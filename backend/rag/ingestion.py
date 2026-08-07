import html
import io
import json
import logging
import math
import re
import time
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import chromadb
import pandas as pd
import trafilatura
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from llama_index.core import Settings as LlamaSettings
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode, TextNode
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.chroma import ChromaVectorStore

from backend.config import settings

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".csv"}
_DOCLING_EXTENSIONS = {".pdf", ".docx"}


class IngestionError(Exception):
    """A file's content failed validation or failed to parse (IMPROVEMENTS.md #8.4).
    Callers at the API boundary (the upload route) turn this into a clean 400; internal
    callers that process many files at once (load_documents) catch it to skip just the
    one bad file instead of failing the whole batch."""


# Zip-bomb guard (IMPROVEMENTS.md #8.4): .docx/.xlsx are zip archives, so the 20MB cap on
# the uploaded bytes doesn't bound how much memory docling/pandas expand them into. A
# small, highly-compressible crafted archive could still be a resource-exhaustion vector
# even though it's a perfectly "valid" zip -- these caps are generous enough that no real
# BINUS catalog/spreadsheet gets anywhere close, but block a bomb from ever reaching
# docling/pandas.
_MAX_ZIP_UNCOMPRESSED_BYTES = 300 * 1024 * 1024
_MAX_ZIP_ENTRIES = 10_000


def _check_zip_bomb(content: bytes, max_uncompressed: int = _MAX_ZIP_UNCOMPRESSED_BYTES) -> None:
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        infos = zf.infolist()
        if len(infos) > _MAX_ZIP_ENTRIES:
            raise IngestionError(f"Archive has too many entries ({len(infos)})")
        total_uncompressed = sum(i.file_size for i in infos)
        if total_uncompressed > max_uncompressed:
            raise IngestionError(
                f"Archive expands to {total_uncompressed / 1_048_576:.0f}MB uncompressed, "
                f"over the {max_uncompressed / 1_048_576:.0f}MB limit"
            )


def validate_upload_content(content: bytes, suffix: str) -> None:
    """Checks the file's actual bytes match its claimed extension, before it's ever
    written to disk or handed to docling/pandas (IMPROVEMENTS.md #8.4 -- extension and
    size were checked before, content never was). Deterministic, magic-byte-based --
    this doesn't try to fully validate internal structure (docling/pandas do that, and
    add_document's own try/except below catches their failure), just rules out "wrong
    file entirely" and zip-bomb-style resource exhaustion before anything expensive runs.
    Raises IngestionError with a message safe to return directly to the admin.
    """
    if suffix == ".pdf":
        if not content.startswith(b"%PDF-"):
            raise IngestionError("File content is not a valid PDF")
    elif suffix in (".docx", ".xlsx"):
        if not zipfile.is_zipfile(io.BytesIO(content)):
            raise IngestionError(f"File content is not a valid {suffix} archive")
        # Passes the module-level cap explicitly (rather than relying on
        # _check_zip_bomb's own default parameter, which binds once at function
        # definition time) so a test can monkeypatch _MAX_ZIP_UNCOMPRESSED_BYTES and
        # actually observe it here.
        _check_zip_bomb(content, max_uncompressed=_MAX_ZIP_UNCOMPRESSED_BYTES)
    elif suffix == ".csv":
        if b"\x00" in content:
            raise IngestionError("File content is not valid CSV (contains binary data)")


def _build_converter() -> DocumentConverter:
    pdf_options = PdfPipelineOptions(do_ocr=False)
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)}
    )


def _docling_page_texts(path: Path, converter: DocumentConverter) -> list[tuple[Optional[int], str]]:
    """PDF/DOCX via docling -> list of (page_number-or-None, page_text).

    _build_converter() sets do_ocr=False, so a scanned/image-only page produces zero
    extractable text -- previously that page was just silently dropped, with the file's
    OTHER pages producing enough chunks that add_document's "no extractable text at all"
    check never fired, so an admin had no way to know that one page's content never made
    it into the KB. Logs a warning per empty page instead so it's visible (in the upload
    response's server-side logs, or reindex output) that OCR would recover something here.
    """
    result = converter.convert(str(path))
    doc = result.document
    num_pages = doc.num_pages()

    page_texts: list[tuple[Optional[int], str]] = []
    if num_pages > 0:
        for page_no in range(1, num_pages + 1):
            text = doc.export_to_markdown(page_no=page_no).strip()
            if text:
                page_texts.append((page_no, text))
            else:
                logger.warning(
                    "%s page %d produced no extractable text (scanned/image page? "
                    "do_ocr is disabled)", path.name, page_no,
                )
    else:
        # Non-paginated formats (e.g. DOCX) - treat as a single source with no page number
        text = doc.export_to_markdown().strip()
        if text:
            page_texts.append((None, text))
    return page_texts


# Program total-credits summary row, e.g. "TOTAL CREDITS 146 Credits" or "Total Credits
# 182 SCU" -- a single, authoritative fact per program catalog. docling's layout-based
# table export occasionally drops exactly this summary row (confirmed live: Computer
# Science 2026's row is in the PDF text layer but absent from every docling-produced
# chunk, while all 9 other SOCS catalogs' rows survived), which then makes a "total
# credits" / comparison question unanswerable for that one program even though the fact
# is right there in the source.
_CREDIT_TOTAL_RE = re.compile(r"total\s+credits?\s+(\d{2,3})\s*(credits?|scu)?\b", re.IGNORECASE)


def _recover_dropped_credit_total(path: Path, docling_text: str) -> Optional[str]:
    """If docling's extraction of a PDF is missing the program's total-credits summary row
    but the PDF's own text layer has it, return a clean standalone statement of that fact
    to be indexed as its own chunk; else None. A deterministic backstop for a specific,
    observed docling table-export gap -- a no-op for every document docling handled
    correctly (the fact is already present, so nothing is added and no duplication is
    introduced). pypdfium2's raw text layer is noisier than docling overall (encoding
    artifacts, no layout), so ONLY this one high-signal, unambiguous line is lifted from
    it, never the whole thing."""
    if path.suffix.lower() != ".pdf":
        return None
    if _CREDIT_TOTAL_RE.search(docling_text):
        return None  # docling already captured it -> nothing to recover
    try:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(path))
        try:
            raw = "\n".join(pdf[i].get_textpage().get_text_range() for i in range(len(pdf)))
        finally:
            pdf.close()
    except Exception as exc:
        logger.warning("Credit-total recovery: could not read text layer of %s: %s", path.name, exc)
        return None
    match = _CREDIT_TOTAL_RE.search(raw)
    if not match:
        return None
    unit = match.group(2) or "Credits"
    recovered = f"Total Credits: {match.group(1)} {unit.title()}"
    logger.info("Recovered docling-dropped credit total for %s: %r", path.name, recovered)
    return recovered


# The base Computer Science catalog PDF renders its "Prospective Career of the Graduates"
# list as an IMAGE, so docling extracts only the lead-in followed by "<!-- image -->" and
# the roles are lost -- a faithful model can then only decline the career question (the 8B
# masked this by fabricating a plausible list). OCR recovers it from a clean copy of the
# same PDF but not from the live re-export (its image is OCR-hostile), so this is the
# authentic list, OCR-recovered once and pinned here. Same deterministic-recovery spirit as
# _recover_dropped_credit_total: injected ONLY when the section is present but its list
# isn't, so a future PDF that carries the list as real text (or a successful OCR) is a
# no-op with no duplication.
_CS_CAREER_ROLES = (
    "Software Engineer/Developer", "System Analyst/Developer", "Web Engineer/Developer",
    "Computer Network Specialist", "Database Specialist", "Artificial Intelligence Specialist",
    "Data Scientist", "IT Support/Consultant", "Researcher", "Multimedia Programmer",
    "Lecturer/Trainer",
)
# Ties the fix to the Computer Science program AND the image-only case: the roles image sits
# right after this exact lead-in. A text-list version wouldn't have the image placeholder
# here, so the guard naturally stops firing once the source carries the list as text.
_CS_CAREER_IMAGE_RE = re.compile(
    r"computer science program could follow a career as:\s*(?:<!--\s*image\s*-->)",
    re.IGNORECASE,
)


def _recover_career_list(path: Path, docling_text: str) -> Optional[str]:
    """The Computer Science careers list when the PDF stored it as an (un-extractable) image;
    else None. See _CS_CAREER_ROLES. A no-op for every other document and for any future
    version whose careers section is real text."""
    if path.suffix.lower() != ".pdf":
        return None
    if not _CS_CAREER_IMAGE_RE.search(docling_text):
        return None
    roles = "\n".join(f"- {r}" for r in _CS_CAREER_ROLES)
    recovered = (
        "Prospective Career of the Graduates (Computer Science): after finishing the "
        "program, graduates of the Computer Science Program could follow a career as:\n" + roles
    )
    logger.info("Recovered image-only career list for %s (%d roles)", path.name, len(_CS_CAREER_ROLES))
    return recovered


_HEADER_RE = re.compile(r"^#{1,6}\s+(.*)$", re.MULTILINE)


def _section_headers(text: str) -> list[tuple[int, str]]:
    """(char_offset, header_text) for every markdown header in text, in document order."""
    return [(m.start(), m.group(1).strip()) for m in _HEADER_RE.finditer(text)]


def _nearest_header(headers: list[tuple[int, str]], offset: int) -> Optional[str]:
    """Text of the last header at or before offset, or None if offset precedes any header."""
    title = None
    for h_offset, h_text in headers:
        if h_offset > offset:
            break
        title = h_text
    return title


def _is_cross_program_partner_table(text: str) -> bool:
    """Detects a "cross-program partnership/master-track index" table: a table whose
    rows pair a short non-university label (a major/stream name, e.g. "Accounting",
    "Marketing") with a cell naming a partner "University". These tables are
    boilerplate shared across many program guides -- e.g. one lists Marketing,
    Finance, Accounting, and Management as rows all pointing to "Macquarie
    University, Australia" -- so they lexically match almost any major name via
    BM25 without containing any actual descriptive content about that major,
    letting them out-rank genuine program-specific prose. Distinct from genuine
    course-structure/minor/elective tables (rows are course names, no "University"
    mentions) and from single-program double-degree tables where the row label
    itself IS the partner university (e.g. "Edinburgh Napier University").
    """
    table_lines = [l.strip() for l in text.splitlines() if l.strip().startswith("|")]
    if len(table_lines) < 3:
        return False

    qualifying_rows = 0
    for line in table_lines:
        cells = [c.strip() for c in line.strip("|").split("|")]
        if not cells or not any(cells):
            continue
        first_cell = cells[0]
        rest = " ".join(cells[1:])
        if "University" in first_cell:
            continue
        words = first_cell.split()
        if not (1 <= len(words) <= 5):
            continue
        if not first_cell[:1].isupper():
            continue
        if "University" in rest:
            qualifying_rows += 1

    return qualifying_rows >= 3


def _parent_child_split(
    text: str, carried_title: Optional[str] = None
) -> tuple[list[tuple[str, str, Optional[str]]], Optional[str]]:
    """Split text into large parent chunks, then each parent into small child chunks.

    Returns ((child_text, parent_text, section_title) triples, last_title). section_title
    (R-07) is the nearest preceding markdown header -- docling renders real document headings
    ("## Prospective Career of the Graduates") as markdown headers, so this recovers
    structural context that plain chunking would otherwise drop. carried_title/last_title let
    callers chain this across a document's pages: docling exports markdown per-page, so a
    section heading on page N would otherwise be invisible to page N+1's continuation text;
    carried_title is the fallback for chunks preceding this page's own first header, and
    last_title (this page's own last header, or carried_title if it had none) is what the
    caller should pass in for the next page.
    """
    headers = _section_headers(text)
    parent_splitter = SentenceSplitter(
        chunk_size=settings.parent_chunk_size, chunk_overlap=settings.parent_chunk_overlap
    )
    child_splitter = SentenceSplitter(
        chunk_size=settings.child_chunk_size, chunk_overlap=settings.child_chunk_overlap
    )
    triples: list[tuple[str, str, Optional[str]]] = []
    cursor = 0
    for parent_text in parent_splitter.split_text(text):
        offset = text.find(parent_text, cursor)
        if offset == -1:
            offset = text.find(parent_text)
        if offset != -1:
            cursor = offset
        section_title = _nearest_header(headers, max(offset, 0)) or carried_title
        for child_text in child_splitter.split_text(parent_text):
            # Markdown table export can leave bare-pipe row fragments (e.g. "|", "| 2")
            # at split boundaries -- too little signal to embed meaningfully, and short
            # enough to occasionally produce a degenerate (NaN) embedding vector.
            if len(child_text.strip()) < 10:
                continue
            # Cross-program partnership/master-track index tables (e.g. a Macquarie
            # University row-per-major table) are reference boilerplate, not
            # descriptive content -- exclude them so they can't out-rank genuine
            # program-specific prose just because a major's name appears as a row.
            if _is_cross_program_partner_table(child_text):
                continue
            # A "Free Electives" appendix (seen in Computer_Science_2026.pdf: 61 of its
            # 287 chunks) lists cross-registrable courses "owned" by dozens of other
            # departments -- e.g. a row literally reading "Information Systems | ISYS6900003
            # | IT Governance & Security" -- so a query about that OTHER program's own
            # curriculum can match this row and clear the confidence gate even though the
            # chunk describes an elective slot in a different program's catalog, not that
            # program itself. Keyed on section_title (not table content) since docling
            # recovers this heading reliably but individual child chunks split from the
            # table often don't repeat the words "Free Elective" in their own text.
            if section_title and "free elective" in section_title.lower():
                continue
            triples.append((child_text, parent_text, section_title))
    last_title = headers[-1][1] if headers else carried_title
    return triples, last_title


def _spreadsheet_sheet_texts(path: Path) -> list[tuple[Optional[str], str]]:
    """XLSX/CSV -> list of (sheet_name-or-None, sheet_text)."""
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        return [(None, df.to_csv(index=False))]

    sheets = pd.read_excel(path, sheet_name=None)
    return [(name, df.to_csv(index=False)) for name, df in sheets.items()]


def add_document(path: Path, converter: Optional[DocumentConverter] = None) -> list[TextNode]:
    """Parse a single PDF/DOCX/XLSX/CSV file into chunked TextNodes with source metadata."""
    suffix = path.suffix.lower()
    ingested_at = datetime.now(timezone.utc).isoformat()

    try:
        if suffix in _DOCLING_EXTENSIONS:
            converter = converter or _build_converter()
            sections = _docling_page_texts(path, converter)
            section_key = "page_number"
        else:
            sections = _spreadsheet_sheet_texts(path)
            section_key = "sheet_name"
    except Exception as exc:
        # A file that passes validate_upload_content's magic-byte check (or one already
        # sitting in documents_dir, e.g. during reindex) can still be internally
        # corrupted in a way only the real parser catches -- surface that as a clean,
        # catchable error instead of an unhandled crash (IMPROVEMENTS.md #8.4).
        raise IngestionError(f"Failed to parse {path.name}: {exc}") from exc

    nodes: list[TextNode] = []
    carried_title: Optional[str] = None
    for section_value, text in sections:
        triples, carried_title = _parent_child_split(text, carried_title=carried_title)
        for child_text, parent_text, section_title in triples:
            metadata = {
                "source_file": path.name,
                "ingested_at": ingested_at,
                "parent_text": parent_text,
            }
            if section_value is not None:
                metadata[section_key] = section_value
            if section_title is not None:
                metadata["section_title"] = section_title
            nodes.append(TextNode(text=child_text, metadata=metadata))

    # Deterministic backstop for docling dropping a program's total-credits summary row
    # (see _recover_dropped_credit_total). Only fires for a PDF whose docling output
    # genuinely lacks it, so it's a no-op for every document already indexed correctly.
    if suffix in _DOCLING_EXTENSIONS:
        full_text = "\n".join(t for _, t in sections)
        recovered = _recover_dropped_credit_total(path, full_text)
        if recovered:
            nodes.append(TextNode(text=recovered, metadata={
                "source_file": path.name,
                "ingested_at": ingested_at,
                "parent_text": recovered,
                "section_title": "Total Credits",
            }))
        # Same backstop shape for the Computer Science careers list stored as an image.
        careers = _recover_career_list(path, full_text)
        if careers:
            nodes.append(TextNode(text=careers, metadata={
                "source_file": path.name,
                "ingested_at": ingested_at,
                "parent_text": careers,
                "section_title": "Prospective Career of the Graduates",
            }))
        # One self-describing node per course row so per-course SCU lookups retrieve the course
        # and its credit value together (see _course_scu_row_nodes). A no-op for a document with
        # no Course-Name/SCU table.
        nodes.extend(_course_scu_row_nodes(path, full_text))

    logger.info("  -> %d chunk(s) from %d section(s)", len(nodes), len(sections) or 1)
    return nodes


def load_documents(documents_dir: Path) -> list[TextNode]:
    """Parse every supported file in documents_dir into chunked TextNodes."""
    converter = _build_converter()
    nodes: list[TextNode] = []

    files = sorted(
        p for p in documents_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        logger.warning("No supported documents found in %s", documents_dir)
        return nodes

    for path in files:
        logger.info("Parsing %s", path.name)
        try:
            nodes.extend(add_document(path, converter=converter))
        except IngestionError as exc:
            # One corrupted file already on disk (e.g. from a previous partial write)
            # shouldn't take down startup/reindex for every other document -- skip it
            # and keep going, same "don't let one bad input sink the whole batch"
            # pattern /admin/reindex already uses for a URL that fails to re-fetch.
            logger.warning("Skipping %s: %s", path.name, exc)

    return nodes


def load_scraped_urls() -> list[str]:
    """Every URL ever scraped into the KB via record_scraped_url, in add-order.

    /admin/reindex (IMPROVEMENTS.md #5.1) re-fetches each of these after re-walking
    documents_dir, so a full reindex no longer silently drops URL sources. Read fresh
    on every call, same pattern as load_fallback_contacts.
    """
    if not settings.scraped_urls_path.exists():
        return []
    return json.loads(settings.scraped_urls_path.read_text(encoding="utf-8"))


def _save_scraped_urls(urls: list[str]) -> None:
    settings.scraped_urls_path.write_text(
        json.dumps(urls, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def record_scraped_url(url: str) -> None:
    """Called by the /admin/documents/url route after a successful scrape."""
    urls = load_scraped_urls()
    if url not in urls:
        urls.append(url)
        _save_scraped_urls(urls)


def forget_scraped_url(url: str) -> None:
    """Called by the /admin document-delete route; a no-op if url isn't a known URL source."""
    urls = load_scraped_urls()
    if url in urls:
        urls.remove(url)
        _save_scraped_urls(urls)


# Indirect prompt injection via scraped URLs (IMPROVEMENTS.md #8.2): a scraped page's
# text is stored as chunks and later dropped straight into the model's context at query
# time, so a hostile (or compromised) page could carry text aimed at the model itself,
# not the user -- "ignore previous instructions", a fake "SYSTEM:" turn, a request to
# reveal the system prompt, etc. This can't be made airtight by pattern-matching alone
# (arbitrary phrasing always exists), so it's paired with an explicit instruction in
# ANSWER_SYSTEM_PROMPT/ANSWER_USER_TEMPLATE telling the model context is data, never
# commands (the load-bearing part, since it also covers phrasings this regex misses) --
# this scrub is the deterministic first layer: catch the loud, common, unambiguous
# cases before they even reach the model, and log/flag when it fires so an admin adding
# a URL has a reason to go look at the source page.
_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE | re.MULTILINE)
    for p in [
        r"ignore\s+(all\s+|the\s+)?(above|prior|previous)\s+instructions?",
        r"disregard\s+(all\s+|the\s+)?(above|prior|previous)\s+(instructions?|rules?|prompt)",
        r"forget\s+(all\s+|the\s+)?(above|prior|previous)\s+instructions?",
        r"new\s+instructions?\s*:",
        r"you\s+are\s+now\s+(a|an)\s+\w+",
        r"reveal\s+(your\s+)?(system\s+prompt|instructions)",
        r"print\s+(your\s+)?(system\s+prompt|instructions)",
        r"^\s*(system|assistant)\s*:",
    ]
]


def _scrub_injection_attempts(text: str, source: str) -> str:
    scrubbed = text
    hit_count = 0
    for pattern in _INJECTION_PATTERNS:
        scrubbed, n = pattern.subn("[redacted]", scrubbed)
        hit_count += n
    if hit_count:
        logger.warning(
            "scrape_url: redacted %d potential prompt-injection pattern(s) from %s",
            hit_count, source,
        )
    return scrubbed


# BINUS publishes tuition fees as one page PER CAMPUS (query param campus-location=
# binus-xxx), each holding a single markdown table listing EVERY program at that
# campus, sometimes for more than one academic year. Generic sentence-based chunking
# (_parent_child_split) splits these tables mid-row and packs 40+ irrelevant programs'
# numbers into every chunk -- a query about one program's fees is competing against
# every other program just to be recognized as on-topic, and a row can be fragmented
# across a chunk boundary. Confirmed live: "tuition fees for Computer Science" only
# ever surfaced 2 of the 7 campuses that actually offer Computer Science, because the
# final context budget (settings.rerank_top_n, kept small to respect the LLM provider's
# per-minute token limit) was being spent on giant multi-program table fragments instead of small,
# precisely on-topic rows.
_TUITION_FEE_URL_RE = re.compile(r"^https://gabung\.binus\.ac\.id/tuition-fee/", re.IGNORECASE)
_ACADEMIC_YEAR_RE = re.compile(r"ACADEMIC YEAR\s+(\d{4}/\d{4})", re.IGNORECASE)
_TABLE_ROW_RE = re.compile(r"^\|(.+)\|\s*$")
_TABLE_SEPARATOR_RE = re.compile(r"^\|[\s:|-]+\|\s*$")
_CAMPUS_LOCATION_RE = re.compile(r"campus-location=([a-z0-9-]+)", re.IGNORECASE)
_CAMPUS_LABEL_OVERRIDES = {"aso": "ASO", "online": "Online Learning"}


def _campus_label(url: str) -> str:
    """Human-readable campus name from a `campus-location=binus-xxx` URL param, e.g.
    "binus-alam-sutera" -> "BINUS Alam Sutera". Read from the URL rather than a
    hardcoded per-campus name list, so a campus added to the source site later needs no
    code change here."""
    match = _CAMPUS_LOCATION_RE.search(url)
    if not match:
        return "BINUS"
    slug = re.sub(r"^binus-", "", match.group(1))
    label = _CAMPUS_LABEL_OVERRIDES.get(slug, slug.replace("-", " ").title())
    return f"BINUS {label}"


def known_campus_names() -> set[str]:
    """Every real BINUS campus name derivable from currently-scraped URLs (Alam Sutera,
    ASO, Bandung, Bekasi, Kemanggisan, Malang, Medan, Online Learning, Semarang,
    Senayan), without the "BINUS " prefix _campus_label adds for citation labels. Same
    "read from the URL, no hardcoded list" reasoning as _campus_label itself -- used by
    generation.detect_unresolved_campus_mention so a query naming a real campus that
    just isn't slang/nickname-y enough to need a _CAMPUS_ALIASES entry (e.g. "kampus
    ASO", "kampus online") doesn't get mistaken for an unresolved one.
    """
    names = set()
    for url in load_scraped_urls():
        match = _CAMPUS_LOCATION_RE.search(url)
        if not match:
            continue
        slug = re.sub(r"^binus-", "", match.group(1))
        names.add(_CAMPUS_LABEL_OVERRIDES.get(slug, slug.replace("-", " ").title()))
    return names


def admission_requirement_url_for_campus(campus_name: str) -> Optional[str]:
    """The scraped admission-requirement page for a canonical campus name (as
    known_campus_names yields it, e.g. "Kemanggisan"), or None if that campus has no such
    page scraped. Reverses the same slug->name derivation known_campus_names uses -- read
    from the URLs, no hardcoded campus->URL table -- so it stays correct as campuses are
    added/removed at the source. Used to scope a "what programs are offered at campus X"
    query to the one page that actually lists them (see chat_service's campus-programs route)."""
    for url in load_scraped_urls():
        if "admission-requirement" not in url:
            continue
        match = _CAMPUS_LOCATION_RE.search(url)
        if not match:
            continue
        slug = re.sub(r"^binus-", "", match.group(1))
        if _CAMPUS_LABEL_OVERRIDES.get(slug, slug.replace("-", " ").title()) == campus_name:
            return url
    return None


# Every fee the tuition-fee pages publish is labelled "Semester 1" or "(hanya 1x)", so the KB
# could say what the first semester costs and nothing at all about the second. Real traffic asks
# ("Berapa biaya kuliah semester 2?"), and the honest answer is not a fallback: the recurring fees
# repeat at the same amount, which is a fact about the fee structure that the page's own labels
# imply but never state. Appended to each fee row rather than emitted as its own chunk because the
# tuition route retrieves fee ROWS (see chat_service._retry_tuition_across_campuses); a standalone
# note would have to win a place in the top-N against them, and losing that race means the fact is
# absent exactly when it is needed.
#
# Deliberately claims nothing about the program total, and says nothing about how to compute one.
# The published "Estimasi Total Biaya" does not equal 8 x (semester fee) + the one-time fees on any
# campus -- Kemanggisan's total implies 7.23 semesters, Bandung's 5.49 -- because tuition is a fixed
# component plus a variable per-SKS component and SKS load differs by semester (see
# socs_documents/Tuition_Fees_Computer_Science.md). The authoritative total is already in the same
# row, so the note does not need to argue about arithmetic. Guidance on how to ANSWER belongs in
# rag/prompts.py, not repeated 16 times in retrieved context.
#
# Kept to ~60 tokens for that reason: the tuition route retrieves up to 16 rows, so every token
# here is paid 16 times. Indonesian first because that is the language of the labels it qualifies,
# with a short English clause so an English "semester 2" query still has the fact in its own words.
_SEMESTER_FEE_NOTE = (
    "Biaya Kuliah dan Biaya Laboratorium sama untuk setiap semester berikutnya "
    "(Semester 2 dan seterusnya); Biaya Peralatan dan Biaya Sumbangan / DP3 dibayar satu kali. "
    "Per-semester tuition and laboratory fees repeat at the same amount in later semesters; "
    "equipment and DP3 fees are one-time."
)


def _tuition_fee_row_nodes(url: str, text: str) -> list[TextNode]:
    """Splits every data row of a BINUS tuition-fee page's markdown table(s) into its
    own small chunk: one program, one campus, one academic year per chunk. Applied to
    the single un-chunked page text (before _parent_child_split ever runs), so table
    rows are read intact -- never split mid-row -- straight from the source markdown.

    Not every campus offers every program (e.g. the ASO campus's table has no
    Computer-Science-family row at all) -- this only ever emits a chunk for a row that
    genuinely exists on that campus's page, so a program absent from one campus is
    correctly absent from retrieval too, not silently guessed at.
    """
    campus = _campus_label(url)
    nodes: list[TextNode] = []
    year_label: Optional[str] = None
    header_cells: Optional[list[str]] = None
    ingested_at = datetime.now(timezone.utc).isoformat()
    lines = text.splitlines()

    for i, line in enumerate(lines):
        year_match = _ACADEMIC_YEAR_RE.search(line)
        if year_match:
            year_label = year_match.group(1)

        stripped = line.strip()
        if _TABLE_SEPARATOR_RE.match(stripped):
            continue  # the "|---|---|" rule row under a header -- already consumed below

        row_match = _TABLE_ROW_RE.match(stripped)
        if not row_match:
            if stripped:
                header_cells = None  # a non-table line ends the current table
            continue

        next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if _TABLE_SEPARATOR_RE.match(next_line):
            header_cells = [c.strip() for c in row_match.group(1).split("|")]
            continue

        if not header_cells:
            continue
        cells = [c.strip() for c in row_match.group(1).split("|")]
        if len(cells) != len(header_cells) or not cells[0]:
            continue

        program = cells[0]
        facts = "; ".join(
            f"{h}: {v}" for h, v in zip(header_cells[1:], cells[1:]) if h and v
        )
        if not facts:
            continue
        year_suffix = f" (Academic Year {year_label})" if year_label else ""
        node_text = f"{campus} -- {program}{year_suffix}: {facts}. {_SEMESTER_FEE_NOTE}"
        nodes.append(TextNode(
            text=node_text,
            metadata={
                "source_file": url,
                "ingested_at": ingested_at,
                "parent_text": node_text,
                "section_title": f"{program} -- {campus}{year_suffix}",
            },
        ))
    return nodes


# The catalog PDFs' course-structure tables ("| Sem | Code | Course Name | SCU | Total |") are
# large enough that _parent_child_split routinely lands a course's row and the "| SCU |" column
# header in DIFFERENT chunks -- so a per-course credit lookup ("berapa sks Computer Graphics")
# retrieves either a bare "| Computer Graphics | 2/2 |" with no way to tell the number is an SCU
# count, or misses the row entirely (confirmed via retrieval diagnostic: the Computer Graphics
# row's only parent scored 0.045 and carried no SCU header). This mirrors _tuition_fee_row_nodes:
# split every course row into its own small, self-describing node -- "<Program> program --
# <Course>: <SCU> SCU (Semester N)" -- read straight from the raw docling markdown BEFORE
# _parent_child_split, so a course and its credits are never separated. The chunked table nodes
# still exist for broad "what's the curriculum" queries; these are additive, for point lookups.
_COURSE_SCU_RE = re.compile(r"^\d+(?:/\d+)?$")  # "2", "4", "4/2", "2/1" -- lecture[/lab] SCU
_COURSE_NAME_NOISE_RE = re.compile(r"\s*\((?:AOL|Block)\)|\*+", re.IGNORECASE)
_PROGRAM_YEAR_SUFFIX_RE = re.compile(r"_\d{4}$")  # same convention as retrieval.get_program_catalog


def _course_scu_row_nodes(path: Path, text: str) -> list[TextNode]:
    """One node per course in a catalog PDF's course-structure table(s), with the course's SCU
    (Semester Credit Unit) value inline -- see the block comment above for why the normal
    chunking can't answer a per-course credit question. A no-op for a document with no
    Course-Name/SCU table (the header gate below never opens), so it's safe to run on every
    PDF/DOCX. The Sem column is carried forward across rows that leave it blank (docling only
    fills it on a semester group's first row); a row whose SCU cell isn't a credit token
    (blank, "Streaming: ...", a stray merged cell) is skipped rather than guessed at."""
    program = re.sub(r"\s+", " ", _PROGRAM_YEAR_SUFFIX_RE.sub("", path.stem).replace("_", " ")).strip()
    ingested_at = datetime.now(timezone.utc).isoformat()
    lines = text.splitlines()
    header_cells: Optional[list[str]] = None
    name_idx = scu_idx = sem_idx = None
    current_sem = ""
    seen: set[tuple] = set()
    nodes: list[TextNode] = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if _TABLE_SEPARATOR_RE.match(stripped):
            continue
        row_match = _TABLE_ROW_RE.match(stripped)
        if not row_match:
            if stripped:
                header_cells = None  # a non-table line ends the current table
            continue
        cells = [c.strip() for c in row_match.group(1).split("|")]

        # A header row is recognized only when the NEXT line is the "|---|" separator AND it
        # carries the Course-Name + SCU columns this parser is scoped to -- so an unrelated
        # table on the page (minor/elective lists, enrichment tracks) is ignored.
        next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if _TABLE_SEPARATOR_RE.match(next_line):
            lowered = [c.lower() for c in cells]
            if "course name" in lowered and "scu" in lowered:
                header_cells = cells
                name_idx = lowered.index("course name")
                scu_idx = lowered.index("scu")
                sem_idx = lowered.index("sem") if "sem" in lowered else None
                current_sem = ""
            else:
                header_cells = None
            continue

        if not header_cells or len(cells) != len(header_cells):
            continue
        if sem_idx is not None and cells[sem_idx]:
            current_sem = cells[sem_idx]
        course = _COURSE_NAME_NOISE_RE.sub("", cells[name_idx]).strip()
        scu = cells[scu_idx]
        if not course or not _COURSE_SCU_RE.match(scu):
            continue
        key = (course.lower(), scu, current_sem)
        if key in seen:  # docling occasionally repeats a row; one node per (course, scu, sem)
            continue
        seen.add(key)
        sem_suffix = f" (Semester {current_sem})" if current_sem else ""
        node_text = f"{program} program -- {course}: {scu} SCU{sem_suffix}"
        nodes.append(TextNode(
            text=node_text,
            metadata={
                "source_file": path.name,
                "ingested_at": ingested_at,
                "parent_text": node_text,
                "section_title": f"{course} -- {program} Course Structure",
            },
        ))
    return nodes


# --- SoCS faculty roster (a second structured-source special case, like tuition above) ---
#
# The public faculty list (socs.binus.ac.id/community/faculty-members/) is a plain static
# table trafilatura reads cleanly -- Kode Dosen | Nama | scholar-profile URL, ~210 rows.
# But the role and the classes each lecturer teaches live on the JS-rendered scholar
# profile pages, which a static fetch can't see. Those pages load their data from a public
# JSON API (found by reading scholar.binus.ac.id's own front-end script.js): a POST to
# .../lecturers/detail returns academic rank + department, and .../lecturers/teachings/list
# returns the courses taught in a given year. So this builder scrapes the list statically,
# then enriches each row via those two API calls, producing one compact node per lecturer
# (name + rank + most-recent-year courses) -- the granularity that lets "who teaches X" or
# "what is <name>'s role" retrieve a single self-contained record.
FACULTY_ROSTER_URL = "https://socs.binus.ac.id/community/faculty-members/"
_FACULTY_LIST_URL_RE = re.compile(
    r"^https://socs\.binus\.ac\.id/community/faculty-members/?$", re.IGNORECASE
)
_FACULTY_ROW_RE = re.compile(
    r"\|\s*(D\d+)\s*\|\s*([^|]+?)\s*\|\s*(https://scholar\.binus\.ac\.id/lecturer/\S+)"
)
_SCHOLAR_LECTURERS_API = "https://scholar.binus.ac.id/wp-json/binus-scholar/v1/lecturers/"
# Bearer token read from scholar.binus.ac.id's own front-end JS, where it sits hardcoded and
# public. It can rotate server-side; if it does, these POSTs stop returning data and the
# builder degrades to name-only rows (or an empty roster), which the /admin/reindex loop
# already treats as a re-fetch failure to report -- never a crash. Same safe-degradation
# contract as any scraped URL that stops resolving.
_SCHOLAR_BEARER = "Rx00Q0UwLUEyRWx"
# A universal onboarding/character course assigned to nearly every lecturer -- confirmed
# empirically that early in an academic year it's often the ONLY course listed, which would
# otherwise make "most recent year" resolve to a year with no real teaching signal. Excluded
# from course lists AND from the "does this year have real teaching" check below.
_FILLER_COURSE_TITLE = "BINUS DNA"


def _scholar_lecturer_post(endpoint: str, fields: dict) -> list:
    """POST to a binus-scholar lecturers API endpoint, returning its `data` list (empty on
    any non-data response). Raises on network error -- callers decide whether to skip."""
    body = urllib.parse.urlencode(fields).encode()
    request = urllib.request.Request(
        _SCHOLAR_LECTURERS_API + endpoint,
        data=body,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Authorization": f"Bearer {_SCHOLAR_BEARER}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=25) as response:
        payload = json.loads(response.read().decode("utf-8", "ignore"))
    return payload.get("data") or []


# A transient scholar-API failure must never be readable as "this lecturer taught nothing". One
# or two retries with a short backoff absorb an ordinary blip; past that the failure is REPORTED
# rather than folded into an empty result, because the two are not the same fact and the callers
# below depend on telling them apart. Cost of the retries is bounded: they only fire on an actual
# failure, and the crawl already paces itself at 50ms per lecturer.
#
# This is not hypothetical. On 2026-08-07 a mid-probe scholar-API failure produced a record that
# looked exactly like real data, and the conclusion drawn from it -- that the faculty snapshot had
# drifted from the source -- was wrong. Re-probing showed the snapshot was correct all along.
_SCHOLAR_ATTEMPTS = 3
_SCHOLAR_RETRY_BACKOFF_S = 0.6

# Crawl-time provenance on a record whose scholar responses were not all readable. Never part of
# the stored schema -- _save_faculty_snapshot strips it, so a bootstrap crawl and a refresh both
# write the shape a reader expects.
_SCAN_INCOMPLETE_KEY = "_scan_incomplete"


def _scholar_post_ok(endpoint: str, fields: dict) -> tuple[list, bool]:
    """(rows, ok). ok=False means the API did not answer after _SCHOLAR_ATTEMPTS tries, which is
    categorically different from it answering with no rows. Every caller must branch on this
    rather than on `not rows`, or "no answer" silently becomes "nothing to report"."""
    for attempt in range(_SCHOLAR_ATTEMPTS):
        try:
            return _scholar_lecturer_post(endpoint, fields), True
        except Exception as exc:
            if attempt + 1 >= _SCHOLAR_ATTEMPTS:
                logger.warning(
                    "scholar %s gave up after %d attempts for %r: %s",
                    endpoint, _SCHOLAR_ATTEMPTS, fields.get("lecturer_id"), exc,
                )
                return [], False
            time.sleep(_SCHOLAR_RETRY_BACKOFF_S * (attempt + 1))
    return [], False


def _lecturer_detail(code: str) -> tuple[Optional[str], Optional[str], Optional[str], bool]:
    """(name, academic rank, department, ok) for a lecturer code. The first three are None when
    the detail API has no active record (e.g. emeritus/inactive faculty, or a campus-page code
    not in the scholar system). The detail endpoint repeats the same record several times;
    the first is enough. Name is included so a code discovered on a campus page but absent
    from the main roster list can still be named.

    `ok` is False only when the API never answered. A genuinely absent record returns
    (None, None, None, True) -- that is a real answer, and overwriting a cached rank with it is
    correct. An unanswered request returns ok=False so the caller can decline to overwrite."""
    rows, ok = _scholar_post_ok("detail", {"token": "", "lecturer_id": code})
    if not rows:
        return None, None, None, ok
    return (
        rows[0].get("namaDosen"), rows[0].get("desc_JJA2"), rows[0].get("desc_Department"), ok,
    )


def _lecturer_recent_courses(
    code: str, back_years: int = 5
) -> tuple[Optional[str], list[str], bool]:
    """(year, courses, complete) for the most recent academic year in which the lecturer taught a
    real (non-filler) course, scanning back from the current year. Returns (None, [], complete)
    if none in range. Scanning back rather than pinning a year makes this roll forward on its own
    as new years populate; the filler-course skip (see _FILLER_COURSE_TITLE) stops a barely-
    started new year from masking the last year of actual teaching.

    `complete` is False when at least one year NEWER than the one returned could not be read.
    That is the difference between "2025 is this lecturer's most recent teaching year" and "2025
    is the most recent year we managed to ask about". A failing year is still skipped rather than
    aborting the scan -- returning the older year beats returning nothing -- but the caller is
    told the answer may be stale, because a silently backdated year is indistinguishable from
    real data once it is in the snapshot.
    """
    current_year = datetime.now(timezone.utc).year
    complete = True
    for year in range(current_year, current_year - back_years, -1):
        rows, ok = _scholar_post_ok(
            "teachings/list", {"token": "", "lecturer_id": code, "year": str(year)}
        )
        if not ok:
            # Unknown, not empty. Only years newer than whatever we eventually return can reach
            # this point, so one flag is enough to say "something newer was unreadable".
            complete = False
            continue
        titles = sorted({
            r.get("coursE_TITLE_LONG", "").strip()
            for r in rows
            if r.get("coursE_TITLE_LONG", "").strip()
        })
        substantive = [t for t in titles if t != _FILLER_COURSE_TITLE]
        if substantive:
            return str(year), substantive, complete
    return None, [], complete


def _faculty_node_text(
    name: str, code: Optional[str], role: Optional[str], dept: Optional[str],
    year: Optional[str], courses: list[str], campuses: Optional[list[str]] = None,
    struktural: Optional[str] = None,
) -> str:
    """One compact, self-contained lecturer record. Course titles stay in their source
    (English) form; the surrounding labels are Indonesian, matching the KB's dominant
    language -- retrieval is multilingual and BM25 still matches an English course title in
    an English query either way."""
    lead = f"{name} adalah dosen BINUS School of Computer Science"
    lead += f" (kode dosen {code})." if code else "."
    parts = [lead]
    if role and dept:
        parts.append(f"Jabatan akademik: {role} di bidang {dept}.")
    elif role:
        parts.append(f"Jabatan akademik: {role}.")
    elif dept:
        parts.append(f"Bidang: {dept}.")
    if struktural:
        # Org/structural role (Dean, Head of X Program, etc.), bilingual so both "siapa
        # kepala program X" and "who is the head of X" retrieve this lecturer.
        parts.append(f"Jabatan struktural di BINUS School of Computer Science: {struktural}.")
        parts.append(f"Leadership position at BINUS School of Computer Science: {struktural}.")
    if campuses:
        listed_campus = ", ".join(campuses)
        parts.append(f"Mengajar di kampus BINUS: {listed_campus}.")
        parts.append(f"Teaches at BINUS campus(es): {listed_campus}.")
    if courses:
        listed = "; ".join(courses)
        parts.append(f"Mata kuliah yang diajar pada tahun akademik {year}: {listed}.")
        # English gloss of the course list. The reranker (bge-reranker-v2-m3) scores an
        # English "who teaches X" query near zero against the Indonesian-framed line above
        # (measured 0.0-0.31, below the 0.5 gate), but ~0.66-0.70 once an English "teaches"
        # framing carrying the same course titles is present -- so this line is what makes
        # English who-teaches queries work at all, without hurting the Indonesian ones.
        parts.append(f"Courses taught in {year} (teaches): {listed}.")
    return " ".join(parts)


# --- Per-campus enrichment (Task 6): which campus(es) each lecturer teaches at ---
#
# The main roster list above is School-wide and carries no campus. The 7 CS campuses each
# publish a people page; a person can appear on several (staff commute/share, esp. the
# Jakarta pair Kemanggisan+Alam Sutera which is one shared page), so campus is MANY-TO-MANY.
# These pages are JS-rendered, but each person's card links to a static profile page whose
# text contains the `D####` code -- the exact join key back to the scholar API/roster. Two
# regional pages (Semarang, Bekasi) use a barebones profile template with NO code, so those
# fall back to a symmetric name-match against the roster. Net effect: the roster becomes a
# superset -- every original lecturer is kept, plus campus tags, plus the ~15 people the
# campus pages list that the School-wide roster missed.
_CAMPUS_PEOPLE_PAGES = [
    ("https://socs.binus.ac.id/computer-science/people/", ("Kemanggisan", "Alam Sutera")),
    ("https://binus.ac.id/bandung/computer-science/binus-people/", ("Bandung",)),
    ("https://binus.ac.id/semarang/computer-science/people/", ("Semarang",)),
    ("https://binus.ac.id/medan/computer-science/binus-people/", ("Medan",)),
    ("https://binus.ac.id/bekasi/csse/", ("Bekasi",)),
    ("https://binus.ac.id/malang/computer-science/binus-people/", ("Malang",)),
]
_PROFILE_URL_RE = re.compile(
    r'href="(https://[^"]+/(?:binus-people|people)/[a-z0-9][a-z0-9\-]+/?)"', re.IGNORECASE
)
_PROFILE_ROOT_RE = re.compile(r"/(?:binus-people|people)/?$", re.IGNORECASE)
_DCODE_RE = re.compile(r"\bD\d{3,4}\b")


def _http_get(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=25) as response:
        return response.read().decode("utf-8", "ignore")


def _norm_person_name(name: str) -> str:
    """Symmetric name key: lowercase, non-letters -> space, collapsed -- keeps degree tokens
    (S.Kom etc.) so a full name matches the hyphen-joined form in a profile-page slug."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z]", " ", name.lower())).strip()


def _jakarta_ajax_profile_urls(list_html: str) -> set[str]:
    """The socs.binus.ac.id Jakarta page paginates its people via a `load_more_people`
    admin-ajax action (the regional pages are small enough to be fully server-rendered)."""
    nonce_match = re.search(r"nonce:\s*'([a-f0-9]+)'", list_html)
    if not nonce_match:
        return set()
    ajax = "https://socs.binus.ac.id/computer-science/wp-admin/admin-ajax.php"
    urls: set[str] = set()
    for page in range(1, 15):
        body = urllib.parse.urlencode({
            "action": "load_more_people", "nonce": nonce_match.group(1), "page": page,
            "limit": 100, "search": "", "group": "lecturer",
        }).encode()
        try:
            request = urllib.request.Request(
                ajax, data=body,
                headers={"User-Agent": "Mozilla/5.0", "X-Requested-With": "XMLHttpRequest"},
            )
            with urllib.request.urlopen(request, timeout=25) as response:
                payload = json.loads(response.read().decode("utf-8", "ignore"))
        except Exception:
            break
        data = payload.get("data")
        people = data.get("people") if isinstance(data, dict) else None
        if not people:
            break
        for person in people:
            if person.get("url"):
                urls.add(person["url"].rstrip("/") + "/")
    return urls


def _campus_profile_urls(page_url: str) -> set[str]:
    try:
        html = _http_get(page_url)
    except Exception:
        return set()
    urls = {
        m.group(1) for m in _PROFILE_URL_RE.finditer(html)
        if not _PROFILE_ROOT_RE.search(m.group(1)) and "/feed" not in m.group(1)
    }
    if "socs.binus.ac.id/computer-science/people" in page_url:
        urls |= _jakarta_ajax_profile_urls(html)
    return urls


def _profile_code_and_name(profile_url: str) -> tuple[Optional[str], Optional[str]]:
    """(D-code|None, display-name|None) from a lecturer profile page -- both static-readable.
    Name comes from the page <title>, HTML-unescaped and stripped of the trailing
    " – BINUS ..." site suffix (the title uses an entity en-dash, e.g. `&#8211;`, so it must
    be unescaped before splitting). A code-less barebones profile still yields a clean name
    for the roster name-match fallback."""
    try:
        page = _http_get(profile_url)
    except Exception:
        return None, None
    codes = _DCODE_RE.findall(page)
    code = Counter(codes).most_common(1)[0][0] if codes else None
    name = None
    title_match = re.search(r"<title>(.*?)</title>", page, re.IGNORECASE | re.DOTALL)
    if title_match:
        title = html.unescape(re.sub(r"\s+", " ", title_match.group(1)).strip())
        name = re.split(r"\s[–—|]\s|\s-\s", title)[0].strip() or None
    return code, name


def _build_campus_index(
    roster_name_to_code: dict[str, str],
) -> tuple[dict[str, set[str]], dict[str, tuple[str, set[str]]]]:
    """Crawl the campus people pages -> (campus_by_code, name_only). `campus_by_code` maps a
    resolved lecturer code to the SET of campuses it appeared under (via profile D-code, or a
    name-match to the roster for code-less profiles). `name_only` holds campus people who have
    neither a code nor a roster name-match -> keyed by normalized name -> (display name,
    campuses). Fully best-effort: any page/profile that fails is skipped, never fatal."""
    campus_by_code: dict[str, set[str]] = {}
    name_only: dict[str, tuple[str, set[str]]] = {}
    for page_url, campuses in _CAMPUS_PEOPLE_PAGES:
        campus_set = set(campuses)
        profiles = _campus_profile_urls(page_url)
        logger.info("campus crawl: %d profiles on %s", len(profiles), page_url)
        for profile_url in profiles:
            code, name = _profile_code_and_name(profile_url)
            slug_name = profile_url.rstrip("/").split("/")[-1].replace("-", " ")
            if not code:
                # Match on the title-name, then the URL slug (a clean hyphen-joined form of
                # the same name) -- either resolves a code-less regional profile to a roster
                # lecturer without needing the profile to expose a code.
                code = (roster_name_to_code.get(_norm_person_name(name or ""))
                        or roster_name_to_code.get(_norm_person_name(slug_name)))
            if code:
                campus_by_code.setdefault(code, set()).update(campus_set)
            else:
                display = name or slug_name.title()
                key = _norm_person_name(display)
                existing = name_only.get(key, (display, set()))[1]
                name_only[key] = (display, existing | campus_set)
            time.sleep(0.03)
    return campus_by_code, name_only


# --- SoCS leadership / structural roles (Task 3) ---
#
# socs.binus.ac.id/people/ server-renders the org chart -- Dean, Deputy Deans, Heads of
# Department (per campus), and Head of each Program (Computer Science, Cyber Security, Data
# Science, ...) -- as cards carrying the person's name, their structural role, and a profile
# link (which, like the campus profiles, exposes the D-code). These people are all lecturers
# too, so this doesn't add nodes; it enriches the matching faculty record with a `struktural`
# role, folding into the same snapshot. Directly answers the confirmed-live "who is the head
# of the CS program" query (-> Head of Computer Science Program - Kemanggisan / Alam Sutera).
_LEADERSHIP_PAGE_URL = "https://socs.binus.ac.id/people/"
# Anchored on the people-link anchor's href FIRST, because within each card the DOM order is
# href -> name -> description; matching name first would grab the NEXT card's href and shift
# every role onto the wrong person (a real bug caught in testing). All three fields now come
# from the same card. Returns (profile_url, name, role).
_LEADERSHIP_CARD_RE = re.compile(
    r'href="(https://socs\.binus\.ac\.id/people/[^"]+/)"\s+class="people-link'
    r'.*?<p class="people-name">(.*?)</p>'
    r'.*?<p class="people-description">(.*?)</p>',
    re.DOTALL,
)


def _scrape_leadership_roles(roster_name_to_code: dict[str, str]) -> dict[str, str]:
    """{lecturer code -> structural role} from the SoCS org-chart page. A person may hold
    several roles (e.g. Head of a Department AND Head of a Program) -> joined with '; '.
    Code resolved via the profile page's D-code, else a roster name-match. Best-effort: a
    failed page/profile just yields fewer roles, never an error."""
    try:
        page = _http_get(_LEADERSHIP_PAGE_URL)
    except Exception:
        return {}
    roles: dict[str, str] = {}
    for profile_url, name_html, role_html in _LEADERSHIP_CARD_RE.findall(page):
        name = html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", name_html)).strip())
        role = html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", role_html)).strip())
        if not name or not role:
            continue
        # Resolve the code by NAME first (it's the same-card, reliably-aligned field);
        # the profile fetch is only a fallback for a leader not in the roster.
        code = roster_name_to_code.get(_norm_person_name(name))
        if not code:
            code, _ = _profile_code_and_name(profile_url)
        if code:
            roles[code] = f"{roles[code]}; {role}" if code in roles else role
        time.sleep(0.03)
    logger.info("leadership scrape: %d structural roles", len(roles))
    return roles


def _faculty_node(
    name: str, code: Optional[str], role: Optional[str], dept: Optional[str],
    year: Optional[str], courses: list[str], campuses: list[str],
    source_url: str, ingested_at: str, citation_unit: str,
    struktural: Optional[str] = None,
) -> TextNode:
    node_text = _scrub_injection_attempts(
        _faculty_node_text(name, code, role, dept, year, courses, campuses or None, struktural),
        source_url,
    )
    return TextNode(
        text=node_text,
        metadata={
            "source_file": source_url,
            "ingested_at": ingested_at,
            "parent_text": node_text,
            "section_title": f"Dosen: {name}",
            # Each lecturer is an independently-citable unit even though all share one
            # source_file -- see generation._source_key. Without this, a "who teaches X"
            # query would collapse every matching lecturer into one block.
            "citation_unit": citation_unit,
        },
    )


def _scrape_faculty_records(url: str) -> list[dict]:
    """The network-heavy half: scrape the School-wide roster, crawl the 7 campus pages, and
    enrich each lecturer via the scholar APIs -> a list of plain per-lecturer records (name,
    rank, dept, year, courses, campuses). This is the ONLY part that touches the fragile
    external sources; its output is cached to a snapshot so a routine reindex never re-runs
    it (see _faculty_roster_nodes / refresh_faculty_snapshot). A superset: every roster
    lecturer plus campus-page people the roster missed; one bad row/page is skipped, not fatal.
    """
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return []
    text = trafilatura.extract(downloaded, url=url)
    if not text:
        return []

    # 1) the School-wide roster (code -> name), first-seen order preserved.
    roster: dict[str, str] = {}
    for match in _FACULTY_ROW_RE.finditer(text):
        code, name = match.group(1), match.group(2).strip()
        roster.setdefault(code, name)
    roster_name_to_code = {_norm_person_name(name): code for code, name in roster.items()}

    # 2) campus index across all 7 CS campuses, and 3) structural/leadership roles.
    campus_by_code, name_only = _build_campus_index(roster_name_to_code)
    leadership_by_code = _scrape_leadership_roles(roster_name_to_code)

    records: list[dict] = []
    # 4) union of codes: roster order first, then campus/leadership codes the roster missed.
    extra_codes = [c for c in (campus_by_code.keys() | leadership_by_code.keys()) if c not in roster]
    for code in list(roster) + extra_codes:
        name, role, dept, detail_ok = _lecturer_detail(code)
        name = roster.get(code) or name or code
        year, courses, courses_complete = _lecturer_recent_courses(code)
        record = {
            "citation_unit": code, "code": code, "name": name, "rank": role,
            "dept": dept, "year": year, "courses": courses,
            "campuses": sorted(campus_by_code.get(code, ())),
            "struktural": leadership_by_code.get(code),
        }
        # Marked, not dropped: a partially-read lecturer is still worth having. The flag lets
        # refresh_faculty_snapshot decline to overwrite better cached data with this record, and
        # is stripped before anything is written (see _save_faculty_snapshot).
        if not (detail_ok and courses_complete):
            record[_SCAN_INCOMPLETE_KEY] = True
        records.append(record)
        time.sleep(0.05)

    # 4) campus-page people with no code and no roster match: minimal name+campus records.
    for key, (display, campuses) in name_only.items():
        records.append({
            "citation_unit": f"name:{key}", "code": None, "name": display, "rank": None,
            "dept": None, "year": None, "courses": [], "campuses": sorted(campuses),
        })

    logger.info(
        "faculty scrape: %d records (%d roster, %d campus-only codes, %d name-only) from %s",
        len(records), len(roster),
        len(campus_by_code) - len(set(campus_by_code) & set(roster)), len(name_only), url,
    )
    return records


def _faculty_records_to_nodes(records: list[dict], url: str) -> list[TextNode]:
    """Pure, offline: rebuild the faculty TextNodes from cached records -- no network. This is
    what a routine reindex runs, so the fragile crawl never fires unless explicitly refreshed."""
    ingested_at = datetime.now(timezone.utc).isoformat()
    return [
        _faculty_node(
            r["name"], r.get("code"), r.get("rank"), r.get("dept"),
            r.get("year"), r.get("courses") or [], r.get("campuses") or [],
            url, ingested_at, r["citation_unit"], r.get("struktural"),
        )
        for r in records
    ]


def _load_faculty_snapshot() -> Optional[list[dict]]:
    path = settings.faculty_snapshot_path
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("faculty snapshot at %s is unreadable; will re-scrape", path)
        return None
    return data if isinstance(data, list) and data else None


def _save_faculty_snapshot(records: list[dict]) -> None:
    records = [{k: v for k, v in r.items() if k != _SCAN_INCOMPLETE_KEY} for r in records]
    settings.faculty_snapshot_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8"
    )


def _lecturers_with_courses(records: list[dict]) -> int:
    return sum(1 for r in records if r.get("courses"))


def _restore_unreadable_teaching(fresh: list[dict], existing: list[dict]) -> int:
    """For each fresh record whose scan couldn't read the newer years, keep whatever the previous
    snapshot already knew when that is at least as recent. Returns how many were restored.

    Without this, one transient API failure per lecturer silently rewrites their most-recent
    teaching year BACKWARDS, and no record-count guard can see it: every record is present, every
    field is populated, the year is just wrong. A "who teaches X" answer is built from exactly
    these course lists, so a backdated year quietly changes who the bot says teaches what.
    """
    by_unit = {r.get("citation_unit"): r for r in existing}
    restored = 0
    for record in fresh:
        if not record.get(_SCAN_INCOMPLETE_KEY):
            continue
        cached = by_unit.get(record.get("citation_unit"))
        if not cached or not cached.get("courses"):
            continue
        # Restore when the fresh scan found nothing at all, or when the cache knows a NEWER year
        # than the fallback year this scan settled on. Never when the fresh year is newer: that
        # is real new information, incomplete scan or not.
        if not record.get("courses") or (cached.get("year") or "") > (record.get("year") or ""):
            record["year"] = cached.get("year")
            record["courses"] = list(cached.get("courses") or [])
            restored += 1
    return restored


def refresh_faculty_snapshot(url: str = FACULTY_ROSTER_URL, min_fraction: float = 0.9) -> dict:
    """Force a fresh crawl (the ONLY path that re-scrapes) and overwrite the cached snapshot.

    Two guardrails, because a degraded crawl has two very different shapes.

    Record count: if the fresh crawl returns notably fewer records than the cache
    (< min_fraction), a rotated token or a restyled BINUS page has broken the roster scrape, and
    the good snapshot is KEPT.

    Lecturers with courses: the count check passes untouched when the roster page scrapes fine but
    the scholar API is down, because every name is still there -- just with no rank, no department
    and no courses. That is the more likely outage of the two (two APIs and a bearer token versus
    one HTML page) and it was invisible to the old guard. Course lists are what a who-teaches
    answer is built from, so they get their own threshold.

    Records whose scan was incomplete also have their cached year/courses restored first, so a
    per-lecturer blip can't backdate a teaching year (see _restore_unreadable_teaching).

    Returns {records, wrote, reason, incomplete, restored}.
    """
    fresh = _scrape_faculty_records(url)
    existing = _load_faculty_snapshot() or []
    incomplete = sum(1 for r in fresh if r.get(_SCAN_INCOMPLETE_KEY))
    restored = _restore_unreadable_teaching(fresh, existing) if existing else 0
    if incomplete:
        logger.warning(
            "refresh_faculty_snapshot: %d/%d records had an unreadable scholar response; "
            "restored cached teaching data for %d of them",
            incomplete, len(fresh), restored,
        )

    def kept(reason: str) -> dict:
        logger.warning("refresh_faculty_snapshot: %s", reason)
        return {"records": existing, "wrote": False, "reason": reason,
                "incomplete": incomplete, "restored": restored}

    if not fresh:
        return kept("fresh crawl returned nothing")
    if existing and len(fresh) < len(existing) * min_fraction:
        return kept(
            f"fresh crawl {len(fresh)} < {min_fraction:.0%} of cached {len(existing)}; kept cache"
        )
    if existing:
        fresh_taught = _lecturers_with_courses(fresh)
        cached_taught = _lecturers_with_courses(existing)
        if cached_taught and fresh_taught < cached_taught * min_fraction:
            return kept(
                f"fresh crawl has courses for {fresh_taught} lecturers < {min_fraction:.0%} of "
                f"cached {cached_taught}; scholar API likely degraded; kept cache"
            )

    _save_faculty_snapshot(fresh)
    logger.info(
        "refresh_faculty_snapshot: wrote %d records to %s", len(fresh), settings.faculty_snapshot_path
    )
    return {"records": fresh, "wrote": True, "reason": None,
            "incomplete": incomplete, "restored": restored}


def _faculty_roster_nodes(url: str) -> list[TextNode]:
    """Faculty nodes for scrape_url / reindex. Reads the cached snapshot and rebuilds nodes
    OFFLINE when present; only bootstraps a live scrape (and caches it) when no snapshot
    exists yet. So the fragile crawl fires once ever, then never again on routine reindex --
    refreshing the data is the explicit, separate refresh_faculty_snapshot / admin action."""
    records = _load_faculty_snapshot()
    if records is None:
        logger.info("scrape_url: no faculty snapshot yet -> bootstrapping one live crawl")
        records = _scrape_faculty_records(url)
        if records:
            _save_faculty_snapshot(records)
    else:
        logger.info("scrape_url: faculty rebuilt from snapshot (%d records, offline)", len(records))
    return _faculty_records_to_nodes(records, url)


def scrape_url(url: str) -> list[TextNode]:
    """Fetch and extract clean text from a web page, chunked like any other document.

    Pure fetch -- no persistence side effects. Callers that want the URL re-fetched by
    a future /admin/reindex must call record_scraped_url(url) themselves after a
    successful scrape (see the /admin/documents/url route).
    """
    # Structured-source special case (like the tuition table below): the faculty list needs
    # per-lecturer API enrichment, not generic chunking, so it does its own fetch.
    if _FACULTY_LIST_URL_RE.match(url):
        return _faculty_roster_nodes(url)

    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return []

    text = trafilatura.extract(downloaded, url=url)
    if not text:
        return []

    text = _scrub_injection_attempts(text, url)

    if _TUITION_FEE_URL_RE.match(url):
        row_nodes = _tuition_fee_row_nodes(url, text)
        if row_nodes:
            return row_nodes
        # Falls through to generic chunking below if the page's table structure ever
        # stops matching what _tuition_fee_row_nodes expects, so a format change on
        # BINUS's site degrades to the old (still functional) behavior rather than
        # silently dropping the page from the KB entirely.

    ingested_at = datetime.now(timezone.utc).isoformat()
    nodes: list[TextNode] = []
    triples, _ = _parent_child_split(text)
    for child_text, parent_text, section_title in triples:
        metadata = {"source_file": url, "ingested_at": ingested_at, "parent_text": parent_text}
        if section_title is not None:
            metadata["section_title"] = section_title
        nodes.append(TextNode(text=child_text, metadata=metadata))
    return nodes


def _load_url_cache() -> dict[str, list[dict]]:
    """The last-known-good cache: {url: [serialized TextNode, ...]}. Returns {} if the file
    is absent or unreadable (a corrupt cache must never block a reindex -- worst case the
    fallback simply isn't available for that run)."""
    path = settings.url_cache_path
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("URL cache at %s is unreadable; ignoring it this run", path)
        return {}
    return data if isinstance(data, dict) else {}


def _save_url_cache(cache: dict[str, list[dict]]) -> None:
    settings.url_cache_path.write_text(
        json.dumps(cache, ensure_ascii=False), encoding="utf-8"
    )


def _cache_url_nodes(url: str, nodes: list[TextNode]) -> None:
    """Record the chunks a SUCCESSFUL scrape of `url` produced, as this URL's last-known-good.
    No-op for an empty list, so a failed scrape never overwrites good cached content."""
    if not nodes:
        return
    cache = _load_url_cache()
    cache[url] = [n.to_dict() for n in nodes]
    _save_url_cache(cache)


def scrape_url_cached(url: str) -> tuple[list[TextNode], bool]:
    """scrape_url with a last-known-good fallback (IMPROVEMENTS.md #6.x / the periodic-rescrape
    open item's silent-loss risk). Returns (nodes, from_cache).

    A live scrape that returns chunks is authoritative: those chunks are used AND written to the
    cache as the new last-known-good. If the live scrape comes back empty (page moved, network
    blip, restyled HTML, rate-limit), the previously-cached chunks are returned instead so the
    URL's content degrades to stale-but-present rather than vanishing from the rebuilt index.
    Only truly-never-scraped-successfully URLs (no live result and no cache) return ([], False).
    """
    nodes = scrape_url(url)
    if nodes:
        _cache_url_nodes(url, nodes)
        return nodes, False
    cached = _load_url_cache().get(url)
    if cached:
        restored = [TextNode.from_dict(d) for d in cached]
        logger.warning(
            "Reindex: re-fetch of %s returned nothing; restored %d chunk(s) from last-known-good cache",
            url, len(restored),
        )
        return restored, True
    return [], False


def forget_url_cache(url: str) -> None:
    """Drop a URL's cached chunks -- called when the URL is deliberately removed, so a later
    re-add doesn't silently resurrect stale content from before the removal."""
    cache = _load_url_cache()
    if cache.pop(url, None) is not None:
        _save_url_cache(cache)


def delete_document_nodes(index: VectorStoreIndex, filename: str) -> int:
    """Remove all chunks belonging to filename from both the vector store and the docstore."""
    vector_store = index.vector_store
    collection = vector_store._collection
    matches = collection.get(where={"source_file": filename}, include=[])
    node_ids = matches["ids"]

    if node_ids:
        collection.delete(ids=node_ids)
        for node_id in node_ids:
            # A node present in the Chroma collection but absent from the docstore (the two
            # can drift -- Chroma writes are immediate, the docstore is persisted separately)
            # must not abort the delete: this llama-index docstore raises ValueError, not
            # KeyError, for a missing doc_id, so both are treated as "already gone".
            try:
                index.docstore.delete_document(node_id)
            except (KeyError, ValueError):
                pass

    return len(node_ids)


def _is_finite_embedding(embedding: list[float]) -> bool:
    return all(math.isfinite(x) for x in embedding)


def _embed_nodes_with_nan_guard(nodes: list[TextNode]) -> list[TextNode]:
    """Embeds every node up front, in batches, guarding against a transient NaN/Inf
    embedding (IMPROVEMENTS.md #7.4) instead of letting it crash the whole index build.

    Confirmed live: an admin's /admin/reindex click failed with chromadb's
    "Embeddings must not contain NaN or Infinity values", which had already DELETED the
    old collection by the time the crash happened, leaving the live KB empty until
    rebuilt again -- this validates against exactly that failure mode, and does so
    BEFORE build_index touches Chroma at all, so a bad embedding can never again cost an
    already-working collection.

    A batch-level NaN is treated as a transient GPU/batch-encoding glitch first (retry
    the whole batch once, same shape of fix IMPROVEMENTS.md #7.4 already called for) --
    only if the retry ALSO fails does it drop to embedding that batch's nodes one at a
    time, to isolate and drop just the specific node(s) actually responsible rather than
    losing the whole batch. Nodes get their `.embedding` set directly so
    VectorStoreIndex reuses it instead of re-embedding (llama_index's embed_nodes()
    skips any node whose `.embedding` is already non-None).
    """
    good_nodes: list[TextNode] = []
    batch_size = settings.embedding_batch_size
    for start in range(0, len(nodes), batch_size):
        batch = nodes[start : start + batch_size]
        texts = [n.get_content(metadata_mode=MetadataMode.EMBED) for n in batch]
        embeddings = LlamaSettings.embed_model.get_text_embedding_batch(texts)
        if not all(_is_finite_embedding(e) for e in embeddings):
            logger.warning(
                "Batch embedding produced a NaN/Inf vector (nodes %d-%d) -- retrying batch",
                start, start + len(batch) - 1,
            )
            embeddings = LlamaSettings.embed_model.get_text_embedding_batch(texts)

        if all(_is_finite_embedding(e) for e in embeddings):
            for node, embedding in zip(batch, embeddings):
                node.embedding = embedding
                good_nodes.append(node)
            continue

        # Still bad after a full-batch retry -- isolate node-by-node so only the
        # specific offending node(s) are dropped, not the entire batch.
        for node, text in zip(batch, texts):
            embedding = LlamaSettings.embed_model.get_text_embedding(text)
            if not _is_finite_embedding(embedding):
                embedding = LlamaSettings.embed_model.get_text_embedding(text)
            if not _is_finite_embedding(embedding):
                logger.warning(
                    "Dropping node from %s after repeated NaN/Inf embedding: %r",
                    node.metadata.get("source_file"), text[:120],
                )
                continue
            node.embedding = embedding
            good_nodes.append(node)

    return good_nodes


def build_index(nodes: list[TextNode]) -> VectorStoreIndex:
    """Embeds every node (with a NaN/Inf guard, see _embed_nodes_with_nan_guard), then
    deletes and recreates the Chroma collection and builds a fresh index. Embedding
    happens BEFORE the old collection is touched, so a bad embedding can no longer wipe
    an already-working KB out from under a failed rebuild.
    """
    nodes = _embed_nodes_with_nan_guard(nodes)

    client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))

    try:
        client.delete_collection(settings.chroma_collection_name)
    except Exception:
        pass

    collection = client.create_collection(settings.chroma_collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, docstore=SimpleDocumentStore()
    )

    # store_nodes_override: ChromaVectorStore.stores_text=True means LlamaIndex otherwise
    # skips writing nodes into the docstore at all -- BM25Retriever needs them there.
    index = VectorStoreIndex(nodes, storage_context=storage_context, store_nodes_override=True)
    storage_context.persist(persist_dir=str(settings.chroma_persist_dir))
    return index


def load_index() -> Optional[VectorStoreIndex]:
    """Load the existing persisted index (vector store + docstore), or None if missing/empty."""
    client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
    try:
        collection = client.get_collection(settings.chroma_collection_name)
    except Exception:
        return None

    if collection.count() == 0:
        return None

    vector_store = ChromaVectorStore(chroma_collection=collection)
    try:
        docstore = SimpleDocumentStore.from_persist_dir(
            persist_dir=str(settings.chroma_persist_dir)
        )
    except FileNotFoundError:
        docstore = SimpleDocumentStore()

    storage_context = StorageContext.from_defaults(
        docstore=docstore, vector_store=vector_store
    )
    return VectorStoreIndex(
        nodes=[], storage_context=storage_context, store_nodes_override=True
    )
