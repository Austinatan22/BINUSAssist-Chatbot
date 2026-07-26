# SOCS Document Collection

Collected 2026-07-07 for the BINUSAssist chatbot knowledge base. All curriculum PDFs are
the official catalogs from the **BINUS Curriculum Center** (https://curriculum.binus.ac.id),
which is the same source the existing KB documents came from. Each program catalog covers
the **high- and medium-priority** items in one document: program overview, **graduate
competencies / learning outcomes**, **curriculum / course structure** (semester-by-semester),
specialization streams, and **career prospects**.

## SOCS @Greater Jakarta undergraduate programs (10/10)

| File | Program | Catalog | Source PDF |
|------|---------|---------|------------|
| Computer_Science_2026.pdf | Computer Science | 2026/B2030 | curriculum.binus.ac.id/files/2012/04/SOCS-Computer-Science-2026-R0.pdf |
| Computer_Science_Global_Class_2026.pdf | Computer Science – Global Class | 2026/B2030 | .../2013/08/SOCS-Computer-Science-GC-2026-R0.pdf |
| Mathematics_and_Computer_Science_2026.pdf | Mathematics & Computer Science | 2026/B2030 | .../2012/05/SOCS-Mathematics-and-Computer-Science-2026-R0.pdf |
| Statistics_and_Computer_Science_2026.pdf | Statistics & Computer Science | 2026/B2030 | .../2012/06/SOCS-Statistics-and-Computer-Science-2026-R0.pdf |
| Software_Engineering_2026.pdf | Software Engineering | 2026/B2030 | .../2023/09/SOCS-Software-Engineering-2026-R0.pdf |
| Data_Science_2026.pdf | Data Science | 2026/B2030 | .../2022/03/SOCS-Data-Science-2026-R0.pdf |
| Artificial_Intelligence_2025.pdf | Artificial Intelligence | 2025/B2029 | .../2024/08/SOCS-Artificial-Intelligence-2025-R0.pdf |
| Mobile_Application_and_Technology_2023.pdf | Mobile Application & Technology | 2023/B2027 | .../2012/04/SOCS-Mobile-Application-Technology-2023.pdf |
| **Cyber_Security_2025.pdf** | **Cyber Security** | 2025/B2029 | .../2015/04/SOCS-Cyber-Security-2025-R0.pdf |
| **Game_Application_and_Technology_2026.pdf** | **Game Application & Technology** | 2026/B2030 | .../2012/08/SOCS-Game-Application-Technology-2026-R0.pdf |

## New to the knowledge base (were missing)

- **Cyber_Security_2025.pdf** — no Cyber Security document was in the KB.
- **Game_Application_and_Technology_2026.pdf** — no Game App & Tech document was in the KB.
- **Data_Science_2026.pdf** — KB had only the older 2025 catalog; this is the current one.

The other 7 programs already exist in `backend/documents/` at the same or matching catalog
year — included here so the full SOCS set is in one place. Copy just the three above into
`backend/documents/` and re-run `python scripts/seed_kb.py` to index the gaps.

## Prospective-student reference notes (not per-program PDFs)

These cover what the curriculum catalogs don't: admission, money, life, outcomes. They are
university-wide living web pages, so the cleanest way to index them is the admin panel's
**Add URL** feature (so they refresh from source) rather than the .md files. Amounts and
requirements change per intake — re-verify before relying on them.

| Note file | Covers | Source URL to add via admin |
|-----------|--------|------------------------------|
| `Admission_Requirements_2026-2027.md` | Entry requirements + step-by-step procedure | https://gabung.binus.ac.id/admission-requirement/ and https://gabung.binus.ac.id/admission-procedure |
| `Tuition_Fees_Computer_Science.md` | Tuition, fees, structure | https://gabung.binus.ac.id/tuition-fee/ |
| `Scholarships.md` | StarTech (SOCS), Widia, TPKS, others | https://binus.ac.id/scholarship/list/ |
| `Why_SOCS_Facilities_Student_Life.md` | Why SOCS, facilities, HIMTI/clubs | https://socs.binus.ac.id/ |
| `Alumni_and_Career_Outcomes.md` | Career paths, testimonial, employability | https://socs.binus.ac.id/computer-science/ |

Note: several alumni/employability claims in `Alumni_and_Career_Outcomes.md` are flagged
*(unverified)* — they come from third-party aggregation, not an official BINUS page. Confirm
with the School before publishing.

## KB cleanup performed 2026-07-07

Scoped the KB down to SOCS only. `backend/documents/` now holds exactly the 10 SOCS catalogs
above. Moved out of the KB (archived, not deleted — fully reversible):
- `backend/documents/_archive/` — 77 out-of-scope catalogs (Business, Design, Information
  Systems, Communication, Engineering, Hotel/Tourism, Law, Accounting, Finance, etc.).
- `backend/documents/_archive/borderline/` — 3 supervisor's-call catalogs: Digital Psychology
  ×2 (SOCS only at regional campuses), Computer Science – International (BINUS International).
- Superseded duplicates archived: `Computer_Science_2025.pdf`, `Data_Science_2025.pdf`.

**Not yet re-seeded** — run `python scripts/seed_kb.py` to rebuild the index on the clean 10
when ready. The Chroma vectorstore still contains the old 87-doc index until then.

## Not collected (require the head of program)

Accreditation certificates (BAN-PT / int'l), faculty research profiles, official current
alumni-outcome data (employment rate, starting salary, top hiring companies), and internal
double-degree agreements are not published as downloadable documents — request these directly.
