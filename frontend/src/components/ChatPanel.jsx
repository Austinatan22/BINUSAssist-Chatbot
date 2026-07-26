import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { PreviewCard } from "@base-ui/react/preview-card";
import {
  ArrowDown,
  ArrowRight,
  Check,
  ChevronLeft,
  ChevronRight,
  Copy,
  ExternalLink,
  FileText,
  Mail,
  MessageCircle,
  MessageSquarePlus,
  RotateCcw,
  Sparkles,
  Square,
  ThumbsDown,
  ThumbsUp,
} from "lucide-react";
import { Textarea } from "@/components/ui/textarea";
import { API_URL, sourceUrl } from "@/lib/api";

// Source lookup by citation id, provided per-message by MessageBubble so CitationPill can
// render a hover preview without threading source data through every pure render function
// (renderContentWithCitations, MarkdownList, ...). A plain object keyed by String(id).
const SourcesContext = createContext(null);

// How close to the bottom (px) counts as "already there" for auto-scroll purposes.
const NEAR_BOTTOM_PX = 120;

// How tall the top/bottom scroll-edge fade reads, in px.
const FADE_SIZE = "40px";

// A mask (not a color overlay) so the edge fade always matches whatever is actually
// behind the scroll area -- including the chat card's own gradient background -- with
// no risk of the fade's assumed color drifting out of sync with it. An earlier version
// used two absolutely-positioned divs with a `from-card to-transparent` gradient, which
// broke the moment the card's background stopped being a flat, single color: the fade's
// solid "from" color no longer matched the gradient-tinted background at that position,
// producing a visible hard-edged rectangle instead of a smooth fade. Masking the
// scrollable element itself sidesteps the problem entirely -- it fades the real pixels
// in place rather than trying to paint a matching color on top of them. `direction`
// lets the same helper serve both the vertical message list ("to bottom") and the
// horizontal follow-up chip strip ("to right").
function scrollEdgeMask(showStart, showEnd, direction = "to bottom") {
  if (!showStart && !showEnd) return undefined;
  const stops = [
    showStart ? "transparent 0%" : "black 0%",
    ...(showStart ? [`black ${FADE_SIZE}`] : []),
    ...(showEnd ? [`black calc(100% - ${FADE_SIZE})`] : []),
    showEnd ? "transparent 100%" : "black 100%",
  ];
  return `linear-gradient(${direction}, ${stops.join(", ")})`;
}

export default function ChatPanel({
  messages,
  sendMessage,
  regenerate,
  isStreaming,
  onStop,
  onCiteClick,
  onNewChat,
}) {
  const [input, setInput] = useState("");
  const [starterQuestions, setStarterQuestions] = useState([]);
  const [starterQuestionsLoading, setStarterQuestionsLoading] = useState(true);
  const bottomRef = useRef(null);
  const textareaRef = useRef(null);
  const scrollRef = useRef(null);
  const [showTopFade, setShowTopFade] = useState(false);
  const [showBottomFade, setShowBottomFade] = useState(false);
  // Ref (not state) so the streaming-driven scroll effect below always reads the
  // latest value without needing to be an effect dependency itself -- a plain state
  // read there would close over a stale value from before the user's last scroll.
  const isNearBottomRef = useRef(true);
  const [showScrollButton, setShowScrollButton] = useState(false);

  const scrollToBottom = (behavior = "smooth") => {
    isNearBottomRef.current = true;
    setShowScrollButton(false);
    bottomRef.current?.scrollIntoView({ behavior });
  };

  const updateFades = () => {
    const el = scrollRef.current;
    if (!el) return;
    setShowTopFade(el.scrollTop > 0);
    setShowBottomFade(el.scrollTop + el.clientHeight < el.scrollHeight - 1);
    const nearBottom =
      el.scrollHeight - el.scrollTop - el.clientHeight < NEAR_BOTTOM_PX;
    isNearBottomRef.current = nearBottom;
    setShowScrollButton(!nearBottom);
  };

  // Streaming yanking the view back down every token is disorienting if the user has
  // scrolled up to re-read an earlier answer -- only auto-follow when they were already
  // near the bottom; otherwise leave them where they are and surface the "jump to
  // latest" button instead (see showScrollButton above).
  useEffect(() => {
    if (isNearBottomRef.current) {
      bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  useEffect(() => {
    updateFades();
  }, [messages, starterQuestions]);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const observer = new ResizeObserver(updateFades);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  // Grows the textarea line-by-line as text wraps, up to 5 lines, then switches to an
  // internal scrollbar instead of growing further.
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    const lineHeight = parseFloat(getComputedStyle(el).lineHeight);
    const maxHeight = lineHeight * 5;
    el.style.height = "auto";
    const contentHeight = el.scrollHeight;
    el.style.height = `${Math.min(contentHeight, maxHeight)}px`;
    el.style.overflowY = contentHeight > maxHeight ? "auto" : "hidden";
  }, [input]);

  useEffect(() => {
    fetch(`${API_URL}/config/starter-questions`)
      .then((res) => (res.ok ? res.json() : []))
      .then(setStarterQuestions)
      .catch(() => setStarterQuestions([]))
      .finally(() => setStarterQuestionsLoading(false));
  }, []);

  const handleSend = () => {
    const text = input.trim();
    if (!text) return;
    setInput("");
    sendMessage(text);
    // The user just asked to see a new answer -- jump to it regardless of wherever
    // they'd scrolled to read something earlier, same as any chat app's send action.
    scrollToBottom("auto");
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Same "finished, latest, not a fallback" condition MessageBubble used to gate the
  // now-removed inline follow-up block -- an older turn's suggestions would be stale
  // once the conversation has moved on, and mid-stream there's nothing to suggest yet.
  const lastMessage = messages[messages.length - 1];
  const followUpQuestions =
    lastMessage?.role === "assistant" &&
    lastMessage.content &&
    !isStreaming &&
    !lastMessage.fallback
      ? (lastMessage.followUps ?? [])
      : [];

  return (
    <div className="flex flex-col h-full gap-4">
      <div className="flex items-center gap-2">
        {onNewChat && messages.length > 0 && (
          <button
            onClick={onNewChat}
            className="flex items-center gap-1.5 self-start text-sm text-muted-foreground hover:text-primary bg-muted/50 hover:bg-muted rounded-md px-3 py-1.5 transition"
          >
            <MessageSquarePlus size={16} />
            New Chat
          </button>
        )}
      </div>
      <div className="relative flex-1 overflow-hidden">
        <div
          ref={scrollRef}
          onScroll={updateFades}
          className="h-full overflow-y-auto"
          style={{
            maskImage: scrollEdgeMask(showTopFade, showBottomFade),
            WebkitMaskImage: scrollEdgeMask(showTopFade, showBottomFade),
          }}
        >
          <div className="w-full max-w-2xl mx-auto space-y-4 h-full">
            {messages.length === 0 && (
              <div className="flex flex-col items-center justify-center gap-8 h-full animate-message-in">
                <div className="flex flex-col items-center gap-4 text-center">
                  <div className="flex items-center justify-center p-2 rounded-2xl bg-gradient-to-br from-primary/15 via-accent/10 to-primary/5">
                    <img
                      src="/binus%20icon.png"
                      alt=""
                      className="h-12 w-12 object-contain"
                    />
                  </div>
                  <div className="flex flex-col items-center gap-1.5">
                    <p className="text-3xl font-semibold text-foreground">
                      How can we help?
                    </p>
                    <p className="text-sm text-muted-foreground max-w-sm">
                      Ask about programs, tuition, admissions, or anything else
                      about BINUS School of Computer Science.
                    </p>
                  </div>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full max-w-xl">
                  {starterQuestionsLoading
                    ? Array.from({ length: 4 }).map((_, i) => (
                        <div
                          key={i}
                          className="flex items-center gap-3 rounded-xl border border-border bg-card px-4 py-3 shadow-sm animate-pulse"
                        >
                          <span className="h-7 w-7 shrink-0 rounded-lg bg-muted" />
                          <span className="h-3 flex-1 rounded bg-muted" />
                        </div>
                      ))
                    : starterQuestions.map((q) => (
                        <button
                          key={q}
                          onClick={() => sendMessage(q)}
                          className="group flex items-center gap-3 text-left rounded-xl border border-border bg-card px-4 py-3 shadow-sm hover:shadow-md hover:border-primary/40 transition"
                        >
                          <span className="flex items-center justify-center h-7 w-7 shrink-0 rounded-lg bg-gradient-to-br from-primary/15 to-primary/5 text-primary">
                            <Sparkles size={14} />
                          </span>
                          <span className="text-sm text-foreground/90 group-hover:text-foreground">
                            {q}
                          </span>
                        </button>
                      ))}
                </div>
              </div>
            )}
            {messages.map((message, i) => (
              <MessageBubble
                key={i}
                message={message}
                question={
                  message.role === "assistant"
                    ? messages[i - 1]?.content
                    : undefined
                }
                onCiteClick={(sourceId) => onCiteClick(i, sourceId)}
                isLast={i === messages.length - 1}
                isStreaming={isStreaming}
                onRegenerate={regenerate}
                onSendSuggestion={sendMessage}
                starterQuestions={starterQuestions}
              />
            ))}
            <div ref={bottomRef} />
          </div>
        </div>
        {showScrollButton && (
          <button
            onClick={() => scrollToBottom()}
            aria-label="Scroll to latest messages"
            className="absolute bottom-4 left-1/2 -translate-x-1/2 flex items-center gap-1.5 rounded-full bg-card/90 backdrop-blur-sm border border-border shadow-md px-3 py-1.5 text-xs text-muted-foreground hover:text-primary hover:border-primary transition"
          >
            <ArrowDown size={14} />
            Scroll to latest
          </button>
        )}
      </div>
      {followUpQuestions.length > 0 && (
        <div className="w-full max-w-2xl mx-auto">
          <FollowUpBar questions={followUpQuestions} onSelect={sendMessage} />
        </div>
      )}
      <div className="w-full max-w-2xl mx-auto">
        <div className="flex flex-col rounded-xl border border-input bg-background shadow-sm p-4">
          <Textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type your question..."
            className="rounded-none min-h-0 resize-none border-0 bg-background px-2 py-0 shadow-none focus-visible:ring-0 dark:bg-background [scrollbar-gutter:stable] transition-none"
            rows={1}
          />
          <div className="flex items-center justify-end mt-2">
            <button
              onClick={isStreaming ? onStop : handleSend}
              disabled={!isStreaming && !input.trim()}
              aria-label={isStreaming ? "Stop generating" : "Send"}
              title={isStreaming ? "Stop generating" : "Send"}
              className="flex items-center justify-center h-8 w-8 rounded-md bg-gradient-to-br from-primary to-primary/80 text-primary-foreground shadow-sm transition hover:brightness-110 disabled:opacity-40 disabled:hover:brightness-100"
            >
              {isStreaming ? (
                <Square size={13} fill="currentColor" />
              ) : (
                <ArrowRight size={16} />
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// Orange (accent), not blue (primary), is this app's citation/sources identity color --
// deliberately given its own consistent color across every citation-related element
// (this pill, the preview card's own badge below, SourcePanel's badges, the collapsed-
// sidebar source chips) so "orange = a citation/source" reads as one coherent visual
// language, distinct from primary's general "interactive element" blue.
const PILL_CLASSES =
  "inline-flex items-center justify-center mx-0.5 h-3.5 min-w-3.5 px-0.5 rounded-sm bg-accent/15 text-accent text-[8px] font-medium align-[15%] hover:bg-accent/25 transition";

function CitationPill({ id, onClick }) {
  const source = useContext(SourcesContext)?.[String(id)];

  // No source metadata for this id (e.g. a citation number the model emitted that has no
  // matching source, or a message rendered outside a provider) -> plain pill, click still
  // opens the sources panel as before.
  if (!source) {
    return (
      <button onClick={onClick} className={PILL_CLASSES}>
        {id}
      </button>
    );
  }

  return (
    <PreviewCard.Root>
      <PreviewCard.Trigger
        delay={250}
        closeDelay={100}
        onClick={onClick}
        render={<button type="button" className={PILL_CLASSES} />}
      >
        {id}
      </PreviewCard.Trigger>
      <PreviewCard.Portal>
        <PreviewCard.Positioner sideOffset={6} className="z-50">
          <PreviewCard.Popup className="max-w-xs rounded-lg border border-border bg-card/95 backdrop-blur-md shadow-lg p-3 text-left">
            <div className="flex items-start gap-2">
              <span className="flex items-center justify-center h-5 min-w-5 px-1 shrink-0 rounded-sm bg-accent/15 text-accent text-[10px] font-medium">
                {id}
              </span>
              <div className="min-w-0">
                <p className="text-xs font-medium text-foreground truncate flex items-center gap-1">
                  <FileText
                    size={11}
                    className="shrink-0 text-muted-foreground"
                  />
                  {source.display_name || source.source_file}
                </p>
                {source.section_title && (
                  <p className="text-[11px] text-muted-foreground truncate">
                    {source.section_title}
                  </p>
                )}
              </div>
            </div>
            {source.snippet && (
              <p className="mt-2 text-[11px] leading-snug text-muted-foreground line-clamp-4">
                {source.snippet}
                {source.snippet.length >= 200 ? "…" : ""}
              </p>
            )}
            <a
              href={sourceUrl(source)}
              target="_blank"
              rel="noopener noreferrer"
              onClick={(e) => e.stopPropagation()}
              className="mt-2 inline-flex items-center gap-1 text-[10px] text-accent/80 hover:text-accent hover:underline"
            >
              <ExternalLink size={10} className="shrink-0" />
              Open source
            </a>
          </PreviewCard.Popup>
        </PreviewCard.Positioner>
      </PreviewCard.Portal>
    </PreviewCard.Root>
  );
}

// Order matters: alternation tries each branch left-to-right at a given position, so
// bold (**) must come before italic (*) or "**bold**" would match italic's single-star
// pattern first (consuming just one leading "*" as literal text before the "real" match).
const TOKEN_PATTERN =
  /(\[\d+\]|https?:\/\/\S+|[\w.+-]+@[\w-]+\.[\w.-]+|\*\*[^*\n]+\*\*|`[^`\n]+`|\*[^*\n]+\*)/g;
const LINK_CLASSES = "underline text-primary hover:no-underline break-words";
// A source re-cited on every bullet of a long list clutters the answer without adding
// information -- once a given source has shown up this many times, later repeats of the
// same [n] are dropped (the first appearances already let the reader jump to that source).
const MAX_CITATION_REPEATS = 3;
// ...but only where dropping it can't break the sentence. A citation is a trailing
// DECORATION when the clause reads fine without it ("Data Science [2]"); it's
// grammatically LOAD-BEARING when it's the object of a connector ("described in [2] and
// [3]"). Reported live: the cap blanked load-bearing citations with no cleanup, leaving
// visibly broken prose -- "This program is described in 2 and ." / "provided in  and ,
// respectively." Sentence-breaking output reads as a bug and costs more credibility than
// a repeated pill ever saves, so the cap now yields whenever grammar depends on the mark.
// (The list-clutter case this cap was added for is separately, and more carefully, handled
// by collapseRepeatedItemCitations, which also cleans up the whitespace it leaves behind.)
const CITATION_CONNECTOR_RE = /\b(?:in|on|at|of|to|from|and|or|with|per|via|see)[\s(]*$/i;
const CITATION_TRAILING_RE = /^[\s.,;:)\]]*$/;

function isDroppableRepeat(parts, i) {
  if (CITATION_CONNECTOR_RE.test(parts.slice(0, i).join(""))) return false;
  return CITATION_TRAILING_RE.test(parts.slice(i + 1).join(""));
}

function renderContentWithCitations(
  content,
  onCiteClick,
  citationCounts = new Map(),
) {
  const parts = content.split(TOKEN_PATTERN);
  return parts.map((part, i) => {
    if (part == null || part === "") return null;
    const citationMatch = part.match(/^\[(\d+)\]$/);
    if (citationMatch) {
      const id = citationMatch[1];
      const count = (citationCounts.get(id) || 0) + 1;
      citationCounts.set(id, count);
      if (count > MAX_CITATION_REPEATS && isDroppableRepeat(parts, i)) return null;
      return (
        <CitationPill key={i} id={id} onClick={() => onCiteClick(Number(id))} />
      );
    }
    if (/^https?:\/\//.test(part)) {
      return (
        <a
          key={i}
          href={part}
          target="_blank"
          rel="noopener noreferrer"
          className={LINK_CLASSES}
        >
          {part}
        </a>
      );
    }
    if (/^[\w.+-]+@[\w-]+\.[\w.-]+$/.test(part)) {
      return (
        <a key={i} href={`mailto:${part}`} className={LINK_CLASSES}>
          {part}
        </a>
      );
    }
    const boldMatch = part.match(/^\*\*([^*\n]+)\*\*$/);
    if (boldMatch) {
      // Recurse so a citation/link inside the bold span (e.g. "**Rp 27,300,000** [1]"
      // written as one bolded run) still renders correctly, not as literal brackets.
      return (
        <strong key={i} className="font-semibold">
          {renderContentWithCitations(
            boldMatch[1],
            onCiteClick,
            citationCounts,
          )}
        </strong>
      );
    }
    const codeMatch = part.match(/^`([^`\n]+)`$/);
    if (codeMatch) {
      // Code spans are literal per CommonMark -- no nested emphasis/citation parsing.
      return (
        <code
          key={i}
          className="px-1 py-0.5 rounded bg-muted-foreground/15 text-[0.9em] font-mono"
        >
          {codeMatch[1]}
        </code>
      );
    }
    const italicMatch = part.match(/^\*([^*\n]+)\*$/);
    if (italicMatch) {
      return (
        <em key={i}>
          {renderContentWithCitations(
            italicMatch[1],
            onCiteClick,
            citationCounts,
          )}
        </em>
      );
    }
    return <span key={i}>{part}</span>;
  });
}

// A list where every bullet cites the same single source (common when one page covers
// several related facts) reads cleaner with the citation shown once rather than repeated
// down the whole list. Collapse a citation only when the exact same set of ids repeats on
// the immediately preceding list item -- if an item cites a different (or additional)
// source, it still gets its own mark.
const CITATION_ID_RE = /\[(\d+)\]/g;

function citationIdsInLine(line) {
  return Array.from(line.matchAll(CITATION_ID_RE), (m) => m[1]);
}

function sameCitationIds(a, b) {
  return (
    a.length > 0 && a.length === b.length && a.every((id, i) => id === b[i])
  );
}

// Every element of `items` IS a list item by construction (splitContentBlocks already
// grouped them), so this only needs to track the previous item's citation set -- no
// per-line "is this even a list item" check like the old whole-content version needed.
// Items are {indent, text} (see ORDERED_ITEM_RE/UNORDERED_ITEM_RE below) -- indent
// passes through untouched, only text is ever collapsed.
function collapseRepeatedItemCitations(items) {
  let prevIds = null;
  return items.map(({ indent, text }) => {
    const ids = citationIdsInLine(text);
    let result = text;
    if (prevIds !== null && sameCitationIds(ids, prevIds)) {
      result = text
        .replace(CITATION_ID_RE, "")
        .replace(/[ \t]+([.,;:])/g, "$1")
        .replace(/[ \t]+$/, "");
    }
    prevIds = ids;
    return { indent, text: result };
  });
}

// Markdown-table detection: a header row, a "|---|---|" separator row, then 1+ body
// rows, each starting/ending with "|". Answers citing tuition/course-structure tables
// (inherently tabular source data) render this shape; without special handling it shows
// up as raw, unreadable pipe-delimited text.
const TABLE_ROW_RE = /^\s*\|.+\|\s*$/;
const TABLE_SEPARATOR_RE = /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$/;
// Block-level line classifiers, checked in this order (a table is detected separately,
// via 2-line lookahead, before any of these run). ORDERED/UNORDERED capture leading
// whitespace (group 1) too -- indentation is the only depth signal the model gives for
// a nested sub-bullet (e.g. "* Core Subjects" with " + Programming" sub-items under
// it), used to render a visually indented flat list rather than fragmenting into
// separate list blocks or leaving the marker character showing as literal text.
const HEADING_RE = /^\s*(#{1,6})\s+(.+?)\s*$/;
const BLOCKQUOTE_RE = /^\s*>\s?(.*)$/;
const ORDERED_ITEM_RE = /^(\s*)(\d+)[.)]\s+(.*)$/;
const UNORDERED_ITEM_RE = /^(\s*)[-*+]\s+(.*)$/;

function parseTableRow(line) {
  return line
    .trim()
    .replace(/^\|/, "")
    .replace(/\|$/, "")
    .split("|")
    .map((cell) => cell.trim());
}

function lineKind(line) {
  if (HEADING_RE.test(line)) return "heading";
  if (BLOCKQUOTE_RE.test(line)) return "blockquote";
  if (ORDERED_ITEM_RE.test(line)) return "ordered";
  if (UNORDERED_ITEM_RE.test(line)) return "unordered";
  return "text";
}

// Splits message content into typed blocks (table/heading/blockquote/list/text) so each
// renders with the right HTML element instead of everything falling back to plain text
// with the raw markdown punctuation still visible.
function splitContentBlocks(content) {
  const lines = content.split("\n");
  const blocks = [];
  let i = 0;
  const isTableStart = (idx) =>
    TABLE_ROW_RE.test(lines[idx] ?? "") &&
    TABLE_SEPARATOR_RE.test(lines[idx + 1] ?? "");

  while (i < lines.length) {
    if (isTableStart(i)) {
      let j = i + 2;
      while (j < lines.length && TABLE_ROW_RE.test(lines[j])) j++;
      blocks.push({ type: "table", lines: lines.slice(i, j) });
      i = j;
      continue;
    }

    const kind = lineKind(lines[i]);

    if (kind === "heading") {
      const m = lines[i].match(HEADING_RE);
      blocks.push({ type: "heading", level: m[1].length, content: m[2] });
      i++;
      continue;
    }

    if (kind === "blockquote") {
      let j = i;
      const quoteLines = [];
      while (j < lines.length && lineKind(lines[j]) === "blockquote") {
        quoteLines.push(lines[j].match(BLOCKQUOTE_RE)[1]);
        j++;
      }
      blocks.push({ type: "blockquote", content: quoteLines.join("\n") });
      i = j;
      continue;
    }

    if (kind === "ordered" || kind === "unordered") {
      let j = i;
      const items = [];
      let start = null;
      while (j < lines.length) {
        const lk = lineKind(lines[j]);
        if (lk === kind) {
          if (kind === "ordered") {
            const m = lines[j].match(ORDERED_ITEM_RE);
            if (start === null) start = parseInt(m[2], 10);
            items.push({ indent: m[1].length, text: m[3] });
          } else {
            const m = lines[j].match(UNORDERED_ITEM_RE);
            items.push({ indent: m[1].length, text: m[2] });
          }
          j++;
        } else if (
          lines[j].trim() === "" &&
          lineKind(lines[j + 1] ?? "") === kind
        ) {
          // A blank line between two same-type items doesn't end the list (a "loose"
          // list) -- skip it without breaking the run.
          j++;
        } else {
          break;
        }
      }
      blocks.push({
        type: "list",
        ordered: kind === "ordered",
        items,
        start: start ?? 1,
      });
      i = j;
      continue;
    }

    let j = i;
    while (
      j < lines.length &&
      !isTableStart(j) &&
      lineKind(lines[j]) === "text"
    )
      j++;
    blocks.push({ type: "text", content: lines.slice(i, j).join("\n") });
    i = j;
  }
  return blocks;
}

function MarkdownTable({ lines, onCiteClick, citationCounts }) {
  const header = parseTableRow(lines[0]);
  const rows = lines.slice(2).map(parseTableRow);
  return (
    <div className="my-2 overflow-x-auto rounded-md border border-border">
      <table className="w-full text-xs border-collapse">
        <thead>
          <tr className="bg-muted/50">
            {header.map((cell, i) => (
              <th
                key={i}
                className="px-2 py-1.5 text-left font-medium border-b border-border"
              >
                {renderContentWithCitations(cell, onCiteClick, citationCounts)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, ri) => (
            <tr
              key={ri}
              className="border-b border-border last:border-0 even:bg-muted/30"
            >
              {row.map((cell, ci) => (
                <td key={ci} className="px-2 py-1.5 align-top">
                  {renderContentWithCitations(
                    cell,
                    onCiteClick,
                    citationCounts,
                  )}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function MarkdownHeading({ level, content, onCiteClick, citationCounts }) {
  const Tag = `h${Math.min(Math.max(level, 1), 6)}`;
  // Scaled one step up from the previous text-base/text-sm pairing to keep the same
  // relative step against the message body's own text-sm -> text-base readability bump
  // -- headings must stay visually larger than body text, not become the same size.
  const sizeClass =
    level <= 2 ? "text-lg font-semibold" : "text-base font-semibold";
  return (
    <Tag className={`${sizeClass} mt-3 mb-1 first:mt-0`}>
      {renderContentWithCitations(content, onCiteClick, citationCounts)}
    </Tag>
  );
}

function MarkdownBlockquote({ content, onCiteClick, citationCounts }) {
  return (
    <blockquote className="border-l-2 border-primary/40 pl-3 my-1.5 text-muted-foreground italic">
      {renderContentWithCitations(content, onCiteClick, citationCounts)}
    </blockquote>
  );
}

function MarkdownList({ ordered, items, start, onCiteClick, citationCounts }) {
  const Tag = ordered ? "ol" : "ul";
  const collapsedItems = collapseRepeatedItemCitations(items);
  const minIndent = Math.min(...items.map((it) => it.indent));
  return (
    <Tag
      className={`my-1 pl-5 space-y-0.5 ${ordered ? "list-decimal" : "list-disc"}`}
      {...(ordered && start !== 1 ? { start } : {})}
    >
      {collapsedItems.map((item, i) => {
        // A flat list with proportional indentation, not a real nested <ul> tree --
        // the model's own indentation (e.g. " + sub-item" under "* parent") is the
        // only depth signal available and isn't reliable enough to build a real list
        // tree from, but showing SOME indentation reads far better than none. ceil
        // (not floor) of half the diff so even a single extra space of indent (this
        // model's own convention, confirmed live) registers as level 1, not level 0.
        const diff = item.indent - minIndent;
        const level = diff > 0 ? Math.min(Math.ceil(diff / 2), 3) : 0;
        return (
          <li
            key={i}
            className={level > 0 && !ordered ? "list-[circle]" : undefined}
            style={level > 0 ? { marginLeft: `${level * 1.1}rem` } : undefined}
          >
            {renderContentWithCitations(item.text, onCiteClick, citationCounts)}
          </li>
        );
      })}
    </Tag>
  );
}

function renderMessageContent(content, onCiteClick) {
  // Shared across every block in the message so the repeat cap applies to the whole
  // answer, not just per-paragraph/per-table.
  const citationCounts = new Map();
  return splitContentBlocks(content).map((block, i) => {
    switch (block.type) {
      case "table":
        return (
          <MarkdownTable
            key={i}
            lines={block.lines}
            onCiteClick={onCiteClick}
            citationCounts={citationCounts}
          />
        );
      case "heading":
        return (
          <MarkdownHeading
            key={i}
            level={block.level}
            content={block.content}
            onCiteClick={onCiteClick}
            citationCounts={citationCounts}
          />
        );
      case "blockquote":
        return (
          <MarkdownBlockquote
            key={i}
            content={block.content}
            onCiteClick={onCiteClick}
            citationCounts={citationCounts}
          />
        );
      case "list":
        return (
          <MarkdownList
            key={i}
            ordered={block.ordered}
            items={block.items}
            start={block.start}
            onCiteClick={onCiteClick}
            citationCounts={citationCounts}
          />
        );
      default:
        return (
          <span key={i}>
            {renderContentWithCitations(
              block.content,
              onCiteClick,
              citationCounts,
            )}
          </span>
        );
    }
  });
}

function FeedbackButtons({
  question,
  answer,
  onRegenerate,
  regenerateDisabled,
}) {
  const [sent, setSent] = useState(null);
  const [copied, setCopied] = useState(false);

  const send = (helpful) => {
    const value = helpful ? "up" : "down";
    if (sent === value) return;
    setSent(value);
    fetch(`${API_URL}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: question ?? "", answer, helpful }),
    }).catch(() => {});
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(answer).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  };

  return (
    <div className="flex mt-3">
      <button
        onClick={handleCopy}
        aria-label="Copy"
        title="Copy all text"
        className="p-1.5 rounded-md transition hover:bg-background text-muted-foreground hover:text-primary"
      >
        {copied ? <Check size={14} /> : <Copy size={14} />}
      </button>
      <button
        onClick={() => send(true)}
        aria-label="Helpful"
        title="Give positive feedback"
        className={`p-1.5 rounded-md transition hover:bg-background ${sent === "up" ? "text-primary" : "text-muted-foreground hover:text-primary"}`}
      >
        <ThumbsUp size={14} />
      </button>
      <button
        onClick={() => send(false)}
        title="Give negative feedback"
        aria-label="Not helpful"
        className={`p-1.5 rounded-md transition hover:bg-background ${sent === "down" ? "text-primary" : "text-muted-foreground hover:text-primary"}`}
      >
        <ThumbsDown size={14} />
      </button>
      {onRegenerate && (
        <button
          onClick={onRegenerate}
          disabled={regenerateDisabled}
          aria-label="Regenerate"
          title="Regenerate response"
          className="p-1.5 rounded-md transition hover:bg-background text-muted-foreground hover:text-primary disabled:opacity-40 disabled:hover:bg-transparent disabled:hover:text-muted-foreground"
        >
          <RotateCcw size={14} />
        </button>
      )}
    </div>
  );
}

function TypingIndicator() {
  return (
    <span className="inline-flex items-center gap-1 py-1" aria-label="Thinking">
      <span className="h-1.5 w-1.5 rounded-full bg-current animate-shimmer-bounce [animation-delay:-0.3s]" />
      <span className="h-1.5 w-1.5 rounded-full bg-current animate-shimmer-bounce [animation-delay:-0.15s]" />
      <span className="h-1.5 w-1.5 rounded-full bg-current animate-shimmer-bounce" />
    </span>
  );
}

// Follow-up suggestions (IMPROVEMENTS.md #9.3), pinned above the composer as a single
// non-wrapping row instead of the flex-wrap layout used elsewhere for chip lists --
// wrapping would stack full-sentence questions into a tall block sitting right above
// where the user is about to type, which is the one place height is most at a premium.
// Overflow scrolls horizontally with the same edge-mask fade as the vertical message
// list (see scrollEdgeMask), plus a click-to-scroll chevron on whichever side has more
// to reveal -- the fade alone signals overflow but gives a mouse-only user (no
// trackpad/touch gesture, and the scrollbar itself is intentionally hidden) no actual
// way to act on it. snap-x/snap-start make wheel/trackpad/drag scrolling settle on a
// chip boundary too, so the chevrons and a raw scroll gesture always agree on where
// "the next chip" is.
function FollowUpBar({ questions, onSelect }) {
  const scrollRef = useRef(null);
  const chipRefs = useRef([]);
  const [showLeftFade, setShowLeftFade] = useState(false);
  const [showRightFade, setShowRightFade] = useState(false);

  const updateFades = () => {
    const el = scrollRef.current;
    if (!el) return;
    setShowLeftFade(el.scrollLeft > 0);
    setShowRightFade(el.scrollLeft + el.clientWidth < el.scrollWidth - 1);
  };

  useEffect(() => {
    updateFades();
  }, [questions]);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const observer = new ResizeObserver(updateFades);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  // Steps to the next chip whose start is past the currently-visible right edge --
  // i.e. the first chip the user can't fully see yet -- rather than an arbitrary pixel
  // distance, so one click always lands cleanly on a chip boundary regardless of how
  // long each question happens to be.
  const scrollToNextChip = () => {
    const container = scrollRef.current;
    if (!container) return;
    const visibleRight = container.scrollLeft + container.clientWidth;
    const next = chipRefs.current.find(
      (chip) => chip && chip.offsetLeft + chip.offsetWidth > visibleRight + 1
    );
    if (next) container.scrollTo({ left: next.offsetLeft, behavior: "smooth" });
  };

  // Mirrors scrollToNextChip going backwards: the last chip whose start is still
  // before the current scroll position, i.e. one chip back from wherever we are now.
  const scrollToPrevChip = () => {
    const container = scrollRef.current;
    if (!container) return;
    const chips = chipRefs.current.filter(Boolean);
    let target = chips[0];
    for (const chip of chips) {
      if (chip.offsetLeft < container.scrollLeft - 1) target = chip;
      else break;
    }
    if (target) container.scrollTo({ left: target.offsetLeft, behavior: "smooth" });
  };

  return (
    <div className="relative flex items-center">
      {showLeftFade && (
        <button
          onClick={scrollToPrevChip}
          aria-label="Scroll suggestions left"
          className="absolute left-0 z-10 flex items-center justify-center h-6 w-6 shrink-0 text-muted-foreground hover:text-primary transition"
        >
          <ChevronLeft size={14} />
        </button>
      )}
      <div
        ref={scrollRef}
        onScroll={updateFades}
        className="no-scrollbar snap-x snap-mandatory flex flex-nowrap items-center gap-2 overflow-x-auto"
        style={{
          maskImage: scrollEdgeMask(showLeftFade, showRightFade, "to right"),
          WebkitMaskImage: scrollEdgeMask(showLeftFade, showRightFade, "to right"),
        }}
      >
        {questions.map((q, i) => (
          <button
            key={q}
            ref={(el) => (chipRefs.current[i] = el)}
            onClick={() => onSelect(q)}
            className="snap-start inline-flex shrink-0 items-center gap-1.5 text-xs border border-primary/40 text-primary rounded-full pl-2 pr-3 py-1 hover:bg-primary hover:text-primary-foreground transition"
          >
            <Sparkles size={11} className="shrink-0" />
            {q}
          </button>
        ))}
      </div>
      {showRightFade && (
        <button
          onClick={scrollToNextChip}
          aria-label="Scroll suggestions right"
          className="absolute right-0 z-10 flex items-center justify-center h-6 w-6 shrink-0 text-muted-foreground hover:text-primary transition"
        >
          <ChevronRight size={14} />
        </button>
      )}
    </div>
  );
}

function MessageBubble({
  message,
  question,
  onCiteClick,
  isLast,
  isStreaming,
  onRegenerate,
  onSendSuggestion,
  starterQuestions,
}) {
  const isUser = message.role === "user";
  // Suggestions/redirect only make sense once this turn has actually finished (not
  // mid-stream) and only on the latest turn -- showing them on an older answer after
  // the conversation has moved on would be stale and cluttered.
  const showTurnActions = !isUser && isLast && message.content && !isStreaming;
  const isLoading = !isUser && !message.content;
  // Source lookup by citation id for the hover preview (SourcesContext). Empty until the
  // stream's `done` event delivers sources, so during streaming citation pills render
  // without a preview and gain one once sources arrive -- no extra work per token.
  const sourcesById = useMemo(() => {
    const map = {};
    for (const s of message.sources ?? []) map[String(s.id)] = s;
    return map;
  }, [message.sources]);

  return (
    <div
      className={`flex gap-2 animate-message-in ${isUser ? "justify-end" : "justify-start"} ${isLoading ? "items-center" : ""}`}
    >
      {!isUser && (
        <img
          src="/binus%20icon.png"
          alt=""
          className={`h-8 w-8 rounded-full shrink-0 object-contain ${isLoading ? "" : "self-start"}`}
        />
      )}
      <div
        className={`flex flex-col max-w-[85%] ${isUser ? "items-end" : "items-start"}`}
      >
        {isLoading ? (
          <TypingIndicator />
        ) : (
          <div
            className={`rounded-xl px-4 py-2 text-sm leading-relaxed whitespace-pre-wrap ${
              isUser
                ? "bg-primary text-primary-foreground"
                : "bg-muted text-foreground"
            }`}
          >
            <SourcesContext.Provider value={sourcesById}>
              {renderMessageContent(message.content, onCiteClick)}
            </SourcesContext.Provider>
            {!isUser && message.sources?.length > 0 && (
              <button
                onClick={() => onCiteClick(null)}
                className="block mt-2 pt-2 text-xs text-gray-500 hover:text-accent transition"
              >
                {message.sources.length} source
                {message.sources.length > 1 ? "s" : ""} — view
              </button>
            )}
          </div>
        )}
        {!isUser && message.content && (
          <FeedbackButtons
            question={question}
            answer={message.content}
            onRegenerate={isLast ? onRegenerate : undefined}
            regenerateDisabled={isStreaming}
          />
        )}
        {/* Follow-up suggestions (IMPROVEMENTS.md #9.3) now render in a sticky bar above
            the composer (see FollowUpBar in ChatPanel's return) instead of inline here --
            they guide what to ask next, which reads more naturally anchored next to where
            the next question actually gets typed, not scrolling away with old messages. */}
        {/* Human-handoff card: the escalation route on a fallback. Rendered from the
            structured `contacts` in the SSE 'done' event rather than concatenated into
            the message text (which is how it used to read -- a wall of name/email/
            WhatsApp lines inside the bubble), so it looks like a support bot's handoff
            rather than a paragraph. */}
        {!isUser && message.fallback && message.contacts?.length > 0 && (
          <div className="mt-2 w-full rounded-lg border border-border bg-card/60 p-3 space-y-2.5">
            {message.contacts.map((c) => (
              <div key={c.email || c.name}>
                <p className="text-xs font-medium text-foreground">{c.name}</p>
                {c.role && <p className="text-[11px] text-muted-foreground">{c.role}</p>}
                <div className="flex flex-wrap items-center gap-2 mt-1.5">
                  {c.email && (
                    <a
                      href={`mailto:${c.email}`}
                      className="inline-flex items-center gap-1 text-[11px] rounded-full border border-primary/40 text-primary px-2.5 py-1 hover:bg-primary hover:text-primary-foreground transition"
                    >
                      <Mail size={11} className="shrink-0" />
                      Email
                    </a>
                  )}
                  {c.whatsapp && (
                    <a
                      href={c.whatsapp}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1 text-[11px] rounded-full border border-primary/40 text-primary px-2.5 py-1 hover:bg-primary hover:text-primary-foreground transition"
                    >
                      <MessageCircle size={11} className="shrink-0" />
                      WhatsApp
                    </a>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
        {/* Fallback redirect (IMPROVEMENTS.md #9.4): a KB content gap doesn't have to be
            a dead end -- resurface the same starter questions shown on a fresh chat. */}
        {showTurnActions &&
          message.fallback &&
          starterQuestions?.length > 0 && (
            <div className="mt-3">
              <p className="text-xs text-muted-foreground mb-1.5">
                Or try one of these:
              </p>
              <div className="flex flex-wrap gap-2">
                {starterQuestions.map((q) => (
                  <button
                    key={q}
                    onClick={() => onSendSuggestion(q)}
                    className="inline-flex items-center gap-1.5 text-xs border border-primary text-primary rounded-full pl-2 pr-3 py-1 hover:bg-primary hover:text-primary-foreground transition"
                  >
                    <Sparkles size={11} className="shrink-0" />
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}
      </div>
    </div>
  );
}
