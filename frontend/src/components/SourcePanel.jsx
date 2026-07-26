import { ExternalLink, FileSearch, X } from 'lucide-react'
import { sourceUrl } from '@/lib/api'

export default function SourcePanel({ sources, highlightId, onToggle, onClose, onCollapse }) {
  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold">Sources</h2>
        {onCollapse && (
          <button
            onClick={onCollapse}
            className="text-muted-foreground hover:text-foreground transition"
            aria-label="Collapse sources"
          >
            <span className="material-symbols-outlined" style={{ fontSize: 20 }}>right_panel_close</span>
          </button>
        )}
        {onClose && (
          <button onClick={onClose} className="text-muted-foreground hover:text-foreground" aria-label="Close">
            <X size={16} />
          </button>
        )}
      </div>
      <div className="-mx-4 my-2 border-b border-border" />
      <div className="flex-1 overflow-y-auto py-3 flex flex-col gap-2">
        {sources.length === 0 && (
          <div className="flex-1 flex flex-col items-center justify-center gap-3 text-center px-4">
            <div className="flex items-center justify-center h-10 w-10 rounded-xl bg-gradient-to-br from-primary/15 via-accent/10 to-primary/5">
              <FileSearch size={18} className="text-muted-foreground" />
            </div>
            <div className="flex flex-col gap-1">
              <p className="text-sm font-medium text-foreground">No sources yet</p>
              <p className="text-xs text-muted-foreground max-w-[200px]">
                Sources cited in the next answer will show up here.
              </p>
            </div>
          </div>
        )}
        {sources.map((source) => {
          const isOpen = highlightId === source.id
          // Relevance bar, not a literal percentage: the confidence gate only checks the
          // TOP source in a returned list (backend/chat_service.py), so #2+ can carry a
          // much lower -- even negative, per the underlying cross-encoder's raw logit --
          // score. A number would misleadingly imply calibrated precision; a clamped,
          // relative fill communicates "how this one compares" without claiming that.
          const relevance =
            source.score != null ? Math.round(Math.min(Math.max(source.score, 0), 1) * 100) : null
          return (
            <div key={source.id} className="border border-border rounded-lg overflow-hidden bg-card shadow-sm">
              <button
                onClick={() => onToggle?.(source.id)}
                className="w-full flex items-start gap-2 px-3 py-2 text-left hover:bg-muted/40 transition"
              >
                <span className="flex items-center justify-center h-5 min-w-5 px-1 mt-0.5 shrink-0 rounded-sm bg-accent/15 text-accent text-[10px] font-medium">
                  {source.id}
                </span>
                <span className="flex-1 min-w-0">
                  {/* The title is the one thing this card must communicate at a glance --
                      everything else (section, page/sheet, relevance) is deliberately a
                      size and a color step down, on its own line below it. */}
                  <span className="block truncate text-sm font-medium text-foreground">
                    {source.display_name || source.source_file}
                  </span>
                  {source.section_title && (
                    <span className="block truncate text-xs text-muted-foreground mt-0.5">
                      {source.section_title}
                    </span>
                  )}
                  {(source.page_number != null || source.sheet_name != null || relevance != null) && (
                    <span className="flex items-center gap-2 mt-1 text-[10px] text-muted-foreground/70">
                      {source.page_number != null && <span>p.{source.page_number}</span>}
                      {source.sheet_name != null && <span>{source.sheet_name}</span>}
                      {relevance != null && (
                        <span
                          className="inline-flex items-center h-1 w-8 rounded-full bg-muted overflow-hidden"
                          title="Relevance to this answer"
                        >
                          <span
                            className="block h-full bg-gradient-to-r from-accent/50 to-accent rounded-full"
                            style={{ width: `${relevance}%` }}
                          />
                        </span>
                      )}
                    </span>
                  )}
                </span>
              </button>
              <div
                className={`grid transition-all duration-300 ease-in-out ${
                  isOpen ? 'grid-rows-[1fr] opacity-100' : 'grid-rows-[0fr] opacity-0'
                }`}
              >
                <div className="overflow-hidden">
                  <div className="px-3 pb-3">
                    <p className="text-xs text-muted-foreground whitespace-pre-wrap">
                      {source.snippet}
                      {source.snippet?.length >= 200 ? '…' : ''}
                    </p>
                    <a
                      href={sourceUrl(source)}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1 mt-2 text-xs text-primary hover:underline"
                    >
                      <ExternalLink size={12} /> Open source
                    </a>
                  </div>
                </div>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
