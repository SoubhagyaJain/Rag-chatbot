import { AlertTriangle, CheckCircle2, X } from "lucide-react";

export interface Citation {
  section_path?: string;
  source_file?: string;
  page_number?: number;
  excerpt?: string;
  score?: number;
  selection_reason?: string;
}

interface Props {
  open: boolean;
  citations: Citation[];
  highlightIndex?: number;
  onClose: () => void;
}

export function CitationPanel({ open, citations, highlightIndex, onClose }: Props) {
  if (!open) return null;

  return (
    <>
      <div className="fixed inset-0 bg-black/20 z-40" onClick={onClose} />
      <aside className="fixed top-0 right-0 h-full w-96 bg-sidebar text-white z-50 shadow-[0_8px_32px_rgba(28,24,48,0.12)] flex flex-col">
        <div className="flex items-center justify-between p-5 border-b border-white/10">
          <h2 className="text-lg font-semibold">Sources ({citations.length})</h2>
          <button onClick={onClose} className="p-2 rounded-xl hover:bg-white/10">
            <X className="w-5 h-5" />
          </button>
        </div>
        <div className="flex-1 overflow-y-auto p-4 space-y-3">
          {citations.map((c, i) => {
            const cited = c.selection_reason === "cited_in_answer";
            const highlighted = highlightIndex === i + 1;
            return (
              <div
                key={i}
                className={`rounded-2xl p-4 border transition-all ${
                  highlighted
                    ? "border-primary bg-primary/10"
                    : "border-white/10 bg-white/5"
                }`}
              >
                <div className="flex items-start gap-2 mb-2">
                  {cited ? (
                    <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                  ) : (
                    <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
                  )}
                  <div className="min-w-0">
                    <p className="text-sm font-medium leading-snug">
                      {c.section_path || c.source_file || `Source ${i + 1}`}
                    </p>
                    <p className="text-xs text-white/50 mt-1">
                      {[c.source_file, c.page_number != null ? `p.${c.page_number}` : null]
                        .filter(Boolean)
                        .join(" · ")}
                    </p>
                  </div>
                </div>
                {c.excerpt && (
                  <p className="text-xs text-white/70 leading-relaxed line-clamp-6">{c.excerpt}</p>
                )}
                {c.score != null && (
                  <p className="text-[10px] text-white/40 mt-2">Relevance {c.score.toFixed(3)}</p>
                )}
              </div>
            );
          })}
        </div>
      </aside>
    </>
  );
}