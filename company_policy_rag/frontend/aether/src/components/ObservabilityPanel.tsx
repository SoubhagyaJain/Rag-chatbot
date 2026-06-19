import { Activity } from "lucide-react";

export interface RetrievalTrace {
  chunk_count?: number;
  chunks?: Array<{
    index: number;
    section_path?: string;
    page_number?: number;
    score?: number;
    excerpt_preview?: string;
  }>;
  stages?: Record<string, number>;
}

interface Props {
  trace?: RetrievalTrace | null;
  timing?: Record<string, number> | null;
}

const STAGE_LABELS: Array<[string, string]> = [
  ["query_rewrite_ms", "Rewrite"],
  ["chroma_retrieve_ms", "Retrieve"],
  ["rerank_filter_ms", "Rerank"],
  ["generation_ms", "Generate"],
  ["faithfulness_guard_ms", "Guard"],
];

export function ObservabilityPanel({ trace, timing }: Props) {
  const stages = trace?.stages ?? timing ?? {};
  const hasStages = Object.values(stages).some((v) => v > 0);
  if (!hasStages && !trace?.chunks?.length) return null;

  return (
    <details className="mt-3 rounded-2xl border border-white/10 bg-white/5 text-white/80">
      <summary className="flex items-center gap-2 px-4 py-2.5 cursor-pointer text-xs font-medium list-none">
        <Activity className="w-3.5 h-3.5 text-highlight" />
        Pipeline · {trace?.chunk_count ?? 0} chunks
        {stages.e2e_ms ? ` · ${Math.round(stages.e2e_ms)}ms` : ""}
      </summary>
      <div className="px-4 pb-3 space-y-3">
        {hasStages && (
          <div className="flex flex-wrap gap-2">
            {STAGE_LABELS.map(([key, label]) => {
              const ms = stages[key];
              if (!ms) return null;
              return (
                <span
                  key={key}
                  className="text-[10px] px-2 py-1 rounded-full bg-white/10 text-white/70"
                >
                  {label} {Math.round(ms)}ms
                </span>
              );
            })}
          </div>
        )}
        {trace?.chunks && trace.chunks.length > 0 && (
          <div className="space-y-1.5 max-h-36 overflow-y-auto">
            {trace.chunks.map((c) => (
              <div key={c.index} className="text-[10px] text-white/60 leading-snug">
                <span className="text-white/80">
                  #{c.index} {c.section_path ?? "?"}
                  {c.page_number != null ? ` p.${c.page_number}` : ""}
                </span>
                {c.score != null && <span className="text-white/40"> · {c.score.toFixed(3)}</span>}
                {c.excerpt_preview && (
                  <p className="text-white/45 truncate">{c.excerpt_preview}</p>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </details>
  );
}