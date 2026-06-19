import { Check, ChevronDown, RefreshCw } from "lucide-react";
import { useState } from "react";
import type { ModelInfo } from "../api/client";

interface Props {
  activeLabel: string;
  models: ModelInfo[];
  activeId: string | null;
  connected: boolean;
  loading: boolean;
  onSelect: (id: string) => void;
  onRefresh: () => void;
}

const BADGE_STYLES: Record<string, string> = {
  Reasoning: "bg-violet-100 text-violet-700",
  Fast: "bg-emerald-100 text-emerald-700",
  Recommended: "bg-amber-100 text-amber-800",
};

export function ModelSelector({
  activeLabel,
  models,
  activeId,
  connected,
  loading,
  onSelect,
  onRefresh,
}: Props) {
  const [open, setOpen] = useState(false);

  return (
    <div className="relative">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-2 px-4 py-2 rounded-full bg-white/60 hover:bg-white border border-violet-200/50 text-sm font-medium text-sidebar transition-all hover:shadow-[0_4px_24px_rgba(124,58,237,0.08)]"
      >
        {loading ? "Loading models…" : activeLabel}
        <ChevronDown className="w-4 h-4 text-sidebar/50" />
      </button>
      {open && (
        <div className="absolute top-full left-0 mt-2 w-80 bg-white rounded-2xl shadow-[0_8px_32px_rgba(28,24,48,0.12)] border border-violet-100 py-2 z-30 max-h-80 overflow-y-auto">
          <div className="flex items-center justify-between px-4 py-2 border-b border-violet-50">
            <span className="text-xs text-sidebar/50">
              {connected ? "Local Ollama models" : "Ollama offline"}
            </span>
            <button onClick={onRefresh} className="p-1 rounded-lg hover:bg-violet-50" aria-label="Refresh">
              <RefreshCw className="w-3.5 h-3.5 text-sidebar/50" />
            </button>
          </div>
          {models.map((m) => (
            <button
              key={m.id}
              onClick={() => {
                onSelect(m.id);
                setOpen(false);
              }}
              className={`w-full text-left px-4 py-3 hover:bg-violet-50 transition-colors ${
                m.id === activeId ? "bg-violet-50/80" : ""
              }`}
            >
              <div className="flex items-start justify-between gap-2">
                <div className="min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium text-sidebar truncate">{m.label}</span>
                    {m.id === activeId && <Check className="w-4 h-4 text-primary shrink-0" />}
                  </div>
                  <p className="text-xs text-sidebar/45 mt-0.5 truncate">
                    {[m.parameter_size, m.quantization, m.family].filter(Boolean).join(" · ") || m.id}
                  </p>
                  {m.badges && m.badges.length > 0 && (
                    <div className="flex flex-wrap gap-1 mt-1.5">
                      {m.badges.map((b) => (
                        <span
                          key={b}
                          className={`text-[10px] px-1.5 py-0.5 rounded-full font-medium ${
                            BADGE_STYLES[b] ?? "bg-gray-100 text-gray-600"
                          }`}
                        >
                          {b}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}