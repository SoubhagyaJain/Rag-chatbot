import { X } from "lucide-react";
import type { CorpusScope, GroundingMode } from "../api/client";

interface Props {
  open: boolean;
  onClose: () => void;
  scope: CorpusScope;
  onScope: (s: CorpusScope) => void;
  grounding: GroundingMode;
  onGrounding: (g: GroundingMode) => void;
  health?: string;
}

export function ConfigPanel({
  open,
  onClose,
  scope,
  onScope,
  grounding,
  onGrounding,
  health,
}: Props) {
  if (!open) return null;

  return (
    <>
      <div className="fixed inset-0 bg-black/20 z-40" onClick={onClose} />
      <aside className="fixed top-0 right-0 h-full w-80 bg-sidebar text-white z-50 shadow-[0_8px_32px_rgba(28,24,48,0.12)] p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-semibold">Configuration</h2>
          <button onClick={onClose} className="p-2 rounded-xl hover:bg-white/10">
            <X className="w-5 h-5" />
          </button>
        </div>
        <label className="block text-sm text-white/60 mb-2">Corpus scope</label>
        <select
          value={scope}
          onChange={(e) => onScope(e.target.value as CorpusScope)}
          className="w-full bg-input rounded-xl px-3 py-2 mb-4 text-sm border border-white/10 outline-none focus:ring-2 focus:ring-primary"
        >
          <option value="all">All corpora</option>
          <option value="policy">Company Policy</option>
          <option value="guidebook">AI Agents Guidebook</option>
        </select>
        <label className="block text-sm text-white/60 mb-2">Grounding mode</label>
        <select
          value={grounding}
          onChange={(e) => onGrounding(e.target.value as GroundingMode)}
          className="w-full bg-input rounded-xl px-3 py-2 mb-4 text-sm border border-white/10 outline-none focus:ring-2 focus:ring-primary"
        >
          <option value="balanced">Balanced</option>
          <option value="strict">Strict</option>
        </select>
        {health && (
          <div className="text-xs text-white/50 whitespace-pre-line mt-4">{health}</div>
        )}
      </aside>
    </>
  );
}