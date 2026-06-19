import { Brain, ChevronDown } from "lucide-react";
import { useState } from "react";

interface Props {
  thinking: string;
}

export function ThinkingBlock({ thinking }: Props) {
  const [open, setOpen] = useState(false);
  if (!thinking.trim()) return null;

  return (
    <div className="mb-2 rounded-2xl border border-violet-200/60 bg-violet-50/80 overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-2 px-4 py-2.5 text-left text-sm text-violet-800 hover:bg-violet-100/60 transition-colors"
      >
        <Brain className="w-4 h-4 shrink-0" />
        <span className="font-medium">Reasoning</span>
        <ChevronDown className={`w-4 h-4 ml-auto transition-transform ${open ? "rotate-180" : ""}`} />
      </button>
      {open && (
        <div className="px-4 pb-3 text-xs text-violet-900/80 whitespace-pre-wrap leading-relaxed max-h-48 overflow-y-auto border-t border-violet-200/40">
          {thinking}
        </div>
      )}
    </div>
  );
}