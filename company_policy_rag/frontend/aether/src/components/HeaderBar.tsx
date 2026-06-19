import { ChevronDown, Download, Settings } from "lucide-react";
import { useState } from "react";
import type { ModelInfo } from "../api/client";

interface Props {
  activeLabel: string;
  models: ModelInfo[];
  activeId: string | null;
  onSelectModel: (id: string) => void;
  onConfig: () => void;
  onExport: () => void;
}

export function HeaderBar({
  activeLabel,
  models,
  activeId,
  onSelectModel,
  onConfig,
  onExport,
}: Props) {
  const [open, setOpen] = useState(false);

  return (
    <header className="flex items-center justify-between px-8 py-4 border-b border-violet-200/40 bg-canvas/80 backdrop-blur-sm">
      <div className="relative">
        <button
          onClick={() => setOpen(!open)}
          className="flex items-center gap-2 px-4 py-2 rounded-full bg-white/60 hover:bg-white border border-violet-200/50 text-sm font-medium text-sidebar transition-all hover:shadow-[0_4px_24px_rgba(124,58,237,0.08)]"
        >
          {activeLabel}
          <ChevronDown className="w-4 h-4 text-sidebar/50" />
        </button>
        {open && (
          <div className="absolute top-full left-0 mt-2 w-56 bg-white rounded-2xl shadow-[0_8px_32px_rgba(28,24,48,0.12)] border border-violet-100 py-2 z-30">
            {models.map((m) => (
              <button
                key={m.id}
                onClick={() => {
                  onSelectModel(m.id);
                  setOpen(false);
                }}
                className={`w-full text-left px-4 py-2 text-sm hover:bg-violet-50 ${
                  m.id === activeId ? "text-primary font-medium" : "text-sidebar"
                }`}
              >
                {m.label}
              </button>
            ))}
          </div>
        )}
      </div>
      <div className="flex items-center gap-2">
        <button
          onClick={onConfig}
          className="flex items-center gap-1.5 px-4 py-2 rounded-full bg-white/60 hover:bg-white border border-violet-200/50 text-sm text-sidebar/80 hover:text-sidebar transition-all"
        >
          <Settings className="w-4 h-4" /> Configuration
        </button>
        <button
          onClick={onExport}
          className="flex items-center gap-1.5 px-4 py-2 rounded-full bg-white/60 hover:bg-white border border-violet-200/50 text-sm text-sidebar/80 hover:text-sidebar transition-all"
        >
          <Download className="w-4 h-4" /> Export
        </button>
      </div>
    </header>
  );
}