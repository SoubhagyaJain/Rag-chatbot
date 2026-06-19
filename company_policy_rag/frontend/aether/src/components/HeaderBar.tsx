import { Download, Settings } from "lucide-react";
import type { ModelInfo } from "../api/client";
import { ModelSelector } from "./ModelSelector";

interface Props {
  activeLabel: string;
  models: ModelInfo[];
  activeId: string | null;
  connected: boolean;
  modelsLoading: boolean;
  onSelectModel: (id: string) => void;
  onRefreshModels: () => void;
  onConfig: () => void;
  onExport: () => void;
}

export function HeaderBar({
  activeLabel,
  models,
  activeId,
  connected,
  modelsLoading,
  onSelectModel,
  onRefreshModels,
  onConfig,
  onExport,
}: Props) {
  return (
    <header className="flex items-center justify-between px-8 py-4 border-b border-violet-200/40 bg-canvas/80 backdrop-blur-sm">
      <ModelSelector
        activeLabel={activeLabel}
        models={models}
        activeId={activeId}
        connected={connected}
        loading={modelsLoading}
        onSelect={onSelectModel}
        onRefresh={onRefreshModels}
      />
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