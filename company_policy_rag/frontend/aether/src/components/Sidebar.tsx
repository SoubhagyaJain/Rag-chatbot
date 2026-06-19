import { BookOpen, FileText, Layers, Plus } from "lucide-react";
import type { CorpusScope } from "../api/client";

interface Props {
  scope: CorpusScope;
  onScope: (s: CorpusScope) => void;
  onNewChat: () => void;
}

const CHATS = [
  { title: "Dress code policy", scope: "policy" as const },
  { title: "Sick leave eligibility", scope: "policy" as const },
  { title: "AI agent building blocks", scope: "guidebook" as const },
];

export function Sidebar({ scope, onScope, onNewChat }: Props) {
  return (
    <aside className="w-[252px] shrink-0 bg-sidebar text-white flex flex-col shadow-[0_8px_32px_rgba(28,24,48,0.12)]">
      <div className="p-5 flex items-center gap-2.5">
        <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-primary to-highlight flex items-center justify-center shadow-[0_0_40px_rgba(168,85,247,0.35)]">
          <Layers className="w-4 h-4" />
        </div>
        <span className="text-lg font-semibold tracking-tight">Aether</span>
      </div>

      <div className="px-4 mb-4">
        <button
          onClick={onNewChat}
          className="w-full py-2.5 px-4 rounded-2xl bg-primary hover:bg-highlight text-white text-sm font-medium transition-all hover:shadow-[0_0_40px_rgba(168,85,247,0.35)] flex items-center justify-center gap-2"
        >
          <Plus className="w-4 h-4" /> New Chat
        </button>
      </div>

      <nav className="flex-1 overflow-y-auto px-3 space-y-6">
        <div>
          <p className="px-2 text-xs font-medium text-white/40 uppercase tracking-wider mb-2">
            Chats
          </p>
          {CHATS.map((c) => (
            <button
              key={c.title}
              onClick={() => onScope(c.scope)}
              className="w-full text-left px-3 py-2 rounded-xl text-sm text-white/60 hover:bg-white/5 hover:text-white transition-all truncate"
            >
              {c.title}
            </button>
          ))}
        </div>
        <div>
          <p className="px-2 text-xs font-medium text-white/40 uppercase tracking-wider mb-2">
            Knowledge Bases
          </p>
          <button
            onClick={() => onScope("policy")}
            className={`w-full text-left px-3 py-2 rounded-xl text-sm flex items-center gap-2 transition-all ${
              scope === "policy" ? "bg-white/10 text-white" : "text-white/70 hover:bg-white/5"
            }`}
          >
            <FileText className="w-4 h-4 text-primary" /> Company Policy
          </button>
          <button
            onClick={() => onScope("guidebook")}
            className={`w-full text-left px-3 py-2 rounded-xl text-sm flex items-center gap-2 mt-0.5 transition-all ${
              scope === "guidebook" ? "bg-white/10 text-white" : "text-white/70 hover:bg-white/5"
            }`}
          >
            <BookOpen className="w-4 h-4 text-highlight" /> AI Agents Guidebook
          </button>
        </div>
      </nav>

      <div className="p-4 m-3 rounded-2xl bg-white/5 border border-white/10">
        <p className="text-xs font-medium text-white/80 mb-1">Upgrade to Premium</p>
        <p className="text-xs text-white/50 mb-3">Advanced eval suites & team workspaces</p>
        <button className="w-full py-1.5 rounded-xl bg-primary/80 hover:bg-primary text-xs font-medium">
          Upgrade
        </button>
      </div>
    </aside>
  );
}