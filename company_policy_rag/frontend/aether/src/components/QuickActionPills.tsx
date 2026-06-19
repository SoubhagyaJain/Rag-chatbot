import { BarChart3, HelpCircle, Search } from "lucide-react";

interface Props {
  onPolicy: () => void;
  onGuidebook: () => void;
  onEval: () => void;
}

export function QuickActionPills({ onPolicy, onGuidebook, onEval }: Props) {
  const pills = [
    { label: "Ask about Policy", icon: HelpCircle, onClick: onPolicy },
    { label: "Explore Guidebook", icon: Search, onClick: onGuidebook },
    { label: "Run Evaluation", icon: BarChart3, onClick: onEval },
  ];

  return (
    <div className="flex flex-wrap gap-3 justify-center mb-6 px-8">
      {pills.map((p) => (
        <button
          key={p.label}
          onClick={p.onClick}
          className="px-5 py-2.5 rounded-full bg-white/70 hover:bg-white border border-violet-200/60 text-sm font-medium text-sidebar shadow-[0_4px_24px_rgba(124,58,237,0.08)] hover:shadow-[0_8px_32px_rgba(28,24,48,0.12)] hover:ring-2 hover:ring-primary/20 transition-all flex items-center gap-2"
        >
          <p.icon className="w-4 h-4 text-primary" />
          {p.label}
        </button>
      ))}
    </div>
  );
}