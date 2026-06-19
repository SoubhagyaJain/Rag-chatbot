import { ArrowRight, Paperclip, SlidersHorizontal, Sliders } from "lucide-react";
import { KeyboardEvent, useState } from "react";

interface Props {
  onSend: (text: string) => void;
  loading: boolean;
  onOpenConfig: () => void;
}

export function ChatInput({ onSend, loading, onOpenConfig }: Props) {
  const [text, setText] = useState("");

  const submit = () => {
    if (!text.trim() || loading) return;
    onSend(text);
    setText("");
  };

  const onKey = (e: KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <div className="px-8 pb-6 pt-2">
      <div className="max-w-3xl mx-auto">
        <div className="bg-input rounded-3xl shadow-[0_8px_32px_rgba(28,24,48,0.12)] border border-white/5 flex items-end gap-2 p-3 pl-4 focus-within:ring-2 focus-within:ring-primary/40 transition-all">
          <div className="flex items-center gap-1 pb-2">
            <button className="p-2 rounded-xl text-white/40 hover:text-white/80 hover:bg-white/5" aria-label="Attach">
              <Paperclip className="w-5 h-5" />
            </button>
            <button
              onClick={onOpenConfig}
              className="p-2 rounded-xl text-white/40 hover:text-white/80 hover:bg-white/5"
              aria-label="Context"
            >
              <SlidersHorizontal className="w-5 h-5" />
            </button>
            <button className="p-2 rounded-xl text-white/40 hover:text-white/80 hover:bg-white/5" aria-label="Advanced">
              <Sliders className="w-5 h-5" />
            </button>
          </div>
          <textarea
            rows={1}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={onKey}
            placeholder="Ask anything about policies or the guidebook…"
            className="flex-1 bg-transparent text-white placeholder-white/40 text-sm resize-none outline-none py-3 max-h-32"
          />
          <button
            onClick={submit}
            disabled={loading}
            className="shrink-0 w-10 h-10 rounded-full bg-primary hover:bg-highlight flex items-center justify-center text-white transition-all hover:shadow-[0_0_40px_rgba(168,85,247,0.35)] disabled:opacity-40"
            aria-label="Send"
          >
            <ArrowRight className="w-5 h-5" />
          </button>
        </div>
        <p className="text-center text-xs text-sidebar/40 mt-2">
          Aether retrieves from indexed documents — answers include source citations.
        </p>
      </div>
    </div>
  );
}