import type { ChatMessage } from "../api/client";

interface Props {
  messages: ChatMessage[];
  loading: boolean;
}

export function ChatThread({ messages, loading }: Props) {
  if (!messages.length && !loading) return null;

  return (
    <div className="max-w-3xl mx-auto w-full space-y-4 pt-6 pb-4">
      {messages.map((m, i) =>
        m.role === "user" ? (
          <div key={i} className="flex justify-end">
            <div className="max-w-[85%] px-4 py-3 rounded-2xl rounded-br-md bg-violet-200/80 text-sidebar text-sm">
              {m.content}
            </div>
          </div>
        ) : (
          <div key={i} className="flex justify-start">
            <div className="max-w-[90%] px-5 py-4 rounded-2xl rounded-bl-md bg-sidebar text-white/90 text-sm shadow-[0_8px_32px_rgba(28,24,48,0.12)] border border-white/5">
              <div className="whitespace-pre-wrap leading-relaxed">{m.content}</div>
              {m.citations && m.citations.length > 0 && (
                <div className="mt-3 pt-2 border-t border-white/10 flex flex-wrap gap-1">
                  {m.citations.slice(0, 3).map((c, j) => (
                    <span
                      key={j}
                      className="inline-block px-2 py-0.5 rounded-lg bg-white/10 text-white/60 text-xs"
                    >
                      {String(c.section_path || c.source_file || "Source")}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        ),
      )}
      {loading && (
        <div className="flex gap-2 items-center text-sidebar/50 text-sm py-2">
          <span className="w-2 h-2 rounded-full bg-primary animate-bounce" />
          <span className="w-2 h-2 rounded-full bg-primary animate-bounce [animation-delay:0.15s]" />
          <span className="w-2 h-2 rounded-full bg-primary animate-bounce [animation-delay:0.3s]" />
          Retrieving and synthesizing…
        </div>
      )}
    </div>
  );
}