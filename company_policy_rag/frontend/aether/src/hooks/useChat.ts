import { useCallback, useState } from "react";
import { ChatMessage, CorpusScope, GroundingMode, sendChat } from "../api/client";

export function useChat(
  scope: CorpusScope,
  model: string | null,
  grounding: GroundingMode,
) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const send = useCallback(
    async (text: string) => {
      const msg = text.trim();
      if (!msg || loading) return;
      setError(null);
      setMessages((m) => [...m, { role: "user", content: msg }]);
      setLoading(true);
      try {
        const res = await sendChat(msg, {
          corpus_scope: scope,
          llm_model: model,
          grounding_mode: grounding,
        });
        setMessages((m) => [
          ...m,
          { role: "assistant", content: res.answer, citations: res.citations },
        ]);
      } catch (e) {
        const errMsg = e instanceof Error ? e.message : "Request failed";
        setError(errMsg);
        setMessages((m) => [
          ...m,
          { role: "assistant", content: `Error: ${errMsg}` },
        ]);
      } finally {
        setLoading(false);
      }
    },
    [scope, model, grounding, loading],
  );

  const clear = useCallback(() => {
    setMessages([]);
    setError(null);
  }, []);

  return { messages, loading, error, send, clear };
}