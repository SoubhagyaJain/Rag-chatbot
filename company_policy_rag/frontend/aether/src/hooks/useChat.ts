import { useCallback, useRef, useState } from "react";
import {
  ChatMessage,
  CorpusScope,
  GroundingMode,
  sendChatStream,
} from "../api/client";

export function useChat(
  scope: CorpusScope,
  model: string | null,
  grounding: GroundingMode,
) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const updateAssistant = useCallback(
    (patch: Partial<ChatMessage>) => {
      setMessages((msgs) => {
        const next = [...msgs];
        const idx = next.length - 1;
        if (idx >= 0 && next[idx].role === "assistant") {
          next[idx] = { ...next[idx], ...patch };
        }
        return next;
      });
    },
    [],
  );

  const send = useCallback(
    async (text: string) => {
      const msg = text.trim();
      if (!msg || loading) return;
      setError(null);
      setMessages((m) => [
        ...m,
        { role: "user", content: msg },
        { role: "assistant", content: "", isStreaming: true },
      ]);
      setLoading(true);

      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      let streamedContent = "";

      try {
        await sendChatStream(
          msg,
          {
            corpus_scope: scope,
            llm_model: model,
            grounding_mode: grounding,
          },
          {
            onRetrievalDone: (trace) => updateAssistant({ retrieval_trace: trace }),
            onThinking: (thinking) => updateAssistant({ thinking }),
            onToken: (token) => {
              streamedContent += token;
              updateAssistant({ content: streamedContent });
            },
            onDone: (payload) => {
              updateAssistant({
                content: payload.answer,
                citations: payload.citations,
                thinking: payload.thinking,
                retrieval_trace: payload.retrieval_trace,
                timing: payload.timing,
                message_id: payload.message_id,
                low_confidence: payload.low_confidence,
                isStreaming: false,
              });
            },
            onError: (errMsg) => {
              setError(errMsg);
              updateAssistant({
                content: `Error: ${errMsg}`,
                isStreaming: false,
              });
            },
          },
          controller.signal,
        );
      } catch (e) {
        if ((e as Error).name === "AbortError") return;
        const errMsg = e instanceof Error ? e.message : "Request failed";
        setError(errMsg);
        updateAssistant({
          content: `Error: ${errMsg}`,
          isStreaming: false,
        });
      } finally {
        setLoading(false);
        abortRef.current = null;
      }
    },
    [scope, model, grounding, loading, updateAssistant],
  );

  const clear = useCallback(() => {
    abortRef.current?.abort();
    setMessages([]);
    setError(null);
    setLoading(false);
  }, []);

  return { messages, loading, error, send, clear };
}