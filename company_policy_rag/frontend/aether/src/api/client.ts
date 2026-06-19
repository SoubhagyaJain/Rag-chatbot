export type CorpusScope = "all" | "policy" | "guidebook";
export type GroundingMode = "balanced" | "strict";

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  citations?: Array<Record<string, unknown>>;
}

export interface ModelInfo {
  id: string;
  label: string;
}

export interface ModelsResponse {
  connected: boolean;
  error?: string;
  active_model: string;
  models: ModelInfo[];
}

export interface HealthResponse {
  index_ready: boolean;
  chunk_count: number;
  ollama_connected: boolean;
  llm_model: string;
  embed_model: string;
}

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, {
    headers: { "Content-Type": "application/json", ...init?.headers },
    ...init,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail || res.statusText);
  }
  return res.json() as Promise<T>;
}

export function fetchModels() {
  return api<ModelsResponse>("/models");
}

export function setActiveModel(model: string) {
  return api<{ active_model: string }>("/models/active", {
    method: "PUT",
    body: JSON.stringify({ model }),
  });
}

export function fetchHealth() {
  return api<HealthResponse>("/health");
}

export function sendChat(
  message: string,
  opts: {
    corpus_scope?: CorpusScope;
    llm_model?: string | null;
    grounding_mode?: GroundingMode;
  } = {},
) {
  return api<{
    answer: string;
    citations: Array<Record<string, unknown>>;
    timing?: Record<string, number>;
    low_confidence: boolean;
  }>("/chat", {
    method: "POST",
    body: JSON.stringify({
      message,
      corpus_scope: opts.corpus_scope ?? "all",
      chat_mode: "direct",
      llm_model: opts.llm_model,
      grounding_mode: opts.grounding_mode,
    }),
  });
}

export function runEval(max_samples = 5) {
  return api<{ job_id: string; status: string }>("/eval/run", {
    method: "POST",
    body: JSON.stringify({ max_samples }),
  });
}