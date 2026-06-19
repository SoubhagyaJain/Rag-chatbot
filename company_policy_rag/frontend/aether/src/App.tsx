import { useCallback, useEffect, useState } from "react";
import {
  CorpusScope,
  fetchHealth,
  GroundingMode,
  runEval,
} from "./api/client";
import { ChatInput } from "./components/ChatInput";
import { ChatThread } from "./components/ChatThread";
import { ConfigPanel } from "./components/ConfigPanel";
import { FeatureCards } from "./components/FeatureCards";
import { HeaderBar } from "./components/HeaderBar";
import { HeroOrb } from "./components/HeroOrb";
import { QuickActionPills } from "./components/QuickActionPills";
import { Sidebar } from "./components/Sidebar";
import { useChat } from "./hooks/useChat";
import { useModels } from "./hooks/useModels";

const PROMPTS = {
  policy: "What is the dress code policy?",
  guidebook: "What are the six building blocks of AI agents?",
};

export default function App() {
  const [scope, setScope] = useState<CorpusScope>("all");
  const [grounding, setGrounding] = useState<GroundingMode>("balanced");
  const [configOpen, setConfigOpen] = useState(false);
  const [healthText, setHealthText] = useState("");
  const [toast, setToast] = useState("");

  const { models, active, activeLabel, select } = useModels();
  const { messages, loading, send, clear } = useChat(scope, active, grounding);

  const showWelcome = messages.length === 0 && !loading;

  useEffect(() => {
    fetchHealth()
      .then((h) =>
        setHealthText(
          `Index: ${h.index_ready ? "Ready" : "Not ready"} (${h.chunk_count} chunks)\nOllama: ${h.ollama_connected ? "Connected" : "Offline"}\nLLM: ${h.llm_model}`,
        ),
      )
      .catch(() => {});
  }, []);

  const notify = (msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(""), 4000);
  };

  const quickPolicy = () => {
    setScope("policy");
    send(PROMPTS.policy);
  };

  const quickGuidebook = () => {
    setScope("guidebook");
    send(PROMPTS.guidebook);
  };

  const handleEval = useCallback(async () => {
    notify("Starting evaluation…");
    try {
      const job = await runEval(5);
      notify(`Eval job ${job.job_id} started`);
    } catch (e) {
      notify(e instanceof Error ? e.message : "Eval failed");
    }
  }, []);

  const exportChat = () => {
    const blob = new Blob(
      [JSON.stringify({ messages, scope, model: active }, null, 2)],
      { type: "application/json" },
    );
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `aether-chat-${Date.now()}.json`;
    a.click();
    notify("Chat exported");
  };

  return (
    <div className="bg-canvas text-sidebar antialiased min-h-screen overflow-hidden flex">
      {toast && (
        <div className="fixed top-6 right-6 z-50 px-5 py-3 rounded-2xl bg-sidebar text-white text-sm shadow-[0_8px_32px_rgba(28,24,48,0.12)]">
          {toast}
        </div>
      )}

      <ConfigPanel
        open={configOpen}
        onClose={() => setConfigOpen(false)}
        scope={scope}
        onScope={setScope}
        grounding={grounding}
        onGrounding={setGrounding}
        health={healthText}
      />

      <Sidebar scope={scope} onScope={setScope} onNewChat={clear} />

      <main className="flex-1 flex flex-col min-w-0">
        <HeaderBar
          activeLabel={activeLabel}
          models={models}
          activeId={active}
          onSelectModel={async (id) => {
            await select(id);
            notify(`Model switched`);
          }}
          onConfig={() => setConfigOpen(true)}
          onExport={exportChat}
        />

        <div className="flex-1 overflow-y-auto">
          {showWelcome && <HeroOrb />}
          {showWelcome && (
            <QuickActionPills
              onPolicy={quickPolicy}
              onGuidebook={quickGuidebook}
              onEval={handleEval}
            />
          )}
          <ChatThread messages={messages} loading={loading} />
          {showWelcome && (
            <FeatureCards
              onPolicy={quickPolicy}
              onGuidebook={quickGuidebook}
              onEval={handleEval}
            />
          )}
        </div>

        <ChatInput
          onSend={send}
          loading={loading}
          onOpenConfig={() => setConfigOpen(true)}
        />
      </main>
    </div>
  );
}