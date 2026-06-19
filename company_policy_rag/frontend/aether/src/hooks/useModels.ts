import { useCallback, useEffect, useState } from "react";
import { fetchModels, ModelInfo, setActiveModel } from "../api/client";

export function useModels() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [active, setActive] = useState<string | null>(null);
  const [connected, setConnected] = useState(false);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const data = await fetchModels();
      setModels(data.models);
      setActive(data.active_model);
      setConnected(data.connected);
    } catch {
      setConnected(false);
    } finally {
      setLoading(false);
    }
  }, []);

  const select = useCallback(async (id: string) => {
    await setActiveModel(id);
    setActive(id);
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const activeLabel =
    models.find((m) => m.id === active)?.label ?? active ?? "Loading…";

  return { models, active, activeLabel, connected, loading, select, refresh };
}