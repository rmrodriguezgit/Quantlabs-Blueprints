import { useEffect, useState } from "react";
import { brainPing, type Ping } from "../brain";

type Light = "green" | "amber" | "red" | "checking";
const LIGHT: Record<Ping["state"], Light> = { ok: "green", slow: "amber", no_model: "amber", offline: "red" };

function describe(ping: Ping): { text: string; title: string } {
  if (ping.state === "offline")
    return {
      text: "Sin modelo · reglas",
      title: `${ping.error ?? "Ollama no responde"}. El Cerebro usa reglas deterministas. Abre Ollama y pulsa para volver a comprobar.`,
    };
  if (ping.state === "no_model")
    return { text: "Ollama sin modelo · reglas", title: `${ping.error ?? "Sin modelo"}: ollama pull qwen3.5:4b` };
  return {
    text: `${ping.model} · ${ping.seconds} s`,
    title: `${ping.state === "slow" ? "Responde, pero lento. " : "Conectado: el modelo generó una respuesta de prueba. "}Ollama en ${ping.url}; respuesta «${ping.reply ?? ""}» en ${ping.seconds} s. Pulsa para volver a comprobar.`,
  };
}

/** Semáforo del modelo local: verde responde, ámbar lento o sin modelo, rojo sin conexión. Pulsa para comprobar. */
export function LlmStatus({ compact = false }: { compact?: boolean }) {
  const [ping, setPing] = useState<Ping | null>(null);
  const [checking, setChecking] = useState(false);
  const check = async () => {
    if (checking) return;
    setChecking(true);
    try {
      setPing(await brainPing());
    } catch (e) {
      setPing({ state: "offline", url: "", model: null, seconds: null, reply: null, error: e instanceof Error ? e.message : String(e) });
    } finally {
      setChecking(false);
    }
  };
  useEffect(() => {
    void check();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps
  const light: Light = checking || !ping ? "checking" : LIGHT[ping.state];
  const info =
    checking || !ping
      ? { text: "Comprobando modelo…", title: "Comprobando que Ollama responde y que el modelo genera texto." }
      : describe(ping);
  return (
    <button
      type="button"
      className={`llm-status ${light}${compact ? " compact" : ""}`}
      onClick={() => void check()}
      disabled={checking}
      title={info.title}
      aria-label={`Estado del modelo local: ${info.text}. Pulsa para comprobar.`}
    >
      <span className="llm-dot" aria-hidden="true" />
      {!compact && <span>{info.text}</span>}
    </button>
  );
}
