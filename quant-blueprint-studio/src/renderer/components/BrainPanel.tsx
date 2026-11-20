import { useEffect, useState } from "react";
import { Sparkles, X, Play, FolderOpen, CircleAlert, CheckCircle2, Cpu, ListChecks } from "lucide-react";
import { brainStatus, planBlueprint, type BrainEngine, type BrainPlan, type BrainStatus } from "../brain";
import { validateGraph } from "../validation";
import type { Graph } from "../types/blueprint";

const KEY = "quantlabs-brain-v1";
const EXAMPLES = [
  "Analiza NVDA 4 años con indicadores técnicos, una red neuronal y backtest",
  "Agente de refuerzo sobre Bitcoin con RSI y MACD",
  "Portafolio de AAPL, MSFT y GOOGL con métricas de riesgo",
  "Laya sobre QQQ: preguntas, ajuste de pesos y retroalimentación",
  "Compara Ridge, Random Forest y boosting en SPY",
];

function restore(): { prompt: string; engine: BrainEngine; plan: BrainPlan | null } {
  try {
    const value = JSON.parse(localStorage.getItem(KEY) ?? "null") as ReturnType<typeof restore> | null;
    if (value && typeof value.prompt === "string") return value;
  } catch {
    /* fresh panel */
  }
  return { prompt: "", engine: "auto", plan: null };
}

export function BrainPanel({
  onClose,
  onOpen,
}: {
  onClose: () => void;
  onOpen: (graph: Graph, title: string, run: boolean) => void;
}) {
  const saved = restore();
  const [prompt, setPrompt] = useState(saved.prompt);
  const [engine, setEngine] = useState<BrainEngine>(saved.engine);
  const [plan, setPlan] = useState<BrainPlan | null>(saved.plan);
  const [status, setStatus] = useState<BrainStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [started, setStarted] = useState(0);
  const [now, setNow] = useState(0);
  const [error, setError] = useState("");
  useEffect(() => {
    brainStatus().then(setStatus, (e: unknown) =>
      setStatus({ available: false, model: null, models: [], reason: e instanceof Error ? e.message : String(e) }),
    );
  }, []);
  useEffect(() => {
    try {
      localStorage.setItem(KEY, JSON.stringify({ prompt, engine, plan }));
    } catch {
      /* in-memory only */
    }
  }, [prompt, engine, plan]);
  useEffect(() => {
    if (!busy) return;
    const t = window.setInterval(() => setNow(Date.now()), 500);
    return () => window.clearInterval(t);
  }, [busy]);
  const build = async () => {
    if (!prompt.trim() || busy) return;
    setBusy(true);
    setError("");
    setStarted(Date.now());
    setNow(Date.now());
    try {
      setPlan(await planBlueprint(prompt, engine));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };
  const issues = plan ? validateGraph(plan.graph) : [];
  const errors = issues.filter((i) => i.severity === "error");
  return (
    <div className="library-backdrop" onMouseDown={(e) => e.target === e.currentTarget && !busy && onClose()}>
      <section className="brain-panel" role="dialog" aria-label="Cerebro IA">
        <header>
          <div>
            <small>SEGUNDO CEREBRO · LOCAL</small>
            <h2>
              <Sparkles size={20} /> Cerebro IA
            </h2>
            <p>
              Describe el análisis. El Cerebro recupera herramientas y plantillas, planifica con Qwen local (o reglas),
              compila solo con nodos del catálogo y valida el grafo. Tú revisas antes de abrirlo.
            </p>
          </div>
          <button onClick={onClose} disabled={busy} aria-label="Cerrar">
            <X size={16} />
          </button>
        </header>
        <div className="brain-status">
          <Cpu size={13} />
          {status === null
            ? "Comprobando Ollama…"
            : status.available
              ? `Qwen local disponible · ${status.model}`
              : `Sin modelo local (${status.reason ?? "Ollama no responde"}). Se usará el planificador por reglas.`}
        </div>
        <div className="brain-input">
          <textarea
            value={prompt}
            maxLength={2000}
            placeholder="Ej.: Analiza Apple 5 años con indicadores, una red neuronal y backtest"
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) void build();
            }}
          />
          <div className="brain-controls">
            <select value={engine} onChange={(e) => setEngine(e.target.value as BrainEngine)} disabled={busy}>
              <option value="auto">Automático (Qwen → reglas)</option>
              <option value="ollama">Solo Qwen</option>
              <option value="rules">Solo reglas (instantáneo)</option>
            </select>
            <button className="execute-button" disabled={busy || !prompt.trim()} onClick={() => void build()}>
              <Sparkles size={14} />
              {busy ? `Construyendo… ${Math.round((now - started) / 1000)} s` : "Construir blueprint (⌘↵)"}
            </button>
          </div>
          <div className="brain-examples">
            {EXAMPLES.map((example) => (
              <button key={example} disabled={busy} onClick={() => setPrompt(example)}>
                {example}
              </button>
            ))}
          </div>
        </div>
        {error && (
          <p className="brain-error">
            <CircleAlert size={14} /> {error}
          </p>
        )}
        {plan && !busy && (
          <div className="brain-result">
            <div className="brain-result-head">
              <div>
                <h3>{plan.title}</h3>
                <p>{plan.objective}</p>
                <p className="muted">
                  Motor: <strong>{plan.engine === "ollama" ? `Qwen · ${plan.model}` : "reglas deterministas"}</strong>
                  {plan.fallback_reason && ` · se usaron reglas porque: ${plan.fallback_reason}`}
                  {plan.detected.tickers && ` · activos detectados: ${plan.detected.tickers.join(", ")}`}
                  {plan.detected.years && ` · ${plan.detected.years} años`}
                </p>
              </div>
              <div className={`brain-check ${plan.check.ok && !errors.length ? "ok" : "bad"}`}>
                {plan.check.ok && !errors.length ? <CheckCircle2 size={15} /> : <CircleAlert size={15} />}
                {plan.check.ok && !errors.length
                  ? "Grafo válido para el motor"
                  : `${plan.check.errors.length + errors.length} problema(s)`}
              </div>
            </div>
            <ol className="brain-steps">
              {plan.steps.map((step) => (
                <li key={step.id}>
                  <strong>{step.label}</strong> <code>{step.tool}</code>
                  {step.why && <span> — {step.why}</span>}
                </li>
              ))}
            </ol>
            {[...plan.check.errors, ...errors.map((e) => e.message)].length > 0 && (
              <ul className="brain-warnings">
                {[...plan.check.errors, ...errors.map((e) => e.message)].map((m, i) => (
                  <li key={i}>{m}</li>
                ))}
              </ul>
            )}
            {(plan.dropped.length > 0 || plan.notes.length > 0) && (
              <details>
                <summary>
                  <ListChecks size={13} /> Ajustes del compilador ({plan.dropped.length + plan.notes.length})
                </summary>
                <ul>
                  {plan.dropped.map((d, i) => (
                    <li key={`d${i}`}>
                      Paso descartado «{d.step}» ({d.tool}): {d.reason}
                    </li>
                  ))}
                  {plan.notes.map((n, i) => (
                    <li key={`n${i}`}>{n}</li>
                  ))}
                </ul>
              </details>
            )}
            {plan.missing.length > 0 && (
              <div className="brain-missing">
                <strong>Herramientas que faltan (no se simulan):</strong>
                <ul>
                  {plan.missing.map((m, i) => (
                    <li key={i}>
                      <b>{m.name}</b>: {m.purpose}
                    </li>
                  ))}
                </ul>
              </div>
            )}
            <details>
              <summary>Fuentes recuperadas (BM25)</summary>
              <ul>
                {plan.retrieved.map((r) => (
                  <li key={r.type + r.id}>
                    {r.type === "tool" ? "Herramienta" : "Plantilla"} · {r.title} · {r.score}
                  </li>
                ))}
              </ul>
            </details>
            <div className="brain-open">
              <button onClick={() => onOpen(plan.graph, plan.title, false)}>
                <FolderOpen size={14} /> Abrir en lienzo
              </button>
              <button
                className="execute-button"
                disabled={!plan.check.ok || errors.length > 0}
                onClick={() => onOpen(plan.graph, plan.title, true)}
              >
                <Play size={14} fill="currentColor" /> Abrir y ejecutar
              </button>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
