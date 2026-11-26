import { useEffect, useState } from "react";
import { Workflow, X, Play, FolderOpen, CircleAlert, CheckCircle2, ListChecks, ThumbsUp, ThumbsDown, Sparkles } from "lucide-react";
import { brainMemory, planBlueprint, ratePlan, setRequestStatus, type BrainEngine, type BrainMemory, type BrainPlan } from "../brain";
import { LlmStatus } from "./LlmStatus";
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
  initialPrompt,
  onClose,
  onOpen,
}: {
  initialPrompt?: string;
  onClose: () => void;
  onOpen: (graph: Graph, title: string, run: boolean) => void;
}) {
  const saved = restore();
  const [prompt, setPrompt] = useState(initialPrompt ?? saved.prompt);
  const [tab, setTab] = useState<"build" | "missing" | "memory">("build");
  const [memory, setMemory] = useState<BrainMemory | null>(null);
  const [rating, setRating] = useState<Record<string, number>>({});
  const loadMemory = () => brainMemory().then(setMemory, () => setMemory(null));
  const [engine, setEngine] = useState<BrainEngine>(saved.engine);
  const [plan, setPlan] = useState<BrainPlan | null>(saved.plan);
  const [busy, setBusy] = useState(false);
  const [started, setStarted] = useState(0);
  const [now, setNow] = useState(0);
  const [error, setError] = useState("");
  useEffect(() => {
    if (tab !== "build") void loadMemory();
  }, [tab]); // eslint-disable-line react-hooks/exhaustive-deps
  const rate = async (value: -1 | 1) => {
    if (!plan?.plan_id) return;
    const next = rating[plan.plan_id] === value ? 0 : value;
    try {
      await ratePlan(plan.plan_id, next);
      setRating((r) => ({ ...r, [plan.plan_id as string]: next }));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };
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
      <section className="brain-panel" role="dialog" aria-label="Cerebro Constructor">
        <header>
          <div>
            <small>SEGUNDO CEREBRO · LOCAL</small>
            <h2>
              <Workflow size={20} /> Cerebro Constructor
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
        <div className="brain-tabs" role="tablist">
          {(
            [
              ["build", "Construir"],
              ["missing", `Herramientas faltantes${memory ? ` · ${memory.requests.filter((r) => r.status === "pending").length}` : ""}`],
              ["memory", "Memoria"],
            ] as const
          ).map(([id, text]) => (
            <button key={id} role="tab" aria-selected={tab === id} className={tab === id ? "active" : ""} onClick={() => setTab(id)}>
              {text}
            </button>
          ))}
          <LlmStatus />
        </div>
        {tab === "missing" && (
          <div className="brain-memory">
            <p className="muted">
              Lo que pediste y el catálogo no cubre. El Cerebro nunca lo simula con otro nodo: queda aquí como ficha para implementarla.
            </p>
            {!memory?.requests.length && <p className="muted">Sin fichas pendientes.</p>}
            {memory?.requests.map((r) => (
              <article key={r.id} className={`brain-request ${r.status}`}>
                <header>
                  <strong>{r.name}</strong>
                  <span>
                    pedida {r.count} vez{r.count > 1 ? "es" : ""}
                  </span>
                  <select
                    value={r.status}
                    onChange={async (e) => {
                      await setRequestStatus(r.id, e.target.value as typeof r.status);
                      void loadMemory();
                    }}
                  >
                    <option value="pending">Pendiente</option>
                    <option value="planned">Planificada</option>
                    <option value="discarded">Descartada</option>
                  </select>
                </header>
                <p>{r.purpose}</p>
                <small>Último enunciado: {r.prompts[r.prompts.length - 1]}</small>
              </article>
            ))}
          </div>
        )}
        {tab === "memory" && (
          <div className="brain-memory">
            <p className="muted">
              Blueprints construidos. Los valorados 👍 ({memory?.liked ?? 0}) se reutilizan cuando un enunciado parecido llega. Guardado solo en este equipo
              {memory ? `: ${memory.path}` : ""}.
            </p>
            <ol className="brain-history">
              {memory?.history.map((h) => (
                <li key={h.id}>
                  <span className="brain-rating">{h.rating === 1 ? "👍" : h.rating === -1 ? "👎" : "·"}</span>
                  <div>
                    <strong>{h.title}</strong>
                    <small>
                      {new Date(h.at * 1000).toLocaleString()} · {h.engine === "ollama" ? h.model : "reglas"} · {h.steps.length} pasos
                    </small>
                    <p>«{h.prompt}»</p>
                  </div>
                  <button disabled={busy} onClick={() => setPrompt(h.prompt)} title="Copiar el enunciado al constructor">
                    Reusar
                  </button>
                </li>
              ))}
            </ol>
          </div>
        )}
        {tab === "build" && (
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
        )}
        {tab === "build" && error && (
          <p className="brain-error">
            <CircleAlert size={14} /> {error}
          </p>
        )}
        {tab === "build" && plan && !busy && (
          <div className="brain-result">
            <div className="brain-result-head">
              <div>
                <h3>
                  <Sparkles size={15} /> {plan.title}
                </h3>
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
              {plan.plan_id && (
                <div className="brain-rate" aria-label="Valorar este blueprint">
                  <span>¿Te sirvió?</span>
                  <button
                    className={rating[plan.plan_id] === 1 ? "active" : ""}
                    onClick={() => void rate(1)}
                    title="Útil: se reutiliza en enunciados parecidos"
                  >
                    <ThumbsUp size={14} />
                  </button>
                  <button className={rating[plan.plan_id] === -1 ? "active" : ""} onClick={() => void rate(-1)} title="No útil">
                    <ThumbsDown size={14} />
                  </button>
                </div>
              )}
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
