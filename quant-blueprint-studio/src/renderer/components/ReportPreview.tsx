import { useEffect, useMemo, useState } from "react";
import { createPortal } from "react-dom";
import { FileText, Printer, X } from "lucide-react";
import glossary from "../data/glossary.json";
import { explainResult, type Explanation } from "../brain";
import { catalog, presentPlot, type Graph, type Result, type StudyReport } from "../types/blueprint";
import { PlotDock } from "./PlotDock";

type Term = { id: string; title: string; body: string; how?: string; tools?: string[] };
const TERMS = (glossary as { terms: Term[] }).terms;

function fmt(v: unknown): string {
  if (typeof v === "number") return Number.isFinite(v) ? (Math.abs(v) >= 1e9 ? v.toExponential(3) : String(Number(v.toPrecision(5)))) : "—";
  if (v === null || v === undefined || v === "") return "—";
  return String(v);
}

function order(graph: Graph): string[] {
  const indegree = new Map(graph.nodes.map((n) => [n.id, 0]));
  graph.edges.forEach((e) => indegree.set(e.target, (indegree.get(e.target) ?? 0) + 1));
  const byX = (a: string, b: string) =>
    (graph.nodes.find((n) => n.id === a)?.position.x ?? 0) - (graph.nodes.find((n) => n.id === b)?.position.x ?? 0);
  const queue = [...indegree].filter(([, d]) => d === 0).map(([id]) => id).sort(byX);
  const out: string[] = [];
  while (queue.length) {
    const id = queue.shift() as string;
    out.push(id);
    graph.edges
      .filter((e) => e.source === id)
      .forEach((e) => {
        const d = (indegree.get(e.target) ?? 1) - 1;
        indegree.set(e.target, d);
        if (d === 0) queue.push(e.target);
      });
    queue.sort(byX);
  }
  return out.length === graph.nodes.length ? out : graph.nodes.map((n) => n.id);
}

function Table({ rows, limit = 15 }: { rows: Record<string, unknown>[]; limit?: number }) {
  const columns = [...new Set(rows.flatMap((r) => Object.keys(r)))].slice(0, 9);
  if (!rows.length) return null;
  return (
    <table className="report-table">
      <thead>
        <tr>
          {columns.map((c) => (
            <th key={c}>{c}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.slice(0, limit).map((r, i) => (
          <tr key={i}>
            {columns.map((c) => (
              <td key={c}>{fmt(r[c])}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/** Printable report of the executed blueprint: one section per step, in execution order. */
export function ReportPreview({ graph, result, title, onClose }: { graph: Graph; result: Result; title: string; onClose: () => void }) {
  const [readings, setReadings] = useState<Record<string, Explanation | "error">>({});
  const [figures, setFigures] = useState(true);
  const studies = useMemo(() => new Map((result.studies ?? []).map((s) => [s.id, s])), [result]);
  useEffect(() => {
    // Deterministic readings only: the report must not depend on a language model.
    (result.studies ?? []).forEach((study) =>
      explainResult(study, "rules").then(
        (value) => setReadings((r) => ({ ...r, [study.id]: value })),
        () => setReadings((r) => ({ ...r, [study.id]: "error" })),
      ),
    );
  }, [result]);
  const ids = order(graph);
  const kinds = new Set(graph.nodes.map((n) => n.data.kind));
  const concepts = TERMS.filter((t) => t.tools?.some((k) => kinds.has(k as never)));
  const sources = graph.nodes
    .filter((n) => n.data.kind === "source" || n.data.kind === "portfolio_data")
    .map((n) => `${n.data.params.ticker ?? n.data.params.tickers ?? "—"} · ${n.data.params.source ?? "yfinance"} · ${n.data.params.years ?? "—"} años`);
  const summary = (id: string) => ((result as unknown as { nodes?: { id: string; rows?: number; columns?: string[] }[] }).nodes)?.find((n) => n.id === id);

  return createPortal(
    <div className="report-backdrop">
      <div className="report-toolbar">
        <strong>
          <FileText size={15} /> Reporte · vista previa
        </strong>
        <label>
          <input type="checkbox" checked={figures} onChange={(e) => setFigures(e.target.checked)} /> Incluir gráficos
        </label>
        <button className="execute-button" onClick={() => window.print()}>
          <Printer size={14} /> Imprimir / guardar PDF
        </button>
        <button onClick={onClose} aria-label="Cerrar">
          <X size={15} />
        </button>
      </div>
      <article className="report-page">
        <header className="report-cover">
          <small>QUANTLABS · BLUEPRINTS · REPORTE DE INVESTIGACIÓN</small>
          <h1>{title}</h1>
          <p>
            {new Date().toLocaleString()} · {graph.nodes.length} pasos · {graph.edges.length} conexiones
          </p>
          {sources.length > 0 && <p>Datos: {sources.join(" | ")}</p>}
          <p className="report-note">
            Documento educativo generado a partir de una ejecución local. Las métricas son históricas y no constituyen asesoría financiera ni promesa de
            rentabilidad.
          </p>
        </header>
        {ids.map((id, index) => {
          const node = graph.nodes.find((n) => n.id === id);
          if (!node) return null;
          const def = catalog[node.data.kind];
          const inputs = graph.edges
            .filter((e) => e.target === id)
            .map((e) => `${graph.nodes.find((n) => n.id === e.source)?.data.label ?? e.source} (${e.targetHandle})`);
          const changed = def.fields.map((f) => ({ f, value: node.data.params[f.key] ?? f.value }));
          const out = summary(id);
          const study: StudyReport | undefined = studies.get(id);
          const reading = readings[id];
          const plots = (result.plots ?? []).filter((p) => p.id === id);
          const risk = [...(result.risk_reports ?? []), ...(result.portfolios ?? [])].filter((r) => r.id === id);
          return (
            <section key={id} className="report-section">
              <header>
                <span className="report-step">Paso {index + 1}</span>
                <h2>{node.data.label || def.label}</h2>
                {node.data.label !== def.label && <small>{def.label}</small>}
              </header>
              <p>
                <strong>Qué hace.</strong> {def.subtitle}. <span className="muted">({def.category})</span>
              </p>
              {inputs.length > 0 && (
                <p>
                  <strong>Datos de entrada.</strong> {inputs.join("; ")}.
                </p>
              )}
              {changed.length > 0 && (
                <dl className="report-params">
                  {changed.map(({ f, value }) => (
                    <div key={f.key} className={value !== f.value ? "changed" : ""}>
                      <dt>{f.label}</dt>
                      <dd>{fmt(value)}</dd>
                    </div>
                  ))}
                </dl>
              )}
              {out?.rows !== undefined && (
                <p className="muted">
                  Salida: {out.rows} filas{out.columns ? ` · ${out.columns.length} columnas (${out.columns.slice(0, 10).join(", ")}${out.columns.length > 10 ? "…" : ""})` : ""}.
                </p>
              )}
              {study && (
                <>
                  <p>
                    <strong>Selección.</strong> {study.selected} · TR {study.splits.train} / VA {study.splits.validation} / TE {study.splits.test}
                  </p>
                  <div className="report-metrics">
                    {Object.entries(study.metrics).map(([k, v]) => (
                      <div key={k}>
                        <small>{k}</small>
                        <strong>{fmt(v)}</strong>
                      </div>
                    ))}
                  </div>
                  <Table rows={study.table} />
                  {study.notes.length > 0 && (
                    <ul className="report-notes">
                      {study.notes.map((n, i) => (
                        <li key={i}>{n}</li>
                      ))}
                    </ul>
                  )}
                  {reading && reading !== "error" && (
                    <div className="report-reading">
                      <h3>Lectura de resultados</h3>
                      <p>{reading.rules.summary}</p>
                      <ul>
                        {[...reading.rules.findings, ...reading.rules.cautions].map((t, i) => (
                          <li key={i}>{t}</li>
                        ))}
                      </ul>
                      {reading.rules.next_steps.length > 0 && (
                        <p>
                          <strong>Siguiente paso.</strong> {reading.rules.next_steps.join(" ")}
                        </p>
                      )}
                    </div>
                  )}
                  {figures && study.charts.length > 0 && (
                    <PlotDock
                      plots={study.charts.map((c, i) => ({ ...c, id: `${study.id}:${i}`, origin: study.selected, source: "study" }))}
                      onInspect={() => undefined}
                    />
                  )}
                </>
              )}
              {risk.map((r) => (
                <div key={r.title}>
                  <p>
                    <strong>{r.title}</strong> · {r.scope} · {r.start} → {r.end}
                  </p>
                  <Table rows={r.metrics as unknown as Record<string, unknown>[]} />
                </div>
              ))}
              {figures && plots.length > 0 && (
                <PlotDock plots={plots.map((p, i) => presentPlot(p, node, i))} onInspect={() => undefined} />
              )}
            </section>
          );
        })}
        {result.metrics && Object.keys(result.metrics).length > 0 && (
          <section className="report-section">
            <header>
              <h2>Métricas del modelo</h2>
            </header>
            <div className="report-metrics">
              {Object.entries(result.metrics).map(([k, v]) => (
                <div key={k}>
                  <small>{k}</small>
                  <strong>{fmt(v)}</strong>
                </div>
              ))}
            </div>
          </section>
        )}
        {concepts.length > 0 && (
          <section className="report-section">
            <header>
              <h2>Conceptos usados</h2>
            </header>
            <dl className="report-glossary">
              {concepts.map((t) => (
                <div key={t.id}>
                  <dt>{t.title}</dt>
                  <dd>
                    {t.body} {t.how}
                  </dd>
                </div>
              ))}
            </dl>
          </section>
        )}
      </article>
    </div>,
    document.body,
  );
}
