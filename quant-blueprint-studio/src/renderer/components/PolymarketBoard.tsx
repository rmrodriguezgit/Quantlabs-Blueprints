import { useState } from "react";
import { LayoutDashboard, Pause, Play } from "lucide-react";
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { DashboardRow, MarketDashboardData, PolymarketDashboardData } from "../types/blueprint";
import { MarketBoard } from "./MarketBoard";

interface Props {
  data: PolymarketDashboardData | MarketDashboardData;
  live: boolean;
  running: boolean;
  onToggle: () => void;
}

const money = (v: unknown) => {
  const n = Number(v);
  if (v == null || !Number.isFinite(n)) return "—";
  return n >= 1e9 ? `$${(n / 1e9).toFixed(2)} B` : n >= 1e6 ? `$${(n / 1e6).toFixed(2)} M` : n >= 1e3 ? `$${(n / 1e3).toFixed(1)} k` : `$${n.toFixed(0)}`;
};
const pct = (v: unknown, digits = 1) => (v == null || !Number.isFinite(Number(v)) ? "—" : `${(Number(v) * 100).toFixed(digits)}%`);
const hours = (v: unknown) => {
  const h = Number(v);
  if (v == null || !Number.isFinite(h)) return "—";
  if (h < 0) return "por resolver";
  return h < 1 ? `${Math.round(h * 60)} min` : h < 48 ? `${h.toFixed(1)} h` : `${Math.round(h / 24)} d`;
};
const PALETTE = ["#7aa2f7", "#9ece6a", "#e0af68", "#bb9af7", "#f7768e", "#7dcfff", "#ff9e64", "#73daca", "#c0caf5", "#a9b1d6", "#db4b4b", "#41a6b5"];

const TABS: { key: keyof PolymarketDashboardData; label: string; hint: string }[] = [
  { key: "top_markets", label: "Más negociados", hint: "Mercados abiertos con mayor volumen en 24 h." },
  { key: "movers", label: "Mayores movimientos", hint: "Mayor cambio de probabilidad en 24 h con al menos $100 negociados (excluye mercados casi resueltos)." },
  { key: "contested", label: "Más disputados", hint: "Probabilidad entre 40 % y 60 % con volumen: el resultado está abierto." },
  { key: "ending_soon", label: "Cierran en 24 h", hint: "Mercados que vencen en las próximas 24 h, por volumen." },
  { key: "most_liquid", label: "Más líquidos", hint: "Mayor liquidez en el libro de órdenes." },
  { key: "wide_spreads", label: "Spreads amplios", hint: "Mercados activos con mayor diferencia bid/ask: precio poco fiable." },
];

function MarketTable({ rows }: { rows: DashboardRow[] }) {
  if (!rows.length) return <p className="muted">Sin mercados en esta vista para el filtro actual.</p>;
  return (
    <div className="board-table-wrap">
      <table className="live-log board-table">
        <thead>
          <tr>
            <th>Mercado</th><th>Categoría</th><th>Líder</th><th>Prob.</th><th>Δ 24 h</th><th>Vol. 24 h</th><th>Liquidez</th><th>Spread</th><th>Cierre</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => {
            const ch = Number(r["cambio_24h"]);
            return (
              <tr key={`${String(r.mercado)}-${i}`}>
                <td className="board-name">
                  <a href={String(r.url)} target="_blank" rel="noreferrer" title={String(r.evento ?? "")}>{String(r.mercado)}</a>
                </td>
                <td>{String(r["categoría"] ?? "")}</td>
                <td>{String(r["outcome_líder"] ?? "")}</td>
                <td>
                  <span className="board-prob"><span style={{ width: pct(r["prob_líder"], 0) }} /></span>
                  {pct(r["prob_líder"], 0)}
                </td>
                <td className={Number.isFinite(ch) ? (ch >= 0 ? "up" : "down") : ""}>
                  {Number.isFinite(ch) && r["cambio_24h"] != null ? `${ch >= 0 ? "+" : ""}${(ch * 100).toFixed(1)}` : "—"}
                </td>
                <td>{money(r["volumen_24h"])}</td>
                <td>{money(r["liquidez"])}</td>
                <td>{r.spread == null ? "—" : Number(r.spread).toFixed(3)}</td>
                <td>{hours(r["horas_al_cierre"])}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/** Polymarket Dashboard: KPIs, categories, rankings and distribution of the whole active market. */
export function PolymarketBoard(props: Props) {
  return props.data.mode === "market" ? (
    <MarketBoard data={props.data} live={props.live} running={props.running} onToggle={props.onToggle} />
  ) : (
    <OverviewBoard {...props} data={props.data} />
  );
}

function OverviewBoard({ data, live, running, onToggle }: Props & { data: PolymarketDashboardData }) {
  const [tab, setTab] = useState<keyof PolymarketDashboardData>("top_markets");
  const k = data.kpis;
  const active = TABS.find((t) => t.key === tab) ?? TABS[0];
  return (
    <section className="live-dashboard board">
      <header>
        <div>
          <LayoutDashboard size={15} /> <strong>Polymarket Dashboard</strong>
          <span className="muted">
            {" "}
            · {String(data.filters.category)} · {String(data.filters.sort)} · {data.filters.pages}×50 eventos
            {data.filters.query ? ` · «${String(data.filters.query)}»` : ""} · {new Date(data.fetched_at).toLocaleTimeString()} ({data.seconds}s)
          </span>
        </div>
        <div className="live-controls">
          <span className="muted">{running ? "actualizando…" : ""}</span>
          <button className={live ? "live-on" : ""} onClick={onToggle}>
            {live ? <Pause size={13} /> : <Play size={13} />} {live ? "Detener ciclo" : "Ciclo en vivo"}
          </button>
        </div>
      </header>

      <div className="board-kpis">
        {[
          ["Volumen 24 h", money(k["volumen_24h"])],
          ["Interés abierto", money(k["interés_abierto"])],
          ["Liquidez", money(k["liquidez"])],
          ["Eventos", String(k.eventos ?? "—")],
          ["Mercados abiertos", String(k.mercados ?? "—")],
          ["Cierran en 24 h", `${k["cierran_24h"] ?? "—"} merc. · ${k["eventos_cierran_24h"] ?? "—"} ev.`],
          ["Disputados (40–60 %)", String(k["mercados_disputados"] ?? "—")],
          ["Spread mediano", k["spread_mediano"] == null ? "—" : Number(k["spread_mediano"]).toFixed(3)],
          ["Top 10 eventos", pct(k["concentración_top10"], 0) + " del vol."],
        ].map(([label, value]) => (
          <div key={label} className="live-card board-kpi">
            <span className="label">{label}</span>
            <span className="value">{value}</span>
          </div>
        ))}
      </div>

      <div className="live-charts">
        <div>
          <h4>Volumen 24 h por {data.filters.category === "todas" ? "categoría" : "subcategoría"}</h4>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={data.categories} layout="vertical" margin={{ left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#ffffff14" />
              <XAxis type="number" tick={{ fontSize: 10 }} tickFormatter={money} />
              <YAxis type="category" dataKey="categoría" tick={{ fontSize: 10 }} width={95} />
              <Tooltip formatter={(v) => money(v)} />
              <Bar dataKey="volumen_24h" isAnimationActive={false}>
                {data.categories.map((_, i) => <Cell key={i} fill={PALETTE[i % PALETTE.length]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div>
          <h4>Distribución de probabilidades (primer outcome)</h4>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={data.probability_histogram}>
              <CartesianGrid strokeDasharray="3 3" stroke="#ffffff14" />
              <XAxis dataKey="tramo" tick={{ fontSize: 9 }} interval={0} angle={-30} textAnchor="end" height={45} />
              <YAxis tick={{ fontSize: 10 }} />
              <Tooltip />
              <Bar dataKey="mercados" fill="#7aa2f7" isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="board-section">
        <div className="board-tabs" role="tablist">
          {TABS.map((t) => (
            <button key={String(t.key)} role="tab" aria-selected={tab === t.key} className={tab === t.key ? "active" : ""} onClick={() => setTab(t.key)}>
              {t.label} <span className="muted">({(data[t.key] as DashboardRow[]).length})</span>
            </button>
          ))}
        </div>
        <p className="muted">{active.hint}</p>
        <MarketTable rows={data[tab] as DashboardRow[]} />
      </div>

      <div className="live-charts">
        <div>
          <h4>Eventos más activos (24 h)</h4>
          <div className="board-table-wrap">
            <table className="live-log board-table">
              <thead>
                <tr><th>Evento</th><th>Categoría</th><th>Vol. 24 h</th><th>Interés abierto</th><th>Mercados</th><th>Cierre</th></tr>
              </thead>
              <tbody>
                {data.top_events.map((r, i) => (
                  <tr key={i}>
                    <td className="board-name"><a href={String(r.url)} target="_blank" rel="noreferrer">{String(r.evento)}</a></td>
                    <td>{String(r["categoría"] ?? "")}</td>
                    <td>{money(r["volumen_24h"])}</td>
                    <td>{money(r["interés_abierto"])}</td>
                    <td>{String(r.mercados)}</td>
                    <td>{hours(r["horas_al_cierre"])}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        <div>
          <h4>Categorías</h4>
          <table className="live-log board-table">
            <thead>
              <tr><th>Categoría</th><th>Eventos</th><th>Vol. 24 h</th><th>Interés abierto</th></tr>
            </thead>
            <tbody>
              {data.categories.map((r, i) => (
                <tr key={i}>
                  <td><span className="board-dot" style={{ background: PALETTE[i % PALETTE.length] }} />{String(r["categoría"])}</td>
                  <td>{String(r.eventos)}</td>
                  <td>{money(r["volumen_24h"])}</td>
                  <td>{money(r["interés_abierto"])}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {data.updown.length > 0 && (
            <>
              <h4 style={{ marginTop: 12 }}>Mercados Up/Down recurrentes</h4>
              <table className="live-log board-table">
                <thead>
                  <tr><th>Activo</th><th>Ventana</th><th>Mercados</th><th>Vol. 24 h</th><th>P(UP) media</th></tr>
                </thead>
                <tbody>
                  {data.updown.map((r, i) => (
                    <tr key={i}>
                      <td>{String(r.activo).toUpperCase()}</td>
                      <td>{String(r.ventana)}</td>
                      <td>{String(r.mercados)}</td>
                      <td>{money(r["volumen_24h"])}</td>
                      <td>{pct(r["prob_up_media"], 0)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}
        </div>
      </div>
      <p className="muted">Instantánea de la Gamma API (eventos activos, páginas descargadas). Datos informativos; no es asesoría financiera.</p>
    </section>
  );
}
