import { useEffect, useState } from "react";
import { LayoutDashboard, Pause, Play } from "lucide-react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { DashboardRow, MarketDashboardData } from "../types/blueprint";

interface Props {
  data: MarketDashboardData;
  live: boolean;
  running: boolean;
  onToggle: () => void;
}

const money = (v: unknown) => {
  const n = Number(v);
  if (v == null || !Number.isFinite(n)) return "—";
  return n >= 1e6 ? `$${(n / 1e6).toFixed(2)} M` : n >= 1e3 ? `$${(n / 1e3).toFixed(1)} k` : `$${n.toFixed(2)}`;
};
const pct = (v: unknown, d = 1) => (v == null || !Number.isFinite(Number(v)) ? "—" : `${(Number(v) * 100).toFixed(d)}%`);
const pts = (v: unknown) => (v == null || !Number.isFinite(Number(v)) ? "—" : `${Number(v) >= 0 ? "+" : ""}${(Number(v) * 100).toFixed(1)} pts`);
const usd2 = (v: unknown) => (v == null || !Number.isFinite(Number(v)) ? "—" : `$${Number(v).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`);
const clock = (s: number | null) => {
  if (s == null) return "—";
  if (s <= 0) return "cerrado / por resolver";
  const t = Math.round(s);
  if (t >= 172800) return `${Math.round(t / 86400)} d`;
  if (t >= 3600) return `${Math.floor(t / 3600)} h ${Math.floor((t % 3600) / 60)} min`;
  return `${Math.floor(t / 60)}:${String(t % 60).padStart(2, "0")}`;
};

/** Dashboard of the single market chosen in Polymarket Market. */
export function MarketBoard({ data, live, running, onToggle }: Props) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, []);
  const m = data.market;
  const k = data.kpis;
  const u = m.underlying;
  const age = (now - Date.parse(data.fetched_at)) / 1000;
  const toClose = k["segundos_al_cierre"] == null ? null : Number(k["segundos_al_cierre"]) - age;
  const upName = m.up_outcome;
  const d = data.decision;
  const siblings = data.siblings.filter((s) => !s.cerrado);
  return (
    <section className="live-dashboard board">
      <header>
        <div>
          <LayoutDashboard size={15} /> <strong>{m.question}</strong>
          <span className="muted">
            {" "}
            · {m.event_title && m.event_title !== m.question ? `${m.event_title} · ` : ""}
            {m.window ? `ventana ${m.window} · ` : ""}
            <a href={m.url} target="_blank" rel="noreferrer">ver en Polymarket</a> · {new Date(data.fetched_at).toLocaleTimeString()} ({data.seconds}s)
          </span>
        </div>
        <div className="live-controls">
          <span className="muted">{running ? "actualizando…" : `dato de hace ${Math.round(age)}s`}</span>
          <button className={live ? "live-on" : ""} onClick={onToggle}>
            {live ? <Pause size={13} /> : <Play size={13} />} {live ? "Detener ciclo" : "Ciclo en vivo"}
          </button>
        </div>
      </header>

      <div className="board-outcomes">
        {m.outcomes.map((name, i) => {
          const p = name === upName ? Number(k.p_up) : m.outcomes.length === 2 ? 1 - Number(k.p_up) : Number(m.outcome_prices[i]);
          return (
            <div key={name} className={`board-outcome ${i === 0 ? "first" : "second"}`}>
              <span className="label">{name}</span>
              <span className="value">{pct(p, 1)}</span>
              <span className="board-prob wide"><span style={{ width: pct(p, 0) }} /></span>
              {data.vwap[name] != null && <span className="muted">VWAP reciente {Number(data.vwap[name]).toFixed(3)}</span>}
            </div>
          );
        })}
        <div className="board-outcome">
          <span className="label">Cierre en</span>
          <span className="value">{clock(toClose)}</span>
          <span className="muted">{m.end ? new Date(m.end).toLocaleString() : ""} · {m.closed ? "cerrado" : m.accepting_orders ? "acepta órdenes" : "pausado"}</span>
        </div>
        {u && u.price_to_beat != null && (
          <div className="board-outcome">
            <span className="label">{u.asset} vs precio a vencer</span>
            <span className={`value ${u.leading === "UP" ? "up" : u.leading === "DOWN" ? "down" : ""}`}>
              {u.diff == null ? "—" : `${u.diff >= 0 ? "▲ +" : "▼ "}${usd2(Math.abs(u.diff))}`}
            </span>
            <span className="muted">{usd2(u.spot)} vs {usd2(u.price_to_beat)} (Chainlink)</span>
          </div>
        )}
      </div>

      <div className="board-kpis">
        {[
          ["Spread", k.spread == null ? "—" : Number(k.spread).toFixed(3)],
          ["Cambio 1 h", pts(k["cambio_1h"])],
          ["Cambio 24 h", pts(k["cambio_24h"])],
          ["Rango (historial + libro)", `${pct(k["mínimo_historial"], 0)} – ${pct(k["máximo_historial"], 0)}`],
          ["Volumen reciente", money(k["volumen_reciente_usd"])],
          ["Volumen 24 h (Gamma)", money(k["volumen_24h"])],
          ["Liquidez", money(k.liquidez)],
          ["Presión hacia " + upName, pct(k["presión_up_pct"], 0)],
          ["Operaciones", `${k.operaciones ?? "—"} · ${k["operaciones_por_min"] == null ? "—" : Number(k["operaciones_por_min"]).toFixed(1)}/min`],
          ["Operación media / mayor", `${money(k["operación_media_usd"])} / ${money(k["mayor_operación_usd"])}`],
          ["Profundidad ±5¢ bid / ask", `${money(k["profundidad_bid_usd"])} / ${money(k["profundidad_ask_usd"])}`],
          ["Desequilibrio libro", k.desequilibrio == null ? "—" : Number(k.desequilibrio).toFixed(2)],
        ].map(([label, value]) => (
          <div key={label} className="live-card board-kpi">
            <span className="label">{label}</span>
            <span className="value">{value}</span>
          </div>
        ))}
      </div>

      {d && (
        <div className={`live-decision ${d.decision === "UP" ? "up" : "down"} board-decision`}>
          <span className="label">Decisión LAYA UP/DOWN</span>
          <span className="big">{d.decision}</span>
          <span>P(UP) {pct(d.probability_up)} · confianza {pct(d.confidence)} · {d.actionable ? "accionable" : "sin señal"}{d.bucket != null ? ` · minuto ${d.bucket + 1}/${d.buckets_total}` : ""}</span>
        </div>
      )}

      <div className="live-charts">
        <div>
          <h4>P({upName}) · historial CLOB</h4>
          <ResponsiveContainer width="100%" height={210}>
            <LineChart data={data.history}>
              <CartesianGrid strokeDasharray="3 3" stroke="#ffffff14" />
              <XAxis dataKey="hora" tick={{ fontSize: 10 }} minTickGap={60} />
              <YAxis domain={[0, 1]} tick={{ fontSize: 10 }} />
              <Tooltip formatter={(v) => pct(v)} />
              <ReferenceLine y={0.5} stroke="#888" strokeDasharray="4 4" />
              <ReferenceLine y={Number(k.p_up)} stroke="#e0af68" strokeDasharray="2 3" label={{ value: "libro ahora", fill: "#e0af68", fontSize: 10, position: "insideTopRight" }} />
              <Line type="stepAfter" dataKey="P" stroke="#7aa2f7" dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div>
          <h4>Profundidad del libro «{upName}» (USD acumulado)</h4>
          {data.depth.length ? (
            <ResponsiveContainer width="100%" height={210}>
              <AreaChart data={data.depth}>
                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff14" />
                <XAxis dataKey="precio" type="number" domain={["dataMin", "dataMax"]} tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 10 }} tickFormatter={money} width={60} />
                <Tooltip formatter={(v) => money(v)} labelFormatter={(v) => `precio ${v}`} />
                <Area type="stepAfter" dataKey="compra" stroke="#9ece6a" fill="#9ece6a33" isAnimationActive={false} connectNulls={false} />
                <Area type="stepBefore" dataKey="venta" stroke="#f7768e" fill="#f7768e33" isAnimationActive={false} connectNulls={false} />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <p className="muted">{toClose != null && toClose < 60 ? "Libro vacío: el mercado está cerrando o ya cerró (Polymarket retira las órdenes al final de la ventana)." : "Libro vacío o no disponible."}</p>
          )}
        </div>
        <div>
          <h4>Flujo de operaciones por minuto (USD hacia cada outcome)</h4>
          <ResponsiveContainer width="100%" height={190}>
            <BarChart data={data.trade_flow}>
              <CartesianGrid strokeDasharray="3 3" stroke="#ffffff14" />
              <XAxis dataKey="minuto" tick={{ fontSize: 10 }} minTickGap={30} />
              <YAxis tick={{ fontSize: 10 }} tickFormatter={money} width={60} />
              <Tooltip formatter={(v) => money(v)} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Bar dataKey="hacia UP" name={`hacia ${upName}`} stackId="f" fill="#9ece6a" isAnimationActive={false} />
              <Bar dataKey="hacia DOWN" name={`hacia ${m.down_outcome}`} stackId="f" fill="#f7768e" isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div>
          <h4>Libro de órdenes (mejores 12 niveles)</h4>
          <div className="board-books">
            {Object.entries(data.book).map(([name, b]) => (
              <table key={name} className="live-log board-table">
                <thead>
                  <tr><th colSpan={3}>«{name}»</th></tr>
                  <tr><th>Bid</th><th>Ask</th><th>USD bid / ask</th></tr>
                </thead>
                <tbody>
                  {Array.from({ length: Math.max(b.bids.length, b.asks.length) }, (_, i) => (
                    <tr key={i}>
                      <td className="up">{b.bids[i] ? Number(b.bids[i].precio).toFixed(3) : ""}</td>
                      <td className="down">{b.asks[i] ? Number(b.asks[i].precio).toFixed(3) : ""}</td>
                      <td>{b.bids[i] ? money(b.bids[i].usd) : "—"} / {b.asks[i] ? money(b.asks[i].usd) : "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ))}
          </div>
        </div>
      </div>

      <div className="live-charts">
        <div>
          <h4>Últimas operaciones</h4>
          <div className="board-table-wrap">
            <table className="live-log board-table">
              <thead>
                <tr><th>Hora</th><th>Operación</th><th>Favorece</th><th>Precio</th><th>Acciones</th><th>USD</th></tr>
              </thead>
              <tbody>
                {data.trades.map((t: DashboardRow, i) => (
                  <tr key={i}>
                    <td>{new Date(String(t.hora)).toLocaleTimeString()}</td>
                    <td>{t.lado === "BUY" ? "compra" : "venta"} «{String(t.outcome)}»</td>
                    <td className={t.hacia === "UP" ? "up" : "down"}>{t.hacia === "UP" ? upName : m.down_outcome}</td>
                    <td>{Number(t.precio).toFixed(3)}</td>
                    <td>{Number(t.acciones).toFixed(1)}</td>
                    <td>{money(t.usd)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        <div>
          {siblings.length > 1 && (
            <>
              <h4>Otros mercados del evento «{m.event_title}»</h4>
              <div className="board-table-wrap">
                <table className="live-log board-table">
                  <thead>
                    <tr><th>Mercado</th><th>Prob.</th><th>Δ 24 h</th><th>Vol. 24 h</th></tr>
                  </thead>
                  <tbody>
                    {siblings.map((s, i) => (
                      <tr key={i} className={s.actual ? "live-current" : undefined}>
                        <td className="board-name">{String(s.mercado)}</td>
                        <td><span className="board-prob"><span style={{ width: pct(s.prob_primero, 0) }} /></span>{pct(s.prob_primero, 1)}</td>
                        <td className={Number(s.cambio_24h) >= 0 ? "up" : "down"}>{pts(s.cambio_24h)}</td>
                        <td>{money(s.volumen_24h)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
          {data.holders.length > 0 && (
            <>
              <h4 style={{ marginTop: 12 }}>Concentración de tenedores (20 mayores por outcome)</h4>
              <table className="live-log board-table">
                <thead>
                  <tr><th>Outcome</th><th>Acciones de los 20</th><th>Mayor / 20</th><th>Top 5 / 20</th></tr>
                </thead>
                <tbody>
                  {data.holders.map((h, i) => (
                    <tr key={i}>
                      <td>{String(h.outcome)}</td>
                      <td>{Number(h.acciones_top).toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                      <td>{pct(h.mayor_tenedor_pct_top, 0)}</td>
                      <td>{pct(h.top5_pct_top, 0)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="muted">Sólo agregados: no se muestran nombres ni direcciones.</p>
            </>
          )}
          {m.resolution_source && (
            <p className="muted">
              Fuente de resolución: {String(m.resolution_source)}
            </p>
          )}
        </div>
      </div>
      {data.notes.length > 0 && <p className="muted">{data.notes.join(" ")}</p>}
      <p className="muted">Datos de Gamma, CLOB y Data API v2 de Polymarket. Informativo; no es asesoría financiera.</p>
    </section>
  );
}
