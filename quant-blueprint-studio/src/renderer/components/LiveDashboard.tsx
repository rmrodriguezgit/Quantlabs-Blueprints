import { Fragment, useEffect, useState } from "react";
import { Activity, Pause, Play } from "lucide-react";
import {
  Bar,
  BarChart,
  Cell,
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { PolymarketSnapshot, StudyReport, UpDownDecision } from "../types/blueprint";

export interface LiveTick {
  time: string;
  market?: PolymarketSnapshot;
  decisions: UpDownDecision[];
}
interface Props {
  ticks: LiveTick[];
  history: StudyReport | undefined;
  live: boolean;
  running: boolean;
  refreshSeconds: number;
  nextAt: number | null;
  onToggle: () => void;
}

const pct = (v: number | null | undefined, d = 1) => (v == null ? "—" : `${(v * 100).toFixed(d)}%`);
const num = (v: number | null | undefined, d = 3) => (v == null ? "—" : v.toFixed(d));
const usd = (v: number | null | undefined) =>
  v == null
    ? "—"
    : "$" + v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: Math.abs(v) < 1 ? 4 : 2 });
const tone = (side: string | null | undefined) => (side === "UP" ? "up" : side === "DOWN" ? "down" : "");
const clock = (s: number | null | undefined) => {
  if (s == null) return "—";
  const t = Math.max(0, Math.round(s));
  return `${Math.floor(t / 60)}:${String(t % 60).padStart(2, "0")}`;
};

/** Real-time Polymarket + LAYA dashboard fed by each cycle of the live loop. */
export function LiveDashboard({ ticks, history, live, running, refreshSeconds, nextAt, onToggle }: Props) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, []);
  const last = ticks[ticks.length - 1];
  const m = last?.market;
  const d = last?.decisions[0];
  if (!m && !d) return null;
  const age = m ? (now - Date.parse(m.fetched_at)) / 1000 : 0;
  const toClose = m?.seconds_to_close != null ? m.seconds_to_close - age : null;
  const timeline = ticks.map((t) => ({
    time: t.time,
    "P(UP) mercado": t.market?.book.best_bid != null && t.market.book.best_ask != null ? (t.market.book.best_bid + t.market.book.best_ask) / 2 : t.market?.price_up ?? null,
    "P(UP) LAYA": t.decisions[0]?.probability_up ?? null,
    "Laya cruda": t.decisions[0]?.laya.P_up ?? null,
  }));
  const u = m?.underlying;
  const startMs = d?.window_start ? Date.parse(d.window_start) : null;
  const path = startMs
    ? ticks
        .filter((t) => t.market?.slug === m?.slug && t.decisions[0])
        .map((t) => ({
          t: (Date.parse(t.market!.fetched_at) - startMs) / 1000,
          LAYA: t.decisions[0].probability_up,
          mercado: t.decisions[0].p_market ?? null,
        }))
        .filter((r) => r.t >= 0)
    : [];
  const stats = (d?.bucket_stats ?? []) as Record<string, string | number>[];
  const midUp =
    m?.book.best_bid != null && m.book.best_ask != null && m.book.best_ask - m.book.best_bid <= 0.1
      ? (m.book.best_bid + m.book.best_ask) / 2
      : (m?.price_up ?? null);
  const contrib = Object.entries(d?.contributions ?? {}).map(([k, v]) => ({ señal: k, contribución: Number(v.toFixed(4)) }));
  const priceChart = history?.charts[0];
  return (
    <section className="live-dashboard">
      <header>
        <div>
          <span className={`live-dot ${live ? "on" : ""}`} />
          <strong>{m?.question ?? d?.question ?? "Polymarket"}</strong>
          <span className="muted">
            {" "}
            · {m?.slug} · ventana {m?.window ?? d?.horizon}
          </span>
        </div>
        <div className="live-controls">
          <span className="muted">
            {running ? "actualizando…" : live && nextAt ? `próximo ciclo en ${Math.max(0, Math.ceil((nextAt - now) / 1000))}s` : `ciclo ${refreshSeconds}s`} · {ticks.length} ciclos
          </span>
          <button className={live ? "live-on" : ""} onClick={onToggle}>
            {live ? <Pause size={13} /> : <Play size={13} />} {live ? "Detener ciclo" : "Ciclo en vivo"}
          </button>
        </div>
      </header>

      {u && (
        <div className="live-price">
          <div className="live-price-figures">
            <div>
              <span className="label">Precio a vencer · {u.asset}</span>
              <span className="value">{usd(u.price_to_beat)}</span>
              <span className="muted">apertura de la ventana · {u.source}</span>
            </div>
            <div>
              <span className="label">{u.asset} ahora</span>
              <span className={`value ${tone(u.leading)}`}>{usd(u.spot)}</span>
              <span className="muted">{u.spot_ts ? `hace ${Math.max(0, Math.round(now / 1000 - u.spot_ts))}s` : "sin feed en vivo"}</span>
            </div>
            <div>
              <span className="label">Diferencia</span>
              <span className={`value ${tone(u.leading)}`}>
                {u.diff == null ? "—" : `${u.diff >= 0 ? "▲ +" : "▼ "}${usd(u.diff)}`}
              </span>
              <span className="muted">
                {u.diff_pct == null ? "" : `${u.diff_pct >= 0 ? "+" : ""}${u.diff_pct.toFixed(3)}%`}
                {u.leading && ` · si cerrara ahora: ${u.leading}`}
                {u.signal && ` · z ${u.signal.z.toFixed(2)} (σ ${u.signal.sigma_pct_min.toFixed(3)}%/min)`}
              </span>
            </div>
            <div>
              <span className="label">Cierre en</span>
              <span className="value">{clock(toClose)}</span>
              <span className="muted">{m?.window} · {m?.end ? new Date(m.end).toLocaleTimeString() : ""}</span>
            </div>
          </div>
          {u.history.length > 1 && (
            <ResponsiveContainer width="100%" height={120}>
              <LineChart data={u.history.map((r) => ({ hora: new Date(r.t * 1000).toLocaleTimeString(), precio: r.p }))}>
                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff14" />
                <XAxis dataKey="hora" tick={{ fontSize: 10 }} minTickGap={50} />
                <YAxis
                  tick={{ fontSize: 10 }}
                  width={70}
                  domain={[
                    (min: number) => Math.min(min, u.price_to_beat ?? min) - 5,
                    (max: number) => Math.max(max, u.price_to_beat ?? max) + 5,
                  ]}
                  tickFormatter={(v: number) => v.toFixed(0)}
                />
                <Tooltip formatter={(v) => usd(Number(v))} />
                {u.price_to_beat != null && (
                  <ReferenceLine y={u.price_to_beat} stroke="#e0af68" strokeDasharray="5 4" label={{ value: "a vencer", fill: "#e0af68", fontSize: 10, position: "insideTopRight" }} />
                )}
                <Line type="monotone" dataKey="precio" stroke={u.leading === "DOWN" ? "#f7768e" : "#9ece6a"} dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>
      )}

      <div className="live-grid">
        <div className={`live-decision ${d?.decision === "UP" ? "up" : "down"}`}>
          <Activity size={16} />
          <span className="big">{d?.decision ?? "—"}</span>
          <span>P(UP) {pct(d?.probability_up)} · confianza {pct(d?.confidence)}</span>
          <span>{d ? (d.actionable ? `accionable (≥ ${pct(d.threshold, 0)})` : `sin señal (< ${pct(d.threshold, 0)})`) : ""}</span>
          <span className="muted">pesos: {d?.weight_source} · Laya {d?.laya.used ? pct(d.laya.P_up) : "off"}</span>
        </div>
        <div className="live-card">
          <h4>Mercado</h4>
          <dl>
            <dt>«{m?.up_outcome}» (mid libro)</dt><dd>{pct(midUp)}</dd>
            <dt>«{m?.down_outcome}»</dt><dd>{pct(midUp == null ? null : 1 - midUp)}</dd>
            <dt>Cierre en</dt><dd>{clock(toClose)}</dd>
            <dt>Estado</dt><dd>{m?.closed ? "cerrado" : m?.accepting_orders ? "abierto" : "pausado"}</dd>
          </dl>
        </div>
        <div className="live-card">
          <h4>Libro (token UP)</h4>
          <dl>
            <dt>Bid / Ask</dt><dd>{num(m?.book.best_bid)} / {num(m?.book.best_ask)}</dd>
            <dt>Spread</dt><dd>{num(m?.book.spread)}</dd>
            <dt>Prof. bid / ask</dt><dd>{Math.round(m?.book.bid_depth ?? 0)} / {Math.round(m?.book.ask_depth ?? 0)}</dd>
            <dt>Desequilibrio</dt><dd>{num(m?.book.imbalance, 2)}</dd>
          </dl>
        </div>
        <div className="live-card">
          <h4>Actividad</h4>
          <dl>
            <dt>Volumen</dt><dd>{Math.round(m?.volume ?? 0).toLocaleString()}</dd>
            <dt>Vol. 24h</dt><dd>{Math.round(m?.volume24h ?? 0).toLocaleString()}</dd>
            <dt>Liquidez</dt><dd>{Math.round(m?.liquidity ?? 0).toLocaleString()}</dd>
            <dt>Dato</dt><dd>hace {Math.round(age)}s</dd>
          </dl>
        </div>
      </div>

      <div className="live-charts">
        <div>
          <h4>Decisión LAYA vs mercado (por ciclo)</h4>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={timeline}>
              <CartesianGrid strokeDasharray="3 3" stroke="#ffffff14" />
              <XAxis dataKey="time" tick={{ fontSize: 10 }} />
              <YAxis domain={[0, 1]} tick={{ fontSize: 10 }} />
              <Tooltip />
              <ReferenceLine y={0.5} stroke="#888" strokeDasharray="4 4" />
              <Line type="monotone" dataKey="P(UP) mercado" stroke="#7aa2f7" dot={false} isAnimationActive={false} />
              <Line type="monotone" dataKey="P(UP) LAYA" stroke="#9ece6a" dot isAnimationActive={false} />
              <Line type="monotone" dataKey="Laya cruda" stroke="#e0af68" dot={false} strokeDasharray="3 3" isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div>
          <h4>Contribución por señal (log-odds)</h4>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={contrib}>
              <CartesianGrid strokeDasharray="3 3" stroke="#ffffff14" />
              <XAxis dataKey="señal" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 10 }} />
              <Tooltip />
              <ReferenceLine y={0} stroke="#888" />
              <Bar dataKey="contribución" fill="#9ece6a" isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        {priceChart && (
          <div className="wide">
            <h4>Historial CLOB · P(UP) / P(DOWN)</h4>
            <ResponsiveContainer width="100%" height={180}>
              <LineChart data={priceChart.rows.slice(-400)}>
                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff14" />
                <XAxis dataKey="date" tick={{ fontSize: 10 }} minTickGap={40} />
                <YAxis domain={[0, 1]} tick={{ fontSize: 10 }} />
                <Tooltip />
                <Line type="stepAfter" dataKey="P_up" stroke="#7aa2f7" dot={false} isAnimationActive={false} />
                <Line type="stepAfter" dataKey="P_down" stroke="#f7768e" dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {d && d.bucket != null && (
        <div className="live-learning">
          <h4>
            Por minuto de la ventana · ahora minuto {d.bucket + 1} de {d.buckets_total} · {Math.round(d.elapsed_s ?? 0)}s transcurridos
          </h4>
          <div className="live-charts">
            <div>
              <h4>P(UP) dentro de la ventana actual</h4>
              <ResponsiveContainer width="100%" height={180}>
                <LineChart data={path}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#ffffff14" />
                  <XAxis dataKey="t" type="number" domain={[0, (d.buckets_total ?? 1) * (d.bucket_s ?? 60)]} tick={{ fontSize: 10 }}
                    tickFormatter={(v: number) => `${Math.floor(v / 60)}:${String(Math.round(v % 60)).padStart(2, "0")}`} />
                  <YAxis domain={[0, 1]} tick={{ fontSize: 10 }} />
                  <Tooltip labelFormatter={(v) => `${Math.round(Number(v))}s`} />
                  <ReferenceLine y={0.5} stroke="#888" strokeDasharray="4 4" />
                  {Array.from({ length: (d.buckets_total ?? 1) - 1 }, (_, i) => (
                    <ReferenceLine key={i} x={(i + 1) * (d.bucket_s ?? 60)} stroke="#ffffff22" />
                  ))}
                  <Line type="monotone" dataKey="LAYA" stroke="#9ece6a" dot isAnimationActive={false} />
                  <Line type="monotone" dataKey="mercado" stroke="#7aa2f7" dot={false} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div>
              <h4>Ventaja frente al mercado por minuto (calibración TE)</h4>
              {stats.length ? (
                <ResponsiveContainer width="100%" height={180}>
                  <BarChart data={stats}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#ffffff14" />
                    <XAxis dataKey="minuto" tick={{ fontSize: 10 }} />
                    <YAxis tick={{ fontSize: 10 }} />
                    <Tooltip />
                    <ReferenceLine y={0} stroke="#888" />
                    <Bar dataKey="ventaja vs mercado" isAnimationActive={false}>
                      {stats.map((r) => (
                        <Cell key={String(r.minuto)} fill={Number(r.minuto) === d.bucket! + 1 ? "#e0af68" : Number(r["ventaja vs mercado"]) > 0 ? "#9ece6a" : "#f7768e"} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <p className="muted">Conecta Laya Signal → Laya Calibrate para ver la ventaja medida en cada minuto.</p>
              )}
            </div>
          </div>
          {stats.length > 0 && (
            <table className="live-log">
              <thead>
                <tr><th>Minuto</th><th>TE n</th><th>Acierto mercado</th><th>Acierto distancia</th><th>Acierto calibrada</th><th>Log-loss mercado</th><th>Log-loss calibrada</th><th>Ventaja</th><th>Online (acierto / n)</th></tr>
              </thead>
              <tbody>
                {stats.map((r) => {
                  const on = d.learning?.by_bucket?.find((b) => b.bucket + 1 === Number(r.minuto));
                  const adv = Number(r["ventaja vs mercado"]);
                  return (
                    <tr key={String(r.minuto)} className={Number(r.minuto) === d.bucket! + 1 ? "live-current" : undefined}>
                      <td>{String(r.minuto)}</td>
                      <td>{String(r["TE n"])}</td>
                      <td>{pct(Number(r["acierto mercado"]))}</td>
                      <td>{pct(Number(r["acierto distancia objetivo"]))}</td>
                      <td>{pct(Number(r["acierto calibrada"]))}</td>
                      <td>{num(Number(r["logloss mercado"]))}</td>
                      <td>{num(Number(r["logloss calibrada"]))}</td>
                      <td className={adv > 0 ? "up" : "down"}>{adv > 0 ? "+" : ""}{num(adv, 4)}</td>
                      <td>{on && on.n ? `${pct(on.hit_rate)} / ${on.n}` : "—"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>
      )}

      {d?.learning && (
        <div className="live-learning">
          <h4>
            Aprendizaje en línea · {d.learning.key} · lr {d.learning.lr}
          </h4>
          <div className="live-grid">
            <div className="live-card">
              <dl>
                <dt>Ventanas resueltas</dt><dd>{d.learning.updates}</dd>
                <dt>Pendientes</dt><dd>{d.learning.pending.length}</dd>
                <dt>Acierto LAYA ({d.learning.window})</dt><dd>{pct(d.learning.hit_rate)}</dd>
                <dt>Acierto mercado</dt><dd>{pct(d.learning.market_hit_rate)}</dd>
                <dt>Log-loss</dt><dd>{num(d.learning.logloss)}</dd>
              </dl>
              {d.learning.learned_now.length > 0 && (
                <p className="muted">
                  Este ciclo aprendió de: {d.learning.learned_now.map((e) => `${e.slug.slice(-10)} → ${e.outcome} ${e.hit ? "✓" : "✗"}`).join(", ")}
                </p>
              )}
            </div>
            <div className="live-card">
              <h4>Pesos inicio → ahora</h4>
              <dl>
                {Object.keys(d.learning.weights).map((k) => (
                  <Fragment key={k}>
                    <dt>{k}</dt>
                    <dd>
                      {num(d.learning?.initial_weights[k], 2)} → <strong>{num(d.learning?.weights[k], 3)}</strong>
                    </dd>
                  </Fragment>
                ))}
              </dl>
            </div>
            <div className="live-card" style={{ gridColumn: "span 2" }}>
              <h4>Evolución de pesos por ventana resuelta</h4>
              {d.learning.history.length ? (
                <ResponsiveContainer width="100%" height={160}>
                  <LineChart data={d.learning.history.map((e, i) => ({ n: i + 1, ...e.weights }))}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#ffffff14" />
                    <XAxis dataKey="n" tick={{ fontSize: 10 }} />
                    <YAxis tick={{ fontSize: 10 }} />
                    <Tooltip />
                    {Object.keys(d.learning.weights).map((k, i) => (
                      <Line key={k} type="monotone" dataKey={k} dot={false} isAnimationActive={false} stroke={["#7aa2f7", "#e0af68", "#9ece6a", "#bb9af7", "#f7768e"][i % 5]} />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <p className="muted">Esperando a que se resuelva la primera ventana (unos minutos tras el cierre).</p>
              )}
            </div>
          </div>
          {d.learning.history.length > 0 && (
            <table className="live-log">
              <thead>
                <tr><th>Ventana</th><th>Decidido</th><th>P(UP)</th><th>Decisión</th><th>Resultado</th><th>Acierto</th><th>Log-loss</th></tr>
              </thead>
              <tbody>
                {[...d.learning.history].reverse().slice(0, 10).map((e) => (
                  <tr key={e.slug}>
                    <td>{e.slug.slice(-24)}</td>
                    <td>{e.decided_at.slice(11, 19)}</td>
                    <td>{pct(e.p_up)}</td>
                    <td className={e.decision === "UP" ? "up" : "down"}>{e.decision}</td>
                    <td className={e.outcome === "UP" ? "up" : "down"}>{e.outcome}</td>
                    <td>{e.hit ? "✓" : "✗"}</td>
                    <td>{num(e.logloss)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      <table className="live-log">
        <thead>
          <tr><th>Hora</th><th>Mercado</th><th>Decisión</th><th>P(UP)</th><th>Conf.</th><th>Laya</th><th>Subyacente</th><th>Δ objetivo</th><th>Spread</th><th>Acción</th></tr>
        </thead>
        <tbody>
          {[...ticks].reverse().slice(0, 15).map((t, i) => {
            const x = t.decisions[0];
            return (
              <tr key={i}>
                <td>{t.time}</td>
                <td>{t.market?.slug?.slice(-24)}</td>
                <td className={x?.decision === "UP" ? "up" : "down"}>{x?.decision}</td>
                <td>{pct(x?.probability_up)}</td>
                <td>{pct(x?.confidence)}</td>
                <td>{x?.laya.used ? pct(x.laya.P_up) : "off"}</td>
                <td>{usd(t.market?.underlying?.spot)}</td>
                <td className={tone(t.market?.underlying?.leading)}>{t.market?.underlying?.diff == null ? "—" : (t.market.underlying.diff >= 0 ? "+" : "") + t.market.underlying.diff.toFixed(2)}</td>
                <td>{num(t.market?.book.spread)}</td>
                <td>{x?.actionable ? "✓" : "—"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </section>
  );
}
