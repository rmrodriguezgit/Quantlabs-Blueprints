import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  Legend,
} from "recharts";
import type { Result } from "../types/blueprint";
const colors = ["#6cc9ac", "#9c9af8", "#f2ad65"];
function Chart({
  data,
  keys,
  percent = false,
}: {
  data: Record<string, string | number>[];
  keys: string[];
  percent?: boolean;
}) {
  return (
    <ResponsiveContainer width="100%" height={173}>
      <LineChart
        data={data}
        margin={{ top: 6, right: 12, bottom: 0, left: -15 }}
      >
        <CartesianGrid
          stroke="#252d39"
          strokeDasharray="3 5"
          vertical={false}
        />
        <XAxis
          dataKey="date"
          tick={{ fontSize: 9, fill: "#778394" }}
          tickFormatter={(s) => String(s).slice(5)}
          minTickGap={45}
          axisLine={false}
          tickLine={false}
        />
        <YAxis
          tick={{ fontSize: 9, fill: "#778394" }}
          width={55}
          tickFormatter={(v: number) =>
            percent ? `${(v * 100).toFixed(0)}%` : Number(v).toPrecision(2)
          }
          axisLine={false}
          tickLine={false}
        />
        <Tooltip
          contentStyle={{
            background: "#171f2b",
            border: "1px solid #344054",
            borderRadius: 6,
            fontSize: 11,
          }}
        />
        <Legend wrapperStyle={{ fontSize: 10 }} />
        {percent && <ReferenceLine y={0} stroke="#465168" />}
        {keys.map((key, i) => (
          <Line
            key={key}
            dataKey={key}
            stroke={colors[i % colors.length]}
            dot={false}
            strokeWidth={1.6}
            isAnimationActive={false}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}
function Empty({ text }: { text: string }) {
  return (
    <div className="chart-empty">
      <span>⌁</span>
      <p>{text}</p>
    </div>
  );
}
export function PerformanceDock({
  result,
  running,
}: {
  result: Result | null;
  running: boolean;
}) {
  return (
    <section className="performance">
      <div className="dock-heading">
        <div>
          PERFORMANCE & ANALYTICS <span>OUT OF SAMPLE</span>
        </div>
        <small>
          {running
            ? "Computing pipeline…"
            : result
              ? `${result.predictions.length} test observations · ${result.history.length} epochs`
              : "Execute a blueprint to populate results"}
        </small>
      </div>
      <div className="panels">
        <article>
          <header>
            <b>01</b>
            <h3>Series decomposition</h3>
            <span>TR / VA / TE</span>
          </header>
          {result?.decomposition.length ? (
            <div className="decomp">
              {(["TR", "VA", "TE"] as const).map((key, i) => (
                <div key={key}>
                  <small style={{ color: colors[i] }}>{key}</small>
                  <ResponsiveContainer width="100%" height={46}>
                    <LineChart data={result.decomposition}>
                      <Tooltip
                        contentStyle={{
                          background: "#171f2b",
                          border: "1px solid #344054",
                          fontSize: 10,
                        }}
                      />
                      <Line
                        dataKey={key}
                        stroke={colors[i]}
                        dot={false}
                        strokeWidth={1.4}
                        isAnimationActive={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              ))}
            </div>
          ) : (
            <Empty text="Trend, seasonality and residuals" />
          )}
        </article>
        <article>
          <header>
            <b>02</b>
            <h3>Actual vs. predicted</h3>
            <span>{result?.target ?? "FORECAST"}</span>
          </header>
          {result?.predictions.length ? (
            <Chart data={result.predictions} keys={["actual", "predicted"]} />
          ) : (
            <Empty text="Unseen test-period forecasts" />
          )}
        </article>
        <article>
          <header>
            <b>03</b>
            <h3>Market backtest</h3>
            <span>−1 / 0 / +1</span>
          </header>
          {result?.backtest.length ? (
            <Chart
              data={result.backtest}
              keys={["strategy", "market"]}
              percent
            />
          ) : (
            <Empty text="Strategy and benchmark equity" />
          )}
        </article>
        <article className="metrics">
          <header>
            <b>04</b>
            <h3>Error summary</h3>
            <span>TEST SET</span>
          </header>
          <div className="metric-grid">
            {["MSE", "RMSE", "MAE", "R2"].map((key) => (
              <div key={key}>
                <small>{key === "R2" ? "R²" : key}</small>
                <strong>
                  {result?.metrics[key] != null
                    ? Number(result.metrics[key]).toPrecision(4)
                    : "—"}
                </strong>
              </div>
            ))}
          </div>
          {result?.trading && (
            <div className="trading-note">
              Return {(result.trading.total_return * 100).toFixed(2)}%{" "}
              <span>·</span> Drawdown{" "}
              {(result.trading.max_drawdown * 100).toFixed(2)}%
            </div>
          )}
        </article>
      </div>
    </section>
  );
}
