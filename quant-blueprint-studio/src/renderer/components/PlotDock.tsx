import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Area,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";
import type { CustomPlot } from "../types/blueprint";
const colors = [
  "#6cc9ac",
  "#9c9af8",
  "#f2ad65",
  "#79b4ef",
  "#ec9dd6",
  "#d5d77e",
];
export function PlotDock({
  plots,
  onInspect,
}: {
  plots: CustomPlot[];
  onInspect: (id: string) => void;
}) {
  if (!plots.length) return null;
  return (
    <section className="plot-dock" aria-label="Custom plot outputs">
      <h2>
        PLOT NODES · {plots.length} chart{plots.length === 1 ? "" : "s"}
      </h2>
      <div>
        {plots.map((plot) => (
          <article
            key={plot.id}
            style={{
              borderTopColor: plot.colors?.[plot.series[0]] ?? colors[0],
            }}
          >
            <header>
              <div className="plot-title">
                <strong>{plot.title}</strong>
                <button
                  onClick={() => onInspect(plot.id)}
                  title="Select this Plot node"
                >
                  Plot {plot.id.slice(0, 8)} ↗
                </button>
              </div>
              <small>
                {plot.origin ?? "Input"} · {plot.source} · {plot.rows.length}{" "}
                observations · {plot.series.join(" / ")}
              </small>
            </header>
            <ResponsiveContainer width="100%" height={230}>
              <ComposedChart
                data={plot.rows}
                margin={{ top: 10, right: 18, bottom: 0, left: 0 }}
              >
                <CartesianGrid stroke="#29313d" strokeDasharray="3 5" />
                <XAxis
                  dataKey={plot.x_key}
                  tick={{ fontSize: 10, fill: "#8898af" }}
                  minTickGap={50}
                />
                <YAxis
                  tick={{ fontSize: 10, fill: "#8898af" }}
                  tickFormatter={(value: number) =>
                    ["equity", "drawdown", "weights"].includes(plot.source)
                      ? `${(value * 100).toFixed(0)}%`
                      : Number(value).toPrecision(3)
                  }
                />
                <Tooltip
                  contentStyle={{
                    background: "#171f2b",
                    border: "1px solid #344054",
                    fontSize: 11,
                  }}
                />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                {plot.series.map((key, i) =>
                  plot.chart_type === "bar" ? (
                    <Bar
                      key={key}
                      dataKey={key}
                      name={plot.labels?.[key] ?? key}
                      fill={plot.colors?.[key] ?? colors[i % colors.length]}
                      isAnimationActive={false}
                    />
                  ) : plot.chart_type === "area" ? (
                    <Area
                      key={key}
                      dataKey={key}
                      name={plot.labels?.[key] ?? key}
                      stroke={plot.colors?.[key] ?? colors[i % colors.length]}
                      fill={plot.colors?.[key] ?? colors[i % colors.length]}
                      fillOpacity={0.12}
                      isAnimationActive={false}
                    />
                  ) : (
                    <Line
                      key={key}
                      dataKey={key}
                      name={plot.labels?.[key] ?? key}
                      stroke={plot.colors?.[key] ?? colors[i % colors.length]}
                      dot={false}
                      strokeWidth={1.5}
                      isAnimationActive={false}
                    />
                  ),
                )}
              </ComposedChart>
            </ResponsiveContainer>
          </article>
        ))}
      </div>
    </section>
  );
}
