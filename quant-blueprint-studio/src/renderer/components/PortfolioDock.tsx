import type {
  PortfolioReport,
  RiskReport,
  MetricRow,
} from "../types/blueprint";
const friendly: Record<string, string> = {
  equal_weight: "Equal weight",
  max_sharpe: "Maximum Sharpe",
  min_variance: "Minimum variance",
  custom: "Custom allocation",
};
const percent = new Set([
  "roi",
  "cagr",
  "var",
  "cvar",
  "alpha",
  "volatility",
  "max_drawdown",
]);
const columns: { key: keyof MetricRow; label: string }[] = [
  { key: "roi", label: "ROI" },
  { key: "cagr", label: "CAGR" },
  { key: "var", label: "VaR" },
  { key: "cvar", label: "CVaR" },
  { key: "sharpe", label: "Sharpe" },
  { key: "sortino", label: "Sortino" },
  { key: "alpha", label: "OLS α / yr" },
  { key: "beta", label: "OLS β" },
  { key: "volatility", label: "Volatility / yr" },
  { key: "max_drawdown", label: "Max drawdown" },
  { key: "ols_r2", label: "OLS R²" },
];
function format(value: string | number | null, key: string): string {
  if (value === null || typeof value !== "number" || !Number.isFinite(value))
    return "—";
  return percent.has(key) ? `${(value * 100).toFixed(2)}%` : value.toFixed(3);
}
function Metrics({ report }: { report: RiskReport }) {
  return (
    <>
      <p className="risk-conventions">
        {report.scope} · {report.start} → {report.end} · benchmark:{" "}
        {report.benchmark ?? "none (alpha/beta unavailable)"} · RF{" "}
        {(report.risk_free_rate * 100).toFixed(2)}% / year ·{" "}
        {report.periods_per_year} periods/year
      </p>
      <p className="risk-conventions">
        Historical VaR / CVaR {(report.confidence * 100).toFixed(1)}% · horizon{" "}
        {report.var_horizon} · positive value = loss · — = undefined ratio or
        unavailable OLS benchmark
      </p>
      <div className="metrics-table-scroll">
        <table>
          <thead>
            <tr>
              <th>Asset / portfolio</th>
              {columns.map((c) => (
                <th key={c.key}>{c.label}</th>
              ))}
              <th>N</th>
            </tr>
          </thead>
          <tbody>
            {report.metrics.map((row) => (
              <tr
                key={row.name}
                className={friendly[row.name] ? "portfolio-metric-row" : ""}
              >
                <th>{friendly[row.name] ?? row.name}</th>
                {columns.map((c) => (
                  <td key={c.key}>{format(row[c.key], c.key)}</td>
                ))}
                <td>{row.observations}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
export function PortfolioDock({
  portfolios,
  reports,
}: {
  portfolios: PortfolioReport[];
  reports: RiskReport[];
}) {
  if (!portfolios.length && !reports.length) return null;
  return (
    <section
      className="portfolio-dock"
      aria-label="Portfolio and risk analytics"
    >
      <h2>PORTFOLIO & RISK ANALYTICS</h2>
      {portfolios.map((report) => (
        <article key={report.id}>
          <header>
            <h3>{report.title}</h3>
            <span>{report.source.toUpperCase()} · HOLDOUT EVALUATION</span>
          </header>
          <div className="portfolio-split">
            <strong>
              Estimation: {report.estimation.start} → {report.estimation.end} (
              {report.estimation.observations} returns)
            </strong>
            <span>
              Long-only · sum 100% · Markowitz cap{" "}
              {(report.estimation.max_weight * 100).toFixed(0)}% ·{" "}
              {report.estimation.covariance} covariance
            </span>
            <span>
              {report.rebalance === "daily"
                ? "Daily rebalancing"
                : "Buy and hold"}{" "}
              · costs {report.cost_bps_per_traded_notional} bps per traded
              notional, including entry/exit; asset rows gross, portfolio rows
              net
            </span>
          </div>
          <Metrics report={report} />
          <h4>Allocation weights · estimated before the holdout</h4>
          <div className="metrics-table-scroll">
            <table className="weights-table">
              <thead>
                <tr>
                  <th>Asset</th>
                  {Object.keys(report.weights).map((name) => (
                    <th key={name}>{friendly[name] ?? name}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {report.assets.map((asset) => (
                  <tr key={asset}>
                    <th>{asset}</th>
                    {Object.entries(report.weights).map(([name, weights]) => (
                      <td key={name}>{(weights[asset] * 100).toFixed(2)}%</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </article>
      ))}
      {reports.map((report) => (
        <article key={report.id}>
          <header>
            <h3>{report.title}</h3>
            <span>RISK METRICS</span>
          </header>
          <Metrics report={report} />
        </article>
      ))}
    </section>
  );
}
