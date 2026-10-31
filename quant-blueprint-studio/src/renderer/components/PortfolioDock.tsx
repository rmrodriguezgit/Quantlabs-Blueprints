import { useState } from "react";
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
const columns = [
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
] as const;
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
export function portfolioMetrics(report: PortfolioReport): MetricRow[] {
  return (
    report.portfolio_metrics ??
    report.metrics.filter((row) => Object.hasOwn(report.weights, row.name))
  );
}
function PortfolioSummary({ report }: { report: PortfolioReport }) {
  const rows = portfolioMetrics(report);
  const [choice, setChoice] = useState("custom");
  const current =
    rows.find((row) => row.name === choice) ??
    rows.find((row) => row.name === "equal_weight") ??
    rows[0];
  if (!current)
    return (
      <p role="alert">
        No hay indicadores de cartera en este resultado. Ejecuta el nodo
        Portfolio / Markowitz.
      </p>
    );
  const labels: Record<string, string> = {
    equal_weight: "Pesos iguales",
    custom: "Portafolio personalizado",
    max_sharpe: "Máximo Sharpe",
    min_variance: "Mínima varianza",
  };
  const names: Record<string, string> = {
    roi: "ROI · retorno total",
    cagr: "CAGR · anual",
    var: `VaR ${(report.confidence * 100).toFixed(1)}%`,
    cvar: `CVaR ${(report.confidence * 100).toFixed(1)}%`,
    sharpe: "Sharpe · anualizado",
    sortino: "Sortino · anualizado",
    alpha: "Alfa OLS · anual",
    beta: "Beta OLS",
    volatility: "Volatilidad · anual",
    max_drawdown: "Drawdown máximo",
    ols_r2: "R² de OLS",
  };
  const units: Record<string, string> = {
    roi: "Retorno de toda la cartera, neto de costes",
    cagr: `Geométrico · ${report.periods_per_year} periodos/año`,
    var: `Pérdida histórica · ${report.var_horizon}`,
    cvar: `Pérdida media de cola · ${report.var_horizon}`,
    sharpe: "Exceso sobre tasa libre de riesgo / volatilidad",
    sortino: "Exceso / desviación negativa",
    alpha: `Exceso ajustado a ${report.benchmark ?? "benchmark no conectado"}`,
    beta: `Sensibilidad a ${report.benchmark ?? "benchmark no conectado"}`,
    volatility: `Desviación estándar × √${report.periods_per_year}`,
    max_drawdown: "Desde el máximo del patrimonio",
    ols_r2: "Ajuste de la regresión frente al benchmark",
  };
  return (
    <section
      className="portfolio-summary"
      aria-label="Indicadores del portafolio completo"
      data-portfolio-strategy={current.name}
    >
      <div className="portfolio-summary-heading">
        <div>
          <h4>Indicadores del portafolio completo</h4>
          <p>
            {labels[current.name] ?? current.name} · {current.observations}{" "}
            observaciones · {report.start} → {report.end}
          </p>
        </div>
        <label>
          Cartera evaluada
          <select
            aria-label="Cartera evaluada"
            value={current.name}
            onChange={(event) => setChoice(event.target.value)}
          >
            {rows.map((row) => (
              <option key={row.name} value={row.name}>
                {labels[row.name] ?? row.name}
              </option>
            ))}
          </select>
        </label>
      </div>
      <p className="risk-conventions">
        Calculados sobre los retornos de la cartera completa, después del
        rebalanceo y los costes; no son un promedio de indicadores de los
        activos. Benchmark: {report.benchmark ?? "ninguno"} · RF{" "}
        {(report.risk_free_rate * 100).toFixed(2)}% anual.
      </p>
      <div className="portfolio-metric-cards">
        {columns.map((column) => (
          <div
            key={column.key}
            data-portfolio-metric={column.key}
            className="portfolio-metric-card"
          >
            <span>{names[column.key] ?? column.label}</span>
            <strong>{format(current[column.key], column.key)}</strong>
            <small>
              {current[column.key] === null
                ? (current.metric_notes?.[column.key] ??
                  "No definido para estos datos.")
                : units[column.key]}
            </small>
          </div>
        ))}
      </div>
      <div className="portfolio-selected-weights">
        <strong>Pesos de esta cartera</strong>
        {Object.entries(report.weights[current.name] ?? {}).map(
          ([asset, weight]) => (
            <span key={asset}>
              {asset} <b>{(weight * 100).toFixed(2)}%</b>
            </span>
          ),
        )}
        <span className="weight-total">
          Total{" "}
          {(
            Object.values(report.weights[current.name] ?? {}).reduce(
              (sum, w) => sum + w,
              0,
            ) * 100
          ).toFixed(2)}
          %
        </span>
      </div>
      {report.benchmark_metrics && (
        <p className="risk-conventions">
          Referencia {report.benchmark}: ROI{" "}
          {format(report.benchmark_metrics.roi, "roi")} · CAGR{" "}
          {format(report.benchmark_metrics.cagr, "cagr")} · volatilidad anual{" "}
          {format(report.benchmark_metrics.volatility, "volatility")}. Activos y
          benchmark: retornos brutos; carteras: netos de costes configurados.
        </p>
      )}
    </section>
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
          <PortfolioSummary report={report} />
          <h4>Comparación completa · activos y carteras</h4>
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
