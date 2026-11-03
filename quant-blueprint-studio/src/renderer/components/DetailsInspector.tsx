import { Settings2, X } from "lucide-react";
import {
  plotPalette,
  catalog,
  type BlueprintNode,
  type Param,
} from "../types/blueprint";
interface Props {
  node: BlueprintNode | undefined;
  onChange: (key: string, value: Param) => void;
  onDelete: () => void;
  disabled: boolean;
  availableFeatures: string[];
  onDuplicate: () => void;
  networkConnected?: boolean;
}
export function DetailsInspector({
  node,
  onChange,
  onDelete,
  disabled,
  availableFeatures,
  onDuplicate,
  networkConnected = false,
}: Props) {
  return (
    <aside className="inspector">
      <div className="aside-heading">
        DETAILS INSPECTOR <Settings2 size={14} />
      </div>
      {node ? (
        <>
          <div className="inspector-title">
            <span
              style={{
                background:
                  node.data.kind === "plot"
                    ? (plotPalette[String(node.data.params.color)] ??
                      catalog[node.data.kind].color)
                    : catalog[node.data.kind].color,
              }}
            />
            {node.data.kind === "plot"
              ? String(node.data.params.title || node.data.label)
              : node.data.label}
          </div>
          <p className="muted">{catalog[node.data.kind].subtitle}</p>
          {["center", "scale_columns"].includes(node.data.kind) && (
            <p className="inspector-hint">
              Añade columnas derivadas y conserva las originales. fit_fraction=1
              es descriptivo. En predicción temporal usa sólo filas TR; Series
              Preparation rechaza columnas ajustadas con filas futuras. Su
              escalador sigue activo: evita normalizar dos veces. Para datos
              temporales usa preferentemente el escalado integrado de Series
              Preparation.
            </p>
          )}
          {node.data.kind === "coordinate" && (
            <p className="inspector-hint">
              Coordenada artificial según el orden de filas; no es una variable
              económica ni un pronóstico.
            </p>
          )}
          {node.data.kind === "network_input" && (
            <p className="muted">
              last: N features from the last bar. flatten: lookback × N inputs.
            </p>
          )}
          {node.data.kind === "curve_fit" && (
            <p className="inspector-hint">
              Optional purple network pin: connect Network Input → Layers →
              Network Output. Connected layers replace widths/activation
              controls. Input size is 1 for dense or the polynomial degree for
              polynomial_net. OLS uses no network.
            </p>
          )}
          {node.data.kind === "network_output" && (
            <p className="muted">
              Connect the purple output to the trainer’s purple network pin.
            </p>
          )}
          {node.data.kind === "features" && (
            <p className="inspector-hint">
              VWAP requires reported volume. For Yahoo FX symbols such as
              EURUSD=X (zero volume), turn VWAP off.
            </p>
          )}
          {node.data.kind === "plot" && (
            <div className="inspector-hint">
              Connect one input: orange for OHLCV/indicators; blue from
              Preparation for TR/VA/TE, Model for predictions/loss, or Backtest
              for equity/signals. Yellow portfolio pins support asset_prices,
              equity, cumulative_log, drawdown and weights. Choose a source and
              comma-separated series. For notebook points + smooth curve:
              scatter_line, X=coordinate, point series=actual. For losses: chart
              index 2, X=epoch, Y=log (positive values only).{" "}
              {availableFeatures.length > 0 && (
                <p>Available columns: {availableFeatures.join(", ")}</p>
              )}
            </div>
          )}
          {node.data.kind === "portfolio_data" && (
            <p className="inspector-hint">
              Adjusted daily closes are joined on common dates. All assets must
              use the same currency; no FX conversion is applied.
            </p>
          )}
          {node.data.kind === "portfolio" && (
            <p className="inspector-hint">
              Long-only, weights sum to 100%. Fit max Sharpe/min variance on
              estimation dates, then compare assets and portfolios on holdout
              dates. Optional custom weights follow ticker order and must sum to
              1.
            </p>
          )}
          {node.data.kind === "risk" && (
            <p className="inspector-hint">
              Connect one main input (frame, model/backtest or portfolio).
              Optional benchmark: orange Close-price frame at the top pin.
              VaR/CVaR use one-period historical losses; ratios and volatility
              are annualized.
            </p>
          )}
          <fieldset disabled={disabled}>
            {catalog[node.data.kind].fields.map((field) => (
              <div className="inspector-field" key={field.key}>
                <label htmlFor={`param-${field.key}`}>{field.label}</label>
                {field.key === "feature_columns" ? (
                  <div className="feature-selector">
                    <div className="feature-actions">
                      <button
                        type="button"
                        onClick={() => onChange(field.key, "")}
                      >
                        All
                      </button>
                      <button
                        type="button"
                        onClick={() =>
                          onChange(
                            field.key,
                            availableFeatures.slice(0, 16).join(","),
                          )
                        }
                      >
                        First 16
                      </button>
                    </div>
                    <small>
                      {String(node.data.params.feature_columns ?? "").trim()
                        ? String(node.data.params.feature_columns)
                            .split(",")
                            .filter(Boolean).length
                        : availableFeatures.length}{" "}
                      input features selected
                    </small>
                    {availableFeatures.length === 0 && (
                      <p>Connect a feature source first.</p>
                    )}
                    {availableFeatures.map((name) => {
                      const raw = String(
                        node.data.params.feature_columns ?? "",
                      ).trim();
                      const selected = raw
                        ? raw.split(",").map((s) => s.trim())
                        : availableFeatures;
                      return (
                        <label key={name}>
                          <input
                            type="checkbox"
                            checked={selected.includes(name)}
                            onChange={(e) => {
                              const next = e.target.checked
                                ? [...selected, name]
                                : selected.filter((n) => n !== name);
                              if (next.length)
                                onChange(field.key, next.join(","));
                            }}
                          />
                          {name}
                        </label>
                      );
                    })}
                  </div>
                ) : field.key === "color" ? (
                  <div className="palette-swatches">
                    {["Auto", ...Object.keys(plotPalette)].map((color) => (
                      <button
                        type="button"
                        key={color}
                        title={color}
                        aria-label={`Color ${color}`}
                        aria-pressed={
                          String(node.data.params.color ?? "Auto") === color
                        }
                        onClick={() => onChange("color", color)}
                      >
                        <i
                          style={{
                            background:
                              plotPalette[color] ??
                              "linear-gradient(90deg,#6cc9ac,#9c9af8)",
                          }}
                        />
                        {color}
                      </button>
                    ))}
                  </div>
                ) : field.options ? (
                  <select
                    disabled={
                      networkConnected &&
                      node.data.kind === "curve_fit" &&
                      field.key === "activation"
                    }
                    id={`param-${field.key}`}
                    value={String(node.data.params[field.key] ?? field.value)}
                    onChange={(e) => onChange(field.key, e.target.value)}
                  >
                    {field.options.map((option) => (
                      <option key={option}>{option}</option>
                    ))}
                  </select>
                ) : typeof field.value === "boolean" ? (
                  <input
                    id={`param-${field.key}`}
                    type="checkbox"
                    checked={Boolean(
                      node.data.params[field.key] ?? field.value,
                    )}
                    onChange={(e) => onChange(field.key, e.target.checked)}
                  />
                ) : (
                  <input
                    id={`param-${field.key}`}
                    type={typeof field.value === "number" ? "number" : "text"}
                    value={String(node.data.params[field.key] ?? field.value)}
                    disabled={
                      (field.key === "input_features" &&
                        node.data.params.auto_features !== false) ||
                      (networkConnected &&
                        node.data.kind === "curve_fit" &&
                        field.key === "widths")
                    }
                    min={field.min}
                    max={field.max}
                    step={field.step ?? 1}
                    onChange={(e) => {
                      if (typeof field.value === "number") {
                        const value = Number(e.target.value);
                        if (Number.isFinite(value)) onChange(field.key, value);
                      } else onChange(field.key, e.target.value);
                    }}
                  />
                )}
              </div>
            ))}
          </fieldset>
          <div className="inspector-hint">
            {node.data.kind === "plot"
              ? "Title, colors and legend names update immediately. Source, type and series apply on the next execution."
              : "Parameters are applied on the next execution. A connected config node overrides matching fields."}
          </div>
          <button disabled={disabled} onClick={onDuplicate}>
            Duplicate node
          </button>
          <button
            className="delete-button"
            disabled={disabled}
            onClick={onDelete}
          >
            <X size={14} /> Remove node
          </button>
        </>
      ) : (
        <div className="inspector-empty">
          <Settings2 size={30} />
          <h3>Inspect a node</h3>
          <p>Select a node to edit its parameters and architecture.</p>
        </div>
      )}
    </aside>
  );
}
