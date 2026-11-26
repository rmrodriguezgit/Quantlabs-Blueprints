import { Settings2, X } from "lucide-react";
import {
  plotPalette,
  catalog,
  type BlueprintNode,
  type Param,
} from "../types/blueprint";
import { PolymarketPicker, OnlineReset } from "./PolymarketPicker";
/** Laya nodes run in Polymarket mode after a Polymarket Market node unless «mode» says otherwise. */
function fieldMode(node: BlueprintNode, polymarketUpstream: boolean): "daily" | "polymarket" {
  const mode = String(node.data.params.mode ?? "auto");
  if (mode === "daily" || mode === "polymarket") return mode;
  return polymarketUpstream ? "polymarket" : "daily";
}
interface Props {
  node: BlueprintNode | undefined;
  onChange: (key: string, value: Param) => void;
  onDelete: () => void;
  disabled: boolean;
  availableFeatures: string[];
  onDuplicate: () => void;
  networkConnected?: boolean;
  frameConnected?: boolean;
  onMarketResolved?: (market: { question: string; slug: string; window: string | null }) => void;
  /** A Polymarket Market node feeds this one (directly or further upstream). */
  polymarketUpstream?: boolean;
}
export function DetailsInspector({
  node,
  onChange,
  onDelete,
  disabled,
  availableFeatures,
  onDuplicate,
  networkConnected = false,
  frameConnected = false,
  onMarketResolved,
  polymarketUpstream = false,
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
          {node.data.label !== catalog[node.data.kind].label && (
            <p className="muted">
              Herramienta: {catalog[node.data.kind].label}
            </p>
          )}
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
          {node.data.kind === "ml_model" && (
            <p className="inspector-hint">
              Compare entrena los mismos candidatos sólo con TR, selecciona por
              MSE VA y evalúa TE una vez. Se usan los datos escalados de Series
              Preparation. La salida sirve para Plot, Backtest y Risk Metrics.
            </p>
          )}
          {node.data.kind === "rl_agent" && (
            <p className="inspector-hint">
              Acciones Short/Cash/Long. Replay sólo TR; VA elige checkpoint; TE
              usa política greedy sin aprender. Compara Double DQN, Q-learning,
              largo y efectivo. El informe se conecta a Plot.
            </p>
          )}
          {node.data.kind === "polymarket_source" && (
            <>
              <PolymarketPicker
                market={String(node.data.params.market ?? catalog.polymarket_source.fields[0].value)}
                upOutcome={String(node.data.params.up_outcome ?? "auto")}
                disabled={disabled}
                onChange={onChange}
                onResolved={onMarketResolved}
              />
              <p className="inspector-hint">
                Lee cualquier mercado binario de Polymarket (Gamma + CLOB). Un
                alias como btc-updown-15m apunta a la ventana activa. Close =
                P(UP) implícita; el libro sólo se escribe en la última fila.
              </p>
            </>
          )}
          {node.data.kind === "yfinance_data" && (
            <p className="inspector-hint">
              Descarga un ticker de Yahoo Finance con las opciones de yfinance.download.
              Límites de Yahoo para intradía: 1m ≈ 7 días (sólo los últimos 30), 2m–90m
              hasta 60 días, 60m/1h hasta 730 días; diario o mayor, sin límite. La fecha
              de fin es exclusiva. El intradía se expresa en la zona horaria elegida.
            </p>
          )}
          {node.data.kind === "polymarket_dashboard" && (
            <p className="inspector-hint">
              Conectado a Polymarket Market muestra el dashboard de ese mercado:
              outcomes, cuenta atrás, subyacente vs precio a vencer, KPIs, historial,
              profundidad y niveles del libro, flujo de operaciones hacia cada
              outcome, últimas operaciones, mercados del mismo evento y concentración
              de tenedores (sólo agregados). Sin conexión, descarga los eventos
              activos de Polymarket (Gamma API), los organiza y
              los presenta como dashboard: KPIs, volumen por categoría, mercados más
              negociados, mayores movimientos 24 h, más disputados, cierran en 24 h,
              más líquidos y spreads amplios. Con una categoría elegida, el desglose
              es por subcategoría. Usa ● En vivo para refrescarlo en ciclo. Categoría,
              orden, páginas y filas sólo aplican sin conexión.
            </p>
          )}
          {node.data.kind === "laya_updown" && (
            <OnlineReset
              params={node.data.params}
              disabled={disabled}
            />
          )}
          {node.data.kind === "laya_updown" && (
            <p className="inspector-hint">
              P(UP) = σ(sesgo + Σ wᵢ·zᵢ) con señales mercado, momentum, Laya y
              desequilibrio del libro. Ajusta los pesos aquí o usa «calibrated»
              con Laya Calibrate aguas arriba. Cada cambio queda en el informe
              (traza por señal y JSON de decisión).
            </p>
          )}
          {node.data.kind === "laya_signal" && (
            <p className="inspector-hint">
              Cada día envía a Laya un estado JSON con indicadores conocidos al
              cierre (RSI, MACD, retornos, volatilidad, posición Bollinger) y
              tres preguntas tipadas. Laya devuelve probabilidades calibradas
              sin generar texto. Edita las preguntas para ver cómo cambian las
              respuestas. La primera ejecución descarga el modelo; después hay
              caché. Tras Polymarket Market (modo auto) pregunta una vez por cada
              ventana resuelta de la serie, con su resultado real, y por la
              ventana actual.
            </p>
          )}
          {node.data.kind === "laya_calibrate" && (
            <p className="inspector-hint">
              Ajuste de pesos: aprende w en p = σ(w·x) con x = [sesgo,
              log(P_up/P_down), logit P(higher), riesgo]. Gradiente sólo con
              TR, VA elige la época, TE se mide una vez. Compara Brier, logloss
              y ECE de Laya cruda vs calibrada. Con Polymarket ajusta sesgo y
              pesos de mercado, momentum y Laya sobre las ventanas resueltas y
              los compara con seguir al mercado.
            </p>
          )}
          {node.data.kind === "laya_feedback" && (
            <p className="inspector-hint">
              Retroalimentación diaria: predice con w, espera el retorno real de
              t+1 y corrige w ← w − lr·(p − y)·x. Parte de los pesos de Laya
              Calibrate si está conectado. Lista los errores de mayor pérdida.
            </p>
          )}
          {node.data.kind === "data_quality" && (
            <p className="inspector-hint">
              Forward fill sólo usa pasado y tiene límite. No se usa backfill ni
              interpolación con observaciones futuras. Los huecos iniciales
              restantes se eliminan.
            </p>
          )}
          {node.data.kind === "quant_features" && (
            <p className="inspector-hint">
              Ventanas móviles causales. Sharpe es indefinido si la volatilidad
              es cero; no se sustituye por cero. Selecciona columnas y aplica
              Data Quality explícitamente antes de entrenar. Usa 12 períodos/año
              para datos mensuales.
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
            <div className="inspector-hint">
              <strong>Se calculan siempre</strong>
              <p>
                LogReturn = log(Close / Close anterior). Volatility = desviación
                móvil de LogReturn (sin anualizar). CumulativeLogReturn = suma
                de LogReturn tras el calentamiento.
              </p>
              <p>
                Estas columnas no necesitan casilla de activación. Se
                seleccionan como entradas en Series Preparation o Learning
                Experiment.
              </p>
              {!frameConnected && (
                <p role="alert">
                  Falta entrada de datos: conecta Market Data → entrada naranja
                  frame. Un nodo desconectado también se intenta ejecutar.
                </p>
              )}
              <details>
                <summary>Columnas de salida previstas</summary>
                <p>{availableFeatures.join(", ")}</p>
              </details>
              <p>
                VWAP necesita volumen reportado. Para EURUSD=X sin volumen,
                desactiva VWAP y retíralo de las columnas seleccionadas por el
                entrenador.
              </p>
            </div>
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
            {node.data.kind.startsWith("laya_") && catalog[node.data.kind].fields.some((f) => f.mode) && (
              <p className="inspector-hint">
                Modo {fieldMode(node, polymarketUpstream) === "polymarket" ? "Polymarket (ventanas resueltas)" : "diario (acciones / yfinance)"}:
                sólo se muestran los parámetros que usa este modo.
              </p>
            )}
            {catalog[node.data.kind].fields
              .filter((field) => !field.mode || field.mode === fieldMode(node, polymarketUpstream))
              .filter((field) => {
                if (!field.showIf) return true;
                const own = node.data.params[field.showIf.key] ?? catalog[node.data.kind].fields.find((f) => f.key === field.showIf!.key)?.value;
                return own === field.showIf.value;
              })
              .map((field) => (
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
