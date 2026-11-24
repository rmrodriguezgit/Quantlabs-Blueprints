import {
  createContext,
  useContext,
  useState,
  useCallback,
  useEffect,
} from "react";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  Handle,
  Position,
  addEdge,
  applyEdgeChanges,
  applyNodeChanges,
  useReactFlow,
  useNodesInitialized,
  type NodeProps,
  type Connection,
  type Edge,
  type OnNodesChange,
  type OnEdgesChange,
} from "@xyflow/react";
import {
  Database,
  SlidersHorizontal,
  Layers3,
  BrainCircuit,
  ChartNoAxesCombined,
  Settings2,
  Workflow,
  Eye,
  EyeOff,
} from "lucide-react";
import {
  studyArchitecture,
  catalog,
  createNode,
  inputHandles,
  plotPalette,
  type BlueprintNode,
  type Graph,
  type Kind,
} from "../types/blueprint";
import "@xyflow/react/dist/style.css";
import { validConnection, payloadLabel } from "../connections";
const icons = {
  dl_recurrent_compare: BrainCircuit,
  dl_tabular_forecast: BrainCircuit,
  dl_sequence_classifier: BrainCircuit,
  dl_anomaly_autoencoder: BrainCircuit,
  ml_select: SlidersHorizontal,
  ml_returns: ChartNoAxesCombined,
  ml_clean: Database,
  ml_split: Layers3,
  ml_train: BrainCircuit,
  ml_evaluate: SlidersHorizontal,
  ml_predict: BrainCircuit,
  ml_visualize: ChartNoAxesCombined,
  ae_split: Layers3,
  sup_target: SlidersHorizontal,
  sup_split: Layers3,
  sup_scale: SlidersHorizontal,
  sup_windows: Layers3,
  sup_init: Settings2,
  sup_mlp: BrainCircuit,
  sup_cnn: BrainCircuit,
  sup_rnn: BrainCircuit,
  sup_optimizer: Settings2,
  sup_regularizer: Settings2,
  sup_loss: Settings2,
  sup_hpo: BrainCircuit,
  sup_train: BrainCircuit,
  sup_select: SlidersHorizontal,
  sup_evaluate: ChartNoAxesCombined,
  rl_env: Layers3,
  rl_qnet: BrainCircuit,
  rl_replay: BrainCircuit,
  rl_target: BrainCircuit,
  rl_sync: BrainCircuit,
  rl_train: BrainCircuit,
  rl_baselines: ChartNoAxesCombined,
  rl_evaluate: ChartNoAxesCombined,
  gen_dcgan_train: BrainCircuit,
  gen_regime_compare: ChartNoAxesCombined,
  gen_windows: Layers3,
  gen_condition: SlidersHorizontal,
  gen_split: Layers3,
  gen_scale: SlidersHorizontal,
  gen_cvae: BrainCircuit,
  gen_reparam: SlidersHorizontal,
  gen_vae_train: BrainCircuit,
  gen_gan_train: BrainCircuit,
  gen_select: SlidersHorizontal,
  gen_evaluate: ChartNoAxesCombined,
  ae_standardize: SlidersHorizontal,
  ae_pca: ChartNoAxesCombined,
  ae_encoder: BrainCircuit,
  ae_decoder: BrainCircuit,
  ae_noise: SlidersHorizontal,
  ae_train: BrainCircuit,
  ae_select: SlidersHorizontal,
  ae_ridge: ChartNoAxesCombined,
  ae_test: ChartNoAxesCombined,
  yfinance_data: Database,
  polymarket_dashboard: ChartNoAxesCombined,
  polymarket_source: Database,
  laya_updown: BrainCircuit,
  laya_signal: BrainCircuit,
  laya_calibrate: SlidersHorizontal,
  laya_feedback: ChartNoAxesCombined,
  data_quality: Database,
  quant_features: ChartNoAxesCombined,
  ml_model: BrainCircuit,
  rl_agent: BrainCircuit,
  resample: Layers3,
  select_columns: SlidersHorizontal,
  coordinate: Layers3,
  center: SlidersHorizontal,
  scale_columns: SlidersHorizontal,
  curve_data: Layers3,
  curve_fit: BrainCircuit,
  study: BrainCircuit,
  source: Database,
  features: SlidersHorizontal,
  prepare: Layers3,
  model: BrainCircuit,
  backtest: ChartNoAxesCombined,
  config: Settings2,
  network_input: Layers3,
  layer: BrainCircuit,
  network_output: Layers3,
  merge: SlidersHorizontal,
  plot: ChartNoAxesCombined,
  portfolio_data: Database,
  portfolio: ChartNoAxesCombined,
  risk: SlidersHorizontal,
};
function nodeSummary(data: BlueprintNode["data"]): string {
  const p = {
    ...Object.fromEntries(
      catalog[data.kind].fields.map((field) => [field.key, field.value]),
    ),
    ...data.params,
  };
  switch (data.kind) {
    case "dl_recurrent_compare":
      return `${[p.use_lstm !== false && "LSTM", p.use_gru !== false && "GRU", p.use_lstm_bi !== false && "LSTM bi", p.use_gru_bi === true && "GRU bi"].filter(Boolean).join(" · ")} · ${p.lookback}×${String(p.features).split(",").filter(Boolean).length || "auto"} → ${p.target} (bps) · vs Ridge`;
    case "dl_tabular_forecast":
      return `In ${String(p.architecture).toUpperCase()} ${p.input_units} → ${p.layers || `${p.hidden_layers}×${p.units} ${p.activation}`} → Out ${p.horizon} · ${p.optimizer}`;
    case "dl_sequence_classifier":
      return `In GRU ${p.input_units} → ${p.layers || `${p.hidden_layers}×${p.units} ${p.activation}`} → Out ${p.classes} clases · ${p.optimizer}`;
    case "dl_anomaly_autoencoder":
      return `In ${p.input_units} → ${p.layers || `${p.hidden_layers}×${p.units}`} → z${p.latent} → espejo → Out · ${p.optimizer}`;
    case "ml_select":
      return `${p.mode} · ${String(p.columns || "todas").slice(0, 32)}`;
    case "ml_returns":
      return `${p.return_type} · ${p.columns} · k=${p.periods}`;
    case "ml_clean":
      return `faltantes ${p.missing} · atípicos ${p.outliers}`;
    case "ml_split":
      return `${p.target || "auto"} ${p.target_mode} · ${p.method} · TR ${p.train_ratio} / VA ${p.val_ratio} · ${p.scaling}`;
    case "ml_train":
      return `${p.algorithm} / ${p.algorithm_cls} · ajuste TR · selección VA`;
    case "ml_evaluate":
      return [p.eval_tr && "TR", p.eval_va && "VA", p.eval_te && "TE"].filter(Boolean).join(" · ") + " · línea base";
    case "ml_predict":
      return `${p.scope} → ${p.output_column}`;
    case "ml_visualize":
      return [p.series && "serie", p.scatter && "dispersión", p.residuals && "residuos", p.importance && "importancia"].filter(Boolean).join(" · ");
    case "gen_windows":
      return `w = ret[i:i+${p.horizon}] · paso ${p.stride}`;
    case "gen_condition":
      return `${[p.use_log_vol && "log σ", p.use_momentum && "momentum"].filter(Boolean).join(" + ")} de ret[i-${p.lookback}:i]`;
    case "gen_split":
      return p.cut_mode === "dates" ? `VAL ${p.val_start} · TEST ${p.test_start} · purga cortes` : `TR ${p.train_ratio} / VA ${p.val_ratio} · purga cortes`;
    case "gen_scale":
      return `media/σ TR · retornos ${p.return_scale} · condición`;
    case "gen_cvae":
      return `cat(x, c) → ${p.widths} → μ, log σ² (ZDIM ${p.latent_dim})`;
    case "gen_reparam":
      return `z = μ + σ ⊙ ε · ${p.mode} · T=${p.temperature}`;
    case "gen_vae_train":
      return `MSE + ${p.beta}·KL · ${p.epochs} épocas · ${p.device}`;
    case "gen_gan_train":
      return `G/D ${p.widths} · ${p.epochs} épocas · d_steps ${p.d_steps}${p.amp ? " · AMP" : ""}`;
    case "gen_select":
      return `ruido fijo (semilla ${p.noise_seed}) · score VA`;
    case "gen_evaluate":
      return `${p.scenarios_per_condition} escenarios/ventana · VaR ${p.var_level}${p.bootstrap_baseline ? " · vs bootstrap" : ""}`;
    case "sup_target":
      return `${String(p.features).split(",").filter(Boolean).length} rasgos · ${p.task === "direction" ? "dirección t+1" : "y_bps intradía t+1"}`;
    case "sup_split":
      return p.cut_mode === "dates" ? `VAL ${p.val_start} · TEST ${p.test_start}` : `TR ${p.train_ratio} / VA ${p.val_ratio}`;
    case "sup_scale":
      return `${p.method} · fit TR${p.scale_target ? " · y_scaler" : ""}`;
    case "sup_windows":
      return `(batch, ${p.lookback}, rasgos)`;
    case "sup_init":
      return `${p.scheme} · semilla ${p.seed}`;
    case "sup_mlp":
      return `${p.name || "MLP"} · ${p.widths} ${p.activation}${Number(p.dropout) ? ` · dropout ${p.dropout}` : ""}${p.include_as_candidate === false ? " · plantilla" : ""}`;
    case "sup_cnn":
      return `Conv1d ${p.channels} · k ${p.kernel} · pool → 1`;
    case "sup_rnn":
      return `${p.bidirectional ? "Bi" : ""}${p.cell} ${p.hidden} · estado final → 1`;
    case "sup_optimizer":
      return `${p.optimizer} · lr ${p.lr}`;
    case "sup_regularizer":
      return `${p.kind}${["Baseline", "BatchNorm"].includes(String(p.kind)) ? "" : ` ${p.value}`}`;
    case "sup_loss":
      return `pérdida ${p.loss}`;
    case "sup_hpo":
      return `${p.trials} pruebas · folds ${p.folds} · lr 10^[${p.log_lr_min}, ${p.log_lr_max}]`;
    case "sup_train":
      return `${p.epochs} épocas · lote ${p.batch_size} · paciencia ${p.patience} · ${p.device}`;
    case "sup_select":
      return "menor pérdida en VA";
    case "sup_evaluate":
      return `TE · costes ${p.cost_bps} bps${p.ridge ? " · vs Ridge" : ""}`;
    case "rl_env":
      return `rasgos + posición · acciones −1/0/1 · ${p.cost_bps} bps`;
    case "rl_qnet":
      return `online + target · ${p.widths} → 3 Q`;
    case "rl_replay":
      return `buffer ${p.capacity} · lote ${p.batch_size} · ε ${p.eps_start}→${p.eps_end}`;
    case "rl_target":
      return `${p.double ? "Double DQN" : "DQN"} · γ ${p.gamma}`;
    case "rl_sync":
      return `copiar cada ${p.sync_every} · checkpoint ${p.checkpoint_metric}`;
    case "rl_train":
      return `${p.epochs} épocas · Huber · clip ${p.grad_clip}`;
    case "rl_baselines":
      return [p.q_learning && "Q-learning", p.always_long && "largo", p.cash && "efectivo"].filter(Boolean).join(" · ");
    case "rl_evaluate":
      return "greedy en TE · sin aprender";
    case "gen_dcgan_train":
      return `z ${p.latent_dim} · emb ${p.embedding_dim} · ${p.channels} canales · ${p.epochs} épocas`;
    case "gen_regime_compare":
      return `${p.scenarios} escenarios × 3 regímenes · semilla ${p.noise_seed}`;
    case "ae_split":
      return `${String(p.features).split(",").filter(Boolean).length} indicadores · y_bps ${p.horizon} filas · TR ${p.train_ratio} / VA ${p.val_ratio}`;
    case "ae_standardize":
      return "fit TR → transform TR/VA/TE";
    case "ae_pca":
      return `PCA(n_components=${p.components})`;
    case "ae_encoder":
      return `Lineal(d, ${p.hidden}) → ${p.activation} → Lineal(${p.hidden}, ${p.latent})`;
    case "ae_decoder":
      return `Lineal(k, ${p.hidden}) → ${p.activation} → Lineal(${p.hidden}, d)`;
    case "ae_noise":
      return `inp = xb + ${p.sigma}·randn_like(xb)`;
    case "ae_train":
      return `${[p.train_ae && "AE", p.train_dae && "DAE"].filter(Boolean).join(" + ")} · ${p.epochs} épocas · paciencia ${p.patience} · ${p.device}`;
    case "ae_select":
      return `menor MSE VA${p.include_pca ? " · incluye PCA" : ""}`;
    case "ae_ridge":
      return `Ridge(alpha=${p.alpha}) · z → y_bps`;
    case "ae_test":
      return "MSE reconstrucción · RMSE/R² y_bps en TE";
    case "yfinance_data":
      return `${String(p.ticker).toUpperCase()} · ${p.interval} · ${p.range_mode === "start_end" ? `${p.start} → ${p.end || "hoy"}` : p.period}${p.auto_adjust === false ? " · sin ajustar" : ""}`;
    case "polymarket_dashboard":
      return `${p.category} · ${p.sort} · ${p.pages}×50 eventos (sin entrada) · o el mercado conectado`;
    case "polymarket_source":
      return `${String(p.market).slice(0, 28)} · ${p.window}`;
    case "laya_updown":
      return `${p.window} · ${p.weight_source} · Laya ${p.use_laya ? "on" : "off"}`;
    case "laya_signal":
      return `${p.checkpoint} · ${p.max_rows} días · 3 preguntas`;
    case "laya_calibrate":
      return `TR ${p.train_ratio} · VA ${p.val_ratio} · lr ${p.lr} · ${p.epochs} épocas`;
    case "laya_feedback":
      return `lr ${p.lr} · conf ≥ ${p.min_confidence} · ${p.positions}`;
    case "data_quality":
      return `${p.duplicates} duplicates · ${p.missing} missing`;
    case "quant_features":
      return `${p.price_column} · ${p.window} bars · ${p.periods_per_year}/year`;
    case "ml_model":
      return `${p.algorithm} · ${p.window_mode} · validation selection`;
    case "rl_agent":
      return `Double DQN / Q-learning · ${p.epochs} episodes`;
    case "select_columns":
      return String(p.columns || "Close");
    case "resample":
      return `${p.frequency} · ${p.aggregation} · ${p.columns}`;
    case "coordinate":
      return `${p.output_column}: ${p.start}…${p.stop}`;
    case "center":
      return `${p.prefix}${p.columns} · fit ${Number(p.fit_fraction) * 100}%`;
    case "scale_columns":
      return `${p.method} · ${p.columns} · fit ${Number(p.fit_fraction) * 100}%`;
    case "curve_data":
      return "Daily Close → monthly last → artificial coordinate";
    case "curve_fit":
      return `${p.name} · ${p.evaluation} · ${p.epochs} epochs`;
    case "study":
      return studyArchitecture(p);
    case "source":
      return `${p.ticker} · ${p.years}Y · ${p.source}`;
    case "features":
      return (
        ["rsi", "ema", "sma", "macd", "vwap", "bb"]
          .filter((k) => p[`use_${k}`] !== false)
          .map((k) => k.toUpperCase())
          .join(" · ") || "Returns + volatility"
      );
    case "prepare":
      return `${String(p.feature_columns ?? "").trim() ? String(p.feature_columns).split(",").length : "auto"} features · ${p.lookback} bars`;
    case "model":
      return `${p.architecture} · ${p.epochs} epochs`;
    case "network_input":
      return `${p.input_features} features · ${p.mode}`;
    case "layer":
      return p.layer_type === "Dense"
        ? `Dense ${p.units} → ${p.activation}`
        : p.layer_type === "Dropout"
          ? `Dropout ${p.rate}`
          : String(p.layer_type);
    case "network_output":
      return `${p.output_units} output · Linear`;
    case "merge":
      return "A + B · date-aligned features";
    case "portfolio_data":
      return `${p.tickers} · ${p.source}`;
    case "portfolio":
      return `Max Sharpe + min variance · ${Number(p.train_fraction) * 100}% estimation`;
    case "risk":
      return `VaR ${Number(p.confidence) * 100}% · RF ${Number(p.risk_free_rate) * 100}%`;
    case "plot":
      return `${String(p.plot_source)} · ${String(p.plot_columns || "default series")}`;
    case "backtest":
      return `Threshold ${p.threshold}`;
    default:
      return "Overrides connected parameters";
  }
}
interface Pin {
  node: string;
  handle: string;
  direction: "source" | "target";
}
const PinContext = createContext<(pin: Pin) => void>(() => {});
/** Nodes whose result is shown as a view (report, plot, dashboard…) and can be switched off with the eye. */
export const VIEW_KINDS: Kind[] = [
  "yfinance_data", "plot", "study", "ml_model", "rl_agent", "curve_fit", "portfolio", "risk", "model", "backtest",
  "laya_signal", "laya_calibrate", "laya_feedback", "laya_updown", "polymarket_source", "polymarket_dashboard",
  "dl_recurrent_compare", "dl_tabular_forecast", "dl_sequence_classifier", "dl_anomaly_autoencoder", "ml_select", "ml_returns", "ml_clean", "ml_split", "ml_train", "ml_evaluate", "ml_predict", "ml_visualize",
  "gen_windows", "gen_condition", "gen_split", "gen_scale", "gen_cvae", "gen_reparam", "gen_vae_train", "gen_gan_train", "gen_select", "gen_evaluate",
  "ae_split", "ae_standardize", "ae_pca", "ae_encoder", "ae_decoder", "ae_noise", "ae_train", "ae_select", "ae_ridge", "ae_test",
  "sup_target", "sup_split", "sup_scale", "sup_windows", "sup_init", "sup_mlp", "sup_cnn", "sup_rnn", "sup_optimizer", "sup_regularizer", "sup_loss", "sup_hpo", "sup_train", "sup_select", "sup_evaluate", "rl_env", "rl_qnet", "rl_replay", "rl_target", "rl_sync", "rl_train", "rl_baselines", "rl_evaluate", "gen_dcgan_train", "gen_regime_compare",
];
/** Hidden views by node id and the toggle, provided by App (kept outside the graph so results survive). */
export const ViewContext = createContext<{ hidden: Set<string>; toggle: (id: string) => void }>({ hidden: new Set(), toggle: () => {} });
/** Worst validation severity and count per node id, provided by App. */
export const IssueContext = createContext<
  Record<string, { severity: "error" | "warning"; count: number; first: string }>
>({});
function visibleInputs(node: BlueprintNode): Record<string, string> {
  return Object.fromEntries(
    Object.entries(inputHandles(node.data.kind)).filter(
      ([id]) =>
        node.data.kind !== "study" ||
        node.data.params.topic === "cdcgan" ||
        !id.startsWith("asset_"),
    ),
  );
}
function VisualNode({ id: nodeId, data, selected }: NodeProps<BlueprintNode>) {
  const openPin = useContext(PinContext);
  const issue = useContext(IssueContext)[nodeId];
  const views = useContext(ViewContext);
  const hidden = views.hidden.has(nodeId);
  const def = catalog[data.kind];
  const Icon = icons[data.kind];
  return (
    <div
      className={`blueprint-node ${selected ? "selected" : ""} ${issue ? `has-${issue.severity}` : ""} ${hidden ? "view-hidden" : ""}`}
      style={
        {
          "--node-color":
            data.kind === "plot"
              ? (plotPalette[String(data.params.color)] ?? def.color)
              : def.color,
        } as React.CSSProperties
      }
    >
      {issue && (
        <span className={`node-issue ${issue.severity}`} title={issue.first}>
          {issue.count}
        </span>
      )}
      <div className="node-category">
        {def.category}
        <span className="node-actions">
          {VIEW_KINDS.includes(data.kind) && (
            <button
              type="button"
              className="node-eye nodrag nopan"
              aria-pressed={!hidden}
              aria-label={hidden ? "Mostrar la vista de esta herramienta" : "Ocultar la vista de esta herramienta"}
              title={hidden ? "Vista oculta · clic para mostrarla" : "Vista visible · clic para ocultarla"}
              onMouseDown={(event) => event.stopPropagation()}
              onClick={(event) => {
                event.stopPropagation();
                views.toggle(nodeId);
              }}
            >
              {hidden ? <EyeOff size={14} /> : <Eye size={14} />}
            </button>
          )}
          <span>•••</span>
        </span>
      </div>
      <div className="node-title">
        <Icon size={19} />
        <strong>
          {data.kind === "plot"
            ? String(data.params.title || data.label)
            : data.label}
        </strong>
      </div>
      <p>{def.subtitle}</p>
      <div className="node-summary">{nodeSummary(data)}</div>
      <div className="node-pins">
        <span>{def.input ?? "source"}</span>
        <span title={payloadLabel(data.kind)}>
          {def.output === "tensor"
            ? payloadLabel(data.kind)
            : (def.output ?? "report")}
        </span>
      </div>
      {Object.entries(inputHandles(data.kind))
        .filter(
          ([id]) =>
            data.kind !== "study" ||
            data.params.topic === "cdcgan" ||
            !id.startsWith("asset_"),
        )
        .map(([id, type], i) => (
          <div key={id}>
            <Handle
              aria-label={`Entrada ${id}: ${type}`}
              title={`${id}: ${type} · Doble clic para conectar o desconectar`}
              tabIndex={0}
              onDoubleClick={(event) => {
                event.stopPropagation();
                openPin({ node: nodeId, handle: id, direction: "target" });
              }}
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  event.stopPropagation();
                  event.preventDefault();
                  openPin({ node: nodeId, handle: id, direction: "target" });
                }
              }}
              type="target"
              position={
                id === "config" || id === "benchmark"
                  ? Position.Top
                  : id === "network" &&
                      ["model", "curve_fit"].includes(data.kind)
                    ? Position.Bottom
                    : Position.Left
              }
              id={id}
              className={type}
              style={
                id === "config" ||
                id === "benchmark" ||
                (id === "network" && ["model", "curve_fit"].includes(data.kind))
                  ? {}
                  : {
                      top:
                        ["merge", "plot", "risk"].includes(data.kind) ||
                        (data.kind === "study" &&
                          data.params.topic === "cdcgan")
                          ? `${44 + i * 15}%`
                          : "82%",
                    }
              }
            />
            {id === "benchmark" && (
              <span className="config-label">benchmark</span>
            )}
            {id === "config" && <span className="config-label">config</span>}
            {id === "network" && ["model", "curve_fit"].includes(data.kind) && (
              <span className="network-label">network</span>
            )}
            {id === "frame_b" && <span className="second-input-label">B</span>}
            {id.startsWith("asset_") && (
              <span
                style={{
                  position: "absolute",
                  left: 10,
                  top: `${44 + i * 15}%`,
                  fontSize: 9,
                }}
              >
                {id}
              </span>
            )}
          </div>
        ))}
      {def.output && (
        <Handle
          aria-label={`Salida ${def.output}: ${payloadLabel(data.kind)}`}
          title={`${def.output} · Doble clic para ver alternativas de conexión`}
          tabIndex={0}
          onDoubleClick={(event) => {
            event.stopPropagation();
            openPin({ node: nodeId, handle: def.output!, direction: "source" });
          }}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              event.stopPropagation();
              event.preventDefault();
              openPin({
                node: nodeId,
                handle: def.output!,
                direction: "source",
              });
            }
          }}
          type="source"
          position={Position.Right}
          id={def.output}
          className={def.output}
          style={{ top: "82%" }}
        />
      )}
    </div>
  );
}
const nodeTypes = { blueprint: VisualNode };
export function Palette({
  onAdd,
  onBuilder,
  disabled,
}: {
  onAdd: (kind: Kind) => void;
  onBuilder: () => void;
  disabled: boolean;
}) {
  const [query, setQuery] = useState("");
  const [area, setArea] = useState<string | null>(null);
  const areas = [...new Set(Object.values(catalog).map((d) => d.category))].sort();
  const areaName = (category: string) => category.replace(/^\d+\s*\/\s*/, "");
  const browsing = !query && area === null;
  return (
    <aside className="palette">
      <div className="aside-heading">
        BIBLIOTECA <span>{Object.keys(catalog).length}</span>
      </div>
      <input
        aria-label="Buscar herramientas"
        placeholder="Buscar en todas las áreas…"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />
      {browsing ? (
        <>
          <p className="muted">Elige un área para ver sus herramientas.</p>
          <nav className="palette-areas" aria-label="Áreas">
            {areas.map((category) => {
              const kinds = (Object.keys(catalog) as Kind[]).filter((k) => catalog[k].category === category);
              const Icon = icons[kinds[0]];
              return (
                <button key={category} onClick={() => setArea(category)} style={{ color: catalog[kinds[0]].color }}>
                  <Icon size={16} />
                  <span>{areaName(category)}</span>
                  <small>{kinds.length}</small>
                  <b>›</b>
                </button>
              );
            })}
          </nav>
        </>
      ) : (
        !query && (
          <button className="palette-back" onClick={() => setArea(null)}>
            ‹ Áreas · <strong>{areaName(area ?? "")}</strong>
          </button>
        )
      )}
      {!browsing && (Object.keys(catalog) as Kind[])
        .filter((kind) => query || catalog[kind].category === area)
        .filter((kind) =>
          `${catalog[kind].label} ${catalog[kind].category} ${catalog[kind].subtitle}`
            .toLowerCase()
            .includes(query.toLowerCase()),
        )
        .sort((a, b) => catalog[a].category.localeCompare(catalog[b].category))
        .map((kind) => {
          const def = catalog[kind];
          const Icon = icons[kind];
          return (
            <div
              key={kind}
              draggable={!disabled}
              role="button"
              tabIndex={disabled ? -1 : 0}
              aria-label={`Add ${def.label}`}
              onClick={() => {
                if (!disabled) onAdd(kind);
              }}
              onKeyDown={(e) => {
                if (!disabled && (e.key === "Enter" || e.key === " ")) {
                  e.preventDefault();
                  onAdd(kind);
                }
              }}
              onDragStart={(e) => {
                e.dataTransfer.setData("application/blueprint", kind);
                e.dataTransfer.effectAllowed = "move";
              }}
              className="palette-item"
            >
              <span style={{ color: def.color }}>
                <Icon size={18} />
              </span>
              <div>
                <strong>{def.label}</strong>
                <small>
                  {def.category} · {def.subtitle}
                </small>
              </div>
              <b>⋮</b>
            </div>
          );
        })}
      <div className="pin-legend">
        <span>
          <i className="portfolio" />
          Portfolio / risk bundle
        </span>
        <span>
          <i className="network" />
          Network architecture
        </span>
        <span>
          <i className="frame" />
          DataFrame
        </span>
        <span>
          <i className="tensor" />
          Tensor bundle
        </span>
        <span>
          <i className="config" />
          Configuration
        </span>
      </div>
      <div className="palette-footer">
        <button className="builder-card" disabled={disabled} onClick={onBuilder} title="Describe el análisis y el Cerebro Constructor arma un flujo validado">
          <span className="builder-emblem">
            <Workflow size={17} />
          </span>
          <span>
            <strong>Cerebro Constructor</strong>
            <small>Construye un flujo</small>
          </span>
        </button>
        <small className="muted">
          {Object.keys(catalog).length} herramientas · {areas.length} áreas
        </small>
      </div>
    </aside>
  );
}
interface Props {
  graph: Graph;
  setGraph: (update: (graph: Graph) => Graph) => void;
  onSelect: (id: string | null) => void;
  disabled: boolean;
}
export function BlueprintCanvas({
  graph,
  setGraph,
  onSelect,
  disabled,
}: Props) {
  const flow = useReactFlow<BlueprintNode, Edge>();
  const initialized = useNodesInitialized();
  const topology = graph.nodes.map((n) => n.id).join();
  useEffect(() => {
    if (initialized) void flow.fitView({ padding: 0.16, duration: 200 });
  }, [initialized, topology, flow]);
  const onNodesChange: OnNodesChange<BlueprintNode> = useCallback(
    (changes) =>
      setGraph((g) => ({ ...g, nodes: applyNodeChanges(changes, g.nodes) })),
    [setGraph],
  );
  const onEdgesChange: OnEdgesChange = useCallback(
    (changes) =>
      setGraph((g) => ({ ...g, edges: applyEdgeChanges(changes, g.edges) })),
    [setGraph],
  );
  const [pin, setPin] = useState<Pin | null>(null);
  const [edgeId, setEdgeId] = useState<string | null>(null);
  const valid = (connection: Connection | Edge) =>
    validConnection(graph, connection);
  const connect = (connection: Connection) => {
    if (disabled) return;
    setGraph((g) =>
      validConnection(g, connection)
        ? { ...g, edges: addEdge({ ...connection, animated: true }, g.edges) }
        : g,
    );
    setPin(null);
  };
  const disconnect = (ids: string[]) => {
    if (disabled) return;
    setGraph((g) => ({
      ...g,
      edges: g.edges.filter((e) => !ids.includes(e.id)),
    }));
    setEdgeId(null);
  };
  const pinNode = graph.nodes.find((n) => n.id === pin?.node);
  const links = pin
    ? graph.edges.filter((e) =>
        pin.direction === "source"
          ? e.source === pin.node && e.sourceHandle === pin.handle
          : e.target === pin.node && e.targetHandle === pin.handle,
      )
    : [];
  const options: { label: string; connection: Connection }[] = [];
  if (pin && pinNode)
    for (const node of graph.nodes) {
      const handles =
        pin.direction === "source"
          ? visibleInputs(node)
          : catalog[node.data.kind].output
            ? {
                [catalog[node.data.kind].output!]:
                  catalog[node.data.kind].output!,
              }
            : {};
      for (const handle of Object.keys(handles)) {
        const connection: Connection =
          pin.direction === "source"
            ? {
                source: pin.node,
                sourceHandle: pin.handle,
                target: node.id,
                targetHandle: handle,
              }
            : {
                source: node.id,
                sourceHandle: handle,
                target: pin.node,
                targetHandle: pin.handle,
              };
        if (valid(connection))
          options.push({
            label: `${node.data.label} · ${handle} (${node.id.slice(0, 6)})`,
            connection,
          });
      }
    }
  const selectedEdge = graph.edges.find((e) => e.id === edgeId);
  useEffect(() => {
    if (disabled) {
      setPin(null);
      setEdgeId(null);
    }
  }, [disabled]);
  useEffect(() => {
    const close = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setPin(null);
        setEdgeId(null);
      }
    };
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, []);
  return (
    <div
      className="canvas"
      onDragOver={(e) => {
        e.preventDefault();
        e.dataTransfer.dropEffect = "move";
      }}
      onDrop={(e) => {
        e.preventDefault();
        if (disabled) return;
        const kind = e.dataTransfer.getData("application/blueprint");
        if (!Object.hasOwn(catalog, kind)) return;
        const position = flow.screenToFlowPosition({
          x: e.clientX,
          y: e.clientY,
        });
        setGraph((g) => ({
          ...g,
          nodes: [
            ...g.nodes,
            createNode(kind as Kind, g.nodes.length, position),
          ],
        }));
      }}
    >
      <div className="canvas-caption">
        <span className="live-dot" /> BLUEPRINT EDITOR <span>/</span> Research
        pipeline
      </div>
      {graph.nodes.length === 0 && (
        <div className="empty-blueprint">
          <h3>New blueprint</h3>
          <p>Click or drag a node from the library to begin.</p>
        </div>
      )}
      <PinContext.Provider
        value={(p) => {
          if (!disabled) {
            setPin(p);
            setEdgeId(null);
          }
        }}
      >
        <ReactFlow<BlueprintNode>
          nodes={graph.nodes}
          edges={graph.edges}
          nodeTypes={nodeTypes}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={connect}
          isValidConnection={valid}
          onNodeClick={(_, node) => onSelect(node.id)}
          onEdgeClick={(_, edge) => {
            setPin(null);
            setEdgeId(edge.id);
          }}
          onEdgeDoubleClick={(_, edge) => {
            setPin(null);
            setEdgeId(edge.id);
          }}
          onPaneClick={() => {
            onSelect(null);
            setPin(null);
            setEdgeId(null);
          }}
          zoomOnDoubleClick={false}
          nodesDraggable={!disabled}
          nodesConnectable={!disabled}
          deleteKeyCode={disabled ? null : ["Backspace", "Delete"]}
          fitView
          fitViewOptions={{ padding: 0.14 }}
          minZoom={0.25}
          maxZoom={1.7}
          colorMode="dark"
        >
          <Background gap={22} size={1} color="#29313d" />
          <Controls showInteractive={false} />
          <MiniMap
            nodeColor={(n) => catalog[(n as BlueprintNode).data.kind].color}
            maskColor="#0d1118bb"
            pannable
            zoomable
          />
        </ReactFlow>
      </PinContext.Provider>
      {pin && pinNode && !disabled && (
        <section
          className="connection-menu nodrag nowheel"
          role="dialog"
          aria-label="Alternativas de conexión"
        >
          <header>
            <strong>
              {pinNode.data.label} · {pin.handle}
            </strong>
            <button onClick={() => setPin(null)} aria-label="Cerrar conexiones">
              ×
            </button>
          </header>
          <h4>Conexiones actuales</h4>
          {links.length ? (
            links.map((edge) => (
              <div key={edge.id}>
                <span>
                  {
                    graph.nodes.find(
                      (n) =>
                        n.id ===
                        (pin.direction === "source"
                          ? edge.target
                          : edge.source),
                    )?.data.label
                  }{" "}
                  ·{" "}
                  {pin.direction === "source"
                    ? edge.targetHandle
                    : edge.sourceHandle}
                </span>
                <button onClick={() => disconnect([edge.id])}>
                  Desconectar
                </button>
              </div>
            ))
          ) : (
            <p>Este conector no tiene enlaces.</p>
          )}
          {links.length > 1 && (
            <button onClick={() => disconnect(links.map((e) => e.id))}>
              Desconectar todos los enlaces del conector
            </button>
          )}
          <button
            disabled={
              !graph.edges.some(
                (e) => e.source === pin.node || e.target === pin.node,
              )
            }
            onClick={() =>
              disconnect(
                graph.edges
                  .filter((e) => e.source === pin.node || e.target === pin.node)
                  .map((e) => e.id),
              )
            }
          >
            Desconectar nodo completo
          </button>
          <h4>Conectar a un nodo existente</h4>
          {options.length ? (
            options.map((option) => (
              <button
                key={JSON.stringify(option.connection)}
                onClick={() => connect(option.connection)}
              >
                {option.label}
              </button>
            ))
          ) : (
            <p>
              No hay entradas o salidas compatibles libres. Añade un nodo o
              desconecta un enlace.
            </p>
          )}
          {pin.direction === "source" && (
            <>
              <h4>Añadir y conectar un nodo</h4>
              {(Object.keys(catalog) as Kind[]).flatMap((kind) => {
                const node = createNode(kind, graph.nodes.length, {
                  x: pinNode.position.x + 360,
                  y: pinNode.position.y + links.length * 100,
                });
                return Object.keys(visibleInputs(node))
                  .filter((handle) =>
                    validConnection(
                      { ...graph, nodes: [...graph.nodes, node] },
                      {
                        source: pin.node,
                        sourceHandle: pin.handle,
                        target: node.id,
                        targetHandle: handle,
                      },
                    ),
                  )
                  .map((handle) => (
                    <button
                      key={`${kind}-${handle}`}
                      onClick={() => {
                        setGraph((g) => {
                          const next = { ...g, nodes: [...g.nodes, node] };
                          const connection = {
                            source: pin.node,
                            sourceHandle: pin.handle,
                            target: node.id,
                            targetHandle: handle,
                          };
                          return validConnection(next, connection)
                            ? {
                                ...next,
                                edges: addEdge(
                                  { ...connection, animated: true },
                                  g.edges,
                                ),
                              }
                            : g;
                        });
                        setPin(null);
                        onSelect(node.id);
                      }}
                    >
                      + {catalog[kind].label} · {handle}
                    </button>
                  ));
              })}
            </>
          )}
        </section>
      )}
      {selectedEdge && !disabled && (
        <div className="connection-menu edge-menu">
          <span>
            {graph.nodes.find((n) => n.id === selectedEdge.source)?.data.label}{" "}
            →{" "}
            {graph.nodes.find((n) => n.id === selectedEdge.target)?.data.label}
          </span>
          <button onClick={() => disconnect([selectedEdge.id])}>
            Desconectar enlace
          </button>
          <button onClick={() => setEdgeId(null)}>Cerrar</button>
        </div>
      )}
      <div className="canvas-foot">
        {graph.nodes.length} nodes <span>·</span> {graph.edges.length}{" "}
        connections <span>·</span> DAG execution
      </div>
    </div>
  );
}
