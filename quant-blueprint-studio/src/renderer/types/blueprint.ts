import type { Edge, Node } from "@xyflow/react";
export type Kind =
  | "dl_recurrent_compare"
  | "dl_tabular_forecast"
  | "dl_sequence_classifier"
  | "dl_anomaly_autoencoder"
  | "ml_select"
  | "ml_returns"
  | "ml_clean"
  | "ml_split"
  | "ml_train"
  | "ml_evaluate"
  | "ml_predict"
  | "ml_visualize"
  | "ae_split"
  | "sup_target"
  | "sup_split"
  | "sup_scale"
  | "sup_windows"
  | "sup_init"
  | "sup_mlp"
  | "sup_cnn"
  | "sup_rnn"
  | "sup_optimizer"
  | "sup_regularizer"
  | "sup_loss"
  | "sup_hpo"
  | "sup_train"
  | "sup_select"
  | "sup_evaluate"
  | "rl_env"
  | "rl_qnet"
  | "rl_replay"
  | "rl_target"
  | "rl_sync"
  | "rl_train"
  | "rl_baselines"
  | "rl_evaluate"
  | "gen_dcgan_train"
  | "gen_regime_compare"
  | "gen_windows"
  | "gen_condition"
  | "gen_split"
  | "gen_scale"
  | "gen_cvae"
  | "gen_reparam"
  | "gen_vae_train"
  | "gen_gan_train"
  | "gen_select"
  | "gen_evaluate"
  | "ae_standardize"
  | "ae_pca"
  | "ae_encoder"
  | "ae_decoder"
  | "ae_noise"
  | "ae_train"
  | "ae_select"
  | "ae_ridge"
  | "ae_test"
  | "yfinance_data"
  | "polymarket_dashboard"
  | "polymarket_source"
  | "laya_updown"
  | "laya_signal"
  | "laya_calibrate"
  | "laya_feedback"
  | "data_quality"
  | "quant_features"
  | "ml_model"
  | "rl_agent"
  | "resample"
  | "select_columns"
  | "coordinate"
  | "center"
  | "scale_columns"
  | "curve_data"
  | "curve_fit"
  | "study"
  | "source"
  | "features"
  | "prepare"
  | "model"
  | "backtest"
  | "config"
  | "network_input"
  | "layer"
  | "network_output"
  | "merge"
  | "plot"
  | "portfolio_data"
  | "portfolio"
  | "risk";
export type Param = string | number | boolean;
export type BlueprintData = {
  label: string;
  kind: Kind;
  params: Record<string, Param>;
};
export type BlueprintNode = Node<BlueprintData, "blueprint">;
export type Graph = { nodes: BlueprintNode[]; edges: Edge[] };
export interface Result {
  dashboards?: (PolymarketDashboardData | MarketDashboardData)[];
  decisions?: UpDownDecision[];
  studies?: StudyReport[];
  scaling_audits?: ScalingAudit[];
  decomposition: { date: string; TR: number; VA: number; TE: number }[];
  predictions: { date: string; actual: number; predicted: number }[];
  backtest: {
    date: string;
    strategy: number;
    market: number;
    signal: number;
  }[];
  metrics: Record<string, number | null>;
  history: { epoch: number; train: number; validation: number }[];
  plots?: CustomPlot[];
  portfolios?: PortfolioReport[];
  risk_reports?: RiskReport[];
  target?: string;
  features?: string[];
  input_shape?: number[];
  trading?: {
    total_return: number;
    max_drawdown: number;
    sharpe: number;
    turnover: number;
  };
}
export interface ScalingAudit {
  id: string;
  node_title: string;
  title: string;
  method: string;
  scope: string;
  fit_rows: number;
  note: string;
  fit_start?: string;
  fit_end?: string;
  inverse_before_metrics: boolean;
  columns: {
    name: string;
    mean: number;
    variance: number;
    center: number;
    scale: number;
    transformed_mean: number;
    transformed_std: number;
    constant: boolean;
  }[];
}
export interface NetworkSummary {
  total_parameters: number;
  trainable_parameters: number;
  layers: {
    name: string;
    operation: string;
    input_shape: number[];
    output_shape: number[];
    parameters: number;
  }[];
}
export type ChartType =
  "line" | "area" | "bar" | "scatter" | "scatter_line" | "histogram";
export interface StudyReport {
  id: string;
  title: string;
  topic: string;
  scope?: string;
  architecture?: NetworkSummary | null;
  market?: PolymarketSnapshot;
  selected: string;
  features: string[];
  splits: { train: number; validation: number; test: number };
  notes: string[];
  table: Record<string, string | number | null>[];
  metrics: Record<string, number | null>;
  charts: {
    title: string;
    rows: Record<string, string | number>[];
    series: string[];
    x_key: string;
    chart_type: ChartType;
  }[];
}
export interface Field {
  key: string;
  label: string;
  value: Param;
  options?: string[];
  min?: number;
  max?: number;
  step?: number;
  /** Only shown when the node runs in this mode (Laya nodes behave differently after Polymarket Market). */
  mode?: "daily" | "polymarket";
  /** Only shown when another parameter has this value (e.g. start/end only with range_mode = start_end). */
  showIf?: { key: string; value: Param };
}
export interface Definition {
  label: string;
  subtitle: string;
  category: string;
  color: string;
  input?: "frame" | "tensor" | "network" | "portfolio";
  output?: "frame" | "tensor" | "config" | "network" | "portfolio";
  fields: Field[];
}
export const catalog: Record<Kind, Definition> = {
  dl_recurrent_compare: {
    label: "DL · LSTM vs GRU (secuencias)",
    subtitle: "Secuencias batch×20×11 · LSTM / GRU / bidireccional · selección VA · vs Ridge en bps",
    category: "12 / DL TOOLS",
    color: "#ff9e64",
    input: "frame",
    output: "frame",
    fields: [
      { key: "features", label: "Indicadores (coma; vacío = todas las numéricas salvo OHLCV)", value: "LogReturn,Lag_1,Lag_2,Lag_3,Lag_4,Lag_5,RSI,MACD,MACDSignal,MACDHistogram,Volatility" },
      { key: "target", label: "Objetivo de la sesión siguiente", value: "next_intraday", options: ["next_intraday", "next_close"] },
      { key: "target_scale", label: "Escala de salida (10000 = bps)", value: 10000, min: 1, max: 100000 },
      { key: "lookback", label: "Sesiones por secuencia", value: 20, min: 2, max: 256 },
      { key: "use_lstm", label: "Comparar LSTM", value: true },
      { key: "use_gru", label: "Comparar GRU", value: true },
      { key: "use_lstm_bi", label: "Comparar LSTM bidireccional", value: true },
      { key: "use_gru_bi", label: "Comparar GRU bidireccional", value: false },
      { key: "hidden_size", label: "Capa de entrada: estado oculto LSTM/GRU", value: 32, min: 2, max: 1024 },
      { key: "rnn_layers", label: "Capas recurrentes apiladas", value: 1, min: 1, max: 4 },
      { key: "hidden_layers", label: "Capas intermedias tras el resumen (0 = como la infografía)", value: 0, min: 0, max: 8 },
      { key: "units", label: "Neuronas por capa intermedia", value: 32, min: 1, max: 2048 },
      { key: "activation", label: "Activación de las capas intermedias", value: "gelu", options: ["gelu", "relu", "tanh", "sigmoid", "swish", "leaky_relu", "elu", "identity"] },
      { key: "dropout", label: "Dropout por capa", value: 0, min: 0, max: 0.9, step: 0.05 },
      { key: "batch_norm", label: "BatchNorm en cada capa", value: false },
      { key: "layers", label: "Capas a medida (64:relu:0.1, 16:tanh)", value: "" },
      { key: "optimizer", label: "Optimizador", value: "adam", options: ["adam", "adamw", "sgd", "rmsprop", "adagrad"] },
      { key: "lr", label: "Learning rate", value: 0.001, min: 0.00001, max: 1, step: 0.0005 },
      { key: "weight_decay", label: "Weight decay (L2)", value: 0, min: 0, max: 1, step: 0.0001 },
      { key: "momentum", label: "Momentum (SGD / RMSprop)", value: 0.9, min: 0, max: 0.99, step: 0.01, showIf: { key: "optimizer", value: "sgd" } },
      { key: "epochs", label: "Épocas máximas", value: 60, min: 0, max: 1000 },
      { key: "batch_size", label: "Tamaño de lote", value: 64, min: 4, max: 4096 },
      { key: "patience", label: "Paciencia (parada temprana en VA)", value: 10, min: 1, max: 200 },
      { key: "ridge_alpha", label: "Ridge α (línea base)", value: 1, min: 0.0001, max: 1000000 },
      { key: "train_ratio", label: "Fracción TR", value: 0.7, min: 0.3, max: 0.9, step: 0.05 },
      { key: "val_ratio", label: "Fracción VA", value: 0.15, min: 0.05, max: 0.4, step: 0.05 },
      { key: "device", label: "Dispositivo", value: "auto", options: ["auto", "cpu", "mps", "cuda"] },
      { key: "seed", label: "Semilla", value: 42, min: 0, max: 99999 },
    ],
  },
  dl_tabular_forecast: {
    label: "DL · Pronóstico de serie",
    subtitle: "Herramienta independiente · MLP / 1D-CNN / LSTM · intervalo",
    category: "12 / DL TOOLS",
    color: "#ff9e64",
    input: "frame",
    output: "frame",
    fields: [
      { key: "column", label: "Columna a pronosticar", value: "LogRet_Close" },
      { key: "lookback", label: "Ventana lookback", value: 20, min: 2, max: 512 },
      { key: "architecture", label: "Capa de entrada", value: "lstm", options: ["lstm", "cnn1d", "mlp"] },
      { key: "input_units", label: "Neuronas / filtros / celdas de entrada", value: 32, min: 1, max: 2048 },
      { key: "input_activation", label: "Activación de entrada (MLP/CNN)", value: "gelu", options: ["gelu", "relu", "tanh", "sigmoid", "swish", "leaky_relu", "elu", "identity"] },
      { key: "hidden_layers", label: "Capas intermedias (nº)", value: 2, min: 0, max: 8 },
      { key: "units", label: "Neuronas por capa intermedia", value: 32, min: 1, max: 2048 },
      { key: "activation", label: "Activación de las capas intermedias", value: "gelu", options: ["gelu", "relu", "tanh", "sigmoid", "swish", "leaky_relu", "elu", "identity"] },
      { key: "dropout", label: "Dropout por capa", value: 0, min: 0, max: 0.9, step: 0.05 },
      { key: "batch_norm", label: "BatchNorm en cada capa", value: false },
      { key: "layers", label: "Capas a medida (64:relu:0.1:bn, 32:tanh; vacío = las de arriba)", value: "" },
      { key: "optimizer", label: "Optimizador", value: "adam", options: ["adam", "adamw", "sgd", "rmsprop", "adagrad"] },
      { key: "lr", label: "Learning rate", value: 0.003, min: 0.00001, max: 1, step: 0.0005 },
      { key: "weight_decay", label: "Weight decay (L2)", value: 0, min: 0, max: 1, step: 0.0001 },
      { key: "momentum", label: "Momentum (SGD / RMSprop)", value: 0.9, min: 0, max: 0.99, step: 0.01, showIf: { key: "optimizer", value: "sgd" } },
      { key: "epochs", label: "Épocas máximas (sin checkpoint)", value: 60, min: 0, max: 1000 },
      { key: "batch_size", label: "Tamaño de lote", value: 64, min: 4, max: 4096 },
      { key: "patience", label: "Paciencia (parada temprana)", value: 10, min: 1, max: 200 },
      { key: "horizon", label: "Capa de salida: pasos a predecir", value: 1, min: 1, max: 64 },
      { key: "holdout", label: "Fracción holdout final", value: 0.2, min: 0.05, max: 0.5, step: 0.05 },
      { key: "coverage", label: "Cobertura del intervalo", value: 0.9, min: 0.5, max: 0.99, step: 0.01 },
      { key: "standardize", label: "Estandarizar con el tramo de ajuste", value: true },
      { key: "device", label: "Dispositivo", value: "auto", options: ["auto", "cpu", "mps", "cuda"] },
      { key: "seed", label: "Semilla", value: 42, min: 0, max: 99999 },
      { key: "fit_on_input", label: "Ajustar con la entrada si no hay checkpoint", value: true },
    ],
  },
  dl_sequence_classifier: {
    label: "DL · Clasificador de secuencias",
    subtitle: "Herramienta independiente · GRU · probabilidades · embedding",
    category: "12 / DL TOOLS",
    color: "#ff9e64",
    input: "frame",
    output: "frame",
    fields: [
      { key: "column", label: "Columna (señal / rendimiento)", value: "LogRet_Close" },
      { key: "lookback", label: "Ventana lookback", value: 20, min: 2, max: 512 },
      { key: "input_units", label: "Capa de entrada GRU: celdas", value: 32, min: 1, max: 2048 },
      { key: "hidden_layers", label: "Capas intermedias (nº)", value: 1, min: 0, max: 8 },
      { key: "units", label: "Neuronas por capa intermedia", value: 16, min: 1, max: 2048 },
      { key: "activation", label: "Activación de las capas intermedias", value: "gelu", options: ["gelu", "relu", "tanh", "sigmoid", "swish", "leaky_relu", "elu", "identity"] },
      { key: "dropout", label: "Dropout por capa", value: 0, min: 0, max: 0.9, step: 0.05 },
      { key: "batch_norm", label: "BatchNorm en cada capa", value: false },
      { key: "layers", label: "Capas a medida (64:relu:0.1:bn, 32:tanh; vacío = las de arriba)", value: "" },
      { key: "optimizer", label: "Optimizador", value: "adam", options: ["adam", "adamw", "sgd", "rmsprop", "adagrad"] },
      { key: "lr", label: "Learning rate", value: 0.003, min: 0.00001, max: 1, step: 0.0005 },
      { key: "weight_decay", label: "Weight decay (L2)", value: 0, min: 0, max: 1, step: 0.0001 },
      { key: "momentum", label: "Momentum (SGD / RMSprop)", value: 0.9, min: 0, max: 0.99, step: 0.01, showIf: { key: "optimizer", value: "sgd" } },
      { key: "epochs", label: "Épocas máximas (sin checkpoint)", value: 60, min: 0, max: 1000 },
      { key: "batch_size", label: "Tamaño de lote", value: 64, min: 4, max: 4096 },
      { key: "patience", label: "Paciencia (parada temprana)", value: 10, min: 1, max: 200 },
      { key: "classes", label: "Capa de salida: clases", value: 2, options: ["2", "3"] },
      { key: "dead_zone", label: "Zona muerta (3 clases: |valor| ≤ plano)", value: 0.002, min: 0, max: 1, step: 0.001, showIf: { key: "classes", value: "3" } },
      { key: "class_weighting", label: "Ponderar la pérdida por frecuencia de clase", value: true },
      { key: "threshold", label: "Probabilidad mínima para emitir etiqueta", value: 0.5, min: 0.05, max: 0.95, step: 0.01 },
      { key: "holdout", label: "Fracción holdout final", value: 0.2, min: 0.05, max: 0.5, step: 0.05 },
      { key: "standardize", label: "Estandarizar con el tramo de ajuste", value: true },
      { key: "device", label: "Dispositivo", value: "auto", options: ["auto", "cpu", "mps", "cuda"] },
      { key: "seed", label: "Semilla", value: 42, min: 0, max: 99999 },
      { key: "fit_on_input", label: "Ajustar con la entrada si no hay checkpoint", value: true },
    ],
  },
  dl_anomaly_autoencoder: {
    label: "DL · Anomalías (autoencoder)",
    subtitle: "Herramienta independiente · error de reconstrucción · umbral",
    category: "12 / DL TOOLS",
    color: "#ff9e64",
    input: "frame",
    output: "frame",
    fields: [
      { key: "columns", label: "Columnas (coma; vacío = numéricas)", value: "" },
      { key: "input_units", label: "Capa de entrada: neuronas", value: 32, min: 1, max: 2048 },
      { key: "input_activation", label: "Activación de entrada", value: "gelu", options: ["gelu", "relu", "tanh", "sigmoid", "swish", "leaky_relu", "elu", "identity"] },
      { key: "hidden_layers", label: "Capas intermedias del codificador (el decodificador las refleja)", value: 1, min: 0, max: 8 },
      { key: "units", label: "Neuronas por capa intermedia", value: 16, min: 1, max: 2048 },
      { key: "activation", label: "Activación de las capas intermedias", value: "gelu", options: ["gelu", "relu", "tanh", "sigmoid", "swish", "leaky_relu", "elu", "identity"] },
      { key: "dropout", label: "Dropout por capa", value: 0, min: 0, max: 0.9, step: 0.05 },
      { key: "batch_norm", label: "BatchNorm en cada capa", value: false },
      { key: "layers", label: "Capas a medida (64:relu:0.1:bn, 32:tanh; vacío = las de arriba)", value: "" },
      { key: "optimizer", label: "Optimizador", value: "adam", options: ["adam", "adamw", "sgd", "rmsprop", "adagrad"] },
      { key: "lr", label: "Learning rate", value: 0.005, min: 0.00001, max: 1, step: 0.0005 },
      { key: "weight_decay", label: "Weight decay (L2)", value: 0, min: 0, max: 1, step: 0.0001 },
      { key: "momentum", label: "Momentum (SGD / RMSprop)", value: 0.9, min: 0, max: 0.99, step: 0.01, showIf: { key: "optimizer", value: "sgd" } },
      { key: "epochs", label: "Épocas máximas (sin checkpoint)", value: 150, min: 0, max: 1000 },
      { key: "batch_size", label: "Tamaño de lote", value: 64, min: 4, max: 4096 },
      { key: "patience", label: "Paciencia (parada temprana)", value: 10, min: 1, max: 200 },
      { key: "latent", label: "Cuello de botella (< nº de columnas)", value: 1, min: 1, max: 64 },
      { key: "error", label: "Error de reconstrucción", value: "mse", options: ["mse", "mae"] },
      { key: "threshold_mode", label: "Umbral", value: "quantile", options: ["quantile", "absolute"] },
      { key: "threshold", label: "Cuantil (0–1) o valor absoluto", value: 0.99, min: 0.0001, max: 1000, step: 0.005 },
      { key: "fit_fraction", label: "Filas iniciales para ajustar", value: 0.7, min: 0.2, max: 1, step: 0.05 },
      { key: "device", label: "Dispositivo", value: "auto", options: ["auto", "cpu", "mps", "cuda"] },
      { key: "seed", label: "Semilla", value: 42, min: 0, max: 99999 },
      { key: "fit_on_input", label: "Ajustar con la entrada si no hay checkpoint", value: true },
    ],
  },
  ml_select: {
    label: "1 · Seleccionar columnas",
    subtitle: "ML Pipeline · conservar / eliminar · renombrar",
    category: "11 / ML PIPELINE",
    color: "#9ece6a",
    input: "frame",
    output: "frame",
    fields: [
      { key: "mode", label: "Modo", value: "keep", options: ["keep", "drop"] },
      { key: "columns", label: "Columnas (coma; vacío = todas)", value: "Close,Volume" },
      { key: "always_keep", label: "Conservar siempre (coma, opcional)", value: "" },
      { key: "numeric_only", label: "Sólo columnas numéricas", value: true },
      { key: "rename", label: "Renombrar (viejo=nuevo, coma)", value: "" },
    ],
  },
  ml_returns: {
    label: "2 · Rendimientos",
    subtitle: "ML Pipeline · simples / logarítmicos · k periodos",
    category: "11 / ML PIPELINE",
    color: "#9ece6a",
    input: "frame",
    output: "frame",
    fields: [
      { key: "columns", label: "Columnas de precio (coma)", value: "Close" },
      { key: "return_type", label: "Tipo", value: "log", options: ["simple", "log", "both"] },
      { key: "periods", label: "Periodos k (Pₜ / Pₜ₋ₖ)", value: 1, min: 1, max: 252 },
      { key: "prefix_simple", label: "Prefijo simple", value: "Ret_", showIf: { key: "return_type", value: "simple" } },
      { key: "prefix_log", label: "Prefijo logarítmico", value: "LogRet_", showIf: { key: "return_type", value: "log" } },
      { key: "replace", label: "Sustituir la columna de precio", value: false },
      { key: "drop_na", label: "Eliminar las primeras filas sin rendimiento", value: true },
    ],
  },
  ml_clean: {
    label: "3 · Limpieza de datos",
    subtitle: "ML Pipeline · faltantes · atípicos · duplicados · constantes",
    category: "11 / ML PIPELINE",
    color: "#9ece6a",
    input: "frame",
    output: "frame",
    fields: [
      { key: "columns", label: "Columnas a limpiar (coma; vacío = numéricas)", value: "" },
      { key: "missing", label: "Faltantes", value: "ffill", options: ["ffill", "drop", "bfill", "ffill_bfill", "interpolate", "zero", "mean", "median", "none"] },
      { key: "outliers", label: "Atípicos", value: "none", options: ["none", "clip_zscore", "clip_quantile", "remove_zscore", "remove_iqr"] },
      { key: "z_threshold", label: "Umbral z", value: 4, min: 1, max: 20, step: 0.5 },
      { key: "q_low", label: "Cuantil inferior", value: 0.01, min: 0, max: 0.5, step: 0.005, showIf: { key: "outliers", value: "clip_quantile" } },
      { key: "q_high", label: "Cuantil superior", value: 0.99, min: 0.5, max: 1, step: 0.005, showIf: { key: "outliers", value: "clip_quantile" } },
      { key: "fit_fraction", label: "Fracción inicial para ajustar estadísticos (sin fuga)", value: 0.7, min: 0.1, max: 1, step: 0.05 },
      { key: "drop_duplicates", label: "Eliminar duplicados", value: true },
      { key: "drop_constant", label: "Eliminar columnas constantes", value: true },
      { key: "drop_remaining_na", label: "Eliminar filas con faltantes restantes", value: true },
      { key: "sort_index", label: "Ordenar por fecha", value: true },
    ],
  },
  ml_split: {
    label: "4 · Dividir TR / VA / TE",
    subtitle: "ML Pipeline · objetivo · rasgos · cronológico · escalado en TR",
    category: "11 / ML PIPELINE",
    color: "#9ece6a",
    input: "frame",
    output: "frame",
    fields: [
      { key: "target", label: "Columna objetivo (vacío = primer rendimiento)", value: "" },
      { key: "target_mode", label: "Objetivo", value: "next", options: ["next", "direction", "same"] },
      { key: "horizon", label: "Horizonte (filas hacia delante)", value: 1, min: 1, max: 252 },
      { key: "task", label: "Tarea", value: "auto", options: ["auto", "regression", "classification"] },
      { key: "features", label: "Rasgos (coma; vacío = resto de numéricas)", value: "" },
      { key: "method", label: "Método", value: "chronological", options: ["chronological", "random"] },
      { key: "train_ratio", label: "Fracción TR (entrenamiento)", value: 0.7, min: 0.3, max: 0.9, step: 0.05 },
      { key: "val_ratio", label: "Fracción VA (validación)", value: 0.15, min: 0.05, max: 0.4, step: 0.05 },
      { key: "gap", label: "Hueco entre conjuntos (filas, embargo)", value: 0, min: 0, max: 500 },
      { key: "scaling", label: "Escalado (ajustado en TR)", value: "standard", options: ["standard", "minmax", "robust", "none"] },
      { key: "keep_future", label: "Conservar filas FUT para el pronóstico", value: true },
      { key: "seed", label: "Semilla (aleatorio)", value: 42, min: 0, max: 99999, showIf: { key: "method", value: "random" } },
    ],
  },
  ml_train: {
    label: "5 · Entrenar modelo ML",
    subtitle: "ML Pipeline · scikit-learn · ajuste TR · selección VA",
    category: "11 / ML PIPELINE",
    color: "#9ece6a",
    input: "frame",
    output: "frame",
    fields: [
      { key: "algorithm", label: "Algoritmo (regresión)", value: "Compare", options: ["Compare", "Linear", "Ridge", "Lasso", "ElasticNet", "Random Forest", "Gradient Boosting", "Histogram Boosting", "SVR", "KNN", "MLP"] },
      { key: "algorithm_cls", label: "Algoritmo (clasificación)", value: "Compare", options: ["Compare", "Logistic", "Random Forest", "Gradient Boosting", "Histogram Boosting", "SVC", "KNN", "MLP"] },
      { key: "alpha", label: "Regularización α / C", value: 1, min: 0.0001, max: 1000, step: 0.1 },
      { key: "estimators", label: "Árboles / iteraciones", value: 200, min: 10, max: 1000 },
      { key: "max_depth", label: "Profundidad máxima", value: 6, min: 1, max: 32 },
      { key: "learning_rate", label: "Learning rate (boosting / MLP)", value: 0.05, min: 0.001, max: 1, step: 0.005 },
      { key: "neighbors", label: "Vecinos (KNN)", value: 10, min: 1, max: 200 },
      { key: "hidden_layers", label: "Capas ocultas MLP (coma)", value: "64,32" },
      { key: "refit_train_val", label: "Reentrenar el ganador con TR+VA", value: false },
      { key: "seed", label: "Semilla", value: 42, min: 0, max: 99999 },
    ],
  },
  ml_evaluate: {
    label: "6 · Evaluar el modelo",
    subtitle: "ML Pipeline · métricas TR/VA/TE · línea base · confusión",
    category: "11 / ML PIPELINE",
    color: "#9ece6a",
    input: "frame",
    output: "frame",
    fields: [
      { key: "eval_te", label: "Evaluar TE (prueba)", value: true },
      { key: "eval_va", label: "Evaluar VA (validación)", value: true },
      { key: "eval_tr", label: "Evaluar TR (entrenamiento, para ver sobreajuste)", value: true },
      { key: "threshold", label: "Umbral de clase 1 (clasificación)", value: 0.5, min: 0.05, max: 0.95, step: 0.01 },
    ],
  },
  ml_predict: {
    label: "7 · Realizar predicciones",
    subtitle: "ML Pipeline · TE · pronóstico FUT · intervalo · señal",
    category: "11 / ML PIPELINE",
    color: "#9ece6a",
    input: "frame",
    output: "frame",
    fields: [
      { key: "scope", label: "Filas a predecir", value: "test_future", options: ["test_future", "test", "future", "validation", "all"] },
      { key: "output_column", label: "Columna de salida", value: "Prediction" },
      { key: "threshold", label: "Umbral de clase 1 (clasificación)", value: 0.5, min: 0.05, max: 0.95, step: 0.01 },
      { key: "interval", label: "Intervalo de predicción (regresión)", value: true },
      { key: "coverage", label: "Cobertura del intervalo", value: 0.9, min: 0.5, max: 0.99, step: 0.01, showIf: { key: "interval", value: true } },
      { key: "signal", label: "Añadir señal ±1", value: true },
      { key: "rows", label: "Filas en la tabla", value: 20, min: 5, max: 200 },
    ],
  },
  ml_visualize: {
    label: "8 · Visualizar resultados",
    subtitle: "ML Pipeline · real vs predicho · residuos · importancia",
    category: "11 / ML PIPELINE",
    color: "#9ece6a",
    input: "frame",
    output: "frame",
    fields: [
      { key: "show_te", label: "Mostrar TE", value: true },
      { key: "show_fut", label: "Mostrar FUT (pronóstico)", value: true },
      { key: "show_va", label: "Mostrar VA", value: true },
      { key: "show_tr", label: "Mostrar TR", value: false },
      { key: "series", label: "Serie real vs predicho", value: true },
      { key: "scatter", label: "Dispersión real vs predicho", value: true },
      { key: "residuals", label: "Histograma de residuos", value: true },
      { key: "cumulative", label: "Estrategia acumulada ilustrativa", value: true },
      { key: "allow_short", label: "Estrategia con cortos", value: true, showIf: { key: "cumulative", value: true } },
      { key: "importance", label: "Importancia de rasgos", value: true },
      { key: "selection", label: "Comparación de modelos", value: true },
      { key: "max_points", label: "Puntos máximos", value: 1500, min: 100, max: 10000 },
    ],
  },
  gen_windows: {
    label: "01 · Crear ventanas",
    subtitle: "Generativos · w = ret[i:i+HORIZON]",
    category: "14 / GENERATIVOS",
    color: "#f7768e",
    input: "frame",
    output: "frame",
    fields: [
      { key: "return_column", label: "Columna de retornos (auto = LogReturn o log Close)", value: "auto" },
      { key: "horizon", label: "HORIZON (sesiones por ventana)", value: 16, min: 4, max: 64 },
      { key: "stride", label: "Paso entre ventanas", value: 1, min: 1, max: 64 },
    ],
  },
  gen_condition: {
    label: "02 · Condición causal",
    subtitle: "Generativos · c = log(std(ret[i-20:i]))",
    category: "14 / GENERATIVOS",
    color: "#f7768e",
    input: "frame",
    output: "frame",
    fields: [
      { key: "lookback", label: "Sesiones anteriores", value: 20, min: 5, max: 252 },
      { key: "use_log_vol", label: "Log-volatilidad", value: true },
      { key: "use_momentum", label: "Momentum (suma de retornos)", value: false },
      { key: "floor", label: "Piso de σ (evita log 0)", value: 1e-8, min: 1e-12, max: 1e-3, step: 1e-8 },
    ],
  },
  gen_split: {
    label: "03 · Separar ventanas",
    subtitle: "Generativos · purga ventanas que cruzan un corte",
    category: "14 / GENERATIVOS",
    color: "#f7768e",
    input: "frame",
    output: "frame",
    fields: [
      { key: "cut_mode", label: "Cortes", value: "ratio", options: ["ratio", "dates"] },
      { key: "train_ratio", label: "Fracción TR", value: 0.7, min: 0.3, max: 0.9, step: 0.05, showIf: { key: "cut_mode", value: "ratio" } },
      { key: "val_ratio", label: "Fracción VA", value: 0.15, min: 0.05, max: 0.4, step: 0.05, showIf: { key: "cut_mode", value: "ratio" } },
      { key: "val_start", label: "VAL_START (AAAA-MM-DD)", value: "2021-01-01", showIf: { key: "cut_mode", value: "dates" } },
      { key: "test_start", label: "TEST_START (AAAA-MM-DD)", value: "2023-01-01", showIf: { key: "cut_mode", value: "dates" } },
    ],
  },
  gen_scale: {
    label: "04 · Escalar con TR",
    subtitle: "Generativos · return_mean/std y condición sólo TR",
    category: "14 / GENERATIVOS",
    color: "#f7768e",
    input: "frame",
    output: "frame",
    fields: [{ key: "return_scale", label: "Escala de retornos", value: "global", options: ["global", "per_step"] }],
  },
  gen_cvae: {
    label: "05 · VAE condicional",
    subtitle: "Generativos · encoder(x, c) → μ, log σ²",
    category: "14 / GENERATIVOS",
    color: "#f7768e",
    input: "frame",
    output: "frame",
    fields: [
      { key: "latent_dim", label: "ZDIM", value: 4, min: 1, max: 32 },
      { key: "widths", label: "Anchos ocultos (coma)", value: "64,64" },
      { key: "logvar_clamp", label: "Clamp de log σ² (±)", value: 8, min: 1, max: 20 },
    ],
  },
  gen_reparam: {
    label: "06 · Reparametrizar",
    subtitle: "Generativos · z = μ + σ ⊙ ε",
    category: "14 / GENERATIVOS",
    color: "#f7768e",
    input: "frame",
    output: "frame",
    fields: [
      { key: "mode", label: "Muestreo en entrenamiento", value: "sample", options: ["sample", "mean"] },
      { key: "temperature", label: "Temperatura al generar", value: 1, min: 0.1, max: 3, step: 0.1 },
      { key: "seed", label: "Semilla de la demostración", value: 0, min: 0, max: 99999 },
    ],
  },
  gen_vae_train: {
    label: "07 · Reconstrucción + KL",
    subtitle: "Generativos · loss = MSE + β·KL · entrena CVAE en TR",
    category: "14 / GENERATIVOS",
    color: "#f7768e",
    input: "frame",
    output: "frame",
    fields: [
      { key: "beta", label: "β (peso KL)", value: 0.01, min: 0, max: 5, step: 0.005 },
      { key: "kl_warmup", label: "Calentamiento KL (épocas, 0 = no)", value: 0, min: 0, max: 200 },
      { key: "epochs", label: "Épocas", value: 60, min: 1, max: 1000 },
      { key: "batch_size", label: "Lote", value: 128, min: 8, max: 2048 },
      { key: "lr", label: "Learning rate", value: 0.001, min: 0.00001, max: 0.1, step: 0.0001 },
      { key: "grad_clip", label: "Clip de gradiente", value: 1, min: 0.01, max: 100, step: 0.1 },
      { key: "snapshot_every", label: "Instantánea cada (épocas)", value: 1, min: 1, max: 50 },
      { key: "device", label: "Dispositivo", value: "cpu", options: ["cpu", "auto", "cuda"] },
      { key: "seed", label: "Semilla", value: 42, min: 0, max: 99999 },
    ],
  },
  gen_gan_train: {
    label: "08 · Generador y discriminador",
    subtitle: "Generativos · cGAN alternado · BCE float32",
    category: "14 / GENERATIVOS",
    color: "#f7768e",
    input: "frame",
    output: "frame",
    fields: [
      { key: "widths", label: "Anchos G/D (coma)", value: "64,64" },
      { key: "epochs", label: "Épocas", value: 60, min: 1, max: 1000 },
      { key: "batch_size", label: "Lote", value: 128, min: 8, max: 2048 },
      { key: "lr_g", label: "Learning rate G", value: 0.0002, min: 0.00001, max: 0.01, step: 0.00005 },
      { key: "lr_d", label: "Learning rate D", value: 0.0002, min: 0.00001, max: 0.01, step: 0.00005 },
      { key: "d_steps", label: "Pasos de D por paso de G", value: 1, min: 1, max: 5 },
      { key: "label_smoothing", label: "Suavizado de etiqueta real", value: 0, min: 0, max: 0.3, step: 0.05 },
      { key: "grad_clip", label: "Clip de gradiente", value: 1, min: 0.01, max: 100, step: 0.1 },
      { key: "amp", label: "AMP float16 (sólo GPU)", value: false },
      { key: "snapshot_every", label: "Instantánea cada (épocas)", value: 1, min: 1, max: 50 },
      { key: "device", label: "Dispositivo", value: "cpu", options: ["cpu", "auto", "cuda"] },
      { key: "seed", label: "Semilla", value: 42, min: 0, max: 99999 },
    ],
  },
  gen_select: {
    label: "09 · Seleccionar sin mirar test",
    subtitle: "Generativos · ruido fijo · score en VA",
    category: "14 / GENERATIVOS",
    color: "#f7768e",
    input: "frame",
    output: "frame",
    fields: [
      { key: "noise_seed", label: "Semilla del ruido fijo", value: 43, min: 0, max: 99999 },
      { key: "w_marginal", label: "Peso W1 marginal", value: 1, min: 0, max: 10, step: 0.1 },
      { key: "w_cumulative", label: "Peso W1 acumulado", value: 1, min: 0, max: 10, step: 0.1 },
      { key: "w_vol_cluster", label: "Peso ACF de |r|", value: 1, min: 0, max: 10, step: 0.1 },
      { key: "w_tail", label: "Peso ES 5 %", value: 1, min: 0, max: 10, step: 0.1 },
    ],
  },
  gen_evaluate: {
    label: "10 · Evaluar escenarios y colas",
    subtitle: "Generativos · distribución · dependencia · VaR en TE",
    category: "14 / GENERATIVOS",
    color: "#f7768e",
    input: "frame",
    output: "frame",
    fields: [
      { key: "scenarios_per_condition", label: "Escenarios por ventana TE", value: 20, min: 1, max: 500 },
      { key: "var_level", label: "Nivel VaR / ES", value: 0.05, min: 0.01, max: 0.2, step: 0.01 },
      { key: "bootstrap_baseline", label: "Comparar con bootstrap TR", value: true },
      { key: "seed", label: "Semilla", value: 7, min: 0, max: 99999 },
    ],
  },
  sup_target: {
    label: "Definir el objetivo",
    subtitle: "U2-03 · U4-01 · y_bps = 10000·log(Close[t+1]/Open[t+1]) o dirección",
    category: "15 / SUPERVISADO U2–U7",
    color: "#7dcfff",
    input: "frame",
    output: "frame",
    fields: [
      { key: "features", label: "Rasgos al cierre t (coma)", value: "LogReturn,Volatility,RSI,EMA,SMA,VWAP,MACD,MACDSignal,BBUpper,BBMiddle,BBLower" },
      { key: "task", label: "Objetivo", value: "regression", options: ["regression", "direction"] },
    ],
  },
  sup_split: {
    label: "Separar el tiempo",
    subtitle: "U2-04 · TR / VA / TE cronológico por fecha de señal y de objetivo",
    category: "15 / SUPERVISADO U2–U7",
    color: "#7dcfff",
    input: "frame",
    output: "frame",
    fields: [
      { key: "cut_mode", label: "Cortes", value: "ratio", options: ["ratio", "dates"] },
      { key: "train_ratio", label: "Fracción TR", value: 0.65, min: 0.3, max: 0.9, step: 0.05, showIf: { key: "cut_mode", value: "ratio" } },
      { key: "val_ratio", label: "Fracción VA", value: 0.15, min: 0.05, max: 0.4, step: 0.05, showIf: { key: "cut_mode", value: "ratio" } },
      { key: "val_start", label: "VAL_START (AAAA-MM-DD)", value: "2024-01-01", showIf: { key: "cut_mode", value: "dates" } },
      { key: "test_start", label: "TEST_START (AAAA-MM-DD)", value: "2025-01-01", showIf: { key: "cut_mode", value: "dates" } },
    ],
  },
  sup_scale: {
    label: "Escalar sólo con train",
    subtitle: "U2-05 · x_scaler y y_scaler ajustados con TR",
    category: "15 / SUPERVISADO U2–U7",
    color: "#7dcfff",
    input: "frame",
    output: "frame",
    fields: [
      { key: "method", label: "Escalador de entradas", value: "standard", options: ["standard", "minmax"] },
      { key: "scale_target", label: "Escalar y_bps (regresión)", value: true },
    ],
  },
  sup_windows: {
    label: "Construir ventanas",
    subtitle: "U6-02 · U7-01 · (batch, LOOKBACK, rasgos)",
    category: "15 / SUPERVISADO U2–U7",
    color: "#7dcfff",
    input: "frame",
    output: "frame",
    fields: [{ key: "lookback", label: "LOOKBACK (sesiones)", value: 20, min: 1, max: 250 }],
  },
  sup_init: {
    label: "Fijar la inicialización",
    subtitle: "U3-02 · U4-02 · mismos pesos iniciales para todas las variantes",
    category: "15 / SUPERVISADO U2–U7",
    color: "#bb9af7",
    input: "frame",
    output: "frame",
    fields: [
      { key: "scheme", label: "Esquema", value: "kaiming_normal", options: ["kaiming_normal", "kaiming_uniform", "xavier_uniform", "pytorch_default"] },
      { key: "seed", label: "Semilla", value: 42, min: 0, max: 99999 },
    ],
  },
  sup_mlp: {
    label: "Construir la MLP",
    subtitle: "U2-06 · U6-08 · Linear → act → … → Linear(·, 1)",
    category: "15 / SUPERVISADO U2–U7",
    color: "#bb9af7",
    input: "frame",
    output: "frame",
    fields: [
      { key: "name", label: "Nombre del modelo", value: "" },
      { key: "widths", label: "Anchos ocultos (coma)", value: "64,32" },
      { key: "activation", label: "Activación", value: "ReLU", options: ["ReLU", "Tanh", "GELU", "SiLU", "LeakyReLU"] },
      { key: "dropout", label: "Dropout", value: 0, min: 0, max: 0.9, step: 0.05 },
      { key: "batchnorm", label: "BatchNorm1d", value: false },
      { key: "include_as_candidate", label: "Competir como candidato (si no, sólo plantilla)", value: true },
    ],
  },
  sup_cnn: {
    label: "Construir la CNN 1D",
    subtitle: "U6-05/06/07 · Conv1d → Conv1d → pooling global → Linear",
    category: "15 / SUPERVISADO U2–U7",
    color: "#bb9af7",
    input: "frame",
    output: "frame",
    fields: [
      { key: "name", label: "Nombre del modelo", value: "CNN_1D" },
      { key: "channels", label: "Canales (dos, coma)", value: "32,64" },
      { key: "kernel", label: "Kernel (sesiones)", value: 3, min: 1, max: 30 },
      { key: "activation", label: "Activación", value: "ReLU", options: ["ReLU", "Tanh", "GELU", "SiLU", "LeakyReLU"] },
      { key: "include_as_candidate", label: "Competir como candidato", value: true },
    ],
  },
  sup_rnn: {
    label: "Construir la red recurrente",
    subtitle: "U7-02…07 · LSTM / GRU · bidireccional · estado final → Linear",
    category: "15 / SUPERVISADO U2–U7",
    color: "#bb9af7",
    input: "frame",
    output: "frame",
    fields: [
      { key: "name", label: "Nombre del modelo", value: "" },
      { key: "cell", label: "Celda", value: "LSTM", options: ["LSTM", "GRU"] },
      { key: "hidden", label: "Unidades ocultas", value: 32, min: 2, max: 512 },
      { key: "bidirectional", label: "Bidireccional (ventana histórica)", value: false },
      { key: "include_as_candidate", label: "Competir como candidato", value: true },
    ],
  },
  sup_optimizer: {
    label: "Definir el optimizador",
    subtitle: "U3-03 · variante de la plantilla con SGD / RMSprop / Adam / AdamW",
    category: "15 / SUPERVISADO U2–U7",
    color: "#e0af68",
    input: "frame",
    output: "frame",
    fields: [
      { key: "optimizer", label: "Optimizador", value: "AdamW", options: ["SGD", "RMSprop", "Adam", "AdamW"] },
      { key: "lr", label: "Learning rate", value: 0.001, min: 0.00001, max: 1, step: 0.0005 },
      { key: "momentum", label: "Momentum (sólo SGD)", value: 0, min: 0, max: 0.99, step: 0.01 },
    ],
  },
  sup_regularizer: {
    label: "Añadir regularización",
    subtitle: "U4-03…07 · variante con Dropout / L1 / L2 / BatchNorm / Early stopping",
    category: "15 / SUPERVISADO U2–U7",
    color: "#e0af68",
    input: "frame",
    output: "frame",
    fields: [
      { key: "kind", label: "Técnica", value: "Dropout", options: ["Baseline", "Dropout", "L1", "L2", "BatchNorm", "EarlyStopping"] },
      { key: "value", label: "p · λ · weight_decay · paciencia", value: 0.3, min: 0, max: 200, step: 0.00001 },
    ],
  },
  sup_loss: {
    label: "Elegir la pérdida",
    subtitle: "U2-07 · MSE / Huber para magnitud · BCE con logits para dirección",
    category: "15 / SUPERVISADO U2–U7",
    color: "#e0af68",
    input: "frame",
    output: "frame",
    fields: [{ key: "loss", label: "Pérdida", value: "auto", options: ["auto", "MSE", "Huber", "BCE"] }],
  },
  sup_hpo: {
    label: "Búsqueda de hiperparámetros",
    subtitle: "U5-02…08 · espacio · muestreo log · folds causales · escaladores por fold",
    category: "15 / SUPERVISADO U2–U7",
    color: "#e0af68",
    input: "frame",
    output: "frame",
    fields: [
      { key: "trials", label: "Pruebas", value: 6, min: 1, max: 50 },
      { key: "width_choices", label: "Anchos posibles", value: "32,64,128" },
      { key: "depth_choices", label: "Profundidades posibles", value: "1,2,3" },
      { key: "log_lr_min", label: "log10(lr) mínimo", value: -4, min: -7, max: 0, step: 0.1 },
      { key: "log_lr_max", label: "log10(lr) máximo", value: -2.5, min: -7, max: 0, step: 0.1 },
      { key: "wd_choices", label: "weight_decay posibles", value: "0,0.0001,0.001" },
      { key: "folds", label: "Cortes de fold (fracción de TR)", value: "0.5,0.75" },
      { key: "epochs", label: "Épocas por fold", value: 12, min: 1, max: 300 },
      { key: "patience", label: "Paciencia por fold", value: 5, min: 1, max: 100 },
      { key: "batch_size", label: "Lote", value: 64, min: 2, max: 1024 },
      { key: "seed", label: "Semilla", value: 42, min: 0, max: 99999 },
    ],
  },
  sup_train: {
    label: "Entrenar con checkpoint",
    subtitle: "U2-08/09 · U3-04…08 · zero_grad → backward → clip → step · mejor VA",
    category: "15 / SUPERVISADO U2–U7",
    color: "#9ece6a",
    input: "frame",
    output: "frame",
    fields: [
      { key: "epochs", label: "Épocas (mismo presupuesto para todos)", value: 30, min: 1, max: 2000 },
      { key: "batch_size", label: "Lote", value: 64, min: 2, max: 4096 },
      { key: "lr", label: "Learning rate (si el candidato no fija uno)", value: 0.001, min: 0.00001, max: 1, step: 0.0005 },
      { key: "patience", label: "Paciencia (0 = sin early stopping)", value: 5, min: 0, max: 500 },
      { key: "restore_best", label: "Restaurar el mejor checkpoint de VA", value: true },
      { key: "grad_clip", label: "Clipping de norma", value: 1, min: 0.01, max: 100, step: 0.1 },
      { key: "amp", label: "AMP (sólo CUDA)", value: false },
      { key: "device", label: "Dispositivo", value: "cpu", options: ["cpu", "auto", "cuda", "mps"] },
      { key: "seed", label: "Semilla (orden de lotes)", value: 42, min: 0, max: 99999 },
    ],
  },
  sup_select: {
    label: "Seleccionar con validación",
    subtitle: "U3-09 · U4-09 · menor pérdida de VA, sin mirar test",
    category: "15 / SUPERVISADO U2–U7",
    color: "#9ece6a",
    input: "frame",
    output: "frame",
    fields: [],
  },
  sup_evaluate: {
    label: "Evaluar en test",
    subtitle: "U2-10 · desescalar · métricas · Ridge · resultado con costes",
    category: "15 / SUPERVISADO U2–U7",
    color: "#9ece6a",
    input: "frame",
    output: "frame",
    fields: [
      { key: "threshold", label: "Umbral de señal (bps o probabilidad)", value: 0, min: -100, max: 100, step: 0.01 },
      { key: "allow_short", label: "Permitir cortos", value: true },
      { key: "cost_bps", label: "Costes por lado (bps)", value: 5, min: 0, max: 100 },
      { key: "ridge", label: "Comparar con Ridge (regresión)", value: true },
      { key: "ridge_alpha", label: "α de Ridge", value: 1, min: 0, max: 1000, step: 0.1 },
    ],
  },
  rl_env: {
    label: "Construir el entorno",
    subtitle: "U10-02/03 · estado = rasgos + posición · acciones {-1, 0, 1} · recompensa neta",
    category: "16 / REFUERZO U10",
    color: "#ff757f",
    input: "frame",
    output: "frame",
    fields: [
      { key: "cost_bps", label: "Costes por unidad de cambio (bps)", value: 5, min: 0, max: 100 },
      { key: "reward_scale", label: "Escala de recompensa", value: 100, min: 1, max: 10000 },
    ],
  },
  rl_qnet: {
    label: "Crear las dos redes",
    subtitle: "U10-04 · online y target · 12 → 64 → 32 → 3",
    category: "16 / REFUERZO U10",
    color: "#ff757f",
    input: "frame",
    output: "frame",
    fields: [
      { key: "widths", label: "Anchos ocultos", value: "64,32" },
      { key: "lr", label: "Learning rate (Adam)", value: 0.001, min: 0.00001, max: 0.1, step: 0.0005 },
      { key: "seed", label: "Semilla", value: 42, min: 0, max: 99999 },
    ],
  },
  rl_replay: {
    label: "Explorar y recordar",
    subtitle: "U10-05/06 · ε-greedy · replay buffer · muestreo aleatorio",
    category: "16 / REFUERZO U10",
    color: "#ff757f",
    input: "frame",
    output: "frame",
    fields: [
      { key: "capacity", label: "Capacidad del buffer", value: 10000, min: 100, max: 1000000 },
      { key: "warmup", label: "Experiencias antes de aprender", value: 128, min: 8, max: 100000 },
      { key: "batch_size", label: "Lote", value: 64, min: 8, max: 4096 },
      { key: "update_every", label: "Actualizar cada (pasos)", value: 4, min: 1, max: 100 },
      { key: "eps_start", label: "ε inicial", value: 1, min: 0, max: 1, step: 0.05 },
      { key: "eps_end", label: "ε final", value: 0.05, min: 0, max: 1, step: 0.01 },
      { key: "eps_decay_steps", label: "Pasos de decaimiento de ε", value: 5000, min: 1, max: 1000000 },
    ],
  },
  rl_target: {
    label: "Objetivo Double DQN",
    subtitle: "U10-07 · online elige, target evalúa · γ",
    category: "16 / REFUERZO U10",
    color: "#ff757f",
    input: "frame",
    output: "frame",
    fields: [
      { key: "gamma", label: "γ (descuento)", value: 0.95, min: 0, max: 0.999, step: 0.005 },
      { key: "double", label: "Double DQN (si no, DQN clásica)", value: true },
    ],
  },
  rl_sync: {
    label: "Sincronizar y seleccionar",
    subtitle: "U10-09 · copiar online → target · checkpoint por VA",
    category: "16 / REFUERZO U10",
    color: "#ff757f",
    input: "frame",
    output: "frame",
    fields: [
      { key: "sync_every", label: "Copiar cada (actualizaciones)", value: 100, min: 1, max: 100000 },
      { key: "checkpoint_metric", label: "Métrica de checkpoint en VA", value: "ROI neto", options: ["ROI neto", "Sharpe"] },
    ],
  },
  rl_train: {
    label: "Actualizar online",
    subtitle: "U10-08 · smooth L1 · clipping · bucle de entrenamiento",
    category: "16 / REFUERZO U10",
    color: "#ff757f",
    input: "frame",
    output: "frame",
    fields: [
      { key: "epochs", label: "Épocas (pasadas por TR)", value: 12, min: 1, max: 500 },
      { key: "grad_clip", label: "Clipping de norma", value: 1, min: 0.01, max: 100, step: 0.1 },
    ],
  },
  rl_baselines: {
    label: "Referencias de política",
    subtitle: "U10-10 · Q-learning tabular · siempre largo · efectivo",
    category: "16 / REFUERZO U10",
    color: "#ff757f",
    input: "frame",
    output: "frame",
    fields: [
      { key: "q_learning", label: "Q-learning tabular", value: true },
      { key: "always_long", label: "Siempre largo", value: true },
      { key: "cash", label: "Efectivo", value: true },
      { key: "epochs", label: "Épocas de Q-learning", value: 12, min: 1, max: 500 },
      { key: "seed", label: "Semilla", value: 42, min: 0, max: 99999 },
    ],
  },
  rl_evaluate: {
    label: "Evaluar sin aprender",
    subtitle: "U10-10 · greedy en test · ROI, Sharpe, drawdown, rotación",
    category: "16 / REFUERZO U10",
    color: "#ff757f",
    input: "frame",
    output: "frame",
    fields: [],
  },
  gen_dcgan_train: {
    label: "08b · cDCGAN 1D",
    subtitle: "Generativos ampliación · embedding de régimen · ConvTranspose1d · D Conv1d",
    category: "14 / GENERATIVOS",
    color: "#f7768e",
    input: "frame",
    output: "frame",
    fields: [
      { key: "latent_dim", label: "Dimensión de z", value: 8, min: 1, max: 64 },
      { key: "embedding_dim", label: "Dimensión del embedding de régimen", value: 3, min: 1, max: 16 },
      { key: "channels", label: "Canales convolucionales", value: 32, min: 4, max: 256 },
      { key: "regime_cut", label: "Corte de régimen (momentum escalado)", value: 0.5, min: 0, max: 3, step: 0.1 },
      { key: "batchnorm", label: "BatchNorm en G", value: true },
      { key: "epochs", label: "Épocas", value: 60, min: 1, max: 1000 },
      { key: "batch_size", label: "Lote", value: 128, min: 8, max: 2048 },
      { key: "lr_g", label: "Learning rate G", value: 0.0002, min: 0.00001, max: 0.01, step: 0.00005 },
      { key: "lr_d", label: "Learning rate D", value: 0.0002, min: 0.00001, max: 0.01, step: 0.00005 },
      { key: "grad_clip", label: "Clip de gradiente", value: 1, min: 0.01, max: 100, step: 0.1 },
      { key: "snapshot_every", label: "Instantánea cada (épocas)", value: 1, min: 1, max: 50 },
      { key: "device", label: "Dispositivo", value: "cpu", options: ["cpu", "auto", "cuda"] },
      { key: "seed", label: "Semilla", value: 42, min: 0, max: 99999 },
    ],
  },
  gen_regime_compare: {
    label: "09b · Comparar regímenes",
    subtitle: "Generativos ampliación · mismo ruido · bajista / lateral / alcista",
    category: "14 / GENERATIVOS",
    color: "#f7768e",
    input: "frame",
    output: "frame",
    fields: [
      { key: "scenarios", label: "Escenarios por régimen", value: 500, min: 10, max: 5000 },
      { key: "noise_seed", label: "Semilla del ruido", value: 43, min: 0, max: 99999 },
    ],
  },
  ae_split: {
    label: "1 · Validar y dividir",
    subtitle: "Autoencoders · controles de entrada · y_bps · TR/VA/TE cronológico",
    category: "13 / AUTOENCODERS",
    color: "#ff9e64",
    input: "frame",
    output: "frame",
    fields: [
      { key: "features", label: "Indicadores (coma)", value: "LogReturn,Volatility,RSI,EMA,SMA,VWAP,MACD,MACDSignal,BBUpper,BBMiddle,BBLower" },
      { key: "target", label: "Rendimiento para y_bps", value: "LogReturn" },
      { key: "horizon", label: "Horizonte de y_bps (filas)", value: 16, min: 1, max: 252 },
      { key: "train_ratio", label: "Fracción TR", value: 0.65, min: 0.3, max: 0.9, step: 0.05 },
      { key: "val_ratio", label: "Fracción VA", value: 0.15, min: 0.05, max: 0.4, step: 0.05 },
      { key: "gap", label: "Hueco mínimo entre conjuntos (filas)", value: 0, min: 0, max: 500 },
    ],
  },
  ae_standardize: {
    label: "2 · Estandarizar (TR)",
    subtitle: "Autoencoders · 01 · StandardScaler ajustado sólo con TR",
    category: "13 / AUTOENCODERS",
    color: "#ff9e64",
    input: "frame",
    output: "frame",
    fields: [],
  },
  ae_pca: {
    label: "3 · Referencia PCA",
    subtitle: "Autoencoders · 02 · compresión lineal",
    category: "13 / AUTOENCODERS",
    color: "#ff9e64",
    input: "frame",
    output: "frame",
    fields: [{ key: "components", label: "Componentes", value: 3, min: 1, max: 10 }],
  },
  ae_encoder: {
    label: "4 · Encoder",
    subtitle: "Autoencoders · 03 · Lineal(d, h) → act → Lineal(h, k)",
    category: "13 / AUTOENCODERS",
    color: "#ff9e64",
    input: "frame",
    output: "frame",
    fields: [
      { key: "hidden", label: "Neuronas ocultas h", value: 32, min: 2, max: 1024 },
      { key: "latent", label: "Código latente k", value: 3, min: 1, max: 10 },
      { key: "activation", label: "Activación", value: "ReLU", options: ["ReLU", "Tanh", "GELU", "SiLU", "LeakyReLU"] },
    ],
  },
  ae_decoder: {
    label: "5 · Decoder",
    subtitle: "Autoencoders · 04–05 · Lineal(k, h) → act → Lineal(h, d) · x̂",
    category: "13 / AUTOENCODERS",
    color: "#ff9e64",
    input: "frame",
    output: "frame",
    fields: [
      { key: "hidden", label: "Neuronas ocultas h", value: 32, min: 2, max: 1024 },
      { key: "activation", label: "Activación", value: "ReLU", options: ["ReLU", "Tanh", "GELU", "SiLU", "LeakyReLU"] },
    ],
  },
  ae_noise: {
    label: "6 · Ruido (denoising)",
    subtitle: "Autoencoders · 06 · inp = xb + σ·randn",
    category: "13 / AUTOENCODERS",
    color: "#ff9e64",
    input: "frame",
    output: "frame",
    fields: [
      { key: "sigma", label: "σ del ruido gaussiano", value: 0.1, min: 0.01, max: 2, step: 0.01 },
      { key: "seed", label: "Semilla (muestra)", value: 42, min: 0, max: 99999 },
    ],
  },
  ae_train: {
    label: "7 · Entrenar AE / DAE",
    subtitle: "Autoencoders · 07 · early stopping VA · controles numéricos",
    category: "13 / AUTOENCODERS",
    color: "#ff9e64",
    input: "frame",
    output: "frame",
    fields: [
      { key: "train_ae", label: "Entrenar AE", value: true },
      { key: "train_dae", label: "Entrenar DAE (requiere Ruido)", value: true },
      { key: "epochs", label: "Épocas máximas", value: 80, min: 1, max: 2000 },
      { key: "batch_size", label: "Tamaño de lote", value: 64, min: 8, max: 4096 },
      { key: "lr", label: "Learning rate", value: 0.001, min: 0.00001, max: 1, step: 0.0005 },
      { key: "weight_decay", label: "Weight decay", value: 0, min: 0, max: 0.1, step: 0.0001 },
      { key: "patience", label: "Paciencia (épocas sin mejora en VA)", value: 12, min: 1, max: 200 },
      { key: "grad_clip", label: "Clipping de gradiente (norma)", value: 1, min: 0, max: 100, step: 0.1 },
      { key: "device", label: "Dispositivo", value: "cpu", options: ["cpu", "auto", "cuda", "mps"] },
      { key: "seed", label: "Semilla", value: 42, min: 0, max: 99999 },
    ],
  },
  ae_select: {
    label: "8 · Seleccionar por reconstrucción",
    subtitle: "Autoencoders · 08 · menor MSE de validación",
    category: "13 / AUTOENCODERS",
    color: "#ff9e64",
    input: "frame",
    output: "frame",
    fields: [{ key: "include_pca", label: "Incluir PCA en la comparación", value: true }],
  },
  ae_ridge: {
    label: "9 · Ridge sobre factores",
    subtitle: "Autoencoders · 09 · utilidad predictiva de z → y_bps",
    category: "13 / AUTOENCODERS",
    color: "#ff9e64",
    input: "frame",
    output: "frame",
    fields: [{ key: "alpha", label: "Regularización α", value: 1, min: 0, max: 1000, step: 0.1 }],
  },
  ae_test: {
    label: "10 · Medir en test",
    subtitle: "Autoencoders · 10 · reconstrucción y predicción en TE",
    category: "13 / AUTOENCODERS",
    color: "#ff9e64",
    input: "frame",
    output: "frame",
    fields: [],
  },
  yfinance_data: {
    label: "yfinance Data",
    subtitle: "Yahoo Finance · ticker · interval · period · ajustes",
    category: "01 / DATA",
    color: "#f2ad65",
    output: "frame",
    fields: [
      { key: "ticker", label: "Ticker de Yahoo (SPY, BTC-USD, EURUSD=X, ^GSPC…)", value: "SPY" },
      {
        key: "interval",
        label: "Intervalo",
        value: "1d",
        options: ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"],
      },
      { key: "range_mode", label: "Rango", value: "period", options: ["period", "start_end"] },
      {
        key: "period",
        label: "Periodo",
        value: "1y",
        options: ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"],
        showIf: { key: "range_mode", value: "period" },
      },
      { key: "start", label: "Inicio (AAAA-MM-DD)", value: "2024-01-01", showIf: { key: "range_mode", value: "start_end" } },
      { key: "end", label: "Fin exclusivo (AAAA-MM-DD, vacío = hoy)", value: "", showIf: { key: "range_mode", value: "start_end" } },
      { key: "auto_adjust", label: "auto_adjust (ajustar OHLC por dividendos y splits)", value: true },
      { key: "back_adjust", label: "back_adjust (sólo sin auto_adjust)", value: false, showIf: { key: "auto_adjust", value: false } },
      { key: "actions", label: "actions (columnas Dividends / StockSplits)", value: false },
      { key: "prepost", label: "prepost (pre y post mercado, sólo intradía)", value: false },
      { key: "repair", label: "repair (corregir errores de precio de Yahoo)", value: false },
      { key: "keepna", label: "keepna (conservar filas sin precio)", value: false },
      { key: "rounding", label: "rounding (redondear a la precisión de Yahoo)", value: false },
      { key: "drop_incomplete", label: "Descartar la última barra si aún no ha cerrado", value: true },
      { key: "timezone", label: "Zona horaria del intradía", value: "America/New_York", options: ["America/New_York", "UTC", "Europe/London", "America/Mexico_City", "Asia/Tokyo"] },
      { key: "timeout", label: "Timeout (s)", value: 20, min: 3, max: 120 },
      { key: "cache_seconds", label: "Caché (s; intradía máx. 60)", value: 300, min: 0, max: 86400 },
    ],
  },
  polymarket_dashboard: {
    label: "Polymarket Dashboard",
    subtitle: "Mercado conectado o todo Polymarket · KPIs · libro · flujo",
    category: "10 / POLYMARKET",
    color: "#7aa2f7",
    output: "frame",
    fields: [
      {
        key: "category",
        label: "Categoría",
        value: "todas",
        options: ["todas", "politics", "crypto", "sports", "finance", "economy", "tech", "ai", "geopolitics", "elections", "culture", "weather"],
      },
      { key: "sort", label: "Ordenar eventos por", value: "volumen_24h", options: ["volumen_24h", "liquidez", "nuevos"] },
      { key: "query", label: "Filtrar por texto en el título (opcional)", value: "" },
      { key: "pages", label: "Páginas de 50 eventos", value: 2, min: 1, max: 10 },
      { key: "top_n", label: "Filas por ranking", value: 15, min: 5, max: 50 },
      { key: "min_volume_24h", label: "Volumen 24 h mínimo ($)", value: 0, min: 0, max: 10000000 },
      { key: "trades", label: "Operaciones recientes (mercado conectado)", value: 200, min: 20, max: 1000 },
      { key: "refresh_seconds", label: "Refresco en vivo (s)", value: 60, min: 15, max: 3600 },
      { key: "cache_seconds", label: "Caché (s)", value: 30, min: 0, max: 3600 },
      { key: "timeout", label: "Timeout por petición (s)", value: 10, min: 1, max: 30 },
    ],
  },
  polymarket_source: {
    label: "Polymarket Market",
    subtitle: "Gamma + CLOB · P(UP) implícita · libro",
    category: "10 / POLYMARKET",
    color: "#7aa2f7",
    output: "frame",
    fields: [
      { key: "market", label: "Mercado (slug, URL, condition_id, token_id o texto)", value: "btc-updown-15m" },
      { key: "window", label: "Ventana temporal", value: "auto", options: ["auto", "5m", "15m", "30m", "1h"] },
      { key: "fallback_window", label: "Ventana si no se detecta", value: "1h", options: ["5m", "15m", "30m", "1h"] },
      { key: "up_outcome", label: "Outcome UP (índice)", value: "auto", options: ["auto", "0", "1"] },
      { key: "history_interval", label: "Historial CLOB", value: "1d", options: ["1h", "6h", "1d", "1w", "max"] },
      { key: "fidelity", label: "Fidelidad (min por punto)", value: 1, min: 1, max: 1440 },
      { key: "cache_seconds", label: "Caché (s)", value: 5, min: 0, max: 3600 },
      { key: "timeout", label: "Timeout por petición (s)", value: 8, min: 1, max: 30 },
      { key: "refresh_seconds", label: "Refresco en vivo (s)", value: 15, min: 5, max: 600 },
    ],
  },
  laya_updown: {
    label: "LAYA UP/DOWN",
    subtitle: "Pesos × señales → decisión UP / DOWN por ventana",
    category: "10 / POLYMARKET",
    color: "#7aa2f7",
    input: "frame",
    output: "frame",
    fields: [
      { key: "window", label: "Horizonte", value: "inherit", options: ["inherit", "5m", "15m", "30m", "1h"] },
      { key: "use_laya", label: "Consultar Laya-MLX", value: true },
      { key: "checkpoint", label: "Checkpoint", value: "laya-mlx", options: ["laya-mlx", "laya-multilingual-mlx"] },
      { key: "weight_source", label: "Origen de pesos", value: "manual", options: ["manual", "calibrated", "online"] },
      { key: "minute_buckets", label: "Pesos por minuto de la ventana", value: true },
      { key: "online_lr", label: "Learning rate online", value: 0.05, min: 0.001, max: 1, step: 0.005 },
      { key: "online_l2", label: "L2 online", value: 0.001, min: 0, max: 0.1, step: 0.001 },
      { key: "commit_at", label: "Decisión que se aprende", value: "first", options: ["first", "last"] },
      { key: "learning_key", label: "Serie de aprendizaje (vacío = auto)", value: "" },
      { key: "w_market", label: "Peso mercado (logit P_up)", value: 1, min: -5, max: 5, step: 0.05 },
      { key: "w_momentum", label: "Peso momentum", value: 0.5, min: -5, max: 5, step: 0.05 },
      { key: "w_laya", label: "Peso Laya (log P_up/P_down)", value: 1, min: -5, max: 5, step: 0.05 },
      { key: "w_imbalance", label: "Peso desequilibrio libro", value: 0.3, min: -5, max: 5, step: 0.05 },
      { key: "w_target", label: "Peso distancia al precio a vencer (z)", value: 1, min: -5, max: 5, step: 0.05 },
      { key: "bias", label: "Sesgo", value: 0, min: -5, max: 5, step: 0.05 },
      { key: "lookback_bars", label: "Lookback (barras)", value: 30, min: 3, max: 2000 },
      { key: "min_confidence", label: "Confianza mínima accionable", value: 0.55, min: 0.5, max: 0.99, step: 0.01 },
    ],
  },
  laya_signal: {
    label: "Laya Signal",
    subtitle: "Laya-MLX · choice · score · noul por día",
    category: "09 / LAYA",
    color: "#9ad1c4",
    input: "frame",
    output: "frame",
    fields: [
      { key: "ticker", label: "Nombre del activo en el estado (auto = ticker de Market Data)", value: "auto", mode: "daily" },
      {
        key: "checkpoint",
        label: "Checkpoint",
        value: "laya-mlx",
        options: ["laya-mlx", "laya-multilingual-mlx"],
      },
      { key: "max_rows", label: "Últimos días a evaluar", value: 500, min: 30, max: 3000, mode: "daily" },
      { key: "mode", label: "Modo (auto = Polymarket si la fuente lo es)", value: "auto", options: ["auto", "daily", "polymarket"] },
      { key: "train_windows", label: "Ventanas resueltas para calibrar", value: 120, min: 20, max: 500, mode: "polymarket" },
      { key: "bucket_s", label: "Tramos dentro de la ventana (s; 0 = un solo momento)", value: "auto", options: ["auto", "0", "60", "120", "300"], mode: "polymarket" },
      { key: "decision_offset_s", label: "Segundos tras el inicio (si tramos = 0)", value: 60, min: 0, max: 3599, mode: "polymarket" },
      { key: "pm_lookback", label: "Lookback (puntos de 1 min)", value: 5, min: 3, max: 60, mode: "polymarket" },
      {
        key: "trend_question",
        label: "Pregunta choice (up/down/flat)",
        value: "Given these technical indicators, what is the most likely direction of the next trading day?",
        mode: "daily",
      },
      {
        key: "risk_question",
        label: "Pregunta score (riesgo 0–2)",
        value: "How risky is holding this asset tomorrow?",
        mode: "daily",
      },
      {
        key: "higher_question",
        label: "Proposición noul (P verdadero)",
        value: "The asset will close higher on the next trading day.",
        mode: "daily",
      },
    ],
  },
  laya_calibrate: {
    label: "Laya Calibrate",
    subtitle: "Ajuste de pesos · logística TR · VA · TE",
    category: "09 / LAYA",
    color: "#9ad1c4",
    input: "frame",
    output: "frame",
    fields: [
      { key: "train_ratio", label: "Fracción TR", value: 0.6, min: 0.3, max: 0.8, step: 0.05 },
      { key: "val_ratio", label: "Fracción VA", value: 0.2, min: 0.05, max: 0.3, step: 0.05 },
      { key: "lr", label: "Learning rate", value: 0.1, min: 0.0001, max: 5, step: 0.01 },
      { key: "epochs", label: "Épocas", value: 300, min: 10, max: 5000 },
      { key: "l2", label: "Regularización L2", value: 0.01, min: 0, max: 10, step: 0.01 },
      { key: "imbalance_weight", label: "Peso fijo del desequilibrio del libro", value: 0, min: -5, max: 5, step: 0.05, mode: "polymarket" },
    ],
  },
  laya_feedback: {
    label: "Laya Feedback",
    subtitle: "Predecir → observar → corregir pesos",
    category: "09 / LAYA",
    color: "#9ad1c4",
    input: "frame",
    output: "frame",
    fields: [
      { key: "lr", label: "Learning rate online", value: 0.01, min: 0, max: 2, step: 0.005 },
      { key: "min_confidence", label: "Confianza mínima para operar", value: 0.52, min: 0.5, max: 0.99, step: 0.01 },
      {
        key: "positions",
        label: "Posiciones",
        value: "long_short",
        options: ["long_short", "long_only"],
      },
      { key: "cost_bps", label: "Coste por cambio (bps)", value: 5, min: 0, max: 100 },
      { key: "rolling_window", label: "Ventana métricas móviles", value: 20, min: 5, max: 120 },
      { key: "start_fraction", label: "Inicio si no hay Calibrate", value: 0.6, min: 0.2, max: 0.9, step: 0.05 },
    ],
  },
  data_quality: {
    label: "Data Quality",
    subtitle: "Sort · duplicates · missing · causal fill",
    category: "01 / DATA",
    color: "#f2ad65",
    input: "frame",
    output: "frame",
    fields: [
      {
        key: "duplicates",
        label: "Duplicate timestamps",
        value: "error",
        options: ["error", "last"],
      },
      {
        key: "missing",
        label: "Missing values",
        value: "drop",
        options: ["error", "drop", "ffill"],
      },
      {
        key: "ffill_limit",
        label: "Maximum causal forward fill",
        value: 3,
        min: 1,
        max: 30,
      },
    ],
  },
  quant_features: {
    label: "Quant Features",
    subtitle: "Returns · rolling risk · drawdown",
    category: "06 / QUANT",
    color: "#d5d77e",
    input: "frame",
    output: "frame",
    fields: [
      { key: "price_column", label: "Price column", value: "Close" },
      { key: "window", label: "Rolling window", value: 20, min: 2, max: 252 },
      {
        key: "periods_per_year",
        label: "Observations per year",
        value: 252,
        min: 1,
        max: 366,
      },
      {
        key: "risk_free_rate",
        label: "Annual risk-free rate",
        value: 0.02,
        min: 0,
        max: 1,
        step: 0.01,
      },
    ],
  },
  ml_model: {
    label: "Classical ML",
    subtitle: "Seven baselines · VA selection · TE evaluation",
    category: "03 / ML",
    color: "#79b4ef",
    input: "tensor",
    output: "tensor",
    fields: [
      {
        key: "algorithm",
        label: "Algorithm",
        value: "Compare",
        options: [
          "Compare",
          "Mean baseline",
          "Ridge",
          "ElasticNet",
          "Random Forest",
          "Histogram Boosting",
          "SVR",
          "KNN",
        ],
      },
      {
        key: "window_mode",
        label: "Window features",
        value: "last",
        options: ["last", "flatten"],
      },
      {
        key: "estimators",
        label: "Trees / boosting iterations",
        value: 100,
        min: 10,
        max: 500,
      },
      { key: "max_depth", label: "Maximum depth", value: 8, min: 1, max: 32 },
      {
        key: "alpha",
        label: "Penalty alpha / SVR C",
        value: 1,
        min: 0.0001,
        max: 1000000,
        step: 0.1,
      },
      { key: "neighbors", label: "KNN neighbors", value: 5, min: 1, max: 100 },
      { key: "seed", label: "Seed", value: 42, min: 0, max: 100000 },
    ],
  },
  rl_agent: {
    label: "RL Trading Agent",
    subtitle: "Double DQN · Q-learning · greedy holdout",
    category: "08 / RL",
    color: "#ec9dd6",
    input: "frame",
    output: "tensor",
    fields: [
      {
        key: "feature_columns",
        label: "State feature columns",
        value:
          "LogReturn,Volatility,RSI,EMA,SMA,VWAP,MACD,MACDSignal,BBUpper,BBMiddle,BBLower",
      },
      {
        key: "lookback",
        label: "Lookback (last state used by current agent)",
        value: 1,
        min: 1,
        max: 120,
      },
      {
        key: "epochs",
        label: "Training episodes",
        value: 12,
        min: 1,
        max: 300,
      },
      {
        key: "lr",
        label: "Learning rate",
        value: 0.001,
        min: 0.000001,
        max: 0.1,
        step: 0.0001,
      },
      {
        key: "train_ratio",
        label: "Training fraction",
        value: 0.65,
        min: 0.3,
        max: 0.85,
        step: 0.01,
      },
      {
        key: "val_ratio",
        label: "Validation fraction",
        value: 0.15,
        min: 0.05,
        max: 0.3,
        step: 0.01,
      },
      {
        key: "cost_bps",
        label: "Cost per position change (bps)",
        value: 5,
        min: 0,
        max: 100,
      },
      { key: "seed", label: "Seed", value: 42, min: 0, max: 100000 },
    ],
  },
  resample: {
    label: "Resample",
    subtitle: "Dated aggregation · reusable",
    category: "02 / TRANSFORM",
    color: "#6cc9ac",
    input: "frame",
    output: "frame",
    fields: [
      { key: "columns", label: "Columns (CSV)", value: "Close" },
      {
        key: "frequency",
        label: "Frequency",
        value: "ME",
        options: ["W-FRI", "ME", "QE", "YE"],
      },
      {
        key: "aggregation",
        label: "Aggregation",
        value: "last",
        options: ["last", "mean", "sum"],
      },
      {
        key: "exclude_incomplete",
        label: "Exclude unfinished period",
        value: true,
      },
    ],
  },
  select_columns: {
    label: "Select Columns",
    subtitle: "Choose columns for a branch",
    category: "02 / TRANSFORM",
    color: "#6cc9ac",
    input: "frame",
    output: "frame",
    fields: [{ key: "columns", label: "Columns (CSV)", value: "Close" }],
  },
  coordinate: {
    label: "Numeric Coordinate",
    subtitle: "Ordered numeric X · no economic predictor",
    category: "02 / TRANSFORM",
    color: "#6cc9ac",
    input: "frame",
    output: "frame",
    fields: [
      { key: "start", label: "Range start", value: -2 },
      { key: "stop", label: "Range end", value: 2 },
      { key: "output_column", label: "New column name", value: "Coordinate" },
    ],
  },
  center: {
    label: "Center Columns",
    subtitle: "Subtract fitting mean · preserve originals",
    category: "02 / TRANSFORM",
    color: "#6cc9ac",
    input: "frame",
    output: "frame",
    fields: [
      { key: "columns", label: "Columns (CSV)", value: "Close" },
      { key: "prefix", label: "New column prefix", value: "Centered_" },
      {
        key: "fit_fraction",
        label: "Leading fitting fraction (1 = descriptive full sample)",
        value: 0.65,
        min: 0.01,
        max: 1,
        step: 0.01,
      },
    ],
  },
  scale_columns: {
    label: "Scale Columns",
    subtitle: "Fit leading rows · transform all · preserve originals",
    category: "02 / TRANSFORM",
    color: "#6cc9ac",
    input: "frame",
    output: "frame",
    fields: [
      { key: "columns", label: "Columns (CSV)", value: "Close" },
      { key: "prefix", label: "New column prefix", value: "Scaled_" },
      {
        key: "method",
        label: "Scaler",
        value: "standard",
        options: ["standard", "minmax"],
      },
      {
        key: "fit_fraction",
        label: "Leading fitting fraction (1 = descriptive full sample)",
        value: 0.65,
        min: 0.01,
        max: 1,
        step: 0.01,
      },
    ],
  },
  curve_data: {
    label: "Monthly Curve Data",
    subtitle: "Monthly last Close · coordinate −2…2",
    category: "07 / NOTEBOOK",
    color: "#f2ad65",
    input: "frame",
    output: "frame",
    fields: [
      {
        key: "exclude_incomplete_month",
        label: "Exclude unfinished current month",
        value: true,
      },
    ],
  },
  curve_fit: {
    label: "Curve Fitting / Benchmark",
    subtitle: "Notebook approximation · OLS / neural",
    category: "07 / NOTEBOOK",
    color: "#ec9dd6",
    input: "frame",
    output: "tensor",
    fields: [
      { key: "name", label: "Benchmark name", value: "MLP 1 → 256 → 1" },
      {
        key: "variant",
        label: "Model family",
        value: "dense",
        options: ["ols", "polynomial_net", "dense"],
      },
      {
        key: "evaluation",
        label: "Evaluation protocol",
        value: "in_sample",
        options: ["in_sample", "temporal"],
      },
      { key: "degree", label: "Polynomial degree", value: 5, min: 1, max: 10 },
      { key: "widths", label: "Hidden widths (comma separated)", value: "256" },
      {
        key: "activation",
        label: "Hidden activation",
        value: "ReLU",
        options: ["ReLU", "Swish", "Tanh", "GELU", "Sigmoid"],
      },
      {
        key: "optimizer",
        label: "Optimizer",
        value: "RMSprop",
        options: ["RMSprop", "Adam", "SGD", "AdamW"],
      },
      {
        key: "epochs",
        label: "Epochs (notebook: 1500 / 2000 / 2500)",
        value: 100,
        min: 1,
        max: 5000,
      },
      { key: "batch_size", label: "Batch size", value: 32, min: 2, max: 1024 },
      {
        key: "lr",
        label: "Learning rate",
        value: 0.001,
        min: 0.000001,
        max: 0.1,
        step: 0.0001,
      },
      {
        key: "normalize",
        label: "Normalize inputs using training rows",
        value: true,
      },
      {
        key: "l2",
        label: "L2 weight penalty (exclude bias)",
        value: 0,
        min: 0,
        max: 1,
        step: 0.001,
      },
      {
        key: "reduce_lr",
        label: "Reduce LR on plateau (factor .5, patience 30)",
        value: false,
      },
      {
        key: "train_ratio",
        label: "Temporal training fraction",
        value: 0.65,
        min: 0.3,
        max: 0.9,
        step: 0.05,
      },
      {
        key: "val_ratio",
        label: "Temporal validation fraction",
        value: 0.15,
        min: 0.05,
        max: 0.5,
        step: 0.05,
      },
      {
        key: "patience",
        label: "Temporal early stopping patience",
        value: 100,
        min: 1,
        max: 1000,
      },
      {
        key: "seed",
        label: "Random seed",
        value: 100,
        min: 0,
        max: 2147483647,
      },
    ],
  },
  study: {
    label: "Learning Experiment (legado)",
    subtitle: "Monolito heredado: usa las herramientas 15 / SUPERVISADO, 13, 14 y 16",
    category: "99 / LEGADO",
    color: "#ec9dd6",
    input: "frame",
    output: "tensor",
    fields: [
      {
        key: "topic",
        label: "Experiment",
        value: "feedforward",
        options: [
          "feedforward",
          "optimizers",
          "regularization",
          "hyperparameters",
          "cnn",
          "recurrent",
          "autoencoders",
          "generative",
          "cdcgan",
          "double_dqn",
        ],
      },
      {
        key: "feature_columns",
        label: "Input feature columns",
        value:
          "LogReturn,Volatility,RSI,EMA,SMA,VWAP,MACD,MACDSignal,BBUpper,BBMiddle,BBLower",
      },
      {
        key: "widths",
        label: "MLP hidden widths (comma separated)",
        value: "64,32",
      },
      {
        key: "lookback",
        label: "Historical sessions per sample",
        value: 1,
        min: 1,
        max: 120,
      },
      {
        key: "train_ratio",
        label: "Training fraction",
        value: 0.65,
        min: 0.3,
        max: 0.85,
        step: 0.05,
      },
      {
        key: "val_ratio",
        label: "Validation fraction",
        value: 0.15,
        min: 0.05,
        max: 0.5,
        step: 0.05,
      },
      {
        key: "epochs",
        label: "Maximum epochs / RL episodes",
        value: 12,
        min: 1,
        max: 300,
      },
      {
        key: "patience",
        label: "Early stopping patience",
        value: 5,
        min: 1,
        max: 100,
      },
      { key: "batch_size", label: "Batch size", value: 64, min: 2, max: 1024 },
      {
        key: "lr",
        label: "Learning rate",
        value: 0.001,
        min: 0.000001,
        max: 0.1,
        step: 0.0001,
      },
      {
        key: "trials",
        label: "HPO random trials / 2 expanding folds",
        value: 6,
        min: 1,
        max: 30,
      },
      {
        key: "latent_dim",
        label: "Latent dimensions",
        value: 3,
        min: 1,
        max: 32,
      },
      {
        key: "horizon",
        label: "Generative horizon (multiple of 4)",
        value: 16,
        min: 4,
        max: 64,
        step: 4,
      },
      {
        key: "cost_bps",
        label: "Cost per position unit / bps",
        value: 5,
        min: 0,
        max: 100,
      },
      {
        key: "threshold_bps",
        label: "Regression decision threshold / bps",
        value: 0,
        min: -100,
        max: 100,
      },
      { key: "seed", label: "Random seed", value: 42, min: 0, max: 2147483647 },
    ],
  },
  source: {
    label: "Market Data",
    subtitle: "OHLCV · daily bars",
    category: "01 / DATA",
    color: "#f2ad65",
    output: "frame",
    fields: [
      {
        key: "source",
        label: "Data source",
        value: "synthetic",
        options: ["synthetic", "yfinance"],
      },
      { key: "ticker", label: "Ticker", value: "SPY" },
      { key: "years", label: "History / years", value: 3, min: 1, max: 20 },
    ],
  },
  features: {
    label: "Technical Features",
    subtitle: "LogReturn · Volatility + selectable indicators",
    category: "02 / TRANSFORM",
    color: "#e6b86a",
    input: "frame",
    output: "frame",
    fields: [
      {
        key: "lags",
        label: "Lag features (0 disables)",
        value: 5,
        min: 0,
        max: 60,
      },
      {
        key: "volatility_window",
        label: "Volatility period (log returns)",
        value: 20,
        min: 2,
        max: 252,
      },
      ...["rsi", "ema", "sma", "macd", "vwap", "bb"].map((name) => ({
        key: `use_${name}`,
        label: `Enable ${name.toUpperCase()}`,
        value: true,
      })),
      { key: "ema_window", label: "EMA period", value: 20, min: 2, max: 252 },
      { key: "sma_window", label: "SMA period", value: 20, min: 2, max: 252 },
      {
        key: "macd_fast",
        label: "MACD fast period",
        value: 12,
        min: 2,
        max: 252,
      },
      {
        key: "macd_slow",
        label: "MACD slow period",
        value: 26,
        min: 2,
        max: 252,
      },
      {
        key: "macd_signal",
        label: "MACD signal period",
        value: 9,
        min: 2,
        max: 252,
      },
      {
        key: "vwap_window",
        label: "Rolling VWAP / daily bars",
        value: 20,
        min: 2,
        max: 252,
      },
      {
        key: "bb_window",
        label: "Bollinger period",
        value: 20,
        min: 2,
        max: 252,
      },
      {
        key: "bb_std",
        label: "Bollinger deviations",
        value: 2,
        min: 0.1,
        max: 5,
        step: 0.1,
      },
      { key: "rsi_window", label: "RSI window", value: 14, min: 2, max: 100 },
    ],
  },
  prepare: {
    label: "Series Preparation",
    subtitle: "Temporal split · scaling",
    category: "02 / TRANSFORM",
    color: "#6cc9ac",
    input: "frame",
    output: "tensor",
    fields: [
      { key: "feature_columns", label: "Input feature columns", value: "" },
      {
        key: "target",
        label: "Prediction target",
        value: "LogReturn",
        options: ["LogReturn", "Close", "Volatility"],
      },
      {
        key: "lookback",
        label: "Sequence length",
        value: 20,
        min: 1,
        max: 120,
      },
      {
        key: "train_ratio",
        label: "Training fraction",
        value: 0.65,
        min: 0.3,
        max: 0.9,
        step: 0.05,
      },
      {
        key: "val_ratio",
        label: "Validation fraction",
        value: 0.15,
        min: 0.05,
        max: 0.5,
        step: 0.05,
      },
      {
        key: "scaler",
        label: "Feature scaler",
        value: "standard",
        options: ["standard", "minmax"],
      },
      {
        key: "period",
        label: "Decomposition period",
        value: 20,
        min: 2,
        max: 120,
      },
    ],
  },
  model: {
    label: "Deep Learning",
    subtitle: "Train · validate · forecast",
    category: "03 / LEARNING",
    color: "#9c9af8",
    input: "tensor",
    output: "tensor",
    fields: [
      {
        key: "architecture",
        label: "Architecture",
        value: "LSTM",
        options: [
          "Custom",
          "Dense",
          "CNN",
          "LSTM",
          "GRU",
          "BiLSTM",
          "BiGRU",
          "CNN-LSTM",
          "Transformer",
          "Autoencoder",
          "DAE",
          "DQN",
        ],
      },
      {
        key: "heads",
        label: "Transformer attention heads",
        value: 4,
        options: ["1", "2", "4", "8"],
      },
      { key: "hidden", label: "Hidden units", value: 32, min: 4, max: 256 },
      { key: "layers", label: "Hidden layers", value: 1, min: 1, max: 4 },
      { key: "kernel", label: "CNN kernel", value: 3, min: 1, max: 20 },
      {
        key: "dropout",
        label: "Dropout",
        value: 0.15,
        min: 0,
        max: 0.8,
        step: 0.05,
      },
      { key: "batch_norm", label: "Batch normalization", value: false },
      { key: "epochs", label: "Maximum epochs", value: 12, min: 1, max: 300 },
      { key: "batch_size", label: "Batch size", value: 32, min: 2, max: 1024 },
      {
        key: "lr",
        label: "Learning rate",
        value: 0.001,
        min: 0.000001,
        max: 1,
        step: 0.0001,
      },
      {
        key: "optimizer",
        label: "Optimizer",
        value: "Adam",
        options: ["Adam", "AdamW", "RMSprop", "SGD"],
      },
      {
        key: "patience",
        label: "Early stopping patience",
        value: 5,
        min: 1,
        max: 100,
      },
      { key: "l1", label: "L1 penalty", value: 0, min: 0, step: 0.00001 },
      { key: "l2", label: "L2 penalty", value: 0.0001, min: 0, step: 0.00001 },
      {
        key: "noise",
        label: "DAE noise",
        value: 0.1,
        min: 0,
        max: 1,
        step: 0.05,
      },
    ],
  },
  backtest: {
    label: "Market Backtest",
    output: "tensor",
    subtitle: "Short / cash / long",
    category: "04 / EVALUATION",
    color: "#79b4ef",
    input: "tensor",
    fields: [
      {
        key: "threshold",
        label: "Signal threshold",
        value: 0.001,
        min: 0,
        step: 0.0001,
      },
      {
        key: "fee_bps",
        label: "Trading fee / bps",
        value: 2,
        min: 0,
        max: 1000,
      },
      {
        key: "slippage_bps",
        label: "Slippage / bps",
        value: 1,
        min: 0,
        max: 1000,
      },
    ],
  },
  network_input: {
    label: "Network Input",
    subtitle: "Declare feature count and window mode",
    category: "03 / ARCHITECTURE",
    color: "#c49af8",
    output: "network",
    fields: [
      {
        key: "auto_features",
        label: "Match selected features automatically",
        value: true,
      },
      {
        key: "input_features",
        label: "Input features",
        value: 16,
        min: 1,
        max: 512,
      },
      {
        key: "mode",
        label: "Window handling",
        value: "last",
        options: ["last", "flatten"],
      },
    ],
  },
  layer: {
    label: "Network Layer",
    subtitle: "Dense · activation · regularization",
    category: "03 / ARCHITECTURE",
    color: "#c49af8",
    input: "network",
    output: "network",
    fields: [
      {
        key: "layer_type",
        label: "Layer type",
        value: "Dense",
        options: [
          "Dense",
          "ReLU",
          "Swish",
          "Tanh",
          "GELU",
          "Sigmoid",
          "LeakyReLU",
          "Dropout",
          "BatchNorm",
        ],
      },
      { key: "units", label: "Neurons (Dense)", value: 64, min: 1, max: 2048 },
      {
        key: "activation",
        label: "Activation (Dense)",
        value: "ReLU",
        options: [
          "ReLU",
          "Swish",
          "Tanh",
          "GELU",
          "Sigmoid",
          "LeakyReLU",
          "Identity",
        ],
      },
      {
        key: "rate",
        label: "Dropout rate",
        value: 0.2,
        min: 0,
        max: 0.95,
        step: 0.05,
      },
    ],
  },
  network_output: {
    label: "Network Output",
    subtitle: "Scalar regression head",
    category: "03 / ARCHITECTURE",
    color: "#c49af8",
    input: "network",
    output: "network",
    fields: [
      {
        key: "output_units",
        label: "Output units (scalar target)",
        value: 1,
        min: 1,
        max: 1,
      },
    ],
  },
  merge: {
    label: "Merge Features",
    subtitle: "Join two feature branches by date",
    category: "02 / TRANSFORM",
    color: "#e6b86a",
    input: "frame",
    output: "frame",
    fields: [],
  },
  plot: {
    label: "Plot",
    subtitle: "Choose data and series to chart",
    category: "05 / VISUALIZATION",
    color: "#ec9dd6",
    fields: [
      {
        key: "hist_bins",
        label: "Histogram bins",
        value: 20,
        min: 5,
        max: 100,
      },
      {
        key: "hist_density",
        label: "Histogram density (instead of count)",
        value: false,
      },
      { key: "x_column", label: "X column (blank = chart default)", value: "" },
      {
        key: "y_scale",
        label: "Y scale",
        value: "linear",
        options: ["linear", "log"],
      },
      {
        key: "scatter_columns",
        label: "Point series for scatter_line",
        value: "actual",
      },
      {
        key: "skip_rows",
        label: "Skip first observations / loss epochs",
        value: 0,
        min: 0,
        max: 5000,
      },

      {
        key: "study_chart",
        label: "Experiment chart index (starts at 0)",
        value: 0,
        min: 0,
        max: 10,
      },
      { key: "title", label: "Chart title", value: "Custom plot" },
      {
        key: "color",
        label: "Chart color / predefined palette",
        value: "Auto",
        options: [
          "Auto",
          "Mint",
          "Violet",
          "Orange",
          "Blue",
          "Pink",
          "Yellow",
          "Coral",
          "Cyan",
        ],
      },
      {
        key: "series_colors",
        label: "Per-series colors (CSV; optional)",
        value: "",
      },
      {
        key: "series_labels",
        label: "Legend names (CSV; optional)",
        value: "",
      },
      {
        key: "plot_source",
        label: "What to plot",
        value: "auto",
        options: [
          "auto",
          "columns",
          "decomposition",
          "predictions",
          "backtest",
          "signals",
          "loss",
          "asset_prices",
          "equity",
          "cumulative_log",
          "drawdown",
          "weights",
        ],
      },
      {
        key: "chart_type",
        label: "Chart type",
        value: "line",
        options: [
          "line",
          "area",
          "bar",
          "scatter",
          "scatter_line",
          "histogram",
        ],
      },
      {
        key: "plot_columns",
        label: "Series (comma-separated; empty = defaults)",
        value: "",
      },
    ],
  },
  portfolio_data: {
    label: "Portfolio Data",
    subtitle: "Multiple assets · aligned adjusted prices",
    category: "06 / PORTFOLIO",
    color: "#d5d77e",
    output: "portfolio",
    fields: [
      {
        key: "source",
        label: "Data source",
        value: "synthetic",
        options: ["synthetic", "yfinance"],
      },
      {
        key: "tickers",
        label: "Assets (comma-separated, 2–20)",
        value: "SPY,QQQ,TLT",
      },
      { key: "benchmark", label: "OLS benchmark ticker", value: "SPY" },
      { key: "years", label: "History / years", value: 3, min: 1, max: 10 },
    ],
  },
  portfolio: {
    label: "Portfolio / Markowitz",
    subtitle: "Equal weight · max Sharpe · min variance",
    category: "06 / PORTFOLIO",
    color: "#d5d77e",
    input: "portfolio",
    output: "portfolio",
    fields: [
      {
        key: "title",
        label: "Portfolio report name",
        value: "Portfolio comparison",
      },
      {
        key: "weights",
        label: "Optional custom weights, in ticker order",
        value: "",
      },
      {
        key: "train_fraction",
        label: "Estimation fraction / rest is holdout",
        value: 0.7,
        min: 0.5,
        max: 0.9,
        step: 0.05,
      },
      {
        key: "max_weight",
        label: "Markowitz maximum asset weight",
        value: 1,
        min: 0.05,
        max: 1,
        step: 0.05,
      },
      {
        key: "covariance",
        label: "Covariance estimator",
        value: "shrinkage",
        options: ["shrinkage", "sample"],
      },
      {
        key: "rebalance",
        label: "Rebalancing policy",
        value: "daily",
        options: ["daily", "buy_hold"],
      },
      {
        key: "risk_free_rate",
        label: "Annual risk-free rate (decimal)",
        value: 0,
        min: -0.9,
        max: 1,
        step: 0.005,
      },
      {
        key: "confidence",
        label: "Historical VaR / CVaR confidence",
        value: 0.95,
        min: 0.8,
        max: 0.999,
        step: 0.01,
      },
      {
        key: "periods_per_year",
        label: "Trading periods per year",
        value: 252,
        min: 1,
        max: 365,
      },
      {
        key: "fee_bps",
        label: "Fee / traded notional (bps)",
        value: 0,
        min: 0,
        max: 1000,
      },
      {
        key: "slippage_bps",
        label: "Slippage (bps)",
        value: 0,
        min: 0,
        max: 1000,
      },
    ],
  },
  risk: {
    label: "Risk Metrics",
    subtitle: "ROI · CAGR · VaR · CVaR · alpha / beta",
    category: "06 / PORTFOLIO",
    color: "#d5d77e",
    output: "portfolio",
    fields: [
      { key: "title", label: "Metrics report name", value: "Risk metrics" },
      { key: "series_name", label: "Asset / strategy name", value: "Asset" },
      {
        key: "return_column",
        label: "DataFrame return column",
        value: "LogReturn",
      },
      {
        key: "return_type",
        label: "DataFrame input type",
        value: "log",
        options: ["log", "simple", "price"],
      },
      {
        key: "benchmark_name",
        label: "Optional benchmark name",
        value: "Benchmark",
      },
      {
        key: "risk_free_rate",
        label: "Annual risk-free rate (decimal)",
        value: 0,
        min: -0.9,
        max: 1,
        step: 0.005,
      },
      {
        key: "confidence",
        label: "Historical VaR / CVaR confidence",
        value: 0.95,
        min: 0.8,
        max: 0.999,
        step: 0.01,
      },
      {
        key: "periods_per_year",
        label: "Trading periods per year",
        value: 252,
        min: 1,
        max: 365,
      },
    ],
  },
  config: {
    label: "Training Config",
    subtitle: "Shared hyperparameters",
    category: "03 / LEARNING",
    color: "#6cc9ac",
    output: "config",
    fields: [
      {
        key: "lr",
        label: "Learning rate",
        value: 0.001,
        min: 0.000001,
        max: 1,
        step: 0.0001,
      },
      { key: "epochs", label: "Maximum epochs", value: 12, min: 1, max: 300 },
    ],
  },
};
export function createNode(
  kind: Kind,
  index: number,
  position = { x: index * 255, y: 100 },
): BlueprintNode {
  const def = catalog[kind];
  return {
    id: crypto.randomUUID(),
    type: "blueprint",
    position,
    data: {
      kind,
      label: def.label,
      params: Object.fromEntries(def.fields.map((f) => [f.key, f.value])),
    },
  };
}
export function defaultGraph(): Graph {
  const nodes = (
    ["source", "features", "prepare", "model", "backtest"] as Kind[]
  ).map((kind, i) => createNode(kind, i));
  return {
    nodes,
    edges: nodes.slice(1).map((n, i) => ({
      id: `edge-${i}`,
      source: nodes[i].id,
      target: n.id,
      sourceHandle: catalog[nodes[i].data.kind].output,
      targetHandle: catalog[n.data.kind].input,
      animated: true,
    })),
  };
}
declare global {
  interface Window {
    quantEngine?: {
      onNewBlueprint?: (callback: () => void) => () => void;
      execute: (graph: Graph) => Promise<unknown>;
      onProgress: (callback: (event: unknown) => void) => () => void;
      request?: (route: string, body?: unknown) => Promise<unknown>;
    };
  }
}

export const exampleFeatures = [
  "Open",
  "High",
  "Low",
  "Close",
  "Volume",
  "LogReturn",
  "Volatility",
  "RSI",
  "EMA",
  "SMA",
  "VWAP",
  "MACD",
  "MACDSignal",
  "BBUpper",
  "BBMiddle",
  "BBLower",
];
export function networkExample(): Graph {
  const graph = defaultGraph();
  const prepare = graph.nodes[2],
    trainer = graph.nodes[3];
  prepare.data.params = {
    ...prepare.data.params,
    feature_columns: exampleFeatures.join(","),
    lookback: 1,
  };
  trainer.data.params.architecture = "Custom";
  graph.nodes.forEach((node, i) => {
    node.position = { x: i * 280, y: 300 };
  });
  const network = (
    ["network_input", "layer", "layer", "network_output"] as Kind[]
  ).map((kind, i) => createNode(kind, i, { x: i * 280, y: 30 }));
  network[1].data.params.units = 64;
  network[2].data.params.units = 32;
  const edges: Edge[] = network.slice(1).map((node, i) => ({
    id: `network-${i}`,
    source: network[i].id,
    target: node.id,
    sourceHandle: "network",
    targetHandle: "network",
    animated: true,
  }));
  edges.push({
    id: "network-trainer",
    source: network[3].id,
    target: trainer.id,
    sourceHandle: "network",
    targetHandle: "network",
    animated: true,
  });
  const chart = createNode("plot", 9, { x: 100, y: 570 });
  chart.data.params = {
    title: "Price, VWAP and Bollinger bands",
    plot_source: "columns",
    plot_columns: "Close,VWAP,BBUpper,BBLower",
    chart_type: "line",
  };
  const loss = createNode("plot", 10, { x: 550, y: 570 });
  loss.data.params = {
    title: "Training and validation loss",
    plot_source: "loss",
    plot_columns: "",
    chart_type: "line",
  };
  edges.push({
    id: "plot-indicators",
    source: graph.nodes[1].id,
    target: chart.id,
    sourceHandle: "frame",
    targetHandle: "frame",
  });
  edges.push({
    id: "plot-loss",
    source: trainer.id,
    target: loss.id,
    sourceHandle: "tensor",
    targetHandle: "tensor",
  });
  return {
    nodes: [...graph.nodes, ...network, chart, loss],
    edges: [...graph.edges, ...edges],
  };
}
export function featureOptions(
  graph: Graph,
  id: string,
  visited = new Set<string>(),
): string[] {
  if (visited.has(id)) return [];
  visited.add(id);
  const node = graph.nodes.find((n) => n.id === id);
  if (!node) return [];
  const incoming = graph.edges.filter(
    (e) => e.target === id && e.sourceHandle === "frame",
  );
  const base = [
    ...new Set(
      incoming.flatMap((e) =>
        featureOptions(graph, e.source, new Set(visited)),
      ),
    ),
  ];
  const split = (v: Param | undefined) => String(v ?? "").split(",").map((s) => s.trim()).filter(Boolean);
  if (node.data.kind === "ml_select") {
    const cols = split(node.data.params.columns ?? "Close,Volume");
    const kept = node.data.params.mode === "drop" ? base.filter((c) => !cols.includes(c)) : cols.length ? cols : base;
    return [...new Set([...kept, ...split(node.data.params.always_keep)])];
  }
  if (node.data.kind === "ml_returns") {
    const type = String(node.data.params.return_type ?? "log");
    const k = Number(node.data.params.periods ?? 1);
    const made = split(node.data.params.columns ?? "Close").flatMap((c) => [
      ...(type !== "log" ? [`${node.data.params.prefix_simple ?? "Ret_"}${c}${k > 1 ? `_${k}` : ""}`] : []),
      ...(type !== "simple" ? [`${node.data.params.prefix_log ?? "LogRet_"}${c}${k > 1 ? `_${k}` : ""}`] : []),
    ]);
    const dropped = node.data.params.replace === true ? split(node.data.params.columns ?? "Close") : [];
    return [...new Set([...base.filter((c) => !dropped.includes(c)), ...made])];
  }
  if (node.data.kind === "ml_clean") return base;
  if (node.data.kind === "dl_tabular_forecast") return [...new Set([...base, "DL_Forecast_1"])];
  if (node.data.kind === "dl_recurrent_compare") return [...new Set([...base, "RNN_Pred"])];
  if (node.data.kind === "dl_sequence_classifier") return [...new Set([...base, "DL_Class"])];
  if (node.data.kind === "dl_anomaly_autoencoder") return [...new Set([...base, "AnomalyScore", "IsAnomaly"])];
  if (node.data.kind === "ml_split") {
    const feats = split(node.data.params.features);
    return [...(feats.length ? feats : base), "Target", "Split"];
  }
  if (["ml_train", "ml_evaluate"].includes(node.data.kind)) return [...new Set([...base, "Prediction", "Proba"])];
  if (node.data.kind === "ml_predict")
    return ["Split", "Actual", String(node.data.params.output_column ?? "Prediction"), "Proba", "Lower", "Upper", "Signal"];
  if (node.data.kind === "ml_visualize") return base;
  if (node.data.kind === "gen_windows") return [...base, "Gen_WindowStart"];
  if (node.data.kind === "gen_condition") return [...new Set([...base, "Gen_CondVol"])];
  if (node.data.kind === "gen_split") return [...new Set([...base, "Split"])];
  if (node.data.kind === "gen_evaluate") return [...new Set([...base, "Gen_VaR_cum", "Real_cum"])];
  if (node.data.kind.startsWith("gen_")) return base;
  if (node.data.kind === "ae_split") return [...split(node.data.params.features ?? ""), "y_bps", "Split"];
  if (node.data.kind === "ae_pca") return [...base, ...Array.from({ length: Number(node.data.params.components ?? 3) }, (_, i) => `PC${i + 1}`)];
  if (node.data.kind === "ae_ridge") return [...new Set([...base, "Z1", "Z2", "Z3", "Pred_bps"])];
  if (node.data.kind.startsWith("ae_")) return base;
  if (node.data.kind === "sup_target") return [...new Set([...base, "y_bps", ...(node.data.params.task === "direction" ? ["Direction"] : [])])];
  if (node.data.kind === "sup_split") return [...new Set([...base, "Split"])];
  if (node.data.kind === "sup_evaluate") return [...new Set([...base, "Prediction"])];
  if (node.data.kind.startsWith("sup_") || node.data.kind.startsWith("rl_")) return base;
  if (node.data.kind === "curve_data") return ["Close", "Coordinate"];
  if (node.data.kind === "yfinance_data")
    return [
      "Open", "High", "Low", "Close", "Volume",
      ...(node.data.params.auto_adjust === false ? ["AdjClose"] : []),
      ...(node.data.params.actions === true ? ["Dividends", "StockSplits"] : []),
    ];
  if (node.data.kind === "polymarket_dashboard")
    return ["mercado", "prob_líder", "cambio_24h", "volumen_24h", "liquidez", "spread", "horas_al_cierre"];
  if (node.data.kind === "polymarket_source")
    return ["P_up", "P_down", "Close", "SecondsToClose", "Book_spread", "Book_imbalance"];
  if (node.data.kind === "laya_updown")
    return [...new Set([...base, "UpDown_P_up", "UpDown_Decision"])];
  const laya: Partial<Record<Kind, string[]>> = {
    laya_signal: ["Laya_P_up", "Laya_P_down", "Laya_P_flat", "Laya_P_higher", "Laya_Risk", "Laya_Confidence"],
    laya_calibrate: ["Laya_P_raw", "Laya_P_cal"],
    laya_feedback: ["Feedback_P", "Position", "StrategyReturn", "Equity", "BuyHold", "RollingHitRate"],
  };
  if (laya[node.data.kind])
    return [...new Set([...base, ...(laya[node.data.kind] ?? [])])];
  if (node.data.kind === "source")
    return ["Open", "High", "Low", "Close", "Volume"];
  if (node.data.kind === "select_columns")
    return String(node.data.params.columns ?? "Close")
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
  if (node.data.kind === "coordinate")
    return [...base, String(node.data.params.output_column ?? "Coordinate")];
  if (["center", "scale_columns"].includes(node.data.kind))
    return [
      ...base,
      ...String(node.data.params.columns ?? "Close")
        .split(",")
        .map(
          (s) =>
            String(
              node.data.params.prefix ??
                (node.data.kind === "center" ? "Centered_" : "Scaled_"),
            ) + s.trim(),
        ),
    ];
  if (node.data.kind === "resample")
    return String(node.data.params.columns ?? "Close")
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
  if (node.data.kind === "quant_features")
    return [
      ...new Set([
        ...base,
        "SimpleReturn",
        "LogReturn",
        "CumulativeLogReturn",
        "RollingVolatility",
        "Drawdown",
        "RollingSharpe",
      ]),
    ];
  if (node.data.kind !== "features") return base;
  const p = effectiveParams(graph, id);
  const names = [
    ...base,
    "LogReturn",
    "CumulativeLogReturn",
    "Volatility",
    ...Array.from(
      { length: Math.max(0, Math.min(60, Number(p.lags ?? 5))) },
      (_, i) => `Lag_${i + 1}`,
    ),
  ];
  for (const [key, columns] of Object.entries({
    rsi: ["RSI"],
    ema: ["EMA"],
    sma: ["SMA"],
    macd: ["MACD", "MACDSignal", "MACDHistogram"],
    vwap: ["VWAP"],
    bb: ["BBMiddle", "BBUpper", "BBLower"],
  })) {
    if (p[`use_${key}`] !== false) names.push(...columns);
  }
  return [...new Set(names)];
}
export function inputHandles(kind: Kind): Record<string, string> {
  const primary = catalog[kind].input;
  const handles: Record<string, string> = primary ? { [primary]: primary } : {};
  if (
    [
      "features",
      "prepare",
      "model",
      "ml_model",
      "backtest",
      "curve_fit",
    ].includes(kind)
  )
    handles.config = "config";
  if (kind === "study") {
    handles.asset_b = "frame";
    handles.asset_c = "frame";
  }
  if (kind === "model" || kind === "curve_fit") handles.network = "network";
  if (kind === "dl_tabular_forecast" || kind === "dl_sequence_classifier" || kind === "dl_anomaly_autoencoder" || kind === "dl_recurrent_compare")
    handles.network = "network"; // optional: Network Input → Layers → Network Output replaces the hidden layers
  if (kind === "merge") handles.frame_b = "frame";
  if (kind === "polymarket_dashboard") handles.frame = "frame";   // optional: one market from Polymarket Market
  if (kind === "plot") {
    handles.frame = "frame";
    handles.tensor = "tensor";
    handles.portfolio = "portfolio";
  }
  if (kind === "risk") {
    handles.frame = "frame";
    handles.tensor = "tensor";
    handles.portfolio = "portfolio";
    handles.benchmark = "frame";
  }
  return handles;
}

export interface CustomPlot {
  id: string;
  title: string;
  origin?: string;
  colors?: Record<string, string>;
  labels?: Record<string, string>;
  chart_type: ChartType;
  y_scale?: "linear" | "log";
  scatter_columns?: string[];
  line_rows?: Record<string, string | number>[];
  source: string;
  x_key: string;
  series: string[];
  rows: Record<string, string | number>[];
}
export interface NetworkBinding {
  trainerId: string;
  inputId?: string;
  prepareId?: string;
  selected: string[];
  unavailable: string[];
  declared: number;
  summary: string;
  issue?: string;
}
export function effectiveParams(
  graph: Graph,
  id: string,
): Record<string, Param> {
  const node = graph.nodes.find((n) => n.id === id);
  if (!node) return {};
  const params = { ...node.data.params };
  const edge = graph.edges.find(
    (e) => e.target === id && e.targetHandle === "config",
  );
  const config = graph.nodes.find((n) => n.id === edge?.source);
  return config?.data.kind === "config"
    ? { ...params, ...config.data.params }
    : params;
}
export function studyArchitecture(params: Record<string, Param>): string {
  const count = String(params.feature_columns ?? "")
    .split(",")
    .filter((s) => s.trim()).length;
  const mlp = `${count} × ${params.lookback ?? 1} → ${String(
    params.widths ?? "64,32",
  )
    .split(",")
    .join(" ReLU → ")} ReLU → 1`;
  switch (params.topic) {
    case "cnn":
      return `${count} → Conv 32 → Conv 64 → pool → 1`;
    case "recurrent":
      return `${count} × ${params.lookback} → LSTM / GRU / Bi → 1`;
    case "autoencoders":
      return `${count} → 32 → ${params.latent_dim ?? 3} → 32 → ${count}`;
    case "generative":
    case "cdcgan":
      return `z ${params.latent_dim} + condition → ${params.horizon} returns`;
    case "double_dqn":
      return `${count} + position → 64 ReLU → 32 ReLU → 3 Q`;
    default:
      return mlp;
  }
}
export function networkBindings(graph: Graph): NetworkBinding[] {
  return graph.nodes
    .filter(
      (n) =>
        n.data.kind === "model" ||
        n.data.kind === "study" ||
        n.data.kind === "curve_fit",
    )
    .map((trainer) => {
      if (trainer.data.kind === "curve_fit") {
        const p = effectiveParams(graph, trainer.id);
        const count =
          p.variant === "polynomial_net" ? Number(p.degree ?? 5) : 1;
        const selected = Array.from(
          { length: Math.max(1, Math.min(10, count)) },
          (_, i) => (count === 1 ? "Coordinate" : `Coordinate^${i + 1}`),
        );
        const network = graph.edges.find(
          (e) => e.target === trainer.id && e.targetHandle === "network",
        );
        let cursor = network?.source;
        const visited = new Set<string>();
        const blocks: string[] = [];
        let input: BlueprintNode | undefined;
        let complete = false;
        while (cursor && !visited.has(cursor)) {
          visited.add(cursor);
          const node = graph.nodes.find((n) => n.id === cursor);
          if (!node) break;
          if (node.data.kind === "network_input") {
            input = node;
            break;
          }
          if (node.data.kind === "network_output") {
            blocks.unshift("1 output");
            complete = cursor === network?.source;
          }
          if (node.data.kind === "layer")
            blocks.unshift(
              node.data.params.layer_type === "Dense"
                ? `${node.data.params.units} ${node.data.params.activation}`
                : String(node.data.params.layer_type),
            );
          cursor = graph.edges.find(
            (e) => e.target === node.id && e.targetHandle === "network",
          )?.source;
        }
        const declared = Number(input?.data.params.input_features ?? count);
        const issue = !featureOptions(graph, trainer.id).includes("Coordinate")
          ? "Connect Monthly Curve Data first."
          : network && p.variant === "ols"
            ? "OLS does not use a network. Disconnect it or choose dense/polynomial_net."
            : network && (!input || !complete)
              ? "Connect Network Input → Layers → Network Output to Curve Fitting."
              : network && declared !== count
                ? `Input mismatch: selected ${count}, Network Input declares ${declared}. Enable automatic input size or correct the count.`
                : undefined;
        return {
          trainerId: trainer.id,
          inputId: input?.id,
          selected,
          unavailable: [],
          declared,
          summary: network
            ? [`${declared} inputs`, ...blocks].join(" → ")
            : p.variant === "dense"
              ? `1 → ${String(p.widths).split(",").join(` ${p.activation} → `)} ${p.activation} → 1`
              : `1 coordinate → polynomial ${p.degree} → 1`,
          issue,
        };
      }
      if (trainer.data.kind === "study") {
        const selected = String(trainer.data.params.feature_columns ?? "")
          .split(",")
          .map((s) => s.trim())
          .filter(Boolean);
        const available = featureOptions(graph, trainer.id);
        const unavailable = selected.filter((s) => !available.includes(s));
        return {
          trainerId: trainer.id,
          selected,
          unavailable,
          declared: selected.length,
          summary: studyArchitecture(trainer.data.params),
          issue: !selected.length
            ? "Select experiment input features."
            : unavailable.length
              ? `Unavailable selected features: ${unavailable.join(", ")}. Enable indicators or deselect them.`
              : undefined,
        };
      }
      const prepEdge = graph.edges.find(
        (e) => e.target === trainer.id && e.targetHandle === "tensor",
      );
      const prep = graph.nodes.find(
        (n) => n.id === prepEdge?.source && n.data.kind === "prepare",
      );
      const params = prep ? effectiveParams(graph, prep.id) : {};
      const available = prep ? featureOptions(graph, prep.id) : [];
      const raw = String(params.feature_columns ?? "").trim();
      const selected = raw
        ? raw
            .split(",")
            .map((n) => n.trim())
            .filter(Boolean)
        : available;
      const unavailable = selected.filter((name) => !available.includes(name));
      let cursor = graph.edges.find(
        (e) => e.target === trainer.id && e.targetHandle === "network",
      )?.source;
      const visited = new Set<string>();
      const blocks: string[] = [];
      let input: BlueprintNode | undefined;
      while (cursor && !visited.has(cursor)) {
        visited.add(cursor);
        const node = graph.nodes.find((n) => n.id === cursor);
        if (!node) break;
        if (node.data.kind === "network_input") {
          input = node;
          break;
        }
        if (node.data.kind === "layer")
          blocks.unshift(
            node.data.params.layer_type === "Dense"
              ? `${node.data.params.units} ${node.data.params.activation}`
              : String(node.data.params.layer_type),
          );
        if (node.data.kind === "network_output")
          blocks.unshift(`${node.data.params.output_units} output`);
        cursor = graph.edges.find(
          (e) => e.target === node.id && e.targetHandle === "network",
        )?.source;
      }
      const declared = Number(
        input?.data.params.input_features ?? selected.length,
      );
      const summary = input
        ? [`${declared} inputs`, ...blocks].join(" → ")
        : `${selected.length} features → ${String(trainer.data.params.architecture ?? "LSTM")} → 1 output`;
      const issue = !prep
        ? "Connect Series Preparation to the trainer."
        : unavailable.length
          ? `Unavailable selected features: ${unavailable.join(", ")}. Enable their indicators or deselect them.`
          : selected.length === 0
            ? "Select input features."
            : input && declared !== selected.length
              ? `Input mismatch: selected ${selected.length}, Network Input declares ${declared}. Enable automatic input size or correct the count.`
              : undefined;
      return {
        trainerId: trainer.id,
        inputId: input?.id,
        prepareId: prep?.id,
        selected,
        unavailable,
        declared,
        summary,
        issue,
      };
    });
}
export function syncNetworkInputs(graph: Graph): Graph {
  const bindings = networkBindings(graph);
  const updates = new Map<string, number>();
  for (const binding of bindings) {
    const node = graph.nodes.find((n) => n.id === binding.inputId);
    const counts = bindings
      .filter((b) => b.inputId === binding.inputId)
      .map((b) => b.selected.length);
    if (
      node &&
      node.data.params.auto_features !== false &&
      binding.selected.length > 0 &&
      counts.every((n) => n === binding.selected.length)
    )
      updates.set(node.id, binding.selected.length);
  }
  return {
    ...graph,
    nodes: graph.nodes.map((n) =>
      updates.has(n.id)
        ? {
            ...n,
            data: {
              ...n.data,
              params: { ...n.data.params, input_features: updates.get(n.id)! },
            },
          }
        : n,
    ),
  };
}

export const plotPalette: Record<string, string> = {
  Mint: "#6cc9ac",
  Violet: "#9c9af8",
  Orange: "#f2ad65",
  Blue: "#79b4ef",
  Pink: "#ec9dd6",
  Yellow: "#d5d77e",
  Coral: "#f07777",
  Cyan: "#56d9dd",
};
export interface MetricRow {
  metric_notes?: Record<string, string>;
  alpha_daily?: number | null;
  name: string;
  roi: number | null;
  cagr: number | null;
  var: number | null;
  cvar: number | null;
  sharpe: number | null;
  sortino: number | null;
  alpha: number | null;
  beta: number | null;
  volatility: number | null;
  max_drawdown: number | null;
  ols_r2: number | null;
  observations: number;
}
export interface RiskReport {
  id: string;
  title: string;
  scope: string;
  start: string;
  end: string;
  confidence: number;
  var_horizon: string;
  risk_free_rate: number;
  periods_per_year: number;
  benchmark: string | null;
  metrics: MetricRow[];
}
export interface PortfolioReport extends RiskReport {
  portfolio_metrics?: MetricRow[];
  asset_metrics?: MetricRow[];
  benchmark_metrics?: MetricRow;
  assets: string[];
  weights: Record<string, Record<string, number>>;
  estimation: {
    start: string;
    end: string;
    observations: number;
    covariance: string;
    long_only: boolean;
    max_weight: number;
  };
  rebalance: string;
  cost_bps_per_traded_notional: number;
  source: string;
  curves: Record<string, string | number>[];
}
export function portfolioExample(): Graph {
  const data = createNode("portfolio_data", 0, { x: 0, y: 150 });
  const allocation = createNode("portfolio", 1, { x: 320, y: 150 });
  const equity = createNode("plot", 2, { x: 650, y: 30 });
  equity.data.params = {
    ...equity.data.params,
    title: "Portfolio holdout · cumulative returns",
    plot_source: "equity",
    color: "Mint",
  };
  const weights = createNode("plot", 3, { x: 650, y: 300 });
  weights.data.params = {
    ...weights.data.params,
    title: "Markowitz allocations",
    plot_source: "weights",
    chart_type: "bar",
    color: "Violet",
  };
  return {
    nodes: [data, allocation, equity, weights],
    edges: [
      {
        id: "portfolio-data",
        source: data.id,
        target: allocation.id,
        sourceHandle: "portfolio",
        targetHandle: "portfolio",
      },
      {
        id: "portfolio-equity",
        source: allocation.id,
        target: equity.id,
        sourceHandle: "portfolio",
        targetHandle: "portfolio",
      },
      {
        id: "portfolio-weights",
        source: allocation.id,
        target: weights.id,
        sourceHandle: "portfolio",
        targetHandle: "portfolio",
      },
    ],
  };
}

export function presentPlot(
  plot: CustomPlot,
  node: BlueprintNode | undefined,
  index: number,
): CustomPlot {
  if (!node || node.data.kind !== "plot") return plot;
  const params = node.data.params;
  const names = Object.keys(plotPalette);
  const selected = String(params.color ?? "Auto");
  const offset =
    selected === "Auto"
      ? index % names.length
      : Math.max(0, names.indexOf(selected));
  const assigned = String(params.series_colors ?? "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
  const labels = String(params.series_labels ?? "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
  const title = String(params.title ?? "").trim();
  return {
    ...plot,
    title:
      title && title !== "Custom plot"
        ? title
        : `${plot.origin ?? "Input"} · ${plot.series.join(", ")}`,
    colors: Object.fromEntries(
      plot.series.map((key, i) => [
        key,
        assigned.length === plot.series.length &&
        assigned.every((name) => name in plotPalette)
          ? plotPalette[assigned[i]]
          : plotPalette[names[(offset + i) % names.length]],
      ]),
    ),
    labels:
      labels.length === plot.series.length
        ? Object.fromEntries(plot.series.map((key, i) => [key, labels[i]]))
        : Object.fromEntries(plot.series.map((key) => [key, key])),
  };
}

export interface UpDownDecision {
  id: string;
  decision: "UP" | "DOWN";
  probability_up: number;
  confidence: number;
  actionable: boolean;
  horizon: string;
  market: string | null;
  question?: string | null;
  as_of: string;
  p_market?: number;
  threshold?: number;
  weights: Record<string, number>;
  weight_source: string;
  signals: Record<string, number>;
  contributions?: Record<string, number>;
  laya: { used: boolean; via?: string; P_up?: number; P_down?: number };
  learning?: OnlineLearning | null;
  bucket?: number | null;
  bucket_s?: number;
  buckets_total?: number;
  elapsed_s?: number;
  window_start?: string | null;
  bucket_stats?: Record<string, string | number>[] | null;
}
export interface OnlineLearning {
  key: string;
  updates: number;
  pending: string[];
  weights: Record<string, number>;
  initial_weights: Record<string, number>;
  hit_rate: number | null;
  market_hit_rate: number | null;
  logloss: number | null;
  window: number;
  lr: number;
  learned_now: { slug: string; bucket?: number | null; p_up: number; outcome: string; hit: number }[];
  bucket?: number | null;
  bucket_updates?: number;
  by_bucket?: { bucket: number; n: number; hit_rate: number | null; market_hit_rate: number | null; logloss: number | null; weights: Record<string, number> }[];
  history: { slug: string; decided_at: string; p_up: number; decision: string; outcome: string; hit: number; logloss: number; weights: Record<string, number> }[];
}
export interface PolymarketSnapshot {
  slug: string;
  question: string;
  condition_id: string;
  up_outcome: string;
  down_outcome: string;
  price_up: number | null;
  price_down: number | null;
  window: string;
  start: string | null;
  end: string | null;
  closed: boolean;
  active: boolean;
  accepting_orders: boolean;
  volume: number;
  liquidity: number;
  volume24h: number;
  seconds_to_close: number | null;
  fetched_at: string;
  underlying?: Underlying | null;
  book: { best_bid: number | null; best_ask: number | null; spread: number | null; bid_depth: number; ask_depth: number; imbalance: number };
}
export interface Underlying {
  asset: string;
  source: string;
  price_to_beat: number | null;
  spot: number | null;
  spot_ts: number | null;
  diff: number | null;
  diff_pct: number | null;
  leading: "UP" | "DOWN" | null;
  signal?: { z: number; diff_pct: number; sigma_pct_min: number; minutes_left: number; source: string } | null;
  close: number | null;
  completed: boolean;
  history: { t: number; p: number }[];
}
export interface DashboardRow {
  [key: string]: string | number | boolean | null;
}
export interface MarketDashboardData {
  id: string;
  mode: "market";
  fetched_at: string;
  seconds: number;
  market: {
    slug: string; question: string; event_title: string | null; event_slug: string | null; outcomes: string[]; outcome_prices: number[];
    up_outcome: string; down_outcome: string; window: string | null; start: string | null; end: string | null; closed: boolean;
    accepting_orders: boolean; resolution_source: string | null; description: string; condition_id: string; url: string; underlying?: Underlying | null;
  };
  kpis: Record<string, number | null>;
  history: { hora: string; P: number }[];
  depth: { precio: number; compra?: number; venta?: number }[];
  book: Record<string, { bids: { precio: number; acciones: number; usd: number }[]; asks: { precio: number; acciones: number; usd: number }[] }>;
  trades: DashboardRow[];
  trade_flow: DashboardRow[];
  vwap: Record<string, number>;
  holders: DashboardRow[];
  siblings: DashboardRow[];
  decision: UpDownDecision | null;
  notes: string[];
}
export interface PolymarketDashboardData {
  id: string;
  mode?: "overview";
  kpis: Record<string, number | string | null>;
  filters: Record<string, string | number>;
  fetched_at: string;
  seconds: number;
  categories: DashboardRow[];
  top_events: DashboardRow[];
  top_markets: DashboardRow[];
  movers: DashboardRow[];
  ending_soon: DashboardRow[];
  contested: DashboardRow[];
  most_liquid: DashboardRow[];
  wide_spreads: DashboardRow[];
  probability_histogram: DashboardRow[];
  updown: DashboardRow[];
}
