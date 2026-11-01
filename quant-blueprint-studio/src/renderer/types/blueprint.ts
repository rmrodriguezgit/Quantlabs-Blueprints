import type { Edge, Node } from "@xyflow/react";
export type Kind =
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
  studies?: StudyReport[];
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
export interface StudyReport {
  id: string;
  title: string;
  topic: string;
  scope?: string;
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
    chart_type: "line" | "area" | "bar";
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
    label: "Learning Experiment",
    subtitle: "Validation selection · isolated holdout",
    category: "06 / EDUCATION",
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
    subtitle: "Select RSI · EMA · SMA · MACD · VWAP · BB",
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
          "Autoencoder",
          "DAE",
          "DQN",
        ],
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
        options: ["Adam", "SGD"],
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
        options: ["ReLU", "Tanh", "GELU", "Sigmoid", "LeakyReLU", "Identity"],
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
        options: ["line", "area", "bar"],
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
      execute: (graph: Graph) => Promise<unknown>;
      onProgress: (callback: (event: unknown) => void) => () => void;
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
  if (node.data.kind === "curve_data") return ["Close", "Coordinate"];
  if (node.data.kind === "source")
    return ["Open", "High", "Low", "Close", "Volume"];
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
  if (["features", "prepare", "model", "backtest", "curve_fit"].includes(kind))
    handles.config = "config";
  if (kind === "study") {
    handles.asset_b = "frame";
    handles.asset_c = "frame";
  }
  if (kind === "model") handles.network = "network";
  if (kind === "merge") handles.frame_b = "frame";
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
  chart_type: "line" | "area" | "bar";
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
        const p = trainer.data.params;
        return {
          trainerId: trainer.id,
          selected: ["Coordinate"],
          unavailable: [],
          declared: 1,
          summary:
            p.variant === "dense"
              ? `1 → ${String(p.widths).split(",").join(` ${p.activation} → `)} ${p.activation} → 1`
              : `1 coordinate → polynomial ${p.degree} → 1`,
          issue: featureOptions(graph, trainer.id).includes("Coordinate")
            ? undefined
            : "Connect Monthly Curve Data first.",
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
