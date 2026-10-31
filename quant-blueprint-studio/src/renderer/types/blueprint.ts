import type { Edge, Node } from "@xyflow/react";
export type Kind =
  "source" | "features" | "prepare" | "model" | "backtest" | "config";
export type Param = string | number | boolean;
export type BlueprintData = {
  label: string;
  kind: Kind;
  params: Record<string, Param>;
};
export type BlueprintNode = Node<BlueprintData, "blueprint">;
export type Graph = { nodes: BlueprintNode[]; edges: Edge[] };
export interface Result {
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
  target?: string;
  trading?: {
    total_return: number;
    max_drawdown: number;
    sharpe: number;
    turnover: number;
  };
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
  input?: "frame" | "tensor";
  output?: "frame" | "tensor" | "config";
  fields: Field[];
}
export const catalog: Record<Kind, Definition> = {
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
      { key: "years", label: "History / years", value: 3, min: 1, max: 10 },
    ],
  },
  features: {
    label: "Technical Features",
    subtitle: "Returns · lags · momentum",
    category: "02 / TRANSFORM",
    color: "#e6b86a",
    input: "frame",
    output: "frame",
    fields: [
      { key: "lags", label: "Lag features", value: 5, min: 1, max: 60 },
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
        min: 2,
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
    edges: nodes
      .slice(1)
      .map((n, i) => ({
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
    quantEngine?: { execute: (graph: Graph) => Promise<unknown> };
  }
}
