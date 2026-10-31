import type { Edge, Node } from "@xyflow/react";
export type Kind =
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
  | "plot";
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
  plots?: CustomPlot[];
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
  input?: "frame" | "tensor" | "network";
  output?: "frame" | "tensor" | "config" | "network";
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
      { key: "title", label: "Chart title", value: "Custom plot" },
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
  if (node.data.kind === "source")
    return ["Open", "High", "Low", "Close", "Volume"];
  if (node.data.kind !== "features") return base;
  const p = effectiveParams(graph, id);
  const names = [
    ...base,
    "LogReturn",
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
  if (["features", "prepare", "model", "backtest"].includes(kind))
    handles.config = "config";
  if (kind === "model") handles.network = "network";
  if (kind === "merge") handles.frame_b = "frame";
  if (kind === "plot") {
    handles.frame = "frame";
    handles.tensor = "tensor";
  }
  return handles;
}

export interface CustomPlot {
  id: string;
  title: string;
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
export function networkBindings(graph: Graph): NetworkBinding[] {
  return graph.nodes
    .filter((n) => n.data.kind === "model")
    .map((trainer) => {
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
