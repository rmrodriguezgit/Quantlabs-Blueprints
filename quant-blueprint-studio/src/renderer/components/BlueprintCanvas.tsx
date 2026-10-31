import { useCallback, useEffect } from "react";
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
const icons = {
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
  const p = data.params;
  switch (data.kind) {
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
function VisualNode({ data, selected }: NodeProps<BlueprintNode>) {
  const def = catalog[data.kind];
  const Icon = icons[data.kind];
  return (
    <div
      className={`blueprint-node ${selected ? "selected" : ""}`}
      style={
        {
          "--node-color":
            data.kind === "plot"
              ? (plotPalette[String(data.params.color)] ?? def.color)
              : def.color,
        } as React.CSSProperties
      }
    >
      <div className="node-category">
        {def.category}
        <span>•••</span>
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
        <span>{def.output ?? "report"}</span>
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
              type="target"
              position={
                id === "config" || id === "benchmark"
                  ? Position.Top
                  : id === "network" && data.kind === "model"
                    ? Position.Bottom
                    : Position.Left
              }
              id={id}
              className={type}
              style={
                id === "config" ||
                id === "benchmark" ||
                (id === "network" && data.kind === "model")
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
            {id === "network" && data.kind === "model" && (
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
  disabled,
}: {
  onAdd: (kind: Kind) => void;
  disabled: boolean;
}) {
  return (
    <aside className="palette">
      <div className="aside-heading">
        NODE LIBRARY <span>{Object.keys(catalog).length}</span>
      </div>
      <p className="muted">Click or drag a node onto the canvas</p>
      {(Object.keys(catalog) as Kind[]).map((kind) => {
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
              <small>{def.subtitle}</small>
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
      <div className="library-note">
        <Layers3 size={20} />
        <strong>Build your research pipeline</strong>
        <p>Connect typed pins, configure nodes, then execute the graph.</p>
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
  const valid = (connection: Connection | Edge): boolean => {
    const src = graph.nodes.find((n) => n.id === connection.source);
    const dst = graph.nodes.find((n) => n.id === connection.target);
    if (
      !src ||
      !dst ||
      src.id === dst.id ||
      !connection.targetHandle ||
      !connection.sourceHandle ||
      inputHandles(dst.data.kind)[connection.targetHandle] !==
        connection.sourceHandle
    )
      return false;
    if (
      graph.edges.some(
        (e) =>
          e.target === dst.id && e.targetHandle === connection.targetHandle,
      )
    )
      return false;
    if (
      dst.data.kind === "plot" &&
      graph.edges.some((e) => e.target === dst.id)
    )
      return false;
    if (
      dst.data.kind === "risk" &&
      connection.targetHandle !== "benchmark" &&
      graph.edges.some(
        (e) => e.target === dst.id && e.targetHandle !== "benchmark",
      )
    )
      return false;
    const visited = new Set<string>();
    const reaches = (id: string): boolean => {
      if (id === src.id) return true;
      if (visited.has(id)) return false;
      visited.add(id);
      return graph.edges
        .filter((e) => e.source === id)
        .some((e) => reaches(e.target));
    };
    return !reaches(dst.id);
  };
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
      <ReactFlow<BlueprintNode>
        nodes={graph.nodes}
        edges={graph.edges}
        nodeTypes={nodeTypes}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={(c) =>
          setGraph((g) => ({
            ...g,
            edges: addEdge({ ...c, animated: true }, g.edges),
          }))
        }
        isValidConnection={valid}
        onNodeClick={(_, node) => onSelect(node.id)}
        onPaneClick={() => onSelect(null)}
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
      <div className="canvas-foot">
        {graph.nodes.length} nodes <span>·</span> {graph.edges.length}{" "}
        connections <span>·</span> DAG execution
      </div>
    </div>
  );
}
