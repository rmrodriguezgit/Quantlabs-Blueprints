import { useCallback } from "react";
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
  catalog,
  createNode,
  type BlueprintNode,
  type Graph,
  type Kind,
} from "../types/blueprint";
import "@xyflow/react/dist/style.css";
const icons = {
  source: Database,
  features: SlidersHorizontal,
  prepare: Layers3,
  model: BrainCircuit,
  backtest: ChartNoAxesCombined,
  config: Settings2,
};
function VisualNode({ data, selected }: NodeProps<BlueprintNode>) {
  const def = catalog[data.kind];
  const Icon = icons[data.kind];
  return (
    <div
      className={`blueprint-node ${selected ? "selected" : ""}`}
      style={{ "--node-color": def.color } as React.CSSProperties}
    >
      <div className="node-category">
        {def.category}
        <span>•••</span>
      </div>
      <div className="node-title">
        <Icon size={19} />
        <strong>{data.label}</strong>
      </div>
      <p>{def.subtitle}</p>
      <div className="node-summary">
        {data.kind === "source"
          ? `${String(data.params.ticker)} · ${String(data.params.years)}Y · ${String(data.params.source)}`
          : data.kind === "model"
            ? `${String(data.params.architecture)} / ${String(data.params.hidden)} units`
            : data.kind === "prepare"
              ? `${String(data.params.lookback)} bars → ${String(data.params.target)}`
              : data.kind === "features"
                ? `${String(data.params.lags)} lags / RSI ${String(data.params.rsi_window)}`
                : data.kind === "backtest"
                  ? `Threshold ${String(data.params.threshold)}`
                  : "Overrides connected parameters"}
      </div>
      <div className="node-pins">
        {def.input ? (
          <span>
            <i className={def.input} />
            {def.input === "frame" ? "DataFrame" : "Tensor bundle"}
          </span>
        ) : (
          <span />
        )}
        {def.output && (
          <span>
            {def.output === "frame"
              ? "DataFrame"
              : def.output === "config"
                ? "Config"
                : "Tensor bundle"}
            <i className={def.output} />
          </span>
        )}
      </div>
      {def.input && (
        <Handle
          type="target"
          position={Position.Left}
          id={def.input}
          className={def.input}
          style={{ top: "82%" }}
        />
      )}
      {def.output && (
        <Handle
          type="source"
          position={Position.Right}
          id={def.output}
          className={def.output}
          style={{ top: "82%" }}
        />
      )}
      {def.input && (
        <>
          <Handle
            type="target"
            position={Position.Top}
            id="config"
            className="config"
          />
          <span className="config-label">config</span>
        </>
      )}
    </div>
  );
}
const nodeTypes = { blueprint: VisualNode };
export function Palette() {
  return (
    <aside className="palette">
      <div className="aside-heading">
        NODE LIBRARY <span>06</span>
      </div>
      <p className="muted">Drag a node onto the canvas</p>
      {(Object.keys(catalog) as Kind[]).map((kind) => {
        const def = catalog[kind];
        const Icon = icons[kind];
        return (
          <div
            key={kind}
            draggable
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
      connection.sourceHandle !== connection.targetHandle
    )
      return false;
    if (
      graph.edges.some(
        (e) =>
          e.target === dst.id && e.targetHandle === connection.targetHandle,
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
        if (!(kind in catalog)) return;
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
