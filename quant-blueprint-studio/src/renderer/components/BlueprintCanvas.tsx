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
import { validConnection } from "../connections";
const icons = {
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
  const p = data.params;
  switch (data.kind) {
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
          aria-label={`Salida ${def.output}`}
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
