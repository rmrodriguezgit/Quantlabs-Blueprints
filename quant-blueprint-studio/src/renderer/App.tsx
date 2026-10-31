import { version } from "../../package.json";
import { useEffect, useRef, useState } from "react";
import { ReactFlowProvider } from "@xyflow/react";
import {
  Activity,
  Play,
  Download,
  Upload,
  RotateCcw,
  CheckCircle2,
  CircleAlert,
} from "lucide-react";
import { BlueprintCanvas, Palette } from "./components/BlueprintCanvas";
import { DetailsInspector } from "./components/DetailsInspector";
import { executeGraph } from "./engine";
import {
  ExecutionPanel,
  type ExecutionEntry,
} from "./components/ExecutionPanel";
import { PlotDock } from "./components/PlotDock";
import { PerformanceDock } from "./components/PerformanceDock";
import {
  catalog,
  defaultGraph,
  networkExample,
  networkBindings,
  syncNetworkInputs,
  featureOptions,
  createNode,
  type Graph,
  type Param,
  type Result,
  type Kind,
} from "./types/blueprint";
function isGraph(value: unknown): value is Graph {
  if (typeof value !== "object" || value === null) return false;
  const g = value as Partial<Graph>;
  return (
    Array.isArray(g.nodes) &&
    g.nodes.length > 0 &&
    g.nodes.length <= 50 &&
    g.nodes.every(
      (n) =>
        typeof n.id === "string" &&
        n.type === "blueprint" &&
        n.data &&
        Object.hasOwn(catalog, n.data.kind) &&
        typeof n.data.params === "object" &&
        n.data.params !== null &&
        Object.values(n.data.params).every((v) =>
          ["string", "number", "boolean"].includes(typeof v),
        ) &&
        n.position &&
        Number.isFinite(n.position.x) &&
        Number.isFinite(n.position.y),
    ) &&
    new Set(g.nodes.map((n) => n.id)).size === g.nodes.length &&
    Array.isArray(g.edges) &&
    g.edges.length <= 150 &&
    g.edges.every(
      (e) =>
        typeof e.id === "string" &&
        typeof e.source === "string" &&
        typeof e.target === "string",
    )
  );
}
function isResult(value: unknown): value is Result {
  if (typeof value !== "object" || !value) return false;
  const r = value as Partial<Result>;
  return (
    Array.isArray(r.predictions) &&
    Array.isArray(r.decomposition) &&
    Array.isArray(r.backtest) &&
    Array.isArray(r.history) &&
    typeof r.metrics === "object" &&
    r.metrics !== null
  );
}
function graphSignature(graph: Graph): string {
  return JSON.stringify({
    nodes: graph.nodes.map(({ id, data }) => ({ id, data })),
    edges: graph.edges.map(
      ({ source, target, sourceHandle, targetHandle }) => ({
        source,
        target,
        sourceHandle,
        targetHandle,
      }),
    ),
  });
}
function loadGraph(): Graph {
  try {
    const saved = JSON.parse(
      localStorage.getItem("quant-blueprint") ?? "null",
    ) as unknown;
    if (isGraph(saved)) return syncNetworkInputs(saved);
  } catch {
    /* Start with the included pipeline if local storage is unavailable. */
  }
  return defaultGraph();
}
export default function App() {
  const [graph, setGraph] = useState<Graph>(loadGraph);
  const [selected, setSelected] = useState<string | null>(null);
  const [result, setResult] = useState<Result | null>(null);
  const [running, setRunning] = useState(false);
  const [message, setMessage] = useState(
    "Ready. Synthetic data pipeline loaded.",
  );
  const [error, setError] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const [entries, setEntries] = useState<ExecutionEntry[]>([]);
  const [elapsed, setElapsed] = useState(0);
  const started = useRef(0);
  const bindings = networkBindings(graph);
  useEffect(() => {
    if (!running) return;
    const timer = window.setInterval(
      () => setElapsed(Math.floor((Date.now() - started.current) / 1000)),
      1000,
    );
    return () => window.clearInterval(timer);
  }, [running]);
  const updateGraph = (update: (g: Graph) => Graph) => {
    setGraph((g) => {
      const next = syncNetworkInputs(update(g));
      if (graphSignature(next) !== graphSignature(g)) {
        setResult(null);
        setEntries([]);
        setError(false);
        setElapsed(0);
        setMessage("Graph changed. Ready to execute.");
      }
      try {
        localStorage.setItem("quant-blueprint", JSON.stringify(next));
      } catch {
        /* In-memory editing remains available. */
      }
      return next;
    });
  };
  const execute = async () => {
    const issue = bindings.find((binding) => binding.issue);
    if (issue) {
      setError(true);
      setMessage(issue.issue ?? "Input configuration mismatch.");
      setSelected(issue.inputId ?? issue.prepareId ?? issue.trainerId);
      return;
    }
    started.current = Date.now();
    setEntries([]);
    setElapsed(0);
    setRunning(true);
    setError(false);
    setResult(null);
    setMessage("Executing DAG · preparing data and training model…");
    try {
      const value = await executeGraph(graph, (event) => {
        const label =
          graph.nodes.find((n) => n.id === event.node)?.data.label ??
          event.node;
        setEntries((previous) => [
          ...previous,
          { ...event, label, time: (Date.now() - started.current) / 1000 },
        ]);
        if (event.status === "running") setMessage(`${label}: calculating…`);
        if (event.status === "epoch")
          setMessage(`${label}: epoch ${event.epoch}/${event.epochs}`);
        if (event.status === "error") setSelected(event.node);
      });
      if (!isResult(value))
        throw new Error("Engine returned an invalid result.");
      setResult(value);
      setMessage(
        `Completed ${graph.nodes.length} nodes · ${value.features?.length ?? "auto"} input features · ${value.predictions.length} out-of-sample observations`,
      );
    } catch (e) {
      setError(true);
      setMessage(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  };
  const exportGraph = () => {
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(graph, null, 2)], { type: "application/json" }),
    );
    const a = document.createElement("a");
    a.href = url;
    a.download = "research-blueprint.json";
    a.click();
    URL.revokeObjectURL(url);
  };
  return (
    <div className="studio">
      <header className="app-header">
        <div className="brand-icon">
          <Activity size={23} />
        </div>
        <div className="brand">
          <strong>
            QUANTLABS<span> / BLUEPRINTS</span>
          </strong>
          <small>VISUAL QUANTITATIVE RESEARCH STUDIO</small>
        </div>
        <span className="workspace-tag">LOCAL WORKSPACE</span>
        <div className="header-right">
          <span className="version">v{version}</span>
          <div className="engine-indicator">
            <span className="live-dot" />
            Python engine
          </div>
        </div>
      </header>
      <div className="toolbar">
        <div>
          <strong>Untitled research</strong>
          <span className="toolbar-divider">/</span>
          <span>Blueprint graph</span>
        </div>
        <div className="toolbar-actions">
          <button
            disabled={running}
            onClick={() => {
              updateGraph(() => networkExample());
              setSelected(null);
              setError(false);
              setMessage(
                "Example loaded: 16 features → Dense 64 ReLU → Dense 32 ReLU → 1 output.",
              );
            }}
          >
            Example: 16 → 64 → 32 → 1
          </button>
          <button
            disabled={running}
            onClick={() => {
              updateGraph(() => defaultGraph());
              setSelected(null);
              setMessage("Default pipeline restored.");
              setError(false);
            }}
            title="Reset graph"
          >
            <RotateCcw size={15} />
            Reset
          </button>
          <button disabled={running} onClick={() => input.current?.click()}>
            <Upload size={15} />
            Import
          </button>
          <button
            title="Save nodes, connections and parameters to a JSON file"
            onClick={exportGraph}
          >
            <Download size={15} />
            Export
          </button>
          <button
            className="execute-button"
            disabled={running}
            onClick={() => void execute()}
          >
            <Play size={15} fill="currentColor" />
            {running ? "Executing…" : "Execute Blueprint"}
          </button>
        </div>
        <input
          ref={input}
          type="file"
          accept="application/json,.json"
          hidden
          onChange={async (e) => {
            const file = e.target.files?.[0];
            e.target.value = "";
            if (!file) return;
            try {
              if (file.size > 1000000)
                throw new Error("Graph file exceeds 1 MB.");
              const value: unknown = JSON.parse(await file.text());
              if (!isGraph(value)) throw new Error("Invalid blueprint graph.");
              updateGraph(() => value);
              setSelected(null);
              setError(false);
              setMessage("Blueprint imported.");
            } catch (err) {
              setError(true);
              setMessage(err instanceof Error ? err.message : String(err));
            }
          }}
        />
      </div>
      <ExecutionPanel
        bindings={bindings}
        entries={entries}
        running={running}
        error={error}
        message={message}
        elapsed={elapsed}
        total={graph.nodes.length}
        onInspect={setSelected}
      />
      <main className={running ? "workspace busy" : "workspace"}>
        <Palette
          disabled={running}
          onAdd={(kind) => {
            const node = createNode(kind, graph.nodes.length, {
              x: 300 + (graph.nodes.length % 4) * 50,
              y: 150 + (graph.nodes.length % 4) * 50,
            });
            updateGraph((g) => ({ ...g, nodes: [...g.nodes, node] }));
            setSelected(node.id);
          }}
        />
        <ReactFlowProvider>
          <BlueprintCanvas
            graph={graph}
            setGraph={updateGraph}
            onSelect={setSelected}
            disabled={running}
          />
        </ReactFlowProvider>
        <DetailsInspector
          node={graph.nodes.find((n) => n.id === selected)}
          availableFeatures={selected ? featureOptions(graph, selected) : []}
          onDuplicate={() => {
            const original = graph.nodes.find((n) => n.id === selected);
            if (!original) return;
            const node = {
              ...original,
              id: crypto.randomUUID(),
              position: {
                x: original.position.x + 45,
                y: original.position.y + 90,
              },
              selected: false,
              data: { ...original.data, params: { ...original.data.params } },
            };
            updateGraph((g) => ({ ...g, nodes: [...g.nodes, node] }));
            setSelected(node.id);
          }}
          disabled={running}
          onChange={(key: string, value: Param) =>
            updateGraph((g) => ({
              ...g,
              nodes: g.nodes.map((n) =>
                n.id === selected
                  ? {
                      ...n,
                      data: {
                        ...n.data,
                        params: { ...n.data.params, [key]: value },
                      },
                    }
                  : n,
              ),
            }))
          }
          onDelete={() => {
            updateGraph((g) => ({
              ...g,
              nodes: g.nodes.filter((n) => n.id !== selected),
              edges: g.edges.filter(
                (e) => e.source !== selected && e.target !== selected,
              ),
            }));
            setSelected(null);
          }}
        />
      </main>
      <PerformanceDock result={result} running={running} />
      <PlotDock plots={result?.plots ?? []} />
      <footer className={`status-bar ${error ? "error" : ""}`}>
        {error ? <CircleAlert size={13} /> : <CheckCircle2 size={13} />}
        <span>{message}</span>
        <small>
          LOCAL COMPUTE <span>•</span>{" "}
          {graph.nodes.some(
            (n) =>
              n.data.kind === ("source" satisfies Kind) &&
              n.data.params.source === "yfinance",
          )
            ? "YFINANCE DATA"
            : "SYNTHETIC DATA"}
        </small>
      </footer>
    </div>
  );
}
