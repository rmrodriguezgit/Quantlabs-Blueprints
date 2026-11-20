import { BlueprintLibrary } from "./components/BlueprintLibrary";
import { ScalingDock } from "./components/ScalingDock";
import { StudyDock } from "./components/StudyDock";
import {
  isGraph,
  parseLibrary,
  downloadLibrary,
  readSaved,
  storeSaved,
  type LibraryEntry,
} from "./library";
import { version } from "../../package.json";
import { useEffect, useMemo, useRef, useState } from "react";
import { ReactFlowProvider } from "@xyflow/react";
import {
  Activity,
  Play,
  Download,
  Upload,
  RotateCcw,
  CheckCircle2,
  CircleAlert,
  Sparkles,
} from "lucide-react";
import { BlueprintCanvas, IssueContext, Palette } from "./components/BlueprintCanvas";
import { BrainPanel } from "./components/BrainPanel";
import { ValidationPanel } from "./components/ValidationPanel";
import { applyFix, repairGraph, validateGraph, type Issue } from "./validation";
import { DetailsInspector } from "./components/DetailsInspector";
import { executeGraph } from "./engine";
import {
  ExecutionPanel,
  type ExecutionEntry,
} from "./components/ExecutionPanel";
import { PortfolioDock } from "./components/PortfolioDock";
import { PlotDock } from "./components/PlotDock";
import { PerformanceDock } from "./components/PerformanceDock";
import {
  catalog,
  defaultGraph,
  networkExample,
  portfolioExample,
  plotPalette,
  presentPlot,
  networkBindings,
  syncNetworkInputs,
  featureOptions,
  createNode,
  type Graph,
  type Param,
  type Result,
  type Kind,
} from "./types/blueprint";
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
    nodes: graph.nodes.map(({ id, data }) => ({
      id,
      data:
        data.kind === "plot"
          ? {
              ...data,
              params: Object.fromEntries(
                Object.entries(data.params).filter(
                  ([key]) =>
                    ![
                      "title",
                      "color",
                      "series_colors",
                      "series_labels",
                    ].includes(key),
                ),
              ),
            }
          : data,
    })),
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
  const [libraryOpen, setLibraryOpen] = useState(false);
  const [title, setTitle] = useState(
    () => localStorage.getItem("quant-blueprint-title") ?? "Untitled research",
  );
  useEffect(() => {
    localStorage.setItem("quant-blueprint-title", title);
  }, [title]);
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
  const [brainOpen, setBrainOpen] = useState(false);
  const [repairChanges, setRepairChanges] = useState<string[]>([]);
  const undo = useRef<Graph[]>([]);
  const issues = useMemo(() => validateGraph(graph), [graph]);
  const nodeIssues = useMemo(() => {
    const map: Record<string, { severity: "error" | "warning"; count: number; first: string }> = {};
    for (const issue of issues) {
      if (!issue.node || issue.severity === "info") continue;
      const current = map[issue.node];
      if (!current) map[issue.node] = { severity: issue.severity, count: 1, first: issue.message };
      else {
        current.count++;
        if (issue.severity === "error" && current.severity !== "error") {
          current.severity = "error";
          current.first = issue.message;
        }
      }
    }
    return map;
  }, [issues]);
  const fixIssue = (issue: Issue) => {
    if (!issue.fix) return;
    undo.current = [...undo.current.slice(-19), graph];
    const fix = issue.fix;
    updateGraph((g) => applyFix(g, fix));
    setRepairChanges([`${issue.fixLabel ?? "Corregido"} — ${issue.message}`]);
  };
  const repairAll = () => {
    const repaired = repairGraph(graph);
    if (!repaired.changes.length) return;
    undo.current = [...undo.current.slice(-19), graph];
    updateGraph(() => repaired.graph);
    setRepairChanges(repaired.changes);
    setError(repaired.remaining.some((i) => i.severity === "error"));
    setMessage(
      `Reparación: ${repaired.changes.length} cambio(s). ${repaired.remaining.filter((i) => i.severity === "error").length} error(es) requieren tu decisión.`,
    );
  };
  const undoRepair = () => {
    const previous = undo.current.pop();
    if (!previous) return;
    updateGraph(() => previous);
    setRepairChanges([]);
    setMessage("Reparación deshecha.");
  };
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
  const execute = async (executionGraph: Graph = graph) => {
    if (!executionGraph.nodes.length) {
      setMessage("Add nodes before executing the blueprint.");
      return;
    }
    const blocking = validateGraph(executionGraph).filter((i) => i.severity === "error");
    if (blocking.length) {
      setError(true);
      setMessage(
        `${blocking.length} error(es) de validación antes de ejecutar: ${blocking[0].message} Usa Reparar o corrige el nodo marcado.`,
      );
      if (blocking[0].node) setSelected(blocking[0].node);
      return;
    }
    const issue = networkBindings(executionGraph).find(
      (binding) => binding.issue,
    );
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
      const value = await executeGraph(executionGraph, (event) => {
        const label =
          executionGraph.nodes.find((n) => n.id === event.node)?.data.label ??
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
      // The final result confirms every node completed; IPC progress can arrive later.
      setEntries((previous) => {
        const completed = new Set(
          previous
            .filter((event) => event.status === "complete")
            .map((event) => event.node),
        );
        return [
          ...previous,
          ...executionGraph.nodes
            .filter((node) => !completed.has(node.id))
            .map((node) => ({
              node: node.id,
              label: node.data.label,
              status: "complete" as const,
              time: (Date.now() - started.current) / 1000,
            })),
        ];
      });
      setMessage(
        value.portfolios?.length
          ? `Completed ${executionGraph.nodes.length} nodes · ${value.portfolios.length} portfolio report(s) · ${value.portfolios[0].metrics[0]?.observations ?? 0} holdout observations`
          : value.studies?.some((s) => s.topic === "curve_fit")
            ? `Completed ${executionGraph.nodes.length} nodes · ${value.studies.length} curve benchmarks · ${value.studies[0].scope ?? "curve fitting"}`
            : value.studies?.length
              ? `Completed ${executionGraph.nodes.length} nodes · ${value.studies[0].selected} selected · ${value.studies[0].splits.test} holdout samples`
              : value.risk_reports?.length
                ? `Completed ${executionGraph.nodes.length} nodes · ${value.risk_reports.length} risk report(s)`
                : `Completed ${executionGraph.nodes.length} nodes · ${value.features?.length ?? "auto"} input features · ${value.predictions.length} out-of-sample observations`,
      );
    } catch (e) {
      setError(true);
      setMessage(e instanceof Error ? e.message : String(e));
    } finally {
      setElapsed(Math.ceil((Date.now() - started.current) / 1000));
      setRunning(false);
    }
  };
  const currentEntry = (): LibraryEntry => ({
    id: crypto.randomUUID(),
    title,
    area: "Personal",
    description: "Editable research blueprint",
    instructions: ["Configure nodes and execute."],
    graph: structuredClone(graph),
  });
  const exportGraph = () =>
    downloadLibrary([currentEntry()], "research-blueprint.json");
  const loadEntry = (entry: LibraryEntry, run: boolean) => {
    try {
      const previous = readSaved();
      if (
        graph.nodes.length > 0 &&
        !previous.some((e) => JSON.stringify(e.graph) === JSON.stringify(graph))
      )
        storeSaved([
          ...previous,
          {
            ...currentEntry(),
            title: `${title} · autosave ${new Date().toLocaleString()}`,
          },
        ]);
      const next = syncNetworkInputs(structuredClone(entry.graph));
      updateGraph(() => next);
      setTitle(entry.title);
      setSelected(null);
      setLibraryOpen(false);
      setMessage(
        next.nodes.length
          ? `${entry.title}: loaded. Edit parameters or execute.`
          : "New blueprint. Click or drag a node from the library to begin.",
      );
      if (run) void execute(next);
    } catch (e) {
      setError(true);
      setMessage(e instanceof Error ? e.message : String(e));
    }
  };
  const newBlueprint = () => {
    if (
      !window.dispatchEvent(
        new Event("quantlabs:before-new", { cancelable: true }),
      )
    )
      return;
    if (running) {
      setMessage(
        "Wait for the current execution before creating a new blueprint.",
      );
      return;
    }
    loadEntry(
      {
        id: crypto.randomUUID(),
        title: "Untitled research",
        area: "Personal",
        description: "Blank research blueprint",
        instructions: ["Add and connect nodes, then execute."],
        graph: { nodes: [], edges: [] },
      },
      false,
    );
  };
  useEffect(
    () => window.quantEngine?.onNewBlueprint?.(newBlueprint),
    [graph, title, running],
  );
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
          <strong>{title}</strong>
          <span className="toolbar-divider">/</span>
          <span>Blueprint graph</span>
        </div>
        <div className="toolbar-actions">
          <button
            disabled={running}
            onClick={newBlueprint}
            title="New empty blueprint; archive current design in Personal"
          >
            New
          </button>
          <button disabled={running} onClick={() => setLibraryOpen(true)}>
            Biblioteca / ejemplos
          </button>
          <button
            className="brain-button"
            disabled={running}
            onClick={() => setBrainOpen(true)}
            title="Describe el análisis y el Cerebro construye un blueprint validado"
          >
            <Sparkles size={15} />
            Cerebro IA
          </button>
          <button
            disabled={running}
            onClick={() => {
              updateGraph(() => portfolioExample());
              setSelected(null);
            }}
          >
            Portfolio / Markowitz example
          </button>
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
            disabled={running || !graph.nodes.length}
            onClick={() => void execute(graph)}
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
              if (file.size > 10000000)
                throw new Error("Blueprint file exceeds 10 MB.");
              const value: unknown = JSON.parse(await file.text());
              const imported = parseLibrary(value);
              if (imported.length === 1) {
                loadEntry(imported[0], false);
                return;
              } else {
                storeSaved([
                  ...readSaved(),
                  ...imported.map((e) => ({
                    ...e,
                    id: crypto.randomUUID(),
                    area: "Personal",
                  })),
                ]);
                setLibraryOpen(true);
              }
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
      {graph.nodes.length > 0 && (
        <ValidationPanel
          issues={issues}
          disabled={running}
          changes={repairChanges}
          canUndo={undo.current.length > 0}
          onSelect={setSelected}
          onFix={fixIssue}
          onRepairAll={repairAll}
          onUndo={undoRepair}
        />
      )}
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
          <IssueContext.Provider value={nodeIssues}>
            <BlueprintCanvas
              graph={graph}
              setGraph={updateGraph}
              onSelect={setSelected}
              disabled={running}
            />
          </IssueContext.Provider>
        </ReactFlowProvider>
        <DetailsInspector
          networkConnected={graph.edges.some(
            (e) => e.target === selected && e.targetHandle === "network",
          )}
          node={graph.nodes.find((n) => n.id === selected)}
          frameConnected={graph.edges.some(
            (edge) => edge.target === selected && edge.targetHandle === "frame",
          )}
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
            if (node.data.kind === "plot") {
              const palette = Object.keys(plotPalette);
              const current = palette.indexOf(
                String(original.data.params.color),
              );
              node.data.params = {
                ...node.data.params,
                title: `${String(original.data.params.title || "Custom plot")} (copy)`,
                color:
                  palette[
                    ((current >= 0
                      ? current
                      : Math.max(
                          (
                            result?.plots ??
                            graph.nodes.filter((n) => n.data.kind === "plot")
                          ).findIndex((n) => n.id === original.id),
                          0,
                        ) % palette.length) +
                      1) %
                      palette.length
                  ],
              };
            }
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
      {graph.nodes.some((n) => n.data.kind === "model") && (
        <PerformanceDock result={result} running={running} />
      )}
      <PortfolioDock
        portfolios={result?.portfolios ?? []}
        reports={result?.risk_reports ?? []}
      />
      <PlotDock
        plots={(result?.plots ?? []).map((plot, index) =>
          presentPlot(
            plot,
            graph.nodes.find((n) => n.id === plot.id),
            index,
          ),
        )}
        onInspect={(id) => {
          setSelected(id);
          document
            .querySelector(".workspace")
            ?.scrollIntoView({ behavior: "smooth" });
        }}
      />
      <ScalingDock
        audits={result?.scaling_audits ?? []}
        onInspect={(id) => {
          setSelected(id);
          document
            .querySelector(".workspace")
            ?.scrollIntoView({ behavior: "smooth" });
        }}
      />
      <StudyDock studies={result?.studies ?? []} onInspect={setSelected} />
      {brainOpen && (
        <BrainPanel
          onClose={() => setBrainOpen(false)}
          onOpen={(brainGraph, brainTitle, run) => {
            setBrainOpen(false);
            setRepairChanges([]);
            loadEntry(
              {
                id: crypto.randomUUID(),
                title: brainTitle,
                area: "Personal",
                description: "Blueprint construido por el Cerebro IA",
                instructions: ["Revisa cada nodo y ejecuta."],
                graph: brainGraph,
              },
              run,
            );
          }}
        />
      )}
      {libraryOpen && (
        <BlueprintLibrary
          graph={graph}
          title={title}
          onClose={() => setLibraryOpen(false)}
          onLoad={loadEntry}
        />
      )}
      <footer className={`status-bar ${error ? "error" : ""}`}>
        {error ? <CircleAlert size={13} /> : <CheckCircle2 size={13} />}
        <span>{message}</span>
        <small>
          LOCAL COMPUTE <span>•</span>{" "}
          {graph.nodes.some(
            (n) =>
              (n.data.kind === ("source" satisfies Kind) ||
                n.data.kind === "portfolio_data") &&
              n.data.params.source === "yfinance",
          )
            ? "YFINANCE DATA"
            : "SYNTHETIC DATA"}
        </small>
      </footer>
    </div>
  );
}
