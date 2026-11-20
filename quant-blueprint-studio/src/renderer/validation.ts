/**
 * Live blueprint validation and deterministic repair.
 *
 * Every rule returns an Issue with a plain-language message and, when the
 * correction is unambiguous, a Fix. Repair applies only those fixes, so every
 * change is reproducible and can be undone. Nothing here executes the graph.
 */
import contracts from "./data/node-contracts.json";
import { acceptsPayload } from "./connections";
import {
  catalog,
  featureOptions,
  inputHandles,
  networkBindings,
  type Graph,
  type Kind,
  type Param,
} from "./types/blueprint";

export type Severity = "error" | "warning" | "info";
export type Fix =
  | { type: "set_param"; node: string; key: string; value: Param }
  | { type: "remove_edge"; edge: string }
  | { type: "connect"; source: string; target: string; pin: string }
  | { type: "reconnect"; source: string; target: string; pin: string };
export interface Issue {
  id: string;
  severity: Severity;
  node?: string;
  message: string;
  fix?: Fix;
  fixLabel?: string;
}

const TERMINAL: Kind[] = [
  "plot",
  "risk",
  "backtest",
  "portfolio",
  "study",
  "rl_agent",
  "curve_fit",
  "ml_model",
  "model",
  "laya_feedback",
];
const COLUMN_PARAMS = ["plot_columns", "feature_columns", "columns"];

/** Parameter as the engine sees it: explicit value, else the catalog default. */
function paramValue(node: Graph["nodes"][number], key: string): Param | undefined {
  return node.data.params[key] ?? catalog[node.data.kind]?.fields.find((f) => f.key === key)?.value;
}

function label(graph: Graph, id: string): string {
  const node = graph.nodes.find((n) => n.id === id);
  return node ? node.data.label || catalog[node.data.kind].label : id;
}

function ancestors(graph: Graph, id: string, seen = new Set<string>()): Set<string> {
  for (const edge of graph.edges)
    if (edge.target === id && !seen.has(edge.source)) {
      seen.add(edge.source);
      ancestors(graph, edge.source, seen);
    }
  return seen;
}

/** Columns arriving at a node's frame inputs (featureOptions gives a node's own output). */
export function incomingColumns(graph: Graph, id: string): string[] {
  return [
    ...new Set(
      graph.edges
        .filter((e) => e.target === id && e.sourceHandle === "frame")
        .flatMap((e) => featureOptions(graph, e.source)),
    ),
  ];
}

/** Nearest node (by canvas distance, upstream-left first) whose output fits the pin, without creating a cycle. */
export function nearestProducer(graph: Graph, targetId: string, pin: string): string | undefined {
  const target = graph.nodes.find((n) => n.id === targetId);
  if (!target) return undefined;
  const downstream = new Set<string>([targetId]);
  let grew = true;
  while (grew) {
    grew = false;
    for (const e of graph.edges)
      if (downstream.has(e.source) && !downstream.has(e.target)) {
        downstream.add(e.target);
        grew = true;
      }
  }
  const pinType = inputHandles(target.data.kind)[pin];
  const wanted = COLUMN_PARAMS.flatMap((key) =>
    String(paramValue(target, key) ?? "")
      .split(",")
      .map((c) => c.trim())
      .filter(Boolean),
  );
  // How many requested columns would arrive if this producer were connected.
  const coverage = (sourceId: string): number => {
    if (!wanted.length || pinType !== "frame") return 0;
    const trial: Graph = {
      ...graph,
      edges: [
        ...graph.edges.filter((e) => !(e.target === targetId && e.targetHandle === pin)),
        { id: "trial", source: sourceId, target: targetId, sourceHandle: "frame", targetHandle: pin },
      ],
    };
    const available = incomingColumns(trial, targetId);
    return wanted.filter((c) => available.includes(c)).length;
  };
  return graph.nodes
    .filter(
      (n) =>
        !downstream.has(n.id) &&
        catalog[n.data.kind].output === pinType &&
        acceptsPayload(n.data.kind, target.data.kind, pin),
    )
    .map((n) => ({
      id: n.id,
      covered: coverage(n.id),
      distance:
        Math.hypot(n.position.x - target.position.x, n.position.y - target.position.y) +
        (n.position.x > target.position.x ? 5000 : 0),
    }))
    .sort((a, b) => b.covered - a.covered || a.distance - b.distance)[0]?.id;
}

function hasCycle(graph: Graph): boolean {
  const indegree = new Map(graph.nodes.map((n) => [n.id, 0]));
  for (const e of graph.edges) indegree.set(e.target, (indegree.get(e.target) ?? 0) + 1);
  const queue = [...indegree].filter(([, d]) => d === 0).map(([id]) => id);
  let visited = 0;
  while (queue.length) {
    const id = queue.shift() as string;
    visited++;
    for (const e of graph.edges)
      if (e.source === id) {
        const d = (indegree.get(e.target) ?? 1) - 1;
        indegree.set(e.target, d);
        if (d === 0) queue.push(e.target);
      }
  }
  return visited !== graph.nodes.length;
}

export function validateGraph(graph: Graph): Issue[] {
  const issues: Issue[] = [];
  const add = (issue: Omit<Issue, "id">) =>
    issues.push({ ...issue, id: `${issues.length}:${issue.node ?? ""}:${issue.message}` });
  if (!graph.nodes.length) return issues;
  if (graph.nodes.length > 50)
    add({ severity: "error", message: "El motor admite como máximo 50 nodos por blueprint." });
  const ids = new Set(graph.nodes.map((n) => n.id));
  const nodes = new Map(graph.nodes.map((n) => [n.id, n]));
  if (!graph.nodes.some((n) => ["source", "portfolio_data"].includes(n.data.kind)))
    add({
      severity: "error",
      message: "No hay fuente de datos. Añade Market Data o Portfolio Data al inicio del flujo.",
    });

  // Connections
  const usedPins = new Set<string>();
  for (const edge of graph.edges) {
    const src = nodes.get(edge.source);
    const dst = nodes.get(edge.target);
    if (!src || !dst || !ids.has(edge.target)) {
      add({ severity: "error", message: "Conexión a un nodo inexistente.", fix: { type: "remove_edge", edge: edge.id }, fixLabel: "Quitar conexión" });
      continue;
    }
    const pin = edge.targetHandle ?? "";
    const pinType = inputHandles(dst.data.kind)[pin];
    const reason =
      catalog[src.data.kind].output !== edge.sourceHandle || pinType !== edge.sourceHandle
        ? `el pin ${edge.sourceHandle ?? "?"} de ${label(graph, src.id)} no corresponde a la entrada ${pin || "?"}`
        : !acceptsPayload(src.data.kind, dst.data.kind, pin)
          ? `${catalog[dst.data.kind].label} no acepta el tensor de ${catalog[src.data.kind].label}`
          : usedPins.has(`${dst.id}:${pin}`)
            ? `la entrada ${pin} ya tiene otra conexión`
            : "";
    usedPins.add(`${dst.id}:${pin}`);
    if (reason)
      add({
        severity: "error",
        node: dst.id,
        message: `Conexión incompatible: ${reason}.`,
        fix: { type: "remove_edge", edge: edge.id },
        fixLabel: "Quitar conexión",
      });
  }
  if (hasCycle(graph))
    add({ severity: "error", message: "Hay un ciclo. Los blueprints deben ser grafos dirigidos acíclicos." });

  for (const node of graph.nodes) {
    const def = catalog[node.data.kind];
    if (!def) {
      add({ severity: "error", node: node.id, message: `Tipo de nodo desconocido: ${String(node.data.kind)}.` });
      continue;
    }
    const name = label(graph, node.id);
    // Parameters: types, ranges and options
    for (const field of def.fields) {
      const value = node.data.params[field.key] ?? field.value;
      if (field.options) {
        if (!field.options.includes(String(value)))
          add({
            severity: "error",
            node: node.id,
            message: `${name}: «${field.label}» = ${String(value)} no es una opción válida (${field.options.join(", ")}).`,
            fix: { type: "set_param", node: node.id, key: field.key, value: field.value },
            fixLabel: `Usar ${String(field.value)}`,
          });
      } else if (typeof field.value === "number") {
        const n = Number(value);
        if (value === "" || !Number.isFinite(n))
          add({
            severity: "error",
            node: node.id,
            message: `${name}: «${field.label}» debe ser un número.`,
            fix: { type: "set_param", node: node.id, key: field.key, value: field.value },
            fixLabel: `Usar ${field.value}`,
          });
        else if ((field.min !== undefined && n < field.min) || (field.max !== undefined && n > field.max)) {
          const clamped = Math.min(field.max ?? n, Math.max(field.min ?? n, n));
          add({
            severity: "error",
            node: node.id,
            message: `${name}: «${field.label}» = ${n} fuera del rango ${field.min ?? "−∞"}–${field.max ?? "∞"}.`,
            fix: { type: "set_param", node: node.id, key: field.key, value: clamped },
            fixLabel: `Ajustar a ${clamped}`,
          });
        }
      }
    }
    // Required primary input
    const primary = (contracts.nodes as Record<string, { input: string | null }>)[node.data.kind]?.input;
    if (primary && !graph.edges.some((e) => e.target === node.id && e.targetHandle === primary)) {
      const producer = nearestProducer(graph, node.id, primary);
      add({
        severity: "error",
        node: node.id,
        message: `${name}: falta la entrada ${primary}.${producer ? "" : " No hay ningún nodo compatible en el lienzo."}`,
        fix: producer ? { type: "connect", source: producer, target: node.id, pin: primary } : undefined,
        fixLabel: producer ? `Conectar desde ${label(graph, producer)}` : undefined,
      });
    }
    // Unused outputs
    if (!TERMINAL.includes(node.data.kind) && def.output && !graph.edges.some((e) => e.source === node.id))
      add({ severity: "info", node: node.id, message: `${name}: su salida no se usa en ningún nodo posterior.` });
    // Columns that upstream nodes do not produce
    const available = incomingColumns(graph, node.id);
    if (available.length)
      for (const key of COLUMN_PARAMS) {
        const raw = paramValue(node, key);
        if (typeof raw !== "string" || !raw.trim()) continue;
        const missing = raw
          .split(",")
          .map((c) => c.trim())
          .filter((c) => c && !available.includes(c));
        if (missing.length) {
          const current = graph.edges.find((e) => e.target === node.id && e.targetHandle === "frame")?.source;
          const better = nearestProducer(graph, node.id, "frame");
          const helps = better && better !== current;
          add({
            severity: "error",
            node: node.id,
            message: `${name}: columnas no producidas aguas arriba: ${missing.join(", ")}. Disponibles: ${available.slice(0, 12).join(", ")}${available.length > 12 ? "…" : ""}.`,
            fix: helps ? { type: "reconnect", source: better, target: node.id, pin: "frame" } : undefined,
            fixLabel: helps ? `Conectar a ${label(graph, better)}` : undefined,
          });
        }
      }
  }

  // Semantic rules that otherwise fail only at execution time
  for (const node of graph.nodes) {
    const name = label(graph, node.id);
    const up = [...ancestors(graph, node.id)].map((id) => nodes.get(id)?.data.kind);
    if (["laya_calibrate", "laya_feedback"].includes(node.data.kind) && graph.edges.some((e) => e.target === node.id) && !up.includes("laya_signal"))
      add({ severity: "error", node: node.id, message: `${name}: necesita Laya Signal aguas arriba (columnas Laya_P_*).` });
    if (node.data.kind === "laya_feedback" && !up.includes("laya_calibrate"))
      add({ severity: "info", node: node.id, message: `${name}: sin Laya Calibrate empieza con los pesos crudos de Laya y en la fracción «Inicio si no hay Calibrate».` });
    if (node.data.kind === "features" && node.data.params.use_vwap !== false) {
      const fx = [...ancestors(graph, node.id)].some((id) => {
        const n = nodes.get(id);
        return n?.data.kind === "source" && n.data.params.source === "yfinance" && /=X$/i.test(String(n.data.params.ticker));
      });
      if (fx)
        add({
          severity: "error",
          node: node.id,
          message: `${name}: los pares de divisas de Yahoo (=X) no reportan volumen; VWAP fallaría.`,
          fix: { type: "set_param", node: node.id, key: "use_vwap", value: false },
          fixLabel: "Desactivar VWAP",
        });
    }
    if (node.data.kind === "laya_signal") {
      const source = [...ancestors(graph, node.id)].map((id) => nodes.get(id)).find((n) => n?.data.kind === "source");
      const years = Number(source?.data.params.years ?? 3);
      if (Number(node.data.params.max_rows ?? 500) > years * 252)
        add({ severity: "warning", node: node.id, message: `${name}: pide más días (${String(node.data.params.max_rows)}) de los que da la fuente (~${years * 252}); se usarán los disponibles.` });
    }
  }
  for (const binding of networkBindings(graph))
    if (binding.issue)
      add({ severity: "error", node: binding.trainerId, message: binding.issue });
  const order: Record<Severity, number> = { error: 0, warning: 1, info: 2 };
  return issues.sort((a, b) => order[a.severity] - order[b.severity]);
}

export function applyFix(graph: Graph, fix: Fix): Graph {
  if (fix.type === "remove_edge") return { ...graph, edges: graph.edges.filter((e) => e.id !== fix.edge) };
  if (fix.type === "set_param")
    return {
      ...graph,
      nodes: graph.nodes.map((n) =>
        n.id === fix.node ? { ...n, data: { ...n.data, params: { ...n.data.params, [fix.key]: fix.value } } } : n,
      ),
    };
  const source = graph.nodes.find((n) => n.id === fix.source);
  const edges =
    fix.type === "reconnect"
      ? graph.edges.filter((e) => !(e.target === fix.target && e.targetHandle === fix.pin))
      : graph.edges;
  if (!source || edges.some((e) => e.target === fix.target && e.targetHandle === fix.pin)) return graph;
  return {
    ...graph,
    edges: [
      ...edges,
      {
        id: `${fix.source}-${fix.target}-${fix.pin}`,
        source: fix.source,
        target: fix.target,
        sourceHandle: catalog[source.data.kind].output,
        targetHandle: fix.pin,
        animated: true,
      },
    ],
  };
}

/** Apply unambiguous fixes one at a time, re-validating after each (bounded to 40). */
export function repairGraph(graph: Graph): { graph: Graph; changes: string[]; remaining: Issue[] } {
  let current = graph;
  const changes: string[] = [];
  const tried = new Set<string>();
  while (changes.length < 40) {
    const issue = validateGraph(current).find(
      (i) => i.fix && i.severity !== "info" && !tried.has(JSON.stringify(i.fix)),
    );
    if (!issue?.fix) break;
    tried.add(JSON.stringify(issue.fix));
    current = applyFix(current, issue.fix);
    changes.push(`${issue.fixLabel ?? "Corregido"} — ${issue.message}`);
  }
  return { graph: current, changes, remaining: validateGraph(current).filter((i) => i.severity !== "info") };
}
