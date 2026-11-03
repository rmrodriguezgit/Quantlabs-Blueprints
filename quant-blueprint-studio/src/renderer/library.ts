import { acceptsPayload } from "./connections";
import { catalog, inputHandles, type Graph } from "./types/blueprint";
export interface LibraryEntry {
  id: string;
  title: string;
  area: string;
  description: string;
  instructions: string[];
  infographic?: string;
  graph: Graph;
}
export interface LibraryBundle {
  format: "quantlabs-blueprint-library";
  schemaVersion: 1;
  entries: LibraryEntry[];
}
export const LIBRARY_KEY = "quantlabs-library-v1";
export function isGraph(value: unknown): value is Graph {
  if (typeof value !== "object" || !value) return false;
  const g = value as Partial<Graph>;
  if (
    !Array.isArray(g.nodes) ||
    g.nodes.length > 50 ||
    !Array.isArray(g.edges) ||
    g.edges.length > 150
  )
    return false;
  if (
    !g.nodes.every(
      (n) =>
        n &&
        typeof n.id === "string" &&
        n.id.length > 0 &&
        n.id.length <= 100 &&
        n.type === "blueprint" &&
        n.data &&
        typeof n.data.label === "string" &&
        Object.hasOwn(catalog, n.data.kind) &&
        n.data.params &&
        typeof n.data.params === "object" &&
        !Array.isArray(n.data.params) &&
        Object.values(n.data.params).every(
          (v) =>
            typeof v === "string" ||
            typeof v === "boolean" ||
            (typeof v === "number" && Number.isFinite(v)),
        ) &&
        n.position &&
        Number.isFinite(n.position.x) &&
        Number.isFinite(n.position.y),
    )
  )
    return false;
  if (new Set(g.nodes.map((n) => n.id)).size !== g.nodes.length) return false;
  const pins = new Set<string>();
  for (const e of g.edges) {
    if (!e || typeof e.id !== "string") return false;
    const src = g.nodes.find((n) => n.id === e.source),
      dst = g.nodes.find((n) => n.id === e.target);
    const pin = `${e.target}:${e.targetHandle}`;
    if (
      !src ||
      !dst ||
      src.id === dst.id ||
      !acceptsPayload(src.data.kind, dst.data.kind, e.targetHandle ?? "") ||
      !e.sourceHandle ||
      !e.targetHandle ||
      e.sourceHandle !== catalog[src.data.kind].output ||
      inputHandles(dst.data.kind)[e.targetHandle] !== e.sourceHandle ||
      pins.has(pin)
    )
      return false;
    pins.add(pin);
  }
  const indegree = new Map(g.nodes.map((n) => [n.id, 0]));
  for (const e of g.edges)
    indegree.set(e.target, (indegree.get(e.target) ?? 0) + 1);
  const queue = [...indegree].filter(([, n]) => n === 0).map(([id]) => id);
  let count = 0;
  while (queue.length) {
    const id = queue.shift();
    count++;
    for (const e of g.edges.filter((e) => e.source === id)) {
      const d = (indegree.get(e.target) ?? 0) - 1;
      indegree.set(e.target, d);
      if (d === 0) queue.push(e.target);
    }
  }
  return count === g.nodes.length;
}
export function parseLibrary(value: unknown): LibraryEntry[] {
  if (isGraph(value))
    return [
      {
        id: crypto.randomUUID(),
        title: "Imported blueprint",
        area: "Personal",
        description: "Legacy JSON graph",
        instructions: ["Load, edit node parameters and execute."],
        graph: value,
      },
    ];
  if (!value || typeof value !== "object")
    throw new Error("Invalid blueprint library.");
  const bundle = value as Partial<LibraryBundle>;
  if (
    bundle.format !== "quantlabs-blueprint-library" ||
    bundle.schemaVersion !== 1 ||
    !Array.isArray(bundle.entries) ||
    !bundle.entries.length ||
    bundle.entries.length > 100
  )
    throw new Error(
      "Unsupported library format/version; expected schemaVersion 1.",
    );
  if (
    !bundle.entries.every(
      (e) =>
        e &&
        typeof e.id === "string" &&
        typeof e.title === "string" &&
        e.title.length > 0 &&
        e.title.length <= 200 &&
        typeof e.area === "string" &&
        typeof e.description === "string" &&
        Array.isArray(e.instructions) &&
        e.instructions.every((s) => typeof s === "string") &&
        (e.infographic === undefined ||
          /^unidad-(?:[2-9]|10)(?:-cdcgan)?\.png$/.test(e.infographic)) &&
        isGraph(e.graph),
    )
  )
    throw new Error("Library contains invalid metadata, pins or graphs.");
  if (new Set(bundle.entries.map((e) => e.id)).size !== bundle.entries.length)
    throw new Error("Duplicate blueprint identifiers.");
  return bundle.entries;
}
export function downloadLibrary(
  entries: LibraryEntry[],
  filename = "quantlabs-blueprints.json",
): void {
  const bundle: LibraryBundle = {
    format: "quantlabs-blueprint-library",
    schemaVersion: 1,
    entries,
  };
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(bundle, null, 2)], { type: "application/json" }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
export function readSaved(): LibraryEntry[] {
  try {
    const value: unknown = JSON.parse(
      localStorage.getItem(LIBRARY_KEY) ?? "null",
    );
    return value ? parseLibrary(value) : [];
  } catch {
    return [];
  }
}
export function storeSaved(entries: LibraryEntry[]): void {
  if (entries.length > 100)
    throw new Error(
      "Personal admite 100 diseños. Exporta y elimina copias antes de guardar más.",
    );
  if (entries.length)
    localStorage.setItem(
      LIBRARY_KEY,
      JSON.stringify({
        format: "quantlabs-blueprint-library",
        schemaVersion: 1,
        entries,
      }),
    );
  else localStorage.removeItem(LIBRARY_KEY);
}
