import {
  isGraph,
  parseLibrary,
  storeSaved,
  LIBRARY_KEY,
  type LibraryEntry,
  type LibraryBundle,
} from "./library";
export const CATALOG_KEY = "quantlabs-catalog-v1";
export interface DeletedExample {
  token: string;
  entry: LibraryEntry;
  source: "builtin" | "personal";
  deletedAt: string;
}
export interface CatalogState {
  schemaVersion: 1;
  overrides: LibraryEntry[];
  hidden: string[];
  trash: DeletedExample[];
}
export interface LibraryWorkspace {
  saved: LibraryEntry[];
  catalog: CatalogState;
}
export const emptyCatalog = (): CatalogState => ({
  schemaVersion: 1,
  overrides: [],
  hidden: [],
  trash: [],
});
export const entryJSON = (entry: LibraryEntry): string =>
  JSON.stringify(
    {
      format: "quantlabs-blueprint-library",
      schemaVersion: 1,
      entries: [entry],
    } satisfies LibraryBundle,
    null,
    2,
  );
function validatedEntries(entries: LibraryEntry[]): void {
  if (entries.length)
    parseLibrary({
      format: "quantlabs-blueprint-library",
      schemaVersion: 1,
      entries,
    });
}
export function validateCatalog(value: unknown): CatalogState {
  if (!value || typeof value !== "object")
    throw new Error("Estado de biblioteca inválido.");
  const state = value as Partial<CatalogState>;
  if (
    state.schemaVersion !== 1 ||
    !Array.isArray(state.overrides) ||
    !Array.isArray(state.hidden) ||
    !Array.isArray(state.trash) ||
    state.trash.length > 1000 ||
    !state.hidden.every((id) => typeof id === "string" && id.length > 0) ||
    new Set(state.hidden).size !== state.hidden.length
  )
    throw new Error("Estado de biblioteca inválido.");
  validatedEntries(state.overrides);
  const tokens = new Set<string>();
  for (const item of state.trash) {
    if (
      !item ||
      typeof item.token !== "string" ||
      tokens.has(item.token) ||
      !["builtin", "personal"].includes(item.source) ||
      typeof item.deletedAt !== "string" ||
      !Number.isFinite(Date.parse(item.deletedAt))
    )
      throw new Error("Papelera inválida.");
    validatedEntries([item.entry]);
    tokens.add(item.token);
  }
  return state as CatalogState;
}
export function readCatalog(): CatalogState {
  const raw = localStorage.getItem(CATALOG_KEY);
  return raw ? validateCatalog(JSON.parse(raw) as unknown) : emptyCatalog();
}
export function resolveExamples(
  builtins: LibraryEntry[],
  workspace: LibraryWorkspace,
): LibraryEntry[] {
  const overrides = new Map(workspace.catalog.overrides.map((e) => [e.id, e]));
  return [
    ...builtins
      .filter((e) => !workspace.catalog.hidden.includes(e.id))
      .map((e) => overrides.get(e.id) ?? e),
    ...workspace.saved,
  ];
}
export function parseEditedEntry(
  text: string,
  original: LibraryEntry,
): LibraryEntry {
  if (new TextEncoder().encode(text).length > 10_000_000)
    throw new Error("El JSON supera 10 MB.");
  let value: unknown;
  try {
    value = JSON.parse(text) as unknown;
  } catch (error) {
    throw new SyntaxError(
      `JSON inválido: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
  if (isGraph(value)) return { ...original, graph: value };
  const entries = parseLibrary(value);
  if (entries.length !== 1) throw new Error("Edita un solo ejemplo por vez.");
  if (entries[0].id !== original.id)
    throw new Error(
      "El id del ejemplo no se cambia. Para crear otro usa Guardar copia o Importar JSON.",
    );
  if (!entries[0].title.trim() || !entries[0].area.trim())
    throw new Error("Nombre y área no pueden estar vacíos.");
  return {
    ...entries[0],
    title: entries[0].title.trim(),
    area: entries[0].area.trim(),
  };
}
export function saveExample(
  workspace: LibraryWorkspace,
  entry: LibraryEntry,
  builtinIds: Set<string>,
  original?: LibraryEntry,
): LibraryWorkspace {
  validatedEntries([entry]);
  if (builtinIds.has(entry.id))
    return {
      ...workspace,
      catalog: {
        ...workspace.catalog,
        overrides: [
          ...workspace.catalog.overrides.filter((e) => e.id !== entry.id),
          ...(original &&
          original.id === entry.id &&
          entryJSON(original) === entryJSON(entry)
            ? []
            : [entry]),
        ],
      },
    };
  return {
    ...workspace,
    saved: [...workspace.saved.filter((e) => e.id !== entry.id), entry],
  };
}
export function deleteExample(
  workspace: LibraryWorkspace,
  entry: LibraryEntry,
  builtinIds: Set<string>,
): LibraryWorkspace {
  const source = builtinIds.has(entry.id) ? "builtin" : "personal";
  if (
    workspace.catalog.trash.some(
      (t) => t.entry.id === entry.id && t.source === source,
    )
  )
    throw new Error("Este ejemplo ya está en la papelera.");
  if (workspace.catalog.trash.length >= 1000)
    throw new Error(
      "La papelera admite 1000 elementos. Recupera elementos antes de borrar más.",
    );
  return {
    saved:
      source === "personal"
        ? workspace.saved.filter((e) => e.id !== entry.id)
        : workspace.saved,
    catalog: {
      ...workspace.catalog,
      hidden:
        source === "builtin"
          ? [...workspace.catalog.hidden, entry.id]
          : workspace.catalog.hidden,
      trash: [
        ...workspace.catalog.trash,
        {
          token: crypto.randomUUID(),
          source,
          entry: structuredClone(entry),
          deletedAt: new Date().toISOString(),
        },
      ],
    },
  };
}
export function restoreExample(
  workspace: LibraryWorkspace,
  token: string,
): { workspace: LibraryWorkspace; id: string } {
  const item = workspace.catalog.trash.find((t) => t.token === token);
  if (!item) throw new Error("El ejemplo ya no está en la papelera.");
  const catalog = {
    ...workspace.catalog,
    trash: workspace.catalog.trash.filter((t) => t.token !== token),
  };
  if (item.source === "builtin")
    return {
      id: item.entry.id,
      workspace: {
        ...workspace,
        catalog: {
          ...catalog,
          hidden: catalog.hidden.filter((id) => id !== item.entry.id),
          overrides: [
            ...catalog.overrides.filter((e) => e.id !== item.entry.id),
            structuredClone(item.entry),
          ],
        },
      },
    };
  const entry = structuredClone(item.entry);
  if (workspace.saved.some((e) => e.id === entry.id))
    entry.id = crypto.randomUUID();
  return {
    id: entry.id,
    workspace: { saved: [...workspace.saved, entry], catalog },
  };
}
export function persistWorkspace(workspace: LibraryWorkspace): void {
  if (workspace.saved.length > 100)
    throw new Error(
      "Personal admite 100 diseños. Exporta y mueve copias a Papelera antes de guardar más.",
    );
  validatedEntries(workspace.saved);
  validateCatalog(workspace.catalog);
  const previous = [LIBRARY_KEY, CATALOG_KEY].map(
    (key) => [key, localStorage.getItem(key)] as const,
  );
  try {
    storeSaved(workspace.saved);
    localStorage.setItem(CATALOG_KEY, JSON.stringify(workspace.catalog));
  } catch (error) {
    let restored = true;
    for (const [key, value] of previous) {
      try {
        if (value === null) localStorage.removeItem(key);
        else localStorage.setItem(key, value);
      } catch {
        restored = false;
      }
    }
    throw new Error(
      restored
        ? "No se pudo guardar en este equipo. Se conservó la biblioteca anterior. Exporta una copia y revisa el espacio disponible."
        : "No se pudo completar el guardado ni restaurar el almacenamiento. Conserva o exporta los diseños de la sesión antes de cerrar.",
      { cause: error },
    );
  }
}
