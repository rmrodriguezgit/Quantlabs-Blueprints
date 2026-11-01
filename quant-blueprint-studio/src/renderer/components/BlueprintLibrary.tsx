import { useEffect, useRef, useState } from "react";
import builtinLibrary from "../data/library.json";
import {
  X,
  BookOpen,
  Download,
  Upload,
  Save,
  Play,
  Code2,
  Pencil,
  Trash2,
  Undo2,
} from "lucide-react";
import {
  downloadLibrary,
  parseLibrary,
  readSaved,
  type LibraryEntry,
} from "../library";
import {
  deleteExample,
  emptyCatalog,
  entryJSON,
  parseEditedEntry,
  persistWorkspace,
  readCatalog,
  resolveExamples,
  restoreExample,
  saveExample,
  type LibraryWorkspace,
} from "../catalog";
import type { Graph } from "../types/blueprint";
export function BlueprintLibrary({
  graph,
  title,
  onClose,
  onLoad,
}: {
  graph: Graph;
  title: string;
  onClose: () => void;
  onLoad: (entry: LibraryEntry, run: boolean) => void;
}) {
  const [builtins] = useState<LibraryEntry[]>(() =>
    parseLibrary(builtinLibrary),
  );
  const [initial] = useState(() => {
    try {
      return {
        workspace: { saved: readSaved(), catalog: readCatalog() },
        error: "",
      };
    } catch (error) {
      return {
        workspace: { saved: readSaved(), catalog: emptyCatalog() },
        error: `No se pudo leer el catálogo: ${String(error)}. No se guardarán cambios sobre este estado.`,
      };
    }
  });
  const [workspace, setWorkspace] = useState<LibraryWorkspace>(
    initial.workspace,
  );
  const [imageExpanded, setImageExpanded] = useState(false);
  const [selected, setSelected] = useState("unit-2");
  const [area, setArea] = useState("All");
  const [name, setName] = useState(title);
  const [message, setMessage] = useState(initial.error);
  const [error, setError] = useState(Boolean(initial.error));
  const [editor, setEditor] = useState<"view" | "edit" | null>(null);
  const [draft, setDraft] = useState("");
  const [showTrash, setShowTrash] = useState(false);
  const file = useRef<HTMLInputElement>(null);
  const builtinIds = new Set(builtins.map((e) => e.id));
  const all = resolveExamples(builtins, workspace);
  const entry = all.find((e) => e.id === selected);
  const dirty = editor === "edit" && draft !== (entry ? entryJSON(entry) : "");
  function fail(err: unknown) {
    setError(true);
    setMessage(err instanceof Error ? err.message : String(err));
  }
  function canLeave(): boolean {
    if (!dirty) return true;
    fail(
      "Guarda los cambios o pulsa Cancelar antes de cambiar de ficha o cerrar.",
    );
    return false;
  }
  function close() {
    if (canLeave()) onClose();
  }
  useEffect(() => {
    const key = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        if (imageExpanded) setImageExpanded(false);
        else close();
      }
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [onClose, dirty, imageExpanded]);
  useEffect(() => {
    const guard = (event: Event) => {
      if (dirty) {
        event.preventDefault();
        fail("Guarda o cancela la edición antes de crear un blueprint nuevo.");
      }
    };
    window.addEventListener("quantlabs:before-new", guard);
    return () => window.removeEventListener("quantlabs:before-new", guard);
  }, [dirty]);
  function commit(next: LibraryWorkspace) {
    if (initial.error) throw new Error(initial.error);
    persistWorkspace(next);
    setWorkspace(next);
  }
  function choose(id: string) {
    if (!canLeave()) return;
    setSelected(id);
    setImageExpanded(false);
    setEditor(null);
    setDraft("");
    setShowTrash(false);
    setError(false);
    setMessage("");
  }
  function openEditor(mode: "view" | "edit") {
    if (!entry || !canLeave()) return;
    setDraft(entryJSON(entry));
    setEditor(mode);
    setError(false);
    setMessage(
      mode === "edit"
        ? "Edita nombre, área, parámetros, nodos o conexiones y pulsa Guardar. El id se conserva."
        : "JSON del ejemplo guardado. Pulsa Editar para modificarlo.",
    );
  }
  function save() {
    if (!entry || editor !== "edit") return;
    try {
      const edited = parseEditedEntry(draft, entry);
      commit(
        saveExample(
          workspace,
          edited,
          builtinIds,
          builtins.find((e) => e.id === edited.id),
        ),
      );
      setDraft(entryJSON(edited));
      setEditor("view");
      setArea("All");
      setError(false);
      setMessage(
        "Cambios guardados en este equipo. Exporta JSON para transferirlos a otro equipo.",
      );
    } catch (err) {
      fail(err);
    }
  }
  function remove() {
    if (!entry || !canLeave()) return;
    try {
      const next = deleteExample(workspace, entry, builtinIds);
      commit(next);
      setSelected(resolveExamples(builtins, next)[0]?.id ?? "");
      setArea("All");
      setEditor(null);
      setImageExpanded(false);
      setError(false);
      setMessage(
        `«${entry.title}» se movió a la Papelera. Puedes restaurarlo.`,
      );
    } catch (err) {
      fail(err);
    }
  }
  return (
    <div className="library-backdrop">
      <section
        className="blueprint-library"
        role="dialog"
        aria-modal="true"
        aria-label="Blueprint library"
      >
        <header>
          <div>
            <small>QUANTLABS / LEARNING LAB</small>
            <h2>
              <BookOpen size={22} /> Biblioteca de Blueprints
            </h2>
            <p>
              {all.length} diseños disponibles · ejemplos y copias personales
            </p>
          </div>
          <button onClick={close} aria-label="Close library">
            <X size={20} />
          </button>
        </header>
        <div className="library-tools">
          <label>
            Área{" "}
            <select
              disabled={showTrash}
              value={area}
              onChange={(e) => {
                if (canLeave()) setArea(e.target.value);
              }}
            >
              {["All", ...new Set(all.map((e) => e.area))].map((a) => (
                <option key={a}>{a}</option>
              ))}
            </select>
          </label>
          <button
            onClick={() => {
              if (canLeave()) file.current?.click();
            }}
          >
            <Upload size={14} />
            Importar JSON
          </button>
          <button
            disabled={!all.length}
            onClick={() => {
              if (canLeave()) downloadLibrary(all);
            }}
          >
            <Download size={14} />
            Exportar biblioteca
          </button>
          <button
            aria-pressed={showTrash}
            onClick={() => {
              if (canLeave()) {
                setShowTrash(!showTrash);
                setEditor(null);
              }
            }}
          >
            <Trash2 size={14} />
            Papelera ({workspace.catalog.trash.length})
          </button>
        </div>
        {showTrash ? (
          <div className="library-trash">
            <h3>Papelera</h3>
            <p>
              Los diseños borrados se conservan aquí hasta que los restaures.
            </p>
            {!workspace.catalog.trash.length ? (
              <p>No hay ejemplos borrados.</p>
            ) : (
              workspace.catalog.trash.map((item) => (
                <article key={item.token}>
                  <div>
                    <strong>{item.entry.title}</strong>
                    <small>
                      {item.entry.area} · {item.entry.graph.nodes.length} nodos
                      · {new Date(item.deletedAt).toLocaleString()}
                    </small>
                  </div>
                  <button
                    onClick={() => {
                      try {
                        const restored = restoreExample(workspace, item.token);
                        commit(restored.workspace);
                        setSelected(restored.id);
                        setArea("All");
                        setShowTrash(false);
                        setEditor(null);
                        setError(false);
                        setMessage(
                          "Ejemplo restaurado con sus nodos, parámetros y metadatos.",
                        );
                      } catch (err) {
                        fail(err);
                      }
                    }}
                  >
                    <Undo2 size={14} />
                    Restaurar
                  </button>
                </article>
              ))
            )}
          </div>
        ) : (
          <div className="library-body">
            <nav aria-label="Library templates">
              {all
                .filter((e) => area === "All" || e.area === area)
                .map((e) => (
                  <button
                    key={e.id}
                    className={selected === e.id ? "active" : ""}
                    onClick={() => choose(e.id)}
                  >
                    <small>{e.area}</small>
                    <strong>{e.title}</strong>
                    <span>
                      {e.graph.nodes.length} nodos ·{" "}
                      {e.infographic ? "Infografía" : "Ejemplo"}
                      {workspace.catalog.overrides.some((o) => o.id === e.id)
                        ? " · Modificado"
                        : ""}
                    </span>
                  </button>
                ))}
            </nav>
            {entry ? (
              <article
                className={
                  editor ? "library-detail json-open" : "library-detail"
                }
              >
                <div className="library-detail-copy">
                  <small>{entry.area}</small>
                  <h3>{entry.title}</h3>
                  <p>{entry.description}</p>
                  <div
                    className="library-actions library-crud"
                    aria-label="Example actions"
                  >
                    <button onClick={() => openEditor("view")}>
                      <Code2 size={14} />
                      Ver JSON
                    </button>
                    <button onClick={() => openEditor("edit")}>
                      <Pencil size={14} />
                      Editar
                    </button>
                    <button disabled={editor !== "edit"} onClick={save}>
                      <Save size={14} />
                      Guardar
                    </button>
                    <button onClick={remove}>
                      <Trash2 size={14} />
                      Borrar
                    </button>
                  </div>
                  {editor ? (
                    <section
                      className="library-json-editor"
                      aria-label="Example JSON editor"
                    >
                      <div>
                        <strong>
                          {editor === "edit" ? "Editar JSON" : "Ver JSON"}
                        </strong>
                        <span>
                          {dirty
                            ? "Cambios sin guardar"
                            : editor === "edit"
                              ? "Edición habilitada"
                              : "Sólo lectura"}
                        </span>
                      </div>
                      <label htmlFor="example-json">JSON del ejemplo</label>
                      <textarea
                        id="example-json"
                        aria-label="JSON del ejemplo"
                        spellCheck={false}
                        readOnly={editor === "view"}
                        maxLength={10_000_000}
                        value={draft}
                        onChange={(e) => setDraft(e.target.value)}
                      />
                      <p>
                        Formato de biblioteca con una ficha, o un grafo
                        nodes/edges. Guardar valida estructura, pines y ciclos;
                        las configuraciones del modelo se revisan al ejecutar.
                      </p>
                      {editor === "edit" && (
                        <div className="library-actions">
                          <button onClick={save}>Guardar cambios</button>
                          <button
                            onClick={() => {
                              setDraft(entryJSON(entry));
                              setEditor("view");
                              setError(false);
                              setMessage(
                                "Edición cancelada. Se conserva la versión guardada.",
                              );
                            }}
                          >
                            Cancelar
                          </button>
                          <button
                            onClick={() => {
                              try {
                                const edited = parseEditedEntry(draft, entry);
                                setDraft(
                                  entryJSON({
                                    ...edited,
                                    graph: structuredClone(graph),
                                  }),
                                );
                                setError(false);
                                setMessage(
                                  "JSON actualizado con el lienzo actual. Pulsa Guardar para confirmar los cambios.",
                                );
                              } catch (err) {
                                fail(err);
                              }
                            }}
                          >
                            Usar lienzo actual
                          </button>
                        </div>
                      )}
                    </section>
                  ) : (
                    <ol>
                      {entry.instructions.map((s, i) => (
                        <li key={i}>{s}</li>
                      ))}
                    </ol>
                  )}
                  <div className="library-actions">
                    <button
                      onClick={() => {
                        if (canLeave()) onLoad(entry, false);
                      }}
                    >
                      Cargar y editar
                    </button>
                    <button
                      className="execute-button"
                      disabled={!entry.graph.nodes.length}
                      onClick={() => {
                        if (canLeave()) onLoad(entry, true);
                      }}
                    >
                      <Play size={14} />
                      Cargar y ejecutar
                    </button>
                    <button
                      onClick={() => {
                        if (canLeave())
                          downloadLibrary([entry], `${entry.id}.json`);
                      }}
                    >
                      Exportar
                    </button>
                  </div>
                </div>
                {!editor && entry.infographic && (
                  <button
                    className={
                      imageExpanded
                        ? "infographic-preview expanded"
                        : "infographic-preview"
                    }
                    onClick={() => setImageExpanded(!imageExpanded)}
                    title={
                      imageExpanded ? "Cerrar ampliación" : "Ampliar infografía"
                    }
                    aria-label={
                      imageExpanded
                        ? "Cerrar infografía ampliada"
                        : "Ampliar infografía"
                    }
                  >
                    <img
                      src={`./infographics/${entry.infographic}`}
                      alt={`Infografía: ${entry.title}`}
                    />
                  </button>
                )}
              </article>
            ) : (
              <p className="muted">
                Selecciona una plantilla o restaura un ejemplo de la Papelera.
              </p>
            )}
          </div>
        )}
        <footer>
          <div>
            <label>
              Guardar el diseño actual{" "}
              <input
                value={name}
                maxLength={200}
                onChange={(e) => setName(e.target.value)}
                placeholder="Nombre del blueprint"
              />
            </label>
            <button
              disabled={!name.trim()}
              onClick={() => {
                if (!canLeave()) return;
                try {
                  const item: LibraryEntry = {
                    id: crypto.randomUUID(),
                    title: name.trim(),
                    area: "Personal",
                    description: "Copia editable del diseño actual.",
                    instructions: [
                      "Ajusta los parámetros y pulsa Execute Blueprint.",
                    ],
                    graph: structuredClone(graph),
                  };
                  commit({ ...workspace, saved: [...workspace.saved, item] });
                  setSelected(item.id);
                  setArea("All");
                  setShowTrash(false);
                  setEditor(null);
                  setError(false);
                  setMessage(
                    "Copia guardada en Personal. Exporta JSON para llevarla a otro equipo.",
                  );
                } catch (err) {
                  fail(err);
                }
              }}
            >
              <Save size={14} />
              Guardar copia
            </button>
          </div>
          <p role="status" className={error ? "library-error" : ""}>
            {message ||
              "Ver JSON, Editar y Guardar modifican la ficha seleccionada. Borrar la mueve a Papelera. Los cambios se conservan en este equipo."}
          </p>
        </footer>
        <input
          ref={file}
          type="file"
          accept="application/json,.json"
          hidden
          onChange={async (e) => {
            const f = e.target.files?.[0];
            e.target.value = "";
            if (!f) return;
            try {
              if (f.size > 10_000_000)
                throw new Error("Library exceeds 10 MB.");
              const imported = parseLibrary(
                JSON.parse(await f.text()) as unknown,
              ).map((v) => ({
                ...v,
                id: crypto.randomUUID(),
                area: "Personal",
              }));
              commit({
                ...workspace,
                saved: [...workspace.saved, ...imported],
              });
              setSelected(imported[0].id);
              setArea("All");
              setShowTrash(false);
              setEditor(null);
              setError(false);
              setMessage(`${imported.length} blueprints importados.`);
            } catch (err) {
              fail(err);
            }
          }}
        />
      </section>
    </div>
  );
}
