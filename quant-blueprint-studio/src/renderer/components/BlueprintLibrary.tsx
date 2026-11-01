import { useEffect, useRef, useState } from "react";
import builtinLibrary from "../data/library.json";
import { X, BookOpen, Download, Upload, Save, Play } from "lucide-react";
import {
  downloadLibrary,
  parseLibrary,
  readSaved,
  storeSaved,
  type LibraryEntry,
} from "../library";
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
  const [saved, setSaved] = useState(readSaved);
  const [imageExpanded, setImageExpanded] = useState(false);
  const [selected, setSelected] = useState("unit-2");
  const [area, setArea] = useState("All");
  const [name, setName] = useState(title);
  const [message, setMessage] = useState("");
  const [error, setError] = useState(false);
  const file = useRef<HTMLInputElement>(null);
  const all = [...builtins, ...saved];
  const entry = all.find((e) => e.id === selected);
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [onClose]);
  function persist(next: LibraryEntry[]) {
    storeSaved(next);
    setSaved(next);
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
              {builtins.length} blueprints · 10 infografías · notebook y
              ejemplos por área
            </p>
          </div>
          <button onClick={onClose} aria-label="Close library">
            <X size={20} />
          </button>
        </header>
        <div className="library-tools">
          <label>
            Área{" "}
            <select value={area} onChange={(e) => setArea(e.target.value)}>
              {["All", ...new Set(all.map((e) => e.area))].map((a) => (
                <option key={a}>{a}</option>
              ))}
            </select>
          </label>
          <button onClick={() => file.current?.click()}>
            <Upload size={14} />
            Importar JSON
          </button>
          <button disabled={!all.length} onClick={() => downloadLibrary(all)}>
            <Download size={14} />
            Exportar biblioteca
          </button>
        </div>
        <div className="library-body">
          <nav aria-label="Library templates">
            {all
              .filter((e) => area === "All" || e.area === area)
              .map((e) => (
                <button
                  key={e.id}
                  className={selected === e.id ? "active" : ""}
                  onClick={() => {
                    setSelected(e.id);
                    setImageExpanded(false);
                  }}
                >
                  <small>{e.area}</small>
                  <strong>{e.title}</strong>
                  <span>
                    {e.graph.nodes.length} nodos ·{" "}
                    {e.infographic ? "Infografía" : "Ejemplo"}
                  </span>
                </button>
              ))}
          </nav>
          {entry ? (
            <article className="library-detail">
              <div className="library-detail-copy">
                <small>{entry.area}</small>
                <h3>{entry.title}</h3>
                <p>{entry.description}</p>
                <ol>
                  {entry.instructions.map((s, i) => (
                    <li key={i}>{s}</li>
                  ))}
                </ol>
                <div className="library-actions">
                  <button onClick={() => onLoad(entry, false)}>
                    Cargar y editar
                  </button>
                  <button
                    className="execute-button"
                    onClick={() => onLoad(entry, true)}
                  >
                    <Play size={14} />
                    Cargar y ejecutar
                  </button>
                  <button
                    onClick={() => downloadLibrary([entry], `${entry.id}.json`)}
                  >
                    Exportar
                  </button>
                  {saved.some((e) => e.id === entry.id) && (
                    <button
                      onClick={() => {
                        try {
                          persist(saved.filter((e) => e.id !== entry.id));
                          setSelected("unit-2");
                        } catch (e) {
                          setMessage(String(e));
                          setError(true);
                        }
                      }}
                    >
                      Eliminar copia
                    </button>
                  )}
                </div>
              </div>
              {entry.infographic && (
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
            <p className="muted">Selecciona una plantilla.</p>
          )}
        </div>
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
                  persist([...saved, item]);
                  setSelected(item.id);
                  setArea("All");
                  setError(false);
                  setMessage(
                    "Copia guardada en este equipo. Exporta JSON para llevarla a otro equipo.",
                  );
                } catch (e) {
                  setError(true);
                  setMessage(String(e));
                }
              }}
            >
              <Save size={14} />
              Guardar copia
            </button>
          </div>
          <p role="status" className={error ? "library-error" : ""}>
            {message ||
              "Al cargar se archiva automáticamente el diseño actual en Personal. El JSON guarda nodos y parámetros; los modelos se entrenan de nuevo al ejecutar."}
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
              persist([...saved, ...imported]);
              setSelected(imported[0].id);
              setArea("All");
              setError(false);
              setMessage(`${imported.length} blueprints importados.`);
            } catch (err) {
              setError(true);
              setMessage(err instanceof Error ? err.message : String(err));
            }
          }}
        />
      </section>
    </div>
  );
}
