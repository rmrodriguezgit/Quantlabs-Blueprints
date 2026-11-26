import { useState } from "react";
import { MessageSquareText, TriangleAlert } from "lucide-react";
import { explainResult, type Explanation } from "../brain";

/** "Explicar este resultado": deterministic reading plus an optional local-model reading. */
export function ExplainBox({ report }: { report: unknown }) {
  const [value, setValue] = useState<Explanation | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [rulesView, setRulesView] = useState(false);
  const run = async () => {
    setBusy(true);
    setError("");
    try {
      setValue(await explainResult(report));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };
  const view = value && (rulesView ? value.rules : value);
  return (
    <div className="explain-box">
      <button disabled={busy} onClick={() => void run()}>
        <MessageSquareText size={14} />
        {busy ? "Interpretando…" : value ? "Volver a explicar" : "Explicar este resultado"}
      </button>
      {error && <p className="brain-error">{error}</p>}
      {view && value && (
        <article>
          <header>
            <small>
              {rulesView || value.engine === "rules" ? "LECTURA POR REGLAS" : `QWEN LOCAL · ${value.model}`}
              {value.fallback_reason && !rulesView && ` · sin modelo: ${value.fallback_reason}`}
            </small>
            {value.engine === "ollama" && (
              <label>
                <input type="checkbox" checked={rulesView} onChange={(e) => setRulesView(e.target.checked)} /> Ver
                reglas
              </label>
            )}
          </header>
          <p>{view.summary}</p>
          {[
            ["Qué dicen las cifras", view.findings],
            ["Cautelas", view.cautions],
            ["Siguiente paso", view.next_steps],
          ].map(
            ([title, items]) =>
              (items as string[]).length > 0 && (
                <section key={title as string}>
                  <strong>{title as string}</strong>
                  <ul>
                    {(items as string[]).map((item, i) => (
                      <li key={i}>{item}</li>
                    ))}
                  </ul>
                </section>
              ),
          )}
          {!rulesView && value.unverified.length > 0 && (
            <p className="explain-unverified">
              <TriangleAlert size={13} /> Cifras no encontradas en el resultado (no verificadas):{" "}
              {value.unverified.join(", ")}
            </p>
          )}
        </article>
      )}
    </div>
  );
}
