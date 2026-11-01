import type { StudyReport } from "../types/blueprint";
import { PlotDock } from "./PlotDock";
export function StudyDock({
  studies,
  onInspect,
}: {
  studies: StudyReport[];
  onInspect: (id: string) => void;
}) {
  const curves = studies.filter((study) => study.topic === "curve_fit");
  const benchmark = curves.flatMap((study) => study.table);
  const columns = [...new Set(benchmark.flatMap((row) => Object.keys(row)))];
  return (
    <>
      {curves.length > 0 && (
        <section className="study-dock" aria-label="Notebook benchmark">
          <header>
            <small>NOTEBOOK / BENCHMARK</small>
            <h2>EUR/USD · comparación de {curves.length} modelos</h2>
            <p>
              Errores en USD/EUR; RMSE pips usa 0.0001. Los tiempos corresponden
              a este equipo y ejecución.
            </p>
          </header>
          <div className="study-table">
            <table>
              <thead>
                <tr>
                  {columns.map((c) => (
                    <th key={c}>{c}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {benchmark.map((row, i) => (
                  <tr key={i}>
                    {columns.map((c) => (
                      <td key={c}>
                        {typeof row[c] === "number"
                          ? Number(row[c]).toPrecision(6)
                          : (row[c] ?? "—")}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {studies.map((study) => {
        const columns = [
          ...new Set(study.table.flatMap((row) => Object.keys(row))),
        ];
        return (
          <section key={study.id} className="study-dock">
            <header>
              <small>LEARNING EXPERIMENT / {study.topic}</small>
              <h2>{study.title}</h2>
              <p>
                {study.scope && <strong>{study.scope} · </strong>}
                Seleccionado: <strong>{study.selected}</strong> ·{" "}
                {study.features.length} características · TR{" "}
                {study.splits.train} / VA {study.splits.validation} / TE{" "}
                {study.splits.test}
              </p>
            </header>
            {study.architecture && (
              <details className="network-summary">
                <summary>
                  Arquitectura ·{" "}
                  {study.architecture.total_parameters.toLocaleString()}{" "}
                  parámetros ·{" "}
                  {study.architecture.trainable_parameters.toLocaleString()}{" "}
                  entrenables
                </summary>
                <div className="study-table">
                  <table>
                    <thead>
                      <tr>
                        {[
                          "Capa",
                          "Operación",
                          "Entrada",
                          "Salida",
                          "Parámetros",
                        ].map((c) => (
                          <th key={c}>{c}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {study.architecture.layers.map((layer) => (
                        <tr key={layer.name}>
                          <td>{layer.name}</td>
                          <td>{layer.operation}</td>
                          <td>{layer.input_shape.join(" × ")}</td>
                          <td>{layer.output_shape.join(" × ")}</td>
                          <td>{layer.parameters}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </details>
            )}
            <div className="study-metrics">
              {Object.entries(study.metrics).map(([key, v]) => (
                <article key={key}>
                  <small>{key}</small>
                  <strong>
                    {v == null ? "N/A" : Number(v).toPrecision(5)}
                  </strong>
                </article>
              ))}
            </div>
            <div className="study-table">
              <table>
                <thead>
                  <tr>
                    {columns.map((c) => (
                      <th key={c}>{c}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {study.table.map((row, i) => (
                    <tr key={i}>
                      {columns.map((c) => (
                        <td key={c}>
                          {typeof row[c] === "number"
                            ? Number(row[c]).toPrecision(5)
                            : (row[c] ?? "—")}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {study.notes.map((n, i) => (
              <p className="muted" key={i}>
                {n}
              </p>
            ))}
            <PlotDock
              plots={study.charts.map((c, i) => ({
                ...c,
                id: study.id + ":" + i,
                origin: study.selected,
                source: "study",
              }))}
              onInspect={() => onInspect(study.id)}
            />
          </section>
        );
      })}
    </>
  );
}
