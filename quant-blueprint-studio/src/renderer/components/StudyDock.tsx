import type { StudyReport } from "../types/blueprint";
import { PlotDock } from "./PlotDock";
export function StudyDock({
  studies,
  onInspect,
}: {
  studies: StudyReport[];
  onInspect: (id: string) => void;
}) {
  return (
    <>
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
                Seleccionado: <strong>{study.selected}</strong> ·{" "}
                {study.features.length} características · TR{" "}
                {study.splits.train} / VA {study.splits.validation} / TE{" "}
                {study.splits.test}
              </p>
            </header>
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
