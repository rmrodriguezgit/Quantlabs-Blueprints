import type { ScalingAudit } from "../types/blueprint";
const number = (value: number) => Number(value).toPrecision(6);
export function ScalingDock({
  audits,
  onInspect,
}: {
  audits: ScalingAudit[];
  onInspect: (id: string) => void;
}) {
  if (!audits.length) return null;
  const firstActive = Math.max(
    0,
    audits.findIndex((a) => a.method === "StandardScaler"),
  );
  return (
    <section className="study-dock scaling-dock" aria-label="Scaling audit">
      <header>
        <small>PREPROCESSING / STANDARD SCALER</small>
        <h2>Normalización · auditoría</h2>
        <p>
          Comprueba qué se escala y con qué observaciones se ajusta. En
          protocolos temporales, VA/TE sólo se transforman. En ajuste completo
          se usan todos los datos de ajuste.
        </p>
      </header>
      <p className="muted">
        StandardScaler usa z = (x − media) / escala, por característica. La
        media ≈ 0 y desviación ≈ 1 se comprueban sobre las filas usadas para
        ajustar; no se exigen en VA/TE. Las columnas constantes quedan en cero.
      </p>
      {audits.map((audit, i) => (
        <details key={`${audit.id}-${i}`} open={i === firstActive}>
          <summary>
            <strong>
              {audit.node_title} · {audit.title}
            </strong>
            <span
              className={
                audit.method === "Disabled" ? "scaler-off" : "scaler-on"
              }
            >
              {audit.method === "Disabled" ? "Sin escalado" : audit.method}
            </span>
            <span>
              {audit.fit_rows} filas · {audit.scope}
            </span>
          </summary>
          <p>
            {audit.note}{" "}
            {audit.fit_start && `${audit.fit_start} → ${audit.fit_end}`}
          </p>
          {audit.inverse_before_metrics && (
            <p>
              La transformación del objetivo se invierte antes de calcular las
              métricas.
            </p>
          )}
          {audit.columns.length > 0 && (
            <div className="study-table">
              <table>
                <thead>
                  <tr>
                    {[
                      "Característica",
                      "Media original",
                      "Varianza (ddof=0)",
                      "Centro",
                      "Divisor",
                      "Media transformada",
                      "Desv. transformada",
                      "Constante",
                    ].map((c) => (
                      <th key={c}>{c}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {audit.columns.map((c) => (
                    <tr key={c.name}>
                      <td>{c.name}</td>
                      {[
                        c.mean,
                        c.variance,
                        c.center,
                        c.scale,
                        c.transformed_mean,
                        c.transformed_std,
                      ].map((v, j) => (
                        <td key={j}>{number(v)}</td>
                      ))}
                      <td>{c.constant ? "Sí" : "No"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <button onClick={() => onInspect(audit.id)}>Inspeccionar nodo</button>
        </details>
      ))}
    </section>
  );
}
