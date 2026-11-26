import { useState } from "react";
import {
  CircleAlert,
  TriangleAlert,
  Info,
  Wrench,
  Undo2,
  ChevronDown,
  ChevronRight,
  CheckCircle2, Brain } from "lucide-react";
import type { Issue } from "../validation";

const icon = {
  error: <CircleAlert size={13} />,
  warning: <TriangleAlert size={13} />,
  info: <Info size={13} />,
};

export function ValidationPanel({
  issues,
  disabled,
  changes,
  canUndo,
  onSelect,
  onFix,
  onRepairAll,
  onUndo,
  execError = null,
  cerebroBusy = false,
  onCerebro,
}: {
  execError?: string | null;
  cerebroBusy?: boolean;
  onCerebro?: () => void;
  issues: Issue[];
  disabled: boolean;
  changes: string[];
  canUndo: boolean;
  onSelect: (node: string) => void;
  onFix: (issue: Issue) => void;
  onRepairAll: () => void;
  onUndo: () => void;
}) {
  const [open, setOpen] = useState(true);
  const [showInfo, setShowInfo] = useState(false);
  const count = (s: Issue["severity"]) => issues.filter((i) => i.severity === s).length;
  const errors = count("error");
  const warnings = count("warning");
  const infos = count("info");
  const fixable = issues.filter((i) => i.fix && i.severity !== "info").length;
  const shown = issues.filter((i) => showInfo || i.severity !== "info");
  const state = errors || execError ? "error" : warnings ? "warning" : "ok";
  return (
    <section className={`validation-panel ${state}`} aria-label="Validación del blueprint">
      <header>
        <button className="validation-toggle" onClick={() => setOpen(!open)} aria-expanded={open}>
          {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          {state === "ok" ? <CheckCircle2 size={14} /> : icon[state]}
          <strong>Validación</strong>
          <span>
            {errors} errores · {warnings} avisos · {infos} notas
          </span>
        </button>
        <div className="validation-actions">
          {infos > 0 && (
            <label>
              <input type="checkbox" checked={showInfo} onChange={(e) => setShowInfo(e.target.checked)} />
              Mostrar notas
            </label>
          )}
          {canUndo && (
            <button disabled={disabled} onClick={onUndo} title="Revertir la última reparación">
              <Undo2 size={14} /> Deshacer
            </button>
          )}
          {onCerebro && (
            <button
              className="repair-button cerebro-fix"
              disabled={disabled || cerebroBusy || (errors === 0 && !execError)}
              onClick={onCerebro}
              title="Cerebro corrige cualquier error: aplica las reparaciones, propone cambios para los errores de validación y de ejecución, y vuelve a ejecutar para comprobarlo (se puede deshacer)"
            >
              <Brain size={14} className={cerebroBusy ? "spin" : undefined} /> {cerebroBusy ? "Cerebro corrigiendo…" : "Corregir con Cerebro"}
            </button>
          )}
          <button
            className="repair-button"
            disabled={disabled || fixable === 0}
            onClick={onRepairAll}
            title="Aplica solo correcciones deterministas: rangos, opciones, conexiones incompatibles y entradas vacías"
          >
            <Wrench size={14} /> Reparar {fixable > 0 ? `(${fixable})` : ""}
          </button>
        </div>
      </header>
      {open && (
        <>
          {execError && (
            <p className="validation-exec-error">
              Error de ejecución: {execError}
            </p>
          )}
          {changes.length > 0 && (
            <details className="validation-changes" open>
              <summary>Última reparación · {changes.length} cambio(s)</summary>
              <ul>
                {changes.map((c, i) => (
                  <li key={i}>{c}</li>
                ))}
              </ul>
            </details>
          )}
          {shown.length === 0 ? (
            <p className="validation-ok">
              Sin problemas detectados antes de ejecutar. Los datos reales aún se comprueban en el motor.
            </p>
          ) : (
            <ul className="validation-list">
              {shown.map((issue) => (
                <li key={issue.id} className={issue.severity}>
                  {icon[issue.severity]}
                  <button
                    className="validation-message"
                    disabled={!issue.node}
                    onClick={() => issue.node && onSelect(issue.node)}
                    title={issue.node ? "Seleccionar el nodo" : undefined}
                  >
                    {issue.message}
                  </button>
                  {issue.fix && (
                    <button className="fix-button" disabled={disabled} onClick={() => onFix(issue)}>
                      {issue.fixLabel ?? "Corregir"}
                    </button>
                  )}
                </li>
              ))}
            </ul>
          )}
        </>
      )}
    </section>
  );
}
