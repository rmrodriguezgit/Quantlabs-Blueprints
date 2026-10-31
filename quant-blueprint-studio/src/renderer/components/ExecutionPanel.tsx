import { useEffect, useRef } from "react";
import type { NetworkBinding } from "../types/blueprint";
import type { ProgressEvent } from "../engine";
export interface ExecutionEntry extends ProgressEvent {
  label: string;
  time: number;
}
interface Props {
  bindings: NetworkBinding[];
  entries: ExecutionEntry[];
  running: boolean;
  error: boolean;
  message: string;
  elapsed: number;
  total: number;
  onInspect: (id: string) => void;
}
export function ExecutionPanel({
  bindings,
  entries,
  running,
  error,
  message,
  elapsed,
  total,
  onInspect,
}: Props) {
  const log = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (log.current) log.current.scrollTop = log.current.scrollHeight;
  }, [entries]);
  const completed = new Set(
    entries.filter((e) => e.status === "complete").map((e) => e.node),
  ).size;
  const latest = entries.at(-1);
  return (
    <section className="execution-panel" aria-label="Execution monitor">
      <div className="network-overview">
        {bindings.map((binding) => (
          <div
            key={binding.trainerId}
            className={binding.issue ? "network-issue" : ""}
          >
            <button
              onClick={() => onInspect(binding.inputId ?? binding.trainerId)}
              title="Inspect network input"
            >
              {binding.summary}
            </button>
            <span>
              {binding.selected.length} selected / {binding.declared} declared{" "}
              {binding.issue ? "· mismatch" : "· matched"}
            </span>
            {binding.issue && <p role="alert">{binding.issue}</p>}
          </div>
        ))}
      </div>
      <div className={`run-monitor ${error ? "failed" : ""}`}>
        <div className="run-heading">
          <strong>
            {running
              ? "⟳ Calculating"
              : error
                ? "Execution stopped"
                : entries.length
                  ? "Execution complete"
                  : "Ready to execute"}
          </strong>
          <span>
            {completed}/{total} nodes complete · {elapsed}s
          </span>
        </div>
        <progress
          max={Math.max(total, 1)}
          value={completed}
          aria-label="Completed nodes"
        />
        {running && (
          <p aria-live="polite">
            {latest
              ? `${latest.label}${latest.status === "epoch" ? ` · epoch ${latest.epoch}/${latest.epochs} · validation loss ${latest.validation?.toPrecision(5)}` : latest.status === "running" ? " · running…" : " · complete"}`
              : "Connecting to Python engine…"}
          </p>
        )}
        {error && (
          <p className="execution-error" role="alert">
            {message}
          </p>
        )}
        <div
          ref={log}
          className="execution-log"
          role="log"
          aria-label="Node execution events"
        >
          {entries.slice(-200).map((entry, i) => (
            <div key={`${entry.time}-${i}`}>
              <time>{entry.time.toFixed(1)}s</time>
              <b>{entry.label}</b>
              <span>
                {entry.status === "epoch"
                  ? `epoch ${entry.epoch}/${entry.epochs} · train ${entry.train?.toPrecision(4)} · validation ${entry.validation?.toPrecision(4)}`
                  : entry.status === "complete"
                    ? `complete${entry.rows !== undefined ? ` · ${entry.rows} rows / ${entry.source_rows ?? entry.rows} source rows` : ""}${entry.features !== undefined ? ` · ${entry.features} features` : ""}`
                    : entry.status === "error"
                      ? entry.detail
                      : "running…"}
              </span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
