import { Settings2, X } from "lucide-react";
import { catalog, type BlueprintNode, type Param } from "../types/blueprint";
interface Props {
  node: BlueprintNode | undefined;
  onChange: (key: string, value: Param) => void;
  onDelete: () => void;
  disabled: boolean;
  availableFeatures: string[];
  onDuplicate: () => void;
}
export function DetailsInspector({
  node,
  onChange,
  onDelete,
  disabled,
  availableFeatures,
  onDuplicate,
}: Props) {
  return (
    <aside className="inspector">
      <div className="aside-heading">
        DETAILS INSPECTOR <Settings2 size={14} />
      </div>
      {node ? (
        <>
          <div className="inspector-title">
            <span style={{ background: catalog[node.data.kind].color }} />
            {node.data.label}
          </div>
          <p className="muted">{catalog[node.data.kind].subtitle}</p>
          {node.data.kind === "network_input" && (
            <p className="muted">
              last: N features from the last bar. flatten: lookback × N inputs.
            </p>
          )}
          {node.data.kind === "network_output" && (
            <p className="muted">
              Connect the purple output to the trainer’s purple network pin.
            </p>
          )}
          <fieldset disabled={disabled}>
            {catalog[node.data.kind].fields.map((field) => (
              <div className="inspector-field" key={field.key}>
                <label htmlFor={`param-${field.key}`}>{field.label}</label>
                {field.key === "feature_columns" ? (
                  <div className="feature-selector">
                    <div className="feature-actions">
                      <button
                        type="button"
                        onClick={() => onChange(field.key, "")}
                      >
                        All
                      </button>
                      <button
                        type="button"
                        onClick={() =>
                          onChange(
                            field.key,
                            availableFeatures.slice(0, 16).join(","),
                          )
                        }
                      >
                        First 16
                      </button>
                    </div>
                    <small>
                      {String(node.data.params.feature_columns ?? "").trim()
                        ? String(node.data.params.feature_columns)
                            .split(",")
                            .filter(Boolean).length
                        : availableFeatures.length}{" "}
                      input features selected
                    </small>
                    {availableFeatures.length === 0 && (
                      <p>Connect a feature source first.</p>
                    )}
                    {availableFeatures.map((name) => {
                      const raw = String(
                        node.data.params.feature_columns ?? "",
                      ).trim();
                      const selected = raw
                        ? raw.split(",").map((s) => s.trim())
                        : availableFeatures;
                      return (
                        <label key={name}>
                          <input
                            type="checkbox"
                            checked={selected.includes(name)}
                            onChange={(e) => {
                              const next = e.target.checked
                                ? [...selected, name]
                                : selected.filter((n) => n !== name);
                              if (next.length)
                                onChange(field.key, next.join(","));
                            }}
                          />
                          {name}
                        </label>
                      );
                    })}
                  </div>
                ) : field.options ? (
                  <select
                    id={`param-${field.key}`}
                    value={String(node.data.params[field.key] ?? field.value)}
                    onChange={(e) => onChange(field.key, e.target.value)}
                  >
                    {field.options.map((option) => (
                      <option key={option}>{option}</option>
                    ))}
                  </select>
                ) : typeof field.value === "boolean" ? (
                  <input
                    id={`param-${field.key}`}
                    type="checkbox"
                    checked={Boolean(
                      node.data.params[field.key] ?? field.value,
                    )}
                    onChange={(e) => onChange(field.key, e.target.checked)}
                  />
                ) : (
                  <input
                    id={`param-${field.key}`}
                    type={typeof field.value === "number" ? "number" : "text"}
                    value={String(node.data.params[field.key] ?? field.value)}
                    min={field.min}
                    max={field.max}
                    step={field.step ?? 1}
                    onChange={(e) => {
                      if (typeof field.value === "number") {
                        const value = Number(e.target.value);
                        if (Number.isFinite(value)) onChange(field.key, value);
                      } else onChange(field.key, e.target.value);
                    }}
                  />
                )}
              </div>
            ))}
          </fieldset>
          <div className="inspector-hint">
            Parameters are applied on the next execution. A connected config
            node overrides matching fields.
          </div>
          <button disabled={disabled} onClick={onDuplicate}>
            Duplicate node
          </button>
          <button
            className="delete-button"
            disabled={disabled}
            onClick={onDelete}
          >
            <X size={14} /> Remove node
          </button>
        </>
      ) : (
        <div className="inspector-empty">
          <Settings2 size={30} />
          <h3>Inspect a node</h3>
          <p>Select a node to edit its parameters and architecture.</p>
        </div>
      )}
    </aside>
  );
}
