import { Settings2, X } from "lucide-react";
import { catalog, type BlueprintNode, type Param } from "../types/blueprint";
interface Props {
  node: BlueprintNode | undefined;
  onChange: (key: string, value: Param) => void;
  onDelete: () => void;
  disabled: boolean;
}
export function DetailsInspector({
  node,
  onChange,
  onDelete,
  disabled,
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
          <fieldset disabled={disabled}>
            {catalog[node.data.kind].fields.map((field) => (
              <label key={field.key}>
                {field.label}
                {field.options ? (
                  <select
                    value={String(node.data.params[field.key] ?? field.value)}
                    onChange={(e) => onChange(field.key, e.target.value)}
                  >
                    {field.options.map((option) => (
                      <option key={option}>{option}</option>
                    ))}
                  </select>
                ) : typeof field.value === "boolean" ? (
                  <input
                    type="checkbox"
                    checked={Boolean(
                      node.data.params[field.key] ?? field.value,
                    )}
                    onChange={(e) => onChange(field.key, e.target.checked)}
                  />
                ) : (
                  <input
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
              </label>
            ))}
          </fieldset>
          <div className="inspector-hint">
            Parameters are applied on the next execution. A connected config
            node overrides matching fields.
          </div>
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
