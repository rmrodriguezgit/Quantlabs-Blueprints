import { useState } from "react";
import { RefreshCw, Search } from "lucide-react";
import { engineRequest } from "../engine";
import type { Param } from "../types/blueprint";

interface Hit {
  slug: string;
  question: string;
  closed: boolean;
  end: string | null;
  outcomes: string[];
  window: string | null;
}
interface Meta {
  slug: string;
  question: string;
  up_outcome: string;
  down_outcome: string;
  price_up: number | null;
  window: string | null;
  end: string | null;
  closed: boolean;
  accepting_orders: boolean;
  liquidity: number;
}
interface Props {
  market: string;
  upOutcome: string;
  disabled: boolean;
  onChange: (key: string, value: Param) => void;
  onResolved?: (market: { question: string; slug: string; window: string | null }) => void;
}

/** Search, select and validate any Polymarket market from the Inspector. */
export function PolymarketPicker({ market, upOutcome, disabled, onChange, onResolved }: Props) {
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<Hit[]>([]);
  const [meta, setMeta] = useState<Meta | null>(null);
  const [status, setStatus] = useState<string>("");
  const [busy, setBusy] = useState(false);

  const run = async (fn: () => Promise<void>) => {
    setBusy(true);
    setStatus("");
    try {
      await fn();
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  };
  const search = () =>
    run(async () => {
      const value = await engineRequest<{ results: Hit[] }>("/api/polymarket/search", { query });
      setHits(value.results);
      if (!value.results.length) setStatus("Sin resultados.");
    });
  const validate = (ref = market) =>
    run(async () => {
      const value = await engineRequest<{ market: Meta }>("/api/polymarket/resolve", { query: ref, up_outcome: upOutcome });
      setMeta(value.market);
      onResolved?.(value.market);
    });

  return (
    <div className="polymarket-picker">
      <div className="polymarket-row">
        <input
          value={query}
          placeholder="Buscar: bitcoin up or down, elecciones…"
          disabled={disabled || busy}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && query.trim() && void search()}
        />
        <button disabled={disabled || busy || !query.trim()} onClick={() => void search()} title="Buscar en Gamma API">
          <Search size={13} />
        </button>
        <button disabled={disabled || busy || !market.trim()} onClick={() => void validate()} title="Validar / refrescar mercado">
          <RefreshCw size={13} className={busy ? "spin" : undefined} />
        </button>
      </div>
      {hits.length > 0 && (
        <ul className="polymarket-hits">
          {hits.map((hit) => (
            <li key={hit.slug}>
              <button
                disabled={disabled}
                onClick={() => {
                  onChange("market", hit.slug);
                  setHits([]);
                  void validate(hit.slug);
                }}
              >
                <strong>{hit.question}</strong>
                <span>
                  {hit.window ?? "—"} · {hit.outcomes.join(" / ")} · {hit.closed ? "cerrado" : "abierto"}
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
      {meta && (
        <div className="polymarket-meta">
          <strong>{meta.question}</strong>
          <span>
            UP = «{meta.up_outcome}» · DOWN = «{meta.down_outcome}» · P(UP) {meta.price_up?.toFixed(3) ?? "—"}
          </span>
          <span>
            Ventana detectada: {meta.window ?? "no detectada"} · {meta.closed ? "cerrado" : meta.accepting_orders ? "acepta órdenes" : "pausado"} · liquidez{" "}
            {Math.round(meta.liquidity).toLocaleString()}
          </span>
        </div>
      )}
      {status && <p className="inspector-hint">{status}</p>}
    </div>
  );
}

/** Online weights live in the engine per series; this wipes one series back to the Inspector weights. */
export function OnlineReset({ params, disabled }: { params: Record<string, Param>; disabled: boolean }) {
  const [key, setKey] = useState(String(params.learning_key ?? ""));
  const [status, setStatus] = useState("");
  if (params.weight_source !== "online") return null;
  return (
    <div className="polymarket-picker">
      <p className="inspector-hint">
        Online: los pesos parten de los sliders y, al resolverse cada ventana, se corrigen con
        w ← w − lr·((p − y)·z + L2·w). Se guardan por serie (p. ej. btc-updown-5m) y sobreviven reinicios.
      </p>
      <div className="polymarket-row">
        <input value={key} placeholder="serie, p. ej. btc-updown-5m" disabled={disabled} onChange={(e) => setKey(e.target.value)} />
        <button
          disabled={disabled || !key.trim()}
          onClick={() =>
            void engineRequest<{ reset: boolean }>("/api/polymarket/learning/reset", { key: key.trim() })
              .then((r) => setStatus(r.reset ? "Pesos aprendidos borrados: el próximo ciclo parte de los sliders." : "No había aprendizaje guardado para esa serie."))
              .catch((e: unknown) => setStatus(e instanceof Error ? e.message : String(e)))
          }
        >
          Reiniciar
        </button>
      </div>
      {status && <p className="inspector-hint">{status}</p>}
    </div>
  );
}
