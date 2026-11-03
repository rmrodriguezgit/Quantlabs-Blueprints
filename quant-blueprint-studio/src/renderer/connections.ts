import contracts from "./data/node-contracts.json";
import type { Connection, Edge } from "@xyflow/react";
import {
  catalog,
  inputHandles,
  type Graph,
  type Kind,
} from "./types/blueprint";
export function acceptsPayload(
  source: Kind,
  target: Kind,
  pin: string,
): boolean {
  if (pin !== "tensor") return true;
  const sources: Record<string, string[]> = contracts.tensorSources;
  return sources[target]?.includes(source) ?? false;
}
export function validConnection(
  graph: Graph,
  connection: Connection | Edge,
): boolean {
  const src = graph.nodes.find((n) => n.id === connection.source);
  const dst = graph.nodes.find((n) => n.id === connection.target);
  if (
    !src ||
    !dst ||
    src.id === dst.id ||
    !acceptsPayload(
      src.data.kind,
      dst.data.kind,
      connection.targetHandle ?? "",
    ) ||
    !connection.targetHandle ||
    !connection.sourceHandle ||
    (dst.data.kind === "study" &&
      dst.data.params.topic !== "cdcgan" &&
      connection.targetHandle.startsWith("asset_")) ||
    catalog[src.data.kind].output !== connection.sourceHandle ||
    inputHandles(dst.data.kind)[connection.targetHandle] !==
      connection.sourceHandle
  )
    return false;
  if (
    graph.edges.some(
      (e) => e.target === dst.id && e.targetHandle === connection.targetHandle,
    )
  )
    return false;
  if (dst.data.kind === "plot" && graph.edges.some((e) => e.target === dst.id))
    return false;
  if (
    dst.data.kind === "risk" &&
    connection.targetHandle !== "benchmark" &&
    graph.edges.some(
      (e) => e.target === dst.id && e.targetHandle !== "benchmark",
    )
  )
    return false;
  const visited = new Set<string>();
  const reaches = (id: string): boolean => {
    if (id === src.id) return true;
    if (visited.has(id)) return false;
    visited.add(id);
    return graph.edges
      .filter((e) => e.source === id)
      .some((e) => reaches(e.target));
  };
  return !reaches(dst.id);
}

export function payloadLabel(kind: Kind): string {
  if (kind === "curve_fit" || kind === "study" || kind === "rl_agent")
    return "Informe de experimento";
  if (kind === "prepare") return "Datos TR / VA / TE";
  if (kind === "model" || kind === "ml_model") return "Modelo y predicciones";
  if (kind === "backtest") return "Resultado de backtest";
  if (kind === "risk") return "Informe de riesgo";
  if (kind === "portfolio_data") return "Datos de cartera";
  if (kind === "portfolio") return "Resultado de cartera";
  return catalog[kind].output ?? "Informe";
}
