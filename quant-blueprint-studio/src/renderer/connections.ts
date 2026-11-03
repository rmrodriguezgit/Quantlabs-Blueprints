import type { Connection, Edge } from "@xyflow/react";
import { catalog, inputHandles, type Graph } from "./types/blueprint";
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
