import type { Graph } from "./types/blueprint";
export interface ProgressEvent {
  node: string;
  status: "running" | "complete" | "epoch" | "error";
  epoch?: number;
  epochs?: number;
  train?: number;
  validation?: number;
  rows?: number;
  source_rows?: number;
  features?: number;
  detail?: string;
}
export function progressEvent(value: unknown): ProgressEvent | null {
  if (typeof value !== "object" || !value) return null;
  const item = value as Record<string, unknown>;
  if (
    typeof item.node !== "string" ||
    !["running", "complete", "epoch", "error"].includes(String(item.status))
  )
    return null;
  return item as unknown as ProgressEvent;
}
export async function executeGraph(
  graph: Graph,
  onProgress: (event: ProgressEvent) => void,
): Promise<unknown> {
  const receive = (value: unknown) => {
    const event = progressEvent(value);
    if (event) onProgress(event);
  };
  if (window.quantEngine) {
    const unsubscribe = window.quantEngine.onProgress(receive);
    try {
      const value = await window.quantEngine.execute(graph);
      if (typeof value === "object" && value && "error" in value)
        throw new Error(String(value.error));
      return value;
    } finally {
      unsubscribe();
    }
  }
  return new Promise((resolve, reject) => {
    const socket = new WebSocket("ws://127.0.0.1:8000/ws/execute");
    let settled = false;
    const fail = (message: string) => {
      if (settled) return;
      settled = true;
      reject(new Error(message));
      socket.close();
    };
    socket.onopen = () => socket.send(JSON.stringify(graph));
    socket.onmessage = (message) => {
      try {
        const value: unknown = JSON.parse(String(message.data));
        if (typeof value !== "object" || !value) return;
        const event = value as Record<string, unknown>;
        if (event.event === "result") {
          settled = true;
          resolve(event.result);
          socket.close();
        } else if (event.event === "error") fail(String(event.detail));
        else receive(value);
      } catch {
        fail("The engine returned an invalid progress message.");
      }
    };
    socket.onerror = () =>
      fail(
        "Cannot connect to the Python engine. Restart the launcher and check port 8000.",
      );
    socket.onclose = () => {
      if (!settled)
        fail(
          "Engine connection closed before completion. Check the launcher console.",
        );
    };
  });
}
