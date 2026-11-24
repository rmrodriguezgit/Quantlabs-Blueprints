import type { Graph } from "./types/blueprint";
/** Browser/dev fallback only; Electron routes through the main-process bridge. */
const ENGINE_HOST = `127.0.0.1:${(import.meta as { env?: Record<string, string | undefined> }).env?.VITE_ENGINE_PORT ?? "8000"}`;
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
    const socket = new WebSocket(`ws://${ENGINE_HOST}/ws/execute`);
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
        `Cannot connect to the Python engine at ${ENGINE_HOST}. Restart the launcher.`,
      );
    socket.onclose = () => {
      if (!settled)
        fail(
          "Engine connection closed before completion. Check the launcher console.",
        );
    };
  });
}

/** JSON request to an allow-listed engine route (Electron bridge, else direct HTTP in dev). */
export async function engineRequest<T>(route: string, body?: unknown): Promise<T> {
  let value: unknown;
  if (window.quantEngine?.request) value = await window.quantEngine.request(route, body);
  else {
    const response = await fetch(`http://${ENGINE_HOST}${route}`, {
      method: body === undefined ? "GET" : "POST",
      headers: { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    value = await response.json();
    if (!response.ok) throw new Error(JSON.stringify((value as { detail?: unknown }).detail ?? response.status));
  }
  // The bridge reports failures as exactly { error }; payloads may carry their own error field.
  if (typeof value === "object" && value && Object.keys(value).length === 1 && "error" in value)
    throw new Error(String((value as { error: unknown }).error));
  return value as T;
}
