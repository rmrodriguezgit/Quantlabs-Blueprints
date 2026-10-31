import { app, BrowserWindow, ipcMain } from "electron";
import path from "node:path";
function createWindow(): void {
  const window = new BrowserWindow({
    width: 1560,
    height: 1000,
    minWidth: 1050,
    minHeight: 720,
    backgroundColor: "#0d1118",
    title: "Quant Blueprint Studio",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });
  window.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
  window.webContents.on("will-navigate", (event) => event.preventDefault());
  if (app.isPackaged || process.argv.includes("--production"))
    void window.loadFile(path.join(__dirname, "../renderer/index.html"));
  else void window.loadURL("http://127.0.0.1:5173");
}
ipcMain.handle(
  "engine:execute",
  async (event, graph: unknown): Promise<unknown> => {
    try {
      return await new Promise<unknown>((resolve, reject) => {
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
            const item = value as Record<string, unknown>;
            if (item.event === "result") {
              settled = true;
              resolve(item.result);
              socket.close();
            } else if (item.event === "error") fail(String(item.detail));
            else if (!event.sender.isDestroyed())
              event.sender.send("engine:progress", value);
          } catch {
            fail("Invalid message received from the Python engine.");
          }
        };
        socket.onerror = () =>
          fail(
            "Cannot connect to Python engine. Restart the launcher (port 8000).",
          );
        socket.onclose = () => {
          if (!settled) fail("Engine connection closed before completion.");
        };
      });
    } catch (error) {
      return { error: error instanceof Error ? error.message : String(error) };
    }
  },
);
void app.whenReady().then(() => {
  createWindow();
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});
app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});
