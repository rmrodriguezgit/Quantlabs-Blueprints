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
  async (_event, graph: unknown): Promise<unknown> => {
    const response = await fetch("http://127.0.0.1:8000/api/execute", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(graph),
    });
    const result: unknown = await response.json();
    if (!response.ok) throw new Error(JSON.stringify(result));
    return result;
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
