import {
  app,
  BrowserWindow,
  ipcMain,
  Menu,
  type MenuItemConstructorOptions,
} from "electron";
import path from "node:path";
const pendingNewBlueprints = new Set<number>();
function createWindow(newBlueprint = false): void {
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
  const webContentsId = window.webContents.id;
  if (newBlueprint) pendingNewBlueprints.add(webContentsId);
  window.on("closed", () => pendingNewBlueprints.delete(webContentsId));
  window.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
  window.webContents.on("will-navigate", (event) => event.preventDefault());
  if (app.isPackaged || process.argv.includes("--production"))
    void window.loadFile(path.join(__dirname, "../renderer/index.html"));
  else void window.loadURL("http://127.0.0.1:5173");
}
ipcMain.on("blueprint:ready", (event) => {
  if (pendingNewBlueprints.delete(event.sender.id))
    event.sender.send("blueprint:new");
});
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
  const menu: MenuItemConstructorOptions[] = [];
  if (process.platform === "darwin") menu.push({ role: "appMenu" });
  menu.push(
    {
      label: "File",
      submenu: [
        {
          label: "New",
          accelerator: "CmdOrCtrl+N",
          click: () => {
            const target =
              BrowserWindow.getFocusedWindow() ??
              BrowserWindow.getAllWindows()[0];
            if (target) target.webContents.send("blueprint:new");
            else createWindow(true);
          },
        },
        { type: "separator" },
        { role: "close" },
      ],
    },
    { role: "editMenu" },
    { role: "viewMenu" },
    { role: "windowMenu" },
  );
  Menu.setApplicationMenu(Menu.buildFromTemplate(menu));
  createWindow();
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});
app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});
