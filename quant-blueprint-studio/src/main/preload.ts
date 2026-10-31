import { contextBridge, ipcRenderer } from "electron";
contextBridge.exposeInMainWorld("quantEngine", {
  execute: (graph: unknown): Promise<unknown> =>
    ipcRenderer.invoke("engine:execute", graph),
});
