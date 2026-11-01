import { contextBridge, ipcRenderer, type IpcRendererEvent } from "electron";
contextBridge.exposeInMainWorld("quantEngine", {
  onNewBlueprint: (callback: () => void): (() => void) => {
    const listener = () => callback();
    ipcRenderer.on("blueprint:new", listener);
    ipcRenderer.send("blueprint:ready");
    return () => {
      ipcRenderer.removeListener("blueprint:new", listener);
    };
  },
  onProgress: (callback: (event: unknown) => void): (() => void) => {
    const listener = (_event: IpcRendererEvent, value: unknown) =>
      callback(value);
    ipcRenderer.on("engine:progress", listener);
    return () => ipcRenderer.removeListener("engine:progress", listener);
  },
  execute: (graph: unknown): Promise<unknown> =>
    ipcRenderer.invoke("engine:execute", graph),
});
