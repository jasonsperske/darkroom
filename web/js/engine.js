// Promise-based client for the engine worker (workers/engine.worker.js).

export class Engine {
  constructor() {
    this.worker = new Worker(new URL('../workers/engine.worker.js', import.meta.url), { type: 'module' });
    this.pending = new Map();
    this.nextId = 1;
    this.ready = new Promise((resolve, reject) => {
      this.worker.onerror = (e) => reject(new Error(e.message || 'engine failed to start'));
      this.worker.onmessage = ({ data }) => {
        if (data.ready) {
          this.worker.onmessage = (ev) => this.#receive(ev.data);
          resolve();
        }
      };
    });
  }

  #receive({ id, ok, result, error, progress }) {
    const p = this.pending.get(id);
    if (!p) return;
    if (progress !== undefined) {
      p.onProgress?.(progress);
      return;
    }
    this.pending.delete(id);
    ok ? p.resolve(result) : p.reject(new Error(error));
  }

  /** Call a handler in the engine. `transfer` moves ArrayBuffers instead of copying them. */
  async call(type, payload = {}, { transfer = [], onProgress } = {}) {
    // Messages sent before the worker has finished loading its wasm could be dropped.
    await this.ready;
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, onProgress });
      this.worker.postMessage({ id, type, payload }, transfer);
    });
  }
}
