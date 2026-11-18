// A pool of compute workers with a simple FIFO job queue.
//
// Why a pool? WebAssembly threads need SharedArrayBuffer, which needs cross-origin isolation
// headers (COOP/COEP) that GitHub Pages can't send. Instead we run N independent wasm
// instances in N workers and split work into pieces (e.g. horizontal strips of an image),
// moving buffers to and from them with zero-copy transfers.

export class WorkerPool {
  constructor(size) {
    this.size = size;
    this.queue = [];
    this.workers = [];
    this.nextId = 1;
    this.ready = Promise.all(Array.from({ length: size }, () => this.#spawn()));
  }

  #spawn() {
    return new Promise((resolve, reject) => {
      const worker = new Worker(new URL('./compute.worker.js', import.meta.url), { type: 'module' });
      const slot = { worker, busy: null };
      worker.onmessage = ({ data }) => {
        if (data.ready) {
          this.simd = data.simd;
          this.workers.push(slot);
          resolve();
          this.#pump();
          return;
        }
        const job = slot.busy;
        slot.busy = null;
        if (data.ok) job.resolve(data.result);
        else job.reject(new Error(data.error));
        if (data.fatal) this.#replace(slot);
        this.#pump();
      };
      worker.onerror = (e) => {
        e.preventDefault();
        if (slot.busy) slot.busy.reject(new Error(e.message || 'worker crashed'));
        slot.busy = null;
        if (this.workers.includes(slot)) this.#replace(slot);
        else reject(new Error(e.message || 'worker failed to start'));
      };
    });
  }

  #replace(slot) {
    slot.worker.terminate();
    this.workers = this.workers.filter((s) => s !== slot);
    this.#spawn().catch(() => {});
  }

  #pump() {
    for (const slot of this.workers) {
      if (slot.busy || !this.queue.length) continue;
      const job = this.queue.shift();
      slot.busy = job;
      slot.worker.postMessage({ id: job.id, type: job.type, payload: job.payload }, job.transfer);
    }
  }

  /** Run a job on the next free worker. `transfer` lists ArrayBuffers to move, not copy. */
  run(type, payload, transfer = []) {
    return new Promise((resolve, reject) => {
      this.queue.push({ id: this.nextId++, type, payload, transfer, resolve, reject });
      this.#pump();
    });
  }
}
