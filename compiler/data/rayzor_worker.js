// Rayzor Worker Thread Runtime
// Loaded by each Web Worker in the thread pool.
// Receives shared memory + WASM module, instantiates with shared imports.

let wasmInstance = null;
let wasmMemory = null;
let threadId = 0;

self.onmessage = async (e) => {
  const msg = e.data;

  if (msg.type === 'init') {
    // Initialize worker with shared memory and WASM module
    threadId = msg.threadId;
    wasmMemory = msg.memory;

    // Instantiate the same WASM module with shared memory
    const imports = buildImports(wasmMemory, msg.table);
    const { instance } = await WebAssembly.instantiate(msg.module, imports);
    wasmInstance = instance;

    // Override memory with shared memory (already set via imports)
    self.postMessage({ type: 'ready', threadId });
  }

  else if (msg.type === 'run') {
    // Execute a closure: table.get(fnIdx)(envPtr)
    const { taskId, fnIdx, envPtr } = msg;
    try {
      const table = wasmInstance.exports.__indirect_function_table;
      const fn = table.get(fnIdx);
      const result = fn ? fn(envPtr) : 0;
      self.postMessage({ type: 'done', taskId, threadId, result });
    } catch (err) {
      self.postMessage({ type: 'error', taskId, threadId, error: err.message });
    }
  }
};

function buildImports(memory, table) {
  // Minimal rayzor runtime for worker threads.
  // Workers share memory but need their own import stubs.
  const rayzor = new Proxy({
    malloc: (size) => {
      // Workers need their own bump allocator region.
      // Use Atomics on a shared heap pointer.
      const view = new Int32Array(memory.buffer);
      const HEAP_PTR_OFFSET = 256; // shared heap pointer at byte 1024
      const aligned = (n) => (n + 7) & ~7;
      const sz = aligned(Number(size));
      const ptr = Atomics.add(view, HEAP_PTR_OFFSET, sz);
      return ptr;
    },
    free: () => {},
    haxe_alloc: (size) => rayzor.malloc(Number(size)),
    haxe_trace_string_struct: (ptr) => {
      if (!ptr || !memory) return;
      const view = new DataView(memory.buffer);
      const dataPtr = view.getUint32(ptr, true);
      const len = view.getUint32(ptr + 4, true);
      if (len <= 0 || len > 1000000) return;
      const bytes = new Uint8Array(memory.buffer, dataPtr, len);
      const str = new TextDecoder().decode(bytes);
      self.postMessage({ type: 'trace', threadId, message: str });
    },
    trace: (ptr) => rayzor.haxe_trace_string_struct(ptr),
  }, {
    get: (target, prop) => {
      if (prop in target) return target[prop];
      return (...args) => {
        if (typeof args[0] === 'bigint') return BigInt(0);
        return 0;
      };
    }
  });

  const wasi = new Proxy({
    fd_write: () => 0,
    environ_get: () => 0,
    environ_sizes_get: () => 0,
    proc_exit: () => {},
    fd_close: () => 0,
    fd_read: () => 0,
    fd_fdstat_get: () => 0,
    fd_filestat_get: () => 0,
    fd_prestat_get: () => 8,
    fd_prestat_dir_name: () => 8,
    path_open: () => 44,
  }, {
    get: (target, prop) => target[prop] ?? (() => 0),
  });

  return {
    rayzor,
    wasi_snapshot_preview1: wasi,
  };
}
