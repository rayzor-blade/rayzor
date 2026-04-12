// Rayzor Worker Thread Runtime
// Loaded by each Web Worker in the thread pool.
// Receives shared memory + WASM module, instantiates with shared imports.

let wasmInstance = null;
let wasmMemory = null;
let threadId = 0;

self.onmessage = async (e) => {
  const msg = e.data;

  if (msg.type === 'init') {
    // Initialize worker with shared memory and WASM module. Note: we do
    // NOT receive the `__indirect_function_table` from the main thread
    // — `WebAssembly.Table` is not structured-cloneable. Each worker
    // gets its own table from `instance.exports.__indirect_function_table`
    // after instantiating the module below; because the module is the
    // same compiled `WebAssembly.Module`, the table has identical func
    // indices, so closures spawned via `Thread.spawn` look up the same
    // function on the worker side.
    threadId = msg.threadId;
    wasmMemory = msg.memory;

    const imports = buildImports(wasmMemory);
    const { instance } = await WebAssembly.instantiate(msg.module, imports);
    wasmInstance = instance;

    self.postMessage({ type: 'ready', threadId });
  }

  else if (msg.type === 'run') {
    // Execute a closure: table.get(fnIdx)(envPtr). The result must be
    // delivered back to the main thread via the shared-memory slot the
    // main thread allocated, not via postMessage — the main thread is
    // blocked inside a synchronous WASM call and can't pump its event
    // loop to receive `postMessage` until `join()` returns. Layout:
    //   +0  i32 ready flag (0 = pending, 1 = done)
    //   +8  i64 closure result
    const { threadId: tid, fnIdx, envPtr, slot } = msg;
    try {
      const table = wasmInstance.exports.__indirect_function_table;
      const fn = table.get(fnIdx);
      let result = 0;
      if (typeof fn === 'function') {
        result = fn(envPtr);
      } else {
        console.error(`[worker ${threadId}] table slot ${fnIdx} is not callable`);
      }
      // Write the result first, then publish the ready flag — matches
      // the acquire-release pattern the main thread spins on.
      const dv = new DataView(wasmMemory.buffer);
      dv.setBigInt64(slot + 8, BigInt(result), true);
      const view = new Int32Array(wasmMemory.buffer);
      Atomics.store(view, slot >> 2, 1);
      Atomics.notify(view, slot >> 2, 1);
    } catch (err) {
      console.error(`[worker ${threadId}] closure threw:`, err);
      // Publish the ready flag even on failure so join() doesn't spin
      // forever. Result slot stays 0, which matches the legacy behavior.
      try {
        const view = new Int32Array(wasmMemory.buffer);
        Atomics.store(view, slot >> 2, 1);
        Atomics.notify(view, slot >> 2, 1);
      } catch (_) { /* nothing we can do */ }
      self.postMessage({ type: 'error', threadId: tid, error: err.message });
    }
  }
};

function buildImports(memory) {
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

  // The Rayzor runtime-wasm library imports its linear memory from
  // `env.memory`. Workers must provide the SAME `WebAssembly.Memory` handle
  // the main thread allocated — this is how all wasm instances end up
  // sharing the SharedArrayBuffer that backs the heap and sync primitives.
  return {
    env: { memory },
    rayzor,
    wasi_snapshot_preview1: wasi,
  };
}
