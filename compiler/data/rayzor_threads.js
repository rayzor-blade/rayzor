// Rayzor Thread Runtime — Browser implementation via Web Workers
// Provides: thread spawn/join, mutex, semaphore, channel, future
//
// Requires: SharedArrayBuffer (COOP+COEP headers), Atomics
//
// Memory layout for sync primitives (in shared WASM memory):
//   Mutex:     [state: i32]  0=unlocked, 1=locked
//   Semaphore: [count: i32]
//   Channel:   [head: i32, tail: i32, closed: i32, cap: i32, data: i64[cap]]

export class RayzorThreadRuntime {
  constructor() {
    this.workers = [];
    this.idleWorkers = [];
    this.nextThreadId = 1;
    this.nextTaskId = 1;
    this.tasks = new Map();    // taskId → { resolve, reject, threadId }
    this.threads = new Map();  // threadId → { done, result, worker }
    this.wasmModule = null;
    this.memory = null;
    this.table = null;

    // Sync primitive + result slot regions. Addresses are assigned in
    // init() via `rayzor_malloc` so dlmalloc on the runtime side treats
    // these bytes as allocated and won't hand them out to the user
    // program. Fixed-address regions (e.g. 2 MiB / 3 MiB) collide with
    // dlmalloc's heap and corrupt it on the first allocation.
    this.SYNC_BASE = 0;           // set in init()
    this.SYNC_POOL_BYTES = 64 * 1024;  // 64 KiB for mutexes/semaphores/etc.
    this.nextSyncSlot = 0;

    this.RESULT_BASE = 0;         // set in init()
    this.RESULT_POOL_BYTES = 64 * 1024; // 64 KiB → 4096 16-byte slots
    this.RESULT_SLOT_BYTES = 16;
    this.nextResultSlot = 0;
    this.freeResultSlots = [];
  }

  _allocResultSlot() {
    if (this.freeResultSlots.length > 0) return this.freeResultSlots.pop();
    const addr = this.RESULT_BASE + this.nextResultSlot * this.RESULT_SLOT_BYTES;
    this.nextResultSlot++;
    return addr;
  }

  _releaseResultSlot(addr) {
    // Reset the ready flag and recycle. The main thread holds the only
    // reference at this point (after join() observed done), so no atomic
    // fence is needed — the next allocator will re-initialize it.
    this.freeResultSlots.push(addr);
  }

  // Initialize the worker pool. Call after WASM instantiation.
  async init(wasmModule, memory, instance, workerUrl) {
    this.wasmModule = wasmModule;
    this.memory = memory;
    this.table = instance.exports.__indirect_function_table;

    // Reserve sync + result slot regions through the runtime's own
    // dlmalloc. If we just picked fixed addresses (e.g. 2 MiB / 3 MiB)
    // they'd land inside dlmalloc's heap and the first user allocation
    // would trample the mutex state and result slots.
    const runtimeMalloc = instance.exports.rayzor_malloc;
    if (typeof runtimeMalloc === 'function') {
      this.SYNC_BASE = runtimeMalloc(this.SYNC_POOL_BYTES);
      this.RESULT_BASE = runtimeMalloc(this.RESULT_POOL_BYTES);
      // Zero the result region so freshly-allocated slots start with
      // ready=0. dlmalloc doesn't guarantee zero-initialized memory.
      const i32 = new Int32Array(
        memory.buffer,
        this.RESULT_BASE,
        this.RESULT_POOL_BYTES >> 2,
      );
      for (let i = 0; i < i32.length; i++) Atomics.store(i32, i, 0);
      console.log(
        `[rayzor:threads] sync region @ 0x${this.SYNC_BASE.toString(16)}, ` +
        `result region @ 0x${this.RESULT_BASE.toString(16)}`,
      );
    } else {
      console.warn(
        '[rayzor:threads] rayzor_malloc export missing; falling back to fixed ' +
        'addresses (may race with dlmalloc)',
      );
      this.SYNC_BASE = 2 * 1024 * 1024;
      this.RESULT_BASE = 3 * 1024 * 1024;
    }

    // Check SharedArrayBuffer support
    if (!(memory.buffer instanceof SharedArrayBuffer)) {
      console.warn('[rayzor:threads] Memory is not shared — threads will run on main thread');
      return;
    }

    const poolSize = typeof navigator !== 'undefined'
      ? (navigator.hardwareConcurrency || 4)
      : 4;

    console.log(`[rayzor:threads] Initializing worker pool (${poolSize} workers)`);

    for (let i = 0; i < poolSize; i++) {
      const worker = new Worker(workerUrl, { type: 'module' });
      const tid = this.nextThreadId++;
      worker._threadId = tid;

      worker.onmessage = (e) => this._handleWorkerMessage(e.data, worker);

      // NOTE: we intentionally do NOT forward `this.table` —
      // `WebAssembly.Table` is not a structured-cloneable type, so
      // postMessage would raise DataCloneError. The worker will grab its
      // own `__indirect_function_table` off the instance exports after it
      // re-instantiates the module on its side.
      worker.postMessage({
        type: 'init',
        threadId: tid,
        module: wasmModule,
        memory: memory,
      });

      // Wait for ready
      await new Promise((resolve) => {
        const handler = (e) => {
          if (e.data.type === 'ready') {
            worker.removeEventListener('message', handler);
            resolve();
          }
        };
        worker.addEventListener('message', handler);
      });

      this.workers.push(worker);
      this.idleWorkers.push(worker);
    }

    console.log(`[rayzor:threads] ${poolSize} workers ready`);
  }

  _handleWorkerMessage(msg, worker) {
    if (msg.type === 'done' || msg.type === 'error') {
      const task = this.tasks.get(msg.taskId);
      if (task) {
        this.tasks.delete(msg.taskId);
        const thread = this.threads.get(msg.threadId);
        if (thread) {
          thread.done = true;
          thread.result = msg.result || 0;
        }
        if (msg.type === 'done') task.resolve(msg.result);
        else task.reject(new Error(msg.error));
      }
      // Return worker to idle pool
      this.idleWorkers.push(worker);
    }
    else if (msg.type === 'trace') {
      console.log(`[thread ${msg.threadId}] ${msg.message}`);
    }
  }

  // ========== Thread API ==========

  spawn(fnIdx, envPtr) {
    const threadId = this.nextThreadId++;

    // Real Worker pool path (requires SharedArrayBuffer). The worker writes
    // the closure result into a shared-memory slot and the main thread
    // busy-polls the `ready` flag in `join()` — we cannot use
    // `Atomics.wait` on the main thread per the HTML spec, but we *can*
    // observe atomic writes from other threads via `Atomics.load` inside
    // a synchronous loop.
    if (this._isShared() && this.idleWorkers.length > 0) {
      const worker = this.idleWorkers.pop();
      const slot = this._allocResultSlot();
      // Reset ready flag before dispatch.
      const view = new Int32Array(this.memory.buffer);
      Atomics.store(view, slot >> 2, 0);
      this.threads.set(threadId, {
        done: false,
        result: 0,
        worker,
        slot,
      });
      worker.postMessage({ type: 'run', threadId, fnIdx, envPtr, slot });
      return threadId;
    }

    // Fallback: no Worker pool / no SharedArrayBuffer. Run synchronously
    // on the main thread so join() returns the cached result.
    this.threads.set(threadId, { done: false, result: 0, worker: null });
    try {
      const table = this.table;
      const fn = table ? table.get(fnIdx) : null;
      const result = fn ? fn(envPtr) : 0;
      this.threads.get(threadId).done = true;
      this.threads.get(threadId).result = result;
    } catch (e) {
      console.warn('[rayzor:threads] spawn fallback failed:', e);
      this.threads.get(threadId).done = true;
      this.threads.get(threadId).result = 0;
    }
    return threadId;
  }

  join(threadId) {
    const thread = this.threads.get(threadId);
    if (!thread) return this._boxJoinResult(0);

    if (thread.done) return this._boxJoinResult(thread.result);

    // Worker-dispatched task: spin on the ready flag. Reads are atomic
    // against the worker's `Atomics.store(ready, 1)` — this is the whole
    // point of shared memory + atomics.
    if (thread.slot !== undefined) {
      const view = new Int32Array(this.memory.buffer);
      const readyIdx = thread.slot >> 2;
      // Small idle twiddle so we don't peg the core. Atomics.wait is
      // forbidden on the main thread, so the busy-wait is the only option
      // for a fully synchronous join().
      while (Atomics.load(view, readyIdx) === 0) {
        // Tight loop. A future optimization could bounce the wait into a
        // dedicated "coordinator" worker via Atomics.waitAsync, but that
        // requires an async WASM ABI.
      }
      const dv = new DataView(this.memory.buffer);
      const result = Number(dv.getBigInt64(thread.slot + 8, true));
      thread.done = true;
      thread.result = result;
      // Reclaim the slot and return the worker to the idle pool.
      this._releaseResultSlot(thread.slot);
      thread.slot = undefined;
      if (thread.worker) {
        this.idleWorkers.push(thread.worker);
        thread.worker = null;
      }
      return this._boxJoinResult(result);
    }

    console.warn('[rayzor:threads] join() on main thread with no slot');
    return this._boxJoinResult(0);
  }

  /// Box an i64 join result as a DynamicValue* so the Haxe-side
  /// Thread<T>.join() unbox path finds type_id=3 (Int) and reads value_ptr.
  /// Matches the native `rayzor_thread_join` → `haxe_box_int_ptr` contract.
  _boxJoinResult(value) {
    if (this._boxInt) return this._boxInt(value);
    return value;
  }

  /// Wire up the host harness' boxing helpers. Must be called by the harness
  /// after construction so join() can return boxed DynamicValue* pointers that
  /// match the native runtime contract.
  setBoxHelpers({ boxInt }) {
    this._boxInt = boxInt;
  }

  isFinished(threadId) {
    const thread = this.threads.get(threadId);
    if (!thread) return 0;
    if (thread.done) return 1;
    // Peek at the shared-memory ready flag for worker-dispatched tasks.
    if (thread.slot !== undefined && this._isShared()) {
      const view = new Int32Array(this.memory.buffer);
      return Atomics.load(view, thread.slot >> 2) === 1 ? 1 : 0;
    }
    return 0;
  }

  yieldNow() {
    // No-op on main thread. In Worker, could yield to microtask queue.
  }

  sleep(ms) {
    // Can't Atomics.wait on non-shared memory, and we can't block the main
    // event loop from a host function anyway. Busy-wait for short durations
    // so short Thread.sleep() calls still work as synchronization points.
    if (ms <= 0) return;
    const end = Date.now() + Math.min(ms, 100);
    while (Date.now() < end) { /* spin */ }
  }

  currentId() {
    return 0; // Main thread = 0
  }

  // ========== Mutex API ==========
  // On shared memory: uses Atomics on an i32 cell in WASM memory.
  // On non-shared memory: falls back to a JS-side handle map so single-threaded
  // browsers (no COOP/COEP or Worker pool) still get correct mutex semantics.
  // Returns RAW primitives — the builtin stdlib mapping declares these as
  // returning i32/bool, not boxed DynamicValue*.

  _isShared() {
    return this.memory && this.memory.buffer instanceof SharedArrayBuffer;
  }

  _allocSyncSlot() {
    const slot = this.SYNC_BASE + this.nextSyncSlot * 4;
    this.nextSyncSlot++;
    return slot;
  }

  mutexInit() {
    if (this._isShared()) {
      const slot = this._allocSyncSlot();
      const view = new Int32Array(this.memory.buffer);
      Atomics.store(view, slot >> 2, 0);
      return slot;
    }
    // Non-shared: allocate a JS handle.
    if (!this._mutexMap) { this._mutexMap = new Map(); this._nextMutex = 1; }
    const id = this._nextMutex++;
    this._mutexMap.set(id, { locked: false });
    return id;
  }

  mutexLock(mutexId) {
    if (this._isShared()) {
      const view = new Int32Array(this.memory.buffer);
      const idx = mutexId >> 2;
      while (true) {
        if (Atomics.compareExchange(view, idx, 0, 1) === 0) return mutexId;
        Atomics.wait(view, idx, 1, 100);
      }
    }
    const m = this._mutexMap && this._mutexMap.get(mutexId);
    if (m) m.locked = true;
    return mutexId;
  }

  mutexTryLock(mutexId) {
    if (this._isShared()) {
      const view = new Int32Array(this.memory.buffer);
      return Atomics.compareExchange(view, mutexId >> 2, 0, 1) === 0 ? 1 : 0;
    }
    const m = this._mutexMap && this._mutexMap.get(mutexId);
    if (!m) return 0;
    if (m.locked) return 0;
    m.locked = true;
    return 1;
  }

  mutexIsLocked(mutexId) {
    if (this._isShared()) {
      const view = new Int32Array(this.memory.buffer);
      return Atomics.load(view, mutexId >> 2) !== 0 ? 1 : 0;
    }
    const m = this._mutexMap && this._mutexMap.get(mutexId);
    return m && m.locked ? 1 : 0;
  }

  mutexGuardGet(mutexId) {
    // Guard just aliases the mutex id — the inner value lives in WASM memory
    // and is accessed by the compiled Haxe code directly.
    return mutexId;
  }

  mutexUnlock(mutexId) {
    if (this._isShared()) {
      const view = new Int32Array(this.memory.buffer);
      const idx = mutexId >> 2;
      Atomics.store(view, idx, 0);
      Atomics.notify(view, idx, 1);
      return;
    }
    const m = this._mutexMap && this._mutexMap.get(mutexId);
    if (m) m.locked = false;
  }

  // ========== Semaphore API ==========

  semaphoreInit(count) {
    const slot = this._allocSyncSlot();
    const view = new Int32Array(this.memory.buffer);
    Atomics.store(view, slot >> 2, count);
    return slot;
  }

  semaphoreAcquire(semId) {
    const view = new Int32Array(this.memory.buffer);
    const idx = semId >> 2;
    while (true) {
      const current = Atomics.load(view, idx);
      if (current > 0 && Atomics.compareExchange(view, idx, current, current - 1) === current) return;
      Atomics.wait(view, idx, 0, 100);
    }
  }

  semaphoreTryAcquire(semId) {
    const view = new Int32Array(this.memory.buffer);
    const idx = semId >> 2;
    const current = Atomics.load(view, idx);
    if (current > 0 && Atomics.compareExchange(view, idx, current, current - 1) === current) return 1;
    return 0;
  }

  // ========== Channel API ==========
  // Simple bounded channel using shared memory ring buffer.

  channelInit() {
    // Allocate: [head:i32, tail:i32, closed:i32, cap:i32, lock:i32, data:i64[64]]
    const CAP = 64;
    const headerSize = 5 * 4; // 5 i32s
    const dataSize = CAP * 8; // 64 i64s
    const totalSlots = Math.ceil((headerSize + dataSize) / 4);
    const baseSlot = this.SYNC_BASE + this.nextSyncSlot * 4;
    this.nextSyncSlot += totalSlots;

    const view = new Int32Array(this.memory.buffer);
    const base = baseSlot >> 2;
    Atomics.store(view, base + 0, 0);   // head
    Atomics.store(view, base + 1, 0);   // tail
    Atomics.store(view, base + 2, 0);   // closed
    Atomics.store(view, base + 3, CAP); // cap
    Atomics.store(view, base + 4, 0);   // lock
    return baseSlot;
  }

  channelSend(chanId, value) {
    const view = new Int32Array(this.memory.buffer);
    const dv = new DataView(this.memory.buffer);
    const base = chanId >> 2;
    const cap = Atomics.load(view, base + 3);
    const headerBytes = 5 * 4;

    // Spin until space available
    while (true) {
      const head = Atomics.load(view, base + 0);
      const tail = Atomics.load(view, base + 1);
      const len = (tail - head + cap) % cap;
      if (len < cap - 1) {
        // Write value at tail position
        const offset = chanId + headerBytes + (tail % cap) * 8;
        dv.setBigUint64(offset, BigInt(value), true);
        Atomics.store(view, base + 1, (tail + 1) % cap);
        Atomics.notify(view, base + 0, 1); // notify receivers
        return;
      }
      Atomics.wait(view, base + 1, tail, 10); // wait for consumer
    }
  }

  channelTrySend(chanId, value) {
    const view = new Int32Array(this.memory.buffer);
    const dv = new DataView(this.memory.buffer);
    const base = chanId >> 2;
    const cap = Atomics.load(view, base + 3);
    const head = Atomics.load(view, base + 0);
    const tail = Atomics.load(view, base + 1);
    const len = (tail - head + cap) % cap;
    if (len >= cap - 1) return 0;
    const headerBytes = 5 * 4;
    const offset = chanId + headerBytes + (tail % cap) * 8;
    dv.setBigUint64(offset, BigInt(value), true);
    Atomics.store(view, base + 1, (tail + 1) % cap);
    Atomics.notify(view, base + 0, 1);
    return 1;
  }

  channelReceive(chanId) {
    const view = new Int32Array(this.memory.buffer);
    const dv = new DataView(this.memory.buffer);
    const base = chanId >> 2;
    const cap = Atomics.load(view, base + 3);
    const headerBytes = 5 * 4;

    while (true) {
      const head = Atomics.load(view, base + 0);
      const tail = Atomics.load(view, base + 1);
      if (head !== tail) {
        const offset = chanId + headerBytes + (head % cap) * 8;
        const value = Number(dv.getBigUint64(offset, true));
        Atomics.store(view, base + 0, (head + 1) % cap);
        Atomics.notify(view, base + 1, 1); // notify senders
        return value;
      }
      if (Atomics.load(view, base + 2) !== 0) return 0; // closed
      Atomics.wait(view, base + 0, head, 10);
    }
  }

  channelTryReceive(chanId) {
    const view = new Int32Array(this.memory.buffer);
    const dv = new DataView(this.memory.buffer);
    const base = chanId >> 2;
    const cap = Atomics.load(view, base + 3);
    const head = Atomics.load(view, base + 0);
    const tail = Atomics.load(view, base + 1);
    if (head === tail) return 0;
    const headerBytes = 5 * 4;
    const offset = chanId + headerBytes + (head % cap) * 8;
    const value = Number(dv.getBigUint64(offset, true));
    Atomics.store(view, base + 0, (head + 1) % cap);
    Atomics.notify(view, base + 1, 1);
    return value;
  }

  channelClose(chanId) {
    const view = new Int32Array(this.memory.buffer);
    Atomics.store(view, (chanId >> 2) + 2, 1);
    Atomics.notify(view, chanId >> 2, 0x7fffffff); // wake all
  }

  channelIsClosed(chanId) {
    const view = new Int32Array(this.memory.buffer);
    return Atomics.load(view, (chanId >> 2) + 2) !== 0 ? 1 : 0;
  }

  channelLen(chanId) {
    const view = new Int32Array(this.memory.buffer);
    const base = chanId >> 2;
    const cap = Atomics.load(view, base + 3);
    const head = Atomics.load(view, base + 0);
    const tail = Atomics.load(view, base + 1);
    return (tail - head + cap) % cap;
  }

  channelCapacity(chanId) {
    const view = new Int32Array(this.memory.buffer);
    return Atomics.load(view, (chanId >> 2) + 3);
  }

  channelIsEmpty(chanId) {
    return this.channelLen(chanId) === 0 ? 1 : 0;
  }

  channelIsFull(chanId) {
    const len = this.channelLen(chanId);
    const cap = this.channelCapacity(chanId);
    return len >= cap - 1 ? 1 : 0;
  }

  // ========== Future API ==========

  futureCreate(fnIdx, envPtr) {
    const threadId = this.spawn(fnIdx, envPtr);
    return threadId; // future handle = thread handle
  }

  futureAwait(futureId) {
    const thread = this.threads.get(futureId);
    if (!thread) return 0;
    if (thread.done) return thread.result;
    // Can't truly block main thread — return 0 and warn
    console.warn('[rayzor:threads] Future.await() on main thread — use .then() instead');
    return 0;
  }

  futureThen(futureId, cbFnIdx, cbEnvPtr) {
    const thread = this.threads.get(futureId);
    if (!thread) return;
    if (thread.done) {
      // Already done — call callback immediately
      const fn = this.table?.get(cbFnIdx);
      if (fn) fn(cbEnvPtr);
      return;
    }
    // Wait for completion then call callback
    if (thread.promise) {
      thread.promise.then(() => {
        const fn = this.table?.get(cbFnIdx);
        if (fn) fn(cbEnvPtr);
      });
    }
  }

  futurePoll(futureId) {
    return this.threads.get(futureId)?.done ? 1 : 0;
  }

  futureIsReady(futureId) {
    return this.futurePoll(futureId);
  }

  futureAll(_arrayPtr) { return 0; } // TODO
  futureAwaitTimeout(_futureId, _ms) { return 0; } // TODO
  futureRace(_arrayPtr) { return 0; } // TODO
  futureCancel(_futureId) {} // TODO
  futureIsCancelled(_futureId) { return 0; } // TODO

  // ========== Build import bindings for WASM ==========

  buildImports() {
    const rt = this;
    return {
      rayzor_thread_spawn: (fnIdx, envPtr) => rt.spawn(fnIdx, envPtr),
      rayzor_thread_join: (tid) => rt.join(tid),
      rayzor_thread_is_finished: (tid) => rt.isFinished(tid),
      rayzor_thread_yield_now: () => rt.yieldNow(),
      rayzor_thread_sleep: (ms) => rt.sleep(ms),
      rayzor_thread_current_id: () => rt.currentId(),
      rayzor_semaphore_init: (n) => rt.semaphoreInit(n),
      rayzor_semaphore_acquire: (id) => rt.semaphoreAcquire(id),
      rayzor_semaphore_try_acquire: (id) => rt.semaphoreTryAcquire(id),
      sys_semaphore_try_acquire_nowait: (id) => rt.semaphoreTryAcquire(id),
      rayzor_channel_init: () => rt.channelInit(),
      rayzor_channel_send: (id, v) => rt.channelSend(id, v),
      rayzor_channel_try_send: (id, v) => rt.channelTrySend(id, v),
      rayzor_channel_receive: (id) => rt.channelReceive(id),
      rayzor_channel_try_receive: (id) => rt.channelTryReceive(id),
      rayzor_channel_close: (id) => rt.channelClose(id),
      rayzor_channel_is_closed: (id) => rt.channelIsClosed(id),
      rayzor_channel_len: (id) => rt.channelLen(id),
      rayzor_channel_capacity: (id) => rt.channelCapacity(id),
      rayzor_channel_is_empty: (id) => rt.channelIsEmpty(id),
      rayzor_channel_is_full: (id) => rt.channelIsFull(id),
      rayzor_mutex_init: () => rt.mutexInit(),
      rayzor_mutex_lock: (id) => rt.mutexLock(id),
      rayzor_mutex_try_lock: (id) => rt.mutexTryLock(id),
      rayzor_mutex_is_locked: (id) => rt.mutexIsLocked(id),
      rayzor_mutex_guard_get: (id) => rt.mutexGuardGet(id),
      rayzor_mutex_unlock: (id) => rt.mutexUnlock(id),
      rayzor_future_create: (fn, env) => rt.futureCreate(fn, env),
      rayzor_future_await: (id) => rt.futureAwait(id),
      rayzor_future_then: (id, fn, env) => rt.futureThen(id, fn, env),
      rayzor_future_poll: (id) => rt.futurePoll(id),
      rayzor_future_is_ready: (id) => rt.futureIsReady(id),
      rayzor_future_all: (arr) => rt.futureAll(arr),
      rayzor_future_await_timeout: (id, ms) => rt.futureAwaitTimeout(id, ms),
      rayzor_future_race: (arr) => rt.futureRace(arr),
      rayzor_future_cancel: (id) => rt.futureCancel(id),
      rayzor_future_is_cancelled: (id) => rt.futureIsCancelled(id),
      rayzor_arc_init: (v) => v,     // Arc on WASM = identity (single heap)
      rayzor_arc_clone: (v) => v,
      rayzor_arc_get: (v) => v,
      rayzor_arc_strong_count: () => 1,
      rayzor_arc_try_unwrap: (v) => v,
      rayzor_arc_as_ptr: (v) => v,
      rayzor_box_init: (v) => v,
      rayzor_box_unbox: (v) => v,
      rayzor_box_raw: (v) => v,
      rayzor_box_free: () => {},
    };
  }
}
