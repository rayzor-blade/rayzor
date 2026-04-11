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

    // Sync primitive allocator (in shared memory)
    this.nextSyncSlot = 0;
    this.SYNC_BASE = 1024 * 1024 * 2; // 2MB offset for sync primitives
  }

  // Initialize the worker pool. Call after WASM instantiation.
  async init(wasmModule, memory, instance, workerUrl) {
    this.wasmModule = wasmModule;
    this.memory = memory;
    this.table = instance.exports.__indirect_function_table;

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

      worker.postMessage({
        type: 'init',
        threadId: tid,
        module: wasmModule,
        memory: memory,
        table: this.table,
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
    const taskId = this.nextTaskId++;

    this.threads.set(threadId, { done: false, result: 0, worker: null });

    if (this.idleWorkers.length > 0) {
      const worker = this.idleWorkers.pop();
      this.threads.get(threadId).worker = worker;

      const promise = new Promise((resolve, reject) => {
        this.tasks.set(taskId, { resolve, reject, threadId });
      });

      worker.postMessage({ type: 'run', taskId, fnIdx, envPtr });
      this.threads.get(threadId).promise = promise;
    } else {
      // No idle workers — run synchronously on main thread (fallback).
      // Resolves the closure body via __indirect_function_table.get(fnIdx)
      // and invokes it with env_ptr, caching the result for join().
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
    }

    return threadId;
  }

  join(threadId) {
    const thread = this.threads.get(threadId);
    if (!thread) return this._boxJoinResult(0);
    // Main thread fallback: the closure already ran synchronously during
    // spawn(), so `thread.done` is true and we return the cached result.
    if (thread.done) return this._boxJoinResult(thread.result);
    // True blocking on main thread requires Atomics.wait on shared memory.
    // With a Worker pool we'd spin here; without it we log and return 0.
    console.warn('[rayzor:threads] join() on main thread with pending Worker task');
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
    return this.threads.get(threadId)?.done ? 1 : 0;
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
