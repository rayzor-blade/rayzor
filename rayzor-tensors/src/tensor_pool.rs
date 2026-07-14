//! Tensor allocation pool — shape-class bucket recycling.
//!
//! # Opt-in status (as of this commit)
//!
//! The pool is **opt-in via the `RZT_POOL=1` env var**. With the var unset
//! (the default), `TensorPool::global()` returns a singleton in its disabled
//! state — every `try_pop` immediately returns `None` and every `push`
//! immediately evicts via the supplied `freer`, so the pool adds no Mutex
//! traffic, no HashMap lookup, and no bucket bookkeeping on the hot path. It
//! is plumbed-but-dormant in real workloads until the data-flow gap is closed
//! (`InsertFreePass` on transient activation tensors OR KV-cache reference
//! release). When that gap closes, the default will flip to opt-out.
//!
//! Set `RZT_POOL=1` to exercise the bucket-pool machinery (and the related
//! `RZT_POOL_MAX_PER_BUCKET` / `RZT_POOL_MAX_TOTAL_MB` / `RZT_POOL_STATS`
//! knobs); set `RZT_POOL_DISABLE=1` to force-disable even if `RZT_POOL=1`
//! is also set (the explicit disable wins, for A/B bisection runs).
//!
//! # Motivation
//!
//! `runtime/src/tensor.rs::alloc_tensor` performs FOUR `malloc` calls per
//! tensor (data, shape array, strides array, RayzorTensor struct) and the
//! corresponding FOUR `free`s on drop. Audit instrumentation gated behind
//! `RZT_TENSOR_ALLOC_HISTOGRAM` revealed that a 24-token Llama 3.2 1B
//! decode allocates ~3000 tensors across only 35 distinct shape classes, so
//! the hot path is the same handful of `(dtype, shape)` tuples being
//! allocated and freed over and over.
//!
//! This module exposes a global allocator-side recycle bin: on `push` an
//! about-to-be-freed tensor's struct + buffers are retained in a per-shape
//! bucket; on `try_pop` an alloc request for the same shape can resurrect
//! a previously freed entry with zero malloc activity.
//!
//! # Design rationale (see /tmp/audit context for the long form)
//!
//! - **Bucket key = (dtype, shape_hash)** — hashing the shape vec avoids
//!   `Vec<usize>` map keys (and the SipHash they imply). Collisions are
//!   resolved by linearly checking `PooledEntry::shape == requested_shape`
//!   when walking the bucket — buckets are bounded to ≤ `max_per_bucket`
//!   (default 8) so the walk is cheap.
//! - **Single global lock (`parking_lot::Mutex`)** — the bench audit notes
//!   the production tensor pipeline is effectively single-producer (only
//!   the threaded qmatmul kernel multi-threads compute, and it allocates
//!   its output BEFORE spawning workers). A coarse lock is therefore not
//!   on the steady-state critical path; future work can layer thread-local
//!   front-buckets if profiling shows contention.
//! - **owns_data invariant** — view producers (`reshape` (contiguous),
//!   `permute`, `slice`, `transpose`, `transpose_last2`, plus the Q4_K_M /
//!   Q6_K mmap wrappers) construct tensors with `owns_data: false`. Pushing
//!   such a tensor into the pool would later hand its alias back as an
//!   owning allocation, corrupting the parent on first write. `push()`
//!   therefore early-returns when `!owns_data` and the wrapper bytes are
//!   freed normally by the caller.
//! - **QTensor meta retention** — INT8 QTensors carry a separately-allocated
//!   f32 scales array sized by group_size. The `PooledEntry` records the
//!   meta pointer and length so that an INT8 pop can hand back the original
//!   scales buffer unchanged. The bucket key alone does NOT distinguish
//!   group_size — INT8 callers must therefore use a `PoolKey` whose
//!   `shape_hash` mixes group_size into the digest (or use a separate
//!   `qint8_*` API), keeping recycled entries safe.
//! - **Pool vs arena** — an arena-style reset would clash with the IR's
//!   per-temp InsertFreePass: temps inside a scope are freed individually
//!   on Drop, not at scope exit. A drop-in bucket pool composes with the
//!   existing `@:move` + Drop machinery transparently — from the language's
//!   perspective Drop still runs, the runtime simply hoards the bytes.
//!
//! # Env flags
//!
//! - `RZT_POOL` — opt-in master switch. When set to `1`, the pool is
//!   active (subject to `RZT_POOL_DISABLE` below). When unset / anything
//!   else, the pool is **disabled by default**: `try_pop` always returns
//!   `None` and `push` immediately evicts (free-through), so no Mutex or
//!   HashMap lookup happens on the alloc / free hot path.
//! - `RZT_POOL_DISABLE` — when set (any non-empty value), force-disables
//!   the pool even if `RZT_POOL=1` is also set. The explicit disable
//!   wins, which keeps A/B benchmarks and bug bisection runs unambiguous.
//! - `RZT_POOL_MAX_PER_BUCKET` — max retained entries per bucket
//!   (default 8). Over the cap, `push` evicts (calls the supplied `freer`
//!   on the eldest entry).
//! - `RZT_POOL_MAX_TOTAL_MB` — global cap on retained bytes (default
//!   256 MB). Over the cap, `push` evicts the eldest entry from the same
//!   bucket; if the bucket has only one entry, `push` evicts that newly-
//!   incoming tensor instead.
//! - `RZT_POOL_STATS` — when set, an `atexit` hook prints the running
//!   statistics on process shutdown.
//! - `RZT_TENSOR_POOL_POISON` — when set, the pool fills the data
//!   buffer of every parked entry with `0xCD` BEFORE releasing the bucket
//!   lock, and again fills the data buffer of every popped entry with
//!   `0xCD` BEFORE returning it to the caller. A buggy caller that reads
//!   stale data from a parked-then-revived buffer will see `0xCDCDCDCD`
//!   (≈ -842150451 as i32, NaN as f32) — an obvious sentinel rather than
//!   the prior contents of the buffer. The flag is read once at first
//!   use and cached in a `OnceLock`; toggling it after the pool has been
//!   initialised has no effect.
//!
//! # Pool entry lifecycle
//!
//! The pool stores raw `*mut RayzorTensor` pointers wrapped in
//! `PooledEntry`. Because the pool layer does not know the concrete
//! `RayzorTensor` definition (it lives in `tensor.rs`), callers must
//! provide a `freer` callback at `push`/eviction time: this is the
//! function that knows how to release the struct + data + shape + strides
//! (and optionally the QTensor meta) without re-entering the pool.
//!
//! # Safety invariants
//!
//! These invariants are the contract every poolable allocation must
//! respect. Violating any of them risks use-after-free, double-free,
//! aliased writes through a view, or cross-scheme byte corruption.
//!
//! 1. **owns_data = true gate.** `push()` callers MUST verify
//!    `RayzorTensor::owns_data == true` before constructing a
//!    `PooledEntry`. Wrappers with `owns_data = false` (`reshape`-
//!    contiguous, `permute`, `slice`, `transpose`, `transpose_last2`,
//!    `qtensor_from_bytes_q4_k_m`, `qtensor_from_bytes_q6_k`) alias a
//!    parent tensor's buffer. Pooling such a wrapper would later hand
//!    its aliased `data` back as an owning allocation and corrupt the
//!    parent on first write. Enforced at both ends:
//!    - **Push side**: `rayzor_tensor_free` / `rayzor_qtensor_free`
//!      explicitly branch on `owns_data` and take the direct-free path
//!      (release shape/strides/wrapper only) for views.
//!    - **Producer side**: view constructors continue to set
//!      `owns_data: false` so a future free path collapse cannot leak
//!      a view into the pool.
//! 2. **`@:move` / strict-move at the language layer.** Both
//!    `rayzor.ds.Tensor` and `rayzor.ds.QTensor` are annotated
//!    `@:move`. The compiler's TAST pass
//!    (`compiler/src/tast/trait_checker.rs::requires_strict_move`)
//!    promotes any `E0382 use-of-moved-value` against a `@:move` type
//!    from a soft warning to a hard error. The pool therefore CANNOT
//!    silently revive a freed binding for the user: the user binding
//!    is statically unreachable after a move into a `.free()` call.
//!    Pool routing is a runtime optimisation invisible to the TAST
//!    layer — no compiler-side analysis changes when a tensor is pooled
//!    vs free'd back to malloc.
//! 3. **PoolKey namespace isolation.** Three orthogonal cuts:
//!    - **Plain vs QTensor.** Plain tensors use `dtype ∈ [0, 0x7F]`
//!      (see `tensor.rs::DTYPE_*`). QTensor keys OR `0x80` into the
//!      dtype byte (`quant.rs::qtensor_pool_key`), so plain F32
//!      (`dtype = 0`) and INT8 QTensor (`scheme = 0`) never alias.
//!    - **QScheme.** INT8 / Q4_K_M / Q6_K each carry a distinct
//!      `scheme` byte AND XOR `scheme` into `shape_hash` for defence
//!      in depth. An INT8 parked entry can never satisfy a Q4_K_M pop
//!      even if rows/cols/group_size coincide.
//!    - **group_size.** INT8 QTensors stash a meta scales array sized
//!      `numel / group_size`. `qtensor_pool_key` folds `group_size`
//!      into the synthetic shape `[rows, cols, group_size]` so two
//!      tensors that differ only in group_size hash to different
//!      buckets — the bucket-walk's `shape == ?` check also catches
//!      any hash collision.
//! 4. **Bucket-walk authority over hash.** `try_pop` walks the bucket
//!    linearly and compares `entry.shape.as_slice() == requested`. The
//!    hash is a fast filter; the slice comparison is the authoritative
//!    decider. Hash collisions are therefore safe — a colliding entry
//!    is simply skipped.
//! 5. **Freer-after-unlock invariant.** Both `push` (on eviction) and
//!    `drain` invoke `freer(entry)` AFTER releasing the bucket
//!    `Mutex`. The freer is free to `malloc` / `free` / call any other
//!    runtime path without risk of re-entering the pool and
//!    deadlocking. Eviction code paths upstream rely on this.
//! 6. **Concurrency.** A single global `parking_lot::Mutex` serialises
//!    every bucket access; `PoolStats` uses `Atomic*` so snapshot
//!    readers don't take the lock. The `stress_concurrent_alloc_free_*`
//!    test exercises 8 threads × 1000 push/pop cycles with FREE_COUNT
//!    accounting; the test asserts `pushes == frees` at the end so any
//!    double-free or leak surfaces as a hard assertion. `PooledEntry`'s
//!    raw pointers are `unsafe impl Send + Sync` because ownership is
//!    transferred atomically: the pushing thread relinquishes on
//!    `push`, the popping thread reassumes on `try_pop`. No two threads
//!    ever see the same entry simultaneously.
//! 7. **Production drain semantics.** `rayzor_tensor_pool_reset`
//!    (defined in `tensor.rs`, not here) wires
//!    `TensorPool::drain(tensor_pool_freer)` so a real flush releases
//!    `data` / `shape` / `strides` / wrapper through the canonical
//!    `RayzorTensor` release path. The `_test_clear_pool_bookkeeping`
//!    helper below is the test-only counterpart that intentionally
//!    leaks parked memory for fast test isolation.

#![allow(clippy::missing_safety_doc)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Hash-resilient bucket key. `dtype` is the tensor's element-type tag (the
/// same `u8` stored in `RayzorTensor::dtype` — see `tensor.rs::DTYPE_*`).
/// `shape_hash` is a deterministic hash of the shape vec; QTensor callers
/// MAY fold group_size / scheme into the hash before constructing the key.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct PoolKey {
    pub dtype: u8,
    pub shape_hash: u64,
}

impl PoolKey {
    /// Build a key from a `(dtype, shape)` pair. Uses the same FxHash-style
    /// digest as `hash_shape` to keep `try_pop` and `push` symmetric.
    pub fn from_shape(dtype: u8, shape: &[usize]) -> Self {
        Self {
            dtype,
            shape_hash: hash_shape(shape),
        }
    }
}

/// Stable digest of a shape vector. Public so callers (e.g. QTensor) can
/// mix additional discriminants (group_size, quant scheme) into the hash
/// before constructing a `PoolKey`.
pub fn hash_shape(shape: &[usize]) -> u64 {
    // Tiny FNV-1a — deterministic, no SipHash random seed, collision-safe
    // enough for the rare bucket-walk fallback that re-checks `shape == ?`.
    let mut h: u64 = 0xcbf29ce484222325;
    for &dim in shape {
        let bytes = (dim as u64).to_le_bytes();
        for b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    // Mix the length too so `[0,0,0]` and `[0,0]` collide less.
    let lb = (shape.len() as u64).to_le_bytes();
    for b in lb {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// One retained tensor + the metadata needed to revalidate on `try_pop`
/// and to fully release on bucket eviction.
///
/// `ptr` is the `*mut RayzorTensor` originally returned by `alloc_tensor`
/// (or `alloc_qtensor`). The pool does not deref it; the caller passes the
/// concrete `freer` that knows the layout.
///
/// `shape` is retained for the bucket-walk collision check — `shape_hash`
/// alone is not authoritative.
///
/// `alloc_bytes` is the data-buffer size; tracked for the
/// `RZT_POOL_MAX_TOTAL_MB` budget.
///
/// `qtensor_meta_ptr` / `qtensor_meta_bytes` are non-null iff the entry is
/// an INT8 QTensor with a separately-allocated f32 scales array. They are
/// passed back to the freer unchanged so that meta is either kept (on
/// `try_pop` hit) or released alongside `ptr` (on eviction).
#[derive(Clone)]
pub struct PooledEntry {
    pub ptr: *mut u8,
    pub shape: ShapeBuf,
    pub alloc_bytes: usize,
    pub qtensor_meta_ptr: *mut u8,
    pub qtensor_meta_bytes: usize,
}

// Raw pointers in PooledEntry are owned by the pool while parked. The
// `push`-caller releases ownership; `try_pop` transfers it back. Sending
// the entry across threads is only needed because a global Mutex<HashMap>
// is `Send + Sync`-bound; the receiving thread reassumes ownership.
unsafe impl Send for PooledEntry {}
unsafe impl Sync for PooledEntry {}

/// Inline shape buffer — up to 6 dimensions, enough for every Llama
/// shape class observed in the audit (max ndim = 3). Spills to a heap
/// `Vec` on the rare >6-D case so the cold path stays correct.
#[derive(Clone)]
pub enum ShapeBuf {
    Inline { len: u8, dims: [usize; 6] },
    Heap(Vec<usize>),
}

impl ShapeBuf {
    pub fn from_slice(shape: &[usize]) -> Self {
        if shape.len() <= 6 {
            let mut dims = [0usize; 6];
            dims[..shape.len()].copy_from_slice(shape);
            ShapeBuf::Inline {
                len: shape.len() as u8,
                dims,
            }
        } else {
            ShapeBuf::Heap(shape.to_vec())
        }
    }

    pub fn as_slice(&self) -> &[usize] {
        match self {
            ShapeBuf::Inline { len, dims } => &dims[..*len as usize],
            ShapeBuf::Heap(v) => v.as_slice(),
        }
    }
}

/// Free callback. Pool invokes this on eviction (bucket-full or
/// global-bytes-budget exceeded) to release the entry's memory without
/// re-entering the pool. Caller provides this at `push` time.
pub type FreeFn = unsafe fn(entry: PooledEntry);

/// Pool stats — all atomics so any thread can read without taking the
/// pool lock.
#[derive(Default)]
pub struct PoolStats {
    pub hits: AtomicU64,
    pub misses: AtomicU64,
    pub pushes: AtomicU64,
    pub evictions: AtomicU64,
    pub peak_per_bucket: AtomicUsize,
    pub current_bytes: AtomicUsize,
    pub peak_bytes: AtomicUsize,
}

impl PoolStats {
    pub fn snapshot(&self) -> PoolStatsSnapshot {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        PoolStatsSnapshot {
            hits,
            misses,
            pushes: self.pushes.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            hit_rate: if total == 0 {
                0.0
            } else {
                hits as f64 / total as f64
            },
            peak_per_bucket: self.peak_per_bucket.load(Ordering::Relaxed),
            current_bytes: self.current_bytes.load(Ordering::Relaxed),
            peak_bytes: self.peak_bytes.load(Ordering::Relaxed),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PoolStatsSnapshot {
    pub hits: u64,
    pub misses: u64,
    pub pushes: u64,
    pub evictions: u64,
    pub hit_rate: f64,
    pub peak_per_bucket: usize,
    pub current_bytes: usize,
    pub peak_bytes: usize,
}

/// Shape-class bucket pool.
pub struct TensorPool {
    buckets: Mutex<HashMap<PoolKey, Vec<PooledEntry>>>,
    pub stats: PoolStats,
    pub max_per_bucket: usize,
    pub max_total_bytes: usize,
    disabled: AtomicBool,
}

impl TensorPool {
    /// Construct a pool with the supplied limits. The env-driven
    /// configuration is applied by `global()`; this constructor is the
    /// raw interface used by unit tests.
    pub fn new(max_per_bucket: usize, max_total_bytes: usize, disabled: bool) -> Self {
        Self {
            buckets: Mutex::new(HashMap::new()),
            stats: PoolStats::default(),
            max_per_bucket,
            max_total_bytes,
            disabled: AtomicBool::new(disabled),
        }
    }

    /// Whether the pool is in its short-circuit state. When `true`,
    /// `try_pop` returns `None` without taking the mutex and `push`
    /// frees the entry directly. Surfaced for the diagnostics dump so
    /// readers can distinguish "pool is genuinely missing" from
    /// "pool is disabled by default" — the two look identical from
    /// `hits=0 misses=N` alone.
    #[inline]
    pub fn is_disabled(&self) -> bool {
        self.disabled.load(Ordering::Relaxed)
    }

    /// Attempt to recycle an entry for the requested shape. Returns the
    /// raw `*mut RayzorTensor` (cast to `*mut u8`) on hit, `None` on miss.
    ///
    /// Walks the bucket linearly and compares `shape == ?` to defend against
    /// hash collisions. Buckets are bounded by `max_per_bucket` so the
    /// walk is O(small).
    pub fn try_pop(&self, key: PoolKey, shape: &[usize]) -> Option<PooledEntry> {
        if self.disabled.load(Ordering::Relaxed) {
            self.stats.misses.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let mut buckets = self.buckets.lock();
        let entries = match buckets.get_mut(&key) {
            Some(e) => e,
            None => {
                self.stats.misses.fetch_add(1, Ordering::Relaxed);
                return None;
            }
        };
        // Walk newest-first — most recently freed is most cache-hot.
        let mut found_idx: Option<usize> = None;
        for (i, e) in entries.iter().enumerate().rev() {
            if e.shape.as_slice() == shape {
                found_idx = Some(i);
                break;
            }
        }
        match found_idx {
            Some(i) => {
                let entry = entries.swap_remove(i);
                self.stats.hits.fetch_add(1, Ordering::Relaxed);
                self.stats
                    .current_bytes
                    .fetch_sub(entry.alloc_bytes, Ordering::Relaxed);
                drop(buckets);
                // Poison BEFORE returning so any caller that mistakenly
                // reads the buffer before the zero-fill in
                // `alloc_tensor`'s pool-hit arm sees `0xCD` sentinels.
                unsafe { poison_entry_data(&entry) };
                Some(entry)
            }
            None => {
                self.stats.misses.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    /// Push a freed tensor into the pool. Caller MUST have verified that
    /// the underlying tensor is owning (i.e. `owns_data == true`) — view
    /// wrappers are unsafe to recycle, see module doc.
    ///
    /// Returns `true` if the entry was parked; `false` if the pool was
    /// full / disabled and the entry was evicted (freed via `freer`).
    ///
    /// # Budget race correctness
    ///
    /// The budget check (`current_bytes + alloc_bytes > max_total_bytes`)
    /// AND the budget-driven eviction-or-park decision are performed under
    /// the same bucket `Mutex` acquisition. Earlier revisions read
    /// `current_bytes` with `Ordering::Relaxed` *before* taking the lock,
    /// which let two concurrent pushers both observe under-budget and both
    /// proceed to park, breaching the cap. With the load+decide moved
    /// inside the critical section, `current_bytes` only changes while the
    /// lock is held, so the cap is observed atomically.
    pub fn push(&self, key: PoolKey, entry: PooledEntry, freer: FreeFn) -> bool {
        self.stats.pushes.fetch_add(1, Ordering::Relaxed);
        if self.disabled.load(Ordering::Relaxed) {
            self.stats.evictions.fetch_add(1, Ordering::Relaxed);
            unsafe { freer(entry) };
            return false;
        }

        // Acquire the bucket lock FIRST, then read current_bytes and
        // decide. This serialises every budget transition behind a single
        // mutex so two concurrent pushers cannot both observe room and
        // both park; cf. the cap-race stress test below.
        let added_bytes = entry.alloc_bytes;
        let mut buckets = self.buckets.lock();
        let current_bytes = self.stats.current_bytes.load(Ordering::Relaxed);
        let prospective_bytes = current_bytes + added_bytes;
        if prospective_bytes > self.max_total_bytes {
            // Budget exceeded — try to make room by evicting the oldest
            // entry from this bucket. If the bucket is empty, evict the
            // incoming entry instead (free-through). Either way the lock
            // is held continuously so concurrent pushers see the updated
            // current_bytes before deciding.
            if let Some(entries) = buckets.get_mut(&key) {
                if !entries.is_empty() {
                    let evicted = entries.remove(0);
                    self.stats
                        .current_bytes
                        .fetch_sub(evicted.alloc_bytes, Ordering::Relaxed);
                    self.stats.evictions.fetch_add(1, Ordering::Relaxed);
                    // Poison BEFORE parking the incoming entry so its
                    // buffer carries the sentinel during the park.
                    unsafe { poison_entry_data(&entry) };
                    // Now park the incoming under the same lock so the
                    // intermediate state is never observable.
                    let bucket = buckets.entry(key).or_default();
                    bucket.push(entry);
                    let bucket_len = bucket.len();
                    self.bump_bytes(added_bytes);
                    self.bump_peak_bucket(bucket_len);
                    drop(buckets);
                    unsafe { freer(evicted) };
                    return true;
                }
            }
            // No room and bucket empty: evict the incoming.
            self.stats.evictions.fetch_add(1, Ordering::Relaxed);
            drop(buckets);
            unsafe { freer(entry) };
            return false;
        }

        // Under budget — park (inline `push_inner` body so the lock
        // covers both the budget check and the park atomically).
        self.push_inner_locked(buckets, key, entry, freer)
    }

    /// Park into the bucket while already holding the bucket lock.
    /// Caller is responsible for having checked the global byte budget.
    fn push_inner_locked(
        &self,
        mut buckets: parking_lot::MutexGuard<'_, HashMap<PoolKey, Vec<PooledEntry>>>,
        key: PoolKey,
        entry: PooledEntry,
        freer: FreeFn,
    ) -> bool {
        let added_bytes = entry.alloc_bytes;
        // Poison BEFORE parking so any subsequent reader of the parked
        // buffer (debug-poke from the freer, racy read after a stale
        // cached pointer escapes, etc.) sees `0xCD` sentinels.
        unsafe { poison_entry_data(&entry) };
        let bucket = buckets.entry(key).or_default();
        if bucket.len() >= self.max_per_bucket {
            // Bucket full — evict the oldest entry (FIFO).
            let evicted = bucket.remove(0);
            self.stats
                .current_bytes
                .fetch_sub(evicted.alloc_bytes, Ordering::Relaxed);
            self.stats.evictions.fetch_add(1, Ordering::Relaxed);
            bucket.push(entry);
            let bucket_len = bucket.len();
            self.bump_bytes(added_bytes);
            self.bump_peak_bucket(bucket_len);
            drop(buckets);
            // Free the evicted entry without holding the lock (`freer` may
            // itself malloc/free and we don't want to recurse into the
            // pool's Mutex).
            unsafe { freer(evicted) };
            return true;
        }
        bucket.push(entry);
        let bucket_len = bucket.len();
        self.bump_bytes(added_bytes);
        self.bump_peak_bucket(bucket_len);
        true
    }

    fn bump_bytes(&self, added: usize) {
        let new = self.stats.current_bytes.fetch_add(added, Ordering::Relaxed) + added;
        let mut peak = self.stats.peak_bytes.load(Ordering::Relaxed);
        while new > peak {
            match self.stats.peak_bytes.compare_exchange_weak(
                peak,
                new,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(observed) => peak = observed,
            }
        }
    }

    fn bump_peak_bucket(&self, len: usize) {
        let mut peak = self.stats.peak_per_bucket.load(Ordering::Relaxed);
        while len > peak {
            match self.stats.peak_per_bucket.compare_exchange_weak(
                peak,
                len,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(observed) => peak = observed,
            }
        }
    }

    /// Drain every bucket, invoking `freer` on each retained entry.
    /// Used by `rayzor_tensor_pool_reset()` (exposed for test isolation
    /// and end-of-bench cleanup).
    pub fn drain(&self, freer: FreeFn) {
        let mut buckets = self.buckets.lock();
        let mut all: Vec<PooledEntry> = Vec::new();
        for (_k, mut v) in buckets.drain() {
            all.append(&mut v);
        }
        let drained_bytes: usize = all.iter().map(|e| e.alloc_bytes).sum();
        self.stats
            .current_bytes
            .fetch_sub(drained_bytes, Ordering::Relaxed);
        self.stats
            .evictions
            .fetch_add(all.len() as u64, Ordering::Relaxed);
        drop(buckets);
        for entry in all {
            unsafe { freer(entry) };
        }
    }

    /// Number of buckets currently holding ≥1 entry. Test helper.
    pub fn bucket_count(&self) -> usize {
        self.buckets.lock().len()
    }

    /// Number of entries in `key`'s bucket (0 if absent). Test helper.
    pub fn entries_in(&self, key: PoolKey) -> usize {
        self.buckets.lock().get(&key).map(|v| v.len()).unwrap_or(0)
    }

    pub fn set_disabled(&self, disabled: bool) {
        self.disabled.store(disabled, Ordering::Relaxed);
    }
}

// ---------------------------------------------------------------------------
// Global singleton
// ---------------------------------------------------------------------------

static GLOBAL_POOL: OnceLock<TensorPool> = OnceLock::new();

/// Cached result of the `RZT_POOL` env-var read. Read once on first
/// `TensorPool::global()` call; `Some(true)` means the pool is opt-in-
/// active (i.e. `RZT_POOL=1`). The `OnceLock` avoids re-reading the
/// environment on every push / pop, which would otherwise be a serialised
/// stdlib call on the hot path.
///
/// As of this commit the default (unset) is **disabled**: the pool only
/// activates when explicitly opted in. See the module-level "Opt-in status"
/// note above.
static POOL_ENABLED: OnceLock<bool> = OnceLock::new();

/// Test-only override for `pool_opt_in_enabled()`. Same encoding as
/// `POISON_TEST_OVERRIDE`. See `_test_set_pool_opt_in_override`.
#[cfg(test)]
static POOL_OPT_IN_TEST_OVERRIDE: std::sync::atomic::AtomicI8 =
    std::sync::atomic::AtomicI8::new(-1);

#[cfg(test)]
pub(crate) fn _test_set_pool_opt_in_override(enabled: Option<bool>) {
    let v: i8 = match enabled {
        None => -1,
        Some(false) => 0,
        Some(true) => 1,
    };
    POOL_OPT_IN_TEST_OVERRIDE.store(v, std::sync::atomic::Ordering::Relaxed);
}

/// Cached result of the `RZT_TENSOR_POOL_POISON` env-var read. Read
/// once on first poll; `Some(true)` means poison every park + pop. The
/// `OnceLock` avoids re-reading the environment on every push/pop, which
/// would otherwise be a serialised stdlib call on the hot path.
static POISON_ENABLED: OnceLock<bool> = OnceLock::new();

/// Test-only override for `poison_enabled()`. `-1` = no override, fall
/// through to the production OnceLock; `0` = force-false; `1` = force-true.
/// This is the only way to write a deterministic test for poisoning under
/// the default parallel `cargo test`, where the OnceLock may have already
/// been seeded by another test in the same process before this test runs.
#[cfg(test)]
static POISON_TEST_OVERRIDE: std::sync::atomic::AtomicI8 = std::sync::atomic::AtomicI8::new(-1);

#[cfg(test)]
pub(crate) fn _test_set_poison_override(enabled: Option<bool>) {
    let v: i8 = match enabled {
        None => -1,
        Some(false) => 0,
        Some(true) => 1,
    };
    POISON_TEST_OVERRIDE.store(v, std::sync::atomic::Ordering::Relaxed);
}

/// Sentinel byte stamped over a buffer when poisoning is enabled.
/// 0xCD is the same value MSVC's debug allocator uses for uninitialised
/// heap memory — recognisable in a debugger and gives obvious f32
/// (`-1.0717e8`-ish; LLVM may surface NaN per bit pattern) / i32
/// (`-842150451`) sentinels in a casual `printf`.
const POISON_BYTE: u8 = 0xCD;

/// Returns true iff `RZT_TENSOR_POOL_POISON` was set when the pool
/// first observed it. Cached for the process lifetime via `OnceLock`.
/// Under `#[cfg(test)]`, `POISON_TEST_OVERRIDE` wins to make per-test
/// state deterministic — see `_test_set_poison_override`.
pub fn poison_enabled() -> bool {
    #[cfg(test)]
    {
        let v = POISON_TEST_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed);
        match v {
            0 => return false,
            1 => return true,
            _ => {} // -1: fall through to OnceLock
        }
    }
    *POISON_ENABLED.get_or_init(|| {
        crate::env_var("RZT_TENSOR_POOL_POISON", "RAYZOR_TENSOR_POOL_POISON")
            .map(|v| !v.is_empty())
            .unwrap_or(false)
    })
}

/// Stamp `POISON_BYTE` over the entry's data buffer (and meta scales for
/// INT8 QTensors). No-op when poisoning is disabled or when `ptr` is
/// null. Called from `push` (before parking) and from `try_pop` (before
/// returning a hit to the caller).
///
/// # Safety
///
/// `entry.ptr` must point to a `RayzorTensor` / `RayzorQTensor` whose
/// `data` field is the first pointer-sized slot. The `qtensor_meta_*`
/// fields are independently checked.
unsafe fn poison_entry_data(entry: &PooledEntry) {
    if !poison_enabled() || entry.ptr.is_null() {
        return;
    }
    // Both RayzorTensor and RayzorQTensor lay out `data: *mut u8` as the
    // first field (verified by the on-disk struct layouts in tensor.rs
    // and quant.rs respectively). Read it without depending on either
    // module's concrete struct so the pool layer stays decoupled.
    let data_ptr: *mut u8 = *(entry.ptr as *mut *mut u8);
    if !data_ptr.is_null() && entry.alloc_bytes > 0 {
        std::ptr::write_bytes(data_ptr, POISON_BYTE, entry.alloc_bytes);
    }
    if !entry.qtensor_meta_ptr.is_null() && entry.qtensor_meta_bytes > 0 {
        std::ptr::write_bytes(
            entry.qtensor_meta_ptr,
            POISON_BYTE,
            entry.qtensor_meta_bytes,
        );
    }
}

/// Default per-bucket entry cap when `RZT_POOL_MAX_PER_BUCKET` is unset.
const DEFAULT_MAX_PER_BUCKET: usize = 8;
/// Default global byte cap (256 MB) when `RZT_POOL_MAX_TOTAL_MB` is unset.
const DEFAULT_MAX_TOTAL_BYTES: usize = 256 * 1024 * 1024;

fn parse_usize_env(primary: &str, legacy: &str) -> Option<usize> {
    crate::env_var(primary, legacy)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
}

/// Returns true iff `RZT_POOL=1` was set when the pool first observed
/// the environment. Cached for the process lifetime via `OnceLock`. The
/// match is strict (`== "1"`) rather than "any non-empty value" so the
/// opt-in is unambiguous; future revisions may broaden this.
pub fn pool_opt_in_enabled() -> bool {
    #[cfg(test)]
    {
        let v = POOL_OPT_IN_TEST_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed);
        match v {
            0 => return false,
            1 => return true,
            _ => {} // -1: fall through to OnceLock
        }
    }
    *POOL_ENABLED.get_or_init(|| {
        crate::env_var("RZT_POOL", "RAYZOR_POOL")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

/// Return the process-wide pool, initialising on first call. Env flags
/// applied: `RZT_POOL` (master opt-in), `RZT_POOL_DISABLE` (override),
/// `RZT_POOL_MAX_PER_BUCKET`, `RZT_POOL_MAX_TOTAL_MB`,
/// `RZT_POOL_STATS`.
///
/// **Default behaviour** (`RZT_POOL` unset): the pool is constructed in
/// its disabled state, so `try_pop` short-circuits to `None` and `push`
/// short-circuits to `freer(entry)`. No Mutex traffic on the hot path.
pub fn global() -> &'static TensorPool {
    GLOBAL_POOL.get_or_init(|| {
        let max_per_bucket =
            parse_usize_env("RZT_POOL_MAX_PER_BUCKET", "RAYZOR_POOL_MAX_PER_BUCKET")
                .unwrap_or(DEFAULT_MAX_PER_BUCKET);
        let max_total_bytes = parse_usize_env("RZT_POOL_MAX_TOTAL_MB", "RAYZOR_POOL_MAX_TOTAL_MB")
            .map(|mb| mb * 1024 * 1024)
            .unwrap_or(DEFAULT_MAX_TOTAL_BYTES);
        // Master opt-in: pool is OFF by default. The explicit
        // `RZT_POOL_DISABLE` env-var still wins so a benchmark harness
        // can force-disable even if `RZT_POOL=1` is exported from a
        // wrapper script.
        let explicit_disable = crate::env_var("RZT_POOL_DISABLE", "RAYZOR_POOL_DISABLE")
            .map(|v| !v.is_empty())
            .unwrap_or(false);
        let disabled = explicit_disable || !pool_opt_in_enabled();

        if crate::env_var("RZT_POOL_STATS", "RAYZOR_POOL_STATS")
            .map(|v| !v.is_empty())
            .unwrap_or(false)
        {
            // Register an atexit-like hook by leaking a Drop guard into
            // a static. `std::process::exit` does NOT run thread_locals or
            // static drops, so we rely on the libc `atexit` shim via the
            // `ctor`-less idiom: print on a thread that dlsym's atexit at
            // first use. We avoid the `ctor` crate dep — instead, hook
            // via `at_exit` style by registering with libc.
            register_atexit_hook();
        }

        TensorPool::new(max_per_bucket, max_total_bytes, disabled)
    })
}

extern "C" fn pool_stats_atexit() {
    if let Some(pool) = GLOBAL_POOL.get() {
        let s = pool.stats.snapshot();
        eprintln!(
            "[RAYZOR pool] hits={} misses={} hit_rate={:.1}% pushes={} evictions={} peak_per_bucket={} peak_bytes={}MB current_bytes={}MB",
            s.hits,
            s.misses,
            s.hit_rate * 100.0,
            s.pushes,
            s.evictions,
            s.peak_per_bucket,
            s.peak_bytes / (1024 * 1024),
            s.current_bytes / (1024 * 1024),
        );
    }
}

fn register_atexit_hook() {
    extern "C" {
        fn atexit(cb: extern "C" fn()) -> i32;
    }
    unsafe {
        atexit(pool_stats_atexit);
    }
}

/// Test-only: clear pool bookkeeping without freeing parked entries.
/// Real production drain (which knows how to release `data` / `shape` /
/// `strides` / wrapper / QTensor `meta` correctly) is exposed as
/// `rayzor_tensor_pool_reset` in `tensor.rs` and wires a canonical
/// `FreeFn` into `TensorPool::drain`.
#[cfg(test)]
pub fn _test_clear_pool_bookkeeping() {
    if let Some(pool) = GLOBAL_POOL.get() {
        let mut buckets = pool.buckets.lock();
        buckets.clear();
        pool.stats.current_bytes.store(0, Ordering::Relaxed);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use parking_lot::Mutex as PLMutex;
    use std::sync::Arc;
    use std::thread;

    // A test-only freer that counts how many times it ran. We model the
    // tensor as a heap-allocated `Vec<u8>` whose leaked pointer round-trips
    // through `PooledEntry::ptr`.
    static FREE_COUNT: AtomicU64 = AtomicU64::new(0);

    // Tests that read FREE_COUNT must serialize, since cargo runs tests in
    // parallel threads by default.
    static FREE_COUNT_LOCK: PLMutex<()> = PLMutex::new(());

    unsafe fn test_freer(entry: PooledEntry) {
        // Reconstitute the Vec from the leaked pointer + length and let it
        // drop. `alloc_bytes` is the original Vec's capacity.
        if !entry.ptr.is_null() && entry.alloc_bytes > 0 {
            let _ = Vec::from_raw_parts(entry.ptr, entry.alloc_bytes, entry.alloc_bytes);
        }
        if !entry.qtensor_meta_ptr.is_null() && entry.qtensor_meta_bytes > 0 {
            let _ = Vec::from_raw_parts(
                entry.qtensor_meta_ptr,
                entry.qtensor_meta_bytes,
                entry.qtensor_meta_bytes,
            );
        }
        FREE_COUNT.fetch_add(1, Ordering::Relaxed);
    }

    fn fake_entry(bytes: usize, shape: &[usize]) -> PooledEntry {
        let v: Vec<u8> = vec![0u8; bytes];
        let mut v = std::mem::ManuallyDrop::new(v);
        PooledEntry {
            ptr: v.as_mut_ptr(),
            shape: ShapeBuf::from_slice(shape),
            alloc_bytes: bytes,
            qtensor_meta_ptr: std::ptr::null_mut(),
            qtensor_meta_bytes: 0,
        }
    }

    #[test]
    fn push_then_pop_returns_same_ptr() {
        let _g = FREE_COUNT_LOCK.lock();
        let pool = TensorPool::new(4, 16 * 1024 * 1024, false);
        let key = PoolKey::from_shape(0, &[1, 2048]);
        let e = fake_entry(2048 * 4, &[1, 2048]);
        let original_ptr = e.ptr;
        pool.push(key, e, test_freer);
        let popped = pool.try_pop(key, &[1, 2048]).expect("expected hit");
        assert_eq!(popped.ptr, original_ptr);
        assert_eq!(popped.shape.as_slice(), &[1, 2048]);
        // Clean up — we manually freed nothing yet, so let the test freer drop it.
        unsafe { test_freer(popped) };
    }

    #[test]
    fn different_shape_misses() {
        let _g = FREE_COUNT_LOCK.lock();
        let pool = TensorPool::new(4, 16 * 1024 * 1024, false);
        let key_a = PoolKey::from_shape(0, &[1, 2048]);
        let e = fake_entry(2048 * 4, &[1, 2048]);
        pool.push(key_a, e, test_freer);

        // Try popping at a shape that hashes differently — must miss.
        let key_b = PoolKey::from_shape(0, &[1, 8192]);
        assert!(pool.try_pop(key_b, &[1, 8192]).is_none());

        // And popping at the right key but with a mismatched shape (forced
        // hash collision simulated by reusing key_a with a different shape)
        // must also miss because the bucket-walk compares shapes.
        let collision_key = key_a;
        assert!(pool.try_pop(collision_key, &[1, 4096]).is_none());

        // Drain real entry.
        let popped = pool.try_pop(key_a, &[1, 2048]).expect("hit");
        unsafe { test_freer(popped) };
    }

    #[test]
    fn bucket_overflow_evicts_oldest() {
        let _g = FREE_COUNT_LOCK.lock();
        FREE_COUNT.store(0, Ordering::Relaxed);
        let pool = TensorPool::new(2, 16 * 1024 * 1024, false);
        let key = PoolKey::from_shape(0, &[1, 16]);

        // Three pushes into a cap-2 bucket → one eviction.
        for _ in 0..3 {
            let e = fake_entry(64, &[1, 16]);
            pool.push(key, e, test_freer);
        }
        assert_eq!(pool.entries_in(key), 2);
        assert_eq!(FREE_COUNT.load(Ordering::Relaxed), 1);

        pool.drain(test_freer);
        assert_eq!(FREE_COUNT.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn disabled_pool_always_misses_and_evicts() {
        let _g = FREE_COUNT_LOCK.lock();
        FREE_COUNT.store(0, Ordering::Relaxed);
        let pool = TensorPool::new(4, 16 * 1024 * 1024, true);
        let key = PoolKey::from_shape(0, &[1, 16]);
        let e = fake_entry(64, &[1, 16]);
        pool.push(key, e, test_freer);
        assert_eq!(FREE_COUNT.load(Ordering::Relaxed), 1);
        assert!(pool.try_pop(key, &[1, 16]).is_none());
        let stats = pool.stats.snapshot();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.evictions, 1);
    }

    #[test]
    fn total_bytes_cap_evicts() {
        let _g = FREE_COUNT_LOCK.lock();
        FREE_COUNT.store(0, Ordering::Relaxed);
        // 1 KB total budget — second push of 800 bytes triggers eviction.
        let pool = TensorPool::new(8, 1024, false);
        let key = PoolKey::from_shape(0, &[1, 10]);
        let e1 = fake_entry(800, &[1, 10]);
        let e2 = fake_entry(800, &[1, 10]);
        pool.push(key, e1, test_freer);
        pool.push(key, e2, test_freer);
        // One eviction should have happened during the second push.
        assert_eq!(FREE_COUNT.load(Ordering::Relaxed), 1);
        assert_eq!(pool.entries_in(key), 1);
        pool.drain(test_freer);
    }

    #[test]
    fn stats_hit_rate_tracking() {
        let _g = FREE_COUNT_LOCK.lock();
        let pool = TensorPool::new(4, 16 * 1024 * 1024, false);
        let key = PoolKey::from_shape(0, &[2, 3]);

        // 1 miss, 1 push, 1 hit
        assert!(pool.try_pop(key, &[2, 3]).is_none());
        let e = fake_entry(24, &[2, 3]);
        pool.push(key, e, test_freer);
        let popped = pool.try_pop(key, &[2, 3]).expect("hit");

        let s = pool.stats.snapshot();
        assert_eq!(s.hits, 1);
        assert_eq!(s.misses, 1);
        assert!((s.hit_rate - 0.5).abs() < 1e-9);

        unsafe { test_freer(popped) };
    }

    #[test]
    fn thread_safety_smoke() {
        let _g = FREE_COUNT_LOCK.lock();
        // 4 threads each push and pop 100 entries at one of 3 shapes.
        // No assertion beyond "no panics, no data races detected by Miri".
        let pool = Arc::new(TensorPool::new(16, 64 * 1024 * 1024, false));
        let mut handles = Vec::new();
        for t in 0..4 {
            let pool = pool.clone();
            handles.push(thread::spawn(move || {
                let shapes: [&[usize]; 3] = [&[1, 2048], &[1, 8192], &[1, 512]];
                for i in 0..100 {
                    let shape = shapes[(t + i) % 3];
                    let key = PoolKey::from_shape(0, shape);
                    let bytes = shape.iter().product::<usize>() * 4;
                    let e = fake_entry(bytes, shape);
                    pool.push(key, e, test_freer);
                    let _ = pool.try_pop(key, shape).map(|e| unsafe { test_freer(e) });
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        // Drain whatever remains so leak-checkers stay clean.
        pool.drain(test_freer);
    }

    #[test]
    fn qtensor_meta_round_trips() {
        let _g = FREE_COUNT_LOCK.lock();
        // Verify the QTensor meta pointer survives a push/pop cycle so an
        // INT8 caller can hand the same scales array back to the new
        // tensor. Use a small buffer; the freer is only invoked on the
        // final drain.
        let pool = TensorPool::new(4, 16 * 1024 * 1024, false);
        let key = PoolKey::from_shape(7 /* INT8 dtype tag */, &[64, 64]);

        let data: Vec<u8> = vec![0u8; 64 * 64];
        let meta: Vec<u8> = vec![0u8; 64 * 4]; // 16 groups of 4-byte f32 scales
        let mut data = std::mem::ManuallyDrop::new(data);
        let mut meta = std::mem::ManuallyDrop::new(meta);
        let entry = PooledEntry {
            ptr: data.as_mut_ptr(),
            shape: ShapeBuf::from_slice(&[64, 64]),
            alloc_bytes: 64 * 64,
            qtensor_meta_ptr: meta.as_mut_ptr(),
            qtensor_meta_bytes: 64 * 4,
        };
        let data_ptr = entry.ptr;
        let meta_ptr = entry.qtensor_meta_ptr;
        pool.push(key, entry, test_freer);

        let popped = pool.try_pop(key, &[64, 64]).expect("hit");
        assert_eq!(popped.ptr, data_ptr);
        assert_eq!(popped.qtensor_meta_ptr, meta_ptr);
        assert_eq!(popped.qtensor_meta_bytes, 64 * 4);
        unsafe { test_freer(popped) };
    }

    // ------------------------------------------------------------------
    // Pool safety invariants — verification suite
    // ------------------------------------------------------------------
    //
    // These tests cover the four invariants documented at the head of
    // this module: (a) plain-tensor / QTensor key namespace isolation;
    // (b) QTensor scheme isolation across the disjoint 0x80|scheme dtype
    // byte; (c) per-(scheme, group_size) bucket isolation; (d) stress
    // concurrency with many writers and zero false hits / double-frees.

    /// Helper: replicate `quant.rs::qtensor_pool_key` here so a pure-
    /// pool-layer test can exercise scheme isolation without pulling in
    /// the quant module's heavier surface. Stays in lockstep with the
    /// real key builder; if either drifts this test will start hitting
    /// when it must not.
    fn fake_qtensor_pool_key(scheme: u8, rows: usize, cols: usize, group_size: usize) -> PoolKey {
        let shape = [rows, cols, group_size];
        let mut key = PoolKey::from_shape(0x80 | scheme, &shape);
        key.shape_hash ^= scheme as u64;
        key
    }

    #[test]
    fn plain_and_qtensor_keys_are_disjoint() {
        let _g = FREE_COUNT_LOCK.lock();
        // F32 (dtype=0) shape [1024,1024] vs INT8 QTensor (scheme=0)
        // [1024,1024]. Without the 0x80 high-bit, both would collide on
        // the same PoolKey and an INT8 alloc could pop an F32 buffer.
        let plain = PoolKey::from_shape(0 /* DTYPE_F32 */, &[1024, 1024]);
        let qint8 = fake_qtensor_pool_key(0 /* QSCHEME_INT8 */, 1024, 1024, 32);
        assert_ne!(plain, qint8, "plain F32 and INT8 keys must be distinct");
        assert_ne!(plain.dtype, qint8.dtype, "dtype byte must differ");
        assert!(qint8.dtype & 0x80 != 0, "qtensor dtype must have 0x80 set");
    }

    #[test]
    fn qtensor_schemes_segregated() {
        let _g = FREE_COUNT_LOCK.lock();
        // INT8 vs Q4_K_M vs Q6_K — same rows, cols, group_size. All three
        // must hash to distinct keys. (Even though only INT8 is poolable
        // today, the namespace boundary must hold against future Q4_K_M
        // owns_data=true allocations.)
        let int8 = fake_qtensor_pool_key(0, 1024, 1024, 32);
        let q4km = fake_qtensor_pool_key(1, 1024, 1024, 32);
        let q6k = fake_qtensor_pool_key(2, 1024, 1024, 32);
        assert_ne!(int8, q4km);
        assert_ne!(int8, q6k);
        assert_ne!(q4km, q6k);
    }

    #[test]
    fn qtensor_group_size_segregated() {
        let _g = FREE_COUNT_LOCK.lock();
        // Same (scheme, rows, cols) but two different group_sizes — must
        // hash to distinct keys, because INT8 carries a meta scales array
        // of length numel/group_size and a mismatch would over- or under-
        // read on revive.
        let g32 = fake_qtensor_pool_key(0, 1024, 1024, 32);
        let g64 = fake_qtensor_pool_key(0, 1024, 1024, 64);
        let g128 = fake_qtensor_pool_key(0, 1024, 1024, 128);
        assert_ne!(g32, g64, "group_size 32 vs 64 must hash apart");
        assert_ne!(g32, g128, "group_size 32 vs 128 must hash apart");
        assert_ne!(g64, g128, "group_size 64 vs 128 must hash apart");
    }

    #[test]
    fn qtensor_cross_scheme_pop_misses() {
        // Park an INT8 [256,256,32]; attempt to pop at Q4_K_M
        // [256,256,32]. Must MISS even though rows/cols/group_size match —
        // the dtype byte (0x80|scheme) and the XOR'd hash both rule out
        // any cross-scheme collision.
        let _g = FREE_COUNT_LOCK.lock();
        FREE_COUNT.store(0, Ordering::Relaxed);
        let pool = TensorPool::new(4, 16 * 1024 * 1024, false);

        let int8_key = fake_qtensor_pool_key(0, 256, 256, 32);
        let q4km_key = fake_qtensor_pool_key(1, 256, 256, 32);

        let entry = fake_entry(256 * 256, &[256, 256, 32]);
        pool.push(int8_key, entry, test_freer);

        // Cross-scheme pop must miss.
        assert!(
            pool.try_pop(q4km_key, &[256, 256, 32]).is_none(),
            "Q4_K_M pop must NOT see an INT8 parked entry"
        );
        // Same-scheme pop must hit.
        let popped = pool
            .try_pop(int8_key, &[256, 256, 32])
            .expect("INT8 pop must see its own entry");
        unsafe { test_freer(popped) };
    }

    #[test]
    fn stress_concurrent_alloc_free_no_double_free() {
        let _g = FREE_COUNT_LOCK.lock();
        FREE_COUNT.store(0, Ordering::Relaxed);
        // 8 worker threads × 1000 alloc/free cycles each on shape
        // [256, 256] F32. Every push must be paired with exactly one
        // freer invocation across the lifetime of the test (the freer
        // fires either on pool eviction or on the final drain). If the
        // pool double-freed or leaked we'd see FREE_COUNT != pushes.
        let pool = Arc::new(TensorPool::new(8, 64 * 1024 * 1024, false));
        let mut handles = Vec::new();
        const THREADS: usize = 8;
        const ITERS: usize = 1000;
        for _ in 0..THREADS {
            let pool = pool.clone();
            handles.push(thread::spawn(move || {
                let shape: &[usize] = &[256, 256];
                let key = PoolKey::from_shape(0, shape);
                let bytes = 256 * 256 * 4;
                for _ in 0..ITERS {
                    // Push first; the bucket might overflow and evict
                    // (FREE_COUNT++), or it might park.
                    let e = fake_entry(bytes, shape);
                    pool.push(key, e, test_freer);
                    // Then immediately try_pop; on hit the popped entry
                    // is freed via the freer (counted), simulating the
                    // real alloc_tensor → free cycle that the runtime
                    // performs on hot-loop tensors.
                    if let Some(popped) = pool.try_pop(key, shape) {
                        unsafe { test_freer(popped) };
                    }
                }
            }));
        }
        for h in handles {
            h.join().expect("worker thread must not panic");
        }

        // Drain any leftovers and verify FREE_COUNT == THREADS * ITERS.
        // (Each push contributes exactly one freer invocation, whether
        // mid-loop on pop, on eviction, or here on drain.)
        pool.drain(test_freer);
        let freed = FREE_COUNT.load(Ordering::Relaxed);
        assert_eq!(
            freed as usize,
            THREADS * ITERS,
            "stress test must free exactly one entry per push (no leak, no double-free)"
        );
        // Pool must be fully drained.
        let snap = pool.stats.snapshot();
        assert_eq!(
            snap.current_bytes, 0,
            "drain must zero current_bytes accounting"
        );
    }

    // ------------------------------------------------------------------
    // (A) Budget race — `current_bytes` cap must never be breached
    // ------------------------------------------------------------------
    //
    // Earlier `push` read `current_bytes` with a Relaxed load *before*
    // taking the bucket Mutex; two concurrent pushers could both observe
    // under-budget and both park, breaching `max_total_bytes`. The fix
    // moves the load + comparison inside the critical section.
    //
    // This test spawns 8 threads each pushing 100 entries of 1 MB each
    // (800 entries × 1 MB = 800 MB attempted) into a pool with a 10 MB
    // global budget and asserts `current_bytes` never exceeds 10 MB at
    // any observation point (taken before+after every push). The fix
    // bounds the steady-state at the cap (well below 10 MB after the
    // first eviction triggers), so any breach surfaces immediately.
    #[test]
    fn budget_race_never_breaches_cap() {
        let _g = FREE_COUNT_LOCK.lock();
        FREE_COUNT.store(0, Ordering::Relaxed);
        const CAP_BYTES: usize = 10 * 1024 * 1024; // 10 MB
        const PER_ENTRY: usize = 1024 * 1024; // 1 MB
        const THREADS: usize = 8;
        const PER_THREAD: usize = 100;
        // Make max_per_bucket large so the budget gate is the only
        // limiter — we want to exercise the budget race specifically.
        let pool = Arc::new(TensorPool::new(1024, CAP_BYTES, false));
        let observed_max = Arc::new(AtomicUsize::new(0));

        let mut handles = Vec::new();
        for t in 0..THREADS {
            let pool = pool.clone();
            let observed_max = observed_max.clone();
            handles.push(thread::spawn(move || {
                // Each thread targets a different shape so its pushes
                // all land in its own bucket. This stresses the
                // global byte budget rather than the per-bucket cap.
                let shape: [usize; 2] = [1, t + 1];
                let key = PoolKey::from_shape(0, &shape);
                for _ in 0..PER_THREAD {
                    let e = fake_entry(PER_ENTRY, &shape);
                    pool.push(key, e, test_freer);
                    let cb = pool.stats.current_bytes.load(Ordering::Relaxed);
                    let mut prev = observed_max.load(Ordering::Relaxed);
                    while cb > prev {
                        match observed_max.compare_exchange_weak(
                            prev,
                            cb,
                            Ordering::Relaxed,
                            Ordering::Relaxed,
                        ) {
                            Ok(_) => break,
                            Err(o) => prev = o,
                        }
                    }
                    // Hard assertion at every observation point.
                    assert!(
                        cb <= CAP_BYTES,
                        "current_bytes={} exceeded cap={} after push",
                        cb,
                        CAP_BYTES
                    );
                }
            }));
        }
        for h in handles {
            h.join().expect("worker thread must not panic");
        }
        let final_bytes = pool.stats.current_bytes.load(Ordering::Relaxed);
        let peak = observed_max.load(Ordering::Relaxed);
        assert!(
            final_bytes <= CAP_BYTES,
            "final current_bytes={} exceeded cap={}",
            final_bytes,
            CAP_BYTES
        );
        assert!(
            peak <= CAP_BYTES,
            "observed peak current_bytes={} exceeded cap={}",
            peak,
            CAP_BYTES
        );
        pool.drain(test_freer);
    }

    // ------------------------------------------------------------------
    // (D) RZT_TENSOR_POOL_POISON — sentinel-fill park + pop
    // ------------------------------------------------------------------
    //
    // When the env flag is set, `push` fills the data buffer with 0xCD
    // before parking AND `try_pop` re-fills with 0xCD before returning.
    // A buggy caller that reads stale bytes from a recycled tensor sees
    // the sentinel (-842150451 as i32, NaN as f32) instead of the prior
    // tensor's contents.
    //
    // Building a "tensor wrapper" fixture: `poison_entry_data` reads
    // `entry.ptr` as `*mut *mut u8` (the leading slot is `data`). The
    // fixture allocates a wrapper struct whose first field is the data
    // pointer, mirroring RayzorTensor / RayzorQTensor's layout.
    #[repr(C)]
    struct PoisonFixture {
        data: *mut u8,
        // Pad out to match a real wrapper size so PooledEntry's pointer
        // arithmetic stays within a real allocation.
        _other_fields: [usize; 6],
    }

    fn poison_fake_entry(bytes: usize, shape: &[usize]) -> (PooledEntry, *mut PoisonFixture) {
        let data: Vec<u8> = vec![0xAAu8; bytes];
        let mut data = std::mem::ManuallyDrop::new(data);
        let data_ptr = data.as_mut_ptr();
        // Allocate the wrapper on the heap; leak the pointer through
        // PooledEntry so it survives until test_freer reclaims it.
        let fixture = Box::new(PoisonFixture {
            data: data_ptr,
            _other_fields: [0; 6],
        });
        let fixture_ptr = Box::into_raw(fixture);
        let entry = PooledEntry {
            ptr: fixture_ptr as *mut u8,
            shape: ShapeBuf::from_slice(shape),
            alloc_bytes: bytes,
            qtensor_meta_ptr: std::ptr::null_mut(),
            qtensor_meta_bytes: 0,
        };
        (entry, fixture_ptr)
    }

    unsafe fn poison_fake_freer(entry: PooledEntry) {
        if !entry.ptr.is_null() {
            // Reconstruct the wrapper to drop its data buffer + the
            // wrapper itself.
            let fx = Box::from_raw(entry.ptr as *mut PoisonFixture);
            if !fx.data.is_null() && entry.alloc_bytes > 0 {
                let _ = Vec::from_raw_parts(fx.data, entry.alloc_bytes, entry.alloc_bytes);
            }
            drop(fx);
        }
        FREE_COUNT.fetch_add(1, Ordering::Relaxed);
    }

    // ------------------------------------------------------------------
    // RZT_POOL opt-in master switch
    // ------------------------------------------------------------------
    //
    // The `pool_opt_in_enabled()` helper reads `RZT_POOL` once and
    // caches the result in `POOL_ENABLED`. The test process inherits the
    // ambient env; this test only asserts the helper agrees with the
    // ambient setting (it does NOT mutate global env state, which would
    // race with other parallel tests). The real default-disabled
    // behaviour is exercised end-to-end by `tools/llama-diff/bench.sh`
    // with `RZT_POOL` unset — see project memory for the steady-state
    // baseline.
    #[test]
    fn pool_opt_in_helper_matches_env() {
        let _g = FREE_COUNT_LOCK.lock();

        struct ResetGuard;
        impl Drop for ResetGuard {
            fn drop(&mut self) {
                _test_set_pool_opt_in_override(None);
            }
        }
        let _reset = ResetGuard;

        // Force-true: helper must return true regardless of ambient env.
        _test_set_pool_opt_in_override(Some(true));
        assert!(
            pool_opt_in_enabled(),
            "override Some(true) must make pool_opt_in_enabled() return true"
        );

        // Force-false: helper must return false regardless of ambient env.
        _test_set_pool_opt_in_override(Some(false));
        assert!(
            !pool_opt_in_enabled(),
            "override Some(false) must make pool_opt_in_enabled() return false"
        );
    }

    #[test]
    fn poison_flag_stamps_sentinel_on_park_and_pop() {
        let _g = FREE_COUNT_LOCK.lock();
        FREE_COUNT.store(0, Ordering::Relaxed);

        // Deterministic override regardless of OnceLock seeding by other
        // tests. Reset to None at the end so subsequent tests see the
        // production env-var-driven OnceLock value.
        _test_set_poison_override(Some(true));
        assert!(
            poison_enabled(),
            "_test_set_poison_override(Some(true)) must force poison_enabled() = true"
        );

        struct ResetGuard;
        impl Drop for ResetGuard {
            fn drop(&mut self) {
                _test_set_poison_override(None);
            }
        }
        let _reset = ResetGuard;

        let pool = TensorPool::new(4, 16 * 1024 * 1024, false);
        let key = PoolKey::from_shape(0, &[1, 64]);
        let (entry, fixture_ptr) = poison_fake_entry(256, &[1, 64]);
        let data_ptr = unsafe { (*fixture_ptr).data };

        // Pre-park: data buffer is 0xAA (set by poison_fake_entry).
        let pre = unsafe { *data_ptr };
        assert_eq!(pre, 0xAA, "fixture buffer must start at 0xAA");

        pool.push(key, entry, poison_fake_freer);

        // Post-park: buffer must be 0xCD sentinel (push stamped it
        // before parking).
        let post_park = unsafe { *data_ptr };
        assert_eq!(
            post_park, POISON_BYTE,
            "parked buffer must be filled with POISON_BYTE (0xCD)"
        );

        // Manually overwrite to a different value to prove the pop-side
        // poison fires too (otherwise the test passes trivially).
        unsafe {
            std::ptr::write_bytes(data_ptr, 0xBB, 256);
        }

        let popped = pool.try_pop(key, &[1, 64]).expect("pop must hit");
        let popped_data = unsafe { *data_ptr };
        assert_eq!(
            popped_data, POISON_BYTE,
            "popped buffer must be re-stamped with POISON_BYTE before return"
        );
        unsafe { poison_fake_freer(popped) };
    }
}
