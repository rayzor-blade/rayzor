# Rayzor Compiler Feature Backlog

This document tracks major features, enhancements, and technical debt for the Rayzor Haxe compiler.

**Status Legend:**
- 🔴 Not Started
- 🟡 In Progress
- 🟢 Complete
- ⏸️ Blocked/On Hold

---

## 1. Generics System 🟢

**Priority:** High
**Complexity:** High
**Dependencies:** Type system, MIR infrastructure
**Status:** ✅ Core Implementation Complete (2026-02-08) — Type erasure approach

### 1.1 Generic Classes End-to-End (Type Erasure)

**Status:** 🟢 Complete (2026-02-08)
**Related Files:**
- `compiler/src/tast/ast_lowering.rs` - Type arg inference from constructor/function args, return type substitution
- `compiler/src/ir/hir_to_mir.rs` - TypeParameter→I64 erasure, field load/store coercion
- `compiler/src/ir/builder.rs` - F64↔I64 bitcast in build_call_direct
- `compiler/src/ir/types.rs` - TypeVar size=8 safety net
- `compiler/src/codegen/cranelift_backend.rs` - TypeVar size/align
- `compiler/src/codegen/llvm_jit_backend.rs` - TypeVar → i64

**What Works:**
- [x] `class Container<T> { var value:T; }` — single type parameter
- [x] `class Pair<A, B> { var first:A; var second:B; }` — multiple type parameters
- [x] Explicit type args: `new Container<Int>(42)`
- [x] Inferred type args: `new Container(42)`, `new Container("hello")`, `new Pair("age", 25)`
- [x] Field access: `container.value` with correct type coercion
- [x] Method return: `container.get()` with correct type coercion
- [x] Int, Float, String types all work through erasure
- [x] GEP stride uses I64 for erased fields (not concrete type)
- [x] F64↔I64 bitcast (not value cast) for float fields

**Architecture:** Type erasure — all type parameters become I64 (8 bytes) at MIR level. One struct layout per generic class regardless of instantiation. Coercion (bitcast for floats, cast for ints) at field load/store and method call boundaries.

**Not Yet Implemented:**
- [x] Generic constraint validation — constrained TypeParameters (`<T:Interface>`) dispatch through fat pointer interface vtables
- [ ] Abstract types with generics support
- [x] Generic functions (standalone, not class methods) — return type inferred from argument types at each call site
- [x] Nested generics (`Container<Container<Int>>`) — fixed stdlib method name collision for TypeParameter receivers + user class name collision with stdlib abstracts

### 1.2 Type System Extensions

**Status:** 🟢 Complete
**Related Files:**
- `compiler/src/ir/types.rs` - IrType::TypeVar and IrType::Generic
- `compiler/src/tast/core.rs` - TypeKind::TypeParameter and GenericInstance

**Tasks:**
- [x] Add `IrType::TypeVar(String)` variant (already existed as TypeVar)
- [x] Add `IrType::Generic { base, type_args }` variant
- [ ] Add `IrType::Union { variants }` for sum types
- [ ] Add type parameter constraints support
- [ ] Implement type parameter substitution
- [ ] Add generic type equivalence checking

### 1.3 TAST Generics Infrastructure

**Status:** 🟢 Complete
**Related Files:**
- `compiler/src/tast/generics.rs` - GenericsEngine facade
- `compiler/src/tast/generic_instantiation.rs` - GenericInstantiator
- `compiler/src/tast/constraint_solver.rs` - ConstraintSolver, UnificationTable
- `compiler/src/tast/tests/generics_test.rs` - Comprehensive test suite

**Tasks:**
- [x] GenericsEngine main facade
- [x] GenericInstantiator for type instantiation
- [x] ConstraintSolver for unification
- [x] ConstraintValidator for constraint checking
- [x] InstantiationCache for performance
- [x] Cycle detection for recursive generics

### 1.4 MIR Builder Enhancements

**Status:** 🟡 Partially Complete
**Related Files:**
- `compiler/src/ir/mir_builder.rs`
- `compiler/src/ir/types.rs` - IrType::TypeVar, IrType::Generic

**Tasks:**
- [x] IrType::TypeVar for type parameters
- [x] IrType::Generic for generic instantiations
- [ ] Add `begin_generic_function()` method
- [ ] Add union creation/extraction instructions
- [ ] Test generic MIR generation

### 1.5 Monomorphization Pass

**Status:** 🟢 Core Complete - Specialization Working
**Related Files:**
- `compiler/src/ir/monomorphize.rs` - MonoKey, Monomorphizer, type substitution
- `compiler/src/ir/builder.rs` - FunctionSignatureBuilder with type_params, build_call_direct_with_type_args
- `compiler/src/ir/instructions.rs` - CallDirect with type_args
- `compiler/src/ir/tast_to_hir.rs` - lower_type_params implementation
- `compiler/src/ir/hir_to_mir.rs` - Class type param propagation to methods, type_args inference

**Tasks:**
- [x] Design monomorphization strategy (lazy vs eager) - Using lazy instantiation
- [x] Implement MonoKey caching (generic_func + type_args)
- [x] Implement type substitution algorithm
- [x] Generate specialized function names (mangling: Container__i32)
- [x] Integrate into compilation pipeline
- [x] Add monomorphization statistics/reporting
- [x] Propagate class type_params to method signatures
- [x] Implement TAST->HIR type_params lowering
- [x] Extract type_args from receiver's class type for instance method calls
- [x] Infer type_args for static generic method calls from argument types
- [x] Generate specialized function bodies (set__i32, id__i32, id__f64)
- [x] Rewrite call sites to use specialized functions
- [ ] Use SymbolFlags::GENERIC to identify monomorphizable types
- [ ] Handle recursive generic instantiation
- [x] Preserve TypeVar in MIR signatures (TypeParameter → TypeVar conversion)

**Reference:** Based on Zyntax proven approach - see GENERICS_DESIGN.md

### 1.6 Standard Library Generics

**Status:** 🟡 Blocked on Enum Support

**Tasks:**
- [ ] Implement `Vec<T>` (generic vector) - Can proceed
- [ ] Implement `Option<T>` (tagged union) - Requires enum type support in AST lowering
- [ ] Implement `Result<T, E>` (tagged union) - Requires enum type support
- [ ] Implement `Array<T>` (Haxe's dynamic array) - Existing haxe.ds.List
- [ ] Implement `Map<K, V>` (hashmap) - Existing haxe.ds.Map
- [ ] Test monomorphization with stdlib types

**Note:** Option<T> and Result<T,E> already exist in haxe.ds but enum constructor
resolution fails during AST lowering. Need to fix enum variant symbol resolution.

---

## 2. Async/Await System 🔴

**Priority:** High
**Complexity:** Very High
**Dependencies:** Generics (Promise<T>), Memory Safety

### 2.1 Async Metadata Support

**Status:** 🔴 Not Started
**Related Files:**
- `parser/src/haxe_parser.rs`
- `compiler/src/tast/ast_lowering.rs`

**Design Note:**

- **NO NEW KEYWORDS** - Maintain Haxe backward compatibility
- Use `@:async` for async functions
- Use `@:await` metadata for await points (NOT a keyword)

**Tasks:**
- [ ] Parser support for `@:async` function metadata
- [ ] Parser support for `@:await` expression metadata (as metadata, not keyword)
- [ ] AST representation for async functions
- [ ] AST representation for @:await expressions
- [ ] TAST lowering for async functions
- [ ] Validate @:await only in @:async contexts

**Acceptance Criteria:**
```haxe
@:async
function fetchData(url: String): Promise<String> {
    var response = @:await httpGet(url);
    var data = @:await parseJson(response);
    return data;
}
```

### 2.2 Promise<T> Type Implementation

**Status:** 🔴 Not Started
**Dependencies:** Generics System

**Tasks:**
- [ ] Define Promise<T> as generic class
- [ ] Implement promise states (Pending, Resolved, Rejected)
- [ ] Implement promise creation
- [ ] Implement resolve/reject mechanisms
- [ ] Implement promise chaining (.then(), .catch())
- [ ] Implement Promise.all(), Promise.race()

### 2.3 Async State Machine Transformation

**Status:** 🟡 Proof of Concept Exists
**Related Files:**
- `compiler/examples/test_cranelift_async_statemachine.rs` (POC)

**Tasks:**
- [ ] Design state machine IR representation
- [ ] Implement async function → state machine lowering
- [ ] Handle suspension points (await expressions)
- [ ] Implement resume continuation mechanism
- [ ] Generate state storage for locals
- [ ] Handle control flow across suspension points
- [ ] Integrate with runtime

**State Machine Example:**

```haxe
@:async
function foo(): Promise<Int> {
    var x = @:await a();  // Suspension point 1
    var y = @:await b();  // Suspension point 2
    return x + y;
}

// Transforms to state machine:
enum State { S0, S1(i64), S2(i64, i64), Done }
fn foo_state_machine(state: &mut State) -> ControlFlow {
    match state {
        S0 => {
            *state = S1(await_start(a()));
            Suspend
        }
        S1(promise_a) => {
            let x = await_get(promise_a);
            *state = S2(x, await_start(b()));
            Suspend
        }
        S2(x, promise_b) => {
            let y = await_get(promise_b);
            *state = Done;
            Return(x + y)
        }
    }
}
```

### 2.4 Async Runtime Implementation

**Status:** 🔴 Not Started

**Tasks:**
- [ ] Implement AsyncRuntime struct
- [ ] Promise registration and tracking
- [ ] Suspended continuation management
- [ ] Event loop implementation
- [ ] Task scheduling
- [ ] Waker/polling mechanism
- [ ] Integration with Cranelift codegen

### 2.5 Error Handling in Async

**Status:** 🔴 Not Started

**Tasks:**
- [ ] Propagate exceptions across await points
- [ ] Implement try/catch in async functions
- [ ] Promise rejection handling
- [ ] Stack trace preservation across suspensions

---

## 3. Concurrency: Lightweight Threads & Message Passing 🟢

**Priority:** Medium-High
**Complexity:** Very High
**Status:** ✅ Core Implementation Complete (2026-01-28)
**Design:** Rayzor Standard Library (extern classes) - See [STDLIB_DESIGN.md](STDLIB_DESIGN.md)

### Implementation Summary

Two threading APIs are fully implemented and tested:

1. **`rayzor.concurrent.*`** - Rayzor's native concurrent primitives
   - Thread, Channel, Arc, Mutex, MutexGuard
   - 29 runtime functions implemented
   - See `test_rayzor_stdlib_e2e.rs` for tests

2. **`sys.thread.*`** - Standard Haxe threading API
   - Thread, Mutex, Lock, Semaphore, Deque, Condition
   - 21 tests covering all primitives
   - See `test_sys_thread.rs` for tests

### 3.1 Lightweight Thread System

**Status:** 🟢 Complete

**Implemented APIs:**

**rayzor.concurrent.Thread:**
- [x] `Thread.spawn(() -> T)` - spawn thread with closure
- [x] `handle.join()` - wait for thread completion and get result
- [x] Runtime: `rayzor_thread_spawn()`, `rayzor_thread_join()`

**sys.thread.Thread:**
- [x] `Thread.create(() -> Void)` - create thread
- [x] `Thread.yield()` - yield execution
- [x] `Thread.sleep(seconds)` - sleep for duration
- [x] `handle.join()` - wait for thread
- [x] `handle.isFinished()` - check completion status
- [x] Runtime: `sys_thread_*` functions

**Closure Capture Semantics:**
- Variables captured by **value** (like Rust), not reference
- Primitives (Int, Bool, Float) are copied
- Objects/Arrays captured as pointer copies (same object)
- Use Deque/Channel for thread-safe communication

**Stdlib (Haxe):**
- [x] `rayzor/concurrent/Thread.hx` extern class
- [x] `sys/thread/Thread.hx` extern class
- [x] Type parameters for thread return values

**Compiler Integration:**
- [x] Thread intrinsic type in compiler
- [x] `lower_thread_spawn()` in stdlib lowering
- [x] `lower_thread_join()` in stdlib lowering
- [ ] Validate Send trait on closure captures (parsing works, validation not enforced)
- [x] MIR instructions for thread operations
- [x] Cranelift codegen integration

**Runtime:**
- [x] Native OS threads (1:1 model, not M:N green threads)
- [x] FFI: `rayzor_thread_spawn()`, `rayzor_thread_join()`, `rayzor_thread_is_finished()`
- [x] FFI: `sys_thread_create()`, `sys_thread_yield()`, `sys_thread_sleep()`

**API Design (Pure Haxe):**
```haxe
import rayzor.concurrent.Thread;

@:derive([Send])
class Counter {
    var count: Int = 0;
    public function increment() { count++; }
}

// Spawn lightweight thread - fire and forget
Thread.spawn(() -> {
    trace("Running in thread");
    var c = new Counter();
    c.increment();
});

// Spawn with result
var handle = Thread.spawn(() -> {
    return 42;
});
var result = handle.join();  // blocks until thread completes

// Compiler validates Send trait on captured variables
var notSend = new NonSendable();
Thread.spawn(() -> {
    use(notSend);  // ERROR: NonSendable does not implement Send
});
```

### 3.2 Channel System (Message Passing)

**Status:** 🟢 Complete

**rayzor.concurrent.Channel:**
- [x] `new Channel<T>(capacity)` - create bounded channel
- [x] `channel.send(value)` - blocking send
- [x] `channel.tryReceive()` - non-blocking receive
- [x] `channel.close()` - close channel
- [x] Runtime: `rayzor_channel_init()`, `rayzor_channel_send()`, `rayzor_channel_try_receive()`, `rayzor_channel_close()`

**sys.thread.Deque<T>:** (Thread-safe double-ended queue)
- [x] `new Deque<T>()` - create deque
- [x] `deque.add(value)` - add to back
- [x] `deque.push(value)` - add to front
- [x] `deque.pop(blocking)` - remove from front
- [x] Runtime: `sys_deque_alloc()`, `sys_deque_add()`, `sys_deque_push()`, `sys_deque_pop()`

**Stdlib (Haxe):**
- [x] `rayzor/concurrent/Channel.hx` extern class
- [x] `sys/thread/Deque.hx` extern class
- [ ] Select class/macro for multi-channel select (future enhancement)

**Compiler Integration:**
- [x] Channel<T> type in compiler
- [x] `lower_channel_*()` functions in stdlib lowering
- [ ] Validate Send trait on channel element type (parsing works, validation not enforced)
- [x] MIR instructions for channel operations
- [x] Cranelift codegen integration

**Runtime:**
- [x] Bounded channels with capacity
- [x] Blocking send/receive
- [x] Non-blocking try_receive
- [x] Channel closing semantics
- [x] FFI: `rayzor_channel_*()` functions (10 total)

### 3.3 Synchronization Primitives

**Status:** 🟢 Complete

**rayzor.concurrent.Mutex:**
- [x] `new Mutex<T>(value)` - create mutex wrapping value
- [x] `mutex.lock()` - acquire lock, returns MutexGuard
- [x] `mutex.tryLock()` - non-blocking lock attempt
- [x] `guard.get()` - access inner value
- [x] `guard.unlock()` - release lock
- [x] Runtime: `rayzor_mutex_init()`, `rayzor_mutex_lock()`, `rayzor_mutex_try_lock()`, `rayzor_mutex_unlock()`

**rayzor.concurrent.Arc:**
- [x] `new Arc<T>(value)` - create atomic reference counted pointer
- [x] `arc.clone()` - increment ref count
- [x] `arc.get()` - access inner value
- [x] `arc.strongCount()` - get reference count
- [x] Runtime: `rayzor_arc_init()`, `rayzor_arc_clone()`, `rayzor_arc_get()`, `rayzor_arc_strong_count()`

**sys.thread.Mutex:**
- [x] `new Mutex()` - create mutex
- [x] `mutex.acquire()` - blocking acquire
- [x] `mutex.tryAcquire()` - non-blocking acquire
- [x] `mutex.release()` - release lock
- [x] Runtime: `Mutex_init()`, `Mutex_lock()`, `Mutex_tryLock()`

**sys.thread.Lock:** (One-shot synchronization)
- [x] `new Lock()` - create lock
- [x] `lock.wait()` - blocking wait
- [x] `lock.wait(timeout)` - wait with timeout
- [x] `lock.release()` - signal waiting thread
- [x] Runtime: `Lock_init()`, `Lock_wait()`, `Lock_wait_timeout()`

**sys.thread.Semaphore:**
- [x] `new Semaphore(count)` - create counting semaphore
- [x] `sem.acquire()` - decrement (blocking)
- [x] `sem.tryAcquire()` - non-blocking decrement
- [x] `sem.release()` - increment
- [x] Runtime: `rayzor_semaphore_init()`, `rayzor_semaphore_acquire()`, `rayzor_semaphore_release()`

**sys.thread.Condition:**
- [x] `new Condition()` - create condition variable
- [x] `cond.acquire()` / `cond.release()` - lock management
- [x] `cond.wait()` - wait for signal
- [x] `cond.signal()` - wake one waiter
- [x] `cond.broadcast()` - wake all waiters
- [x] Runtime: `sys_condition_*()` functions

### 3.4 Send and Sync Traits

**Status:** 🟡 Parsing Complete, Validation Not Enforced
**Dependencies:** Derived Traits System
**Design:** See [SEND_SYNC_VALIDATION.md](SEND_SYNC_VALIDATION.md) for validation strategy

**Completed:**
- [x] `Send` and `Sync` in `DerivedTrait` enum
- [x] `@:derive([Send, Sync])` parsing works
- [x] Classes can be annotated with Send/Sync

**Not Yet Enforced:**
- [ ] Compile-time validation that captured variables are Send
- [ ] Compile-time validation that channel element types are Send
- [ ] Auto-derivation rules (struct is Send if all fields are Send)
- [ ] Closure capture analysis for Send validation

**Note:** The threading system works correctly at runtime. Send/Sync annotations are parsed but not enforced at compile time. This is a future enhancement for compile-time safety guarantees.

### 3.5 Memory Safety Integration

**Status:** 🟢 Runtime Complete, Compile-time Validation Partial

**Completed:**
- [x] Arc for shared ownership across threads
- [x] Mutex for interior mutability
- [x] MutexGuard for RAII-style lock management
- [x] Channel for ownership transfer between threads

**Not Yet Enforced:**
- [ ] Validate Send/Sync at MIR level
- [ ] Compile-time data race prevention
- [ ] Enforce "no shared mutable state" rule at compile time

---

## 4. Derived Trait Enforcement 🟡

**Priority:** High
**Complexity:** Medium
**Status:** Infrastructure Complete, Enforcement Partial

**Related Files:**
- `compiler/src/tast/node.rs` - DerivedTrait enum
- `compiler/src/tast/ast_lowering.rs` - Trait extraction/validation
- `compiler/docs/memory_safety_wiki.md`

### 4.1 Existing Traits (Implemented)

- [x] Clone - Explicit deep copy
- [x] Copy - Implicit bitwise copy
- [x] Debug - toString() generation (not enforced)
- [x] Default - default() static method (not enforced)

### 4.2 Equality Traits

**Status:** 🔴 Not Started

**Tasks:**
- [ ] Implement PartialEq enforcement
  - Generate `==` operator implementation
  - Validate all fields support equality
- [ ] Implement Eq enforcement
  - Requires PartialEq
  - Validate reflexivity, symmetry, transitivity
- [ ] Generate equality methods in MIR
- [ ] Test equality with complex types

**Example:**
```haxe
@:derive([PartialEq, Eq])
class Point {
    public var x: Int;
    public var y: Int;
}

var p1 = new Point(1, 2);
var p2 = new Point(1, 2);
trace(p1 == p2);  // true (auto-generated)
```

### 4.3 Ordering Traits

**Status:** 🔴 Not Started

**Tasks:**
- [ ] Implement PartialOrd enforcement
  - Generate `<`, `<=`, `>`, `>=` operators
  - Requires PartialEq
  - Validate all fields are PartialOrd
- [ ] Implement Ord enforcement
  - Requires PartialOrd + Eq
  - Validate total ordering (antisymmetric, transitive)
  - Generate `compare()` method
- [ ] Support custom comparison logic
- [ ] Test ordering with collections (sorting)

**Example:**
```haxe
@:derive([PartialEq, Eq, PartialOrd, Ord])
class Student {
    public var name: String;
    public var grade: Int;
}

var students = [student1, student2, student3];
students.sort();  // Uses auto-generated Ord
```

### 4.4 Hash Trait

**Status:** 🔴 Not Started

**Tasks:**
- [ ] Implement Hash enforcement
  - Generate `hash()` method
  - Validate all fields are hashable
  - Ensure hash consistency with Eq
- [ ] Implement hash combining algorithm
- [ ] Integrate with HashMap<K, V> (requires K: Hash + Eq)
- [ ] Test hash distribution and collisions

**Example:**
```haxe
@:derive([PartialEq, Eq, Hash])
class Key {
    public var id: Int;
    public var name: String;
}

var map = new HashMap<Key, String>();
map.set(new Key(1, "foo"), "value");
```

### 4.5 Default Trait

**Status:** 🟡 Defined, Not Enforced

**Tasks:**
- [ ] Generate `default()` static method
- [ ] Validate all fields have defaults
- [ ] Support custom default values via `@:default(value)`
- [ ] Integrate with constructors

### 4.6 Debug Trait

**Status:** 🟡 Defined, Not Enforced

**Tasks:**
- [ ] Generate `toString()` method
- [ ] Format nested structures
- [ ] Handle circular references
- [ ] Customizable formatting via metadata

---

## 5. Memory Safety Enhancements 🟢

**Status:** 🟢 Infrastructure Complete, Critical Fixes Applied

### 5.1 Completed

- [x] MIR Safety Validator infrastructure
- [x] Symbol-to-register mapping
- [x] Pipeline integration
- [x] Use-after-move detection (infrastructure)
- [x] @:derive([Clone, Copy]) validation
- [x] **Alloc instruction side effects** - Fixed LICM hoisting allocations out of loops (2026-01-28)
- [x] **Break/continue drop scope preservation** - Fixed scope stack corruption on control flow (2026-01-28)
- [x] **Tracked allocator for debugging** - Available in runtime but using libc malloc/free in production

### 5.2 Enhancement Needed

**Status:** 🟡 In Progress

**Tasks:**
- [ ] Enhance OwnershipAnalyzer to track move operations
- [ ] Mark variables as Moved in OwnershipGraph
- [ ] Implement borrow conflict detection
- [ ] Implement lifetime constraint checking
- [ ] Add more granular error messages
- [ ] Test with real safety violations

---

## 6. Standard Library Implementation 🟡

**Status:** Partial (~55% by function count)
**Last Audit:** 2025-11-27

### 6.1 Implementation Coverage Summary

| Category | Classes | Functions | Status |
|----------|---------|-----------|--------|
| Core Types (String, Array, Math) | 3 | 55 | ✅ String ✅, Array ✅, Math ✅ |
| Concurrency (Thread, Arc, Mutex, Channel) | 5 | 32 | ✅ 100% |
| System I/O (Sys) | 1 | 10/20 | 🟡 50% |
| Standard Utilities (Std, Type, Reflect) | 3 | 5/15 | 🟡 33% |
| File System (File, FileSystem, etc.) | 6 | 22/25 | 🟡 88% |
| Date | 6 | 17/17 | ✅ 100% |
| Networking (Socket, Host, SSL) | 6 | 0 | 🔴 0% |
| Data Structures (Maps, List) | 4 | 18/30 | 🟡 60% |
| **Total** | **37** | **159** | **~65%** |

### 6.2 Core Types Status

**String Class - VERIFIED STABLE ✅ (2025-11-25):**
- [x] length - get string length (haxe_string_len)
- [x] charAt(index) - get character at index (haxe_string_char_at_ptr)
- [x] charCodeAt(index) - get ASCII code at index (haxe_string_char_code_at_ptr)
- [x] indexOf(needle, startIndex) - find substring (haxe_string_index_of_ptr)
- [x] lastIndexOf(needle, startIndex) - find last occurrence (haxe_string_last_index_of_ptr)
- [x] substr(pos, len) - extract substring by position (haxe_string_substr_ptr)
- [x] substring(start, end) - extract substring by indices (haxe_string_substring_ptr)
- [x] toUpperCase() - convert to uppercase (haxe_string_upper)
- [x] toLowerCase() - convert to lowercase (haxe_string_lower)
- [x] toString() - copy string (haxe_string_copy)
- [x] String.fromCharCode(code) - create from char code (haxe_string_from_char_code)
- [x] split(delimiter) - split string (haxe_string_split_ptr)

> ✅ **Verified:** All 12 String methods tested and working with correct type handling.
> Runtime functions use i32 for Int parameters, Ptr(String) for String parameters.

**Array<T> Class - VERIFIED WORKING ✅ (2025-11-25):**
- [x] length - get array length (haxe_array_length)
- [x] push(item) - add element (haxe_array_push)
- [x] pop() - remove and return last element (haxe_array_pop_ptr)
- [x] arr[index] - index access (haxe_array_get_i64)
- [ ] slice(start, end) - needs testing
- [ ] reverse() - needs testing
- [ ] insert(pos, item) - needs testing
- [ ] remove(item) - needs testing

> ✅ **Verified:** Core Array operations (push, pop, length, index access) working.
> Values stored as 64-bit with proper i32->i64 extension for consistent elem_size.

**Math Class - VERIFIED WORKING ✅ (2025-11-25):**
- [x] Math.abs(x) - absolute value (haxe_math_abs)
- [x] Math.floor(x) - floor (haxe_math_floor)
- [x] Math.ceil(x) - ceiling (haxe_math_ceil)
- [x] Math.sqrt(x) - square root (haxe_math_sqrt)
- [x] Math.sin(x) - sine (haxe_math_sin)
- [x] Math.cos(x) - cosine (haxe_math_cos)
- [x] Math.min(a,b), Math.max(a,b) - min/max
- [x] Math.pow(base,exp), Math.exp(x), Math.log(x)
- [ ] Math.random() - needs testing

> ✅ **Verified:** All Math operations use f64 parameter/return types via `get_extern_function_signature`.

**Concurrency Primitives (32 functions) - VERIFIED STABLE:**
- [x] Thread<T> - 8 functions (spawn, join, isFinished, yieldNow, sleep, currentId)
- [x] Arc<T> - 6 functions (init, clone, get, strongCount, tryUnwrap, asPtr)
- [x] Mutex<T> - 6 functions (init, lock, tryLock, isLocked, guardGet, unlock)
- [x] MutexGuard<T> - 2 functions (get, unlock)
- [x] Channel<T> - 10 functions (init, send, trySend, receive, tryReceive, close, etc.)

**Memory Management (5 functions):**
- [x] Vec<u8> - malloc, realloc, free, len, capacity

### 6.3 Partially Implemented 🟡

**Sys Class (10/20 functions) - VERIFIED ✅ (2025-11-27):**
- [x] print (int/float/bool)
- [x] println
- [x] exit
- [x] time - Sys.time()
- [x] cpuTime - Sys.cpuTime()
- [x] systemName - Sys.systemName()
- [x] getCwd - Sys.getCwd()
- [x] getEnv - Sys.getEnv(key)
- [x] putEnv - Sys.putEnv(key, value)
- [x] sleep - Sys.sleep(seconds)
- [x] programPath - Sys.programPath()
- [x] command - Sys.command(cmd)
- [ ] args (only count implemented)
- [ ] setCwd
- [ ] executablePath
- [ ] getChar (runtime exists, not tested)
- [ ] stdin, stdout, stderr

### 6.4 Not Implemented - HIGH PRIORITY 🔴

**Priority 1: Standard Utilities**

**Std Class - VERIFIED ✅ (2025-11-27):**
- [x] Std.string(v) - convert value to string
- [x] Std.int(f) - convert float to int
- [x] Std.parseInt(s) - parse string to int
- [x] Std.parseFloat(s) - parse string to float
- [x] Std.random(max) - random int 0..max-1
- [ ] Std.is(v, t) - type check (requires RTTI)
- [ ] Std.downcast<T>(v, c) - safe downcast (requires RTTI)

**Type Class** - Runtime reflection
```haxe
extern class Type {
    static function getClass<T>(o:T):Class<T>;
    static function getClassName(c:Class<Dynamic>):String;
    static function getSuperClass(c:Class<Dynamic>):Class<Dynamic>;
    static function getInstanceFields(c:Class<Dynamic>):Array<String>;
    static function createInstance<T>(c:Class<T>, args:Array<Dynamic>):T;
    static function createEmptyInstance<T>(c:Class<T>):T;
    static function typeof(v:Dynamic):ValueType;
    static function enumIndex(e:EnumValue):Int;
    // ... more methods
}
```

**Reflect Class** - Dynamic field access
```haxe
extern class Reflect {
    static function field(o:Dynamic, name:String):Dynamic;
    static function setField(o:Dynamic, name:String, value:Dynamic):Void;
    static function hasField(o:Dynamic, name:String):Bool;
    static function fields(o:Dynamic):Array<String>;
    static function isFunction(f:Dynamic):Bool;
    static function callMethod(o:Dynamic, func:Dynamic, args:Array<Dynamic>):Dynamic;
    static function deleteField(o:Dynamic, name:String):Bool;
    static function copy<T>(o:Null<T>):Null<T>;
}
```

**Priority 2: File System I/O - VERIFIED ✅ (2025-11-27)**

**FileSystem Class:**
- [x] exists(path) - check if path exists
- [x] isDirectory(path) - check if path is directory
- [x] isFile(path) - check if path is file
- [x] createDirectory(path) - create directory
- [x] deleteDirectory(path) - delete directory
- [x] deleteFile(path) - delete file
- [x] rename(oldPath, newPath) - rename/move file
- [x] fullPath(relativePath) - get full absolute path
- [x] absolutePath(relativePath) - get absolute path
- [x] stat(path) - file/directory stats (returns FileStat with size, mtime, etc.)
- [x] readDirectory(path) - list directory contents (returns Array<String>)

**File Class:**
- [x] getContent(path) - read file as string
- [x] saveContent(path, content) - write string to file
- [x] copy(src, dst) - copy file
- [x] read(path) - open for reading (FileInput) - runtime impl done
- [x] write(path) - open for writing (FileOutput) - runtime impl done
- [x] append(path) - open for appending - runtime impl done
- [x] update(path) - open for updating - runtime impl done
- [ ] getBytes(path) - read file as bytes (needs haxe.io.Bytes)
- [ ] saveBytes(path, bytes) - write bytes to file (needs haxe.io.Bytes)

**FileInput/FileOutput Classes:** 🟡 Runtime Implemented (2025-11-28)
- [x] readByte() - read single byte
- [x] writeByte(c) - write single byte
- [x] close() - close file handle
- [x] flush() - flush output buffer
- [x] tell() - get current position
- [x] eof() - check if at end of file
- [x] seek(p, pos) - seek to position (runtime impl done)
- [ ] readBytes/writeBytes - needs haxe.io.Bytes type
- [ ] readLine/readAll - needs full Input class support
- **Note**: Type inference for extern class return types needs fixing.
  Using explicit type annotation works: `var output:FileOutput = File.write(...)`

### 6.5 Not Implemented - MEDIUM PRIORITY 🔴

**Date Class:** ✅ Complete (2025-11-28)
- [x] new(year, month, day, hour, min, sec) - constructor
- [x] now() - get current date/time
- [x] fromTime(t) - create from timestamp (milliseconds)
- [x] fromString(s) - parse from string
- [x] getTime() - get timestamp in milliseconds
- [x] getHours/Minutes/Seconds() - local timezone
- [x] getFullYear/Month/Date/Day() - local timezone
- [x] getUTCHours/Minutes/Seconds() - UTC
- [x] getUTCFullYear/Month/Date/Day() - UTC
- [x] getTimezoneOffset() - timezone offset in minutes
- [x] toString() - format as "YYYY-MM-DD HH:MM:SS"

**Data Structure Classes** 🟡 Partial
- [x] IntMap<T> - Integer key hash map (runtime impl done)
- [x] StringMap<T> - String key hash map (runtime impl done)
- [ ] ObjectMap<K,V> - Object key hash map (needs RTTI)
- [ ] List<T> - Linked list

**Exception/Stack Trace**
- Exception class with stack trace
- NativeStackTrace for capture

### 6.6 Not Implemented - LOW PRIORITY 🔴

**Networking (requires async)**
- Host - DNS resolution
- Socket - TCP/UDP sockets
- sys.ssl.* - SSL/TLS support

**System Threading (alternative to rayzor.concurrent)**
- sys.thread.Lock
- sys.thread.Mutex (different from rayzor.concurrent.Mutex)
- sys.thread.Tls<T>
- sys.thread.Semaphore
- sys.thread.Condition
- sys.thread.Deque

**Compile-Time Features (N/A for JIT)**
- MacroType - Macro metaprogramming

### 6.7 Implementation Plan

**Phase 1: Standard Utilities (Est. 3-4 days)**
1. Implement Std class runtime functions
2. Basic Type class (getClassName, typeof, is)
3. Basic Reflect class (field, setField, hasField)

**Phase 2: Complete Sys Class (Est. 2 days)**
1. Environment variables (getEnv, putEnv)
2. Working directory (getCwd, setCwd)
3. System info (systemName, cpuTime)
4. Command execution (command)

**Phase 3: File System I/O (Est. 4-5 days)**
1. FileSystem class (exists, stat, directory ops)
2. File class (content read/write)
3. FileInput/FileOutput streams

**Phase 4: Date/Time (Est. 1-2 days)**
1. Date class with all methods
2. Date formatting and parsing

**Phase 5: Data Structures (Est. 2-3 days)**
1. IntMap<T> with runtime backing
2. StringMap<T> with runtime backing
3. ObjectMap<K,V> (may need generics)
4. List<T> implementation

**Phase 6: Advanced Features (Future)**
1. Networking (requires async infrastructure)
2. SSL/TLS support
3. Enhanced reflection

### 6.8 Runtime Implementation Strategy

Each extern class requires:
1. **Haxe Declaration** - `compiler/haxe-std/<Class>.hx` with `extern class` and `@:native` metadata
2. **Rust Runtime** - `runtime/src/haxe_<class>.rs` with C-ABI functions
3. **Symbol Registration** - Add to `runtime/src/plugin_impl.rs`
4. **Stdlib Mapping** - Add to `compiler/src/stdlib/runtime_mapping.rs` if needed

**Example Pattern:**
```rust
// runtime/src/haxe_std.rs
#[no_mangle]
pub extern "C" fn haxe_std_parse_int(s: *const u8, len: usize) -> i64 {
    // Implementation
}

// runtime/src/plugin_impl.rs
inventory::submit! { RayzorSymbol::new("haxe_std_parse_int", haxe_std_parse_int as *const ()) }
```

---

## 7. Error Recovery & Diagnostics 🟡

**Status:** Basic Implementation

**Tasks:**
- [ ] Enhanced error recovery in parser
- [ ] Better error messages with suggestions
- [ ] Error codes and categories
- [ ] IDE integration (LSP)
- [ ] Warning levels and configuration
- [ ] Error aggregation and reporting

---

## 8. Optimization Passes 🟡

**Status:** Basic Passes Implemented

### Implemented
- [x] Dead code elimination
- [x] Constant folding
- [x] Copy propagation

### Needed
- [ ] Inlining (method and function)
- [ ] Loop optimizations (unrolling, invariant hoisting)
- [ ] Escape analysis
- [ ] Devirtualization
- [ ] Tail call optimization
- [ ] SIMD vectorization

---

## 9. Testing Infrastructure 🟡

**Status:** Comprehensive Test Suite with CI Infrastructure

### 9.1 Completed

- [x] **600/600 tests passing** (100% pass rate as of 2026-01-28)
- [x] **Docker stress test environment** (`ci/bench-test/`) for reproducible amd64 testing
- [x] **SIGSEGV signal handler** for crash diagnosis with stack traces
- [x] **Automated stress testing** (20-iteration default, configurable)
- [x] Comprehensive unit tests across parser, runtime, compiler
- [x] E2E integration tests (test_rayzor_stdlib_e2e, test_core_types_e2e)
- [x] Macro system tests (113 unit tests)
- [x] Tiered JIT stress tests (20/20 stability)

### 9.2 In Progress / Needed

**Tasks:**
- [ ] Comprehensive generics test suite
- [ ] Async/await integration tests
- [ ] Memory safety violation tests (edge cases)
- [ ] Performance benchmarks (formal suite)
- [ ] Fuzzing infrastructure
- [x] CI/CD GitHub Actions integration ✅ (benchmarks workflow on push/schedule)

---

## 10. Documentation 🟡

**Status:** Core Documentation Exists

**Tasks:**
- [ ] Complete API documentation
- [ ] Generics user guide
- [ ] Async/await tutorial
- [ ] Concurrency guide
- [ ] Memory safety best practices
- [ ] Performance tuning guide
- [ ] Migration guide (from Haxe)
- [ ] Contributing guide

---

## Implementation Priority Order

### Phase 1: Foundation (Mostly Complete)
1. ✅ Memory safety infrastructure
2. ✅ Property access support (getter/setter)
3. 🟡 Derived traits (Clone, Copy, Send, Sync parsing)
4. 🔴 Generic metadata pipeline integration

### Phase 2: JIT Execution ✅ COMPLETE (2026-01-28)

5. ✅ **JIT Execution - Runtime concurrency primitives** (29 functions implemented)
6. ✅ **JIT Execution - Cranelift integration** (plugin system working)
7. ✅ **JIT Execution - E2E test execution** (tiered backend 20/20 stress tests)

### Phase 3: Core Features
8. 🔴 Generics type system
9. 🔴 Monomorphization
10. 🔴 Equality and ordering traits
11. 🔴 Hash trait

### Phase 4: Advanced Features
12. 🔴 Async/await infrastructure
13. 🔴 Promise<T> implementation
14. 🔴 State machine transformation

### Phase 5: Concurrency Safety
15. 🔴 Send/Sync validation (compiler-enforced)
16. 🔴 Capture analysis for closures
17. 🔴 Thread safety validation in MIR

### Phase 6: Polish
18. 🔴 Performance optimization
19. 🔴 Comprehensive testing
20. 🔴 Complete documentation

### Current Blockers

**For JIT Execution:** ✅ RESOLVED (2026-01-28)
- ~~Missing runtime concurrency primitives~~ ✅ All 29 functions implemented
- ~~Missing Cranelift symbol registration~~ ✅ Plugin system working
- ~~Broken test examples~~ ✅ Fixed and passing
- ~~Alloc instruction LICM hoisting~~ ✅ Fixed - Alloc now has side effects
- ~~Break/continue drop scope corruption~~ ✅ Fixed - State preserved across branches

**For Full Concurrency Support:**
1. ✅ JIT execution works (tiered backend 20/20 stress tests passing)
2. 🔴 Send/Sync trait validation (design exists, not implemented)
3. 🔴 Capture analysis for closures

**Remaining Work:**
1. Generics constraint validation and abstract types
2. Async/await state machine transformation
3. Full RTTI for Type/Reflect classes

---

---

## 11. Haxe Property Access Support 🟡

**Priority:** Medium
**Complexity:** Medium
**Status:** Infrastructure Complete - Basic structure in place, method call generation pending

**Related Files:**
- `compiler/src/tast/node.rs` - PropertyAccessInfo and PropertyAccessor types ✅
- `compiler/src/tast/ast_lowering.rs` - Property lowering with accessor info ✅
- `compiler/src/ir/hir.rs` - HirClassField with property_access ✅
- `compiler/src/ir/tast_to_hir.rs` - Property info propagation ✅
- `compiler/src/ir/hir_to_mir.rs` - Field access lowering with property checks ✅
- `parser/src/haxe_ast.rs` - PropertyAccess enum ✅

### Current State

**What Works:** ✅
- @:coreType extern classes (Array, String) properties route through StdlibMapping
- `array.length` → `haxe_array_length()` runtime call
- PropertyAccessInfo stored in TAST TypedField
- Property accessor info propagated through HIR
- property_access_map populated during MIR lowering
- Field access checks for property getters (infrastructure)

**What's Missing:** ❌
- Method call generation for custom getters (placeholder only)
- Setter call generation in lower_lvalue_write
- Method name resolution (get_x/set_x convention)
- Full enforcement of property access modes (null, never) - partially done

### Property Access Modes

```haxe
// 1. Direct field access
var x(default, default):Int;

// 2. Read-only
var length(default, null):Int;

// 3. Custom getter/setter (naming convention)
var x(get, set):Int;
function get_x():Int { return _x * 2; }
function set_x(v:Int):Int { _x = v; return v; }

// 4. Custom named accessors
var y(getY, setY):Int;
function getY():Int { return _y; }
function setY(v:Int):Int { _y = v; return v; }

// 5. Never/Null access control
var z(get, never):Int;  // Read-only via getter
```

### Implementation Tasks

**Phase 1: TAST Storage** ✅ COMPLETE
- [x] Add `PropertyAccessInfo` and `PropertyAccessor` to `TypedField` struct
- [x] Store getter/setter information during `lower_class_field()`
- [x] Convert PropertyAccess to PropertyAccessor in ast_lowering
- [x] Add property_access field to HirClassField
- [x] Propagate property info through TAST→HIR→MIR pipeline

**Phase 2: MIR Infrastructure** ✅ COMPLETE
- [x] Add `property_access_map` to HirToMirContext
- [x] Populate property_access_map in register_class_metadata
- [x] Update `lower_field_access()` to check property info
- [x] Add checks for Null/Never accessors

**Phase 3: Method Call Generation** ✅ COMPLETE
- [x] Change PropertyAccessor::Method to store InternedString (method name) instead of SymbolId
- [x] Generate method calls for custom getters in lower_field_access
- [x] Update `lower_lvalue_write()` for custom setters
- [x] Handle read-only properties (null/never setter)
- [x] Error on write to read-only property

**Phase 4: Method Name Resolution** ✅ COMPLETE
- [x] Store method names in convert_property_accessor
- [x] Derive `get_<name>` and `set_<name>` from PropertyAccess::Custom("get")
- [x] Support custom accessor names PropertyAccess::Custom("getMyX")
- [x] Look up accessor methods in function_map by name during MIR lowering
- [x] Error on missing accessor methods with helpful message

**Phase 5: Testing** (1 day) - TODO
- [ ] Test all PropertyAccess modes (default, get, set, custom, null, never)
- [ ] Test read-only properties
- [ ] Test write-only properties (rare)
- [ ] Test property inheritance
- [ ] Test error messages for violations

### Current Workaround

For @:coreType extern classes in stdlib:
- Manually add property mappings to StdlibMapping
- Example: `Array.length` → `haxe_array_length` (0-param getter)
- Works because @:coreType has NO actual fields

### Acceptance Criteria

- [x] PropertyAccessInfo stored and propagated through compilation pipeline
- [x] property_access_map populated for all properties
- [x] Field access checks for property accessors (infrastructure)
- [x] Properties with `(get, set)` call `get_x()/set_x()` methods
- [x] Properties with custom names `(getX, setX)` call those methods
- [x] Read-only properties `(get, null)` allow read but error on write
- [x] Default properties `(default, default)` use direct field access
- [ ] All test cases pass for property access modes (basic test passes, needs comprehensive suite)

### Progress Summary

**Fully Implemented (Phases 1-4):**
1. Added PropertyAccessInfo and PropertyAccessor types to TAST
2. PropertyAccessor::Method stores InternedString (method name)
3. convert_property_accessor derives get_x/set_x from PropertyAccess::Custom("get")
4. Property info propagates TAST→HIR→MIR
5. property_access_map populated during register_class_metadata
6. Getter calls generated in lower_field_access with method name lookup
7. Setter calls generated in lower_lvalue_write with method name lookup
8. Read-only property enforcement (error on write to Null/Never setter)
9. Write-only property enforcement (error on read from Null/Never getter)
10. Proper error reporting for missing getter/setter methods
11. All existing tests pass (7/7 e2e tests)
12. Basic property test passes (test_property.hx with getter/setter)

**Remaining (Phase 5):**
1. Comprehensive test suite for all property modes
2. Property inheritance tests
3. Edge case testing (static properties, property overrides, etc.)

---

## 12. JIT Execution (Cranelift Backend) 🟢

**Priority:** High
**Complexity:** Medium-High
**Status:** ✅ Complete - Tiered JIT Working (2026-01-28)

**Related Files:**
- `compiler/src/codegen/cranelift_backend.rs` - Cranelift JIT backend ✅
- `compiler/examples/test_full_pipeline_cranelift.rs` - Full pipeline test (needs update)
- `compiler/examples/test_rayzor_stdlib_e2e.rs` - E2E tests (currently at L4 MIR validation)
- `runtime/src/lib.rs` - Runtime library (missing concurrency primitives)

### Current State

**What Works:** ✅
- Cranelift backend infrastructure exists
- MIR → Cranelift IR compilation
- Basic JIT compilation for simple functions
- Full pipeline: Haxe → AST → TAST → HIR → MIR → Cranelift
- Runtime: malloc, realloc, free, Vec, String, Array, Math functions
- All 7/7 e2e tests compile to MIR and pass validation (L4)

**What's Missing (Blockers for L5/L6):** ❌

**Critical Blockers:**

1. **~~Missing Runtime Implementations~~** ✅ RESOLVED (2025-11-16)
   - **Status:** ✅ All 29 concurrency runtime functions implemented in `runtime/src/concurrency.rs`
   - **Stdlib:** ✅ Extern declarations exist in `compiler/src/stdlib/{thread,channel,sync}.rs`
   - **Runtime:** ✅ C-ABI implementations using std::thread/Arc/Mutex/mpsc
   - **Plugin:** ✅ All symbols registered in `runtime/src/plugin_impl.rs`
   - **Verification:** ✅ All 7 e2e tests compile and pass MIR validation

   **Implementation Details:**
   - Thread: wraps std::thread::JoinHandle (spawn, join, is_finished, yield, sleep, current_id)
   - Arc: wraps std::sync::Arc (init, clone, get, strong_count, try_unwrap, as_ptr)
   - Mutex: wraps std::sync::Mutex (init, lock, try_lock, is_locked, guard_get, unlock)
   - Channel: wraps std::sync::mpsc (init with bounded/unbounded, send, receive, close, query ops)

   **Note:** Thread spawn uses placeholder - proper closure invocation requires FFI trampoline (enhancement for later)

2. **~~Broken Test Examples~~** ✅ RESOLVED (2025-11-16)
   - **Status:** ✅ `test_full_pipeline_cranelift.rs` fixed and passing
   - **Changes:** Updated to use CompilationUnit API, added runtime symbols
   - **Tests:** All 3 JIT execution tests pass (add, max, sumToN)

3. **~~Missing L5/L6 Infrastructure~~** ✅ RESOLVED (Infrastructure Complete, 2025-11-16)
   - **Status:** ✅ L5/L6 infrastructure implemented and working
   - **What works:**
     - Cranelift backend compiles and executes code (test_full_pipeline_cranelift.rs)
     - L5 (Codegen) level compiles MIR to native code with runtime symbols
     - L6 (Execution) level retrieves function pointers and verifies executability
   - **Known Issue:**
     - ⚠️ Function signature conflicts when compiling multiple stdlib modules
     - This is a Cranelift backend limitation (function redeclaration with different signatures)
     - Affects full e2e test execution but not the infrastructure itself
   - **Remaining work:**
     - Actual function execution with parameter passing
     - Result validation and assertions
     - Fixing Cranelift signature conflict issue

### Implementation Plan

**Phase 1: Runtime Concurrency Primitives** ✅ COMPLETE
- [x] Create `runtime/src/concurrency.rs` module
- [x] Implement `rayzor_thread_spawn()` using std::thread
- [x] Implement `rayzor_thread_join()`
- [x] Implement Arc primitives (init, clone, drop)
- [x] Implement Mutex primitives (init, lock, unlock)
- [x] Implement Channel primitives (init, send, receive, try_receive, close)
- [x] Export symbols in runtime/src/lib.rs
- [x] Add FFI signatures for Cranelift integration

**Phase 2: Cranelift Runtime Integration** ✅ COMPLETE
- [x] Register runtime function symbols in plugin system
- [x] All 29 symbols registered in `runtime/src/plugin_impl.rs`
- [x] Symbols available via `rayzor_runtime::plugin_impl::get_plugin()`
- [x] Verified all 7 e2e tests compile with symbols present

**Phase 3: Fix Test Examples** ✅ COMPLETE
- [x] Fix `test_full_pipeline_cranelift.rs` AstLowering API usage
- [x] Update to use CompilationUnit instead of manual lowering
- [x] Verify basic arithmetic/control flow execution works
- [x] All 3 tests pass: add, max (if/else), sumToN (while loop with SSA)

**Phase 4: E2E Execution Tests** ⚠️ PARTIAL (Infrastructure Complete)
- [x] Add L5 (Codegen) support to test_rayzor_stdlib_e2e.rs
- [x] Add L6 (Execution) support with function pointer verification
- [x] Implement compilation harness with Cranelift + runtime symbols
- [ ] Add actual function execution (currently blocked by signature conflicts)
- [ ] Add expected output/behavior validation
- [ ] Test all 7 concurrency test cases end-to-end

**Status:** Infrastructure is complete and working. Execution blocked by Cranelift function signature conflicts when compiling multiple stdlib modules. This is a backend limitation, not an infrastructure issue.

**Phase 5: Documentation & Polish** (1 day)
- [ ] Document runtime API for concurrency primitives
- [ ] Add execution examples to README
- [ ] Performance benchmarks (JIT vs interpretation)
- [ ] Update BACKLOG with JIT execution status

### Current Status (2025-11-16)

E2E test infrastructure now supports all levels:
- ✅ L1: TAST lowering
- ✅ L2: HIR lowering
- ✅ L3: MIR lowering
- ✅ L4: MIR validation (extern functions registered, CFG valid)
- ✅ L5: Codegen (Cranelift JIT compilation with runtime symbols)
- ✅ L6: Execution (function pointer verification, ready for execution)

**Default behavior:** Tests run to L4 (MIR Validation) for backward compatibility.
**L5/L6 capability:** Infrastructure complete - use `.expect_level(TestLevel::Codegen)` or `.expect_level(TestLevel::Execution)` to test JIT compilation and execution.

**Known Limitation:** Cranelift function signature conflicts when compiling multiple stdlib modules. This affects full e2e execution but does NOT affect single-file tests (test_full_pipeline_cranelift.rs works perfectly).

### Acceptance Criteria

- [x] All runtime concurrency functions implemented and exported (29 functions)
- [x] Cranelift backend registers all runtime symbols (via plugin system)
- [x] test_full_pipeline_cranelift compiles and runs (3/3 tests passing)
- [x] L5 (Codegen) infrastructure working in e2e tests
- [x] L6 (Execution) infrastructure working in e2e tests
- [ ] All 7 e2e tests reach L5/L6 (blocked by Cranelift signature conflicts)
- [ ] Thread spawn/join executes correctly (placeholder implementation, needs FFI trampoline)
- [ ] Arc/Mutex synchronization works (runtime code ready, needs execution tests)
- [ ] Channel send/receive works (runtime code ready, needs execution tests)
- [x] No memory leaks or crashes (verified for arithmetic/control flow tests)

### Estimated Timeline

**Total: 7-8 days**
- Runtime primitives: 2-3 days
- Cranelift integration: 1 day
- Fix test examples: 1 day
- E2E execution tests: 2 days
- Documentation: 1 day

### Dependencies

- ✅ MIR lowering (complete)
- ✅ Stdlib mapping (complete)
- ✅ Property access (complete)
- ✅ Runtime concurrency primitives (complete - 29 functions)
- ✅ Cranelift symbol registration (complete - plugin system)
- ✅ Alloc instruction side effects (fixed 2026-01-28)
- ✅ Break/continue drop scope state (fixed 2026-01-28)

---

## 13. Inline C / TinyCC Runtime API 🟢

**Priority:** Medium
**Complexity:** Medium-High
**Dependencies:** TCC linker integration (complete), stdlib infrastructure
**Status:** ✅ Core Complete (2026-01-31)

### Overview

TinyCC is exposed as a first-class API in `rayzor.runtime.CC` for runtime C compilation, plus `untyped __c__()` syntax for inline C code with automatic TCC lifecycle management. See [runtime/CC_FEATURES.md](runtime/CC_FEATURES.md) for full documentation.

### 13.1 Explicit API: `rayzor.runtime.CC` Extern Class

**Status:** 🟢 Complete

**Related Files:**
- `compiler/haxe-std/rayzor/runtime/CC.hx` - Extern class declaration
- `runtime/src/tinycc_runtime.rs` - Rust runtime (16 functions)
- `runtime/src/plugin_impl.rs` - Symbol registration
- `compiler/src/stdlib/runtime_mapping.rs` - Stdlib mappings

**Implemented Methods:**
- [x] `CC.create()` — create TCC context (output to memory)
- [x] `cc.compile(code)` — compile C source string (panics on failure)
- [x] `cc.addSymbol(name, value)` — register symbol for `extern long` access
- [x] `cc.relocate()` — link and relocate into executable memory (panics on failure)
- [x] `cc.getSymbol(name)` — get function/symbol address (panics if not found)
- [x] `cc.addFramework(name)` — load macOS framework or shared library via dlopen
- [x] `cc.addIncludePath(path)` — add include search directory
- [x] `cc.addFile(path)` — add .c, .o, .a, .dylib/.so/.dll file
- [x] `cc.delete()` — free TCC context (JIT code remains valid)
- [x] `CC.call0(fn)` through `CC.call3(fn, a, b, c)` — call JIT functions

**E2E Tests:** 6 tests in `compiler/examples/test_cc_e2e.rs`

### 13.2 Inline C: `untyped __c__()` Syntax

**Status:** 🟢 Complete

**Related Files:**
- `compiler/src/ir/hir_to_mir.rs` — `lower_inline_code()` (~200 lines)
- `compiler/src/tast/ast_lowering.rs` — metadata parsing

**Features:**
- [x] `untyped __c__("C code")` — auto-manages TCC lifecycle (create → compile → relocate → call → delete)
- [x] Argument passing via `{0}`, `{1}`, ... placeholders → `extern long __argN` symbols
- [x] Return value support (long → Int)
- [x] Module-local `@:cstruct` typedef auto-injection (no manual `cdef()` needed)
- [x] System header support (`#include <string.h>`, etc.) with auto-discovered SDK paths
- [x] Error handling — TCC compile/relocate/symbol errors trigger panics (catchable via try-catch)

**E2E Tests:** Tests 13-16 in `compiler/examples/test_cstruct_e2e.rs`

### 13.3 Metadata for 3rd Party Library Integration

**Status:** 🟢 Complete

All metadata works on both classes and functions. When `__c__()` is used, metadata from the enclosing function and all module-local classes is collected automatically.

**Related Files:**
- `compiler/src/tast/symbols.rs` — `frameworks`, `c_includes`, `c_sources`, `c_libs` fields on Symbol
- `compiler/src/tast/ast_lowering.rs` — metadata parsing (class-level + method-level)
- `compiler/src/ir/hir_to_mir.rs` — collection and injection in `lower_inline_code()`

**Implemented Metadata:**
- [x] `@:frameworks(["Accelerate"])` — load macOS frameworks, add SDK header paths
- [x] `@:cInclude(["/opt/homebrew/include"])` — add include search directories
- [x] `@:cSource(["vendor/stb_image.c"])` — compile additional C source files into TCC context
- [x] `@:clib(["sqlite3"])` — discover and load libraries via `pkg-config` (cross-platform)

**`@:clib` pkg-config discovery:**
- Runs `pkg-config --cflags <name>` → extracts `-I` paths → `tcc_add_include_path()`
- Runs `pkg-config --libs <name>` → extracts `-L`/`-l` → `dlopen()` libraries
- Cross-platform: macOS (brew), Linux (apt), Windows/MSYS2 (pacman)

**E2E Tests:** Tests 17-19 in `compiler/examples/test_cstruct_e2e.rs` (frameworks, function-level frameworks, raylib raymath)

### 13.4 @:cstruct C-Compatible Memory Layout

**Status:** 🟢 Complete

**Related Files:**
- `compiler/src/ir/hir_to_mir.rs` — cstruct layout computation, cdef generation, auto-injection
- `compiler/src/tast/ast_lowering.rs` — `@:cstruct` metadata extraction

**Features:**
- [x] `@:cstruct` metadata — flat C-compatible memory layout (no object header)
- [x] Field read/write via byte offsets
- [x] `cdef()` static method — returns C typedef string for explicit use
- [x] Auto-injection of module-local `@:cstruct` typedefs into `__c__()` contexts
- [x] Dependency resolution — nested cstructs included in topological order
- [x] Supported field types: Int (long), Float (double), Bool (int), Ptr<T> (void*/T*), Usize (size_t), CString (char*)

**E2E Tests:** Tests 1-12 in `compiler/examples/test_cstruct_e2e.rs`

### 13.5 System Path Discovery

**Status:** 🟢 Complete

- [x] macOS: auto-discovers CommandLineTools/Xcode SDK via candidate paths, adds `<SDK>/usr/include`
- [x] macOS: framework headers from `<SDK>/System/Library/Frameworks/<Name>.framework/Headers/`
- [x] Linux: probes `/usr/include`, `/usr/local/include`
- [x] TCC lib path set to vendored `compiler/vendor/tinycc/` (includes `tccdefs.h`)
- [x] `-nostdlib` flag prevents TCC from loading macOS `.tbd` stubs (incompatible with TCC linker)
- [x] Symbol resolution via `dlsym(RTLD_DEFAULT)` during `tcc_relocate`

### 13.6 CString Extern Abstract

**Status:** 🟢 Complete

- [x] `CString.from(s)` — allocate null-terminated copy from Haxe String
- [x] `cs.toHaxeString()` — convert back to Haxe String
- [x] `cs.raw()` — get raw `char*` address as Int
- [x] `CString.fromRaw(addr)` — wrap existing `char*`
- [x] `cs.free()` — free the buffer
- [x] CString fields in `@:cstruct` map to `char*` in C typedef

### 13.7 Remaining / Future Enhancements

- [ ] Source caching: hash C source to avoid recompiling identical `__c__()` blocks
- [ ] `@:unsafe` metadata warning when using `__c__` (currently allowed without annotation)
- [ ] CC.addClib() explicit API method (currently `@:clib` metadata only)
- [ ] Windows: test MSYS2/MinGW pkg-config integration end-to-end

### Test Summary

| Test File | Tests | Status |
|-----------|-------|--------|
| `test_cstruct_e2e.rs` | 19 | ✅ 19/19 PASS |
| `test_cc_e2e.rs` | 6 | ✅ 6/6 PASS |
| `test_systems_e2e.rs` | 8 | ✅ 8/8 PASS |

---

## 14. SIMD & Tensor / GPU Compute 🟡

**Priority:** High
**Complexity:** Very High
**Dependencies:** SIMD4f (complete), Plugin system (complete)

### 14.1 SIMD4f ✅ COMPLETE (2026-01-31)

- [x] 128-bit SIMD vector (4×f32) as @:coreType abstract
- [x] Tuple literal syntax: `var v:SIMD4f = (1.0, 2.0, 3.0, 4.0)`
- [x] @:from Array literal with heap allocation warning
- [x] Zero-cost operators: +, -, *, / via VectorBinOp
- [x] Math ops: sqrt, abs, neg, min, max, ceil, floor, round
- [x] Compound ops: clamp, lerp, normalize, cross3, distance, len
- [x] Cranelift + LLVM backend support
- [x] 16 E2E tests passing

### 14.2 rayzor.ds.Tensor (CPU) 🔴

- [ ] Tensor type with shape/strides/dtype (extern class, runtime in Rust)
- [ ] DType enum (F32, F16, BF16, I32, I8, U8)
- [ ] Construction: zeros, ones, full, fromArray, rand
- [ ] View ops: reshape, transpose, permute, slice (no-copy via strides)
- [ ] Elementwise ops: add, sub, mul, div, exp, log, sqrt
- [ ] Reductions: sum, mean, max, min
- [ ] Linear algebra: matmul, dot
- [ ] Activations: relu, gelu, silu, softmax
- [ ] Normalization: layerNorm, rmsNorm
- [ ] SIMD4f vectorized CPU paths for f32 ops

### 14.3 rayzor-gpu Plugin 🟡

GPU compute is a **packaged plugin** (not core stdlib) — keeps core lean, optional dependency.
Strategy: Tinygrad-style source code emission (Kernel IR → text per backend → runtime compile).

**Phase 1 ✅ Metal device + buffers + NativePlugin**
- [x] Metal device init (MTLDevice + MTLCommandQueue)
- [x] GPU buffer management (create from Tensor, alloc, readback to Tensor, free)
- [x] NativePlugin architecture (`declare_native_methods!` macro) — no compiler core changes
- [x] Haxe API: `GPUCompute.create()`, `.createBuffer()`, `.toTensor()`, `.freeBuffer()`

**Phase 2 ✅ Kernel IR + MSL codegen**
- [x] KernelOp IR enum (Add, Sub, Mul, Div, Neg, Abs, Sqrt, Exp, Log, Relu)
- [x] MSL source code generation (binary + unary elementwise kernels)
- [x] Metal shader compilation (MSL → MTLComputePipelineState)
- [x] Compute command dispatch (threadgroup sizing, buffer binding)
- [x] KernelCache: HashMap<(KernelOp, dtype), CompiledKernel>

**Phase 3 ✅ Elementwise ops API**
- [x] Binary ops: gpu.add/sub/mul/div(bufA, bufB) → bufResult
- [x] Unary ops: gpu.neg/abs/sqrt/exp/log/relu(buf) → bufResult
- [x] 15 GPU tests passing (codegen + Metal integration + ops)

**Phase 4 — Reductions + Matmul**
- [ ] Tree-reduction kernels (sum, mean, max, min) with threadgroup shared memory
- [ ] Tiled 16x16 shared-memory matmul
- [ ] Dot product

**Phase 5 — Compute Data Structures (@:gpuStruct)**
- [ ] `@:gpuStruct` annotation (GPU-aligned flat structs, 4-byte floats)
- [ ] Structured buffer create/alloc/read
- [ ] MSL/CUDA typedef generation via `gpuDef()`

**Phase 6 — Kernel Fusion**
- [ ] Lazy evaluation DAG for elementwise op chains
- [ ] Fused kernel codegen (e.g., `a.add(b).mul(c).relu()` → single kernel)

**Phase 7 — Additional Backends**
- [ ] CUDA backend (NVRTC) — NVIDIA GPUs
- [ ] WebGPU backend (wgpu) — cross-platform
- [ ] Vulkan backend (SPIR-V) — Windows/Linux/Android
- [ ] OpenCL backend — cross-platform legacy

### 14.5 Operator Overloading for GPU/Tensor Types 🔴

- [ ] Exercise existing `@:op` annotations on Tensor (add E2E tests using `a + b` syntax)
- [ ] Add `@:op` overloading to GpuBuffer (requires ctx back-pointer in buffer struct)
- [ ] Verify abstract type `@:op` support works end-to-end (currently only extern class tested)

### 14.4 Interpreter SIMD Correctness 🔴

- [ ] Integrate `wide` crate for real SIMD in interpreter (currently returns void)
- [ ] Or: force-promote SIMD functions to skip Tier 0
- [ ] TCC Linker SIMD gap on Linux (final tier lacks SIMD)

---

## 15. AOT Compilation & Static Linking 🟢

**Priority:** High
**Complexity:** High
**Dependencies:** LLVM backend (complete), Runtime staticlib (complete), Tree-shaking (complete)

### 15.1 AOT Compiler Driver

**Status:** 🟢 Complete (2026-02-01)

**Related Files:**
- `compiler/src/codegen/aot_compiler.rs` — AOT compilation pipeline
- `compiler/src/codegen/llvm_aot_backend.rs` — AOT-specific LLVM operations (free functions, separate from JIT)
- `compiler/src/bin/rayzor_build.rs` — CLI binary
- `compiler/src/codegen/llvm_jit_backend.rs` — Shared LLVM codegen (aot_mode flag)

**Architecture:**
```
Haxe Source (.hx) → MIR → MIR Optimize (O2 cap) → Tree-shake → LLVM IR → LLVM O3 → Object File (.o) → Native Executable
```

**Tasks:**
- [x] `AotCompiler` struct with compile pipeline (parse → optimize → tree-shake → LLVM → link)
- [x] Generate LLVM IR `main()` wrapper that calls Haxe entry point
- [x] `llvm_aot_backend.rs` with free functions for AOT operations (no JIT regression)
- [x] `compile_to_object_file()` with configurable target triple
- [x] Support all LLVM target triples for cross-compilation (`init_llvm_aot` → `Target::initialize_all`)
- [x] Platform-specific linker invocation (macOS/Linux/Windows)
- [x] Runtime library discovery (`librayzor_runtime.a`) — 4 search paths
- [x] Multiple output formats: exe, obj, llvm-ir, llvm-bc, asm

### 15.2 CLI Interface (`rayzor-build`)

**Status:** 🟢 Complete (2026-02-01)

**Usage:**
```bash
rayzor-build -O2 -o hello hello.hx                              # Host target
rayzor-build --target aarch64-unknown-linux-gnu -o hello hello.hx # Cross-compile
rayzor-build --emit llvm-ir -o hello.ll hello.hx                  # Emit IR
rayzor-build --emit asm -o hello.s hello.hx                       # Emit assembly
rayzor-build -O3 -v -o hello hello.hx                            # Verbose O3
```

**Tasks:**
- [x] Argument parsing (--target, --emit, -O, --runtime-dir, --linker, --sysroot, --strip, -v)
- [x] Default output naming
- [x] Verbose compilation progress output with timing
- [x] Error messages for missing runtime / linker

### 15.3 Static Linking

**Status:** 🟢 Complete (2026-02-01)

**Design:** Link `librayzor_runtime.a` (Rust staticlib) directly into native binary. No shared library dependencies beyond system libc/libm/libpthread.

**Tasks:**
- [x] macOS linking (clang + frameworks: CoreFoundation, Security)
- [x] Linux linking (clang/gcc + -lc -lm -lpthread -ldl)
- [x] Windows linking (kernel32.lib, ws2_32.lib, userenv.lib, bcrypt.lib)
- [ ] Fully static linking with musl
- [x] Strip debug symbols option (--strip)

### 15.4 Cross-Compilation

**Status:** 🟡 Infrastructure Complete, Testing Needed

**Tasks:**
- [x] Configurable target triple in LLVM codegen
- [x] Sysroot support for cross-compilation (--sysroot flag)
- [ ] Runtime library for target arch (build-on-demand or user-provided)
- [ ] CI testing for cross-compilation (x86_64 → aarch64, etc.)

### 15.5 LLVM Codegen Performance Optimizations

**Status:** 🟢 Complete (2026-02-01)

Two optimizations applied to the shared LLVM codegen (benefits both JIT and AOT):

**1. Math Intrinsics:** Known runtime math functions (`haxe_math_sqrt`, `haxe_math_abs`, `haxe_math_floor`, `haxe_math_ceil`, `haxe_math_round`, `haxe_math_sin`, `haxe_math_cos`, `haxe_math_exp`, `haxe_math_log`, `haxe_math_pow`) replaced with inline LLVM intrinsic wrappers (e.g. `@llvm.sqrt.f64` → single `fsqrt` instruction). Wrappers use `alwaysinline` + `Internal` linkage.

**2. Stack Allocation:** Fixed-size `Alloc` instructions use `alloca` (stack) instead of `malloc` (heap). `Free` instructions become no-ops. Profiling showed **89% of mandelbrot time was in malloc/free**. Dynamic-count allocations still use malloc.

**Benchmark Results (mandelbrot, 875×500, 1000 max iterations):**

| Target | Before (2026-01-31) | After (2026-02-01) | Speedup |
|--------|---------------------|---------------------|---------|
| rayzor-llvm (JIT) | 893ms | **343ms** | **2.6x** |
| rayzor-tiered (JIT) | 874ms | **153ms** | **5.7x** |
| rayzor-precompiled-tiered | 914ms | **154ms** | **5.9x** |
| AOT native binary | 870ms | **155ms** | **5.6x** |
| rayzor-cranelift | 2840ms | 2869ms | — (no LLVM) |

---

## 16. Haxe Language Feature Gap Analysis 🔴

**Priority:** Critical — these gaps block real-world Haxe code from compiling
**Last Audit:** 2026-02-08 (cross-referenced against https://haxe.org/manual/introduction.html)

### Gap Priority Matrix

Features are ranked by **impact** (how much real Haxe code they block) and **complexity** (implementation effort). P0 = must-have for any non-trivial program, P1 = needed for idiomatic Haxe, P2 = advanced/nice-to-have.

| # | Feature | Priority | Complexity | Status | Blocks |
|---|---------|----------|------------|--------|--------|
| 1 | Enum variants + pattern matching (ADTs) | P0 | High | 🟢 Complete | switch, Option, Result |
| 2 | Interface dispatch (vtables) | P0 | High | 🟢 Complete | polymorphism, stdlib |
| 3 | try/catch exception handling | P0 | High | 🟢 Complete | error handling |
| 4 | Closures as first-class values | P0 | High | 🟢 Complete | callbacks, HOFs |
| 5 | Array.map/filter/sort (higher-order) | P0 | Medium | 🟢 Complete | functional patterns |
| 6 | String interpolation | P0 | Low | 🟢 Complete | basic string formatting |
| 7 | for-in range (`0...n`) | P0 | Low | 🟡 Partial | basic loops |
| 8 | Static extensions (`using`) | P1 | Medium | 🟢 Complete | idiomatic Haxe |
| 9 | Safe cast (`cast(expr, Type)`) | P1 | Medium | 🔴 Not started | type-safe downcasting |
| 10 | Generics instantiation end-to-end | P1 | High | 🟢 Complete | generic classes/functions |
| 11 | Property get/set dispatch | P1 | Medium | 🟡 Mostly done | encapsulation |
| 12 | EReg (regex runtime) | P1 | Medium | 🟢 Complete | text processing |
| 13 | Enum methods + statics | P1 | Medium | 🔴 Not started | rich enums |
| 14 | Abstract types (operator overloading) | P1 | High | 🟡 Partial | custom types |
| 15 | Dynamic type operations | P1 | Medium | 🟡 Partial | interop, JSON |
| 16 | Type parameters on functions | P1 | Medium | 🟢 Complete | generic functions |
| 17 | Null safety (`Null<T>`) | P2 | Medium | 🔴 Not started | null checks |
| 18 | Structural subtyping | P2 | Medium | 🔴 Not started | structural interfaces |
| 19 | `@:forward` on abstracts | P2 | Medium | 🔴 Not started | delegation |
| 20 | Macros (compile-time) | P2 | Very High | 🔴 Not started | metaprogramming |
| 21 | Map literal syntax | P2 | Low | 🔴 Not started | `["key" => val]` |
| 22 | Array comprehension | P2 | Medium | 🔴 Not started | `[for (x in arr) x*2]` |
| 23 | `Std.is()` / `Std.downcast()` (RTTI) | P2 | Medium | 🔴 Not started | runtime type checks |

---

### 16.1 Enum Variants + Pattern Matching (ADTs) 🟢

**Priority:** P0 — Critical
**Status:** ✅ Complete (2026-02-08)

**What Works:**
- Enum declaration parsing and TAST lowering
- Simple discriminant enums (`Color.Red` = integer)
- Boxed parameterized variants (`Option.Some(42)` = heap [tag][value])
- Enum RTTI for trace (`trace(Color.Red)` → "Red")
- `switch` on enum values with `case Some(v):` destructuring
- Wildcard `_` and variable binding in patterns
- `default` / catch-all case
- Or-patterns (`case A | B:`)
- Multiple patterns per case
- Bitcast i64→Ptr for boxed enum scrutinee in pattern tests

**Not Yet Implemented:**

- [ ] Guard expressions in match arms (`case v if v > 0:`)
- [ ] Exhaustiveness checking (warn on missing cases)
- [ ] `EnumValue` API (`Type.enumIndex()`, `Type.enumParameters()`)
- [ ] Nested pattern matching (`case Pair(Some(x), _):`)

### 16.2 Interface Dispatch (Vtables) 🟢

**Priority:** P0 — Critical
**Status:** ✅ Complete (2026-02-08)

**What Works:**
- Fat pointer vtable: `{obj_ptr: i64, fn_ptr_0: i64, ...}` per interface assignment
- `interface_method_names` + `interface_vtables` maps built during type registration
- Two-pass type registration (interfaces first, then classes) for correct ordering
- `wrap_in_interface_fat_ptr()` allocates and populates fat pointer at assignment
- Interface dispatch in Variable callee path via `CallIndirect`
- `build_function_ref()` for vtable fn_ptr construction
- Works for Let bindings and Assign statements

**Not Yet Implemented:**

- [ ] Multiple interface implementation (`class Foo implements Bar implements Baz`)
- [ ] Interface inheritance (`interface A extends B`)
- [ ] `Std.is(obj, IMyInterface)` runtime check
- [ ] Fat pointer lifecycle management (free on scope exit)

### 16.3 Try/Catch Exception Handling 🟢

**Priority:** P0 — Critical
**Status:** ✅ Complete (2026-02-08)

**Implementation:** setjmp/longjmp with thread-local handler stack.

**What Works:**
- `runtime/src/exception.rs`: Thread-local `ExceptionState` with handler stack
- `rayzor_exception_push_handler()`, `rayzor_exception_pop_handler()`, `rayzor_throw()`, `rayzor_get_exception()`
- Expression-level `HirExprKind::TryCatch` handler with full setjmp/longjmp pattern
- Statement-level `lower_try_catch()` also implemented
- `throw expr` → `CallDirect` to `rayzor_throw()` (no backend changes needed)
- Catch block with `Dynamic` type matching
- Normal control flow preserved (try without throw skips catch)

**Not Yet Implemented:**

- [ ] Typed catch matching (`catch (e:String)` vs `catch (e:Int)`)
- [ ] Multiple catch blocks with type discrimination
- [ ] Finally block execution
- [ ] Exception propagation through uncaught functions (cross-function unwinding)
- [ ] `haxe.Exception` base class
- [ ] Stack trace capture on throw

### 16.4 Closures as First-Class Values 🟢

**Priority:** P0 — Critical
**Status:** ✅ Complete (2026-02-08)

**What Works:**
- Lambda parsing (`() -> expr`, `(x) -> expr`)
- Store closure in variable (`var f = (x) -> x * 2;`)
- Call stored closure (`f(10)`) via `CallIndirect`
- Closure environment capture (env_ptr always first param, even without captures)
- Closure struct: `{fn_ptr: i64, env_ptr: i64}` — 16 bytes on heap
- Cranelift backend: `MakeClosure`, `ClosureFunc`, `ClosureEnv`, `CallIndirect`
- LLVM backend: Full closure support (MakeClosure, ClosureFunc, ClosureEnv, CallIndirect)
- Indirect call parameter type inference from callee's function type

**Not Yet Implemented:**

- [ ] Pass closure as function argument (`arr.map((x) -> x * 2)`)
- [ ] Partial application / bind
- [ ] `Reflect.isFunction()` support

### 16.5 Higher-Order Array Methods 🟢

**Priority:** P0 — Critical (depends on 16.4 Closures)
**Status:** ✅ Complete (2026-02-08)

**What Works:**

- [x] `arr.map(f)` — transform elements
- [x] `arr.filter(f)` — select elements
- [x] `arr.sort(f)` — sort with comparator

**Not Yet Implemented:**

- [ ] `arr.indexOf(v)` — find element
- [ ] `arr.contains(v)` — check membership
- [ ] `arr.iterator()` — for-in iteration
- [ ] `arr.join(sep)` — string join
- [ ] `arr.concat(other)` — concatenate
- [ ] `arr.copy()` — shallow copy
- [ ] `arr.splice(pos, len)` — remove range
- [ ] `arr.slice(pos, end)` — sub-array
- [ ] `arr.reverse()` — reverse in-place
- [ ] `arr.remove(v)` — remove first occurrence
- [ ] `arr.insert(pos, v)` — insert at position

### 16.6 String Interpolation 🟢

**Priority:** P0 — Low complexity, high impact
**Status:** ✅ Complete (already implemented — parser, AST, TAST, HIR desugaring all work)

**What Works:**
- Single-quote string interpolation: `'Hello $name, you are ${age + 1} years old'`
- Simple variable interpolation: `$varName`
- Expression interpolation: `${expr}`
- Desugared to string concatenation during AST lowering

### 16.7 For-in Range Iteration 🟡

**Priority:** P0
**Current State:** `for (v in iterable)` works for arrays. `0...n` range syntax is partially supported.

**What's Missing:**
- [ ] `IntIterator` (`0...10` creates IntIterator with hasNext/next)
- [ ] `for (i in 0...10)` full support
- [ ] Custom iterator protocol (`hasNext()` + `next()`)
- [ ] `do...while` loop
- [ ] Labeled break/continue (`break label`)

### 16.8 Static Extensions (`using`) 🟢

**Priority:** P1
**Status:** ✅ Complete (2026-02-08)

**What Works:**
- `using MyTools;` imports static extension methods at file level
- Method resolution: `x.myMethod()` rewrites to `MyTools.myMethod(x)` when `MyTools` has matching static method
- Multiple `using` imports in scope
- Extension methods on basic types (Int, String, Array)
- Priority: local methods > extensions (placeholder check triggers extension lookup)
- Multi-argument extension methods (`x.add(3)` → `IntTools.add(x, 3)`)

**Implementation:** Already existed in ast_lowering.rs — `lower_using()` registers using modules, `find_static_extension_method()` resolves calls, method call desugaring converts to `StaticMethodCall` with receiver prepended as first argument.

### 16.9 Safe Cast 🔴

**Priority:** P1
**Current State:** `cast(expr, Type)` syntax not implemented. Unsafe `cast expr` may partially work.

**What's Missing:**
- [ ] `cast(expr, Type)` — returns null on failure (safe cast)
- [ ] `cast expr` — unchecked cast (unsafe, for FFI/interop)
- [ ] Runtime type check before cast
- [ ] Integration with RTTI system

### 16.10 Abstract Types 🟡

**Priority:** P1
**Current State:** Parser handles abstract declarations. `@:coreType` extern abstracts work (SIMD4f, CString). Operator overloading via `@:op` partially works.

**What's Missing:**
- [ ] User-defined abstract types with underlying type (`abstract MyInt(Int)`)
- [ ] Implicit conversions (`@:from`, `@:to`)
- [ ] `@:op(A + B)` on non-extern abstracts
- [ ] Abstract enum (`abstract Color(Int) { var Red = 0; var Blue = 1; }`)
- [ ] `@:forward` — delegate methods to underlying type
- [ ] `@:enum` abstracts
- [ ] `this` in abstract methods refers to underlying value

### 16.11 Dynamic Type 🟡

**Priority:** P1
**Current State:** `Dynamic` type exists in type system. Boxing/unboxing works for basic types. Anonymous objects use Dynamic for field types.

**What's Missing:**
- [ ] `Dynamic` field access (`obj.anyField` without compile-time check)
- [ ] `Dynamic` method calls
- [ ] `Dynamic` arithmetic operations
- [ ] `Dynamic` → typed coercion at assignment
- [ ] Reflect.field/setField on Dynamic objects
- [ ] JSON parsing returns Dynamic

### 16.12 EReg (Regular Expressions) 🟢

**Priority:** P1
**Status:** Complete

**Implemented:**

- [x] `~/pattern/flags` literal syntax (parser → TAST → HIR → MIR)
- [x] `new EReg(pattern, flags)` constructor
- [x] `match()`, `matched()`, `matchedLeft()`, `matchedRight()` instance methods
- [x] `replace()` with global/non-global modes
- [x] `split()` — non-global splits at first match
- [x] `matchSub()` with optional length param (2-arg and 3-arg overloads)
- [x] `EReg.escape()` static method
- [x] Regex flags: `g` (global), `i` (case-insensitive), `m` (multiline), `s` (dotall)
- [x] Runtime backed by Rust `regex` crate (runtime/src/ereg.rs)
- [x] Regex literal properly typed as EReg class for method resolution

**Deferred:**

- `matchedPos()` — returns anonymous object `{pos:Int, len:Int}`, needs MIR wrapper
- `map()` — needs passing Haxe closure to runtime

### 16.13 Enum Methods and Statics 🔴

**Priority:** P1
**Current State:** Enums are data-only. No methods or static members.

**What's Missing:**
- [ ] Methods on enum types
- [ ] Static methods on enums
- [ ] `Type.getEnumConstructs()` — list variant names
- [ ] `Type.createEnum()` — create variant by name/index

### 16.14 Null Safety 🔴

**Priority:** P2
**Current State:** No null safety enforcement. Null is a valid value for any reference type.

**What's Missing:**
- [ ] `Null<T>` wrapper type
- [ ] Null-check operator `?.` (optional chaining)
- [ ] Null coalescing `??`
- [ ] Compile-time null flow analysis
- [ ] `@:notNull` metadata

### 16.15 Structural Subtyping 🔴

**Priority:** P2
**Current State:** Typedef structure types partially work. Anonymous object shapes work.

**What's Missing:**
- [ ] Structural type compatibility (pass `{x:Int, y:Int, z:Int}` where `{x:Int, y:Int}` expected)
- [ ] Structural interfaces (any object with matching fields satisfies the type)
- [ ] Compile-time structural matching

### 16.16 Map Literal Syntax 🔴

**Priority:** P2
**Current State:** IntMap/StringMap exist as runtime types. No literal syntax.

**What's Missing:**
- [ ] `["key1" => val1, "key2" => val2]` map literal syntax
- [ ] Type inference for map key/value types
- [ ] `for (key => value in map)` iteration

### 16.17 Array Comprehension 🔴

**Priority:** P2
**Current State:** Not implemented.

**What's Missing:**
- [ ] `[for (x in arr) x * 2]` — array comprehension
- [ ] `[for (x in arr) if (x > 0) x]` — filtered comprehension
- [ ] Nested comprehensions

### 16.18 RTTI (Runtime Type Information) 🟡

**Priority:** P2
**Current State:** Basic enum RTTI exists. Type IDs assigned. Anonymous object shapes registered.

**What's Missing:**
- [ ] `Std.is(value, Type)` — runtime type checking
- [ ] `Std.downcast(value, Type)` — safe downcast
- [ ] `Type.getClass(obj)` — get class of object
- [ ] `Type.getClassName(cls)` — get class name as string
- [ ] `Type.getInstanceFields(cls)` — list fields
- [ ] `Type.getSuperClass(cls)` — class hierarchy
- [ ] `Type.typeof(value)` — get ValueType enum
- [ ] Full class metadata at runtime

### 16.19 Macros (Compile-Time) 🔴

**Priority:** P2 — Very high complexity, low near-term priority
**Current State:** Macro parser infrastructure exists (113 unit tests). No execution.

**What's Missing:**
- [ ] Compile-time expression evaluation
- [ ] `macro` keyword functions
- [ ] Expression reification (`macro $v`, `macro $a{expr}`)
- [ ] `Context` and `Compiler` APIs
- [ ] Build macros (`@:build`, `@:autoBuild`)
- [ ] `#if` / `#else` conditional compilation (preprocessor)

---

### Updated Implementation Priority Order (2026-02-08)

#### Tier 1: Language Fundamentals (blocks real programs) ✅ COMPLETE

1. ✅ **Enum variants + pattern matching** (16.1) — unlocks Option/Result, switch expressions
2. ✅ **Closures as first-class values** (16.4) — unlocks callbacks, HOFs, Array.map
3. ✅ **String interpolation** (16.6) — already implemented
4. ✅ **try/catch exception handling** (16.3) — setjmp/longjmp based
5. ✅ **Interface dispatch** (16.2) — fat pointer vtables

#### Tier 2: Idiomatic Haxe (blocks Haxe-style code)
6. ✅ **Higher-order Array methods** (16.5) — map/filter/sort with closure callbacks
7. ✅ **Static extensions** (16.8) — `using` keyword
8. **Generics end-to-end** (16.10, existing 1.x) — unblock generic containers
9. ✅ **EReg** (16.12) — regex support (match, replace, split, escape, regex literals)
10. **Abstract types** (16.10) — user-defined abstracts

#### Tier 3: Completeness (polish and compatibility)
11. **Safe cast** (16.9)
12. **Dynamic type ops** (16.11)
13. **Null safety** (16.14)
14. **RTTI** (16.18)
15. **Map literals** (16.16)
16. **Array comprehension** (16.17)
17. **Macros** (16.19)

---

## Known Issues

### Deref Coercion for Wrapper Types
**Status:** Not Implemented
**Affected Types:** Arc<T>, MutexGuard<T>, and similar wrapper types

Wrapper types like `Arc<T>` and `MutexGuard<T>` were designed to transparently forward method/field access to their inner type (similar to Rust's `Deref` trait). Currently, users must explicitly call `.get()` to access the inner value.

**Workaround:** Use explicit `.get()` calls:
```haxe
var arc = Arc.init(42);
var value = arc.get();  // Must explicitly call .get()
// Instead of: var value = arc;  // Would implicitly deref
```

**Future Implementation:**
- Detect method/field access on wrapper types
- Automatically insert `.get()` calls during MIR lowering
- Handle nested wrappers (e.g., `Arc<Mutex<T>>`)

### @:native Metadata Ignored on Extern Abstract Methods
**Status:** Bug (workaround in place)
**Affected Types:** `rayzor.CString`, `rayzor.Usize`, `rayzor.Ptr`, and any stdlib extern abstract

`@:native` metadata on extern abstract method declarations (e.g., `@:native("to_haxe_string") public function toHaxeString():String`) is not processed during stdlib BLADE cache loading. The `symbol.native_name` field remains `None` for all stdlib extern abstract methods.

This means `get_stdlib_runtime_info` cannot use the declared native name to look up the correct runtime mapping — it must use the Haxe method name (`symbol.name`) instead.

**Workaround:** Runtime mapping keys in `runtime_mapping.rs` use Haxe method names (e.g., `"toHaxeString"`) instead of the `@:native` names (e.g., `"to_haxe_string"`). This works but defeats the purpose of `@:native`.

**Root Cause:** Stdlib types are loaded via the BLADE cache path, which deserializes pre-built symbols. The `@:native` metadata processing added in `lower_function_from_field` (ast_lowering.rs) only runs for user-defined types, not for stdlib types loaded from cache.

**Fix Required:**
- Process `@:native` metadata during BLADE cache deserialization, or
- Process `@:native` on extern abstract methods during stdlib loading (post-cache), or
- Store `native_name` in the BLADE cache format itself

### String Concatenation with Trace
**Status:** ✅ FIXED (2026-01-30)
**Issue:** ~~Using string concatenation inside trace causes misaligned pointer dereference~~ — Resolved by using MIR register types instead of HIR types for string concat operands, and changing `int_to_string` to accept I64 directly.

```haxe
// Now works:
trace("Length: " + v.length());  // ✅
trace("The point is: " + p);    // ✅ (calls toString())
```

---

## Technical Debt

- [ ] Remove DEBUG log statements cleanly (without breaking code)
- [ ] Consolidate error handling (CompilationError vs custom errors)
- [ ] Reduce warnings in codebase
- [ ] Improve type inference completeness
- [ ] Refactor HIR/MIR distinction (clarify naming)
- [ ] Performance profiling and bottleneck identification
- [x] Fix test_full_pipeline_cranelift.rs API usage ✅
- [x] Fix Alloc instruction LICM hoisting ✅ (2026-01-28)
- [x] Fix break/continue drop scope corruption ✅ (2026-01-28)

---

## Notes

- **Generics** are foundational for async (Promise<T>) and concurrency (Channel<T>)
- **Send/Sync** require derived trait infrastructure to be complete
- **Async state machines** build on generics and memory safety
- Implementation should follow dependency order to avoid rework

**Last Updated:** 2026-02-07 (Haxe Language Feature Gap Analysis)

## Recent Progress (Session 2026-01-31 - Inline C / TinyCC Runtime API)

**TinyCC Runtime API:** ✅ Complete

- ✅ **`rayzor.runtime.CC` extern class** — 13 methods (create, compile, relocate, getSymbol, addSymbol, addFramework, addIncludePath, addFile, delete, call0-call3)
- ✅ **`untyped __c__()` inline C syntax** — auto-manages TCC lifecycle, argument passing via `{0}`/`{1}` placeholders, return values, module-local `@:cstruct` auto-injection
- ✅ **`@:cstruct` metadata** — C-compatible memory layout, `cdef()` static method, nested struct dependency resolution, field types: Int/Float/Bool/Ptr/Usize/CString
- ✅ **`rayzor.CString` extern abstract** — from/toHaxeString/raw/fromRaw/free, maps to `char*` in cstruct
- ✅ **System path discovery** — macOS SDK auto-detection, Linux `/usr/include`, TCC vendored headers
- ✅ **`@:frameworks(["Accelerate"])`** — load macOS frameworks + SDK headers into TCC context (class or function level)
- ✅ **`@:cInclude(["/path"])`** — add include search paths (class or function level)
- ✅ **`@:cSource(["file.c"])`** — compile additional C sources into TCC context (class or function level)
- ✅ **`@:clib(["sqlite3"])`** — pkg-config discovery for cross-platform library loading (class or function level)
- ✅ **TCC error handling** — compile/relocate/symbol errors trigger panics (catchable via try-catch)
- ✅ **Raylib raymath E2E test** — `@:cInclude` with header-only raylib math library (Vector2Length, Clamp, Lerp)
- ✅ **CC_FEATURES.md** — comprehensive documentation of all CC/TCC features

**Runtime Functions:** 16 Rust functions in `runtime/src/tinycc_runtime.rs`, all registered in `plugin_impl.rs`

**E2E Tests:**
- ✅ test_cstruct_e2e: **19/19 PASS** (cstruct, CString, inline C, frameworks, cInclude, raylib)
- ✅ test_cc_e2e: **6/6 PASS** (explicit CC API)
- ✅ test_systems_e2e: **8/8 PASS** (Box, Ptr, Ref, Usize, Arc)

**Commits:** c0d3597 → 147d557 (8 commits across sessions)

---

## Recent Progress (Session 2026-01-30b - String Concat & Vec Fixes)

**String Concatenation Fix:** ✅ Complete

- ✅ **`int_to_string` accepts I64 directly** — removed redundant I32→I64 cast inside wrapper, matching `haxe_string_from_int(i64)` runtime signature
- ✅ **MIR register types for string concat** — HIR types from generic methods (e.g. `Vec<Int>.length()`) resolve as `Ptr(Void)`; now uses `builder.get_register_type()` which reflects correct runtime mapping types
- ✅ **Cranelift BitCast I32↔I64** — added sextend/ireduce support in BitCast handler
- ✅ **Vec push I32→I64 sign-extend** — array literal push uses `build_cast` instead of `build_bitcast`
- ✅ **String concat ABI** — renamed `haxe_string_concat` to `haxe_string_concat_sret` to avoid symbol conflict

**E2E Test Results:**

- ✅ test_vec_e2e: **5/5 PASS** (was 1/2 FAIL)
- ✅ test_tostring_concat: **3/3 PASS**
- ✅ test_core_types_e2e: **25/25 PASS**

**Commit:** 0f9136d

---

## Recent Progress (Session 2026-01-30 - TCC Linker & Benchmark CI)

**TCC In-Process Linker Integration:** ✅ Complete
- ✅ **TCC linker replaces system linker + dlopen** for LLVM AOT object files on Linux
  - Vendored TinyCC source via `cc` crate in `build.rs` (no system TCC install needed)
  - Feature-gated behind `tcc-linker` cargo feature
  - `-nostdlib` to avoid libc/libtcc1 dependency; manually registers libc symbols (malloc, realloc, calloc, free, memcpy, memset, memmove, abort)
  - ELF object files loaded via `tcc_add_file()`, relocated in-memory via `tcc_relocate()`
  - Function pointers extracted via `tcc_get_symbol()`
  - **Files:** `compiler/src/codegen/tcc_linker.rs` (new), `compiler/build.rs`, `compiler/Cargo.toml`

- ✅ **Fixed TCC relocation errors on CI**
  - `library 'c' not found` → fixed with `-nostdlib`
  - `undefined symbol 'realloc'` → fixed by registering libc symbols via `tcc_add_symbol`
  - `R_X86_64_32[S] out of range` → fixed by keeping LLVM `RelocMode::PIC` (TCC allocates at arbitrary addresses)

- ✅ **Tiered backend LLVM upgrade working on CI**
  - mandelbrot_class_small: tiered 2.17x faster than Cranelift-only
  - nbody: tiered 1.08x faster than Cranelift-only
  - LLVM tier promotion fires correctly during benchmark execution

**Benchmark CI Infrastructure:**
- ✅ **GitHub Actions benchmark workflow** (`.github/workflows/benchmarks.yml`)
  - Runs on push to main and weekly schedule
  - Stores results as JSON artifacts with system info
  - HTML chart generation with historical comparison
- ✅ **System info in benchmark output** — OS, arch, CPU cores, RAM, hostname
  - Displays in both console output and HTML chart page
  - Preserved across JSON result merges

**E2E Test Results (verified 2026-01-30):**
- ✅ test_core_types_e2e: **25/25 PASS** (was 20/25 in backlog)
- ✅ test_rayzor_stdlib_e2e: **9/9 PASS**
- ✅ test_enum_trace: **3/3 PASS**
- ✅ test_enum_resolution: **2/2 PASS**
- ✅ test_enum_option_result: **4/4 PASS**
- ✅ test_vec_e2e: **5/5 PASS** (was 1/2 FAIL — fixed Vec bitcast I32→I64 + string concat)

**Remaining Issues:**
- ✅ ~~Vec bitcast error (I32→I64)~~ — Fixed with sextend/ireduce in Cranelift BitCast handler
- ✅ ~~String concatenation ABI mismatch~~ — Fixed by using MIR register types + renamed sret variant

---

## Recent Progress (Session 2026-01-28 - Enum Trace)

**Enum Trace & RTTI:**
- ✅ **Enum trace with RTTI variant name lookup** - Simple enums: `trace(Color.Red)` → "Red"
  - Registered enum types in runtime type system with variant names
  - `haxe_trace_enum(type_id, discriminant)` looks up variant name from RTTI
  - **Commit:** d4ea44c

- ✅ **Boxed enum representation for parameterized variants** - `trace(MyResult.Ok(42))` → "Ok(42)"
  - Heap-allocated enums with layout: `[tag:i32][pad:i32][field0:i64][field1:i64]...`
  - Simple enums (no params) remain as plain i64 discriminants
  - GEP element size fix in Cranelift backend for correct field offset calculation
  - `haxe_trace_enum_boxed(type_id, ptr)` reads tag + parameters from memory
  - ParamType RTTI for type-aware parameter printing (Int, Float, Bool, String, Object)
  - EnumVariantBuilder type alias for clippy compliance
  - **Commit:** 5cef9ee

- ✅ **Rustfmt formatting cleanup** in hir_to_mir.rs and ast_lowering.rs
  - **Commit:** dd623f3

**Test Status:**
- ✅ **600/600 tests passing** (100% pass rate)

---

## Recent Progress (Session 2026-01-28 - Bug Fixes)

**Critical Bug Fixes:**
- ✅ **Fixed Alloc instruction side effects** - `IrInstruction::Alloc` was not marked as having side effects in `has_side_effects()`, allowing LICM (Loop-Invariant Code Motion) to hoist allocations out of loops. This caused all loop iterations to reuse the same pointer, leading to double-frees and heap corruption (SIGSEGV at 0xf0 in pthread_mutex_lock).
  - **File:** `compiler/src/ir/instructions.rs`
  - **Fix:** Added `IrInstruction::Alloc { .. }` to the `has_side_effects()` match expression
  - **Verification:** 20/20 stress test runs passing on Docker/QEMU amd64 emulation

- ✅ **Fixed break/continue drop scope state** - Break and continue statements were not preserving drop scope state across branches, causing scope stack corruption and invalid SSA phi node updates.
  - **File:** `compiler/src/ir/hir_to_mir.rs`
  - **Fix:** Save/restore drop state around break/continue paths, update phi nodes with exit values
  - **Commit:** b08c502

- ✅ **Reverted to libc malloc/free** - After the Alloc side-effects fix was verified, switched back from tracked allocator to libc malloc/free for optimal performance.
  - **File:** `compiler/src/codegen/cranelift_backend.rs`
  - **Impact:** Restored performance while maintaining stability

**CI Infrastructure:**
- ✅ **Added Docker stress test environment** (`ci/bench-test/`)
  - `Dockerfile`: Alpine Linux + Rust + LLVM 18 for reproducible amd64 testing
  - `seghandler.c`: Signal handler for capturing SIGSEGV with stack traces
  - `stress-test.sh`: Automated stress test script (20 iterations default)
  - Usage: `docker build --platform linux/amd64 -t rayzor-bench-test -f ci/bench-test/Dockerfile .`

**Test Status:**
- ✅ **600/600 tests passing** (100% pass rate)
- ✅ **576 total commits** in repository
- ✅ All tiered backend stress tests pass (20/20 runs on Docker/QEMU)

---

## Recent Progress (Session 2025-11-27)

**Completed:**
- ✅ **Std class verified working** (5/7 methods)
  - Std.int, Std.string, Std.parseInt, Std.parseFloat, Std.random all passing
  - Created test_std_class.rs with comprehensive tests
  - Std.is and Std.downcast require full RTTI system (deferred)
- ✅ **Sys class extended** (10/20 methods now working)
  - Added Sys.command() - shell command execution
  - Added Sys.getChar() - stdin character reading
  - All 8 tested methods passing: time, cpuTime, systemName, getCwd, getEnv, putEnv, sleep, programPath, command
  - Created test_sys_class.rs with comprehensive tests
- ✅ **File I/O complete** (15/20 operations - 75%)
  - FileSystem: exists, isDirectory, isFile, createDirectory, deleteDirectory, deleteFile, rename, fullPath, absolutePath, stat, readDirectory
  - File: getContent, saveContent, copy
  - Created test_file_io.rs with 9 comprehensive tests
  - All tests passing reliably
  - Added HaxeFileStat struct with Unix metadata (gid, uid, size, mtime, etc.)
  - Fixed MIR verifier errors for extern class methods with runtime mappings

**Key Implementation Details:**
- Added haxe_sys_command() in runtime/src/haxe_sys.rs (shell execution via sh -c)
- Added haxe_sys_get_char() in runtime/src/haxe_sys.rs (stdin reading)
- Added haxe_filesystem_stat() returning HaxeFileStat with full Unix metadata
- Added haxe_filesystem_read_directory() returning Array<String>
- Added haxe_filesystem_is_file() extension function
- Fixed extern class method lowering to skip MIR stub generation for runtime-mapped methods
- Fixed runtime mapping return types (primitive vs complex) for FileSystem methods
- Added TypeTable::iter() for type iteration
- Pre-load stdlib imports before compilation for typedef availability

**Test Results:**
- test_std_class.rs: 5/5 tests passing
- test_sys_class.rs: 8/8 tests passing
- test_file_io.rs: 9/9 tests passing

**Stdlib Coverage Update:**
- Overall coverage increased from ~55% to ~58%
- Core types: ✅ Complete (String, Array, Math)
- Concurrency: ✅ Complete (Thread, Arc, Mutex, Channel)
- Sys class: 🟡 50% (10/20 functions)
- Std class: 🟡 70% (5/7 functions)
- File I/O: 🟡 75% (15/20 functions)

---

## Recent Progress (Session 2025-11-25)

**Completed:**
- ✅ **String class fully implemented and verified stable**
  - 12 String methods working: length, charAt, charCodeAt, indexOf, lastIndexOf, substr, substring, toUpperCase, toLowerCase, toString, fromCharCode, split
  - Fixed type inference for String method arguments (using TAST expression types)
  - Fixed return type handling (I32 for Int-returning methods, Ptr(String) for String-returning)
  - Fixed static method return type extraction from Function types
  - Added extern function lookup in build_call_direct for correct register typing
- ✅ Created comprehensive test_string_class.rs with all String methods
- ✅ All String tests passing reliably (3/3 stability runs)

**Key Fixes:**
1. `hir_to_mir.rs`: Use `self.convert_type(arg.ty)` for accurate argument types
2. `hir_to_mir.rs`: Return type mapping based on method name (I32 vs Ptr(String))
3. `hir_to_mir.rs`: Extract return type from Function types for static methods
4. `builder.rs`: Check `extern_functions` in `build_call_direct` for correct return types

**Session 2025-11-25 (Continued):**
- ✅ **Array class core operations verified**
  - Fixed Array.pop() - created haxe_array_pop_ptr that returns value directly
  - Fixed ptr_conversion to extend i32 to i64 for consistent 8-byte elem_size
  - Push, pop, length, and index access all working
- ✅ **Math class fully verified**
  - Added get_extern_function_signature() for Math function f64 signatures
  - All Math functions (abs, floor, ceil, sqrt, sin, cos, etc.) working with f64
- ✅ **Key fixes:**
  - `hir_to_mir.rs`: i32->i64 extension in ptr_conversion for array operations
  - `hir_to_mir.rs`: Separate get_extern_function_signature for extern-only sigs
  - `runtime/haxe_array.rs`: New haxe_array_pop_ptr returning value directly

**Next Steps:**
1. Test remaining Array methods (slice, reverse, insert, remove)
2. Test Math.random()
3. Consider consolidating String/Array/Math into single stdlib test suite
4. Run test_core_types_e2e.rs to validate all core types

---

## Recent Progress (Session 2025-11-24)

**Completed:**
- ✅ ARM64 macOS JIT stability (MAP_JIT + pthread_jit_write_protect_np)
- ✅ Cranelift fork PR review feedback addressed
- ✅ 100% stability (20/20 test runs passing)
- ✅ Comprehensive stdlib audit completed
- ✅ Implementation plan documented in Section 6

**Stdlib Audit Findings:**
- 37 extern classes identified in haxe-std
- 94 runtime functions exist (~43% coverage)
- ⚠️ String, Array, Math need stability verification (may be outdated)
- ✅ Concurrency primitives verified stable (Thread, Arc, Mutex, Channel)
- High priority gaps: Std, Type, Reflect, File I/O

**Next Steps:**
1. Verify String, Array, Math runtime stability
2. Implement Std class (string, parseInt, parseFloat, is)
3. Implement basic Type/Reflect for runtime type info
4. Complete Sys class (env vars, cwd, command execution)
5. File System I/O

---

## Known Issues (Technical Debt)

### Phi Node Bug: Variables with Limited Scope in If/Else ✅ RESOLVED

**Status:** ✅ Fixed
**Priority:** High
**Discovered:** 2025-12-02
**Resolved:** 2025-12-03
**Fix Commit:** d5ab906

**Problem:** When a variable is defined in only one branch of an if/else statement, the compiler incorrectly generates block parameters for the merge block that reference the variable from both branches, even though it's only defined in one.

**Example:**
```haxe
var acquired = mutex.tryAcquire();
if (acquired) {
    var acquired2 = mutex.tryAcquire();  // Only defined in true branch
    trace(acquired2);
} else {
    trace("failed");
}
// acquired2 is NOT used here, but compiler generates bad phi nodes
```

**Cranelift Verifier Error:**
```
inst32 (jump block2(v12)): uses value v12 from non-dominating inst16
```

**Cranelift IR Analysis:**
```
block3 (true branch):
    v12 = call fn9(v4)  // acquired2 defined here
    jump block1(v12)    // Correct - v12 exists

block2 (false branch):
    // v12 NOT defined in this branch
    jump block1(v12)    // ❌ ERROR! v12 doesn't exist here

block1(v23: i64):  // Merge block expects parameter
    // Uses v23 (phi result from both branches)
```

**Root Cause Hypothesis:** **Variable scope information is lost during TAST→HIR→MIR lowering**. When phi nodes/block parameters are created for merge points, the system has lost track of which variables are defined in which branches, so it tries to create phi nodes for ALL variables that appear in the symbol table, even those only defined in one branch.

**Investigation Findings:**

1. **Compilation Pipeline Confirmed:**
   - TAST → HIR (via `tast_to_hir` in `lowering.rs`)
   - HIR → MIR (via `hir_to_mir` in `hir_to_mir.rs`)
   - MIR → Cranelift IR (via `cranelift_backend.rs`)

2. **Attempted Fix in Wrong Location:**
   - Modified `hir_to_mir.rs::lower_if_statement` to only collect phi values for pre-existing variables
   - Debug output confirmed this function is **NOT called** for the test code
   - The `hir_to_mir.rs::lower_if_statement` function may be dead code or used for different AST types

3. **Actual Code Path:**
   - `lowering.rs::lower_if_statement` (TAST→HIR) is called
   - This function does **NOT** create phi nodes explicitly
   - Phi nodes/block parameters appear in final IR but are not created in lowering code

4. **Data Loss Theory:**
   - During TAST→HIR lowering, variable scope information (which branch defines which variable) is discarded
   - Later, when SSA form requires phi nodes, the system has no way to know `acquired2` only exists in one branch
   - It sees `acquired2` in the symbol table and creates a phi node for it, assuming it exists in all branches

**Next Investigation Steps:**

1. Add debug output to `lowering.rs::lower_if_statement` to confirm it's the active code path
2. Check if HIR representation preserves variable scope information
3. Search for SSA conversion or phi insertion passes that run after lowering
4. Examine MIR builder to see if it auto-creates phis on branch merges
5. Dump MIR before Cranelift to see if phi nodes already exist
6. Check Cranelift backend's `collect_phi_args_with_coercion` - why doesn't it error when value is missing?

**Potential Fix Locations:**

1. **lowering.rs**: Track variable scope during TAST→HIR and only create merge points for variables defined in all branches
2. **SSA pass**: If there's an SSA conversion pass, it needs variable liveness analysis before creating phis
3. **MIR builder**: If builder auto-creates phis, it needs scope-aware logic
4. **Cranelift backend**: Should validate that phi incoming values exist in their source blocks

**Files Modified (Investigation):**
- `compiler/src/ir/hir_to_mir.rs` lines 5314, 5346, 5355, 5360, 5364, 5384, 5403, 5425-5463 (added debug output)
- Changes may need to be reverted and applied to correct location once found

**Test Case:**
`/Users/amaterasu/Vibranium/rayzor/compiler/examples/test_deque_condition.rs` - `test_mutex_try_acquire()` function

---

## ✅ RESOLUTION (2025-12-03)

**Root Cause Confirmed:**
1. **TAST→HIR**: Block expressions in conditionals were not handled (unimplemented expression error)
2. **HIR→MIR**: Phi node generation used fallback logic that violated SSA dominance by using values from wrong branches

**The Fix:**

**1. compiler/src/ir/lowering.rs (TAST→HIR):**
- Added Block expression handling (lines 463-471) to process statements within block expressions
- Modified `lower_conditional` (lines 1243-1298) to detect Block expressions and use proper control flow with basic blocks instead of select operations

**2. compiler/src/ir/hir_to_mir.rs (HIR→MIR):**
- Added pre-check before phi node creation (lines 7557-7566) to skip variables that don't exist in all non-terminated branches
- Fixed phi incoming edge logic (lines 7583-7617) to only use values that exist in each specific branch

**Key Change:**
```rust
// BEFORE: Used values from wrong branches (violated dominance)
let val = else_reg.unwrap_or(before_reg.unwrap_or(sample_reg));
                                               // └─> from then branch!

// AFTER: Only use values that exist in current branch
if let Some(val) = else_reg.or(before_reg) {
    self.builder.add_phi_incoming(merge_block, phi_reg, else_end_block, val);
}
```

**Test Results:**
- ✅ test_deque_condition: 3/3 PASS (was failing with verifier error)
- ✅ test_rayzor_stdlib_e2e: 8/8 PASS (no regression)
- ✅ Zero Cranelift verifier errors
- ✅ All pre-existing test failures remain unchanged (verified with git stash)

**Files Modified:**
- `compiler/src/ir/lowering.rs`: +92 lines (Block expr handling, control flow conditionals)
- `compiler/src/ir/hir_to_mir.rs`: +36 lines (phi node validation logic)

---

### String ABI Inconsistency ⏸️

**Problem:** Multiple incompatible `HaxeString` struct definitions exist in the runtime:
- `haxe_sys.rs`: `{ptr: *const u8, len: usize}` = 16 bytes
- `string.rs`: `{ptr: *mut u8, len: usize, cap: usize}` = 24 bytes
- `haxe_string.rs`: possibly different definition

**Impact:**
- String concatenation crashes due to ABI mismatch
- Functions returning `HaxeString` by value have struct return ABI issues on ARM64
- Cannot safely pass strings between different runtime modules

**Fix Required:**
1. Consolidate to single `HaxeString` definition
2. All string functions should return `*mut HaxeString` (pointer) to avoid struct return ABI issues
3. Update stdlib and HIR-to-MIR lowering to use pointer-based string handling consistently

**Workaround:** Use `haxe_string_literal` which returns a pointer, avoid string concatenation for now.

### Deref Coercion for Wrapper Types ⏸️

**Problem:** Arc, MutexGuard, and similar wrapper types were initially expected to implicitly inherit methods/fields of their inner type (like Rust's Deref coercion), but this is not implemented.

**Current Workaround:** Explicitly call `.get()` on Arc/MutexGuard to access the inner value.

**Example:**
```haxe
var arc = new Arc<Int>(42);
// arc.someMethod();  // Would need Deref coercion
arc.get().someMethod();  // Works with explicit .get()
```

---

## Recent Progress (Session 2025-11-16)

**Completed:**
- ✅ Property access infrastructure (TAST, HIR, MIR)
- ✅ Property getter method call generation
- ✅ Property setter method call generation
- ✅ Method name resolution (get_x/set_x convention)
- ✅ Read/write-only property enforcement
- ✅ All 7/7 e2e tests pass MIR validation

**Identified Blockers:**
- ❌ Runtime concurrency primitives missing (thread, arc, mutex, channel)
- ❌ Cranelift symbol registration for runtime functions
- ❌ E2E test execution infrastructure (L5/L6)

**Next Steps:**
1. Implement runtime concurrency primitives
2. Register runtime symbols in Cranelift backend
3. Enable JIT execution for e2e tests

## Recent Progress (Session 2025-12-03)

**Completed:**
- ✅ Fixed phi node bug for branch-local variables (Cranelift verifier error)
- ✅ Added Block expression handling in TAST→HIR lowering
- ✅ Fixed control flow for conditionals with Block expressions
- ✅ Validated phi node generation logic in HIR→MIR
- ✅ All test_rayzor_stdlib_e2e tests pass (8/8)
- ✅ All test_deque_condition tests pass (3/3, previously failing)
- ✅ Comprehensive regression testing (no regressions introduced)
- ✅ Created detailed test failure investigation plan

**Test Suite Status:**
- ✅ test_rayzor_stdlib_e2e: 8/8 PASS (100%)
- ✅ test_deque_condition: 3/3 PASS (100%)
- ✅ test_generics_e2e: Compilation successful
- ⚠️ test_core_types_e2e: 25/25 PASS (100%) ✅ (was 20/25, fixed 2026-01-30)
- ⚠️ test_vec_e2e: 1/2 PASS (50%) - vec_int_basic fails (bitcast I32→I64), vec_float_basic hangs

**Identified Issues (Pre-existing):**
- ❌ Missing extern function: haxe_array_get
- ❌ Field index not found: Array.length, String fields
- ❌ Wrong instruction type: iadd.f64 instead of fadd.f64
- ❌ Return value handling broken in Vec methods
- ❌ Class registration issues for String/Array

**Documentation:**
- 📋 TEST_FAILURES_PLAN.md: Detailed investigation and fix plan for 10 failing tests
- 📋 BACKLOG.md: Updated with phi node bug resolution

**Next Steps:**
See TEST_FAILURES_PLAN.md for prioritized fix strategy:
1. Fix integration_math_array (iadd.f64 instruction bug) - High ROI
2. Add haxe_array_get extern function - Enables array operations
3. Fix test_vec_e2e return value handling - All 5 tests
4. Fix array_slice field access - Array manipulation
5. Fix string_split class registration - String utilities
