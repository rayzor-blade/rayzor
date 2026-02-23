# Macro System Architecture

The Rayzor macro system executes Haxe `macro` functions at compile time, transforming AST expressions before type checking and code generation. It supports expression macros, `@:build` macros, reification, the `haxe.macro.Context` API, and class-aware dispatch.

## Pipeline Position

```
Source → Parse → AST
                  │
                  ├── Registry scan (macro definitions, @:build metadata)
                  ├── ClassRegistry scan (class constructors, methods, fields)
                  │
                  ▼
              Macro Expansion  ← this module
                  │
                  ▼
              Expanded AST → TAST → HIR → MIR → Codegen
```

Macro expansion runs **between parsing and type checking**. The expanded AST is then lowered through the normal compilation pipeline.

## Execution Model: Tiered Interpretation

The interpreter uses a morsel-parallelism-inspired **tiered execution** strategy with two backends:

### Tree-Walker (Cold Path)

The default execution path. A recursive evaluator (`interpreter.rs`, ~2400 lines) dispatches on `ExprKind` variants via a 35+ arm match. Variables are resolved by string-keyed scope chain lookup (O(n) depth).

Used for:
- All macros when `RAYZOR_MACRO_VM` is not set
- Cold macros (called fewer than threshold times)
- Fallback when bytecode compilation or VM execution fails

### Bytecode VM (Hot Path)

A stack-based VM (`bytecode/vm.rs`) that executes flat `Vec<u8>` bytecode compiled from AST. Local variables are accessed by index into the stack (O(1)).

Used for:
- Hot macros that cross the call-count threshold
- Automatically promoted by the `MorselScheduler`

### MorselScheduler

The tiering scheduler in `interpreter.rs` tracks per-macro call counts:

```
Call 1..N-1  → Tree-walker (profiling: increment count)
Call N       → Tree-walker + compile macro + compile class dependencies ("morsel")
Call N+1..   → Bytecode VM (fast path)
```

A **morsel** is a macro + its transitive class dependencies, batch-compiled as a unit. This amortizes compilation cost: one-shot macros never pay it, while hot macros amortize it over subsequent calls.

**Configuration:**
- `RAYZOR_MACRO_VM=1` — enable the bytecode VM and tiering scheduler
- `RAYZOR_MACRO_VM_THRESHOLD=N` — promote after N tree-walker calls (default: 2)

## Bytecode VM

### Instruction Set

62 opcodes with variable-length encoding (1–5 bytes per instruction):

| Category | Opcodes |
|----------|---------|
| Stack | `Const(u16)`, `PushNull`, `PushTrue`, `PushFalse`, `PushInt0`, `PushInt1`, `Pop`, `Dup`, `Swap` |
| Locals | `LoadLocal(u16)`, `StoreLocal(u16)`, `DefineLocal(u16)`, `LoadUpvalue(u16)` |
| Arithmetic | `Add`, `Sub`, `Mul`, `Div`, `Mod`, `Neg`, `Incr`, `Decr` |
| Comparison | `Eq`, `NotEq`, `Lt`, `Le`, `Gt`, `Ge` |
| Bitwise | `BitAnd`, `BitOr`, `BitXor`, `Shl`, `Shr`, `Ushr`, `BitNot` |
| Logic | `Not`, `NullCoal` |
| Control | `Jump(i16)`, `JumpIfFalse(i16)`, `JumpIfTrue(i16)`, `JumpIfFalseKeep(i16)`, `JumpIfTrueKeep(i16)` |
| Calls | `Call(u8)`, `CallMethod(u16,u8)`, `CallStatic(u16,u16,u8)`, `CallBuiltin(u16,u8)`, `CallMacroDef(u16,u8)` |
| Fields | `GetField(u16)`, `SetField(u16)`, `SetFieldLocal(u16,u16)`, `GetFieldOpt(u16)`, `GetIndex`, `SetIndex` |
| Construction | `MakeArray(u16)`, `MakeObject(u16)`, `MakeMap(u16)`, `MakeClosure(u16)`, `NewObject(u16,u8)` |
| Macro | `Reify`, `DollarSplice(u16)`, `MacroWrap` |
| Return | `Return`, `ReturnNull` |

### Compilation Unit (Chunk)

```rust
struct Chunk {
    code: Vec<u8>,                    // flat bytecode
    constants: Vec<MacroValue>,       // literal pool (strings, ints, floats)
    local_count: u16,                 // stack slots needed
    params: Vec<CompiledParam>,       // parameter metadata
    closures: Vec<Chunk>,             // nested closure chunks
    local_names: Vec<(u16, String)>,  // slot→name (for reification env reconstruction)
    name: String,                     // for debug/disassembly
}
```

### VM Execution

```rust
struct MacroVm {
    stack: Vec<MacroValue>,           // operand stack
    frames: Vec<CallFrame>,           // call frame stack
    class_chunks: HashMap<String, CompiledClassInfo>,  // compiled class data
}

struct CallFrame {
    chunk: Arc<Chunk>,
    ip: usize,                        // instruction pointer (byte offset)
    bp: usize,                        // base pointer (stack index for locals)
}
```

Locals are accessed as `stack[frame.bp + slot]` — O(1) vs the tree-walker's O(n) scope chain.

### Class Dispatch

When the VM encounters class-related opcodes:

- **`NewObject`**: Creates an object with `__type__` + instance var defaults, then pushes a constructor frame (slot 0 = `this`)
- **`CallMethod`**: Reads `__type__` from the object, looks up the compiled instance method, pushes a method frame (slot 0 = `this`)
- **`CallStatic`**: Looks up the compiled static method by class name, pushes a frame
- **`SetFieldLocal`**: Modifies a field directly on the object in a local slot via `Arc::make_mut` (avoids losing mutations when the base is a local variable)

## Core Components

### Registry (`registry.rs`)

Scans parsed `HaxeFile` declarations for `macro` functions. Stores `MacroDefinition` entries (name, params, AST body). Also stores compiled bytecode chunks once a macro is promoted.

### ClassRegistry (`class_registry.rs`)

Scans parsed files for class declarations. Stores constructors, instance methods, static methods, and field variable info. The macro interpreter falls back to ClassRegistry when hardcoded dispatch (Std, Math, etc.) doesn't match.

### Interpreter (`interpreter.rs`)

Tree-walking evaluator. Handles all `ExprKind` variants: literals, variables, binary/unary ops, control flow (if/while/for/switch/break/continue/return), function calls, field access, object/array construction, closures, and special expressions (cast, type check, string interpolation).

Also hosts the `MorselScheduler` for tiered VM promotion.

### Reification Engine (`reification.rs`)

Handles `macro { ... }` blocks — converts runtime values back into AST expressions:

| Splice | Meaning | Example |
|--------|---------|---------|
| `$v{expr}` | Value splice | `$v{42}` → `ExprKind::Int(42)` |
| `$i{expr}` | Identifier splice | `$i{"foo"}` → `ExprKind::Ident("foo")` |
| `$e{expr}` | Expression splice | `$e{someExpr}` → the expr itself |
| `$a{expr}` | Array splice | Splices array elements into parent |
| `$p{expr}` | Path splice | Dot-separated path construction |
| `$b{expr}` | Block splice | Block expression construction |

### Context API (`context_api.rs`)

Implements `haxe.macro.Context` methods that bridge macro execution with the compiler's internal state:

- `Context.parse(str)` — parse a string as a Haxe expression
- `Context.currentPos()` — current source position
- `Context.getType(name)` — look up a type
- `Context.getLocalType()` — enclosing class type
- `Context.getBuildFields()` — fields of the class being built

### Build Macros (`build_macros.rs`)

Processes `@:build(MacroName.method())` and `@:autoBuild` metadata on classes. Build macros receive the class fields and can add, remove, or modify them at compile time.

### Environment (`environment.rs`)

Lexical scope stack for the tree-walking interpreter. Supports `push_scope()` / `pop_scope()` with variable shadowing.

### Value Types (`value.rs`)

```rust
enum MacroValue {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(Arc<String>),
    Array(Arc<Vec<MacroValue>>),
    Object(Arc<HashMap<String, MacroValue>>),
    Enum { tag: String, params: Vec<MacroValue> },
    Expr(Box<Expr>),
    Type(Box<MacroType>),
    Function(Box<MacroFunction>),
    Position(Box<SourceLocation>),
}
```

Strings, Arrays, and Objects use `Arc` for cheap cloning with copy-on-write via `Arc::make_mut`.

## Benchmarks

Macro benchmarks (`compiler/benches/macro_bench.rs`) model real-world patterns:

| Benchmark | Pattern | What it tests |
|-----------|---------|---------------|
| `tink_serialize` | Object construction + method calls in loops | Class-heavy single-call macros |
| `build_accessors` | Field processing with validation chains | ORM/@:build patterns |
| `lookup_table` | Hash table construction at compile time | Compile-time data structures |
| `assert_rewrite` | Many independent macro call sites | Multi-call simple macros |
| `helper_chain` | Deep cross-class method delegation | Layered macro helper architectures |
| `multi_entity_schema` | Nested entity×field iteration | Schema/protobuf code generation |
| `full_pipeline` | End-to-end through compilation | Real-world compile latency |

### Performance

With tiered VM (`RAYZOR_MACRO_VM=1`, threshold=2):

- **One-shot macros**: 0% overhead (never compiled, always tree-walker)
- **Hot multi-call macros**: promoted to bytecode, ~neutral to slight improvement
- **Previous eager compilation**: caused 30–44% regressions on one-shot macros; tiering eliminates this entirely
