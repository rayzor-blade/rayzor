# Rayzor Compiler - Implementation Roadmap

## 🎯 Goal: Production-Ready Haxe Compiler

**Timeline**: 3-6 months to production readiness
**Current State**: 40-45% complete
**Target**: Support real-world Haxe projects including stdlib

---

## Phase 1: Critical Foundation Fixes (Weeks 1-3) 🔴
**Goal**: Unblock the compilation pipeline

### Week 1: Fix HIR Type Preservation
**Priority**: CRITICAL - Blocking everything
**Owner**: Compiler Team Lead

1. **Fix Type Resolution in HIR** (2 days)
   ```rust
   // In tast_to_hir.rs
   - Remove all TypeId::from_raw(0)
   - Use actual types from expr.expr_type
   - Add proper type lookups from type_table
   ```
   - [ ] Fix make_bool_literal() to use type_table
   - [ ] Fix make_null_literal() to use Optional<T> type
   - [ ] Fix string interpolation type preservation
   - [ ] Fix method type lookups from symbol_table
   - [ ] Add type validation in HIR construction

2. **Fix Symbol Resolution** (2 days)
   ```rust
   // Add proper symbol tracking
   - Maintain symbol_table references
   - Fix method symbol resolution
   - Add symbol validation
   ```
   - [ ] Create symbol resolution context
   - [ ] Fix SymbolId::from_raw(0) usage
   - [ ] Add symbol existence validation
   - [ ] Implement symbol scope tracking

3. **Add Error Recovery** (1 day)
   ```rust
   enum LoweringResult<T> {
       Ok(T),
       Error(HirError),
       Partial(T, Vec<HirError>)
   }
   ```
   - [ ] Implement graceful degradation
   - [ ] Add error nodes that preserve structure
   - [ ] Continue processing after errors
   - [ ] Collect all errors for reporting

### Week 2: Complete HIR Desugaring
**Priority**: HIGH - Required for correct lowering

1. **Complete Array Comprehension Desugaring** (2 days)
   ```rust
   // [for (x in xs) if (cond) expr] →
   // { let arr = []; for (x in xs) { if (cond) arr.push(expr); } arr }
   ```
   **Status**: ✅ **MOSTLY COMPLETE**
   **Location**: `/compiler/src/ir/tast_to_hir.rs:1340-1450`
   - [x] ~~Create temporary array variable with unique name~~ ✅
   - [x] ~~Build nested for loops for multiple iterators~~ ✅
   - [ ] Add condition checks as if statements (BACKLOG: TAST doesn't have filter field)
   - [x] ~~Generate array push calls~~ ✅
   - [x] ~~Return final array~~ ✅

   **Note**: Filter support (`if (cond)`) requires TAST changes - TypedComprehensionFor needs condition field

2. **Pattern Matching Desugaring** (3 days)
   ```rust
   // switch(x) { case Some(v): ... } →
   // if (x.tag == "Some") { let v = x.value; ... }
   ```
   - [ ] Convert patterns to conditions
   - [ ] Extract pattern bindings
   - [ ] Build if-else chains
   - [ ] Handle guard conditions
   - [ ] Support nested patterns

### Week 3: Fix MIR Lowering
**Priority**: HIGH - Need working MIR for optimization

1. **Complete HIR to MIR Lowering** (3 days)
   - [ ] Fix remaining TODO items in hir_to_mir.rs
   - [x] ~~Implement closure lowering~~ ✅ **COMPLETED 2025-01-13**
     - Infrastructure complete: MakeClosure, ClosureFunc, ClosureEnv instructions
     - Lambda function generation working
     - Cranelift codegen implemented
     - Test passing: [test_closure_infrastructure.rs](examples/test_closure_infrastructure.rs)
     - Docs: [CLOSURE_IMPLEMENTATION.md](CLOSURE_IMPLEMENTATION.md)
   - [x] ~~Complete lambda body lowering~~ ✅ **COMPLETED 2025-01-13**
     - Lambda bodies now contain actual executable code (not stubs)
     - Parameters accessible within lambda bodies
     - Return type correctly extracted from function type signature
     - Nested function context management working
     - Tests passing: [test_lambda_execution.rs](examples/test_lambda_execution.rs), [test_simple_indirect_call.rs](examples/test_simple_indirect_call.rs)
   - [x] ~~Implement closure environment allocation~~ ✅ **COMPLETED 2025-01-13**
     - Capture analysis implemented: detects free variables in lambda bodies
     - Environment allocation on stack via Cranelift stack slots
     - Environment loading in lambda bodies via pointer arithmetic
     - Closure invocation with automatic environment passing
     - Full pipeline working: capture → allocate → load → invoke
     - Test passing: [test_closure_call.rs](examples/test_closure_call.rs)
     - Result: Closures fully functional! Can create, capture, and call closures correctly
   - [ ] Add exception handling
   - [ ] Complete pattern lowering
   - [ ] Add phi node generation

2. **Add MIR Validation** (2 days)
   - [x] ~~Validate closure instructions~~ ✅ **COMPLETED 2025-01-13**
   - [ ] Validate SSA form
   - [ ] Check type consistency
   - [ ] Verify control flow
   - [ ] Add debug assertions

---

## Phase 1.5: Technical Debt from Week 1 🟠
**Goal**: Address foundational issues discovered during implementation**

### ClassHierarchyInfo Extension
**Priority**: HIGH - Blocking proper class metadata
**Location**: `/compiler/src/tast/type_checker.rs:970`
```rust
pub struct ClassHierarchyInfo {
    pub superclass: Option<TypeId>,
    pub interfaces: Vec<TypeId>,
    pub all_supertypes: HashSet<TypeId>,
    pub depth: usize,
    // NEED TO ADD:
    pub is_final: bool,
    pub is_abstract: bool,
    pub is_extern: bool,
    pub is_interface: bool,
    pub sealed_to: Option<Vec<TypeId>>,
}
```
- [ ] Add new fields to ClassHierarchyInfo
- [ ] Update register_class_hierarchy to populate fields
- [ ] Fix HIR lowering to use actual values

## Phase 2: Core Type System Completion (Weeks 4-7) 🟡
**Goal**: Support real Haxe language features

### Week 4: Package System Implementation
**Priority**: CRITICAL - Required for multi-file projects

1. **Implement Package Resolution** (3 days)
   ```haxe
   package com.example.utils;
   import com.example.data.Model;
   ```
   - [ ] Add package declaration support
   - [ ] Implement import resolution
   - [ ] Add package visibility checking
   - [ ] Support wildcard imports
   - [ ] Handle import aliases

2. **Multi-file Compilation** (2 days)
   - [ ] Create compilation unit manager
   - [ ] Add dependency resolution
   - [ ] Implement incremental compilation
   - [ ] Add circular dependency detection

### Week 5: Abstract Types
**Priority**: HIGH - Core Haxe feature

1. **Implement Abstract Type System** (3 days)
   ```haxe
   abstract Float(Float) from Float to Float {
     @:op(A + B) function add(rhs:Float):Float;
   }
   ```
   - [ ] Add abstract type representation
   - [ ] Implement implicit casts (from/to)
   - [ ] Add operator overloading
   - [ ] Support abstract enums
   - [ ] Add @:forward metadata

2. **Abstract Type Checking** (2 days)
   - [ ] Validate cast rules
   - [ ] Check operator usage
   - [ ] Implement type unification
   - [ ] Add error messages

### Week 6: Null Safety
**Priority**: HIGH - Modern requirement

1. **Implement Null Safety Analysis** (3 days)
   ```haxe
   var x:Null<String> = null;
   x?.length; // Safe navigation
   ```
   - [ ] Add null state tracking
   - [ ] Implement flow-sensitive analysis
   - [ ] Add null check detection
   - [ ] Support safe navigation (?.)
   - [ ] Add null coalescing (??)

2. **Null Safety Diagnostics** (2 days)
   - [ ] Add null dereference warnings
   - [ ] Suggest null checks
   - [ ] Add @:nullSafety metadata
   - [ ] Implement strict mode

### Week 7: Advanced Generics
**Priority**: MEDIUM - Required for complex code

1. **Multiple Type Constraints** (2 days)
   ```haxe
   class Container<T:Readable & Writable> { }
   ```
   - [ ] Parse multiple constraints
   - [ ] Implement constraint checking
   - [ ] Add constraint propagation

2. **Variance Completion** (3 days)
   - [ ] Complete variance checking
   - [ ] Add variance inference
   - [ ] Fix generic method calls
   - [ ] Support F-bounded polymorphism

---

## Phase 3: Code Generation (Weeks 8-10) 🟢
**Goal**: Produce executable output

### Week 8: LLVM Backend Foundation
**Priority**: CRITICAL - Need output

1. **LLVM Integration** (3 days)
   - [ ] Add llvm-sys dependency
   - [erstellen LLVM context
   - [ ] Create module builder
   - [ ] Implement type mapping
   - [ ] Add function generation

2. **Basic Code Generation** (2 days)
   - [ ] Generate main function
   - [ ] Implement arithmetic ops
   - [ ] Add function calls
   - [ ] Support control flow
   - [ ] Handle returns

### Week 9: Type Lowering to LLVM
**Priority**: HIGH

1. **Primitive Types** (2 days)
   - [ ] Map Haxe types to LLVM
   - [ ] Implement boxing/unboxing
   - [ ] Add string handling
   - [ ] Support arrays

2. **Object Model** (3 days)
   - [ ] Design object layout
   - [ ] Implement vtables
   - [ ] Add method dispatch
   - [ ] Support inheritance
   - [ ] Handle interfaces

### Week 10: Runtime Support
**Priority**: HIGH

1. **Memory Management** (3 days) ✅ **COMPLETED**
   - [x] Ownership-based memory management (no GC for typed code)
   - [x] Drop analysis with AutoDrop/RuntimeManaged/NoDrop behaviors
   - [x] Escape analysis for stack allocation optimization
   - [x] Runtime allocator (rayzor_malloc/rayzor_free)

2. **Standard Library Bridge** (2 days)
   - [ ] Implement builtin functions
   - [ ] Add string operations
   - [ ] Support array methods
   - [ ] Add I/O primitives

---

## Phase 4: Production Features (Weeks 11-14) 🔵
**Goal**: Support real-world projects

### Week 11: Macro System
**Priority**: MEDIUM - Core Haxe feature

1. **Macro Parser** (3 days)
   ```haxe
   macro function assert(expr:Expr):Expr { }
   ```
   - [ ] Add macro keyword support
   - [ ] Parse build macros
   - [ ] Support expression macros
   - [ ] Add reification

2. **Macro Expansion** (2 days)
   - [ ] Implement macro context
   - [ ] Add AST manipulation API
   - [ ] Support macro caching
   - [ ] Add error handling

### Week 12: Optimization Pipeline
**Priority**: MEDIUM - Performance

1. **Advanced Optimizations** (3 days)
   - [ ] Implement inlining
   - [ ] Add loop optimizations
   - [ ] Support devirtualization
   - [ ] Add escape analysis

2. **Link-Time Optimization** (2 days)
   - [ ] Whole program analysis
   - [ ] Dead code elimination
   - [ ] Cross-module inlining

### Week 13: Build System
**Priority**: HIGH - Usability

1. **Hxml Support** (2 days)
   ```
   -cp src
   -main Main
   -js output.js
   ```
   - [ ] Parse hxml files
   - [ ] Support all flags
   - [ ] Add configuration

2. **Incremental Compilation** (3 days)
   - [ ] Implement dependency tracking
   - [ ] Add cache management
   - [ ] Support hot reload
   - [ ] Add watch mode

### Week 14: Testing & Validation
**Priority**: CRITICAL - Quality

1. **Test Suite** (3 days)
   - [ ] Port Haxe unit tests
   - [ ] Add regression tests
   - [ ] Create benchmarks
   - [ ] Add fuzzing

2. **Standard Library** (2 days)
   - [ ] Test stdlib compilation
   - [ ] Fix compatibility issues
   - [ ] Add missing features
   - [ ] Validate output

---

## Phase 5: Additional Backends (Weeks 15-18) ⚡
**Goal**: Multiple target support

### JavaScript Backend (Week 15-16)
- [ ] AST to JS transformer
- [ ] Source map generation
- [ ] Module system support
- [ ] Optimization passes

### VM/Interpreter (Week 17-18)
- [ ] Bytecode design
- [ ] Interpreter loop
- [ ] JIT compilation
- [ ] Debugger support

---

## Success Metrics

### Milestone 1 (Week 3): Pipeline Fixed ✅
- [ ] HIR preserves all type information
- [ ] All desugaring complete
- [ ] MIR generation working
- [ ] Can compile simple programs

### Milestone 2 (Week 7): Type System Complete ✅
- [ ] Multi-file compilation working
- [ ] Abstract types supported
- [ ] Null safety implemented
- [ ] Can compile medium complexity code

### Milestone 3 (Week 10): Code Generation ✅
- [ ] LLVM backend producing executables
- [ ] Basic runtime working
- [ ] Can run simple programs
- [ ] Performance acceptable

### Milestone 4 (Week 14): Production Ready ✅
- [ ] Standard library compiles
- [ ] Macro system working
- [ ] Build system complete
- [ ] Can compile real projects

### Final Goal (Week 18): Full Compiler ✅
- [ ] Multiple backends
- [ ] Optimization pipeline complete
- [ ] IDE support ready
- [ ] Performance competitive

---

## Resource Requirements

### Team Size
- **Minimum**: 2-3 developers
- **Optimal**: 4-5 developers
- **Roles**:
  - IR/Optimization specialist
  - Type system expert
  - Code generation engineer
  - Build system/tooling developer

### Dependencies
- LLVM 15+ (for backend)
- Rust 1.70+ (compiler implementation)
- Node.js (for JS backend testing)

### Infrastructure
- CI/CD pipeline
- Benchmark servers
- Test infrastructure
- Documentation system

---

## Risk Mitigation

### High Risk Items
1. **LLVM Complexity**: Consider simpler backend first (C generation)
2. **Macro System**: Can defer to Phase 6 if needed
3. **Memory Management**: Ownership-based (completed, no GC needed for typed code)

### Fallback Plans
- **If behind schedule**: Focus on single backend (JS or LLVM)
- **If type system complex**: Defer advanced features
- **If performance issues**: Profile and optimize critical path

---

## Weekly Status Tracking

### Week 1 Status ✅ COMPLETED
- [x] HIR type preservation (100%)
  - Removed all TypeId::from_raw(0)
  - Using proper type table methods
  - Preserving types from TAST expressions
- [x] Symbol resolution (100%)
  - Proper symbol lookups from symbol table
  - Made get_class_hierarchy pub(crate)
- [x] Error recovery (100%)
  - Added LoweringResult enum
  - Created error expression/statement nodes
  - Graceful degradation on errors
- **Blockers**: None
- **Next**: Week 2 - HIR Desugaring

### Week 2 Status ✅ COMPLETED  
- [x] Pattern Matching Desugaring (80%)
  - Basic patterns (variable, wildcard, literal) working
  - Guard patterns supported
  - Complex patterns (constructor, array, object) blocked on runtime support
- [x] ClassHierarchyInfo Extension (100%)
  - Added all missing fields (is_final, is_abstract, is_extern, is_interface, sealed_to)
  - Updated all instantiation sites
  - HIR now uses real values from hierarchy
- [x] Method Symbol Resolution (100%)
  - Using extern classes from haxe-std/
  - Array.hx, String.hx loaded by stdlib_loader
  - Method lookup via symbol table implemented
- [x] Array Comprehension Desugaring (100%)
  - Full desugaring to nested loops implemented
  - Method resolution for Array.push working
  - Temporary variable generation working
  - Block expressions properly returning array
- **Blockers**: None
- **Next**: Week 3 - MIR Lowering

### Week 3 Status ✅ COMPLETED
- [x] Complete HIR to MIR lowering (100%)
  - Exception handling (try-catch-finally) with landing pads
  - Conditional expressions (ternary) with phi nodes
  - Pattern matching lowering for switch statements
  - Closure/lambda lowering with capture analysis
  - Array/Map/Object literal lowering
- [x] MIR Validation (100%)
  - SSA form validation with phi node checks
  - Type consistency checking
  - Control flow integrity validation
  - Register def-use analysis
- [x] Integration Tests (100%)
  - Created comprehensive test suite
  - Full pipeline testing (Source → AST → TAST → HIR → MIR)
  - Test coverage for all new features
- **Blockers**: None
- **Next**: Week 4 - Package System Implementation

### Technical Debt Backlog
**Resolved:**
1. ✅ **ClassHierarchyInfo Extension** - COMPLETED
2. ✅ **Array Comprehension Full Desugaring** - COMPLETED

**Still Outstanding:**
1. **Array Comprehension Filter Support** - Need to add if-conditions inside comprehensions
2. **Method Abstract Detection** - Need to check @:abstract metadata + empty body
3. **Import/Module Field Lowering** - Empty implementations need completion
4. **Type Parameter Constraints** - Not lowering generic constraints properly
5. **Complex Pattern Matching** - Constructor, array, object patterns need runtime support

[Continue updating weekly...]

---

## Definition of Done

### For Each Feature
- [ ] Implementation complete
- [ ] Unit tests passing
- [ ] Integration tests added
- [ ] Documentation written
- [ ] Performance acceptable
- [ ] Error messages helpful

### For Each Phase
- [ ] All features implemented
- [ ] Test suite passing
- [ ] Benchmarks acceptable
- [ ] Documentation complete
- [ ] Code reviewed
- [ ] Merged to main

### For Production Release
- [ ] All phases complete
- [ ] Standard library compiles
- [ ] Real projects tested
- [ ] Performance validated
- [ ] Documentation published
- [ ] Release notes prepared

---

## Appendix: Technical Decisions

### Why Fix HIR First?
- Blocks entire pipeline
- Relatively quick fix (1 week)
- Enables parallel work

### Why LLVM over other backends?
- Industry standard
- Great optimization
- Multiple target support
- Good documentation

### Why Implement Abstracts Early?
- Core Haxe feature
- Used extensively in stdlib
- Blocks many programs

### Why Defer Macros?
- Complex feature
- Can work without initially
- Time consuming

---

## Contact & Ownership

**Project Lead**: [Name]
**Technical Lead**: [Name]
**Status Updates**: Weekly on [Day]
**Repository**: github.com/org/rayzor
**Documentation**: docs.rayzor.dev
**Issue Tracker**: github.com/org/rayzor/issues