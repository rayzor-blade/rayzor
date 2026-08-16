//! HIR to MIR lowering.
//!
//! MIR is SSA with phi nodes; HIR is not. Lowering therefore does three things
//! at once: it flattens control flow into basic blocks, resolves HIR names to
//! MIR ids, and inserts the boxing, drops and dispatch that the source leaves
//! implicit.
//!
//! `HirToMirContext` carries the whole of that state and is implemented across
//! this directory:
//!
//! - `decl` — what must exist before any body is lowered: signatures, metadata, vtables
//! - `stmt`, `expr`, `field` — the lowering itself
//! - `resolve` — HIR names and symbols to MIR ids
//! - `helpers` — shared utilities: boxing, drops, allocation, type ids

use crate::ir::drop_analysis::{DropBehavior, DropPointAnalyzer, DropPoints};
use crate::ir::hir::*;
use crate::ir::{
    BinaryOp, CallingConvention, CompareOp, EnvironmentLayout, FunctionKind,
    FunctionSignatureBuilder, IrBasicBlock, IrBlockId, IrBuilder, IrEnumVariant, IrField,
    IrFunction, IrFunctionId, IrFunctionSignature, IrGlobal, IrGlobalId, IrId, IrInstruction,
    IrLocal, IrModule, IrParameter, IrPhiNode, IrSourceLocation, IrTerminator, IrType, IrTypeDef,
    IrTypeDefId, IrTypeDefinition, IrValue, Linkage, UnaryOp,
};
use crate::stdlib::{IrTypeDescriptor, MethodSignature, StdlibMapping};
use crate::tast::symbols::SymbolFlags;
use crate::tast::{
    InternedString, SourceLocation, StringInterner, SymbolId, SymbolTable, TypeId, TypeKind,
    TypeTable,
};
use log::{debug, trace, warn};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

/// Which `haxe_box_*_ptr` helper to use when wrapping a primitive
/// MIR value as a `DynamicValue*`. Keeps the runtime-symbol names
/// centralised in `box_primitive_as_dynamic`.
#[derive(Clone, Copy, Debug)]
pub(crate) enum PrimBoxKind {
    Int,
    Float,
    Bool,
}

/// Layout information for a single field in a @:cstruct class
#[derive(Debug, Clone)]
pub(crate) struct CStructFieldLayout {
    symbol_id: SymbolId,
    name: String,
    byte_offset: u32,
    ir_type: IrType,
    c_type: String, // "double", "long", "int", "void*", "Vec2", etc.
}

/// Precomputed C-compatible layout for a @:cstruct class
#[derive(Debug, Clone)]
pub(crate) struct CStructLayout {
    fields: Vec<CStructFieldLayout>,
    total_size: u32,
    alignment: u32,
    c_name: String,
    /// All dependency typedefs (for nested cstructs), in topological order
    dep_cdefs: Vec<String>,
    /// This struct's own typedef line (without deps)
    own_cdef: String,
    /// Full cdef: deps + own typedef
    cdef_string: String,
}

/// Layout information for a single field in a @:gpuStruct class
#[derive(Debug, Clone)]
pub(crate) struct GpuStructFieldLayout {
    symbol_id: SymbolId,
    name: String,
    byte_offset: u32,
    /// IR type used on CPU side (F32 for Float fields, I32 for Int, etc.)
    ir_type: IrType,
    /// MSL type string ("float", "int", "uint", etc.)
    msl_type: String,
    /// Size in bytes on GPU
    size: u32,
    /// WGSL vertex format code for use in vertex buffer layouts.
    /// 0=Float32, 1=Float32x2, 2=Float32x3, 3=Float32x4, 4=Sint32, 5=Uint32
    vertex_format: u32,
}

/// Precomputed GPU-compatible layout for a @:gpuStruct class
#[derive(Debug, Clone)]
pub(crate) struct GpuStructLayout {
    fields: Vec<GpuStructFieldLayout>,
    total_size: u32,
    alignment: u32,
    msl_name: String,
    /// MSL struct typedef string
    msl_typedef: String,
    /// Dependency typedefs for nested @:gpuStruct (in topological order)
    dep_typedefs: Vec<String>,
}

/// Context for lowering HIR to MIR
pub struct HirToMirContext<'a> {
    /// MIR builder
    builder: IrBuilder,

    /// Mapping from HIR symbols to MIR registers (for variables/parameters)
    symbol_map: BTreeMap<SymbolId, IrId>,

    /// Resolved IR types for pattern-bound variables (overrides convert_type for TypeParameter types)
    symbol_ir_types: BTreeMap<SymbolId, IrType>,

    /// Concrete TypeIds for pattern-bound variables (for toString dispatch on generic fields)
    symbol_type_ids: BTreeMap<SymbolId, TypeId>,

    /// Target TypeId for the next object literal lowering — set by Return /
    /// Let / Assign / call-arg handlers when they know the expected wider
    /// typedef type, so `lower_object_literal` computes slot indices over the
    /// FULL field set (including optional fields absent from the literal).
    /// Writer and reader must sort over the same shape or slot indices diverge.
    object_literal_target_ty: Option<TypeId>,
    /// Declared type of the enclosing `var x:T = <call>` — lets erased-generic
    /// returns (inferred `Channel<T>.receive()`) pick the right unbox when the
    /// receiver's type parameter carries no information.
    let_target_type_hint: Option<TypeId>,

    /// Mapping from HIR function symbols to MIR function IDs
    function_map: BTreeMap<SymbolId, crate::ir::IrFunctionId>,

    /// Mapping from global variable symbols to MIR global IDs
    /// Used for static class fields and module-level variables
    global_symbol_map: BTreeMap<SymbolId, IrGlobalId>,

    /// Qualified name -> global id for globals defined in ALREADY-LOWERED
    /// modules. Modules are lowered separately and each starts with an empty
    /// globals table, so a static declared in an import is otherwise invisible
    /// to the module that reads it.
    ///
    /// Safe to hold ids: imports are renumbered into disjoint ranges
    /// (`old_id + import_base`) BEFORE the reading module is lowered, and
    /// backends key global storage by raw id, so these are final.
    external_globals: BTreeMap<String, (IrGlobalId, IrType)>,

    /// External function map from previously compiled modules (e.g., stdlib)
    /// These are functions defined in other modules that can be called from this module
    external_function_map: BTreeMap<SymbolId, crate::ir::IrFunctionId>,

    /// Name-based external function map for cross-file lookups
    /// This is used when SymbolIds don't match (e.g., "StringTools.startsWith" -> IrFunctionId)
    external_function_name_map: BTreeMap<String, crate::ir::IrFunctionId>,

    /// Mapping from HIR blocks to MIR blocks
    block_map: BTreeMap<usize, IrBlockId>,

    /// Loop context for break/continue
    loop_stack: Vec<LoopContext>,

    /// Stack of loop-carried symbol sets — one entry per active loop whose
    /// body drop-scope is on `drop_scope_stack`. A symbol is loop-carried when
    /// it is declared in an ENCLOSING scope but assigned inside the loop body
    /// (so its value flows out through the loop-carried / exit phi). Its owned
    /// heap value must NOT be freed by the loop-body `exit_drop_scope`: the phi
    /// keeps the pointer live past the loop, so freeing it is a use-after-free.
    loop_carried_symbols: Vec<BTreeSet<SymbolId>>,

    /// Current HIR module being processed
    current_module: Option<String>,

    /// Current class being processed (for @:jsImport propagation to methods)
    current_class_symbol: Option<SymbolId>,

    /// Error accumulator
    errors: Vec<LoweringError>,

    /// Warning accumulator (non-fatal diagnostics like exhaustiveness checks)
    diagnostics: Vec<diagnostics::Diagnostic>,

    /// SSA-derived optimization hints extracted from HIR metadata
    /// These are queried from DFG during HIR lowering and passed to MIR
    ssa_hints: SsaOptimizationHints,

    /// Counter for generating unique lambda names
    lambda_counter: u32,

    /// Dynamic global initializers (globals needing runtime initialization)
    dynamic_globals: Vec<(SymbolId, HirExpr)>,

    /// String interner for resolving InternedString to actual strings
    string_interner: &'a StringInterner,

    /// Type table for type conversion. Borrowed once at context creation;
    /// lowering never mutates it.
    type_table: &'a TypeTable,

    /// Track closure registers and their environment pointers
    /// Maps: closure_function_pointer_register -> environment_pointer_register
    closure_environments: BTreeMap<IrId, IrId>,

    /// Mapping from field SymbolId to (class_type_id, field_index)
    /// This allows us to find the index of a field within its class
    field_index_map: BTreeMap<SymbolId, (TypeId, u32)>,

    /// Reverse index: TypeId → Vec<(field_symbol, field_index)>
    /// Built lazily from field_index_map. Invalidated when field_index_map changes.
    fields_by_type_cache: Option<BTreeMap<TypeId, Vec<(SymbolId, u32)>>>,

    /// Mapping from (typedef_type_id, field_name) to field_index for anonymous struct fields
    /// This allows field access on typedef'd anonymous structs like FileStat
    /// where field symbols may be created at access sites rather than typedef declaration
    typedef_field_map: BTreeMap<(TypeId, InternedString), u32>,

    /// Mapping from field SymbolId to PropertyAccessInfo (for properties with custom getters/setters)
    /// This allows us to route property access through the appropriate getter/setter methods
    property_access_map: BTreeMap<SymbolId, crate::tast::PropertyAccessInfo>,

    /// Mapping from class TypeId to constructor IrFunctionId
    /// This allows new expressions to find the constructor by class type
    constructor_map: BTreeMap<TypeId, IrFunctionId>,

    /// Reflective constructor wrappers keyed by class TypeId.
    /// Each wrapper has a uniform ABI `(obj_ptr, args_array)` and invokes the
    /// real constructor after unboxing dynamic arguments.
    constructor_reflect_wrappers: BTreeMap<TypeId, IrFunctionId>,

    /// Thunks for bound method references (`obj.method` lvalue).
    /// Map: method's `IrFunctionId` → thunk `IrFunctionId`. The thunk
    /// has closure-ABI signature `(env: *u8, ...method_params_without_this)`
    /// and forwards to the method after loading `this` from `env[0]`.
    /// Cached so `var f = obj.m; var g = obj.m;` reuses one thunk per
    /// underlying method.
    method_ref_thunks: BTreeMap<IrFunctionId, IrFunctionId>,

    /// Thunks for virtual/interface dispatch slots. The generic indirect-call
    /// ABI prepends a closure env pointer, while class methods are declared as
    /// `(this, ...args)`. These thunks use `(env, this, ...args)` and forward to
    /// the real method, ignoring `env`.
    vtable_dispatch_thunks: BTreeMap<IrFunctionId, IrFunctionId>,

    /// Mapping from qualified class name to constructor IrFunctionId
    /// This is a fallback when TypeIds don't match (e.g., across separately compiled files)
    constructor_name_map: BTreeMap<String, IrFunctionId>,
    /// Owning class (bare name) per constructor, used to verify candidates
    /// found through id-space puns before they are called.
    constructor_owner_map: BTreeMap<IrFunctionId, String>,

    /// Reference to HIR type declarations for inheritance lookup
    /// Needed to access parent class fields during field inheritance
    current_hir_types: &'a indexmap::IndexMap<TypeId, HirTypeDecl>,

    /// Standard library runtime function mapping
    stdlib_mapping: &'a StdlibMapping,

    /// Symbol table for resolving symbols
    symbol_table: &'a SymbolTable,

    /// Track when we're inside a lambda and what its environment layout is
    current_env_layout: Option<EnvironmentLayout>,

    /// Current class type for instance methods (used to resolve implicit field accesses)
    current_this_type: Option<TypeId>,

    /// Mapping from variable SymbolIds to their monomorphized stdlib class names
    /// Used to track Vec<Int> -> VecI32, Vec<Float> -> VecF64, etc.
    /// This is needed because extern generic classes don't have proper TypeIds in the type table
    monomorphized_var_types: BTreeMap<SymbolId, String>,

    /// Mapping from result register IrIds to the stdlib class they belong to.
    /// Used to disambiguate method calls on TypeParameter receivers when multiple
    /// stdlib classes have the same method name (e.g., Arc.get vs MutexGuard.get).
    /// Set by the Dynamic/TypeParameter method dispatch handler after MIR wrapper calls.
    register_class_hints: BTreeMap<IrId, String>,

    /// Result registers from cross-context interface method calls whose
    /// HIR result type was `Dynamic` (because TAST couldn't resolve the
    /// iface method's return type) but for which we successfully
    /// re-resolved a concrete TypeId via `interface_method_return_types`
    /// at MIR-lowering time. Lets `Let` propagate the concrete type
    /// down through variable bindings so subsequent receiver dispatch
    /// sees the real type instead of falling into the Dynamic path.
    interface_call_result_types: BTreeMap<IrId, TypeId>,
    /// Registers holding values that `maybe_box_value` actually boxed
    /// (Dynamic box protocol). Register-keyed and per-function (cleared with
    /// `symbol_map`): consumers use it to tell a genuine box apart from raw
    /// payloads that share the same `Ptr(Void)` register type.
    boxed_value_regs: BTreeSet<IrId>,

    /// Variables whose declared HIR type is `Dynamic` but whose RHS at
    /// the binding site was a tracked `interface_call_result_types`
    /// register. Receiver-type lookups (Call method dispatch, field
    /// access) consult this map and treat the variable as the
    /// concrete type instead of `Dynamic`.
    var_concrete_overrides: BTreeMap<SymbolId, TypeId>,

    /// Dedup set for `emit_iface_return_diagnostic` so the same
    /// `(method, file, line, col)` doesn't dump dozens of identical
    /// warnings when a generic helper is called many times.
    iface_diag_seen: BTreeSet<(String, u32, u32, u32)>,

    /// Enums that need RTTI registration
    /// Maps enum SymbolId -> (runtime_type_id, enum_name, variant_names)
    enums_for_registration: BTreeMap<SymbolId, (u32, String, Vec<String>)>,

    /// Counter for generating unique wrapper function names
    next_wrapper_id: u32,

    /// Maps variable SymbolId to the IrId holding its current heap-allocated value
    /// When a variable is reassigned, we free its old value before assigning the new one
    owned_heap_values: BTreeMap<SymbolId, IrId>,

    /// Stack of scopes, each containing the variables that own heap allocations in that scope
    /// When a scope exits, we emit Free for all owned values in that scope
    /// Inner Vec contains (SymbolId, IrId) pairs of owned values
    drop_scope_stack: Vec<Vec<(SymbolId, IrId)>>,

    /// Set of IrIds that are temporaries needing drop after use
    /// These are intermediate values from expressions (e.g., `new Complex(...).mul(...)`)
    /// that need to be freed after the containing expression completes
    temp_heap_values: Vec<IrId>,

    /// Drop point analysis results for current function
    /// Contains last-use information for precise drop insertion
    current_drop_points: Option<DropPoints>,

    /// Current statement index during lowering (for drop point matching)
    current_stmt_index: usize,

    /// Symbols that have been reassigned in the current scope
    /// These should be skipped at scope exit (they're freed at reassignment time)
    reassigned_in_scope: BTreeSet<SymbolId>,

    /// Precomputed C-compatible layouts for @:cstruct classes
    /// Maps class TypeId to its CStructLayout
    cstruct_layouts: BTreeMap<TypeId, CStructLayout>,

    /// Precomputed GPU-compatible layouts for @:gpuStruct classes
    /// Maps class TypeId to its GpuStructLayout
    gpu_struct_layouts: BTreeMap<TypeId, GpuStructLayout>,

    /// Lazily-initialized TCC runtime function IDs for __c__ inline code
    tcc_func_ids: Option<TccFuncIds>,

    /// Current function's SymbolId (for function-level metadata like @:frameworks)
    current_function_symbol: Option<SymbolId>,

    /// Mapping from sorted anonymous field names to shape_id (for runtime shape table)
    /// Key: comma-joined sorted field names — Value: shape_id (u32)
    anonymous_shapes: BTreeMap<String, u32>,

    /// Next shape ID to allocate (incremented each time a new shape is registered)
    next_anon_shape_id: u32,

    /// Interface method ordering: maps interface SymbolId → ordered list of method names
    interface_method_names: BTreeMap<SymbolId, Vec<InternedString>>,

    /// Interface method return types: maps (interface SymbolId, method name) → return TypeId
    interface_method_return_types: BTreeMap<(SymbolId, InternedString), TypeId>,

    /// Interface vtables: maps (class SymbolId, interface SymbolId) → method SymbolIds
    /// Each entry's method order matches the interface's method_names order
    interface_vtables: BTreeMap<(SymbolId, SymbolId), Vec<SymbolId>>,

    /// Interface inheritance: maps interface SymbolId → parent interface SymbolIds
    /// Used to create transitive vtables for implementing classes
    interface_extends: BTreeMap<SymbolId, Vec<SymbolId>>,

    /// Registers holding interface fat pointers built at call boundaries
    /// (`maybe_materialize_for_call` Path 3) that wrap a class arg for an
    /// interface-typed parameter. Such wrappers may escape via the callee
    /// (e.g. pushed into a long-lived `Array<I>`), so callers must NOT add
    /// them to `temp_heap_values` — auto-freeing them after the call would
    /// be a use-after-free for the array reference. Trades a transient leak
    /// for correctness.
    /// TODO: retain only when the call returns or stores the fat pointer.
    interface_wrapped_args: std::collections::BTreeSet<IrId>,

    /// Class method lookup: maps (class_symbol, method_name) → method SymbolId
    /// Populated during register_class_metadata for iterator protocol dispatch
    class_method_symbols: BTreeMap<(SymbolId, InternedString), SymbolId>,

    /// Constrained type parameter info for function parameters.
    /// Maps IrFunctionId → Vec<(param_index, interface_SymbolId)>
    /// Used at call sites to wrap arguments in fat pointers for constrained type params.
    constrained_param_interfaces: BTreeMap<IrFunctionId, Vec<(usize, SymbolId)>>,

    /// Abstract type @:from conversion rules.
    /// Maps abstract name (InternedString) → list of from-rules.
    abstract_from_rules: BTreeMap<InternedString, Vec<HirCastRule>>,

    /// Abstract type @:to conversion rules.
    /// Maps abstract name (InternedString) → list of to-rules.
    abstract_to_rules: BTreeMap<InternedString, Vec<HirCastRule>>,

    /// Abstract @:forward rules.
    /// Maps abstract SymbolId → (underlying_type, forwarded method names). Empty vec = forward all.
    abstract_forward_rules: BTreeMap<SymbolId, (TypeId, Vec<InternedString>)>,

    /// Symbols known to hold boxed DynamicValue pointers (from maybe_box_value in Let/Assign).
    /// Used by the Dynamic BinaryOp path to distinguish actual boxed Dynamic values from
    /// raw i64 values that happen to have Dynamic type (e.g., untyped lambda params).
    boxed_dynamic_symbols: BTreeSet<SymbolId>,

    /// Symbols known to carry `Type.typeof(...)` boxed ValueType payloads.
    /// These values are runtime enum payload pointers encoded as i64 and need
    /// dedicated formatting paths for trace/Std.string/interpolation parity.
    value_type_symbols: BTreeSet<SymbolId>,

    /// Symbols holding raw anonymous object handles (not boxed DynamicValue*).
    /// These come from runtime functions like `haxe_ereg_matched_pos_anon` that return
    /// Ptr(U8) anon handles directly. Field access on these must use haxe_reflect_field
    /// directly, without haxe_unbox_reference_ptr (which would corrupt the raw handle).
    raw_anon_symbols: BTreeSet<SymbolId>,

    /// Allocation sizes for class instances (in bytes).
    /// Keyed by the class declaration's SymbolId — the only id space that is
    /// stable across compilation contexts. TypeIds are context-local and must
    /// never key this map: a punned or shifted TypeId silently returns another
    /// class's size and under-allocates.
    class_alloc_sizes: BTreeMap<SymbolId, u64>,

    /// Name-keyed class allocation sizes (qualified_name → bytes).
    /// TypeIds are unstable across compilation contexts. This map provides a stable
    /// fallback for classes compiled in separate units (e.g., stdlib Exception).
    class_alloc_sizes_by_name: BTreeMap<String, u64>,

    /// Parsed-AST declaration index (shared from CompilationUnit). Supplies
    /// the declared instance-field count for a `new C(...)` whose class
    /// compiles later than this module — the arg-count guess under-allocates
    /// (ctor default params / fields set post-ctor overflow the block).
    static_sig_index:
        Option<std::rc::Rc<std::cell::RefCell<crate::tast::sig_index::StaticSigIndex>>>,

    /// Maps class registration TypeId → class SymbolId. Survives type_table overwrites.
    /// Used for field disambiguation when multiple classes have same-named fields.
    class_type_to_symbol: BTreeMap<TypeId, SymbolId>,

    /// Maps field SymbolId → class name string for BLADE cache serialization.
    field_class_names: BTreeMap<SymbolId, String>,

    /// (child_class_symbol, method_name) for methods with is_override=true
    override_methods: BTreeSet<(SymbolId, InternedString)>,

    /// class SymbolId → parent class SymbolId (from extends)
    class_parent_map: BTreeMap<SymbolId, SymbolId>,

    /// (class_symbol, method_name) → method SymbolId (all class methods)
    class_method_by_name: BTreeMap<(SymbolId, InternedString), SymbolId>,

    /// class SymbolId → [(method_name, slot_index)] — virtual method slots
    class_virtual_slots: BTreeMap<SymbolId, Vec<(InternedString, u32)>>,

    /// class SymbolId → [method_SymbolId per slot] — most-derived implementation
    class_vtables: BTreeMap<SymbolId, Vec<SymbolId>>,

    /// base method SymbolId → (slot_index, defining_class) — checked at call sites
    virtual_dispatch_info: BTreeMap<SymbolId, (u32, SymbolId)>,

    /// Parameter default expressions for functions/constructors.
    /// Keyed by IrFunctionId, value is Vec matching user-visible params (excludes implicit 'this').
    /// Only populated for functions that have at least one parameter with a default value.
    function_param_defaults: BTreeMap<IrFunctionId, Vec<Option<HirExpr>>>,

    /// Constructor param counts from import modules (for BLADE-cached constructors).
    /// fill_default_args uses this when the constructor isn't in the local module yet.
    external_constructor_param_counts: BTreeMap<IrFunctionId, usize>,

    /// Parameter types for ALL functions from imported modules.
    /// Used by fill_default_args to fill missing optional params with correct types.
    external_function_param_types: BTreeMap<IrFunctionId, Vec<IrType>>,

    /// HIR parameter types for functions/constructors.
    /// Keyed by IrFunctionId, value is Vec<TypeId> of param types (excludes implicit 'this').
    /// Used for structural subtyping: detecting class→anon or wider-anon→anon at call sites.
    function_param_hir_types: BTreeMap<IrFunctionId, Vec<TypeId>>,

    /// Per-parameter qualified type *names* for imported functions whose
    /// HIR didn't go through this context's lowering (BLADE-cache loads
    /// and cross-file fresh-compiled imports). MIR alone erases both
    /// Class and Interface to `Ptr(Void)`, so without these names
    /// `maybe_materialize_for_call` Path 3 silently skips the
    /// class→interface fat-pointer wrap for any imported constructor.
    /// `None` entries mark slots Path 3 should ignore (primitive, anon, …).
    external_function_param_iface_names: BTreeMap<IrFunctionId, Vec<Option<String>>>,

    /// Current function's return TypeId, set in lower_function_body.
    /// Used by the Return handler to box values for Null<T> return types.
    current_function_return_type: Option<TypeId>,

    /// Structural subtyping: tracks variables where an anon-typed variable is backed by
    /// a different runtime representation (class pointer or wider anonymous object).
    /// Field access is redirected at compile time; materialization only at escape points.
    anon_views: BTreeMap<SymbolId, AnonBacking>,

    /// Registers that hold async function results (Future handles).
    /// Method calls like `.await()` on these registers are dispatched to Future_await.
    async_result_registers: BTreeSet<IrId>,

    /// Classes that derive PartialEq — enables field-by-field == / != codegen
    derive_partial_eq_classes: BTreeSet<SymbolId>,

    /// Classes that derive PartialOrd — enables field-by-field < / <= / > / >= codegen
    derive_partial_ord_classes: BTreeSet<SymbolId>,

    /// Classes that derive Hash — enables synthetic hashCode() method
    derive_hash_classes: BTreeSet<SymbolId>,

    /// Classes that derive Clone — enables synthetic clone() method
    derive_clone_classes: BTreeSet<SymbolId>,

    /// Classes annotated with `@:move` — values use strict move semantics.
    /// Tracked in parallel with `derive_clone_classes` so MIR lowering can
    /// emit `MarkMoved` / `CheckLive` opcodes for use-after-move analysis.
    derive_move_classes: BTreeSet<SymbolId>,

    /// Classes annotated with `@:shared` — `.clone()` lowers to an atomic
    /// reference-count increment (runtime-safe aliasing) and move-tracking
    /// is suppressed for bindings of these classes. Mutually exclusive with
    /// `@:move`; see W0030 diagnostic for the conflict case.
    derive_shared_classes: BTreeSet<SymbolId>,

    /// IrIds bound to `@:move`-class locals. Reads of these registers
    /// receive a `CheckLive` guard; consumes receive a `MarkMoved` marker.
    strict_move_locals: BTreeSet<IrId>,

    /// Tier B: extern classes whose @:derive(Clone) delegates to a runtime function
    /// (e.g. rayzor.ds.Tensor → "rayzor_tensor_clone"). Looked up in lower_derived_clone
    /// before field-by-field synthesis; if present, emits a single direct call.
    derive_clone_extern_fns: BTreeMap<SymbolId, &'static str>,

    /// Classes that derive Copy — implicit shallow copy at Let/Assign/Call boundaries
    derive_copy_classes: BTreeSet<SymbolId>,

    /// Classes that derive Drop — user-defined destructor called before Free
    derive_drop_classes: BTreeSet<SymbolId>,

    /// Maps IrId (allocation register) → class SymbolId for Drop dispatch
    ir_to_drop_class: BTreeMap<IrId, SymbolId>,

    /// @:manualDrop classes — compiler does NOT auto-free
    manual_drop_classes: BTreeSet<SymbolId>,

    /// Classes that derive Debug — enables synthetic toString() method
    derive_debug_classes: BTreeSet<SymbolId>,

    /// Classes that derive Default — enables synthetic static default() method
    derive_default_classes: BTreeSet<SymbolId>,

    /// Instance fields for classes with derive traits.
    /// Maps class SymbolId → list of (field_symbol, field_type, gep_index) for non-static fields.
    /// Used by PartialEq/PartialOrd/Hash codegen to iterate fields in order.
    class_instance_fields: BTreeMap<SymbolId, Vec<(SymbolId, TypeId, u32)>>,
    /// Debug-only: how many times register_type_metadata ran in THIS context.
    /// Distinguishes "registration never ran here" from "ran but saw no classes"
    /// when class_instance_fields comes up empty at a field access.
    dbg_type_meta_calls: usize,

    /// Field default value expressions — from @:default(value) metadata or field initializers.
    /// Maps field SymbolId → HirExpr to use as default in @:derive(Default).
    field_default_exprs: BTreeMap<SymbolId, HirExpr>,

    /// Custom debug format strings from @:debugFormat("pattern") metadata.
    /// Pattern uses {fieldName} placeholders, e.g. "({x}, {y})".
    debug_format_strings: BTreeMap<SymbolId, String>,
}

/// Tracks the backing representation of an anonymous-typed variable.
/// Used for deferred structural subtyping: the variable holds the original value
/// (class pointer or wider anon handle), and field access is redirected at compile time.
#[derive(Debug, Clone)]
pub(crate) enum AnonBacking {
    /// Value is a class pointer. Access fields via GEP at mapped indices.
    Class {
        class_symbol: SymbolId,
        class_type: TypeId,
        /// (target_field_name, gep_index, field_type_id)
        field_map: Vec<(String, u32, TypeId)>,
    },
    /// Value is a wider anonymous object. Access fields via remapped sorted index.
    WiderAnon {
        source_type: TypeId,
        /// (target_field_name, source_sorted_index, field_type_id)
        field_map: Vec<(String, usize, TypeId)>,
    },
}

/// TCC runtime function IDs for __c__ inline code lowering
#[derive(Debug, Clone, Copy)]
pub(crate) struct TccFuncIds {
    create: IrFunctionId,           // rayzor_tcc_create() -> I64
    compile: IrFunctionId,          // rayzor_tcc_compile(I64, PtrString) -> I32
    add_value_symbol: IrFunctionId, // rayzor_tcc_add_value_symbol(I64, PtrString, I64) -> I64
    relocate: IrFunctionId,         // rayzor_tcc_relocate(I64) -> I32
    get_symbol: IrFunctionId,       // rayzor_tcc_get_symbol(I64, PtrString) -> I64
    delete: IrFunctionId,           // rayzor_tcc_delete(I64) -> Void
    call0: IrFunctionId,            // rayzor_tcc_call0(I64) -> I64
    free_value: IrFunctionId,       // rayzor_tcc_free_value(I64) -> Void
    string_replace: IrFunctionId, // haxe_string_replace(PtrString, PtrString, PtrString) -> PtrString
    add_framework: IrFunctionId,  // rayzor_tcc_add_framework(I64, PtrString) -> I32
    add_include_path: IrFunctionId, // rayzor_tcc_add_include_path(I64, PtrString) -> I32
    add_file: IrFunctionId,       // rayzor_tcc_add_file(I64, PtrString) -> I32
    add_clib: IrFunctionId,       // rayzor_tcc_add_clib(I64, PtrString) -> I32
}

/// SSA-derived optimization hints from DFG analysis
/// These guide MIR generation and optimization without rebuilding SSA
#[derive(Debug, Default)]
pub(crate) struct SsaOptimizationHints {
    /// Functions that are inline candidates (small, simple control flow)
    inline_candidates: BTreeSet<SymbolId>,

    /// Functions with straight-line code (no branches, optimize aggressively)
    straight_line_functions: BTreeSet<SymbolId>,

    /// Functions with complex control flow (many phi nodes, careful optimization)
    complex_control_flow_functions: BTreeSet<SymbolId>,

    /// Functions with common subexpressions (CSE opportunities)
    cse_opportunities: BTreeSet<SymbolId>,
}

#[derive(Debug)]
pub(crate) struct LoopContext {
    continue_block: IrBlockId,
    break_block: IrBlockId,
    label: Option<SymbolId>,
    /// Maps symbol IDs to their exit block phi registers
    /// When breaking, we need to add incoming edges to these phi nodes
    exit_phi_nodes: BTreeMap<SymbolId, IrId>,
    /// Maps symbol IDs to their update block phi registers (for range/for loops).
    /// When continuing, we need to add incoming edges to these phi nodes
    /// so that the update block can merge values from both the continue path
    /// and the normal body-end path.
    continue_phi_nodes: BTreeMap<SymbolId, IrId>,
}

#[derive(Debug)]
pub struct LoweringError {
    pub message: String,
    pub location: SourceLocation,
}

/// Outcome of resolving a field's (class, slot) by bare name. `Ambiguous`
/// carries candidates that DISAGREE on the slot — consuming callers must
/// hard-fail rather than pick one; existence probes may treat it as "found".
pub(crate) enum FieldIndexResolution {
    None,
    Unique(TypeId, u32),
    Ambiguous(Vec<(TypeId, u32)>),
}

/// Context for lambda function generation (Two-Pass Architecture)
pub(crate) struct LambdaContext {
    /// The function ID of the lambda
    func_id: IrFunctionId,
    /// Entry block of the lambda
    entry_block: IrBlockId,
    /// Offset for parameter registers (0 if no env, 1 if has env)
    param_offset: u32,
    /// Environment layout (if captures exist)
    env_layout: Option<EnvironmentLayout>,
}

/// Saved state for restoring after lambda generation
pub(crate) struct SavedLoweringState {
    current_function: Option<IrFunctionId>,
    current_block: Option<IrBlockId>,
    symbol_map: BTreeMap<SymbolId, IrId>,
    current_env_layout: Option<EnvironmentLayout>,
    // Drop tracking state - must be saved/restored to prevent lambda bodies
    // from inheriting (and freeing) the parent function's owned values
    owned_heap_values: BTreeMap<SymbolId, IrId>,
    drop_scope_stack: Vec<Vec<(SymbolId, IrId)>>,
    loop_carried_symbols: Vec<BTreeSet<SymbolId>>,
    temp_heap_values: Vec<IrId>,
    reassigned_in_scope: BTreeSet<SymbolId>,
    current_drop_points: Option<DropPoints>,
    current_stmt_index: usize,
    boxed_dynamic_symbols: BTreeSet<SymbolId>,
    value_type_symbols: BTreeSet<SymbolId>,
    anon_views: BTreeMap<SymbolId, AnonBacking>,
    raw_anon_symbols: BTreeSet<SymbolId>,
    // Per-function @:move tracking — snapshot before switching into a
    // lambda/async/inner function so the outer body's MarkMoved/CheckLive
    // emission resumes correctly after restore.
    strict_move_locals: BTreeSet<IrId>,
}

/// Process-global class field layouts, keyed by NAME.
///
/// `class_instance_fields` is per-lowering-context and populated only from that
/// context's own `hir_module.types`, so a class TYPE crosses the module
/// boundary while its LAYOUT does not. This map carries the layout across.
///
/// Stores only (field name -> GEP index). Deliberately NOT TypeId: type ids are
/// per-context indices and are meaningless in a consumer. The access site
/// already knows the field's own type, so only the index has to travel.
/// Keyed by name because SymbolIds drift across contexts (both the class's and
/// the field's). The key is the CANONICAL class name — the fully-qualified name
/// when the symbol has one (`nue.arch.LlamaModel`), the bare name otherwise.
static CLASS_FIELD_LAYOUTS: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashMap<String, Vec<(String, u32)>>>,
> = std::sync::OnceLock::new();

/// Bare-name alias: `LlamaModel` -> Some(canonical), or None once two DIFFERENT
/// canonical classes have claimed the same bare name (poisoned). A consumer
/// context often only knows the bare name (its forwarded stub may carry no
/// qualified name), so bare lookups must work — but an ambiguous bare name must
/// fail loudly rather than resolve to whichever class registered first: a wrong
/// GEP index is silent memory corruption, strictly worse than a compile error.
static CLASS_NAME_ALIAS: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashMap<String, Option<String>>>,
> = std::sync::OnceLock::new();

fn class_field_layouts(
) -> &'static std::sync::Mutex<std::collections::HashMap<String, Vec<(String, u32)>>> {
    CLASS_FIELD_LAYOUTS.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

fn class_name_alias() -> &'static std::sync::Mutex<std::collections::HashMap<String, Option<String>>>
{
    CLASS_NAME_ALIAS.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

/// Record one field's index under its class's canonical name. Called from the
/// same loop that populates `field_index_map`, which runs for every registered
/// class. Idempotent: re-registration writes the same layout.
fn record_class_field(qualified: Option<&str>, bare: &str, field_name: &str, idx: u32) {
    let canonical = qualified.unwrap_or(bare);
    if let Ok(mut m) = class_field_layouts().lock() {
        let v = m.entry(canonical.to_string()).or_default();
        if !v.iter().any(|(n, _)| n == field_name) {
            v.push((field_name.to_string(), idx));
        }
    }
    if let Ok(mut a) = class_name_alias().lock() {
        match a.get(bare) {
            None => {
                a.insert(bare.to_string(), Some(canonical.to_string()));
            }
            Some(Some(existing)) if existing != canonical => {
                // Second distinct canonical class with this bare name: poison.
                a.insert(bare.to_string(), None);
            }
            _ => {}
        }
    }
}

/// Look up a field's GEP index by class name (qualified or bare) + field name.
/// Bare names route through the alias map so a poisoned (ambiguous) bare name
/// returns None and the caller errors loudly instead of guessing.
fn lookup_class_field_gep(class_name: &str, field_name: &str) -> Option<u32> {
    let canonical: String = {
        let a = class_name_alias().lock().ok()?;
        match a.get(class_name) {
            Some(Some(c)) => c.clone(),
            Some(None) => return None,      // ambiguous bare name — refuse
            None => class_name.to_string(), // may itself be a canonical key
        }
    };
    let m = class_field_layouts().lock().ok()?;
    m.get(&canonical)?
        .iter()
        .find(|(n, _)| n == field_name)
        .map(|(_, idx)| *idx)
}

mod decl;
mod expr;
mod field;
mod helpers;
/// `HirToMirContext`'s methods, split out to keep this file navigable.
mod resolve;
mod stmt;
mod types;

pub(crate) enum MirBinaryOp {
    Binary(BinaryOp),
    Compare(CompareOp),
}

/// Map a SIMD vector IrType to its stdlib class name, so instance-method
/// dispatch picks the right wrapper set: i32x4 → SIMD4i32, i32x8 → SIMD8i32,
/// i8x16 → SIMD16i8, i8x32 → SIMD32i8, everything else (f32x4) → SIMD4f.
/// Both element type AND lane count matter — the 128- and 256-bit twins have
/// identical method names bound to different-width MIR wrappers.
fn simd_vector_class(ty: &IrType) -> &'static str {
    match ty {
        IrType::Vector { element, count }
            if matches!(element.as_ref(), IrType::I32 | IrType::U32) =>
        {
            if *count == 8 {
                "rayzor_SIMD8i32"
            } else {
                "rayzor_SIMD4i32"
            }
        }
        IrType::Vector { element, count }
            if matches!(element.as_ref(), IrType::I8 | IrType::U8) =>
        {
            if *count == 32 {
                "rayzor_SIMD32i8"
            } else {
                "rayzor_SIMD16i8"
            }
        }
        _ => "rayzor_SIMD4f",
    }
}

/// Public API for HIR to MIR lowering
pub fn lower_hir_to_mir(
    hir_module: &HirModule,
    string_interner: &StringInterner,
    type_table: &Rc<RefCell<TypeTable>>,
    symbol_table: &SymbolTable,
) -> Result<IrModule, Vec<LoweringError>> {
    lower_hir_to_mir_with_externals(
        hir_module,
        string_interner,
        type_table,
        symbol_table,
        BTreeMap::new(),
    )
}

/// Public API for HIR to MIR lowering with external function references
///
/// The `external_functions` map contains SymbolId -> IrFunctionId mappings for functions
/// defined in other modules (e.g., stdlib) that can be called from this module.
pub fn lower_hir_to_mir_with_externals(
    hir_module: &HirModule,
    string_interner: &StringInterner,
    type_table: &Rc<RefCell<TypeTable>>,
    symbol_table: &SymbolTable,
    external_functions: BTreeMap<SymbolId, IrFunctionId>,
) -> Result<IrModule, Vec<LoweringError>> {
    // Borrow once: lowering never mutates the type table.
    let type_table_ref = type_table.borrow();
    let stdlib_mapping = StdlibMapping::builtin();
    let mut context = HirToMirContext::new(
        hir_module.name.clone(),
        hir_module.metadata.source_file.clone(),
        string_interner,
        &type_table_ref,
        &hir_module.types,
        symbol_table,
        stdlib_mapping,
    );

    context.external_function_map = external_functions;

    context.lower_module(hir_module)
}

/// Result of MIR lowering that includes both the module and the function mappings
pub struct MirLoweringResult {
    /// The compiled MIR module
    pub module: IrModule,
    /// Mapping from HIR function symbols to MIR function IDs
    /// This can be used to build the external_functions map for other modules
    pub function_map: BTreeMap<SymbolId, IrFunctionId>,
    /// Field index mappings for class instances (SymbolId -> (TypeId, field_index))
    /// Exported so that importing modules can resolve field access on imported classes
    pub field_index_map: BTreeMap<SymbolId, (TypeId, u32)>,
    /// Property accessor info for class properties
    pub property_access_map: BTreeMap<SymbolId, crate::tast::PropertyAccessInfo>,
    /// Constructor name map (class qualified name -> constructor IrFunctionId)
    /// Exported so that importing modules can resolve `new ClassName()` across files
    pub constructor_name_map: BTreeMap<String, IrFunctionId>,
    /// Class allocation sizes (TypeId -> byte size)
    /// Exported so importing modules know how much memory to allocate for `new ClassName()`
    pub class_alloc_sizes: BTreeMap<SymbolId, u64>,
    /// Name-keyed class allocation sizes (class_name → bytes).
    /// Stable across compilation contexts where TypeIds differ.
    pub class_alloc_sizes_by_name: BTreeMap<String, u64>,
    /// Class method symbols (class_symbol, method_name) -> method_symbol
    /// Exported so importing modules can resolve iterator protocol methods
    pub class_method_symbols: BTreeMap<(SymbolId, InternedString), SymbolId>,
    /// Class TypeId → SymbolId mapping. Survives type_table overwrites.
    /// Used for field disambiguation when multiple classes have same-named fields.
    pub class_type_to_symbol: BTreeMap<TypeId, SymbolId>,
    /// Field SymbolId → class name string. Used by BLADE cache to store field entries
    /// with class names that survive across compilation contexts (where TypeIds differ).
    pub field_class_names: BTreeMap<SymbolId, String>,
    /// Interface metadata — accumulated across files so cross-file interface
    /// dispatch (e.g. `var t:Tokenizer = ...; t.method()` in a file that
    /// imports the interface declared elsewhere) can resolve methods, look
    /// up vtable layout, and wrap class pointers in interface fat pointers.
    pub interface_method_names: BTreeMap<SymbolId, Vec<InternedString>>,
    pub interface_method_return_types: BTreeMap<(SymbolId, InternedString), TypeId>,
    pub interface_extends: BTreeMap<SymbolId, Vec<SymbolId>>,
    pub interface_vtables: BTreeMap<(SymbolId, SymbolId), Vec<SymbolId>>,
    /// HIR-level param types per function id. Captured here so the BLADE
    /// cache can persist them by qualified name and rebuild the map for
    /// imported functions in a future compilation context. Without it,
    /// `maybe_materialize_for_call` Path 3 skips the class→interface
    /// fat-pointer wrap for imported constructors.
    pub function_param_hir_types: BTreeMap<IrFunctionId, Vec<TypeId>>,
    /// Non-fatal diagnostics from the lowering pass (e.g., exhaustiveness warnings)
    pub diagnostics: Vec<diagnostics::Diagnostic>,
}

/// Lower HIR to MIR and return both the module and function mappings
///
/// This is useful when you need to compile multiple modules and have later modules
/// call functions from earlier modules (e.g., user code calling stdlib functions).
pub fn lower_hir_to_mir_with_function_map(
    hir_module: &HirModule,
    string_interner: &StringInterner,
    type_table: &Rc<RefCell<TypeTable>>,
    symbol_table: &SymbolTable,
    external_functions: BTreeMap<SymbolId, IrFunctionId>,
    external_functions_by_name: BTreeMap<String, IrFunctionId>,
    external_globals: BTreeMap<String, (IrGlobalId, IrType)>,
    stdlib_mapping: &StdlibMapping,
    external_field_index_map: BTreeMap<SymbolId, (TypeId, u32)>,
    external_property_access_map: BTreeMap<SymbolId, crate::tast::PropertyAccessInfo>,
    external_constructor_name_map: BTreeMap<String, IrFunctionId>,
    external_class_alloc_sizes: BTreeMap<SymbolId, u64>,
    external_class_method_symbols: BTreeMap<(SymbolId, InternedString), SymbolId>,
    external_class_type_to_symbol: BTreeMap<TypeId, SymbolId>,
    external_constructor_param_counts: BTreeMap<IrFunctionId, usize>,
    external_function_param_types: BTreeMap<IrFunctionId, Vec<IrType>>,
    external_class_alloc_sizes_by_name: BTreeMap<String, u64>,
    external_interface_method_names: BTreeMap<SymbolId, Vec<InternedString>>,
    external_interface_method_return_types: BTreeMap<(SymbolId, InternedString), TypeId>,
    external_interface_extends: BTreeMap<SymbolId, Vec<SymbolId>>,
    external_interface_vtables: BTreeMap<(SymbolId, SymbolId), Vec<SymbolId>>,
    external_function_param_iface_names: BTreeMap<IrFunctionId, Vec<Option<String>>>,
    external_field_class_names: BTreeMap<SymbolId, String>,
    static_sig_index: Option<
        std::rc::Rc<std::cell::RefCell<crate::tast::sig_index::StaticSigIndex>>,
    >,
) -> Result<MirLoweringResult, Vec<LoweringError>> {
    let type_table_ref = type_table.borrow();
    let mut context = HirToMirContext::new(
        hir_module.name.clone(),
        hir_module.metadata.source_file.clone(),
        string_interner,
        &type_table_ref,
        &hir_module.types,
        symbol_table,
        stdlib_mapping,
    );
    context.static_sig_index = static_sig_index;

    context.external_function_map = external_functions;
    context.external_function_name_map = external_functions_by_name;
    context.external_globals = external_globals;

    context.field_index_map = external_field_index_map;
    context.property_access_map = external_property_access_map;

    // The constructor_name_map key is the owning class's name, which also seeds
    // the owner map used to verify pun-derived candidates.
    for (class_name, func_id) in &external_constructor_name_map {
        let bare = class_name.rsplit('.').next().unwrap_or(class_name);
        context
            .constructor_owner_map
            .entry(*func_id)
            .or_insert_with(|| bare.to_string());
    }
    context.constructor_name_map = external_constructor_name_map;

    context.class_alloc_sizes = external_class_alloc_sizes;
    context.class_method_symbols = external_class_method_symbols;
    context.class_type_to_symbol = external_class_type_to_symbol;
    context.class_alloc_sizes_by_name = external_class_alloc_sizes_by_name;

    // Per-function HIR param qualified names let Path 3 of
    // `maybe_materialize_for_call` recover class→interface wrap decisions
    // for imported constructors.
    context.external_function_param_iface_names = external_function_param_iface_names;

    // Interface metadata from imports, so cross-file interface dispatch
    // (assignment to interface-typed var, vtable-based call, fat-pointer wrap)
    // finds the method ordering and vtable for interfaces declared elsewhere.
    context.interface_method_names = external_interface_method_names;
    context.interface_method_return_types = external_interface_method_return_types;
    context.interface_extends = external_interface_extends;
    context.interface_vtables = external_interface_vtables;
    // Imported field→owner-class names let the property-getter name-based
    // fallback in `lower_field_access` reject cross-class matches (e.g.
    // `Bytes.length` must not resolve to `StringBuf.get_length`).
    context.field_class_names = external_field_class_names;

    // Also populate the name-keyed map from the SymbolId-keyed one.
    for (&sym_id, &size) in &context.class_alloc_sizes {
        if let Some(sym) = symbol_table.get_symbol(sym_id) {
            let name = sym
                .qualified_name
                .and_then(|n| string_interner.get(n))
                .or_else(|| string_interner.get(sym.name));
            if let Some(name) = name {
                context
                    .class_alloc_sizes_by_name
                    .insert(name.to_string(), size);
            }
        }
    }

    // Both feed fill_default_args for cross-module optional params.
    context.external_constructor_param_counts = external_constructor_param_counts;
    context.external_function_param_types = external_function_param_types;

    let mut module = context.lower_module(hir_module)?;

    // Reverse map func_id -> qualified_name for external functions: lets the
    // blade cache resolve cross-module references when function ids change
    // between sessions (different renumbering bases).
    if !context.external_function_name_map.is_empty() {
        let mut ext_id_to_name: BTreeMap<IrFunctionId, String> = BTreeMap::new();
        for (name, &func_id) in &context.external_function_name_map {
            ext_id_to_name
                .entry(func_id)
                .or_insert_with(|| name.clone());
        }
        for func in module.functions.values() {
            for block in func.cfg.blocks.values() {
                for inst in &block.instructions {
                    let ref_ids: Vec<IrFunctionId> = match inst {
                        crate::ir::IrInstruction::CallDirect { func_id, .. }
                        | crate::ir::IrInstruction::FunctionRef { func_id, .. }
                        | crate::ir::IrInstruction::MakeClosure { func_id, .. } => {
                            vec![*func_id]
                        }
                        _ => vec![],
                    };
                    for fid in ref_ids {
                        if !module.functions.contains_key(&fid)
                            && !module.extern_functions.contains_key(&fid)
                        {
                            if let Some(name) = ext_id_to_name.get(&fid) {
                                module.external_function_names.insert(fid, name.clone());
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(MirLoweringResult {
        module,
        function_map: context.function_map,
        field_index_map: context.field_index_map,
        property_access_map: context.property_access_map,
        constructor_name_map: context.constructor_name_map,
        class_alloc_sizes: context.class_alloc_sizes,
        class_alloc_sizes_by_name: context.class_alloc_sizes_by_name,
        class_method_symbols: context.class_method_symbols,
        class_type_to_symbol: context.class_type_to_symbol,
        field_class_names: context.field_class_names,
        interface_method_names: context.interface_method_names,
        interface_method_return_types: context.interface_method_return_types,
        interface_extends: context.interface_extends,
        interface_vtables: context.interface_vtables,
        function_param_hir_types: context.function_param_hir_types,
        diagnostics: context.diagnostics,
    })
}

impl<'a> HirToMirContext<'a> {
    /// The text an interned name stands for.
    ///
    /// An `InternedString` is an index into a per-process interner, so anything
    /// that reaches MIR as a name — a local's name, a string literal, a metadata
    /// key compared against a literal — has to be resolved through the interner
    /// that produced it. An unresolvable id yields an empty string rather than a
    /// placeholder that could be mistaken for a real name.
    pub(crate) fn interned_str(&self, name: crate::tast::InternedString) -> &'a str {
        self.string_interner.get(name).unwrap_or("")
    }

    /// Create a new lowering context
    pub fn new(
        module_name: String,
        source_file: String,
        string_interner: &'a StringInterner,
        type_table: &'a TypeTable,
        hir_types: &'a indexmap::IndexMap<TypeId, HirTypeDecl>,
        symbol_table: &'a SymbolTable,
        stdlib_mapping: &'a StdlibMapping,
    ) -> Self {
        let mut ctx = Self {
            builder: IrBuilder::new(module_name.clone(), source_file),
            symbol_map: BTreeMap::new(),
            symbol_ir_types: BTreeMap::new(),
            symbol_type_ids: BTreeMap::new(),
            object_literal_target_ty: None,
            let_target_type_hint: None,
            function_map: BTreeMap::new(),
            global_symbol_map: BTreeMap::new(),
            external_globals: BTreeMap::new(),
            external_function_map: BTreeMap::new(),
            external_function_name_map: BTreeMap::new(),
            block_map: BTreeMap::new(),
            loop_stack: Vec::new(),
            loop_carried_symbols: Vec::new(),
            current_module: Some(module_name),
            current_class_symbol: None,
            errors: Vec::new(),
            diagnostics: Vec::new(),
            ssa_hints: SsaOptimizationHints::default(),
            lambda_counter: 0,
            dynamic_globals: Vec::new(),
            string_interner,
            type_table,
            closure_environments: BTreeMap::new(),
            field_index_map: BTreeMap::new(),
            fields_by_type_cache: None,
            typedef_field_map: BTreeMap::new(),
            property_access_map: BTreeMap::new(),
            constructor_map: BTreeMap::new(),
            constructor_reflect_wrappers: BTreeMap::new(),
            method_ref_thunks: BTreeMap::new(),
            vtable_dispatch_thunks: BTreeMap::new(),
            constructor_name_map: BTreeMap::new(),
            constructor_owner_map: BTreeMap::new(),
            current_hir_types: hir_types,
            stdlib_mapping,
            symbol_table,
            current_env_layout: None,
            current_this_type: None,
            monomorphized_var_types: BTreeMap::new(),
            register_class_hints: BTreeMap::new(),
            interface_call_result_types: BTreeMap::new(),
            boxed_value_regs: BTreeSet::new(),
            var_concrete_overrides: BTreeMap::new(),
            iface_diag_seen: BTreeSet::new(),
            enums_for_registration: BTreeMap::new(),
            next_wrapper_id: 0,
            owned_heap_values: BTreeMap::new(),
            drop_scope_stack: Vec::new(),
            temp_heap_values: Vec::new(),
            current_drop_points: None,
            current_stmt_index: 0,
            reassigned_in_scope: BTreeSet::new(),
            cstruct_layouts: BTreeMap::new(),
            gpu_struct_layouts: BTreeMap::new(),
            tcc_func_ids: None,
            current_function_symbol: None,
            anonymous_shapes: BTreeMap::new(),
            next_anon_shape_id: 0,
            interface_method_names: BTreeMap::new(),
            interface_method_return_types: BTreeMap::new(),
            interface_vtables: BTreeMap::new(),
            interface_extends: BTreeMap::new(),
            interface_wrapped_args: std::collections::BTreeSet::new(),
            class_method_symbols: BTreeMap::new(),
            constrained_param_interfaces: BTreeMap::new(),
            abstract_from_rules: BTreeMap::new(),
            abstract_to_rules: BTreeMap::new(),
            abstract_forward_rules: BTreeMap::new(),
            boxed_dynamic_symbols: BTreeSet::new(),
            value_type_symbols: BTreeSet::new(),
            raw_anon_symbols: BTreeSet::new(),
            class_alloc_sizes: BTreeMap::new(),
            class_alloc_sizes_by_name: BTreeMap::new(),
            static_sig_index: None,
            class_type_to_symbol: BTreeMap::new(),
            field_class_names: BTreeMap::new(),
            override_methods: BTreeSet::new(),
            class_parent_map: BTreeMap::new(),
            class_method_by_name: BTreeMap::new(),
            class_virtual_slots: BTreeMap::new(),
            class_vtables: BTreeMap::new(),
            virtual_dispatch_info: BTreeMap::new(),
            function_param_defaults: BTreeMap::new(),
            external_constructor_param_counts: BTreeMap::new(),
            external_function_param_types: BTreeMap::new(),
            function_param_hir_types: BTreeMap::new(),
            external_function_param_iface_names: BTreeMap::new(),
            current_function_return_type: None,
            anon_views: BTreeMap::new(),
            async_result_registers: BTreeSet::new(),
            derive_partial_eq_classes: BTreeSet::new(),
            derive_partial_ord_classes: BTreeSet::new(),
            derive_hash_classes: BTreeSet::new(),
            derive_clone_classes: BTreeSet::new(),
            derive_move_classes: BTreeSet::new(),
            derive_shared_classes: BTreeSet::new(),
            strict_move_locals: BTreeSet::new(),
            derive_clone_extern_fns: BTreeMap::new(),
            derive_copy_classes: BTreeSet::new(),
            derive_drop_classes: BTreeSet::new(),
            ir_to_drop_class: BTreeMap::new(),
            manual_drop_classes: BTreeSet::new(),
            derive_debug_classes: BTreeSet::new(),
            derive_default_classes: BTreeSet::new(),
            class_instance_fields: BTreeMap::new(),
            dbg_type_meta_calls: 0,
            field_default_exprs: BTreeMap::new(),
            debug_format_strings: BTreeMap::new(),
        };

        // Must exist before any body is lowered: allocation and drop emission
        // reference them directly.
        ctx.declare_malloc();
        ctx.declare_free();

        ctx
    }
}
