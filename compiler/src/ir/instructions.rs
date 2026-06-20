//! IR Instructions
//!
//! Defines the instruction set for the intermediate representation.
//! Instructions are low-level operations that map directly to machine operations.

use super::{IrFunctionId, IrGlobalId, IrId, IrSourceLocation, IrType, IrValue};
use serde::{Deserialize, Serialize};

/// Reference to a specific field within a struct type.
/// Attached to GEP instructions for typed field access in backends.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructFieldRef {
    /// Name of the struct/class type (e.g., "Body", "Constraint")
    pub struct_name: String,
    /// Name of the field (e.g., "x", "mass")
    pub field_name: String,
    /// Field index in the struct layout (0-based, index 0 = __type_id header)
    pub field_index: u32,
}

/// Ownership transfer semantics for values
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OwnershipMode {
    /// Transfer ownership (move semantics)
    Move,
    /// Borrow immutably (shared reference)
    BorrowImmutable,
    /// Borrow mutably (exclusive reference)
    BorrowMutable,
    /// Copy value (for Copy types)
    Copy,
    /// Clone value (explicit deep copy)
    Clone,
}

/// Lifetime annotation for borrowed values
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LifetimeId(pub u32);

impl LifetimeId {
    pub fn static_lifetime() -> Self {
        Self(0)
    }

    pub fn new(id: u32) -> Self {
        Self(id)
    }
}

/// IR instruction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IrInstruction {
    // === Value Operations ===
    /// Load constant value
    Const { dest: IrId, value: IrValue },

    /// Copy value from one register to another (for Copy types)
    Copy { dest: IrId, src: IrId },

    /// SSA barrier — value-flow identity that no MIR pass and no
    /// Cranelift egraph step may look through. Lowers to a real
    /// backend instruction (`select_spectre_guard` on Cranelift,
    /// `llvm.assume(true) + asm("" :: "r"(src))` on LLVM, a plain
    /// value-copy in the interpreter).
    ///
    /// Semantics: `dest = src` value-wise; the barrier itself has
    /// NO side effect. It exists only as an opacity hint to
    /// optimizers — Copy / Add-0 / store-load / branch-phi all got
    /// folded by some pass in the past; this primitive is built to
    /// survive.
    ///
    /// Used at effectful CallDirect result sites where the value is
    /// consumed across a block boundary, to dodge the
    /// "Cranelift egraph elaborates effectful instruction" class
    /// of bugs (see bugs_sys_getenv_in_ctor_residual and
    /// bugs_sys_call_in_generation_method) and the downstream
    /// "block cascade to Unreachable" symptom that's same family.
    ///
    /// CopyPropagationPass MUST treat this as opaque (do not
    /// substitute `dest` uses with `src`) — see scalar_replacement
    /// and copy_propagation pass implementations.
    SsaBarrier { dest: IrId, src: IrId, ty: IrType },

    /// Move value (transfer ownership)
    Move { dest: IrId, src: IrId },

    /// Mark a value as moved without producing a destination.
    /// Used by the ownership analysis to signal that `src` has been transferred
    /// away. Backends may treat this as a hint / no-op, but later `CheckLive`
    /// uses of the same id are expected to diagnose a use-after-move.
    MarkMoved { src: IrId },

    /// Assert that `src` is still live (not moved) at this program point.
    /// Carries an `IrSourceLocation` so the borrow-checker / runtime can report
    /// where the offending use occurred (mirrors how `DebugLoc` carries
    /// location info).
    CheckLive {
        src: IrId,
        location: IrSourceLocation,
    },

    /// Create immutable borrow/reference
    BorrowImmutable {
        dest: IrId,
        src: IrId,
        lifetime: LifetimeId,
    },

    /// Create mutable borrow/reference
    BorrowMutable {
        dest: IrId,
        src: IrId,
        lifetime: LifetimeId,
    },

    /// Clone value (explicit deep copy)
    Clone { dest: IrId, src: IrId },

    /// End borrow/drop reference (for explicit lifetime management)
    EndBorrow { borrow: IrId },

    /// Load value from memory
    Load { dest: IrId, ptr: IrId, ty: IrType },

    /// Store value to memory
    Store {
        ptr: IrId,
        value: IrId,
        /// Optional store type (inferred from value's register type).
        /// Enables backends to emit correctly-typed stores (f64 stays in XMM).
        #[serde(default)]
        store_ty: Option<IrType>,
    },

    /// Load value from a global variable
    LoadGlobal {
        dest: IrId,
        global_id: IrGlobalId,
        ty: IrType,
    },

    /// Store value to a global variable
    StoreGlobal { global_id: IrGlobalId, value: IrId },

    // === Arithmetic Operations ===
    /// Binary arithmetic operation
    BinOp {
        dest: IrId,
        op: BinaryOp,
        left: IrId,
        right: IrId,
    },

    /// Unary operation
    UnOp {
        dest: IrId,
        op: UnaryOp,
        operand: IrId,
    },

    /// Compare operation
    Cmp {
        dest: IrId,
        op: CompareOp,
        left: IrId,
        right: IrId,
    },

    // === Control Flow ===
    /// Unconditional jump
    Jump { target: IrId },

    /// Conditional branch
    Branch {
        condition: IrId,
        true_target: IrId,
        false_target: IrId,
    },

    /// Switch/jump table
    Switch {
        value: IrId,
        default_target: IrId,
        cases: Vec<(IrValue, IrId)>,
    },

    /// Direct function call (callee known at compile time)
    CallDirect {
        dest: Option<IrId>,
        func_id: IrFunctionId,
        args: Vec<IrId>,
        /// Ownership mode for each argument (Move, BorrowImmutable, BorrowMutable, Copy, Clone)
        arg_ownership: Vec<OwnershipMode>,
        /// Type arguments for generic function instantiation (empty if non-generic)
        type_args: Vec<IrType>,
        /// Whether this is a tail call (can reuse caller's stack frame)
        #[serde(default)]
        is_tail_call: bool,
    },

    /// Indirect function call (callee computed at runtime)
    CallIndirect {
        dest: Option<IrId>,
        func_ptr: IrId,
        args: Vec<IrId>,
        signature: IrType,
        /// Ownership mode for each argument
        arg_ownership: Vec<OwnershipMode>,
        /// Whether this is a tail call (can reuse caller's stack frame)
        #[serde(default)]
        is_tail_call: bool,
    },

    /// Return from function
    Return { value: Option<IrId> },

    // === Memory Operations ===
    /// Allocate memory
    Alloc {
        dest: IrId,
        ty: IrType,
        count: Option<IrId>,
    },

    /// Free memory
    Free { ptr: IrId },

    /// Get element pointer (GEP)
    GetElementPtr {
        dest: IrId,
        ptr: IrId,
        indices: Vec<IrId>,
        ty: IrType,
        /// Optional struct context for typed field access.
        /// Enables C backend to emit `body->x` instead of pointer arithmetic,
        /// and LLVM backend to add TBAA metadata.
        #[serde(default)]
        struct_context: Option<StructFieldRef>,
    },

    /// Memory copy
    MemCopy { dest: IrId, src: IrId, size: IrId },

    /// Memory set
    MemSet { dest: IrId, value: IrId, size: IrId },

    // === Type Operations ===
    /// Type cast
    Cast {
        dest: IrId,
        src: IrId,
        from_ty: IrType,
        to_ty: IrType,
    },

    /// Bitcast (reinterpret bits)
    BitCast { dest: IrId, src: IrId, ty: IrType },

    // === Exception Handling ===
    /// Throw exception
    Throw { exception: IrId },

    /// Begin exception handler
    LandingPad {
        dest: IrId,
        ty: IrType,
        clauses: Vec<LandingPadClause>,
    },

    /// Resume exception propagation
    Resume { exception: IrId },

    // === Special Operations ===
    /// Phi node for SSA form
    Phi {
        dest: IrId,
        incoming: Vec<(IrId, IrId)>, // (value, predecessor block)
    },

    /// Select (ternary) operation
    Select {
        dest: IrId,
        condition: IrId,
        true_val: IrId,
        false_val: IrId,
    },

    /// Extract value from aggregate
    ExtractValue {
        dest: IrId,
        aggregate: IrId,
        indices: Vec<u32>,
    },

    /// Insert value into aggregate
    InsertValue {
        dest: IrId,
        aggregate: IrId,
        value: IrId,
        indices: Vec<u32>,
    },

    /// Create a closure (allocates environment, captures variables)
    MakeClosure {
        dest: IrId,
        func_id: IrFunctionId,
        captured_values: Vec<IrId>, // Values to capture in environment
    },

    /// Extract the function pointer from a closure
    ClosureFunc { dest: IrId, closure: IrId },

    /// Extract the environment pointer from a closure
    ClosureEnv { dest: IrId, closure: IrId },

    // === Union/Sum Type Operations ===
    /// Create a union value with discriminant
    CreateUnion {
        dest: IrId,
        discriminant: u32,
        value: IrId,
        ty: IrType,
    },

    /// Extract discriminant from union
    ExtractDiscriminant { dest: IrId, union_val: IrId },

    /// Extract value from union variant
    ExtractUnionValue {
        dest: IrId,
        union_val: IrId,
        discriminant: u32,
        value_ty: IrType,
    },

    // === Struct Operations ===
    /// Create struct from field values
    CreateStruct {
        dest: IrId,
        ty: IrType,
        fields: Vec<IrId>,
    },

    // === Pointer Operations ===
    /// Pointer arithmetic: ptr + offset
    PtrAdd {
        dest: IrId,
        ptr: IrId,
        offset: IrId,
        ty: IrType,
    },

    // === Special Values ===
    /// Undefined value (uninitialized)
    Undef { dest: IrId, ty: IrType },

    /// Function reference (for function pointers)
    FunctionRef { dest: IrId, func_id: IrFunctionId },

    /// Panic/abort execution
    Panic { message: Option<String> },

    /// Debug location marker
    DebugLoc { location: IrSourceLocation },

    /// Inline assembly
    InlineAsm {
        dest: Option<IrId>,
        asm: String,
        inputs: Vec<(String, IrId)>,
        outputs: Vec<(String, IrType)>,
        clobbers: Vec<String>,
    },

    // === SIMD Vector Operations ===
    /// Load contiguous elements into a SIMD vector
    VectorLoad {
        dest: IrId,
        ptr: IrId,
        vec_ty: IrType, // Must be IrType::Vector
    },

    /// Store SIMD vector to contiguous memory
    VectorStore {
        ptr: IrId,
        value: IrId,
        vec_ty: IrType,
    },

    /// SIMD binary operation (element-wise)
    VectorBinOp {
        dest: IrId,
        op: BinaryOp,
        left: IrId,
        right: IrId,
        vec_ty: IrType,
    },

    /// Broadcast scalar to all vector lanes (splat)
    VectorSplat {
        dest: IrId,
        scalar: IrId,
        vec_ty: IrType,
    },

    /// Extract scalar element from vector
    VectorExtract {
        dest: IrId,
        vector: IrId,
        index: u8, // Lane index (0-15 for 16-element vectors)
        /// Lane element type. Carried explicitly so backends that don't track
        /// SSA value types (the wasm backend) know whether to emit f32x4 vs
        /// i32x4/i8x16/... lane ops — register-type inference is lost across
        /// inlining, so this must travel with the instruction.
        elem_ty: IrType,
    },

    /// Insert scalar into vector lane
    VectorInsert {
        dest: IrId,
        vector: IrId,
        scalar: IrId,
        index: u8,
        /// Lane element type (see VectorExtract::elem_ty).
        elem_ty: IrType,
    },

    /// Horizontal reduction (e.g., sum all elements)
    VectorReduce {
        dest: IrId,
        op: BinaryOp, // Add, Mul, And, Or, Xor for reductions
        vector: IrId,
        /// Lane element type (see VectorExtract::elem_ty).
        elem_ty: IrType,
    },

    /// SIMD element-wise unary operation (sqrt, abs, neg, ceil, floor, round)
    VectorUnaryOp {
        dest: IrId,
        op: VectorUnaryOpKind,
        operand: IrId,
        vec_ty: IrType,
    },

    /// SIMD element-wise min/max
    VectorMinMax {
        dest: IrId,
        op: VectorMinMaxKind,
        left: IrId,
        right: IrId,
        vec_ty: IrType,
    },

    /// Fused widening dot-product accumulate: `dest = acc + dot(a, b)`, where
    /// `a`/`b` are 16-lane i8 vectors and the 16 i8×i8 products are summed in
    /// groups of 4 into the 4 i32 lanes of `dest` (and added to `acc`). This is
    /// the quantized-matmul primitive: it lowers to a single AArch64 SDOT
    /// (`vdotq_s32`) on native and to `i32x4.relaxed_dot_i8x16_i7x16_add_s` on
    /// wasm (which wasmtime 47 lowers to SDOT). For cross-runtime-deterministic
    /// results the `b` operand must stay in i7 range (−64..63) — quant weights
    /// (4-bit nibbles 0..15, 6-bit 0..63) satisfy this.
    VectorDot {
        dest: IrId,
        acc: IrId,
        a: IrId,
        b: IrId,
    },

    // === Atomic memory operations (sequentially consistent) ===
    /// Atomic load. dest = *ptr.
    AtomicLoad { dest: IrId, ptr: IrId, ty: IrType },
    /// Atomic store. *ptr = value.
    AtomicStore { ptr: IrId, value: IrId, ty: IrType },
    /// Atomic RMW. dest = old *ptr; *ptr = op(old, value).
    AtomicRmw {
        dest: IrId,
        op: AtomicRmwOp,
        ptr: IrId,
        value: IrId,
        ty: IrType,
    },
    /// Atomic compare-and-swap. dest = old *ptr; if old==expected then *ptr=replacement.
    AtomicCas {
        dest: IrId,
        ptr: IrId,
        expected: IrId,
        replacement: IrId,
        ty: IrType,
    },
}

/// Binary operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinaryOp {
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Rem,

    // Bitwise
    And,
    Or,
    Xor,
    Shl,
    Shr,
    Ushr,

    // Floating point
    FAdd,
    FSub,
    FMul,
    FDiv,
    FRem,
}

/// Unary operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnaryOp {
    // Arithmetic
    Neg,

    // Bitwise
    Not,

    // Floating point
    FNeg,
}

/// SIMD vector unary operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VectorUnaryOpKind {
    Sqrt,
    Abs,
    Neg,
    Ceil,
    Floor,
    Trunc,
    Round, // nearest
}

/// SIMD vector min/max operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VectorMinMaxKind {
    Min,
    Max,
}

/// Atomic read-modify-write operation kind (SC ordering implied)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AtomicRmwOp {
    Add,
    Sub,
    And,
    Or,
    Xor,
    Xchg,
}

/// Comparison operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompareOp {
    // Integer comparisons
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,

    // Unsigned comparisons
    ULt,
    ULe,
    UGt,
    UGe,

    // Floating point comparisons
    FEq,
    FNe,
    FLt,
    FLe,
    FGt,
    FGe,

    // Floating point ordered/unordered
    FOrd,
    FUno,
}

/// Landing pad clause for exception handling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LandingPadClause {
    /// Catch specific exception type
    Catch(IrType),
    /// Filter exceptions
    Filter(Vec<IrType>),
}

impl IrInstruction {
    /// Get the destination register if this instruction produces a value
    pub fn dest(&self) -> Option<IrId> {
        match self {
            IrInstruction::Const { dest, .. } |
            IrInstruction::Copy { dest, .. } |
            IrInstruction::SsaBarrier { dest, .. } |
            IrInstruction::Move { dest, .. } |
            IrInstruction::BorrowImmutable { dest, .. } |
            IrInstruction::BorrowMutable { dest, .. } |
            IrInstruction::Clone { dest, .. } |
            IrInstruction::Load { dest, .. } |
            IrInstruction::LoadGlobal { dest, .. } |
            IrInstruction::BinOp { dest, .. } |
            IrInstruction::UnOp { dest, .. } |
            IrInstruction::Cmp { dest, .. } |
            IrInstruction::Alloc { dest, .. } |
            IrInstruction::GetElementPtr { dest, .. } |
            IrInstruction::MemCopy { dest, .. } |
            IrInstruction::MemSet { dest, .. } |
            IrInstruction::Cast { dest, .. } |
            IrInstruction::BitCast { dest, .. } |
            IrInstruction::LandingPad { dest, .. } |
            IrInstruction::Phi { dest, .. } |
            IrInstruction::Select { dest, .. } |
            IrInstruction::ExtractValue { dest, .. } |
            IrInstruction::InsertValue { dest, .. } |
            IrInstruction::MakeClosure { dest, .. } |
            IrInstruction::ClosureFunc { dest, .. } |
            IrInstruction::ClosureEnv { dest, .. } |
            IrInstruction::CreateUnion { dest, .. } |
            IrInstruction::ExtractDiscriminant { dest, .. } |
            IrInstruction::ExtractUnionValue { dest, .. } |
            IrInstruction::CreateStruct { dest, .. } |
            IrInstruction::PtrAdd { dest, .. } |
            IrInstruction::Undef { dest, .. } |
            IrInstruction::FunctionRef { dest, .. } |
            // Vector instructions
            IrInstruction::VectorLoad { dest, .. } |
            IrInstruction::VectorBinOp { dest, .. } |
            IrInstruction::VectorSplat { dest, .. } |
            IrInstruction::VectorExtract { dest, .. } |
            IrInstruction::VectorInsert { dest, .. } |
            IrInstruction::VectorReduce { dest, .. } |
            IrInstruction::VectorUnaryOp { dest, .. } |
            IrInstruction::VectorMinMax { dest, .. } |
            IrInstruction::VectorDot { dest, .. } |
            // Atomic instructions (AtomicStore omitted → falls to _ => None)
            IrInstruction::AtomicLoad { dest, .. } |
            IrInstruction::AtomicRmw { dest, .. } |
            IrInstruction::AtomicCas { dest, .. } => Some(*dest),

            IrInstruction::CallDirect { dest, .. } |
            IrInstruction::CallIndirect { dest, .. } |
            IrInstruction::InlineAsm { dest, .. } => *dest,

            _ => None,
        }
    }

    /// Replace the destination register of this instruction
    pub fn replace_dest(&mut self, new_dest: IrId) {
        match self {
            IrInstruction::Const { dest, .. }
            | IrInstruction::Copy { dest, .. }
            | IrInstruction::SsaBarrier { dest, .. }
            | IrInstruction::Move { dest, .. }
            | IrInstruction::BorrowImmutable { dest, .. }
            | IrInstruction::BorrowMutable { dest, .. }
            | IrInstruction::Clone { dest, .. }
            | IrInstruction::Load { dest, .. }
            | IrInstruction::LoadGlobal { dest, .. }
            | IrInstruction::MemCopy { dest, .. }
            | IrInstruction::MemSet { dest, .. }
            | IrInstruction::BinOp { dest, .. }
            | IrInstruction::UnOp { dest, .. }
            | IrInstruction::Cmp { dest, .. }
            | IrInstruction::Alloc { dest, .. }
            | IrInstruction::GetElementPtr { dest, .. }
            | IrInstruction::Cast { dest, .. }
            | IrInstruction::BitCast { dest, .. }
            | IrInstruction::LandingPad { dest, .. }
            | IrInstruction::Phi { dest, .. }
            | IrInstruction::Select { dest, .. }
            | IrInstruction::ExtractValue { dest, .. }
            | IrInstruction::InsertValue { dest, .. }
            | IrInstruction::MakeClosure { dest, .. }
            | IrInstruction::ClosureFunc { dest, .. }
            | IrInstruction::ClosureEnv { dest, .. }
            | IrInstruction::CreateUnion { dest, .. }
            | IrInstruction::ExtractDiscriminant { dest, .. }
            | IrInstruction::ExtractUnionValue { dest, .. }
            | IrInstruction::CreateStruct { dest, .. }
            | IrInstruction::PtrAdd { dest, .. }
            | IrInstruction::Undef { dest, .. }
            | IrInstruction::FunctionRef { dest, .. }
            | IrInstruction::VectorLoad { dest, .. }
            | IrInstruction::VectorBinOp { dest, .. }
            | IrInstruction::VectorSplat { dest, .. }
            | IrInstruction::VectorExtract { dest, .. }
            | IrInstruction::VectorInsert { dest, .. }
            | IrInstruction::VectorReduce { dest, .. }
            | IrInstruction::VectorUnaryOp { dest, .. }
            | IrInstruction::VectorMinMax { dest, .. }
            | IrInstruction::VectorDot { dest, .. }
            | IrInstruction::AtomicLoad { dest, .. }
            | IrInstruction::AtomicRmw { dest, .. }
            | IrInstruction::AtomicCas { dest, .. } => *dest = new_dest,

            IrInstruction::CallDirect { dest, .. }
            | IrInstruction::CallIndirect { dest, .. }
            | IrInstruction::InlineAsm { dest, .. } => *dest = Some(new_dest),

            _ => {}
        }
    }

    /// Get all registers used by this instruction
    pub fn uses(&self) -> Vec<IrId> {
        match self {
            IrInstruction::Copy { src, .. } => vec![*src],
            IrInstruction::SsaBarrier { src, .. } => vec![*src],
            IrInstruction::Load { ptr, .. } => vec![*ptr],
            IrInstruction::Store { ptr, value, .. } => vec![*ptr, *value],
            IrInstruction::BinOp { left, right, .. } => vec![*left, *right],
            IrInstruction::UnOp { operand, .. } => vec![*operand],
            IrInstruction::Cmp { left, right, .. } => vec![*left, *right],
            IrInstruction::Branch { condition, .. } => vec![*condition],
            IrInstruction::Switch { value, .. } => vec![*value],
            IrInstruction::CallDirect { args, .. } => {
                // CallDirect uses function ID (not a register), so only args are register uses
                args.clone()
            }
            IrInstruction::CallIndirect { func_ptr, args, .. } => {
                let mut uses = vec![*func_ptr];
                uses.extend(args);
                uses
            }
            IrInstruction::Return { value } => value.map(|v| vec![v]).unwrap_or_default(),
            IrInstruction::Alloc { count, .. } => count.map(|c| vec![c]).unwrap_or_default(),
            IrInstruction::Free { ptr } => vec![*ptr],
            IrInstruction::GetElementPtr { ptr, indices, .. } => {
                let mut uses = vec![*ptr];
                uses.extend(indices);
                uses
            }
            IrInstruction::MemCopy { dest, src, size } => vec![*dest, *src, *size],
            IrInstruction::MemSet { dest, value, size } => vec![*dest, *value, *size],
            IrInstruction::Cast { src, .. } | IrInstruction::BitCast { src, .. } => vec![*src],
            IrInstruction::Throw { exception } => vec![*exception],
            IrInstruction::Resume { exception } => vec![*exception],
            IrInstruction::Phi { incoming, .. } => incoming.iter().map(|(val, _)| *val).collect(),
            IrInstruction::Select {
                condition,
                true_val,
                false_val,
                ..
            } => {
                vec![*condition, *true_val, *false_val]
            }
            IrInstruction::ExtractValue { aggregate, .. } => vec![*aggregate],
            IrInstruction::InsertValue {
                aggregate, value, ..
            } => vec![*aggregate, *value],
            IrInstruction::InlineAsm { inputs, .. } => inputs.iter().map(|(_, id)| *id).collect(),
            // Vector instructions
            IrInstruction::VectorLoad { ptr, .. } => vec![*ptr],
            IrInstruction::VectorStore { ptr, value, .. } => vec![*ptr, *value],
            IrInstruction::VectorBinOp { left, right, .. } => vec![*left, *right],
            IrInstruction::VectorSplat { scalar, .. } => vec![*scalar],
            IrInstruction::VectorExtract { vector, .. } => vec![*vector],
            IrInstruction::VectorInsert { vector, scalar, .. } => vec![*vector, *scalar],
            IrInstruction::VectorReduce { vector, .. } => vec![*vector],
            IrInstruction::VectorUnaryOp { operand, .. } => vec![*operand],
            IrInstruction::VectorMinMax { left, right, .. } => vec![*left, *right],
            IrInstruction::VectorDot { acc, a, b, .. } => vec![*acc, *a, *b],
            // Atomic instructions
            IrInstruction::AtomicLoad { ptr, .. } => vec![*ptr],
            IrInstruction::AtomicStore { ptr, value, .. } => vec![*ptr, *value],
            IrInstruction::AtomicRmw { ptr, value, .. } => vec![*ptr, *value],
            IrInstruction::AtomicCas {
                ptr,
                expected,
                replacement,
                ..
            } => vec![*ptr, *expected, *replacement],
            // Move/borrow instructions
            IrInstruction::Move { src, .. } => vec![*src],
            IrInstruction::MarkMoved { src } => vec![*src],
            IrInstruction::CheckLive { src, .. } => vec![*src],
            IrInstruction::BorrowImmutable { src, .. } => vec![*src],
            IrInstruction::BorrowMutable { src, .. } => vec![*src],
            IrInstruction::Clone { src, .. } => vec![*src],
            IrInstruction::EndBorrow { borrow } => vec![*borrow],
            // Closure instructions
            IrInstruction::MakeClosure {
                captured_values, ..
            } => captured_values.clone(),
            IrInstruction::ClosureFunc { closure, .. } => vec![*closure],
            IrInstruction::ClosureEnv { closure, .. } => vec![*closure],
            // Union/struct instructions
            IrInstruction::CreateUnion { value, .. } => vec![*value],
            IrInstruction::ExtractDiscriminant { union_val, .. } => vec![*union_val],
            IrInstruction::ExtractUnionValue { union_val, .. } => vec![*union_val],
            IrInstruction::CreateStruct { fields, .. } => fields.clone(),
            // Pointer arithmetic
            IrInstruction::PtrAdd { ptr, offset, .. } => vec![*ptr, *offset],
            // Global variable access
            IrInstruction::LoadGlobal { .. } => vec![], // No register uses, just global_id
            IrInstruction::StoreGlobal { value, .. } => vec![*value],
            // No uses for these
            IrInstruction::Const { .. }
            | IrInstruction::Jump { .. }
            | IrInstruction::LandingPad { .. }
            | IrInstruction::Undef { .. }
            | IrInstruction::FunctionRef { .. }
            | IrInstruction::Panic { .. }
            | IrInstruction::DebugLoc { .. } => vec![],
        }
    }

    /// Check if this is a terminator instruction
    pub fn is_terminator(&self) -> bool {
        matches!(
            self,
            IrInstruction::Jump { .. }
                | IrInstruction::Branch { .. }
                | IrInstruction::Switch { .. }
                | IrInstruction::Return { .. }
                | IrInstruction::Throw { .. }
                | IrInstruction::Resume { .. }
        )
    }

    /// Check if this instruction has side effects
    pub fn has_side_effects(&self) -> bool {
        matches!(
            self,
            IrInstruction::Alloc { .. }
                | IrInstruction::Store { .. }
                | IrInstruction::StoreGlobal { .. }
                | IrInstruction::CallDirect { .. }
                | IrInstruction::CallIndirect { .. }
                | IrInstruction::Free { .. }
                | IrInstruction::MemCopy { .. }
                | IrInstruction::MemSet { .. }
                | IrInstruction::Throw { .. }
                | IrInstruction::InlineAsm { .. }
                | IrInstruction::AtomicStore { .. }
                | IrInstruction::AtomicRmw { .. }
                | IrInstruction::AtomicCas { .. }
                | IrInstruction::AtomicLoad { .. }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instruction_properties() {
        let add = IrInstruction::BinOp {
            dest: IrId::new(1),
            op: BinaryOp::Add,
            left: IrId::new(2),
            right: IrId::new(3),
        };

        assert_eq!(add.dest(), Some(IrId::new(1)));
        assert_eq!(add.uses(), vec![IrId::new(2), IrId::new(3)]);
        assert!(!add.is_terminator());
        assert!(!add.has_side_effects());

        let ret = IrInstruction::Return {
            value: Some(IrId::new(1)),
        };

        assert!(ret.is_terminator());
        assert_eq!(ret.uses(), vec![IrId::new(1)]);
    }
}
