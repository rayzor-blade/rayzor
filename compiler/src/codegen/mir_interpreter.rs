//! MIR Register-Based Interpreter
//!
//! Provides instant startup by directly interpreting MIR without compilation.
//! Performance: ~5-10x native speed (suitable for cold paths and development)
//!
//! ## Design
//! - **Register-based execution** (not stack-based) - matches MIR's SSA form
//! - Direct mapping from IrId to interpreter registers
//! - Support for all MIR instructions
//! - FFI calls to runtime functions
//! - GC-safe (no raw pointers in interpreter state)
//!
//! ## Why Register-Based?
//! 1. MIR is already in SSA form with explicit registers (IrId)
//! 2. ~30% faster than stack-based (see Lua 5.x vs 4.x benchmarks)
//! 3. Fewer instructions needed (no push/pop overhead)
//! 4. Direct 1:1 mapping from MIR to interpreter state
//!
//! ## NaN Boxing (Performance Optimization)
//! Primitive values are stored using NaN boxing - a technique that packs
//! type tag and value into a 64-bit IEEE 754 double. This provides:
//! - 8 bytes per value instead of 16-32 bytes
//! - Copy semantics (no Clone overhead)
//! - Better cache locality
//!
//! Layout: `[0x7FFC_0000_0000_0000 | tag:4 | payload:48]`
//! - When exponent bits != 0x7FF, it's a regular f64
//! - Otherwise, tag bits identify: Ptr, I32, Bool, Null, etc.

use crate::ir::{
    BinaryOp, CompareOp, FunctionKind, IrBasicBlock, IrBlockId, IrExternFunction, IrFunction,
    IrFunctionId, IrFunctionSignature, IrId, IrInstruction, IrModule, IrTerminator, IrType,
    IrValue, UnaryOp,
};
// SmallVec disabled temporarily - reverting to Vec to debug Linux CI heap corruption
use std::collections::HashMap;
use std::sync::Arc;

// ============================================================================
// NaN Boxing Implementation
// ============================================================================

/// NaN-boxed value for efficient interpreter storage
///
/// Uses IEEE 754 NaN boxing to pack type tags and values into 64 bits.
/// This eliminates heap allocation for primitives and enables Copy semantics.
///
/// ## Encoding
/// - Regular doubles are stored as-is (when exponent != 0x7FF)
/// - Special values use the NaN space: `NAN_TAG | type_tag | payload`
///
/// ## Type Tags (in bits 48-51)
/// - 0x0: Pointer (48-bit address)
/// - 0x1: I32 (32-bit signed integer)
/// - 0x2: I64 (64-bit integer, needs 2 words for full range, we use payload)
/// - 0x3: Bool (0 or 1)
/// - 0x4: Null
/// - 0x5: Void
/// - 0x6: Function ID (32-bit)
/// - 0x7: Heap object (index into heap table)
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct NanBoxedValue(u64);

impl std::fmt::Debug for NanBoxedValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_f64() {
            write!(f, "F64({})", self.as_f64())
        } else {
            let tag = self.tag();
            match tag {
                NanBoxedValue::TAG_PTR => write!(f, "Ptr({:#x})", self.payload()),
                NanBoxedValue::TAG_I32 => write!(f, "I32({})", self.as_i32()),
                NanBoxedValue::TAG_I64 => write!(f, "I64({})", self.as_i64_lossy()),
                NanBoxedValue::TAG_BOOL => write!(f, "Bool({})", self.as_bool()),
                NanBoxedValue::TAG_NULL => write!(f, "Null"),
                NanBoxedValue::TAG_VOID => write!(f, "Void"),
                NanBoxedValue::TAG_FUNC => write!(f, "Func({})", self.payload() as u32),
                NanBoxedValue::TAG_HEAP => write!(f, "Heap({})", self.payload() as u32),
                _ => write!(f, "Unknown({:#x})", self.0),
            }
        }
    }
}

impl NanBoxedValue {
    /// NaN tag prefix - all special values have this prefix
    /// This is a quiet NaN that's distinct from any valid double
    // NaN box base: quiet NaN with no bits set in tag area (bits 48-51)
    // 0x7FF8 ensures this is in the quiet NaN range without conflicting with tags
    const NAN_TAG: u64 = 0x7FF8_0000_0000_0000;

    /// Mask for extracting the tag (bits 48-50, excluding the quiet NaN bit at 51)
    /// This gives us 8 possible tag values (0-7), which is enough for our types
    const TAG_MASK: u64 = 0x0007_0000_0000_0000;

    /// Mask for extracting the payload (bits 0-47)
    const PAYLOAD_MASK: u64 = 0x0000_FFFF_FFFF_FFFF;

    /// Type tags
    const TAG_PTR: u64 = 0x0000_0000_0000_0000; // Pointer (48-bit)
    const TAG_I32: u64 = 0x0001_0000_0000_0000; // 32-bit signed int
    const TAG_I64: u64 = 0x0002_0000_0000_0000; // 64-bit int (lossy, 48 bits)
    const TAG_BOOL: u64 = 0x0003_0000_0000_0000; // Boolean
    const TAG_NULL: u64 = 0x0004_0000_0000_0000; // Null
    const TAG_VOID: u64 = 0x0005_0000_0000_0000; // Void
    const TAG_FUNC: u64 = 0x0006_0000_0000_0000; // Function ID
    const TAG_HEAP: u64 = 0x0007_0000_0000_0000; // Heap object index

    /// Create a void value
    #[inline(always)]
    pub const fn void() -> Self {
        Self(Self::NAN_TAG | Self::TAG_VOID)
    }

    /// Create a null value
    #[inline(always)]
    pub const fn null() -> Self {
        Self(Self::NAN_TAG | Self::TAG_NULL)
    }

    /// Create from f64 (stored directly as IEEE 754 double)
    #[inline(always)]
    pub fn from_f64(v: f64) -> Self {
        Self(v.to_bits())
    }

    /// Create from f32 (converted to f64)
    #[inline(always)]
    pub fn from_f32(v: f32) -> Self {
        Self::from_f64(v as f64)
    }

    /// Create from i32
    #[inline(always)]
    pub fn from_i32(v: i32) -> Self {
        Self(Self::NAN_TAG | Self::TAG_I32 | (v as u32 as u64))
    }

    /// Create from i64 (lossy - only 48 bits preserved)
    #[inline(always)]
    pub fn from_i64(v: i64) -> Self {
        Self(Self::NAN_TAG | Self::TAG_I64 | ((v as u64) & Self::PAYLOAD_MASK))
    }

    /// Create from bool
    #[inline(always)]
    pub fn from_bool(v: bool) -> Self {
        Self(Self::NAN_TAG | Self::TAG_BOOL | (v as u64))
    }

    /// Create from raw pointer
    #[inline(always)]
    pub fn from_ptr(ptr: usize) -> Self {
        // On 64-bit systems, only lower 48 bits of pointers are typically used
        Self(Self::NAN_TAG | Self::TAG_PTR | ((ptr as u64) & Self::PAYLOAD_MASK))
    }

    /// Create from function ID
    #[inline(always)]
    pub fn from_func_id(id: u32) -> Self {
        Self(Self::NAN_TAG | Self::TAG_FUNC | (id as u64))
    }

    /// Create from heap object index
    #[inline(always)]
    pub fn from_heap_index(index: u32) -> Self {
        Self(Self::NAN_TAG | Self::TAG_HEAP | (index as u64))
    }

    /// Check if this is a regular f64 (not a tagged value)
    #[inline(always)]
    pub fn is_f64(&self) -> bool {
        // Check if exponent bits are NOT all 1s (0x7FF)
        // OR if it's a canonical NaN but not our tagged format
        let exp_bits = (self.0 >> 52) & 0x7FF;
        if exp_bits != 0x7FF {
            return true;
        }
        // It's some kind of NaN - check if it's our tagged format
        // Check if the high bits indicate a quiet NaN (used for tagged values)
        // Valid f64s have different bit patterns in the exponent+mantissa area
        (self.0 & 0x7FF8_0000_0000_0000) != Self::NAN_TAG
    }

    /// Get the tag bits
    #[inline(always)]
    fn tag(&self) -> u64 {
        self.0 & Self::TAG_MASK
    }

    /// Get the payload bits
    #[inline(always)]
    fn payload(&self) -> u64 {
        self.0 & Self::PAYLOAD_MASK
    }

    /// Extract as f64
    #[inline(always)]
    pub fn as_f64(&self) -> f64 {
        f64::from_bits(self.0)
    }

    /// Extract as i32
    #[inline(always)]
    pub fn as_i32(&self) -> i32 {
        (self.0 & 0xFFFF_FFFF) as i32
    }

    /// Extract as i64 (lossy - sign extended from 48 bits)
    #[inline(always)]
    pub fn as_i64_lossy(&self) -> i64 {
        let payload = self.payload();
        // Sign extend from 48 bits
        if payload & 0x0000_8000_0000_0000 != 0 {
            (payload | 0xFFFF_0000_0000_0000) as i64
        } else {
            payload as i64
        }
    }

    /// Extract as bool
    #[inline(always)]
    pub fn as_bool(&self) -> bool {
        (self.0 & 1) != 0
    }

    /// Extract as pointer
    #[inline(always)]
    pub fn as_ptr(&self) -> usize {
        self.payload() as usize
    }

    /// Extract as function ID
    #[inline(always)]
    pub fn as_func_id(&self) -> u32 {
        self.payload() as u32
    }

    /// Extract as heap object index
    #[inline(always)]
    pub fn as_heap_index(&self) -> u32 {
        self.payload() as u32
    }

    /// Check if null
    #[inline(always)]
    pub fn is_null(&self) -> bool {
        self.0 == (Self::NAN_TAG | Self::TAG_NULL)
    }

    /// Check if void
    #[inline(always)]
    pub fn is_void(&self) -> bool {
        self.0 == (Self::NAN_TAG | Self::TAG_VOID)
    }

    /// Check if this is a heap object
    #[inline(always)]
    pub fn is_heap(&self) -> bool {
        !self.is_f64() && self.tag() == Self::TAG_HEAP
    }

    /// Check if this is a pointer
    #[inline(always)]
    pub fn is_ptr(&self) -> bool {
        !self.is_f64() && self.tag() == Self::TAG_PTR
    }

    /// Check if this is an i32
    #[inline(always)]
    pub fn is_i32(&self) -> bool {
        !self.is_f64() && self.tag() == Self::TAG_I32
    }

    /// Check if this is a bool
    #[inline(always)]
    pub fn is_bool(&self) -> bool {
        !self.is_f64() && self.tag() == Self::TAG_BOOL
    }
}

impl Default for NanBoxedValue {
    fn default() -> Self {
        Self::void()
    }
}

// ============================================================================
// NaN-Boxed Binary Operations (Ultra-Fast Path)
// ============================================================================

impl NanBoxedValue {
    /// Fast binary operation on NaN-boxed values
    /// Returns None if types don't match (fallback to slow path needed)
    #[inline(always)]
    pub fn binary_op(self, op: BinaryOp, other: Self) -> Option<Self> {
        // Fast path: both are i32 (most common in Haxe)
        if self.is_i32() && other.is_i32() {
            let l = self.as_i32();
            let r = other.as_i32();
            return Some(match op {
                BinaryOp::Add => Self::from_i32(l.wrapping_add(r)),
                BinaryOp::Sub => Self::from_i32(l.wrapping_sub(r)),
                BinaryOp::Mul => Self::from_i32(l.wrapping_mul(r)),
                BinaryOp::Div => {
                    if r != 0 {
                        Self::from_i32(l / r)
                    } else {
                        return None;
                    }
                }
                BinaryOp::Rem => {
                    if r != 0 {
                        Self::from_i32(l % r)
                    } else {
                        return None;
                    }
                }
                BinaryOp::And => Self::from_i32(l & r),
                BinaryOp::Or => Self::from_i32(l | r),
                BinaryOp::Xor => Self::from_i32(l ^ r),
                BinaryOp::Shl => Self::from_i32(l << (r & 31)),
                BinaryOp::Shr => Self::from_i32(l >> (r & 31)),
                BinaryOp::Ushr => Self::from_i32(((l as u32) >> (r & 31)) as i32),
                _ => return None,
            });
        }

        // Fast path: both are f64
        if self.is_f64() && other.is_f64() {
            let l = self.as_f64();
            let r = other.as_f64();
            return Some(match op {
                BinaryOp::Add | BinaryOp::FAdd => Self::from_f64(l + r),
                BinaryOp::Sub | BinaryOp::FSub => Self::from_f64(l - r),
                BinaryOp::Mul | BinaryOp::FMul => Self::from_f64(l * r),
                BinaryOp::Div | BinaryOp::FDiv => Self::from_f64(l / r),
                BinaryOp::Rem | BinaryOp::FRem => Self::from_f64(l % r),
                _ => return None,
            });
        }

        None // Fallback to slow path
    }

    /// Fast comparison operation on NaN-boxed values
    #[inline(always)]
    pub fn compare_op(self, op: CompareOp, other: Self) -> Option<bool> {
        // Fast path: both are i32
        if self.is_i32() && other.is_i32() {
            let l = self.as_i32();
            let r = other.as_i32();
            return Some(match op {
                CompareOp::Eq => l == r,
                CompareOp::Ne => l != r,
                CompareOp::Lt => l < r,
                CompareOp::Le => l <= r,
                CompareOp::Gt => l > r,
                CompareOp::Ge => l >= r,
                _ => return None,
            });
        }

        // Fast path: both are f64
        if self.is_f64() && other.is_f64() {
            let l = self.as_f64();
            let r = other.as_f64();
            return Some(match op {
                CompareOp::Eq | CompareOp::FEq => l == r,
                CompareOp::Ne | CompareOp::FNe => l != r,
                CompareOp::Lt | CompareOp::FLt => l < r,
                CompareOp::Le | CompareOp::FLe => l <= r,
                CompareOp::Gt | CompareOp::FGt => l > r,
                CompareOp::Ge | CompareOp::FGe => l >= r,
                _ => return None,
            });
        }

        // Fast path: both are bool (for logical comparisons)
        if self.is_bool() && other.is_bool() {
            let l = self.as_bool();
            let r = other.as_bool();
            return Some(match op {
                CompareOp::Eq => l == r,
                CompareOp::Ne => l != r,
                _ => return None,
            });
        }

        None // Fallback to slow path
    }

    /// Fast unary operation on NaN-boxed values
    #[inline(always)]
    pub fn unary_op(self, op: UnaryOp) -> Option<Self> {
        // Fast path: i32
        if self.is_i32() {
            let v = self.as_i32();
            return Some(match op {
                UnaryOp::Neg => Self::from_i32(v.wrapping_neg()),
                UnaryOp::Not => Self::from_i32(!v),
                _ => return None,
            });
        }

        // Fast path: f64
        if self.is_f64() {
            let v = self.as_f64();
            return Some(match op {
                UnaryOp::Neg | UnaryOp::FNeg => Self::from_f64(-v),
                _ => return None,
            });
        }

        // Fast path: bool (uses Not for logical negation)
        if self.is_bool() {
            let v = self.as_bool();
            return Some(match op {
                UnaryOp::Not => Self::from_bool(!v),
                _ => return None,
            });
        }

        None // Fallback to slow path
    }
}

/// Heap-allocated objects for complex types that don't fit in NaN boxing
#[derive(Clone, Debug)]
pub enum HeapObject {
    /// String value
    String(String),
    /// Array of values
    Array(Vec<NanBoxedValue>),
    /// Struct with fields
    Struct(Vec<NanBoxedValue>),
    /// Full i64 when 48 bits isn't enough
    I64(i64),
    /// Full u64
    U64(u64),
}

/// Heap storage for complex objects
#[derive(Debug, Default)]
pub struct ObjectHeap {
    objects: Vec<HeapObject>,
}

impl ObjectHeap {
    pub fn new() -> Self {
        Self {
            objects: Vec::new(),
        }
    }

    /// Allocate a new object and return its index
    pub fn alloc(&mut self, obj: HeapObject) -> u32 {
        let index = self.objects.len() as u32;
        self.objects.push(obj);
        index
    }

    /// Get an object by index
    pub fn get(&self, index: u32) -> Option<&HeapObject> {
        self.objects.get(index as usize)
    }

    /// Get a mutable object by index
    pub fn get_mut(&mut self, index: u32) -> Option<&mut HeapObject> {
        self.objects.get_mut(index as usize)
    }
}

// ============================================================================
// Opcode-Based Dispatch (Performance Optimization)
// ============================================================================

/// Instruction opcode for dispatch table lookup
///
/// This enum mirrors IrInstruction variants but as a simple discriminant.
/// Used for computed goto-style dispatch via function pointer table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Opcode {
    Const = 0,
    Copy = 1,
    Move = 2,
    BinOp = 3,
    UnOp = 4,
    Cmp = 5,
    Load = 6,
    Store = 7,
    Alloc = 8,
    Free = 9,
    GetElementPtr = 10,
    PtrAdd = 11,
    CallDirect = 12,
    CallIndirect = 13,
    Cast = 14,
    BitCast = 15,
    CreateStruct = 16,
    ExtractValue = 17,
    InsertValue = 18,
    CreateUnion = 19,
    ExtractDiscriminant = 20,
    ExtractUnionValue = 21,
    Select = 22,
    FunctionRef = 23,
    MakeClosure = 24,
    ClosureFunc = 25,
    ClosureEnv = 26,
    BorrowImmutable = 27,
    BorrowMutable = 28,
    Clone = 29,
    EndBorrow = 30,
    MemCopy = 31,
    MemSet = 32,
    Undef = 33,
    Panic = 34,
    DebugLoc = 35,
    Phi = 36,
    Jump = 37,
    Branch = 38,
    Switch = 39,
    Return = 40,
    Throw = 41,
    LandingPad = 42,
    Resume = 43,
    InlineAsm = 44,
    // SIMD Vector operations
    VectorLoad = 45,
    VectorStore = 46,
    VectorBinOp = 47,
    VectorSplat = 48,
    VectorExtract = 49,
    VectorInsert = 50,
    VectorReduce = 51,
    VectorUnaryOp = 52,
    VectorMinMax = 53,
    // Global variable access
    LoadGlobal = 54,
    StoreGlobal = 55,
    // Sentinel for table size
    _Count = 56,
}

impl Opcode {
    /// Get opcode from IrInstruction (fast discriminant extraction)
    #[inline(always)]
    pub fn from_instruction(instr: &IrInstruction) -> Self {
        match instr {
            IrInstruction::Const { .. } => Opcode::Const,
            IrInstruction::Copy { .. } => Opcode::Copy,
            IrInstruction::Move { .. } => Opcode::Move,
            IrInstruction::BinOp { .. } => Opcode::BinOp,
            IrInstruction::UnOp { .. } => Opcode::UnOp,
            IrInstruction::Cmp { .. } => Opcode::Cmp,
            IrInstruction::Load { .. } => Opcode::Load,
            IrInstruction::Store { .. } => Opcode::Store,
            IrInstruction::Alloc { .. } => Opcode::Alloc,
            IrInstruction::Free { .. } => Opcode::Free,
            IrInstruction::GetElementPtr { .. } => Opcode::GetElementPtr,
            IrInstruction::PtrAdd { .. } => Opcode::PtrAdd,
            IrInstruction::CallDirect { .. } => Opcode::CallDirect,
            IrInstruction::CallIndirect { .. } => Opcode::CallIndirect,
            IrInstruction::Cast { .. } => Opcode::Cast,
            IrInstruction::BitCast { .. } => Opcode::BitCast,
            IrInstruction::CreateStruct { .. } => Opcode::CreateStruct,
            IrInstruction::ExtractValue { .. } => Opcode::ExtractValue,
            IrInstruction::InsertValue { .. } => Opcode::InsertValue,
            IrInstruction::CreateUnion { .. } => Opcode::CreateUnion,
            IrInstruction::ExtractDiscriminant { .. } => Opcode::ExtractDiscriminant,
            IrInstruction::ExtractUnionValue { .. } => Opcode::ExtractUnionValue,
            IrInstruction::Select { .. } => Opcode::Select,
            IrInstruction::FunctionRef { .. } => Opcode::FunctionRef,
            IrInstruction::MakeClosure { .. } => Opcode::MakeClosure,
            IrInstruction::ClosureFunc { .. } => Opcode::ClosureFunc,
            IrInstruction::ClosureEnv { .. } => Opcode::ClosureEnv,
            IrInstruction::BorrowImmutable { .. } => Opcode::BorrowImmutable,
            IrInstruction::BorrowMutable { .. } => Opcode::BorrowMutable,
            IrInstruction::Clone { .. } => Opcode::Clone,
            IrInstruction::EndBorrow { .. } => Opcode::EndBorrow,
            IrInstruction::MemCopy { .. } => Opcode::MemCopy,
            IrInstruction::MemSet { .. } => Opcode::MemSet,
            IrInstruction::Undef { .. } => Opcode::Undef,
            IrInstruction::Panic { .. } => Opcode::Panic,
            IrInstruction::DebugLoc { .. } => Opcode::DebugLoc,
            IrInstruction::Phi { .. } => Opcode::Phi,
            IrInstruction::Jump { .. } => Opcode::Jump,
            IrInstruction::Branch { .. } => Opcode::Branch,
            IrInstruction::Switch { .. } => Opcode::Switch,
            IrInstruction::Return { .. } => Opcode::Return,
            IrInstruction::Throw { .. } => Opcode::Throw,
            IrInstruction::LandingPad { .. } => Opcode::LandingPad,
            IrInstruction::Resume { .. } => Opcode::Resume,
            IrInstruction::InlineAsm { .. } => Opcode::InlineAsm,
            // SIMD Vector operations
            IrInstruction::VectorLoad { .. } => Opcode::VectorLoad,
            IrInstruction::VectorStore { .. } => Opcode::VectorStore,
            IrInstruction::VectorBinOp { .. } => Opcode::VectorBinOp,
            IrInstruction::VectorSplat { .. } => Opcode::VectorSplat,
            IrInstruction::VectorExtract { .. } => Opcode::VectorExtract,
            IrInstruction::VectorInsert { .. } => Opcode::VectorInsert,
            IrInstruction::VectorReduce { .. } => Opcode::VectorReduce,
            IrInstruction::VectorUnaryOp { .. } => Opcode::VectorUnaryOp,
            IrInstruction::VectorMinMax { .. } => Opcode::VectorMinMax,
            // Global variable access
            IrInstruction::LoadGlobal { .. } => Opcode::LoadGlobal,
            IrInstruction::StoreGlobal { .. } => Opcode::StoreGlobal,
        }
    }
}

/// Pre-decoded instruction for faster dispatch
///
/// Stores the opcode separately from the instruction data,
/// allowing direct dispatch table lookup without pattern matching.
#[derive(Clone)]
pub struct DecodedInstruction {
    pub opcode: Opcode,
    pub instr: IrInstruction,
}

impl DecodedInstruction {
    #[inline(always)]
    pub fn new(instr: IrInstruction) -> Self {
        Self {
            opcode: Opcode::from_instruction(&instr),
            instr,
        }
    }
}

/// Pre-decoded basic block for faster interpretation
pub struct DecodedBlock {
    pub instructions: Vec<DecodedInstruction>,
    pub terminator: IrTerminator,
}

impl DecodedBlock {
    pub fn from_block(block: &IrBasicBlock) -> Self {
        Self {
            instructions: block
                .instructions
                .iter()
                .map(|i| DecodedInstruction::new(i.clone()))
                .collect(),
            terminator: block.terminator.clone(),
        }
    }
}

// ============================================================================
// Legacy InterpValue (kept for compatibility, will be phased out)
// ============================================================================

/// MIR interpreter value (boxed for GC safety)
#[derive(Clone, Debug)]
pub enum InterpValue {
    Void,
    Bool(bool),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    F32(f32),
    F64(f64),
    Ptr(usize), // Raw pointer (for FFI)
    Null,
    /// String value (owned)
    String(String),
    /// Array value
    Array(Vec<InterpValue>),
    /// Struct value (fields by index)
    Struct(Vec<InterpValue>),
    /// Function reference
    Function(IrFunctionId),
}

impl Default for InterpValue {
    fn default() -> Self {
        InterpValue::Void
    }
}

impl InterpValue {
    /// Convert to boolean (for conditionals)
    pub fn to_bool(&self) -> Result<bool, InterpError> {
        match self {
            InterpValue::Bool(b) => Ok(*b),
            InterpValue::I32(n) => Ok(*n != 0),
            InterpValue::I64(n) => Ok(*n != 0),
            InterpValue::Ptr(p) => Ok(*p != 0),
            // Void and Null are falsy
            InterpValue::Void => Ok(false),
            InterpValue::Null => Ok(false),
            _ => Err(InterpError::TypeError(format!(
                "Cannot convert {:?} to bool",
                self
            ))),
        }
    }

    /// Convert to i64 (for integer operations)
    pub fn to_i64(&self) -> Result<i64, InterpError> {
        match self {
            InterpValue::I8(n) => Ok(*n as i64),
            InterpValue::I16(n) => Ok(*n as i64),
            InterpValue::I32(n) => Ok(*n as i64),
            InterpValue::I64(n) => Ok(*n),
            InterpValue::U8(n) => Ok(*n as i64),
            InterpValue::U16(n) => Ok(*n as i64),
            InterpValue::U32(n) => Ok(*n as i64),
            InterpValue::U64(n) => Ok(*n as i64),
            InterpValue::Bool(b) => Ok(if *b { 1 } else { 0 }),
            InterpValue::Ptr(p) => Ok(*p as i64),
            // Float to int truncates the decimal part
            InterpValue::F32(f) => Ok(*f as i64),
            InterpValue::F64(f) => Ok(*f as i64),
            // Void and Null convert to 0 (common convention in many languages)
            InterpValue::Void => Ok(0),
            InterpValue::Null => Ok(0),
            _ => Err(InterpError::TypeError(format!(
                "Cannot convert {:?} to i64",
                self
            ))),
        }
    }

    /// Encode a function reference the same way NaN-boxed registers store it.
    fn to_function_bits(&self) -> Option<i64> {
        match self {
            InterpValue::Function(id) => Some(NanBoxedValue::from_func_id(id.0).0 as i64),
            _ => None,
        }
    }

    /// Convert to f64 (for floating point operations)
    pub fn to_f64(&self) -> Result<f64, InterpError> {
        match self {
            InterpValue::F32(n) => Ok(*n as f64),
            InterpValue::F64(n) => Ok(*n),
            InterpValue::I32(n) => Ok(*n as f64),
            InterpValue::I64(n) => Ok(*n as f64),
            // Void and Null convert to 0.0
            InterpValue::Void => Ok(0.0),
            InterpValue::Null => Ok(0.0),
            _ => Err(InterpError::TypeError(format!(
                "Cannot convert {:?} to f64",
                self
            ))),
        }
    }

    /// Check if this value is a floating-point type
    pub fn is_float(&self) -> bool {
        matches!(self, InterpValue::F32(_) | InterpValue::F64(_))
    }

    /// Convert to usize (for pointer operations)
    pub fn to_usize(&self) -> Result<usize, InterpError> {
        match self {
            InterpValue::Ptr(p) => Ok(*p),
            InterpValue::I64(n) => Ok(*n as usize),
            InterpValue::U64(n) => Ok(*n as usize),
            InterpValue::I32(n) => Ok(*n as usize),
            InterpValue::U32(n) => Ok(*n as usize),
            // Void and Null convert to 0 (null pointer)
            InterpValue::Void => Ok(0),
            InterpValue::Null => Ok(0),
            _ => Err(InterpError::TypeError(format!(
                "Cannot convert {:?} to usize",
                self
            ))),
        }
    }

    /// Convert InterpValue to NanBoxedValue for efficient storage
    ///
    /// Primitives are packed directly; complex types go to the heap.
    pub fn to_nan_boxed(&self, heap: &mut ObjectHeap) -> NanBoxedValue {
        match self {
            InterpValue::Void => NanBoxedValue::void(),
            InterpValue::Null => NanBoxedValue::null(),
            InterpValue::Bool(b) => NanBoxedValue::from_bool(*b),
            InterpValue::I8(n) => NanBoxedValue::from_i32(*n as i32),
            InterpValue::I16(n) => NanBoxedValue::from_i32(*n as i32),
            InterpValue::I32(n) => NanBoxedValue::from_i32(*n),
            InterpValue::I64(n) => {
                // Check if it fits in 48 bits
                if *n >= -(1i64 << 47) && *n < (1i64 << 47) {
                    NanBoxedValue::from_i64(*n)
                } else {
                    // Full precision i64 needs heap allocation
                    let idx = heap.alloc(HeapObject::I64(*n));
                    NanBoxedValue::from_heap_index(idx)
                }
            }
            InterpValue::U8(n) => NanBoxedValue::from_i32(*n as i32),
            InterpValue::U16(n) => NanBoxedValue::from_i32(*n as i32),
            InterpValue::U32(n) => NanBoxedValue::from_i64(*n as i64),
            InterpValue::U64(n) => {
                // Check if it fits in 48 bits
                if *n < (1u64 << 48) {
                    NanBoxedValue::from_i64(*n as i64)
                } else {
                    let idx = heap.alloc(HeapObject::U64(*n));
                    NanBoxedValue::from_heap_index(idx)
                }
            }
            InterpValue::F32(n) => NanBoxedValue::from_f32(*n),
            InterpValue::F64(n) => NanBoxedValue::from_f64(*n),
            InterpValue::Ptr(p) => NanBoxedValue::from_ptr(*p),
            InterpValue::String(s) => {
                let idx = heap.alloc(HeapObject::String(s.clone()));
                NanBoxedValue::from_heap_index(idx)
            }
            InterpValue::Array(arr) => {
                // Convert array elements recursively
                let boxed: Vec<NanBoxedValue> = arr.iter().map(|v| v.to_nan_boxed(heap)).collect();
                let idx = heap.alloc(HeapObject::Array(boxed));
                NanBoxedValue::from_heap_index(idx)
            }
            InterpValue::Struct(fields) => {
                let boxed: Vec<NanBoxedValue> =
                    fields.iter().map(|v| v.to_nan_boxed(heap)).collect();
                let idx = heap.alloc(HeapObject::Struct(boxed));
                NanBoxedValue::from_heap_index(idx)
            }
            InterpValue::Function(id) => NanBoxedValue::from_func_id(id.0),
        }
    }

    /// Convert NanBoxedValue back to InterpValue
    ///
    /// This is used for compatibility with existing code paths.
    pub fn from_nan_boxed(val: NanBoxedValue, heap: &ObjectHeap) -> Self {
        if val.is_f64() {
            return InterpValue::F64(val.as_f64());
        }

        if val.is_null() {
            return InterpValue::Null;
        }

        if val.is_void() {
            return InterpValue::Void;
        }

        if val.is_i32() {
            return InterpValue::I32(val.as_i32());
        }

        if val.is_bool() {
            return InterpValue::Bool(val.as_bool());
        }

        if val.is_ptr() {
            return InterpValue::Ptr(val.as_ptr());
        }

        if val.is_heap() {
            let idx = val.as_heap_index();
            if let Some(obj) = heap.get(idx) {
                return match obj {
                    HeapObject::String(s) => InterpValue::String(s.clone()),
                    HeapObject::Array(arr) => {
                        let values: Vec<InterpValue> = arr
                            .iter()
                            .map(|v| InterpValue::from_nan_boxed(*v, heap))
                            .collect();
                        InterpValue::Array(values)
                    }
                    HeapObject::Struct(fields) => {
                        let values: Vec<InterpValue> = fields
                            .iter()
                            .map(|v| InterpValue::from_nan_boxed(*v, heap))
                            .collect();
                        InterpValue::Struct(values)
                    }
                    HeapObject::I64(n) => InterpValue::I64(*n),
                    HeapObject::U64(n) => InterpValue::U64(*n),
                };
            }
        }

        // Fallback for function IDs and other tagged values
        let tag = val.0 & NanBoxedValue::TAG_MASK;
        if tag == NanBoxedValue::TAG_FUNC {
            return InterpValue::Function(IrFunctionId(val.as_func_id()));
        }

        // For i64 tagged values
        if tag == NanBoxedValue::TAG_I64 {
            return InterpValue::I64(val.as_i64_lossy());
        }

        // Default fallback
        InterpValue::Void
    }
}

/// Register file for a single function execution frame
/// Uses NaN boxing for efficient value storage (8 bytes per register, Copy semantics)
#[derive(Debug)]
struct RegisterFile {
    /// Register values indexed by IrId.as_u32()
    /// Pre-allocated to function's max register count for speed
    /// Uses NanBoxedValue for 3-5x performance improvement over InterpValue
    registers: Vec<NanBoxedValue>,
}

impl RegisterFile {
    fn new(register_count: usize) -> Self {
        Self {
            registers: vec![NanBoxedValue::void(); register_count],
        }
    }

    /// Get register value (O(1) access, Copy semantics - no clone needed!)
    #[inline(always)]
    fn get(&self, reg: IrId) -> NanBoxedValue {
        self.registers[reg.as_u32() as usize]
    }

    /// Set register value
    #[inline(always)]
    fn set(&mut self, reg: IrId, value: NanBoxedValue) {
        let idx = reg.as_u32() as usize;
        if idx >= self.registers.len() {
            self.registers.resize(idx + 1, NanBoxedValue::void());
        }
        self.registers[idx] = value;
    }

    /// Set from InterpValue (conversion at boundary)
    #[inline(always)]
    fn set_from_interp(&mut self, reg: IrId, value: InterpValue, heap: &mut ObjectHeap) {
        self.set(reg, value.to_nan_boxed(heap));
    }

    /// Get as InterpValue (conversion at boundary)
    #[inline(always)]
    fn get_as_interp(&self, reg: IrId, heap: &ObjectHeap) -> InterpValue {
        InterpValue::from_nan_boxed(self.get(reg), heap)
    }
}

/// Interpreter execution frame (one per function call)
#[derive(Debug)]
struct InterpreterFrame {
    function_id: IrFunctionId,
    registers: RegisterFile, // Register-based storage (fast O(1) access)
    current_block: IrBlockId,
    prev_block: Option<IrBlockId>, // For phi node resolution
}

/// Result of executing a terminator (uses NanBoxedValue for efficiency)
enum TerminatorResult {
    Continue(IrBlockId),
    Return(NanBoxedValue),
}

/// MIR Register-Based Interpreter
///
/// Uses NaN boxing for efficient register storage and supports
/// optional pre-decoded block caching for faster repeated execution.
pub struct MirInterpreter {
    /// Runtime function pointers (for FFI calls)
    runtime_symbols: HashMap<String, *const u8>,

    /// Call stack (frames with register files)
    stack: Vec<InterpreterFrame>,

    /// Maximum stack depth (prevent stack overflow)
    max_stack_depth: usize,

    /// Heap memory for allocations (simple bump allocator)
    heap: Vec<u8>,

    /// Next heap allocation offset
    heap_offset: usize,

    /// Object heap for NaN-boxed complex objects (strings, arrays, structs)
    object_heap: ObjectHeap,

    /// Pre-decoded block cache for faster execution
    /// Key: (function_id.0, block_id.0)
    decoded_blocks: HashMap<(u32, u32), DecodedBlock>,

    /// Whether to use cached decoded blocks
    use_decoded_cache: bool,

    /// Track heap allocations from system allocator for proper free
    /// Maps pointer address to allocation size for correct deallocation
    heap_allocations: HashMap<usize, usize>,

    /// Iteration counter for hot loop detection
    /// When exceeded, interpreter signals need for JIT compilation
    iteration_count: u64,

    /// Maximum iterations before triggering JIT bailout
    /// Default: 10000 (signals hot code that should be JIT compiled)
    max_iterations: u64,

    /// Global variable store - maps global IDs to their values
    /// Used for static class fields and module-level variables
    global_store: HashMap<crate::ir::IrGlobalId, NanBoxedValue>,
}

// Safety: MirInterpreter can be sent across threads
// The runtime_symbols are function pointers that remain valid
unsafe impl Send for MirInterpreter {}
unsafe impl Sync for MirInterpreter {}

impl MirInterpreter {
    fn widen_value_for_object_slot(&self, val: InterpValue) -> InterpValue {
        match val {
            InterpValue::Bool(_)
            | InterpValue::I8(_)
            | InterpValue::I16(_)
            | InterpValue::I32(_)
            | InterpValue::U8(_)
            | InterpValue::U16(_)
            | InterpValue::U32(_) => InterpValue::I64(val.to_i64().unwrap_or(0)),
            other => other,
        }
    }

    /// Create a new interpreter
    pub fn new() -> Self {
        Self {
            runtime_symbols: HashMap::new(),
            stack: Vec::new(),
            max_stack_depth: 1000,
            heap: vec![0u8; 1024 * 1024], // 1MB heap
            heap_offset: 0,
            object_heap: ObjectHeap::new(),
            decoded_blocks: HashMap::new(),
            use_decoded_cache: true, // Enable by default for performance
            heap_allocations: HashMap::new(),
            iteration_count: 0,
            max_iterations: 10_000, // Trigger JIT bailout after 10k iterations
            global_store: HashMap::new(),
        }
    }

    /// Create a new interpreter with decoded block caching disabled
    pub fn new_without_cache() -> Self {
        let mut interp = Self::new();
        interp.use_decoded_cache = false;
        interp
    }

    /// Enable or disable decoded block caching
    pub fn set_use_decoded_cache(&mut self, enabled: bool) {
        self.use_decoded_cache = enabled;
        if !enabled {
            self.decoded_blocks.clear();
        }
    }

    /// Clear the decoded block cache
    pub fn clear_decoded_cache(&mut self) {
        self.decoded_blocks.clear();
    }

    /// Create interpreter with runtime symbols for FFI calls
    pub fn with_symbols(symbols: &[(&str, *const u8)]) -> Self {
        let mut interp = Self::new();
        for (name, ptr) in symbols {
            interp.runtime_symbols.insert(name.to_string(), *ptr);
        }
        interp
    }

    /// Register a runtime symbol
    pub fn register_symbol(&mut self, name: &str, ptr: *const u8) {
        self.runtime_symbols.insert(name.to_string(), ptr);
    }

    /// Set the maximum iterations before triggering JIT bailout
    pub fn set_max_iterations(&mut self, max: u64) {
        self.max_iterations = max;
    }

    /// Reset the iteration counter (called when switching to JIT)
    pub fn reset_iteration_count(&mut self) {
        self.iteration_count = 0;
    }

    /// Get current iteration count
    pub fn iteration_count(&self) -> u64 {
        self.iteration_count
    }

    /// Calculate the maximum register ID used in a function
    fn calculate_register_count(function: &IrFunction) -> usize {
        let mut max_reg = 0usize;

        // Check parameters
        for param in &function.signature.parameters {
            max_reg = max_reg.max(param.reg.as_u32() as usize + 1);
        }

        // Check all instructions for destination registers
        for block in function.cfg.blocks.values() {
            for instr in &block.instructions {
                if let Some(dest) = instr.dest() {
                    max_reg = max_reg.max(dest.as_u32() as usize + 1);
                }
            }
            // Check phi nodes
            for phi in &block.phi_nodes {
                max_reg = max_reg.max(phi.dest.as_u32() as usize + 1);
            }
        }

        // Add some headroom for temporaries
        max_reg + 16
    }

    /// Execute a function and return the result
    pub fn execute(
        &mut self,
        module: &IrModule,
        func_id: IrFunctionId,
        args: Vec<InterpValue>,
    ) -> Result<InterpValue, InterpError> {
        let function = module
            .functions
            .get(&func_id)
            .ok_or(InterpError::FunctionNotFound(func_id))?;

        // Check stack depth
        if self.stack.len() >= self.max_stack_depth {
            return Err(InterpError::StackOverflow);
        }

        // Pre-calculate register count for efficient allocation
        let register_count = Self::calculate_register_count(function);

        // Create new frame with pre-allocated register file
        let mut frame = InterpreterFrame {
            function_id: func_id,
            registers: RegisterFile::new(register_count),
            current_block: function.cfg.entry_block,
            prev_block: None,
        };

        // Bind arguments to parameter registers - convert InterpValue to NanBoxedValue at boundary
        for (i, param) in function.signature.parameters.iter().enumerate() {
            if let Some(arg) = args.get(i) {
                let boxed = arg.to_nan_boxed(&mut self.object_heap);
                frame.registers.set(param.reg, boxed);
            }
        }

        self.stack.push(frame);

        // Execute blocks until return (uses NanBoxedValue internally)
        let result = self.execute_function(module, function)?;

        self.stack.pop();

        // Convert NanBoxedValue back to InterpValue at the boundary
        Ok(InterpValue::from_nan_boxed(result, &self.object_heap))
    }

    /// Get the current frame (mutable)
    fn current_frame_mut(&mut self) -> &mut InterpreterFrame {
        self.stack.last_mut().expect("No active frame")
    }

    /// Get the current frame
    fn current_frame(&self) -> &InterpreterFrame {
        self.stack.last().expect("No active frame")
    }

    /// Execute function body using NaN-boxed registers (internal, fast path)
    fn execute_function(
        &mut self,
        module: &IrModule,
        function: &IrFunction,
    ) -> Result<NanBoxedValue, InterpError> {
        loop {
            // Hot loop detection: check if we've exceeded iteration threshold
            self.iteration_count += 1;
            if self.iteration_count >= self.max_iterations {
                // Reset counter for next function call
                self.iteration_count = 0;
                // Signal that this function should be JIT compiled
                return Err(InterpError::JitBailout(function.id));
            }

            let block_id = self.current_frame().current_block;
            let block = function
                .cfg
                .blocks
                .get(&block_id)
                .ok_or(InterpError::BlockNotFound(block_id))?;

            // Execute phi nodes first (using prev_block for value selection)
            self.execute_phi_nodes(block)?;

            // Execute instructions
            for instr in &block.instructions {
                self.execute_instruction(module, function, instr)?;
            }

            // Execute terminator
            match self.execute_terminator(module, function, &block.terminator)? {
                TerminatorResult::Continue(next_block) => {
                    let frame = self.current_frame_mut();
                    frame.prev_block = Some(frame.current_block);
                    frame.current_block = next_block;
                }
                TerminatorResult::Return(value) => {
                    return Ok(value);
                }
            }
        }
    }

    /// Execute phi nodes at the beginning of a block (uses NanBoxedValue - Copy, no clone!)
    fn execute_phi_nodes(&mut self, block: &IrBasicBlock) -> Result<(), InterpError> {
        let prev_block = self.current_frame().prev_block;

        // Collect phi values first to avoid interference
        // NanBoxedValue is Copy, so this is very efficient
        let mut phi_values: Vec<(IrId, NanBoxedValue)> = Vec::new();

        for phi in &block.phi_nodes {
            // Find the value from the previous block
            if let Some(prev) = prev_block {
                for (pred_block, value_reg) in &phi.incoming {
                    if *pred_block == prev {
                        // NanBoxedValue is Copy - no clone needed!
                        let value = self.current_frame().registers.get(*value_reg);
                        phi_values.push((phi.dest, value));
                        break;
                    }
                }
            }
        }

        // Apply phi values
        for (dest, value) in phi_values {
            self.current_frame_mut().registers.set(dest, value);
        }

        Ok(())
    }

    /// Execute a single instruction using NaN-boxed register operations
    ///
    /// Hot paths use NanBoxedValue directly (Copy semantics, no allocation).
    /// Complex operations fall back to InterpValue conversion.
    fn execute_instruction(
        &mut self,
        module: &IrModule,
        function: &IrFunction,
        instr: &IrInstruction,
    ) -> Result<(), InterpError> {
        match instr {
            // === Value Operations (NaN-boxed fast path) ===
            IrInstruction::Const { dest, value } => {
                // Convert IR value to NanBoxedValue (fast for primitives)
                let val = self.ir_value_to_nanboxed(value)?;
                self.current_frame_mut().registers.set(*dest, val);
            }

            IrInstruction::Copy { dest, src } => {
                // NanBoxedValue is Copy - no clone needed!
                let val = self.current_frame().registers.get(*src);
                self.current_frame_mut().registers.set(*dest, val);
            }

            IrInstruction::Move { dest, src } => {
                // NanBoxedValue is Copy - move is same as copy
                let val = self.current_frame().registers.get(*src);
                self.current_frame_mut().registers.set(*dest, val);
            }

            // === Arithmetic Operations (NaN-boxed fast path) ===
            IrInstruction::BinOp {
                dest,
                op,
                left,
                right,
            } => {
                let l = self.current_frame().registers.get(*left);
                let r = self.current_frame().registers.get(*right);

                // Try fast NaN-boxed path first
                if let Some(result) = l.binary_op(*op, r) {
                    self.current_frame_mut().registers.set(*dest, result);
                } else {
                    // Fall back to InterpValue slow path
                    let l_interp = InterpValue::from_nan_boxed(l, &self.object_heap);
                    let r_interp = InterpValue::from_nan_boxed(r, &self.object_heap);
                    let result = self.eval_binary_op(*op, l_interp, r_interp)?;
                    let boxed = result.to_nan_boxed(&mut self.object_heap);
                    self.current_frame_mut().registers.set(*dest, boxed);
                }
            }

            IrInstruction::UnOp { dest, op, operand } => {
                let val = self.current_frame().registers.get(*operand);

                // Try fast NaN-boxed path first
                if let Some(result) = val.unary_op(*op) {
                    self.current_frame_mut().registers.set(*dest, result);
                } else {
                    // Fall back to InterpValue slow path
                    let val_interp = InterpValue::from_nan_boxed(val, &self.object_heap);
                    let result = self.eval_unary_op(*op, val_interp)?;
                    let boxed = result.to_nan_boxed(&mut self.object_heap);
                    self.current_frame_mut().registers.set(*dest, boxed);
                }
            }

            IrInstruction::Cmp {
                dest,
                op,
                left,
                right,
            } => {
                let l = self.current_frame().registers.get(*left);
                let r = self.current_frame().registers.get(*right);

                // Try fast NaN-boxed path first
                if let Some(result) = l.compare_op(*op, r) {
                    self.current_frame_mut()
                        .registers
                        .set(*dest, NanBoxedValue::from_bool(result));
                } else {
                    // Fall back to InterpValue slow path
                    let l_interp = InterpValue::from_nan_boxed(l, &self.object_heap);
                    let r_interp = InterpValue::from_nan_boxed(r, &self.object_heap);
                    let result = self.eval_compare_op(*op, l_interp, r_interp)?;
                    self.current_frame_mut()
                        .registers
                        .set(*dest, NanBoxedValue::from_bool(result));
                }
            }

            // === Memory Operations (convert at boundary for compatibility) ===
            IrInstruction::Load { dest, ptr, ty } => {
                let ptr_val = self.current_frame().registers.get(*ptr);
                let ptr_interp = InterpValue::from_nan_boxed(ptr_val, &self.object_heap);
                let result = self.load_from_ptr(ptr_interp, ty)?;
                let boxed = result.to_nan_boxed(&mut self.object_heap);
                self.current_frame_mut().registers.set(*dest, boxed);
            }

            IrInstruction::Store { ptr, value, .. } => {
                let ptr_val = self.current_frame().registers.get(*ptr);
                let val = self.current_frame().registers.get(*value);
                let ptr_interp = InterpValue::from_nan_boxed(ptr_val, &self.object_heap);
                let mut val_interp = InterpValue::from_nan_boxed(val, &self.object_heap);
                let store_to_object_slot = function
                    .register_types
                    .get(ptr)
                    .and_then(|ty| match ty {
                        IrType::Ptr(inner) => {
                            Some(!matches!(inner.as_ref(), IrType::U8 | IrType::I8))
                        }
                        _ => None,
                    })
                    .unwrap_or(false);

                if store_to_object_slot {
                    val_interp = self.widen_value_for_object_slot(val_interp);
                }
                self.store_to_ptr(ptr_interp, val_interp)?;
            }

            IrInstruction::Alloc { dest, ty, count } => {
                let size = ty.size();
                let count_val = if let Some(c) = count {
                    let val = self.current_frame().registers.get(*c);
                    if val.is_i32() {
                        val.as_i32() as usize
                    } else {
                        InterpValue::from_nan_boxed(val, &self.object_heap).to_i64()? as usize
                    }
                } else {
                    1
                };
                let total_size = size * count_val;
                let ptr = self.alloc_heap(total_size)?;
                self.current_frame_mut()
                    .registers
                    .set(*dest, NanBoxedValue::from_ptr(ptr));
            }

            IrInstruction::Free { ptr } => {
                // Free heap-allocated memory
                let ptr_val = self.current_frame().registers.get(*ptr);
                if ptr_val.is_ptr() {
                    let ptr_addr = ptr_val.as_ptr();
                    self.free_heap(ptr_addr);
                }
            }

            IrInstruction::GetElementPtr {
                dest,
                ptr,
                indices,
                ty,
                ..
            } => {
                let ptr_val = self.current_frame().registers.get(*ptr);
                let base_ptr = if ptr_val.is_ptr() {
                    ptr_val.as_ptr()
                } else {
                    InterpValue::from_nan_boxed(ptr_val, &self.object_heap).to_usize()?
                };
                let mut offset = 0usize;
                let elem_size = match ty {
                    IrType::Ptr(inner) => match inner.as_ref() {
                        IrType::U8 | IrType::I8 => 1,
                        _ => 8,
                    },
                    _ => 8,
                };

                // Calculate offset based on indices and type
                for idx in indices {
                    let idx_val = self.current_frame().registers.get(*idx);
                    let idx_int = if idx_val.is_i32() {
                        idx_val.as_i32() as usize
                    } else {
                        InterpValue::from_nan_boxed(idx_val, &self.object_heap).to_i64()? as usize
                    };
                    offset += idx_int * elem_size;
                }

                self.current_frame_mut()
                    .registers
                    .set(*dest, NanBoxedValue::from_ptr(base_ptr + offset));
            }

            IrInstruction::PtrAdd {
                dest,
                ptr,
                offset,
                ty,
            } => {
                let ptr_val = self.current_frame().registers.get(*ptr);
                let base_ptr = if ptr_val.is_ptr() {
                    ptr_val.as_ptr()
                } else {
                    InterpValue::from_nan_boxed(ptr_val, &self.object_heap).to_usize()?
                };
                let offset_val = self.current_frame().registers.get(*offset);
                let offset_int = if offset_val.is_i32() {
                    offset_val.as_i32() as usize
                } else {
                    InterpValue::from_nan_boxed(offset_val, &self.object_heap).to_i64()? as usize
                };
                let elem_size = match ty {
                    IrType::Ptr(inner) => inner.size(),
                    _ => ty.size(),
                };
                self.current_frame_mut().registers.set(
                    *dest,
                    NanBoxedValue::from_ptr(base_ptr + offset_int * elem_size),
                );
            }

            // === Function Calls (convert at boundary for FFI compatibility) ===
            IrInstruction::CallDirect {
                dest,
                func_id,
                args,
                ..
            } => {
                // Collect argument values - convert from NanBoxedValue to InterpValue
                let arg_values: Vec<InterpValue> = args
                    .iter()
                    .map(|a| {
                        InterpValue::from_nan_boxed(
                            self.current_frame().registers.get(*a),
                            &self.object_heap,
                        )
                    })
                    .collect();

                // Check if it's a user function or extern
                let result = if let Some(func) = module.functions.get(func_id) {
                    // Check the function kind - ExternC functions need FFI
                    if func.kind == FunctionKind::ExternC {
                        // Extern function - use FFI with the function's signature
                        self.call_ffi_for_function(func, &arg_values)?
                    } else if func.cfg.blocks.is_empty() {
                        // Function with no blocks - try extern_functions or FFI
                        if let Some(extern_fn) = module.extern_functions.get(func_id) {
                            self.call_extern_with_signature(extern_fn, &arg_values)?
                        } else {
                            // Try runtime symbols as fallback
                            self.call_ffi_for_function(func, &arg_values)?
                        }
                    } else {
                        // Regular user function - execute recursively
                        self.execute(module, *func_id, arg_values)?
                    }
                } else if let Some(extern_fn) = module.extern_functions.get(func_id) {
                    // FFI call to extern function with full signature info
                    self.call_extern_with_signature(extern_fn, &arg_values)?
                } else {
                    return Err(InterpError::FunctionNotFound(*func_id));
                };

                if let Some(d) = dest {
                    // Convert result back to NanBoxedValue
                    let boxed = result.to_nan_boxed(&mut self.object_heap);
                    self.current_frame_mut().registers.set(*d, boxed);
                }
            }

            IrInstruction::CallIndirect {
                dest,
                func_ptr,
                args,
                signature,
                ..
            } => {
                let ptr_val = self.current_frame().registers.get(*func_ptr);

                // Collect argument values - convert from NanBoxedValue to InterpValue
                let arg_values: Vec<InterpValue> = args
                    .iter()
                    .map(|a| {
                        InterpValue::from_nan_boxed(
                            self.current_frame().registers.get(*a),
                            &self.object_heap,
                        )
                    })
                    .collect();

                let result = if ptr_val.is_heap() {
                    let idx = ptr_val.as_heap_index();
                    match self.object_heap.get(idx) {
                        Some(HeapObject::Struct(fields)) if !fields.is_empty() => {
                            let func_interp =
                                InterpValue::from_nan_boxed(fields[0], &self.object_heap);
                            if let InterpValue::Function(func_id) = func_interp {
                                let env_idx = self
                                    .object_heap
                                    .alloc(HeapObject::Struct(fields[1..].to_vec()));
                                let mut full_args = vec![InterpValue::Ptr(env_idx as usize)];
                                full_args.extend(arg_values);

                                if module.functions.contains_key(&func_id) {
                                    self.execute(module, func_id, full_args)?
                                } else if let Some(extern_fn) =
                                    module.extern_functions.get(&func_id)
                                {
                                    self.call_extern_with_signature(extern_fn, &full_args)?
                                } else {
                                    return Err(InterpError::FunctionNotFound(func_id));
                                }
                            } else {
                                return Err(InterpError::TypeError(format!(
                                    "Closure first field is not a function: {:?}",
                                    func_interp
                                )));
                            }
                        }
                        Some(HeapObject::I64(raw_val)) => {
                            self.call_indirect_raw_i64(module, *raw_val, &arg_values, signature)?
                        }
                        Some(HeapObject::U64(raw_val)) if *raw_val != 0 => self
                            .call_indirect_raw_i64(
                                module,
                                *raw_val as i64,
                                &arg_values,
                                signature,
                            )?,
                        _ => {
                            return Err(InterpError::TypeError(format!(
                                "Cannot call non-closure heap value: {:?}",
                                InterpValue::from_nan_boxed(ptr_val, &self.object_heap)
                            )));
                        }
                    }
                } else if !ptr_val.is_f64()
                    && (ptr_val.0 & NanBoxedValue::TAG_MASK) == NanBoxedValue::TAG_FUNC
                {
                    let func_id = IrFunctionId(ptr_val.as_func_id());
                    if module.functions.contains_key(&func_id) {
                        self.execute(module, func_id, arg_values)?
                    } else if let Some(extern_fn) = module.extern_functions.get(&func_id) {
                        self.call_extern_with_signature(extern_fn, &arg_values)?
                    } else {
                        return Err(InterpError::FunctionNotFound(func_id));
                    }
                } else if ptr_val.is_ptr() {
                    let ptr = ptr_val.as_ptr();
                    if let IrType::Function {
                        params,
                        return_type,
                        ..
                    } = signature
                    {
                        self.call_ffi_ptr_with_types(ptr, &arg_values, params, return_type)?
                    } else {
                        self.call_ffi_ptr_simple(ptr, &arg_values)?
                    }
                } else {
                    let ptr_interp = InterpValue::from_nan_boxed(ptr_val, &self.object_heap);
                    match ptr_interp {
                        InterpValue::I64(raw_val) if raw_val != 0 => {
                            self.call_indirect_raw_i64(module, raw_val, &arg_values, signature)?
                        }
                        _ => {
                            return Err(InterpError::TypeError(format!(
                                "Cannot call non-function value: {:?}",
                                ptr_interp
                            )));
                        }
                    }
                };

                if let Some(d) = dest {
                    // Convert result back to NanBoxedValue
                    let boxed = result.to_nan_boxed(&mut self.object_heap);
                    self.current_frame_mut().registers.set(*d, boxed);
                }
            }

            // === Type Operations (convert at boundary) ===
            IrInstruction::Cast {
                dest,
                src,
                from_ty: _,
                to_ty,
            } => {
                let val = self.current_frame().registers.get(*src);
                let val_interp = InterpValue::from_nan_boxed(val, &self.object_heap);
                let result = self.cast_value(val_interp, to_ty)?;
                let boxed = result.to_nan_boxed(&mut self.object_heap);
                self.current_frame_mut().registers.set(*dest, boxed);
            }

            IrInstruction::BitCast { dest, src, ty: _ } => {
                // Bitcast preserves the bits - NanBoxedValue is Copy, no clone needed
                let val = self.current_frame().registers.get(*src);
                self.current_frame_mut().registers.set(*dest, val);
            }

            // === Struct Operations (use object heap for complex types) ===
            IrInstruction::CreateStruct {
                dest,
                ty: _,
                fields,
            } => {
                // Collect field values as NanBoxedValue
                let field_values: Vec<NanBoxedValue> = fields
                    .iter()
                    .map(|f| self.current_frame().registers.get(*f))
                    .collect();
                // Store in object heap
                let idx = self.object_heap.alloc(HeapObject::Struct(field_values));
                self.current_frame_mut()
                    .registers
                    .set(*dest, NanBoxedValue::from_heap_index(idx));
            }

            IrInstruction::ExtractValue {
                dest,
                aggregate,
                indices,
            } => {
                let agg = self.current_frame().registers.get(*aggregate);
                // Fast path: heap object - copy value first to avoid borrow conflict
                if agg.is_heap() {
                    let idx = agg.as_heap_index();
                    let extracted_val = {
                        if let Some(HeapObject::Struct(fields)) = self.object_heap.get(idx) {
                            if let Some(&first_idx) = indices.first() {
                                fields.get(first_idx as usize).copied()
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    };
                    if let Some(val) = extracted_val {
                        self.current_frame_mut().registers.set(*dest, val);
                        return Ok(());
                    }
                }
                // Slow path: convert and extract
                let agg_interp = InterpValue::from_nan_boxed(agg, &self.object_heap);
                let result = self.extract_value(agg_interp, indices)?;
                let boxed = result.to_nan_boxed(&mut self.object_heap);
                self.current_frame_mut().registers.set(*dest, boxed);
            }

            IrInstruction::InsertValue {
                dest,
                aggregate,
                value,
                indices,
            } => {
                let agg = self.current_frame().registers.get(*aggregate);
                let val = self.current_frame().registers.get(*value);
                // Convert to InterpValue, modify, convert back
                let mut agg_interp = InterpValue::from_nan_boxed(agg, &self.object_heap);
                let val_interp = InterpValue::from_nan_boxed(val, &self.object_heap);
                self.insert_value(&mut agg_interp, indices, val_interp)?;
                let boxed = agg_interp.to_nan_boxed(&mut self.object_heap);
                self.current_frame_mut().registers.set(*dest, boxed);
            }

            // === Union Operations (use object heap for struct storage) ===
            IrInstruction::CreateUnion {
                dest,
                discriminant,
                value,
                ty: _,
            } => {
                let val = self.current_frame().registers.get(*value);
                // Store as struct: [discriminant, value]
                let fields = vec![NanBoxedValue::from_i32(*discriminant as i32), val];
                let idx = self.object_heap.alloc(HeapObject::Struct(fields));
                self.current_frame_mut()
                    .registers
                    .set(*dest, NanBoxedValue::from_heap_index(idx));
            }

            IrInstruction::ExtractDiscriminant { dest, union_val } => {
                let union_v = self.current_frame().registers.get(*union_val);
                if union_v.is_heap() {
                    let idx = union_v.as_heap_index();
                    // Copy value before setting register to avoid borrow conflict
                    let discriminant_val = self.object_heap.get(idx).and_then(|obj| match obj {
                        HeapObject::Struct(fields) if !fields.is_empty() => Some(fields[0]),
                        _ => None,
                    });
                    if let Some(val) = discriminant_val {
                        self.current_frame_mut().registers.set(*dest, val);
                        return Ok(());
                    }
                }
                return Err(InterpError::TypeError("Expected union value".to_string()));
            }

            IrInstruction::ExtractUnionValue {
                dest,
                union_val,
                discriminant: _,
                value_ty: _,
            } => {
                let union_v = self.current_frame().registers.get(*union_val);
                if union_v.is_heap() {
                    let idx = union_v.as_heap_index();
                    // Copy value before setting register to avoid borrow conflict
                    let union_data_val = self.object_heap.get(idx).and_then(|obj| match obj {
                        HeapObject::Struct(fields) if fields.len() > 1 => Some(fields[1]),
                        _ => None,
                    });
                    if let Some(val) = union_data_val {
                        self.current_frame_mut().registers.set(*dest, val);
                        return Ok(());
                    }
                }
                return Err(InterpError::TypeError(
                    "Expected union value with data".to_string(),
                ));
            }

            // === Select Operation (NaN-boxed fast path) ===
            IrInstruction::Select {
                dest,
                condition,
                true_val,
                false_val,
            } => {
                let cond = self.current_frame().registers.get(*condition);
                let is_true = if cond.is_bool() {
                    cond.as_bool()
                } else {
                    InterpValue::from_nan_boxed(cond, &self.object_heap).to_bool()?
                };
                // NanBoxedValue is Copy - no clone needed!
                let result = if is_true {
                    self.current_frame().registers.get(*true_val)
                } else {
                    self.current_frame().registers.get(*false_val)
                };
                self.current_frame_mut().registers.set(*dest, result);
            }

            // === Function Reference ===
            IrInstruction::FunctionRef { dest, func_id } => {
                self.current_frame_mut()
                    .registers
                    .set(*dest, NanBoxedValue::from_func_id(func_id.0));
            }

            // === Closure Operations (use object heap) ===
            IrInstruction::MakeClosure {
                dest,
                func_id,
                captured_values,
            } => {
                // Collect captured values as NanBoxedValue
                let mut closure_data: Vec<NanBoxedValue> =
                    vec![NanBoxedValue::from_func_id(func_id.0)];
                for v in captured_values {
                    closure_data.push(self.current_frame().registers.get(*v));
                }
                let idx = self.object_heap.alloc(HeapObject::Struct(closure_data));
                self.current_frame_mut()
                    .registers
                    .set(*dest, NanBoxedValue::from_heap_index(idx));
            }

            IrInstruction::ClosureFunc { dest, closure } => {
                let closure_val = self.current_frame().registers.get(*closure);
                if closure_val.is_heap() {
                    let idx = closure_val.as_heap_index();
                    // Copy value before setting register to avoid borrow conflict
                    let func_val = self.object_heap.get(idx).and_then(|obj| match obj {
                        HeapObject::Struct(fields) if !fields.is_empty() => Some(fields[0]),
                        _ => None,
                    });
                    if let Some(val) = func_val {
                        self.current_frame_mut().registers.set(*dest, val);
                        return Ok(());
                    }
                }
                return Err(InterpError::TypeError("Expected closure value".to_string()));
            }

            IrInstruction::ClosureEnv { dest, closure } => {
                let closure_val = self.current_frame().registers.get(*closure);
                if closure_val.is_heap() {
                    let idx = closure_val.as_heap_index();
                    if let Some(HeapObject::Struct(fields)) = self.object_heap.get(idx) {
                        if fields.len() > 1 {
                            // Return environment as a new struct (skip function pointer)
                            let env: Vec<NanBoxedValue> = fields[1..].to_vec();
                            let env_idx = self.object_heap.alloc(HeapObject::Struct(env));
                            self.current_frame_mut()
                                .registers
                                .set(*dest, NanBoxedValue::from_heap_index(env_idx));
                            return Ok(());
                        }
                    }
                }
                // Empty environment
                let idx = self.object_heap.alloc(HeapObject::Struct(vec![]));
                self.current_frame_mut()
                    .registers
                    .set(*dest, NanBoxedValue::from_heap_index(idx));
            }

            // === Borrowing (no-op in interpreter - NanBoxedValue is Copy) ===
            IrInstruction::BorrowImmutable { dest, src, .. }
            | IrInstruction::BorrowMutable { dest, src, .. } => {
                let val = self.current_frame().registers.get(*src);
                self.current_frame_mut().registers.set(*dest, val);
            }

            IrInstruction::Clone { dest, src } => {
                // NanBoxedValue is Copy - this is a true no-op copy
                let val = self.current_frame().registers.get(*src);
                self.current_frame_mut().registers.set(*dest, val);
            }

            IrInstruction::EndBorrow { .. } => {
                // No-op in interpreter
            }

            // === Memory Operations (need conversion for ptr operations) ===
            IrInstruction::MemCopy { dest, src, size } => {
                let dest_val = self.current_frame().registers.get(*dest);
                let src_val = self.current_frame().registers.get(*src);
                let size_val = self.current_frame().registers.get(*size);

                let dest_ptr = if dest_val.is_ptr() {
                    dest_val.as_ptr()
                } else {
                    InterpValue::from_nan_boxed(dest_val, &self.object_heap).to_usize()?
                };
                let src_ptr = if src_val.is_ptr() {
                    src_val.as_ptr()
                } else {
                    InterpValue::from_nan_boxed(src_val, &self.object_heap).to_usize()?
                };
                let size_int = if size_val.is_i32() {
                    size_val.as_i32() as usize
                } else {
                    InterpValue::from_nan_boxed(size_val, &self.object_heap).to_usize()?
                };

                // Copy bytes
                for i in 0..size_int {
                    if src_ptr + i < self.heap.len() && dest_ptr + i < self.heap.len() {
                        self.heap[dest_ptr + i] = self.heap[src_ptr + i];
                    }
                }
            }

            IrInstruction::MemSet { dest, value, size } => {
                let dest_val = self.current_frame().registers.get(*dest);
                let val = self.current_frame().registers.get(*value);
                let size_val = self.current_frame().registers.get(*size);

                let dest_ptr = if dest_val.is_ptr() {
                    dest_val.as_ptr()
                } else {
                    InterpValue::from_nan_boxed(dest_val, &self.object_heap).to_usize()?
                };
                let byte_val = if val.is_i32() {
                    val.as_i32() as u8
                } else {
                    InterpValue::from_nan_boxed(val, &self.object_heap).to_i64()? as u8
                };
                let size_int = if size_val.is_i32() {
                    size_val.as_i32() as usize
                } else {
                    InterpValue::from_nan_boxed(size_val, &self.object_heap).to_usize()?
                };

                // Set bytes
                for i in 0..size_int {
                    if dest_ptr + i < self.heap.len() {
                        self.heap[dest_ptr + i] = byte_val;
                    }
                }
            }

            // === Undefined/Special ===
            IrInstruction::Undef { dest, .. } => {
                self.current_frame_mut()
                    .registers
                    .set(*dest, NanBoxedValue::void());
            }

            IrInstruction::Panic { message } => {
                return Err(InterpError::Panic(
                    message.clone().unwrap_or_else(|| "panic".to_string()),
                ));
            }

            IrInstruction::DebugLoc { .. } => {
                // No-op: debug info
            }

            // === Phi nodes are handled separately ===
            IrInstruction::Phi { .. } => {
                // Phi nodes are processed at block entry
            }

            // === Control flow (should be terminators) ===
            IrInstruction::Jump { .. }
            | IrInstruction::Branch { .. }
            | IrInstruction::Switch { .. }
            | IrInstruction::Return { .. } => {
                // These should be terminators, not instructions
                // They're handled by execute_terminator
            }

            // === Exception handling (simplified) ===
            IrInstruction::Throw { exception } => {
                let exc = self.current_frame().registers.get(*exception);
                return Err(InterpError::Exception(format!("{:?}", exc)));
            }

            IrInstruction::LandingPad { dest, .. } => {
                // Simplified: just store a null value
                self.current_frame_mut()
                    .registers
                    .set(*dest, NanBoxedValue::null());
            }

            IrInstruction::Resume { .. } => {
                return Err(InterpError::Exception("resumed exception".to_string()));
            }

            // === Inline Assembly (not supported in interpreter) ===
            IrInstruction::InlineAsm { dest, .. } => {
                if let Some(d) = dest {
                    self.current_frame_mut()
                        .registers
                        .set(*d, NanBoxedValue::void());
                }
            }

            // === SIMD Vector Operations (not yet supported in interpreter) ===
            // These are lowered by the JIT compiler; interpreter falls back to void
            IrInstruction::VectorLoad { dest, .. } => {
                self.current_frame_mut()
                    .registers
                    .set(*dest, NanBoxedValue::void());
            }
            IrInstruction::VectorStore { .. } => {
                // Vector store has no dest, nothing to do
            }
            IrInstruction::VectorBinOp { dest, .. } => {
                self.current_frame_mut()
                    .registers
                    .set(*dest, NanBoxedValue::void());
            }
            IrInstruction::VectorSplat { dest, .. } => {
                self.current_frame_mut()
                    .registers
                    .set(*dest, NanBoxedValue::void());
            }
            IrInstruction::VectorExtract { dest, .. } => {
                self.current_frame_mut()
                    .registers
                    .set(*dest, NanBoxedValue::void());
            }
            IrInstruction::VectorInsert { dest, .. } => {
                self.current_frame_mut()
                    .registers
                    .set(*dest, NanBoxedValue::void());
            }
            IrInstruction::VectorReduce { dest, .. } => {
                self.current_frame_mut()
                    .registers
                    .set(*dest, NanBoxedValue::void());
            }
            IrInstruction::VectorUnaryOp { dest, .. } => {
                self.current_frame_mut()
                    .registers
                    .set(*dest, NanBoxedValue::void());
            }
            IrInstruction::VectorMinMax { dest, .. } => {
                self.current_frame_mut()
                    .registers
                    .set(*dest, NanBoxedValue::void());
            }

            // === Global Variable Access ===
            IrInstruction::LoadGlobal {
                dest,
                global_id,
                ty: _,
            } => {
                // Load value from global variable store
                let val = if let Some(&stored_val) = self.global_store.get(global_id) {
                    stored_val
                } else if let Some(global) = module.globals.get(global_id) {
                    // Global not yet initialized - use module's initializer as default
                    if let Some(ref init) = global.initializer {
                        self.ir_value_to_nanboxed(init)?
                    } else {
                        NanBoxedValue::null()
                    }
                } else {
                    // Global not found - return null
                    NanBoxedValue::null()
                };
                tracing::debug!("[INTERP] LoadGlobal {:?} = {:?}", global_id, val);
                self.current_frame_mut().registers.set(*dest, val);
            }

            IrInstruction::StoreGlobal { global_id, value } => {
                // Store value to global variable store
                let val = self.current_frame().registers.get(*value);
                tracing::debug!("[INTERP] StoreGlobal {:?} = {:?}", global_id, val);
                self.global_store.insert(*global_id, val);
            }
        }
        Ok(())
    }

    /// Execute a block terminator
    /// Execute terminator (returns NanBoxedValue for efficiency)
    fn execute_terminator(
        &mut self,
        _module: &IrModule,
        _function: &IrFunction,
        terminator: &IrTerminator,
    ) -> Result<TerminatorResult, InterpError> {
        match terminator {
            IrTerminator::Branch { target } => Ok(TerminatorResult::Continue(*target)),

            IrTerminator::CondBranch {
                condition,
                true_target,
                false_target,
            } => {
                let cond = self.current_frame().registers.get(*condition);
                // Fast path for NaN-boxed bool
                let is_true = if cond.is_bool() {
                    cond.as_bool()
                } else {
                    // Slow path: convert to InterpValue
                    InterpValue::from_nan_boxed(cond, &self.object_heap).to_bool()?
                };
                if is_true {
                    Ok(TerminatorResult::Continue(*true_target))
                } else {
                    Ok(TerminatorResult::Continue(*false_target))
                }
            }

            IrTerminator::Switch {
                value,
                cases,
                default,
            } => {
                let val = self.current_frame().registers.get(*value);
                // Fast path for NaN-boxed i32
                let switch_val = if val.is_i32() {
                    val.as_i32() as i64
                } else {
                    // Slow path: convert to InterpValue
                    InterpValue::from_nan_boxed(val, &self.object_heap).to_i64()?
                };
                for (case_val, target) in cases {
                    if *case_val == switch_val {
                        return Ok(TerminatorResult::Continue(*target));
                    }
                }
                Ok(TerminatorResult::Continue(*default))
            }

            IrTerminator::Return { value } => {
                // NanBoxedValue is Copy - no clone needed
                let result = if let Some(v) = value {
                    self.current_frame().registers.get(*v)
                } else {
                    NanBoxedValue::void()
                };
                Ok(TerminatorResult::Return(result))
            }

            IrTerminator::Unreachable => Err(InterpError::RuntimeError(
                "Reached unreachable code".to_string(),
            )),

            IrTerminator::NoReturn { .. } => Err(InterpError::RuntimeError(
                "NoReturn terminator executed".to_string(),
            )),
        }
    }

    /// Convert IrValue to InterpValue
    fn ir_value_to_interp(&self, value: &IrValue) -> Result<InterpValue, InterpError> {
        match value {
            IrValue::Void => Ok(InterpValue::Void),
            IrValue::Undef => Ok(InterpValue::Void),
            IrValue::Null => Ok(InterpValue::Null),
            IrValue::Bool(b) => Ok(InterpValue::Bool(*b)),
            IrValue::I8(n) => Ok(InterpValue::I8(*n)),
            IrValue::I16(n) => Ok(InterpValue::I16(*n)),
            IrValue::I32(n) => Ok(InterpValue::I32(*n)),
            IrValue::I64(n) => Ok(InterpValue::I64(*n)),
            IrValue::U8(n) => Ok(InterpValue::U8(*n)),
            IrValue::U16(n) => Ok(InterpValue::U16(*n)),
            IrValue::U32(n) => Ok(InterpValue::U32(*n)),
            IrValue::U64(n) => Ok(InterpValue::U64(*n)),
            IrValue::F32(n) => Ok(InterpValue::F32(*n)),
            IrValue::F64(n) => Ok(InterpValue::F64(*n)),
            IrValue::String(s) => Ok(InterpValue::String(s.clone())),
            IrValue::Array(arr) => {
                let values: Result<Vec<_>, _> =
                    arr.iter().map(|v| self.ir_value_to_interp(v)).collect();
                Ok(InterpValue::Array(values?))
            }
            IrValue::Struct(fields) => {
                let values: Result<Vec<_>, _> =
                    fields.iter().map(|v| self.ir_value_to_interp(v)).collect();
                Ok(InterpValue::Struct(values?))
            }
            IrValue::Function(func_id) => Ok(InterpValue::Function(*func_id)),
            IrValue::Closure {
                function,
                environment,
            } => {
                let env = self.ir_value_to_interp(environment)?;
                Ok(InterpValue::Struct(vec![
                    InterpValue::Function(*function),
                    env,
                ]))
            }
        }
    }

    /// Convert IrValue directly to NanBoxedValue (fast path for primitives)
    ///
    /// This avoids the InterpValue intermediate for common cases.
    fn ir_value_to_nanboxed(&mut self, value: &IrValue) -> Result<NanBoxedValue, InterpError> {
        match value {
            // Fast path: primitives go directly to NaN-boxed format
            IrValue::Void => Ok(NanBoxedValue::void()),
            IrValue::Undef => Ok(NanBoxedValue::void()),
            IrValue::Null => Ok(NanBoxedValue::null()),
            IrValue::Bool(b) => Ok(NanBoxedValue::from_bool(*b)),
            IrValue::I8(n) => Ok(NanBoxedValue::from_i32(*n as i32)),
            IrValue::I16(n) => Ok(NanBoxedValue::from_i32(*n as i32)),
            IrValue::I32(n) => Ok(NanBoxedValue::from_i32(*n)),
            IrValue::I64(n) => Ok(NanBoxedValue::from_i64(*n)),
            IrValue::U8(n) => Ok(NanBoxedValue::from_i32(*n as i32)),
            IrValue::U16(n) => Ok(NanBoxedValue::from_i32(*n as i32)),
            IrValue::U32(n) => Ok(NanBoxedValue::from_i64(*n as i64)),
            IrValue::U64(n) => Ok(NanBoxedValue::from_i64(*n as i64)),
            IrValue::F32(n) => Ok(NanBoxedValue::from_f32(*n)),
            IrValue::F64(n) => Ok(NanBoxedValue::from_f64(*n)),
            IrValue::Function(func_id) => Ok(NanBoxedValue::from_func_id(func_id.0)),

            // Slow path: complex types need heap allocation
            IrValue::String(s) => {
                let idx = self.object_heap.alloc(HeapObject::String(s.clone()));
                Ok(NanBoxedValue::from_heap_index(idx))
            }
            IrValue::Array(arr) => {
                let values: Result<Vec<_>, _> =
                    arr.iter().map(|v| self.ir_value_to_nanboxed(v)).collect();
                let idx = self.object_heap.alloc(HeapObject::Array(values?));
                Ok(NanBoxedValue::from_heap_index(idx))
            }
            IrValue::Struct(fields) => {
                let values: Result<Vec<_>, _> = fields
                    .iter()
                    .map(|v| self.ir_value_to_nanboxed(v))
                    .collect();
                let idx = self.object_heap.alloc(HeapObject::Struct(values?));
                Ok(NanBoxedValue::from_heap_index(idx))
            }
            IrValue::Closure {
                function,
                environment,
            } => {
                let env = self.ir_value_to_nanboxed(environment)?;
                let func = NanBoxedValue::from_func_id(function.0);
                let idx = self.object_heap.alloc(HeapObject::Struct(vec![func, env]));
                Ok(NanBoxedValue::from_heap_index(idx))
            }
        }
    }

    /// Evaluate a binary operation with specialized fast paths
    ///
    /// Uses type-specialized handlers for common integer operations
    /// to avoid the overhead of generic type conversion.
    fn eval_binary_op(
        &self,
        op: BinaryOp,
        left: InterpValue,
        right: InterpValue,
    ) -> Result<InterpValue, InterpError> {
        // Fast path: both operands are i32 (most common case in Haxe)
        if let (InterpValue::I32(l), InterpValue::I32(r)) = (&left, &right) {
            return match op {
                BinaryOp::Add => Ok(InterpValue::I32(l.wrapping_add(*r))),
                BinaryOp::Sub => Ok(InterpValue::I32(l.wrapping_sub(*r))),
                BinaryOp::Mul => Ok(InterpValue::I32(l.wrapping_mul(*r))),
                BinaryOp::Div => {
                    if *r == 0 {
                        return Err(InterpError::RuntimeError("Division by zero".to_string()));
                    }
                    Ok(InterpValue::I32(l / r))
                }
                BinaryOp::Rem => {
                    if *r == 0 {
                        return Err(InterpError::RuntimeError("Modulo by zero".to_string()));
                    }
                    Ok(InterpValue::I32(l % r))
                }
                BinaryOp::And => Ok(InterpValue::I32(l & r)),
                BinaryOp::Or => Ok(InterpValue::I32(l | r)),
                BinaryOp::Xor => Ok(InterpValue::I32(l ^ r)),
                BinaryOp::Shl => Ok(InterpValue::I32(l << (r & 31))),
                BinaryOp::Shr => Ok(InterpValue::I32(l >> (r & 31))),
                BinaryOp::Ushr => Ok(InterpValue::I32(((*l as u32) >> (r & 31)) as i32)),
                _ => self.eval_binary_op_slow(op, left, right),
            };
        }

        // Fast path: both operands are i64
        if let (InterpValue::I64(l), InterpValue::I64(r)) = (&left, &right) {
            return match op {
                BinaryOp::Add => Ok(InterpValue::I64(l.wrapping_add(*r))),
                BinaryOp::Sub => Ok(InterpValue::I64(l.wrapping_sub(*r))),
                BinaryOp::Mul => Ok(InterpValue::I64(l.wrapping_mul(*r))),
                BinaryOp::Div => {
                    if *r == 0 {
                        return Err(InterpError::RuntimeError("Division by zero".to_string()));
                    }
                    Ok(InterpValue::I64(l / r))
                }
                BinaryOp::Rem => {
                    if *r == 0 {
                        return Err(InterpError::RuntimeError("Modulo by zero".to_string()));
                    }
                    Ok(InterpValue::I64(l % r))
                }
                BinaryOp::And => Ok(InterpValue::I64(l & r)),
                BinaryOp::Or => Ok(InterpValue::I64(l | r)),
                BinaryOp::Xor => Ok(InterpValue::I64(l ^ r)),
                BinaryOp::Shl => Ok(InterpValue::I64(l << (r & 63))),
                BinaryOp::Shr => Ok(InterpValue::I64(l >> (r & 63))),
                BinaryOp::Ushr => Ok(InterpValue::I64(((*l as u64) >> (r & 63)) as i64)),
                _ => self.eval_binary_op_slow(op, left, right),
            };
        }

        // Fast path: both operands are f64
        if let (InterpValue::F64(l), InterpValue::F64(r)) = (&left, &right) {
            return match op {
                BinaryOp::Add | BinaryOp::FAdd => Ok(InterpValue::F64(l + r)),
                BinaryOp::Sub | BinaryOp::FSub => Ok(InterpValue::F64(l - r)),
                BinaryOp::Mul | BinaryOp::FMul => Ok(InterpValue::F64(l * r)),
                BinaryOp::Div | BinaryOp::FDiv => Ok(InterpValue::F64(l / r)),
                BinaryOp::Rem | BinaryOp::FRem => Ok(InterpValue::F64(l % r)),
                _ => self.eval_binary_op_slow(op, left, right),
            };
        }

        // Slow path: mixed types or less common types
        self.eval_binary_op_slow(op, left, right)
    }

    /// Slow path for binary operations (type conversion required)
    fn eval_binary_op_slow(
        &self,
        op: BinaryOp,
        left: InterpValue,
        right: InterpValue,
    ) -> Result<InterpValue, InterpError> {
        // Check if either operand is a float - if so, use float arithmetic for Add/Sub/Mul/Div/Rem
        let either_is_float = left.is_float() || right.is_float();

        match op {
            // Arithmetic operations - use float if either operand is float
            BinaryOp::Add => {
                if either_is_float {
                    let l = left.to_f64()?;
                    let r = right.to_f64()?;
                    Ok(InterpValue::F64(l + r))
                } else {
                    let l = left.to_i64()?;
                    let r = right.to_i64()?;
                    Ok(InterpValue::I64(l.wrapping_add(r)))
                }
            }
            BinaryOp::Sub => {
                if either_is_float {
                    let l = left.to_f64()?;
                    let r = right.to_f64()?;
                    Ok(InterpValue::F64(l - r))
                } else {
                    let l = left.to_i64()?;
                    let r = right.to_i64()?;
                    Ok(InterpValue::I64(l.wrapping_sub(r)))
                }
            }
            BinaryOp::Mul => {
                if either_is_float {
                    let l = left.to_f64()?;
                    let r = right.to_f64()?;
                    Ok(InterpValue::F64(l * r))
                } else {
                    let l = left.to_i64()?;
                    let r = right.to_i64()?;
                    Ok(InterpValue::I64(l.wrapping_mul(r)))
                }
            }
            BinaryOp::Div => {
                if either_is_float {
                    let l = left.to_f64()?;
                    let r = right.to_f64()?;
                    Ok(InterpValue::F64(l / r))
                } else {
                    let l = left.to_i64()?;
                    let r = right.to_i64()?;
                    if r == 0 {
                        return Err(InterpError::RuntimeError("Division by zero".to_string()));
                    }
                    Ok(InterpValue::I64(l / r))
                }
            }
            BinaryOp::Rem => {
                if either_is_float {
                    let l = left.to_f64()?;
                    let r = right.to_f64()?;
                    Ok(InterpValue::F64(l % r))
                } else {
                    let l = left.to_i64()?;
                    let r = right.to_i64()?;
                    if r == 0 {
                        return Err(InterpError::RuntimeError("Modulo by zero".to_string()));
                    }
                    Ok(InterpValue::I64(l % r))
                }
            }

            // Bitwise operations
            BinaryOp::And => {
                let l = left.to_i64()?;
                let r = right.to_i64()?;
                Ok(InterpValue::I64(l & r))
            }
            BinaryOp::Or => {
                let l = left.to_i64()?;
                let r = right.to_i64()?;
                Ok(InterpValue::I64(l | r))
            }
            BinaryOp::Xor => {
                let l = left.to_i64()?;
                let r = right.to_i64()?;
                Ok(InterpValue::I64(l ^ r))
            }
            BinaryOp::Shl => {
                let l = left.to_i64()?;
                let r = right.to_i64()?;
                Ok(InterpValue::I64(l << (r & 63)))
            }
            BinaryOp::Shr => {
                let l = left.to_i64()?;
                let r = right.to_i64()?;
                Ok(InterpValue::I64(l >> (r & 63)))
            }
            BinaryOp::Ushr => {
                let l = left.to_i64()?;
                let r = right.to_i64()?;
                Ok(InterpValue::I64(((l as u64) >> (r & 63)) as i64))
            }

            // Floating point arithmetic
            BinaryOp::FAdd => {
                let l = left.to_f64()?;
                let r = right.to_f64()?;
                Ok(InterpValue::F64(l + r))
            }
            BinaryOp::FSub => {
                let l = left.to_f64()?;
                let r = right.to_f64()?;
                Ok(InterpValue::F64(l - r))
            }
            BinaryOp::FMul => {
                let l = left.to_f64()?;
                let r = right.to_f64()?;
                Ok(InterpValue::F64(l * r))
            }
            BinaryOp::FDiv => {
                let l = left.to_f64()?;
                let r = right.to_f64()?;
                Ok(InterpValue::F64(l / r))
            }
            BinaryOp::FRem => {
                let l = left.to_f64()?;
                let r = right.to_f64()?;
                Ok(InterpValue::F64(l % r))
            }
        }
    }

    /// Evaluate a unary operation
    fn eval_unary_op(&self, op: UnaryOp, operand: InterpValue) -> Result<InterpValue, InterpError> {
        match op {
            UnaryOp::Neg => {
                let val = operand.to_i64()?;
                Ok(InterpValue::I64(-val))
            }
            UnaryOp::Not => {
                let val = operand.to_i64()?;
                Ok(InterpValue::I64(!val))
            }
            UnaryOp::FNeg => {
                let val = operand.to_f64()?;
                Ok(InterpValue::F64(-val))
            }
        }
    }

    /// Evaluate a comparison operation
    fn eval_compare_op(
        &self,
        op: CompareOp,
        left: InterpValue,
        right: InterpValue,
    ) -> Result<bool, InterpError> {
        match op {
            // Integer comparisons (signed)
            CompareOp::Eq => Ok(left.to_i64()? == right.to_i64()?),
            CompareOp::Ne => Ok(left.to_i64()? != right.to_i64()?),
            CompareOp::Lt => Ok(left.to_i64()? < right.to_i64()?),
            CompareOp::Le => Ok(left.to_i64()? <= right.to_i64()?),
            CompareOp::Gt => Ok(left.to_i64()? > right.to_i64()?),
            CompareOp::Ge => Ok(left.to_i64()? >= right.to_i64()?),

            // Unsigned comparisons
            CompareOp::ULt => Ok((left.to_i64()? as u64) < (right.to_i64()? as u64)),
            CompareOp::ULe => Ok((left.to_i64()? as u64) <= (right.to_i64()? as u64)),
            CompareOp::UGt => Ok((left.to_i64()? as u64) > (right.to_i64()? as u64)),
            CompareOp::UGe => Ok((left.to_i64()? as u64) >= (right.to_i64()? as u64)),

            // Floating point comparisons
            CompareOp::FEq => Ok(left.to_f64()? == right.to_f64()?),
            CompareOp::FNe => Ok(left.to_f64()? != right.to_f64()?),
            CompareOp::FLt => Ok(left.to_f64()? < right.to_f64()?),
            CompareOp::FLe => Ok(left.to_f64()? <= right.to_f64()?),
            CompareOp::FGt => Ok(left.to_f64()? > right.to_f64()?),
            CompareOp::FGe => Ok(left.to_f64()? >= right.to_f64()?),

            // Floating point ordered/unordered
            CompareOp::FOrd => {
                let l = left.to_f64()?;
                let r = right.to_f64()?;
                Ok(!l.is_nan() && !r.is_nan())
            }
            CompareOp::FUno => {
                let l = left.to_f64()?;
                let r = right.to_f64()?;
                Ok(l.is_nan() || r.is_nan())
            }
        }
    }

    /// Cast a value to a target type
    fn cast_value(&self, val: InterpValue, to_ty: &IrType) -> Result<InterpValue, InterpError> {
        match to_ty {
            IrType::Bool => Ok(InterpValue::Bool(val.to_bool()?)),
            IrType::I8 => Ok(InterpValue::I8(val.to_i64()? as i8)),
            IrType::I16 => Ok(InterpValue::I16(val.to_i64()? as i16)),
            IrType::I32 => Ok(InterpValue::I32(val.to_i64()? as i32)),
            IrType::I64 => Ok(InterpValue::I64(val.to_i64()?)),
            IrType::U8 => Ok(InterpValue::U8(val.to_i64()? as u8)),
            IrType::U16 => Ok(InterpValue::U16(val.to_i64()? as u16)),
            IrType::U32 => Ok(InterpValue::U32(val.to_i64()? as u32)),
            IrType::U64 => Ok(InterpValue::U64(val.to_i64()? as u64)),
            IrType::F32 => Ok(InterpValue::F32(val.to_f64()? as f32)),
            IrType::F64 => Ok(InterpValue::F64(val.to_f64()?)),
            IrType::Ptr(_) => Ok(InterpValue::Ptr(val.to_usize()?)),
            _ => Ok(val), // For other types, pass through
        }
    }

    /// Load a value from a pointer
    /// Uses raw pointer access since alloc_heap returns system allocator pointers
    fn load_from_ptr(&self, ptr: InterpValue, ty: &IrType) -> Result<InterpValue, InterpError> {
        let addr = ptr.to_usize()?;

        // Null pointer check
        if addr == 0 {
            return Err(InterpError::RuntimeError(
                "Null pointer dereference".to_string(),
            ));
        }

        // Use raw pointer access since alloc_heap uses system allocator
        // SAFETY: We trust that valid pointers come from alloc_heap or extern functions
        unsafe {
            match ty {
                IrType::Bool => Ok(InterpValue::Bool(*(addr as *const u8) != 0)),
                IrType::I8 => Ok(InterpValue::I8(*(addr as *const i8))),
                IrType::U8 => Ok(InterpValue::U8(*(addr as *const u8))),
                IrType::I16 => Ok(InterpValue::I16(*(addr as *const i16))),
                IrType::U16 => Ok(InterpValue::U16(*(addr as *const u16))),
                IrType::I32 => Ok(InterpValue::I32(*(addr as *const i32))),
                IrType::U32 => Ok(InterpValue::U32(*(addr as *const u32))),
                IrType::I64 => Ok(InterpValue::I64(*(addr as *const i64))),
                IrType::U64 => Ok(InterpValue::U64(*(addr as *const u64))),
                IrType::F32 => Ok(InterpValue::F32(*(addr as *const f32))),
                IrType::F64 => Ok(InterpValue::F64(*(addr as *const f64))),
                IrType::Ptr(_) => Ok(InterpValue::Ptr(*(addr as *const usize))),
                _ => {
                    // For other types, return a placeholder
                    Ok(InterpValue::Void)
                }
            }
        }
    }

    /// Store a value to a pointer
    /// Uses raw pointer access since alloc_heap returns system allocator pointers
    fn store_to_ptr(&mut self, ptr: InterpValue, val: InterpValue) -> Result<(), InterpError> {
        let addr = ptr.to_usize()?;

        // Null pointer check
        if addr == 0 {
            return Err(InterpError::RuntimeError("Null pointer write".to_string()));
        }

        // Use raw pointer access since alloc_heap uses system allocator
        // SAFETY: We trust that valid pointers come from alloc_heap or extern functions
        unsafe {
            match val {
                InterpValue::Bool(b) => {
                    *(addr as *mut u8) = if b { 1 } else { 0 };
                }
                InterpValue::I8(n) => {
                    *(addr as *mut i8) = n;
                }
                InterpValue::U8(n) => {
                    *(addr as *mut u8) = n;
                }
                InterpValue::I16(n) => {
                    *(addr as *mut i16) = n;
                }
                InterpValue::U16(n) => {
                    *(addr as *mut u16) = n;
                }
                InterpValue::I32(n) => {
                    *(addr as *mut i32) = n;
                }
                InterpValue::U32(n) => {
                    *(addr as *mut u32) = n;
                }
                InterpValue::I64(n) => {
                    *(addr as *mut i64) = n;
                }
                InterpValue::U64(n) => {
                    *(addr as *mut u64) = n;
                }
                InterpValue::F32(n) => {
                    *(addr as *mut f32) = n;
                }
                InterpValue::F64(n) => {
                    *(addr as *mut f64) = n;
                }
                InterpValue::Ptr(p) => {
                    *(addr as *mut usize) = p;
                }
                InterpValue::Function(id) => {
                    *(addr as *mut i64) = NanBoxedValue::from_func_id(id.0).0 as i64;
                }
                _ => {}
            }
        }

        Ok(())
    }

    /// Allocate memory on the heap using system allocator
    /// Uses a fixed alignment of 8 bytes which is suitable for most types
    fn alloc_heap(&mut self, size: usize) -> Result<usize, InterpError> {
        if size == 0 {
            return Ok(0); // Null pointer for zero-size allocations
        }

        // Round up size to alignment for proper deallocation
        let aligned_size = (size + 7) & !7;

        // Use system allocator with layout stored for later deallocation
        let layout = std::alloc::Layout::from_size_align(aligned_size, 8)
            .map_err(|_| InterpError::RuntimeError("Invalid allocation layout".to_string()))?;

        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            return Err(InterpError::RuntimeError("Allocation failed".to_string()));
        }

        let ptr_val = ptr as usize;

        // Track this allocation with its size so we can free it correctly
        self.heap_allocations.insert(ptr_val, aligned_size);

        Ok(ptr_val)
    }

    /// Free heap-allocated memory
    fn free_heap(&mut self, ptr: usize) {
        if ptr == 0 {
            return; // Don't free null pointers
        }

        // Look up the allocation size and verify this was a tracked allocation
        if let Some(size) = self.heap_allocations.remove(&ptr) {
            let layout = std::alloc::Layout::from_size_align(size, 8).unwrap();
            unsafe {
                std::alloc::dealloc(ptr as *mut u8, layout);
            }
        }
        // If not in heap_allocations, it might be an invalid pointer - don't free it
    }

    /// Extract a value from an aggregate using indices
    fn extract_value(&self, agg: InterpValue, indices: &[u32]) -> Result<InterpValue, InterpError> {
        let mut current = agg;
        for &idx in indices {
            match current {
                InterpValue::Struct(fields) | InterpValue::Array(fields) => {
                    if (idx as usize) < fields.len() {
                        current = fields[idx as usize].clone();
                    } else {
                        return Err(InterpError::RuntimeError(format!(
                            "Index {} out of bounds",
                            idx
                        )));
                    }
                }
                _ => {
                    return Err(InterpError::TypeError(
                        "Cannot extract from non-aggregate".to_string(),
                    ));
                }
            }
        }
        Ok(current)
    }

    /// Insert a value into an aggregate using indices
    fn insert_value(
        &self,
        agg: &mut InterpValue,
        indices: &[u32],
        val: InterpValue,
    ) -> Result<(), InterpError> {
        if indices.is_empty() {
            *agg = val;
            return Ok(());
        }

        let idx = indices[0] as usize;
        match agg {
            InterpValue::Struct(fields) | InterpValue::Array(fields) => {
                if idx < fields.len() {
                    if indices.len() == 1 {
                        fields[idx] = val;
                    } else {
                        self.insert_value(&mut fields[idx], &indices[1..], val)?;
                    }
                } else {
                    return Err(InterpError::RuntimeError(format!(
                        "Index {} out of bounds",
                        idx
                    )));
                }
            }
            _ => {
                return Err(InterpError::TypeError(
                    "Cannot insert into non-aggregate".to_string(),
                ));
            }
        }
        Ok(())
    }

    fn interp_value_to_haxe_string(
        &self,
        value: &InterpValue,
    ) -> Result<Option<String>, InterpError> {
        match value {
            InterpValue::String(s) => Ok(Some(s.clone())),
            InterpValue::Null => Ok(Some("null".to_string())),
            InterpValue::Ptr(ptr) => {
                if *ptr == 0 {
                    return Ok(Some("null".to_string()));
                }
                let haxe_string =
                    unsafe { &*(*ptr as *const rayzor_runtime::haxe_string::HaxeString) };
                if haxe_string.ptr.is_null() {
                    return Ok(Some(String::new()));
                }
                let bytes = unsafe { std::slice::from_raw_parts(haxe_string.ptr, haxe_string.len) };
                Ok(Some(String::from_utf8_lossy(bytes).into_owned()))
            }
            InterpValue::Struct(fields) => {
                if let Some(first) = fields.first() {
                    if let Some(s) = self.interp_value_to_haxe_string(first)? {
                        return Ok(Some(s));
                    }
                    if let InterpValue::Ptr(ptr) = first {
                        let len = fields.get(1).and_then(|v| v.to_usize().ok()).unwrap_or(0);
                        if *ptr == 0 {
                            return Ok(Some("null".to_string()));
                        }
                        let bytes = unsafe { std::slice::from_raw_parts(*ptr as *const u8, len) };
                        return Ok(Some(String::from_utf8_lossy(bytes).into_owned()));
                    }
                }
                Ok(None)
            }
            _ => Ok(None),
        }
    }

    /// Call an extern function by name (without signature - uses built-in handlers)
    fn call_extern(
        &mut self,
        name: &str,
        args: &[InterpValue],
    ) -> Result<InterpValue, InterpError> {
        // Built-in functions (simple implementations for common operations)
        match name {
            "trace" | "haxe_print" | "print" => {
                // Print function - handle all numeric types
                if let Some(arg) = args.first() {
                    match arg {
                        InterpValue::String(s) => println!("{}", s),
                        InterpValue::I8(n) => println!("{}", n),
                        InterpValue::I16(n) => println!("{}", n),
                        InterpValue::I32(n) => println!("{}", n),
                        InterpValue::I64(n) => println!("{}", n),
                        InterpValue::U8(n) => println!("{}", n),
                        InterpValue::U16(n) => println!("{}", n),
                        InterpValue::U32(n) => println!("{}", n),
                        InterpValue::U64(n) => println!("{}", n),
                        InterpValue::F32(n) => println!("{}", n),
                        InterpValue::F64(n) => println!("{}", n),
                        InterpValue::Bool(b) => println!("{}", b),
                        InterpValue::Ptr(p) => println!("<ptr:{:#x}>", p),
                        InterpValue::Null => println!("null"),
                        InterpValue::Void => println!("<void>"),
                        other => println!("{:?}", other),
                    }
                }
                Ok(InterpValue::Void)
            }
            // Handle haxe_trace_string_struct which takes a struct containing the string
            "haxe_trace_string_struct" => {
                if let Some(arg) = args.first() {
                    if let Some(text) = self.interp_value_to_haxe_string(arg)? {
                        rayzor_runtime::haxe_sys::haxe_trace_string(text.as_ptr(), text.len());
                        return Ok(InterpValue::Void);
                    }
                    match arg {
                        InterpValue::String(s) => {
                            rayzor_runtime::haxe_sys::haxe_trace_string(s.as_ptr(), s.len());
                        }
                        InterpValue::Struct(fields) => {
                            // The struct typically has (ptr, len, cap). Accept any integer-like
                            // length representation so interpreted string traces match native backends.
                            if let Some(first) = fields.first() {
                                match first {
                                    InterpValue::String(s) => {
                                        rayzor_runtime::haxe_sys::haxe_trace_string(
                                            s.as_ptr(),
                                            s.len(),
                                        );
                                    }
                                    InterpValue::Ptr(ptr) => {
                                        let len = fields
                                            .get(1)
                                            .and_then(|v| v.to_usize().ok())
                                            .unwrap_or(0);
                                        if *ptr != 0 && len > 0 {
                                            rayzor_runtime::haxe_sys::haxe_trace_string(
                                                *ptr as *const u8,
                                                len,
                                            );
                                        } else {
                                            rayzor_runtime::haxe_sys::haxe_trace_string(
                                                std::ptr::null(),
                                                0,
                                            );
                                        }
                                    }
                                    _ => println!("{:?}", first),
                                }
                            } else {
                                rayzor_runtime::haxe_sys::haxe_trace_string(std::ptr::null(), 0);
                            }
                        }
                        InterpValue::Ptr(ptr) => {
                            if *ptr != 0 {
                                rayzor_runtime::haxe_sys::haxe_trace_string_struct(
                                    *ptr as *const rayzor_runtime::haxe_string::HaxeString,
                                );
                            } else {
                                rayzor_runtime::haxe_sys::haxe_trace_string(std::ptr::null(), 0);
                            }
                        }
                        other => println!("{:?}", other),
                    }
                }
                Ok(InterpValue::Void)
            }
            "haxe_string_from_int" => {
                let value = args
                    .first()
                    .map(|arg| arg.to_i64())
                    .transpose()?
                    .unwrap_or(0);
                Ok(InterpValue::String(value.to_string()))
            }
            "haxe_string_from_float" => {
                let value = args
                    .first()
                    .map(|arg| arg.to_f64())
                    .transpose()?
                    .unwrap_or(0.0);
                Ok(InterpValue::String(value.to_string()))
            }
            "haxe_string_from_bool" => {
                let value = args
                    .first()
                    .map(|arg| arg.to_bool())
                    .transpose()?
                    .unwrap_or(false);
                Ok(InterpValue::String(
                    if value { "true" } else { "false" }.to_string(),
                ))
            }
            // Handle haxe_trace_int for tracing integers
            "haxe_trace_int" => {
                if let Some(arg) = args.first() {
                    match arg {
                        InterpValue::I32(n) => rayzor_runtime::haxe_sys::haxe_trace_int(*n as i64),
                        InterpValue::I64(n) => rayzor_runtime::haxe_sys::haxe_trace_int(*n),
                        InterpValue::I8(n) => rayzor_runtime::haxe_sys::haxe_trace_int(*n as i64),
                        InterpValue::I16(n) => rayzor_runtime::haxe_sys::haxe_trace_int(*n as i64),
                        InterpValue::U8(n) => rayzor_runtime::haxe_sys::haxe_trace_int(*n as i64),
                        InterpValue::U16(n) => rayzor_runtime::haxe_sys::haxe_trace_int(*n as i64),
                        InterpValue::U32(n) => rayzor_runtime::haxe_sys::haxe_trace_int(*n as i64),
                        InterpValue::U64(n) => rayzor_runtime::haxe_sys::haxe_trace_int(*n as i64),
                        other => println!("{:?}", other),
                    }
                }
                Ok(InterpValue::Void)
            }
            "haxe_string_length" => {
                let len = args
                    .first()
                    .map(|arg| self.interp_value_to_haxe_string(arg))
                    .transpose()?
                    .flatten()
                    .map(|s| s.len() as i32)
                    .unwrap_or(0);
                Ok(InterpValue::I32(len))
            }
            "haxe_string_concat" => {
                if args.len() >= 2 {
                    let left = self
                        .interp_value_to_haxe_string(&args[0])?
                        .unwrap_or_else(|| format!("{:?}", &args[0]));
                    let right = self
                        .interp_value_to_haxe_string(&args[1])?
                        .unwrap_or_else(|| format!("{:?}", &args[1]));
                    return Ok(InterpValue::String(format!("{}{}", left, right)));
                }
                Ok(InterpValue::String(String::new()))
            }
            "haxe_std_int" => {
                let x = args
                    .first()
                    .map(|arg| arg.to_f64())
                    .transpose()?
                    .unwrap_or(0.0);
                let truncated = if x.is_nan() {
                    0
                } else if x.is_infinite() {
                    if x.is_sign_positive() {
                        i64::MAX
                    } else {
                        i64::MIN
                    }
                } else {
                    x.trunc() as i64
                };
                Ok(InterpValue::I64(truncated))
            }
            "malloc" => {
                let size = args
                    .first()
                    .map(|arg| arg.to_usize())
                    .transpose()?
                    .unwrap_or(0);
                Ok(InterpValue::Ptr(self.alloc_heap(size)?))
            }
            "free" => {
                if let Some(ptr) = args.first() {
                    self.free_heap(ptr.to_usize()?);
                }
                Ok(InterpValue::Void)
            }
            _ => {
                // Check if we have a registered symbol (call without signature info)
                if let Some(&ptr) = self.runtime_symbols.get(name) {
                    // Without signature info, we can only handle simple cases
                    return self.call_ffi_ptr_simple(ptr as usize, args);
                }
                // Unknown extern - return void
                tracing::warn!("Unknown extern function: {}", name);
                Ok(InterpValue::Void)
            }
        }
    }

    /// Call an IrFunction as FFI (for extern functions with empty blocks)
    fn call_ffi_for_function(
        &mut self,
        func: &IrFunction,
        args: &[InterpValue],
    ) -> Result<InterpValue, InterpError> {
        // First check built-ins
        let builtin_result = self.call_extern(&func.name, args);
        if let Ok(ref val) = builtin_result {
            if !matches!(val, InterpValue::Void) || func.signature.return_type == IrType::Void {
                // If we got a non-void result, or the function is supposed to return void,
                // use the builtin result
                if !func.name.starts_with("Unknown") {
                    return builtin_result;
                }
            }
        }

        // Check if we have a registered symbol
        if let Some(&ptr) = self.runtime_symbols.get(&func.name) {
            return self.call_ffi_with_signature(ptr as usize, args, &func.signature);
        }

        // No symbol found - return default value (warning already logged by builtin check)
        tracing::warn!("Function symbol not found for FFI: {}", func.name);
        Ok(self.default_value_for_type(&func.signature.return_type))
    }

    /// Call an extern function with its full signature for proper FFI
    fn call_extern_with_signature(
        &mut self,
        extern_fn: &IrExternFunction,
        args: &[InterpValue],
    ) -> Result<InterpValue, InterpError> {
        // First check built-ins
        let builtin_result = self.call_extern(&extern_fn.name, args);
        if let Ok(ref val) = builtin_result {
            if !matches!(val, InterpValue::Void) || extern_fn.signature.return_type == IrType::Void
            {
                // If we got a non-void result, or the function is supposed to return void,
                // use the builtin result
                if !extern_fn.name.starts_with("Unknown") {
                    return builtin_result;
                }
            }
        }

        // Check if we have a registered symbol
        if let Some(&ptr) = self.runtime_symbols.get(&extern_fn.name) {
            return self.call_ffi_with_signature(ptr as usize, args, &extern_fn.signature);
        }

        // No symbol found
        tracing::warn!("Extern function not found: {}", extern_fn.name);
        Ok(self.default_value_for_type(&extern_fn.signature.return_type))
    }

    /// Call a function through a raw pointer with signature (proper FFI)
    ///
    /// This uses unsafe Rust to call native function pointers with the correct
    /// calling convention based on the IrFunctionSignature.
    fn call_ffi_with_signature(
        &self,
        ptr: usize,
        args: &[InterpValue],
        signature: &IrFunctionSignature,
    ) -> Result<InterpValue, InterpError> {
        // Convert arguments to native representation (SmallVec disabled - using Vec for stability)
        let native_args: Vec<NativeValue> = args
            .iter()
            .zip(signature.parameters.iter())
            .map(|(arg, param)| self.interp_to_native(arg, &param.ty))
            .collect::<Result<_, _>>()?;

        // Call the function based on arity and return type
        let result = unsafe { self.call_native_fn(ptr, &native_args, &signature.return_type)? };

        // Convert result back to InterpValue
        self.native_to_interp(result, &signature.return_type)
    }

    /// Simple FFI call without signature (for backward compatibility)
    fn call_ffi_ptr_simple(
        &self,
        ptr: usize,
        args: &[InterpValue],
    ) -> Result<InterpValue, InterpError> {
        // Without signature info, we infer types from arguments and assume i64 return
        // SmallVec disabled - using Vec for stability
        let native_args: Vec<NativeValue> = args
            .iter()
            .map(|arg| self.interp_to_native_inferred(arg))
            .collect::<Result<_, _>>()?;

        let result = unsafe { self.call_native_fn(ptr, &native_args, &IrType::I64)? };

        self.native_to_interp(result, &IrType::I64)
    }

    /// FFI call with explicit parameter and return types (from IrType::Function)
    fn call_ffi_ptr_with_types(
        &self,
        ptr: usize,
        args: &[InterpValue],
        param_types: &[IrType],
        return_type: &IrType,
    ) -> Result<InterpValue, InterpError> {
        // Convert arguments to native representation using the explicit types
        // SmallVec disabled - using Vec for stability
        let native_args: Vec<NativeValue> = args
            .iter()
            .enumerate()
            .map(|(i, arg)| {
                let ty = param_types.get(i).unwrap_or(&IrType::I64);
                self.interp_to_native(arg, ty)
            })
            .collect::<Result<_, _>>()?;

        let result = unsafe { self.call_native_fn(ptr, &native_args, return_type)? };

        self.native_to_interp(result, return_type)
    }

    fn call_indirect_raw_i64(
        &mut self,
        module: &IrModule,
        raw_val: i64,
        arg_values: &[InterpValue],
        signature: &IrType,
    ) -> Result<InterpValue, InterpError> {
        let nan_boxed = NanBoxedValue(raw_val as u64);
        let interp_val = InterpValue::from_nan_boxed(nan_boxed, &self.object_heap);

        match interp_val {
            InterpValue::Function(func_id) => {
                if module.functions.contains_key(&func_id) {
                    self.execute(module, func_id, arg_values.to_vec())
                } else if let Some(extern_fn) = module.extern_functions.get(&func_id) {
                    self.call_extern_with_signature(extern_fn, arg_values)
                } else {
                    Err(InterpError::FunctionNotFound(func_id))
                }
            }
            InterpValue::Ptr(heap_idx) => {
                if let Some(HeapObject::Struct(fields)) = self.object_heap.get(heap_idx as u32) {
                    let func_nan = fields[0];
                    let func_interp = InterpValue::from_nan_boxed(func_nan, &self.object_heap);
                    if let InterpValue::Function(func_id) = func_interp {
                        if module.functions.contains_key(&func_id) {
                            self.execute(module, func_id, arg_values.to_vec())
                        } else if let Some(extern_fn) = module.extern_functions.get(&func_id) {
                            self.call_extern_with_signature(extern_fn, arg_values)
                        } else {
                            Err(InterpError::FunctionNotFound(func_id))
                        }
                    } else {
                        Err(InterpError::TypeError(format!(
                            "Closure first field is not a function: {:?}",
                            func_interp
                        )))
                    }
                } else {
                    Err(InterpError::TypeError(format!(
                        "Raw indirect target decoded to pointer-like value without closure backing: 0x{:x}",
                        raw_val as u64
                    )))
                }
            }
            _ => Err(InterpError::TypeError(format!(
                "Raw indirect target is not callable: 0x{:x}",
                raw_val as u64
            ))),
        }
    }

    /// Convert InterpValue to native representation for FFI
    fn interp_to_native(&self, val: &InterpValue, ty: &IrType) -> Result<NativeValue, InterpError> {
        match ty {
            IrType::Void => Ok(NativeValue::Void),
            IrType::Bool => Ok(NativeValue::U8(if val.to_bool()? { 1 } else { 0 })),
            IrType::I8 => Ok(NativeValue::I8(val.to_i64()? as i8)),
            IrType::I16 => Ok(NativeValue::I16(val.to_i64()? as i16)),
            IrType::I32 => Ok(NativeValue::I32(val.to_i64()? as i32)),
            IrType::I64 => Ok(NativeValue::I64(
                val.to_function_bits().unwrap_or(val.to_i64()?),
            )),
            IrType::U8 => Ok(NativeValue::U8(val.to_i64()? as u8)),
            IrType::U16 => Ok(NativeValue::U16(val.to_i64()? as u16)),
            IrType::U32 => Ok(NativeValue::U32(val.to_i64()? as u32)),
            IrType::U64 => Ok(NativeValue::U64(
                val.to_function_bits().unwrap_or(val.to_i64()?) as u64,
            )),
            IrType::F32 => Ok(NativeValue::F32(val.to_f64()? as f32)),
            IrType::F64 => Ok(NativeValue::F64(val.to_f64()?)),
            IrType::Ptr(_) | IrType::Ref(_) => Ok(NativeValue::Ptr(val.to_usize()?)),
            IrType::String => {
                // For string FFI, we pass a pointer to the string data
                match val {
                    InterpValue::String(s) => Ok(NativeValue::Ptr(s.as_ptr() as usize)),
                    InterpValue::Ptr(p) => Ok(NativeValue::Ptr(*p)),
                    _ => Ok(NativeValue::Ptr(0)),
                }
            }
            _ => {
                // For other types, try to pass as pointer
                Ok(NativeValue::Ptr(val.to_usize().unwrap_or(0)))
            }
        }
    }

    /// Convert InterpValue to native, inferring type from the value
    fn interp_to_native_inferred(&self, val: &InterpValue) -> Result<NativeValue, InterpError> {
        match val {
            InterpValue::Void => Ok(NativeValue::Void),
            InterpValue::Bool(b) => Ok(NativeValue::U8(if *b { 1 } else { 0 })),
            InterpValue::I8(n) => Ok(NativeValue::I8(*n)),
            InterpValue::I16(n) => Ok(NativeValue::I16(*n)),
            InterpValue::I32(n) => Ok(NativeValue::I32(*n)),
            InterpValue::I64(n) => Ok(NativeValue::I64(*n)),
            InterpValue::U8(n) => Ok(NativeValue::U8(*n)),
            InterpValue::U16(n) => Ok(NativeValue::U16(*n)),
            InterpValue::U32(n) => Ok(NativeValue::U32(*n)),
            InterpValue::U64(n) => Ok(NativeValue::U64(*n)),
            InterpValue::F32(n) => Ok(NativeValue::F32(*n)),
            InterpValue::F64(n) => Ok(NativeValue::F64(*n)),
            InterpValue::Ptr(p) => Ok(NativeValue::Ptr(*p)),
            InterpValue::Null => Ok(NativeValue::Ptr(0)),
            InterpValue::String(s) => Ok(NativeValue::Ptr(s.as_ptr() as usize)),
            InterpValue::Function(id) => {
                Ok(NativeValue::I64(NanBoxedValue::from_func_id(id.0).0 as i64))
            }
            InterpValue::Array(_) | InterpValue::Struct(_) => {
                // Pass these as pointers (though this is a fallback)
                Ok(NativeValue::Ptr(0))
            }
        }
    }

    /// Convert native value back to InterpValue
    fn native_to_interp(&self, val: NativeValue, ty: &IrType) -> Result<InterpValue, InterpError> {
        match ty {
            IrType::Void => Ok(InterpValue::Void),
            IrType::Bool => match val {
                NativeValue::U8(n) => Ok(InterpValue::Bool(n != 0)),
                NativeValue::I32(n) => Ok(InterpValue::Bool(n != 0)),
                NativeValue::I64(n) => Ok(InterpValue::Bool(n != 0)),
                _ => Ok(InterpValue::Bool(false)),
            },
            IrType::I8 => Ok(InterpValue::I8(val.to_i64() as i8)),
            IrType::I16 => Ok(InterpValue::I16(val.to_i64() as i16)),
            IrType::I32 => Ok(InterpValue::I32(val.to_i64() as i32)),
            IrType::I64 => Ok(InterpValue::I64(val.to_i64())),
            IrType::U8 => Ok(InterpValue::U8(val.to_i64() as u8)),
            IrType::U16 => Ok(InterpValue::U16(val.to_i64() as u16)),
            IrType::U32 => Ok(InterpValue::U32(val.to_i64() as u32)),
            IrType::U64 => Ok(InterpValue::U64(val.to_i64() as u64)),
            IrType::F32 => Ok(InterpValue::F32(val.to_f64() as f32)),
            IrType::F64 => Ok(InterpValue::F64(val.to_f64())),
            IrType::Ptr(_) | IrType::Ref(_) => match val {
                NativeValue::Ptr(p) => Ok(InterpValue::Ptr(p)),
                _ => Ok(InterpValue::Ptr(val.to_i64() as usize)),
            },
            _ => Ok(InterpValue::I64(val.to_i64())),
        }
    }

    /// Get default value for a type
    fn default_value_for_type(&self, ty: &IrType) -> InterpValue {
        match ty {
            IrType::Void => InterpValue::Void,
            IrType::Bool => InterpValue::Bool(false),
            IrType::I8 => InterpValue::I8(0),
            IrType::I16 => InterpValue::I16(0),
            IrType::I32 => InterpValue::I32(0),
            IrType::I64 => InterpValue::I64(0),
            IrType::U8 => InterpValue::U8(0),
            IrType::U16 => InterpValue::U16(0),
            IrType::U32 => InterpValue::U32(0),
            IrType::U64 => InterpValue::U64(0),
            IrType::F32 => InterpValue::F32(0.0),
            IrType::F64 => InterpValue::F64(0.0),
            IrType::Ptr(_) | IrType::Ref(_) => InterpValue::Ptr(0),
            IrType::String => InterpValue::String(String::new()),
            _ => InterpValue::Void,
        }
    }

    /// Call a native function pointer with given arguments
    ///
    /// # Safety
    /// The caller must ensure:
    /// - `ptr` points to a valid function
    /// - `args` match the function's expected signature
    /// - The return type matches the function's actual return type
    unsafe fn call_native_fn(
        &self,
        ptr: usize,
        args: &[NativeValue],
        return_type: &IrType,
    ) -> Result<NativeValue, InterpError> {
        // Convert arguments to u64 for the trampoline
        let arg_values: Vec<u64> = args.iter().map(|a| a.to_u64()).collect();

        // Dispatch based on arity (0-8 arguments supported)
        let result = match arg_values.len() {
            0 => self.call_fn_0(ptr, return_type),
            1 => self.call_fn_1(ptr, arg_values[0], return_type),
            2 => self.call_fn_2(ptr, arg_values[0], arg_values[1], return_type),
            3 => self.call_fn_3(
                ptr,
                arg_values[0],
                arg_values[1],
                arg_values[2],
                return_type,
            ),
            4 => self.call_fn_4(
                ptr,
                arg_values[0],
                arg_values[1],
                arg_values[2],
                arg_values[3],
                return_type,
            ),
            5 => self.call_fn_5(
                ptr,
                arg_values[0],
                arg_values[1],
                arg_values[2],
                arg_values[3],
                arg_values[4],
                return_type,
            ),
            6 => self.call_fn_6(
                ptr,
                arg_values[0],
                arg_values[1],
                arg_values[2],
                arg_values[3],
                arg_values[4],
                arg_values[5],
                return_type,
            ),
            7 => self.call_fn_7(
                ptr,
                arg_values[0],
                arg_values[1],
                arg_values[2],
                arg_values[3],
                arg_values[4],
                arg_values[5],
                arg_values[6],
                return_type,
            ),
            8 => self.call_fn_8(
                ptr,
                arg_values[0],
                arg_values[1],
                arg_values[2],
                arg_values[3],
                arg_values[4],
                arg_values[5],
                arg_values[6],
                arg_values[7],
                return_type,
            ),
            n => {
                return Err(InterpError::RuntimeError(format!(
                    "FFI calls with {} arguments not supported (max 8)",
                    n
                )));
            }
        };

        result
    }

    // FFI trampoline functions for different arities
    // These use the system C calling convention (extern "C")

    unsafe fn call_fn_0(&self, ptr: usize, ret_ty: &IrType) -> Result<NativeValue, InterpError> {
        if ret_ty.is_float() {
            let f: extern "C" fn() -> f64 = std::mem::transmute(ptr);
            Ok(NativeValue::F64(f()))
        } else {
            let f: extern "C" fn() -> u64 = std::mem::transmute(ptr);
            Ok(NativeValue::U64(f()))
        }
    }

    unsafe fn call_fn_1(
        &self,
        ptr: usize,
        a0: u64,
        ret_ty: &IrType,
    ) -> Result<NativeValue, InterpError> {
        if ret_ty.is_float() {
            let f: extern "C" fn(u64) -> f64 = std::mem::transmute(ptr);
            Ok(NativeValue::F64(f(a0)))
        } else {
            let f: extern "C" fn(u64) -> u64 = std::mem::transmute(ptr);
            Ok(NativeValue::U64(f(a0)))
        }
    }

    unsafe fn call_fn_2(
        &self,
        ptr: usize,
        a0: u64,
        a1: u64,
        ret_ty: &IrType,
    ) -> Result<NativeValue, InterpError> {
        if ret_ty.is_float() {
            let f: extern "C" fn(u64, u64) -> f64 = std::mem::transmute(ptr);
            Ok(NativeValue::F64(f(a0, a1)))
        } else {
            let f: extern "C" fn(u64, u64) -> u64 = std::mem::transmute(ptr);
            Ok(NativeValue::U64(f(a0, a1)))
        }
    }

    unsafe fn call_fn_3(
        &self,
        ptr: usize,
        a0: u64,
        a1: u64,
        a2: u64,
        ret_ty: &IrType,
    ) -> Result<NativeValue, InterpError> {
        if ret_ty.is_float() {
            let f: extern "C" fn(u64, u64, u64) -> f64 = std::mem::transmute(ptr);
            Ok(NativeValue::F64(f(a0, a1, a2)))
        } else {
            let f: extern "C" fn(u64, u64, u64) -> u64 = std::mem::transmute(ptr);
            Ok(NativeValue::U64(f(a0, a1, a2)))
        }
    }

    unsafe fn call_fn_4(
        &self,
        ptr: usize,
        a0: u64,
        a1: u64,
        a2: u64,
        a3: u64,
        ret_ty: &IrType,
    ) -> Result<NativeValue, InterpError> {
        if ret_ty.is_float() {
            let f: extern "C" fn(u64, u64, u64, u64) -> f64 = std::mem::transmute(ptr);
            Ok(NativeValue::F64(f(a0, a1, a2, a3)))
        } else {
            let f: extern "C" fn(u64, u64, u64, u64) -> u64 = std::mem::transmute(ptr);
            Ok(NativeValue::U64(f(a0, a1, a2, a3)))
        }
    }

    unsafe fn call_fn_5(
        &self,
        ptr: usize,
        a0: u64,
        a1: u64,
        a2: u64,
        a3: u64,
        a4: u64,
        ret_ty: &IrType,
    ) -> Result<NativeValue, InterpError> {
        if ret_ty.is_float() {
            let f: extern "C" fn(u64, u64, u64, u64, u64) -> f64 = std::mem::transmute(ptr);
            Ok(NativeValue::F64(f(a0, a1, a2, a3, a4)))
        } else {
            let f: extern "C" fn(u64, u64, u64, u64, u64) -> u64 = std::mem::transmute(ptr);
            Ok(NativeValue::U64(f(a0, a1, a2, a3, a4)))
        }
    }

    unsafe fn call_fn_6(
        &self,
        ptr: usize,
        a0: u64,
        a1: u64,
        a2: u64,
        a3: u64,
        a4: u64,
        a5: u64,
        ret_ty: &IrType,
    ) -> Result<NativeValue, InterpError> {
        if ret_ty.is_float() {
            let f: extern "C" fn(u64, u64, u64, u64, u64, u64) -> f64 = std::mem::transmute(ptr);
            Ok(NativeValue::F64(f(a0, a1, a2, a3, a4, a5)))
        } else {
            let f: extern "C" fn(u64, u64, u64, u64, u64, u64) -> u64 = std::mem::transmute(ptr);
            Ok(NativeValue::U64(f(a0, a1, a2, a3, a4, a5)))
        }
    }

    unsafe fn call_fn_7(
        &self,
        ptr: usize,
        a0: u64,
        a1: u64,
        a2: u64,
        a3: u64,
        a4: u64,
        a5: u64,
        a6: u64,
        ret_ty: &IrType,
    ) -> Result<NativeValue, InterpError> {
        if ret_ty.is_float() {
            let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64) -> f64 =
                std::mem::transmute(ptr);
            Ok(NativeValue::F64(f(a0, a1, a2, a3, a4, a5, a6)))
        } else {
            let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64) -> u64 =
                std::mem::transmute(ptr);
            Ok(NativeValue::U64(f(a0, a1, a2, a3, a4, a5, a6)))
        }
    }

    unsafe fn call_fn_8(
        &self,
        ptr: usize,
        a0: u64,
        a1: u64,
        a2: u64,
        a3: u64,
        a4: u64,
        a5: u64,
        a6: u64,
        a7: u64,
        ret_ty: &IrType,
    ) -> Result<NativeValue, InterpError> {
        if ret_ty.is_float() {
            let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64) -> f64 =
                std::mem::transmute(ptr);
            Ok(NativeValue::F64(f(a0, a1, a2, a3, a4, a5, a6, a7)))
        } else {
            let f: extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64) -> u64 =
                std::mem::transmute(ptr);
            Ok(NativeValue::U64(f(a0, a1, a2, a3, a4, a5, a6, a7)))
        }
    }
}

/// Native value representation for FFI calls
///
/// Uses a flat representation that can be easily converted to/from u64
/// for passing through the FFI trampoline.
#[derive(Debug, Clone, Copy)]
enum NativeValue {
    Void,
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    F32(f32),
    F64(f64),
    Ptr(usize),
}

impl NativeValue {
    /// Convert to u64 for FFI (all integer/pointer types fit in u64)
    fn to_u64(&self) -> u64 {
        match self {
            NativeValue::Void => 0,
            NativeValue::I8(n) => *n as i64 as u64,
            NativeValue::I16(n) => *n as i64 as u64,
            NativeValue::I32(n) => *n as i64 as u64,
            NativeValue::I64(n) => *n as u64,
            NativeValue::U8(n) => *n as u64,
            NativeValue::U16(n) => *n as u64,
            NativeValue::U32(n) => *n as u64,
            NativeValue::U64(n) => *n,
            NativeValue::F32(n) => (*n as f64).to_bits(),
            NativeValue::F64(n) => n.to_bits(),
            NativeValue::Ptr(p) => *p as u64,
        }
    }

    /// Convert to i64
    fn to_i64(&self) -> i64 {
        match self {
            NativeValue::Void => 0,
            NativeValue::I8(n) => *n as i64,
            NativeValue::I16(n) => *n as i64,
            NativeValue::I32(n) => *n as i64,
            NativeValue::I64(n) => *n,
            NativeValue::U8(n) => *n as i64,
            NativeValue::U16(n) => *n as i64,
            NativeValue::U32(n) => *n as i64,
            NativeValue::U64(n) => *n as i64,
            NativeValue::F32(n) => *n as i64,
            NativeValue::F64(n) => *n as i64,
            NativeValue::Ptr(p) => *p as i64,
        }
    }

    /// Convert to f64
    fn to_f64(&self) -> f64 {
        match self {
            NativeValue::Void => 0.0,
            NativeValue::I8(n) => *n as f64,
            NativeValue::I16(n) => *n as f64,
            NativeValue::I32(n) => *n as f64,
            NativeValue::I64(n) => *n as f64,
            NativeValue::U8(n) => *n as f64,
            NativeValue::U16(n) => *n as f64,
            NativeValue::U32(n) => *n as f64,
            NativeValue::U64(n) => *n as f64,
            NativeValue::F32(n) => *n as f64,
            NativeValue::F64(n) => *n,
            NativeValue::Ptr(p) => *p as f64,
        }
    }
}

impl Default for MirInterpreter {
    fn default() -> Self {
        Self::new()
    }
}

/// Interpreter error types
#[derive(Debug)]
pub enum InterpError {
    FunctionNotFound(IrFunctionId),
    BlockNotFound(IrBlockId),
    StackOverflow,
    TypeError(String),
    RuntimeError(String),
    Panic(String),
    Exception(String),
    /// Hot loop detected - signal to promote function to JIT
    JitBailout(IrFunctionId),
}

impl std::fmt::Display for InterpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InterpError::FunctionNotFound(id) => write!(f, "Function not found: {:?}", id),
            InterpError::BlockNotFound(id) => write!(f, "Block not found: {:?}", id),
            InterpError::StackOverflow => write!(f, "Stack overflow"),
            InterpError::TypeError(msg) => write!(f, "Type error: {}", msg),
            InterpError::RuntimeError(msg) => write!(f, "Runtime error: {}", msg),
            InterpError::Panic(msg) => write!(f, "Panic: {}", msg),
            InterpError::Exception(msg) => write!(f, "Exception: {}", msg),
            InterpError::JitBailout(id) => write!(f, "JIT bailout requested for {:?}", id),
        }
    }
}

impl std::error::Error for InterpError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interp_value_conversions() {
        let i32_val = InterpValue::I32(42);
        assert_eq!(i32_val.to_i64().unwrap(), 42);
        assert_eq!(i32_val.to_bool().unwrap(), true);

        let zero_val = InterpValue::I32(0);
        assert_eq!(zero_val.to_bool().unwrap(), false);

        let f64_val = InterpValue::F64(3.14);
        assert!((f64_val.to_f64().unwrap() - 3.14).abs() < 0.001);
    }

    #[test]
    fn test_register_file() {
        let mut regs = RegisterFile::new(10);
        let mut heap = ObjectHeap::new();
        regs.set(IrId::new(0), InterpValue::I32(100).to_nan_boxed(&mut heap));
        regs.set(
            IrId::new(5),
            InterpValue::Bool(true).to_nan_boxed(&mut heap),
        );

        let _100 = InterpValue::I32(100).to_nan_boxed(&mut heap);
        let _true = InterpValue::Bool(true).to_nan_boxed(&mut heap);
        let _void = InterpValue::Void.to_nan_boxed(&mut heap);
        assert!(matches!(regs.get(IrId::new(0)), _100));
        assert!(matches!(regs.get(IrId::new(5)), _true));
        assert!(matches!(regs.get(IrId::new(3)), _void));
    }

    #[test]
    fn test_binary_ops() {
        let interp = MirInterpreter::new();

        // Integer operations
        let result = interp
            .eval_binary_op(BinaryOp::Add, InterpValue::I64(10), InterpValue::I64(20))
            .unwrap();
        assert!(matches!(result, InterpValue::I64(30)));

        let result = interp
            .eval_binary_op(BinaryOp::Mul, InterpValue::I64(5), InterpValue::I64(6))
            .unwrap();
        assert!(matches!(result, InterpValue::I64(30)));

        // Floating point operations
        let result = interp
            .eval_binary_op(BinaryOp::FAdd, InterpValue::F64(1.5), InterpValue::F64(2.5))
            .unwrap();
        if let InterpValue::F64(v) = result {
            assert!((v - 4.0).abs() < 0.001);
        } else {
            panic!("Expected F64");
        }
    }

    #[test]
    fn test_compare_ops() {
        let interp = MirInterpreter::new();

        assert!(interp
            .eval_compare_op(CompareOp::Lt, InterpValue::I64(5), InterpValue::I64(10))
            .unwrap());
        assert!(!interp
            .eval_compare_op(CompareOp::Lt, InterpValue::I64(10), InterpValue::I64(5))
            .unwrap());
        assert!(interp
            .eval_compare_op(CompareOp::Eq, InterpValue::I64(5), InterpValue::I64(5))
            .unwrap());
    }

    // FFI test functions (extern "C" for proper ABI)
    extern "C" fn test_add(a: u64, b: u64) -> u64 {
        a + b
    }

    extern "C" fn test_mul(a: u64, b: u64, c: u64) -> u64 {
        a * b * c
    }

    extern "C" fn test_f64_add(a: u64, b: u64) -> f64 {
        let a = f64::from_bits(a);
        let b = f64::from_bits(b);
        a + b
    }

    #[test]
    fn test_ffi_call_simple() {
        let mut interp = MirInterpreter::new();

        // Register the test_add function
        interp.register_symbol("test_add", test_add as *const u8);

        // Call through simple FFI (without signature)
        let result = interp
            .call_ffi_ptr_simple(
                test_add as *const () as usize,
                &[InterpValue::I64(5), InterpValue::I64(3)],
            )
            .unwrap();

        match result {
            InterpValue::I64(n) => assert_eq!(n, 8),
            _ => panic!("Expected I64 result, got {:?}", result),
        }
    }

    #[test]
    fn test_ffi_call_with_types() {
        let interp = MirInterpreter::new();

        // Call with explicit type information
        let param_types = vec![IrType::I64, IrType::I64, IrType::I64];
        let return_type = IrType::I64;

        let result = interp
            .call_ffi_ptr_with_types(
                test_mul as *const () as usize,
                &[
                    InterpValue::I64(2),
                    InterpValue::I64(3),
                    InterpValue::I64(4),
                ],
                &param_types,
                &return_type,
            )
            .unwrap();

        match result {
            InterpValue::I64(n) => assert_eq!(n, 24), // 2 * 3 * 4 = 24
            _ => panic!("Expected I64 result, got {:?}", result),
        }
    }

    #[test]
    fn test_ffi_call_float_return() {
        let interp = MirInterpreter::new();

        // Call function that returns f64
        let param_types = vec![IrType::F64, IrType::F64];
        let return_type = IrType::F64;

        let result = interp
            .call_ffi_ptr_with_types(
                test_f64_add as *const () as usize,
                &[InterpValue::F64(1.5), InterpValue::F64(2.5)],
                &param_types,
                &return_type,
            )
            .unwrap();

        match result {
            InterpValue::F64(n) => assert!((n - 4.0).abs() < 0.001),
            _ => panic!("Expected F64 result, got {:?}", result),
        }
    }

    #[test]
    fn test_builtin_haxe_std_int() {
        let mut interp = MirInterpreter::new();

        let positive = interp
            .call_extern("haxe_std_int", &[InterpValue::F64(5.9)])
            .expect("positive std int");
        match positive {
            InterpValue::I64(n) => assert_eq!(n, 5),
            other => panic!("Expected I64 result, got {:?}", other),
        }

        let negative = interp
            .call_extern("haxe_std_int", &[InterpValue::F64(-5.9)])
            .expect("negative std int");
        match negative {
            InterpValue::I64(n) => assert_eq!(n, -5),
            other => panic!("Expected I64 result, got {:?}", other),
        }
    }

    #[test]
    fn test_native_value_conversions() {
        // Test u64 conversion for various types
        assert_eq!(NativeValue::I32(42).to_u64(), 42);
        assert_eq!(NativeValue::I64(-1).to_u64(), u64::MAX);
        assert_eq!(NativeValue::U8(255).to_u64(), 255);
        assert_eq!(NativeValue::Ptr(0x12345678).to_u64(), 0x12345678);

        // Test i64 conversion
        assert_eq!(NativeValue::I32(-5).to_i64(), -5);
        assert_eq!(NativeValue::U32(100).to_i64(), 100);

        // Test f64 conversion
        assert!((NativeValue::F32(3.14f32).to_f64() - 3.14f64).abs() < 0.01);
        assert_eq!(NativeValue::I64(42).to_f64(), 42.0);
    }
}
