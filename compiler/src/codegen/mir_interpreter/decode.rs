//! MIR to the interpreter's typed form.
//!
//! Every value gets the machine type the Cranelift lowering gives it, and every
//! conversion that lowering inserts implicitly is made an explicit operation
//! here, so an interpreted function computes the values compiled code would.
//! Where the lowering would refuse an instruction, the decoded function traps
//! at that instruction instead.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use crate::ir::{
    AtomicRmwOp, BinaryOp, CallingConvention, CompareOp, IrBlockId, IrFunction, IrFunctionId, IrId,
    IrInstruction, IrModule, IrTerminator, IrType, IrValue, Linkage, UnaryOp,
};

use super::Program;

pub(super) type Reg = u32;

/// A value's machine type: the Cranelift type the lowering gives it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Mt {
    I8,
    I16,
    I32,
    I64,
    F32,
    F64,
    /// A SIMD vector. The interpreter hands these to compiled code.
    V128,
}

impl Mt {
    /// `mir_type_to_cranelift_static`; `None` for `Void`.
    pub(crate) fn of(ty: &IrType) -> Option<Mt> {
        Some(match ty {
            IrType::Void => return None,
            IrType::I8 | IrType::U8 | IrType::Bool => Mt::I8,
            IrType::I16 | IrType::U16 => Mt::I16,
            IrType::I32 | IrType::U32 => Mt::I32,
            IrType::F32 => Mt::F32,
            IrType::F64 => Mt::F64,
            IrType::Vector { .. } => Mt::V128,
            _ => Mt::I64,
        })
    }

    /// A phi's block parameter type, which maps fewer types than `of`.
    fn phi(ty: &IrType) -> Mt {
        match ty {
            IrType::I8 | IrType::Bool => Mt::I8,
            IrType::I16 => Mt::I16,
            IrType::I32 => Mt::I32,
            IrType::F32 => Mt::F32,
            IrType::F64 => Mt::F64,
            IrType::Vector { .. } => Mt::V128,
            _ => Mt::I64,
        }
    }

    pub(crate) fn bits(self) -> u32 {
        match self {
            Mt::I8 => 8,
            Mt::I16 => 16,
            Mt::I32 | Mt::F32 => 32,
            Mt::I64 | Mt::F64 => 64,
            Mt::V128 => 128,
        }
    }

    pub(crate) fn bytes(self) -> usize {
        (self.bits() / 8) as usize
    }

    pub(crate) fn is_int(self) -> bool {
        matches!(self, Mt::I8 | Mt::I16 | Mt::I32 | Mt::I64)
    }

    pub(crate) fn is_float(self) -> bool {
        matches!(self, Mt::F32 | Mt::F64)
    }
}

/// The low `mt` bits of `bits`, zero-extended: how a register holds a value.
#[inline(always)]
pub(crate) fn canon(bits: u64, mt: Mt) -> u64 {
    match mt {
        Mt::I8 => bits & 0xff,
        Mt::I16 => bits & 0xffff,
        Mt::I32 | Mt::F32 => bits & 0xffff_ffff,
        _ => bits,
    }
}

#[inline(always)]
pub(crate) fn sext(bits: u64, mt: Mt) -> i64 {
    match mt {
        Mt::I8 => bits as u8 as i8 as i64,
        Mt::I16 => bits as u16 as i16 as i64,
        Mt::I32 => bits as u32 as i32 as i64,
        _ => bits as i64,
    }
}

#[inline(always)]
pub(crate) fn float(bits: u64, mt: Mt) -> f64 {
    if mt == Mt::F32 {
        f32::from_bits(bits as u32) as f64
    } else {
        f64::from_bits(bits)
    }
}

#[inline(always)]
pub(crate) fn float_bits(v: f64, mt: Mt) -> u64 {
    if mt == Mt::F32 {
        (v as f32).to_bits() as u64
    } else {
        v.to_bits()
    }
}

/// One Cranelift conversion instruction.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Conv {
    Sext(Mt, Mt),
    Uext(Mt, Mt),
    Reduce(Mt),
    SToF(Mt, Mt),
    UToF(Mt, Mt),
    /// `fcvt_to_sint`: traps on NaN and on overflow.
    FToS(Mt, Mt),
    FToU(Mt, Mt),
    FToSSat(Mt, Mt),
    FPromote,
    FDemote,
    /// A reinterpretation, or a width change Cranelift leaves to the backend.
    Bits(Mt),
}

impl Conv {
    pub(crate) fn to(self) -> Mt {
        match self {
            Conv::Sext(_, t)
            | Conv::Uext(_, t)
            | Conv::SToF(_, t)
            | Conv::UToF(_, t)
            | Conv::FToS(_, t)
            | Conv::FToU(_, t)
            | Conv::FToSSat(_, t) => t,
            Conv::Reduce(t) | Conv::Bits(t) => t,
            Conv::FPromote => Mt::F64,
            Conv::FDemote => Mt::F32,
        }
    }

    pub(crate) fn apply(self, bits: u64) -> Result<u64, &'static str> {
        Ok(match self {
            Conv::Sext(from, to) => canon(sext(bits, from) as u64, to),
            Conv::Uext(_, to) | Conv::Reduce(to) | Conv::Bits(to) => canon(bits, to),
            Conv::SToF(from, to) => float_bits(sext(bits, from) as f64, to),
            Conv::UToF(_, to) => float_bits(bits as f64, to),
            Conv::FToS(from, to) => {
                let v = float(bits, from);
                if v.is_nan() {
                    return Err("bad conversion to integer");
                }
                let t = v.trunc();
                let half = 2f64.powi(to.bits() as i32 - 1);
                if t < -half || t >= half {
                    return Err("integer overflow");
                }
                canon(t as i64 as u64, to)
            }
            Conv::FToU(from, to) => {
                let v = float(bits, from);
                if v.is_nan() {
                    return Err("bad conversion to integer");
                }
                let t = v.trunc();
                if t < 0.0 || t >= 2f64.powi(to.bits() as i32) {
                    return Err("integer overflow");
                }
                canon(t as u64, to)
            }
            Conv::FToSSat(from, to) => {
                let v = float(bits, from);
                let n = match to {
                    Mt::I8 => v as i8 as i64,
                    Mt::I16 => v as i16 as i64,
                    Mt::I32 => v as i32 as i64,
                    _ => v as i64,
                };
                canon(n as u64, to)
            }
            Conv::FPromote => (f32::from_bits(bits as u32) as f64).to_bits(),
            Conv::FDemote => (f64::from_bits(bits) as f32).to_bits() as u64,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum IntOp {
    Add,
    Sub,
    Mul,
    SDiv,
    UDiv,
    SRem,
    URem,
    And,
    Or,
    Xor,
    Shl,
    SShr,
    UShr,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum FloatOp {
    Add,
    Sub,
    Mul,
    Div,
    /// `a - trunc(a / b) * b`.
    Rem,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum IntCc {
    Eq,
    Ne,
    Slt,
    Sle,
    Sgt,
    Sge,
    Ult,
    Ule,
    Ugt,
    Uge,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum FloatCc {
    Eq,
    /// Unordered or not equal.
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Ord,
    Uno,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Intrinsic {
    Sqrt,
    Abs,
    Floor,
    Ceil,
    Round,
}

/// A call argument.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Src {
    Reg(Reg),
    Imm(u64),
    /// The address of a frame slot.
    Slot(u32),
}

#[derive(Clone, Debug)]
pub(crate) enum Target {
    /// A function with a body, entered in the interpreter.
    Interp {
        module: usize,
        func: IrFunctionId,
    },
    Native(usize),
}

#[derive(Debug)]
pub(crate) struct Call {
    pub target: Target,
    /// In the callee's ABI order, hidden parameters first.
    pub args: Vec<(Src, Mt)>,
    /// What the callee returns in a register.
    pub ret: Option<Mt>,
    /// A struct return: the slot the callee writes, and its size.
    pub sret: Option<(u32, u32)>,
    pub dest: Option<Reg>,
}

#[derive(Debug)]
pub(crate) struct IndirectCall {
    pub closure: Reg,
    /// Excluding the environment, which comes from the closure.
    pub args: Vec<(Reg, Mt)>,
    pub ret: Option<Mt>,
    pub dest: Option<Reg>,
}

#[derive(Debug)]
pub(crate) enum Op {
    Const {
        dest: Reg,
        bits: u64,
    },
    Copy {
        dest: Reg,
        src: Reg,
    },
    Conv {
        dest: Reg,
        src: Reg,
        conv: Conv,
    },
    /// `haxe_string_literal(ptr, len)` on bytes that live for the process.
    Str {
        dest: Reg,
        literal: usize,
        ptr: usize,
        len: usize,
    },
    IBin {
        dest: Reg,
        op: IntOp,
        mt: Mt,
        a: Reg,
        b: Reg,
    },
    FBin {
        dest: Reg,
        op: FloatOp,
        mt: Mt,
        a: Reg,
        b: Reg,
    },
    /// `a * b + c`, rounded once (`RAYZOR_FMA=1`).
    Fma {
        dest: Reg,
        mt: Mt,
        a: Reg,
        b: Reg,
        c: Reg,
        neg_a: bool,
        neg_c: bool,
    },
    ICmp {
        dest: Reg,
        cc: IntCc,
        mt: Mt,
        a: Reg,
        b: Reg,
    },
    FCmp {
        dest: Reg,
        cc: FloatCc,
        mt: Mt,
        a: Reg,
        b: Reg,
    },
    INeg {
        dest: Reg,
        mt: Mt,
        a: Reg,
    },
    FNeg {
        dest: Reg,
        mt: Mt,
        a: Reg,
    },
    BNot {
        dest: Reg,
        mt: Mt,
        a: Reg,
    },
    IsZero {
        dest: Reg,
        a: Reg,
    },
    Select {
        dest: Reg,
        cond: Reg,
        a: Reg,
        b: Reg,
    },
    Load {
        dest: Reg,
        mt: Mt,
        ptr: Reg,
        offset: i32,
    },
    Store {
        mt: Mt,
        ptr: Reg,
        val: Reg,
        offset: i32,
    },
    SlotAddr {
        dest: Reg,
        offset: u32,
    },
    /// `base + sext(index) * scale`.
    AddrIndex {
        dest: Reg,
        base: Reg,
        index: Reg,
        index_mt: Mt,
        scale: i64,
    },
    ArrayHeader {
        dest: Reg,
        array: Reg,
        offset: i32,
        guarded: bool,
    },
    ArrayGetPtr {
        dest: Reg,
        array: Reg,
        index: Reg,
        index_mt: Mt,
    },
    Math {
        dest: Reg,
        kind: Intrinsic,
        arg: Reg,
        mt: Mt,
    },
    GlobalLoad {
        dest: Reg,
        mt: Mt,
        slot: GlobalSlot,
    },
    GlobalStore {
        src: Reg,
        mt: Mt,
        slot: GlobalSlot,
    },
    Call(Box<Call>),
    CallIndirect(Box<IndirectCall>),
    /// The callee's code address: compiled code or a trampoline.
    FnAddr {
        dest: Reg,
        module: usize,
        func: IrFunctionId,
    },
    /// A heap closure object `{fn, env}`, as `FunctionRef` and `MakeClosure` build.
    Closure {
        dest: Reg,
        func: FnRef,
        captures: Vec<(Reg, Mt)>,
    },
    Env {
        dest: Reg,
    },
    /// `malloc(64)` and a 64-byte copy, as the lowering does.
    CloneObject {
        dest: Reg,
        src: Reg,
    },
    Free {
        ptr: Reg,
    },
    MemCopy {
        dest: Reg,
        src: Reg,
        size: Reg,
        size_mt: Mt,
    },
    MemSet {
        dest: Reg,
        val: Reg,
        size: Reg,
        size_mt: Mt,
    },
    AtomicLoad {
        dest: Reg,
        mt: Mt,
        ptr: Reg,
    },
    AtomicStore {
        mt: Mt,
        ptr: Reg,
        val: Reg,
    },
    AtomicRmw {
        dest: Reg,
        mt: Mt,
        op: AtomicRmwOp,
        ptr: Reg,
        val: Reg,
    },
    AtomicCas {
        dest: Reg,
        mt: Mt,
        ptr: Reg,
        expected: Reg,
        replacement: Reg,
    },
    MarkMoved {
        slot: u32,
    },
    CheckLive {
        slot: u32,
        name: Arc<str>,
        line: u32,
    },
    /// `_setjmp(buf)`, run so that the rest of the frame executes inside it.
    SetJmp {
        dest: Reg,
        buf: Reg,
    },
    /// Follows a call to `rayzor_exception_pop_handler`: the try whose
    /// `setjmp` this frame is running inside may have ended.
    HandlerPopped,
    /// Hand this function to compiled code.
    Bail,
    Trap {
        message: Arc<str>,
    },
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum GlobalSlot {
    /// The runtime's slot for this global, read and written in place.
    Address(usize),
    /// Through `rayzor_global_load` / `rayzor_global_store`.
    Runtime { id: u32, load: usize, store: usize },
}

#[derive(Clone, Debug)]
pub(crate) enum FnRef {
    Address(usize),
    Function { module: usize, func: IrFunctionId },
}

#[derive(Debug, Default)]
pub(crate) struct Edge {
    pub block: u32,
    /// Parallel moves into the target's phis.
    pub moves: Vec<(Reg, Reg, Option<Conv>, Option<Conv>)>,
}

#[derive(Debug)]
pub(crate) enum Term {
    Jump(Edge),
    Brif {
        cond: Reg,
        then: Edge,
        otherwise: Edge,
    },
    Switch {
        value: Reg,
        mt: Mt,
        cases: Vec<(i64, Edge)>,
        default: Edge,
    },
    Return(Option<Reg>),
    /// Copy a struct into the caller's return slot.
    ReturnSret {
        value: Reg,
        fields: Vec<(i32, Mt)>,
    },
    Trap(Arc<str>),
}

#[derive(Debug)]
pub(crate) struct Block {
    pub ops: Vec<Op>,
    pub term: Term,
}

/// A function's native signature.
#[derive(Clone, Debug)]
pub(crate) struct Abi {
    pub sret: bool,
    pub hidden_env: bool,
    /// Every parameter in order, hidden ones first.
    pub params: Vec<Mt>,
    pub ret: Option<Mt>,
}

#[derive(Debug)]
pub(crate) struct Function {
    pub id: IrFunctionId,
    pub name: Arc<str>,
    pub abi: Abi,
    pub return_type: IrType,
    /// MIR parameters in order, after the hidden ones.
    pub params: Vec<(Reg, Mt)>,
    /// A lambda's first parameter is its environment.
    pub explicit_env: bool,
    pub registers: usize,
    pub slot_words: usize,
    pub live_slots: usize,
    pub entry: u32,
    pub blocks: Vec<Block>,
}

/// Whether `function` takes a hidden environment parameter in compiled code.
pub(crate) fn needs_env(
    function: &IrFunction,
    id: IrFunctionId,
    indirect: &BTreeSet<IrFunctionId>,
) -> bool {
    crate::ir::abi::needs_env_param(function, id, indirect)
}

/// The native signature compiled code gives `function`.
pub(crate) fn abi_of(function: &IrFunction, hidden_env: bool) -> Abi {
    let sig = &function.signature;
    let sret = sig.uses_sret;
    let is_extern = function.cfg.blocks.is_empty();
    let c_extern = is_extern && sig.calling_convention == CallingConvention::C && !cfg!(windows);
    let mut params = Vec::with_capacity(sig.parameters.len() + 2);
    if sret {
        params.push(Mt::I64);
    }
    if hidden_env {
        params.push(Mt::I64);
    }
    for p in &sig.parameters {
        let widened =
            c_extern && matches!(p.ty, IrType::I32 | IrType::U32 | IrType::Bool | IrType::I8);
        params.push(if widened {
            Mt::I64
        } else {
            Mt::of(&p.ty).unwrap_or(Mt::I64)
        });
    }
    let ret = if sret { None } else { Mt::of(&sig.return_type) };
    Abi {
        sret,
        hidden_env,
        params,
        ret,
    }
}

/// `CraneliftBackend::type_size`: struct field and pointee sizes.
fn field_size(ty: &IrType) -> usize {
    match ty {
        IrType::I8 | IrType::U8 | IrType::Bool => 1,
        IrType::I16 | IrType::U16 => 2,
        IrType::I32 | IrType::U32 | IrType::F32 => 4,
        IrType::Void => 0,
        IrType::Vector { element, count } => field_size(element) * count,
        _ => 8,
    }
}

/// The lowering's `type_size` for stack allocations.
fn alloca_size(ty: &IrType) -> usize {
    match ty {
        IrType::I8 | IrType::U8 | IrType::Bool => 1,
        IrType::I16 | IrType::U16 => 2,
        IrType::I32 | IrType::U32 | IrType::F32 => 4,
        IrType::Array(element, count) => alloca_size(element) * count,
        _ => 8,
    }
}

fn struct_layout(ty: &IrType) -> Option<Vec<(i32, Mt, usize)>> {
    let IrType::Struct { fields, .. } = ty else {
        return None;
    };
    let mut offset = 0usize;
    let mut out = Vec::with_capacity(fields.len());
    for f in fields {
        out.push((
            offset as i32,
            Mt::of(&f.ty).unwrap_or(Mt::I64),
            field_size(&f.ty),
        ));
        offset += field_size(&f.ty);
    }
    Some(out)
}

fn struct_size(ty: &IrType) -> usize {
    struct_layout(ty)
        .map(|l| l.iter().map(|f| f.2).sum())
        .unwrap_or(8)
}

fn is_signed(ty: &IrType) -> bool {
    matches!(ty, IrType::I8 | IrType::I16 | IrType::I32 | IrType::I64)
}

fn is_unsigned(ty: &IrType) -> bool {
    matches!(ty, IrType::U8 | IrType::U16 | IrType::U32 | IrType::U64)
}

fn is_primitive_int(ty: &IrType) -> bool {
    is_signed(ty) || is_unsigned(ty) || *ty == IrType::Bool
}

/// The Cranelift type of a primitive integer register, or `I64`.
fn int_register_mt(ty: Option<&IrType>) -> Mt {
    match ty {
        Some(IrType::I8 | IrType::U8 | IrType::Bool) => Mt::I8,
        Some(IrType::I16 | IrType::U16) => Mt::I16,
        Some(IrType::I32 | IrType::U32) => Mt::I32,
        _ => Mt::I64,
    }
}

/// The lowering's integer-to-integer resize at a call or phi edge.
fn resize(from: Mt, to: Mt, signed: bool) -> Option<Conv> {
    if from == to || !from.is_int() || !to.is_int() {
        None
    } else if from.bits() > to.bits() {
        Some(Conv::Reduce(to))
    } else if signed {
        Some(Conv::Sext(from, to))
    } else {
        Some(Conv::Uext(from, to))
    }
}

/// Conversion of a phi argument to its block parameter type.
fn phi_coercion(from: Mt, to: Mt) -> (Option<Conv>, Option<Conv>) {
    use Mt::*;
    let one = |c| (Some(c), None);
    match (from, to) {
        (a, b) if a == b => (None, None),
        (I64, I32) | (I64, I8) | (I32, I8) => one(Conv::Reduce(to)),
        (I32, I64) | (I8, I32) | (I8, I64) => one(Conv::Sext(from, to)),
        (I8 | I32 | I64, F32 | F64) => one(Conv::SToF(from, to)),
        (F32 | F64, I8) => (Some(Conv::FToS(from, I32)), Some(Conv::Reduce(I8))),
        (F32 | F64, I32 | I64) => one(Conv::FToS(from, to)),
        (F32, F64) => one(Conv::FPromote),
        (F64, F32) => one(Conv::FDemote),
        _ => one(Conv::Bits(to)),
    }
}

/// Coercion of a closure-call argument to the call-site signature.
fn indirect_arg_coercion(from: Mt, to: Mt) -> Option<Conv> {
    if from == to {
        None
    } else if from.is_int() && to.is_int() {
        if from.bits() < to.bits() {
            if from == Mt::I8 {
                Some(Conv::Uext(from, to))
            } else {
                Some(Conv::Sext(from, to))
            }
        } else {
            Some(Conv::Reduce(to))
        }
    } else if from.is_int() && to.is_float() {
        Some(Conv::SToF(from, to))
    } else if from == Mt::F32 && to == Mt::F64 {
        Some(Conv::FPromote)
    } else if from == Mt::F64 && to == Mt::F32 {
        Some(Conv::FDemote)
    } else {
        None
    }
}

/// Coercion of an argument to a Haxe-ABI callee's parameter.
fn haxe_arg_coercion(from: Mt, to: Mt) -> Option<Conv> {
    use Mt::*;
    match (from, to) {
        (a, b) if a == b => None,
        (I32, I64) => Some(Conv::Sext(I32, I64)),
        (I64, I32) => Some(Conv::Reduce(I32)),
        (I32 | I64, F64 | F32) => Some(Conv::SToF(from, to)),
        (F64, I64 | I32) => Some(Conv::FToS(F64, to)),
        _ => None,
    }
}

/// Integer promotion of an argument to a C extern defined by MIR.
fn c_extern_arg_coercion(from: Mt, param: &IrType) -> Option<Conv> {
    match from {
        Mt::I32 | Mt::I8 => match param {
            IrType::I32 | IrType::I64 | IrType::NullInt | IrType::Bool => {
                Some(Conv::Sext(from, Mt::I64))
            }
            IrType::U32 | IrType::U64 => Some(Conv::Uext(from, Mt::I64)),
            IrType::F64 => Some(Conv::SToF(from, Mt::F64)),
            IrType::F32 => Some(Conv::SToF(from, Mt::F32)),
            _ => None,
        },
        Mt::I64 => match param {
            IrType::F64 => Some(Conv::SToF(Mt::I64, Mt::F64)),
            IrType::F32 => Some(Conv::SToF(Mt::I64, Mt::F32)),
            _ => None,
        },
        _ => None,
    }
}

/// Addresses the decoder resolves once per process.
pub(crate) struct Externs<'a> {
    pub program: &'a Program,
}

impl Externs<'_> {
    fn symbol(&self, name: &str) -> Option<usize> {
        match name {
            "malloc" => return Some(libc::malloc as *const () as usize),
            "realloc" => return Some(libc::realloc as *const () as usize),
            "free" => return Some(libc::free as *const () as usize),
            _ => {}
        }
        self.program.symbol(name)
    }
}

struct Decoder<'a> {
    module: &'a IrModule,
    module_addr: usize,
    function: &'a IrFunction,
    externs: Externs<'a>,
    indirect: &'a BTreeSet<IrFunctionId>,
    mts: Vec<Option<Mt>>,
    next_reg: u32,
    consts: HashMap<Reg, i64>,
    /// Registers an `fmul` produced, with its operands and block.
    fmuls: HashMap<Reg, (Reg, Reg, u32)>,
    slot_bytes: usize,
    live: BTreeMap<IrId, u32>,
    block_index: BTreeMap<IrBlockId, u32>,
    current_block: u32,
    env_param: bool,
}

/// Decode `func_id` of `module`.
pub(crate) fn decode(
    program: &Program,
    module: &IrModule,
    func_id: IrFunctionId,
    indirect: &BTreeSet<IrFunctionId>,
) -> Result<Function, String> {
    let function = module
        .functions
        .get(&func_id)
        .ok_or_else(|| format!("function {:?} not found", func_id))?;
    if function.cfg.blocks.is_empty() {
        return Err(format!("`{}` has no body", function.name));
    }

    let mut max_reg = function.next_reg_id;
    let mut bump = |id: IrId| {
        if id.is_valid() {
            max_reg = max_reg.max(id.as_u32() + 1);
        }
    };
    for p in &function.signature.parameters {
        bump(p.reg);
    }
    // Every register named anywhere, defined or not, gets a slot.
    for block in function.cfg.blocks.values() {
        for phi in &block.phi_nodes {
            bump(phi.dest);
            for (_, v) in &phi.incoming {
                bump(*v);
            }
        }
        for inst in &block.instructions {
            if let Some(d) = inst.dest() {
                bump(d);
            }
            for u in inst.uses() {
                bump(u);
            }
        }
        match &block.terminator {
            IrTerminator::CondBranch { condition: r, .. }
            | IrTerminator::Switch { value: r, .. }
            | IrTerminator::Return { value: Some(r) } => bump(*r),
            _ => {}
        }
    }

    let hidden_env = needs_env(function, func_id, indirect);
    let explicit_env = function
        .signature
        .parameters
        .first()
        .is_some_and(|p| p.name == "env");
    let abi = abi_of(function, hidden_env);

    let mut d = Decoder {
        module,
        module_addr: module as *const IrModule as usize,
        function,
        externs: Externs { program },
        indirect,
        mts: vec![None; max_reg as usize],
        next_reg: max_reg,
        consts: HashMap::new(),
        fmuls: HashMap::new(),
        slot_bytes: 0,
        live: BTreeMap::new(),
        block_index: BTreeMap::new(),
        current_block: 0,
        env_param: hidden_env || explicit_env,
    };

    let mut params = Vec::new();
    for p in &function.signature.parameters {
        let mt = Mt::of(&p.ty).unwrap_or(Mt::I64);
        d.set_mt(p.reg.as_u32(), mt);
        params.push((p.reg.as_u32(), mt));
    }
    for block in function.cfg.blocks.values() {
        for phi in &block.phi_nodes {
            d.set_mt(phi.dest.as_u32(), Mt::phi(&phi.ty));
        }
        for inst in &block.instructions {
            if let IrInstruction::MarkMoved { src } | IrInstruction::CheckLive { src, .. } = inst {
                let n = d.live.len() as u32;
                d.live.entry(*src).or_insert(n);
            }
        }
    }

    let order = reverse_post_order(function);
    for (i, id) in order.iter().enumerate() {
        d.block_index.insert(*id, i as u32);
    }
    let mut blocks = Vec::with_capacity(order.len());
    for (i, id) in order.iter().enumerate() {
        d.current_block = i as u32;
        let block = &function.cfg.blocks[id];
        let mut ops = Vec::with_capacity(block.instructions.len());
        for inst in &block.instructions {
            d.instruction(inst, &mut ops);
        }
        let term = d.terminator(*id, &block.terminator, &mut ops);
        blocks.push(Block { ops, term });
    }

    Ok(Function {
        id: func_id,
        name: Arc::from(function.name.as_str()),
        abi,
        return_type: function.signature.return_type.clone(),
        params,
        explicit_env,
        registers: d.next_reg as usize,
        slot_words: d.slot_bytes.div_ceil(8),
        live_slots: d.live.len(),
        entry: d.block_index[&function.cfg.entry_block],
        blocks,
    })
}

/// Blocks in the order the lowering visits them: reverse post-order from the
/// entry, then any it cannot reach.
fn reverse_post_order(function: &IrFunction) -> Vec<IrBlockId> {
    let mut rpo = Vec::new();
    let mut visited = BTreeSet::new();
    let mut stack = vec![(function.cfg.entry_block, false)];
    while let Some((id, done)) = stack.pop() {
        if done {
            if function.cfg.blocks.contains_key(&id) {
                rpo.push(id);
            }
            continue;
        }
        if !visited.insert(id) {
            continue;
        }
        stack.push((id, true));
        let successors: Vec<IrBlockId> = match function.cfg.blocks.get(&id).map(|b| &b.terminator) {
            Some(IrTerminator::Branch { target }) => vec![*target],
            Some(IrTerminator::CondBranch {
                true_target,
                false_target,
                ..
            }) => vec![*true_target, *false_target],
            Some(IrTerminator::Switch { cases, default, .. }) => {
                let mut t: Vec<_> = cases.iter().map(|(_, t)| *t).collect();
                t.push(*default);
                t
            }
            _ => vec![],
        };
        for s in successors.into_iter().rev() {
            if !visited.contains(&s) {
                stack.push((s, false));
            }
        }
    }
    rpo.reverse();
    for id in function.cfg.blocks.keys() {
        if !visited.contains(id) {
            rpo.push(*id);
        }
    }
    rpo
}

fn trap(message: impl Into<String>) -> Op {
    Op::Trap {
        message: Arc::from(message.into()),
    }
}

impl Decoder<'_> {
    fn set_mt(&mut self, reg: Reg, mt: Mt) {
        let i = reg as usize;
        if i >= self.mts.len() {
            self.mts.resize(i + 1, None);
        }
        self.mts[i] = Some(mt);
    }

    fn mt(&self, reg: IrId) -> Mt {
        self.mt_of(reg.as_u32())
    }

    fn mt_of(&self, reg: Reg) -> Mt {
        self.mts
            .get(reg as usize)
            .copied()
            .flatten()
            .unwrap_or(Mt::I64)
    }

    fn temp(&mut self, mt: Mt) -> Reg {
        let r = self.next_reg;
        self.next_reg += 1;
        self.set_mt(r, mt);
        r
    }

    /// `src` converted by `conv`, through a new register when there is one.
    fn coerce(&mut self, ops: &mut Vec<Op>, src: Reg, conv: Option<Conv>) -> Reg {
        match conv {
            None => src,
            Some(conv) => {
                let dest = self.temp(conv.to());
                ops.push(Op::Conv { dest, src, conv });
                dest
            }
        }
    }

    fn slot(&mut self, size: usize) -> u32 {
        let offset = self.slot_bytes.next_multiple_of(8);
        self.slot_bytes = offset + size.max(1);
        offset as u32
    }

    fn reg_type(&self, reg: IrId) -> Option<&IrType> {
        self.function.register_types.get(&reg)
    }

    fn local_type(&self, reg: IrId) -> Option<&IrType> {
        self.function.locals.get(&reg).map(|l| &l.ty)
    }

    /// The first phi, in block order, defining any of `regs`.
    fn phi_type(&self, regs: &[IrId]) -> Option<&IrType> {
        self.function
            .cfg
            .blocks
            .values()
            .flat_map(|b| b.phi_nodes.iter())
            .find(|phi| regs.contains(&phi.dest))
            .map(|phi| &phi.ty)
    }

    fn fn_ref(&self, func: IrFunctionId) -> Result<FnRef, String> {
        if let Some(f) = self.module.functions.get(&func) {
            if f.cfg.blocks.is_empty() {
                return self
                    .externs
                    .symbol(&f.name)
                    .map(FnRef::Address)
                    .ok_or_else(|| format!("unresolved extern `{}`", f.name));
            }
            return Ok(FnRef::Function {
                module: self.module_addr,
                func,
            });
        }
        if let Some(e) = self.module.extern_functions.get(&func) {
            return self
                .externs
                .symbol(&e.name)
                .map(FnRef::Address)
                .ok_or_else(|| format!("unresolved extern `{}`", e.name));
        }
        Err(format!("function {:?} not found", func))
    }

    fn instruction(&mut self, inst: &IrInstruction, ops: &mut Vec<Op>) {
        if let Err(message) = self.try_instruction(inst, ops) {
            if let Some(dest) = inst.dest() {
                if self
                    .mts
                    .get(dest.as_u32() as usize)
                    .copied()
                    .flatten()
                    .is_none()
                {
                    self.set_mt(dest.as_u32(), Mt::I64);
                }
            }
            ops.push(trap(message));
        }
    }

    fn try_instruction(&mut self, inst: &IrInstruction, ops: &mut Vec<Op>) -> Result<(), String> {
        use IrInstruction as I;
        match inst {
            I::Const { dest, value } => self.constant(dest.as_u32(), value, ops)?,

            I::Copy { dest, src }
            | I::Move { dest, src }
            | I::SsaBarrier { dest, src, .. }
            | I::BorrowImmutable { dest, src, .. }
            | I::BorrowMutable { dest, src, .. }
            | I::ClosureFunc { dest, closure: src } => {
                let mt = self.mt(*src);
                self.set_mt(dest.as_u32(), mt);
                if let Some(&c) = self.consts.get(&src.as_u32()) {
                    self.consts.insert(dest.as_u32(), c);
                }
                ops.push(Op::Copy {
                    dest: dest.as_u32(),
                    src: src.as_u32(),
                });
            }

            I::Clone { dest, src } => {
                self.set_mt(dest.as_u32(), Mt::I64);
                ops.push(Op::CloneObject {
                    dest: dest.as_u32(),
                    src: src.as_u32(),
                });
            }

            I::EndBorrow { .. } | I::DebugLoc { .. } | I::Phi { .. } => {}
            I::Jump { .. } | I::Branch { .. } | I::Switch { .. } | I::Return { .. } => {}

            I::MarkMoved { src } => ops.push(Op::MarkMoved {
                slot: self.live[src],
            }),
            I::CheckLive { src, location } => {
                let name = self
                    .function
                    .locals
                    .get(src)
                    .map(|l| l.name.clone())
                    .filter(|n| !n.is_empty())
                    .or_else(|| {
                        self.function
                            .signature
                            .parameters
                            .iter()
                            .find(|p| p.reg == *src)
                            .map(|p| p.name.clone())
                            .filter(|n| !n.is_empty())
                    })
                    .unwrap_or_else(|| "<anon>".to_string());
                ops.push(Op::CheckLive {
                    slot: self.live[src],
                    name: Arc::from(name),
                    line: location.line,
                });
            }

            I::BinOp {
                dest,
                op,
                left,
                right,
            } => self.binop(*dest, *op, *left, *right, ops)?,
            I::UnOp { dest, op, operand } => self.unop(*dest, *op, *operand, ops)?,
            I::Cmp {
                dest,
                op,
                left,
                right,
            } => self.cmp(*dest, *op, *left, *right, ops)?,

            I::Load { dest, ptr, ty } => {
                let mt = Mt::of(ty).ok_or("load of a void value")?;
                self.set_mt(dest.as_u32(), mt);
                ops.push(Op::Load {
                    dest: dest.as_u32(),
                    mt,
                    ptr: ptr.as_u32(),
                    offset: 0,
                });
            }

            I::Store {
                ptr,
                value,
                store_ty,
            } => {
                let val_mt = self.mt(*value);
                // Object fields are 8-byte slots; a narrow integer fills one.
                let field = matches!(
                    self.reg_type(*ptr),
                    Some(IrType::Ptr(inner)) if !matches!(inner.as_ref(), IrType::U8 | IrType::I8)
                );
                let float_field =
                    matches!(store_ty, Some(IrType::F64 | IrType::F32)) || val_mt.is_float();
                let mut val = value.as_u32();
                let mut mt = val_mt;
                if field && val_mt.is_int() && val_mt.bits() < 64 && !float_field {
                    val = self.coerce(ops, val, Some(Conv::Sext(val_mt, Mt::I64)));
                    mt = Mt::I64;
                }
                ops.push(Op::Store {
                    mt,
                    ptr: ptr.as_u32(),
                    val,
                    offset: 0,
                });
            }

            I::Alloc { dest, ty, count } => {
                let unit = alloca_size(ty);
                let size = match count {
                    Some(c) => unit * self.consts.get(&c.as_u32()).copied().unwrap_or(1) as usize,
                    None if matches!(ty, IrType::Any | IrType::Ptr(_) | IrType::Ref(_)) => {
                        unit * 16
                    }
                    None => unit,
                };
                let offset = self.slot(size);
                self.set_mt(dest.as_u32(), Mt::I64);
                ops.push(Op::SlotAddr {
                    dest: dest.as_u32(),
                    offset,
                });
            }

            I::Free { ptr } => ops.push(Op::Free { ptr: ptr.as_u32() }),

            I::GetElementPtr {
                dest,
                ptr,
                indices,
                ty,
                ..
            } => {
                if indices.len() != 1 {
                    return Err(format!(
                        "GEP with {} indices not yet supported",
                        indices.len()
                    ));
                }
                let scale = match ty {
                    IrType::Ptr(inner) if matches!(inner.as_ref(), IrType::I8) => 1,
                    _ => 8,
                };
                self.set_mt(dest.as_u32(), Mt::I64);
                ops.push(Op::AddrIndex {
                    dest: dest.as_u32(),
                    base: ptr.as_u32(),
                    index: indices[0].as_u32(),
                    index_mt: self.mt(indices[0]),
                    scale,
                });
            }

            I::PtrAdd {
                dest,
                ptr,
                offset,
                ty,
            } => {
                let IrType::Ptr(pointee) = ty else {
                    return Err(format!("PtrAdd on non-pointer type: {:?}", ty));
                };
                self.set_mt(dest.as_u32(), Mt::I64);
                ops.push(Op::AddrIndex {
                    dest: dest.as_u32(),
                    base: ptr.as_u32(),
                    index: offset.as_u32(),
                    index_mt: self.mt(*offset),
                    scale: field_size(pointee) as i64,
                });
            }

            I::ArrayHeader {
                dest,
                array,
                field,
                guarded,
            } => {
                self.set_mt(dest.as_u32(), Mt::I64);
                ops.push(Op::ArrayHeader {
                    dest: dest.as_u32(),
                    array: array.as_u32(),
                    offset: field.byte_offset() as i32,
                    guarded: *guarded,
                });
            }

            I::MemCopy { dest, src, size } => ops.push(Op::MemCopy {
                dest: dest.as_u32(),
                src: src.as_u32(),
                size: size.as_u32(),
                size_mt: self.mt(*size),
            }),
            I::MemSet { dest, value, size } => ops.push(Op::MemSet {
                dest: dest.as_u32(),
                val: value.as_u32(),
                size: size.as_u32(),
                size_mt: self.mt(*size),
            }),

            I::CallDirect {
                dest,
                func_id,
                args,
                ..
            } => {
                self.call_direct(dest.map(|d| d.as_u32()), *func_id, args, ops)?;
                let name = self
                    .module
                    .extern_functions
                    .get(func_id)
                    .map(|e| e.name.as_str())
                    .or_else(|| self.module.functions.get(func_id).map(|f| f.name.as_str()));
                if name == Some("rayzor_exception_pop_handler") {
                    ops.push(Op::HandlerPopped);
                }
            }

            I::CallIndirect {
                dest,
                func_ptr,
                args,
                signature,
                ..
            } => {
                let IrType::Function {
                    params,
                    return_type,
                    ..
                } = signature
                else {
                    return Err(format!(
                        "Invalid signature type for CallIndirect: {:?}",
                        signature
                    ));
                };
                if matches!(return_type.as_ref(), IrType::Struct { .. }) {
                    return Err("closure call returning a struct".to_string());
                }
                let expected: Vec<Mt> = params.iter().filter_map(Mt::of).collect();
                let mut coerced = Vec::with_capacity(args.len());
                for (i, a) in args.iter().enumerate() {
                    let from = self.mt(*a);
                    let to = expected.get(i).copied().unwrap_or(from);
                    let r = self.coerce(ops, a.as_u32(), indirect_arg_coercion(from, to));
                    coerced.push((r, self.mt_of(r)));
                }
                let ret = Mt::of(return_type);
                let dest = dest.map(|d| d.as_u32());
                if let Some(d) = dest {
                    match ret {
                        Some(mt) => self.set_mt(d, mt),
                        None => {
                            let (mt, _) = self.zero_for(d);
                            self.set_mt(d, mt);
                        }
                    }
                }
                if ret.is_none() {
                    if let Some(d) = dest {
                        let (_, bits) = self.zero_for(d);
                        ops.push(Op::CallIndirect(Box::new(IndirectCall {
                            closure: func_ptr.as_u32(),
                            args: coerced,
                            ret: None,
                            dest: None,
                        })));
                        ops.push(Op::Const { dest: d, bits });
                        return Ok(());
                    }
                }
                ops.push(Op::CallIndirect(Box::new(IndirectCall {
                    closure: func_ptr.as_u32(),
                    args: coerced,
                    ret,
                    dest,
                })));
            }

            I::Cast {
                dest,
                src,
                from_ty,
                to_ty,
            } => {
                let from = self.mt(*src);
                let to = Mt::of(to_ty).ok_or("cast to void")?;
                use Mt::*;
                let conv = match (from, to) {
                    (a, b) if a == b => None,
                    (I32 | I64, F64 | F32) => Some(Conv::SToF(from, to)),
                    (F64 | F32, I32 | I64) => Some(Conv::FToS(from, to)),
                    (F32, F64) => Some(Conv::FPromote),
                    (F64, F32) => Some(Conv::FDemote),
                    (I8, F32 | F64) => Some(Conv::SToF(I8, to)),
                    (I32, I64) if *from_ty == IrType::U32 => Some(Conv::Uext(I32, I64)),
                    (I32, I64) => Some(Conv::Sext(I32, I64)),
                    (I64, I32) => Some(Conv::Reduce(I32)),
                    (I8, I32 | I64) => Some(Conv::Uext(I8, to)),
                    (I32 | I64, I8) => Some(Conv::Reduce(I8)),
                    _ => {
                        return Err(format!(
                            "Unsupported cast from {:?} ({:?}) to {:?}",
                            from, from_ty, to_ty
                        ));
                    }
                };
                self.set_mt(dest.as_u32(), to);
                self.move_conv(dest.as_u32(), src.as_u32(), conv, ops);
            }

            I::BitCast { dest, src, ty } => {
                let from = self.mt(*src);
                let to = Mt::of(ty).ok_or("bitcast to void")?;
                use Mt::*;
                let conv = match (from, to) {
                    (F64, I64) | (F32, I32) | (I64, F64) | (I32, F32) => Some(Conv::Bits(to)),
                    (I32, I64) => Some(Conv::Sext(I32, I64)),
                    (I64, I32) => Some(Conv::Reduce(I32)),
                    (a, b) if a.is_int() && b.is_int() && b.bits() > a.bits() => {
                        Some(Conv::Uext(a, b))
                    }
                    (a, b) if a.is_int() && b.is_int() && b.bits() < a.bits() => {
                        Some(Conv::Reduce(b))
                    }
                    (a, b) if a == b => None,
                    _ => return Err(format!("Unsupported bitcast from {:?} to {:?}", from, to)),
                };
                self.set_mt(dest.as_u32(), to);
                self.move_conv(dest.as_u32(), src.as_u32(), conv, ops);
            }

            I::CreateStruct { dest, ty, fields } => {
                let layout = struct_layout(ty)
                    .ok_or_else(|| format!("CreateStruct with non-struct type: {:?}", ty))?;
                let size: usize = layout.iter().map(|f| f.2).sum();
                let offset = self.slot(size);
                let d = dest.as_u32();
                self.set_mt(d, Mt::I64);
                ops.push(Op::SlotAddr { dest: d, offset });
                for (field, (field_offset, _, _)) in fields.iter().zip(&layout) {
                    ops.push(Op::Store {
                        mt: self.mt(*field),
                        ptr: d,
                        val: field.as_u32(),
                        offset: *field_offset,
                    });
                }
            }

            I::CreateUnion {
                dest,
                discriminant,
                value,
                ..
            } => {
                let offset = self.slot(16);
                let d = dest.as_u32();
                self.set_mt(d, Mt::I64);
                let tag = self.temp(Mt::I32);
                ops.push(Op::SlotAddr { dest: d, offset });
                ops.push(Op::Const {
                    dest: tag,
                    bits: *discriminant as u64,
                });
                ops.push(Op::Store {
                    mt: Mt::I32,
                    ptr: d,
                    val: tag,
                    offset: 0,
                });
                ops.push(Op::Store {
                    mt: self.mt(*value),
                    ptr: d,
                    val: value.as_u32(),
                    offset: 8,
                });
            }

            I::ExtractDiscriminant { dest, union_val } => {
                self.set_mt(dest.as_u32(), Mt::I32);
                ops.push(Op::Load {
                    dest: dest.as_u32(),
                    mt: Mt::I32,
                    ptr: union_val.as_u32(),
                    offset: 0,
                });
            }

            I::ExtractUnionValue {
                dest,
                union_val,
                value_ty,
                ..
            } => {
                let mt = Mt::of(value_ty).unwrap_or(Mt::I64);
                self.set_mt(dest.as_u32(), mt);
                ops.push(Op::Load {
                    dest: dest.as_u32(),
                    mt,
                    ptr: union_val.as_u32(),
                    offset: 8,
                });
            }

            I::ExtractValue {
                dest,
                aggregate,
                indices,
            } => {
                if indices.len() != 1 {
                    return Err(format!(
                        "ExtractValue with multiple indices not yet supported: {:?}",
                        indices
                    ));
                }
                let ty = self
                    .aggregate_type(*aggregate)
                    .ok_or_else(|| format!("Type not found for aggregate {:?}", aggregate))?;
                let layout = struct_layout(&ty)
                    .ok_or_else(|| format!("ExtractValue on non-struct type: {:?}", ty))?;
                let (offset, mt, _) = *layout
                    .get(indices[0] as usize)
                    .ok_or_else(|| format!("Field index {} out of bounds", indices[0]))?;
                self.set_mt(dest.as_u32(), mt);
                ops.push(Op::Load {
                    dest: dest.as_u32(),
                    mt,
                    ptr: aggregate.as_u32(),
                    offset,
                });
            }

            I::Select {
                dest,
                condition,
                true_val,
                false_val,
            } => {
                let mt = self.mt(*true_val);
                self.set_mt(dest.as_u32(), mt);
                ops.push(Op::Select {
                    dest: dest.as_u32(),
                    cond: condition.as_u32(),
                    a: true_val.as_u32(),
                    b: false_val.as_u32(),
                });
            }

            I::FunctionRef { dest, func_id } => {
                let func = self.fn_ref(*func_id)?;
                self.set_mt(dest.as_u32(), Mt::I64);
                ops.push(Op::Closure {
                    dest: dest.as_u32(),
                    func,
                    captures: Vec::new(),
                });
            }

            I::MakeClosure {
                dest,
                func_id,
                captured_values,
            } => {
                let func = self.fn_ref(*func_id)?;
                let mut captures = Vec::with_capacity(captured_values.len());
                for v in captured_values {
                    let mt = self.mt(*v);
                    // Environment slots are 8 bytes; narrow integers widen.
                    let r = if matches!(mt, Mt::I32 | Mt::I8) {
                        self.coerce(ops, v.as_u32(), Some(Conv::Sext(mt, Mt::I64)))
                    } else {
                        v.as_u32()
                    };
                    captures.push((r, self.mt_of(r)));
                }
                self.set_mt(dest.as_u32(), Mt::I64);
                ops.push(Op::Closure {
                    dest: dest.as_u32(),
                    func,
                    captures,
                });
            }

            I::ClosureEnv { dest, .. } => {
                self.set_mt(dest.as_u32(), Mt::I64);
                if self.env_param {
                    ops.push(Op::Env {
                        dest: dest.as_u32(),
                    });
                } else {
                    ops.push(Op::Const {
                        dest: dest.as_u32(),
                        bits: 0,
                    });
                }
            }

            I::Undef { dest, ty } => {
                let (mt, bits) = match ty {
                    IrType::NullInt => (Mt::I64, crate::ir::NULL_INT as u64),
                    _ => match Mt::of(ty) {
                        Some(Mt::V128) | None => (Mt::I64, 0),
                        Some(mt) => (mt, 0),
                    },
                };
                self.set_mt(dest.as_u32(), mt);
                ops.push(Op::Const {
                    dest: dest.as_u32(),
                    bits,
                });
            }

            I::LoadGlobal {
                dest,
                global_id,
                ty,
            } => {
                let mt = match ty {
                    IrType::I32 => Mt::I32,
                    IrType::F32 => Mt::F32,
                    IrType::F64 => Mt::F64,
                    IrType::Bool => Mt::I8,
                    _ => Mt::I64,
                };
                let slot = self.global_slot(global_id.0)?;
                self.set_mt(dest.as_u32(), mt);
                ops.push(Op::GlobalLoad {
                    dest: dest.as_u32(),
                    mt,
                    slot,
                });
            }

            I::StoreGlobal { global_id, value } => {
                let slot = self.global_slot(global_id.0)?;
                ops.push(Op::GlobalStore {
                    src: value.as_u32(),
                    mt: self.mt(*value),
                    slot,
                });
            }

            I::AtomicLoad { dest, ptr, ty } => {
                let mt = Mt::of(ty).ok_or("atomic load of void")?;
                self.set_mt(dest.as_u32(), mt);
                ops.push(Op::AtomicLoad {
                    dest: dest.as_u32(),
                    mt,
                    ptr: ptr.as_u32(),
                });
            }
            I::AtomicStore { ptr, value, .. } => ops.push(Op::AtomicStore {
                mt: self.mt(*value),
                ptr: ptr.as_u32(),
                val: value.as_u32(),
            }),
            I::AtomicRmw {
                dest,
                op,
                ptr,
                value,
                ty,
            } => {
                let mt = Mt::of(ty).ok_or("atomic rmw on void")?;
                self.set_mt(dest.as_u32(), mt);
                ops.push(Op::AtomicRmw {
                    dest: dest.as_u32(),
                    mt,
                    op: *op,
                    ptr: ptr.as_u32(),
                    val: value.as_u32(),
                });
            }
            I::AtomicCas {
                dest,
                ptr,
                expected,
                replacement,
                ..
            } => {
                let mt = self.mt(*expected);
                self.set_mt(dest.as_u32(), mt);
                ops.push(Op::AtomicCas {
                    dest: dest.as_u32(),
                    mt,
                    ptr: ptr.as_u32(),
                    expected: expected.as_u32(),
                    replacement: replacement.as_u32(),
                });
            }

            I::VectorLoad { dest, .. }
            | I::VectorBinOp { dest, .. }
            | I::VectorSplat { dest, .. }
            | I::VectorInsert { dest, .. }
            | I::VectorUnaryOp { dest, .. }
            | I::VectorMinMax { dest, .. }
            | I::VectorDot { dest, .. }
            | I::VectorShuffle { dest, .. }
            | I::VectorConvert { dest, .. }
            | I::VectorNarrow { dest, .. } => {
                self.set_mt(dest.as_u32(), Mt::V128);
                ops.push(Op::Bail);
            }
            I::VectorExtract { dest, elem_ty, .. } | I::VectorReduce { dest, elem_ty, .. } => {
                self.set_mt(dest.as_u32(), Mt::of(elem_ty).unwrap_or(Mt::I64));
                ops.push(Op::Bail);
            }
            I::VectorStore { .. } => ops.push(Op::Bail),

            I::Panic { message } => {
                return Err(message.clone().unwrap_or_else(|| "panic".to_string()));
            }

            I::InsertValue { .. }
            | I::Throw { .. }
            | I::LandingPad { .. }
            | I::Resume { .. }
            | I::InlineAsm { .. } => {
                return Err(format!("Unsupported instruction: {:?}", inst));
            }
        }
        Ok(())
    }

    fn move_conv(&mut self, dest: Reg, src: Reg, conv: Option<Conv>, ops: &mut Vec<Op>) {
        match conv {
            None => ops.push(Op::Copy { dest, src }),
            Some(conv) => ops.push(Op::Conv { dest, src, conv }),
        }
    }

    fn constant(&mut self, dest: Reg, value: &IrValue, ops: &mut Vec<Op>) -> Result<(), String> {
        let (mt, bits) = match value {
            IrValue::I8(v) => (Mt::I8, *v as u8 as u64),
            IrValue::U8(v) => (Mt::I8, *v as u64),
            IrValue::Bool(v) => (Mt::I8, *v as u64),
            IrValue::I16(v) => (Mt::I16, *v as u16 as u64),
            IrValue::U16(v) => (Mt::I16, *v as u64),
            IrValue::I32(v) => (Mt::I32, *v as u32 as u64),
            IrValue::U32(v) => (Mt::I32, *v as u64),
            IrValue::I64(v) => (Mt::I64, *v as u64),
            IrValue::U64(v) => (Mt::I64, *v),
            IrValue::F32(v) => (Mt::F32, v.to_bits() as u64),
            IrValue::F64(v) => (Mt::F64, v.to_bits()),
            IrValue::Null => (Mt::I64, 0),
            IrValue::String(s) => {
                let literal = self
                    .externs
                    .symbol("haxe_string_literal")
                    .ok_or("unresolved extern `haxe_string_literal`")?;
                let bytes = self.externs.program.literal(s);
                self.set_mt(dest, Mt::I64);
                ops.push(Op::Str {
                    dest,
                    literal,
                    ptr: bytes.as_ptr() as usize,
                    len: bytes.len(),
                });
                return Ok(());
            }
            IrValue::Function(func) => {
                self.set_mt(dest, Mt::I64);
                match self.fn_ref(*func)? {
                    FnRef::Address(addr) => ops.push(Op::Const {
                        dest,
                        bits: addr as u64,
                    }),
                    FnRef::Function { module, func } => ops.push(Op::FnAddr { dest, module, func }),
                }
                return Ok(());
            }
            other => return Err(format!("Unsupported constant value: {:?}", other)),
        };
        if mt.is_int() {
            self.consts.insert(dest, sext(bits, mt));
        }
        self.set_mt(dest, mt);
        ops.push(Op::Const { dest, bits });
        Ok(())
    }

    /// The type an `ExtractValue` aggregate was declared or loaded with.
    fn aggregate_type(&self, aggregate: IrId) -> Option<IrType> {
        let f = self.function;
        if let Some(p) = f.signature.parameters.iter().find(|p| p.reg == aggregate) {
            return Some(p.ty.clone());
        }
        if let Some(l) = f.locals.get(&aggregate) {
            return Some(l.ty.clone());
        }
        for block in f.cfg.blocks.values() {
            for inst in &block.instructions {
                if let IrInstruction::Load { dest, ty, .. } = inst {
                    if *dest == aggregate {
                        return Some(ty.clone());
                    }
                }
            }
        }
        None
    }

    fn global_slot(&self, id: u32) -> Result<GlobalSlot, String> {
        if let Some(slot_fn) = self.externs.symbol("rayzor_global_slot") {
            let slot_fn: extern "C" fn(i64) -> *mut u64 = unsafe { std::mem::transmute(slot_fn) };
            return Ok(GlobalSlot::Address(slot_fn(i64::from(id)) as usize));
        }
        match (
            self.externs.symbol("rayzor_global_load"),
            self.externs.symbol("rayzor_global_store"),
        ) {
            (Some(load), Some(store)) => Ok(GlobalSlot::Runtime { id, load, store }),
            _ => Err("no runtime global store".to_string()),
        }
    }

    /// A register named for a callee that returns nothing reads as zero.
    fn zero_for(&self, reg: Reg) -> (Mt, u64) {
        let ty = self.function.register_types.get(&IrId::new(reg));
        if ty == Some(&IrType::NullInt) {
            return (Mt::I64, crate::ir::NULL_INT as u64);
        }
        match ty.and_then(Mt::of) {
            Some(Mt::V128) | None => (Mt::I64, 0),
            Some(mt) => (mt, 0),
        }
    }

    fn binop(
        &mut self,
        dest: IrId,
        op: BinaryOp,
        left: IrId,
        right: IrId,
        ops: &mut Vec<Op>,
    ) -> Result<(), String> {
        let ty = self
            .reg_type(dest)
            .or_else(|| self.reg_type(left))
            .or_else(|| self.reg_type(right))
            .or_else(|| self.local_type(dest))
            .or_else(|| self.phi_type(&[left, right]))
            .cloned()
            .ok_or_else(|| format!("Type not found for BinOp dest {:?}", dest))?;
        let (l_mt, r_mt) = (self.mt(left), self.mt(right));
        if l_mt == Mt::V128 || r_mt == Mt::V128 {
            self.set_mt(dest.as_u32(), Mt::V128);
            ops.push(Op::Bail);
            return Ok(());
        }
        let expected = match ty {
            IrType::I32 | IrType::U32 | IrType::Bool => Mt::I32,
            IrType::F32 => Mt::F32,
            IrType::F64 => Mt::F64,
            _ => Mt::I64,
        };
        let both_int = l_mt.is_int() && r_mt.is_int();
        let larger = if l_mt.bits() >= r_mt.bits() {
            l_mt
        } else {
            r_mt
        };
        let integer_only = matches!(
            op,
            BinaryOp::And
                | BinaryOp::Or
                | BinaryOp::Xor
                | BinaryOp::Shl
                | BinaryOp::Shr
                | BinaryOp::Ushr
        );
        let op_ty = if integer_only {
            let operand = if both_int {
                larger
            } else if l_mt.is_int() {
                l_mt
            } else if r_mt.is_int() {
                r_mt
            } else if expected.is_int() {
                expected
            } else {
                Mt::I32
            };
            if expected.is_int() && expected.bits() > operand.bits() {
                expected
            } else {
                operand
            }
        } else if both_int && !expected.is_int() {
            larger
        } else if larger.is_int() && expected.is_int() && larger.bits() > expected.bits() {
            larger
        } else {
            expected
        };
        let float_named = matches!(
            op,
            BinaryOp::FAdd | BinaryOp::FSub | BinaryOp::FMul | BinaryOp::FDiv | BinaryOp::FRem
        );
        let use_float = !integer_only
            && (l_mt.is_float() || r_mt.is_float() || op_ty.is_float() || float_named);
        let signed = is_signed(&ty);
        let signed_coercion = matches!(op, BinaryOp::Shr) || signed || !is_unsigned(&ty);

        let operand_conv = |v: Mt, other: Mt| -> Option<Conv> {
            if integer_only {
                if v == op_ty {
                    None
                } else if v.is_float() {
                    Some(if signed_coercion {
                        Conv::FToS(v, op_ty)
                    } else {
                        Conv::FToU(v, op_ty)
                    })
                } else {
                    resize(v, op_ty, signed_coercion)
                }
            } else if use_float && v.is_int() {
                let float_ty = if other.is_float() { other } else { Mt::F64 };
                Some(if signed || v == Mt::I32 || v == Mt::I64 {
                    Conv::SToF(v, float_ty)
                } else {
                    Conv::UToF(v, float_ty)
                })
            } else if v != op_ty && v.is_int() && op_ty.is_int() {
                Some(if signed {
                    Conv::Sext(v, op_ty)
                } else {
                    Conv::Uext(v, op_ty)
                })
            } else {
                None
            }
        };
        let (l_conv, r_conv) = (operand_conv(l_mt, r_mt), operand_conv(r_mt, l_mt));
        let a = self.coerce(ops, left.as_u32(), l_conv);
        let b = self.coerce(ops, right.as_u32(), r_conv);
        let (a_mt, b_mt) = (self.mt_of(a), self.mt_of(b));

        let block = self.current_block;
        let result: Reg;
        let result_mt: Mt;
        if use_float && a_mt.is_float() && b_mt.is_float() {
            // Mixed widths would not verify; compute at the wider one.
            let mt = if a_mt == Mt::F64 || b_mt == Mt::F64 {
                Mt::F64
            } else {
                Mt::F32
            };
            let a = if a_mt == mt {
                a
            } else {
                self.coerce(ops, a, Some(Conv::FPromote))
            };
            let b = if b_mt == mt {
                b
            } else {
                self.coerce(ops, b, Some(Conv::FPromote))
            };
            let fop = match op {
                BinaryOp::Add | BinaryOp::FAdd => FloatOp::Add,
                BinaryOp::Sub | BinaryOp::FSub => FloatOp::Sub,
                BinaryOp::Mul | BinaryOp::FMul => FloatOp::Mul,
                BinaryOp::Div | BinaryOp::FDiv => FloatOp::Div,
                _ => FloatOp::Rem,
            };
            result_mt = mt;
            result = self.temp(mt);
            let fused = if crate::codegen::instruction_lowering::scalar_fma_enabled() {
                match fop {
                    FloatOp::Add | FloatOp::Sub => {
                        let from_l = self.fmuls.get(&a).filter(|m| m.2 == block).copied();
                        let from_r = self.fmuls.get(&b).filter(|m| m.2 == block).copied();
                        match (fop, from_l, from_r) {
                            (FloatOp::Add, Some((x, y, _)), _) => Some((x, y, b, false, false)),
                            (FloatOp::Add, None, Some((x, y, _))) => Some((x, y, a, false, false)),
                            (FloatOp::Sub, Some((x, y, _)), _) => Some((x, y, b, false, true)),
                            (FloatOp::Sub, None, Some((x, y, _))) => Some((x, y, a, true, false)),
                            _ => None,
                        }
                    }
                    _ => None,
                }
            } else {
                None
            };
            match fused {
                Some((x, y, c, neg_a, neg_c)) => ops.push(Op::Fma {
                    dest: result,
                    mt,
                    a: x,
                    b: y,
                    c,
                    neg_a,
                    neg_c,
                }),
                None => {
                    ops.push(Op::FBin {
                        dest: result,
                        op: fop,
                        mt,
                        a,
                        b,
                    });
                    if matches!(fop, FloatOp::Mul) {
                        self.fmuls.insert(result, (a, b, block));
                    }
                }
            }
        } else {
            let mt = if a_mt.is_int() { a_mt } else { op_ty };
            let iop = match op {
                BinaryOp::Add | BinaryOp::FAdd => IntOp::Add,
                BinaryOp::Sub | BinaryOp::FSub => IntOp::Sub,
                BinaryOp::Mul | BinaryOp::FMul => IntOp::Mul,
                BinaryOp::Div | BinaryOp::FDiv if signed => IntOp::SDiv,
                BinaryOp::Div | BinaryOp::FDiv => IntOp::UDiv,
                BinaryOp::Rem | BinaryOp::FRem if signed => IntOp::SRem,
                BinaryOp::Rem | BinaryOp::FRem => IntOp::URem,
                BinaryOp::And => IntOp::And,
                BinaryOp::Or => IntOp::Or,
                BinaryOp::Xor => IntOp::Xor,
                BinaryOp::Shl => IntOp::Shl,
                BinaryOp::Shr => IntOp::SShr,
                BinaryOp::Ushr => IntOp::UShr,
            };
            result_mt = mt;
            result = self.temp(mt);
            ops.push(Op::IBin {
                dest: result,
                op: iop,
                mt,
                a,
                b,
            });
        }

        // The result returns to the expected width.
        let final_conv = if result_mt.is_float() && expected.is_int() {
            Some(if signed {
                Conv::FToS(result_mt, expected)
            } else {
                Conv::FToU(result_mt, expected)
            })
        } else if result_mt.is_int() && expected.is_int() && result_mt != expected {
            if result_mt.bits() < expected.bits() {
                Some(if signed {
                    Conv::Sext(result_mt, expected)
                } else {
                    Conv::Uext(result_mt, expected)
                })
            } else {
                Some(Conv::Reduce(expected))
            }
        } else {
            None
        };
        let d = dest.as_u32();
        self.set_mt(d, final_conv.map(|c| c.to()).unwrap_or(result_mt));
        if final_conv.is_none() {
            if let Some(m) = self.fmuls.get(&result).copied() {
                self.fmuls.insert(d, m);
            }
        }
        self.move_conv(d, result, final_conv, ops);
        Ok(())
    }

    fn unop(
        &mut self,
        dest: IrId,
        op: UnaryOp,
        operand: IrId,
        ops: &mut Vec<Op>,
    ) -> Result<(), String> {
        let ty = self
            .reg_type(dest)
            .or_else(|| self.reg_type(operand))
            .or_else(|| self.local_type(dest))
            .or_else(|| self.phi_type(&[operand]))
            .cloned()
            .ok_or_else(|| format!("Type not found for UnOp dest {:?}", dest))?;
        let v_mt = self.mt(operand);
        let mut v = operand.as_u32();
        let ty_float = matches!(ty, IrType::F32 | IrType::F64);
        if matches!(op, UnaryOp::Neg | UnaryOp::FNeg) && ty_float {
            let float_ty = if ty == IrType::F32 { Mt::F32 } else { Mt::F64 };
            let conv = if v_mt.is_int() {
                Some(Conv::SToF(v_mt, float_ty))
            } else if v_mt == Mt::F32 && float_ty == Mt::F64 {
                Some(Conv::FPromote)
            } else if v_mt == Mt::F64 && float_ty == Mt::F32 {
                Some(Conv::FDemote)
            } else {
                None
            };
            v = self.coerce(ops, v, conv);
        }
        let mt = self.mt_of(v);
        let d = dest.as_u32();
        match op {
            UnaryOp::Neg if ty_float => {
                self.set_mt(d, mt);
                ops.push(Op::FNeg { dest: d, mt, a: v });
            }
            UnaryOp::Neg => {
                self.set_mt(d, mt);
                ops.push(Op::INeg { dest: d, mt, a: v });
            }
            UnaryOp::Not if ty == IrType::Bool => {
                self.set_mt(d, Mt::I8);
                ops.push(Op::IsZero { dest: d, a: v });
            }
            UnaryOp::Not => {
                self.set_mt(d, mt);
                ops.push(Op::BNot { dest: d, mt, a: v });
            }
            UnaryOp::FNeg => {
                self.set_mt(d, mt);
                ops.push(Op::FNeg { dest: d, mt, a: v });
            }
        }
        Ok(())
    }

    fn cmp(
        &mut self,
        dest: IrId,
        op: CompareOp,
        left: IrId,
        right: IrId,
        ops: &mut Vec<Op>,
    ) -> Result<(), String> {
        let ty = self
            .reg_type(left)
            .or_else(|| self.local_type(left))
            .or_else(|| self.phi_type(&[left]))
            .cloned()
            .ok_or_else(|| format!("Type not found for Cmp operand {:?}", left))?;
        let (l_mt, r_mt) = (self.mt(left), self.mt(right));
        let d = dest.as_u32();
        self.set_mt(d, Mt::I8);
        let is_float_op = matches!(
            op,
            CompareOp::FEq
                | CompareOp::FNe
                | CompareOp::FLt
                | CompareOp::FLe
                | CompareOp::FGt
                | CompareOp::FGe
                | CompareOp::FOrd
                | CompareOp::FUno
        );
        let (mut a, mut b) = (left.as_u32(), right.as_u32());
        if (l_mt.is_float() && r_mt.is_float())
            || (is_float_op && (l_mt.is_float() || r_mt.is_float()))
        {
            if l_mt.is_float() && r_mt.is_int() {
                b = self.coerce(ops, b, Some(Conv::SToF(r_mt, l_mt)));
            } else if l_mt.is_int() && r_mt.is_float() {
                a = self.coerce(ops, a, Some(Conv::SToF(l_mt, r_mt)));
            }
            let (a_mt, b_mt) = (self.mt_of(a), self.mt_of(b));
            let mt = if a_mt == Mt::F64 || b_mt == Mt::F64 {
                Mt::F64
            } else {
                Mt::F32
            };
            if a_mt != mt {
                a = self.coerce(ops, a, Some(Conv::FPromote));
            }
            if b_mt != mt {
                b = self.coerce(ops, b, Some(Conv::FPromote));
            }
            let cc = match op {
                CompareOp::Eq | CompareOp::FEq => FloatCc::Eq,
                CompareOp::Ne | CompareOp::FNe => FloatCc::Ne,
                CompareOp::Lt | CompareOp::FLt => FloatCc::Lt,
                CompareOp::Le | CompareOp::FLe => FloatCc::Le,
                CompareOp::Gt | CompareOp::FGt => FloatCc::Gt,
                CompareOp::Ge | CompareOp::FGe => FloatCc::Ge,
                CompareOp::FOrd => FloatCc::Ord,
                CompareOp::FUno => FloatCc::Uno,
                _ => return Err(format!("Invalid float comparison: {:?}", op)),
            };
            ops.push(Op::FCmp {
                dest: d,
                cc,
                mt,
                a,
                b,
            });
            return Ok(());
        }
        if l_mt.is_float() {
            a = self.coerce(ops, a, Some(Conv::FToSSat(l_mt, Mt::I64)));
        }
        if r_mt.is_float() {
            b = self.coerce(ops, b, Some(Conv::FToSSat(r_mt, Mt::I64)));
        }
        let (a_mt, b_mt) = (self.mt_of(a), self.mt_of(b));
        let signed = is_signed(&ty);
        if a_mt != b_mt && a_mt.is_int() && b_mt.is_int() {
            if a_mt.bits() > b_mt.bits() {
                b = self.coerce(ops, b, resize(b_mt, a_mt, signed));
            } else {
                a = self.coerce(ops, a, resize(a_mt, b_mt, signed));
            }
        }
        let cc = match op {
            CompareOp::Eq => IntCc::Eq,
            CompareOp::Ne => IntCc::Ne,
            CompareOp::Lt => IntCc::Slt,
            CompareOp::Le => IntCc::Sle,
            CompareOp::Gt => IntCc::Sgt,
            CompareOp::Ge => IntCc::Sge,
            CompareOp::ULt => IntCc::Ult,
            CompareOp::ULe => IntCc::Ule,
            CompareOp::UGt => IntCc::Ugt,
            CompareOp::UGe => IntCc::Uge,
            _ => return Err(format!("Invalid int comparison: {:?}", op)),
        };
        ops.push(Op::ICmp {
            dest: d,
            cc,
            mt: self.mt_of(a),
            a,
            b,
        });
        Ok(())
    }

    /// The lowering inlines these runtime calls; the interpreter computes the
    /// same values. `None` when the call goes to the runtime after all.
    fn intrinsic(
        &mut self,
        name: &str,
        dest: Option<Reg>,
        args: &[(Reg, Mt)],
        ops: &mut Vec<Op>,
    ) -> Option<()> {
        let kind = match name {
            "haxe_math_sqrt" => Some(Intrinsic::Sqrt),
            "haxe_math_abs" => Some(Intrinsic::Abs),
            "haxe_math_floor" => Some(Intrinsic::Floor),
            "haxe_math_ceil" => Some(Intrinsic::Ceil),
            "haxe_math_round" => Some(Intrinsic::Round),
            _ => None,
        };
        if let Some(kind) = kind {
            let &[(arg, mt)] = args else { return None };
            if !(mt.is_float() || mt == Mt::I32 || mt == Mt::I64) {
                return None;
            }
            let arg = if mt.is_float() {
                arg
            } else {
                self.coerce(ops, arg, Some(Conv::SToF(mt, Mt::F64)))
            };
            let mt = self.mt_of(arg);
            let result_mt = match kind {
                Intrinsic::Sqrt | Intrinsic::Abs => mt,
                _ => Mt::I32,
            };
            let dest = dest.unwrap_or_else(|| self.temp(result_mt));
            self.set_mt(dest, result_mt);
            ops.push(Op::Math {
                dest,
                kind,
                arg,
                mt,
            });
            return Some(());
        }
        match name {
            "haxe_std_int" => {
                let &[(arg, mt)] = args else { return None };
                let conv = if mt.is_float() {
                    Some(Conv::FToSSat(mt, Mt::I32))
                } else if mt == Mt::I64 {
                    Some(Conv::Reduce(Mt::I32))
                } else {
                    None
                };
                if let Some(d) = dest {
                    self.set_mt(d, conv.map(|c| c.to()).unwrap_or(mt));
                    self.move_conv(d, arg, conv, ops);
                }
                Some(())
            }
            "haxe_array_length" => {
                let &[(arr, _)] = args else { return None };
                if let Some(d) = dest {
                    self.set_mt(d, Mt::I64);
                    ops.push(Op::Load {
                        dest: d,
                        mt: Mt::I64,
                        ptr: arr,
                        offset: 8,
                    });
                }
                Some(())
            }
            "haxe_array_get_ptr" => {
                let &[(arr, _), (index, index_mt)] = args else {
                    return None;
                };
                if let Some(d) = dest {
                    self.set_mt(d, Mt::I64);
                    ops.push(Op::ArrayGetPtr {
                        dest: d,
                        array: arr,
                        index,
                        index_mt,
                    });
                }
                Some(())
            }
            _ => None,
        }
    }

    fn call_direct(
        &mut self,
        dest: Option<Reg>,
        func_id: IrFunctionId,
        args: &[IrId],
        ops: &mut Vec<Op>,
    ) -> Result<(), String> {
        // A runtime function declared in the extern table.
        if let Some(ext) = self.module.extern_functions.get(&func_id) {
            let sig = &ext.signature;
            let c_conv = sig.calling_convention == CallingConvention::C && !cfg!(windows);
            let declared: Vec<Mt> = sig
                .parameters
                .iter()
                .map(|p| match p.ty {
                    IrType::I32 | IrType::U32 if c_conv => Mt::I64,
                    _ => Mt::of(&p.ty).unwrap_or(Mt::I64),
                })
                .collect();
            let mut coerced = Vec::with_capacity(args.len());
            for (i, a) in args.iter().enumerate() {
                let from = self.mt(*a);
                let conv = if c_conv {
                    let expected = declared.get(i).copied();
                    if matches!(from, Mt::I32 | Mt::I8) && expected == Some(Mt::I64) {
                        Some(Conv::Sext(from, Mt::I64))
                    } else {
                        match sig.parameters.get(i).map(|p| &p.ty) {
                            Some(IrType::F64) if matches!(from, Mt::I32 | Mt::I64) => {
                                Some(Conv::SToF(from, Mt::F64))
                            }
                            Some(IrType::F32) if matches!(from, Mt::I32 | Mt::I64) => {
                                Some(Conv::SToF(from, Mt::F32))
                            }
                            _ => None,
                        }
                    }
                } else {
                    None
                };
                let r = self.coerce(ops, a.as_u32(), conv);
                coerced.push((r, self.mt_of(r)));
            }
            if ext.name == "_setjmp" {
                return self.setjmp(dest, &coerced, ops);
            }
            if self.intrinsic(&ext.name, dest, &coerced, ops).is_some() {
                return Ok(());
            }
            let addr = self
                .externs
                .symbol(&ext.name)
                .ok_or_else(|| format!("unresolved extern `{}`", ext.name))?;
            let ret = Mt::of(&sig.return_type);
            let args = pad(coerced, &declared, &[]);
            return self.emit_call(Target::Native(addr), args, ret, None, dest, ops);
        }

        let called = self
            .module
            .functions
            .get(&func_id)
            .ok_or_else(|| format!("Called function {:?} not found in module", func_id))?;
        let name = called.name.as_str();
        let libc = matches!(name, "malloc" | "realloc" | "free");
        let is_extern = called.cfg.blocks.is_empty();
        let sig = &called.signature;
        let uses_sret = sig.uses_sret && !is_extern;
        let is_lambda = name.starts_with("<lambda");
        let c_conv = sig.calling_convention == CallingConvention::C;
        let add_env = !is_extern
            && !is_lambda
            && !c_conv
            && !libc
            && needs_env(called, func_id, self.indirect);
        let c_extern = c_conv
            && (is_extern || called.attributes.linkage == Linkage::External)
            && !cfg!(windows);

        let mut coerced = Vec::with_capacity(args.len());
        for (i, a) in args.iter().enumerate() {
            let from = self.mt(*a);
            let conv = match sig.parameters.get(i) {
                Some(p) if c_extern => c_extern_arg_coercion(from, &p.ty),
                Some(p) => haxe_arg_coercion(from, Mt::of(&p.ty).unwrap_or(Mt::I64)),
                None => haxe_arg_coercion(from, Mt::I64),
            };
            let r = self.coerce(ops, a.as_u32(), conv);
            coerced.push((r, self.mt_of(r)));
        }

        if is_extern {
            if name == "_setjmp" {
                return self.setjmp(dest, &coerced, ops);
            }
            if self.intrinsic(name, dest, &coerced, ops).is_some() {
                return Ok(());
            }
        }

        let mut hidden: Vec<(Src, Mt)> = Vec::new();
        if uses_sret {
            hidden.push((Src::Imm(0), Mt::I64));
        }
        if add_env {
            hidden.push((Src::Imm(0), Mt::I64));
        }
        // An extern's small integer parameters are widened to i64 under the C ABI.
        let widen = is_extern && c_conv && !cfg!(windows);
        let user_declared: Vec<Mt> = sig
            .parameters
            .iter()
            .map(|p| {
                if widen && matches!(p.ty, IrType::I32 | IrType::U32 | IrType::Bool | IrType::I8) {
                    Mt::I64
                } else {
                    Mt::of(&p.ty).unwrap_or(Mt::I64)
                }
            })
            .collect();
        if coerced.len() > user_declared.len() {
            return Err(format!(
                "call to `{}` passes {} arguments, it takes {}",
                name,
                coerced.len(),
                user_declared.len()
            ));
        }
        let null_int: Vec<bool> = sig
            .parameters
            .iter()
            .map(|p| p.ty == IrType::NullInt)
            .collect();
        let mut call_args = hidden;
        call_args.extend(pad(coerced, &user_declared, &null_int));

        let target = if is_extern || libc {
            Target::Native(
                self.externs
                    .symbol(name)
                    .ok_or_else(|| format!("unresolved extern `{}`", name))?,
            )
        } else {
            Target::Interp {
                module: self.module_addr,
                func: func_id,
            }
        };

        if sig.return_type == IrType::Void {
            self.emit_call(target, call_args, None, None, None, ops)?;
            if let Some(d) = dest {
                let (mt, bits) = self.zero_for(d);
                self.set_mt(d, mt);
                ops.push(Op::Const { dest: d, bits });
            }
            return Ok(());
        }
        if uses_sret {
            let size = struct_size(&sig.return_type).max(8);
            let offset = self.slot(size);
            call_args[0] = (Src::Slot(offset), Mt::I64);
            return self.emit_call(
                target,
                call_args,
                None,
                Some((offset, size as u32)),
                dest,
                ops,
            );
        }
        let ret = Mt::of(&sig.return_type);
        let Some(d) = dest else {
            return self.emit_call(target, call_args, ret, None, None, ops);
        };
        // An integer result resizes to its register when both sides are integers.
        let ret_mt = ret.unwrap_or(Mt::I64);
        let dest_ty = self.function.register_types.get(&IrId::new(d));
        let expected = int_register_mt(dest_ty);
        let conv = if is_primitive_int(&sig.return_type)
            && dest_ty.is_some_and(is_primitive_int)
            && ret_mt != expected
            && ret_mt.is_int()
        {
            if ret_mt.bits() > expected.bits() {
                Some(Conv::Reduce(expected))
            } else {
                Some(Conv::Sext(ret_mt, expected))
            }
        } else {
            None
        };
        match conv {
            None => self.emit_call(target, call_args, ret, None, Some(d), ops),
            Some(conv) => {
                let raw = self.temp(ret_mt);
                self.emit_call(target, call_args, ret, None, Some(raw), ops)?;
                self.set_mt(d, conv.to());
                ops.push(Op::Conv {
                    dest: d,
                    src: raw,
                    conv,
                });
                Ok(())
            }
        }
    }

    fn setjmp(
        &mut self,
        dest: Option<Reg>,
        args: &[(Reg, Mt)],
        ops: &mut Vec<Op>,
    ) -> Result<(), String> {
        let (buf, _) = *args.first().ok_or("_setjmp without a buffer")?;
        let dest = dest.unwrap_or_else(|| self.temp(Mt::I32));
        self.set_mt(dest, Mt::I32);
        ops.push(Op::SetJmp { dest, buf });
        Ok(())
    }

    fn emit_call(
        &mut self,
        target: Target,
        args: Vec<(Src, Mt)>,
        ret: Option<Mt>,
        sret: Option<(u32, u32)>,
        dest: Option<Reg>,
        ops: &mut Vec<Op>,
    ) -> Result<(), String> {
        if let Some(d) = dest {
            let mt = if sret.is_some() {
                Mt::I64
            } else {
                match ret {
                    Some(mt) => mt,
                    None => self.zero_for(d).0,
                }
            };
            self.set_mt(d, mt);
            if ret.is_none() && sret.is_none() {
                let (_, bits) = self.zero_for(d);
                ops.push(Op::Call(Box::new(Call {
                    target,
                    args,
                    ret: None,
                    sret: None,
                    dest: None,
                })));
                ops.push(Op::Const { dest: d, bits });
                return Ok(());
            }
        }
        ops.push(Op::Call(Box::new(Call {
            target,
            args,
            ret,
            sret,
            dest,
        })));
        Ok(())
    }

    fn terminator(&mut self, from: IrBlockId, term: &IrTerminator, ops: &mut Vec<Op>) -> Term {
        match self.try_terminator(from, term, ops) {
            Ok(t) => t,
            Err(message) => Term::Trap(Arc::from(message)),
        }
    }

    fn edge(&mut self, from: IrBlockId, to: IrBlockId) -> Result<Edge, String> {
        let block = *self
            .block_index
            .get(&to)
            .ok_or_else(|| format!("Branch target {:?} not found", to))?;
        let target = &self.function.cfg.blocks[&to];
        let mut moves = Vec::with_capacity(target.phi_nodes.len());
        for phi in &target.phi_nodes {
            let (_, value) = phi
                .incoming
                .iter()
                .find(|(b, _)| *b == from)
                .ok_or_else(|| {
                    format!(
                        "No incoming value for phi node {:?} from block {:?}",
                        phi.dest, from
                    )
                })?;
            let (c1, c2) = phi_coercion(self.mt(*value), Mt::phi(&phi.ty));
            moves.push((phi.dest.as_u32(), value.as_u32(), c1, c2));
        }
        Ok(Edge { block, moves })
    }

    fn try_terminator(
        &mut self,
        from: IrBlockId,
        term: &IrTerminator,
        ops: &mut Vec<Op>,
    ) -> Result<Term, String> {
        Ok(match term {
            IrTerminator::Branch { target } => Term::Jump(self.edge(from, *target)?),
            IrTerminator::CondBranch {
                condition,
                true_target,
                false_target,
            } => Term::Brif {
                cond: condition.as_u32(),
                then: self.edge(from, *true_target)?,
                otherwise: self.edge(from, *false_target)?,
            },
            IrTerminator::Switch {
                value,
                cases,
                default,
            } => {
                let mut out = Vec::with_capacity(cases.len());
                for (v, t) in cases {
                    out.push((*v, self.edge(from, *t)?));
                }
                Term::Switch {
                    value: value.as_u32(),
                    mt: self.mt(*value),
                    cases: out,
                    default: self.edge(from, *default)?,
                }
            }
            IrTerminator::Return { value } => {
                let sig = &self.function.signature;
                if sig.uses_sret {
                    let Some(v) = value else {
                        return Ok(Term::Return(None));
                    };
                    let ty = self
                        .reg_type(*v)
                        .or_else(|| self.local_type(*v))
                        .cloned()
                        .ok_or_else(|| format!("Cannot find type for return value {:?}", v))?;
                    let layout = struct_layout(&ty)
                        .ok_or_else(|| format!("sret with non-struct type: {:?}", ty))?;
                    return Ok(Term::ReturnSret {
                        value: v.as_u32(),
                        fields: layout.iter().map(|(o, mt, _)| (*o, *mt)).collect(),
                    });
                }
                match value {
                    Some(v) => {
                        let from = self.mt(*v);
                        let conv = match (from, Mt::of(&sig.return_type)) {
                            (Mt::I64, Some(Mt::I32)) => Some(Conv::Reduce(Mt::I32)),
                            (Mt::I32, Some(Mt::I64)) => Some(Conv::Sext(Mt::I32, Mt::I64)),
                            _ => None,
                        };
                        Term::Return(Some(self.coerce(ops, v.as_u32(), conv)))
                    }
                    None if sig.return_type != IrType::Void => {
                        return Err(format!(
                            "Function '{}' has return type {:?} but encountered Return with no value",
                            self.function.name, sig.return_type
                        ));
                    }
                    None => Term::Return(None),
                }
            }
            IrTerminator::Unreachable => Term::Trap(Arc::from("unreachable code")),
            IrTerminator::NoReturn { .. } => Term::Trap(Arc::from("NoReturn terminator executed")),
        })
    }
}

/// Arguments padded to the declared count: a missing trailing argument is
/// zero of its type, or null for a `Null<Int>`.
fn pad(args: Vec<(Reg, Mt)>, declared: &[Mt], null_int: &[bool]) -> Vec<(Src, Mt)> {
    let mut out: Vec<(Src, Mt)> = args.into_iter().map(|(r, mt)| (Src::Reg(r), mt)).collect();
    for i in out.len()..declared.len() {
        let bits = if null_int.get(i).copied().unwrap_or(false) {
            crate::ir::NULL_INT as u64
        } else {
            0
        };
        out.push((Src::Imm(bits), declared[i]));
    }
    out
}
