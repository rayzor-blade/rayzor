//! MIR Dump Utility
//!
//! Pretty-prints MIR in a human-readable format similar to LLVM IR / Cranelift CLIF.
//! Useful for debugging optimization passes.

use super::{
    BinaryOp, CompareOp, IrBasicBlock, IrBlockId, IrControlFlowGraph, IrFunction, IrId,
    IrInstruction, IrModule, IrPhiNode, IrTerminator, IrType, IrValue,
};
use std::fmt::Write;

/// Dump an entire module to a string.
pub fn dump_module(module: &IrModule) -> String {
    let mut out = String::new();
    writeln!(out, "; Module: {}", module.name).unwrap();
    writeln!(out, "; Functions: {}", module.functions.len()).unwrap();
    writeln!(out).unwrap();

    // Sort functions by ID for consistent output
    let mut func_ids: Vec<_> = module.functions.keys().collect();
    func_ids.sort_by_key(|id| id.0);

    for &func_id in &func_ids {
        let func = &module.functions[func_id];
        writeln!(out, "; {} = @{}", func_id, func.name).unwrap();
        writeln!(out, "{}", dump_function(func)).unwrap();
    }

    out
}

/// Dump a single function to a string.
pub fn dump_function(func: &IrFunction) -> String {
    let mut out = String::new();

    // Function signature
    let params: Vec<String> = func
        .signature
        .parameters
        .iter()
        .map(|p| format!("{}: {}", p.reg, dump_type(&p.ty)))
        .collect();

    writeln!(
        out,
        "fn @{}({}) -> {} {{",
        func.name,
        params.join(", "),
        dump_type(&func.signature.return_type)
    )
    .unwrap();

    // Dump CFG
    writeln!(out, "{}", dump_cfg(&func.cfg)).unwrap();

    writeln!(out, "}}").unwrap();
    out
}

/// Dump a CFG to a string.
pub fn dump_cfg(cfg: &IrControlFlowGraph) -> String {
    let mut out = String::new();

    // Sort blocks by ID for consistent output
    let mut block_ids: Vec<_> = cfg.blocks.keys().collect();
    block_ids.sort_by_key(|id| id.0);

    for &block_id in &block_ids {
        let block = &cfg.blocks[block_id];
        write!(out, "{}", dump_block(block)).unwrap();
    }

    out
}

/// Dump a basic block to a string.
pub fn dump_block(block: &IrBasicBlock) -> String {
    let mut out = String::new();

    // Block header
    let label = block
        .label
        .as_ref()
        .map(|l| format!(" ; {}", l))
        .unwrap_or_default();
    writeln!(out, "  {}:{}", block.id, label).unwrap();

    // Predecessors
    if !block.predecessors.is_empty() {
        let preds: Vec<String> = block
            .predecessors
            .iter()
            .map(|p| format!("{}", p))
            .collect();
        writeln!(out, "    ; preds: {}", preds.join(", ")).unwrap();
    }

    // Phi nodes
    for phi in &block.phi_nodes {
        writeln!(out, "    {}", dump_phi(phi)).unwrap();
    }

    // Instructions
    for inst in &block.instructions {
        writeln!(out, "    {}", dump_instruction(inst)).unwrap();
    }

    // Terminator
    writeln!(out, "    {}", dump_terminator(&block.terminator)).unwrap();
    writeln!(out).unwrap();

    out
}

/// Dump a phi node to a string.
pub fn dump_phi(phi: &IrPhiNode) -> String {
    let incoming: Vec<String> = phi
        .incoming
        .iter()
        .map(|(block, val)| format!("[{}: {}]", block, val))
        .collect();

    format!(
        "{} = phi {} {}",
        phi.dest,
        dump_type(&phi.ty),
        incoming.join(", ")
    )
}

/// Dump an instruction to a string.
pub fn dump_instruction(inst: &IrInstruction) -> String {
    match inst {
        IrInstruction::Const { dest, value } => {
            format!("{} = const {}", dest, dump_value(value))
        }
        IrInstruction::Copy { dest, src } => {
            format!("{} = copy {}", dest, src)
        }
        IrInstruction::Move { dest, src } => {
            format!("{} = move {}", dest, src)
        }
        IrInstruction::Clone { dest, src } => {
            format!("{} = clone {}", dest, src)
        }
        IrInstruction::BorrowImmutable {
            dest,
            src,
            lifetime,
        } => {
            format!("{} = borrow_imm {} (lifetime {})", dest, src, lifetime.0)
        }
        IrInstruction::BorrowMutable {
            dest,
            src,
            lifetime,
        } => {
            format!("{} = borrow_mut {} (lifetime {})", dest, src, lifetime.0)
        }
        IrInstruction::EndBorrow { borrow } => {
            format!("end_borrow {}", borrow)
        }
        IrInstruction::BinOp {
            dest,
            op,
            left,
            right,
        } => {
            format!("{} = {} {}, {}", dest, dump_binop(op), left, right)
        }
        IrInstruction::UnOp { dest, op, operand } => {
            format!("{} = {} {}", dest, dump_unaryop(op), operand)
        }
        IrInstruction::Cmp {
            dest,
            op,
            left,
            right,
        } => {
            format!("{} = cmp {} {}, {}", dest, dump_cmpop(op), left, right)
        }
        IrInstruction::Load { dest, ptr, ty } => {
            format!("{} = load {} {}", dest, dump_type(ty), ptr)
        }
        IrInstruction::Store { ptr, value, .. } => {
            format!("store {}, {}", ptr, value)
        }
        IrInstruction::Alloc { dest, ty, count } => {
            if let Some(cnt) = count {
                format!("{} = alloc {} x {}", dest, dump_type(ty), cnt)
            } else {
                format!("{} = alloc {}", dest, dump_type(ty))
            }
        }
        IrInstruction::Free { ptr } => {
            format!("free {}", ptr)
        }
        IrInstruction::GetElementPtr {
            dest,
            ptr,
            indices,
            ty,
            ..
        } => {
            let idx_str: Vec<String> = indices.iter().map(|i| format!("{}", i)).collect();
            format!(
                "{} = gep {} {}, [{}]",
                dest,
                dump_type(ty),
                ptr,
                idx_str.join(", ")
            )
        }
        IrInstruction::PtrAdd {
            dest,
            ptr,
            offset,
            ty,
        } => {
            format!(
                "{} = ptradd {}, {} (type {})",
                dest,
                ptr,
                offset,
                dump_type(ty)
            )
        }
        IrInstruction::CallDirect {
            dest,
            func_id,
            args,
            ..
        } => {
            let args_str: Vec<String> = args.iter().map(|a| format!("{}", a)).collect();
            if let Some(d) = dest {
                format!("{} = call {}({})", d, func_id, args_str.join(", "))
            } else {
                format!("call {}({})", func_id, args_str.join(", "))
            }
        }
        IrInstruction::CallIndirect {
            dest,
            func_ptr,
            args,
            ..
        } => {
            let args_str: Vec<String> = args.iter().map(|a| format!("{}", a)).collect();
            if let Some(d) = dest {
                format!(
                    "{} = call_indirect {}({})",
                    d,
                    func_ptr,
                    args_str.join(", ")
                )
            } else {
                format!("call_indirect {}({})", func_ptr, args_str.join(", "))
            }
        }
        IrInstruction::Cast {
            dest,
            src,
            from_ty,
            to_ty,
        } => {
            format!(
                "{} = cast {} {} to {}",
                dest,
                dump_type(from_ty),
                src,
                dump_type(to_ty)
            )
        }
        IrInstruction::BitCast { dest, src, ty } => {
            format!("{} = bitcast {} to {}", dest, src, dump_type(ty))
        }
        IrInstruction::Select {
            dest,
            condition,
            true_val,
            false_val,
        } => {
            format!(
                "{} = select {}, {}, {}",
                dest, condition, true_val, false_val
            )
        }
        IrInstruction::Undef { dest, ty } => {
            format!("{} = undef {}", dest, dump_type(ty))
        }
        IrInstruction::Return { value } => {
            if let Some(v) = value {
                format!("ret {}", v)
            } else {
                "ret void".to_string()
            }
        }
        IrInstruction::MemCopy { dest, src, size } => {
            format!("memcpy {}, {}, {}", dest, src, size)
        }
        IrInstruction::MemSet { dest, value, size } => {
            format!("memset {}, {}, {}", dest, value, size)
        }
        IrInstruction::LoadGlobal {
            dest,
            global_id,
            ty,
        } => {
            format!("{} = load_global {} @g{}", dest, dump_type(ty), global_id.0)
        }
        IrInstruction::StoreGlobal { global_id, value } => {
            format!("store_global @g{}, {}", global_id.0, value)
        }
        IrInstruction::MakeClosure {
            dest,
            func_id,
            captured_values,
        } => {
            let caps: Vec<String> = captured_values.iter().map(|v| format!("{}", v)).collect();
            format!("{} = make_closure {}, [{}]", dest, func_id, caps.join(", "))
        }
        IrInstruction::Throw { exception } => {
            format!("throw {}", exception)
        }
        IrInstruction::Phi { dest, incoming } => {
            let inc: Vec<String> = incoming
                .iter()
                .map(|(val, block)| format!("[{}: {}]", block, val))
                .collect();
            format!("{} = phi {}", dest, inc.join(", "))
        }
        IrInstruction::ExtractValue {
            dest,
            aggregate,
            indices,
        } => {
            let fields: String = indices.iter().map(|i| format!(".{}", i)).collect();
            format!("{} = {}{}", dest, aggregate, fields)
        }
        IrInstruction::InsertValue {
            dest,
            aggregate,
            value,
            indices,
        } => {
            let idx: Vec<String> = indices.iter().map(|i| format!("{}", i)).collect();
            format!(
                "{} = insert_value {}, {}, [{}]",
                dest,
                aggregate,
                value,
                idx.join(", ")
            )
        }
        IrInstruction::Jump { target } => {
            format!("jump {}", target)
        }
        IrInstruction::Branch {
            condition,
            true_target,
            false_target,
        } => {
            format!("branch {}, {}, {}", condition, true_target, false_target)
        }
        IrInstruction::Switch {
            value,
            default_target,
            cases,
        } => {
            let cases_str: Vec<String> = cases
                .iter()
                .map(|(val, target)| format!("{} => {}", dump_value(val), target))
                .collect();
            format!(
                "switch {} [{}] default {}",
                value,
                cases_str.join(", "),
                default_target
            )
        }
        IrInstruction::LandingPad { dest, ty, .. } => {
            format!("{} = landing_pad {}", dest, dump_type(ty))
        }
        IrInstruction::Resume { exception } => {
            format!("resume {}", exception)
        }
        IrInstruction::CreateUnion {
            dest,
            discriminant,
            value,
            ty,
        } => {
            let (type_name, variant_name) = match ty {
                IrType::Union { name, variants } => {
                    let vname = variants
                        .get(*discriminant as usize)
                        .map(|v| v.name.as_str())
                        .unwrap_or("?");
                    (name.as_str(), vname)
                }
                _ => ("?", "?"),
            };
            format!(
                "{} = union {} {{ {}: {} }}",
                dest, type_name, variant_name, value
            )
        }
        IrInstruction::CreateStruct { dest, ty, fields } => {
            let type_name = match ty {
                IrType::Struct { name, .. } => name.clone(),
                _ => dump_type(ty),
            };
            let field_strs: Vec<String> = fields.iter().map(|f| format!("{}", f)).collect();
            format!(
                "{} = struct {} {{ {} }}",
                dest,
                type_name,
                field_strs.join(", ")
            )
        }
        IrInstruction::ExtractDiscriminant { dest, union_val } => {
            format!("{} = discriminant {}", dest, union_val)
        }
        IrInstruction::ExtractUnionValue {
            dest,
            union_val,
            discriminant,
            value_ty,
        } => {
            format!(
                "{} = extract {} {} .{}",
                dest,
                dump_type(value_ty),
                union_val,
                discriminant
            )
        }
        IrInstruction::FunctionRef { dest, func_id } => {
            format!("{} = fn_ref @fn{}", dest, func_id.0)
        }
        IrInstruction::VectorBinOp {
            dest,
            op,
            left,
            right,
            vec_ty,
        } => {
            let prefix = vec_type_prefix(vec_ty);
            format!(
                "{} = {}.{} {}, {}",
                dest,
                prefix,
                format!("{:?}", op).to_lowercase(),
                left,
                right
            )
        }
        IrInstruction::VectorUnaryOp {
            dest,
            op,
            operand,
            vec_ty,
        } => {
            let prefix = vec_type_prefix(vec_ty);
            format!(
                "{} = {}.{} {}",
                dest,
                prefix,
                format!("{:?}", op).to_lowercase(),
                operand
            )
        }
        IrInstruction::VectorSplat {
            dest,
            scalar,
            vec_ty,
        } => {
            let prefix = vec_type_prefix(vec_ty);
            format!("{} = {}.splat {}", dest, prefix, scalar)
        }
        IrInstruction::VectorExtract {
            dest,
            vector,
            index,
        } => {
            format!("{} = simd4f.extract {}[{}]", dest, vector, index)
        }
        IrInstruction::VectorInsert {
            dest,
            vector,
            scalar,
            index,
        } => {
            format!("{} = simd4f.insert {}[{}], {}", dest, vector, index, scalar)
        }
        IrInstruction::VectorReduce {
            dest, op, vector, ..
        } => {
            format!(
                "{} = simd4f.reduce.{} {}",
                dest,
                format!("{:?}", op).to_lowercase(),
                vector
            )
        }
        IrInstruction::VectorLoad { dest, ptr, vec_ty } => {
            let prefix = vec_type_prefix(vec_ty);
            format!("{} = {}.load {}", dest, prefix, ptr)
        }
        IrInstruction::VectorStore { ptr, value, vec_ty } => {
            let prefix = vec_type_prefix(vec_ty);
            format!("{}.store {}, {}", prefix, ptr, value)
        }
        IrInstruction::VectorMinMax {
            dest,
            op,
            left,
            right,
            vec_ty,
        } => {
            let prefix = vec_type_prefix(vec_ty);
            format!(
                "{} = {}.{} {}, {}",
                dest,
                prefix,
                format!("{:?}", op).to_lowercase(),
                left,
                right
            )
        }
        _ => format!("{:?}", inst),
    }
}

/// Dump a terminator to a string.
pub fn dump_terminator(term: &IrTerminator) -> String {
    match term {
        IrTerminator::Branch { target } => {
            format!("br {}", target)
        }
        IrTerminator::CondBranch {
            condition,
            true_target,
            false_target,
        } => {
            format!("br_if {}, {}, {}", condition, true_target, false_target)
        }
        IrTerminator::Switch {
            value,
            cases,
            default,
        } => {
            let cases_str: Vec<String> = cases
                .iter()
                .map(|(val, target)| format!("{} => {}", val, target))
                .collect();
            format!(
                "switch {} [{}] default {}",
                value,
                cases_str.join(", "),
                default
            )
        }
        IrTerminator::Return { value } => {
            if let Some(v) = value {
                format!("ret {}", v)
            } else {
                "ret void".to_string()
            }
        }
        IrTerminator::Unreachable => "unreachable".to_string(),
        IrTerminator::NoReturn { call } => {
            format!("noreturn {}", call)
        }
    }
}

/// Dump a type to a string.
/// Get a SIMD type prefix like "simd4f", "simd4i" from a vector type.
fn vec_type_prefix(ty: &IrType) -> String {
    match ty {
        IrType::Vector { element, count } => {
            let elem = match element.as_ref() {
                IrType::F32 => "f",
                IrType::F64 => "d",
                IrType::I32 => "i",
                IrType::I64 => "l",
                _ => "x",
            };
            format!("simd{}{}", count, elem)
        }
        _ => "simd".to_string(),
    }
}

pub fn dump_type(ty: &IrType) -> String {
    match ty {
        IrType::Void => "void".to_string(),
        IrType::Bool => "bool".to_string(),
        IrType::I8 => "i8".to_string(),
        IrType::I16 => "i16".to_string(),
        IrType::I32 => "i32".to_string(),
        IrType::I64 => "i64".to_string(),
        IrType::U8 => "u8".to_string(),
        IrType::U16 => "u16".to_string(),
        IrType::U32 => "u32".to_string(),
        IrType::U64 => "u64".to_string(),
        IrType::F32 => "f32".to_string(),
        IrType::F64 => "f64".to_string(),
        IrType::Ptr(inner) => format!("*{}", dump_type(inner)),
        IrType::Ref(inner) => format!("&{}", dump_type(inner)),
        IrType::Array(elem, size) => format!("[{} x {}]", dump_type(elem), size),
        IrType::Slice(elem) => format!("[{}]", dump_type(elem)),
        IrType::Vector { element, count } => format!("<{} x {}>", count, dump_type(element)),
        IrType::String => "string".to_string(),
        IrType::Struct { name, fields } => {
            if fields.is_empty() {
                format!("%{}", name)
            } else {
                let f: Vec<String> = fields.iter().map(|f| dump_type(&f.ty)).collect();
                format!("%{}{{ {} }}", name, f.join(", "))
            }
        }
        IrType::Union { name, .. } => format!("union %{}", name),
        IrType::Opaque { name, .. } => format!("opaque({})", name),
        IrType::TypeVar(name) => format!("?{}", name),
        IrType::Generic { base, type_args } => {
            let args: Vec<String> = type_args.iter().map(|t| dump_type(t)).collect();
            format!("{}<{}>", dump_type(base), args.join(", "))
        }
        IrType::Any => "any".to_string(),
        IrType::Function {
            params,
            return_type,
            varargs,
        } => {
            let p: Vec<String> = params.iter().map(|t| dump_type(t)).collect();
            let va = if *varargs { ", ..." } else { "" };
            format!("fn({}{}) -> {}", p.join(", "), va, dump_type(return_type))
        }
    }
}

/// Dump a value to a string.
pub fn dump_value(value: &IrValue) -> String {
    match value {
        IrValue::Void => "void".to_string(),
        IrValue::Undef => "undef".to_string(),
        IrValue::Null => "null".to_string(),
        IrValue::Bool(b) => format!("{}", b),
        IrValue::I8(v) => format!("{}i8", v),
        IrValue::I16(v) => format!("{}i16", v),
        IrValue::I32(v) => format!("{}i32", v),
        IrValue::I64(v) => format!("{}i64", v),
        IrValue::U8(v) => format!("{}u8", v),
        IrValue::U16(v) => format!("{}u16", v),
        IrValue::U32(v) => format!("{}u32", v),
        IrValue::U64(v) => format!("{}u64", v),
        IrValue::F32(v) => format!("{}f32", v),
        IrValue::F64(v) => format!("{}f64", v),
        IrValue::String(s) => format!("\"{}\"", s.escape_default()),
        IrValue::Array(elems) => {
            let e: Vec<String> = elems.iter().map(|v| dump_value(v)).collect();
            format!("[{}]", e.join(", "))
        }
        IrValue::Struct(fields) => {
            let f: Vec<String> = fields.iter().map(|v| dump_value(v)).collect();
            format!("{{ {} }}", f.join(", "))
        }
        IrValue::Function(id) => format!("@fn{}", id.0),
        IrValue::Closure {
            function,
            environment,
        } => {
            format!("closure(@fn{}, {})", function.0, dump_value(environment))
        }
    }
}

/// Dump a binary operator to a string.
pub fn dump_binop(op: &BinaryOp) -> String {
    match op {
        BinaryOp::Add => "add",
        BinaryOp::Sub => "sub",
        BinaryOp::Mul => "mul",
        BinaryOp::Div => "div",
        BinaryOp::Rem => "rem",
        BinaryOp::And => "and",
        BinaryOp::Or => "or",
        BinaryOp::Xor => "xor",
        BinaryOp::Shl => "shl",
        BinaryOp::Shr => "shr",
        BinaryOp::Ushr => "ushr",
        BinaryOp::FAdd => "fadd",
        BinaryOp::FSub => "fsub",
        BinaryOp::FMul => "fmul",
        BinaryOp::FDiv => "fdiv",
        BinaryOp::FRem => "frem",
    }
    .to_string()
}

/// Dump a unary operator to a string.
pub fn dump_unaryop(op: &super::UnaryOp) -> String {
    match op {
        super::UnaryOp::Neg => "neg",
        super::UnaryOp::Not => "not",
        super::UnaryOp::FNeg => "fneg",
    }
    .to_string()
}

/// Dump a comparison operator to a string.
pub fn dump_cmpop(op: &CompareOp) -> String {
    match op {
        CompareOp::Eq => "eq",
        CompareOp::Ne => "ne",
        CompareOp::Lt => "lt",
        CompareOp::Le => "le",
        CompareOp::Gt => "gt",
        CompareOp::Ge => "ge",
        CompareOp::ULt => "ult",
        CompareOp::ULe => "ule",
        CompareOp::UGt => "ugt",
        CompareOp::UGe => "uge",
        CompareOp::FEq => "feq",
        CompareOp::FNe => "fne",
        CompareOp::FLt => "flt",
        CompareOp::FLe => "fle",
        CompareOp::FGt => "fgt",
        CompareOp::FGe => "fge",
        CompareOp::FOrd => "ford",
        CompareOp::FUno => "funo",
    }
    .to_string()
}

/// Dump a specific function by name from a module.
pub fn dump_function_by_name(module: &IrModule, name: &str) -> Option<String> {
    for func in module.functions.values() {
        if func.name == name {
            return Some(dump_function(func));
        }
    }
    None
}
