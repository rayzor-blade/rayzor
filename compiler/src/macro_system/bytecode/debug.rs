//! Disassembler for macro bytecode chunks.
//!
//! Provides human-readable output of compiled bytecode for debugging.

use super::chunk::Chunk;
use super::opcode::{Op, Reader};

/// Disassemble a chunk into a human-readable string.
pub fn disassemble(chunk: &Chunk) -> String {
    let mut out = String::new();
    out.push_str(&format!("=== {} ===\n", chunk.name));
    out.push_str(&format!(
        "locals: {}, params: {}, constants: {}, closures: {}\n",
        chunk.local_count,
        chunk.params.len(),
        chunk.constants.len(),
        chunk.closures.len(),
    ));

    if !chunk.local_names.is_empty() {
        out.push_str("local names:");
        for (slot, name) in &chunk.local_names {
            out.push_str(&format!(" [{}]={}", slot, name));
        }
        out.push('\n');
    }

    out.push_str("--- bytecode ---\n");

    let mut reader = Reader::new(&chunk.code);
    while reader.ip < chunk.code.len() {
        let offset = reader.ip;
        let Some(op) = reader.read_op() else {
            out.push_str(&format!(
                "{:04x}  ??? (0x{:02x})\n",
                offset, chunk.code[offset]
            ));
            reader.ip += 1;
            continue;
        };

        let mut line = format!("{:04x}  {:<20}", offset, op.name());

        match op {
            // No operands
            Op::PushNull
            | Op::PushTrue
            | Op::PushFalse
            | Op::PushInt0
            | Op::PushInt1
            | Op::Pop
            | Op::Add
            | Op::Sub
            | Op::Mul
            | Op::Div
            | Op::Mod
            | Op::Eq
            | Op::NotEq
            | Op::Lt
            | Op::Le
            | Op::Gt
            | Op::Ge
            | Op::BitAnd
            | Op::BitOr
            | Op::BitXor
            | Op::Shl
            | Op::Shr
            | Op::Ushr
            | Op::NullCoal
            | Op::Not
            | Op::Neg
            | Op::BitNot
            | Op::Incr
            | Op::Decr
            | Op::Return
            | Op::ReturnNull
            | Op::GetIndex
            | Op::SetIndex
            | Op::Reify
            | Op::MacroWrap
            | Op::Dup
            | Op::Swap => {}

            // u16 operand
            Op::Const => {
                let idx = reader.read_u16();
                let val_str = if let Some(val) = chunk.constants.get(idx as usize) {
                    format!("{}", val)
                } else {
                    "???".to_string()
                };
                line.push_str(&format!("{:<6} ; {}", idx, val_str));
            }
            Op::LoadLocal | Op::StoreLocal | Op::DefineLocal => {
                let slot = reader.read_u16();
                let name = chunk
                    .local_names
                    .iter()
                    .find(|(s, _)| *s == slot)
                    .map(|(_, n)| n.as_str())
                    .unwrap_or("");
                if name.is_empty() {
                    line.push_str(&format!("{}", slot));
                } else {
                    line.push_str(&format!("{:<6} ; {}", slot, name));
                }
            }
            Op::LoadUpvalue => {
                let slot = reader.read_u16();
                line.push_str(&format!("{}", slot));
            }
            Op::GetField | Op::SetField | Op::GetFieldOpt => {
                let idx = reader.read_u16();
                let name = const_string(chunk, idx);
                line.push_str(&format!("{:<6} ; .{}", idx, name));
            }
            Op::SetFieldLocal => {
                let slot = reader.read_u16();
                let idx = reader.read_u16();
                let field_name = const_string(chunk, idx);
                let local_name = chunk
                    .local_names
                    .iter()
                    .find(|(s, _)| *s == slot)
                    .map(|(_, n)| n.as_str())
                    .unwrap_or("?");
                line.push_str(&format!(
                    "[{}].{} ; {}.{}",
                    slot, idx, local_name, field_name
                ));
            }
            Op::MakeArray | Op::MakeObject | Op::MakeMap => {
                let count = reader.read_u16();
                line.push_str(&format!("{}", count));
            }
            Op::MakeClosure => {
                let idx = reader.read_u16();
                let cname = chunk
                    .closures
                    .get(idx as usize)
                    .map(|c| c.name.as_str())
                    .unwrap_or("???");
                line.push_str(&format!("{:<6} ; {}", idx, cname));
            }
            Op::DollarSplice => {
                let idx = reader.read_u16();
                let name = const_string(chunk, idx);
                line.push_str(&format!("{:<6} ; ${}", idx, name));
            }

            // i16 operand (jumps)
            Op::Jump
            | Op::JumpIfFalse
            | Op::JumpIfTrue
            | Op::JumpIfFalseKeep
            | Op::JumpIfTrueKeep => {
                let offset_val = reader.read_i16();
                let target = (reader.ip as isize + offset_val as isize) as usize;
                line.push_str(&format!("{:<6} ; -> {:04x}", offset_val, target));
            }

            // u8 operand
            Op::Call => {
                let nargs = reader.read_u8();
                line.push_str(&format!("nargs={}", nargs));
            }

            // u16 + u8 operands
            Op::CallMethod => {
                let name_idx = reader.read_u16();
                let nargs = reader.read_u8();
                let name = const_string(chunk, name_idx);
                line.push_str(&format!(".{}({})", name, nargs));
            }
            Op::CallBuiltin => {
                let id = reader.read_u16();
                let nargs = reader.read_u8();
                if id == 0xFFFF {
                    line.push_str(&format!("throw({})", nargs));
                } else {
                    line.push_str(&format!("builtin#{}({})", id, nargs));
                }
            }
            Op::CallMacroDef => {
                let id = reader.read_u16();
                let nargs = reader.read_u8();
                line.push_str(&format!("macro#{}({})", id, nargs));
            }
            Op::NewObject => {
                let name_idx = reader.read_u16();
                let nargs = reader.read_u8();
                let name = const_string(chunk, name_idx);
                line.push_str(&format!("new {}({})", name, nargs));
            }

            // u16 + u16 + u8 operands
            Op::CallStatic => {
                let class_idx = reader.read_u16();
                let method_idx = reader.read_u16();
                let nargs = reader.read_u8();
                let class = const_string(chunk, class_idx);
                let method = const_string(chunk, method_idx);
                line.push_str(&format!("{}.{}({})", class, method, nargs));
            }
        }

        out.push_str(&line);
        out.push('\n');
    }

    // Disassemble closures recursively
    for (i, closure) in chunk.closures.iter().enumerate() {
        out.push_str(&format!("\n--- closure[{}] ---\n", i));
        out.push_str(&disassemble(closure));
    }

    out
}

fn const_string(chunk: &Chunk, idx: u16) -> String {
    chunk
        .constants
        .get(idx as usize)
        .map(|v| v.to_display_string())
        .unwrap_or_else(|| "???".to_string())
}

#[cfg(test)]
mod tests {
    use super::super::super::value::MacroParam;
    use super::super::compiler::BytecodeCompiler;
    use super::*;
    use parser::{BinaryOp, BlockElement, Expr, ExprKind, Span};

    fn make_expr(kind: ExprKind) -> Expr {
        Expr {
            kind,
            span: Span { start: 0, end: 0 },
        }
    }

    #[test]
    fn test_disassemble_simple() {
        let body = make_expr(ExprKind::Return(Some(Box::new(make_expr(
            ExprKind::Binary {
                left: Box::new(make_expr(ExprKind::Ident("x".to_string()))),
                op: BinaryOp::Add,
                right: Box::new(make_expr(ExprKind::Int(1))),
            },
        )))));
        let params = vec![MacroParam {
            name: "x".to_string(),
            optional: false,
            rest: false,
            default_value: None,
        }];

        let chunk = BytecodeCompiler::compile("addOne", &params, &body).unwrap();
        let output = disassemble(&chunk);

        assert!(output.contains("=== addOne ==="));
        assert!(output.contains("LoadLocal"));
        assert!(output.contains("Add"));
        assert!(output.contains("Return"));
    }

    #[test]
    fn test_disassemble_with_jump() {
        let body = make_expr(ExprKind::If {
            cond: Box::new(make_expr(ExprKind::Bool(true))),
            then_branch: Box::new(make_expr(ExprKind::Int(1))),
            else_branch: Some(Box::new(make_expr(ExprKind::Int(2)))),
        });
        let chunk = BytecodeCompiler::compile("test", &[], &body).unwrap();
        let output = disassemble(&chunk);

        assert!(output.contains("JumpIfFalse"));
        assert!(output.contains("Jump"));
        assert!(output.contains("->"));
    }
}
