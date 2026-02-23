//! Bytecode opcodes for the macro VM.
//!
//! Instructions are variable-length encoded: `[opcode: u8][operand bytes...]`.
//! Operands use little-endian encoding: u16 = 2 bytes, i16 = 2 bytes signed.

/// Opcode discriminants. Each variant maps to a unique u8 tag.
///
/// Instructions that take operands encode them inline in the bytecode stream
/// after the opcode byte. The `Emitter` and `Reader` handle encoding/decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Op {
    // === Constants & Stack (0x00-0x06) ===
    /// Push constants[idx] onto the stack.
    Const = 0x00,
    /// Push Null.
    PushNull = 0x01,
    /// Push Bool(true).
    PushTrue = 0x02,
    /// Push Bool(false).
    PushFalse = 0x03,
    /// Push Int(0).
    PushInt0 = 0x04,
    /// Push Int(1).
    PushInt1 = 0x05,
    /// Discard top of stack.
    Pop = 0x06,

    // === Local Variables (0x07-0x0A) ===
    /// Push locals[slot] onto the stack.
    LoadLocal = 0x07,
    /// locals[slot] = peek top of stack (does NOT pop).
    StoreLocal = 0x08,
    /// locals[slot] = pop top, then push value back (for var decl).
    DefineLocal = 0x09,
    /// Push upvalues[slot] (for closures).
    LoadUpvalue = 0x0A,

    // === Binary Ops (0x10-0x22) ===
    Add = 0x10,
    Sub = 0x11,
    Mul = 0x12,
    Div = 0x13,
    Mod = 0x14,
    Eq = 0x15,
    NotEq = 0x16,
    Lt = 0x17,
    Le = 0x18,
    Gt = 0x19,
    Ge = 0x1A,
    BitAnd = 0x1B,
    BitOr = 0x1C,
    BitXor = 0x1D,
    Shl = 0x1E,
    Shr = 0x1F,
    Ushr = 0x20,
    NullCoal = 0x21,

    // === Unary Ops (0x28-0x2C) ===
    Not = 0x28,
    Neg = 0x29,
    BitNot = 0x2A,
    Incr = 0x2B,
    Decr = 0x2C,

    // === Control Flow (0x30-0x34) ===
    /// Unconditional jump by signed i16 offset from next instruction.
    Jump = 0x30,
    /// Pop top; jump if falsy.
    JumpIfFalse = 0x31,
    /// Pop top; jump if truthy.
    JumpIfTrue = 0x32,
    /// Peek top (no pop); jump if falsy (for short-circuit &&).
    JumpIfFalseKeep = 0x33,
    /// Peek top (no pop); jump if truthy (for short-circuit ||).
    JumpIfTrueKeep = 0x34,

    // === Function Calls (0x38-0x3F) ===
    /// Pop function value, pop N args, call, push result. Operand: u8 nargs.
    Call = 0x38,
    /// Pop base, call method. Operands: u16 name_const_idx, u8 nargs.
    CallMethod = 0x39,
    /// Call static method. Operands: u16 class_const_idx, u16 method_const_idx, u8 nargs.
    CallStatic = 0x3A,
    /// Call builtin by ID. Operands: u16 builtin_id, u8 nargs.
    CallBuiltin = 0x3B,
    /// Call registered macro def. Operands: u16 macro_id, u8 nargs.
    CallMacroDef = 0x3C,
    /// Pop value, return from current frame.
    Return = 0x3D,
    /// Return Null from current frame.
    ReturnNull = 0x3E,

    // === Field & Index (0x40-0x44) ===
    /// Pop base, push base.field. Operand: u16 name_const_idx.
    GetField = 0x40,
    /// Pop value, pop base, base.field = value, push value. Operand: u16 name_const_idx.
    SetField = 0x41,
    /// Optional chaining: pop base, push base?.field (Null if base is Null). Operand: u16 name_const_idx.
    GetFieldOpt = 0x42,
    /// Pop index, pop base, push base[index].
    GetIndex = 0x43,
    /// Pop value, pop index, pop base, base[index] = value, push value.
    SetIndex = 0x44,

    // === Collection Construction (0x48-0x4C) ===
    /// Pop N values, build Array. Operand: u16 count.
    MakeArray = 0x48,
    /// Pop N (const_name_idx, value) pairs, build Object. Operand: u16 count.
    MakeObject = 0x49,
    /// Pop N (key, value) pairs, build Object. Operand: u16 count.
    MakeMap = 0x4A,
    /// Build closure from closures[chunk_idx]. Operand: u16 chunk_idx.
    MakeClosure = 0x4B,
    /// Construct class. Operands: u16 class_name_const_idx, u8 nargs.
    NewObject = 0x4C,

    // === Reification & Macros (0x50-0x52) ===
    /// Pop expr, reify with current env, push result.
    Reify = 0x50,
    /// Pop value, splice via kind. Operand: u16 kind_const_idx.
    DollarSplice = 0x51,
    /// Pop inner expression, wrap as MacroValue::Expr.
    MacroWrap = 0x52,

    /// Set field on a local variable in-place. Operands: u16 local_slot, u16 name_const_idx.
    /// Pop value from stack, modify local directly, push value.
    SetFieldLocal = 0x45,

    // === Misc (0x58-0x59) ===
    /// Duplicate top of stack.
    Dup = 0x58,
    /// Swap top two stack elements.
    Swap = 0x59,
}

impl Op {
    /// Decode an opcode from its u8 discriminant.
    /// Returns None for unknown/reserved opcodes.
    pub fn from_byte(byte: u8) -> Option<Op> {
        match byte {
            0x00 => Some(Op::Const),
            0x01 => Some(Op::PushNull),
            0x02 => Some(Op::PushTrue),
            0x03 => Some(Op::PushFalse),
            0x04 => Some(Op::PushInt0),
            0x05 => Some(Op::PushInt1),
            0x06 => Some(Op::Pop),

            0x07 => Some(Op::LoadLocal),
            0x08 => Some(Op::StoreLocal),
            0x09 => Some(Op::DefineLocal),
            0x0A => Some(Op::LoadUpvalue),

            0x10 => Some(Op::Add),
            0x11 => Some(Op::Sub),
            0x12 => Some(Op::Mul),
            0x13 => Some(Op::Div),
            0x14 => Some(Op::Mod),
            0x15 => Some(Op::Eq),
            0x16 => Some(Op::NotEq),
            0x17 => Some(Op::Lt),
            0x18 => Some(Op::Le),
            0x19 => Some(Op::Gt),
            0x1A => Some(Op::Ge),
            0x1B => Some(Op::BitAnd),
            0x1C => Some(Op::BitOr),
            0x1D => Some(Op::BitXor),
            0x1E => Some(Op::Shl),
            0x1F => Some(Op::Shr),
            0x20 => Some(Op::Ushr),
            0x21 => Some(Op::NullCoal),

            0x28 => Some(Op::Not),
            0x29 => Some(Op::Neg),
            0x2A => Some(Op::BitNot),
            0x2B => Some(Op::Incr),
            0x2C => Some(Op::Decr),

            0x30 => Some(Op::Jump),
            0x31 => Some(Op::JumpIfFalse),
            0x32 => Some(Op::JumpIfTrue),
            0x33 => Some(Op::JumpIfFalseKeep),
            0x34 => Some(Op::JumpIfTrueKeep),

            0x38 => Some(Op::Call),
            0x39 => Some(Op::CallMethod),
            0x3A => Some(Op::CallStatic),
            0x3B => Some(Op::CallBuiltin),
            0x3C => Some(Op::CallMacroDef),
            0x3D => Some(Op::Return),
            0x3E => Some(Op::ReturnNull),

            0x40 => Some(Op::GetField),
            0x41 => Some(Op::SetField),
            0x42 => Some(Op::GetFieldOpt),
            0x43 => Some(Op::GetIndex),
            0x44 => Some(Op::SetIndex),
            0x45 => Some(Op::SetFieldLocal),

            0x48 => Some(Op::MakeArray),
            0x49 => Some(Op::MakeObject),
            0x4A => Some(Op::MakeMap),
            0x4B => Some(Op::MakeClosure),
            0x4C => Some(Op::NewObject),

            0x50 => Some(Op::Reify),
            0x51 => Some(Op::DollarSplice),
            0x52 => Some(Op::MacroWrap),

            0x58 => Some(Op::Dup),
            0x59 => Some(Op::Swap),

            _ => None,
        }
    }

    /// Return the name of this opcode for debugging.
    pub fn name(self) -> &'static str {
        match self {
            Op::Const => "Const",
            Op::PushNull => "PushNull",
            Op::PushTrue => "PushTrue",
            Op::PushFalse => "PushFalse",
            Op::PushInt0 => "PushInt0",
            Op::PushInt1 => "PushInt1",
            Op::Pop => "Pop",
            Op::LoadLocal => "LoadLocal",
            Op::StoreLocal => "StoreLocal",
            Op::DefineLocal => "DefineLocal",
            Op::LoadUpvalue => "LoadUpvalue",
            Op::Add => "Add",
            Op::Sub => "Sub",
            Op::Mul => "Mul",
            Op::Div => "Div",
            Op::Mod => "Mod",
            Op::Eq => "Eq",
            Op::NotEq => "NotEq",
            Op::Lt => "Lt",
            Op::Le => "Le",
            Op::Gt => "Gt",
            Op::Ge => "Ge",
            Op::BitAnd => "BitAnd",
            Op::BitOr => "BitOr",
            Op::BitXor => "BitXor",
            Op::Shl => "Shl",
            Op::Shr => "Shr",
            Op::Ushr => "Ushr",
            Op::NullCoal => "NullCoal",
            Op::Not => "Not",
            Op::Neg => "Neg",
            Op::BitNot => "BitNot",
            Op::Incr => "Incr",
            Op::Decr => "Decr",
            Op::Jump => "Jump",
            Op::JumpIfFalse => "JumpIfFalse",
            Op::JumpIfTrue => "JumpIfTrue",
            Op::JumpIfFalseKeep => "JumpIfFalseKeep",
            Op::JumpIfTrueKeep => "JumpIfTrueKeep",
            Op::Call => "Call",
            Op::CallMethod => "CallMethod",
            Op::CallStatic => "CallStatic",
            Op::CallBuiltin => "CallBuiltin",
            Op::CallMacroDef => "CallMacroDef",
            Op::Return => "Return",
            Op::ReturnNull => "ReturnNull",
            Op::GetField => "GetField",
            Op::SetField => "SetField",
            Op::GetFieldOpt => "GetFieldOpt",
            Op::GetIndex => "GetIndex",
            Op::SetIndex => "SetIndex",
            Op::SetFieldLocal => "SetFieldLocal",
            Op::MakeArray => "MakeArray",
            Op::MakeObject => "MakeObject",
            Op::MakeMap => "MakeMap",
            Op::MakeClosure => "MakeClosure",
            Op::NewObject => "NewObject",
            Op::Reify => "Reify",
            Op::DollarSplice => "DollarSplice",
            Op::MacroWrap => "MacroWrap",
            Op::Dup => "Dup",
            Op::Swap => "Swap",
        }
    }

    /// Number of inline operand bytes following this opcode.
    pub fn operand_size(self) -> usize {
        match self {
            // No operands
            Op::PushNull | Op::PushTrue | Op::PushFalse | Op::PushInt0 | Op::PushInt1 | Op::Pop => {
                0
            }
            Op::Add | Op::Sub | Op::Mul | Op::Div | Op::Mod => 0,
            Op::Eq | Op::NotEq | Op::Lt | Op::Le | Op::Gt | Op::Ge => 0,
            Op::BitAnd | Op::BitOr | Op::BitXor | Op::Shl | Op::Shr | Op::Ushr | Op::NullCoal => 0,
            Op::Not | Op::Neg | Op::BitNot | Op::Incr | Op::Decr => 0,
            Op::Return | Op::ReturnNull => 0,
            Op::GetIndex | Op::SetIndex => 0,
            Op::Reify | Op::MacroWrap => 0,
            Op::Dup | Op::Swap => 0,

            // u16 operand (2 bytes)
            Op::Const | Op::LoadLocal | Op::StoreLocal | Op::DefineLocal | Op::LoadUpvalue => 2,
            Op::GetField | Op::SetField | Op::GetFieldOpt => 2,
            Op::SetFieldLocal => 4, // u16 local_slot + u16 name_const_idx
            Op::MakeArray | Op::MakeObject | Op::MakeMap | Op::MakeClosure => 2,
            Op::DollarSplice => 2,

            // i16 operand (2 bytes)
            Op::Jump
            | Op::JumpIfFalse
            | Op::JumpIfTrue
            | Op::JumpIfFalseKeep
            | Op::JumpIfTrueKeep => 2,

            // u8 operand (1 byte)
            Op::Call => 1,

            // u16 + u8 operands (3 bytes)
            Op::CallMethod | Op::CallBuiltin | Op::CallMacroDef | Op::NewObject => 3,

            // u16 + u16 + u8 operands (5 bytes)
            Op::CallStatic => 5,
        }
    }
}

/// Bytecode emitter — writes opcodes and operands into a `Vec<u8>`.
pub struct Emitter {
    pub code: Vec<u8>,
}

impl Emitter {
    pub fn new() -> Self {
        Self { code: Vec::new() }
    }

    /// Current byte offset (for jump patching).
    pub fn offset(&self) -> usize {
        self.code.len()
    }

    /// Emit an opcode with no operands.
    pub fn emit_op(&mut self, op: Op) {
        debug_assert_eq!(op.operand_size(), 0, "{} requires operands", op.name());
        self.code.push(op as u8);
    }

    /// Emit an opcode with a u16 operand.
    pub fn emit_u16(&mut self, op: Op, val: u16) {
        self.code.push(op as u8);
        self.code.extend_from_slice(&val.to_le_bytes());
    }

    /// Emit an opcode with an i16 operand (jumps).
    pub fn emit_i16(&mut self, op: Op, val: i16) {
        self.code.push(op as u8);
        self.code.extend_from_slice(&val.to_le_bytes());
    }

    /// Emit an opcode with a u8 operand.
    pub fn emit_u8(&mut self, op: Op, val: u8) {
        self.code.push(op as u8);
        self.code.push(val);
    }

    /// Emit an opcode with u16 + u8 operands.
    pub fn emit_u16_u8(&mut self, op: Op, a: u16, b: u8) {
        self.code.push(op as u8);
        self.code.extend_from_slice(&a.to_le_bytes());
        self.code.push(b);
    }

    /// Emit an opcode with u16 + u16 operands (SetFieldLocal).
    pub fn emit_u16_u16(&mut self, op: Op, a: u16, b: u16) {
        self.code.push(op as u8);
        self.code.extend_from_slice(&a.to_le_bytes());
        self.code.extend_from_slice(&b.to_le_bytes());
    }

    /// Emit an opcode with u16 + u16 + u8 operands (CallStatic).
    pub fn emit_u16_u16_u8(&mut self, op: Op, a: u16, b: u16, c: u8) {
        self.code.push(op as u8);
        self.code.extend_from_slice(&a.to_le_bytes());
        self.code.extend_from_slice(&b.to_le_bytes());
        self.code.push(c);
    }

    /// Emit a jump instruction with a placeholder offset (0).
    /// Returns the byte offset of the i16 operand for later patching.
    pub fn emit_jump(&mut self, op: Op) -> usize {
        self.code.push(op as u8);
        let patch_offset = self.code.len();
        self.code.extend_from_slice(&0i16.to_le_bytes());
        patch_offset
    }

    /// Patch a previously emitted jump's i16 offset.
    /// The offset is relative to the byte AFTER the operand (i.e., next instruction).
    pub fn patch_jump(&mut self, patch_offset: usize) {
        let target = self.code.len();
        let base = patch_offset + 2; // past the i16 operand
        let offset = (target as isize - base as isize) as i16;
        self.code[patch_offset..patch_offset + 2].copy_from_slice(&offset.to_le_bytes());
    }

    /// Emit a backward jump to `target_offset`.
    pub fn emit_loop(&mut self, op: Op, target_offset: usize) {
        self.code.push(op as u8);
        let base = self.code.len() + 2; // past the i16 we're about to write
        let offset = (target_offset as isize - base as isize) as i16;
        self.code.extend_from_slice(&offset.to_le_bytes());
    }
}

/// Bytecode reader — decodes opcodes and operands from a `&[u8]` slice.
pub struct Reader<'a> {
    code: &'a [u8],
    pub ip: usize,
}

impl<'a> Reader<'a> {
    pub fn new(code: &'a [u8]) -> Self {
        Self { code, ip: 0 }
    }

    /// Read the next opcode byte and advance IP.
    pub fn read_op(&mut self) -> Option<Op> {
        if self.ip >= self.code.len() {
            return None;
        }
        let byte = self.code[self.ip];
        self.ip += 1;
        Op::from_byte(byte)
    }

    /// Read a u16 operand (little-endian) and advance IP.
    pub fn read_u16(&mut self) -> u16 {
        let lo = self.code[self.ip] as u16;
        let hi = self.code[self.ip + 1] as u16;
        self.ip += 2;
        lo | (hi << 8)
    }

    /// Read an i16 operand (little-endian) and advance IP.
    pub fn read_i16(&mut self) -> i16 {
        let bytes = [self.code[self.ip], self.code[self.ip + 1]];
        self.ip += 2;
        i16::from_le_bytes(bytes)
    }

    /// Read a u8 operand and advance IP.
    pub fn read_u8(&mut self) -> u8 {
        let val = self.code[self.ip];
        self.ip += 1;
        val
    }

    /// Apply a signed offset to the IP (for jumps).
    pub fn jump(&mut self, offset: i16) {
        self.ip = (self.ip as isize + offset as isize) as usize;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_opcodes_roundtrip() {
        // Verify every Op variant can be decoded from its byte representation
        let all_ops = [
            Op::Const,
            Op::PushNull,
            Op::PushTrue,
            Op::PushFalse,
            Op::PushInt0,
            Op::PushInt1,
            Op::Pop,
            Op::LoadLocal,
            Op::StoreLocal,
            Op::DefineLocal,
            Op::LoadUpvalue,
            Op::Add,
            Op::Sub,
            Op::Mul,
            Op::Div,
            Op::Mod,
            Op::Eq,
            Op::NotEq,
            Op::Lt,
            Op::Le,
            Op::Gt,
            Op::Ge,
            Op::BitAnd,
            Op::BitOr,
            Op::BitXor,
            Op::Shl,
            Op::Shr,
            Op::Ushr,
            Op::NullCoal,
            Op::Not,
            Op::Neg,
            Op::BitNot,
            Op::Incr,
            Op::Decr,
            Op::Jump,
            Op::JumpIfFalse,
            Op::JumpIfTrue,
            Op::JumpIfFalseKeep,
            Op::JumpIfTrueKeep,
            Op::Call,
            Op::CallMethod,
            Op::CallStatic,
            Op::CallBuiltin,
            Op::CallMacroDef,
            Op::Return,
            Op::ReturnNull,
            Op::GetField,
            Op::SetField,
            Op::GetFieldOpt,
            Op::GetIndex,
            Op::SetIndex,
            Op::SetFieldLocal,
            Op::MakeArray,
            Op::MakeObject,
            Op::MakeMap,
            Op::MakeClosure,
            Op::NewObject,
            Op::Reify,
            Op::DollarSplice,
            Op::MacroWrap,
            Op::Dup,
            Op::Swap,
        ];

        for &op in &all_ops {
            let byte = op as u8;
            let decoded = Op::from_byte(byte)
                .unwrap_or_else(|| panic!("failed to decode {:?} (0x{:02x})", op, byte));
            assert_eq!(decoded, op, "roundtrip failed for {:?}", op);
        }

        // Verify we have the expected count
        assert_eq!(all_ops.len(), 62, "expected 62 opcodes");
    }

    #[test]
    fn test_emitter_simple_ops() {
        let mut e = Emitter::new();
        e.emit_op(Op::PushNull);
        e.emit_op(Op::PushTrue);
        e.emit_op(Op::Add);
        e.emit_op(Op::Return);

        assert_eq!(e.code, vec![0x01, 0x02, 0x10, 0x3D]);
    }

    #[test]
    fn test_emitter_u16_operand() {
        let mut e = Emitter::new();
        e.emit_u16(Op::Const, 258); // 0x0102 LE = [0x02, 0x01]

        assert_eq!(e.code, vec![0x00, 0x02, 0x01]);
    }

    #[test]
    fn test_emitter_jump_patching() {
        let mut e = Emitter::new();
        e.emit_op(Op::PushTrue); // offset 0
        let patch = e.emit_jump(Op::JumpIfFalse); // offset 1, operand at 2-3
        e.emit_op(Op::PushInt0); // offset 4
        e.emit_op(Op::Return); // offset 5
        e.patch_jump(patch); // target = 6, base = 4, offset = 2
        e.emit_op(Op::PushInt1); // offset 6
        e.emit_op(Op::Return); // offset 7

        let mut r = Reader::new(&e.code);

        // PushTrue
        assert_eq!(r.read_op(), Some(Op::PushTrue));
        // JumpIfFalse with offset 2
        assert_eq!(r.read_op(), Some(Op::JumpIfFalse));
        let offset = r.read_i16();
        assert_eq!(offset, 2);
        // PushInt0
        assert_eq!(r.read_op(), Some(Op::PushInt0));
        // Return
        assert_eq!(r.read_op(), Some(Op::Return));
        // PushInt1
        assert_eq!(r.read_op(), Some(Op::PushInt1));
        // Return
        assert_eq!(r.read_op(), Some(Op::Return));
    }

    #[test]
    fn test_emitter_backward_loop() {
        let mut e = Emitter::new();
        let loop_start = e.offset(); // 0
        e.emit_op(Op::PushTrue); // offset 0, size 1
        e.emit_i16(Op::JumpIfFalse, 10); // offset 1, size 3 (opcode + i16)
        e.emit_op(Op::Pop); // offset 4, size 1
        e.emit_loop(Op::Jump, loop_start); // offset 5, target=0, base=8, offset=-8

        let mut r = Reader::new(&e.code);
        r.ip = 5; // go to the backward Jump
        assert_eq!(r.read_op(), Some(Op::Jump));
        let offset = r.read_i16();
        assert_eq!(offset, -8);
        r.jump(offset);
        assert_eq!(r.ip, 0);
    }

    #[test]
    fn test_emitter_call_static() {
        let mut e = Emitter::new();
        e.emit_u16_u16_u8(Op::CallStatic, 100, 200, 3);

        assert_eq!(e.code.len(), 6); // 1 opcode + 2 + 2 + 1
        let mut r = Reader::new(&e.code);
        assert_eq!(r.read_op(), Some(Op::CallStatic));
        assert_eq!(r.read_u16(), 100);
        assert_eq!(r.read_u16(), 200);
        assert_eq!(r.read_u8(), 3);
    }

    #[test]
    fn test_reader_sequential() {
        let mut e = Emitter::new();
        e.emit_u16(Op::Const, 42);
        e.emit_op(Op::Dup);
        e.emit_op(Op::Add);
        e.emit_u16(Op::StoreLocal, 0);
        e.emit_op(Op::Return);

        let mut r = Reader::new(&e.code);
        assert_eq!(r.read_op(), Some(Op::Const));
        assert_eq!(r.read_u16(), 42);
        assert_eq!(r.read_op(), Some(Op::Dup));
        assert_eq!(r.read_op(), Some(Op::Add));
        assert_eq!(r.read_op(), Some(Op::StoreLocal));
        assert_eq!(r.read_u16(), 0);
        assert_eq!(r.read_op(), Some(Op::Return));
        assert_eq!(r.read_op(), None);
    }

    #[test]
    fn test_operand_sizes() {
        // Verify operand_size matches the emitter's encoding
        assert_eq!(Op::PushNull.operand_size(), 0);
        assert_eq!(Op::Add.operand_size(), 0);
        assert_eq!(Op::Return.operand_size(), 0);
        assert_eq!(Op::Const.operand_size(), 2);
        assert_eq!(Op::LoadLocal.operand_size(), 2);
        assert_eq!(Op::Jump.operand_size(), 2);
        assert_eq!(Op::Call.operand_size(), 1);
        assert_eq!(Op::CallMethod.operand_size(), 3);
        assert_eq!(Op::CallStatic.operand_size(), 5);
        assert_eq!(Op::NewObject.operand_size(), 3);
    }

    #[test]
    fn test_unknown_opcode() {
        assert_eq!(Op::from_byte(0xFF), None);
        assert_eq!(Op::from_byte(0x0B), None); // gap between LoadUpvalue and Add
        assert_eq!(Op::from_byte(0x60), None); // beyond defined range
    }
}
