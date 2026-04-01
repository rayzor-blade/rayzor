//! Stack-based virtual machine for executing compiled macro bytecode.
//!
//! The VM executes `Chunk` bytecode produced by the `BytecodeCompiler`.
//! It uses a flat operand stack with indexed local variable access (O(1)
//! instead of the tree-walker's O(n) scope chain scan).

use super::super::value::{MacroFunction, MacroParam, MacroValue};
use super::chunk::Chunk;
use super::opcode::{Op, Reader};
use parser::Span;
use std::collections::BTreeMap;
use std::sync::Arc;

/// Compiled class metadata for VM dispatch.
#[derive(Debug, Clone)]
pub struct CompiledClassInfo {
    /// Constructor chunk (slot 0 = this, returns this).
    pub constructor: Option<Arc<Chunk>>,
    /// Instance methods: name → chunk (slot 0 = this).
    pub instance_methods: BTreeMap<String, Arc<Chunk>>,
    /// Static methods: name → chunk.
    pub static_methods: BTreeMap<String, Arc<Chunk>>,
    /// Instance variable names and their default values.
    pub instance_vars: Vec<(String, MacroValue)>,
}

/// Error during VM execution.
#[derive(Debug)]
pub struct VmError {
    pub message: String,
    pub span: Option<Span>,
}

impl VmError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            span: None,
        }
    }

    fn with_span(message: impl Into<String>, span: Span) -> Self {
        Self {
            message: message.into(),
            span: Some(span),
        }
    }
}

impl std::fmt::Display for VmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

/// A call frame in the VM's call stack.
struct CallFrame {
    /// The chunk being executed.
    chunk: Arc<Chunk>,
    /// Instruction pointer (byte offset into chunk.code).
    ip: usize,
    /// Base pointer: index into the VM stack where this frame's locals start.
    bp: usize,
}

/// The macro bytecode VM.
pub struct MacroVm {
    /// Operand stack.
    stack: Vec<MacroValue>,
    /// Call frame stack.
    frames: Vec<CallFrame>,
    /// Trace output (from trace() calls).
    pub trace_output: Vec<String>,
    /// Compiled class data for constructor/method dispatch.
    class_chunks: BTreeMap<String, CompiledClassInfo>,
}

impl MacroVm {
    pub fn new() -> Self {
        Self {
            stack: Vec::with_capacity(256),
            frames: Vec::with_capacity(16),
            trace_output: Vec::new(),
            class_chunks: BTreeMap::new(),
        }
    }

    /// Set compiled class data for class-aware dispatch.
    pub fn set_class_chunks(&mut self, chunks: BTreeMap<String, CompiledClassInfo>) {
        self.class_chunks = chunks;
    }

    /// Execute a compiled chunk with the given arguments.
    /// Returns the result value.
    pub fn execute(
        &mut self,
        chunk: Arc<Chunk>,
        args: Vec<MacroValue>,
    ) -> Result<MacroValue, VmError> {
        self.stack.clear();
        self.frames.clear();
        self.trace_output.clear();

        let bp = 0;

        // Pre-allocate local slots and fill with args/Null
        for i in 0..chunk.local_count as usize {
            if i < args.len() {
                self.stack.push(args[i].clone());
            } else {
                self.stack.push(MacroValue::Null);
            }
        }

        self.frames.push(CallFrame { chunk, ip: 0, bp });

        self.run()
    }

    /// Main execution loop.
    fn run(&mut self) -> Result<MacroValue, VmError> {
        loop {
            let frame_idx = self.frames.len() - 1;
            let frame = &mut self.frames[frame_idx];
            let chunk = &frame.chunk;

            if frame.ip >= chunk.code.len() {
                // End of bytecode — implicit return Null
                return Ok(MacroValue::Null);
            }

            let op_byte = chunk.code[frame.ip];
            let op = match Op::from_byte(op_byte) {
                Some(op) => op,
                None => {
                    let span = chunk.span_at(frame.ip);
                    return Err(VmError {
                        message: format!("unknown opcode 0x{:02x} at offset {}", op_byte, frame.ip),
                        span,
                    });
                }
            };

            // Advance past the opcode byte
            frame.ip += 1;

            match op {
                // === Constants & Stack ===
                Op::Const => {
                    let idx = self.read_u16(frame_idx);
                    let val = self.frames[frame_idx].chunk.constants[idx as usize].clone();
                    self.stack.push(val);
                }
                Op::PushNull => self.stack.push(MacroValue::Null),
                Op::PushTrue => self.stack.push(MacroValue::Bool(true)),
                Op::PushFalse => self.stack.push(MacroValue::Bool(false)),
                Op::PushInt0 => self.stack.push(MacroValue::Int(0)),
                Op::PushInt1 => self.stack.push(MacroValue::Int(1)),
                Op::Pop => {
                    self.stack.pop();
                }

                // === Local Variables ===
                Op::LoadLocal => {
                    let slot = self.read_u16(frame_idx);
                    let bp = self.frames[frame_idx].bp;
                    let val = self.stack[bp + slot as usize].clone();
                    self.stack.push(val);
                }
                Op::StoreLocal => {
                    let slot = self.read_u16(frame_idx);
                    let bp = self.frames[frame_idx].bp;
                    let val = self.stack.last().cloned().unwrap_or(MacroValue::Null);
                    self.stack[bp + slot as usize] = val;
                }
                Op::DefineLocal => {
                    let slot = self.read_u16(frame_idx);
                    let bp = self.frames[frame_idx].bp;
                    let val = self.stack.last().cloned().unwrap_or(MacroValue::Null);
                    // Ensure slot exists
                    let idx = bp + slot as usize;
                    if idx < self.stack.len() {
                        self.stack[idx] = val;
                    } else {
                        // Extend stack to fit
                        while self.stack.len() <= idx {
                            self.stack.push(MacroValue::Null);
                        }
                        self.stack[idx] = val;
                    }
                }
                Op::LoadUpvalue => {
                    let _slot = self.read_u16(frame_idx);
                    // TODO: upvalue support for closures
                    self.stack.push(MacroValue::Null);
                }

                // === Binary Ops ===
                Op::Add => self.binary_op(|a, b| vm_add(a, b))?,
                Op::Sub => self.binary_op(|a, b| vm_sub(a, b))?,
                Op::Mul => self.binary_op(|a, b| vm_mul(a, b))?,
                Op::Div => self.binary_op(|a, b| vm_div(a, b))?,
                Op::Mod => self.binary_op(|a, b| vm_mod(a, b))?,
                Op::Eq => self.binary_op(|a, b| Ok(MacroValue::Bool(a == b)))?,
                Op::NotEq => self.binary_op(|a, b| Ok(MacroValue::Bool(a != b)))?,
                Op::Lt => self.binary_op(|a, b| vm_compare(a, b, |o| o.is_lt()))?,
                Op::Le => self.binary_op(|a, b| vm_compare(a, b, |o| o.is_le()))?,
                Op::Gt => self.binary_op(|a, b| vm_compare(a, b, |o| o.is_gt()))?,
                Op::Ge => self.binary_op(|a, b| vm_compare(a, b, |o| o.is_ge()))?,
                Op::BitAnd => self.int_binary_op(|a, b| a & b)?,
                Op::BitOr => self.int_binary_op(|a, b| a | b)?,
                Op::BitXor => self.int_binary_op(|a, b| a ^ b)?,
                Op::Shl => self.int_binary_op(|a, b| a << (b & 31))?,
                Op::Shr => self.int_binary_op(|a, b| a >> (b & 31))?,
                Op::Ushr => self.int_binary_op(|a, b| ((a as u64) >> ((b as u64) & 31)) as i64)?,
                Op::NullCoal => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    if matches!(a, MacroValue::Null) {
                        self.stack.push(b);
                    } else {
                        self.stack.push(a);
                    }
                }

                // === Unary Ops ===
                Op::Not => {
                    let val = self.pop()?;
                    self.stack.push(MacroValue::Bool(!val.is_truthy()));
                }
                Op::Neg => {
                    let val = self.pop()?;
                    match val {
                        MacroValue::Int(i) => self.stack.push(MacroValue::Int(-i)),
                        MacroValue::Float(f) => self.stack.push(MacroValue::Float(-f)),
                        _ => {
                            return Err(
                                self.error(frame_idx, format!("cannot negate {}", val.type_name()))
                            )
                        }
                    }
                }
                Op::BitNot => {
                    let val = self.pop()?;
                    match val {
                        MacroValue::Int(i) => self.stack.push(MacroValue::Int(!i)),
                        _ => {
                            return Err(self
                                .error(frame_idx, format!("cannot bit-not {}", val.type_name())))
                        }
                    }
                }
                Op::Incr => {
                    let val = self.pop()?;
                    match val {
                        MacroValue::Int(i) => self.stack.push(MacroValue::Int(i + 1)),
                        MacroValue::Float(f) => self.stack.push(MacroValue::Float(f + 1.0)),
                        _ => {
                            return Err(self
                                .error(frame_idx, format!("cannot increment {}", val.type_name())))
                        }
                    }
                }
                Op::Decr => {
                    let val = self.pop()?;
                    match val {
                        MacroValue::Int(i) => self.stack.push(MacroValue::Int(i - 1)),
                        MacroValue::Float(f) => self.stack.push(MacroValue::Float(f - 1.0)),
                        _ => {
                            return Err(self
                                .error(frame_idx, format!("cannot decrement {}", val.type_name())))
                        }
                    }
                }

                // === Control Flow ===
                Op::Jump => {
                    let offset = self.read_i16(frame_idx);
                    let frame = &mut self.frames[frame_idx];
                    frame.ip = (frame.ip as isize + offset as isize) as usize;
                }
                Op::JumpIfFalse => {
                    let offset = self.read_i16(frame_idx);
                    let val = self.pop()?;
                    if !val.is_truthy() {
                        let frame = &mut self.frames[frame_idx];
                        frame.ip = (frame.ip as isize + offset as isize) as usize;
                    }
                }
                Op::JumpIfTrue => {
                    let offset = self.read_i16(frame_idx);
                    let val = self.pop()?;
                    if val.is_truthy() {
                        let frame = &mut self.frames[frame_idx];
                        frame.ip = (frame.ip as isize + offset as isize) as usize;
                    }
                }
                Op::JumpIfFalseKeep => {
                    let offset = self.read_i16(frame_idx);
                    let val = self.stack.last().cloned().unwrap_or(MacroValue::Null);
                    if !val.is_truthy() {
                        let frame = &mut self.frames[frame_idx];
                        frame.ip = (frame.ip as isize + offset as isize) as usize;
                    }
                }
                Op::JumpIfTrueKeep => {
                    let offset = self.read_i16(frame_idx);
                    let val = self.stack.last().cloned().unwrap_or(MacroValue::Null);
                    if val.is_truthy() {
                        let frame = &mut self.frames[frame_idx];
                        frame.ip = (frame.ip as isize + offset as isize) as usize;
                    }
                }

                // === Function Calls ===
                Op::Call => {
                    let nargs = self.read_u8(frame_idx);
                    let mut args = Vec::with_capacity(nargs as usize);
                    for _ in 0..nargs {
                        args.push(self.pop()?);
                    }
                    args.reverse();
                    let callee = self.pop()?;

                    match callee {
                        MacroValue::Function(func) => {
                            let result = self.call_function(&func, args)?;
                            self.stack.push(result);
                        }
                        _ => {
                            return Err(self
                                .error(frame_idx, format!("cannot call {}", callee.type_name())));
                        }
                    }
                }
                Op::CallMethod => {
                    let name_idx = self.read_u16(frame_idx);
                    let nargs = self.read_u8(frame_idx);
                    let mut args = Vec::with_capacity(nargs as usize);
                    for _ in 0..nargs {
                        args.push(self.pop()?);
                    }
                    args.reverse();
                    let base = self.pop()?;

                    let method_name = self.get_string_constant(frame_idx, name_idx)?;

                    // Check for compiled class instance methods
                    let class_method = if let MacroValue::Object(ref obj) = base {
                        if let Some(MacroValue::String(ref type_name)) = obj.get("__type__") {
                            self.class_chunks
                                .get(type_name.as_ref())
                                .and_then(|ci| ci.instance_methods.get(&method_name))
                                .cloned()
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    if let Some(method_chunk) = class_method {
                        // Execute compiled instance method: this=base at slot 0
                        let bp = self.stack.len();
                        self.stack.push(base); // slot 0 = this
                        for (i, param) in method_chunk.params.iter().enumerate() {
                            let val = args.get(i).cloned().unwrap_or_else(|| {
                                if param.optional {
                                    MacroValue::Null
                                } else {
                                    MacroValue::Null
                                }
                            });
                            self.stack.push(val);
                        }
                        let filled = 1 + method_chunk.params.len();
                        for _ in filled..method_chunk.local_count as usize {
                            self.stack.push(MacroValue::Null);
                        }
                        self.frames.push(CallFrame {
                            chunk: method_chunk,
                            ip: 0,
                            bp,
                        });
                        // Method will run and Return the result
                    } else {
                        // Fall back to built-in method dispatch
                        let result = self.call_method(base, &method_name, args)?;
                        self.stack.push(result);
                    }
                }
                Op::CallStatic => {
                    let class_idx = self.read_u16(frame_idx);
                    let method_idx = self.read_u16(frame_idx);
                    let nargs = self.read_u8(frame_idx);
                    let mut args = Vec::with_capacity(nargs as usize);
                    for _ in 0..nargs {
                        args.push(self.pop()?);
                    }
                    args.reverse();

                    let class_name = self.get_string_constant(frame_idx, class_idx)?;
                    let method_name = self.get_string_constant(frame_idx, method_idx)?;

                    // Check for compiled class static methods
                    let static_method = self
                        .class_chunks
                        .get(&class_name)
                        .and_then(|ci| ci.static_methods.get(&method_name))
                        .cloned();

                    if let Some(method_chunk) = static_method {
                        // Execute compiled static method
                        let bp = self.stack.len();
                        for (i, param) in method_chunk.params.iter().enumerate() {
                            let val = args.get(i).cloned().unwrap_or_else(|| {
                                if param.optional {
                                    MacroValue::Null
                                } else {
                                    MacroValue::Null
                                }
                            });
                            self.stack.push(val);
                        }
                        let filled = method_chunk.params.len();
                        for _ in filled..method_chunk.local_count as usize {
                            self.stack.push(MacroValue::Null);
                        }
                        self.frames.push(CallFrame {
                            chunk: method_chunk,
                            ip: 0,
                            bp,
                        });
                        // Static method will run and Return the result
                    } else {
                        // Fall back to built-in static dispatch
                        let result = self.call_static(&class_name, &method_name, args)?;
                        self.stack.push(result);
                    }
                }
                Op::CallBuiltin => {
                    let builtin_id = self.read_u16(frame_idx);
                    let nargs = self.read_u8(frame_idx);
                    let mut args = Vec::with_capacity(nargs as usize);
                    for _ in 0..nargs {
                        args.push(self.pop()?);
                    }
                    args.reverse();

                    if builtin_id == 0xFFFF {
                        // Throw sentinel
                        let msg = if args.is_empty() {
                            MacroValue::Null
                        } else {
                            args.into_iter().next().unwrap()
                        };
                        return Err(VmError::new(format!("throw: {}", msg.to_display_string())));
                    }

                    // Other builtins — TODO
                    self.stack.push(MacroValue::Null);
                }
                Op::CallMacroDef => {
                    let _macro_id = self.read_u16(frame_idx);
                    let nargs = self.read_u8(frame_idx);
                    for _ in 0..nargs {
                        self.pop()?;
                    }
                    // TODO: macro def calls
                    self.stack.push(MacroValue::Null);
                }
                Op::Return => {
                    let val = self.pop()?;
                    if self.frames.len() <= 1 {
                        return Ok(val);
                    }
                    let frame = self.frames.pop().unwrap();
                    // Truncate stack back to caller's level
                    self.stack.truncate(frame.bp);
                    self.stack.push(val);
                }
                Op::ReturnNull => {
                    if self.frames.len() <= 1 {
                        return Ok(MacroValue::Null);
                    }
                    let frame = self.frames.pop().unwrap();
                    self.stack.truncate(frame.bp);
                    self.stack.push(MacroValue::Null);
                }

                // === Field & Index ===
                Op::GetField => {
                    let name_idx = self.read_u16(frame_idx);
                    let base = self.pop()?;
                    let field_name = self.get_string_constant(frame_idx, name_idx)?;
                    let result = self.get_field(base, &field_name)?;
                    self.stack.push(result);
                }
                Op::SetField => {
                    let name_idx = self.read_u16(frame_idx);
                    let val = self.pop()?;
                    let mut base = self.pop()?;
                    let field_name = self.get_string_constant(frame_idx, name_idx)?;
                    self.set_field(&mut base, &field_name, val.clone())?;
                    self.stack.push(val);
                }
                Op::SetFieldLocal => {
                    let slot = self.read_u16(frame_idx);
                    let name_idx = self.read_u16(frame_idx);
                    let val = self.pop()?;
                    let field_name = self.get_string_constant(frame_idx, name_idx)?;
                    let bp = self.frames[frame_idx].bp;
                    let idx = bp + slot as usize;
                    let is_object = matches!(&self.stack[idx], MacroValue::Object(_));
                    if is_object {
                        if let MacroValue::Object(obj) = &mut self.stack[idx] {
                            Arc::make_mut(obj).insert(field_name, val.clone());
                        }
                    } else {
                        let type_name = self.stack[idx].type_name().to_string();
                        return Err(
                            self.error(frame_idx, format!("cannot set field on {}", type_name))
                        );
                    }
                    self.stack.push(val);
                }
                Op::GetFieldOpt => {
                    let name_idx = self.read_u16(frame_idx);
                    let base = self.pop()?;
                    if matches!(base, MacroValue::Null) {
                        self.stack.push(MacroValue::Null);
                    } else {
                        let field_name = self.get_string_constant(frame_idx, name_idx)?;
                        let result = self.get_field(base, &field_name)?;
                        self.stack.push(result);
                    }
                }
                Op::GetIndex => {
                    let index = self.pop()?;
                    let base = self.pop()?;
                    let result = self.get_index(base, index)?;
                    self.stack.push(result);
                }
                Op::SetIndex => {
                    let val = self.pop()?;
                    let index = self.pop()?;
                    let mut base = self.pop()?;
                    self.set_index(&mut base, index, val.clone())?;
                    self.stack.push(val);
                }

                // === Collection Construction ===
                Op::MakeArray => {
                    let count = self.read_u16(frame_idx);
                    let start = self.stack.len() - count as usize;
                    let items: Vec<MacroValue> = self.stack.drain(start..).collect();
                    self.stack.push(MacroValue::Array(Arc::new(items)));
                }
                Op::MakeObject => {
                    let count = self.read_u16(frame_idx);
                    let mut map = BTreeMap::new();
                    // Stack has N pairs of (name_string, value)
                    let start = self.stack.len() - (count as usize * 2);
                    let items: Vec<MacroValue> = self.stack.drain(start..).collect();
                    for pair in items.chunks(2) {
                        if let MacroValue::String(name) = &pair[0] {
                            map.insert(name.to_string(), pair[1].clone());
                        }
                    }
                    self.stack.push(MacroValue::Object(Arc::new(map)));
                }
                Op::MakeMap => {
                    let count = self.read_u16(frame_idx);
                    let mut map = BTreeMap::new();
                    let start = self.stack.len() - (count as usize * 2);
                    let items: Vec<MacroValue> = self.stack.drain(start..).collect();
                    for pair in items.chunks(2) {
                        let key = pair[0].to_display_string();
                        map.insert(key, pair[1].clone());
                    }
                    self.stack.push(MacroValue::Object(Arc::new(map)));
                }
                Op::MakeClosure => {
                    let chunk_idx = self.read_u16(frame_idx);
                    let sub_chunk =
                        self.frames[frame_idx].chunk.closures[chunk_idx as usize].clone();
                    let params: Vec<MacroParam> = sub_chunk
                        .params
                        .iter()
                        .map(|p| {
                            let name = sub_chunk
                                .local_names
                                .iter()
                                .find(|(slot, _)| *slot == p.slot)
                                .map(|(_, n)| n.clone())
                                .unwrap_or_default();
                            MacroParam {
                                name,
                                optional: p.optional,
                                rest: false,
                                default_value: None,
                            }
                        })
                        .collect();

                    // Capture current environment for the closure
                    let bp = self.frames[frame_idx].bp;
                    let mut captures = BTreeMap::new();
                    for (slot, name) in &self.frames[frame_idx].chunk.local_names {
                        let idx = bp + *slot as usize;
                        if idx < self.stack.len() {
                            captures.insert(name.clone(), self.stack[idx].clone());
                        }
                    }

                    let func = MacroFunction {
                        name: sub_chunk.name.clone(),
                        params,
                        body: Arc::new(parser::Expr {
                            kind: parser::ExprKind::Null,
                            span: Span { start: 0, end: 0 },
                        }),
                        captures,
                    };
                    self.stack.push(MacroValue::Function(Arc::new(func)));
                }
                Op::NewObject => {
                    let name_idx = self.read_u16(frame_idx);
                    let nargs = self.read_u8(frame_idx);
                    let mut args = Vec::with_capacity(nargs as usize);
                    for _ in 0..nargs {
                        args.push(self.pop()?);
                    }
                    args.reverse();

                    let class_name = self.get_string_constant(frame_idx, name_idx)?;

                    // Check if we have compiled class data
                    if let Some(class_info) = self.class_chunks.get(&class_name) {
                        // Create object with __type__ and instance var defaults
                        let mut map = BTreeMap::new();
                        map.insert(
                            "__type__".to_string(),
                            MacroValue::String(Arc::from(class_name.as_str())),
                        );
                        for (name, default_val) in &class_info.instance_vars {
                            map.insert(name.clone(), default_val.clone());
                        }
                        let obj = MacroValue::Object(Arc::new(map));

                        if let Some(ctor_chunk) = class_info.constructor.clone() {
                            // Execute constructor: push frame with this=obj at slot 0
                            let bp = self.stack.len();
                            // Pre-allocate locals: slot 0 = this, then params, then remaining
                            self.stack.push(obj); // slot 0 = this
                            for (i, param) in ctor_chunk.params.iter().enumerate() {
                                let val = args.get(i).cloned().unwrap_or_else(|| {
                                    if param.optional {
                                        MacroValue::Null
                                    } else {
                                        MacroValue::Null
                                    }
                                });
                                self.stack.push(val);
                            }
                            // Fill remaining local slots with Null
                            let filled = 1 + ctor_chunk.params.len();
                            for _ in filled..ctor_chunk.local_count as usize {
                                self.stack.push(MacroValue::Null);
                            }
                            self.frames.push(CallFrame {
                                chunk: ctor_chunk,
                                ip: 0,
                                bp,
                            });
                            // Constructor will run in the main loop and Return `this`
                        } else {
                            // No constructor — push object directly
                            self.stack.push(obj);
                        }
                    } else {
                        // Fallback: generic object construction
                        let mut map = BTreeMap::new();
                        map.insert(
                            "__type__".to_string(),
                            MacroValue::String(Arc::from(class_name.as_str())),
                        );
                        for (i, arg) in args.into_iter().enumerate() {
                            map.insert(format!("_{}", i), arg);
                        }
                        self.stack.push(MacroValue::Object(Arc::new(map)));
                    }
                }

                // === Reification & Macros ===
                Op::Reify | Op::DollarSplice | Op::MacroWrap => {
                    if op == Op::DollarSplice {
                        let _kind_idx = self.read_u16(frame_idx);
                    }
                    // TODO: reification support — for now, pass through
                }

                // === Misc ===
                Op::Dup => {
                    let val = self.stack.last().cloned().unwrap_or(MacroValue::Null);
                    self.stack.push(val);
                }
                Op::Swap => {
                    let len = self.stack.len();
                    if len >= 2 {
                        self.stack.swap(len - 1, len - 2);
                    }
                }
            }
        }
    }

    // === Helper methods ===

    fn read_u16(&mut self, frame_idx: usize) -> u16 {
        let frame = &mut self.frames[frame_idx];
        let lo = frame.chunk.code[frame.ip] as u16;
        let hi = frame.chunk.code[frame.ip + 1] as u16;
        frame.ip += 2;
        lo | (hi << 8)
    }

    fn read_i16(&mut self, frame_idx: usize) -> i16 {
        let frame = &mut self.frames[frame_idx];
        let bytes = [frame.chunk.code[frame.ip], frame.chunk.code[frame.ip + 1]];
        frame.ip += 2;
        i16::from_le_bytes(bytes)
    }

    fn read_u8(&mut self, frame_idx: usize) -> u8 {
        let frame = &mut self.frames[frame_idx];
        let val = frame.chunk.code[frame.ip];
        frame.ip += 1;
        val
    }

    fn pop(&mut self) -> Result<MacroValue, VmError> {
        self.stack
            .pop()
            .ok_or_else(|| VmError::new("stack underflow"))
    }

    fn error(&self, frame_idx: usize, message: String) -> VmError {
        let span = self
            .frames
            .get(frame_idx)
            .and_then(|f| f.chunk.span_at(f.ip.saturating_sub(1)));
        VmError { message, span }
    }

    fn get_string_constant(&self, frame_idx: usize, idx: u16) -> Result<String, VmError> {
        let chunk = &self.frames[frame_idx].chunk;
        match chunk.constants.get(idx as usize) {
            Some(MacroValue::String(s)) => Ok(s.to_string()),
            Some(other) => Err(VmError::new(format!(
                "expected string constant at index {}, got {}",
                idx,
                other.type_name()
            ))),
            None => Err(VmError::new(format!(
                "constant index {} out of bounds (pool size: {})",
                idx,
                chunk.constants.len()
            ))),
        }
    }

    /// Execute a binary operation.
    fn binary_op(
        &mut self,
        f: impl FnOnce(MacroValue, MacroValue) -> Result<MacroValue, VmError>,
    ) -> Result<(), VmError> {
        let b = self.pop()?;
        let a = self.pop()?;
        let result = f(a, b)?;
        self.stack.push(result);
        Ok(())
    }

    /// Execute an integer binary operation.
    fn int_binary_op(&mut self, f: impl FnOnce(i64, i64) -> i64) -> Result<(), VmError> {
        let b = self.pop()?;
        let a = self.pop()?;
        let ai = a
            .as_int()
            .ok_or_else(|| VmError::new(format!("expected Int, got {}", a.type_name())))?;
        let bi = b
            .as_int()
            .ok_or_else(|| VmError::new(format!("expected Int, got {}", b.type_name())))?;
        self.stack.push(MacroValue::Int(f(ai, bi)));
        Ok(())
    }

    /// Get a field from a value.
    fn get_field(&self, base: MacroValue, name: &str) -> Result<MacroValue, VmError> {
        match &base {
            MacroValue::Object(obj) => Ok(obj.get(name).cloned().unwrap_or(MacroValue::Null)),
            MacroValue::Array(arr) => match name {
                "length" => Ok(MacroValue::Int(arr.len() as i64)),
                _ => Ok(MacroValue::Null),
            },
            MacroValue::String(s) => match name {
                "length" => Ok(MacroValue::Int(s.len() as i64)),
                _ => Ok(MacroValue::Null),
            },
            MacroValue::Null => Err(VmError::new("cannot access field on null")),
            _ => Ok(MacroValue::Null),
        }
    }

    /// Set a field on a value.
    fn set_field(
        &mut self,
        base: &mut MacroValue,
        name: &str,
        value: MacroValue,
    ) -> Result<(), VmError> {
        match base {
            MacroValue::Object(obj) => {
                Arc::make_mut(obj).insert(name.to_string(), value);
                Ok(())
            }
            _ => Err(VmError::new(format!(
                "cannot set field on {}",
                base.type_name()
            ))),
        }
    }

    /// Get an element by index.
    fn get_index(&self, base: MacroValue, index: MacroValue) -> Result<MacroValue, VmError> {
        match (&base, &index) {
            (MacroValue::Array(arr), MacroValue::Int(i)) => {
                let idx = *i as usize;
                Ok(arr.get(idx).cloned().unwrap_or(MacroValue::Null))
            }
            (MacroValue::Object(obj), MacroValue::String(key)) => {
                Ok(obj.get(key.as_ref()).cloned().unwrap_or(MacroValue::Null))
            }
            (MacroValue::String(s), MacroValue::Int(i)) => {
                let idx = *i as usize;
                Ok(s.chars()
                    .nth(idx)
                    .map(|c| MacroValue::from_str(&c.to_string()))
                    .unwrap_or(MacroValue::Null))
            }
            _ => Err(VmError::new(format!(
                "cannot index {} with {}",
                base.type_name(),
                index.type_name()
            ))),
        }
    }

    /// Set an element by index.
    fn set_index(
        &mut self,
        base: &mut MacroValue,
        index: MacroValue,
        value: MacroValue,
    ) -> Result<(), VmError> {
        match (base, &index) {
            (MacroValue::Array(arr), MacroValue::Int(i)) => {
                let idx = *i as usize;
                let arr = Arc::make_mut(arr);
                while arr.len() <= idx {
                    arr.push(MacroValue::Null);
                }
                arr[idx] = value;
                Ok(())
            }
            (MacroValue::Object(obj), MacroValue::String(key)) => {
                Arc::make_mut(obj).insert(key.to_string(), value);
                Ok(())
            }
            (base_val, _) => Err(VmError::new(format!(
                "cannot set index on {}",
                base_val.type_name()
            ))),
        }
    }

    /// Call a function value.
    fn call_function(
        &mut self,
        _func: &MacroFunction,
        _args: Vec<MacroValue>,
    ) -> Result<MacroValue, VmError> {
        // TODO: full function call support with new frame
        // For now, return Null for non-bytecode functions
        Ok(MacroValue::Null)
    }

    /// Call a method on a value.
    fn call_method(
        &mut self,
        base: MacroValue,
        method: &str,
        args: Vec<MacroValue>,
    ) -> Result<MacroValue, VmError> {
        match &base {
            MacroValue::Array(arr) => self.call_array_method(arr, method, args),
            MacroValue::String(s) => self.call_string_method(s, method, args),
            MacroValue::Object(obj) => {
                // Check if the object has a function field with this name
                if let Some(MacroValue::Function(func)) = obj.get(method) {
                    return self.call_function(func, args);
                }
                Ok(MacroValue::Null)
            }
            _ => Err(VmError::new(format!(
                "cannot call method '{}' on {}",
                method,
                base.type_name()
            ))),
        }
    }

    /// Call a static method.
    fn call_static(
        &mut self,
        class: &str,
        method: &str,
        args: Vec<MacroValue>,
    ) -> Result<MacroValue, VmError> {
        match (class, method) {
            ("trace" | "Sys", "println") | (_, "trace")
                if method == "trace" || method == "println" =>
            {
                let msg = args
                    .iter()
                    .map(|a| a.to_display_string())
                    .collect::<Vec<_>>()
                    .join(",");
                self.trace_output.push(msg);
                Ok(MacroValue::Null)
            }
            ("Std", "string") => {
                if let Some(val) = args.first() {
                    Ok(MacroValue::from_string(val.to_display_string()))
                } else {
                    Ok(MacroValue::from_str("null"))
                }
            }
            ("Std", "parseInt") => {
                if let Some(MacroValue::String(s)) = args.first() {
                    match s.parse::<i64>() {
                        Ok(i) => Ok(MacroValue::Int(i)),
                        Err(_) => Ok(MacroValue::Null),
                    }
                } else {
                    Ok(MacroValue::Null)
                }
            }
            ("Std", "parseFloat") => {
                if let Some(MacroValue::String(s)) = args.first() {
                    match s.parse::<f64>() {
                        Ok(f) => Ok(MacroValue::Float(f)),
                        Err(_) => Ok(MacroValue::Null),
                    }
                } else {
                    Ok(MacroValue::Null)
                }
            }
            ("Math", "floor") => {
                if let Some(val) = args.first().and_then(|a| a.as_float()) {
                    Ok(MacroValue::Int(val.floor() as i64))
                } else {
                    Ok(MacroValue::Null)
                }
            }
            ("Math", "ceil") => {
                if let Some(val) = args.first().and_then(|a| a.as_float()) {
                    Ok(MacroValue::Int(val.ceil() as i64))
                } else {
                    Ok(MacroValue::Null)
                }
            }
            ("Math", "abs") => {
                if let Some(val) = args.first() {
                    match val {
                        MacroValue::Int(i) => Ok(MacroValue::Int(i.abs())),
                        MacroValue::Float(f) => Ok(MacroValue::Float(f.abs())),
                        _ => Ok(MacroValue::Null),
                    }
                } else {
                    Ok(MacroValue::Null)
                }
            }
            ("Math", "max") => {
                if args.len() >= 2 {
                    let a = args[0].as_float().unwrap_or(0.0);
                    let b = args[1].as_float().unwrap_or(0.0);
                    Ok(MacroValue::Float(a.max(b)))
                } else {
                    Ok(MacroValue::Null)
                }
            }
            ("Math", "min") => {
                if args.len() >= 2 {
                    let a = args[0].as_float().unwrap_or(0.0);
                    let b = args[1].as_float().unwrap_or(0.0);
                    Ok(MacroValue::Float(a.min(b)))
                } else {
                    Ok(MacroValue::Null)
                }
            }
            _ => {
                // Unknown static call — check if it's a bare trace() call pattern
                if class == "trace" {
                    let msg = args
                        .iter()
                        .map(|a| a.to_display_string())
                        .collect::<Vec<_>>()
                        .join(",");
                    self.trace_output.push(msg);
                    return Ok(MacroValue::Null);
                }
                Ok(MacroValue::Null)
            }
        }
    }

    /// Call an array method.
    fn call_array_method(
        &self,
        arr: &Arc<Vec<MacroValue>>,
        method: &str,
        args: Vec<MacroValue>,
    ) -> Result<MacroValue, VmError> {
        match method {
            "push" => {
                let mut new_arr = (**arr).clone();
                for arg in args {
                    new_arr.push(arg);
                }
                Ok(MacroValue::Array(Arc::new(new_arr)))
            }
            "pop" => {
                let mut new_arr = (**arr).clone();
                let val = new_arr.pop().unwrap_or(MacroValue::Null);
                Ok(val)
            }
            "length" => Ok(MacroValue::Int(arr.len() as i64)),
            "join" => {
                let sep = args
                    .first()
                    .and_then(|a| a.as_string().map(|s| s.to_string()))
                    .unwrap_or_else(|| ",".to_string());
                let result: Vec<String> = arr.iter().map(|v| v.to_display_string()).collect();
                Ok(MacroValue::from_string(result.join(&sep)))
            }
            "map" => {
                // map with a function — returns the array for now
                Ok(MacroValue::Array(arr.clone()))
            }
            "filter" => Ok(MacroValue::Array(arr.clone())),
            "indexOf" => {
                if let Some(target) = args.first() {
                    for (i, item) in arr.iter().enumerate() {
                        if item == target {
                            return Ok(MacroValue::Int(i as i64));
                        }
                    }
                    Ok(MacroValue::Int(-1))
                } else {
                    Ok(MacroValue::Int(-1))
                }
            }
            "contains" => {
                if let Some(target) = args.first() {
                    Ok(MacroValue::Bool(arr.iter().any(|item| item == target)))
                } else {
                    Ok(MacroValue::Bool(false))
                }
            }
            "slice" => {
                let start = args.first().and_then(|a| a.as_int()).unwrap_or(0) as usize;
                let end = args
                    .get(1)
                    .and_then(|a| a.as_int())
                    .map(|i| i as usize)
                    .unwrap_or(arr.len());
                let sliced: Vec<MacroValue> =
                    arr[start.min(arr.len())..end.min(arr.len())].to_vec();
                Ok(MacroValue::Array(Arc::new(sliced)))
            }
            "concat" => {
                let mut new_arr = (**arr).clone();
                if let Some(MacroValue::Array(other)) = args.first() {
                    new_arr.extend(other.iter().cloned());
                }
                Ok(MacroValue::Array(Arc::new(new_arr)))
            }
            "reverse" => {
                let mut new_arr = (**arr).clone();
                new_arr.reverse();
                Ok(MacroValue::Array(Arc::new(new_arr)))
            }
            "iterator" => {
                // Return the array itself as an iterator-like value
                Ok(MacroValue::Array(arr.clone()))
            }
            _ => Ok(MacroValue::Null),
        }
    }

    /// Call a string method.
    fn call_string_method(
        &self,
        s: &Arc<str>,
        method: &str,
        args: Vec<MacroValue>,
    ) -> Result<MacroValue, VmError> {
        match method {
            "length" => Ok(MacroValue::Int(s.len() as i64)),
            "charAt" => {
                let idx = args.first().and_then(|a| a.as_int()).unwrap_or(0) as usize;
                Ok(s.chars()
                    .nth(idx)
                    .map(|c| MacroValue::from_str(&c.to_string()))
                    .unwrap_or(MacroValue::from_str("")))
            }
            "indexOf" => {
                if let Some(MacroValue::String(needle)) = args.first() {
                    match s.find(needle.as_ref()) {
                        Some(pos) => Ok(MacroValue::Int(pos as i64)),
                        None => Ok(MacroValue::Int(-1)),
                    }
                } else {
                    Ok(MacroValue::Int(-1))
                }
            }
            "substring" | "substr" => {
                let start = args.first().and_then(|a| a.as_int()).unwrap_or(0) as usize;
                let end = args
                    .get(1)
                    .and_then(|a| a.as_int())
                    .map(|i| i as usize)
                    .unwrap_or(s.len());
                let result: String = s
                    .chars()
                    .skip(start)
                    .take(end.saturating_sub(start))
                    .collect();
                Ok(MacroValue::from_string(result))
            }
            "toLowerCase" => Ok(MacroValue::from_string(s.to_lowercase())),
            "toUpperCase" => Ok(MacroValue::from_string(s.to_uppercase())),
            "split" => {
                if let Some(MacroValue::String(sep)) = args.first() {
                    let parts: Vec<MacroValue> =
                        s.split(sep.as_ref()).map(MacroValue::from_str).collect();
                    Ok(MacroValue::Array(Arc::new(parts)))
                } else {
                    Ok(MacroValue::Array(Arc::new(vec![MacroValue::String(
                        s.clone(),
                    )])))
                }
            }
            "trim" => Ok(MacroValue::from_string(s.trim().to_string())),
            "startsWith" => {
                if let Some(MacroValue::String(prefix)) = args.first() {
                    Ok(MacroValue::Bool(s.starts_with(prefix.as_ref())))
                } else {
                    Ok(MacroValue::Bool(false))
                }
            }
            "endsWith" => {
                if let Some(MacroValue::String(suffix)) = args.first() {
                    Ok(MacroValue::Bool(s.ends_with(suffix.as_ref())))
                } else {
                    Ok(MacroValue::Bool(false))
                }
            }
            "replace" => {
                if args.len() >= 2 {
                    if let (Some(MacroValue::String(from)), Some(MacroValue::String(to))) =
                        (args.first(), args.get(1))
                    {
                        Ok(MacroValue::from_string(s.replacen(
                            from.as_ref(),
                            to.as_ref(),
                            1,
                        )))
                    } else {
                        Ok(MacroValue::String(s.clone()))
                    }
                } else {
                    Ok(MacroValue::String(s.clone()))
                }
            }
            _ => Ok(MacroValue::Null),
        }
    }
}

// === Arithmetic helpers ===

fn vm_add(a: MacroValue, b: MacroValue) -> Result<MacroValue, VmError> {
    match (&a, &b) {
        (MacroValue::Int(x), MacroValue::Int(y)) => Ok(MacroValue::Int(x + y)),
        (MacroValue::Float(x), MacroValue::Float(y)) => Ok(MacroValue::Float(x + y)),
        (MacroValue::Int(x), MacroValue::Float(y)) => Ok(MacroValue::Float(*x as f64 + y)),
        (MacroValue::Float(x), MacroValue::Int(y)) => Ok(MacroValue::Float(x + *y as f64)),
        // String concatenation
        (MacroValue::String(_), _) | (_, MacroValue::String(_)) => Ok(MacroValue::from_string(
            format!("{}{}", a.to_display_string(), b.to_display_string()),
        )),
        _ => Err(VmError::new(format!(
            "cannot add {} and {}",
            a.type_name(),
            b.type_name()
        ))),
    }
}

fn vm_sub(a: MacroValue, b: MacroValue) -> Result<MacroValue, VmError> {
    match (&a, &b) {
        (MacroValue::Int(x), MacroValue::Int(y)) => Ok(MacroValue::Int(x - y)),
        (MacroValue::Float(x), MacroValue::Float(y)) => Ok(MacroValue::Float(x - y)),
        (MacroValue::Int(x), MacroValue::Float(y)) => Ok(MacroValue::Float(*x as f64 - y)),
        (MacroValue::Float(x), MacroValue::Int(y)) => Ok(MacroValue::Float(x - *y as f64)),
        _ => Err(VmError::new(format!(
            "cannot subtract {} and {}",
            a.type_name(),
            b.type_name()
        ))),
    }
}

fn vm_mul(a: MacroValue, b: MacroValue) -> Result<MacroValue, VmError> {
    match (&a, &b) {
        (MacroValue::Int(x), MacroValue::Int(y)) => Ok(MacroValue::Int(x * y)),
        (MacroValue::Float(x), MacroValue::Float(y)) => Ok(MacroValue::Float(x * y)),
        (MacroValue::Int(x), MacroValue::Float(y)) => Ok(MacroValue::Float(*x as f64 * y)),
        (MacroValue::Float(x), MacroValue::Int(y)) => Ok(MacroValue::Float(x * *y as f64)),
        _ => Err(VmError::new(format!(
            "cannot multiply {} and {}",
            a.type_name(),
            b.type_name()
        ))),
    }
}

fn vm_div(a: MacroValue, b: MacroValue) -> Result<MacroValue, VmError> {
    match (&a, &b) {
        (MacroValue::Int(x), MacroValue::Int(y)) => {
            if *y == 0 {
                return Err(VmError::new("division by zero"));
            }
            Ok(MacroValue::Int(x / y))
        }
        (MacroValue::Float(x), MacroValue::Float(y)) => Ok(MacroValue::Float(x / y)),
        (MacroValue::Int(x), MacroValue::Float(y)) => Ok(MacroValue::Float(*x as f64 / y)),
        (MacroValue::Float(x), MacroValue::Int(y)) => Ok(MacroValue::Float(x / *y as f64)),
        _ => Err(VmError::new(format!(
            "cannot divide {} and {}",
            a.type_name(),
            b.type_name()
        ))),
    }
}

fn vm_mod(a: MacroValue, b: MacroValue) -> Result<MacroValue, VmError> {
    match (&a, &b) {
        (MacroValue::Int(x), MacroValue::Int(y)) => {
            if *y == 0 {
                return Err(VmError::new("modulo by zero"));
            }
            Ok(MacroValue::Int(x % y))
        }
        (MacroValue::Float(x), MacroValue::Float(y)) => Ok(MacroValue::Float(x % y)),
        (MacroValue::Int(x), MacroValue::Float(y)) => Ok(MacroValue::Float(*x as f64 % y)),
        (MacroValue::Float(x), MacroValue::Int(y)) => Ok(MacroValue::Float(x % *y as f64)),
        _ => Err(VmError::new(format!(
            "cannot modulo {} and {}",
            a.type_name(),
            b.type_name()
        ))),
    }
}

fn vm_compare(
    a: MacroValue,
    b: MacroValue,
    pred: impl FnOnce(std::cmp::Ordering) -> bool,
) -> Result<MacroValue, VmError> {
    let ord = match (&a, &b) {
        (MacroValue::Int(x), MacroValue::Int(y)) => x.cmp(y),
        (MacroValue::Float(x), MacroValue::Float(y)) => {
            x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal)
        }
        (MacroValue::Int(x), MacroValue::Float(y)) => (*x as f64)
            .partial_cmp(y)
            .unwrap_or(std::cmp::Ordering::Equal),
        (MacroValue::Float(x), MacroValue::Int(y)) => x
            .partial_cmp(&(*y as f64))
            .unwrap_or(std::cmp::Ordering::Equal),
        (MacroValue::String(x), MacroValue::String(y)) => x.cmp(y),
        _ => {
            return Err(VmError::new(format!(
                "cannot compare {} and {}",
                a.type_name(),
                b.type_name()
            )));
        }
    };
    Ok(MacroValue::Bool(pred(ord)))
}

#[cfg(test)]
mod tests {
    use super::super::compiler::BytecodeCompiler;
    use super::*;

    fn compile_and_run(expr: &parser::Expr) -> Result<MacroValue, VmError> {
        let chunk =
            BytecodeCompiler::compile_expr_standalone(expr).map_err(|e| VmError::new(e.message))?;
        let mut vm = MacroVm::new();
        vm.execute(Arc::new(chunk), vec![])
    }

    fn make_expr(kind: parser::ExprKind) -> parser::Expr {
        parser::Expr {
            kind,
            span: Span { start: 0, end: 0 },
        }
    }

    #[test]
    fn test_vm_int_literal() {
        let result = compile_and_run(&make_expr(parser::ExprKind::Int(42))).unwrap();
        assert_eq!(result, MacroValue::Int(42));
    }

    #[test]
    fn test_vm_int_zero_one() {
        let r0 = compile_and_run(&make_expr(parser::ExprKind::Int(0))).unwrap();
        assert_eq!(r0, MacroValue::Int(0));
        let r1 = compile_and_run(&make_expr(parser::ExprKind::Int(1))).unwrap();
        assert_eq!(r1, MacroValue::Int(1));
    }

    #[test]
    fn test_vm_float_literal() {
        let result = compile_and_run(&make_expr(parser::ExprKind::Float(3.14))).unwrap();
        assert_eq!(result, MacroValue::Float(3.14));
    }

    #[test]
    fn test_vm_string_literal() {
        let result =
            compile_and_run(&make_expr(parser::ExprKind::String("hello".to_string()))).unwrap();
        assert_eq!(result, MacroValue::String(Arc::from("hello")));
    }

    #[test]
    fn test_vm_bool_null() {
        assert_eq!(
            compile_and_run(&make_expr(parser::ExprKind::Bool(true))).unwrap(),
            MacroValue::Bool(true)
        );
        assert_eq!(
            compile_and_run(&make_expr(parser::ExprKind::Bool(false))).unwrap(),
            MacroValue::Bool(false)
        );
        assert_eq!(
            compile_and_run(&make_expr(parser::ExprKind::Null)).unwrap(),
            MacroValue::Null
        );
    }

    #[test]
    fn test_vm_add_ints() {
        // 2 + 3
        let expr = make_expr(parser::ExprKind::Binary {
            left: Box::new(make_expr(parser::ExprKind::Int(2))),
            op: parser::BinaryOp::Add,
            right: Box::new(make_expr(parser::ExprKind::Int(3))),
        });
        assert_eq!(compile_and_run(&expr).unwrap(), MacroValue::Int(5));
    }

    #[test]
    fn test_vm_add_floats() {
        let expr = make_expr(parser::ExprKind::Binary {
            left: Box::new(make_expr(parser::ExprKind::Float(1.5))),
            op: parser::BinaryOp::Add,
            right: Box::new(make_expr(parser::ExprKind::Float(2.5))),
        });
        assert_eq!(compile_and_run(&expr).unwrap(), MacroValue::Float(4.0));
    }

    #[test]
    fn test_vm_string_concat() {
        let expr = make_expr(parser::ExprKind::Binary {
            left: Box::new(make_expr(parser::ExprKind::String("hello ".to_string()))),
            op: parser::BinaryOp::Add,
            right: Box::new(make_expr(parser::ExprKind::String("world".to_string()))),
        });
        assert_eq!(
            compile_and_run(&expr).unwrap(),
            MacroValue::String(Arc::from("hello world"))
        );
    }

    #[test]
    fn test_vm_arithmetic() {
        // (10 - 3) * 2 = 14
        let sub = make_expr(parser::ExprKind::Binary {
            left: Box::new(make_expr(parser::ExprKind::Int(10))),
            op: parser::BinaryOp::Sub,
            right: Box::new(make_expr(parser::ExprKind::Int(3))),
        });
        let expr = make_expr(parser::ExprKind::Binary {
            left: Box::new(sub),
            op: parser::BinaryOp::Mul,
            right: Box::new(make_expr(parser::ExprKind::Int(2))),
        });
        assert_eq!(compile_and_run(&expr).unwrap(), MacroValue::Int(14));
    }

    #[test]
    fn test_vm_comparison() {
        let expr = make_expr(parser::ExprKind::Binary {
            left: Box::new(make_expr(parser::ExprKind::Int(5))),
            op: parser::BinaryOp::Lt,
            right: Box::new(make_expr(parser::ExprKind::Int(10))),
        });
        assert_eq!(compile_and_run(&expr).unwrap(), MacroValue::Bool(true));

        let expr2 = make_expr(parser::ExprKind::Binary {
            left: Box::new(make_expr(parser::ExprKind::Int(10))),
            op: parser::BinaryOp::Eq,
            right: Box::new(make_expr(parser::ExprKind::Int(10))),
        });
        assert_eq!(compile_and_run(&expr2).unwrap(), MacroValue::Bool(true));
    }

    #[test]
    fn test_vm_unary_neg() {
        let expr = make_expr(parser::ExprKind::Unary {
            op: parser::UnaryOp::Neg,
            expr: Box::new(make_expr(parser::ExprKind::Int(42))),
        });
        assert_eq!(compile_and_run(&expr).unwrap(), MacroValue::Int(-42));
    }

    #[test]
    fn test_vm_unary_not() {
        let expr = make_expr(parser::ExprKind::Unary {
            op: parser::UnaryOp::Not,
            expr: Box::new(make_expr(parser::ExprKind::Bool(true))),
        });
        assert_eq!(compile_and_run(&expr).unwrap(), MacroValue::Bool(false));
    }

    #[test]
    fn test_vm_var_decl_and_load() {
        // { var x = 42; x }
        let block = make_expr(parser::ExprKind::Block(vec![
            parser::BlockElement::Expr(make_expr(parser::ExprKind::Var {
                name: "x".to_string(),
                type_hint: None,
                expr: Some(Box::new(make_expr(parser::ExprKind::Int(42)))),
            })),
            parser::BlockElement::Expr(make_expr(parser::ExprKind::Ident("x".to_string()))),
        ]));
        assert_eq!(compile_and_run(&block).unwrap(), MacroValue::Int(42));
    }

    #[test]
    fn test_vm_var_assignment() {
        // { var x = 10; x = x + 5; x }
        let block = make_expr(parser::ExprKind::Block(vec![
            parser::BlockElement::Expr(make_expr(parser::ExprKind::Var {
                name: "x".to_string(),
                type_hint: None,
                expr: Some(Box::new(make_expr(parser::ExprKind::Int(10)))),
            })),
            parser::BlockElement::Expr(make_expr(parser::ExprKind::Assign {
                left: Box::new(make_expr(parser::ExprKind::Ident("x".to_string()))),
                op: parser::AssignOp::Assign,
                right: Box::new(make_expr(parser::ExprKind::Binary {
                    left: Box::new(make_expr(parser::ExprKind::Ident("x".to_string()))),
                    op: parser::BinaryOp::Add,
                    right: Box::new(make_expr(parser::ExprKind::Int(5))),
                })),
            })),
            parser::BlockElement::Expr(make_expr(parser::ExprKind::Ident("x".to_string()))),
        ]));
        assert_eq!(compile_and_run(&block).unwrap(), MacroValue::Int(15));
    }

    #[test]
    fn test_vm_if_else() {
        // if (true) 42 else 99
        let expr = make_expr(parser::ExprKind::If {
            cond: Box::new(make_expr(parser::ExprKind::Bool(true))),
            then_branch: Box::new(make_expr(parser::ExprKind::Int(42))),
            else_branch: Some(Box::new(make_expr(parser::ExprKind::Int(99)))),
        });
        assert_eq!(compile_and_run(&expr).unwrap(), MacroValue::Int(42));

        // if (false) 42 else 99
        let expr2 = make_expr(parser::ExprKind::If {
            cond: Box::new(make_expr(parser::ExprKind::Bool(false))),
            then_branch: Box::new(make_expr(parser::ExprKind::Int(42))),
            else_branch: Some(Box::new(make_expr(parser::ExprKind::Int(99)))),
        });
        assert_eq!(compile_and_run(&expr2).unwrap(), MacroValue::Int(99));
    }

    #[test]
    fn test_vm_ternary() {
        // true ? 1 : 2
        let expr = make_expr(parser::ExprKind::Ternary {
            cond: Box::new(make_expr(parser::ExprKind::Bool(true))),
            then_expr: Box::new(make_expr(parser::ExprKind::Int(1))),
            else_expr: Box::new(make_expr(parser::ExprKind::Int(2))),
        });
        assert_eq!(compile_and_run(&expr).unwrap(), MacroValue::Int(1));
    }

    #[test]
    fn test_vm_short_circuit_and() {
        // false && error  → false (should not evaluate right side)
        let expr = make_expr(parser::ExprKind::Binary {
            left: Box::new(make_expr(parser::ExprKind::Bool(false))),
            op: parser::BinaryOp::And,
            right: Box::new(make_expr(parser::ExprKind::Bool(true))),
        });
        assert_eq!(compile_and_run(&expr).unwrap(), MacroValue::Bool(false));

        // true && false → false
        let expr2 = make_expr(parser::ExprKind::Binary {
            left: Box::new(make_expr(parser::ExprKind::Bool(true))),
            op: parser::BinaryOp::And,
            right: Box::new(make_expr(parser::ExprKind::Bool(false))),
        });
        assert_eq!(compile_and_run(&expr2).unwrap(), MacroValue::Bool(false));
    }

    #[test]
    fn test_vm_short_circuit_or() {
        // true || anything → true
        let expr = make_expr(parser::ExprKind::Binary {
            left: Box::new(make_expr(parser::ExprKind::Bool(true))),
            op: parser::BinaryOp::Or,
            right: Box::new(make_expr(parser::ExprKind::Bool(false))),
        });
        assert_eq!(compile_and_run(&expr).unwrap(), MacroValue::Bool(true));
    }

    #[test]
    fn test_vm_while_loop() {
        // { var x = 0; while (x < 5) { x = x + 1; } x }
        let block = make_expr(parser::ExprKind::Block(vec![
            parser::BlockElement::Expr(make_expr(parser::ExprKind::Var {
                name: "x".to_string(),
                type_hint: None,
                expr: Some(Box::new(make_expr(parser::ExprKind::Int(0)))),
            })),
            parser::BlockElement::Expr(make_expr(parser::ExprKind::While {
                cond: Box::new(make_expr(parser::ExprKind::Binary {
                    left: Box::new(make_expr(parser::ExprKind::Ident("x".to_string()))),
                    op: parser::BinaryOp::Lt,
                    right: Box::new(make_expr(parser::ExprKind::Int(5))),
                })),
                body: Box::new(make_expr(parser::ExprKind::Block(vec![
                    parser::BlockElement::Expr(make_expr(parser::ExprKind::Assign {
                        left: Box::new(make_expr(parser::ExprKind::Ident("x".to_string()))),
                        op: parser::AssignOp::Assign,
                        right: Box::new(make_expr(parser::ExprKind::Binary {
                            left: Box::new(make_expr(parser::ExprKind::Ident("x".to_string()))),
                            op: parser::BinaryOp::Add,
                            right: Box::new(make_expr(parser::ExprKind::Int(1))),
                        })),
                    })),
                ]))),
            })),
            parser::BlockElement::Expr(make_expr(parser::ExprKind::Ident("x".to_string()))),
        ]));
        assert_eq!(compile_and_run(&block).unwrap(), MacroValue::Int(5));
    }

    #[test]
    fn test_vm_return() {
        // { return 42; 99 }
        let block = make_expr(parser::ExprKind::Block(vec![
            parser::BlockElement::Expr(make_expr(parser::ExprKind::Return(Some(Box::new(
                make_expr(parser::ExprKind::Int(42)),
            ))))),
            parser::BlockElement::Expr(make_expr(parser::ExprKind::Int(99))),
        ]));
        assert_eq!(compile_and_run(&block).unwrap(), MacroValue::Int(42));
    }

    #[test]
    fn test_vm_array_literal() {
        let expr = make_expr(parser::ExprKind::Array(vec![
            make_expr(parser::ExprKind::Int(1)),
            make_expr(parser::ExprKind::Int(2)),
            make_expr(parser::ExprKind::Int(3)),
        ]));
        let result = compile_and_run(&expr).unwrap();
        match result {
            MacroValue::Array(arr) => {
                assert_eq!(arr.len(), 3);
                assert_eq!(arr[0], MacroValue::Int(1));
                assert_eq!(arr[1], MacroValue::Int(2));
                assert_eq!(arr[2], MacroValue::Int(3));
            }
            _ => panic!("expected Array, got {:?}", result),
        }
    }

    #[test]
    fn test_vm_null_coalesce() {
        // null ?? 42
        let expr = make_expr(parser::ExprKind::Binary {
            left: Box::new(make_expr(parser::ExprKind::Null)),
            op: parser::BinaryOp::NullCoal,
            right: Box::new(make_expr(parser::ExprKind::Int(42))),
        });
        assert_eq!(compile_and_run(&expr).unwrap(), MacroValue::Int(42));

        // 10 ?? 42
        let expr2 = make_expr(parser::ExprKind::Binary {
            left: Box::new(make_expr(parser::ExprKind::Int(10))),
            op: parser::BinaryOp::NullCoal,
            right: Box::new(make_expr(parser::ExprKind::Int(42))),
        });
        assert_eq!(compile_and_run(&expr2).unwrap(), MacroValue::Int(10));
    }

    #[test]
    fn test_vm_bitwise_ops() {
        // 0xFF & 0x0F = 15
        let expr = make_expr(parser::ExprKind::Binary {
            left: Box::new(make_expr(parser::ExprKind::Int(0xFF))),
            op: parser::BinaryOp::BitAnd,
            right: Box::new(make_expr(parser::ExprKind::Int(0x0F))),
        });
        assert_eq!(compile_and_run(&expr).unwrap(), MacroValue::Int(15));

        // 1 << 4 = 16
        let expr2 = make_expr(parser::ExprKind::Binary {
            left: Box::new(make_expr(parser::ExprKind::Int(1))),
            op: parser::BinaryOp::Shl,
            right: Box::new(make_expr(parser::ExprKind::Int(4))),
        });
        assert_eq!(compile_and_run(&expr2).unwrap(), MacroValue::Int(16));
    }

    #[test]
    fn test_vm_function_params() {
        // Compile a function body that returns param + 10
        // function(x) { return x + 10; }
        let body = make_expr(parser::ExprKind::Return(Some(Box::new(make_expr(
            parser::ExprKind::Binary {
                left: Box::new(make_expr(parser::ExprKind::Ident("x".to_string()))),
                op: parser::BinaryOp::Add,
                right: Box::new(make_expr(parser::ExprKind::Int(10))),
            },
        )))));

        let params = vec![MacroParam {
            name: "x".to_string(),
            optional: false,
            rest: false,
            default_value: None,
        }];

        let chunk = BytecodeCompiler::compile("test", &params, &body).unwrap();
        let mut vm = MacroVm::new();
        let result = vm
            .execute(Arc::new(chunk), vec![MacroValue::Int(32)])
            .unwrap();
        assert_eq!(result, MacroValue::Int(42));
    }

    #[test]
    fn test_vm_compound_assignment() {
        // { var x = 10; x += 5; x }
        let block = make_expr(parser::ExprKind::Block(vec![
            parser::BlockElement::Expr(make_expr(parser::ExprKind::Var {
                name: "x".to_string(),
                type_hint: None,
                expr: Some(Box::new(make_expr(parser::ExprKind::Int(10)))),
            })),
            parser::BlockElement::Expr(make_expr(parser::ExprKind::Assign {
                left: Box::new(make_expr(parser::ExprKind::Ident("x".to_string()))),
                op: parser::AssignOp::AddAssign,
                right: Box::new(make_expr(parser::ExprKind::Int(5))),
            })),
            parser::BlockElement::Expr(make_expr(parser::ExprKind::Ident("x".to_string()))),
        ]));
        assert_eq!(compile_and_run(&block).unwrap(), MacroValue::Int(15));
    }

    #[test]
    fn test_vm_division() {
        // 10 / 3
        let expr = make_expr(parser::ExprKind::Binary {
            left: Box::new(make_expr(parser::ExprKind::Int(10))),
            op: parser::BinaryOp::Div,
            right: Box::new(make_expr(parser::ExprKind::Int(3))),
        });
        assert_eq!(compile_and_run(&expr).unwrap(), MacroValue::Int(3));

        // 10.0 / 3.0
        let expr2 = make_expr(parser::ExprKind::Binary {
            left: Box::new(make_expr(parser::ExprKind::Float(10.0))),
            op: parser::BinaryOp::Div,
            right: Box::new(make_expr(parser::ExprKind::Float(3.0))),
        });
        let result = compile_and_run(&expr2).unwrap();
        if let MacroValue::Float(f) = result {
            assert!((f - 3.333333333333333).abs() < 1e-10);
        } else {
            panic!("expected Float");
        }
    }
}
