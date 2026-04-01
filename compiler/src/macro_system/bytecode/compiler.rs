//! Bytecode compiler: AST → Chunk.
//!
//! Walks a Haxe `Expr` AST and emits bytecode instructions into a `Chunk`.
//! Variable names are resolved to local slot indices at compile time for
//! O(1) access at runtime.

use super::super::value::{MacroParam, MacroValue};
use super::chunk::{Chunk, CompiledParam, UpvalueDesc};
use super::opcode::{Emitter, Op};
use parser::{AssignOp, BinaryOp, BlockElement, Expr, ExprKind, Span, UnaryOp};
use std::collections::BTreeMap;
use std::sync::Arc;

/// Error during bytecode compilation.
#[derive(Debug)]
pub struct CompileError {
    pub message: String,
    pub span: Span,
}

impl CompileError {
    fn new(message: impl Into<String>, span: Span) -> Self {
        Self {
            message: message.into(),
            span,
        }
    }
}

/// Resolves variable names to local slot indices with lexical scoping.
struct LocalResolver {
    /// Stack of scopes. Each scope maps name → slot index.
    scopes: Vec<BTreeMap<String, u16>>,
    /// Next slot to allocate.
    next_slot: u16,
}

impl LocalResolver {
    fn new() -> Self {
        Self {
            scopes: vec![BTreeMap::new()],
            next_slot: 0,
        }
    }

    /// Define a new local variable, returning its slot index.
    fn define(&mut self, name: &str) -> u16 {
        let slot = self.next_slot;
        self.next_slot += 1;
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.to_string(), slot);
        }
        slot
    }

    /// Resolve a variable name to its slot index.
    /// Searches from innermost to outermost scope.
    fn resolve(&self, name: &str) -> Option<u16> {
        for scope in self.scopes.iter().rev() {
            if let Some(&slot) = scope.get(name) {
                return Some(slot);
            }
        }
        None
    }

    fn push_scope(&mut self) {
        self.scopes.push(BTreeMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }
}

/// Loop context for break/continue jump patching.
struct LoopContext {
    /// Byte offset of the loop condition (for Continue → backward jump).
    condition_offset: usize,
    /// Byte offsets of Break jump operands that need patching to loop exit.
    break_patches: Vec<usize>,
}

/// Bytecode compiler that transforms an Expr AST into a Chunk.
pub struct BytecodeCompiler {
    emitter: Emitter,
    chunk: Chunk,
    locals: LocalResolver,
    loop_stack: Vec<LoopContext>,
}

impl BytecodeCompiler {
    /// Compile a macro function body into a Chunk.
    ///
    /// `params` are the function parameters to bind as the first local slots.
    /// `body` is the function body expression to compile.
    pub fn compile(name: &str, params: &[MacroParam], body: &Expr) -> Result<Chunk, CompileError> {
        let mut compiler = Self {
            emitter: Emitter::new(),
            chunk: Chunk::new(name),
            locals: LocalResolver::new(),
            loop_stack: Vec::new(),
        };

        // Allocate parameter slots
        for param in params {
            let slot = compiler.locals.define(&param.name);
            compiler.chunk.register_local_name(slot, param.name.clone());
            compiler.chunk.params.push(CompiledParam {
                slot,
                optional: param.optional,
                default_chunk: None, // TODO: compile default expressions
            });
        }

        // Compile body
        compiler.compile_expr(body)?;

        // Ensure there's an implicit return
        compiler.emitter.emit_op(Op::Return);

        // Finalize
        compiler.chunk.code = compiler.emitter.code;
        compiler.chunk.local_count = compiler.locals.next_slot;
        Ok(compiler.chunk)
    }

    /// Compile a class method body into a Chunk.
    ///
    /// `this` is bound as local slot 0. Constructor bodies get an implicit
    /// `LoadLocal(0); Return` appended so they return the constructed object.
    pub fn compile_method(
        name: &str,
        params: &[parser::FunctionParam],
        body: &Expr,
        is_constructor: bool,
    ) -> Result<Chunk, CompileError> {
        let mut compiler = Self {
            emitter: Emitter::new(),
            chunk: Chunk::new(name),
            locals: LocalResolver::new(),
            loop_stack: Vec::new(),
        };

        // Slot 0 = this
        let this_slot = compiler.locals.define("this");
        compiler
            .chunk
            .register_local_name(this_slot, "this".to_string());

        // Allocate parameter slots
        for param in params {
            let slot = compiler.locals.define(&param.name);
            compiler.chunk.register_local_name(slot, param.name.clone());
            compiler.chunk.params.push(CompiledParam {
                slot,
                optional: param.optional,
                default_chunk: None,
            });
        }

        // Compile body
        compiler.compile_expr(body)?;

        if is_constructor {
            // Constructor returns `this`
            compiler.emitter.emit_op(Op::Pop); // discard body result
            compiler.emitter.emit_u16(Op::LoadLocal, 0); // push this
        }

        // Implicit return
        compiler.emitter.emit_op(Op::Return);

        compiler.chunk.code = compiler.emitter.code;
        compiler.chunk.local_count = compiler.locals.next_slot;
        Ok(compiler.chunk)
    }

    /// Compile a standalone expression (for testing).
    pub fn compile_expr_standalone(expr: &Expr) -> Result<Chunk, CompileError> {
        Self::compile("<expr>", &[], expr)
    }

    /// Compile an expression, leaving its result value on the stack.
    fn compile_expr(&mut self, expr: &Expr) -> Result<(), CompileError> {
        self.chunk.add_span(self.emitter.offset(), expr.span);

        match &expr.kind {
            // === Literals ===
            ExprKind::Int(i) => match *i {
                0 => self.emitter.emit_op(Op::PushInt0),
                1 => self.emitter.emit_op(Op::PushInt1),
                _ => {
                    let idx = self.chunk.add_constant(MacroValue::Int(*i));
                    self.emitter.emit_u16(Op::Const, idx);
                }
            },
            ExprKind::Float(f) => {
                let idx = self.chunk.add_constant(MacroValue::Float(*f));
                self.emitter.emit_u16(Op::Const, idx);
            }
            ExprKind::String(s) => {
                let idx = self
                    .chunk
                    .add_constant(MacroValue::String(Arc::from(s.as_str())));
                self.emitter.emit_u16(Op::Const, idx);
            }
            ExprKind::Bool(true) => self.emitter.emit_op(Op::PushTrue),
            ExprKind::Bool(false) => self.emitter.emit_op(Op::PushFalse),
            ExprKind::Null => self.emitter.emit_op(Op::PushNull),

            // === This ===
            ExprKind::This => {
                if let Some(slot) = self.locals.resolve("this") {
                    self.emitter.emit_u16(Op::LoadLocal, slot);
                } else {
                    return Err(CompileError::new("'this' not defined", expr.span));
                }
            }

            // === Identifiers ===
            ExprKind::Ident(name) => {
                if let Some(slot) = self.locals.resolve(name) {
                    self.emitter.emit_u16(Op::LoadLocal, slot);
                } else {
                    return Err(CompileError::new(
                        format!("undefined variable '{}'", name),
                        expr.span,
                    ));
                }
            }

            // === Variable declarations ===
            ExprKind::Var {
                name, expr: init, ..
            }
            | ExprKind::Final {
                name, expr: init, ..
            } => {
                if let Some(init_expr) = init {
                    self.compile_expr(init_expr)?;
                } else {
                    self.emitter.emit_op(Op::PushNull);
                }
                let slot = self.locals.define(name);
                self.chunk.register_local_name(slot, name.clone());
                self.emitter.emit_u16(Op::DefineLocal, slot);
            }

            // === Assignment ===
            ExprKind::Assign { left, op, right } => {
                self.compile_assignment(left, op, right, expr.span)?;
            }

            // === Binary operations ===
            ExprKind::Binary { left, op, right } => {
                match op {
                    // Short-circuit logical AND
                    BinaryOp::And => {
                        self.compile_expr(left)?;
                        let jump = self.emitter.emit_jump(Op::JumpIfFalseKeep);
                        self.emitter.emit_op(Op::Pop);
                        self.compile_expr(right)?;
                        self.emitter.patch_jump(jump);
                    }
                    // Short-circuit logical OR
                    BinaryOp::Or => {
                        self.compile_expr(left)?;
                        let jump = self.emitter.emit_jump(Op::JumpIfTrueKeep);
                        self.emitter.emit_op(Op::Pop);
                        self.compile_expr(right)?;
                        self.emitter.patch_jump(jump);
                    }
                    _ => {
                        self.compile_expr(left)?;
                        self.compile_expr(right)?;
                        self.emit_binary_op(op, expr.span)?;
                    }
                }
            }

            // === Unary operations ===
            ExprKind::Unary { op, expr: inner } => {
                // Pre-increment/decrement on identifiers need special handling
                match op {
                    UnaryOp::PreIncr | UnaryOp::PreDecr => {
                        self.compile_pre_inc_dec(op, inner, expr.span)?;
                    }
                    UnaryOp::PostIncr | UnaryOp::PostDecr => {
                        self.compile_post_inc_dec(op, inner, expr.span)?;
                    }
                    _ => {
                        self.compile_expr(inner)?;
                        match op {
                            UnaryOp::Not => self.emitter.emit_op(Op::Not),
                            UnaryOp::Neg => self.emitter.emit_op(Op::Neg),
                            UnaryOp::BitNot => self.emitter.emit_op(Op::BitNot),
                            _ => unreachable!(),
                        }
                    }
                }
            }

            // === Ternary ===
            ExprKind::Ternary {
                cond,
                then_expr,
                else_expr,
            } => {
                self.compile_expr(cond)?;
                let else_jump = self.emitter.emit_jump(Op::JumpIfFalse);
                self.compile_expr(then_expr)?;
                let end_jump = self.emitter.emit_jump(Op::Jump);
                self.emitter.patch_jump(else_jump);
                self.compile_expr(else_expr)?;
                self.emitter.patch_jump(end_jump);
            }

            // === Parentheses ===
            ExprKind::Paren(inner) => {
                self.compile_expr(inner)?;
            }

            // === Block ===
            ExprKind::Block(elements) => {
                self.compile_block(elements, expr.span)?;
            }

            // === If/else ===
            ExprKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                self.compile_expr(cond)?;
                let else_jump = self.emitter.emit_jump(Op::JumpIfFalse);
                self.compile_expr(then_branch)?;
                if let Some(else_br) = else_branch {
                    let end_jump = self.emitter.emit_jump(Op::Jump);
                    self.emitter.patch_jump(else_jump);
                    self.compile_expr(else_br)?;
                    self.emitter.patch_jump(end_jump);
                } else {
                    let end_jump = self.emitter.emit_jump(Op::Jump);
                    self.emitter.patch_jump(else_jump);
                    self.emitter.emit_op(Op::PushNull);
                    self.emitter.patch_jump(end_jump);
                }
            }

            // === While loop ===
            ExprKind::While { cond, body } => {
                let loop_start = self.emitter.offset();
                self.loop_stack.push(LoopContext {
                    condition_offset: loop_start,
                    break_patches: Vec::new(),
                });

                self.compile_expr(cond)?;
                let exit_jump = self.emitter.emit_jump(Op::JumpIfFalse);

                self.compile_expr(body)?;
                self.emitter.emit_op(Op::Pop); // discard body value

                self.emitter.emit_loop(Op::Jump, loop_start);
                self.emitter.patch_jump(exit_jump);

                // Patch all break jumps
                let ctx = self.loop_stack.pop().unwrap();
                for patch in ctx.break_patches {
                    self.emitter.patch_jump(patch);
                }

                self.emitter.emit_op(Op::PushNull); // while evaluates to null
            }

            // === Do-while loop ===
            ExprKind::DoWhile { body, cond } => {
                let loop_start = self.emitter.offset();
                self.loop_stack.push(LoopContext {
                    condition_offset: loop_start,
                    break_patches: Vec::new(),
                });

                self.compile_expr(body)?;
                self.emitter.emit_op(Op::Pop);

                self.compile_expr(cond)?;
                // Jump back if truthy
                let exit_jump = self.emitter.emit_jump(Op::JumpIfFalse);
                self.emitter.emit_loop(Op::Jump, loop_start);
                self.emitter.patch_jump(exit_jump);

                let ctx = self.loop_stack.pop().unwrap();
                for patch in ctx.break_patches {
                    self.emitter.patch_jump(patch);
                }

                self.emitter.emit_op(Op::PushNull);
            }

            // === Return ===
            ExprKind::Return(value) => {
                if let Some(val_expr) = value {
                    self.compile_expr(val_expr)?;
                    self.emitter.emit_op(Op::Return);
                } else {
                    self.emitter.emit_op(Op::ReturnNull);
                }
            }

            // === Break ===
            ExprKind::Break => {
                let patch = self.emitter.emit_jump(Op::Jump);
                if let Some(ctx) = self.loop_stack.last_mut() {
                    ctx.break_patches.push(patch);
                }
            }

            // === Continue ===
            ExprKind::Continue => {
                if let Some(ctx) = self.loop_stack.last() {
                    let target = ctx.condition_offset;
                    self.emitter.emit_loop(Op::Jump, target);
                }
            }

            // === For-in loop ===
            ExprKind::For {
                var, iter, body, ..
            } => {
                self.compile_for_in(var, iter, body, expr.span)?;
            }

            // === Array literal ===
            ExprKind::Array(elements) => {
                for elem in elements {
                    self.compile_expr(elem)?;
                }
                self.emitter.emit_u16(Op::MakeArray, elements.len() as u16);
            }

            // === Object literal ===
            ExprKind::Object(fields) => {
                for field in fields {
                    let name_idx = self.chunk.intern_string(&field.name);
                    self.emitter.emit_u16(Op::Const, name_idx);
                    self.compile_expr(&field.expr)?;
                }
                self.emitter.emit_u16(Op::MakeObject, fields.len() as u16);
            }

            // === Map literal ===
            ExprKind::Map(pairs) => {
                for (key, value) in pairs {
                    self.compile_expr(key)?;
                    self.compile_expr(value)?;
                }
                self.emitter.emit_u16(Op::MakeMap, pairs.len() as u16);
            }

            // === String interpolation ===
            ExprKind::StringInterpolation(parts) => {
                self.compile_string_interpolation(parts)?;
            }

            // === Field access ===
            ExprKind::Field {
                expr: base,
                field,
                is_optional,
            } => {
                self.compile_expr(base)?;
                let name_idx = self.chunk.intern_string(field);
                if *is_optional {
                    self.emitter.emit_u16(Op::GetFieldOpt, name_idx);
                } else {
                    self.emitter.emit_u16(Op::GetField, name_idx);
                }
            }

            // === Index access ===
            ExprKind::Index {
                expr: base, index, ..
            } => {
                self.compile_expr(base)?;
                self.compile_expr(index)?;
                self.emitter.emit_op(Op::GetIndex);
            }

            // === Function call ===
            ExprKind::Call { expr: callee, args } => {
                self.compile_call(callee, args, expr.span)?;
            }

            // === New (constructor) ===
            ExprKind::New {
                type_path, args, ..
            } => {
                let class_name = if type_path.package.is_empty() {
                    type_path.name.clone()
                } else {
                    format!("{}.{}", type_path.package.join("."), type_path.name)
                };
                for arg in args {
                    self.compile_expr(arg)?;
                }
                let name_idx = self.chunk.intern_string(&class_name);
                self.emitter
                    .emit_u16_u8(Op::NewObject, name_idx, args.len() as u8);
            }

            // === Pass-through expressions ===
            ExprKind::Cast { expr: inner, .. }
            | ExprKind::TypeCheck { expr: inner, .. }
            | ExprKind::Meta { expr: inner, .. }
            | ExprKind::Untyped(inner)
            | ExprKind::Inline(inner) => {
                self.compile_expr(inner)?;
            }

            // === Throw ===
            ExprKind::Throw(inner) => {
                self.compile_expr(inner)?;
                // Throw is compiled as a special builtin call
                self.emitter.emit_u16_u8(Op::CallBuiltin, 0xFFFF, 1); // sentinel for throw
            }

            // === Switch ===
            ExprKind::Switch {
                expr: scrutinee,
                cases,
                default,
            } => {
                self.compile_switch(scrutinee, cases, default.as_deref(), expr.span)?;
            }

            // === Try/catch/finally ===
            ExprKind::Try {
                expr: try_body,
                catches,
                finally_block,
            } => {
                // Simplified: just compile try body, if it errors fall through
                // Full exception handling is Phase 6
                self.compile_expr(try_body)?;
                // TODO: proper try/catch bytecode in Phase 6
                let _ = (catches, finally_block);
            }

            // === Function literal ===
            ExprKind::Function(func) => {
                self.compile_function_literal(func, expr.span)?;
            }

            // === Arrow function ===
            ExprKind::Arrow { params, expr: body } => {
                self.compile_arrow(params, body, expr.span)?;
            }

            // === Macro/Reify/Dollar ===
            ExprKind::Macro(inner) | ExprKind::Reify(inner) => {
                self.compile_expr(inner)?;
                self.emitter.emit_op(Op::Reify);
            }
            ExprKind::DollarIdent { name, arg } => {
                if let Some(arg_expr) = arg {
                    self.compile_expr(arg_expr)?;
                    let kind_idx = self.chunk.intern_string(name);
                    self.emitter.emit_u16(Op::DollarSplice, kind_idx);
                } else if let Some(slot) = self.locals.resolve(name) {
                    self.emitter.emit_u16(Op::LoadLocal, slot);
                } else {
                    return Err(CompileError::new(
                        format!("undefined variable '${}'", name),
                        expr.span,
                    ));
                }
            }

            _ => {
                return Err(CompileError::new(
                    format!(
                        "unsupported expression: {:?}",
                        std::mem::discriminant(&expr.kind)
                    ),
                    expr.span,
                ));
            }
        }

        Ok(())
    }

    /// Compile a block of statements.
    fn compile_block(
        &mut self,
        elements: &[BlockElement],
        _span: Span,
    ) -> Result<(), CompileError> {
        self.locals.push_scope();

        if elements.is_empty() {
            self.emitter.emit_op(Op::PushNull);
        } else {
            for (i, elem) in elements.iter().enumerate() {
                match elem {
                    BlockElement::Expr(expr) => {
                        self.compile_expr(expr)?;
                        // Pop intermediate values, keep the last one
                        if i < elements.len() - 1 {
                            self.emitter.emit_op(Op::Pop);
                        }
                    }
                    BlockElement::Import(_) | BlockElement::Using(_) => {
                        // Skip imports in macro context
                        if i == elements.len() - 1 {
                            self.emitter.emit_op(Op::PushNull);
                        }
                    }
                    _ => {
                        if i == elements.len() - 1 {
                            self.emitter.emit_op(Op::PushNull);
                        }
                    }
                }
            }
        }

        self.locals.pop_scope();
        Ok(())
    }

    /// Emit a binary operation opcode.
    fn emit_binary_op(&mut self, op: &BinaryOp, span: Span) -> Result<(), CompileError> {
        match op {
            BinaryOp::Add => self.emitter.emit_op(Op::Add),
            BinaryOp::Sub => self.emitter.emit_op(Op::Sub),
            BinaryOp::Mul => self.emitter.emit_op(Op::Mul),
            BinaryOp::Div => self.emitter.emit_op(Op::Div),
            BinaryOp::Mod => self.emitter.emit_op(Op::Mod),
            BinaryOp::Eq => self.emitter.emit_op(Op::Eq),
            BinaryOp::NotEq => self.emitter.emit_op(Op::NotEq),
            BinaryOp::Lt => self.emitter.emit_op(Op::Lt),
            BinaryOp::Le => self.emitter.emit_op(Op::Le),
            BinaryOp::Gt => self.emitter.emit_op(Op::Gt),
            BinaryOp::Ge => self.emitter.emit_op(Op::Ge),
            BinaryOp::BitAnd => self.emitter.emit_op(Op::BitAnd),
            BinaryOp::BitOr => self.emitter.emit_op(Op::BitOr),
            BinaryOp::BitXor => self.emitter.emit_op(Op::BitXor),
            BinaryOp::Shl => self.emitter.emit_op(Op::Shl),
            BinaryOp::Shr => self.emitter.emit_op(Op::Shr),
            BinaryOp::Ushr => self.emitter.emit_op(Op::Ushr),
            BinaryOp::NullCoal => self.emitter.emit_op(Op::NullCoal),
            BinaryOp::And | BinaryOp::Or => unreachable!("handled via short-circuit"),
            _ => {
                return Err(CompileError::new(
                    format!("unsupported binary op: {:?}", op),
                    span,
                ));
            }
        }
        Ok(())
    }

    /// Compile pre-increment/decrement on an identifier.
    fn compile_pre_inc_dec(
        &mut self,
        op: &UnaryOp,
        inner: &Expr,
        span: Span,
    ) -> Result<(), CompileError> {
        if let ExprKind::Ident(name) = &inner.kind {
            if let Some(slot) = self.locals.resolve(name) {
                self.emitter.emit_u16(Op::LoadLocal, slot);
                match op {
                    UnaryOp::PreIncr => self.emitter.emit_op(Op::Incr),
                    UnaryOp::PreDecr => self.emitter.emit_op(Op::Decr),
                    _ => unreachable!(),
                }
                self.emitter.emit_u16(Op::StoreLocal, slot);
                // StoreLocal doesn't pop, so the incremented value is on top
            } else {
                return Err(CompileError::new(
                    format!("undefined variable '{}'", name),
                    span,
                ));
            }
        } else {
            return Err(CompileError::new("pre-inc/dec requires identifier", span));
        }
        Ok(())
    }

    /// Compile post-increment/decrement on an identifier.
    fn compile_post_inc_dec(
        &mut self,
        op: &UnaryOp,
        inner: &Expr,
        span: Span,
    ) -> Result<(), CompileError> {
        if let ExprKind::Ident(name) = &inner.kind {
            if let Some(slot) = self.locals.resolve(name) {
                self.emitter.emit_u16(Op::LoadLocal, slot); // push original
                self.emitter.emit_op(Op::Dup); // dup for returning original
                match op {
                    UnaryOp::PostIncr => self.emitter.emit_op(Op::Incr),
                    UnaryOp::PostDecr => self.emitter.emit_op(Op::Decr),
                    _ => unreachable!(),
                }
                self.emitter.emit_u16(Op::StoreLocal, slot); // store incremented
                self.emitter.emit_op(Op::Pop); // pop the incremented, leaving original
            } else {
                return Err(CompileError::new(
                    format!("undefined variable '{}'", name),
                    span,
                ));
            }
        } else {
            return Err(CompileError::new("post-inc/dec requires identifier", span));
        }
        Ok(())
    }

    /// Compile assignment.
    fn compile_assignment(
        &mut self,
        left: &Expr,
        op: &AssignOp,
        right: &Expr,
        span: Span,
    ) -> Result<(), CompileError> {
        match &left.kind {
            ExprKind::Ident(name) => {
                if *op != AssignOp::Assign {
                    // Compound assignment: load current value, compute, store
                    if let Some(slot) = self.locals.resolve(name) {
                        self.emitter.emit_u16(Op::LoadLocal, slot);
                        self.compile_expr(right)?;
                        self.emit_compound_op(op, span)?;
                        self.emitter.emit_u16(Op::StoreLocal, slot);
                    } else {
                        return Err(CompileError::new(
                            format!("undefined variable '{}'", name),
                            span,
                        ));
                    }
                } else {
                    self.compile_expr(right)?;
                    if let Some(slot) = self.locals.resolve(name) {
                        self.emitter.emit_u16(Op::StoreLocal, slot);
                    } else {
                        // Variable doesn't exist — define it (mimics tree-walker behavior)
                        let slot = self.locals.define(name);
                        self.chunk.register_local_name(slot, name.clone());
                        self.emitter.emit_u16(Op::DefineLocal, slot);
                    }
                }
            }
            ExprKind::Field {
                expr: base, field, ..
            } => {
                // Check if base is a local variable — use SetFieldLocal for in-place mutation
                let local_slot = if let ExprKind::Ident(name) = &base.kind {
                    self.locals.resolve(name)
                } else {
                    None
                };

                if *op != AssignOp::Assign {
                    // Compound field assignment
                    if let Some(slot) = local_slot {
                        // Optimized path: read from local, compute, write back to local
                        let name_idx = self.chunk.intern_string(field);
                        self.emitter.emit_u16(Op::LoadLocal, slot);
                        self.emitter.emit_u16(Op::GetField, name_idx);
                        self.compile_expr(right)?;
                        self.emit_compound_op(op, span)?;
                        self.emitter.emit_u16_u16(Op::SetFieldLocal, slot, name_idx);
                    } else {
                        self.compile_expr(base)?;
                        self.emitter.emit_op(Op::Dup); // keep base for SetField
                        let name_idx = self.chunk.intern_string(field);
                        self.emitter.emit_u16(Op::GetField, name_idx);
                        self.compile_expr(right)?;
                        self.emit_compound_op(op, span)?;
                        self.emitter.emit_u16(Op::SetField, name_idx);
                    }
                } else {
                    let name_idx = self.chunk.intern_string(field);
                    if let Some(slot) = local_slot {
                        // Optimized: modify local in-place
                        self.compile_expr(right)?;
                        self.emitter.emit_u16_u16(Op::SetFieldLocal, slot, name_idx);
                    } else {
                        self.compile_expr(base)?;
                        self.compile_expr(right)?;
                        self.emitter.emit_u16(Op::SetField, name_idx);
                    }
                }
            }
            ExprKind::Index {
                expr: base, index, ..
            } => {
                if *op != AssignOp::Assign {
                    // Compound index assignment
                    self.compile_expr(base)?;
                    self.compile_expr(index)?;
                    self.emitter.emit_op(Op::Dup); // TODO: need to dup both base and index
                                                   // This is simplified; full implementation in Phase 5
                    self.compile_expr(right)?;
                    self.emit_compound_op(op, span)?;
                    self.emitter.emit_op(Op::SetIndex);
                } else {
                    self.compile_expr(base)?;
                    self.compile_expr(index)?;
                    self.compile_expr(right)?;
                    self.emitter.emit_op(Op::SetIndex);
                }
            }
            _ => {
                return Err(CompileError::new("invalid assignment target", span));
            }
        }
        Ok(())
    }

    /// Emit the binary op for a compound assignment.
    fn emit_compound_op(&mut self, op: &AssignOp, span: Span) -> Result<(), CompileError> {
        match op {
            AssignOp::AddAssign => self.emitter.emit_op(Op::Add),
            AssignOp::SubAssign => self.emitter.emit_op(Op::Sub),
            AssignOp::MulAssign => self.emitter.emit_op(Op::Mul),
            AssignOp::DivAssign => self.emitter.emit_op(Op::Div),
            AssignOp::ModAssign => self.emitter.emit_op(Op::Mod),
            AssignOp::AndAssign => self.emitter.emit_op(Op::BitAnd),
            AssignOp::OrAssign => self.emitter.emit_op(Op::BitOr),
            AssignOp::XorAssign => self.emitter.emit_op(Op::BitXor),
            AssignOp::ShlAssign => self.emitter.emit_op(Op::Shl),
            AssignOp::ShrAssign => self.emitter.emit_op(Op::Shr),
            AssignOp::Assign => unreachable!(),
            _ => {
                return Err(CompileError::new(
                    format!("unsupported compound op: {:?}", op),
                    span,
                ));
            }
        }
        Ok(())
    }

    /// Compile a for-in loop.
    fn compile_for_in(
        &mut self,
        var: &str,
        iter: &Expr,
        body: &Expr,
        _span: Span,
    ) -> Result<(), CompileError> {
        self.locals.push_scope();

        // Evaluate iterable and store to temp
        self.compile_expr(iter)?;
        let iter_slot = self.locals.define("__iter");
        self.emitter.emit_u16(Op::DefineLocal, iter_slot);
        self.emitter.emit_op(Op::Pop);

        // Get length: iter.length
        self.emitter.emit_u16(Op::LoadLocal, iter_slot);
        let length_idx = self.chunk.intern_string("length");
        self.emitter.emit_u16(Op::GetField, length_idx);
        let len_slot = self.locals.define("__len");
        self.emitter.emit_u16(Op::DefineLocal, len_slot);
        self.emitter.emit_op(Op::Pop);

        // Counter = 0
        self.emitter.emit_op(Op::PushInt0);
        let idx_slot = self.locals.define("__idx");
        self.emitter.emit_u16(Op::DefineLocal, idx_slot);
        self.emitter.emit_op(Op::Pop);

        // Loop var
        self.emitter.emit_op(Op::PushNull);
        let var_slot = self.locals.define(var);
        self.chunk.register_local_name(var_slot, var.to_string());
        self.emitter.emit_u16(Op::DefineLocal, var_slot);
        self.emitter.emit_op(Op::Pop);

        // Loop condition
        let loop_start = self.emitter.offset();
        self.loop_stack.push(LoopContext {
            condition_offset: loop_start,
            break_patches: Vec::new(),
        });

        self.emitter.emit_u16(Op::LoadLocal, idx_slot);
        self.emitter.emit_u16(Op::LoadLocal, len_slot);
        self.emitter.emit_op(Op::Lt);
        let exit_jump = self.emitter.emit_jump(Op::JumpIfFalse);

        // iter[idx] → loop var
        self.emitter.emit_u16(Op::LoadLocal, iter_slot);
        self.emitter.emit_u16(Op::LoadLocal, idx_slot);
        self.emitter.emit_op(Op::GetIndex);
        self.emitter.emit_u16(Op::StoreLocal, var_slot);
        self.emitter.emit_op(Op::Pop);

        // Body
        self.compile_expr(body)?;
        self.emitter.emit_op(Op::Pop);

        // Increment counter
        self.emitter.emit_u16(Op::LoadLocal, idx_slot);
        self.emitter.emit_op(Op::Incr);
        self.emitter.emit_u16(Op::StoreLocal, idx_slot);
        self.emitter.emit_op(Op::Pop);

        // Back to condition
        self.emitter.emit_loop(Op::Jump, loop_start);
        self.emitter.patch_jump(exit_jump);

        let ctx = self.loop_stack.pop().unwrap();
        for patch in ctx.break_patches {
            self.emitter.patch_jump(patch);
        }

        self.locals.pop_scope();
        self.emitter.emit_op(Op::PushNull);
        Ok(())
    }

    /// Compile string interpolation.
    fn compile_string_interpolation(
        &mut self,
        parts: &[parser::StringPart],
    ) -> Result<(), CompileError> {
        if parts.is_empty() {
            let idx = self.chunk.add_constant(MacroValue::String(Arc::from("")));
            self.emitter.emit_u16(Op::Const, idx);
            return Ok(());
        }

        // Compile first part
        let mut count = 0;
        for part in parts {
            match part {
                parser::StringPart::Literal(s) => {
                    let idx = self
                        .chunk
                        .add_constant(MacroValue::String(Arc::from(s.as_str())));
                    self.emitter.emit_u16(Op::Const, idx);
                    count += 1;
                }
                parser::StringPart::Interpolation(expr) => {
                    self.compile_expr(expr)?;
                    // Will be converted to string via Add semantics
                    count += 1;
                }
            }
        }

        // Concatenate all parts with Add
        for _ in 1..count {
            self.emitter.emit_op(Op::Add);
        }
        Ok(())
    }

    /// Compile a function call.
    fn compile_call(
        &mut self,
        callee: &Expr,
        args: &[Expr],
        span: Span,
    ) -> Result<(), CompileError> {
        match &callee.kind {
            // Static method call: Class.method(args)
            ExprKind::Field {
                expr: base,
                field: method,
                ..
            } => {
                if let ExprKind::Ident(class_name) = &base.kind {
                    // Check if this is a known static call pattern
                    let class_idx = self.chunk.intern_string(class_name);
                    let method_idx = self.chunk.intern_string(method);

                    for arg in args {
                        self.compile_expr(arg)?;
                    }
                    self.emitter.emit_u16_u16_u8(
                        Op::CallStatic,
                        class_idx,
                        method_idx,
                        args.len() as u8,
                    );
                } else {
                    // Method call on expression: expr.method(args)
                    self.compile_expr(base)?;
                    for arg in args {
                        self.compile_expr(arg)?;
                    }
                    let method_idx = self.chunk.intern_string(method);
                    self.emitter
                        .emit_u16_u8(Op::CallMethod, method_idx, args.len() as u8);
                }
            }
            _ => {
                // General call: callee(args)
                self.compile_expr(callee)?;
                for arg in args {
                    self.compile_expr(arg)?;
                }
                self.emitter.emit_u8(Op::Call, args.len() as u8);
            }
        }
        Ok(())
    }

    /// Compile a switch expression.
    fn compile_switch(
        &mut self,
        scrutinee: &Expr,
        cases: &[parser::Case],
        default: Option<&Expr>,
        _span: Span,
    ) -> Result<(), CompileError> {
        // Compile scrutinee and store in temp
        self.compile_expr(scrutinee)?;
        let temp_slot = self.locals.define("__switch");
        self.emitter.emit_u16(Op::DefineLocal, temp_slot);
        self.emitter.emit_op(Op::Pop);

        let mut end_patches = Vec::new();

        for case in cases {
            // For each pattern in the case
            let mut next_case_patch = None;
            for pattern in &case.patterns {
                match pattern {
                    parser::Pattern::Const(pat_expr) => {
                        self.emitter.emit_u16(Op::LoadLocal, temp_slot);
                        self.compile_expr(pat_expr)?;
                        self.emitter.emit_op(Op::Eq);
                        next_case_patch = Some(self.emitter.emit_jump(Op::JumpIfFalse));
                    }
                    parser::Pattern::Underscore => {
                        // Always matches — no test needed
                    }
                    parser::Pattern::Var(name) => {
                        // Bind scrutinee to variable
                        self.emitter.emit_u16(Op::LoadLocal, temp_slot);
                        let var_slot = self.locals.define(name);
                        self.chunk.register_local_name(var_slot, name.clone());
                        self.emitter.emit_u16(Op::DefineLocal, var_slot);
                        self.emitter.emit_op(Op::Pop);
                    }
                    _ => {
                        // Unsupported pattern — skip
                        next_case_patch = Some(self.emitter.emit_jump(Op::Jump));
                    }
                }
            }

            // Compile case body
            self.compile_expr(&case.body)?;
            end_patches.push(self.emitter.emit_jump(Op::Jump));

            // Patch the "next case" jump
            if let Some(patch) = next_case_patch {
                self.emitter.patch_jump(patch);
            }
        }

        // Default case
        if let Some(default_body) = default {
            self.compile_expr(default_body)?;
        } else {
            self.emitter.emit_op(Op::PushNull);
        }

        // Patch all end jumps
        for patch in end_patches {
            self.emitter.patch_jump(patch);
        }

        Ok(())
    }

    /// Compile a function literal.
    fn compile_function_literal(
        &mut self,
        func: &parser::Function,
        _span: Span,
    ) -> Result<(), CompileError> {
        let params: Vec<MacroParam> = func
            .params
            .iter()
            .map(|p| MacroParam {
                name: p.name.clone(),
                optional: p.optional,
                rest: p.rest,
                default_value: p.default_value.clone(),
            })
            .collect();

        if let Some(body) = &func.body {
            let chunk = BytecodeCompiler::compile(&func.name, &params, body)
                .map_err(|e| CompileError::new(e.message, _span))?;
            let chunk_idx = self.chunk.add_closure(chunk);
            self.emitter.emit_u16(Op::MakeClosure, chunk_idx);
        } else {
            self.emitter.emit_op(Op::PushNull);
        }
        Ok(())
    }

    /// Compile an arrow function.
    fn compile_arrow(
        &mut self,
        params: &[parser::ArrowParam],
        body: &Expr,
        span: Span,
    ) -> Result<(), CompileError> {
        let macro_params: Vec<MacroParam> = params
            .iter()
            .map(|p| MacroParam {
                name: p.name.clone(),
                optional: false,
                rest: false,
                default_value: None,
            })
            .collect();

        let chunk = BytecodeCompiler::compile("<arrow>", &macro_params, body)
            .map_err(|e| CompileError::new(e.message, span))?;
        let chunk_idx = self.chunk.add_closure(chunk);
        self.emitter.emit_u16(Op::MakeClosure, chunk_idx);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::opcode::Reader;
    use super::*;

    fn make_expr(kind: ExprKind) -> Expr {
        Expr {
            kind,
            span: Span { start: 0, end: 0 },
        }
    }

    #[test]
    fn test_compile_int_literal() {
        let expr = make_expr(ExprKind::Int(42));
        let chunk = BytecodeCompiler::compile_expr_standalone(&expr).unwrap();

        let mut r = Reader::new(&chunk.code);
        assert_eq!(r.read_op(), Some(Op::Const));
        let idx = r.read_u16();
        assert_eq!(chunk.constants[idx as usize], MacroValue::Int(42));
        assert_eq!(r.read_op(), Some(Op::Return)); // implicit return
    }

    #[test]
    fn test_compile_int_zero_one_optimized() {
        let expr0 = make_expr(ExprKind::Int(0));
        let chunk0 = BytecodeCompiler::compile_expr_standalone(&expr0).unwrap();
        assert_eq!(chunk0.code[0], Op::PushInt0 as u8);

        let expr1 = make_expr(ExprKind::Int(1));
        let chunk1 = BytecodeCompiler::compile_expr_standalone(&expr1).unwrap();
        assert_eq!(chunk1.code[0], Op::PushInt1 as u8);
    }

    #[test]
    fn test_compile_bool_null() {
        let t = make_expr(ExprKind::Bool(true));
        let chunk_t = BytecodeCompiler::compile_expr_standalone(&t).unwrap();
        assert_eq!(chunk_t.code[0], Op::PushTrue as u8);

        let f = make_expr(ExprKind::Bool(false));
        let chunk_f = BytecodeCompiler::compile_expr_standalone(&f).unwrap();
        assert_eq!(chunk_f.code[0], Op::PushFalse as u8);

        let n = make_expr(ExprKind::Null);
        let chunk_n = BytecodeCompiler::compile_expr_standalone(&n).unwrap();
        assert_eq!(chunk_n.code[0], Op::PushNull as u8);
    }

    #[test]
    fn test_compile_binary_add() {
        // 2 + 3
        let expr = make_expr(ExprKind::Binary {
            left: Box::new(make_expr(ExprKind::Int(2))),
            op: BinaryOp::Add,
            right: Box::new(make_expr(ExprKind::Int(3))),
        });
        let chunk = BytecodeCompiler::compile_expr_standalone(&expr).unwrap();

        let mut r = Reader::new(&chunk.code);
        // Const(2)
        assert_eq!(r.read_op(), Some(Op::Const));
        let idx = r.read_u16();
        assert_eq!(chunk.constants[idx as usize], MacroValue::Int(2));
        // Const(3)
        assert_eq!(r.read_op(), Some(Op::Const));
        let idx = r.read_u16();
        assert_eq!(chunk.constants[idx as usize], MacroValue::Int(3));
        // Add
        assert_eq!(r.read_op(), Some(Op::Add));
        // Return
        assert_eq!(r.read_op(), Some(Op::Return));
    }

    #[test]
    fn test_compile_var_decl_and_load() {
        // { var x = 42; x }
        let block = make_expr(ExprKind::Block(vec![
            BlockElement::Expr(make_expr(ExprKind::Var {
                name: "x".to_string(),
                type_hint: None,
                expr: Some(Box::new(make_expr(ExprKind::Int(42)))),
            })),
            BlockElement::Expr(make_expr(ExprKind::Ident("x".to_string()))),
        ]));

        let chunk = BytecodeCompiler::compile_expr_standalone(&block).unwrap();

        let mut r = Reader::new(&chunk.code);
        // Const(42)
        assert_eq!(r.read_op(), Some(Op::Const));
        let _ = r.read_u16();
        // DefineLocal(0) -- x is slot 0
        assert_eq!(r.read_op(), Some(Op::DefineLocal));
        let slot = r.read_u16();
        assert_eq!(slot, 0);
        // Pop (intermediate value from var decl)
        assert_eq!(r.read_op(), Some(Op::Pop));
        // LoadLocal(0) -- load x
        assert_eq!(r.read_op(), Some(Op::LoadLocal));
        let slot = r.read_u16();
        assert_eq!(slot, 0);
        // Return
        assert_eq!(r.read_op(), Some(Op::Return));
    }

    #[test]
    fn test_compile_short_circuit_and() {
        // true && false
        let expr = make_expr(ExprKind::Binary {
            left: Box::new(make_expr(ExprKind::Bool(true))),
            op: BinaryOp::And,
            right: Box::new(make_expr(ExprKind::Bool(false))),
        });
        let chunk = BytecodeCompiler::compile_expr_standalone(&expr).unwrap();

        let mut r = Reader::new(&chunk.code);
        assert_eq!(r.read_op(), Some(Op::PushTrue));
        assert_eq!(r.read_op(), Some(Op::JumpIfFalseKeep));
        let _offset = r.read_i16();
        assert_eq!(r.read_op(), Some(Op::Pop));
        assert_eq!(r.read_op(), Some(Op::PushFalse));
        assert_eq!(r.read_op(), Some(Op::Return));
    }

    #[test]
    fn test_compile_if_else() {
        // if (true) 1 else 2
        let expr = make_expr(ExprKind::If {
            cond: Box::new(make_expr(ExprKind::Bool(true))),
            then_branch: Box::new(make_expr(ExprKind::Int(1))),
            else_branch: Some(Box::new(make_expr(ExprKind::Int(2)))),
        });
        let chunk = BytecodeCompiler::compile_expr_standalone(&expr).unwrap();

        // Verify it compiles without error and contains jump instructions
        let code = &chunk.code;
        assert!(code.contains(&(Op::JumpIfFalse as u8)));
        assert!(code.contains(&(Op::Jump as u8)));
    }

    #[test]
    fn test_compile_return() {
        let expr = make_expr(ExprKind::Return(Some(Box::new(make_expr(ExprKind::Int(
            99,
        ))))));
        let chunk = BytecodeCompiler::compile_expr_standalone(&expr).unwrap();

        let mut r = Reader::new(&chunk.code);
        assert_eq!(r.read_op(), Some(Op::Const));
        let _ = r.read_u16();
        assert_eq!(r.read_op(), Some(Op::Return));
    }

    #[test]
    fn test_compile_unary_neg() {
        let expr = make_expr(ExprKind::Unary {
            op: UnaryOp::Neg,
            expr: Box::new(make_expr(ExprKind::Int(5))),
        });
        let chunk = BytecodeCompiler::compile_expr_standalone(&expr).unwrap();

        let mut r = Reader::new(&chunk.code);
        assert_eq!(r.read_op(), Some(Op::Const));
        let _ = r.read_u16();
        assert_eq!(r.read_op(), Some(Op::Neg));
        assert_eq!(r.read_op(), Some(Op::Return));
    }

    #[test]
    fn test_compile_array_literal() {
        // [1, 2, 3]
        let expr = make_expr(ExprKind::Array(vec![
            make_expr(ExprKind::Int(1)),
            make_expr(ExprKind::Int(2)),
            make_expr(ExprKind::Int(3)),
        ]));
        let chunk = BytecodeCompiler::compile_expr_standalone(&expr).unwrap();

        // Should contain MakeArray(3)
        assert!(chunk.code.contains(&(Op::MakeArray as u8)));
    }
}
