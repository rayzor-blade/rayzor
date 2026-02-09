//! Tree-walking interpreter for Haxe macro function bodies.
//!
//! Evaluates Haxe expressions at compile time, producing MacroValue results.
//! The interpreter handles:
//! - Literals, variables, assignments
//! - Binary/unary operations
//! - Control flow (if/else, switch, for, while, return, break, continue)
//! - Function calls (including macro-to-macro calls)
//! - Field access, array indexing, object construction
//! - Built-in functions (trace, Std.string, etc.)

use super::ast_bridge::{self, span_to_location};
use super::environment::Environment;
use super::errors::MacroError;
use super::registry::MacroRegistry;
use super::reification::ReificationEngine;
use super::value::{MacroFunction, MacroParam, MacroValue};
use crate::tast::SourceLocation;
use parser::{AssignOp, BinaryOp, Expr, ExprKind, Span, UnaryOp};

/// Maximum call depth for macro function calls
const DEFAULT_MAX_CALL_DEPTH: usize = 256;

/// The macro interpreter evaluates Haxe AST expressions at compile time.
pub struct MacroInterpreter {
    /// Variable environment with lexical scoping
    env: Environment,
    /// Macro registry for looking up macro definitions
    registry: MacroRegistry,
    /// Current call depth (for recursion detection)
    call_depth: usize,
    /// Maximum call depth
    max_call_depth: usize,
    /// Accumulated trace output
    trace_output: Vec<String>,
}

impl MacroInterpreter {
    pub fn new(registry: MacroRegistry) -> Self {
        Self {
            env: Environment::new(),
            registry,
            call_depth: 0,
            max_call_depth: DEFAULT_MAX_CALL_DEPTH,
            trace_output: Vec::new(),
        }
    }

    /// Get a reference to the environment
    pub fn env(&self) -> &Environment {
        &self.env
    }

    /// Get a mutable reference to the environment
    pub fn env_mut(&mut self) -> &mut Environment {
        &mut self.env
    }

    /// Get the macro registry
    pub fn registry(&self) -> &MacroRegistry {
        &self.registry
    }

    /// Get a mutable reference to the macro registry
    pub fn registry_mut(&mut self) -> &mut MacroRegistry {
        &mut self.registry
    }

    /// Get accumulated trace output
    pub fn trace_output(&self) -> &[String] {
        &self.trace_output
    }

    /// Take accumulated trace output, draining the internal buffer
    pub fn take_trace_output(&mut self) -> Vec<String> {
        std::mem::take(&mut self.trace_output)
    }

    /// Define a variable in the current environment scope
    pub fn define_variable(&mut self, name: &str, value: MacroValue) {
        self.env.define(name, value);
    }

    /// Evaluate an expression, returning a MacroValue
    pub fn eval_expr(&mut self, expr: &Expr) -> Result<MacroValue, MacroError> {
        let location = span_to_location(expr.span);

        match &expr.kind {
            // --- Literals ---
            ExprKind::Int(i) => Ok(MacroValue::Int(*i)),
            ExprKind::Float(f) => Ok(MacroValue::Float(*f)),
            ExprKind::String(s) => Ok(MacroValue::String(s.clone())),
            ExprKind::Bool(b) => Ok(MacroValue::Bool(*b)),
            ExprKind::Null => Ok(MacroValue::Null),
            ExprKind::This => self
                .env
                .get("this")
                .cloned()
                .ok_or(MacroError::UndefinedVariable {
                    name: "this".to_string(),
                    location,
                }),

            // --- Identifiers ---
            ExprKind::Ident(name) => {
                self.env
                    .get(name)
                    .cloned()
                    .ok_or(MacroError::UndefinedVariable {
                        name: name.clone(),
                        location,
                    })
            }

            // --- Variable declarations ---
            ExprKind::Var {
                name, expr: init, ..
            } => {
                let value = if let Some(init_expr) = init {
                    self.eval_expr(init_expr)?
                } else {
                    MacroValue::Null
                };
                self.env.define(name, value.clone());
                Ok(value)
            }

            ExprKind::Final {
                name, expr: init, ..
            } => {
                let value = if let Some(init_expr) = init {
                    self.eval_expr(init_expr)?
                } else {
                    MacroValue::Null
                };
                self.env.define(name, value.clone());
                Ok(value)
            }

            // --- Assignment ---
            ExprKind::Assign { left, op, right } => self.eval_assignment(left, op, right),

            // --- Binary operations ---
            ExprKind::Binary { left, op, right } => {
                // Short-circuit for logical operators
                if *op == BinaryOp::And {
                    let left_val = self.eval_expr(left)?;
                    if !left_val.is_truthy() {
                        return Ok(MacroValue::Bool(false));
                    }
                    let right_val = self.eval_expr(right)?;
                    return Ok(MacroValue::Bool(right_val.is_truthy()));
                }
                if *op == BinaryOp::Or {
                    let left_val = self.eval_expr(left)?;
                    if left_val.is_truthy() {
                        return Ok(MacroValue::Bool(true));
                    }
                    let right_val = self.eval_expr(right)?;
                    return Ok(MacroValue::Bool(right_val.is_truthy()));
                }

                let left_val = self.eval_expr(left)?;
                let right_val = self.eval_expr(right)?;
                ast_bridge::apply_binary_op(op, &left_val, &right_val, location)
            }

            // --- Unary operations ---
            ExprKind::Unary { op, expr: inner } => {
                let val = self.eval_expr(inner)?;
                self.apply_unary_op(op, &val, location)
            }

            // --- Ternary ---
            ExprKind::Ternary {
                cond,
                then_expr,
                else_expr,
            } => {
                let cond_val = self.eval_expr(cond)?;
                if cond_val.is_truthy() {
                    self.eval_expr(then_expr)
                } else {
                    self.eval_expr(else_expr)
                }
            }

            // --- Parenthesized expression ---
            ExprKind::Paren(inner) => self.eval_expr(inner),
            ExprKind::Tuple(_) => Err(MacroError::RuntimeError {
                message: "Tuple literals not supported in macro context".to_string(),
                location: SourceLocation::unknown(),
            }),

            // --- Block ---
            ExprKind::Block(elements) => self.eval_block(elements),

            // --- If/else ---
            ExprKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let cond_val = self.eval_expr(cond)?;
                if cond_val.is_truthy() {
                    self.eval_expr(then_branch)
                } else if let Some(else_br) = else_branch {
                    self.eval_expr(else_br)
                } else {
                    Ok(MacroValue::Null)
                }
            }

            // --- While loop ---
            ExprKind::While { cond, body } => {
                loop {
                    let cond_val = self.eval_expr(cond)?;
                    if !cond_val.is_truthy() {
                        break;
                    }
                    match self.eval_expr(body) {
                        Err(MacroError::Break) => break,
                        Err(MacroError::Continue) => continue,
                        Err(e) => return Err(e),
                        Ok(_) => {}
                    }
                }
                Ok(MacroValue::Null)
            }

            // --- Do-while loop ---
            ExprKind::DoWhile { body, cond } => {
                loop {
                    match self.eval_expr(body) {
                        Err(MacroError::Break) => break,
                        Err(MacroError::Continue) => {}
                        Err(e) => return Err(e),
                        Ok(_) => {}
                    }
                    let cond_val = self.eval_expr(cond)?;
                    if !cond_val.is_truthy() {
                        break;
                    }
                }
                Ok(MacroValue::Null)
            }

            // --- For loop (Haxe: for (x in iter)) ---
            ExprKind::For {
                var, iter, body, ..
            } => {
                let iter_val = self.eval_expr(iter)?;
                let items = self.get_iterable(&iter_val, location)?;

                self.env.push_scope();
                for item in items {
                    self.env.define(var, item);
                    match self.eval_expr(body) {
                        Err(MacroError::Break) => break,
                        Err(MacroError::Continue) => continue,
                        Err(e) => {
                            self.env.pop_scope();
                            return Err(e);
                        }
                        Ok(_) => {}
                    }
                }
                self.env.pop_scope();
                Ok(MacroValue::Null)
            }

            // --- Switch ---
            ExprKind::Switch {
                expr: switch_expr,
                cases,
                default,
            } => {
                let val = self.eval_expr(switch_expr)?;
                for case in cases {
                    for pattern in &case.patterns {
                        if self.match_pattern(&val, pattern)? {
                            // Check guard
                            if let Some(guard) = &case.guard {
                                let guard_val = self.eval_expr(guard)?;
                                if !guard_val.is_truthy() {
                                    continue;
                                }
                            }
                            return self.eval_expr(&case.body);
                        }
                    }
                }
                if let Some(def) = default {
                    self.eval_expr(def)
                } else {
                    Ok(MacroValue::Null)
                }
            }

            // --- Return ---
            ExprKind::Return(value) => {
                let ret_val = if let Some(expr) = value {
                    self.eval_expr(expr)?
                } else {
                    MacroValue::Null
                };
                Err(MacroError::Return {
                    value: Some(Box::new(ret_val)),
                })
            }

            // --- Break / Continue ---
            ExprKind::Break => Err(MacroError::Break),
            ExprKind::Continue => Err(MacroError::Continue),

            // --- Throw ---
            ExprKind::Throw(inner) => {
                let val = self.eval_expr(inner)?;
                Err(MacroError::RuntimeError {
                    message: format!("uncaught exception: {}", val.to_display_string()),
                    location,
                })
            }

            // --- Try/catch ---
            ExprKind::Try {
                expr: try_expr,
                catches,
                finally_block,
            } => {
                let result = self.eval_expr(try_expr);
                match result {
                    Ok(val) => {
                        if let Some(finally) = finally_block {
                            self.eval_expr(finally)?;
                        }
                        Ok(val)
                    }
                    Err(e) if !e.is_control_flow() => {
                        // Try to match a catch block (use first matching catch)
                        let err_val = MacroValue::String(e.to_string());
                        if let Some(catch) = catches.first() {
                            self.env.push_scope();
                            self.env.define(&catch.var, err_val);
                            let catch_result = self.eval_expr(&catch.body);
                            self.env.pop_scope();

                            if let Some(finally) = finally_block {
                                self.eval_expr(finally)?;
                            }
                            catch_result
                        } else {
                            if let Some(finally) = finally_block {
                                self.eval_expr(finally)?;
                            }
                            Err(e)
                        }
                    }
                    Err(e) => {
                        // Control flow errors propagate through try/catch
                        if let Some(finally) = finally_block {
                            let _ = self.eval_expr(finally);
                        }
                        Err(e)
                    }
                }
            }

            // --- Function call ---
            ExprKind::Call { expr: callee, args } => self.eval_call(callee, args, location),

            // --- Field access ---
            ExprKind::Field {
                expr: base, field, ..
            } => {
                let base_val = self.eval_expr(base)?;
                self.field_access(&base_val, field, location)
            }

            // --- Array indexing ---
            ExprKind::Index { expr: base, index } => {
                let base_val = self.eval_expr(base)?;
                let idx_val = self.eval_expr(index)?;
                self.index_access(&base_val, &idx_val, location)
            }

            // --- Array literal ---
            ExprKind::Array(elements) => {
                let mut values = Vec::with_capacity(elements.len());
                for elem in elements {
                    values.push(self.eval_expr(elem)?);
                }
                Ok(MacroValue::Array(values))
            }

            // --- Object literal ---
            ExprKind::Object(fields) => {
                let mut map = std::collections::HashMap::new();
                for field in fields {
                    let val = self.eval_expr(&field.expr)?;
                    map.insert(field.name.clone(), val);
                }
                Ok(MacroValue::Object(map))
            }

            // --- Map literal ---
            ExprKind::Map(entries) => {
                let mut map = std::collections::HashMap::new();
                for (key, value) in entries {
                    let k = self.eval_expr(key)?;
                    let v = self.eval_expr(value)?;
                    let key_str = k.to_display_string();
                    map.insert(key_str, v);
                }
                Ok(MacroValue::Object(map))
            }

            // --- Function literal ---
            ExprKind::Function(func) => {
                let params: Vec<MacroParam> = func
                    .params
                    .iter()
                    .map(|p| MacroParam::from_function_param(p))
                    .collect();

                let body = func.body.as_ref().cloned().unwrap_or(Box::new(Expr {
                    kind: ExprKind::Null,
                    span: Span::default(),
                }));

                Ok(MacroValue::Function(MacroFunction {
                    name: func.name.clone(),
                    params,
                    body,
                    captures: self.env.capture_all(),
                }))
            }

            // --- Arrow function ---
            ExprKind::Arrow { params, expr: body } => {
                let macro_params: Vec<MacroParam> = params
                    .iter()
                    .map(|p| MacroParam {
                        name: p.name.clone(),
                        optional: false,
                        rest: false,
                        default_value: None,
                    })
                    .collect();

                Ok(MacroValue::Function(MacroFunction {
                    name: String::new(),
                    params: macro_params,
                    body: body.clone(),
                    captures: self.env.capture_all(),
                }))
            }

            // --- New object construction ---
            ExprKind::New {
                type_path, args, ..
            } => {
                let name = if type_path.package.is_empty() {
                    type_path.name.clone()
                } else {
                    format!("{}.{}", type_path.package.join("."), type_path.name)
                };

                let mut arg_vals = Vec::with_capacity(args.len());
                for arg in args {
                    arg_vals.push(self.eval_expr(arg)?);
                }

                // Special handling for known types
                match name.as_str() {
                    "Map" | "haxe.ds.Map" => {
                        Ok(MacroValue::Object(std::collections::HashMap::new()))
                    }
                    "Array" => Ok(MacroValue::Array(Vec::new())),
                    _ => {
                        // Generic object construction
                        let mut obj = std::collections::HashMap::new();
                        obj.insert("__type__".to_string(), MacroValue::String(name));
                        obj.insert("__args__".to_string(), MacroValue::Array(arg_vals));
                        Ok(MacroValue::Object(obj))
                    }
                }
            }

            // --- String interpolation ---
            ExprKind::StringInterpolation(parts) => {
                let mut result = String::new();
                for part in parts {
                    match part {
                        parser::StringPart::Literal(s) => result.push_str(s),
                        parser::StringPart::Interpolation(expr) => {
                            let val = self.eval_expr(expr)?;
                            result.push_str(&val.to_display_string());
                        }
                    }
                }
                Ok(MacroValue::String(result))
            }

            // --- Cast ---
            ExprKind::Cast { expr: inner, .. } => {
                // In macro context, casts are no-ops
                self.eval_expr(inner)
            }

            // --- Type check ---
            ExprKind::TypeCheck { expr: inner, .. } => {
                // In macro context, type checks just evaluate the expression
                self.eval_expr(inner)
            }

            // --- Macro expression ---
            ExprKind::Macro(inner) => {
                // macro expr — the expression should be reified into an Expr value
                Ok(MacroValue::Expr(inner.clone()))
            }

            // --- Reification block ---
            ExprKind::Reify(inner) => {
                // macro { ... } — reify the expression
                ReificationEngine::reify_expr(inner, &self.env)
            }

            // --- Dollar identifier ---
            ExprKind::DollarIdent { name, arg } => {
                if let Some(arg_expr) = arg {
                    // $kind{expr} — evaluate the arg and splice
                    let val = self.eval_expr(arg_expr)?;
                    let result_expr = ReificationEngine::splice_value(name, val, expr.span)?;
                    Ok(MacroValue::Expr(Box::new(result_expr)))
                } else {
                    // $name — lookup in environment
                    self.env
                        .get(name)
                        .cloned()
                        .ok_or(MacroError::UndefinedVariable {
                            name: format!("${}", name),
                            location,
                        })
                }
            }

            // --- Metadata expression ---
            ExprKind::Meta { expr: inner, .. } => {
                // Metadata annotations are ignored during interpretation
                self.eval_expr(inner)
            }

            // --- Untyped ---
            ExprKind::Untyped(inner) => self.eval_expr(inner),

            // --- Inline ---
            ExprKind::Inline(inner) => self.eval_expr(inner),

            // Unsupported expressions
            _ => Err(MacroError::UnsupportedOperation {
                operation: format!("expression kind: {:?}", std::mem::discriminant(&expr.kind)),
                location,
            }),
        }
    }

    /// Evaluate a block of elements, returning the last value
    fn eval_block(&mut self, elements: &[parser::BlockElement]) -> Result<MacroValue, MacroError> {
        self.env.push_scope();
        let mut last = MacroValue::Null;
        for elem in elements {
            match elem {
                parser::BlockElement::Expr(expr) => {
                    last = self.eval_expr(expr)?;
                }
                _ => {
                    // Imports/using/conditionals in blocks — skip for macro context
                }
            }
        }
        self.env.pop_scope();
        Ok(last)
    }

    /// Evaluate an assignment expression
    fn eval_assignment(
        &mut self,
        left: &Expr,
        op: &AssignOp,
        right: &Expr,
    ) -> Result<MacroValue, MacroError> {
        let location = span_to_location(left.span);
        let right_val = self.eval_expr(right)?;

        // For compound assignment, get current value and apply operation
        let new_val = if *op != AssignOp::Assign {
            let current = self.eval_expr(left)?;
            let bin_op = match op {
                AssignOp::AddAssign => BinaryOp::Add,
                AssignOp::SubAssign => BinaryOp::Sub,
                AssignOp::MulAssign => BinaryOp::Mul,
                AssignOp::DivAssign => BinaryOp::Div,
                AssignOp::ModAssign => BinaryOp::Mod,
                AssignOp::AndAssign => BinaryOp::BitAnd,
                AssignOp::OrAssign => BinaryOp::BitOr,
                AssignOp::XorAssign => BinaryOp::BitXor,
                AssignOp::ShlAssign => BinaryOp::Shl,
                AssignOp::ShrAssign => BinaryOp::Shr,
                AssignOp::UshrAssign => BinaryOp::Ushr,
                AssignOp::Assign => unreachable!(),
            };
            ast_bridge::apply_binary_op(&bin_op, &current, &right_val, location)?
        } else {
            right_val
        };

        // Assign to the target
        match &left.kind {
            ExprKind::Ident(name) => {
                if !self.env.set(name, new_val.clone()) {
                    // Variable doesn't exist yet — define it
                    self.env.define(name, new_val.clone());
                }
                Ok(new_val)
            }
            ExprKind::Field {
                expr: base, field, ..
            } => {
                let mut base_val = self.eval_expr(base)?;
                if let MacroValue::Object(ref mut map) = base_val {
                    map.insert(field.clone(), new_val.clone());
                    // Re-assign the modified object back
                    self.assign_base(base, base_val)?;
                    Ok(new_val)
                } else {
                    Err(MacroError::TypeError {
                        message: format!(
                            "cannot assign field '{}' on {}",
                            field,
                            base_val.type_name()
                        ),
                        location,
                    })
                }
            }
            ExprKind::Index { expr: base, index } => {
                let mut base_val = self.eval_expr(base)?;
                let idx = self.eval_expr(index)?;
                match (&mut base_val, &idx) {
                    (MacroValue::Array(arr), MacroValue::Int(i)) => {
                        let idx = *i as usize;
                        if idx < arr.len() {
                            arr[idx] = new_val.clone();
                        }
                        self.assign_base(base, base_val)?;
                        Ok(new_val)
                    }
                    (MacroValue::Object(map), _) => {
                        map.insert(idx.to_display_string(), new_val.clone());
                        self.assign_base(base, base_val)?;
                        Ok(new_val)
                    }
                    _ => Err(MacroError::TypeError {
                        message: "cannot index-assign on this type".to_string(),
                        location,
                    }),
                }
            }
            _ => Err(MacroError::TypeError {
                message: "invalid assignment target".to_string(),
                location,
            }),
        }
    }

    /// Re-assign a base expression after mutation
    fn assign_base(&mut self, base: &Expr, value: MacroValue) -> Result<(), MacroError> {
        match &base.kind {
            ExprKind::Ident(name) => {
                self.env.set(name, value);
                Ok(())
            }
            _ => Ok(()), // nested assignments on temporaries are discarded
        }
    }

    /// Evaluate a function call
    fn eval_call(
        &mut self,
        callee: &Expr,
        args: &[Expr],
        location: SourceLocation,
    ) -> Result<MacroValue, MacroError> {
        // Evaluate arguments
        let mut arg_vals = Vec::with_capacity(args.len());
        for arg in args {
            arg_vals.push(self.eval_expr(arg)?);
        }

        // Check for built-in functions first
        match &callee.kind {
            ExprKind::Ident(name) => {
                if let Some(result) = self.try_builtin(name, &arg_vals, location)? {
                    return Ok(result);
                }
                // Look up in environment
                if let Some(func_val) = self.env.get(name).cloned() {
                    return self.call_value(func_val, arg_vals, location);
                }
                // Look up in macro registry
                if let Some(macro_def) = self.registry.find_macro_by_name(name).cloned() {
                    return self.call_macro_def(&macro_def, arg_vals, location);
                }
                Err(MacroError::UndefinedVariable {
                    name: name.clone(),
                    location,
                })
            }
            ExprKind::Field {
                expr: base, field, ..
            } => {
                // Method call: base.field(args)
                let base_val = self.eval_expr(base)?;
                self.method_call(&base_val, field, arg_vals, location)
            }
            _ => {
                // Evaluate callee as expression (e.g., function variable)
                let func_val = self.eval_expr(callee)?;
                self.call_value(func_val, arg_vals, location)
            }
        }
    }

    /// Call a MacroValue::Function
    fn call_value(
        &mut self,
        func: MacroValue,
        args: Vec<MacroValue>,
        location: SourceLocation,
    ) -> Result<MacroValue, MacroError> {
        match func {
            MacroValue::Function(macro_func) => self.call_function(&macro_func, args, location),
            _ => Err(MacroError::TypeError {
                message: format!("cannot call {}", func.type_name()),
                location,
            }),
        }
    }

    /// Execute a MacroFunction with given arguments
    fn call_function(
        &mut self,
        func: &MacroFunction,
        args: Vec<MacroValue>,
        location: SourceLocation,
    ) -> Result<MacroValue, MacroError> {
        self.call_depth += 1;
        if self.call_depth > self.max_call_depth {
            self.call_depth -= 1;
            return Err(MacroError::RecursionLimitExceeded {
                macro_name: func.name.clone(),
                depth: self.call_depth,
                max_depth: self.max_call_depth,
                location,
            });
        }

        // Set up function scope with captures and parameters
        self.env.push_scope();

        // Restore captured variables
        for (name, value) in &func.captures {
            self.env.define(name, value.clone());
        }

        // Bind parameters
        for (i, param) in func.params.iter().enumerate() {
            let value = if let Some(arg) = args.get(i) {
                arg.clone()
            } else if param.optional {
                MacroValue::Null
            } else if let Some(default) = &param.default_value {
                self.eval_expr(default)?
            } else {
                MacroValue::Null
            };
            self.env.define(&param.name, value);
        }

        // Execute body
        let result = match self.eval_expr(&func.body) {
            Ok(val) => Ok(val),
            Err(MacroError::Return { value }) => Ok(value.map(|v| *v).unwrap_or(MacroValue::Null)),
            Err(e) => Err(e),
        };

        self.env.pop_scope();
        self.call_depth -= 1;
        result
    }

    /// Call a registered macro definition
    fn call_macro_def(
        &mut self,
        def: &super::registry::MacroDefinition,
        args: Vec<MacroValue>,
        location: SourceLocation,
    ) -> Result<MacroValue, MacroError> {
        let func = MacroFunction {
            name: def.name.clone(),
            params: def.params.clone(),
            body: def.body.clone(),
            captures: std::collections::HashMap::new(),
        };
        self.call_function(&func, args, location)
    }

    /// Try to execute a built-in function
    fn try_builtin(
        &mut self,
        name: &str,
        args: &[MacroValue],
        location: SourceLocation,
    ) -> Result<Option<MacroValue>, MacroError> {
        match name {
            "trace" => {
                let parts: Vec<String> = args.iter().map(|v| v.to_display_string()).collect();
                let msg = parts.join(", ");
                self.trace_output.push(msg);
                Ok(Some(MacroValue::Null))
            }
            _ => Ok(None),
        }
    }

    /// Call a method on a value
    fn method_call(
        &mut self,
        base: &MacroValue,
        method: &str,
        args: Vec<MacroValue>,
        location: SourceLocation,
    ) -> Result<MacroValue, MacroError> {
        // Check for static method calls (e.g., Std.string(), Math.abs())
        if let MacroValue::String(ref s) = base {
            // String methods would be handled here in a full implementation
            return self.string_method(s, method, args, location);
        }

        match base {
            MacroValue::Array(arr) => self.array_method(arr, method, args, location),
            MacroValue::Object(obj) => {
                // Check if the field is a function
                if let Some(MacroValue::Function(func)) = obj.get(method) {
                    self.call_function(func, args, location)
                } else {
                    self.object_method(obj, method, args, location)
                }
            }
            _ => {
                // Check for well-known static classes
                // The base might be an identifier that represents a class
                Err(MacroError::TypeError {
                    message: format!("cannot call method '{}' on {}", method, base.type_name()),
                    location,
                })
            }
        }
    }

    /// Built-in array methods
    fn array_method(
        &mut self,
        arr: &[MacroValue],
        method: &str,
        args: Vec<MacroValue>,
        location: SourceLocation,
    ) -> Result<MacroValue, MacroError> {
        match method {
            "length" => Ok(MacroValue::Int(arr.len() as i64)),
            "push" => {
                // Arrays in our system are immutable values — push returns a new array
                let mut new_arr = arr.to_vec();
                for arg in args {
                    new_arr.push(arg);
                }
                Ok(MacroValue::Array(new_arr))
            }
            "pop" => {
                let mut new_arr = arr.to_vec();
                let popped = new_arr.pop().unwrap_or(MacroValue::Null);
                Ok(popped)
            }
            "concat" => {
                let mut new_arr = arr.to_vec();
                for arg in args {
                    if let MacroValue::Array(other) = arg {
                        new_arr.extend(other);
                    }
                }
                Ok(MacroValue::Array(new_arr))
            }
            "join" => {
                let sep = args.first().and_then(|a| a.as_string()).unwrap_or(",");
                let parts: Vec<String> = arr.iter().map(|v| v.to_display_string()).collect();
                Ok(MacroValue::String(parts.join(sep)))
            }
            "map" => {
                if let Some(MacroValue::Function(func)) = args.first() {
                    let mut result = Vec::with_capacity(arr.len());
                    for item in arr {
                        let mapped = self.call_function(func, vec![item.clone()], location)?;
                        result.push(mapped);
                    }
                    Ok(MacroValue::Array(result))
                } else {
                    Err(MacroError::TypeError {
                        message: "Array.map() requires a function argument".to_string(),
                        location,
                    })
                }
            }
            "filter" => {
                if let Some(MacroValue::Function(func)) = args.first() {
                    let mut result = Vec::new();
                    for item in arr {
                        let keep = self.call_function(func, vec![item.clone()], location)?;
                        if keep.is_truthy() {
                            result.push(item.clone());
                        }
                    }
                    Ok(MacroValue::Array(result))
                } else {
                    Err(MacroError::TypeError {
                        message: "Array.filter() requires a function argument".to_string(),
                        location,
                    })
                }
            }
            "indexOf" => {
                let needle = args.first().unwrap_or(&MacroValue::Null);
                for (i, item) in arr.iter().enumerate() {
                    if item == needle {
                        return Ok(MacroValue::Int(i as i64));
                    }
                }
                Ok(MacroValue::Int(-1))
            }
            "contains" => {
                let needle = args.first().unwrap_or(&MacroValue::Null);
                Ok(MacroValue::Bool(arr.iter().any(|item| item == needle)))
            }
            _ => Err(MacroError::UnsupportedOperation {
                operation: format!("Array.{}()", method),
                location,
            }),
        }
    }

    /// Built-in string methods
    fn string_method(
        &self,
        s: &str,
        method: &str,
        args: Vec<MacroValue>,
        location: SourceLocation,
    ) -> Result<MacroValue, MacroError> {
        match method {
            "length" => Ok(MacroValue::Int(s.len() as i64)),
            "charAt" => {
                let idx = args.first().and_then(|a| a.as_int()).unwrap_or(0) as usize;
                Ok(MacroValue::String(
                    s.chars()
                        .nth(idx)
                        .map(|c| c.to_string())
                        .unwrap_or_default(),
                ))
            }
            "indexOf" => {
                let needle = args.first().and_then(|a| a.as_string()).unwrap_or("");
                Ok(MacroValue::Int(
                    s.find(needle).map(|i| i as i64).unwrap_or(-1),
                ))
            }
            "split" => {
                let delim = args.first().and_then(|a| a.as_string()).unwrap_or("");
                let parts: Vec<MacroValue> = s
                    .split(delim)
                    .map(|part| MacroValue::String(part.to_string()))
                    .collect();
                Ok(MacroValue::Array(parts))
            }
            "substring" | "substr" => {
                let start = args.first().and_then(|a| a.as_int()).unwrap_or(0).max(0) as usize;
                let end = args
                    .get(1)
                    .and_then(|a| a.as_int())
                    .map(|e| e.max(0) as usize)
                    .unwrap_or(s.len());
                let result: String = s.chars().skip(start).take(end - start).collect();
                Ok(MacroValue::String(result))
            }
            "toUpperCase" => Ok(MacroValue::String(s.to_uppercase())),
            "toLowerCase" => Ok(MacroValue::String(s.to_lowercase())),
            "toString" => Ok(MacroValue::String(s.to_string())),
            _ => Err(MacroError::UnsupportedOperation {
                operation: format!("String.{}()", method),
                location,
            }),
        }
    }

    /// Built-in object methods
    fn object_method(
        &self,
        _obj: &std::collections::HashMap<String, MacroValue>,
        method: &str,
        _args: Vec<MacroValue>,
        location: SourceLocation,
    ) -> Result<MacroValue, MacroError> {
        Err(MacroError::UnsupportedOperation {
            operation: format!("Object.{}()", method),
            location,
        })
    }

    /// Access a field on a value
    fn field_access(
        &self,
        base: &MacroValue,
        field: &str,
        location: SourceLocation,
    ) -> Result<MacroValue, MacroError> {
        match base {
            MacroValue::Object(map) => Ok(map.get(field).cloned().unwrap_or(MacroValue::Null)),
            MacroValue::Array(arr) => match field {
                "length" => Ok(MacroValue::Int(arr.len() as i64)),
                _ => Err(MacroError::TypeError {
                    message: format!("Array has no field '{}'", field),
                    location,
                }),
            },
            MacroValue::String(s) => match field {
                "length" => Ok(MacroValue::Int(s.len() as i64)),
                _ => Err(MacroError::TypeError {
                    message: format!("String has no field '{}'", field),
                    location,
                }),
            },
            MacroValue::Expr(expr) => {
                // Field access on reified expressions (e.g., expr.expr, expr.pos)
                match field {
                    "pos" => Ok(MacroValue::Position(span_to_location(expr.span))),
                    _ => Err(MacroError::TypeError {
                        message: format!("Expr has no field '{}'", field),
                        location,
                    }),
                }
            }
            _ => Err(MacroError::TypeError {
                message: format!("cannot access field '{}' on {}", field, base.type_name()),
                location,
            }),
        }
    }

    /// Access an array/map element by index
    fn index_access(
        &self,
        base: &MacroValue,
        index: &MacroValue,
        location: SourceLocation,
    ) -> Result<MacroValue, MacroError> {
        match (base, index) {
            (MacroValue::Array(arr), MacroValue::Int(i)) => {
                let idx = *i as usize;
                Ok(arr.get(idx).cloned().unwrap_or(MacroValue::Null))
            }
            (MacroValue::Object(map), MacroValue::String(key)) => {
                Ok(map.get(key.as_str()).cloned().unwrap_or(MacroValue::Null))
            }
            (MacroValue::Object(map), other) => {
                let key = other.to_display_string();
                Ok(map.get(&key).cloned().unwrap_or(MacroValue::Null))
            }
            (MacroValue::String(s), MacroValue::Int(i)) => {
                let idx = *i as usize;
                Ok(MacroValue::String(
                    s.chars()
                        .nth(idx)
                        .map(|c| c.to_string())
                        .unwrap_or_default(),
                ))
            }
            _ => Err(MacroError::TypeError {
                message: format!(
                    "cannot index {} with {}",
                    base.type_name(),
                    index.type_name()
                ),
                location,
            }),
        }
    }

    /// Apply a unary operator
    fn apply_unary_op(
        &self,
        op: &UnaryOp,
        val: &MacroValue,
        location: SourceLocation,
    ) -> Result<MacroValue, MacroError> {
        match op {
            UnaryOp::Not => Ok(MacroValue::Bool(!val.is_truthy())),
            UnaryOp::Neg => match val {
                MacroValue::Int(i) => Ok(MacroValue::Int(-i)),
                MacroValue::Float(f) => Ok(MacroValue::Float(-f)),
                _ => Err(MacroError::TypeError {
                    message: format!("cannot negate {}", val.type_name()),
                    location,
                }),
            },
            UnaryOp::BitNot => match val {
                MacroValue::Int(i) => Ok(MacroValue::Int(!i)),
                _ => Err(MacroError::TypeError {
                    message: format!("cannot bit-not {}", val.type_name()),
                    location,
                }),
            },
            UnaryOp::PreIncr | UnaryOp::PostIncr => match val {
                MacroValue::Int(i) => Ok(MacroValue::Int(i + 1)),
                MacroValue::Float(f) => Ok(MacroValue::Float(f + 1.0)),
                _ => Err(MacroError::TypeError {
                    message: format!("cannot increment {}", val.type_name()),
                    location,
                }),
            },
            UnaryOp::PreDecr | UnaryOp::PostDecr => match val {
                MacroValue::Int(i) => Ok(MacroValue::Int(i - 1)),
                MacroValue::Float(f) => Ok(MacroValue::Float(f - 1.0)),
                _ => Err(MacroError::TypeError {
                    message: format!("cannot decrement {}", val.type_name()),
                    location,
                }),
            },
        }
    }

    /// Get an iterable sequence from a value
    fn get_iterable(
        &self,
        value: &MacroValue,
        location: SourceLocation,
    ) -> Result<Vec<MacroValue>, MacroError> {
        match value {
            MacroValue::Array(arr) => Ok(arr.clone()),
            MacroValue::Object(map) => {
                // Iterate over keys
                Ok(map.keys().map(|k| MacroValue::String(k.clone())).collect())
            }
            _ => Err(MacroError::TypeError {
                message: format!("cannot iterate over {}", value.type_name()),
                location,
            }),
        }
    }

    /// Match a value against a pattern (for switch cases)
    fn match_pattern(
        &mut self,
        value: &MacroValue,
        pattern: &parser::Pattern,
    ) -> Result<bool, MacroError> {
        match pattern {
            parser::Pattern::Const(expr) => {
                let pattern_val = self.eval_expr(expr)?;
                Ok(value == &pattern_val)
            }
            parser::Pattern::Var(name) => {
                // Variable pattern always matches, binding the value
                self.env.define(name, value.clone());
                Ok(true)
            }
            parser::Pattern::Underscore => Ok(true),
            _ => {
                // Constructor patterns, or patterns etc. — simplified matching
                Ok(false)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eval(source: &str) -> Result<MacroValue, MacroError> {
        let full_source = format!("class Test {{ static function main() {{ {} }} }}", source);
        let file =
            parser::parse_haxe_file("test.hx", &full_source, false).expect("parse should succeed");

        let registry = MacroRegistry::new();
        let mut interp = MacroInterpreter::new(registry);

        // Find the main function body
        if let Some(decl) = file.declarations.first() {
            if let parser::TypeDeclaration::Class(class) = decl {
                for field in &class.fields {
                    if let parser::ClassFieldKind::Function(func) = &field.kind {
                        if func.name == "main" {
                            if let Some(body) = &func.body {
                                return match interp.eval_expr(body) {
                                    Ok(val) => Ok(val),
                                    // Unwrap return control flow into a normal result
                                    Err(MacroError::Return { value }) => {
                                        Ok(value.map(|v| *v).unwrap_or(MacroValue::Null))
                                    }
                                    Err(e) => Err(e),
                                };
                            }
                        }
                    }
                }
            }
        }
        Ok(MacroValue::Null)
    }

    #[test]
    fn test_literal_eval() {
        assert_eq!(eval("return 42;").unwrap(), MacroValue::Int(42));
        assert_eq!(
            eval("return \"hello\";").unwrap(),
            MacroValue::String("hello".to_string())
        );
        assert_eq!(eval("return true;").unwrap(), MacroValue::Bool(true));
        assert_eq!(eval("return null;").unwrap(), MacroValue::Null);
    }

    #[test]
    fn test_arithmetic() {
        assert_eq!(eval("return 2 + 3;").unwrap(), MacroValue::Int(5));
        assert_eq!(eval("return 10 - 4;").unwrap(), MacroValue::Int(6));
        assert_eq!(eval("return 3 * 7;").unwrap(), MacroValue::Int(21));
        assert_eq!(eval("return 15 / 3;").unwrap(), MacroValue::Int(5));
        assert_eq!(eval("return 10 % 3;").unwrap(), MacroValue::Int(1));
    }

    #[test]
    fn test_string_concat() {
        assert_eq!(
            eval("return \"hello\" + \" world\";").unwrap(),
            MacroValue::String("hello world".to_string())
        );
    }

    #[test]
    fn test_comparison() {
        assert_eq!(eval("return 1 < 2;").unwrap(), MacroValue::Bool(true));
        assert_eq!(eval("return 2 > 1;").unwrap(), MacroValue::Bool(true));
        assert_eq!(eval("return 1 == 1;").unwrap(), MacroValue::Bool(true));
        assert_eq!(eval("return 1 != 2;").unwrap(), MacroValue::Bool(true));
    }

    #[test]
    fn test_variable_declaration() {
        assert_eq!(eval("var x = 42; return x;").unwrap(), MacroValue::Int(42));
    }

    #[test]
    fn test_if_else() {
        assert_eq!(
            eval("if (true) return 1; else return 2;").unwrap(),
            MacroValue::Int(1)
        );
        assert_eq!(
            eval("if (false) return 1; else return 2;").unwrap(),
            MacroValue::Int(2)
        );
    }

    #[test]
    fn test_while_loop() {
        assert_eq!(
            eval("var x = 0; while (x < 5) { x = x + 1; } return x;").unwrap(),
            MacroValue::Int(5)
        );
    }

    #[test]
    fn test_array_literal() {
        let result = eval("return [1, 2, 3];").unwrap();
        match result {
            MacroValue::Array(arr) => {
                assert_eq!(arr.len(), 3);
                assert_eq!(arr[0], MacroValue::Int(1));
                assert_eq!(arr[2], MacroValue::Int(3));
            }
            _ => panic!("expected Array"),
        }
    }

    #[test]
    fn test_array_index() {
        assert_eq!(
            eval("var arr = [10, 20, 30]; return arr[1];").unwrap(),
            MacroValue::Int(20)
        );
    }

    #[test]
    fn test_object_literal() {
        let result = eval("return { x: 1, y: 2 };").unwrap();
        match result {
            MacroValue::Object(map) => {
                assert_eq!(map.get("x"), Some(&MacroValue::Int(1)));
                assert_eq!(map.get("y"), Some(&MacroValue::Int(2)));
            }
            _ => panic!("expected Object"),
        }
    }

    #[test]
    fn test_field_access() {
        assert_eq!(
            eval("var obj = { x: 42 }; return obj.x;").unwrap(),
            MacroValue::Int(42)
        );
    }

    #[test]
    fn test_trace() {
        let full_source =
            "class Test { static function main() { trace(\"hello\"); return null; } }";
        let file =
            parser::parse_haxe_file("test.hx", full_source, false).expect("parse should succeed");

        let registry = MacroRegistry::new();
        let mut interp = MacroInterpreter::new(registry);

        if let Some(parser::TypeDeclaration::Class(class)) = file.declarations.first() {
            for field in &class.fields {
                if let parser::ClassFieldKind::Function(func) = &field.kind {
                    if let Some(body) = &func.body {
                        let _ = interp.eval_expr(body);
                    }
                }
            }
        }

        assert_eq!(interp.trace_output(), &["hello"]);
    }

    #[test]
    fn test_function_literal() {
        assert_eq!(
            eval("var add = function(a, b) { return a + b; }; return add(3, 4);").unwrap(),
            MacroValue::Int(7)
        );
    }

    #[test]
    fn test_logical_operators() {
        assert_eq!(
            eval("return true && false;").unwrap(),
            MacroValue::Bool(false)
        );
        assert_eq!(
            eval("return false || true;").unwrap(),
            MacroValue::Bool(true)
        );
        assert_eq!(eval("return !true;").unwrap(), MacroValue::Bool(false));
    }

    #[test]
    fn test_break_in_while() {
        assert_eq!(
            eval("var x = 0; while (true) { x = x + 1; if (x >= 3) break; } return x;").unwrap(),
            MacroValue::Int(3)
        );
    }

    #[test]
    fn test_for_loop() {
        assert_eq!(
            eval("var sum = 0; for (x in [1, 2, 3]) { sum = sum + x; } return sum;").unwrap(),
            MacroValue::Int(6)
        );
    }

    #[test]
    fn test_ternary() {
        assert_eq!(eval("return true ? 1 : 2;").unwrap(), MacroValue::Int(1));
        assert_eq!(eval("return false ? 1 : 2;").unwrap(), MacroValue::Int(2));
    }

    #[test]
    fn test_compound_assignment() {
        assert_eq!(
            eval("var x = 10; x += 5; return x;").unwrap(),
            MacroValue::Int(15)
        );
        assert_eq!(
            eval("var x = 10; x -= 3; return x;").unwrap(),
            MacroValue::Int(7)
        );
    }

    // ===== Edge case tests (Phase 7) =====

    #[test]
    fn test_nested_if_else() {
        assert_eq!(
            eval("var x = 10; if (x > 5) { if (x > 8) { return 1; } else { return 2; } } else { return 3; }").unwrap(),
            MacroValue::Int(1)
        );
    }

    #[test]
    fn test_try_catch_recovery() {
        // Try/catch should catch runtime errors
        let source = r#"
            try {
                var x = 1 / 0;
                return x;
            } catch (e) {
                return -1;
            }
        "#;
        let result = eval(source);
        assert_eq!(result.unwrap(), MacroValue::Int(-1));
    }

    #[test]
    fn test_division_by_zero_error() {
        let result = eval("return 10 / 0;");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(!err.is_control_flow());
    }

    #[test]
    fn test_undefined_variable_error() {
        let result = eval("return nonexistent;");
        assert!(result.is_err());
        match result.unwrap_err() {
            MacroError::UndefinedVariable { name, .. } => {
                assert_eq!(name, "nonexistent");
            }
            other => panic!("expected UndefinedVariable, got {:?}", other),
        }
    }

    #[test]
    fn test_nested_array_access() {
        assert_eq!(
            eval("var arr = [[1, 2], [3, 4]]; return arr[1][0];").unwrap(),
            MacroValue::Int(3)
        );
    }

    #[test]
    fn test_string_comparison() {
        assert_eq!(
            eval("return \"abc\" == \"abc\";").unwrap(),
            MacroValue::Bool(true)
        );
        assert_eq!(
            eval("return \"abc\" == \"def\";").unwrap(),
            MacroValue::Bool(false)
        );
    }

    #[test]
    fn test_null_handling() {
        assert_eq!(eval("return null;").unwrap(), MacroValue::Null);
        assert_eq!(
            eval("return null == null;").unwrap(),
            MacroValue::Bool(true)
        );
    }

    #[test]
    fn test_boolean_negation() {
        assert_eq!(eval("return !true;").unwrap(), MacroValue::Bool(false));
        assert_eq!(eval("return !false;").unwrap(), MacroValue::Bool(true));
    }

    #[test]
    fn test_numeric_negation() {
        assert_eq!(eval("return -42;").unwrap(), MacroValue::Int(-42));
    }

    #[test]
    fn test_sequential_returns() {
        // Only the first return should take effect
        assert_eq!(
            eval("var x = 0; x = 1; return x;").unwrap(),
            MacroValue::Int(1)
        );
    }

    #[test]
    fn test_while_break() {
        assert_eq!(
            eval("var i = 0; while (true) { i += 1; if (i >= 5) { break; } } return i;").unwrap(),
            MacroValue::Int(5)
        );
    }

    #[test]
    fn test_while_continue() {
        assert_eq!(
            eval("var sum = 0; var i = 0; while (i < 10) { i += 1; if (i % 2 == 0) { continue; } sum += i; } return sum;").unwrap(),
            MacroValue::Int(25) // 1+3+5+7+9
        );
    }

    #[test]
    fn test_float_arithmetic() {
        assert_eq!(eval("return 3.14 * 2.0;").unwrap(), MacroValue::Float(6.28));
    }

    #[test]
    fn test_mixed_int_float_arithmetic() {
        // When mixing int and float, result should be float
        let result = eval("return 3 + 0.14;").unwrap();
        match result {
            MacroValue::Float(f) => assert!((f - 3.14).abs() < 0.001),
            other => panic!("expected Float, got {:?}", other),
        }
    }
}
