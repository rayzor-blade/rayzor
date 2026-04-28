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
use super::bytecode::{BytecodeCompiler, MacroVm};
use super::class_registry::ClassRegistry;
use super::environment::Environment;
use super::errors::MacroError;
use super::registry::MacroRegistry;
use super::reification::ReificationEngine;
use super::value::{MacroFunction, MacroParam, MacroValue};
use crate::tast::SourceLocation;
use parser::{AssignOp, BinaryOp, Expr, ExprKind, Span, UnaryOp};
use std::collections::BTreeMap;
use std::sync::Arc;

/// Maximum call depth for macro function calls
const DEFAULT_MAX_CALL_DEPTH: usize = 256;

/// Morsel-parallelism-inspired tiering scheduler.
///
/// Tracks per-macro call counts and promotes hot macros to bytecode.
/// A "morsel" is a macro + its transitive class dependencies — when a macro
/// crosses the call threshold, the scheduler batch-compiles the macro and
/// all classes it might use (the full connected subgraph).
///
/// Cold macros (called < threshold times) never pay compilation cost.
struct MorselScheduler {
    /// Per-macro call counts (keyed by qualified name).
    call_counts: BTreeMap<String, u32>,
    /// Promotion threshold: compile after this many tree-walker calls.
    threshold: u32,
    /// Whether class chunks have been compiled and transferred to the VM.
    classes_compiled: bool,
}

/// Cached data extracted from ClassRegistry for a user-defined class.
/// Avoids re-cloning instance_vars, constructor body, and params on every `new`.
struct CachedClassData {
    qualified_name: String,
    instance_vars: Vec<(String, Option<Arc<Expr>>)>,
    constructor: Option<(Arc<Expr>, Vec<parser::FunctionParam>)>,
}

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
    /// Import map: short class name → fully qualified name
    /// e.g., "Context" → "haxe.macro.Context"
    import_map: BTreeMap<String, String>,
    /// Class registry for fallback dispatch to any imported/user class
    class_registry: Option<Arc<ClassRegistry>>,
    /// Cache of extracted class data for constructor calls (avoids re-cloning)
    class_data_cache: BTreeMap<String, Arc<CachedClassData>>,
    /// Bytecode VM instance (created when RAYZOR_MACRO_VM=1).
    vm: Option<MacroVm>,
    /// Morsel scheduler for tiered compilation (only active when VM is present).
    scheduler: Option<MorselScheduler>,
    /// Macro context for @:build macros. When set, `Context.getBuildFields()`
    /// and `Context.getLocalClass()` return the class being built. Without
    /// this, those APIs return empty/null.
    pub macro_context: Option<super::context_api::MacroContext>,
    /// Stack of classes whose macros are currently executing.
    /// Bare identifier calls inside a macro body (e.g. `parseValue(...)`
    /// inside `tink.Json.parse`) fall back to this class's static methods
    /// when not found as builtins, env vars, or registered macros.
    /// Stack so nested macro calls restore the outer class on return.
    macro_class_stack: Vec<String>,
}

impl MacroInterpreter {
    /// Create a new scheduler+VM pair if bytecode tiering is enabled.
    fn make_vm_and_scheduler() -> (Option<MacroVm>, Option<MorselScheduler>) {
        if std::env::var("RAYZOR_MACRO_VM").is_ok() {
            let threshold = std::env::var("RAYZOR_MACRO_VM_THRESHOLD")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(2u32);
            (
                Some(MacroVm::new()),
                Some(MorselScheduler {
                    call_counts: BTreeMap::new(),
                    threshold,
                    classes_compiled: false,
                }),
            )
        } else {
            (None, None)
        }
    }

    pub fn new(registry: MacroRegistry) -> Self {
        let (vm, scheduler) = Self::make_vm_and_scheduler();
        Self {
            env: Environment::new(),
            registry,
            call_depth: 0,
            max_call_depth: DEFAULT_MAX_CALL_DEPTH,
            trace_output: Vec::new(),
            import_map: BTreeMap::new(),
            class_registry: None,
            class_data_cache: BTreeMap::new(),
            vm,
            scheduler,
            macro_context: None,
            macro_class_stack: Vec::new(),
        }
    }

    /// Create an interpreter with import mappings from the source file.
    pub fn with_imports(registry: MacroRegistry, import_map: BTreeMap<String, String>) -> Self {
        let (vm, scheduler) = Self::make_vm_and_scheduler();
        Self {
            env: Environment::new(),
            registry,
            call_depth: 0,
            max_call_depth: DEFAULT_MAX_CALL_DEPTH,
            trace_output: Vec::new(),
            import_map,
            class_registry: None,
            class_data_cache: BTreeMap::new(),
            vm,
            scheduler,
            macro_context: None,
            macro_class_stack: Vec::new(),
        }
    }

    /// Create an interpreter with import mappings and a class registry for extended dispatch.
    /// Class compilation is deferred — the morsel scheduler compiles classes on-demand
    /// when the first macro crosses the call-count threshold.
    pub fn with_class_registry(
        registry: MacroRegistry,
        import_map: BTreeMap<String, String>,
        class_registry: Arc<ClassRegistry>,
    ) -> Self {
        let (vm, scheduler) = Self::make_vm_and_scheduler();
        Self {
            env: Environment::new(),
            registry,
            call_depth: 0,
            max_call_depth: DEFAULT_MAX_CALL_DEPTH,
            trace_output: Vec::new(),
            import_map,
            class_registry: Some(class_registry),
            class_data_cache: BTreeMap::new(),
            vm,
            scheduler,
            macro_context: None,
            macro_class_stack: Vec::new(),
        }
    }

    /// Push the name of the class whose macro is about to execute.
    /// See `macro_class_stack` on the struct for why this exists.
    pub fn push_macro_class(&mut self, class_name: String) {
        self.macro_class_stack.push(class_name);
    }

    /// Pop the most recently pushed macro class context.
    pub fn pop_macro_class(&mut self) -> Option<String> {
        self.macro_class_stack.pop()
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
            ExprKind::String(s) => Ok(MacroValue::String(Arc::from(s.as_str()))),
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
                if let Some(v) = self.env.get(name).cloned() {
                    return Ok(v);
                }
                // Bare enum-constructor identifiers from haxe.macro.Expr —
                // `APublic`, `AInline`, etc. — appear unbound in build macros
                // because the interpreter doesn't load enum declarations.
                // Map them to plain strings so `value_to_class_field` (which
                // matches on `"Public" | "Static" | ...`) accepts either
                // constructor form.
                if let Some(stripped) = enum_ident_as_string(name) {
                    return Ok(MacroValue::String(Arc::from(stripped)));
                }
                Err(MacroError::UndefinedVariable {
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
            ExprKind::Unary { op, expr: inner } => match op {
                // Pre/post inc/dec need lvalue access: compute the new value,
                // then write it back through the same target machinery
                // `eval_assignment` uses. Evaluating `inner` once and handing
                // it to `apply_unary_op` silently drops the store, so `x++`
                // becomes a no-op and loops like `while (i<n) i++` never
                // terminate.
                UnaryOp::PreIncr | UnaryOp::PostIncr | UnaryOp::PreDecr | UnaryOp::PostDecr => {
                    self.eval_inc_dec(op, inner, location)
                }
                _ => {
                    let val = self.eval_expr(inner)?;
                    self.apply_unary_op(op, &val, location)
                }
            },

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
                        let err_val = MacroValue::String(Arc::from(e.to_string().as_str()));
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
                Ok(MacroValue::Array(Arc::new(values)))
            }

            // --- Object literal ---
            ExprKind::Object(fields) => {
                let mut map = std::collections::BTreeMap::new();
                for field in fields {
                    let val = self.eval_expr(&field.expr)?;
                    map.insert(field.name.clone(), val);
                }
                Ok(MacroValue::Object(Arc::new(map)))
            }

            // --- Map literal ---
            ExprKind::Map(entries) => {
                let mut map = std::collections::BTreeMap::new();
                for (key, value) in entries {
                    let k = self.eval_expr(key)?;
                    let v = self.eval_expr(value)?;
                    let key_str = k.to_display_string();
                    map.insert(key_str, v);
                }
                Ok(MacroValue::Object(Arc::new(map)))
            }

            // --- Function literal ---
            ExprKind::Function(func) => {
                let params: Vec<MacroParam> = func
                    .params
                    .iter()
                    .map(|p| MacroParam::from_function_param(p))
                    .collect();

                let body: Arc<Expr> = func
                    .body
                    .as_ref()
                    .map(|b| Arc::from(b.as_ref().clone()))
                    .unwrap_or_else(|| {
                        Arc::new(Expr {
                            kind: ExprKind::Null,
                            span: Span::default(),
                        })
                    });

                let free_vars = get_free_vars_for_closure(&body, &params);
                Ok(MacroValue::Function(Arc::new(MacroFunction {
                    name: func.name.clone(),
                    params,
                    body,
                    captures: self.env.capture_used(&free_vars),
                })))
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

                let arrow_body = Arc::from(body.as_ref().clone());
                let free_vars = get_free_vars_for_closure(&arrow_body, &macro_params);
                Ok(MacroValue::Function(Arc::new(MacroFunction {
                    name: String::new(),
                    params: macro_params,
                    body: arrow_body,
                    captures: self.env.capture_used(&free_vars),
                })))
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

                // Resolve through imports (e.g., "Process" → "sys.io.Process")
                let resolved_name = self.resolve_class_name(&name);

                // Special handling for known types
                match resolved_name.as_str() {
                    "Map" | "haxe.ds.Map" => Ok(MacroValue::Object(Arc::new(
                        std::collections::BTreeMap::new(),
                    ))),
                    "Array" => Ok(MacroValue::Array(Arc::new(Vec::new()))),
                    "sys.io.Process" => self.construct_process(arg_vals, location),
                    "sys.io.File" => self.construct_file(arg_vals, location),
                    _ => {
                        // Check cache first, then ClassRegistry for user-defined constructor
                        let cached = if let Some(cached) =
                            self.class_data_cache.get(resolved_name.as_str())
                        {
                            Some(Arc::clone(cached))
                        } else {
                            // Cache miss: extract from registry and cache
                            let data = self.class_registry.as_ref().and_then(|cr| {
                                cr.find_class(&resolved_name).map(|ci| {
                                    Arc::new(CachedClassData {
                                        qualified_name: ci.qualified_name.clone(),
                                        instance_vars: ci
                                            .instance_vars
                                            .iter()
                                            .map(|v| (v.name.clone(), v.init_expr.clone()))
                                            .collect(),
                                        constructor: ci
                                            .constructor
                                            .as_ref()
                                            .map(|c| (c.body.clone(), c.params.clone())),
                                    })
                                })
                            });
                            if let Some(ref data) = data {
                                self.class_data_cache
                                    .insert(resolved_name.to_string(), Arc::clone(data));
                            }
                            data
                        };

                        if let Some(cd) = cached {
                            return self.construct_from_registry_data(
                                &cd.qualified_name,
                                &cd.instance_vars,
                                cd.constructor.as_ref(),
                                arg_vals,
                                location,
                            );
                        }

                        // Generic object construction (no class found)
                        let mut obj = std::collections::BTreeMap::new();
                        obj.insert(
                            "__type__".to_string(),
                            MacroValue::String(Arc::from(name.as_str())),
                        );
                        obj.insert(
                            "__args__".to_string(),
                            MacroValue::Array(Arc::new(arg_vals)),
                        );
                        Ok(MacroValue::Object(Arc::new(obj)))
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
                Ok(MacroValue::String(Arc::from(result.as_str())))
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
                // macro expr — reify the expression, resolving dollar-idents from the environment
                // e.g., `macro trace($e{msg})` resolves $e{msg} using the current env
                ReificationEngine::reify_expr(inner, &self.env)
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
                    Ok(MacroValue::Expr(Arc::new(result_expr)))
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
        let mut result: Result<MacroValue, MacroError> = Ok(MacroValue::Null);
        for elem in elements {
            match elem {
                parser::BlockElement::Expr(expr) => match self.eval_expr(expr) {
                    Ok(val) => result = Ok(val),
                    Err(e) => {
                        result = Err(e);
                        break;
                    }
                },
                _ => {
                    // Imports/using/conditionals in blocks — skip for macro context
                }
            }
        }
        // CRITICAL: pop the scope on EVERY exit path, not just success.
        // The earlier `let last = self.eval_expr(expr)?;` form short-circuited
        // out before reaching `pop_scope()`, leaking one block scope per
        // early-return / break / continue / throw. Over a few recursive
        // calls the scope stack grew unboundedly, and identifier lookups
        // found stale outer-scope `pos` values instead of the current
        // function's, which caused mutual-recursion macros (parseValue ↔
        // parseArray ↔ parseNumber in tink.Json) to spin forever.
        self.env.pop_scope();
        result
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
                // Fast path: this.field = value or ident.field = value
                // Mutates the object in-place in the environment, avoiding
                // eval→clone→COW→reassign overhead (O(N²) for N-field constructors)
                let base_var_name = match &base.kind {
                    ExprKind::This => Some("this"),
                    ExprKind::Ident(name) => Some(name.as_str()),
                    _ => None,
                };
                if let Some(var_name) = base_var_name {
                    if self
                        .env
                        .mutate_object_field(var_name, field, new_val.clone())
                    {
                        return Ok(new_val);
                    }
                }
                // Fallback: complex base expressions (e.g. a.b.field = value)
                let mut base_val = self.eval_expr(base)?;
                if let MacroValue::Object(ref mut arc_map) = base_val {
                    Arc::make_mut(arc_map).insert(field.clone(), new_val.clone());
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
                    (MacroValue::Array(arc_arr), MacroValue::Int(i)) => {
                        let idx = *i as usize;
                        let arr = Arc::make_mut(arc_arr);
                        if idx < arr.len() {
                            arr[idx] = new_val.clone();
                        }
                        self.assign_base(base, base_val)?;
                        Ok(new_val)
                    }
                    (MacroValue::Object(arc_map), _) => {
                        Arc::make_mut(arc_map).insert(idx.to_display_string(), new_val.clone());
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
            ExprKind::This => {
                self.env.set("this", value);
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
                // Look up as a sibling static method of the class whose macro
                // is currently executing. This lets a macro body call helper
                // functions defined in the same class (e.g. `tink.Json.parse`
                // internally calls `parseValue`).
                if let Some(class_name) = self.macro_class_stack.last().cloned() {
                    if let Some(result) =
                        self.try_static_call(&class_name, name, &arg_vals, location)?
                    {
                        return Ok(result);
                    }
                }
                // Enum-constructor style call from haxe.macro.Expr:
                // `FFun({args:[], ret:..., expr:...})` should produce
                // `{kind: "FFun", args:[], ret:..., expr:...}` so the
                // build-macro consumer (`value_to_class_field`) can read
                // the variant tag through the nested `kind` field.
                if let Some(tag) = enum_ctor_tag(name) {
                    return Ok(build_enum_ctor_value(tag, &arg_vals));
                }
                Err(MacroError::UndefinedVariable {
                    name: name.clone(),
                    location,
                })
            }
            ExprKind::Field {
                expr: base, field, ..
            } => {
                // Check for static class calls: Class.method(...) or qualified.path.Class.method(...)
                if let ExprKind::Ident(class_name) = &base.kind {
                    // Simple: Context.parse(...)
                    if let Some(result) =
                        self.try_static_call(class_name, field, &arg_vals, location)?
                    {
                        return Ok(result);
                    }
                } else if let Some((qualified_class, method_name)) =
                    extract_qualified_call(base, field)
                {
                    // Qualified: haxe.macro.Context.parse(...)
                    if let Some(result) =
                        self.try_static_call(&qualified_class, method_name, &arg_vals, location)?
                    {
                        return Ok(result);
                    }
                }

                // Method call: base.field(args)
                let base_val = self.eval_expr(base)?;
                let result = self.method_call(&base_val, field, arg_vals, location)?;

                // For mutating array methods (push, pop, splice, unshift, etc.),
                // update the variable in-place so the mutation is visible.
                let is_mutating = matches!(
                    field.as_str(),
                    "push" | "pop" | "shift" | "unshift" | "splice" | "sort" | "reverse"
                );
                if is_mutating {
                    if let ExprKind::Ident(var_name) = &base.kind {
                        if matches!(result, MacroValue::Array(_)) {
                            // Update the variable with the new array
                            if !self.env.set(var_name, result.clone()) {
                                self.env.define(var_name, result.clone());
                            }
                        }
                    }
                }

                Ok(result)
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
        match &func {
            MacroValue::Function(macro_func) => self.call_function(macro_func, args, location),
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

        // Bind parameters (use define_owned to avoid &str → to_string() allocation).
        // Default-value evaluation can fail; that error must NOT short-circuit
        // before pop_scope, or we'd leak the function scope.
        let mut bind_err: Option<MacroError> = None;
        for (i, param) in func.params.iter().enumerate() {
            let value = if let Some(arg) = args.get(i) {
                arg.clone()
            } else if param.optional {
                MacroValue::Null
            } else if let Some(default) = &param.default_value {
                match self.eval_expr(default) {
                    Ok(v) => v,
                    Err(e) => {
                        bind_err = Some(e);
                        break;
                    }
                }
            } else {
                MacroValue::Null
            };
            self.env.define_owned(param.name.clone(), value);
        }

        let result = if let Some(e) = bind_err {
            Err(e)
        } else {
            // Execute body
            match self.eval_expr(&func.body) {
                Ok(val) => Ok(val),
                Err(MacroError::Return { value }) => {
                    Ok(value.map(|v| *v).unwrap_or(MacroValue::Null))
                }
                Err(e) => Err(e),
            }
        };

        self.env.pop_scope();
        self.call_depth -= 1;
        result
    }

    /// Call a registered macro definition.
    ///
    /// Uses morsel-parallelism-inspired tiered execution:
    /// 1. If already compiled → execute via bytecode VM (fast path)
    /// 2. Otherwise → tree-walker, then profile and maybe promote
    fn call_macro_def(
        &mut self,
        def: &super::registry::MacroDefinition,
        args: Vec<MacroValue>,
        location: SourceLocation,
    ) -> Result<MacroValue, MacroError> {
        // Fast path: already promoted to bytecode → execute via VM
        if let Some(vm) = &mut self.vm {
            if let Some(chunk) = self.registry.get_compiled(&def.qualified_name) {
                match vm.execute(chunk, args.clone()) {
                    Ok(result) => {
                        // Collect trace output from VM
                        self.trace_output.append(&mut vm.trace_output);
                        return Ok(result);
                    }
                    Err(_) => {
                        // Bytecode execution failed — fall through to tree-walker
                    }
                }
            }
        }

        // Slow path: tree-walker
        let func = MacroFunction {
            name: def.name.clone(),
            params: def.params.clone(),
            body: def.body.clone(),
            captures: std::collections::BTreeMap::new(),
        };
        // Track the class that contains this macro so bare-identifier calls
        // in the body (e.g. `parseValue(...)` inside `tink.Json.parse`)
        // can resolve to sibling static helpers via the class registry.
        let class_name = def
            .qualified_name
            .rsplit_once('.')
            .map(|(cls, _)| cls.to_string());
        if let Some(cls) = &class_name {
            self.macro_class_stack.push(cls.clone());
        }
        let result = self.call_function(&func, args, location);
        if class_name.is_some() {
            self.macro_class_stack.pop();
        }
        let result = result?;

        // Profile: check if this macro should be promoted to bytecode.
        // Morsel scheduling: when a macro crosses the threshold, batch-compile
        // the macro + all its class dependencies (the "morsel").
        if let Some(scheduler) = &mut self.scheduler {
            let count = scheduler
                .call_counts
                .entry(def.qualified_name.clone())
                .or_insert(0);
            *count += 1;
            if *count == scheduler.threshold {
                // Promote this macro: compile to bytecode
                if let Ok(chunk) =
                    BytecodeCompiler::compile(&def.qualified_name, &def.params, &def.body)
                {
                    self.registry
                        .insert_compiled(def.qualified_name.clone(), Arc::new(chunk));
                }

                // Compile class morsel: batch-compile all class dependencies
                // on first promotion (deferred from interpreter construction).
                if !scheduler.classes_compiled {
                    if let Some(cr) = &self.class_registry {
                        self.registry.compile_classes(cr);
                        // Transfer compiled classes to the VM
                        let class_chunks = self
                            .registry
                            .get_compiled_classes()
                            .iter()
                            .map(|(k, v)| {
                                (
                                    k.clone(),
                                    super::bytecode::CompiledClassInfo {
                                        constructor: v.constructor.clone(),
                                        instance_methods: v.instance_methods.clone(),
                                        static_methods: v.static_methods.clone(),
                                        instance_vars: v.instance_vars.clone(),
                                    },
                                )
                            })
                            .collect();
                        if let Some(vm) = &mut self.vm {
                            vm.set_class_chunks(class_chunks);
                        }
                    }
                    scheduler.classes_compiled = true;
                }
            }
        }

        Ok(result)
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

    /// Resolve a class name through imports.
    /// Returns the fully qualified name if the bare name was imported,
    /// otherwise returns the original name.
    ///
    /// Cross-file macro invocation note: the `import_map` is always the
    /// CALLER file's imports, not the macro-defining file's. A macro body
    /// in `tink/Json.hx` that references `Context` will fail to resolve
    /// via `import_map` when called from `Main.hx` (which doesn't
    /// `import haxe.macro.Context`). Fall back to the ClassRegistry's
    /// short-name index so registered classes resolve by their bare name
    /// regardless of the call site's imports.
    fn resolve_class_name(&self, class_name: &str) -> String {
        if let Some(qualified) = self.import_map.get(class_name) {
            return qualified.clone();
        }
        if let Some(cr) = &self.class_registry {
            if let Some(info) = cr.find_class(class_name) {
                return info.qualified_name.clone();
            }
        }
        class_name.to_string()
    }

    /// Try to dispatch a static class method call (e.g., Context.parse, Std.string)
    fn try_static_call(
        &mut self,
        class_name: &str,
        method: &str,
        args: &[MacroValue],
        location: SourceLocation,
    ) -> Result<Option<MacroValue>, MacroError> {
        // Resolve bare class names through imports
        let resolved = self.resolve_class_name(class_name);
        match resolved.as_str() {
            "haxe.macro.Context" => {
                // Use the stored macro_context if available (set by @:build
                // pipeline so Context.getBuildFields() returns class fields).
                // Fall back to a fresh empty context for expression macros.
                let result = if let Some(ref mut ctx) = self.macro_context {
                    ctx.dispatch(method, args, location)?
                } else {
                    let mut ctx = super::context_api::MacroContext::new();
                    ctx.dispatch(method, args, location)?
                };
                Ok(Some(result))
            }
            "Std" => match method {
                "string" => {
                    let val = args.first().unwrap_or(&MacroValue::Null);
                    Ok(Some(MacroValue::String(Arc::from(
                        val.to_display_string().as_str(),
                    ))))
                }
                "int" | "parseInt" => {
                    let val = args.first().unwrap_or(&MacroValue::Null);
                    match val {
                        MacroValue::String(s) => {
                            let n = s.parse::<i64>().unwrap_or(0);
                            Ok(Some(MacroValue::Int(n)))
                        }
                        MacroValue::Int(n) => Ok(Some(MacroValue::Int(*n))),
                        MacroValue::Float(f) => Ok(Some(MacroValue::Int(*f as i64))),
                        _ => Ok(Some(MacroValue::Int(0))),
                    }
                }
                "parseFloat" => {
                    let val = args.first().unwrap_or(&MacroValue::Null);
                    match val {
                        MacroValue::String(s) => {
                            let f = s.parse::<f64>().unwrap_or(0.0);
                            Ok(Some(MacroValue::Float(f)))
                        }
                        MacroValue::Float(f) => Ok(Some(MacroValue::Float(*f))),
                        MacroValue::Int(n) => Ok(Some(MacroValue::Float(*n as f64))),
                        _ => Ok(Some(MacroValue::Float(0.0))),
                    }
                }
                _ => Ok(None),
            },
            "Math" => match method {
                "abs" => {
                    let val = args.first().unwrap_or(&MacroValue::Null);
                    match val {
                        MacroValue::Int(n) => Ok(Some(MacroValue::Int(n.abs()))),
                        MacroValue::Float(f) => Ok(Some(MacroValue::Float(f.abs()))),
                        _ => Ok(None),
                    }
                }
                "floor" => {
                    let val = args.first().unwrap_or(&MacroValue::Null);
                    match val {
                        MacroValue::Float(f) => Ok(Some(MacroValue::Int(f.floor() as i64))),
                        MacroValue::Int(n) => Ok(Some(MacroValue::Int(*n))),
                        _ => Ok(None),
                    }
                }
                "ceil" => {
                    let val = args.first().unwrap_or(&MacroValue::Null);
                    match val {
                        MacroValue::Float(f) => Ok(Some(MacroValue::Int(f.ceil() as i64))),
                        MacroValue::Int(n) => Ok(Some(MacroValue::Int(*n))),
                        _ => Ok(None),
                    }
                }
                "max" => {
                    let a = args.first().and_then(|v| v.as_float()).unwrap_or(0.0);
                    let b = args.get(1).and_then(|v| v.as_float()).unwrap_or(0.0);
                    Ok(Some(MacroValue::Float(a.max(b))))
                }
                "min" => {
                    let a = args.first().and_then(|v| v.as_float()).unwrap_or(0.0);
                    let b = args.get(1).and_then(|v| v.as_float()).unwrap_or(0.0);
                    Ok(Some(MacroValue::Float(a.min(b))))
                }
                _ => Ok(None),
            },
            "sys.io.File" => match method {
                "getContent" => {
                    let path = args.first().map(|v| value_to_string(v)).unwrap_or_default();
                    match std::fs::read_to_string(&path) {
                        Ok(content) => Ok(Some(MacroValue::String(Arc::from(content.as_str())))),
                        Err(e) => Err(MacroError::RuntimeError {
                            message: format!("File.getContent('{}') failed: {}", path, e),
                            location,
                        }),
                    }
                }
                "saveContent" => {
                    let path = args.first().map(|v| value_to_string(v)).unwrap_or_default();
                    let content = args.get(1).map(|v| value_to_string(v)).unwrap_or_default();
                    std::fs::write(&path, &content).map_err(|e| MacroError::RuntimeError {
                        message: format!("File.saveContent('{}') failed: {}", path, e),
                        location,
                    })?;
                    Ok(Some(MacroValue::Null))
                }
                _ => Ok(None),
            },
            "Sys" | "sys.Sys" => match method {
                "getCwd" => {
                    let cwd = std::env::current_dir()
                        .map(|p| p.to_string_lossy().to_string())
                        .unwrap_or_default();
                    Ok(Some(MacroValue::String(Arc::from(cwd.as_str()))))
                }
                "getEnv" => {
                    let key = args.first().map(|v| value_to_string(v)).unwrap_or_default();
                    let val = std::env::var(&key).unwrap_or_default();
                    Ok(Some(MacroValue::String(Arc::from(val.as_str()))))
                }
                "systemName" => {
                    let name = if cfg!(target_os = "macos") {
                        "Mac"
                    } else if cfg!(target_os = "linux") {
                        "Linux"
                    } else if cfg!(target_os = "windows") {
                        "Windows"
                    } else {
                        "Unknown"
                    };
                    Ok(Some(MacroValue::String(Arc::from(name))))
                }
                _ => Ok(None),
            },
            "StringTools" | "haxe.StringTools" => match method {
                "trim" => {
                    let s = args.first().and_then(|v| v.as_string()).unwrap_or("");
                    Ok(Some(MacroValue::String(Arc::from(s.trim()))))
                }
                "replace" => {
                    let s = args
                        .first()
                        .and_then(|v| v.as_string())
                        .unwrap_or("")
                        .to_string();
                    let from = args.get(1).and_then(|v| v.as_string()).unwrap_or("");
                    let to = args.get(2).and_then(|v| v.as_string()).unwrap_or("");
                    Ok(Some(MacroValue::String(Arc::from(
                        s.replace(from, to).as_str(),
                    ))))
                }
                _ => Ok(None),
            },
            "haxe.macro.ComplexTypeTools" | "ComplexTypeTools" => match method {
                // `ComplexTypeTools.toString(c)` — convert a ComplexType
                // back to its source-level Haxe representation. The full
                // Haxe stdlib delegates to `Printer.printComplexType`,
                // which we don't run; instead we extract the leaf type
                // name from the parser's `Type::Path { path: TypePath {
                // name, ... }, ... }` debug format that the macro
                // pipeline uses for the kind object's `type` field. This
                // handles the common cases (`Int`, `String`, `Bool`,
                // user classes) used by `@:build` introspection helpers
                // like tink.Json's `describe()`.
                "toString" => {
                    let raw = args.first().and_then(|v| v.as_string()).unwrap_or("");
                    let s = if let Some(idx) = raw.find("name: \"") {
                        let rest = &raw[idx + 7..];
                        rest.find('"').map(|end| &rest[..end]).unwrap_or("Dynamic")
                    } else if raw.is_empty() {
                        "Dynamic"
                    } else {
                        // Couldn't find the leaf — surface the raw form
                        // so the user can spot the inability to print.
                        raw
                    };
                    Ok(Some(MacroValue::String(Arc::from(s))))
                }
                _ => Ok(None),
            },
            _ => {
                // Fallback: check ClassRegistry for user/stdlib class
                let resolved_owned = resolved.clone();
                let maybe_method = self
                    .class_registry
                    .as_ref()
                    .and_then(|cr| cr.find_static_method(&resolved_owned, method))
                    .map(|m| (m.body.clone(), m.params.clone()));
                if let Some((body, params)) = maybe_method {
                    // Depth tracking — prevents runaway recursion from
                    // crashing the compiler when a macro body has a bug
                    // (stack overflow aborts the process; we'd rather
                    // surface a diagnostic and move on).
                    self.call_depth += 1;
                    if self.call_depth > self.max_call_depth {
                        self.call_depth -= 1;
                        return Err(MacroError::RecursionLimitExceeded {
                            macro_name: format!("{}.{}", resolved_owned, method),
                            depth: self.call_depth,
                            max_depth: self.max_call_depth,
                            location,
                        });
                    }
                    self.env.push_scope();
                    for (i, param) in params.iter().enumerate() {
                        let val = args.get(i).cloned().unwrap_or(MacroValue::Null);
                        self.env.define(&param.name, val);
                    }
                    let result = self.eval_expr(&body);
                    self.env.pop_scope();
                    self.call_depth -= 1;
                    return match result {
                        Ok(val) => Ok(Some(val)),
                        Err(MacroError::Return { value: Some(v) }) => Ok(Some(*v)),
                        Err(MacroError::Return { value: None }) => Ok(Some(MacroValue::Null)),
                        Err(e) => Err(e),
                    };
                }
                Ok(None)
            }
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
        // Unwrap Expr-wrapped values so method calls work on concrete types
        // (e.g., s.charAt(0) where s is MacroValue::Expr(String("hello")))
        let unwrapped = if matches!(base, MacroValue::Expr(_)) {
            let u = ast_bridge::unwrap_expr_value(base);
            if matches!(u, MacroValue::Expr(_)) {
                None
            } else {
                Some(u)
            }
        } else {
            None
        };
        let base = unwrapped.as_ref().unwrap_or(base);

        // Check for static method calls (e.g., Std.string(), Math.abs())
        if let MacroValue::String(ref s) = base {
            // String methods would be handled here in a full implementation
            return self.string_method(s, method, args, location);
        }

        match base {
            MacroValue::Array(arr) => self.array_method(arr.as_ref(), method, args, location),
            MacroValue::Object(obj) => {
                // Check if the field is a function
                if let Some(MacroValue::Function(func)) = obj.get(method) {
                    self.call_function(func.as_ref(), args, location)
                } else {
                    self.object_method(obj.as_ref(), method, args, location)
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
                Ok(MacroValue::Array(Arc::new(new_arr)))
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
                        new_arr.extend(other.iter().cloned());
                    }
                }
                Ok(MacroValue::Array(Arc::new(new_arr)))
            }
            "join" => {
                let sep = args.first().and_then(|a| a.as_string()).unwrap_or(",");
                let parts: Vec<String> = arr.iter().map(|v| v.to_display_string()).collect();
                Ok(MacroValue::String(Arc::from(parts.join(sep).as_str())))
            }
            "map" => {
                if let Some(MacroValue::Function(func)) = args.first() {
                    let mut result = Vec::with_capacity(arr.len());
                    for item in arr {
                        let mapped =
                            self.call_function(func.as_ref(), vec![item.clone()], location)?;
                        result.push(mapped);
                    }
                    Ok(MacroValue::Array(Arc::new(result)))
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
                        let keep =
                            self.call_function(func.as_ref(), vec![item.clone()], location)?;
                        if keep.is_truthy() {
                            result.push(item.clone());
                        }
                    }
                    Ok(MacroValue::Array(Arc::new(result)))
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
                Ok(MacroValue::String(Arc::from(
                    s.chars()
                        .nth(idx)
                        .map(|c| c.to_string())
                        .unwrap_or_default()
                        .as_str(),
                )))
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
                    .map(|part| MacroValue::String(Arc::from(part)))
                    .collect();
                Ok(MacroValue::Array(Arc::new(parts)))
            }
            "substring" => {
                // Haxe `String.substring(startIndex, endIndex)` — chars from
                // startIndex up to (but not including) endIndex.
                let start = args.first().and_then(|a| a.as_int()).unwrap_or(0).max(0) as usize;
                let end = args
                    .get(1)
                    .and_then(|a| a.as_int())
                    .map(|e| e.max(0) as usize)
                    .unwrap_or(s.len());
                let take = end.saturating_sub(start);
                let result: String = s.chars().skip(start).take(take).collect();
                Ok(MacroValue::String(Arc::from(result.as_str())))
            }
            "substr" => {
                // Haxe `String.substr(pos, ?len)` — `len` characters starting
                // at `pos`. Critically *not* the same as `substring(start, end)`:
                // the second argument is a LENGTH, not an end index. Treating
                // them identically caused tink.Json's `s.substr(pos, 4) ==
                // "true"` literal-detection to read only 3 characters and
                // miss every keyword.
                let pos = args.first().and_then(|a| a.as_int()).unwrap_or(0);
                let s_len = s.chars().count() as i64;
                // Negative `pos` counts from the end (Haxe spec).
                let start = if pos < 0 {
                    (s_len + pos).max(0) as usize
                } else {
                    (pos as usize).min(s_len as usize)
                };
                let len = match args.get(1).and_then(|a| a.as_int()) {
                    Some(n) if n < 0 => {
                        // Negative length: stop at `len` from the end.
                        let absn = (-n) as i64;
                        (s_len - start as i64 - absn).max(0) as usize
                    }
                    Some(n) => n as usize,
                    None => s_len as usize - start,
                };
                let result: String = s.chars().skip(start).take(len).collect();
                Ok(MacroValue::String(Arc::from(result.as_str())))
            }
            "toUpperCase" => Ok(MacroValue::String(Arc::from(s.to_uppercase().as_str()))),
            "toLowerCase" => Ok(MacroValue::String(Arc::from(s.to_lowercase().as_str()))),
            "toString" => Ok(MacroValue::String(Arc::from(s))),
            _ => Err(MacroError::UnsupportedOperation {
                operation: format!("String.{}()", method),
                location,
            }),
        }
    }

    /// Built-in object methods
    fn object_method(
        &mut self,
        obj: &std::collections::BTreeMap<String, MacroValue>,
        method: &str,
        _args: Vec<MacroValue>,
        location: SourceLocation,
    ) -> Result<MacroValue, MacroError> {
        // `.get()` on plain objects acts as a Ref/Null<T> dereference — it
        // returns the object itself. This lets Haxe macro idioms like
        // `Context.getLocalClass().get().name` work when the underlying
        // class is modeled as an Object.
        if method == "get" && _args.is_empty() && !obj.contains_key("__type__") {
            return Ok(MacroValue::Object(Arc::new(obj.clone())));
        }

        // Check __type__ for type-specific method dispatch
        let type_name = obj
            .get("__type__")
            .and_then(|v| v.as_string().map(|s| s.to_string()));

        match type_name.as_deref() {
            Some("sys.io.ProcessOutput" | "sys.io.ProcessStdout") => match method {
                "readAll" | "readLine" => {
                    // Return the stored data as a string (Bytes → String via toString)
                    Ok(obj
                        .get("__data__")
                        .cloned()
                        .unwrap_or(MacroValue::String(Arc::from(""))))
                }
                "toString" => Ok(obj
                    .get("__data__")
                    .cloned()
                    .unwrap_or(MacroValue::String(Arc::from("")))),
                _ => Err(MacroError::UnsupportedOperation {
                    operation: format!("ProcessOutput.{}()", method),
                    location,
                }),
            },
            Some("sys.io.Process") => match method {
                "close" | "kill" => Ok(MacroValue::Null),
                "exitCode" => Ok(obj
                    .get("__exitCode__")
                    .cloned()
                    .unwrap_or(MacroValue::Int(0))),
                _ => Err(MacroError::UnsupportedOperation {
                    operation: format!("Process.{}()", method),
                    location,
                }),
            },
            _ => {
                // Fallback: check ClassRegistry for instance methods
                if let Some(ref tn) = type_name {
                    if let Some(ref cr) = self.class_registry {
                        if let Some(method_info) = cr.find_instance_method(tn, method) {
                            let body = method_info.body.clone();
                            let params = method_info.params.clone();
                            self.env.push_scope();
                            self.env
                                .define("this", MacroValue::Object(Arc::new(obj.clone())));
                            for (i, param) in params.iter().enumerate() {
                                let val = _args.get(i).cloned().unwrap_or(MacroValue::Null);
                                self.env.define(&param.name, val);
                            }
                            let result = self.eval_expr(&body);
                            self.env.pop_scope();
                            return match result {
                                Ok(val) => Ok(val),
                                Err(MacroError::Return { value: Some(v) }) => Ok(*v),
                                Err(MacroError::Return { value: None }) => Ok(MacroValue::Null),
                                Err(e) => Err(e),
                            };
                        }
                    }
                }
                Err(MacroError::UnsupportedOperation {
                    operation: format!("Object.{}()", method),
                    location,
                })
            }
        }
    }

    /// Construct a sys.io.Process — runs a real subprocess at compile time
    fn construct_process(
        &self,
        args: Vec<MacroValue>,
        location: SourceLocation,
    ) -> Result<MacroValue, MacroError> {
        let cmd =
            args.first()
                .map(|v| value_to_string(v))
                .ok_or_else(|| MacroError::TypeError {
                    message: "sys.io.Process requires a command string".to_string(),
                    location,
                })?;

        let proc_args: Vec<String> = if let Some(arr_val) = args.get(1) {
            // Unwrap Expr if needed
            let unwrapped = ast_bridge::unwrap_expr_value(arr_val);
            match unwrapped {
                MacroValue::Array(arr) => arr.iter().map(|v| value_to_string(v)).collect(),
                _ => Vec::new(),
            }
        } else {
            Vec::new()
        };

        // Execute the process
        let output = std::process::Command::new(&cmd)
            .args(&proc_args)
            .output()
            .map_err(|e| MacroError::RuntimeError {
                message: format!("failed to execute '{}': {}", cmd, e),
                location,
            })?;

        let stdout_str = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr_str = String::from_utf8_lossy(&output.stderr).to_string();
        let exit_code = output.status.code().unwrap_or(-1) as i64;

        // Build stdout pseudo-object (supports .readAll() / .toString())
        let mut stdout_obj = std::collections::BTreeMap::new();
        stdout_obj.insert(
            "__type__".to_string(),
            MacroValue::String(Arc::from("sys.io.ProcessStdout")),
        );
        stdout_obj.insert(
            "__data__".to_string(),
            MacroValue::String(Arc::from(stdout_str.as_str())),
        );

        // Build stderr pseudo-object
        let mut stderr_obj = std::collections::BTreeMap::new();
        stderr_obj.insert(
            "__type__".to_string(),
            MacroValue::String(Arc::from("sys.io.ProcessOutput")),
        );
        stderr_obj.insert(
            "__data__".to_string(),
            MacroValue::String(Arc::from(stderr_str.as_str())),
        );

        // Build Process object
        let mut proc_obj = std::collections::BTreeMap::new();
        proc_obj.insert(
            "__type__".to_string(),
            MacroValue::String(Arc::from("sys.io.Process")),
        );
        proc_obj.insert(
            "stdout".to_string(),
            MacroValue::Object(Arc::new(stdout_obj)),
        );
        proc_obj.insert(
            "stderr".to_string(),
            MacroValue::Object(Arc::new(stderr_obj)),
        );
        proc_obj.insert("__exitCode__".to_string(), MacroValue::Int(exit_code));

        Ok(MacroValue::Object(Arc::new(proc_obj)))
    }

    /// Construct a sys.io.File — compile-time file I/O
    fn construct_file(
        &self,
        _args: Vec<MacroValue>,
        _location: SourceLocation,
    ) -> Result<MacroValue, MacroError> {
        // File construction is typically done via static methods (File.getContent)
        let mut obj = std::collections::BTreeMap::new();
        obj.insert(
            "__type__".to_string(),
            MacroValue::String(Arc::from("sys.io.File")),
        );
        Ok(MacroValue::Object(Arc::new(obj)))
    }

    /// Construct an object from pre-extracted ClassRegistry data.
    /// Takes cloned data to avoid borrow conflicts with &mut self.
    fn construct_from_registry_data(
        &mut self,
        qualified_name: &str,
        instance_vars: &[(String, Option<Arc<Expr>>)],
        constructor: Option<&(Arc<Expr>, Vec<parser::FunctionParam>)>,
        args: Vec<MacroValue>,
        _location: SourceLocation,
    ) -> Result<MacroValue, MacroError> {
        // Create the object with __type__ and instance var defaults
        let mut obj = std::collections::BTreeMap::new();
        obj.insert(
            "__type__".to_string(),
            MacroValue::String(Arc::from(qualified_name)),
        );

        // Initialize instance vars with their default init expressions
        for (name, init_expr) in instance_vars {
            let val = if let Some(ref init) = init_expr {
                self.eval_expr(init).unwrap_or(MacroValue::Null)
            } else {
                MacroValue::Null
            };
            obj.insert(name.clone(), val);
        }

        // If there's a constructor, run it with `this` bound to the object
        if let Some((ctor_body, ctor_params)) = constructor {
            let ctor_body = ctor_body.clone();
            let ctor_params = ctor_params.clone();

            self.env.push_scope();
            self.env
                .define("this", MacroValue::Object(Arc::new(obj.clone())));

            // Bind constructor params
            for (i, param) in ctor_params.iter().enumerate() {
                let val = args.get(i).cloned().unwrap_or(MacroValue::Null);
                self.env.define(&param.name, val);
            }

            // Execute constructor body
            let result = self.eval_expr(&ctor_body);
            match result {
                Ok(_) | Err(MacroError::Return { .. }) => {}
                Err(e) => {
                    self.env.pop_scope();
                    return Err(e);
                }
            }

            // Extract the (potentially mutated) `this` from the environment
            let final_obj = self
                .env
                .get("this")
                .cloned()
                .unwrap_or(MacroValue::Object(Arc::new(obj)));
            self.env.pop_scope();
            Ok(final_obj)
        } else {
            Ok(MacroValue::Object(Arc::new(obj)))
        }
    }

    /// Access a field on a value
    fn field_access(
        &self,
        base: &MacroValue,
        field: &str,
        location: SourceLocation,
    ) -> Result<MacroValue, MacroError> {
        // AST-introspection fields on `Expr` values (`.expr`, `.pos`) must
        // be routed to the `MacroValue::Expr` arm below — they are how
        // build-macro / parse-macro code reads the variant tag and source
        // position of a reified `Expr`. If we unwrap first, an
        // `Expr(ExprKind::String("x"))` becomes a `MacroValue::String` and
        // `.expr` then errors with "String has no field 'expr'", which is
        // exactly the failure that broke `tink.Json.parse('{"x":1}')`'s
        // key extraction.
        //
        // For all other fields (e.g., `.length`), the unwrap is the right
        // call: it lets `s.length` work when `s` is an `Expr` wrapping a
        // string literal.
        let preserve_expr = matches!(field, "expr" | "pos");
        let unwrapped = if !preserve_expr && matches!(base, MacroValue::Expr(_)) {
            let u = ast_bridge::unwrap_expr_value(base);
            if matches!(u, MacroValue::Expr(_)) {
                None
            } else {
                Some(u)
            }
        } else {
            None
        };
        let base = unwrapped.as_ref().unwrap_or(base);
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
                    "expr" => {
                        // Return the ExprDef as an enum-like value
                        Ok(ast_bridge::expr_kind_to_value(&expr.kind, expr.span))
                    }
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
                Ok(map.get(&**key).cloned().unwrap_or(MacroValue::Null))
            }
            (MacroValue::Object(map), other) => {
                let key = other.to_display_string();
                Ok(map.get(&key).cloned().unwrap_or(MacroValue::Null))
            }
            (MacroValue::String(s), MacroValue::Int(i)) => {
                let idx = *i as usize;
                Ok(MacroValue::String(Arc::from(
                    s.chars()
                        .nth(idx)
                        .map(|c| c.to_string())
                        .unwrap_or_default()
                        .as_str(),
                )))
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
            UnaryOp::PreIncr
            | UnaryOp::PostIncr
            | UnaryOp::PreDecr
            | UnaryOp::PostDecr => {
                // Inc/dec are handled by `eval_inc_dec` in the caller because
                // they need lvalue write-back; they should never flow through
                // this helper. If we see one here it indicates a routing bug.
                Err(MacroError::TypeError {
                    message: format!(
                        "internal: apply_unary_op called with inc/dec {:?}; \
                         should be routed through eval_inc_dec",
                        op
                    ),
                    location,
                })
            }
        }
    }

    /// Evaluate a pre/post inc/dec expression with lvalue write-back.
    ///
    /// Haxe semantics: PostIncr returns the original value, PreIncr returns
    /// the new value (same for Decr). `Int` stays `Int`, `Float` stays `Float`
    /// — no promotion.
    ///
    /// The three target shapes mirror `eval_assignment`:
    /// - `Ident` — read, update, `env.set`.
    /// - `Field` — fast path via `mutate_object_field`, fallback via
    ///   `assign_base` on the evaluated base `MacroValue::Object`.
    /// - `Index` — COW through `assign_base`, mirroring the `Assign` path.
    fn eval_inc_dec(
        &mut self,
        op: &UnaryOp,
        target: &Expr,
        location: SourceLocation,
    ) -> Result<MacroValue, MacroError> {
        let is_incr = matches!(op, UnaryOp::PreIncr | UnaryOp::PostIncr);
        let is_pre = matches!(op, UnaryOp::PreIncr | UnaryOp::PreDecr);

        let add_one = |val: &MacroValue| -> Result<MacroValue, MacroError> {
            match val {
                MacroValue::Int(i) => Ok(MacroValue::Int(if is_incr { i + 1 } else { i - 1 })),
                MacroValue::Float(f) => Ok(MacroValue::Float(if is_incr {
                    f + 1.0
                } else {
                    f - 1.0
                })),
                _ => Err(MacroError::TypeError {
                    message: format!(
                        "cannot {} {}",
                        if is_incr { "increment" } else { "decrement" },
                        val.type_name()
                    ),
                    location,
                }),
            }
        };

        match &target.kind {
            ExprKind::Ident(name) => {
                let old = self
                    .env
                    .get(name)
                    .cloned()
                    .ok_or_else(|| MacroError::UndefinedVariable {
                        name: name.clone(),
                        location,
                    })?;
                let new = add_one(&old)?;
                if !self.env.set(name, new.clone()) {
                    self.env.define(name, new.clone());
                }
                Ok(if is_pre { new } else { old })
            }
            ExprKind::Field {
                expr: base, field, ..
            } => {
                // Read the current field value through a normal field access.
                let base_val = self.eval_expr(base)?;
                let old = self.field_access(&base_val, field, location)?;
                let new = add_one(&old)?;
                // Write back — reuse eval_assignment's field path by going
                // through the fast/fallback machinery.
                let base_var_name = match &base.kind {
                    ExprKind::This => Some("this"),
                    ExprKind::Ident(n) => Some(n.as_str()),
                    _ => None,
                };
                if let Some(var_name) = base_var_name {
                    if self.env.mutate_object_field(var_name, field, new.clone()) {
                        return Ok(if is_pre { new } else { old });
                    }
                }
                // Fallback: COW the base object and reassign.
                let mut base_val = self.eval_expr(base)?;
                if let MacroValue::Object(ref mut arc_map) = base_val {
                    Arc::make_mut(arc_map).insert(field.clone(), new.clone());
                    self.assign_base(base, base_val)?;
                    Ok(if is_pre { new } else { old })
                } else {
                    Err(MacroError::TypeError {
                        message: format!(
                            "cannot {} field '{}' on {}",
                            if is_incr { "increment" } else { "decrement" },
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
                let old = self.index_access(&base_val, &idx, location)?;
                let new = add_one(&old)?;
                match (&mut base_val, &idx) {
                    (MacroValue::Array(arc_arr), MacroValue::Int(i)) => {
                        let i = *i as usize;
                        let arr = Arc::make_mut(arc_arr);
                        if i < arr.len() {
                            arr[i] = new.clone();
                        }
                        self.assign_base(base, base_val)?;
                        Ok(if is_pre { new } else { old })
                    }
                    (MacroValue::Object(arc_map), _) => {
                        Arc::make_mut(arc_map)
                            .insert(idx.to_display_string(), new.clone());
                        self.assign_base(base, base_val)?;
                        Ok(if is_pre { new } else { old })
                    }
                    _ => Err(MacroError::TypeError {
                        message: format!(
                            "cannot {} indexed element on {}",
                            if is_incr { "increment" } else { "decrement" },
                            base_val.type_name()
                        ),
                        location,
                    }),
                }
            }
            _ => Err(MacroError::TypeError {
                message: "inc/dec target must be a variable, field access, or indexed element"
                    .to_string(),
                location,
            }),
        }
    }

    /// Get an iterable sequence from a value
    fn get_iterable(
        &self,
        value: &MacroValue,
        location: SourceLocation,
    ) -> Result<Vec<MacroValue>, MacroError> {
        match value {
            MacroValue::Array(arr) => Ok(arr.as_ref().clone()),
            MacroValue::Object(map) => {
                // Iterate over keys
                Ok(map
                    .keys()
                    .map(|k| MacroValue::String(Arc::from(k.as_str())))
                    .collect())
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
            parser::Pattern::Constructor { path, params } => {
                // `case FVar(t, _):` over a value like
                //   { kind: "FVar", type: <T>, expr: <E>, __args__: [<T>, <E>] }
                // We match the path's leaf name against the object's `kind`
                // tag and bind each pattern param against the same-index
                // entry of `__args__`.
                let ctor_name = &path.name;
                let obj = match value {
                    MacroValue::Object(map) => map,
                    MacroValue::Enum(_, variant, payload) => {
                        // Enum-shaped values (e.g. `EConst(CString(s))`
                        // produced by `expr_kind_to_value`) match by
                        // variant name + positional payload.
                        if &**variant != ctor_name.as_str() {
                            return Ok(false);
                        }
                        if params.len() > payload.len() {
                            return Ok(false);
                        }
                        for (sub_pat, sub_val) in params.iter().zip(payload.iter()) {
                            if !self.match_pattern(sub_val, sub_pat)? {
                                return Ok(false);
                            }
                        }
                        return Ok(true);
                    }
                    _ => return Ok(false),
                };
                // Verify the variant tag.
                match obj.get("kind") {
                    Some(MacroValue::String(s)) if &**s == ctor_name.as_str() => {}
                    _ => return Ok(false),
                }
                // Pull positional args. If `__args__` is missing the kind
                // object was constructed without positional info — match
                // only if no params were expected.
                let args: Vec<MacroValue> = match obj.get("__args__") {
                    Some(MacroValue::Array(arr)) => arr.iter().cloned().collect(),
                    _ => Vec::new(),
                };
                if params.len() > args.len() {
                    return Ok(false);
                }
                for (sub_pat, sub_val) in params.iter().zip(args.iter()) {
                    if !self.match_pattern(sub_val, sub_pat)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            parser::Pattern::Or(alts) => {
                for alt in alts {
                    if self.match_pattern(value, alt)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            parser::Pattern::Null => Ok(matches!(value, MacroValue::Null)),
            _ => {
                // Array, Object, Type, Extractor patterns — not supported
                // yet by the macro tree-walker. Return false so the case
                // doesn't silently appear to match.
                Ok(false)
            }
        }
    }
}

/// Collect all free variable references from an expression AST.
///
/// Walks the expression tree and collects all `Ident` names that appear,
/// excluding names that are bound by local `var` declarations or function parameters.
/// This is used for selective closure capture — only variables actually referenced
/// in the closure body need to be captured from the enclosing scope.
/// Recognise enum-constructor calls from `haxe.macro.Expr.FieldType` —
/// e.g. `FFun({args, ret, expr})` — and report the variant tag we want
/// to surface as the `kind` field of the resulting object. None means
/// the identifier isn't an enum constructor and should fall through to
/// the normal undefined-variable error.
fn enum_ctor_tag(name: &str) -> Option<&'static str> {
    match name {
        // haxe.macro.Expr.FieldType
        "FVar" => Some("FVar"),
        "FFun" => Some("FFun"),
        "FProp" => Some("FProp"),
        // haxe.macro.Expr.ComplexType
        "TPath" => Some("TPath"),
        "TFunction" => Some("TFunction"),
        "TAnonymous" => Some("TAnonymous"),
        "TParent" => Some("TParent"),
        "TExtend" => Some("TExtend"),
        "TOptional" => Some("TOptional"),
        _ => None,
    }
}

/// Construct the object form of an enum-constructor call.
///
/// For tagged variants like `FFun(payload)` where the single argument is
/// already an Object, we merge the payload's fields into the result and
/// store the variant tag under `kind` — that's the shape
/// `value_to_class_field` reads.
///
/// For other shapes (multiple args, non-object args), we fall back to
/// `{kind: tag, args: [...]}`.
fn build_enum_ctor_value(tag: &'static str, args: &[MacroValue]) -> MacroValue {
    let mut map = std::collections::BTreeMap::new();
    map.insert(
        "kind".to_string(),
        MacroValue::String(Arc::from(tag)),
    );
    if args.len() == 1 {
        if let MacroValue::Object(payload) = &args[0] {
            for (k, v) in payload.iter() {
                if k != "kind" {
                    map.insert(k.clone(), v.clone());
                }
            }
            return MacroValue::Object(Arc::new(map));
        }
    }
    map.insert(
        "args".to_string(),
        MacroValue::Array(Arc::new(args.to_vec())),
    );
    MacroValue::Object(Arc::new(map))
}

/// Recognise the bare-identifier form of `haxe.macro.Expr.Access` enum
/// constructors so build-macro bodies that use `APublic`, `AInline`, etc.
/// can be interpreted without an actual enum table.
///
/// Returns the matching string token (`"Public"`, `"Inline"`, …) — the
/// representation `value_to_class_field` already understands.
fn enum_ident_as_string(name: &str) -> Option<&'static str> {
    match name {
        // haxe.macro.Expr.Access — used in @:build-generated `Field.access`.
        "APublic" => Some("Public"),
        "APrivate" => Some("Private"),
        "AStatic" => Some("Static"),
        "AOverride" => Some("Override"),
        "ADynamic" => Some("Dynamic"),
        "AInline" => Some("Inline"),
        "AMacro" => Some("Macro"),
        "AFinal" => Some("Final"),
        "AExtern" => Some("Extern"),
        _ => None,
    }
}

fn collect_free_vars(
    expr: &Expr,
    bound: &mut std::collections::BTreeSet<String>,
    free: &mut std::collections::BTreeSet<String>,
) {
    match &expr.kind {
        ExprKind::Ident(name) => {
            if !bound.contains(name) {
                free.insert(name.clone());
            }
        }
        ExprKind::Var {
            name, expr: init, ..
        }
        | ExprKind::Final {
            name, expr: init, ..
        } => {
            if let Some(init) = init {
                collect_free_vars(init, bound, free);
            }
            bound.insert(name.clone());
        }
        ExprKind::Function(func) => {
            let mut inner_bound = bound.clone();
            for p in &func.params {
                inner_bound.insert(p.name.clone());
            }
            if let Some(body) = &func.body {
                collect_free_vars(body, &mut inner_bound, free);
            }
        }
        ExprKind::Arrow { params, expr: body } => {
            let mut inner_bound = bound.clone();
            for p in params {
                inner_bound.insert(p.name.clone());
            }
            collect_free_vars(body, &mut inner_bound, free);
        }
        ExprKind::Block(elements) => {
            let mut block_bound = bound.clone();
            for elem in elements {
                if let parser::BlockElement::Expr(e) = elem {
                    collect_free_vars(e, &mut block_bound, free);
                }
            }
        }
        ExprKind::For {
            var, iter, body, ..
        } => {
            collect_free_vars(iter, bound, free);
            let mut for_bound = bound.clone();
            for_bound.insert(var.clone());
            collect_free_vars(body, &mut for_bound, free);
        }
        ExprKind::While { cond, body } | ExprKind::DoWhile { body, cond } => {
            collect_free_vars(cond, bound, free);
            collect_free_vars(body, bound, free);
        }
        ExprKind::If {
            cond,
            then_branch,
            else_branch,
        } => {
            collect_free_vars(cond, bound, free);
            collect_free_vars(then_branch, bound, free);
            if let Some(e) = else_branch {
                collect_free_vars(e, bound, free);
            }
        }
        ExprKind::Binary { left, right, .. } => {
            collect_free_vars(left, bound, free);
            collect_free_vars(right, bound, free);
        }
        ExprKind::Unary { expr: inner, .. } => {
            collect_free_vars(inner, bound, free);
        }
        ExprKind::Call { expr: callee, args } => {
            collect_free_vars(callee, bound, free);
            for a in args {
                collect_free_vars(a, bound, free);
            }
        }
        ExprKind::Field { expr: obj, .. } => {
            collect_free_vars(obj, bound, free);
        }
        ExprKind::Index { expr: obj, index } => {
            collect_free_vars(obj, bound, free);
            collect_free_vars(index, bound, free);
        }
        ExprKind::Array(items) => {
            for item in items {
                collect_free_vars(item, bound, free);
            }
        }
        ExprKind::Object(fields) => {
            for f in fields {
                collect_free_vars(&f.expr, bound, free);
            }
        }
        ExprKind::Assign { left, right, .. } => {
            collect_free_vars(left, bound, free);
            collect_free_vars(right, bound, free);
        }
        ExprKind::Return(Some(inner)) | ExprKind::Throw(inner) | ExprKind::Paren(inner) => {
            collect_free_vars(inner, bound, free);
        }
        ExprKind::Ternary {
            cond,
            then_expr,
            else_expr,
        } => {
            collect_free_vars(cond, bound, free);
            collect_free_vars(then_expr, bound, free);
            collect_free_vars(else_expr, bound, free);
        }
        ExprKind::Switch {
            expr: scrutinee,
            cases,
            default,
        } => {
            collect_free_vars(scrutinee, bound, free);
            for case in cases {
                collect_free_vars(&case.body, bound, free);
            }
            if let Some(d) = default {
                collect_free_vars(d, bound, free);
            }
        }
        ExprKind::Try {
            expr: body,
            catches,
            ..
        } => {
            collect_free_vars(body, bound, free);
            for c in catches {
                let mut catch_bound = bound.clone();
                catch_bound.insert(c.var.clone());
                collect_free_vars(&c.body, &mut catch_bound, free);
            }
        }
        ExprKind::StringInterpolation(parts) => {
            for part in parts {
                if let parser::StringPart::Interpolation(e) = part {
                    collect_free_vars(e, bound, free);
                }
            }
        }
        ExprKind::Cast { expr: inner, .. }
        | ExprKind::Macro(inner)
        | ExprKind::TypeCheck { expr: inner, .. } => {
            collect_free_vars(inner, bound, free);
        }
        ExprKind::New { args, .. } => {
            for a in args {
                collect_free_vars(a, bound, free);
            }
        }
        _ => {}
    }
}

/// Get the set of free variables in a function body, excluding parameter names
fn get_free_vars_for_closure(
    body: &Expr,
    params: &[MacroParam],
) -> std::collections::BTreeSet<String> {
    let mut bound = std::collections::BTreeSet::new();
    for p in params {
        bound.insert(p.name.clone());
    }
    let mut free = std::collections::BTreeSet::new();
    collect_free_vars(body, &mut bound, &mut free);
    free
}

/// Extract a string from a MacroValue, unwrapping Expr wrappers.
fn value_to_string(val: &MacroValue) -> String {
    match val {
        MacroValue::String(s) => s.to_string(),
        MacroValue::Int(n) => n.to_string(),
        MacroValue::Float(f) => f.to_string(),
        MacroValue::Expr(_) => {
            let unwrapped = ast_bridge::unwrap_expr_value(val);
            if matches!(unwrapped, MacroValue::Expr(_)) {
                val.to_display_string()
            } else {
                value_to_string(&unwrapped)
            }
        }
        _ => val.to_display_string(),
    }
}

/// Extract a qualified class name and method from a nested Field chain.
///
/// For `haxe.macro.Context.parse(...)`, the callee is:
///   Field { expr: Field { expr: Field { Ident("haxe"), "macro" }, "Context" }, "parse" }
///
/// This function is called with `base = Field{..., "Context"}` and `method = "parse"`.
/// It walks the nested Field chain to build "haxe.macro.Context" and returns it
/// along with the method name.
fn extract_qualified_call<'a>(base: &'a Expr, method: &'a str) -> Option<(String, &'a str)> {
    // base should be a Field chain ending in the class name
    // Walk the chain to collect path segments
    let mut segments = Vec::new();
    let mut current = base;
    loop {
        match &current.kind {
            ExprKind::Field {
                expr: inner, field, ..
            } => {
                segments.push(field.as_str());
                current = inner;
            }
            ExprKind::Ident(name) => {
                segments.push(name.as_str());
                break;
            }
            _ => return None,
        }
    }
    // segments are in reverse order: ["Context", "macro", "haxe"]
    segments.reverse();
    let qualified = segments.join(".");
    Some((qualified, method))
}

/// Build an import map from a parsed file's import declarations.
///
/// Maps short class names to their fully qualified paths:
/// - `import haxe.macro.Context;` → "Context" → "haxe.macro.Context"
/// - `import haxe.macro.Context as Ctx;` → "Ctx" → "haxe.macro.Context"
/// - `import haxe.macro.*;` → not resolved here (wildcard)
pub fn build_import_map(imports: &[parser::Import]) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    for import in imports {
        match &import.mode {
            parser::ImportMode::Normal => {
                // import haxe.macro.Context → short name is last segment
                if let Some(last) = import.path.last() {
                    let qualified = import.path.join(".");
                    map.insert(last.clone(), qualified);
                }
            }
            parser::ImportMode::Alias(alias) => {
                // import haxe.macro.Context as Ctx → alias maps to full path
                let qualified = import.path.join(".");
                map.insert(alias.clone(), qualified);
            }
            parser::ImportMode::Field(_)
            | parser::ImportMode::Wildcard
            | parser::ImportMode::WildcardWithExclusions(_) => {
                // Wildcard and field imports not tracked for class resolution
            }
        }
    }
    map
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
            MacroValue::from_str("hello")
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
            MacroValue::from_str("hello world")
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

    // Phase 5 regression guards.
    //
    // Before Phase 5, `apply_unary_op` computed `i+1` for `PostIncr` but
    // dropped the result (never writing it back through the lvalue), so
    // `x++` was a no-op. This silently broke any macro using `while (i<n) i++`
    // — including tink.Json.parseValue, which recursed to the stack-limit
    // trying to advance past `[`.
    //
    // Bytecode VM does this correctly; tree-walker runs first until the
    // tiering threshold, so the bug only hits slow-path macros.
    #[test]
    fn test_post_increment_ident() {
        assert_eq!(
            eval("var x = 5; var old = x++; return old + x * 10;").unwrap(),
            MacroValue::Int(5 + 6 * 10)
        );
    }

    #[test]
    fn test_pre_increment_ident() {
        assert_eq!(
            eval("var x = 5; var new_ = ++x; return new_ + x * 10;").unwrap(),
            MacroValue::Int(6 + 6 * 10)
        );
    }

    #[test]
    fn test_post_decrement_ident() {
        assert_eq!(
            eval("var x = 5; var old = x--; return old + x * 10;").unwrap(),
            MacroValue::Int(5 + 4 * 10)
        );
    }

    #[test]
    fn test_pre_decrement_ident() {
        assert_eq!(
            eval("var x = 5; var new_ = --x; return new_ + x * 10;").unwrap(),
            MacroValue::Int(4 + 4 * 10)
        );
    }

    /// The canonical failing loop from tink.Json.parseValue — this would
    /// recurse infinitely before Phase 5 because `i++` was a no-op.
    #[test]
    fn test_while_loop_with_increment_terminates() {
        assert_eq!(
            eval("var i = 0; while (i < 10) i++; return i;").unwrap(),
            MacroValue::Int(10)
        );
    }

    #[test]
    fn test_increment_keeps_type() {
        // Int stays Int, Float stays Float — no promotion.
        assert!(matches!(
            eval("var x = 0; x++; return x;").unwrap(),
            MacroValue::Int(1)
        ));
        assert!(matches!(
            eval("var x = 0.5; x++; return x;").unwrap(),
            MacroValue::Float(f) if (f - 1.5).abs() < 1e-9
        ));
    }

    #[test]
    fn test_increment_object_field() {
        assert_eq!(
            eval("var obj = { count: 10 }; obj.count++; return obj.count;").unwrap(),
            MacroValue::Int(11)
        );
    }

    #[test]
    fn test_pre_increment_object_field_returns_new() {
        assert_eq!(
            eval("var obj = { count: 10 }; var n = ++obj.count; return n;").unwrap(),
            MacroValue::Int(11)
        );
    }

    #[test]
    fn test_post_increment_array_element() {
        assert_eq!(
            eval("var arr = [10, 20, 30]; arr[1]++; return arr[1];").unwrap(),
            MacroValue::Int(21)
        );
    }

    #[test]
    fn test_post_increment_array_element_returns_old() {
        assert_eq!(
            eval("var arr = [10, 20, 30]; var old = arr[1]++; return old;").unwrap(),
            MacroValue::Int(20)
        );
    }

    /// Regression guard for the JSON parsing bug fixes (substr/substring,
    /// eval_block scope leak, field-access unwrap).

    /// `String.substr(pos, len)` and `String.substring(start, end)` must
    /// be different. Treating them identically caused
    /// `s.substr(pos, 4) == "true"` to silently fail every keyword check
    /// in tink.Json — substr returned 3 chars instead of 4.
    #[test]
    fn test_string_substr_length_argument() {
        // substr(start, len) — len characters starting at start.
        assert_eq!(
            eval("return \"abcdefgh\".substr(1, 4);").unwrap(),
            MacroValue::from_str("bcde")
        );
        // substr with no length — to end of string.
        assert_eq!(
            eval("return \"abcdef\".substr(2);").unwrap(),
            MacroValue::from_str("cdef")
        );
        // Negative pos counts from end.
        assert_eq!(
            eval("return \"abcdef\".substr(-3, 2);").unwrap(),
            MacroValue::from_str("de")
        );
        // substring(start, end) — different semantics.
        assert_eq!(
            eval("return \"abcdefgh\".substring(1, 4);").unwrap(),
            MacroValue::from_str("bcd")
        );
    }

    /// Block scopes must be popped on early returns. Before the fix
    /// `eval_block` used `eval_expr(expr)?` which short-circuited past
    /// `pop_scope`, leaking one scope per `return`. Mutually recursive
    /// macro helpers grew an unbounded scope stack and identifier
    /// lookups found stale outer-scope values, causing infinite loops
    /// (tink.Json.parseValue ↔ parseArray ↔ parseNumber).
    #[test]
    fn test_block_scope_popped_on_early_return() {
        // Deeply nested early returns must not leak scopes — if they
        // did, this test would hang or produce wrong results because
        // the outer `pos` would shadow each inner `pos` in turn.
        let source = r#"
            var pos = 0;
            for (i in 0...10) {
                if (i > 5) {
                    pos = 99; // late assignment
                    if (i == 7) {
                        // early-return-style nested block
                    }
                }
            }
            return pos;
        "#;
        assert_eq!(eval(source).unwrap(), MacroValue::Int(99));
    }

    /// `expr.expr` and `expr.pos` on a `MacroValue::Expr` must return AST
    /// metadata, not unwrap the wrapped value. Build/parse macros use
    /// `key.expr.expr` (with key being a parseString-returned Object) to
    /// pattern-match on `EConst(CString(_))` — the unwrap step was making
    /// `key.expr` collapse into a bare `MacroValue::String`, so `.expr`
    /// then errored with "String has no field 'expr'".
    #[test]
    fn test_field_access_preserves_expr_for_introspection() {
        // Build an Object with an Expr field (simulating
        // `{expr: macro $v{result}, ...}` from parseString)
        let mut interp = MacroInterpreter::new(MacroRegistry::new());
        let inner = parser::Expr {
            kind: parser::ExprKind::String("hello".to_string()),
            span: parser::Span::default(),
        };
        let mut obj = std::collections::BTreeMap::new();
        obj.insert("expr".to_string(), MacroValue::Expr(Arc::new(inner)));
        obj.insert("endPos".to_string(), MacroValue::Int(7));
        let key_obj = MacroValue::Object(Arc::new(obj));

        // Access `.expr` once — gets the Expr value (not unwrapped).
        let key_expr = interp
            .field_access(&key_obj, "expr", crate::tast::SourceLocation::unknown())
            .unwrap();
        assert!(
            matches!(key_expr, MacroValue::Expr(_)),
            "key.expr should preserve MacroValue::Expr, got {:?}",
            key_expr
        );

        // Access `.expr` again — gets the ExprDef enum representation.
        let kind_value = interp
            .field_access(&key_expr, "expr", crate::tast::SourceLocation::unknown())
            .unwrap();
        assert!(
            matches!(kind_value, MacroValue::Enum(_, _, _)),
            "key.expr.expr should produce a MacroValue::Enum, got {:?}",
            kind_value
        );

        // Sanity: unwrap-friendly fields still work — `.length` on an
        // Expr-wrapped string falls through to String.length.
        let len = interp
            .field_access(&key_expr, "length", crate::tast::SourceLocation::unknown())
            .unwrap();
        assert_eq!(len, MacroValue::Int(5));
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
