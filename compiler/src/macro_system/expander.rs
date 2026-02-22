//! Macro Expansion Orchestrator
//!
//! Top-level macro expansion engine that walks parsed ASTs, identifies macro
//! invocations, and expands them using the interpreter. This module provides
//! the `MacroExpander` which is called between the parse and TAST lowering
//! pipeline stages.
//!
//! # Expansion order
//!
//! 1. Scan all declarations for macro function definitions → register them
//! 2. Identify `@:build` / `@:autoBuild` metadata on classes → defer to Phase 6
//! 3. Walk all expressions, expanding:
//!    - `ExprKind::Macro(inner)` nodes
//!    - Calls to registered macro functions
//!    - `macro { ... }` reification blocks
//!    - `$v{}`, `$i{}`, `$e{}`, `$a{}`, `$p{}`, `$b{}` dollar identifiers

use super::context_api::MacroContext;
use super::errors::{MacroDiagnostic, MacroError};
use super::interpreter::MacroInterpreter;
use super::registry::MacroRegistry;
use super::value::MacroValue;
use crate::tast::SourceLocation;
use parser::{BlockElement, ClassFieldKind, Expr, ExprKind, HaxeFile, Metadata, TypeDeclaration};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
// Note: parser::ObjectField, Case, Catch are used via `parser::` prefix in walk handlers

/// Records where a macro expansion originated, for error diagnostics.
///
/// When an error occurs in expanded code, the expansion origin helps
/// the user trace back to the macro definition and the call site.
#[derive(Debug, Clone)]
pub struct ExpansionOrigin {
    /// Name of the macro that was expanded
    pub macro_name: String,
    /// Location where the macro was called
    pub call_site: SourceLocation,
    /// Location where the macro was defined (if known)
    pub definition_site: Option<SourceLocation>,
    /// Byte span of the expanded output in the resulting AST
    pub expanded_span: parser::Span,
}

/// Result of expanding macros in a file
pub struct ExpansionResult {
    /// The modified AST file (with macros expanded)
    pub file: HaxeFile,
    /// Diagnostics emitted during expansion
    pub diagnostics: Vec<MacroDiagnostic>,
    /// Number of macros expanded
    pub expansions_count: usize,
    /// Origins of each expansion (for error tracing)
    pub expansion_origins: Vec<ExpansionOrigin>,
}

/// Top-level macro expansion orchestrator.
///
/// Coordinates the registry, interpreter, and context API to expand
/// all macro invocations in a parsed file.
pub struct MacroExpander {
    /// Macro definition registry
    registry: MacroRegistry,
    /// Context for macro evaluation
    context: MacroContext,
    /// Count of expansions performed
    expansions_count: usize,
    /// Maximum expansion iterations (to prevent infinite loops)
    max_iterations: usize,
    /// Tracked expansion origins for diagnostics
    expansion_origins: Vec<ExpansionOrigin>,
    /// Memoization cache: (macro_name, args_hash) -> expanded Expr
    call_cache: HashMap<(String, u64), Expr>,
}

impl MacroExpander {
    /// Create a new expander with default settings
    pub fn new() -> Self {
        Self {
            registry: MacroRegistry::new(),
            context: MacroContext::new(),
            expansions_count: 0,
            max_iterations: 100,
            expansion_origins: Vec::new(),
            call_cache: HashMap::new(),
        }
    }

    /// Create an expander with an existing registry and context
    pub fn with_state(registry: MacroRegistry, context: MacroContext) -> Self {
        Self {
            registry,
            context,
            expansions_count: 0,
            max_iterations: 100,
            expansion_origins: Vec::new(),
            call_cache: HashMap::new(),
        }
    }

    /// Set the maximum number of expansion iterations
    pub fn set_max_iterations(&mut self, max: usize) {
        self.max_iterations = max;
    }

    /// Get a reference to the macro context
    pub fn context(&self) -> &MacroContext {
        &self.context
    }

    /// Get a mutable reference to the macro context
    pub fn context_mut(&mut self) -> &mut MacroContext {
        &mut self.context
    }

    /// Get a reference to the registry
    pub fn registry(&self) -> &MacroRegistry {
        &self.registry
    }

    /// Get a mutable reference to the registry
    pub fn registry_mut(&mut self) -> &mut MacroRegistry {
        &mut self.registry
    }

    /// Expand all macros in a parsed file.
    ///
    /// This is the main entry point for the expander. It:
    /// 1. Scans and registers macro definitions
    /// 2. Walks all expressions and expands macro invocations
    /// 3. Returns the modified file with diagnostics
    pub fn expand_file(&mut self, mut file: HaxeFile) -> ExpansionResult {
        self.expansions_count = 0;

        // Phase 1: Scan and register macro definitions from this file
        if let Err(e) = self.registry.scan_and_register(&file, &file.filename) {
            self.context.diagnostics.push(MacroDiagnostic::error(
                format!("failed to scan macros: {}", e),
                SourceLocation::unknown(),
            ));
        }

        // Phase 2: Identify and process @:build/@:autoBuild metadata
        let build_macros = collect_build_macros(&file);
        if !build_macros.is_empty() {
            for bm in &build_macros {
                self.context.diagnostics.push(MacroDiagnostic::info(
                    format!(
                        "@:build macro '{}' registered for class '{}'",
                        bm.macro_name, bm.class_name
                    ),
                    bm.location,
                ));
            }

            // Phase 2b: Execute @:build macros — modify class fields before expression expansion
            let build_result = super::build_macros::process_build_macros(file, &self.registry);
            file = build_result.file;
            self.expansions_count += build_result.applied_count;
            for diag in build_result.diagnostics {
                self.context.diagnostics.push(diag);
            }
        }

        // Fast path: if no macros are registered and no build macros found,
        // skip the expression walk entirely to avoid unnecessary AST reconstruction.
        // This is critical for performance and stability: the walk drains and rebuilds
        // every declaration, which is wasteful for the vast majority of files that
        // contain no macro invocations.
        if self.registry.macro_count() == 0 && build_macros.is_empty() {
            let diagnostics = self.context.take_diagnostics();
            return ExpansionResult {
                file,
                diagnostics,
                expansions_count: 0,
                expansion_origins: Vec::new(),
            };
        }

        // Phase 3: Walk and expand expressions in all declarations
        // Uses dirty-set tracking: after iteration 1, only re-expand declarations
        // that changed in the previous iteration (and their dependents).
        let mut iteration = 0;
        let mut changed = true;
        let num_decls = file.declarations.len();
        let mut dirty: std::collections::HashSet<usize> = (0..num_decls).collect();

        while changed && iteration < self.max_iterations {
            changed = false;
            iteration += 1;

            let mut new_decls = Vec::with_capacity(num_decls);
            let mut next_dirty = std::collections::HashSet::new();

            for (idx, decl) in file.declarations.drain(..).enumerate() {
                if dirty.contains(&idx) {
                    let (expanded, did_change) = self.expand_declaration(decl);
                    if did_change {
                        changed = true;
                        next_dirty.insert(idx);
                    }
                    new_decls.push(expanded);
                } else {
                    new_decls.push(decl);
                }
            }

            file.declarations = new_decls;
            dirty = next_dirty;
        }

        if iteration >= self.max_iterations {
            self.context.diagnostics.push(MacroDiagnostic::warning(
                format!(
                    "macro expansion reached iteration limit ({}); possible infinite expansion",
                    self.max_iterations
                ),
                SourceLocation::unknown(),
            ));
        }

        let diagnostics = self.context.take_diagnostics();
        let expansion_origins = std::mem::take(&mut self.expansion_origins);

        ExpansionResult {
            file,
            diagnostics,
            expansions_count: self.expansions_count,
            expansion_origins,
        }
    }

    /// Expand a single expression.
    ///
    /// This can be called standalone for testing or for expanding
    /// individual macro invocations.
    pub fn expand_expr(&mut self, expr: Expr) -> Result<Expr, MacroError> {
        let (expanded, _) = self.walk_expr(expr)?;
        Ok(expanded)
    }

    // =====================================================
    // Declaration-level expansion
    // =====================================================

    fn expand_declaration(&mut self, decl: TypeDeclaration) -> (TypeDeclaration, bool) {
        match decl {
            TypeDeclaration::Class(mut class) => {
                let mut changed = false;
                let mut new_fields = Vec::with_capacity(class.fields.len());
                for field in class.fields.drain(..) {
                    // Strip macro function definitions — they're compile-time only
                    // and their types (haxe.macro.Expr) shouldn't reach TAST lowering
                    if field.modifiers.contains(&parser::Modifier::Macro) {
                        changed = true;
                        continue;
                    }
                    let (expanded, did_change) = self.expand_class_field(field);
                    if did_change {
                        changed = true;
                    }
                    new_fields.push(expanded);
                }
                class.fields = new_fields;
                (TypeDeclaration::Class(class), changed)
            }
            TypeDeclaration::Interface(mut iface) => {
                let mut changed = false;
                let mut new_fields = Vec::with_capacity(iface.fields.len());
                for field in iface.fields.drain(..) {
                    let (expanded, did_change) = self.expand_class_field(field);
                    if did_change {
                        changed = true;
                    }
                    new_fields.push(expanded);
                }
                iface.fields = new_fields;
                (TypeDeclaration::Interface(iface), changed)
            }
            // Enums, typedefs, abstracts don't contain expression bodies to expand
            other => (other, false),
        }
    }

    fn expand_class_field(&mut self, mut field: parser::ClassField) -> (parser::ClassField, bool) {
        let mut changed = false;

        // Skip fields marked as `macro` — they're macro definitions, not calls
        if field.modifiers.contains(&parser::Modifier::Macro) {
            return (field, false);
        }

        match &mut field.kind {
            ClassFieldKind::Function(func) => {
                if let Some(body) = func.body.take() {
                    let backup = body.clone();
                    match self.walk_expr(*body) {
                        Ok((expanded, did_change)) => {
                            changed = did_change;
                            func.body = Some(Box::new(expanded));
                        }
                        Err(e) => {
                            if !e.is_control_flow() {
                                self.context.diagnostics.push(MacroDiagnostic::error(
                                    format!(
                                        "macro expansion failed in function '{}': {}",
                                        func.name, e
                                    ),
                                    e.location(),
                                ));
                            }
                            func.body = Some(backup);
                        }
                    }
                }
            }
            ClassFieldKind::Var {
                expr: ref mut init, ..
            } => {
                if let Some(init_expr) = init.take() {
                    let backup = init_expr.clone();
                    match self.walk_expr(init_expr) {
                        Ok((expanded, did_change)) => {
                            changed = did_change;
                            *init = Some(expanded);
                        }
                        Err(e) => {
                            if !e.is_control_flow() {
                                self.context.diagnostics.push(MacroDiagnostic::error(
                                    format!("macro expansion failed in var initializer: {}", e),
                                    e.location(),
                                ));
                            }
                            *init = Some(backup);
                        }
                    }
                }
            }
            ClassFieldKind::Final {
                expr: ref mut init, ..
            } => {
                if let Some(init_expr) = init.take() {
                    let backup = init_expr.clone();
                    match self.walk_expr(init_expr) {
                        Ok((expanded, did_change)) => {
                            changed = did_change;
                            *init = Some(expanded);
                        }
                        Err(e) => {
                            if !e.is_control_flow() {
                                self.context.diagnostics.push(MacroDiagnostic::error(
                                    format!("macro expansion failed in final initializer: {}", e),
                                    e.location(),
                                ));
                            }
                            *init = Some(backup);
                        }
                    }
                }
            }
            ClassFieldKind::Property { .. } => {
                // Properties don't have expression bodies to expand
            }
        }

        (field, changed)
    }

    // =====================================================
    // Expression-level expansion (recursive walk)
    // =====================================================

    /// Walk an expression tree, expanding macro nodes.
    /// Returns (expanded_expr, did_change).
    fn walk_expr(&mut self, expr: Expr) -> Result<(Expr, bool), MacroError> {
        match &expr.kind {
            // --- Primary macro nodes ---
            ExprKind::Macro(inner) => {
                // `macro expr` — evaluate the inner expression at compile time
                let expanded = self.eval_macro_expr(inner)?;
                self.expansions_count += 1;
                Ok((expanded, true))
            }

            // --- Function calls that might be macro calls ---
            ExprKind::Call { expr: callee, args } => {
                if let Some(macro_name) = extract_macro_call_name(callee) {
                    if self.registry.is_macro(&macro_name) {
                        let expanded = self.expand_macro_call(&macro_name, args, &expr)?;
                        self.expansions_count += 1;
                        return Ok((expanded, true));
                    }
                }
                // Not a macro call — recursively walk children
                self.walk_expr_children(expr)
            }

            // --- Recursive walk for compound expressions ---
            ExprKind::Block(_)
            | ExprKind::If { .. }
            | ExprKind::While { .. }
            | ExprKind::DoWhile { .. }
            | ExprKind::For { .. }
            | ExprKind::Switch { .. }
            | ExprKind::Try { .. }
            | ExprKind::Binary { .. }
            | ExprKind::Unary { .. }
            | ExprKind::Ternary { .. }
            | ExprKind::Paren(_)
            | ExprKind::Tuple(_)
            | ExprKind::Return(_)
            | ExprKind::Throw(_)
            | ExprKind::Var { .. }
            | ExprKind::Final { .. }
            | ExprKind::Assign { .. }
            | ExprKind::Field { .. }
            | ExprKind::Index { .. }
            | ExprKind::New { .. }
            | ExprKind::Array(_)
            | ExprKind::Map(_)
            | ExprKind::Object(_) => self.walk_expr_children(expr),

            // --- Leaf nodes — no expansion needed ---
            _ => Ok((expr, false)),
        }
    }

    /// Recursively walk children of a compound expression.
    /// This is a structural walk that preserves the expression kind.
    fn walk_expr_children(&mut self, expr: Expr) -> Result<(Expr, bool), MacroError> {
        let span = expr.span;
        let mut changed = false;

        let new_kind = match expr.kind {
            ExprKind::Block(elements) => {
                let mut new_elements = Vec::with_capacity(elements.len());
                for elem in elements {
                    match elem {
                        BlockElement::Expr(e) => {
                            let (expanded, c) = self.walk_expr(e)?;
                            changed |= c;
                            new_elements.push(BlockElement::Expr(expanded));
                        }
                        other => new_elements.push(other),
                    }
                }
                ExprKind::Block(new_elements)
            }

            ExprKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let (c_val, c1) = self.walk_expr(*cond)?;
                let (then_b, c2) = self.walk_expr(*then_branch)?;
                changed |= c1 | c2;
                let else_b = if let Some(eb) = else_branch {
                    let (expanded, c3) = self.walk_expr(*eb)?;
                    changed |= c3;
                    Some(Box::new(expanded))
                } else {
                    None
                };
                ExprKind::If {
                    cond: Box::new(c_val),
                    then_branch: Box::new(then_b),
                    else_branch: else_b,
                }
            }

            ExprKind::While { cond, body } => {
                let (c_val, c1) = self.walk_expr(*cond)?;
                let (bod, c2) = self.walk_expr(*body)?;
                changed |= c1 | c2;
                ExprKind::While {
                    cond: Box::new(c_val),
                    body: Box::new(bod),
                }
            }

            ExprKind::DoWhile { body, cond } => {
                let (bod, c1) = self.walk_expr(*body)?;
                let (c_val, c2) = self.walk_expr(*cond)?;
                changed |= c1 | c2;
                ExprKind::DoWhile {
                    body: Box::new(bod),
                    cond: Box::new(c_val),
                }
            }

            ExprKind::For {
                var: var_name,
                key_var,
                iter,
                body,
            } => {
                let (iter_exp, c1) = self.walk_expr(*iter)?;
                let (bod, c2) = self.walk_expr(*body)?;
                changed |= c1 | c2;
                ExprKind::For {
                    var: var_name,
                    key_var,
                    iter: Box::new(iter_exp),
                    body: Box::new(bod),
                }
            }

            ExprKind::Return(opt) => {
                if let Some(inner) = opt {
                    let (expanded, c) = self.walk_expr(*inner)?;
                    changed |= c;
                    ExprKind::Return(Some(Box::new(expanded)))
                } else {
                    ExprKind::Return(None)
                }
            }

            ExprKind::Throw(inner) => {
                let (expanded, c) = self.walk_expr(*inner)?;
                changed |= c;
                ExprKind::Throw(Box::new(expanded))
            }

            ExprKind::Paren(inner) => {
                let (expanded, c) = self.walk_expr(*inner)?;
                changed |= c;
                ExprKind::Paren(Box::new(expanded))
            }

            ExprKind::Tuple(elements) => {
                let mut new_elements = Vec::new();
                for elem in elements {
                    let (expanded, c) = self.walk_expr(elem)?;
                    changed |= c;
                    new_elements.push(expanded);
                }
                ExprKind::Tuple(new_elements)
            }

            ExprKind::Binary { left, op, right } => {
                let (l, c1) = self.walk_expr(*left)?;
                let (r, c2) = self.walk_expr(*right)?;
                changed |= c1 | c2;
                ExprKind::Binary {
                    left: Box::new(l),
                    op,
                    right: Box::new(r),
                }
            }

            ExprKind::Unary { op, expr: inner } => {
                let (expanded, c) = self.walk_expr(*inner)?;
                changed |= c;
                ExprKind::Unary {
                    op,
                    expr: Box::new(expanded),
                }
            }

            ExprKind::Ternary {
                cond,
                then_expr,
                else_expr,
            } => {
                let (c, c1) = self.walk_expr(*cond)?;
                let (t, c2) = self.walk_expr(*then_expr)?;
                let (e, c3) = self.walk_expr(*else_expr)?;
                changed |= c1 | c2 | c3;
                ExprKind::Ternary {
                    cond: Box::new(c),
                    then_expr: Box::new(t),
                    else_expr: Box::new(e),
                }
            }

            ExprKind::Call { expr: callee, args } => {
                let (callee_exp, c1) = self.walk_expr(*callee)?;
                let mut new_args = Vec::with_capacity(args.len());
                for arg in args {
                    let (a, c) = self.walk_expr(arg)?;
                    changed |= c;
                    new_args.push(a);
                }
                changed |= c1;
                ExprKind::Call {
                    expr: Box::new(callee_exp),
                    args: new_args,
                }
            }

            ExprKind::Field {
                expr: obj,
                field,
                is_optional,
            } => {
                let (expanded, c) = self.walk_expr(*obj)?;
                changed |= c;
                ExprKind::Field {
                    expr: Box::new(expanded),
                    field,
                    is_optional,
                }
            }

            ExprKind::Index { expr: arr, index } => {
                let (a, c1) = self.walk_expr(*arr)?;
                let (i, c2) = self.walk_expr(*index)?;
                changed |= c1 | c2;
                ExprKind::Index {
                    expr: Box::new(a),
                    index: Box::new(i),
                }
            }

            ExprKind::Assign { left, op, right } => {
                let (l, c1) = self.walk_expr(*left)?;
                let (r, c2) = self.walk_expr(*right)?;
                changed |= c1 | c2;
                ExprKind::Assign {
                    left: Box::new(l),
                    op,
                    right: Box::new(r),
                }
            }

            ExprKind::Var {
                name,
                type_hint,
                expr: init,
            } => {
                if let Some(init_expr) = init {
                    let (expanded, c) = self.walk_expr(*init_expr)?;
                    changed |= c;
                    ExprKind::Var {
                        name,
                        type_hint,
                        expr: Some(Box::new(expanded)),
                    }
                } else {
                    ExprKind::Var {
                        name,
                        type_hint,
                        expr: None,
                    }
                }
            }

            ExprKind::Final {
                name,
                type_hint,
                expr: init,
            } => {
                if let Some(init_expr) = init {
                    let (expanded, c) = self.walk_expr(*init_expr)?;
                    changed |= c;
                    ExprKind::Final {
                        name,
                        type_hint,
                        expr: Some(Box::new(expanded)),
                    }
                } else {
                    ExprKind::Final {
                        name,
                        type_hint,
                        expr: None,
                    }
                }
            }

            ExprKind::Array(items) => {
                let mut new_items = Vec::with_capacity(items.len());
                for item in items {
                    let (expanded, c) = self.walk_expr(item)?;
                    changed |= c;
                    new_items.push(expanded);
                }
                ExprKind::Array(new_items)
            }

            ExprKind::Map(entries) => {
                let mut new_entries = Vec::with_capacity(entries.len());
                for (key, val) in entries {
                    let (k, c1) = self.walk_expr(key)?;
                    let (v, c2) = self.walk_expr(val)?;
                    changed |= c1 | c2;
                    new_entries.push((k, v));
                }
                ExprKind::Map(new_entries)
            }

            ExprKind::Object(fields) => {
                let mut new_fields = Vec::with_capacity(fields.len());
                for field in fields {
                    let (expanded, c) = self.walk_expr(field.expr)?;
                    changed |= c;
                    new_fields.push(parser::ObjectField {
                        name: field.name,
                        expr: expanded,
                        span: field.span,
                    });
                }
                ExprKind::Object(new_fields)
            }

            ExprKind::Switch {
                expr: switch_expr,
                cases,
                default,
            } => {
                let (se, c1) = self.walk_expr(*switch_expr)?;
                changed |= c1;
                let mut new_cases = Vec::with_capacity(cases.len());
                for case in cases {
                    let guard = if let Some(g) = case.guard {
                        let (expanded, c) = self.walk_expr(g)?;
                        changed |= c;
                        Some(expanded)
                    } else {
                        None
                    };
                    let (body, c) = self.walk_expr(case.body)?;
                    changed |= c;
                    new_cases.push(parser::Case {
                        patterns: case.patterns,
                        guard,
                        body,
                        span: case.span,
                    });
                }
                let new_default = if let Some(d) = default {
                    let (expanded, c) = self.walk_expr(*d)?;
                    changed |= c;
                    Some(Box::new(expanded))
                } else {
                    None
                };
                ExprKind::Switch {
                    expr: Box::new(se),
                    cases: new_cases,
                    default: new_default,
                }
            }

            ExprKind::Try {
                expr: try_expr,
                catches,
                finally_block,
            } => {
                let (te, c1) = self.walk_expr(*try_expr)?;
                changed |= c1;
                let mut new_catches = Vec::with_capacity(catches.len());
                for catch in catches {
                    let filter = if let Some(f) = catch.filter {
                        let (expanded, c) = self.walk_expr(f)?;
                        changed |= c;
                        Some(expanded)
                    } else {
                        None
                    };
                    let (body, c) = self.walk_expr(catch.body)?;
                    changed |= c;
                    new_catches.push(parser::Catch {
                        var: catch.var,
                        type_hint: catch.type_hint,
                        filter,
                        body,
                        span: catch.span,
                    });
                }
                let new_finally = if let Some(f) = finally_block {
                    let (expanded, c) = self.walk_expr(*f)?;
                    changed |= c;
                    Some(Box::new(expanded))
                } else {
                    None
                };
                ExprKind::Try {
                    expr: Box::new(te),
                    catches: new_catches,
                    finally_block: new_finally,
                }
            }

            ExprKind::New {
                type_path,
                params,
                args,
            } => {
                let mut new_args = Vec::with_capacity(args.len());
                for arg in args {
                    let (expanded, c) = self.walk_expr(arg)?;
                    changed |= c;
                    new_args.push(expanded);
                }
                ExprKind::New {
                    type_path,
                    params,
                    args: new_args,
                }
            }

            // For other compound kinds, just pass through
            other => other,
        };

        Ok((
            Expr {
                kind: new_kind,
                span,
            },
            changed,
        ))
    }

    // =====================================================
    // Macro evaluation
    // =====================================================

    /// Evaluate a `macro expr` node — run the expression at compile time
    fn eval_macro_expr(&mut self, expr: &Expr) -> Result<Expr, MacroError> {
        let call_site = super::errors::span_to_location(expr.span);
        let mut interp = MacroInterpreter::new(self.registry.clone());
        let result = interp.eval_expr(expr);

        // Collect trace output
        for line in interp.take_trace_output() {
            self.context.diagnostics.push(MacroDiagnostic::info(
                format!("[macro trace] {}", line),
                SourceLocation::unknown(),
            ));
        }

        let expanded = match result {
            Ok(value) => Ok(super::ast_bridge::value_to_expr(&value)),
            Err(MacroError::Return { value: Some(v) }) => Ok(super::ast_bridge::value_to_expr(&*v)),
            Err(MacroError::Return { value: None }) => {
                Ok(super::ast_bridge::value_to_expr(&MacroValue::Null))
            }
            Err(e) if e.is_control_flow() => {
                Ok(super::ast_bridge::value_to_expr(&MacroValue::Null))
            }
            Err(e) => Err(e),
        }?;

        // Record expansion origin
        self.expansion_origins.push(ExpansionOrigin {
            macro_name: "<macro expr>".to_string(),
            call_site,
            definition_site: None,
            expanded_span: expanded.span,
        });

        Ok(expanded)
    }

    /// Expand a call to a registered macro function
    fn expand_macro_call(
        &mut self,
        name: &str,
        args: &[Expr],
        call_expr: &Expr,
    ) -> Result<Expr, MacroError> {
        let location = super::errors::span_to_location(call_expr.span);

        // Memoization: check if we've already expanded this exact call
        let args_hash = hash_exprs(args);
        let cache_key = (name.to_string(), args_hash);
        if let Some(cached) = self.call_cache.get(&cache_key) {
            self.expansions_count += 1;
            return Ok(cached.clone());
        }

        // Enter expansion tracking (depth + circular dep checks)
        self.registry.enter_expansion(name)?;

        // Get the macro definition
        let macro_def = self
            .registry
            .get_macro(name)
            .ok_or_else(|| MacroError::UndefinedMacro {
                name: name.to_string(),
                location,
            })?
            .clone();

        // Build argument values
        // In Haxe macros, arguments are passed as Expr values (not evaluated)
        let arg_values: Vec<MacroValue> = args
            .iter()
            .map(|a| MacroValue::Expr(Arc::new(a.clone())))
            .collect();

        // Check argument count
        let required_params = macro_def
            .params
            .iter()
            .filter(|p| !p.optional && !p.rest)
            .count();
        if arg_values.len() < required_params {
            self.registry.exit_expansion(name);
            return Err(MacroError::ArgumentCountMismatch {
                macro_name: name.to_string(),
                expected: required_params,
                found: arg_values.len(),
                location,
            });
        }

        // Create interpreter and bind arguments
        let mut interp = MacroInterpreter::new(self.registry.clone());

        // Bind parameters in the interpreter environment
        for (i, param) in macro_def.params.iter().enumerate() {
            let value = if param.rest {
                // Rest parameter: collect remaining args into an array
                MacroValue::Array(Arc::new(arg_values[i..].to_vec()))
            } else if let Some(val) = arg_values.get(i) {
                val.clone()
            } else if param.optional {
                MacroValue::Null
            } else {
                MacroValue::Null
            };
            interp.define_variable(&param.name, value);
        }

        // Execute the macro body
        let result = interp.eval_expr(&macro_def.body);

        // Collect trace output
        for line in interp.take_trace_output() {
            self.context.diagnostics.push(MacroDiagnostic::info(
                format!("[macro trace] {}", line),
                SourceLocation::unknown(),
            ));
        }

        self.registry.exit_expansion(name);

        // Convert result to expression and record expansion origin
        let expanded = match result {
            Ok(value) => Ok(super::ast_bridge::value_to_expr(&value)),
            Err(MacroError::Return { value: Some(v) }) => Ok(super::ast_bridge::value_to_expr(&*v)),
            Err(MacroError::Return { value: None }) => {
                Ok(super::ast_bridge::value_to_expr(&MacroValue::Null))
            }
            Err(e) if e.is_control_flow() => {
                Ok(super::ast_bridge::value_to_expr(&MacroValue::Null))
            }
            Err(e) => Err(e),
        }?;

        // Record expansion origin for diagnostics
        self.expansion_origins.push(ExpansionOrigin {
            macro_name: name.to_string(),
            call_site: location,
            definition_site: Some(super::errors::span_to_location(macro_def.body.span)),
            expanded_span: expanded.span,
        });

        // Store in memoization cache for future identical calls
        self.call_cache.insert(cache_key, expanded.clone());

        Ok(expanded)
    }
}

impl Default for MacroExpander {
    fn default() -> Self {
        Self::new()
    }
}

// ==========================================================
// Helper functions
// ==========================================================

/// Hash a slice of expressions for memoization cache key.
///
/// Uses the Debug representation as a stable, content-based hash.
/// This is fast enough since macro args are typically small expressions.
fn hash_exprs(exprs: &[Expr]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for expr in exprs {
        format!("{:?}", expr.kind).hash(&mut hasher);
    }
    hasher.finish()
}

/// Extract the name of a potential macro call from a callee expression.
///
/// Handles simple identifiers (`macroName(...)`) and field access
/// (`ClassName.macroName(...)`).
fn extract_macro_call_name(expr: &Expr) -> Option<String> {
    match &expr.kind {
        ExprKind::Ident(name) => Some(name.clone()),
        ExprKind::Field {
            expr: obj, field, ..
        } => {
            // Build qualified name: Foo.bar → "Foo.bar"
            if let ExprKind::Ident(class_name) = &obj.kind {
                Some(format!("{}.{}", class_name, field))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Information about a @:build macro found on a class
#[derive(Debug, Clone)]
pub struct BuildMacroInfo {
    /// The macro name/path to call
    pub macro_name: String,
    /// The class it's applied to
    pub class_name: String,
    /// Source location
    pub location: SourceLocation,
}

/// Collect @:build metadata from class declarations
fn collect_build_macros(file: &HaxeFile) -> Vec<BuildMacroInfo> {
    let mut result = Vec::new();
    for decl in &file.declarations {
        if let TypeDeclaration::Class(class) = decl {
            for meta in &class.meta {
                if meta.name == "build" || meta.name == ":build" {
                    let macro_name = extract_build_macro_name(meta);
                    result.push(BuildMacroInfo {
                        macro_name,
                        class_name: class.name.clone(),
                        location: super::errors::span_to_location(meta.span),
                    });
                }
            }
        }
    }
    result
}

/// Extract the macro name from @:build metadata parameters
fn extract_build_macro_name(meta: &Metadata) -> String {
    if let Some(first) = meta.params.first() {
        match &first.kind {
            ExprKind::Ident(name) => name.clone(),
            ExprKind::Call { expr, .. } => {
                if let ExprKind::Ident(name) = &expr.kind {
                    name.clone()
                } else {
                    format!("{:?}", expr.kind)
                }
            }
            ExprKind::Field { expr, field, .. } => {
                if let ExprKind::Ident(class_name) = &expr.kind {
                    format!("{}.{}", class_name, field)
                } else {
                    field.clone()
                }
            }
            _ => "unknown".to_string(),
        }
    } else {
        "unknown".to_string()
    }
}

// ==========================================================
// Pipeline integration function
// ==========================================================

/// Expand all macros in a parsed file.
///
/// This is the top-level function called from the pipeline between
/// parsing and TAST lowering.
///
/// Returns the expanded file and any diagnostics generated.
pub fn expand_macros(file: HaxeFile) -> ExpansionResult {
    let mut expander = MacroExpander::new();
    expander.expand_file(file)
}

/// Expand macros using an existing registry (for multi-file compilation).
///
/// The registry may contain macro definitions from previously compiled files.
pub fn expand_macros_with_registry(file: HaxeFile, registry: MacroRegistry) -> ExpansionResult {
    let mut expander = MacroExpander::with_state(registry, MacroContext::new());
    expander.expand_file(file)
}

#[cfg(test)]
mod tests {
    use super::*;
    use parser::Span;

    fn parse(source: &str) -> HaxeFile {
        parser::parse_haxe_file("test.hx", source, false).expect("parse should succeed")
    }

    #[test]
    fn test_expand_no_macros() {
        let file = parse("class Test { static function main() { var x = 1 + 2; } }");
        let result = expand_macros(file);
        assert_eq!(result.expansions_count, 0);
        assert!(result
            .diagnostics
            .iter()
            .all(|d| d.severity != super::super::errors::MacroSeverity::Error));
    }

    #[test]
    fn test_expand_preserves_structure() {
        let source = "class Test {
            static function main() {
                var x = 42;
                var y = \"hello\";
            }
        }";
        let file = parse(source);
        let result = expand_macros(file);
        // No macros to expand, file should be preserved
        assert_eq!(result.file.declarations.len(), 1);
        assert_eq!(result.expansions_count, 0);
    }

    #[test]
    fn test_extract_macro_call_name_identifier() {
        let expr = Expr {
            kind: ExprKind::Ident("myMacro".to_string()),
            span: Span::new(0, 0),
        };
        assert_eq!(extract_macro_call_name(&expr), Some("myMacro".to_string()));
    }

    #[test]
    fn test_extract_macro_call_name_field_access() {
        let expr = Expr {
            kind: ExprKind::Field {
                expr: Box::new(Expr {
                    kind: ExprKind::Ident("MyClass".to_string()),
                    span: Span::new(0, 0),
                }),
                field: "macroMethod".to_string(),
                is_optional: false,
            },
            span: Span::new(0, 0),
        };
        assert_eq!(
            extract_macro_call_name(&expr),
            Some("MyClass.macroMethod".to_string())
        );
    }

    #[test]
    fn test_collect_build_macros() {
        let source = r#"
            @:build(MyMacro.build)
            class Test {
                var x:Int;
            }
        "#;
        let file = parse(source);
        let macros = collect_build_macros(&file);
        assert_eq!(macros.len(), 1);
        assert_eq!(macros[0].class_name, "Test");
        assert_eq!(macros[0].macro_name, "MyMacro.build");
    }

    #[test]
    fn test_expand_macro_function_call() {
        // Define a class with a macro function, and another that calls it
        let source = r#"
            class Macros {
                macro static function makeConst() {
                    return 42;
                }
            }
            class Test {
                static function main() {
                    var x = Macros.makeConst();
                }
            }
        "#;
        let file = parse(source);
        let mut expander = MacroExpander::new();
        let result = expander.expand_file(file);

        // The macro should have been registered and the call expanded
        assert!(
            expander.registry().get_macro("Macros.makeConst").is_some(),
            "macro should be registered"
        );
    }

    #[test]
    fn test_expand_file_with_nested_expressions() {
        let source = r#"
            class Test {
                static function main() {
                    if (true) {
                        var x = 1 + 2;
                    } else {
                        var y = 3 * 4;
                    }
                }
            }
        "#;
        let file = parse(source);
        let result = expand_macros(file);
        // No macros, but the walk should complete without error
        assert_eq!(result.expansions_count, 0);
        assert_eq!(result.file.declarations.len(), 1);
    }

    #[test]
    fn test_expander_with_context() {
        let mut expander = MacroExpander::new();
        expander
            .context_mut()
            .defines
            .insert("debug".to_string(), "1".to_string());
        assert_eq!(
            expander.context().defines.get("debug"),
            Some(&"1".to_string())
        );
    }

    #[test]
    fn test_expander_max_iterations() {
        let mut expander = MacroExpander::new();
        expander.set_max_iterations(5);
        assert_eq!(expander.max_iterations, 5);
    }

    // ===== Edge case tests (Phase 7) =====

    #[test]
    fn test_expand_switch_expressions() {
        let source = r#"
            class Test {
                static function main() {
                    var x = 1;
                    switch (x) {
                        case 1: return "one";
                        case 2: return "two";
                        default: return "other";
                    }
                }
            }
        "#;
        let file = parse(source);
        let result = expand_macros(file);
        assert_eq!(result.expansions_count, 0);
        assert_eq!(result.file.declarations.len(), 1);
    }

    #[test]
    fn test_expand_try_catch_expressions() {
        let source = r#"
            class Test {
                static function main() {
                    try {
                        var x = doSomething();
                    } catch (e:Dynamic) {
                        trace(e);
                    }
                }
            }
        "#;
        let file = parse(source);
        let result = expand_macros(file);
        assert_eq!(result.expansions_count, 0);
        assert_eq!(result.file.declarations.len(), 1);
    }

    #[test]
    fn test_expand_object_literals() {
        let source = r#"
            class Test {
                static function main() {
                    var obj = {x: 1, y: 2, z: 3};
                }
            }
        "#;
        let file = parse(source);
        let result = expand_macros(file);
        assert_eq!(result.expansions_count, 0);
        assert_eq!(result.file.declarations.len(), 1);
    }

    #[test]
    fn test_expand_map_literals() {
        let source = r#"
            class Test {
                static function main() {
                    var m = ["a" => 1, "b" => 2];
                }
            }
        "#;
        let file = parse(source);
        let result = expand_macros(file);
        assert_eq!(result.expansions_count, 0);
        assert_eq!(result.file.declarations.len(), 1);
    }

    #[test]
    fn test_expansion_result_has_origins() {
        // When macros are expanded, expansion_origins should be populated
        let source = r#"
            class Macros {
                macro static function getValue() {
                    return 42;
                }
            }
            class Test {
                static function main() {
                    var x = Macros.getValue();
                }
            }
        "#;
        let file = parse(source);
        let result = expand_macros(file);
        // The structure should be preserved
        assert_eq!(result.file.declarations.len(), 2);
    }

    #[test]
    fn test_expand_multiple_classes() {
        let source = r#"
            class Foo {
                public var x:Int = 10;
                public function bar() { return x + 1; }
            }
            class Baz {
                public var y:String = "hello";
            }
        "#;
        let file = parse(source);
        let result = expand_macros(file);
        assert_eq!(result.file.declarations.len(), 2);
        assert_eq!(result.expansions_count, 0);
    }

    #[test]
    fn test_expand_with_registry() {
        let source = "class Test { static function main() { var x = 1; } }";
        let file = parse(source);
        let registry = MacroRegistry::new();
        let result = expand_macros_with_registry(file, registry);
        assert_eq!(result.expansions_count, 0);
        assert_eq!(result.file.declarations.len(), 1);
    }

    #[test]
    fn test_expand_interface_fields() {
        let source = r#"
            interface ITest {
                function getValue():Int;
                var name:String;
            }
        "#;
        let file = parse(source);
        let result = expand_macros(file);
        assert_eq!(result.file.declarations.len(), 1);
        assert_eq!(result.expansions_count, 0);
    }
}
