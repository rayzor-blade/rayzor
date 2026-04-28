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

use super::class_registry::ClassRegistry;
use super::context_api::MacroContext;
use super::errors::{MacroDiagnostic, MacroError};
use super::interpreter::MacroInterpreter;
use super::registry::MacroRegistry;
use super::value::MacroValue;
use crate::tast::SourceLocation;
use parser::{BlockElement, ClassFieldKind, Expr, ExprKind, HaxeFile, Metadata, TypeDeclaration};
use std::collections::BTreeMap;
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
    /// Human-readable representation of the expanded result
    pub expanded_text: Option<String>,
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
    call_cache: BTreeMap<(String, u64), Expr>,
    /// Import map for the current file: short name → qualified name
    import_map: BTreeMap<String, String>,
    /// Class registry for macro interpreter fallback dispatch
    class_registry: Arc<ClassRegistry>,
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
            call_cache: BTreeMap::new(),
            import_map: BTreeMap::new(),
            class_registry: Arc::new(ClassRegistry::new()),
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
            call_cache: BTreeMap::new(),
            import_map: BTreeMap::new(),
            class_registry: Arc::new(ClassRegistry::new()),
        }
    }

    /// Create an expander with a class registry for extended class dispatch
    pub fn with_class_registry(class_registry: ClassRegistry) -> Self {
        Self {
            registry: MacroRegistry::new(),
            context: MacroContext::new(),
            expansions_count: 0,
            max_iterations: 100,
            expansion_origins: Vec::new(),
            call_cache: BTreeMap::new(),
            import_map: BTreeMap::new(),
            class_registry: Arc::new(class_registry),
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

        // Build import map from file imports for macro interpreter resolution
        self.import_map = super::interpreter::build_import_map(&file.imports);

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

            // Phase 2b: Execute @:build macros — modify class fields before
            // expression expansion. Pass the ClassRegistry so bare-name
            // references inside the build-macro body (e.g. `Context`,
            // sibling static helpers) resolve via the short-name index.
            let build_result = super::build_macros::process_build_macros_with_class_registry(
                file,
                &self.registry,
                Some(self.class_registry.clone()),
            );
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
        let mut dirty: std::collections::BTreeSet<usize> = (0..num_decls).collect();

        while changed && iteration < self.max_iterations {
            changed = false;
            iteration += 1;

            let mut new_decls = Vec::with_capacity(num_decls);
            let mut next_dirty = std::collections::BTreeSet::new();

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
            // --- Reification fragments ---
            //
            // `macro expr`, `macro { ... }` (Reify), and `$v{...}` / `$e{...}`
            // (DollarIdent) are reification syntax: they build an `Expr` value
            // at the moment the enclosing function is executed, using that
            // function's live local scope. They are NOT compile-time constants
            // we can eagerly evaluate during the structural walk.
            //
            // Evaluating them here with a fresh empty-scope interpreter causes
            // spurious `undefined variable` errors for every local the
            // reification references (see tink/Json.hx parseString/etc.).
            // Leave them in the AST untouched; the macro interpreter handles
            // them correctly when it actually runs the containing function.
            ExprKind::Macro(_) | ExprKind::Reify(_) | ExprKind::DollarIdent { .. } => {
                Ok((expr, false))
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
        let mut interp = MacroInterpreter::with_class_registry(
            self.registry.clone(),
            self.import_map.clone(),
            self.class_registry.clone(),
        );
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
            expanded_text: Some(compact_expr_string(&expanded)),
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

        // Expand nested macro calls in arguments BEFORE entering expansion tracking.
        // This prevents false circular-dependency detection when the same macro is
        // used in both the call and its arguments (e.g., square(square(3))).
        let mut expanded_args = Vec::with_capacity(args.len());
        for arg in args {
            let (expanded, _) = self.walk_expr(arg.clone())?;
            expanded_args.push(expanded);
        }

        // Memoization: check if we've already expanded this exact call
        let args_hash = hash_exprs(&expanded_args);
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

        // Build argument values — args are passed as Expr values (not evaluated),
        // but nested macro calls have already been expanded above.
        let arg_values: Vec<MacroValue> = expanded_args
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
        let mut interp = MacroInterpreter::with_class_registry(
            self.registry.clone(),
            self.import_map.clone(),
            self.class_registry.clone(),
        );

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

        // Push the macro's class context so bare-identifier calls in the
        // body (e.g. `parseValue(...)` inside `tink.Json.parse`) can resolve
        // to sibling static helpers via the class registry.
        let class_context = macro_def
            .qualified_name
            .rsplit_once('.')
            .map(|(cls, _)| cls.to_string());
        if let Some(cls) = &class_context {
            interp.push_macro_class(cls.clone());
        }

        // Execute the macro body
        let result = interp.eval_expr(&macro_def.body);

        if class_context.is_some() {
            interp.pop_macro_class();
        }

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
            expanded_text: Some(compact_expr_string(&expanded)),
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
/// Hash expressions for memoization cache key. Uses both source spans AND
/// Compact string representation of an expanded expression (for IDE hints).
/// Shows the result value, truncated to 80 chars for readability.
fn compact_expr_string(expr: &Expr) -> String {
    let s = match &expr.kind {
        ExprKind::Int(n) => n.to_string(),
        ExprKind::Float(f) => f.to_string(),
        ExprKind::String(s) => format!("\"{}\"", s),
        ExprKind::Bool(b) => b.to_string(),
        ExprKind::Ident(id) => id.clone(),
        ExprKind::Null => "null".to_string(),
        ExprKind::Array(elems) => {
            let inner: Vec<String> = elems.iter().map(compact_expr_string).collect();
            format!("[{}]", inner.join(", "))
        }
        ExprKind::Binary { left, op, right } => {
            format!(
                "{} {:?} {}",
                compact_expr_string(left),
                op,
                compact_expr_string(right)
            )
        }
        ExprKind::Unary { op, expr: inner } => {
            format!("{:?}{}", op, compact_expr_string(inner))
        }
        ExprKind::Paren(inner) => format!("({})", compact_expr_string(inner)),
        ExprKind::Call { expr: callee, args } => {
            let args_str: Vec<String> = args.iter().map(compact_expr_string).collect();
            format!("{}({})", compact_expr_string(callee), args_str.join(", "))
        }
        ExprKind::Field {
            expr: obj,
            field,
            is_optional,
        } => {
            let op = if *is_optional { "?." } else { "." };
            format!("{}{}{}", compact_expr_string(obj), op, field)
        }
        ExprKind::Block(stmts) => {
            if stmts.len() == 1 {
                format!("{:?}", stmts[0])
            } else {
                format!("{{ ... {} statements }}", stmts.len())
            }
        }
        _ => format!("{:?}", expr.kind),
    };
    if s.len() > 80 {
        format!("{}...", &s[..77])
    } else {
        s
    }
}

/// a Debug representation of the expression kind, so expanded macro results
/// with Span::default() (from nested macro calls) don't collide.
fn hash_exprs(exprs: &[Expr]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for expr in exprs {
        expr.span.hash(&mut hasher);
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
            // Build qualified name for simple and FQN paths:
            //   Foo.bar         → "Foo.bar"
            //   tink.Json.parse → "tink.Json.parse"
            // Recurse so arbitrary-depth `pkg.sub.Class.method` works.
            let prefix = extract_macro_call_name(obj)?;
            Some(format!("{}.{}", prefix, field))
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

/// Extract the macro name from @:build metadata parameters.
/// Delegates to the shared helper so nested FQN forms like
/// `@:build(tink.Json.build)` resolve to `"tink.Json.build"` rather than
/// the leaf `"build"`.
fn extract_build_macro_name(meta: &Metadata) -> String {
    let name = super::registry::extract_macro_name_from_meta(meta);
    if name.is_empty() {
        "unknown".to_string()
    } else {
        name
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

/// Expand macros and also scan sibling/dependency files for macro definitions.
///
/// This is critical for cross-file macro discovery: when `Main.hx` calls
/// `tink.Json.parse(...)` and `tink.Json.parse` is a macro defined in
/// `tink/Json.hx`, we must scan `tink/Json.hx` first so the registry
/// knows about the macro before expansion walks the call site.
///
/// Without this, the macro call silently falls through to regular method
/// resolution and (for classes with matching simple names to stdlib) gets
/// routed to the stdlib method. Phase 2 of the macro correctness plan.
pub fn expand_macros_with_dependencies(
    file: HaxeFile,
    class_registry: ClassRegistry,
    dependency_files: &[HaxeFile],
) -> ExpansionResult {
    let mut expander = MacroExpander::with_class_registry(class_registry);

    // Pre-scan dependency files so their macro definitions are in the
    // registry before we expand the main file. Errors from scanning
    // dependencies are ignored — they'll surface when the dep file is
    // compiled on its own.
    for dep in dependency_files {
        if dep.filename == file.filename {
            continue; // skip the file we're about to expand
        }
        let _ = expander.registry.scan_and_register(dep, &dep.filename);
    }

    expander.expand_file(file)
}

/// Expand macros with a class registry for extended class dispatch.
///
/// The class registry allows the macro interpreter to call methods on any
/// class found in stdlib, imports, or user files — not just hardcoded builtins.
pub fn expand_macros_with_class_registry(
    file: HaxeFile,
    class_registry: ClassRegistry,
) -> ExpansionResult {
    let mut expander = MacroExpander::with_class_registry(class_registry);
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

    /// Phase 3 regression guard: nested FQN `@:build(tink.Json.build)` must
    /// produce the full qualified name. Previously `extract_build_macro_name`
    /// in this file walked only one `Field` level and returned `"build"`.
    #[test]
    fn test_collect_build_macros_nested_fqn() {
        let source = r#"
            @:build(tink.Json.build)
            class User {
                var name:String;
            }
        "#;
        let file = parse(source);
        let macros = collect_build_macros(&file);
        assert_eq!(macros.len(), 1);
        assert_eq!(macros[0].class_name, "User");
        assert_eq!(macros[0].macro_name, "tink.Json.build");
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

    /// Phase 4 regression guard: helper functions containing reification
    /// expressions (`macro $v{x}`, `macro true`) must not fail with
    /// "undefined variable" during the structural walk. Reification only
    /// makes sense when the macro interpreter is actually running the
    /// function body with live scope — during the pre-expansion walk,
    /// reification fragments should be left untouched.
    ///
    /// Before Phase 4, `tink/Json.hx`'s parseString/parseNumber/parseArray/
    /// parseObject each produced "undefined variable in macro: 'X'" errors
    /// during file-level expansion.
    #[test]
    fn test_reification_in_helper_function_not_evaluated() {
        let source = r#"
            class Helper {
                static function wrapInt(n:Int) {
                    return macro $v{n};
                }
                static function wrapBool(b:Bool) {
                    return macro $v{b};
                }
            }
        "#;
        let file = parse(source);
        let result = expand_macros(file);

        // No errors — reification fragments left alone.
        let errors: Vec<_> = result
            .diagnostics
            .iter()
            .filter(|d| matches!(d.severity, super::super::errors::MacroSeverity::Error))
            .collect();
        assert!(
            errors.is_empty(),
            "unexpected diagnostics: {:?}",
            errors.iter().map(|d| d.message.clone()).collect::<Vec<_>>()
        );
        // The class should still have both functions with their reification
        // bodies intact (no expansion count bumped).
        assert_eq!(result.expansions_count, 0);
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
