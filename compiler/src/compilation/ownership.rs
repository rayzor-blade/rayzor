//! The ownership check reported against a typed file.

use super::*;

impl CompilationUnit {

    /// Check for ownership violations (use-after-move) in the TAST.
    /// Returns diagnostics that can be printed via print_mir_diagnostics.
    pub(crate) fn check_ownership_violations(&self, typed_file: &TypedFile) -> Vec<diagnostics::Diagnostic> {
        use crate::semantic_graph::{MoveType, OwnershipGraph};
        use crate::tast::{ScopeId, TypedExpressionKind, TypedStatement};

        let mut ownership_graph = OwnershipGraph::new();

        // Walk all classes and standalone functions to populate the ownership graph
        for class in &typed_file.classes {
            for method in &class.methods {
                let scope = ScopeId::from_raw(method.symbol_id.as_raw());
                for param in &method.parameters {
                    ownership_graph.add_variable(param.symbol_id, param.param_type, scope);
                }
                Self::populate_ownership_stmts(&mut ownership_graph, &method.body);
            }
            for ctor in &class.constructors {
                let scope = ScopeId::from_raw(ctor.symbol_id.as_raw());
                for param in &ctor.parameters {
                    ownership_graph.add_variable(param.symbol_id, param.param_type, scope);
                }
                Self::populate_ownership_stmts(&mut ownership_graph, &ctor.body);
            }
        }
        for func in &typed_file.functions {
            let scope = ScopeId::from_raw(func.symbol_id.as_raw());
            for param in &func.parameters {
                ownership_graph.add_variable(param.symbol_id, param.param_type, scope);
            }
            Self::populate_ownership_stmts(&mut ownership_graph, &func.body);
        }

        // Check for use-after-move violations
        let violations = ownership_graph.check_use_after_move();
        let mut diagnostics = Vec::new();

        // Build a trait checker so we can drop violations on `Copy` types —
        // Int / Float / Bool and any class deriving Copy are pass-by-value
        // semantically (matches Haxe + Rust), so `i++` in a for-loop or
        // passing an Int to a function does NOT consume the variable. The
        // ownership graph conservatively records all variable references
        // as moves; this filter removes the false positives at diagnostic
        // emission time. Classes that genuinely need move semantics (no
        // Copy derive) still surface their warnings.
        // Stdlib classes (Tensor, QTensor, …) live in `loaded_stdlib_typed_files`,
        // not on the current `typed_file`. Without folding them into the trait
        // checker's class map, `@:move` annotations on stdlib types are silently
        // inert at user-call sites (requires_strict_move would return false
        // because the class wouldn't be found at all). Chain every loaded stdlib
        // file's classes into the lookup so cross-file move semantics fire.
        let mut trait_checker = crate::tast::trait_checker::TraitChecker::new(
            self.type_table.as_ref(),
            &self.symbol_table,
            &self.string_interner,
            &typed_file.classes,
        );
        for stdlib_file in &self.loaded_stdlib_typed_files {
            trait_checker = trait_checker.extend_classes(&stdlib_file.classes);
        }

        for violation in violations {
            if let crate::semantic_graph::OwnershipViolation::UseAfterMove {
                variable,
                use_location,
                move_location,
                ..
            } = violation
            {
                // Skip if the variable is Copy — `i` in `i++`, primitives
                // passed to functions, etc. shouldn't fire the warning.
                // Also decide up-front whether the variable's class is
                // `@:move`-annotated; if so, the diagnostic is a hard error
                // (linear/affine semantics) rather than a soft warning.
                // `@:move` (strict_q) takes precedence over auto-Copy: when the
                // user explicitly opts into move semantics, we must NOT silently
                // treat the value as Copy even if all its fields happen to be
                // Copy-able. `is_copy` only skips the diagnostic when there is
                // no `@:move` annotation.
                let mut strict = false;
                if let Some(node) = ownership_graph.variables.get(&variable) {
                    // An unresolved type cannot be shown to have move semantics.
                    // Reporting one anyway turns a resolution gap into a
                    // use-after-move accusation against code that has none.
                    if !node.variable_type.is_valid() {
                        continue;
                    }
                    {
                        // `@:shared` short-circuits the entire diagnostic.
                        // Bindings of shared classes (e.g. rayzor.ds.Tensor)
                        // are reference-counted at runtime; aliasing them
                        // after a `.clone()` (which is now an atomic
                        // refcount increment) is not a use-after-move and
                        // must not produce E0382 — neither error nor
                        // warning. Skip ahead of the is_copy check so we
                        // don't even traverse the per-callsite work.
                        if trait_checker.requires_shared(node.variable_type) {
                            continue;
                        }
                        let strict_q = trait_checker.requires_strict_move(node.variable_type);
                        // Copy OR Clone: neither is consumed by being read. Copy
                        // covers Int/Float/Bool; Clone additionally covers String,
                        // which owns a heap buffer (so it is not Copy) but is
                        // immutable and duplicable — passing one to a function
                        // does not end the caller's binding, and reporting that it
                        // does was the single remaining false positive on ordinary
                        // Haxe. `@:move` still wins: an explicit opt-in to move
                        // semantics is never overridden by a derivable trait.
                        if !strict_q
                            && (trait_checker.is_copy(node.variable_type)
                                || trait_checker.is_clone(node.variable_type))
                        {
                            continue;
                        }
                        // `@:move` types belong to the MIR move analysis now.
                        // It works on bindings over the control-flow graph and
                        // reads `@:borrow` / `@:owned` / `@:consume`, none of
                        // which this pass knows about — so leaving it enabled
                        // here means it contradicts the analysis that is right,
                        // reporting a borrowed argument as consumed.
                        if strict_q {
                            continue;
                        }
                        strict = strict_q;
                    }
                }

                let var_name = self.get_symbol_name(variable, typed_file);
                // Opt-in debug for triaging E0382 sites — set RAYZOR_DEBUG_E0382 to
                // print each violation's (var, symbol, file, line, col) so the
                // diagnostic's "Main.hx fallback" rendering can be cross-referenced
                // against the real source location.
                if std::env::var("RAYZOR_DEBUG_E0382").is_ok() {
                    eprintln!("[E0382-DEBUG] var={} sym={} typed_file={} severity={} move_file_id={} move_line={} move_col={} use_file_id={} use_line={} use_col={}",
                        var_name,
                        variable.as_raw(),
                        typed_file.metadata.file_path,
                        if strict { "Error" } else { "Warning" },
                        move_location.file_id,
                        move_location.line,
                        move_location.column,
                        use_location.file_id,
                        use_location.line,
                        use_location.column,
                    );
                }
                let file_id = diagnostics::FileId::new(use_location.file_id as usize);
                // Span the entire identifier — we know `var_name` so the
                // end position is start + len-bytes. Previously this
                // was start + 1, highlighting just the first character.
                // var_name is the source identifier as the parser saw it
                // (Haxe identifiers are ASCII-only so byte_len == char_len
                // for column math).
                let name_byte_len = var_name.len();
                let use_start = diagnostics::SourcePosition::new(
                    use_location.line as usize,
                    use_location.column as usize,
                    use_location.byte_offset as usize,
                );
                let use_end = diagnostics::SourcePosition::new(
                    use_location.line as usize,
                    use_location.column as usize + name_byte_len,
                    use_location.byte_offset as usize + name_byte_len,
                );
                let use_span = diagnostics::SourceSpan::new(use_start, use_end, file_id);

                let move_start = diagnostics::SourcePosition::new(
                    move_location.line as usize,
                    move_location.column as usize,
                    move_location.byte_offset as usize,
                );
                let move_end = diagnostics::SourcePosition::new(
                    move_location.line as usize,
                    move_location.column as usize + name_byte_len,
                    move_location.byte_offset as usize + name_byte_len,
                );
                let move_span = diagnostics::SourceSpan::new(move_start, move_end, file_id);

                let help = if strict {
                    vec![
                        format!(
                            "`{}` is declared `@:move`, so its values cannot be aliased after a move.",
                            var_name
                        ),
                        format!(
                            "Clone the value explicitly (`var copy = {}.clone();`) or restructure the code so the original binding is no longer reachable.",
                            var_name
                        ),
                    ]
                } else {
                    vec![format!(
                        "Consider cloning: `var copy = {}.clone();`",
                        var_name
                    )]
                };
                let diag = diagnostics::Diagnostic {
                    severity: if strict {
                        diagnostics::DiagnosticSeverity::Error
                    } else {
                        diagnostics::DiagnosticSeverity::Warning
                    },
                    code: Some("E0382".to_string()),
                    message: format!("use of moved value: `{}`", var_name),
                    span: use_span.clone(),
                    labels: vec![
                        diagnostics::Label::primary(use_span, "value used here after move"),
                        diagnostics::Label::secondary(move_span, "value moved here"),
                    ],
                    suggestions: vec![],
                    notes: vec![],
                    help,
                };
                diagnostics.push(diag);
            }
        }

        diagnostics
    }


    /// Walk statements to populate ownership graph (moves and uses).
    pub(crate) fn populate_ownership_stmts(
        graph: &mut crate::semantic_graph::OwnershipGraph,
        stmts: &[crate::tast::TypedStatement],
    ) {
        use crate::semantic_graph::MoveType;
        use crate::tast::{ScopeId, TypedExpressionKind, TypedStatement};

        for stmt in stmts {
            match stmt {
                TypedStatement::VarDeclaration {
                    symbol_id,
                    var_type,
                    initializer,
                    ..
                } => {
                    let scope = ScopeId::from_raw(symbol_id.as_raw());
                    graph.add_variable(*symbol_id, *var_type, scope);
                    if let Some(init) = initializer {
                        if let TypedExpressionKind::Variable { symbol_id: src } = &init.kind {
                            graph.add_move(
                                *src,
                                Some(*symbol_id),
                                init.source_location,
                                MoveType::Explicit,
                            );
                        }
                        Self::populate_ownership_expr(graph, init);
                    }
                }
                TypedStatement::Expression { expression, .. } => {
                    Self::populate_ownership_expr(graph, expression);
                }
                TypedStatement::Return { value, .. } => {
                    if let Some(expr) = value {
                        Self::populate_ownership_expr(graph, expr);
                    }
                }
                TypedStatement::If {
                    condition,
                    then_branch,
                    else_branch,
                    ..
                } => {
                    Self::populate_ownership_expr(graph, condition);
                    Self::populate_ownership_stmts(
                        graph,
                        std::slice::from_ref(then_branch.as_ref()),
                    );
                    if let Some(else_stmt) = else_branch {
                        Self::populate_ownership_stmts(
                            graph,
                            std::slice::from_ref(else_stmt.as_ref()),
                        );
                    }
                }
                TypedStatement::While {
                    condition, body, ..
                } => {
                    Self::populate_ownership_expr(graph, condition);
                    Self::populate_ownership_stmts(graph, std::slice::from_ref(body.as_ref()));
                }
                TypedStatement::Block { statements, .. } => {
                    Self::populate_ownership_stmts(graph, statements);
                }
                _ => {}
            }
        }
    }


    /// Walk expressions to record moves (function call args) and uses (variable refs).
    pub(crate) fn populate_ownership_expr(
        graph: &mut crate::semantic_graph::OwnershipGraph,
        expr: &crate::tast::TypedExpression,
    ) {
        use crate::semantic_graph::MoveType;
        use crate::tast::TypedExpressionKind;

        match &expr.kind {
            TypedExpressionKind::Variable { symbol_id } => {
                graph.record_use(*symbol_id, expr.source_location);
            }
            TypedExpressionKind::FieldAccess { object, .. } => {
                Self::populate_ownership_expr(graph, object);
            }
            TypedExpressionKind::FunctionCall {
                function,
                arguments,
                ..
            } => {
                Self::populate_ownership_expr(graph, function);
                for arg in arguments {
                    if let TypedExpressionKind::Variable { symbol_id } = &arg.kind {
                        graph.add_move(
                            *symbol_id,
                            None,
                            arg.source_location,
                            MoveType::FunctionCall,
                        );
                    }
                    Self::populate_ownership_expr(graph, arg);
                }
            }
            TypedExpressionKind::MethodCall {
                receiver,
                arguments,
                ..
            } => {
                Self::populate_ownership_expr(graph, receiver);
                for arg in arguments {
                    if let TypedExpressionKind::Variable { symbol_id } = &arg.kind {
                        graph.add_move(
                            *symbol_id,
                            None,
                            arg.source_location,
                            MoveType::FunctionCall,
                        );
                    }
                    Self::populate_ownership_expr(graph, arg);
                }
            }
            TypedExpressionKind::StaticMethodCall { arguments, .. } => {
                for arg in arguments {
                    if let TypedExpressionKind::Variable { symbol_id } = &arg.kind {
                        graph.add_move(
                            *symbol_id,
                            None,
                            arg.source_location,
                            MoveType::FunctionCall,
                        );
                    }
                    Self::populate_ownership_expr(graph, arg);
                }
            }
            TypedExpressionKind::Block { statements, .. } => {
                Self::populate_ownership_stmts(graph, statements);
            }
            TypedExpressionKind::Conditional {
                condition,
                then_expr,
                else_expr,
                ..
            } => {
                Self::populate_ownership_expr(graph, condition);
                Self::populate_ownership_expr(graph, then_expr);
                if let Some(e) = else_expr {
                    Self::populate_ownership_expr(graph, e);
                }
            }
            TypedExpressionKind::BinaryOp { left, right, .. } => {
                Self::populate_ownership_expr(graph, left);
                Self::populate_ownership_expr(graph, right);
            }
            TypedExpressionKind::UnaryOp { operand, .. } => {
                Self::populate_ownership_expr(graph, operand);
            }
            TypedExpressionKind::ArrayAccess { array, index, .. } => {
                Self::populate_ownership_expr(graph, array);
                Self::populate_ownership_expr(graph, index);
            }
            _ => {}
        }
    }


    /// Get variable name from SymbolId via symbol table.
    pub(crate) fn get_symbol_name(&self, symbol: crate::tast::SymbolId, _typed_file: &TypedFile) -> String {
        if let Some(sym) = self.symbol_table.get_symbol(symbol) {
            if let Some(name) = self.string_interner.get(sym.name) {
                if !name.is_empty() {
                    return name.to_string();
                }
            }
        }
        format!("var_{}", symbol.as_raw())
    }
}
