//! Call expressions, and the method resolution behind them.

use super::*;
use crate::tast::node::HasSourceLocation;
use crate::tast::{core::*, node::MemoryEffects, node::*, type_resolution, *};
use parser::{
    AbstractDecl, BinaryOp, BlockElement, ClassDecl, ClassField, ClassFieldKind, EnumConstructor,
    EnumDecl, Expr, ExprKind, Function, FunctionParam, HaxeFile, Import, InterfaceDecl, Metadata,
    Modifier, ModuleField, Package, Type, TypeDeclaration, TypeParam, TypedefDecl, UnaryOp, Using,
};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fmt;
use std::rc::Rc;
use tracing::warn;

impl<'a> AstLowering<'a> {
    /// Whether an expression spells `_` anywhere, which only a pattern does.
    fn expr_has_wildcard(expr: &Expr) -> bool {
        match &expr.kind {
            ExprKind::Ident(name) => name == "_",
            ExprKind::Call { expr: callee, args } => {
                Self::expr_has_wildcard(callee) || args.iter().any(Self::expr_has_wildcard)
            }
            ExprKind::Array(elements) => elements.iter().any(Self::expr_has_wildcard),
            _ => false,
        }
    }

    /// The pattern an expression spells, for `e.match(pattern)`.
    ///
    /// The argument was parsed in expression position, so it arrives as a call
    /// or an identifier. Mirrors the pattern parser: a bare dotted name is
    /// `Var` whatever its case (a later stage decides constructor or binding),
    /// a call is a constructor, and `_` is the wildcard.
    fn pattern_from_expr(expr: &Expr) -> Option<parser::haxe_ast::Pattern> {
        use parser::haxe_ast::{Pattern, TypePath};
        fn dotted_path(expr: &Expr) -> Option<Vec<String>> {
            match &expr.kind {
                ExprKind::Ident(name) => Some(vec![name.clone()]),
                ExprKind::Field { expr, field, .. } => {
                    let mut parts = dotted_path(expr)?;
                    parts.push(field.clone());
                    Some(parts)
                }
                _ => None,
            }
        }
        match &expr.kind {
            ExprKind::Ident(name) if name == "_" => Some(Pattern::Underscore),
            ExprKind::Null => Some(Pattern::Null),
            ExprKind::Int(_)
            | ExprKind::Float(_)
            | ExprKind::String(_)
            | ExprKind::Bool(_) => Some(Pattern::Const(expr.clone())),
            ExprKind::Ident(_) | ExprKind::Field { .. } => {
                Some(Pattern::Var(dotted_path(expr)?.join(".")))
            }
            ExprKind::Array(elements) => Some(Pattern::Array(
                elements
                    .iter()
                    .map(Self::pattern_from_expr)
                    .collect::<Option<Vec<_>>>()?,
            )),
            ExprKind::Call { expr: callee, args } => {
                let mut parts = dotted_path(callee)?;
                let name = parts.pop()?;
                let params = args
                    .iter()
                    .map(Self::pattern_from_expr)
                    .collect::<Option<Vec<_>>>()?;
                Some(Pattern::Constructor {
                    path: TypePath {
                        package: parts,
                        name,
                        sub: None,
                    },
                    params,
                })
            }
            _ => None,
        }
    }

    /// The class `class_symbol` extends, even when this context never lowered it.
    ///
    /// `class_parents` holds only what THIS lowering registered, so the chain
    /// ends at the first class from another module -- one hop resolves and the
    /// grandparent does not. The signature index records `extends` by name for
    /// every file it has seen, which is what crosses that boundary.
    fn parent_class_symbol(&self, class_symbol: SymbolId) -> Option<SymbolId> {
        if let Some(&parent) = self.class_parents.get(&class_symbol) {
            return Some(parent);
        }
        let child = self
            .context
            .symbol_table
            .get_symbol(class_symbol)
            .and_then(|s| s.qualified_name.or(Some(s.name)))
            .and_then(|n| self.context.string_interner.get(n))?
            .to_string();
        let index = self.static_sig_index.as_ref()?.clone();
        let parent_name = index.borrow_mut().parent_of(&child)?;
        let short = parent_name.rsplit('.').next().unwrap_or(&parent_name);
        let matches = self.context.symbol_table.find_symbols(|sym| {
            if sym.kind != crate::tast::symbols::SymbolKind::Class {
                return false;
            }
            let qn = sym
                .qualified_name
                .and_then(|q| self.context.string_interner.get(q));
            if qn == Some(parent_name.as_str()) {
                return true;
            }
            qn.is_none() && self.context.string_interner.get(sym.name) == Some(short)
        });
        matches.into_iter().next().map(|s| s.id)
    }

    /// Build the parser-level AST `expr.get()` for deref coercion.
    /// Re-running `lower_expression` on the result re-uses the normal
    /// method-call lowering path so the inner type is recovered through
    /// signature substitution rather than direct `MethodCall` synthesis.
    fn synth_get_call_expr(&self, expr: &Expr) -> Expr {
        let span = expr.span;
        let getter = Expr {
            kind: parser::ExprKind::Field {
                expr: Box::new(expr.clone()),
                field: "get".to_string(),
                is_optional: false,
            },
            span,
        };
        Expr {
            kind: parser::ExprKind::Call {
                expr: Box::new(getter),
                args: Vec::new(),
            },
            span,
        }
    }

    /// Find the `get()` method on an auto-deref wrapper class.
    /// Checks both `class_methods` (user-class compilation unit) and the
    /// class's symbol-table scope (extern classes loaded from BLADE).
    pub(crate) fn find_wrapper_get_method(&self, class_sym: SymbolId) -> Option<SymbolId> {
        let get_name_str = "get";
        if let Some(methods) = self.class_methods.get(&class_sym) {
            for (name, sym, _) in methods {
                if self.context.string_interner.get(*name) == Some(get_name_str) {
                    return Some(*sym);
                }
            }
        }
        // Fallback: extern classes loaded from BLADE register their methods
        // in the class's scope, not in `class_methods`.
        let class_sym_info = self.context.symbol_table.get_symbol(class_sym)?;
        let get_interned = self.context.string_interner.get_id(get_name_str)?;
        let method_sym = self
            .context
            .symbol_table
            .lookup_symbol(class_sym_info.scope_id, get_interned)?;
        if method_sym.kind == crate::tast::symbols::SymbolKind::Function {
            Some(method_sym.id)
        } else {
            None
        }
    }

    /// Fill in known stdlib static method types when only placeholder symbols are available.
    ///
    /// This keeps return types stable even when stdlib class bodies are not fully lowered.
    fn ensure_known_static_method_type(
        &mut self,
        class_symbol: SymbolId,
        method_name: InternedString,
        method_symbol: SymbolId,
    ) {
        let class_name = self
            .context
            .symbol_table
            .get_symbol(class_symbol)
            .and_then(|s| self.context.string_interner.get(s.name));
        let method_name_str = self.context.string_interner.get(method_name);

        // Keep Type.typeof statically typed as Dynamic -> ValueType.
        // Runtime mapping may provide an ordinal-based placeholder signature, but
        // language-level typing must remain ValueType for parity (trace/switch).
        if class_name == Some("Type") && method_name_str == Some("typeof") {
            let dynamic_type = self.context.type_table.borrow().dynamic_type();
            let value_type = self
                .resolve_type_by_name("ValueType")
                .unwrap_or(dynamic_type);
            let should_update = self
                .context
                .symbol_table
                .get_symbol(method_symbol)
                .map(|s| {
                    let current_type = s.type_id;
                    let type_table = self.context.type_table.borrow();
                    match type_table.get(current_type).map(|t| &t.kind) {
                        Some(crate::tast::core::TypeKind::Function {
                            params,
                            return_type,
                            ..
                        }) => {
                            params.len() != 1
                                || params[0] != dynamic_type
                                || *return_type != value_type
                        }
                        _ => true,
                    }
                })
                .unwrap_or(true);

            if should_update {
                let fn_type = self
                    .context
                    .type_table
                    .borrow_mut()
                    .create_function_type(vec![dynamic_type], value_type);
                self.context
                    .symbol_table
                    .update_symbol_type(method_symbol, fn_type);
            }
            return;
        }

        let has_type = self
            .context
            .symbol_table
            .get_symbol(method_symbol)
            .map(|s| s.type_id.is_valid())
            .unwrap_or(false);
        if has_type {
            return;
        }

        // General on-demand signature resolution. A static's declaring file
        // can TAST-lower AFTER a call site that references it (import /
        // convergence ordering), so the symbol resolved here is an untyped
        // pre-registration or placeholder — and an untyped factory return
        // decays the whole downstream chain into guessed dynamic dispatch.
        // Recover the DECLARED signature from the parsed AST via the
        // program-wide sig index (parsing the declaring file on demand if it
        // hasn't been seen yet), mirroring the per-class pre-registration
        // loop in `lower_class_declaration`.
        if let Some(sig) = self.resolve_declared_method_sig(class_symbol, method_name, true) {
            self.apply_declared_sig(method_symbol, &sig);
            self.context
                .symbol_table
                .add_symbol_flags(method_symbol, crate::tast::symbols::SymbolFlags::STATIC);
            return;
        }
    }

    /// Lower a declared signature's AST type hints into a function type
    /// (unannotated positions become Dynamic, mirroring the per-class
    /// pre-registration loop) and stamp it on `method_symbol`.
    fn apply_declared_sig(
        &mut self,
        method_symbol: SymbolId,
        sig: &crate::tast::sig_index::StaticMethodSig,
    ) -> TypeId {
        let dynamic_type = self.context.type_table.borrow().dynamic_type();
        let param_types: Vec<TypeId> = sig
            .params
            .iter()
            .map(|hint| match hint {
                Some(t) => self.lower_type(t).unwrap_or(dynamic_type),
                None => dynamic_type,
            })
            .collect();
        let return_type = match &sig.return_type {
            Some(t) => self.lower_type(t).unwrap_or(dynamic_type),
            None => dynamic_type,
        };
        let fn_ty = self
            .context
            .type_table
            .borrow_mut()
            .create_function_type(param_types, return_type);
        self.context
            .symbol_table
            .update_symbol_type(method_symbol, fn_ty);
        fn_ty
    }

    /// Resolve a method symbol for a given receiver and method name
    fn resolve_method_symbol(
        &mut self,
        receiver: &TypedExpression,
        method_name: InternedString,
    ) -> SymbolId {
        // Try to resolve method from receiver's type
        match &receiver.kind {
            TypedExpressionKind::This { this_type } => {
                let _ = this_type;
                if let Some(class_symbol) = self.context.class_context_stack.last() {
                    if let Some(methods) = self.class_methods.get(class_symbol) {
                        if let Some((_, method_symbol, _)) =
                            methods.iter().find(|(name, _, _)| *name == method_name)
                        {
                            return *method_symbol;
                        }
                    }
                }
            }
            TypedExpressionKind::Variable { symbol_id } => {
                // Try to resolve method from variable's type
                if let Some(symbol) = self.context.symbol_table.get_symbol(*symbol_id) {
                    if let Some(class_symbol) = self.resolve_type_to_class_symbol(symbol.type_id) {
                        if let Some(found) =
                            self.resolve_class_method_symbol(class_symbol, method_name)
                        {
                            return found;
                        }
                        // An inherited method is declared by an ancestor, and
                        // every strategy above is keyed on the receiver's own
                        // class. Missing it hands back a placeholder whose type
                        // is unknown, so the call's return type is unresolved --
                        // an Int comes back as a pointer and printing it ends
                        // the process with no output.
                        let mut current = class_symbol;
                        let mut seen: std::collections::BTreeSet<SymbolId> =
                            std::collections::BTreeSet::new();
                        seen.insert(current);
                        while let Some(parent) = self.parent_class_symbol(current) {
                            if !seen.insert(parent) {
                                break; // cyclic `extends`; the type checker reports it
                            }
                            if let Some(found) =
                                self.resolve_class_method_symbol(parent, method_name)
                            {
                                return found;
                            }
                            current = parent;
                        }
                    }
                }
            }
            TypedExpressionKind::MethodCall { .. } | TypedExpressionKind::FunctionCall { .. } => {
                // For method chains like z.mul(z).add(c), the receiver is a MethodCall.
                // We need to infer the type of that expression and resolve the method on it.
                if let Ok(receiver_type) = self.infer_expression_type(&receiver.kind) {
                    if let Some(class_symbol) = self.resolve_type_to_class_symbol(receiver_type) {
                        // Check local class_methods first (classes in this compilation unit)
                        if let Some(methods) = self.class_methods.get(&class_symbol) {
                            if let Some((_, method_symbol, _)) =
                                methods.iter().find(|(name, _, _)| *name == method_name)
                            {
                                return *method_symbol;
                            }
                        }
                        // Fallback: check shared symbol table's class scope
                        // (for classes compiled in a different compilation unit / package)
                        if let Some(class_sym) = self.context.symbol_table.get_symbol(class_symbol)
                        {
                            if let Some(method_sym) = self
                                .context
                                .symbol_table
                                .lookup_symbol(class_sym.scope_id, method_name)
                            {
                                if method_sym.kind == crate::tast::symbols::SymbolKind::Function {
                                    return method_sym.id;
                                }
                            }
                        }
                    }
                }
            }
            TypedExpressionKind::New { class_type, .. } => {
                // For `new Complex().method()`, resolve method on the class type
                if let Some(class_symbol) = self.resolve_type_to_class_symbol(*class_type) {
                    if let Some(methods) = self.class_methods.get(&class_symbol) {
                        if let Some((_, method_symbol, _)) =
                            methods.iter().find(|(name, _, _)| *name == method_name)
                        {
                            return *method_symbol;
                        }
                    }
                    // Fallback: shared symbol table (cross-package classes)
                    if let Some(class_sym) = self.context.symbol_table.get_symbol(class_symbol) {
                        if let Some(method_sym) = self
                            .context
                            .symbol_table
                            .lookup_symbol(class_sym.scope_id, method_name)
                        {
                            if method_sym.kind == crate::tast::symbols::SymbolKind::Function {
                                return method_sym.id;
                            }
                        }
                    }
                }
            }
            _ => {
                // For other receiver types, try general type inference
                if let Ok(receiver_type) = self.infer_expression_type(&receiver.kind) {
                    if let Some(class_symbol) = self.resolve_type_to_class_symbol(receiver_type) {
                        if let Some(methods) = self.class_methods.get(&class_symbol) {
                            if let Some((_, method_symbol, _)) =
                                methods.iter().find(|(name, _, _)| *name == method_name)
                            {
                                return *method_symbol;
                            }
                        }
                        // Fallback: shared symbol table (cross-package classes)
                        if let Some(class_sym) = self.context.symbol_table.get_symbol(class_symbol)
                        {
                            if let Some(method_sym) = self
                                .context
                                .symbol_table
                                .lookup_symbol(class_sym.scope_id, method_name)
                            {
                                if method_sym.kind == crate::tast::symbols::SymbolKind::Function {
                                    return method_sym.id;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Fallback: Try to resolve using the receiver expression's type
        // (may differ from the variable symbol's type for extern classes).
        // Route through `resolve_class_method_symbol` so the phantom-class
        // fallback (Strategy 4) fires for typedef'd receivers — e.g.
        // `this.bytes.sub(...)` where `bytes:haxe.io.Bytes` resolves to a
        // BLADE phantom Class with no methods and the real `rayzor.Bytes`
        // class holds the method elsewhere.
        if let Some(class_symbol) = self.resolve_type_to_class_symbol(receiver.expr_type) {
            if let Some(found) = self.resolve_class_method_symbol(class_symbol, method_name) {
                return found;
            }
        }

        // Last resort: scan ALL class_methods for a match by name.
        // This handles extern classes where TypeId is invalid but the method
        // definition symbols have full metadata (qualified_name, native_name, etc.).
        //
        // CRITICAL: when the receiver's type is Dynamic (or otherwise
        // unresolved), exclude the class we're currently lowering from
        // candidates. Class_methods is populated incrementally during
        // lowering — if only the current class's methods are visible at
        // this point, the scan would always return the enclosing function
        // (the "lexical match" trap), producing a silent self-recursive
        // call at runtime. Excluding the enclosing class forces the
        // lookup to either find a real cross-class candidate or fall
        // through to the placeholder path (which gets a qualified name
        // and goes through proper Dynamic dispatch downstream).
        let receiver_is_dynamic = {
            let tt = self.context.type_table.borrow();
            tt.get(receiver.expr_type)
                .map(|t| matches!(t.kind, crate::tast::core::TypeKind::Dynamic))
                .unwrap_or(false)
        };
        let current_class = if receiver_is_dynamic {
            self.context.class_context_stack.last().copied()
        } else {
            None
        };
        {
            // A TYPED receiver extends the lexical-match-trap guard: a lone
            // same-named method on an UNRELATED class must not win either
            // (`gate:Linear` + `.forward` in SwiGLU.hx found only
            // SwiGLU.forward here — Linear lowers later — and the call
            // self-bound into infinite recursion). Only candidates whose
            // class matches the receiver's may enter; none matching falls
            // through to the qualified placeholder below.
            let receiver_bare_for_filter = self
                .get_class_name_for_type(receiver.expr_type)
                .map(|c| c.rsplit('.').next().unwrap_or(&c).to_string());
            let mut found: Option<(SymbolId, SymbolId)> = None; // (class_sym, method_sym)
            let mut ambiguous = false;
            let mut all_matches: Vec<(SymbolId, SymbolId)> = Vec::new();
            for (class_sym, methods) in &self.class_methods {
                if Some(*class_sym) == current_class {
                    continue;
                }
                // The filter only bit when the receiver's class could be
                // named. A builtin cannot -- Array<Int> yields nothing -- so
                // every candidate was admitted and a lone same-named method
                // won by default: declaring `abstract Wrap(Array<Int>)` with
                // a `push` captured EVERY Array.push in the program, and the
                // forwarding body then called itself forever. An unnameable
                // receiver is the case where a name match is least justified,
                // not most. Require the name, and let the qualified
                // placeholder below handle what this cannot resolve.
                let Some(rb) = receiver_bare_for_filter.as_deref() else {
                    continue;
                };
                let owner_bare = self
                    .context
                    .symbol_table
                    .get_symbol(*class_sym)
                    .and_then(|s| self.context.string_interner.get(s.name));
                if owner_bare.is_some_and(|o| o != rb) {
                    continue;
                }
                if let Some((_, method_symbol, _)) =
                    methods.iter().find(|(name, _, _)| *name == method_name)
                {
                    all_matches.push((*class_sym, *method_symbol));
                    if found.is_some() {
                        ambiguous = true;
                    }
                    found = Some((*class_sym, *method_symbol));
                }
            }
            if let Some((_, method_symbol)) = found {
                if !ambiguous {
                    return method_symbol;
                }
                // Ambiguous: try to disambiguate using receiver's class name
                // Get receiver class name from the expression type (may be qualified)
                let receiver_class_name = self.get_class_name_for_type(receiver.expr_type);
                if let Some(ref class_name) = receiver_class_name {
                    // Extract bare name from qualified (e.g., "sys.io.FileOutput" -> "FileOutput")
                    let bare_name = class_name.rsplit('.').next().unwrap_or(class_name);
                    for (class_sym, method_sym) in &all_matches {
                        if let Some(sym) = self.context.symbol_table.get_symbol(*class_sym) {
                            // Match against bare name or qualified name
                            let sym_name = self.context.string_interner.get(sym.name).unwrap_or("");
                            let sym_qname = sym
                                .qualified_name
                                .and_then(|qn| self.context.string_interner.get(qn))
                                .unwrap_or("");
                            if sym_name == bare_name
                                || sym_name == class_name.as_str()
                                || sym_qname == class_name.as_str()
                            {
                                return *method_sym;
                            }
                        }
                    }
                }
            }
        }

        // Create a method symbol placeholder if we can't resolve it
        // Set qualified name based on receiver's class to help MIR disambiguation
        if std::env::var_os("RAYZOR_SYM_DEBUG").is_some() {
            let m = self.context.string_interner.get(method_name).unwrap_or("?");
            let k = self.context.type_table.borrow().get(receiver.expr_type).map(|t| format!("{:?}", t.kind));
            eprintln!("[sym] member-synth {m} recv={} class={:?}", k.unwrap_or_default().chars().take(80).collect::<String>(), self.get_class_name_for_type(receiver.expr_type));
            for cand in self.context.symbol_table.find_symbols(|s| s.name == method_name).into_iter().take(8) {
                let qn = cand.qualified_name.and_then(|q| self.context.string_interner.get(q)).unwrap_or("-");
                eprintln!("[sym]   cand {:?} kind={:?} qn={qn} type_valid={} scope={:?}", cand.id, cand.kind, cand.type_id.is_valid(), cand.scope_id);
            }
        }
        let new_symbol = self.context.symbol_table.create_function(method_name);
        if let Some(class_name) = self.get_class_name_for_type(receiver.expr_type) {
            let method_name_str = self.context.string_interner.get(method_name).unwrap_or("");
            let qname = format!("{}.{}", class_name, method_name_str);
            let qname_interned = self.context.intern_string(&qname);
            if let Some(sym) = self.context.symbol_table.get_symbol_mut(new_symbol) {
                sym.qualified_name = Some(qname_interned);
            }
        }
        new_symbol
    }

    /// Resolve a method symbol within a specific class context, preferring
    /// qualified-name matching to avoid cross-class collisions on short names.
    pub(crate) fn resolve_class_method_symbol(
        &self,
        class_symbol: SymbolId,
        method_name: InternedString,
    ) -> Option<SymbolId> {
        // Strategy 1: local class_methods table (same lowering instance)
        if let Some(methods) = self.class_methods.get(&class_symbol) {
            if let Some((_, symbol, _)) = methods.iter().find(|(name, _, _)| *name == method_name) {
                return Some(*symbol);
            }
        }

        let class_sym = self.context.symbol_table.get_symbol(class_symbol)?;

        // Strategy 2: exact qualified-name match Class.method
        // A root-package type has no qualified name; its bare name is the path.
        if let (Some(class_qname), Some(method_name_str)) = (
            class_sym
                .qualified_name
                .or(Some(class_sym.name))
                .and_then(|qn| self.context.string_interner.get(qn)),
            self.context.string_interner.get(method_name),
        ) {
            let expected_qname = format!("{}.{}", class_qname, method_name_str);
            if let Some(sym) = self
                .context
                .symbol_table
                .find_symbols(|sym| {
                    sym.kind == crate::tast::symbols::SymbolKind::Function
                        && sym.name == method_name
                        && sym
                            .qualified_name
                            .and_then(|qn| self.context.string_interner.get(qn))
                            .map(|qn| qn == expected_qname)
                            .unwrap_or(false)
                })
                .into_iter()
                .next()
            {
                return Some(sym.id);
            }

            // Strategy 2b: package-qualified suffix match. A `@:coreType extern
            // abstract` loaded from the stdlib (Atomic/Box/Ptr) carries a BARE
            // class qualified_name (`Atomic`) while its methods are registered
            // with the fully-packaged qname (`rayzor.Atomic.of`). The exact
            // match above misses, leaving the call bound to a typeless
            // placeholder whose return decays to Dynamic. Match a typed Function
            // whose qname ends with `.<class_bare_name>.<method>` so the real
            // pre-typed method is recovered. Restricted to symbols with a valid
            // type so it never prefers a fresh placeholder.
            let class_bare = class_qname.rsplit('.').next().unwrap_or(class_qname);
            let suffix = format!(".{}.{}", class_bare, method_name_str);
            if let Some(sym) = self
                .context
                .symbol_table
                .find_symbols(|sym| {
                    sym.kind == crate::tast::symbols::SymbolKind::Function
                        && sym.name == method_name
                        && sym.type_id.is_valid()
                        && sym
                            .qualified_name
                            .and_then(|qn| self.context.string_interner.get(qn))
                            .map(|qn| qn.ends_with(&suffix))
                            .unwrap_or(false)
                })
                .into_iter()
                .next()
            {
                return Some(sym.id);
            }
        }

        // Strategy 3: class scope fallback. A pre-registered class's
        // `scope_id` can be the ROOT scope (no class scope exists until its
        // file lowers), where a same-named static of an UNRELATED class also
        // lives — verify ownership by qualified name before accepting
        // (`Linear.fromQuant` must not resolve to `nue.Embedding.fromQuant`).
        if let Some(sym) = self
            .context
            .symbol_table
            .lookup_symbol(class_sym.scope_id, method_name)
        {
            let owned_elsewhere = (|| {
                let sym_qn = self.context.string_interner.get(sym.qualified_name?)?;
                let class_qn = self
                    .context
                    .string_interner
                    .get(class_sym.qualified_name?)?;
                let (owner, _) = sym_qn.rsplit_once('.')?;
                let class_bare = class_qn.rsplit('.').next().unwrap_or(class_qn);
                let owner_bare = owner.rsplit('.').next().unwrap_or(owner);
                Some(owner != class_qn && owner_bare != class_bare)
            })()
            .unwrap_or(false);
            if !owned_elsewhere {
                return Some(sym.id);
            }
        }

        // Strategy 4: phantom-class fallback. BLADE can produce a Class symbol
        // for a typedef target whose qualified name embeds the typedef's
        // package (e.g., a `Bytes` Class with qname `haxe.io.Bytes` and an
        // empty scope) while the real underlying class lives elsewhere as
        // another `Bytes` Class with methods registered in its own scope.
        // If the resolved class symbol has no methods of its own, walk the
        // symbol table for ALL same-named classes and pick one whose scope
        // does have this method.
        let short_class_name = class_sym.name;
        let mut candidates = self.context.symbol_table.find_symbols(|s| {
            s.kind == crate::tast::symbols::SymbolKind::Class
                && s.name == short_class_name
                && s.id != class_symbol
        });
        candidates.sort_by_key(|s| s.id);
        for cand in candidates {
            if let Some(sym) = self
                .context
                .symbol_table
                .lookup_symbol(cand.scope_id, method_name)
            {
                if sym.kind == crate::tast::symbols::SymbolKind::Function {
                    return Some(sym.id);
                }
            }
        }

        None
    }

    /// Try to find a static extension method in using modules
    /// Returns (class_symbol, method_symbol) if found
    fn find_static_extension_method(
        &self,
        method_name: InternedString,
        _receiver_type: TypeId,
    ) -> Option<(SymbolId, SymbolId)> {
        // Check each using module for a static method with this name
        for (_class_name, class_symbol) in &self.using_modules {
            // First, check local class_methods (for classes lowered in this instance)
            if let Some(methods) = self.class_methods.get(class_symbol) {
                for (meth_name, meth_symbol, is_static) in methods {
                    if *meth_name == method_name && *is_static {
                        return Some((*class_symbol, *meth_symbol));
                    }
                }
            }

            // Then, check the shared symbol table for methods registered by other lowering passes
            // Look up the class symbol to get its scope, then search for the method
            if let Some(class_sym) = self.context.symbol_table.get_symbol(*class_symbol) {
                // The class should have a scope ID where its members are registered
                // Try to find a method with the given name in that scope
                if let Some(method_sym) = self
                    .context
                    .symbol_table
                    .lookup_symbol(class_sym.scope_id, method_name)
                {
                    // Check if it's a static method by looking at its modifiers or kind
                    if method_sym.kind == crate::tast::symbols::SymbolKind::Function {
                        return Some((*class_symbol, method_sym.id));
                    }
                }
            }
        }
        None
    }

    fn boxing_param_types(&mut self, callee: &Expr) -> Option<Vec<TypeId>> {
        match &callee.kind {
            ExprKind::Ident(name) => {
                let name_interned = self.context.string_interner.intern(name);
                if let Some(sym) = self
                    .context
                    .symbol_table
                    .lookup_symbol(self.context.current_scope, name_interned)
                {
                    if let Some(params) = self.function_param_types_from_symbol(sym.id) {
                        return Some(params);
                    }
                }
                let class_sym = *self.context.class_context_stack.last()?;
                let method_sym = self.resolve_class_method_symbol(class_sym, name_interned)?;
                self.function_param_types_from_symbol(method_sym)
            }
            ExprKind::Field {
                expr: obj, field, ..
            } => {
                if let ExprKind::Ident(cls_name) = &obj.kind {
                    let cls_name_interned = self.context.string_interner.intern(cls_name);
                    let class_sym = self
                        .context
                        .symbol_table
                        .lookup_symbol(self.context.current_scope, cls_name_interned)
                        .filter(|s| s.kind == crate::tast::symbols::SymbolKind::Class)
                        .map(|s| s.id)
                        .or_else(|| self.resolve_class_like_symbol_by_name(cls_name_interned));
                    if let Some(cls_id) = class_sym {
                        let method_name = self.context.string_interner.intern(field);
                        if let Some(method_sym) =
                            self.resolve_class_method_symbol(cls_id, method_name)
                        {
                            return self.function_param_types_from_symbol(method_sym);
                        }
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// If `formal` is a genuine `Dynamic` parameter and `arg` is a primitive
    /// VALUE type, wrap `arg` in an implicit cast to Dynamic so it is boxed
    /// during lowering (a raw inline primitive handed to a `Dynamic` param is
    /// read back as a bogus `DynamicValue` and crashes `Std.string`).
    ///
    /// Restricted to primitive value types (Int/Float/Bool) — these are the only
    /// args that genuinely need boxing: stored inline, a raw primitive handed to
    /// a `Dynamic` param is misread as a pointer. Reference types are left raw on
    /// purpose: class/array/enum values are already pointers and are often FFI
    /// handles (SIMD, quant, generics) that must reach the callee unwrapped, and
    /// a raw String is consumed directly by `trace` and similar `Dynamic` sinks.
    /// (Correct String→Dynamic for `Std.string` needs a tagged representation
    /// that those sinks also accept; out of scope here.)
    /// Wrap an argument in the abstract's `@:from` conversion when the formal
    /// parameter is that abstract and the argument is not already one.
    /// `lazy(2)` with `lazy(l:Lazy<A>)` and `@:from static ofConst<T>(c:T)`
    /// must lower as `lazy(ofConst(2))` — without the call, the raw argument
    /// flows into the abstract's representation and whatever consumes it
    /// (`this()` on an abstract over a function type jumps through the
    /// integer 2). The clause form (`from Y`) is representation-compatible
    /// and correctly stays uncoerced.
    fn coerce_arg_via_abstract_from(
        &mut self,
        arg: TypedExpression,
        formal: Option<TypeId>,
    ) -> TypedExpression {
        use crate::tast::core::TypeKind;
        let Some(formal_ty) = formal else {
            return arg;
        };
        // The formal's abstract symbol, through generic instantiation.
        let abstract_symbol = {
            let tt = self.context.type_table.borrow();
            let mut resolved = formal_ty;
            let mut hops = 0;
            loop {
                hops += 1;
                if hops > 8 {
                    break None;
                }
                match tt.get(resolved).map(|ti| &ti.kind) {
                    Some(TypeKind::GenericInstance { base_type, .. }) => resolved = *base_type,
                    Some(TypeKind::Abstract { symbol_id, .. }) => break Some(*symbol_id),
                    _ => break None,
                }
            }
        };
        let Some(abstract_symbol) = abstract_symbol else {
            return arg;
        };
        // Already that abstract? No conversion.
        let arg_is_same_abstract = {
            let tt = self.context.type_table.borrow();
            let mut resolved = arg.expr_type;
            let mut hops = 0;
            loop {
                hops += 1;
                if hops > 8 {
                    break false;
                }
                match tt.get(resolved).map(|ti| &ti.kind) {
                    Some(TypeKind::GenericInstance { base_type, .. }) => resolved = *base_type,
                    Some(TypeKind::Abstract { symbol_id, .. }) => {
                        break *symbol_id == abstract_symbol
                    }
                    _ => break false,
                }
            }
        };
        if arg_is_same_abstract {
            return arg;
        }
        let Some(candidates) = self.abstract_from_methods.get(&abstract_symbol) else {
            return arg;
        };
        // First method whose parameter accepts the argument's type: an exact
        // type, a matching primitive kind, or the abstract's own type
        // parameter (which accepts anything).
        let pick = {
            let tt = self.context.type_table.borrow();
            let arg_kind = tt.get(arg.expr_type).map(|ti| ti.kind.clone());
            candidates
                .iter()
                .find(|(_, param_ty)| {
                    if *param_ty == arg.expr_type {
                        return true;
                    }
                    match (tt.get(*param_ty).map(|ti| &ti.kind), arg_kind.as_ref()) {
                        (Some(TypeKind::TypeParameter { .. }), _) => true,
                        (Some(TypeKind::Dynamic), _) => true,
                        (Some(a), Some(b)) => {
                            std::mem::discriminant(a) == std::mem::discriminant(b)
                                && matches!(
                                    a,
                                    TypeKind::Int
                                        | TypeKind::Float
                                        | TypeKind::Bool
                                        | TypeKind::String
                                )
                        }
                        _ => false,
                    }
                })
                .map(|&(method_symbol, _)| method_symbol)
        };
        let Some(method_symbol) = pick else {
            return arg;
        };
        let location = arg.source_location.clone();
        TypedExpression {
            expr_type: formal_ty,
            kind: crate::tast::node::TypedExpressionKind::StaticMethodCall {
                class_symbol: abstract_symbol,
                method_symbol,
                arguments: vec![arg],
                type_arguments: Vec::new(),
            },
            usage: crate::tast::node::VariableUsage::Copy,
            lifetime_id: crate::tast::LifetimeId::first(),
            source_location: location,
            metadata: crate::tast::node::ExpressionMetadata::default(),
        }
    }

    fn coerce_arg_to_dynamic_param(
        &mut self,
        arg: TypedExpression,
        formal: Option<TypeId>,
    ) -> TypedExpression {
        use crate::tast::core::TypeKind;
        let Some(formal_ty) = formal else {
            return arg;
        };
        let should_box = {
            let tt = self.context.type_table.borrow();
            let formal_is_dyn =
                matches!(tt.get(formal_ty).map(|t| &t.kind), Some(TypeKind::Dynamic));
            formal_is_dyn
                && matches!(
                    tt.get(arg.expr_type).map(|t| &t.kind),
                    Some(TypeKind::Int) | Some(TypeKind::Float) | Some(TypeKind::Bool)
                )
        };
        if !should_box {
            return arg;
        }
        // Implicit (is_safe) cast so hir_to_mir takes the boxing arm — an Unsafe
        // cast to Dynamic is treated as a no-op reinterpret and would NOT box.
        let loc = arg.source_location;
        let kind = TypedExpressionKind::Cast {
            expression: Box::new(arg),
            target_type: formal_ty,
            cast_kind: CastKind::Implicit,
        };
        let usage = self.determine_variable_usage(&kind);
        let lifetime_id = self.assign_lifetime(&kind, &formal_ty);
        let metadata = self.analyze_expression_metadata(&kind);
        TypedExpression {
            expr_type: formal_ty,
            kind,
            usage,
            lifetime_id,
            source_location: loc,
            metadata,
        }
    }

    pub(crate) fn lower_call_expression(
        &mut self,
        expression: &Expr,
        expr: &Expr,
        args: &[Expr],
    ) -> LoweringResult<TypedExpression> {
        // A call the expander deferred (its macro body asks the typer a
        // question) is re-expanded HERE, where locals upstream of this site
        // are typed and this lowering can answer as the MacroTyper. The
        // result replaces the call and lowers in its place.
        if !self.deferred_macro_calls.is_empty() {
            let key = (expression.span.start, expression.span.end);
            if let Some(name) = self.deferred_macro_calls.get(&key).cloned() {
                let cell = self
                    .deferred_macro_expander
                    .expect("deferred call recorded without its expander");
                let expanded = {
                    let mut typer =
                        super::super::macro_defer::DeferredMacroTyper { lowering: self };
                    cell.borrow_mut()
                        .expand_deferred_call(&name, expression, &mut typer)
                };
                match expanded {
                    Ok(result) => return self.lower_expression(&result),
                    Err(e) => {
                        return Err(LoweringError::SemanticError {
                            message: format!("macro '{}' failed during typing: {}", name, e),
                            location: self.context.create_location_from_span(expression.span),
                        });
                    }
                }
            }
        }

        // `e.match(pattern)` is sugar for `switch (e) { case pattern: true;
        // default: false; }`. Lowered as an ordinary call the pattern is read
        // as an expression, so `_` resolves as a variable and the call fails
        // with "Cannot find name '_'". `EReg.match(s:String)` is a real method,
        // so only a receiver with no `match` member of its own is desugared.
        if let ExprKind::Field {
            expr: receiver_expr,
            field,
            ..
        } = &expr.kind
        {
            if field == "match" && args.len() == 1 {
                let receiver = self.lower_expression(receiver_expr)?;
                // An enum receiver, or an argument spelling a wildcard, is a
                // pattern. Anything else keeps the method it names -- asking
                // whether the receiver's class declares `match` is not enough,
                // because a stdlib class like `EReg` resolves to nothing in
                // this context and its real `match(s:String)` would be eaten.
                let receiver_is_enum = {
                    let tt = self.context.type_table.borrow();
                    tt.get(receiver.expr_type)
                        .map(|t| matches!(t.kind, crate::tast::core::TypeKind::Enum { .. }))
                        .unwrap_or(false)
                };
                if receiver_is_enum || Self::expr_has_wildcard(&args[0]) {
                    if let Some(pattern) = Self::pattern_from_expr(&args[0]) {
                        let span = expression.span;
                        let switch_expr = Expr {
                            kind: ExprKind::Switch {
                                expr: Box::new((**receiver_expr).clone()),
                                cases: vec![parser::haxe_ast::Case {
                                    patterns: vec![pattern],
                                    guard: None,
                                    body: Expr {
                                        kind: ExprKind::Bool(true),
                                        span,
                                    },
                                    span,
                                }],
                                default: Some(Box::new(Expr {
                                    kind: ExprKind::Bool(false),
                                    span,
                                })),
                            },
                            span,
                        };
                        return self.lower_expression(&switch_expr);
                    }
                }
            }
        }

        // Intercept f.bind(args...) before lowering args (args may contain `_` placeholder)
        if let ExprKind::Field {
            expr: receiver_expr,
            field,
            ..
        } = &expr.kind
        {
            if field == "bind" {
                let receiver = self.lower_expression(receiver_expr)?;
                let is_func_type = {
                    let tt = self.context.type_table.borrow();
                    tt.get(receiver.expr_type)
                        .map(|t| matches!(t.kind, crate::tast::core::TypeKind::Function { .. }))
                        .unwrap_or(false)
                };
                if is_func_type {
                    return self.lower_bind_expression(expression, receiver, args);
                }
                // Not function-typed — fall through to normal method call handling
            }
        }

        // Try to peek the callee's formal parameter types so we can supply
        // expected lambda signatures to untyped function literals like
        // `function(i, n) { ... }` (Phase 1c WorkerPool callback case). This
        // is best-effort: if we can't statically resolve a callee, we just
        // lower args with no hints and the existing behavior applies.
        //
        // Re-entrancy guard: `resolve_callee_*` lowers the receiver, which
        // for a call-valued receiver re-enters here. Suppress nested hint
        // resolution during that lowering so a chain of call receivers stays
        // O(depth) instead of O(2^depth) — otherwise `WorkerPool.parallelRows`
        // and similar bodies hang the compiler. See `suppress_callee_hint`.
        let (expected_param_types, expected_arg_types) = if self.suppress_callee_hint {
            (None, None)
        } else {
            self.suppress_callee_hint = true;
            let p = self.resolve_callee_param_types(expr);
            let a = self.resolve_callee_formal_param_types(expr);
            self.suppress_callee_hint = false;
            (p, a)
        };

        let arg_exprs = args
            .iter()
            .enumerate()
            .map(|(i, arg)| {
                let hint = expected_param_types
                    .as_ref()
                    .and_then(|ps| ps.get(i).cloned())
                    .flatten();
                let arg_type_hint = expected_arg_types
                    .as_ref()
                    .and_then(|ts| ts.get(i).copied());
                self.expected_lambda_params_stack.push(hint);
                self.expected_arg_type_stack.push(arg_type_hint);
                let result = self
                    .lower_value_expression(arg)
                    .map(|typed| self.instantiate_function_reference(typed, arg_type_hint));
                self.expected_arg_type_stack.pop();
                self.expected_lambda_params_stack.pop();
                result
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Box primitive value-type arguments passed to `Dynamic` parameters.
        // Formal param types come from the import-aware `boxing_param_types`
        // (covers cross-module STATIC calls that the lambda-hint resolver
        // misses), falling back to the already-computed `expected_arg_types`
        // (covers instance method calls and same-module callees). The latter is
        // the UNMODIFIED lambda-hint resolver — reused read-only here so the
        // boxing decision cannot affect closure inference.
        let arg_exprs = {
            let boxing_formals = if self.suppress_callee_hint {
                None
            } else {
                self.boxing_param_types(expr)
            };
            if boxing_formals.is_none() && expected_arg_types.is_none() {
                arg_exprs
            } else {
                let mut coerced: Vec<TypedExpression> = Vec::with_capacity(arg_exprs.len());
                for (i, a) in arg_exprs.into_iter().enumerate() {
                    let formal = boxing_formals
                        .as_ref()
                        .and_then(|f| f.get(i).copied())
                        .or_else(|| expected_arg_types.as_ref().and_then(|f| f.get(i).copied()));
                    let a = self.coerce_arg_via_abstract_from(a, formal);
                    coerced.push(self.coerce_arg_to_dynamic_param(a, formal));
                }
                coerced
            }
        };

        // Check if this is a method call (field access being called)
        let kind = match &expr.kind {
            ExprKind::Field {
                expr: obj_expr,
                field,
                is_optional: field_is_optional,
            } => {
                let is_optional_call = *field_is_optional;
                // Check if this is a static method call (Class.method)
                if let ExprKind::Ident(class_name) = &obj_expr.kind {
                    let class_name_interned = self.context.intern_string(class_name);

                    // Try to resolve as a class symbol
                    if let Some(symbol_id) =
                        self.resolve_class_like_symbol_by_name(class_name_interned)
                    {
                        if let Some(symbol) = self.context.symbol_table.get_symbol(symbol_id) {
                            // Check if this symbol represents a class declaration (not just a variable of class type)
                            // `@:coreType extern abstract` types (Atomic, Box, Ptr) carry
                            // SymbolKind::Abstract; their static methods (Atomic.of, Box.init)
                            // must take the same static-call path as classes so the call keeps
                            // its declared concrete return type (Atomic<T>/Box<T>) instead of
                            // falling through to the instance path and decaying to Dynamic.
                            if symbol.kind == crate::tast::symbols::SymbolKind::Class
                                || symbol.kind == crate::tast::symbols::SymbolKind::Abstract
                            {
                                // This is a class name, so this is a static method call
                                //
                                // For extern classes (Std, Math, Sys, etc.), the type_id may be invalid
                                // because they don't have concrete type definitions in the type table.
                                // In that case, use the symbol_id directly as the class_symbol.
                                let class_symbol = if symbol.type_id == TypeId::invalid() {
                                    // Extern class - use the symbol_id directly
                                    symbol_id
                                } else if let Ok(type_table) = self.context.type_table.try_borrow()
                                {
                                    // Try to get the class symbol from the type table
                                    if let Some(type_info) = type_table.get(symbol.type_id) {
                                        if let crate::tast::core::TypeKind::Class {
                                            symbol_id: ts_symbol,
                                            ..
                                        } = &type_info.kind
                                        {
                                            *ts_symbol
                                        } else {
                                            // Type exists but isn't a Class - use symbol_id as fallback
                                            symbol_id
                                        }
                                    } else {
                                        // Type not in table - use symbol_id as fallback
                                        symbol_id
                                    }
                                } else {
                                    // Can't borrow type table - use symbol_id as fallback
                                    symbol_id
                                };

                                // This is a static method call
                                let method_name = self.context.intern_string(field);

                                // Look for the method in this class:
                                // 1. local class_methods
                                // 2. exact qualified-name match
                                // 3. class scope fallback
                                // 4. create placeholder as last resort
                                let method_symbol = {
                                    if let Some(sym) =
                                        self.resolve_class_method_symbol(class_symbol, method_name)
                                    {
                                        sym
                                    } else {
                                        // Strategy 4: create placeholder with qualified name
                                        let new_symbol =
                                            self.context.symbol_table.create_function(method_name);
                                        if let Some(class_sym) =
                                            self.context.symbol_table.get_symbol(class_symbol)
                                        {
                                            if let Some(class_qname) = class_sym
                                                .qualified_name
                                                .and_then(|qn| self.context.string_interner.get(qn))
                                            {
                                                let method_qname = format!(
                                                    "{}.{}",
                                                    class_qname,
                                                    self.context
                                                        .string_interner
                                                        .get(method_name)
                                                        .unwrap_or("")
                                                );
                                                let method_qname_interned =
                                                    self.context.intern_string(&method_qname);
                                                if let Some(sym_mut) = self
                                                    .context
                                                    .symbol_table
                                                    .get_symbol_mut(new_symbol)
                                                {
                                                    sym_mut.qualified_name =
                                                        Some(method_qname_interned);
                                                }
                                            }
                                        }
                                        new_symbol
                                    }
                                };

                                self.ensure_known_static_method_type(
                                    class_symbol,
                                    method_name,
                                    method_symbol,
                                );

                                // Get method return type by extracting it from the Function type
                                // (must be done before arg_exprs is moved into StaticMethodCall)
                                let expr_type = if let Some(symbol) =
                                    self.context.symbol_table.get_symbol(method_symbol)
                                {
                                    let type_table = self.context.type_table.borrow();
                                    if let Some(method_type) = type_table.get(symbol.type_id) {
                                        match &method_type.kind {
                                            crate::tast::core::TypeKind::Function {
                                                params,
                                                return_type,
                                                ..
                                            } => {
                                                let ret = *return_type;
                                                let params_owned = params.clone();
                                                // If return type is a TypeParameter, infer from arguments
                                                if type_table.is_type_parameter(ret) {
                                                    let mut inferred = ret;
                                                    for (i, param_ty) in
                                                        params_owned.iter().enumerate()
                                                    {
                                                        if *param_ty == ret && i < arg_exprs.len() {
                                                            inferred = arg_exprs[i].expr_type;
                                                            break;
                                                        }
                                                    }
                                                    inferred
                                                } else if let Some(ret_info) = type_table.get(ret) {
                                                    // Check if return type has TypeParameter args that need substitution.
                                                    // This handles both GenericInstance (e.g., Array<T>) and Class/Interface
                                                    // types whose definition carries type_args (e.g., Thread<T> stored as
                                                    // Class { type_args: [T] } rather than GenericInstance).
                                                    let (base_type_opt, ret_type_args_opt) = match &ret_info.kind {
                                                        crate::tast::core::TypeKind::GenericInstance {
                                                            base_type,
                                                            type_args: ret_type_args,
                                                            ..
                                                        } => (Some(*base_type), Some(ret_type_args.clone())),
                                                        crate::tast::core::TypeKind::Class {
                                                            type_args: ret_type_args,
                                                            ..
                                                        } | crate::tast::core::TypeKind::Interface {
                                                            type_args: ret_type_args,
                                                            ..
                                                        } if !ret_type_args.is_empty() && ret_type_args.iter().any(|ta| {
                                                            type_table.get(*ta).map_or(false, |info| {
                                                                matches!(info.kind, crate::tast::core::TypeKind::TypeParameter { .. })
                                                            })
                                                        }) => {
                                                            // Class/Interface with unresolved TypeParameter args — treat ret itself as base
                                                            (Some(ret), Some(ret_type_args.clone()))
                                                        }
                                                        _ => (None, None),
                                                    };
                                                    if let (Some(base_type), Some(ret_type_args)) =
                                                        (base_type_opt, ret_type_args_opt)
                                                    {
                                                        let mut subs: Vec<(TypeId, TypeId)> =
                                                            Vec::new();
                                                        for ret_ta in ret_type_args.iter() {
                                                            if let Some(ta_info) =
                                                                type_table.get(*ret_ta)
                                                            {
                                                                if let crate::tast::core::TypeKind::TypeParameter {
                                                                    symbol_id: tp_sym,
                                                                    ..
                                                                } = &ta_info.kind
                                                                {
                                                                    for (pi, param_ty) in params_owned.iter().enumerate() {
                                                                        if pi >= arg_exprs.len() {
                                                                            continue;
                                                                        }
                                                                        let arg_ty = arg_exprs[pi].expr_type;
                                                                        if let Some(concrete) =
                                                                            Self::match_type_param_in_types(
                                                                                *tp_sym,
                                                                                *param_ty,
                                                                                arg_ty,
                                                                                &type_table,
                                                                            )
                                                                        {
                                                                            subs.push((*ret_ta, concrete));
                                                                            break;
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }

                                                        if !subs.is_empty() {
                                                            let base_type_val = base_type;
                                                            let new_args: Vec<TypeId> =
                                                                ret_type_args
                                                                    .iter()
                                                                    .map(|ta| {
                                                                        subs.iter()
                                                                            .find(|(old, _)| {
                                                                                old == ta
                                                                            })
                                                                            .map(|(_, new)| *new)
                                                                            .unwrap_or(*ta)
                                                                    })
                                                                    .collect();
                                                            drop(type_table);
                                                            self.context
                                                                .type_table
                                                                .borrow_mut()
                                                                .create_generic_instance(
                                                                    base_type_val,
                                                                    new_args,
                                                                )
                                                        } else {
                                                            ret
                                                        }
                                                    } else {
                                                        ret
                                                    }
                                                } else {
                                                    ret
                                                }
                                            }
                                            _ => symbol.type_id,
                                        }
                                    } else {
                                        symbol.type_id
                                    }
                                } else {
                                    self.context.type_table.borrow().dynamic_type()
                                };

                                let kind = TypedExpressionKind::StaticMethodCall {
                                    class_symbol,
                                    method_symbol,
                                    arguments: arg_exprs,
                                    type_arguments: Vec::new(),
                                };

                                let usage = VariableUsage::Copy;
                                let lifetime_id = self.assign_lifetime(&kind, &expr_type);
                                let metadata = self.analyze_expression_metadata(&kind);

                                // Calculate the span for the field name specifically
                                // The field appears after the object expression and a dot
                                let field_span = parser::haxe_ast::Span::new(
                                    obj_expr.span.end + 1, // +1 for the dot
                                    obj_expr.span.end + 1 + field.len(),
                                );

                                return Ok(TypedExpression {
                                    expr_type,
                                    kind,
                                    usage,
                                    lifetime_id,
                                    source_location: self.context.span_to_location(&field_span),
                                    metadata,
                                });
                            }

                            // Check if this is an enum constructor call like MyResult.Ok(42)
                            if symbol.kind == crate::tast::symbols::SymbolKind::Enum {
                                let enum_symbol = symbol_id;
                                let variant_name = self.context.intern_string(field);

                                // Look up the enum variant
                                if let Some(variants) =
                                    self.context.symbol_table.get_enum_variants(enum_symbol)
                                {
                                    for &variant_id in variants {
                                        if let Some(variant_sym) =
                                            self.context.symbol_table.get_symbol(variant_id)
                                        {
                                            if variant_sym.name == variant_name {
                                                // This is an enum constructor call
                                                // Create a func_expr representing the enum variant
                                                let mut func_expr = TypedExpression {
                                                    expr_type: variant_sym.type_id,
                                                    kind: TypedExpressionKind::Variable {
                                                        symbol_id: variant_id,
                                                    },
                                                    usage: VariableUsage::Borrow,
                                                    lifetime_id: crate::tast::LifetimeId::first(),
                                                    source_location: self.context.create_location(),
                                                    metadata: ExpressionMetadata::default(),
                                                };

                                                // Instantiate the enum constructor type for proper return type
                                                func_expr = self
                                                    .instantiate_enum_constructor_type(
                                                        variant_id, &arg_exprs, func_expr,
                                                    )?;

                                                let kind = TypedExpressionKind::FunctionCall {
                                                    function: Box::new(func_expr),
                                                    arguments: arg_exprs,
                                                    type_arguments: Vec::new(),
                                                };

                                                let expr_type =
                                                    self.infer_expression_type(&kind)?;
                                                let usage = self.determine_variable_usage(&kind);
                                                let lifetime_id =
                                                    self.assign_lifetime(&kind, &expr_type);
                                                let metadata =
                                                    self.analyze_expression_metadata(&kind);

                                                return Ok(TypedExpression {
                                                    expr_type,
                                                    kind,
                                                    usage,
                                                    lifetime_id,
                                                    source_location: self
                                                        .context
                                                        .span_to_location(&expression.span),
                                                    metadata,
                                                });
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Handle multi-segment qualified static method calls
                // e.g., haxe.io.Bytes.alloc(8), sys.io.File.getContent(path)
                // extract_qualified_path recursively collects dotted path segments
                fn extract_qualified_path_for_call(expr: &parser::Expr) -> Option<Vec<String>> {
                    match &expr.kind {
                        ExprKind::Ident(name) => Some(vec![name.clone()]),
                        ExprKind::Field {
                            expr: inner_expr,
                            field,
                            ..
                        } => {
                            let mut path = extract_qualified_path_for_call(inner_expr)?;
                            path.push(field.clone());
                            Some(path)
                        }
                        _ => None,
                    }
                }

                if let Some(qualified_parts) = extract_qualified_path_for_call(obj_expr) {
                    if qualified_parts.len() >= 2 {
                        // qualified_parts = ["haxe", "io", "Bytes"], field = "alloc"
                        let class_name = qualified_parts.last().unwrap();
                        let class_name_interned = self.context.intern_string(class_name);
                        let qualified_class_name = qualified_parts.join(".");
                        let qualified_class_interned =
                            self.context.intern_string(&qualified_class_name);

                        // Build QualifiedPath for namespace resolver
                        let package_interned: Vec<_> = qualified_parts[..qualified_parts.len() - 1]
                            .iter()
                            .map(|p| self.context.intern_string(p))
                            .collect();
                        let qpath = crate::tast::namespace::QualifiedPath::new(
                            package_interned,
                            class_name_interned,
                        );

                        // Try to resolve the class
                        let symbol_id_opt = self
                            .context
                            .namespace_resolver
                            .lookup_symbol(&qpath)
                            .or_else(|| {
                                self.context
                                    .symbol_table
                                    .lookup_symbol(
                                        crate::tast::ScopeId::first(),
                                        qualified_class_interned,
                                    )
                                    .map(|s| s.id)
                            })
                            .or_else(|| {
                                self.resolve_symbol_in_scope_hierarchy(qualified_class_interned)
                            })
                            .or_else(|| {
                                self.resolve_class_like_symbol_by_name(class_name_interned)
                            });

                        if let Some(symbol_id) = symbol_id_opt {
                            if let Some(symbol) = self.context.symbol_table.get_symbol(symbol_id) {
                                // For TypeAlias, resolve through the alias chain to find
                                // the underlying class (e.g., haxe.io.Bytes -> rayzor.Bytes)
                                // For TypeAlias, resolve through alias chain to find underlying class
                                let (resolved_symbol_id, resolved_kind) = if symbol.kind
                                    == crate::tast::symbols::SymbolKind::TypeAlias
                                {
                                    // Extract placeholder name if target is unresolved
                                    let (resolved_type, placeholder_name) = {
                                        let type_table = self.context.type_table.borrow();
                                        let resolved =
                                            Self::resolve_alias_chain(&type_table, symbol.type_id);
                                        let ph_name = type_table.get(resolved).and_then(|ti| {
                                            if let crate::tast::core::TypeKind::Placeholder {
                                                name,
                                            } = &ti.kind
                                            {
                                                self.context
                                                    .string_interner
                                                    .get(*name)
                                                    .map(|s| s.to_string())
                                            } else {
                                                None
                                            }
                                        });
                                        (resolved, ph_name)
                                    };

                                    if let Some(ref ph_name) = placeholder_name {
                                        // TypeAlias target is unresolved — try to find
                                        // the class by the placeholder name in scope
                                        let ph_interned = self.context.intern_string(ph_name);
                                        let target_sym_id_opt = self
                                            .resolve_symbol_in_scope_hierarchy(ph_interned)
                                            .or_else(|| {
                                                // For qualified names (e.g. "rayzor.Bytes"), split
                                                // into package + short name and look up via namespace
                                                if ph_name.contains('.') {
                                                    let parts: Vec<&str> =
                                                        ph_name.rsplitn(2, '.').collect();
                                                    if parts.len() == 2 {
                                                        let class_name = parts[0];
                                                        let package_str = parts[1];
                                                        let pkg_parts: Vec<InternedString> =
                                                            package_str
                                                                .split('.')
                                                                .map(|p| {
                                                                    self.context.intern_string(p)
                                                                })
                                                                .collect();
                                                        let class_int =
                                                            self.context.intern_string(class_name);
                                                        let qp = crate::tast::namespace::QualifiedPath::new(pkg_parts.clone(), class_int);
                                                        if let Some(sid) = self.context.namespace_resolver.lookup_symbol(&qp) {
                                                            return Some(sid);
                                                        }
                                                        // Also try short name in root scope
                                                        if let Some(sym) = self.context.symbol_table.lookup_symbol(ScopeId::first(), class_int) {
                                                            return Some(sym.id);
                                                        }
                                                    }
                                                    None
                                                } else {
                                                    None
                                                }
                                            });
                                        if let Some(target_sym_id) = target_sym_id_opt {
                                            if let Some(target_sym) =
                                                self.context.symbol_table.get_symbol(target_sym_id)
                                            {
                                                if target_sym.kind
                                                    == crate::tast::symbols::SymbolKind::Class
                                                {
                                                    (target_sym_id, target_sym.kind)
                                                } else {
                                                    // Found but not a class — trigger loading
                                                    return Err(LoweringError::UnresolvedType {
                                                        type_name: ph_name.clone(),
                                                        location: self
                                                            .context
                                                            .create_location_from_span(
                                                                expression.span,
                                                            ),
                                                    });
                                                }
                                            } else {
                                                return Err(LoweringError::UnresolvedType {
                                                    type_name: ph_name.clone(),
                                                    location: self
                                                        .context
                                                        .create_location_from_span(expression.span),
                                                });
                                            }
                                        } else {
                                            // Not in scope — trigger on-demand loading
                                            return Err(LoweringError::UnresolvedType {
                                                type_name: ph_name.clone(),
                                                location: self
                                                    .context
                                                    .create_location_from_span(expression.span),
                                            });
                                        }
                                    } else {
                                        // Target resolved — check if it's a Class
                                        let type_table = self.context.type_table.borrow();
                                        if let Some(type_info) = type_table.get(resolved_type) {
                                            if let crate::tast::core::TypeKind::Class {
                                                symbol_id: class_sym,
                                                ..
                                            } = &type_info.kind
                                            {
                                                let kind = self
                                                    .context
                                                    .symbol_table
                                                    .get_symbol(*class_sym)
                                                    .map(|s| s.kind)
                                                    .unwrap_or(symbol.kind);
                                                (*class_sym, kind)
                                            } else {
                                                (symbol_id, symbol.kind)
                                            }
                                        } else {
                                            (symbol_id, symbol.kind)
                                        }
                                    }
                                } else {
                                    (symbol_id, symbol.kind)
                                };

                                if resolved_kind == crate::tast::symbols::SymbolKind::Class {
                                    // Resolved the qualified class — now handle as static method call
                                    let class_symbol = if let Ok(type_table) =
                                        self.context.type_table.try_borrow()
                                    {
                                        if let Some(type_info) = self
                                            .context
                                            .symbol_table
                                            .get_symbol(resolved_symbol_id)
                                            .and_then(|s| type_table.get(s.type_id))
                                        {
                                            if let crate::tast::core::TypeKind::Class {
                                                symbol_id: ts_symbol,
                                                ..
                                            } = &type_info.kind
                                            {
                                                *ts_symbol
                                            } else {
                                                resolved_symbol_id
                                            }
                                        } else {
                                            resolved_symbol_id
                                        }
                                    } else {
                                        resolved_symbol_id
                                    };

                                    let method_name = self.context.intern_string(field);

                                    let method_symbol = {
                                        if let Some(sym) = self
                                            .resolve_class_method_symbol(class_symbol, method_name)
                                        {
                                            sym
                                        } else {
                                            let new_symbol = self
                                                .context
                                                .symbol_table
                                                .create_function(method_name);
                                            if let Some(class_sym) =
                                                self.context.symbol_table.get_symbol(class_symbol)
                                            {
                                                if let Some(class_qname) =
                                                    class_sym.qualified_name.and_then(|qn| {
                                                        self.context.string_interner.get(qn)
                                                    })
                                                {
                                                    let method_qname = format!(
                                                        "{}.{}",
                                                        class_qname,
                                                        self.context
                                                            .string_interner
                                                            .get(method_name)
                                                            .unwrap_or("")
                                                    );
                                                    let method_qname_interned =
                                                        self.context.intern_string(&method_qname);
                                                    if let Some(sym_mut) = self
                                                        .context
                                                        .symbol_table
                                                        .get_symbol_mut(new_symbol)
                                                    {
                                                        sym_mut.qualified_name =
                                                            Some(method_qname_interned);
                                                    }
                                                }
                                            }
                                            new_symbol
                                        }
                                    };

                                    self.ensure_known_static_method_type(
                                        class_symbol,
                                        method_name,
                                        method_symbol,
                                    );

                                    let expr_type = if let Some(symbol) =
                                        self.context.symbol_table.get_symbol(method_symbol)
                                    {
                                        let type_table = self.context.type_table.borrow();
                                        if let Some(method_type) = type_table.get(symbol.type_id) {
                                            match &method_type.kind {
                                                crate::tast::core::TypeKind::Function {
                                                    return_type,
                                                    params,
                                                    ..
                                                } => {
                                                    let ret = *return_type;
                                                    let params_owned = params.clone();
                                                    if type_table.is_type_parameter(ret) {
                                                        let mut inferred = ret;
                                                        for (i, param_ty) in
                                                            params_owned.iter().enumerate()
                                                        {
                                                            if *param_ty == ret
                                                                && i < arg_exprs.len()
                                                            {
                                                                inferred = arg_exprs[i].expr_type;
                                                                break;
                                                            }
                                                        }
                                                        inferred
                                                    } else {
                                                        ret
                                                    }
                                                }
                                                _ => symbol.type_id,
                                            }
                                        } else {
                                            symbol.type_id
                                        }
                                    } else {
                                        self.context.type_table.borrow().dynamic_type()
                                    };

                                    let kind = TypedExpressionKind::StaticMethodCall {
                                        class_symbol,
                                        method_symbol,
                                        arguments: arg_exprs,
                                        type_arguments: Vec::new(),
                                    };

                                    let usage = VariableUsage::Copy;
                                    let lifetime_id = self.assign_lifetime(&kind, &expr_type);
                                    let metadata = self.analyze_expression_metadata(&kind);

                                    let field_span = parser::haxe_ast::Span::new(
                                        obj_expr.span.end + 1,
                                        obj_expr.span.end + 1 + field.len(),
                                    );

                                    return Ok(TypedExpression {
                                        expr_type,
                                        kind,
                                        usage,
                                        lifetime_id,
                                        source_location: self.context.span_to_location(&field_span),
                                        metadata,
                                    });
                                }
                            }
                        } else {
                            // Class not found — only return UnresolvedType if the first
                            // segment is a known package prefix. Otherwise fall through
                            // to field access (e.g., a.b.c.process() is field chain, not package)
                            let first_part = &qualified_parts[0];
                            if matches!(
                                first_part.as_str(),
                                "haxe"
                                    | "rayzor"
                                    | "sys"
                                    | "cpp"
                                    | "cs"
                                    | "java"
                                    | "python"
                                    | "lua"
                                    | "eval"
                                    | "neko"
                                    | "hl"
                                    | "flash"
                            ) {
                                return Err(LoweringError::UnresolvedType {
                                    type_name: qualified_class_name,
                                    location: self
                                        .context
                                        .create_location_from_span(expression.span),
                                });
                            }
                        }
                    }
                }

                // Not a static call, proceed with instance method call
                let mut receiver_expr = self.lower_expression(obj_expr)?;
                let method_name = self.context.intern_string(field);

                // Deref coercion (method call): if the receiver is an
                // auto-deref wrapper (`Arc<T>` / `MutexGuard<T>`) and the
                // method doesn't exist on the wrapper itself, transparently
                // rewrite `wrapper.method(args)` as `wrapper.get().method(args)`.
                // Mirrors lower_field_expression's hook — synthesise the
                // parser-level `obj_expr.get()` Call and re-run
                // lower_expression so the standard method-call type
                // inference applies. Now works because cross-file generic
                // class metadata is shared on SymbolTable.
                let should_deref_method = self
                    .resolve_type_to_class_symbol(receiver_expr.expr_type)
                    .map(|class_sym| {
                        if !self.is_auto_deref_wrapper_class(class_sym) {
                            return false;
                        }
                        let on_wrapper = self
                            .class_methods
                            .get(&class_sym)
                            .map(|methods| methods.iter().any(|(n, _, _)| *n == method_name))
                            .unwrap_or(false)
                            || self
                                .context
                                .symbol_table
                                .get_symbol(class_sym)
                                .and_then(|cs| {
                                    self.context
                                        .symbol_table
                                        .lookup_symbol(cs.scope_id, method_name)
                                })
                                .map(|s| s.kind == crate::tast::symbols::SymbolKind::Function)
                                .unwrap_or(false);
                        !on_wrapper
                    })
                    .unwrap_or(false);
                if should_deref_method {
                    let synth = self.synth_get_call_expr(obj_expr);
                    if let Ok(rewritten) = self.lower_expression(&synth) {
                        receiver_expr = rewritten;
                    }
                }

                // Closure-valued data field call: `obj.fieldFn(args)` where
                // `fieldFn` is a *field* of function type (not a method). Lower it
                // as an indirect call through the field value — the same shape as
                // `var f = obj.fieldFn; f(args)`. Dispatching it as a method traps
                // at runtime because there is no method body of that name.
                if let Some((field_sym, fn_type)) =
                    self.resolve_function_typed_field(receiver_expr.expr_type, method_name)
                {
                    let lifetime_id = receiver_expr.lifetime_id;
                    let field_access = TypedExpression {
                        kind: TypedExpressionKind::FieldAccess {
                            object: Box::new(receiver_expr),
                            field_symbol: field_sym,
                            is_optional: is_optional_call,
                        },
                        expr_type: fn_type,
                        usage: VariableUsage::Borrow,
                        lifetime_id,
                        source_location: self.context.span_to_location(&expression.span),
                        metadata: ExpressionMetadata::default(),
                    };
                    let kind = TypedExpressionKind::FunctionCall {
                        function: Box::new(field_access),
                        arguments: arg_exprs,
                        type_arguments: Vec::new(),
                    };
                    let expr_type = self.infer_expression_type(&kind)?;
                    let usage = self.determine_variable_usage(&kind);
                    let lifetime_id = self.assign_lifetime(&kind, &expr_type);
                    let metadata = self.analyze_expression_metadata(&kind);
                    return Ok(TypedExpression {
                        expr_type,
                        kind,
                        usage,
                        lifetime_id,
                        source_location: self.context.span_to_location(&expression.span),
                        metadata,
                    });
                }

                // First, try to resolve as a regular method on the receiver
                let method_symbol = self.resolve_method_symbol(&receiver_expr, method_name);

                // Check if the resolved symbol is a placeholder (newly created function)
                // If so, try to find a static extension method from 'using' modules
                let is_placeholder = self
                    .context
                    .symbol_table
                    .get_symbol(method_symbol)
                    .map(|s| s.kind == crate::tast::symbols::SymbolKind::Function)
                    .unwrap_or(false);

                if is_placeholder {
                    // Try to find a static extension method
                    if let Some((class_symbol, static_method_symbol)) =
                        self.find_static_extension_method(method_name, receiver_expr.expr_type)
                    {
                        // Found a static extension! Convert to static method call
                        // with receiver as first argument
                        let mut new_args = vec![receiver_expr];
                        new_args.extend(arg_exprs);

                        TypedExpressionKind::StaticMethodCall {
                            class_symbol,
                            method_symbol: static_method_symbol,
                            arguments: new_args,
                            type_arguments: Vec::new(),
                        }
                    } else {
                        // No static extension found, use regular method call
                        TypedExpressionKind::MethodCall {
                            receiver: Box::new(receiver_expr),
                            method_symbol,
                            arguments: arg_exprs,
                            type_arguments: Vec::new(),
                            is_optional: is_optional_call,
                        }
                    }
                } else {
                    // Method was found on the receiver, use it
                    TypedExpressionKind::MethodCall {
                        receiver: Box::new(receiver_expr),
                        method_symbol,
                        arguments: arg_exprs,
                        type_arguments: Vec::new(),
                        is_optional: is_optional_call,
                    }
                }
            }
            _ => {
                // Regular function call
                let mut func_expr = self.lower_expression(expr)?;

                // Check if this is an enum constructor call and instantiate its type
                if let TypedExpressionKind::Variable { symbol_id } = &func_expr.kind {
                    if let Some(symbol) = self.context.symbol_table.get_symbol(*symbol_id) {
                        let mut resolved_id = *symbol_id;
                        let mut is_variant =
                            symbol.kind == crate::tast::symbols::SymbolKind::EnumVariant;

                        // Name collision fix: if we resolved to a TYPE but this is a call
                        // with args (e.g., `Error("oops")`, `Bool(v != 0)`), search for an
                        // EnumVariant with the same name. A type called with args is only
                        // valid as an enum constructor, so a same-named variant must win.
                        // Covers `Result.Error` vs the `haxe.io.Error` enum type AND
                        // `MetaValue.Bool` vs the builtin `Bool` Abstract type (the variant
                        // is shadowed in scope by the primitive; without this the
                        // construction silently elides to a value-less return -> W0020 ->
                        // SIGILL). Selection prefers an imported parent enum; falls back to
                        // a single unambiguous candidate for same-file (unimported) enums.
                        if !is_variant
                            && matches!(
                                symbol.kind,
                                crate::tast::symbols::SymbolKind::Enum
                                    | crate::tast::symbols::SymbolKind::Abstract
                                    | crate::tast::symbols::SymbolKind::Class
                                    | crate::tast::symbols::SymbolKind::Interface
                                    | crate::tast::symbols::SymbolKind::TypeAlias
                            )
                            && !arg_exprs.is_empty()
                        {
                            let name = symbol.name;
                            // Match by NAME STRING, not InternedString id: the colliding
                            // type symbol (e.g. builtin `Bool`) and the enum variant
                            // (`MetaValue.Bool`) are interned in different contexts and get
                            // different ids, so an id-only compare finds zero candidates.
                            let name_str = self
                                .context
                                .string_interner
                                .get(name)
                                .map(|s| s.to_string());
                            // Collect all enum variants (id, name) owned, releasing the
                            // symbol_table borrow before re-borrowing the interner.
                            let all_variants: Vec<_> = self
                                .context
                                .symbol_table
                                .all_symbols()
                                .filter(|s| s.kind == crate::tast::symbols::SymbolKind::EnumVariant)
                                .map(|s| (s.id, s.name))
                                .collect();
                            let candidates: Vec<_> = all_variants
                                .into_iter()
                                .filter(|(_, vn)| {
                                    *vn == name
                                        || (name_str.is_some()
                                            && self
                                                .context
                                                .string_interner
                                                .get(*vn)
                                                .map(|s| s.to_string())
                                                == name_str)
                                })
                                .map(|(id, _)| id)
                                .collect();

                            // Build set of imported qualified names from user imports
                            let mut imported_qnames = std::collections::BTreeSet::new();
                            for scope_id in [self.context.current_scope, ScopeId::first()] {
                                for entry in self.context.import_resolver.get_imports(scope_id) {
                                    let qn: String = entry
                                        .package_path
                                        .package
                                        .iter()
                                        .filter_map(|s| self.context.string_interner.get(*s))
                                        .chain(std::iter::once(
                                            self.context
                                                .string_interner
                                                .get(entry.package_path.name)
                                                .unwrap_or(""),
                                        ))
                                        .collect::<Vec<_>>()
                                        .join(".");
                                    imported_qnames.insert(qn);
                                }
                            }

                            // Pick the candidate whose parent enum's qualified name is imported
                            for &candidate_id in &candidates {
                                if let Some(parent_enum) = self
                                    .context
                                    .symbol_table
                                    .find_parent_enum_for_constructor(candidate_id)
                                {
                                    if let Some(parent_sym) =
                                        self.context.symbol_table.get_symbol(parent_enum)
                                    {
                                        if let Some(qn) = parent_sym.qualified_name {
                                            let qn_str =
                                                self.context.string_interner.get(qn).unwrap_or("");
                                            if imported_qnames.contains(qn_str) {
                                                if let Some(variant_sym) = self
                                                    .context
                                                    .symbol_table
                                                    .get_symbol(candidate_id)
                                                {
                                                    resolved_id = candidate_id;
                                                    is_variant = true;
                                                    func_expr = TypedExpression {
                                                        kind: TypedExpressionKind::Variable {
                                                            symbol_id: resolved_id,
                                                        },
                                                        expr_type: variant_sym.type_id,
                                                        usage: func_expr.usage.clone(),
                                                        lifetime_id: func_expr.lifetime_id,
                                                        source_location: func_expr.source_location,
                                                        metadata: func_expr.metadata.clone(),
                                                    };
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            // Same-file / unimported enum: the import-based match above
                            // didn't fire (the parent enum is declared in the same file as
                            // its consumer, so it is never in `imported_qnames`). Dedupe the
                            // candidates by their parent enum's qualified name — import
                            // compilation registers the same variant in multiple contexts,
                            // so raw `candidates.len()` is often >1 for one logical variant.
                            // If all candidates share ONE parent enum, the variant is
                            // unambiguous — prefer it. This is the `MetaValue.Bool` case.
                            if !is_variant && !candidates.is_empty() {
                                let mut parent_qns = std::collections::BTreeSet::new();
                                for &c in &candidates {
                                    if let Some(pe) = self
                                        .context
                                        .symbol_table
                                        .find_parent_enum_for_constructor(c)
                                    {
                                        if let Some(ps) = self.context.symbol_table.get_symbol(pe) {
                                            // Key by qualified name when set, else the
                                            // bare enum name — a same-file enum's parent
                                            // often has no qualified_name, and multiple
                                            // registrations would otherwise look distinct.
                                            let key = ps
                                                .qualified_name
                                                .or(Some(ps.name))
                                                .and_then(|q| self.context.string_interner.get(q))
                                                .map(|s| s.to_string())
                                                .unwrap_or_else(|| format!("#{:?}", pe));
                                            parent_qns.insert(key);
                                        }
                                    }
                                }
                                if parent_qns.len() == 1 {
                                    let candidate_id = candidates[0];
                                    if let Some(variant_sym) =
                                        self.context.symbol_table.get_symbol(candidate_id)
                                    {
                                        resolved_id = candidate_id;
                                        is_variant = true;
                                        func_expr = TypedExpression {
                                            kind: TypedExpressionKind::Variable {
                                                symbol_id: resolved_id,
                                            },
                                            expr_type: variant_sym.type_id,
                                            usage: func_expr.usage.clone(),
                                            lifetime_id: func_expr.lifetime_id,
                                            source_location: func_expr.source_location,
                                            metadata: func_expr.metadata.clone(),
                                        };
                                    }
                                }
                            }
                        }

                        if is_variant {
                            // This is an enum constructor - instantiate its function type
                            func_expr = self.instantiate_enum_constructor_type(
                                resolved_id,
                                &arg_exprs,
                                func_expr,
                            )?;
                        }
                    }
                }

                // Check if this is an unqualified call to a method on the current class.
                // In Haxe, `calculate(10, 20)` inside a class method is `this.calculate(10, 20)`,
                // and `staticMethod()` is `ClassName.staticMethod()`.
                if let TypedExpressionKind::Variable { symbol_id } = &func_expr.kind {
                    let method_info =
                        self.context
                            .class_context_stack
                            .last()
                            .and_then(|class_sym| {
                                self.class_methods.get(class_sym).and_then(|methods| {
                                    methods
                                        .iter()
                                        .find(|(_, sym, _)| *sym == *symbol_id)
                                        .map(|(_, _, is_static)| (*class_sym, *is_static))
                                })
                            });

                    if let Some((class_symbol, is_static)) = method_info {
                        let method_symbol = *symbol_id;
                        // Get the return type of the method
                        let return_type = {
                            let sym = self.context.symbol_table.get_symbol(method_symbol);
                            if let Some(sym) = sym {
                                let type_table = self.context.type_table.borrow();
                                if let Some(method_type) = type_table.get(sym.type_id) {
                                    match &method_type.kind {
                                        crate::tast::core::TypeKind::Function {
                                            params,
                                            return_type,
                                            ..
                                        } => {
                                            let ret = *return_type;
                                            // If return type is a TypeParameter, infer from arguments
                                            if type_table.is_type_parameter(ret) {
                                                let mut inferred = ret;
                                                for (i, param_ty) in params.iter().enumerate() {
                                                    if *param_ty == ret && i < arg_exprs.len() {
                                                        inferred = arg_exprs[i].expr_type;
                                                        break;
                                                    }
                                                }
                                                inferred
                                            } else {
                                                ret
                                            }
                                        }
                                        _ => sym.type_id,
                                    }
                                } else {
                                    sym.type_id
                                }
                            } else {
                                func_expr.expr_type
                            }
                        };

                        let kind = if is_static {
                            // Static methods: create StaticMethodCall with the class symbol
                            TypedExpressionKind::StaticMethodCall {
                                class_symbol,
                                method_symbol,
                                arguments: arg_exprs,
                                type_arguments: Vec::new(),
                            }
                        } else {
                            // Instance methods: create MethodCall with implicit `this` receiver
                            let this_name = self.context.intern_string("this");
                            let this_symbol = self
                                .resolve_symbol_in_scope_hierarchy(this_name)
                                .unwrap_or_else(|| {
                                    self.context.symbol_table.create_variable(this_name)
                                });
                            let this_type = self
                                .context
                                .class_context_stack
                                .last()
                                .and_then(|cs| self.context.symbol_table.get_symbol(*cs))
                                .map(|s| s.type_id)
                                .unwrap_or_else(|| self.context.type_table.borrow().dynamic_type());
                            let receiver = TypedExpression {
                                expr_type: this_type,
                                kind: TypedExpressionKind::Variable {
                                    symbol_id: this_symbol,
                                },
                                usage: VariableUsage::Copy,
                                lifetime_id: crate::tast::LifetimeId::first(),
                                source_location: self.context.create_location(),
                                metadata: ExpressionMetadata::default(),
                            };
                            TypedExpressionKind::MethodCall {
                                receiver: Box::new(receiver),
                                method_symbol,
                                arguments: arg_exprs,
                                type_arguments: Vec::new(),
                                is_optional: false,
                            }
                        };

                        let usage = VariableUsage::Copy;
                        let lifetime_id = self.assign_lifetime(&kind, &return_type);
                        let metadata = self.analyze_expression_metadata(&kind);
                        return Ok(TypedExpression {
                            expr_type: return_type,
                            kind,
                            usage,
                            lifetime_id,
                            source_location: self.context.create_location(),
                            metadata,
                        });
                    }
                }

                TypedExpressionKind::FunctionCall {
                    function: Box::new(func_expr),
                    arguments: arg_exprs,
                    type_arguments: Vec::new(),
                }
            }
        };

        // Build the TypedExpression for the non-early-return paths
        let expr_type = self.infer_expression_type(&kind)?;
        let usage = self.determine_variable_usage(&kind);
        let lifetime_id = self.assign_lifetime(&kind, &expr_type);
        let metadata = self.analyze_expression_metadata(&kind);

        Ok(TypedExpression {
            expr_type,
            kind,
            usage,
            lifetime_id,
            source_location: self.context.span_to_location(&expression.span),
            metadata,
        })
    }

    /// A reference to a generic function (`Reflect.compare`, typed `(T, T) -> Int`)
    /// passed where a concrete function type of the same arity is expected takes
    /// that type, as a lambda in the same position is typed from its hint.
    fn instantiate_function_reference(
        &self,
        mut typed: TypedExpression,
        hint: Option<TypeId>,
    ) -> TypedExpression {
        use crate::tast::core::TypeKind;
        let Some(hint) = hint else { return typed };
        if matches!(typed.kind, TypedExpressionKind::FunctionLiteral { .. }) {
            return typed;
        }
        let tt = self.context.type_table.borrow();
        let is_param = |t: TypeId| {
            matches!(tt.get(t).map(|i| &i.kind), Some(TypeKind::TypeParameter { .. }))
        };
        let generic = match tt.get(typed.expr_type).map(|i| &i.kind) {
            Some(TypeKind::Function { params, .. }) => {
                Some(params.clone()).filter(|p| p.iter().any(|t| is_param(*t)))
            }
            _ => None,
        };
        let concrete = match tt.get(hint).map(|i| &i.kind) {
            Some(TypeKind::Function { params, .. }) => {
                Some(params.clone()).filter(|p| !p.iter().any(|t| is_param(*t)))
            }
            _ => None,
        };
        if let (Some(g), Some(c)) = (generic, concrete) {
            if g.len() == c.len() {
                typed.expr_type = hint;
            }
        }
        typed
    }

    /// Infer the return type of a method call, substituting type parameters from the receiver.
    pub(crate) fn structural_method_return_type(
        &self,
        receiver_type: TypeId,
        method: crate::tast::InternedString,
    ) -> Option<TypeId> {
        let trace = std::env::var_os("RAYZOR_STRUCT_TRACE").is_some();
        let mname = self
            .context
            .string_interner
            .get(method)
            .unwrap_or("?")
            .to_string();
        let mut current = receiver_type;
        // What each instantiated alias on the way binds its declaration's
        // parameters to; the structure itself mentions only the parameters.
        let mut bindings: Vec<(SymbolId, TypeId)> = Vec::new();
        for step in 0..8 {
            let kind = {
                let tt = self.context.type_table.borrow();
                tt.get(current).map(|ti| ti.kind.clone())
            };
            if trace {
                eprintln!(
                    "[STRUCT] m={mname} step={step} ty={current:?} kind={:?}",
                    kind.as_ref().map(std::mem::discriminant)
                );
            }
            let kind = kind?;
            match kind {
                crate::tast::core::TypeKind::Anonymous { fields } => {
                    let field = fields.iter().find(|f| f.name == method)?;
                    let r = {
                        let tt = self.context.type_table.borrow();
                        match tt.get(field.type_id).map(|i| &i.kind) {
                            Some(crate::tast::core::TypeKind::Function { return_type, .. }) => {
                                Some(*return_type)
                            }
                            _ => None,
                        }
                    };
                    if trace {
                        eprintln!("[STRUCT]   -> {r:?}");
                    }
                    return r.map(|t| self.substitute_alias_args(t, &bindings));
                }
                crate::tast::core::TypeKind::TypeAlias {
                    symbol_id,
                    target_type,
                    type_args,
                } => {
                    bindings.extend(self.alias_bindings(symbol_id, &type_args));
                    current = target_type
                }
                crate::tast::core::TypeKind::GenericInstance { base_type, .. } => {
                    current = base_type
                }
                // A typedef pre-registered as a class keeps that symbol kind, so
                // its instantiation is a Class node whose arguments bind the
                // alias's parameters all the same.
                crate::tast::core::TypeKind::Class {
                    symbol_id,
                    type_args,
                } => {
                    bindings.extend(self.alias_bindings(symbol_id, &type_args));
                    let resolved = {
                        self.context
                            .type_table
                            .borrow()
                            .resolve_type_alias(symbol_id)
                    };
                    if trace {
                        eprintln!("[STRUCT]   class -> alias {resolved:?}");
                    }
                    if resolved == current {
                        return None;
                    }
                    current = resolved;
                }
                _ => {
                    if trace {
                        eprintln!("[STRUCT]   stop");
                    }
                    return None;
                }
            }
        }
        None
    }

    /// What an instantiated alias binds each parameter of its declaration to.
    fn alias_bindings(&self, alias: SymbolId, args: &[TypeId]) -> Vec<(SymbolId, TypeId)> {
        let tt = self.context.type_table.borrow();
        let Some(decl) = self
            .context
            .symbol_table
            .get_symbol(alias)
            .and_then(|s| tt.get(s.type_id))
        else {
            return Vec::new();
        };
        let crate::tast::core::TypeKind::TypeAlias { type_args: params, .. } = &decl.kind else {
            return Vec::new();
        };
        params
            .iter()
            .zip(args)
            .filter_map(|(p, a)| match tt.get(*p).map(|i| &i.kind) {
                Some(crate::tast::core::TypeKind::TypeParameter { symbol_id, .. }) if p != a => {
                    Some((*symbol_id, *a))
                }
                _ => None,
            })
            .collect()
    }

    /// `ty` with every bound parameter replaced; a node is rebuilt only where
    /// something under it changed.
    fn substitute_alias_args(&self, ty: TypeId, bindings: &[(SymbolId, TypeId)]) -> TypeId {
        use crate::tast::core::{AnonymousField, TypeKind};
        if bindings.is_empty() {
            return ty;
        }
        let kind = self.context.type_table.borrow().get(ty).map(|i| i.kind.clone());
        let sub = |ids: &[TypeId]| -> Vec<TypeId> {
            ids.iter()
                .map(|t| self.substitute_alias_args(*t, bindings))
                .collect()
        };
        let rebuilt = match kind {
            Some(TypeKind::TypeParameter { symbol_id, .. }) => {
                return match bindings.iter().find(|(p, _)| *p == symbol_id) {
                    Some((_, a)) => self.substitute_alias_args(*a, bindings),
                    None => ty,
                };
            }
            Some(TypeKind::TypeAlias {
                symbol_id,
                target_type,
                type_args,
            }) => {
                let args = sub(&type_args);
                if args == type_args {
                    return ty;
                }
                TypeKind::TypeAlias {
                    symbol_id,
                    target_type,
                    type_args: args,
                }
            }
            Some(TypeKind::GenericInstance {
                base_type,
                type_args,
                instantiation_cache_id,
            }) => {
                let args = sub(&type_args);
                if args == type_args {
                    return ty;
                }
                TypeKind::GenericInstance {
                    base_type,
                    type_args: args,
                    instantiation_cache_id,
                }
            }
            Some(TypeKind::Array { element_type }) => {
                let elem = self.substitute_alias_args(element_type, bindings);
                if elem == element_type {
                    return ty;
                }
                TypeKind::Array { element_type: elem }
            }
            Some(TypeKind::Function {
                params,
                return_type,
                effects,
            }) => {
                let p = sub(&params);
                let r = self.substitute_alias_args(return_type, bindings);
                if p == params && r == return_type {
                    return ty;
                }
                TypeKind::Function {
                    params: p,
                    return_type: r,
                    effects,
                }
            }
            Some(TypeKind::Anonymous { fields }) => {
                let rebuilt: Vec<AnonymousField> = fields
                    .iter()
                    .map(|f| AnonymousField {
                        name: f.name,
                        type_id: self.substitute_alias_args(f.type_id, bindings),
                        is_public: f.is_public,
                        optional: f.optional,
                    })
                    .collect();
                if rebuilt.iter().zip(&fields).all(|(a, b)| a.type_id == b.type_id) {
                    return ty;
                }
                TypeKind::Anonymous { fields: rebuilt }
            }
            _ => return ty,
        };
        self.context.type_table.borrow_mut().create_type(rebuilt)
    }

    pub(crate) fn infer_method_call_return_type(
        &mut self,
        method_symbol: SymbolId,
        receiver_type: TypeId,
    ) -> LoweringResult<TypeId> {
        // Phase 1: Collect all necessary information with immutable borrow
        let substitution_result = {
            let type_table = self.context.type_table.borrow();

            // Get the method symbol
            let method_type_id = match self.context.symbol_table.get_symbol(method_symbol) {
                Some(symbol) if symbol.type_id.is_valid() => symbol.type_id,
                _ => {
                    // Method symbol has no type info. This happens for true
                    // built-ins (Array/String placeholders) AND for a method
                    // that a cross-module / typedef receiver bound to a TYPELESS
                    // PLACEHOLDER symbol — e.g. `bytes.sub(...)` where `bytes :
                    // haxe.io.Bytes` (a typedef for rayzor.Bytes whose phantom
                    // Class has an empty scope). Falling straight to
                    // `infer_builtin_method_type` there silently decays the
                    // return type to Dynamic (no Bytes arm), and a later
                    // `.toString()` on that Dynamic mis-dispatches to an
                    // arbitrary concrete class. So FIRST recover the real
                    // DECLARED method from the receiver class via
                    // `resolve_class_method_symbol` (which already handles the
                    // typedef / phantom-class case and only returns a
                    // valid-typed symbol) and use its declared return type.
                    drop(type_table);
                    let method_name_intern = self
                        .context
                        .symbol_table
                        .get_symbol(method_symbol)
                        .map(|s| s.name);
                    let recv_class_symbol = {
                        let tt = self.context.type_table.borrow();
                        let mut cur = receiver_type;
                        let mut found = None;
                        for _ in 0..8 {
                            match tt.get(cur).map(|ti| &ti.kind) {
                                Some(crate::tast::core::TypeKind::Class { symbol_id, .. }) => {
                                    found = Some(*symbol_id);
                                    break;
                                }
                                Some(crate::tast::core::TypeKind::TypeAlias {
                                    target_type,
                                    ..
                                }) => cur = *target_type,
                                Some(crate::tast::core::TypeKind::GenericInstance {
                                    base_type,
                                    ..
                                }) => cur = *base_type,
                                // `Null<C>.method()` — the declaring class is C.
                                Some(crate::tast::core::TypeKind::Optional {
                                    inner_type, ..
                                }) => cur = *inner_type,
                                _ => break,
                            }
                        }
                        found
                    };
                    if let (Some(mn), Some(cs)) = (method_name_intern, recv_class_symbol) {
                        if let Some(real_method) = self.resolve_class_method_symbol(cs, mn) {
                            let real_valid = real_method != method_symbol
                                && self
                                    .context
                                    .symbol_table
                                    .get_symbol(real_method)
                                    .map_or(false, |s| s.type_id.is_valid());
                            if real_valid {
                                return self
                                    .infer_method_call_return_type(real_method, receiver_type);
                            }
                        }
                        // No typed declaration symbol exists anywhere yet (the
                        // receiver class's file TAST-lowers later). Recover the
                        // DECLARED signature from the parsed AST via the sig
                        // index and type this symbol from it, so the call's
                        // return doesn't decay to Dynamic (whose later member
                        // dispatch guesses among same-named methods).
                        if let Some(sig) = self.resolve_declared_method_sig(cs, mn, false) {
                            let fn_ty = self.apply_declared_sig(method_symbol, &sig);
                            let ret = {
                                let tt = self.context.type_table.borrow();
                                match tt.get(fn_ty).map(|i| &i.kind) {
                                    Some(crate::tast::core::TypeKind::Function {
                                        return_type,
                                        ..
                                    }) => Some(*return_type),
                                    _ => None,
                                }
                            };
                            if let Some(ret) = ret {
                                return Ok(ret);
                            }
                        }
                    }
                    if let Some(mn) = method_name_intern {
                        if let Some(ret) = self.structural_method_return_type(receiver_type, mn) {
                            return Ok(ret);
                        }
                    }

                    // Use infer_builtin_method_type to get the method's function type,
                    // then extract the return type from it.
                    let method_func_type =
                        self.infer_builtin_method_type(receiver_type, method_symbol)?;
                    let type_table = self.context.type_table.borrow();
                    return match type_table.get(method_func_type) {
                        Some(info) => match &info.kind {
                            crate::tast::core::TypeKind::Function { return_type, .. } => {
                                Ok(*return_type)
                            }
                            _ => Ok(method_func_type), // Not a function type — it's the type itself (e.g., length: Int)
                        },
                        None => Ok(type_table.dynamic_type()),
                    };
                }
            };

            // Get the method's function type
            let return_type = match type_table.get(method_type_id) {
                Some(method_type) => match &method_type.kind {
                    crate::tast::core::TypeKind::Function { return_type, .. } => *return_type,
                    _ => return Ok(type_table.dynamic_type()),
                },
                None => return Ok(type_table.dynamic_type()),
            };

            // Compute the substitution
            let sub_result =
                self.compute_type_substitution(return_type, receiver_type, &type_table);
            if matches!(sub_result, TypeSubstitutionResult::NoChange(_)) {
                if let Some(rt_info) = type_table.get(return_type) {
                    if let crate::tast::core::TypeKind::TypeParameter {
                        symbol_id: ret_sym, ..
                    } = &rt_info.kind
                    {
                        if let Some(recv_info) = type_table.get(receiver_type) {
                            if let crate::tast::core::TypeKind::GenericInstance {
                                base_type,
                                type_args: recv_type_args,
                                ..
                            } = &recv_info.kind
                            {
                                if let Some(base_info) = type_table.get(*base_type) {
                                    if let crate::tast::core::TypeKind::Class {
                                        type_args: base_params,
                                        ..
                                    } = &base_info.kind
                                    {}
                                }
                            }
                        }
                    }
                }
            }
            sub_result
        };

        // Phase 2: Create new type if needed (with mutable borrow)
        match substitution_result {
            TypeSubstitutionResult::NoChange(type_id) => Ok(type_id),
            TypeSubstitutionResult::DirectSubstitution(type_id) => Ok(type_id),
            TypeSubstitutionResult::NeedGenericInstance {
                base_type,
                type_args,
            } => Ok(self
                .context
                .type_table
                .borrow_mut()
                .create_generic_instance(base_type, type_args)),
            TypeSubstitutionResult::NeedClassInstance {
                symbol_id,
                type_args,
            } => Ok(self
                .context
                .type_table
                .borrow_mut()
                .create_class_type(symbol_id, type_args)),
            TypeSubstitutionResult::NeedOptional { inner_type } => Ok(self
                .context
                .type_table
                .borrow_mut()
                .create_optional_type(inner_type)),
            TypeSubstitutionResult::NeedTypeAlias {
                symbol_id,
                target_type,
                type_args,
            } => Ok(self.context.type_table.borrow_mut().create_type(
                crate::tast::core::TypeKind::TypeAlias {
                    symbol_id,
                    target_type,
                    type_args,
                },
            )),
        }
    }

    pub(crate) fn function_param_types_from_symbol(&self, sym_id: SymbolId) -> Option<Vec<TypeId>> {
        let sym = self.context.symbol_table.get_symbol(sym_id)?;
        let type_table = self.context.type_table.borrow();
        let t = type_table.get(sym.type_id)?;
        if let crate::tast::core::TypeKind::Function { params, .. } = &t.kind {
            Some(params.clone())
        } else {
            None
        }
    }
}
