//! Symbol and type resolution.

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
    /// Resolve a TypeId through TypeAlias chains to find the underlying type.
    pub(crate) fn resolve_alias_chain(type_table: &TypeTable, type_id: TypeId) -> TypeId {
        let mut current = type_id;
        for _ in 0..10 {
            match type_table.get(current).map(|t| &t.kind) {
                Some(TypeKind::TypeAlias { target_type, .. }) => current = *target_type,
                _ => break,
            }
        }
        current
    }

    /// Find the parent enum symbol for an enum constructor
    pub(crate) fn find_parent_enum_for_constructor(
        &self,
        constructor_symbol: SymbolId,
    ) -> Option<SymbolId> {
        self.context
            .symbol_table
            .find_parent_enum_for_constructor(constructor_symbol)
    }

    /// Resolve a symbol by walking up the scope hierarchy
    pub(crate) fn resolve_symbol_in_scope_hierarchy(
        &self,
        name: InternedString,
    ) -> Option<SymbolId> {
        let mut current_scope = self.context.current_scope;

        loop {
            // Check if symbol exists in current scope
            if let Some(symbol) = self.context.symbol_table.lookup_symbol(current_scope, name) {
                return Some(symbol.id);
            }

            // Get parent scope
            if let Some(scope) = self.context.scope_tree.get_scope(current_scope) {
                if let Some(parent_id) = scope.parent_id {
                    current_scope = parent_id;
                } else {
                    // No parent scope
                    break;
                }
            } else {
                // Invalid scope
                break;
            }
        }

        // Check if the symbol is a field of the current class (implicit this access)
        if let Some(class_symbol) = self.context.class_context_stack.last() {
            if let Some(field_list) = self.class_fields.get(class_symbol) {
                for (field_name, field_symbol, _is_static) in field_list {
                    if *field_name == name {
                        return Some(*field_symbol);
                    }
                }
            }
        }

        // Check if the symbol is a method of the current class
        if let Some(class_symbol) = self.context.class_context_stack.last() {
            if let Some(methods) = self.class_methods.get(class_symbol) {
                for (method_name, method_symbol, _) in methods {
                    if *method_name == name {
                        return Some(*method_symbol);
                    }
                }
            }
        }

        // Inherited members. `class_fields`/`class_methods` above hold only what
        // this context lowered, so a parent from another module is absent from
        // both -- which is why a bare `eq(...)` inside `class T extends
        // unit.Test` failed to resolve while the same code with a same-module
        // parent worked. Each ancestor's members live in its own scope in the
        // shared symbol table, so walk the chain and look there.
        if let Some(class_symbol) = self.context.class_context_stack.last().copied() {
            if std::env::var("RAYZOR_INHERIT_DEBUG").is_ok() {
                eprintln!(
                    "[lookup] name={:?} class={:?} parent={:?} parent_scope_has={:?}",
                    self.context.string_interner.get(name),
                    class_symbol,
                    self.class_parents.get(&class_symbol),
                    self.class_parents.get(&class_symbol).and_then(|p| {
                        self.context.symbol_table.get_symbol(*p).map(|s| {
                            self.context
                                .symbol_table
                                .lookup_symbol(s.scope_id, name)
                                .is_some()
                        })
                    })
                );
            }
            let mut current = class_symbol;
            let mut seen: std::collections::BTreeSet<SymbolId> = std::collections::BTreeSet::new();
            seen.insert(current);
            while let Some(&parent) = self.class_parents.get(&current) {
                if !seen.insert(parent) {
                    break; // cyclic `extends`; the type checker reports it
                }
                if let Some(members) = self.class_methods.get(&parent) {
                    if let Some((_, sym, _)) = members.iter().find(|(n, _, _)| *n == name) {
                        return Some(*sym);
                    }
                }
                if let Some(members) = self.class_fields.get(&parent) {
                    if let Some((_, sym, _)) = members.iter().find(|(n, _, _)| *n == name) {
                        return Some(*sym);
                    }
                }
                // The declaration-level source of truth: the parent's own scope,
                // populated when ITS module registered its methods.
                if let Some(parent_scope) = self
                    .context
                    .symbol_table
                    .get_symbol(parent)
                    .map(|s| s.scope_id)
                {
                    if let Some(sym) = self.context.symbol_table.lookup_symbol(parent_scope, name) {
                        return Some(sym.id);
                    }
                }
                current = parent;
            }
        }

        // Fallback: explicitly check the global root scope (ScopeId::first())
        // This is needed for symbols like enum variants that are registered globally
        // but may not be reachable through the current scope's parent chain
        // (e.g., imported enums from other packages)
        let root_scope = ScopeId::first();
        if current_scope != root_scope {
            if let Some(symbol) = self.context.symbol_table.lookup_symbol(root_scope, name) {
                return Some(symbol.id);
            }
        }

        None
    }

    /// Resolve built-in types
    pub(crate) fn resolve_builtin_type(&self, name: &str) -> Option<TypeId> {
        let type_table = self.context.type_table.borrow();
        match name {
            "Int" => Some(type_table.int_type()),
            "Float" => Some(type_table.float_type()),
            "String" => Some(type_table.string_type()),
            "Bool" => Some(type_table.bool_type()),
            "Dynamic" => Some(type_table.dynamic_type()),
            "Void" => Some(type_table.void_type()),
            "Array" => {
                // Array<T> needs type parameter, return dynamic array for now
                let dynamic_type = type_table.dynamic_type();
                drop(type_table); // Release borrow before mutable borrow
                Some(
                    self.context
                        .type_table
                        .borrow_mut()
                        .create_array_type(dynamic_type),
                )
            }
            _ => None,
        }
    }

    /// Declared signature for `class_symbol::method_name` from the parsed-AST
    /// sig index: the class's qualified name first (following typedef
    /// aliases), then its bare name — accepted only when a UNIQUE indexed
    /// class declares the method, never a pick among candidates.
    pub(crate) fn resolve_declared_method_sig(
        &mut self,
        class_symbol: SymbolId,
        method_name: InternedString,
        is_static: bool,
    ) -> Option<crate::tast::sig_index::StaticMethodSig> {
        let index = self.static_sig_index.as_ref()?.clone();
        let (qname, bare): (Option<String>, Option<String>) = {
            let sym = self.context.symbol_table.get_symbol(class_symbol)?;
            (
                sym.qualified_name
                    .and_then(|q| self.context.string_interner.get(q))
                    .map(str::to_string),
                self.context
                    .string_interner
                    .get(sym.name)
                    .map(str::to_string),
            )
        };
        let method = self.context.string_interner.get(method_name)?.to_string();
        let resolver: &super::namespace::NamespaceResolver = self.context.namespace_resolver;
        let resolve_file = |q: &str| resolver.resolve_qualified_path_to_file_force(q);
        let mut index = index.borrow_mut();
        if let Some(q) = &qname {
            if let Some(sig) = index.resolve(q, &method, is_static, &resolve_file) {
                return Some(sig);
            }
        }
        if let Some(b) = &bare {
            if qname.as_ref() != Some(b) {
                if let Some(sig) = index.resolve(b, &method, is_static, &resolve_file) {
                    return Some(sig);
                }
            }
        }
        None
    }

    /// If `field_name` names a data field of *function* type on the class of
    /// `receiver_type`, return `(field_symbol, function_type)`. This lets a call
    /// `obj.fieldFn(args)` be lowered as an indirect call through the field value
    /// (like `var f = obj.fieldFn; f(args)`) instead of a method dispatch — there
    /// is no method body to dispatch to, so the method path traps at runtime.
    pub(crate) fn resolve_function_typed_field(
        &self,
        receiver_type: TypeId,
        field_name: InternedString,
    ) -> Option<(SymbolId, TypeId)> {
        let class_sym = self.resolve_type_to_class_symbol(receiver_type)?;
        let field_sym = self.lookup_data_field(class_sym, field_name)?;
        let sym = self.context.symbol_table.get_symbol(field_sym)?;
        // A method would be SymbolKind::Function — that's the method path, not this.
        if sym.kind == crate::tast::symbols::SymbolKind::Function {
            return None;
        }
        let fn_type = sym.type_id;
        let is_fn = self
            .context
            .type_table
            .borrow()
            .get(fn_type)
            .map(|t| matches!(t.kind, crate::tast::core::TypeKind::Function { .. }))
            .unwrap_or(false);
        if is_fn {
            Some((field_sym, fn_type))
        } else {
            None
        }
    }

    /// Resolve a class-like symbol by simple name.
    ///
    /// First tries lexical scope resolution, then falls back to a global symbol
    /// table scan for Class/Abstract/TypeAlias symbols with that name.
    pub(crate) fn resolve_class_like_symbol_by_name(
        &self,
        name: InternedString,
    ) -> Option<SymbolId> {
        if let Some(symbol_id) = self.resolve_symbol_in_scope_hierarchy(name) {
            if let Some(sym) = self.context.symbol_table.get_symbol(symbol_id) {
                if matches!(
                    sym.kind,
                    crate::tast::symbols::SymbolKind::Class
                        | crate::tast::symbols::SymbolKind::Abstract
                        | crate::tast::symbols::SymbolKind::TypeAlias
                        | crate::tast::symbols::SymbolKind::Enum
                ) {
                    return Some(symbol_id);
                }
            }
        }

        let mut matches = self.context.symbol_table.find_symbols(|sym| {
            sym.name == name
                && matches!(
                    sym.kind,
                    crate::tast::symbols::SymbolKind::Class
                        | crate::tast::symbols::SymbolKind::Abstract
                        | crate::tast::symbols::SymbolKind::TypeAlias
                        | crate::tast::symbols::SymbolKind::Enum
                )
        });

        if matches.is_empty() {
            return None;
        }
        if matches.len() == 1 {
            return Some(matches[0].id);
        }

        if let Some(name_str) = self.context.string_interner.get(name) {
            if let Some(sym) = matches.iter().find(|sym| {
                sym.qualified_name
                    .and_then(|qn| self.context.string_interner.get(qn))
                    .map(|qn| qn == name_str)
                    .unwrap_or(false)
            }) {
                return Some(sym.id);
            }
        }

        Some(matches.remove(0).id)
    }

    /// Try to resolve an enum constructor using the switch discriminant type
    /// This is needed for Haxe pattern matching where `case Some(v):` needs to
    /// be resolved as `Option.Some` based on the switch expression's type
    pub(crate) fn resolve_enum_constructor_from_discriminant(
        &self,
        constructor_name: InternedString,
    ) -> Option<SymbolId> {
        // Get the current switch discriminant type
        let discriminant_type = self.context.switch_discriminant_type?;

        // Get the type to find the enum symbol
        let type_table = self.context.type_table.borrow();

        // Recursively unwrap GenericInstance to find the base enum
        // Handles nested generics like Option<Option<Int>>
        let mut current_type_id = discriminant_type;
        let enum_symbol = loop {
            let ty = type_table.get(current_type_id)?;
            match &ty.kind {
                crate::tast::core::TypeKind::Enum { symbol_id, .. } => break *symbol_id,
                crate::tast::core::TypeKind::GenericInstance { base_type, .. } => {
                    // Continue unwrapping to find the base enum
                    current_type_id = *base_type;
                }
                _ => return None,
            }
        };
        drop(type_table);

        // Look up the enum's variants
        let variants = self.context.symbol_table.get_enum_variants(enum_symbol)?;

        // Find the variant with the matching name
        for &variant_id in variants {
            if let Some(variant_symbol) = self.context.symbol_table.get_symbol(variant_id) {
                if variant_symbol.name == constructor_name {
                    return Some(variant_id);
                }
            }
        }

        None
    }

    /// Lower a function parameter
    /// Inspect a call's callee expression and, when each formal parameter
    /// is itself a function type with concrete parameter types, return a
    /// `Vec<Option<Vec<TypeId>>>` matching `args` positions. Used so that a
    /// lambda argument passed to `parallelFor(items, fn:(idx:Int, node:Int)->Void)`
    /// can pick up `[Int, Int]` for its untyped `function(i, n)` params.
    /// Each inner `Option` is `Some(param_types)` only if THAT formal slot
    /// is a function type — non-function slots return `None` so we don't
    /// accidentally inject lambda types into unrelated args.
    pub(crate) fn resolve_callee_param_types(
        &mut self,
        callee: &Expr,
    ) -> Option<Vec<Option<Vec<TypeId>>>> {
        // Resolve callee's formal-parameter type list. Two cases handled:
        //   (1) Direct call to a free function / static method via Ident
        //   (2) Instance/static method call via Field { obj, field, .. }
        let formal_param_types = self.resolve_callee_formal_param_types(callee)?;

        let type_table = self.context.type_table.borrow();
        Some(
            formal_param_types
                .iter()
                .map(|pty| {
                    let t = type_table.get(*pty)?;
                    if let crate::tast::core::TypeKind::Function { params, .. } = &t.kind {
                        Some(params.clone())
                    } else {
                        None
                    }
                })
                .collect(),
        )
    }

    /// Best-effort lookup of a callee's formal parameter `TypeId`s. Returns
    /// `None` if the callee can't be statically resolved (e.g. lambda call,
    /// runtime extern with no signature, dynamic dispatch through a value
    /// whose type isn't a `Function`). Callers should treat `None` as "no
    /// expected types available — lower args with the existing defaults."
    pub(crate) fn resolve_callee_formal_param_types(
        &mut self,
        callee: &Expr,
    ) -> Option<Vec<TypeId>> {
        match &callee.kind {
            ExprKind::Ident(name) => {
                let name_interned = self.context.string_interner.intern(name);
                // Direct scope lookup first: free functions and locals/params
                // holding a closure value (a local shadows a same-name method).
                if let Some(sym) = self
                    .context
                    .symbol_table
                    .lookup_symbol(self.context.current_scope, name_interned)
                {
                    if let Some(params) = self.function_param_types_from_symbol(sym.id) {
                        return Some(params);
                    }
                }
                // Fall back: an unqualified call inside a class method is a
                // call to a method of the current class (`run(...)` ==
                // `ThisClass.run(...)`). The bare scope lookup misses
                // same-class static/instance methods, so consult the
                // `class_methods` table the real call path uses (9827+).
                // Without this, a lambda passed to a same-class method gets
                // no expected-param-types hint and its untyped params default
                // to Dynamic → `*void` MIR formals → the caller passes raw
                // i32 that the body unboxes (deref of a small int → SIGSEGV).
                let class_sym = *self.context.class_context_stack.last()?;
                let method_sym = self.resolve_class_method_symbol(class_sym, name_interned)?;
                self.function_param_types_from_symbol(method_sym)
            }
            ExprKind::Field {
                expr: obj, field, ..
            } => {
                // Static call: `ClassName.method(...)` — receiver is an Ident
                // whose symbol is a Class. Look up the method in that class.
                if let ExprKind::Ident(cls_name) = &obj.kind {
                    let cls_name_interned = self.context.string_interner.intern(cls_name);
                    if let Some(sym) = self
                        .context
                        .symbol_table
                        .lookup_symbol(self.context.current_scope, cls_name_interned)
                    {
                        if sym.kind == crate::tast::symbols::SymbolKind::Class {
                            let method_name = self.context.string_interner.intern(field);
                            if let Some(method_sym) =
                                self.resolve_class_method_symbol(sym.id, method_name)
                            {
                                return self.function_param_types_from_symbol(method_sym);
                            }
                        }
                    }
                }
                // Instance call: lower the receiver to find its class, then
                // look up the method in that class's scope.
                let receiver_typed = self.lower_expression(obj).ok()?;
                let receiver_type_id = receiver_typed.expr_type;
                let class_symbol = {
                    let type_table = self.context.type_table.borrow();
                    let t = type_table.get(receiver_type_id)?;
                    match &t.kind {
                        crate::tast::core::TypeKind::Class { symbol_id, .. } => Some(*symbol_id),
                        _ => None,
                    }
                }?;
                let method_name = self.context.string_interner.intern(field);
                let method_sym = self.resolve_class_method_symbol(class_symbol, method_name)?;
                self.function_param_types_from_symbol(method_sym)
            }
            _ => None,
        }
    }

    /// Find a `TypeAlias` symbol in the shared type table whose qualified
    /// name matches `qname` and whose target resolves (directly, or through
    /// further aliases) to `Anonymous`. Mirrors
    /// `hir_to_mir::find_typedef_anonymous_target_by_name` one layer up, at
    /// TAST type inference: a cross-module structural typedef reference
    /// (e.g. a field typed `metadata:ModelMetadata` where `ModelMetadata`
    /// is itself `typedef ModelMetadata = {...}`) can resolve to a bare
    /// synthetic `Class` symbol instead of the real `TypeAlias`/`Anonymous`
    /// chain (see `resolve_type_reference`'s TypeAlias arms). Without this
    /// recovery, `infer_expression_type`'s `FieldAccess` shape-read loop
    /// can't recognise the receiver as Anonymous and falls through to the
    /// Dynamic fallback in `infer_builtin_method_type`.
    pub(crate) fn find_typedef_anonymous_target_by_qname(&self, qname: &str) -> Option<TypeId> {
        let type_table = self.context.type_table.borrow();
        for (_tid, ti) in type_table.iter() {
            if let crate::tast::core::TypeKind::TypeAlias {
                symbol_id,
                target_type,
                ..
            } = &ti.kind
            {
                let matches = self
                    .context
                    .symbol_table
                    .get_symbol(*symbol_id)
                    .and_then(|s| {
                        s.qualified_name
                            .and_then(|n| self.context.string_interner.get(n))
                    })
                    == Some(qname);
                if matches {
                    let mut resolved = *target_type;
                    let mut hops = 0;
                    while hops < 8 {
                        match type_table.get(resolved).map(|t| &t.kind) {
                            Some(crate::tast::core::TypeKind::TypeAlias {
                                target_type, ..
                            }) => {
                                resolved = *target_type;
                                hops += 1;
                            }
                            _ => break,
                        }
                    }
                    if matches!(
                        type_table.get(resolved).map(|t| &t.kind),
                        Some(crate::tast::core::TypeKind::Anonymous { .. })
                    ) {
                        return Some(resolved);
                    }
                }
            }
        }
        None
    }

    /// Resolve a type by name (helper for overload parsing)
    pub(crate) fn resolve_type_by_name(
        &mut self,
        type_name: &str,
    ) -> Result<TypeId, LoweringError> {
        match type_name {
            "Void" => Ok(self.context.type_table.borrow().void_type()),
            "Int" => Ok(self.context.type_table.borrow().int_type()),
            "Float" => Ok(self.context.type_table.borrow().float_type()),
            "Bool" => Ok(self.context.type_table.borrow().bool_type()),
            "String" => Ok(self.context.type_table.borrow().string_type()),
            "Dynamic" => Ok(self.context.type_table.borrow().dynamic_type()),
            _ => {
                // Try to resolve as a class/interface/enum name using scope hierarchy
                let interned_name = self.context.intern_string(type_name);
                if let Some(symbol_id) = self.resolve_symbol_in_scope_hierarchy(interned_name) {
                    if let Some(symbol) = self.context.symbol_table.get_symbol(symbol_id) {
                        Ok(symbol.type_id)
                    } else {
                        Err(LoweringError::UnresolvedType {
                            type_name: type_name.to_string(),
                            location: SourceLocation::unknown(),
                        })
                    }
                } else {
                    Err(LoweringError::UnresolvedType {
                        type_name: type_name.to_string(),
                        location: SourceLocation::unknown(),
                    })
                }
            }
        }
    }

    /// Infer return type from function body by looking at return statements
    /// The type a return inside this expression yields, if any.
    ///
    /// Only the shapes a brace-less body can take: the return itself, and the
    /// block or conditional a body may be wrapped in. Anything else has no
    /// return to find.
    fn find_return_type_in_expression(&self, expr: &TypedExpression) -> Option<TypeId> {
        match &expr.kind {
            TypedExpressionKind::Return { value } => Some(
                value
                    .as_ref()
                    .map(|v| v.expr_type)
                    .unwrap_or_else(|| self.context.type_table.borrow().void_type()),
            ),
            TypedExpressionKind::Block { statements, .. } => statements
                .iter()
                .find_map(|s| self.find_return_type_in_statement(s)),
            TypedExpressionKind::Conditional {
                then_expr,
                else_expr,
                ..
            } => self.find_return_type_in_expression(then_expr).or_else(|| {
                else_expr
                    .as_ref()
                    .and_then(|e| self.find_return_type_in_expression(e))
            }),
            _ => None,
        }
    }

    /// Find return type from a statement (recursively search nested blocks)
    pub(crate) fn find_return_type_in_statement(&self, stmt: &TypedStatement) -> Option<TypeId> {
        match stmt {
            // A brace-less body -- `function get_high() return this.high;` --
            // lowers to a single expression statement wrapping a return, and
            // the whole body is wrapped this way regardless. Without this arm
            // the search finds no return at all and the function is typed
            // void, which is how Int64's accessors came to declare `-> void`
            // and then return an i32 from the body.
            TypedStatement::Expression { expression, .. } => {
                self.find_return_type_in_expression(expression)
            }
            TypedStatement::Return { value, .. } => {
                if let Some(expr) = value {
                    Some(expr.expr_type)
                } else {
                    Some(self.context.type_table.borrow().void_type())
                }
            }
            TypedStatement::Block { statements, .. } => {
                for s in statements {
                    if let Some(ret_type) = self.find_return_type_in_statement(s) {
                        return Some(ret_type);
                    }
                }
                None
            }
            TypedStatement::If {
                then_branch,
                else_branch,
                ..
            } => {
                // Check then branch
                if let Some(ret_type) = self.find_return_type_in_statement(then_branch.as_ref()) {
                    return Some(ret_type);
                }
                // Check else branch
                if let Some(else_stmt) = else_branch {
                    if let Some(ret_type) = self.find_return_type_in_statement(else_stmt.as_ref()) {
                        return Some(ret_type);
                    }
                }
                None
            }
            TypedStatement::While { body, .. }
            | TypedStatement::For { body, .. }
            | TypedStatement::ForIn { body, .. } => {
                self.find_return_type_in_statement(body.as_ref())
            }
            TypedStatement::Switch {
                cases,
                default_case,
                ..
            } => {
                for case in cases {
                    if let Some(ret_type) = self.find_return_type_in_statement(&case.body) {
                        return Some(ret_type);
                    }
                }
                if let Some(default) = default_case {
                    if let Some(ret_type) = self.find_return_type_in_statement(default.as_ref()) {
                        return Some(ret_type);
                    }
                }
                None
            }
            TypedStatement::Try {
                body,
                catch_clauses,
                finally_block,
                ..
            } => {
                if let Some(ret_type) = self.find_return_type_in_statement(body.as_ref()) {
                    return Some(ret_type);
                }
                for catch in catch_clauses {
                    if let Some(ret_type) = self.find_return_type_in_statement(&catch.body) {
                        return Some(ret_type);
                    }
                }
                if let Some(finally_stmt) = finally_block {
                    if let Some(ret_type) =
                        self.find_return_type_in_statement(finally_stmt.as_ref())
                    {
                        return Some(ret_type);
                    }
                }
                None
            }
            _ => None,
        }
    }
}
