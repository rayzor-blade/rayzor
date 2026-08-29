//! Abstract declarations and their underlying types.

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
    /// The span of a write to `this`, if the expression contains one.
    ///
    /// Deliberately conservative: it walks the shapes a method body actually
    /// uses, and an expression form it does not recognise yields None. A missed
    /// write means no diagnostic, which is exactly the behaviour without this
    /// check -- whereas a false positive would reject a legal program.
    fn find_this_write(expr: &parser::Expr) -> Option<parser::Span> {
        use parser::ExprKind as E;
        let is_this = |e: &parser::Expr| matches!(&e.kind, E::This);
        match &expr.kind {
            E::Assign { left, .. } if is_this(left) => Some(expr.span),
            E::Unary { op, expr: inner }
                if is_this(inner)
                    && matches!(
                        op,
                        parser::UnaryOp::PreIncr
                            | parser::UnaryOp::PostIncr
                            | parser::UnaryOp::PreDecr
                            | parser::UnaryOp::PostDecr
                    ) =>
            {
                Some(expr.span)
            }
            E::Assign { left, right, .. } => {
                Self::find_this_write(left).or_else(|| Self::find_this_write(right))
            }
            E::Unary { expr: inner, .. } => Self::find_this_write(inner),
            E::Block(stmts) => stmts.iter().find_map(|el| match el {
                parser::BlockElement::Expr(e) => Self::find_this_write(e),
                _ => None,
            }),
            E::Binary { left, right, .. } => {
                Self::find_this_write(left).or_else(|| Self::find_this_write(right))
            }
            E::If {
                cond,
                then_branch,
                else_branch,
            } => Self::find_this_write(cond)
                .or_else(|| Self::find_this_write(then_branch))
                .or_else(|| else_branch.as_ref().and_then(|e| Self::find_this_write(e))),
            E::While { cond, body } => {
                Self::find_this_write(cond).or_else(|| Self::find_this_write(body))
            }
            E::For { iter, body, .. } => {
                Self::find_this_write(iter).or_else(|| Self::find_this_write(body))
            }
            E::Return(Some(inner)) => Self::find_this_write(inner),
            E::Paren(inner) => Self::find_this_write(inner),
            E::Call { expr: callee, args } => Self::find_this_write(callee)
                .or_else(|| args.iter().find_map(|e| Self::find_this_write(e))),
            _ => None,
        }
    }

    /// Lower an abstract declaration
    pub(crate) fn lower_abstract_declaration(
        &mut self,
        abstract_decl: &AbstractDecl,
    ) -> LoweringResult<TypedDeclaration> {
        let abstract_name = self.context.intern_string(&abstract_decl.name);

        // Reuse the symbol created by the declaration pre-pass. Creating a
        // second abstract symbol here disconnects qualified field lookups from
        // the field initializers lowered below.
        let abstract_symbol = self
            .context
            .symbol_table
            .lookup_symbol(ScopeId::first(), abstract_name)
            .filter(|entry| entry.kind == crate::tast::SymbolKind::Abstract)
            .map(|entry| entry.id)
            .unwrap_or_else(|| {
                self.context
                    .symbol_table
                    .create_abstract_in_scope(abstract_name, ScopeId::first())
            });

        // Update qualified name (full path including class hierarchy)
        self.context.update_symbol_qualified_name(abstract_symbol);

        // Extract @:native metadata for abstracts
        let mut abstract_meta_flags =
            self.extract_metadata_flags(&abstract_decl.meta, abstract_symbol);
        // Also check for modifiers (extern, final, etc)
        for modifier in &abstract_decl.modifiers {
            match modifier {
                parser::haxe_ast::Modifier::Extern => {
                    abstract_meta_flags =
                        abstract_meta_flags.union(crate::tast::symbols::SymbolFlags::EXTERN);
                }
                parser::haxe_ast::Modifier::Final => {
                    abstract_meta_flags =
                        abstract_meta_flags.union(crate::tast::symbols::SymbolFlags::FINAL);
                }
                _ => {}
            }
        }
        if let Some(sym) = self.context.symbol_table.get_symbol_mut(abstract_symbol) {
            sym.flags = sym.flags.union(abstract_meta_flags);
        }

        // Extract @:forward metadata params (method/field names to forward to underlying type)
        let forward_fields: Vec<InternedString> = abstract_decl
            .meta
            .iter()
            .find(|m| {
                let name = m.name.strip_prefix(':').unwrap_or(&m.name);
                name == "forward"
            })
            .map(|m| {
                m.params
                    .iter()
                    .filter_map(|p| {
                        if let parser::haxe_ast::ExprKind::Ident(name) = &p.kind {
                            Some(self.context.intern_string(name))
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();

        // Add abstract to the root scope so it can be resolved for forward references
        self.context
            .scope_tree
            .get_scope_mut(ScopeId::first())
            .expect("Root scope should exist")
            .add_symbol(abstract_symbol, abstract_name);

        // Enter abstract scope with name
        let existing_scope = self
            .context
            .symbol_table
            .get_symbol(abstract_symbol)
            .map(|symbol| symbol.scope_id)
            .filter(|scope_id| {
                *scope_id != ScopeId::first()
                    && *scope_id != ScopeId::invalid()
                    && self
                        .context
                        .scope_tree
                        .get_scope(*scope_id)
                        .is_some_and(|scope| scope.kind == ScopeKind::Class)
            });
        let abstract_scope = if let Some(scope_id) = existing_scope {
            self.context.current_scope = scope_id;
            scope_id
        } else {
            self.context
                .enter_named_scope(ScopeKind::Class, abstract_name)
        };
        if let Some(symbol) = self.context.symbol_table.get_symbol_mut(abstract_symbol) {
            symbol.scope_id = abstract_scope;
        }

        // Process type parameters
        let type_params = self.lower_type_parameters(&abstract_decl.type_params)?;
        let mut type_param_map: BTreeMap<InternedString, TypeId> = BTreeMap::new();
        for tp in &type_params {
            let interned_name = tp.name;
            // Convert constraints to ConstraintKind for symbol table
            let constraint_kinds = tp
                .constraints
                .iter()
                .map(|_| {
                    crate::tast::type_checker::ConstraintKind::Implements {
                        interface_type: TypeId::invalid(), // Placeholder, will be resolved later
                    }
                })
                .collect();
            let symbol_id = self
                .context
                .symbol_table
                .create_type_parameter(interned_name, constraint_kinds);
            let type_id = self.context.type_table.borrow_mut().create_type_parameter(
                symbol_id,
                tp.constraints.clone(),
                tp.variance.into(),
            );
            type_param_map.insert(tp.name, type_id);
        }
        self.context.push_type_parameters(type_param_map);

        // Process underlying type
        let underlying_type = match &abstract_decl.underlying {
            Some(underlying) => Some(self.lower_type(underlying)?),
            None => {
                // @:coreType abstracts (e.g. `@:coreType abstract Void {}`) AND
                // extern/@:native abstracts (e.g. `@:native("rayzor::Ptr") extern
                // abstract Ptr<T>` — also Usize/Ref/Box) are OPAQUE: their
                // representation lives in the native impl, not a Haxe underlying
                // type. The stdlib-merge path tolerates the missing underlying,
                // but the import-compile path used to hard-error here, so when a
                // user file imported Ptr/Usize the module compiled to empty MIR
                // and every call into it trapped (udf #0xc11f / wasm unreachable)
                // at the importer's call sites. Treat them as opaque (None).
                let is_opaque = abstract_decl
                    .meta
                    .iter()
                    .any(|m| m.name == "coreType" || m.name == "native");
                if is_opaque {
                    None
                } else {
                    return Err(LoweringError::IncompleteImplementation {
                        feature: format!(
                            "Abstract type '{}' missing underlying type",
                            abstract_decl.name
                        ),
                        location: self.context.create_location_from_span(abstract_decl.span),
                    });
                }
            }
        };

        // Register the abstract type in the type table with the underlying type
        // so that resolve_this_type can return the underlying type for `this`
        {
            let abstract_type_id = self.context.type_table.borrow_mut().create_abstract_type(
                abstract_symbol,
                underlying_type,
                Vec::new(),
            );
            self.context
                .symbol_table
                .update_symbol_type(abstract_symbol, abstract_type_id);
        }

        // Process from/to types
        let from_types = abstract_decl
            .from
            .iter()
            .map(|t| self.lower_type(t))
            .collect::<Result<Vec<_>, _>>()?;
        let to_types = abstract_decl
            .to
            .iter()
            .map(|t| self.lower_type(t))
            .collect::<Result<Vec<_>, _>>()?;

        // Initialize class_fields for this abstract so field tracking works (needed for enum abstract)
        self.class_fields.entry(abstract_symbol).or_default();
        // Initialize class_methods so the pre-pass-typed abstract methods are
        // reachable by resolve_class_method_symbol Strategy 1 at call sites.
        // The class_methods map is the same mechanism regular classes use and
        // keeps `@:coreType` static calls (Atomic.of, Box.init) bound to their
        // typed method instead of a Dynamic placeholder.
        self.class_methods
            .entry(abstract_symbol)
            .or_insert_with(Vec::new);

        // Push abstract onto class context stack so `this` resolves correctly in method bodies
        self.context.class_context_stack.push(abstract_symbol);

        // Pre-pass: create symbols (with pre-computed signature types) for
        // every member BEFORE lowering any body, mirroring the class
        // lowering's method pre-pass at lower_class_declaration. Abstracts
        // previously lowered members strictly in source order, so a method
        // body referencing a member declared LATER in the file failed
        // resolution — e.g. haxe.Int64's `copy()` (line ~43) reads `high`,
        // a property declared at line ~445 ("Cannot find name 'high'"),
        // and Int32's operator methods call the `clamp` static declared
        // below them ("Cannot find name 'clamp'"). These fired as
        // [IMPORT[...]] errors on every fresh stdlib compile and were
        // tolerated only by retry/fallback machinery.
        for field in &abstract_decl.fields {
            let is_static = field
                .modifiers
                .iter()
                .any(|m| matches!(m, parser::haxe_ast::Modifier::Static));
            match &field.kind {
                ClassFieldKind::Function(func) => {
                    if func.name == "new" {
                        continue;
                    }
                    let method_name = self.context.intern_string(&func.name);
                    if self
                        .context
                        .symbol_table
                        .lookup_symbol(abstract_scope, method_name)
                        .is_some()
                    {
                        continue;
                    }
                    let sym = self
                        .context
                        .symbol_table
                        .create_function_in_scope(method_name, abstract_scope);
                    if let Some(scope) = self.context.scope_tree.get_scope_mut(abstract_scope) {
                        scope.add_symbol(sym, method_name);
                    }
                    if is_static {
                        self.context
                            .symbol_table
                            .add_symbol_flags(sym, crate::tast::symbols::SymbolFlags::STATIC);
                    }
                    let param_types: Vec<TypeId> = func
                        .params
                        .iter()
                        .map(|p| {
                            if let Some(ref type_hint) = p.type_hint {
                                self.lower_type(type_hint).unwrap_or_else(|_| {
                                    self.context.type_table.borrow().dynamic_type()
                                })
                            } else {
                                self.context.type_table.borrow().dynamic_type()
                            }
                        })
                        .collect();
                    let return_type = if let Some(ref ret_type) = func.return_type {
                        self.lower_type(ret_type)
                            .unwrap_or_else(|_| self.context.type_table.borrow().dynamic_type())
                    } else {
                        self.context.type_table.borrow().dynamic_type()
                    };
                    let function_type = self
                        .context
                        .type_table
                        .borrow_mut()
                        .create_function_type(param_types, return_type);
                    self.context
                        .symbol_table
                        .update_symbol_type(sym, function_type);
                    // Register in class_methods so static (Atomic.of) and instance
                    // (cell.asPtr()) calls resolve to this typed symbol via
                    // resolve_class_method_symbol Strategy 1.
                    if let Some(methods) = self.class_methods.get_mut(&abstract_symbol) {
                        methods.push((method_name, sym, is_static));
                    }
                }
                ClassFieldKind::Var {
                    name, type_hint, ..
                }
                | ClassFieldKind::Final {
                    name, type_hint, ..
                }
                | ClassFieldKind::Property {
                    name, type_hint, ..
                } => {
                    let member_name = self.context.intern_string(name);
                    if self
                        .context
                        .symbol_table
                        .lookup_symbol(abstract_scope, member_name)
                        .is_some()
                    {
                        continue;
                    }
                    let member_type = if let Some(th) = type_hint {
                        self.lower_type(th)
                            .unwrap_or_else(|_| self.context.type_table.borrow().dynamic_type())
                    } else {
                        self.context.type_table.borrow().dynamic_type()
                    };
                    let tracked_symbol =
                        self.class_fields.get(&abstract_symbol).and_then(|fields| {
                            fields
                                .iter()
                                .find(|(name, _, _)| *name == member_name)
                                .map(|(_, symbol, _)| *symbol)
                        });
                    let sym = tracked_symbol
                        .unwrap_or_else(|| self.context.symbol_table.create_variable(member_name));
                    self.context
                        .symbol_table
                        .update_symbol_type(sym, member_type);
                    if let Some(s) = self.context.symbol_table.get_symbol_mut(sym) {
                        s.kind = crate::tast::SymbolKind::Field;
                    }
                    let effective_static = is_static || abstract_decl.is_enum_abstract;
                    if effective_static {
                        self.context
                            .symbol_table
                            .add_symbol_flags(sym, crate::tast::symbols::SymbolFlags::STATIC);
                    }
                    if tracked_symbol.is_none() {
                        self.class_fields
                            .get_mut(&abstract_symbol)
                            .expect("abstract field map was initialized")
                            .push((member_name, sym, effective_static));
                    }
                    let qualified = format!("{}.{}", abstract_decl.name, name);
                    if let Some(symbol) = self.context.symbol_table.get_symbol_mut(sym) {
                        symbol.qualified_name =
                            Some(self.context.string_interner.intern(&qualified));
                    }
                    if let Some(scope) = self.context.scope_tree.get_scope_mut(abstract_scope) {
                        scope.add_symbol(sym, member_name);
                    }
                    // Enum-abstract constants are reachable bare (`eq(1, Red)`)
                    // as well as qualified (`Color.Red`), so also alias them into
                    // the module scope for bare-name resolution.
                    if abstract_decl.is_enum_abstract {
                        self.context.symbol_table.add_symbol_alias(
                            sym,
                            ScopeId::first(),
                            member_name,
                        );
                        let root = self
                            .context
                            .scope_tree
                            .get_scope_mut(ScopeId::first())
                            .expect("Root scope should exist");
                        if !root.has_symbol(member_name) {
                            root.add_symbol(sym, member_name);
                        }
                    }
                }
            }
        }

        // Process fields - separate fields, methods, and constructors
        let mut fields = Vec::with_capacity(abstract_decl.fields.len());
        let mut methods = Vec::with_capacity(abstract_decl.fields.len());
        let mut constructors = Vec::with_capacity(2); // Most abstracts have 0-2 constructors

        // Enum-abstract constants without an explicit initializer get an auto
        // value derived from the underlying type: Int/Float increment, String
        // uses the field name, Bool alternates false/true.
        let underlying_kind = underlying_type.and_then(|mut type_id| {
            let type_table = self.context.type_table.borrow();
            for _ in 0..16 {
                let kind = type_table.get(type_id)?.kind.clone();
                match kind {
                    crate::tast::TypeKind::Abstract {
                        underlying: Some(next),
                        ..
                    }
                    | crate::tast::TypeKind::TypeAlias {
                        target_type: next, ..
                    } => type_id = next,
                    other => return Some(other),
                }
            }
            None
        });
        let mut next_int: i64 = 0;
        let mut next_bool = false;

        for field in &abstract_decl.fields {
            match &field.kind {
                ClassFieldKind::Function(func) => {
                    if func.name == "new" {
                        // Constructor
                        match self.lower_function_from_field(field, func) {
                            Ok(typed_function) => {
                                constructors.push(typed_function);
                            }
                            Err(e) => self.context.add_error(e),
                        }
                    } else {
                        // Haxe permits writing to an abstract's `this` only from an
                        // inline function -- constructors excepted, which may assign
                        // it whether or not they are inline. Verified against Haxe
                        // 4.3.6, which rejects the non-inline method form with the
                        // message reproduced below and accepts `public function
                        // new(i) this = i;` unchanged.
                        //
                        // Without this the write is simply discarded: the receiver is
                        // passed by value, the body mutates a copy, and the caller
                        // sees its original value with no diagnostic at all.
                        let is_inline = field
                            .modifiers
                            .iter()
                            .any(|m| matches!(m, parser::Modifier::Inline));
                        if !is_inline {
                            if let Some(body) = &func.body {
                                if let Some(span) = Self::find_this_write(body) {
                                    self.context.add_error(LoweringError::SemanticError {
                                        message: "Abstract 'this' value can only \
                                                      be modified inside an inline \
                                                      function"
                                            .to_string(),
                                        location: self.context.create_location_from_span(span),
                                    });
                                }
                            }
                        }
                        // Regular method
                        match self.lower_function_from_field(field, func) {
                            Ok(typed_function) => {
                                methods.push(typed_function);
                            }
                            Err(e) => self.context.add_error(e),
                        }
                    }
                }
                _ => {
                    // Handle regular fields (var, final, property)
                    let member_name = match &field.kind {
                        ClassFieldKind::Var { name, .. }
                        | ClassFieldKind::Final { name, .. }
                        | ClassFieldKind::Property { name, .. } => {
                            Some(self.context.intern_string(name))
                        }
                        ClassFieldKind::Function(_) => None,
                    };
                    let pre_registered_symbol = member_name.and_then(|name| {
                        self.class_fields.get(&abstract_symbol).and_then(|fields| {
                            fields
                                .iter()
                                .find(|(field_name, _, _)| *field_name == name)
                                .map(|(_, symbol, _)| *symbol)
                        })
                    });
                    match self.lower_field_with_symbol(field, pre_registered_symbol) {
                        Ok(mut typed_field) => {
                            // For enum abstracts, all var fields are implicitly static
                            if abstract_decl.is_enum_abstract && !typed_field.is_static {
                                typed_field.is_static = true;
                                self.context.symbol_table.add_symbol_flags(
                                    typed_field.symbol_id,
                                    crate::tast::symbols::SymbolFlags::STATIC,
                                );
                                // Also update class_fields tracking
                                if let Some(field_list) =
                                    self.class_fields.get_mut(&abstract_symbol)
                                {
                                    if let Some(entry) = field_list
                                        .iter_mut()
                                        .find(|(_, sym, _)| *sym == typed_field.symbol_id)
                                    {
                                        entry.2 = true;
                                    }
                                }
                            }
                            if abstract_decl.is_enum_abstract {
                                if let Some(ref init) = typed_field.initializer {
                                    // Track an explicit Int constant so the next
                                    // auto-valued constant continues from it.
                                    if let TypedExpressionKind::Literal {
                                        value: LiteralValue::Int(i),
                                    } = &init.kind
                                    {
                                        next_int = i + 1;
                                    }
                                } else if let Some(auto_kind) = match &underlying_kind {
                                    Some(crate::tast::TypeKind::Int) => {
                                        let v = next_int;
                                        next_int += 1;
                                        Some(ExprKind::Int(v))
                                    }
                                    Some(crate::tast::TypeKind::Float) => {
                                        let v = next_int as f64;
                                        next_int += 1;
                                        Some(ExprKind::Float(v))
                                    }
                                    Some(crate::tast::TypeKind::String) => {
                                        let name = self
                                            .context
                                            .string_interner
                                            .get(typed_field.name)
                                            .unwrap_or("")
                                            .to_string();
                                        Some(ExprKind::String(name))
                                    }
                                    Some(crate::tast::TypeKind::Bool) => {
                                        let v = next_bool;
                                        next_bool = !next_bool;
                                        Some(ExprKind::Bool(v))
                                    }
                                    _ => None,
                                } {
                                    let auto_expr = Expr {
                                        kind: auto_kind,
                                        span: field.span,
                                    };
                                    match self.lower_expression(&auto_expr) {
                                        Ok(t) => typed_field.initializer = Some(t),
                                        Err(e) => self.context.add_error(e),
                                    }
                                }
                            }
                            fields.push(typed_field);
                        }
                        Err(e) => self.context.add_error(e),
                    }
                }
            }
        }

        // Pop abstract from class context stack
        self.context.class_context_stack.pop();

        self.context.pop_type_parameters();
        self.context.exit_scope();

        let typed_abstract = crate::tast::node::TypedAbstract {
            symbol_id: abstract_symbol,
            name: abstract_name,
            underlying_type,
            type_parameters: type_params,
            fields,
            methods,
            constructors,
            from_types,
            to_types,
            forward_fields,
            is_enum_abstract: abstract_decl.is_enum_abstract,
            visibility: self.lower_access(&abstract_decl.access),
            source_location: self.context.create_location_from_span(abstract_decl.span),
        };

        Ok(TypedDeclaration::Abstract(typed_abstract))
    }
}
