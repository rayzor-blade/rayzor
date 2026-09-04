//! Expression lowering: the dispatcher and the simple forms.

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
    /// A wildcard import naming a TYPE brings that type's statics into scope.
    ///
    /// Haxe spells two different things the same way. When the last segment of
    /// `import a.b.*` is a package, the wildcard imports the types in it; when
    /// it is a type, it imports that type's static fields. Only the package
    /// reading was implemented, so `import utest.Assert.*` was resolved as the
    /// package `utest` and a bare `isTrue(...)` became a search for the type
    /// `utest.isTrue`, which does not exist -- the call then failed as an
    /// unknown name. This adds the field reading; the package one is unchanged
    /// and still handled by the namespace resolver.
    ///
    /// Both the enclosing scope and the module scope are searched, because an
    /// import written at the top of a file belongs to the latter.
    fn resolve_wildcard_static_import(&self, name: InternedString) -> Option<SymbolId> {
        let probe = crate::debug_flags::wildcard_log();
        // The whole chain, not just the innermost and outermost: an import
        // written at the top of a file is registered on that file's scope,
        // which is neither the body's scope nor the root.
        let mut scope = self.context.current_scope;
        loop {
            for import in self.context.import_resolver.get_imports(scope) {
                if !import.is_wildcard || import.exclusions.contains(&name) {
                    continue;
                }
                // A root-package owner has an empty package path, which the
                // namespace resolver does not key on -- `import Helper.*` for a
                // class in no package resolves to nothing there. Fall back to
                // the scope chain, which is where such a type is registered.
                let owner = self
                    .context
                    .namespace_resolver
                    .lookup_symbol(&import.package_path)
                    .or_else(|| {
                        if import.package_path.package.is_empty() {
                            self.resolve_symbol_in_scope_hierarchy(import.package_path.name)
                        } else {
                            None
                        }
                    });
                if probe {
                    let owner_name = self
                        .context
                        .string_interner
                        .get(import.package_path.name)
                        .unwrap_or("<?>")
                        .to_string();
                    eprintln!(
                        "[wildcard] scope={:?} owner_name={} pkg_len={} resolved={:?} fields={:?}",
                        scope,
                        owner_name,
                        import.package_path.package.len(),
                        owner,
                        owner.and_then(|o| self.class_fields.get(&o).map(|f| f.len())),
                    );
                }
                let Some(owner) = owner else { continue };
                if let Some(fields) = self.class_fields.get(&owner) {
                    if let Some((_, field_symbol, _)) = fields
                        .iter()
                        .find(|(field_name, _, is_static)| *field_name == name && *is_static)
                    {
                        return Some(*field_symbol);
                    }
                }
            }
            match self
                .context
                .scope_tree
                .get_scope(scope)
                .and_then(|sc| sc.parent_id)
            {
                Some(parent) => scope = parent,
                None => break,
            }
        }
        if probe {
            eprintln!("[wildcard] no static import matched {:?}", name);
        }
        None
    }

    /// Peek a simple AST expression's type WITHOUT lowering it — for binding an
    /// inferred empty-array element type from the first push/index-assign before
    /// that statement is lowered. Handles literals, in-scope identifiers, and
    /// arithmetic; `None` means "leave Array<Dynamic>" (no regression).
    fn peek_ast_expr_type(&mut self, e: &Expr) -> Option<TypeId> {
        match &e.kind {
            ExprKind::Float(_) => Some(self.context.type_table.borrow().float_type()),
            ExprKind::Int(_) => Some(self.context.type_table.borrow().int_type()),
            ExprKind::Bool(_) => Some(self.context.type_table.borrow().bool_type()),
            ExprKind::String(_) | ExprKind::StringInterpolation(_) => {
                Some(self.context.type_table.borrow().string_type())
            }
            ExprKind::Ident(name) => {
                let interned = self.context.intern_string(name);
                let sym = self.resolve_symbol_in_scope_hierarchy(interned)?;
                self.context.symbol_table.get_symbol(sym).map(|s| s.type_id)
            }
            ExprKind::Binary { left, right, .. } => {
                let lt = self.peek_ast_expr_type(left);
                let rt = self.peek_ast_expr_type(right);
                let ftype = self.context.type_table.borrow().float_type();
                if lt == Some(ftype) || rt == Some(ftype) {
                    Some(ftype)
                } else {
                    lt.or(rt)
                }
            }
            _ => None,
        }
    }

    /// Lower an expression as a statement
    pub(crate) fn lower_expression_as_statement(
        &mut self,
        expr: &Expr,
    ) -> LoweringResult<TypedStatement> {
        let typed_expr = self.lower_expression(expr)?;
        Ok(TypedStatement::Expression {
            expression: typed_expr,
            source_location: self.context.create_location(),
        })
    }

    /// Lower an expression
    pub(crate) fn lower_expression(
        &mut self,
        expression: &Expr,
    ) -> LoweringResult<TypedExpression> {
        let kind = match &expression.kind {
            ExprKind::Int(value) => TypedExpressionKind::Literal {
                value: LiteralValue::Int(*value),
            },
            ExprKind::Float(value) => TypedExpressionKind::Literal {
                value: LiteralValue::Float(*value),
            },
            ExprKind::String(value) => TypedExpressionKind::Literal {
                value: LiteralValue::String(value.clone()),
            },
            ExprKind::Bool(value) => TypedExpressionKind::Literal {
                value: LiteralValue::Bool(*value),
            },
            ExprKind::Null => TypedExpressionKind::Null,
            ExprKind::Regex { pattern, flags } => TypedExpressionKind::Literal {
                value: LiteralValue::RegexWithFlags {
                    pattern: pattern.clone(),
                    flags: flags.clone(),
                },
            },
            ExprKind::Ident(name) => {
                let id_name = self.context.intern_string(name);
                let prefer = self
                    .expected_arg_type_stack
                    .last()
                    .copied()
                    .flatten()
                    .or(self.context.expected_return_type);
                let preferred_abstract_field = prefer.and_then(|expected_ty| {
                    let expected_abstract = {
                        let type_table = self.context.type_table.borrow();
                        type_table.get(expected_ty).and_then(|ty| match &ty.kind {
                            crate::tast::core::TypeKind::Abstract { symbol_id, .. } => {
                                Some(*symbol_id)
                            }
                            _ => None,
                        })
                    }?;
                    self.class_fields
                        .get(&expected_abstract)
                        .and_then(|fields| {
                            fields
                                .iter()
                                .find(|(field_name, _, is_static)| {
                                    *field_name == id_name && *is_static
                                })
                                .map(|(_, field_symbol, _)| *field_symbol)
                        })
                });
                // Need to resolve symbol by walking up the scope hierarchy
                let mut symbol_id = match self
                    .resolve_symbol_in_scope_hierarchy(id_name)
                    .or(preferred_abstract_field)
                    // Last, so a local, a parameter or a field still shadows it.
                    .or_else(|| self.resolve_wildcard_static_import(id_name))
                {
                    Some(s) => s,
                    None => {
                        // Abstract-method implicit `this`: in an abstract, `this` IS the
                        // underlying value, so a bare member name that isn't a local/param
                        // is `this.<name>` — which dispatches to the underlying type's field
                        // or the abstract's property getter. Synthesizing the field access
                        // (rather than resolving the bare name to a storage-less property
                        // symbol) reads the right slot AND is declaration-order-independent.
                        // Without this, e.g. haxe.Int64.copy() reading bare `high`/`low`
                        // (properties declared ~400 lines below) failed "Cannot find name".
                        let in_abstract = self
                            .context
                            .class_context_stack
                            .last()
                            .and_then(|s| self.context.symbol_table.get_symbol(*s))
                            .map(|s| s.kind == crate::tast::symbols::SymbolKind::Abstract)
                            .unwrap_or(false);
                        if in_abstract {
                            let this_expr = Expr {
                                kind: ExprKind::This,
                                span: expression.span,
                            };
                            let field_expr = Expr {
                                kind: ExprKind::Field {
                                    expr: Box::new(this_expr),
                                    field: name.clone(),
                                    is_optional: false,
                                },
                                span: expression.span,
                            };
                            return self.lower_expression(&field_expr);
                        }
                        if std::env::var_os("RAYZOR_RESOLVE_TRACE").is_some() {
                            eprintln!(
                                "[RESOLVE_TRACE] unresolved '{}' in scope {:?}; table-wide same-name symbols:",
                                name, self.context.current_scope
                            );
                            for sym in self.context.symbol_table.find_symbols(|s| {
                                self.context
                                    .string_interner
                                    .get(s.name)
                                    .is_some_and(|n| n == name.as_str())
                            }) {
                                eprintln!(
                                    "[RESOLVE_TRACE]   id={:?} kind={:?} scope={:?} type={:?}",
                                    sym.id, sym.kind, sym.scope_id, sym.type_id
                                );
                            }
                        }
                        return Err(LoweringError::UnresolvedSymbol {
                            name: name.clone(),
                            location: self.context.create_location_from_span(expression.span),
                        });
                    }
                };

                // Enum-abstract constants share the module namespace with
                // ordinary types. If an unrelated class already owns the bare
                // name (for example `unit.Bar` versus `Foo.Bar`), the root
                // scope cannot represent Haxe's expected-type disambiguation.
                // Redirect to the expected abstract's field before applying
                // ordinary enum-variant disambiguation below.
                if let Some(expected_abstract) = prefer.and_then(|expected_ty| {
                    let type_table = self.context.type_table.borrow();
                    type_table.get(expected_ty).and_then(|ty| match &ty.kind {
                        crate::tast::core::TypeKind::Abstract { symbol_id, .. } => Some(*symbol_id),
                        _ => None,
                    })
                }) {
                    if let Some(field_symbol) =
                        self.class_fields
                            .get(&expected_abstract)
                            .and_then(|fields| {
                                fields
                                    .iter()
                                    .find(|(field_name, _, is_static)| {
                                        *field_name == id_name && *is_static
                                    })
                                    .map(|(_, field_symbol, _)| *field_symbol)
                            })
                    {
                        symbol_id = field_symbol;
                    }
                }

                // Enum-variant disambiguation. If the scope-walk found an enum
                // variant but the expected arg type is a *different* enum,
                // re-resolve to that enum's variant of the same name when one
                // exists. This catches the cross-file shadowing where, e.g.,
                // `Tensor.zeros([…], F32)` in a file that has both `DType.F32`
                // and `MetaValue.F32` in scope would otherwise pick the first
                // one the scope walk found (which is often the wrong one).
                // See bugs_dtype_enum_cross_file_pointer.
                //
                // Falls back to `expected_return_type` (the enclosing
                // function's declared return type) when there's no
                // in-progress call argument — covers `return F32;` inside a
                // function declared `:DType`, which otherwise hits the same
                // collision (confirmed: `nue`'s `inferDType():DType` with
                // `return F32;` resolved to `MetaValue.F32`, a boxed variant
                // from an unrelated 13-variant enum, instead of `DType.F32`).
                if let Some(expected_ty) = prefer {
                    let needs_reresolve = {
                        let sym = self.context.symbol_table.get_symbol(symbol_id);
                        let sym_is_variant = sym
                            .map(|s| s.kind == crate::tast::symbols::SymbolKind::EnumVariant)
                            .unwrap_or(false);
                        if !sym_is_variant {
                            false
                        } else {
                            // Look at expected_ty's TypeKind: if it's an Enum E,
                            // and the resolved variant's parent enum != E, we want
                            // to look up the variant within E's scope.
                            let type_table = self.context.type_table.borrow();
                            let expected_enum_sym =
                                type_table.get(expected_ty).and_then(|t| match &t.kind {
                                    crate::tast::core::TypeKind::Enum { symbol_id, .. } => {
                                        Some(*symbol_id)
                                    }
                                    _ => None,
                                });
                            if let Some(expected_enum) = expected_enum_sym {
                                let resolved_parent_enum = self
                                    .context
                                    .symbol_table
                                    .find_parent_enum_for_constructor(symbol_id);
                                resolved_parent_enum.map_or(true, |p| p != expected_enum)
                            } else {
                                false
                            }
                        }
                    };
                    if needs_reresolve {
                        if let Some(expected_enum_sym) = {
                            let type_table = self.context.type_table.borrow();
                            type_table.get(expected_ty).and_then(|t| match &t.kind {
                                crate::tast::core::TypeKind::Enum { symbol_id, .. } => {
                                    Some(*symbol_id)
                                }
                                _ => None,
                            })
                        } {
                            if let Some(variants) = self
                                .context
                                .symbol_table
                                .get_enum_variants(expected_enum_sym)
                            {
                                for &variant_sym in variants {
                                    if let Some(vsym) =
                                        self.context.symbol_table.get_symbol(variant_sym)
                                    {
                                        if vsym.name == id_name {
                                            symbol_id = variant_sym;
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    // No expected type disambiguates this bare identifier. If it
                    // resolved to an enum VARIANT and a same-named variant exists
                    // on a DIFFERENT enum, the scope walk's pick is a guess
                    // (whichever module registered first) — hard-fail instead
                    // of silently building the wrong enum's value.
                    let is_variant = self
                        .context
                        .symbol_table
                        .get_symbol(symbol_id)
                        .map(|s| s.kind == crate::tast::symbols::SymbolKind::EnumVariant)
                        .unwrap_or(false);
                    if is_variant {
                        let variant_name = self
                            .context
                            .symbol_table
                            .get_symbol(symbol_id)
                            .map(|s| s.name);
                        let mut parent_enum_names: Vec<String> = Vec::new();
                        if let Some(vname) = variant_name {
                            // A constructor bound only under its qualified name
                            // is not reachable as a bare `Some` here, so it is
                            // not a candidate the scope walk could have picked.
                            let same_named = self.context.symbol_table.find_symbols(|s| {
                                s.kind == crate::tast::symbols::SymbolKind::EnumVariant
                                    && s.name == vname
                                    && !s
                                        .flags
                                        .contains(crate::tast::symbols::SymbolFlags::QUALIFIED_ONLY)
                            });
                            for v in same_named {
                                let parent = self
                                    .context
                                    .symbol_table
                                    .find_parent_enum_for_constructor(v.id);
                                let Some(parent) = parent else { continue };
                                let Some(psym) = self.context.symbol_table.get_symbol(parent)
                                else {
                                    continue;
                                };
                                // Identity = qualified name when present, else bare
                                // name: the same source enum re-registered across
                                // contexts must not count twice.
                                let pname = psym
                                    .qualified_name
                                    .or(Some(psym.name))
                                    .and_then(|n| self.context.string_interner.get(n))
                                    .unwrap_or("?")
                                    .to_string();
                                if !parent_enum_names.contains(&pname) {
                                    parent_enum_names.push(pname);
                                }
                            }
                        }
                        if parent_enum_names.len() > 1 {
                            let candidates = parent_enum_names
                                .iter()
                                .map(|p| format!("{}.{}", p, name))
                                .collect::<Vec<_>>()
                                .join(", ");
                            return Err(LoweringError::AmbiguousSymbol {
                                message: format!(
                                    "E0804: ambiguous enum variant `{}`: matches {} and no expected type disambiguates. Qualify it with the enum name",
                                    name, candidates
                                ),
                                location: self.context.create_location_from_span(expression.span),
                            });
                        }
                    }
                }

                // Check if this symbol is an instance VAR field of the current class
                // (not a method). If so, we need to create a FieldAccess with implicit
                // `this` receiver: `i = value` → `this.i = value`.
                let is_instance_field =
                    if let Some(class_symbol) = self.context.class_context_stack.last() {
                        let is_in_fields = self
                            .class_fields
                            .get(class_symbol)
                            .map(|fields| {
                                fields.iter().any(|(_, field_sym, is_static)| {
                                    *field_sym == symbol_id && !is_static
                                })
                            })
                            .unwrap_or(false);
                        // Exclude methods — they are handled by the call resolution path
                        let is_method = self
                            .class_methods
                            .get(class_symbol)
                            .map(|methods| {
                                methods
                                    .iter()
                                    .any(|(_, method_sym, _)| *method_sym == symbol_id)
                            })
                            .unwrap_or(false);
                        is_in_fields && !is_method
                    } else {
                        false
                    };

                if is_instance_field && !self.in_static_method {
                    // Create implicit `this` receiver for instance field access
                    // in non-static methods/constructors.
                    let this_name = self.context.intern_string("this");
                    let this_symbol = self
                        .resolve_symbol_in_scope_hierarchy(this_name)
                        .unwrap_or_else(|| self.context.symbol_table.create_variable(this_name));
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
                    TypedExpressionKind::FieldAccess {
                        object: Box::new(receiver),
                        field_symbol: symbol_id,
                        is_optional: false,
                    }
                } else {
                    TypedExpressionKind::Variable { symbol_id }
                }
            }
            ExprKind::Binary { left, op, right } => {
                // Special handling for `is` operator: `expr is Type`
                if matches!(op, BinaryOp::Is) {
                    let left_expr = self.lower_expression(left)?;
                    // The right side is a type name parsed as an expression (Ident)
                    // Extract the type name and resolve it
                    // Build a TypePath from the right-hand expression
                    let type_path = match &right.kind {
                        ExprKind::Ident(name) => parser::TypePath {
                            package: vec![],
                            name: name.clone(),
                            sub: None,
                        },
                        ExprKind::Field { expr, field, .. } => {
                            // Handle qualified names like `pack.Type`
                            let mut parts = Vec::new();
                            fn collect_parts(e: &Expr, parts: &mut Vec<String>) {
                                match &e.kind {
                                    ExprKind::Ident(n) => parts.push(n.clone()),
                                    ExprKind::Field { expr, field, .. } => {
                                        collect_parts(expr, parts);
                                        parts.push(field.clone());
                                    }
                                    _ => {}
                                }
                            }
                            collect_parts(expr, &mut parts);
                            parser::TypePath {
                                package: parts,
                                name: field.clone(),
                                sub: None,
                            }
                        }
                        _ => {
                            return Err(LoweringError::UnresolvedType {
                                type_name: format!("{:?}", right.kind),
                                location: self.context.create_location_from_span(expression.span),
                            });
                        }
                    };
                    // Resolve via full type resolution (handles user classes, imports, namespaces)
                    let check_type = self.resolve_type_path(&type_path)?;
                    TypedExpressionKind::Is {
                        expression: Box::new(left_expr),
                        check_type,
                    }
                } else {
                    let left_expr = self.lower_expression(left)?;
                    // For ==/!= the LHS's static type is the expected type of
                    // the RHS — disambiguates a bare enum-variant comparand
                    // (`v == Red` with `v:ColorA`).
                    let is_eq = matches!(op, BinaryOp::Eq | BinaryOp::NotEq);
                    if is_eq {
                        self.expected_arg_type_stack.push(Some(left_expr.expr_type));
                    }
                    let right_result = self.lower_expression(right);
                    if is_eq {
                        self.expected_arg_type_stack.pop();
                    }
                    let right_expr = right_result?;
                    let typed_op = self.lower_binary_operator(op)?;

                    TypedExpressionKind::BinaryOp {
                        left: Box::new(left_expr),
                        operator: typed_op,
                        right: Box::new(right_expr),
                    }
                }
            }
            ExprKind::Unary { op, expr } => {
                let operand_expr = self.lower_expression(expr)?;
                let typed_op = self.lower_unary_operator(op)?;

                TypedExpressionKind::UnaryOp {
                    operator: typed_op,
                    operand: Box::new(operand_expr),
                }
            }
            ExprKind::Call { expr, args } => {
                // Monomorph rewrite: `arr.push(e)` on an untyped empty array binds
                // its element type from `e` (before the call lowers, so the
                // receiver resolves to the concrete Array<T>).
                if args.len() == 1 {
                    if let ExprKind::Field {
                        expr: recv, field, ..
                    } = &expr.kind
                    {
                        if field == "push" {
                            self.try_bind_inferred_array(recv, &args[0]);
                        }
                    }
                }
                return self.lower_call_expression(expression, expr, args);
            }
            ExprKind::Field {
                expr,
                field,
                is_optional,
            } => {
                // Char-literal magic: `'X'.code` inlines the character code at
                // compile time (documented in String.hx: `"x".code` "inline the
                // character code at compile time"). The receiver is a single-char
                // string literal; fold it to the Int constant here so it never
                // depends on resolving a `code` member on String (there is none).
                // Without this, StringTools/EReg etc. fail to compile under import
                // isolation (`'+'.code` falls through to String field resolution),
                // and `case 'X'.code:` switch patterns require a constant.
                if field == "code" {
                    if let ExprKind::String(s) = &expr.kind {
                        let mut chars = s.chars();
                        if let (Some(c), None) = (chars.next(), chars.next()) {
                            let int_expr = Expr {
                                kind: ExprKind::Int(c as i64),
                                span: expression.span,
                            };
                            return self.lower_expression(&int_expr);
                        }
                    }
                }
                return self.lower_field_expression(expression, expr, field, *is_optional);
            }
            ExprKind::Index { expr, index } => {
                let array_expr = self.lower_expression(expr)?;
                let index_expr = self.lower_expression(index)?;

                TypedExpressionKind::ArrayAccess {
                    array: Box::new(array_expr),
                    index: Box::new(index_expr),
                }
            }
            ExprKind::Assign { left, op, right } => {
                // Monomorph rewrite: `arr[i] = e` on an untyped empty array binds
                // its element type from `e` before the target lowers.
                if let ExprKind::Index { expr: recv, .. } = &left.kind {
                    self.try_bind_inferred_array(recv, right);
                }
                let target_expr = self.lower_expression(left)?;
                // Same `@:multiType` propagation as Var declarations: when
                // the RHS is a bare `new C()`, the LHS's static type seeds
                // the type args so e.g. `values = new Map()` against a
                // `Map<String, V>` field still routes through StringMap.
                // Without this the field assignment elides at MIR lowering
                // and the constructor body becomes a no-op.
                let value_expr = {
                    let prev_hint = self.context.expected_new_type_hint;
                    self.context.expected_new_type_hint = Some(target_expr.expr_type);
                    // The LHS's static type is also the expected type of the
                    // RHS — disambiguates a bare enum-variant RHS
                    // (`v = Red` with `v:ColorA`).
                    self.expected_arg_type_stack
                        .push(Some(target_expr.expr_type));
                    let result = self.lower_expression(right);
                    self.expected_arg_type_stack.pop();
                    self.context.expected_new_type_hint = prev_hint;
                    result?
                };

                match op {
                    parser::AssignOp::Assign => {
                        // Simple assignment: target = value
                        TypedExpressionKind::BinaryOp {
                            left: Box::new(target_expr),
                            operator: BinaryOperator::Assign,
                            right: Box::new(value_expr),
                        }
                    }
                    _ => {
                        // Compound assignment: target op= value
                        // This needs to be: target = target op value
                        let target_clone = target_expr.clone();

                        // Map compound assignment operators to their corresponding binary operators
                        let binary_op = match op {
                            parser::AssignOp::AddAssign => BinaryOperator::Add,
                            parser::AssignOp::SubAssign => BinaryOperator::Sub,
                            parser::AssignOp::MulAssign => BinaryOperator::Mul,
                            parser::AssignOp::DivAssign => BinaryOperator::Div,
                            parser::AssignOp::ModAssign => BinaryOperator::Mod,
                            parser::AssignOp::AndAssign => BinaryOperator::BitAnd,
                            parser::AssignOp::OrAssign => BinaryOperator::BitOr,
                            parser::AssignOp::XorAssign => BinaryOperator::BitXor,
                            parser::AssignOp::ShlAssign => BinaryOperator::Shl,
                            parser::AssignOp::ShrAssign => BinaryOperator::Shr,
                            parser::AssignOp::UshrAssign => BinaryOperator::Ushr,
                            parser::AssignOp::Assign => unreachable!(), // Handled above
                        };

                        // Create the binary operation: target op value
                        let binary_expr = TypedExpression {
                            expr_type: target_expr.expr_type,
                            kind: TypedExpressionKind::BinaryOp {
                                left: Box::new(target_clone),
                                operator: binary_op,
                                right: Box::new(value_expr),
                            },
                            usage: VariableUsage::Copy,
                            lifetime_id: crate::tast::LifetimeId::first(),
                            source_location: self.context.create_location(),
                            metadata: ExpressionMetadata::default(),
                        };

                        // Now assign the result back to target: target = (target op value)
                        TypedExpressionKind::BinaryOp {
                            left: Box::new(target_expr),
                            operator: BinaryOperator::Assign,
                            right: Box::new(binary_expr),
                        }
                    }
                }
            }
            ExprKind::New {
                type_path,
                params,
                args,
            } => {
                // Resolve the base class type from type_path.
                // `mut` because the `expected_new_type_hint` block below may
                // re-target construction to the concrete container when the
                // hint and base disagree (e.g. hint `StringMap<V>` from
                // `var m:Map<String,V>` annotation, base `Map` from
                // `new Map()` call site).
                let mut base_class_type_id = self.resolve_type_path(type_path)?;

                // Lower constructor arguments
                let arg_exprs = args
                    .iter()
                    .map(|arg| self.lower_expression(arg))
                    .collect::<Result<Vec<_>, _>>()?;

                // Lower type arguments from params
                let mut type_args = params
                    .iter()
                    .map(|param| self.lower_type(param))
                    .collect::<Result<Vec<_>, _>>()?;

                // If the call site omitted explicit `<...>` params (e.g.
                // `var m:Map<String,Int> = new Map();`), borrow them from
                // the surrounding `expected_new_type_hint` when the hint
                // is a generic instance of the same class. This is what
                // makes `@:multiType` resolution fire on bare `new Map()`
                // — without it the New site has zero type args and
                // `maybe_resolve_multitype_map` bails, leaving Map
                // unresolved and the whole containing function body
                // silently elided at MIR lowering.
                // Fill in `<...>` type args from `expected_new_type_hint`
                // when the call site omitted them. The hint comes from the
                // surrounding Var-declaration or Assign LHS type. We extract
                // (symbol, type_args) and copy them in when either:
                //   (a) hint and base classes match — straightforward
                //       `var m:Foo<Int> = new Foo()` propagation, or
                //   (b) base is the `Map` abstract and the hint is a
                //       Map-concrete (StringMap/IntMap/etc.) — the LHS
                //       annotation already underwent multiType resolution
                //       in `lower_type`, so the hint carries `[V]` (the
                //       concrete arity) but we want the abstract's
                //       `[K, V]` here so `maybe_resolve_multitype_map`
                //       below re-runs cleanly and supplies the right class
                //       name (`haxe.ds.StringMap`). Just-using-the-hint
                //       loses that name and the construction would fall
                //       through to a generic allocator.
                if type_args.is_empty() {
                    if let Some(hint_ty) = self.context.expected_new_type_hint {
                        let tt = self.context.type_table.borrow();
                        fn extract_sym_and_args(
                            tt: &crate::tast::core::TypeTable,
                            ty: TypeId,
                        ) -> Option<(SymbolId, Vec<TypeId>)> {
                            let info = tt.get(ty)?;
                            match &info.kind {
                                crate::tast::core::TypeKind::Class {
                                    symbol_id,
                                    type_args,
                                }
                                | crate::tast::core::TypeKind::Abstract {
                                    symbol_id,
                                    type_args,
                                    ..
                                }
                                | crate::tast::core::TypeKind::Interface {
                                    symbol_id,
                                    type_args,
                                    ..
                                } => Some((*symbol_id, type_args.clone())),
                                crate::tast::core::TypeKind::GenericInstance {
                                    base_type,
                                    type_args,
                                    ..
                                } => {
                                    let base_sym = tt.get(*base_type).and_then(|b| match &b.kind {
                                        crate::tast::core::TypeKind::Class {
                                            symbol_id, ..
                                        }
                                        | crate::tast::core::TypeKind::Abstract {
                                            symbol_id, ..
                                        }
                                        | crate::tast::core::TypeKind::Interface {
                                            symbol_id,
                                            ..
                                        } => Some(*symbol_id),
                                        _ => None,
                                    });
                                    base_sym.map(|s| (s, type_args.clone()))
                                }
                                _ => None,
                            }
                        }
                        let hint_parts = extract_sym_and_args(&tt, hint_ty);
                        let base_sym =
                            extract_sym_and_args(&tt, base_class_type_id).map(|(s, _)| s);
                        let base_is_abstract_map = matches!(
                            tt.get(base_class_type_id).map(|t| &t.kind),
                            Some(crate::tast::core::TypeKind::Abstract { .. })
                        ) && type_path.name == "Map";
                        // Is the hint itself a Map-family type? Cross-module,
                        // the field annotation `Map<K, V>` may NOT have been
                        // resolved to a concrete (StringMap / …) — the hint
                        // arrives as the *abstract* `Map` (or its TypeAlias)
                        // still carrying `[K, V]`, and `base_class_type_id`
                        // for `new Map()` may resolve to a TypeAlias rather
                        // than the Abstract, so `base_is_abstract_map` is
                        // false. Recognising the Map family on the hint side
                        // lets us recover `[K, V]` regardless of how the base
                        // resolved. Without this the New site has zero args,
                        // multiType resolution bails, and the construction
                        // falls through to `BalancedTree` (which fails to
                        // monomorphize cross-module → trap stub → SIGILL).
                        let hint_map_kind = hint_parts.as_ref().and_then(|(s, _)| {
                            self.context
                                .symbol_table
                                .get_symbol(*s)
                                .and_then(|sym| self.context.string_interner.get(sym.name))
                                .map(|n| n.to_string())
                        });
                        let constructing_map = type_path.name == "Map";
                        drop(tt);
                        if let Some((hint_sym, hint_args)) = hint_parts {
                            let hint_name = hint_map_kind.as_deref().unwrap_or("");
                            if Some(hint_sym) == base_sym && !hint_args.is_empty() {
                                type_args = hint_args;
                            } else if base_is_abstract_map && hint_args.len() == 1 {
                                // `Map<K, V>` was resolved to a concrete
                                // with arity 1 (StringMap<V> / IntMap<V>) —
                                // we lost K. There's no general way to
                                // recover it from the concrete, but we can
                                // re-derive from the concrete's class name.
                                if let Some(k_ty) =
                                    self.recover_map_key_type_from_concrete(hint_sym)
                                {
                                    type_args = vec![k_ty, hint_args[0]];
                                }
                            } else if base_is_abstract_map && hint_args.len() == 2 {
                                // `Map<K, V>` resolved to `ObjectMap<K, V>`
                                // / `EnumValueMap<K, V>` — both keep arity 2.
                                type_args = hint_args;
                            } else if constructing_map
                                && matches!(hint_name, "Map" | "EnumValueMap" | "ObjectMap")
                                && hint_args.len() == 2
                            {
                                // Cross-module fallback: hint is the abstract
                                // `Map<K, V>` (or an arity-2 concrete) still
                                // carrying both args. Use them directly so
                                // multiType resolution below picks the right
                                // concrete container from K's kind.
                                type_args = hint_args;
                            } else if constructing_map
                                && matches!(hint_name, "StringMap" | "IntMap")
                                && hint_args.len() == 1
                            {
                                // Cross-module: hint is an arity-1 concrete
                                // (StringMap<V> / IntMap<V>); recover K from
                                // the concrete's class identity.
                                if let Some(k_ty) =
                                    self.recover_map_key_type_from_concrete(hint_sym)
                                {
                                    type_args = vec![k_ty, hint_args[0]];
                                }
                            }
                        }
                    }
                }

                // Fill in any argument the call site left out from the class's
                // declared default: `class Foo<T = String>` means `new Foo()`
                // binds T to String. Without it the parameter stays unresolved
                // and its values render as `<unknown type N>` -- wrong, and
                // silent.
                if let Some(base_sym) = {
                    let tt = self.context.type_table.borrow();
                    tt.get(base_class_type_id).and_then(|t| match &t.kind {
                        crate::tast::core::TypeKind::Class { symbol_id, .. } => Some(*symbol_id),
                        _ => None,
                    })
                } {
                    if let Some(defaults) = self
                        .context
                        .symbol_table
                        .get_class_type_param_defaults(base_sym)
                        .cloned()
                    {
                        // Only trailing positions can default, which is also the
                        // only place Haxe allows one.
                        for d in defaults.iter().skip(type_args.len()) {
                            match d {
                                Some(ty) => type_args.push(*ty),
                                None => break,
                            }
                        }
                    }
                }

                // If type arguments are provided, create an instantiated type
                // e.g., new Array<Thread<Int>>() should have type Array<Thread<Int>>, not just Array
                let actual_class_type = if !type_args.is_empty() {
                    let symbol_id_opt = {
                        let type_table = self.context.type_table.borrow();
                        if let Some(base_type_info) = type_table.get(base_class_type_id) {
                            match &base_type_info.kind {
                                crate::tast::core::TypeKind::Class { symbol_id, .. } => {
                                    Some((*symbol_id, false)) // (symbol_id, is_array)
                                }
                                crate::tast::core::TypeKind::Array { .. } => {
                                    Some((SymbolId::invalid(), true)) // Mark as array type
                                }
                                _ => None,
                            }
                        } else {
                            None
                        }
                    };

                    if let Some((symbol_id, is_array)) = symbol_id_opt {
                        if is_array && type_args.len() == 1 {
                            self.context
                                .type_table
                                .borrow_mut()
                                .create_array_type(type_args[0])
                        } else if !is_array {
                            self.context
                                .type_table
                                .borrow_mut()
                                .create_class_type(symbol_id, type_args.clone())
                        } else {
                            base_class_type_id
                        }
                    } else {
                        base_class_type_id
                    }
                } else {
                    // No explicit type args — try to infer from constructor argument types
                    self.infer_type_args_from_constructor(base_class_type_id, &arg_exprs)
                        .unwrap_or(base_class_type_id)
                };

                // `@:multiType` abstract resolution. `Map<K, V>` in haxe-std is
                // declared as `@:multiType` and selects a concrete underlying
                // (StringMap / IntMap / EnumValueMap / ObjectMap) per K. We
                // don't parse `@:multiType` metadata yet, but the rule for
                // `haxe.ds.Map` is fixed and load-bearing — `new Map<String,
                // V>()` must construct a `StringMap<V>` (extern, routed
                // through `haxe_stringmap_*` runtime), not fall through to
                // `BalancedTree`. Anything else returns garbage at runtime
                // with no diagnostic (the typechecker silently picks the
                // first `IMap` implementer when no rule is honored). See
                // memory/feedback_no_silent_dispatch_fallthrough.md.
                let (final_class_type, final_type_args, final_class_name) =
                    self.maybe_resolve_multitype_map(actual_class_type, type_args, type_path);

                let class_name_str = match final_class_name {
                    Some(name) => name,
                    None if type_path.package.is_empty() => type_path.name.clone(),
                    None => format!("{}.{}", type_path.package.join("."), type_path.name),
                };
                let interned_class_name = self.context.string_interner.intern(&class_name_str);

                TypedExpressionKind::New {
                    class_type: final_class_type,
                    arguments: arg_exprs,
                    type_arguments: final_type_args,
                    class_name: Some(interned_class_name),
                }
            }
            // Cast doesn't exist in ExprKind, remove this variant
            ExprKind::Ternary {
                cond,
                then_expr,
                else_expr,
            } => {
                let cond_expr = self.lower_expression(cond)?;
                let then_expression = self.lower_expression(then_expr)?;
                let else_expression = Some(Box::new(self.lower_expression(else_expr)?));

                TypedExpressionKind::Conditional {
                    condition: Box::new(cond_expr),
                    then_expr: Box::new(then_expression),
                    else_expr: else_expression,
                }
            }
            ExprKind::Block(block_elements) => {
                // Handle block expressions with error recovery
                let mut statements = Vec::new();
                let block_scope = self.context.enter_scope(ScopeKind::Block);

                for elem in block_elements {
                    match elem {
                        parser::BlockElement::Expr(expr) => {
                            // Check if this is a variable declaration expression
                            match &expr.kind {
                                parser::ExprKind::Var { .. } | parser::ExprKind::Final { .. } => {
                                    // Variable declaration - lower as expression and convert to statement
                                    match self.lower_expression(expr) {
                                        Ok(typed_expr) => {
                                            // Extract the declaration info to create a proper statement
                                            if let TypedExpressionKind::VarDeclarationExpr {
                                                symbol_id,
                                                var_type,
                                                initializer,
                                            } = typed_expr.kind
                                            {
                                                statements.push(TypedStatement::VarDeclaration {
                                                    symbol_id,
                                                    var_type,
                                                    initializer: Some(*initializer),
                                                    mutability: crate::tast::symbols::Mutability::Mutable,
                                                    source_location: self
                                                        .context
                                                        .span_to_location(&expr.span),
                                                });
                                            } else if let TypedExpressionKind::FinalDeclarationExpr {
                                                symbol_id,
                                                var_type,
                                                initializer,
                                            } = typed_expr.kind
                                            {
                                                statements.push(TypedStatement::VarDeclaration {
                                                    symbol_id,
                                                    var_type,
                                                    initializer: Some(*initializer),
                                                    mutability: crate::tast::symbols::Mutability::Immutable,
                                                    source_location: self
                                                        .context
                                                        .span_to_location(&expr.span),
                                                });
                                            }
                                        }
                                        Err(e) => {
                                            // Collect error and continue processing other statements
                                            self.collected_errors.push(e);
                                        }
                                    }
                                }
                                parser::ExprKind::Return(_) => {
                                    // Return expression - convert to Return statement
                                    match self.lower_expression(expr) {
                                        Ok(typed_expr) => {
                                            if let TypedExpressionKind::Return { value } =
                                                typed_expr.kind
                                            {
                                                statements.push(TypedStatement::Return {
                                                    value: value.map(|v| *v),
                                                    source_location: self
                                                        .context
                                                        .span_to_location(&expr.span),
                                                });
                                            } else {
                                                // Fallback: wrap as expression statement
                                                statements.push(TypedStatement::Expression {
                                                    expression: typed_expr,
                                                    source_location: self
                                                        .context
                                                        .span_to_location(&expr.span),
                                                });
                                            }
                                        }
                                        Err(e) => {
                                            // Collect error and continue processing other statements
                                            self.collected_errors.push(e);
                                        }
                                    }
                                }
                                parser::ExprKind::Function(func) if !func.name.is_empty() => {
                                    // `function foo() {...}` in statement position
                                    // declares `foo`. The expression form yields an
                                    // anonymous literal, so without binding the name
                                    // here the declaration is unreachable and every
                                    // later call reports an unknown name.
                                    match self.lower_expression(expr) {
                                        Ok(typed_expr) => {
                                            let fn_name = self.context.intern_string(&func.name);
                                            let fn_type = typed_expr.expr_type;
                                            let fn_symbol = self
                                                .context
                                                .symbol_table
                                                .create_variable_with_type(
                                                    fn_name,
                                                    self.context.current_scope,
                                                    fn_type,
                                                );
                                            if let Some(scope) = self
                                                .context
                                                .scope_tree
                                                .get_scope_mut(self.context.current_scope)
                                            {
                                                scope.add_symbol(fn_symbol, fn_name);
                                            }
                                            statements.push(TypedStatement::VarDeclaration {
                                                symbol_id: fn_symbol,
                                                var_type: fn_type,
                                                initializer: Some(typed_expr),
                                                mutability:
                                                    crate::tast::symbols::Mutability::Mutable,
                                                source_location: self
                                                    .context
                                                    .span_to_location(&expr.span),
                                            });
                                        }
                                        Err(e) => {
                                            self.collected_errors.push(e);
                                        }
                                    }
                                }
                                _ => {
                                    // Regular expression - lower and wrap in statement
                                    match self.lower_expression(expr) {
                                        Ok(typed_expr) => {
                                            statements.push(TypedStatement::Expression {
                                                expression: typed_expr,
                                                source_location: self
                                                    .context
                                                    .span_to_location(&expr.span),
                                            });
                                        }
                                        Err(e) => {
                                            // Collect error and continue processing other statements
                                            self.collected_errors.push(e);
                                        }
                                    }
                                }
                            }
                        }
                        parser::BlockElement::Import(_)
                        | parser::BlockElement::Using(_)
                        | parser::BlockElement::Conditional(_) => {
                            // Skip imports, using statements, and conditional compilation for now
                            // These should be handled at the module level
                        }
                    }
                }

                // Leave the block scope
                let parent_scope = self
                    .context
                    .scope_tree
                    .get_scope(block_scope)
                    .and_then(|scope| scope.parent_id)
                    .unwrap_or(ScopeId::first());
                self.context.current_scope = parent_scope;

                TypedExpressionKind::Block {
                    statements,
                    scope_id: block_scope,
                }
            }
            ExprKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let cond_expr = self.lower_expression(cond)?;
                let then_expr = self.lower_expression(then_branch)?;
                let else_expr = if let Some(else_branch) = else_branch {
                    Some(Box::new(self.lower_expression(else_branch)?))
                } else {
                    None
                };

                TypedExpressionKind::Conditional {
                    condition: Box::new(cond_expr),
                    then_expr: Box::new(then_expr),
                    else_expr,
                }
            }
            ExprKind::While { cond, body } => {
                // Convert while expressions to statement form for proper CFG handling
                let cond_expr = self.lower_expression(cond)?;
                let body_stmt = self.convert_expression_to_statement(body)?;

                // Create a while statement and wrap it in a block expression
                let while_stmt = TypedStatement::While {
                    condition: cond_expr,
                    body: Box::new(body_stmt),
                    source_location: SourceLocation::unknown(),
                };

                // Return block expression containing the while statement
                TypedExpressionKind::Block {
                    statements: vec![while_stmt],
                    scope_id: ScopeId::from_raw(self.context.next_scope_id()),
                }
            }
            ExprKind::DoWhile { body, cond } => {
                // Convert do-while expressions to statement form
                let body_stmt = self.convert_expression_to_statement(body)?;
                let cond_expr = self.lower_expression(cond)?;

                // Create a do-while statement (add to TAST if missing)
                // Convert do-while to equivalent control flow:
                // { body; while(cond) { body } }
                let body_block = TypedStatement::Block {
                    statements: vec![body_stmt.clone()],
                    scope_id: ScopeId::from_raw(self.context.next_scope_id()),
                    source_location: SourceLocation::unknown(),
                };

                let while_stmt = TypedStatement::While {
                    condition: cond_expr,
                    body: Box::new(body_stmt),
                    source_location: SourceLocation::unknown(),
                };

                // Return block that executes body once, then while loop
                TypedExpressionKind::Block {
                    statements: vec![body_block, while_stmt],
                    scope_id: ScopeId::from_raw(self.context.next_scope_id()),
                }
            }
            ExprKind::For {
                var,
                key_var,
                iter,
                body,
            } => {
                return self.lower_for_expression(expression, var, key_var.as_deref(), iter, body);
            }
            ExprKind::Array(elements) => {
                let element_exprs = elements
                    .iter()
                    .map(|elem| self.lower_expression(elem))
                    .collect::<Result<Vec<_>, _>>()?;

                TypedExpressionKind::ArrayLiteral {
                    elements: element_exprs,
                }
            }
            ExprKind::Return(expr) => {
                let return_expr = if let Some(expr) = expr {
                    Some(Box::new(self.lower_expression(expr)?))
                } else {
                    None
                };

                TypedExpressionKind::Return { value: return_expr }
            }
            ExprKind::Break => TypedExpressionKind::Break,
            ExprKind::Continue => TypedExpressionKind::Continue,
            // Is doesn't exist in ExprKind, remove this variant
            ExprKind::Throw(expr) => {
                let expression = self.lower_expression(expr)?;
                TypedExpressionKind::Throw {
                    expression: Box::new(expression),
                }
            }
            ExprKind::Switch {
                expr,
                cases,
                default,
            } => {
                // Lower the discriminant expression
                let discriminant = Box::new(self.lower_expression(expr)?);

                // Store the discriminant type for use in pattern matching
                // This allows resolving enum constructor names like "Some" to "Option.Some"
                let prev_switch_type = self.context.switch_discriminant_type;
                self.context.switch_discriminant_type = Some(discriminant.expr_type);

                // Check if this is a switch expression or switch statement
                // In a switch expression, all cases must have expression values
                // In a switch statement, cases contain statements (like return)
                let is_expression = cases.iter().all(|case| {
                    // Check if the case body is a simple expression (not a block with statements)
                    !matches!(
                        &case.body.kind,
                        ExprKind::Block(_)
                            | ExprKind::Return(_)
                            | ExprKind::Break
                            | ExprKind::Continue
                            | ExprKind::Throw(_)
                    )
                });

                let result = if is_expression {
                    // Lower as switch expression
                    let mut typed_cases = Vec::with_capacity(cases.len());
                    for case in cases {
                        let typed_case = self.lower_switch_case_expression(case)?;
                        typed_cases.push(typed_case);
                    }

                    // Lower the default case if present
                    let default_case = if let Some(default_expr) = default {
                        Some(Box::new(self.lower_expression(default_expr)?))
                    } else {
                        None
                    };

                    TypedExpressionKind::Switch {
                        discriminant,
                        cases: typed_cases,
                        default_case,
                    }
                } else {
                    // Switch statement - lower as a block with switch statement
                    // For now, we'll lower it as a switch expression but mark it as void type
                    let mut typed_cases = Vec::with_capacity(cases.len());
                    for case in cases {
                        let typed_case = self.lower_switch_case(case)?;
                        typed_cases.push(typed_case);
                    }

                    // Lower the default case if present
                    let default_case = if let Some(default_expr) = default {
                        Some(Box::new(self.lower_expression(default_expr)?))
                    } else {
                        None
                    };

                    // Create a switch that returns void
                    TypedExpressionKind::Switch {
                        discriminant,
                        cases: typed_cases,
                        default_case,
                    }
                };

                // Restore previous switch type
                self.context.switch_discriminant_type = prev_switch_type;

                result
            }
            ExprKind::Try {
                expr,
                catches,
                finally_block,
            } => {
                // Lower the try expression
                let try_expr = Box::new(self.lower_expression(expr)?);

                // Lower catch clauses
                let mut catch_clauses = Vec::new();
                for catch in catches {
                    let typed_catch = self.lower_catch_clause(catch)?;
                    catch_clauses.push(typed_catch);
                }

                // Lower finally block if present
                let typed_finally = if let Some(finally_expr) = finally_block {
                    Some(Box::new(self.lower_expression(finally_expr)?))
                } else {
                    None
                };

                TypedExpressionKind::Try {
                    try_expr,
                    catch_clauses,
                    finally_block: typed_finally,
                }
            }
            ExprKind::This => {
                // Find current class context
                let this_type = if let Some(current_class) = self.context.class_context_stack.last()
                {
                    type_resolution::resolve_this_type(
                        &self.context.type_table,
                        self.context.symbol_table,
                        Some(*current_class),
                    )
                } else {
                    self.context.type_table.borrow().dynamic_type()
                };
                TypedExpressionKind::This { this_type }
            }
            ExprKind::Super => {
                // Find current class context and get super type
                let super_type =
                    if let Some(current_class) = self.context.class_context_stack.last() {
                        type_resolution::resolve_super_type(
                            &self.context.type_table,
                            self.context.symbol_table,
                            Some(*current_class),
                        )
                    } else {
                        self.context.type_table.borrow().dynamic_type()
                    };
                TypedExpressionKind::Super { super_type }
            }
            ExprKind::Map(entries) => {
                // Map literal: ["key1" => value1, "key2" => value2]
                let mut typed_entries = Vec::with_capacity(entries.len());
                for (key_expr, value_expr) in entries {
                    let key = self.lower_expression(key_expr)?;
                    let value = self.lower_expression(value_expr)?;
                    typed_entries.push(TypedMapEntry {
                        key,
                        value,
                        source_location: self.context.create_location(),
                    });
                }
                TypedExpressionKind::MapLiteral {
                    entries: typed_entries,
                }
            }
            ExprKind::Object(fields) => {
                // Object literal
                let mut typed_fields = Vec::with_capacity(fields.len());
                for field in fields {
                    let value = self.lower_expression(&field.expr)?;
                    let field_name = self.context.intern_string(&field.name);
                    typed_fields.push(TypedObjectField {
                        name: field_name,
                        value,
                        source_location: self.context.create_location(),
                    });
                }
                TypedExpressionKind::ObjectLiteral {
                    fields: typed_fields,
                }
            }
            ExprKind::StringInterpolation(parts) => {
                let mut typed_parts = Vec::new();
                for part in parts {
                    match part {
                        parser::StringPart::Literal(text) => {
                            typed_parts.push(StringInterpolationPart::String(text.clone()));
                        }
                        parser::StringPart::Interpolation(expr) => {
                            let typed_expr = self.lower_expression(expr)?;
                            typed_parts.push(StringInterpolationPart::Expression(typed_expr));
                        }
                    }
                }
                TypedExpressionKind::StringInterpolation { parts: typed_parts }
            }
            ExprKind::Paren(expr) => {
                // Parentheses just pass through the inner expression
                return self.lower_expression(expr);
            }
            ExprKind::Tuple(elements) => {
                // Standalone tuple without a known target type — desugar to array literal.
                // e.g., (1, 2, 3) becomes [1, 2, 3]
                let array_expr = parser::Expr {
                    kind: parser::ExprKind::Array(elements.clone()),
                    span: expression.span,
                };
                return self.lower_expression(&array_expr);
            }
            ExprKind::Cast { expr, type_hint } => {
                let typed_expr = self.lower_expression(expr)?;
                let (target_type, cast_kind) = if let Some(hint) = type_hint {
                    // cast(expr, Type) — safe/explicit cast
                    (self.lower_type(hint)?, CastKind::Explicit)
                } else {
                    // `cast expr` takes its type from the context — `var a:A =
                    // cast e` is a cast to A. Typing it Dynamic instead loses
                    // that, and every later use of the binding then takes the
                    // Dynamic path: a field read would unbox a value that was
                    // never boxed and dereference the result.
                    // Only a class or interface target is adopted. An abstract
                    // reached this way would be routed through its @:from
                    // conversions, which is a different operation than
                    // reinterpreting the value.
                    let from_context = self
                        .expected_arg_type_stack
                        .last()
                        .copied()
                        .flatten()
                        .or(self.context.expected_return_type)
                        .filter(|ty| {
                            matches!(
                                self.context.type_table.borrow().get(*ty).map(|t| &t.kind),
                                Some(TypeKind::Class { .. }) | Some(TypeKind::Interface { .. })
                            )
                        });
                    match from_context {
                        Some(ty) => (ty, CastKind::Unsafe),
                        None => (
                            self.context.type_table.borrow().dynamic_type(),
                            CastKind::Unsafe,
                        ),
                    }
                };
                TypedExpressionKind::Cast {
                    expression: Box::new(typed_expr),
                    target_type,
                    cast_kind,
                }
            }
            ExprKind::TypeCheck { expr, type_hint } => {
                // (expr : Type) is a type check hint — returns the value (not a boolean).
                // It asserts at compile time that expr is compatible with Type.
                // At runtime, it acts as an implicit cast (identity for same type, coercion otherwise).
                let typed_expr = self.lower_expression(expr)?;
                let target_type = self.lower_type(type_hint)?;

                TypedExpressionKind::Cast {
                    expression: Box::new(typed_expr),
                    target_type,
                    cast_kind: CastKind::Checked,
                }
            }
            ExprKind::Function(func) => {
                // Function expression/lambda - create a new scope for the function body
                let function_scope = self.context.enter_scope(ScopeKind::Function);

                // If the surrounding call set up an expected lambda signature
                // (e.g. `parallelFor(3, function(i, n) {...})` where the
                // formal param is `fn:(idx:Int, node:Int)->Void`), use the
                // expected param types to fill in any *untyped* lambda
                // parameters. Without this, untyped params default to
                // `Dynamic` and MIR ends up with `*void` formal types — the
                // caller passes i32 args reinterpreted as pointers and the
                // body dereferences address 0/1/2 producing "null" garbage.
                let expected_params = self
                    .expected_lambda_params_stack
                    .last()
                    .and_then(|p| p.clone());

                // Lower parameters - they will be automatically registered in the function scope
                let mut parameters = Vec::new();
                for (i, param) in func.params.iter().enumerate() {
                    let expected_ty = expected_params.as_ref().and_then(|ps| ps.get(i).copied());
                    let param_result = if param.type_hint.is_none() && expected_ty.is_some() {
                        self.lower_function_param_with_type(param, expected_ty.unwrap())?
                    } else {
                        self.lower_function_param(param)?
                    };
                    parameters.push(param_result);
                }

                // Lower function body in the new scope
                let body = if let Some(body_expr) = &func.body {
                    self.lower_function_body(body_expr)?
                } else {
                    Vec::new()
                };

                // Determine return type: explicit annotation > infer from body > void
                let return_type = if let Some(ret_type) = &func.return_type {
                    self.lower_type(ret_type)?
                } else {
                    self.infer_return_type_from_body(&body)
                };

                // Exit the function scope
                self.context.exit_scope();

                TypedExpressionKind::FunctionLiteral {
                    parameters,
                    body,
                    return_type,
                }
            }
            ExprKind::Arrow { params, expr } => {
                // Arrow function: x -> x * 2 or (x:Int) -> x * 2
                let function_scope = self.context.enter_scope(ScopeKind::Function);

                // Same expected-types-from-surrounding-call mechanism as
                // ExprKind::Function above — see that branch for rationale.
                let expected_params = self
                    .expected_lambda_params_stack
                    .last()
                    .and_then(|p| p.clone());

                let mut typed_params = Vec::new();
                for (i, param) in params.iter().enumerate() {
                    let param_interned = self.context.string_interner.intern(&param.name);

                    // Use type annotation if present, then expected-from-context,
                    // otherwise fall back to dynamic.
                    let param_type = if let Some(ref type_hint) = param.type_hint {
                        self.lower_type(type_hint)?
                    } else if let Some(expected) =
                        expected_params.as_ref().and_then(|ps| ps.get(i).copied())
                    {
                        expected
                    } else {
                        self.context.type_table.borrow().dynamic_type()
                    };

                    // Create symbol WITH the correct type so body expressions
                    // (like x * 2) resolve the variable to the right type
                    let param_symbol = self.context.symbol_table.create_variable_with_type(
                        param_interned,
                        self.context.current_scope,
                        param_type,
                    );

                    typed_params.push(TypedParameter {
                        symbol_id: param_symbol,
                        name: param_interned,
                        param_type,
                        is_optional: false,
                        default_value: None,
                        mutability: crate::tast::symbols::Mutability::Immutable,
                        ownership: Default::default(),
                        source_location: self.context.span_to_location(&expression.span),
                    });
                }

                // Lower arrow body in the new scope
                // For block bodies like () -> { ...; return x; }, use lower_function_body
                // to get flat statements so return type inference works correctly.
                // For simple expressions like () -> x * 2, lower as expression directly.
                let (body, return_type) = if matches!(&expr.kind, ExprKind::Block(_)) {
                    let body = self.lower_function_body(expr)?;
                    let return_type = self.infer_return_type_from_body(&body);
                    (body, return_type)
                } else {
                    let body_expr = self.lower_expression(expr)?;
                    let return_type = body_expr.expr_type;
                    let body = vec![TypedStatement::Expression {
                        expression: body_expr.clone(),
                        source_location: body_expr.source_location,
                    }];
                    (body, return_type)
                };

                // Exit the function scope
                self.context.exit_scope();

                TypedExpressionKind::FunctionLiteral {
                    parameters: typed_params,
                    body,
                    return_type,
                }
            }
            ExprKind::Var {
                name,
                type_hint,
                expr,
            } => {
                // Variable declaration as expression: `var x = 5` returns 5
                let var_name = self.context.intern_string(name);

                // Resolve target type FIRST for type-directed desugaring (e.g., tuples)
                let declared_type = if let Some(th) = type_hint {
                    Some(self.lower_type(th)?)
                } else {
                    None
                };

                // Check for tuple → SIMD4f.make() desugaring
                if let (Some(init_expr), Some(target_ty)) = (expr.as_ref(), declared_type) {
                    if let ExprKind::Tuple(elements) = &init_expr.kind {
                        if let Some(desugared) =
                            self.try_desugar_tuple_to_make(elements, target_ty, expression)?
                        {
                            // Successfully desugared tuple to a static method call.
                            // Wrap in VarDeclarationExpr.
                            let var_symbol = self.context.symbol_table.create_variable_with_type(
                                var_name,
                                self.context.current_scope,
                                target_ty,
                            );
                            if let Some(scope) = self
                                .context
                                .scope_tree
                                .get_scope_mut(self.context.current_scope)
                            {
                                scope.add_symbol(var_symbol, var_name);
                            }
                            return Ok(TypedExpression {
                                kind: TypedExpressionKind::VarDeclarationExpr {
                                    symbol_id: var_symbol,
                                    var_type: target_ty,
                                    initializer: Box::new(desugared),
                                },
                                expr_type: target_ty,
                                usage: VariableUsage::Copy,
                                lifetime_id: LifetimeId::from_raw(1),
                                source_location: self.context.span_to_location(&expression.span),
                                metadata: ExpressionMetadata::default(),
                            });
                        }
                    }
                }

                // An empty `[]` declared as a Map is Haxe's empty-map literal, not an
                // empty array. Built as an array it produced a HaxeArray whose static
                // type said Map, so the two disagreed: `m[k] = v` dispatched on the
                // static type to the map setter while the object underneath was an
                // array, and the value went nowhere.
                //
                // `Map<K,V>` resolves to the CONCRETE map class for its key --
                // StringMap, IntMap, ObjectMap, EnumValueMap -- so the check is on
                // those names, not on "Map", and the constructed class is whichever
                // one the type already picked.
                //
                // Only the EMPTY literal is redirected: a non-empty `[a, b]` is an
                // array whatever the hint says, and `[k => v]` is a Map node already.
                if let (Some(init_expr), Some(target_ty)) = (expr.as_ref(), declared_type) {
                    if matches!(&init_expr.kind, ExprKind::Array(e) if e.is_empty()) {
                        use crate::tast::core::TypeKind;
                        let map_class = {
                            let tt = self.context.type_table.borrow();
                            match tt.get(target_ty).map(|t| &t.kind) {
                                Some(TypeKind::Class { symbol_id, .. })
                                | Some(TypeKind::Abstract { symbol_id, .. }) => self
                                    .context
                                    .symbol_table
                                    .get_symbol(*symbol_id)
                                    .and_then(|sy| self.context.string_interner.get(sy.name))
                                    .filter(|n| {
                                        matches!(
                                            *n,
                                            "Map"
                                                | "StringMap"
                                                | "IntMap"
                                                | "ObjectMap"
                                                | "EnumValueMap"
                                        )
                                    })
                                    .map(|n| n.to_string()),
                                _ => None,
                            }
                        };
                        if let Some(class_name) = map_class {
                            let ctor = parser::Expr {
                                kind: ExprKind::New {
                                    type_path: parser::TypePath {
                                        package: Vec::new(),
                                        name: class_name,
                                        sub: None,
                                    },
                                    params: Vec::new(),
                                    args: Vec::new(),
                                },
                                span: init_expr.span,
                            };
                            let rebuilt = parser::Expr {
                                kind: ExprKind::Var {
                                    name: name.clone(),
                                    type_hint: type_hint.clone(),
                                    expr: Some(Box::new(ctor)),
                                },
                                span: expression.span,
                            };
                            return self.lower_expression(&rebuilt);
                        }
                    }
                }

                // Check for implicit @:from conversion (e.g., array literal → abstract type)
                // Array/object literals assigned to abstract types need an explicit Cast node
                // so the MIR Cast handler can look up and call the @:from conversion function.
                // Simple literals (int, float, string) and variables are handled by the
                // MIR Let handler's maybe_abstract_from_convert() instead.
                if let (Some(init_expr), Some(target_ty)) = (expr.as_ref(), declared_type) {
                    let needs_cast = self.is_abstract_type(target_ty)
                        && matches!(&init_expr.kind, ExprKind::Array(_) | ExprKind::Object(_));
                    if needs_cast {
                        // Lower the initializer, then wrap in an implicit cast to the abstract type
                        let array_expr = self.lower_expression(init_expr)?;
                        let cast_expr = TypedExpression {
                            kind: TypedExpressionKind::Cast {
                                expression: Box::new(array_expr),
                                target_type: target_ty,
                                cast_kind: crate::tast::node::CastKind::Implicit,
                            },
                            expr_type: target_ty,
                            usage: VariableUsage::Copy,
                            lifetime_id: LifetimeId::from_raw(1),
                            source_location: self.context.span_to_location(&expression.span),
                            metadata: ExpressionMetadata::default(),
                        };
                        let var_symbol = self.context.symbol_table.create_variable_with_type(
                            var_name,
                            self.context.current_scope,
                            target_ty,
                        );
                        if let Some(scope) = self
                            .context
                            .scope_tree
                            .get_scope_mut(self.context.current_scope)
                        {
                            scope.add_symbol(var_symbol, var_name);
                        }
                        return Ok(TypedExpression {
                            kind: TypedExpressionKind::VarDeclarationExpr {
                                symbol_id: var_symbol,
                                var_type: target_ty,
                                initializer: Box::new(cast_expr),
                            },
                            expr_type: target_ty,
                            usage: VariableUsage::Copy,
                            lifetime_id: LifetimeId::from_raw(1),
                            source_location: self.context.span_to_location(&expression.span),
                            metadata: ExpressionMetadata::default(),
                        });
                    }
                }

                // Lower initializer expression first if it exists.
                //
                // Pass the declared type as an `expected_new_type_hint` so a
                // bare `new C()` initializer can pick up type arguments from
                // the variable's annotation. This matters for `@:multiType`
                // abstracts like `Map`: `var m:Map<String,Int> = new Map()`
                // must resolve to `StringMap<Int>`, but without the hint the
                // New site sees zero type args and falls through to the
                // (broken) abstract path. Restored on the way out — nested
                // declarations shouldn't leak their hint.
                let initializer = if let Some(init_expr) = expr {
                    let prev_hint = self.context.expected_new_type_hint;
                    self.context.expected_new_type_hint = declared_type;
                    // The annotation is also the expected type of the whole
                    // initializer — lets a bare enum-variant identifier
                    // (`var v:ColorA = Red`) disambiguate against same-named
                    // variants on other enums.
                    // A declared function type also types an untyped lambda
                    // initializer's parameters, as a call's formal does for its
                    // lambda argument.
                    let lambda_hint: Option<Vec<TypeId>> = declared_type.and_then(|dt| {
                        let tt = self.context.type_table.borrow();
                        match tt.get(dt).map(|t| &t.kind) {
                            Some(crate::tast::core::TypeKind::Function { params, .. }) => {
                                Some(params.clone())
                            }
                            _ => None,
                        }
                    });
                    self.expected_lambda_params_stack.push(lambda_hint);
                    self.expected_arg_type_stack.push(declared_type);
                    let result = self.lower_expression(init_expr);
                    self.expected_arg_type_stack.pop();
                    self.expected_lambda_params_stack.pop();
                    self.context.expected_new_type_hint = prev_hint;
                    result?
                } else {
                    // Default to null if no initializer
                    TypedExpression {
                        kind: TypedExpressionKind::Null,
                        expr_type: self.context.type_table.borrow().dynamic_type(),
                        usage: VariableUsage::Copy,
                        lifetime_id: LifetimeId::from_raw(1),
                        source_location: self.context.span_to_location(&expression.span),
                        metadata: ExpressionMetadata::default(),
                    }
                };

                // Determine variable type (use already-resolved declared_type if available)
                let var_type = if let Some(dt) = declared_type {
                    dt
                } else {
                    initializer.expr_type
                };

                // Create the variable symbol with the correct type
                let var_symbol = self.context.symbol_table.create_variable_with_type(
                    var_name,
                    self.context.current_scope,
                    var_type,
                );

                // Add the variable to the current scope so it can be resolved later
                if let Some(scope) = self
                    .context
                    .scope_tree
                    .get_scope_mut(self.context.current_scope)
                {
                    scope.add_symbol(var_symbol, var_name);
                }

                // Monomorph rewrite: an UNTYPED empty array literal (`var x = []`)
                // gets element type Dynamic here; track it so the first
                // `x.push(e)` / `x[i] = e` can bind it to e's concrete type (see
                // bind_inferred_array_element). Skip if the user annotated a type.
                if declared_type.is_none()
                    && expr
                        .as_ref()
                        .map(|e| matches!(&e.kind, ExprKind::Array(els) if els.is_empty()))
                        .unwrap_or(false)
                {
                    let loc = self.context.span_to_location(&expression.span);
                    self.empty_array_inferred.insert(var_symbol, loc);
                }

                TypedExpressionKind::VarDeclarationExpr {
                    symbol_id: var_symbol,
                    var_type,
                    initializer: Box::new(initializer),
                }
            }
            ExprKind::Final {
                name,
                type_hint,
                expr,
            } => {
                // Final declaration as expression: `final x = 5` returns 5
                let var_name = self.context.intern_string(name);

                // Resolve target type FIRST for type-directed desugaring
                let declared_type = if let Some(th) = type_hint {
                    Some(self.lower_type(th)?)
                } else {
                    None
                };

                // Check for tuple → SIMD4f.make() desugaring
                if let (Some(init_expr), Some(target_ty)) = (expr.as_ref(), declared_type) {
                    if let ExprKind::Tuple(elements) = &init_expr.kind {
                        if let Some(desugared) =
                            self.try_desugar_tuple_to_make(elements, target_ty, expression)?
                        {
                            let var_symbol = self.context.symbol_table.create_variable_with_type(
                                var_name,
                                self.context.current_scope,
                                target_ty,
                            );
                            if let Some(scope) = self
                                .context
                                .scope_tree
                                .get_scope_mut(self.context.current_scope)
                            {
                                scope.add_symbol(var_symbol, var_name);
                            }
                            return Ok(TypedExpression {
                                kind: TypedExpressionKind::FinalDeclarationExpr {
                                    symbol_id: var_symbol,
                                    var_type: target_ty,
                                    initializer: Box::new(desugared),
                                },
                                expr_type: target_ty,
                                usage: VariableUsage::Copy,
                                lifetime_id: LifetimeId::from_raw(1),
                                source_location: self.context.span_to_location(&expression.span),
                                metadata: ExpressionMetadata::default(),
                            });
                        }
                    }
                }

                // Final variables must have an initializer
                let initializer = if let Some(init_expr) = expr {
                    // Annotation = expected type of the initializer (bare
                    // enum-variant disambiguation, mirroring `var`).
                    // A declared function type also types an untyped lambda
                    // initializer's parameters, as a call's formal does for its
                    // lambda argument.
                    let lambda_hint: Option<Vec<TypeId>> = declared_type.and_then(|dt| {
                        let tt = self.context.type_table.borrow();
                        match tt.get(dt).map(|t| &t.kind) {
                            Some(crate::tast::core::TypeKind::Function { params, .. }) => {
                                Some(params.clone())
                            }
                            _ => None,
                        }
                    });
                    self.expected_lambda_params_stack.push(lambda_hint);
                    self.expected_arg_type_stack.push(declared_type);
                    let result = self.lower_expression(init_expr);
                    self.expected_arg_type_stack.pop();
                    self.expected_lambda_params_stack.pop();
                    result?
                } else {
                    return Err(LoweringError::IncompleteImplementation {
                        feature: "Final declaration without initializer".to_string(),
                        location: self.context.span_to_location(&expression.span),
                    });
                };

                // Determine variable type
                let var_type = if let Some(dt) = declared_type {
                    dt
                } else {
                    // Infer type from initializer
                    initializer.expr_type
                };

                // Create the variable symbol with the correct type
                let var_symbol = self.context.symbol_table.create_variable_with_type(
                    var_name,
                    self.context.current_scope,
                    var_type,
                );

                // Add the variable to the current scope so it can be resolved later
                if let Some(scope) = self
                    .context
                    .scope_tree
                    .get_scope_mut(self.context.current_scope)
                {
                    scope.add_symbol(var_symbol, var_name);
                }

                TypedExpressionKind::FinalDeclarationExpr {
                    symbol_id: var_symbol,
                    var_type,
                    initializer: Box::new(initializer),
                }
            }
            ExprKind::Meta { meta, expr } => {
                // Metadata annotation: @:meta expr
                if std::env::var("RAYZOR_META_LOG").is_ok_and(|v| v != "0") {
                    eprintln!(
                        "[meta] @:{} on {:?} at {:?}",
                        meta.name,
                        std::mem::discriminant(&expr.kind),
                        self.context.span_to_location(&meta.span)
                    );
                }
                let inner_expr = self.lower_expression(expr)?;

                // Convert parser metadata to typed metadata
                let typed_meta = TypedMetadata {
                    name: self.context.intern_string(&meta.name),
                    params: meta
                        .params
                        .iter()
                        .map(|param_expr| self.lower_expression(param_expr))
                        .collect::<Result<Vec<_>, _>>()?,
                    source_location: self.context.span_to_location(&meta.span),
                };

                TypedExpressionKind::Meta {
                    metadata: vec![typed_meta],
                    expression: Box::new(inner_expr),
                }
            }
            ExprKind::DollarIdent { name, arg } => {
                // Dollar identifier: $type, $v{...}, $i{...}, etc.
                let arg_expr = if let Some(arg_expr) = arg {
                    Some(Box::new(self.lower_expression(arg_expr)?))
                } else {
                    None
                };

                TypedExpressionKind::DollarIdent {
                    name: self.context.intern_string(name),
                    arg: arg_expr,
                }
            }
            ExprKind::Untyped(expr) => {
                // Untyped expression: untyped expr
                // Just lower the inner expression, the "untyped" is more of a compiler hint
                self.lower_expression(expr)?.kind
            }
            ExprKind::Macro(expr) => {
                // Macro expression: macro expr
                // Lower as macro expression in TAST
                let inner_expr = self.lower_expression(expr)?;
                let macro_name = self.context.intern_string("macro");
                let macro_symbol = self.context.symbol_table.create_variable(macro_name);
                TypedExpressionKind::MacroExpression {
                    macro_symbol,
                    arguments: vec![inner_expr],
                }
            }
            ExprKind::Inline(expr) => {
                // Inline expression: inline expr
                // The 'inline' modifier is a hint to the compiler to inline the call
                // For now, just lower the inner expression (inlining would happen in a later pass)
                self.lower_expression(expr)?.kind
            }
            ExprKind::Reify(expr) => {
                // Macro reification: $expr
                // This is similar to DollarIdent but for expressions
                let inner_expr = self.lower_expression(expr)?;
                TypedExpressionKind::DollarIdent {
                    name: self.context.intern_string("reify"),
                    arg: Some(Box::new(inner_expr)),
                }
            }
            ExprKind::ArrayComprehension { for_parts, expr } => {
                // Array comprehension: [for (i in 0...10) i * 2]
                let expr_location = self.context.span_to_location(&expression.span);
                let comprehension =
                    self.lower_array_comprehension(for_parts, expr, &expr_location)?;
                comprehension.kind
            }
            ExprKind::MapComprehension {
                for_parts,
                key,
                value,
            } => {
                // Map comprehension: [for (i in 0...10) i => i * 2]
                let expr_location = self.context.span_to_location(&expression.span);
                let comprehension =
                    self.lower_map_comprehension(for_parts, key, value, &expr_location)?;
                comprehension.kind
            }
            ExprKind::CompilerSpecific { target, code, args } => {
                // Compiler-specific code: __c__("code {0}", arg0)
                let code_expr = self.lower_expression(code)?;
                let lowered_args = args
                    .iter()
                    .filter_map(|a| self.lower_expression(a).ok())
                    .collect();
                TypedExpressionKind::CompilerSpecific {
                    target: self.context.intern_string(target),
                    code: Box::new(code_expr),
                    args: lowered_args,
                }
            }
            // For now, handle remaining expression types with placeholders
            _ => {
                // Return a placeholder expression for unhandled cases
                TypedExpressionKind::Literal {
                    value: LiteralValue::String("unhandled_expression".to_string()),
                }
            }
        };

        // Determine expression type based on kind
        let expr_type = self.infer_expression_type(&kind)?;

        // Determine ownership usage based on expression kind
        let usage = self.determine_variable_usage(&kind);

        // Assign lifetime based on expression scope and type
        let lifetime_id = self.assign_lifetime(&kind, &expr_type);

        // Analyze expression metadata
        let metadata = self.analyze_expression_metadata(&kind);

        let typed_expr = TypedExpression {
            expr_type,
            kind,
            usage,
            lifetime_id,
            source_location: self.context.span_to_location(&expression.span),
            metadata,
        };

        // // Debug switch expressions
        // match &typed_expr.kind {
        //     TypedExpressionKind::Switch { .. } => {
        //         eprintln!(
        //             "DEBUG: Created switch expression with type: {:?}",
        //             typed_expr.expr_type
        //         );
        //     }
        //     _ => {}
        // }

        Ok(typed_expr)
    }

    /// Lower a function call expression (ExprKind::Call).
    /// Extracted from lower_expression to reduce stack frame size.
    #[inline(never)]
    /// Resolve a callee's formal parameter types for the Dynamic-boxing decision
    /// ONLY. Kept separate from `resolve_callee_formal_param_types` (which feeds
    /// lambda-param inference via `expected_arg_types`) so the import-aware
    /// resolution used for boxing cannot influence closure-param inference — the
    /// two concerns must not share a resolver.
    ///
    /// Handles `Class.method(...)` static calls (import-aware, so cross-module
    /// callees resolve) and free / same-class `name(...)` calls. Returns None
    /// for instance method calls (same-module boxing is still covered by
    /// `maybe_materialize_for_call`).

    /// Lower a literal
    fn lower_literal(&mut self, literal: &parser::StringPart) -> LoweringResult<LiteralValue> {
        match literal {
            parser::StringPart::Literal(text) => Ok(LiteralValue::String(text.clone())),
            parser::StringPart::Interpolation(expr) => {
                // String interpolation expressions should not be converted to literals
                // They should be handled as part of StringInterpolation expression
                Err(LoweringError::InternalError {
                    message: "String interpolation part cannot be converted to literal value"
                        .to_string(),
                    location: self.context.create_location_from_span(expr.span),
                })
            }
        }
    }

    /// Convert an expression to a statement for proper CFG handling
    fn convert_expression_to_statement(&mut self, expr: &Expr) -> LoweringResult<TypedStatement> {
        let typed_expr = self.lower_expression(expr)?;

        // Wrap expression in an expression statement
        Ok(TypedStatement::Expression {
            expression: typed_expr,
            source_location: SourceLocation::unknown(),
        })
    }

    /// Assign lifetime based on expression scope and type (simplified for TAST)
    fn assign_lifetime(&self, kind: &TypedExpressionKind, expr_type: &TypeId) -> LifetimeId {
        match kind {
            TypedExpressionKind::Literal { .. } => {
                // Literals have static lifetime
                LifetimeId::static_lifetime()
            }
            TypedExpressionKind::Variable { symbol_id } => {
                // Variables have lifetime tied to their declaring scope
                if let Some(symbol) = self.context.symbol_table.get_symbol(*symbol_id) {
                    symbol.lifetime_id
                } else {
                    LifetimeId::first() // Default lifetime for TAST
                }
            }
            _ => {
                // Default to current scope lifetime - detailed analysis in semantic graph
                LifetimeId::from_raw(1) // TODO: Use proper lifetime ID generation
            }
        }
    }

    /// Analyze expression metadata for optimization and error reporting
    fn analyze_expression_metadata(&self, kind: &TypedExpressionKind) -> ExpressionMetadata {
        let mut metadata = ExpressionMetadata::default();

        match kind {
            TypedExpressionKind::Literal { .. } => {
                metadata.is_constant = true;
                metadata.has_side_effects = false;
                metadata.can_throw = false;
                metadata.complexity_score = 1;
            }
            TypedExpressionKind::Variable { .. } => {
                metadata.is_constant = false;
                metadata.has_side_effects = false;
                metadata.can_throw = false;
                metadata.complexity_score = 1;
            }
            TypedExpressionKind::FunctionCall { .. } => {
                metadata.is_constant = false;
                metadata.has_side_effects = true; // Assume function calls have side effects
                metadata.can_throw = true; // Assume function calls can throw
                metadata.complexity_score = 10;
            }
            TypedExpressionKind::BinaryOp { operator, .. } => {
                metadata.is_constant = false;
                metadata.complexity_score = 2;

                match operator {
                    BinaryOperator::Assign
                    | BinaryOperator::AddAssign
                    | BinaryOperator::SubAssign
                    | BinaryOperator::MulAssign
                    | BinaryOperator::DivAssign
                    | BinaryOperator::ModAssign => {
                        metadata.has_side_effects = true;
                        metadata.can_throw = false;
                    }
                    BinaryOperator::Div | BinaryOperator::Mod => {
                        metadata.has_side_effects = false;
                        metadata.can_throw = true; // Division by zero
                    }
                    _ => {
                        metadata.has_side_effects = false;
                        metadata.can_throw = false;
                    }
                }
            }
            TypedExpressionKind::New { .. } => {
                metadata.is_constant = false;
                metadata.has_side_effects = true; // Memory allocation
                metadata.can_throw = true; // Constructor can throw
                metadata.complexity_score = 5;
            }
            _ => {
                metadata.is_constant = false;
                metadata.has_side_effects = false;
                metadata.can_throw = false;
                metadata.complexity_score = 1;
            }
        }

        metadata
    }

    /// Convert an expression to a statement
    fn lower_expression_to_statement(
        &mut self,
        expr: &parser::Expr,
    ) -> Result<TypedStatement, LoweringError> {
        let typed_expr = self.lower_expression(expr)?;
        Ok(TypedStatement::Expression {
            expression: typed_expr,
            source_location: self.context.span_to_location(&expr.span),
        })
    }

    /// Convert a parser expression to a string representation
    /// Used for extracting metadata parameter values
    pub(crate) fn expr_to_string(&self, expr: &parser::Expr) -> String {
        match &expr.kind {
            parser::ExprKind::String(s) => s.clone(),
            parser::ExprKind::Ident(id) => id.clone(),
            parser::ExprKind::Int(n) => n.to_string(),
            parser::ExprKind::Float(f) => f.to_string(),
            parser::ExprKind::Bool(b) => b.to_string(),
            parser::ExprKind::Binary { left, op, right } => {
                format!(
                    "{} {:?} {}",
                    self.expr_to_string(left),
                    op,
                    self.expr_to_string(right)
                )
            }
            parser::ExprKind::Unary { op, expr: operand } => {
                format!("{:?}{}", op, self.expr_to_string(operand))
            }
            parser::ExprKind::Paren(inner) => {
                format!("({})", self.expr_to_string(inner))
            }
            parser::ExprKind::Tuple(elements) => {
                let parts: Vec<_> = elements.iter().map(|e| self.expr_to_string(e)).collect();
                format!("({})", parts.join(", "))
            }
            _ => format!("{:?}", expr.kind), // Fallback for complex expressions
        }
    }

    /// Analyze if function body can throw exceptions
    pub(crate) fn analyze_can_throw(&self, body: &Option<Box<parser::Expr>>) -> bool {
        if let Some(body_expr) = body {
            self.expr_can_throw(body_expr)
        } else {
            false
        }
    }

    /// Check if an expression can throw
    fn expr_can_throw(&self, expr: &parser::Expr) -> bool {
        match &expr.kind {
            parser::ExprKind::Throw(_) => true,
            parser::ExprKind::Block(elements) => elements.iter().any(|elem| {
                if let parser::BlockElement::Expr(e) = elem {
                    self.expr_can_throw(e)
                } else {
                    false
                }
            }),
            parser::ExprKind::If {
                then_branch,
                else_branch,
                ..
            } => {
                self.expr_can_throw(then_branch)
                    || else_branch
                        .as_ref()
                        .map_or(false, |e| self.expr_can_throw(e))
            }
            parser::ExprKind::Switch { cases, default, .. } => {
                cases.iter().any(|c| self.expr_can_throw(&c.body))
                    || default.as_ref().map_or(false, |d| self.expr_can_throw(d))
            }
            parser::ExprKind::Try { .. } => false, // Try blocks handle exceptions
            parser::ExprKind::While { body, .. }
            | parser::ExprKind::DoWhile { body, .. }
            | parser::ExprKind::For { body, .. } => self.expr_can_throw(body),
            _ => false,
        }
    }

    /// Detect if function is async based on @:async metadata
    pub(crate) fn detect_async_kind(&self, meta: &[parser::haxe_ast::Metadata]) -> AsyncKind {
        for m in meta {
            let name = m.name.strip_prefix(':').unwrap_or(&m.name);
            if name == "async" {
                return AsyncKind::Async;
            }
        }
        AsyncKind::Sync
    }

    /// Analyze if function is pure (no side effects)
    pub(crate) fn analyze_is_pure(&self, body: &Option<Box<parser::Expr>>) -> bool {
        if let Some(body_expr) = body {
            self.expr_is_pure(body_expr)
        } else {
            true // No body means pure
        }
    }

    /// Check if an expression is pure
    fn expr_is_pure(&self, expr: &parser::Expr) -> bool {
        match &expr.kind {
            // Pure expressions
            parser::ExprKind::Int(_)
            | parser::ExprKind::Float(_)
            | parser::ExprKind::String(_)
            | parser::ExprKind::Bool(_)
            | parser::ExprKind::Null
            | parser::ExprKind::Ident(_) => true,

            // Assignments and mutations are impure
            parser::ExprKind::Assign { .. } => false,
            parser::ExprKind::Unary { op, .. } => !matches!(
                op,
                parser::UnaryOp::PreIncr
                    | parser::UnaryOp::PreDecr
                    | parser::UnaryOp::PostIncr
                    | parser::UnaryOp::PostDecr
            ),

            // Function calls might have side effects
            parser::ExprKind::Call { .. } | parser::ExprKind::New { .. } => false,

            // Recursively check compound expressions
            parser::ExprKind::Binary { left, right, .. } => {
                self.expr_is_pure(left) && self.expr_is_pure(right)
            }
            parser::ExprKind::Block(elements) => elements.iter().all(|elem| {
                if let parser::BlockElement::Expr(e) = elem {
                    self.expr_is_pure(e)
                } else {
                    true
                }
            }),

            _ => false, // Conservative: assume impure
        }
    }

    /// Calculate cyclomatic complexity of function body
    pub(crate) fn calculate_complexity(&self, body: &Option<Box<parser::Expr>>) -> u32 {
        if let Some(body_expr) = body {
            1 + self.expr_complexity(body_expr)
        } else {
            1
        }
    }

    /// Calculate expression complexity
    fn expr_complexity(&self, expr: &parser::Expr) -> u32 {
        match &expr.kind {
            parser::ExprKind::If {
                then_branch,
                else_branch,
                ..
            } => {
                1 + self.expr_complexity(then_branch)
                    + else_branch.as_ref().map_or(0, |e| self.expr_complexity(e))
            }
            parser::ExprKind::Switch { cases, default, .. } => {
                cases.len() as u32
                    + cases
                        .iter()
                        .map(|c| self.expr_complexity(&c.body))
                        .sum::<u32>()
                    + default.as_ref().map_or(0, |d| self.expr_complexity(d))
            }
            parser::ExprKind::While { body, .. }
            | parser::ExprKind::DoWhile { body, .. }
            | parser::ExprKind::For { body, .. } => 1 + self.expr_complexity(body),
            parser::ExprKind::Binary { op, left, right } => {
                let base = match op {
                    parser::BinaryOp::And | parser::BinaryOp::Or => 1,
                    _ => 0,
                };
                base + self.expr_complexity(left) + self.expr_complexity(right)
            }
            parser::ExprKind::Block(elements) => elements
                .iter()
                .map(|elem| {
                    if let parser::BlockElement::Expr(e) = elem {
                        self.expr_complexity(e)
                    } else {
                        0
                    }
                })
                .sum(),
            parser::ExprKind::Try { catches, .. } => catches.len() as u32,
            _ => 0,
        }
    }
}

mod binding;
mod calls;
mod exceptions;
mod fields;
mod loops;
mod operators;
mod patterns;
mod switch;
