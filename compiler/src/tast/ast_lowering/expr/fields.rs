//! Field access, property accessors and array-access wrappers.

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
    /// Convert parser PropertyAccess to TAST PropertyAccessor
    ///
    /// For "get" or "set", we derive the method name as "get_fieldname" or "set_fieldname"
    /// For custom names, we use the name directly
    ///
    /// The method name is stored as InternedString and resolved to SymbolId during MIR lowering
    pub(crate) fn convert_property_accessor(
        &mut self,
        access: &parser::PropertyAccess,
        field_name: &str,
        is_getter: bool,
    ) -> crate::tast::PropertyAccessor {
        match access {
            parser::PropertyAccess::Default => crate::tast::PropertyAccessor::Default,
            parser::PropertyAccess::Null => crate::tast::PropertyAccessor::Null,
            parser::PropertyAccess::Never => crate::tast::PropertyAccessor::Never,
            parser::PropertyAccess::Dynamic => crate::tast::PropertyAccessor::Dynamic,
            parser::PropertyAccess::Custom(method_name) => {
                // If the custom name is just "get" or "set", derive the full method name
                let full_method_name = if method_name == "get" || method_name == "set" {
                    format!("{}_{}", method_name, field_name)
                } else {
                    method_name.clone()
                };

                // Intern the method name for later resolution during MIR lowering
                let interned_name = self.context.intern_string(&full_method_name);
                crate::tast::PropertyAccessor::Method(interned_name)
            }
        }
    }

    /// Extract the first type argument from a generic class type
    /// (e.g. `T` from `Arc<T>`), used to type the synthesised `.get()`
    /// call in deref coercion.
    fn extract_wrapper_inner_type(&self, wrapper_type: TypeId) -> Option<TypeId> {
        let type_table = self.context.type_table.borrow();
        let ti = type_table.get(wrapper_type)?;
        match &ti.kind {
            crate::tast::core::TypeKind::Class { type_args, .. }
            | crate::tast::core::TypeKind::GenericInstance { type_args, .. } => {
                type_args.first().copied()
            }
            _ => None,
        }
    }

    /// Find a field in a class by symbol
    fn find_field_in_class(
        &self,
        class_symbol: &SymbolId,
        field_symbol: SymbolId,
    ) -> Option<(InternedString, TypeId, bool)> {
        if let Some(fields) = self.class_fields.get(class_symbol) {
            fields
                .iter()
                .find(|(_, symbol, _)| *symbol == field_symbol)
                .map(|(name, field_symbol, is_static)| {
                    let field_type = if let Some(field_sym) =
                        self.context.symbol_table.get_symbol(*field_symbol)
                    {
                        field_sym.type_id
                    } else {
                        self.context.type_table.borrow().dynamic_type()
                    };
                    (*name, field_type, *is_static)
                })
        } else {
            None
        }
    }

    /// Look up a *data field* (not a method) by name on a class.
    /// Methods live in `class_methods`; fields in `class_fields`. Used to tell a
    /// closure-valued field apart from a method at a call site.
    pub(crate) fn lookup_data_field(
        &self,
        class_sym: SymbolId,
        field_name: InternedString,
    ) -> Option<SymbolId> {
        let fields = self.class_fields.get(&class_sym)?;
        fields
            .iter()
            .find(|(n, _, _)| *n == field_name)
            .map(|(_, sym, _)| *sym)
    }

    /// Lower a field access expression (ExprKind::Field).
    /// Extracted from lower_expression to reduce stack frame size.
    #[inline(never)]
    pub(crate) fn lower_field_expression(
        &mut self,
        expression: &Expr,
        expr: &Expr,
        field: &str,
        is_optional: bool,
    ) -> LoweringResult<TypedExpression> {
        // Helper function to extract a fully qualified path from nested Field expressions
        // For example: rayzor.concurrent.Thread -> vec!["rayzor", "concurrent", "Thread"]
        fn extract_qualified_path(expr: &parser::Expr) -> Option<Vec<String>> {
            match &expr.kind {
                ExprKind::Ident(name) => Some(vec![name.clone()]),
                ExprKind::Field {
                    expr: inner_expr,
                    field,
                    ..
                } => {
                    let mut path = extract_qualified_path(inner_expr)?;
                    path.push(field.clone());
                    Some(path)
                }
                _ => None, // Not a qualified path
            }
        }

        // Try to extract a fully qualified path (e.g., rayzor.concurrent.Thread)
        if let Some(mut path) = extract_qualified_path(expr) {
            path.push(field.to_string()); // Add the final field (e.g., "spawn")

            // Before attempting qualified type/package resolution, check if the base
            // identifier is a local variable or parameter. If so, this is a field
            // access chain (a.b.c.process()), NOT a qualified type path.
            let base_name_interned = self.context.intern_string(&path[0]);
            let base_is_local_var = self
                .resolve_symbol_in_scope_hierarchy(base_name_interned)
                .and_then(|id| self.context.symbol_table.get_symbol(id))
                .map(|sym| {
                    matches!(
                        sym.kind,
                        crate::tast::symbols::SymbolKind::Variable
                            | crate::tast::symbols::SymbolKind::Parameter
                            | crate::tast::symbols::SymbolKind::Field
                    )
                })
                .unwrap_or(false);

            // Try to resolve this as a package.Class.staticMethod pattern
            // Start from the full path and work backwards to find the class
            // Skip this if the base is a local variable (field access chain)
            for split_point in (1..if base_is_local_var { 1 } else { path.len() }).rev() {
                let package_and_class = &path[..split_point];
                let remaining = &path[split_point..];

                // Try to resolve the package+class part as a symbol
                // For "rayzor.concurrent.Thread.spawn", try:
                // - "rayzor.concurrent.Thread" (class) with "spawn" (method)
                // - "rayzor.concurrent" (class) with "Thread.spawn" (not valid, skip)
                // - "rayzor" (class) with "concurrent.Thread.spawn" (not valid, skip)

                // For static field access like rayzor.concurrent.Thread.spawn:
                // - path = ["rayzor", "concurrent", "Thread", "spawn"]
                // - When split at 2: package_and_class=["rayzor", "concurrent"], remaining=["Thread", "spawn"]
                // - Package = ["rayzor", "concurrent"]
                // - Class = remaining[0] = "Thread"
                // - Field = remaining[1] = "spawn"
                //
                // For class name access like rayzor.concurrent.Thread:
                // - path = ["rayzor", "concurrent", "Thread"]
                // - When split at 2: package_and_class=["rayzor", "concurrent"], remaining=["Thread"]
                // - Package = ["rayzor", "concurrent"]
                // - Class = remaining[0] = "Thread"
                // - Field = None (just accessing the class itself)
                if remaining.len() == 1 {
                    // Just accessing a class name (e.g., rayzor.concurrent.Thread)
                    let package_parts = package_and_class;
                    let class_name = &remaining[0];

                    let class_name_interned = self.context.intern_string(class_name);

                    // Build fully qualified class name
                    let qualified_class_name = if package_parts.is_empty() {
                        class_name.clone()
                    } else {
                        format!("{}.{}", package_parts.join("."), class_name)
                    };
                    let qualified_class_interned =
                        self.context.intern_string(&qualified_class_name);

                    // Construct QualifiedPath for namespace resolver
                    let qualified_path = {
                        let package_interned: Vec<_> = package_parts
                            .iter()
                            .map(|p| self.context.intern_string(p))
                            .collect();
                        crate::tast::namespace::QualifiedPath::new(
                            package_interned,
                            class_name_interned,
                        )
                    };

                    // Try to resolve the class
                    let symbol_id_opt = self
                        .context
                        .namespace_resolver
                        .lookup_symbol(&qualified_path)
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
                        .or_else(|| self.resolve_class_like_symbol_by_name(class_name_interned));

                    if let Some(symbol_id) = symbol_id_opt {
                        if let Some(symbol) = self.context.symbol_table.get_symbol(symbol_id) {
                            if symbol.kind == crate::tast::symbols::SymbolKind::Class
                                || symbol.kind == crate::tast::symbols::SymbolKind::Enum
                            {
                                // Return a reference to the class/enum itself
                                let class_type = symbol.type_id;
                                return Ok(TypedExpression {
                                    expr_type: class_type,
                                    kind: TypedExpressionKind::Variable { symbol_id },
                                    usage: VariableUsage::Borrow,
                                    lifetime_id: crate::tast::LifetimeId::first(),
                                    source_location: self.context.create_location(),
                                    metadata: ExpressionMetadata::default(),
                                });
                            }
                        }
                    } else if package_parts.len() >= 2
                        || (!package_parts.is_empty()
                            && matches!(
                                package_parts[0].as_str(),
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
                            ))
                    {
                        // Qualified class not found AND looks like a package path
                        // Either has 2+ package components OR starts with known stdlib/project package
                        // This indicates a package path like rayzor.concurrent.Thread or haxe.ds.StringMap
                        // Return UnresolvedType to trigger on-demand loading
                        return Err(LoweringError::UnresolvedType {
                            type_name: qualified_class_name.clone(),
                            location: self.context.create_location_from_span(expression.span),
                        });
                    }
                } else if remaining.len() == 2 {
                    let package_parts = package_and_class; // Full package path
                    let class_name = &remaining[0]; // Class is first element of remaining
                    let field_name = &remaining[1]; // Field is second element of remaining

                    let class_name_interned = self.context.intern_string(class_name);
                    let field_name_interned = self.context.intern_string(field_name);

                    // Build fully qualified class name for fallback lookup
                    let qualified_class_name = if package_parts.is_empty() {
                        class_name.clone()
                    } else {
                        format!("{}.{}", package_parts.join("."), class_name)
                    };
                    let qualified_class_interned =
                        self.context.intern_string(&qualified_class_name);

                    // Construct QualifiedPath for namespace resolver
                    let qualified_path = {
                        let package_interned: Vec<_> = package_parts
                            .iter()
                            .map(|p| self.context.intern_string(p))
                            .collect();
                        crate::tast::namespace::QualifiedPath::new(
                            package_interned,
                            class_name_interned,
                        )
                    };

                    // Try to resolve the class using the namespace resolver

                    let symbol_id_opt = self
                        .context
                        .namespace_resolver
                        .lookup_symbol(&qualified_path)
                        .or_else(|| {
                            // Fallback: Try to look up in root scope using full path string
                            self.context
                                .symbol_table
                                .lookup_symbol(
                                    crate::tast::ScopeId::first(), // Root scope
                                    qualified_class_interned,
                                )
                                .map(|s| s.id)
                        })
                        .or_else(|| {
                            self.resolve_symbol_in_scope_hierarchy(qualified_class_interned)
                        })
                        .or_else(|| self.resolve_class_like_symbol_by_name(class_name_interned));

                    if let Some(symbol_id) = symbol_id_opt {
                        if let Some(symbol) = self.context.symbol_table.get_symbol(symbol_id) {
                            if symbol.kind == crate::tast::symbols::SymbolKind::Class {
                                // Found the class! Now look up the static field
                                {
                                    let field_info =
                                        if let Some(fields) = self.class_fields.get(&symbol_id) {
                                            fields
                                                .iter()
                                                .find(|(name, _, _)| *name == field_name_interned)
                                                .map(|(_, symbol, is_static)| (*symbol, *is_static))
                                        } else {
                                            None
                                        };

                                    if let Some((field_symbol, _is_static)) = field_info {
                                        let expr_type = if let Some(field) =
                                            self.find_field_in_class(&symbol_id, field_symbol)
                                        {
                                            field.1 // field type
                                        } else {
                                            self.context.type_table.borrow().dynamic_type()
                                        };

                                        let kind = TypedExpressionKind::StaticFieldAccess {
                                            class_symbol: symbol_id,
                                            field_symbol,
                                        };

                                        let usage = VariableUsage::Copy;
                                        let lifetime_id = self.assign_lifetime(&kind, &expr_type);
                                        let metadata = self.analyze_expression_metadata(&kind);

                                        return Ok(TypedExpression {
                                            expr_type,
                                            kind,
                                            usage,
                                            lifetime_id,
                                            source_location: self.context.create_location(),
                                            metadata,
                                        });
                                    }
                                }
                            } else if symbol.kind == crate::tast::symbols::SymbolKind::Enum {
                                // Found an enum! Look up the variant by field name
                                if let Some(variants) =
                                    self.context.symbol_table.get_enum_variants(symbol_id)
                                {
                                    for &variant_id in variants {
                                        if let Some(variant_sym) =
                                            self.context.symbol_table.get_symbol(variant_id)
                                        {
                                            if variant_sym.name == field_name_interned {
                                                let variant_type = variant_sym.type_id;
                                                let kind = TypedExpressionKind::Variable {
                                                    symbol_id: variant_id,
                                                };
                                                let usage = VariableUsage::Borrow;
                                                let lifetime_id =
                                                    self.assign_lifetime(&kind, &variant_type);
                                                let metadata =
                                                    self.analyze_expression_metadata(&kind);

                                                return Ok(TypedExpression {
                                                    expr_type: variant_type,
                                                    kind,
                                                    usage,
                                                    lifetime_id,
                                                    source_location: self
                                                        .context
                                                        .create_location_from_span(expression.span),
                                                    metadata,
                                                });
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else if package_parts.len() >= 2
                        || (!package_parts.is_empty()
                            && matches!(
                                package_parts[0].as_str(),
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
                            ))
                    {
                        // Qualified class not found AND looks like a package path
                        // Either has 2+ package components OR starts with known stdlib/project package
                        // This indicates a package path like rayzor.concurrent.Thread or haxe.ds.StringMap
                        // Return UnresolvedType to trigger on-demand loading
                        return Err(LoweringError::UnresolvedType {
                            type_name: qualified_class_name.clone(),
                            location: self.context.create_location_from_span(expression.span),
                        });
                    }
                }
            }
        }

        // Check if the expression is an identifier that refers to a class (static access)
        if let ExprKind::Ident(class_name) = &expr.kind {
            let class_name_interned = self.context.intern_string(class_name);

            // Try to resolve as a class or enum symbol
            if let Some(symbol_id) = self.resolve_class_like_symbol_by_name(class_name_interned) {
                // Extract symbol kind to release the borrow before calling intern_string
                let symbol_kind = self
                    .context
                    .symbol_table
                    .get_symbol(symbol_id)
                    .map(|s| s.kind);

                // Check if this symbol represents a class declaration (not just a variable of class type)
                if symbol_kind == Some(crate::tast::symbols::SymbolKind::Class) {
                    // This is a class name, so this is static field access
                    let class_symbol = symbol_id;
                    let field_name = self.context.intern_string(field);

                    // Look for the field in this class and check if it's static
                    let field_info = if let Some(fields) = self.class_fields.get(&class_symbol) {
                        fields
                            .iter()
                            .find(|(name, _, _)| *name == field_name)
                            .map(|(_, symbol, is_static)| (*symbol, *is_static))
                    } else {
                        None
                    };

                    if let Some((field_symbol, _is_static)) = field_info {
                        // Create StaticFieldAccess for any Class.field syntax
                        // The type checker will validate if it's allowed
                        let expr_type = if let Some(field) =
                            self.find_field_in_class(&class_symbol, field_symbol)
                        {
                            field.1 // field type
                        } else {
                            self.context.type_table.borrow().dynamic_type()
                        };

                        let kind = TypedExpressionKind::StaticFieldAccess {
                            class_symbol,
                            field_symbol,
                        };

                        let usage = VariableUsage::Copy;
                        let lifetime_id = self.assign_lifetime(&kind, &expr_type);
                        let metadata = self.analyze_expression_metadata(&kind);

                        // Calculate the span for the field name specifically
                        // The field appears after the object expression and a dot
                        let field_span = parser::haxe_ast::Span::new(
                            expr.span.end + 1, // +1 for the dot
                            expr.span.end + 1 + field.len(),
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

                // Check if this is an enum and the field is a variant
                if symbol_kind == Some(crate::tast::symbols::SymbolKind::Enum) {
                    let enum_symbol = symbol_id;
                    let variant_name = self.context.intern_string(field);

                    // Look up enum variants
                    if let Some(variants) = self.context.symbol_table.get_enum_variants(enum_symbol)
                    {
                        for &variant_id in variants {
                            if let Some(variant_sym) =
                                self.context.symbol_table.get_symbol(variant_id)
                            {
                                if variant_sym.name == variant_name {
                                    let variant_type = variant_sym.type_id;
                                    let kind = TypedExpressionKind::Variable {
                                        symbol_id: variant_id,
                                    };
                                    let usage = VariableUsage::Borrow;
                                    let lifetime_id = self.assign_lifetime(&kind, &variant_type);
                                    let metadata = self.analyze_expression_metadata(&kind);

                                    return Ok(TypedExpression {
                                        expr_type: variant_type,
                                        kind,
                                        usage,
                                        lifetime_id,
                                        source_location: self
                                            .context
                                            .create_location_from_span(expression.span),
                                        metadata,
                                    });
                                }
                            }
                        }
                    }
                }

                // Check if this is an abstract (enum abstract) and the field is a static value
                if symbol_kind == Some(crate::tast::symbols::SymbolKind::Abstract) {
                    let abstract_symbol = symbol_id;
                    let field_name = self.context.intern_string(field);

                    if let Some(fields) = self.class_fields.get(&abstract_symbol) {
                        if let Some((_, field_symbol, _)) =
                            fields.iter().find(|(name, _, _)| *name == field_name)
                        {
                            let field_symbol = *field_symbol;
                            let expr_type = self
                                .context
                                .symbol_table
                                .get_symbol(field_symbol)
                                .map(|s| s.type_id)
                                .unwrap_or_else(|| self.context.type_table.borrow().dynamic_type());

                            let kind = TypedExpressionKind::StaticFieldAccess {
                                class_symbol: abstract_symbol,
                                field_symbol,
                            };

                            let usage = VariableUsage::Copy;
                            let lifetime_id = self.assign_lifetime(&kind, &expr_type);
                            let metadata = self.analyze_expression_metadata(&kind);

                            return Ok(TypedExpression {
                                expr_type,
                                kind,
                                usage,
                                lifetime_id,
                                source_location: self
                                    .context
                                    .create_location_from_span(expression.span),
                                metadata,
                            });
                        }
                    }
                }
            }
        }

        // Not a static access, proceed with instance field access
        let mut obj_expr = self.lower_expression(expr)?;
        let field_name = self.context.intern_string(field);

        // Helper: look up a field or method by name in a class, checking both
        // class_fields and class_methods. Methods are tracked separately from fields,
        // so we must check both to resolve instance method calls like `obj.lock()`.
        let resolve_in_class =
            |this: &Self, class_sym: &SymbolId, name: InternedString| -> Option<SymbolId> {
                if let Some(fields) = this.class_fields.get(class_sym) {
                    if let Some((_, sym, _)) = fields.iter().find(|(n, _, _)| *n == name) {
                        return Some(*sym);
                    }
                }
                if let Some(methods) = this.class_methods.get(class_sym) {
                    if let Some((_, sym, _)) = methods.iter().find(|(n, _, _)| *n == name) {
                        return Some(*sym);
                    }
                }
                None
            };

        // Deref coercion: if the receiver is an auto-deref wrapper
        // (`Arc<T>` / `MutexGuard<T>`) and the field doesn't exist on the
        // wrapper itself, transparently rewrite `wrapper.field` as
        // `wrapper.get().field`. Avoids forcing every concurrency
        // program to call `.get()` explicitly.
        // Deref coercion: rewrite `wrapper.field` as `wrapper.get().field`
        // when the field doesn't exist on the wrapper. Synthesises the
        // MethodCall directly with `infer_method_call_return_type` to get
        // the substituted concrete inner type.
        if let Some(class_sym) = self.resolve_type_to_class_symbol(obj_expr.expr_type) {
            let field_on_wrapper = resolve_in_class(self, &class_sym, field_name).is_some();
            if !field_on_wrapper && self.is_auto_deref_wrapper_class(class_sym) {
                if let Some(get_sym) = self.find_wrapper_get_method(class_sym) {
                    let inner_type = self
                        .infer_method_call_return_type(get_sym, obj_expr.expr_type)
                        .ok()
                        .or_else(|| self.extract_wrapper_inner_type(obj_expr.expr_type))
                        .unwrap_or_else(|| self.context.type_table.borrow().dynamic_type());
                    let location = obj_expr.source_location;
                    let lifetime_id = obj_expr.lifetime_id;
                    obj_expr = TypedExpression {
                        kind: TypedExpressionKind::MethodCall {
                            receiver: Box::new(obj_expr),
                            method_symbol: get_sym,
                            arguments: Vec::new(),
                            type_arguments: Vec::new(),
                            is_optional: false,
                        },
                        expr_type: inner_type,
                        usage: VariableUsage::Borrow,
                        lifetime_id,
                        source_location: location,
                        metadata: ExpressionMetadata::default(),
                    };
                }
            }
        }

        // For field access, we need to look up the field symbol from the object's type
        // Create type parameter with deferred constraint resolution
        // But we can try to resolve it if the object is 'this'
        let field_symbol = match &obj_expr.kind {
            TypedExpressionKind::This { this_type: _ } => {
                // If accessing field on 'this', try to find it in current class
                if let Some(class_symbol) = self.context.class_context_stack.last() {
                    resolve_in_class(self, class_symbol, field_name)
                        .unwrap_or_else(|| self.context.symbol_table.create_field(field_name))
                } else {
                    self.context.symbol_table.create_field(field_name)
                }
            }
            TypedExpressionKind::Variable { symbol_id } => {
                // If accessing field on a variable/parameter, try to resolve from its type
                if let Some(symbol) = self.context.symbol_table.get_symbol(*symbol_id) {
                    if let Some(class_symbol) = self.resolve_type_to_class_symbol(symbol.type_id) {
                        resolve_in_class(self, &class_symbol, field_name)
                            .unwrap_or_else(|| self.context.symbol_table.create_field(field_name))
                    } else {
                        // Can't resolve object type to class, create placeholder
                        self.context.symbol_table.create_field(field_name)
                    }
                } else {
                    // Object symbol not found, create placeholder
                    self.context.symbol_table.create_field(field_name)
                }
            }
            _ => {
                // For other expression kinds (chained calls, etc.), try to resolve
                // from the expression's type to find methods/fields
                let obj_type = obj_expr.expr_type;
                if let Some(class_symbol) = self.resolve_type_to_class_symbol(obj_type) {
                    resolve_in_class(self, &class_symbol, field_name)
                        .unwrap_or_else(|| self.context.symbol_table.create_field(field_name))
                } else {
                    self.context.symbol_table.create_field(field_name)
                }
            }
        };

        // Method-as-value (bound method reference): if the resolved
        // symbol is a function (instance method on the receiver's
        // class), emit `MethodReference` rather than a `FieldAccess`.
        // This site is reached only for *standalone* `obj.method`
        // expressions — `lower_call_expression` routes
        // `obj.method(args)` through its own field-callee branch
        // before reaching here, so converting unconditionally here
        // can't accidentally break the invocation path.
        let is_method = self
            .context
            .symbol_table
            .get_symbol(field_symbol)
            .map(|s| s.kind == crate::tast::symbols::SymbolKind::Function)
            .unwrap_or(false);
        if is_method {
            // The expression's type is the method's function type,
            // which the symbol already carries (or Dynamic as a
            // safe fallback for unresolved generic methods).
            let method_fn_type = self
                .context
                .symbol_table
                .get_symbol(field_symbol)
                .map(|s| s.type_id)
                .filter(|tid| tid.is_valid())
                .unwrap_or_else(|| self.context.type_table.borrow().dynamic_type());
            let kind = TypedExpressionKind::MethodReference {
                receiver: Box::new(obj_expr),
                method_symbol: field_symbol,
            };
            let usage = VariableUsage::Borrow;
            let lifetime_id = self.assign_lifetime(&kind, &method_fn_type);
            let metadata = self.analyze_expression_metadata(&kind);
            return Ok(TypedExpression {
                expr_type: method_fn_type,
                kind,
                usage,
                lifetime_id,
                source_location: self.context.span_to_location(&expression.span),
                metadata,
            });
        }

        let kind = TypedExpressionKind::FieldAccess {
            object: Box::new(obj_expr),
            field_symbol,
            is_optional,
        };

        // Build the TypedExpression for the non-early-return path
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

    /// Check if function has @:arrayAccess metadata
    pub(crate) fn has_array_access_metadata(&self, metadata: &[parser::Metadata]) -> bool {
        metadata.iter().any(|m| m.name == "arrayAccess")
    }
}
