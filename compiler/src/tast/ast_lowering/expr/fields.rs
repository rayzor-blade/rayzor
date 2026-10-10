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
    pub(crate) fn forwarded_static_expression(&mut self, expression: &Expr) -> Option<Expr> {
        fn path(expression: &Expr) -> Option<Vec<String>> {
            match &expression.kind {
                ExprKind::Ident(name) => Some(vec![name.clone()]),
                ExprKind::Field {
                    expr,
                    field,
                    is_optional: false,
                } => {
                    let mut parts = path(expr)?;
                    parts.push(field.clone());
                    Some(parts)
                }
                _ => None,
            }
        }
        let ExprKind::Field {
            expr,
            field,
            is_optional,
        } = &expression.kind
        else {
            return None;
        };
        let parts = path(expr)?;
        let first = self.context.intern_string(parts.first()?);
        let base = self.resolve_symbol_in_scope_hierarchy(first);
        if base
            .and_then(|id| self.context.symbol_table.get_symbol(id))
            .is_some_and(|symbol| {
                matches!(
                    symbol.kind,
                    SymbolKind::Variable | SymbolKind::Parameter | SymbolKind::Field
                )
            })
        {
            return None;
        }
        let name = self.context.intern_string(parts.last()?);
        let symbol = if parts.len() == 1 {
            base.or_else(|| self.resolve_class_like_symbol_by_name(name))?
        } else {
            let package = parts[..parts.len() - 1]
                .iter()
                .map(|part| self.context.intern_string(part))
                .collect();
            let qualified = crate::tast::namespace::QualifiedPath::new(package, name);
            self.context.namespace_resolver.lookup_symbol(&qualified)?
        };
        let declared = self.context.symbol_table.get_symbol(symbol)?;
        if !matches!(declared.kind, SymbolKind::Abstract | SymbolKind::TypeAlias) {
            return None;
        }
        let original = self
            .resolve_type_to_class_symbol(declared.type_id)
            .unwrap_or(symbol);
        let member = self.context.intern_string(field);
        let index = self.static_sig_index.as_ref()?.clone();
        let mut owner = original;
        let mut seen = std::collections::BTreeSet::new();
        loop {
            if !seen.insert(owner) {
                return None;
            }
            let symbol = self.context.symbol_table.get_symbol(owner)?;
            let owner_type = symbol.type_id;
            let owner_name = self
                .context
                .string_interner
                .get(symbol.qualified_name.unwrap_or(symbol.name))?;
            if !index.borrow_mut().forwards_static(owner_name, field) {
                break;
            }
            if self.lookup_data_field(owner, member).is_some()
                || self
                    .resolve_declared_method_sig(owner, member, true)
                    .is_some()
            {
                break;
            }
            let underlying = {
                let table = self.context.type_table.borrow();
                match table.get(owner_type).map(|ty| &ty.kind) {
                    Some(TypeKind::Abstract {
                        underlying: Some(underlying),
                        ..
                    }) => *underlying,
                    _ => table.resolve_abstract_underlying(owner)?,
                }
            };
            owner = self.resolve_type_to_class_symbol(underlying)?;
        }
        if owner == original {
            return None;
        }
        let target = self.context.symbol_table.get_symbol(owner)?;
        let target = self
            .context
            .string_interner
            .get(target.qualified_name.unwrap_or(target.name))?;
        let mut parts = target.split('.');
        let mut object = Expr {
            kind: ExprKind::Ident(parts.next()?.to_string()),
            span: expr.span,
        };
        for part in parts {
            object = Expr {
                kind: ExprKind::Field {
                    expr: Box::new(object),
                    field: part.to_string(),
                    is_optional: false,
                },
                span: expr.span,
            };
        }
        Some(Expr {
            kind: ExprKind::Field {
                expr: Box::new(object),
                field: field.clone(),
                is_optional: *is_optional,
            },
            span: expression.span,
        })
    }

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
        let mut current = Some(class_sym);
        let mut visited = std::collections::BTreeSet::new();
        while let Some(class) = current {
            if !visited.insert(class) {
                break;
            }
            if let Some(field) = self
                .class_fields
                .get(&class)
                .and_then(|fields| fields.iter().find(|(name, _, _)| *name == field_name))
            {
                return Some(field.1);
            }
            current = self.parent_class_symbol(class);
        }
        None
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
        if let Some(forwarded) = self.forwarded_static_expression(expression) {
            return self.lower_expression(&forwarded);
        }
        if field == "new" {
            if let Some(value) = self.lower_constructor_value(expression, expr)? {
                return Ok(value);
            }
        }
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
                            // Any type a bare identifier would name: `haxe.Int64.make`
                            // reaches the abstract the way `Int64.make` does.
                            use crate::tast::symbols::SymbolKind;
                            if matches!(
                                symbol.kind,
                                SymbolKind::Class
                                    | SymbolKind::Enum
                                    | SymbolKind::Abstract
                                    | SymbolKind::Interface
                                    | SymbolKind::TypeAlias
                            ) {
                                // Return a reference to the type itself
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
                    } else if package_parts
                        .iter()
                        .any(|p| p.starts_with(|c: char| c.is_ascii_uppercase()))
                    {
                        // `pkg.Type.value`: the last "package" part is a type,
                        // so a shorter split names it with this as its field.
                        continue;
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
                    } else if !package_parts
                        .iter()
                        .any(|p| p.starts_with(|c: char| c.is_ascii_uppercase()))
                        && (package_parts.len() >= 2
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
                                )))
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
                    // A class declared but not yet compiled (an import cycle):
                    // nothing says what `field` is, so fail and let the import
                    // retry compile this file after the class.
                    if !self.class_fields.contains_key(&class_symbol)
                        && self
                            .resolve_class_method_symbol(class_symbol, field_name)
                            .is_none()
                    {
                        return Err(LoweringError::UnresolvedType {
                            type_name: class_name.clone(),
                            location: self.context.create_location_from_span(expression.span),
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
                if let Some(field) = this.lookup_data_field(*class_sym, name) {
                    return Some(field);
                }
                if let Some(methods) = this.class_methods.get(class_sym) {
                    if let Some((_, sym, _)) = methods.iter().find(|(n, _, _)| *n == name) {
                        return Some(*sym);
                    }
                }
                // An interface's methods live in its own scope, not these tables.
                let is_interface = this
                    .context
                    .symbol_table
                    .get_symbol(*class_sym)
                    .is_some_and(|s| s.kind == crate::tast::symbols::SymbolKind::Interface);
                if is_interface {
                    return this.resolve_class_method_symbol(*class_sym, name);
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
            TypedExpressionKind::This { this_type } => {
                let this_type = *this_type;
                let current = self.context.class_context_stack.last().copied();
                let in_abstract = current.is_some_and(|class| {
                    self.context
                        .symbol_table
                        .get_symbol(class)
                        .is_some_and(|s| s.kind == crate::tast::symbols::SymbolKind::Abstract)
                });
                if in_abstract {
                    // An abstract's `this` is its underlying value: that type's
                    // members, then static extensions, then the abstract's own.
                    let underlying = self
                        .resolve_type_to_class_symbol(this_type)
                        .filter(|class| Some(*class) != current)
                        .and_then(|class| resolve_in_class(self, &class, field_name));
                    match underlying {
                        Some(symbol) => symbol,
                        None if self
                            .find_static_extension_method(field_name, this_type)
                            .is_some() =>
                        {
                            self.context.symbol_table.create_field(field_name)
                        }
                        None => current
                            .and_then(|class| resolve_in_class(self, &class, field_name))
                            .unwrap_or_else(|| self.context.symbol_table.create_field(field_name)),
                    }
                } else {
                    // The current class, else the class `this` is.
                    current
                        .and_then(|class_symbol| resolve_in_class(self, &class_symbol, field_name))
                        .or_else(|| {
                            let class_symbol = self.resolve_type_to_class_symbol(this_type)?;
                            resolve_in_class(self, &class_symbol, field_name)
                        })
                        .unwrap_or_else(|| self.context.symbol_table.create_field(field_name))
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

        if self.static_through_abstract_instance(&obj_expr, field_symbol) && {
            let name = self.context.intern_string(field);
            self.find_static_extension_method(name, obj_expr.expr_type)
                .is_none()
        } {
            return Err(LoweringError::SemanticError {
                message: format!(
                    "Invalid call to static function {field} through abstract instance"
                ),
                location: self.context.span_to_location(&expression.span),
            });
        }

        // A dynamic method read as a value takes its current binding; a plain
        // override is the binding itself, as calls through it ignore the slot.
        if self.is_method_symbol(field_symbol)
            && self
                .resolve_type_to_class_symbol(obj_expr.expr_type)
                .is_some_and(|class| {
                    self.has_original_body(class, field) && !self.plainly_overridden(class, field)
                })
        {
            let read = self.dynamic_method_read(Some(expr), field, expression.span);
            return self.lower_expression(&read);
        }

        // Method-as-value (bound method reference): if the resolved
        // symbol is a function (instance method on the receiver's
        // class), emit `MethodReference` rather than a `FieldAccess`.
        // This site is reached only for *standalone* `obj.method`
        // expressions — `lower_call_expression` routes
        // `obj.method(args)` through its own field-callee branch
        // before reaching here, so converting unconditionally here
        // can't accidentally break the invocation path.
        let is_placeholder = self
            .context
            .symbol_table
            .get_symbol(field_symbol)
            .is_some_and(|s| {
                s.kind == crate::tast::symbols::SymbolKind::Field && !s.type_id.is_valid()
            });
        if is_placeholder && !is_optional && field != "new" {
            if let Some(value) =
                self.lower_resolve_macro_field(expression, expr, field, obj_expr.expr_type)?
            {
                return Ok(value);
            }
            if let Some(value) =
                self.lower_unresolved_method_value(expression, expr, field, &obj_expr)?
            {
                return Ok(value);
            }
        }
        let is_method = self
            .context
            .symbol_table
            .get_symbol(field_symbol)
            .map(|s| s.kind == crate::tast::symbols::SymbolKind::Function)
            .unwrap_or(false);
        if is_method {
            if let Some(value) =
                self.lower_interface_method_value(expression, expr, field, field_symbol, &obj_expr)?
            {
                return Ok(value);
            }
            // A static method read through a variable holding the class is
            // that class's static, not a method bound to the class value.
            let receiver_is_local = match &obj_expr.kind {
                TypedExpressionKind::Variable { symbol_id } => self
                    .context
                    .symbol_table
                    .get_symbol(*symbol_id)
                    .is_some_and(|s| {
                        matches!(
                            s.kind,
                            crate::tast::symbols::SymbolKind::Variable
                                | crate::tast::symbols::SymbolKind::Parameter
                        )
                    }),
                _ => false,
            };
            let static_owner = self
                .context
                .symbol_table
                .get_symbol(field_symbol)
                .filter(|s| s.is_static())
                .and_then(|_| self.resolve_type_to_class_symbol(obj_expr.expr_type));
            if let Some(owner) = static_owner.filter(|_| receiver_is_local) {
                if let Some(owner_expr) = self.class_path_expr(owner, expression.span) {
                    let direct = Expr {
                        kind: ExprKind::Field {
                            expr: Box::new(owner_expr),
                            field: field.to_string(),
                            is_optional: false,
                        },
                        span: expression.span,
                    };
                    return self.lower_expression(&direct);
                }
            }
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
        let expr_type = self.registered_alias_as_array(expr_type);
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
    /// `@:arrayAccess`, or its operator spelling `@:op([])`.
    /// `recv.m` on an interface receiver: `{ var r = recv; function(a, ..)
    /// return r.m(a, ..); }`, so the call dispatches through the interface.
    fn lower_interface_method_value(
        &mut self,
        expression: &Expr,
        receiver: &Expr,
        method: &str,
        method_symbol: SymbolId,
        lowered_receiver: &TypedExpression,
    ) -> LoweringResult<Option<TypedExpression>> {
        let is_interface = {
            let tt = self.context.type_table.borrow();
            let mut ty = lowered_receiver.expr_type;
            for _ in 0..4 {
                match tt.get(ty).map(|t| &t.kind) {
                    Some(TypeKind::TypeAlias { target_type, .. }) => ty = *target_type,
                    _ => break,
                }
            }
            matches!(
                tt.get(ty).map(|t| &t.kind),
                Some(TypeKind::Interface { .. })
            )
        };
        if !is_interface {
            return Ok(None);
        }
        let Some(param_types) = self.function_param_types_from_symbol(method_symbol) else {
            return Ok(None);
        };
        let returns_void = {
            let tt = self.context.type_table.borrow();
            let fn_ty = self
                .context
                .symbol_table
                .get_symbol(method_symbol)
                .map(|s| s.type_id);
            match fn_ty.and_then(|t| tt.get(t)).map(|t| &t.kind) {
                Some(TypeKind::Function { return_type, .. }) => {
                    matches!(tt.get(*return_type).map(|t| &t.kind), Some(TypeKind::Void))
                }
                _ => false,
            }
        };
        let span = expression.span;
        let at = |kind: ExprKind| Expr { kind, span };
        let recv = "__iface_recv".to_string();
        let names: Vec<String> = (0..param_types.len())
            .map(|i| format!("__iface_arg{i}"))
            .collect();
        let call = at(ExprKind::Call {
            expr: Box::new(at(ExprKind::Field {
                expr: Box::new(at(ExprKind::Ident(recv.clone()))),
                field: method.to_string(),
                is_optional: false,
            })),
            args: names
                .iter()
                .map(|n| at(ExprKind::Ident(n.clone())))
                .collect(),
        });
        let body = if returns_void {
            call
        } else {
            at(ExprKind::Return(Some(Box::new(call))))
        };
        let literal = at(ExprKind::Function(Function {
            name: String::new(),
            type_params: Vec::new(),
            params: names
                .iter()
                .map(|n| FunctionParam {
                    meta: Vec::new(),
                    name: n.clone(),
                    type_hint: None,
                    optional: false,
                    rest: false,
                    default_value: None,
                    span,
                })
                .collect(),
            return_type: None,
            body: Some(Box::new(body)),
            span,
        }));
        let block = at(ExprKind::Block(vec![
            BlockElement::Expr(at(ExprKind::Var {
                name: recv,
                type_hint: None,
                expr: Some(Box::new(receiver.clone())),
            })),
            BlockElement::Expr(literal),
        ]));
        self.expected_lambda_params_stack.push(Some(param_types));
        let lowered = self.lower_expression(&block);
        self.expected_lambda_params_stack.pop();
        lowered.map(Some)
    }

    /// `recv.m` as a value where only the call form `recv.m(..)` resolves: a
    /// method of a builtin Array or String, a static extension, or a method
    /// an abstract forwards to its underlying class.
    fn lower_unresolved_method_value(
        &mut self,
        expression: &Expr,
        receiver: &Expr,
        method: &str,
        lowered_receiver: &TypedExpression,
    ) -> LoweringResult<Option<TypedExpression>> {
        let method_name = self.context.intern_string(method);
        let dynamic = self.context.type_table.borrow().dynamic_type();
        let receiver_ty = {
            let tt = self.context.type_table.borrow();
            let mut ty = lowered_receiver.expr_type;
            for _ in 0..4 {
                match tt.get(ty).map(|t| &t.kind) {
                    Some(TypeKind::TypeAlias { target_type, .. }) => ty = *target_type,
                    _ => break,
                }
            }
            ty
        };
        let builtin = {
            let tt = self.context.type_table.borrow();
            match tt.get(receiver_ty).map(|t| &t.kind) {
                Some(TypeKind::Array { element_type }) => Some(("Array", Some(*element_type))),
                Some(TypeKind::String) => Some(("String", None)),
                _ => None,
            }
        };
        let builtin_sig = builtin.and_then(|(class, element)| {
            let index = self.static_sig_index.as_ref()?;
            let resolver: &super::namespace::NamespaceResolver = self.context.namespace_resolver;
            let resolve_file = |q: &str| resolver.resolve_qualified_path_to_file_force(q);
            let sig = index
                .borrow_mut()
                .resolve(class, method, false, &resolve_file)?;
            Some((sig, element))
        });
        if let Some((sig, element)) = builtin_sig {
            // The class's `T` is the element type; the method's own type
            // parameters are left Dynamic.
            let mut bindings = BTreeMap::new();
            bindings.insert(self.context.intern_string("T"), element.unwrap_or(dynamic));
            for param in &sig.type_params {
                bindings.insert(self.context.intern_string(&param.name), dynamic);
            }
            let params = sig
                .params
                .iter()
                .enumerate()
                .map(|(i, _)| (dynamic, sig.optional.get(i).copied().unwrap_or(false)))
                .collect();
            let returns_void = matches!(
                &sig.return_type,
                Some(parser::Type::Path { path, .. }) if path.name == "Void"
            );
            let declared = DeclaredSig {
                params: sig.params.clone(),
                return_type: sig.return_type.clone(),
                bindings,
            };
            return self
                .method_value_closure(
                    expression,
                    Some(receiver),
                    method,
                    params,
                    returns_void,
                    Some(declared),
                )
                .map(Some);
        }

        // A static extension in scope: the call form passes the receiver first.
        if let Some((class, ext)) =
            self.find_static_extension_method(method_name, lowered_receiver.expr_type)
        {
            let Some(types) = self.function_param_types_from_symbol(ext) else {
                return Ok(None);
            };
            let optional = self
                .resolve_declared_method_sig(class, method_name, true)
                .map(|sig| sig.optional)
                .unwrap_or_default();
            let params = types
                .iter()
                .enumerate()
                .skip(1)
                .map(|(i, t)| (*t, optional.get(i).copied().unwrap_or(false)))
                .collect();
            let returns_void = self.method_returns_void(ext);
            return self
                .method_value_closure(
                    expression,
                    Some(receiver),
                    method,
                    params,
                    returns_void,
                    None,
                )
                .map(Some);
        }

        // An enum's constructor read through a value of the enum type: an
        // enum value has no fields, so the receiver is the enum itself.
        let enum_symbol = match self
            .context
            .type_table
            .borrow()
            .get(receiver_ty)
            .map(|t| &t.kind)
        {
            Some(TypeKind::Enum { symbol_id, .. }) => Some(*symbol_id),
            _ => None,
        };
        if let Some(enum_symbol) = enum_symbol {
            let is_variant = self
                .context
                .symbol_table
                .get_symbol(enum_symbol)
                .and_then(|e| {
                    self.context
                        .symbol_table
                        .lookup_symbol(e.scope_id, method_name)
                })
                .is_some_and(|v| v.kind == crate::tast::symbols::SymbolKind::EnumVariant);
            if is_variant {
                if let Some(enum_expr) = self.class_path_expr(enum_symbol, expression.span) {
                    let direct = Expr {
                        kind: ExprKind::Field {
                            expr: Box::new(enum_expr),
                            field: method.to_string(),
                            is_optional: false,
                        },
                        span: expression.span,
                    };
                    return self.lower_expression(&direct).map(Some);
                }
            }
        }

        // A method of the class an abstract wraps (`@:forward`).
        let underlying = {
            let tt = self.context.type_table.borrow();
            match tt.get(receiver_ty).map(|t| &t.kind) {
                Some(TypeKind::Abstract {
                    symbol_id,
                    underlying,
                    ..
                }) => underlying.or_else(|| tt.resolve_abstract_underlying(*symbol_id)),
                _ => None,
            }
        };
        let forwarded = underlying
            .and_then(|u| self.resolve_type_to_class_symbol(u))
            .and_then(|class| self.resolve_class_method_symbol(class, method_name));
        if let Some(target) = forwarded {
            let Some(types) = self.function_param_types_from_symbol(target) else {
                return Ok(None);
            };
            let params = types.into_iter().map(|t| (t, false)).collect();
            let returns_void = self.method_returns_void(target);
            return self
                .method_value_closure(
                    expression,
                    Some(receiver),
                    method,
                    params,
                    returns_void,
                    None,
                )
                .map(Some);
        }
        Ok(None)
    }

    /// The expression naming a class by its qualified path (`a.b.C`).
    fn class_path_expr(&self, class: SymbolId, span: parser::Span) -> Option<Expr> {
        let sym = self.context.symbol_table.get_symbol(class)?;
        let path = sym
            .qualified_name
            .or(Some(sym.name))
            .and_then(|n| self.context.string_interner.get(n))?
            .to_string();
        let mut parts = path.split('.');
        let first = parts.next()?;
        let mut expr = Expr {
            kind: ExprKind::Ident(first.to_string()),
            span,
        };
        for part in parts {
            expr = Expr {
                kind: ExprKind::Field {
                    expr: Box::new(expr),
                    field: part.to_string(),
                    is_optional: false,
                },
                span,
            };
        }
        Some(expr)
    }

    fn method_returns_void(&self, method: SymbolId) -> bool {
        let tt = self.context.type_table.borrow();
        let fn_ty = self
            .context
            .symbol_table
            .get_symbol(method)
            .map(|s| s.type_id);
        match fn_ty.and_then(|t| tt.get(t)).map(|t| &t.kind) {
            Some(TypeKind::Function { return_type, .. }) => {
                matches!(tt.get(*return_type).map(|t| &t.kind), Some(TypeKind::Void))
            }
            _ => false,
        }
    }

    /// `{ var r = recv; function(a, ..) return r.m(a, ..); }`, or with no
    /// receiver `function(a, ..) return m(a, ..)`. An optional parameter left
    /// null drops it and those after it from the call, so the method applies
    /// its own defaults.
    pub(crate) fn method_value_closure(
        &mut self,
        expression: &Expr,
        receiver: Option<&Expr>,
        method: &str,
        params: Vec<(TypeId, bool)>,
        returns_void: bool,
        declared: Option<DeclaredSig>,
    ) -> LoweringResult<TypedExpression> {
        let span = expression.span;
        let at = |kind: ExprKind| Expr { kind, span };
        let recv = format!("__method_recv{}", self.context.next_scope_id());
        let names: Vec<String> = (0..params.len())
            .map(|i| format!("__method_arg{i}"))
            .collect();
        let callee = match receiver {
            Some(_) => at(ExprKind::Field {
                expr: Box::new(at(ExprKind::Ident(recv.clone()))),
                field: method.to_string(),
                is_optional: false,
            }),
            None => at(ExprKind::Ident(method.to_string())),
        };
        let call = |n: usize| {
            at(ExprKind::Call {
                expr: Box::new(callee.clone()),
                args: names[..n]
                    .iter()
                    .map(|a| at(ExprKind::Ident(a.clone())))
                    .collect(),
            })
        };
        let mut body = call(names.len());
        for (i, (_, optional)) in params.iter().enumerate().rev() {
            if !*optional {
                continue;
            }
            body = at(ExprKind::If {
                cond: Box::new(at(ExprKind::Binary {
                    left: Box::new(at(ExprKind::Ident(names[i].clone()))),
                    op: parser::BinaryOp::Eq,
                    right: Box::new(at(ExprKind::Null)),
                })),
                then_branch: Box::new(call(i)),
                else_branch: Some(Box::new(body)),
            });
        }
        if !returns_void {
            body = at(ExprKind::Return(Some(Box::new(body))));
        }
        let literal = at(ExprKind::Function(Function {
            name: String::new(),
            type_params: Vec::new(),
            params: names
                .iter()
                .zip(&params)
                .enumerate()
                .map(|(i, (n, (_, optional)))| FunctionParam {
                    meta: Vec::new(),
                    name: n.clone(),
                    type_hint: declared
                        .as_ref()
                        .and_then(|d| d.params.get(i).cloned().flatten()),
                    optional: *optional,
                    rest: false,
                    default_value: None,
                    span,
                })
                .collect(),
            return_type: declared.as_ref().and_then(|d| d.return_type.clone()),
            body: Some(Box::new(body)),
            span,
        }));
        let block = match receiver {
            Some(receiver) => at(ExprKind::Block(vec![
                BlockElement::Expr(at(ExprKind::Var {
                    name: recv,
                    type_hint: None,
                    expr: Some(Box::new(receiver.clone())),
                })),
                BlockElement::Expr(literal),
            ])),
            None => literal,
        };
        let bound = declared.map(|d| d.bindings);
        let has_bindings = bound.is_some();
        if let Some(bindings) = bound {
            self.context.push_type_parameters(bindings);
            self.expected_lambda_params_stack.push(None);
        } else {
            self.expected_lambda_params_stack
                .push(Some(params.iter().map(|(t, _)| *t).collect()));
        }
        let lowered = self.lower_expression(&block);
        self.expected_lambda_params_stack.pop();
        if has_bindings {
            self.context.pop_type_parameters();
        }
        lowered
    }

    /// `C.new` as a value: `function(a, ..) return new C(a, ..)`, its
    /// parameters typed from the constructor's.
    fn lower_constructor_value(
        &mut self,
        expression: &Expr,
        target: &Expr,
    ) -> LoweringResult<Option<TypedExpression>> {
        let ExprKind::Ident(name) = &target.kind else {
            return Ok(None);
        };
        let interned = self.context.intern_string(name);
        let Some(symbol) = self.resolve_class_like_symbol_by_name(interned) else {
            return Ok(None);
        };
        let Some((kind, scope)) = self
            .context
            .symbol_table
            .get_symbol(symbol)
            .map(|s| (s.kind, s.scope_id))
        else {
            return Ok(None);
        };
        let ctor = match kind {
            crate::tast::symbols::SymbolKind::Class => self
                .class_constructor_symbols
                .get(&symbol)
                .copied()
                .or_else(|| self.context.symbol_table.get_class_constructor(symbol)),
            crate::tast::symbols::SymbolKind::Abstract => {
                let new_name = self.context.intern_string("new");
                self.context
                    .symbol_table
                    .lookup_symbol(scope, new_name)
                    .filter(|s| s.kind == crate::tast::symbols::SymbolKind::Function)
                    .map(|s| s.id)
            }
            _ => None,
        };
        if ctor.is_none()
            && kind == crate::tast::symbols::SymbolKind::Class
            && self.class_lacks_constructor(symbol)
        {
            let path = self
                .context
                .symbol_table
                .display_type_path(symbol, self.context.string_interner)
                .unwrap_or_else(|| name.clone());
            return Err(LoweringError::SemanticError {
                message: format!("{path} does not have a constructor"),
                location: self.context.create_location_from_span(expression.span),
            });
        }
        // A class without a constructor of its own takes its parent's parameters.
        let ctor = ctor.or_else(|| {
            if kind != crate::tast::symbols::SymbolKind::Class {
                return None;
            }
            let mut seen = std::collections::BTreeSet::new();
            let mut current = self.parent_class_symbol(symbol);
            while let Some(class) = current {
                if !seen.insert(class) {
                    break;
                }
                let own = self
                    .class_constructor_symbols
                    .get(&class)
                    .copied()
                    .or_else(|| self.context.symbol_table.get_class_constructor(class));
                if own.is_some() {
                    return own;
                }
                current = self.parent_class_symbol(class);
            }
            None
        });
        let Some(param_types) = ctor.and_then(|c| self.function_param_types_from_symbol(c)) else {
            return Ok(None);
        };
        let span = expression.span;
        let at = |kind: ExprKind| Expr { kind, span };
        let names: Vec<String> = (0..param_types.len())
            .map(|i| format!("__ctor_arg{i}"))
            .collect();
        let construct = at(ExprKind::New {
            type_path: parser::TypePath {
                package: Vec::new(),
                name: name.clone(),
                sub: None,
            },
            params: Vec::new(),
            args: names
                .iter()
                .map(|n| at(ExprKind::Ident(n.clone())))
                .collect(),
        });
        let literal = at(ExprKind::Function(Function {
            name: String::new(),
            type_params: Vec::new(),
            params: names
                .iter()
                .map(|n| FunctionParam {
                    meta: Vec::new(),
                    name: n.clone(),
                    type_hint: None,
                    optional: false,
                    rest: false,
                    default_value: None,
                    span,
                })
                .collect(),
            return_type: None,
            body: Some(Box::new(at(ExprKind::Return(Some(Box::new(construct)))))),
            span,
        }));
        self.expected_lambda_params_stack.push(Some(param_types));
        let lowered = self.lower_expression(&literal);
        self.expected_lambda_params_stack.pop();
        lowered.map(Some)
    }

    /// No constructor of its own, none inherited, and none its declarations
    /// could supply once lowered.
    fn class_lacks_constructor(&self, class: SymbolId) -> bool {
        let has_ctor = |c: SymbolId| {
            self.class_constructor_symbols.contains_key(&c)
                || self.context.symbol_table.get_class_constructor(c).is_some()
        };
        let mut seen = std::collections::BTreeSet::new();
        let mut current = Some(class);
        while let Some(c) = current {
            if !seen.insert(c) || has_ctor(c) {
                return false;
            }
            current = self.parent_class_symbol(c);
        }
        let Some(index) = self.static_sig_index.as_ref() else {
            return false;
        };
        let Some(name) = self.context.symbol_table.get_symbol(class).and_then(|s| {
            self.context
                .string_interner
                .get(s.qualified_name.unwrap_or(s.name))
        }) else {
            return false;
        };
        index.borrow_mut().lacks_constructor(name)
    }

    pub(crate) fn has_array_access_metadata(&self, metadata: &[parser::Metadata]) -> bool {
        metadata.iter().any(|m| {
            m.name == "arrayAccess"
                || (m.name == "op"
                    && matches!(
                        m.params.first().map(|p| &p.kind),
                        Some(parser::ExprKind::Array(items)) if items.is_empty()
                    ))
        })
    }
}

/// A method's declared parameter and return annotations, with the type
/// parameters they mention bound.
pub(crate) struct DeclaredSig {
    params: Vec<Option<parser::Type>>,
    return_type: Option<parser::Type>,
    bindings: BTreeMap<InternedString, TypeId>,
}
