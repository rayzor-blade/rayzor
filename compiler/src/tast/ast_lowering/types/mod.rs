//! Type annotations and type paths.

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
    /// Lower a type annotation
    pub(crate) fn lower_type(&mut self, type_annotation: &Type) -> LoweringResult<TypeId> {
        match type_annotation {
            Type::Path { path, params, .. } => {
                let name = if path.package.is_empty() {
                    path.name.clone()
                } else {
                    format!("{}.{}", path.package.join("."), path.name)
                };

                // Haxe Type Resolution Order:

                // 1. Check if it's a type parameter (in generic contexts)
                let interned_name = self.context.intern_string(&name);
                if let Some(type_param) = self.context.resolve_type_parameter(interned_name) {
                    return Ok(type_param);
                }

                // 2. Try to resolve as a built-in type (covers basic types first)
                // IMPORTANT: Skip "Array" when type params are present (e.g., Array<Body>).
                // resolve_builtin_type("Array") returns Array<Dynamic>, discarding params.
                // Instead, resolve the element type and create Array<ElementType>.
                if name == "Array" && !params.is_empty() {
                    let element_type = self.lower_type(&params[0])?;
                    return Ok(self
                        .context
                        .type_table
                        .borrow_mut()
                        .create_array_type(element_type));
                }
                if name == "Null" && params.len() == 1 {
                    let inner_type = self.lower_type(&params[0])?;
                    return Ok(self
                        .context
                        .type_table
                        .borrow_mut()
                        .create_optional_type(inner_type));
                }
                if let Some(builtin_type) = self.resolve_builtin_type(&name) {
                    return Ok(builtin_type);
                }

                let interned_name = self.context.intern_string(&name);

                // 3. Module-level types (current module/file scope)
                // 4. Imported types (already registered during import processing)
                // 5. Top-level and standard library types (already in root scope)

                // IMPORTANT: First try to resolve through the import system.
                // This ensures we get the symbol with the correct qualified_name (e.g., "rayzor.Bytes")
                // rather than a duplicate symbol that may have been created without package context.
                let import_candidates = self.context.import_resolver.resolve_type(
                    interned_name,
                    self.context.current_scope,
                    self.context.namespace_resolver,
                );

                let import_resolved_symbol = import_candidates
                    .first()
                    .and_then(|qualified_path| {
                        self.context
                            .namespace_resolver
                            .lookup_symbol(qualified_path)
                    })
                    .and_then(|sym_id| self.context.symbol_table.get_symbol(sym_id))
                    .map(|s| (s.id, s.kind.clone()));

                // If import resolution found a symbol, use it. Otherwise fall back to scope lookup.
                let symbol_info = import_resolved_symbol
                    .or_else(|| {
                        self.context
                            .symbol_table
                            .lookup_symbol(self.context.current_scope, interned_name)
                            .or_else(|| {
                                self.context.symbol_table.lookup_symbol(
                                    ScopeId::first(), // Root scope contains imports and top-level types
                                    interned_name,
                                )
                            })
                            .map(|s| (s.id, s.kind.clone()))
                    })
                    // For qualified names (e.g. "rayzor.Bytes"), also try looking up via
                    // the namespace resolver using package path + short name
                    .or_else(|| {
                        if name.contains('.') {
                            let parts: Vec<&str> = name.rsplitn(2, '.').collect();
                            if parts.len() == 2 {
                                let class_name = parts[0]; // "Bytes"
                                let package_str = parts[1]; // "rayzor"
                                let package_parts: Vec<InternedString> = package_str
                                    .split('.')
                                    .map(|p| self.context.intern_string(p))
                                    .collect();
                                let class_interned = self.context.intern_string(class_name);
                                let qp = crate::tast::namespace::QualifiedPath::new(
                                    package_parts,
                                    class_interned,
                                );
                                if let Some(sym_id) =
                                    self.context.namespace_resolver.lookup_symbol(&qp)
                                {
                                    if let Some(sym) = self.context.symbol_table.get_symbol(sym_id)
                                    {
                                        return Some((sym.id, sym.kind.clone()));
                                    }
                                }
                                // Also try short name lookup in root scope
                                if let Some(sym) = self
                                    .context
                                    .symbol_table
                                    .lookup_symbol(ScopeId::first(), class_interned)
                                {
                                    // Verify the symbol's qualified name matches
                                    if let Some(qn) = sym.qualified_name {
                                        if let Some(qn_str) = self.context.string_interner.get(qn) {
                                            if qn_str == name {
                                                return Some((sym.id, sym.kind.clone()));
                                            }
                                        }
                                    }
                                    // If no qualified name set, accept it as a match
                                    // (extern classes may not have qualified names set)
                                    return Some((sym.id, sym.kind.clone()));
                                }
                            }
                            None
                        } else {
                            None
                        }
                    });

                if let Some((symbol_id, symbol_kind)) = symbol_info {
                    // Process type arguments if present (now the symbol borrow is dropped)
                    let type_arg_ids = if !params.is_empty() {
                        let mut result = Vec::new();
                        for arg in params {
                            result.push(self.lower_type(arg)?);
                        }
                        result
                    } else {
                        Vec::new()
                    };

                    // Create appropriate type based on symbol kind
                    match symbol_kind {
                        crate::tast::SymbolKind::Class => {
                            // `@:multiType` resolution when `Map` resolved to a
                            // stale `Class` placeholder. The root `Map.hx`
                            // typedef (`typedef Map<K,V> = haxe.ds.Map<K,V>`)
                            // is pre-registered as a `SymbolKind::Class`
                            // placeholder, and that kind is never upgraded to
                            // `TypeAlias` once its body is lowered. When a user
                            // module is compiled BEFORE `haxe/ds/Map.hx` (the
                            // case for cross-module imports), the only `Map`
                            // symbol visible is this Class placeholder, so the
                            // TypeAlias / Abstract multiType arms below are
                            // never reached and a `Map<K,V>` field annotation
                            // keeps the abstract type. Every `get`/`set` then
                            // dispatches to the default `BalancedTree`
                            // implementer (which fails to monomorphize
                            // cross-module → trap stub → SIGILL). Resolve the
                            // multiType here too so the field/param/return type
                            // becomes the concrete container regardless of the
                            // symbol's (possibly stale) kind or compile order.
                            if path.name == "Map"
                                && (path.package.is_empty()
                                    || path.package == vec!["haxe".to_string(), "ds".to_string()])
                                && type_arg_ids.len() == 2
                            {
                                if let Some(resolved) =
                                    self.resolve_multitype_map_to_concrete(&type_arg_ids)
                                {
                                    return Ok(resolved);
                                }
                            }
                            // Check if this class already has a type from pre-registration
                            if let Some(symbol) = self.context.symbol_table.get_symbol(symbol_id) {
                                if symbol.type_id.is_valid() && type_arg_ids.is_empty() {
                                    // Use the existing type from pre-registration
                                    return Ok(symbol.type_id);
                                }
                            }
                            Ok(self
                                .context
                                .type_table
                                .borrow_mut()
                                .create_class_type(symbol_id, type_arg_ids))
                        }
                        crate::tast::SymbolKind::Interface => {
                            // Check if this interface already has a type from pre-registration
                            if let Some(symbol) = self.context.symbol_table.get_symbol(symbol_id) {
                                if symbol.type_id.is_valid() && type_arg_ids.is_empty() {
                                    // Use the existing type from pre-registration
                                    return Ok(symbol.type_id);
                                }
                            }
                            Ok(self
                                .context
                                .type_table
                                .borrow_mut()
                                .create_interface_type(symbol_id, type_arg_ids))
                        }
                        crate::tast::SymbolKind::Enum => Ok(self
                            .context
                            .type_table
                            .borrow_mut()
                            .create_enum_type(symbol_id, type_arg_ids)),
                        crate::tast::SymbolKind::TypeAlias => {
                            // `@:multiType` resolution through a typedef. The
                            // standard library exposes `Map` two ways:
                            //   - `haxe.ds.Map<K, V>` — the `@:multiType`
                            //     abstract, resolved in the Abstract arm below.
                            //   - `Map<K, V> = haxe.ds.Map<K, V>` — a typedef
                            //     in the root `Map.hx`.
                            // Same-module the bare `Map` binds to the abstract;
                            // cross-module it binds to the typedef and lands
                            // HERE. Without resolving the multiType through the
                            // alias, the field/return/param keeps an abstract
                            // `Map` type, every member call (`get`/`set`)
                            // dispatches to the default `BalancedTree`
                            // implementer (which fails to monomorphize
                            // cross-module → trap stub → SIGILL), and the
                            // construction never routes through `StringMap`.
                            if path.name == "Map"
                                && (path.package.is_empty()
                                    || path.package == vec!["haxe".to_string(), "ds".to_string()])
                                && type_arg_ids.len() == 2
                            {
                                if let Some(resolved) =
                                    self.resolve_multitype_map_to_concrete(&type_arg_ids)
                                {
                                    return Ok(resolved);
                                }
                            }
                            // For type aliases, we need to get the target type
                            let target_type = type_resolution::resolve_type_alias(
                                self.context.type_table,
                                self.context.symbol_table,
                                symbol_id,
                            );
                            Ok(self.context.type_table.borrow_mut().create_type(
                                crate::tast::core::TypeKind::TypeAlias {
                                    symbol_id,
                                    target_type,
                                    type_args: type_arg_ids,
                                },
                            ))
                        }
                        crate::tast::SymbolKind::Abstract => {
                            // `@:multiType` resolution at type-annotation level.
                            // `Map<K, V>` is the only multiType abstract today;
                            // declaring `var foo:Map<String, V>` (as a local, a
                            // field, a parameter, or a return) must immediately
                            // resolve to the concrete container per K so that
                            // every downstream operation (construction,
                            // dispatch, field load/store) sees the same type.
                            // Resolving only at the New site, as the previous
                            // fix did, fixed construction but left dispatch
                            // through Map-typed fields silently broken — the
                            // MIR lowerer ended up elided the call entirely.
                            if path.name == "Map"
                                && (path.package.is_empty()
                                    || path.package == vec!["haxe".to_string(), "ds".to_string()])
                                && type_arg_ids.len() == 2
                            {
                                if let Some(resolved) =
                                    self.resolve_multitype_map_to_concrete(&type_arg_ids)
                                {
                                    return Ok(resolved);
                                }
                            }
                            // Reuse the type the declaration interned, which
                            // carries the underlying. Building a fresh one here
                            // discards it at every annotation site, and a
                            // missing underlying lowers to a 32-bit slot -- so
                            // a static of an abstract over any reference type
                            // gets half a pointer and faults on first use, and
                            // one over Float loses everything after the point.
                            // Mirrors the Class and Interface arms above.
                            if let Some(symbol) = self.context.symbol_table.get_symbol(symbol_id) {
                                if symbol.type_id.is_valid() && type_arg_ids.is_empty() {
                                    // An abstract over Dynamic is the exception.
                                    // Dynamic is pointer-shaped, but a scalar
                                    // assigned into one is not boxed on the way
                                    // in, so honouring the underlying here hands
                                    // the backend a raw integer to dereference.
                                    // Leave those on the old representation until
                                    // that assignment boxes.
                                    let over_dynamic = {
                                        let tt = self.context.type_table.borrow();
                                        tt.get(symbol.type_id)
                                            .and_then(|ti| match &ti.kind {
                                                crate::tast::core::TypeKind::Abstract {
                                                    underlying: Some(u),
                                                    ..
                                                } => tt.get(*u).map(|u| {
                                                    matches!(
                                                        u.kind,
                                                        crate::tast::core::TypeKind::Dynamic
                                                    )
                                                }),
                                                _ => None,
                                            })
                                            .unwrap_or(false)
                                    };
                                    if !over_dynamic {
                                        return Ok(symbol.type_id);
                                    }
                                }
                            }
                            Ok(self.context.type_table.borrow_mut().create_type(
                                crate::tast::core::TypeKind::Abstract {
                                    symbol_id,
                                    underlying: None,
                                    type_args: type_arg_ids,
                                },
                            ))
                        }
                        _ => {
                            // For other symbol kinds, return dynamic type
                            Ok(self.context.type_table.borrow().dynamic_type())
                        }
                    }
                } else {
                    // Symbol not found, this might be a forward reference
                    // Create a placeholder type and defer resolution
                    let placeholder_type = self.context.type_table.borrow_mut().create_type(
                        crate::tast::core::TypeKind::Placeholder {
                            name: interned_name,
                        },
                    );

                    // Record this for later resolution
                    self.resolution_state
                        .deferred_resolutions
                        .push(DeferredTypeResolution {
                            type_name: name.clone(),
                            location: self.context.create_location(),
                            type_params: params.iter().map(|_| "T".to_string()).collect(), // TODO: extract actual param names
                            target_type_id: placeholder_type,
                        });

                    Ok(placeholder_type)
                }
            }
            Type::Function { params, ret, .. } => {
                let param_types = params
                    .iter()
                    .map(|param| self.lower_type(param))
                    .collect::<Result<Vec<_>, _>>()?;

                let return_type_id = self.lower_type(ret)?;

                // Create function type with default effects. `is_send = true`
                // is the right default here — this branch builds a function
                // type from a declaration (no capture analysis available),
                // and bare function declarations are Send-by-default.
                let effects = crate::tast::core::FunctionEffects {
                    can_throw: false,
                    is_async: false,
                    is_pure: false,
                    memory_effects: crate::tast::core::MemoryEffects::None,
                    is_send: true,
                };

                Ok(self.context.type_table.borrow_mut().create_type(
                    crate::tast::core::TypeKind::Function {
                        params: param_types,
                        return_type: return_type_id,
                        effects,
                    },
                ))
            }
            Type::Anonymous { fields, .. } => {
                // Create proper anonymous type with fields
                let mut anonymous_fields = Vec::new();
                for field in fields {
                    let field_type_id = self.lower_type(&field.type_hint)?;
                    let field_name = self.context.intern_string(&field.name);
                    anonymous_fields.push(crate::tast::core::AnonymousField {
                        name: field_name,
                        type_id: field_type_id,
                        is_public: true, // Anonymous fields are typically public
                        optional: field.optional,
                    });
                }

                Ok(self.context.type_table.borrow_mut().create_type(
                    crate::tast::core::TypeKind::Anonymous {
                        fields: anonymous_fields,
                    },
                ))
            }
            Type::Optional { inner, .. } => {
                let inner_type_id = self.lower_type(inner)?;
                Ok(self
                    .context
                    .type_table
                    .borrow_mut()
                    .create_optional_type(inner_type_id))
            }
            Type::Parenthesis { inner, .. } => {
                // Just unwrap parentheses
                self.lower_type(inner)
            }
            Type::Intersection { left, right, .. } => {
                let left_type_id = self.lower_type(left)?;
                let right_type_id = self.lower_type(right)?;

                // If both sides resolve to Anonymous types, merge their fields
                // into a single Anonymous type (right side wins on name conflicts)
                let merged = {
                    let type_table = self.context.type_table.borrow();
                    let left_resolved = Self::resolve_alias_chain(&type_table, left_type_id);
                    let right_resolved = Self::resolve_alias_chain(&type_table, right_type_id);
                    let left_anon = type_table.get(left_resolved).and_then(|t| {
                        if let TypeKind::Anonymous { fields } = &t.kind {
                            Some(fields.clone())
                        } else {
                            None
                        }
                    });
                    let right_anon = type_table.get(right_resolved).and_then(|t| {
                        if let TypeKind::Anonymous { fields } = &t.kind {
                            Some(fields.clone())
                        } else {
                            None
                        }
                    });
                    match (left_anon, right_anon) {
                        (Some(left_fields), Some(right_fields)) => {
                            // Merge: start with left fields, override/add right fields
                            let mut merged_fields = left_fields;
                            for rf in right_fields {
                                if let Some(existing) =
                                    merged_fields.iter_mut().find(|f| f.name == rf.name)
                                {
                                    *existing = rf;
                                } else {
                                    merged_fields.push(rf);
                                }
                            }
                            Some(merged_fields)
                        }
                        _ => None,
                    }
                };

                if let Some(merged_fields) = merged {
                    Ok(self
                        .context
                        .type_table
                        .borrow_mut()
                        .create_type(TypeKind::Anonymous {
                            fields: merged_fields,
                        }))
                } else {
                    Ok(self
                        .context
                        .type_table
                        .borrow_mut()
                        .create_type(TypeKind::Intersection {
                            types: vec![left_type_id, right_type_id],
                        }))
                }
            }
            Type::Wildcard { .. } => {
                // Wildcard types are used in type parameters, return Unknown type
                Ok(self.context.type_table.borrow().unknown_type())
            }
        }
    }

    pub(crate) fn resolve_type_path(
        &mut self,
        type_path: &parser::TypePath,
    ) -> LoweringResult<TypeId> {
        // Try to resolve using the import resolver first — user imports take priority
        // over top-level stdlib builtins (Array, Map, etc.)
        let name_interned = self.context.string_interner.intern(&type_path.name);
        let candidates = self.context.import_resolver.resolve_type(
            name_interned,
            self.context.current_scope,
            self.context.namespace_resolver,
        );

        if !candidates.is_empty() {
            // Use the first candidate (in a full implementation, we'd handle ambiguity)
            let qualified_path = &candidates[0];
            if let Some(symbol_id) = self
                .context
                .namespace_resolver
                .lookup_symbol(qualified_path)
            {
                if let Some(symbol) = self.context.symbol_table.get_symbol(symbol_id) {
                    let sym_id = symbol.id;
                    let type_id = symbol.type_id;
                    return Ok(self.ensure_symbol_has_class_type(sym_id, type_id));
                }
            }
        }

        // Only check top-level stdlib builtins when no import candidates exist.
        // This ensures user-defined classes with the same name as builtins (e.g., Map)
        // take priority when explicitly imported.
        if candidates.is_empty() && type_path.package.is_empty() && type_path.sub.is_none() {
            if let Some(builtin_type) = self.resolve_builtin_type(&type_path.name) {
                return Ok(builtin_type);
            }
        }

        // If not found through imports, try direct resolution
        let qualified_path = if type_path.package.is_empty() {
            // Try to find in current package first
            if let Some(current_package) = self.context.current_package {
                let package_segments = self
                    .context
                    .namespace_resolver
                    .find_symbols_by_name(name_interned, current_package);
                if let Some((_, symbol_id)) = package_segments.first() {
                    if let Some(symbol) = self.context.symbol_table.get_symbol(*symbol_id) {
                        let sym_id = symbol.id;
                        let type_id = symbol.type_id;
                        return Ok(self.ensure_symbol_has_class_type(sym_id, type_id));
                    }
                }
            }

            // Otherwise, treat as a simple name
            super::namespace::QualifiedPath::simple(name_interned)
        } else {
            // Create a qualified path from the package
            let package_path: Vec<_> = type_path
                .package
                .iter()
                .map(|s| self.context.string_interner.intern(s))
                .collect();
            super::namespace::QualifiedPath::new(package_path, name_interned)
        };

        // Try to resolve from namespace
        if let Some(symbol_id) = self
            .context
            .namespace_resolver
            .lookup_symbol(&qualified_path)
        {
            if let Some(symbol) = self.context.symbol_table.get_symbol(symbol_id) {
                let sym_id = symbol.id;
                let type_id = symbol.type_id;
                return Ok(self.ensure_symbol_has_class_type(sym_id, type_id));
            }
        }

        // Construct the full path for fallback
        let full_path = if type_path.package.is_empty() {
            type_path.name.clone()
        } else {
            format!("{}.{}", type_path.package.join("."), type_path.name)
        };

        // Try to resolve from symbol table (legacy path)
        let interned_name = self.context.intern_string(&full_path);

        if let Some(symbol) = self.context.symbol_table.lookup_symbol(
            ScopeId::first(), // Look in root scope for type definitions
            interned_name,
        ) {
            let sym_id = symbol.id;
            let type_id = symbol.type_id;
            return Ok(self.ensure_symbol_has_class_type(sym_id, type_id));
        }

        // For qualified names (e.g., "haxe.ds.IntMap"), also try the simple name
        // in root scope. Extern classes loaded via load_imports_efficiently may be
        // registered under their simple name, not the fully qualified path.
        if !type_path.package.is_empty() {
            if let Some(symbol) = self
                .context
                .symbol_table
                .lookup_symbol(ScopeId::first(), name_interned)
            {
                let sym_id = symbol.id;
                let type_id = symbol.type_id;
                return Ok(self.ensure_symbol_has_class_type(sym_id, type_id));
            }
        }

        // Type not found - create a placeholder and defer resolution
        let placeholder_type = self
            .context
            .type_table
            .borrow_mut()
            .create_type(crate::tast::core::TypeKind::Unknown);

        // Add to deferred resolutions for later processing
        self.resolution_state
            .deferred_resolutions
            .push(DeferredTypeResolution {
                type_name: full_path.clone(),
                target_type_id: placeholder_type,
                location: self.context.create_location(),
                type_params: Vec::new(), // For constructor calls, we don't need type params here
            });

        Ok(placeholder_type)
    }

    /// Resolve deferred type references (second pass)
    pub(crate) fn resolve_deferred_types(&mut self) -> LoweringResult<()> {
        let deferred = std::mem::take(&mut self.resolution_state.deferred_resolutions);

        for deferred_type in deferred {
            // Try to resolve the type now that all declarations have been processed
            // Parse the qualified type name into package path and simple name
            let parts: Vec<&str> = deferred_type.type_name.split('.').collect();

            let symbol = if parts.len() > 1 {
                // Qualified name like "haxe.iterators.ArrayIterator"
                let (package_parts, name) = parts.split_at(parts.len() - 1);
                let package: Vec<InternedString> = package_parts
                    .iter()
                    .map(|s| self.context.intern_string(s))
                    .collect();
                let name = self.context.intern_string(name[0]);

                let qualified_path =
                    crate::tast::namespace::QualifiedPath::new(package.clone(), name);

                // First try namespace resolver for qualified path lookup
                if let Some(symbol_id) = self
                    .context
                    .namespace_resolver
                    .lookup_symbol(&qualified_path)
                {
                    self.context.symbol_table.get_symbol(symbol_id)
                } else {
                    // Namespace lookup failed, try root scope lookup for backward compatibility
                    let interned_name = self.context.intern_string(&deferred_type.type_name);
                    self.context
                        .symbol_table
                        .lookup_symbol(ScopeId::first(), interned_name)
                }
            } else {
                // Simple name - look up in symbol table root scope
                let interned_name = self.context.intern_string(&deferred_type.type_name);
                self.context
                    .symbol_table
                    .lookup_symbol(ScopeId::first(), interned_name)
            };

            if let Some(symbol) = symbol {
                // Create the actual type based on the symbol kind
                let resolved_type = match symbol.kind {
                    crate::tast::SymbolKind::Class => self
                        .context
                        .type_table
                        .borrow_mut()
                        .create_class_type(symbol.id, Vec::new()),
                    crate::tast::SymbolKind::Interface => self
                        .context
                        .type_table
                        .borrow_mut()
                        .create_interface_type(symbol.id, Vec::new()),
                    crate::tast::SymbolKind::Enum => self
                        .context
                        .type_table
                        .borrow_mut()
                        .create_enum_type(symbol.id, Vec::new()),
                    crate::tast::SymbolKind::Abstract => {
                        self.context.type_table.borrow_mut().create_type(
                            crate::tast::core::TypeKind::Abstract {
                                symbol_id: symbol.id,
                                underlying: None,
                                type_args: Vec::new(),
                            },
                        )
                    }
                    crate::tast::SymbolKind::TypeAlias => {
                        let target_type = self.context.type_table.borrow().dynamic_type();
                        self.context.type_table.borrow_mut().create_type(
                            crate::tast::core::TypeKind::TypeAlias {
                                symbol_id: symbol.id,
                                target_type,
                                type_args: Vec::new(),
                            },
                        )
                    }
                    _ => {
                        // For other symbol kinds, keep as placeholder
                        continue;
                    }
                };

                // Record the mapping from placeholder to resolved type
                self.resolution_state
                    .placeholder_to_real
                    .insert(deferred_type.target_type_id, resolved_type);

                // TODO: Update all references to the placeholder type
                // For now, we've recorded the mapping which can be used later
            } else {
                // Still unresolved - only report errors for user-authored types.
                // Internal/synthetic stdlib references (file_id = u32::MAX) are
                // from lazy-loaded stdlib files whose transitive dependencies
                // may not be loaded yet. These are not user errors.
                let is_internal = deferred_type.location.file_id == u32::MAX
                    && deferred_type.location.byte_offset == 0;
                if !is_internal {
                    self.context.errors.push(LoweringError::UnresolvedType {
                        type_name: deferred_type.type_name,
                        location: deferred_type.location,
                    });
                }
                // Continue processing other deferred types
            }
        }

        Ok(())
    }
}

mod generics;
mod predicates;
