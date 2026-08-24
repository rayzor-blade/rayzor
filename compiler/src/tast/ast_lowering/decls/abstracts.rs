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
    /// Lower an abstract declaration
    pub(crate) fn lower_abstract_declaration(
        &mut self,
        abstract_decl: &AbstractDecl,
    ) -> LoweringResult<TypedDeclaration> {
        let abstract_name = self.context.intern_string(&abstract_decl.name);

        let abstract_symbol = self
            .context
            .symbol_table
            .create_abstract_in_scope(abstract_name, ScopeId::first());

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
        let abstract_scope = self
            .context
            .enter_named_scope(ScopeKind::Class, abstract_name);

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
        self.class_fields.insert(abstract_symbol, Vec::new());
        // Initialize class_methods so the pre-pass-typed abstract methods are
        // reachable by resolve_class_method_symbol Strategy 1 at call sites.
        // The abstract symbol's scope_id stays at root (ScopeId::first()), so
        // the scope-based Strategy 3 cannot find methods registered in the
        // abstract's inner scope; the class_methods map is the same mechanism
        // regular classes use and keeps `@:coreType` static calls (Atomic.of,
        // Box.init) bound to their typed method instead of a Dynamic placeholder.
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
                    let sym = self.context.symbol_table.create_variable(member_name);
                    self.context
                        .symbol_table
                        .update_symbol_type(sym, member_type);
                    if let Some(s) = self.context.symbol_table.get_symbol_mut(sym) {
                        s.kind = crate::tast::SymbolKind::Field;
                    }
                    if is_static {
                        self.context
                            .symbol_table
                            .add_symbol_flags(sym, crate::tast::symbols::SymbolFlags::STATIC);
                    }
                    if let Some(scope) = self.context.scope_tree.get_scope_mut(abstract_scope) {
                        scope.add_symbol(sym, member_name);
                    }
                }
            }
        }

        // Process fields - separate fields, methods, and constructors
        let mut fields = Vec::with_capacity(abstract_decl.fields.len());
        let mut methods = Vec::with_capacity(abstract_decl.fields.len());
        let mut constructors = Vec::with_capacity(2); // Most abstracts have 0-2 constructors

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
                    match self.lower_field(field) {
                        Ok(mut typed_field) => {
                            // For enum abstracts, all var fields are implicitly static
                            if abstract_decl.is_enum_abstract && !typed_field.is_static {
                                typed_field.is_static = true;
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
