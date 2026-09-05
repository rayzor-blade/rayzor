//! Type parameters, substitution and generic instantiation.

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
    /// Lower type parameters
    pub(crate) fn lower_type_parameters(
        &mut self,
        type_params: &[TypeParam],
    ) -> LoweringResult<Vec<TypedTypeParameter>> {
        let mut result = Vec::new();
        for type_param in type_params {
            result.push(self.lower_type_parameter(type_param)?);
        }
        Ok(result)
    }

    /// Lower a single type parameter
    fn lower_type_parameter(
        &mut self,
        type_param: &TypeParam,
    ) -> LoweringResult<TypedTypeParameter> {
        let name = self.context.intern_string(&type_param.name);

        // Process constraints - but handle them specially if they reference type parameters
        let mut constraints = Vec::new();
        let mut deferred_constraints = Vec::new();

        for constraint in &type_param.constraints {
            // Check if this constraint might reference type parameters that aren't defined yet
            if self.type_might_reference_undefined_params(constraint) {
                // Create a placeholder for now
                let placeholder_type = self.context.type_table.borrow_mut().create_type(
                    crate::tast::core::TypeKind::Placeholder {
                        name: self.context.intern_string("<deferred_constraint>"),
                    },
                );
                constraints.push(placeholder_type);
                deferred_constraints.push((constraint.clone(), placeholder_type));
            } else {
                // Safe to lower now
                constraints.push(self.lower_type(constraint)?);
            }
        }

        // Store deferred constraints for later resolution
        for (constraint_type, placeholder) in deferred_constraints {
            if let Type::Path { path, params, .. } = &constraint_type {
                let type_name = if path.package.is_empty() {
                    path.name.clone()
                } else {
                    format!("{}.{}", path.package.join("."), path.name)
                };

                self.resolution_state
                    .deferred_resolutions
                    .push(DeferredTypeResolution {
                        type_name,
                        location: self.context.create_location(),
                        type_params: params.iter().map(|p| format!("{:?}", p)).collect(),
                        target_type_id: placeholder,
                    });
            }
        }

        // Convert TypeId constraints to ConstraintKind for symbol table
        let constraint_kinds: Vec<super::type_checker::ConstraintKind> = constraints
            .iter()
            .map(|&type_id| super::type_checker::ConstraintKind::Implements {
                interface_type: type_id,
            })
            .collect();

        // Create type parameter symbol with proper type
        let symbol_id = self
            .context
            .symbol_table
            .create_type_parameter(name, constraint_kinds);
        let param_type_id = self.context.type_table.borrow_mut().create_type_parameter(
            symbol_id,
            constraints.clone(),
            Variance::Invariant,
        );

        // The declared default, lowered here so inference can reach it.
        // The declared default, lowered here so inference can reach it.
        let default_type = match &type_param.default_type {
            Some(ty) => self.lower_type(ty).ok(),
            None => None,
        };

        Ok(TypedTypeParameter {
            symbol_id,
            name,
            constraints,
            variance: TypeVariance::Invariant, // Default variance
            default_type,
            source_location: self.context.create_location(), // TODO: Get span from type_param
        })
    }

    /// Build the concrete `StringMap<V>` / `IntMap<V>` / `ObjectMap<K,V>`
    /// type for a `Map<K, V>` annotation. Returns `None` if K is a type
    /// parameter / Dynamic / unresolved (where we can't safely pick one),
    /// or if the concrete container class isn't loaded yet. The returned
    /// TypeId is a fully-formed class type and can be substituted wherever
    /// the original `Map<K,V>` Abstract TypeId would have appeared.
    /// The type a `Map` key selects its container by: an abstract key picks the
    /// container its UNDERLYING type would, which is what haxe's multiType
    /// resolution does (it follows the key before choosing). `UInt` is
    /// `abstract UInt(Int)` and belongs in an IntMap; an `enum abstract Foo(String)`
    /// belongs in a StringMap. Without this an abstract key matched no arm, the
    /// selection returned None, and the caller's fallback path recursed until the
    /// compiler's stack ran out — `new Map<UInt,String>()` overflowed as soon as it
    /// was indexed.
    fn map_key_repr(&self, key_ty: TypeId) -> TypeId {
        let mut current = key_ty;
        // Abstracts can nest (`abstract Id(UInt)`), and a broken declaration could
        // point one at itself, so the walk is bounded.
        for _ in 0..16 {
            let underlying = {
                let tt = self.context.type_table.borrow();
                match tt.get(current).map(|t| &t.kind) {
                    Some(crate::tast::core::TypeKind::Abstract {
                        underlying: Some(u),
                        ..
                    }) => Some(*u),
                    Some(crate::tast::core::TypeKind::TypeAlias { target_type, .. }) => {
                        Some(*target_type)
                    }
                    _ => None,
                }
            };
            match underlying {
                Some(next) if next != current => current = next,
                _ => break,
            }
        }
        current
    }

    /// A `Map<K,V>` whose key is a type parameter can never pick a container:
    /// Haxe forbids constructing one there, but such a map is still held and
    /// dispatched on, and the abstract's own methods already go through its
    /// underlying `IMap<K,V>`. Typing the slot as that interface makes the
    /// value a fat pointer, which is what those methods read.
    pub(crate) fn multitype_map_underlying_imap(&self, type_args: &[TypeId]) -> Option<TypeId> {
        if type_args.len() != 2 {
            return None;
        }
        let key_repr = self.map_key_repr(type_args[0]);
        let key_is_param = matches!(
            self.context.type_table.borrow().get(key_repr).map(|t| &t.kind),
            Some(crate::tast::core::TypeKind::TypeParameter { .. })
        );
        if !key_is_param {
            return None;
        }
        let imap = self
            .context
            .string_interner
            .intern("haxe.Constraints.IMap");
        let bare = self.context.string_interner.intern("IMap");
        let sym = self
            .context
            .symbol_table
            .resolve_qualified_name(imap)
            .or_else(|| {
                self.context
                    .symbol_table
                    .lookup_symbol(ScopeId::first(), bare)
                    .map(|s| s.id)
            })
            .filter(|id| {
                self.context
                    .symbol_table
                    .get_symbol(*id)
                    .map(|s| matches!(s.kind, crate::tast::SymbolKind::Interface))
                    .unwrap_or(false)
            })?;
        Some(self.context.type_table.borrow_mut().create_type(
            crate::tast::core::TypeKind::Interface {
                symbol_id: sym,
                type_args: type_args.to_vec(),
            },
        ))
    }

    pub(crate) fn resolve_multitype_map_to_concrete(&self, type_args: &[TypeId]) -> Option<TypeId> {
        if type_args.len() != 2 {
            return None;
        }
        let key_ty = type_args[0];
        let value_ty = type_args[1];
        // Choose by what the key IS underneath; keep the declared key in the
        // container's type arguments so the map still reads as Map<Foo, V>.
        let key_repr = self.map_key_repr(key_ty);

        let (concrete_name, concrete_type_args): (&str, Vec<TypeId>) = {
            let tt = self.context.type_table.borrow();
            let key_kind = tt.get(key_repr).map(|t| &t.kind);
            match key_kind {
                Some(crate::tast::core::TypeKind::String) => ("StringMap", vec![value_ty]),
                Some(crate::tast::core::TypeKind::Int) => ("IntMap", vec![value_ty]),
                Some(crate::tast::core::TypeKind::Enum { .. }) => {
                    ("EnumValueMap", vec![key_ty, value_ty])
                }
                Some(crate::tast::core::TypeKind::Class { .. })
                | Some(crate::tast::core::TypeKind::Interface { .. })
                | Some(crate::tast::core::TypeKind::Anonymous { .. })
                | Some(crate::tast::core::TypeKind::GenericInstance { .. }) => {
                    ("ObjectMap", vec![key_ty, value_ty])
                }
                _ => return None,
            }
        };

        // Packaged types are published under their qualified name; the simple
        // name is only bound when the module was registered from source.
        let qualified_interned = self
            .context
            .string_interner
            .intern(&format!("haxe.ds.{concrete_name}"));
        let concrete_name_interned = self.context.string_interner.intern(concrete_name);
        let symbol = self
            .context
            .symbol_table
            .lookup_symbol(ScopeId::first(), qualified_interned)
            .or_else(|| {
                self.context
                    .symbol_table
                    .lookup_symbol(ScopeId::first(), concrete_name_interned)
            })?;
        Some(
            self.context
                .type_table
                .borrow_mut()
                .create_class_type(symbol.id, concrete_type_args),
        )
    }

    pub(crate) fn maybe_resolve_multitype_map(
        &self,
        original_class_type: TypeId,
        original_type_args: Vec<TypeId>,
        type_path: &parser::TypePath,
    ) -> (TypeId, Vec<TypeId>, Option<String>) {
        // Detect `Map` (with or without `haxe.ds.` package qualifier).
        let is_map = type_path.name == "Map"
            && (type_path.package.is_empty()
                || type_path.package == vec!["haxe".to_string(), "ds".to_string()]);
        if !is_map || original_type_args.len() != 2 {
            return (original_class_type, original_type_args, None);
        }

        let key_ty = original_type_args[0];
        let value_ty = original_type_args[1];
        let key_repr = self.map_key_repr(key_ty);

        // Pick the concrete container name + remaining type args from K's kind.
        let (concrete_name, concrete_type_args): (&str, Vec<TypeId>) = {
            let tt = self.context.type_table.borrow();
            let key_kind = tt.get(key_repr).map(|t| &t.kind);
            match key_kind {
                Some(crate::tast::core::TypeKind::String) => ("StringMap", vec![value_ty]),
                Some(crate::tast::core::TypeKind::Int) => ("IntMap", vec![value_ty]),
                Some(crate::tast::core::TypeKind::Enum { .. }) => {
                    ("EnumValueMap", vec![key_ty, value_ty])
                }
                Some(crate::tast::core::TypeKind::Class { .. })
                | Some(crate::tast::core::TypeKind::Interface { .. })
                | Some(crate::tast::core::TypeKind::Anonymous { .. })
                | Some(crate::tast::core::TypeKind::GenericInstance { .. }) => {
                    ("ObjectMap", vec![key_ty, value_ty])
                }
                _ => {
                    // Unknown / type-parameter / dynamic K — leave as-is.
                    // Honors the "no silent fallthrough" principle: we can't
                    // pick a concrete safely, so the caller continues with
                    // the abstract resolution path (which will surface any
                    // failure modes downstream rather than guessing here).
                    return (original_class_type, original_type_args, None);
                }
            }
        };

        // Resolve the concrete class symbol. `haxe.ds.StringMap` and friends
        // are packaged, so they are published under their qualified name;
        // the simple name is only bound when the module was registered from
        // source. Try the qualified name first and fall back to the simple
        // one so both registration paths resolve to the same class.
        let qualified = format!("haxe.ds.{concrete_name}");
        let qualified_interned = self.context.string_interner.intern(&qualified);
        let concrete_name_interned = self.context.string_interner.intern(concrete_name);
        let resolved = self
            .context
            .symbol_table
            .lookup_symbol(ScopeId::first(), qualified_interned)
            .or_else(|| {
                self.context
                    .symbol_table
                    .lookup_symbol(ScopeId::first(), concrete_name_interned)
            });
        let symbol = match resolved {
            Some(sym) => sym,
            None => {
                // Concrete class isn't loaded — keep the original to avoid
                // a worse downstream failure. Caller will see whatever the
                // abstract resolution produces.
                return (original_class_type, original_type_args, None);
            }
        };

        let concrete_class_type = self
            .context
            .type_table
            .borrow_mut()
            .create_class_type(symbol.id, concrete_type_args.clone());

        let qualified_name = format!("haxe.ds.{}", concrete_name);
        (
            concrete_class_type,
            concrete_type_args,
            Some(qualified_name),
        )
    }

    pub(crate) fn infer_type_args_from_constructor(
        &self,
        class_type_id: TypeId,
        args: &[TypedExpression],
    ) -> Option<TypeId> {
        // Get class symbol
        let class_symbol = {
            let tt = self.context.type_table.borrow();
            let ti = tt.get(class_type_id)?;
            match &ti.kind {
                crate::tast::core::TypeKind::Class { symbol_id, .. } => *symbol_id,
                _ => return None,
            }
        };

        // Check if class has type parameters. Prefer the per-instance map
        // (current file's lowering) but fall back to the shared SymbolTable
        // for cross-file lookups (e.g. user file calling `new Arc(value)`
        // when `Arc.hx` was lowered in a different ast_lowering instance).
        let type_param_ids: Vec<TypeId> =
            if let Some(ids) = self.class_type_params.get(&class_symbol) {
                ids.clone()
            } else if let Some(ids) = self
                .context
                .symbol_table
                .get_class_type_params(class_symbol)
            {
                ids.clone()
            } else {
                return None;
            };
        if type_param_ids.is_empty() {
            return None;
        }

        // Get constructor symbol and its function type — same fallback chain.
        let ctor_symbol =
            if let Some(s) = self.class_constructor_symbols.get(&class_symbol).copied() {
                s
            } else if let Some(s) = self
                .context
                .symbol_table
                .get_class_constructor(class_symbol)
            {
                s
            } else {
                return None;
            };
        let ctor_type_id = self.context.symbol_table.get_symbol(ctor_symbol)?.type_id;
        let param_type_ids = {
            let tt = self.context.type_table.borrow();
            let ti = tt.get(ctor_type_id)?;
            match &ti.kind {
                crate::tast::core::TypeKind::Function { params, .. } => params.clone(),
                _ => return None,
            }
        };

        // Match TypeParameter params against argument types
        let mut tp_to_concrete: BTreeMap<TypeId, TypeId> = BTreeMap::new();
        {
            let tt = self.context.type_table.borrow();
            for (i, param_ty) in param_type_ids.iter().enumerate() {
                if i >= args.len() {
                    break;
                }
                if let Some(param_info) = tt.get(*param_ty) {
                    if matches!(
                        param_info.kind,
                        crate::tast::core::TypeKind::TypeParameter { .. }
                    ) {
                        tp_to_concrete.insert(*param_ty, args[i].expr_type);
                    }
                }
            }
        }

        if tp_to_concrete.is_empty() {
            return None;
        }

        // Build ordered type_args matching the class's type parameter order
        let type_args: Vec<TypeId> = type_param_ids
            .iter()
            .map(|tp_id| {
                tp_to_concrete
                    .get(tp_id)
                    .copied()
                    .unwrap_or_else(|| self.context.type_table.borrow().dynamic_type())
            })
            .collect();

        Some(
            self.context
                .type_table
                .borrow_mut()
                .create_class_type(class_symbol, type_args),
        )
    }

    /// Lower a function from a class field (includes field metadata)

    /// Check if a type might reference type parameters that aren't defined yet
    fn type_might_reference_undefined_params(&mut self, type_annotation: &Type) -> bool {
        match type_annotation {
            Type::Path { path, params, .. } => {
                let name = if path.package.is_empty() {
                    &path.name
                } else {
                    return false; // Qualified paths are not type parameters
                };

                // Check if any type arguments contain references to type parameters
                // For example, in Sortable<T>, the T might not be defined yet
                for param in params {
                    if self.type_might_reference_undefined_params(param) {
                        return true;
                    }

                    // Check if this is a simple type parameter reference
                    if let Type::Path {
                        path: param_path,
                        params: param_params,
                        ..
                    } = param
                    {
                        if param_path.package.is_empty() && param_params.is_empty() {
                            // This looks like a type parameter reference (e.g., T)
                            // Check if it's NOT a built-in type
                            if self.resolve_builtin_type(&param_path.name).is_none() {
                                // It's not a built-in, so it might be a type parameter
                                // Check if it's already defined
                                let interned_param_name =
                                    self.context.intern_string(&param_path.name);
                                if self
                                    .context
                                    .resolve_type_parameter(interned_param_name)
                                    .is_none()
                                {
                                    return true;
                                }
                            }
                        }
                    }
                }

                // Check if the base type exists and can be resolved
                // Only defer if we can't resolve the base type or if it has arity issues
                if !params.is_empty() {
                    // Try to resolve the base type to see if it exists
                    let base_type_name = if path.package.is_empty() {
                        &path.name
                    } else {
                        // Qualified names should be resolvable - don't defer
                        return false;
                    };

                    // Check if this is a known interface/class that can accept type parameters
                    // Try to find the type in the symbol table without interning a new string
                    // We can check for common interface names directly
                    match base_type_name.as_str() {
                        "Comparable" | "Iterable" | "Iterator" | "Array" | "Map" => {
                            // These are well-known generic types, don't defer
                            return false;
                        }
                        _ => {
                            // For other types, be conservative and defer for now
                            // This could be improved with better symbol resolution
                            return true;
                        }
                    }
                }

                false
            }
            Type::Function { params, ret, .. } => {
                // Check function parameter and return types
                params
                    .iter()
                    .any(|p| self.type_might_reference_undefined_params(p))
                    || self.type_might_reference_undefined_params(ret)
            }
            Type::Anonymous { fields, .. } => {
                // Check anonymous type fields
                fields
                    .iter()
                    .any(|f| self.type_might_reference_undefined_params(&f.type_hint))
            }
            Type::Optional { inner, .. } => self.type_might_reference_undefined_params(inner),
            Type::Parenthesis { inner, .. } => self.type_might_reference_undefined_params(inner),
            Type::Intersection { left, right, .. } => {
                self.type_might_reference_undefined_params(left)
                    || self.type_might_reference_undefined_params(right)
            }
            Type::Wildcard { .. } => false,
        }
    }

    /// Compute what substitution is needed (without creating new types)
    pub(crate) fn compute_type_substitution(
        &self,
        return_type: TypeId,
        receiver_type: TypeId,
        type_table: &std::cell::Ref<'_, crate::tast::TypeTable>,
    ) -> TypeSubstitutionResult {
        // Get receiver's substitution info
        let receiver_type_info = match type_table.get(receiver_type) {
            Some(info) => info,
            None => return TypeSubstitutionResult::NoChange(return_type),
        };

        // Extract base type parameters and type arguments from receiver
        let (base_type_params, type_args) = match &receiver_type_info.kind {
            crate::tast::core::TypeKind::GenericInstance {
                base_type,
                type_args,
                ..
            } => {
                // Get the base class's type parameters
                if let Some(base_info) = type_table.get(*base_type) {
                    match &base_info.kind {
                        crate::tast::core::TypeKind::Class {
                            type_args: params, ..
                        }
                        | crate::tast::core::TypeKind::Interface {
                            type_args: params, ..
                        } => (params.clone(), type_args.clone()),
                        _ => return TypeSubstitutionResult::NoChange(return_type),
                    }
                } else {
                    return TypeSubstitutionResult::NoChange(return_type);
                }
            }
            crate::tast::core::TypeKind::Class {
                symbol_id,
                type_args,
            }
            | crate::tast::core::TypeKind::Interface {
                symbol_id,
                type_args,
            } if !type_args.is_empty() => {
                // Concrete generic class instance like `Mutex<State>` —
                // type_args holds the concrete substitution. Derive the
                // base parameter list from the class symbol's TypeParameter
                // record (lifted onto SymbolTable so cross-file lookups
                // see it).
                let params = self
                    .context
                    .symbol_table
                    .get_class_type_params(*symbol_id)
                    .cloned()
                    .or_else(|| self.class_type_params.get(symbol_id).cloned())
                    .unwrap_or_default();
                if params.is_empty() {
                    // The class's TypeParameter list was never registered for
                    // this `class_symbol` — this happens for stdlib extern
                    // containers (StringMap / IntMap / ObjectMap) produced by
                    // `@:multiType` Map resolution, whose declaration was
                    // lowered in a *different* ast_lowering instance, so neither
                    // the per-instance `class_type_params` map nor the shared
                    // SymbolTable mirror has them. Without params we cannot do
                    // symbol/name-based substitution, but the receiver still
                    // carries concrete `type_args`. For a return type that is a
                    // bare (or Optional-wrapped) TypeParameter — e.g.
                    // `StringMap<V>.get : Null<V>` — fall back to a positional
                    // substitution against the *last* type_arg (the value type
                    // for every Map container, and the sole arg for
                    // single-parameter containers). Leaving it abstract makes
                    // `maybe_box_for_optional` box the value with a placeholder
                    // type tag, corrupting enum/reference results.
                    if !type_args.is_empty() {
                        // Peel an optional Optional wrapper to find a bare
                        // TypeParameter return.
                        let bare_tp = match type_table.get(return_type).map(|i| &i.kind) {
                            Some(crate::tast::core::TypeKind::TypeParameter { .. }) => true,
                            Some(crate::tast::core::TypeKind::Optional { inner_type }) => matches!(
                                type_table.get(*inner_type).map(|i| &i.kind),
                                Some(crate::tast::core::TypeKind::TypeParameter { .. })
                            ),
                            _ => false,
                        };
                        if bare_tp {
                            let concrete = *type_args.last().unwrap();
                            return match type_table.get(return_type).map(|i| &i.kind) {
                                Some(crate::tast::core::TypeKind::Optional { .. }) => {
                                    // Reuse an existing `Optional<concrete>` if present.
                                    for (tid, tinfo) in type_table.iter() {
                                        if let crate::tast::core::TypeKind::Optional {
                                            inner_type: ex,
                                        } = &tinfo.kind
                                        {
                                            if *ex == concrete {
                                                return TypeSubstitutionResult::DirectSubstitution(
                                                    tid,
                                                );
                                            }
                                        }
                                    }
                                    TypeSubstitutionResult::NeedOptional {
                                        inner_type: concrete,
                                    }
                                }
                                _ => TypeSubstitutionResult::DirectSubstitution(concrete),
                            };
                        }
                    }
                    return TypeSubstitutionResult::NoChange(return_type);
                }
                (params, type_args.clone())
            }
            _ => return TypeSubstitutionResult::NoChange(return_type),
        };

        // Get the return type info
        let return_type_info = match type_table.get(return_type) {
            Some(info) => info,
            None => return TypeSubstitutionResult::NoChange(return_type),
        };

        match &return_type_info.kind {
            crate::tast::core::TypeKind::TypeParameter { symbol_id, .. } => {
                // Direct type parameter - find and substitute
                // First try exact SymbolId match
                for (i, param_type_id) in base_type_params.iter().enumerate() {
                    if let Some(param_info) = type_table.get(*param_type_id) {
                        if let crate::tast::core::TypeKind::TypeParameter {
                            symbol_id: param_sym,
                            ..
                        } = &param_info.kind
                        {
                            if param_sym == symbol_id {
                                if i < type_args.len() {
                                    return TypeSubstitutionResult::DirectSubstitution(
                                        type_args[i],
                                    );
                                }
                            }
                        }
                    }
                }
                // Fallback: name-based matching for extern class methods where the method's
                // type parameter T has a different SymbolId than the class definition's T
                let ret_param_name = self
                    .context
                    .symbol_table
                    .get_symbol(*symbol_id)
                    .map(|s| s.name);
                if let Some(ret_name) = ret_param_name {
                    for (i, param_type_id) in base_type_params.iter().enumerate() {
                        if let Some(param_info) = type_table.get(*param_type_id) {
                            if let crate::tast::core::TypeKind::TypeParameter {
                                symbol_id: param_sym,
                                ..
                            } = &param_info.kind
                            {
                                let param_name = self
                                    .context
                                    .symbol_table
                                    .get_symbol(*param_sym)
                                    .map(|s| s.name);
                                if param_name == Some(ret_name) && i < type_args.len() {
                                    return TypeSubstitutionResult::DirectSubstitution(
                                        type_args[i],
                                    );
                                }
                            }
                        }
                    }
                }
                TypeSubstitutionResult::NoChange(return_type)
            }
            crate::tast::core::TypeKind::GenericInstance {
                base_type,
                type_args: ret_type_args,
                ..
            } => {
                // Generic return type - need to substitute type args recursively
                let mut new_type_args = Vec::with_capacity(ret_type_args.len());
                let mut changed = false;

                for arg in ret_type_args {
                    match self.compute_type_substitution(*arg, receiver_type, type_table) {
                        TypeSubstitutionResult::NoChange(_) => new_type_args.push(*arg),
                        TypeSubstitutionResult::DirectSubstitution(new_arg) => {
                            new_type_args.push(new_arg);
                            changed = true;
                        }
                        TypeSubstitutionResult::NeedGenericInstance { .. }
                        | TypeSubstitutionResult::NeedClassInstance { .. }
                        | TypeSubstitutionResult::NeedOptional { .. }
                        | TypeSubstitutionResult::NeedTypeAlias { .. } => {
                            // Would need to create nested type - for now just use the original
                            // This is a limitation, but handles most common cases
                            new_type_args.push(*arg);
                        }
                    }
                }

                if changed {
                    // Check if this exact type already exists
                    if let Some(existing) =
                        self.find_existing_generic_instance(*base_type, &new_type_args, type_table)
                    {
                        return TypeSubstitutionResult::DirectSubstitution(existing);
                    }
                    return TypeSubstitutionResult::NeedGenericInstance {
                        base_type: *base_type,
                        type_args: new_type_args,
                    };
                }
                TypeSubstitutionResult::NoChange(return_type)
            }
            crate::tast::core::TypeKind::Class {
                symbol_id,
                type_args: ret_type_args,
            } if !ret_type_args.is_empty() => {
                // Generic class return type (e.g. `MutexGuard<T>` from
                // `Mutex.lock()`). Same recursive substitution as the
                // `GenericInstance` arm — needed for nested wrappers like
                // `Arc<Mutex<T>>` where `lock()`'s return type is
                // serialised as a Class with TypeParameter args rather
                // than a GenericInstance.
                let mut new_type_args = Vec::with_capacity(ret_type_args.len());
                let mut changed = false;
                for arg in ret_type_args {
                    match self.compute_type_substitution(*arg, receiver_type, type_table) {
                        TypeSubstitutionResult::NoChange(_) => new_type_args.push(*arg),
                        TypeSubstitutionResult::DirectSubstitution(new_arg) => {
                            new_type_args.push(new_arg);
                            changed = true;
                        }
                        TypeSubstitutionResult::NeedGenericInstance { .. }
                        | TypeSubstitutionResult::NeedClassInstance { .. }
                        | TypeSubstitutionResult::NeedOptional { .. }
                        | TypeSubstitutionResult::NeedTypeAlias { .. } => {
                            new_type_args.push(*arg);
                        }
                    }
                }
                if changed {
                    return TypeSubstitutionResult::NeedClassInstance {
                        symbol_id: *symbol_id,
                        type_args: new_type_args,
                    };
                }
                TypeSubstitutionResult::NoChange(return_type)
            }
            // An alias carries its arguments on its own node (`Iterator<T>`), so
            // the substitution rebuilds the alias over the substituted arguments.
            crate::tast::core::TypeKind::TypeAlias {
                symbol_id,
                target_type,
                type_args: ret_type_args,
            } => {
                let mut new_type_args = Vec::with_capacity(ret_type_args.len());
                let mut changed = false;
                for arg in ret_type_args {
                    match self.compute_type_substitution(*arg, receiver_type, type_table) {
                        TypeSubstitutionResult::DirectSubstitution(new_arg) => {
                            new_type_args.push(new_arg);
                            changed = true;
                        }
                        _ => new_type_args.push(*arg),
                    }
                }
                if changed {
                    return TypeSubstitutionResult::NeedTypeAlias {
                        symbol_id: *symbol_id,
                        target_type: *target_type,
                        type_args: new_type_args,
                    };
                }
                TypeSubstitutionResult::NoChange(return_type)
            }
            // `Null<V>` (e.g. `Map<K,V>.get` return) — substitute the inner type
            // so the result is `Optional<concrete>` rather than `Optional<V>`.
            // Leaving it abstract makes the caller box the value via
            // `haxe_box_typed_ptr` with an unresolved (0) tag, which boxes
            // enum/reference values as Int and corrupts the subsequent match.
            crate::tast::core::TypeKind::Optional { inner_type } => {
                let inner = *inner_type;
                match self.compute_type_substitution(inner, receiver_type, type_table) {
                    TypeSubstitutionResult::DirectSubstitution(new_inner) => {
                        // Reuse an existing `Optional<new_inner>` if present.
                        for (tid, tinfo) in type_table.iter() {
                            if let crate::tast::core::TypeKind::Optional { inner_type: ex } =
                                &tinfo.kind
                            {
                                if *ex == new_inner {
                                    return TypeSubstitutionResult::DirectSubstitution(tid);
                                }
                            }
                        }
                        TypeSubstitutionResult::NeedOptional {
                            inner_type: new_inner,
                        }
                    }
                    // Nested creation (Optional<GenericInstance<V>> etc.) — keep
                    // the original; the common Map<K,V>.get case is the bare-V path.
                    _ => TypeSubstitutionResult::NoChange(return_type),
                }
            }
            _ => TypeSubstitutionResult::NoChange(return_type),
        }
    }

    /// Substitute type parameters in a type with actual type arguments from a receiver type.
    ///
    /// For example, if we have:
    /// - return_type = T (a TypeParameter)
    /// - receiver_type = Arc<Channel<Int>> (a GenericInstance)
    ///
    /// This function will substitute T with Channel<Int>.
    fn substitute_type_params_in_type(
        &self,
        return_type: TypeId,
        receiver_type: TypeId,
        type_table: &std::cell::Ref<'_, crate::tast::TypeTable>,
    ) -> TypeId {
        // Collect all necessary info in one pass, then we can release the borrow if needed

        // Get the receiver's substitution info (base_type_params and type_args)
        let substitution_info: Option<(Vec<TypeId>, Vec<TypeId>)> = {
            let receiver_type_info = match type_table.get(receiver_type) {
                Some(info) => info,
                None => return return_type,
            };

            match &receiver_type_info.kind {
                crate::tast::core::TypeKind::GenericInstance {
                    base_type,
                    type_args,
                    ..
                } => {
                    // Get the base class's type parameters
                    if let Some(base_info) = type_table.get(*base_type) {
                        match &base_info.kind {
                            crate::tast::core::TypeKind::Class {
                                type_args: params, ..
                            }
                            | crate::tast::core::TypeKind::Interface {
                                type_args: params, ..
                            } => Some((params.clone(), type_args.clone())),
                            _ => None,
                        }
                    } else {
                        None
                    }
                }
                _ => None,
            }
        };

        let (base_type_params, type_args) = match substitution_info {
            Some(info) => info,
            None => return return_type,
        };

        // Get the return type info
        let return_type_info = match type_table.get(return_type) {
            Some(info) => info,
            None => return return_type,
        };

        // Now substitute based on return type kind
        match &return_type_info.kind {
            crate::tast::core::TypeKind::TypeParameter { symbol_id, .. } => {
                // Find which type parameter this is and substitute
                for (i, param_type_id) in base_type_params.iter().enumerate() {
                    if let Some(param_info) = type_table.get(*param_type_id) {
                        if let crate::tast::core::TypeKind::TypeParameter {
                            symbol_id: param_sym,
                            ..
                        } = &param_info.kind
                        {
                            if param_sym == symbol_id {
                                // Found matching type parameter, substitute with type argument
                                if i < type_args.len() {
                                    return type_args[i];
                                }
                            }
                        }
                    }
                }
                return_type
            }
            crate::tast::core::TypeKind::GenericInstance {
                base_type,
                type_args: ret_type_args,
                ..
            } => {
                // Recursively substitute type parameters in the type arguments
                // E.g., Arc<T>.clone() returns Arc<T>, substitute T in the Arc<T>
                // First, collect all info we need
                let base = *base_type;
                let ret_args: Vec<TypeId> = ret_type_args.clone();

                let mut new_type_args = Vec::with_capacity(ret_args.len());
                let mut changed = false;
                for arg in &ret_args {
                    let substituted =
                        self.substitute_type_params_in_type(*arg, receiver_type, type_table);
                    if substituted != *arg {
                        changed = true;
                    }
                    new_type_args.push(substituted);
                }
                if changed {
                    // Need to create a new type - but we can't mutably borrow here
                    // Return a signal that we need to create a new type
                    // For now, we'll use an existing type if it matches, or return the original
                    // Actually, let's check if the type already exists
                    // This is a limitation - we may need to refactor more significantly
                    // For now, try to find if the substituted type exists
                    if let Some(existing) =
                        self.find_existing_generic_instance(base, &new_type_args, type_table)
                    {
                        return existing;
                    }
                    // Fallback: return original (the substitution will need to happen elsewhere)
                    return return_type;
                }
                return_type
            }
            crate::tast::core::TypeKind::Class {
                symbol_id: _sym_id,
                type_args: class_type_args,
            } if !class_type_args.is_empty() => {
                // For class types with type args that are type parameters, substitute them
                let class_args: Vec<TypeId> = class_type_args.clone();

                let mut new_type_args = Vec::with_capacity(class_args.len());
                let mut changed = false;
                for arg in &class_args {
                    let substituted =
                        self.substitute_type_params_in_type(*arg, receiver_type, type_table);
                    if substituted != *arg {
                        changed = true;
                    }
                    new_type_args.push(substituted);
                }
                if changed {
                    // Same limitation as above
                    return return_type;
                }
                return_type
            }
            _ => return_type,
        }
    }

    /// Try to find an existing GenericInstance with the given base type and type args
    fn find_existing_generic_instance(
        &self,
        base_type: TypeId,
        type_args: &[TypeId],
        type_table: &std::cell::Ref<'_, crate::tast::TypeTable>,
    ) -> Option<TypeId> {
        // Search through existing types to find a matching GenericInstance
        // This is O(n) but avoids the borrow conflict
        for (type_id, type_info) in type_table.iter() {
            if let crate::tast::core::TypeKind::GenericInstance {
                base_type: existing_base,
                type_args: existing_args,
                ..
            } = &type_info.kind
            {
                if *existing_base == base_type && existing_args == type_args {
                    return Some(type_id);
                }
            }
        }
        None
    }

    /// Recursively match a parameter type against an argument type to find where
    /// a specific TypeParameter appears, and extract the concrete type from the
    /// argument at the same structural position.
    ///
    /// For example, if param_ty is `Function { return_type: T }` and arg_ty is
    /// `Function { return_type: Int }`, this returns `Some(Int)` for target T.
    pub(crate) fn match_type_param_in_types(
        target_sym: SymbolId,
        param_ty: TypeId,
        arg_ty: TypeId,
        type_table: &std::cell::Ref<'_, crate::tast::TypeTable>,
    ) -> Option<TypeId> {
        let param_info = type_table.get(param_ty)?;
        match &param_info.kind {
            crate::tast::core::TypeKind::TypeParameter { symbol_id, .. } => {
                if *symbol_id == target_sym {
                    Some(arg_ty)
                } else {
                    None
                }
            }
            crate::tast::core::TypeKind::Function {
                params: fn_params,
                return_type: fn_ret,
                ..
            } => {
                let arg_info = type_table.get(arg_ty)?;
                if let crate::tast::core::TypeKind::Function {
                    params: arg_fn_params,
                    return_type: arg_fn_ret,
                    ..
                } = &arg_info.kind
                {
                    // Check return type
                    if let Some(result) = Self::match_type_param_in_types(
                        target_sym,
                        *fn_ret,
                        *arg_fn_ret,
                        type_table,
                    ) {
                        return Some(result);
                    }
                    // Check function parameters
                    for (fp, ap) in fn_params.iter().zip(arg_fn_params.iter()) {
                        if let Some(result) =
                            Self::match_type_param_in_types(target_sym, *fp, *ap, type_table)
                        {
                            return Some(result);
                        }
                    }
                }
                None
            }
            crate::tast::core::TypeKind::GenericInstance { type_args, .. } => {
                let arg_info = type_table.get(arg_ty)?;
                if let crate::tast::core::TypeKind::GenericInstance {
                    type_args: arg_type_args,
                    ..
                } = &arg_info.kind
                {
                    for (ta, ata) in type_args.iter().zip(arg_type_args.iter()) {
                        if let Some(result) =
                            Self::match_type_param_in_types(target_sym, *ta, *ata, type_table)
                        {
                            return Some(result);
                        }
                    }
                }
                None
            }
            _ => None,
        }
    }
}
