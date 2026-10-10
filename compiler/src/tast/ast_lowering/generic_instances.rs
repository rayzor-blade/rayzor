//! `@:generic` specialisation: one copy of a class or method per set of
//! concrete type arguments, named `Name_Arg1_Arg2` as Haxe names it. A copy is
//! the template's AST with each type parameter renamed to a placeholder that
//! lowers straight to its argument, so `new T()` constructs that class.
//! Arguments that are not concrete leave the use on the erased template.

use super::const_generics::Substitution;
use super::{AstLowering, LoweringResult};
use crate::tast::core::TypeKind;
use crate::tast::node::{TypedClass, TypedExpression};
use crate::tast::{InternedString, ScopeId, SymbolFlags, SymbolId, SymbolKind, TypeId, Visibility};
use parser::haxe_ast::{
    Access, ClassDecl, ClassField, ClassFieldKind, Expr, ExprKind, HaxeFile, Metadata, Modifier,
    Span, Type, TypeDeclaration, TypeParam, TypePath,
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone)]
pub(super) struct GenericClass {
    class: ClassDecl,
    pack: Vec<String>,
    /// The `pack._Module` prefix Haxe prints for a module-private template.
    private_prefix: Option<String>,
}

#[derive(Clone)]
pub(super) struct GenericMethod {
    field: ClassField,
    class_params: Vec<String>,
    pack: Vec<String>,
}

fn is_generic(meta: &[Metadata]) -> bool {
    meta.iter()
        .any(|meta| meta.name.trim_start_matches(':') == "generic")
}

fn qualified_name(pack: &[String], name: &str) -> String {
    pack.iter()
        .map(String::as_str)
        .chain(std::iter::once(name))
        .collect::<Vec<_>>()
        .join(".")
}

impl AstLowering<'_> {
    pub(super) fn seed_generic_templates(&mut self, file: &HaxeFile) {
        let pack = file
            .package
            .as_ref()
            .map(|p| p.path.clone())
            .unwrap_or_default();
        let module = std::path::Path::new(&file.filename)
            .file_stem()
            .and_then(|stem| stem.to_str())
            .map(str::to_string);
        for declaration in &file.declarations {
            let TypeDeclaration::Class(class) = declaration else {
                continue;
            };
            self.seed_generic_methods(class, &pack);
            if !is_generic(&class.meta)
                || class.type_params.is_empty()
                || super::const_generics::is_template(class)
                || class.modifiers.contains(&Modifier::Extern)
            {
                continue;
            }
            let private_prefix = match (&class.access, &module) {
                (Some(Access::Private), Some(module)) => {
                    Some(qualified_name(&pack, &format!("_{module}")))
                }
                _ => None,
            };
            self.generic_class_templates.insert(
                qualified_name(&pack, &class.name),
                GenericClass {
                    class: class.clone(),
                    pack: pack.clone(),
                    private_prefix,
                },
            );
        }
    }

    /// A method is specialised only where no override could be dispatched to
    /// instead of the copy.
    fn seed_generic_methods(&mut self, class: &ClassDecl, pack: &[String]) {
        let owner = qualified_name(pack, &class.name);
        let class_params: Vec<String> = class.type_params.iter().map(|p| p.name.clone()).collect();
        for field in &class.fields {
            let ClassFieldKind::Function(function) = &field.kind else {
                continue;
            };
            if field.modifiers.contains(&Modifier::Override) {
                self.generic_overridden_methods
                    .insert(function.name.clone());
            }
            if !is_generic(&field.meta)
                || function.type_params.is_empty()
                || function.body.is_none()
                || function.name == "new"
                || field.modifiers.iter().any(|m| {
                    matches!(
                        m,
                        Modifier::Override | Modifier::Extern | Modifier::Macro | Modifier::Dynamic
                    )
                })
            {
                continue;
            }
            self.generic_method_templates.insert(
                (owner.clone(), function.name.clone()),
                GenericMethod {
                    field: field.clone(),
                    class_params: class_params.clone(),
                    pack: pack.to_vec(),
                },
            );
        }
    }

    /// The type as a specialisation's name spells it; None when it is not
    /// concrete (a type parameter, a monomorph, Dynamic, a function type).
    fn generic_type_name(&self, ty: TypeId, depth: u32) -> Option<String> {
        if depth > 8 {
            return None;
        }
        let kind = self.context.type_table.borrow().get(ty)?.kind.clone();
        let (head, args) = match kind {
            TypeKind::Void => ("Void".to_string(), Vec::new()),
            TypeKind::Bool => ("Bool".to_string(), Vec::new()),
            TypeKind::Int => ("Int".to_string(), Vec::new()),
            TypeKind::Float => ("Float".to_string(), Vec::new()),
            TypeKind::String => ("String".to_string(), Vec::new()),
            TypeKind::Array { element_type } => ("Array".to_string(), vec![element_type]),
            TypeKind::Optional { inner_type } => ("Null".to_string(), vec![inner_type]),
            TypeKind::Map {
                key_type,
                value_type,
            } => ("haxe_ds_Map".to_string(), vec![key_type, value_type]),
            TypeKind::Class {
                symbol_id,
                type_args,
            }
            | TypeKind::Interface {
                symbol_id,
                type_args,
            }
            | TypeKind::Enum {
                symbol_id,
                type_args,
            }
            | TypeKind::Abstract {
                symbol_id,
                type_args,
                ..
            }
            | TypeKind::TypeAlias {
                symbol_id,
                type_args,
                ..
            } => {
                let path = self
                    .context
                    .symbol_table
                    .display_type_path(symbol_id, self.context.string_interner)?;
                let path = path
                    .strip_prefix("StdTypes.")
                    .unwrap_or(&path)
                    .replace('.', "_");
                (path, type_args)
            }
            _ => return None,
        };
        let mut name = head;
        for arg in args {
            name.push('_');
            name.push_str(&self.generic_type_name(arg, depth + 1)?);
        }
        Some(name)
    }

    /// A name `lower_type` and `new` resolve straight to `ty`.
    fn generic_placeholder(&mut self, ty: TypeId) -> String {
        if let Some((name, _)) = self
            .generic_placeholders
            .iter()
            .find(|(_, bound)| **bound == ty)
        {
            return name.clone();
        }
        let name = format!("__generic{}", self.generic_placeholders.len());
        self.generic_placeholders.insert(name.clone(), ty);
        name
    }

    pub(crate) fn generic_placeholder_type(&self, path: &TypePath) -> Option<TypeId> {
        if !path.package.is_empty() || path.sub.is_some() {
            return None;
        }
        self.generic_placeholders.get(&path.name).copied()
    }

    /// The class a type names, through aliases, with its type arguments.
    fn generic_class_of(&self, ty: TypeId) -> Option<(SymbolId, Vec<TypeId>)> {
        let mut ty = ty;
        for _ in 0..8 {
            let kind = self.context.type_table.borrow().get(ty)?.kind.clone();
            match kind {
                TypeKind::Class {
                    symbol_id,
                    type_args,
                } => return Some((symbol_id, type_args)),
                TypeKind::GenericInstance {
                    base_type,
                    type_args,
                    ..
                } => {
                    let base = self
                        .context
                        .type_table
                        .borrow()
                        .get(base_type)?
                        .kind
                        .clone();
                    return match base {
                        TypeKind::Class { symbol_id, .. } => Some((symbol_id, type_args)),
                        _ => None,
                    };
                }
                TypeKind::TypeAlias {
                    symbol_id,
                    target_type,
                    type_args,
                } => {
                    let bindings = self.alias_bindings(symbol_id, &type_args);
                    let next = self.substitute_alias_args(target_type, &bindings);
                    if next == ty {
                        return None;
                    }
                    ty = next;
                }
                _ => return None,
            }
        }
        None
    }

    fn generic_template_of(&self, symbol: SymbolId) -> Option<String> {
        let symbol = self.context.symbol_table.get_symbol(symbol)?;
        if symbol.kind != SymbolKind::Class {
            return None;
        }
        let name = self
            .context
            .string_interner
            .get(symbol.qualified_name.unwrap_or(symbol.name))?;
        self.generic_class_templates
            .contains_key(name)
            .then(|| name.to_string())
    }

    fn class_type_of(&mut self, symbol: SymbolId) -> TypeId {
        let ty = self
            .context
            .symbol_table
            .get_symbol(symbol)
            .map_or(TypeId::invalid(), |s| s.type_id);
        self.ensure_symbol_has_class_type(symbol, ty)
    }

    /// The specialisation of `template` for `args`, lowered on first use. None
    /// when an argument is not concrete or the copy does not lower, which
    /// leaves every use on the template.
    fn generic_class_instance(
        &mut self,
        template: &str,
        args: &[TypeId],
    ) -> LoweringResult<Option<TypeId>> {
        // Keyed by the names Haxe gives the arguments: two TypeIds for one
        // type must share a copy, as they share its name.
        let Some(names) = args
            .iter()
            .map(|arg| self.generic_type_name(*arg, 0))
            .collect::<Option<Vec<_>>>()
        else {
            return Ok(None);
        };
        let key = (template.to_string(), names.clone());
        if let Some(instance) = self.generic_class_instances.get(&key).copied() {
            return Ok(instance.map(|symbol| self.class_type_of(symbol)));
        }
        let Some(generic) = self.generic_class_templates.get(template).cloned() else {
            return Ok(None);
        };
        if self.macro_probe_depth > 0 || args.len() != generic.class.type_params.len() {
            return Ok(None);
        }
        let mut class = generic.class.clone();
        class.name = format!("{}_{}", class.name, names.join("_"));
        let mut types = BTreeMap::new();
        for (param, arg) in generic.class.type_params.iter().zip(args) {
            types.insert(param.name.clone(), self.generic_placeholder(*arg));
        }
        class.type_params.clear();
        class
            .meta
            .retain(|meta| meta.name.trim_start_matches(':') != "generic");
        Substitution::of_types(types).class(&mut class);
        let instance_name = class.name.clone();
        let declaration = TypeDeclaration::Class(class);
        let marks = self.error_marks();
        let mut instance = None;
        let lowered = self.in_generated_context(&generic.pack, |this| {
            let TypeDeclaration::Class(class) = &declaration else {
                return Ok(None);
            };
            this.pre_register_declaration(&declaration)?;
            let Some(symbol) = this.generated_class_symbol(&generic.pack, &instance_name) else {
                return Ok(None);
            };
            // Cached before the body lowers, so the copy can name itself.
            instance = Some(symbol);
            this.generic_class_instances
                .insert(key.clone(), Some(symbol));
            this.generic_class_origins
                .insert(symbol, template.to_string());
            if let Some(prefix) = &generic.private_prefix {
                let path = this
                    .context
                    .intern_string(&format!("{prefix}.{instance_name}"));
                this.context
                    .symbol_table
                    .set_private_type_path(symbol, path);
            }
            this.seed_generic_methods(class, &generic.pack);
            this.pre_register_class_fields(class)?;
            this.lower_declaration(&declaration).map(Some)
        });
        match lowered {
            Ok(Some(typed)) if self.errors_since(marks) == 0 => {
                self.generated_declarations.push(typed);
                Ok(instance.map(|symbol| self.class_type_of(symbol)))
            }
            _ => {
                self.rollback_errors(marks);
                self.generic_class_instances.insert(key, None);
                if let Some(symbol) = instance {
                    self.generic_class_origins.remove(&symbol);
                }
                Ok(None)
            }
        }
    }

    fn generated_class_symbol(&self, pack: &[String], name: &str) -> Option<SymbolId> {
        let name = self.context.string_interner.intern(name);
        let package: Vec<InternedString> = pack
            .iter()
            .map(|p| self.context.string_interner.intern(p))
            .collect();
        let path = if package.is_empty() {
            crate::tast::namespace::QualifiedPath::simple(name)
        } else {
            crate::tast::namespace::QualifiedPath::new(package, name)
        };
        self.context
            .namespace_resolver
            .lookup_symbol(&path)
            .or_else(|| {
                self.context
                    .symbol_table
                    .lookup_symbol(ScopeId::first(), name)
                    .map(|s| s.id)
            })
            .filter(|symbol| {
                self.context
                    .symbol_table
                    .get_symbol(*symbol)
                    .is_some_and(|s| s.kind == SymbolKind::Class)
            })
    }

    fn error_marks(&self) -> (usize, usize, usize) {
        (
            self.context.errors.len(),
            self.collected_errors.len(),
            self.resolution_state.deferred_resolutions.len(),
        )
    }

    fn errors_since(&self, marks: (usize, usize, usize)) -> usize {
        self.context.errors.len().saturating_sub(marks.0)
            + self.collected_errors.len().saturating_sub(marks.1)
    }

    /// A copy that does not lower leaves no trace: the use falls back to the
    /// template as if no copy had been attempted.
    fn rollback_errors(&mut self, marks: (usize, usize, usize)) {
        self.context.errors.truncate(marks.0);
        self.collected_errors.truncate(marks.1);
        self.resolution_state.deferred_resolutions.truncate(marks.2);
    }

    /// `Template<Args>` written as a type: the specialisation, when every
    /// argument is concrete.
    pub(crate) fn generic_class_annotation(
        &mut self,
        symbol: SymbolId,
        params: &[Type],
    ) -> LoweringResult<Option<TypeId>> {
        if self.generic_class_templates.is_empty() || params.is_empty() {
            return Ok(None);
        }
        let Some(template) = self.generic_template_of(symbol) else {
            return Ok(None);
        };
        let mut args = Vec::with_capacity(params.len());
        for param in params {
            match self.lower_type(param) {
                Ok(ty) => args.push(ty),
                Err(_) => return Ok(None),
            }
        }
        self.generic_class_instance(&template, &args)
    }

    /// An alias whose target is a template it binds concretely.
    pub(crate) fn generic_alias_target(
        &mut self,
        alias: SymbolId,
        target: TypeId,
        args: &[TypeId],
    ) -> LoweringResult<Option<TypeId>> {
        if self.generic_class_templates.is_empty() || args.is_empty() {
            return Ok(None);
        }
        let bindings = self.alias_bindings(alias, args);
        let target = self.substitute_alias_args(target, &bindings);
        let Some((symbol, type_args)) = self.generic_class_of(target) else {
            return Ok(None);
        };
        let Some(template) = self.generic_template_of(symbol) else {
            return Ok(None);
        };
        self.generic_class_instance(&template, &type_args)
    }

    /// The specialisation a `new` of a template constructs: its arguments
    /// written out, else taken from the expected type, else bound from the
    /// constructor's annotated parameters.
    pub(crate) fn generic_construction(
        &mut self,
        class_type: TypeId,
        type_path: &TypePath,
        params: &[Type],
        args: &[TypedExpression],
        span: Span,
    ) -> LoweringResult<Option<TypeId>> {
        if self.generic_class_templates.is_empty() {
            return Ok(None);
        }
        let Some((symbol, type_args)) = self.generic_class_of(class_type) else {
            return Ok(None);
        };
        let Some(template) = self.generic_template_of(symbol) else {
            return Ok(None);
        };
        if let Some(instance) = self.generic_class_instance(&template, &type_args)? {
            return Ok(Some(instance));
        }
        if !params.is_empty() {
            let annotation = Type::Path {
                path: type_path.clone(),
                params: params.to_vec(),
                span,
            };
            if let Ok(annotated) = self.lower_type(&annotation)
                && let Some(instance) = self.generic_instance_type(annotated, &template)?
            {
                return Ok(Some(instance));
            }
        }
        if let Some(hint) = self.context.expected_new_type_hint
            && let Some(instance) = self.generic_instance_type(hint, &template)?
        {
            return Ok(Some(instance));
        }
        let Some(inferred) = self.infer_generic_class_args(&template, &type_args, args) else {
            return Ok(None);
        };
        self.generic_class_instance(&template, &inferred)
    }

    /// `ty` as a specialisation of `template`: one already, or the template
    /// with concrete arguments.
    fn generic_instance_type(
        &mut self,
        ty: TypeId,
        template: &str,
    ) -> LoweringResult<Option<TypeId>> {
        let Some((symbol, args)) = self.generic_class_of(ty) else {
            return Ok(None);
        };
        if self
            .generic_class_origins
            .get(&symbol)
            .is_some_and(|origin| origin == template)
        {
            return Ok(Some(self.class_type_of(symbol)));
        }
        if self.generic_template_of(symbol).as_deref() == Some(template) {
            return self.generic_class_instance(template, &args);
        }
        Ok(None)
    }

    /// The template's arguments, each one `known` leaves open bound from the
    /// constructor's annotated parameters against the argument types.
    fn infer_generic_class_args(
        &mut self,
        template: &str,
        known: &[TypeId],
        args: &[TypedExpression],
    ) -> Option<Vec<TypeId>> {
        let generic = self.generic_class_templates.get(template)?.clone();
        let constructor = generic
            .class
            .fields
            .iter()
            .find_map(|field| match &field.kind {
                ClassFieldKind::Function(function) if function.name == "new" => Some(function),
                _ => None,
            })?;
        let (fresh, symbols) = self.fresh_type_params(&generic.class.type_params);
        self.context.push_type_parameters(fresh);
        let mut bindings = Vec::new();
        for (param, arg) in constructor.params.iter().zip(args) {
            if param.rest {
                break;
            }
            let Some(hint) = &param.type_hint else {
                continue;
            };
            if let Ok(declared) = self.lower_type(hint) {
                self.unify_type_args(declared, arg.expr_type, 0, &mut bindings);
            }
        }
        self.context.pop_type_parameters();
        symbols
            .iter()
            .enumerate()
            .map(|(index, symbol)| {
                known
                    .get(index)
                    .copied()
                    .filter(|ty| self.generic_type_name(*ty, 0).is_some())
                    .or_else(|| self.concrete_binding(&bindings, *symbol))
            })
            .collect()
    }

    /// Unconstrained type variables standing for `params`, by name, and their
    /// symbols in declaration order.
    fn fresh_type_params(
        &mut self,
        params: &[TypeParam],
    ) -> (BTreeMap<InternedString, TypeId>, Vec<SymbolId>) {
        let mut map = BTreeMap::new();
        let mut symbols = Vec::with_capacity(params.len());
        for param in params {
            let name = self.context.intern_string(&param.name);
            let symbol = self
                .context
                .symbol_table
                .create_type_parameter(name, Vec::new());
            let ty = self.context.type_table.borrow_mut().create_type_parameter(
                symbol,
                Vec::new(),
                crate::tast::Variance::Invariant,
            );
            map.insert(name, ty);
            symbols.push(symbol);
        }
        (map, symbols)
    }

    fn concrete_binding(
        &self,
        bindings: &[(SymbolId, TypeId)],
        symbol: SymbolId,
    ) -> Option<TypeId> {
        bindings
            .iter()
            .filter(|(bound, _)| *bound == symbol)
            .map(|(_, ty)| *ty)
            .find(|ty| self.generic_type_name(*ty, 0).is_some())
    }

    /// The name a `new` records for a class reached through a placeholder or
    /// a specialisation, where the written path no longer names it.
    pub(crate) fn generic_constructed_name(
        &self,
        type_path: &TypePath,
        class_type: TypeId,
        specialised: bool,
    ) -> Option<String> {
        if !specialised && self.generic_placeholder_type(type_path).is_none() {
            return None;
        }
        let kind = self
            .context
            .type_table
            .borrow()
            .get(class_type)?
            .kind
            .clone();
        match kind {
            TypeKind::String => Some("String".to_string()),
            TypeKind::Array { .. } => Some("Array".to_string()),
            TypeKind::Class { symbol_id, .. } | TypeKind::Abstract { symbol_id, .. } => {
                let symbol = self.context.symbol_table.get_symbol(symbol_id)?;
                self.context
                    .string_interner
                    .get(symbol.qualified_name.unwrap_or(symbol.name))
                    .map(str::to_string)
            }
            _ => None,
        }
    }

    /// The class's own type parameters, by name.
    fn class_param_map(
        &self,
        class: SymbolId,
        names: &[String],
    ) -> BTreeMap<InternedString, TypeId> {
        let Some(params) = self.class_type_params.get(&class) else {
            return BTreeMap::new();
        };
        names
            .iter()
            .zip(params)
            .map(|(name, ty)| (self.context.string_interner.intern(name), *ty))
            .collect()
    }

    /// What the receiver's type arguments bind the class's parameters to.
    fn receiver_param_bindings(
        &self,
        class: SymbolId,
        receiver: TypeId,
    ) -> Vec<(SymbolId, TypeId)> {
        let Some((_, args)) = self.generic_class_of(receiver) else {
            return Vec::new();
        };
        let Some(params) = self.class_type_params.get(&class) else {
            return Vec::new();
        };
        let tt = self.context.type_table.borrow();
        params
            .iter()
            .zip(args)
            .filter_map(|(param, arg)| match tt.get(*param).map(|t| &t.kind) {
                Some(TypeKind::TypeParameter { symbol_id, .. }) => Some((*symbol_id, arg)),
                _ => None,
            })
            .collect()
    }

    fn static_receiver_class(&mut self, receiver: &Expr) -> Option<SymbolId> {
        let ExprKind::Ident(owner) = &receiver.kind else {
            return None;
        };
        let name = self.context.intern_string(owner);
        let symbol = self.resolve_symbol_in_scope_hierarchy(name)?;
        (self.context.symbol_table.get_symbol(symbol)?.kind == SymbolKind::Class).then_some(symbol)
    }

    /// A call of a `@:generic` method of a class lowered in this file,
    /// rewritten to the specialisation its arguments select. Method type
    /// parameters bind from the argument types, then through their
    /// constraints; one that no argument mentions takes its sole constraint.
    pub(crate) fn generic_method_callee(
        &mut self,
        callee: &Expr,
        args: &[Expr],
    ) -> LoweringResult<Option<Expr>> {
        if self.generic_method_templates.is_empty() || self.macro_probe_depth > 0 {
            return Ok(None);
        }
        let (receiver, name): (Option<&Expr>, String) = match &callee.kind {
            ExprKind::Field { expr, field, .. } => (Some(expr.as_ref()), field.clone()),
            ExprKind::Ident(name) => (None, name.clone()),
            _ => return Ok(None),
        };
        if self.generic_overridden_methods.contains(&name)
            || !self
                .generic_method_templates
                .keys()
                .any(|(_, method)| *method == name)
        {
            return Ok(None);
        }
        let (class, receiver_type, call_is_static) = match receiver {
            None => {
                let Some(&class) = self.context.class_context_stack.last() else {
                    return Ok(None);
                };
                let id = self.context.intern_string(&name);
                let is_method = self
                    .resolve_symbol_in_scope_hierarchy(id)
                    .and_then(|symbol| self.context.symbol_table.get_symbol(symbol))
                    .is_some_and(|symbol| symbol.kind == SymbolKind::Function);
                if !is_method {
                    return Ok(None);
                }
                (class, None, None)
            }
            Some(Expr {
                kind: ExprKind::Super,
                ..
            }) => return Ok(None),
            Some(receiver) => match self.static_receiver_class(receiver) {
                Some(class) => (class, None, Some(true)),
                None => {
                    let marks = self.error_marks();
                    let Ok(typed) = self.lower_expression(receiver) else {
                        self.rollback_errors(marks);
                        return Ok(None);
                    };
                    match self.class_value_owner(&typed) {
                        Some(class) => (class, None, Some(true)),
                        None => {
                            let Some(class) = self.resolve_type_to_class_symbol(typed.expr_type)
                            else {
                                return Ok(None);
                            };
                            (class, Some(typed.expr_type), Some(false))
                        }
                    }
                }
            },
        };
        let Some(owner) = self.context.symbol_table.get_symbol(class).and_then(|s| {
            self.context
                .string_interner
                .get(s.qualified_name.unwrap_or(s.name))
                .map(str::to_string)
        }) else {
            return Ok(None);
        };
        let Some(generic) = self
            .generic_method_templates
            .get(&(owner, name.clone()))
            .cloned()
        else {
            return Ok(None);
        };
        let ClassFieldKind::Function(function) = &generic.field.kind else {
            return Ok(None);
        };
        let is_static = generic.field.modifiers.contains(&Modifier::Static);
        if call_is_static.is_some_and(|call| call != is_static)
            || !self.class_methods.contains_key(&class)
        {
            return Ok(None);
        }

        let class_params = self.class_param_map(class, &generic.class_params);
        let (fresh, method_params) = self.fresh_type_params(&function.type_params);
        self.context.push_type_parameters(class_params.clone());
        self.context.push_type_parameters(fresh);
        let constraints: Vec<Vec<TypeId>> = function
            .type_params
            .iter()
            .map(|param| {
                param
                    .constraints
                    .iter()
                    .filter_map(|constraint| self.lower_type(constraint).ok())
                    .collect()
            })
            .collect();
        let declared: Vec<Option<TypeId>> = function
            .params
            .iter()
            .map(|param| {
                param
                    .type_hint
                    .as_ref()
                    .and_then(|hint| self.lower_type(hint).ok())
            })
            .collect();
        self.context.pop_type_parameters();
        self.context.pop_type_parameters();

        let own: BTreeSet<SymbolId> = method_params.iter().copied().collect();
        let mut in_values = BTreeSet::new();
        for ty in declared.iter().flatten() {
            self.collect_type_param_symbols(*ty, 0, &mut in_values);
        }
        let mut bindings = receiver_type
            .map(|ty| self.receiver_param_bindings(class, ty))
            .unwrap_or_default();
        for (index, arg) in args.iter().enumerate() {
            let Some(param) = function.params.get(index) else {
                break;
            };
            if param.rest {
                break;
            }
            let Some(Some(declared)) = declared.get(index) else {
                continue;
            };
            // A literal function's parameters are typed by the parameter it
            // is passed to, so it binds nothing on its own.
            if matches!(
                arg.kind,
                ExprKind::Null | ExprKind::Function(_) | ExprKind::Arrow { .. }
            ) {
                continue;
            }
            let mut mentioned = BTreeSet::new();
            self.collect_type_param_symbols(*declared, 0, &mut mentioned);
            if mentioned.is_disjoint(&own) {
                continue;
            }
            let marks = self.error_marks();
            let Ok(actual) = self.lower_expression(arg) else {
                self.rollback_errors(marks);
                return Ok(None);
            };
            self.unify_type_args(*declared, actual.expr_type, 0, &mut bindings);
        }
        let mut bound = 0;
        loop {
            for (index, symbol) in method_params.iter().enumerate() {
                match self.concrete_binding(&bindings, *symbol) {
                    Some(ty) => {
                        for constraint in &constraints[index] {
                            self.unify_type_args(*constraint, ty, 0, &mut bindings);
                        }
                    }
                    None if !in_values.contains(symbol) && constraints[index].len() == 1 => {
                        let concrete: Vec<(SymbolId, TypeId)> = bindings
                            .iter()
                            .copied()
                            .filter(|(_, ty)| self.generic_type_name(*ty, 0).is_some())
                            .collect();
                        let ty = self.substitute_alias_args(constraints[index][0], &concrete);
                        if self.generic_type_name(ty, 0).is_some() {
                            bindings.push((*symbol, ty));
                        }
                    }
                    None => {}
                }
            }
            let now = method_params
                .iter()
                .filter(|symbol| self.concrete_binding(&bindings, **symbol).is_some())
                .count();
            if now == bound {
                break;
            }
            bound = now;
        }

        let mut names = Vec::with_capacity(method_params.len());
        let mut placeholders = BTreeMap::new();
        for (param, symbol) in function.type_params.iter().zip(&method_params) {
            let Some(ty) = self.concrete_binding(&bindings, *symbol) else {
                return Ok(None);
            };
            let Some(type_name) = self.generic_type_name(ty, 0) else {
                return Ok(None);
            };
            names.push(type_name);
            placeholders.insert(param.name.clone(), self.generic_placeholder(ty));
        }
        let specialised = format!("{name}_{}", names.join("_"));
        match self
            .generic_method_instances
            .get(&(class, specialised.clone()))
        {
            Some(true) => {}
            Some(false) => return Ok(None),
            None => {
                if !self.instantiate_generic_method(
                    class,
                    &generic,
                    &specialised,
                    placeholders,
                    class_params,
                ) {
                    return Ok(None);
                }
            }
        }
        let kind = match &callee.kind {
            ExprKind::Field {
                expr, is_optional, ..
            } => ExprKind::Field {
                expr: expr.clone(),
                field: specialised,
                is_optional: *is_optional,
            },
            _ => ExprKind::Ident(specialised),
        };
        Ok(Some(Expr {
            kind,
            span: callee.span,
        }))
    }

    /// Lower the copy of `generic` named `specialised` into `class`, beside
    /// the methods it already has. False when the copy does not lower.
    fn instantiate_generic_method(
        &mut self,
        class: SymbolId,
        generic: &GenericMethod,
        specialised: &str,
        placeholders: BTreeMap<String, String>,
        class_params: BTreeMap<InternedString, TypeId>,
    ) -> bool {
        let key = (class, specialised.to_string());
        let Some(class_scope) = self
            .context
            .symbol_table
            .get_symbol(class)
            .map(|s| s.scope_id)
        else {
            return false;
        };
        let mut field = generic.field.clone();
        field
            .meta
            .retain(|meta| meta.name.trim_start_matches(':') != "generic");
        let ClassFieldKind::Function(function) = &mut field.kind else {
            return false;
        };
        function.name = specialised.to_string();
        function.type_params.clear();
        Substitution::of_types(placeholders).function(function);
        let function = function.clone();
        let is_static = field.modifiers.contains(&Modifier::Static);
        let visibility = match field.access {
            Some(Access::Public) => Visibility::Public,
            Some(Access::Private) => Visibility::Private,
            None => Visibility::Internal,
        };

        // Registered before the body lowers, so a recursive call finds it. A
        // second lowering of this file finds the symbol the first one made.
        let name = self.context.intern_string(specialised);
        let existing = self
            .context
            .symbol_table
            .lookup_symbol(class_scope, name)
            .filter(|s| s.kind == SymbolKind::Function)
            .map(|s| s.id);
        let symbol = match existing {
            Some(symbol) => symbol,
            None => {
                let symbol = self
                    .context
                    .symbol_table
                    .create_function_in_scope(name, class_scope);
                if let Some(scope) = self.context.scope_tree.get_scope_mut(class_scope) {
                    scope.add_symbol(symbol, name);
                }
                symbol
            }
        };
        if is_static {
            self.context
                .symbol_table
                .add_symbol_flags(symbol, SymbolFlags::STATIC);
        }
        self.context.update_symbol_qualified_name(symbol);
        if let Some(entry) = self.context.symbol_table.get_symbol_mut(symbol) {
            entry.visibility = visibility;
        }
        if let Some(methods) = self.class_methods.get_mut(&class)
            && !methods.iter().any(|(_, method, _)| *method == symbol)
        {
            methods.push((name, symbol, is_static));
        }
        self.generic_method_instances.insert(key.clone(), true);

        let marks = self.error_marks();
        let method_name = self.current_method_name;
        let lowered = self.in_generated_context(&generic.pack, |this| {
            this.context.current_scope = class_scope;
            this.context.class_context_stack.push(class);
            this.context.push_type_parameters(class_params);
            this.lower_function_from_field(&field, &function)
        });
        self.current_method_name = method_name;
        match lowered {
            Ok(typed) if self.errors_since(marks) == 0 => {
                self.generic_method_functions.push((class, typed));
                true
            }
            _ => {
                self.rollback_errors(marks);
                if let Some(methods) = self.class_methods.get_mut(&class) {
                    methods.retain(|(_, method, _)| *method != symbol);
                }
                if let Some(fields) = self.class_fields.get_mut(&class) {
                    fields.retain(|(_, method, _)| *method != symbol);
                }
                self.generic_method_instances.insert(key, false);
                false
            }
        }
    }

    /// Hand each lowered method copy to the class it belongs to.
    pub(crate) fn attach_generic_methods(&mut self, classes: &mut [TypedClass]) {
        for (class, method) in std::mem::take(&mut self.generic_method_functions) {
            if let Some(typed) = classes.iter_mut().find(|typed| typed.symbol_id == class) {
                typed.methods.push(method);
            }
        }
    }
}
