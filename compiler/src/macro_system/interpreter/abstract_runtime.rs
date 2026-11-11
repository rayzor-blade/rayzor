use super::*;
use crate::macro_system::value::MacroFunctionContext;

impl MacroInterpreter {
    pub(super) fn function_context(&self) -> MacroFunctionContext {
        MacroFunctionContext {
            imports: self.import_map.clone(),
            source_file: self.source_file.clone(),
            owner: self.macro_class_stack.last().cloned(),
        }
    }

    pub(super) fn type_name_path(&self, path: &parser::TypePath) -> String {
        let mut parts = path.package.clone();
        parts.push(path.name.clone());
        if let Some(sub) = &path.sub {
            parts.push(sub.clone());
        }
        self.resolve_class_name(&parts.join("."))
    }

    pub(super) fn value_matches_type(&self, value: &MacroValue, ty: &parser::Type) -> bool {
        self.value_matches_type_at_depth(value, ty, 0)
    }

    fn value_matches_type_at_depth(
        &self,
        value: &MacroValue,
        ty: &parser::Type,
        depth: usize,
    ) -> bool {
        use parser::Type;
        if depth >= 64 {
            return false;
        }
        match ty {
            Type::Optional { inner, .. } => {
                matches!(value, MacroValue::Null)
                    || self.value_matches_type_at_depth(value, inner, depth + 1)
            }
            Type::Parenthesis { inner, .. } | Type::Named { inner, .. } => {
                self.value_matches_type_at_depth(value, inner, depth + 1)
            }
            Type::Function { .. } => matches!(value, MacroValue::Function(_)),
            Type::Anonymous { fields, .. } => value
                .as_object()
                .is_some_and(|object| fields.iter().all(|field| object.contains_key(&field.name))),
            Type::Path { path, params, .. } => {
                let name = self.type_name_path(path);
                match name.as_str() {
                    "Dynamic" => true,
                    "String" => matches!(value, MacroValue::String(_)),
                    "Int" => matches!(value, MacroValue::Int(_)),
                    "Float" | "Single" => {
                        matches!(value, MacroValue::Float(_) | MacroValue::Int(_))
                    }
                    "Bool" => matches!(value, MacroValue::Bool(_)),
                    "haxe.macro.Position" | "haxe.macro.Expr.Position" => {
                        matches!(value.underlying(), MacroValue::Position(_))
                    }
                    "Array" => matches!(value, MacroValue::Array(_)),
                    "haxe.ds.Vector" | "Vector" => {
                        matches!(value,MacroValue::Abstract(owner,_) if owner.as_ref()=="haxe.ds.Vector")
                    }
                    "IntIterator" => value.as_object().is_some_and(|object| {
                        object.get("__type__").and_then(MacroValue::as_string)
                            == Some("IntIterator")
                    }),
                    "Null" => {
                        matches!(value, MacroValue::Null)
                            || params.first().is_some_and(|ty| {
                                self.value_matches_type_at_depth(value, ty, depth + 1)
                            })
                    }
                    // Structural: anything with an `iterator()`, as `Lambda` takes it.
                    "Iterable" | "KeyValueIterable" => {
                        let method = if name == "Iterable" {
                            "iterator"
                        } else {
                            "keyValueIterator"
                        };
                        matches!(value, MacroValue::Array(_) | MacroValue::Map(_))
                            || value.as_object().is_some_and(|object| {
                                object.contains_key(method)
                                    || object
                                        .get("__type__")
                                        .and_then(MacroValue::as_string)
                                        .is_some_and(|owner| {
                                            self.class_registry.as_ref().is_some_and(|registry| {
                                                registry
                                                    .find_instance_method(owner, method)
                                                    .is_some()
                                            })
                                        })
                            })
                    }
                    _ if path.name == "Expr" => is_expr_value(value),
                    _ => {
                        if let MacroValue::Enum(owner, _, _) = value {
                            // A haxe.macro enum value carries its short name
                            // (`ExprDef`) for `haxe.macro.Expr.ExprDef`.
                            let builtin_owner = !owner.contains('.')
                                && name.starts_with("haxe.macro.")
                                && name.rsplit('.').next() == Some(owner.as_ref());
                            if owner.as_ref() == name
                                || builtin_owner
                                || self
                                    .class_registry
                                    .as_ref()
                                    .is_some_and(|registry| registry.is_subclass(&owner, &name))
                            {
                                return true;
                            }
                        }
                        if let MacroValue::Abstract(owner, _) = value {
                            if owner.as_ref() == name {
                                return true;
                            }
                        }
                        if let Some(MacroValue::String(owner)) = value
                            .as_object()
                            .and_then(|object| object.get("__type__").cloned())
                        {
                            if self
                                .class_registry
                                .as_ref()
                                .is_some_and(|registry| registry.is_subclass(&owner, &name))
                            {
                                return true;
                            }
                        }
                        self.class_registry
                            .as_ref()
                            .and_then(|registry| registry.find_alias(&name))
                            .is_some_and(|(ty, _)| match ty {
                                Type::Anonymous { fields, .. } => {
                                    value.as_object().is_some_and(|object| {
                                        fields.iter().all(|field| object.contains_key(&field.name))
                                    })
                                }
                                Type::Path { path, .. } => {
                                    let leaf = path.sub.as_deref().unwrap_or(&path.name);
                                    match leaf {
                                        "Array" => matches!(value,MacroValue::Array(_)),
                                        "Vector" => matches!(value,MacroValue::Abstract(owner,_) if owner.as_ref()=="haxe.ds.Vector"),
                                        _ => self.value_matches_type_at_depth(value, ty, depth + 1),
                                    }
                                }
                                _ => false,
                            })
                    }
                }
            }
            _ => false,
        }
    }

    pub(super) fn coerce_value(
        &mut self,
        value: MacroValue,
        ty: Option<&parser::Type>,
        location: SourceLocation,
    ) -> Result<MacroValue, MacroError> {
        let Some(ty) = ty else { return Ok(value) };
        self.coerce_type(value, ty, location, 0)
    }

    fn coerce_type(
        &mut self,
        value: MacroValue,
        ty: &parser::Type,
        location: SourceLocation,
        depth: usize,
    ) -> Result<MacroValue, MacroError> {
        use parser::Type;
        if depth >= 64 {
            return Err(MacroError::TypeError {
                message: "cyclic macro conversion".into(),
                location,
            });
        }
        match ty {
            Type::Optional { inner, .. }
            | Type::Parenthesis { inner, .. }
            | Type::Named { inner, .. } => self.coerce_type(value, inner, location, depth + 1),
            Type::Anonymous { fields, .. } => {
                let MacroValue::Object(original) = &value else {
                    return Ok(value);
                };
                let mut updates = Vec::new();
                for field in fields {
                    if let Some(original) = original.read().get(&field.name) {
                        let converted = self.coerce_type(
                            original.clone(),
                            &field.type_hint,
                            location,
                            depth + 1,
                        )?;
                        if !original.same_reference(&converted) && original != &converted {
                            updates.push((field.name.clone(), converted));
                        }
                    }
                }
                if updates.is_empty() {
                    return Ok(value);
                }
                original.write(|fields| fields.extend(updates));
                Ok(value)
            }
            Type::Path { path, params, .. } => {
                if path.package.is_empty()
                    && path.sub.is_none()
                    && !self.import_map.contains_key(&path.name)
                    && matches!(
                        (path.name.as_str(), &value),
                        ("String", MacroValue::String(_))
                            | ("Int", MacroValue::Int(_))
                            | ("Float", MacroValue::Float(_) | MacroValue::Int(_))
                            | ("Bool", MacroValue::Bool(_))
                            | ("Array", MacroValue::Array(_))
                            | ("Dynamic", _)
                    )
                {
                    return Ok(value);
                }
                let name = self.type_name_path(path);
                if path.name == "Null" {
                    return match params.first() {
                        Some(inner) => self.coerce_type(value, inner, location, depth + 1),
                        None => Ok(value),
                    };
                }
                if let Some((target, imports)) = self
                    .class_registry
                    .as_ref()
                    .and_then(|registry| registry.find_alias(&name))
                    .cloned()
                {
                    let previous = std::mem::replace(&mut self.import_map, imports.clone());
                    let result = self.coerce_type(value, &target, location, depth + 1);
                    self.import_map = previous;
                    return result;
                }
                if let MacroValue::Object(fields) = &value
                    && !fields.read().contains_key("__type__")
                    && self
                        .class_registry
                        .as_ref()
                        .is_some_and(|registry| registry.is_struct_init(&name))
                {
                    return self.struct_init_instance(&name, fields, location);
                }
                if let MacroValue::Abstract(owner, inner) = &value {
                    if owner.as_ref() == name {
                        return Ok(value);
                    }
                    if self.value_matches_type(inner, ty) {
                        return Ok(inner.as_ref().clone());
                    }
                    if !matches!(name.as_str(), "String" | "Int" | "Float" | "Bool" | "Array")
                        && self
                            .class_registry
                            .as_ref()
                            .and_then(|registry| registry.find_executable_class(&name))
                            .is_none()
                    {
                        return Ok(value);
                    }
                    let owner = owner.to_string();
                    let to = self
                        .class_registry
                        .as_ref()
                        .and_then(|registry| registry.find_executable_class(&owner))
                        .and_then(|class| class.abstract_decl.as_ref())
                        .and_then(|decl| {
                            decl.fields.iter().find_map(|field| {
                                let parser::ClassFieldKind::Function(method) = &field.kind else {
                                    return None;
                                };
                                (field
                                    .meta
                                    .iter()
                                    .any(|meta| meta.name.trim_start_matches(':') == "to")
                                    && (method.return_type.is_none()
                                        || method.return_type.as_ref().is_some_and(|target| {
                                            match target {
                                                Type::Path { path, .. } => {
                                                    path.name == name || path.name == path_name(ty)
                                                }
                                                _ => false,
                                            }
                                        })))
                                .then(|| method.name.clone())
                            })
                        });
                    if let Some(method) = to {
                        return self.invoke_abstract(
                            &owner,
                            inner.as_ref().clone(),
                            &method,
                            vec![],
                            location,
                        );
                    }
                    if self.value_matches_type(inner, ty) {
                        return Ok(inner.as_ref().clone());
                    }
                }
                let declaration =
                    self.class_registry
                        .as_ref()
                        .and_then(|registry| registry.find_executable_class(&name))
                        .and_then(|class| {
                            class.abstract_decl.clone().map(|decl| {
                                (decl, class.imports.clone(), class.qualified_name.clone())
                            })
                        });
                let Some((decl, imports, owner)) = declaration else {
                    return Ok(value);
                };
                if matches!(value, MacroValue::Null)
                    && decl
                        .underlying
                        .as_ref()
                        .is_none_or(|ty| path_name(ty) != "Dynamic")
                {
                    return Ok(value);
                }
                let previous = std::mem::replace(&mut self.import_map, imports.clone());
                let accepted = decl
                    .from
                    .iter()
                    .any(|source| self.value_matches_type(&value, source));
                let from = decl.fields.iter().find_map(|field| {
                    let parser::ClassFieldKind::Function(method) = &field.kind else {
                        return None;
                    };
                    (field
                        .meta
                        .iter()
                        .any(|meta| meta.name.trim_start_matches(':') == "from")
                        && method
                            .params
                            .first()
                            .and_then(|param| param.type_hint.as_ref())
                            .is_some_and(|source| self.value_matches_type(&value, source)))
                    .then(|| method.name.clone())
                });
                let underlying = decl
                    .underlying
                    .as_ref()
                    .is_some_and(|underlying| self.value_matches_type(&value, underlying));
                self.import_map = previous;
                if accepted {
                    return Ok(MacroValue::Abstract(Arc::from(owner), Arc::new(value)));
                }
                if let Some(method) = from {
                    let converted = self
                        .try_registered_static_call(&owner, &method, &[value], location)?
                        .ok_or_else(|| MacroError::TypeError {
                            message: format!("missing abstract conversion {owner}.{method}"),
                            location,
                        })?;
                    if matches!(&converted, MacroValue::Abstract(name, _) if name.as_ref() == owner)
                    {
                        return Ok(converted);
                    }
                    return Ok(MacroValue::Abstract(Arc::from(owner), Arc::new(converted)));
                }
                if underlying {
                    return Ok(MacroValue::Abstract(Arc::from(owner), Arc::new(value)));
                }
                Ok(value)
            }
            _ => Ok(value),
        }
    }

    pub(super) fn construct_abstract(
        &mut self,
        owner: &str,
        args: Vec<MacroValue>,
        location: SourceLocation,
    ) -> Result<MacroValue, MacroError> {
        let class = self
            .class_registry
            .as_ref()
            .and_then(|registry| registry.find_executable_class(owner))
            .unwrap();
        let constructor = class
            .constructor
            .as_ref()
            .map(|method| (method.body.clone(), method.params.clone()));
        let value =
            self.construct_from_registry_data(owner, &[], constructor.as_ref(), args, location)?;
        Ok(MacroValue::Abstract(Arc::from(owner), Arc::new(value)))
    }

    pub(super) fn run_parent_constructor(
        &mut self,
        owner: &str,
        args: Vec<MacroValue>,
        location: SourceLocation,
    ) -> Result<MacroValue, MacroError> {
        let method = self
            .class_registry
            .as_ref()
            .and_then(|registry| registry.find_executable_class(owner))
            .and_then(|class| class.constructor.as_ref())
            .map(|method| (method.body.clone(), method.params.clone()));
        let Some((body, params)) = method else {
            return Ok(MacroValue::Null);
        };
        self.invoke_declared(
            owner,
            body,
            &params,
            None,
            args,
            self.env.get("this"),
            location,
        )
    }

    /// `{f: v, ...}` typed as a `@:structInit` class becomes an instance in
    /// place: its constructor takes the fields by parameter name; without one
    /// the fields are assigned over the declared defaults.
    fn struct_init_instance(
        &mut self,
        name: &str,
        fields: &Shared<BTreeMap<String, MacroValue>>,
        location: SourceLocation,
    ) -> Result<MacroValue, MacroError> {
        let literal: BTreeMap<String, MacroValue> = fields
            .read()
            .iter()
            .map(|(field, value)| (field.clone(), value.clone()))
            .collect();
        let params = self
            .class_registry
            .as_ref()
            .and_then(|registry| registry.find_executable_class(name))
            .and_then(|class| class.constructor.as_ref())
            .map(|constructor| constructor.params.clone());
        let Some(params) = params else {
            let types = self
                .class_registry
                .as_ref()
                .and_then(|registry| registry.find_executable_class(name))
                .map(|class| (class.field_types.clone(), class.imports.clone()));
            let mut values = Vec::with_capacity(literal.len());
            for (field, value) in literal {
                let declared = types.as_ref().and_then(|(types, imports)| {
                    Some((types.get(&field)?.clone(), imports.clone()))
                });
                let value = match declared {
                    Some((ty, imports)) => {
                        let previous = std::mem::replace(&mut self.import_map, imports);
                        let value = self.coerce_value(value, Some(&ty), location);
                        self.import_map = previous;
                        value?
                    }
                    None => value,
                };
                values.push((field, value));
            }
            let instance = self.construct_named(name, name.to_string(), Vec::new(), location)?;
            return Ok(Self::become_instance(fields, &instance, values));
        };
        let args = params
            .iter()
            .map(|param| {
                literal
                    .get(&param.name)
                    .cloned()
                    .unwrap_or(MacroValue::Null)
            })
            .collect();
        let instance = self.construct_named(name, name.to_string(), args, location)?;
        Ok(Self::become_instance(fields, &instance, Vec::new()))
    }

    /// The literal object takes the instance's fields, then `overrides`, so
    /// every reference to it, a map key among them, sees one instance.
    fn become_instance(
        literal: &Shared<BTreeMap<String, MacroValue>>,
        instance: &MacroValue,
        overrides: Vec<(String, MacroValue)>,
    ) -> MacroValue {
        let MacroValue::Object(built) = instance else {
            return instance.clone();
        };
        let mut fields = (*built.read()).clone();
        fields.extend(overrides);
        literal.set(fields);
        MacroValue::Object(literal.clone())
    }

    pub(super) fn try_extension(
        &mut self,
        value: &MacroValue,
        method: &str,
        args: &[MacroValue],
        location: SourceLocation,
    ) -> Result<Option<MacroValue>, MacroError> {
        let owners = self
            .import_map
            .get(USING_OWNERS_KEY)
            .map(|owners| {
                owners
                    .split('\n')
                    .rev()
                    .map(str::to_string)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let Some(registry) = self.class_registry.clone() else {
            return Ok(None);
        };
        for module in owners {
            let owners = registry
                .using_members
                .get(&module)
                .cloned()
                .unwrap_or_else(|| vec![module]);
            for owner in owners {
                let Some(class) = registry.find_executable_class(&owner) else {
                    continue;
                };
                let Some(candidate) = class.static_methods.get(method) else {
                    continue;
                };
                let previous = std::mem::replace(&mut self.import_map, class.imports.clone());
                let accepts = candidate.params.first().is_some_and(|param| {
                    param
                        .type_hint
                        .as_ref()
                        .is_none_or(|ty| self.value_matches_type(value, ty))
                });
                self.import_map = previous;
                if accepts {
                    let mut all = vec![value.clone()];
                    all.extend_from_slice(args);
                    return self.try_registered_static_call(&owner, method, &all, location);
                }
            }
        }
        Ok(None)
    }

    pub(super) fn abstract_array_access(&self, owner: &str, index: &MacroValue) -> Option<String> {
        let class = self.class_registry.as_ref()?.find_executable_class(owner)?;
        class
            .abstract_decl
            .as_ref()?
            .fields
            .iter()
            .find_map(|field| {
                let parser::ClassFieldKind::Function(method) = &field.kind else {
                    return None;
                };
                (field
                    .meta
                    .iter()
                    .any(|meta| meta.name.trim_start_matches(':') == "arrayAccess")
                    && method.params.len() == 1
                    && method.params[0]
                        .type_hint
                        .as_ref()
                        .is_none_or(|ty| self.value_matches_type(index, ty)))
                .then(|| method.name.clone())
            })
    }

    pub(super) fn abstract_binary(
        &mut self,
        op: &parser::BinaryOp,
        left: &MacroValue,
        right: &MacroValue,
        location: SourceLocation,
    ) -> Result<Option<MacroValue>, MacroError> {
        let Some(registry) = self.class_registry.clone() else {
            return Ok(None);
        };
        let negated = *op == parser::BinaryOp::NotEq;
        let declared_op = if negated { &parser::BinaryOp::Eq } else { op };
        for value in [left, right] {
            let MacroValue::Abstract(owner, _) = value else {
                continue;
            };
            let Some(class) = registry.find_executable_class(owner) else {
                continue;
            };
            let Some(decl) = &class.abstract_decl else {
                continue;
            };
            let mut candidates = Vec::new();
            for field in &decl.fields {
                let parser::ClassFieldKind::Function(method) = &field.kind else {
                    continue;
                };
                if field.meta.iter().any(|meta|meta.name.trim_start_matches(':')=="op" && meta.params.first().is_some_and(|expr|matches!(&expr.kind,parser::ExprKind::Binary {op:declared,..} if declared==declared_op))) {
                    candidates.push(method);
                }
            }
            let previous = std::mem::replace(&mut self.import_map, class.imports.clone());
            let method = candidates
                .iter()
                .find(|method| {
                    method
                        .params
                        .iter()
                        .zip([left, right])
                        .all(|(param, value)| {
                            param
                                .type_hint
                                .as_ref()
                                .is_none_or(|ty| {
                                    matches!(ty, parser::Type::Path { path, .. } if path.package.is_empty() && method.type_params.iter().any(|parameter| parameter.name == path.name))
                                        || self.value_matches_type(value, ty)
                                })
                        })
                })
                .or_else(|| candidates.first())
                .map(|method| method.name.clone());
            self.import_map = previous;
            if let Some(method) = method {
                let result = self.try_registered_static_call(
                    owner,
                    &method,
                    &[left.clone(), right.clone()],
                    location,
                )?;
                return Ok(result.map(|value| {
                    if negated {
                        MacroValue::Bool(!value.is_truthy())
                    } else {
                        value
                    }
                }));
            }
        }
        Ok(None)
    }

    pub(super) fn match_quoted_value(
        &mut self,
        value: &MacroValue,
        pattern: &MacroValue,
    ) -> Result<bool, MacroError> {
        if let MacroValue::Expr(expr) = pattern {
            match &expr.kind {
                ExprKind::Ident(name) if name == "_" => return Ok(true),
                ExprKind::DollarIdent { name, arg: None } => {
                    self.env.define(name, value.clone());
                    return Ok(true);
                }
                ExprKind::DollarIdent {
                    name,
                    arg: Some(pattern),
                } => {
                    if let ExprKind::Ident(binding) = &pattern.kind {
                        self.env.define(
                            binding,
                            if name == "v" {
                                ast_bridge::unwrap_expr_value(value)
                            } else {
                                value.clone()
                            },
                        );
                        return Ok(true);
                    }
                    return self.match_pattern(value, &pattern_of_expr(pattern));
                }
                _ => {}
            }
            let actual = match value {
                MacroValue::Expr(expr) => ast_bridge::expr_kind_to_value(&expr.kind, expr.span),
                MacroValue::Object(object) => object
                    .read()
                    .get("expr")
                    .cloned()
                    .unwrap_or(MacroValue::Null),
                _ => return Ok(false),
            };
            let pattern = ast_bridge::expr_kind_to_value(&expr.kind, expr.span);
            return self.match_quoted_value(&actual, &pattern);
        }
        match (value, pattern) {
            (MacroValue::Enum(_, actual, args), MacroValue::Enum(_, expected, patterns))
                if actual == expected && args.len() == patterns.len() =>
            {
                for (value, pattern) in args.iter().zip(patterns.iter()) {
                    if !self.match_quoted_value(value, pattern)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            (MacroValue::Array(values), MacroValue::Array(patterns))
                if values.read().len() == patterns.read().len() =>
            {
                for (value, pattern) in values.read().iter().zip(patterns.read().iter()) {
                    if !self.match_quoted_value(value, pattern)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            (MacroValue::Object(values), MacroValue::Object(patterns)) => {
                for (name, pattern) in patterns
                    .read()
                    .iter()
                    .filter(|(name, _)| name.as_str() != "pos")
                {
                    if !self.match_quoted_value(
                        values.read().get(name).unwrap_or(&MacroValue::Null),
                        pattern,
                    )? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            _ => Ok(value == pattern),
        }
    }

    pub(super) fn invoke_abstract(
        &mut self,
        owner: &str,
        this: MacroValue,
        method: &str,
        args: Vec<MacroValue>,
        location: SourceLocation,
    ) -> Result<MacroValue, MacroError> {
        let method = self
            .class_registry
            .as_ref()
            .and_then(|registry| registry.find_instance_method(owner, method))
            .map(|method| {
                (
                    method.body.clone(),
                    method.params.clone(),
                    method.return_type.clone(),
                    method.owner.clone(),
                )
            })
            .ok_or_else(|| MacroError::UnsupportedOperation {
                operation: format!("{owner}.{method}"),
                location,
            })?;
        self.invoke_declared(
            &method.3,
            method.0,
            &method.1,
            method.2,
            args,
            Some(this),
            location,
        )
    }

    pub(super) fn invoke_declared(
        &mut self,
        owner: &str,
        body: Arc<Expr>,
        params: &[parser::FunctionParam],
        return_type: Option<parser::Type>,
        args: Vec<MacroValue>,
        this: Option<MacroValue>,
        location: SourceLocation,
    ) -> Result<MacroValue, MacroError> {
        let context = self
            .class_registry
            .as_ref()
            .and_then(|registry| registry.find_executable_class(owner))
            .map(|class| MacroFunctionContext {
                imports: class.imports.clone(),
                source_file: Some(class.source_file.clone()),
                owner: Some(class.qualified_name.clone()),
            });
        let mut func = MacroFunction {
            name: format!("{owner}.<method>"),
            params: params.iter().map(MacroParam::from_function_param).collect(),
            body,
            captures: Default::default(),
            context,
            return_type,
            static_target: None,
        };
        self.env.push_scope();
        if let Some(value) = &this {
            self.env.define("this", value.clone());
            func.captures = self
                .env
                .capture_used(&std::collections::BTreeSet::from(["this".to_string()]));
        }
        let result = self.call_function(&func, args, location);
        let updated = self.env.get("this");
        self.env.pop_scope();
        self.updated_this = None;
        if let Some((old, new)) = this.zip(updated) {
            self.replace_references(&old, &new);
            if !old.haxe_equals(&new) {
                self.updated_this = Some(new);
            }
        }
        result
    }
}

fn path_name(ty: &parser::Type) -> &str {
    match ty {
        parser::Type::Path { path, .. } => &path.name,
        _ => "",
    }
}
