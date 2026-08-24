//! Pattern matching: destructuring, bindings and placeholders.

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
    /// Check if a pattern contains variables that need binding
    pub(crate) fn pattern_has_variables(&self, pattern: &parser::Pattern) -> bool {
        use parser::Pattern;
        match pattern {
            Pattern::Var(_) => true,
            Pattern::Constructor { params, .. } => {
                // If the constructor has parameters, check if any are variables
                params.iter().any(|p| self.pattern_has_variables(p))
            }
            Pattern::Array(patterns) => {
                // Check if any array element patterns have variables
                patterns.iter().any(|p| self.pattern_has_variables(p))
            }
            Pattern::ArrayRest { elements, rest, .. } => {
                // Check elements and rest variable
                rest.is_some() || elements.iter().any(|p| self.pattern_has_variables(p))
            }
            Pattern::Object { fields, .. } => {
                // Check if any field patterns have variables
                fields.iter().any(|(_, p)| self.pattern_has_variables(p))
            }
            Pattern::Type { var, .. } => {
                // Type patterns always bind a variable
                true
            }
            Pattern::Or(patterns) => {
                // Or patterns can have variables in any branch
                patterns.iter().any(|p| self.pattern_has_variables(p))
            }
            Pattern::Const(_) | Pattern::Null | Pattern::Underscore => false,
            Pattern::Extractor { .. } => {
                // Extractors might bind variables - for now assume they do
                true
            }
        }
    }

    /// Bind pattern variables in the current scope
    pub(crate) fn bind_pattern_variables(
        &mut self,
        pattern: &parser::Pattern,
    ) -> Result<Vec<(InternedString, SymbolId)>, LoweringError> {
        self.bind_pattern_variables_typed(pattern, None)
    }

    /// Bind pattern variables, propagating an expected type when known
    /// (e.g. for `case JString(s):` where `s` should be typed as String
    /// from the JString variant's parameter type).
    fn bind_pattern_variables_typed(
        &mut self,
        pattern: &parser::Pattern,
        expected_type: Option<TypeId>,
    ) -> Result<Vec<(InternedString, SymbolId)>, LoweringError> {
        use parser::Pattern;
        match pattern {
            Pattern::Var(var_name) => {
                let interned_name = self.context.intern_string(var_name);
                let type_id = expected_type.unwrap_or(TypeId::invalid());
                let var_symbol = self.context.symbol_table.create_variable_with_type(
                    interned_name,
                    self.context.current_scope,
                    type_id,
                );

                self.context
                    .scope_tree
                    .get_scope_mut(self.context.current_scope)
                    .expect("Current scope should exist")
                    .add_symbol(var_symbol, interned_name);

                Ok(vec![(interned_name, var_symbol)])
            }
            Pattern::Constructor { path, params } => {
                // Resolve the constructor's parameter types so sub-pattern
                // variable bindings get proper type info (e.g. JString(s)
                // where s is String). Without this, `s.length` later fails
                // because the destructured variable has TypeId::invalid().
                let ctor_name = self.context.intern_string(&path.name);
                let ctor_sym = self
                    .resolve_enum_constructor_from_discriminant(ctor_name)
                    .or_else(|| self.resolve_symbol_in_scope_hierarchy(ctor_name));

                let mut param_types: Vec<Option<TypeId>> = vec![None; params.len()];
                if let Some(sym_id) = ctor_sym {
                    if let Some(sym) = self.context.symbol_table.get_symbol(sym_id) {
                        let ctor_type_id = sym.type_id;
                        if ctor_type_id != TypeId::invalid() {
                            let type_table = self.context.type_table.borrow();
                            if let Some(ty) = type_table.get(ctor_type_id) {
                                if let crate::tast::core::TypeKind::Function {
                                    params: tparams,
                                    ..
                                } = &ty.kind
                                {
                                    for (i, &tid) in tparams.iter().enumerate() {
                                        if i < param_types.len() {
                                            param_types[i] = Some(tid);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                let mut bindings = Vec::new();
                for (i, param) in params.iter().enumerate() {
                    let param_ty = param_types.get(i).copied().unwrap_or(None);
                    bindings.extend(self.bind_pattern_variables_typed(param, param_ty)?);
                }
                Ok(bindings)
            }
            Pattern::Array(patterns) => {
                // Bind variables in each array element pattern
                let mut bindings = Vec::new();
                for pattern in patterns {
                    bindings.extend(self.bind_pattern_variables(pattern)?);
                }
                Ok(bindings)
            }
            Pattern::ArrayRest { elements, rest, .. } => {
                // Bind variables in element patterns
                let mut bindings = Vec::new();
                for pattern in elements {
                    bindings.extend(self.bind_pattern_variables(pattern)?);
                }

                // Bind the rest variable if present
                if let Some(rest_var) = rest {
                    let interned_name = self.context.intern_string(rest_var);
                    let var_symbol = self
                        .context
                        .symbol_table
                        .create_variable_in_scope(interned_name, self.context.current_scope);
                    self.context
                        .scope_tree
                        .get_scope_mut(self.context.current_scope)
                        .expect("Current scope should exist")
                        .add_symbol(var_symbol, interned_name);
                    bindings.push((interned_name, var_symbol));
                }
                Ok(bindings)
            }
            Pattern::Object { fields, .. } => {
                // Bind variables in each field pattern
                let mut bindings = Vec::new();
                for (_, field_pattern) in fields {
                    bindings.extend(self.bind_pattern_variables(field_pattern)?);
                }
                Ok(bindings)
            }
            Pattern::Type { var, .. } => {
                // Bind the typed variable
                let interned_name = self.context.intern_string(var);
                let var_symbol = self
                    .context
                    .symbol_table
                    .create_variable_in_scope(interned_name, self.context.current_scope);
                self.context
                    .scope_tree
                    .get_scope_mut(self.context.current_scope)
                    .expect("Current scope should exist")
                    .add_symbol(var_symbol, interned_name);
                Ok(vec![(interned_name, var_symbol)])
            }
            Pattern::Or(patterns) => {
                // All alternatives bind the same variables, but their payload
                // types can differ: a cross-module-imported enum can leave an
                // early variant's payload Dynamic while later alternatives are
                // concrete (e.g. `U8(x:Dynamic)|I8(x:Int)|..|I64(x:Int)`).
                // Binding from the FIRST alternative alone types `x` Dynamic,
                // which forces the switch-as-expression result temp to `*void`
                // and a spurious `haxe_unbox_int_ptr` of the raw payload at
                // return → SIGSEGV. Prefer the first alternative whose payloads
                // all resolve to concrete types so `x` gets a value type; fall
                // back to the first alternative when none are concrete.
                let mut chosen_idx = 0;
                for (i, p) in patterns.iter().enumerate() {
                    if self.constructor_pattern_payloads_concrete(p) {
                        chosen_idx = i;
                        break;
                    }
                }
                let mut bindings = Vec::new();
                if let Some(p) = patterns.get(chosen_idx) {
                    bindings = self.bind_pattern_variables(p)?;
                }
                // TODO: Validate that all branches bind the same variables
                Ok(bindings)
            }
            Pattern::Const(_) | Pattern::Null | Pattern::Underscore => {
                // These patterns don't bind variables
                Ok(vec![])
            }
            Pattern::Extractor { .. } => {
                // Extractors are complex - for now skip binding
                // TODO: Implement extractor pattern variable binding
                Ok(vec![])
            }
        }
    }

    /// Pick which pattern to bind an or-pattern's variables from. Scans the
    /// case's patterns (expanding an `Or` into its alternatives) and returns
    /// the first alternative whose constructor payloads are all concrete,
    /// falling back to `default` when none qualify. Keeps a bound variable from
    /// being typed `Dynamic` off a leading cross-module variant.
    pub(crate) fn pick_concrete_binding_pattern<'p>(
        &mut self,
        case: &'p parser::Case,
        default: &'p parser::Pattern,
    ) -> &'p parser::Pattern {
        let mut candidates: Vec<&'p parser::Pattern> = Vec::new();
        for p in &case.patterns {
            match p {
                parser::Pattern::Or(alts) => candidates.extend(alts.iter()),
                other => candidates.push(other),
            }
        }
        for c in candidates {
            if self.constructor_pattern_payloads_concrete(c) {
                return c;
            }
        }
        default
    }

    /// True if `p` is a constructor pattern whose every destructured payload
    /// resolves to a concrete (non-Dynamic/Unknown/invalid) type. Used to pick
    /// a well-typed alternative when binding an or-pattern's variables, so a
    /// cross-module enum whose leading variant lost its payload type to Dynamic
    /// doesn't poison the bound variable's type (see `Pattern::Or` above).
    fn constructor_pattern_payloads_concrete(&mut self, p: &parser::Pattern) -> bool {
        let (path, params) = match p {
            parser::Pattern::Constructor { path, params } if !params.is_empty() => (path, params),
            _ => return false,
        };
        let ctor_name = self.context.intern_string(&path.name);
        let ctor_sym = match self
            .resolve_enum_constructor_from_discriminant(ctor_name)
            .or_else(|| self.resolve_symbol_in_scope_hierarchy(ctor_name))
        {
            Some(s) => s,
            None => return false,
        };
        let ctor_type_id = match self.context.symbol_table.get_symbol(ctor_sym) {
            Some(sym) if sym.type_id != TypeId::invalid() => sym.type_id,
            _ => return false,
        };
        let type_table = self.context.type_table.borrow();
        let tparams = match type_table.get(ctor_type_id).map(|t| &t.kind) {
            Some(crate::tast::core::TypeKind::Function { params, .. }) => params.clone(),
            _ => return false,
        };
        if tparams.len() < params.len() {
            return false;
        }
        (0..params.len()).all(|i| {
            let tid = tparams[i];
            tid != TypeId::invalid()
                && !matches!(
                    type_table.get(tid).map(|t| &t.kind),
                    None | Some(crate::tast::core::TypeKind::Dynamic)
                        | Some(crate::tast::core::TypeKind::Unknown)
                )
        })
    }

    /// Create a constructor expression for pattern matching
    fn create_constructor_expression(
        &mut self,
        pattern: &parser::Pattern,
    ) -> Result<TypedExpression, LoweringError> {
        self.create_constructor_expression_with_bindings(pattern, vec![])
    }

    /// Create a constructor expression with pre-resolved variable bindings
    pub(crate) fn create_constructor_expression_with_bindings(
        &mut self,
        pattern: &parser::Pattern,
        variable_bindings: Vec<(InternedString, SymbolId)>,
    ) -> Result<TypedExpression, LoweringError> {
        use parser::Pattern;
        match pattern {
            Pattern::Constructor { path, params } => {
                // Resolve the constructor symbol
                let constructor_name = self.context.intern_string(&path.name);

                // First try to resolve from switch discriminant type (for enum pattern matching)
                // Then fall back to scope hierarchy lookup
                let constructor_symbol = self
                    .resolve_enum_constructor_from_discriminant(constructor_name)
                    .or_else(|| self.resolve_symbol_in_scope_hierarchy(constructor_name))
                    .ok_or_else(|| LoweringError::UnresolvedSymbol {
                        name: path.name.clone(),
                        location: SourceLocation::new(0, 0, 0, 0),
                    })?;

                if params.is_empty() {
                    // Simple constructor like Red, Green, Blue
                    let constructor_type = if let Some(symbol) =
                        self.context.symbol_table.get_symbol(constructor_symbol)
                    {
                        symbol.type_id
                    } else {
                        self.context.type_table.borrow().dynamic_type()
                    };

                    Ok(TypedExpression {
                        kind: TypedExpressionKind::Variable {
                            symbol_id: constructor_symbol,
                        },
                        expr_type: constructor_type,
                        usage: VariableUsage::Borrow,
                        lifetime_id: LifetimeId::from_raw(1),
                        source_location: SourceLocation::new(0, 0, 0, 0),
                        metadata: ExpressionMetadata::default(),
                    })
                } else {
                    // Constructor with parameters - use pattern placeholder for complex patterns
                    self.create_pattern_placeholder_with_bindings(pattern, variable_bindings)
                }
            }
            Pattern::ArrayRest { .. } | Pattern::Object { .. } | Pattern::Type { .. } => {
                // Complex patterns that need later compilation
                self.create_pattern_placeholder_with_bindings(pattern, variable_bindings)
            }
            Pattern::Or(_) => {
                // Or patterns (`case A(id) | B(id):`) must preserve all
                // alternatives so MIR lowering can emit a pattern test
                // that ORs each alternative's discriminant check. Falling
                // through to `lower_pattern_to_expression` here would
                // collapse to the first alternative only — the
                // exhaustiveness checker would then report missing
                // variants and (more importantly) the runtime test would
                // miss every other alternative.
                self.create_pattern_placeholder_with_bindings(pattern, variable_bindings)
            }
            Pattern::Var(name) => {
                // Check if this identifier is actually an enum variant (e.g., "None")
                // The parser produces Var for bare identifiers like `case None:`
                let interned_name = self.context.intern_string(name);
                if let Some(variant_sym_id) =
                    self.resolve_enum_constructor_from_discriminant(interned_name)
                {
                    let ct = self
                        .context
                        .symbol_table
                        .get_symbol(variant_sym_id)
                        .map(|s| s.type_id)
                        .unwrap_or_else(|| self.context.type_table.borrow().dynamic_type());
                    Ok(TypedExpression {
                        kind: TypedExpressionKind::Variable {
                            symbol_id: variant_sym_id,
                        },
                        expr_type: ct,
                        usage: VariableUsage::Borrow,
                        lifetime_id: LifetimeId::from_raw(1),
                        source_location: SourceLocation::new(0, 0, 0, 0),
                        metadata: ExpressionMetadata::default(),
                    })
                } else {
                    // Variable binding pattern (e.g., `case v if v > 0:`)
                    // Use PatternPlaceholder to preserve variable bindings for HIR lowering
                    self.create_pattern_placeholder_with_bindings(pattern, variable_bindings)
                }
            }
            _ => {
                // For simple patterns, fall back to regular pattern conversion
                self.lower_pattern_to_expression(pattern)
            }
        }
    }

    /// Create a pattern placeholder for complex patterns that need later compilation
    fn create_pattern_placeholder(
        &mut self,
        pattern: &parser::Pattern,
    ) -> Result<TypedExpression, LoweringError> {
        self.create_pattern_placeholder_with_bindings(pattern, vec![])
    }

    /// Create a pattern placeholder with pre-resolved variable bindings
    fn create_pattern_placeholder_with_bindings(
        &mut self,
        pattern: &parser::Pattern,
        variable_bindings: Vec<(InternedString, SymbolId)>,
    ) -> Result<TypedExpression, LoweringError> {
        let source_location = SourceLocation::new(0, 0, 0, 0); // TODO: get actual pattern location
        Ok(TypedExpression {
            kind: TypedExpressionKind::PatternPlaceholder {
                pattern: pattern.clone(),
                source_location,
                variable_bindings,
            },
            expr_type: self.context.type_table.borrow().dynamic_type(),
            usage: VariableUsage::Borrow,
            lifetime_id: LifetimeId::from_raw(1),
            source_location,
            metadata: ExpressionMetadata::default(),
        })
    }

    /// Convert a pattern to an expression for case values
    pub(crate) fn lower_pattern_to_expression(
        &mut self,
        pattern: &parser::Pattern,
    ) -> Result<TypedExpression, LoweringError> {
        use parser::Pattern;

        match pattern {
            Pattern::Const(expr) => {
                // Convert constant expression directly
                self.lower_expression(expr)
            }
            Pattern::Var(name) => {
                // Check if this identifier is actually an enum variant (e.g., "None")
                let interned_name = self.context.intern_string(name);
                if let Some(variant_sym_id) =
                    self.resolve_enum_constructor_from_discriminant(interned_name)
                {
                    let ct = self
                        .context
                        .symbol_table
                        .get_symbol(variant_sym_id)
                        .map(|s| s.type_id)
                        .unwrap_or_else(|| self.context.type_table.borrow().dynamic_type());
                    return Ok(TypedExpression {
                        kind: TypedExpressionKind::Variable {
                            symbol_id: variant_sym_id,
                        },
                        expr_type: ct,
                        usage: VariableUsage::Borrow,
                        lifetime_id: LifetimeId::from_raw(1),
                        source_location: SourceLocation::new(0, 0, 0, 0),
                        metadata: ExpressionMetadata::default(),
                    });
                }

                // Variable patterns bind a new variable in the case body
                let var_symbol = self.context.symbol_table.create_variable(interned_name);

                // Register in current scope for the case body
                let current_scope = self.context.current_scope;
                if let Some(scope) = self.context.scope_tree.get_scope_mut(current_scope) {
                    scope.add_symbol(var_symbol, interned_name);
                }

                // Return a wildcard pattern expression
                Ok(TypedExpression {
                    kind: TypedExpressionKind::Null, // Placeholder for wildcard
                    expr_type: self.context.type_table.borrow().dynamic_type(),
                    usage: VariableUsage::Borrow,
                    lifetime_id: LifetimeId::first(),
                    source_location: self.context.create_location(),
                    metadata: ExpressionMetadata::default(),
                })
            }
            Pattern::Constructor { path, params } => {
                // Resolve the constructor symbol
                let constructor_name = self.context.intern_string(&path.name);

                // First try to resolve from switch discriminant type (for enum pattern matching)
                // Then fall back to scope hierarchy lookup
                let constructor_symbol = self
                    .resolve_enum_constructor_from_discriminant(constructor_name)
                    .or_else(|| self.resolve_symbol_in_scope_hierarchy(constructor_name))
                    .ok_or_else(|| LoweringError::UnresolvedSymbol {
                        name: path.name.clone(),
                        location: SourceLocation::new(0, 0, 0, 0),
                    })?;

                if params.is_empty() {
                    // Simple constructor like Red, Green, Blue
                    let constructor_var = TypedExpressionKind::Variable {
                        symbol_id: constructor_symbol,
                    };

                    // Get the constructor's type
                    let constructor_type = if let Some(symbol) =
                        self.context.symbol_table.get_symbol(constructor_symbol)
                    {
                        symbol.type_id
                    } else {
                        self.context.type_table.borrow().dynamic_type()
                    };

                    Ok(TypedExpression {
                        kind: constructor_var,
                        expr_type: constructor_type,
                        usage: VariableUsage::Borrow,
                        lifetime_id: LifetimeId::from_raw(1),
                        source_location: SourceLocation::new(0, 0, 0, 0),
                        metadata: ExpressionMetadata::default(),
                    })
                } else {
                    // Constructor with parameters like RGB(255, 0, 0)
                    let mut arg_exprs = Vec::new();
                    for param_pattern in params {
                        let arg_expr = self.lower_pattern_to_expression(param_pattern)?;
                        arg_exprs.push(arg_expr);
                    }

                    // Get the constructor's type
                    let constructor_type = if let Some(symbol) =
                        self.context.symbol_table.get_symbol(constructor_symbol)
                    {
                        symbol.type_id
                    } else {
                        self.context.type_table.borrow().dynamic_type()
                    };

                    // Create the constructor variable expression
                    let mut constructor_expr = TypedExpression {
                        kind: TypedExpressionKind::Variable {
                            symbol_id: constructor_symbol,
                        },
                        expr_type: constructor_type,
                        usage: VariableUsage::Borrow,
                        lifetime_id: LifetimeId::from_raw(1),
                        source_location: SourceLocation::new(0, 0, 0, 0),
                        metadata: ExpressionMetadata::default(),
                    };

                    // Check if this is a generic enum constructor and instantiate its type
                    if let Some(symbol) = self.context.symbol_table.get_symbol(constructor_symbol) {
                        if symbol.kind == crate::tast::symbols::SymbolKind::EnumVariant {
                            constructor_expr = self.instantiate_enum_constructor_type(
                                constructor_symbol,
                                &arg_exprs,
                                constructor_expr,
                            )?;
                        }
                    }

                    Ok(TypedExpression {
                        kind: TypedExpressionKind::FunctionCall {
                            function: Box::new(constructor_expr),
                            arguments: arg_exprs,
                            type_arguments: Vec::new(),
                        },
                        expr_type: self.context.type_table.borrow().dynamic_type(), // Will be updated by type inference
                        usage: VariableUsage::Borrow,
                        lifetime_id: LifetimeId::from_raw(1),
                        source_location: SourceLocation::new(0, 0, 0, 0),
                        metadata: ExpressionMetadata::default(),
                    })
                }
            }
            Pattern::Array(patterns) => {
                // Array patterns like [1, 2, 3]
                let mut elements = Vec::new();
                for pattern in patterns {
                    elements.push(self.lower_pattern_to_expression(pattern)?);
                }

                Ok(TypedExpression {
                    kind: TypedExpressionKind::ArrayLiteral { elements },
                    expr_type: self.context.type_table.borrow().dynamic_type(),
                    usage: VariableUsage::Borrow,
                    lifetime_id: LifetimeId::from_raw(1),
                    source_location: SourceLocation::new(0, 0, 0, 0),
                    metadata: ExpressionMetadata::default(),
                })
            }
            Pattern::Null => {
                // Null pattern
                Ok(TypedExpression {
                    kind: TypedExpressionKind::Null,
                    expr_type: self.context.type_table.borrow().dynamic_type(),
                    usage: VariableUsage::Borrow,
                    lifetime_id: LifetimeId::from_raw(1),
                    source_location: SourceLocation::new(0, 0, 0, 0),
                    metadata: ExpressionMetadata::default(),
                })
            }
            Pattern::Underscore => {
                // Wildcard pattern. tast_to_hir uses TypedExpressionKind::Null as
                // its wildcard sentinel (Pattern::Var follows the same path), so
                // emit Null here rather than Bool(true) — using Bool(true) caused
                // tast_to_hir to treat the wildcard as a `case true:` literal,
                // which then lowered to `cmp eq scrutinee, true` and left the
                // wildcard body unreachable for non-bool scrutinees.
                Ok(TypedExpression {
                    kind: TypedExpressionKind::Null,
                    expr_type: self.context.type_table.borrow().dynamic_type(),
                    usage: VariableUsage::Borrow,
                    lifetime_id: LifetimeId::from_raw(1),
                    source_location: SourceLocation::new(0, 0, 0, 0),
                    metadata: ExpressionMetadata::default(),
                })
            }
            Pattern::Or(patterns) => {
                // Or patterns like 1 | 2 | 3
                // For now, just use the first pattern
                // TODO: Proper OR pattern handling requires different switch compilation
                if let Some(first) = patterns.first() {
                    self.lower_pattern_to_expression(first)
                } else {
                    Err(LoweringError::IncompleteImplementation {
                        feature: "Empty OR pattern".to_string(),
                        location: SourceLocation::new(0, 0, 0, 0),
                    })
                }
            }
            Pattern::Object { fields } => {
                // Object pattern: {x: 42, y: "hello"}
                // Convert to object literal expression
                let mut typed_fields = Vec::new();

                for (field_name, field_pattern) in fields {
                    // Recursively convert the field pattern to expression
                    let field_expr = self.lower_pattern_to_expression(field_pattern)?;
                    let interned_name = self.context.intern_string(field_name);

                    typed_fields.push(TypedObjectField {
                        name: interned_name,
                        value: field_expr,
                        source_location: SourceLocation::new(0, 0, 0, 0),
                    });
                }

                let field_types: Vec<(InternedString, TypeId)> = typed_fields
                    .iter()
                    .map(|f| (f.name, f.value.expr_type))
                    .collect();

                let kind = TypedExpressionKind::ObjectLiteral {
                    fields: typed_fields,
                };

                Ok(TypedExpression {
                    kind,
                    expr_type: {
                        // Extract field types for type inference

                        type_resolution::infer_object_literal_type(
                            &self.context.type_table,
                            &field_types,
                        )
                    },
                    usage: VariableUsage::Borrow,
                    lifetime_id: LifetimeId::from_raw(1),
                    source_location: SourceLocation::new(0, 0, 0, 0),
                    metadata: ExpressionMetadata::default(),
                })
            }

            Pattern::ArrayRest { elements, rest } => {
                // Array rest pattern: [first, ...rest]
                // Convert elements to expressions
                let mut typed_elements = Vec::new();

                for element_pattern in elements {
                    let element_expr = self.lower_pattern_to_expression(element_pattern)?;
                    typed_elements.push(element_expr);
                }

                // Handle the rest variable if present
                if let Some(rest_name) = rest {
                    // Create a variable expression for the rest binding
                    let rest_interned = self.context.intern_string(rest_name);

                    // Look up or create symbol for rest variable
                    let rest_symbol = if let Some(symbol_id) =
                        self.resolve_symbol_in_scope_hierarchy(rest_interned)
                    {
                        symbol_id
                    } else {
                        // Create new variable symbol in current scope
                        let symbol_id = self
                            .context
                            .symbol_table
                            .create_variable_in_scope(rest_interned, self.context.current_scope);

                        self.context
                            .scope_tree
                            .get_scope_mut(self.context.current_scope)
                            .expect("Current scope should exist")
                            .add_symbol(symbol_id, rest_interned);

                        symbol_id
                    };

                    let rest_expr = TypedExpression {
                        kind: TypedExpressionKind::Variable {
                            symbol_id: rest_symbol,
                        },
                        expr_type: self
                            .context
                            .type_table
                            .borrow_mut()
                            .create_array_type(self.context.type_table.borrow().dynamic_type()), // Array type with dynamic elements
                        usage: VariableUsage::Borrow,
                        lifetime_id: LifetimeId::from_raw(1),
                        source_location: SourceLocation::new(0, 0, 0, 0),
                        metadata: ExpressionMetadata::default(),
                    };

                    // Add rest to the elements
                    typed_elements.push(rest_expr);
                }

                let kind = TypedExpressionKind::ArrayLiteral {
                    elements: typed_elements,
                };

                Ok(TypedExpression {
                    kind,
                    expr_type: self
                        .context
                        .type_table
                        .borrow_mut()
                        .create_array_type(self.context.type_table.borrow().dynamic_type()), // Array type with dynamic elements
                    usage: VariableUsage::Borrow,
                    lifetime_id: LifetimeId::from_raw(1),
                    source_location: SourceLocation::new(0, 0, 0, 0),
                    metadata: ExpressionMetadata::default(),
                })
            }

            Pattern::Type { var, type_hint } => {
                // Type pattern: (s:String)
                // Create a variable expression with type constraint
                let var_interned = self.context.intern_string(var);

                // Look up or create symbol for the typed variable
                let var_symbol =
                    if let Some(symbol_id) = self.resolve_symbol_in_scope_hierarchy(var_interned) {
                        symbol_id
                    } else {
                        // Create new variable symbol in current scope
                        let symbol_id = self
                            .context
                            .symbol_table
                            .create_variable_in_scope(var_interned, self.context.current_scope);

                        self.context
                            .scope_tree
                            .get_scope_mut(self.context.current_scope)
                            .expect("Current scope should exist")
                            .add_symbol(symbol_id, var_interned);

                        symbol_id
                    };

                // Resolve the type hint to get the proper type
                let resolved_type = self.lower_type(type_hint)?;

                let kind = TypedExpressionKind::Variable {
                    symbol_id: var_symbol,
                };

                Ok(TypedExpression {
                    kind,
                    expr_type: resolved_type, // Use the type constraint from the pattern
                    usage: VariableUsage::Borrow,
                    lifetime_id: LifetimeId::from_raw(1),
                    source_location: SourceLocation::new(0, 0, 0, 0),
                    metadata: ExpressionMetadata::default(),
                })
            }

            Pattern::Extractor { .. } => {
                // Extractor patterns require runtime evaluation - not implemented
                Err(LoweringError::IncompleteImplementation {
                    feature: format!("Extractor pattern to expression conversion: {:?}", pattern),
                    location: SourceLocation::new(0, 0, 0, 0),
                })
            }
        }
    }

    /// Try to desugar a tuple literal to a static method call (e.g., SIMD4f.make()).
    /// Returns Ok(Some(expr)) if desugared, Ok(None) if the target type doesn't support tuple construction.
    pub(crate) fn try_desugar_tuple_to_make(
        &mut self,
        elements: &[parser::Expr],
        target_ty: crate::tast::TypeId,
        original_expr: &parser::Expr,
    ) -> LoweringResult<Option<TypedExpression>> {
        use crate::tast::core::TypeKind;

        // Check if target type is an abstract or class with a known native name
        let (class_symbol_id, native_name) = {
            let type_table = self.context.type_table.borrow();
            if let Some(type_info) = type_table.get(target_ty) {
                let sym_id = match &type_info.kind {
                    TypeKind::Abstract { symbol_id, .. } => Some(*symbol_id),
                    TypeKind::Class { symbol_id, .. } => Some(*symbol_id),
                    _ => None,
                };
                if let Some(sid) = sym_id {
                    let nn = self.context.symbol_table.get_symbol(sid).and_then(|s| {
                        s.native_name
                            .and_then(|nn| self.context.string_interner.get(nn))
                            .map(|s| s.to_string())
                    });
                    (Some(sid), nn)
                } else {
                    (None, None)
                }
            } else {
                (None, None)
            }
        };

        // Currently only SIMD4f supports tuple construction
        // Check by native_name or symbol name
        let is_simd4f = match (class_symbol_id, native_name.as_deref()) {
            (Some(_), Some("rayzor::SIMD4f")) => true,
            (Some(sid), _) => {
                // Fallback: check symbol name directly
                self.context
                    .symbol_table
                    .get_symbol(sid)
                    .and_then(|s| self.context.string_interner.get(s.name))
                    .map(|n| n == "SIMD4f")
                    .unwrap_or(false)
            }
            _ => false,
        };
        let class_symbol_id = match (is_simd4f, class_symbol_id) {
            (true, Some(sid)) => sid,
            _ => return Ok(None),
        };

        // Validate element count
        if elements.len() != 4 {
            return Err(LoweringError::InternalError {
                message: format!(
                    "SIMD4f tuple literal requires exactly 4 elements, got {}",
                    elements.len()
                ),
                location: self.context.span_to_location(&original_expr.span),
            });
        }

        // AST-level rewriting: construct a synthetic `SIMD4f.make(e1, e2, e3, e4)` expression
        // and lower it through the normal path, which handles static method resolution.
        let span = original_expr.span;
        let simd_ident = parser::Expr {
            kind: parser::ExprKind::Ident("SIMD4f".to_string()),
            span,
        };
        let field_access = parser::Expr {
            kind: parser::ExprKind::Field {
                expr: Box::new(simd_ident),
                field: "make".to_string(),
                is_optional: false,
            },
            span,
        };
        let args: Vec<parser::Expr> = elements.to_vec();
        let make_call = parser::Expr {
            kind: parser::ExprKind::Call {
                expr: Box::new(field_access),
                args,
            },
            span,
        };

        let lowered = self.lower_expression(&make_call)?;
        Ok(Some(lowered))
    }
}
