//! Enum declarations, variants and constructors.

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
    /// Infer the type of an enum constructor call with generic type instantiation
    fn infer_enum_constructor_type(
        &mut self,
        constructor_symbol: SymbolId,
        arguments: &[TypedExpression],
    ) -> LoweringResult<TypeId> {
        // println!(
        //     "DEBUG: Inferring enum constructor type for symbol {:?} with {} arguments",
        //     constructor_symbol,
        //     arguments.len()
        // );

        // Find the parent enum for this constructor
        let parent_enum = self
            .find_parent_enum_for_constructor(constructor_symbol)
            .ok_or_else(|| LoweringError::InternalError {
                message: "Could not find parent enum for constructor".to_string(),
                location: self.context.create_location(),
            })?;

        // println!(
        //     "DEBUG: Found parent enum {:?} for constructor {:?}",
        //     parent_enum, constructor_symbol
        // );

        // Get the parent enum's type information
        if let Some(enum_symbol) = self.context.symbol_table.get_symbol(parent_enum) {
            let enum_type_info = self
                .context
                .type_table
                .borrow()
                .get(enum_symbol.type_id)
                .ok_or_else(|| LoweringError::InternalError {
                    message: "Could not get type info for enum".to_string(),
                    location: self.context.create_location(),
                })?
                .clone();

            match &enum_type_info.kind {
                crate::tast::core::TypeKind::Enum { type_args, .. } => {
                    if type_args.is_empty() {
                        // Non-generic enum, just return the enum type
                        // println!(
                        //     "DEBUG: Non-generic enum, returning enum type {:?}",
                        //     enum_symbol.type_id
                        // );
                        return Ok(enum_symbol.type_id);
                    }

                    // Generic enum - need to infer type arguments from constructor arguments
                    // println!(
                    //     "DEBUG: Generic enum with {} type parameters",
                    //     type_args.len()
                    // );

                    // Infer type parameters from constructor arguments
                    let mut inferred_types = Vec::new();

                    // Match argument types to constructor parameter types
                    for (i, arg) in arguments.iter().enumerate() {
                        if i < type_args.len() {
                            inferred_types.push(arg.expr_type);
                        }
                    }

                    // Fill remaining type parameters with dynamic type
                    while inferred_types.len() < type_args.len() {
                        inferred_types.push(self.context.type_table.borrow().dynamic_type());
                    }

                    // Create properly instantiated enum type
                    if !inferred_types.is_empty() {
                        let instantiated_type = self
                            .context
                            .type_table
                            .borrow_mut()
                            .create_enum_type(parent_enum, inferred_types);
                        return Ok(instantiated_type);
                    }

                    // Non-generic enum
                    Ok(self
                        .context
                        .type_table
                        .borrow_mut()
                        .create_enum_type(parent_enum, vec![]))
                }
                _ => {
                    // Not an enum type
                    Ok(self.context.type_table.borrow().dynamic_type())
                }
            }
        } else {
            Ok(self.context.type_table.borrow().dynamic_type())
        }
    }

    /// Instantiate the function type of an enum constructor based on call arguments
    pub(crate) fn instantiate_enum_constructor_type(
        &mut self,
        constructor_symbol: SymbolId,
        arguments: &[TypedExpression],
        mut func_expr: TypedExpression,
    ) -> LoweringResult<TypedExpression> {
        // println!(
        //     "DEBUG: Instantiating constructor function type for symbol {:?} with {} arguments",
        //     constructor_symbol,
        //     arguments.len()
        // );

        // Find the parent enum for this constructor
        let parent_enum = self
            .find_parent_enum_for_constructor(constructor_symbol)
            .ok_or_else(|| LoweringError::InternalError {
                message: "Could not find parent enum for constructor".to_string(),
                location: self.context.create_location(),
            })?;

        // println!(
        //     "DEBUG: Found parent enum {:?} for constructor {:?}",
        //     parent_enum, constructor_symbol
        // );

        // Get the parent enum's type information
        if let Some(enum_symbol) = self.context.symbol_table.get_symbol(parent_enum) {
            let enum_type_info = self
                .context
                .type_table
                .borrow()
                .get(enum_symbol.type_id)
                .ok_or_else(|| LoweringError::InternalError {
                    message: "Could not get type info for enum".to_string(),
                    location: self.context.create_location(),
                })?
                .clone();

            match &enum_type_info.kind {
                crate::tast::core::TypeKind::Enum { type_args, .. } => {
                    if type_args.is_empty() {
                        // Non-generic enum - ensure function type has enum as return type
                        // This is critical for type inference of parameterized enum constructors
                        let constructor_sym =
                            self.context.symbol_table.get_symbol(constructor_symbol);
                        if let Some(sym) = constructor_sym {
                            // Extract params first, then drop the borrow
                            let params_opt = {
                                let type_table = self.context.type_table.borrow();
                                if let Some(func_type_info) = type_table.get(sym.type_id) {
                                    if let crate::tast::core::TypeKind::Function {
                                        params, ..
                                    } = &func_type_info.kind
                                    {
                                        Some(params.clone())
                                    } else {
                                        None
                                    }
                                } else {
                                    None
                                }
                            };
                            // Now create the function type with the enum as return type
                            if let Some(params) = params_opt {
                                let corrected_func_type = self
                                    .context
                                    .type_table
                                    .borrow_mut()
                                    .create_function_type(params, enum_symbol.type_id);
                                func_expr.expr_type = corrected_func_type;
                            }
                        }
                        return Ok(func_expr);
                    }

                    // Generic enum - need to infer type arguments from constructor arguments
                    // println!(
                    //     "DEBUG: Generic enum with {} type parameters",
                    //     type_args.len()
                    // );

                    // Infer type arguments from constructor arguments
                    if !arguments.is_empty() && !type_args.is_empty() {
                        // Get the original constructor's function type params
                        let original_params = {
                            if let Some(sym) =
                                self.context.symbol_table.get_symbol(constructor_symbol)
                            {
                                let type_table = self.context.type_table.borrow();
                                if let Some(ty) = type_table.get(sym.type_id) {
                                    if let crate::tast::core::TypeKind::Function {
                                        params, ..
                                    } = &ty.kind
                                    {
                                        params.clone()
                                    } else {
                                        vec![]
                                    }
                                } else {
                                    vec![]
                                }
                            } else {
                                vec![]
                            }
                        };

                        // Infer the concrete type parameter T from the first argument.
                        // Two cases:
                        //   1. Param type is T directly (e.g., Leaf(value:T)) → arg type IS T
                        //   2. Param type is Enum<T> (e.g., Node(left:Tree<T>)) → extract T from arg's type args
                        let inferred_type = {
                            let arg_type = arguments[0].expr_type;
                            let first_param_type = original_params.first().copied();
                            let type_table = self.context.type_table.borrow();

                            let mut result = arg_type; // default: assume arg IS the type param

                            if let Some(param_tid) = first_param_type {
                                if let Some(param_ty) = type_table.get(param_tid) {
                                    match &param_ty.kind {
                                        // Param is the enum type itself (e.g., Tree<T>)
                                        // Extract the type args from the argument's enum type
                                        crate::tast::core::TypeKind::Enum {
                                            symbol_id: enum_sym,
                                            ..
                                        } if *enum_sym == parent_enum => {
                                            if let Some(arg_ty) = type_table.get(arg_type) {
                                                if let crate::tast::core::TypeKind::Enum {
                                                    type_args: arg_ta,
                                                    ..
                                                } = &arg_ty.kind
                                                {
                                                    if let Some(&first_ta) = arg_ta.first() {
                                                        result = first_ta;
                                                    }
                                                }
                                            }
                                        }
                                        // Param is a type parameter directly → arg type IS T
                                        crate::tast::core::TypeKind::TypeParameter { .. } => {
                                            result = arg_type;
                                        }
                                        _ => {}
                                    }
                                }
                            }
                            result
                        };

                        // Create instantiated enum type with inferred type args
                        let instantiated_enum_type = self
                            .context
                            .type_table
                            .borrow_mut()
                            .create_enum_type(parent_enum, vec![inferred_type]);

                        // Substitute each original param type with its instantiated version
                        let instantiated_params: Vec<TypeId> = original_params
                            .iter()
                            .map(|&param_type| {
                                let type_table = self.context.type_table.borrow();
                                if let Some(ty) = type_table.get(param_type) {
                                    match &ty.kind {
                                        crate::tast::core::TypeKind::TypeParameter { .. } => {
                                            inferred_type
                                        }
                                        crate::tast::core::TypeKind::Enum { symbol_id, .. }
                                            if *symbol_id == parent_enum =>
                                        {
                                            instantiated_enum_type
                                        }
                                        _ => param_type,
                                    }
                                } else {
                                    param_type
                                }
                            })
                            .collect();

                        // Create instantiated function type with correct param count and types
                        let instantiated_function_type = self
                            .context
                            .type_table
                            .borrow_mut()
                            .create_function_type(instantiated_params, instantiated_enum_type);

                        func_expr.expr_type = instantiated_function_type;
                        return Ok(func_expr);
                    }

                    // Fallback - couldn't infer
                    // println!("DEBUG: Could not infer type arguments, using original function type");
                    Ok(func_expr)
                }
                _ => {
                    // Not an enum type
                    Ok(func_expr)
                }
            }
        } else {
            Ok(func_expr)
        }
    }

    /// Bring an imported enum's constructors into scope under their bare names.
    ///
    /// `import haxe.ds.Option;` makes `Some`/`None` usable unqualified. A
    /// packaged enum restored from the BLADE manifest publishes its
    /// constructors under qualified names, so the bare names only exist once
    /// something imports the enum — which is exactly Haxe's own rule.
    pub(crate) fn import_enum_constructors(&mut self, enum_symbol: SymbolId) {
        if self
            .context
            .symbol_table
            .get_symbol(enum_symbol)
            .map(|s| s.kind)
            != Some(crate::tast::symbols::SymbolKind::Enum)
        {
            return;
        }
        let Some(variants) = self
            .context
            .symbol_table
            .get_enum_variants(enum_symbol)
            .cloned()
        else {
            return;
        };
        for variant in variants {
            let Some(name) = self
                .context
                .symbol_table
                .get_symbol(variant)
                .map(|s| s.name)
            else {
                continue;
            };
            // A name already bound in the root scope belongs to whoever claimed
            // it; an import adds constructors, it never displaces them.
            self.context
                .symbol_table
                .add_symbol_alias(variant, ScopeId::first(), name);
            // The bare name now reaches it, so it counts as a candidate again
            // wherever the compiler weighs same-named constructors.
            self.context
                .symbol_table
                .clear_symbol_flags(variant, crate::tast::symbols::SymbolFlags::QUALIFIED_ONLY);
            let root = self
                .context
                .scope_tree
                .get_scope_mut(ScopeId::first())
                .expect("Root scope should exist");
            if !root.has_symbol(name) {
                root.add_symbol(variant, name);
            }
        }
    }

    /// Lower an enum declaration
    /// Public wrapper for lower_enum_declaration, used when loading from BLADE cache
    pub fn lower_enum_declaration_public(
        &mut self,
        enum_decl: &EnumDecl,
    ) -> LoweringResult<TypedDeclaration> {
        self.lower_enum_declaration(enum_decl)
    }

    pub(crate) fn lower_enum_declaration(
        &mut self,
        enum_decl: &EnumDecl,
    ) -> LoweringResult<TypedDeclaration> {
        let enum_name = self.context.intern_string(&enum_decl.name);

        // Look up existing symbol from pre-registration, or create a new one
        let enum_symbol = if let Some(existing_symbol) = self
            .context
            .symbol_table
            .lookup_symbol(ScopeId::first(), enum_name)
        {
            existing_symbol.id
        } else {
            let new_symbol = self
                .context
                .symbol_table
                .create_enum_in_scope(enum_name, ScopeId::first());
            self.context.update_symbol_qualified_name(new_symbol);
            self.context
                .scope_tree
                .get_scope_mut(ScopeId::first())
                .expect("Root scope should exist")
                .add_symbol(new_symbol, enum_name);
            new_symbol
        };

        // Enter enum scope with name
        let enum_scope = self.context.enter_named_scope(ScopeKind::Enum, enum_name);

        // Process type parameters
        let type_params = self.lower_type_parameters(&enum_decl.type_params)?;
        let mut type_param_map: BTreeMap<InternedString, TypeId> = BTreeMap::new();
        let mut type_param_ids = Vec::new();
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
            type_param_ids.push(type_id);
        }
        self.context.push_type_parameters(type_param_map);

        // Create the enum type
        let enum_type_id = self
            .context
            .type_table
            .borrow_mut()
            .create_enum_type(enum_symbol, type_param_ids);

        // Update the enum symbol with its type
        self.context
            .symbol_table
            .update_symbol_type(enum_symbol, enum_type_id);

        // Process variants
        let mut variants = Vec::with_capacity(enum_decl.constructors.len());
        for variant in &enum_decl.constructors {
            variants.push(self.lower_enum_variant(variant, enum_type_id, enum_symbol)?);
        }

        self.context.pop_type_parameters();
        self.context.exit_scope();

        let typed_enum = TypedEnum {
            symbol_id: enum_symbol,
            name: enum_name,
            variants,
            type_parameters: type_params,
            visibility: self.lower_access(&enum_decl.access),
            source_location: self.context.create_location_from_span(enum_decl.span),
        };

        Ok(TypedDeclaration::Enum(typed_enum))
    }

    /// Lower an enum variant
    fn lower_enum_variant(
        &mut self,
        variant: &EnumConstructor,
        enum_type_id: TypeId,
        enum_symbol: SymbolId,
    ) -> LoweringResult<TypedEnumVariant> {
        let variant_name = self.context.intern_string(&variant.name);
        // Reuse pre-registered variant symbol if it exists AND belongs to the same
        // parent enum. Without this check, name collisions (e.g., Result.Error vs
        // haxe.io.Error enum type) cause the wrong symbol to be reused.
        let variant_symbol = if let Some(existing) = self
            .context
            .symbol_table
            .lookup_symbol(ScopeId::first(), variant_name)
        {
            if existing.kind == crate::tast::symbols::SymbolKind::EnumVariant {
                // Verify it belongs to the same parent enum
                let is_same_parent = self
                    .context
                    .symbol_table
                    .find_parent_enum_for_constructor(existing.id)
                    .map(|p| p == enum_symbol)
                    .unwrap_or(false);
                if is_same_parent {
                    existing.id
                } else {
                    self.context.symbol_table.create_enum_variant_in_scope(
                        variant_name,
                        ScopeId::first(),
                        enum_symbol,
                    )
                }
            } else {
                // Name collision with non-variant symbol (e.g., enum type) — create fresh variant
                self.context.symbol_table.create_enum_variant_in_scope(
                    variant_name,
                    ScopeId::first(),
                    enum_symbol,
                )
            }
        } else {
            self.context.symbol_table.create_enum_variant_in_scope(
                variant_name,
                ScopeId::first(),
                enum_symbol,
            )
        };

        // Process parameters first to get their types
        let mut parameters = Vec::new();
        let mut param_types = Vec::new();
        for param in &variant.params {
            let typed_param = self.lower_parameter(param)?;
            param_types.push(typed_param.param_type);
            parameters.push(typed_param);
        }

        // For enum constructors, we store the generic constructor type
        // The actual type will be instantiated when the constructor is used
        let constructor_type = if param_types.is_empty() {
            // No parameters: constructor will return the enum type directly
            enum_type_id
        } else {
            // Has parameters: create a function type that preserves generics
            // This will be a generic function if the enum is generic
            self.context
                .type_table
                .borrow_mut()
                .create_function_type(param_types, enum_type_id)
        };

        // Update the symbol with the proper type
        self.context
            .symbol_table
            .update_symbol_type(variant_symbol, constructor_type);

        Ok(TypedEnumVariant {
            name: variant_name,
            parameters,
            source_location: self.context.create_location(),
        })
    }
}
