//! TAST to HIR Lowering
//!
//! This module converts the Typed AST (TAST) to High-level IR (HIR).
//! HIR preserves most source-level constructs while adding:
//! - Resolved symbols and types
//! - Lifetime and ownership information
//! - Desugared syntax (e.g., for-in to iterators)

use tracing::{debug, warn};

use crate::ir::hir::*;
use crate::semantic_graph::SemanticGraphs;
use crate::stdlib::{MethodSignature, StdlibMapping};
use crate::tast::{
    node::*, InternedString, LifetimeId, ScopeId, SourceLocation, StringInterner, SymbolId,
    SymbolTable, TypeId, TypeKind, TypeTable, Visibility,
};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

/// Context for lowering TAST to HIR
pub struct TastToHirContext<'a> {
    /// Symbol table from TAST
    symbol_table: &'a SymbolTable,

    /// Type table from TAST
    type_table: &'a Rc<RefCell<TypeTable>>,

    /// String interner from TAST
    string_interner: &'a mut StringInterner,

    /// Semantic graphs for additional information
    semantic_graphs: Option<&'a SemanticGraphs>,

    /// Current module being built
    module: HirModule,

    /// Current scope
    current_scope: ScopeId,

    /// Current lifetime
    current_lifetime: LifetimeId,

    /// Loop labels for break/continue
    loop_labels: Vec<Option<SymbolId>>,

    /// Error accumulator
    errors: Vec<LoweringError>,

    /// Counter for generating unique temporary variable names
    temp_var_counter: u32,

    /// Current file being processed (for validation)
    current_file: Option<&'a TypedFile>,

    /// Standard library runtime function mapping
    stdlib_mapping: StdlibMapping,

    /// Inline variable values (for static inline vars that need constant evaluation)
    /// Maps symbol ID to the evaluated literal value (preserving type)
    inline_var_values: HashMap<SymbolId, HirLiteral>,
}

#[derive(Debug)]
pub struct LoweringError {
    pub message: String,
    pub location: SourceLocation,
}

/// Result type for lowering operations that can recover from errors
pub enum LoweringResult<T> {
    /// Successful lowering
    Ok(T),
    /// Complete failure - cannot produce any result
    Error(LoweringError),
    /// Partial success - produced a result but with errors
    Partial(T, Vec<LoweringError>),
}

impl<T> LoweringResult<T> {
    /// Convert to Result, treating partial success as success
    pub fn to_result(self) -> Result<T, Vec<LoweringError>> {
        match self {
            LoweringResult::Ok(value) => Ok(value),
            LoweringResult::Partial(value, _) => Ok(value),
            LoweringResult::Error(err) => Err(vec![err]),
        }
    }

    /// Check if this is a successful result (Ok or Partial)
    pub fn is_successful(&self) -> bool {
        !matches!(self, LoweringResult::Error(_))
    }

    /// Extract any errors from the result
    pub fn errors(&self) -> Vec<LoweringError> {
        match self {
            LoweringResult::Ok(_) => vec![],
            LoweringResult::Error(err) => vec![err.clone()],
            LoweringResult::Partial(_, errors) => errors.clone(),
        }
    }
}

impl Clone for LoweringError {
    fn clone(&self) -> Self {
        Self {
            message: self.message.clone(),
            location: self.location,
        }
    }
}

impl<'a> TastToHirContext<'a> {
    /// Create a new lowering context
    pub fn new(
        symbol_table: &'a SymbolTable,
        type_table: &'a Rc<RefCell<TypeTable>>,
        string_interner: &'a mut StringInterner,
        module_name: String,
    ) -> Self {
        Self {
            symbol_table,
            type_table,
            string_interner,
            semantic_graphs: None,
            module: HirModule {
                name: module_name,
                imports: Vec::new(),
                types: indexmap::IndexMap::new(), // Use IndexMap for deterministic ordering
                functions: indexmap::IndexMap::new(),
                globals: HashMap::new(),
                metadata: HirMetadata {
                    source_file: String::new(),
                    language_version: "1.0".to_string(),
                    target_platforms: vec!["js".to_string()],
                    optimization_hints: Vec::new(),
                },
            },
            current_scope: ScopeId::from_raw(0),
            current_lifetime: LifetimeId::from_raw(0),
            loop_labels: Vec::new(),
            errors: Vec::new(),
            temp_var_counter: 0,
            current_file: None,
            stdlib_mapping: StdlibMapping::new(),
            inline_var_values: HashMap::new(),
        }
    }

    /// Set semantic graphs for additional analysis info
    pub fn set_semantic_graphs(&mut self, graphs: &'a SemanticGraphs) {
        self.semantic_graphs = Some(graphs);
    }

    /// Helper methods to get builtin types from type table
    fn get_void_type(&self) -> TypeId {
        self.type_table.borrow().void_type()
    }

    fn get_bool_type(&self) -> TypeId {
        self.type_table.borrow().bool_type()
    }

    fn get_string_type(&self) -> TypeId {
        self.type_table.borrow().string_type()
    }

    fn get_null_type(&self) -> TypeId {
        // Use dynamic type for null for now
        self.type_table.borrow().dynamic_type()
    }

    fn get_dynamic_type(&self) -> TypeId {
        self.type_table.borrow().dynamic_type()
    }

    /// Pre-process static variables to evaluate their constant values
    /// This handles cases like: static inline var SOLAR_MASS = 4.0 * PI * PI;
    /// where PI is also a static inline var
    fn evaluate_inline_static_vars(&mut self, file: &TypedFile) {
        // Collect all static fields that need evaluation (from classes and abstracts)
        let static_fields: Vec<_> = file
            .classes
            .iter()
            .flat_map(|class| class.fields.iter())
            .chain(file.abstracts.iter().flat_map(|abs| abs.fields.iter()))
            .filter(|field| field.is_static && field.initializer.is_some())
            .map(|field| (field.symbol_id, field.initializer.as_ref().unwrap().clone()))
            .collect();

        // Keep evaluating until no new values are discovered (fixpoint)
        loop {
            let prev_count = self.inline_var_values.len();

            for (symbol_id, init_expr) in &static_fields {
                if !self.inline_var_values.contains_key(symbol_id) {
                    if let Some(literal) = self.try_evaluate_const_expr(&init_expr) {
                        self.inline_var_values.insert(*symbol_id, literal);
                    }
                }
            }

            // Stop when no new values were discovered
            if self.inline_var_values.len() == prev_count {
                break;
            }
        }
    }

    /// Try to evaluate a constant expression to a literal value
    /// Returns Some(HirLiteral) if the expression can be fully evaluated at compile time
    /// Preserves the type (Int vs Float) of the original expression
    fn try_evaluate_const_expr(&self, expr: &TypedExpression) -> Option<HirLiteral> {
        use crate::tast::node::{BinaryOperator, UnaryOperator};

        match &expr.kind {
            TypedExpressionKind::Literal { value } => {
                // Only handle numeric literals for constant evaluation
                match value {
                    LiteralValue::Int(i) => Some(HirLiteral::Int(*i)),
                    LiteralValue::Float(f) => Some(HirLiteral::Float(*f)),
                    LiteralValue::Bool(b) => Some(HirLiteral::Bool(*b)),
                    _ => None, // Strings and other literals not supported in const eval
                }
            }
            TypedExpressionKind::Variable { symbol_id, .. } => {
                // Check if this is an already-evaluated inline variable
                self.inline_var_values.get(symbol_id).cloned()
            }
            TypedExpressionKind::BinaryOp {
                operator,
                left,
                right,
            } => {
                let lhs = self.try_evaluate_const_expr(left)?;
                let rhs = self.try_evaluate_const_expr(right)?;

                // Evaluate based on types - prefer float if either operand is float
                match (&lhs, &rhs) {
                    (HirLiteral::Float(l), HirLiteral::Float(r)) => match operator {
                        BinaryOperator::Add => Some(HirLiteral::Float(l + r)),
                        BinaryOperator::Sub => Some(HirLiteral::Float(l - r)),
                        BinaryOperator::Mul => Some(HirLiteral::Float(l * r)),
                        BinaryOperator::Div => Some(HirLiteral::Float(l / r)),
                        _ => None,
                    },
                    (HirLiteral::Float(l), HirLiteral::Int(r)) => {
                        let r = *r as f64;
                        match operator {
                            BinaryOperator::Add => Some(HirLiteral::Float(l + r)),
                            BinaryOperator::Sub => Some(HirLiteral::Float(l - r)),
                            BinaryOperator::Mul => Some(HirLiteral::Float(l * r)),
                            BinaryOperator::Div => Some(HirLiteral::Float(l / r)),
                            _ => None,
                        }
                    }
                    (HirLiteral::Int(l), HirLiteral::Float(r)) => {
                        let l = *l as f64;
                        match operator {
                            BinaryOperator::Add => Some(HirLiteral::Float(l + r)),
                            BinaryOperator::Sub => Some(HirLiteral::Float(l - r)),
                            BinaryOperator::Mul => Some(HirLiteral::Float(l * r)),
                            BinaryOperator::Div => Some(HirLiteral::Float(l / r)),
                            _ => None,
                        }
                    }
                    (HirLiteral::Int(l), HirLiteral::Int(r)) => match operator {
                        BinaryOperator::Add => Some(HirLiteral::Int(l + r)),
                        BinaryOperator::Sub => Some(HirLiteral::Int(l - r)),
                        BinaryOperator::Mul => Some(HirLiteral::Int(l * r)),
                        BinaryOperator::Div => Some(HirLiteral::Int(l / r)),
                        BinaryOperator::Mod => Some(HirLiteral::Int(l % r)),
                        // Bitwise operations
                        BinaryOperator::Shl => Some(HirLiteral::Int(l << r)),
                        BinaryOperator::Shr => Some(HirLiteral::Int(l >> r)),
                        BinaryOperator::BitAnd => Some(HirLiteral::Int(l & r)),
                        BinaryOperator::BitOr => Some(HirLiteral::Int(l | r)),
                        BinaryOperator::BitXor => Some(HirLiteral::Int(l ^ r)),
                        _ => None,
                    },
                    _ => None,
                }
            }
            TypedExpressionKind::UnaryOp { operator, operand } => {
                let val = self.try_evaluate_const_expr(operand)?;
                match operator {
                    UnaryOperator::Neg => match val {
                        HirLiteral::Float(f) => Some(HirLiteral::Float(-f)),
                        HirLiteral::Int(i) => Some(HirLiteral::Int(-i)),
                        _ => None,
                    },
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Check if class is final
    fn is_class_final(&self, class_symbol: SymbolId) -> bool {
        if let Some(hierarchy) = self.symbol_table.get_class_hierarchy(class_symbol) {
            hierarchy.is_final
        } else {
            false
        }
    }

    /// Check if class is abstract
    fn is_class_abstract(&self, class_symbol: SymbolId) -> bool {
        if let Some(hierarchy) = self.symbol_table.get_class_hierarchy(class_symbol) {
            hierarchy.is_abstract
        } else {
            false
        }
    }

    /// Lookup enum type from a constructor symbol
    fn lookup_enum_type_from_constructor(&self, constructor: SymbolId) -> TypeId {
        // Look up constructor symbol to find its parent enum type
        if let Some(symbol_info) = self.symbol_table.get_symbol(constructor) {
            // If this is an enum variant, its type should be the enum itself
            if symbol_info.type_id != TypeId::invalid() {
                return symbol_info.type_id;
            }
            // Try to find the parent enum through the symbol's scope
            // This would require more context about the symbol hierarchy
            // For now, return the symbol's type if available
        }
        // Fallback to dynamic type
        self.get_dynamic_type()
    }

    /// Lower a typed file to HIR module
    pub fn lower_file(&mut self, file: &'a TypedFile) -> Result<HirModule, Vec<LoweringError>> {
        // Set current file for validation
        self.current_file = Some(file);

        // Lower imports
        for import in &file.imports {
            self.lower_import(import);
        }

        // Pre-process: Evaluate inline static variables
        // This must happen before lowering classes so that references to inline vars
        // (like SOLAR_MASS = 4.0 * PI * PI) can be resolved
        self.evaluate_inline_static_vars(file);

        // Lower type declarations (classes, interfaces, enums, etc.)
        for class in &file.classes {
            self.lower_class(class);
        }

        for interface in &file.interfaces {
            self.lower_interface(interface);
        }

        for enum_decl in &file.enums {
            self.lower_enum(enum_decl);
        }

        // Lower abstract types
        for abstract_decl in &file.abstracts {
            self.lower_abstract(abstract_decl);
        }

        // Lower type aliases
        for alias in &file.type_aliases {
            self.lower_type_alias(alias);
        }

        // Lower module-level functions
        for function in &file.functions {
            let hir_func = self.lower_function(function);
            self.module.functions.insert(function.symbol_id, hir_func);
        }

        // Lower module fields (global variables)
        for field in &file.module_fields {
            self.lower_module_field(field);
        }

        if self.errors.is_empty() {
            Ok(self.module.clone())
        } else {
            Err(std::mem::take(&mut self.errors))
        }
    }

    /// Lower a class declaration
    fn lower_class(&mut self, class: &TypedClass) {
        let mut hir_fields = Vec::new();
        let mut hir_methods = Vec::new();
        let mut hir_constructor = None;

        // Process constructors (take only the first)
        if let Some(constructor) = class.constructors.first() {
            hir_constructor = Some(self.lower_constructor(constructor));
        }

        // Process methods
        for method in &class.methods {
            hir_methods.push(HirMethod {
                function: self.lower_function(method),
                visibility: self.convert_visibility(method.visibility),
                is_static: method.is_static,
                is_override: method.metadata.is_override,
                is_abstract: false, // Abstract methods would have no body
            });
        }

        // Process fields
        for field in &class.fields {
            hir_fields.push(HirClassField {
                symbol_id: field.symbol_id, // Preserve symbol_id for field access lowering
                name: field.name.clone(),
                ty: field.field_type,
                init: field.initializer.as_ref().map(|e| self.lower_expression(e)),
                visibility: self.convert_visibility(field.visibility),
                is_static: field.is_static,
                is_final: !matches!(field.mutability, crate::tast::Mutability::Mutable),
                property_access: field.property_access.clone(), // Preserve property accessor info
            });
        }

        // Create type ID from symbol ID (simplified)
        let type_id = TypeId::from_raw(class.symbol_id.as_raw());

        let hir_class = HirClass {
            symbol_id: class.symbol_id,
            name: class.name.clone(),
            type_params: self.lower_type_params(&class.type_parameters),
            extends: class.super_class,
            implements: class.interfaces.clone(),
            fields: hir_fields,
            methods: hir_methods,
            constructor: hir_constructor,
            metadata: Vec::new(), // Convert from TypedMetadata when available
            // Look up class hierarchy info from SymbolTable
            is_final: self.is_class_final(class.symbol_id),
            is_abstract: self.is_class_abstract(class.symbol_id),
            is_extern: false, // Would be in effects or metadata when available
        };

        self.module
            .types
            .insert(type_id, HirTypeDecl::Class(hir_class));
    }

    /// Lower an interface declaration
    fn lower_interface(&mut self, interface: &TypedInterface) {
        // Extract interface methods from method signatures
        let hir_methods: Vec<HirInterfaceMethod> = interface
            .methods
            .iter()
            .map(|method| {
                HirInterfaceMethod {
                    name: method.name.clone(),
                    type_params: Vec::new(), // Type params are on the method signature in TAST
                    params: method
                        .parameters
                        .iter()
                        .map(|p| self.lower_param(p))
                        .collect(),
                    return_type: method.return_type,
                }
            })
            .collect();

        // Interfaces don't have fields in Haxe, only properties (which are methods)
        let hir_fields: Vec<HirInterfaceField> = Vec::new();

        // Create type ID from symbol ID (simplified)
        let type_id = TypeId::from_raw(interface.symbol_id.as_raw());

        let hir_interface = HirInterface {
            symbol_id: interface.symbol_id,
            name: interface.name.clone(),
            type_params: self.lower_type_params(&interface.type_parameters),
            extends: interface.extends.clone(),
            fields: hir_fields,
            methods: hir_methods,
            metadata: Vec::new(), // Interfaces can have metadata
        };

        self.module
            .types
            .insert(type_id, HirTypeDecl::Interface(hir_interface));
    }

    /// Lower an enum declaration
    fn lower_enum(&mut self, enum_decl: &TypedEnum) {
        let mut hir_variants = Vec::new();

        for (i, variant) in enum_decl.variants.iter().enumerate() {
            // Extract variant fields from parameters
            // In Haxe, enum variants can have parameters like: Value(x: Int, y: String)
            let hir_fields: Vec<HirEnumField> = variant
                .parameters
                .iter()
                .map(|param| HirEnumField {
                    name: param.name.clone(),
                    ty: param.param_type,
                })
                .collect();

            hir_variants.push(HirEnumVariant {
                name: variant.name.clone(),
                fields: hir_fields,
                discriminant: Some(i as i32),
            });
        }

        // Create type ID from symbol ID (simplified)
        let type_id = TypeId::from_raw(enum_decl.symbol_id.as_raw());

        let hir_enum = HirEnum {
            symbol_id: enum_decl.symbol_id,
            name: enum_decl.name.clone(),
            type_params: self.lower_type_params(&enum_decl.type_parameters),
            variants: hir_variants,
            metadata: Vec::new(),
        };

        self.module
            .types
            .insert(type_id, HirTypeDecl::Enum(hir_enum));
    }

    /// Lower an abstract type
    fn lower_abstract(&mut self, abstract_decl: &TypedAbstract) {
        // Create type ID from symbol ID (simplified)
        let type_id = TypeId::from_raw(abstract_decl.symbol_id.as_raw());

        // Extract implicit conversion rules
        // @:from rules allow implicit conversion FROM other types TO this abstract
        let mut from_rules: Vec<HirCastRule> = abstract_decl
            .from_types
            .iter()
            .map(|&from_ty| {
                HirCastRule {
                    from_type: from_ty,
                    to_type: type_id,
                    is_implicit: true,
                    cast_function: None, // Keyword from — no function needed
                }
            })
            .collect();

        // @:to rules allow implicit conversion FROM this abstract TO other types
        let mut to_rules: Vec<HirCastRule> = abstract_decl
            .to_types
            .iter()
            .map(|&to_ty| {
                HirCastRule {
                    from_type: type_id,
                    to_type: to_ty,
                    is_implicit: true,
                    cast_function: None, // Keyword to — no function needed
                }
            })
            .collect();

        // Scan methods for @:from/@:to metadata and associate conversion functions
        for method in &abstract_decl.methods {
            if method.metadata.is_from_conversion
                && method.is_static
                && !method.parameters.is_empty()
            {
                // @:from static function: first param type is the source type
                let param_type = method.parameters[0].param_type;
                // Check if there's already a keyword rule for this source type
                let existing = from_rules.iter_mut().find(|r| r.from_type == param_type);
                if let Some(rule) = existing {
                    rule.cast_function = Some(method.symbol_id);
                } else {
                    // @:from method defines a new conversion rule (no keyword clause needed)
                    from_rules.push(HirCastRule {
                        from_type: param_type,
                        to_type: type_id,
                        is_implicit: true,
                        cast_function: Some(method.symbol_id),
                    });
                }
            }
            if method.metadata.is_to_conversion && !method.is_static {
                // @:to instance function: return type is the target type
                let target_type = method.return_type;
                let existing = to_rules.iter_mut().find(|r| r.to_type == target_type);
                if let Some(rule) = existing {
                    rule.cast_function = Some(method.symbol_id);
                } else {
                    to_rules.push(HirCastRule {
                        from_type: type_id,
                        to_type: target_type,
                        is_implicit: true,
                        cast_function: Some(method.symbol_id),
                    });
                }
            }
        }

        // Extract operator overloads from methods
        // Methods with @:op metadata become operators
        let operators: Vec<HirOperatorOverload> = abstract_decl
            .methods
            .iter()
            .filter_map(|_method| {
                // Check if method has @:op metadata
                // For now, return None as operators are extracted via metadata
                // which we'll handle when macro infrastructure is ready
                None
            })
            .collect();

        // Extract abstract fields
        let fields: Vec<HirAbstractField> = abstract_decl
            .fields
            .iter()
            .map(|field| {
                HirAbstractField {
                    name: field.name.clone(),
                    ty: field.field_type,
                    getter: None, // Will be resolved during type checking
                    setter: None, // Will be resolved during type checking
                }
            })
            .collect();

        // Lower methods (same pattern as classes)
        let hir_methods: Vec<HirMethod> = abstract_decl
            .methods
            .iter()
            .map(|method| HirMethod {
                function: self.lower_function(method),
                visibility: self.convert_visibility(method.visibility),
                is_static: method.is_static,
                is_override: false,
                is_abstract: false,
            })
            .collect();

        // Abstract constructors use value-wrap at the call site (new MyInt(42) → 42).
        // Don't lower the constructor body since `this = value` isn't a standard assignment.
        let hir_constructor = None;

        let hir_abstract = HirAbstract {
            symbol_id: abstract_decl.symbol_id,
            name: abstract_decl.name.clone(),
            type_params: self.lower_type_params(&abstract_decl.type_parameters),
            underlying: abstract_decl
                .underlying_type
                .unwrap_or_else(|| self.get_dynamic_type()),
            from_rules,
            to_rules,
            operators,
            fields,
            methods: hir_methods,
            constructor: hir_constructor,
            metadata: Vec::new(),
        };

        self.module
            .types
            .insert(type_id, HirTypeDecl::Abstract(hir_abstract));
    }

    /// Lower a type alias
    fn lower_type_alias(&mut self, alias: &TypedTypeAlias) {
        // Create type ID from symbol ID (simplified)
        let type_id = TypeId::from_raw(alias.symbol_id.as_raw());

        let hir_alias = HirTypeAlias {
            symbol_id: alias.symbol_id,
            name: alias.name.clone(),
            type_params: self.lower_type_params(&alias.type_parameters),
            aliased_type: alias.target_type,
        };

        self.module
            .types
            .insert(type_id, HirTypeDecl::TypeAlias(hir_alias));
    }

    /// Lower a function
    fn lower_function(&mut self, function: &TypedFunction) -> HirFunction {
        // debug!(" TAST→HIR: Function {:?} has {} statements in body",
        //           self.string_interner.get(function.name),
        //           function.body.len());

        let hir_body = if !function.body.is_empty() {
            Some(self.lower_block(&function.body))
        } else {
            None
        };

        // Check if this is the main function
        let main_name = self.string_interner.intern("main");
        let is_main = function.name == main_name;

        // Extract optimization hints from SemanticGraphs/DFG if available
        let mut metadata = self.extract_function_metadata(&function.metadata);
        if let Some(semantic_graphs) = self.semantic_graphs {
            metadata.extend(self.extract_ssa_optimization_hints(function, semantic_graphs));
        }

        // Get qualified name from symbol table
        let qualified_name = self
            .symbol_table
            .get_symbol(function.symbol_id)
            .and_then(|sym| sym.qualified_name);

        HirFunction {
            symbol_id: function.symbol_id,
            name: function.name.clone(),
            qualified_name,
            type_params: self.lower_type_params(&function.type_parameters),
            params: function
                .parameters
                .iter()
                .map(|p| self.lower_param(p))
                .collect(),
            return_type: function.return_type,
            body: hir_body,
            metadata,
            is_inline: function.effects.is_inline,
            is_macro: false,  // TODO: Extract from function metadata
            is_extern: false, // TODO: Extract from function metadata
            calling_convention: HirCallingConvention::Haxe,
            is_main,
        }
    }

    /// Extract optimization hints from SSA analysis in SemanticGraphs
    /// This queries the DFG without rebuilding SSA - following the architectural principle
    fn extract_ssa_optimization_hints(
        &self,
        function: &TypedFunction,
        semantic_graphs: &SemanticGraphs,
    ) -> Vec<HirAttribute> {
        let mut hints = Vec::new();

        // Query DFG for this specific function using its symbol_id
        if let Some(dfg) = semantic_graphs.data_flow.get(&function.symbol_id) {
            // Check if function is in valid SSA form
            if dfg.is_valid_ssa() {
                // Extract optimization hints from SSA analysis

                // 1. Variable usage patterns from SSA
                let total_ssa_vars = dfg.ssa_variables.len();
                if total_ssa_vars < 10 {
                    let hint_name = self.string_interner.intern("optimization_hint");
                    let hint_value = self.string_interner.intern("few_locals");
                    hints.push(HirAttribute {
                        name: hint_name,
                        args: vec![HirAttributeArg::Literal(HirLiteral::String(hint_value))],
                    });
                }

                // 2. Dead code detection from SSA
                let dead_nodes: usize = dfg
                    .nodes
                    .values()
                    .filter(|n| n.uses.is_empty() && !n.metadata.has_side_effects)
                    .count();
                if dead_nodes > 0 {
                    let dead_code_name = self.string_interner.intern("dead_code_count");
                    hints.push(HirAttribute {
                        name: dead_code_name,
                        args: vec![HirAttributeArg::Literal(HirLiteral::Int(dead_nodes as i64))],
                    });
                }

                // 3. Phi node complexity (indicates control flow complexity)
                let phi_count = dfg.metadata.phi_node_count;
                if phi_count == 0 {
                    let hint_name = self.string_interner.intern("optimization_hint");
                    let hint_value = self.string_interner.intern("straight_line_code");
                    hints.push(HirAttribute {
                        name: hint_name,
                        args: vec![HirAttributeArg::Literal(HirLiteral::String(hint_value))],
                    });
                } else if phi_count > 20 {
                    let hint_name = self.string_interner.intern("optimization_hint");
                    let hint_value = self.string_interner.intern("complex_control_flow");
                    hints.push(HirAttribute {
                        name: hint_name,
                        args: vec![HirAttributeArg::Literal(HirLiteral::String(hint_value))],
                    });
                }

                // 4. Value numbering opportunities
                if dfg.value_numbering.expr_to_value.len() > 5 {
                    let hint_name = self.string_interner.intern("optimization_hint");
                    let hint_value = self.string_interner.intern("common_subexpressions");
                    hints.push(HirAttribute {
                        name: hint_name,
                        args: vec![HirAttributeArg::Literal(HirLiteral::String(hint_value))],
                    });
                }

                // 5. Inlining hints based on function size and SSA complexity
                let node_count = dfg.nodes.len();
                if node_count < 10 && phi_count < 3 {
                    let inline_name = self.string_interner.intern("inline_candidate");
                    hints.push(HirAttribute {
                        name: inline_name,
                        args: vec![HirAttributeArg::Literal(HirLiteral::Bool(true))],
                    });
                }
            }
        }

        // Query CFG for control flow patterns
        if let Some(cfg) = semantic_graphs.control_flow.get(&function.symbol_id) {
            let block_count = cfg.blocks.len();
            if block_count == 1 {
                let hint_name = self.string_interner.intern("optimization_hint");
                let hint_value = self.string_interner.intern("single_block");
                hints.push(HirAttribute {
                    name: hint_name,
                    args: vec![HirAttributeArg::Literal(HirLiteral::String(hint_value))],
                });
            }
        }

        hints
    }

    /// Lower a constructor
    fn lower_constructor(&mut self, method: &TypedFunction) -> HirConstructor {
        let mut body = self.lower_block(&method.body);

        // Extract super() call from constructor body if present.
        // In Haxe, super(args) must be the first statement in a child constructor.
        // We scan the first statement for Call { callee: Super, args } and extract it
        // into HirConstructor.super_call so the MIR lowering can emit the parent
        // constructor call before the child's field initializations.
        let mut super_call = None;
        // super() may be inside a nested Block (the constructor body gets wrapped).
        // Scan both top-level and one-level-deep block statements.
        let stmts_to_scan: &mut Vec<HirStatement> = &mut body.statements;
        // If the body is a single Block expression, unwrap it
        if stmts_to_scan.len() == 1 {
            if let HirStatement::Expr(expr) = &stmts_to_scan[0] {
                if let HirExprKind::Block(inner_block) = &expr.kind {
                    // Replace body with the inner block's contents
                    let inner_stmts = inner_block.statements.clone();
                    *stmts_to_scan = inner_stmts;
                }
            }
        }
        if let Some(first_stmt) = stmts_to_scan.first() {
            if let HirStatement::Expr(expr) = first_stmt {
                if let HirExprKind::Call { callee, args, .. } = &expr.kind {
                    if matches!(callee.kind, HirExprKind::Super) {
                        super_call = Some(HirSuperCall { args: args.clone() });
                    }
                }
            }
        }
        if super_call.is_some() {
            body.statements.remove(0);
        }

        HirConstructor {
            params: method
                .parameters
                .iter()
                .map(|p| self.lower_param(p))
                .collect(),
            super_call,
            field_inits: Vec::new(),
            body,
        }
    }

    /// Lower a statement
    fn lower_statement(&mut self, stmt: &TypedStatement) -> HirStatement {
        match stmt {
            TypedStatement::VarDeclaration {
                symbol_id,
                var_type,
                initializer,
                mutability,
                ..
            } => {
                use crate::tast::Mutability;
                let is_mutable = matches!(mutability, Mutability::Mutable);

                HirStatement::Let {
                    pattern: HirPattern::Variable {
                        name: self.get_symbol_name(*symbol_id),
                        symbol: *symbol_id,
                    },
                    type_hint: Some(*var_type),
                    init: initializer.as_ref().map(|e| self.lower_expression(e)),
                    is_mutable,
                }
            }
            TypedStatement::Expression { expression, .. } => {
                HirStatement::Expr(self.lower_expression(expression))
            }
            TypedStatement::Return { value, .. } => {
                HirStatement::Return(value.as_ref().map(|e| self.lower_expression(e)))
            }
            TypedStatement::Break { target_loop, .. } => {
                // Use the SymbolId directly instead of converting to string
                HirStatement::Break(*target_loop)
            }
            TypedStatement::Continue { target_loop, .. } => {
                // Use the SymbolId directly
                HirStatement::Continue(*target_loop)
            }
            TypedStatement::Throw { exception, .. } => {
                HirStatement::Throw(self.lower_expression(exception))
            }
            TypedStatement::If {
                condition,
                then_branch,
                else_branch,
                ..
            } => HirStatement::If {
                condition: self.lower_expression(condition),
                then_branch: self.lower_block(std::slice::from_ref(then_branch)),
                else_branch: else_branch
                    .as_ref()
                    .map(|s| self.lower_block(std::slice::from_ref(&**s))),
            },
            TypedStatement::Switch {
                discriminant,
                cases,
                default_case,
                ..
            } => {
                let mut hir_cases = Vec::new();

                for case in cases {
                    // TAST TypedSwitchCase has case_value and body
                    let patterns = vec![self.lower_case_value_pattern(&case.case_value)];

                    let guard = None; // TAST doesn't have guard on switch cases
                    let body = self.lower_statement_as_block(&case.body);

                    hir_cases.push(HirMatchCase {
                        patterns,
                        guard,
                        body,
                    });
                }

                // Add default case if present
                if let Some(default) = default_case {
                    hir_cases.push(HirMatchCase {
                        patterns: vec![HirPattern::Wildcard],
                        guard: None,
                        body: self.lower_block(std::slice::from_ref(&**default)),
                    });
                }

                HirStatement::Switch {
                    scrutinee: self.lower_expression(discriminant),
                    cases: hir_cases,
                }
            }
            TypedStatement::While {
                condition, body, ..
            } => {
                self.loop_labels.push(None);
                let hir_stmt = HirStatement::While {
                    label: None,
                    condition: self.lower_expression(condition),
                    body: self.lower_block(std::slice::from_ref(body)),
                    continue_update: None,
                };
                self.loop_labels.pop();
                hir_stmt
            }
            // Note: DoWhile might not exist in TAST, handle via While
            TypedStatement::For {
                init,
                condition,
                update,
                body,
                ..
            } => {
                // Desugar for loop to while loop with separate continue_update
                let mut statements = Vec::new();

                // Add init statement if present
                if let Some(init) = init {
                    statements.push(self.lower_statement(init));
                }

                // Create while loop with update in continue_update (not body)
                self.loop_labels.push(None);
                let while_body = self.lower_block(std::slice::from_ref(body));
                let continue_update = update.as_ref().map(|upd| {
                    let update_stmt = HirStatement::Expr(self.lower_expression(upd));
                    HirBlock::new(vec![update_stmt], self.current_scope)
                });

                let while_stmt = HirStatement::While {
                    label: None,
                    condition: condition
                        .as_ref()
                        .map(|e| self.lower_expression(e))
                        .unwrap_or_else(|| self.make_bool_literal(true)),
                    body: while_body,
                    continue_update,
                };
                self.loop_labels.pop();

                statements.push(while_stmt);

                // Wrap in a block
                HirStatement::Expr(HirExpr::new(
                    HirExprKind::Block(HirBlock::new(statements, self.current_scope)),
                    self.get_void_type(),
                    self.current_lifetime,
                    stmt.source_location(),
                ))
            }
            TypedStatement::ForIn {
                value_var,
                key_var,
                iterable,
                body,
                ..
            } => {
                debug!(
                    " [tast_to_hir ForIn]: Creating HirStatement::ForIn from TypedStatement::ForIn"
                );
                debug!(
                    " [tast_to_hir ForIn]: value_var={:?}, key_var={:?}",
                    value_var, key_var
                );
                debug!(
                    " [tast_to_hir ForIn]: iterable.expr_type={:?}",
                    iterable.expr_type
                );
                self.loop_labels.push(None);

                // Create pattern from value_var and optional key_var
                let pattern = if let Some(key_var) = key_var {
                    // Key-value iteration: key => value
                    HirPattern::Tuple(vec![
                        HirPattern::Variable {
                            name: self.get_symbol_name(*key_var),
                            symbol: *key_var,
                        },
                        HirPattern::Variable {
                            name: self.get_symbol_name(*value_var),
                            symbol: *value_var,
                        },
                    ])
                } else {
                    // Simple iteration
                    HirPattern::Variable {
                        name: self.get_symbol_name(*value_var),
                        symbol: *value_var,
                    }
                };

                let hir_stmt = HirStatement::ForIn {
                    label: None,
                    pattern,
                    iterator: self.lower_expression(iterable),
                    body: self.lower_block(std::slice::from_ref(body)),
                };
                self.loop_labels.pop();
                hir_stmt
            }
            TypedStatement::Try {
                body,
                catch_clauses,
                finally_block,
                ..
            } => {
                let hir_catches = catch_clauses
                    .iter()
                    .map(|c| HirCatchClause {
                        exception_type: c.exception_type,
                        exception_var: c.exception_variable,
                        body: self.lower_block(std::slice::from_ref(&c.body)),
                    })
                    .collect();

                HirStatement::TryCatch {
                    try_block: self.lower_block(std::slice::from_ref(body)),
                    catches: hir_catches,
                    finally_block: finally_block
                        .as_ref()
                        .map(|s| self.lower_block(std::slice::from_ref(&**s))),
                }
            }
            TypedStatement::Block { statements, .. } => HirStatement::Expr(HirExpr::new(
                HirExprKind::Block(self.lower_block(statements)),
                self.get_void_type(),
                self.current_lifetime,
                stmt.source_location(),
            )),
            TypedStatement::Assignment { target, value, .. } => {
                // ARRAY ACCESS OVERLOADING: Check if target is array access with @:arrayAccess set method
                if let TypedExpressionKind::ArrayAccess { array, index } = &target.kind {
                    if let Some((set_method, _abstract_symbol)) =
                        self.find_array_access_method(array.expr_type, "set")
                    {
                        debug!(
                            ": Found array access set method for type {:?}: method symbol {:?}",
                            array.expr_type, set_method
                        );

                        // Rewrite array assignment to method call:  `a[i] = v` → `a.set(i, v)`
                        // Create arguments: [index, value]
                        let args = vec![(**index).clone(), value.clone()];

                        if let Some(inlined) = self.try_inline_abstract_method(
                            array,
                            set_method,
                            &args,
                            value.expr_type,
                            target.source_location,
                        ) {
                            debug!(": Successfully inlined array access set method!");
                            // Convert to expression statement
                            return HirStatement::Expr(inlined);
                        } else {
                            debug!(": Failed to inline array access set method");
                            // Fall through to normal assignment
                        }
                    }
                }

                // Default: normal assignment
                HirStatement::Assign {
                    lhs: self.lower_lvalue(target),
                    rhs: self.lower_expression(value),
                    op: None,
                }
            }
            TypedStatement::PatternMatch {
                value,
                patterns,
                source_location,
            } => {
                // Desugar pattern matching to a series of if-else statements
                // match value { pattern1 => body1, pattern2 => body2 } becomes:
                // let _match = value;
                // if (pattern1 matches _match) { body1 } else if (pattern2 matches _match) { body2 }

                self.desugar_pattern_match(value, patterns, *source_location)
            }
            _ => {
                // Use error recovery to continue processing
                self.make_error_stmt("Unsupported statement type", stmt.source_location())
            }
        }
    }

    /// Check if this method call should be mapped to a stdlib runtime function
    ///
    /// Returns (class_name, method_name, is_static) if this is a stdlib method
    fn get_stdlib_method_info(
        &self,
        receiver_type: TypeId,
        method_symbol: SymbolId,
    ) -> Option<(&'static str, &'static str, bool)> {
        // Get the method name from the symbol table
        let method_name = if let Some(symbol) = self.symbol_table.get_symbol(method_symbol) {
            self.string_interner.get(symbol.name)?
        } else {
            return None;
        };

        // Get the type name from the receiver type
        let type_table = self.type_table.borrow();
        let type_info = type_table.get(receiver_type)?;

        // Get class name from type - prefer lowered @:native name for namespacing
        let class_name = match &type_info.kind {
            TypeKind::String => "String",
            TypeKind::Array { .. } => "Array",
            TypeKind::Class { symbol_id, .. } => {
                if let Some(class_info) = self.symbol_table.get_symbol(*symbol_id) {
                    // Use lowered native name if available (e.g., "rayzor::concurrent::Arc" -> "rayzor_concurrent_Arc")
                    if let Some(native_interned) = class_info.native_name {
                        if let Some(native_str) = self.string_interner.get(native_interned) {
                            let lowered = native_str.replace("::", "_");
                            Box::leak(lowered.into_boxed_str())
                        } else {
                            self.string_interner.get(class_info.name)?
                        }
                    } else {
                        self.string_interner.get(class_info.name)?
                    }
                } else {
                    return None;
                }
            }
            _ => return None, // Not a stdlib type
        };

        drop(type_table);

        // Check if this is actually a stdlib class using the mapping
        // This replaces the hardcoded list ["Math", "Sys", "String", "Array"]
        if !self.stdlib_mapping.is_stdlib_class(class_name) {
            return None; // Not a stdlib class
        }

        // Determine if methods are static by checking the mapping
        // This replaces hardcoded matches!(class_name, "Math" | "Sys")
        let is_static = self.stdlib_mapping.class_has_static_methods(class_name);

        if self
            .stdlib_mapping
            .has_mapping(class_name, method_name, is_static)
        {
            // Get the class name as a 'static str from the mapping registry
            // This replaces the hardcoded match statement
            let class_static = self.stdlib_mapping.get_class_static_str(class_name)?;

            // For method names, we need to use a leaked string for now
            // In production, we'd maintain a static registry
            let method_static: &'static str = Box::leak(method_name.to_string().into_boxed_str());

            Some((class_static, method_static, is_static))
        } else {
            None
        }
    }

    /// Lower an expression
    fn lower_expression(&mut self, expr: &TypedExpression) -> HirExpr {
        let kind = match &expr.kind {
            TypedExpressionKind::Literal { value } => {
                HirExprKind::Literal(self.lower_literal(value))
            }
            TypedExpressionKind::Variable { symbol_id } => {
                // First check if this is a pre-evaluated inline variable (e.g., SOLAR_MASS = 4.0 * PI * PI)
                if let Some(literal) = self.inline_var_values.get(symbol_id).cloned() {
                    return HirExpr {
                        kind: HirExprKind::Literal(literal),
                        ty: expr.expr_type,
                        lifetime: LifetimeId::from_raw(1), // Static lifetime for constants
                        source_location: expr.source_location,
                    };
                }

                // Check if this variable is actually a static field reference
                // This happens when you access a static field without the class prefix
                // e.g., `SIZE` instead of `Main.SIZE` when inside class Main
                let mut inlined_static = None;

                if let Some(file) = &self.current_file {
                    // Look through all classes to find if this symbol is a static field
                    for class in &file.classes {
                        for field in &class.fields {
                            if field.symbol_id == *symbol_id && field.is_static {
                                // Found a static field - try to inline its constant value
                                if let Some(ref init_expr) = field.initializer {
                                    if let TypedExpressionKind::Literal { value } = &init_expr.kind
                                    {
                                        // Inline the constant value from the static field
                                        let lowered = self.lower_literal(value);
                                        inlined_static = Some(HirExprKind::Literal(lowered));
                                    }
                                }
                                break;
                            }
                        }
                        if inlined_static.is_some() {
                            break;
                        }
                    }
                }

                if let Some(hir_kind) = inlined_static {
                    hir_kind
                } else {
                    HirExprKind::Variable {
                        symbol: *symbol_id,
                        capture_mode: None, // TODO: Determine from context
                    }
                }
            }
            TypedExpressionKind::This { .. } => HirExprKind::This,
            TypedExpressionKind::Super { .. } => HirExprKind::Super,
            TypedExpressionKind::Null => HirExprKind::Null,
            TypedExpressionKind::FieldAccess {
                object,
                field_symbol,
                ..
            } => HirExprKind::Field {
                object: Box::new(self.lower_expression(object)),
                field: *field_symbol,
            },
            TypedExpressionKind::StaticFieldAccess {
                class_symbol,
                field_symbol,
            } => {
                // First check pre-evaluated inline values (handles enum abstract fields and
                // static inline vars that were pre-computed by evaluate_inline_static_vars)
                if let Some(literal) = self.inline_var_values.get(field_symbol).cloned() {
                    return HirExpr {
                        kind: HirExprKind::Literal(literal),
                        ty: expr.expr_type,
                        lifetime: LifetimeId::from_raw(1),
                        source_location: expr.source_location,
                    };
                }

                // Static fields with constant initializers should be inlined
                // For non-constant static fields, we would need global data storage
                let mut inlined_value = None;

                if let Some(file) = &self.current_file {
                    // Find the class or abstract by symbol
                    let field_iter: Vec<&crate::tast::node::TypedField> = file
                        .classes
                        .iter()
                        .filter(|c| c.symbol_id == *class_symbol)
                        .flat_map(|c| c.fields.iter())
                        .chain(
                            file.abstracts
                                .iter()
                                .filter(|a| a.symbol_id == *class_symbol)
                                .flat_map(|a| a.fields.iter()),
                        )
                        .collect();

                    for field in field_iter {
                        if field.symbol_id == *field_symbol && field.is_static {
                            if let Some(ref init_expr) = field.initializer {
                                if let TypedExpressionKind::Literal { value } = &init_expr.kind {
                                    inlined_value =
                                        Some(HirExprKind::Literal(self.lower_literal(value)));
                                }
                            }
                            break;
                        }
                    }
                }

                if let Some(hir_kind) = inlined_value {
                    hir_kind
                } else {
                    // Fallback: treat as variable (may not work for all cases)
                    debug!(
                        "WARNING: Static field access without constant initializer may not work"
                    );
                    HirExprKind::Variable {
                        symbol: *field_symbol,
                        capture_mode: None,
                    }
                }
            }
            TypedExpressionKind::ArrayAccess { array, index } => {
                // ARRAY ACCESS OVERLOADING: Check if array type has @:arrayAccess get method
                if let Some((get_method, _abstract_symbol)) =
                    self.find_array_access_method(array.expr_type, "get")
                {
                    debug!(
                        ": Found array access get method for type {:?}: method symbol {:?}",
                        array.expr_type, get_method
                    );

                    // Rewrite array access to method call:  `a[i]` → `a.get(i)`
                    // Then try to inline it using existing infrastructure
                    if let Some(inlined) = self.try_inline_abstract_method(
                        array,
                        get_method,
                        &[(**index).clone()],
                        expr.expr_type,
                        expr.source_location,
                    ) {
                        debug!(": Successfully inlined array access get method!");
                        return inlined;
                    } else {
                        debug!(": Failed to inline array access get method, falling back to method call");
                        // TODO: Fall back to method call if inlining fails
                    }
                }

                // Default: convert to normal array index operation
                HirExprKind::Index {
                    object: Box::new(self.lower_expression(array)),
                    index: Box::new(self.lower_expression(index)),
                }
            }
            TypedExpressionKind::FunctionCall {
                function,
                type_arguments,
                arguments,
                ..
            } => {
                HirExprKind::Call {
                    callee: Box::new(self.lower_expression(function)),
                    type_args: type_arguments.clone(),
                    args: arguments.iter().map(|a| self.lower_expression(a)).collect(),
                    is_method: false, // TODO: Determine from context
                }
            }
            TypedExpressionKind::MethodCall {
                receiver,
                method_symbol,
                type_arguments,
                arguments,
            } => {
                // Check if this is an abstract type method that should be inlined
                if let Some(inlined) = self.try_inline_abstract_method(
                    receiver,
                    *method_symbol,
                    arguments,
                    expr.expr_type,
                    expr.source_location,
                ) {
                    // println!("DEBUG lower_expression: Method inlined successfully!");
                    return inlined;
                }

                // Check if this is a stdlib method that should be mapped to runtime function
                if let Some((class_name, method_name, _is_static)) =
                    self.get_stdlib_method_info(receiver.expr_type, *method_symbol)
                {
                    // println!("DEBUG: Mapping stdlib method {}.{} to runtime function", class_name, method_name);

                    let sig = MethodSignature {
                        class: class_name,
                        method: method_name,
                        is_static: _is_static,
                        is_constructor: false, // Not checking for constructors in this code path
                        param_count: arguments.len(), // Use actual arg count for lookup
                    };

                    if let Some(_mapping) = self.stdlib_mapping.get(&sig) {
                        // println!("DEBUG: Found runtime mapping for {}.{}", class_name, method_name);
                        // For now, continue with normal lowering
                        // The actual runtime function call will be generated in MIR lowering
                        // based on the method signature
                    }
                }

                // println!("DEBUG lower_expression: Method NOT inlined, lowering normally");

                // Desugar method call to function call with receiver as first argument
                // receiver.method(args) becomes method(receiver, args)
                // This makes the calling convention explicit for later lowering

                let receiver_expr = self.lower_expression(receiver);
                let mut call_args = vec![receiver_expr];
                call_args.extend(arguments.iter().map(|a| self.lower_expression(a)));

                HirExprKind::Call {
                    callee: Box::new(HirExpr::new(
                        HirExprKind::Variable {
                            symbol: *method_symbol,
                            capture_mode: None,
                        },
                        // TODO: Get actual method type from symbol table
                        expr.expr_type,
                        self.current_lifetime,
                        expr.source_location,
                    )),
                    type_args: type_arguments.clone(),
                    args: call_args,
                    is_method: true, // Mark that this was originally a method call
                }
            }
            TypedExpressionKind::New {
                class_type,
                type_arguments,
                arguments,
                class_name: tast_class_name,
            } => {
                // Validate constructor exists and is accessible
                self.validate_constructor(*class_type, arguments.len(), expr.source_location);

                // Use class_name from TAST if available, otherwise extract from TypeId
                // (for cases where TypeId might be invalid, e.g., extern stdlib classes)
                let class_name = tast_class_name.or_else(|| {
                    self.type_table
                        .borrow()
                        .get(*class_type)
                        .and_then(|type_ref| {
                            if let crate::tast::TypeKind::Class { symbol_id, .. } = &type_ref.kind {
                                self.symbol_table.get_symbol(*symbol_id).map(|sym| sym.name)
                            } else {
                                None
                            }
                        })
                });

                HirExprKind::New {
                    class_type: *class_type,
                    type_args: type_arguments.clone(),
                    args: arguments.iter().map(|a| self.lower_expression(a)).collect(),
                    class_name,
                }
            }
            TypedExpressionKind::UnaryOp { operator, operand } => {
                // OPERATOR OVERLOADING: Check if operand has abstract type with @:op metadata
                if let Some((method_symbol, _abstract_symbol)) =
                    self.find_unary_operator_method(operand.expr_type, operator)
                {
                    // debug!(": Found unary operator method for {:?} on type {:?}: method symbol {:?}",
                    //           operator, operand.expr_type, method_symbol);

                    // Rewrite unary operation to method call:  `-a` → `a.negate()`
                    // Then try to inline it using existing infrastructure
                    if let Some(inlined) = self.try_inline_abstract_method(
                        operand,
                        method_symbol,
                        &[], // No arguments for unary operators
                        expr.expr_type,
                        expr.source_location,
                    ) {
                        // debug!(": Successfully inlined unary operator method!");
                        return inlined;
                    } else {
                        // debug!(": Failed to inline unary operator method, falling back to method call");
                        // TODO: Fall back to method call if inlining fails
                    }
                }

                // Default: convert to normal unary operation
                HirExprKind::Unary {
                    op: self.convert_unary_op(operator),
                    operand: Box::new(self.lower_expression(operand)),
                }
            }
            TypedExpressionKind::BinaryOp {
                left,
                operator,
                right,
            } => {
                // ARRAY ACCESS OVERLOADING: Check if this is an assignment to array access with @:arrayAccess set method
                if *operator == BinaryOperator::Assign {
                    if let TypedExpressionKind::ArrayAccess { array, index } = &left.kind {
                        if let Some((set_method, _abstract_symbol)) =
                            self.find_array_access_method(array.expr_type, "set")
                        {
                            // debug!(": Found array access set method in BinaryOp for type {:?}: method symbol {:?}",
                            //           array.expr_type, set_method);

                            // Rewrite array assignment to method call:  `a[i] = v` → `a.set(i, v)`
                            // Create arguments: [index, value]
                            let args = vec![(**index).clone(), (**right).clone()];

                            if let Some(inlined) = self.try_inline_abstract_method(
                                array,
                                set_method,
                                &args,
                                right.expr_type, // Return value is typically the assigned value
                                expr.source_location,
                            ) {
                                // debug!(": Successfully inlined array access set method in BinaryOp!");
                                // Return the inlined set method call
                                return inlined;
                            } else {
                                // debug!(": Failed to inline array access set method in BinaryOp");
                                // Fall through to normal assignment
                            }
                        }
                    }
                }

                // OPERATOR OVERLOADING: Check if left operand has abstract type with @:op metadata
                if let Some((method_symbol, _abstract_symbol)) =
                    self.find_binary_operator_method(left.expr_type, operator)
                {
                    // debug!(": Found operator method for {:?} on type {:?}: method symbol {:?}",
                    //           operator, left.expr_type, method_symbol);

                    // Rewrite binary operation to method call:  `a + b` → `a.add(b)`
                    // Then try to inline it using existing infrastructure
                    if let Some(inlined) = self.try_inline_abstract_method(
                        left,
                        method_symbol,
                        &[(**right).clone()],
                        expr.expr_type,
                        expr.source_location,
                    ) {
                        // debug!(": Successfully inlined operator method!");
                        return inlined;
                    } else {
                        // debug!(": Operator method found but not inlined, generating method call");
                        // Fall through to generate a regular method call
                        // TODO: Generate MethodCall instead of Binary
                    }
                }

                // Check if this is an assignment operator
                match operator {
                    BinaryOperator::Assign
                    | BinaryOperator::AddAssign
                    | BinaryOperator::SubAssign
                    | BinaryOperator::MulAssign
                    | BinaryOperator::DivAssign
                    | BinaryOperator::ModAssign => {
                        // Assignments in expression position need special handling
                        // In HIR, assignments are statements, not expressions
                        // We need to create a block that performs the assignment and returns the value

                        // Create an assignment statement
                        let assign_stmt = HirStatement::Assign {
                            lhs: self.lower_lvalue(left),
                            rhs: self.lower_expression(right),
                            op: match operator {
                                BinaryOperator::AddAssign => Some(HirBinaryOp::Add),
                                BinaryOperator::SubAssign => Some(HirBinaryOp::Sub),
                                BinaryOperator::MulAssign => Some(HirBinaryOp::Mul),
                                BinaryOperator::DivAssign => Some(HirBinaryOp::Div),
                                BinaryOperator::ModAssign => Some(HirBinaryOp::Mod),
                                _ => None, // Simple assignment
                            },
                        };

                        // Create a variable reference to the assigned value
                        let result_expr = self.lower_expression(left);

                        // Wrap in a block that performs assignment and returns the value
                        HirExprKind::Block(HirBlock {
                            statements: vec![assign_stmt],
                            expr: Some(Box::new(result_expr)),
                            scope: self.current_scope,
                        })
                    }
                    _ => {
                        // Regular binary operators
                        HirExprKind::Binary {
                            op: self.convert_binary_op(operator),
                            lhs: Box::new(self.lower_expression(left)),
                            rhs: Box::new(self.lower_expression(right)),
                        }
                    }
                }
            }
            TypedExpressionKind::Cast {
                expression,
                target_type,
                cast_kind,
            } => {
                use CastKind;
                HirExprKind::Cast {
                    expr: Box::new(self.lower_expression(expression)),
                    target: *target_type,
                    is_safe: !matches!(cast_kind, CastKind::Unsafe),
                }
            }
            TypedExpressionKind::Conditional {
                condition,
                then_expr,
                else_expr,
            } => HirExprKind::If {
                condition: Box::new(self.lower_expression(condition)),
                then_expr: Box::new(self.lower_expression(then_expr)),
                else_expr: Box::new(
                    else_expr
                        .as_ref()
                        .map(|e| self.lower_expression(e))
                        .unwrap_or_else(|| self.make_null_literal()),
                ),
            },
            TypedExpressionKind::FunctionLiteral {
                parameters,
                body,
                return_type: _,
            } => {
                // Compute captured variables from the lambda body
                let param_symbols: std::collections::HashSet<_> =
                    parameters.iter().map(|p| p.symbol_id).collect();

                let captures = self.compute_captures(body, &param_symbols);

                debug!(
                    "DEBUG TAST->HIR: Lambda has {} parameters, found {} captures",
                    parameters.len(),
                    captures.len()
                );
                for capture in &captures {
                    debug!("  Captured symbol: {:?}", capture.symbol);
                }

                HirExprKind::Lambda {
                    params: parameters.iter().map(|p| self.lower_param(p)).collect(),
                    body: Box::new(self.lower_statements_as_expr(body)),
                    captures,
                }
            }
            TypedExpressionKind::ArrayLiteral { elements } => HirExprKind::Array {
                elements: elements.iter().map(|e| self.lower_expression(e)).collect(),
            },
            TypedExpressionKind::ObjectLiteral { fields, .. } => HirExprKind::ObjectLiteral {
                fields: fields
                    .iter()
                    .map(|f| (f.name.clone(), self.lower_expression(&f.value)))
                    .collect(),
            },
            TypedExpressionKind::MapLiteral { entries } => HirExprKind::Map {
                entries: entries
                    .iter()
                    .map(|entry| {
                        (
                            self.lower_expression(&entry.key),
                            self.lower_expression(&entry.value),
                        )
                    })
                    .collect(),
            },
            TypedExpressionKind::Block {
                statements,
                scope_id,
            } => {
                let hir_block = HirBlock {
                    statements: statements.iter().map(|s| self.lower_statement(s)).collect(),
                    expr: None, // Block expression result handled separately
                    scope: *scope_id,
                };
                HirExprKind::Block(hir_block)
            }
            TypedExpressionKind::StringInterpolation { parts } => {
                // Desugar string interpolation to concatenation
                // "Hello ${name}!" becomes "Hello " + name + "!"
                // This simplifies later optimization and code generation

                if parts.is_empty() {
                    return HirExpr::new(
                        HirExprKind::Literal(HirLiteral::String(self.intern_str(""))),
                        self.get_string_type(),
                        self.current_lifetime,
                        expr.source_location,
                    );
                }

                let mut result = None;

                for part in parts {
                    let part_expr = match part {
                        StringInterpolationPart::String(s) => {
                            HirExpr::new(
                                HirExprKind::Literal(HirLiteral::String(self.intern_str(s))),
                                expr.expr_type, // Use the expression's string type
                                self.current_lifetime,
                                expr.source_location,
                            )
                        }
                        StringInterpolationPart::Expression(e) => {
                            // TODO: Add toString() conversion if needed
                            self.lower_expression(e)
                        }
                    };

                    result = match result {
                        None => Some(part_expr),
                        Some(left) => {
                            // Create concatenation: left + part_expr
                            Some(HirExpr::new(
                                HirExprKind::Binary {
                                    op: HirBinaryOp::Add, // String concatenation
                                    lhs: Box::new(left),
                                    rhs: Box::new(part_expr),
                                },
                                self.get_string_type(),
                                self.current_lifetime,
                                expr.source_location,
                            ))
                        }
                    };
                }

                result.map(|e| e.kind).unwrap_or_else(|| {
                    HirExprKind::Literal(HirLiteral::String(self.intern_str("")))
                })
            }
            TypedExpressionKind::ArrayComprehension {
                for_parts,
                expression,
                element_type,
            } => {
                // Desugar array comprehension to a loop that builds an array
                self.desugar_array_comprehension(for_parts, expression, *element_type)
            }
            TypedExpressionKind::Return { value } => {
                // Return as an expression creates a block that never returns normally
                let return_stmt =
                    HirStatement::Return(value.as_ref().map(|v| self.lower_expression(v)));
                let block = HirBlock {
                    statements: vec![return_stmt],
                    expr: None,
                    scope: self.current_scope,
                };
                HirExprKind::Block(block)
            }
            TypedExpressionKind::Break => {
                // Break as an expression
                let break_stmt = HirStatement::Break(None); // TODO: Handle labeled breaks
                let block = HirBlock {
                    statements: vec![break_stmt],
                    expr: None,
                    scope: self.current_scope,
                };
                HirExprKind::Block(block)
            }
            TypedExpressionKind::Continue => {
                // Continue as an expression
                let continue_stmt = HirStatement::Continue(None); // TODO: Handle labeled continues
                let block = HirBlock {
                    statements: vec![continue_stmt],
                    expr: None,
                    scope: self.current_scope,
                };
                HirExprKind::Block(block)
            }
            TypedExpressionKind::Switch {
                discriminant,
                cases,
                default_case,
            } => {
                // Check if any case has constructor patterns (enum matching)
                let has_constructor_patterns = cases.iter().any(|case| {
                    matches!(
                        &case.case_value.kind,
                        TypedExpressionKind::PatternPlaceholder { pattern, .. }
                            if matches!(pattern, parser::Pattern::Constructor { .. })
                    ) || matches!(
                        &case.case_value.kind,
                        TypedExpressionKind::FunctionCall { function, .. }
                            if matches!(&function.kind, TypedExpressionKind::Variable { symbol_id }
                                if self.symbol_table.get_symbol(*symbol_id)
                                    .map(|s| s.kind == crate::tast::symbols::SymbolKind::EnumVariant)
                                    .unwrap_or(false))
                    ) || matches!(
                        &case.case_value.kind,
                        TypedExpressionKind::Variable { symbol_id }
                            if self.symbol_table.get_symbol(*symbol_id)
                                .map(|s| s.kind == crate::tast::symbols::SymbolKind::EnumVariant)
                                .unwrap_or(false)
                    )
                });

                if has_constructor_patterns {
                    // Lower as a proper switch statement with pattern matching
                    let scrutinee_expr = self.lower_expression(discriminant);
                    let mut hir_cases = Vec::new();

                    for case in cases {
                        let patterns = vec![self.lower_case_value_pattern(&case.case_value)];
                        let body = match &case.body {
                            TypedStatement::Expression { expression, .. } => {
                                let body_stmt =
                                    HirStatement::Expr(self.lower_expression(expression));
                                HirBlock {
                                    statements: vec![body_stmt],
                                    expr: None,
                                    scope: self.current_scope,
                                }
                            }
                            _ => HirBlock {
                                statements: vec![self.lower_statement(&case.body)],
                                expr: None,
                                scope: self.current_scope,
                            },
                        };

                        hir_cases.push(HirMatchCase {
                            patterns,
                            guard: None,
                            body,
                        });
                    }

                    // Add default case if present
                    if let Some(default) = default_case {
                        hir_cases.push(HirMatchCase {
                            patterns: vec![HirPattern::Wildcard],
                            guard: None,
                            body: HirBlock {
                                statements: vec![HirStatement::Expr(
                                    self.lower_expression(default),
                                )],
                                expr: None,
                                scope: self.current_scope,
                            },
                        });
                    }

                    let switch_stmt = HirStatement::Switch {
                        scrutinee: scrutinee_expr,
                        cases: hir_cases,
                    };

                    // Wrap the switch statement in a block expression
                    let block = HirBlock {
                        statements: vec![switch_stmt],
                        expr: None,
                        scope: self.current_scope,
                    };

                    HirExprKind::Block(block)
                } else {
                    // Simple value matching: convert to if-then-else chain
                    let discriminant_expr = self.lower_expression(discriminant);
                    let mut current_expr = default_case
                        .as_ref()
                        .map(|expr| self.lower_expression(expr))
                        .unwrap_or_else(|| self.make_null_literal());

                    // Build if-then-else chain from right to left
                    for case in cases.iter().rev() {
                        let case_value = self.lower_expression(&case.case_value);
                        let case_body = match &case.body {
                            TypedStatement::Expression { expression, .. } => {
                                self.lower_expression(expression)
                            }
                            _ => {
                                let block = HirBlock {
                                    statements: vec![self.lower_statement(&case.body)],
                                    expr: None,
                                    scope: self.current_scope,
                                };
                                HirExpr::new(
                                    HirExprKind::Block(block),
                                    expr.expr_type,
                                    self.current_lifetime,
                                    SourceLocation::unknown(),
                                )
                            }
                        };

                        let condition = HirExpr::new(
                            HirExprKind::Binary {
                                lhs: Box::new(discriminant_expr.clone()),
                                op: crate::ir::hir::HirBinaryOp::Eq,
                                rhs: Box::new(case_value),
                            },
                            self.get_bool_type(),
                            self.current_lifetime,
                            SourceLocation::unknown(),
                        );

                        current_expr = HirExpr::new(
                            HirExprKind::If {
                                condition: Box::new(condition),
                                then_expr: Box::new(case_body),
                                else_expr: Box::new(current_expr),
                            },
                            self.get_dynamic_type(),
                            self.current_lifetime,
                            SourceLocation::unknown(),
                        );
                    }

                    current_expr.kind
                }
            }
            TypedExpressionKind::Throw { expression } => {
                // Throw as an expression creates a block that throws
                let throw_expr = self.lower_expression(expression);
                let throw_stmt = HirStatement::Throw(throw_expr);
                let block = HirBlock {
                    statements: vec![throw_stmt],
                    expr: None,
                    scope: self.current_scope,
                };
                HirExprKind::Block(block)
            }
            TypedExpressionKind::Try {
                try_expr,
                catch_clauses,
                finally_block,
            } => {
                // Lower try-catch-finally expression to HIR
                let try_body = self.lower_expression(try_expr);
                let catch_handlers = catch_clauses
                    .iter()
                    .map(|clause| {
                        // Lower the catch body
                        let body = match &clause.body {
                            TypedStatement::Expression { expression, .. } => {
                                self.lower_expression(expression)
                            }
                            _ => {
                                let stmt = self.lower_statement(&clause.body);
                                // Wrap statement in a block expression
                                HirExpr::new(
                                    HirExprKind::Block(HirBlock {
                                        statements: vec![stmt],
                                        expr: None,
                                        scope: self.current_scope,
                                    }),
                                    self.get_void_type(),
                                    self.current_lifetime,
                                    SourceLocation::unknown(),
                                )
                            }
                        };

                        crate::ir::hir::HirCatchHandler {
                            exception_var: clause.exception_variable,
                            exception_type: clause.exception_type,
                            guard: clause
                                .filter
                                .as_ref()
                                .map(|f| Box::new(self.lower_expression(f))),
                            body: Box::new(body),
                        }
                    })
                    .collect();

                let finally_expr = finally_block
                    .as_ref()
                    .map(|f| Box::new(self.lower_expression(f)));

                HirExprKind::TryCatch {
                    try_expr: Box::new(try_body),
                    catch_handlers,
                    finally_expr,
                }
            }
            TypedExpressionKind::StaticMethodCall {
                class_symbol,
                method_symbol,
                type_arguments,
                arguments,
            } => {
                // Lower static method call to a regular function call
                // Static methods are just functions in the class namespace
                HirExprKind::Call {
                    callee: Box::new(HirExpr::new(
                        HirExprKind::Variable {
                            symbol: *method_symbol,
                            capture_mode: None,
                        },
                        expr.expr_type,
                        expr.lifetime_id,
                        expr.source_location,
                    )),
                    type_args: type_arguments.clone(),
                    args: arguments
                        .iter()
                        .map(|arg| self.lower_expression(arg))
                        .collect(),
                    is_method: false, // Static methods are regular function calls
                }
            }
            TypedExpressionKind::PatternPlaceholder {
                pattern,
                source_location,
                ..
            } => {
                // Complex patterns (object, type, enum constructor, array rest) are
                // not yet compiled to decision trees. For now, lower as a wildcard
                // match (null) so the switch case compiles. This is semantically a
                // placeholder — proper pattern compilation is future work.
                HirExprKind::Null
            }
            TypedExpressionKind::StaticFieldAccess {
                class_symbol,
                field_symbol,
            } => {
                // Static field access is a variable reference to the field symbol
                HirExprKind::Variable {
                    symbol: *field_symbol,
                    capture_mode: None,
                }
            }
            TypedExpressionKind::While {
                condition,
                then_expr,
            } => {
                // While loops as expressions: wrap in a block with a while statement
                let cond = self.lower_expression(condition);
                let body_expr = self.lower_expression(then_expr);
                let while_stmt = HirStatement::While {
                    label: None,
                    condition: cond,
                    body: HirBlock {
                        statements: vec![HirStatement::Expr(body_expr)],
                        expr: None,
                        scope: self.current_scope,
                    },
                    continue_update: None,
                };
                HirExprKind::Block(HirBlock {
                    statements: vec![while_stmt],
                    expr: None,
                    scope: self.current_scope,
                })
            }
            TypedExpressionKind::For {
                variable,
                iterable,
                body,
            } => {
                let _iter_expr = self.lower_expression(iterable);
                let body_expr = self.lower_expression(body);
                let while_stmt = HirStatement::While {
                    label: None,
                    condition: self.make_bool_literal(true),
                    body: HirBlock {
                        statements: vec![HirStatement::Expr(body_expr)],
                        expr: None,
                        scope: self.current_scope,
                    },
                    continue_update: None,
                };
                HirExprKind::Block(HirBlock {
                    statements: vec![while_stmt],
                    expr: None,
                    scope: self.current_scope,
                })
            }
            TypedExpressionKind::ForIn {
                value_var,
                key_var,
                iterable,
                body,
            } => {
                let _iter_expr = self.lower_expression(iterable);
                let body_expr = self.lower_expression(body);
                let while_stmt = HirStatement::While {
                    label: None,
                    condition: self.make_bool_literal(true),
                    body: HirBlock {
                        statements: vec![HirStatement::Expr(body_expr)],
                        expr: None,
                        scope: self.current_scope,
                    },
                    continue_update: None,
                };
                HirExprKind::Block(HirBlock {
                    statements: vec![while_stmt],
                    expr: None,
                    scope: self.current_scope,
                })
            }
            TypedExpressionKind::Is {
                expression,
                check_type,
            } => {
                let inner = self.lower_expression(expression);
                HirExprKind::TypeCheck {
                    expr: Box::new(inner),
                    expected: *check_type,
                }
            }
            TypedExpressionKind::Meta {
                metadata,
                expression,
            } => {
                // Metadata-annotated expression — just lower the inner expression
                return self.lower_expression(expression);
            }
            TypedExpressionKind::MacroExpression {
                macro_symbol,
                arguments,
            } => {
                // Macro expression — lower as a function call
                HirExprKind::Call {
                    callee: Box::new(HirExpr::new(
                        HirExprKind::Variable {
                            symbol: *macro_symbol,
                            capture_mode: None,
                        },
                        expr.expr_type,
                        expr.lifetime_id,
                        expr.source_location,
                    )),
                    type_args: Vec::new(),
                    args: arguments
                        .iter()
                        .map(|arg| self.lower_expression(arg))
                        .collect(),
                    is_method: false,
                }
            }
            TypedExpressionKind::CompilerSpecific { target, code, args } => {
                let target_str = self
                    .string_interner
                    .get(*target)
                    .unwrap_or("unknown")
                    .to_string();
                let code_expr = self.lower_expression(code);
                let hir_args = args.iter().map(|a| self.lower_expression(a)).collect();
                HirExprKind::InlineCode {
                    target: target_str,
                    code: Box::new(code_expr),
                    args: hir_args,
                }
            }
            // Handle remaining expression types with error recovery
            _ => {
                let error_msg = self.get_expression_type_name(&expr.kind);
                // Use error recovery but still return a valid HIR node
                return self.make_error_expr(&error_msg, expr.source_location);
            }
        };

        HirExpr::new(kind, expr.expr_type, expr.lifetime_id, expr.source_location)
    }

    /// Lower a block of statements
    fn lower_block(&mut self, statements: &[TypedStatement]) -> HirBlock {
        // Debug: Print statement types
        for (i, stmt) in statements.iter().enumerate() {
            let stmt_type = match stmt {
                TypedStatement::VarDeclaration { .. } => "VarDeclaration",
                TypedStatement::Assignment { .. } => "Assignment",
                TypedStatement::Expression { .. } => "Expression",
                TypedStatement::Return { .. } => "Return",
                TypedStatement::If { .. } => "If",
                TypedStatement::While { .. } => "While",
                TypedStatement::For { .. } => "For",
                TypedStatement::Break { .. } => "Break",
                TypedStatement::Continue { .. } => "Continue",
                TypedStatement::Block { .. } => "Block",
                _ => "Other",
            };
            // debug!(" TAST statement {}: {}", i, stmt_type);
        }

        let hir_stmts: Vec<_> = statements.iter().map(|s| self.lower_statement(s)).collect();

        // Debug: Print HIR statement types
        for (i, stmt) in hir_stmts.iter().enumerate() {
            let stmt_type = match stmt {
                HirStatement::Let { .. } => "Let",
                HirStatement::Assign { .. } => "Assign",
                HirStatement::Expr(_) => "Expr",
                HirStatement::Return(_) => "Return",
                HirStatement::Break(_) => "Break",
                HirStatement::Continue(_) => "Continue",
                _ => "Other",
            };
            // debug!(" HIR statement {}: {}", i, stmt_type);
        }

        HirBlock::new(hir_stmts, self.current_scope)
    }

    // Helper methods...

    fn lower_pattern(&mut self, pattern: &TypedPattern) -> HirPattern {
        match pattern {
            TypedPattern::Variable { symbol_id, .. } => HirPattern::Variable {
                name: self.get_symbol_name(*symbol_id),
                symbol: *symbol_id,
            },
            TypedPattern::Wildcard { .. } => HirPattern::Wildcard,
            TypedPattern::Literal { value, .. } => {
                HirPattern::Literal(self.lower_expression_as_literal(value))
            }
            TypedPattern::Constructor {
                constructor, args, ..
            } => {
                // Extract enum type from constructor symbol
                HirPattern::Constructor {
                    // TODO: Get actual enum type from constructor symbol
                    enum_type: self.lookup_enum_type_from_constructor(*constructor),
                    variant: self.get_symbol_name(*constructor),
                    fields: args.iter().map(|f| self.lower_pattern(f)).collect(),
                }
            }
            // TAST doesn't have Tuple pattern, handle via Array
            TypedPattern::Array { elements, rest, .. } => HirPattern::Array {
                elements: elements.iter().map(|e| self.lower_pattern(e)).collect(),
                rest: rest.as_ref().map(|r| Box::new(self.lower_pattern(r))),
            },
            TypedPattern::Object { fields, .. } => {
                HirPattern::Object {
                    fields: fields
                        .iter()
                        .map(|f| {
                            (
                                self.intern_str(&f.field_name),
                                self.lower_pattern(&f.pattern),
                            )
                        })
                        .collect(),
                    rest: false, // TODO: Extract from pattern
                }
            }
            // TAST doesn't have Or pattern directly
            TypedPattern::Guard { pattern, guard } => HirPattern::Guard {
                pattern: Box::new(self.lower_pattern(pattern)),
                condition: self.lower_expression(guard),
            },
            TypedPattern::Extractor { .. } => {
                // Extractor patterns need special handling
                HirPattern::Wildcard
            }
        }
    }

    fn lower_lvalue(&mut self, expr: &TypedExpression) -> HirLValue {
        match &expr.kind {
            TypedExpressionKind::Variable { symbol_id } => HirLValue::Variable(*symbol_id),
            TypedExpressionKind::FieldAccess {
                object,
                field_symbol,
                ..
            } => HirLValue::Field {
                object: Box::new(self.lower_expression(object)),
                field: *field_symbol,
            },
            TypedExpressionKind::ArrayAccess { array, index } => HirLValue::Index {
                object: Box::new(self.lower_expression(array)),
                index: Box::new(self.lower_expression(index)),
            },
            _ => {
                self.add_error("Invalid assignment target", expr.source_location);
                // Return invalid symbol - this will be caught during validation
                HirLValue::Variable(SymbolId::invalid())
            }
        }
    }

    fn lower_literal(&mut self, lit: &LiteralValue) -> HirLiteral {
        match lit {
            LiteralValue::Int(i) => HirLiteral::Int(*i),
            LiteralValue::Float(f) => HirLiteral::Float(*f),
            LiteralValue::String(s) => HirLiteral::String(self.intern_str(s)),
            LiteralValue::Bool(b) => HirLiteral::Bool(*b),
            LiteralValue::Char(c) => HirLiteral::String(self.intern_str(&c.to_string())),
            LiteralValue::Regex(pattern) => HirLiteral::Regex {
                pattern: self.intern_str(pattern),
                flags: self.intern_str(""),
            },
            LiteralValue::RegexWithFlags { pattern, flags } => HirLiteral::Regex {
                pattern: self.intern_str(pattern),
                flags: self.intern_str(flags),
            },
        }
    }

    fn convert_unary_op(&self, op: &UnaryOperator) -> HirUnaryOp {
        match op {
            UnaryOperator::Not => HirUnaryOp::Not,
            UnaryOperator::Neg => HirUnaryOp::Neg,
            UnaryOperator::BitNot => HirUnaryOp::BitNot,
            UnaryOperator::PreInc => HirUnaryOp::PreIncr,
            UnaryOperator::PreDec => HirUnaryOp::PreDecr,
            UnaryOperator::PostInc => HirUnaryOp::PostIncr,
            UnaryOperator::PostDec => HirUnaryOp::PostDecr,
        }
    }

    fn convert_binary_op(&self, op: &BinaryOperator) -> HirBinaryOp {
        match op {
            BinaryOperator::Add => HirBinaryOp::Add,
            BinaryOperator::Sub => HirBinaryOp::Sub,
            BinaryOperator::Mul => HirBinaryOp::Mul,
            BinaryOperator::Div => HirBinaryOp::Div,
            BinaryOperator::Mod => HirBinaryOp::Mod,
            BinaryOperator::Eq => HirBinaryOp::Eq,
            BinaryOperator::Ne => HirBinaryOp::Ne,
            BinaryOperator::Lt => HirBinaryOp::Lt,
            BinaryOperator::Le => HirBinaryOp::Le,
            BinaryOperator::Gt => HirBinaryOp::Gt,
            BinaryOperator::Ge => HirBinaryOp::Ge,
            BinaryOperator::And => HirBinaryOp::And,
            BinaryOperator::Or => HirBinaryOp::Or,
            BinaryOperator::BitAnd => HirBinaryOp::BitAnd,
            BinaryOperator::BitOr => HirBinaryOp::BitOr,
            BinaryOperator::BitXor => HirBinaryOp::BitXor,
            BinaryOperator::Shl => HirBinaryOp::Shl,
            BinaryOperator::Shr => HirBinaryOp::Shr,
            // Range and NullCoalesce might not exist in TAST
            _ => HirBinaryOp::Add, // Default fallback
        }
    }

    fn convert_visibility(&self, vis: Visibility) -> HirVisibility {
        match vis {
            Visibility::Public => HirVisibility::Public,
            Visibility::Private => HirVisibility::Private,
            Visibility::Protected => HirVisibility::Protected,
            Visibility::Internal => HirVisibility::Internal,
        }
    }

    fn lower_import(&mut self, import: &TypedImport) {
        // Convert module path from InternedString to String
        let module_path_str = self
            .string_interner
            .get(import.module_path)
            .unwrap_or("Unknown");
        let module_path = module_path_str.split('.').map(|s| s.to_string()).collect();

        // Convert imported symbols from InternedString to SymbolId
        // For now, we just track the symbol names - actual symbol resolution happens during import
        let imported_symbols = if let Some(symbols) = &import.imported_symbols {
            // Create symbol IDs for each imported symbol name
            symbols
                .iter()
                .filter_map(|&sym_str| {
                    // Look up the symbol in the symbol table
                    // The symbol should have been registered during TAST lowering
                    // if let Some(name) = self.string_interner.get(sym_str) {
                    //     debug!(": Lowering import symbol: {}", name);
                    // }
                    None // For now, we don't track the actual SymbolIds in HIR imports
                })
                .collect()
        } else {
            Vec::new()
        };

        // Convert alias if present
        let alias = import
            .alias
            .and_then(|a| self.string_interner.get(a).map(|s| s.to_string()));

        let hir_import = HirImport {
            module_path,
            imported_symbols,
            alias,
            is_static_extension: false, // TODO: detect 'using' imports
        };

        self.module.imports.push(hir_import);
    }

    fn lower_module_field(&mut self, _field: &TypedModuleField) {
        // TODO: Implement module field lowering
    }

    fn lower_type_params(&mut self, params: &[TypedTypeParameter]) -> Vec<HirTypeParam> {
        params
            .iter()
            .map(|param| {
                HirTypeParam {
                    name: param.name.clone(),
                    bounds: param.constraints.clone(),
                    default: None, // TODO: Add default type support if needed
                }
            })
            .collect()
    }

    fn lower_param(&mut self, param: &TypedParameter) -> HirParam {
        HirParam {
            symbol_id: param.symbol_id, // Preserve symbol ID for variable lookup in MIR!
            name: param.name.clone(),
            ty: param.param_type,
            default: param
                .default_value
                .as_ref()
                .map(|e| self.lower_expression(e)),
            is_optional: param.is_optional,
            is_rest: false, // TODO: Extract from parameter metadata
        }
    }

    fn lower_metadata(&mut self, _metadata: &[TypedMetadata]) -> Vec<HirAttribute> {
        // TODO: Implement metadata lowering
        Vec::new()
    }

    /// Desugar array comprehension into a loop that builds an array
    /// [for (x in xs) if (condition) expression] becomes:
    /// {
    ///     let result = [];
    ///     for (x in xs) {
    ///         if (condition) {
    ///             result.push(expression);
    ///         }
    ///     }
    ///     result
    /// }
    fn desugar_array_comprehension(
        &mut self,
        for_parts: &[TypedComprehensionFor],
        expression: &TypedExpression,
        element_type: TypeId,
    ) -> HirExprKind {
        // Full desugaring: [for (x in xs) if (cond) expr] becomes:
        // {
        //     let _tmp = [];
        //     for (x in xs) {
        //         if (cond) {
        //             _tmp.push(expr);
        //         }
        //     }
        //     _tmp
        // }

        let mut statements = Vec::new();

        // 1. Create temporary array variable
        let (temp_name, temp_symbol) = self.gen_temp_var();
        let array_type = self.type_table.borrow_mut().create_array_type(element_type);

        // Create empty array literal
        let empty_array = HirExpr::new(
            HirExprKind::Array {
                elements: Vec::new(),
            },
            array_type,
            self.current_lifetime,
            SourceLocation::unknown(),
        );

        // let _tmp = []
        statements.push(HirStatement::Let {
            pattern: HirPattern::Variable {
                name: temp_name.clone(),
                symbol: temp_symbol,
            },
            type_hint: Some(array_type),
            init: Some(empty_array),
            is_mutable: true,
        });

        // 2. Build nested for loops
        let mut current_body = match self.build_comprehension_body(
            expression,
            temp_symbol,
            temp_name.clone(),
            array_type,
        ) {
            Ok(body) => body,
            Err(err) => {
                // If we can't build the comprehension body, return an error expression
                let error_expr = self.make_error_expr(&err, SourceLocation::unknown());
                return error_expr.kind;
            }
        };

        // Iterate through for parts in reverse to build nested structure
        for for_part in for_parts.iter().rev() {
            let pattern = if let Some(key_var) = for_part.key_var_symbol {
                // Key-value iteration
                HirPattern::Tuple(vec![
                    HirPattern::Variable {
                        name: self.get_symbol_name(key_var),
                        symbol: key_var,
                    },
                    HirPattern::Variable {
                        name: self.get_symbol_name(for_part.var_symbol),
                        symbol: for_part.var_symbol,
                    },
                ])
            } else {
                // Simple iteration
                HirPattern::Variable {
                    name: self.get_symbol_name(for_part.var_symbol),
                    symbol: for_part.var_symbol,
                }
            };

            // BACKLOG: TypedComprehensionFor doesn't have a condition field
            // Haxe array comprehensions support filters like: [for (x in xs) if (x > 0) x * 2]
            // Need to check how filters are represented in TAST
            // For now, no filter support in desugaring

            // Create for-in loop
            let for_stmt = HirStatement::ForIn {
                label: None,
                pattern,
                iterator: self.lower_expression(&for_part.iterator),
                body: current_body,
            };

            current_body = HirBlock::new(vec![for_stmt], self.current_scope);
        }

        // Add the nested loops to statements
        statements.extend(current_body.statements);

        // 3. Return the temporary array variable
        let result_expr = HirExpr::new(
            HirExprKind::Variable {
                symbol: temp_symbol,
                capture_mode: None,
            },
            array_type,
            self.current_lifetime,
            SourceLocation::unknown(),
        );

        // Create block that evaluates to the array
        let block = HirBlock {
            statements,
            expr: Some(Box::new(result_expr)),
            scope: self.current_scope,
        };

        HirExprKind::Block(block)
    }

    /// Build the innermost body for array comprehension that pushes to the array
    fn build_comprehension_body(
        &mut self,
        expression: &TypedExpression,
        array_symbol: SymbolId,
        array_name: InternedString,
        array_type: TypeId,
    ) -> Result<HirBlock, String> {
        // Create _tmp.push(expression)
        let array_ref = HirExpr::new(
            HirExprKind::Variable {
                symbol: array_symbol,
                capture_mode: None,
            },
            array_type,
            self.current_lifetime,
            SourceLocation::unknown(),
        );

        // BACKLOG: Need proper method resolution for Array.push
        // See src/ir/BACKLOG.md section 2a
        // For now, we cannot properly resolve the push method symbol
        // This blocks correct array comprehension desugaring

        // Attempt to look up push method from Array type
        let push_symbol = self.lookup_array_push_method(array_type)
            .ok_or_else(|| {
                format!("Cannot resolve Array.push method for comprehension desugaring. Array type methods must be loaded from haxe-std")
            })?;

        // Create method call: array.push(element)
        let push_call = HirExpr::new(
            HirExprKind::Call {
                callee: Box::new(HirExpr::new(
                    HirExprKind::Field {
                        object: Box::new(array_ref),
                        field: push_symbol,
                    },
                    self.get_dynamic_type(), // BACKLOG: Need actual method type
                    self.current_lifetime,
                    SourceLocation::unknown(),
                )),
                type_args: Vec::new(),
                args: vec![self.lower_expression(expression)],
                is_method: true,
            },
            self.get_void_type(),
            self.current_lifetime,
            SourceLocation::unknown(),
        );

        Ok(HirBlock::new(
            vec![HirStatement::Expr(push_call)],
            self.current_scope,
        ))
    }

    fn make_bool_literal(&self, value: bool) -> HirExpr {
        HirExpr::new(
            HirExprKind::Literal(HirLiteral::Bool(value)),
            self.get_bool_type(),
            self.current_lifetime,
            SourceLocation::unknown(),
        )
    }

    fn make_null_literal(&self) -> HirExpr {
        HirExpr::new(
            HirExprKind::Null,
            self.get_null_type(),
            self.current_lifetime,
            SourceLocation::unknown(),
        )
    }

    fn add_error(&mut self, msg: &str, location: SourceLocation) {
        self.errors.push(LoweringError {
            message: msg.to_string(),
            location,
        });
    }

    /// Create an error expression node that preserves structure during error recovery
    /// Get a human-readable name for an expression type
    fn get_expression_type_name(&self, kind: &TypedExpressionKind) -> String {
        match kind {
            TypedExpressionKind::Literal { .. } => "literal expression".to_string(),
            TypedExpressionKind::Variable { .. } => "variable reference".to_string(),
            TypedExpressionKind::This { .. } => "'this' reference".to_string(),
            TypedExpressionKind::Super { .. } => "'super' reference".to_string(),
            TypedExpressionKind::Null => "null literal".to_string(),
            TypedExpressionKind::FieldAccess { .. } => "field access".to_string(),
            TypedExpressionKind::ArrayAccess { .. } => "array access".to_string(),
            TypedExpressionKind::FunctionCall { .. } => "function call".to_string(),
            TypedExpressionKind::MethodCall { .. } => "method call".to_string(),
            TypedExpressionKind::StaticMethodCall { .. } => "static method call".to_string(),
            TypedExpressionKind::New { .. } => "object construction (new)".to_string(),
            TypedExpressionKind::UnaryOp { .. } => "unary operation".to_string(),
            TypedExpressionKind::BinaryOp { .. } => "binary operation".to_string(),
            TypedExpressionKind::Cast { .. } => "type cast".to_string(),
            TypedExpressionKind::Conditional { .. } => {
                "conditional expression (ternary)".to_string()
            }
            TypedExpressionKind::FunctionLiteral { .. } => "function literal".to_string(),
            TypedExpressionKind::ArrayLiteral { .. } => "array literal".to_string(),
            TypedExpressionKind::ObjectLiteral { .. } => "object literal".to_string(),
            TypedExpressionKind::MapLiteral { .. } => "map literal".to_string(),
            TypedExpressionKind::Block { .. } => "block expression".to_string(),
            TypedExpressionKind::StringInterpolation { .. } => "string interpolation".to_string(),
            TypedExpressionKind::ArrayComprehension { .. } => "array comprehension".to_string(),
            TypedExpressionKind::Return { .. } => "return statement".to_string(),
            TypedExpressionKind::Break => "break statement".to_string(),
            TypedExpressionKind::Continue => "continue statement".to_string(),
            TypedExpressionKind::Switch { .. } => "switch expression".to_string(),
            TypedExpressionKind::Throw { .. } => "throw expression".to_string(),
            TypedExpressionKind::Try { .. } => "try-catch expression".to_string(),
            TypedExpressionKind::VarDeclarationExpr { .. } => "variable declaration".to_string(),
            TypedExpressionKind::FinalDeclarationExpr { .. } => "final declaration".to_string(),
            _ => "unsupported expression type".to_string(),
        }
    }

    fn make_error_expr(&mut self, msg: &str, location: SourceLocation) -> HirExpr {
        self.add_error(msg, location);
        HirExpr::new(
            // Use Untyped as a placeholder for error expressions
            HirExprKind::Untyped(Box::new(HirExpr::new(
                HirExprKind::Null,
                self.get_dynamic_type(),
                self.current_lifetime,
                location,
            ))),
            self.get_dynamic_type(),
            self.current_lifetime,
            location,
        )
    }

    /// Create an error statement node
    fn make_error_stmt(&mut self, msg: &str, location: SourceLocation) -> HirStatement {
        self.add_error(msg, location);
        // Return a no-op statement
        HirStatement::Expr(self.make_error_expr(msg, location))
    }

    /// Get symbol name from symbol table
    fn get_symbol_name(&self, symbol_id: SymbolId) -> InternedString {
        // Look up symbol name from the symbol table
        if let Some(symbol_info) = self.symbol_table.get_symbol(symbol_id) {
            symbol_info.name
        } else {
            // Fallback for invalid symbols
            let name = format!("unknown_sym_{}", symbol_id.as_raw());
            self.intern_str(&name)
        }
    }

    /// Intern a string
    fn intern_str(&self, s: &str) -> InternedString {
        self.string_interner.intern(s)
    }

    /// Generate a unique temporary variable name
    fn gen_temp_var(&mut self) -> (InternedString, SymbolId) {
        let name = format!("_tmp{}", self.temp_var_counter);
        self.temp_var_counter += 1;
        let interned = self.intern_str(&name);
        // Create a synthetic symbol ID for the temporary
        let symbol_id = SymbolId::from_raw(u32::MAX - self.temp_var_counter);
        (interned, symbol_id)
    }

    /// Look up the push method for Array type
    fn lookup_array_push_method(&self, array_type: TypeId) -> Option<SymbolId> {
        // Array is loaded from haxe-std/Array.hx as an extern class
        // The push method should be registered in the symbol table
        //
        // NOTE: This is a temporary workaround. In the future, we need to:
        // 1. Add package declarations to haxe-std files
        // 2. Properly resolve methods through type information
        // 3. Support global root-level Haxe standard modules

        // Strategy 1: Look for "Array.push" qualified name
        let array_push_qualified = self.string_interner.intern("Array.push");
        if let Some(symbol_id) = self
            .symbol_table
            .resolve_qualified_name(array_push_qualified)
        {
            return Some(symbol_id);
        }

        // Strategy 2: Try "haxe.Array.push" (if from std package)
        let haxe_array_push = self.string_interner.intern("haxe.Array.push");
        if let Some(symbol_id) = self.symbol_table.resolve_qualified_name(haxe_array_push) {
            return Some(symbol_id);
        }

        // Strategy 3: Search for ANY symbol with qualified_name ending in "Array.push"
        // This handles cases where Array is in different packages (stdlib, test, etc.)
        for symbol in self.symbol_table.all_symbols() {
            if let Some(qualified) = symbol.qualified_name {
                if let Some(qual_str) = self.string_interner.get(qualified) {
                    // Match any qualified name ending with ".Array.push"
                    // This will match "test.comprehensive.Array.push", "haxe.Array.push", etc.
                    if qual_str.ends_with(".Array.push") {
                        return Some(symbol.id);
                    }
                }
            }
            // Also check the name field directly
            if let Some(name_str) = self.string_interner.get(symbol.name) {
                if name_str == "Array.push" || name_str == "haxe.Array.push" {
                    return Some(symbol.id);
                }
            }
        }

        // If we still can't find it, the standard library wasn't properly loaded
        None
    }

    /// Desugar pattern matching statement to if-else chain
    fn desugar_pattern_match(
        &mut self,
        value: &TypedExpression,
        patterns: &[TypedPatternCase],
        source_location: SourceLocation,
    ) -> HirStatement {
        if patterns.is_empty() {
            return self.make_error_stmt("Pattern match with no cases", source_location);
        }

        // Generate temp variable to hold matched value
        let (match_var_name, match_var_symbol) = self.gen_temp_var();

        // Create let statement for matched value
        let match_let = HirStatement::Let {
            pattern: HirPattern::Variable {
                name: match_var_name.clone(),
                symbol: match_var_symbol,
            },
            type_hint: Some(value.expr_type),
            init: Some(self.lower_expression(value)),
            is_mutable: false,
        };

        // Build if-else chain from patterns
        let mut else_branch: Option<HirBlock> = None;

        // Process patterns in reverse to build nested if-else
        for (i, case) in patterns.iter().enumerate().rev() {
            let is_last = i == patterns.len() - 1;

            // Generate condition from pattern
            let (condition, bindings) =
                self.pattern_to_condition(&case.pattern, match_var_symbol, value.expr_type);

            // Add guard condition if present
            let final_condition = if let Some(guard) = &case.guard {
                // Combine pattern condition with guard: pattern_cond && guard
                HirExpr::new(
                    HirExprKind::Binary {
                        op: HirBinaryOp::And,
                        lhs: Box::new(condition),
                        rhs: Box::new(self.lower_expression(guard)),
                    },
                    self.get_bool_type(),
                    self.current_lifetime,
                    source_location,
                )
            } else {
                condition
            };

            // Build body with bindings
            let mut body_stmts = Vec::new();

            // Add variable bindings from pattern
            for (bind_symbol, bind_expr) in bindings {
                body_stmts.push(HirStatement::Let {
                    pattern: HirPattern::Variable {
                        name: self.get_symbol_name(bind_symbol),
                        symbol: bind_symbol,
                    },
                    type_hint: None, // Let type inference handle it
                    init: Some(bind_expr),
                    is_mutable: false,
                });
            }

            // Add the actual case body
            body_stmts.push(self.lower_statement(&case.body));

            let then_branch = HirBlock::new(body_stmts, self.current_scope);

            // Create if statement
            let if_stmt = HirStatement::If {
                condition: final_condition,
                then_branch,
                else_branch,
            };

            // This if becomes the else branch for the previous case
            else_branch = Some(HirBlock::new(vec![if_stmt], self.current_scope));
        }

        // Combine match variable declaration with the if-else chain
        let mut statements = vec![match_let];
        if let Some(else_block) = else_branch {
            // The else_block contains the entire if-else chain
            statements.extend(else_block.statements);
        }

        // Wrap in a block statement
        HirStatement::Expr(HirExpr::new(
            HirExprKind::Block(HirBlock::new(statements, self.current_scope)),
            self.get_void_type(),
            self.current_lifetime,
            source_location,
        ))
    }

    /// Convert a pattern to a condition expression and extract variable bindings
    fn pattern_to_condition(
        &mut self,
        pattern: &TypedPattern,
        match_var: SymbolId,
        match_type: TypeId,
    ) -> (HirExpr, Vec<(SymbolId, HirExpr)>) {
        let mut bindings = Vec::new();

        let condition = match pattern {
            TypedPattern::Wildcard { .. } => {
                // Wildcard always matches
                self.make_bool_literal(true)
            }
            TypedPattern::Variable {
                symbol_id,
                source_location,
                ..
            } => {
                // Variable pattern always matches and creates a binding
                let match_expr = HirExpr::new(
                    HirExprKind::Variable {
                        symbol: match_var,
                        capture_mode: None,
                    },
                    match_type,
                    self.current_lifetime,
                    *source_location,
                );
                bindings.push((*symbol_id, match_expr));
                self.make_bool_literal(true)
            }
            TypedPattern::Literal {
                value,
                source_location,
            } => {
                // Literal pattern: match_var == literal
                let match_expr = HirExpr::new(
                    HirExprKind::Variable {
                        symbol: match_var,
                        capture_mode: None,
                    },
                    match_type,
                    self.current_lifetime,
                    *source_location,
                );
                HirExpr::new(
                    HirExprKind::Binary {
                        op: HirBinaryOp::Eq,
                        lhs: Box::new(match_expr),
                        rhs: Box::new(self.lower_expression(value)),
                    },
                    self.get_bool_type(),
                    self.current_lifetime,
                    *source_location,
                )
            }
            TypedPattern::Constructor {
                constructor,
                args,
                source_location,
                ..
            } => {
                // Constructor pattern: check type and extract fields
                // BACKLOG: Need proper enum variant checking
                // This requires:
                // 1. Runtime type information for enum variants
                // 2. Field extraction from enum constructors
                // 3. Nested pattern matching for constructor arguments

                // For now, create a placeholder that always fails
                self.add_error(
                    "Constructor patterns not yet supported in desugaring",
                    *source_location,
                );
                self.make_bool_literal(false)
            }
            TypedPattern::Array {
                elements,
                rest,
                source_location,
                ..
            } => {
                // Array pattern: check length and elements
                // BACKLOG: Need proper array pattern matching
                // This requires:
                // 1. Array length checking
                // 2. Element extraction and matching
                // 3. Rest pattern handling

                self.add_error(
                    "Array patterns not yet supported in desugaring",
                    *source_location,
                );
                self.make_bool_literal(false)
            }
            TypedPattern::Object {
                fields,
                source_location,
                ..
            } => {
                // Object pattern: check fields
                // BACKLOG: Need proper object pattern matching
                // This requires:
                // 1. Field existence checking
                // 2. Field extraction and matching
                // 3. Nested pattern matching for field values

                self.add_error(
                    "Object patterns not yet supported in desugaring",
                    *source_location,
                );
                self.make_bool_literal(false)
            }
            TypedPattern::Guard { pattern, guard } => {
                // Guard pattern: pattern && guard
                let (pattern_cond, pattern_bindings) =
                    self.pattern_to_condition(pattern, match_var, match_type);
                bindings.extend(pattern_bindings);

                HirExpr::new(
                    HirExprKind::Binary {
                        op: HirBinaryOp::And,
                        lhs: Box::new(pattern_cond),
                        rhs: Box::new(self.lower_expression(guard)),
                    },
                    self.get_bool_type(),
                    self.current_lifetime,
                    guard.source_location,
                )
            }
            TypedPattern::Extractor {
                source_location, ..
            } => {
                // Extractor pattern: needs special handling
                // BACKLOG: Extractor patterns require method calls
                self.add_error(
                    "Extractor patterns not yet supported in desugaring",
                    *source_location,
                );
                self.make_bool_literal(false)
            }
        };

        (condition, bindings)
    }

    /// Extract function metadata
    fn extract_function_metadata(&self, metadata: &FunctionMetadata) -> Vec<HirAttribute> {
        let mut attrs = Vec::new();

        // Extract complexity score as metadata
        if metadata.complexity_score > 0 {
            let complexity_name = self.string_interner.intern("complexity");
            attrs.push(HirAttribute {
                name: complexity_name,
                args: vec![HirAttributeArg::Literal(HirLiteral::Int(
                    metadata.complexity_score as i64,
                ))],
            });
        }

        // Extract override marker
        if metadata.is_override {
            let override_name = self.string_interner.intern("override");
            attrs.push(HirAttribute {
                name: override_name,
                args: vec![],
            });
        }

        // Extract recursive marker
        if metadata.is_recursive {
            let recursive_name = self.string_interner.intern("recursive");
            attrs.push(HirAttribute {
                name: recursive_name,
                args: vec![],
            });
        }

        attrs
    }

    /// Lower a switch case value to a pattern
    fn lower_case_value_pattern(&mut self, expr: &TypedExpression) -> HirPattern {
        // Convert constant expression to pattern
        match &expr.kind {
            TypedExpressionKind::Literal { value } => {
                HirPattern::Literal(self.lower_literal(value))
            }
            // Simple enum variant without parameters (e.g., case Red:)
            TypedExpressionKind::Variable { symbol_id } => {
                if let Some(sym) = self.symbol_table.get_symbol(*symbol_id) {
                    if sym.kind == crate::tast::symbols::SymbolKind::EnumVariant {
                        // Find the parent enum type
                        let enum_type = self
                            .symbol_table
                            .find_parent_enum_for_constructor(*symbol_id)
                            .and_then(|parent_id| self.symbol_table.get_symbol(parent_id))
                            .map(|parent_sym| parent_sym.type_id)
                            .unwrap_or(expr.expr_type);
                        return HirPattern::Constructor {
                            enum_type,
                            variant: sym.name,
                            fields: vec![],
                        };
                    }
                }
                HirPattern::Wildcard
            }
            // Enum constructor with parameters (e.g., case Ok(_):, case Some(x):)
            TypedExpressionKind::FunctionCall {
                function,
                arguments,
                ..
            } => {
                if let TypedExpressionKind::Variable { symbol_id } = &function.kind {
                    if let Some(sym) = self.symbol_table.get_symbol(*symbol_id) {
                        if sym.kind == crate::tast::symbols::SymbolKind::EnumVariant {
                            let enum_type = self
                                .symbol_table
                                .find_parent_enum_for_constructor(*symbol_id)
                                .and_then(|parent_id| self.symbol_table.get_symbol(parent_id))
                                .map(|parent_sym| parent_sym.type_id)
                                .unwrap_or(function.expr_type);
                            let fields: Vec<HirPattern> = arguments
                                .iter()
                                .map(|arg| self.lower_case_value_pattern(arg))
                                .collect();
                            return HirPattern::Constructor {
                                enum_type,
                                variant: sym.name,
                                fields,
                            };
                        }
                    }
                }
                HirPattern::Wildcard
            }
            // Pattern placeholder from TAST (constructor patterns with params like Ok(_))
            TypedExpressionKind::PatternPlaceholder {
                pattern,
                variable_bindings,
                ..
            } => self.lower_parser_pattern_to_hir_with_bindings(pattern, variable_bindings),
            // Null expression is used as placeholder for wildcard patterns
            TypedExpressionKind::Null => HirPattern::Wildcard,
            _ => HirPattern::Wildcard,
        }
    }

    /// Convert a parser::Pattern to HirPattern for switch case matching
    fn lower_parser_pattern_to_hir(&mut self, pattern: &parser::Pattern) -> HirPattern {
        self.lower_parser_pattern_to_hir_with_bindings(pattern, &[])
    }

    /// Convert a parser::Pattern to HirPattern using pre-resolved variable bindings
    fn lower_parser_pattern_to_hir_with_bindings(
        &mut self,
        pattern: &parser::Pattern,
        variable_bindings: &[(InternedString, SymbolId)],
    ) -> HirPattern {
        match pattern {
            parser::Pattern::Constructor { path, params } => {
                let name_interned = self.string_interner.intern(&path.name);
                // Look up the enum variant symbol
                if let Some(sym) = self
                    .symbol_table
                    .lookup_symbol(crate::tast::ScopeId::first(), name_interned)
                {
                    let sym_id = sym.id;
                    if sym.kind == crate::tast::symbols::SymbolKind::EnumVariant {
                        let enum_type = self
                            .symbol_table
                            .find_parent_enum_for_constructor(sym_id)
                            .and_then(|parent_id| self.symbol_table.get_symbol(parent_id))
                            .map(|parent_sym| parent_sym.type_id)
                            .unwrap_or(TypeId::invalid());
                        let fields: Vec<HirPattern> = params
                            .iter()
                            .map(|p| {
                                self.lower_parser_pattern_to_hir_with_bindings(p, variable_bindings)
                            })
                            .collect();
                        return HirPattern::Constructor {
                            enum_type,
                            variant: name_interned,
                            fields,
                        };
                    }
                }
                HirPattern::Wildcard
            }
            parser::Pattern::Underscore => HirPattern::Wildcard,
            parser::Pattern::Var(name) => {
                // Variable patterns bind the matched value
                let name_interned = self.string_interner.intern(name);

                // First check pre-resolved bindings from AST lowering (most reliable)
                if let Some((_, sym_id)) =
                    variable_bindings.iter().find(|(n, _)| *n == name_interned)
                {
                    return HirPattern::Variable {
                        name: name_interned,
                        symbol: *sym_id,
                    };
                }

                // Fall back to scope-based lookup
                let found = self
                    .symbol_table
                    .lookup_symbol(self.current_scope, name_interned)
                    .map(|s| s.id)
                    .or_else(|| {
                        // Search all symbols for a variable with this name.
                        // Use max_by_key on SymbolId to get the most recently created
                        // symbol, ensuring deterministic behavior regardless of
                        // HashMap iteration order.
                        self.symbol_table
                            .all_symbols()
                            .filter(|s| {
                                s.name == name_interned
                                    && s.kind == crate::tast::symbols::SymbolKind::Variable
                            })
                            .max_by_key(|s| s.id)
                            .map(|s| s.id)
                    });
                if let Some(sym_id) = found {
                    HirPattern::Variable {
                        name: name_interned,
                        symbol: sym_id,
                    }
                } else {
                    HirPattern::Wildcard
                }
            }
            parser::Pattern::Const(expr) => {
                // Best effort: try to lower as a literal
                HirPattern::Wildcard
            }
            parser::Pattern::Null => HirPattern::Wildcard,
            _ => HirPattern::Wildcard,
        }
    }

    /// Convert a single statement to a block
    fn lower_statement_as_block(&mut self, stmt: &TypedStatement) -> HirBlock {
        HirBlock {
            statements: vec![self.lower_statement(stmt)],
            expr: None,
            scope: self.current_scope,
        }
    }

    /// Convert statements to an expression
    fn lower_statements_as_expr(&mut self, stmts: &[TypedStatement]) -> HirExpr {
        let block = self.lower_block(stmts);
        HirExpr::new(
            HirExprKind::Block(block),
            self.get_dynamic_type(), // Block type will be inferred
            self.current_lifetime,
            SourceLocation::unknown(),
        )
    }

    /// Look up method type from symbol table
    fn lookup_method_type(&mut self, method_symbol: SymbolId, receiver_type: TypeId) -> TypeId {
        // Try to get the method's type from the symbol table
        if let Some(symbol_info) = self.symbol_table.get_symbol(method_symbol) {
            if symbol_info.type_id != TypeId::invalid() {
                return symbol_info.type_id;
            }
        }

        // Fallback: return the receiver type (method should return something compatible)
        self.add_error(
            &format!("Method type not found for symbol {:?}", method_symbol),
            SourceLocation::unknown(),
        );
        receiver_type
    }

    /// Look up enum type from constructor symbol
    fn lookup_enum_type(&mut self, constructor: SymbolId) -> TypeId {
        // Look up the constructor's parent enum type
        if let Some(symbol_info) = self.symbol_table.get_symbol(constructor) {
            if symbol_info.type_id != TypeId::invalid() {
                // The constructor's type should reference the enum
                return symbol_info.type_id;
            }
        }

        self.add_error(
            &format!("Enum type not found for constructor {:?}", constructor),
            SourceLocation::unknown(),
        );
        self.get_dynamic_type() // Fallback to dynamic
    }

    /// Find a method with matching @:op metadata for a binary operator
    /// Returns (method_symbol, abstract_symbol) if found
    fn find_binary_operator_method(
        &self,
        operand_type: TypeId,
        operator: &BinaryOperator,
    ) -> Option<(SymbolId, SymbolId)> {
        // Check if this type is an abstract type
        let type_table = self.type_table.borrow();
        let type_info = type_table.get(operand_type)?;
        let abstract_symbol = match &type_info.kind {
            TypeKind::Abstract { symbol_id, .. } => *symbol_id,
            _ => return None, // Not an abstract type
        };
        drop(type_table);

        // Get the abstract definition from the current file
        let current_file = self.current_file?;

        // Search all abstracts for the one matching our symbol
        for abstract_def in &current_file.abstracts {
            if abstract_def.symbol_id != abstract_symbol {
                continue;
            }

            // Found the abstract, now search for a method with matching @:op metadata
            for method in &abstract_def.methods {
                for (op_str, _params) in &method.metadata.operator_metadata {
                    if let Some(parsed_op) = Self::parse_operator_from_metadata(op_str) {
                        // Compare using discriminant to match operator variants
                        if std::mem::discriminant(&parsed_op) == std::mem::discriminant(operator) {
                            // debug!(": Matched operator {:?} to method {} with metadata '{}'",
                            //           operator, self.string_interner.get(method.name).unwrap_or("<unknown>"), op_str);
                            return Some((method.symbol_id, abstract_symbol));
                        }
                    }
                }
            }
        }

        None
    }

    /// Parse operator metadata string to extract the operator type
    /// Format: "<ident> <Op> <ident>" for binary, "<Op><ident>" for unary
    /// e.g. "lhs Add rhs" → Some(BinaryOperator::Add) with 2 operands
    fn parse_operator_from_metadata(op_str: &str) -> Option<BinaryOperator> {
        // Split the string into tokens
        let tokens: Vec<&str> = op_str.split_whitespace().collect();

        // Binary operator pattern: "ident Op ident" (3 tokens)
        if tokens.len() == 3 {
            // Middle token should be the operator
            let operator = tokens[1];

            // Match against known binary operators
            match operator {
                "Add" => Some(BinaryOperator::Add),
                "Sub" => Some(BinaryOperator::Sub),
                "Mul" => Some(BinaryOperator::Mul),
                "Div" => Some(BinaryOperator::Div),
                "Mod" => Some(BinaryOperator::Mod),
                "Eq" => Some(BinaryOperator::Eq),
                "NotEq" | "Ne" => Some(BinaryOperator::Ne),
                "Lt" => Some(BinaryOperator::Lt),
                "Le" => Some(BinaryOperator::Le),
                "Gt" => Some(BinaryOperator::Gt),
                "Ge" => Some(BinaryOperator::Ge),
                _ => {
                    warn!(
                        "WARNING: Unknown binary operator in metadata: '{}'",
                        operator
                    );
                    None
                }
            }
        } else {
            // Not a valid binary operator pattern
            warn!(
                "WARNING: Invalid binary operator pattern (expected 3 tokens, got {}): '{}'",
                tokens.len(),
                op_str
            );
            None
        }
    }

    /// Find a method with matching @:op metadata for a unary operator
    /// Returns (method_symbol, abstract_symbol) if found
    fn find_unary_operator_method(
        &self,
        operand_type: TypeId,
        operator: &UnaryOperator,
    ) -> Option<(SymbolId, SymbolId)> {
        // Check if this type is an abstract type
        let type_table = self.type_table.borrow();
        let type_info = type_table.get(operand_type)?;
        let abstract_symbol = match &type_info.kind {
            TypeKind::Abstract { symbol_id, .. } => *symbol_id,
            _ => return None, // Not an abstract type
        };
        drop(type_table);

        // Get the abstract definition from the current file
        let current_file = self.current_file?;

        // Search all abstracts for the one matching our symbol
        for abstract_def in &current_file.abstracts {
            if abstract_def.symbol_id != abstract_symbol {
                continue;
            }

            // Found the abstract, now search for a method with matching @:op metadata
            for method in &abstract_def.methods {
                for (op_str, _params) in &method.metadata.operator_metadata {
                    if let Some(parsed_op) = Self::parse_unary_operator_from_metadata(op_str) {
                        // Compare using discriminant to match operator variants
                        if std::mem::discriminant(&parsed_op) == std::mem::discriminant(operator) {
                            // debug!(": Matched unary operator {:?} to method {} with metadata '{}'",
                            //           operator, self.string_interner.get(method.name).unwrap_or("<unknown>"), op_str);
                            return Some((method.symbol_id, abstract_symbol));
                        }
                    }
                }
            }
        }

        None
    }

    /// Parse unary operator metadata string to extract the operator type
    /// Format: "<Op><ident>" for prefix unary, "<ident><Op>" for postfix
    /// e.g. "Negvalue" → Some(UnaryOperator::Neg)
    /// e.g. "valuePreInc" → Some(UnaryOperator::PreInc)
    fn parse_unary_operator_from_metadata(op_str: &str) -> Option<UnaryOperator> {
        // Try to match unary operators
        // Common patterns:
        // - "Neg<ident>" for negation: -A
        // - "Not<ident>" for logical not: !A
        // - "BitNot<ident>" for bitwise not: ~A
        // - "PreInc<ident>" for pre-increment: ++A
        // - "<ident>PostInc" for post-increment: A++
        // - "PreDec<ident>" for pre-decrement: --A
        // - "<ident>PostDec" for post-decrement: A--

        if op_str.starts_with("Neg") {
            Some(UnaryOperator::Neg)
        } else if op_str.starts_with("Not") && !op_str.starts_with("NotEq") {
            // Careful: "NotEq" is a binary operator, "Not" is unary
            Some(UnaryOperator::Not)
        } else if op_str.starts_with("BitNot") {
            Some(UnaryOperator::BitNot)
        } else if op_str.starts_with("PreInc") || op_str.contains("PreInc") {
            Some(UnaryOperator::PreInc)
        } else if op_str.ends_with("PostInc") || op_str.contains("PostInc") {
            Some(UnaryOperator::PostInc)
        } else if op_str.starts_with("PreDec") || op_str.contains("PreDec") {
            Some(UnaryOperator::PreDec)
        } else if op_str.ends_with("PostDec") || op_str.contains("PostDec") {
            Some(UnaryOperator::PostDec)
        } else {
            // Not a recognized unary operator pattern
            None
        }
    }

    /// Find a method with @:arrayAccess metadata for array access operations
    /// Returns (method_symbol, abstract_symbol) if found
    /// method_name should be "get" for read access or "set" for write access
    fn find_array_access_method(
        &self,
        operand_type: TypeId,
        method_name: &str,
    ) -> Option<(SymbolId, SymbolId)> {
        // Check if this type is an abstract type
        let type_table = self.type_table.borrow();
        let type_info = type_table.get(operand_type)?;
        let abstract_symbol = match &type_info.kind {
            TypeKind::Abstract { symbol_id, .. } => *symbol_id,
            _ => return None, // Not an abstract type
        };
        drop(type_table);

        // Get the abstract definition from the current file
        let current_file = self.current_file?;

        // Search all abstracts for the one matching our symbol
        for abstract_def in &current_file.abstracts {
            if abstract_def.symbol_id != abstract_symbol {
                continue;
            }

            // Found the abstract, now search for a method with @:arrayAccess metadata
            for method in &abstract_def.methods {
                // Check if this method has @:arrayAccess metadata
                if method.metadata.is_array_access {
                    // Check if the method name matches what we're looking for
                    if let Some(name_str) = self.string_interner.get(method.name) {
                        if name_str == method_name {
                            debug!(
                                "DEBUG: Matched array access '{}' to method {} on abstract '{}'",
                                method_name,
                                name_str,
                                self.string_interner
                                    .get(abstract_def.name)
                                    .unwrap_or("<unknown>")
                            );
                            return Some((method.symbol_id, abstract_symbol));
                        }
                    }
                }
            }
        }

        None
    }

    /// Try to inline an abstract type method call
    /// Returns Some(inlined_expr) if successful, None otherwise
    fn try_inline_abstract_method(
        &mut self,
        receiver: &TypedExpression,
        method_symbol: SymbolId,
        arguments: &[TypedExpression],
        result_type: TypeId,
        source_location: SourceLocation,
    ) -> Option<HirExpr> {
        // Get the current file being processed
        if self.current_file.is_none() {
            return None;
        }
        let current_file = self.current_file?;

        // Try to find which abstract type (if any) contains this method
        // We search by method symbol OR by method name (as fallback for symbol mismatch)
        let method_name = if let Some(sym) = self.symbol_table.get_symbol(method_symbol) {
            sym.name
        } else {
            return None;
        };

        let mut found_abstract: Option<&crate::tast::node::TypedAbstract> = None;
        let mut found_method: Option<&crate::tast::node::TypedFunction> = None;

        for abstract_def in &current_file.abstracts {
            // Try to find method by symbol ID first
            if let Some(method) = abstract_def
                .methods
                .iter()
                .find(|m| m.symbol_id == method_symbol)
            {
                found_abstract = Some(abstract_def);
                found_method = Some(method);
                break;
            }
            // Fallback: match by name
            if let Some(method) = abstract_def.methods.iter().find(|m| m.name == method_name) {
                found_abstract = Some(abstract_def);
                found_method = Some(method);
                break;
            }
        }

        if found_abstract.is_none() || found_method.is_none() {
            return None; // Not an abstract method
        }

        let abstract_def = found_abstract.unwrap();
        let method = found_method.unwrap();

        // Found the abstract method — try to inline it

        // Inline the method body
        // Build a parameter mapping: parameter symbols -> argument expressions (already lowered to HIR)
        let lowered_arguments: Vec<HirExpr> = arguments
            .iter()
            .map(|arg| self.lower_expression(arg))
            .collect();

        let mut param_map: HashMap<SymbolId, HirExpr> = HashMap::new();

        // Map parameters to arguments
        if method.parameters.len() == arguments.len() {
            for (param, lowered_arg) in method.parameters.iter().zip(lowered_arguments.into_iter())
            {
                // println!("DEBUG: Mapping parameter {:?} to argument: {:?}", param.symbol_id, lowered_arg);
                param_map.insert(param.symbol_id, lowered_arg);
            }
        } else {
            // println!("DEBUG: Parameter count mismatch: method has {} params, call has {} args",
            //     method.parameters.len(), arguments.len());
        }
        // println!("DEBUG: param_map has {} entries", param_map.len());

        // Lower the receiver once
        let lowered_receiver = self.lower_expression(receiver);

        // Extract the return expression from the method body.
        // The body may be structured as:
        //   1. Direct: [Return { value: Some(expr) }]
        //   2. Wrapped: [Expression { Block { [Return { value: Some(expr) }] } }]
        // The second form is common when the parser wraps method bodies in a Block.
        let return_expr = if method.body.len() == 1 {
            match method.body.first() {
                Some(TypedStatement::Return {
                    value: Some(expr), ..
                }) => Some(expr),
                Some(TypedStatement::Expression { expression, .. }) => {
                    // Unwrap Expression(Block([Return(...)]))
                    if let TypedExpressionKind::Block { statements, .. } = &expression.kind {
                        if statements.len() == 1 {
                            if let TypedStatement::Return {
                                value: Some(expr), ..
                            } = &statements[0]
                            {
                                Some(expr)
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                }
                _ => None,
            }
        } else {
            None
        };

        if let Some(return_expr) = return_expr {
            let inlined = self.inline_expression_deep(
                return_expr,
                &lowered_receiver,
                &param_map,
                result_type,
            );
            return Some(inlined);
        }

        // For more complex methods, we'd need to:
        // 1. Create a block scope
        // 2. Lower all statements
        // 3. Replace `this` and parameter references
        // For now, fall back to regular method call
        // println!("DEBUG: Method body too complex to inline (has {} statements)", method.body.len());
        None
    }

    /// Deeply inline an expression, replacing `this` and parameters with concrete HIR expressions
    /// The `expected_type` parameter is used to fix type information for expressions like `new Abstract(...)`
    /// which may have incorrect type IDs in the TAST.
    fn inline_expression_deep(
        &mut self,
        expr: &TypedExpression,
        this_replacement: &HirExpr,
        param_map: &HashMap<SymbolId, HirExpr>,
        expected_type: TypeId,
    ) -> HirExpr {
        let kind_name = match &expr.kind {
            TypedExpressionKind::Literal { .. } => "Literal",
            TypedExpressionKind::Variable { .. } => "Variable",
            TypedExpressionKind::FieldAccess { .. } => "FieldAccess",
            TypedExpressionKind::MethodCall { .. } => "MethodCall",
            TypedExpressionKind::BinaryOp { .. } => "BinaryOp",
            TypedExpressionKind::UnaryOp { .. } => "UnaryOp",
            TypedExpressionKind::This { .. } => "This",
            TypedExpressionKind::New { .. } => "New",
            _ => "Other",
        };
        // println!("DEBUG inline_expression_deep: Processing {}", kind_name);
        match &expr.kind {
            // If it's a `this` reference, replace it with the receiver
            TypedExpressionKind::This { .. } => this_replacement.clone(),

            // If it's a variable reference, check if it's a parameter
            TypedExpressionKind::Variable { symbol_id, .. } => {
                if let Some(replacement) = param_map.get(symbol_id) {
                    // println!("DEBUG inline_expression_deep: Substituting variable {:?} with: {:?}", symbol_id, replacement);
                    replacement.clone()
                } else {
                    // println!("DEBUG inline_expression_deep: Variable {:?} not in param_map, lowering normally", symbol_id);
                    // Not a parameter, lower normally
                    self.lower_expression(expr)
                }
            }

            // For method calls on parameters, optimize identity methods like toInt()
            TypedExpressionKind::MethodCall {
                receiver: inner_receiver,
                method_symbol,
                type_arguments,
                arguments,
            } => {
                // println!("DEBUG inline_expression_deep: Found MethodCall on {:?}", method_symbol);

                // Special optimization: if receiver is a parameter and method is an identity method (toInt, etc.)
                // just return the substituted parameter value directly
                if let TypedExpressionKind::Variable { symbol_id } = &inner_receiver.kind {
                    if let Some(replacement) = param_map.get(symbol_id) {
                        // println!("DEBUG inline_expression_deep: Receiver is parameter {:?}", symbol_id);

                        // Check if this is an identity method
                        // First get the symbol, then get its name
                        if let Some(symbol) = self.symbol_table.get_symbol(*method_symbol) {
                            if let Some(method_name_str) = self.string_interner.get(symbol.name) {
                                // println!("DEBUG inline_expression_deep: Method name is {}", method_name_str);
                                if method_name_str == "toInt"
                                    || method_name_str == "toFloat"
                                    || method_name_str == "toString"
                                {
                                    // println!("DEBUG inline_expression_deep: Optimizing identity method - returning receiver");
                                    return replacement.clone();
                                }
                            }
                        }
                    }
                }

                // Otherwise, substitute receiver and arguments and create call
                // println!("DEBUG inline_expression_deep: Creating Call expression for method");
                let lowered_receiver = self.inline_expression_deep(
                    inner_receiver,
                    this_replacement,
                    param_map,
                    inner_receiver.expr_type,
                );
                let lowered_args: Vec<HirExpr> = arguments
                    .iter()
                    .map(|arg| {
                        self.inline_expression_deep(arg, this_replacement, param_map, arg.expr_type)
                    })
                    .collect();

                let mut call_args = vec![lowered_receiver];
                call_args.extend(lowered_args);

                HirExpr::new(
                    HirExprKind::Call {
                        callee: Box::new(HirExpr::new(
                            HirExprKind::Variable {
                                symbol: *method_symbol,
                                capture_mode: None,
                            },
                            expr.expr_type,
                            self.current_lifetime,
                            expr.source_location,
                        )),
                        type_args: type_arguments.clone(),
                        args: call_args,
                        is_method: true,
                    },
                    expr.expr_type,
                    self.current_lifetime,
                    expr.source_location,
                )
            }

            // For binary operations, recursively inline both operands
            TypedExpressionKind::BinaryOp {
                operator,
                left,
                right,
            } => {
                let lowered_left =
                    self.inline_expression_deep(left, this_replacement, param_map, left.expr_type);
                let lowered_right = self.inline_expression_deep(
                    right,
                    this_replacement,
                    param_map,
                    right.expr_type,
                );

                HirExpr::new(
                    HirExprKind::Binary {
                        op: self.convert_binary_op(operator),
                        lhs: Box::new(lowered_left),
                        rhs: Box::new(lowered_right),
                    },
                    expr.expr_type,
                    self.current_lifetime,
                    expr.source_location,
                )
            }

            // For unary operations, recursively inline the operand
            TypedExpressionKind::UnaryOp { operator, operand } => {
                let lowered_operand = self.inline_expression_deep(
                    operand,
                    this_replacement,
                    param_map,
                    operand.expr_type,
                );

                HirExpr::new(
                    HirExprKind::Unary {
                        op: self.convert_unary_op(operator),
                        operand: Box::new(lowered_operand),
                    },
                    expr.expr_type,
                    self.current_lifetime,
                    expr.source_location,
                )
            }

            // For New expressions, recursively inline constructor arguments
            TypedExpressionKind::New {
                class_type,
                type_arguments,
                arguments,
                class_name: tast_class_name,
            } => {
                let lowered_args: Vec<HirExpr> = arguments
                    .iter()
                    .map(|arg| {
                        self.inline_expression_deep(arg, this_replacement, param_map, expected_type)
                    })
                    .collect();

                // Fix the type information: if class_type is UNKNOWN/invalid, use expected_type
                let fixed_class_type = if *class_type == TypeId::invalid() {
                    // debug!(": Fixing New expression type from UNKNOWN to {:?}", expected_type);
                    expected_type
                } else {
                    *class_type
                };

                // Use class_name from TAST if available, otherwise extract from TypeId
                let class_name = tast_class_name.or_else(|| {
                    self.type_table
                        .borrow()
                        .get(fixed_class_type)
                        .and_then(|type_ref| {
                            if let crate::tast::TypeKind::Class { symbol_id, .. } = &type_ref.kind {
                                self.symbol_table
                                    .get_symbol(*symbol_id)
                                    .and_then(|sym| Some(sym.name))
                            } else {
                                None
                            }
                        })
                });

                HirExpr::new(
                    HirExprKind::New {
                        class_type: fixed_class_type,
                        type_args: type_arguments.clone(),
                        args: lowered_args,
                        class_name,
                    },
                    expected_type, // Use expected_type instead of expr.expr_type
                    self.current_lifetime,
                    expr.source_location,
                )
            }

            // For checked casts (from type annotations like `(rhs : Int)`),
            // recursively inline to substitute parameters. Only handle Checked
            // casts to avoid interfering with other cast kinds during compilation.
            TypedExpressionKind::Cast {
                expression,
                target_type,
                cast_kind: CastKind::Checked,
            } => {
                let lowered_inner = self.inline_expression_deep(
                    expression,
                    this_replacement,
                    param_map,
                    expression.expr_type,
                );
                HirExpr::new(
                    HirExprKind::Cast {
                        expr: Box::new(lowered_inner),
                        target: *target_type,
                        is_safe: true,
                    },
                    expr.expr_type,
                    self.current_lifetime,
                    expr.source_location,
                )
            }

            // For field access on parameters (e.g., `rhs.toInt()`),
            // recursively inline the object expression
            TypedExpressionKind::FieldAccess {
                object,
                field_symbol,
            } if matches!(&object.kind, TypedExpressionKind::Variable { symbol_id } if param_map.contains_key(symbol_id))
                || matches!(&object.kind, TypedExpressionKind::This { .. }) =>
            {
                let lowered_object = self.inline_expression_deep(
                    object,
                    this_replacement,
                    param_map,
                    object.expr_type,
                );
                HirExpr::new(
                    HirExprKind::Field {
                        object: Box::new(lowered_object),
                        field: *field_symbol,
                    },
                    expr.expr_type,
                    self.current_lifetime,
                    expr.source_location,
                )
            }

            // For other expressions, lower them normally
            _ => self.lower_expression(expr),
        }
    }

    /// Lower an expression to a literal pattern
    fn lower_expression_as_literal(&mut self, expr: &TypedExpression) -> HirLiteral {
        match &expr.kind {
            TypedExpressionKind::Literal { value } => self.lower_literal(value),
            _ => HirLiteral::Bool(false), // Default
        }
    }

    /// Validate that a constructor exists for the given class type and argument count
    fn validate_constructor(
        &mut self,
        class_type: TypeId,
        arg_count: usize,
        location: SourceLocation,
    ) {
        // Check if we have a current file being processed
        if let Some(file) = &self.current_file {
            // Look up the class type in the original TAST data
            for class in &file.classes {
                // Get the type symbol for this class
                if let Some(class_symbol_id) = self.type_table.borrow().get_type_symbol(class_type)
                {
                    // Check if this matches our class symbol
                    if class.symbol_id == class_symbol_id {
                        // Check if the class has a constructor
                        let has_constructor = !class.constructors.is_empty();

                        if !has_constructor {
                            // In Haxe, classes without explicit constructors have
                            // an implicit default (no-arg) constructor.
                            if arg_count > 0 {
                                let class_name_str = self
                                    .string_interner
                                    .get(class.name)
                                    .unwrap_or("?")
                                    .to_string();
                                let error_msg = format!(
                                    "Class '{}' has no constructor but 'new' was called with {} arguments",
                                    class_name_str,
                                    arg_count
                                );
                                self.add_error(&error_msg, location);
                            }
                            return;
                        }

                        // Basic validation passed
                        // Note: Enhanced validation features are tracked in BACKLOG.md
                        return;
                    }
                }
            }
        }

        // Class not found in current file - might be from imported module
        // Enhanced cross-module lookup tracked in BACKLOG.md
    }

    /// Compute captured variables for a lambda/closure
    ///
    /// Returns a list of variables that are:
    /// 1. Referenced in the lambda body
    /// 2. NOT lambda parameters
    /// 3. NOT defined locally within the lambda
    ///
    /// These are free variables that need to be captured from the enclosing scope.
    fn compute_captures(
        &self,
        body: &[TypedStatement],
        param_symbols: &std::collections::HashSet<SymbolId>,
    ) -> Vec<HirCapture> {
        use std::collections::{HashMap, HashSet};

        // Collect all variable references with their types
        let mut referenced_vars: HashMap<SymbolId, TypeId> = HashMap::new();
        for stmt in body {
            self.collect_var_refs_stmt(stmt, &mut referenced_vars);
        }

        // Remove parameters and local definitions
        let mut locally_defined = param_symbols.clone();
        for stmt in body {
            self.collect_local_defs_stmt(stmt, &mut locally_defined);
        }

        // Free variables are those referenced but not locally defined
        let captures: Vec<_> = referenced_vars
            .into_iter()
            .filter(|(sym, _)| !locally_defined.contains(sym))
            .map(|(symbol, ty)| HirCapture {
                symbol,
                mode: HirCaptureMode::ByValue, // Default to by-value capture
                ty,
            })
            .collect();

        captures
    }

    /// Collect all variable references in a statement
    fn collect_var_refs_stmt(
        &self,
        stmt: &TypedStatement,
        refs: &mut std::collections::HashMap<SymbolId, TypeId>,
    ) {
        match stmt {
            TypedStatement::Expression { expression, .. } => {
                self.collect_var_refs_expr(expression, refs);
            }
            TypedStatement::VarDeclaration { initializer, .. } => {
                if let Some(init) = initializer {
                    self.collect_var_refs_expr(init, refs);
                }
            }
            TypedStatement::Return { value, .. } => {
                if let Some(val) = value {
                    self.collect_var_refs_expr(val, refs);
                }
            }
            TypedStatement::If {
                condition,
                then_branch,
                else_branch,
                ..
            } => {
                self.collect_var_refs_expr(condition, refs);
                self.collect_var_refs_stmt(then_branch, refs);
                if let Some(else_stmt) = else_branch {
                    self.collect_var_refs_stmt(else_stmt, refs);
                }
            }
            TypedStatement::While {
                condition, body, ..
            } => {
                self.collect_var_refs_expr(condition, refs);
                self.collect_var_refs_stmt(body, refs);
            }
            _ => {} // Other statement types
        }
    }

    /// Collect all variable references in an expression
    fn collect_var_refs_expr(
        &self,
        expr: &TypedExpression,
        refs: &mut std::collections::HashMap<SymbolId, TypeId>,
    ) {
        match &expr.kind {
            TypedExpressionKind::Variable { symbol_id, .. } => {
                // Store the variable with its type from the expression
                refs.insert(*symbol_id, expr.expr_type);
            }
            TypedExpressionKind::FieldAccess { object, .. } => {
                // Field access like msg.value - need to capture the object (msg)
                self.collect_var_refs_expr(object, refs);
            }
            TypedExpressionKind::ArrayAccess { array, index, .. } => {
                self.collect_var_refs_expr(array, refs);
                self.collect_var_refs_expr(index, refs);
            }
            TypedExpressionKind::MethodCall {
                receiver,
                arguments,
                ..
            } => {
                self.collect_var_refs_expr(receiver, refs);
                for arg in arguments {
                    self.collect_var_refs_expr(arg, refs);
                }
            }
            TypedExpressionKind::BinaryOp { left, right, .. } => {
                self.collect_var_refs_expr(left, refs);
                self.collect_var_refs_expr(right, refs);
            }
            TypedExpressionKind::UnaryOp { operand, .. } => {
                self.collect_var_refs_expr(operand, refs);
            }
            TypedExpressionKind::FunctionCall {
                function,
                arguments,
                ..
            } => {
                self.collect_var_refs_expr(function, refs);
                for arg in arguments {
                    self.collect_var_refs_expr(arg, refs);
                }
            }
            TypedExpressionKind::Return { value, .. } => {
                if let Some(val) = value {
                    self.collect_var_refs_expr(val, refs);
                }
            }
            TypedExpressionKind::Block { statements, .. } => {
                for stmt in statements {
                    self.collect_var_refs_stmt(stmt, refs);
                }
            }
            _ => {} // Other expression types - add as needed
        }
    }

    /// Collect all locally defined variables in a statement
    fn collect_local_defs_stmt(
        &self,
        stmt: &TypedStatement,
        defs: &mut std::collections::HashSet<SymbolId>,
    ) {
        match stmt {
            TypedStatement::Expression { expression, .. } => {
                // FIX: Expression statements can contain Block expressions with local definitions
                self.collect_local_defs_expr(expression, defs);
            }
            TypedStatement::VarDeclaration { symbol_id, .. } => {
                defs.insert(*symbol_id);
            }
            TypedStatement::Block { statements, .. } => {
                // Recursively collect from all statements in the block
                for s in statements {
                    self.collect_local_defs_stmt(s, defs);
                }
            }
            TypedStatement::If {
                then_branch,
                else_branch,
                ..
            } => {
                self.collect_local_defs_stmt(then_branch, defs);
                if let Some(else_stmt) = else_branch {
                    self.collect_local_defs_stmt(else_stmt, defs);
                }
            }
            TypedStatement::While { body, .. } => {
                self.collect_local_defs_stmt(body, defs);
            }
            TypedStatement::For { init, body, .. } => {
                // For loops can have init declarations
                if let Some(init_stmt) = init {
                    self.collect_local_defs_stmt(init_stmt, defs);
                }
                self.collect_local_defs_stmt(body, defs);
            }
            TypedStatement::ForIn {
                value_var,
                key_var,
                body,
                ..
            } => {
                // For-in loops define iteration variables
                defs.insert(*value_var);
                if let Some(key) = key_var {
                    defs.insert(*key);
                }
                self.collect_local_defs_stmt(body, defs);
            }
            _ => {} // Other statement types
        }
    }

    /// Collect all locally defined variables in an expression (e.g., Block expressions)
    fn collect_local_defs_expr(
        &self,
        expr: &TypedExpression,
        defs: &mut std::collections::HashSet<SymbolId>,
    ) {
        match &expr.kind {
            TypedExpressionKind::Block { statements, .. } => {
                for stmt in statements {
                    self.collect_local_defs_stmt(stmt, defs);
                }
            }
            TypedExpressionKind::While { then_expr, .. } => {
                self.collect_local_defs_expr(then_expr, defs);
            }
            TypedExpressionKind::For { variable, body, .. } => {
                // For expressions define iteration variable
                defs.insert(*variable);
                self.collect_local_defs_expr(body, defs);
            }
            TypedExpressionKind::ForIn {
                value_var,
                key_var,
                body,
                ..
            } => {
                // For-in expressions define iteration variables
                defs.insert(*value_var);
                if let Some(key) = key_var {
                    defs.insert(*key);
                }
                self.collect_local_defs_expr(body, defs);
            }
            TypedExpressionKind::Conditional {
                then_expr,
                else_expr,
                ..
            } => {
                self.collect_local_defs_expr(then_expr, defs);
                if let Some(else_e) = else_expr {
                    self.collect_local_defs_expr(else_e, defs);
                }
            }
            _ => {} // Other expression types don't define local variables
        }
    }

    /// Copy parent class methods to child class's HIR method list
    /// This enables method inheritance - child instances can call parent methods
    /// Child methods should already be in the list so they override parent methods
    fn copy_parent_methods_to_hir(
        &mut self,
        child_methods: &mut Vec<HirMethod>,
        parent_type_id: TypeId,
    ) {
        // First, resolve the parent TypeId to get the parent class's SymbolId
        // parent_type_id might be an instance type, we need to find the declaration
        let parent_symbol = {
            let type_table = self.type_table.borrow();
            if let Some(type_info) = type_table.get(parent_type_id) {
                if let crate::tast::TypeKind::Class { symbol_id, .. } = &type_info.kind {
                    Some(*symbol_id)
                } else {
                    None
                }
            } else {
                None
            }
        };

        if let Some(parent_sym) = parent_symbol {
            // Find the parent class in module.types by matching symbol_id
            // We need to search because the TypeId used as key might be different from parent_type_id
            for (_type_id, type_decl) in &self.module.types {
                if let HirTypeDecl::Class(parent_class) = type_decl {
                    if parent_class.symbol_id == parent_sym {
                        // Found the parent class! Clone and add its methods
                        let parent_methods = parent_class.methods.clone();

                        // Add parent methods to the child's method list
                        // They go AFTER child methods, so child methods are found first (enabling overriding)
                        for parent_method in parent_methods {
                            child_methods.push(parent_method);
                        }
                        return;
                    }
                }
            }
        }
    }
}

/// Public entry point for TAST to HIR lowering
pub fn lower_tast_to_hir(
    file: &TypedFile,
    symbol_table: &SymbolTable,
    type_table: &Rc<RefCell<TypeTable>>,
    string_interner: &mut StringInterner,
    semantic_graphs: Option<&SemanticGraphs>,
) -> Result<HirModule, Vec<LoweringError>> {
    let mut context = TastToHirContext::new(
        symbol_table,
        type_table,
        string_interner,
        file.metadata
            .package_name
            .as_ref()
            .map(|n| n.to_string())
            .unwrap_or_else(|| "main".to_string()),
    );

    if let Some(graphs) = semantic_graphs {
        context.set_semantic_graphs(graphs);
    }

    context.lower_file(file)
}
