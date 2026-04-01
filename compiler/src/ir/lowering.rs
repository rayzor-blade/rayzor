//! TAST to HIR Lowering
//!
//! This module implements the lowering from Typed AST (TAST) to High-level IR (HIR).
//! The lowering process uses information from semantic analysis to generate optimized IR.

use super::{
    BinaryOp, CallingConvention, CompareOp, FunctionSignatureBuilder, InlineHint, IrBlockId,
    IrBuilder, IrId, IrModule, IrSourceLocation, IrType, IrValue, Linkage, UnaryOp,
};
use crate::semantic_graph::{
    analysis::analysis_engine::{AnalysisResults, HIRLoweringHints},
    SemanticGraphs,
};
use crate::tast::{
    node::{
        BinaryOperator, FunctionEffects, HasSourceLocation, LiteralValue, TypedCatchClause,
        TypedExpression, TypedExpressionKind, TypedFile, TypedFunction, TypedParameter,
        TypedStatement, TypedSwitchCase, UnaryOperator,
    },
    InternedString, SourceLocation, SymbolId, SymbolTable, Type, TypeId, TypeKind, TypeTable,
    Visibility,
};
use std::cell::RefCell;
use std::collections::BTreeMap;

/// Context for lowering TAST to HIR
pub struct LoweringContext<'a> {
    /// IR builder
    builder: IrBuilder,

    /// Symbol table from TAST
    symbol_table: &'a SymbolTable,

    /// Type table from TAST
    type_table: &'a RefCell<TypeTable>,

    /// Semantic analysis results
    analysis_results: Option<&'a AnalysisResults>,

    /// HIR lowering hints from analysis
    lowering_hints: Option<&'a HIRLoweringHints>,

    /// Mapping from TAST symbols to HIR registers
    symbol_map: BTreeMap<SymbolId, IrId>,

    /// Mapping from TAST types to HIR types
    type_cache: BTreeMap<TypeId, IrType>,

    /// Current loop context (for break/continue)
    loop_stack: Vec<LoopContext>,

    /// Error accumulator
    errors: Vec<LoweringError>,
}

/// Loop context for break/continue lowering
#[derive(Debug, Clone)]
struct LoopContext {
    /// Block to jump to for continue
    continue_block: IrBlockId,

    /// Block to jump to for break
    break_block: IrBlockId,
}

/// Lowering error
#[derive(Debug)]
pub struct LoweringError {
    pub message: String,
    pub location: SourceLocation,
}

impl<'a> LoweringContext<'a> {
    /// Create a new lowering context
    pub fn new(
        module_name: String,
        source_file: String,
        symbol_table: &'a SymbolTable,
        type_table: &'a RefCell<TypeTable>,
    ) -> Self {
        Self {
            builder: IrBuilder::new(module_name, source_file),
            symbol_table,
            type_table,
            analysis_results: None,
            lowering_hints: None,
            symbol_map: BTreeMap::new(),
            type_cache: BTreeMap::new(),
            loop_stack: Vec::new(),
            errors: Vec::new(),
        }
    }

    /// Set analysis results for optimization
    pub fn set_analysis_results(&mut self, results: &'a AnalysisResults) {
        self.analysis_results = Some(results);
    }

    /// Set HIR lowering hints
    pub fn set_lowering_hints(&mut self, hints: &'a HIRLoweringHints) {
        self.lowering_hints = Some(hints);
    }

    /// Register a symbol-to-register mapping for memory safety validation
    fn register_symbol_mapping(&mut self, symbol_id: SymbolId, register: IrId) {
        // Add to local symbol map
        self.symbol_map.insert(symbol_id, register);

        // Add to module-level mapping for MirSafetyValidator
        self.builder
            .module
            .symbol_to_register
            .insert(symbol_id, register);
        self.builder
            .module
            .register_to_symbol
            .insert(register, symbol_id);
    }

    /// Lower a typed file to HIR module
    pub fn lower_file(&mut self, file: &TypedFile) -> Result<IrModule, Vec<LoweringError>> {
        // Set module metadata
        self.builder.module.metadata.language_version = "1.0".to_string();

        // Lower all functions
        for function in &file.functions {
            self.lower_function(function);
        }

        // TODO: Lower global variables, type definitions, etc.

        if self.errors.is_empty() {
            Ok(std::mem::replace(
                &mut self.builder.module,
                IrModule::new(String::new(), String::new()),
            ))
        } else {
            Err(std::mem::take(&mut self.errors))
        }
    }

    /// Lower a function to HIR
    fn lower_function(&mut self, function: &TypedFunction) {
        // Clear per-function state
        self.symbol_map.clear();

        // Convert source location
        let source_loc = self.convert_source_location(&function.source_location);
        self.builder.set_source_location(source_loc);

        // Build function signature
        let signature = self.build_function_signature(function);

        // Get function name (may need mangling)
        let name = self.get_function_name(function);

        // Start building the function
        let func_id = self
            .builder
            .start_function(function.symbol_id, name, signature);

        // Map parameters to their registers
        for (i, param) in function.parameters.iter().enumerate() {
            if let Some(reg) = self
                .builder
                .current_function()
                .and_then(|f| f.get_param_reg(i))
            {
                self.register_symbol_mapping(param.symbol_id, reg);
            }
        }

        // Check if this function should be inlined
        if let Some(hints) = self.lowering_hints {
            if hints.inlinable_functions.contains(&function.symbol_id) {
                if let Some(func) = self.builder.module.functions.get_mut(&func_id) {
                    func.attributes.inline = super::InlineHint::Hint;
                }
            }
        }

        // Lower function body
        for statement in &function.body {
            self.lower_statement(statement);
        }

        // Add implicit return if needed
        if let Some(current_block) = self.builder.current_block() {
            if let Some(func) = self.builder.current_function() {
                if let Some(block) = func.cfg.get_block(current_block) {
                    if !block.is_terminated() {
                        // Add return based on function return type
                        let ret_type = &func.signature.return_type;
                        if *ret_type == IrType::Void {
                            self.builder.build_return(None);
                        } else {
                            // Return default value
                            let default_val = self.builder.build_const(ret_type.default_value());
                            self.builder.build_return(default_val);
                        }
                    }
                }
            }
        }

        self.builder.finish_function();
    }

    /// Build function signature from TAST
    fn build_function_signature(&mut self, function: &TypedFunction) -> super::IrFunctionSignature {
        let mut sig_builder = FunctionSignatureBuilder::new();

        // Add parameters
        for param in &function.parameters {
            let param_type = self.lower_type(param.param_type);
            sig_builder = sig_builder.param(param.name.to_string(), param_type);
        }

        // Set return type
        let return_type = self.lower_type(function.return_type);
        sig_builder = sig_builder.returns(return_type);

        // TODO: Determine if function can throw
        sig_builder = sig_builder.can_throw(false);

        sig_builder.build()
    }

    /// Get mangled function name
    fn get_function_name(&self, function: &TypedFunction) -> String {
        // TODO: Implement proper name mangling
        function.name.to_string()
    }

    /// Lower a statement
    fn lower_statement(&mut self, statement: &TypedStatement) {
        let source_loc = self.convert_source_location(&statement.source_location());
        self.builder.set_source_location(source_loc);

        match statement {
            TypedStatement::Expression { expression, .. } => {
                // Evaluate expression for side effects
                self.lower_expression(expression);
            }

            TypedStatement::VarDeclaration {
                symbol_id,
                initializer,
                ..
            } => {
                // Get variable type from symbol table
                let var_type = if let Some(symbol) = self.symbol_table.get_symbol(*symbol_id) {
                    self.lower_type(symbol.type_id)
                } else {
                    IrType::Any
                };

                // Allocate storage for the variable
                let var_reg = self
                    .builder
                    .declare_local(format!("var_{}", symbol_id.as_raw()), var_type.clone());

                if let Some(var_reg) = var_reg {
                    // Map the symbol to its register
                    self.register_symbol_mapping(*symbol_id, var_reg);

                    // If there's an initializer, evaluate it and store
                    if let Some(init_expr) = initializer {
                        if let Some(init_value) = self.lower_expression(init_expr) {
                            // For now, just copy the value
                            // In a more complete implementation, we might need to handle
                            // different storage strategies based on the type
                            self.builder.build_copy(init_value);
                        }
                    }
                }
            }

            TypedStatement::Assignment { target, value, .. } => {
                self.lower_assignment(target, value);
            }

            TypedStatement::Return { value, .. } => {
                let value = value.as_ref().map(|e| self.lower_expression(e)).flatten();
                self.builder.build_return(value);
            }

            TypedStatement::If {
                condition,
                then_branch,
                else_branch,
                ..
            } => {
                self.lower_if_statement(condition, then_branch, else_branch.as_ref().map(|s| &**s));
            }

            TypedStatement::While {
                condition, body, ..
            } => {
                self.lower_while_loop(condition, body);
            }

            TypedStatement::For {
                init,
                condition,
                update,
                body,
                ..
            } => {
                self.lower_for_loop(
                    init.as_ref().map(|s| &**s),
                    condition.as_ref(),
                    update.as_ref(),
                    body,
                );
            }

            TypedStatement::Break { .. } => {
                if let Some(loop_ctx) = self.loop_stack.last() {
                    self.builder.build_branch(loop_ctx.break_block);
                } else {
                    self.add_error("Break outside of loop", statement.source_location());
                }
            }

            TypedStatement::Continue { .. } => {
                if let Some(loop_ctx) = self.loop_stack.last() {
                    self.builder.build_branch(loop_ctx.continue_block);
                } else {
                    self.add_error("Continue outside of loop", statement.source_location());
                }
            }

            TypedStatement::Block { statements, .. } => {
                for stmt in statements {
                    self.lower_statement(stmt);
                }
            }

            TypedStatement::Switch {
                discriminant,
                cases,
                default_case,
                ..
            } => {
                // Convert default_case from Option<Box<TypedStatement>> to Option<&[TypedStatement]>
                let default_stmts = default_case
                    .as_ref()
                    .map(|stmt| std::slice::from_ref(&**stmt));
                self.lower_switch_statement(discriminant, cases, default_stmts);
            }

            TypedStatement::Try {
                body,
                catch_clauses,
                finally_block,
                ..
            } => {
                // TODO: Implement proper try statement lowering
                self.add_error(
                    "Try statement lowering not implemented",
                    statement.source_location(),
                );
            }

            TypedStatement::Throw { exception, .. } => {
                let _exception = self.lower_expression(&exception);
                // TODO: Implement throw instruction properly
                // For now, just generate unreachable
                self.builder.build_unreachable();
            }

            _ => {
                self.add_error(
                    format!("Unimplemented statement kind: {:?}", statement),
                    statement.source_location(),
                );
            }
        }
    }

    /// Lower an expression
    fn lower_expression(&mut self, expr: &TypedExpression) -> Option<IrId> {
        let source_loc = self.convert_source_location(&expr.source_location);
        self.builder.set_source_location(source_loc);

        use crate::tast::node::TypedExpressionKind as EK;

        match &expr.kind {
            EK::Literal { value } if matches!(value, LiteralValue::Int(_)) => {
                let ty = self.lower_type(expr.expr_type);
                if let LiteralValue::Int(int_val) = value {
                    self.builder.build_int(*int_val, ty)
                } else {
                    None
                }
            }

            EK::Literal { value } if matches!(value, LiteralValue::Float(_)) => {
                if let LiteralValue::Float(float_val) = value {
                    let ir_value = match self.lower_type(expr.expr_type) {
                        IrType::F32 => IrValue::F32(*float_val as f32),
                        IrType::F64 => IrValue::F64(*float_val),
                        _ => IrValue::F64(*float_val),
                    };
                    self.builder.build_const(ir_value)
                } else {
                    None
                }
            }

            EK::Literal { value } if matches!(value, LiteralValue::Bool(_)) => {
                if let LiteralValue::Bool(bool_val) = value {
                    self.builder.build_bool(*bool_val)
                } else {
                    None
                }
            }

            EK::Literal { value } if matches!(value, LiteralValue::String(_)) => {
                if let LiteralValue::String(string_val) = value {
                    self.builder.build_string(string_val.to_string())
                } else {
                    None
                }
            }

            EK::Null => self.builder.build_null(),

            EK::Variable { symbol_id } => self.symbol_map.get(symbol_id).copied(),

            EK::BinaryOp {
                left,
                operator,
                right,
            } => self.lower_binary_op(operator, left, right),

            EK::UnaryOp { operator, operand } => self.lower_unary_op(operator, operand),

            // Note: Assignment is typically a statement, not an expression
            // If we need assignment expressions, add proper support
            // EK::Assignment { target, value } => {
            //     self.lower_assignment(target, value)
            // }
            EK::FunctionCall {
                function,
                arguments,
                ..
            } => self.lower_call(function, arguments, expr.expr_type),

            EK::FieldAccess {
                object,
                field_symbol,
                ..
            } => {
                // Lower the object expression
                let obj_reg = self.lower_expression(object)?;

                // Get the field index from the symbol
                let field_index = self.get_field_index_from_symbol(*field_symbol)?;

                // Calculate field pointer using GEP
                let field_type = self.lower_type(expr.expr_type);
                let zero_idx = self.builder.build_int(0, IrType::I64)?;
                let field_idx = self.builder.build_int(field_index as i64, IrType::I64)?;
                let field_ptr = self.builder.build_gep(
                    obj_reg,
                    vec![zero_idx, field_idx],
                    field_type.clone(),
                )?;

                // Load the field value
                self.builder.build_load(field_ptr, field_type)
            }

            EK::ArrayAccess { array, index } => self.lower_array_access(array, index),

            EK::ArrayLiteral { elements } => self.lower_array_literal(elements, expr.expr_type),

            EK::ObjectLiteral { fields, .. } => {
                // Convert TypedObjectField to expected format
                let converted_fields: Vec<(String, TypedExpression)> = fields
                    .iter()
                    .map(|f| (f.name.to_string(), f.value.clone()))
                    .collect();
                self.lower_object_literal(&converted_fields, expr.expr_type)
            }

            EK::Cast {
                expression,
                target_type,
                ..
            } => self.lower_cast(expression, *target_type),

            EK::Conditional {
                condition,
                then_expr,
                else_expr,
            } => self.lower_conditional(condition, then_expr, else_expr.as_ref().map(|e| &**e)),

            EK::Block {
                statements,
                scope_id,
            } => {
                // Lower all statements in the block
                for stmt in statements {
                    self.lower_statement(stmt);
                }
                // Blocks used in conditionals return void
                use crate::ir::types::IrValue;
                self.builder.build_const(IrValue::Void)
            }

            _ => {
                self.add_error(
                    format!("Unimplemented expression kind: {:?}", expr.kind),
                    expr.source_location.clone(),
                );
                None
            }
        }
    }

    /// Lower a binary operation
    fn lower_binary_op(
        &mut self,
        op: &BinaryOperator,
        left: &TypedExpression,
        right: &TypedExpression,
    ) -> Option<IrId> {
        // Use BinaryOperator enum

        // Handle short-circuit operators specially
        match op {
            BinaryOperator::And => return self.lower_logical_and(left, right),
            BinaryOperator::Or => return self.lower_logical_or(left, right),
            _ => {}
        }

        let left_val = self.lower_expression(left)?;
        let right_val = self.lower_expression(right)?;

        let is_float = self.lower_type(left.expr_type).is_float();

        match op {
            BinaryOperator::Add => self.builder.build_add(left_val, right_val, is_float),
            BinaryOperator::Sub => self.builder.build_sub(left_val, right_val, is_float),
            BinaryOperator::Mul => self.builder.build_mul(left_val, right_val, is_float),
            BinaryOperator::Div => self.builder.build_div(left_val, right_val, is_float),
            BinaryOperator::Mod => {
                let op = if is_float {
                    BinaryOp::FRem
                } else {
                    BinaryOp::Rem
                };
                self.builder.build_binop(op, left_val, right_val)
            }
            BinaryOperator::Eq => {
                let op = if is_float {
                    CompareOp::FEq
                } else {
                    CompareOp::Eq
                };
                self.builder.build_cmp(op, left_val, right_val)
            }
            BinaryOperator::Ne => {
                let op = if is_float {
                    CompareOp::FNe
                } else {
                    CompareOp::Ne
                };
                self.builder.build_cmp(op, left_val, right_val)
            }
            BinaryOperator::Lt => {
                let op = if is_float {
                    CompareOp::FLt
                } else {
                    CompareOp::Lt
                };
                self.builder.build_cmp(op, left_val, right_val)
            }
            BinaryOperator::Le => {
                let op = if is_float {
                    CompareOp::FLe
                } else {
                    CompareOp::Le
                };
                self.builder.build_cmp(op, left_val, right_val)
            }
            BinaryOperator::Gt => {
                let op = if is_float {
                    CompareOp::FGt
                } else {
                    CompareOp::Gt
                };
                self.builder.build_cmp(op, left_val, right_val)
            }
            BinaryOperator::Ge => {
                let op = if is_float {
                    CompareOp::FGe
                } else {
                    CompareOp::Ge
                };
                self.builder.build_cmp(op, left_val, right_val)
            }
            BinaryOperator::BitAnd => self.builder.build_binop(BinaryOp::And, left_val, right_val),
            BinaryOperator::BitOr => self.builder.build_binop(BinaryOp::Or, left_val, right_val),
            BinaryOperator::BitXor => self.builder.build_binop(BinaryOp::Xor, left_val, right_val),
            BinaryOperator::Shl => self.builder.build_binop(BinaryOp::Shl, left_val, right_val),
            BinaryOperator::Shr => self.builder.build_binop(BinaryOp::Shr, left_val, right_val),
            _ => {
                self.add_error(
                    format!("Unimplemented binary operator: {:?}", op),
                    left.source_location,
                );
                None
            }
        }
    }

    /// Lower logical AND with short-circuit evaluation
    fn lower_logical_and(
        &mut self,
        left: &TypedExpression,
        right: &TypedExpression,
    ) -> Option<IrId> {
        let right_block = self.builder.create_block()?;
        let merge_block = self.builder.create_block()?;

        // Evaluate left side
        let left_val = self.lower_expression(left)?;

        // Branch based on left value
        self.builder
            .build_cond_branch(left_val, right_block, merge_block)?;

        // Right block: evaluate right side
        self.builder.switch_to_block(right_block);
        let right_val = self.lower_expression(right)?;
        self.builder.build_branch(merge_block)?;

        // Merge block: phi node
        self.builder.switch_to_block(merge_block);
        let phi = self.builder.build_phi(merge_block, IrType::Bool)?;

        // Add incoming values
        let current_block = self.builder.current_block()?;
        let false_val = self.builder.build_bool(false)?;
        self.builder
            .add_phi_incoming(merge_block, phi, current_block, false_val)?;
        self.builder
            .add_phi_incoming(merge_block, phi, right_block, right_val)?;

        Some(phi)
    }

    /// Lower logical OR with short-circuit evaluation
    fn lower_logical_or(
        &mut self,
        left: &TypedExpression,
        right: &TypedExpression,
    ) -> Option<IrId> {
        let right_block = self.builder.create_block()?;
        let merge_block = self.builder.create_block()?;

        // Evaluate left side
        let left_val = self.lower_expression(left)?;

        // Branch based on left value
        self.builder
            .build_cond_branch(left_val, merge_block, right_block)?;

        // Right block: evaluate right side
        self.builder.switch_to_block(right_block);
        let right_val = self.lower_expression(right)?;
        self.builder.build_branch(merge_block)?;

        // Merge block: phi node
        self.builder.switch_to_block(merge_block);
        let phi = self.builder.build_phi(merge_block, IrType::Bool)?;

        // Add incoming values
        let current_block = self.builder.current_block()?;
        let true_val = self.builder.build_bool(true)?;
        self.builder
            .add_phi_incoming(merge_block, phi, current_block, true_val)?;
        self.builder
            .add_phi_incoming(merge_block, phi, right_block, right_val)?;

        Some(phi)
    }

    /// Lower type from TAST to HIR
    fn lower_type(&mut self, type_id: TypeId) -> IrType {
        // Check cache first
        if let Some(cached) = self.type_cache.get(&type_id) {
            return cached.clone();
        }

        let ty = self.type_table.borrow().get(type_id).cloned();
        let ir_type = match ty {
            Some(Type { kind, .. }) => self.convert_type_kind(&kind),
            None => IrType::Any, // Fallback for unknown types
        };

        self.type_cache.insert(type_id, ir_type.clone());
        ir_type
    }

    /// Convert TAST type kind to HIR type
    fn convert_type_kind(&mut self, kind: &TypeKind) -> IrType {
        match kind {
            TypeKind::Void => IrType::Void,
            TypeKind::Int => IrType::I32,   // Default int size
            TypeKind::Float => IrType::F64, // Default float size
            TypeKind::Bool => IrType::Bool,
            TypeKind::String => IrType::String,
            TypeKind::Array { element_type } => {
                IrType::Slice(Box::new(self.lower_type(*element_type)))
            }
            TypeKind::Class { .. } | TypeKind::Interface { .. } => {
                // Classes and interfaces are represented as pointers
                IrType::Ptr(Box::new(IrType::Opaque {
                    name: "object".to_string(),
                    size: std::mem::size_of::<usize>(),
                    align: std::mem::align_of::<usize>(),
                }))
            }
            TypeKind::Function {
                params,
                return_type,
                ..
            } => {
                IrType::Function {
                    params: params.iter().map(|&ty| self.lower_type(ty)).collect(),
                    return_type: Box::new(self.lower_type(*return_type)),
                    varargs: false, // TODO: Handle varargs
                }
            }
            TypeKind::Dynamic => IrType::Any,
            TypeKind::Unknown => IrType::Ptr(Box::new(IrType::Void)), // Null values
            _ => IrType::Any,                                         // Fallback
        }
    }

    /// Convert source location
    fn convert_source_location(&self, loc: &SourceLocation) -> IrSourceLocation {
        IrSourceLocation {
            file_id: loc.file_id,
            line: loc.line,
            column: loc.column,
        }
    }

    /// Add an error
    fn add_error(&mut self, message: impl Into<String>, location: SourceLocation) {
        self.errors.push(LoweringError {
            message: message.into(),
            location,
        });
    }

    // Stub implementations for remaining methods

    fn lower_unary_op(&mut self, op: &UnaryOperator, operand: &TypedExpression) -> Option<IrId> {
        let operand_val = self.lower_expression(operand)?;

        // Use UnaryOperator enum
        match op {
            UnaryOperator::Neg => self.builder.build_unop(UnaryOp::Neg, operand_val),
            UnaryOperator::Not => self.builder.build_unop(UnaryOp::Not, operand_val),
            UnaryOperator::BitNot => self.builder.build_unop(UnaryOp::Not, operand_val),
            _ => {
                self.add_error(
                    format!("Unimplemented unary operator: {:?}", op),
                    operand.source_location,
                );
                None
            }
        }
    }

    fn lower_variable_declaration(
        &mut self,
        _pattern: &crate::tast::node::TypedPattern,
        _value: Option<&TypedExpression>,
    ) {
        // TODO: Implement pattern matching and variable declaration
        self.add_error(
            "Variable declaration lowering not implemented",
            SourceLocation::unknown(),
        );
    }

    fn lower_if_statement(
        &mut self,
        condition: &TypedExpression,
        then_stmt: &TypedStatement,
        else_stmt: Option<&TypedStatement>,
    ) {
        // Create blocks for the if statement
        let then_block = match self.builder.create_block() {
            Some(block) => block,
            None => return,
        };
        let merge_block = match self.builder.create_block() {
            Some(block) => block,
            None => return,
        };

        let else_block = if else_stmt.is_some() {
            match self.builder.create_block() {
                Some(block) => block,
                None => return,
            }
        } else {
            merge_block
        };

        // Evaluate condition
        if let Some(cond_val) = self.lower_expression(condition) {
            // Branch based on condition
            self.builder
                .build_cond_branch(cond_val, then_block, else_block);

            // Lower then branch
            self.builder.switch_to_block(then_block);
            self.lower_statement(then_stmt);
            // Jump to merge block if not already terminated
            if let Some(current_block) = self.builder.current_block() {
                if let Some(func) = self.builder.current_function() {
                    if let Some(block) = func.cfg.get_block(current_block) {
                        if !block.is_terminated() {
                            self.builder.build_branch(merge_block);
                        }
                    }
                }
            }

            // Lower else branch if present
            if let Some(else_stmt) = else_stmt {
                self.builder.switch_to_block(else_block);
                self.lower_statement(else_stmt);
                // Jump to merge block if not already terminated
                if let Some(current_block) = self.builder.current_block() {
                    if let Some(func) = self.builder.current_function() {
                        if let Some(block) = func.cfg.get_block(current_block) {
                            if !block.is_terminated() {
                                self.builder.build_branch(merge_block);
                            }
                        }
                    }
                }
            }

            // Continue in merge block
            self.builder.switch_to_block(merge_block);
        }
    }

    fn lower_while_loop(&mut self, condition: &TypedExpression, body: &TypedStatement) {
        // Create blocks for the while loop
        let cond_block = match self.builder.create_block() {
            Some(block) => block,
            None => return,
        };
        let body_block = match self.builder.create_block() {
            Some(block) => block,
            None => return,
        };
        let exit_block = match self.builder.create_block() {
            Some(block) => block,
            None => return,
        };

        // Jump to condition block
        self.builder.build_branch(cond_block);

        // Condition block
        self.builder.switch_to_block(cond_block);
        if let Some(cond_val) = self.lower_expression(condition) {
            self.builder
                .build_cond_branch(cond_val, body_block, exit_block);
        }

        // Body block
        self.builder.switch_to_block(body_block);

        // Push loop context for break/continue
        self.loop_stack.push(LoopContext {
            continue_block: cond_block,
            break_block: exit_block,
        });

        // Lower loop body
        self.lower_statement(body);

        // Pop loop context
        self.loop_stack.pop();

        // Jump back to condition check
        if let Some(current_block) = self.builder.current_block() {
            if let Some(func) = self.builder.current_function() {
                if let Some(block) = func.cfg.get_block(current_block) {
                    if !block.is_terminated() {
                        self.builder.build_branch(cond_block);
                    }
                }
            }
        }

        // Continue in exit block
        self.builder.switch_to_block(exit_block);
    }

    fn lower_for_loop(
        &mut self,
        init: Option<&TypedStatement>,
        condition: Option<&TypedExpression>,
        update: Option<&TypedExpression>,
        body: &TypedStatement,
    ) {
        // Lower initialization statement if present
        if let Some(init_stmt) = init {
            self.lower_statement(init_stmt);
        }

        // Create blocks for the for loop
        let cond_block = match self.builder.create_block() {
            Some(block) => block,
            None => return,
        };
        let body_block = match self.builder.create_block() {
            Some(block) => block,
            None => return,
        };
        let update_block = match self.builder.create_block() {
            Some(block) => block,
            None => return,
        };
        let exit_block = match self.builder.create_block() {
            Some(block) => block,
            None => return,
        };

        // Jump to condition block
        self.builder.build_branch(cond_block);

        // Condition block
        self.builder.switch_to_block(cond_block);
        if let Some(cond_expr) = condition {
            if let Some(cond_val) = self.lower_expression(cond_expr) {
                self.builder
                    .build_cond_branch(cond_val, body_block, exit_block);
            }
        } else {
            // No condition means infinite loop
            self.builder.build_branch(body_block);
        }

        // Body block
        self.builder.switch_to_block(body_block);

        // Push loop context for break/continue
        self.loop_stack.push(LoopContext {
            continue_block: update_block,
            break_block: exit_block,
        });

        // Lower loop body
        self.lower_statement(body);

        // Pop loop context
        self.loop_stack.pop();

        // Jump to update block
        if let Some(current_block) = self.builder.current_block() {
            if let Some(func) = self.builder.current_function() {
                if let Some(block) = func.cfg.get_block(current_block) {
                    if !block.is_terminated() {
                        self.builder.build_branch(update_block);
                    }
                }
            }
        }

        // Update block
        self.builder.switch_to_block(update_block);
        if let Some(update_expr) = update {
            self.lower_expression(update_expr);
        }
        // Jump back to condition
        self.builder.build_branch(cond_block);

        // Continue in exit block
        self.builder.switch_to_block(exit_block);
    }

    fn lower_switch_statement(
        &mut self,
        _value: &TypedExpression,
        _cases: &[crate::tast::node::TypedSwitchCase],
        _default: Option<&[TypedStatement]>,
    ) {
        // TODO: Implement switch statement lowering
        self.add_error(
            "Switch statement lowering not implemented",
            SourceLocation::unknown(),
        );
    }

    fn lower_try_statement(
        &mut self,
        body: &[TypedStatement],
        catches: &[crate::tast::node::TypedCatchClause],
        finally: Option<&[TypedStatement]>,
    ) {
        // Create landing pad for exception handling
        let Some(try_bb) = self.builder.create_block_with_label("try".to_string()) else {
            return;
        };
        let Some(catch_bb) = self.builder.create_block_with_label("catch".to_string()) else {
            return;
        };
        let Some(finally_bb) = self.builder.create_block_with_label("finally".to_string()) else {
            return;
        };
        let Some(continue_bb) = self.builder.create_block_with_label("continue".to_string()) else {
            return;
        };

        // Jump to try block
        self.builder.build_branch(try_bb);
        self.builder.switch_to_block(try_bb);

        // Lower try block with exception handling
        for stmt in body {
            self.lower_statement(stmt);
        }

        // Jump to finally block if no exception
        self.builder.build_branch(finally_bb);

        // Lower catch blocks
        self.builder.switch_to_block(catch_bb);
        for catch in catches {
            // Create a block for this catch clause
            let Some(this_catch_bb) = self
                .builder
                .create_block_with_label("catch_clause".to_string())
            else {
                continue;
            };
            self.builder.switch_to_block(this_catch_bb);

            // Bind the exception variable
            let exception_type = self.lower_type(catch.exception_type);
            if let Some(exception_ptr) = self.builder.build_alloc(exception_type, None) {
                self.register_symbol_mapping(catch.exception_variable, exception_ptr);
            }

            // Lower catch block statements
            self.lower_statement(&catch.body);

            // Jump to finally block
            self.builder.build_branch(finally_bb);
        }

        // Lower finally block if present
        self.builder.switch_to_block(finally_bb);
        if let Some(finally_stmts) = finally {
            for stmt in finally_stmts {
                self.lower_statement(stmt);
            }
        }

        // Continue with normal execution
        self.builder.build_branch(continue_bb);
        self.builder.switch_to_block(continue_bb);
    }

    fn lower_assignment(
        &mut self,
        target: &TypedExpression,
        value: &TypedExpression,
    ) -> Option<IrId> {
        // Evaluate the value expression
        let value_reg = self.lower_expression(value)?;

        // Handle different assignment targets
        match &target.kind {
            TypedExpressionKind::Variable { symbol_id } => {
                // Simple variable assignment
                if let Some(&var_reg) = self.symbol_map.get(symbol_id) {
                    // Copy the value to the variable's register
                    self.builder.build_copy(value_reg)
                } else {
                    self.add_error(
                        format!("Undefined variable in assignment: {:?}", symbol_id),
                        target.source_location,
                    );
                    None
                }
            }
            TypedExpressionKind::FieldAccess {
                object,
                field_symbol,
                ..
            } => {
                // Field assignment: obj.field = value
                if let Some(obj_reg) = self.lower_expression(object) {
                    // TODO: Implement proper field offset calculation
                    // For now, we'll use a simplified approach
                    self.add_error(
                        "Field assignment not yet fully implemented",
                        target.source_location,
                    );
                    None
                } else {
                    None
                }
            }
            TypedExpressionKind::ArrayAccess { array, index } => {
                // Array element assignment: arr[index] = value
                let array_reg = self.lower_expression(array)?;
                let index_reg = self.lower_expression(index)?;

                // Calculate element pointer using GEP
                let target_type = self.lower_type(target.expr_type);
                let elem_ptr = self
                    .builder
                    .build_gep(array_reg, vec![index_reg], target_type)?;

                // Store value at the calculated address
                self.builder.build_store(elem_ptr, value_reg);

                Some(value_reg)
            }
            _ => {
                self.add_error(
                    format!("Invalid assignment target: {:?}", target.kind),
                    target.source_location,
                );
                None
            }
        }
    }

    fn lower_call(
        &mut self,
        callee: &TypedExpression,
        arguments: &[TypedExpression],
        result_type: TypeId,
    ) -> Option<IrId> {
        // Lower the callee expression to get the function pointer
        let func_ptr = self.lower_expression(callee)?;

        // Lower all arguments
        let mut arg_regs = Vec::new();
        for arg in arguments {
            if let Some(arg_reg) = self.lower_expression(arg) {
                arg_regs.push(arg_reg);
            } else {
                self.add_error("Failed to lower function argument", arg.source_location);
                return None;
            }
        }

        // Get the result type
        let ret_type = self.lower_type(result_type);

        // Build the indirect call instruction (legacy lowering doesn't distinguish direct/indirect)
        self.builder
            .build_call_indirect(func_ptr, arg_regs, ret_type)
    }

    /// Get field index from a field symbol ID
    fn get_field_index_from_symbol(&self, field_symbol: SymbolId) -> Option<usize> {
        // Look up the field symbol in the symbol table
        if let Some(symbol) = self.symbol_table.get_symbol(field_symbol) {
            // Extract field index from symbol metadata
            // For now, use a simple index based on symbol ID
            // In production, this would look up the actual field index from type info
            Some(field_symbol.as_raw() as usize % 10) // Simplified for now
        } else {
            None
        }
    }

    fn lower_member_access(&mut self, object: &TypedExpression, member: &str) -> Option<IrId> {
        // Lower the object expression
        let obj_reg = self.lower_expression(object)?;

        // Find field index by name
        // TODO: Properly extract field index from type table
        let field_index = 0usize; // Simplified for now

        // Calculate field pointer using GEP
        let field_type = self.lower_type(object.expr_type);
        let zero_idx = self.builder.build_int(0, IrType::I64)?;
        let field_idx = self.builder.build_int(field_index as i64, IrType::I64)?;
        let field_ptr =
            self.builder
                .build_gep(obj_reg, vec![zero_idx, field_idx], field_type.clone())?;

        // Load the field value
        self.builder.build_load(field_ptr, field_type)
    }

    fn lower_array_access(
        &mut self,
        array: &TypedExpression,
        index: &TypedExpression,
    ) -> Option<IrId> {
        // Lower array and index expressions
        let array_reg = self.lower_expression(array)?;
        let index_reg = self.lower_expression(index)?;

        // Get element type
        let elem_type = match self.lower_type(array.expr_type) {
            IrType::Array(elem, _) => *elem,
            IrType::Slice(elem) => *elem,
            _ => {
                self.add_error("Array access on non-array type", array.source_location);
                return None;
            }
        };

        // Calculate element pointer
        let elem_ptr = self
            .builder
            .build_gep(array_reg, vec![index_reg], elem_type.clone())?;

        // Load the element
        self.builder.build_load(elem_ptr, elem_type)
    }

    fn lower_array_literal(
        &mut self,
        elements: &[TypedExpression],
        array_type: TypeId,
    ) -> Option<IrId> {
        // Get array type information
        let arr_type = self.lower_type(array_type);
        let (elem_type, _) = match &arr_type {
            IrType::Array(elem, size) => (elem.as_ref().clone(), Some(*size)),
            IrType::Slice(elem) => (elem.as_ref().clone(), None),
            _ => {
                self.add_error(
                    "Array literal with non-array type",
                    SourceLocation::unknown(),
                );
                return None;
            }
        };

        // Allocate array
        let count = self.builder.build_int(elements.len() as i64, IrType::I32)?;
        let array_ptr = self.builder.build_alloc(elem_type.clone(), Some(count))?;

        // Initialize elements
        for (i, elem_expr) in elements.iter().enumerate() {
            if let Some(elem_val) = self.lower_expression(elem_expr) {
                // Calculate element pointer
                let index = self.builder.build_int(i as i64, IrType::I32)?;
                let elem_ptr = self
                    .builder
                    .build_gep(array_ptr, vec![index], elem_type.clone())?;

                // Store element
                self.builder.build_store(elem_ptr, elem_val);
            }
        }

        Some(array_ptr)
    }

    fn lower_object_literal(
        &mut self,
        fields: &[(String, TypedExpression)],
        object_type: TypeId,
    ) -> Option<IrId> {
        // Get the struct type
        let struct_type = self.lower_type(object_type);

        // Allocate memory for the object
        let obj_ptr = self.builder.build_alloc(struct_type.clone(), None)?;

        // Initialize each field
        for (index, (_field_name, field_expr)) in fields.iter().enumerate() {
            // Lower the field value
            let field_value = self.lower_expression(field_expr)?;

            // Calculate field pointer
            let field_type = self.lower_type(field_expr.expr_type);
            let zero_idx = self.builder.build_int(0, IrType::I64)?;
            let field_idx = self.builder.build_int(index as i64, IrType::I64)?;
            let field_ptr =
                self.builder
                    .build_gep(obj_ptr, vec![zero_idx, field_idx], field_type)?;

            // Store the field value
            self.builder.build_store(field_ptr, field_value);
        }

        // Return the object pointer
        Some(obj_ptr)
    }

    fn lower_cast(&mut self, value: &TypedExpression, target_type: TypeId) -> Option<IrId> {
        let value_reg = self.lower_expression(value)?;
        let from_ty = self.lower_type(value.expr_type);
        let to_ty = self.lower_type(target_type);

        self.builder.build_cast(value_reg, from_ty, to_ty)
    }

    fn lower_conditional(
        &mut self,
        condition: &TypedExpression,
        then_expr: &TypedExpression,
        else_expr: Option<&TypedExpression>,
    ) -> Option<IrId> {
        use crate::tast::node::TypedExpressionKind as EK;

        // Check if we have Block expressions - if so, use control flow instead of select
        let has_blocks = matches!(then_expr.kind, EK::Block { .. })
            || else_expr
                .as_ref()
                .map_or(false, |e| matches!(e.kind, EK::Block { .. }));

        if has_blocks {
            // Use control flow for Block expressions
            // Create blocks for the conditional
            let then_block = self.builder.create_block()?;
            let merge_block = self.builder.create_block()?;
            let else_block = if else_expr.is_some() {
                self.builder.create_block()?
            } else {
                merge_block
            };

            // Evaluate condition and branch
            let cond_val = self.lower_expression(condition)?;
            self.builder
                .build_cond_branch(cond_val, then_block, else_block);

            // Lower then branch
            self.builder.switch_to_block(then_block);
            let _then_val = self.lower_expression(then_expr)?;
            // Jump to merge block if not already terminated
            if let Some(current_block) = self.builder.current_block() {
                if let Some(func) = self.builder.current_function() {
                    if let Some(block) = func.cfg.get_block(current_block) {
                        if !block.is_terminated() {
                            self.builder.build_branch(merge_block);
                        }
                    }
                }
            }

            // Lower else branch if present
            if let Some(else_expr) = else_expr {
                self.builder.switch_to_block(else_block);
                let _else_val = self.lower_expression(else_expr)?;
                // Jump to merge block if not already terminated
                if let Some(current_block) = self.builder.current_block() {
                    if let Some(func) = self.builder.current_function() {
                        if let Some(block) = func.cfg.get_block(current_block) {
                            if !block.is_terminated() {
                                self.builder.build_branch(merge_block);
                            }
                        }
                    }
                }
            }

            // Continue in merge block
            self.builder.switch_to_block(merge_block);

            // Return void for block-based conditionals
            use crate::ir::types::IrValue;
            self.builder.build_const(IrValue::Void)
        } else {
            // Use select for simple value expressions
            let cond_val = self.lower_expression(condition)?;
            let then_val = self.lower_expression(then_expr)?;

            if let Some(else_expr) = else_expr {
                let else_val = self.lower_expression(else_expr)?;
                self.builder.build_select(cond_val, then_val, else_val)
            } else {
                // For conditional without else, we need to handle it differently
                // For now, just return the then value if condition is true, null otherwise
                let null_val = self.builder.build_null()?;
                self.builder.build_select(cond_val, then_val, null_val)
            }
        }
    }

    fn lower_ternary(
        &mut self,
        condition: &TypedExpression,
        true_expr: &TypedExpression,
        false_expr: &TypedExpression,
    ) -> Option<IrId> {
        let cond_val = self.lower_expression(condition)?;
        let true_val = self.lower_expression(true_expr)?;
        let false_val = self.lower_expression(false_expr)?;

        self.builder.build_select(cond_val, true_val, false_val)
    }
}

/// Public API for TAST to HIR lowering
pub fn lower_to_hir(
    file: &TypedFile,
    symbol_table: &SymbolTable,
    type_table: &RefCell<TypeTable>,
    _semantic_graphs: Option<&SemanticGraphs>,
    analysis_results: Option<&AnalysisResults>,
) -> Result<IrModule, Vec<LoweringError>> {
    let module_name = "main".to_string(); // TODO: Get from file metadata
    let source_file = String::new(); // TODO: Get from file metadata

    let mut context = LoweringContext::new(module_name, source_file, symbol_table, type_table);

    // Set analysis results if available
    if let Some(results) = analysis_results {
        context.set_analysis_results(results);

        // TODO: Add HIR lowering hints when lifetime issues are resolved
        // let hints = results.get_hir_hints();
        // context.set_lowering_hints(&hints);
    }

    context.lower_file(file)
}

// TODO: Add proper tests once the HIR lowering is stabilized
