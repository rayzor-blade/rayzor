// Lambda/Function Literal Support for DFG Builder

use std::collections::BTreeSet;

use crate::{
    semantic_graph::dfg_builder::SsaConstructionState,
    tast::{
        node::{TypedExpression, TypedExpressionKind, TypedParameter, TypedStatement},
        SsaVariableId, SymbolId, TypeId,
    },
};

/// Information about a captured variable
#[derive(Debug, Clone)]
pub struct CapturedVariable {
    pub(crate) symbol_id: SymbolId,
    pub(crate) ssa_var_id: SsaVariableId,
    pub(crate) capture_type: TypeId,
}

/// Free variable visitor for lambda capture analysis
pub struct FreeVariableVisitor<'a> {
    /// Variables bound in the current scope (parameters + locals)
    pub(crate) bound_variables: BTreeSet<SymbolId>,
    /// Free variables found (need to be captured)
    pub(crate) free_variables: BTreeSet<SymbolId>,
    /// Current SSA state for variable lookup
    pub(crate) ssa_state: &'a SsaConstructionState,
}

impl<'a> FreeVariableVisitor<'a> {
    pub(crate) fn new(parameters: &[TypedParameter], ssa_state: &'a SsaConstructionState) -> Self {
        let mut bound_variables = BTreeSet::new();

        // Parameters are bound in the lambda scope
        for param in parameters {
            bound_variables.insert(param.symbol_id);
        }

        Self {
            bound_variables,
            free_variables: BTreeSet::new(),
            ssa_state,
        }
    }

    /// Visit statements to find free variables
    pub(crate) fn visit_statements(&mut self, statements: &[TypedStatement]) {
        for statement in statements {
            self.visit_statement(statement);
        }
    }

    /// Visit a single statement
    fn visit_statement(&mut self, statement: &TypedStatement) {
        match statement {
            TypedStatement::VarDeclaration {
                symbol_id,
                initializer,
                ..
            } => {
                // Variable declaration binds a new variable
                self.bound_variables.insert(*symbol_id);

                // But check the initializer for free variables
                if let Some(init_expr) = initializer {
                    self.visit_expression(init_expr);
                }
            }

            TypedStatement::Expression { expression, .. } => {
                self.visit_expression(expression);
            }

            TypedStatement::Assignment { target, value, .. } => {
                self.visit_expression(target);
                self.visit_expression(value);
            }

            TypedStatement::If {
                condition,
                then_branch,
                else_branch,
                ..
            } => {
                self.visit_expression(condition);
                self.visit_statement(then_branch);
                if let Some(else_stmt) = else_branch {
                    self.visit_statement(else_stmt);
                }
            }

            TypedStatement::While {
                condition, body, ..
            } => {
                self.visit_expression(condition);
                self.visit_statement(body);
            }

            TypedStatement::For {
                init,
                condition,
                update,
                body,
                ..
            } => {
                if let Some(init_stmt) = init {
                    self.visit_statement(init_stmt);
                }
                if let Some(cond) = condition {
                    self.visit_expression(cond);
                }
                if let Some(upd) = update {
                    self.visit_expression(upd);
                }
                self.visit_statement(body);
            }

            TypedStatement::Block { statements, .. } => {
                // Save current bound variables
                let saved_bound = self.bound_variables.clone();

                // Visit block statements
                self.visit_statements(statements);

                // Restore bound variables (block scope ends)
                self.bound_variables = saved_bound;
            }

            TypedStatement::Return { value, .. } => {
                if let Some(val) = value {
                    self.visit_expression(val);
                }
            }

            TypedStatement::Throw { exception, .. } => {
                self.visit_expression(exception);
            }

            TypedStatement::Try {
                body,
                catch_clauses,
                finally_block,
                ..
            } => {
                self.visit_statement(body);

                for catch in catch_clauses {
                    let saved_bound = self.bound_variables.clone();
                    self.bound_variables.insert(catch.exception_variable);
                    self.visit_statement(&catch.body);
                    self.bound_variables = saved_bound;
                }

                if let Some(finally) = finally_block {
                    self.visit_statement(finally);
                }
            }

            _ => {} // Other statement types
        }
    }

    /// Visit an expression
    fn visit_expression(&mut self, expression: &TypedExpression) {
        match &expression.kind {
            TypedExpressionKind::Variable { symbol_id } => {
                // Check if this variable is free (not bound in lambda scope)
                if !self.bound_variables.contains(symbol_id) {
                    self.free_variables.insert(*symbol_id);
                }
            }

            TypedExpressionKind::FieldAccess { object, .. } => {
                self.visit_expression(object);
            }

            TypedExpressionKind::ArrayAccess { array, index } => {
                self.visit_expression(array);
                self.visit_expression(index);
            }

            TypedExpressionKind::FunctionCall {
                function,
                arguments,
                ..
            } => {
                self.visit_expression(function);
                for arg in arguments {
                    self.visit_expression(arg);
                }
            }

            TypedExpressionKind::MethodCall {
                receiver,
                arguments,
                ..
            } => {
                self.visit_expression(receiver);
                for arg in arguments {
                    self.visit_expression(arg);
                }
            }

            TypedExpressionKind::BinaryOp { left, right, .. } => {
                self.visit_expression(left);
                self.visit_expression(right);
            }

            TypedExpressionKind::UnaryOp { operand, .. } => {
                self.visit_expression(operand);
            }

            TypedExpressionKind::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
                self.visit_expression(condition);
                self.visit_expression(then_expr);
                if let Some(else_expr) = else_expr {
                    self.visit_expression(else_expr);
                }
            }

            TypedExpressionKind::ArrayLiteral { elements } => {
                for elem in elements {
                    self.visit_expression(elem);
                }
            }

            TypedExpressionKind::ObjectLiteral { fields } => {
                for field in fields {
                    self.visit_expression(&field.value);
                }
            }

            TypedExpressionKind::FunctionLiteral {
                parameters, body, ..
            } => {
                // Nested lambda - need to handle its captures too
                let mut nested_visitor = FreeVariableVisitor::new(parameters, self.ssa_state);
                nested_visitor.visit_statements(body);

                // Free variables of nested lambda are also free in this context
                for free_var in nested_visitor.free_variables {
                    if !self.bound_variables.contains(&free_var) {
                        self.free_variables.insert(free_var);
                    }
                }
            }

            TypedExpressionKind::Cast { expression, .. } => {
                self.visit_expression(expression);
            }

            TypedExpressionKind::New { arguments, .. } => {
                for arg in arguments {
                    self.visit_expression(arg);
                }
            }

            _ => {} // Literals, This, Super, Null don't have free variables
        }
    }
}

#[cfg(test)]
mod lambda_tests {
    use super::*;
    use crate::{
        semantic_graph::{
            dfg_builder::DfgBuilder, ConstantValue, DataFlowNode, DataFlowNodeKind,
            GraphConstructionOptions, NodeMetadata,
        },
        tast::{
            collections::new_id_set, node::*, BlockId, DataFlowNodeId, Mutability, SourceLocation,
            StringInterner,
        },
    };

    /// Create a simple lambda that captures a variable
    fn create_capturing_lambda() -> TypedExpression {
        let interner = StringInterner::new();
        // Lambda: (y) -> x + y  (where x is captured)
        let x_ref = TypedExpression {
            expr_type: TypeId::from_raw(1), // int
            kind: TypedExpressionKind::Variable {
                symbol_id: SymbolId::from_raw(10), // x
            },
            usage: VariableUsage::Borrow,
            lifetime_id: crate::tast::LifetimeId::static_lifetime(),
            source_location: SourceLocation::unknown(),
            metadata: ExpressionMetadata::default(),
        };

        let y_ref = TypedExpression {
            expr_type: TypeId::from_raw(1), // int
            kind: TypedExpressionKind::Variable {
                symbol_id: SymbolId::from_raw(11), // y (parameter)
            },
            usage: VariableUsage::Borrow,
            lifetime_id: crate::tast::LifetimeId::static_lifetime(),
            source_location: SourceLocation::unknown(),
            metadata: ExpressionMetadata::default(),
        };

        let add_expr = TypedExpression {
            expr_type: TypeId::from_raw(1), // int
            kind: TypedExpressionKind::BinaryOp {
                left: Box::new(x_ref),
                operator: BinaryOperator::Add,
                right: Box::new(y_ref),
            },
            usage: VariableUsage::Move,
            lifetime_id: crate::tast::LifetimeId::static_lifetime(),
            source_location: SourceLocation::unknown(),
            metadata: ExpressionMetadata::default(),
        };

        let return_stmt = TypedStatement::Return {
            value: Some(add_expr),
            source_location: SourceLocation::unknown(),
        };

        let param = TypedParameter {
            symbol_id: SymbolId::from_raw(11),
            name: interner.intern("y"),
            param_type: TypeId::from_raw(1), // int
            is_optional: false,
            default_value: None,
            mutability: Mutability::Immutable,
            source_location: SourceLocation::unknown(),
        };

        TypedExpression {
            expr_type: TypeId::from_raw(100), // Function type
            kind: TypedExpressionKind::FunctionLiteral {
                parameters: vec![param],
                body: vec![return_stmt],
                return_type: TypeId::from_raw(1), // int
            },
            usage: VariableUsage::Move,
            lifetime_id: crate::tast::LifetimeId::static_lifetime(),
            source_location: SourceLocation::unknown(),
            metadata: ExpressionMetadata::default(),
        }
    }

    #[test]
    fn test_lambda_with_captures() {
        let mut builder = DfgBuilder::new(GraphConstructionOptions::default());

        // Set up outer variable x
        let x_symbol = SymbolId::from_raw(10);
        let x_ssa = builder.allocate_ssa_variable(x_symbol, TypeId::from_raw(1));
        builder.push_ssa_variable(x_symbol, x_ssa);

        // Create defining node for x
        let x_def_node = DataFlowNode {
            id: DataFlowNodeId::from_raw(100),
            kind: DataFlowNodeKind::Constant {
                value: ConstantValue::Int(5),
            },
            value_type: TypeId::from_raw(1),
            source_location: SourceLocation::unknown(),
            operands: vec![],
            uses: new_id_set(),
            defines: Some(x_ssa),
            basic_block: BlockId::from_raw(1),
            metadata: NodeMetadata::default(),
        };
        builder.dfg.add_node(x_def_node);

        // Build lambda expression
        let lambda_expr = create_capturing_lambda();
        let lambda_node_id = builder.build_expression(&lambda_expr).unwrap();

        // Verify lambda was created
        let lambda_node = builder.dfg.get_node(lambda_node_id).unwrap();
        assert!(matches!(lambda_node.kind, DataFlowNodeKind::Load { .. }));

        // Check metadata
        assert_eq!(
            lambda_node.metadata.annotations.get("closure"),
            Some(&"true".to_string())
        );
        assert_eq!(
            lambda_node.metadata.annotations.get("capture_count"),
            Some(&"1".to_string())
        );

        // Verify allocation and store nodes were created
        let alloc_nodes = builder
            .dfg
            .nodes
            .values()
            .filter(|n| matches!(n.kind, DataFlowNodeKind::Allocation { .. }))
            .count();
        assert!(alloc_nodes >= 1);

        let store_nodes = builder
            .dfg
            .nodes
            .values()
            .filter(|n| matches!(n.kind, DataFlowNodeKind::Store { .. }))
            .count();
        assert!(store_nodes >= 1); // At least one store for the captured variable
    }

    #[test]
    fn test_lambda_without_captures() {
        let mut builder = DfgBuilder::new(GraphConstructionOptions::default());
        let interner = StringInterner::new();
        // Lambda with no captures: (x) -> x * 2
        let param = TypedParameter {
            symbol_id: SymbolId::from_raw(20),
            name: interner.intern("x"),
            param_type: TypeId::from_raw(1),
            is_optional: false,
            default_value: None,
            mutability: Mutability::Immutable,
            source_location: SourceLocation::unknown(),
        };

        let x_ref = TypedExpression {
            expr_type: TypeId::from_raw(1),
            kind: TypedExpressionKind::Variable {
                symbol_id: SymbolId::from_raw(20),
            },
            usage: VariableUsage::Borrow,
            lifetime_id: crate::tast::LifetimeId::static_lifetime(),
            source_location: SourceLocation::unknown(),
            metadata: ExpressionMetadata::default(),
        };

        let two = TypedExpression {
            expr_type: TypeId::from_raw(1),
            kind: TypedExpressionKind::Literal {
                value: LiteralValue::Int(2),
            },
            usage: VariableUsage::Copy,
            lifetime_id: crate::tast::LifetimeId::static_lifetime(),
            source_location: SourceLocation::unknown(),
            metadata: ExpressionMetadata::default(),
        };

        let mul_expr = TypedExpression {
            expr_type: TypeId::from_raw(1),
            kind: TypedExpressionKind::BinaryOp {
                left: Box::new(x_ref),
                operator: BinaryOperator::Mul,
                right: Box::new(two),
            },
            usage: VariableUsage::Move,
            lifetime_id: crate::tast::LifetimeId::static_lifetime(),
            source_location: SourceLocation::unknown(),
            metadata: ExpressionMetadata::default(),
        };

        let return_stmt = TypedStatement::Return {
            value: Some(mul_expr),
            source_location: SourceLocation::unknown(),
        };

        let lambda_expr = TypedExpression {
            expr_type: TypeId::from_raw(100),
            kind: TypedExpressionKind::FunctionLiteral {
                parameters: vec![param],
                body: vec![return_stmt],
                return_type: TypeId::from_raw(1),
            },
            usage: VariableUsage::Move,
            lifetime_id: crate::tast::LifetimeId::static_lifetime(),
            source_location: SourceLocation::unknown(),
            metadata: ExpressionMetadata::default(),
        };

        let lambda_node_id = builder.build_expression(&lambda_expr).unwrap();

        // Verify lambda was created
        let lambda_node = builder.dfg.get_node(lambda_node_id).unwrap();
        assert_eq!(
            lambda_node.metadata.annotations.get("capture_count"),
            Some(&"0".to_string())
        );

        // Should have allocation but no capture stores
        let store_nodes = builder
            .dfg
            .nodes
            .values()
            .filter(|n| matches!(n.kind, DataFlowNodeKind::Store { .. }))
            .count();
        assert_eq!(store_nodes, 0); // No captures means no stores
    }
}
