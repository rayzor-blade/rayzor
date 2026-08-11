//! Finds variables a region modifies, and the enclosing loop context.

use super::*;
use crate::ir::drop_analysis::{DropBehavior, DropPointAnalyzer, DropPoints};
use crate::ir::hir::*;
use crate::ir::{
    BinaryOp, CallingConvention, CompareOp, EnvironmentLayout, FunctionKind,
    FunctionSignatureBuilder, IrBasicBlock, IrBlockId, IrBuilder, IrEnumVariant, IrField,
    IrFunction, IrFunctionId, IrFunctionSignature, IrGlobal, IrGlobalId, IrId, IrInstruction,
    IrLocal, IrModule, IrParameter, IrPhiNode, IrSourceLocation, IrTerminator, IrType, IrTypeDef,
    IrTypeDefId, IrTypeDefinition, IrValue, Linkage, UnaryOp,
};
use crate::stdlib::{IrTypeDescriptor, MethodSignature, StdlibMapping};
use crate::tast::symbols::SymbolFlags;
use crate::tast::{
    InternedString, SourceLocation, StringInterner, SymbolId, SymbolTable, TypeId, TypeKind,
    TypeTable,
};
use log::{debug, trace, warn};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

impl<'a> HirToMirContext<'a> {
    /// Find all variables that are modified (assigned) in a block
    /// This is used for SSA phi node insertion in loops
    pub(crate) fn find_modified_variables_in_block(
        &self,
        block: &HirBlock,
    ) -> std::collections::BTreeSet<SymbolId> {
        let mut modified = std::collections::BTreeSet::new();

        for stmt in &block.statements {
            self.find_modified_variables_in_statement(stmt, &mut modified);
        }
        // Also check the block's trailing expression (block-as-value)
        if let Some(expr) = &block.expr {
            self.find_modified_variables_in_expression(expr, &mut modified);
        }

        modified
    }

    /// Recursively find modified variables in a statement
    pub(crate) fn find_modified_variables_in_statement(
        &self,
        stmt: &HirStatement,
        modified: &mut std::collections::BTreeSet<SymbolId>,
    ) {
        match stmt {
            HirStatement::Let { pattern, .. } => {
                // Variable declarations are modifications
                // debug!("Found Let statement");
                self.collect_pattern_variables(pattern, modified);
            }
            HirStatement::Expr(expr) => {
                // debug!("Found Expr statement with kind: {:?}", std::mem::discriminant(&expr.kind));
                self.find_modified_variables_in_expression(expr, modified);
            }
            HirStatement::Assign { lhs, rhs, .. } => {
                // Assignments modify the left-hand side
                match lhs {
                    HirLValue::Variable(symbol) => {
                        modified.insert(*symbol);
                    }
                    _ => {}
                }
                self.find_modified_variables_in_expression(rhs, modified);
            }
            HirStatement::If {
                then_branch,
                else_branch,
                ..
            } => {
                for stmt in &then_branch.statements {
                    self.find_modified_variables_in_statement(stmt, modified);
                }
                if let Some(else_blk) = else_branch {
                    for stmt in &else_blk.statements {
                        self.find_modified_variables_in_statement(stmt, modified);
                    }
                }
            }
            HirStatement::While { body, .. } | HirStatement::DoWhile { body, .. } => {
                for stmt in &body.statements {
                    self.find_modified_variables_in_statement(stmt, modified);
                }
            }
            HirStatement::ForIn { body, .. } => {
                for stmt in &body.statements {
                    self.find_modified_variables_in_statement(stmt, modified);
                }
            }
            HirStatement::Label { block, .. } => {
                for stmt in &block.statements {
                    self.find_modified_variables_in_statement(stmt, modified);
                }
            }
            _ => {}
        }
    }

    /// Find modified variables in an expression (assignments)
    pub(crate) fn find_modified_variables_in_expression(
        &self,
        expr: &HirExpr,
        modified: &mut std::collections::BTreeSet<SymbolId>,
    ) {
        match &expr.kind {
            HirExprKind::Binary { lhs, rhs, .. } => {
                self.find_modified_variables_in_expression(lhs, modified);
                self.find_modified_variables_in_expression(rhs, modified);
            }
            HirExprKind::Unary { operand, .. } => {
                self.find_modified_variables_in_expression(operand, modified);
            }
            HirExprKind::If {
                then_expr,
                else_expr,
                ..
            } => {
                self.find_modified_variables_in_expression(then_expr, modified);
                self.find_modified_variables_in_expression(else_expr, modified);
            }
            HirExprKind::Call { args, .. } => {
                for arg in args {
                    self.find_modified_variables_in_expression(arg, modified);
                }
            }
            HirExprKind::Block(block) => {
                for stmt in &block.statements {
                    self.find_modified_variables_in_statement(stmt, modified);
                }
            }
            _ => {}
        }
    }

    pub(crate) fn find_loop_context(&self, label: Option<&SymbolId>) -> Option<&LoopContext> {
        if let Some(label) = label {
            self.loop_stack
                .iter()
                .rev()
                .find(|ctx| ctx.label.as_ref() == Some(label))
        } else {
            self.loop_stack.last()
        }
    }

    /// Check if a TypeId refers to a TypeParameter in the type table.
    /// If so, returns the type param name for tag-based generic dispatch.
    pub(crate) fn resolve_type_param_name(&self, type_id: crate::tast::TypeId) -> Option<String> {
        let type_table = self.type_table;
        if let Some(ti) = type_table.get(type_id) {
            if let crate::tast::TypeKind::TypeParameter {
                symbol_id,
                constraints,
                ..
            } = &ti.kind
            {
                if constraints.is_empty() || !self.has_interface_constraint(constraints) {
                    return self
                        .symbol_table
                        .get_symbol(*symbol_id)
                        .and_then(|s| self.string_interner.get(s.name))
                        .map(|s| s.to_string());
                }
            }
        }
        None
    }
}
