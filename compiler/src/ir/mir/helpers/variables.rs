//! Variable-reference collection, and the SSA hints carried over from HIR.

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
    /// Collect all variables referenced in a block
    pub(crate) fn collect_referenced_variables_in_block(
        &self,
        block: &HirBlock,
        vars: &mut std::collections::BTreeSet<SymbolId>,
    ) {
        for stmt in &block.statements {
            self.collect_referenced_variables_in_stmt(stmt, vars);
        }
        // Also check the block's trailing expression (block-as-value)
        if let Some(expr) = &block.expr {
            self.collect_referenced_variables_in_expr(expr, vars);
        }
    }

    /// Collect all variables referenced in an expression
    pub(crate) fn collect_referenced_variables_in_expr(
        &self,
        expr: &HirExpr,
        vars: &mut std::collections::BTreeSet<SymbolId>,
    ) {
        match &expr.kind {
            HirExprKind::Variable { symbol, .. } => {
                vars.insert(*symbol);
            }
            HirExprKind::Binary { lhs, rhs, .. } => {
                self.collect_referenced_variables_in_expr(lhs, vars);
                self.collect_referenced_variables_in_expr(rhs, vars);
            }
            HirExprKind::Unary { operand, .. } => {
                self.collect_referenced_variables_in_expr(operand, vars);
            }
            HirExprKind::If {
                condition,
                then_expr,
                else_expr,
                ..
            } => {
                self.collect_referenced_variables_in_expr(condition, vars);
                self.collect_referenced_variables_in_expr(then_expr, vars);
                self.collect_referenced_variables_in_expr(else_expr, vars);
            }
            HirExprKind::Call { callee, args, .. } => {
                self.collect_referenced_variables_in_expr(callee, vars);
                for arg in args {
                    self.collect_referenced_variables_in_expr(arg, vars);
                }
            }
            HirExprKind::Block(block) => {
                self.collect_referenced_variables_in_block(block, vars);
            }
            HirExprKind::Field { object, .. } => {
                self.collect_referenced_variables_in_expr(object, vars);
            }
            HirExprKind::Index { object, index } => {
                self.collect_referenced_variables_in_expr(object, vars);
                self.collect_referenced_variables_in_expr(index, vars);
            }
            HirExprKind::New { args, .. } => {
                for a in args {
                    self.collect_referenced_variables_in_expr(a, vars);
                }
            }
            HirExprKind::Cast { expr, .. }
            | HirExprKind::TypeCheck { expr, .. }
            | HirExprKind::Reification { expr } => {
                self.collect_referenced_variables_in_expr(expr, vars);
            }
            HirExprKind::Untyped(inner) => {
                self.collect_referenced_variables_in_expr(inner, vars);
            }
            HirExprKind::Lambda { body, .. } => {
                self.collect_referenced_variables_in_expr(body, vars);
            }
            HirExprKind::MethodReference { receiver, .. } => {
                self.collect_referenced_variables_in_expr(receiver, vars);
            }
            HirExprKind::Array { elements } => {
                for e in elements {
                    self.collect_referenced_variables_in_expr(e, vars);
                }
            }
            HirExprKind::Map { entries } => {
                for (k, v) in entries {
                    self.collect_referenced_variables_in_expr(k, vars);
                    self.collect_referenced_variables_in_expr(v, vars);
                }
            }
            HirExprKind::ObjectLiteral { fields } => {
                for (_, v) in fields {
                    self.collect_referenced_variables_in_expr(v, vars);
                }
            }
            // A variable mutated inside a try/catch/finally lowered as an
            // EXPRESSION (`try { x += 1; } catch (e) {}` inside a block) must
            // still be collected, or an enclosing loop won't build its phi.
            HirExprKind::TryCatch {
                try_expr,
                catch_handlers,
                finally_expr,
            } => {
                self.collect_referenced_variables_in_expr(try_expr, vars);
                for h in catch_handlers {
                    if let Some(g) = &h.guard {
                        self.collect_referenced_variables_in_expr(g, vars);
                    }
                    self.collect_referenced_variables_in_expr(&h.body, vars);
                }
                if let Some(f) = finally_expr {
                    self.collect_referenced_variables_in_expr(f, vars);
                }
            }
            _ => {}
        }
    }

    /// Collect all variables referenced in a statement
    pub(crate) fn collect_referenced_variables_in_stmt(
        &self,
        stmt: &HirStatement,
        vars: &mut std::collections::BTreeSet<SymbolId>,
    ) {
        match stmt {
            HirStatement::Let { init, .. } => {
                if let Some(expr) = init {
                    self.collect_referenced_variables_in_expr(expr, vars);
                }
            }
            HirStatement::Expr(expr) => {
                self.collect_referenced_variables_in_expr(expr, vars);
            }
            HirStatement::Assign { lhs, rhs, .. } => {
                match lhs {
                    HirLValue::Variable(sym) => {
                        vars.insert(*sym);
                    }
                    HirLValue::Field { object, .. } => {
                        self.collect_referenced_variables_in_expr(object, vars);
                    }
                    HirLValue::Index { object, index } => {
                        self.collect_referenced_variables_in_expr(object, vars);
                        self.collect_referenced_variables_in_expr(index, vars);
                    }
                }
                self.collect_referenced_variables_in_expr(rhs, vars);
            }
            HirStatement::Return(Some(expr)) => {
                self.collect_referenced_variables_in_expr(expr, vars);
            }
            HirStatement::If {
                condition,
                then_branch,
                else_branch,
                ..
            } => {
                self.collect_referenced_variables_in_expr(condition, vars);
                self.collect_referenced_variables_in_block(then_branch, vars);
                if let Some(else_blk) = else_branch {
                    self.collect_referenced_variables_in_block(else_blk, vars);
                }
            }
            HirStatement::While {
                condition, body, ..
            }
            | HirStatement::DoWhile {
                condition, body, ..
            } => {
                self.collect_referenced_variables_in_expr(condition, vars);
                self.collect_referenced_variables_in_block(body, vars);
            }
            _ => {}
        }
    }

    /// Snapshot the current SSA register for each tracked variable, for building
    /// merge phis at a convergence point (used by try/catch lowering).
    pub(crate) fn capture_tracked_values(
        &self,
        tracked: &BTreeMap<SymbolId, (IrId, IrType)>,
    ) -> BTreeMap<SymbolId, IrId> {
        let mut vals: BTreeMap<SymbolId, IrId> = BTreeMap::new();
        for s in tracked.keys() {
            if let Some(&reg) = self.symbol_map.get(s) {
                vals.insert(*s, reg);
            }
        }
        vals
    }

    /// True iff `sym` is an actual function-parameter symbol.
    ///
    /// Loop-phi construction excludes parameters, which need no header phi.
    /// Identify them by SYMBOL KIND, never by register: `var i = lo` aliases
    /// `i` to `lo`'s register instead of emitting a copy, so a register
    /// comparison would misclassify such a local as a parameter and deny it
    /// the loop phi it needs.
    pub(crate) fn is_parameter_symbol(&self, sym: &SymbolId) -> bool {
        self.symbol_table
            .get_symbol(*sym)
            .map(|s| s.kind == crate::tast::symbols::SymbolKind::Parameter)
            .unwrap_or(false)
    }

    pub(crate) fn extract_ssa_hints_from_hir(&mut self, hir_module: &HirModule) {
        for (symbol_id, func) in &hir_module.functions {
            // Parse optimization hints from function metadata
            for attr in &func.metadata {
                let attr_name = self.interned_str(attr.name);
                match attr_name {
                    "inline_candidate" => {
                        self.ssa_hints.inline_candidates.insert(*symbol_id);
                    }
                    "optimization_hint" => {
                        if let Some(HirAttributeArg::Literal(HirLiteral::String(hint))) =
                            attr.args.first()
                        {
                            match self.interned_str(*hint) {
                                "straight_line_code" => {
                                    self.ssa_hints.straight_line_functions.insert(*symbol_id);
                                }
                                "complex_control_flow" => {
                                    self.ssa_hints
                                        .complex_control_flow_functions
                                        .insert(*symbol_id);
                                }
                                "common_subexpressions" => {
                                    self.ssa_hints.cse_opportunities.insert(*symbol_id);
                                }
                                _ => {}
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}
