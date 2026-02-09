//! Drop Point Analysis for Automatic Memory Deallocation
//!
//! This module provides last-use analysis to determine where to insert Free
//! instructions for heap-allocated values. It implements a lightweight analysis
//! that runs during HIR-to-MIR lowering.
//!
//! ## Algorithm
//!
//! 1. Pre-scan each function to find all variable uses
//! 2. Track the "last use" statement index for each variable
//! 3. During lowering, emit Free after the last use of heap-allocated values
//!
//! ## Example
//!
//! ```haxe
//! function iterate() {
//!     var z = new Complex(0, 0);  // z allocated here
//!     for (i in 0...100) {
//!         z = z.mul(z).add(c);    // old z freed, new z allocated
//!         if (z.abs() > 2.0) {
//!             return i;            // z still alive
//!         }
//!     }
//!     return 1000;                 // z's last use is in loop, but function returns
//! }
//! ```
//!
//! The analyzer tracks that z is used in the loop and determines drop points.

use super::hir::{HirBlock, HirExpr, HirExprKind, HirLValue, HirPattern, HirStatement};
use crate::tast::SymbolId;
use std::collections::{HashMap, HashSet};

/// Defines how a type should be dropped/cleaned up
///
/// This is part of the Drop trait system that determines automatic memory management behavior.
/// The behavior is determined at compile time based on the type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DropBehavior {
    /// Compiler generates Free instruction for this type.
    /// Used for user-defined classes allocated with `new`.
    AutoDrop,

    /// Runtime manages the lifetime, no Free instruction generated.
    /// Used for stdlib types like Thread, Channel, Arc, Mutex.
    /// These types have their own reference counting or cleanup mechanisms.
    RuntimeManaged,

    /// No cleanup needed for this type.
    /// Used for primitives, arrays (runtime-managed buffers), and Dynamic.
    NoDrop,
}

/// Drop point information for a function
#[derive(Debug, Clone, Default)]
pub struct DropPoints {
    /// Variables that need drop and their last-use statement index
    /// The index refers to the position in the flattened statement list
    pub last_use: HashMap<SymbolId, LastUseInfo>,

    /// Variables that are heap-allocated (from `new` expressions)
    pub heap_allocated: HashSet<SymbolId>,

    /// Variables that escape (returned, stored in fields, passed to functions)
    /// Used by last-use analysis to prevent premature freeing
    pub escaping: HashSet<SymbolId>,

    /// Variables captured by lambdas - these truly escape the function scope
    /// and should NOT be freed at scope exit (the closure owns them)
    pub lambda_captures: HashSet<SymbolId>,
}

/// Information about a variable's last use
#[derive(Debug, Clone)]
pub struct LastUseInfo {
    /// Statement index where variable is last used
    pub statement_index: usize,

    /// Whether the last use is in a loop (may need deferred drop)
    pub in_loop: bool,

    /// Whether the variable is reassigned (drop happens at reassignment)
    pub is_reassigned: bool,

    /// Block depth at last use (for scope-based drop)
    pub block_depth: usize,
}

/// Analyzer for computing drop points in HIR functions
pub struct DropPointAnalyzer {
    /// Current statement index during traversal
    current_stmt_idx: usize,

    /// Current block depth (for scope tracking)
    current_depth: usize,

    /// Whether we're inside a loop
    in_loop: bool,

    /// Collected uses: variable -> list of (stmt_idx, in_loop, depth)
    uses: HashMap<SymbolId, Vec<(usize, bool, usize)>>,

    /// Variables assigned from `new` expressions
    heap_vars: HashSet<SymbolId>,

    /// Variables that are reassigned
    reassigned: HashSet<SymbolId>,

    /// Variables that escape the function (passed to functions, returned, etc.)
    escaping: HashSet<SymbolId>,

    /// Variables captured by lambdas (truly escape scope)
    lambda_captures: HashSet<SymbolId>,
}

impl DropPointAnalyzer {
    pub fn new() -> Self {
        Self {
            current_stmt_idx: 0,
            current_depth: 0,
            in_loop: false,
            uses: HashMap::new(),
            heap_vars: HashSet::new(),
            reassigned: HashSet::new(),
            escaping: HashSet::new(),
            lambda_captures: HashSet::new(),
        }
    }

    /// Analyze a function body and return drop points
    pub fn analyze_function(&mut self, body: &HirBlock) -> DropPoints {
        // Reset state
        self.current_stmt_idx = 0;
        self.current_depth = 0;
        self.in_loop = false;
        self.uses.clear();
        self.heap_vars.clear();
        self.reassigned.clear();
        self.escaping.clear();
        self.lambda_captures.clear();

        // Traverse the function body
        self.analyze_block(body);

        // Compute last use for each variable
        let mut last_use = HashMap::new();
        for (symbol, use_list) in &self.uses {
            if let Some(&(stmt_idx, in_loop, depth)) = use_list.last() {
                last_use.insert(
                    *symbol,
                    LastUseInfo {
                        statement_index: stmt_idx,
                        in_loop,
                        is_reassigned: self.reassigned.contains(symbol),
                        block_depth: depth,
                    },
                );
            }
        }

        DropPoints {
            last_use,
            heap_allocated: self.heap_vars.clone(),
            escaping: self.escaping.clone(),
            lambda_captures: self.lambda_captures.clone(),
        }
    }

    fn analyze_block(&mut self, block: &HirBlock) {
        self.current_depth += 1;

        for stmt in &block.statements {
            self.analyze_statement(stmt);
            self.current_stmt_idx += 1;
        }

        if let Some(expr) = &block.expr {
            self.analyze_expr(expr);
        }

        self.current_depth -= 1;
    }

    fn analyze_statement(&mut self, stmt: &HirStatement) {
        match stmt {
            HirStatement::Let { pattern, init, .. } => {
                // Check if init is a heap allocation
                if let Some(init_expr) = init {
                    let is_heap = self.is_heap_allocation(init_expr);

                    if is_heap {
                        if let Some(symbol) = self.pattern_symbol(pattern) {
                            self.heap_vars.insert(symbol);
                        }
                    }

                    self.analyze_expr(init_expr);
                }
            }

            HirStatement::Assign { lhs, rhs, .. } => {
                // Track reassignment
                if let HirLValue::Variable(symbol) = lhs {
                    if self.heap_vars.contains(symbol) {
                        self.reassigned.insert(*symbol);
                    }

                    // If RHS is heap allocation, track it
                    if self.is_heap_allocation(rhs) {
                        self.heap_vars.insert(*symbol);
                    }
                }

                self.analyze_lvalue(lhs);
                self.analyze_expr(rhs);
            }

            HirStatement::Expr(expr) => {
                self.analyze_expr(expr);
            }

            HirStatement::Return(expr_opt) => {
                if let Some(expr) = expr_opt {
                    // Mark returned variables as escaping
                    self.mark_escaping(expr);
                    self.analyze_expr(expr);
                }
            }

            HirStatement::If {
                condition,
                then_branch,
                else_branch,
            } => {
                self.analyze_expr(condition);
                self.analyze_block(then_branch);
                if let Some(else_block) = else_branch {
                    self.analyze_block(else_block);
                }
            }

            HirStatement::While {
                label: _,
                condition,
                body,
                continue_update,
            } => {
                let was_in_loop = self.in_loop;
                self.in_loop = true;

                self.analyze_expr(condition);
                self.analyze_block(body);
                if let Some(update) = continue_update {
                    self.analyze_block(update);
                }

                self.in_loop = was_in_loop;
            }

            HirStatement::ForIn {
                label: _,
                pattern: _,
                iterator,
                body,
            } => {
                let was_in_loop = self.in_loop;
                self.in_loop = true;

                self.analyze_expr(iterator);
                self.analyze_block(body);

                self.in_loop = was_in_loop;
            }

            _ => {}
        }
    }

    fn analyze_expr(&mut self, expr: &HirExpr) {
        match &expr.kind {
            HirExprKind::Variable { symbol, .. } => {
                self.record_use(*symbol);
            }

            HirExprKind::Call { callee, args, .. } => {
                self.analyze_expr(callee);
                for arg in args {
                    // Mark arguments as escaping for now - scope-based drops handle final cleanup
                    // but last-use analysis should not free values passed to functions
                    self.mark_escaping(arg);
                    self.analyze_expr(arg);
                }
            }

            HirExprKind::Field { object, .. } => {
                self.analyze_expr(object);
            }

            HirExprKind::Index { object, index } => {
                self.analyze_expr(object);
                self.analyze_expr(index);
            }

            HirExprKind::Binary { lhs, rhs, .. } => {
                self.analyze_expr(lhs);
                self.analyze_expr(rhs);
            }

            HirExprKind::Unary { operand, .. } => {
                self.analyze_expr(operand);
            }

            HirExprKind::If {
                condition,
                then_expr,
                else_expr,
            } => {
                self.analyze_expr(condition);
                self.analyze_expr(then_expr);
                self.analyze_expr(else_expr);
            }

            HirExprKind::Block(block) => {
                self.analyze_block(block);
            }

            HirExprKind::New { args, .. } => {
                for arg in args {
                    self.analyze_expr(arg);
                }
            }

            HirExprKind::Array { elements } => {
                for elem in elements {
                    self.analyze_expr(elem);
                }
            }

            HirExprKind::Cast { expr, .. } => {
                self.analyze_expr(expr);
            }

            HirExprKind::Lambda { captures, body, .. } => {
                // CRITICAL: Record captured variables as "used" at the lambda
                // expression. This prevents the drop analyzer from freeing
                // captured variables before they're used by the closure.
                for capture in captures {
                    self.record_use(capture.symbol);
                    // Mark as escaping for last-use analysis (don't free early)
                    self.escaping.insert(capture.symbol);
                    // Also mark as lambda capture - these TRULY escape scope
                    // and should NOT be freed at scope exit (closure owns them)
                    self.lambda_captures.insert(capture.symbol);
                }
                // Also analyze the body in case there are nested expressions
                self.analyze_expr(body);
            }

            _ => {}
        }
    }

    fn analyze_lvalue(&mut self, lvalue: &HirLValue) {
        match lvalue {
            HirLValue::Variable(symbol) => {
                self.record_use(*symbol);
            }
            HirLValue::Field { object, .. } => {
                self.analyze_expr(object);
            }
            HirLValue::Index { object, index } => {
                self.analyze_expr(object);
                self.analyze_expr(index);
            }
        }
    }

    fn record_use(&mut self, symbol: SymbolId) {
        let uses = self.uses.entry(symbol).or_insert_with(Vec::new);
        uses.push((self.current_stmt_idx, self.in_loop, self.current_depth));
    }

    fn is_heap_allocation(&self, expr: &HirExpr) -> bool {
        matches!(&expr.kind, HirExprKind::New { .. })
    }

    fn pattern_symbol(&self, pattern: &HirPattern) -> Option<SymbolId> {
        match pattern {
            HirPattern::Variable { symbol, .. } => Some(*symbol),
            _ => None,
        }
    }

    fn mark_escaping(&mut self, expr: &HirExpr) {
        // Mark variables used in the expression as potentially escaping
        match &expr.kind {
            HirExprKind::Variable { symbol, .. } => {
                self.escaping.insert(*symbol);
            }
            _ => {}
        }
    }
}

/// Check if a variable should be dropped based on drop point analysis
pub fn should_drop_at_statement(
    drop_points: &DropPoints,
    symbol: SymbolId,
    current_stmt_idx: usize,
) -> bool {
    // Don't drop escaping variables
    if drop_points.escaping.contains(&symbol) {
        return false;
    }

    // Only drop heap-allocated variables
    if !drop_points.heap_allocated.contains(&symbol) {
        return false;
    }

    // Check if this is the last use
    if let Some(last_use) = drop_points.last_use.get(&symbol) {
        // If reassigned, drops happen at reassignment points, not last use
        if last_use.is_reassigned {
            return false;
        }

        // Drop after last use
        return current_stmt_idx == last_use.statement_index;
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tests would go here
}
