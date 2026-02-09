//! Generic optimization trait for different IR levels
//!
//! This module provides a trait that allows optimization passes to work generically
//! across different IR representations (HIR, MIR, LIR).

use super::optimization::{OptimizationPass, OptimizationResult};
use super::validation::ValidationError;
use super::IrType;
use std::any::Any;
use std::fmt::Debug;

/// Trait for IR modules that can be optimized
pub trait OptimizableModule: Debug {
    /// Get the IR level name (e.g., "HIR", "MIR", "LIR")
    fn ir_level(&self) -> &'static str;

    /// Get module name
    fn module_name(&self) -> &str;

    /// Run a specific optimization pass on this module
    fn run_optimization(&mut self, pass: &mut dyn OptimizationPass) -> OptimizationResult;

    /// Validate the module
    fn validate(&self) -> Result<(), Vec<ValidationError>>;

    /// Get statistics about the module
    fn get_stats(&self) -> ModuleStats;
}

/// Statistics about an IR module
#[derive(Debug, Default)]
pub struct ModuleStats {
    pub functions: usize,
    pub blocks: usize,
    pub instructions: usize,
    pub globals: usize,
    pub types: usize,
}

/// Generic optimization function that can work with any IR level
pub fn optimize<M: OptimizableModule>(
    module: &mut M,
    passes: Vec<Box<dyn OptimizationPass>>,
    validate_after_each: bool,
) -> Result<OptimizationResult, Vec<ValidationError>> {
    let mut total_result = OptimizationResult::unchanged();

    println!(
        "Optimizing {} module: {}",
        module.ir_level(),
        module.module_name()
    );

    for mut pass in passes {
        let pass_name = pass.name();
        println!("  Running pass: {}", pass_name);

        let result = module.run_optimization(&mut *pass);

        if result.modified {
            println!(
                "    Modified: {} instructions eliminated, {} blocks eliminated",
                result.instructions_eliminated, result.blocks_eliminated
            );
        }

        total_result = total_result.combine(result);

        // Optionally validate after each pass
        if validate_after_each {
            module.validate()?;
        }
    }

    // Always validate at the end
    module.validate()?;

    Ok(total_result)
}

// Implementation for MIR (existing IrModule)
impl OptimizableModule for super::IrModule {
    fn ir_level(&self) -> &'static str {
        "MIR"
    }

    fn module_name(&self) -> &str {
        &self.name
    }

    fn run_optimization(&mut self, pass: &mut dyn OptimizationPass) -> OptimizationResult {
        pass.run_on_module(self)
    }

    fn validate(&self) -> Result<(), Vec<ValidationError>> {
        super::validation::validate_module(self)
    }

    fn get_stats(&self) -> ModuleStats {
        let mut stats = ModuleStats::default();
        stats.functions = self.functions.len();
        stats.globals = self.globals.len();

        for func in self.functions.values() {
            stats.blocks += func.cfg.blocks.len();
            for block in func.cfg.blocks.values() {
                stats.instructions += block.instructions.len();
            }
        }

        stats
    }
}

// Implementation for HIR
impl OptimizableModule for super::hir::HirModule {
    fn ir_level(&self) -> &'static str {
        "HIR"
    }

    fn module_name(&self) -> &str {
        &self.name
    }

    fn run_optimization(&mut self, pass: &mut dyn OptimizationPass) -> OptimizationResult {
        // HIR-specific optimization
        // For now, we'll use a simple adapter pattern
        // In practice, we'd have HIR-specific optimization passes

        // For HIR, we need HIR-specific passes
        // Standard MIR passes won't work on HIR
        // This is a placeholder - in practice, we'd have HIR-specific passes
        OptimizationResult::unchanged()
    }

    fn validate(&self) -> Result<(), Vec<ValidationError>> {
        // HIR validation
        validate_hir(self)
    }

    fn get_stats(&self) -> ModuleStats {
        let mut stats = ModuleStats::default();
        stats.functions = self.functions.len();
        stats.types = self.types.len();
        stats.globals = self.globals.len();

        // Count blocks and statements in functions
        for func in self.functions.values() {
            if let Some(body) = &func.body {
                stats.blocks += 1; // The body itself is a block
                stats.instructions += count_statements_in_block(body);
            }
        }

        stats
    }
}

/// Trait for HIR-specific optimization passes
pub trait HirOptimizationPass: OptimizationPass {
    /// Optimize a HIR module
    fn optimize_hir(&mut self, module: &mut super::hir::HirModule) -> OptimizationResult;
}

/// HIR optimization pass: dead code elimination using call graph analysis
pub struct HirDeadCodeElimination<'a> {
    /// Optional call graph for more accurate analysis
    call_graph: Option<&'a crate::semantic_graph::CallGraph>,
    /// Entry points to preserve
    entry_points: std::collections::HashSet<crate::tast::SymbolId>,
}

impl<'a> HirDeadCodeElimination<'a> {
    /// Create a new dead code elimination pass
    pub fn new() -> Self {
        Self {
            call_graph: None,
            entry_points: std::collections::HashSet::new(),
        }
    }

    /// Set the call graph for analysis
    pub fn with_call_graph(mut self, call_graph: &'a crate::semantic_graph::CallGraph) -> Self {
        self.call_graph = Some(call_graph);
        self
    }

    /// Add an entry point to preserve
    pub fn add_entry_point(mut self, symbol: crate::tast::SymbolId) -> Self {
        self.entry_points.insert(symbol);
        self
    }
}

impl<'a> OptimizationPass for HirDeadCodeElimination<'a> {
    fn name(&self) -> &'static str {
        "hir-dead-code-elimination"
    }

    fn run_on_module(&mut self, _module: &mut super::IrModule) -> OptimizationResult {
        // This is for MIR, not used for HIR
        OptimizationResult::unchanged()
    }
}

impl<'a> HirOptimizationPass for HirDeadCodeElimination<'a> {
    fn optimize_hir(&mut self, module: &mut super::hir::HirModule) -> OptimizationResult {
        let mut result = OptimizationResult::unchanged();

        // Collect all reachable functions
        let mut reachable_functions = std::collections::HashSet::new();

        // If we have a call graph, use it for precise analysis
        if let Some(call_graph) = self.call_graph {
            // Start from entry points and exported functions
            for func in module.functions.values() {
                if func.is_entry_point() {
                    self.entry_points.insert(func.symbol_id);
                }
            }

            // Find all reachable functions from entry points
            for &entry_point in &self.entry_points {
                if call_graph.functions.contains(&entry_point) {
                    let reachable = call_graph.reachable_functions(entry_point);
                    reachable_functions.extend(reachable);
                }
            }
        } else {
            // Fallback: collect all function calls from all function bodies
            for func in module.functions.values() {
                if let Some(body) = &func.body {
                    collect_called_functions(body, &mut reachable_functions);
                }
            }

            // Add entry points
            for func in module.functions.values() {
                if func.is_entry_point() {
                    reachable_functions.insert(func.symbol_id);
                }
            }
        }

        // Remove unreachable functions
        let original_count = module.functions.len();
        module
            .functions
            .retain(|symbol_id, _func| reachable_functions.contains(symbol_id));

        let removed = original_count - module.functions.len();
        if removed > 0 {
            result.modified = true;
            result.instructions_eliminated = removed;
            eprintln!("HIR DCE: Removed {} unreachable functions", removed);
        }

        result
    }
}

/// Helper function to count statements in a HIR block
fn count_statements_in_block(block: &super::hir::HirBlock) -> usize {
    let mut count = block.statements.len();
    if block.expr.is_some() {
        count += 1;
    }

    // Recursively count nested blocks
    for stmt in &block.statements {
        use super::hir::HirStatement;
        count += match stmt {
            HirStatement::If {
                then_branch,
                else_branch,
                ..
            } => {
                count_statements_in_block(then_branch)
                    + else_branch
                        .as_ref()
                        .map_or(0, |b| count_statements_in_block(b))
            }
            HirStatement::While { body, .. }
            | HirStatement::DoWhile { body, .. }
            | HirStatement::ForIn { body, .. } => count_statements_in_block(body),
            HirStatement::Label { block, .. } => count_statements_in_block(block),
            HirStatement::TryCatch {
                try_block,
                catches,
                finally_block,
            } => {
                count_statements_in_block(try_block)
                    + catches
                        .iter()
                        .map(|c| count_statements_in_block(&c.body))
                        .sum::<usize>()
                    + finally_block
                        .as_ref()
                        .map_or(0, |b| count_statements_in_block(b))
            }
            HirStatement::Switch { cases, .. } => cases
                .iter()
                .map(|c| count_statements_in_block(&c.body))
                .sum(),
            _ => 0,
        };
    }

    count
}

/// Helper function to collect called functions from a HIR block
fn collect_called_functions(
    block: &super::hir::HirBlock,
    called: &mut std::collections::HashSet<crate::tast::SymbolId>,
) {
    // Check expressions in the block
    if let Some(expr) = &block.expr {
        collect_called_functions_in_expr(expr, called);
    }

    // Check statements
    for stmt in &block.statements {
        use super::hir::HirStatement;
        match stmt {
            HirStatement::Expr(expr)
            | HirStatement::Return(Some(expr))
            | HirStatement::Throw(expr) => {
                collect_called_functions_in_expr(expr, called);
            }
            HirStatement::Let {
                init: Some(expr), ..
            }
            | HirStatement::Assign { rhs: expr, .. } => {
                collect_called_functions_in_expr(expr, called);
            }
            HirStatement::If {
                condition,
                then_branch,
                else_branch,
            } => {
                collect_called_functions_in_expr(condition, called);
                collect_called_functions(then_branch, called);
                if let Some(else_b) = else_branch {
                    collect_called_functions(else_b, called);
                }
            }
            HirStatement::While {
                condition,
                body,
                continue_update,
                ..
            } => {
                collect_called_functions_in_expr(condition, called);
                collect_called_functions(body, called);
                if let Some(update) = continue_update {
                    collect_called_functions(update, called);
                }
            }
            HirStatement::DoWhile {
                body, condition, ..
            } => {
                collect_called_functions_in_expr(condition, called);
                collect_called_functions(body, called);
            }
            HirStatement::ForIn { iterator, body, .. } => {
                collect_called_functions_in_expr(iterator, called);
                collect_called_functions(body, called);
            }
            HirStatement::Switch { scrutinee, cases } => {
                collect_called_functions_in_expr(scrutinee, called);
                for case in cases {
                    if let Some(guard) = &case.guard {
                        collect_called_functions_in_expr(guard, called);
                    }
                    collect_called_functions(&case.body, called);
                }
            }
            HirStatement::TryCatch {
                try_block,
                catches,
                finally_block,
            } => {
                collect_called_functions(try_block, called);
                for catch in catches {
                    collect_called_functions(&catch.body, called);
                }
                if let Some(finally) = finally_block {
                    collect_called_functions(finally, called);
                }
            }
            HirStatement::Label { block, .. } => {
                collect_called_functions(block, called);
            }
            _ => {}
        }
    }
}

/// Helper function to collect called functions from a HIR expression
fn collect_called_functions_in_expr(
    expr: &super::hir::HirExpr,
    called: &mut std::collections::HashSet<crate::tast::SymbolId>,
) {
    use super::hir::HirExprKind;

    match &expr.kind {
        HirExprKind::Call { callee, args, .. } => {
            // Check if the callee is a direct function reference
            if let HirExprKind::Variable { symbol, .. } = &callee.kind {
                called.insert(*symbol);
            }
            collect_called_functions_in_expr(callee, called);
            for arg in args {
                collect_called_functions_in_expr(arg, called);
            }
        }
        HirExprKind::Lambda { body, .. } => {
            collect_called_functions_in_expr(body, called);
        }
        HirExprKind::Block(block) => {
            collect_called_functions(block, called);
        }
        HirExprKind::If {
            condition,
            then_expr,
            else_expr,
        } => {
            collect_called_functions_in_expr(condition, called);
            collect_called_functions_in_expr(then_expr, called);
            collect_called_functions_in_expr(else_expr, called);
        }
        HirExprKind::Binary { lhs, rhs, .. }
        | HirExprKind::Index {
            object: lhs,
            index: rhs,
        } => {
            collect_called_functions_in_expr(lhs, called);
            collect_called_functions_in_expr(rhs, called);
        }
        HirExprKind::Unary { operand, .. }
        | HirExprKind::Field {
            object: operand, ..
        }
        | HirExprKind::Cast { expr: operand, .. }
        | HirExprKind::Untyped(operand) => {
            collect_called_functions_in_expr(operand, called);
        }
        HirExprKind::Array { elements } => {
            for elem in elements {
                collect_called_functions_in_expr(elem, called);
            }
        }
        HirExprKind::Map { entries } => {
            for (key, value) in entries {
                collect_called_functions_in_expr(key, called);
                collect_called_functions_in_expr(value, called);
            }
        }
        HirExprKind::ObjectLiteral { fields } => {
            for (_, value) in fields {
                collect_called_functions_in_expr(value, called);
            }
        }
        HirExprKind::New { args, .. } => {
            for arg in args {
                collect_called_functions_in_expr(arg, called);
            }
        }
        _ => {}
    }
}

/// Basic HIR validation
fn validate_hir(module: &super::hir::HirModule) -> Result<(), Vec<ValidationError>> {
    let mut errors = Vec::new();

    // Check that all functions have unique names
    let mut function_names = std::collections::HashSet::new();
    for func in module.functions.values() {
        if !function_names.insert(&func.name) {
            errors.push(ValidationError {
                kind: super::validation::ValidationErrorKind::InvalidOperand {
                    instruction: format!("function {}", func.name),
                    reason: format!("Duplicate function name: {}", func.name),
                },
                function: None,
                block: None,
                instruction: Some(format!("function {}", func.name)),
            });
        }
    }

    // HIR doesn't track main function requirement yet
    // This would be added when HIR becomes more feature-complete

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}
