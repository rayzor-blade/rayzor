//! Complete Haxe compilation pipeline: Source -> AST -> TAST -> HIR
//!
//! This module provides the main compilation pipeline that takes Haxe source code
//! and transforms it through the following stages:
//! 1. Parse source code to AST using the enhanced parser
//! 2. Lower AST to TAST with type checking and semantic analysis
//! 3. Validate the resulting TAST for correctness
//! 4. Generate semantic graphs for advanced analysis
//! 5. Lower TAST to HIR (High-level Intermediate Representation)
//! 6. Optimize HIR for target platform
//!

use log::{debug, error, info, warn};

use crate::error_codes::error_registry;
use crate::ir::{
    hir::HirModule,
    hir_to_mir::lower_hir_to_mir,
    optimizable::{optimize, OptimizableModule},
    optimization::{OptimizationResult, PassManager},
    tast_to_hir::lower_tast_to_hir,
    validation::{validate_module, ValidationError},
    IrModule,
};
use crate::semantic_graph::{builder::CfgBuilder, GraphConstructionOptions, SemanticGraphs};
use crate::tast::type_flow_guard::{FlowSafetyError, FlowSafetyResults, TypeFlowGuard};
use crate::tast::{
    node::{FileMetadata, SafetyMode, TypedFile},
    string_intern::{InternedString, StringInterner},
    SourceLocation, SymbolId, SymbolTable, TypeId, TypeTable,
};

// Use the parser's public interface
use parser::{haxe_ast::HaxeFile, parse_haxe_file_with_diagnostics, ParseResult};

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;
use std::{rc::Rc, sync::Arc};

/// Main compilation pipeline for Haxe source code
pub struct HaxeCompilationPipeline {
    /// String interner shared across compilation units
    string_interner: Rc<RefCell<StringInterner>>,

    /// Pipeline configuration
    pub(crate) config: PipelineConfig,

    /// Compilation statistics
    stats: PipelineStats,
}

/// Configuration for the compilation pipeline
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Enable detailed type checking
    pub strict_type_checking: bool,

    /// Enable lifetime analysis
    pub enable_lifetime_analysis: bool,

    /// Enable ownership tracking (required for memory safety)
    pub enable_ownership_analysis: bool,

    /// Enable borrow checking (required for memory safety)
    pub enable_borrow_checking: bool,

    /// Enable hot reload support (for development builds)
    pub enable_hot_reload: bool,

    /// Optimization level (0 = debug, 1 = basic, 2 = aggressive)
    pub optimization_level: u8,

    /// Collect detailed statistics
    pub collect_statistics: bool,

    /// Maximum number of errors before stopping
    pub max_errors: usize,

    /// Target execution mode for compilation
    pub target_platform: TargetPlatform,

    /// Enable colored error output
    pub enable_colored_errors: bool,

    /// Enable semantic graph generation
    pub enable_semantic_analysis: bool,

    /// Enable HIR lowering phase
    pub enable_hir_lowering: bool,

    /// Enable HIR optimization passes
    pub enable_hir_optimization: bool,

    /// Enable HIR validation
    pub enable_hir_validation: bool,

    /// Enable MIR lowering phase
    pub enable_mir_lowering: bool,

    /// Enable MIR optimization passes
    pub enable_mir_optimization: bool,

    /// Enable basic flow-sensitive analysis during type checking
    pub enable_flow_sensitive_analysis: bool,

    /// Enable enhanced flow analysis with CFG/DFG integration
    pub enable_enhanced_flow_analysis: bool,

    /// Enable memory safety analysis (lifetime and ownership)
    pub enable_memory_safety_analysis: bool,

    /// Enable macro expansion between parsing and TAST lowering
    pub enable_macro_expansion: bool,
}

/// Target execution modes for the hybrid VM/compiler system
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetPlatform {
    /// Direct interpretation for fastest iteration during development
    Interpreter,

    /// Cranelift JIT compilation for fast compile times and good performance
    CraneliftJIT,

    /// LLVM AOT compilation for maximum performance in shipping builds
    LLVM,

    /// WebAssembly target for browser and universal deployment
    WebAssembly,

    /// Legacy transpilation targets (for compatibility)
    Legacy(LegacyTarget),
}

/// Legacy transpilation targets from traditional Haxe
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LegacyTarget {
    JavaScript,
    Neko,
    HashLink,
    Cpp,
    Java,
    CSharp,
    Python,
    Lua,
}

/// Statistics collected during compilation
#[derive(Debug, Clone, Default)]
pub struct PipelineStats {
    /// Number of files processed
    pub files_processed: usize,

    /// Total lines of code
    pub total_loc: usize,

    /// Parse time in microseconds
    pub parse_time_us: u64,

    /// Macro expansion time in microseconds
    pub macro_expansion_time_us: u64,

    /// AST lowering time in microseconds
    pub lowering_time_us: u64,

    /// Type checking time in microseconds
    pub type_checking_time_us: u64,

    /// Total compilation time in microseconds
    pub total_time_us: u64,

    /// Number of warnings generated
    pub warning_count: usize,

    /// Number of errors encountered
    pub error_count: usize,

    /// Semantic analysis time in microseconds
    pub semantic_analysis_time_us: u64,

    /// HIR lowering time in microseconds
    pub hir_lowering_time_us: u64,

    /// HIR optimization time in microseconds
    pub hir_optimization_time_us: u64,

    /// HIR validation time in microseconds
    pub hir_validation_time_us: u64,

    /// MIR lowering time in microseconds
    pub mir_lowering_time_us: u64,

    /// MIR optimization time in microseconds
    pub mir_optimization_time_us: u64,

    /// Flow-sensitive analysis time in microseconds
    pub flow_analysis_time_us: u64,

    /// Enhanced flow analysis time in microseconds
    pub enhanced_flow_analysis_time_us: u64,

    /// Memory safety analysis time in microseconds
    pub memory_safety_analysis_time_us: u64,

    /// Memory usage statistics
    pub memory_stats: MemoryStats,
}

/// Memory usage statistics
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Peak memory usage in bytes
    pub peak_memory_bytes: usize,

    /// AST size in bytes
    pub ast_size_bytes: usize,

    /// TAST size in bytes
    pub tast_size_bytes: usize,

    /// String interner size in bytes
    pub string_interner_bytes: usize,

    /// HIR size in bytes
    pub hir_size_bytes: usize,
}

/// Result of compilation pipeline
#[derive(Debug, Clone)]
pub struct CompilationResult {
    /// Successfully compiled TAST files
    pub typed_files: Vec<TypedFile>,

    /// Successfully lowered HIR modules (high-level IR)
    pub hir_modules: Vec<Arc<HirModule>>,

    /// Successfully lowered MIR modules (mid-level IR in SSA form)
    pub mir_modules: Vec<Arc<IrModule>>,

    /// Semantic analysis results
    pub semantic_graphs: Vec<Arc<SemanticGraphs>>,

    /// Compilation errors encountered
    pub errors: Vec<CompilationError>,

    /// Compilation warnings
    pub warnings: Vec<CompilationWarning>,

    /// Pipeline statistics
    pub stats: PipelineStats,
}

/// Compilation error with detailed information
#[derive(Debug, Clone)]
pub struct CompilationError {
    /// Error message
    pub message: String,

    /// Source location of the error
    pub location: SourceLocation,

    /// Error category
    pub category: ErrorCategory,

    /// Optional suggestion for fixing the error
    pub suggestion: Option<String>,

    /// Related errors (for cascading issues)
    pub related_errors: Vec<String>,
}

impl CompilationError {
    /// Convert to a standard Diagnostic for formatted output
    pub fn to_diagnostic(&self, source_map: &diagnostics::SourceMap) -> diagnostics::Diagnostic {
        use diagnostics::{
            Diagnostic, DiagnosticSeverity, FileId, Label, SourcePosition, SourceSpan,
        };

        let file_id = FileId::new(self.location.file_id as usize);

        // Calculate line/column from byte offset using SourceMap
        // Try to underline the whole identifier/token, not just one character
        let (start_pos, end_pos) = if let Some(pos) =
            source_map.offset_to_position(file_id, self.location.byte_offset as usize)
        {
            // Estimate token length from the error message - look for variable/field names in quotes
            // e.g., "Use after move: variable 'myResource' was moved" -> extract 'myResource'
            let token_len = self
                .message
                .split('\'')
                .nth(1) // Get first quoted string
                .map(|name| name.len())
                .unwrap_or(1) // Default to 1 if no quoted name found
                .max(1); // Ensure at least 1

            let end_offset = self.location.byte_offset as usize + token_len;
            let end_pos = source_map
                .offset_to_position(file_id, end_offset)
                .unwrap_or_else(|| {
                    SourcePosition::new(pos.line, pos.column + token_len, end_offset)
                });
            (pos, end_pos)
        } else {
            // Fallback if offset calculation fails
            let start_pos = SourcePosition::new(
                self.location.line as usize,
                self.location.column as usize,
                self.location.byte_offset as usize,
            );
            let end_pos = SourcePosition::new(
                self.location.line as usize,
                (self.location.column + 1) as usize,
                (self.location.byte_offset + 1) as usize,
            );
            (start_pos, end_pos)
        };

        let span = SourceSpan::new(start_pos, end_pos, file_id);

        // Split multi-line suggestions into separate help items for better color coding
        let help_items = self
            .suggestion
            .as_ref()
            .map(|s| {
                s.lines()
                    .filter(|line| !line.trim().is_empty())
                    .map(|line| line.trim().to_string())
                    .collect::<Vec<String>>()
            })
            .unwrap_or_default();

        Diagnostic {
            severity: DiagnosticSeverity::Error,
            code: Some(self.category.error_code().to_string()),
            message: self.message.clone(),
            span: span.clone(),
            labels: vec![Label::primary(span, self.message.clone())],
            suggestions: vec![],
            notes: self.related_errors.clone(),
            help: help_items,
        }
    }
}

/// Compilation warning
#[derive(Debug, Clone)]
pub struct CompilationWarning {
    /// Warning message
    pub message: String,

    /// Source location of the warning
    pub location: SourceLocation,

    /// Warning category
    pub category: WarningCategory,

    /// Whether this warning can be suppressed
    pub suppressible: bool,
}

impl CompilationWarning {
    /// Convert to a standard Diagnostic for formatted output
    pub fn to_diagnostic(&self, _source_map: &diagnostics::SourceMap) -> diagnostics::Diagnostic {
        use diagnostics::{
            Diagnostic, DiagnosticSeverity, FileId, Label, SourcePosition, SourceSpan,
        };

        // Create span from source location
        let start_pos = SourcePosition::new(
            self.location.line as usize,
            self.location.column as usize,
            self.location.byte_offset as usize,
        );
        let end_pos = SourcePosition::new(
            self.location.line as usize,
            (self.location.column + 1) as usize,
            (self.location.byte_offset + 1) as usize,
        );
        let span = SourceSpan::new(
            start_pos,
            end_pos,
            FileId::new(self.location.file_id as usize),
        );

        Diagnostic {
            severity: DiagnosticSeverity::Warning,
            code: Some(format!("{:?}", self.category)),
            message: self.message.clone(),
            span: span.clone(),
            labels: vec![Label::primary(span, self.message.clone())],
            suggestions: vec![],
            notes: vec![],
            help: vec![],
        }
    }
}

/// Categories of compilation errors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    /// Syntax error in source code
    ParseError,

    /// Type error (type mismatch, undefined type, etc.)
    TypeError,

    /// Symbol resolution error (undefined variable, etc.)
    SymbolError,

    /// Ownership/borrowing error
    OwnershipError,

    /// Lifetime error
    LifetimeError,

    /// Concurrency safety error (Send/Sync violations)
    ConcurrencyError,

    /// Import/module error
    ImportError,

    /// HIR lowering error
    HIRLoweringError,

    /// HIR optimization error
    HIROptimizationError,

    /// HIR validation error
    HIRValidationError,

    /// Semantic analysis error
    SemanticAnalysisError,

    /// Macro expansion error
    MacroExpansionError,

    /// Internal compiler error
    InternalError,
}

impl ErrorCategory {
    /// Get the error code for this category
    ///
    /// Error code ranges:
    /// - E0001-E0099: Parser/syntax errors (defined in diagnostics/haxe.rs)
    /// - E0100-E0199: Type system errors
    /// - E0200-E0299: Symbol resolution errors
    /// - E0300-E0399: Ownership/lifetime/concurrency errors
    /// - E0400-E0499: Import/module errors
    /// - E0500-E0599: HIR errors
    /// - E0600-E0699: Semantic analysis errors
    /// - E0700-E0799: Macro expansion errors
    /// - E9999: Internal compiler errors
    pub fn error_code(&self) -> &'static str {
        match self {
            // Parser errors use E0001-E0099 range (delegated to parser)
            ErrorCategory::ParseError => "E0001",

            // Type system errors: E0100-E0199
            ErrorCategory::TypeError => "E0100",

            // Symbol resolution errors: E0200-E0299
            ErrorCategory::SymbolError => "E0200",

            // Ownership/lifetime/concurrency errors: E0300-E0399
            ErrorCategory::OwnershipError => "E0300",
            ErrorCategory::LifetimeError => "E0301",
            ErrorCategory::ConcurrencyError => "E0302",

            // Import/module errors: E0400-E0499
            ErrorCategory::ImportError => "E0400",

            // HIR errors: E0500-E0599
            ErrorCategory::HIRLoweringError => "E0500",
            ErrorCategory::HIROptimizationError => "E0501",
            ErrorCategory::HIRValidationError => "E0502",

            // Semantic analysis errors: E0600-E0699
            ErrorCategory::SemanticAnalysisError => "E0600",

            // Macro expansion errors: E0700-E0799
            ErrorCategory::MacroExpansionError => "E0700",

            // Internal compiler errors
            ErrorCategory::InternalError => "E9999",
        }
    }
}

/// Categories of compilation warnings
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarningCategory {
    /// Unused variable, function, etc.
    UnusedCode,

    /// Deprecated feature usage
    Deprecated,

    /// Potential performance issue
    Performance,

    /// Style/convention warning
    Style,

    /// Potential correctness issue
    Correctness,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            strict_type_checking: true,
            enable_lifetime_analysis: true,
            enable_ownership_analysis: true,
            enable_borrow_checking: true,
            enable_hot_reload: false,
            optimization_level: 1,
            collect_statistics: true,
            max_errors: 100,
            target_platform: TargetPlatform::CraneliftJIT,
            enable_colored_errors: true,
            enable_semantic_analysis: true,
            enable_hir_lowering: true,
            enable_hir_optimization: true,
            enable_hir_validation: true,
            enable_mir_lowering: true,
            enable_mir_optimization: true,
            enable_flow_sensitive_analysis: true,
            enable_enhanced_flow_analysis: true,
            enable_memory_safety_analysis: true,
            enable_macro_expansion: true,
        }
    }
}

impl PipelineConfig {
    /// Disable all analysis passes that don't affect code generation.
    /// Skips flow analysis, ownership/lifetime/borrow checking, semantic
    /// analysis, HIR validation, macro expansion, and statistics collection.
    /// Use `rayzor check` to run analysis separately.
    pub fn skip_analysis(mut self) -> Self {
        self.enable_lifetime_analysis = false;
        self.enable_ownership_analysis = false;
        self.enable_borrow_checking = false;
        self.enable_semantic_analysis = false;
        self.enable_hir_validation = false;
        self.enable_flow_sensitive_analysis = false;
        self.enable_enhanced_flow_analysis = false;
        self.enable_memory_safety_analysis = false;
        // NOTE: macro expansion is NOT disabled — it's a correctness feature, not analysis
        self.collect_statistics = false;
        self
    }

    /// Configuration for development builds with hot reload
    pub fn development() -> Self {
        Self {
            strict_type_checking: true,
            enable_lifetime_analysis: true,
            enable_ownership_analysis: true,
            enable_borrow_checking: true,
            enable_hot_reload: true,
            optimization_level: 0,
            collect_statistics: true,
            max_errors: 100,
            target_platform: TargetPlatform::Interpreter,
            enable_colored_errors: true,
            enable_semantic_analysis: true,
            enable_hir_lowering: false, // Skip HIR for interpreter mode
            enable_hir_optimization: false,
            enable_hir_validation: false,
            enable_mir_lowering: false, // Skip MIR for interpreter mode
            enable_mir_optimization: false,
            enable_flow_sensitive_analysis: true, // Keep flow analysis for safety
            enable_enhanced_flow_analysis: false,
            enable_memory_safety_analysis: true,
            enable_macro_expansion: true,
        }
    }

    /// Configuration for release builds with maximum performance
    pub fn release() -> Self {
        Self {
            strict_type_checking: true,
            enable_lifetime_analysis: true,
            enable_ownership_analysis: true,
            enable_borrow_checking: true,
            enable_hot_reload: false,
            optimization_level: 2,
            collect_statistics: false,
            max_errors: 100,
            target_platform: TargetPlatform::LLVM,
            enable_colored_errors: true,
            enable_semantic_analysis: true,
            enable_hir_lowering: true,
            enable_hir_optimization: true,
            enable_hir_validation: true,
            enable_mir_lowering: true,
            enable_mir_optimization: true,
            enable_flow_sensitive_analysis: true,
            enable_enhanced_flow_analysis: true,
            enable_memory_safety_analysis: true,
            enable_macro_expansion: true,
        }
    }

    /// Configuration for WebAssembly builds
    pub fn webassembly() -> Self {
        Self {
            strict_type_checking: true,
            enable_lifetime_analysis: true,
            enable_ownership_analysis: true,
            enable_borrow_checking: true,
            enable_hot_reload: false,
            optimization_level: 2,
            collect_statistics: false,
            max_errors: 100,
            target_platform: TargetPlatform::WebAssembly,
            enable_colored_errors: false, // No colors for web output
            enable_semantic_analysis: true,
            enable_hir_lowering: true,
            enable_hir_optimization: true,
            enable_hir_validation: true,
            enable_mir_lowering: true,
            enable_mir_optimization: true,
            enable_flow_sensitive_analysis: true,
            enable_enhanced_flow_analysis: true,
            enable_memory_safety_analysis: true,
            enable_macro_expansion: true,
        }
    }

    /// Set whether to enable colored error output
    pub fn with_colored_errors(mut self, enabled: bool) -> Self {
        self.enable_colored_errors = enabled;
        self
    }
}

impl HaxeCompilationPipeline {
    /// Create a new compilation pipeline with default configuration
    pub fn new() -> Self {
        Self::with_config(PipelineConfig::default())
    }

    /// Create a compilation pipeline with custom configuration
    pub fn with_config(config: PipelineConfig) -> Self {
        let string_interner = Rc::new(RefCell::new(StringInterner::new()));

        Self {
            string_interner,
            config,
            stats: PipelineStats::default(),
        }
    }

    /// Compile a single Haxe source file
    pub fn compile_file<P: AsRef<Path>>(
        &mut self,
        file_path: P,
        source: &str,
    ) -> CompilationResult {
        let start_time = std::time::Instant::now();
        let mut result = CompilationResult {
            typed_files: Vec::new(),
            hir_modules: Vec::new(),
            mir_modules: Vec::new(),
            semantic_graphs: Vec::new(),
            errors: Vec::new(),
            warnings: Vec::new(),
            stats: PipelineStats::default(),
        };

        // Stage 1: Parse source code to AST
        let parse_start = std::time::Instant::now();
        let parse_result = self.parse_source(file_path.as_ref(), source);
        self.stats.parse_time_us += parse_start.elapsed().as_micros() as u64;
        match parse_result {
            Ok((ast_file, source_map)) => {
                // Stage 1.5: Macro expansion (if enabled)
                let ast_file = if self.config.enable_macro_expansion {
                    let macro_start = std::time::Instant::now();
                    let expansion = crate::macro_system::expand_macros(ast_file);
                    self.stats.macro_expansion_time_us += macro_start.elapsed().as_micros() as u64;

                    // Collect macro diagnostics
                    for diag in &expansion.diagnostics {
                        match diag.severity {
                            crate::macro_system::MacroSeverity::Error => {
                                result.errors.push(diag.to_compilation_error());
                            }
                            crate::macro_system::MacroSeverity::Warning
                            | crate::macro_system::MacroSeverity::Info => {
                                result.warnings.push(diag.to_compilation_warning());
                            }
                        }
                    }

                    expansion.file
                } else {
                    ast_file
                };

                // Stage 2: Lower AST to TAST
                let lowering_start = std::time::Instant::now();
                match self.lower_ast_to_tast(ast_file, file_path.as_ref(), source, source_map) {
                    Ok((mut typed_file, lowering_errors, symbol_table, type_table, scope_tree)) => {
                        self.stats.lowering_time_us += lowering_start.elapsed().as_micros() as u64;

                        // Add any type errors from lowering/type checking
                        result.errors.extend(lowering_errors);

                        // Stage 2b: Detect program-level safety mode (check Main class for @:safety)
                        let program_safety_mode = typed_file.detect_program_safety_mode();

                        // Stage 2c: Validate strict mode if enabled
                        if let Some(SafetyMode::Strict) = program_safety_mode {
                            // In strict mode, all classes must have @:safety annotation
                            for class in &typed_file.classes {
                                if !class.has_safety_annotation() {
                                    let class_name = typed_file
                                        .get_string(class.name)
                                        .unwrap_or_else(|| "<unknown>".to_string());
                                    result.errors.push(CompilationError {
                                        message: format!(
                                            "Class '{}' must have @:safety annotation when program uses strict safety mode",
                                            class_name
                                        ),
                                        location: class.source_location,
                                        category: ErrorCategory::TypeError,
                                        suggestion: Some(format!(
                                            "Add @:safety annotation to class '{}' or use @:safety(strict=false) on Main class",
                                            class_name
                                        )),
                                        related_errors: Vec::new(),
                                    });
                                }
                            }
                        }

                        // Stage 3: Validate TAST
                        if let Err(validation_errors) = self.validate_tast(&typed_file) {
                            result.errors.extend(validation_errors);
                        }

                        // Stage 4: Generate semantic graphs (if enabled)
                        let semantic_graphs = if self.config.enable_semantic_analysis {
                            let semantic_start = std::time::Instant::now();
                            match self.build_semantic_graphs(
                                &typed_file,
                                &symbol_table,
                                &type_table,
                                &scope_tree,
                            ) {
                                Ok(graphs) => {
                                    self.stats.semantic_analysis_time_us +=
                                        semantic_start.elapsed().as_micros() as u64;

                                    // Stage 4b: Enhanced flow analysis with CFG/DFG integration (if enabled)
                                    if self.config.enable_enhanced_flow_analysis {
                                        let enhanced_flow_start = std::time::Instant::now();
                                        let enhanced_flow_errors = self.run_enhanced_flow_analysis(
                                            &typed_file,
                                            &graphs,
                                            &symbol_table,
                                            &type_table,
                                        );
                                        result.errors.extend(enhanced_flow_errors);
                                        self.stats.enhanced_flow_analysis_time_us +=
                                            enhanced_flow_start.elapsed().as_micros() as u64;
                                    }

                                    // Stage 4c: Memory safety analysis (if enabled)
                                    if self.config.enable_memory_safety_analysis {
                                        let memory_safety_start = std::time::Instant::now();
                                        let memory_safety_errors = self.run_memory_safety_analysis(
                                            &typed_file,
                                            &graphs,
                                            &symbol_table,
                                            &type_table,
                                        );

                                        // Check if we're in strict safety mode
                                        if let Some(SafetyMode::Strict) =
                                            typed_file.get_program_safety_mode()
                                        {
                                            // In strict mode, safety errors are fatal
                                            if !memory_safety_errors.is_empty() {
                                                result.errors.extend(memory_safety_errors);
                                                // Add a summary error
                                                result.errors.push(CompilationError {
                                                    message: "Compilation stopped: Safety violations in strict mode".to_string(),
                                                    location: SourceLocation::unknown(),
                                                    category: ErrorCategory::OwnershipError,
                                                    suggestion: Some("Fix all safety violations above, or use @:safety(false) for non-strict mode".to_string()),
                                                    related_errors: Vec::new(),
                                                });
                                                self.stats.memory_safety_analysis_time_us +=
                                                    memory_safety_start.elapsed().as_micros()
                                                        as u64;

                                                // Store partial results and return early
                                                result
                                                    .semantic_graphs
                                                    .push(Arc::new(graphs.clone()));
                                                result.typed_files.push(typed_file);
                                                return result;
                                            }
                                        } else if typed_file.uses_manual_memory() {
                                            // Non-strict mode: Display warnings but continue
                                            if !memory_safety_errors.is_empty() {
                                                warn!("\n⚠️  Memory Safety Warnings (non-strict mode):");
                                                warn!("   The following issues were found but compilation will continue.");
                                                warn!("   Unannotated classes will use ARC (atomic reference counting).\n");
                                                for err in &memory_safety_errors {
                                                    warn!(
                                                        "   {} at {}:{}",
                                                        err.message,
                                                        err.location.line,
                                                        err.location.column
                                                    );
                                                    if let Some(ref suggestion) = err.suggestion {
                                                        warn!("     Suggestion: {}", suggestion);
                                                    }
                                                }
                                                warn!("");
                                            }
                                            // Convert errors to warnings for reporting
                                            for err in memory_safety_errors {
                                                result.warnings.push(CompilationWarning {
                                                    message: err.message,
                                                    location: err.location,
                                                    category: WarningCategory::Correctness,
                                                    suppressible: false, // Safety warnings in non-strict mode are important
                                                });
                                            }
                                        } else {
                                            // Default mode: no safety analysis was run anyway
                                            result.errors.extend(memory_safety_errors);
                                        }

                                        self.stats.memory_safety_analysis_time_us +=
                                            memory_safety_start.elapsed().as_micros() as u64;
                                    }

                                    result.semantic_graphs.push(Arc::new(graphs.clone()));
                                    Some(graphs)
                                }
                                Err(semantic_errors) => {
                                    result.errors.extend(semantic_errors);
                                    None
                                }
                            }
                        } else {
                            None
                        };

                        // Stage 5: Lower TAST to HIR (if enabled)
                        // Skip HIR lowering if we have fatal errors
                        if self.config.enable_hir_lowering && result.errors.is_empty() {
                            let hir_start = std::time::Instant::now();
                            match self.lower_tast_to_hir(
                                &typed_file,
                                semantic_graphs.as_ref(),
                                &symbol_table,
                                &type_table,
                            ) {
                                Ok(hir_module) => {
                                    self.stats.hir_lowering_time_us +=
                                        hir_start.elapsed().as_micros() as u64;

                                    // Stage 6: Validate HIR (if enabled)
                                    if self.config.enable_hir_validation {
                                        let validation_start = std::time::Instant::now();
                                        if let Err(hir_validation_errors) =
                                            self.validate_hir(&hir_module)
                                        {
                                            result.errors.extend(hir_validation_errors);
                                        }
                                        self.stats.hir_validation_time_us +=
                                            validation_start.elapsed().as_micros() as u64;
                                    }

                                    // Stage 7: Optimize HIR (if enabled)
                                    let final_hir = if self.config.enable_hir_optimization {
                                        let opt_start = std::time::Instant::now();
                                        // Pass semantic graphs (including call graph) to optimizer
                                        let optimized =
                                            self.optimize_hir(hir_module, semantic_graphs.as_ref());
                                        self.stats.hir_optimization_time_us +=
                                            opt_start.elapsed().as_micros() as u64;
                                        optimized
                                    } else {
                                        hir_module
                                    };

                                    // Store HIR module
                                    let hir_arc = Arc::new(final_hir.clone());
                                    result.hir_modules.push(hir_arc);

                                    // Stage 8: Lower HIR to MIR (if enabled)
                                    if self.config.enable_mir_lowering {
                                        let mir_start = std::time::Instant::now();
                                        match self.lower_hir_to_mir(
                                            &final_hir,
                                            &type_table,
                                            &symbol_table,
                                            semantic_graphs.as_ref(),
                                            &typed_file,
                                        ) {
                                            Ok(mir_module) => {
                                                self.stats.mir_lowering_time_us +=
                                                    mir_start.elapsed().as_micros() as u64;

                                                // Stage 9: Optimize MIR (if enabled)
                                                let final_mir =
                                                    if self.config.enable_mir_optimization {
                                                        let opt_start = std::time::Instant::now();
                                                        let optimized =
                                                            self.optimize_mir(mir_module);
                                                        self.stats.mir_optimization_time_us +=
                                                            opt_start.elapsed().as_micros() as u64;
                                                        optimized
                                                    } else {
                                                        mir_module
                                                    };

                                                // Store MIR module
                                                result.mir_modules.push(Arc::new(final_mir));
                                            }
                                            Err(mir_errors) => {
                                                result.errors.extend(mir_errors);
                                            }
                                        }
                                    }
                                }
                                Err(hir_errors) => {
                                    result.errors.extend(hir_errors);
                                }
                            }
                        }

                        // Always add the typed file, even if there are type errors
                        // This allows constraint validation tests to work properly
                        result.typed_files.push(typed_file);
                    }
                    Err(fatal_lowering_errors) => {
                        // Only reach here for fatal errors that prevent TAST generation
                        result.errors.extend(fatal_lowering_errors);
                    }
                }
            }
            Err(parse_errors) => {
                result.errors.extend(parse_errors);
            }
        }

        // Update statistics
        self.stats.files_processed += 1;
        self.stats.total_loc += source.lines().count();
        self.stats.total_time_us += start_time.elapsed().as_micros() as u64;
        self.stats.error_count += result.errors.len();
        self.stats.warning_count += result.warnings.len();

        result.stats = self.stats.clone();
        result
    }

    /// Compile multiple Haxe source files
    pub fn compile_files<P: AsRef<Path>>(&mut self, files: &[(P, String)]) -> CompilationResult {
        use crate::compilation::{CompilationConfig, CompilationUnit};

        let start_time = std::time::Instant::now();
        let mut result = CompilationResult {
            typed_files: Vec::new(),
            hir_modules: Vec::new(),
            mir_modules: Vec::new(),
            semantic_graphs: Vec::new(),
            errors: Vec::new(),
            warnings: Vec::new(),
            stats: PipelineStats::default(),
        };

        // Use CompilationUnit which shares symbol table, type table, and
        // namespace resolver across all files — enabling cross-package resolution.
        let config = CompilationConfig::default();
        let mut unit = CompilationUnit::new(config);

        // Add all files to the compilation unit
        for (file_path, source) in files {
            let file_name = file_path.as_ref().to_str().unwrap_or("unknown");
            if let Err(e) = unit.add_file(source, file_name) {
                result.errors.push(CompilationError {
                    message: format!("Parse error: {}", e),
                    location: SourceLocation::unknown(),
                    category: ErrorCategory::ParseError,
                    suggestion: None,
                    related_errors: Vec::new(),
                });
            }
        }

        // Lower all files together with shared state
        let typed_files = match unit.lower_to_tast() {
            Ok(files) => files,
            Err(errors) => {
                result.errors.extend(errors);
                self.stats.total_time_us += start_time.elapsed().as_micros() as u64;
                result.stats = self.stats.clone();
                return result;
            }
        };

        // Run remaining pipeline stages on each typed file using the unit's shared state.
        // Downstream stages (semantic graphs, HIR, MIR) operate per-file, so
        // cross-file field/method references may not be fully resolvable —
        // failures in these stages are reported as warnings, not errors.
        for mut typed_file in typed_files {
            // Stage 2b: Detect program-level safety mode
            let _program_safety_mode = typed_file.detect_program_safety_mode();

            // Stage 3: Validate TAST
            if let Err(validation_errors) = self.validate_tast(&typed_file) {
                result.errors.extend(validation_errors);
            }

            // Stage 4: Semantic graphs (if enabled)
            let semantic_graphs = if self.config.enable_semantic_analysis {
                match self.build_semantic_graphs(
                    &typed_file,
                    &unit.symbol_table,
                    &unit.type_table,
                    &unit.scope_tree,
                ) {
                    Ok(graphs) => {
                        // Stage 4b: Enhanced flow analysis
                        if self.config.enable_enhanced_flow_analysis {
                            let flow_errors = self.run_enhanced_flow_analysis(
                                &typed_file,
                                &graphs,
                                &unit.symbol_table,
                                &unit.type_table,
                            );
                            for err in flow_errors {
                                result.warnings.push(CompilationWarning {
                                    message: err.message,
                                    location: err.location,
                                    category: WarningCategory::Correctness,
                                    suppressible: true,
                                });
                            }
                        }

                        result.semantic_graphs.push(Arc::new(graphs.clone()));
                        Some(graphs)
                    }
                    Err(semantic_errors) => {
                        for err in semantic_errors {
                            result.warnings.push(CompilationWarning {
                                message: err.message,
                                location: err.location,
                                category: WarningCategory::Correctness,
                                suppressible: true,
                            });
                        }
                        None
                    }
                }
            } else {
                None
            };

            // Stage 5: Lower TAST to HIR (if enabled and no fatal errors)
            if self.config.enable_hir_lowering && result.errors.is_empty() {
                match self.lower_tast_to_hir(
                    &typed_file,
                    semantic_graphs.as_ref(),
                    &unit.symbol_table,
                    &unit.type_table,
                ) {
                    Ok(hir_module) => {
                        // Stage 6: Validate HIR
                        if self.config.enable_hir_validation {
                            if let Err(hir_errors) = self.validate_hir(&hir_module) {
                                for err in hir_errors {
                                    result.warnings.push(CompilationWarning {
                                        message: err.message,
                                        location: err.location,
                                        category: WarningCategory::Correctness,
                                        suppressible: true,
                                    });
                                }
                            }
                        }

                        // Stage 7: Optimize HIR
                        let final_hir = if self.config.enable_hir_optimization {
                            self.optimize_hir(hir_module, semantic_graphs.as_ref())
                        } else {
                            hir_module
                        };

                        let hir_arc = Arc::new(final_hir.clone());
                        result.hir_modules.push(hir_arc);

                        // Stage 8: Lower HIR to MIR
                        if self.config.enable_mir_lowering {
                            match self.lower_hir_to_mir(
                                &final_hir,
                                &unit.type_table,
                                &unit.symbol_table,
                                semantic_graphs.as_ref(),
                                &typed_file,
                            ) {
                                Ok(mir_module) => {
                                    // Stage 9: Optimize MIR
                                    let final_mir = if self.config.enable_mir_optimization {
                                        self.optimize_mir(mir_module)
                                    } else {
                                        mir_module
                                    };
                                    result.mir_modules.push(Arc::new(final_mir));
                                }
                                Err(mir_errors) => {
                                    for err in mir_errors {
                                        result.warnings.push(CompilationWarning {
                                            message: err.message,
                                            location: err.location,
                                            category: WarningCategory::Correctness,
                                            suppressible: true,
                                        });
                                    }
                                }
                            }
                        }
                    }
                    Err(hir_errors) => {
                        for err in hir_errors {
                            result.warnings.push(CompilationWarning {
                                message: err.message,
                                location: err.location,
                                category: WarningCategory::Correctness,
                                suppressible: true,
                            });
                        }
                    }
                }
            }

            result.typed_files.push(typed_file);
        }

        // Update statistics
        self.stats.files_processed += files.len();
        self.stats.total_loc += files.iter().map(|(_, s)| s.lines().count()).sum::<usize>();
        self.stats.total_time_us += start_time.elapsed().as_micros() as u64;
        self.stats.error_count += result.errors.len();
        self.stats.warning_count += result.warnings.len();
        result.stats = self.stats.clone();
        result
    }

    /// Parse source code to AST and return both AST and SourceMap
    fn parse_source(
        &mut self,
        file_path: &Path,
        source: &str,
    ) -> Result<(HaxeFile, diagnostics::SourceMap), Vec<CompilationError>> {
        let file_name = file_path.to_str().unwrap_or("unknown");
        match parse_haxe_file_with_diagnostics(file_name, source) {
            Ok(parse_result) => {
                // Check if there are any errors in the diagnostics
                if parse_result.diagnostics.has_errors() {
                    let compilation_errors = parse_result
                        .diagnostics
                        .diagnostics
                        .into_iter()
                        .filter(|d| d.severity == diagnostics::DiagnosticSeverity::Error)
                        .map(|d| CompilationError {
                            message: d.message,
                            location: SourceLocation::new(
                                d.span.start.line as u32,
                                d.span.start.column as u32,
                                d.span.end.line as u32,
                                d.span.end.column as u32,
                            ),
                            category: ErrorCategory::ParseError,
                            suggestion: if d.help.is_empty() {
                                None
                            } else {
                                Some(d.help.join(" "))
                            },
                            related_errors: d.notes,
                        })
                        .collect();
                    Err(compilation_errors)
                } else {
                    Ok((parse_result.file, parse_result.source_map))
                }
            }
            Err(parse_error_str) => {
                let compilation_errors = vec![CompilationError {
                    message: format!("Parse error: {}", parse_error_str),
                    location: SourceLocation::new(0, 0, 0, 0), // Default location
                    category: ErrorCategory::ParseError,
                    suggestion: None,
                    related_errors: Vec::new(),
                }];
                Err(compilation_errors)
            }
        }
    }

    /// Lower AST to TAST with type checking
    fn lower_ast_to_tast(
        &mut self,
        ast_file: HaxeFile,
        file_path: &Path,
        source: &str,
        source_map: diagnostics::SourceMap,
    ) -> Result<
        (
            TypedFile,
            Vec<CompilationError>,
            SymbolTable,
            Rc<RefCell<TypeTable>>,
            crate::tast::ScopeTree,
        ),
        Vec<CompilationError>,
    > {
        use crate::tast::ast_lowering::AstLowering;
        use crate::tast::type_checking_pipeline::type_check_with_diagnostics;
        use crate::tast::{ScopeId, ScopeTree, SymbolTable, TypeTable};
        use diagnostics::ErrorFormatter;
        use std::cell::RefCell;

        // Create the necessary infrastructure for AST lowering
        // Estimate capacity based on AST size
        let estimated_symbols = (ast_file.declarations.len() * 20) // Rough estimate: 20 symbols per type
            .max(100); // Minimum 100 symbols

        let mut symbol_table = SymbolTable::with_capacity(estimated_symbols);
        let type_table = Rc::new(RefCell::new(TypeTable::with_capacity(estimated_symbols)));
        let mut scope_tree = ScopeTree::new(ScopeId::from_raw(0));

        // Use the source_map from the parser - it already has the file added
        // The parser has already added the file to the source map, so we don't need to add it again
        let file_name = file_path.to_str().unwrap_or("unknown");
        // Assume file_id is 0 since the parser adds it as the first file
        let file_id = diagnostics::FileId::new(0);

        // Now proceed with AST lowering using resolved types
        let mut binding = self.string_interner.borrow_mut();

        // Create namespace and import resolvers
        let mut namespace_resolver = crate::tast::namespace::NamespaceResolver::new();
        let mut import_resolver = crate::tast::namespace::ImportResolver::new();

        let mut lowering = AstLowering::new(
            &mut binding,
            Rc::clone(&self.string_interner),
            &mut symbol_table,
            &type_table,
            &mut scope_tree,
            &mut namespace_resolver,
            &mut import_resolver,
        );
        // Initialize span converter with proper filename
        lowering.initialize_span_converter_with_filename(
            file_id.as_usize() as u32,
            source.to_string(),
            file_name.to_string(),
        );

        // Lower the AST to TAST with error recovery
        // AstLowering now collects ALL errors within a file and continues processing
        // This allows us to report multiple errors per file in a single compilation
        let (mut typed_file, mut type_errors): (TypedFile, Vec<CompilationError>) =
            match lowering.lower_file(&ast_file) {
                Ok(typed_file) => {
                    // Collect any non-fatal errors accumulated during lowering
                    let non_fatal_errors: Vec<CompilationError> = lowering
                        .get_all_errors()
                        .iter()
                        .map(|err| {
                            let (message, suggestion) = self.extract_lowering_error_message(err);
                            CompilationError {
                                message,
                                location: self.extract_location_from_lowering_error(err),
                                category: self.categorize_lowering_error(err),
                                suggestion,
                                related_errors: Vec::new(),
                            }
                        })
                        .collect();
                    (typed_file, non_fatal_errors)
                }
                Err(_lowering_error) => {
                    // Extract ALL collected errors from the lowering context
                    let all_lowering_errors = lowering.get_all_errors();

                    // Convert all lowering errors to CompilationErrors
                    let compilation_errors: Vec<CompilationError> = all_lowering_errors
                        .iter()
                        .map(|err| {
                            let (message, suggestion) = self.extract_lowering_error_message(err);
                            CompilationError {
                                message,
                                location: self.extract_location_from_lowering_error(err),
                                category: self.categorize_lowering_error(err),
                                suggestion,
                                related_errors: Vec::new(),
                            }
                        })
                        .collect();

                    // Create a minimal TAST so we can continue pipeline and find more errors
                    // This allows us to report ALL errors, not just the first one
                    // IMPORTANT: Use the same string interner as the pipeline to ensure symbol names can be resolved
                    let minimal_tast = TypedFile::new(Rc::clone(&self.string_interner));

                    (minimal_tast, compilation_errors)
                }
            };

        // Run type checking with diagnostics
        let diagnostics = type_check_with_diagnostics(
            &mut typed_file,
            &type_table,
            &symbol_table,
            &scope_tree,
            &binding,
            &source_map,
        )
        .unwrap_or_else(|_| diagnostics::Diagnostics::new());

        // Add type checking errors to the collection (type_errors already initialized above with lowering errors)
        if !diagnostics.is_empty() {
            // Convert only Error-severity diagnostics to CompilationErrors.
            // Hints and warnings (e.g., dead code) are informational and
            // should not block compilation or count as errors.
            for diagnostic in diagnostics
                .diagnostics
                .iter()
                .filter(|d| matches!(d.severity, diagnostics::DiagnosticSeverity::Error))
            {
                // Extract the span location
                let location = SourceLocation {
                    file_id: diagnostic.span.file_id.as_usize() as u32,
                    line: diagnostic.span.start.line as u32,
                    column: diagnostic.span.start.column as u32,
                    byte_offset: diagnostic.span.start.byte_offset as u32,
                };

                // Build a self-contained message that includes error code, notes, and help text
                let mut message = diagnostic.message.clone();
                if let Some(ref code) = diagnostic.code {
                    message = format!("[{}] {}", code, message);
                }
                if !diagnostic.notes.is_empty() {
                    message = format!("{}\n{}", message, diagnostic.notes.join("\n"));
                }
                if !diagnostic.help.is_empty() {
                    message = format!("{}. {}", message, diagnostic.help.join(" "));
                }
                let compilation_error = CompilationError {
                    message,
                    location,
                    category: ErrorCategory::TypeError,
                    suggestion: if diagnostic.help.is_empty() {
                        None
                    } else {
                        Some(diagnostic.help.join(" "))
                    },
                    related_errors: diagnostic.notes.clone(),
                };

                type_errors.push(compilation_error);
            }
        }

        // Stage 2b: Basic flow-sensitive analysis (if enabled)
        if self.config.enable_flow_sensitive_analysis {
            let flow_start = std::time::Instant::now();
            let flow_errors = self.run_basic_flow_analysis(&typed_file, &symbol_table, &type_table);
            type_errors.extend(flow_errors);
            self.stats.flow_analysis_time_us += flow_start.elapsed().as_micros() as u64;
        }

        // Return the typed file along with any type errors and the symbol/type tables/scope tree for later stages
        Ok((
            typed_file,
            type_errors,
            symbol_table,
            type_table,
            scope_tree,
        ))
    }

    /// Validate the resulting TAST for correctness
    fn validate_tast(&self, typed_file: &TypedFile) -> Result<(), Vec<CompilationError>> {
        let mut errors = Vec::new();

        // Validate functions
        for function in &typed_file.functions {
            if let Err(function_errors) = self.validate_function(function) {
                errors.extend(function_errors);
            }
        }

        // Validate classes
        for class in &typed_file.classes {
            if let Err(class_errors) = self.validate_class(class) {
                errors.extend(class_errors);
            }
        }

        // Validate interfaces
        for interface in &typed_file.interfaces {
            if let Err(interface_errors) = self.validate_interface(interface) {
                errors.extend(interface_errors);
            }
        }

        // Validate enums
        for enum_def in &typed_file.enums {
            if let Err(enum_errors) = self.validate_enum(enum_def) {
                errors.extend(enum_errors);
            }
        }

        // Validate abstracts
        for abstract_def in &typed_file.abstracts {
            if let Err(abstract_errors) = self.validate_abstract(abstract_def) {
                errors.extend(abstract_errors);
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Extract package name from AST file
    fn extract_package_name(&self, ast_file: &HaxeFile) -> Option<String> {
        // Look for package declaration in AST
        // This is a simplified implementation
        ast_file.package.as_ref().map(|pkg| pkg.path.join("."))
    }

    /// Convert parser span to source location (placeholder implementation)
    fn convert_span_to_location(&self, line: u32, column: u32) -> SourceLocation {
        SourceLocation::new(line, column, line, column + 1)
    }

    /// Validate a function in the TAST
    fn validate_function(
        &self,
        function: &crate::tast::node::TypedFunction,
    ) -> Result<(), Vec<CompilationError>> {
        let mut errors = Vec::new();

        // Check function body consistency
        if function.body.is_empty() && !function.effects.is_pure {
            // Empty non-pure functions might be suspicious
        }

        // Validate parameter types
        for param in &function.parameters {
            if !self.is_valid_type_id(param.param_type) {
                errors.push(CompilationError {
                    message: format!(
                        "Invalid parameter type for '{}'",
                        self.get_string_from_interned(param.name)
                    ),
                    location: param.source_location,
                    category: ErrorCategory::TypeError,
                    suggestion: Some("Check that the type is properly defined".to_string()),
                    related_errors: Vec::new(),
                });
            }
        }

        // Validate return type
        if !self.is_valid_type_id(function.return_type) {
            errors.push(CompilationError {
                message: "Invalid return type".to_string(),
                location: function.source_location,
                category: ErrorCategory::TypeError,
                suggestion: Some("Check that the return type is properly defined".to_string()),
                related_errors: Vec::new(),
            });
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Validate a class in the TAST
    fn validate_class(
        &self,
        class: &crate::tast::node::TypedClass,
    ) -> Result<(), Vec<CompilationError>> {
        let mut errors = Vec::new();

        // Check for duplicate method names
        let mut method_names = std::collections::HashSet::new();
        for method in &class.methods {
            let method_name = self.get_string_from_interned(method.name);
            if !method_names.insert(method_name.clone()) {
                errors.push(CompilationError {
                    message: format!("Duplicate method name: '{}'", method_name),
                    location: method.source_location,
                    category: ErrorCategory::SymbolError,
                    suggestion: Some(
                        "Rename one of the methods or use method overloading".to_string(),
                    ),
                    related_errors: Vec::new(),
                });
            }
        }

        // Validate field types
        for field in &class.fields {
            if !self.is_valid_type_id(field.field_type) {
                errors.push(CompilationError {
                    message: format!("Invalid field type for '{}'", field.name),
                    location: field.source_location,
                    category: ErrorCategory::TypeError,
                    suggestion: Some("Check that the field type is properly defined".to_string()),
                    related_errors: Vec::new(),
                });
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Validate an interface in the TAST
    fn validate_interface(
        &self,
        interface: &crate::tast::node::TypedInterface,
    ) -> Result<(), Vec<CompilationError>> {
        let mut errors = Vec::new();

        // Check for duplicate method signatures
        let mut method_signatures = std::collections::HashSet::new();
        for method in &interface.methods {
            let signature = format!("{}:{}", method.name, "type"); // Simplified signature
            if !method_signatures.insert(signature.clone()) {
                errors.push(CompilationError {
                    message: format!("Duplicate method signature: '{}'", signature),
                    location: method.source_location,
                    category: ErrorCategory::SymbolError,
                    suggestion: Some("Remove duplicate method or change signature".to_string()),
                    related_errors: Vec::new(),
                });
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Validate an enum in the TAST
    fn validate_enum(
        &self,
        enum_def: &crate::tast::node::TypedEnum,
    ) -> Result<(), Vec<CompilationError>> {
        let mut errors = Vec::new();

        // Check for duplicate variant names
        let mut variant_names = std::collections::HashSet::new();
        for variant in &enum_def.variants {
            if !variant_names.insert(variant.name.clone()) {
                errors.push(CompilationError {
                    message: format!("Duplicate enum variant: '{}'", variant.name),
                    location: variant.source_location,
                    category: ErrorCategory::SymbolError,
                    suggestion: Some("Rename one of the variants".to_string()),
                    related_errors: Vec::new(),
                });
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Validate an abstract type in the TAST
    fn validate_abstract(
        &self,
        abstract_def: &crate::tast::node::TypedAbstract,
    ) -> Result<(), Vec<CompilationError>> {
        let mut errors = Vec::new();

        // Validate underlying type if present
        if let Some(underlying_type) = abstract_def.underlying_type {
            if !self.is_valid_type_id(underlying_type) {
                errors.push(CompilationError {
                    message: "Invalid underlying type for abstract".to_string(),
                    location: abstract_def.source_location,
                    category: ErrorCategory::TypeError,
                    suggestion: Some(
                        "Check that the underlying type is properly defined".to_string(),
                    ),
                    related_errors: Vec::new(),
                });
            }
        }

        // Validate from/to conversion types
        for &from_type in &abstract_def.from_types {
            if !self.is_valid_type_id(from_type) {
                errors.push(CompilationError {
                    message: "Invalid 'from' conversion type".to_string(),
                    location: abstract_def.source_location,
                    category: ErrorCategory::TypeError,
                    suggestion: Some(
                        "Check that the conversion type is properly defined".to_string(),
                    ),
                    related_errors: Vec::new(),
                });
            }
        }

        for &to_type in &abstract_def.to_types {
            if !self.is_valid_type_id(to_type) {
                errors.push(CompilationError {
                    message: "Invalid 'to' conversion type".to_string(),
                    location: abstract_def.source_location,
                    category: ErrorCategory::TypeError,
                    suggestion: Some(
                        "Check that the conversion type is properly defined".to_string(),
                    ),
                    related_errors: Vec::new(),
                });
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Check if a type ID is valid (placeholder implementation)
    fn is_valid_type_id(&self, type_id: TypeId) -> bool {
        // In a real implementation, this would check against a type table
        type_id.is_valid()
    }

    /// Get string from interned string (helper method)
    fn get_string_from_interned(&self, interned: crate::tast::InternedString) -> String {
        self.string_interner
            .borrow()
            .get(interned)
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("<invalid:#{}>", interned.as_raw()))
    }

    /// Extract raw message and suggestion from lowering error
    fn extract_lowering_error_message(
        &self,
        error: &crate::tast::ast_lowering::LoweringError,
    ) -> (String, Option<String>) {
        use crate::tast::ast_lowering::LoweringError;

        match error {
            LoweringError::UnresolvedSymbol { name, .. } => (
                format!("Cannot find symbol '{}'", name),
                Some("Make sure the symbol is declared and in scope".to_string()),
            ),

            LoweringError::UnresolvedType { type_name, .. } => (
                format!("Cannot find type '{}'", type_name),
                Some("Check that the type is imported or defined".to_string()),
            ),

            LoweringError::DuplicateSymbol { name, .. } => (
                format!("Duplicate definition of '{}'", name),
                Some("Rename one of the symbols or remove the duplicate definition".to_string()),
            ),

            LoweringError::GenericParameterError { message, .. } => (
                message.clone(),
                Some(
                    "Check the type definition to see how many type parameters it expects"
                        .to_string(),
                ),
            ),

            LoweringError::InvalidModifiers { modifiers, .. } => (
                format!("Invalid modifier combination: {}", modifiers.join(", ")),
                Some("Remove the conflicting modifiers".to_string()),
            ),

            LoweringError::TypeInferenceError { expression, .. } => (
                format!("Cannot infer type for expression '{}'", expression),
                Some("Try adding a type annotation like 'var x: Type = ...'".to_string()),
            ),

            LoweringError::LifetimeError { message, .. } => (
                format!("Lifetime error: {}", message),
                Some("Ensure that references don't outlive their referents".to_string()),
            ),

            LoweringError::OwnershipError { message, .. } => (
                format!("Ownership error: {}", message),
                Some("Check ownership constraints and borrowing rules".to_string()),
            ),

            LoweringError::IncompleteImplementation { feature, .. } => (
                format!("Feature not yet implemented: {}", feature),
                Some(
                    "Try using an alternative approach or wait for this feature to be implemented"
                        .to_string(),
                ),
            ),

            LoweringError::InternalError { message, .. } => (
                format!("Internal compiler error: {}", message),
                Some("Please report this issue to the compiler developers".to_string()),
            ),
        }
    }

    /// Categorize a lowering error into the correct ErrorCategory
    fn categorize_lowering_error(
        &self,
        error: &crate::tast::ast_lowering::LoweringError,
    ) -> ErrorCategory {
        use crate::tast::ast_lowering::LoweringError;

        match error {
            LoweringError::UnresolvedSymbol { .. } | LoweringError::DuplicateSymbol { .. } => {
                ErrorCategory::SymbolError
            }

            LoweringError::UnresolvedType { .. }
            | LoweringError::GenericParameterError { .. }
            | LoweringError::TypeInferenceError { .. } => ErrorCategory::TypeError,

            LoweringError::LifetimeError { .. } => ErrorCategory::LifetimeError,

            LoweringError::OwnershipError { .. } => ErrorCategory::OwnershipError,

            LoweringError::InvalidModifiers { .. } => ErrorCategory::TypeError,

            LoweringError::IncompleteImplementation { .. }
            | LoweringError::InternalError { .. } => ErrorCategory::InternalError,
        }
    }

    /// Extract source location from lowering error
    fn extract_location_from_lowering_error(
        &self,
        error: &crate::tast::ast_lowering::LoweringError,
    ) -> crate::tast::SourceLocation {
        use crate::tast::ast_lowering::LoweringError;

        match error {
            LoweringError::UnresolvedSymbol { location, .. } => location.clone(),
            LoweringError::UnresolvedType { location, .. } => location.clone(),
            LoweringError::DuplicateSymbol {
                duplicate_location, ..
            } => duplicate_location.clone(),
            LoweringError::InvalidModifiers { location, .. } => location.clone(),
            LoweringError::InternalError { location, .. } => location.clone(),
            LoweringError::GenericParameterError { location, .. } => location.clone(),
            LoweringError::TypeInferenceError { location, .. } => location.clone(),
            LoweringError::LifetimeError { location, .. } => location.clone(),
            LoweringError::OwnershipError { location, .. } => location.clone(),
            LoweringError::IncompleteImplementation { location, .. } => location.clone(),
            /* // TODO: Add these when error variants are added
            LoweringError::TypeResolution { location, .. } => location.clone(),
            LoweringError::ConstraintViolation { location, .. } => location.clone(),
            */
        }
    }

    /// Build semantic graphs for advanced analysis
    fn build_semantic_graphs(
        &mut self,
        typed_file: &TypedFile,
        symbol_table: &SymbolTable,
        type_table: &Rc<RefCell<TypeTable>>,
        scope_tree: &crate::tast::ScopeTree,
    ) -> Result<SemanticGraphs, Vec<CompilationError>> {
        use crate::semantic_graph::dfg_builder::DfgBuilder;
        use crate::semantic_graph::GraphConstructionError;
        use crate::tast::type_checker::TypeChecker;

        info!(
            "Building semantic graphs for {} functions",
            typed_file.functions.len()
        );

        let options = GraphConstructionOptions {
            build_call_graph: true,
            build_ownership_graph: self.config.enable_ownership_analysis,
            convert_to_ssa: true,
            eliminate_dead_code: self.config.optimization_level > 0,
            max_function_size: if self.config.optimization_level == 0 {
                5000
            } else {
                10000
            },
            collect_statistics: self.config.collect_statistics,
        };

        // Build CFGs using CfgBuilder
        let mut cfg_builder = CfgBuilder::new(options.clone());
        match cfg_builder.build_file(typed_file) {
            Ok(cfgs) => {
                // Create SemanticGraphs from the built CFGs
                let mut graphs = SemanticGraphs::new();
                graphs.control_flow = cfgs.clone();

                // Build DFGs for each function - required for full ownership analysis
                // Create TypeChecker inside loop to avoid borrow checker issues
                let mut dfg_builder = DfgBuilder::new(options);

                for function in &typed_file.functions {
                    if let Some(cfg) = graphs.control_flow.get(&function.symbol_id) {
                        // Create TypeChecker for this iteration (avoids borrow checker issues)
                        let string_interner_ref = self.string_interner.borrow();
                        let mut type_checker = TypeChecker::new(
                            type_table,
                            symbol_table,
                            scope_tree,
                            &*string_interner_ref,
                        );

                        match dfg_builder.build_dfg(cfg, function, &mut type_checker) {
                            Ok(dfg) => {
                                info!(
                                    "✓ Built DFG for function {:?} with {} nodes",
                                    function.symbol_id,
                                    dfg.nodes.len()
                                );
                                graphs.data_flow.insert(function.symbol_id, dfg);
                            }
                            Err(e) => {
                                // DFG construction failed - log but continue
                                warn!(
                                    "✗ Failed to build DFG for function {:?}: {:?}",
                                    function.symbol_id, e
                                );
                            }
                        }
                        // string_interner_ref dropped here, releasing borrow
                    }
                }

                // Build ownership graph from TAST for safety analysis
                self.populate_ownership_graph(&mut graphs, typed_file);

                Ok(graphs)
            }
            Err(graph_error) => {
                let compilation_errors = vec![CompilationError {
                    message: match &graph_error {
                        GraphConstructionError::InvalidTAST { message, .. } => {
                            format!("Invalid TAST: {}", message)
                        }
                        GraphConstructionError::TypeError { message } => {
                            format!("Type error: {}", message)
                        }
                        GraphConstructionError::InvalidCFG { message, .. } => {
                            format!("Invalid CFG: {}", message)
                        }
                        GraphConstructionError::UnresolvedSymbol { symbol_name, .. } => {
                            format!("Unresolved symbol: {}", symbol_name)
                        }
                        GraphConstructionError::MissingTypeInfo {
                            node_description, ..
                        } => format!("Missing type info: {}", node_description),
                        GraphConstructionError::InternalError { message } => {
                            format!("Internal error: {}", message)
                        }
                        GraphConstructionError::DominanceAnalysisFailed(message) => {
                            format!("Dominance analysis failed: {}", message)
                        }
                    },
                    location: match &graph_error {
                        GraphConstructionError::InvalidTAST { location, .. }
                        | GraphConstructionError::InvalidCFG { location, .. }
                        | GraphConstructionError::UnresolvedSymbol { location, .. }
                        | GraphConstructionError::MissingTypeInfo { location, .. } => {
                            location.clone()
                        }
                        GraphConstructionError::InternalError { .. }
                        | GraphConstructionError::TypeError { .. }
                        | GraphConstructionError::DominanceAnalysisFailed(_) => {
                            // These are module-level errors, use file start location
                            SourceLocation::new(1, 1, 1, 1)
                        }
                    },
                    category: ErrorCategory::SemanticAnalysisError,
                    suggestion: None,
                    related_errors: Vec::new(),
                }];

                Err(compilation_errors)
            }
        }
    }

    /// Populate ownership graph from TAST for memory safety analysis
    fn populate_ownership_graph(&self, graphs: &mut SemanticGraphs, typed_file: &TypedFile) {
        use crate::semantic_graph::MoveType;
        use crate::tast::{ScopeId, TypedExpressionKind, TypedStatement};

        // Walk all functions/methods to find variable declarations and assignments
        for class in &typed_file.classes {
            for method in &class.methods {
                let method_scope = ScopeId::from_raw(method.symbol_id.as_raw()); // Use symbol ID as scope proxy

                // Add method parameters as owned variables
                for param in &method.parameters {
                    graphs.ownership_graph.add_variable(
                        param.symbol_id,
                        param.param_type,
                        method_scope,
                    );
                }

                // Walk method body
                self.populate_ownership_from_statements(&mut graphs.ownership_graph, &method.body);
            }

            // Handle constructors
            for constructor in &class.constructors {
                let constructor_scope = ScopeId::from_raw(constructor.symbol_id.as_raw());

                for param in &constructor.parameters {
                    graphs.ownership_graph.add_variable(
                        param.symbol_id,
                        param.param_type,
                        constructor_scope,
                    );
                }

                self.populate_ownership_from_statements(
                    &mut graphs.ownership_graph,
                    &constructor.body,
                );
            }
        }

        // Standalone functions
        for function in &typed_file.functions {
            let function_scope = ScopeId::from_raw(function.symbol_id.as_raw());

            for param in &function.parameters {
                graphs.ownership_graph.add_variable(
                    param.symbol_id,
                    param.param_type,
                    function_scope,
                );
            }

            self.populate_ownership_from_statements(&mut graphs.ownership_graph, &function.body);
        }
    }

    /// Walk statements and populate ownership edges
    fn populate_ownership_from_statements(
        &self,
        ownership_graph: &mut crate::semantic_graph::OwnershipGraph,
        statements: &[crate::tast::TypedStatement],
    ) {
        use crate::semantic_graph::MoveType;
        use crate::tast::{ScopeId, TypedExpressionKind, TypedStatement};

        for stmt in statements {
            match stmt {
                TypedStatement::VarDeclaration {
                    symbol_id,
                    var_type,
                    initializer,
                    ..
                } => {
                    // Add variable to ownership graph
                    let var_scope = ScopeId::from_raw(symbol_id.as_raw()); // Use symbol ID as scope proxy
                    ownership_graph.add_variable(*symbol_id, *var_type, var_scope);

                    // Check if initialized from another variable (this is a move)
                    if let Some(init_expr) = initializer {
                        if let TypedExpressionKind::Variable {
                            symbol_id: source_var,
                        } = &init_expr.kind
                        {
                            // This is a move: var y = x;
                            ownership_graph.add_move(
                                *source_var,
                                Some(*symbol_id),
                                init_expr.source_location,
                                MoveType::Explicit,
                            );
                        }
                        // Also check for nested uses/moves in the initializer
                        self.check_expression_for_use(ownership_graph, init_expr);
                    }
                }
                TypedStatement::Expression { expression, .. } => {
                    self.check_expression_for_use(ownership_graph, expression);
                }
                TypedStatement::Return { value, .. } => {
                    if let Some(ret_expr) = value {
                        self.check_expression_for_use(ownership_graph, ret_expr);
                    }
                }
                TypedStatement::If {
                    condition,
                    then_branch,
                    else_branch,
                    ..
                } => {
                    self.check_expression_for_use(ownership_graph, condition);
                    self.populate_ownership_from_statements(
                        ownership_graph,
                        std::slice::from_ref(then_branch.as_ref()),
                    );
                    if let Some(else_stmt) = else_branch {
                        self.populate_ownership_from_statements(
                            ownership_graph,
                            std::slice::from_ref(else_stmt.as_ref()),
                        );
                    }
                }
                TypedStatement::While {
                    condition, body, ..
                } => {
                    self.check_expression_for_use(ownership_graph, condition);
                    self.populate_ownership_from_statements(
                        ownership_graph,
                        std::slice::from_ref(body.as_ref()),
                    );
                }
                TypedStatement::Block { statements, .. } => {
                    self.populate_ownership_from_statements(ownership_graph, statements);
                }
                _ => {}
            }
        }
    }

    /// Check if an expression uses a variable (for use-after-move detection)
    fn check_expression_for_use(
        &self,
        ownership_graph: &mut crate::semantic_graph::OwnershipGraph,
        expr: &crate::tast::TypedExpression,
    ) {
        use crate::semantic_graph::MoveType;
        use crate::tast::TypedExpressionKind;

        match &expr.kind {
            TypedExpressionKind::Variable { symbol_id } => {
                // Record this as a use site for the variable
                ownership_graph.record_use(*symbol_id, expr.source_location);
            }
            TypedExpressionKind::FieldAccess { object, .. } => {
                self.check_expression_for_use(ownership_graph, object);
            }
            TypedExpressionKind::FunctionCall {
                function,
                arguments,
                ..
            } => {
                self.check_expression_for_use(ownership_graph, function);
                for arg in arguments {
                    // If a variable is passed as a function argument, record it as a move
                    if let TypedExpressionKind::Variable { symbol_id } = &arg.kind {
                        ownership_graph.add_move(
                            *symbol_id,
                            None,
                            arg.source_location,
                            MoveType::FunctionCall,
                        );
                    }
                    self.check_expression_for_use(ownership_graph, arg);
                }
            }
            TypedExpressionKind::MethodCall {
                receiver,
                arguments,
                ..
            } => {
                self.check_expression_for_use(ownership_graph, receiver);
                for arg in arguments {
                    if let TypedExpressionKind::Variable { symbol_id } = &arg.kind {
                        ownership_graph.add_move(
                            *symbol_id,
                            None,
                            arg.source_location,
                            MoveType::FunctionCall,
                        );
                    }
                    self.check_expression_for_use(ownership_graph, arg);
                }
            }
            TypedExpressionKind::StaticMethodCall { arguments, .. } => {
                for arg in arguments {
                    if let TypedExpressionKind::Variable { symbol_id } = &arg.kind {
                        ownership_graph.add_move(
                            *symbol_id,
                            None,
                            arg.source_location,
                            MoveType::FunctionCall,
                        );
                    }
                    self.check_expression_for_use(ownership_graph, arg);
                }
            }
            TypedExpressionKind::FunctionLiteral {
                parameters, body, ..
            } => {
                // Lambda/closure captures variables - this is a MOVE!
                // Use CaptureAnalyzer to determine what variables are captured
                use crate::tast::capture_analyzer::CaptureAnalyzer;

                // Create analyzer with a dummy scope (we just need to find captures)
                let analyzer = CaptureAnalyzer::new(crate::tast::ScopeId::from_raw(0));
                let capture_analysis = analyzer.analyze_function_literal(parameters, body);

                // Each captured variable is moved into the closure environment
                for captured_var in &capture_analysis.captures {
                    debug!("OWNERSHIP DEBUG: Lambda captures variable {:?}, adding move to ownership graph", captured_var.symbol_id);
                    ownership_graph.add_move(
                        captured_var.symbol_id,
                        None, // Moved into closure environment (no destination variable)
                        expr.source_location,
                        MoveType::Explicit,
                    );
                }

                // Also recursively check the body for nested lambdas
                for stmt in body {
                    self.check_statement_for_use(ownership_graph, stmt);
                }
            }
            _ => {}
        }
    }

    /// Check if a statement uses variables (helper for checking lambda bodies)
    fn check_statement_for_use(
        &self,
        ownership_graph: &mut crate::semantic_graph::OwnershipGraph,
        stmt: &crate::tast::TypedStatement,
    ) {
        use crate::tast::TypedStatement;

        match stmt {
            TypedStatement::Expression { expression, .. } => {
                self.check_expression_for_use(ownership_graph, expression);
            }
            TypedStatement::VarDeclaration { initializer, .. } => {
                if let Some(init_expr) = initializer {
                    self.check_expression_for_use(ownership_graph, init_expr);
                }
            }
            TypedStatement::Return { value, .. } => {
                if let Some(ret_expr) = value {
                    self.check_expression_for_use(ownership_graph, ret_expr);
                }
            }
            TypedStatement::If {
                condition,
                then_branch,
                else_branch,
                ..
            } => {
                self.check_expression_for_use(ownership_graph, condition);
                self.check_statement_for_use(ownership_graph, then_branch);
                if let Some(else_stmt) = else_branch {
                    self.check_statement_for_use(ownership_graph, else_stmt);
                }
            }
            TypedStatement::While {
                condition, body, ..
            } => {
                self.check_expression_for_use(ownership_graph, condition);
                self.check_statement_for_use(ownership_graph, body);
            }
            TypedStatement::Block { statements, .. } => {
                for s in statements {
                    self.check_statement_for_use(ownership_graph, s);
                }
            }
            _ => {}
        }
    }

    /// Lower TAST to HIR
    fn lower_tast_to_hir(
        &mut self,
        typed_file: &TypedFile,
        semantic_graphs: Option<&SemanticGraphs>,
        symbol_table: &SymbolTable,
        type_table: &Rc<RefCell<TypeTable>>,
    ) -> Result<HirModule, Vec<CompilationError>> {
        // Use the new TAST to HIR lowering
        match lower_tast_to_hir(
            typed_file,
            symbol_table,
            type_table,
            &mut *self.string_interner.borrow_mut(),
            semantic_graphs,
        ) {
            Ok(hir_module) => Ok(hir_module),
            Err(lowering_errors) => {
                let compilation_errors = lowering_errors
                    .into_iter()
                    .map(|err| CompilationError {
                        message: err.message,
                        location: err.location,
                        category: ErrorCategory::HIRLoweringError,
                        suggestion: None,
                        related_errors: Vec::new(),
                    })
                    .collect();

                Err(compilation_errors)
            }
        }
    }

    /// Lower HIR to MIR (mid-level IR in SSA form)
    /// Enforces memory safety rules before lowering in strict mode
    fn lower_hir_to_mir(
        &mut self,
        hir_module: &HirModule,
        type_table: &Rc<RefCell<TypeTable>>,
        symbol_table: &SymbolTable,
        semantic_graphs: Option<&SemanticGraphs>,
        typed_file: &TypedFile,
    ) -> Result<IrModule, Vec<CompilationError>> {
        // MEMORY SAFETY ENFORCEMENT: Check ownership/lifetime violations before MIR lowering
        // In strict mode (@:safety(true)), we must reject code with memory safety violations
        if let Some(safety_mode) = typed_file.program_safety_mode {
            if safety_mode == crate::tast::SafetyMode::Strict {
                if let Some(graphs) = semantic_graphs {
                    // Check for ownership violations
                    let violations = self.check_memory_safety_violations(
                        typed_file,
                        graphs,
                        symbol_table,
                        type_table,
                    );
                    if !violations.is_empty() {
                        debug!("\n⛔ MEMORY SAFETY ENFORCEMENT: Blocking MIR lowering due to {} violation(s) in strict mode", violations.len());
                        return Err(violations);
                    } else {
                        debug!("✅ MEMORY SAFETY: All checks passed, proceeding to MIR lowering");
                    }
                }
            }
        }

        match lower_hir_to_mir(
            hir_module,
            &*self.string_interner.borrow(),
            type_table,
            symbol_table,
        ) {
            Ok(mir_module) => {
                // MIR SAFETY VALIDATION: Enforce memory safety at MIR level
                // This validates that MIR operations respect semantic analysis constraints
                if let Some(graphs) = semantic_graphs {
                    use crate::ir::validation::MirSafetyValidator;

                    if let Err(validation_errors) =
                        MirSafetyValidator::validate(&mir_module, graphs)
                    {
                        debug!(
                            "\n⛔ MIR SAFETY VALIDATION: Found {} violation(s)",
                            validation_errors.len()
                        );

                        let compilation_errors = validation_errors
                            .into_iter()
                            .map(|err| CompilationError {
                                message: format!("MIR safety violation: {:?}", err.kind),
                                location: SourceLocation::new(1, 1, 1, 1),
                                category: ErrorCategory::OwnershipError,
                                suggestion: None,
                                related_errors: Vec::new(),
                            })
                            .collect();

                        return Err(compilation_errors);
                    } else {
                        debug!("✅ MIR SAFETY VALIDATION: All checks passed");
                    }
                }

                Ok(mir_module)
            }
            Err(lowering_errors) => {
                let compilation_errors = lowering_errors
                    .into_iter()
                    .map(|err| CompilationError {
                        message: err.message,
                        location: err.location,
                        category: ErrorCategory::HIRLoweringError,
                        suggestion: None,
                        related_errors: Vec::new(),
                    })
                    .collect();

                Err(compilation_errors)
            }
        }
    }

    /// Optimize MIR modules
    fn optimize_mir(&mut self, mut mir_module: IrModule) -> IrModule {
        use crate::ir::optimization::OptimizationLevel;

        // Map config optimization level to OptimizationLevel enum
        let opt_level = match self.config.optimization_level {
            0 => OptimizationLevel::O0, // Debug mode: no optimization
            1 => OptimizationLevel::O1, // Basic: DCE, const fold, copy prop
            2 => OptimizationLevel::O2, // Standard: + CSE, LICM, CFG simplify
            _ => OptimizationLevel::O3, // Aggressive: + GVN, inlining, tail call opt
        };

        let mut pass_manager = PassManager::for_level(opt_level);
        let result = pass_manager.run(&mut mir_module);

        // Log optimization statistics
        if result.modified {
            tracing::debug!("MIR optimization modified module: {:?}", result.stats);
        }

        mir_module
    }

    /// Validate HIR for correctness
    fn validate_hir(&self, hir_module: &HirModule) -> Result<(), Vec<CompilationError>> {
        // Use the OptimizableModule trait for validation
        match hir_module.validate() {
            Ok(()) => Ok(()),
            Err(validation_errors) => {
                let compilation_errors = validation_errors.into_iter().map(|err| {
                    CompilationError {
                        message: format!("HIR validation error: {:?}", err.kind),
                        location: SourceLocation::new(1, 1, 1, 1), // HIR validation errors apply to whole module
                        category: ErrorCategory::HIRValidationError,
                        suggestion: Some("Check the HIR module structure and ensure all invariants are satisfied".to_string()),
                        related_errors: Vec::new(),
                    }
                }).collect();

                Err(compilation_errors)
            }
        }
    }

    /// Optimize HIR using optimization passes
    fn optimize_hir(
        &self,
        mut hir_module: HirModule,
        semantic_graphs: Option<&SemanticGraphs>,
    ) -> HirModule {
        // Use semantic graphs for more precise optimization

        // Create HIR-specific optimization passes
        let mut passes: Vec<Box<dyn crate::ir::optimization::OptimizationPass>> = vec![];

        // Add dead code elimination with call graph if available
        if let Some(graphs) = semantic_graphs {
            let mut dce = crate::ir::optimizable::HirDeadCodeElimination::new()
                .with_call_graph(&graphs.call_graph);

            // Find entry points
            for func in hir_module.functions.values() {
                if func.is_entry_point() {
                    dce = dce.add_entry_point(func.symbol_id);
                }
            }

            // Note: We can't box HirDeadCodeElimination directly because it has a lifetime
            // For now, just run it directly
            if let crate::ir::optimization::OptimizationResult { modified: true, .. } =
                crate::ir::optimizable::HirOptimizationPass::optimize_hir(&mut dce, &mut hir_module)
            {
                info!("HIR dead code elimination removed unreachable functions");
            }
        }

        // Run other generic optimization passes if any
        if !passes.is_empty() {
            match optimize(&mut hir_module, passes, false) {
                Ok(result) => {
                    if result.modified {
                        info!("HIR optimization modified the module");
                    }
                }
                Err(validation_errors) => {
                    error!(
                        "HIR validation failed after optimization: {:?}",
                        validation_errors
                    );
                }
            }
        }

        hir_module
    }

    /// Get pipeline statistics
    pub fn stats(&self) -> &PipelineStats {
        &self.stats
    }

    /// Reset pipeline statistics
    pub fn reset_stats(&mut self) {
        self.stats = PipelineStats::default();
    }

    /// Stage 2b: Run basic flow-sensitive analysis during type checking
    fn run_basic_flow_analysis(
        &self,
        typed_file: &TypedFile,
        symbol_table: &SymbolTable,
        type_table: &Rc<RefCell<TypeTable>>,
    ) -> Vec<CompilationError> {
        let mut type_flow_guard = TypeFlowGuard::new(symbol_table, type_table);

        // Perform basic flow analysis without CFG/DFG (they're built in stage 4)
        let flow_safety_results = type_flow_guard.analyze_file(typed_file);

        // Only collect actual errors — warnings (dead code, unreachable code) are
        // informational and should not be treated as compilation errors.
        self.convert_flow_safety_errors(flow_safety_results.errors)
    }

    /// Stage 4b: Run enhanced flow analysis with CFG/DFG integration
    fn run_enhanced_flow_analysis(
        &self,
        typed_file: &TypedFile,
        semantic_graphs: &SemanticGraphs,
        symbol_table: &SymbolTable,
        type_table: &Rc<RefCell<TypeTable>>,
    ) -> Vec<CompilationError> {
        let mut type_flow_guard = TypeFlowGuard::new(symbol_table, type_table);
        let mut errors = Vec::new();

        // Enhanced analysis using CFG and DFG from semantic graphs
        for function in &typed_file.functions {
            // Get CFG and DFG for function if available
            if let Some(cfg) = semantic_graphs.cfg_for_function(function.symbol_id) {
                if let Some(dfg) = semantic_graphs.dfg_for_function(function.symbol_id) {
                    type_flow_guard.analyze_function_safety(function, cfg, dfg);
                }
            }
        }

        // Collect enhanced analysis results
        let results = type_flow_guard.into_results();
        errors.extend(self.convert_flow_safety_errors(results.errors));

        // Track enhanced analysis metrics
        if self.config.collect_statistics {
            // Store metrics for reporting
            // These would be aggregated with other metrics
        }

        errors
    }

    /// Helper: Get variable name from SymbolId for diagnostics
    fn get_variable_name(
        &self,
        symbol_id: SymbolId,
        symbol_table: &SymbolTable,
        typed_file: &TypedFile,
    ) -> String {
        // First try symbol table
        if let Some(sym) = symbol_table.get_symbol(symbol_id) {
            // sym.name is an InternedString, resolve it using typed_file's string interner
            let interner = typed_file.string_interner.borrow();
            debug!(
                "DEBUG: Trying to resolve InternedString({}) from interner with {} strings",
                sym.name.as_raw(),
                interner.len()
            );
            if let Some(name_str) = interner.get(sym.name) {
                debug!(
                    "DEBUG get_variable_name: Symbol {} -> '{}'",
                    symbol_id.as_raw(),
                    name_str
                );
                return name_str.to_string();
            } else {
                debug!("DEBUG get_variable_name: Symbol {} found but couldn't resolve interned string {} in interner (interner has {} strings)",
                    symbol_id.as_raw(), sym.name.as_raw(), interner.len());
                // Try to iterate through all strings to see what's there
                debug!("DEBUG: Dumping first 100 strings in interner:");
                for i in 0..100.min(interner.len()) {
                    if let Some(s) = interner.get(unsafe { InternedString::from_raw(i as u32) }) {
                        debug!("  [{}] = '{}'", i, s);
                    }
                }
            }
        } else {
            debug!(
                "DEBUG get_variable_name: Symbol {} NOT in symbol table",
                symbol_id.as_raw()
            );
        }

        // Try to find the variable in typed_file declarations
        // Check all functions for parameters
        for func in &typed_file.functions {
            for param in &func.parameters {
                if param.symbol_id == symbol_id {
                    return typed_file
                        .get_string(param.name)
                        .unwrap_or_else(|| format!("param#{}", symbol_id.as_raw()));
                }
            }
        }

        // Check classes
        for class in &typed_file.classes {
            for method in &class.methods {
                for param in &method.parameters {
                    if param.symbol_id == symbol_id {
                        return typed_file
                            .get_string(param.name)
                            .unwrap_or_else(|| format!("param#{}", symbol_id.as_raw()));
                    }
                }
            }
            for constructor in &class.constructors {
                for param in &constructor.parameters {
                    if param.symbol_id == symbol_id {
                        return typed_file
                            .get_string(param.name)
                            .unwrap_or_else(|| format!("param#{}", symbol_id.as_raw()));
                    }
                }
            }
        }

        // TODO: Local variables are not in the symbol table and VarDeclaration doesn't store names
        // Need to populate symbol table with local variables during type checking
        // For now, fall back to a readable format
        format!("variable#{}", symbol_id.as_raw())
    }

    /// Check memory safety violations for strict mode enforcement
    /// This is called before MIR lowering to block unsafe code
    fn check_memory_safety_violations(
        &self,
        typed_file: &TypedFile,
        semantic_graphs: &SemanticGraphs,
        symbol_table: &SymbolTable,
        type_table: &Rc<RefCell<TypeTable>>,
    ) -> Vec<CompilationError> {
        // Reuse the existing memory safety analysis
        self.run_memory_safety_analysis(typed_file, semantic_graphs, symbol_table, type_table)
    }

    /// Stage 4c: Run memory safety analysis (lifetime and ownership)
    fn run_memory_safety_analysis(
        &self,
        typed_file: &TypedFile,
        semantic_graphs: &SemanticGraphs,
        symbol_table: &SymbolTable,
        type_table: &Rc<RefCell<TypeTable>>,
    ) -> Vec<CompilationError> {
        let mut errors = Vec::new();

        // Only run if ownership/lifetime analysis is enabled
        if !self.config.enable_ownership_analysis && !self.config.enable_lifetime_analysis {
            return errors;
        }

        // For concurrent code (using threads), ALWAYS check ownership to prevent data races
        // For non-concurrent code, only check if @:safety mode is enabled
        // Check if any imported symbols include concurrent primitives (Thread, Channel, Mutex, Arc)
        let uses_concurrency = typed_file.imports.iter().any(|imp| {
            // Check if the import's module path contains "concurrent"
            // This covers both explicit imports (e.g., import rayzor.concurrent.Thread)
            // and wildcard imports (e.g., import rayzor.concurrent.*)
            let interner = self.string_interner.borrow();
            if let Some(path) = interner.get(imp.module_path) {
                path.contains("concurrent")
            } else {
                false
            }
        });

        if typed_file.program_safety_mode.is_none() && !uses_concurrency {
            // Program uses default runtime-managed memory and doesn't use threads, skip memory safety analysis
            return errors;
        }

        // Use the semantic graph's ownership analysis capabilities
        if self.config.enable_ownership_analysis {
            // The ownership graph is already built in semantic_graphs
            // Validate basic ownership graph structure
            if let Err(ownership_error) = semantic_graphs.ownership_graph.validate() {
                // Extract a reasonable source location based on the error type
                let location = match &ownership_error {
                    crate::semantic_graph::OwnershipValidationError::InvalidLifetime {
                        variable,
                        ..
                    }
                    | crate::semantic_graph::OwnershipValidationError::InvalidBorrow {
                        variable,
                        ..
                    }
                    | crate::semantic_graph::OwnershipValidationError::InvalidMove {
                        variable,
                        ..
                    } => {
                        // Try to find the function that contains this variable
                        typed_file
                            .functions
                            .iter()
                            .find(|f| {
                                // This is a heuristic - in production, we'd need better mapping
                                f.symbol_id == *variable
                                    || f.parameters
                                        .iter()
                                        .any(|p| p.name.as_raw() == variable.as_raw())
                            })
                            .map(|f| f.source_location)
                            .unwrap_or_else(|| {
                                // Fall back to file-level location
                                SourceLocation::new(1, 1, 1, 1)
                            })
                    }
                };

                errors.push(CompilationError {
                    message: format!("Ownership validation error: {:?}", ownership_error),
                    location,
                    category: ErrorCategory::OwnershipError,
                    suggestion: Some(
                        "Check that all ownership constraints are properly satisfied".to_string(),
                    ),
                    related_errors: Vec::new(),
                });
            }

            // Check for use-after-move violations
            let use_after_move_violations = semantic_graphs.ownership_graph.check_use_after_move();
            for violation in use_after_move_violations {
                let (message, location, suggestion) = match violation {
                    crate::semantic_graph::OwnershipViolation::UseAfterMove {
                        variable,
                        use_location,
                        move_location,
                        ..
                    } => {
                        let var_name = self.get_variable_name(variable, symbol_table, typed_file);

                        let suggestion = format!(
                            "To fix this use-after-move error, you can:\n\
                             1. Clone the value before moving: `var y = {0}.clone();`\n\
                             2. Use a borrow instead: Add `@:borrow` annotation to the parameter\n\
                             3. Use the value after the move instead of before\n\
                             4. For shared ownership, use `@:rc` or `@:arc` on the class\n\
                             Note: Variable was moved at line {1}, then used at line {2}.",
                            var_name, move_location.line, use_location.line
                        );

                        (
                            format!(
                                "Use after move: variable '{}' was moved at line {} and used at line {}",
                                var_name, move_location.line, use_location.line
                            ),
                            use_location,
                            suggestion,
                        )
                    }
                    crate::semantic_graph::OwnershipViolation::AliasingViolation {
                        variable,
                        mutable_borrow_locations,
                        immutable_borrow_locations,
                    } => {
                        let var_name = self.get_variable_name(variable, symbol_table, typed_file);
                        let suggestion = format!(
                            "Aliasing violation: cannot have mutable and immutable borrows simultaneously.\n\
                             To fix:\n\
                             1. Use only immutable borrows (@:borrow) if you don't need to modify\n\
                             2. Ensure mutable borrows end before creating new borrows\n\
                             3. Consider using 'final' for read-only access\n\
                             4. For shared mutable state, use @:atomic or @:arc with locks"
                        );
                        (
                            format!(
                                "Aliasing violation: variable '{}' has {} mutable and {} immutable borrows",
                                var_name,
                                mutable_borrow_locations.len(),
                                immutable_borrow_locations.len()
                            ),
                            // Use first mutable borrow location as the primary location
                            mutable_borrow_locations.first()
                                .or(immutable_borrow_locations.first())
                                .cloned()
                                .unwrap_or_else(|| {
                                    // Fall back to a reasonable default location
                                    SourceLocation::new(1, 1, 1, 1)
                                }),
                            suggestion
                        )
                    }
                    crate::semantic_graph::OwnershipViolation::DanglingPointer {
                        variable,
                        use_location,
                        expired_lifetime,
                    } => {
                        let var_name = self.get_variable_name(variable, symbol_table, typed_file);
                        let suggestion = format!(
                            "Dangling pointer: reference outlives the object it points to.\n\
                             To fix:\n\
                             1. Extend the lifetime of the object\n\
                             2. Clone the value instead of borrowing: `{}.clone()`\n\
                             3. Use owned values instead of references\n\
                             4. For shared ownership, use @:rc or @:arc on the class",
                            var_name
                        );
                        (
                            format!(
                                "Dangling pointer: variable '{}' used after its lifetime expired",
                                var_name
                            ),
                            use_location,
                            suggestion,
                        )
                    }
                    crate::semantic_graph::OwnershipViolation::DoubleFree {
                        variable,
                        first_free: _,
                        second_free,
                    } => {
                        let var_name = self.get_variable_name(variable, symbol_table, typed_file);
                        (
                            format!(
                                "Double free: variable '{}' was freed at two locations",
                                var_name
                            ),
                            second_free, // Use second free location as primary
                            "Ensure that each resource is freed exactly once".to_string(),
                        )
                    }
                };

                errors.push(CompilationError {
                    message,
                    location,
                    category: ErrorCategory::OwnershipError,
                    suggestion: Some(suggestion),
                    related_errors: Vec::new(),
                });
            }

            // Run detailed ownership analysis using OwnershipAnalyzer
            // Only analyze functions in classes with @:safety annotation
            use crate::semantic_graph::analysis::ownership_analyzer::{
                FunctionAnalysisContext as OwnershipContext, OwnershipAnalyzer,
            };

            // Check if any class has @:safety annotation
            let has_safety_classes = typed_file.classes.iter().any(|c| c.has_safety_annotation());

            if has_safety_classes {
                let mut ownership_analyzer = OwnershipAnalyzer::new();
                for (function_id, cfg) in &semantic_graphs.control_flow {
                    // Check if this function belongs to a @:safety class
                    let function_has_safety = typed_file.classes.iter().any(|class| {
                        if !class.has_safety_annotation() {
                            return false;
                        }
                        class.methods.iter().any(|m| m.symbol_id == *function_id)
                            || class
                                .constructors
                                .iter()
                                .any(|c| c.symbol_id == *function_id)
                    });

                    if !function_has_safety {
                        continue; // Skip analysis for non-@:safety classes
                    }

                    if let Some(dfg) = semantic_graphs.data_flow.get(function_id) {
                        let context = OwnershipContext {
                            function_id: *function_id,
                            cfg,
                            dfg,
                            call_graph: &semantic_graphs.call_graph,
                            ownership_graph: &semantic_graphs.ownership_graph,
                        };

                        match ownership_analyzer.analyze_function(&context) {
                            Ok(ownership_violations) => {
                                // Process violations from ownership analysis
                                for violation in &ownership_violations {
                                    // Extract the specific source location from the violation
                                    let (message, location, suggestion) = match violation {
                                    crate::semantic_graph::analysis::ownership_analyzer::OwnershipViolation::UseAfterMove {
                                        variable,
                                        use_location,
                                        move_location,
                                        move_destination,
                                    } => {
                                        let var_name = self.get_variable_name(*variable, symbol_table, typed_file);
                                        let dest_info = move_destination.map(|d| {
                                            let dest_name = self.get_variable_name(d, symbol_table, typed_file);
                                            format!(" (moved to '{}')", dest_name)
                                        }).unwrap_or_default();
                                        (
                                            format!(
                                                "Use after move: variable '{}' was moved at line {} and used at line {}{}",
                                                var_name,
                                                move_location.line,
                                                use_location.line,
                                                dest_info
                                            ),
                                            use_location.clone(),
                                            "Consider cloning the value or restructuring to avoid the move"
                                        )
                                    },
                                    crate::semantic_graph::analysis::ownership_analyzer::OwnershipViolation::DoubleMove {
                                        variable,
                                        first_move,
                                        second_move,
                                    } => {
                                        let var_name = self.get_variable_name(*variable, symbol_table, typed_file);
                                        (
                                            format!(
                                                "Double move: variable '{}' was moved at line {} and again at line {}",
                                                var_name, first_move.line, second_move.line
                                            ),
                                            second_move.clone(),
                                            "A variable can only be moved once; consider cloning or borrowing instead"
                                        )
                                    },
                                    crate::semantic_graph::analysis::ownership_analyzer::OwnershipViolation::BorrowConflict {
                                        variable,
                                        mutable_borrow,
                                        conflicting_borrow,
                                        conflict_type,
                                    } => {
                                        let var_name = self.get_variable_name(*variable, symbol_table, typed_file);
                                        let conflict_desc = match conflict_type {
                                            crate::semantic_graph::analysis::ownership_analyzer::BorrowConflictType::MultipleMutableBorrows => "multiple mutable borrows",
                                            crate::semantic_graph::analysis::ownership_analyzer::BorrowConflictType::MutableWithImmutable => "mutable borrow while immutably borrowed",
                                            crate::semantic_graph::analysis::ownership_analyzer::BorrowConflictType::BorrowOfMovedVariable => "borrow of moved variable",
                                        };
                                        (
                                            format!(
                                                "Borrow conflict: variable '{}' has {} at line {}",
                                                var_name, conflict_desc, mutable_borrow.line
                                            ),
                                            conflicting_borrow.clone(),
                                            "Ensure that mutable and immutable borrows don't overlap"
                                        )
                                    },
                                    crate::semantic_graph::analysis::ownership_analyzer::OwnershipViolation::MoveOfBorrowedVariable {
                                        variable,
                                        move_location,
                                        active_borrows,
                                    } => {
                                        let var_name = self.get_variable_name(*variable, symbol_table, typed_file);
                                        (
                                            format!(
                                                "Cannot move variable '{}': it has {} active borrow(s)",
                                                var_name,
                                                active_borrows.len()
                                            ),
                                            move_location.clone(),
                                            "Cannot move a variable while it is borrowed; wait for borrows to end"
                                        )
                                    },
                                    crate::semantic_graph::analysis::ownership_analyzer::OwnershipViolation::BorrowOutlivesOwner {
                                        borrowed_variable,
                                        borrower,
                                        borrow_location,
                                        owner_end_location,
                                    } => {
                                        let borrowed_name = self.get_variable_name(*borrowed_variable, symbol_table, typed_file);
                                        let borrower_name = self.get_variable_name(*borrower, symbol_table, typed_file);
                                        (
                                            format!(
                                                "Borrow of '{}' by '{}' outlives the owner (ends at line {})",
                                                borrowed_name,
                                                borrower_name,
                                                owner_end_location.line
                                            ),
                                            borrow_location.clone(),
                                            "Ensure that borrows don't outlive the data they reference"
                                        )
                                    },
                                };

                                    errors.push(CompilationError {
                                        message,
                                        location,
                                        category: ErrorCategory::OwnershipError,
                                        suggestion: Some(suggestion.to_string()),
                                        related_errors: Vec::new(),
                                    });
                                }
                            }
                            Err(ownership_error) => {
                                let location = typed_file
                                    .functions
                                    .iter()
                                    .find(|f| f.symbol_id == *function_id)
                                    .map(|f| f.source_location)
                                    .unwrap_or_else(|| SourceLocation::new(1, 1, 1, 1));

                                errors.push(CompilationError {
                                    message: format!(
                                        "Ownership analysis error: {:?}",
                                        ownership_error
                                    ),
                                    location,
                                    category: ErrorCategory::OwnershipError,
                                    suggestion: Some(
                                        "Check function ownership constraints".to_string(),
                                    ),
                                    related_errors: Vec::new(),
                                });
                            }
                        }
                    }
                }
            } // end if has_safety_classes
        }

        // Use lifetime analysis from semantic graph analyzers
        // Only analyze functions in classes with @:safety annotation
        if self.config.enable_lifetime_analysis {
            use crate::semantic_graph::analysis::lifetime_analyzer::LifetimeAnalyzer;
            use crate::semantic_graph::analysis::ownership_analyzer::FunctionAnalysisContext;

            // Check if any class has @:safety annotation
            let has_safety_classes = typed_file.classes.iter().any(|c| c.has_safety_annotation());

            if has_safety_classes {
                // Create lifetime analyzer and run on each function
                let mut lifetime_analyzer = LifetimeAnalyzer::new();
                for (function_id, cfg) in &semantic_graphs.control_flow {
                    // Check if this function belongs to a @:safety class
                    let function_has_safety = typed_file.classes.iter().any(|class| {
                        if !class.has_safety_annotation() {
                            return false;
                        }
                        class.methods.iter().any(|m| m.symbol_id == *function_id)
                            || class
                                .constructors
                                .iter()
                                .any(|c| c.symbol_id == *function_id)
                    });

                    if !function_has_safety {
                        continue; // Skip analysis for non-@:safety classes
                    }

                    if let Some(dfg) = semantic_graphs.data_flow.get(function_id) {
                        // Create analysis context
                        let context = FunctionAnalysisContext {
                            function_id: *function_id,
                            cfg,
                            dfg,
                            call_graph: &semantic_graphs.call_graph,
                            ownership_graph: &semantic_graphs.ownership_graph,
                        };

                        // Run lifetime analysis
                        match lifetime_analyzer.analyze_function(&context) {
                            Ok(lifetime_result) => {
                                // Process any violations found
                                for violation in &lifetime_result.violations {
                                    use crate::semantic_graph::analysis::lifetime_analyzer::LifetimeViolation;

                                    let (message, location, suggestion) = match violation {
                                    LifetimeViolation::UseAfterFree { variable, use_location, end_of_lifetime, .. } => (
                                        format!("Use of variable after its lifetime has ended"),
                                        use_location.clone(),
                                        Some(format!("Variable lifetime ended at line {}, column {}",
                                            end_of_lifetime.line, end_of_lifetime.column))
                                    ),
                                    LifetimeViolation::DanglingReference { reference, referent, reference_location, referent_end_location } => (
                                        format!("Reference outlives the data it points to"),
                                        reference_location.clone(),
                                        Some(format!("Referenced data lifetime ended at line {}, column {}",
                                            referent_end_location.line, referent_end_location.column))
                                    ),
                                    LifetimeViolation::ReturnLocalReference { local_variable, return_location, .. } => (
                                        format!("Cannot return reference to local variable"),
                                        return_location.clone(),
                                        Some("Local variables are destroyed when function returns".to_string())
                                    ),
                                    LifetimeViolation::ConflictingConstraints { conflict_explanation, .. } => {
                                        // Use default location since constraints don't have specific source locations
                                        let location = typed_file.functions.iter()
                                            .find(|f| f.symbol_id == *function_id)
                                            .map(|f| f.source_location)
                                            .unwrap_or_else(|| SourceLocation::new(1, 1, 1, 1));
                                        (
                                            format!("Conflicting lifetime constraints: {}", conflict_explanation),
                                            location,
                                            Some("Check lifetime annotations and variable scopes".to_string())
                                        )
                                    },
                                };

                                    errors.push(CompilationError {
                                        message,
                                        location,
                                        category: ErrorCategory::LifetimeError,
                                        suggestion,
                                        related_errors: Vec::new(),
                                    });
                                }
                            }
                            Err(lifetime_error) => {
                                // Get function location for the error
                                let location = typed_file
                                    .functions
                                    .iter()
                                    .find(|f| f.symbol_id == *function_id)
                                    .map(|f| f.source_location)
                                    .unwrap_or_else(|| SourceLocation::new(1, 1, 1, 1));

                                errors.push(CompilationError {
                                    message: format!(
                                        "Lifetime analysis error: {:?}",
                                        lifetime_error
                                    ),
                                    location,
                                    category: ErrorCategory::LifetimeError,
                                    suggestion: Some(
                                        "Check lifetime annotations and constraints".to_string(),
                                    ),
                                    related_errors: Vec::new(),
                                });
                            }
                        }
                    }
                }
            } // end if has_safety_classes
        }

        errors
    }

    /// Convert TypeFlowGuard FlowSafetyError to CompilationWarning
    fn convert_flow_safety_warning(&self, warning: FlowSafetyError) -> CompilationError {
        let (message, category) = match &warning {
            FlowSafetyError::DeadCode { .. } => {
                ("Dead code detected".to_string(), ErrorCategory::TypeError)
            }
            _ => (format!("Warning: {:?}", warning), ErrorCategory::TypeError),
        };

        let location = match &warning {
            FlowSafetyError::DeadCode { location } => location.clone(),
            FlowSafetyError::UninitializedVariable { location, .. }
            | FlowSafetyError::NullDereference { location, .. } => location.clone(),
            _ => {
                // For warnings without specific locations, use file start
                SourceLocation::new(1, 1, 1, 1)
            }
        };

        CompilationError {
            message,
            location,
            category,
            suggestion: None,
            related_errors: Vec::new(),
        }
    }

    /// Convert TypeFlowGuard FlowSafetyError to CompilationError
    fn convert_flow_safety_errors(
        &self,
        flow_errors: Vec<FlowSafetyError>,
    ) -> Vec<CompilationError> {
        flow_errors
            .into_iter()
            .map(|err| {
                let (message, category) = match &err {
                    FlowSafetyError::UninitializedVariable {
                        variable,
                        location: _,
                    }
                    | FlowSafetyError::UseOfUninitializedVariable {
                        variable,
                        location: _,
                    } => (
                        format!("Use of uninitialized variable: {:?}", variable),
                        ErrorCategory::TypeError,
                    ),
                    FlowSafetyError::NullDereference {
                        variable,
                        location: _,
                    }
                    | FlowSafetyError::NullPointerDereference {
                        variable,
                        location: _,
                    } => (
                        format!("Potential null pointer dereference: {:?}", variable),
                        ErrorCategory::TypeError,
                    ),
                    FlowSafetyError::DeadCode { location: _ } => (
                        "Unreachable code detected".to_string(),
                        ErrorCategory::TypeError, // Dead code is often a type/logic error
                    ),
                    FlowSafetyError::ResourceLeak {
                        resource,
                        location: _,
                    } => (
                        format!("Resource leak detected: {:?}", resource),
                        ErrorCategory::OwnershipError,
                    ),
                    FlowSafetyError::UseAfterFree {
                        variable,
                        use_location: _,
                        free_location: _,
                    } => (
                        format!("Use after free: {:?}", variable),
                        ErrorCategory::LifetimeError,
                    ),
                    FlowSafetyError::UseAfterMove {
                        variable,
                        use_location: _,
                        move_location: _,
                    } => (
                        format!("Use after move: {:?}", variable),
                        ErrorCategory::OwnershipError,
                    ),
                    FlowSafetyError::InvalidBorrow {
                        variable,
                        location: _,
                        reason,
                    } => (
                        format!("Invalid borrow of {:?}: {}", variable, reason),
                        ErrorCategory::OwnershipError,
                    ),
                    FlowSafetyError::DanglingReference {
                        reference,
                        location: _,
                    } => (
                        format!("Dangling reference: {:?}", reference),
                        ErrorCategory::LifetimeError,
                    ),
                    FlowSafetyError::TypeError { message } => {
                        (message.clone(), ErrorCategory::TypeError)
                    }
                    FlowSafetyError::EffectMismatch {
                        expected,
                        actual,
                        location: _,
                    } => (
                        format!("Effect mismatch: expected {}, got {}", expected, actual),
                        ErrorCategory::TypeError,
                    ),
                    FlowSafetyError::NullAssignedToNotNull {
                        variable,
                        location: _,
                    } => (
                        format!("Cannot assign null to @:notNull variable: {:?}", variable),
                        ErrorCategory::TypeError,
                    ),
                    FlowSafetyError::NullableAssignedToNotNull {
                        variable,
                        location: _,
                    } => (
                        format!(
                            "Cannot assign potentially-null value to @:notNull variable: {:?}",
                            variable
                        ),
                        ErrorCategory::TypeError,
                    ),
                };

                let location = match &err {
                    FlowSafetyError::UninitializedVariable { location, .. }
                    | FlowSafetyError::NullDereference { location, .. }
                    | FlowSafetyError::UseOfUninitializedVariable { location, .. }
                    | FlowSafetyError::NullPointerDereference { location, .. }
                    | FlowSafetyError::DeadCode { location }
                    | FlowSafetyError::ResourceLeak { location, .. }
                    | FlowSafetyError::DanglingReference { location, .. }
                    | FlowSafetyError::InvalidBorrow { location, .. }
                    | FlowSafetyError::NullAssignedToNotNull { location, .. }
                    | FlowSafetyError::NullableAssignedToNotNull { location, .. } => {
                        location.clone()
                    }
                    FlowSafetyError::UseAfterFree { use_location, .. }
                    | FlowSafetyError::UseAfterMove { use_location, .. } => use_location.clone(),
                    FlowSafetyError::TypeError { .. } => {
                        // Type errors should be caught earlier with proper locations,
                        // but if we get here, use a reasonable fallback
                        SourceLocation::new(1, 1, 1, 1)
                    }
                    FlowSafetyError::EffectMismatch { location, .. } => location.clone(),
                };

                CompilationError {
                    message,
                    location,
                    category,
                    suggestion: None,
                    related_errors: Vec::new(),
                }
            })
            .collect()
    }
}

impl Default for HaxeCompilationPipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function to compile a single Haxe file
pub fn compile_haxe_file<P: AsRef<Path>>(file_path: P, source: &str) -> CompilationResult {
    let mut pipeline = HaxeCompilationPipeline::new();
    pipeline.compile_file(file_path, source)
}

/// Convenience function to compile Haxe source code without a file
pub fn compile_haxe_source(source: &str) -> CompilationResult {
    compile_haxe_file("inline.hx", source)
}

/// Convenience function to compile multiple Haxe files
pub fn compile_haxe_files<P: AsRef<Path>>(files: &[(P, String)]) -> CompilationResult {
    let mut pipeline = HaxeCompilationPipeline::new();
    pipeline.compile_files(files)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_creation() {
        let pipeline = HaxeCompilationPipeline::new();
        assert_eq!(pipeline.stats.files_processed, 0);
        assert!(pipeline.config.strict_type_checking);
    }

    #[test]
    fn test_compile_simple_haxe() {
        let source = r#"
            class Main {
                static function main() {
                    trace("Hello, World!");
                }
            }
        "#;

        let result = compile_haxe_file("test.hx", source);

        // Should successfully parse even if type checking fails
        assert!(result.stats.files_processed > 0);
    }

    // #[test]
    // fn test_config_customization() {
    //     let config = PipelineConfig {
    //         strict_type_checking: false,
    //         enable_lifetime_analysis: false,
    //         target_platform: TargetPlatform::Cpp,
    //         ..Default::default()
    //     };

    //     let pipeline = HaxeCompilationPipeline::with_config(config);
    //     assert!(!pipeline.config.strict_type_checking);
    //     assert_eq!(pipeline.config.target_platform, TargetPlatform::Cpp);
    // }
}
