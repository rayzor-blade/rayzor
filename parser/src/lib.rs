#![allow(elided_lifetimes_in_paths)]
#![allow(mismatched_lifetime_syntaxes)]
#![allow(deprecated)]

use nom::IResult;

// New complete Haxe AST and parser
pub mod haxe_ast;
pub mod haxe_parser;
pub mod haxe_parser_decls;
pub mod haxe_parser_expr;
pub mod haxe_parser_expr2;
pub mod haxe_parser_expr3;
pub mod haxe_parser_types;
pub mod incremental_parser;
pub mod incremental_parser_enhanced;
pub mod preprocessor;

// Modules that provide enhanced error context
pub mod enhanced_context;
// pub mod context_integration;
pub mod custom_error;
pub mod error_syntax;

pub mod enhanced_incremental_parser;

// New tokenizer + recursive descent parser infrastructure
pub mod lexer;
pub mod rd;
pub mod token;
pub mod token_stream;

// TODO: The following module needs refactoring to use diagnostics crate:
// pub mod position_parser;

// Error modules have been removed - use diagnostics crate instead

// Re-export diagnostics from the diagnostics crate
pub use diagnostics::*;

// Haxe-specific diagnostics
pub use diagnostics::haxe::HaxeDiagnostics;

// Export new Haxe parser
pub use haxe_ast::*;
pub use haxe_parser::{
    parse_haxe_file, parse_haxe_file_with_debug, parse_haxe_file_with_diagnostics,
    ParseResult, HAXE_KEYWORDS,
};
pub use incremental_parser_enhanced::{
    parse_incrementally_enhanced, IncrementalParseResult as EnhancedParseResult,
};

// #[cfg(test)]
// mod test_dollar_simple;

// #[cfg(test)]
// mod test_macro_quick;

// Legacy nom result type for compatibility
pub type NomParseResult<'a, T> = IResult<&'a str, T, nom::error::Error<&'a str>>;

// use ast::{Span, HaxeFile, PackageDecl, ImportDecl, Declaration};
// pub use ast::*;

// Removed duplicate type definitions to avoid conflicts with ast.rs

// ============================================================================
