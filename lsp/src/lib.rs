//! Rayzor Language Server Protocol implementation.
//!
//! Provides IDE integration for Haxe development with the Rayzor compiler:
//! - Live diagnostics (errors/warnings on save)
//! - Hover information (types, documentation)
//! - Go-to-definition navigation
//! - Symbol completions
//!
//! Usage: `rayzor lsp` starts the server on stdio.

pub mod analysis;
mod context;
mod diagnostics;
mod server;

pub use server::run_lsp;
