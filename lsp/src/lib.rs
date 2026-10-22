//! Rayzor Language Server Protocol implementation.
//!
//! Editor support for Haxe through the Rayzor compiler: diagnostics as you
//! type, hover, go to definition, references and highlights, completion and
//! semantic tokens. Member completion and signature help at a call are not
//! there yet.
//!
//! Usage: `rayzor lsp` starts the server on stdio.

pub mod analysis;
mod binary_response;
mod context;
mod diagnostics;
mod documentation;
mod editor_cache;
mod project;
mod server;
mod session_cache;
mod stdlib;
mod structural_fields;
mod transport;
mod type_references;
mod workspace_index;

pub use server::{run_lsp, serve, spawn_session};
