//! WASM Component Model wrapper.
//!
//! Converts a core WASM module (with WASI P1 imports) into a WASI P2 Component
//! using the `wit-component` crate and the official P1→P2 adapter.
//!
//! This is the final step in the WASM compilation pipeline:
//!   MIR → core WASM → linked core WASM → **P2 Component**
//!
//! The output Component runs on any WASI P2 runtime (wasmtime, jco, browser
//! with @bytecodealliance/preview2-shim).

use wit_component::ComponentEncoder;

/// The WASI P1→P2 command adapter (for modules with `_start`).
/// Downloaded from wasmtime releases and embedded at compile time.
const WASI_ADAPTER_COMMAND: &[u8] =
    include_bytes!("../../data/wasi_snapshot_preview1.command.wasm");

/// The WASI P1→P2 reactor adapter (for library modules without `_start`).
const WASI_ADAPTER_REACTOR: &[u8] =
    include_bytes!("../../data/wasi_snapshot_preview1.reactor.wasm");

/// Component kind determines which adapter to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComponentKind {
    /// Executable component with `_start` entry point (default for `rayzor build`).
    Command,
    /// Library component without `_start` (for rpkg WASM components).
    Reactor,
}

/// Wrap a core WASM module as a WASI P2 Component.
///
/// Takes the linked core WASM binary (with `wasi_snapshot_preview1` imports)
/// and produces a WASM Component that uses WASI P2 interfaces.
///
/// # Arguments
/// * `core_wasm` - The linked core WASM module bytes
/// * `kind` - Whether this is a Command (has `_start`) or Reactor (library)
///
/// # Returns
/// The encoded WASM Component bytes, ready for wasmtime or jco.
pub fn wrap_as_component(core_wasm: &[u8], kind: ComponentKind) -> Result<Vec<u8>, String> {
    let adapter = match kind {
        ComponentKind::Command => WASI_ADAPTER_COMMAND,
        ComponentKind::Reactor => WASI_ADAPTER_REACTOR,
    };

    let mut encoder = ComponentEncoder::default()
        .module(core_wasm)
        .map_err(|e| format!("Failed to load core module into component encoder: {e}"))?
        .adapter("wasi_snapshot_preview1", adapter)
        .map_err(|e| format!("Failed to load WASI adapter: {e}"))?
        .validate(false); // Skip validation for speed — our linker already validates

    let component_bytes = encoder
        .encode()
        .map_err(|e| format!("Failed to encode WASM Component: {e}"))?;

    Ok(component_bytes)
}
