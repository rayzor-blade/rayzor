//! Records what the free-insertion pass decided, for inspection.
//!
//! The pass answers three questions per allocation — what kind of thing it is,
//! whether anything still reads it, and where its release belongs — and until
//! now answered them silently. A wrong answer surfaces as a corrupted heap far
//! from the decision that caused it, so the decisions are worth reading
//! directly.
//!
//! Set `RAYZOR_DUMP_FREE_GRAPH` to a path and every compiled function is
//! appended there as JSON: the control-flow graph, the loops in it, each
//! allocation with its classification and aliases, and the release the pass
//! chose. Nothing is written when the variable is unset.

use serde::Serialize;
use std::sync::Mutex;

/// How an allocation is released. The pass must match the release to the
/// allocator: a string's header is reclaimed by its own free, an array's
/// buffer is released before its header, and only a plain allocation is a
/// bare `free`.
#[derive(Serialize, Clone, Debug, PartialEq, Eq)]
pub enum Release {
    StringFree,
    ArrayFreeThenHeader,
    AnonDrop,
    PlainFree,
}

/// Where in the function the release was placed.
#[derive(Serialize, Clone, Debug, PartialEq, Eq)]
pub enum Site {
    /// End of an iteration, at the loop's latch.
    Latch,
    /// A path that leaves the function.
    Exit,
    /// Released at the point the pass found its last use.
    LastUse,
}

#[derive(Serialize, Clone, Debug)]
pub struct ReleaseRecord {
    pub block: u32,
    pub how: Release,
    pub site: Site,
}

#[derive(Serialize, Clone, Debug)]
pub struct AllocRecord {
    pub id: u32,
    /// `string`, `array`, `anon` or `plain` — what the pass believes it is.
    pub kind: String,
    pub def_block: Option<u32>,
    /// The instruction that produced it, rendered for reading.
    pub def_inst: String,
    /// Whether the pass judged it to need releasing at all.
    pub needs_free: bool,
    /// Values that alias or are derived from it; a release must consider all
    /// of them when deciding whether anything still reads the allocation.
    pub derived: Vec<u32>,
    pub releases: Vec<ReleaseRecord>,
}

#[derive(Serialize, Clone, Debug)]
pub struct BlockRecord {
    pub id: u32,
    pub successors: Vec<u32>,
    pub terminator: String,
    pub instructions: usize,
}

#[derive(Serialize, Clone, Debug)]
pub struct LoopRecord {
    pub header: u32,
    pub latch: u32,
    pub body: Vec<u32>,
}

#[derive(Serialize, Clone, Debug)]
pub struct FunctionRecord {
    pub name: String,
    pub qualified_name: Option<String>,
    pub blocks: Vec<BlockRecord>,
    pub loops: Vec<LoopRecord>,
    pub allocs: Vec<AllocRecord>,
}

static RECORDS: Mutex<Vec<FunctionRecord>> = Mutex::new(Vec::new());

/// Whether recording is on. Checked before any work is done, so the pass pays
/// nothing for the instrumentation when the variable is unset.
pub fn enabled() -> bool {
    std::env::var_os("RAYZOR_DUMP_FREE_GRAPH").is_some()
}

pub fn record(function: FunctionRecord) {
    if let Ok(mut records) = RECORDS.lock() {
        records.push(function);
    }
}

/// Write everything recorded so far. Called after each module, so the file is
/// complete even when a later module fails to compile.
pub fn flush() {
    let Some(path) = std::env::var_os("RAYZOR_DUMP_FREE_GRAPH") else {
        return;
    };
    let Ok(records) = RECORDS.lock() else {
        return;
    };
    if records.is_empty() {
        return;
    }
    if let Ok(json) = serde_json::to_string_pretty(&*records) {
        let _ = std::fs::write(path, json);
    }
}
