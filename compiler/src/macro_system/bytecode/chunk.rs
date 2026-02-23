//! Compilation unit for the macro bytecode VM.
//!
//! A `Chunk` represents a compiled function body: flat bytecode, constants pool,
//! local variable metadata, and nested closure chunks.

use super::super::value::MacroValue;
use parser::Span;

/// A compiled bytecode chunk for a single function or macro body.
#[derive(Debug, Clone)]
pub struct Chunk {
    /// Flat bytecode instructions.
    pub code: Vec<u8>,

    /// Constants pool: literals, interned strings (field names, class names),
    /// and other values referenced by `Const(idx)` instructions.
    pub constants: Vec<MacroValue>,

    /// Number of local variable slots needed (computed at compile time).
    /// Includes parameters, `this`, and all local variables.
    pub local_count: u16,

    /// Parameter binding metadata.
    pub params: Vec<CompiledParam>,

    /// Source span map: `(byte_offset, span)` for error reporting.
    /// Only populated at instruction start offsets. Looked up via binary search.
    pub span_map: Vec<(usize, Span)>,

    /// Upvalue capture descriptors (for closures).
    pub upvalues: Vec<UpvalueDesc>,

    /// Sub-chunks for closures defined within this function.
    /// Referenced by `MakeClosure(chunk_idx)`.
    pub closures: Vec<Chunk>,

    /// Function name (for error messages and debugging).
    pub name: String,

    /// Local variable names by slot index (for reification env reconstruction).
    /// Only names that might be needed for dollar-splicing are stored.
    pub local_names: Vec<(u16, String)>,
}

/// A compiled parameter binding.
#[derive(Debug, Clone)]
pub struct CompiledParam {
    /// Local slot index where this parameter's value is stored.
    pub slot: u16,

    /// Whether this parameter is optional.
    pub optional: bool,

    /// Compiled default value expression (if parameter is optional with a default).
    /// Evaluated at call time when the argument is not provided.
    pub default_chunk: Option<Box<Chunk>>,
}

/// Describes how a closure captures a variable from an enclosing scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpvalueDesc {
    /// Capture from the immediately enclosing function's local slot.
    Local(u16),
    /// Capture from the enclosing function's upvalue slot (transitive capture).
    Upvalue(u16),
}

impl Chunk {
    /// Create a new empty chunk with the given function name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            code: Vec::new(),
            constants: Vec::new(),
            local_count: 0,
            params: Vec::new(),
            span_map: Vec::new(),
            upvalues: Vec::new(),
            closures: Vec::new(),
            name: name.into(),
            local_names: Vec::new(),
        }
    }

    /// Add a constant to the pool and return its index.
    /// Does NOT deduplicate — use `intern_string` for string dedup.
    pub fn add_constant(&mut self, val: MacroValue) -> u16 {
        let idx = self.constants.len();
        assert!(idx <= u16::MAX as usize, "constants pool overflow");
        self.constants.push(val);
        idx as u16
    }

    /// Intern a string constant, returning its index.
    /// Deduplicates: if the string already exists in the pool, returns its index.
    pub fn intern_string(&mut self, s: &str) -> u16 {
        for (i, c) in self.constants.iter().enumerate() {
            if let MacroValue::String(ref cs) = c {
                if &**cs == s {
                    return i as u16;
                }
            }
        }
        self.add_constant(MacroValue::String(std::sync::Arc::from(s)))
    }

    /// Record a source span for the instruction at the given byte offset.
    pub fn add_span(&mut self, byte_offset: usize, span: Span) {
        self.span_map.push((byte_offset, span));
    }

    /// Look up the source span for a byte offset (binary search).
    /// Returns the span of the nearest instruction at or before the given offset.
    pub fn span_at(&self, byte_offset: usize) -> Option<Span> {
        match self
            .span_map
            .binary_search_by_key(&byte_offset, |&(off, _)| off)
        {
            Ok(i) => Some(self.span_map[i].1),
            Err(0) => None,
            Err(i) => Some(self.span_map[i - 1].1),
        }
    }

    /// Register a local variable name for reification support.
    pub fn register_local_name(&mut self, slot: u16, name: String) {
        self.local_names.push((slot, name));
    }

    /// Add a sub-chunk for a closure and return its index.
    pub fn add_closure(&mut self, chunk: Chunk) -> u16 {
        let idx = self.closures.len();
        assert!(idx <= u16::MAX as usize, "closures overflow");
        self.closures.push(chunk);
        idx as u16
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_chunk_new() {
        let chunk = Chunk::new("test_func");
        assert_eq!(chunk.name, "test_func");
        assert!(chunk.code.is_empty());
        assert!(chunk.constants.is_empty());
        assert_eq!(chunk.local_count, 0);
    }

    #[test]
    fn test_add_constant() {
        let mut chunk = Chunk::new("test");
        let idx0 = chunk.add_constant(MacroValue::Int(42));
        let idx1 = chunk.add_constant(MacroValue::Float(3.14));
        assert_eq!(idx0, 0);
        assert_eq!(idx1, 1);
        assert_eq!(chunk.constants.len(), 2);
    }

    #[test]
    fn test_intern_string_dedup() {
        let mut chunk = Chunk::new("test");
        let idx0 = chunk.intern_string("hello");
        let idx1 = chunk.intern_string("world");
        let idx2 = chunk.intern_string("hello"); // duplicate
        assert_eq!(idx0, 0);
        assert_eq!(idx1, 1);
        assert_eq!(idx2, 0); // reused
        assert_eq!(chunk.constants.len(), 2);
    }

    #[test]
    fn test_span_lookup() {
        let mut chunk = Chunk::new("test");
        chunk.add_span(0, Span { start: 10, end: 15 });
        chunk.add_span(3, Span { start: 20, end: 25 });
        chunk.add_span(7, Span { start: 30, end: 35 });

        // Exact matches
        assert_eq!(chunk.span_at(0), Some(Span { start: 10, end: 15 }));
        assert_eq!(chunk.span_at(3), Some(Span { start: 20, end: 25 }));
        assert_eq!(chunk.span_at(7), Some(Span { start: 30, end: 35 }));

        // Between entries — returns nearest before
        assert_eq!(chunk.span_at(1), Some(Span { start: 10, end: 15 }));
        assert_eq!(chunk.span_at(5), Some(Span { start: 20, end: 25 }));
        assert_eq!(chunk.span_at(10), Some(Span { start: 30, end: 35 }));
    }

    #[test]
    fn test_add_closure() {
        let mut parent = Chunk::new("parent");
        let child = Chunk::new("closure_0");
        let idx = parent.add_closure(child);
        assert_eq!(idx, 0);
        assert_eq!(parent.closures.len(), 1);
        assert_eq!(parent.closures[0].name, "closure_0");
    }

    #[test]
    fn test_register_local_name() {
        let mut chunk = Chunk::new("test");
        chunk.register_local_name(0, "this".to_string());
        chunk.register_local_name(1, "x".to_string());
        assert_eq!(chunk.local_names.len(), 2);
        assert_eq!(chunk.local_names[0], (0, "this".to_string()));
        assert_eq!(chunk.local_names[1], (1, "x".to_string()));
    }
}
