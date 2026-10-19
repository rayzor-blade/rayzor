//! Where each resolved name in a typed file sits in its source.

use super::definition_locations::name_location;
use super::node::TypedFile;
use super::walk::{SymbolSink, walk_file};
use super::{SourceLocation, StringInterner, SymbolId, SymbolTable};

/// A name in the source and the symbol it resolves to. `at` is the name's
/// first byte; it spans `len` bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SymbolReference {
    pub symbol: SymbolId,
    pub at: SourceLocation,
    pub len: u32,
}

/// How far into an expression a member's `.name` may sit: past its receiver.
const MEMBER_WINDOW: usize = 512;

/// The references written in `file`. A name the TAST implies but the source
/// does not spell (an unqualified static's class, an implicit `this`) has none.
pub fn collect_references(
    file: &TypedFile,
    source: &str,
    symbols: &SymbolTable,
    interner: &StringInterner,
) -> Vec<SymbolReference> {
    let mut sink = Collect {
        source: source.as_bytes(),
        symbols,
        interner,
        out: Vec::new(),
    };
    walk_file(file, &mut sink);
    sink.out.sort_by_key(|r| (r.at.file_id, r.at.byte_offset));
    sink.out.dedup();
    sink.out
}

struct Collect<'a> {
    source: &'a [u8],
    symbols: &'a SymbolTable,
    interner: &'a StringInterner,
    out: Vec<SymbolReference>,
}

impl SymbolSink for Collect<'_> {
    fn declares(&mut self, _symbol: SymbolId, _at: SourceLocation) {}

    fn refers(&mut self, symbol: SymbolId, at: SourceLocation, member: bool) {
        if !at.is_valid() {
            return;
        }
        let Some(name) = self
            .symbols
            .get_symbol(symbol)
            .and_then(|s| self.interner.get(s.name))
        else {
            return;
        };
        let found = member
            .then(|| name_location(self.source, at, name, MEMBER_WINDOW, true))
            .flatten()
            .or_else(|| name_location(self.source, at, name, name.len(), false));
        if let Some(at) = found {
            self.out.push(SymbolReference {
                symbol,
                at,
                len: name.len() as u32,
            });
        }
    }
}
