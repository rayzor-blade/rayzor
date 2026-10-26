//! Where each resolved name in a typed file sits in its source.

use super::definition_locations::name_location;
use super::node::TypedFile;
use super::walk::{SymbolSink, walk_file};
use super::{SourceLocation, StringInterner, SymbolId, SymbolTable, TypeId, TypeKind, TypeTable};

/// A name in the source and the symbol it resolves to. `at` is the name's
/// first byte; it spans `len` bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SymbolReference {
    pub symbol: SymbolId,
    pub at: SourceLocation,
    pub len: u32,
    /// The instantiated type at this use; invalid for a type-only reference.
    pub type_id: TypeId,
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
    types: &TypeTable,
) -> Vec<SymbolReference> {
    let mut sink = Collect {
        source: source.as_bytes(),
        symbols,
        interner,
        types,
        markup: parser::lexer::Lexer::new(source)
            .tokenize()
            .unwrap_or_default()
            .into_iter()
            .filter(|token| token.kind == parser::token::TokenKind::MarkupLit)
            .map(|token| token.span)
            .collect(),
        constructed_tags: std::collections::HashSet::new(),
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
    types: &'a TypeTable,
    markup: Vec<parser::Span>,
    constructed_tags: std::collections::HashSet<(u32, usize)>,
    out: Vec<SymbolReference>,
}

impl SymbolSink for Collect<'_> {
    fn declares(&mut self, _symbol: SymbolId, _at: SourceLocation) {}

    fn constructs(&mut self, mut ty: TypeId, at: SourceLocation) {
        let type_id = ty;
        let mut symbol = None;
        for _ in 0..128 {
            ty = self.types.resolve_alias_type(ty);
            match self.types.get(ty).map(|ty| &ty.kind) {
                Some(TypeKind::Class { symbol_id, .. } | TypeKind::Abstract { symbol_id, .. }) => {
                    symbol = Some(*symbol_id);
                    break;
                }
                Some(TypeKind::GenericInstance { base_type, .. }) if *base_type != ty => {
                    ty = *base_type
                }
                _ => return,
            }
        }
        let Some(symbol) = symbol else { return };
        let mut start = at.byte_offset as usize;
        if !self
            .markup
            .iter()
            .any(|span| span.start <= start && start < span.end)
        {
            return;
        }
        if self.source.get(start) == Some(&b'<') {
            start += 1;
        }
        if start == 0 || self.source.get(start - 1) != Some(&b'<') {
            return;
        }
        let len = self.source[start..]
            .iter()
            .take_while(|byte| {
                byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.' | b':')
            })
            .count();
        if len == 0 {
            return;
        }
        // The outer constructor owns the tag; generated helper constructors
        // can inherit its position too.
        if !self.constructed_tags.insert((at.file_id, start)) {
            return;
        }
        let before = &self.source[..start];
        let row = 1 + before.iter().filter(|byte| **byte == b'\n').count() as u32;
        let column = start
            - before
                .iter()
                .rposition(|byte| *byte == b'\n')
                .map_or(0, |offset| offset + 1)
            + 1;
        self.out.push(SymbolReference {
            symbol,
            at: SourceLocation::new(at.file_id, row, column as u32, start as u32),
            len: len as u32,
            type_id,
        });
    }

    fn refers(&mut self, symbol: SymbolId, at: SourceLocation, member: bool) {
        self.refers_typed(symbol, at, member, TypeId::invalid());
    }

    fn refers_typed(
        &mut self,
        symbol: SymbolId,
        at: SourceLocation,
        member: bool,
        type_id: TypeId,
    ) {
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
                type_id,
            });
        }
    }
}
