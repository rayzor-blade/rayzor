//! Gives each symbol a typed file declares the location of its name.

use super::node::TypedFile;
use super::walk::{SymbolSink, walk_file};
use super::{SourceLocation, StringInterner, SymbolId, SymbolTable};

/// Records, for every symbol `file` declares and that has no location yet, the
/// position of its name in `source`: the first whole-word occurrence after the
/// declaring node's start, outside comments, strings and metadata.
pub fn record_definition_locations(
    file: &TypedFile,
    source: &str,
    symbols: &mut SymbolTable,
    interner: &StringInterner,
) {
    let mut stamp = Stamp {
        source: source.as_bytes(),
        symbols,
        interner,
    };
    for declaration in &file.enums {
        for symbol in stamp
            .symbols
            .get_enum_variants(declaration.symbol_id)
            .cloned()
            .unwrap_or_default()
        {
            if let Some(variant) = stamp.symbols.get_symbol(symbol).and_then(|symbol| {
                declaration
                    .variants
                    .iter()
                    .find(|variant| variant.name == symbol.name)
            }) {
                stamp.declares(symbol, variant.source_location);
            }
        }
    }
    walk_file(file, &mut stamp);
}

struct Stamp<'a> {
    source: &'a [u8],
    symbols: &'a mut SymbolTable,
    interner: &'a StringInterner,
}

impl SymbolSink for Stamp<'_> {
    fn declares(&mut self, symbol: SymbolId, at: SourceLocation) {
        if !at.is_valid() {
            return;
        }
        let Some(sym) = self.symbols.get_symbol(symbol) else {
            return;
        };
        if sym.definition_location.is_valid() {
            return;
        }
        let Some(name) = self.interner.get(sym.name) else {
            return;
        };
        let location = name_location(self.source, at, name, NAME_WINDOW, false).unwrap_or(at);
        if let Some(sym) = self.symbols.get_symbol_mut(symbol) {
            sym.definition_location = location;
        }
    }
}

/// How far past a declaration's start its name may appear: modifiers,
/// metadata and doc comments come first, never a body.
const NAME_WINDOW: usize = 4096;

/// The first whole-word `name` within `window` bytes of `at`, outside
/// comments, strings and metadata; with `after_dot`, only one following a `.`.
pub(crate) fn name_location(
    source: &[u8],
    at: SourceLocation,
    name: &str,
    window: usize,
    after_dot: bool,
) -> Option<SourceLocation> {
    let start = at.byte_offset as usize;
    if name.is_empty() || start >= source.len() {
        return None;
    }
    let end = source.len().min(start + window);
    let name = name.as_bytes();
    let (mut line, mut column) = (at.line, at.column);
    let mut i = start;
    let advance = |from: usize, to: usize, line: &mut u32, column: &mut u32| {
        for &b in &source[from..to] {
            if b == b'\n' {
                *line += 1;
                *column = 1;
            } else {
                *column += 1;
            }
        }
    };
    while i < end {
        let skip_to = match source[i] {
            b'/' if source.get(i + 1) == Some(&b'/') => find(source, i, b"\n").unwrap_or(end),
            b'/' if source.get(i + 1) == Some(&b'*') => {
                find(source, i + 2, b"*/").map_or(end, |j| j + 2)
            }
            q @ (b'"' | b'\'') => skip_string(source, i, q),
            b'@' => skip_metadata(source, i),
            b if is_ident(b) => {
                let word_end = (i..source.len())
                    .find(|&j| !is_ident(source[j]))
                    .unwrap_or(source.len());
                let dotted = || {
                    source[..i]
                        .iter()
                        .rev()
                        .find(|b| !b.is_ascii_whitespace())
                        .is_some_and(|&b| b == b'.')
                };
                if &source[i..word_end] == name && (!after_dot || dotted()) {
                    advance(start, i, &mut line, &mut column);
                    return Some(SourceLocation::new(at.file_id, line, column, i as u32));
                }
                word_end
            }
            _ => i + 1,
        };
        i = skip_to.max(i + 1);
    }
    None
}

fn is_ident(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b >= 0x80
}

fn find(source: &[u8], from: usize, needle: &[u8]) -> Option<usize> {
    source
        .get(from..)?
        .windows(needle.len())
        .position(|w| w == needle)
        .map(|p| from + p)
}

fn skip_string(source: &[u8], start: usize, quote: u8) -> usize {
    let mut i = start + 1;
    while i < source.len() {
        match source[i] {
            b'\\' => i += 2,
            b if b == quote => return i + 1,
            _ => i += 1,
        }
    }
    source.len()
}

/// Past `@name`, `@:name` or `@:name(...)`, with balanced parentheses.
fn skip_metadata(source: &[u8], start: usize) -> usize {
    let mut i = start + 1;
    if source.get(i) == Some(&b':') {
        i += 1;
    }
    while i < source.len() && (is_ident(source[i]) || source[i] == b'.') {
        i += 1;
    }
    if source.get(i) != Some(&b'(') {
        return i;
    }
    let mut depth = 0usize;
    while i < source.len() {
        match source[i] {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return i + 1;
                }
            }
            q @ (b'"' | b'\'') => {
                i = skip_string(source, i, q);
                continue;
            }
            _ => {}
        }
        i += 1;
    }
    source.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(offset: u32) -> SourceLocation {
        SourceLocation::new(3, 1, offset + 1, offset)
    }

    #[test]
    fn finds_the_name_after_modifiers_and_metadata() {
        let src = "@:native(\"add\") /* add */ public static function add(a:Int)";
        let loc = name_location(src.as_bytes(), at(0), "add", NAME_WINDOW, false).unwrap();
        assert_eq!(&src[loc.byte_offset as usize..][..4], "add(");
        assert_eq!(loc.column, loc.byte_offset + 1);
        assert_eq!(loc.file_id, 3);
    }

    #[test]
    fn counts_lines_and_skips_partial_words() {
        let src = "/** total */\nvar totals = 1;\nvar total = 2;";
        let loc = name_location(src.as_bytes(), at(0), "total", NAME_WINDOW, false).unwrap();
        assert_eq!((loc.line, loc.column), (3, 5));
    }
}
