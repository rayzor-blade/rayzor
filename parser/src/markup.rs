//! Inline markup is one expression token; its contents belong to the macro.

use crate::lexer::LexError;

pub(crate) fn scan(source: &[u8], start: usize) -> Result<usize, LexError> {
    let error = |message: &str| LexError {
        message: message.into(),
        offset: start,
    };
    let mut pos = start;
    let mut root: Option<&[u8]> = None;
    let mut depth = 0;
    while pos < source.len() {
        if source[pos..].starts_with(b"<!--") {
            pos = terminator(source, pos + 4, b"-->")
                .ok_or_else(|| error("unterminated markup comment"))?;
        } else if source[pos..].starts_with(b"<![CDATA[") {
            pos = terminator(source, pos + 9, b"]]>")
                .ok_or_else(|| error("unterminated markup CDATA"))?;
        } else if source[pos] == b'{' {
            pos = braces(source, pos).ok_or_else(|| error("unterminated markup expression"))?;
        } else if source[pos] == b'<' {
            pos += 1;
            let closing = source.get(pos) == Some(&b'/');
            pos += usize::from(closing);
            let name = pos;
            while source.get(pos).is_some_and(|b| {
                b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b':' | b'.')
            }) {
                pos += 1;
            }
            if pos == name {
                return Err(error("expected markup tag name"));
            }
            let tag = &source[name..pos];
            while pos < source.len() && source[pos] != b'>' {
                pos = match source[pos] {
                    b'\'' | b'"' => {
                        quoted(source, pos).ok_or_else(|| error("unterminated markup attribute"))?
                    }
                    b'{' => braces(source, pos)
                        .ok_or_else(|| error("unterminated markup expression"))?,
                    _ => pos + 1,
                };
            }
            if pos == source.len() {
                return Err(error("unterminated markup tag"));
            }
            let empty = source.get(pos.wrapping_sub(1)) == Some(&b'/');
            pos += 1;
            let root_tag = *root.get_or_insert(tag);
            if tag == root_tag {
                if closing {
                    depth -= 1;
                } else if !empty {
                    depth += 1;
                }
                if depth == 0 {
                    return Ok(pos);
                }
            }
        } else {
            pos += 1;
        }
    }
    Err(error("unterminated markup"))
}

fn terminator(source: &[u8], start: usize, end: &[u8]) -> Option<usize> {
    source[start..]
        .windows(end.len())
        .position(|w| w == end)
        .map(|p| start + p + end.len())
}

fn quoted(source: &[u8], start: usize) -> Option<usize> {
    let mut pos = start + 1;
    while pos < source.len() {
        if source[pos] == source[start] {
            return Some(pos + 1);
        }
        pos += if source[pos] == b'\\' { 2 } else { 1 };
    }
    None
}

fn braces(source: &[u8], start: usize) -> Option<usize> {
    let mut depth = 1;
    let mut pos = start + 1;
    while pos < source.len() {
        if source[pos..].starts_with(b"//") {
            pos = terminator(source, pos + 2, b"\n")?;
        } else if source[pos..].starts_with(b"/*") {
            pos = terminator(source, pos + 2, b"*/")?;
        } else {
            match source[pos] {
                b'\'' | b'"' => {
                    pos = quoted(source, pos)?;
                    continue;
                }
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(pos + 1);
                    }
                }
                _ => {}
            }
            pos += 1;
        }
    }
    None
}
