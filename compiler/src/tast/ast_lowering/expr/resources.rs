//! `untyped __resources__()`: the resources embedded with `--resource`, as
//! the table `haxe.Resource` reads.

use super::*;
use parser::{Expr, ExprKind, ObjectField, Span};

impl<'a> AstLowering<'a> {
    /// The resource table literal: one `{name, data, str}` per resource, the
    /// bytes in `data` as Base64 and `str` null, so every entry has one shape
    /// and binary content needs no string literal of its own.
    pub(crate) fn resource_table_expr(&self, span: Span) -> Expr {
        let node = |kind| Expr { kind, span };
        let field = |name: &str, expr| ObjectField {
            name: name.to_string(),
            expr,
            span,
        };
        let entries = self
            .resources
            .iter()
            .flat_map(|resources| resources.iter())
            .map(|(name, bytes)| {
                node(ExprKind::Object(vec![
                    field("name", node(ExprKind::String(name.clone()))),
                    field("data", node(ExprKind::String(base64(bytes)))),
                    field("str", node(ExprKind::Null)),
                ]))
            })
            .collect();
        node(ExprKind::Array(entries))
    }
}

/// Standard Base64 with padding, the form `haxe.crypto.Base64.decode` reads.
fn base64(bytes: &[u8]) -> String {
    const CHARS: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b = [
            chunk[0],
            chunk.get(1).copied().unwrap_or(0),
            chunk.get(2).copied().unwrap_or(0),
        ];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(CHARS[((n >> (18 - 6 * i)) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::base64;

    #[test]
    fn base64_pads_partial_groups() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
        assert_eq!(
            base64("H\u{e9}llo World !".as_bytes()),
            "SMOpbGxvIFdvcmxkICE="
        );
    }
}
