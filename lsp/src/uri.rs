//! Conversions between `file:` URIs and filesystem paths.
//!
//! Clients percent-encode every byte of a path outside the unreserved set and
//! `/`, so a path is only recovered by decoding all of them, as UTF-8.

/// The bytes a URI path keeps as they are: the characters RFC 3986 allows in a path segment
/// (unreserved, sub-delimiters, `:` and `@`) and `/`. Clients encode more than this, which is why
/// reading a URI decodes every escape; writing one changes only what a path cannot hold.
fn is_plain(byte: u8) -> bool {
    byte.is_ascii_alphanumeric()
        || matches!(
            byte,
            b'-' | b'.'
                | b'_'
                | b'~'
                | b'/'
                | b'!'
                | b'$'
                | b'&'
                | b'\''
                | b'('
                | b')'
                | b'*'
                | b'+'
                | b','
                | b';'
                | b'='
                | b'@'
        )
}

fn hex(digit: u8) -> Option<u8> {
    match digit {
        b'0'..=b'9' => Some(digit - b'0'),
        b'a'..=b'f' => Some(digit - b'a' + 10),
        b'A'..=b'F' => Some(digit - b'A' + 10),
        _ => None,
    }
}

fn decode(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let escaped = (bytes[i] == b'%')
            .then(|| Some((hex(*bytes.get(i + 1)?)?, hex(*bytes.get(i + 2)?)?)))
            .flatten();
        match escaped {
            Some((high, low)) => {
                out.push(high << 4 | low);
                i += 3;
            }
            None => {
                out.push(bytes[i]);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// The path a `file:` URI names, or `None` for any other scheme or a remote host.
pub(crate) fn path_of(uri: &str) -> Option<String> {
    let rest = uri.strip_prefix("file://")?;
    let path = match rest.strip_prefix("localhost") {
        Some(path) if path.starts_with('/') => path,
        _ if rest.starts_with('/') => rest,
        _ => return None,
    };
    let path = decode(path.split(['?', '#']).next().unwrap_or(path));
    // `file:///C:/dir` names `C:/dir`.
    let drive = path.as_bytes();
    Some(
        if drive.len() >= 3
            && drive[0] == b'/'
            && drive[1].is_ascii_alphabetic()
            && drive[2] == b':'
        {
            path[1..].to_string()
        } else {
            path
        },
    )
}

/// The `file:` URI of `path`.
pub(crate) fn uri_of(path: &str) -> String {
    let path = path.replace('\\', "/");
    let drive = path.as_bytes();
    let rooted = !path.starts_with('/')
        && drive.len() >= 2
        && drive[0].is_ascii_alphabetic()
        && drive[1] == b':';
    let mut out = String::from("file://");
    if rooted {
        out.push('/');
    }
    for (i, &byte) in path.as_bytes().iter().enumerate() {
        if is_plain(byte) || (rooted && i == 1 && byte == b':') {
            out.push(byte as char);
        } else {
            out.push_str(&format!("%{byte:02X}"));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_every_percent_escape_as_utf8() {
        assert_eq!(
            path_of("file:///work/a%20b%23c%25d%28e%29%26%2B%C3%A9/Main.hx").unwrap(),
            "/work/a b#c%d(e)&+é/Main.hx"
        );
        assert_eq!(path_of("file:///a%2fb").unwrap(), "/a/b");
    }

    #[test]
    fn keeps_a_stray_percent_and_ignores_query_and_fragment() {
        assert_eq!(path_of("file:///100%/x%zz").unwrap(), "/100%/x%zz");
        assert_eq!(path_of("file:///a/b.hx#frag").unwrap(), "/a/b.hx");
    }

    #[test]
    fn rejects_other_schemes_and_remote_hosts() {
        assert_eq!(path_of("untitled:Untitled-1"), None);
        assert_eq!(path_of("file://server/share/a.hx"), None);
        assert_eq!(path_of("file://localhost/a.hx").unwrap(), "/a.hx");
    }

    #[test]
    fn names_a_windows_drive() {
        assert_eq!(path_of("file:///C:/dir/a.hx").unwrap(), "C:/dir/a.hx");
        assert_eq!(path_of("file:///c%3A/dir/a.hx").unwrap(), "c:/dir/a.hx");
        assert_eq!(uri_of("C:\\dir\\a b.hx"), "file:///C:/dir/a%20b.hx");
    }

    #[test]
    fn a_path_survives_the_round_trip() {
        for path in [
            "/work/Pocket Studio #1 é 100%/src/Main.hx",
            "/a(1)/b&c/d+e/f?g=h",
            "/plain/path/Main.hx",
            "/日本語/ファイル.hx",
        ] {
            assert_eq!(path_of(&uri_of(path)).as_deref(), Some(path), "{path}");
        }
    }

    #[test]
    fn writes_only_what_a_path_cannot_hold() {
        assert_eq!(uri_of("/a b/c(1)/1,0,0"), "file:///a%20b/c(1)/1,0,0");
        assert_eq!(uri_of("/a#b?c%d"), "file:///a%23b%3Fc%25d");
        assert_eq!(uri_of("/é"), "file:///%C3%A9");
    }
}
