use super::*;

fn parts(path: &str) -> (Option<&str>, &str, Option<&str>) {
    if matches!(path, "." | "..") {
        return (Some(path), "", None);
    }
    let (directory, file) = match path.rfind(['/', '\\']) {
        Some(separator) => (Some(&path[..separator]), &path[separator + 1..]),
        None => (None, path),
    };
    let (file, extension) = match file.rsplit_once('.') {
        Some((file, extension)) => (file, Some(extension)),
        None => (file, None),
    };
    (directory, file, extension)
}

fn trailing(path: &str) -> String {
    let separator = path.rfind(['/', '\\']);
    match separator {
        Some(index) if index + 1 == path.len() => path.to_string(),
        Some(index) => format!("{path}{}", &path[index..index + 1]),
        None => format!("{path}/"),
    }
}

fn normalize(path: &str) -> String {
    let path = path.replace('\\', "/");
    if path == "/" {
        return path;
    }
    let mut target = Vec::new();
    for token in path.split('/') {
        if token == ".." && target.last().is_some_and(|last| *last != "..") {
            target.pop();
        } else if token.is_empty() {
            if !target.is_empty() || path.starts_with('/') {
                target.push(token);
            }
        } else if token != "." {
            target.push(token);
        }
    }
    let mut out = String::new();
    let (mut colon, mut slashes) = (false, false);
    for c in target.join("/").chars() {
        match c {
            ':' => {
                out.push(c);
                colon = true;
            }
            '/' if !colon => slashes = true,
            c => {
                colon = false;
                if slashes {
                    out.push('/');
                    slashes = false;
                }
                out.push(c);
            }
        }
    }
    out
}

/// Portable Haxe path rules apply independently of the compiler host OS.
pub(super) fn call(method: &str, args: &[MacroValue]) -> Option<MacroValue> {
    if method == "join" {
        let MacroValue::Array(paths) = args.first()? else {
            return None;
        };
        let mut out = String::new();
        for value in paths.read().iter() {
            let Some(path) = value.as_string().filter(|path| !path.is_empty()) else {
                continue;
            };
            if !out.is_empty() {
                out = trailing(&out);
            }
            out.push_str(path);
        }
        return Some(MacroValue::from_str(&normalize(&out)));
    }
    let path = args.first()?.as_string()?;
    if method == "isAbsolute" {
        return Some(MacroValue::Bool(
            path.starts_with('/') || path.starts_with("\\\\") || path.chars().nth(1) == Some(':'),
        ));
    }
    let result = match method {
        "normalize" => normalize(path),
        "addTrailingSlash" => trailing(path),
        "removeTrailingSlashes" => path.trim_end_matches(['/', '\\']).to_string(),
        "directory" => parts(path).0.unwrap_or("").to_string(),
        "extension" => parts(path).2.unwrap_or("").to_string(),
        "withoutDirectory" => {
            let (_, file, extension) = parts(path);
            extension.map_or_else(|| file.to_string(), |ext| format!("{file}.{ext}"))
        }
        "withoutExtension" | "withExtension" => {
            let (directory, file, _) = parts(path);
            let mut result = directory.map_or_else(String::new, |directory| {
                let separator = path
                    .rfind(['/', '\\'])
                    .map_or("/", |index| &path[index..index + 1]);
                format!("{directory}{separator}")
            });
            result.push_str(file);
            if method == "withExtension"
                && let Some(extension) = args.get(1).and_then(MacroValue::as_string)
            {
                result.push('.');
                result.push_str(extension);
            }
            result
        }
        _ => return None,
    };
    Some(MacroValue::from_str(&result))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn portable_path_methods_preserve_haxe_separators_and_dotfiles() {
        for (path, directory, file, extension, stem) in [
            ("dir/a.svg", "dir", "a.svg", "svg", "dir/a"),
            ("C:\\dir\\a.svg", "C:\\dir", "a.svg", "svg", "C:\\dir\\a"),
            ("/é/板.svg", "/é", "板.svg", "svg", "/é/板"),
            (".gitignore", "", ".gitignore", "gitignore", ""),
            (".", ".", "", "", "./"),
            ("..", "..", "", "", "../"),
            ("/", "", "", "", "/"),
            ("a.", "", "a.", "", "a"),
        ] {
            for (method, expected) in [
                ("directory", directory),
                ("withoutDirectory", file),
                ("extension", extension),
                ("withoutExtension", stem),
            ] {
                assert_eq!(
                    call(method, &[MacroValue::from_str(path)])
                        .unwrap()
                        .as_string(),
                    Some(expected),
                    "{method}({path})"
                );
            }
        }
        for (path, expected) in [
            ("a/../b", "b"),
            ("/a//b/", "/a/b"),
            ("http://a/b", "http://a/b"),
            ("C:\\a\\..\\b", "C:/b"),
            ("../a/../../b", "../../b"),
            ("/", "/"),
        ] {
            assert_eq!(normalize(path), expected, "{path}");
        }
        assert_eq!(
            call(
                "join",
                &[MacroValue::array(vec![
                    MacroValue::Null,
                    MacroValue::from_str("a\\b"),
                    MacroValue::from_str(""),
                    MacroValue::from_str("../c")
                ])]
            )
            .unwrap()
            .as_string(),
            Some("a/c")
        );
    }
}
