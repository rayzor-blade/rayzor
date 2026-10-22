//! Declaration comments belong to symbols, so declarations and references share
//! the same hover, completion and signature-help documentation.

use compiler::compilation::CompilationUnit;
use parser::lexer::Lexer;
use parser::token::{Token, TokenKind};
use std::collections::BTreeMap;

pub(crate) struct SourceComments<'a> {
    source: &'a str,
    tokens: Vec<Token>,
}

impl<'a> SourceComments<'a> {
    pub(crate) fn new(source: &'a str) -> Option<Self> {
        Some(Self {
            source,
            tokens: Lexer::new(source).tokenize().ok()?,
        })
    }

    pub(crate) fn at(&self, offset: usize) -> Option<String> {
        let index = self
            .tokens
            .binary_search_by_key(&offset, |t| t.span.start)
            .ok()?;
        comment_before(self.source, &self.tokens, index)
    }
}

pub(crate) fn record(unit: &mut CompilationUnit) {
    let mut files = BTreeMap::<_, Vec<_>>::new();
    for symbol in unit.symbol_table.all_symbols() {
        let at = symbol.definition_location;
        if at.is_valid() && symbol.documentation.is_none() {
            files
                .entry(at.file_id)
                .or_default()
                .push((symbol.id, at.byte_offset as usize));
        }
    }
    for (file_id, symbols) in files {
        let disk_source;
        let source = if let Some(source) = unit.source_of(file_id) {
            source
        } else {
            disk_source = unit
                .file_path_of(file_id)
                .and_then(|path| std::fs::read_to_string(path).ok());
            let Some(source) = disk_source.as_deref() else {
                continue;
            };
            source
        };
        let Some(comments) = SourceComments::new(source) else {
            continue;
        };
        let docs: Vec<_> = symbols
            .into_iter()
            .filter_map(|(id, offset)| {
                comments
                    .at(offset)
                    .map(|doc| (id, unit.string_interner.intern(&doc)))
            })
            .collect();
        for (id, doc) in docs {
            unit.symbol_table.get_symbol_mut(id).unwrap().documentation = Some(doc);
        }
    }
    // Builtin/extern symbols may have no lowered body or source location.
    // Their parsed declarations still carry the offsets of their comments.
    let mut extern_docs = BTreeMap::new();
    for file in &unit.stdlib_files {
        let Some(comments) = file.input.as_deref().and_then(SourceComments::new) else {
            continue;
        };
        let package = file
            .package
            .as_ref()
            .map(|p| p.path.join("."))
            .unwrap_or_default();
        for decl in &file.declarations {
            let (name, fields, metadata) = match decl {
                parser::TypeDeclaration::Class(c) => (&c.name, c.fields.as_slice(), &c.meta),
                parser::TypeDeclaration::Interface(c) => (&c.name, c.fields.as_slice(), &c.meta),
                parser::TypeDeclaration::Abstract(c) => (&c.name, c.fields.as_slice(), &c.meta),
                parser::TypeDeclaration::Enum(e) => (&e.name, &[][..], &e.meta),
                parser::TypeDeclaration::Typedef(t) => (&t.name, &[][..], &t.meta),
                _ => continue,
            };
            let owner = if package.is_empty() {
                name.clone()
            } else {
                format!("{package}.{name}")
            };
            let start = metadata.first().map_or(decl.span().start, |m| m.span.start);
            if let Some(doc) = comments.at(start) {
                extern_docs.insert(owner.clone(), unit.string_interner.intern(&doc));
            }
            for field in fields {
                let name = match &field.kind {
                    parser::ClassFieldKind::Function(f) => &f.name,
                    parser::ClassFieldKind::Var { name, .. }
                    | parser::ClassFieldKind::Final { name, .. }
                    | parser::ClassFieldKind::Property { name, .. } => name,
                };
                if let Some(doc) = comments.at(field.span.start) {
                    extern_docs
                        .insert(format!("{owner}.{name}"), unit.string_interner.intern(&doc));
                }
            }
        }
    }
    let docs: Vec<_> = unit
        .symbol_table
        .all_symbols()
        .filter(|s| s.documentation.is_none())
        .filter_map(|s| {
            let name = unit.symbol_table.build_qualified_name(
                s.id,
                &unit.scope_tree,
                &unit.string_interner,
            );
            let name = unit.string_interner.get(name)?;
            extern_docs.get(name).map(|&doc| (s.id, doc))
        })
        .collect();
    for (id, doc) in docs {
        unit.symbol_table.get_symbol_mut(id).unwrap().documentation = Some(doc);
    }
}

/// Walk only declaration prefixes, never an earlier declaration or expression.
/// Token gaps contain comments, but never comment-like text inside strings.
fn comment_before(source: &str, tokens: &[Token], mut index: usize) -> Option<String> {
    loop {
        let start = index.checked_sub(1).map_or(0, |i| tokens[i].span.end);
        let end = tokens[index].span.start;
        if let Some(doc) = comments_in_gap(source, start, end) {
            return Some(doc);
        }
        let previous = index.checked_sub(1)?;
        let text = &source[tokens[previous].span.start..tokens[previous].span.end];
        if matches!(
            text,
            "public"
                | "private"
                | "static"
                | "inline"
                | "extern"
                | "override"
                | "dynamic"
                | "macro"
                | "overload"
                | "final"
                | "abstract"
                | "class"
                | "interface"
                | "enum"
                | "typedef"
                | "function"
                | "var"
        ) {
            index = previous;
        } else {
            index = metadata_start(tokens, previous)?;
        }
    }
}

fn metadata_start(tokens: &[Token], mut index: usize) -> Option<usize> {
    if tokens[index].kind == TokenKind::RParen {
        let mut depth = 1;
        while depth > 0 {
            index = index.checked_sub(1)?;
            match tokens[index].kind {
                TokenKind::LParen => depth -= 1,
                TokenKind::RParen => depth += 1,
                _ => {}
            }
        }
        index = index.checked_sub(1)?;
    }
    if tokens[index].kind != TokenKind::Ident {
        return None;
    }
    index = index.checked_sub(1)?;
    while tokens[index].kind == TokenKind::Dot {
        index = index.checked_sub(1)?;
        if tokens[index].kind != TokenKind::Ident {
            return None;
        }
        index = index.checked_sub(1)?;
    }
    if tokens[index].kind == TokenKind::Colon {
        index = index.checked_sub(1)?;
    }
    matches!(tokens[index].kind, TokenKind::At | TokenKind::AtColon).then_some(index)
}

fn comments_in_gap(source: &str, start: usize, end: usize) -> Option<String> {
    let gap = source.get(start..end)?;
    let mut rest = gap;
    let mut comments = Vec::new();
    while !rest.is_empty() {
        let trimmed = rest.trim_start();
        // A blank line separates commentary from the next declaration.
        if rest[..rest.len() - trimmed.len()].matches('\n').count() > 1 {
            comments.clear();
        }
        rest = trimmed;
        if rest.is_empty() {
            break;
        }
        let offset = end - rest.len();
        let line_start = source[..offset].rfind('\n').map_or(0, |i| i + 1);
        let trailing = !source[line_start..offset].trim().is_empty();
        let (comment, remaining) = if let Some(body) = rest.strip_prefix("/*") {
            let close = body.find("*/")?;
            (normalize_block(&body[..close]), &body[close + 2..])
        } else {
            let body = rest.strip_prefix("//")?;
            let close = body.find('\n').unwrap_or(body.len());
            (
                body[..close]
                    .strip_prefix('/')
                    .unwrap_or(&body[..close])
                    .trim()
                    .to_string(),
                &body[close..],
            )
        };
        // `var old = 1; // old's comment` must not document the next field.
        if !trailing || rest.starts_with("/**") {
            comments.push(comment);
        }
        rest = remaining;
    }
    let doc = comments.join("\n");
    (!doc.trim().is_empty()).then(|| render_tags(&doc))
}

/// HaxeDoc and JSDoc tags share a Markdown presentation. Fenced examples retain
/// their contents verbatim, even when a code line looks like a documentation tag.
fn render_tags(doc: &str) -> String {
    let mut parts = Vec::new();
    let mut tag = "";
    let mut body = Vec::new();
    let mut fence: Option<(char, usize)> = None;
    for line in doc.lines() {
        let trimmed = line.trim_start();
        let marker = trimmed.chars().next().filter(|c| matches!(c, '`' | '~'));
        let count = marker.map_or(0, |c| trimmed.chars().take_while(|&v| v == c).count());
        if let Some((c, length)) = fence {
            if marker == Some(c) && count >= length && trimmed[count..].trim().is_empty() {
                fence = None;
            }
            body.push(line);
            continue;
        }
        if count >= 3 {
            fence = marker.map(|c| (c, count));
            body.push(line);
            continue;
        }
        let candidate = trimmed.strip_prefix('@').and_then(|s| {
            let end = s.find(char::is_whitespace).unwrap_or(s.len());
            let name = &s[..end];
            matches!(
                name,
                "param"
                    | "arg"
                    | "argument"
                    | "return"
                    | "returns"
                    | "throws"
                    | "exception"
                    | "see"
                    | "deprecated"
                    | "example"
            )
            .then_some((name, s[end..].trim_start()))
        });
        if let Some((name, text)) =
            candidate.filter(|_| !line.starts_with("    ") && !line.starts_with('\t'))
        {
            if !body.is_empty() || !tag.is_empty() {
                parts.push(render_tag(tag, &body.join("\n")));
            }
            tag = name;
            body = vec![text];
        } else {
            body.push(line);
        }
    }
    if !body.is_empty() || !tag.is_empty() {
        parts.push(render_tag(tag, &body.join("\n")));
    }
    parts
        .into_iter()
        .filter(|p| !p.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn render_tag(tag: &str, body: &str) -> String {
    let body = body.trim();
    let (ty, text) = doc_type(body);
    match tag {
        "" => body.to_string(),
        "param" | "arg" | "argument" => {
            let end = if text.starts_with('[') {
                text.find(']').map(|i| i + 1)
            } else {
                text.find(char::is_whitespace)
            }
            .unwrap_or(text.len());
            let (name, description) = text.split_at(end);
            let description = description
                .trim_start()
                .trim_start_matches("- ")
                .trim_start();
            let ty = ty.map_or(String::new(), |ty| format!(" (`{ty}`)"));
            let description = if description.is_empty() {
                String::new()
            } else {
                format!(": {description}")
            };
            format!("**Parameter** `{name}`{ty}{description}")
        }
        "return" | "returns" | "throws" | "exception" => {
            let label = if matches!(tag, "return" | "returns") {
                "Returns"
            } else {
                "Throws"
            };
            let ty = ty.map_or(String::new(), |ty| format!("`{ty}` "));
            format!("**{label}**\n\n{ty}{text}").trim_end().to_string()
        }
        "example" => {
            let example = if body.starts_with("```") || body.starts_with("~~~") {
                body.to_string()
            } else {
                // A longer fence keeps backticks in sample code intact.
                let mut fence = "```".to_string();
                while body.contains(&fence) {
                    fence.push('`');
                }
                format!("{fence}haxe\n{body}\n{fence}")
            };
            format!("**Example**\n\n{example}")
        }
        "see" => format!("**See also**\n\n{body}"),
        "deprecated" => format!("**Deprecated**\n\n{body}").trim_end().to_string(),
        _ => body.to_string(),
    }
}

/// Optional JSDoc `{Type}` prefix, including nested object type braces.
fn doc_type(body: &str) -> (Option<&str>, &str) {
    if !body.starts_with('{') {
        return (None, body);
    }
    let mut depth = 0;
    for (i, c) in body.char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return (Some(&body[1..i]), body[i + 1..].trim_start());
                }
            }
            _ => {}
        }
    }
    (None, body)
}

fn normalize_block(body: &str) -> String {
    let body = body.strip_prefix('*').unwrap_or(body);
    let body = body.strip_suffix('*').unwrap_or(body);
    let mut lines: Vec<_> = body.lines().collect();
    let first = if lines.first().is_some_and(|s| !s.trim().is_empty()) {
        Some(lines.remove(0).trim())
    } else {
        None
    };
    while lines.first().is_some_and(|s| s.trim().is_empty()) {
        lines.remove(0);
    }
    while lines.last().is_some_and(|s| s.trim().is_empty()) {
        lines.pop();
    }
    let starred = lines
        .iter()
        .all(|s| s.trim().is_empty() || s.trim_start().starts_with('*'));
    let rest = if starred {
        lines
            .iter()
            .map(|line| {
                let text = line.trim_start().strip_prefix('*').unwrap_or("");
                text.strip_prefix(' ').unwrap_or(text).trim_end()
            })
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        let indent = lines
            .iter()
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.len() - s.trim_start().len())
            .min()
            .unwrap_or(0);
        lines
            .iter()
            .map(|s| s.get(indent..).unwrap_or("").trim_end())
            .collect::<Vec<_>>()
            .join("\n")
    };
    match first {
        Some(first) if !rest.is_empty() => format!("{first}\n{rest}"),
        Some(first) => first.to_string(),
        None => rest,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(source: &str, name: &str) -> Option<String> {
        let tokens = Lexer::new(source).tokenize().unwrap();
        let index = tokens
            .iter()
            .position(|t| &source[t.span.start..t.span.end] == name)
            .unwrap();
        comment_before(source, &tokens, index)
    }

    #[test]
    fn preserves_markdown_across_metadata_and_modifiers() {
        let source = "/**\n * Adds **values**.\n *\n * ```haxe\n *   add(1);\n * ```\n */\n@:native(\"/* no */\")\npublic static inline function add() {}";
        assert_eq!(
            doc(source, "add").as_deref(),
            Some("Adds **values**.\n\n```haxe\n  add(1);\n```")
        );
        assert_eq!(
            doc("/** Haxe-style.\n    Indented.\n*/\nclass C {}", "C").as_deref(),
            Some("Haxe-style.\nIndented.")
        );
    }

    #[test]
    fn accepts_adjacent_comments_but_not_trailing_or_detached_comments() {
        assert_eq!(
            doc("// First.\n// Second.\nvar count:Int;", "count").as_deref(),
            Some("First.\nSecond.")
        );
        assert_eq!(doc("var old = 1; // Old.\nvar count:Int;", "count"), None);
        assert_eq!(doc("/* License. */\n\nclass C {}", "C"), None);
        assert_eq!(
            doc("var old = \"/* Old. */\";\nvar count:Int;", "count"),
            None
        );
        assert_eq!(
            doc("/** Old. */\nvar old:Int;\nvar count:Int;", "count"),
            None
        );
        assert_eq!(
            doc("class C { /** Count. **/ var count:Int; }", "count").as_deref(),
            Some("Count.")
        );
        assert_eq!(
            doc("/**\n\tThe number.\n\t**/\nvar count:Int;", "count").as_deref(),
            Some("The number.")
        );
    }

    #[test]
    fn formats_haxedoc_and_jsdoc_tags_without_rewriting_examples() {
        for end in ["*/", "**/"] {
            let source = format!(
                "/**\n * Adds a value.\n * @param {{Int}} [value=1] - The value.\n *   Continued description.\n * @returns {{Int}} The sum.\n * @throws {{Error}} If invalid.\n * @see Other.add\n * @deprecated Use sum instead.\n * @example\n * ```haxe\n * @param // This is sample text.\n * ```\n {end}\nfunction add() {{}}"
            );
            let doc = doc(&source, "add").unwrap();
            for expected in [
                "Adds a value.",
                "**Parameter** `[value=1]` (`Int`): The value.\n  Continued description.",
                "**Returns**\n\n`Int` The sum.",
                "**Throws**\n\n`Error` If invalid.",
                "**See also**\n\nOther.add",
                "**Deprecated**\n\nUse sum instead.",
                "**Example**\n\n```haxe\n@param // This is sample text.\n```",
            ] {
                assert!(doc.contains(expected), "{expected}: {doc}");
            }
        }
        assert_eq!(
            render_tags("@param value Input.\n@return Result."),
            "**Parameter** `value`: Input.\n\n**Returns**\n\nResult."
        );
        assert_eq!(
            render_tags("@example\nadd(1);"),
            "**Example**\n\n```haxe\nadd(1);\n```"
        );
        assert_eq!(render_tags("@custom Keep this."), "@custom Keep this.");
    }
}
