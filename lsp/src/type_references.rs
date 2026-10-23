//! References to types that have no expression node in the typed tree.

use compiler::compilation::CompilationUnit;
use compiler::tast::references::SymbolReference;
use compiler::tast::symbols::SymbolKind;
use compiler::tast::{SourceLocation, SymbolId};
use parser::token::TokenKind;
use std::collections::{BTreeMap, BTreeSet};

pub(crate) fn record(
    file: &parser::HaxeFile,
    source: &str,
    file_id: u32,
    unit: &mut CompilationUnit,
    workspace: &crate::workspace_index::WorkspaceIndex,
) {
    let Ok(tokens) = parser::lexer::Lexer::new(source).tokenize() else {
        return;
    };
    let mut types = BTreeMap::<String, SymbolId>::new();
    for import in &file.imports {
        if let Some(path) = unit
            .namespace_resolver
            .resolve_qualified_path_to_file_force(&import.path.join("."))
        {
            workspace.ensure_disk_file(&path);
        }
    }
    for imported in unit.stdlib_files.iter().chain(&unit.user_files) {
        workspace.ensure_parsed_file(imported);
    }
    let mut qualified_types = BTreeSet::new();
    for (index, _) in tokens.iter().enumerate().filter(|(_, token)| {
        token.kind == TokenKind::Ident && source.as_bytes()[token.span.start].is_ascii_uppercase()
    }) {
        let mut first = index;
        while first >= 2
            && tokens[first - 1].kind == TokenKind::Dot
            && tokens[first - 2].kind == TokenKind::Ident
        {
            first -= 2;
        }
        if first != index && source.as_bytes()[tokens[first].span.start].is_ascii_lowercase() {
            qualified_types.insert(
                tokens[first..=index]
                    .iter()
                    .map(|token| &source[token.span.start..token.span.end])
                    .collect::<String>(),
            );
        }
    }
    for path in qualified_types {
        if let Some(path) = unit
            .namespace_resolver
            .resolve_qualified_path_to_file_force(&path)
        {
            workspace.ensure_disk_file(&path);
        }
    }
    let source_types = workspace.type_identities();
    for symbol in unit.symbol_table.all_symbols().filter(|s| {
        matches!(
            s.kind,
            SymbolKind::Class
                | SymbolKind::Interface
                | SymbolKind::Enum
                | SymbolKind::Abstract
                | SymbolKind::TypeAlias
        )
    }) {
        let path = symbol.qualified_name.unwrap_or_else(|| {
            unit.symbol_table.build_qualified_name(
                symbol.id,
                &unit.scope_tree,
                &unit.string_interner,
            )
        });
        if let Some(path) = unit.string_interner.get(path) {
            let slot = types.entry(path.to_string()).or_insert(symbol.id);
            if symbol.definition_location.is_valid() {
                *slot = symbol.id;
            }
            if let Some(declaration) = source_types.get(path) {
                types
                    .entry(declaration.qualified.clone())
                    .or_insert(symbol.id);
            }
        }
    }
    let package = file
        .package
        .as_ref()
        .map(|p| p.path.join("."))
        .unwrap_or_default();
    let prefix = if package.is_empty() {
        String::new()
    } else {
        format!("{package}.")
    };
    let mut visible: BTreeMap<String, String> = types
        .keys()
        .filter_map(|path| {
            let name = path.strip_prefix(&prefix)?;
            (!name.contains('.')).then(|| (name.to_string(), path.clone()))
        })
        .collect();
    for import in &file.imports {
        let path = import.path.join(".");
        match &import.mode {
            parser::ImportMode::Normal | parser::ImportMode::Alias(_) => {
                let name = match &import.mode {
                    parser::ImportMode::Alias(alias) => alias.as_str(),
                    _ => import.path.last().map(String::as_str).unwrap_or(""),
                };
                if types.contains_key(&path) {
                    visible.insert(name.into(), path);
                }
            }
            parser::ImportMode::Wildcard | parser::ImportMode::WildcardWithExclusions(_) => {
                let prefix = format!("{path}.");
                for path in types.keys() {
                    if let Some(name) = path.strip_prefix(&prefix)
                        && !name.contains('.')
                        && !matches!(&import.mode, parser::ImportMode::WildcardWithExclusions(excluded) if excluded.iter().any(|s|s == name))
                    {
                        visible.entry(name.into()).or_insert_with(|| path.clone());
                    }
                }
            }
            _ => {}
        }
    }
    let mut occupied: BTreeSet<_> = unit
        .symbol_references
        .iter()
        .filter(|r| r.at.file_id == file_id)
        .map(|r| r.at.byte_offset as usize)
        .collect();
    occupied.extend(
        unit.symbol_table
            .all_symbols()
            .filter(|s| {
                s.definition_location.is_valid() && s.definition_location.file_id == file_id
            })
            .map(|s| s.definition_location.byte_offset as usize),
    );
    for (index, token) in tokens
        .iter()
        .enumerate()
        .filter(|(_, t)| t.kind == TokenKind::Ident)
    {
        if occupied.contains(&token.span.start) {
            continue;
        }
        let name = &source[token.span.start..token.span.end];
        let mut first = index;
        while first >= 2
            && tokens[first - 1].kind == TokenKind::Dot
            && tokens[first - 2].kind == TokenKind::Ident
        {
            first -= 2;
        }
        let symbol = if first != index {
            let path = tokens[first..=index]
                .iter()
                .map(|t| &source[t.span.start..t.span.end])
                .collect::<String>();
            types.get(&path)
        } else {
            // A missing expression reference must not turn a shadowing local
            // into a similarly named type from another module.
            if unit.symbol_table.all_symbols().any(|s| {
                matches!(s.kind, SymbolKind::Variable | SymbolKind::Parameter)
                    && unit.string_interner.get(s.name) == Some(name)
                    && s.definition_location.file_id == file_id
                    && s.definition_location.byte_offset as usize <= token.span.start
            }) {
                continue;
            }
            visible.get(name).and_then(|path| types.get(path))
        };
        let Some(&symbol) = symbol else {
            continue;
        };
        let before = &source[..token.span.start];
        let line = before.bytes().filter(|&b| b == b'\n').count() as u32 + 1;
        let column = before
            .rsplit('\n')
            .next()
            .unwrap_or("")
            .encode_utf16()
            .count() as u32
            + 1;
        unit.symbol_references.push(SymbolReference {
            symbol,
            at: SourceLocation::new(file_id, line, column, token.span.start as u32),
            len: name.encode_utf16().count() as u32,
            type_id: compiler::tast::TypeId::invalid(),
        });
    }
}
