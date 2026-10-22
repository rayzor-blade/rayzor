//! Source identities for fields of structural typedefs.

use compiler::compilation::CompilationUnit;
use compiler::tast::{SourceLocation, SymbolId, TypeId, TypeKind};
use parser::{AnonField, Type};

pub(crate) fn source_fields(ty: &Type) -> Vec<&AnonField> {
    match ty {
        Type::Anonymous { fields, .. } => fields.iter().collect(),
        Type::Parenthesis { inner, .. } | Type::Optional { inner, .. } => source_fields(inner),
        Type::Intersection { left, right, .. } => {
            let mut fields = source_fields(left);
            fields.extend(source_fields(right));
            fields
        }
        _ => Vec::new(),
    }
}

pub(crate) fn alias_owner(mut ty: TypeId, unit: &CompilationUnit) -> Option<SymbolId> {
    let table = unit.type_table.borrow();
    for _ in 0..64 {
        match &table.get(ty)?.kind {
            TypeKind::TypeAlias {
                symbol_id,
                target_type,
                ..
            } => {
                if matches!(table.get(*target_type)?.kind, TypeKind::Anonymous { .. }) {
                    return Some(*symbol_id);
                }
                ty = *target_type;
            }
            TypeKind::Optional { inner_type } => ty = *inner_type,
            TypeKind::Reference { target_type, .. } => ty = *target_type,
            _ => return None,
        }
    }
    None
}

pub(crate) fn record(
    file: &parser::HaxeFile,
    source: &str,
    file_id: u32,
    unit: &mut CompilationUnit,
) {
    let Ok(tokens) = parser::lexer::Lexer::new(source).tokenize() else {
        return;
    };
    let comments = crate::documentation::SourceComments::new(source);
    let package = file
        .package
        .as_ref()
        .map(|p| p.path.join("."))
        .unwrap_or_default();
    for declaration in &file.declarations {
        let parser::TypeDeclaration::Typedef(alias) = declaration else {
            continue;
        };
        let owner = if package.is_empty() {
            alias.name.clone()
        } else {
            format!("{package}.{}", alias.name)
        };
        let target = unit.symbol_table.all_symbols().find_map(|symbol| {
            let qualified = symbol
                .qualified_name
                .and_then(|name| unit.string_interner.get(name));
            if qualified != Some(owner.as_str()) {
                return None;
            }
            let table = unit.type_table.borrow();
            matches!(table.get(symbol.type_id)?.kind, TypeKind::TypeAlias { .. })
                .then_some(symbol.type_id)
        });
        let Some(target) = target else { continue };
        let mut pending = vec![(owner, &alias.type_def, target)];
        while let Some((owner, ast_type, target)) = pending.pop() {
            let fields = {
                let table = unit.type_table.borrow();
                match table
                    .get(table.resolve_alias_type(target))
                    .map(|ty| &ty.kind)
                {
                    Some(TypeKind::Anonymous { fields }) => fields.clone(),
                    _ => continue,
                }
            };
            for field in source_fields(ast_type) {
                let Some(typed) = fields.iter().find(|typed| {
                    unit.string_interner.get(typed.name) == Some(field.name.as_str())
                }) else {
                    continue;
                };
                let Some(token) = tokens.iter().find(|token| {
                    token.kind == parser::token::TokenKind::Ident
                        && token.span.start >= field.span.start
                        && token.span.end <= field.span.end
                        && source.get(token.span.start..token.span.end) == Some(field.name.as_str())
                }) else {
                    continue;
                };
                let qualified = format!("{owner}.{}", field.name);
                let interned = unit.string_interner.intern(&qualified);
                let existing = unit
                    .symbol_table
                    .all_symbols()
                    .find(|s| s.qualified_name == Some(interned))
                    .map(|s| s.id);
                let id = existing.unwrap_or_else(|| unit.symbol_table.create_field(typed.name));
                let before = &source[..token.span.start];
                let line = before.bytes().filter(|&b| b == b'\n').count() as u32 + 1;
                let column = before
                    .rsplit('\n')
                    .next()
                    .unwrap_or("")
                    .encode_utf16()
                    .count() as u32
                    + 1;
                let doc = comments
                    .as_ref()
                    .and_then(|comments| comments.at(field.span.start))
                    .map(|doc| unit.string_interner.intern(&doc));
                let symbol = unit.symbol_table.get_symbol_mut(id).unwrap();
                symbol.type_id = typed.type_id;
                symbol.qualified_name = Some(interned);
                symbol.definition_location =
                    SourceLocation::new(file_id, line, column, token.span.start as u32);
                symbol.documentation = doc;
                if field.optional {
                    symbol
                        .flags
                        .insert(compiler::tast::symbols::SymbolFlags::OPTIONAL);
                }
                pending.push((qualified, &field.type_hint, typed.type_id));
            }
        }
    }
}
