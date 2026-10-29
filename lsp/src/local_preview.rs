//! Provisional local facts from a disposable compiler graph.

use crate::{
    analysis,
    binary_response::Responses,
    editor_cache::{Name, Snapshot},
};
use compiler::{
    compilation::{CompilationConfig, CompilationUnit},
    tast::{SymbolId, TypeId, TypeKind},
};
use serde_json::json;
use std::{collections::HashSet, path::PathBuf};

pub(crate) fn infer(
    context: &crate::context::LspContext,
    uri: &str,
    source: &str,
    config: CompilationConfig,
    paths: &[PathBuf],
) -> Option<Snapshot> {
    let started = std::time::Instant::now();
    let _preview = compiler::macro_system::budget::preview();
    let filename = crate::context::uri_to_path(uri);
    let mut preprocessor = parser::preprocessor::PreprocessorConfig::default();
    preprocessor.add_defines(config.user_defines.iter().chain(&config.extra_defines));
    preprocessor.defines.insert("eval".into());
    let text = parser::preprocessor::preprocess(source, &preprocessor);
    let mut unit = CompilationUnit::new_editor_preview(config);
    for path in paths {
        unit.add_source_path(path.clone());
    }
    unit.load_stdlib().ok()?;
    unit.add_file(source, &filename).ok()?;
    // RAYZOR_LSP_PREVIEW_TRACE logs provisional types and errors without changing inference.
    let trace = std::env::var_os("RAYZOR_LSP_PREVIEW_TRACE").is_some();
    if let Err(errors) = unit.lower_to_tast()
        && trace
    {
        for error in errors.iter().take(8) {
            eprintln!("lsp-preview-error: {}", error.message);
        }
    }
    context.record_editor_references(&filename, &text, &mut unit);
    let file_id = unit.file_id_of(&filename)?;
    let index = analysis::FileSymbolIndex::build(
        &unit.symbol_table,
        &unit.string_interner,
        &unit.symbol_references,
        file_id,
        source,
    );
    let mut snapshot = Snapshot::default();
    for (&(line, column), &(len, id)) in &index.names {
        compiler::cooperative::checkpoint();
        let Some(symbol) = unit.symbol_table.get_symbol(id) else {
            continue;
        };
        use compiler::tast::symbols::SymbolKind;
        let contract = matches!(
            symbol.kind,
            SymbolKind::Class
                | SymbolKind::Interface
                | SymbolKind::Abstract
                | SymbolKind::Enum
                | SymbolKind::TypeAlias
                | SymbolKind::Function
        );
        if !contract
            && !matches!(
                symbol.kind,
                compiler::tast::symbols::SymbolKind::Variable
                    | compiler::tast::symbols::SymbolKind::Parameter
                    | compiler::tast::symbols::SymbolKind::Field
                    | compiler::tast::symbols::SymbolKind::Property
                    | compiler::tast::symbols::SymbolKind::EnumVariant
            )
        {
            continue;
        }
        let offset = symbol.definition_location.byte_offset as usize;
        let name = unit.string_interner.get(symbol.name)?;
        let explicit = unit
            .source_of(symbol.definition_location.file_id)
            .and_then(|source| source.get(offset..))
            .and_then(|source| source.strip_prefix(name))
            .is_some_and(|after| after.trim_start().starts_with(':'));
        let type_id = index.type_at(line, column).unwrap_or(symbol.type_id);
        let resolved = contract || known(type_id, &unit, explicit, &mut HashSet::new());
        if trace {
            eprintln!(
                "lsp-preview-name: {line}:{column} {name}:{} resolved={resolved}",
                analysis::format_type_name(
                    type_id,
                    &unit.type_table.borrow(),
                    &unit.string_interner,
                    &unit.symbol_table
                )
            );
        }
        if !resolved {
            continue;
        }
        let Some(mut hover) =
            context.render_hover(&unit, &index, source, id, (line, column), contract)
        else {
            continue;
        };
        hover.push_str("\n\n*Provisional while indexing.*");
        let mut responses = Responses::from([(
            "textDocument/hover",
            json!({"contents":{"kind":"markdown","value":hover}}),
        )]);
        responses.mark_syntax();
        snapshot.names.push(Name {
            line: line - 1,
            column: column - 1,
            len,
            metadata: None,
            responses,
        });
    }
    if std::env::var_os("RAYZOR_LSP_PROFILE").is_some() {
        eprintln!(
            "lsp-preview: total_ms={:.2} names={} file={filename}",
            started.elapsed().as_secs_f64() * 1000.0,
            snapshot.names.len()
        );
    }
    snapshot.dependencies = unit.macro_dependencies.snapshot();
    Some(snapshot)
}

fn known(id: TypeId, unit: &CompilationUnit, explicit: bool, seen: &mut HashSet<TypeId>) -> bool {
    if !seen.insert(id) {
        return true;
    }
    let table = unit.type_table.borrow();
    let Some(ty) = table.get(id) else {
        return false;
    };
    let named = |symbol: SymbolId| {
        unit.symbol_table
            .display_type_path(symbol, &unit.string_interner)
            .is_none_or(|path| !path.starts_with("haxe.macro."))
    };
    let check = |id, seen: &mut HashSet<TypeId>| known(id, unit, explicit, seen);
    match &ty.kind {
        TypeKind::Unknown | TypeKind::Error | TypeKind::Placeholder { .. } | TypeKind::Void => {
            false
        }
        TypeKind::Dynamic | TypeKind::TypeParameter { .. } => explicit,
        TypeKind::Class {
            symbol_id,
            type_args,
        }
        | TypeKind::Interface {
            symbol_id,
            type_args,
        }
        | TypeKind::Enum {
            symbol_id,
            type_args,
        }
        | TypeKind::Abstract {
            symbol_id,
            type_args,
            ..
        } => named(*symbol_id) && type_args.iter().all(|id| check(*id, seen)),
        TypeKind::TypeAlias {
            symbol_id,
            target_type,
            type_args,
        } => {
            named(*symbol_id)
                && type_args.iter().all(|id| check(*id, seen))
                && check(*target_type, seen)
        }
        TypeKind::Function {
            params,
            return_type,
            ..
        } => {
            params.iter().all(|id| check(*id, seen))
                && (table
                    .get(*return_type)
                    .is_some_and(|ty| matches!(ty.kind, TypeKind::Void))
                    || check(*return_type, seen))
        }
        TypeKind::Array { element_type } => check(*element_type, seen),
        TypeKind::Map {
            key_type,
            value_type,
        } => check(*key_type, seen) && check(*value_type, seen),
        TypeKind::Optional { inner_type } => check(*inner_type, seen),
        TypeKind::GenericInstance {
            base_type,
            type_args,
            ..
        } => check(*base_type, seen) && type_args.iter().all(|id| check(*id, seen)),
        TypeKind::Anonymous { fields } => fields.iter().all(|field| check(field.type_id, seen)),
        TypeKind::Union { types } | TypeKind::Intersection { types } => {
            types.iter().all(|id| check(*id, seen))
        }
        TypeKind::Reference { target_type, .. } => check(*target_type, seen),
        _ => true,
    }
}
