//! Symbol analysis for LSP features.
//!
//! Builds a per-file index mapping source positions to symbols,
//! enabling hover, go-to-definition, completions, and find-references.

use compiler::tast::{StringInterner, SymbolId, SymbolTable, TypeId, TypeTable};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

/// Index mapping source positions to symbols for a single file.
/// Built by walking the TAST after compilation.
pub struct FileSymbolIndex {
    /// file_id for this file (matches SourceLocation.file_id in symbols)
    pub file_id: u32,
    /// Definition positions: (line, col) → SymbolId
    /// Symbols defined in this file (class names, method names, variables)
    pub definitions: BTreeMap<(u32, u32), SymbolId>,
    /// All symbols defined in this file, for iteration
    pub defined_symbols: Vec<SymbolId>,
}

impl FileSymbolIndex {
    /// Build the index from the symbol table for a given file_id.
    pub fn build(symbol_table: &SymbolTable, file_id: u32) -> Self {
        let mut definitions = BTreeMap::new();
        let mut defined_symbols = Vec::new();

        for sym in symbol_table.all_symbols() {
            let loc = sym.definition_location;
            if loc.file_id == file_id && loc.is_valid() {
                definitions.insert((loc.line, loc.column), sym.id);
                defined_symbols.push(sym.id);
            }
        }

        Self {
            file_id,
            definitions,
            defined_symbols,
        }
    }

    /// Find the symbol closest to the given position.
    /// Looks for a definition at (line, col), then searches the same line
    /// for the nearest symbol by column.
    pub fn find_symbol_at(
        &self,
        line: u32,
        col: u32,
        symbol_table: &SymbolTable,
        interner: &StringInterner,
    ) -> Option<SymbolId> {
        // Exact match
        if let Some(&sym_id) = self.definitions.get(&(line, col)) {
            return Some(sym_id);
        }

        // Search same line: find symbol whose name spans the cursor position
        let mut best: Option<(u32, SymbolId)> = None;
        for (&(def_line, def_col), &sym_id) in &self.definitions {
            if def_line != line {
                continue;
            }
            // Check if cursor is within the symbol name span
            let name_len = symbol_table
                .get_symbol(sym_id)
                .and_then(|s| interner.get(s.name))
                .map(|n| n.len() as u32)
                .unwrap_or(1);

            if col >= def_col && col < def_col + name_len {
                return Some(sym_id);
            }

            // Track closest symbol before cursor position
            if def_col <= col {
                let dist = col - def_col;
                if best.map(|(d, _)| dist < d).unwrap_or(true) {
                    best = Some((dist, sym_id));
                }
            }
        }

        // Return closest if within reasonable range (10 chars)
        best.filter(|(dist, _)| *dist < 10).map(|(_, id)| id)
    }
}

/// Format hover information for a symbol.
pub fn format_hover(
    sym_id: SymbolId,
    symbol_table: &SymbolTable,
    type_table: &Rc<RefCell<TypeTable>>,
    interner: &StringInterner,
) -> Option<String> {
    let sym = symbol_table.get_symbol(sym_id)?;
    let name = interner.get(sym.name)?;
    let type_table = type_table.borrow();

    let mut parts = Vec::new();

    // Code block with signature
    let sig = format_symbol_signature(sym, &type_table, interner);
    parts.push(format!("```haxe\n{}\n```", sig));

    // Qualified context (class name)
    if let Some(qn) = sym.qualified_name.and_then(|qn| interner.get(qn)) {
        if qn != name {
            parts.push(format!("*{}*", qn));
        }
    }

    // Documentation
    if let Some(doc) = sym.documentation.and_then(|d| interner.get(d)) {
        parts.push(doc.to_string());
    }

    Some(parts.join("\n\n"))
}

/// Format a symbol's type signature as a Haxe-style string.
fn format_symbol_signature(
    sym: &compiler::tast::symbols::Symbol,
    type_table: &TypeTable,
    interner: &StringInterner,
) -> String {
    use compiler::tast::symbols::SymbolKind;

    let name = interner.get(sym.name).unwrap_or("?");
    let vis = match sym.visibility {
        compiler::tast::symbols::Visibility::Public => "public ",
        compiler::tast::symbols::Visibility::Private => "private ",
        _ => "",
    };
    let static_kw = if sym.flags.contains(compiler::tast::symbols::SymbolFlags::STATIC) {
        "static "
    } else {
        ""
    };

    match sym.kind {
        SymbolKind::Function => {
            let type_str = format_type(sym.type_id, type_table, interner);
            format!("{}{}function {}{}", vis, static_kw, name, type_str)
        }
        SymbolKind::Variable | SymbolKind::Field => {
            let type_str = format_type_name(sym.type_id, type_table, interner);
            format!("{}{}var {}:{}", vis, static_kw, name, type_str)
        }
        SymbolKind::Class => format!("{}class {}", vis, name),
        SymbolKind::Interface => format!("{}interface {}", vis, name),
        SymbolKind::Enum => format!("{}enum {}", vis, name),
        SymbolKind::TypeAlias => {
            let type_str = format_type_name(sym.type_id, type_table, interner);
            format!("typedef {} = {}", name, type_str)
        }
        _ => {
            let type_str = format_type_name(sym.type_id, type_table, interner);
            format!("{}:{}", name, type_str)
        }
    }
}

/// Format a function type as `(param:Type, ...):ReturnType`.
fn format_type(type_id: TypeId, type_table: &TypeTable, interner: &StringInterner) -> String {
    if let Some(ti) = type_table.get(type_id) {
        if let compiler::tast::TypeKind::Function {
            params,
            return_type,
            ..
        } = &ti.kind
        {
            let param_strs: Vec<String> = params
                .iter()
                .map(|p| format_type_name(*p, type_table, interner))
                .collect();
            let ret = format_type_name(*return_type, type_table, interner);
            return format!("({}):{}",  param_strs.join(", "), ret);
        }
    }
    format_type_name(type_id, type_table, interner)
}

/// Format a type as a human-readable name.
fn format_type_name(type_id: TypeId, type_table: &TypeTable, interner: &StringInterner) -> String {
    if let Some(ti) = type_table.get(type_id) {
        match &ti.kind {
            compiler::tast::TypeKind::Void => "Void".to_string(),
            compiler::tast::TypeKind::Bool => "Bool".to_string(),
            compiler::tast::TypeKind::Int => "Int".to_string(),
            compiler::tast::TypeKind::Float => "Float".to_string(),
            compiler::tast::TypeKind::String => "String".to_string(),
            compiler::tast::TypeKind::Dynamic => "Dynamic".to_string(),
            compiler::tast::TypeKind::Class { symbol_id, .. } => {
                interner
                    .get(
                        symbol_table_name(*symbol_id, type_table)
                            .unwrap_or_default(),
                    )
                    .unwrap_or("Class")
                    .to_string()
            }
            compiler::tast::TypeKind::Array { element_type, .. } => {
                format!("Array<{}>", format_type_name(*element_type, type_table, interner))
            }
            compiler::tast::TypeKind::Optional { inner_type, .. } => {
                format!("Null<{}>", format_type_name(*inner_type, type_table, interner))
            }
            compiler::tast::TypeKind::Function { params, return_type, .. } => {
                let param_strs: Vec<String> = params
                    .iter()
                    .map(|p| format_type_name(*p, type_table, interner))
                    .collect();
                let ret = format_type_name(*return_type, type_table, interner);
                format!("({}) -> {}", param_strs.join(", "), ret)
            }
            _ => format!("{}", ti.kind),
        }
    } else {
        "Unknown".to_string()
    }
}

fn symbol_table_name(
    _symbol_id: SymbolId,
    _type_table: &TypeTable,
) -> Option<compiler::tast::InternedString> {
    // TODO: look up symbol name from type_table's class info
    None
}

/// Collect completion items from the symbol table for a given scope context.
pub fn collect_completions(
    symbol_table: &SymbolTable,
    type_table: &Rc<RefCell<TypeTable>>,
    interner: &StringInterner,
    file_id: u32,
) -> Vec<CompletionEntry> {
    let mut entries = Vec::new();
    let type_table = type_table.borrow();

    for sym in symbol_table.all_symbols() {
        // Skip internal/anonymous symbols
        let name = match interner.get(sym.name) {
            Some(n) if !n.is_empty() && !n.starts_with('_') && !n.starts_with("__") => n,
            _ => continue,
        };

        // Skip symbols from different files (unless exported)
        if sym.definition_location.file_id != file_id && !sym.is_exported {
            continue;
        }

        let kind = match sym.kind {
            compiler::tast::symbols::SymbolKind::Function => CompletionKind::Function,
            compiler::tast::symbols::SymbolKind::Variable => CompletionKind::Variable,
            compiler::tast::symbols::SymbolKind::Field => CompletionKind::Field,
            compiler::tast::symbols::SymbolKind::Class => CompletionKind::Class,
            compiler::tast::symbols::SymbolKind::Interface => CompletionKind::Interface,
            compiler::tast::symbols::SymbolKind::Enum => CompletionKind::Enum,
            compiler::tast::symbols::SymbolKind::TypeAlias => CompletionKind::TypeAlias,
            _ => CompletionKind::Variable,
        };

        let detail = format_type_name(sym.type_id, &type_table, interner);
        let doc = sym
            .documentation
            .and_then(|d| interner.get(d))
            .map(|s| s.to_string());

        entries.push(CompletionEntry {
            label: name.to_string(),
            kind,
            detail,
            documentation: doc,
        });
    }

    entries
}

/// A completion entry ready for LSP conversion.
pub struct CompletionEntry {
    pub label: String,
    pub kind: CompletionKind,
    pub detail: String,
    pub documentation: Option<String>,
}

#[derive(Clone, Copy)]
pub enum CompletionKind {
    Function,
    Variable,
    Field,
    Class,
    Interface,
    Enum,
    TypeAlias,
}

impl CompletionKind {
    pub fn to_lsp(self) -> lsp_types::CompletionItemKind {
        match self {
            Self::Function => lsp_types::CompletionItemKind::FUNCTION,
            Self::Variable => lsp_types::CompletionItemKind::VARIABLE,
            Self::Field => lsp_types::CompletionItemKind::FIELD,
            Self::Class => lsp_types::CompletionItemKind::CLASS,
            Self::Interface => lsp_types::CompletionItemKind::INTERFACE,
            Self::Enum => lsp_types::CompletionItemKind::ENUM,
            Self::TypeAlias => lsp_types::CompletionItemKind::TYPE_PARAMETER,
        }
    }
}
