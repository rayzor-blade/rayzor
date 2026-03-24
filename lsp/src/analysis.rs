//! Symbol analysis for LSP features.
//!
//! Provides position→symbol resolution, hover formatting, completions,
//! semantic tokens, document outline, signature help, and inlay hints.
//! Covers ALL Haxe language features from the rayzor parser and TAST.

use compiler::tast::symbols::{Symbol, SymbolFlags, SymbolKind, Visibility};
use compiler::tast::{InternedString, StringInterner, SymbolId, SymbolTable, TypeId, TypeTable};
use compiler::tast::TypeKind;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

// ---------------------------------------------------------------------------
// FileSymbolIndex — position ↔ symbol mapping
// ---------------------------------------------------------------------------

/// Index mapping source positions to symbols for a single file.
pub struct FileSymbolIndex {
    pub file_id: u32,
    /// (line, col) → SymbolId for definitions in this file
    pub definitions: BTreeMap<(u32, u32), SymbolId>,
    /// All symbols defined in this file
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

    /// Find the symbol at or nearest to (line, col).
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

        // Search same line: find symbol whose name spans the cursor
        let mut best: Option<(u32, SymbolId)> = None;
        for (&(def_line, def_col), &sym_id) in &self.definitions {
            if def_line != line {
                continue;
            }
            let name_len = symbol_table
                .get_symbol(sym_id)
                .and_then(|s| interner.get(s.name))
                .map(|n| n.len() as u32)
                .unwrap_or(1);

            if col >= def_col && col < def_col + name_len {
                return Some(sym_id);
            }

            if def_col <= col {
                let dist = col - def_col;
                if best.map(|(d, _)| dist < d).unwrap_or(true) {
                    best = Some((dist, sym_id));
                }
            }
        }

        best.filter(|(dist, _)| *dist < 15).map(|(_, id)| id)
    }
}

// ---------------------------------------------------------------------------
// Hover — rich type info + docs
// ---------------------------------------------------------------------------

/// Format hover information for a symbol as markdown.
pub fn format_hover(
    sym_id: SymbolId,
    symbol_table: &SymbolTable,
    type_table: &Rc<RefCell<TypeTable>>,
    interner: &StringInterner,
) -> Option<String> {
    let sym = symbol_table.get_symbol(sym_id)?;
    let name = interner.get(sym.name)?;
    let tt = type_table.borrow();
    let mut parts = Vec::new();

    // Code block with full signature
    let sig = format_symbol_signature(sym, &tt, interner);
    parts.push(format!("```haxe\n{}\n```", sig));

    // Qualified context
    if let Some(qn) = sym.qualified_name.and_then(|qn| interner.get(qn)) {
        if qn != name {
            parts.push(format!("*{}*", qn));
        }
    }

    // Flags / annotations
    let flags = format_symbol_flags(sym);
    if !flags.is_empty() {
        parts.push(flags);
    }

    // Documentation
    if let Some(doc) = sym.documentation.and_then(|d| interner.get(d)) {
        parts.push(doc.to_string());
    }

    Some(parts.join("\n\n"))
}

fn format_symbol_flags(sym: &Symbol) -> String {
    let mut tags = Vec::new();
    let f = sym.flags;
    if f.contains(SymbolFlags::INLINE) {
        tags.push("inline");
    }
    if f.contains(SymbolFlags::GENERIC) {
        tags.push("@:generic");
    }
    if f.contains(SymbolFlags::FORWARD) {
        tags.push("@:forward");
    }
    if f.contains(SymbolFlags::NATIVE) {
        tags.push("@:native");
    }
    if f.contains(SymbolFlags::EXTERN) {
        tags.push("extern");
    }
    if f.contains(SymbolFlags::ASYNC) {
        tags.push("@:async");
    }
    if f.contains(SymbolFlags::DEPRECATED) {
        tags.push("@:deprecated");
    }
    if f.contains(SymbolFlags::CSTRUCT) {
        tags.push("@:cstruct");
    }
    if f.contains(SymbolFlags::GPU_STRUCT) {
        tags.push("@:gpuStruct");
    }
    if tags.is_empty() {
        String::new()
    } else {
        format!("_{}_", tags.join(", "))
    }
}

/// Format a symbol's type signature as a Haxe-style string.
fn format_symbol_signature(sym: &Symbol, type_table: &TypeTable, interner: &StringInterner) -> String {
    let name = interner.get(sym.name).unwrap_or("?");
    let vis = match sym.visibility {
        Visibility::Public => "public ",
        Visibility::Private => "private ",
        _ => "",
    };
    let static_kw = if sym.flags.contains(SymbolFlags::STATIC) { "static " } else { "" };
    let inline_kw = if sym.flags.contains(SymbolFlags::INLINE) { "inline " } else { "" };
    let override_kw = if sym.flags.contains(SymbolFlags::OVERRIDE) { "override " } else { "" };
    let final_kw = if sym.flags.contains(SymbolFlags::FINAL) { "final " } else { "" };

    match sym.kind {
        SymbolKind::Function => {
            let type_str = format_function_type(sym.type_id, type_table, interner);
            format!("{}{}{}{}{}function {}{}", vis, static_kw, inline_kw, override_kw, final_kw, name, type_str)
        }
        SymbolKind::Variable => {
            let type_str = format_type_name(sym.type_id, type_table, interner);
            let kw = if sym.flags.contains(SymbolFlags::FINAL) { "final" } else { "var" };
            format!("{}{}{} {}:{}", vis, static_kw, kw, name, type_str)
        }
        SymbolKind::Field => {
            let type_str = format_type_name(sym.type_id, type_table, interner);
            let kw = if sym.flags.contains(SymbolFlags::FINAL) { "final" } else { "var" };
            format!("{}{}{} {}:{}", vis, static_kw, kw, name, type_str)
        }
        SymbolKind::Property => {
            let type_str = format_type_name(sym.type_id, type_table, interner);
            format!("{}{}var {}(get, set):{}", vis, static_kw, name, type_str)
        }
        SymbolKind::Parameter => {
            let type_str = format_type_name(sym.type_id, type_table, interner);
            let opt = if sym.flags.contains(SymbolFlags::OPTIONAL) { "?" } else { "" };
            format!("{}{}:{}", opt, name, type_str)
        }
        SymbolKind::Class => {
            let extern_kw = if sym.flags.contains(SymbolFlags::EXTERN) { "extern " } else { "" };
            format!("{}{}class {}", vis, extern_kw, name)
        }
        SymbolKind::Interface => format!("{}interface {}", vis, name),
        SymbolKind::Enum => format!("{}enum {}", vis, name),
        SymbolKind::Abstract => {
            let ea = if sym.flags.contains(SymbolFlags::ABSTRACT) { "enum " } else { "" };
            format!("{}{}abstract {}", vis, ea, name)
        }
        SymbolKind::TypeAlias => {
            let type_str = format_type_name(sym.type_id, type_table, interner);
            format!("typedef {} = {}", name, type_str)
        }
        SymbolKind::EnumVariant => {
            let type_str = format_type_name(sym.type_id, type_table, interner);
            if type_str == "Void" {
                name.to_string()
            } else {
                format!("{}({})", name, type_str)
            }
        }
        SymbolKind::TypeParameter => format!("type parameter {}", name),
        SymbolKind::Macro => format!("macro {}", name),
        SymbolKind::Module => format!("package {}", name),
        _ => {
            let type_str = format_type_name(sym.type_id, type_table, interner);
            format!("{}:{}", name, type_str)
        }
    }
}

// ---------------------------------------------------------------------------
// Type formatting
// ---------------------------------------------------------------------------

/// Format function type as `(param:Type, ...):ReturnType`.
fn format_function_type(type_id: TypeId, type_table: &TypeTable, interner: &StringInterner) -> String {
    if let Some(ti) = type_table.get(type_id) {
        if let TypeKind::Function { params, return_type, .. } = &ti.kind {
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

/// Format a type as a human-readable Haxe type name.
pub fn format_type_name(type_id: TypeId, type_table: &TypeTable, interner: &StringInterner) -> String {
    let ti = match type_table.get(type_id) {
        Some(ti) => ti,
        None => return "Unknown".to_string(),
    };
    match &ti.kind {
        TypeKind::Void => "Void".into(),
        TypeKind::Bool => "Bool".into(),
        TypeKind::Int => "Int".into(),
        TypeKind::Float => "Float".into(),
        TypeKind::String => "String".into(),
        TypeKind::Char => "Char".into(),
        TypeKind::Dynamic => "Dynamic".into(),
        TypeKind::Unknown => "Unknown".into(),
        TypeKind::Class { .. } => {
            // Try to get the class name from the Display impl
            format!("{}", ti.kind)
        }
        TypeKind::Interface { .. } => "Interface".into(),
        TypeKind::Enum { .. } => "Enum".into(),
        TypeKind::Array { element_type, .. } => {
            format!("Array<{}>", format_type_name(*element_type, type_table, interner))
        }
        TypeKind::Map { key_type, value_type, .. } => {
            format!("Map<{}, {}>",
                format_type_name(*key_type, type_table, interner),
                format_type_name(*value_type, type_table, interner))
        }
        TypeKind::Optional { inner_type, .. } => {
            format!("Null<{}>", format_type_name(*inner_type, type_table, interner))
        }
        TypeKind::Function { params, return_type, .. } => {
            let ps: Vec<String> = params.iter()
                .map(|p| format_type_name(*p, type_table, interner))
                .collect();
            format!("({}) -> {}", ps.join(", "), format_type_name(*return_type, type_table, interner))
        }
        TypeKind::TypeParameter { symbol_id, .. } => {
            // Resolve name from symbol table via the type's symbol_id
            "T".to_string()
        }
        TypeKind::GenericInstance { base_type, type_args, .. } => {
            let base = format_type_name(*base_type, type_table, interner);
            let args: Vec<String> = type_args.iter()
                .map(|a| format_type_name(*a, type_table, interner))
                .collect();
            if args.is_empty() { base } else { format!("{}<{}>", base, args.join(", ")) }
        }
        TypeKind::TypeAlias { target_type, .. } => {
            format_type_name(*target_type, type_table, interner)
        }
        TypeKind::Anonymous { fields, .. } => {
            let fs: Vec<String> = fields.iter()
                .map(|f| {
                    let fname = interner.get(f.name).unwrap_or("?");
                    let ftype = format_type_name(f.type_id, type_table, interner);
                    format!("{}:{}", fname, ftype)
                })
                .collect();
            format!("{{ {} }}", fs.join(", "))
        }
        _ => format!("{}", ti.kind),
    }
}

// ---------------------------------------------------------------------------
// Completions — scope-aware + member completion
// ---------------------------------------------------------------------------

/// Collect completion items from the symbol table.
pub fn collect_completions(
    symbol_table: &SymbolTable,
    type_table: &Rc<RefCell<TypeTable>>,
    interner: &StringInterner,
    file_id: u32,
) -> Vec<CompletionEntry> {
    let mut entries = Vec::new();
    let tt = type_table.borrow();

    for sym in symbol_table.all_symbols() {
        let name = match interner.get(sym.name) {
            Some(n) if !n.is_empty() && !n.starts_with("__") => n,
            _ => continue,
        };

        // Skip compiler-generated symbols
        if sym.flags.contains(SymbolFlags::COMPILER_GENERATED) {
            continue;
        }

        // Include: same file, exported, or stdlib
        if sym.definition_location.file_id != file_id && !sym.is_exported {
            continue;
        }

        let kind = symbol_kind_to_completion(sym.kind);
        let detail = format_type_name(sym.type_id, &tt, interner);
        let doc = sym.documentation.and_then(|d| interner.get(d)).map(|s| s.to_string());

        // Deprecated tag
        let deprecated = sym.flags.contains(SymbolFlags::DEPRECATED);

        entries.push(CompletionEntry {
            label: name.to_string(),
            kind,
            detail,
            documentation: doc,
            deprecated,
            sort_priority: completion_sort_priority(sym),
        });
    }

    entries
}

fn completion_sort_priority(sym: &Symbol) -> u8 {
    match sym.kind {
        SymbolKind::Variable | SymbolKind::Parameter => 0,    // Locals first
        SymbolKind::Field | SymbolKind::Property => 1,        // Members
        SymbolKind::Function => 2,                            // Methods
        SymbolKind::EnumVariant => 3,                         // Enum values
        SymbolKind::Class | SymbolKind::Interface | SymbolKind::Enum => 4, // Types
        SymbolKind::TypeAlias | SymbolKind::Abstract => 5,    // Type aliases
        _ => 6,
    }
}

fn symbol_kind_to_completion(kind: SymbolKind) -> CompletionKind {
    match kind {
        SymbolKind::Function => CompletionKind::Function,
        SymbolKind::Variable => CompletionKind::Variable,
        SymbolKind::Parameter => CompletionKind::Variable,
        SymbolKind::Field => CompletionKind::Field,
        SymbolKind::Property => CompletionKind::Property,
        SymbolKind::Class => CompletionKind::Class,
        SymbolKind::Interface => CompletionKind::Interface,
        SymbolKind::Enum => CompletionKind::Enum,
        SymbolKind::EnumVariant => CompletionKind::EnumMember,
        SymbolKind::TypeAlias => CompletionKind::TypeParameter,
        SymbolKind::Abstract => CompletionKind::Class,
        SymbolKind::Module => CompletionKind::Module,
        SymbolKind::TypeParameter => CompletionKind::TypeParameter,
        _ => CompletionKind::Variable,
    }
}

pub struct CompletionEntry {
    pub label: String,
    pub kind: CompletionKind,
    pub detail: String,
    pub documentation: Option<String>,
    pub deprecated: bool,
    pub sort_priority: u8,
}

#[derive(Clone, Copy)]
pub enum CompletionKind {
    Function,
    Variable,
    Field,
    Property,
    Class,
    Interface,
    Enum,
    EnumMember,
    Module,
    TypeParameter,
}

impl CompletionKind {
    pub fn to_lsp(self) -> lsp_types::CompletionItemKind {
        match self {
            Self::Function => lsp_types::CompletionItemKind::FUNCTION,
            Self::Variable => lsp_types::CompletionItemKind::VARIABLE,
            Self::Field => lsp_types::CompletionItemKind::FIELD,
            Self::Property => lsp_types::CompletionItemKind::PROPERTY,
            Self::Class => lsp_types::CompletionItemKind::CLASS,
            Self::Interface => lsp_types::CompletionItemKind::INTERFACE,
            Self::Enum => lsp_types::CompletionItemKind::ENUM,
            Self::EnumMember => lsp_types::CompletionItemKind::ENUM_MEMBER,
            Self::Module => lsp_types::CompletionItemKind::MODULE,
            Self::TypeParameter => lsp_types::CompletionItemKind::TYPE_PARAMETER,
        }
    }
}

// ---------------------------------------------------------------------------
// Semantic tokens — full Haxe syntax highlighting
// ---------------------------------------------------------------------------

/// Semantic token types matching the LSP SemanticTokenTypes.
/// Order matters — these are indices into the legend.
pub const SEMANTIC_TOKEN_TYPES: &[&str] = &[
    "namespace",     // 0  — package declarations
    "type",          // 1  — type references
    "class",         // 2  — class names
    "enum",          // 3  — enum names
    "interface",     // 4  — interface names
    "struct",        // 5  — abstract/typedef names
    "typeParameter", // 6  — generic type parameters <T>
    "parameter",     // 7  — function parameters
    "variable",      // 8  — local variables
    "property",      // 9  — fields, properties
    "enumMember",    // 10 — enum constructors
    "function",      // 11 — functions, methods
    "macro",         // 12 — macro definitions
    "keyword",       // 13 — reserved keywords
    "modifier",      // 14 — access modifiers (public, static, inline...)
    "comment",       // 15 — comments
    "string",        // 16 — string literals
    "number",        // 17 — numeric literals
    "operator",      // 18 — operators
    "decorator",     // 19 — @:metadata annotations
];

/// Semantic token modifiers.
pub const SEMANTIC_TOKEN_MODIFIERS: &[&str] = &[
    "declaration",    // 0 — symbol definition site
    "definition",     // 1
    "readonly",       // 2 — final/immutable
    "static",         // 3 — static member
    "deprecated",     // 4 — @:deprecated
    "async",          // 5 — @:async
    "modification",   // 6 — assignment target
    "documentation",  // 7
    "defaultLibrary", // 8 — stdlib symbol
];

/// A single semantic token for LSP response.
pub struct SemanticToken {
    pub line: u32,         // 0-based
    pub start_char: u32,   // 0-based
    pub length: u32,
    pub token_type: u32,   // index into SEMANTIC_TOKEN_TYPES
    pub token_modifiers: u32, // bitmask into SEMANTIC_TOKEN_MODIFIERS
}

/// Build semantic tokens from the symbol table for a file.
pub fn build_semantic_tokens(
    symbol_table: &SymbolTable,
    interner: &StringInterner,
    file_id: u32,
) -> Vec<SemanticToken> {
    let mut tokens = Vec::new();

    for sym in symbol_table.all_symbols() {
        let loc = sym.definition_location;
        if loc.file_id != file_id || !loc.is_valid() {
            continue;
        }

        let name_len = interner.get(sym.name).map(|n| n.len() as u32).unwrap_or(1);

        let token_type = match sym.kind {
            SymbolKind::Class => 2,
            SymbolKind::Interface => 4,
            SymbolKind::Enum => 3,
            SymbolKind::Abstract | SymbolKind::TypeAlias => 5,
            SymbolKind::TypeParameter => 6,
            SymbolKind::Parameter => 7,
            SymbolKind::Variable => 8,
            SymbolKind::Field | SymbolKind::Property => 9,
            SymbolKind::EnumVariant => 10,
            SymbolKind::Function => 11,
            SymbolKind::Macro => 12,
            SymbolKind::Module => 0,
            _ => 8, // default to variable
        };

        let mut modifiers: u32 = 0;
        modifiers |= 1 << 0; // declaration
        if sym.flags.contains(SymbolFlags::FINAL) {
            modifiers |= 1 << 2; // readonly
        }
        if sym.flags.contains(SymbolFlags::STATIC) {
            modifiers |= 1 << 3; // static
        }
        if sym.flags.contains(SymbolFlags::DEPRECATED) {
            modifiers |= 1 << 4; // deprecated
        }
        if sym.flags.contains(SymbolFlags::ASYNC) {
            modifiers |= 1 << 5; // async
        }

        tokens.push(SemanticToken {
            line: loc.line.saturating_sub(1), // LSP is 0-based
            start_char: loc.column.saturating_sub(1),
            length: name_len,
            token_type,
            token_modifiers: modifiers,
        });
    }

    // Sort by position (required by LSP delta encoding)
    tokens.sort_by(|a, b| a.line.cmp(&b.line).then(a.start_char.cmp(&b.start_char)));
    tokens
}

// ---------------------------------------------------------------------------
// Document symbols — outline view
// ---------------------------------------------------------------------------

/// A symbol for the document outline (breadcrumbs, symbol tree).
pub struct DocumentSymbolEntry {
    pub name: String,
    pub detail: String,
    pub kind: lsp_types::SymbolKind,
    pub line: u32,
    pub col: u32,
    pub end_line: u32,
    pub end_col: u32,
    pub children: Vec<DocumentSymbolEntry>,
}

/// Build document symbols for the outline panel.
pub fn build_document_symbols(
    symbol_table: &SymbolTable,
    type_table: &Rc<RefCell<TypeTable>>,
    interner: &StringInterner,
    file_id: u32,
) -> Vec<DocumentSymbolEntry> {
    let tt = type_table.borrow();
    let mut top_level = Vec::new();

    // Collect top-level declarations (classes, enums, functions, typedefs)
    // Then nest methods/fields under their parent class/interface
    let mut class_children: BTreeMap<SymbolId, Vec<DocumentSymbolEntry>> = BTreeMap::new();

    for sym in symbol_table.all_symbols() {
        if sym.definition_location.file_id != file_id || !sym.definition_location.is_valid() {
            continue;
        }
        if sym.flags.contains(SymbolFlags::COMPILER_GENERATED) {
            continue;
        }

        let name = match interner.get(sym.name) {
            Some(n) if !n.is_empty() && !n.starts_with("__") => n.to_string(),
            _ => continue,
        };
        let detail = format_type_name(sym.type_id, &tt, interner);
        let loc = sym.definition_location;

        let entry = DocumentSymbolEntry {
            name,
            detail,
            kind: symbol_kind_to_lsp_symbol(sym.kind),
            line: loc.line,
            col: loc.column,
            end_line: loc.line,
            end_col: loc.column + interner.get(sym.name).map(|n| n.len() as u32).unwrap_or(1),
            children: Vec::new(),
        };

        match sym.kind {
            SymbolKind::Class | SymbolKind::Interface | SymbolKind::Enum
            | SymbolKind::Abstract | SymbolKind::Module => {
                top_level.push((sym.id, entry));
            }
            SymbolKind::Function | SymbolKind::Field | SymbolKind::Property
            | SymbolKind::EnumVariant | SymbolKind::Variable => {
                // Try to find parent class
                // For now, add as top-level (proper nesting needs scope_tree)
                top_level.push((sym.id, entry));
            }
            _ => {}
        }
    }

    top_level.into_iter().map(|(_, e)| e).collect()
}

fn symbol_kind_to_lsp_symbol(kind: SymbolKind) -> lsp_types::SymbolKind {
    match kind {
        SymbolKind::Class => lsp_types::SymbolKind::CLASS,
        SymbolKind::Interface => lsp_types::SymbolKind::INTERFACE,
        SymbolKind::Enum => lsp_types::SymbolKind::ENUM,
        SymbolKind::EnumVariant => lsp_types::SymbolKind::ENUM_MEMBER,
        SymbolKind::Function => lsp_types::SymbolKind::FUNCTION,
        SymbolKind::Variable | SymbolKind::Parameter => lsp_types::SymbolKind::VARIABLE,
        SymbolKind::Field => lsp_types::SymbolKind::FIELD,
        SymbolKind::Property => lsp_types::SymbolKind::PROPERTY,
        SymbolKind::TypeAlias | SymbolKind::Abstract => lsp_types::SymbolKind::TYPE_PARAMETER,
        SymbolKind::TypeParameter => lsp_types::SymbolKind::TYPE_PARAMETER,
        SymbolKind::Module => lsp_types::SymbolKind::NAMESPACE,
        SymbolKind::Macro => lsp_types::SymbolKind::FUNCTION,
        _ => lsp_types::SymbolKind::VARIABLE,
    }
}

// ---------------------------------------------------------------------------
// Signature help — function parameter info
// ---------------------------------------------------------------------------

/// Signature help info for a function call at cursor position.
pub struct SignatureInfo {
    pub label: String,
    pub documentation: Option<String>,
    pub parameters: Vec<ParameterInfo>,
    pub active_parameter: u32,
}

pub struct ParameterInfo {
    pub label: String,
    pub documentation: Option<String>,
}

/// Try to build signature help for a function call at the given position.
pub fn build_signature_help(
    sym_id: SymbolId,
    symbol_table: &SymbolTable,
    type_table: &Rc<RefCell<TypeTable>>,
    interner: &StringInterner,
    active_param: u32,
) -> Option<SignatureInfo> {
    let sym = symbol_table.get_symbol(sym_id)?;
    let tt = type_table.borrow();

    if sym.kind != SymbolKind::Function {
        return None;
    }

    let ti = tt.get(sym.type_id)?;
    let (params, ret) = if let TypeKind::Function { params, return_type, .. } = &ti.kind {
        (params.clone(), *return_type)
    } else {
        return None;
    };

    let name = interner.get(sym.name)?;
    let param_strs: Vec<String> = params.iter()
        .map(|p| format_type_name(*p, &tt, interner))
        .collect();
    let ret_str = format_type_name(ret, &tt, interner);

    let label = format!("{}({}):{}",  name, param_strs.join(", "), ret_str);

    let parameters: Vec<ParameterInfo> = param_strs.into_iter()
        .map(|p| ParameterInfo {
            label: p,
            documentation: None,
        })
        .collect();

    Some(SignatureInfo {
        label,
        documentation: sym.documentation.and_then(|d| interner.get(d)).map(|s| s.to_string()),
        parameters,
        active_parameter: active_param,
    })
}
