//! Symbol analysis for LSP features.
//!
//! Position→symbol resolution over the names the compiler recorded, hover
//! formatting, completions, semantic tokens, outline and signature help.

use compiler::tast::TypeKind;
use compiler::tast::references::SymbolReference;
use compiler::tast::symbols::{Symbol, SymbolFlags, SymbolKind, Visibility};
use compiler::tast::{StringInterner, SymbolId, SymbolTable, TypeId, TypeTable};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

// ---------------------------------------------------------------------------
// FileSymbolIndex — position ↔ symbol mapping
// ---------------------------------------------------------------------------

/// The names written in one file, by position: where its symbols are
/// declared and where resolved names refer to them.
pub struct FileSymbolIndex {
    pub file_id: u32,
    /// (line, start column) → (length, symbol), both 1-based.
    pub names: BTreeMap<(u32, u32), (u32, SymbolId)>,
}

impl FileSymbolIndex {
    /// Declaration signatures remain useful when a body cannot be lowered.
    pub fn record_source_declarations(
        file: &parser::HaxeFile,
        source: &str,
        file_id: u32,
        unit: &mut compiler::compilation::CompilationUnit,
    ) {
        let Ok(tokens) = parser::lexer::Lexer::new(source).tokenize() else {
            return;
        };
        let package = file
            .package
            .as_ref()
            .map(|p| p.path.join("."))
            .unwrap_or_default();
        let qualify = |name: &str| {
            if package.is_empty() {
                name.to_string()
            } else {
                format!("{package}.{name}")
            }
        };
        let mut declarations = Vec::new();
        for declaration in &file.declarations {
            let (name, fields) = match declaration {
                parser::TypeDeclaration::Class(c) => (&c.name, c.fields.as_slice()),
                parser::TypeDeclaration::Interface(c) => (&c.name, c.fields.as_slice()),
                parser::TypeDeclaration::Abstract(c) => (&c.name, c.fields.as_slice()),
                parser::TypeDeclaration::Typedef(t) => (&t.name, &[][..]),
                parser::TypeDeclaration::Enum(e) => {
                    let owner = qualify(&e.name);
                    for constructor in &e.constructors {
                        declarations.push((
                            format!("{owner}.{}", constructor.name),
                            constructor.name.as_str(),
                            constructor.span,
                        ));
                    }
                    (&e.name, &[][..])
                }
                _ => continue,
            };
            let owner = qualify(name);
            declarations.push((owner.clone(), name.as_str(), declaration.span()));
            for field in fields {
                let (name, span) = match &field.kind {
                    parser::ClassFieldKind::Function(f) => (f.name.as_str(), f.span),
                    parser::ClassFieldKind::Var { name, .. }
                    | parser::ClassFieldKind::Final { name, .. }
                    | parser::ClassFieldKind::Property { name, .. } => (name.as_str(), field.span),
                };
                declarations.push((format!("{owner}.{name}"), name, span));
            }
        }
        let mut symbols_by_name: BTreeMap<String, Vec<SymbolId>> = BTreeMap::new();
        for symbol in unit
            .symbol_table
            .all_symbols()
            .filter(|s| !s.definition_location.is_valid())
        {
            let owner = (symbol.kind == SymbolKind::EnumVariant)
                .then(|| {
                    unit.symbol_table
                        .find_parent_enum_for_constructor(symbol.id)
                })
                .flatten();
            let name = unit.symbol_table.build_qualified_name(
                owner.unwrap_or(symbol.id),
                &unit.scope_tree,
                &unit.string_interner,
            );
            if let Some(name) = unit.string_interner.get(name) {
                let name = match owner {
                    Some(_) => format!(
                        "{name}.{}",
                        unit.string_interner.get(symbol.name).unwrap_or("")
                    ),
                    None => name.to_string(),
                };
                symbols_by_name.entry(name).or_default().push(symbol.id);
            }
        }
        for (qualified, name, span) in declarations {
            let Some(token) = tokens.iter().find(|t| {
                t.span.start >= span.start
                    && t.span.end <= span.end
                    && t.kind == parser::token::TokenKind::Ident
                    && &source[t.span.start..t.span.end] == name
            }) else {
                continue;
            };
            let ids = symbols_by_name.get(&qualified).cloned().unwrap_or_default();
            let before = &source[..token.span.start];
            let line = before.bytes().filter(|&b| b == b'\n').count() as u32 + 1;
            let column = before
                .rsplit('\n')
                .next()
                .unwrap_or("")
                .encode_utf16()
                .count() as u32
                + 1;
            for id in ids {
                unit.symbol_table
                    .get_symbol_mut(id)
                    .unwrap()
                    .definition_location = compiler::tast::SourceLocation::new(
                    file_id,
                    line,
                    column,
                    token.span.start as u32,
                );
            }
        }
    }

    pub fn build(
        symbol_table: &SymbolTable,
        interner: &StringInterner,
        references: &[SymbolReference],
        file_id: u32,
    ) -> Self {
        let mut names = BTreeMap::new();
        for r in references.iter().filter(|r| r.at.file_id == file_id) {
            names.insert((r.at.line, r.at.column), (r.len, r.symbol));
        }
        for sym in symbol_table.all_symbols() {
            let loc = sym.definition_location;
            if loc.file_id == file_id && loc.is_valid() {
                let len = interner.get(sym.name).map_or(1, |n| n.len() as u32);
                names.insert((loc.line, loc.column), (len, sym.id));
            }
        }
        Self { file_id, names }
    }

    /// The symbol whose name covers (line, col), both 1-based.
    pub fn find_symbol_at(&self, line: u32, col: u32) -> Option<SymbolId> {
        let (&(l, start), &(len, sym)) = self.names.range(..=(line, col)).next_back()?;
        (l == line && col < start + len).then_some(sym)
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
    let sig = format_symbol_signature(sym, &tt, interner, symbol_table);
    parts.push(format!("```haxe\n{}\n```", sig));
    if sym.kind == SymbolKind::Function
        && !tt
            .get(sym.type_id)
            .is_some_and(|t| matches!(t.kind, TypeKind::Function { .. }))
    {
        parts.push("*Signature not yet resolved.*".to_string());
    }

    // Qualified context
    if let Some(qn) = sym.qualified_name.and_then(|qn| interner.get(qn))
        && qn != name
    {
        parts.push(format!("*{}*", qn));
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
fn format_symbol_signature(
    sym: &Symbol,
    type_table: &TypeTable,
    interner: &StringInterner,
    symbol_table: &SymbolTable,
) -> String {
    let name = interner.get(sym.name).unwrap_or("?");
    let vis = match sym.visibility {
        Visibility::Public => "public ",
        Visibility::Private => "private ",
        _ => "",
    };
    let static_kw = if sym.flags.contains(SymbolFlags::STATIC) {
        "static "
    } else {
        ""
    };
    let inline_kw = if sym.flags.contains(SymbolFlags::INLINE) {
        "inline "
    } else {
        ""
    };
    let override_kw = if sym.flags.contains(SymbolFlags::OVERRIDE) {
        "override "
    } else {
        ""
    };
    let final_kw = if sym.flags.contains(SymbolFlags::FINAL) {
        "final "
    } else {
        ""
    };

    match sym.kind {
        SymbolKind::Function => {
            let type_str = format_function_type(sym.type_id, type_table, interner, symbol_table);
            format!(
                "{}{}{}{}{}function {}{}",
                vis, static_kw, inline_kw, override_kw, final_kw, name, type_str
            )
        }
        SymbolKind::Variable => {
            let type_str = format_type_name(sym.type_id, type_table, interner, symbol_table);
            let kw = if sym.flags.contains(SymbolFlags::FINAL) {
                "final"
            } else {
                "var"
            };
            format!("{} {}:{}", kw, name, type_str)
        }
        SymbolKind::Field => {
            let type_str = format_type_name(sym.type_id, type_table, interner, symbol_table);
            let kw = if sym.flags.contains(SymbolFlags::FINAL) {
                "final"
            } else {
                "var"
            };
            format!("{}{}{} {}:{}", vis, static_kw, kw, name, type_str)
        }
        SymbolKind::Property => {
            let type_str = format_type_name(sym.type_id, type_table, interner, symbol_table);
            format!("{}{}var {}(get, set):{}", vis, static_kw, name, type_str)
        }
        SymbolKind::Parameter => {
            let type_str = format_type_name(sym.type_id, type_table, interner, symbol_table);
            let opt = if sym.flags.contains(SymbolFlags::OPTIONAL) {
                "?"
            } else {
                ""
            };
            format!("{}{}:{}", opt, name, type_str)
        }
        SymbolKind::Class => {
            let extern_kw = if sym.flags.contains(SymbolFlags::EXTERN) {
                "extern "
            } else {
                ""
            };
            format!("{}{}class {}", vis, extern_kw, name)
        }
        SymbolKind::Interface => format!("{}interface {}", vis, name),
        SymbolKind::Enum => format!("{}enum {}", vis, name),
        SymbolKind::Abstract => {
            let ea = if sym.flags.contains(SymbolFlags::ABSTRACT) {
                "enum "
            } else {
                ""
            };
            format!("{}{}abstract {}", vis, ea, name)
        }
        SymbolKind::TypeAlias => {
            let target = type_table
                .get(sym.type_id)
                .and_then(|ty| match ty.kind {
                    TypeKind::TypeAlias { target_type, .. } => Some(target_type),
                    _ => None,
                })
                .unwrap_or(sym.type_id);
            let type_str = format_type_name(target, type_table, interner, symbol_table);
            format!("typedef {} = {}", name, type_str)
        }
        SymbolKind::EnumVariant => {
            let type_str = format_type_name(sym.type_id, type_table, interner, symbol_table);
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
            let type_str = format_type_name(sym.type_id, type_table, interner, symbol_table);
            format!("{}:{}", name, type_str)
        }
    }
}

// ---------------------------------------------------------------------------
// Type formatting
// ---------------------------------------------------------------------------

/// Format function type as `(param:Type, ...):ReturnType`.
fn format_function_type(
    type_id: TypeId,
    type_table: &TypeTable,
    interner: &StringInterner,
    symbol_table: &SymbolTable,
) -> String {
    if let Some(ti) = type_table.get(type_id)
        && let TypeKind::Function {
            params,
            return_type,
            ..
        } = &ti.kind
    {
        let param_strs: Vec<String> = params
            .iter()
            .map(|p| format_type_name(*p, type_table, interner, symbol_table))
            .collect();
        let ret = format_type_name(*return_type, type_table, interner, symbol_table);
        return format!("({}):{}", param_strs.join(", "), ret);
    }
    "(…)".to_string()
}

/// Format a type using declaration names from the analysis that owns its IDs.
pub fn format_type_name(
    type_id: TypeId,
    type_table: &TypeTable,
    interner: &StringInterner,
    symbol_table: &SymbolTable,
) -> String {
    format_type_at(type_id, type_table, interner, symbol_table, 0)
}

fn format_type_at(
    type_id: TypeId,
    type_table: &TypeTable,
    interner: &StringInterner,
    symbols: &SymbolTable,
    depth: usize,
) -> String {
    if depth >= 64 {
        return "…".into();
    }
    let Some(ty) = type_table.get(type_id) else {
        return "Unknown".into();
    };
    let nested = |id| format_type_at(id, type_table, interner, symbols, depth + 1);
    let named = |symbol, args: &[TypeId]| {
        let name = symbols
            .display_type_path(symbol, interner)
            .unwrap_or_else(|| "_".into());
        if args.is_empty() {
            name
        } else {
            format!(
                "{}<{}>",
                name,
                args.iter()
                    .map(|&id| nested(id))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
    };
    match &ty.kind {
        TypeKind::Void => "Void".into(),
        TypeKind::Bool => "Bool".into(),
        TypeKind::Int => "Int".into(),
        TypeKind::Float => "Float".into(),
        TypeKind::String => "String".into(),
        TypeKind::Char => "Char".into(),
        TypeKind::Dynamic => "Dynamic".into(),
        TypeKind::Unknown => "Unknown".into(),
        TypeKind::Error => "<error>".into(),
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
        }
        | TypeKind::TypeAlias {
            symbol_id,
            type_args,
            ..
        } => named(*symbol_id, type_args),
        TypeKind::TypeParameter { symbol_id, .. } => symbols
            .get_symbol(*symbol_id)
            .and_then(|s| interner.get(s.name))
            .unwrap_or("T")
            .into(),
        TypeKind::Placeholder { name } | TypeKind::ConstArgument { value: name } => {
            interner.get(*name).unwrap_or("_").into()
        }
        TypeKind::Array { element_type } => format!("Array<{}>", nested(*element_type)),
        TypeKind::Map {
            key_type,
            value_type,
        } => format!("Map<{}, {}>", nested(*key_type), nested(*value_type)),
        TypeKind::Optional { inner_type } => format!("Null<{}>", nested(*inner_type)),
        TypeKind::Function {
            params,
            return_type,
            ..
        } => format!(
            "({}) -> {}",
            params
                .iter()
                .map(|&p| nested(p))
                .collect::<Vec<_>>()
                .join(", "),
            nested(*return_type)
        ),
        TypeKind::GenericInstance {
            base_type,
            type_args,
            ..
        } => {
            let base = nested(*base_type);
            if type_args.is_empty() {
                base
            } else {
                format!(
                    "{}<{}>",
                    base,
                    type_args
                        .iter()
                        .map(|&a| nested(a))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
        }
        TypeKind::Anonymous { fields } => format!(
            "{{ {} }}",
            fields
                .iter()
                .map(|f| format!(
                    "{}:{}",
                    interner.get(f.name).unwrap_or("_"),
                    nested(f.type_id)
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        TypeKind::Union { types } => types
            .iter()
            .map(|&t| nested(t))
            .collect::<Vec<_>>()
            .join(" | "),
        TypeKind::Intersection { types } => types
            .iter()
            .map(|&t| nested(t))
            .collect::<Vec<_>>()
            .join(" & "),
        TypeKind::Reference { target_type, .. } => nested(*target_type),
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
        let detail = format_type_name(sym.type_id, &tt, interner, symbol_table);
        let doc = sym
            .documentation
            .and_then(|d| interner.get(d))
            .map(|s| s.to_string());

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
        SymbolKind::Variable | SymbolKind::Parameter => 0, // Locals first
        SymbolKind::Field | SymbolKind::Property => 1,     // Members
        SymbolKind::Function => 2,                         // Methods
        SymbolKind::EnumVariant => 3,                      // Enum values
        SymbolKind::Class | SymbolKind::Interface | SymbolKind::Enum => 4, // Types
        SymbolKind::TypeAlias | SymbolKind::Abstract => 5, // Type aliases
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
    pub line: u32,       // 0-based
    pub start_char: u32, // 0-based
    pub length: u32,
    pub token_type: u32,      // index into SEMANTIC_TOKEN_TYPES
    pub token_modifiers: u32, // bitmask into SEMANTIC_TOKEN_MODIFIERS
}

/// Semantic tokens for every declared or resolved name in the file.
pub fn build_semantic_tokens(
    symbol_table: &SymbolTable,
    index: &FileSymbolIndex,
) -> Vec<SemanticToken> {
    let mut tokens = Vec::new();

    for (&(line, column), &(name_len, sym_id)) in &index.names {
        let Some(sym) = symbol_table.get_symbol(sym_id) else {
            continue;
        };
        let loc = sym.definition_location;
        let declaration = loc.file_id == index.file_id && (loc.line, loc.column) == (line, column);

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
        if declaration {
            modifiers |= 1 << 0;
        }
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
            line: line.saturating_sub(1), // LSP is 0-based
            start_char: column.saturating_sub(1),
            length: name_len,
            token_type,
            token_modifiers: modifiers,
        });
    }

    // `names` is ordered by position, as LSP delta encoding requires.
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

    // Collect top-level declarations (classes, enums, functions, typedefs)
    // TODO: nest methods/fields under their parent class using scope_tree
    let mut top_level = Vec::new();

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
        let detail = format_type_name(sym.type_id, &tt, interner, symbol_table);
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
            SymbolKind::Class
            | SymbolKind::Interface
            | SymbolKind::Enum
            | SymbolKind::Abstract
            | SymbolKind::Module => {
                top_level.push((sym.id, entry));
            }
            SymbolKind::Function
            | SymbolKind::Field
            | SymbolKind::Property
            | SymbolKind::EnumVariant
            | SymbolKind::Variable => {
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
    let (params, ret) = if let TypeKind::Function {
        params,
        return_type,
        ..
    } = &ti.kind
    {
        (params.clone(), *return_type)
    } else {
        return None;
    };

    let name = interner.get(sym.name)?;
    let param_strs: Vec<String> = params
        .iter()
        .map(|p| format_type_name(*p, &tt, interner, symbol_table))
        .collect();
    let ret_str = format_type_name(ret, &tt, interner, symbol_table);

    let label = format!("{}({}):{}", name, param_strs.join(", "), ret_str);

    let parameters: Vec<ParameterInfo> = param_strs
        .into_iter()
        .map(|p| ParameterInfo {
            label: p,
            documentation: None,
        })
        .collect();

    Some(SignatureInfo {
        label,
        documentation: sym
            .documentation
            .and_then(|d| interner.get(d))
            .map(|s| s.to_string()),
        parameters,
        active_parameter: active_param,
    })
}

// ---------------------------------------------------------------------------
// Member completion and calls under the cursor
// ---------------------------------------------------------------------------

fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// The byte offset of (line, col), both 1-based, in `text`.
fn offset_of(text: &str, line: u32, col: u32) -> Option<usize> {
    let start: usize = text
        .split_inclusive('\n')
        .take(line.checked_sub(1)? as usize)
        .map(str::len)
        .sum();
    let at = start + col.checked_sub(1)? as usize;
    (at <= text.len()).then_some(at)
}

/// The (line, col), both 1-based, of byte offset `at` in `text`.
fn position_of(text: &str, at: usize) -> (u32, u32) {
    let before = &text.as_bytes()[..at];
    let line = before.iter().filter(|&&b| b == b'\n').count() as u32 + 1;
    let line_start = before
        .iter()
        .rposition(|&b| b == b'\n')
        .map_or(0, |p| p + 1);
    (line, (at - line_start) as u32 + 1)
}

/// When the cursor at (line, col) follows `receiver.` and part of a member
/// name, where `receiver` starts and the name itself.
pub fn receiver_before(text: &str, line: u32, col: u32) -> Option<(u32, u32, String)> {
    let bytes = text.as_bytes();
    let mut i = offset_of(text, line, col)?;
    while i > 0 && is_ident_byte(bytes[i - 1]) {
        i -= 1;
    }
    while i > 0 && bytes[i - 1].is_ascii_whitespace() {
        i -= 1;
    }
    if i == 0 || bytes[i - 1] != b'.' {
        return None;
    }
    i -= 1;
    while i > 0 && bytes[i - 1].is_ascii_whitespace() {
        i -= 1;
    }
    let end = i;
    while i > 0 && is_ident_byte(bytes[i - 1]) {
        i -= 1;
    }
    let name = &text[i..end];
    (i < end && name != "this").then(|| {
        let (line, col) = position_of(text, i);
        (line, col, name.to_string())
    })
}

/// The symbol `name` most likely means at (line, col): the closest earlier
/// declaration of it in the file, else a type of that name, else any member
/// of that name. For a name the last analysis did not record where it is
/// written, as in text being typed.
pub fn resolve_name(
    name: &str,
    file_id: u32,
    line: u32,
    col: u32,
    symbol_table: &SymbolTable,
    interner: &StringInterner,
) -> Option<SymbolId> {
    let named = symbol_table.find_symbols(|s| interner.get(s.name) == Some(name));
    let earlier = named
        .iter()
        .filter(|s| {
            let at = s.definition_location;
            at.file_id == file_id && (at.line, at.column) < (line, col)
        })
        .max_by_key(|s| (s.definition_location.line, s.definition_location.column));
    let is_type = |s: &&&Symbol| {
        matches!(
            s.kind,
            SymbolKind::Class | SymbolKind::Interface | SymbolKind::Abstract | SymbolKind::Enum
        )
    };
    earlier
        .or_else(|| named.iter().find(is_type))
        .or_else(|| named.first())
        .map(|s| s.id)
}

/// The class a value of type `type_id` is an instance of.
pub(crate) fn class_of_type(
    type_id: TypeId,
    symbol_table: &SymbolTable,
    type_table: &TypeTable,
    interner: &StringInterner,
) -> Option<SymbolId> {
    let named = |name: &str| {
        symbol_table
            .find_symbols(|s| {
                s.kind == SymbolKind::Class
                    && interner.get(s.name) == Some(name)
                    && s.qualified_name.and_then(|q| interner.get(q)) == Some(name)
            })
            .first()
            .map(|s| s.id)
    };
    match &type_table.get(type_id)?.kind {
        TypeKind::Class { symbol_id, .. }
        | TypeKind::Interface { symbol_id, .. }
        | TypeKind::Enum { symbol_id, .. }
        | TypeKind::Abstract { symbol_id, .. } => Some(*symbol_id),
        TypeKind::GenericInstance { base_type, .. } => {
            class_of_type(*base_type, symbol_table, type_table, interner)
        }
        TypeKind::Optional { inner_type, .. } => {
            class_of_type(*inner_type, symbol_table, type_table, interner)
        }
        TypeKind::TypeAlias { target_type, .. } | TypeKind::Reference { target_type, .. } => {
            class_of_type(*target_type, symbol_table, type_table, interner)
        }
        TypeKind::String => named("String"),
        TypeKind::Array { .. } => named("Array"),
        _ => None,
    }
}

/// What can follow `receiver.`: a type's statics and constructors, or an
/// instance's fields and methods, its superclasses' included.
pub fn stdlib_member_completions(
    receiver: SymbolId,
    unit: &compiler::compilation::CompilationUnit,
) -> Vec<CompletionEntry> {
    let Some(symbol) = unit.symbol_table.get_symbol(receiver) else {
        return Vec::new();
    };
    let is_type = symbol.kind == SymbolKind::Class;
    let owner = if is_type {
        Some(symbol.id)
    } else {
        class_of_type(
            symbol.type_id,
            &unit.symbol_table,
            &unit.type_table.borrow(),
            &unit.string_interner,
        )
    };
    let Some(owner) = owner.and_then(|id| unit.symbol_table.get_symbol(id)) else {
        return Vec::new();
    };
    let name = unit
        .string_interner
        .get(owner.qualified_name.unwrap_or(owner.name));
    for file in &unit.stdlib_files {
        let comments = file
            .input
            .as_deref()
            .and_then(crate::documentation::SourceComments::new);
        let package = file
            .package
            .as_ref()
            .map(|p| p.path.join("."))
            .unwrap_or_default();
        for declaration in &file.declarations {
            let parser::TypeDeclaration::Class(class) = declaration else {
                continue;
            };
            let qualified = if package.is_empty() {
                class.name.clone()
            } else {
                format!("{package}.{}", class.name)
            };
            if name != Some(qualified.as_str()) {
                continue;
            }
            return class
                .fields
                .iter()
                .filter(|f| {
                    f.access != Some(parser::Access::Private)
                        && f.modifiers.contains(&parser::Modifier::Static) == is_type
                })
                .filter_map(|field| {
                    let (name, kind, expr) = match &field.kind {
                        parser::ClassFieldKind::Function(f) => {
                            let mut f = f.clone();
                            f.body = None;
                            (
                                f.name.clone(),
                                CompletionKind::Function,
                                parser::ExprKind::Function(f),
                            )
                        }
                        parser::ClassFieldKind::Var {
                            name, type_hint, ..
                        }
                        | parser::ClassFieldKind::Final {
                            name, type_hint, ..
                        }
                        | parser::ClassFieldKind::Property {
                            name, type_hint, ..
                        } => (
                            name.clone(),
                            CompletionKind::Field,
                            parser::ExprKind::Var {
                                name: name.clone(),
                                type_hint: type_hint.clone(),
                                expr: None,
                                attrs: Default::default(),
                            },
                        ),
                    };
                    if name == "new" {
                        return None;
                    }
                    let detail = compiler::macro_system::printer::Printer::new().print_expr(
                        &compiler::macro_system::MacroValue::Expr(std::sync::Arc::new(
                            parser::Expr {
                                kind: expr,
                                span: field.span,
                            },
                        )),
                    );
                    Some(CompletionEntry {
                        label: name,
                        kind,
                        detail,
                        documentation: comments.as_ref().and_then(|c| c.at(field.span.start)),
                        deprecated: false,
                        sort_priority: 0,
                    })
                })
                .collect();
        }
    }
    Vec::new()
}

/// Members supplied by typed declarations and inherited classes.
pub fn member_completions(
    receiver: SymbolId,
    symbol_table: &SymbolTable,
    type_table: &Rc<RefCell<TypeTable>>,
    interner: &StringInterner,
    class_fields: impl Fn(SymbolId) -> Vec<(SymbolId, bool)>,
) -> Vec<CompletionEntry> {
    let tt = type_table.borrow();
    let Some(sym) = symbol_table.get_symbol(receiver) else {
        return Vec::new();
    };
    let is_type = matches!(
        sym.kind,
        SymbolKind::Class | SymbolKind::Interface | SymbolKind::Abstract | SymbolKind::Enum
    );
    let mut class = if is_type {
        Some(sym.id)
    } else {
        class_of_type(sym.type_id, symbol_table, &tt, interner)
    };
    let mut entries: Vec<CompletionEntry> = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    while let Some(class_id) = class {
        let Some(prefix) = symbol_table
            .get_symbol(class_id)
            .and_then(|c| c.qualified_name.or(Some(c.name)))
            .and_then(|q| interner.get(q))
            .map(|q| format!("{q}."))
        else {
            break;
        };
        let fields = class_fields(class_id);
        let declared = symbol_table.all_symbols().filter(|m| {
            m.qualified_name
                .and_then(|q| interner.get(q))
                .and_then(|q| q.strip_prefix(prefix.as_str()))
                .is_some_and(|rest| Some(rest) == interner.get(m.name))
        });
        // A field's entry says whether it is static; a declared member's
        // flags do.
        let members = fields
            .iter()
            .filter_map(|&(f, is_static)| Some((symbol_table.get_symbol(f)?, Some(is_static))))
            .chain(declared.map(|m| (m, None)));
        for (member, is_static) in members {
            let wanted = match (member.kind, is_static) {
                (_, Some(is_static)) => is_static == is_type,
                (SymbolKind::Field | SymbolKind::Property | SymbolKind::Function, None) => {
                    member.flags.contains(SymbolFlags::STATIC) == is_type
                }
                (SymbolKind::EnumVariant, None) => is_type,
                _ => false,
            };
            let Some(name) = interner.get(member.name) else {
                continue;
            };
            if !wanted || name.starts_with("__") || name == "new" || !seen.insert(name.to_string())
            {
                continue;
            }
            entries.push(CompletionEntry {
                label: name.to_string(),
                kind: symbol_kind_to_completion(member.kind),
                detail: format_type_name(member.type_id, &tt, interner, symbol_table),
                documentation: member
                    .documentation
                    .and_then(|d| interner.get(d))
                    .map(str::to_string),
                deprecated: member.flags.contains(SymbolFlags::DEPRECATED),
                sort_priority: completion_sort_priority(member),
            });
        }
        // An instance has its superclasses' members; Haxe does not inherit
        // statics, so `Sub.baseStatic` is not offered.
        class = (!is_type)
            .then(|| symbol_table.get_class_super_type(class_id))
            .flatten()
            .and_then(|t| class_of_type(t, symbol_table, &tt, interner));
    }
    entries
}

/// A call whose argument list the cursor is in.
pub struct CallAtCursor {
    /// Where the callee's name starts, 1-based.
    pub callee_line: u32,
    pub callee_col: u32,
    pub callee: String,
    /// Which argument, from 0.
    pub argument: u32,
}

/// The innermost call whose `(` is open at (line, col), 1-based, skipping
/// strings and comments.
pub fn enclosing_call(text: &str, line: u32, col: u32) -> Option<CallAtCursor> {
    let end = offset_of(text, line, col)?;
    let bytes = text.as_bytes();
    // (bracket, offset, commas seen at its depth)
    let mut open: Vec<(u8, usize, u32)> = Vec::new();
    let mut i = 0;
    while i < end {
        match bytes[i] {
            b'/' if bytes.get(i + 1) == Some(&b'/') => {
                while i < end && bytes[i] != b'\n' {
                    i += 1;
                }
                continue;
            }
            b'/' if bytes.get(i + 1) == Some(&b'*') => {
                i += 2;
                while i + 1 < end && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                    i += 1;
                }
                i += 2;
                continue;
            }
            q @ (b'"' | b'\'') => {
                i += 1;
                while i < end && bytes[i] != q {
                    i += if bytes[i] == b'\\' { 2 } else { 1 };
                }
            }
            b @ (b'(' | b'[' | b'{') => open.push((b, i, 0)),
            b')' | b']' | b'}' => {
                open.pop();
            }
            b',' => {
                if let Some(top) = open.last_mut() {
                    top.2 += 1;
                }
            }
            _ => {}
        }
        i += 1;
    }
    let &(bracket, paren, argument) = open.last()?;
    if bracket != b'(' {
        return None;
    }
    let mut j = paren;
    while j > 0 && bytes[j - 1].is_ascii_whitespace() {
        j -= 1;
    }
    let name_end = j;
    while j > 0 && is_ident_byte(bytes[j - 1]) {
        j -= 1;
    }
    let name = &text[j..name_end];
    const NOT_CALLS: &[&str] = &[
        "if", "while", "for", "switch", "catch", "function", "return",
    ];
    if name.is_empty() || NOT_CALLS.contains(&name) {
        return None;
    }
    let (callee_line, callee_col) = position_of(text, j);
    Some(CallAtCursor {
        callee_line,
        callee_col,
        callee: name.to_string(),
        argument,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unresolved_function_hover_keeps_the_name_and_explains_missing_signature() {
        let interner = StringInterner::new();
        let mut symbols = SymbolTable::new();
        let method = symbols.create_function(interner.intern("getUI16"));
        let types = Rc::new(RefCell::new(TypeTable::new()));
        for ty in [TypeId::invalid(), types.borrow().unknown_type()] {
            symbols.get_symbol_mut(method).unwrap().type_id = ty;
            let hover = format_hover(method, &symbols, &types, &interner).unwrap();
            assert!(hover.contains("public function getUI16(…)"), "{hover}");
            assert!(hover.contains("Signature not yet resolved"), "{hover}");
            assert!(!hover.contains("getUI16Unknown"), "{hover}");
        }
    }

    #[test]
    fn finds_the_receiver_before_a_dot() {
        let text = "var a = 1;\nfoo.ba";
        assert_eq!(receiver_before(text, 2, 7), Some((2, 1, "foo".into())));
        assert_eq!(receiver_before("foo . ", 1, 7), Some((1, 1, "foo".into())));
        assert_eq!(receiver_before("foo", 1, 4), None);
        assert_eq!(receiver_before("this.x", 1, 7), None);
    }

    #[test]
    fn finds_the_open_call_and_argument() {
        let text = "x(1, f(\"a,(\", [2, 3]), ";
        let call = enclosing_call(text, 1, text.len() as u32 + 1).unwrap();
        assert_eq!(
            (call.callee_line, call.callee_col, call.argument),
            (1, 1, 2)
        );
        let call = enclosing_call(text, 1, 9).unwrap();
        assert_eq!((call.callee_col, call.argument), (6, 0));
        assert!(enclosing_call("if (a, ", 1, 8).is_none());
        assert!(enclosing_call("f(a) ", 1, 6).is_none());
    }
}
