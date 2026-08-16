//! The `.bsym` symbol manifest: every declaration in the standard library,
//! extracted from parsed source.
//!
//! This lives outside the compiler crate so the compiler's own build script can
//! produce the manifest at build time. Extraction needs a parser and nothing
//! else — Haxe declarations carry explicit type annotations, so the manifest is
//! derived from the AST without type checking.

use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

/// Errors that can occur during BLADE operations
#[derive(Debug)]
pub enum BladeError {
    /// I/O error
    Io(std::io::Error),

    /// Serialization error
    Serialization(postcard::Error),

    /// Invalid magic number
    InvalidMagic,

    /// Unsupported version
    UnsupportedVersion(u32),

    /// Compression/decompression error
    Compression(String),
}

impl std::fmt::Display for BladeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BladeError::Io(e) => write!(f, "I/O error: {}", e),
            BladeError::Serialization(e) => {
                let detail = e.to_string();
                write!(f, "Serialization error: {}", detail)?;
                if looks_like_stale_postcard_schema(&detail) {
                    write!(
                        f,
                        " (the .blade/.rzb file was probably built with an older compiler; clear .rayzor/cache and rebuild the bundle)"
                    )?;
                }
                Ok(())
            }
            BladeError::Compression(e) => write!(f, "Compression error: {}", e),
            BladeError::InvalidMagic => write!(f, "Invalid BLADE magic number"),
            BladeError::UnsupportedVersion(v) => write!(f, "Unsupported BLADE version: {}", v),
        }
    }
}

fn looks_like_stale_postcard_schema(detail: &str) -> bool {
    detail.contains("Found a bool that wasn't 0 or 1")
        || detail.contains("DeserializeBadBool")
        || detail.contains("DeserializeUnexpectedEnd")
        || detail.contains("SerdeDeCustom")
        || detail.contains("enum")
}

impl std::error::Error for BladeError {}

impl From<std::io::Error> for BladeError {
    fn from(e: std::io::Error) -> Self {
        BladeError::Io(e)
    }
}

impl From<postcard::Error> for BladeError {
    fn from(e: postcard::Error) -> Self {
        BladeError::Serialization(e)
    }
}

// ============================================================================
// BLADE Symbol Format - Pre-resolved symbol storage for fast startup
// ============================================================================

/// Magic number for symbol manifest files
const SYMBOL_MAGIC: &[u8; 4] = b"BSYM";

/// Current symbol format version
const SYMBOL_VERSION: u32 = 2;

/// A type as the manifest records it.
///
/// Types used to be written out as source text and parsed again on load. That
/// cost a parse for every signature in the library on every process start, and
/// the text could not express everything a type is: an anonymous structure, an
/// intersection and a nested function type all came back as a placeholder, and
/// an unannotated declaration was indistinguishable from one written
/// `Dynamic`. Recording the shape means loading resolves names instead of
/// re-deriving syntax.
///
/// Names are kept as written, package and all, rather than as ids: an id is
/// meaningful only in the context that minted it, so a manifest that stored
/// one could not be read by any other compilation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BladeType {
    /// A named type — `Int`, `haxe.io.Bytes`, `Array<String>`.
    Path {
        package: Vec<String>,
        name: String,
        /// The member of a module, when the type is not the module's own.
        sub: Option<String>,
        params: Vec<BladeType>,
    },
    /// `(A, B) -> C`
    Function {
        params: Vec<BladeType>,
        ret: Box<BladeType>,
    },
    /// `{ a: Int, b: String }`
    Anonymous { fields: Vec<BladeAnonField> },
    /// `Null<T>`
    Optional(Box<BladeType>),
    /// `A & B`
    Intersection {
        left: Box<BladeType>,
        right: Box<BladeType>,
    },
    /// `?` — a type the source left to inference.
    Wildcard,
}

impl BladeType {
    /// The type a declaration carries when it names one it does not have,
    /// keeping "unannotated" distinct from an explicit `Dynamic`.
    pub fn dynamic() -> Self {
        BladeType::Path {
            package: Vec::new(),
            name: "Dynamic".to_string(),
            sub: None,
            params: Vec::new(),
        }
    }
}

/// How a property is read or written.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BladeAccess {
    /// The field itself.
    Default,
    /// Readable or writable only from inside the class.
    Null,
    /// Not accessible.
    Never,
    /// Resolved at runtime.
    Dynamic,
    /// A named accessor method.
    Method(String),
}

/// The accessors of a property, as declared.
///
/// A property reads and writes through methods, so a consumer that restores it
/// as a plain field lowers `x.value` to a load from a slot that does not exist.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BladeProperty {
    pub getter: BladeAccess,
    pub setter: BladeAccess,
}

/// A field of an anonymous structure.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BladeAnonField {
    pub name: String,
    pub ty: BladeType,
}

/// Complete symbol information for a field
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BladeFieldInfo {
    /// Field name
    pub name: String,
    /// Declared type, or `None` where the source annotated none.
    pub field_type: Option<BladeType>,
    /// Is this field public?
    pub is_public: bool,
    /// Is this a static field?
    pub is_static: bool,
    /// Is this field final/immutable?
    pub is_final: bool,
    /// Does this field have a default value?
    pub has_default: bool,
    /// Accessors, where the declaration is a property rather than a plain
    /// field. `None` for an ordinary field.
    #[serde(default)]
    pub property: Option<BladeProperty>,
}

/// Parameter information for methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BladeParamInfo {
    /// Parameter name
    pub name: String,
    /// Declared type, or `None` where the source annotated none.
    pub param_type: Option<BladeType>,
    /// Does this parameter have a default value?
    pub has_default: bool,
    /// Is this parameter optional (nullable)?
    pub is_optional: bool,
}

/// Complete symbol information for a method
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BladeMethodInfo {
    /// Method name
    pub name: String,
    /// Method parameters
    pub params: Vec<BladeParamInfo>,
    /// Declared return type, or `None` where the source annotated none.
    pub return_type: Option<BladeType>,
    /// Is this method public?
    pub is_public: bool,
    /// Is this a static method?
    pub is_static: bool,
    /// Is this an inline method?
    pub is_inline: bool,
    /// Type parameters for generic methods
    pub type_params: Vec<String>,
    /// Native name from `@:native` metadata (e.g. `to_haxe_string`).
    /// Used by stdlib runtime mapping when the Haxe method name differs
    /// from the FFI symbol. `#[serde(default)]` so older cache files
    /// without this field deserialise as `None`.
    #[serde(default)]
    pub native_name: Option<String>,
}

/// Complete symbol information for a class
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BladeClassInfo {
    /// Class name (short name, e.g., "Bytes")
    pub name: String,
    /// Package path (e.g., ["haxe", "io"])
    pub package: Vec<String>,
    /// Superclass qualified name (if any)
    pub extends: Option<String>,
    /// Implemented interfaces
    pub implements: Vec<String>,
    /// Type parameters (e.g., ["T", "U"])
    pub type_params: Vec<String>,
    /// Is this an extern class?
    pub is_extern: bool,
    /// Is this an abstract class?
    pub is_abstract: bool,
    /// Is this a final class?
    pub is_final: bool,
    /// Instance fields
    pub fields: Vec<BladeFieldInfo>,
    /// Instance methods
    pub methods: Vec<BladeMethodInfo>,
    /// Static fields
    pub static_fields: Vec<BladeFieldInfo>,
    /// Static methods
    pub static_methods: Vec<BladeMethodInfo>,
    /// Constructor info (if any)
    pub constructor: Option<BladeMethodInfo>,
    /// Native name from @:native metadata (e.g., "rayzor::concurrent::Arc")
    /// Lowered form replaces :: with _ for symbol resolution
    pub native_name: Option<String>,
}

/// Enum variant information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BladeEnumVariantInfo {
    /// Variant name
    pub name: String,
    /// Constructor parameters (for variants with data)
    pub params: Vec<BladeParamInfo>,
    /// Ordinal index
    pub index: usize,
}

/// Complete symbol information for an enum
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BladeEnumInfo {
    /// Enum name
    pub name: String,
    /// Package path
    pub package: Vec<String>,
    /// Type parameters
    pub type_params: Vec<String>,
    /// Enum variants
    pub variants: Vec<BladeEnumVariantInfo>,
    /// Is this an extern enum?
    pub is_extern: bool,
}

/// Type alias information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BladeTypeAliasInfo {
    /// Alias name
    pub name: String,
    /// Package path
    pub package: Vec<String>,
    /// Type parameters
    pub type_params: Vec<String>,
    /// The type this alias names.
    pub target_type: BladeType,
}

/// Abstract type information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BladeAbstractInfo {
    /// Abstract name
    pub name: String,
    /// Package path
    pub package: Vec<String>,
    /// Type parameters
    pub type_params: Vec<String>,
    /// The representation this abstract wraps.
    pub underlying_type: BladeType,
    /// Forward fields (from @:forward)
    pub forward_fields: Vec<String>,
    /// From types (implicit conversions from)
    pub from_types: Vec<String>,
    /// To types (implicit conversions to)
    pub to_types: Vec<String>,
    /// Methods defined on the abstract
    pub methods: Vec<BladeMethodInfo>,
    /// Static methods defined on the abstract
    pub static_methods: Vec<BladeMethodInfo>,
    /// Native name from @:native metadata (e.g., "rayzor::Ptr")
    pub native_name: Option<String>,
}

/// All type information for a module
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BladeTypeInfo {
    /// Classes defined in this module
    pub classes: Vec<BladeClassInfo>,
    /// Enums defined in this module
    pub enums: Vec<BladeEnumInfo>,
    /// Type aliases defined in this module
    pub type_aliases: Vec<BladeTypeAliasInfo>,
    /// Abstract types defined in this module
    pub abstracts: Vec<BladeAbstractInfo>,
}

/// Symbol manifest for all pre-compiled stdlib symbols
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BladeSymbolManifest {
    /// Magic number for validation
    magic: [u8; 4],
    /// Format version
    version: u32,
    /// Compiler version
    pub compiler_version: String,
    /// Build timestamp
    pub build_timestamp: u64,
    /// All modules in this manifest
    pub modules: Vec<BladeModuleSymbols>,
}

/// Symbols for a single module
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BladeModuleSymbols {
    /// Module name (e.g., "haxe.io.Bytes")
    pub name: String,
    /// Source file path
    pub source_path: String,
    /// Source hash for cache validation
    pub source_hash: u64,
    /// Type information
    pub types: BladeTypeInfo,
    /// Dependencies (other modules this depends on)
    pub dependencies: Vec<String>,
}

/// Timestamp for build metadata. Deterministic by default (0) so identical
/// sources produce bit-identical artifacts — same-input bundles must hash
/// equal for cache validation and A/B integrity. Set `SOURCE_DATE_EPOCH`
/// (the reproducible-builds convention) to embed a specific epoch instead.
/// Nothing reads these fields programmatically; they are debug metadata.
pub fn build_metadata_timestamp() -> u64 {
    std::env::var("SOURCE_DATE_EPOCH")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0)
}

/// Save a symbol manifest to file
pub fn save_symbol_manifest(
    path: impl AsRef<Path>,
    modules: Vec<BladeModuleSymbols>,
) -> Result<(), BladeError> {
    let now = build_metadata_timestamp();

    let manifest = BladeSymbolManifest {
        magic: *SYMBOL_MAGIC,
        version: SYMBOL_VERSION,
        compiler_version: env!("CARGO_PKG_VERSION").to_string(),
        build_timestamp: now,
        modules,
    };

    let bytes = postcard::to_allocvec(&manifest)?;
    fs::write(path, bytes)?;

    Ok(())
}

/// Load a symbol manifest from file
pub fn load_symbol_manifest(path: impl AsRef<Path>) -> Result<BladeSymbolManifest, BladeError> {
    // mmap for zero-copy deserialization of symbol manifest
    let file = std::fs::File::open(path.as_ref())?;
    let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(BladeError::Io)?;
    load_symbol_manifest_from_bytes(&mmap)
}

/// Load a symbol manifest from in-memory bytes.
pub fn load_symbol_manifest_from_bytes(bytes: &[u8]) -> Result<BladeSymbolManifest, BladeError> {
    let manifest: BladeSymbolManifest = postcard::from_bytes(bytes)?;

    if &manifest.magic != SYMBOL_MAGIC {
        return Err(BladeError::InvalidMagic);
    }

    if manifest.version != SYMBOL_VERSION {
        return Err(BladeError::UnsupportedVersion(manifest.version));
    }

    Ok(manifest)
}

pub fn extract_type_info_from_ast(haxe_file: &parser::HaxeFile) -> BladeTypeInfo {
    let mut type_info = BladeTypeInfo::default();

    let package: Vec<String> = haxe_file
        .package
        .as_ref()
        .map(|p| p.path.clone())
        .unwrap_or_default();

    for decl in &haxe_file.declarations {
        match decl {
            parser::TypeDeclaration::Class(class) => {
                let is_extern = class.modifiers.contains(&parser::Modifier::Extern);
                let native_name = extract_native_meta(&class.meta);
                let extends = class.extends.as_ref().map(type_to_qualified_name);
                let implements: Vec<String> = class
                    .implements
                    .iter()
                    .map(type_to_qualified_name)
                    .collect();
                let type_params: Vec<String> =
                    class.type_params.iter().map(|tp| tp.name.clone()).collect();

                let mut fields: Vec<BladeFieldInfo> = Vec::new();
                let mut static_fields: Vec<BladeFieldInfo> = Vec::new();
                let mut methods: Vec<BladeMethodInfo> = Vec::new();
                let mut static_methods: Vec<BladeMethodInfo> = Vec::new();
                let mut constructor: Option<BladeMethodInfo> = None;

                for field in &class.fields {
                    let is_static = field.modifiers.contains(&parser::Modifier::Static);
                    let is_inline = field.modifiers.contains(&parser::Modifier::Inline);
                    let is_public = matches!(field.access, Some(parser::Access::Public));

                    match &field.kind {
                        parser::ClassFieldKind::Var {
                            name,
                            type_hint,
                            expr,
                        } => {
                            let field_info = BladeFieldInfo {
                                name: name.clone(),
                                field_type: type_hint.as_ref().map(type_to_blade),
                                is_public,
                                is_static,
                                is_final: false,
                                has_default: expr.is_some(),
                                property: None,
                            };
                            if is_static {
                                static_fields.push(field_info);
                            } else {
                                fields.push(field_info);
                            }
                        }
                        parser::ClassFieldKind::Final {
                            name,
                            type_hint,
                            expr,
                        } => {
                            let field_info = BladeFieldInfo {
                                name: name.clone(),
                                field_type: type_hint.as_ref().map(type_to_blade),
                                is_public,
                                is_static,
                                is_final: true,
                                has_default: expr.is_some(),
                                property: None,
                            };
                            if is_static {
                                static_fields.push(field_info);
                            } else {
                                fields.push(field_info);
                            }
                        }
                        parser::ClassFieldKind::Property {
                            name,
                            type_hint,
                            getter,
                            setter,
                            ..
                        } => {
                            let field_info = BladeFieldInfo {
                                name: name.clone(),
                                field_type: type_hint.as_ref().map(type_to_blade),
                                is_public,
                                is_static,
                                is_final: false,
                                has_default: false,
                                property: Some(BladeProperty {
                                    getter: access_to_blade(getter, name, true),
                                    setter: access_to_blade(setter, name, false),
                                }),
                            };
                            if is_static {
                                static_fields.push(field_info);
                            } else {
                                fields.push(field_info);
                            }
                        }
                        parser::ClassFieldKind::Function(func) => {
                            let method_info = extract_method_from_ast(
                                func,
                                &field.meta,
                                is_public,
                                is_static,
                                is_inline,
                            );
                            if func.name == "new" {
                                constructor = Some(method_info);
                            } else if is_static {
                                static_methods.push(method_info);
                            } else {
                                methods.push(method_info);
                            }
                        }
                    }
                }

                type_info.classes.push(BladeClassInfo {
                    name: class.name.clone(),
                    package: package.clone(),
                    extends,
                    implements,
                    type_params,
                    is_extern,
                    is_abstract: false,
                    is_final: class.modifiers.contains(&parser::Modifier::Final),
                    fields,
                    methods,
                    static_fields,
                    static_methods,
                    constructor,
                    native_name,
                });
            }
            parser::TypeDeclaration::Enum(enum_decl) => {
                let type_params: Vec<String> = enum_decl
                    .type_params
                    .iter()
                    .map(|tp| tp.name.clone())
                    .collect();

                let variants: Vec<BladeEnumVariantInfo> = enum_decl
                    .constructors
                    .iter()
                    .enumerate()
                    .map(|(idx, v)| {
                        let params: Vec<BladeParamInfo> = v
                            .params
                            .iter()
                            .map(|p| BladeParamInfo {
                                name: p.name.clone(),
                                param_type: p.type_hint.as_ref().map(type_to_blade),
                                has_default: p.default_value.is_some(),
                                is_optional: p.optional,
                            })
                            .collect();
                        BladeEnumVariantInfo {
                            name: v.name.clone(),
                            params,
                            index: idx,
                        }
                    })
                    .collect();

                type_info.enums.push(BladeEnumInfo {
                    name: enum_decl.name.clone(),
                    package: package.clone(),
                    type_params,
                    variants,
                    is_extern: false,
                });
            }
            parser::TypeDeclaration::Typedef(typedef) => {
                let type_params: Vec<String> = typedef
                    .type_params
                    .iter()
                    .map(|tp| tp.name.clone())
                    .collect();

                type_info.type_aliases.push(BladeTypeAliasInfo {
                    name: typedef.name.clone(),
                    package: package.clone(),
                    type_params,
                    target_type: type_to_blade(&typedef.type_def),
                });
            }
            parser::TypeDeclaration::Abstract(abstract_decl) => {
                let native_name = extract_native_meta(&abstract_decl.meta);
                let type_params: Vec<String> = abstract_decl
                    .type_params
                    .iter()
                    .map(|tp| tp.name.clone())
                    .collect();

                let underlying_type = abstract_decl
                    .underlying
                    .as_ref()
                    .map(type_to_blade)
                    .unwrap_or_else(BladeType::dynamic);

                let from_types: Vec<String> = abstract_decl
                    .from
                    .iter()
                    .map(type_to_qualified_name)
                    .collect();
                let to_types: Vec<String> = abstract_decl
                    .to
                    .iter()
                    .map(type_to_qualified_name)
                    .collect();

                let mut methods: Vec<BladeMethodInfo> = Vec::new();
                let mut static_methods: Vec<BladeMethodInfo> = Vec::new();

                for field in &abstract_decl.fields {
                    if let parser::ClassFieldKind::Function(func) = &field.kind {
                        let is_static = field.modifiers.contains(&parser::Modifier::Static);
                        let is_inline = field.modifiers.contains(&parser::Modifier::Inline);
                        let is_public = matches!(field.access, Some(parser::Access::Public));
                        let method_info = extract_method_from_ast(
                            func,
                            &field.meta,
                            is_public,
                            is_static,
                            is_inline,
                        );
                        if is_static {
                            static_methods.push(method_info);
                        } else {
                            methods.push(method_info);
                        }
                    }
                }

                type_info.abstracts.push(BladeAbstractInfo {
                    name: abstract_decl.name.clone(),
                    package: package.clone(),
                    type_params,
                    underlying_type,
                    forward_fields: vec![],
                    from_types,
                    to_types,
                    methods,
                    static_methods,
                    native_name,
                });
            }
            parser::TypeDeclaration::Interface(iface) => {
                let extends: Option<String> = iface.extends.first().map(type_to_qualified_name);
                let implements: Vec<String> = iface
                    .extends
                    .iter()
                    .skip(1)
                    .map(type_to_qualified_name)
                    .collect();
                let type_params: Vec<String> =
                    iface.type_params.iter().map(|tp| tp.name.clone()).collect();

                let mut methods: Vec<BladeMethodInfo> = Vec::new();
                for field in &iface.fields {
                    if let parser::ClassFieldKind::Function(func) = &field.kind {
                        let is_public = true;
                        let is_static = field.modifiers.contains(&parser::Modifier::Static);
                        let is_inline = field.modifiers.contains(&parser::Modifier::Inline);
                        let method_info = extract_method_from_ast(
                            func,
                            &field.meta,
                            is_public,
                            is_static,
                            is_inline,
                        );
                        methods.push(method_info);
                    }
                }

                type_info.classes.push(BladeClassInfo {
                    name: iface.name.clone(),
                    package: package.clone(),
                    extends,
                    implements,
                    type_params,
                    is_extern: false,
                    is_abstract: true,
                    is_final: false,
                    fields: vec![],
                    methods,
                    static_fields: vec![],
                    static_methods: vec![],
                    constructor: None,
                    native_name: None,
                });
            }
            parser::TypeDeclaration::Conditional(_) => {}
        }
    }

    type_info
}

fn discover_stdlib_files(stdlib_path: &Path) -> Vec<(String, PathBuf)> {
    let mut files = Vec::new();
    discover_files_recursive(stdlib_path, stdlib_path, &mut files);
    files.sort_by(|a, b| a.0.cmp(&b.0));
    files
}

fn discover_files_recursive(
    base_path: &Path,
    current_path: &Path,
    files: &mut Vec<(String, PathBuf)>,
) {
    let mut dir_paths: Vec<std::path::PathBuf> = match std::fs::read_dir(current_path) {
        Ok(e) => e.filter_map(|entry| entry.ok().map(|e| e.path())).collect(),
        Err(_) => return,
    };
    dir_paths.sort(); // Deterministic ordering

    for path in dir_paths {
        if path.is_dir() {
            let dir_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

            if dir_name.starts_with('.') || dir_name.starts_with('_') {
                continue;
            }
            let skip_dirs = [
                "cpp", "cs", "flash", "hl", "java", "js", "lua", "neko", "php", "python", "eval",
            ];
            if skip_dirs.contains(&dir_name) {
                continue;
            }

            discover_files_recursive(base_path, &path, files);
        } else if path.extension().map(|e| e == "hx").unwrap_or(false) {
            let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

            if file_name == "import.hx" {
                continue;
            }

            if let Ok(relative) = path.strip_prefix(base_path) {
                let module_name = relative
                    .to_string_lossy()
                    .replace(['/', '\\'], ".")
                    .replace(".hx", "");
                files.push((module_name, path.clone()));
            }
        }
    }
}

fn hash_string(s: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}

fn extract_native_meta(meta: &[parser::Metadata]) -> Option<String> {
    for m in meta {
        let name = m.name.strip_prefix(':').unwrap_or(&m.name);
        if name == "native" {
            if let Some(first_param) = m.params.first() {
                if let parser::ExprKind::String(native_name) = &first_param.kind {
                    return Some(native_name.clone());
                }
            }
        }
    }
    None
}

fn extract_method_from_ast(
    func: &parser::Function,
    field_meta: &[parser::Metadata],
    is_public: bool,
    is_static: bool,
    is_inline: bool,
) -> BladeMethodInfo {
    let params: Vec<BladeParamInfo> = func
        .params
        .iter()
        .map(|p| BladeParamInfo {
            name: p.name.clone(),
            param_type: p.type_hint.as_ref().map(type_to_blade),
            has_default: p.default_value.is_some(),
            is_optional: p.optional,
        })
        .collect();

    let type_params: Vec<String> = func.type_params.iter().map(|tp| tp.name.clone()).collect();

    BladeMethodInfo {
        name: func.name.clone(),
        params,
        return_type: func.return_type.as_ref().map(type_to_blade),
        is_public,
        is_static,
        is_inline,
        type_params,
        native_name: extract_native_meta(field_meta),
    }
}

/// The name a type refers to, for the places a manifest records a reference
/// rather than a type — a supertype, an implemented interface, a conversion
/// target. Type arguments are not part of the reference.
fn type_to_qualified_name(ty: &parser::Type) -> String {
    match ty {
        parser::Type::Path { path, .. } => {
            let mut name = if path.package.is_empty() {
                path.name.clone()
            } else {
                format!("{}.{}", path.package.join("."), path.name)
            };
            if let Some(sub) = &path.sub {
                name = format!("{}.{}", name, sub);
            }
            name
        }
        parser::Type::Parenthesis { inner, .. } => type_to_qualified_name(inner),
        other => format!("{:?}", std::mem::discriminant(other)),
    }
}

/// Record a declared accessor. A bare `get`/`set` names a method derived from
/// the field, which is the form the rest of the compiler resolves.
fn access_to_blade(
    access: &parser::PropertyAccess,
    field_name: &str,
    is_getter: bool,
) -> BladeAccess {
    match access {
        parser::PropertyAccess::Default => BladeAccess::Default,
        parser::PropertyAccess::Null => BladeAccess::Null,
        parser::PropertyAccess::Never => BladeAccess::Never,
        parser::PropertyAccess::Dynamic => BladeAccess::Dynamic,
        parser::PropertyAccess::Custom(method) => {
            let _ = is_getter;
            let name = if method == "get" || method == "set" {
                format!("{method}_{field_name}")
            } else {
                method.clone()
            };
            BladeAccess::Method(name)
        }
    }
}

/// Record a parsed type as the manifest stores it.
///
/// A direct transcription: no name is resolved and nothing is looked up, so
/// extraction still needs only a parser. `Parenthesis` carries no meaning of
/// its own and collapses to what it wraps.
fn type_to_blade(ty: &parser::Type) -> BladeType {
    match ty {
        parser::Type::Path { path, params, .. } => BladeType::Path {
            package: path.package.clone(),
            name: path.name.clone(),
            sub: path.sub.clone(),
            params: params.iter().map(type_to_blade).collect(),
        },
        parser::Type::Function { params, ret, .. } => BladeType::Function {
            params: params.iter().map(type_to_blade).collect(),
            ret: Box::new(type_to_blade(ret)),
        },
        parser::Type::Anonymous { fields, .. } => BladeType::Anonymous {
            fields: fields
                .iter()
                .map(|f| BladeAnonField {
                    name: f.name.clone(),
                    ty: type_to_blade(&f.type_hint),
                })
                .collect(),
        },
        parser::Type::Optional { inner, .. } => BladeType::Optional(Box::new(type_to_blade(inner))),
        parser::Type::Parenthesis { inner, .. } => type_to_blade(inner),
        parser::Type::Intersection { left, right, .. } => BladeType::Intersection {
            left: Box::new(type_to_blade(left)),
            right: Box::new(type_to_blade(right)),
        },
        parser::Type::Wildcard { .. } => BladeType::Wildcard,
    }
}
/// Extract every declaration in a standard library rooted at `stdlib_root`.
///
/// Modules are visited in sorted order so the manifest is byte-reproducible
/// from the same sources. A module that declares nothing is omitted. Paths are
/// recorded relative to the root, so the manifest does not depend on where the
/// tree happened to live when it was built.
pub fn build_manifest(stdlib_root: &Path) -> Vec<BladeModuleSymbols> {
    let mut modules = Vec::new();

    for (module_name, file_path) in discover_stdlib_files(stdlib_root) {
        let Ok(source) = fs::read_to_string(&file_path) else {
            continue;
        };
        let relative = file_path
            .strip_prefix(stdlib_root)
            .unwrap_or(&file_path)
            .to_string_lossy()
            .replace('\\', "/");
        let Ok(haxe_file) = parser::parse_haxe_file(&relative, &source, true) else {
            continue;
        };

        let types = extract_type_info_from_ast(&haxe_file);
        if types.classes.is_empty()
            && types.enums.is_empty()
            && types.type_aliases.is_empty()
            && types.abstracts.is_empty()
        {
            continue;
        }

        modules.push(BladeModuleSymbols {
            source_hash: hash_string(&relative),
            name: module_name,
            source_path: relative,
            types,
            dependencies: Vec::new(),
        });
    }

    modules
}
