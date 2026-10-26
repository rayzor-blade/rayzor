//! Class Registry for Macro Interpreter
//!
//! Scans parsed HaxeFiles for class declarations and stores their methods/constructors
//! as AST bodies. The macro interpreter consults this registry as a fallback when
//! hardcoded class dispatch (Std, Math, etc.) doesn't match.

use parser::{
    Access, ClassFieldKind, Expr, FunctionParam, HaxeFile, Metadata, Modifier, TypeDeclaration,
};
use std::collections::BTreeMap;
use std::sync::Arc;

/// Info about a single method (static or instance)
pub struct MethodInfo {
    pub name: String,
    pub params: Vec<FunctionParam>,
    pub body: Arc<Expr>,
    pub is_static: bool,
    pub return_type: Option<parser::Type>,
    pub owner: String,
    pub location: crate::tast::SourceLocation,
}

/// Info about a single field variable
pub struct FieldVarInfo {
    pub name: String,
    pub init_expr: Option<Arc<Expr>>,
    pub is_static: bool,
}

/// Info about a registered class
pub struct ClassInfo {
    /// Path of the defining file.
    pub source_file: String,
    pub name: String,
    pub qualified_name: String,
    pub constructor: Option<MethodInfo>,
    pub static_methods: BTreeMap<String, MethodInfo>,
    pub instance_methods: BTreeMap<String, MethodInfo>,
    pub instance_vars: Vec<FieldVarInfo>,
    pub static_vars: Vec<FieldVarInfo>,
    pub imports: Arc<BTreeMap<String, String>>,
    pub parent: Option<String>,
    pub abstract_decl: Option<Arc<parser::AbstractDecl>>,
    pub field_types: BTreeMap<String, parser::Type>,
}

pub(crate) struct GenericBuildInfo {
    pub class: parser::ClassDecl,
    pub pack: Vec<String>,
    pub module: String,
    pub imports: BTreeMap<String, String>,
}

struct TypeDefinition {
    qualified_name: String,
    source_file: String,
    runtime_class: bool,
    parents: Vec<String>,
    module: String,
    metadata: Vec<Metadata>,
    fields: Vec<FieldDefinition>,
    is_abstract: bool,
    is_enum_abstract: bool,
    constructors: BTreeMap<String, (usize, usize)>,
}

struct FieldDefinition {
    name: String,
    span: parser::Span,
    metadata: Vec<Metadata>,
    is_static: bool,
    is_enum_value: bool,
    init_expr: Option<Arc<Expr>>,
}

/// Registry of all known classes for macro interpretation.
///
/// Built from parsed HaxeFiles (stdlib + imports + user files) before macro expansion.
/// Macro-time static values, `Class.field` → value, shared by every
/// registry of one compile.
pub type MacroStatics = Arc<std::sync::Mutex<BTreeMap<String, super::value::MacroValue>>>;

/// The interpreter falls back to this registry when hardcoded class dispatch doesn't match.
pub struct ClassRegistry {
    /// qualified_name → ClassInfo
    classes: BTreeMap<String, ClassInfo>,
    pub(super) using_members: BTreeMap<String, Vec<String>>,
    pub(super) aliases: BTreeMap<String, (parser::Type, Arc<BTreeMap<String, String>>)>,
    /// short_name → qualified_name (for unambiguous lookups)
    short_name_index: BTreeMap<String, String>,
    type_definitions: BTreeMap<String, Arc<TypeDefinition>>,
    reflected_definitions: std::sync::Mutex<BTreeMap<String, Option<Arc<TypeDefinition>>>>,
    enum_constants: BTreeMap<String, std::collections::BTreeSet<String>>,
    enum_constructors: BTreeMap<String, BTreeMap<String, usize>>,
    generic_builds: BTreeMap<String, Arc<GenericBuildInfo>>,
    /// Static variables' current values, `Class.field` → value. Shared by
    /// every macro call of the compile, as macro-time statics are.
    statics: MacroStatics,
    pub(super) dependencies: super::dependencies::Dependencies,
    class_paths: Vec<std::path::PathBuf>,
    defines: BTreeMap<String, String>,
    source_overrides: Arc<BTreeMap<std::path::PathBuf, String>>,
    reflection_config: Option<(
        parser::preprocessor::PreprocessorConfig,
        parser::preprocessor::PreprocessorConfig,
        Vec<std::path::PathBuf>,
    )>,
}

impl ClassRegistry {
    pub fn new() -> Self {
        Self {
            classes: BTreeMap::new(),
            aliases: BTreeMap::new(),
            using_members: BTreeMap::new(),
            short_name_index: BTreeMap::new(),
            type_definitions: BTreeMap::new(),
            reflected_definitions: Default::default(),
            enum_constants: BTreeMap::new(),
            enum_constructors: BTreeMap::new(),
            generic_builds: BTreeMap::new(),
            statics: MacroStatics::default(),
            dependencies: Default::default(),
            class_paths: Vec::new(),
            defines: BTreeMap::new(),
            source_overrides: Default::default(),
            reflection_config: None,
        }
    }

    /// Read and write macro-time statics in `statics`, the compile's shared
    /// store, instead of a store of this registry's own.
    pub fn use_statics(&mut self, statics: MacroStatics) {
        self.statics = statics;
    }

    pub fn use_dependencies(&mut self, dependencies: super::dependencies::Dependencies) {
        self.dependencies = dependencies;
    }

    pub fn set_class_paths(&mut self, paths: Vec<std::path::PathBuf>) {
        self.class_paths = paths;
    }

    pub fn set_defines(&mut self, defines: BTreeMap<String, String>) {
        self.defines = defines;
    }

    pub fn set_source_overrides(&mut self, sources: Arc<BTreeMap<std::path::PathBuf, String>>) {
        self.source_overrides = sources;
    }

    pub(crate) fn set_reflection_config(
        &mut self,
        project: parser::preprocessor::PreprocessorConfig,
        stdlib: parser::preprocessor::PreprocessorConfig,
        paths: Vec<std::path::PathBuf>,
    ) {
        self.reflection_config = Some((
            project,
            stdlib,
            paths
                .into_iter()
                .map(|path| path.canonicalize().unwrap_or(path))
                .collect(),
        ));
    }

    pub(crate) fn macro_context(&self) -> super::context_api::MacroContext {
        let mut context = super::context_api::MacroContext::new();
        context.class_paths = self.class_paths.clone();
        context.defines = self.defines.clone();
        context.dependencies = self.dependencies.clone();
        context
    }

    /// The static variable `name` declared by `class_name`, with the class's
    /// qualified name.
    pub fn find_static_var(&self, class_name: &str, name: &str) -> Option<(String, Arc<Expr>)> {
        let enum_value = self
            .type_definitions
            .get(class_name)
            .and_then(|definition| {
                definition
                    .fields
                    .iter()
                    .find(|field| field.is_enum_value && field.name == name)
                    .map(|field| (definition.qualified_name.clone(), field.init_expr.clone()))
            });
        let (qualified, init) = if let Some(value) = enum_value {
            value
        } else if let Some(class) = self.find_class(class_name) {
            let var = class.static_vars.iter().find(|v| v.name == name)?;
            (class.qualified_name.clone(), var.init_expr.clone())
        } else {
            let definition = self.type_definitions.get(class_name)?;
            let field = definition
                .fields
                .iter()
                .find(|field| field.is_static && field.name == name)?;
            (definition.qualified_name.clone(), field.init_expr.clone())
        };
        let init = init.unwrap_or_else(|| {
            Arc::new(Expr {
                kind: parser::ExprKind::Null,
                span: parser::Span::default(),
            })
        });
        Some((qualified, init))
    }

    /// Every macro-time static's current value.
    pub fn statics_snapshot(&self) -> BTreeMap<String, super::value::MacroValue> {
        self.statics.lock().map(|s| s.clone()).unwrap_or_default()
    }

    /// Put every macro-time static back to `snapshot`.
    pub fn restore_statics(&self, snapshot: BTreeMap<String, super::value::MacroValue>) {
        if let Ok(mut statics) = self.statics.lock() {
            *statics = snapshot;
        }
    }

    pub fn static_value(&self, key: &str) -> Option<super::value::MacroValue> {
        self.statics.lock().ok()?.get(key).cloned()
    }

    pub fn set_static(&self, key: String, value: super::value::MacroValue) {
        if let Ok(mut statics) = self.statics.lock() {
            statics.insert(key, value);
        }
    }

    pub(crate) fn replace_static_references(
        &self,
        old: &super::value::MacroValue,
        new: &super::value::MacroValue,
        updates: &mut super::value::ReferenceUpdates,
    ) {
        if let Ok(mut statics) = self.statics.lock() {
            for value in statics.values_mut() {
                value.replace_reference(old, new, updates);
            }
        }
    }

    /// Register all classes from a single HaxeFile.
    pub fn register_file(&mut self, file: &HaxeFile) {
        let mut file_imports = super::interpreter::build_file_import_map(file);
        let package_prefix = match &file.package {
            Some(pkg) if !pkg.path.is_empty() => format!("{}.", pkg.path.join(".")),
            _ => String::new(),
        };

        let module_name = std::path::Path::new(&file.filename)
            .file_stem()
            .and_then(|name| name.to_str())
            .unwrap_or("");
        for declaration in &file.declarations {
            let name = match declaration {
                TypeDeclaration::Class(c) => &c.name,
                TypeDeclaration::Interface(c) => &c.name,
                TypeDeclaration::Abstract(c) => &c.name,
                TypeDeclaration::Enum(c) => &c.name,
                TypeDeclaration::Typedef(c) => &c.name,
                _ => continue,
            };
            file_imports.insert(name.clone(), format!("{package_prefix}{name}"));
        }
        let members = file
            .declarations
            .iter()
            .filter_map(|declaration| match declaration {
                TypeDeclaration::Class(class) => Some(format!("{package_prefix}{}", class.name)),
                TypeDeclaration::Abstract(class) => Some(format!("{package_prefix}{}", class.name)),
                TypeDeclaration::Typedef(alias) => match &alias.type_def {
                    parser::Type::Path { path, .. } => Some(
                        path.package
                            .iter()
                            .chain(std::iter::once(&path.name))
                            .chain(path.sub.iter())
                            .cloned()
                            .collect::<Vec<_>>()
                            .join("."),
                    ),
                    _ => None,
                },
                _ => None,
            })
            .collect();
        self.using_members
            .insert(format!("{package_prefix}{module_name}"), members);
        let file_imports = Arc::new(file_imports);
        for decl in &file.declarations {
            let (name, access, metadata, fields, is_abstract, is_enum_abstract) = match decl {
                TypeDeclaration::Class(c) => (
                    &c.name,
                    c.access,
                    &c.meta,
                    c.fields.as_slice(),
                    false,
                    false,
                ),
                TypeDeclaration::Interface(c) => (
                    &c.name,
                    c.access,
                    &c.meta,
                    c.fields.as_slice(),
                    false,
                    false,
                ),
                TypeDeclaration::Abstract(a) => (
                    &a.name,
                    a.access,
                    &a.meta,
                    a.fields.as_slice(),
                    true,
                    a.is_enum_abstract,
                ),
                TypeDeclaration::Typedef(t) => (&t.name, t.access, &t.meta, &[][..], false, false),
                TypeDeclaration::Enum(e) => (&e.name, e.access, &e.meta, &[][..], false, false),
                _ => continue,
            };
            let underlying = match decl {
                TypeDeclaration::Abstract(a) => match &a.underlying {
                    Some(parser::Type::Path { path, .. }) => Some(path.name.as_str()),
                    _ => None,
                },
                _ => None,
            };
            let mut next_value: i64 = 0;
            let definition = Arc::new(TypeDefinition {
                source_file: file.filename.clone(),
                qualified_name: format!("{package_prefix}{name}"),
                runtime_class: matches!(
                    decl,
                    TypeDeclaration::Class(_) | TypeDeclaration::Interface(_)
                ),
                parents: {
                    let types: Vec<&parser::Type> = match decl {
                        TypeDeclaration::Class(c) => {
                            c.extends.iter().chain(&c.implements).collect()
                        }
                        TypeDeclaration::Interface(c) => c.extends.iter().collect(),
                        _ => Vec::new(),
                    };
                    types
                        .into_iter()
                        .filter_map(|t| {
                            let parser::Type::Path { path, .. } = t else {
                                return None;
                            };
                            let mut parts = path.package.clone();
                            parts.push(path.name.clone());
                            parts.extend(path.sub.clone());
                            let name = parts.join(".");
                            Some(if path.package.is_empty() {
                                file_imports
                                    .get(&name)
                                    .cloned()
                                    .unwrap_or_else(|| format!("{package_prefix}{name}"))
                            } else {
                                name
                            })
                        })
                        .collect()
                },
                module: format!("{package_prefix}{module_name}"),
                metadata: metadata.clone(),
                is_abstract,
                is_enum_abstract,
                constructors: match decl {
                    TypeDeclaration::Enum(declaration) => declaration
                        .constructors
                        .iter()
                        .map(|constructor| {
                            (
                                constructor.name.clone(),
                                (
                                    constructor.params.len(),
                                    constructor
                                        .params
                                        .iter()
                                        .filter(|param| {
                                            !param.optional && param.default_value.is_none()
                                        })
                                        .count(),
                                ),
                            )
                        })
                        .collect(),
                    _ => BTreeMap::new(),
                },
                fields: fields
                    .iter()
                    .map(|field| {
                        let (name, is_enum_value) = match &field.kind {
                            ClassFieldKind::Function(f) => (&f.name, false),
                            ClassFieldKind::Var { name, .. }
                            | ClassFieldKind::Final { name, .. } => (
                                name,
                                is_enum_abstract && !field.modifiers.contains(&Modifier::Static),
                            ),
                            ClassFieldKind::Property { name, .. } => (name, false),
                        };
                        let mut init_expr = match &field.kind {
                            ClassFieldKind::Var { expr, .. }
                            | ClassFieldKind::Final { expr, .. }
                            | ClassFieldKind::Property { expr, .. } => expr.clone(),
                            _ => None,
                        };
                        if is_enum_value {
                            if let Some(Expr {
                                kind: parser::ExprKind::Int(value),
                                ..
                            }) = &init_expr
                            {
                                next_value = value.saturating_add(1);
                            } else if init_expr.is_none() {
                                let kind = match underlying {
                                    Some("Int") => Some(parser::ExprKind::Int(next_value)),
                                    Some("Float") => {
                                        Some(parser::ExprKind::Float(next_value as f64))
                                    }
                                    Some("String") => Some(parser::ExprKind::String(name.clone())),
                                    _ => None,
                                };
                                init_expr = kind.map(|kind| Expr {
                                    kind,
                                    span: field.span,
                                });
                                next_value = next_value.saturating_add(1);
                            }
                        }
                        FieldDefinition {
                            name: name.clone(),
                            span: field.span,
                            metadata: field.meta.clone(),
                            is_static: field.modifiers.contains(&Modifier::Static) || is_enum_value,
                            is_enum_value,
                            init_expr: init_expr.map(Arc::new),
                        }
                    })
                    .collect(),
            });
            if let Some(previous) = self.type_definitions.get(&definition.qualified_name) {
                for field in previous.fields.iter().filter(|field| field.is_enum_value) {
                    if let Some(owners) = self.enum_constants.get_mut(&field.name) {
                        owners.remove(&previous.qualified_name);
                    }
                }
                for name in previous.constructors.keys() {
                    if let Some(owners) = self.enum_constructors.get_mut(name) {
                        owners.remove(&previous.qualified_name);
                    }
                }
            }
            for field in definition.fields.iter().filter(|field| field.is_enum_value) {
                self.enum_constants
                    .entry(field.name.clone())
                    .or_default()
                    .insert(definition.qualified_name.clone());
            }
            for (name, &(arity, _)) in &definition.constructors {
                self.enum_constructors
                    .entry(name.clone())
                    .or_default()
                    .insert(definition.qualified_name.clone(), arity);
            }
            self.type_definitions
                .insert(format!("{package_prefix}{name}"), definition.clone());
            let module_path = if access == Some(Access::Private) {
                format!("{package_prefix}_{module_name}.{name}")
            } else {
                format!("{package_prefix}{module_name}.{name}")
            };
            self.type_definitions.insert(module_path, definition);
        }

        for decl in &file.declarations {
            if let TypeDeclaration::Typedef(alias) = decl {
                self.aliases.insert(
                    format!("{package_prefix}{}", alias.name),
                    (alias.type_def.clone(), file_imports.clone()),
                );
            }
            let (class, abstract_decl) = match decl {
                TypeDeclaration::Class(class) => (std::borrow::Cow::Borrowed(class), None),
                TypeDeclaration::Abstract(decl) => (
                    std::borrow::Cow::Owned(parser::ClassDecl {
                        name: decl.name.clone(),
                        fields: decl.fields.clone(),
                        meta: decl.meta.clone(),
                        access: decl.access,
                        modifiers: decl.modifiers.clone(),
                        type_params: decl.type_params.clone(),
                        extends: None,
                        implements: vec![],
                        span: decl.span,
                        is_abstract: false,
                    }),
                    Some(Arc::new(decl.clone())),
                ),
                _ => continue,
            };
            {
                let qualified_name = format!("{}{}", package_prefix, class.name);
                if class
                    .meta
                    .iter()
                    .any(|m| m.name.trim_start_matches(':') == "genericBuild")
                {
                    self.generic_builds.insert(
                        qualified_name.clone(),
                        Arc::new(GenericBuildInfo {
                            class: class.as_ref().clone(),
                            pack: file
                                .package
                                .as_ref()
                                .map(|p| p.path.clone())
                                .unwrap_or_default(),
                            module: module_name.to_string(),
                            imports: super::interpreter::build_file_import_map(file),
                        }),
                    );
                }
                let mut info = ClassInfo {
                    source_file: file.filename.clone(),
                    name: class.name.clone(),
                    qualified_name: qualified_name.clone(),
                    constructor: None,
                    static_methods: BTreeMap::new(),
                    instance_methods: BTreeMap::new(),
                    instance_vars: Vec::new(),
                    static_vars: Vec::new(),
                    imports: file_imports.clone(),
                    parent: class.extends.as_ref().and_then(|ty| match ty {
                        parser::Type::Path { path, .. } => Some(if path.package.is_empty() {
                            file_imports
                                .get(&path.name)
                                .cloned()
                                .unwrap_or_else(|| format!("{package_prefix}{}", path.name))
                        } else {
                            path.package
                                .iter()
                                .chain(std::iter::once(&path.name))
                                .cloned()
                                .collect::<Vec<_>>()
                                .join(".")
                        }),
                        _ => None,
                    }),
                    abstract_decl,
                    field_types: class
                        .fields
                        .iter()
                        .filter_map(|field| match &field.kind {
                            ClassFieldKind::Var {
                                name,
                                type_hint: Some(ty),
                                ..
                            }
                            | ClassFieldKind::Final {
                                name,
                                type_hint: Some(ty),
                                ..
                            }
                            | ClassFieldKind::Property {
                                name,
                                type_hint: Some(ty),
                                ..
                            } => Some((name.clone(), ty.clone())),
                            _ => None,
                        })
                        .collect(),
                };

                for field in &class.fields {
                    let is_static = field.modifiers.contains(&Modifier::Static);
                    let is_macro = field.modifiers.contains(&Modifier::Macro);

                    match &field.kind {
                        ClassFieldKind::Function(func) => {
                            // Skip macro functions — they're handled by MacroRegistry
                            if is_macro {
                                continue;
                            }

                            let body = match &func.body {
                                Some(body) => Arc::new((**body).clone()),
                                None => continue, // No body = extern/abstract, skip
                            };

                            let method = MethodInfo {
                                name: func.name.clone(),
                                params: func.params.clone(),
                                body,
                                is_static,
                                return_type: func.return_type.clone(),
                                owner: qualified_name.clone(),
                                location: super::errors::location_in_source(
                                    file.input.as_deref(),
                                    func.span,
                                ),
                            };

                            if func.name == "new" {
                                info.constructor = Some(method);
                            } else if is_static {
                                info.static_methods.insert(func.name.clone(), method);
                            } else {
                                info.instance_methods.insert(func.name.clone(), method);
                            }
                        }
                        ClassFieldKind::Var { name, expr, .. }
                        | ClassFieldKind::Property { name, expr, .. } => {
                            let init = expr.as_ref().map(|e| Arc::new(e.clone()));
                            let var_info = FieldVarInfo {
                                name: name.clone(),
                                init_expr: init,
                                is_static,
                            };
                            if is_static {
                                info.static_vars.push(var_info);
                            } else {
                                info.instance_vars.push(var_info);
                            }
                        }
                        ClassFieldKind::Final { name, expr, .. } => {
                            let init = expr.as_ref().map(|e| Arc::new(e.clone()));
                            let var_info = FieldVarInfo {
                                name: name.clone(),
                                init_expr: init,
                                is_static,
                            };
                            if is_static {
                                info.static_vars.push(var_info);
                            } else {
                                info.instance_vars.push(var_info);
                            }
                        }
                    }
                }

                // Update short name index (only if unambiguous)
                if !self.short_name_index.contains_key(&class.name) {
                    self.short_name_index
                        .insert(class.name.clone(), qualified_name.clone());
                }

                self.classes.insert(qualified_name, info);
            }
        }
    }

    pub(crate) fn generic_build(&self, name: &str) -> Option<Arc<GenericBuildInfo>> {
        let class = self.find_class(name)?;
        self.generic_builds.get(&class.qualified_name).cloned()
    }

    pub(crate) fn has_generic_builds(&self) -> bool {
        !self.generic_builds.is_empty()
    }

    pub fn enrich_type_view(&self, view: super::value::MacroValue) -> super::value::MacroValue {
        self.enrich_typed_view(view, None)
    }

    /// Runtime imports retain source declarations only when reflection requests them.
    fn reflected_definition(&self, name: &str, module: &str) -> Option<Arc<TypeDefinition>> {
        if let Some(definition) = self.type_definitions.get(name) {
            return Some(definition.clone());
        }
        if let Some(definition) = self.reflected_definitions.lock().unwrap().get(name) {
            return definition.clone();
        }
        let relative = format!("{}.hx", module.replace('.', "/"));
        let source = self.class_paths.iter().find_map(|root| {
            let path = root.join(&relative);
            let path = path.canonicalize().unwrap_or(path);
            let source = self
                .source_overrides
                .get(&path)
                .cloned()
                .or_else(|| std::fs::read_to_string(&path).ok())?;
            self.dependencies.register_source(&path);
            Some((path, source))
        });
        let definition = source.and_then(|(path, source)| {
            let mut config = self
                .reflection_config
                .as_ref()
                .map(|(project, stdlib, paths)| {
                    if paths.iter().any(|root| path.starts_with(root)) {
                        stdlib.clone()
                    } else {
                        project.clone()
                    }
                })
                .unwrap_or_else(|| parser::preprocessor::PreprocessorConfig {
                    defines: self
                        .defines
                        .keys()
                        .filter(|name| name.as_str() != "macro")
                        .cloned()
                        .collect(),
                    values: self.defines.clone(),
                });
            config.defines.remove("macro");
            config.values.remove("macro");
            let file = parser::haxe_parser::parse_haxe_file_with_config(
                &path.to_string_lossy(),
                &source,
                true,
                true,
                &config,
            )
            .ok()?;
            let mut declarations = ClassRegistry::new();
            declarations.register_file(&file);
            let definition = declarations.type_definitions.get(name).cloned();
            self.reflected_definitions.lock().unwrap().extend(
                declarations
                    .type_definitions
                    .into_iter()
                    .map(|(name, definition)| (name, Some(definition))),
            );
            definition
        });
        self.reflected_definitions
            .lock()
            .unwrap()
            .insert(name.to_string(), definition.clone());
        definition
    }

    pub fn enrich_typed_view(
        &self,
        view: super::value::MacroValue,
        declared_type: Option<crate::tast::TypeId>,
    ) -> super::value::MacroValue {
        use super::value::MacroValue as V;
        let V::Object(mut view) = view else {
            return view;
        };
        let Some(name) = view.get("name").and_then(V::as_string) else {
            return V::Object(view);
        };
        let mut path = match view.get("pack") {
            Some(V::Array(pack)) => pack.iter().map(V::to_display_string).collect::<Vec<_>>(),
            _ => Vec::new(),
        };
        path.push(name.to_string());
        let module = view.get("module").and_then(V::as_string).unwrap_or("");
        let Some(definition) = self.reflected_definition(&path.join("."), module) else {
            return V::Object(view);
        };
        let view = Arc::make_mut(&mut view);
        view.insert(
            "module".to_string(),
            V::String(Arc::from(definition.module.as_str())),
        );
        view.insert(
            "pack".to_string(),
            V::Array(Arc::new(
                definition
                    .module
                    .rsplit_once('.')
                    .map_or("", |(pack, _)| pack)
                    .split('.')
                    .filter(|part| !part.is_empty())
                    .map(V::from_str)
                    .collect(),
            )),
        );
        let mut metadata = metadata_entries(&definition.metadata);
        if let Some(V::Object(meta)) = view.get("meta")
            && let Some(V::Array(entries)) = meta.get("__meta__")
        {
            for entry in entries.iter() {
                let name = match entry {
                    V::Object(entry) => entry.get("name"),
                    _ => None,
                };
                if !metadata.iter().any(|candidate| matches!(candidate, V::Object(candidate) if candidate.get("name") == name)) {
                    metadata.push(entry.clone());
                }
            }
        }
        if definition.is_enum_abstract {
            metadata.push(metadata_entry(":enum"));
        }
        view.insert("meta".to_string(), metadata_access(metadata));
        let mut implementation_fields = Vec::new();
        for (key, is_static) in [("fields", false), ("statics", true)] {
            let typed_fields = match view.get(key) {
                Some(V::Object(reference)) => match reference.get("__ref__") {
                    Some(V::Array(fields)) => fields.as_ref().clone(),
                    _ => Vec::new(),
                },
                _ => Vec::new(),
            };
            let mut fields = Vec::new();
            for field in definition
                .fields
                .iter()
                .filter(|f| f.is_static == is_static)
            {
                let mut entry = typed_fields
                    .iter()
                    .find_map(|v| match v {
                        V::Object(o)
                            if o.get("name").and_then(V::as_string) == Some(&field.name) =>
                        {
                            Some(o.as_ref().clone())
                        }
                        _ => None,
                    })
                    .unwrap_or_default();
                entry.insert(
                    "name".to_string(),
                    V::String(Arc::from(field.name.as_str())),
                );
                entry.insert(
                    "pos".to_string(),
                    V::Position(super::context_api::source_position(
                        &definition.source_file,
                        field.span,
                    )),
                );
                entry.insert(
                    "__field_expr__".to_string(),
                    field
                        .init_expr
                        .as_ref()
                        .map(|expr| V::Expr(expr.clone()))
                        .unwrap_or(V::Null),
                );
                if field.is_enum_value {
                    if let Some(ty) = declared_type {
                        entry.insert("type".to_string(), V::Type(ty));
                    }
                    entry.insert("__enum_value__".to_string(), V::Bool(true));
                }
                let mut metadata = metadata_entries(&field.metadata);
                if definition.is_abstract {
                    metadata.push(metadata_entry(":impl"));
                }
                if field.is_enum_value {
                    metadata.push(metadata_entry(":enum"));
                }
                entry.insert("meta".to_string(), metadata_access(metadata));
                let entry = V::Object(Arc::new(entry));
                if definition.is_abstract {
                    implementation_fields.push(entry.clone());
                }
                fields.push(entry);
            }
            for entry in typed_fields {
                let name = match &entry {
                    V::Object(entry) => entry.get("name").and_then(V::as_string),
                    _ => None,
                };
                if !definition
                    .fields
                    .iter()
                    .any(|f| f.is_static == is_static && Some(f.name.as_str()) == name)
                {
                    if definition.is_abstract {
                        implementation_fields.push(entry.clone());
                    }
                    fields.push(entry);
                }
            }
            view.insert(key.to_string(), reference(V::Array(Arc::new(fields))));
        }
        if definition.is_abstract {
            let mut implementation = view.clone();
            implementation.insert(
                "statics".to_string(),
                reference(V::Array(Arc::new(implementation_fields))),
            );
            implementation.insert(
                "fields".to_string(),
                reference(V::Array(Arc::new(Vec::new()))),
            );
            view.insert(
                "impl".to_string(),
                reference(V::Object(Arc::new(implementation))),
            );
        }
        V::Object(Arc::new(view.clone()))
    }

    /// Register all classes from multiple HaxeFiles.
    pub fn register_files(&mut self, files: &[HaxeFile]) {
        for file in files {
            self.register_file(file);
        }
    }

    pub(crate) fn enum_constant_owner(&self, name: &str) -> Option<String> {
        let owners = self.enum_constants.get(name)?;
        (owners.len() == 1)
            .then(|| owners.first().cloned())
            .flatten()
    }

    pub(crate) fn enum_constructor_required(&self, owner: &str, name: &str) -> Option<usize> {
        self.type_definitions
            .get(owner)?
            .constructors
            .get(name)
            .map(|(_, required)| *required)
    }

    pub(crate) fn enum_constructor(
        &self,
        owner: Option<&str>,
        name: &str,
    ) -> Option<(String, usize)> {
        if let Some(owner) = owner {
            let definition = self.type_definitions.get(owner)?;
            return Some((
                definition.qualified_name.clone(),
                definition.constructors.get(name)?.0,
            ));
        }
        let constructors = self.enum_constructors.get(name)?;
        (constructors.len() == 1)
            .then(|| {
                constructors
                    .first_key_value()
                    .map(|(name, &arity)| (name.clone(), arity))
            })
            .flatten()
    }

    pub(crate) fn enum_constructor_in_scope(
        &self,
        scope: &str,
        name: &str,
    ) -> Option<(String, usize)> {
        if let Some(found) = self.enum_constructor(Some(scope), name) {
            return Some(found);
        }
        let mut found = self
            .enum_constructors
            .get(name)?
            .iter()
            .filter(|(owner, _)| {
                owner
                    .rsplit_once('.')
                    .is_some_and(|(package, _)| package == scope)
                    || self
                        .type_definitions
                        .get(*owner)
                        .is_some_and(|definition| definition.module == scope)
            });
        let (owner, &arity) = found.next()?;
        found.next().is_none().then(|| (owner.clone(), arity))
    }

    /// Find a class by name (tries exact match, then short name index).
    pub fn find_class(&self, name: &str) -> Option<&ClassInfo> {
        if let Some(info) = self.classes.get(name) {
            return Some(info);
        }
        if let Some(qualified) = self.short_name_index.get(name) {
            return self.classes.get(qualified);
        }
        let definition = self.type_definitions.get(name)?;
        let short_name = name.rsplit('.').next()?;
        let package = definition.module.rsplit_once('.').map(|(pack, _)| pack);
        let qualified = package
            .map(|pack| format!("{pack}.{short_name}"))
            .unwrap_or_else(|| short_name.to_string());
        self.classes.get(&qualified)
    }

    pub fn runtime_class_name(&self, name: &str) -> Option<&str> {
        if let Some(info) = self.find_class(name) {
            return Some(&info.qualified_name);
        }
        if let Some(info) = self.type_definitions.get(name).filter(|d| d.runtime_class) {
            return Some(&info.qualified_name);
        }
        let mut names = self
            .type_definitions
            .values()
            .filter(|d| d.runtime_class && d.qualified_name.rsplit('.').next() == Some(name))
            .map(|d| d.qualified_name.as_str());
        let first = names.next()?;
        names.all(|other| other == first).then_some(first)
    }

    pub fn is_subclass(&self, actual: &str, expected: &str) -> bool {
        let mut pending = vec![actual.to_string()];
        let mut seen = std::collections::BTreeSet::new();
        while let Some(name) = pending.pop() {
            if !seen.insert(name.clone()) {
                continue;
            }
            if name == expected {
                return true;
            }
            if let Some(definition) = self.type_definitions.get(&name) {
                if definition.qualified_name == expected {
                    return true;
                }
                pending.extend(definition.parents.iter().cloned());
            }
        }
        false
    }

    /// `Context.getLocalClass().get()` for a class known only by its source:
    /// path, metadata and fields, each method's `expr()` a `TFunction` whose
    /// arguments carry their metadata.
    pub fn local_class_view(&self, qualified_name: &str) -> Option<super::value::MacroValue> {
        use super::value::MacroValue as V;
        let info = self.classes.get(qualified_name)?;
        let pack: Vec<V> = qualified_name
            .rsplit_once('.')
            .map(|(pack, _)| pack.split('.').map(|p| V::String(Arc::from(p))).collect())
            .unwrap_or_default();
        let base = BTreeMap::from([
            ("name".to_string(), V::String(Arc::from(info.name.as_str()))),
            ("pack".to_string(), V::Array(Arc::new(pack))),
        ]);
        let V::Object(mut view) = self.enrich_type_view(V::Object(Arc::new(base))) else {
            return None;
        };
        let view_mut = Arc::make_mut(&mut view);
        for (key, methods) in [
            ("statics", &info.static_methods),
            ("fields", &info.instance_methods),
        ] {
            let Some(V::Object(reference)) = view_mut.get_mut(key) else {
                continue;
            };
            let Some(V::Array(fields)) = Arc::make_mut(reference).get_mut("__ref__") else {
                continue;
            };
            for field in Arc::make_mut(fields).iter_mut() {
                let V::Object(field) = field else { continue };
                let name = field.get("name").map(V::to_display_string);
                if let Some(method) = name.and_then(|n| methods.get(&n)) {
                    Arc::make_mut(field).insert("__expr__".to_string(), typed_function(method));
                }
            }
        }
        Some(V::Object(view))
    }

    /// Find a static method on a class.
    pub fn find_static_method(&self, class_name: &str, method: &str) -> Option<&MethodInfo> {
        self.find_class(class_name)
            .and_then(|c| c.static_methods.get(method))
    }

    /// Find a constructor on a class.
    pub fn find_constructor(&self, class_name: &str) -> Option<&MethodInfo> {
        self.find_class(class_name)
            .and_then(|c| c.constructor.as_ref())
    }

    /// Find an instance method on a class.
    pub(super) fn parent_class(&self, class: &ClassInfo) -> Option<&ClassInfo> {
        let parent = class.parent.as_deref()?;
        if let Some(parent) = self.find_class(parent) {
            return Some(parent);
        }
        let short = parent.rsplit('.').next()?;
        if let Some((mut package, _)) = class.qualified_name.rsplit_once('.') {
            while let Some((enclosing, _)) = package.rsplit_once('.') {
                if let Some(parent) = self.find_class(&format!("{enclosing}.{short}")) {
                    return Some(parent);
                }
                package = enclosing;
            }
        }
        for imported in class
            .imports
            .get(super::interpreter::WILDCARD_OWNERS_KEY)
            .into_iter()
            .flat_map(|paths| paths.split('\n'))
        {
            if let Some(parent) = self.find_class(&format!("{imported}.{short}")) {
                return Some(parent);
            }
        }
        None
    }

    pub fn find_instance_method(&self, class_name: &str, method: &str) -> Option<&MethodInfo> {
        let mut class = self.find_class(class_name)?;
        for _ in 0..128 {
            if let Some(found) = class.instance_methods.get(method) {
                return Some(found);
            }
            class = self.parent_class(class)?;
        }
        None
    }

    /// Get the number of registered classes.
    pub fn class_count(&self) -> usize {
        self.classes.len()
    }

    /// Iterate over all registered class names.
    pub fn iter_class_names(&self) -> impl Iterator<Item = &str> {
        self.classes.keys().map(|s| s.as_str())
    }
}

fn reference(value: super::value::MacroValue) -> super::value::MacroValue {
    super::value::MacroValue::Object(Arc::new(BTreeMap::from([("__ref__".to_string(), value)])))
}

fn metadata_entry(name: &str) -> super::value::MacroValue {
    use super::value::MacroValue as V;
    V::Object(Arc::new(BTreeMap::from([
        ("name".to_string(), V::String(Arc::from(name))),
        ("params".to_string(), V::Array(Arc::new(Vec::new()))),
    ])))
}

fn metadata_entries(metadata: &[Metadata]) -> Vec<super::value::MacroValue> {
    use super::value::MacroValue as V;
    metadata
        .iter()
        .map(|m| {
            let name = if m.compile_time {
                format!(":{}", m.name.trim_start_matches(':'))
            } else {
                m.name.clone()
            };
            V::Object(Arc::new(BTreeMap::from([
                ("name".to_string(), V::String(Arc::from(name))),
                (
                    "params".to_string(),
                    V::Array(Arc::new(
                        m.params
                            .iter()
                            .map(|p| V::Expr(Arc::new(p.clone())))
                            .collect(),
                    )),
                ),
                (
                    "pos".to_string(),
                    V::Position(super::ast_bridge::span_to_location(m.span)),
                ),
            ])))
        })
        .collect()
}

/// A method's `ClassField.expr()`: a `TypedExpr` over `TFunction`. Only the
/// arguments are filled, each a `TVar` with its name and metadata.
fn typed_function(method: &MethodInfo) -> super::value::MacroValue {
    use super::value::MacroValue as V;
    let object = |entries: Vec<(&str, V)>| {
        V::Object(Arc::new(
            entries
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect(),
        ))
    };
    let args = method
        .params
        .iter()
        .enumerate()
        .map(|(id, param)| {
            let var = object(vec![
                ("id", V::Int(id as i64)),
                ("name", V::String(Arc::from(param.name.as_str()))),
                ("meta", metadata_access(metadata_entries(&param.meta))),
                ("capture", V::Bool(false)),
                ("t", V::Null),
                ("extra", V::Null),
            ]);
            object(vec![("v", var), ("value", V::Null)])
        })
        .collect();
    let function = object(vec![
        ("args", V::Array(Arc::new(args))),
        ("t", V::Null),
        ("expr", V::Null),
    ]);
    object(vec![
        (
            "expr",
            V::Enum(
                Arc::from("TypedExprDef"),
                Arc::from("TFunction"),
                Arc::new(vec![function]),
            ),
        ),
        ("t", V::Null),
        ("pos", V::Null),
    ])
}

fn metadata_access(entries: Vec<super::value::MacroValue>) -> super::value::MacroValue {
    use super::value::MacroValue as V;
    V::Object(Arc::new(BTreeMap::from([(
        "__meta__".to_string(),
        V::Array(Arc::new(entries)),
    )])))
}
