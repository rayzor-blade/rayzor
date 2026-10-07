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
}

/// Info about a single field variable
pub struct FieldVarInfo {
    pub name: String,
    pub init_expr: Option<Arc<Expr>>,
    pub is_static: bool,
}

/// Info about a registered class
pub struct ClassInfo {
    pub name: String,
    pub qualified_name: String,
    pub constructor: Option<MethodInfo>,
    pub static_methods: BTreeMap<String, MethodInfo>,
    pub instance_methods: BTreeMap<String, MethodInfo>,
    pub instance_vars: Vec<FieldVarInfo>,
    pub static_vars: Vec<FieldVarInfo>,
    pub imports: Arc<BTreeMap<String, String>>,
}

pub(crate) struct GenericBuildInfo {
    pub class: parser::ClassDecl,
    pub pack: Vec<String>,
    pub module: String,
    pub imports: BTreeMap<String, String>,
}

struct TypeDefinition {
    module: String,
    metadata: Vec<Metadata>,
    fields: Vec<FieldDefinition>,
    is_abstract: bool,
    is_enum_abstract: bool,
}

struct FieldDefinition {
    name: String,
    metadata: Vec<Metadata>,
    is_static: bool,
    is_enum_value: bool,
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
    /// short_name → qualified_name (for unambiguous lookups)
    short_name_index: BTreeMap<String, String>,
    type_definitions: BTreeMap<String, Arc<TypeDefinition>>,
    generic_builds: BTreeMap<String, Arc<GenericBuildInfo>>,
    /// Static variables' current values, `Class.field` → value. Shared by
    /// every macro call of the compile, as macro-time statics are.
    statics: MacroStatics,
}

impl ClassRegistry {
    pub fn new() -> Self {
        Self {
            classes: BTreeMap::new(),
            short_name_index: BTreeMap::new(),
            type_definitions: BTreeMap::new(),
            generic_builds: BTreeMap::new(),
            statics: MacroStatics::default(),
        }
    }

    /// Read and write macro-time statics in `statics`, the compile's shared
    /// store, instead of a store of this registry's own.
    pub fn use_statics(&mut self, statics: MacroStatics) {
        self.statics = statics;
    }

    /// The static variable `name` declared by `class_name`, with the class's
    /// qualified name.
    pub fn find_static_var(&self, class_name: &str, name: &str) -> Option<(String, Arc<Expr>)> {
        let class = self.find_class(class_name)?;
        let var = class.static_vars.iter().find(|v| v.name == name)?;
        let init = var.init_expr.clone().unwrap_or_else(|| {
            Arc::new(Expr {
                kind: parser::ExprKind::Null,
                span: parser::Span::default(),
            })
        });
        Some((class.qualified_name.clone(), init))
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

    /// Register all classes from a single HaxeFile.
    pub fn register_file(&mut self, file: &HaxeFile) {
        let file_imports = Arc::new(super::interpreter::build_import_map(&file.imports));
        let package_prefix = match &file.package {
            Some(pkg) if !pkg.path.is_empty() => format!("{}.", pkg.path.join(".")),
            _ => String::new(),
        };

        let module_name = std::path::Path::new(&file.filename)
            .file_stem()
            .and_then(|name| name.to_str())
            .unwrap_or("");
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
            let definition = Arc::new(TypeDefinition {
                module: format!("{package_prefix}{module_name}"),
                metadata: metadata.clone(),
                is_abstract,
                is_enum_abstract,
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
                        FieldDefinition {
                            name: name.clone(),
                            metadata: field.meta.clone(),
                            is_static: field.modifiers.contains(&Modifier::Static) || is_enum_value,
                            is_enum_value,
                        }
                    })
                    .collect(),
            });
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
            if let TypeDeclaration::Class(class) = decl {
                let qualified_name = format!("{}{}", package_prefix, class.name);
                if class
                    .meta
                    .iter()
                    .any(|m| m.name.trim_start_matches(':') == "genericBuild")
                {
                    self.generic_builds.insert(
                        qualified_name.clone(),
                        Arc::new(GenericBuildInfo {
                            class: class.clone(),
                            pack: file
                                .package
                                .as_ref()
                                .map(|p| p.path.clone())
                                .unwrap_or_default(),
                            module: module_name.to_string(),
                            imports: super::interpreter::build_import_map(&file.imports),
                        }),
                    );
                }
                let mut info = ClassInfo {
                    name: class.name.clone(),
                    qualified_name: qualified_name.clone(),
                    constructor: None,
                    static_methods: BTreeMap::new(),
                    instance_methods: BTreeMap::new(),
                    instance_vars: Vec::new(),
                    static_vars: Vec::new(),
                    imports: file_imports.clone(),
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
                            };

                            if func.name == "new" {
                                info.constructor = Some(method);
                            } else if is_static {
                                info.static_methods.insert(func.name.clone(), method);
                            } else {
                                info.instance_methods.insert(func.name.clone(), method);
                            }
                        }
                        ClassFieldKind::Var { name, expr, .. } => {
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
                        ClassFieldKind::Property { .. } => {
                            // Properties are accessed via getter/setter methods, skip for now
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
        let Some(definition) = self.type_definitions.get(&path.join(".")) else {
            return V::Object(view);
        };
        let view = Arc::make_mut(&mut view);
        view.insert(
            "module".to_string(),
            V::String(Arc::from(definition.module.as_str())),
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
    pub fn find_instance_method(&self, class_name: &str, method: &str) -> Option<&MethodInfo> {
        self.find_class(class_name)
            .and_then(|c| c.instance_methods.get(method))
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

fn metadata_access(entries: Vec<super::value::MacroValue>) -> super::value::MacroValue {
    use super::value::MacroValue as V;
    V::Object(Arc::new(BTreeMap::from([(
        "__meta__".to_string(),
        V::Array(Arc::new(entries)),
    )])))
}
