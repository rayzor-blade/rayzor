//! Class Registry for Macro Interpreter
//!
//! Scans parsed HaxeFiles for class declarations and stores their methods/constructors
//! as AST bodies. The macro interpreter consults this registry as a fallback when
//! hardcoded class dispatch (Std, Math, etc.) doesn't match.

use parser::{ClassFieldKind, Expr, FunctionParam, HaxeFile, Modifier, TypeDeclaration};
use std::collections::HashMap;
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
    pub static_methods: HashMap<String, MethodInfo>,
    pub instance_methods: HashMap<String, MethodInfo>,
    pub instance_vars: Vec<FieldVarInfo>,
    pub static_vars: Vec<FieldVarInfo>,
}

/// Registry of all known classes for macro interpretation.
///
/// Built from parsed HaxeFiles (stdlib + imports + user files) before macro expansion.
/// The interpreter falls back to this registry when hardcoded class dispatch doesn't match.
pub struct ClassRegistry {
    /// qualified_name → ClassInfo
    classes: HashMap<String, ClassInfo>,
    /// short_name → qualified_name (for unambiguous lookups)
    short_name_index: HashMap<String, String>,
}

impl ClassRegistry {
    pub fn new() -> Self {
        Self {
            classes: HashMap::new(),
            short_name_index: HashMap::new(),
        }
    }

    /// Register all classes from a single HaxeFile.
    pub fn register_file(&mut self, file: &HaxeFile) {
        let package_prefix = match &file.package {
            Some(pkg) if !pkg.path.is_empty() => format!("{}.", pkg.path.join(".")),
            _ => String::new(),
        };

        for decl in &file.declarations {
            if let TypeDeclaration::Class(class) = decl {
                let qualified_name = format!("{}{}", package_prefix, class.name);
                let mut info = ClassInfo {
                    name: class.name.clone(),
                    qualified_name: qualified_name.clone(),
                    constructor: None,
                    static_methods: HashMap::new(),
                    instance_methods: HashMap::new(),
                    instance_vars: Vec::new(),
                    static_vars: Vec::new(),
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
        None
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
}
