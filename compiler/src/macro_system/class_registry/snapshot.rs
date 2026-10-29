//! Portable stdlib declarations shared by macro registries of one compilation.

use super::*;
use bsym::{BladeMethodInfo, BladeType};

fn source_type(ty: &BladeType) -> parser::Type {
    let span = parser::Span::default();
    match ty {
        BladeType::Path {
            package,
            name,
            sub,
            params,
        } => parser::Type::Path {
            path: parser::TypePath {
                package: package.clone(),
                name: name.clone(),
                sub: sub.clone(),
            },
            params: params.iter().map(source_type).collect(),
            span,
        },
        BladeType::Function { params, ret } => parser::Type::Function {
            params: params.iter().map(source_type).collect(),
            ret: Box::new(source_type(ret)),
            span,
        },
        BladeType::Anonymous { fields } => parser::Type::Anonymous {
            fields: fields
                .iter()
                .map(|field| parser::AnonField {
                    name: field.name.clone(),
                    optional: field.is_optional,
                    type_hint: source_type(&field.ty),
                    access: parser::AnonFieldAccess {
                        is_final: field.is_final,
                        getter: source_access(&field.getter),
                        setter: source_access(&field.setter),
                    },
                    span,
                })
                .collect(),
            span,
        },
        BladeType::Optional(inner) => parser::Type::Optional {
            inner: Box::new(source_type(inner)),
            span,
        },
        BladeType::Intersection { left, right } => parser::Type::Intersection {
            left: Box::new(source_type(left)),
            right: Box::new(source_type(right)),
            span,
        },
        BladeType::Wildcard => parser::Type::Wildcard { span },
    }
}

fn source_access(access: &bsym::BladeAccess) -> parser::PropertyAccess {
    match access {
        bsym::BladeAccess::Default => parser::PropertyAccess::Default,
        bsym::BladeAccess::Null => parser::PropertyAccess::Null,
        bsym::BladeAccess::Never => parser::PropertyAccess::Never,
        bsym::BladeAccess::Dynamic => parser::PropertyAccess::Dynamic,
        bsym::BladeAccess::Method(name) => parser::PropertyAccess::Custom(name.clone()),
    }
}

fn signature(method: &BladeMethodInfo) -> MethodSignature {
    let span = parser::Span::default();
    MethodSignature {
        function: parser::Function {
            name: method.name.clone(),
            type_params: method.type_params.iter().enumerate().map(|(i, name)| parser::TypeParam {
                name: name.clone(),
                constraints: method.type_param_constraints.get(i).into_iter().flatten().map(source_type).collect(),
                variance: parser::Variance::Invariant, meta: Vec::new(), default_type: None, span,
            }).collect(),
            params: method.params.iter().map(|param| {
                let ty = param.param_type.as_ref().map(source_type);
                parser::FunctionParam {
                    name: param.name.clone(),
                    rest: matches!(&ty, Some(parser::Type::Path { path, .. }) if path.name == "Rest" && path.package == ["haxe"]),
                    type_hint: ty, optional: param.is_optional,
                    default_value: param.has_default.then(|| Box::new(Expr { kind: parser::ExprKind::Null, span })),
                    meta: Vec::new(), span,
                }
            }).collect(),
            return_type: method.return_type.as_ref().map(source_type), body: None, span,
        },
        is_static: method.is_static, is_extern: method.is_extern,
    }
}

fn qualified(package: &[String], name: &str) -> String {
    package
        .iter()
        .map(String::as_str)
        .chain(std::iter::once(name))
        .collect::<Vec<_>>()
        .join(".")
}

fn method_field(method: &BladeMethodInfo) -> FieldDefinition {
    let span = parser::Span::default();
    let mut metadata = Vec::new();
    for (enabled, name) in [(method.is_from, ":from"), (method.is_to, ":to")] {
        if enabled {
            metadata.push(Metadata {
                compile_time: true,
                name: name.into(),
                params: Vec::new(),
                span,
            });
        }
    }
    if let Some(native) = &method.native_name {
        metadata.push(Metadata {
            compile_time: true,
            name: ":native".into(),
            params: vec![Expr {
                kind: parser::ExprKind::String(native.clone()),
                span,
            }],
            span,
        });
    }
    FieldDefinition {
        name: method.name.clone(),
        is_static: method.is_static,
        is_enum_value: false,
        init_expr: None,
        span,
        metadata,
    }
}

impl ClassRegistry {
    pub(crate) fn register_manifest(
        &mut self,
        manifest: &bsym::BladeSymbolManifest,
        roots: &[std::path::PathBuf],
    ) {
        for module in &manifest.modules {
            let source_file = roots
                .iter()
                .map(|root| root.join(&module.source_path))
                .find(|path| path.is_file())
                .unwrap_or_else(|| {
                    roots
                        .first()
                        .cloned()
                        .unwrap_or_default()
                        .join(&module.source_path)
                })
                .to_string_lossy()
                .into_owned();
            let source_view = Arc::new(std::sync::OnceLock::new());
            for class in &module.types.classes {
                let name = qualified(&class.package, &class.name);
                let methods: Vec<_> = class
                    .methods
                    .iter()
                    .chain(&class.static_methods)
                    .chain(class.constructor.iter())
                    .collect();
                let imports = Arc::new(BTreeMap::from([(class.name.clone(), name.clone())]));
                let mut fields: Vec<_> = class
                    .fields
                    .iter()
                    .chain(&class.static_fields)
                    .map(|field| FieldDefinition {
                        name: field.name.clone(),
                        is_static: field.is_static,
                        is_enum_value: false,
                        init_expr: None,
                        span: Default::default(),
                        metadata: Vec::new(),
                    })
                    .collect();
                fields.extend(methods.iter().map(|method| method_field(method)));
                let mut parents: Vec<_> = class.extends.iter().cloned().collect();
                parents.extend(class.implements.iter().cloned());
                self.insert_snapshot_definition(TypeDefinition {
                    qualified_name: name.clone(),
                    source_file: source_file.clone(),
                    runtime_class: true,
                    module: module.name.clone(),
                    metadata: Vec::new(),
                    fields,
                    parents,
                    is_abstract: false,
                    is_enum_abstract: false,
                    constructors: BTreeMap::new(),
                });
                self.short_name_index
                    .entry(class.name.clone())
                    .or_insert(name.clone());
                self.classes.insert(
                    name.clone(),
                    ClassInfo {
                        source_file: source_file.clone(),
                        name: class.name.clone(),
                        qualified_name: name,
                        constructor: None,
                        static_methods: BTreeMap::new(),
                        instance_methods: BTreeMap::new(),
                        static_vars: Vec::new(),
                        instance_vars: Vec::new(),
                        imports,
                        parent: class.extends.clone(),
                        abstract_decl: None,
                        field_types: class
                            .fields
                            .iter()
                            .chain(&class.static_fields)
                            .filter_map(|field| {
                                field
                                    .field_type
                                    .as_ref()
                                    .map(|ty| (field.name.clone(), source_type(ty)))
                            })
                            .collect(),
                        method_signatures: methods
                            .iter()
                            .map(|method| (method.name.clone(), signature(method)))
                            .collect(),
                        source_view: Some(source_view.clone()),
                    },
                );
            }
            for decl in &module.types.abstracts {
                let name = qualified(&decl.package, &decl.name);
                self.insert_snapshot_definition(TypeDefinition {
                    qualified_name: name.clone(),
                    source_file: source_file.clone(),
                    runtime_class: false,
                    module: module.name.clone(),
                    metadata: Vec::new(),
                    fields: decl
                        .methods
                        .iter()
                        .chain(&decl.static_methods)
                        .map(method_field)
                        .collect(),
                    parents: Vec::new(),
                    is_abstract: true,
                    is_enum_abstract: false,
                    constructors: BTreeMap::new(),
                });
                self.short_name_index
                    .entry(decl.name.clone())
                    .or_insert(name.clone());
                self.classes.insert(
                    name.clone(),
                    ClassInfo {
                        source_file: source_file.clone(),
                        name: decl.name.clone(),
                        qualified_name: name,
                        constructor: None,
                        static_methods: BTreeMap::new(),
                        instance_methods: BTreeMap::new(),
                        static_vars: Vec::new(),
                        instance_vars: Vec::new(),
                        imports: Arc::new(BTreeMap::new()),
                        parent: None,
                        abstract_decl: None,
                        field_types: BTreeMap::new(),
                        method_signatures: decl
                            .methods
                            .iter()
                            .chain(&decl.static_methods)
                            .map(|method| (method.name.clone(), signature(method)))
                            .collect(),
                        source_view: Some(source_view.clone()),
                    },
                );
            }
            for decl in &module.types.enums {
                let name = qualified(&decl.package, &decl.name);
                let constructors: BTreeMap<_, _> = decl
                    .variants
                    .iter()
                    .map(|variant| {
                        self.enum_constructors
                            .entry(variant.name.clone())
                            .or_default()
                            .insert(name.clone(), variant.params.len());
                        (
                            variant.name.clone(),
                            (
                                variant.params.len(),
                                variant
                                    .params
                                    .iter()
                                    .filter(|p| !p.is_optional && !p.has_default)
                                    .count(),
                            ),
                        )
                    })
                    .collect();
                self.insert_snapshot_definition(TypeDefinition {
                    qualified_name: name,
                    source_file: source_file.clone(),
                    runtime_class: false,
                    module: module.name.clone(),
                    metadata: Vec::new(),
                    fields: Vec::new(),
                    parents: Vec::new(),
                    is_abstract: false,
                    is_enum_abstract: false,
                    constructors,
                });
            }
            for alias in &module.types.type_aliases {
                let name = qualified(&alias.package, &alias.name);
                self.aliases.insert(
                    name.clone(),
                    (source_type(&alias.target_type), Arc::new(BTreeMap::new())),
                );
                self.insert_snapshot_definition(TypeDefinition {
                    qualified_name: name,
                    source_file: source_file.clone(),
                    runtime_class: false,
                    module: module.name.clone(),
                    metadata: Vec::new(),
                    fields: Vec::new(),
                    parents: Vec::new(),
                    is_abstract: false,
                    is_enum_abstract: false,
                    constructors: BTreeMap::new(),
                });
            }
        }
    }

    fn insert_snapshot_definition(&mut self, definition: TypeDefinition) {
        self.runtime_short_names.take();
        let alias = format!(
            "{}.{}",
            definition.module,
            definition.qualified_name.rsplit('.').next().unwrap()
        );
        let definition = Arc::new(definition);
        self.type_definitions.insert(alias, definition.clone());
        self.type_definitions
            .insert(definition.qualified_name.clone(), definition);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::macro_system::{MacroInterpreter, MacroRegistry, MacroValue};

    fn evaluate(
        classes: Arc<ClassRegistry>,
        expression: &str,
    ) -> Result<MacroValue, super::super::super::errors::MacroError> {
        let tokens = parser::lexer::Lexer::new(expression).tokenize().unwrap();
        let mut stream = parser::token_stream::TokenStream::new(&tokens, expression);
        let expr = parser::rd::RdParser::new(&mut stream, expression, "<snapshot-test>")
            .parse_expression()
            .unwrap();
        MacroInterpreter::with_class_registry(MacroRegistry::new(), Default::default(), classes)
            .eval_expr(&expr)
    }

    fn declarations() -> Arc<ClassRegistry> {
        let mut unit = crate::compilation::CompilationUnit::new(Default::default());
        unit.load_stdlib().unwrap();
        assert!(unit.stdlib_files.is_empty());
        Arc::new(unit.macro_class_registry())
    }

    #[test]
    fn snapshot_signatures_are_available_without_loading_bodies() {
        let classes = declarations();
        let signature = classes.method_signature("Sys", "command").unwrap();
        assert!(signature.is_static);
        assert_eq!(
            signature
                .function
                .params
                .iter()
                .map(|p| p.name.as_str())
                .collect::<Vec<_>>(),
            ["cmd", "args"]
        );
        assert!(signature.function.params[1].optional);
        assert!(
            matches!(&signature.function.return_type, Some(parser::Type::Path { path, .. }) if path.name == "Int")
        );
        assert!(
            classes
                .find_class("Sys")
                .unwrap()
                .source_view
                .as_ref()
                .unwrap()
                .get()
                .is_none()
        );
        assert!(classes.find_alias("haxe.io.BytesData").is_some());
        assert!(
            classes
                .method_signature("haxe.io.Bytes", "getData")
                .is_some()
        );
        let error = evaluate(classes, "Sys.command()").unwrap_err().to_string();
        assert!(error.contains("expects 1..2 arguments"), "{error}");
    }

    #[test]
    fn source_only_compiles_still_expose_portable_macro_declarations() {
        let mut unit =
            crate::compilation::CompilationUnit::new(crate::compilation::CompilationConfig {
                enable_cache: false,
                ..Default::default()
            });
        unit.load_stdlib().unwrap();
        let classes = Arc::new(unit.macro_class_registry());
        assert!(classes.method_signature("Sys", "command").is_some());
        assert_eq!(
            evaluate(classes, "haxe.io.Path.normalize('a/../b')").unwrap(),
            MacroValue::from_str("b")
        );
    }

    #[test]
    fn cached_stdlib_bodies_load_on_demand_and_remain_shared() {
        let classes = declarations();
        assert_eq!(
            evaluate(classes.clone(), "haxe.io.Path.normalize('a/../b')").unwrap(),
            MacroValue::from_str("b")
        );
        assert_eq!(
            evaluate(
                classes.clone(),
                "Sys.getEnvOr('__RAYZOR_MISSING_ENV_1__', '__RAYZOR_MISSING_ENV_2__')"
            )
            .unwrap(),
            MacroValue::Null
        );
        let first = classes.find_executable_class("Sys").unwrap() as *const ClassInfo;
        let second = classes.find_executable_class("Sys").unwrap() as *const ClassInfo;
        assert_eq!(first, second);
        assert!(classes.find_static_method("Sys", "getEnvOr").is_some());
    }

    #[test]
    fn source_declarations_override_snapshot_signatures() {
        let mut classes = Arc::try_unwrap(declarations()).ok().unwrap();
        let file = parser::parse_haxe_file("Path.hx", "package haxe.io; class Path { public static function normalize(value:Int):String return 'local'; }", true).unwrap();
        classes.register_file(&file);
        let classes = Arc::new(classes);
        assert_eq!(
            evaluate(classes, "haxe.io.Path.normalize(7)").unwrap(),
            MacroValue::from_str("local")
        );
    }

    #[test]
    fn portable_types_keep_optional_fields_and_generic_constraints() {
        let file = parser::parse_haxe_file("Types.hx", "typedef Props = { ?label:String; required:Null<Int>; }; class Factory { public static function make<T:{length:Int}>(value:T):T return value; }", true).unwrap();
        let symbols = bsym::extract_type_info_from_ast(&file);
        let parser::Type::Anonymous { fields, .. } =
            source_type(&symbols.type_aliases[0].target_type)
        else {
            panic!("structure")
        };
        assert!(fields[0].optional);
        assert!(!fields[1].optional);
        let method = signature(&symbols.classes[0].static_methods[0]);
        assert_eq!(method.function.type_params[0].name, "T");
        assert_eq!(method.function.type_params[0].constraints.len(), 1);
    }
}
