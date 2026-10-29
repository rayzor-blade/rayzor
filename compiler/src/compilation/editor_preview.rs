//! Disposable editor inference using import signatures instead of their bodies.

use super::*;

impl CompilationUnit {
    /// Preview graphs contain incomplete imports and must never enter BLADE.
    pub fn new_editor_preview(mut config: CompilationConfig) -> Self {
        config.typecheck_only = true;
        config.record_references = true;
        config.continue_after_macro_errors = true;
        config.enable_cache = false;
        config.profile_typecheck = false;
        config.pipeline_config.enable_macro_expansion = false;
        let mut unit = Self::new(config);
        unit.editor_preview = true;
        unit
    }
}

pub(super) fn signatures(file: &mut parser::HaxeFile) {
    fn simple(expr: &parser::Expr) -> bool {
        use parser::ExprKind::*;
        match &expr.kind {
            Int(_) | Float(_) | String(_) | Bool(_) | Null | Ident(_) | This => true,
            Field { expr, .. } | Paren(expr) | TypeCheck { expr, .. } => simple(expr),
            Array(values) | Tuple(values) => values.len() <= 32 && values.iter().all(simple),
            Object(fields) => fields.len() <= 32 && fields.iter().all(|field| simple(&field.expr)),
            Call { expr, args } => simple(expr) && args.len() <= 32 && args.iter().all(simple),
            New { args, .. } => args.len() <= 32 && args.iter().all(simple),
            _ => false,
        }
    }
    fn dynamic(span: parser::Span) -> parser::Type {
        parser::Type::Path {
            path: parser::TypePath {
                package: Vec::new(),
                name: "Dynamic".into(),
                sub: None,
            },
            params: Vec::new(),
            span,
        }
    }
    fn function(function: &mut parser::Function) {
        function.body = None;
        if function.name != "new" && function.return_type.is_none() {
            function.return_type = Some(parser::Type::Path {
                path: parser::TypePath {
                    package: Vec::new(),
                    name: "Dynamic".into(),
                    sub: None,
                },
                params: Vec::new(),
                span: function.span,
            });
        }
    }
    for declaration in &mut file.declarations {
        let fields = match declaration {
            parser::TypeDeclaration::Class(class) => {
                class.meta.retain(|meta| {
                    !matches!(
                        meta.name.trim_start_matches(':'),
                        "build" | "autoBuild" | "genericBuild"
                    )
                });
                &mut class.fields
            }
            parser::TypeDeclaration::Interface(class) => &mut class.fields,
            parser::TypeDeclaration::Abstract(class) => &mut class.fields,
            _ => continue,
        };
        for field in fields {
            match &mut field.kind {
                parser::ClassFieldKind::Function(method) => function(method),
                parser::ClassFieldKind::Var {
                    type_hint, expr, ..
                }
                | parser::ClassFieldKind::Final {
                    type_hint, expr, ..
                }
                | parser::ClassFieldKind::Property {
                    type_hint, expr, ..
                } => {
                    if type_hint.is_some() {
                        *expr = None;
                    } else if expr.as_ref().is_some_and(|expr| !simple(expr)) {
                        *type_hint = Some(dynamic(field.span));
                        *expr = None;
                    }
                }
            }
        }
    }
    for field in &mut file.module_fields {
        if let parser::ModuleFieldKind::Function(method) = &mut field.kind {
            function(method);
        }
    }
    let imports = std::mem::take(&mut file.imports);
    let uses = CompilationUnit::extract_all_dependencies(file);
    file.imports = imports
        .into_iter()
        .filter(|import| {
            let short = match &import.mode {
                parser::ImportMode::Alias(alias) => Some(alias.as_str()),
                _ => import.path.last().map(String::as_str),
            };
            matches!(
                import.mode,
                parser::ImportMode::Wildcard | parser::ImportMode::WildcardWithExclusions(_)
            ) || uses.iter().any(|name| {
                let name = name.strip_prefix("new:").unwrap_or(name);
                short.is_some_and(|short| name == short || name.starts_with(&format!("{short}.")))
                    || name == import.path.join(".")
            })
        })
        .collect();
}
