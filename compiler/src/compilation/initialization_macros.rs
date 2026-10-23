//! Initialization macros run before user modules and share their macro statics.

use super::*;
use crate::macro_system::{ClassRegistry, MacroInterpreter, MacroRegistry};

impl CompilationUnit {
    pub(crate) fn macro_class_registry(&self) -> ClassRegistry {
        let mut classes = ClassRegistry::new();
        classes.use_statics(self.macro_statics.clone());
        classes.use_dependencies(self.macro_dependencies.clone());
        classes.set_class_paths(
            self.namespace_resolver
                .get_source_paths()
                .iter()
                .chain(self.config.stdlib_paths.iter())
                .cloned()
                .collect(),
        );
        let mut config = self.preprocessor_config();
        config.add_defines(&self.config.user_defines);
        config.defines.insert("macro".into());
        let defines = config
            .defines
            .into_iter()
            .map(|name| {
                let value = config
                    .values
                    .get(&name)
                    .cloned()
                    .unwrap_or_else(|| "1".into());
                (name, value)
            })
            .collect();
        classes.set_defines(defines);
        classes
    }

    pub fn run_initialization_macros(&mut self, calls: &[String]) -> Vec<String> {
        let mut errors = Vec::new();
        for call in calls {
            if let Err(error) = self.run_initialization_macro(call) {
                errors.push(format!("Initialization macro `{call}`: {error}"));
            }
        }
        errors
    }

    fn run_initialization_macro(&mut self, call: &str) -> Result<(), String> {
        let tokens = parser::lexer::Lexer::new(call)
            .tokenize()
            .map_err(|e| e.to_string())?;
        let mut stream = parser::token_stream::TokenStream::new(&tokens, call);
        let expr = parser::rd::RdParser::new(&mut stream, call, "<initialization-macro>")
            .parse_expression()
            .map_err(|e| e.message)?;
        if !stream.is_eof() {
            return Err("unexpected text after macro call".into());
        }
        let parser::ExprKind::Call { expr: callee, .. } = &expr.kind else {
            return Err("expected a static method call".into());
        };
        fn path(e: &parser::Expr) -> Option<String> {
            match &e.kind {
                parser::ExprKind::Ident(name) => Some(name.clone()),
                parser::ExprKind::Field { expr, field, .. } => {
                    Some(format!("{}.{}", path(expr)?, field))
                }
                _ => None,
            }
        }
        let qualified = path(callee).ok_or("expected a qualified macro name")?;
        let (module, _) = qualified
            .rsplit_once('.')
            .ok_or("expected Class.method()")?;
        self.load_initialization_macro_modules(module)?;
        let mut classes = self.macro_class_registry();
        classes.register_files(&self.initialization_macro_files);
        let mut registry = MacroRegistry::new();
        for file in &self.initialization_macro_files {
            registry
                .scan_and_register(file, &file.filename)
                .map_err(|e| e.to_string())?;
        }
        let context = classes.macro_context();
        let mut interpreter = MacroInterpreter::with_class_registry(
            registry,
            Default::default(),
            std::sync::Arc::new(classes),
        );
        interpreter.macro_context = Some(context);
        let _budget = crate::macro_system::budget::invocation();
        let result = interpreter.eval_expr(&expr).map_err(|e| e.to_string());
        self.global_metadata.extend(interpreter.global_metadata);
        self.macro_hooks.extend(interpreter.hooks);
        self.config.user_defines.extend(
            interpreter
                .compiler_defines
                .into_iter()
                .map(|(name, value)| format!("{name}={value}")),
        );
        result.map(|_| ())
    }

    pub(crate) fn load_initialization_macro_modules(&mut self, root: &str) -> Result<(), String> {
        let mut pending = vec![root.to_string()];
        let mut visited = BTreeSet::new();
        while let Some(mut name) = pending.pop() {
            let requested = name.clone();
            if !visited.insert(name.clone()) {
                continue;
            }
            let mut file = self
                .namespace_resolver
                .resolve_qualified_path_to_file_force(&name);
            while file.is_none() {
                let Some((parent, _)) = name.rsplit_once('.') else {
                    break;
                };
                name = parent.to_string();
                file = self
                    .namespace_resolver
                    .resolve_qualified_path_to_file_force(&name);
            }
            let Some(path) = file else {
                if requested == root {
                    return Err(format!("cannot find macro module {root}"));
                }
                continue;
            };
            let filename = path.to_string_lossy().into_owned();
            if self
                .initialization_macro_files
                .iter()
                .any(|f| f.filename == filename)
            {
                continue;
            }
            let source = self.read_source(&path).map_err(|e| e.to_string())?;
            let mut config = self.preprocessor_config_for(&filename);
            config.defines.insert("macro".into());
            let parsed = parser::haxe_parser::parse_haxe_file_with_config(
                &filename, &source, true, true, &config,
            )?;
            let dependencies = Self::extract_all_dependencies(&parsed);
            pending.extend(Self::enclosing_package_candidates(&parsed, &dependencies));
            pending.extend(dependencies);
            self.initialization_macro_files.push(parsed);
        }
        Ok(())
    }
}
