//! Type substitution at a generic-build use site.

use super::class_registry::{ClassRegistry, GenericBuildInfo};
use super::context_api::MacroContext;
use super::errors::MacroError;
use super::interpreter::MacroInterpreter;
use super::registry::MacroRegistry;
use parser::Expr;
use std::sync::Arc;

pub(crate) struct GenericBuildEngine {
    pub registry: MacroRegistry,
    pub class_registry: Arc<ClassRegistry>,
}

impl GenericBuildEngine {
    pub(crate) fn definition(
        &self,
        name: &str,
    ) -> Option<Arc<super::class_registry::GenericBuildInfo>> {
        self.class_registry.generic_build(name)
    }

    pub(crate) fn evaluate(
        &self,
        info: &super::class_registry::GenericBuildInfo,
        local_type: crate::tast::TypeId,
        args: Option<&[Expr]>,
        span: parser::Span,
        typer: &mut dyn super::context_api::MacroTyper,
    ) -> Result<parser::Type, MacroError> {
        let location = super::errors::span_to_location(span);
        let call = info
            .class
            .meta
            .iter()
            .find(|m| m.name.trim_start_matches(':') == "genericBuild")
            .and_then(|m| m.params.first())
            .ok_or_else(|| MacroError::ContextError {
                method: "genericBuild".to_string(),
                message: "expected a macro call".to_string(),
                location,
            })?;
        let mut context = MacroContext::new();
        context.set_call_position(location);
        let mut pack = info.pack.clone();
        if matches!(info.class.access, Some(parser::Access::Private)) {
            pack.push(format!("_{}", info.module));
        }
        let qualified_name = pack
            .iter()
            .cloned()
            .chain(std::iter::once(info.class.name.clone()))
            .collect::<Vec<_>>()
            .join(".");
        context.set_build_class(super::context_api::BuildClassContext {
            class_name: info.class.name.clone(),
            qualified_name,
            symbol_id: None,
            pack,
            fields: super::build_macros::class_fields_to_build_fields(&info.class.fields),
        });
        context.current_class = Some(info.class.name.clone());
        context.current_module = Some(
            info.pack
                .iter()
                .cloned()
                .chain(std::iter::once(info.module.clone()))
                .collect::<Vec<_>>()
                .join("."),
        );
        context.local_type = Some(local_type);
        context.call_arguments = args.map(<[Expr]>::to_vec);
        // The interpreter and its context are dropped before the live typer returns.
        unsafe {
            context.set_typer(typer);
        }
        let mut interp = MacroInterpreter::with_class_registry(
            self.registry.clone(),
            info.imports.clone(),
            self.class_registry.clone(),
        );
        interp.macro_context = Some(context);
        let macro_name = super::registry::extract_macro_name_from_meta(
            info.class
                .meta
                .iter()
                .find(|m| m.name.trim_start_matches(':') == "genericBuild")
                .unwrap(),
        );
        let result = if let Some(def) = self.registry.find_macro_by_name(&macro_name) {
            let args = match &call.kind {
                parser::ExprKind::Call { args, .. } => args
                    .iter()
                    .cloned()
                    .map(|e| super::value::MacroValue::Expr(Arc::new(e)))
                    .collect(),
                _ => Vec::new(),
            };
            interp.set_import_map((*def.imports).clone());
            interp.call_macro_def(def, args, location)
        } else {
            interp.eval_expr(call)
        };
        interp.macro_context.as_mut().unwrap().clear_typer();
        let value = match result {
            Ok(v) => v,
            Err(MacroError::Return { value: Some(v) }) => *v,
            Err(e) => return Err(e),
        };
        if !interp.defined_types.is_empty() {
            return Err(MacroError::ContextError {
                method: "genericBuild".to_string(),
                message: "types defined during generic building are not registered yet".to_string(),
                location,
            });
        }
        super::expr_adt::type_of_value(&value, span).ok_or_else(|| MacroError::ContextError {
            method: "genericBuild".to_string(),
            message: "expected a ComplexType result".to_string(),
            location,
        })
    }
}
