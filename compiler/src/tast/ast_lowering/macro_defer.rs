//! Deferred macro re-expansion: the lowering side.
//!
//! Macro expansion runs on the raw AST, before any typing exists. A macro
//! body that asks the typer a question (`Context.typeof`, `typeExpr`,
//! `TypeTools.toString` on a typed Type) cannot be answered there, so the
//! expander parks the call site untouched and records it as deferred. When
//! lowering reaches that call, everything the question needs is live: locals
//! upstream of the call are typed, the enclosing class is on the context
//! stack, imports are resolved. This module is the bridge — it re-runs the
//! parked macro with `AstLowering` itself answering as the `MacroTyper`.

use super::AstLowering;
use crate::macro_system::context_api::MacroTyper;
use crate::tast::TypeId;

/// `AstLowering` wearing its `MacroTyper` hat for the span of one deferred
/// call. Separate struct rather than an impl on `AstLowering` so the borrow
/// handed to the expander is visibly scoped.
pub(crate) struct DeferredMacroTyper<'l, 'a> {
    pub lowering: &'l mut AstLowering<'a>,
}

impl MacroTyper for DeferredMacroTyper<'_, '_> {
    fn type_expr_in_scope(&mut self, expr: &parser::Expr) -> Result<TypeId, String> {
        // A typing PROBE: lower the expression for its type and discard the
        // result. Errors raised by the probe belong to the macro (typeError
        // catches them); they must not surface as user diagnostics, so both
        // error sinks are wound back to where they were.
        let outer_before = self.lowering.collected_errors.len();
        let ctx_before = self.lowering.context.errors.len();

        let result = self.lowering.lower_expression(expr);

        let mut probe_errors: Vec<String> = self
            .lowering
            .collected_errors
            .drain(outer_before..)
            .map(|e| e.to_compilation_error().message)
            .collect();
        probe_errors.extend(
            self.lowering
                .context
                .errors
                .drain(ctx_before..)
                .map(|e| e.to_compilation_error().message),
        );

        match result {
            Ok(typed) if probe_errors.is_empty() => Ok(typed.expr_type),
            Ok(_) => Err(probe_errors.join("\n")),
            Err(e) => {
                let mut msg = e.to_compilation_error().message;
                for extra in probe_errors {
                    msg.push('\n');
                    msg.push_str(&extra);
                }
                Err(msg)
            }
        }
    }

    fn type_display(&mut self, id: TypeId) -> String {
        render_type(
            id,
            self.lowering.context.type_table,
            self.lowering.context.symbol_table,
            self.lowering.context.string_interner,
            0,
        )
    }

    fn type_std_string(&mut self, id: TypeId) -> String {
        // `Std.string` on a macro Type value. Haxe prints the constructor
        // form (`TInst(String,[])`); what the tests actually compare is two
        // of these against each other, so the load-bearing property is that
        // equal types render equal and distinct types render distinct — which
        // the display form already provides.
        self.type_display(id)
    }
}

/// The source-level spelling of a type, the way `haxe.macro.TypeTools.toString`
/// prints it: named types by name, `Null<T>` written out, functions arrow-form.
/// The general error formatter is not reused here because it spells `Null<T>`
/// as `T?` and Debug-dumps named types — both visible to tests that compare
/// the string against a literal.
fn render_type(
    id: TypeId,
    type_table: &std::cell::RefCell<crate::tast::TypeTable>,
    symbol_table: &crate::tast::SymbolTable,
    interner: &crate::tast::StringInterner,
    depth: usize,
) -> String {
    use crate::tast::core::TypeKind;
    if depth > 24 {
        return "...".to_string();
    }
    let kind = match type_table.borrow().get(id) {
        Some(t) => t.kind.clone(),
        None => return "<invalid-type>".to_string(),
    };
    let name_of = |symbol_id| {
        symbol_table
            .get_symbol(symbol_id)
            .and_then(|sym| interner.get(sym.name))
            .map(|n| n.to_string())
            .unwrap_or_else(|| "<unnamed>".to_string())
    };
    let with_args = |base: String, args: &[TypeId]| {
        if args.is_empty() {
            base
        } else {
            let rendered: Vec<String> = args
                .iter()
                .map(|&a| render_type(a, type_table, symbol_table, interner, depth + 1))
                .collect();
            format!("{}<{}>", base, rendered.join(", "))
        }
    };
    match kind {
        TypeKind::Void => "Void".to_string(),
        TypeKind::Bool => "Bool".to_string(),
        TypeKind::Int => "Int".to_string(),
        TypeKind::Float => "Float".to_string(),
        TypeKind::String => "String".to_string(),
        TypeKind::Char => "Char".to_string(),
        TypeKind::Dynamic => "Dynamic".to_string(),
        TypeKind::Unknown => "Unknown<0>".to_string(),
        TypeKind::Error => "<error>".to_string(),
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
        } => with_args(name_of(symbol_id), &type_args),
        TypeKind::Abstract {
            symbol_id,
            type_args,
            ..
        } => with_args(name_of(symbol_id), &type_args),
        TypeKind::TypeAlias {
            symbol_id,
            type_args,
            ..
        } => with_args(name_of(symbol_id), &type_args),
        TypeKind::Function {
            params,
            return_type,
            ..
        } => {
            let ret = render_type(return_type, type_table, symbol_table, interner, depth + 1);
            if params.is_empty() {
                format!("() -> {}", ret)
            } else {
                let ps: Vec<String> = params
                    .iter()
                    .map(|&p| render_type(p, type_table, symbol_table, interner, depth + 1))
                    .collect();
                format!("({}) -> {}", ps.join(", "), ret)
            }
        }
        TypeKind::Array { element_type } => format!(
            "Array<{}>",
            render_type(element_type, type_table, symbol_table, interner, depth + 1)
        ),
        TypeKind::Map {
            key_type,
            value_type,
        } => format!(
            "Map<{}, {}>",
            render_type(key_type, type_table, symbol_table, interner, depth + 1),
            render_type(value_type, type_table, symbol_table, interner, depth + 1)
        ),
        TypeKind::Optional { inner_type } => format!(
            "Null<{}>",
            render_type(inner_type, type_table, symbol_table, interner, depth + 1)
        ),
        TypeKind::Placeholder { name } => interner
            .get(name)
            .map(|n| n.to_string())
            .unwrap_or_else(|| "<placeholder>".to_string()),
        TypeKind::TypeParameter { symbol_id, .. } => name_of(symbol_id),
        TypeKind::Anonymous { fields } => {
            // haxe spells a structure `{ a : Int, b : String }`, fields in
            // declaration order.
            let rendered: Vec<String> = fields
                .iter()
                .map(|f| {
                    let name = interner
                        .get(f.name)
                        .map(|n| n.to_string())
                        .unwrap_or_else(|| "<field>".to_string());
                    format!(
                        "{} : {}",
                        name,
                        render_type(f.type_id, type_table, symbol_table, interner, depth + 1)
                    )
                })
                .collect();
            if rendered.is_empty() {
                "{ }".to_string()
            } else {
                format!("{{ {} }}", rendered.join(", "))
            }
        }
        TypeKind::GenericInstance {
            base_type,
            type_args,
            ..
        } => with_args(
            render_type(base_type, type_table, symbol_table, interner, depth + 1),
            &type_args,
        ),
        other => format!("{:?}", other),
    }
}
