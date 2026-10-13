//! XML type information for classes annotated with `@:rtti`.

use super::*;

fn escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

impl AstLowering<'_> {
    fn rtti_symbol_path(&self, id: SymbolId) -> String {
        self.context
            .symbol_table
            .get_symbol(id)
            .and_then(|s| {
                self.context
                    .string_interner
                    .get(s.qualified_name.unwrap_or(s.name))
            })
            .unwrap_or("Dynamic")
            .to_owned()
    }

    fn rtti_type_xml(&self, ty: TypeId, depth: usize) -> String {
        if depth > 32 {
            return "<unknown/>".into();
        }
        let table = self.context.type_table.borrow();
        let Some(info) = table.get(ty) else {
            return "<unknown/>".into();
        };
        let named = |tag: &str, path: &str, args: &[TypeId]| {
            let children: String = args
                .iter()
                .map(|ty| self.rtti_type_xml(*ty, depth + 1))
                .collect();
            format!("<{tag} path=\"{}\">{children}</{tag}>", escape(path))
        };
        match &info.kind {
            TypeKind::Void => named("x", "Void", &[]),
            TypeKind::Bool => named("x", "Bool", &[]),
            TypeKind::Int => named("x", "Int", &[]),
            TypeKind::Float => named("x", "Float", &[]),
            TypeKind::String => named("c", "String", &[]),
            TypeKind::Class {
                symbol_id,
                type_args,
            }
            | TypeKind::Interface {
                symbol_id,
                type_args,
            } => named("c", &self.rtti_symbol_path(*symbol_id), type_args),
            TypeKind::Enum {
                symbol_id,
                type_args,
            } => named("e", &self.rtti_symbol_path(*symbol_id), type_args),
            TypeKind::Abstract {
                symbol_id,
                type_args,
                ..
            } => named("x", &self.rtti_symbol_path(*symbol_id), type_args),
            TypeKind::TypeAlias {
                symbol_id,
                type_args,
                ..
            } => named("t", &self.rtti_symbol_path(*symbol_id), type_args),
            TypeKind::TypeParameter { symbol_id, .. } => {
                named("c", &self.rtti_symbol_path(*symbol_id), &[])
            }
            TypeKind::Array { element_type } => named("c", "Array", &[*element_type]),
            TypeKind::Map {
                key_type,
                value_type,
            } => named("t", "Map", &[*key_type, *value_type]),
            TypeKind::Optional { inner_type } => named("x", "Null", &[*inner_type]),
            TypeKind::Dynamic => "<d/>".into(),
            TypeKind::Function {
                params,
                return_type,
                labels,
                ..
            } => {
                let names = labels
                    .iter()
                    .map(|label| {
                        let name = label
                            .name
                            .and_then(|name| self.context.string_interner.get(name))
                            .unwrap_or("");
                        format!("{}{name}", if label.optional { "?" } else { "" })
                    })
                    .collect::<Vec<_>>()
                    .join(":");
                let children: String = params
                    .iter()
                    .chain(std::iter::once(return_type))
                    .map(|ty| self.rtti_type_xml(*ty, depth + 1))
                    .collect();
                format!("<f a=\"{}\">{children}</f>", escape(&names))
            }
            TypeKind::Anonymous { fields } => {
                let children: String = fields
                    .iter()
                    .map(|field| {
                        let name = self.context.string_interner.get(field.name).unwrap_or("");
                        format!(
                            "<{name}>{}</{name}>",
                            self.rtti_type_xml(field.type_id, depth + 1)
                        )
                    })
                    .collect();
                format!("<a>{children}</a>")
            }
            TypeKind::Reference { target_type, .. } => self.rtti_type_xml(*target_type, depth + 1),
            _ => "<unknown/>".into(),
        }
    }

    fn rtti_expression(&self, expr: &Expr) -> String {
        if let ExprKind::String(value) = &expr.kind {
            return serde_json::to_string(value).unwrap_or_default();
        }
        if let Some(text) = self
            .context
            .span_converter
            .as_ref()
            .and_then(|converter| converter.source_slice(expr.span))
            .filter(|text| !text.trim().is_empty())
        {
            return text.trim().to_owned();
        }
        match &expr.kind {
            ExprKind::Int(value) => value.to_string(),
            ExprKind::Float(value) => value.to_string(),
            ExprKind::Bool(value) => value.to_string(),
            ExprKind::Null => "null".into(),
            _ => String::new(),
        }
    }

    pub(super) fn class_rtti_xml(
        &self,
        decl: &ClassDecl,
        symbol: SymbolId,
        fields: &[TypedField],
        methods: &[TypedFunction],
        constructors: &[TypedFunction],
    ) -> String {
        let params = decl
            .type_params
            .iter()
            .map(|p| p.name.as_str())
            .collect::<Vec<_>>()
            .join(":");
        let mut xml = format!(
            "<class path=\"{}\" params=\"{}\">",
            escape(&self.rtti_symbol_path(symbol)),
            escape(&params)
        );
        for field in &decl.fields {
            let (name, expr) = match &field.kind {
                ClassFieldKind::Function(function) => (function.name.as_str(), None),
                ClassFieldKind::Var { name, expr, .. }
                | ClassFieldKind::Final { name, expr, .. }
                | ClassFieldKind::Property { name, expr, .. } => (name.as_str(), expr.as_ref()),
            };
            xml.push_str(&format!("<{name}"));
            if field.modifiers.contains(&Modifier::Static) {
                xml.push_str(" static=\"1\"");
            }
            if field.access == Some(parser::Access::Public) {
                xml.push_str(" public=\"1\"");
            }
            if field.modifiers.contains(&Modifier::Override) {
                xml.push_str(" override=\"1\"");
            }
            if let Some(expr) = expr {
                xml.push_str(&format!(
                    " expr=\"{}\"",
                    escape(&self.rtti_expression(expr))
                ));
            }
            if let ClassFieldKind::Function(function) = &field.kind {
                xml.push_str(" set=\"method\">");
                if let Some(typed) = methods
                    .iter()
                    .chain(constructors)
                    .find(|method| self.context.string_interner.get(method.name) == Some(name))
                {
                    let args = function
                        .params
                        .iter()
                        .map(|p| {
                            format!(
                                "{}{}",
                                if p.optional || p.default_value.is_some() {
                                    "?"
                                } else {
                                    ""
                                },
                                p.name
                            )
                        })
                        .collect::<Vec<_>>()
                        .join(":");
                    let values = function
                        .params
                        .iter()
                        .map(|p| {
                            p.default_value
                                .as_deref()
                                .map(|e| self.rtti_expression(e))
                                .unwrap_or_default()
                        })
                        .collect::<Vec<_>>()
                        .join(":");
                    xml.push_str(&format!(
                        "<f a=\"{}\" v=\"{}\">",
                        escape(&args),
                        escape(&values)
                    ));
                    for param in &typed.parameters {
                        xml.push_str(&self.rtti_type_xml(param.param_type, 0));
                    }
                    xml.push_str(&self.rtti_type_xml(typed.return_type, 0));
                    xml.push_str("</f>");
                } else {
                    xml.push_str("<unknown/>");
                }
            } else {
                xml.push('>');
                if let Some(typed) = fields
                    .iter()
                    .find(|field| self.context.string_interner.get(field.name) == Some(name))
                {
                    xml.push_str(&self.rtti_type_xml(typed.field_type, 0));
                } else {
                    xml.push_str("<unknown/>");
                }
            }
            xml.push_str(&format!("</{name}>"));
        }
        xml.push_str("</class>");
        xml
    }
}
