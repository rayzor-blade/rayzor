//! Preserve runtime metadata before TAST discards parser annotations. The JSON
//! payload is stored in MIR attributes so cached modules retain the same data.
use crate::ir::IrModule;
use parser::{ClassFieldKind, Expr, ExprKind, HaxeFile, Metadata, Modifier, TypeDeclaration};
use serde_json::{Map, Value};

fn constant(expr: &Expr) -> Option<Value> {
    match &expr.kind {
        ExprKind::Null => Some(Value::Null),
        ExprKind::Bool(v) => Some(Value::Bool(*v)),
        ExprKind::Int(v) => Some((*v).into()),
        ExprKind::Float(v) => serde_json::Number::from_f64(*v).map(Value::Number),
        ExprKind::String(v) => Some(Value::String(v.clone())),
        ExprKind::Paren(inner) => constant(inner),
        ExprKind::Unary {
            op: parser::UnaryOp::Neg,
            expr,
        } => match constant(expr)? {
            Value::Number(v) => v
                .as_i64()
                .and_then(i64::checked_neg)
                .map(Value::from)
                .or_else(|| {
                    v.as_f64()
                        .and_then(|v| serde_json::Number::from_f64(-v))
                        .map(Value::Number)
                }),
            _ => None,
        },
        ExprKind::Array(values) => values
            .iter()
            .map(constant)
            .collect::<Option<Vec<_>>>()
            .map(Value::Array),
        ExprKind::Object(fields) => fields
            .iter()
            .map(|f| Some((f.name.clone(), constant(&f.expr)?)))
            .collect::<Option<Map<_, _>>>()
            .map(Value::Object),
        _ => None,
    }
}

fn metadata(entries: &[Metadata], source: &str) -> Result<Value, String> {
    let mut result = Map::new();
    for entry in entries {
        // The parser normalizes away ':'. Its span retains the distinction
        // between @runtime and @:compiler annotations.
        if source
            .get(entry.span.start..entry.span.end)
            .is_some_and(|s| s.trim_start().starts_with("@:"))
        {
            continue;
        }
        let args = entry
            .params
            .iter()
            .map(constant)
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| format!("Non-constant runtime metadata argument in @{}", entry.name))?;
        result.insert(
            entry.name.clone(),
            if args.is_empty() {
                Value::Null
            } else {
                Value::Array(args)
            },
        );
    }
    Ok(Value::Object(result))
}

pub(super) fn attach(module: &mut IrModule, file: &HaxeFile, source: &str) -> Result<(), String> {
    let package = file
        .package
        .as_ref()
        .map(|p| p.path.join("."))
        .unwrap_or_default();
    for decl in &file.declarations {
        let (name, meta, fields, constructors) = match decl {
            TypeDeclaration::Class(c) => (&c.name, &c.meta, c.fields.as_slice(), &[][..]),
            TypeDeclaration::Interface(c) => (&c.name, &c.meta, c.fields.as_slice(), &[][..]),
            TypeDeclaration::Enum(e) => (&e.name, &e.meta, &[][..], e.constructors.as_slice()),
            _ => continue,
        };
        let qualified = if package.is_empty() {
            name.clone()
        } else {
            format!("{package}.{name}")
        };
        let Some(id) = module
            .types
            .values()
            .find(|ty| ty.name == qualified || ty.name == *name)
            .and_then(|ty| ty.runtime_type_id)
        else {
            continue;
        };
        let mut instance = Map::new();
        let mut statics = Map::new();
        for field in fields {
            let name = match &field.kind {
                ClassFieldKind::Function(f) => &f.name,
                ClassFieldKind::Var { name, .. }
                | ClassFieldKind::Final { name, .. }
                | ClassFieldKind::Property { name, .. } => name,
            };
            let value = metadata(&field.meta, source)?;
            if value.as_object().is_some_and(|v| !v.is_empty()) {
                if field.modifiers.contains(&Modifier::Static) {
                    statics.insert(name.clone(), value);
                } else {
                    instance.insert(name.clone(), value);
                }
            }
        }
        for ctor in constructors {
            let value = metadata(&ctor.meta, source)?;
            if value.as_object().is_some_and(|v| !v.is_empty()) {
                instance.insert(ctor.name.clone(), value);
            }
        }
        for (kind, data) in [
            ("type", metadata(meta, source)?),
            ("fields", Value::Object(instance)),
            ("statics", Value::Object(statics)),
        ] {
            module
                .metadata
                .attributes
                .insert(format!("runtime_meta:{id}:{kind}"), data.to_string());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn runtime_annotations_exclude_compiler_annotations() {
        let source = "@tag @answer(42) @nested({text: 'hello', values: [1, 2]}) @:keep class A {}";
        let file = parser::rd::rd_parse(source, "A.hx", false, false).unwrap();
        let parser::TypeDeclaration::Class(class) = &file.declarations[0] else {
            panic!("class")
        };
        let value = super::metadata(&class.meta, source).unwrap();
        assert_eq!(
            value,
            serde_json::json!({"tag": null, "answer": [42], "nested": [{"text": "hello", "values": [1, 2]}]})
        );
    }
}
