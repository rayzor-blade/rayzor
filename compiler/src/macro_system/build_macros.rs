//! Build Macros Support
//!
//! Implements `@:build` and `@:autoBuild` metadata-driven macro expansion.
//!
//! - `@:build(MacroClass.buildMethod)` — Calls a build macro on a class,
//!   passing its fields via `Context.getBuildFields()`. The macro returns
//!   modified fields which replace the class's original fields.
//!
//! - `@:autoBuild` — Applied to interfaces; when a class implements the
//!   interface, the build macro is automatically applied.
//!
//! # Processing Order
//!
//! Build macros are processed before regular macro expansion so that
//! any macro calls in the generated fields are expanded in the normal
//! expression expansion pass.

use super::context_api::{
    BuildClassContext, BuildField, BuildFieldKind, FieldAccess, FieldMeta, MacroContext,
};
use super::errors::{MacroDiagnostic, MacroError};
use super::interpreter::MacroInterpreter;
use super::registry::MacroRegistry;
use super::value::MacroValue;
use crate::tast::SourceLocation;
use parser::{
    ClassDecl, ClassField, ClassFieldKind, Expr, ExprKind, HaxeFile, InterfaceDecl, Metadata,
    Modifier, TypeDeclaration,
};
use std::sync::Arc;

/// Result of processing build macros in a file
pub struct BuildMacroResult {
    /// The modified AST file
    pub file: HaxeFile,
    /// Diagnostics emitted during build macro processing
    pub diagnostics: Vec<MacroDiagnostic>,
    /// Number of build macros applied
    pub applied_count: usize,
}

/// Process all @:build and @:autoBuild macros in a parsed file.
///
/// This should be called before regular macro expansion so that
/// generated fields are available for expression-level expansion.
pub fn process_build_macros(mut file: HaxeFile, registry: &MacroRegistry) -> BuildMacroResult {
    let mut diagnostics = Vec::new();
    let mut applied_count = 0;

    // Collect @:autoBuild interfaces first
    let auto_build_interfaces = collect_auto_build_interfaces(&file);

    // Process each declaration
    let mut new_decls = Vec::with_capacity(file.declarations.len());
    for decl in file.declarations.drain(..) {
        match decl {
            TypeDeclaration::Class(mut class) => {
                // Check for direct @:build metadata
                let build_metas: Vec<_> = class
                    .meta
                    .iter()
                    .filter(|m| m.name == "build" || m.name == ":build")
                    .cloned()
                    .collect();

                for meta in &build_metas {
                    match apply_build_macro(&mut class, meta, registry) {
                        Ok(()) => {
                            applied_count += 1;
                            diagnostics.push(MacroDiagnostic::info(
                                format!("@:build macro applied to class '{}'", class.name),
                                super::errors::span_to_location(meta.span),
                            ));
                        }
                        Err(e) => {
                            diagnostics.push(MacroDiagnostic::error(
                                format!("@:build macro failed on '{}': {}", class.name, e),
                                e.location(),
                            ));
                        }
                    }
                }

                // Check for @:autoBuild from implemented interfaces
                for auto_build in &auto_build_interfaces {
                    if class_implements(&class, &auto_build.interface_name) {
                        match apply_build_macro(&mut class, &auto_build.build_meta, registry) {
                            Ok(()) => {
                                applied_count += 1;
                                diagnostics.push(MacroDiagnostic::info(
                                    format!(
                                        "@:autoBuild from '{}' applied to class '{}'",
                                        auto_build.interface_name, class.name
                                    ),
                                    super::errors::span_to_location(auto_build.build_meta.span),
                                ));
                            }
                            Err(e) => {
                                diagnostics.push(MacroDiagnostic::error(
                                    format!(
                                        "@:autoBuild from '{}' failed on '{}': {}",
                                        auto_build.interface_name, class.name, e
                                    ),
                                    e.location(),
                                ));
                            }
                        }
                    }
                }

                new_decls.push(TypeDeclaration::Class(class));
            }
            other => new_decls.push(other),
        }
    }
    file.declarations = new_decls;

    BuildMacroResult {
        file,
        diagnostics,
        applied_count,
    }
}

/// Apply a single @:build macro to a class.
///
/// Steps:
/// 1. Extract the macro function name from the metadata
/// 2. Convert class fields to BuildField representations
/// 3. Set up a MacroContext with build class context
/// 4. Call the build macro function
/// 5. Apply returned fields back to the class
fn apply_build_macro(
    class: &mut ClassDecl,
    meta: &Metadata,
    registry: &MacroRegistry,
) -> Result<(), MacroError> {
    let location = super::errors::span_to_location(meta.span);

    // Step 1: Extract macro function name
    let macro_name = extract_build_macro_name(meta);
    if macro_name.is_empty() {
        return Err(MacroError::InvalidDefinition {
            message: "@:build metadata requires a macro function name".to_string(),
            location,
        });
    }

    // Step 2: Convert class fields to BuildField representations
    let build_fields = class_fields_to_build_fields(&class.fields);

    // Step 3: Set up context with build class info
    let mut context = MacroContext::new();
    context.set_call_position(location);
    context.set_build_class(BuildClassContext {
        class_name: class.name.clone(),
        qualified_name: class.name.clone(),
        symbol_id: None,
        fields: build_fields,
    });
    context.current_class = Some(class.name.clone());

    // Step 4: Call the build macro function
    let macro_def = registry.get_macro(&macro_name);

    let result = if let Some(def) = macro_def {
        // Macro is in the registry — execute it
        let mut interp = MacroInterpreter::new(registry.clone());
        let eval_result = interp.eval_expr(&def.body);

        match eval_result {
            Ok(val) => val,
            Err(MacroError::Return { value: Some(v) }) => *v,
            Err(MacroError::Return { value: None }) => MacroValue::Null,
            Err(e) if e.is_control_flow() => MacroValue::Null,
            Err(e) => return Err(e),
        }
    } else {
        // Macro not found — this could be from another file
        // For now, return the original fields unchanged
        return Err(MacroError::UndefinedMacro {
            name: macro_name,
            location,
        });
    };

    // Step 5: Apply returned fields
    // The macro should return an Array of Field objects
    if let MacroValue::Array(field_values) = result {
        let new_fields = values_to_class_fields(&field_values, class);
        class.fields = new_fields;
    }
    // If the macro returns null or non-array, fields remain unchanged

    Ok(())
}

// ==========================================================
// @:autoBuild support
// ==========================================================

/// Information about an @:autoBuild interface
struct AutoBuildInfo {
    interface_name: String,
    build_meta: Metadata,
}

/// Collect all interfaces with @:autoBuild metadata
fn collect_auto_build_interfaces(file: &HaxeFile) -> Vec<AutoBuildInfo> {
    let mut result = Vec::new();
    for decl in &file.declarations {
        if let TypeDeclaration::Interface(iface) = decl {
            for meta in &iface.meta {
                if meta.name == "autoBuild" || meta.name == ":autoBuild" {
                    // The @:autoBuild meta should contain or reference a @:build macro
                    // In Haxe, @:autoBuild on an interface means any implementing class
                    // gets the interface's @:build macro applied
                    if let Some(build_meta) = find_build_meta_on_interface(iface) {
                        result.push(AutoBuildInfo {
                            interface_name: iface.name.clone(),
                            build_meta: build_meta.clone(),
                        });
                    }
                }
            }
        }
    }
    result
}

/// Find @:build metadata on an interface (for @:autoBuild propagation)
fn find_build_meta_on_interface(iface: &InterfaceDecl) -> Option<&Metadata> {
    iface
        .meta
        .iter()
        .find(|m| m.name == "build" || m.name == ":build")
}

/// Check if a class implements a given interface (by name)
fn class_implements(class: &ClassDecl, interface_name: &str) -> bool {
    class.implements.iter().any(|t| {
        // Check if the type path matches the interface name
        format!("{:?}", t).contains(interface_name)
    })
}

// ==========================================================
// Conversion helpers
// ==========================================================

/// Convert parser ClassField list to BuildField representations
fn class_fields_to_build_fields(fields: &[ClassField]) -> Vec<BuildField> {
    fields.iter().map(class_field_to_build_field).collect()
}

/// Convert a single ClassField to a BuildField
fn class_field_to_build_field(field: &ClassField) -> BuildField {
    let (name, kind) = match &field.kind {
        ClassFieldKind::Var {
            name,
            type_hint,
            expr,
        } => {
            let kind = BuildFieldKind::Var {
                type_hint: type_hint.as_ref().map(|t| format!("{:?}", t)),
                expr: expr.as_ref().map(|e| Box::new(e.clone())),
            };
            (name.clone(), kind)
        }
        ClassFieldKind::Final {
            name,
            type_hint,
            expr,
        } => {
            let kind = BuildFieldKind::Var {
                type_hint: type_hint.as_ref().map(|t| format!("{:?}", t)),
                expr: expr.as_ref().map(|e| Box::new(e.clone())),
            };
            (name.clone(), kind)
        }
        ClassFieldKind::Property {
            name,
            type_hint,
            getter,
            setter,
        } => {
            let kind = BuildFieldKind::Property {
                get: format!("{:?}", getter),
                set: format!("{:?}", setter),
                type_hint: type_hint.as_ref().map(|t| format!("{:?}", t)),
            };
            (name.clone(), kind)
        }
        ClassFieldKind::Function(func) => {
            let kind = BuildFieldKind::Function {
                params: func.params.iter().map(|p| p.name.clone()).collect(),
                return_type: func.return_type.as_ref().map(|t| format!("{:?}", t)),
                body: func.body.clone(),
            };
            (func.name.clone(), kind)
        }
    };

    // Convert access modifiers
    let mut access = Vec::new();
    if let Some(parser::Access::Public) = &field.access {
        access.push(FieldAccess::Public);
    }
    if let Some(parser::Access::Private) = &field.access {
        access.push(FieldAccess::Private);
    }
    for modifier in &field.modifiers {
        match modifier {
            Modifier::Static => access.push(FieldAccess::Static),
            Modifier::Override => access.push(FieldAccess::Override),
            Modifier::Inline => access.push(FieldAccess::Inline),
            Modifier::Dynamic => access.push(FieldAccess::Dynamic),
            Modifier::Final => access.push(FieldAccess::Final),
            Modifier::Extern => access.push(FieldAccess::Extern),
            _ => {}
        }
    }

    // Convert metadata
    let meta: Vec<FieldMeta> = field
        .meta
        .iter()
        .map(|m| FieldMeta {
            name: m.name.clone(),
            params: m
                .params
                .iter()
                .map(|p| MacroValue::Expr(Arc::new(p.clone())))
                .collect(),
            pos: super::errors::span_to_location(m.span),
        })
        .collect();

    BuildField {
        name,
        kind,
        access,
        pos: super::errors::span_to_location(field.span),
        doc: None, // Doc comments not tracked in parser ClassField
        meta,
    }
}

/// Convert MacroValue field objects back to parser ClassField list.
///
/// This is a best-effort conversion — complex expressions in the macro
/// output are preserved as-is when they're Expr values.
fn values_to_class_fields(values: &[MacroValue], _class: &ClassDecl) -> Vec<ClassField> {
    values.iter().filter_map(value_to_class_field).collect()
}

/// Convert a single MacroValue (Object) back to a ClassField
fn value_to_class_field(value: &MacroValue) -> Option<ClassField> {
    let obj = match value {
        MacroValue::Object(o) => o,
        _ => return None,
    };

    let name = obj.get("name")?.as_string()?.to_string();

    // Determine field kind from the object
    let kind_obj = obj.get("kind");
    let kind_str = kind_obj
        .and_then(|k| {
            if let MacroValue::Object(ko) = k {
                ko.get("kind").and_then(|v| v.as_string()).map(String::from)
            } else {
                k.as_string().map(String::from)
            }
        })
        .unwrap_or_else(|| "FVar".to_string());

    let field_kind = match kind_str.as_str() {
        "FFun" | "function" => {
            let params = Vec::new(); // Simplified — params from kind_obj
            let body = kind_obj
                .and_then(|k| {
                    if let MacroValue::Object(ko) = k {
                        ko.get("expr")
                    } else {
                        None
                    }
                })
                .and_then(|v| {
                    if let MacroValue::Expr(e) = v {
                        Some(Box::new((**e).clone()))
                    } else {
                        None
                    }
                });

            ClassFieldKind::Function(parser::Function {
                name: name.clone(),
                type_params: Vec::new(),
                params,
                return_type: None,
                body,
                span: parser::Span::new(0, 0),
            })
        }
        _ => {
            // Default to Var
            let expr = kind_obj
                .and_then(|k| {
                    if let MacroValue::Object(ko) = k {
                        ko.get("expr")
                    } else {
                        None
                    }
                })
                .and_then(|v| {
                    if let MacroValue::Expr(e) = v {
                        Some(Box::new((**e).clone()))
                    } else {
                        None
                    }
                });

            ClassFieldKind::Var {
                name: name.clone(),
                type_hint: None,
                expr: expr.map(|e| *e),
            }
        }
    };

    // Parse access modifiers
    let access_arr = obj.get("access").and_then(|v| v.as_array());
    let mut modifiers = Vec::new();
    let mut access_val = None;

    if let Some(arr) = access_arr {
        for a in arr {
            if let MacroValue::String(s) = a {
                match &**s {
                    "Public" => access_val = Some(parser::Access::Public),
                    "Private" => access_val = Some(parser::Access::Private),
                    "Static" => modifiers.push(Modifier::Static),
                    "Override" => modifiers.push(Modifier::Override),
                    "Inline" => modifiers.push(Modifier::Inline),
                    "Final" => modifiers.push(Modifier::Final),
                    "Dynamic" => modifiers.push(Modifier::Dynamic),
                    "Extern" => modifiers.push(Modifier::Extern),
                    _ => {}
                }
            }
        }
    }

    Some(ClassField {
        meta: Vec::new(),
        access: access_val,
        modifiers,
        kind: field_kind,
        span: parser::Span::new(0, 0),
    })
}

/// Extract the macro function name from @:build metadata parameters
fn extract_build_macro_name(meta: &Metadata) -> String {
    if let Some(first) = meta.params.first() {
        match &first.kind {
            ExprKind::Ident(name) => name.clone(),
            ExprKind::Call { expr, .. } => {
                if let ExprKind::Ident(name) = &expr.kind {
                    name.clone()
                } else if let ExprKind::Field { expr, field, .. } = &expr.kind {
                    if let ExprKind::Ident(class_name) = &expr.kind {
                        format!("{}.{}", class_name, field)
                    } else {
                        field.clone()
                    }
                } else {
                    String::new()
                }
            }
            ExprKind::Field { expr, field, .. } => {
                if let ExprKind::Ident(class_name) = &expr.kind {
                    format!("{}.{}", class_name, field)
                } else {
                    field.clone()
                }
            }
            _ => String::new(),
        }
    } else {
        String::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use parser::Span;

    fn parse(source: &str) -> HaxeFile {
        parser::parse_haxe_file("test.hx", source, false).expect("parse should succeed")
    }

    #[test]
    fn test_class_fields_to_build_fields() {
        let source = r#"
            class Test {
                public var x:Int = 10;
                public static function hello() {
                    return "hello";
                }
            }
        "#;
        let file = parse(source);
        if let TypeDeclaration::Class(class) = &file.declarations[0] {
            let build_fields = class_fields_to_build_fields(&class.fields);
            assert_eq!(build_fields.len(), 2);
            assert_eq!(build_fields[0].name, "x");
            assert!(matches!(build_fields[0].kind, BuildFieldKind::Var { .. }));
            assert_eq!(build_fields[1].name, "hello");
            assert!(matches!(
                build_fields[1].kind,
                BuildFieldKind::Function { .. }
            ));
        } else {
            panic!("expected class declaration");
        }
    }

    #[test]
    fn test_build_field_access_modifiers() {
        let source = r#"
            class Test {
                public static inline function compute() { return 42; }
            }
        "#;
        let file = parse(source);
        if let TypeDeclaration::Class(class) = &file.declarations[0] {
            let build_fields = class_fields_to_build_fields(&class.fields);
            assert_eq!(build_fields.len(), 1);
            assert!(build_fields[0].access.contains(&FieldAccess::Public));
            assert!(build_fields[0].access.contains(&FieldAccess::Static));
            assert!(build_fields[0].access.contains(&FieldAccess::Inline));
        }
    }

    #[test]
    fn test_extract_build_macro_name_simple() {
        let meta = Metadata {
            name: "build".to_string(),
            params: vec![Expr {
                kind: ExprKind::Ident("myBuildMacro".to_string()),
                span: Span::new(0, 0),
            }],
            span: Span::new(0, 0),
        };
        assert_eq!(extract_build_macro_name(&meta), "myBuildMacro");
    }

    #[test]
    fn test_extract_build_macro_name_qualified() {
        let meta = Metadata {
            name: "build".to_string(),
            params: vec![Expr {
                kind: ExprKind::Field {
                    expr: Box::new(Expr {
                        kind: ExprKind::Ident("MacroUtils".to_string()),
                        span: Span::new(0, 0),
                    }),
                    field: "build".to_string(),
                    is_optional: false,
                },
                span: Span::new(0, 0),
            }],
            span: Span::new(0, 0),
        };
        assert_eq!(extract_build_macro_name(&meta), "MacroUtils.build");
    }

    #[test]
    fn test_process_build_macros_no_macros() {
        let source = "class Test { var x:Int = 42; }";
        let file = parse(source);
        let registry = MacroRegistry::new();
        let result = process_build_macros(file, &registry);
        assert_eq!(result.applied_count, 0);
        assert_eq!(result.file.declarations.len(), 1);
    }

    #[test]
    fn test_collect_auto_build_interfaces() {
        let source = r#"
            @:autoBuild
            @:build(MyMacro.autoBuild)
            interface Trackable {
                function getId():String;
            }
        "#;
        let file = parse(source);
        let infos = collect_auto_build_interfaces(&file);
        assert_eq!(infos.len(), 1);
        assert_eq!(infos[0].interface_name, "Trackable");
    }

    #[test]
    fn test_value_to_class_field_simple() {
        let mut obj = std::collections::HashMap::new();
        obj.insert("name".to_string(), MacroValue::from_str("myVar"));
        let field = value_to_class_field(&MacroValue::Object(Arc::new(obj)));
        assert!(field.is_some());
        let field = field.unwrap();
        match &field.kind {
            ClassFieldKind::Var { name, .. } => assert_eq!(name, "myVar"),
            _ => panic!("expected Var field"),
        }
    }

    #[test]
    fn test_value_to_class_field_with_access() {
        let mut obj = std::collections::HashMap::new();
        obj.insert("name".to_string(), MacroValue::from_str("test"));
        obj.insert(
            "access".to_string(),
            MacroValue::Array(Arc::new(vec![
                MacroValue::from_str("Public"),
                MacroValue::from_str("Static"),
            ])),
        );
        let field = value_to_class_field(&MacroValue::Object(Arc::new(obj)));
        assert!(field.is_some());
        let field = field.unwrap();
        assert_eq!(field.access, Some(parser::Access::Public));
        assert!(field.modifiers.contains(&Modifier::Static));
    }
}
