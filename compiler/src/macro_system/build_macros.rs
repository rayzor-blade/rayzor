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
pub fn process_build_macros(file: HaxeFile, registry: &MacroRegistry) -> BuildMacroResult {
    process_build_macros_with_class_registry(file, registry, None)
}

/// Like [`process_build_macros`] but threads a `ClassRegistry` to the
/// interpreter so build-macro bodies can resolve sibling helpers and
/// imported classes by short name (e.g. `Context` → `haxe.macro.Context`).
pub fn process_build_macros_with_class_registry(
    mut file: HaxeFile,
    registry: &MacroRegistry,
    class_registry: Option<Arc<super::class_registry::ClassRegistry>>,
) -> BuildMacroResult {
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
                    match apply_build_macro(&mut class, meta, registry, class_registry.clone()) {
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
                        match apply_build_macro(
                            &mut class,
                            &auto_build.build_meta,
                            registry,
                            class_registry.clone(),
                        ) {
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
    class_registry: Option<Arc<super::class_registry::ClassRegistry>>,
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

    // Step 4: Call the build macro function.
    // Try exact FQN first; fall back to simple-name lookup so partial names
    // still resolve (e.g. `@:build(Json.build)` after `import tink.Json` — the
    // registry key is the FQN `tink.Json.build`).
    let macro_def = registry
        .get_macro(&macro_name)
        .or_else(|| registry.find_macro_by_name(&macro_name));

    let result = if let Some(def) = macro_def {
        // Macro is in the registry — execute it. Thread the ClassRegistry
        // when available so bare-name references inside the build-macro
        // body (e.g. `Context`, `FFun`, sibling static helpers) resolve
        // via the short-name index. Without this, every `Context.*` call
        // in a build macro fails with `undefined variable: 'Context'`
        // because @:build is dispatched before the caller's import_map
        // could ever matter.
        let mut interp = if let Some(cr) = class_registry {
            MacroInterpreter::with_class_registry(
                registry.clone(),
                std::collections::BTreeMap::new(),
                cr,
            )
        } else {
            MacroInterpreter::new(registry.clone())
        };
        // Seed the macro_class_stack with the macro's defining class so
        // the interpreter's bare-identifier fallback (see interpreter.rs
        // eval_call's Ident arm) finds sibling static helpers when the
        // build macro delegates to them.
        if let Some((defining_class, _)) = def.qualified_name.rsplit_once('.') {
            interp.push_macro_class(defining_class.to_string());
        }
        // Pass the build context so Context.getBuildFields() returns class fields
        interp.macro_context = Some(context);
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
/// `BuildField`'s on-the-wire representation drops parameter type info
/// and return type info (params come over as bare name strings, types
/// come over as debug-formatted strings). For fields the build macro
/// passed through unchanged, that loss would silently break the original
/// declaration — e.g. a constructor `new(host, port, maxConn)` would lose
/// its parameters and the body would fail with `Cannot find name 'maxConn'`.
///
/// To preserve fidelity we resolve each rebuilt field by name against the
/// original class. If the original has the same field name and matching
/// kind, swap in the original's full `ClassFieldKind` — this keeps params,
/// return types, type hints, and bodies intact for unchanged fields.
/// Newly synthesised fields (no name match) keep the round-trip-rebuilt
/// form.
fn values_to_class_fields(values: &[MacroValue], class: &ClassDecl) -> Vec<ClassField> {
    let originals: std::collections::BTreeMap<String, &ClassField> = class
        .fields
        .iter()
        .map(|f| (field_name(f).to_string(), f))
        .collect();

    values
        .iter()
        .filter_map(value_to_class_field)
        .map(|mut rebuilt| {
            if let Some(original) = originals.get(field_name(&rebuilt)) {
                // Same kind in both → replace with original to recover
                // params / return types / type hints / bodies that the
                // BuildField wire format dropped.
                let same_kind = matches!(
                    (&rebuilt.kind, &original.kind),
                    (ClassFieldKind::Function(_), ClassFieldKind::Function(_))
                        | (ClassFieldKind::Var { .. }, ClassFieldKind::Var { .. })
                        | (ClassFieldKind::Final { .. }, ClassFieldKind::Final { .. })
                        | (
                            ClassFieldKind::Property { .. },
                            ClassFieldKind::Property { .. }
                        )
                );
                if same_kind {
                    rebuilt.kind = original.kind.clone();
                    rebuilt.span = original.span;
                    // Merge original metadata that the macro may have
                    // dropped, but keep any access/modifiers explicitly
                    // set by the rebuilt form.
                    if rebuilt.meta.is_empty() {
                        rebuilt.meta = original.meta.clone();
                    }
                }
            }
            rebuilt
        })
        .collect()
}

fn field_name(field: &ClassField) -> &str {
    match &field.kind {
        ClassFieldKind::Var { name, .. }
        | ClassFieldKind::Final { name, .. }
        | ClassFieldKind::Property { name, .. } => name,
        ClassFieldKind::Function(func) => &func.name,
    }
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

/// Extract the macro function name from @:build metadata parameters.
/// Delegates to the shared helper so nested FQN forms like
/// `@:build(tink.Json.build)` resolve to `"tink.Json.build"` rather than
/// the leaf `"build"`.
fn extract_build_macro_name(meta: &Metadata) -> String {
    super::registry::extract_macro_name_from_meta(meta)
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

    /// Nested FQN — `@:build(tink.Json.build)` — must not collapse to `"build"`.
    /// This was the Phase 3 bug: the chained `Field(Field(...))` shape caused
    /// `@:build` lookup to fail with "undefined macro: 'build'" even when the
    /// macro was correctly registered under `tink.Json.build`.
    #[test]
    fn test_extract_build_macro_name_nested_fqn() {
        let meta = Metadata {
            name: "build".to_string(),
            params: vec![Expr {
                kind: ExprKind::Field {
                    expr: Box::new(Expr {
                        kind: ExprKind::Field {
                            expr: Box::new(Expr {
                                kind: ExprKind::Ident("tink".to_string()),
                                span: Span::new(0, 0),
                            }),
                            field: "Json".to_string(),
                            is_optional: false,
                        },
                        span: Span::new(0, 0),
                    }),
                    field: "build".to_string(),
                    is_optional: false,
                },
                span: Span::new(0, 0),
            }],
            span: Span::new(0, 0),
        };
        assert_eq!(extract_build_macro_name(&meta), "tink.Json.build");
    }

    /// Parameterised form `@:build(M.make(arg))` — callee is a nested Field;
    /// extractor should unwrap Call and still produce the qualified path.
    #[test]
    fn test_extract_build_macro_name_nested_call() {
        let meta = Metadata {
            name: "build".to_string(),
            params: vec![Expr {
                kind: ExprKind::Call {
                    expr: Box::new(Expr {
                        kind: ExprKind::Field {
                            expr: Box::new(Expr {
                                kind: ExprKind::Field {
                                    expr: Box::new(Expr {
                                        kind: ExprKind::Ident("pkg".to_string()),
                                        span: Span::new(0, 0),
                                    }),
                                    field: "M".to_string(),
                                    is_optional: false,
                                },
                                span: Span::new(0, 0),
                            }),
                            field: "make".to_string(),
                            is_optional: false,
                        },
                        span: Span::new(0, 0),
                    }),
                    args: vec![],
                },
                span: Span::new(0, 0),
            }],
            span: Span::new(0, 0),
        };
        assert_eq!(extract_build_macro_name(&meta), "pkg.M.make");
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

    /// Phase 6 regression guard: when a build macro returns the original
    /// `Context.getBuildFields()` array unchanged (or with new fields
    /// appended), the round-trip through `BuildField` MUST preserve each
    /// original field's full kind — params, return type, type hints,
    /// body — not collapse a `function new(host, port, maxConn, debug)`
    /// into a parameterless function whose body references undefined
    /// names.
    #[test]
    fn test_build_macro_preserves_original_function_params() {
        let source = r#"
            class Builder {
                macro public static function build():Array<Field> {
                    return Context.getBuildFields();
                }
            }
            @:build(Builder.build)
            class Target {
                public var x:Int;
                public function new(a:Int, b:String, c:Bool) {
                    this.x = a;
                }
            }
        "#;
        let file = parse(source);
        let mut registry = MacroRegistry::new();
        registry.scan_and_register(&file, "test.hx").unwrap();

        let mut class_registry = super::super::class_registry::ClassRegistry::new();
        class_registry.register_file(&file);

        let result = process_build_macros_with_class_registry(
            file,
            &registry,
            Some(Arc::new(class_registry)),
        );

        let target = result
            .file
            .declarations
            .iter()
            .find_map(|d| match d {
                TypeDeclaration::Class(c) if c.name == "Target" => Some(c),
                _ => None,
            })
            .expect("Target class");

        let new_fn = target
            .fields
            .iter()
            .find(|f| match &f.kind {
                ClassFieldKind::Function(func) => func.name == "new",
                _ => false,
            })
            .expect("new function");

        match &new_fn.kind {
            ClassFieldKind::Function(func) => {
                assert_eq!(
                    func.params.len(),
                    3,
                    "constructor lost its params on round-trip; \
                     got {:?}",
                    func.params
                );
                assert_eq!(func.params[0].name, "a");
                assert_eq!(func.params[1].name, "b");
                assert_eq!(func.params[2].name, "c");
                assert!(
                    func.body.is_some(),
                    "constructor lost its body on round-trip"
                );
            }
            _ => unreachable!(),
        }
    }

    /// Phase 5.5 regression guard: build macros that reference bare class
    /// names (e.g. `Context` imported in the defining file) must resolve
    /// via the ClassRegistry when passed, not fail with
    /// `undefined variable: 'Context'`.
    #[test]
    fn test_build_macro_resolves_context_via_class_registry() {
        let source = r#"
            class Builder {
                macro public static function build():Array<Int> {
                    Context.currentPos();
                    return [];
                }
            }
            @:build(Builder.build)
            class Target {
                var x:Int;
            }
        "#;
        let file = parse(source);

        // Register the macro and Context class in the class registry
        let mut registry = MacroRegistry::new();
        registry.scan_and_register(&file, "test.hx").unwrap();

        // Simulate haxe.macro.Context being available
        let mut class_registry = super::super::class_registry::ClassRegistry::new();
        class_registry.register_file(&file);
        let ctx_source = r#"
            package haxe.macro;
            class Context {
                public static function currentPos():Int { return 0; }
            }
        "#;
        let ctx_file = parse(ctx_source);
        class_registry.register_file(&ctx_file);

        let result = process_build_macros_with_class_registry(
            file,
            &registry,
            Some(Arc::new(class_registry)),
        );

        // The build macro should NOT error out with undefined Context.
        let errs: Vec<_> = result
            .diagnostics
            .iter()
            .filter(|d| matches!(d.severity, super::super::errors::MacroSeverity::Error))
            .map(|d| d.message.clone())
            .collect();
        assert!(
            errs.is_empty(),
            "unexpected build macro errors: {:?}",
            errs
        );
        assert_eq!(result.applied_count, 1);
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
        let mut obj = std::collections::BTreeMap::new();
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
        let mut obj = std::collections::BTreeMap::new();
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
