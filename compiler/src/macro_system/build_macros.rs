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

    let file_pack: Vec<String> = file
        .package
        .as_ref()
        .map(|p| p.path.clone())
        .unwrap_or_default();
    let file_module = std::path::Path::new(&file.filename)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string();
    // Collect @:autoBuild interfaces first
    let auto_build_interfaces = collect_auto_build_interfaces(&file);
    // Which @:autoBuild types each class sits below, decided before the
    // declarations are taken apart.
    let auto_build_targets: std::collections::BTreeMap<String, Vec<usize>> = file
        .declarations
        .iter()
        .filter_map(|d| match d {
            TypeDeclaration::Class(c) => Some(c),
            _ => None,
        })
        .map(|c| {
            let hits = auto_build_interfaces
                .iter()
                .enumerate()
                .filter(|(_, a)| class_implements_in(&file.declarations, c, &a.interface_name))
                .map(|(i, _)| i)
                .collect();
            (c.name.clone(), hits)
        })
        .collect();

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
                    match apply_build_macro(
                        &mut class,
                        meta,
                        registry,
                        class_registry.clone(),
                        &file_pack,
                        &file_module,
                    ) {
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
                let targets = auto_build_targets
                    .get(&class.name)
                    .cloned()
                    .unwrap_or_default();
                for (index, auto_build) in auto_build_interfaces.iter().enumerate() {
                    if targets.contains(&index) {
                        match apply_build_macro(
                            &mut class,
                            &auto_build.build_meta,
                            registry,
                            class_registry.clone(),
                            &file_pack,
                            &file_module,
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
    pack: &[String],
    module: &str,
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
    let mut class_pack = pack.to_vec();
    if matches!(class.access, Some(parser::Access::Private)) {
        class_pack.push(format!("_{}", module));
    }
    let qualified_name = class_pack
        .iter()
        .cloned()
        .chain(std::iter::once(class.name.clone()))
        .collect::<Vec<_>>()
        .join(".");
    context.set_build_class(BuildClassContext {
        class_name: class.name.clone(),
        qualified_name,
        symbol_id: None,
        fields: build_fields,
        pack: class_pack,
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
        // The body resolves names through its own module's imports.
        let mut interp = if let Some(cr) = class_registry {
            MacroInterpreter::with_class_registry(registry.clone(), (*def.imports).clone(), cr)
        } else {
            MacroInterpreter::with_imports(registry.clone(), (*def.imports).clone())
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
    } else if let (Some(cr), Some(call)) = (class_registry, meta.params.first()) {
        // An ordinary static function may build a class too: evaluate the
        // `@:build(...)` call itself against the class registry.
        let mut interp =
            MacroInterpreter::with_class_registry(registry.clone(), Default::default(), cr);
        interp.macro_context = Some(context);
        match interp.eval_expr(call) {
            Ok(val) => val,
            Err(MacroError::Return { value: Some(v) }) => *v,
            Err(MacroError::UndefinedVariable { .. }) => {
                return Err(MacroError::UndefinedMacro {
                    name: macro_name,
                    location,
                });
            }
            Err(e) => return Err(e),
        }
    } else {
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
    // `@:autoBuild(call)` on a class or interface carries the build call
    // itself; it is applied as `@:build(call)` to every type below it.
    let mut result = Vec::new();
    for decl in &file.declarations {
        let (name, meta) = match decl {
            TypeDeclaration::Interface(iface) => (&iface.name, &iface.meta),
            TypeDeclaration::Class(class) => (&class.name, &class.meta),
            _ => continue,
        };
        for m in meta {
            if m.name != "autoBuild" && m.name != ":autoBuild" {
                continue;
            }
            // A bare `@:autoBuild` propagates the type's own `@:build`.
            let build_meta = if m.params.is_empty() {
                match meta
                    .iter()
                    .find(|b| b.name == "build" || b.name == ":build")
                {
                    Some(b) => b.clone(),
                    None => continue,
                }
            } else {
                Metadata {
                    name: "build".to_string(),
                    params: m.params.clone(),
                    span: m.span,
                    compile_time: true,
                }
            };
            result.push(AutoBuildInfo {
                interface_name: name.clone(),
                build_meta,
            });
        }
    }
    result
}

/// The declared name of a type reference.
fn type_name(t: &parser::Type) -> Option<&str> {
    match t {
        parser::Type::Path { path, .. } => Some(&path.name),
        _ => None,
    }
}

/// Whether `class` extends or implements `ancestor`, directly or through
/// other types this file declares.
fn class_implements_in(file_decls: &[TypeDeclaration], class: &ClassDecl, ancestor: &str) -> bool {
    let mut pending: Vec<String> = class
        .extends
        .iter()
        .chain(class.implements.iter())
        .filter_map(|t| type_name(t).map(str::to_string))
        .collect();
    let mut seen = std::collections::BTreeSet::new();
    while let Some(name) = pending.pop() {
        if name == ancestor {
            return true;
        }
        if !seen.insert(name.clone()) {
            continue;
        }
        for decl in file_decls {
            match decl {
                TypeDeclaration::Class(c) if c.name == name => pending.extend(
                    c.extends
                        .iter()
                        .chain(c.implements.iter())
                        .filter_map(|t| type_name(t).map(str::to_string)),
                ),
                TypeDeclaration::Interface(i) if i.name == name => pending.extend(
                    i.extends
                        .iter()
                        .filter_map(|t| type_name(t).map(str::to_string)),
                ),
                _ => {}
            }
        }
    }
    false
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
pub(crate) fn value_to_class_field(value: &MacroValue) -> Option<ClassField> {
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
            // `{args, ret}` of the FFun payload, as `macro class` and
            // hand-built fields carry them.
            let kind_field = |name: &str| match kind_obj {
                Some(MacroValue::Object(ko)) => ko.get(name).cloned(),
                _ => None,
            };
            let span = parser::Span::new(0, 0);
            let params: Vec<parser::FunctionParam> = match kind_field("args") {
                Some(MacroValue::Array(args)) => args
                    .iter()
                    .filter_map(|a| {
                        let MacroValue::Object(a) = a else {
                            return None;
                        };
                        Some(parser::FunctionParam {
                            meta: Vec::new(),
                            name: a.get("name")?.as_string()?.to_string(),
                            type_hint: a
                                .get("type")
                                .and_then(|t| super::expr_adt::type_of_value(t, span)),
                            optional: matches!(a.get("opt"), Some(MacroValue::Bool(true))),
                            rest: false,
                            default_value: a
                                .get("value")
                                .and_then(|v| super::expr_adt::try_expr_of(v, span))
                                .map(Box::new),
                            span,
                        })
                    })
                    .collect(),
                _ => Vec::new(),
            };
            let declared_ret =
                kind_field("ret").and_then(|t| super::expr_adt::type_of_value(&t, span));
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
                        // Function bodies in Haxe AST are conventionally
                        // `Block`s. A reified single-expression body like
                        // `macro return $v{n}` lands here as a bare
                        // `Return(...)`, and the downstream HIR/MIR pipeline
                        // needs the surrounding Block to allocate a function
                        // scope and emit a proper terminator. Without the
                        // wrap, the generated method's MIR ends up as
                        // `unreachable`, which then SIGILLs at runtime.
                        let raw = (**e).clone();
                        let wrapped = match &raw.kind {
                            parser::ExprKind::Block(_) => raw,
                            _ => parser::Expr {
                                kind: parser::ExprKind::Block(vec![parser::BlockElement::Expr(
                                    raw.clone(),
                                )]),
                                span: raw.span,
                            },
                        };
                        Some(Box::new(wrapped))
                    } else {
                        None
                    }
                });

            // Best-effort return-type inference. The macro source declares
            // the return type via `ret: macro :Int`, but reification turns
            // that into an opaque placeholder we can't recover statically.
            // Without ANY return type the downstream type checker tends to
            // fall back to `Dynamic`, which then breaks generic uses like
            // `trace(t.fieldCount())` (the value gets routed through a
            // `DynamicValue*`-expecting formatter even though it's actually
            // an `Int`). Sniff the body for a terminal `Return` of a
            // primitive literal and synthesise a matching `Type::Path` so
            // the call sites see the right concrete type.
            let return_type = body.as_deref().and_then(|b| {
                fn primitive_type_path(name: &str) -> parser::Type {
                    parser::Type::Path {
                        path: parser::TypePath {
                            package: Vec::new(),
                            name: name.to_string(),
                            sub: None,
                        },
                        params: Vec::new(),
                        span: parser::Span::new(0, 0),
                    }
                }
                fn sniff(expr: &parser::Expr) -> Option<parser::Type> {
                    match &expr.kind {
                        parser::ExprKind::Block(elems) => {
                            elems.iter().rev().find_map(|e| match e {
                                parser::BlockElement::Expr(ex) => sniff(ex),
                                _ => None,
                            })
                        }
                        parser::ExprKind::Return(Some(inner)) => sniff(inner),
                        parser::ExprKind::Int(_) => Some(primitive_type_path("Int")),
                        parser::ExprKind::Float(_) => Some(primitive_type_path("Float")),
                        parser::ExprKind::Bool(_) => Some(primitive_type_path("Bool")),
                        parser::ExprKind::String(_) => Some(primitive_type_path("String")),
                        parser::ExprKind::Array(_) => {
                            // Don't try to be clever about element types
                            // here — leaving as None lets TAST infer.
                            None
                        }
                        _ => None,
                    }
                }
                sniff(b)
            });
            let return_type = declared_ret.or(return_type);

            ClassFieldKind::Function(parser::Function {
                name: name.clone(),
                type_params: Vec::new(),
                params,
                return_type,
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
            compile_time: true,
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
            compile_time: true,
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
            compile_time: true,
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
            compile_time: true,
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
        assert!(errs.is_empty(), "unexpected build macro errors: {:?}", errs);
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
