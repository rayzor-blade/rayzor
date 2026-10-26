//! Comprehensive integration tests for the Haxe macro system.
//!
//! Tests the full macro pipeline: parsing → registration → interpretation
//! → expansion → AST output. Covers all subsystems:
//! - MacroValue operations and conversions
//! - Environment scoping
//! - Registry scanning and lookup
//! - AST bridge (bidirectional Expr ↔ MacroValue)
//! - Reification engine ($v, $i, $e, $a, $p, $b)
//! - Interpreter (expression evaluation, control flow, builtins)
//! - Expander (full-file macro expansion)
//! - Build macros (@:build, @:autoBuild)
//! - Context API (diagnostics, defines)
//! - Error handling and diagnostics

use compiler::macro_system::*;
use parser::{BinaryOp, Expr, ExprKind, Span};
use std::sync::Arc;

// ================================================================
// HELPERS
// ================================================================

fn parse(source: &str) -> parser::HaxeFile {
    parser::parse_haxe_file("test.hx", source, false).expect("parse should succeed")
}

fn default_span() -> Span {
    Span::default()
}

fn int_expr(val: i64) -> Expr {
    Expr {
        kind: ExprKind::Int(val),
        span: default_span(),
    }
}

fn float_expr(val: f64) -> Expr {
    Expr {
        kind: ExprKind::Float(val),
        span: default_span(),
    }
}

fn str_expr(val: &str) -> Expr {
    Expr {
        kind: ExprKind::String(val.to_string()),
        span: default_span(),
    }
}

fn bool_expr(val: bool) -> Expr {
    Expr {
        kind: ExprKind::Bool(val),
        span: default_span(),
    }
}

fn null_expr() -> Expr {
    Expr {
        kind: ExprKind::Null,
        span: default_span(),
    }
}

fn ident_expr(name: &str) -> Expr {
    Expr {
        kind: ExprKind::Ident(name.to_string()),
        span: default_span(),
    }
}

fn unknown_loc() -> compiler::tast::SourceLocation {
    compiler::tast::SourceLocation::unknown()
}

// ================================================================
// 1. MACRO VALUE TESTS
// ================================================================

#[test]
fn test_macro_value_type_names() {
    assert_eq!(MacroValue::Null.type_name(), "Null");
    assert_eq!(MacroValue::Bool(true).type_name(), "Bool");
    assert_eq!(MacroValue::Int(42).type_name(), "Int");
    assert_eq!(MacroValue::Float(1.23).type_name(), "Float");
    assert_eq!(MacroValue::from_str("hi").type_name(), "String");
    assert_eq!(MacroValue::Array(Arc::new(vec![])).type_name(), "Array");
    assert_eq!(
        MacroValue::Object(Arc::new(std::collections::BTreeMap::new())).type_name(),
        "Object"
    );
}

#[test]
fn test_macro_value_truthiness() {
    // Falsy values
    assert!(!MacroValue::Null.is_truthy());
    assert!(!MacroValue::Bool(false).is_truthy());
    assert!(!MacroValue::Int(0).is_truthy());
    assert!(!MacroValue::Float(0.0).is_truthy());
    assert!(!MacroValue::from_str("").is_truthy());
    assert!(!MacroValue::Array(Arc::new(vec![])).is_truthy());

    // Truthy values
    assert!(MacroValue::Bool(true).is_truthy());
    assert!(MacroValue::Int(1).is_truthy());
    assert!(MacroValue::Int(-1).is_truthy());
    assert!(MacroValue::Float(0.1).is_truthy());
    assert!(MacroValue::from_str("x").is_truthy());
    assert!(MacroValue::Array(Arc::new(vec![MacroValue::Null])).is_truthy());
    assert!(MacroValue::Object(Arc::new(std::collections::BTreeMap::new())).is_truthy());
}

#[test]
fn test_macro_value_conversions() {
    // as_int
    assert_eq!(MacroValue::Int(42).as_int(), Some(42));
    assert_eq!(MacroValue::Float(3.7).as_int(), Some(3));
    assert_eq!(MacroValue::Bool(true).as_int(), Some(1));
    assert_eq!(MacroValue::Bool(false).as_int(), Some(0));
    assert_eq!(MacroValue::from_str("x").as_int(), None);

    // as_float
    assert_eq!(MacroValue::Float(1.23).as_float(), Some(1.23));
    assert_eq!(MacroValue::Int(5).as_float(), Some(5.0));
    assert_eq!(MacroValue::Null.as_float(), None);

    // as_string
    assert_eq!(MacroValue::from_str("hello").as_string(), Some("hello"));
    assert_eq!(MacroValue::Int(42).as_string(), None);

    // as_bool
    assert_eq!(MacroValue::Bool(true).as_bool(), Some(true));
    assert_eq!(MacroValue::Bool(false).as_bool(), Some(false));
    assert_eq!(MacroValue::Int(1).as_bool(), None);
}

#[test]
fn test_macro_value_display_string() {
    assert_eq!(MacroValue::Null.to_display_string(), "null");
    assert_eq!(MacroValue::Bool(true).to_display_string(), "true");
    assert_eq!(MacroValue::Int(42).to_display_string(), "42");
    assert_eq!(MacroValue::from_str("hi").to_display_string(), "hi");

    let arr = MacroValue::Array(Arc::new(vec![MacroValue::Int(1), MacroValue::Int(2)]));
    assert_eq!(arr.to_display_string(), "[1,2]");

    // Std.string spells an enum value by its constructor alone.
    let enumv = MacroValue::Enum(Arc::from("Color"), Arc::from("Red"), Arc::new(vec![]));
    assert_eq!(enumv.to_display_string(), "Red");

    let enumv_args = MacroValue::Enum(
        Arc::from("Option"),
        Arc::from("Some"),
        Arc::new(vec![MacroValue::Int(5), MacroValue::Int(6)]),
    );
    assert_eq!(enumv_args.to_display_string(), "Some(5,6)");
}

#[test]
fn test_macro_value_equality() {
    assert_eq!(MacroValue::Null, MacroValue::Null);
    assert_eq!(MacroValue::Bool(true), MacroValue::Bool(true));
    assert_ne!(MacroValue::Bool(true), MacroValue::Bool(false));
    assert_eq!(MacroValue::Int(42), MacroValue::Int(42));
    assert_ne!(MacroValue::Int(1), MacroValue::Int(2));
    assert_eq!(MacroValue::from_str("a"), MacroValue::from_str("a"));

    // Cross-type int/float equality
    assert_eq!(MacroValue::Int(5), MacroValue::Float(5.0));
    assert_eq!(MacroValue::Float(3.0), MacroValue::Int(3));
    assert_ne!(MacroValue::Int(5), MacroValue::Float(5.1));

    // Array equality
    assert_eq!(
        MacroValue::Array(Arc::new(vec![MacroValue::Int(1)])),
        MacroValue::Array(Arc::new(vec![MacroValue::Int(1)]))
    );
    assert_ne!(
        MacroValue::Array(Arc::new(vec![MacroValue::Int(1)])),
        MacroValue::Array(Arc::new(vec![MacroValue::Int(2)]))
    );
}

// ================================================================
// 2. ENVIRONMENT TESTS
// ================================================================

#[test]
fn test_environment_nested_scoping() {
    let mut env = Environment::new();
    assert_eq!(env.depth(), 0);

    env.define("x", MacroValue::Int(1));
    env.define("y", MacroValue::Int(2));

    // Inner scope shadows x but y is still visible
    env.push_scope();
    env.define("x", MacroValue::Int(10));
    env.define("z", MacroValue::Int(30));
    assert_eq!(env.get("x"), Some(MacroValue::Int(10)));
    assert_eq!(env.get("y"), Some(MacroValue::Int(2)));
    assert_eq!(env.get("z"), Some(MacroValue::Int(30)));
    assert_eq!(env.depth(), 1);

    // Pop inner scope
    env.pop_scope();
    assert_eq!(env.get("x"), Some(MacroValue::Int(1)));
    assert_eq!(env.get("y"), Some(MacroValue::Int(2)));
    assert_eq!(env.get("z"), None);
    assert_eq!(env.depth(), 0);
}

#[test]
fn test_environment_set_updates_outer_scope() {
    let mut env = Environment::new();
    env.define("counter", MacroValue::Int(0));

    env.push_scope();
    // set should find and update the outer scope's variable
    assert!(env.set("counter", MacroValue::Int(5)));
    assert_eq!(env.get("counter"), Some(MacroValue::Int(5)));

    env.pop_scope();
    // Outer scope should reflect the update
    assert_eq!(env.get("counter"), Some(MacroValue::Int(5)));
}

#[test]
fn test_environment_capture_all() {
    let mut env = Environment::new();
    env.define("a", MacroValue::Int(1));
    env.push_scope();
    env.define("a", MacroValue::Int(2)); // shadows
    env.define("b", MacroValue::Int(3));

    let captured = env.capture_all();
    // Inner value wins for shadowed variables
    assert_eq!(captured.get("a"), Some(&MacroValue::Int(2)));
    assert_eq!(captured.get("b"), Some(&MacroValue::Int(3)));
    assert_eq!(captured.len(), 2);
}

#[test]
fn test_environment_visible_names_sorted() {
    let mut env = Environment::new();
    env.define("z", MacroValue::Null);
    env.define("a", MacroValue::Null);
    env.push_scope();
    env.define("m", MacroValue::Null);

    let names = env.visible_names();
    assert_eq!(names, vec!["a", "m", "z"]);
}

// ================================================================
// 3. REGISTRY TESTS
// ================================================================

#[test]
fn test_registry_scan_and_lookup() {
    let mut registry = MacroRegistry::new();
    let source = r#"
package tools;

class MacroLib {
    public macro static function expand():Void {
        return;
    }

    public static function regular():Void {}
}
"#;
    let file = parse(source);
    registry
        .scan_and_register(&file, "tools.hx")
        .expect("scan should succeed");

    // Only the macro function should be registered
    assert_eq!(registry.macro_count(), 1);
    assert!(registry.is_macro("expand"));
    assert!(registry.is_macro("tools.MacroLib.expand"));
    assert!(!registry.is_macro("regular"));

    // Lookup by qualified name
    let def = registry
        .get_macro("tools.MacroLib.expand")
        .expect("should find by qualified name");
    assert_eq!(def.name, "expand");
    assert_eq!(def.qualified_name, "tools.MacroLib.expand");

    // Lookup by simple name
    let def2 = registry
        .find_macro_by_name("expand")
        .expect("should find by simple name");
    assert_eq!(def2.name, "expand");
}

#[test]
fn test_registry_build_macro_detection() {
    let mut registry = MacroRegistry::new();
    let source = r#"
@:build(MacroTools.addFields)
class MyClass {
    public var x:Int;
}
"#;
    let file = parse(source);
    registry
        .scan_and_register(&file, "test.hx")
        .expect("scan should succeed");

    let build_macros = registry.build_macros();
    assert_eq!(build_macros.len(), 1);
    assert_eq!(build_macros[0].target_class, "MyClass");
    assert!(build_macros[0].macro_name.contains("MacroTools"));
}

#[test]
fn test_registry_expansion_depth_and_circular_detection() {
    let mut registry = MacroRegistry::new();
    registry.set_max_depth(3);

    // Normal expansion tracking
    assert!(registry.enter_expansion("a").is_ok());
    assert_eq!(registry.expansion_depth(), 1);
    assert!(registry.enter_expansion("b").is_ok());
    assert_eq!(registry.expansion_depth(), 2);
    assert!(registry.enter_expansion("c").is_ok());
    assert_eq!(registry.expansion_depth(), 3);

    // Exceed depth
    let err = registry.enter_expansion("d").unwrap_err();
    assert!(matches!(
        err,
        MacroError::RecursionLimitExceeded {
            depth: 4,
            max_depth: 3,
            ..
        }
    ));

    // Exit all
    registry.exit_expansion("c");
    registry.exit_expansion("b");

    // Test circular detection: 'a' is still being expanded
    let err = registry.enter_expansion("a").unwrap_err();
    assert!(matches!(err, MacroError::CircularDependency { .. }));
    if let MacroError::CircularDependency { chain, .. } = err {
        assert_eq!(chain, vec!["a".to_string(), "a".to_string()]);
    }

    // Clean up
    registry.exit_expansion("a");
    assert_eq!(registry.expansion_depth(), 0);
}

// ================================================================
// 4. AST BRIDGE TESTS
// ================================================================

#[test]
fn test_ast_bridge_literal_round_trips() {
    // Int
    let val = expr_to_value(&int_expr(42)).unwrap();
    assert_eq!(val, MacroValue::Int(42));
    let back = value_to_expr(&val);
    assert!(matches!(back.kind, ExprKind::Int(42)));

    // Float
    let val = expr_to_value(&float_expr(1.23)).unwrap();
    if let MacroValue::Float(f) = val {
        assert!((f - 1.23).abs() < 1e-10);
    } else {
        panic!("expected Float");
    }

    // String
    let val = expr_to_value(&str_expr("hello")).unwrap();
    assert_eq!(val, MacroValue::from_str("hello"));
    let back = value_to_expr(&val);
    assert!(matches!(back.kind, ExprKind::String(ref s) if s == "hello"));

    // Bool
    let val = expr_to_value(&bool_expr(true)).unwrap();
    assert_eq!(val, MacroValue::Bool(true));

    // Null
    let val = expr_to_value(&null_expr()).unwrap();
    assert_eq!(val, MacroValue::Null);
}

#[test]
fn test_ast_bridge_array_round_trip() {
    let arr = Expr {
        kind: ExprKind::Array(vec![int_expr(1), int_expr(2), int_expr(3)]),
        span: default_span(),
    };
    let val = expr_to_value(&arr).unwrap();
    assert!(matches!(&val, MacroValue::Array(items) if items.len() == 3));

    let back = value_to_expr(&val);
    assert!(matches!(&back.kind, ExprKind::Array(items) if items.len() == 3));
}

#[test]
fn test_ast_bridge_negative_values() {
    // Negative int: should produce Unary(Neg, Int(5))
    let val = MacroValue::Int(-5);
    let expr = value_to_expr(&val);
    match &expr.kind {
        ExprKind::Unary {
            op: parser::UnaryOp::Neg,
            expr: inner,
        } => {
            assert!(matches!(inner.kind, ExprKind::Int(5)));
        }
        _ => panic!("expected unary neg for negative int"),
    }

    // Negative float
    let val = MacroValue::Float(-2.5);
    let expr = value_to_expr(&val);
    match &expr.kind {
        ExprKind::Unary {
            op: parser::UnaryOp::Neg,
            expr: inner,
        } => {
            assert!(matches!(inner.kind, ExprKind::Float(f) if (f - 2.5).abs() < 1e-10));
        }
        _ => panic!("expected unary neg for negative float"),
    }
}

#[test]
fn test_ast_bridge_non_literal_wraps_as_expr() {
    let val = expr_to_value(&ident_expr("someVar")).unwrap();
    assert!(matches!(val, MacroValue::Expr(_)));
}

#[test]
fn test_binary_operations_comprehensive() {
    let loc = unknown_loc();

    // Arithmetic
    assert_eq!(
        apply_binary_op(
            &BinaryOp::Add,
            &MacroValue::Int(3),
            &MacroValue::Int(4),
            loc
        )
        .unwrap(),
        MacroValue::Int(7)
    );
    assert_eq!(
        apply_binary_op(
            &BinaryOp::Sub,
            &MacroValue::Int(10),
            &MacroValue::Int(3),
            loc
        )
        .unwrap(),
        MacroValue::Int(7)
    );
    assert_eq!(
        apply_binary_op(
            &BinaryOp::Mul,
            &MacroValue::Int(6),
            &MacroValue::Int(7),
            loc
        )
        .unwrap(),
        MacroValue::Int(42)
    );
    assert_eq!(
        apply_binary_op(
            &BinaryOp::Div,
            &MacroValue::Int(20),
            &MacroValue::Int(4),
            loc
        )
        .unwrap(),
        MacroValue::Int(5)
    );
    assert_eq!(
        apply_binary_op(
            &BinaryOp::Mod,
            &MacroValue::Int(10),
            &MacroValue::Int(3),
            loc
        )
        .unwrap(),
        MacroValue::Int(1)
    );

    // Float arithmetic
    let result = apply_binary_op(
        &BinaryOp::Add,
        &MacroValue::Float(1.5),
        &MacroValue::Float(2.5),
        loc,
    )
    .unwrap();
    assert_eq!(result, MacroValue::Float(4.0));

    // Mixed int/float
    let result = apply_binary_op(
        &BinaryOp::Add,
        &MacroValue::Int(1),
        &MacroValue::Float(2.5),
        loc,
    )
    .unwrap();
    assert_eq!(result, MacroValue::Float(3.5));

    // String concatenation
    assert_eq!(
        apply_binary_op(
            &BinaryOp::Add,
            &MacroValue::from_str("hello"),
            &MacroValue::from_str(" world"),
            loc
        )
        .unwrap(),
        MacroValue::from_str("hello world")
    );

    // String + non-string
    assert_eq!(
        apply_binary_op(
            &BinaryOp::Add,
            &MacroValue::from_str("count: "),
            &MacroValue::Int(42),
            loc
        )
        .unwrap(),
        MacroValue::from_str("count: 42")
    );

    // Comparison
    assert_eq!(
        apply_binary_op(&BinaryOp::Eq, &MacroValue::Int(1), &MacroValue::Int(1), loc).unwrap(),
        MacroValue::Bool(true)
    );
    assert_eq!(
        apply_binary_op(
            &BinaryOp::NotEq,
            &MacroValue::Int(1),
            &MacroValue::Int(2),
            loc
        )
        .unwrap(),
        MacroValue::Bool(true)
    );
    assert_eq!(
        apply_binary_op(&BinaryOp::Lt, &MacroValue::Int(1), &MacroValue::Int(2), loc).unwrap(),
        MacroValue::Bool(true)
    );
    assert_eq!(
        apply_binary_op(&BinaryOp::Ge, &MacroValue::Int(5), &MacroValue::Int(3), loc).unwrap(),
        MacroValue::Bool(true)
    );

    // Logical
    assert_eq!(
        apply_binary_op(
            &BinaryOp::And,
            &MacroValue::Bool(true),
            &MacroValue::Bool(false),
            loc
        )
        .unwrap(),
        MacroValue::Bool(false)
    );
    assert_eq!(
        apply_binary_op(
            &BinaryOp::Or,
            &MacroValue::Bool(false),
            &MacroValue::Bool(true),
            loc
        )
        .unwrap(),
        MacroValue::Bool(true)
    );

    // Bitwise
    assert_eq!(
        apply_binary_op(
            &BinaryOp::BitAnd,
            &MacroValue::Int(0xFF),
            &MacroValue::Int(0x0F),
            loc
        )
        .unwrap(),
        MacroValue::Int(0x0F)
    );
    assert_eq!(
        apply_binary_op(
            &BinaryOp::BitOr,
            &MacroValue::Int(0xF0),
            &MacroValue::Int(0x0F),
            loc
        )
        .unwrap(),
        MacroValue::Int(0xFF)
    );
    assert_eq!(
        apply_binary_op(
            &BinaryOp::Shl,
            &MacroValue::Int(1),
            &MacroValue::Int(4),
            loc
        )
        .unwrap(),
        MacroValue::Int(16)
    );

    // Null coalescing
    assert_eq!(
        apply_binary_op(
            &BinaryOp::NullCoal,
            &MacroValue::Null,
            &MacroValue::Int(99),
            loc
        )
        .unwrap(),
        MacroValue::Int(99)
    );
    assert_eq!(
        apply_binary_op(
            &BinaryOp::NullCoal,
            &MacroValue::Int(1),
            &MacroValue::Int(99),
            loc
        )
        .unwrap(),
        MacroValue::Int(1)
    );

    // Range
    let range = apply_binary_op(
        &BinaryOp::Range,
        &MacroValue::Int(0),
        &MacroValue::Int(5),
        loc,
    )
    .unwrap();
    let mut interpreter = MacroInterpreter::new(MacroRegistry::new());
    interpreter.define_variable("range", range);
    let probe = parse(
        "class Probe {static function main() {var sum=0; for (value in range) sum+=value; return sum;}}",
    );
    let parser::TypeDeclaration::Class(class) = &probe.declarations[0] else {
        panic!()
    };
    let parser::ClassFieldKind::Function(function) = &class.fields[0].kind else {
        panic!()
    };
    assert!(
        matches!(interpreter.eval_expr(function.body.as_ref().unwrap()),Err(MacroError::Return {value:Some(value)}) if *value==MacroValue::Int(10))
    );

    // Division by zero
    let err = apply_binary_op(
        &BinaryOp::Div,
        &MacroValue::Int(1),
        &MacroValue::Int(0),
        loc,
    )
    .unwrap_err();
    assert!(matches!(err, MacroError::DivisionByZero { .. }));

    // Modulo by zero
    let err = apply_binary_op(
        &BinaryOp::Mod,
        &MacroValue::Int(1),
        &MacroValue::Int(0),
        loc,
    )
    .unwrap_err();
    assert!(matches!(err, MacroError::DivisionByZero { .. }));
}

// ================================================================
// 5. INTERPRETER TESTS
// ================================================================

#[test]
fn test_interpreter_arithmetic() {
    let registry = MacroRegistry::new();
    let mut interp = MacroInterpreter::new(registry);

    // Simple addition: 2 + 3
    let expr = Expr {
        kind: ExprKind::Binary {
            op: BinaryOp::Add,
            left: Box::new(int_expr(2)),
            right: Box::new(int_expr(3)),
        },
        span: default_span(),
    };
    let result = interp.eval_expr(&expr).unwrap();
    assert_eq!(result, MacroValue::Int(5));
}

#[test]
fn test_interpreter_string_concat() {
    let registry = MacroRegistry::new();
    let mut interp = MacroInterpreter::new(registry);

    let expr = Expr {
        kind: ExprKind::Binary {
            op: BinaryOp::Add,
            left: Box::new(str_expr("hello")),
            right: Box::new(str_expr(" world")),
        },
        span: default_span(),
    };
    let result = interp.eval_expr(&expr).unwrap();
    assert_eq!(result, MacroValue::from_str("hello world"));
}

#[test]
fn test_interpreter_nested_arithmetic() {
    let registry = MacroRegistry::new();
    let mut interp = MacroInterpreter::new(registry);

    // (2 + 3) * 4
    let expr = Expr {
        kind: ExprKind::Binary {
            op: BinaryOp::Mul,
            left: Box::new(Expr {
                kind: ExprKind::Binary {
                    op: BinaryOp::Add,
                    left: Box::new(int_expr(2)),
                    right: Box::new(int_expr(3)),
                },
                span: default_span(),
            }),
            right: Box::new(int_expr(4)),
        },
        span: default_span(),
    };
    let result = interp.eval_expr(&expr).unwrap();
    assert_eq!(result, MacroValue::Int(20));
}

#[test]
fn test_interpreter_comparison() {
    let registry = MacroRegistry::new();
    let mut interp = MacroInterpreter::new(registry);

    // 5 > 3
    let expr = Expr {
        kind: ExprKind::Binary {
            op: BinaryOp::Gt,
            left: Box::new(int_expr(5)),
            right: Box::new(int_expr(3)),
        },
        span: default_span(),
    };
    let result = interp.eval_expr(&expr).unwrap();
    assert_eq!(result, MacroValue::Bool(true));
}

#[test]
fn test_interpreter_unary_negation() {
    let registry = MacroRegistry::new();
    let mut interp = MacroInterpreter::new(registry);

    // -42
    let expr = Expr {
        kind: ExprKind::Unary {
            op: parser::UnaryOp::Neg,
            expr: Box::new(int_expr(42)),
        },
        span: default_span(),
    };
    let result = interp.eval_expr(&expr).unwrap();
    assert_eq!(result, MacroValue::Int(-42));
}

#[test]
fn test_interpreter_unary_not() {
    let registry = MacroRegistry::new();
    let mut interp = MacroInterpreter::new(registry);

    // !true
    let expr = Expr {
        kind: ExprKind::Unary {
            op: parser::UnaryOp::Not,
            expr: Box::new(bool_expr(true)),
        },
        span: default_span(),
    };
    let result = interp.eval_expr(&expr).unwrap();
    assert_eq!(result, MacroValue::Bool(false));
}

#[test]
fn test_interpreter_array_construction() {
    let registry = MacroRegistry::new();
    let mut interp = MacroInterpreter::new(registry);

    let expr = Expr {
        kind: ExprKind::Array(vec![int_expr(10), int_expr(20), int_expr(30)]),
        span: default_span(),
    };
    let result = interp.eval_expr(&expr).unwrap();
    assert_eq!(
        result,
        MacroValue::Array(Arc::new(vec![
            MacroValue::Int(10),
            MacroValue::Int(20),
            MacroValue::Int(30)
        ]))
    );
}

#[test]
fn test_interpreter_null_and_bool_literals() {
    let registry = MacroRegistry::new();
    let mut interp = MacroInterpreter::new(registry);

    assert_eq!(interp.eval_expr(&null_expr()).unwrap(), MacroValue::Null);
    assert_eq!(
        interp.eval_expr(&bool_expr(true)).unwrap(),
        MacroValue::Bool(true)
    );
    assert_eq!(
        interp.eval_expr(&bool_expr(false)).unwrap(),
        MacroValue::Bool(false)
    );
}

#[test]
fn test_interpreter_variable_access() {
    let registry = MacroRegistry::new();
    let mut interp = MacroInterpreter::new(registry);

    interp.env_mut().define("myVar", MacroValue::Int(100));
    let result = interp.eval_expr(&ident_expr("myVar")).unwrap();
    assert_eq!(result, MacroValue::Int(100));
}

#[test]
fn test_interpreter_undefined_variable_error() {
    let registry = MacroRegistry::new();
    let mut interp = MacroInterpreter::new(registry);

    let result = interp.eval_expr(&ident_expr("undefined_var"));
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        matches!(err, MacroError::UndefinedVariable { .. })
            || matches!(err, MacroError::RuntimeError { .. })
    );
}

// ================================================================
// 6. REIFICATION TESTS
// ================================================================

#[test]
fn test_reification_simple_expr() {
    let env = Environment::new();

    // Reify a simple integer literal
    let expr = int_expr(42);
    let result = ReificationEngine::reify_expr(&expr, &env).unwrap();

    // Should produce MacroValue::Expr containing the int literal
    match result {
        MacroValue::Expr(e) => {
            assert!(matches!(e.kind, ExprKind::Int(42)));
        }
        _ => panic!("expected Expr value from reification"),
    }
}

#[test]
fn test_reification_string_expr() {
    let env = Environment::new();
    let expr = str_expr("hello");
    let result = ReificationEngine::reify_expr(&expr, &env).unwrap();

    match result {
        MacroValue::Expr(e) => {
            assert!(matches!(e.kind, ExprKind::String(ref s) if s == "hello"));
        }
        _ => panic!("expected Expr value"),
    }
}

#[test]
fn test_reification_preserves_structure() {
    let env = Environment::new();

    // Reify a binary expression: 1 + 2
    let expr = Expr {
        kind: ExprKind::Binary {
            op: BinaryOp::Add,
            left: Box::new(int_expr(1)),
            right: Box::new(int_expr(2)),
        },
        span: default_span(),
    };
    let result = ReificationEngine::reify_expr(&expr, &env).unwrap();

    match result {
        MacroValue::Expr(e) => {
            assert!(matches!(e.kind, ExprKind::Binary { .. }));
        }
        _ => panic!("expected Expr value"),
    }
}

// ================================================================
// 7. EXPANDER TESTS
// ================================================================

#[test]
fn test_expander_no_macros() {
    let source = r#"
class Simple {
    public var x:Int = 5;
    public function new() {}
}
"#;
    let file = parse(source);
    let result = expand_macros(file);

    assert_eq!(result.expansions_count, 0);
    assert!(result.diagnostics.is_empty());
}

#[test]
fn test_expander_with_macro_function() {
    let source = r#"
class Tools {
    macro static function double(n:Int):Int {
        return n * 2;
    }
}
"#;
    let file = parse(source);
    let result = expand_macros(file);

    // The macro function should be recognized (even if not called)
    assert!(result.diagnostics.is_empty());
}

#[test]
fn test_expander_preserves_non_macro_code() {
    let source = r#"
class MyClass {
    public var x:Int;
    public var y:String;

    public function new(x:Int, y:String) {
        this.x = x;
        this.y = y;
    }

    public function getSum():Int {
        return x + 1;
    }
}
"#;
    let file = parse(source);
    let result = expand_macros(file);

    // File should pass through unchanged
    assert_eq!(result.expansions_count, 0);
    assert_eq!(result.file.declarations.len(), 1);
}

#[test]
fn test_expander_with_registry() {
    let source = r#"
class Macros {
    macro static function myMacro():Void {
        trace("hello");
    }
}

class User {
    public var name:String;
}
"#;
    let file = parse(source);

    // Pre-scan to verify macro registration works
    let mut registry = MacroRegistry::new();
    let scan_file = parse(source);
    registry
        .scan_and_register(&scan_file, "test.hx")
        .expect("scan should succeed");
    assert!(registry.is_macro("myMacro"));

    // Now expand with a fresh registry (consumed by value)
    let result = expand_macros_with_registry(file, MacroRegistry::new());
    assert!(result.diagnostics.is_empty());
}

#[test]
fn test_expander_multiple_classes() {
    let source = r#"
class A {
    public var x:Int = 1;
}

class B {
    public var y:String = "hello";
}

class C {
    public var z:Bool = true;
}
"#;
    let file = parse(source);
    let result = expand_macros(file);
    assert_eq!(result.file.declarations.len(), 3);
}

// ================================================================
// 8. CONTEXT API TESTS
// ================================================================

#[test]
fn test_context_diagnostics() {
    let mut ctx = MacroContext::new();
    let _err = ctx.error("test error", unknown_loc());
    ctx.warning("test warning", unknown_loc());

    assert_eq!(ctx.diagnostics.len(), 2);
    assert_eq!(ctx.diagnostics[0].severity, MacroSeverity::Error);
    assert_eq!(ctx.diagnostics[0].message, "test error");
    assert_eq!(ctx.diagnostics[1].severity, MacroSeverity::Warning);
    assert_eq!(ctx.diagnostics[1].message, "test warning");
}

#[test]
fn test_context_defines() {
    let mut ctx = MacroContext::new();
    ctx.defines.insert("debug".to_string(), "true".to_string());
    ctx.defines.insert("version".to_string(), "1.0".to_string());

    assert!(ctx.defines.contains_key("debug"));
    assert_eq!(ctx.defines.get("version"), Some(&"1.0".to_string()));
    assert!(!ctx.defines.contains_key("release"));
}

#[test]
fn test_context_build_class() {
    let named_type = |name: &str| parser::Type::Path {
        path: parser::TypePath {
            package: vec![],
            name: name.into(),
            sub: None,
        },
        params: vec![],
        span: default_span(),
    };
    let mut ctx = MacroContext::new();
    ctx.build_class = Some(BuildClassContext {
        class_name: "MyClass".to_string(),
        qualified_name: "com.example.MyClass".to_string(),
        symbol_id: None,
        fields: vec![
            BuildField {
                name: "x".to_string(),
                kind: BuildFieldKind::Var {
                    type_hint: Some(named_type("Int")),
                    expr: None,
                },
                access: vec![FieldAccess::Public],
                pos: unknown_loc(),
                doc: None,
                meta: vec![],
            },
            BuildField {
                name: "y".to_string(),
                kind: BuildFieldKind::Var {
                    type_hint: Some(named_type("String")),
                    expr: None,
                },
                access: vec![FieldAccess::Public],
                pos: unknown_loc(),
                doc: None,
                meta: vec![],
            },
        ],
        pack: vec!["com".to_string(), "example".to_string()],
    });

    let build = ctx.build_class.as_ref().unwrap();
    assert_eq!(build.class_name, "MyClass");
    assert_eq!(build.fields.len(), 2);
    assert_eq!(build.fields[0].name, "x");
    assert_eq!(build.fields[1].name, "y");
}

#[test]
fn test_context_defined_types() {
    let mut ctx = MacroContext::new();
    ctx.defined_types.push(DefinedType {
        name: "GeneratedClass".to_string(),
        pack: vec!["gen".to_string()],
        kind: DefinedTypeKind::Class,
        fields: vec![],
        field_values: vec![],
        type_params: vec![],
        pos: unknown_loc(),
    });

    assert_eq!(ctx.defined_types.len(), 1);
    assert_eq!(ctx.defined_types[0].name, "GeneratedClass");
    assert_eq!(ctx.defined_types[0].pack, vec!["gen".to_string()]);
}

// ================================================================
// 9. ERROR HANDLING TESTS
// ================================================================

#[test]
fn test_error_codes() {
    let loc = unknown_loc();

    assert_eq!(
        MacroError::UndefinedMacro {
            name: "x".to_string(),
            location: loc
        }
        .error_code(),
        "E0701"
    );
    assert_eq!(
        MacroError::ArgumentCountMismatch {
            macro_name: "x".to_string(),
            expected: 2,
            found: 1,
            location: loc
        }
        .error_code(),
        "E0702"
    );
    assert_eq!(
        MacroError::TypeError {
            message: "".to_string(),
            location: loc
        }
        .error_code(),
        "E0703"
    );
    assert_eq!(
        MacroError::RuntimeError {
            message: "".to_string(),
            location: loc
        }
        .error_code(),
        "E0704"
    );
    assert_eq!(
        MacroError::RecursionLimitExceeded {
            macro_name: "x".to_string(),
            depth: 300,
            max_depth: 256,
            location: loc
        }
        .error_code(),
        "E0705"
    );
    assert_eq!(
        MacroError::CircularDependency {
            chain: vec!["a".to_string(), "b".to_string(), "a".to_string()],
            location: loc
        }
        .error_code(),
        "E0706"
    );
    assert_eq!(
        MacroError::ReificationError {
            message: "".to_string(),
            location: loc
        }
        .error_code(),
        "E0707"
    );
    assert_eq!(
        MacroError::InvalidDefinition {
            message: "".to_string(),
            location: loc
        }
        .error_code(),
        "E0708"
    );
    assert_eq!(
        MacroError::ContextError {
            method: "getType".to_string(),
            message: "not found".to_string(),
            location: loc
        }
        .error_code(),
        "E0709"
    );
    assert_eq!(
        MacroError::UndefinedVariable {
            name: "x".to_string(),
            location: loc
        }
        .error_code(),
        "E0710"
    );
    assert_eq!(
        MacroError::UnsupportedOperation {
            operation: "".to_string(),
            location: loc
        }
        .error_code(),
        "E0711"
    );
    assert_eq!(
        MacroError::DivisionByZero { location: loc }.error_code(),
        "E0712"
    );
}

#[test]
fn test_error_display_messages() {
    let loc = unknown_loc();

    let err = MacroError::UndefinedMacro {
        name: "doStuff".to_string(),
        location: loc,
    };
    assert_eq!(format!("{}", err), "undefined macro: 'doStuff'");

    let err = MacroError::ArgumentCountMismatch {
        macro_name: "myMacro".to_string(),
        expected: 2,
        found: 0,
        location: loc,
    };
    assert_eq!(
        format!("{}", err),
        "macro 'myMacro' expects 2 argument(s), found 0"
    );

    let err = MacroError::CircularDependency {
        chain: vec!["a".to_string(), "b".to_string(), "a".to_string()],
        location: loc,
    };
    assert_eq!(format!("{}", err), "circular macro dependency: a -> b -> a");

    let err = MacroError::DivisionByZero { location: loc };
    assert_eq!(format!("{}", err), "division by zero in macro evaluation");
}

#[test]
fn test_error_control_flow_detection() {
    assert!(MacroError::Return { value: None }.is_control_flow());
    assert!(MacroError::Break.is_control_flow());
    assert!(MacroError::Continue.is_control_flow());

    assert!(
        !MacroError::DivisionByZero {
            location: unknown_loc()
        }
        .is_control_flow()
    );
    assert!(
        !MacroError::TypeError {
            message: "".to_string(),
            location: unknown_loc()
        }
        .is_control_flow()
    );
}

#[test]
fn test_error_to_compilation_error() {
    let err = MacroError::UndefinedMacro {
        name: "myMacro".to_string(),
        location: unknown_loc(),
    };
    let comp_err = err.to_compilation_error();
    assert!(comp_err.message.contains("E0701"));
    assert!(comp_err.message.contains("myMacro"));
    assert!(comp_err.suggestion.is_some());
}

#[test]
fn test_diagnostic_creation() {
    let loc = unknown_loc();

    let diag = MacroDiagnostic::error("something failed", loc);
    assert_eq!(diag.severity, MacroSeverity::Error);
    assert_eq!(diag.message, "something failed");

    let diag = MacroDiagnostic::warning("be careful", loc).with_suggestion("try this instead");
    assert_eq!(diag.severity, MacroSeverity::Warning);
    assert_eq!(diag.suggestion, Some("try this instead".to_string()));

    let diag = MacroDiagnostic::info("FYI", loc);
    assert_eq!(diag.severity, MacroSeverity::Info);
}

// ================================================================
// 10. BUILD MACRO TESTS
// ================================================================

#[test]
fn test_build_macro_field_extraction() {
    let source = r#"
@:build(MacroTools.addFields)
class MyEntity {
    public var name:String;
    private var id:Int;
    public function process():Void {}
}
"#;
    let file = parse(source);
    let mut registry = MacroRegistry::new();
    registry
        .scan_and_register(&file, "test.hx")
        .expect("scan should succeed");

    // Build macros should be detected
    let build_macros = registry.build_macros();
    assert_eq!(build_macros.len(), 1);
    assert_eq!(build_macros[0].target_class, "MyEntity");
}

#[test]
fn test_build_macro_multiple_on_class() {
    let source = r#"
@:build(Macros.addToString)
@:build(Macros.addSerialize)
@:build(Macros.addValidation)
class User {
    public var name:String;
    public var email:String;
}
"#;
    let file = parse(source);
    let mut registry = MacroRegistry::new();
    registry
        .scan_and_register(&file, "test.hx")
        .expect("scan should succeed");

    let build_macros = registry.build_macros();
    assert_eq!(build_macros.len(), 3);
    // All target the same class
    for bm in build_macros {
        assert_eq!(bm.target_class, "User");
    }
}

// ================================================================
// 11. INTEGRATION: FULL PIPELINE TESTS
// ================================================================

#[test]
fn test_full_pipeline_macro_scan_and_expand() {
    let source = r#"
package test;

class MacroUtils {
    macro static function constValue():Int {
        return 42;
    }

    static function normalMethod():Void {
        trace("not a macro");
    }
}

class App {
    public var value:Int;

    public function new() {
        this.value = 0;
    }
}
"#;
    // Verify macro registration via scanning
    let mut registry = MacroRegistry::new();
    let scan_file = parse(source);
    registry
        .scan_and_register(&scan_file, "test.hx")
        .expect("scan should succeed");
    assert_eq!(registry.macro_count(), 1);
    assert!(registry.is_macro("constValue"));
    assert!(!registry.is_macro("normalMethod"));

    // Expand macros (registry consumed by value)
    let file = parse(source);
    let result = expand_macros_with_registry(file, registry);

    // No expansion errors
    assert!(result.diagnostics.is_empty());

    // Both classes should be preserved
    assert_eq!(result.file.declarations.len(), 2);
}

#[test]
fn test_full_pipeline_with_package() {
    let source = r#"
package com.example;

class Lib {
    macro static function greet():String {
        return "hello";
    }
}
"#;
    // Verify qualified name registration via scanning
    let mut registry = MacroRegistry::new();
    let file = parse(source);
    registry
        .scan_and_register(&file, "test.hx")
        .expect("scan should succeed");

    // Should be registered with full qualified name
    assert!(registry.is_macro("com.example.Lib.greet"));
    assert!(registry.is_macro("greet")); // Also findable by simple name

    // Expand should also work
    let file2 = parse(source);
    let result = expand_macros(file2);
    assert!(result.diagnostics.is_empty());
}

#[test]
fn test_full_pipeline_multiple_macro_classes() {
    let source = r#"
class MacroA {
    macro static function a():Int { return 1; }
}

class MacroB {
    macro static function b():Int { return 2; }
}

class MacroC {
    macro static function c():Int { return 3; }
    macro static function d():Int { return 4; }
}
"#;
    // Verify registration via scanning
    let mut registry = MacroRegistry::new();
    let file = parse(source);
    registry
        .scan_and_register(&file, "test.hx")
        .expect("scan should succeed");

    assert_eq!(registry.macro_count(), 4);
    assert!(registry.is_macro("a"));
    assert!(registry.is_macro("b"));
    assert!(registry.is_macro("c"));
    assert!(registry.is_macro("d"));

    // Expansion should work cleanly
    let file2 = parse(source);
    let result = expand_macros(file2);
    assert!(result.diagnostics.is_empty());
}

#[test]
fn test_full_pipeline_empty_file() {
    let file = parse("class Empty {}");
    let result = expand_macros(file);

    assert_eq!(result.expansions_count, 0);
    assert!(result.diagnostics.is_empty());
}

#[test]
fn diagnostic_context_does_not_replace_the_live_typer() {
    for body in [
        r#"
        var expected = haxe.macro.Context.getExpectedType();
        return macro 1;
        "#,
        r#"
        try {
            haxe.macro.Context.typeof(macro 1);
        } catch (error:Dynamic) {
            return macro $v{error.message};
        }
        return macro "";
        "#,
    ] {
        let source = format!(
            "class Main {{
                macro static function probe() {{ {body} }}
                static function main() {{ var value = probe(); }}
            }}"
        );
        let result = expand_macros(parse(&source));
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        assert_eq!(result.expansions_count, 0);
        assert!(!result.deferred.is_empty(), "{body}");
        // Removing macro declarations can trigger another expansion walk.
        // Each deferred entry must still point to the original call site.
        for call in result.deferred {
            assert_eq!(call.name, "probe");
            assert_eq!(&source[call.span.start..call.span.end], "probe()");
        }
    }
}

#[test]
fn typed_macro_helpers_convert_abstracts_across_closures_and_inheritance() {
    let source = r#"
import haxe.macro.Expr;
typedef TemplateData = {text:String};
@:forward abstract Template(TemplateData) from TemplateData {
    @:from static function fromExpr(e:Expr):Template {
        switch e { case macro @:markup $body: e = body; case _: }
        return switch e.expr {
            case EConst(CString(text)): {text:text};
            case _: throw "a template must contain text";
        }
    }
}
class Base {
    var initial:Int;
    public function new(source:Template) initial = source.text.length;
    public function measure(source:Template):Int return initial + source.text.length;
}
class Worker extends Base { public function new(source:Template) {super(source);} }
class Builder {
    public static macro function make(template:Expr):Expr {
        function local(source:Template):Int {
            var worker = new Worker(template);
            return source.text.length + worker.measure(template);
        }
        return macro $v{local(template)};
    }
}
class Main { static function main() return Builder.make(<div/>); }
"#;
    let file = parse(source);
    let mut classes = class_registry::ClassRegistry::new();
    classes.register_file(&file);
    let mut expander = MacroExpander::with_class_registry(classes);
    let result = expander.expand_file(file);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert!(
        result
            .expansion_origins
            .iter()
            .any(|origin| origin.expanded_text.as_deref() == Some("18")),
        "{:?}",
        result.expansion_origins
    );
}

#[test]
fn quoted_loop_bindings_splice_their_identifier() {
    let source = r#"
class Builder {
    public static macro function make() {
        var name = "piece";
        return macro for ($i{name} in [1, 2]) $i{name};
    }
}
class Main { static function main() Builder.make(); }
"#;
    let result = expander::expand_macros(parse(source));
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    let expansion = result.expansion_origins[0].expanded_text.as_ref().unwrap();
    assert!(
        expansion.contains("for (piece in") && expansion.contains(") piece"),
        "{expansion}"
    );
}

#[test]
fn generated_constructor_positions_follow_macro_arguments() {
    let source = r#"
import haxe.macro.Expr;
import haxe.macro.Context;
class Builder {
    public static macro function make(e:Expr):Expr {
        var position = Context.getPosInfos(e.pos);
        if (position.file != "test.hx") throw "lost caller filename";
        var tag = Context.makePosition({file:position.file, min:position.min + 1, max:position.min + 10});
        return macro @:pos(tag) new Widget();
    }
}
class Main { static function main() return Builder.make(<left-card/>); }
"#;
    let file = parse(source);
    let mut classes = class_registry::ClassRegistry::new();
    classes.register_file(&file);
    let result = MacroExpander::with_class_registry(classes).expand_file(file);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    let origin = &result.expansion_origins[0];
    assert_eq!(
        origin.expanded_span.start,
        source.find("left-card").unwrap()
    );
    assert_eq!(origin.expanded_span.end, origin.expanded_span.start + 9);
    assert!(
        origin
            .expanded_text
            .as_ref()
            .unwrap()
            .contains("new Widget")
    );
}

#[test]
fn macro_methods_isolate_locals_and_preserve_shared_receivers() {
    let source = r#"
class Counter {
    var pos:Int = 1;
    public function new() {}
    public function bump():Void pos++;
    public function read():Int return pos;
    public function captured():Void->Int return () -> pos;
}
class Tools { public static function mutate(box:{n:Int}):Void box.n += 1; }
class Builder {
    public static macro function make() {
        var counter = new Counter();
        var pos = 100;
        counter.bump();
        var read = counter.captured();
        var box = {n:1};
        Tools.mutate(box);
        return macro $v{pos + counter.read() + read() + box.n};
    }
}
class Main { static function main() return Builder.make(); }
"#;
    let file = parse(source);
    let mut classes = class_registry::ClassRegistry::new();
    classes.register_file(&file);
    let result = MacroExpander::with_class_registry(classes).expand_file(file);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(
        result.expansion_origins[0].expanded_text.as_deref(),
        Some("106")
    );
}

#[test]
fn macro_regex_matches_and_captures_without_enum_pattern_dispatch() {
    let source = r#"
class Builder {
    public static macro function make() {
        var lower = ~/[A-Z]/;
        if (lower.match("div")) throw "lowercase tag rejected";
        var capture = new EReg("([a-z]+)-([0-9]+)", "");
        if (!capture.match("<menu-42/>")) throw "missing match";
        var position = capture.matchedPos();
        return macro $v{capture.matched(1) + ":" + capture.matched(2) + ":" + position.pos + ":" + position.len};
    }
}
class Main { static function main() return Builder.make(); }
"#;
    let result = expander::expand_macros(parse(source));
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(
        result.expansion_origins[0].expanded_text.as_deref(),
        Some("\"menu:42:1:7\"")
    );
}

#[test]
fn macro_parent_classes_resolve_through_enclosing_packages() {
    let parent = parser::parse_haxe_file("outer/Base.hx", "package outer; class Base { var value:Int; public function new(n:Int) value = n; public function read():Int return value; }", false).unwrap();
    let child = parser::parse_haxe_file(
        "outer/inner/Child.hx",
        "package outer.inner; class Child extends Base { public function new() super(7); }",
        false,
    )
    .unwrap();
    let file = parse(
        "class Builder { public static macro function make() return macro $v{new outer.inner.Child().read()}; } class Main { static function main() return Builder.make(); }",
    );
    let mut classes = class_registry::ClassRegistry::new();
    classes.register_files(&[parent, child]);
    classes.register_file(&file);
    let result = MacroExpander::with_class_registry(classes).expand_file(file);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(
        result.expansion_origins[0].expanded_text.as_deref(),
        Some("7")
    );
}

#[test]
fn missing_macro_typer_suspends_before_catch_fallback() {
    let source = r#"
import haxe.macro.Context;
class Builder {
    public static macro function make() {
        var type = try Context.getType("Widget") catch (_:Dynamic) null;
        if (type == null) throw "unknown widget";
        return macro new Widget();
    }
}
class Main { static function main() return Builder.make(); }
"#;
    let result = expander::expand_macros(parse(source));
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(result.deferred.len(), 1);
}

#[test]
fn generic_abstract_operators_select_the_matching_operand_order() {
    let source = r#"
abstract Continue(Dynamic) {
    @:op(a + b) static function then<A>(e:Continue, a:A):A return a;
    @:op(a + b) static function before<A>(a:A, e:Continue):A return a;
}
class Builder {
    public static macro function make() {
        var next:Continue = null;
        return macro $v{(next + "left") + ("right" + next)};
    }
}
class Main { static function main() return Builder.make(); }
"#;
    let file = parse(source);
    let mut classes = class_registry::ClassRegistry::new();
    classes.register_file(&file);
    let result = MacroExpander::with_class_registry(classes).expand_file(file);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(
        result.expansion_origins[0].expanded_text.as_deref(),
        Some("\"leftright\"")
    );
}

#[test]
fn macro_position_values_admit_declared_extension_helpers() {
    let source = r#"
using PositionTools;
class PositionTools {
    public static function sanitize(p:haxe.macro.Expr.Position) return p;
}
class Builder {
    public static macro function make(e:haxe.macro.Expr) return macro $v{haxe.macro.Context.getPosInfos(e.pos.sanitize()).min};
}
class Main { static function main() return Builder.make(7); }
"#;
    let file = parse(source);
    let mut classes = class_registry::ClassRegistry::new();
    classes.register_file(&file);
    let result = MacroExpander::with_class_registry(classes).expand_file(file);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(
        result.expansion_origins[0].expanded_text.as_deref(),
        Some(source.rfind('7').unwrap().to_string().as_str())
    );
}

#[test]
fn macro_typed_catches_match_the_thrown_class() {
    let source = r#"
class Case { public function new() {} }
class Problem { public function new() {} }
class Builder {
    public static macro function make() {
        var answer = try { throw new Problem(); 0; } catch (_:Case) 1 catch (_:Problem) 2;
        return macro $v{answer};
    }
}
class Main { static function main() return Builder.make(); }
"#;
    let file = parse(source);
    let mut classes = class_registry::ClassRegistry::new();
    classes.register_file(&file);
    let result = MacroExpander::with_class_registry(classes).expand_file(file);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(
        result.expansion_origins[0].expanded_text.as_deref(),
        Some("2")
    );
}

#[test]
fn abstract_inequality_negates_declared_equality() {
    let source = r#"
typedef Data = {value:String};
@:forward abstract Slice(Data) from Data {
    @:op(a == b) static function equals(a:Slice, b:String):Bool return a.value == b;
}
class Builder {
    public static macro function make() {
        var slice:Slice = {value:"menu"};
        return macro $v{(slice == "menu") && !(slice != "menu") && slice != "other"};
    }
}
class Main { static function main() return Builder.make(); }
"#;
    let file = parse(source);
    let mut classes = class_registry::ClassRegistry::new();
    classes.register_file(&file);
    let result = MacroExpander::with_class_registry(classes).expand_file(file);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(
        result.expansion_origins[0].expanded_text.as_deref(),
        Some("true")
    );
}

#[test]
fn macro_optional_enum_payloads_and_structural_alternatives_match() {
    let source = r#"
enum OptionalValue { Value(text:String, ?kind:String); }
class Builder {
    public static macro function make() {
        var expr = EConst(CString("class"));
        var text = switch expr { case EConst(CString(s, _)): s; case _: throw "lost optional Constant payload"; }
        var optional = Value(text);
        var valid = switch optional { case Value("class", null): true; case _: false; }
        var tag = {name:"Signal"};
        var reactive = switch tag { case {name:"Signal" | "Computed"}: true; case _: false; }
        return macro $v{valid && reactive};
    }
}
class Main { static function main() return Builder.make(); }
"#;
    let file = parse(source);
    let mut classes = class_registry::ClassRegistry::new();
    classes.register_file(&file);
    let result = MacroExpander::with_class_registry(classes).expand_file(file);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(
        result.expansion_origins[0].expanded_text.as_deref(),
        Some("true")
    );
}

#[test]
fn quoted_splices_and_comprehensions_keep_lexical_captures() {
    for (body, expected) in [
        (
            r#"var style = macro ashui.types.Style;
            function keyword(type:String, value:String) return () -> macro $style.$type.$value;
            var make = keyword("Display", "Flex"); return make();"#,
            "ashui.types.Style.Display.Flex",
        ),
        (
            r#"function scale(value:Int) return () -> [for (i in [1, 2]) if (i > 0) i * value];
            var make = scale(3); return macro $v{make()[1]};"#,
            "6",
        ),
        (
            r#"var source = macro null;
            function positioned() return () -> macro @:pos(source.pos) 1;
            var make = positioned(); return make();"#,
            "1",
        ),
    ] {
        let file = parse(&format!(
            "class Builder {{ public static macro function make() {{ {body} }} }} class Main {{ static function main() return Builder.make(); }}"
        ));
        let mut classes = class_registry::ClassRegistry::new();
        classes.register_file(&file);
        let result = MacroExpander::with_class_registry(classes).expand_file(file);
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        assert_eq!(
            result
                .expansion_origins
                .last()
                .and_then(|origin| origin.expanded_text.as_deref()),
            Some(expected)
        );
    }
}

#[test]
fn whole_object_pattern_alternatives_bind_the_selected_payload() {
    let source = r#"
enum Kind { Text(value:String); Wrap(child:Dynamic, target:Dynamic); }
class Builder {
    public static macro function make() {
        var inputs = [{expr:Wrap({expr:Text("wrapped")}, null)}, {expr:Text("plain")}];
        var names = [for (input in inputs) switch input {
            case {expr:Wrap({expr:Text(value)}, _)} | {expr:Text(value)}: value;
            case _: throw "wrong expression";
        }];
        return macro $v{names.join(",")};
    }
}
class Main { static function main() return Builder.make(); }
"#;
    let file = parse(source);
    let mut classes = class_registry::ClassRegistry::new();
    classes.register_file(&file);
    let result = MacroExpander::with_class_registry(classes).expand_file(file);
    assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
    assert_eq!(
        result
            .expansion_origins
            .last()
            .and_then(|origin| origin.expanded_text.as_deref()),
        Some("\"wrapped,plain\"")
    );
}
