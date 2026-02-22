//! Bidirectional conversion between parser AST nodes and MacroValues.
//!
//! This module handles:
//! - Converting AST expression nodes to MacroValue representations
//!   (for when the interpreter encounters literal values in macro code)
//! - Converting MacroValue back to AST expressions
//!   (for when expanded macro results need to be injected back into the AST)

use super::errors::MacroError;
use super::value::MacroValue;
use crate::tast::SourceLocation;
use parser::{BinaryOp, Expr, ExprKind, Span};
use std::collections::HashMap;
use std::sync::Arc;

/// Convert a parser AST expression literal to a MacroValue.
///
/// This handles the subset of expressions that can be directly represented
/// as compile-time values (literals, arrays, objects, etc.).
pub fn expr_to_value(expr: &Expr) -> Result<MacroValue, MacroError> {
    match &expr.kind {
        ExprKind::Int(i) => Ok(MacroValue::Int(*i)),
        ExprKind::Float(f) => Ok(MacroValue::Float(*f)),
        ExprKind::String(s) => Ok(MacroValue::String(Arc::from(s.as_str()))),
        ExprKind::Bool(b) => Ok(MacroValue::Bool(*b)),
        ExprKind::Null => Ok(MacroValue::Null),

        ExprKind::Array(elements) => {
            let mut values = Vec::with_capacity(elements.len());
            for elem in elements {
                values.push(expr_to_value(elem)?);
            }
            Ok(MacroValue::Array(Arc::new(values)))
        }

        ExprKind::Object(fields) => {
            let mut map = HashMap::new();
            for field in fields {
                let value = expr_to_value(&field.expr)?;
                map.insert(field.name.clone(), value);
            }
            Ok(MacroValue::Object(Arc::new(map)))
        }

        ExprKind::Map(entries) => {
            let mut map = HashMap::new();
            for (key, value) in entries {
                let key_str = match &key.kind {
                    ExprKind::String(s) => s.clone(),
                    ExprKind::Int(i) => i.to_string(),
                    _ => {
                        return Err(MacroError::TypeError {
                            message: format!(
                                "map key must be a string or int literal, found {:?}",
                                key.kind
                            ),
                            location: span_to_location(key.span),
                        })
                    }
                };
                map.insert(key_str, expr_to_value(value)?);
            }
            Ok(MacroValue::Object(Arc::new(map)))
        }

        // Unary negation of a literal
        ExprKind::Unary {
            op: parser::UnaryOp::Neg,
            expr: inner,
        } => match &inner.kind {
            ExprKind::Int(i) => Ok(MacroValue::Int(-i)),
            ExprKind::Float(f) => Ok(MacroValue::Float(-f)),
            _ => Ok(MacroValue::Expr(Arc::new(expr.clone()))),
        },

        // For non-literal expressions, wrap as an Expr value
        // The interpreter will evaluate these when needed
        _ => Ok(MacroValue::Expr(Arc::new(expr.clone()))),
    }
}

/// Convert a MacroValue back to a parser AST expression.
///
/// This is used when macro expansion results need to be inserted
/// back into the AST for further compilation.
pub fn value_to_expr(value: &MacroValue) -> Expr {
    let span = Span::default();
    match value {
        MacroValue::Null => Expr {
            kind: ExprKind::Null,
            span,
        },
        MacroValue::Bool(b) => Expr {
            kind: ExprKind::Bool(*b),
            span,
        },
        MacroValue::Int(i) => {
            if *i < 0 {
                Expr {
                    kind: ExprKind::Unary {
                        op: parser::UnaryOp::Neg,
                        expr: Box::new(Expr {
                            kind: ExprKind::Int(-i),
                            span,
                        }),
                    },
                    span,
                }
            } else {
                Expr {
                    kind: ExprKind::Int(*i),
                    span,
                }
            }
        }
        MacroValue::Float(f) => {
            if *f < 0.0 {
                Expr {
                    kind: ExprKind::Unary {
                        op: parser::UnaryOp::Neg,
                        expr: Box::new(Expr {
                            kind: ExprKind::Float(-f),
                            span,
                        }),
                    },
                    span,
                }
            } else {
                Expr {
                    kind: ExprKind::Float(*f),
                    span,
                }
            }
        }
        MacroValue::String(s) => Expr {
            kind: ExprKind::String(s.to_string()),
            span,
        },
        MacroValue::Array(items) => {
            let exprs: Vec<Expr> = items.iter().map(|v| value_to_expr(v)).collect();
            Expr {
                kind: ExprKind::Array(exprs),
                span,
            }
        }
        MacroValue::Object(fields) => {
            let obj_fields: Vec<parser::ObjectField> = fields
                .iter()
                .map(|(name, value)| parser::ObjectField {
                    name: name.clone(),
                    expr: value_to_expr(value),
                    span,
                })
                .collect();
            Expr {
                kind: ExprKind::Object(obj_fields),
                span,
            }
        }
        MacroValue::Enum(enum_name, variant, args) => {
            // Represent as EnumName.Variant(args...)
            let base = Expr {
                kind: ExprKind::Field {
                    expr: Box::new(Expr {
                        kind: ExprKind::Ident(enum_name.to_string()),
                        span,
                    }),
                    field: variant.to_string(),
                    is_optional: false,
                },
                span,
            };
            if args.is_empty() {
                base
            } else {
                let arg_exprs: Vec<Expr> = args.iter().map(|v| value_to_expr(v)).collect();
                Expr {
                    kind: ExprKind::Call {
                        expr: Box::new(base),
                        args: arg_exprs,
                    },
                    span,
                }
            }
        }
        MacroValue::Expr(expr) => expr.as_ref().clone(),
        MacroValue::Type(_type_id) => {
            // Type references can't be directly expressed; use a placeholder
            Expr {
                kind: ExprKind::Null,
                span,
            }
        }
        MacroValue::Function(_) => {
            // Functions can't be directly converted back to AST
            Expr {
                kind: ExprKind::Null,
                span,
            }
        }
        MacroValue::Position(loc) => {
            // Represent position as an object literal { file: ..., min: ..., max: ... }
            let fields = vec![
                parser::ObjectField {
                    name: "file".to_string(),
                    expr: Expr {
                        kind: ExprKind::String(format!("file_{}", loc.file_id)),
                        span,
                    },
                    span,
                },
                parser::ObjectField {
                    name: "min".to_string(),
                    expr: Expr {
                        kind: ExprKind::Int(loc.byte_offset as i64),
                        span,
                    },
                    span,
                },
                parser::ObjectField {
                    name: "max".to_string(),
                    expr: Expr {
                        kind: ExprKind::Int(loc.byte_offset as i64),
                        span,
                    },
                    span,
                },
            ];
            Expr {
                kind: ExprKind::Object(fields),
                span,
            }
        }
    }
}

/// Unwrap a `MacroValue::Expr` to a concrete value.
///
/// When macro parameters are passed as Expr (AST nodes), this function
/// extracts the concrete compile-time value from simple literal expressions.
/// For non-literal expressions, the Expr is returned as-is.
pub fn unwrap_expr_value(val: &MacroValue) -> MacroValue {
    match val {
        MacroValue::Expr(expr) => {
            // Try to extract a concrete value from the expression
            match expr_to_value(expr) {
                Ok(MacroValue::Expr(_)) => val.clone(), // Still an Expr, leave as-is
                Ok(concrete) => concrete,
                Err(_) => val.clone(),
            }
        }
        other => other.clone(),
    }
}

/// Apply a binary operation on two MacroValues.
///
/// Used by the interpreter for evaluating binary expressions.
/// Automatically unwraps Expr-wrapped values (from macro parameters) to
/// concrete values before performing the operation.
pub fn apply_binary_op(
    op: &BinaryOp,
    left: &MacroValue,
    right: &MacroValue,
    location: SourceLocation,
) -> Result<MacroValue, MacroError> {
    // Unwrap Expr-wrapped values (macro parameters) to concrete values
    let left = &unwrap_expr_value(left);
    let right = &unwrap_expr_value(right);
    match op {
        // Arithmetic
        BinaryOp::Add => apply_add(left, right, location),
        BinaryOp::Sub => apply_numeric_op(left, right, |a, b| a - b, |a, b| a - b, location),
        BinaryOp::Mul => apply_numeric_op(left, right, |a, b| a * b, |a, b| a * b, location),
        BinaryOp::Div => {
            // Check for division by zero
            match right {
                MacroValue::Int(0) => Err(MacroError::DivisionByZero { location }),
                MacroValue::Float(f) if *f == 0.0 => Err(MacroError::DivisionByZero { location }),
                _ => apply_numeric_op(left, right, |a, b| a / b, |a, b| a / b, location),
            }
        }
        BinaryOp::Mod => match right {
            MacroValue::Int(0) => Err(MacroError::DivisionByZero { location }),
            _ => apply_numeric_op(left, right, |a, b| a % b, |a, b| a % b, location),
        },

        // Comparison
        BinaryOp::Eq => Ok(MacroValue::Bool(values_equal(left, right))),
        BinaryOp::NotEq => Ok(MacroValue::Bool(!values_equal(left, right))),
        BinaryOp::Lt => {
            compare_values(left, right, |ord| ord == std::cmp::Ordering::Less, location)
        }
        BinaryOp::Le => compare_values(
            left,
            right,
            |ord| ord != std::cmp::Ordering::Greater,
            location,
        ),
        BinaryOp::Gt => compare_values(
            left,
            right,
            |ord| ord == std::cmp::Ordering::Greater,
            location,
        ),
        BinaryOp::Ge => {
            compare_values(left, right, |ord| ord != std::cmp::Ordering::Less, location)
        }

        // Logical
        BinaryOp::And => Ok(MacroValue::Bool(left.is_truthy() && right.is_truthy())),
        BinaryOp::Or => Ok(MacroValue::Bool(left.is_truthy() || right.is_truthy())),

        // Bitwise
        BinaryOp::BitAnd => apply_int_op(left, right, |a, b| a & b, location),
        BinaryOp::BitOr => apply_int_op(left, right, |a, b| a | b, location),
        BinaryOp::BitXor => apply_int_op(left, right, |a, b| a ^ b, location),
        BinaryOp::Shl => apply_int_op(left, right, |a, b| a << b, location),
        BinaryOp::Shr => apply_int_op(left, right, |a, b| a >> b, location),
        BinaryOp::Ushr => apply_int_op(
            left,
            right,
            |a, b| ((a as u64) >> (b as u64)) as i64,
            location,
        ),

        // Range (a...b)
        BinaryOp::Range => {
            let start = left.as_int().ok_or_else(|| MacroError::TypeError {
                message: "interval start must be Int".to_string(),
                location,
            })?;
            let end = right.as_int().ok_or_else(|| MacroError::TypeError {
                message: "interval end must be Int".to_string(),
                location,
            })?;
            let arr: Vec<MacroValue> = (start..end).map(MacroValue::Int).collect();
            Ok(MacroValue::Array(Arc::new(arr)))
        }

        // Arrow (for function types, not typically used in macro values)
        BinaryOp::Arrow => Err(MacroError::UnsupportedOperation {
            operation: "arrow operator in macro context".to_string(),
            location,
        }),

        // Null coalescing
        BinaryOp::NullCoal => {
            if matches!(left, MacroValue::Null) {
                Ok(right.clone())
            } else {
                Ok(left.clone())
            }
        }

        // Other operators
        _ => Err(MacroError::UnsupportedOperation {
            operation: format!("binary operator {:?}", op),
            location,
        }),
    }
}

// --- Helper functions ---

fn apply_add(
    left: &MacroValue,
    right: &MacroValue,
    location: SourceLocation,
) -> Result<MacroValue, MacroError> {
    match (left, right) {
        // String concatenation
        (MacroValue::String(a), b) => Ok(MacroValue::String(Arc::from(
            format!("{}{}", a, b.to_display_string()).as_str(),
        ))),
        (a, MacroValue::String(b)) => Ok(MacroValue::String(Arc::from(
            format!("{}{}", a.to_display_string(), b).as_str(),
        ))),
        // Numeric addition
        (MacroValue::Int(a), MacroValue::Int(b)) => Ok(MacroValue::Int(a + b)),
        (MacroValue::Float(a), MacroValue::Float(b)) => Ok(MacroValue::Float(a + b)),
        (MacroValue::Int(a), MacroValue::Float(b)) => Ok(MacroValue::Float(*a as f64 + b)),
        (MacroValue::Float(a), MacroValue::Int(b)) => Ok(MacroValue::Float(a + *b as f64)),
        // Array concatenation
        (MacroValue::Array(a), MacroValue::Array(b)) => {
            let mut result = a.as_ref().clone();
            result.extend(b.iter().cloned());
            Ok(MacroValue::Array(Arc::new(result)))
        }
        _ => Err(MacroError::TypeError {
            message: format!("cannot add {} and {}", left.type_name(), right.type_name()),
            location,
        }),
    }
}

fn apply_numeric_op(
    left: &MacroValue,
    right: &MacroValue,
    int_op: impl Fn(i64, i64) -> i64,
    float_op: impl Fn(f64, f64) -> f64,
    location: SourceLocation,
) -> Result<MacroValue, MacroError> {
    match (left, right) {
        (MacroValue::Int(a), MacroValue::Int(b)) => Ok(MacroValue::Int(int_op(*a, *b))),
        (MacroValue::Float(a), MacroValue::Float(b)) => Ok(MacroValue::Float(float_op(*a, *b))),
        (MacroValue::Int(a), MacroValue::Float(b)) => {
            Ok(MacroValue::Float(float_op(*a as f64, *b)))
        }
        (MacroValue::Float(a), MacroValue::Int(b)) => {
            Ok(MacroValue::Float(float_op(*a, *b as f64)))
        }
        _ => Err(MacroError::TypeError {
            message: format!(
                "arithmetic operation requires numeric types, found {} and {}",
                left.type_name(),
                right.type_name()
            ),
            location,
        }),
    }
}

fn apply_int_op(
    left: &MacroValue,
    right: &MacroValue,
    op: impl Fn(i64, i64) -> i64,
    location: SourceLocation,
) -> Result<MacroValue, MacroError> {
    let a = left.as_int().ok_or_else(|| MacroError::TypeError {
        message: format!("bitwise operation requires Int, found {}", left.type_name()),
        location,
    })?;
    let b = right.as_int().ok_or_else(|| MacroError::TypeError {
        message: format!(
            "bitwise operation requires Int, found {}",
            right.type_name()
        ),
        location,
    })?;
    Ok(MacroValue::Int(op(a, b)))
}

fn values_equal(left: &MacroValue, right: &MacroValue) -> bool {
    left == right
}

fn compare_values(
    left: &MacroValue,
    right: &MacroValue,
    pred: impl Fn(std::cmp::Ordering) -> bool,
    location: SourceLocation,
) -> Result<MacroValue, MacroError> {
    let ordering = match (left, right) {
        (MacroValue::Int(a), MacroValue::Int(b)) => a.cmp(b),
        (MacroValue::Float(a), MacroValue::Float(b)) => {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        }
        (MacroValue::Int(a), MacroValue::Float(b)) => (*a as f64)
            .partial_cmp(b)
            .unwrap_or(std::cmp::Ordering::Equal),
        (MacroValue::Float(a), MacroValue::Int(b)) => a
            .partial_cmp(&(*b as f64))
            .unwrap_or(std::cmp::Ordering::Equal),
        (MacroValue::String(a), MacroValue::String(b)) => a.cmp(b),
        _ => {
            return Err(MacroError::TypeError {
                message: format!(
                    "cannot compare {} and {}",
                    left.type_name(),
                    right.type_name()
                ),
                location,
            })
        }
    };
    Ok(MacroValue::Bool(pred(ordering)))
}

/// Convert a parser Span to a SourceLocation
pub fn span_to_location(span: Span) -> SourceLocation {
    super::errors::span_to_location(span)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_literal_to_value_and_back() {
        // Int round-trip
        let expr = Expr {
            kind: ExprKind::Int(42),
            span: Span::default(),
        };
        let val = expr_to_value(&expr).unwrap();
        assert_eq!(val, MacroValue::Int(42));
        let back = value_to_expr(&val);
        assert_eq!(back.kind, ExprKind::Int(42));

        // String round-trip
        let expr = Expr {
            kind: ExprKind::String("hello".to_string()),
            span: Span::default(),
        };
        let val = expr_to_value(&expr).unwrap();
        assert_eq!(val, MacroValue::String(Arc::from("hello")));
        let back = value_to_expr(&val);
        assert_eq!(back.kind, ExprKind::String("hello".to_string()));

        // Bool
        let expr = Expr {
            kind: ExprKind::Bool(true),
            span: Span::default(),
        };
        let val = expr_to_value(&expr).unwrap();
        assert_eq!(val, MacroValue::Bool(true));

        // Null
        let expr = Expr {
            kind: ExprKind::Null,
            span: Span::default(),
        };
        let val = expr_to_value(&expr).unwrap();
        assert_eq!(val, MacroValue::Null);

        // Float
        let expr = Expr {
            kind: ExprKind::Float(3.14),
            span: Span::default(),
        };
        let val = expr_to_value(&expr).unwrap();
        assert!(matches!(val, MacroValue::Float(f) if (f - 3.14).abs() < 1e-10));
    }

    #[test]
    fn test_array_round_trip() {
        let expr = Expr {
            kind: ExprKind::Array(vec![
                Expr {
                    kind: ExprKind::Int(1),
                    span: Span::default(),
                },
                Expr {
                    kind: ExprKind::Int(2),
                    span: Span::default(),
                },
                Expr {
                    kind: ExprKind::Int(3),
                    span: Span::default(),
                },
            ]),
            span: Span::default(),
        };
        let val = expr_to_value(&expr).unwrap();
        match &val {
            MacroValue::Array(items) => {
                assert_eq!(items.len(), 3);
                assert_eq!(items[0], MacroValue::Int(1));
                assert_eq!(items[1], MacroValue::Int(2));
                assert_eq!(items[2], MacroValue::Int(3));
            }
            _ => panic!("expected Array"),
        }

        let back = value_to_expr(&val);
        match &back.kind {
            ExprKind::Array(items) => {
                assert_eq!(items.len(), 3);
            }
            _ => panic!("expected Array expr"),
        }
    }

    #[test]
    fn test_non_literal_wraps_as_expr() {
        let expr = Expr {
            kind: ExprKind::Ident("someVar".to_string()),
            span: Span::default(),
        };
        let val = expr_to_value(&expr).unwrap();
        assert!(matches!(val, MacroValue::Expr(_)));
    }

    #[test]
    fn test_binary_ops() {
        let loc = SourceLocation::unknown();

        // Int arithmetic
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
                &MacroValue::Int(3),
                &MacroValue::Int(4),
                loc
            )
            .unwrap(),
            MacroValue::Int(12)
        );
        assert_eq!(
            apply_binary_op(
                &BinaryOp::Div,
                &MacroValue::Int(10),
                &MacroValue::Int(3),
                loc
            )
            .unwrap(),
            MacroValue::Int(3)
        );

        // String concat
        assert_eq!(
            apply_binary_op(
                &BinaryOp::Add,
                &MacroValue::String(Arc::from("hello")),
                &MacroValue::String(Arc::from(" world")),
                loc
            )
            .unwrap(),
            MacroValue::String(Arc::from("hello world"))
        );

        // Comparison
        assert_eq!(
            apply_binary_op(&BinaryOp::Lt, &MacroValue::Int(1), &MacroValue::Int(2), loc).unwrap(),
            MacroValue::Bool(true)
        );
        assert_eq!(
            apply_binary_op(&BinaryOp::Eq, &MacroValue::Int(1), &MacroValue::Int(1), loc).unwrap(),
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
    }

    #[test]
    fn test_division_by_zero() {
        let loc = SourceLocation::unknown();
        let result = apply_binary_op(
            &BinaryOp::Div,
            &MacroValue::Int(10),
            &MacroValue::Int(0),
            loc,
        );
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            MacroError::DivisionByZero { .. }
        ));
    }

    #[test]
    fn test_null_coalescing() {
        let loc = SourceLocation::unknown();
        assert_eq!(
            apply_binary_op(
                &BinaryOp::NullCoal,
                &MacroValue::Null,
                &MacroValue::Int(42),
                loc
            )
            .unwrap(),
            MacroValue::Int(42)
        );
        assert_eq!(
            apply_binary_op(
                &BinaryOp::NullCoal,
                &MacroValue::Int(1),
                &MacroValue::Int(42),
                loc
            )
            .unwrap(),
            MacroValue::Int(1)
        );
    }

    #[test]
    fn test_negative_int_round_trip() {
        let val = MacroValue::Int(-5);
        let expr = value_to_expr(&val);
        // Should be Unary { Neg, Int(5) }
        match &expr.kind {
            ExprKind::Unary {
                op: parser::UnaryOp::Neg,
                expr: inner,
            } => {
                assert_eq!(inner.kind, ExprKind::Int(5));
            }
            _ => panic!("expected unary neg for negative int"),
        }
    }
}
