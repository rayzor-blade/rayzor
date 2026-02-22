//! Reification Engine for Haxe Macros
//!
//! Handles the conversion between macro-time code and AST construction.
//!
//! In Haxe, `macro { expr }` creates an expression that, when evaluated at
//! compile time, produces an AST representation of `expr`. Dollar identifiers
//! within reification blocks allow splicing runtime values into the constructed AST:
//!
//! - `$v{value}` — splice a value as a constant expression
//! - `$i{ident}` — splice as an identifier
//! - `$e{expr}` — splice an expression directly
//! - `$a{array}` — splice array elements
//! - `$p{path}` — splice as a type path
//! - `$b{block}` — splice as a block of statements

use super::ast_bridge::{self, span_to_location};
use super::environment::Environment;
use super::errors::MacroError;
use super::value::MacroValue;
use parser::{Expr, ExprKind, Span};
use std::sync::Arc;

/// The reification engine processes macro blocks and dollar identifiers.
pub struct ReificationEngine;

impl ReificationEngine {
    /// Reify an expression from a `macro { ... }` block.
    ///
    /// This takes the AST inside a macro block and converts it into a
    /// MacroValue::Expr that represents the expression tree. Dollar
    /// identifiers within the expression are evaluated against the
    /// provided environment and spliced into the result.
    pub fn reify_expr(expr: &Expr, env: &Environment) -> Result<MacroValue, MacroError> {
        let reified = Self::process_expr(expr, env)?;
        Ok(MacroValue::Expr(Arc::new(reified)))
    }

    /// Process an expression, resolving dollar identifiers against the environment.
    ///
    /// Non-dollar expressions are returned as-is (they become literal AST nodes).
    /// Dollar identifiers are evaluated and their results are spliced into the output.
    fn process_expr(expr: &Expr, env: &Environment) -> Result<Expr, MacroError> {
        match &expr.kind {
            // Dollar identifier — splice from environment
            ExprKind::DollarIdent { name, arg } => {
                Self::process_dollar_ident(name, arg.as_deref(), env, expr.span)
            }

            // Recursively process sub-expressions in compound nodes
            ExprKind::Block(elements) => {
                let mut new_elements = Vec::with_capacity(elements.len());
                for elem in elements {
                    match elem {
                        parser::BlockElement::Expr(e) => {
                            let processed = Self::process_expr(e, env)?;
                            new_elements.push(parser::BlockElement::Expr(processed));
                        }
                        other => new_elements.push(other.clone()),
                    }
                }
                Ok(Expr {
                    kind: ExprKind::Block(new_elements),
                    span: expr.span,
                })
            }

            ExprKind::Call { expr: callee, args } => {
                let new_callee = Self::process_expr(callee, env)?;
                let new_args: Result<Vec<Expr>, MacroError> =
                    args.iter().map(|a| Self::process_expr(a, env)).collect();
                Ok(Expr {
                    kind: ExprKind::Call {
                        expr: Box::new(new_callee),
                        args: new_args?,
                    },
                    span: expr.span,
                })
            }

            ExprKind::Field {
                expr: base,
                field,
                is_optional,
            } => {
                let new_base = Self::process_expr(base, env)?;
                Ok(Expr {
                    kind: ExprKind::Field {
                        expr: Box::new(new_base),
                        field: field.clone(),
                        is_optional: *is_optional,
                    },
                    span: expr.span,
                })
            }

            ExprKind::Binary { left, op, right } => {
                let new_left = Self::process_expr(left, env)?;
                let new_right = Self::process_expr(right, env)?;
                Ok(Expr {
                    kind: ExprKind::Binary {
                        left: Box::new(new_left),
                        op: *op,
                        right: Box::new(new_right),
                    },
                    span: expr.span,
                })
            }

            ExprKind::Unary { op, expr: inner } => {
                let new_inner = Self::process_expr(inner, env)?;
                Ok(Expr {
                    kind: ExprKind::Unary {
                        op: *op,
                        expr: Box::new(new_inner),
                    },
                    span: expr.span,
                })
            }

            ExprKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let new_cond = Self::process_expr(cond, env)?;
                let new_then = Self::process_expr(then_branch, env)?;
                let new_else = else_branch
                    .as_ref()
                    .map(|e| Self::process_expr(e, env))
                    .transpose()?;
                Ok(Expr {
                    kind: ExprKind::If {
                        cond: Box::new(new_cond),
                        then_branch: Box::new(new_then),
                        else_branch: new_else.map(Box::new),
                    },
                    span: expr.span,
                })
            }

            ExprKind::Return(inner) => {
                let new_inner = inner
                    .as_ref()
                    .map(|e| Self::process_expr(e, env))
                    .transpose()?;
                Ok(Expr {
                    kind: ExprKind::Return(new_inner.map(Box::new)),
                    span: expr.span,
                })
            }

            ExprKind::Var {
                name,
                type_hint,
                expr: init,
            } => {
                let new_init = init
                    .as_ref()
                    .map(|e| Self::process_expr(e, env))
                    .transpose()?;
                Ok(Expr {
                    kind: ExprKind::Var {
                        name: name.clone(),
                        type_hint: type_hint.clone(),
                        expr: new_init.map(Box::new),
                    },
                    span: expr.span,
                })
            }

            ExprKind::Array(elements) => {
                let new_elems: Result<Vec<Expr>, MacroError> = elements
                    .iter()
                    .map(|e| Self::process_expr(e, env))
                    .collect();
                Ok(Expr {
                    kind: ExprKind::Array(new_elems?),
                    span: expr.span,
                })
            }

            ExprKind::Assign { left, op, right } => {
                let new_left = Self::process_expr(left, env)?;
                let new_right = Self::process_expr(right, env)?;
                Ok(Expr {
                    kind: ExprKind::Assign {
                        left: Box::new(new_left),
                        op: *op,
                        right: Box::new(new_right),
                    },
                    span: expr.span,
                })
            }

            ExprKind::Index { expr: base, index } => {
                let new_base = Self::process_expr(base, env)?;
                let new_index = Self::process_expr(index, env)?;
                Ok(Expr {
                    kind: ExprKind::Index {
                        expr: Box::new(new_base),
                        index: Box::new(new_index),
                    },
                    span: expr.span,
                })
            }

            ExprKind::Paren(inner) => {
                let new_inner = Self::process_expr(inner, env)?;
                Ok(Expr {
                    kind: ExprKind::Paren(Box::new(new_inner)),
                    span: expr.span,
                })
            }

            ExprKind::Tuple(elements) => {
                let new_elements = elements
                    .iter()
                    .map(|e| Self::process_expr(e, env))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Expr {
                    kind: ExprKind::Tuple(new_elements),
                    span: expr.span,
                })
            }

            // Leaf nodes — return as-is
            _ => Ok(expr.clone()),
        }
    }

    /// Process a dollar identifier, resolving it against the environment.
    ///
    /// Supported forms:
    /// - `$v{expr}` — value splice: evaluate expr, convert to constant AST node
    /// - `$i{expr}` — identifier splice: evaluate expr (must be string), use as identifier
    /// - `$e{expr}` — expression splice: evaluate expr (must be Expr), insert directly
    /// - `$a{expr}` — array splice: evaluate expr (must be Array<Expr>), splice elements
    /// - `$p{expr}` — path splice: evaluate expr (must be string), use as type path
    /// - `$b{expr}` — block splice: evaluate expr (must be Array<Expr>), use as block
    /// - `$name` (no arg) — simple identifier splice from environment
    fn process_dollar_ident(
        name: &str,
        arg: Option<&Expr>,
        env: &Environment,
        span: Span,
    ) -> Result<Expr, MacroError> {
        let location = span_to_location(span);

        match (name, arg) {
            // $v{expr} — value splice
            ("v", Some(_arg_expr)) => {
                // In a full implementation, we'd evaluate arg_expr in the macro
                // environment to get a MacroValue, then convert it to an AST literal.
                // For now, we look up the expression and convert.
                Err(MacroError::ReificationError {
                    message: "$v{} requires interpreter evaluation (implemented in Phase 3)"
                        .to_string(),
                    location,
                })
            }

            // $i{expr} — identifier splice
            ("i", Some(_arg_expr)) => Err(MacroError::ReificationError {
                message: "$i{} requires interpreter evaluation (implemented in Phase 3)"
                    .to_string(),
                location,
            }),

            // $e{expr} — expression splice
            ("e", Some(_arg_expr)) => Err(MacroError::ReificationError {
                message: "$e{} requires interpreter evaluation (implemented in Phase 3)"
                    .to_string(),
                location,
            }),

            // $a{expr} — array splice
            ("a", Some(_arg_expr)) => Err(MacroError::ReificationError {
                message: "$a{} requires interpreter evaluation (implemented in Phase 3)"
                    .to_string(),
                location,
            }),

            // $p{expr} — path splice
            ("p", Some(_arg_expr)) => Err(MacroError::ReificationError {
                message: "$p{} requires interpreter evaluation (implemented in Phase 3)"
                    .to_string(),
                location,
            }),

            // $b{expr} — block splice
            ("b", Some(_arg_expr)) => Err(MacroError::ReificationError {
                message: "$b{} requires interpreter evaluation (implemented in Phase 3)"
                    .to_string(),
                location,
            }),

            // $name (no argument) — simple variable splice from environment
            (var_name, None) => {
                if let Some(value) = env.get(var_name) {
                    Ok(ast_bridge::value_to_expr(value))
                } else {
                    Err(MacroError::UndefinedVariable {
                        name: format!("${}", var_name),
                        location,
                    })
                }
            }

            // Unknown dollar identifier with arg
            (unknown, Some(_)) => Err(MacroError::ReificationError {
                message: format!("unknown dollar identifier: ${}{{...}}", unknown),
                location,
            }),
        }
    }

    /// Process a dollar identifier with an already-evaluated argument value.
    ///
    /// This is the version called by the interpreter after evaluating the
    /// argument expression.
    pub fn splice_value(kind: &str, value: MacroValue, span: Span) -> Result<Expr, MacroError> {
        let location = span_to_location(span);

        match kind {
            // $v{value} — convert value to a constant expression
            "v" => Ok(ast_bridge::value_to_expr(&value)),

            // $i{value} — value must be a string, used as identifier
            "i" => match value {
                MacroValue::String(name) => Ok(Expr {
                    kind: ExprKind::Ident(name.to_string()),
                    span,
                }),
                _ => Err(MacroError::ReificationError {
                    message: format!("$i{{}} expects a String value, got {}", value.type_name()),
                    location,
                }),
            },

            // $e{value} — value must be an Expr, splice directly
            "e" => match value {
                MacroValue::Expr(expr) => Ok((*expr).clone()),
                _ => Err(MacroError::ReificationError {
                    message: format!("$e{{}} expects an Expr value, got {}", value.type_name()),
                    location,
                }),
            },

            // $a{value} — value must be Array<Expr>, splice as array literal
            "a" => match value {
                MacroValue::Array(items) => {
                    let exprs: Result<Vec<Expr>, MacroError> = items
                        .iter()
                        .map(|item| match item {
                            MacroValue::Expr(e) => Ok((**e).clone()),
                            other => Ok(ast_bridge::value_to_expr(other)),
                        })
                        .collect();
                    Ok(Expr {
                        kind: ExprKind::Array(exprs?),
                        span,
                    })
                }
                _ => Err(MacroError::ReificationError {
                    message: format!("$a{{}} expects an Array value, got {}", value.type_name()),
                    location,
                }),
            },

            // $p{value} — value must be a string, parse as dotted type path
            "p" => match value {
                MacroValue::String(path_str) => {
                    let parts: Vec<&str> = path_str.split('.').collect();
                    if parts.is_empty() {
                        return Err(MacroError::ReificationError {
                            message: "$p{} path string is empty".to_string(),
                            location,
                        });
                    }
                    // Build a chain of Field expressions for the path
                    let mut result = Expr {
                        kind: ExprKind::Ident(parts[0].to_string()),
                        span,
                    };
                    for part in &parts[1..] {
                        result = Expr {
                            kind: ExprKind::Field {
                                expr: Box::new(result),
                                field: part.to_string(),
                                is_optional: false,
                            },
                            span,
                        };
                    }
                    Ok(result)
                }
                _ => Err(MacroError::ReificationError {
                    message: format!("$p{{}} expects a String path, got {}", value.type_name()),
                    location,
                }),
            },

            // $b{value} — value must be Array<Expr>, splice as block
            "b" => match value {
                MacroValue::Array(items) => {
                    let elements: Result<Vec<parser::BlockElement>, MacroError> = items
                        .iter()
                        .map(|item| match item {
                            MacroValue::Expr(e) => Ok(parser::BlockElement::Expr((**e).clone())),
                            other => {
                                Ok(parser::BlockElement::Expr(ast_bridge::value_to_expr(other)))
                            }
                        })
                        .collect();
                    Ok(Expr {
                        kind: ExprKind::Block(elements?),
                        span,
                    })
                }
                _ => Err(MacroError::ReificationError {
                    message: format!("$b{{}} expects an Array value, got {}", value.type_name()),
                    location,
                }),
            },

            unknown => Err(MacroError::ReificationError {
                message: format!("unknown splice kind: ${}", unknown),
                location,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reify_simple_literal() {
        let env = Environment::new();
        let expr = Expr {
            kind: ExprKind::Int(42),
            span: Span::default(),
        };
        let result = ReificationEngine::reify_expr(&expr, &env).unwrap();
        match result {
            MacroValue::Expr(e) => {
                assert_eq!(e.kind, ExprKind::Int(42));
            }
            _ => panic!("expected Expr"),
        }
    }

    #[test]
    fn test_reify_with_dollar_var() {
        let mut env = Environment::new();
        env.define("x", MacroValue::Int(42));

        let expr = Expr {
            kind: ExprKind::DollarIdent {
                name: "x".to_string(),
                arg: None,
            },
            span: Span::default(),
        };
        let result = ReificationEngine::reify_expr(&expr, &env).unwrap();
        match result {
            MacroValue::Expr(e) => {
                assert_eq!(e.kind, ExprKind::Int(42));
            }
            _ => panic!("expected Expr with int value"),
        }
    }

    #[test]
    fn test_reify_undefined_dollar_var() {
        let env = Environment::new();
        let expr = Expr {
            kind: ExprKind::DollarIdent {
                name: "unknown".to_string(),
                arg: None,
            },
            span: Span::default(),
        };
        let result = ReificationEngine::reify_expr(&expr, &env);
        assert!(result.is_err());
    }

    #[test]
    fn test_splice_value_v() {
        let result =
            ReificationEngine::splice_value("v", MacroValue::Int(42), Span::default()).unwrap();
        assert_eq!(result.kind, ExprKind::Int(42));
    }

    #[test]
    fn test_splice_value_i() {
        let result =
            ReificationEngine::splice_value("i", MacroValue::from_str("myVar"), Span::default())
                .unwrap();
        assert_eq!(result.kind, ExprKind::Ident("myVar".to_string()));
    }

    #[test]
    fn test_splice_value_i_type_error() {
        let result = ReificationEngine::splice_value("i", MacroValue::Int(42), Span::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_splice_value_e() {
        let inner_expr = Expr {
            kind: ExprKind::String("hello".to_string()),
            span: Span::default(),
        };
        let result = ReificationEngine::splice_value(
            "e",
            MacroValue::Expr(Arc::new(inner_expr.clone())),
            Span::default(),
        )
        .unwrap();
        assert_eq!(result.kind, inner_expr.kind);
    }

    #[test]
    fn test_splice_value_p() {
        let result = ReificationEngine::splice_value(
            "p",
            MacroValue::from_str("com.example.MyClass"),
            Span::default(),
        )
        .unwrap();
        // Should be a chain: Field(Field(Ident("com"), "example"), "MyClass")
        match &result.kind {
            ExprKind::Field {
                expr: mid, field, ..
            } => {
                assert_eq!(field, "MyClass");
                match &mid.kind {
                    ExprKind::Field {
                        expr: base,
                        field: mid_field,
                        ..
                    } => {
                        assert_eq!(mid_field, "example");
                        assert_eq!(base.kind, ExprKind::Ident("com".to_string()));
                    }
                    _ => panic!("expected nested Field"),
                }
            }
            _ => panic!("expected Field expression"),
        }
    }

    #[test]
    fn test_splice_value_a() {
        let items = vec![
            MacroValue::Expr(Arc::new(Expr {
                kind: ExprKind::Int(1),
                span: Span::default(),
            })),
            MacroValue::Expr(Arc::new(Expr {
                kind: ExprKind::Int(2),
                span: Span::default(),
            })),
        ];
        let result = ReificationEngine::splice_value(
            "a",
            MacroValue::Array(Arc::new(items)),
            Span::default(),
        )
        .unwrap();
        match &result.kind {
            ExprKind::Array(elems) => {
                assert_eq!(elems.len(), 2);
                assert_eq!(elems[0].kind, ExprKind::Int(1));
                assert_eq!(elems[1].kind, ExprKind::Int(2));
            }
            _ => panic!("expected Array"),
        }
    }

    #[test]
    fn test_splice_value_b() {
        let items = vec![MacroValue::Expr(Arc::new(Expr {
            kind: ExprKind::Return(Some(Box::new(Expr {
                kind: ExprKind::Int(42),
                span: Span::default(),
            }))),
            span: Span::default(),
        }))];
        let result = ReificationEngine::splice_value(
            "b",
            MacroValue::Array(Arc::new(items)),
            Span::default(),
        )
        .unwrap();
        match &result.kind {
            ExprKind::Block(elements) => {
                assert_eq!(elements.len(), 1);
            }
            _ => panic!("expected Block"),
        }
    }

    #[test]
    fn test_reify_block_with_dollar() {
        let mut env = Environment::new();
        env.define("val", MacroValue::from_str("test"));

        let expr = Expr {
            kind: ExprKind::Block(vec![
                parser::BlockElement::Expr(Expr {
                    kind: ExprKind::DollarIdent {
                        name: "val".to_string(),
                        arg: None,
                    },
                    span: Span::default(),
                }),
                parser::BlockElement::Expr(Expr {
                    kind: ExprKind::Int(42),
                    span: Span::default(),
                }),
            ]),
            span: Span::default(),
        };

        let result = ReificationEngine::reify_expr(&expr, &env).unwrap();
        match result {
            MacroValue::Expr(e) => match &e.kind {
                ExprKind::Block(elements) => {
                    assert_eq!(elements.len(), 2);
                    // First element should be the spliced string
                    match &elements[0] {
                        parser::BlockElement::Expr(e) => {
                            assert_eq!(e.kind, ExprKind::String("test".to_string()));
                        }
                        _ => panic!("expected Expr block element"),
                    }
                }
                _ => panic!("expected Block"),
            },
            _ => panic!("expected Expr"),
        }
    }
}
