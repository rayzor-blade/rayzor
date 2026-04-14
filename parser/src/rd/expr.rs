//! Expression parser using precedence climbing (Pratt parsing).

use super::error::ParseError;
use super::RdParser;
use crate::haxe_ast::*;
use crate::haxe_parser_expr::unescape_string;
use crate::token::TokenKind;

impl<'a, 'b> RdParser<'a, 'b> {
    /// Parse an expression.
    pub fn parse_expression(&mut self) -> Result<Expr, ParseError> {
        let expr = self.parse_assignment()?;
        // Check for inline type check: `expr : Type` (inside parens, switch, etc.)
        // Only consume if we're inside parens (next after type would be `)`)
        // This is a heuristic — the : might be a ternary else or object field
        Ok(expr)
    }

    fn parse_assignment(&mut self) -> Result<Expr, ParseError> {
        let left = self.parse_ternary()?;

        if self.stream.at_any(&[
            TokenKind::Assign,
            TokenKind::PlusAssign,
            TokenKind::MinusAssign,
            TokenKind::StarAssign,
            TokenKind::SlashAssign,
            TokenKind::PercentAssign,
            TokenKind::AmpAssign,
            TokenKind::PipeAssign,
            TokenKind::CaretAssign,
            TokenKind::ShlAssign,
            TokenKind::ShrAssign,
            TokenKind::UshrAssign,
        ]) {
            let op = match self.stream.advance().kind {
                TokenKind::Assign => AssignOp::Assign,
                TokenKind::PlusAssign => AssignOp::AddAssign,
                TokenKind::MinusAssign => AssignOp::SubAssign,
                TokenKind::StarAssign => AssignOp::MulAssign,
                TokenKind::SlashAssign => AssignOp::DivAssign,
                TokenKind::PercentAssign => AssignOp::ModAssign,
                TokenKind::AmpAssign => AssignOp::AndAssign,
                TokenKind::PipeAssign => AssignOp::OrAssign,
                TokenKind::CaretAssign => AssignOp::XorAssign,
                TokenKind::ShlAssign => AssignOp::ShlAssign,
                TokenKind::ShrAssign => AssignOp::ShrAssign,
                TokenKind::UshrAssign => AssignOp::UshrAssign,
                _ => unreachable!(),
            };
            let right = self.parse_assignment()?;
            let span = left.span.merge(right.span);
            return Ok(Expr {
                kind: ExprKind::Assign {
                    left: Box::new(left),
                    op,
                    right: Box::new(right),
                },
                span,
            });
        }

        Ok(left)
    }

    fn parse_ternary(&mut self) -> Result<Expr, ParseError> {
        let cond = self.parse_binary(0)?;

        if self.stream.eat(TokenKind::Question).is_some() {
            let then_expr = self.parse_expression()?;
            self.stream.expect(TokenKind::Colon)?;
            let else_expr = self.parse_expression()?;
            let span = cond.span.merge(else_expr.span);
            return Ok(Expr {
                kind: ExprKind::Ternary {
                    cond: Box::new(cond),
                    then_expr: Box::new(then_expr),
                    else_expr: Box::new(else_expr),
                },
                span,
            });
        }

        Ok(cond)
    }

    /// Precedence-climbing binary expression parser.
    fn parse_binary(&mut self, min_prec: u8) -> Result<Expr, ParseError> {
        let mut left = self.parse_unary()?;

        loop {
            let (op, prec, right_assoc) = match self.stream.peek().kind {
                TokenKind::QuestionQuestion => (BinaryOp::NullCoal, 2, true),
                TokenKind::PipePipe => (BinaryOp::Or, 3, false),
                TokenKind::AmpAmp => (BinaryOp::And, 4, false),
                TokenKind::Pipe => (BinaryOp::BitOr, 5, false),
                TokenKind::Caret => (BinaryOp::BitXor, 6, false),
                TokenKind::Amp => (BinaryOp::BitAnd, 7, false),
                TokenKind::Eq => (BinaryOp::Eq, 8, false),
                TokenKind::NotEq => (BinaryOp::NotEq, 8, false),
                TokenKind::Lt => (BinaryOp::Lt, 9, false),
                TokenKind::Le => (BinaryOp::Le, 9, false),
                TokenKind::Gt => (BinaryOp::Gt, 9, false),
                TokenKind::Ge => (BinaryOp::Ge, 9, false),
                TokenKind::KwIs => (BinaryOp::Is, 9, false),
                TokenKind::DotDotDot => (BinaryOp::Range, 10, false),
                TokenKind::Shl => (BinaryOp::Shl, 11, false),
                TokenKind::Shr => (BinaryOp::Shr, 11, false),
                TokenKind::Ushr => (BinaryOp::Ushr, 11, false),
                TokenKind::Plus => (BinaryOp::Add, 12, false),
                TokenKind::Minus => (BinaryOp::Sub, 12, false),
                TokenKind::Star => (BinaryOp::Mul, 13, false),
                TokenKind::Slash => (BinaryOp::Div, 13, false),
                TokenKind::Percent => (BinaryOp::Mod, 13, false),
                _ => break,
            };

            if prec < min_prec {
                break;
            }

            self.stream.advance();
            let next_prec = if right_assoc { prec } else { prec + 1 };
            let right = self.parse_binary(next_prec)?;
            let span = left.span.merge(right.span);
            left = Expr {
                kind: ExprKind::Binary {
                    left: Box::new(left),
                    op,
                    right: Box::new(right),
                },
                span,
            };
        }

        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<Expr, ParseError> {
        let start = self.stream.current_offset();

        match self.stream.peek().kind {
            TokenKind::Bang => {
                self.stream.advance();
                let expr = self.parse_unary()?;
                Ok(Expr {
                    span: Span::new(start, expr.span.end),
                    kind: ExprKind::Unary {
                        op: UnaryOp::Not,
                        expr: Box::new(expr),
                    },
                })
            }
            TokenKind::Minus => {
                self.stream.advance();
                let expr = self.parse_unary()?;
                Ok(Expr {
                    span: Span::new(start, expr.span.end),
                    kind: ExprKind::Unary {
                        op: UnaryOp::Neg,
                        expr: Box::new(expr),
                    },
                })
            }
            TokenKind::Tilde => {
                self.stream.advance();
                let expr = self.parse_unary()?;
                Ok(Expr {
                    span: Span::new(start, expr.span.end),
                    kind: ExprKind::Unary {
                        op: UnaryOp::BitNot,
                        expr: Box::new(expr),
                    },
                })
            }
            TokenKind::PlusPlus => {
                self.stream.advance();
                let expr = self.parse_unary()?;
                Ok(Expr {
                    span: Span::new(start, expr.span.end),
                    kind: ExprKind::Unary {
                        op: UnaryOp::PreIncr,
                        expr: Box::new(expr),
                    },
                })
            }
            TokenKind::MinusMinus => {
                self.stream.advance();
                let expr = self.parse_unary()?;
                Ok(Expr {
                    span: Span::new(start, expr.span.end),
                    kind: ExprKind::Unary {
                        op: UnaryOp::PreDecr,
                        expr: Box::new(expr),
                    },
                })
            }
            _ => self.parse_postfix(),
        }
    }

    fn parse_postfix(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_primary()?;

        loop {
            match self.stream.peek().kind {
                TokenKind::Dot => {
                    self.stream.advance();
                    let field = self.stream.current_text().to_string();
                    let end = self.stream.peek().span.end;
                    self.stream.advance();
                    expr = Expr {
                        span: Span::new(expr.span.start, end),
                        kind: ExprKind::Field {
                            expr: Box::new(expr),
                            field,
                            is_optional: false,
                        },
                    };
                }
                TokenKind::QuestionDot => {
                    self.stream.advance();
                    let field = self.stream.current_text().to_string();
                    let end = self.stream.peek().span.end;
                    self.stream.advance();
                    expr = Expr {
                        span: Span::new(expr.span.start, end),
                        kind: ExprKind::Field {
                            expr: Box::new(expr),
                            field,
                            is_optional: true,
                        },
                    };
                }
                TokenKind::LBracket => {
                    self.stream.advance();
                    let index = self.parse_expression()?;
                    let end = self.stream.expect(TokenKind::RBracket)?;
                    expr = Expr {
                        span: Span::new(expr.span.start, end.end),
                        kind: ExprKind::Index {
                            expr: Box::new(expr),
                            index: Box::new(index),
                        },
                    };
                }
                TokenKind::LParen => {
                    let args = self.parse_call_args()?;
                    let end = self.stream.span_from(expr.span.start).end;
                    expr = Expr {
                        span: Span::new(expr.span.start, end),
                        kind: ExprKind::Call {
                            expr: Box::new(expr),
                            args,
                        },
                    };
                }
                TokenKind::PlusPlus => {
                    let end = self.stream.peek().span.end;
                    self.stream.advance();
                    expr = Expr {
                        span: Span::new(expr.span.start, end),
                        kind: ExprKind::Unary {
                            op: UnaryOp::PostIncr,
                            expr: Box::new(expr),
                        },
                    };
                }
                TokenKind::MinusMinus => {
                    let end = self.stream.peek().span.end;
                    self.stream.advance();
                    expr = Expr {
                        span: Span::new(expr.span.start, end),
                        kind: ExprKind::Unary {
                            op: UnaryOp::PostDecr,
                            expr: Box::new(expr),
                        },
                    };
                }
                _ => break,
            }
        }

        Ok(expr)
    }

    fn parse_call_args(&mut self) -> Result<Vec<Expr>, ParseError> {
        self.stream.expect(TokenKind::LParen)?;
        let mut args = Vec::new();
        while !self.stream.at(TokenKind::RParen) && !self.stream.is_eof() {
            args.push(self.parse_expression()?);
            if !self.stream.at(TokenKind::RParen) {
                self.stream.eat(TokenKind::Comma);
            }
        }
        self.stream.expect(TokenKind::RParen)?;
        Ok(args)
    }

    fn parse_primary(&mut self) -> Result<Expr, ParseError> {
        let start = self.stream.current_offset();
        let token = self.stream.peek().clone();

        match token.kind {
            TokenKind::IntLit => {
                let text = token.text(self.source);
                self.stream.advance();
                let val = if text.starts_with("0x") || text.starts_with("0X") {
                    i64::from_str_radix(&text[2..], 16).unwrap_or(0)
                } else if text.starts_with('0')
                    && text.len() > 1
                    && text.as_bytes()[1].is_ascii_digit()
                {
                    // Octal: 0755 → 493
                    i64::from_str_radix(&text[1..], 8).unwrap_or(0)
                } else {
                    text.parse::<i64>().unwrap_or(0)
                };
                Ok(Expr {
                    kind: ExprKind::Int(val),
                    span: token.span,
                })
            }
            TokenKind::FloatLit => {
                let text = token.text(self.source);
                self.stream.advance();
                Ok(Expr {
                    kind: ExprKind::Float(text.parse().unwrap_or(0.0)),
                    span: token.span,
                })
            }
            TokenKind::StringLit => {
                let text = token.text(self.source);
                self.stream.advance();
                let quote = text.as_bytes()[0];
                let inner = &text[1..text.len() - 1];

                // Single-quoted strings with $ are interpolated in Haxe
                if quote == b'\'' && inner.contains('$') {
                    let parts = self.parse_string_interpolation_parts(inner, token.span.start + 1);
                    Ok(Expr {
                        kind: ExprKind::StringInterpolation(parts),
                        span: token.span,
                    })
                } else {
                    Ok(Expr {
                        kind: ExprKind::String(unescape_string(inner)),
                        span: token.span,
                    })
                }
            }
            TokenKind::KwTrue => {
                self.stream.advance();
                Ok(Expr {
                    kind: ExprKind::Bool(true),
                    span: token.span,
                })
            }
            TokenKind::KwFalse => {
                self.stream.advance();
                Ok(Expr {
                    kind: ExprKind::Bool(false),
                    span: token.span,
                })
            }
            TokenKind::KwNull => {
                self.stream.advance();
                Ok(Expr {
                    kind: ExprKind::Null,
                    span: token.span,
                })
            }
            TokenKind::KwThis => {
                self.stream.advance();
                Ok(Expr {
                    kind: ExprKind::This,
                    span: token.span,
                })
            }
            TokenKind::KwSuper => {
                self.stream.advance();
                Ok(Expr {
                    kind: ExprKind::Super,
                    span: token.span,
                })
            }
            TokenKind::Ident => {
                let name = token.text(self.source).to_string();
                // Compiler-specific code block: __js__("code"), __cpp__("code", arg0, ...)
                if name.starts_with("__") && name.ends_with("__") && name.len() > 4 {
                    self.stream.advance();
                    self.stream.expect(TokenKind::LParen)?;
                    let code = self.parse_expression()?;
                    let mut args = Vec::new();
                    while self.stream.at(TokenKind::Comma) {
                        self.stream.advance();
                        args.push(self.parse_expression()?);
                    }
                    let end = self.stream.current_offset();
                    self.stream.expect(TokenKind::RParen)?;
                    return Ok(Expr {
                        kind: ExprKind::CompilerSpecific {
                            target: name,
                            code: Box::new(code),
                            args,
                        },
                        span: Span::new(start, end),
                    });
                }
                self.stream.advance();
                Ok(Expr {
                    kind: ExprKind::Ident(name),
                    span: token.span,
                })
            }
            TokenKind::KwNew => {
                self.stream.advance();
                let path = self.parse_type_path()?;
                let params = if self.stream.at(TokenKind::Lt) {
                    self.parse_type_args()?
                } else {
                    Vec::new()
                };
                let args = self.parse_call_args()?;
                Ok(Expr {
                    kind: ExprKind::New {
                        type_path: path,
                        params,
                        args,
                    },
                    span: self.stream.span_from(start),
                })
            }
            TokenKind::KwIf => self.parse_if_expr(),
            TokenKind::KwWhile => self.parse_while_expr(),
            TokenKind::KwFor => self.parse_for_expr(),
            TokenKind::KwReturn => {
                self.stream.advance();
                let value = if !self.stream.at(TokenKind::Semicolon)
                    && !self.stream.at(TokenKind::RBrace)
                    && !self.stream.is_eof()
                {
                    Some(Box::new(self.parse_expression()?))
                } else {
                    None
                };
                let end = value.as_ref().map(|v| v.span.end).unwrap_or(token.span.end);
                Ok(Expr {
                    kind: ExprKind::Return(value),
                    span: Span::new(start, end),
                })
            }
            TokenKind::KwBreak => {
                self.stream.advance();
                Ok(Expr {
                    kind: ExprKind::Break,
                    span: token.span,
                })
            }
            TokenKind::KwContinue => {
                self.stream.advance();
                Ok(Expr {
                    kind: ExprKind::Continue,
                    span: token.span,
                })
            }
            TokenKind::KwThrow => {
                self.stream.advance();
                let expr = self.parse_expression()?;
                let end = expr.span.end;
                Ok(Expr {
                    kind: ExprKind::Throw(Box::new(expr)),
                    span: Span::new(start, end),
                })
            }
            TokenKind::KwVar => {
                self.stream.advance();
                let name = self.stream.current_text().to_string();
                self.stream.advance();
                let type_hint = if self.stream.eat(TokenKind::Colon).is_some() {
                    Some(self.parse_type()?)
                } else {
                    None
                };
                let expr = if self.stream.eat(TokenKind::Assign).is_some() {
                    Some(Box::new(self.parse_expression()?))
                } else {
                    None
                };
                let end = expr
                    .as_ref()
                    .map(|e| e.span.end)
                    .unwrap_or(self.stream.current_offset());
                Ok(Expr {
                    kind: ExprKind::Var {
                        name,
                        type_hint,
                        expr,
                    },
                    span: Span::new(start, end),
                })
            }
            TokenKind::KwFinal => {
                self.stream.advance();
                let name = self.stream.current_text().to_string();
                self.stream.advance();
                let type_hint = if self.stream.eat(TokenKind::Colon).is_some() {
                    Some(self.parse_type()?)
                } else {
                    None
                };
                let expr = if self.stream.eat(TokenKind::Assign).is_some() {
                    Some(Box::new(self.parse_expression()?))
                } else {
                    None
                };
                let end = expr
                    .as_ref()
                    .map(|e| e.span.end)
                    .unwrap_or(self.stream.current_offset());
                Ok(Expr {
                    kind: ExprKind::Final {
                        name,
                        type_hint,
                        expr,
                    },
                    span: Span::new(start, end),
                })
            }
            TokenKind::KwDo => self.parse_do_while_expr(),
            TokenKind::KwSwitch => self.parse_switch_expr(),
            TokenKind::KwTry => self.parse_try_expr(),
            TokenKind::KwCast => self.parse_cast_expr(),
            TokenKind::KwUntyped => {
                self.stream.advance();
                let expr = self.parse_expression()?;
                let end = expr.span.end;
                Ok(Expr {
                    kind: ExprKind::Untyped(Box::new(expr)),
                    span: Span::new(start, end),
                })
            }
            TokenKind::KwMacro => {
                // Reification: `macro expr` produces an AST Expr at compile time.
                // Examples: `macro null`, `macro $v{x}`, `macro return $v{x}`,
                //           `macro [$a{arr}]`, `macro :Int` (type reification).
                self.stream.advance();
                // Type reification: `macro :Type` — uncommon but legal.
                // Represented as a special Macro-wrapped type expression. We
                // parse the type and wrap it as a Macro-wrapped Ident for now,
                // since the reification engine only needs the token span.
                if self.stream.at(TokenKind::Colon) {
                    self.stream.advance();
                    let _ty = self.parse_type()?;
                    let end = self.stream.current_offset();
                    // Emit a placeholder that the reification engine can
                    // recognize. The original source span preserves the info.
                    return Ok(Expr {
                        kind: ExprKind::Macro(Box::new(Expr {
                            kind: ExprKind::Ident("__macro_type__".to_string()),
                            span: Span::new(start, end),
                        })),
                        span: Span::new(start, end),
                    });
                }
                let inner = self.parse_expression()?;
                let end = inner.span.end;
                Ok(Expr {
                    kind: ExprKind::Macro(Box::new(inner)),
                    span: Span::new(start, end),
                })
            }
            TokenKind::KwFunction => self.parse_function_literal(),
            TokenKind::DollarIdent => {
                let text = token.text(self.source).to_string();
                self.stream.advance();
                // $v{expr} — dollar identifier with braced argument
                let arg = if self.stream.at(TokenKind::LBrace) {
                    self.stream.advance();
                    let arg_expr = self.parse_expression()?;
                    self.stream.expect(TokenKind::RBrace)?;
                    Some(Box::new(arg_expr))
                } else {
                    None
                };
                let end = self.stream.current_offset();
                Ok(Expr {
                    kind: ExprKind::DollarIdent {
                        name: text[1..].to_string(),
                        arg,
                    },
                    span: Span::new(start, end),
                })
            }
            TokenKind::RegexLit => {
                let text = token.text(self.source);
                self.stream.advance();
                // Parse ~/pattern/flags
                let inner = &text[2..]; // skip ~/
                let last_slash = inner.rfind('/').unwrap_or(inner.len());
                let pattern = inner[..last_slash].to_string();
                let flags = inner[last_slash + 1..].to_string();
                Ok(Expr {
                    kind: ExprKind::Regex { pattern, flags },
                    span: token.span,
                })
            }
            TokenKind::LParen => self.parse_paren_or_arrow(start),
            TokenKind::LBracket => self.parse_array_or_map(),
            TokenKind::LBrace => self.parse_block_or_object(),
            _ => Err(ParseError::new(
                &format!("expected expression, found '{}'", token.text(self.source)),
                token.span,
            )),
        }
    }

    fn parse_if_expr(&mut self) -> Result<Expr, ParseError> {
        let start = self.stream.current_offset();
        self.stream.expect(TokenKind::KwIf)?;
        self.stream.expect(TokenKind::LParen)?;
        let cond = self.parse_expression()?;
        self.stream.expect(TokenKind::RParen)?;
        let then_branch = self.parse_expression()?;
        // Consume optional semicolon before else (common in blocks)
        self.stream.eat(TokenKind::Semicolon);
        let else_branch = if self.stream.eat(TokenKind::KwElse).is_some() {
            Some(Box::new(self.parse_expression()?))
        } else {
            None
        };
        let end = else_branch
            .as_ref()
            .map(|e| e.span.end)
            .unwrap_or(then_branch.span.end);
        Ok(Expr {
            kind: ExprKind::If {
                cond: Box::new(cond),
                then_branch: Box::new(then_branch),
                else_branch,
            },
            span: Span::new(start, end),
        })
    }

    fn parse_while_expr(&mut self) -> Result<Expr, ParseError> {
        let start = self.stream.current_offset();
        self.stream.expect(TokenKind::KwWhile)?;
        self.stream.expect(TokenKind::LParen)?;
        let cond = self.parse_expression()?;
        self.stream.expect(TokenKind::RParen)?;
        let body = self.parse_expression()?;
        let end = body.span.end;
        Ok(Expr {
            kind: ExprKind::While {
                cond: Box::new(cond),
                body: Box::new(body),
            },
            span: Span::new(start, end),
        })
    }

    fn parse_for_expr(&mut self) -> Result<Expr, ParseError> {
        let start = self.stream.current_offset();
        self.stream.expect(TokenKind::KwFor)?;
        self.stream.expect(TokenKind::LParen)?;
        let var_name = self.stream.current_text().to_string();
        self.stream.advance();

        // Check for key => value syntax: `for (key => value in map)`
        let key_var = if self.stream.eat(TokenKind::FatArrow).is_some() {
            let key = var_name.clone();
            let value_name = self.stream.current_text().to_string();
            self.stream.advance();
            // var_name becomes value, key_var is the key
            self.stream.expect(TokenKind::KwIn)?;
            let iter = self.parse_expression()?;
            self.stream.expect(TokenKind::RParen)?;
            let body = self.parse_expression()?;
            let end = body.span.end;
            return Ok(Expr {
                kind: ExprKind::For {
                    var: value_name,
                    key_var: Some(key),
                    iter: Box::new(iter),
                    body: Box::new(body),
                },
                span: Span::new(start, end),
            });
        } else {
            None
        };

        self.stream.expect(TokenKind::KwIn)?;
        let iter = self.parse_expression()?;
        self.stream.expect(TokenKind::RParen)?;
        let body = self.parse_expression()?;
        let end = body.span.end;
        Ok(Expr {
            kind: ExprKind::For {
                var: var_name,
                key_var,
                iter: Box::new(iter),
                body: Box::new(body),
            },
            span: Span::new(start, end),
        })
    }

    fn parse_switch_expr(&mut self) -> Result<Expr, ParseError> {
        let start = self.stream.current_offset();
        self.stream.expect(TokenKind::KwSwitch)?;
        self.stream.expect(TokenKind::LParen)?;
        let scrutinee = self.parse_expression()?;
        // Handle inline type check: `switch (cast this : Int) { ... }`
        let scrutinee = if self.stream.eat(TokenKind::Colon).is_some() {
            let ty = self.parse_type()?;
            let span = scrutinee.span.merge(ty.span());
            Expr {
                kind: ExprKind::TypeCheck {
                    expr: Box::new(scrutinee),
                    type_hint: ty,
                },
                span,
            }
        } else {
            scrutinee
        };
        self.stream.expect(TokenKind::RParen)?;
        self.stream.expect(TokenKind::LBrace)?;

        let mut cases = Vec::new();
        let mut default = None;

        while !self.stream.at(TokenKind::RBrace) && !self.stream.is_eof() {
            if self.stream.eat(TokenKind::KwCase).is_some() {
                let case_start = self.stream.current_offset();
                let pattern = self.parse_case_pattern()?;
                // Guard: `case v if (condition):`
                let guard = if self.stream.eat(TokenKind::KwIf).is_some() {
                    self.stream.expect(TokenKind::LParen)?;
                    let g = self.parse_expression()?;
                    self.stream.expect(TokenKind::RParen)?;
                    Some(g)
                } else {
                    None
                };
                self.stream.expect(TokenKind::Colon)?;
                let mut body_exprs = Vec::new();
                while !self.stream.at(TokenKind::KwCase)
                    && !self.stream.at(TokenKind::KwDefault)
                    && !self.stream.at(TokenKind::RBrace)
                    && !self.stream.is_eof()
                {
                    body_exprs.push(self.parse_expression()?);
                    self.stream.eat(TokenKind::Semicolon);
                }
                // Wrap body in block if multiple statements
                let body = if body_exprs.len() == 1 {
                    body_exprs.pop().unwrap()
                } else {
                    let elements: Vec<BlockElement> =
                        body_exprs.into_iter().map(BlockElement::Expr).collect();
                    let span = self.stream.span_from(case_start);
                    Expr {
                        kind: ExprKind::Block(elements),
                        span,
                    }
                };
                cases.push(Case {
                    patterns: vec![pattern],
                    guard,
                    body,
                    span: self.stream.span_from(case_start),
                });
            } else if self.stream.eat(TokenKind::KwDefault).is_some() {
                self.stream.expect(TokenKind::Colon)?;
                let mut body_exprs = Vec::new();
                while !self.stream.at(TokenKind::RBrace) && !self.stream.is_eof() {
                    body_exprs.push(self.parse_expression()?);
                    self.stream.eat(TokenKind::Semicolon);
                }
                let def_body = if body_exprs.len() == 1 {
                    body_exprs.pop().unwrap()
                } else {
                    let elements: Vec<BlockElement> =
                        body_exprs.into_iter().map(BlockElement::Expr).collect();
                    Expr {
                        kind: ExprKind::Block(elements),
                        span: self.stream.span_from(start),
                    }
                };
                default = Some(Box::new(def_body));
            } else {
                self.stream.advance();
            }
        }

        let end = self.stream.expect(TokenKind::RBrace)?;
        Ok(Expr {
            kind: ExprKind::Switch {
                expr: Box::new(scrutinee),
                cases,
                default,
            },
            span: Span::new(start, end.end),
        })
    }

    fn parse_try_expr(&mut self) -> Result<Expr, ParseError> {
        let start = self.stream.current_offset();
        self.stream.expect(TokenKind::KwTry)?;
        let body = self.parse_expression()?;

        let mut catches = Vec::new();
        while self.stream.eat(TokenKind::KwCatch).is_some() {
            let catch_start = self.stream.current_offset();
            self.stream.expect(TokenKind::LParen)?;
            let var_name = self.stream.current_text().to_string();
            self.stream.advance();
            let type_hint = if self.stream.eat(TokenKind::Colon).is_some() {
                Some(self.parse_type()?)
            } else {
                None
            };
            self.stream.expect(TokenKind::RParen)?;
            // Parse optional catch guard: catch (e:Type) if (condition)
            let filter = if self.stream.at(TokenKind::KwIf) {
                self.stream.advance(); // skip 'if'
                Some(self.parse_expression()?)
            } else {
                None
            };
            let catch_body = self.parse_expression()?;
            catches.push(Catch {
                var: var_name,
                type_hint,
                filter,
                body: catch_body,
                span: self.stream.span_from(catch_start),
            });
        }

        let end = catches.last().map(|c| c.span.end).unwrap_or(body.span.end);
        Ok(Expr {
            kind: ExprKind::Try {
                expr: Box::new(body),
                catches,
                finally_block: None,
            },
            span: Span::new(start, end),
        })
    }

    fn parse_cast_expr(&mut self) -> Result<Expr, ParseError> {
        let start = self.stream.current_offset();
        self.stream.expect(TokenKind::KwCast)?;
        if self.stream.at(TokenKind::LParen) {
            self.stream.advance();
            let expr = self.parse_expression()?;
            let type_hint = if self.stream.eat(TokenKind::Comma).is_some() {
                Some(self.parse_type()?)
            } else {
                None
            };
            self.stream.expect(TokenKind::RParen)?;
            Ok(Expr {
                kind: ExprKind::Cast {
                    expr: Box::new(expr),
                    type_hint,
                },
                span: self.stream.span_from(start),
            })
        } else {
            let expr = self.parse_unary()?;
            Ok(Expr {
                kind: ExprKind::Cast {
                    expr: Box::new(expr),
                    type_hint: None,
                },
                span: self.stream.span_from(start),
            })
        }
    }

    fn parse_function_literal(&mut self) -> Result<Expr, ParseError> {
        let func = self.parse_function_decl()?;
        Ok(Expr {
            span: func.span,
            kind: ExprKind::Function(func),
        })
    }

    /// Parse `(...)` — paren expr, type check, or arrow function.
    /// Uses save/restore for disambiguation.
    fn parse_paren_or_arrow(&mut self, start: usize) -> Result<Expr, ParseError> {
        self.stream.advance(); // skip (

        // Empty: () -> expr
        if self.stream.at(TokenKind::RParen) {
            self.stream.advance();
            if self.stream.eat(TokenKind::Arrow).is_some() {
                let body = self.parse_expression()?;
                let end = body.span.end;
                return Ok(Expr {
                    kind: ExprKind::Arrow {
                        params: Vec::new(),
                        expr: Box::new(body),
                    },
                    span: Span::new(start, end),
                });
            }
            return Ok(Expr {
                kind: ExprKind::Block(Vec::new()),
                span: self.stream.span_from(start),
            });
        }

        // Try arrow function: save position, attempt to parse params
        let saved = self.stream.save();
        if let Ok(params) = self.try_parse_arrow_params() {
            if self.stream.eat(TokenKind::RParen).is_some()
                && self.stream.eat(TokenKind::Arrow).is_some()
            {
                let body = self.parse_expression()?;
                let end = body.span.end;
                return Ok(Expr {
                    kind: ExprKind::Arrow {
                        params,
                        expr: Box::new(body),
                    },
                    span: Span::new(start, end),
                });
            }
        }
        // Restore and parse as regular paren expression
        self.stream.restore(saved);

        let inner = self.parse_expression()?;
        // Type check: (expr : Type)
        if self.stream.eat(TokenKind::Colon).is_some() {
            let ty = self.parse_type()?;
            let end = self.stream.expect(TokenKind::RParen)?;
            return Ok(Expr {
                kind: ExprKind::TypeCheck {
                    expr: Box::new(inner),
                    type_hint: ty,
                },
                span: Span::new(start, end.end),
            });
        }
        let end = self.stream.expect(TokenKind::RParen)?;
        // Arrow after close paren: (x) -> expr
        if self.stream.eat(TokenKind::Arrow).is_some() {
            let body = self.parse_expression()?;
            let end = body.span.end;
            let params = match &inner.kind {
                ExprKind::Ident(name) => vec![ArrowParam {
                    name: name.clone(),
                    type_hint: None,
                }],
                _ => vec![],
            };
            return Ok(Expr {
                kind: ExprKind::Arrow {
                    params,
                    expr: Box::new(body),
                },
                span: Span::new(start, end),
            });
        }
        Ok(Expr {
            kind: ExprKind::Paren(Box::new(inner)),
            span: Span::new(start, end.end),
        })
    }

    /// Try to parse arrow function parameters: `(a:Int, b:String)`
    fn try_parse_arrow_params(&mut self) -> Result<Vec<ArrowParam>, ParseError> {
        let mut params = Vec::new();
        loop {
            if self.stream.at(TokenKind::RParen) {
                break;
            }
            let name = self.stream.current_text().to_string();
            if !self.stream.at(TokenKind::Ident) {
                return Err(ParseError::new(
                    "expected parameter name",
                    self.stream.peek().span,
                ));
            }
            self.stream.advance();
            let type_hint = if self.stream.eat(TokenKind::Colon).is_some() {
                Some(self.parse_type()?)
            } else {
                None
            };
            params.push(ArrowParam { name, type_hint });
            if !self.stream.at(TokenKind::RParen) && self.stream.eat(TokenKind::Comma).is_none() {
                return Err(ParseError::new("expected , or )", self.stream.peek().span));
            }
        }
        Ok(params)
    }

    /// Parse string interpolation parts from single-quoted string content.
    /// Handles `$ident` and `${expr}` interpolation.
    fn parse_string_interpolation_parts(
        &mut self,
        inner: &str,
        base_offset: usize,
    ) -> Vec<StringPart> {
        let mut parts = Vec::new();
        let mut current = String::new();
        let bytes = inner.as_bytes();
        let mut i = 0;

        while i < bytes.len() {
            if bytes[i] == b'$' && i + 1 < bytes.len() {
                // Flush literal
                if !current.is_empty() {
                    parts.push(StringPart::Literal(std::mem::take(&mut current)));
                }

                if bytes[i + 1] == b'{' {
                    // ${expr} — find matching }
                    i += 2;
                    let expr_start = i;
                    let mut depth = 1;
                    while i < bytes.len() && depth > 0 {
                        if bytes[i] == b'{' {
                            depth += 1;
                        }
                        if bytes[i] == b'}' {
                            depth -= 1;
                        }
                        if depth > 0 {
                            i += 1;
                        }
                    }
                    let expr_str = &inner[expr_start..i];
                    if i < bytes.len() {
                        i += 1;
                    } // skip }

                    // Parse the expression inside ${}
                    if let Ok(file) = crate::rd::rd_parse(
                        &format!(
                            "class _ {{ static function _() {{ return {}; }} }}",
                            expr_str
                        ),
                        "<interp>",
                        false,
                        false,
                    ) {
                        if let Some(TypeDeclaration::Class(c)) = file.declarations.first() {
                            if let Some(ClassFieldKind::Function(f)) =
                                c.fields.first().map(|f| &f.kind)
                            {
                                if let Some(body) = &f.body {
                                    if let ExprKind::Block(elements) = &body.kind {
                                        if let Some(BlockElement::Expr(Expr {
                                            kind: ExprKind::Return(Some(expr)),
                                            ..
                                        })) = elements.first()
                                        {
                                            parts.push(StringPart::Interpolation(*expr.clone()));
                                            continue;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    // Fallback: treat as identifier
                    parts.push(StringPart::Interpolation(Expr {
                        kind: ExprKind::Ident(expr_str.to_string()),
                        span: Span::new(
                            base_offset + expr_start,
                            base_offset + expr_start + expr_str.len(),
                        ),
                    }));
                } else if bytes[i + 1].is_ascii_alphabetic() || bytes[i + 1] == b'_' {
                    // $ident
                    i += 1;
                    let ident_start = i;
                    while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_')
                    {
                        i += 1;
                    }
                    let ident = &inner[ident_start..i];
                    parts.push(StringPart::Interpolation(Expr {
                        kind: ExprKind::Ident(ident.to_string()),
                        span: Span::new(base_offset + ident_start, base_offset + i),
                    }));
                } else {
                    // Just a $ followed by non-ident char
                    current.push('$');
                    i += 1;
                }
            } else if bytes[i] == b'\\' && i + 1 < bytes.len() {
                // Escape sequence
                current.push(bytes[i] as char);
                current.push(bytes[i + 1] as char);
                i += 2;
            } else {
                current.push(bytes[i] as char);
                i += 1;
            }
        }

        if !current.is_empty() {
            parts.push(StringPart::Literal(current));
        }

        parts
    }

    /// Parse a switch case pattern.
    fn parse_case_pattern(&mut self) -> Result<Pattern, ParseError> {
        match self.stream.peek().kind {
            // Underscore wildcard: `case _:`
            TokenKind::Ident if self.stream.current_text() == "_" => {
                self.stream.advance();
                Ok(Pattern::Underscore)
            }
            // Null pattern
            TokenKind::KwNull => {
                self.stream.advance();
                Ok(Pattern::Null)
            }
            // Literal patterns
            TokenKind::IntLit => {
                let expr = self.parse_primary()?;
                Ok(Pattern::Const(expr))
            }
            TokenKind::FloatLit => {
                let expr = self.parse_primary()?;
                Ok(Pattern::Const(expr))
            }
            TokenKind::StringLit => {
                let expr = self.parse_primary()?;
                Ok(Pattern::Const(expr))
            }
            TokenKind::KwTrue | TokenKind::KwFalse => {
                let expr = self.parse_primary()?;
                Ok(Pattern::Const(expr))
            }
            TokenKind::Minus => {
                // Negative number literal: -42
                let expr = self.parse_unary()?;
                Ok(Pattern::Const(expr))
            }
            // Array pattern: [a, b, c]
            TokenKind::LBracket => {
                self.stream.advance();
                let mut elements = Vec::new();
                while !self.stream.at(TokenKind::RBracket) && !self.stream.is_eof() {
                    elements.push(self.parse_case_pattern()?);
                    if !self.stream.at(TokenKind::RBracket) {
                        self.stream.eat(TokenKind::Comma);
                    }
                }
                self.stream.expect(TokenKind::RBracket)?;
                Ok(Pattern::Array(elements))
            }
            // Identifier: variable capture or constructor
            TokenKind::Ident => {
                let name = self.stream.current_text().to_string();
                self.stream.advance();

                // Check for constructor pattern: Name(args) or Name.Variant
                if self.stream.at(TokenKind::LParen) {
                    // Constructor: Some(v) or MkPair(a, b)
                    self.stream.advance();
                    let mut params = Vec::new();
                    while !self.stream.at(TokenKind::RParen) && !self.stream.is_eof() {
                        params.push(self.parse_case_pattern()?);
                        if !self.stream.at(TokenKind::RParen) {
                            self.stream.eat(TokenKind::Comma);
                        }
                    }
                    self.stream.expect(TokenKind::RParen)?;
                    Ok(Pattern::Constructor {
                        path: TypePath {
                            package: Vec::new(),
                            name,
                            sub: None,
                        },
                        params,
                    })
                } else if self.stream.at(TokenKind::Dot) {
                    // Qualified: Enum.Value or Enum.Value(args)
                    let mut parts = vec![name];
                    while self.stream.eat(TokenKind::Dot).is_some() {
                        parts.push(self.stream.current_text().to_string());
                        self.stream.advance();
                    }
                    let final_name = parts.pop().unwrap();
                    if self.stream.at(TokenKind::LParen) {
                        self.stream.advance();
                        let mut params = Vec::new();
                        while !self.stream.at(TokenKind::RParen) && !self.stream.is_eof() {
                            params.push(self.parse_case_pattern()?);
                            if !self.stream.at(TokenKind::RParen) {
                                self.stream.eat(TokenKind::Comma);
                            }
                        }
                        self.stream.expect(TokenKind::RParen)?;
                        Ok(Pattern::Constructor {
                            path: TypePath {
                                package: parts,
                                name: final_name,
                                sub: None,
                            },
                            params,
                        })
                    } else {
                        // Qualified constant: Enum.Value
                        let path_str = parts.join(".");
                        Ok(Pattern::Const(Expr {
                            kind: ExprKind::Field {
                                expr: Box::new(Expr {
                                    kind: ExprKind::Ident(path_str),
                                    span: Span::default(),
                                }),
                                field: final_name,
                                is_optional: false,
                            },
                            span: Span::default(),
                        }))
                    }
                } else {
                    // Simple identifier — variable capture
                    Ok(Pattern::Var(name))
                }
            }
            _ => {
                // Fallback: parse as expression and wrap as Const
                let expr = self.parse_expression()?;
                Ok(Pattern::Const(expr))
            }
        }
    }

    fn parse_do_while_expr(&mut self) -> Result<Expr, ParseError> {
        let start = self.stream.current_offset();
        self.stream.expect(TokenKind::KwDo)?;
        let body = self.parse_expression()?;
        self.stream.expect(TokenKind::KwWhile)?;
        self.stream.expect(TokenKind::LParen)?;
        let cond = self.parse_expression()?;
        self.stream.expect(TokenKind::RParen)?;
        let end = cond.span.end;
        Ok(Expr {
            kind: ExprKind::DoWhile {
                body: Box::new(body),
                cond: Box::new(cond),
            },
            span: Span::new(start, end),
        })
    }

    /// Parse `[...]` — array literal or map literal (if contains `=>`)
    fn parse_array_or_map(&mut self) -> Result<Expr, ParseError> {
        let start = self.stream.current_offset();
        self.stream.expect(TokenKind::LBracket)?;

        if self.stream.at(TokenKind::RBracket) {
            let end = self.stream.expect(TokenKind::RBracket)?;
            return Ok(Expr {
                kind: ExprKind::Array(Vec::new()),
                span: Span::new(start, end.end),
            });
        }

        // Parse first element to check for `=>`
        let first = self.parse_expression()?;

        if self.stream.eat(TokenKind::FatArrow).is_some() {
            // Map literal: [key => value, ...]
            let first_val = self.parse_expression()?;
            let mut pairs = vec![(first, first_val)];
            while self.stream.eat(TokenKind::Comma).is_some()
                && !self.stream.at(TokenKind::RBracket)
            {
                let key = self.parse_expression()?;
                self.stream.expect(TokenKind::FatArrow)?;
                let val = self.parse_expression()?;
                pairs.push((key, val));
            }
            let end = self.stream.expect(TokenKind::RBracket)?;
            Ok(Expr {
                kind: ExprKind::Map(pairs),
                span: Span::new(start, end.end),
            })
        } else {
            // Array literal
            let mut items = vec![first];
            while self.stream.eat(TokenKind::Comma).is_some()
                && !self.stream.at(TokenKind::RBracket)
            {
                items.push(self.parse_expression()?);
            }
            let end = self.stream.expect(TokenKind::RBracket)?;
            Ok(Expr {
                kind: ExprKind::Array(items),
                span: Span::new(start, end.end),
            })
        }
    }

    /// Parse `{...}` — block or object literal.
    /// Object if: `{ ident: expr }` or `{ }` (empty)
    fn parse_block_or_object(&mut self) -> Result<Expr, ParseError> {
        let start = self.stream.current_offset();

        // Empty braces = empty block
        if self.stream.peek_at(0).kind == TokenKind::LBrace
            && self.stream.peek_at(1).kind == TokenKind::RBrace
        {
            self.stream.advance(); // {
            let end = self.stream.expect(TokenKind::RBrace)?;
            return Ok(Expr {
                kind: ExprKind::Block(Vec::new()),
                span: Span::new(start, end.end),
            });
        }

        // Check if it's an object literal: { ident: ... } or { "string": ... }
        let is_object = self.stream.peek_at(0).kind == TokenKind::LBrace
            && (self.stream.peek_at(1).kind == TokenKind::Ident
                || self.stream.peek_at(1).kind == TokenKind::StringLit)
            && self.stream.peek_at(2).kind == TokenKind::Colon;

        if is_object {
            self.stream.advance(); // {
            let mut fields = Vec::new();
            while !self.stream.at(TokenKind::RBrace) && !self.stream.is_eof() {
                let field_name = self.stream.current_text().to_string();
                // Strip quotes for string keys
                let field_name = if field_name.starts_with('"') || field_name.starts_with('\'') {
                    field_name[1..field_name.len() - 1].to_string()
                } else {
                    field_name
                };
                self.stream.advance();
                self.stream.expect(TokenKind::Colon)?;
                let value = self.parse_expression()?;
                fields.push(ObjectField {
                    name: field_name,
                    expr: value,
                    span: self.stream.span_from(start),
                });
                if !self.stream.at(TokenKind::RBrace) {
                    self.stream.eat(TokenKind::Comma);
                }
            }
            let end = self.stream.expect(TokenKind::RBrace)?;
            return Ok(Expr {
                kind: ExprKind::Object(fields),
                span: Span::new(start, end.end),
            });
        }

        self.parse_block_expr()
    }

    fn parse_block_expr(&mut self) -> Result<Expr, ParseError> {
        let start = self.stream.current_offset();
        self.stream.expect(TokenKind::LBrace)?;
        let mut elements = Vec::new();
        while !self.stream.at(TokenKind::RBrace) && !self.stream.is_eof() {
            elements.push(BlockElement::Expr(self.parse_expression()?));
            self.stream.eat(TokenKind::Semicolon);
        }
        let end = self.stream.expect(TokenKind::RBrace)?;
        Ok(Expr {
            kind: ExprKind::Block(elements),
            span: Span::new(start, end.end),
        })
    }

    /// Parse a type path for `new` expressions
    fn parse_type_path(&mut self) -> Result<TypePath, ParseError> {
        let mut package = Vec::new();
        let mut name = self.stream.current_text().to_string();
        self.stream.advance();

        while self.stream.at(TokenKind::Dot) {
            if self.stream.peek_at(1).kind == TokenKind::Ident
                || self.stream.peek_at(1).kind.is_keyword()
            {
                self.stream.advance(); // skip dot
                package.push(name);
                name = self.stream.current_text().to_string();
                self.stream.advance();
            } else {
                break;
            }
        }

        Ok(TypePath {
            package,
            name,
            sub: None,
        })
    }

    fn parse_type_args(&mut self) -> Result<Vec<Type>, ParseError> {
        self.stream.expect(TokenKind::Lt)?;
        let mut args = Vec::new();
        while !self.stream.at(TokenKind::Gt) && !self.stream.is_eof() {
            args.push(self.parse_type()?);
            if !self.stream.at(TokenKind::Gt) {
                self.stream.eat(TokenKind::Comma);
            }
        }
        self.stream.expect(TokenKind::Gt)?;
        Ok(args)
    }
}
