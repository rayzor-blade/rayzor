//! Expression parser using precedence climbing (Pratt parsing).

use crate::haxe_ast::*;
use crate::token::TokenKind;
use super::RdParser;
use super::error::ParseError;

impl<'a, 'b> RdParser<'a, 'b> {
    /// Parse an expression.
    pub fn parse_expression(&mut self) -> Result<Expr, ParseError> {
        self.parse_assignment()
    }

    fn parse_assignment(&mut self) -> Result<Expr, ParseError> {
        let left = self.parse_ternary()?;

        if self.stream.at_any(&[
            TokenKind::Assign, TokenKind::PlusAssign, TokenKind::MinusAssign,
            TokenKind::StarAssign, TokenKind::SlashAssign, TokenKind::PercentAssign,
            TokenKind::AmpAssign, TokenKind::PipeAssign, TokenKind::CaretAssign,
            TokenKind::ShlAssign, TokenKind::ShrAssign, TokenKind::UshrAssign,
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
                    kind: ExprKind::Unary { op: UnaryOp::Not, expr: Box::new(expr) },
                })
            }
            TokenKind::Minus => {
                self.stream.advance();
                let expr = self.parse_unary()?;
                Ok(Expr {
                    span: Span::new(start, expr.span.end),
                    kind: ExprKind::Unary { op: UnaryOp::Neg, expr: Box::new(expr) },
                })
            }
            TokenKind::Tilde => {
                self.stream.advance();
                let expr = self.parse_unary()?;
                Ok(Expr {
                    span: Span::new(start, expr.span.end),
                    kind: ExprKind::Unary { op: UnaryOp::BitNot, expr: Box::new(expr) },
                })
            }
            TokenKind::PlusPlus => {
                self.stream.advance();
                let expr = self.parse_unary()?;
                Ok(Expr {
                    span: Span::new(start, expr.span.end),
                    kind: ExprKind::Unary { op: UnaryOp::PreIncr, expr: Box::new(expr) },
                })
            }
            TokenKind::MinusMinus => {
                self.stream.advance();
                let expr = self.parse_unary()?;
                Ok(Expr {
                    span: Span::new(start, expr.span.end),
                    kind: ExprKind::Unary { op: UnaryOp::PreDecr, expr: Box::new(expr) },
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
                        kind: ExprKind::Field { expr: Box::new(expr), field, is_optional: false },
                    };
                }
                TokenKind::QuestionDot => {
                    self.stream.advance();
                    let field = self.stream.current_text().to_string();
                    let end = self.stream.peek().span.end;
                    self.stream.advance();
                    expr = Expr {
                        span: Span::new(expr.span.start, end),
                        kind: ExprKind::Field { expr: Box::new(expr), field, is_optional: true },
                    };
                }
                TokenKind::LBracket => {
                    self.stream.advance();
                    let index = self.parse_expression()?;
                    let end = self.stream.expect(TokenKind::RBracket)?;
                    expr = Expr {
                        span: Span::new(expr.span.start, end.end),
                        kind: ExprKind::Index { expr: Box::new(expr), index: Box::new(index) },
                    };
                }
                TokenKind::LParen => {
                    let args = self.parse_call_args()?;
                    let end = self.stream.span_from(expr.span.start).end;
                    expr = Expr {
                        span: Span::new(expr.span.start, end),
                        kind: ExprKind::Call { expr: Box::new(expr), args },
                    };
                }
                TokenKind::PlusPlus => {
                    let end = self.stream.peek().span.end;
                    self.stream.advance();
                    expr = Expr {
                        span: Span::new(expr.span.start, end),
                        kind: ExprKind::Unary { op: UnaryOp::PostIncr, expr: Box::new(expr) },
                    };
                }
                TokenKind::MinusMinus => {
                    let end = self.stream.peek().span.end;
                    self.stream.advance();
                    expr = Expr {
                        span: Span::new(expr.span.start, end),
                        kind: ExprKind::Unary { op: UnaryOp::PostDecr, expr: Box::new(expr) },
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
                } else {
                    text.parse::<i64>().unwrap_or(0)
                };
                Ok(Expr { kind: ExprKind::Int(val), span: token.span })
            }
            TokenKind::FloatLit => {
                let text = token.text(self.source);
                self.stream.advance();
                Ok(Expr { kind: ExprKind::Float(text.parse().unwrap_or(0.0)), span: token.span })
            }
            TokenKind::StringLit => {
                let text = token.text(self.source);
                self.stream.advance();
                let inner = &text[1..text.len() - 1];
                Ok(Expr { kind: ExprKind::String(inner.to_string()), span: token.span })
            }
            TokenKind::KwTrue => { self.stream.advance(); Ok(Expr { kind: ExprKind::Bool(true), span: token.span }) }
            TokenKind::KwFalse => { self.stream.advance(); Ok(Expr { kind: ExprKind::Bool(false), span: token.span }) }
            TokenKind::KwNull => { self.stream.advance(); Ok(Expr { kind: ExprKind::Null, span: token.span }) }
            TokenKind::KwThis => { self.stream.advance(); Ok(Expr { kind: ExprKind::This, span: token.span }) }
            TokenKind::KwSuper => { self.stream.advance(); Ok(Expr { kind: ExprKind::Super, span: token.span }) }
            TokenKind::Ident => {
                let name = token.text(self.source).to_string();
                self.stream.advance();
                Ok(Expr { kind: ExprKind::Ident(name), span: token.span })
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
                Ok(Expr { kind: ExprKind::New { type_path: path, params, args }, span: self.stream.span_from(start) })
            }
            TokenKind::KwIf => self.parse_if_expr(),
            TokenKind::KwWhile => self.parse_while_expr(),
            TokenKind::KwFor => self.parse_for_expr(),
            TokenKind::KwReturn => {
                self.stream.advance();
                let value = if !self.stream.at(TokenKind::Semicolon)
                    && !self.stream.at(TokenKind::RBrace) && !self.stream.is_eof()
                {
                    Some(Box::new(self.parse_expression()?))
                } else { None };
                let end = value.as_ref().map(|v| v.span.end).unwrap_or(token.span.end);
                Ok(Expr { kind: ExprKind::Return(value), span: Span::new(start, end) })
            }
            TokenKind::KwBreak => { self.stream.advance(); Ok(Expr { kind: ExprKind::Break, span: token.span }) }
            TokenKind::KwContinue => { self.stream.advance(); Ok(Expr { kind: ExprKind::Continue, span: token.span }) }
            TokenKind::KwThrow => {
                self.stream.advance();
                let expr = self.parse_expression()?;
                let end = expr.span.end;
                Ok(Expr { kind: ExprKind::Throw(Box::new(expr)), span: Span::new(start, end) })
            }
            TokenKind::KwVar => {
                self.stream.advance();
                let name = self.stream.current_text().to_string();
                self.stream.advance();
                let type_hint = if self.stream.eat(TokenKind::Colon).is_some() { Some(self.parse_type()?) } else { None };
                let expr = if self.stream.eat(TokenKind::Assign).is_some() { Some(Box::new(self.parse_expression()?)) } else { None };
                let end = expr.as_ref().map(|e| e.span.end).unwrap_or(self.stream.current_offset());
                Ok(Expr { kind: ExprKind::Var { name, type_hint, expr }, span: Span::new(start, end) })
            }
            TokenKind::KwFinal => {
                self.stream.advance();
                let name = self.stream.current_text().to_string();
                self.stream.advance();
                let type_hint = if self.stream.eat(TokenKind::Colon).is_some() { Some(self.parse_type()?) } else { None };
                let expr = if self.stream.eat(TokenKind::Assign).is_some() { Some(Box::new(self.parse_expression()?)) } else { None };
                let end = expr.as_ref().map(|e| e.span.end).unwrap_or(self.stream.current_offset());
                Ok(Expr { kind: ExprKind::Final { name, type_hint, expr }, span: Span::new(start, end) })
            }
            TokenKind::KwSwitch => self.parse_switch_expr(),
            TokenKind::KwTry => self.parse_try_expr(),
            TokenKind::KwCast => self.parse_cast_expr(),
            TokenKind::KwFunction => self.parse_function_literal(),
            TokenKind::LParen => {
                self.stream.advance();
                let inner = self.parse_expression()?;
                let end = self.stream.expect(TokenKind::RParen)?;
                Ok(Expr { kind: ExprKind::Paren(Box::new(inner)), span: Span::new(start, end.end) })
            }
            TokenKind::LBracket => {
                self.stream.advance();
                let mut items = Vec::new();
                while !self.stream.at(TokenKind::RBracket) && !self.stream.is_eof() {
                    items.push(self.parse_expression()?);
                    if !self.stream.at(TokenKind::RBracket) { self.stream.eat(TokenKind::Comma); }
                }
                let end = self.stream.expect(TokenKind::RBracket)?;
                Ok(Expr { kind: ExprKind::Array(items), span: Span::new(start, end.end) })
            }
            TokenKind::LBrace => self.parse_block_expr(),
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
        let else_branch = if self.stream.eat(TokenKind::KwElse).is_some() {
            Some(Box::new(self.parse_expression()?))
        } else { None };
        let end = else_branch.as_ref().map(|e| e.span.end).unwrap_or(then_branch.span.end);
        Ok(Expr {
            kind: ExprKind::If { cond: Box::new(cond), then_branch: Box::new(then_branch), else_branch },
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
            kind: ExprKind::While { cond: Box::new(cond), body: Box::new(body) },
            span: Span::new(start, end),
        })
    }

    fn parse_for_expr(&mut self) -> Result<Expr, ParseError> {
        let start = self.stream.current_offset();
        self.stream.expect(TokenKind::KwFor)?;
        self.stream.expect(TokenKind::LParen)?;
        let var_name = self.stream.current_text().to_string();
        self.stream.advance();
        self.stream.expect(TokenKind::KwIn)?;
        let iter = self.parse_expression()?;
        self.stream.expect(TokenKind::RParen)?;
        let body = self.parse_expression()?;
        let end = body.span.end;
        Ok(Expr {
            kind: ExprKind::For { var: var_name, key_var: None, iter: Box::new(iter), body: Box::new(body) },
            span: Span::new(start, end),
        })
    }

    fn parse_switch_expr(&mut self) -> Result<Expr, ParseError> {
        let start = self.stream.current_offset();
        self.stream.expect(TokenKind::KwSwitch)?;
        self.stream.expect(TokenKind::LParen)?;
        let scrutinee = self.parse_expression()?;
        self.stream.expect(TokenKind::RParen)?;
        self.stream.expect(TokenKind::LBrace)?;

        let mut cases = Vec::new();
        let mut default = None;

        while !self.stream.at(TokenKind::RBrace) && !self.stream.is_eof() {
            if self.stream.eat(TokenKind::KwCase).is_some() {
                let case_start = self.stream.current_offset();
                let pattern = self.parse_expression()?;
                self.stream.expect(TokenKind::Colon)?;
                let mut body_exprs = Vec::new();
                while !self.stream.at(TokenKind::KwCase) && !self.stream.at(TokenKind::KwDefault)
                    && !self.stream.at(TokenKind::RBrace) && !self.stream.is_eof()
                {
                    body_exprs.push(self.parse_expression()?);
                    self.stream.eat(TokenKind::Semicolon);
                }
                // Wrap body in block if multiple statements
                let body = if body_exprs.len() == 1 {
                    body_exprs.pop().unwrap()
                } else {
                    let elements: Vec<BlockElement> = body_exprs.into_iter().map(BlockElement::Expr).collect();
                    let span = self.stream.span_from(case_start);
                    Expr { kind: ExprKind::Block(elements), span }
                };
                cases.push(Case {
                    patterns: vec![Pattern::Const(pattern)],
                    guard: None,
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
                    let elements: Vec<BlockElement> = body_exprs.into_iter().map(BlockElement::Expr).collect();
                    Expr { kind: ExprKind::Block(elements), span: self.stream.span_from(start) }
                };
                default = Some(Box::new(def_body));
            } else {
                self.stream.advance();
            }
        }

        let end = self.stream.expect(TokenKind::RBrace)?;
        Ok(Expr {
            kind: ExprKind::Switch { expr: Box::new(scrutinee), cases, default },
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
            } else { None };
            self.stream.expect(TokenKind::RParen)?;
            let catch_body = self.parse_expression()?;
            catches.push(Catch {
                var: var_name,
                type_hint,
                filter: None,
                body: catch_body,
                span: self.stream.span_from(catch_start),
            });
        }

        let end = catches.last().map(|c| c.span.end).unwrap_or(body.span.end);
        Ok(Expr {
            kind: ExprKind::Try { expr: Box::new(body), catches, finally_block: None },
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
            } else { None };
            self.stream.expect(TokenKind::RParen)?;
            Ok(Expr {
                kind: ExprKind::Cast { expr: Box::new(expr), type_hint },
                span: self.stream.span_from(start),
            })
        } else {
            let expr = self.parse_unary()?;
            Ok(Expr {
                kind: ExprKind::Cast { expr: Box::new(expr), type_hint: None },
                span: self.stream.span_from(start),
            })
        }
    }

    fn parse_function_literal(&mut self) -> Result<Expr, ParseError> {
        let func = self.parse_function_decl()?;
        Ok(Expr { span: func.span, kind: ExprKind::Function(func) })
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
        Ok(Expr { kind: ExprKind::Block(elements), span: Span::new(start, end.end) })
    }

    /// Parse a type path for `new` expressions
    fn parse_type_path(&mut self) -> Result<TypePath, ParseError> {
        let mut package = Vec::new();
        let mut name = self.stream.current_text().to_string();
        self.stream.advance();

        while self.stream.at(TokenKind::Dot) {
            if self.stream.peek_at(1).kind == TokenKind::Ident || self.stream.peek_at(1).kind.is_keyword() {
                self.stream.advance(); // skip dot
                package.push(name);
                name = self.stream.current_text().to_string();
                self.stream.advance();
            } else {
                break;
            }
        }

        Ok(TypePath { package, name, sub: None })
    }

    fn parse_type_args(&mut self) -> Result<Vec<Type>, ParseError> {
        self.stream.expect(TokenKind::Lt)?;
        let mut args = Vec::new();
        while !self.stream.at(TokenKind::Gt) && !self.stream.is_eof() {
            args.push(self.parse_type()?);
            if !self.stream.at(TokenKind::Gt) { self.stream.eat(TokenKind::Comma); }
        }
        self.stream.expect(TokenKind::Gt)?;
        Ok(args)
    }
}
