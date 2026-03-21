//! Type expression parser for the recursive descent parser.

use crate::haxe_ast::*;
use crate::token::TokenKind;
use super::RdParser;
use super::error::ParseError;

impl<'a, 'b> RdParser<'a, 'b> {
    /// Parse a type expression.
    pub fn parse_type(&mut self) -> Result<Type, ParseError> {
        let left = self.parse_basic_type()?;

        // Check for function type: `Int -> String -> Void`
        if self.stream.at(TokenKind::Arrow) {
            let mut params = vec![left];
            while self.stream.eat(TokenKind::Arrow).is_some() {
                params.push(self.parse_basic_type()?);
            }
            let ret = params.pop().unwrap();
            let span = params
                .first()
                .map(|p| p.span())
                .unwrap_or(ret.span())
                .merge(ret.span());
            return Ok(Type::Function {
                params,
                ret: Box::new(ret),
                span,
            });
        }

        // Check for intersection type: `Type & { extraField: Int }`
        if self.stream.at(TokenKind::Amp) {
            self.stream.advance();
            let right = self.parse_basic_type()?;
            let span = left.span().merge(right.span());
            return Ok(Type::Intersection {
                left: Box::new(left),
                right: Box::new(right),
                span,
            });
        }

        Ok(left)
    }

    fn parse_basic_type(&mut self) -> Result<Type, ParseError> {
        let start = self.stream.current_offset();

        // Optional type: `?Int`
        if self.stream.eat(TokenKind::Question).is_some() {
            let inner = self.parse_basic_type()?;
            let span = Span::new(start, inner.span().end);
            return Ok(Type::Optional {
                inner: Box::new(inner),
                span,
            });
        }

        // Parenthesized type: `(Int)`
        if self.stream.at(TokenKind::LParen) {
            self.stream.advance();
            let inner = self.parse_type()?;
            self.stream.expect(TokenKind::RParen)?;
            let span = self.stream.span_from(start);
            return Ok(Type::Parenthesis {
                inner: Box::new(inner),
                span,
            });
        }

        // Anonymous structure: `{ x:Int, y:String }`
        if self.stream.at(TokenKind::LBrace) {
            return self.parse_anonymous_type(start);
        }

        // Named type: `Int`, `Array<Int>`, `com.example.MyClass`
        self.parse_named_type(start)
    }

    fn parse_named_type(&mut self, start: usize) -> Result<Type, ParseError> {
        let mut package = Vec::new();

        // Parse the first identifier
        let first = self.stream.current_text().to_string();
        if !self.stream.at(TokenKind::Ident) && !self.stream.peek().kind.is_keyword() {
            // Special case: `Dynamic` is sometimes used as a type
            return Err(ParseError::new(
                &format!("expected type, found '{}'", first),
                self.stream.peek().span,
            ));
        }
        self.stream.advance();

        // Check for dotted path: `com.example.MyClass`
        let mut name = first;
        while self.stream.at(TokenKind::Dot) && !self.stream.is_eof() {
            // Peek ahead to see if next is an identifier (not a field access)
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

        // Check for type parameters: `Array<Int>`
        let params = if self.stream.at(TokenKind::Lt) {
            self.parse_type_param_args()?
        } else {
            Vec::new()
        };

        let span = self.stream.span_from(start);

        Ok(Type::Path {
            path: TypePath {
                package,
                name,
                sub: None,
            },
            params,
            span,
        })
    }

    fn parse_anonymous_type(&mut self, start: usize) -> Result<Type, ParseError> {
        self.stream.expect(TokenKind::LBrace)?;
        let mut fields = Vec::new();

        while !self.stream.at(TokenKind::RBrace) && !self.stream.is_eof() {
            let field_start = self.stream.current_offset();
            let optional = self.stream.eat(TokenKind::Question).is_some();
            let field_name = self.stream.current_text().to_string();
            self.stream.advance();
            self.stream.expect(TokenKind::Colon)?;
            let type_hint = self.parse_type()?;

            fields.push(AnonField {
                name: field_name,
                optional,
                type_hint,
                span: self.stream.span_from(field_start),
            });

            if !self.stream.at(TokenKind::RBrace) {
                self.stream.eat(TokenKind::Comma);
            }
        }

        let end_span = self.stream.expect(TokenKind::RBrace)?;

        Ok(Type::Anonymous {
            fields,
            span: Span::new(start, end_span.end),
        })
    }

    fn parse_type_param_args(&mut self) -> Result<Vec<Type>, ParseError> {
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

    /// Parse type parameter declarations: `<T, U:Constraint>`
    pub fn parse_type_params(&mut self) -> Result<Vec<TypeParam>, ParseError> {
        if !self.stream.at(TokenKind::Lt) {
            return Ok(Vec::new());
        }
        self.stream.advance();

        let mut params = Vec::new();
        while !self.stream.at(TokenKind::Gt) && !self.stream.is_eof() {
            let param_start = self.stream.current_offset();
            let name = self.stream.current_text().to_string();
            self.stream.advance();

            let mut constraints = Vec::new();
            if self.stream.eat(TokenKind::Colon).is_some() {
                constraints.push(self.parse_type()?);
            }

            params.push(TypeParam {
                name,
                constraints,
                variance: Variance::Invariant,
                span: self.stream.span_from(param_start),
            });

            if !self.stream.at(TokenKind::Gt) {
                self.stream.eat(TokenKind::Comma);
            }
        }

        self.stream.expect(TokenKind::Gt)?;
        Ok(params)
    }
}

// Helper to get span from Type
impl Type {
    pub fn span(&self) -> Span {
        match self {
            Type::Path { span, .. } => *span,
            Type::Function { span, .. } => *span,
            Type::Anonymous { span, .. } => *span,
            Type::Optional { span, .. } => *span,
            Type::Parenthesis { span, .. } => *span,
            Type::Intersection { span, .. } => *span,
            Type::Wildcard { span } => *span,
        }
    }
}
