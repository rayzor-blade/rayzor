//! Token stream cursor for the recursive descent parser.
//!
//! Provides peek/advance/expect/eat operations over a flat token array.

use crate::haxe_ast::Span;
use crate::token::{Token, TokenKind};

/// Cursor over a token stream for the parser to consume.
pub struct TokenStream<'a> {
    tokens: &'a [Token],
    pos: usize,
    source: &'a str,
    /// The unconsumed suffix of a token split in a type context.
    remainder: Option<Token>,
    consumed_remainder: Option<Token>,
}

#[derive(Clone)]
pub struct StreamPosition {
    pos: usize,
    remainder: Option<Token>,
}

impl<'a> TokenStream<'a> {
    pub fn new(tokens: &'a [Token], source: &'a str) -> Self {
        Self {
            tokens,
            pos: 0,
            source,
            remainder: None,
            consumed_remainder: None,
        }
    }

    /// Look at the current token without consuming it.
    pub fn peek(&self) -> &Token {
        self.remainder.as_ref().unwrap_or_else(|| {
            self.tokens.get(self.pos).unwrap_or_else(|| {
                self.tokens
                    .last()
                    .expect("token stream must have at least EOF")
            })
        })
    }

    /// Look ahead by n tokens (0 = current).
    pub fn peek_at(&self, n: usize) -> &Token {
        if n == 0 {
            return self.peek();
        }
        self.tokens.get(self.pos + n).unwrap_or_else(|| {
            self.tokens
                .last()
                .expect("token stream must have at least EOF")
        })
    }

    /// Consume and return the current token.
    pub fn advance(&mut self) -> &Token {
        if let Some(token) = self.remainder.take() {
            self.consumed_remainder = Some(token);
            self.pos += 1;
            return self.consumed_remainder.as_ref().unwrap();
        }
        let pos = self.pos;
        if self.peek().kind != TokenKind::Eof {
            self.pos += 1;
        }
        &self.tokens[pos]
    }

    /// Check if the current token matches the given kind.
    pub fn at(&self, kind: TokenKind) -> bool {
        self.peek().kind == kind
    }

    /// Check if the current token matches any of the given kinds.
    pub fn at_any(&self, kinds: &[TokenKind]) -> bool {
        kinds.contains(&self.peek().kind)
    }

    /// Consume the current token if it matches, return it. Otherwise None.
    pub fn eat(&mut self, kind: TokenKind) -> Option<&Token> {
        if self.at(kind) {
            Some(self.advance())
        } else {
            None
        }
    }

    /// Consume the current token if it matches, or return an error.
    pub fn expect(&mut self, kind: TokenKind) -> Result<Span, ParseError> {
        if self.at(kind) {
            let token = self.advance();
            Ok(token.span)
        } else {
            let token = self.peek();
            Err(ParseError {
                message: format!(
                    "expected {:?}, found {:?} ('{}')",
                    kind,
                    token.kind,
                    token.text(self.source)
                ),
                span: token.span,
            })
        }
    }

    /// Consume one generic closing bracket, leaving any operator suffix
    /// available to the enclosing type or assignment expression.
    pub fn expect_closing_gt(&mut self) -> Result<Span, ParseError> {
        let token = self.peek();
        let suffix = match token.kind {
            TokenKind::Gt => return self.expect(TokenKind::Gt),
            TokenKind::Ge => TokenKind::Assign,
            TokenKind::Shr => TokenKind::Gt,
            TokenKind::Ushr => TokenKind::Shr,
            TokenKind::ShrAssign => TokenKind::Ge,
            TokenKind::UshrAssign => TokenKind::ShrAssign,
            _ => {
                return Err(ParseError {
                    message: format!(
                        "expected '>', found {:?} ('{}')",
                        token.kind,
                        token.text(self.source)
                    ),
                    span: token.span,
                });
            }
        };
        let claimed = Span {
            start: token.span.start,
            end: token.span.start + 1,
        };
        self.remainder = Some(Token::new(suffix, claimed.end, token.span.end));
        Ok(claimed)
    }

    /// Whether the current token starts with a generic closing bracket.
    pub fn at_closing_gt(&self) -> bool {
        matches!(
            self.peek().kind,
            TokenKind::Gt
                | TokenKind::Ge
                | TokenKind::Shr
                | TokenKind::Ushr
                | TokenKind::ShrAssign
                | TokenKind::UshrAssign
        )
    }

    /// Get the text of the current token.
    pub fn current_text(&self) -> &'a str {
        self.peek().text(self.source)
    }

    /// The text of the token `n` ahead.
    pub fn text_at(&self, n: usize) -> &'a str {
        self.peek_at(n).text(self.source)
    }

    /// Get the byte offset of the current position.
    pub fn current_offset(&self) -> usize {
        self.peek().span.start
    }

    /// Create a span from a start offset to the current position.
    pub fn span_from(&self, start: usize) -> Span {
        let end = if let Some(remainder) = &self.remainder {
            remainder.span.start
        } else if self.pos > 0 {
            self.tokens[self.pos - 1].span.end
        } else {
            start
        };
        Span { start, end }
    }

    /// Get the source string.
    pub fn source(&self) -> &'a str {
        self.source
    }

    /// Check if we're at end of file.
    pub fn is_eof(&self) -> bool {
        self.at(TokenKind::Eof)
    }

    /// Save current position for backtracking.
    pub fn save(&self) -> StreamPosition {
        StreamPosition {
            pos: self.pos,
            remainder: self.remainder.clone(),
        }
    }

    /// Restore to a saved position.
    pub fn restore(&mut self, position: StreamPosition) {
        self.pos = position.pos;
        self.remainder = position.remainder;
    }
}

/// Parse error with span information.
#[derive(Debug, Clone)]
pub struct ParseError {
    pub message: String,
    pub span: Span,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "parse error at {}..{}: {}",
            self.span.start, self.span.end, self.message
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;

    #[test]
    fn test_peek_advance() {
        let source = "var x = 42;";
        let tokens = Lexer::new(source).tokenize().unwrap();
        let mut stream = TokenStream::new(&tokens, source);

        assert!(stream.at(TokenKind::KwVar));
        stream.advance();
        assert!(stream.at(TokenKind::Ident));
        assert_eq!(stream.current_text(), "x");
        stream.advance();
        assert!(stream.at(TokenKind::Assign));
    }

    #[test]
    fn test_expect() {
        let source = "class Foo";
        let tokens = Lexer::new(source).tokenize().unwrap();
        let mut stream = TokenStream::new(&tokens, source);

        assert!(stream.expect(TokenKind::KwClass).is_ok());
        assert!(stream.expect(TokenKind::Ident).is_ok());
        assert!(stream.expect(TokenKind::LBrace).is_err()); // should fail
    }

    #[test]
    fn test_eat() {
        let source = "if (true)";
        let tokens = Lexer::new(source).tokenize().unwrap();
        let mut stream = TokenStream::new(&tokens, source);

        assert!(stream.eat(TokenKind::KwIf).is_some());
        assert!(stream.eat(TokenKind::KwElse).is_none()); // doesn't match
        assert!(stream.eat(TokenKind::LParen).is_some());
    }

    #[test]
    fn generic_closers_preserve_assignment_suffix_and_backtracking() {
        for (source, closers) in [
            (">=x", 1),
            (">>=x", 2),
            (">>>=x", 3),
            (">>x", 2),
            (">>>x", 3),
        ] {
            let tokens = Lexer::new(source).tokenize().unwrap();
            let mut stream = TokenStream::new(&tokens, source);
            let original = stream.save();
            for index in 0..closers {
                assert!(stream.at_closing_gt());
                let saved = stream.save();
                let span = stream.expect_closing_gt().unwrap();
                assert_eq!(
                    span,
                    Span {
                        start: index,
                        end: index + 1
                    }
                );
                assert_eq!(stream.span_from(0).end, index + 1);
                stream.restore(saved);
                assert_eq!(stream.expect_closing_gt().unwrap(), span);
                assert_eq!(stream.peek_at(0), stream.peek());
            }
            if source.contains('=') {
                assert_eq!(stream.peek_at(1).text(source), "x");
                let assignment = stream.advance();
                assert_eq!(assignment.kind, TokenKind::Assign);
                assert_eq!(
                    assignment.span,
                    Span {
                        start: closers,
                        end: closers + 1
                    }
                );
            }
            assert_eq!(stream.current_text(), "x");
            stream.restore(original);
            assert_eq!(stream.current_text(), &source[..source.len() - 1]);
        }
    }

    #[test]
    fn test_save_restore() {
        let source = "a b c";
        let tokens = Lexer::new(source).tokenize().unwrap();
        let mut stream = TokenStream::new(&tokens, source);

        let saved = stream.save();
        stream.advance(); // a
        stream.advance(); // b
        assert_eq!(stream.current_text(), "c");
        stream.restore(saved);
        assert_eq!(stream.current_text(), "a");
    }
}
