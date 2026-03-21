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
}

impl<'a> TokenStream<'a> {
    pub fn new(tokens: &'a [Token], source: &'a str) -> Self {
        Self {
            tokens,
            pos: 0,
            source,
        }
    }

    /// Look at the current token without consuming it.
    pub fn peek(&self) -> &Token {
        self.tokens.get(self.pos).unwrap_or_else(|| {
            self.tokens
                .last()
                .expect("token stream must have at least EOF")
        })
    }

    /// Look ahead by n tokens (0 = current).
    pub fn peek_at(&self, n: usize) -> &Token {
        self.tokens.get(self.pos + n).unwrap_or_else(|| {
            self.tokens
                .last()
                .expect("token stream must have at least EOF")
        })
    }

    /// Consume and return the current token.
    pub fn advance(&mut self) -> &Token {
        let token = self.peek();
        if token.kind != TokenKind::Eof {
            self.pos += 1;
        }
        &self.tokens[self.pos - 1]
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
            Ok(token.span.clone())
        } else {
            let token = self.peek();
            Err(ParseError {
                message: format!(
                    "expected {:?}, found {:?} ('{}')",
                    kind,
                    token.kind,
                    token.text(self.source)
                ),
                span: token.span.clone(),
            })
        }
    }

    /// Get the text of the current token.
    pub fn current_text(&self) -> &'a str {
        self.peek().text(self.source)
    }

    /// Get the byte offset of the current position.
    pub fn current_offset(&self) -> usize {
        self.peek().span.start
    }

    /// Create a span from a start offset to the current position.
    pub fn span_from(&self, start: usize) -> Span {
        let end = if self.pos > 0 {
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
    pub fn save(&self) -> usize {
        self.pos
    }

    /// Restore to a saved position.
    pub fn restore(&mut self, pos: usize) {
        self.pos = pos;
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
