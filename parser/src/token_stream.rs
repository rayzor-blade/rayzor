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
    /// Counts the `>` slots already virtually consumed from a leading
    /// `>>` (Shr) or `>>>` (Ushr) token by `expect_closing_gt`. Each
    /// call increments this until the token is fully claimed, then
    /// `advance()` resets it and moves past the token.
    ///
    /// - `0`: no virtual consumption yet; treat current token as whole.
    /// - `1`: one `>` slot taken — still pointing at the same Shr/Ushr.
    /// - `2`: two `>` slots taken — only meaningful for Ushr.
    split_gt_consumed: u8,
}

impl<'a> TokenStream<'a> {
    pub fn new(tokens: &'a [Token], source: &'a str) -> Self {
        Self {
            tokens,
            pos: 0,
            source,
            split_gt_consumed: 0,
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

    /// Consume the closing `>` of a type-parameter list, virtually
    /// splitting a multi-`>` token on the fly. This handles nested
    /// generics like `Array<Thread<Int>>` (closing `>>` = Shr) and
    /// `Array<Array<Array<Int>>>` (closing `>>>` = Ushr).
    ///
    /// Each call claims one `>` slot:
    /// - At `Gt`: advance fully.
    /// - At `Shr` (worth 2 slots): increment counter; on slot 2, advance.
    /// - At `Ushr` (worth 3 slots): increment counter; on slot 3, advance.
    /// - Anything else: standard expect-Gt error.
    pub fn expect_closing_gt(&mut self) -> Result<Span, ParseError> {
        if self.at(TokenKind::Gt) {
            let token = self.advance();
            return Ok(token.span);
        }
        let token = self.peek();
        let total_slots = match token.kind {
            TokenKind::Shr => 2,
            TokenKind::Ushr => 3,
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
        let slot_idx = self.split_gt_consumed;
        let span_start = token.span.start + slot_idx as usize;
        let span_end = span_start + 1;
        let claimed = Span {
            start: span_start,
            end: span_end,
        };
        if (slot_idx as u32) + 1 == total_slots {
            // Last slot — consume the whole token and reset.
            self.split_gt_consumed = 0;
            self.advance();
        } else {
            // Virtual consume — token stays, counter advances.
            self.split_gt_consumed = slot_idx + 1;
        }
        Ok(claimed)
    }

    /// Like `at(Gt)` but also returns true at `>>` (Shr) or `>>>` (Ushr)
    /// — the lexer's multi-char tokens that the parser splits on the
    /// fly for nested generic close. Use this for the loop condition
    /// inside `parse_type_param_args` and friends.
    pub fn at_closing_gt(&self) -> bool {
        let kind = self.peek().kind;
        kind == TokenKind::Gt || kind == TokenKind::Shr || kind == TokenKind::Ushr
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
