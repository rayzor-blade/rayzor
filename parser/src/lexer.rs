//! Single-pass lexer for Haxe source code.
//!
//! Converts source text into a `Vec<Token>` in O(n) time with no backtracking.
//! Comments and whitespace are discarded. Keywords are recognized via match.

use crate::token::{Token, TokenKind};

/// Lexer error
#[derive(Debug, Clone)]
pub struct LexError {
    pub message: String,
    pub offset: usize,
}

impl std::fmt::Display for LexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Lex error at byte {}: {}", self.offset, self.message)
    }
}

/// Single-pass lexer that produces a flat token stream.
pub struct Lexer<'a> {
    source: &'a [u8],
    pos: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(source: &'a str) -> Self {
        Self {
            source: source.as_bytes(),
            pos: 0,
        }
    }

    /// Tokenize the entire source. O(n) single pass.
    pub fn tokenize(mut self) -> Result<Vec<Token>, LexError> {
        let mut tokens = Vec::with_capacity(self.source.len() / 4); // rough estimate

        loop {
            self.skip_whitespace_and_comments();
            if self.pos >= self.source.len() {
                tokens.push(Token::new(TokenKind::Eof, self.pos, self.pos));
                break;
            }
            let token = self.next_token()?;
            tokens.push(token);
        }

        Ok(tokens)
    }

    fn peek(&self) -> u8 {
        if self.pos < self.source.len() {
            self.source[self.pos]
        } else {
            0
        }
    }

    fn peek_at(&self, offset: usize) -> u8 {
        let idx = self.pos + offset;
        if idx < self.source.len() {
            self.source[idx]
        } else {
            0
        }
    }

    fn advance(&mut self) -> u8 {
        let ch = self.peek();
        if self.pos < self.source.len() {
            self.pos += 1;
        }
        ch
    }

    fn skip_whitespace_and_comments(&mut self) {
        loop {
            // Skip whitespace
            while self.pos < self.source.len() && self.source[self.pos].is_ascii_whitespace() {
                self.pos += 1;
            }

            if self.pos >= self.source.len() {
                return;
            }

            // Skip line comments: //
            if self.peek() == b'/' && self.peek_at(1) == b'/' {
                self.pos += 2;
                while self.pos < self.source.len() && self.source[self.pos] != b'\n' {
                    self.pos += 1;
                }
                continue;
            }

            // Skip block comments: /* ... */
            if self.peek() == b'/' && self.peek_at(1) == b'*' {
                self.pos += 2;
                let mut depth = 1;
                while self.pos + 1 < self.source.len() && depth > 0 {
                    if self.source[self.pos] == b'/' && self.source[self.pos + 1] == b'*' {
                        depth += 1;
                        self.pos += 2;
                    } else if self.source[self.pos] == b'*' && self.source[self.pos + 1] == b'/' {
                        depth -= 1;
                        self.pos += 2;
                    } else {
                        self.pos += 1;
                    }
                }
                continue;
            }

            break;
        }
    }

    fn next_token(&mut self) -> Result<Token, LexError> {
        let start = self.pos;
        let ch = self.advance();

        match ch {
            // Single-char punctuation
            b'(' => Ok(Token::new(TokenKind::LParen, start, self.pos)),
            b')' => Ok(Token::new(TokenKind::RParen, start, self.pos)),
            b'{' => Ok(Token::new(TokenKind::LBrace, start, self.pos)),
            b'}' => Ok(Token::new(TokenKind::RBrace, start, self.pos)),
            b'[' => Ok(Token::new(TokenKind::LBracket, start, self.pos)),
            b']' => Ok(Token::new(TokenKind::RBracket, start, self.pos)),
            b',' => Ok(Token::new(TokenKind::Comma, start, self.pos)),
            b';' => Ok(Token::new(TokenKind::Semicolon, start, self.pos)),
            b'^' => {
                if self.peek() == b'=' {
                    self.advance();
                    Ok(Token::new(TokenKind::CaretAssign, start, self.pos))
                } else {
                    Ok(Token::new(TokenKind::Caret, start, self.pos))
                }
            }

            // Tilde — could be start of regex ~/pattern/
            b'~' => {
                if self.peek() == b'/' {
                    self.lex_regex(start)
                } else {
                    Ok(Token::new(TokenKind::Tilde, start, self.pos))
                }
            }

            // Dot — could be . or ...
            b'.' => {
                if self.peek() == b'.' && self.peek_at(1) == b'.' {
                    self.pos += 2;
                    Ok(Token::new(TokenKind::DotDotDot, start, self.pos))
                } else {
                    Ok(Token::new(TokenKind::Dot, start, self.pos))
                }
            }

            // Colon
            b':' => Ok(Token::new(TokenKind::Colon, start, self.pos)),

            // Hash
            b'#' => Ok(Token::new(TokenKind::Hash, start, self.pos)),

            // At — could be @ or @:
            b'@' => {
                if self.peek() == b':' {
                    self.advance();
                    Ok(Token::new(TokenKind::AtColon, start, self.pos))
                } else {
                    Ok(Token::new(TokenKind::At, start, self.pos))
                }
            }

            // Question — could be ? or ?. or ??
            b'?' => {
                if self.peek() == b'.' {
                    self.advance();
                    Ok(Token::new(TokenKind::QuestionDot, start, self.pos))
                } else if self.peek() == b'?' {
                    self.advance();
                    Ok(Token::new(TokenKind::QuestionQuestion, start, self.pos))
                } else {
                    Ok(Token::new(TokenKind::Question, start, self.pos))
                }
            }

            // Equals — = or == or =>
            b'=' => {
                if self.peek() == b'=' {
                    self.advance();
                    Ok(Token::new(TokenKind::Eq, start, self.pos))
                } else if self.peek() == b'>' {
                    self.advance();
                    Ok(Token::new(TokenKind::FatArrow, start, self.pos))
                } else {
                    Ok(Token::new(TokenKind::Assign, start, self.pos))
                }
            }

            // Bang — ! or !=
            b'!' => {
                if self.peek() == b'=' {
                    self.advance();
                    Ok(Token::new(TokenKind::NotEq, start, self.pos))
                } else {
                    Ok(Token::new(TokenKind::Bang, start, self.pos))
                }
            }

            // Less — < or <= or << or <<=
            b'<' => {
                if self.peek() == b'=' {
                    self.advance();
                    Ok(Token::new(TokenKind::Le, start, self.pos))
                } else if self.peek() == b'<' {
                    self.advance();
                    if self.peek() == b'=' {
                        self.advance();
                        Ok(Token::new(TokenKind::ShlAssign, start, self.pos))
                    } else {
                        Ok(Token::new(TokenKind::Shl, start, self.pos))
                    }
                } else {
                    Ok(Token::new(TokenKind::Lt, start, self.pos))
                }
            }

            // Greater — > or >= or >> or >>= or >>> or >>>=
            b'>' => {
                if self.peek() == b'=' {
                    self.advance();
                    Ok(Token::new(TokenKind::Ge, start, self.pos))
                } else if self.peek() == b'>' {
                    self.advance();
                    if self.peek() == b'>' {
                        self.advance();
                        if self.peek() == b'=' {
                            self.advance();
                            Ok(Token::new(TokenKind::UshrAssign, start, self.pos))
                        } else {
                            Ok(Token::new(TokenKind::Ushr, start, self.pos))
                        }
                    } else if self.peek() == b'=' {
                        self.advance();
                        Ok(Token::new(TokenKind::ShrAssign, start, self.pos))
                    } else {
                        Ok(Token::new(TokenKind::Shr, start, self.pos))
                    }
                } else {
                    Ok(Token::new(TokenKind::Gt, start, self.pos))
                }
            }

            // Plus — + or ++ or +=
            b'+' => {
                if self.peek() == b'+' {
                    self.advance();
                    Ok(Token::new(TokenKind::PlusPlus, start, self.pos))
                } else if self.peek() == b'=' {
                    self.advance();
                    Ok(Token::new(TokenKind::PlusAssign, start, self.pos))
                } else {
                    Ok(Token::new(TokenKind::Plus, start, self.pos))
                }
            }

            // Minus — - or -- or -= or ->
            b'-' => {
                if self.peek() == b'-' {
                    self.advance();
                    Ok(Token::new(TokenKind::MinusMinus, start, self.pos))
                } else if self.peek() == b'=' {
                    self.advance();
                    Ok(Token::new(TokenKind::MinusAssign, start, self.pos))
                } else if self.peek() == b'>' {
                    self.advance();
                    Ok(Token::new(TokenKind::Arrow, start, self.pos))
                } else {
                    Ok(Token::new(TokenKind::Minus, start, self.pos))
                }
            }

            // Star — * or *=
            b'*' => {
                if self.peek() == b'=' {
                    self.advance();
                    Ok(Token::new(TokenKind::StarAssign, start, self.pos))
                } else {
                    Ok(Token::new(TokenKind::Star, start, self.pos))
                }
            }

            // Slash — / or /= (comments already handled in skip_whitespace_and_comments)
            b'/' => {
                if self.peek() == b'=' {
                    self.advance();
                    Ok(Token::new(TokenKind::SlashAssign, start, self.pos))
                } else {
                    Ok(Token::new(TokenKind::Slash, start, self.pos))
                }
            }

            // Percent — % or %=
            b'%' => {
                if self.peek() == b'=' {
                    self.advance();
                    Ok(Token::new(TokenKind::PercentAssign, start, self.pos))
                } else {
                    Ok(Token::new(TokenKind::Percent, start, self.pos))
                }
            }

            // Ampersand — & or && or &=
            b'&' => {
                if self.peek() == b'&' {
                    self.advance();
                    Ok(Token::new(TokenKind::AmpAmp, start, self.pos))
                } else if self.peek() == b'=' {
                    self.advance();
                    Ok(Token::new(TokenKind::AmpAssign, start, self.pos))
                } else {
                    Ok(Token::new(TokenKind::Amp, start, self.pos))
                }
            }

            // Pipe — | or || or |=
            b'|' => {
                if self.peek() == b'|' {
                    self.advance();
                    Ok(Token::new(TokenKind::PipePipe, start, self.pos))
                } else if self.peek() == b'=' {
                    self.advance();
                    Ok(Token::new(TokenKind::PipeAssign, start, self.pos))
                } else {
                    Ok(Token::new(TokenKind::Pipe, start, self.pos))
                }
            }

            // String literals
            b'"' | b'\'' => self.lex_string(ch, start),

            // Dollar identifiers
            b'$' => {
                while self.pos < self.source.len()
                    && (self.source[self.pos].is_ascii_alphanumeric()
                        || self.source[self.pos] == b'_')
                {
                    self.pos += 1;
                }
                Ok(Token::new(TokenKind::DollarIdent, start, self.pos))
            }

            // Numbers
            b'0'..=b'9' => self.lex_number(start),

            // Identifiers and keywords
            b'a'..=b'z' | b'A'..=b'Z' | b'_' => self.lex_identifier(start),

            _ => {
                // Skip unknown bytes (UTF-8 multibyte chars in identifiers)
                if ch >= 0x80 {
                    // UTF-8 continuation — skip the full character
                    while self.pos < self.source.len() && self.source[self.pos] >= 0x80
                        && self.source[self.pos] < 0xC0
                    {
                        self.pos += 1;
                    }
                    // Treat as identifier
                    self.lex_identifier(start)
                } else {
                    Err(LexError {
                        message: format!("unexpected character: '{}'", ch as char),
                        offset: start,
                    })
                }
            }
        }
    }

    fn lex_string(&mut self, quote: u8, start: usize) -> Result<Token, LexError> {
        // Already consumed the opening quote
        loop {
            if self.pos >= self.source.len() {
                return Err(LexError {
                    message: "unterminated string literal".to_string(),
                    offset: start,
                });
            }
            let ch = self.source[self.pos];
            if ch == b'\\' {
                // Skip escape sequence
                self.pos += 1;
                if self.pos < self.source.len() {
                    self.pos += 1;
                }
                continue;
            }
            if ch == quote {
                self.pos += 1; // consume closing quote
                return Ok(Token::new(TokenKind::StringLit, start, self.pos));
            }
            self.pos += 1;
        }
    }

    fn lex_number(&mut self, start: usize) -> Result<Token, LexError> {
        // Check for hex: 0x...
        if self.source[start] == b'0' && self.pos < self.source.len()
            && (self.source[self.pos] == b'x' || self.source[self.pos] == b'X')
        {
            self.pos += 1; // skip 'x'
            while self.pos < self.source.len()
                && (self.source[self.pos].is_ascii_hexdigit() || self.source[self.pos] == b'_')
            {
                self.pos += 1;
            }
            return Ok(Token::new(TokenKind::IntLit, start, self.pos));
        }

        // Decimal digits
        while self.pos < self.source.len() && self.source[self.pos].is_ascii_digit() {
            self.pos += 1;
        }

        // Check for float: digits followed by '.' then more digits
        if self.pos < self.source.len() && self.source[self.pos] == b'.'
            && self.pos + 1 < self.source.len() && self.source[self.pos + 1].is_ascii_digit()
        {
            self.pos += 1; // skip '.'
            while self.pos < self.source.len() && self.source[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
            // Exponent: e or E
            if self.pos < self.source.len()
                && (self.source[self.pos] == b'e' || self.source[self.pos] == b'E')
            {
                self.pos += 1;
                if self.pos < self.source.len()
                    && (self.source[self.pos] == b'+' || self.source[self.pos] == b'-')
                {
                    self.pos += 1;
                }
                while self.pos < self.source.len() && self.source[self.pos].is_ascii_digit() {
                    self.pos += 1;
                }
            }
            return Ok(Token::new(TokenKind::FloatLit, start, self.pos));
        }

        Ok(Token::new(TokenKind::IntLit, start, self.pos))
    }

    fn lex_identifier(&mut self, start: usize) -> Result<Token, LexError> {
        while self.pos < self.source.len()
            && (self.source[self.pos].is_ascii_alphanumeric()
                || self.source[self.pos] == b'_')
        {
            self.pos += 1;
        }

        let text = std::str::from_utf8(&self.source[start..self.pos]).unwrap_or("");

        // Check for keyword
        let kind = TokenKind::keyword_from_str(text).unwrap_or(TokenKind::Ident);
        Ok(Token::new(kind, start, self.pos))
    }

    fn lex_regex(&mut self, start: usize) -> Result<Token, LexError> {
        // Already consumed '~', consume '/'
        self.advance();
        // Read until closing '/'
        loop {
            if self.pos >= self.source.len() {
                return Err(LexError {
                    message: "unterminated regex literal".to_string(),
                    offset: start,
                });
            }
            let ch = self.source[self.pos];
            if ch == b'\\' {
                self.pos += 2; // skip escape
                continue;
            }
            if ch == b'/' {
                self.pos += 1; // consume closing '/'
                // Read flags: g, i, m, s, u
                while self.pos < self.source.len() && self.source[self.pos].is_ascii_alphabetic() {
                    self.pos += 1;
                }
                return Ok(Token::new(TokenKind::RegexLit, start, self.pos));
            }
            self.pos += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lex(source: &str) -> Vec<Token> {
        Lexer::new(source).tokenize().unwrap()
    }

    fn kinds(source: &str) -> Vec<TokenKind> {
        lex(source).into_iter().map(|t| t.kind).collect()
    }

    #[test]
    fn test_empty() {
        assert_eq!(kinds(""), vec![TokenKind::Eof]);
    }

    #[test]
    fn test_keywords() {
        assert_eq!(
            kinds("class function var if else"),
            vec![
                TokenKind::KwClass,
                TokenKind::KwFunction,
                TokenKind::KwVar,
                TokenKind::KwIf,
                TokenKind::KwElse,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_operators() {
        assert_eq!(
            kinds("+ ++ += -> => == != <= >= && || ?? ?."),
            vec![
                TokenKind::Plus,
                TokenKind::PlusPlus,
                TokenKind::PlusAssign,
                TokenKind::Arrow,
                TokenKind::FatArrow,
                TokenKind::Eq,
                TokenKind::NotEq,
                TokenKind::Le,
                TokenKind::Ge,
                TokenKind::AmpAmp,
                TokenKind::PipePipe,
                TokenKind::QuestionQuestion,
                TokenKind::QuestionDot,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_numbers() {
        assert_eq!(
            kinds("42 3.14 0xFF"),
            vec![
                TokenKind::IntLit,
                TokenKind::FloatLit,
                TokenKind::IntLit,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_strings() {
        assert_eq!(
            kinds(r#""hello" 'world' "esc\"ape""#),
            vec![
                TokenKind::StringLit,
                TokenKind::StringLit,
                TokenKind::StringLit,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_comments_skipped() {
        assert_eq!(
            kinds("a // comment\nb /* block */ c"),
            vec![
                TokenKind::Ident,
                TokenKind::Ident,
                TokenKind::Ident,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_punctuation() {
        assert_eq!(
            kinds("( ) { } [ ] . , ; : @ @: ..."),
            vec![
                TokenKind::LParen, TokenKind::RParen,
                TokenKind::LBrace, TokenKind::RBrace,
                TokenKind::LBracket, TokenKind::RBracket,
                TokenKind::Dot, TokenKind::Comma,
                TokenKind::Semicolon, TokenKind::Colon,
                TokenKind::At, TokenKind::AtColon,
                TokenKind::DotDotDot,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_shift_operators() {
        assert_eq!(
            kinds("<< >> >>> <<= >>= >>>="),
            vec![
                TokenKind::Shl, TokenKind::Shr, TokenKind::Ushr,
                TokenKind::ShlAssign, TokenKind::ShrAssign, TokenKind::UshrAssign,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn test_deltablue_snippet() {
        // A realistic snippet from deltablue.hx
        let source = r#"
            class Strength {
                public var value:Int;
                public function new(v:Int) {
                    this.value = v;
                }
                public static function stronger(s1:Strength, s2:Strength):Bool {
                    return s1.value < s2.value;
                }
            }
        "#;
        let tokens = lex(source);
        // Should tokenize without errors
        assert!(tokens.last().unwrap().kind == TokenKind::Eof);
        // Count non-EOF tokens
        let count = tokens.iter().filter(|t| t.kind != TokenKind::Eof).count();
        assert!(count > 30, "expected >30 tokens, got {}", count);
    }

    #[test]
    fn test_regex_literal() {
        assert_eq!(
            kinds("~/pattern/gi"),
            vec![TokenKind::RegexLit, TokenKind::Eof]
        );
    }

    #[test]
    fn test_dollar_ident() {
        assert_eq!(
            kinds("$type $a"),
            vec![TokenKind::DollarIdent, TokenKind::DollarIdent, TokenKind::Eof]
        );
    }
}
