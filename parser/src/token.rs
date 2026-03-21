//! Token types for the Haxe lexer.
//!
//! Tokens are the output of the lexer and input to the recursive descent parser.
//! Each token carries its kind, byte-offset span, and raw text slice.

use crate::haxe_ast::Span;

/// A single token produced by the lexer.
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

impl Token {
    pub fn new(kind: TokenKind, start: usize, end: usize) -> Self {
        Self {
            kind,
            span: Span { start, end },
        }
    }

    pub fn is_eof(&self) -> bool {
        self.kind == TokenKind::Eof
    }

    /// Get the raw text from source
    pub fn text<'a>(&self, source: &'a str) -> &'a str {
        &source[self.span.start..self.span.end]
    }
}

/// Token kind — every distinct syntactic element.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokenKind {
    // === Literals ===
    IntLit,
    FloatLit,
    StringLit, // double-quoted "..." or single-quoted '...'
    RegexLit,  // ~/pattern/flags

    // === Identifiers ===
    Ident,
    DollarIdent, // $ident (macro reification)

    // === Keywords ===
    KwAbstract,
    KwBreak,
    KwCase,
    KwCast,
    KwCatch,
    KwClass,
    KwContinue,
    KwDefault,
    KwDo,
    KwDynamic,
    KwElse,
    KwEnum,
    KwExtends,
    KwExtern,
    KwFalse,
    KwFinal,
    KwFinally,
    KwFor,
    KwFunction,
    KwIf,
    KwImplements,
    KwImport,
    KwIn,
    KwInline,
    KwInterface,
    KwIs,
    KwMacro,
    KwNew,
    KwNull,
    KwOverride,
    KwPackage,
    KwPrivate,
    KwPublic,
    KwReturn,
    KwStatic,
    KwSuper,
    KwSwitch,
    KwThis,
    KwThrow,
    KwTrue,
    KwTry,
    KwTypedef,
    KwUntyped,
    KwUsing,
    KwVar,
    KwWhile,

    // === Punctuation ===
    LParen,           // (
    RParen,           // )
    LBrace,           // {
    RBrace,           // }
    LBracket,         // [
    RBracket,         // ]
    Dot,              // .
    DotDotDot,        // ...
    Comma,            // ,
    Semicolon,        // ;
    Colon,            // :
    At,               // @
    AtColon,          // @:
    Arrow,            // ->
    FatArrow,         // =>
    Question,         // ?
    QuestionDot,      // ?.
    QuestionQuestion, // ??
    Hash,             // #

    // === Operators ===
    Assign,        // =
    Eq,            // ==
    NotEq,         // !=
    Lt,            // <
    Le,            // <=
    Gt,            // >
    Ge,            // >=
    Plus,          // +
    Minus,         // -
    Star,          // *
    Slash,         // /
    Percent,       // %
    PlusAssign,    // +=
    MinusAssign,   // -=
    StarAssign,    // *=
    SlashAssign,   // /=
    PercentAssign, // %=
    Amp,           // &
    Pipe,          // |
    Caret,         // ^
    Tilde,         // ~
    AmpAmp,        // &&
    PipePipe,      // ||
    AmpAssign,     // &=
    PipeAssign,    // |=
    CaretAssign,   // ^=
    Shl,           // <<
    Shr,           // >>
    Ushr,          // >>>
    ShlAssign,     // <<=
    ShrAssign,     // >>=
    UshrAssign,    // >>>=
    Bang,          // !
    PlusPlus,      // ++
    MinusMinus,    // --

    // === Special ===
    Eof,
}

impl TokenKind {
    /// Check if this token kind is a keyword
    pub fn is_keyword(&self) -> bool {
        matches!(
            self,
            TokenKind::KwAbstract
                | TokenKind::KwBreak
                | TokenKind::KwCase
                | TokenKind::KwCast
                | TokenKind::KwCatch
                | TokenKind::KwClass
                | TokenKind::KwContinue
                | TokenKind::KwDefault
                | TokenKind::KwDo
                | TokenKind::KwDynamic
                | TokenKind::KwElse
                | TokenKind::KwEnum
                | TokenKind::KwExtends
                | TokenKind::KwExtern
                | TokenKind::KwFalse
                | TokenKind::KwFinal
                | TokenKind::KwFinally
                | TokenKind::KwFor
                | TokenKind::KwFunction
                | TokenKind::KwIf
                | TokenKind::KwImplements
                | TokenKind::KwImport
                | TokenKind::KwIn
                | TokenKind::KwInline
                | TokenKind::KwInterface
                | TokenKind::KwIs
                | TokenKind::KwMacro
                | TokenKind::KwNew
                | TokenKind::KwNull
                | TokenKind::KwOverride
                | TokenKind::KwPackage
                | TokenKind::KwPrivate
                | TokenKind::KwPublic
                | TokenKind::KwReturn
                | TokenKind::KwStatic
                | TokenKind::KwSuper
                | TokenKind::KwSwitch
                | TokenKind::KwThis
                | TokenKind::KwThrow
                | TokenKind::KwTrue
                | TokenKind::KwTry
                | TokenKind::KwTypedef
                | TokenKind::KwUntyped
                | TokenKind::KwUsing
                | TokenKind::KwVar
                | TokenKind::KwWhile
        )
    }

    /// Look up keyword from identifier text
    pub fn keyword_from_str(s: &str) -> Option<TokenKind> {
        match s {
            "abstract" => Some(TokenKind::KwAbstract),
            "break" => Some(TokenKind::KwBreak),
            "case" => Some(TokenKind::KwCase),
            "cast" => Some(TokenKind::KwCast),
            "catch" => Some(TokenKind::KwCatch),
            "class" => Some(TokenKind::KwClass),
            "continue" => Some(TokenKind::KwContinue),
            "default" => Some(TokenKind::KwDefault),
            "do" => Some(TokenKind::KwDo),
            "dynamic" => Some(TokenKind::KwDynamic),
            "else" => Some(TokenKind::KwElse),
            "enum" => Some(TokenKind::KwEnum),
            "extends" => Some(TokenKind::KwExtends),
            "extern" => Some(TokenKind::KwExtern),
            "false" => Some(TokenKind::KwFalse),
            "final" => Some(TokenKind::KwFinal),
            "finally" => Some(TokenKind::KwFinally),
            "for" => Some(TokenKind::KwFor),
            "function" => Some(TokenKind::KwFunction),
            "if" => Some(TokenKind::KwIf),
            "implements" => Some(TokenKind::KwImplements),
            "import" => Some(TokenKind::KwImport),
            "in" => Some(TokenKind::KwIn),
            "inline" => Some(TokenKind::KwInline),
            "interface" => Some(TokenKind::KwInterface),
            "is" => Some(TokenKind::KwIs),
            "macro" => Some(TokenKind::KwMacro),
            "new" => Some(TokenKind::KwNew),
            "null" => Some(TokenKind::KwNull),
            "override" => Some(TokenKind::KwOverride),
            "package" => Some(TokenKind::KwPackage),
            "private" => Some(TokenKind::KwPrivate),
            "public" => Some(TokenKind::KwPublic),
            "return" => Some(TokenKind::KwReturn),
            "static" => Some(TokenKind::KwStatic),
            "super" => Some(TokenKind::KwSuper),
            "switch" => Some(TokenKind::KwSwitch),
            "this" => Some(TokenKind::KwThis),
            "throw" => Some(TokenKind::KwThrow),
            "true" => Some(TokenKind::KwTrue),
            "try" => Some(TokenKind::KwTry),
            "typedef" => Some(TokenKind::KwTypedef),
            "untyped" => Some(TokenKind::KwUntyped),
            "using" => Some(TokenKind::KwUsing),
            "var" => Some(TokenKind::KwVar),
            "while" => Some(TokenKind::KwWhile),
            _ => None,
        }
    }
}
