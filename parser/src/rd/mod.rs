//! Recursive Descent Parser for Haxe
//!
//! Hand-written parser that consumes a TokenStream and produces the same
//! AST types as the nom-based parser. No backtracking, O(n) parsing.

pub mod decls;
pub mod error;
pub mod expr;
pub mod types;

use crate::haxe_ast::*;
use crate::lexer::Lexer;
use crate::token::TokenKind;
use crate::token_stream::TokenStream;
use error::ParseError;

/// Parse a Haxe source file using the recursive descent parser.
pub fn rd_parse(
    source: &str,
    file_name: &str,
    _is_import_file: bool,
    debug: bool,
) -> Result<HaxeFile, Vec<ParseError>> {
    let tokens = Lexer::new(source)
        .tokenize()
        .map_err(|e| vec![ParseError::new(&e.message, Span::new(e.offset, e.offset + 1))])?;

    let mut stream = TokenStream::new(&tokens, source);
    let mut parser = RdParser::new(&mut stream, source, file_name);
    let file = parser.parse_file(debug)?;
    Ok(file)
}

/// The recursive descent parser state.
pub struct RdParser<'a, 'b> {
    pub(crate) stream: &'a mut TokenStream<'b>,
    pub(crate) source: &'b str,
    pub(crate) file_name: String,
    pub(crate) errors: Vec<ParseError>,
}

impl<'a, 'b> RdParser<'a, 'b> {
    pub fn new(stream: &'a mut TokenStream<'b>, source: &'b str, file_name: &str) -> Self {
        Self {
            stream,
            source,
            file_name: file_name.to_string(),
            errors: Vec::new(),
        }
    }

    /// Parse a complete Haxe file.
    pub fn parse_file(&mut self, debug: bool) -> Result<HaxeFile, Vec<ParseError>> {
        let start = self.stream.current_offset();

        // Package declaration
        let package = if self.stream.at(TokenKind::KwPackage) {
            Some(self.parse_package()?)
        } else {
            None
        };

        let mut imports = Vec::new();
        let mut usings = Vec::new();
        let mut declarations = Vec::new();
        let mut module_fields = Vec::new();

        // Parse top-level elements
        while !self.stream.is_eof() {
            // Import
            if self.stream.at(TokenKind::KwImport) {
                imports.push(self.parse_import()?);
                continue;
            }

            // Using
            if self.stream.at(TokenKind::KwUsing) {
                usings.push(self.parse_using()?);
                continue;
            }

            // Try type declaration or module field
            if self.is_at_type_declaration() {
                match self.parse_type_declaration() {
                    Ok(decl) => declarations.push(decl),
                    Err(e) => {
                        self.errors.push(e);
                        self.synchronize();
                    }
                }
                continue;
            }

            // Module-level field (var, final, function with modifiers/metadata)
            if self.is_at_module_field() {
                match self.parse_module_field() {
                    Ok(field) => module_fields.push(field),
                    Err(e) => {
                        self.errors.push(e);
                        self.synchronize();
                    }
                }
                continue;
            }

            // Skip conditional compilation directives
            if self.stream.at(TokenKind::Hash) {
                self.skip_conditional_block();
                continue;
            }

            // Unknown token — skip
            if !self.stream.is_eof() {
                self.stream.advance();
            }
        }

        let span = Span::new(start, self.source.len());

        if !self.errors.is_empty() {
            return Err(self.errors.clone());
        }

        Ok(HaxeFile {
            filename: self.file_name.clone(),
            input: if debug {
                Some(self.source.to_string())
            } else {
                None
            },
            package,
            imports,
            using: usings,
            module_fields,
            declarations,
            span,
        })
    }

    /// Check if current position starts a type declaration.
    fn is_at_type_declaration(&self) -> bool {
        // Look past metadata and modifiers for a declaration keyword
        let mut i = 0;
        loop {
            let tok = self.stream.peek_at(i);
            match tok.kind {
                TokenKind::At | TokenKind::AtColon => {
                    // Skip metadata: @name or @:name(...)
                    i += 1;
                    // Skip metadata name
                    if self.stream.peek_at(i).kind == TokenKind::Ident
                        || self.stream.peek_at(i).kind.is_keyword()
                    {
                        i += 1;
                    }
                    // Skip params if present
                    if self.stream.peek_at(i).kind == TokenKind::LParen {
                        let mut depth = 1;
                        i += 1;
                        while depth > 0 {
                            match self.stream.peek_at(i).kind {
                                TokenKind::LParen => depth += 1,
                                TokenKind::RParen => depth -= 1,
                                TokenKind::Eof => return false,
                                _ => {}
                            }
                            i += 1;
                        }
                    }
                }
                TokenKind::KwPublic
                | TokenKind::KwPrivate
                | TokenKind::KwStatic
                | TokenKind::KwInline
                | TokenKind::KwExtern
                | TokenKind::KwFinal
                | TokenKind::KwOverride => {
                    i += 1;
                }
                TokenKind::KwClass
                | TokenKind::KwInterface
                | TokenKind::KwEnum
                | TokenKind::KwTypedef
                | TokenKind::KwAbstract => {
                    return true;
                }
                _ => return false,
            }
        }
    }

    /// Check if current position starts a module-level field.
    fn is_at_module_field(&self) -> bool {
        let mut i = 0;
        loop {
            let tok = self.stream.peek_at(i);
            match tok.kind {
                TokenKind::At | TokenKind::AtColon => {
                    i += 1;
                    if self.stream.peek_at(i).kind == TokenKind::Ident
                        || self.stream.peek_at(i).kind.is_keyword()
                    {
                        i += 1;
                    }
                    if self.stream.peek_at(i).kind == TokenKind::LParen {
                        let mut depth = 1;
                        i += 1;
                        while depth > 0 {
                            match self.stream.peek_at(i).kind {
                                TokenKind::LParen => depth += 1,
                                TokenKind::RParen => depth -= 1,
                                TokenKind::Eof => return false,
                                _ => {}
                            }
                            i += 1;
                        }
                    }
                }
                TokenKind::KwPublic
                | TokenKind::KwPrivate
                | TokenKind::KwStatic
                | TokenKind::KwInline
                | TokenKind::KwExtern
                | TokenKind::KwFinal
                | TokenKind::KwOverride => {
                    i += 1;
                }
                TokenKind::KwVar | TokenKind::KwFunction => return true,
                _ => return false,
            }
        }
    }

    /// Skip to next top-level declaration for error recovery.
    fn synchronize(&mut self) {
        loop {
            if self.stream.is_eof() {
                break;
            }
            match self.stream.peek().kind {
                TokenKind::KwClass
                | TokenKind::KwInterface
                | TokenKind::KwEnum
                | TokenKind::KwTypedef
                | TokenKind::KwAbstract
                | TokenKind::KwImport
                | TokenKind::KwUsing
                | TokenKind::KwPackage => break,
                TokenKind::RBrace => {
                    self.stream.advance();
                    break;
                }
                _ => {
                    self.stream.advance();
                }
            }
        }
    }

    /// Skip a conditional compilation block (#if ... #end)
    fn skip_conditional_block(&mut self) {
        self.stream.advance(); // skip #
        let mut depth = 1;
        while !self.stream.is_eof() && depth > 0 {
            let text = self.stream.current_text();
            if self.stream.at(TokenKind::Hash) {
                self.stream.advance();
                let next = self.stream.current_text();
                if next == "if" {
                    depth += 1;
                    self.stream.advance();
                } else if next == "end" {
                    depth -= 1;
                    self.stream.advance();
                } else {
                    // #else, #elseif, etc.
                }
            } else {
                self.stream.advance();
            }
        }
    }

    // === Shared helpers ===

    /// Parse a dotted path: `a.b.c`
    pub(crate) fn parse_dotted_path(&mut self) -> Result<Vec<String>, ParseError> {
        let mut path = Vec::new();

        // First segment must be an identifier
        let text = self.stream.current_text().to_string();
        if !self.stream.at(TokenKind::Ident) && !self.stream.peek().kind.is_keyword() {
            return Err(ParseError::new(
                &format!("expected identifier, found '{}'", text),
                self.stream.peek().span,
            ));
        }
        path.push(text);
        self.stream.advance();

        // Additional segments separated by dots
        while self.stream.eat(TokenKind::Dot).is_some() {
            let text = self.stream.current_text().to_string();
            if self.stream.at(TokenKind::Ident) || self.stream.peek().kind.is_keyword() {
                path.push(text);
                self.stream.advance();
            } else if self.stream.at(TokenKind::Star) {
                path.push("*".to_string());
                self.stream.advance();
                break;
            } else {
                break;
            }
        }

        Ok(path)
    }

    /// Parse metadata list: `@meta @:native("name")` etc.
    pub(crate) fn parse_metadata_list(&mut self) -> Vec<Metadata> {
        let mut metas = Vec::new();
        while self.stream.at(TokenKind::At) || self.stream.at(TokenKind::AtColon) {
            if let Ok(meta) = self.parse_metadata() {
                metas.push(meta);
            } else {
                break;
            }
        }
        metas
    }

    fn parse_metadata(&mut self) -> Result<Metadata, ParseError> {
        let start = self.stream.current_offset();

        let is_colon = self.stream.at(TokenKind::AtColon);
        self.stream.advance(); // skip @ or @:

        // Name (can be keyword like "native", "forward", etc.)
        let name = if is_colon {
            let text = self.stream.current_text().to_string();
            self.stream.advance();
            text
        } else {
            let text = self.stream.current_text().to_string();
            self.stream.advance();
            text
        };

        // Optional parameters
        let params = if self.stream.at(TokenKind::LParen) {
            self.stream.advance(); // skip (
            let mut args = Vec::new();
            while !self.stream.at(TokenKind::RParen) && !self.stream.is_eof() {
                match self.parse_expression() {
                    Ok(expr) => args.push(expr),
                    Err(_) => {
                        // Skip to closing paren
                        while !self.stream.at(TokenKind::RParen) && !self.stream.is_eof() {
                            self.stream.advance();
                        }
                        break;
                    }
                }
                if !self.stream.at(TokenKind::RParen) {
                    self.stream.eat(TokenKind::Comma);
                }
            }
            self.stream.eat(TokenKind::RParen);
            args
        } else {
            Vec::new()
        };

        let span = self.stream.span_from(start);
        let full_name = if is_colon {
            name
        } else {
            name
        };

        Ok(Metadata {
            name: full_name,
            params,
            span,
        })
    }

    /// Parse access and modifiers: `public static inline`
    pub(crate) fn parse_access_and_modifiers(&mut self) -> (Option<Access>, Vec<Modifier>) {
        let mut access = None;
        let mut modifiers = Vec::new();

        loop {
            match self.stream.peek().kind {
                TokenKind::KwPublic => {
                    access = Some(Access::Public);
                    self.stream.advance();
                }
                TokenKind::KwPrivate => {
                    access = Some(Access::Private);
                    self.stream.advance();
                }
                TokenKind::KwStatic => {
                    modifiers.push(Modifier::Static);
                    self.stream.advance();
                }
                TokenKind::KwInline => {
                    modifiers.push(Modifier::Inline);
                    self.stream.advance();
                }
                TokenKind::KwOverride => {
                    modifiers.push(Modifier::Override);
                    self.stream.advance();
                }
                TokenKind::KwExtern => {
                    modifiers.push(Modifier::Extern);
                    self.stream.advance();
                }
                TokenKind::KwFinal => {
                    modifiers.push(Modifier::Final);
                    self.stream.advance();
                }
                TokenKind::KwDynamic => {
                    modifiers.push(Modifier::Dynamic);
                    self.stream.advance();
                }
                TokenKind::KwMacro => {
                    modifiers.push(Modifier::Macro);
                    self.stream.advance();
                }
                _ => break,
            }
        }

        (access, modifiers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_empty_file() {
        let result = rd_parse("", "test.hx", false, false);
        assert!(result.is_ok());
        let file = result.unwrap();
        assert!(file.declarations.is_empty());
    }

    #[test]
    fn test_parse_package() {
        let result = rd_parse("package com.example;", "test.hx", false, false);
        assert!(result.is_ok());
        let file = result.unwrap();
        assert!(file.package.is_some());
        assert_eq!(file.package.unwrap().path, vec!["com", "example"]);
    }

    #[test]
    fn test_parse_import() {
        let result = rd_parse("import haxe.ds.StringMap;", "test.hx", false, false);
        assert!(result.is_ok());
        let file = result.unwrap();
        assert_eq!(file.imports.len(), 1);
        assert_eq!(
            file.imports[0].path,
            vec!["haxe", "ds", "StringMap"]
        );
    }

    #[test]
    fn test_parse_empty_class() {
        let result = rd_parse("class Foo {}", "test.hx", false, false);
        assert!(result.is_ok());
        let file = result.unwrap();
        assert_eq!(file.declarations.len(), 1);
        if let TypeDeclaration::Class(c) = &file.declarations[0] {
            assert_eq!(c.name, "Foo");
            assert!(c.fields.is_empty());
        } else {
            panic!("expected class declaration");
        }
    }

    #[test]
    fn test_parse_class_with_extends() {
        let result = rd_parse(
            "class Bar extends Foo implements IBaz {}",
            "test.hx",
            false,
            false,
        );
        assert!(result.is_ok());
        let file = result.unwrap();
        if let TypeDeclaration::Class(c) = &file.declarations[0] {
            assert_eq!(c.name, "Bar");
            assert!(c.extends.is_some());
            assert_eq!(c.implements.len(), 1);
        } else {
            panic!("expected class declaration");
        }
    }

    #[test]
    fn test_parse_enum() {
        let result = rd_parse(
            "enum Color { Red; Green; Blue; }",
            "test.hx",
            false,
            false,
        );
        assert!(result.is_ok());
        let file = result.unwrap();
        if let TypeDeclaration::Enum(e) = &file.declarations[0] {
            assert_eq!(e.name, "Color");
            assert_eq!(e.constructors.len(), 3);
        } else {
            panic!("expected enum declaration");
        }
    }
}
