//! Declaration parsing: class, interface, enum, typedef, abstract, import, package.

use crate::haxe_ast::*;
use crate::token::TokenKind;
use super::RdParser;
use super::error::ParseError;

impl<'a, 'b> RdParser<'a, 'b> {
    /// Parse `package com.example;`
    pub fn parse_package(&mut self) -> Result<Package, ParseError> {
        let start = self.stream.current_offset();
        self.stream.expect(TokenKind::KwPackage)?;
        let path = self.parse_dotted_path()?;
        self.stream.eat(TokenKind::Semicolon);
        Ok(Package {
            path,
            span: self.stream.span_from(start),
        })
    }

    /// Parse `import haxe.ds.StringMap;`
    pub fn parse_import(&mut self) -> Result<Import, ParseError> {
        let start = self.stream.current_offset();
        self.stream.expect(TokenKind::KwImport)?;
        let path = self.parse_dotted_path()?;

        let mode = if self.stream.at(TokenKind::KwIn) || self.stream.current_text() == "as" {
            // import X as Alias or import X in Alias
            self.stream.advance();
            let alias = self.stream.current_text().to_string();
            self.stream.advance();
            ImportMode::Alias(alias)
        } else if path.last().map_or(false, |s| s == "*") {
            ImportMode::Wildcard
        } else {
            ImportMode::Normal
        };

        self.stream.eat(TokenKind::Semicolon);
        Ok(Import {
            path,
            mode,
            span: self.stream.span_from(start),
        })
    }

    /// Parse `using Lambda;`
    pub fn parse_using(&mut self) -> Result<Using, ParseError> {
        let start = self.stream.current_offset();
        self.stream.expect(TokenKind::KwUsing)?;
        let path = self.parse_dotted_path()?;
        self.stream.eat(TokenKind::Semicolon);
        Ok(Using {
            path,
            span: self.stream.span_from(start),
        })
    }

    /// Parse a type declaration (class, interface, enum, typedef, abstract).
    pub fn parse_type_declaration(&mut self) -> Result<TypeDeclaration, ParseError> {
        let meta = self.parse_metadata_list();
        let (access, modifiers) = self.parse_access_and_modifiers();

        match self.stream.peek().kind {
            TokenKind::KwClass => {
                let mut decl = self.parse_class()?;
                decl.meta = meta;
                decl.access = access;
                decl.modifiers = modifiers;
                Ok(TypeDeclaration::Class(decl))
            }
            TokenKind::KwInterface => {
                let mut decl = self.parse_interface()?;
                decl.meta = meta;
                decl.access = access;
                decl.modifiers = modifiers;
                Ok(TypeDeclaration::Interface(decl))
            }
            TokenKind::KwEnum => {
                // Could be `enum abstract` or regular `enum`
                if self.stream.peek_at(1).kind == TokenKind::KwAbstract {
                    let mut decl = self.parse_enum_abstract()?;
                    decl.meta = meta;
                    decl.access = access;
                    decl.modifiers = modifiers;
                    Ok(TypeDeclaration::Abstract(decl))
                } else {
                    let mut decl = self.parse_enum()?;
                    decl.meta = meta;
                    decl.access = access;
                    Ok(TypeDeclaration::Enum(decl))
                }
            }
            TokenKind::KwTypedef => {
                let mut decl = self.parse_typedef()?;
                decl.meta = meta;
                decl.access = access;
                Ok(TypeDeclaration::Typedef(decl))
            }
            TokenKind::KwAbstract => {
                let mut decl = self.parse_abstract()?;
                decl.meta = meta;
                decl.access = access;
                decl.modifiers = modifiers;
                Ok(TypeDeclaration::Abstract(decl))
            }
            _ => Err(ParseError::new(
                &format!(
                    "expected type declaration, found '{}'",
                    self.stream.current_text()
                ),
                self.stream.peek().span,
            )),
        }
    }

    /// Parse `class Foo<T> extends Bar implements IBaz { ... }`
    fn parse_class(&mut self) -> Result<ClassDecl, ParseError> {
        let start = self.stream.current_offset();
        self.stream.expect(TokenKind::KwClass)?;

        let name = self.stream.current_text().to_string();
        self.stream.advance();

        let type_params = self.parse_type_params()?;

        let extends = if self.stream.eat(TokenKind::KwExtends).is_some() {
            Some(self.parse_type()?)
        } else {
            None
        };

        let mut implements = Vec::new();
        while self.stream.eat(TokenKind::KwImplements).is_some() {
            implements.push(self.parse_type()?);
        }

        let fields = self.parse_class_body()?;

        Ok(ClassDecl {
            meta: Vec::new(),
            access: None,
            modifiers: Vec::new(),
            name,
            type_params,
            extends,
            implements,
            fields,
            span: self.stream.span_from(start),
        })
    }

    /// Parse `interface IFoo extends IBar { ... }`
    fn parse_interface(&mut self) -> Result<InterfaceDecl, ParseError> {
        let start = self.stream.current_offset();
        self.stream.expect(TokenKind::KwInterface)?;

        let name = self.stream.current_text().to_string();
        self.stream.advance();

        let type_params = self.parse_type_params()?;

        let mut extends = Vec::new();
        while self.stream.eat(TokenKind::KwExtends).is_some() {
            extends.push(self.parse_type()?);
            // Allow comma-separated extends
            self.stream.eat(TokenKind::Comma);
        }

        let fields = self.parse_class_body()?;

        Ok(InterfaceDecl {
            meta: Vec::new(),
            access: None,
            modifiers: Vec::new(),
            name,
            type_params,
            extends,
            fields,
            span: self.stream.span_from(start),
        })
    }

    /// Parse `enum Color { Red; Green; Blue(v:Int); }`
    fn parse_enum(&mut self) -> Result<EnumDecl, ParseError> {
        let start = self.stream.current_offset();
        self.stream.expect(TokenKind::KwEnum)?;

        let name = self.stream.current_text().to_string();
        self.stream.advance();

        let type_params = self.parse_type_params()?;

        self.stream.expect(TokenKind::LBrace)?;

        let mut constructors = Vec::new();
        while !self.stream.at(TokenKind::RBrace) && !self.stream.is_eof() {
            let meta = self.parse_metadata_list();
            let ctor_start = self.stream.current_offset();
            let ctor_name = self.stream.current_text().to_string();
            self.stream.advance();

            let params = if self.stream.at(TokenKind::LParen) {
                self.parse_function_params()?
            } else {
                Vec::new()
            };

            self.stream.eat(TokenKind::Semicolon);

            constructors.push(EnumConstructor {
                meta,
                name: ctor_name,
                params,
                span: self.stream.span_from(ctor_start),
            });
        }

        self.stream.expect(TokenKind::RBrace)?;

        Ok(EnumDecl {
            meta: Vec::new(),
            access: None,
            name,
            type_params,
            constructors,
            span: self.stream.span_from(start),
        })
    }

    /// Parse `enum abstract Color(Int) { ... }`
    fn parse_enum_abstract(&mut self) -> Result<AbstractDecl, ParseError> {
        let start = self.stream.current_offset();
        self.stream.expect(TokenKind::KwEnum)?;
        self.stream.expect(TokenKind::KwAbstract)?;

        let name = self.stream.current_text().to_string();
        self.stream.advance();

        let underlying = if self.stream.at(TokenKind::LParen) {
            self.stream.advance();
            let ty = self.parse_type()?;
            self.stream.expect(TokenKind::RParen)?;
            Some(ty)
        } else {
            None
        };

        let fields = self.parse_class_body()?;

        Ok(AbstractDecl {
            meta: Vec::new(),
            access: None,
            modifiers: Vec::new(),
            name,
            type_params: Vec::new(),
            underlying,
            from: Vec::new(),
            to: Vec::new(),
            fields,
            is_enum_abstract: true,
            span: self.stream.span_from(start),
        })
    }

    /// Parse `typedef Foo = { x:Int, y:String };`
    fn parse_typedef(&mut self) -> Result<TypedefDecl, ParseError> {
        let start = self.stream.current_offset();
        self.stream.expect(TokenKind::KwTypedef)?;

        let name = self.stream.current_text().to_string();
        self.stream.advance();

        let type_params = self.parse_type_params()?;

        self.stream.expect(TokenKind::Assign)?;
        let type_def = self.parse_type()?;
        self.stream.eat(TokenKind::Semicolon);

        Ok(TypedefDecl {
            meta: Vec::new(),
            access: None,
            name,
            type_params,
            type_def,
            span: self.stream.span_from(start),
        })
    }

    /// Parse `abstract Foo(Int) from Int to Int { ... }`
    fn parse_abstract(&mut self) -> Result<AbstractDecl, ParseError> {
        let start = self.stream.current_offset();
        self.stream.expect(TokenKind::KwAbstract)?;

        let name = self.stream.current_text().to_string();
        self.stream.advance();

        let type_params = self.parse_type_params()?;

        let underlying = if self.stream.at(TokenKind::LParen) {
            self.stream.advance();
            let ty = self.parse_type()?;
            self.stream.expect(TokenKind::RParen)?;
            Some(ty)
        } else {
            None
        };

        // Parse from/to clauses
        let mut from = Vec::new();
        let mut to = Vec::new();
        loop {
            if self.stream.current_text() == "from" {
                self.stream.advance();
                from.push(self.parse_type()?);
            } else if self.stream.current_text() == "to" {
                self.stream.advance();
                to.push(self.parse_type()?);
            } else {
                break;
            }
        }

        let fields = if self.stream.at(TokenKind::LBrace) {
            self.parse_class_body()?
        } else {
            Vec::new()
        };

        Ok(AbstractDecl {
            meta: Vec::new(),
            access: None,
            modifiers: Vec::new(),
            name,
            type_params,
            underlying,
            from,
            to,
            fields,
            is_enum_abstract: false,
            span: self.stream.span_from(start),
        })
    }

    /// Parse class body: `{ field1; field2; ... }`
    fn parse_class_body(&mut self) -> Result<Vec<ClassField>, ParseError> {
        self.stream.expect(TokenKind::LBrace)?;
        let mut fields = Vec::new();

        while !self.stream.at(TokenKind::RBrace) && !self.stream.is_eof() {
            // Skip conditional compilation inside class bodies
            if self.stream.at(TokenKind::Hash) {
                self.skip_conditional_block();
                continue;
            }

            match self.parse_class_field() {
                Ok(field) => fields.push(field),
                Err(e) => {
                    self.errors.push(e);
                    // Skip to next field or closing brace
                    while !self.stream.is_eof()
                        && !self.stream.at(TokenKind::RBrace)
                        && !self.is_at_class_field_start()
                    {
                        self.stream.advance();
                    }
                }
            }
        }

        self.stream.expect(TokenKind::RBrace)?;
        Ok(fields)
    }

    fn is_at_class_field_start(&self) -> bool {
        matches!(
            self.stream.peek().kind,
            TokenKind::KwPublic
                | TokenKind::KwPrivate
                | TokenKind::KwStatic
                | TokenKind::KwInline
                | TokenKind::KwOverride
                | TokenKind::KwExtern
                | TokenKind::KwFinal
                | TokenKind::KwDynamic
                | TokenKind::KwVar
                | TokenKind::KwFunction
                | TokenKind::At
                | TokenKind::AtColon
        )
    }

    /// Parse a single class field (var, final, function, property).
    fn parse_class_field(&mut self) -> Result<ClassField, ParseError> {
        let start = self.stream.current_offset();

        let meta = self.parse_metadata_list();
        let (access, modifiers) = self.parse_access_and_modifiers();

        let kind = if self.stream.at(TokenKind::KwVar) || self.stream.at(TokenKind::KwFinal) {
            self.parse_var_or_property_field()?
        } else if self.stream.at(TokenKind::KwFunction) {
            ClassFieldKind::Function(self.parse_function_decl()?)
        } else {
            return Err(ParseError::new(
                &format!(
                    "expected field declaration, found '{}'",
                    self.stream.current_text()
                ),
                self.stream.peek().span,
            ));
        };

        Ok(ClassField {
            meta,
            access,
            modifiers,
            kind,
            span: self.stream.span_from(start),
        })
    }

    /// Parse `var name:Type = expr;` or `var name(get, set):Type;`
    fn parse_var_or_property_field(&mut self) -> Result<ClassFieldKind, ParseError> {
        let is_final = self.stream.at(TokenKind::KwFinal);
        self.stream.advance(); // skip var or final

        let name = self.stream.current_text().to_string();
        self.stream.advance();

        // Check for property syntax: var x(get, set):Type
        if self.stream.at(TokenKind::LParen) {
            self.stream.advance();
            let getter = self.parse_property_access()?;
            self.stream.expect(TokenKind::Comma)?;
            let setter = self.parse_property_access()?;
            self.stream.expect(TokenKind::RParen)?;

            let type_hint = if self.stream.eat(TokenKind::Colon).is_some() {
                Some(self.parse_type()?)
            } else {
                None
            };
            self.stream.eat(TokenKind::Semicolon);

            return Ok(ClassFieldKind::Property {
                name,
                type_hint,
                getter,
                setter,
            });
        }

        let type_hint = if self.stream.eat(TokenKind::Colon).is_some() {
            Some(self.parse_type()?)
        } else {
            None
        };

        let expr = if self.stream.eat(TokenKind::Assign).is_some() {
            Some(self.parse_expression()?)
        } else {
            None
        };

        self.stream.eat(TokenKind::Semicolon);

        if is_final {
            Ok(ClassFieldKind::Final {
                name,
                type_hint,
                expr,
            })
        } else {
            Ok(ClassFieldKind::Var {
                name,
                type_hint,
                expr,
            })
        }
    }

    fn parse_property_access(&mut self) -> Result<PropertyAccess, ParseError> {
        let text = self.stream.current_text().to_string();
        self.stream.advance();
        match text.as_str() {
            "default" => Ok(PropertyAccess::Default),
            "null" => Ok(PropertyAccess::Null),
            "never" => Ok(PropertyAccess::Never),
            "dynamic" => Ok(PropertyAccess::Dynamic),
            "get" | "set" => Ok(PropertyAccess::Custom(text)),
            _ => Ok(PropertyAccess::Custom(text)),
        }
    }

    /// Parse `function name<T>(params):RetType { body }`
    pub(crate) fn parse_function_decl(&mut self) -> Result<Function, ParseError> {
        let start = self.stream.current_offset();
        self.stream.expect(TokenKind::KwFunction)?;

        let name = if self.stream.at(TokenKind::Ident)
            || self.stream.at(TokenKind::KwNew)
            || self.stream.peek().kind.is_keyword()
        {
            let n = self.stream.current_text().to_string();
            self.stream.advance();
            n
        } else {
            String::new() // anonymous function
        };

        let type_params = self.parse_type_params()?;
        let params = self.parse_function_params()?;

        let return_type = if self.stream.eat(TokenKind::Colon).is_some() {
            Some(self.parse_type()?)
        } else {
            None
        };

        let body = if self.stream.at(TokenKind::LBrace) {
            Some(Box::new(self.parse_expression()?))
        } else if self.stream.at(TokenKind::Semicolon) {
            self.stream.advance();
            None
        } else {
            // Expression body (for inline functions)
            Some(Box::new(self.parse_expression()?))
        };

        Ok(Function {
            name,
            type_params,
            params,
            return_type,
            body,
            span: self.stream.span_from(start),
        })
    }

    /// Parse function parameters: `(a:Int, b:String = "default", ...rest)`
    pub(crate) fn parse_function_params(&mut self) -> Result<Vec<FunctionParam>, ParseError> {
        self.stream.expect(TokenKind::LParen)?;
        let mut params = Vec::new();

        while !self.stream.at(TokenKind::RParen) && !self.stream.is_eof() {
            let param = self.parse_function_param()?;
            params.push(param);
            if !self.stream.at(TokenKind::RParen) {
                self.stream.eat(TokenKind::Comma);
            }
        }

        self.stream.expect(TokenKind::RParen)?;
        Ok(params)
    }

    fn parse_function_param(&mut self) -> Result<FunctionParam, ParseError> {
        let start = self.stream.current_offset();
        let meta = self.parse_metadata_list();

        let optional = self.stream.eat(TokenKind::Question).is_some();
        let rest = self.stream.eat(TokenKind::DotDotDot).is_some();

        let name = self.stream.current_text().to_string();
        self.stream.advance();

        let type_hint = if self.stream.eat(TokenKind::Colon).is_some() {
            Some(self.parse_type()?)
        } else {
            None
        };

        let default_value = if self.stream.eat(TokenKind::Assign).is_some() {
            Some(Box::new(self.parse_expression()?))
        } else {
            None
        };

        Ok(FunctionParam {
            meta,
            name,
            type_hint,
            optional,
            rest,
            default_value,
            span: self.stream.span_from(start),
        })
    }

    /// Parse module-level field: `var x = 10;` or `function foo() {}`
    pub fn parse_module_field(&mut self) -> Result<ModuleField, ParseError> {
        let start = self.stream.current_offset();
        let meta = self.parse_metadata_list();
        let (access, modifiers) = self.parse_access_and_modifiers();

        let kind = if self.stream.at(TokenKind::KwVar) {
            self.stream.advance();
            let name = self.stream.current_text().to_string();
            self.stream.advance();
            let type_hint = if self.stream.eat(TokenKind::Colon).is_some() {
                Some(self.parse_type()?)
            } else {
                None
            };
            let expr = if self.stream.eat(TokenKind::Assign).is_some() {
                Some(self.parse_expression()?)
            } else {
                None
            };
            self.stream.eat(TokenKind::Semicolon);
            ModuleFieldKind::Var {
                name,
                type_hint,
                expr,
            }
        } else if self.stream.at(TokenKind::KwFinal) {
            self.stream.advance();
            let name = self.stream.current_text().to_string();
            self.stream.advance();
            let type_hint = if self.stream.eat(TokenKind::Colon).is_some() {
                Some(self.parse_type()?)
            } else {
                None
            };
            let expr = if self.stream.eat(TokenKind::Assign).is_some() {
                Some(self.parse_expression()?)
            } else {
                None
            };
            self.stream.eat(TokenKind::Semicolon);
            ModuleFieldKind::Final {
                name,
                type_hint,
                expr,
            }
        } else if self.stream.at(TokenKind::KwFunction) {
            ModuleFieldKind::Function(self.parse_function_decl()?)
        } else {
            return Err(ParseError::new(
                &format!(
                    "expected module field, found '{}'",
                    self.stream.current_text()
                ),
                self.stream.peek().span,
            ));
        };

        Ok(ModuleField {
            meta,
            access,
            modifiers,
            kind,
            span: self.stream.span_from(start),
        })
    }
}
