//! Declaration parsing: class, interface, enum, typedef, abstract, import, package.

use super::error::ParseError;
use super::RdParser;
use crate::haxe_ast::*;
use crate::token::TokenKind;

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

        let (path, mode) = if self.stream.at(TokenKind::KwIn) || self.stream.current_text() == "as"
        {
            // import X as Alias or import X in Alias
            self.stream.advance();
            let alias = self.stream.current_text().to_string();
            self.stream.advance();
            (path, ImportMode::Alias(alias))
        } else if path.last().is_some_and(|s| s == "*") {
            // import com.example.Module.* [except A, B, C] → wildcard with optional exclusions
            let mut p = path;
            p.pop();
            if self.stream.current_text() == "except" {
                self.stream.advance();
                let mut exclusions = Vec::new();
                exclusions.push(self.stream.current_text().to_string());
                self.stream.advance();
                while self.stream.eat(TokenKind::Comma).is_some() {
                    exclusions.push(self.stream.current_text().to_string());
                    self.stream.advance();
                }
                (p, ImportMode::WildcardWithExclusions(exclusions))
            } else {
                (p, ImportMode::Wildcard)
            }
        } else if path.len() >= 2
            && path
                .last()
                .is_some_and(|s| s.starts_with(|c: char| c.is_ascii_lowercase()))
        {
            // import com.example.Type.staticField → field import
            let mut p = path;
            let field = p.pop().unwrap();
            (p, ImportMode::Field(field))
        } else {
            (path, ImportMode::Normal)
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
                // `abstract class Foo {}` declares a class that cannot be
                // instantiated directly -- not an abstract TYPE. Reading it as
                // one took `class` for the type's name and asked what its
                // underlying type was. Nothing here rejects `new Foo()` yet;
                // what the keyword changes is instantiation, and the shape of
                // the declaration is a class either way.
                if self.stream.peek_at(1).kind == TokenKind::KwClass {
                    self.stream.advance(); // 'abstract'
                    let mut decl = self.parse_class()?;
                    decl.meta = meta;
                    decl.access = access;
                    decl.modifiers = modifiers;
                    return Ok(TypeDeclaration::Class(decl));
                }
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
            // Allow comma-separated: implements A, B
            while self.stream.eat(TokenKind::Comma).is_some() {
                implements.push(self.parse_type()?);
            }
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
        if self.stream.eat(TokenKind::KwExtends).is_some() {
            extends.push(self.parse_type()?);
            // Allow comma-separated: extends A, B, C
            while self.stream.eat(TokenKind::Comma).is_some() {
                extends.push(self.parse_type()?);
            }
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

        let (from, to) = self.parse_abstract_conversions()?;
        let fields = self.parse_class_body()?;

        Ok(AbstractDecl {
            meta: Vec::new(),
            access: None,
            modifiers: Vec::new(),
            name,
            type_params: Vec::new(),
            underlying,
            from,
            to,
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

        let (from, to) = self.parse_abstract_conversions()?;

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

    /// Parse the contextual `from Type` / `to Type` clauses shared by
    /// regular abstracts and enum abstracts.
    fn parse_abstract_conversions(&mut self) -> Result<(Vec<Type>, Vec<Type>), ParseError> {
        let mut from = Vec::new();
        let mut to = Vec::new();

        loop {
            match self.stream.current_text() {
                "from" => {
                    self.stream.advance();
                    from.push(self.parse_type()?);
                }
                "to" => {
                    self.stream.advance();
                    to.push(self.parse_type()?);
                }
                _ => break,
            }
        }

        Ok((from, to))
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
        // `overload` is not a keyword, so it arrives as an identifier; a field
        // may still open with it.
        if self.stream.peek().kind == TokenKind::Ident && self.stream.current_text() == "overload" {
            return true;
        }
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
                | TokenKind::KwMacro
                // `abstract function foo():T;` declares a method with no body
                // inside an abstract class.
                | TokenKind::KwAbstract
                | TokenKind::At
                | TokenKind::AtColon
        )
    }

    /// Parse a single class field (var, final, function, property).
    fn parse_class_field(&mut self) -> Result<ClassField, ParseError> {
        let start = self.stream.current_offset();

        let meta = self.parse_metadata_list();
        // Consumed HERE rather than in `parse_access_and_modifiers`, which also
        // serves module-level declarations: there `abstract` opens an abstract
        // TYPE (`abstract Void` in StdTypes), and eating it leaves the type
        // dispatch looking at the name.
        //
        // Both keywords parse to declarations the compiler cannot yet honour --
        // an abstract method lowers as a body-less function and traps when
        // called, and overload resolution picks the first candidate. Parsing
        // them anyway is the point: the corpus scores what the compiler does,
        // and a parse error hides those gaps from it entirely.
        let mut saw_field_keyword = true;
        while saw_field_keyword {
            saw_field_keyword = self.stream.eat(TokenKind::KwAbstract).is_some();
            if self.stream.peek().kind == TokenKind::Ident
                && self.stream.current_text() == "overload"
            {
                self.stream.advance();
                saw_field_keyword = true;
            }
        }
        let (access, mut modifiers) = self.parse_access_and_modifiers();
        // `overload` is an identifier in Haxe and may appear between ordinary
        // modifiers, e.g. `extern inline overload static function f(...)`.
        // Consume it here, then continue collecting modifiers after it.
        while self.stream.peek().kind == TokenKind::Ident
            && self.stream.current_text() == "overload"
        {
            self.stream.advance();
            let (_, more_modifiers) = self.parse_access_and_modifiers();
            modifiers.extend(more_modifiers);
        }
        // Either order: `abstract public function` and `public abstract function`.
        while self.stream.eat(TokenKind::KwAbstract).is_some() {}

        // `final` is collected as a Modifier by parse_access_and_modifiers,
        // so when the user wrote `final array:Array<T>;` we're now positioned
        // on the bare identifier `array` with no remaining keyword token.
        // Mirror the module-field path: if Final was consumed and current is
        // an Ident, treat it as a `final`-typed var field.
        let has_final_modifier = modifiers.iter().any(|m| matches!(m, Modifier::Final));

        let kind = if self.stream.at(TokenKind::KwVar) || self.stream.at(TokenKind::KwFinal) {
            self.parse_var_or_property_field()?
        } else if has_final_modifier && self.stream.at(TokenKind::Ident) {
            // `final name:Type [= expr];` — the `final` token was already
            // consumed as a modifier. Reuse the var-field parser path by
            // synthesising a name+type+expr triple.
            let name = self.stream.current_text().to_string();
            self.stream.advance();
            let type_hint = if self.stream.eat(TokenKind::Colon).is_some() {
                Some(self.parse_type()?)
            } else {
                None
            };
            let default = if self.stream.eat(TokenKind::Assign).is_some() {
                Some(self.parse_expression()?)
            } else {
                None
            };
            self.stream.eat(TokenKind::Semicolon);
            ClassFieldKind::Final {
                name,
                type_hint,
                expr: default,
            }
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

            // Property with initializer: `var CHARS(default, null) = "...";`
            // The AST's `Property` variant has no slot for an init expression,
            // so downgrade to a `Var` when an `=` follows. We lose the access
            // restriction (`(default, null)` = read-only outside the class),
            // but the runtime value still initialises correctly — which is
            // what stdlib files like Base64.hx rely on. Without this RD
            // errors on the `=` and falls back to the legacy parser.
            if self.stream.at(TokenKind::Assign) {
                self.stream.advance();
                let expr = Some(self.parse_expression()?);
                self.stream.eat(TokenKind::Semicolon);
                return Ok(ClassFieldKind::Var {
                    name,
                    type_hint,
                    expr,
                });
            }

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
            // Expression body, e.g. `function f():T return expr;`. The
            // trailing `;` is the function declaration's terminator, not
            // part of the expression — eat it here so the caller's class-
            // body loop doesn't see a stray `;` and emit a parse error
            // (the RD-parser failure then triggers the nom fallback, which
            // doesn't know the parameterised arrow type form and silently
            // returns an empty file).
            let expr = self.parse_expression()?;
            self.stream.eat(TokenKind::Semicolon);
            Some(Box::new(expr))
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

        // `final` is consumed by `parse_access_and_modifiers` as a modifier.
        // Detect `final <ident>` patterns by checking modifiers.
        let has_final_modifier = modifiers.iter().any(|m| matches!(m, Modifier::Final));

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
        } else if self.stream.at(TokenKind::KwFinal)
            || (has_final_modifier && self.stream.at(TokenKind::Ident))
        {
            // `final x: Int = 42;` — the `final` keyword was already consumed
            // as a modifier, or it's still pending as a token.
            if self.stream.at(TokenKind::KwFinal) {
                self.stream.advance();
            }
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
