//! The document outline: a file's type declarations and their members, read from its syntax.
//!
//! The outline needs no types, so it is answered whatever state the type check is in, and it
//! leaves out locals and anything else a type check records.

use lsp_types::{DocumentSymbol, Position, Range, SymbolKind};
use parser::haxe_ast::{ClassField, ClassFieldKind, HaxeFile, TypeDeclaration};
use parser::token::Token;

struct Source<'a> {
    text: &'a str,
    tokens: Vec<Token>,
    line_starts: Vec<usize>,
}

impl Source<'_> {
    fn position(&self, offset: usize) -> Position {
        let offset = offset.min(self.text.len());
        let line = self.line_starts.partition_point(|&start| start <= offset) - 1;
        let start = self.line_starts[line];
        let column = self
            .text
            .get(start..offset)
            .map_or(0, |before| before.encode_utf16().count());
        Position::new(line as u32, column as u32)
    }

    fn range(&self, start: usize, end: usize) -> Range {
        Range::new(self.position(start), self.position(end))
    }

    /// The first token spelled `name` inside `start..end`: where a declaration's name is.
    fn name_span(&self, name: &str, start: usize, end: usize) -> Option<(usize, usize)> {
        self.tokens
            .iter()
            .find(|token| {
                token.span.start >= start
                    && token.span.end <= end
                    && self.text.get(token.span.start..token.span.end) == Some(name)
            })
            .map(|token| (token.span.start, token.span.end))
    }

    fn symbol(
        &self,
        name: &str,
        kind: SymbolKind,
        detail: Option<String>,
        span: (usize, usize),
        children: Vec<DocumentSymbol>,
    ) -> Option<DocumentSymbol> {
        let (name_start, name_end) = self.name_span(name, span.0, span.1)?;
        #[allow(deprecated)] // DocumentSymbol::deprecated
        Some(DocumentSymbol {
            name: name.to_string(),
            detail: detail.filter(|detail| !detail.is_empty()),
            kind,
            tags: None,
            deprecated: None,
            range: self.range(span.0, span.1),
            selection_range: self.range(name_start, name_end),
            children: (!children.is_empty()).then_some(children),
        })
    }

    /// The text after a member's name up to where its body or value starts: `(a:Int):Void`, `:Int`.
    fn after_name(&self, name: &str, start: usize, end: usize, stop: Option<usize>) -> String {
        let Some((_, name_end)) = self.name_span(name, start, end) else {
            return String::new();
        };
        let stop = stop.unwrap_or(end).clamp(name_end, self.text.len());
        let text = self.text.get(name_end..stop).unwrap_or_default();
        let text = text.split(['=', ';']).next().unwrap_or_default();
        text.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    fn members(&self, fields: &[ClassField]) -> Vec<DocumentSymbol> {
        fields
            .iter()
            .filter_map(|field| {
                let span = (field.span.start, field.span.end);
                match &field.kind {
                    ClassFieldKind::Function(function) => {
                        let kind = if function.name == "new" {
                            SymbolKind::CONSTRUCTOR
                        } else {
                            SymbolKind::METHOD
                        };
                        let stop = function.body.as_ref().map(|body| body.span.start);
                        let detail = self.after_name(&function.name, span.0, span.1, stop);
                        self.symbol(&function.name, kind, Some(detail), span, Vec::new())
                    }
                    ClassFieldKind::Var { name, .. } | ClassFieldKind::Final { name, .. } => {
                        let detail = self.after_name(name, span.0, span.1, None);
                        self.symbol(name, SymbolKind::FIELD, Some(detail), span, Vec::new())
                    }
                    ClassFieldKind::Property { name, .. } => {
                        let detail = self.after_name(name, span.0, span.1, None);
                        self.symbol(name, SymbolKind::PROPERTY, Some(detail), span, Vec::new())
                    }
                }
            })
            .collect()
    }

    fn declarations(&self, file: &HaxeFile) -> Vec<DocumentSymbol> {
        file.declarations
            .iter()
            .filter_map(|declaration| {
                let span = declaration.span();
                let span = (span.start, span.end);
                match declaration {
                    TypeDeclaration::Class(class) => self.symbol(
                        &class.name,
                        SymbolKind::CLASS,
                        None,
                        span,
                        self.members(&class.fields),
                    ),
                    TypeDeclaration::Interface(interface) => self.symbol(
                        &interface.name,
                        SymbolKind::INTERFACE,
                        None,
                        span,
                        self.members(&interface.fields),
                    ),
                    TypeDeclaration::Abstract(abstract_decl) => {
                        let mut members = self.members(&abstract_decl.fields);
                        if abstract_decl.is_enum_abstract {
                            // The values of an enum abstract are its non-static variables.
                            for (member, field) in members.iter_mut().zip(&abstract_decl.fields) {
                                if matches!(
                                    field.kind,
                                    ClassFieldKind::Var { .. } | ClassFieldKind::Final { .. }
                                ) && !field
                                    .modifiers
                                    .contains(&parser::haxe_ast::Modifier::Static)
                                {
                                    member.kind = SymbolKind::ENUM_MEMBER;
                                }
                            }
                        }
                        self.symbol(&abstract_decl.name, SymbolKind::STRUCT, None, span, members)
                    }
                    TypeDeclaration::Enum(enumeration) => {
                        let constructors = enumeration
                            .constructors
                            .iter()
                            .filter_map(|constructor| {
                                self.symbol(
                                    &constructor.name,
                                    SymbolKind::ENUM_MEMBER,
                                    None,
                                    (constructor.span.start, constructor.span.end),
                                    Vec::new(),
                                )
                            })
                            .collect();
                        self.symbol(
                            &enumeration.name,
                            SymbolKind::ENUM,
                            None,
                            span,
                            constructors,
                        )
                    }
                    TypeDeclaration::Typedef(alias) => {
                        let fields = crate::structural_fields::source_fields(&alias.type_def)
                            .iter()
                            .filter_map(|field| {
                                self.symbol(
                                    &field.name,
                                    SymbolKind::FIELD,
                                    None,
                                    (field.span.start, field.span.end),
                                    Vec::new(),
                                )
                            })
                            .collect();
                        self.symbol(&alias.name, SymbolKind::STRUCT, None, span, fields)
                    }
                    _ => None,
                }
            })
            .collect()
    }
}

/// The outline of `source`, or `None` when it does not parse.
pub(crate) fn document_symbols(
    source: &str,
    filename: &str,
    defines: &[String],
) -> Option<Vec<DocumentSymbol>> {
    let mut config = parser::preprocessor::PreprocessorConfig::default();
    config.defines.insert("eval".into());
    config.add_defines(defines.iter());
    let text = parser::preprocessor::preprocess(source, &config);
    let file = parser::rd::rd_parse(&text, filename, false, false).ok()?;
    let tokens = parser::lexer::Lexer::new(&text).tokenize().ok()?;
    let line_starts = std::iter::once(0)
        .chain(text.match_indices('\n').map(|(at, _)| at + 1))
        .collect();
    Some(
        Source {
            text: &text,
            tokens,
            line_starts,
        }
        .declarations(&file),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn outline(source: &str) -> Vec<DocumentSymbol> {
        document_symbols(source, "Main.hx", &[]).unwrap()
    }

    #[test]
    fn members_nest_under_their_type_and_locals_are_left_out() {
        let source = "package p;\nclass Main {\n  public var count:Int = 0;\n  public function new() {}\n  public function add(a:Int, b:Int):Int {\n    var total = a + b;\n    return total;\n  }\n}\nenum Color { Red; Rgb(r:Int); }\ntypedef Point = { x:Int, y:Int };\n";
        let symbols = outline(source);
        let names: Vec<_> = symbols.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["Main", "Color", "Point"]);
        let main = &symbols[0];
        let members: Vec<_> = main
            .children
            .as_ref()
            .unwrap()
            .iter()
            .map(|s| (s.name.as_str(), s.kind))
            .collect();
        assert_eq!(
            members,
            [
                ("count", SymbolKind::FIELD),
                ("new", SymbolKind::CONSTRUCTOR),
                ("add", SymbolKind::METHOD)
            ]
        );
        let all = format!("{symbols:?}");
        assert!(!all.contains("total"), "{all}");
        let colors = symbols[1].children.as_ref().unwrap();
        assert_eq!(colors.len(), 2);
        assert_eq!(colors[1].kind, SymbolKind::ENUM_MEMBER);
        let fields = symbols[2].children.as_ref().unwrap();
        assert_eq!(
            fields.iter().map(|f| f.name.as_str()).collect::<Vec<_>>(),
            ["x", "y"]
        );
    }

    #[test]
    fn a_declaration_covers_its_body_and_selects_its_name() {
        let source = "class Main {\n  public function add(a:Int):Int {\n    return a;\n  }\n}\n";
        let symbols = outline(source);
        let class = &symbols[0];
        assert_eq!(class.range.start, Position::new(0, 0));
        assert_eq!(class.range.end.line, 4);
        assert_eq!(
            class.selection_range,
            Range::new(Position::new(0, 6), Position::new(0, 10))
        );
        let method = &class.children.as_ref().unwrap()[0];
        assert_eq!(method.range.start.line, 1);
        assert_eq!(method.range.end.line, 3);
        assert_eq!(method.selection_range.start, Position::new(1, 18));
        assert_eq!(method.detail.as_deref(), Some("(a:Int):Int"));
    }

    #[test]
    fn columns_count_utf16_units() {
        let source = "class Main {\n  /* é😀 */ var x:Int;\n  function f() {}\n}\n";
        let symbols = outline(source);
        let field = &symbols[0].children.as_ref().unwrap()[0];
        assert_eq!(field.name, "x");
        assert_eq!(field.selection_range.start, Position::new(1, 16));
        assert_eq!(field.selection_range.end, Position::new(1, 17));
    }

    #[test]
    fn source_that_does_not_parse_has_no_outline() {
        assert!(document_symbols("class {{{", "Main.hx", &[]).is_none());
    }
}
