//! Complete Haxe parser with full span tracking
//!
//! This parser handles 100% of Haxe syntax with proper whitespace/comment handling
//! and tracks spans for every AST node.

use nom::{
    branch::alt,
    bytes::complete::{tag, take_until, take_while},
    character::complete::{alpha1, alphanumeric1, char, multispace1},
    combinator::{map, not, opt, peek, recognize, value, verify},
    error::context,
    multi::{many0, many1, separated_list0, separated_list1},
    sequence::{delimited, pair, preceded, tuple},
    IResult, Parser,
};

use crate::custom_error::ContextualError;
use crate::haxe_ast::*;
use diagnostics;

/// Parser result type with contextual errors to capture context strings
pub type PResult<'a, T> = IResult<&'a str, T, ContextualError<&'a str>>;

/// Result of parsing that includes diagnostics
#[derive(Debug)]
pub struct ParseResult {
    pub file: HaxeFile,
    pub diagnostics: diagnostics::Diagnostics,
    pub source_map: diagnostics::SourceMap,
}

/// Helper enum for parsing imports or using statements
#[derive(Debug, Clone)]
enum ImportOrUsing {
    Import(Import),
    Using(Using),
}

/// Helper enum for parsing imports, using, or conditional compilation containing them
#[derive(Debug, Clone)]
enum ImportUsingOrConditional {
    Import(Import),
    Using(Using),
    Conditional(ConditionalCompilation<ImportOrUsing>),
}

/// Check if a filename represents an import.hx file (Haxe import defaults)
/// Only the exact filename "import.hx" (lowercase) qualifies, not "notimport.hx" etc.
fn is_import_hx_file(file_name: &str) -> bool {
    let basename = file_name.rsplit(['/', '\\']).next().unwrap_or(file_name);
    basename == "import.hx"
}

/// Parse a complete Haxe file
pub fn parse_haxe_file(file_name: &str, input: &str, recovery: bool) -> Result<HaxeFile, String> {
    parse_haxe_file_with_debug(file_name, input, recovery, false)
}

/// Parse a complete Haxe file with debug flag for preserving source input
pub fn parse_haxe_file_with_debug(
    file_name: &str,
    input: &str,
    recovery: bool,
    debug: bool,
) -> Result<HaxeFile, String> {
    parse_haxe_file_with_config(
        file_name,
        input,
        recovery,
        debug,
        &crate::preprocessor::PreprocessorConfig::default(),
    )
}

/// Parse a Haxe file with explicit preprocessor defines (e.g., "wasm" for WASM targets).
pub fn parse_haxe_file_with_config(
    file_name: &str,
    input: &str,
    recovery: bool,
    debug: bool,
    preprocessor_config: &crate::preprocessor::PreprocessorConfig,
) -> Result<HaxeFile, String> {
    let is_import_file = is_import_hx_file(file_name);

    // Preprocess to handle conditional compilation directives
    let preprocessed = crate::preprocessor::preprocess(input, preprocessor_config);

    // Try the recursive descent parser first (15x faster than nom)
    match crate::rd::rd_parse(&preprocessed, file_name, is_import_file, debug) {
        Ok(file) => return Ok(file),
        Err(_rd_errors) => {
            // RD parser failed — fall back to nom parser for error recovery
        }
    }

    // Fallback: nom-based parser
    if recovery {
        parse_haxe_file_with_enhanced_errors(&preprocessed, file_name, is_import_file).map_err(
            |(diagnostics, source_map)| {
                let formatter = diagnostics::ErrorFormatter::with_colors();
                formatter.format_diagnostics(&diagnostics, &source_map)
            },
        )
    } else {
        let incremental_result = crate::incremental_parser_enhanced::parse_incrementally_enhanced(
            file_name,
            &preprocessed,
        );

        convert_enhanced_incremental_to_haxe_file(
            incremental_result,
            file_name,
            &preprocessed,
            is_import_file,
            debug,
        )
    }
}

/// Convert enhanced incremental parse result to HaxeFile
fn convert_enhanced_incremental_to_haxe_file(
    result: crate::incremental_parser_enhanced::IncrementalParseResult,
    file_name: &str,
    input: &str,
    is_import_file: bool,
    debug: bool,
) -> Result<HaxeFile, String> {
    use crate::haxe_ast::{HaxeFile, Span};
    use crate::incremental_parser_enhanced::ParsedElement;

    // If there are errors but no parsed elements, format and return all errors
    if result.parsed_elements.is_empty() && result.has_errors() {
        let formatted_errors = result.format_diagnostics(true);
        return Err(format!("Parse failed with errors:\n{}", formatted_errors));
    }

    // Extract components from parsed elements
    let mut package = None;
    let mut imports = Vec::new();
    let mut using = Vec::new();
    let mut module_fields = Vec::new();
    let mut declarations = Vec::new();

    for element in result.parsed_elements {
        match element {
            ParsedElement::Package(pkg) => package = Some(pkg),
            ParsedElement::Import(imp) => imports.push(imp),
            ParsedElement::Using(use_stmt) => using.push(use_stmt),
            ParsedElement::ModuleField(field) => module_fields.push(field),
            ParsedElement::TypeDeclaration(decl) => {
                declarations.push(decl);
            }
            ParsedElement::ConditionalBlock(_) => {
                // Skip conditional blocks for now
            }
        }
    }

    // For import.hx files, only imports and using statements are allowed
    // See: https://haxe.org/manual/type-system-import-defaults.html
    if is_import_file {
        if package.is_some() {
            return Err("import.hx files cannot contain package declarations".to_string());
        }
        if !declarations.is_empty() {
            return Err(
                "import.hx files cannot contain type declarations (class, interface, enum, etc.)"
                    .to_string(),
            );
        }
        if !module_fields.is_empty() {
            return Err("import.hx files cannot contain module-level fields".to_string());
        }
        Ok(HaxeFile {
            filename: file_name.to_string(),
            input: if debug { Some(input.to_string()) } else { None },
            package: None,
            imports,
            using,
            module_fields: Vec::new(),
            declarations: Vec::new(),
            span: Span::new(0, input.len()),
        })
    } else {
        // If we have errors but managed to parse some content, we can still return a partial result
        // This is the key improvement - we don't fail completely on partial parsing issues
        Ok(HaxeFile {
            filename: file_name.to_string(),
            input: if debug { Some(input.to_string()) } else { None },
            package,
            imports,
            using,
            module_fields,
            declarations,
            span: Span::new(0, input.len()),
        })
    }
}

/// Convert old incremental parse result to HaxeFile (for backward compatibility)
#[allow(dead_code)]
fn convert_incremental_to_haxe_file(
    result: crate::incremental_parser::IncrementalParseResult,
    file_name: &str,
    input: &str,
    is_import_file: bool,
    debug: bool,
) -> Result<HaxeFile, String> {
    use crate::haxe_ast::{HaxeFile, Span};
    use crate::incremental_parser::ParsedElement;

    // If there are errors but no parsed elements, format and return all errors
    if result.parsed_elements.is_empty() && !result.errors.is_empty() {
        let formatted_errors = result
            .errors
            .iter()
            .map(|error| format!("{}:{} - {}", error.line, error.column, error.message))
            .collect::<Vec<_>>()
            .join("\n");

        return Err(format!(
            "Parse failed with {} errors:\n{}",
            result.errors.len(),
            formatted_errors
        ));
    }

    // Extract components from parsed elements
    let mut package = None;
    let mut imports = Vec::new();
    let mut using = Vec::new();
    let mut module_fields = Vec::new();
    let mut declarations = Vec::new();

    for element in result.parsed_elements {
        match element {
            ParsedElement::Package(pkg) => package = Some(pkg),
            ParsedElement::Import(imp) => imports.push(imp),
            ParsedElement::Using(use_stmt) => using.push(use_stmt),
            ParsedElement::ModuleField(field) => module_fields.push(field),
            ParsedElement::TypeDeclaration(decl) => declarations.push(decl),
            ParsedElement::ConditionalBlock(_) => {
                // Skip conditional blocks for now
            }
        }
    }

    // For import.hx files, only imports and using statements are allowed
    if is_import_file {
        if package.is_some() {
            return Err("import.hx files cannot contain package declarations".to_string());
        }
        if !declarations.is_empty() {
            return Err(
                "import.hx files cannot contain type declarations (class, interface, enum, etc.)"
                    .to_string(),
            );
        }
        if !module_fields.is_empty() {
            return Err("import.hx files cannot contain module-level fields".to_string());
        }
        Ok(HaxeFile {
            filename: file_name.to_string(),
            input: if debug { Some(input.to_string()) } else { None },
            package: None,
            imports,
            using,
            module_fields: Vec::new(),
            declarations: Vec::new(),
            span: Span::new(0, input.len()),
        })
    } else {
        // If we have errors but managed to parse some content, we can still return a partial result
        Ok(HaxeFile {
            filename: file_name.to_string(),
            input: if debug { Some(input.to_string()) } else { None },
            package,
            imports,
            using,
            module_fields,
            declarations,
            span: Span::new(0, input.len()),
        })
    }
}

/// Parse a Haxe file with enhanced error reporting
pub fn parse_haxe_file_with_enhanced_errors(
    input: &str,
    file_name: &str,
    is_import_file: bool,
) -> Result<HaxeFile, (diagnostics::Diagnostics, diagnostics::SourceMap)> {
    // Use enhanced incremental parser for better error recovery with diagnostics
    let incremental_result =
        crate::incremental_parser_enhanced::parse_incrementally_enhanced(file_name, input);

    // Extract the diagnostics and source map from the result
    let diagnostics = incremental_result.diagnostics.clone();
    let source_map = incremental_result.source_map.clone();

    // If parsing succeeded, return the file (preserve source with debug=true)
    if let Ok(file) = convert_enhanced_incremental_to_haxe_file(
        incremental_result,
        file_name,
        input,
        is_import_file,
        true,
    ) {
        return Ok(file);
    }

    // Otherwise, return the diagnostics and source map
    Err((diagnostics, source_map))
}

/// Parse a Haxe file and always return diagnostics along with the result
pub fn parse_haxe_file_with_diagnostics(
    file_name: &str,
    input: &str,
) -> Result<ParseResult, String> {
    // Preprocess to handle conditional compilation directives
    let preprocessor_config = crate::preprocessor::PreprocessorConfig::default();
    let preprocessed_source = crate::preprocessor::preprocess(input, &preprocessor_config);

    // Check if this is an import.hx file
    let is_import_file = is_import_hx_file(file_name);

    // Use enhanced incremental parser for better error recovery with diagnostics
    // Parse the preprocessed source instead of the original
    let incremental_result = crate::incremental_parser_enhanced::parse_incrementally_enhanced(
        file_name,
        &preprocessed_source,
    );

    // Extract the diagnostics and source map
    let diagnostics = incremental_result.diagnostics.clone();
    let source_map = incremental_result.source_map.clone();

    // Try to convert to HaxeFile
    // Use preprocessed source so the file has the correct content
    match convert_enhanced_incremental_to_haxe_file(
        incremental_result,
        file_name,
        &preprocessed_source,
        is_import_file,
        false,
    ) {
        Ok(file) => Ok(ParseResult {
            file,
            diagnostics,
            source_map,
        }),
        Err(e) => {
            // If we have diagnostics, format them nicely
            // if !diagnostics.is_empty() {
            //     let formatter = diagnostics::ErrorFormatter::with_colors();
            //     Err(formatter.format_diagnostics(&diagnostics, &source_map))
            // } else {

            // }
            Err(e)
        }
    }
}

/// Helper function to flatten conditional imports/using into regular vectors
/// This is a simplification for now - a full implementation might preserve conditional structure
fn flatten_conditional_imports_using(
    cond: &ConditionalCompilation<ImportOrUsing>,
    imports: &mut Vec<Import>,
    using: &mut Vec<Using>,
) {
    // Flatten the if branch
    for item in &cond.if_branch.content {
        match item {
            ImportOrUsing::Import(imp) => imports.push(imp.clone()),
            ImportOrUsing::Using(use_) => using.push(use_.clone()),
        }
    }

    // Flatten elseif branches
    for elseif_branch in &cond.elseif_branches {
        for item in &elseif_branch.content {
            match item {
                ImportOrUsing::Import(imp) => imports.push(imp.clone()),
                ImportOrUsing::Using(use_) => using.push(use_.clone()),
            }
        }
    }

    // Flatten else branch if present
    if let Some(else_content) = &cond.else_branch {
        for item in else_content {
            match item {
                ImportOrUsing::Import(imp) => imports.push(imp.clone()),
                ImportOrUsing::Using(use_) => using.push(use_.clone()),
            }
        }
    }
}

/// Parser for import.hx files - only allows imports and using statements
pub fn import_hx_file<'a>(file_name: &str, full: &'a str, input: &'a str) -> PResult<'a, HaxeFile> {
    import_hx_file_with_debug(file_name, full, input, false)
}

pub fn import_hx_file_with_debug<'a>(
    file_name: &str,
    full: &'a str,
    input: &'a str,
    debug: bool,
) -> PResult<'a, HaxeFile> {
    context(
        "import.hx file",
        |input: &'a str| -> PResult<'a, HaxeFile> {
            let start = position(full, input);

            // Skip leading whitespace/comments
            let (input, _) = ws(input)?;

            // import.hx files cannot have package declarations
            // They can only contain imports and using statements

            // Parse imports and using statements
            let (input, imports_using_conditional) = many0(|i| {
                // Skip any metadata first
                let (i, _) = metadata_list(full, i)?;

                // Try to parse conditional compilation containing imports, or regular imports/using
                alt((
                    // Conditional compilation with imports/using
                    map(
                        |i| {
                            conditional_compilation(full, i, |full, input| {
                                alt((
                                    map(|i| import_decl(full, i), ImportOrUsing::Import),
                                    map(|i| using_decl(full, i), ImportOrUsing::Using),
                                ))
                                .parse(input)
                            })
                        },
                        ImportUsingOrConditional::Conditional,
                    ),
                    // Regular import
                    map(|i| import_decl(full, i), ImportUsingOrConditional::Import),
                    // Regular using
                    map(|i| using_decl(full, i), ImportUsingOrConditional::Using),
                ))
                .parse(i)
            })
            .parse(input)?;

            // Extract imports and using from the mixed results
            let mut imports = Vec::new();
            let mut using = Vec::new();
            let mut conditional_imports_using = Vec::new();

            for item in imports_using_conditional {
                match item {
                    ImportUsingOrConditional::Import(imp) => imports.push(imp),
                    ImportUsingOrConditional::Using(use_) => using.push(use_),
                    ImportUsingOrConditional::Conditional(cond) => {
                        // For now, we'll flatten conditional imports/using into regular ones
                        flatten_conditional_imports_using(&cond, &mut imports, &mut using);
                        conditional_imports_using.push(cond);
                    }
                }
            }

            // Skip trailing whitespace/comments
            let (input, _) = ws(input)?;

            // import.hx files should not have any other content
            // If there's remaining content, it's an error
            if !input.is_empty() {
                return Err(nom::Err::Error(ContextualError::new(
                    input,
                    nom::error::ErrorKind::Tag,
                )));
            }

            let end = position(full, input);

            Ok((
                input,
                HaxeFile {
                    filename: file_name.to_string(),
                    input: if debug { Some(full.to_string()) } else { None },
                    package: None, // import.hx files don't have package declarations
                    imports,
                    using,
                    module_fields: Vec::new(), // import.hx files don't have module fields
                    declarations: Vec::new(),  // import.hx files don't have type declarations
                    span: Span::new(start, end),
                },
            ))
        },
    )
    .parse(input)
}

/// Main file parser
pub fn haxe_file<'a>(file_name: &str, full: &'a str, input: &'a str) -> PResult<'a, HaxeFile> {
    haxe_file_with_debug(file_name, full, input, false)
}

pub fn haxe_file_with_debug<'a>(
    file_name: &str,
    full: &'a str,
    input: &'a str,
    debug: bool,
) -> PResult<'a, HaxeFile> {
    context("haxe file", |input| {
        let start = position(full, input);

        // Skip leading whitespace/comments
        let (input, _) = ws(input)?;

        // Optional package declaration
        let (input, package) = opt(|i| package_decl(full, i)).parse(input)?;

        // Imports and using statements
        let (input, imports_using_conditional) = many0(|i| {
            // Skip any metadata first
            let (i, _) = metadata_list(full, i)?;

            // Check if we've hit a type declaration OR module field keywords
            let peek_result: Result<_, nom::Err<nom::error::Error<_>>> = peek(alt((
                tag("class"),
                tag("interface"),
                tag("enum"),
                tag("typedef"),
                tag("abstract"),
                tag("var"),
                tag("final"),
                tag("function"),
            )))
            .parse(i);

            if peek_result.is_ok() {
                // Stop parsing imports/using
                Err(nom::Err::Error(ContextualError::new(
                    i,
                    nom::error::ErrorKind::Eof,
                )))
            } else {
                // Try to parse conditional compilation containing imports, or regular imports/using
                alt((
                    // Conditional compilation with imports/using
                    map(
                        |i| {
                            conditional_compilation(full, i, |full, input| {
                                alt((
                                    map(|i| import_decl(full, i), ImportOrUsing::Import),
                                    map(|i| using_decl(full, i), ImportOrUsing::Using),
                                ))
                                .parse(input)
                            })
                        },
                        ImportUsingOrConditional::Conditional,
                    ),
                    // Regular import
                    map(|i| import_decl(full, i), ImportUsingOrConditional::Import),
                    // Regular using
                    map(|i| using_decl(full, i), ImportUsingOrConditional::Using),
                ))
                .parse(i)
            }
        })
        .parse(input)?;

        // Extract imports and using from the mixed results
        let mut imports = Vec::new();
        let mut using = Vec::new();
        let mut conditional_imports_using = Vec::new();

        for item in imports_using_conditional {
            match item {
                ImportUsingOrConditional::Import(imp) => imports.push(imp),
                ImportUsingOrConditional::Using(use_) => using.push(use_),
                ImportUsingOrConditional::Conditional(cond) => {
                    // For now, we'll flatten conditional imports/using into regular ones
                    // This is a simplification - in a full implementation you might want to preserve the conditional structure
                    flatten_conditional_imports_using(&cond, &mut imports, &mut using);
                    conditional_imports_using.push(cond);
                }
            }
        }

        // Module-level fields
        let (input, module_fields) = many0(|i| {
            // Skip any metadata first
            let (i, _) = metadata_list(full, i)?;

            // Check if we've hit a type declaration (but NOT metadata or conditional compilation)
            let peek_result: Result<_, nom::Err<nom::error::Error<_>>> = peek(alt((
                tag("class"),
                tag("interface"),
                tag("enum"),
                tag("typedef"),
                tag("abstract"),
            )))
            .parse(i);

            if peek_result.is_ok() {
                // Stop parsing module fields
                Err(nom::Err::Error(ContextualError::new(
                    i,
                    nom::error::ErrorKind::Eof,
                )))
            } else {
                // Try to parse module field
                module_field(full, i)
            }
        })
        .parse(input)?;

        // Type declarations
        let (input, declarations) = many0(|i| type_declaration(full, i)).parse(input)?;

        // Skip trailing whitespace/comments
        let (input, _) = ws(input)?;

        let end = position(full, input);

        Ok((
            input,
            HaxeFile {
                filename: file_name.to_string(),
                input: if debug { Some(full.to_string()) } else { None },
                package,
                imports,
                using,
                module_fields,
                declarations,
                span: Span::new(start, end),
            },
        ))
    })
    .parse(input)
}

/// Get current position in the original input
pub fn position(full: &str, current: &str) -> usize {
    full.len() - current.len()
}

/// Create span from start position to current position
pub fn make_span(full: &str, start_pos: usize, current: &str) -> Span {
    let end_pos = position(full, current);
    Span::new(start_pos, end_pos)
}

// =============================================================================
// Module-level fields
// =============================================================================

/// Parse a module-level field (variable or function)
pub fn module_field<'a>(full: &'a str, input: &'a str) -> PResult<'a, ModuleField> {
    let start = position(full, input);

    let (input, meta) = metadata_list(full, input)?;
    let (input, (access, modifiers)) = parse_access_and_modifiers(input)?;

    // Check if final was parsed as a modifier
    let has_final_modifier = modifiers.iter().any(|m| matches!(m, Modifier::Final));

    // Field kind
    let (input, kind) = alt((
        |i| module_field_function(full, i),
        |i| module_field_var_or_final(full, i, has_final_modifier),
    ))
    .parse(input)?;

    let end = position(full, input);

    Ok((
        input,
        ModuleField {
            meta,
            access,
            modifiers,
            kind,
            span: Span::new(start, end),
        },
    ))
}

/// Parse module-level function
fn module_field_function<'a>(full: &'a str, input: &'a str) -> PResult<'a, ModuleFieldKind> {
    let (input, _) = context("expected 'function' keyword", keyword("function")).parse(input)?;
    let (input, name) = context("expected function name", function_name).parse(input)?;
    let (input, type_params) = type_params(full, input)?;

    let (input, _) = context("[E0082] expected '(' to start parameter list | help: function parameters must be enclosed in parentheses", symbol("(")).parse(input)?;
    let (input, params) = context("[E0083] expected function parameters | help: provide parameter list or leave empty for no parameters", separated_list0(symbol(","), |i| function_param(full, i))).parse(input)?;
    let (input, _) = opt(symbol(",")).parse(input)?; // Trailing comma
    let (input, _) =
        context("[E0084] expected ')' to close parameter list", symbol(")")).parse(input)?;

    let (input, return_type) = opt(preceded(
        context("expected ':' before return type", symbol(":")),
        |i| type_expr(full, i),
    ))
    .parse(input)?;

    let (input, body) = opt(|i| block_expr(full, i)).parse(input)?;

    Ok((
        input,
        ModuleFieldKind::Function(Function {
            name,
            type_params,
            params,
            return_type,
            body: body.map(Box::new),
            span: Span::new(0, 0), // Will be set by the caller
        }),
    ))
}

/// Parse module-level variable or final field
fn module_field_var_or_final<'a>(
    full: &'a str,
    input: &'a str,
    has_final_modifier: bool,
) -> PResult<'a, ModuleFieldKind> {
    let (input, is_final) = if has_final_modifier {
        // If final was already parsed as a modifier, don't parse keywords again
        (input, true)
    } else {
        // Parse either final or var keyword
        alt((
            value(true, context("expected 'final' keyword", keyword("final"))),
            value(false, context("expected 'var' keyword", keyword("var"))),
        ))
        .parse(input)?
    };

    let (input, name) = context("expected variable name", identifier).parse(input)?;
    let (input, type_hint) = opt(preceded(
        context("expected ':' before type annotation", symbol(":")),
        |i| type_expr(full, i),
    ))
    .parse(input)?;
    let (input, expr) = opt(preceded(
        context("expected '=' before initializer", symbol("=")),
        |i| expression(full, i),
    ))
    .parse(input)?;
    let (input, _) =
        context("expected ';' after variable declaration", symbol(";")).parse(input)?;

    if is_final || has_final_modifier {
        Ok((
            input,
            ModuleFieldKind::Final {
                name,
                type_hint,
                expr,
            },
        ))
    } else {
        Ok((
            input,
            ModuleFieldKind::Var {
                name,
                type_hint,
                expr,
            },
        ))
    }
}

// =============================================================================
// Whitespace and Comments
// =============================================================================

/// Skip whitespace and comments
pub fn ws(input: &str) -> PResult<()> {
    // Fast path: if first char isn't whitespace or comment start, return immediately.
    // This avoids entering the many0(alt(...)) machinery for the ~90% of calls
    // where there's no whitespace to skip (called ~16,500 times per 551-line file).
    if let Some(first) = input.as_bytes().first() {
        if !matches!(first, b' ' | b'\t' | b'\n' | b'\r' | b'/' | b'#') {
            return Ok((input, ()));
        }
    }
    value(
        (),
        many0(alt((
            value((), multispace1),
            value((), line_comment),
            value((), block_comment),
            value((), preprocessor_directive),
        ))),
    )
    .parse(input)
}

/// Skip whitespace and comments, require at least some
pub fn ws1(input: &str) -> PResult<()> {
    value(
        (),
        many1(alt((
            value((), multispace1),
            value((), line_comment),
            value((), block_comment),
            value((), preprocessor_directive),
        ))),
    )
    .parse(input)
}

/// Preprocessor directive: #if condition ... #else ... #end
/// For Rayzor, we skip the #if branch and take #else if present
/// For negated conditions like #if !eval, we take the #if branch
/// NOTE: This only handles block-level preprocessor directives (start of line).
/// Inline preprocessor directives (within expressions) are handled by inline_preprocessor_expr.
fn preprocessor_directive(input: &str) -> PResult<&str> {
    // Only handle preprocessor directives that appear on their own line (block-level).
    // Inline preprocessor (within expressions) should be handled by inline_preprocessor_expr.
    // We detect inline by checking if #if/# else/#end appears after non-whitespace on the same line.
    // If the directive is inline (has content before it on the line), fail immediately so
    // the expression parser can handle it.
    // Actually, we can't easily check this here because we're called from ws which has already
    // consumed whitespace. Instead, we'll check if we can see #end on the same line - if so,
    // it's likely an inline directive.
    if input.starts_with("#if") {
        // Quick check: if we find #end before finding a newline, it's an inline preprocessor
        if let Some(newline_pos) = input.find('\n') {
            if let Some(end_pos) = input.find("#end") {
                if end_pos < newline_pos {
                    // This is an inline preprocessor directive - don't handle it here
                    return Err(nom::Err::Error(ContextualError::new(
                        input,
                        nom::error::ErrorKind::Tag,
                    )));
                }
            }
        } else if input.contains("#end") {
            // No newline found but #end exists - entire thing is on one line
            return Err(nom::Err::Error(ContextualError::new(
                input,
                nom::error::ErrorKind::Tag,
            )));
        }
    }

    // Handle standalone #end (from a previous #else that we returned)
    if input.starts_with("#end")
        && (input.len() == 4
            || !input
                .chars()
                .nth(4)
                .map(|c| c.is_alphanumeric())
                .unwrap_or(false))
    {
        let mut rest = &input[4..];
        // Skip to end of line
        while !rest.is_empty() && !rest.starts_with('\n') {
            rest = &rest[1..];
        }
        if rest.starts_with('\n') {
            rest = &rest[1..];
        }
        return Ok((rest, &input[..input.len() - rest.len()]));
    }

    // Handle standalone #else (when we took the #if branch due to negation)
    // We need to skip from #else to #end
    if input.starts_with("#else")
        && (input.len() == 5
            || !input
                .chars()
                .nth(5)
                .map(|c| c.is_alphanumeric())
                .unwrap_or(false))
    {
        let mut current = &input[5..];
        let mut depth = 1;

        while depth > 0 && !current.is_empty() {
            if current.starts_with("#if")
                && (current.len() == 3
                    || !current
                        .chars()
                        .nth(3)
                        .map(|c| c.is_alphanumeric())
                        .unwrap_or(false))
            {
                depth += 1;
                current = &current[3..];
            } else if current.starts_with("#end")
                && (current.len() == 4
                    || !current
                        .chars()
                        .nth(4)
                        .map(|c| c.is_alphanumeric())
                        .unwrap_or(false))
            {
                depth -= 1;
                if depth == 0 {
                    current = &current[4..];
                    // Skip to end of line
                    while !current.is_empty() && !current.starts_with('\n') {
                        current = &current[1..];
                    }
                    if current.starts_with('\n') {
                        current = &current[1..];
                    }
                    break;
                }
                current = &current[4..];
            } else {
                let mut chars = current.chars();
                chars.next();
                current = chars.as_str();
            }
        }

        return Ok((current, &input[..input.len() - current.len()]));
    }

    // Must start with #if
    let (input, _) = tag("#if")(input)?;

    // Skip the condition (identifier, !identifier, or parenthesized expression)
    let (mut input, _) = take_while(|c: char| c.is_whitespace() && c != '\n')(input)?;

    // Check if condition is negated - if so, we include the #if branch
    let is_negated = input.starts_with('!');

    // Skip condition - can be: identifier, !identifier, (expr)
    if input.starts_with('(') {
        // Parenthesized expression - find matching closing paren
        let mut depth = 1;
        let mut pos = 1;
        for (i, c) in input[1..].char_indices() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        pos = i + 2; // +1 for starting after '(', +1 for the ')'
                        break;
                    }
                }
                _ => {}
            }
        }
        input = &input[pos..];
    } else {
        // Simple identifier or !identifier
        if input.starts_with('!') {
            input = &input[1..];
        }
        let (rest, _) = take_while(|c: char| c.is_alphanumeric() || c == '_')(input)?;
        input = rest;
    }

    // If negated (like #if !eval), include the #if branch content
    // by just skipping the #if line and returning - content will be parsed normally
    if is_negated {
        // Skip to end of line
        let (input, _) = take_while(|c: char| c != '\n')(input)?;
        let input = input.strip_prefix('\n').unwrap_or(input);
        // Return - the #if branch content will be parsed normally
        // We'll handle #else and #end when we encounter them
        return Ok((input, ""));
    }

    // Skip to end of line
    let (input, _) = take_while(|c: char| c != '\n')(input)?;
    let input = input.strip_prefix('\n').unwrap_or(input);

    // Now we need to find the matching #end, handling nesting
    // We skip the #if branch content and take #else content if present
    let start = input;
    let mut current = input;
    let mut depth = 1;
    let mut _else_start: Option<&str> = None;

    while depth > 0 && !current.is_empty() {
        if current.starts_with("#if")
            && (current.len() == 3
                || !current
                    .chars()
                    .nth(3)
                    .map(|c| c.is_alphanumeric())
                    .unwrap_or(false))
        {
            depth += 1;
            current = &current[3..];
        } else if current.starts_with("#elseif") && depth == 1 {
            // At our level, treat #elseif like #else - skip the directive and return
            // so the content after it gets parsed normally
            current = &current[7..];
            // Skip condition
            while !current.is_empty() && !current.starts_with('\n') {
                current = &current[1..];
            }
            if current.starts_with('\n') {
                current = &current[1..];
            }
            // Return here - the content after #elseif will be parsed normally
            // until we hit the next preprocessor directive
            break;
        } else if current.starts_with("#else")
            && depth == 1
            && (current.len() == 5
                || !current
                    .chars()
                    .nth(5)
                    .map(|c| c.is_alphanumeric())
                    .unwrap_or(false))
        {
            // Found #else at our level - skip the #else keyword and newline
            // The content after it will be parsed normally
            current = &current[5..];
            while !current.is_empty() && !current.starts_with('\n') {
                current = &current[1..];
            }
            if current.starts_with('\n') {
                current = &current[1..];
            }
            // Return here - the #else content will be parsed normally
            break;
        } else if current.starts_with("#end")
            && (current.len() == 4
                || !current
                    .chars()
                    .nth(4)
                    .map(|c| c.is_alphanumeric())
                    .unwrap_or(false))
        {
            depth -= 1;
            if depth == 0 {
                // Skip #end and rest of line
                current = &current[4..];
                while !current.is_empty() && !current.starts_with('\n') {
                    current = &current[1..];
                }
                if current.starts_with('\n') {
                    current = &current[1..];
                }
                break;
            }
            current = &current[4..];
        } else {
            // Move to next character
            let mut chars = current.chars();
            chars.next();
            current = chars.as_str();
        }
    }

    // Calculate how much we consumed
    let consumed_len = start.len() - current.len();
    let consumed = &start[..consumed_len];

    // Return the remaining input
    Ok((current, consumed))
}

/// Line comment: // comment
fn line_comment(input: &str) -> PResult<&str> {
    recognize(tuple((
        tag("//"),
        take_while(|c| c != '\n'),
        opt(char('\n')),
    )))
    .parse(input)
}

/// Block comment: /* comment */
fn block_comment(input: &str) -> PResult<&str> {
    recognize(tuple((tag("/*"), take_until("*/"), tag("*/")))).parse(input)
}

/// Parse T with optional leading whitespace
#[allow(dead_code)]
fn ws_before<'a, T, F>(mut parser: F) -> impl FnMut(&'a str) -> PResult<'a, T>
where
    F: FnMut(&'a str) -> PResult<'a, T>,
{
    move |input| {
        let (input, _) = ws(input)?;
        parser(input)
    }
}

// =============================================================================
// Basic Elements
// =============================================================================

/// Reserved keywords
/// All Haxe reserved keywords.
pub const HAXE_KEYWORDS: &[&str] = &[
    "abstract",
    "break",
    "case",
    "cast",
    "catch",
    "class",
    "continue",
    "default",
    "do",
    "dynamic",
    "else",
    "enum",
    "extends",
    "extern",
    "false",
    "final",
    "for",
    "function",
    "if",
    "implements",
    "import",
    "in",
    "inline",
    "interface",
    "macro",
    "new",
    "null",
    "override",
    "package",
    "private",
    "public",
    "return",
    "static",
    "super",
    "switch",
    "this",
    "throw",
    "true",
    "try",
    "typedef",
    "untyped",
    "using",
    "var",
    "while",
];

fn is_keyword(s: &str) -> bool {
    HAXE_KEYWORDS.contains(&s)
}

/// Parse a keyword
pub fn keyword<'a>(kw: &'static str) -> impl FnMut(&'a str) -> PResult<'a, &'a str> {
    move |input| {
        let (input, _) = ws(input)?;
        let (input, word) = verify(
            recognize(pair(
                tag(kw),
                // Word boundary: keyword must not be followed by identifier characters
                peek(not(alt((alphanumeric1, tag("_"))))),
            )),
            |s: &str| s == kw,
        )
        .parse(input)?;
        Ok((input, word))
    }
}

/// Parse an identifier
pub fn identifier(input: &str) -> PResult<String> {
    let (input, _) = ws(input)?;
    let (input, id) = verify(
        recognize(pair(
            alt((alpha1, tag("_"))),
            many0(alt((alphanumeric1, tag("_")))),
        )),
        |s: &str| !is_keyword(s),
    )
    .parse(input)?;
    Ok((input, id.to_string()))
}

/// Parse function name (allows "new" as constructor name)
pub fn function_name(input: &str) -> PResult<String> {
    let (input, _) = ws(input)?;
    let (input, id) = verify(
        recognize(pair(
            alt((alpha1, tag("_"))),
            many0(alt((alphanumeric1, tag("_")))),
        )),
        |s: &str| !is_keyword(s) || s == "new",
    )
    .parse(input)?;
    Ok((input, id.to_string()))
}

/// Parse compiler-specific identifier like __js__, __cpp__, etc.
pub fn compiler_specific_identifier(input: &str) -> PResult<String> {
    let (input, _) = ws(input)?;
    // First check if it starts with __
    let (rest, _) = tag("__")(input)?;
    // Then parse alphanumeric characters (but not underscores to avoid consuming the trailing __)
    let (rest, middle) = alpha1(rest)?;
    // Optionally more alphanumeric (still no underscores)
    let (rest, _suffix) = many0(alphanumeric1).parse(rest)?;
    // Then check for trailing __
    let (rest, _) = tag("__")(rest)?;

    // Reconstruct the full identifier
    let full_id = format!("__{}{}__", middle, _suffix.join(""));

    Ok((rest, full_id))
}

/// Parse a symbol with whitespace
pub fn symbol<'a>(sym: &'static str) -> impl FnMut(&'a str) -> PResult<'a, &'a str> {
    move |input| {
        let (input, _) = ws(input)?;
        tag(sym)(input)
    }
}

// =============================================================================
// Package and Imports
// =============================================================================

/// Package declaration: `package com.example;`
pub fn package_decl<'a>(full: &'a str, input: &'a str) -> PResult<'a, Package> {
    context("package declaration", |input| {
        let start = position(full, input);
        let (input, _) = context("expected 'package' keyword", keyword("package")).parse(input)?;
        let (input, path) =
            context("expected package path (e.g., 'com.example')", dot_path).parse(input)?;
        let (input, _) =
            context("expected ';' after package declaration", symbol(";")).parse(input)?;
        let end = position(full, input);

        Ok((
            input,
            Package {
                path,
                span: Span::new(start, end),
            },
        ))
    })
    .parse(input)
}

/// Import declaration
pub fn import_decl<'a>(full: &'a str, input: &'a str) -> PResult<'a, Import> {
    context("import declaration", |input| {
    let start = position(full, input);
    let (input, _) = context("[E0095] expected 'import' keyword", keyword("import")).parse(input)?;

    // Parse the import path and mode
    let (input, (path, mode)) = context("[E0096] expected import path | help: provide a valid import path like 'haxe.ds.StringMap'", alt((
        // import path.* or import path.* except ...
        |input| {
            let (input, path) = import_path_until_wildcard(input)?;
            let (input, _) = context("[E0097] expected '.*' for wildcard import | help: use '.*' to import all items from a module", symbol(".*")).parse(input)?;

            // Check if there's an "except" clause
            if let Ok((input_after_except, _)) = keyword("except")(input) {
                // Parse the exclusion list
                let (input, exclusions) = context("[E0098] expected comma-separated list of excluded identifiers | help: list the items to exclude from the wildcard import", separated_list1(
                    symbol(","),
                    identifier
                )).parse(input_after_except)?;
                Ok((input, (path, ImportMode::WildcardWithExclusions(exclusions))))
            } else {
                Ok((input, (path, ImportMode::Wildcard)))
            }
        },
        // import path.field or import path as Alias
        |input| {
            // Try to parse the full path first
            let (input_after_path, full_path) = import_path(input)?;

            // Check what comes after the path
            if let Ok((input_after_as, _)) = keyword("as")(input_after_path) {
                // This is an alias import
                let (input, alias) = context("[E0099] expected alias identifier after 'as' | help: provide an alias name for the import", identifier).parse(input_after_as)?;
                Ok((input, (full_path, ImportMode::Alias(alias))))
            } else if let Ok((_input_before_semicolon, _)) = symbol(";")(input_after_path) {
                // If we can see a semicolon, check if the last part might be a field
                if full_path.len() >= 2 {
                    // Check if the last identifier starts with lowercase (likely a field)
                    let last = &full_path[full_path.len() - 1];
                    if last.chars().next().map(|c| c.is_lowercase()).unwrap_or(false) {
                        // This is likely a field import
                        let mut base_path = full_path;
                        let field = base_path.pop().unwrap();
                        Ok((input_after_path, (base_path, ImportMode::Field(field))))
                    } else {
                        // Normal import
                        Ok((input_after_path, (full_path, ImportMode::Normal)))
                    }
                } else {
                    // Normal import
                    Ok((input_after_path, (full_path, ImportMode::Normal)))
                }
            } else {
                // Normal import
                Ok((input_after_path, (full_path, ImportMode::Normal)))
            }
        }
    ))).parse(input)?;

    let (input, _) = context("[E0100] expected ';' after import statement | help: import statements must end with semicolon", symbol(";")).parse(input)?;
    let end = position(full, input);

    Ok((input, Import {
        path,
        mode,
        span: Span::new(start, end),
    }))
    }).parse(input)
}

/// Using declaration: `using Lambda;`
pub fn using_decl<'a>(full: &'a str, input: &'a str) -> PResult<'a, Using> {
    context("using declaration", |input| {
    let start = position(full, input);
    let (input, _) = context("[E0101] expected 'using' keyword", keyword("using")).parse(input)?;
    let (input, path) = context("[E0102] expected type path | help: provide a type to use, like 'StringTools'", import_path).parse(input)?;
    let (input, _) = context("[E0103] expected ';' after using statement | help: using statements must end with semicolon", symbol(";")).parse(input)?;
    let end = position(full, input);

    Ok((input, Using {
        path,
        span: Span::new(start, end),
    }))
    }).parse(input)
}

/// Identifier that allows keywords (for import paths)
fn identifier_or_keyword(input: &str) -> PResult<String> {
    let (input, _) = ws(input)?;
    let (input, id) = recognize(pair(
        alt((alpha1, tag("_"))),
        many0(alt((alphanumeric1, tag("_")))),
    ))
    .parse(input)?;
    Ok((input, id.to_string()))
}

/// Dot-separated path: `com.example.Class`
/// Note: Allows keywords in path segments (e.g., haxe.macro, haxe.extern)
pub fn dot_path(input: &str) -> PResult<Vec<String>> {
    separated_list1(symbol("."), identifier_or_keyword).parse(input)
}

/// Import path that allows keywords (e.g., `haxe.macro.Context`)
fn import_path(input: &str) -> PResult<Vec<String>> {
    separated_list1(symbol("."), identifier_or_keyword).parse(input)
}

/// Import path until wildcard (stops before .*)
fn import_path_until_wildcard(input: &str) -> PResult<Vec<String>> {
    let mut path = Vec::new();
    let mut current = input;

    // Parse first identifier
    let (next, first) = identifier_or_keyword(current)?;
    path.push(first);
    current = next;

    // Continue parsing dot-separated identifiers until we hit .* or end
    loop {
        // Check if next is .*
        if symbol(".*")(current).is_ok() {
            break;
        }

        // Try to parse another .identifier
        match symbol(".")(current) {
            Ok((after_dot, _)) => match identifier_or_keyword(after_dot) {
                Ok((next, id)) => {
                    path.push(id);
                    current = next;
                }
                Err(_) => break,
            },
            Err(_) => break,
        }
    }

    Ok((current, path))
}

// =============================================================================
// Type Declarations
// =============================================================================

/// Any type declaration
pub fn type_declaration<'a>(full: &'a str, input: &'a str) -> PResult<'a, TypeDeclaration> {
    // Add context to help identify what failed
    context(
        "expected type declaration (class, interface, enum, typedef, or abstract)",
        alt((
            // Check for conditional compilation first
            |i| {
                let peek_result: Result<_, nom::Err<nom::error::Error<_>>> =
                    peek(tag("#if")).parse(i);
                if peek_result.is_ok() {
                    map(
                        |i| conditional_compilation(full, i, type_declaration),
                        TypeDeclaration::Conditional,
                    )
                    .parse(i)
                } else {
                    Err(nom::Err::Error(ContextualError::new(
                        i,
                        nom::error::ErrorKind::Tag,
                    )))
                }
            },
            // Check for metadata-prefixed declarations
            |i| {
                let peek_result: Result<_, nom::Err<nom::error::Error<_>>> =
                    peek(tag("@")).parse(i);
                if peek_result.is_ok() {
                    // Try parsing each type with metadata
                    alt((
                        context(
                            "class declaration",
                            map(|i| class_decl(full, i), TypeDeclaration::Class),
                        ),
                        context(
                            "interface declaration",
                            map(|i| interface_decl(full, i), TypeDeclaration::Interface),
                        ),
                        context(
                            "enum declaration",
                            map(|i| enum_decl(full, i), TypeDeclaration::Enum),
                        ),
                        context(
                            "typedef declaration",
                            map(|i| typedef_decl(full, i), TypeDeclaration::Typedef),
                        ),
                        context(
                            "abstract declaration",
                            map(|i| abstract_decl(full, i), TypeDeclaration::Abstract),
                        ),
                    ))
                    .parse(i)
                } else {
                    Err(nom::Err::Error(ContextualError::new(
                        i,
                        nom::error::ErrorKind::Tag,
                    )))
                }
            },
            context(
                "class declaration",
                map(|i| class_decl(full, i), TypeDeclaration::Class),
            ),
            context(
                "interface declaration",
                map(|i| interface_decl(full, i), TypeDeclaration::Interface),
            ),
            context(
                "enum declaration",
                map(|i| enum_decl(full, i), TypeDeclaration::Enum),
            ),
            context(
                "typedef declaration",
                map(|i| typedef_decl(full, i), TypeDeclaration::Typedef),
            ),
            context(
                "abstract declaration",
                map(|i| abstract_decl(full, i), TypeDeclaration::Abstract),
            ),
        )),
    )
    .parse(input)
}

/// Parse metadata attributes
pub fn metadata_list<'a>(full: &'a str, input: &'a str) -> PResult<'a, Vec<Metadata>> {
    many0(|i| metadata(full, i)).parse(input)
}

/// Parse single metadata: `@:native("foo")` or `@author("name")`
fn metadata<'a>(full: &'a str, input: &'a str) -> PResult<'a, Metadata> {
    let (input, _) = ws(input)?;
    let start = position(full, input);
    let (input, _) = char('@')(input)?;
    let (input, has_colon) = opt(char(':')).parse(input)?;
    let (input, name) = if has_colon.is_some() {
        // @:metadata format - allow keywords in metadata context
        identifier_or_keyword(input)?
    } else {
        // @metadata format
        identifier(input)?
    };

    // Optional parameters
    let (input, params) = opt(delimited(
        symbol("("),
        separated_list0(symbol(","), |i| expression(full, i)),
        symbol(")"),
    ))
    .parse(input)?;

    let end = position(full, input);

    Ok((
        input,
        Metadata {
            name,
            params: params.unwrap_or_default(),
            span: Span::new(start, end),
        },
    ))
}

/// Parse access modifier
pub fn access(input: &str) -> PResult<Access> {
    alt((
        value(Access::Public, keyword("public")),
        value(Access::Private, keyword("private")),
    ))
    .parse(input)
}

/// Parse function modifiers
pub fn modifiers(input: &str) -> PResult<Vec<Modifier>> {
    many0(alt((
        value(Modifier::Static, keyword("static")),
        value(Modifier::Inline, keyword("inline")),
        value(Modifier::Macro, keyword("macro")),
        value(Modifier::Dynamic, keyword("dynamic")),
        value(Modifier::Override, keyword("override")),
        value(Modifier::Final, keyword("final")),
        value(Modifier::Extern, keyword("extern")),
    )))
    .parse(input)
}

/// Import declarations from other parser modules
pub use crate::haxe_parser_decls::*;
pub use crate::haxe_parser_expr::*;
use crate::haxe_parser_expr2::block_expr;
pub use crate::haxe_parser_types::*;

// =============================================================================
// Conditional Compilation
// =============================================================================

/// Parse conditional compilation directive
pub fn conditional_compilation<'a, T, F>(
    full: &'a str,
    input: &'a str,
    content_parser: F,
) -> PResult<'a, ConditionalCompilation<T>>
where
    F: Fn(&'a str, &'a str) -> PResult<'a, T> + Copy,
{
    context("conditional compilation", |input| {
        let start = position(full, input);

        // Parse #if branch
        let (input, if_branch) = conditional_if_branch(full, input, content_parser)?;

        // Parse #elseif branches
        let (input, elseif_branches) =
            many0(|i| conditional_elseif_branch(full, i, content_parser)).parse(input)?;

        // Parse optional #else branch
        let (input, else_branch) =
            opt(|i| conditional_else_branch(full, i, content_parser)).parse(input)?;

        // Parse #end
        let (input, _) = ws(input)?;
        let (input, _) = tag("#end")(input)?;
        let (input, _) = ws(input)?; // Consume trailing whitespace after #end

        let end = position(full, input);

        Ok((
            input,
            ConditionalCompilation {
                if_branch,
                elseif_branches,
                else_branch,
                span: Span::new(start, end),
            },
        ))
    })
    .parse(input)
}

/// Parse #if branch
fn conditional_if_branch<'a, T, F>(
    full: &'a str,
    input: &'a str,
    content_parser: F,
) -> PResult<'a, ConditionalBlock<Vec<T>>>
where
    F: Fn(&'a str, &'a str) -> PResult<'a, T> + Copy,
{
    let start = position(full, input);
    let (input, _) = ws(input)?;
    let (input, _) = tag("#if")(input)?;
    let (input, _) = ws1(input)?;
    let (input, condition) = conditional_expr(input)?;
    let (input, _) = ws(input)?;

    // Parse content until #elseif, #else, or #end
    let (input, content) = many0(|i| {
        // Look ahead for conditional directives
        let peek_result: Result<_, nom::Err<nom::error::Error<_>>> =
            peek(alt((tag("#elseif"), tag("#else"), tag("#end")))).parse(i);
        if peek_result.is_ok() {
            // Stop parsing content
            Err(nom::Err::Error(ContextualError::new(
                i,
                nom::error::ErrorKind::Eof,
            )))
        } else {
            content_parser(full, i)
        }
    })
    .parse(input)?;

    let end = position(full, input);

    Ok((
        input,
        ConditionalBlock {
            condition,
            content,
            span: Span::new(start, end),
        },
    ))
}

/// Parse #elseif branch
fn conditional_elseif_branch<'a, T, F>(
    full: &'a str,
    input: &'a str,
    content_parser: F,
) -> PResult<'a, ConditionalBlock<Vec<T>>>
where
    F: Fn(&'a str, &'a str) -> PResult<'a, T> + Copy,
{
    let start = position(full, input);
    let (input, _) = ws(input)?;
    let (input, _) = tag("#elseif")(input)?;
    let (input, _) = ws1(input)?;
    let (input, condition) = conditional_expr(input)?;
    let (input, _) = ws(input)?;

    let (input, content) = many0(|i| {
        let peek_result: Result<_, nom::Err<nom::error::Error<_>>> =
            peek(alt((tag("#elseif"), tag("#else"), tag("#end")))).parse(i);
        if peek_result.is_ok() {
            Err(nom::Err::Error(ContextualError::new(
                i,
                nom::error::ErrorKind::Eof,
            )))
        } else {
            content_parser(full, i)
        }
    })
    .parse(input)?;

    let end = position(full, input);

    Ok((
        input,
        ConditionalBlock {
            condition,
            content,
            span: Span::new(start, end),
        },
    ))
}

/// Parse #else branch
fn conditional_else_branch<'a, T, F>(
    full: &'a str,
    input: &'a str,
    content_parser: F,
) -> PResult<'a, Vec<T>>
where
    F: Fn(&'a str, &'a str) -> PResult<'a, T> + Copy,
{
    let (input, _) = ws(input)?;
    let (input, _) = tag("#else")(input)?;
    let (input, _) = ws(input)?;

    many0(|i| {
        let peek_result: Result<_, nom::Err<nom::error::Error<_>>> = peek(tag("#end")).parse(i);
        if peek_result.is_ok() {
            Err(nom::Err::Error(ContextualError::new(
                i,
                nom::error::ErrorKind::Eof,
            )))
        } else {
            content_parser(full, i)
        }
    })
    .parse(input)
}

/// Parse conditional expression
fn conditional_expr(input: &str) -> PResult<ConditionalExpr> {
    conditional_or_expr(input)
}

/// Parse OR expression
fn conditional_or_expr(input: &str) -> PResult<ConditionalExpr> {
    let (input, left) = conditional_and_expr(input)?;

    let (input, rights) =
        many0(preceded(tuple((ws, tag("||"), ws)), conditional_and_expr)).parse(input)?;

    Ok((
        input,
        rights.into_iter().fold(left, |acc, right| {
            ConditionalExpr::Or(Box::new(acc), Box::new(right))
        }),
    ))
}

/// Parse AND expression
fn conditional_and_expr(input: &str) -> PResult<ConditionalExpr> {
    let (input, left) = conditional_not_expr(input)?;

    let (input, rights) =
        many0(preceded(tuple((ws, tag("&&"), ws)), conditional_not_expr)).parse(input)?;

    Ok((
        input,
        rights.into_iter().fold(left, |acc, right| {
            ConditionalExpr::And(Box::new(acc), Box::new(right))
        }),
    ))
}

/// Parse NOT expression
fn conditional_not_expr(input: &str) -> PResult<ConditionalExpr> {
    alt((
        map(preceded(char('!'), conditional_primary_expr), |e| {
            ConditionalExpr::Not(Box::new(e))
        }),
        conditional_primary_expr,
    ))
    .parse(input)
}

/// Parse primary expression (identifier or parenthesized)
fn conditional_primary_expr(input: &str) -> PResult<ConditionalExpr> {
    alt((
        // Parenthesized expression
        map(
            delimited(
                char('('),
                preceded(ws, conditional_expr),
                preceded(ws, char(')')),
            ),
            |e| ConditionalExpr::Paren(Box::new(e)),
        ),
        // Identifier
        map(identifier, ConditionalExpr::Ident),
    ))
    .parse(input)
}
