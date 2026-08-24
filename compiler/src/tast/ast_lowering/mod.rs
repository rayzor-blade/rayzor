//! AST to TAST Lowering
//!
//! This module converts the parser's AST representation into the compiler's
//! Typed Abstract Syntax Tree (TAST) representation, handling:
//! - Symbol resolution and creation
//! - Type annotation processing
//! - Scope management
//! - Error collection and reporting
//!
//! ## Error Recovery
//!
//! This module implements error recovery to collect all errors within a file
//! before stopping compilation. When lowering encounters errors, it:
//! - Collects errors in `collected_errors` and `context.errors` vectors
//! - Continues processing to find additional errors
//! - Returns all collected errors to the pipeline
//!
//! **Implementation Details**:
//! - Top-level declarations (imports, using, module fields, type declarations) use error collection
//! - Expression-level and function body errors still use early returns (future enhancement)
//! - The pipeline extracts all errors from `context.errors` when lowering fails
//!
//! **Future Enhancement**: Extend error recovery into function bodies and expressions
//! to collect all errors within individual functions. This requires placeholder values
//! for failed expressions to maintain type safety.

use crate::tast::node::HasSourceLocation;
use crate::tast::{core::*, node::MemoryEffects, node::*, type_resolution, *};
use parser::{
    AbstractDecl, BinaryOp, BlockElement, ClassDecl, ClassField, ClassFieldKind, EnumConstructor,
    EnumDecl, Expr, ExprKind, Function, FunctionParam, HaxeFile, Import, InterfaceDecl, Metadata,
    Modifier, ModuleField, Package, Type, TypeDeclaration, TypeParam, TypedefDecl, UnaryOp, Using,
};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fmt;
use std::rc::Rc;
use tracing::warn;

/// Record every `this.<field> = <param>` in an expression tree, marking a
/// parameter ambiguous once it is seen feeding a second field.
fn collect_this_field_stores<'a>(
    expr: &'a Expr,
    params: &std::collections::BTreeSet<&str>,
    out: &mut BTreeMap<&'a str, Option<&'a str>>,
) {
    if let ExprKind::Assign { left, right, .. } = &expr.kind {
        if let (
            ExprKind::Field {
                expr: recv, field, ..
            },
            ExprKind::Ident(name),
        ) = (&left.kind, &right.kind)
        {
            if matches!(recv.kind, ExprKind::This) && params.contains(name.as_str()) {
                out.entry(name.as_str())
                    .and_modify(|slot| {
                        if slot.is_some_and(|f| f != field.as_str()) {
                            *slot = None;
                        }
                    })
                    .or_insert(Some(field.as_str()));
            }
        }
    }
    // Recursion covers the block-shaped containers a body uses, and `return`,
    // because a brace-less setter puts the store inside one:
    // `function set_high(x) return this.high = x;`. Missing that left the
    // parameter Dynamic, and storing a Dynamic into an Int32 field writes
    // eight bytes into four.
    //
    // Anything else yields no inference, which leaves the parameter Dynamic --
    // the same answer as before, never a wrong one.
    match &expr.kind {
        ExprKind::Return(Some(inner)) => {
            collect_this_field_stores(inner, params, out);
        }
        ExprKind::Block(elements) => {
            for element in elements {
                if let BlockElement::Expr(e) = element {
                    collect_this_field_stores(e, params, out);
                }
            }
        }
        ExprKind::If {
            then_branch,
            else_branch,
            ..
        } => {
            collect_this_field_stores(then_branch, params, out);
            if let Some(e) = else_branch {
                collect_this_field_stores(e, params, out);
            }
        }
        ExprKind::While { body, .. } | ExprKind::For { body, .. } => {
            collect_this_field_stores(body, params, out);
        }
        _ => {}
    }
}

/// Top-level Haxe standard library classes that are always implicitly
/// available, matching the `.hx` files in the haxe-std root. Registered as
/// resolvable symbols wherever the full declarations have not been loaded;
/// every registration site skips names that already resolve, so a class
/// loaded from its real source is never shadowed by this backstop.
pub(crate) const TOPLEVEL_STDLIB_CLASSES: &[&str] = &[
    // Core types from StdTypes.hx
    "Dynamic",
    "Class",
    "Enum",
    "EnumValue",
    "Type",
    "Any",
    "Unknown",
    // Collections — Array must be here for array literals
    "Array",
    "Map",
    "List",
    "Vector",
    // String handling
    "String",
    "StringBuf",
    "StringTools",
    "UnicodeString",
    // Utilities
    "Date",
    "DateTools",
    "Math",
    "Reflect",
    "Std",
    "Sys",
    "Lambda",
    // Iteration
    "IntIterator",
    // Other stdlib
    "EReg",
    "Xml",
    "UInt",
    // System types living in packages but commonly used unqualified
    "File",
    "FileSystem",
    "Json",
    "Timer",
    "Bytes",
    "Int32",
    "Int64",
];

/// Convert parser Variance to TAST Variance
impl From<parser::Variance> for Variance {
    fn from(variance: parser::Variance) -> Self {
        match variance {
            parser::Variance::Invariant => Variance::Invariant,
            parser::Variance::Covariant => Variance::Covariant,
            parser::Variance::Contravariant => Variance::Contravariant,
        }
    }
}

/// Convert TAST node TypeVariance to core Variance
impl From<TypeVariance> for Variance {
    fn from(variance: TypeVariance) -> Self {
        match variance {
            TypeVariance::Invariant => Variance::Invariant,
            TypeVariance::Covariant => Variance::Covariant,
            TypeVariance::Contravariant => Variance::Contravariant,
        }
    }
}

/// Errors that can occur during AST lowering
#[derive(Debug, Clone)]
pub enum LoweringError {
    /// Symbol resolution failed
    UnresolvedSymbol {
        name: String,
        location: SourceLocation,
    },
    /// Type resolution failed
    UnresolvedType {
        type_name: String,
        location: SourceLocation,
    },
    /// Duplicate symbol definition
    DuplicateSymbol {
        name: String,
        original_location: SourceLocation,
        duplicate_location: SourceLocation,
    },
    /// Invalid modifier combination
    InvalidModifiers {
        modifiers: Vec<String>,
        location: SourceLocation,
    },
    /// Generic type parameter error
    GenericParameterError {
        message: String,
        location: SourceLocation,
    },
    /// Internal lowering error
    InternalError {
        message: String,
        location: SourceLocation,
    },
    /// Type inference failed
    TypeInferenceError {
        expression: String,
        location: SourceLocation,
    },
    /// Lifetime analysis failed
    LifetimeError {
        message: String,
        location: SourceLocation,
    },
    /// Ownership analysis failed
    OwnershipError {
        message: String,
        location: SourceLocation,
    },
    /// Incomplete lowering - missing implementation
    IncompleteImplementation {
        feature: String,
        location: SourceLocation,
    },
    /// A bare name matches multiple unrelated declarations and no context
    /// disambiguates — resolving to any one of them would be a guess.
    AmbiguousSymbol {
        message: String,
        location: SourceLocation,
    },
}

impl fmt::Display for LoweringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoweringError::UnresolvedSymbol { name, location } => {
                write!(
                    f,
                    "Unresolved symbol '{}' at {}:{}:{}",
                    name, location.file_id, location.line, location.column
                )
            }
            LoweringError::UnresolvedType {
                type_name,
                location,
            } => {
                write!(
                    f,
                    "Unresolved type '{}' at {}:{}:{}",
                    type_name, location.file_id, location.line, location.column
                )
            }
            LoweringError::DuplicateSymbol {
                name,
                original_location,
                duplicate_location,
            } => {
                write!(
                    f,
                    "Duplicate symbol '{}' (originally defined at {}:{}:{}) redefined at {}:{}:{}",
                    name,
                    original_location.file_id,
                    original_location.line,
                    original_location.column,
                    duplicate_location.file_id,
                    duplicate_location.line,
                    duplicate_location.column
                )
            }
            LoweringError::InvalidModifiers {
                modifiers,
                location,
            } => {
                write!(
                    f,
                    "Invalid modifier combination {:?} at {}:{}:{}",
                    modifiers, location.file_id, location.line, location.column
                )
            }
            LoweringError::GenericParameterError { message, location } => {
                write!(
                    f,
                    "Generic parameter error: {} at {}:{}:{}",
                    message, location.file_id, location.line, location.column
                )
            }
            LoweringError::InternalError { message, location } => {
                write!(
                    f,
                    "Internal error: {} at {}:{}:{}",
                    message, location.file_id, location.line, location.column
                )
            }
            LoweringError::TypeInferenceError {
                expression,
                location,
            } => {
                write!(
                    f,
                    "Type inference failed for '{}' at {}:{}:{}",
                    expression, location.file_id, location.line, location.column
                )
            }
            LoweringError::LifetimeError { message, location } => {
                write!(
                    f,
                    "Lifetime error: {} at {}:{}:{}",
                    message, location.file_id, location.line, location.column
                )
            }
            LoweringError::OwnershipError { message, location } => {
                write!(
                    f,
                    "Ownership error: {} at {}:{}:{}",
                    message, location.file_id, location.line, location.column
                )
            }
            LoweringError::IncompleteImplementation { feature, location } => {
                write!(
                    f,
                    "Incomplete implementation for '{}' at {}:{}:{}",
                    feature, location.file_id, location.line, location.column
                )
            }
            LoweringError::AmbiguousSymbol { message, location } => {
                write!(
                    f,
                    "{} at {}:{}:{}",
                    message, location.file_id, location.line, location.column
                )
            }
        }
    }
}

impl LoweringError {
    /// Convert LoweringError to CompilationError for formatted diagnostic output
    pub fn to_compilation_error(&self) -> crate::pipeline::CompilationError {
        use crate::pipeline::{CompilationError, ErrorCategory};

        match self {
            LoweringError::UnresolvedSymbol { name, location } => CompilationError {
                message: format!("Cannot find name '{}'", name),
                location: location.clone(),
                category: ErrorCategory::SymbolError,
                suggestion: Some(format!("Check if '{}' is imported or defined", name)),
                related_errors: vec![],
            },
            LoweringError::UnresolvedType {
                type_name,
                location,
            } => CompilationError {
                message: format!("Cannot find type '{}'", type_name),
                location: location.clone(),
                category: ErrorCategory::TypeError,
                suggestion: Some(format!(
                    "Check if type '{}' is imported or defined",
                    type_name
                )),
                related_errors: vec![],
            },
            LoweringError::DuplicateSymbol {
                name,
                original_location,
                duplicate_location,
            } => CompilationError {
                message: format!("Duplicate definition of '{}'", name),
                location: duplicate_location.clone(),
                category: ErrorCategory::SymbolError,
                suggestion: Some("Remove or rename one of the conflicting definitions".to_string()),
                related_errors: vec![format!(
                    "First defined at {}:{}",
                    original_location.line, original_location.column
                )],
            },
            LoweringError::InvalidModifiers {
                modifiers,
                location,
            } => CompilationError {
                message: format!("Invalid modifier combination: {}", modifiers.join(", ")),
                location: location.clone(),
                category: ErrorCategory::TypeError,
                suggestion: Some(
                    "Check Haxe documentation for valid modifier combinations".to_string(),
                ),
                related_errors: vec![],
            },
            LoweringError::GenericParameterError { message, location } => CompilationError {
                message: message.clone(),
                location: location.clone(),
                category: ErrorCategory::TypeError,
                suggestion: None,
                related_errors: vec![],
            },
            LoweringError::InternalError { message, location } => CompilationError {
                message: format!("Internal compiler error: {}", message),
                location: location.clone(),
                category: ErrorCategory::TypeError,
                suggestion: Some("This is a compiler bug - please report it".to_string()),
                related_errors: vec![],
            },
            LoweringError::TypeInferenceError {
                expression,
                location,
            } => CompilationError {
                message: format!("Cannot infer type for expression: {}", expression),
                location: location.clone(),
                category: ErrorCategory::TypeError,
                suggestion: Some("Add an explicit type annotation".to_string()),
                related_errors: vec![],
            },
            LoweringError::LifetimeError { message, location } => {
                // Provide context-sensitive suggestions for lifetime errors
                let suggestion = if message.contains("dangling") || message.contains("outlive") {
                    Some("Consider extending the lifetime of the referenced data or copying the value".to_string())
                } else if message.contains("borrow") {
                    Some("Ensure borrows do not outlive the data they reference".to_string())
                } else {
                    Some(
                        "Review lifetime annotations and ensure data lifetimes are compatible"
                            .to_string(),
                    )
                };

                CompilationError {
                    message: message.clone(),
                    location: location.clone(),
                    category: ErrorCategory::LifetimeError,
                    suggestion,
                    related_errors: vec![],
                }
            }
            LoweringError::OwnershipError { message, location } => {
                // Provide context-sensitive suggestions for ownership errors
                let suggestion = if message.contains("moved") || message.contains("use after move")
                {
                    Some("Value was moved - consider cloning the value or restructuring to avoid the move".to_string())
                } else if message.contains("borrow") && message.contains("mutable") {
                    Some("Cannot have mutable and immutable borrows simultaneously - resolve conflicting borrows".to_string())
                } else if message.contains("borrow") {
                    Some("Borrow checker violation - ensure borrows follow Rust-style ownership rules".to_string())
                } else {
                    Some("Review ownership rules: each value has one owner, moves transfer ownership".to_string())
                };

                CompilationError {
                    message: message.clone(),
                    location: location.clone(),
                    category: ErrorCategory::OwnershipError,
                    suggestion,
                    related_errors: vec![],
                }
            }
            LoweringError::IncompleteImplementation { feature, location } => CompilationError {
                message: format!("Feature not yet implemented: {}", feature),
                location: location.clone(),
                category: ErrorCategory::TypeError,
                suggestion: Some("This feature is planned for a future release".to_string()),
                related_errors: vec![],
            },
            LoweringError::AmbiguousSymbol { message, location } => CompilationError {
                message: message.clone(),
                location: location.clone(),
                category: ErrorCategory::SymbolError,
                suggestion: Some(
                    "Qualify the name (Enum.Variant / annotate the type) so a single declaration matches".to_string(),
                ),
                related_errors: vec![],
            },
        }
    }
}

/// Result type for lowering operations
pub type LoweringResult<T> = Result<T, LoweringError>;

/// Information extracted from modifiers
#[derive(Debug, Clone, Default)]
pub struct ModifierInfo {
    pub visibility: Visibility,
    pub is_static: bool,
    pub is_override: bool,
    pub is_inline: bool,
    pub is_dynamic: bool,
    pub is_macro: bool,
    pub is_final: bool,
    pub is_extern: bool,
    pub is_abstract: bool,
    pub other_modifiers: Vec<String>,
}

/// Deferred type resolution for forward references
#[derive(Debug, Clone)]
pub struct DeferredTypeResolution {
    pub type_name: String,
    pub location: SourceLocation,
    pub type_params: Vec<String>,
    pub target_type_id: TypeId, // The placeholder TypeId that will be replaced
}

/// Simple two-pass type resolution state
#[derive(Debug, Default)]
pub struct TypeResolutionState {
    pub deferred_resolutions: Vec<DeferredTypeResolution>,
    pub placeholder_to_real: BTreeMap<TypeId, TypeId>,
}

impl ModifierInfo {
    pub fn new() -> Self {
        Self {
            visibility: Visibility::Internal, // Default visibility
            is_static: false,
            is_override: false,
            is_inline: false,
            is_dynamic: false,
            is_macro: false,
            is_final: false,
            is_extern: false,
            is_abstract: false,
            other_modifiers: Vec::new(),
        }
    }
}

/// Typed declaration wrapper for lowering
#[derive(Debug, Clone)]
pub enum TypedDeclaration {
    Function(TypedFunction),
    Class(TypedClass),
    Interface(TypedInterface),
    Enum(TypedEnum),
    TypeAlias(TypedTypeAlias),
    Abstract(TypedAbstract),
}

/// Typed typedef declaration for lowering
#[derive(Debug, Clone)]
pub struct TypedTypedef {
    pub symbol_id: SymbolId,
    pub name: String,
    pub target_type: TypeId,
    pub type_parameters: Vec<TypedTypeParameter>,
    pub visibility: Visibility,
    pub source_location: SourceLocation,
}

/// Context for AST lowering operations
pub struct LoweringContext<'a> {
    pub string_interner: &'a mut StringInterner,
    /// Shared reference to the string interner (for TypedFile creation)
    pub string_interner_rc: Rc<RefCell<StringInterner>>,
    pub symbol_table: &'a mut SymbolTable,
    pub type_table: &'a RefCell<TypeTable>,
    pub scope_tree: &'a mut ScopeTree,
    pub current_scope: ScopeId,
    pub errors: Vec<LoweringError>,
    pub type_parameter_stack: Vec<BTreeMap<InternedString, TypeId>>,
    pub span_converter: Option<super::span_conversion::SpanConverter>,
    /// Stack of class symbols we're currently inside (for method resolution)
    pub class_context_stack: Vec<SymbolId>,
    /// Namespace resolver for type path resolution
    pub namespace_resolver: &'a mut super::namespace::NamespaceResolver,
    /// Import resolver for import management
    pub import_resolver: &'a mut super::namespace::ImportResolver,
    /// Current package context
    pub current_package: Option<super::namespace::PackageId>,
    /// Current switch discriminant type (for resolving enum constructors in pattern matching)
    pub switch_discriminant_type: Option<TypeId>,
    /// When set, an `ExprKind::New` with no explicit `<...>` params can borrow
    /// the type arguments from this hint to make `var m:Map<String,Int> = new Map()`
    /// behave the same as `var m = new Map<String,Int>()`. Used only to seed
    /// `@:multiType` abstract resolution (e.g. `Map` → `StringMap`); a wrong
    /// hint is harmless because the New site re-runs through type checking
    /// against the declared variable type either way.
    pub expected_new_type_hint: Option<TypeId>,
    /// The enclosing function/method's declared return type, set before its
    /// body is lowered (mirrors `switch_discriminant_type`). Used to
    /// disambiguate a bare enum-variant identifier (`return F32;`) when
    /// MULTIPLE imported enums declare a variant with the same name —
    /// `resolve_symbol_in_scope_hierarchy` is a plain first-match scope
    /// walk with no type awareness, so e.g. `rayzor.ds.DType.F32` (index 0,
    /// unboxed) and `nue.loader.GGUFReader.MetaValue.F32` (index 8, boxed
    /// with a Float payload) collide and whichever got registered first
    /// wins — silently returning the wrong enum's boxed/garbage value.
    pub expected_return_type: Option<TypeId>,
}

impl<'a> LoweringContext<'a> {
    pub fn new(
        string_interner: &'a mut StringInterner,
        string_interner_rc: Rc<RefCell<StringInterner>>,
        symbol_table: &'a mut SymbolTable,
        type_table: &'a RefCell<TypeTable>,
        scope_tree: &'a mut ScopeTree,
        current_scope: ScopeId,
        namespace_resolver: &'a mut super::namespace::NamespaceResolver,
        import_resolver: &'a mut super::namespace::ImportResolver,
    ) -> Self {
        Self {
            string_interner,
            string_interner_rc,
            symbol_table,
            type_table,
            scope_tree,
            current_scope,
            errors: Vec::with_capacity(16), // Most files have <16 errors
            type_parameter_stack: Vec::with_capacity(4), // Most type nesting is <4 deep
            span_converter: None,
            class_context_stack: Vec::with_capacity(4), // Most class nesting is <4 deep
            namespace_resolver,
            import_resolver,
            current_package: None,
            switch_discriminant_type: None,
            expected_new_type_hint: None,
            expected_return_type: None,
        }
    }

    /// Add an error to the context
    pub fn add_error(&mut self, error: LoweringError) {
        self.errors.push(error);
    }

    /// Clear the current package context (used for stdlib loading)
    pub fn clear_package_context(&mut self) {
        self.current_package = None;
    }

    /// Enter a new scope
    pub fn enter_scope(&mut self, _scope_kind: ScopeKind) -> ScopeId {
        let new_scope = self.scope_tree.create_scope(Some(self.current_scope));
        self.current_scope = new_scope;
        new_scope
    }

    /// Enter a new named scope (for classes, interfaces, etc.)
    pub fn enter_named_scope(&mut self, scope_kind: ScopeKind, name: InternedString) -> ScopeId {
        let new_scope = self.scope_tree.create_scope(Some(self.current_scope));
        self.current_scope = new_scope;
        // Set the name and kind on the scope
        if let Some(scope) = self.scope_tree.get_scope_mut(new_scope) {
            scope.name = Some(name);
            scope.kind = scope_kind;
        }
        new_scope
    }

    /// Exit the current scope
    pub fn exit_scope(&mut self) {
        if let Some(scope) = self.scope_tree.get_scope(self.current_scope) {
            if let Some(parent) = scope.parent_id {
                self.current_scope = parent;
            }
        }
    }

    /// Push type parameters onto the stack
    pub fn push_type_parameters(&mut self, type_params: BTreeMap<InternedString, TypeId>) {
        self.type_parameter_stack.push(type_params);
    }

    /// Pop type parameters from the stack
    pub fn pop_type_parameters(&mut self) {
        self.type_parameter_stack.pop();
    }

    /// Resolve a type parameter
    pub fn resolve_type_parameter(&self, name: InternedString) -> Option<TypeId> {
        for scope in self.type_parameter_stack.iter().rev() {
            if let Some(&type_id) = scope.get(&name) {
                return Some(type_id);
            }
        }
        None
    }

    /// Intern a string
    pub fn intern_string(&mut self, s: &str) -> InternedString {
        self.string_interner.intern(s)
    }

    /// Create a source location from parser span
    pub fn create_location_from_span(&self, span: parser::Span) -> SourceLocation {
        if let Some(converter) = &self.span_converter {
            converter.convert_span(span)
        } else {
            // Fallback to basic offset-only location when no converter available
            SourceLocation::new(0, 0, 0, span.start as u32)
        }
    }

    /// Create a source location (fallback when span not available)
    pub fn create_location(&self) -> SourceLocation {
        SourceLocation::unknown()
    }

    /// Generate next scope ID for new scopes
    pub fn next_scope_id(&mut self) -> u32 {
        // Create a new scope and return its raw ID
        let scope = self.scope_tree.create_scope(Some(self.current_scope));
        scope.as_raw()
    }

    /// Convert a span to a source location (alias for create_location_from_span)
    pub fn span_to_location(&self, span: &parser::Span) -> SourceLocation {
        self.create_location_from_span(*span)
    }

    /// Update the qualified name for a symbol based on its scope chain
    pub fn update_symbol_qualified_name(&mut self, symbol_id: SymbolId) {
        self.symbol_table
            .update_qualified_name(symbol_id, self.scope_tree, self.string_interner);
    }

    /// Initialize the span converter with source text
    pub fn initialize_span_converter(&mut self, file_id: u32, source_text: String) {
        self.initialize_span_converter_with_filename(
            file_id,
            source_text,
            format!("file_{}.hx", file_id),
        );
    }

    /// Initialize the span converter with source text and specific filename.
    /// `file_id` is the COMPILATION-PIPELINE-LEVEL identifier (set by the
    /// outer source loader); it is stamped on every SourceLocation the
    /// converter produces so cross-file lowerings can be distinguished
    /// downstream (ownership diagnostics, the RAYZOR_DEBUG_E0382 dump,
    /// renderer attribution). The previous body silently dropped the
    /// file_id, so every TypedExpression span ended up tagged file_id=0
    /// regardless of the actual source file — see
    /// bugs_diagnostic_span_file_id_always_zero.
    pub fn initialize_span_converter_with_filename(
        &mut self,
        file_id: u32,
        source_text: String,
        file_name: String,
    ) {
        self.span_converter = Some(super::span_conversion::SpanConverter::with_file_and_id(
            file_name,
            source_text,
            file_id,
        ));
    }
}

/// Main AST lowering implementation
pub struct AstLowering<'a> {
    context: LoweringContext<'a>,
    resolution_state: TypeResolutionState,
    /// Temporary storage for classes being built (symbol_id -> class methods)
    class_methods: BTreeMap<SymbolId, Vec<(InternedString, SymbolId, bool)>>, // (name, symbol, is_static)
    class_fields: BTreeMap<SymbolId, Vec<(InternedString, SymbolId, bool)>>, // (name, symbol, is_static)
    /// Child class symbol -> parent class symbol, for resolving inherited
    /// members. `class_methods`/`class_fields` only ever hold what THIS
    /// compilation context lowered, so a parent from another module contributes
    /// nothing to them; its members are reachable only through its own scope in
    /// the shared symbol table, which is what this map makes walkable.
    class_parents: BTreeMap<SymbolId, SymbolId>,
    /// Skip internal stdlib loading (used when CompilationUnit handles it)
    skip_stdlib_loading: bool,
    /// Program-wide declared static signatures (shared from CompilationUnit).
    /// Lets `ensure_known_static_method_type` type a static whose declaring
    /// file hasn't lowered yet from its parsed declaration.
    static_sig_index: Option<std::rc::Rc<RefCell<crate::tast::sig_index::StaticSigIndex>>>,
    /// Skip pre-registration pass (used when CompilationUnit has already pre-registered all files)
    skip_pre_registration: bool,
    /// Collected errors during lowering (for error recovery)
    pub collected_errors: Vec<LoweringError>,
    /// Symbols declared as an untyped empty array literal (`var x = []`), whose
    /// element type is still the placeholder `Dynamic` and should be bound (the
    /// monomorph rewrite) from the FIRST `x.push(e)` / `x[i] = e` — at which
    /// point `e`'s type is known (earlier statements are already lowered). Maps
    /// the symbol to its declaration location (for the "uncertain element type"
    /// warning). Once bound the symbol is removed. Keeps numeric arrays unboxed
    /// (`Array<Float>` routes through the f64 push/get path, not the truncating
    /// generic one).
    empty_array_inferred: std::collections::BTreeMap<SymbolId, SourceLocation>,
    /// Subset of `empty_array_inferred` that was USED (pushed/index-assigned)
    /// but whose element type could not be determined at compile time — if a
    /// symbol is still here AND still unbound at end of file, it stayed
    /// `Array<Dynamic>` and earns a `Correctness` warning (a later peekable push
    /// clears it, per "until another push says otherwise").
    empty_array_used_uncertain: std::collections::BTreeSet<SymbolId>,
    /// Active 'using' modules for static extension resolution
    /// Maps module name (e.g., "StringTools") to class symbol ID
    using_modules: Vec<(InternedString, SymbolId)>,
    /// Pending 'using' modules that need to be loaded (not yet compiled)
    /// These are module paths like "StringTools" that were used but only pre-registered
    pub pending_usings: Vec<String>,
    /// Whether we're currently lowering a static method body (no `this` available)
    in_static_method: bool,
    /// Ordered type parameter TypeIds for each generic class (class_symbol → [TypeParam TypeIds])
    class_type_params: BTreeMap<SymbolId, Vec<TypeId>>,
    /// Constructor symbol for each class (class_symbol → constructor SymbolId)
    class_constructor_symbols: BTreeMap<SymbolId, SymbolId>,
    /// Stack of expected lambda parameter types per active call-arg position.
    /// Pushed before lowering an argument expression to a function whose formal
    /// parameter at that position is a function type with concrete parameter
    /// types. Pulled by `ExprKind::Function`/`ExprKind::Arrow` to fill in
    /// untyped parameters (`function(i, n)` getting `(idx:Int, node:Int)`
    /// from the formal `fn:(idx:Int, node:Int)->Void`). Without this, lambda
    /// params default to `Dynamic` → MIR signature `(*void,*void,*void)` →
    /// caller passes i32 reinterpreted as pointers and the lambda
    /// dereferences address 0/1/2 producing garbage / null.
    expected_lambda_params_stack: Vec<Option<Vec<TypeId>>>,
    /// Per-arg expected type for the current call/init context. Lets
    /// `lower_expression(Ident("F32"))` disambiguate between enum variants
    /// with the same simple name (e.g. `DType.F32` vs `MetaValue.F32`) by
    /// preferring the variant whose parent enum matches the expected type.
    /// `None` at the top means no usable hint (untyped lambda call, dynamic
    /// dispatch, …) — the existing scope-walk resolution applies.
    expected_arg_type_stack: Vec<Option<TypeId>>,
    /// Re-entrancy guard for the best-effort callee-hint resolution.
    /// `resolve_callee_*` lowers a call's RECEIVER (`lower_expression(obj)`)
    /// to find its method; when that receiver is itself a call this re-enters
    /// `lower_call_expression`, which would resolve hints again — and because
    /// each call resolves hints twice (param + formal), a chain of
    /// call-valued receivers blows up to O(2^depth) redundant lowerings
    /// (observed as a multi-second / unbounded compile hang on
    /// `WorkerPool.parallelRows`). The hint is purely best-effort, so while we
    /// are resolving one we suppress nested hint resolution: the receiver is
    /// still lowered (once) for its type, just without recursively re-pricing
    /// the hint at every level. The real per-call hint is still computed when
    /// that nested call is lowered for real, outside this guard.
    suppress_callee_hint: bool,
}

/// Result of type parameter substitution for generic method return types
#[derive(Debug)]
pub(crate) enum TypeSubstitutionResult {
    /// No substitution needed, return this type as-is
    NoChange(TypeId),
    /// Direct substitution to this type (type parameter was replaced)
    DirectSubstitution(TypeId),
    /// Need to create a new GenericInstance with these type arguments
    NeedGenericInstance {
        base_type: TypeId,
        type_args: Vec<TypeId>,
    },
    /// Need to create a concrete generic Class with these type arguments
    /// (e.g. `MutexGuard<State>` substituted from `MutexGuard<T>`).
    /// Used when the return type is a `TypeKind::Class` with non-empty
    /// `type_args`, distinguishing from `GenericInstance`.
    NeedClassInstance {
        symbol_id: SymbolId,
        type_args: Vec<TypeId>,
    },
    /// Need to create an `Optional<T>` (`Null<T>`) with this substituted inner
    /// type — e.g. `Map<K,V>.get` returns `Null<V>`; without this the inner
    /// stays an abstract `V`, and the caller boxes the value with an
    /// unresolved type tag (corrupting enum/reference values).
    NeedOptional { inner_type: TypeId },
}

mod context;
mod decls;
mod expr;
mod imports;
mod infer;
mod metadata;
mod resolve;
mod stdlib;
mod stmt;
mod traits;
mod types;
mod validate;

/// Convenience function to lower a Haxe file
pub fn lower_haxe_file(
    file: &HaxeFile,
    string_interner: &mut StringInterner,
    string_interner_rc: Rc<RefCell<StringInterner>>,
    symbol_table: &mut SymbolTable,
    type_table: &RefCell<TypeTable>,
    scope_tree: &mut ScopeTree,
    namespace_resolver: &mut super::namespace::NamespaceResolver,
    import_resolver: &mut super::namespace::ImportResolver,
) -> LoweringResult<TypedFile> {
    let mut lowering = AstLowering::new(
        string_interner,
        string_interner_rc,
        symbol_table,
        type_table,
        scope_tree,
        namespace_resolver,
        import_resolver,
    );
    lowering.lower_file(file)
}
