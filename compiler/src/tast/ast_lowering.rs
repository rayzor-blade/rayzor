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
    AbstractDecl, BinaryOp, ClassDecl, ClassField, ClassFieldKind, EnumConstructor, EnumDecl, Expr,
    ExprKind, Function, FunctionParam, HaxeFile, Import, InterfaceDecl, Metadata, Modifier,
    ModuleField, Package, Type, TypeDeclaration, TypeParam, TypedefDecl, UnaryOp, Using,
};
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;
use tracing::warn;

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
    pub placeholder_to_real: HashMap<TypeId, TypeId>,
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
    pub type_parameter_stack: Vec<HashMap<InternedString, TypeId>>,
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
    pub fn push_type_parameters(&mut self, type_params: HashMap<InternedString, TypeId>) {
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

    /// Initialize the span converter with source text and specific filename
    pub fn initialize_span_converter_with_filename(
        &mut self,
        file_id: u32,
        source_text: String,
        file_name: String,
    ) {
        self.span_converter = Some(super::span_conversion::SpanConverter::with_file(
            file_name,
            source_text,
        ));
    }
}

/// Main AST lowering implementation
pub struct AstLowering<'a> {
    context: LoweringContext<'a>,
    resolution_state: TypeResolutionState,
    /// Temporary storage for classes being built (symbol_id -> class methods)
    class_methods: HashMap<SymbolId, Vec<(InternedString, SymbolId, bool)>>, // (name, symbol, is_static)
    class_fields: HashMap<SymbolId, Vec<(InternedString, SymbolId, bool)>>, // (name, symbol, is_static)
    /// Skip internal stdlib loading (used when CompilationUnit handles it)
    skip_stdlib_loading: bool,
    /// Skip pre-registration pass (used when CompilationUnit has already pre-registered all files)
    skip_pre_registration: bool,
    /// Collected errors during lowering (for error recovery)
    pub collected_errors: Vec<LoweringError>,
    /// Active 'using' modules for static extension resolution
    /// Maps module name (e.g., "StringTools") to class symbol ID
    using_modules: Vec<(InternedString, SymbolId)>,
    /// Pending 'using' modules that need to be loaded (not yet compiled)
    /// These are module paths like "StringTools" that were used but only pre-registered
    pub pending_usings: Vec<String>,
    /// Whether we're currently lowering a static method body (no `this` available)
    in_static_method: bool,
    /// Ordered type parameter TypeIds for each generic class (class_symbol → [TypeParam TypeIds])
    class_type_params: HashMap<SymbolId, Vec<TypeId>>,
    /// Constructor symbol for each class (class_symbol → constructor SymbolId)
    class_constructor_symbols: HashMap<SymbolId, SymbolId>,
}

/// Result of type parameter substitution for generic method return types
enum TypeSubstitutionResult {
    /// No substitution needed, return this type as-is
    NoChange(TypeId),
    /// Direct substitution to this type (type parameter was replaced)
    DirectSubstitution(TypeId),
    /// Need to create a new GenericInstance with these type arguments
    NeedGenericInstance {
        base_type: TypeId,
        type_args: Vec<TypeId>,
    },
}

impl<'a> AstLowering<'a> {
    /// Extract SymbolFlags and optional native name from metadata entries.
    /// Shared across class, abstract, and other type declarations.
    fn extract_metadata_flags(
        &mut self,
        meta_list: &[parser::haxe_ast::Metadata],
        symbol_id: SymbolId,
    ) -> crate::tast::symbols::SymbolFlags {
        use crate::tast::symbols::SymbolFlags;

        let mut flags = SymbolFlags::NONE;
        for meta in meta_list {
            let name = meta.name.strip_prefix(':').unwrap_or(&meta.name);
            match name {
                "generic" => flags = flags.union(SymbolFlags::GENERIC),
                "final" => flags = flags.union(SymbolFlags::FINAL),
                "forward" => flags = flags.union(SymbolFlags::FORWARD),
                "extern" => flags = flags.union(SymbolFlags::EXTERN),
                "native" => {
                    flags = flags.union(SymbolFlags::NATIVE);
                    if let Some(first_param) = meta.params.first() {
                        if let parser::haxe_ast::ExprKind::String(native_str) = &first_param.kind {
                            let native_interned = self.context.string_interner.intern(&native_str);
                            if let Some(sym) = self.context.symbol_table.get_symbol_mut(symbol_id) {
                                sym.native_name = Some(native_interned);
                            }
                        }
                    }
                }
                "cstruct" => {
                    flags = flags.union(SymbolFlags::CSTRUCT);
                    let no_mangle = meta.params.iter().any(|p| {
                        matches!(&p.kind, parser::haxe_ast::ExprKind::Ident(s) if s == "NoMangle")
                    });
                    if no_mangle {
                        flags = flags.union(SymbolFlags::NO_MANGLE);
                    }
                }
                "gpuStruct" => {
                    flags = flags.union(SymbolFlags::GPU_STRUCT);
                }
                "no_mangle" => flags = flags.union(SymbolFlags::NO_MANGLE),
                "frameworks" | "cInclude" | "cSource" | "clib" => {
                    // @:frameworks(["Accelerate"]), @:cInclude(["vendor/stb"]), @:cSource(["lib.c"])
                    if let Some(first_param) = meta.params.first() {
                        if let parser::haxe_ast::ExprKind::Array(elements) = &first_param.kind {
                            let mut names = Vec::new();
                            for elem in elements {
                                if let parser::haxe_ast::ExprKind::String(s) = &elem.kind {
                                    names.push(self.context.string_interner.intern(s));
                                }
                            }
                            if !names.is_empty() {
                                if let Some(sym) =
                                    self.context.symbol_table.get_symbol_mut(symbol_id)
                                {
                                    match name {
                                        "frameworks" => sym.frameworks = Some(names),
                                        "cInclude" => sym.c_includes = Some(names),
                                        "cSource" => sym.c_sources = Some(names),
                                        "clib" => sym.c_libs = Some(names),
                                        _ => {}
                                    }
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        flags
    }

    /// Resolve a TypeId through TypeAlias chains to find the underlying type.
    fn resolve_alias_chain(type_table: &TypeTable, type_id: TypeId) -> TypeId {
        let mut current = type_id;
        for _ in 0..10 {
            match type_table.get(current).map(|t| &t.kind) {
                Some(TypeKind::TypeAlias { target_type, .. }) => current = *target_type,
                _ => break,
            }
        }
        current
    }

    /// Get the current class symbol if we're in a class context
    fn get_current_class_symbol(&self) -> Option<SymbolId> {
        self.context.class_context_stack.last().copied()
    }

    /// Infer the type of an enum constructor call with generic type instantiation
    fn infer_enum_constructor_type(
        &mut self,
        constructor_symbol: SymbolId,
        arguments: &[TypedExpression],
    ) -> LoweringResult<TypeId> {
        // println!(
        //     "DEBUG: Inferring enum constructor type for symbol {:?} with {} arguments",
        //     constructor_symbol,
        //     arguments.len()
        // );

        // Find the parent enum for this constructor
        let parent_enum = self
            .find_parent_enum_for_constructor(constructor_symbol)
            .ok_or_else(|| LoweringError::InternalError {
                message: "Could not find parent enum for constructor".to_string(),
                location: self.context.create_location(),
            })?;

        // println!(
        //     "DEBUG: Found parent enum {:?} for constructor {:?}",
        //     parent_enum, constructor_symbol
        // );

        // Get the parent enum's type information
        if let Some(enum_symbol) = self.context.symbol_table.get_symbol(parent_enum) {
            let enum_type_info = self
                .context
                .type_table
                .borrow()
                .get(enum_symbol.type_id)
                .ok_or_else(|| LoweringError::InternalError {
                    message: "Could not get type info for enum".to_string(),
                    location: self.context.create_location(),
                })?
                .clone();

            match &enum_type_info.kind {
                crate::tast::core::TypeKind::Enum { type_args, .. } => {
                    if type_args.is_empty() {
                        // Non-generic enum, just return the enum type
                        // println!(
                        //     "DEBUG: Non-generic enum, returning enum type {:?}",
                        //     enum_symbol.type_id
                        // );
                        return Ok(enum_symbol.type_id);
                    }

                    // Generic enum - need to infer type arguments from constructor arguments
                    // println!(
                    //     "DEBUG: Generic enum with {} type parameters",
                    //     type_args.len()
                    // );

                    // Infer type parameters from constructor arguments
                    let mut inferred_types = Vec::new();

                    // Match argument types to constructor parameter types
                    for (i, arg) in arguments.iter().enumerate() {
                        if i < type_args.len() {
                            inferred_types.push(arg.expr_type);
                        }
                    }

                    // Fill remaining type parameters with dynamic type
                    while inferred_types.len() < type_args.len() {
                        inferred_types.push(self.context.type_table.borrow().dynamic_type());
                    }

                    // Create properly instantiated enum type
                    if !inferred_types.is_empty() {
                        let instantiated_type = self
                            .context
                            .type_table
                            .borrow_mut()
                            .create_enum_type(parent_enum, inferred_types);
                        return Ok(instantiated_type);
                    }

                    // Non-generic enum
                    Ok(self
                        .context
                        .type_table
                        .borrow_mut()
                        .create_enum_type(parent_enum, vec![]))
                }
                _ => {
                    // Not an enum type
                    Ok(self.context.type_table.borrow().dynamic_type())
                }
            }
        } else {
            Ok(self.context.type_table.borrow().dynamic_type())
        }
    }

    /// Instantiate the function type of an enum constructor based on call arguments
    fn instantiate_enum_constructor_type(
        &mut self,
        constructor_symbol: SymbolId,
        arguments: &[TypedExpression],
        mut func_expr: TypedExpression,
    ) -> LoweringResult<TypedExpression> {
        // println!(
        //     "DEBUG: Instantiating constructor function type for symbol {:?} with {} arguments",
        //     constructor_symbol,
        //     arguments.len()
        // );

        // Find the parent enum for this constructor
        let parent_enum = self
            .find_parent_enum_for_constructor(constructor_symbol)
            .ok_or_else(|| LoweringError::InternalError {
                message: "Could not find parent enum for constructor".to_string(),
                location: self.context.create_location(),
            })?;

        // println!(
        //     "DEBUG: Found parent enum {:?} for constructor {:?}",
        //     parent_enum, constructor_symbol
        // );

        // Get the parent enum's type information
        if let Some(enum_symbol) = self.context.symbol_table.get_symbol(parent_enum) {
            let enum_type_info = self
                .context
                .type_table
                .borrow()
                .get(enum_symbol.type_id)
                .ok_or_else(|| LoweringError::InternalError {
                    message: "Could not get type info for enum".to_string(),
                    location: self.context.create_location(),
                })?
                .clone();

            match &enum_type_info.kind {
                crate::tast::core::TypeKind::Enum { type_args, .. } => {
                    if type_args.is_empty() {
                        // Non-generic enum - ensure function type has enum as return type
                        // This is critical for type inference of parameterized enum constructors
                        let constructor_sym =
                            self.context.symbol_table.get_symbol(constructor_symbol);
                        if let Some(sym) = constructor_sym {
                            // Extract params first, then drop the borrow
                            let params_opt = {
                                let type_table = self.context.type_table.borrow();
                                if let Some(func_type_info) = type_table.get(sym.type_id) {
                                    if let crate::tast::core::TypeKind::Function {
                                        params, ..
                                    } = &func_type_info.kind
                                    {
                                        Some(params.clone())
                                    } else {
                                        None
                                    }
                                } else {
                                    None
                                }
                            };
                            // Now create the function type with the enum as return type
                            if let Some(params) = params_opt {
                                let corrected_func_type = self
                                    .context
                                    .type_table
                                    .borrow_mut()
                                    .create_function_type(params, enum_symbol.type_id);
                                func_expr.expr_type = corrected_func_type;
                            }
                        }
                        return Ok(func_expr);
                    }

                    // Generic enum - need to infer type arguments from constructor arguments
                    // println!(
                    //     "DEBUG: Generic enum with {} type parameters",
                    //     type_args.len()
                    // );

                    // Infer type arguments from constructor arguments
                    if !arguments.is_empty() && !type_args.is_empty() {
                        // Get the original constructor's function type params
                        let original_params = {
                            if let Some(sym) =
                                self.context.symbol_table.get_symbol(constructor_symbol)
                            {
                                let type_table = self.context.type_table.borrow();
                                if let Some(ty) = type_table.get(sym.type_id) {
                                    if let crate::tast::core::TypeKind::Function {
                                        params, ..
                                    } = &ty.kind
                                    {
                                        params.clone()
                                    } else {
                                        vec![]
                                    }
                                } else {
                                    vec![]
                                }
                            } else {
                                vec![]
                            }
                        };

                        // Infer the concrete type parameter T from the first argument.
                        // Two cases:
                        //   1. Param type is T directly (e.g., Leaf(value:T)) → arg type IS T
                        //   2. Param type is Enum<T> (e.g., Node(left:Tree<T>)) → extract T from arg's type args
                        let inferred_type = {
                            let arg_type = arguments[0].expr_type;
                            let first_param_type = original_params.first().copied();
                            let type_table = self.context.type_table.borrow();

                            let mut result = arg_type; // default: assume arg IS the type param

                            if let Some(param_tid) = first_param_type {
                                if let Some(param_ty) = type_table.get(param_tid) {
                                    match &param_ty.kind {
                                        // Param is the enum type itself (e.g., Tree<T>)
                                        // Extract the type args from the argument's enum type
                                        crate::tast::core::TypeKind::Enum {
                                            symbol_id: enum_sym,
                                            ..
                                        } if *enum_sym == parent_enum => {
                                            if let Some(arg_ty) = type_table.get(arg_type) {
                                                if let crate::tast::core::TypeKind::Enum {
                                                    type_args: arg_ta,
                                                    ..
                                                } = &arg_ty.kind
                                                {
                                                    if let Some(&first_ta) = arg_ta.first() {
                                                        result = first_ta;
                                                    }
                                                }
                                            }
                                        }
                                        // Param is a type parameter directly → arg type IS T
                                        crate::tast::core::TypeKind::TypeParameter { .. } => {
                                            result = arg_type;
                                        }
                                        _ => {}
                                    }
                                }
                            }
                            result
                        };

                        // Create instantiated enum type with inferred type args
                        let instantiated_enum_type = self
                            .context
                            .type_table
                            .borrow_mut()
                            .create_enum_type(parent_enum, vec![inferred_type]);

                        // Substitute each original param type with its instantiated version
                        let instantiated_params: Vec<TypeId> = original_params
                            .iter()
                            .map(|&param_type| {
                                let type_table = self.context.type_table.borrow();
                                if let Some(ty) = type_table.get(param_type) {
                                    match &ty.kind {
                                        crate::tast::core::TypeKind::TypeParameter { .. } => {
                                            inferred_type
                                        }
                                        crate::tast::core::TypeKind::Enum { symbol_id, .. }
                                            if *symbol_id == parent_enum =>
                                        {
                                            instantiated_enum_type
                                        }
                                        _ => param_type,
                                    }
                                } else {
                                    param_type
                                }
                            })
                            .collect();

                        // Create instantiated function type with correct param count and types
                        let instantiated_function_type = self
                            .context
                            .type_table
                            .borrow_mut()
                            .create_function_type(instantiated_params, instantiated_enum_type);

                        func_expr.expr_type = instantiated_function_type;
                        return Ok(func_expr);
                    }

                    // Fallback - couldn't infer
                    // println!("DEBUG: Could not infer type arguments, using original function type");
                    Ok(func_expr)
                }
                _ => {
                    // Not an enum type
                    Ok(func_expr)
                }
            }
        } else {
            Ok(func_expr)
        }
    }

    /// Find the parent enum symbol for an enum constructor
    fn find_parent_enum_for_constructor(&self, constructor_symbol: SymbolId) -> Option<SymbolId> {
        self.context
            .symbol_table
            .find_parent_enum_for_constructor(constructor_symbol)
    }

    /// Resolve a symbol by walking up the scope hierarchy
    fn resolve_symbol_in_scope_hierarchy(&self, name: InternedString) -> Option<SymbolId> {
        let mut current_scope = self.context.current_scope;

        loop {
            // Check if symbol exists in current scope
            if let Some(symbol) = self.context.symbol_table.lookup_symbol(current_scope, name) {
                return Some(symbol.id);
            }

            // Get parent scope
            if let Some(scope) = self.context.scope_tree.get_scope(current_scope) {
                if let Some(parent_id) = scope.parent_id {
                    current_scope = parent_id;
                } else {
                    // No parent scope
                    break;
                }
            } else {
                // Invalid scope
                break;
            }
        }

        // Check if the symbol is a field of the current class (implicit this access)
        if let Some(class_symbol) = self.context.class_context_stack.last() {
            if let Some(field_list) = self.class_fields.get(class_symbol) {
                for (field_name, field_symbol, _is_static) in field_list {
                    if *field_name == name {
                        return Some(*field_symbol);
                    }
                }
            }
        }

        // Check if the symbol is a method of the current class
        if let Some(class_symbol) = self.context.class_context_stack.last() {
            if let Some(methods) = self.class_methods.get(class_symbol) {
                for (method_name, method_symbol, _) in methods {
                    if *method_name == name {
                        return Some(*method_symbol);
                    }
                }
            }
        }

        // Fallback: explicitly check the global root scope (ScopeId::first())
        // This is needed for symbols like enum variants that are registered globally
        // but may not be reachable through the current scope's parent chain
        // (e.g., imported enums from other packages)
        let root_scope = ScopeId::first();
        if current_scope != root_scope {
            if let Some(symbol) = self.context.symbol_table.lookup_symbol(root_scope, name) {
                return Some(symbol.id);
            }
        }

        None
    }

    pub fn new(
        string_interner: &'a mut StringInterner,
        string_interner_rc: Rc<RefCell<StringInterner>>,
        symbol_table: &'a mut SymbolTable,
        type_table: &'a RefCell<TypeTable>,
        scope_tree: &'a mut ScopeTree,
        namespace_resolver: &'a mut super::namespace::NamespaceResolver,
        import_resolver: &'a mut super::namespace::ImportResolver,
    ) -> Self {
        let root_scope = ScopeId::first(); // Use first scope as root
        let context = LoweringContext::new(
            string_interner,
            string_interner_rc,
            symbol_table,
            type_table,
            scope_tree,
            root_scope,
            namespace_resolver,
            import_resolver,
        );

        Self {
            context,
            resolution_state: TypeResolutionState::default(),
            class_methods: HashMap::new(),
            class_fields: HashMap::new(),
            skip_stdlib_loading: false,
            skip_pre_registration: false,
            collected_errors: Vec::new(),
            using_modules: Vec::new(),
            pending_usings: Vec::new(),
            in_static_method: false,
            class_type_params: HashMap::new(),
            class_constructor_symbols: HashMap::new(),
        }
    }

    /// Set whether to skip internal stdlib loading (for CompilationUnit)
    pub fn set_skip_stdlib_loading(&mut self, skip: bool) {
        self.skip_stdlib_loading = skip;
    }

    /// Set whether to skip pre-registration pass (for CompilationUnit with two-pass compilation)
    pub fn set_skip_pre_registration(&mut self, skip: bool) {
        self.skip_pre_registration = skip;
    }

    /// Get all collected errors from both context and collected_errors
    pub fn get_all_errors(&self) -> Vec<LoweringError> {
        let mut all_errors = Vec::new();
        all_errors.extend(self.context.errors.clone());
        all_errors.extend(self.collected_errors.clone());
        all_errors
    }

    /// Set the package context explicitly (used for stdlib loading with "haxe" package)
    pub fn set_package_context(&mut self, package_name: &str) {
        let package_path: Vec<_> = package_name
            .split('.')
            .map(|s| self.context.string_interner.intern(s))
            .collect();
        let package_id = self
            .context
            .namespace_resolver
            .get_or_create_package(package_path);
        self.context.current_package = Some(package_id);
    }

    /// Set the package context from a parsed package path (Vec of path segments)
    pub fn set_package_from_parts(&mut self, parts: &[String]) {
        if !parts.is_empty() {
            self.set_package_context(&parts.join("."));
        }
    }

    /// Clear the current package context (used for root scope)
    pub fn clear_package_context(&mut self) {
        self.context.clear_package_context();
    }

    /// Initialize span converter for proper source location tracking
    pub fn initialize_span_converter(&mut self, file_id: u32, source_text: String) {
        self.context.initialize_span_converter(file_id, source_text);
    }

    /// Initialize span converter with specific filename for proper source location tracking
    pub fn initialize_span_converter_with_filename(
        &mut self,
        file_id: u32,
        source_text: String,
        file_name: String,
    ) {
        self.context
            .initialize_span_converter_with_filename(file_id, source_text, file_name);
    }

    /// Lower a complete Haxe file to TAST

    pub fn lower_file(&mut self, file: &HaxeFile) -> LoweringResult<TypedFile> {
        // Optimizer barrier

        // Create TypedFile with the shared interner from the pipeline
        let mut typed_file = TypedFile::new(Rc::clone(&self.context.string_interner_rc));

        // Process package declaration
        if let Some(package) = &file.package {
            typed_file.metadata.package_name = Some(package.path.join("."));

            // Create or get package in namespace resolver
            let package_path: Vec<_> = package
                .path
                .iter()
                .map(|s| self.context.string_interner.intern(s))
                .collect();
            let package_id = self
                .context
                .namespace_resolver
                .get_or_create_package(package_path.clone());
            self.context.current_package = Some(package_id);

            // Create package scope with full qualified name
            let package_name_str = package.path.join(".");
            let package_name_interned = self.context.string_interner.intern(&package_name_str);
            let package_scope = self
                .context
                .enter_named_scope(ScopeKind::Package, package_name_interned);
        }

        // Set file metadata
        typed_file.metadata.timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Set file name if available
        if !typed_file.metadata.file_path.is_empty() {
            if let Some(file_name) = typed_file.metadata.file_path.split('/').last() {
                typed_file.metadata.file_name =
                    Some(self.context.string_interner.intern(file_name));
            }
        }

        // Load standard library types from source files
        // (skip if CompilationUnit is handling stdlib loading)
        if !self.skip_stdlib_loading {
            self.load_standard_library()?;
        } else {
            // Even when skipping full stdlib loading, we MUST register top-level stdlib
            // symbols (Math, Std, etc.) so they can be resolved without explicit imports.
            // This is required for Haxe semantics where these classes are implicitly available.
            self.register_toplevel_stdlib_symbols();
        }

        // Process import.hx files in the current directory hierarchy
        if let Err(e) = self.process_import_hx_files(&file) {
            self.collected_errors.push(e);
        }

        // Process imports - collect errors but continue
        for import in &file.imports {
            match self.lower_import(import) {
                Ok(typed_import) => typed_file.imports.push(typed_import),
                Err(e) => self.collected_errors.push(e),
            }
        }

        // Process using statements - collect errors but continue
        for using in &file.using {
            match self.lower_using(using) {
                Ok(typed_using) => typed_file.using_statements.push(typed_using),
                Err(e) => self.collected_errors.push(e),
            }
        }

        // Process module-level fields - collect errors but continue
        for module_field in &file.module_fields {
            match self.lower_module_field(module_field) {
                Ok(typed_field) => typed_file.module_fields.push(typed_field),
                Err(e) => self.collected_errors.push(e),
            }
        }

        // First pass: Pre-register all type declarations in the symbol table
        // Skip this if CompilationUnit has already pre-registered all files
        if !self.skip_pre_registration {
            for declaration in &file.declarations {
                if let Err(e) = self.pre_register_declaration(declaration) {
                    self.collected_errors.push(e);
                }
            }
        }

        // Pass 1.5: Pre-register class fields for ALL classes before method bodies are lowered.
        // This enables forward references (e.g., NBody referencing Body fields when Body is
        // declared later in the file). Without this, field resolution falls back to placeholders
        // with Dynamic type, causing incorrect code generation.
        for declaration in &file.declarations {
            if let TypeDeclaration::Class(class_decl) = declaration {
                if let Err(e) = self.pre_register_class_fields(class_decl) {
                    self.collected_errors.push(e);
                }
            }
        }

        // Second pass: Process declarations with full type resolution
        for (i, declaration) in file.declarations.iter().enumerate() {
            match self.lower_declaration(declaration) {
                Ok(typed_decl) => match typed_decl {
                    TypedDeclaration::Function(func) => typed_file.functions.push(func),
                    TypedDeclaration::Class(class) => typed_file.classes.push(class),
                    TypedDeclaration::Interface(interface) => typed_file.interfaces.push(interface),
                    TypedDeclaration::Enum(enum_decl) => typed_file.enums.push(enum_decl),
                    TypedDeclaration::TypeAlias(alias) => typed_file.type_aliases.push(alias),
                    TypedDeclaration::Abstract(abstract_decl) => {
                        typed_file.abstracts.push(abstract_decl);
                    }
                },
                Err(e) => self.context.add_error(e),
            }
        }

        // Resolve any deferred type references (second pass)
        if let Err(e) = self.resolve_deferred_types() {
            self.collected_errors.push(e);
        }

        // Combine all errors from both context and collected_errors
        let mut all_errors = Vec::new();
        all_errors.extend(self.context.errors.clone());
        all_errors.extend(self.collected_errors.clone());

        // Check for errors - if any, return the first one but all are collected
        if !all_errors.is_empty() {
            // Store all errors in context for pipeline to access
            self.context.errors = all_errors.clone();
            return Err(all_errors.into_iter().next().unwrap());
        }

        Ok(typed_file)
    }

    /// Pre-register all type declarations in a file (first pass only)
    /// This registers class/interface/enum/typedef/abstract names in the namespace
    /// without lowering their bodies. Used for multi-file compilation where all
    /// type names need to be available before any file is fully compiled.
    pub fn pre_register_file(&mut self, file: &HaxeFile) -> LoweringResult<()> {
        // Process package declaration to set up the namespace context
        if let Some(package) = &file.package {
            // Create or get package in namespace resolver
            let package_path: Vec<_> = package
                .path
                .iter()
                .map(|s| self.context.string_interner.intern(s))
                .collect();
            let package_id = self
                .context
                .namespace_resolver
                .get_or_create_package(package_path.clone());
            self.context.current_package = Some(package_id);
        }

        // Pre-register all type declarations
        for declaration in &file.declarations {
            if let Err(e) = self.pre_register_declaration(declaration) {
                self.collected_errors.push(e);
            }
        }

        // Reset package context for next file
        self.context.current_package = None;

        // Return any errors that occurred during pre-registration
        if !self.collected_errors.is_empty() {
            return Err(self.collected_errors.pop().unwrap());
        }

        Ok(())
    }

    /// Process import.hx files in the directory hierarchy
    fn process_import_hx_files(&mut self, _current_file: &HaxeFile) -> LoweringResult<()> {
        use crate::tast::stdlib_loader::{StdLibConfig, StdLibLoader};
        use std::path::PathBuf;

        // Determine the current file's directory
        // In a real implementation, we'd get this from the compilation context
        // For now, we'll use a simple approach
        let current_dir = PathBuf::from("/Users/amaterasu/Vibranium/rayzor/compiler/examples");

        // Create a loader for import.hx files
        let mut config = StdLibConfig::default();
        config.load_import_hx = true;
        config.std_paths = vec![]; // We're not loading std lib here
        config.default_imports = vec![]; // No default imports

        let mut loader = StdLibLoader::new(config);

        // Look for import.hx files in the current directory and parent directories
        let mut search_dir = current_dir.clone();
        let mut import_files = Vec::new();

        loop {
            let import_hx_files = loader.load_import_hx(&search_dir);
            import_files.extend(import_hx_files);

            // Move to parent directory
            match search_dir.parent() {
                Some(parent) => search_dir = parent.to_path_buf(),
                None => break,
            }

            // Stop at project root (avoid going too far up)
            if search_dir.ends_with("rayzor") {
                break;
            }
        }

        // Process import.hx files in reverse order (parent directories first)
        import_files.reverse();

        for import_file in import_files {
            // Process imports from import.hx
            for import in &import_file.imports {
                self.process_import_from_import_hx(import)?;
            }

            // Process using statements from import.hx
            for using in &import_file.using {
                self.process_using_from_import_hx(using)?;
            }

            // Process type declarations from import.hx
            // Pre-register first (creates symbols in scope), then fully lower enums
            // so their variant constructor types are properly set (not left as TypeId::invalid)
            for declaration in &import_file.declarations {
                self.pre_register_declaration(declaration)?;
            }
            for declaration in &import_file.declarations {
                if let TypeDeclaration::Enum(enum_decl) = declaration {
                    let _ = self.lower_enum_declaration(enum_decl);
                }
            }
        }

        Ok(())
    }

    /// Process an import from import.hx file
    fn process_import_from_import_hx(&mut self, import: &Import) -> LoweringResult<()> {
        // Similar to lower_import, but adds to global scope
        let imported_symbols = match &import.mode {
            parser::ImportMode::Normal => import.path.last().map(|s| vec![s.as_str()]),
            parser::ImportMode::Alias(alias) => Some(vec![alias.as_str()]),
            parser::ImportMode::Field(field) => Some(vec![field.as_str()]),
            parser::ImportMode::Wildcard => None,
            parser::ImportMode::WildcardWithExclusions(_) => None,
        };

        // Register imported symbols in the symbol table for type resolution
        if let Some(ref symbols) = imported_symbols {
            for symbol_name in symbols {
                let interned_name = self.context.intern_string(symbol_name);

                // Build qualified path for namespace resolver lookup
                let qualified_path = super::namespace::QualifiedPath {
                    package: import.path[..import.path.len() - 1]
                        .iter()
                        .map(|s| self.context.intern_string(s))
                        .collect(),
                    name: interned_name,
                };

                // Check if symbol already exists in namespace resolver (from compiled dependencies)
                // This is preferred over scope hierarchy because namespace has the real types
                let imported_symbol = if let Some(existing) = self
                    .context
                    .namespace_resolver
                    .lookup_symbol(&qualified_path)
                {
                    existing
                } else if let Some(existing) = self.resolve_symbol_in_scope_hierarchy(interned_name)
                {
                    existing
                } else {
                    // Create a placeholder symbol for the imported type
                    let new_sym = self
                        .context
                        .symbol_table
                        .create_class_in_scope(interned_name, ScopeId::first());

                    // Update qualified name
                    self.context.update_symbol_qualified_name(new_sym);
                    new_sym
                };

                // Add to root scope so it can be resolved globally
                self.context
                    .scope_tree
                    .get_scope_mut(ScopeId::first())
                    .expect("Root scope should exist")
                    .add_symbol(imported_symbol, interned_name);
            }
        }

        Ok(())
    }

    /// Process a using statement from import.hx file
    fn process_using_from_import_hx(&mut self, using: &Using) -> LoweringResult<()> {
        // Register the using statement globally
        // In a full implementation, we'd track these for static extension resolution
        let _path = using.path.join(".");
        // TODO: Track global using statements for static extension resolution
        Ok(())
    }

    /// Load standard library types from source files
    fn load_standard_library(&mut self) -> LoweringResult<()> {
        use crate::tast::stdlib_loader::{StdLibConfig, StdLibLoader};
        use std::path::PathBuf;

        // IMPORTANT: Save current package context and clear it for stdlib loading
        // Standard library symbols should NOT inherit the current file's package prefix
        let saved_package = self.context.current_package.take();

        // Configure standard library paths
        let mut config = StdLibConfig::default();
        config.std_paths = vec![
            // Look for standard library files relative to the compiler crate
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("haxe-std"),
        ];
        config.default_imports = vec![
            "StdTypes.hx".to_string(),
            "String.hx".to_string(),
            "Array.hx".to_string(),
            "Iterator.hx".to_string(),
            "Std.hx".to_string(),
        ];

        let mut loader = StdLibLoader::new(config);

        // Load default imports (top-level types)
        let std_files = loader.load_default_imports();

        // Process each standard library file
        for std_file in std_files {
            // First pass: pre-register all types so they can reference each other
            for declaration in &std_file.declarations {
                self.pre_register_declaration(declaration)?;
            }
        }

        // Second pass: fully lower the standard library declarations to register methods
        // This is critical for things like Array.push to be available
        let std_files = loader.load_default_imports();
        for std_file in std_files {
            for declaration in &std_file.declarations {
                // Fully lower each declaration to register all methods and fields
                match declaration {
                    TypeDeclaration::Class(class_decl) => {
                        // Check if this is the Array class
                        if class_decl.name == "Array" {
                            // println!("DEBUG: Processing Array class from standard library");
                            // println!("DEBUG: Array has {} fields", class_decl.fields.len());
                            for field in &class_decl.fields {
                                if let ClassFieldKind::Function(func) = &field.kind {
                                    // println!("DEBUG: Array method: {}", func.name);
                                }
                            }
                        }

                        // Lower the class to register all its methods
                        let _ = self.lower_class_declaration(class_decl);
                    }
                    TypeDeclaration::Interface(interface_decl) => {
                        let _ = self.lower_interface_declaration(interface_decl);
                    }
                    TypeDeclaration::Enum(enum_decl) => {
                        let _ = self.lower_enum_declaration(enum_decl);
                    }
                    TypeDeclaration::Typedef(typedef_decl) => {
                        let _ = self.lower_typedef_declaration(typedef_decl);
                    }
                    TypeDeclaration::Abstract(abstract_decl) => {
                        let _ = self.lower_abstract_declaration(abstract_decl);
                    }
                    _ => {
                        // Package declarations etc.
                    }
                }
            }
        }

        // Register root-level stdlib classes that need to be resolvable by name.
        // These are top-level Haxe standard library classes (Std.hx, Sys.hx, Math.hx, etc.)
        // that exist as extern classes and need their symbols pre-registered so the compiler
        // can resolve references like `Std.int()` or `Math.sin()` before loading the full definitions.
        let toplevel_stdlib_classes = [
            "Dynamic",
            "Class",
            "Enum",
            "Type",
            "Any",
            "Unknown",
            "Map",
            "List",
            "Vector",
            "Date",
            "Math",
            "Reflect",
            "Std",
            "Sys",
            "File",
            "FileSystem",
            "StringBuf",
            "StringTools",
            "EReg",
            "Xml",
            "Json",
            "Timer",
            "Bytes",
            "Int32",
            "Int64",
            "UInt",
        ];

        for type_name in &toplevel_stdlib_classes {
            let interned_name = self.context.intern_string(type_name);

            // Skip if this class was already registered (e.g., from loading Std.hx).
            // Creating a duplicate symbol would shadow the fully-lowered one that has
            // methods and fields registered, breaking static method resolution.
            if let Some(_existing) = self
                .context
                .symbol_table
                .lookup_symbol(ScopeId::first(), interned_name)
            {
                continue;
            }

            let builtin_symbol = self
                .context
                .symbol_table
                .create_class_in_scope(interned_name, ScopeId::first());

            // Update qualified name
            self.context.update_symbol_qualified_name(builtin_symbol);

            // Add to root scope for global resolution
            self.context
                .scope_tree
                .get_scope_mut(ScopeId::first())
                .expect("Root scope should exist")
                .add_symbol(builtin_symbol, interned_name);
        }

        // Register built-in global functions
        let builtin_functions = [
            ("trace", vec!["Dynamic"], "Void"), // trace(value: Dynamic): Void
        ];

        for (func_name, param_types, return_type) in builtin_functions {
            let func_name_interned = self.context.intern_string(func_name);

            // Create parameter types
            let mut param_type_ids = Vec::new();
            for param_type_name in param_types {
                let param_type_id = match param_type_name {
                    "Dynamic" => self.context.type_table.borrow().dynamic_type(),
                    "Int" => self.context.type_table.borrow().int_type(),
                    "String" => self.context.type_table.borrow().string_type(),
                    "Float" => self.context.type_table.borrow().float_type(),
                    "Bool" => self.context.type_table.borrow().bool_type(),
                    "Void" => self.context.type_table.borrow().void_type(),
                    _ => self.context.type_table.borrow().dynamic_type(),
                };
                param_type_ids.push(param_type_id);
            }

            // Create return type
            let return_type_id = match return_type {
                "Dynamic" => self.context.type_table.borrow().dynamic_type(),
                "Int" => self.context.type_table.borrow().int_type(),
                "String" => self.context.type_table.borrow().string_type(),
                "Float" => self.context.type_table.borrow().float_type(),
                "Bool" => self.context.type_table.borrow().bool_type(),
                "Void" => self.context.type_table.borrow().void_type(),
                _ => self.context.type_table.borrow().dynamic_type(),
            };

            // Create function type (not just return type)
            let function_type_id = self
                .context
                .type_table
                .borrow_mut()
                .create_function_type(param_type_ids, return_type_id);

            // Create function symbol manually with proper scope
            use crate::tast::{
                LifetimeId, Mutability, SourceLocation, Symbol, SymbolFlags, SymbolKind, Visibility,
            };

            let func_symbol_id = SymbolId::from_raw(self.context.symbol_table.len() as u32);
            let func_symbol = Symbol {
                id: func_symbol_id,
                name: func_name_interned,
                kind: SymbolKind::Function,
                type_id: function_type_id, // Use function type, not return type
                scope_id: ScopeId::first(),
                lifetime_id: LifetimeId::invalid(),
                visibility: Visibility::Public,
                mutability: Mutability::Immutable,
                definition_location: SourceLocation::unknown(),
                is_used: false,
                is_exported: false,
                documentation: None,
                flags: SymbolFlags::NONE,
                package_id: None,
                qualified_name: None,
                native_name: None,
                frameworks: None,
                c_includes: None,
                c_sources: None,
                c_libs: None,
            };

            // Add symbol to symbol table
            self.context.symbol_table.add_symbol(func_symbol);

            // Add to root scope for global resolution
            self.context
                .scope_tree
                .get_scope_mut(ScopeId::first())
                .expect("Root scope should exist")
                .add_symbol(func_symbol_id, func_name_interned);
        }

        // Restore the original package context
        self.context.current_package = saved_package;

        Ok(())
    }

    /// Register top-level stdlib symbols (Math, Std, etc.) for implicit availability.
    ///
    /// In Haxe, these classes are always available without explicit imports.
    /// This method is called separately from load_standard_library() to support
    /// lazy stdlib loading where we want to skip parsing/processing stdlib files
    /// but still need these symbols to be resolvable.
    fn register_toplevel_stdlib_symbols(&mut self) {
        // Top-level Haxe standard library classes that are always implicitly available.
        // These match the .hx files in haxe-std/ root directory.
        let toplevel_stdlib_classes = [
            // Core types from StdTypes.hx
            "Dynamic",
            "Class",
            "Enum",
            "EnumValue",
            "Type",
            "Any",
            "Unknown",
            // Collections - CRITICAL: Array must be here for array literals
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
            // System types (may be in subdirectories but commonly used)
            "File",
            "FileSystem",
            "Json",
            "Timer",
            "Bytes",
            "Int32",
            "Int64",
        ];

        for type_name in &toplevel_stdlib_classes {
            let interned_name = self.context.intern_string(type_name);

            // Check if already registered (avoid duplicates)
            if self
                .resolve_symbol_in_scope_hierarchy(interned_name)
                .is_some()
            {
                continue;
            }

            let builtin_symbol = self
                .context
                .symbol_table
                .create_class_in_scope(interned_name, ScopeId::first());

            // Update qualified name
            self.context.update_symbol_qualified_name(builtin_symbol);

            // Add to root scope for global resolution
            self.context
                .scope_tree
                .get_scope_mut(ScopeId::first())
                .expect("Root scope should exist")
                .add_symbol(builtin_symbol, interned_name);
        }

        // Register built-in global functions (trace)
        self.register_builtin_functions();
    }

    /// Register built-in global functions like trace()
    fn register_builtin_functions(&mut self) {
        let builtin_functions = [("trace", vec!["Dynamic"], "Void")];

        for (func_name, param_types, return_type) in builtin_functions {
            let func_name_interned = self.context.intern_string(func_name);

            // Check if already registered
            if self
                .resolve_symbol_in_scope_hierarchy(func_name_interned)
                .is_some()
            {
                continue;
            }

            // Create parameter types
            let mut param_type_ids = Vec::new();
            for param_type_name in param_types {
                let param_type_id = match param_type_name {
                    "Dynamic" => self.context.type_table.borrow().dynamic_type(),
                    "Int" => self.context.type_table.borrow().int_type(),
                    "String" => self.context.type_table.borrow().string_type(),
                    "Float" => self.context.type_table.borrow().float_type(),
                    "Bool" => self.context.type_table.borrow().bool_type(),
                    "Void" => self.context.type_table.borrow().void_type(),
                    _ => self.context.type_table.borrow().dynamic_type(),
                };
                param_type_ids.push(param_type_id);
            }

            // Create return type
            let return_type_id = match return_type {
                "Dynamic" => self.context.type_table.borrow().dynamic_type(),
                "Int" => self.context.type_table.borrow().int_type(),
                "String" => self.context.type_table.borrow().string_type(),
                "Float" => self.context.type_table.borrow().float_type(),
                "Bool" => self.context.type_table.borrow().bool_type(),
                "Void" => self.context.type_table.borrow().void_type(),
                _ => self.context.type_table.borrow().dynamic_type(),
            };

            // Create function type
            let function_type_id = self
                .context
                .type_table
                .borrow_mut()
                .create_function_type(param_type_ids, return_type_id);

            // Create function symbol
            use crate::tast::{
                LifetimeId, Mutability, SourceLocation, Symbol, SymbolFlags, SymbolKind, Visibility,
            };

            let func_symbol_id = SymbolId::from_raw(self.context.symbol_table.len() as u32);
            let func_symbol = Symbol {
                id: func_symbol_id,
                name: func_name_interned,
                kind: SymbolKind::Function,
                type_id: function_type_id,
                scope_id: ScopeId::first(),
                lifetime_id: LifetimeId::invalid(),
                visibility: Visibility::Public,
                mutability: Mutability::Immutable,
                definition_location: SourceLocation::unknown(),
                is_used: false,
                is_exported: false,
                documentation: None,
                flags: SymbolFlags::NONE,
                package_id: None,
                qualified_name: None,
                native_name: None,
                frameworks: None,
                c_includes: None,
                c_sources: None,
                c_libs: None,
            };

            self.context.symbol_table.add_symbol(func_symbol);

            self.context
                .scope_tree
                .get_scope_mut(ScopeId::first())
                .expect("Root scope should exist")
                .add_symbol(func_symbol_id, func_name_interned);
        }
    }

    /// Register a symbol with package information
    fn register_symbol_with_package(&mut self, symbol_id: SymbolId, name: &str) {
        if let Some(package_id) = self.context.current_package {
            let interned_name = self.context.string_interner.intern(name);

            // Register symbol in namespace
            self.context
                .namespace_resolver
                .register_symbol(package_id, interned_name, symbol_id);

            // Update symbol with package info and qualified name
            if let Some(symbol) = self.context.symbol_table.get_symbol_mut(symbol_id) {
                symbol.package_id = Some(package_id);

                // Create qualified name
                if let Some(package) = self.context.namespace_resolver.get_package(package_id) {
                    let qualified_name = if package.full_path.is_empty() {
                        name.to_string()
                    } else {
                        format!(
                            "{}.{}",
                            package
                                .full_path
                                .iter()
                                .map(|&s| self
                                    .context
                                    .string_interner
                                    .get(s)
                                    .unwrap_or("<unknown>"))
                                .collect::<Vec<_>>()
                                .join("."),
                            name
                        )
                    };
                    symbol.qualified_name =
                        Some(self.context.string_interner.intern(&qualified_name));
                }
            }
        }
    }

    /// Pre-register type declarations in the symbol table (first pass)
    pub fn pre_register_declaration(
        &mut self,
        declaration: &TypeDeclaration,
    ) -> LoweringResult<()> {
        match declaration {
            TypeDeclaration::Class(class_decl) => {
                let class_name = self.context.intern_string(&class_decl.name);

                // Check if this class already exists in the root scope (from a previous compilation)
                // If so, skip pre-registration to avoid creating duplicate symbols
                if self
                    .context
                    .symbol_table
                    .lookup_symbol(ScopeId::first(), class_name)
                    .is_some()
                {
                    // Class already pre-registered, skip
                    return Ok(());
                }

                let class_symbol = self
                    .context
                    .symbol_table
                    .create_class_in_scope(class_name, ScopeId::first());

                // Register symbol with package information (also sets qualified name)
                self.register_symbol_with_package(class_symbol, &class_decl.name);

                // Create the corresponding type for this class
                let class_type = self.context.type_table.borrow_mut().create_type(
                    crate::tast::core::TypeKind::Class {
                        symbol_id: class_symbol,
                        type_args: Vec::new(), // Will be updated during full lowering
                    },
                );

                // Set the symbol's type_id to link it to the type
                self.context
                    .symbol_table
                    .update_symbol_type(class_symbol, class_type);

                // Register the type-to-symbol mapping so we can look up symbols from types
                self.context
                    .symbol_table
                    .register_type_symbol_mapping(class_type, class_symbol);

                // Add to root scope for global resolution
                self.context
                    .scope_tree
                    .get_scope_mut(ScopeId::first())
                    .expect("Root scope should exist")
                    .add_symbol(class_symbol, class_name);
            }
            TypeDeclaration::Interface(interface_decl) => {
                let interface_name = self.context.intern_string(&interface_decl.name);

                // Check if this interface already exists in the root scope
                if self
                    .context
                    .symbol_table
                    .lookup_symbol(ScopeId::first(), interface_name)
                    .is_some()
                {
                    return Ok(());
                }

                let interface_symbol = self
                    .context
                    .symbol_table
                    .create_interface_in_scope(interface_name, ScopeId::first());

                // Register symbol with package information (also sets qualified name)
                self.register_symbol_with_package(interface_symbol, &interface_decl.name);

                // Create the corresponding type for this interface
                let interface_type = self.context.type_table.borrow_mut().create_type(
                    crate::tast::core::TypeKind::Interface {
                        symbol_id: interface_symbol,
                        type_args: Vec::new(), // Will be updated during full lowering
                    },
                );

                // Set the symbol's type_id to link it to the type
                self.context
                    .symbol_table
                    .update_symbol_type(interface_symbol, interface_type);

                // Register the type-to-symbol mapping so we can look up symbols from types
                self.context
                    .symbol_table
                    .register_type_symbol_mapping(interface_type, interface_symbol);

                // Add to root scope for global resolution
                self.context
                    .scope_tree
                    .get_scope_mut(ScopeId::first())
                    .expect("Root scope should exist")
                    .add_symbol(interface_symbol, interface_name);
            }
            TypeDeclaration::Enum(enum_decl) => {
                let enum_name = self.context.intern_string(&enum_decl.name);

                // Check if this enum already exists in the root scope
                if self
                    .context
                    .symbol_table
                    .lookup_symbol(ScopeId::first(), enum_name)
                    .is_some()
                {
                    return Ok(());
                }

                let enum_symbol = self
                    .context
                    .symbol_table
                    .create_enum_in_scope(enum_name, ScopeId::first());

                // Register symbol with package information (also sets qualified name)
                self.register_symbol_with_package(enum_symbol, &enum_decl.name);

                // Add to root scope for global resolution
                self.context
                    .scope_tree
                    .get_scope_mut(ScopeId::first())
                    .expect("Root scope should exist")
                    .add_symbol(enum_symbol, enum_name);

                // IMPORTANT: Also pre-register enum variants so they can be resolved
                // during pattern matching even before the enum is fully lowered
                for variant in &enum_decl.constructors {
                    let variant_name = self.context.intern_string(&variant.name);
                    let variant_symbol = self.context.symbol_table.create_enum_variant_in_scope(
                        variant_name,
                        ScopeId::first(),
                        enum_symbol,
                    );

                    // Add variant to root scope for global resolution
                    self.context
                        .scope_tree
                        .get_scope_mut(ScopeId::first())
                        .expect("Root scope should exist")
                        .add_symbol(variant_symbol, variant_name);
                }
            }
            TypeDeclaration::Typedef(typedef_decl) => {
                let typedef_name = self.context.intern_string(&typedef_decl.name);

                // Check if this typedef already exists in the root scope
                if self
                    .context
                    .symbol_table
                    .lookup_symbol(ScopeId::first(), typedef_name)
                    .is_some()
                {
                    return Ok(());
                }

                let typedef_symbol = self
                    .context
                    .symbol_table
                    .create_class_in_scope(typedef_name, ScopeId::first()); // Reuse class for typedefs

                // Register symbol with package information (also sets qualified name)
                self.register_symbol_with_package(typedef_symbol, &typedef_decl.name);

                // Add to root scope for global resolution
                self.context
                    .scope_tree
                    .get_scope_mut(ScopeId::first())
                    .expect("Root scope should exist")
                    .add_symbol(typedef_symbol, typedef_name);
            }
            TypeDeclaration::Abstract(abstract_decl) => {
                let abstract_name = self.context.intern_string(&abstract_decl.name);

                // Check if this abstract already exists in the root scope
                if let Some(existing) = self
                    .context
                    .symbol_table
                    .lookup_symbol(ScopeId::first(), abstract_name)
                {
                    let existing_id = existing.id;
                    // The symbol may have been created as SymbolKind::Class by import resolution
                    // (which doesn't know the declaration kind). Fix it to Abstract now that we
                    // know the actual declaration type. We must fix BOTH:
                    // 1. The symbol kind (Class -> Abstract)
                    // 2. The type in the type table (create a new Abstract type, since types are immutable)
                    let needs_fix = self
                        .context
                        .symbol_table
                        .get_symbol(existing_id)
                        .map(|s| s.kind == crate::tast::SymbolKind::Class)
                        .unwrap_or(false);
                    if needs_fix {
                        if let Some(sym) = self.context.symbol_table.get_symbol_mut(existing_id) {
                            sym.kind = crate::tast::SymbolKind::Abstract;
                        }
                        // Create a proper Abstract type to replace the Class type
                        let abstract_type = self
                            .context
                            .type_table
                            .borrow_mut()
                            .create_abstract_type(existing_id, None, Vec::new());
                        self.context
                            .symbol_table
                            .update_symbol_type(existing_id, abstract_type);
                        self.context
                            .symbol_table
                            .register_type_symbol_mapping(abstract_type, existing_id);
                    }
                    return Ok(());
                }

                let abstract_symbol = self
                    .context
                    .symbol_table
                    .create_abstract_in_scope(abstract_name, ScopeId::first());

                // Register symbol with package information (also sets qualified name)
                self.register_symbol_with_package(abstract_symbol, &abstract_decl.name);

                // Add to root scope for global resolution
                self.context
                    .scope_tree
                    .get_scope_mut(ScopeId::first())
                    .expect("Root scope should exist")
                    .add_symbol(abstract_symbol, abstract_name);
            }
            TypeDeclaration::Conditional(_) => {
                // Skip conditional compilation blocks in pre-registration
            }
        }
        Ok(())
    }

    /// Pre-register class fields for forward reference resolution.
    /// This runs after pre_register_declaration (which creates class type entries)
    /// but before full lowering, so that field access on forward-referenced classes
    /// can resolve field names and types correctly.
    fn pre_register_class_fields(&mut self, class_decl: &parser::ClassDecl) -> LoweringResult<()> {
        let class_name = self.context.intern_string(&class_decl.name);

        // Look up the pre-registered class symbol
        let class_symbol = match self
            .context
            .symbol_table
            .lookup_symbol(ScopeId::first(), class_name)
        {
            Some(entry) => entry.id,
            None => return Ok(()), // Not pre-registered, skip
        };

        // If class_fields already has entries for this class, skip (already registered)
        if self.class_fields.contains_key(&class_symbol) {
            return Ok(());
        }

        // Initialize the field list
        self.class_fields.insert(class_symbol, Vec::new());

        // Register each var/final/property field
        for field in &class_decl.fields {
            let (field_name, type_hint) = match &field.kind {
                parser::ClassFieldKind::Var {
                    name, type_hint, ..
                } => (name.clone(), type_hint.as_ref()),
                parser::ClassFieldKind::Final {
                    name, type_hint, ..
                } => (name.clone(), type_hint.as_ref()),
                parser::ClassFieldKind::Property {
                    name, type_hint, ..
                } => (name.clone(), type_hint.as_ref()),
                parser::ClassFieldKind::Function(_) => continue, // Skip methods
            };

            let is_static = field
                .modifiers
                .iter()
                .any(|m| matches!(m, parser::Modifier::Static));

            // Resolve the field type from the type hint
            let field_type = if let Some(th) = type_hint {
                self.lower_type(th)
                    .unwrap_or_else(|_| self.context.type_table.borrow().dynamic_type())
            } else {
                self.context.type_table.borrow().dynamic_type()
            };

            let interned_name = self.context.intern_string(&field_name);
            let field_symbol = self.context.symbol_table.create_variable(interned_name);

            // Set the field's type
            self.context
                .symbol_table
                .update_symbol_type(field_symbol, field_type);

            // Mark as field
            if let Some(sym) = self.context.symbol_table.get_symbol_mut(field_symbol) {
                sym.kind = crate::tast::SymbolKind::Field;
            }

            // Add to class_fields
            if let Some(field_list) = self.class_fields.get_mut(&class_symbol) {
                field_list.push((interned_name, field_symbol, is_static));
            }
        }

        Ok(())
    }

    /// Extract module name from file
    fn extract_module_name(&self, file: &HaxeFile) -> String {
        if let Some(package) = &file.package {
            package.path.join(".")
        } else {
            "default".to_string()
        }
    }

    /// Lower an import declaration
    fn lower_import(&mut self, import: &Import) -> LoweringResult<TypedImport> {
        let imported_symbols = match &import.mode {
            parser::ImportMode::Normal => import
                .path
                .last()
                .map(|s| vec![self.context.intern_string(s)]),
            parser::ImportMode::Alias(alias) => Some(vec![self.context.intern_string(alias)]),
            parser::ImportMode::Field(field) => Some(vec![self.context.intern_string(field)]),
            parser::ImportMode::Wildcard => None,
            parser::ImportMode::WildcardWithExclusions(_) => None,
        };

        let alias = match &import.mode {
            parser::ImportMode::Alias(alias) => Some(self.context.intern_string(alias)),
            _ => None,
        };

        // Create import entry for the import resolver
        let package_path: Vec<_> = import
            .path
            .iter()
            .take(import.path.len().saturating_sub(1)) // All but last element are package path
            .map(|s| self.context.string_interner.intern(s))
            .collect();

        let type_name = import
            .path
            .last()
            .map(|s| self.context.string_interner.intern(s))
            .unwrap_or_else(|| self.context.string_interner.intern("Unknown"));

        let qualified_path = super::namespace::QualifiedPath::new(package_path, type_name);

        // println!("Debug qualified path: {:?}", import.path);

        let alias_interned = alias;

        let exclusions = match &import.mode {
            parser::ImportMode::WildcardWithExclusions(excl) => excl
                .iter()
                .map(|e| self.context.string_interner.intern(e))
                .collect(),
            _ => Vec::new(),
        };

        // Clone qualified_path before moving it into import_entry
        let qualified_path_for_lookup = qualified_path.clone();

        let import_entry = super::namespace::ImportEntry {
            package_path: qualified_path,
            alias: alias_interned,
            exclusions,
            is_wildcard: matches!(
                import.mode,
                parser::ImportMode::Wildcard | parser::ImportMode::WildcardWithExclusions(_)
            ),
            location: self.context.create_location_from_span(import.span),
        };

        // Add import to current scope
        self.context
            .import_resolver
            .add_import(self.context.current_scope, import_entry);

        // Register imported symbols in the symbol table for type resolution
        if let Some(ref symbols) = imported_symbols {
            for &symbol_name in symbols {
                // IMPORTANT: Check if this symbol was already pre-registered (e.g., from stdlib loading)
                // If so, reuse that symbol instead of creating a duplicate
                // First try to look up by the full qualified path from the import
                let imported_symbol = if let Some(existing_symbol) = self
                    .context
                    .namespace_resolver
                    .lookup_symbol(&qualified_path_for_lookup)
                {
                    // Reuse the pre-registered symbol from namespace
                    existing_symbol
                } else if let Some(existing_symbol) =
                    self.resolve_symbol_in_scope_hierarchy(symbol_name)
                {
                    // Symbol already exists (likely from pre-registration)
                    // Reuse the existing symbol regardless of its kind (Class, Enum, Interface, etc.)
                    // This preserves the correct type info from the compiled file
                    if let Some(sym) = self.context.symbol_table.get_symbol(existing_symbol) {
                        // CRITICAL FIX: If the symbol has an invalid type_id, create a type for it
                        // This happens for extern classes that were created as placeholders but
                        // never had their type assigned
                        if !sym.type_id.is_valid()
                            && sym.kind == crate::tast::symbols::SymbolKind::Class
                        {
                            let class_type = self.context.type_table.borrow_mut().create_type(
                                crate::tast::core::TypeKind::Class {
                                    symbol_id: existing_symbol,
                                    type_args: Vec::new(),
                                },
                            );
                            self.context
                                .symbol_table
                                .update_symbol_type(existing_symbol, class_type);
                            self.context
                                .symbol_table
                                .register_type_symbol_mapping(class_type, existing_symbol);
                        }
                        existing_symbol
                    } else {
                        // Symbol ID exists but can't get symbol data - create new
                        let new_sym = self
                            .context
                            .symbol_table
                            .create_class_in_scope(symbol_name, ScopeId::first());
                        // Use the full import path as the qualified name
                        let full_qualified_name = import.path.join(".");
                        if let Some(sym) = self.context.symbol_table.get_symbol_mut(new_sym) {
                            sym.qualified_name =
                                Some(self.context.string_interner.intern(&full_qualified_name));
                        }
                        new_sym
                    }
                } else if let Some(existing) = self
                    .context
                    .symbol_table
                    .lookup_symbol(ScopeId::first(), symbol_name)
                {
                    // Symbol exists in root scope (from a previously compiled file)
                    // Reuse it to preserve correct type kind (Abstract, Enum, etc.)
                    existing.id
                } else {
                    // Create a placeholder symbol for the imported type
                    let new_sym = self
                        .context
                        .symbol_table
                        .create_class_in_scope(symbol_name, ScopeId::first());

                    // CRITICAL FIX: Set the qualified name to the FULL import path, not just the class name.
                    // When importing "rayzor.Bytes", the qualified name should be "rayzor.Bytes", not just "Bytes".
                    // This is needed for runtime mapping to work correctly (e.g., "rayzor_Bytes" pattern matching).
                    let full_qualified_name = import.path.join(".");
                    if let Some(sym) = self.context.symbol_table.get_symbol_mut(new_sym) {
                        sym.qualified_name =
                            Some(self.context.string_interner.intern(&full_qualified_name));
                    }

                    // CRITICAL: For imported classes (especially extern classes like StringMap),
                    // we must create a class type and link it to the symbol. Without this,
                    // new StringMap<Int>() will have TypeId::invalid() because the symbol has no type.
                    let class_type = self.context.type_table.borrow_mut().create_type(
                        crate::tast::core::TypeKind::Class {
                            symbol_id: new_sym,
                            type_args: Vec::new(), // Type args are applied at instantiation
                        },
                    );
                    self.context
                        .symbol_table
                        .update_symbol_type(new_sym, class_type);
                    self.context
                        .symbol_table
                        .register_type_symbol_mapping(class_type, new_sym);

                    new_sym
                };

                // Add to root scope so it can be resolved
                // Note: If symbol was pre-registered, it should already be in the scope,
                // but adding it again is idempotent
                self.context
                    .scope_tree
                    .get_scope_mut(ScopeId::first())
                    .expect("Root scope should exist")
                    .add_symbol(imported_symbol, symbol_name);
            }
        }

        Ok(TypedImport {
            module_path: self.context.intern_string(&import.path.join(".")),
            imported_symbols,
            alias,
            source_location: self.context.create_location_from_span(import.span),
        })
    }

    /// Lower a using declaration
    fn lower_using(&mut self, using: &Using) -> LoweringResult<TypedUsing> {
        let module_path_str = using.path.join(".");
        let module_path = self.context.intern_string(&module_path_str);

        // Try to resolve the using module to a class symbol for static extension resolution
        // The module path is typically just the class name (e.g., "StringTools")
        // or a qualified path (e.g., "haxe.StringTools")
        let class_name = using
            .path
            .last()
            .map(|s| s.as_str())
            .unwrap_or(&module_path_str);
        let class_name_interned = self.context.intern_string(class_name);

        // First try to find via namespace resolver (handles qualified paths)
        let package_path: Vec<_> = using
            .path
            .iter()
            .take(using.path.len().saturating_sub(1))
            .map(|s| self.context.string_interner.intern(s))
            .collect();
        let qualified_path =
            super::namespace::QualifiedPath::new(package_path, class_name_interned);

        let class_symbol_id = if let Some(symbol_id) = self
            .context
            .namespace_resolver
            .lookup_symbol(&qualified_path)
        {
            // Found via namespace resolver
            Some(symbol_id)
        } else if let Some(class_symbol) = self
            .context
            .symbol_table
            .lookup_symbol(ScopeId::first(), class_name_interned)
        {
            // Found in global scope - but check if this symbol was actually lowered
            // If scope_id is ScopeId(0), it was only pre-registered but not lowered
            // In that case, search for a symbol with the same name that WAS lowered
            if class_symbol.scope_id == ScopeId::first() {
                // This symbol wasn't lowered - search for one that was
                let mut found_lowered = None;
                for sym in self
                    .context
                    .symbol_table
                    .symbols_of_kind(crate::tast::symbols::SymbolKind::Class)
                {
                    if sym.name == class_name_interned && sym.scope_id != ScopeId::first() {
                        found_lowered = Some(sym.id);
                        break;
                    }
                }
                if found_lowered.is_some() {
                    found_lowered
                } else {
                    // No lowered symbol found, use pre-registered one (will trigger loading)
                    Some(class_symbol.id)
                }
            } else {
                Some(class_symbol.id)
            }
        } else {
            None
        };

        if let Some(symbol_id) = class_symbol_id {
            // Found the class - register it for static extension resolution
            // Check if the class has been fully compiled (scope_id should not be ScopeId::first() or ScopeId(0))
            let needs_loading = if let Some(sym) = self.context.symbol_table.get_symbol(symbol_id) {
                // If scope_id is still the root scope (ScopeId(0)), the class was only pre-registered
                // and not actually compiled with its method bodies
                sym.scope_id == ScopeId::first()
            } else {
                true
            };

            if needs_loading {
                // Queue the module for loading - the compilation unit will load it
                self.pending_usings.push(module_path_str.clone());
            }

            self.using_modules.push((class_name_interned, symbol_id));
        }
        // Note: If class not found, static extensions will still work through the
        // "LAST RESORT" mechanism in hir_to_mir.rs which searches all stdlib classes

        Ok(TypedUsing {
            module_path,
            target_type: None, // TODO: Handle target type if specified
            source_location: self.context.create_location_from_span(using.span),
        })
    }

    /// Lower a module field
    fn lower_module_field(
        &mut self,
        module_field: &ModuleField,
    ) -> LoweringResult<TypedModuleField> {
        let field_name = match &module_field.kind {
            parser::ModuleFieldKind::Var { name, .. } => name.clone(),
            parser::ModuleFieldKind::Final { name, .. } => name.clone(),
            parser::ModuleFieldKind::Function(func) => func.name.clone(),
        };

        let interned_name = self.context.intern_string(&field_name);
        let field_symbol = self.context.symbol_table.create_variable(interned_name);

        let kind = match &module_field.kind {
            parser::ModuleFieldKind::Var {
                name: _,
                type_hint,
                expr,
            } => {
                let field_type = if let Some(type_hint) = type_hint {
                    self.lower_type(type_hint)?
                } else {
                    self.context.type_table.borrow().dynamic_type()
                };

                let initializer = if let Some(expr) = expr {
                    Some(self.lower_expression(expr)?)
                } else {
                    None
                };

                TypedModuleFieldKind::Var {
                    field_type,
                    initializer,
                    mutability: crate::tast::Mutability::Mutable,
                }
            }
            parser::ModuleFieldKind::Final {
                name: _,
                type_hint,
                expr,
            } => {
                let field_type = if let Some(type_hint) = type_hint {
                    self.lower_type(type_hint)?
                } else {
                    self.context.type_table.borrow().dynamic_type()
                };

                let initializer = if let Some(expr) = expr {
                    Some(self.lower_expression(expr)?)
                } else {
                    None
                };

                TypedModuleFieldKind::Final {
                    field_type,
                    initializer,
                }
            }
            parser::ModuleFieldKind::Function(func) => {
                TypedModuleFieldKind::Function(self.lower_function_object(func)?)
            }
        };

        Ok(TypedModuleField {
            symbol_id: field_symbol,
            name: interned_name,
            kind,
            visibility: self.lower_access(&module_field.access),
            source_location: self.context.create_location_from_span(module_field.span),
        })
    }

    /// Lower a declaration
    fn lower_declaration(
        &mut self,
        declaration: &TypeDeclaration,
    ) -> LoweringResult<TypedDeclaration> {
        match declaration {
            TypeDeclaration::Class(class_decl) => self.lower_class_declaration(class_decl),
            TypeDeclaration::Interface(interface_decl) => {
                self.lower_interface_declaration(interface_decl)
            }
            TypeDeclaration::Enum(enum_decl) => self.lower_enum_declaration(enum_decl),
            TypeDeclaration::Typedef(typedef_decl) => self.lower_typedef_declaration(typedef_decl),
            TypeDeclaration::Abstract(abstract_decl) => {
                self.lower_abstract_declaration(abstract_decl)
            }
            TypeDeclaration::Conditional(conditional) => {
                // Process conditional compilation by evaluating compile-time conditions
                // This requires compile-time flag evaluation which should be done in preprocessing
                return Err(LoweringError::IncompleteImplementation {
                    feature:
                        "Conditional compilation blocks should be expanded during preprocessing"
                            .to_string(),
                    location: self.context.create_location_from_span(conditional.span),
                });
            }
        }
    }

    /// Lower a class declaration

    fn lower_class_declaration(
        &mut self,
        class_decl: &ClassDecl,
    ) -> LoweringResult<TypedDeclaration> {
        let class_name = self.context.intern_string(&class_decl.name);

        // Look up the existing symbol that was created during pre-registration
        let class_symbol = if let Some(existing_symbol) = self
            .context
            .symbol_table
            .lookup_symbol(ScopeId::first(), class_name)
        {
            existing_symbol.id
        } else {
            let new_symbol = self
                .context
                .symbol_table
                .create_class_in_scope(class_name, ScopeId::first());
            // Update qualified name (full path including class hierarchy)
            self.context.update_symbol_qualified_name(new_symbol);
            // Add class to the root scope so it can be resolved for forward references
            self.context
                .scope_tree
                .get_scope_mut(ScopeId::first())
                .expect("Root scope should exist")
                .add_symbol(new_symbol, class_name);
            new_symbol
        };

        // Enter class scope with name
        let class_scope = self.context.enter_named_scope(ScopeKind::Class, class_name);

        // Note: The class symbol remains in the parent scope, while its members are in class_scope
        // This is correct because the class name should be accessible from outside

        // Update the class symbol's scope_id to point to the class scope where its methods are registered
        // This is crucial for static extension resolution to find methods by scope lookup
        if let Some(sym) = self.context.symbol_table.get_symbol_mut(class_symbol) {
            sym.scope_id = class_scope;
        }

        // Push class onto context stack for method resolution
        self.context.class_context_stack.push(class_symbol);

        // Initialize method and field tracking for this class
        self.class_methods.insert(class_symbol, Vec::new());
        self.class_fields.insert(class_symbol, Vec::new());

        // Extract metadata flags from @:generic, @:final, @:native, etc.
        let mut symbol_flags = self.extract_metadata_flags(&class_decl.meta, class_symbol);
        // Also check for modifiers (final, extern, etc)
        for modifier in &class_decl.modifiers {
            match modifier {
                parser::haxe_ast::Modifier::Final => {
                    symbol_flags = symbol_flags.union(crate::tast::symbols::SymbolFlags::FINAL);
                }
                parser::haxe_ast::Modifier::Extern => {
                    symbol_flags = symbol_flags.union(crate::tast::symbols::SymbolFlags::EXTERN);
                }
                _ => {}
            }
        }
        // Apply flags to the class symbol
        if !symbol_flags.is_empty() {
            self.context
                .symbol_table
                .add_symbol_flags(class_symbol, symbol_flags);
        }

        // Process type parameters
        let type_params = self.lower_type_parameters(&class_decl.type_params)?;
        let mut type_param_map: HashMap<InternedString, TypeId> =
            HashMap::with_capacity(type_params.len());
        for tp in &type_params {
            let interned_name = tp.name;
            // Convert constraints to ConstraintKind for symbol table
            let constraint_kinds = tp
                .constraints
                .iter()
                .map(|_| {
                    crate::tast::type_checker::ConstraintKind::Implements {
                        interface_type: TypeId::invalid(), // Placeholder, will be resolved later
                    }
                })
                .collect();
            let symbol_id = self
                .context
                .symbol_table
                .create_type_parameter(interned_name, constraint_kinds);
            let type_id = self.context.type_table.borrow_mut().create_type_parameter(
                symbol_id,
                tp.constraints.clone(),
                tp.variance.into(),
            );
            type_param_map.insert(tp.name, type_id);
        }
        self.context.push_type_parameters(type_param_map.clone());

        // Store ordered type parameter TypeIds for generic type inference
        if !type_param_map.is_empty() {
            let ordered_tp_ids: Vec<TypeId> = type_params
                .iter()
                .filter_map(|tp| type_param_map.get(&tp.name).copied())
                .collect();
            self.class_type_params.insert(class_symbol, ordered_tp_ids);
        }

        // Process extends clause
        let extends = if let Some(extends_type) = &class_decl.extends {
            Some(self.lower_type(extends_type)?)
        } else {
            None
        };

        // Copy parent FIELDS and METHODS before processing child's members
        // This ensures:
        // 1. Field inheritance works (constructor can access parent fields)
        // 2. Method inheritance works (child methods can call parent methods)
        // 3. Method overriding works (child methods will replace parent methods during processing)
        if let Some(parent_type_id) = extends {
            self.copy_parent_fields(parent_type_id, class_symbol);
            self.copy_parent_methods(parent_type_id, class_symbol);
        }

        // Process implements clause
        let implements = class_decl
            .implements
            .iter()
            .map(|t| self.lower_type(t))
            .collect::<Result<Vec<_>, _>>()?;

        // PRE-REGISTER ALL METHODS before lowering any method bodies.
        // This is critical for intra-class method calls: when main() calls iterate(),
        // iterate must already be registered in the class scope, even if it's defined later.
        // We also pre-compute function types from signatures to enable forward references.
        for field in &class_decl.fields {
            if let ClassFieldKind::Function(func) = &field.kind {
                let method_name = self.context.intern_string(&func.name);
                let is_static = field
                    .modifiers
                    .iter()
                    .any(|m| matches!(m, parser::haxe_ast::Modifier::Static));

                // Get or create the method symbol
                let method_symbol = if let Some(existing) = self
                    .context
                    .symbol_table
                    .lookup_symbol(class_scope, method_name)
                {
                    existing.id
                } else {
                    // Create the function symbol in the class scope
                    let sym = self
                        .context
                        .symbol_table
                        .create_function_in_scope(method_name, class_scope);

                    // Add to the class scope so it can be resolved during method body lowering
                    if let Some(scope) = self.context.scope_tree.get_scope_mut(class_scope) {
                        scope.add_symbol(sym, method_name);
                    }

                    // Mark as static if applicable (needed for resolution)
                    if is_static {
                        self.context
                            .symbol_table
                            .add_symbol_flags(sym, crate::tast::symbols::SymbolFlags::STATIC);
                    }

                    sym
                };

                // Pre-compute function type from AST signature for forward reference resolution
                let param_types: Vec<TypeId> = func
                    .params
                    .iter()
                    .map(|p| {
                        if let Some(ref type_hint) = p.type_hint {
                            self.lower_type(type_hint)
                                .unwrap_or_else(|_| self.context.type_table.borrow().dynamic_type())
                        } else {
                            self.context.type_table.borrow().dynamic_type()
                        }
                    })
                    .collect();
                let return_type = if let Some(ref ret_type) = func.return_type {
                    self.lower_type(ret_type)
                        .unwrap_or_else(|_| self.context.type_table.borrow().dynamic_type())
                } else {
                    self.context.type_table.borrow().dynamic_type()
                };
                let function_type = self
                    .context
                    .type_table
                    .borrow_mut()
                    .create_function_type(param_types, return_type);
                self.context
                    .symbol_table
                    .update_symbol_type(method_symbol, function_type);

                // Add to class_methods for forward reference resolution in lower_call_expression
                if func.name != "new" {
                    if let Some(methods_list) = self.class_methods.get_mut(&class_symbol) {
                        if !methods_list.iter().any(|(name, _, _)| *name == method_name) {
                            methods_list.push((method_name, method_symbol, is_static));
                        }
                    }
                } else {
                    // Store constructor symbol for generic type inference
                    self.class_constructor_symbols
                        .insert(class_symbol, method_symbol);
                }
            }
        }

        // Process fields, methods, and constructors separately
        let mut fields = Vec::with_capacity(class_decl.fields.len());
        let mut methods = Vec::with_capacity(class_decl.fields.len()); // Initially allocate for all fields
        let mut constructors = Vec::with_capacity(2); // Most classes have 0-2 constructors

        for (field_idx, field) in class_decl.fields.iter().enumerate() {
            match &field.kind {
                ClassFieldKind::Function(func) => {
                    // Handle functions as methods or constructors
                    match self.lower_function_from_field(field, func) {
                        Ok(typed_function) => {
                            if func.name == "new" {
                                constructors.push(typed_function);
                            } else {
                                // Track method name and symbol for resolution
                                let method_name = self.context.intern_string(&func.name);
                                let method_symbol = typed_function.symbol_id;
                                if let Some(methods_list) =
                                    self.class_methods.get_mut(&class_symbol)
                                {
                                    // Check if a method with this name already exists (from parent)
                                    // If so, replace it (method overriding)
                                    if let Some(existing_idx) = methods_list
                                        .iter()
                                        .position(|(name, _, _)| *name == method_name)
                                    {
                                        // Override parent method
                                        methods_list[existing_idx] =
                                            (method_name, method_symbol, typed_function.is_static);
                                    } else {
                                        // New method, add to list
                                        methods_list.push((
                                            method_name,
                                            method_symbol,
                                            typed_function.is_static,
                                        ));
                                    }
                                }
                                methods.push(typed_function);
                            }
                        }
                        Err(e) => self.context.add_error(e),
                    }
                }
                _ => {
                    // Handle regular fields (var, final, property)
                    match self.lower_field(field) {
                        Ok(typed_field) => fields.push(typed_field),
                        Err(e) => self.context.add_error(e),
                    }
                }
            }
        }

        // Note: Parent fields and methods were already copied before processing members
        // This ensures:
        // 1. Field/method inheritance works (child can access parent members)
        // 2. Method overriding works (child methods replace parent methods during processing)

        // Process modifiers
        let modifiers = self.lower_modifiers(&class_decl.modifiers)?;

        self.context.pop_type_parameters();

        // Pop class from context stack
        self.context.class_context_stack.pop();

        self.context.exit_scope();

        // Auto-inject synthetic cdef() static method for @:cstruct classes
        if symbol_flags.is_cstruct() {
            let cdef_name = self.context.intern_string("cdef");
            // Create a symbol for the synthetic method
            let cdef_symbol = self
                .context
                .symbol_table
                .create_function_in_scope(cdef_name, class_scope);
            self.context
                .symbol_table
                .add_symbol_flags(cdef_symbol, crate::tast::symbols::SymbolFlags::STATIC);
            if let Some(scope) = self.context.scope_tree.get_scope_mut(class_scope) {
                scope.add_symbol(cdef_symbol, cdef_name);
            }
            // Set return type to String
            let string_type = self.context.type_table.borrow().string_type();
            let fn_type = self
                .context
                .type_table
                .borrow_mut()
                .create_function_type(vec![], string_type);
            self.context
                .symbol_table
                .update_symbol_type(cdef_symbol, fn_type);

            // Register in class_methods
            if let Some(methods_list) = self.class_methods.get_mut(&class_symbol) {
                methods_list.push((cdef_name, cdef_symbol, true));
            }

            // Add synthetic TypedFunction (empty body — intercepted at MIR level)
            methods.push(crate::tast::node::TypedFunction {
                symbol_id: cdef_symbol,
                name: cdef_name,
                parameters: vec![],
                return_type: string_type,
                body: vec![],
                visibility: crate::tast::symbols::Visibility::Public,
                effects: crate::tast::node::FunctionEffects {
                    can_throw: false,
                    async_kind: crate::tast::node::AsyncKind::Sync,
                    is_pure: true,
                    is_inline: true,
                    exception_types: vec![],
                    memory_effects: crate::tast::node::MemoryEffects::default(),
                    resource_effects: crate::tast::node::ResourceEffects::default(),
                },
                type_parameters: vec![],
                is_static: true,
                source_location: crate::tast::symbols::SourceLocation {
                    file_id: 0,
                    line: 0,
                    column: 0,
                    byte_offset: 0,
                },
                metadata: crate::tast::node::FunctionMetadata {
                    complexity_score: 0,
                    statement_count: 0,
                    is_recursive: false,
                    call_count: 0,
                    is_override: false,
                    overload_signatures: vec![],
                    operator_metadata: vec![],
                    is_array_access: false,
                    is_from_conversion: false,
                    is_to_conversion: false,
                    memory_annotations: vec![],
                },
            });
        }

        // Auto-inject synthetic gpuDef(), gpuSize(), gpuAlignment() for @:gpuStruct classes
        if symbol_flags.is_gpu_struct() {
            let string_type = self.context.type_table.borrow().string_type();
            let int_type = self.context.type_table.borrow().int_type();

            // Helper to create a synthetic static method with empty body
            let synthetic_names = [
                ("gpuDef", string_type),
                ("gpuSize", int_type),
                ("gpuAlignment", int_type),
            ];
            for (name_str, ret_type) in &synthetic_names {
                let method_name = self.context.intern_string(name_str);
                let method_symbol = self
                    .context
                    .symbol_table
                    .create_function_in_scope(method_name, class_scope);
                self.context
                    .symbol_table
                    .add_symbol_flags(method_symbol, crate::tast::symbols::SymbolFlags::STATIC);
                if let Some(scope) = self.context.scope_tree.get_scope_mut(class_scope) {
                    scope.add_symbol(method_symbol, method_name);
                }
                let fn_type = self
                    .context
                    .type_table
                    .borrow_mut()
                    .create_function_type(vec![], *ret_type);
                self.context
                    .symbol_table
                    .update_symbol_type(method_symbol, fn_type);

                if let Some(methods_list) = self.class_methods.get_mut(&class_symbol) {
                    methods_list.push((method_name, method_symbol, true));
                }

                methods.push(crate::tast::node::TypedFunction {
                    symbol_id: method_symbol,
                    name: method_name,
                    parameters: vec![],
                    return_type: *ret_type,
                    body: vec![],
                    visibility: crate::tast::symbols::Visibility::Public,
                    effects: crate::tast::node::FunctionEffects {
                        can_throw: false,
                        async_kind: crate::tast::node::AsyncKind::Sync,
                        is_pure: true,
                        is_inline: true,
                        exception_types: vec![],
                        memory_effects: crate::tast::node::MemoryEffects::default(),
                        resource_effects: crate::tast::node::ResourceEffects::default(),
                    },
                    type_parameters: vec![],
                    is_static: true,
                    source_location: crate::tast::symbols::SourceLocation {
                        file_id: 0,
                        line: 0,
                        column: 0,
                        byte_offset: 0,
                    },
                    metadata: crate::tast::node::FunctionMetadata {
                        complexity_score: 0,
                        statement_count: 0,
                        is_recursive: false,
                        call_count: 0,
                        is_override: false,
                        overload_signatures: vec![],
                        operator_metadata: vec![],
                        is_array_access: false,
                        is_from_conversion: false,
                        is_to_conversion: false,
                        memory_annotations: vec![],
                    },
                });
            }
        }

        // Extract memory safety annotations from metadata
        let memory_annotations = self.extract_memory_annotations(&class_decl.meta);

        // Extract derived traits from @:derive metadata
        let mut derived_traits = self.extract_derived_traits(class_decl);

        // Create typed class first (needed for validation)
        let typed_class = TypedClass {
            symbol_id: class_symbol,
            name: class_name,
            super_class: extends,
            interfaces: implements,
            fields: fields.clone(),
            methods: methods,
            constructors: constructors,
            type_parameters: type_params,
            visibility: self.lower_access(&class_decl.access),
            source_location: self.context.create_location_from_span(class_decl.span),
            memory_annotations,
            derived_traits: derived_traits.clone(),
        };

        // Validate derived traits against field types
        self.validate_derived_traits(&typed_class, &mut derived_traits, &class_decl.name);

        // Update derived_traits after validation (may have been modified)
        let mut typed_class = typed_class;
        typed_class.derived_traits = derived_traits;

        Ok(TypedDeclaration::Class(typed_class))
    }

    /// Lower an interface declaration
    fn lower_interface_declaration(
        &mut self,
        interface_decl: &InterfaceDecl,
    ) -> LoweringResult<TypedDeclaration> {
        let interface_name = self.context.intern_string(&interface_decl.name);

        // Look up the existing symbol that was created during pre-registration
        let interface_symbol = if let Some(existing_symbol) = self
            .context
            .symbol_table
            .lookup_symbol(ScopeId::first(), interface_name)
        {
            existing_symbol.id
        } else {
            let new_symbol = self
                .context
                .symbol_table
                .create_interface_in_scope(interface_name, ScopeId::first());
            // Update qualified name (full path including class hierarchy)
            self.context.update_symbol_qualified_name(new_symbol);
            // Add interface to the root scope so it can be resolved for forward references
            self.context
                .scope_tree
                .get_scope_mut(ScopeId::first())
                .expect("Root scope should exist")
                .add_symbol(new_symbol, interface_name);
            new_symbol
        };

        // Enter interface scope with name
        let interface_scope = self
            .context
            .enter_named_scope(ScopeKind::Interface, interface_name);

        // Process type parameters
        let type_params = self.lower_type_parameters(&interface_decl.type_params)?;
        let mut type_param_map: HashMap<InternedString, TypeId> =
            HashMap::with_capacity(type_params.len());
        for tp in &type_params {
            let interned_name = tp.name;
            // Convert constraints to ConstraintKind for symbol table
            let constraint_kinds = tp
                .constraints
                .iter()
                .map(|_| {
                    crate::tast::type_checker::ConstraintKind::Implements {
                        interface_type: TypeId::invalid(), // Placeholder, will be resolved later
                    }
                })
                .collect();
            let symbol_id = self
                .context
                .symbol_table
                .create_type_parameter(interned_name, constraint_kinds);
            let type_id = self.context.type_table.borrow_mut().create_type_parameter(
                symbol_id,
                tp.constraints.clone(),
                tp.variance.into(),
            );
            type_param_map.insert(tp.name, type_id);
        }
        self.context.push_type_parameters(type_param_map);

        // Process extends clause
        let extends = interface_decl
            .extends
            .iter()
            .map(|t| self.lower_type(t))
            .collect::<Result<Vec<_>, _>>()?;

        // Process fields - separate method signatures from other fields
        let mut method_signatures = Vec::with_capacity(interface_decl.fields.len());
        for field in &interface_decl.fields {
            match &field.kind {
                ClassFieldKind::Function(func) => {
                    // Interface methods are just signatures, not full implementations
                    match self.lower_function_signature(field, func) {
                        Ok(method_sig) => method_signatures.push(method_sig),
                        Err(e) => self.context.add_error(e),
                    }
                }
                _ => {
                    // Interfaces can have property signatures too
                    // Interfaces can have property signatures and constants
                    // These are handled separately in the interface specification
                }
            }
        }

        // Process modifiers
        let modifiers = self.lower_modifiers(&interface_decl.modifiers)?;

        self.context.pop_type_parameters();
        self.context.exit_scope();

        let typed_interface = TypedInterface {
            symbol_id: interface_symbol,
            name: interface_name,
            extends,
            methods: method_signatures,
            type_parameters: type_params,
            visibility: self.lower_access(&interface_decl.access),
            source_location: self.context.create_location_from_span(interface_decl.span),
        };

        Ok(TypedDeclaration::Interface(typed_interface))
    }

    /// Lower an enum declaration
    /// Public wrapper for lower_enum_declaration, used when loading from BLADE cache
    pub fn lower_enum_declaration_public(
        &mut self,
        enum_decl: &EnumDecl,
    ) -> LoweringResult<TypedDeclaration> {
        self.lower_enum_declaration(enum_decl)
    }

    fn lower_enum_declaration(&mut self, enum_decl: &EnumDecl) -> LoweringResult<TypedDeclaration> {
        let enum_name = self.context.intern_string(&enum_decl.name);

        // Look up existing symbol from pre-registration, or create a new one
        let enum_symbol = if let Some(existing_symbol) = self
            .context
            .symbol_table
            .lookup_symbol(ScopeId::first(), enum_name)
        {
            existing_symbol.id
        } else {
            let new_symbol = self
                .context
                .symbol_table
                .create_enum_in_scope(enum_name, ScopeId::first());
            self.context.update_symbol_qualified_name(new_symbol);
            self.context
                .scope_tree
                .get_scope_mut(ScopeId::first())
                .expect("Root scope should exist")
                .add_symbol(new_symbol, enum_name);
            new_symbol
        };

        // Enter enum scope with name
        let enum_scope = self.context.enter_named_scope(ScopeKind::Enum, enum_name);

        // Process type parameters
        let type_params = self.lower_type_parameters(&enum_decl.type_params)?;
        let mut type_param_map: HashMap<InternedString, TypeId> =
            HashMap::with_capacity(type_params.len());
        let mut type_param_ids = Vec::new();
        for tp in &type_params {
            let interned_name = tp.name;
            // Convert constraints to ConstraintKind for symbol table
            let constraint_kinds = tp
                .constraints
                .iter()
                .map(|_| {
                    crate::tast::type_checker::ConstraintKind::Implements {
                        interface_type: TypeId::invalid(), // Placeholder, will be resolved later
                    }
                })
                .collect();
            let symbol_id = self
                .context
                .symbol_table
                .create_type_parameter(interned_name, constraint_kinds);
            let type_id = self.context.type_table.borrow_mut().create_type_parameter(
                symbol_id,
                tp.constraints.clone(),
                tp.variance.into(),
            );
            type_param_map.insert(tp.name, type_id);
            type_param_ids.push(type_id);
        }
        self.context.push_type_parameters(type_param_map);

        // Create the enum type
        let enum_type_id = self
            .context
            .type_table
            .borrow_mut()
            .create_enum_type(enum_symbol, type_param_ids);

        // Update the enum symbol with its type
        self.context
            .symbol_table
            .update_symbol_type(enum_symbol, enum_type_id);

        // Process variants
        let mut variants = Vec::with_capacity(enum_decl.constructors.len());
        for variant in &enum_decl.constructors {
            variants.push(self.lower_enum_variant(variant, enum_type_id, enum_symbol)?);
        }

        self.context.pop_type_parameters();
        self.context.exit_scope();

        let typed_enum = TypedEnum {
            symbol_id: enum_symbol,
            name: enum_name,
            variants,
            type_parameters: type_params,
            visibility: self.lower_access(&enum_decl.access),
            source_location: self.context.create_location_from_span(enum_decl.span),
        };

        Ok(TypedDeclaration::Enum(typed_enum))
    }

    /// Lower a typedef declaration
    fn lower_typedef_declaration(
        &mut self,
        typedef_decl: &TypedefDecl,
    ) -> LoweringResult<TypedDeclaration> {
        let typedef_name = self.context.intern_string(&typedef_decl.name);

        // Look up existing symbol or create a new one
        let typedef_symbol = if let Some(existing_symbol) = self
            .context
            .symbol_table
            .lookup_symbol(ScopeId::first(), typedef_name)
        {
            existing_symbol.id
        } else {
            let new_symbol = self
                .context
                .symbol_table
                .create_type_alias_in_scope(typedef_name, ScopeId::first());
            // Update qualified name (full path including package/module)
            self.context.update_symbol_qualified_name(new_symbol);
            // Add typedef to the root scope so it can be resolved
            self.context
                .scope_tree
                .get_scope_mut(ScopeId::first())
                .expect("Root scope should exist")
                .add_symbol(new_symbol, typedef_name);
            new_symbol
        };

        // Process type parameters FIRST and push them onto the stack
        let type_params = self.lower_type_parameters(&typedef_decl.type_params)?;

        // Build type parameter map for the stack
        let mut type_param_map = HashMap::new();
        for type_param in &type_params {
            // Type parameter already has a symbol_id from lower_type_parameters
            // Create a TypeId for this parameter
            let variance = match type_param.variance {
                TypeVariance::Covariant => Variance::Covariant,
                TypeVariance::Contravariant => Variance::Contravariant,
                TypeVariance::Invariant => Variance::Invariant,
            };
            let type_var = self.context.type_table.borrow_mut().create_type_parameter(
                type_param.symbol_id,
                type_param.constraints.clone(),
                variance,
            );
            type_param_map.insert(type_param.name, type_var);
        }

        // Push type parameters onto stack so they're available when lowering the typedef body
        self.context.push_type_parameters(type_param_map);

        // Now process target type (can reference type parameters)
        let target_type = self.lower_type(&typedef_decl.type_def)?;

        // Pop type parameters from stack
        self.context.pop_type_parameters();

        // Create the TypeAlias type in the type table and set it on the symbol
        let type_arg_ids: Vec<TypeId> = type_params
            .iter()
            .map(|tp| {
                self.context.type_table.borrow_mut().create_type_parameter(
                    tp.symbol_id,
                    tp.constraints.clone(),
                    tp.variance.into(),
                )
            })
            .collect();

        let typedef_type = self.context.type_table.borrow_mut().create_type(
            crate::tast::core::TypeKind::TypeAlias {
                symbol_id: typedef_symbol,
                target_type,
                type_args: type_arg_ids,
            },
        );

        // Set the type on the symbol so it can be resolved later
        self.context
            .symbol_table
            .update_symbol_type(typedef_symbol, typedef_type);

        let typed_typedef = TypedTypeAlias {
            symbol_id: typedef_symbol,
            name: typedef_name,
            target_type,
            type_parameters: type_params,
            visibility: self.lower_access(&typedef_decl.access),
            source_location: self.context.create_location_from_span(typedef_decl.span),
        };

        Ok(TypedDeclaration::TypeAlias(typed_typedef))
    }

    /// Lower an abstract declaration
    fn lower_abstract_declaration(
        &mut self,
        abstract_decl: &AbstractDecl,
    ) -> LoweringResult<TypedDeclaration> {
        let abstract_name = self.context.intern_string(&abstract_decl.name);

        let abstract_symbol = self
            .context
            .symbol_table
            .create_abstract_in_scope(abstract_name, ScopeId::first());

        // Update qualified name (full path including class hierarchy)
        self.context.update_symbol_qualified_name(abstract_symbol);

        // Extract @:native metadata for abstracts
        let mut abstract_meta_flags =
            self.extract_metadata_flags(&abstract_decl.meta, abstract_symbol);
        // Also check for modifiers (extern, final, etc)
        for modifier in &abstract_decl.modifiers {
            match modifier {
                parser::haxe_ast::Modifier::Extern => {
                    abstract_meta_flags =
                        abstract_meta_flags.union(crate::tast::symbols::SymbolFlags::EXTERN);
                }
                parser::haxe_ast::Modifier::Final => {
                    abstract_meta_flags =
                        abstract_meta_flags.union(crate::tast::symbols::SymbolFlags::FINAL);
                }
                _ => {}
            }
        }
        if let Some(sym) = self.context.symbol_table.get_symbol_mut(abstract_symbol) {
            sym.flags = sym.flags.union(abstract_meta_flags);
        }

        // Extract @:forward metadata params (method/field names to forward to underlying type)
        let forward_fields: Vec<InternedString> = abstract_decl
            .meta
            .iter()
            .find(|m| {
                let name = m.name.strip_prefix(':').unwrap_or(&m.name);
                name == "forward"
            })
            .map(|m| {
                m.params
                    .iter()
                    .filter_map(|p| {
                        if let parser::haxe_ast::ExprKind::Ident(name) = &p.kind {
                            Some(self.context.intern_string(name))
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();

        // Add abstract to the root scope so it can be resolved for forward references
        self.context
            .scope_tree
            .get_scope_mut(ScopeId::first())
            .expect("Root scope should exist")
            .add_symbol(abstract_symbol, abstract_name);

        // Enter abstract scope with name
        let abstract_scope = self
            .context
            .enter_named_scope(ScopeKind::Class, abstract_name);

        // Process type parameters
        let type_params = self.lower_type_parameters(&abstract_decl.type_params)?;
        let mut type_param_map: HashMap<InternedString, TypeId> =
            HashMap::with_capacity(type_params.len());
        for tp in &type_params {
            let interned_name = tp.name;
            // Convert constraints to ConstraintKind for symbol table
            let constraint_kinds = tp
                .constraints
                .iter()
                .map(|_| {
                    crate::tast::type_checker::ConstraintKind::Implements {
                        interface_type: TypeId::invalid(), // Placeholder, will be resolved later
                    }
                })
                .collect();
            let symbol_id = self
                .context
                .symbol_table
                .create_type_parameter(interned_name, constraint_kinds);
            let type_id = self.context.type_table.borrow_mut().create_type_parameter(
                symbol_id,
                tp.constraints.clone(),
                tp.variance.into(),
            );
            type_param_map.insert(tp.name, type_id);
        }
        self.context.push_type_parameters(type_param_map);

        // Process underlying type
        let underlying_type = match &abstract_decl.underlying {
            Some(underlying) => Some(self.lower_type(underlying)?),
            None => {
                // Core types like @:coreType abstract Void {} don't have underlying types
                // Check if this is a core type
                let is_core_type = abstract_decl.meta.iter().any(|m| m.name == "coreType");
                if is_core_type {
                    None
                } else {
                    return Err(LoweringError::IncompleteImplementation {
                        feature: format!(
                            "Abstract type '{}' missing underlying type",
                            abstract_decl.name
                        ),
                        location: self.context.create_location_from_span(abstract_decl.span),
                    });
                }
            }
        };

        // Register the abstract type in the type table with the underlying type
        // so that resolve_this_type can return the underlying type for `this`
        {
            let abstract_type_id = self.context.type_table.borrow_mut().create_abstract_type(
                abstract_symbol,
                underlying_type,
                Vec::new(),
            );
            self.context
                .symbol_table
                .update_symbol_type(abstract_symbol, abstract_type_id);
        }

        // Process from/to types
        let from_types = abstract_decl
            .from
            .iter()
            .map(|t| self.lower_type(t))
            .collect::<Result<Vec<_>, _>>()?;
        let to_types = abstract_decl
            .to
            .iter()
            .map(|t| self.lower_type(t))
            .collect::<Result<Vec<_>, _>>()?;

        // Initialize class_fields for this abstract so field tracking works (needed for enum abstract)
        self.class_fields.insert(abstract_symbol, Vec::new());

        // Push abstract onto class context stack so `this` resolves correctly in method bodies
        self.context.class_context_stack.push(abstract_symbol);

        // Process fields - separate fields, methods, and constructors
        let mut fields = Vec::with_capacity(abstract_decl.fields.len());
        let mut methods = Vec::with_capacity(abstract_decl.fields.len());
        let mut constructors = Vec::with_capacity(2); // Most abstracts have 0-2 constructors

        for field in &abstract_decl.fields {
            match &field.kind {
                ClassFieldKind::Function(func) => {
                    if func.name == "new" {
                        // Constructor
                        match self.lower_function_from_field(field, func) {
                            Ok(typed_function) => {
                                constructors.push(typed_function);
                            }
                            Err(e) => self.context.add_error(e),
                        }
                    } else {
                        // Regular method
                        match self.lower_function_from_field(field, func) {
                            Ok(typed_function) => {
                                methods.push(typed_function);
                            }
                            Err(e) => self.context.add_error(e),
                        }
                    }
                }
                _ => {
                    // Handle regular fields (var, final, property)
                    match self.lower_field(field) {
                        Ok(mut typed_field) => {
                            // For enum abstracts, all var fields are implicitly static
                            if abstract_decl.is_enum_abstract && !typed_field.is_static {
                                typed_field.is_static = true;
                                // Also update class_fields tracking
                                if let Some(field_list) =
                                    self.class_fields.get_mut(&abstract_symbol)
                                {
                                    if let Some(entry) = field_list
                                        .iter_mut()
                                        .find(|(_, sym, _)| *sym == typed_field.symbol_id)
                                    {
                                        entry.2 = true;
                                    }
                                }
                            }
                            fields.push(typed_field);
                        }
                        Err(e) => self.context.add_error(e),
                    }
                }
            }
        }

        // Pop abstract from class context stack
        self.context.class_context_stack.pop();

        self.context.pop_type_parameters();
        self.context.exit_scope();

        let typed_abstract = crate::tast::node::TypedAbstract {
            symbol_id: abstract_symbol,
            name: abstract_name,
            underlying_type,
            type_parameters: type_params,
            fields,
            methods,
            constructors,
            from_types,
            to_types,
            forward_fields,
            is_enum_abstract: abstract_decl.is_enum_abstract,
            visibility: self.lower_access(&abstract_decl.access),
            source_location: self.context.create_location_from_span(abstract_decl.span),
        };

        Ok(TypedDeclaration::Abstract(typed_abstract))
    }

    /// Lower a function declaration (not used anymore - functions are in module fields)
    fn lower_function_declaration(
        &mut self,
        function_decl: &Function,
    ) -> LoweringResult<TypedDeclaration> {
        let function_name = self.context.intern_string(&function_decl.name);
        let function_symbol = self.context.symbol_table.create_function(function_name);

        // Enter function scope
        let function_scope = self.context.enter_scope(ScopeKind::Function);

        // Process type parameters
        let type_params = self.lower_type_parameters(&function_decl.type_params)?;
        let type_param_map: HashMap<InternedString, TypeId> = type_params
            .iter()
            .map(|tp| (tp.name, TypeId::invalid()))
            .collect();
        self.context.push_type_parameters(type_param_map);

        // Process parameters
        let mut parameters = Vec::with_capacity(function_decl.params.len());
        for param in &function_decl.params {
            parameters.push(self.lower_parameter(param)?);
        }

        // Process return type
        let return_type = if let Some(ret_type) = &function_decl.return_type {
            self.lower_type(ret_type)?
        } else {
            self.context.type_table.borrow().void_type()
        };

        // Process body
        let body = if let Some(body_expr) = &function_decl.body {
            // Convert expression to statement
            vec![self.lower_expression_as_statement(body_expr)?]
        } else {
            Vec::new()
        };

        // Process modifiers - skip for now

        self.context.pop_type_parameters();
        self.context.exit_scope();

        let typed_function = TypedFunction {
            symbol_id: function_symbol,
            name: function_name,
            parameters,
            return_type,
            body,
            visibility: Visibility::Public,
            effects: crate::tast::node::FunctionEffects::default(),
            type_parameters: type_params,
            is_static: false, // Top-level functions are not static
            source_location: self.context.create_location_from_span(function_decl.span),
            metadata: FunctionMetadata::default(),
        };

        Ok(TypedDeclaration::Function(typed_function))
    }

    /// Lower type parameters
    fn lower_type_parameters(
        &mut self,
        type_params: &[TypeParam],
    ) -> LoweringResult<Vec<TypedTypeParameter>> {
        let mut result = Vec::new();
        for type_param in type_params {
            result.push(self.lower_type_parameter(type_param)?);
        }
        Ok(result)
    }

    /// Lower a single type parameter
    fn lower_type_parameter(
        &mut self,
        type_param: &TypeParam,
    ) -> LoweringResult<TypedTypeParameter> {
        let name = self.context.intern_string(&type_param.name);

        // Process constraints - but handle them specially if they reference type parameters
        let mut constraints = Vec::new();
        let mut deferred_constraints = Vec::new();

        for constraint in &type_param.constraints {
            // Check if this constraint might reference type parameters that aren't defined yet
            if self.type_might_reference_undefined_params(constraint) {
                // Create a placeholder for now
                let placeholder_type = self.context.type_table.borrow_mut().create_type(
                    crate::tast::core::TypeKind::Placeholder {
                        name: self.context.intern_string("<deferred_constraint>"),
                    },
                );
                constraints.push(placeholder_type);
                deferred_constraints.push((constraint.clone(), placeholder_type));
            } else {
                // Safe to lower now
                constraints.push(self.lower_type(constraint)?);
            }
        }

        // Store deferred constraints for later resolution
        for (constraint_type, placeholder) in deferred_constraints {
            if let Type::Path { path, params, .. } = &constraint_type {
                let type_name = if path.package.is_empty() {
                    path.name.clone()
                } else {
                    format!("{}.{}", path.package.join("."), path.name)
                };

                self.resolution_state
                    .deferred_resolutions
                    .push(DeferredTypeResolution {
                        type_name,
                        location: self.context.create_location(),
                        type_params: params.iter().map(|p| format!("{:?}", p)).collect(),
                        target_type_id: placeholder,
                    });
            }
        }

        // Convert TypeId constraints to ConstraintKind for symbol table
        let constraint_kinds: Vec<super::type_checker::ConstraintKind> = constraints
            .iter()
            .map(|&type_id| super::type_checker::ConstraintKind::Implements {
                interface_type: type_id,
            })
            .collect();

        // Create type parameter symbol with proper type
        let symbol_id = self
            .context
            .symbol_table
            .create_type_parameter(name, constraint_kinds);
        let param_type_id = self.context.type_table.borrow_mut().create_type_parameter(
            symbol_id,
            constraints.clone(),
            Variance::Invariant,
        );

        Ok(TypedTypeParameter {
            symbol_id,
            name,
            constraints,
            variance: TypeVariance::Invariant, // Default variance
            source_location: self.context.create_location(), // TODO: Get span from type_param
        })
    }

    /// Convert parser PropertyAccess to TAST PropertyAccessor
    ///
    /// For "get" or "set", we derive the method name as "get_fieldname" or "set_fieldname"
    /// For custom names, we use the name directly
    ///
    /// The method name is stored as InternedString and resolved to SymbolId during MIR lowering
    fn convert_property_accessor(
        &mut self,
        access: &parser::PropertyAccess,
        field_name: &str,
        is_getter: bool,
    ) -> crate::tast::PropertyAccessor {
        match access {
            parser::PropertyAccess::Default => crate::tast::PropertyAccessor::Default,
            parser::PropertyAccess::Null => crate::tast::PropertyAccessor::Null,
            parser::PropertyAccess::Never => crate::tast::PropertyAccessor::Never,
            parser::PropertyAccess::Dynamic => crate::tast::PropertyAccessor::Dynamic,
            parser::PropertyAccess::Custom(method_name) => {
                // If the custom name is just "get" or "set", derive the full method name
                let full_method_name = if method_name == "get" || method_name == "set" {
                    format!("{}_{}", method_name, field_name)
                } else {
                    method_name.clone()
                };

                // Intern the method name for later resolution during MIR lowering
                let interned_name = self.context.intern_string(&full_method_name);
                crate::tast::PropertyAccessor::Method(interned_name)
            }
        }
    }

    /// Lower a field
    fn lower_field(&mut self, field: &ClassField) -> LoweringResult<TypedField> {
        let (field_name, field_type, initializer, mutability, is_static, property_access) =
            match &field.kind {
                ClassFieldKind::Var {
                    name,
                    type_hint,
                    expr,
                } => {
                    // Lower initializer first so we can infer type from it
                    let initializer = if let Some(expr) = expr {
                        Some(self.lower_expression(expr)?)
                    } else {
                        None
                    };

                    let field_type = if let Some(type_hint) = type_hint {
                        self.lower_type(type_hint)?
                    } else if let Some(ref init_expr) = initializer {
                        // Infer type from initializer expression
                        init_expr.expr_type
                    } else {
                        self.context.type_table.borrow().dynamic_type()
                    };

                    let is_static = field
                        .modifiers
                        .iter()
                        .any(|m| matches!(m, parser::Modifier::Static));

                    (
                        name.clone(),
                        field_type,
                        initializer,
                        crate::tast::Mutability::Mutable,
                        is_static,
                        None, // No property access for regular var fields
                    )
                }
                ClassFieldKind::Final {
                    name,
                    type_hint,
                    expr,
                } => {
                    // Lower initializer first so we can infer type from it
                    let initializer = if let Some(expr) = expr {
                        Some(self.lower_expression(expr)?)
                    } else {
                        None
                    };

                    let field_type = if let Some(type_hint) = type_hint {
                        self.lower_type(type_hint)?
                    } else if let Some(ref init_expr) = initializer {
                        // Infer type from initializer expression
                        init_expr.expr_type
                    } else {
                        self.context.type_table.borrow().dynamic_type()
                    };

                    let is_static = field
                        .modifiers
                        .iter()
                        .any(|m| matches!(m, parser::Modifier::Static));

                    (
                        name.clone(),
                        field_type,
                        initializer,
                        crate::tast::Mutability::Immutable,
                        is_static,
                        None, // No property access for final fields
                    )
                }
                ClassFieldKind::Property {
                    name,
                    type_hint,
                    getter,
                    setter,
                } => {
                    // Handle property with getter/setter
                    let field_type = if let Some(type_hint) = type_hint {
                        self.lower_type(type_hint)?
                    } else {
                        self.context.type_table.borrow().dynamic_type()
                    };
                    let is_static = field
                        .modifiers
                        .iter()
                        .any(|m| matches!(m, parser::Modifier::Static));

                    // Properties are generally mutable unless they only have getters
                    let mutability = match (getter, setter) {
                        (_, parser::PropertyAccess::Never) => crate::tast::Mutability::Immutable,
                        (_, parser::PropertyAccess::Null) => crate::tast::Mutability::Immutable,
                        _ => crate::tast::Mutability::Mutable,
                    };

                    // Convert parser PropertyAccess to TAST PropertyAccessor
                    // TODO: Resolve method names to SymbolIds in a second pass after all methods are lowered
                    let getter_accessor = self.convert_property_accessor(getter, name, true);
                    let setter_accessor = self.convert_property_accessor(setter, name, false);

                    let property_info = Some(crate::tast::PropertyAccessInfo {
                        getter: getter_accessor,
                        setter: setter_accessor,
                    });

                    (
                        name.clone(),
                        field_type,
                        None,
                        mutability,
                        is_static,
                        property_info,
                    )
                }
                ClassFieldKind::Function(func) => {
                    // Functions should be handled separately as methods, not fields
                    // Return placeholder for now
                    let field_type = self.context.type_table.borrow().dynamic_type();
                    let is_static = field
                        .modifiers
                        .iter()
                        .any(|m| matches!(m, parser::Modifier::Static));

                    (
                        func.name.clone(),
                        field_type,
                        None,
                        crate::tast::Mutability::Immutable,
                        is_static,
                        None, // No property access for function fields
                    )
                }
            };

        let interned_field_name = self.context.intern_string(&field_name);
        let field_symbol = self
            .context
            .symbol_table
            .create_variable(interned_field_name);

        // Update the field symbol with its type
        self.context
            .symbol_table
            .update_symbol_type(field_symbol, field_type);

        // Add field symbol to current class scope for resolution
        if let Some(scope) = self
            .context
            .scope_tree
            .get_scope_mut(self.context.current_scope)
        {
            scope.add_symbol(field_symbol, interned_field_name);
        }

        // Track field in the current class for implicit this resolution
        if let Some(class_symbol) = self.context.class_context_stack.last() {
            if let Some(field_list) = self.class_fields.get_mut(class_symbol) {
                field_list.push((interned_field_name, field_symbol, is_static));
            }
        }

        // Process modifiers and access separately
        let modifier_info = self.lower_modifiers(&field.modifiers)?;
        let visibility = self.lower_access(&field.access);

        Ok(TypedField {
            symbol_id: field_symbol,
            name: interned_field_name,
            field_type,
            initializer,
            mutability,
            visibility, // Use visibility from access keyword (public/private), not from modifiers
            is_static: modifier_info.is_static,
            property_access,
            source_location: self.context.create_location_from_span(field.span),
        })
    }

    /// Lower an enum variant
    fn lower_enum_variant(
        &mut self,
        variant: &EnumConstructor,
        enum_type_id: TypeId,
        enum_symbol: SymbolId,
    ) -> LoweringResult<TypedEnumVariant> {
        let variant_name = self.context.intern_string(&variant.name);
        // Reuse pre-registered variant symbol if it exists, otherwise create a new one
        let variant_symbol = if let Some(existing) = self
            .context
            .symbol_table
            .lookup_symbol(ScopeId::first(), variant_name)
        {
            existing.id
        } else {
            self.context.symbol_table.create_enum_variant_in_scope(
                variant_name,
                ScopeId::first(),
                enum_symbol,
            )
        };

        // Process parameters first to get their types
        let mut parameters = Vec::new();
        let mut param_types = Vec::new();
        for param in &variant.params {
            let typed_param = self.lower_parameter(param)?;
            param_types.push(typed_param.param_type);
            parameters.push(typed_param);
        }

        // For enum constructors, we store the generic constructor type
        // The actual type will be instantiated when the constructor is used
        let constructor_type = if param_types.is_empty() {
            // No parameters: constructor will return the enum type directly
            enum_type_id
        } else {
            // Has parameters: create a function type that preserves generics
            // This will be a generic function if the enum is generic
            self.context
                .type_table
                .borrow_mut()
                .create_function_type(param_types, enum_type_id)
        };

        // Update the symbol with the proper type
        self.context
            .symbol_table
            .update_symbol_type(variant_symbol, constructor_type);

        Ok(TypedEnumVariant {
            name: variant_name,
            parameters,
            source_location: self.context.create_location(),
        })
    }

    /// Lower a function object
    fn lower_function_object(&mut self, func: &Function) -> LoweringResult<TypedFunction> {
        let function_name = self.context.intern_string(&func.name);
        let function_symbol = self.context.symbol_table.create_function(function_name);

        // Enter function scope
        let function_scope = self.context.enter_scope(ScopeKind::Function);

        // Process type parameters
        let type_params = self.lower_type_parameters(&func.type_params)?;
        let type_param_map: HashMap<InternedString, TypeId> = type_params
            .iter()
            .map(|tp| (tp.name, TypeId::invalid()))
            .collect();
        self.context.push_type_parameters(type_param_map);

        // Process parameters
        let mut parameters = Vec::new();
        for param in &func.params {
            parameters.push(self.lower_parameter(param)?);
        }

        // Process return type
        let return_type = if let Some(ret_type) = &func.return_type {
            self.lower_type(ret_type)?
        } else {
            self.context.type_table.borrow().void_type()
        };

        // Process body
        let body = if let Some(body_expr) = &func.body {
            vec![self.lower_expression_as_statement(body_expr)?]
        } else {
            Vec::new()
        };

        self.context.pop_type_parameters();
        self.context.exit_scope();

        Ok(TypedFunction {
            symbol_id: function_symbol,
            name: function_name,
            parameters,
            return_type,
            body,
            visibility: Visibility::Public,
            effects: crate::tast::node::FunctionEffects::default(),
            type_parameters: Vec::new(), // TODO: Convert type parameters
            is_static: false,            // Constructors are not static
            source_location: self.context.create_location(),
            metadata: FunctionMetadata::default(),
        })
    }

    /// Infer generic type arguments from constructor argument types.
    /// When `new Container(42)` is written without explicit `<Int>`, this matches
    /// constructor param types (TypeParameter) against argument types to infer type args.
    fn infer_type_args_from_constructor(
        &self,
        class_type_id: TypeId,
        args: &[TypedExpression],
    ) -> Option<TypeId> {
        // Get class symbol
        let class_symbol = {
            let tt = self.context.type_table.borrow();
            let ti = tt.get(class_type_id)?;
            match &ti.kind {
                crate::tast::core::TypeKind::Class { symbol_id, .. } => *symbol_id,
                _ => return None,
            }
        };

        // Check if class has type parameters
        let type_param_ids = self.class_type_params.get(&class_symbol)?;
        if type_param_ids.is_empty() {
            return None;
        }

        // Get constructor symbol and its function type
        let ctor_symbol = self.class_constructor_symbols.get(&class_symbol)?;
        let ctor_type_id = self.context.symbol_table.get_symbol(*ctor_symbol)?.type_id;
        let param_type_ids = {
            let tt = self.context.type_table.borrow();
            let ti = tt.get(ctor_type_id)?;
            match &ti.kind {
                crate::tast::core::TypeKind::Function { params, .. } => params.clone(),
                _ => return None,
            }
        };

        // Match TypeParameter params against argument types
        let mut tp_to_concrete: HashMap<TypeId, TypeId> = HashMap::new();
        {
            let tt = self.context.type_table.borrow();
            for (i, param_ty) in param_type_ids.iter().enumerate() {
                if i >= args.len() {
                    break;
                }
                if let Some(param_info) = tt.get(*param_ty) {
                    if matches!(
                        param_info.kind,
                        crate::tast::core::TypeKind::TypeParameter { .. }
                    ) {
                        tp_to_concrete.insert(*param_ty, args[i].expr_type);
                    }
                }
            }
        }

        if tp_to_concrete.is_empty() {
            return None;
        }

        // Build ordered type_args matching the class's type parameter order
        let type_args: Vec<TypeId> = type_param_ids
            .iter()
            .map(|tp_id| {
                tp_to_concrete
                    .get(tp_id)
                    .copied()
                    .unwrap_or_else(|| self.context.type_table.borrow().dynamic_type())
            })
            .collect();

        Some(
            self.context
                .type_table
                .borrow_mut()
                .create_class_type(class_symbol, type_args),
        )
    }

    /// Lower a function from a class field (includes field metadata)

    fn lower_function_from_field(
        &mut self,
        field: &ClassField,
        func: &Function,
    ) -> LoweringResult<TypedFunction> {
        let function_name = self.context.intern_string(&func.name);

        // Get function symbol - may have been pre-registered during class declaration
        // This ensures the method is associated with its class
        let current_class = self.context.class_context_stack.last().copied();

        // Use the current scope as the class scope since we're inside the class
        // The class symbol itself is in the parent scope, but methods are in the class scope
        let class_scope = if current_class.is_some() {
            self.context.current_scope
        } else {
            ScopeId::first() // Fallback to root scope
        };

        // Look up the pre-registered function symbol, or create a new one if not found
        // (constructors named "new" are not pre-registered since they're handled specially)
        let function_symbol = if let Some(existing) = self
            .context
            .symbol_table
            .lookup_symbol(class_scope, function_name)
        {
            existing.id
        } else {
            // Create the function symbol in the class scope (e.g., for constructors)
            self.context
                .symbol_table
                .create_function_in_scope(function_name, class_scope)
        };

        // Update qualified name (full path including class hierarchy)
        self.context.update_symbol_qualified_name(function_symbol);

        // DEBUG: Check if qualified name was set correctly
        if let Some(sym) = self.context.symbol_table.get_symbol(function_symbol) {
            let qname = sym
                .qualified_name
                .and_then(|qn| self.context.string_interner.get(qn))
                .unwrap_or("<none>");
        }

        // Also track this method in our class_fields for field resolution
        if let Some(class_symbol) = current_class {
            if let Some(fields_list) = self.class_fields.get_mut(&class_symbol) {
                // Check if field has static modifier
                let is_static = field
                    .modifiers
                    .iter()
                    .any(|m| matches!(m, Modifier::Static));
                fields_list.push((function_name, function_symbol, is_static));
            }
        }

        // Process @:native metadata on method fields (sets native_name on the method symbol)
        for meta in &field.meta {
            let name = meta.name.strip_prefix(':').unwrap_or(&meta.name);
            if name == "native" {
                if let Some(first_param) = meta.params.first() {
                    if let parser::haxe_ast::ExprKind::String(native_str) = &first_param.kind {
                        let native_interned = self.context.string_interner.intern(native_str);
                        if let Some(sym) = self.context.symbol_table.get_symbol_mut(function_symbol)
                        {
                            sym.native_name = Some(native_interned);
                        }
                    }
                }
            } else if matches!(name, "frameworks" | "cInclude" | "cSource" | "clib") {
                if let Some(first_param) = meta.params.first() {
                    if let parser::haxe_ast::ExprKind::Array(elements) = &first_param.kind {
                        let mut names = Vec::new();
                        for elem in elements {
                            if let parser::haxe_ast::ExprKind::String(s) = &elem.kind {
                                names.push(self.context.string_interner.intern(s));
                            }
                        }
                        if !names.is_empty() {
                            if let Some(sym) =
                                self.context.symbol_table.get_symbol_mut(function_symbol)
                            {
                                match name {
                                    "frameworks" => sym.frameworks = Some(names),
                                    "cInclude" => sym.c_includes = Some(names),
                                    "cSource" => sym.c_sources = Some(names),
                                    "clib" => sym.c_libs = Some(names),
                                    _ => {}
                                }
                            }
                        }
                    }
                }
            }
        }

        // Enter function scope
        let function_scope = self.context.enter_scope(ScopeKind::Function);

        // Process type parameters
        let type_params = self.lower_type_parameters(&func.type_params)?;
        let mut type_param_map: HashMap<InternedString, TypeId> =
            HashMap::with_capacity(type_params.len());
        for tp in &type_params {
            let interned_name = tp.name;
            // Convert constraints to ConstraintKind for symbol table
            let constraint_kinds = tp
                .constraints
                .iter()
                .map(|_| {
                    crate::tast::type_checker::ConstraintKind::Implements {
                        interface_type: TypeId::invalid(), // Placeholder, will be resolved later
                    }
                })
                .collect();
            let symbol_id = self
                .context
                .symbol_table
                .create_type_parameter(interned_name, constraint_kinds);
            let type_id = self.context.type_table.borrow_mut().create_type_parameter(
                symbol_id,
                tp.constraints.clone(),
                tp.variance.into(),
            );
            type_param_map.insert(tp.name, type_id);
        }
        self.context.push_type_parameters(type_param_map);

        // Process parameters
        let mut parameters = Vec::new();
        for param in &func.params {
            parameters.push(self.lower_parameter(param)?);
        }

        // Check if this is a static method BEFORE lowering the body, so that
        // the implicit `this` logic in identifier resolution knows whether
        // `this` is available.
        let is_static_method = field
            .modifiers
            .iter()
            .any(|m| matches!(m, parser::Modifier::Static));
        let prev_static = self.in_static_method;
        self.in_static_method = is_static_method;

        // Process body first (we need it to infer return type if not specified)
        let (body, body_statements_for_inference) = if let Some(body_expr) = &func.body {
            let typed_expr = self.lower_expression(body_expr)?;
            // Extract statements for return type inference
            let stmts_for_inference = match &typed_expr.kind {
                TypedExpressionKind::Block { statements, .. } => statements.clone(),
                _ => vec![],
            };
            let body = vec![TypedStatement::Expression {
                expression: typed_expr,
                source_location: self.context.span_to_location(&body_expr.span),
            }];
            (body, stmts_for_inference)
        } else {
            (Vec::new(), Vec::new())
        };

        // Restore static method flag
        self.in_static_method = prev_static;

        // Process return type - if not specified, infer from body
        let return_type = if let Some(ret_type) = &func.return_type {
            self.lower_type(ret_type)?
        } else {
            // Try to infer return type from return statements in the body.
            // Use the unwrapped block statements for inference since the body
            // is now wrapped in an Expression(Block(...)) for consistency.
            if body_statements_for_inference.is_empty() {
                self.infer_return_type_from_body(&body)
            } else {
                self.infer_return_type_from_body(&body_statements_for_inference)
            }
        };

        // Create function type and update symbol
        let param_types: Vec<TypeId> = parameters.iter().map(|p| p.param_type).collect();
        let function_type = self
            .context
            .type_table
            .borrow_mut()
            .create_function_type(param_types, return_type);

        // Update the symbol with its type
        self.context
            .symbol_table
            .update_symbol_type(function_symbol, function_type);

        // Process field modifiers and access
        let modifier_info = self.lower_modifiers(&field.modifiers)?;
        let visibility = self.lower_access(&field.access);

        // Process @:overload metadata
        let overload_signatures = self.process_overload_metadata(&field.meta)?;

        // Process @:op metadata for operator overloading
        let operator_metadata = self.process_operator_metadata(&field.meta)?;

        // Check for @:arrayAccess metadata
        let is_array_access = self.has_array_access_metadata(&field.meta);

        // Check for @:from / @:to metadata (abstract implicit conversions)
        let is_from_conversion = field.meta.iter().any(|m| m.name == "from");
        let is_to_conversion = field.meta.iter().any(|m| m.name == "to");

        self.context.pop_type_parameters();
        self.context.exit_scope();

        let body_len = body.len();

        Ok(TypedFunction {
            symbol_id: function_symbol,
            name: function_name,
            parameters,
            return_type,
            body,
            visibility,
            effects: crate::tast::node::FunctionEffects {
                can_throw: self.analyze_can_throw(&func.body),
                async_kind: self.detect_async_kind(&func),
                is_pure: self.analyze_is_pure(&func.body),
                is_inline: modifier_info.is_inline,
                exception_types: vec![],
                memory_effects: MemoryEffects::default(),
                resource_effects: ResourceEffects::default(),
            },
            type_parameters: type_params,
            is_static: modifier_info.is_static,
            source_location: self.context.create_location_from_span(field.span),
            metadata: FunctionMetadata {
                complexity_score: self.calculate_complexity(&func.body),
                statement_count: body_len,
                is_recursive: false, // Recursion detection requires call graph analysis
                call_count: 0,
                is_override: modifier_info.is_override,
                overload_signatures,
                operator_metadata,
                is_array_access,
                is_from_conversion,
                is_to_conversion,
                memory_annotations: self.extract_memory_annotations(&field.meta),
            },
        })
    }

    /// Lower a function signature for interfaces (no body, just signature)
    fn lower_function_signature(
        &mut self,
        field: &ClassField,
        func: &Function,
    ) -> LoweringResult<TypedMethodSignature> {
        let function_name = self.context.intern_string(&func.name);
        let function_symbol = self.context.symbol_table.create_function(function_name);

        // Enter function scope
        let function_scope = self.context.enter_scope(ScopeKind::Function);

        // Process type parameters
        let type_params = self.lower_type_parameters(&func.type_params)?;
        let mut type_param_map: HashMap<InternedString, TypeId> =
            HashMap::with_capacity(type_params.len());
        for tp in &type_params {
            let interned_name = tp.name;
            // Convert constraints to ConstraintKind for symbol table
            let constraint_kinds = tp
                .constraints
                .iter()
                .map(|_| {
                    crate::tast::type_checker::ConstraintKind::Implements {
                        interface_type: TypeId::invalid(), // Placeholder, will be resolved later
                    }
                })
                .collect();
            let symbol_id = self
                .context
                .symbol_table
                .create_type_parameter(interned_name, constraint_kinds);
            let type_id = self.context.type_table.borrow_mut().create_type_parameter(
                symbol_id,
                tp.constraints.clone(),
                tp.variance.into(),
            );
            type_param_map.insert(tp.name, type_id);
        }
        self.context.push_type_parameters(type_param_map);

        // Process parameters
        let mut parameters = Vec::new();
        for param in &func.params {
            parameters.push(self.lower_parameter(param)?);
        }

        // Process return type
        let return_type = if let Some(ret_type) = &func.return_type {
            self.lower_type(ret_type)?
        } else {
            self.context.type_table.borrow().void_type()
        };

        // Create function type and update symbol
        let param_types: Vec<TypeId> = parameters.iter().map(|p| p.param_type).collect();
        let function_type = self
            .context
            .type_table
            .borrow_mut()
            .create_function_type(param_types, return_type);

        // Update the symbol with its type
        self.context
            .symbol_table
            .update_symbol_type(function_symbol, function_type);

        // Interface methods have no body
        let body: Vec<TypedStatement> = Vec::new();

        // Process field modifiers and access
        let modifier_info = self.lower_modifiers(&field.modifiers)?;
        let visibility = self.lower_access(&field.access);

        self.context.pop_type_parameters();
        self.context.exit_scope();

        Ok(TypedMethodSignature {
            name: function_name,
            parameters,
            return_type,
            effects: crate::tast::node::FunctionEffects {
                can_throw: false,            // Interface methods are pure signatures
                async_kind: AsyncKind::Sync, // Async detection not needed for now
                is_pure: true,               // Interface methods are pure signatures
                is_inline: modifier_info.is_inline,
                exception_types: vec![],
                memory_effects: MemoryEffects::default(),
                resource_effects: ResourceEffects::default(),
            },
            source_location: self.context.create_location_from_span(field.span),
        })
    }

    /// Lower a parameter
    fn lower_parameter(&mut self, parameter: &FunctionParam) -> LoweringResult<TypedParameter> {
        let param_name = self.context.intern_string(&parameter.name);
        // Create the parameter symbol with the current scope
        let param_symbol = self
            .context
            .symbol_table
            .create_variable_in_scope(param_name, self.context.current_scope);

        // Add parameter to the current (function) scope so it can be resolved
        if let Some(scope) = self
            .context
            .scope_tree
            .get_scope_mut(self.context.current_scope)
        {
            scope.add_symbol(param_symbol, param_name);
        }

        let param_type = if let Some(type_annotation) = &parameter.type_hint {
            self.lower_type(type_annotation)?
        } else {
            self.context.type_table.borrow().dynamic_type()
        };

        // Update the parameter symbol with its type
        self.context
            .symbol_table
            .update_symbol_type(param_symbol, param_type);

        let default_value = if let Some(default) = &parameter.default_value {
            Some(self.lower_expression(default)?)
        } else {
            None
        };

        Ok(TypedParameter {
            symbol_id: param_symbol,
            name: param_name,
            param_type: param_type,
            is_optional: parameter.optional,
            default_value,
            mutability: crate::tast::Mutability::Immutable,
            source_location: self.context.create_location_from_span(parameter.span),
        })
    }

    /// Lower a type annotation
    fn lower_type(&mut self, type_annotation: &Type) -> LoweringResult<TypeId> {
        match type_annotation {
            Type::Path { path, params, .. } => {
                let name = if path.package.is_empty() {
                    path.name.clone()
                } else {
                    format!("{}.{}", path.package.join("."), path.name)
                };

                // Haxe Type Resolution Order:

                // 1. Check if it's a type parameter (in generic contexts)
                let interned_name = self.context.intern_string(&name);
                if let Some(type_param) = self.context.resolve_type_parameter(interned_name) {
                    return Ok(type_param);
                }

                // 2. Try to resolve as a built-in type (covers basic types first)
                // IMPORTANT: Skip "Array" when type params are present (e.g., Array<Body>).
                // resolve_builtin_type("Array") returns Array<Dynamic>, discarding params.
                // Instead, resolve the element type and create Array<ElementType>.
                if name == "Array" && !params.is_empty() {
                    let element_type = self.lower_type(&params[0])?;
                    return Ok(self
                        .context
                        .type_table
                        .borrow_mut()
                        .create_array_type(element_type));
                }
                if let Some(builtin_type) = self.resolve_builtin_type(&name) {
                    return Ok(builtin_type);
                }

                let interned_name = self.context.intern_string(&name);

                // 3. Module-level types (current module/file scope)
                // 4. Imported types (already registered during import processing)
                // 5. Top-level and standard library types (already in root scope)

                // IMPORTANT: First try to resolve through the import system.
                // This ensures we get the symbol with the correct qualified_name (e.g., "rayzor.Bytes")
                // rather than a duplicate symbol that may have been created without package context.
                let import_candidates = self.context.import_resolver.resolve_type(
                    interned_name,
                    self.context.current_scope,
                    self.context.namespace_resolver,
                );

                let import_resolved_symbol = import_candidates
                    .first()
                    .and_then(|qualified_path| {
                        self.context
                            .namespace_resolver
                            .lookup_symbol(qualified_path)
                    })
                    .and_then(|sym_id| self.context.symbol_table.get_symbol(sym_id))
                    .map(|s| (s.id, s.kind.clone()));

                // If import resolution found a symbol, use it. Otherwise fall back to scope lookup.
                let symbol_info = import_resolved_symbol.or_else(|| {
                    self.context
                        .symbol_table
                        .lookup_symbol(self.context.current_scope, interned_name)
                        .or_else(|| {
                            self.context.symbol_table.lookup_symbol(
                                ScopeId::first(), // Root scope contains imports and top-level types
                                interned_name,
                            )
                        })
                        .map(|s| (s.id, s.kind.clone()))
                });

                if let Some((symbol_id, symbol_kind)) = symbol_info {
                    // Process type arguments if present (now the symbol borrow is dropped)
                    let type_arg_ids = if !params.is_empty() {
                        let mut result = Vec::new();
                        for arg in params {
                            result.push(self.lower_type(arg)?);
                        }
                        result
                    } else {
                        Vec::new()
                    };

                    // Create appropriate type based on symbol kind
                    match symbol_kind {
                        crate::tast::SymbolKind::Class => {
                            // Check if this class already has a type from pre-registration
                            if let Some(symbol) = self.context.symbol_table.get_symbol(symbol_id) {
                                if symbol.type_id.is_valid() && type_arg_ids.is_empty() {
                                    // Use the existing type from pre-registration
                                    return Ok(symbol.type_id);
                                }
                            }
                            Ok(self
                                .context
                                .type_table
                                .borrow_mut()
                                .create_class_type(symbol_id, type_arg_ids))
                        }
                        crate::tast::SymbolKind::Interface => {
                            // Check if this interface already has a type from pre-registration
                            if let Some(symbol) = self.context.symbol_table.get_symbol(symbol_id) {
                                if symbol.type_id.is_valid() && type_arg_ids.is_empty() {
                                    // Use the existing type from pre-registration
                                    return Ok(symbol.type_id);
                                }
                            }
                            Ok(self
                                .context
                                .type_table
                                .borrow_mut()
                                .create_interface_type(symbol_id, type_arg_ids))
                        }
                        crate::tast::SymbolKind::Enum => Ok(self
                            .context
                            .type_table
                            .borrow_mut()
                            .create_enum_type(symbol_id, type_arg_ids)),
                        crate::tast::SymbolKind::TypeAlias => {
                            // For type aliases, we need to get the target type
                            let target_type = type_resolution::resolve_type_alias(
                                self.context.type_table,
                                self.context.symbol_table,
                                symbol_id,
                            );
                            Ok(self.context.type_table.borrow_mut().create_type(
                                crate::tast::core::TypeKind::TypeAlias {
                                    symbol_id,
                                    target_type,
                                    type_args: type_arg_ids,
                                },
                            ))
                        }
                        crate::tast::SymbolKind::Abstract => {
                            let underlying = None; // Abstract enums have no explicit underlying type
                            Ok(self.context.type_table.borrow_mut().create_type(
                                crate::tast::core::TypeKind::Abstract {
                                    symbol_id,
                                    underlying,
                                    type_args: type_arg_ids,
                                },
                            ))
                        }
                        _ => {
                            // For other symbol kinds, return dynamic type
                            Ok(self.context.type_table.borrow().dynamic_type())
                        }
                    }
                } else {
                    // Symbol not found, this might be a forward reference
                    // Create a placeholder type and defer resolution
                    let placeholder_type = self.context.type_table.borrow_mut().create_type(
                        crate::tast::core::TypeKind::Placeholder {
                            name: interned_name,
                        },
                    );

                    // Record this for later resolution
                    self.resolution_state
                        .deferred_resolutions
                        .push(DeferredTypeResolution {
                            type_name: name.clone(),
                            location: self.context.create_location(),
                            type_params: params.iter().map(|_| "T".to_string()).collect(), // TODO: extract actual param names
                            target_type_id: placeholder_type,
                        });

                    Ok(placeholder_type)
                }
            }
            Type::Function { params, ret, .. } => {
                let param_types = params
                    .iter()
                    .map(|param| self.lower_type(param))
                    .collect::<Result<Vec<_>, _>>()?;

                let return_type_id = self.lower_type(ret)?;

                // Create function type with default effects
                let effects = crate::tast::core::FunctionEffects {
                    can_throw: false,
                    is_async: false,
                    is_pure: false,
                    memory_effects: crate::tast::core::MemoryEffects::None,
                };

                Ok(self.context.type_table.borrow_mut().create_type(
                    crate::tast::core::TypeKind::Function {
                        params: param_types,
                        return_type: return_type_id,
                        effects,
                    },
                ))
            }
            Type::Anonymous { fields, .. } => {
                // Create proper anonymous type with fields
                let mut anonymous_fields = Vec::new();
                for field in fields {
                    let field_type_id = self.lower_type(&field.type_hint)?;
                    let field_name = self.context.intern_string(&field.name);
                    anonymous_fields.push(crate::tast::core::AnonymousField {
                        name: field_name,
                        type_id: field_type_id,
                        is_public: true, // Anonymous fields are typically public
                        optional: field.optional,
                    });
                }

                Ok(self.context.type_table.borrow_mut().create_type(
                    crate::tast::core::TypeKind::Anonymous {
                        fields: anonymous_fields,
                    },
                ))
            }
            Type::Optional { inner, .. } => {
                let inner_type_id = self.lower_type(inner)?;
                Ok(self
                    .context
                    .type_table
                    .borrow_mut()
                    .create_optional_type(inner_type_id))
            }
            Type::Parenthesis { inner, .. } => {
                // Just unwrap parentheses
                self.lower_type(inner)
            }
            Type::Intersection { left, right, .. } => {
                let left_type_id = self.lower_type(left)?;
                let right_type_id = self.lower_type(right)?;

                // If both sides resolve to Anonymous types, merge their fields
                // into a single Anonymous type (right side wins on name conflicts)
                let merged = {
                    let type_table = self.context.type_table.borrow();
                    let left_resolved = Self::resolve_alias_chain(&type_table, left_type_id);
                    let right_resolved = Self::resolve_alias_chain(&type_table, right_type_id);
                    let left_anon = type_table.get(left_resolved).and_then(|t| {
                        if let TypeKind::Anonymous { fields } = &t.kind {
                            Some(fields.clone())
                        } else {
                            None
                        }
                    });
                    let right_anon = type_table.get(right_resolved).and_then(|t| {
                        if let TypeKind::Anonymous { fields } = &t.kind {
                            Some(fields.clone())
                        } else {
                            None
                        }
                    });
                    match (left_anon, right_anon) {
                        (Some(left_fields), Some(right_fields)) => {
                            // Merge: start with left fields, override/add right fields
                            let mut merged_fields = left_fields;
                            for rf in right_fields {
                                if let Some(existing) =
                                    merged_fields.iter_mut().find(|f| f.name == rf.name)
                                {
                                    *existing = rf;
                                } else {
                                    merged_fields.push(rf);
                                }
                            }
                            Some(merged_fields)
                        }
                        _ => None,
                    }
                };

                if let Some(merged_fields) = merged {
                    Ok(self
                        .context
                        .type_table
                        .borrow_mut()
                        .create_type(TypeKind::Anonymous {
                            fields: merged_fields,
                        }))
                } else {
                    Ok(self
                        .context
                        .type_table
                        .borrow_mut()
                        .create_type(TypeKind::Intersection {
                            types: vec![left_type_id, right_type_id],
                        }))
                }
            }
            Type::Wildcard { .. } => {
                // Wildcard types are used in type parameters, return Unknown type
                Ok(self.context.type_table.borrow().unknown_type())
            }
        }
    }

    /// Resolve a TypePath to a TypeId for constructor calls
    fn resolve_type_path(&mut self, type_path: &parser::TypePath) -> LoweringResult<TypeId> {
        // First check if it's a built-in type
        if type_path.package.is_empty() && type_path.sub.is_none() {
            if let Some(builtin_type) = self.resolve_builtin_type(&type_path.name) {
                return Ok(builtin_type);
            }
        }

        // Try to resolve using the import resolver first
        let name_interned = self.context.string_interner.intern(&type_path.name);
        let candidates = self.context.import_resolver.resolve_type(
            name_interned,
            self.context.current_scope,
            self.context.namespace_resolver,
        );

        if !candidates.is_empty() {
            // Use the first candidate (in a full implementation, we'd handle ambiguity)
            let qualified_path = &candidates[0];
            if let Some(symbol_id) = self
                .context
                .namespace_resolver
                .lookup_symbol(qualified_path)
            {
                if let Some(symbol) = self.context.symbol_table.get_symbol(symbol_id) {
                    return Ok(symbol.type_id);
                }
            }
        }

        // If not found through imports, try direct resolution
        let qualified_path = if type_path.package.is_empty() {
            // Try to find in current package first
            if let Some(current_package) = self.context.current_package {
                let package_segments = self
                    .context
                    .namespace_resolver
                    .find_symbols_by_name(name_interned, current_package);
                if let Some((_, symbol_id)) = package_segments.first() {
                    if let Some(symbol) = self.context.symbol_table.get_symbol(*symbol_id) {
                        return Ok(symbol.type_id);
                    }
                }
            }

            // Otherwise, treat as a simple name
            super::namespace::QualifiedPath::simple(name_interned)
        } else {
            // Create a qualified path from the package
            let package_path: Vec<_> = type_path
                .package
                .iter()
                .map(|s| self.context.string_interner.intern(s))
                .collect();
            super::namespace::QualifiedPath::new(package_path, name_interned)
        };

        // Try to resolve from namespace
        if let Some(symbol_id) = self
            .context
            .namespace_resolver
            .lookup_symbol(&qualified_path)
        {
            if let Some(symbol) = self.context.symbol_table.get_symbol(symbol_id) {
                return Ok(symbol.type_id);
            }
        }

        // Construct the full path for fallback
        let full_path = if type_path.package.is_empty() {
            type_path.name.clone()
        } else {
            format!("{}.{}", type_path.package.join("."), type_path.name)
        };

        // Try to resolve from symbol table (legacy path)
        let interned_name = self.context.intern_string(&full_path);

        if let Some(symbol) = self.context.symbol_table.lookup_symbol(
            ScopeId::first(), // Look in root scope for type definitions
            interned_name,
        ) {
            // Return the type associated with this symbol
            Ok(symbol.type_id)
        } else {
            // Type not found - create a placeholder and defer resolution
            let placeholder_type = self
                .context
                .type_table
                .borrow_mut()
                .create_type(crate::tast::core::TypeKind::Unknown);

            // Add to deferred resolutions for later processing
            self.resolution_state
                .deferred_resolutions
                .push(DeferredTypeResolution {
                    type_name: full_path.clone(),
                    target_type_id: placeholder_type,
                    location: self.context.create_location(),
                    type_params: Vec::new(), // For constructor calls, we don't need type params here
                });

            Ok(placeholder_type)
        }
    }

    /// Resolve built-in types
    fn resolve_builtin_type(&self, name: &str) -> Option<TypeId> {
        let type_table = self.context.type_table.borrow();
        match name {
            "Int" => Some(type_table.int_type()),
            "Float" => Some(type_table.float_type()),
            "String" => Some(type_table.string_type()),
            "Bool" => Some(type_table.bool_type()),
            "Dynamic" => Some(type_table.dynamic_type()),
            "Void" => Some(type_table.void_type()),
            "Array" => {
                // Array<T> needs type parameter, return dynamic array for now
                let dynamic_type = type_table.dynamic_type();
                drop(type_table); // Release borrow before mutable borrow
                Some(
                    self.context
                        .type_table
                        .borrow_mut()
                        .create_array_type(dynamic_type),
                )
            }
            _ => None,
        }
    }

    /// Lower modifiers and extract static, override, etc.
    fn lower_modifiers(&mut self, modifiers: &[Modifier]) -> LoweringResult<ModifierInfo> {
        let mut modifier_info = ModifierInfo::default();

        for modifier in modifiers {
            match modifier {
                parser::Modifier::Static => modifier_info.is_static = true,
                parser::Modifier::Override => modifier_info.is_override = true,
                parser::Modifier::Inline => modifier_info.is_inline = true,
                parser::Modifier::Dynamic => modifier_info.is_dynamic = true,
                parser::Modifier::Macro => modifier_info.is_macro = true,
                parser::Modifier::Final => modifier_info.is_final = true,
                parser::Modifier::Extern => modifier_info.is_extern = true,
            }
        }

        Ok(modifier_info)
    }

    /// Lower access modifiers (separate from other modifiers)
    fn lower_access(&mut self, access: &Option<parser::Access>) -> Visibility {
        match access {
            Some(parser::Access::Public) => Visibility::Public,
            Some(parser::Access::Private) => Visibility::Private,
            None => Visibility::Internal, // Default visibility
        }
    }

    /// Extract memory safety annotations from metadata
    fn extract_memory_annotations(
        &self,
        metadata: &[parser::Metadata],
    ) -> Vec<crate::tast::MemoryAnnotation> {
        metadata
            .iter()
            .filter_map(|meta| {
                // Parse parameters if present (e.g., @:safety(strict=true))
                if !meta.params.is_empty() {
                    let params = self.parse_metadata_params(&meta.params);
                    crate::tast::MemoryAnnotation::from_metadata_with_params(&meta.name, &params)
                } else {
                    crate::tast::MemoryAnnotation::from_metadata_name(&meta.name)
                }
            })
            .collect()
    }

    /// Extract derived traits from @:derive metadata
    /// Example: @:derive([Clone, Copy]) or @:derive(Clone)
    fn extract_derived_traits(
        &self,
        class_decl: &parser::ClassDecl,
    ) -> Vec<crate::tast::DerivedTrait> {
        let trait_names = class_decl.get_derive_traits();

        let mut derived_traits = Vec::new();
        for trait_name in trait_names {
            if let Some(trait_) = crate::tast::DerivedTrait::from_str(&trait_name) {
                derived_traits.push(trait_);
            } else {
                warn!(
                    "Warning: Unknown derived trait '{}' in @:derive",
                    trait_name
                );
            }
        }

        // Validate trait dependencies (e.g., Eq requires PartialEq)
        let mut missing_deps = Vec::new();
        for trait_ in &derived_traits {
            for required in trait_.requires() {
                if !derived_traits.contains(&required) {
                    missing_deps.push((trait_.as_str(), required.as_str()));
                }
            }
        }

        if !missing_deps.is_empty() {
            warn!(
                "Warning: Missing required trait dependencies for class '{}':",
                class_decl.name
            );
            for (trait_, required) in missing_deps {
                warn!("  - {} requires {}", trait_, required);
            }
        }

        // Check if class has @:rc or @:arc - these require Clone
        let has_rc = class_decl
            .meta
            .iter()
            .any(|m| m.name == "rc" || m.name == "arc");
        if has_rc && !derived_traits.contains(&crate::tast::DerivedTrait::Clone) {
            eprintln!(
                "ERROR: Class '{}' has @:rc/@:arc metadata but does not derive Clone",
                class_decl.name
            );
            eprintln!("  Reference counted types must be Clone to support shared ownership");
            eprintln!("  Add @:derive(Clone) to fix this error");

            // Auto-add Clone for RC types to prevent compilation errors
            // User will see the warning above
            eprintln!("  Note: Automatically adding Clone trait for @:rc class");
            derived_traits.push(crate::tast::DerivedTrait::Clone);
        }

        // If Copy is derived, automatically add Clone (Copy implies Clone)
        if derived_traits.contains(&crate::tast::DerivedTrait::Copy)
            && !derived_traits.contains(&crate::tast::DerivedTrait::Clone)
        {
            derived_traits.push(crate::tast::DerivedTrait::Clone);
        }

        derived_traits
    }

    /// Validate that derived traits are compatible with field types
    fn validate_derived_traits(
        &self,
        typed_class: &TypedClass,
        derived_traits: &mut Vec<crate::tast::DerivedTrait>,
        class_name: &str,
    ) {
        use crate::tast::DerivedTrait;

        let has_clone = derived_traits.contains(&DerivedTrait::Clone);
        let has_copy = derived_traits.contains(&DerivedTrait::Copy);

        // Validate Clone: all fields must be Clone
        if has_clone {
            let mut non_clone_fields = Vec::new();

            for field in &typed_class.fields {
                if !self.is_type_clone(field.field_type) {
                    let field_name_str = self
                        .context
                        .string_interner
                        .get(field.name)
                        .unwrap_or("?")
                        .to_string();
                    non_clone_fields.push(field_name_str);
                }
            }

            if !non_clone_fields.is_empty() {
                eprintln!(
                    "ERROR: Class '{}' derives Clone but has non-Clone fields:",
                    class_name
                );
                for field_name in &non_clone_fields {
                    eprintln!("  - Field '{}' is not Clone", field_name);
                }
                eprintln!("  All fields must derive Clone or be primitive Copy types");
                eprintln!("  Consider adding @:derive(Clone) to field types or removing Clone from this class");

                // Remove Clone trait to prevent incorrect codegen
                derived_traits.retain(|t| *t != DerivedTrait::Clone);
            }
        }

        // Validate Copy: all fields must be Copy
        if has_copy {
            let mut non_copy_fields = Vec::new();

            for field in &typed_class.fields {
                if !self.is_type_copy(field.field_type) {
                    let field_name_str = self
                        .context
                        .string_interner
                        .get(field.name)
                        .unwrap_or("?")
                        .to_string();
                    non_copy_fields.push(field_name_str);
                }
            }

            if !non_copy_fields.is_empty() {
                eprintln!(
                    "ERROR: Class '{}' derives Copy but has non-Copy fields:",
                    class_name
                );
                for field_name in &non_copy_fields {
                    eprintln!("  - Field '{}' is not Copy", field_name);
                }
                eprintln!("  Copy types can only contain primitive Copy types (Int, Float, Bool)");
                eprintln!("  or other classes that derive Copy");
                eprintln!("  Consider using Clone instead of Copy, or remove Copy from this class");

                // Remove Copy trait to prevent incorrect codegen
                derived_traits.retain(|t| *t != DerivedTrait::Copy);
                // Also remove Clone if it was auto-added by Copy
                if !has_clone {
                    derived_traits.retain(|t| *t != DerivedTrait::Clone);
                }
            }
        }
    }

    /// Check if a type implements Clone
    fn is_type_clone(&self, type_id: TypeId) -> bool {
        let type_table = self.context.type_table.borrow();

        if let Some(type_info) = type_table.get(type_id) {
            match &type_info.kind {
                // Primitive types are implicitly Copy (and thus Clone)
                crate::tast::core::TypeKind::Int
                | crate::tast::core::TypeKind::Float
                | crate::tast::core::TypeKind::Bool
                | crate::tast::core::TypeKind::Void => true,

                // String is Clone but not Copy
                crate::tast::core::TypeKind::String => true,

                // Class types: check if they derive Clone
                crate::tast::core::TypeKind::Class { symbol_id, .. } => {
                    // TODO: Look up class and check derived_traits
                    // For now, assume classes are Clone (conservative)
                    true
                }

                // Arrays and Maps are Clone if their element types are Clone
                crate::tast::core::TypeKind::Array { element_type } => {
                    self.is_type_clone(*element_type)
                }

                // Other types default to not Clone
                _ => false,
            }
        } else {
            false
        }
    }

    /// Check if a type implements Copy
    fn is_type_copy(&self, type_id: TypeId) -> bool {
        let type_table = self.context.type_table.borrow();

        if let Some(type_info) = type_table.get(type_id) {
            match &type_info.kind {
                // Only primitive types are Copy
                crate::tast::core::TypeKind::Int
                | crate::tast::core::TypeKind::Float
                | crate::tast::core::TypeKind::Bool => true,

                // Class types: check if they derive Copy
                crate::tast::core::TypeKind::Class { symbol_id, .. } => {
                    // TODO: Look up class and check if it derives Copy
                    // For now, assume classes are NOT Copy (safe default)
                    false
                }

                // String, Arrays, and other heap types are NOT Copy
                _ => false,
            }
        } else {
            false
        }
    }

    /// Parse metadata parameters from expressions
    /// Converts Expr nodes to String values (positional parameters)
    /// e.g., @:safety(true) -> ["true"], @:author("Name") -> ["Name"]
    fn parse_metadata_params(&self, params: &[parser::Expr]) -> Vec<String> {
        params
            .iter()
            .filter_map(|expr| {
                match &expr.kind {
                    // Boolean literals
                    parser::ExprKind::Bool(b) => Some(b.to_string()),
                    // Integer literals
                    parser::ExprKind::Int(n) => Some(n.to_string()),
                    // String literals
                    parser::ExprKind::String(s) => Some(s.clone()),
                    // Identifiers (e.g., true, false, null)
                    parser::ExprKind::Ident(name) => Some(name.clone()),
                    // Float literals
                    parser::ExprKind::Float(f) => Some(f.to_string()),
                    // Skip other expression types
                    _ => None,
                }
            })
            .collect()
    }

    /// Resolve a TypeId to the underlying class symbol if it's a class type
    fn resolve_type_to_class_symbol(&self, type_id: TypeId) -> Option<SymbolId> {
        let type_table = self.context.type_table.borrow();
        self.resolve_type_to_class_symbol_inner(&type_table, type_id)
    }

    /// Inner helper that takes a borrowed type table to allow recursive calls
    fn resolve_type_to_class_symbol_inner(
        &self,
        type_table: &std::cell::Ref<'_, crate::tast::TypeTable>,
        type_id: TypeId,
    ) -> Option<SymbolId> {
        if let Some(type_info) = type_table.get(type_id) {
            match &type_info.kind {
                crate::tast::core::TypeKind::Class { symbol_id, .. } => Some(*symbol_id),
                crate::tast::core::TypeKind::GenericInstance { base_type, .. } => {
                    // For generic instances like Thread<Int>, resolve the base type
                    self.resolve_type_to_class_symbol_inner(type_table, *base_type)
                }
                crate::tast::core::TypeKind::TypeAlias { target_type, .. } => {
                    // For type aliases like `typedef Bytes = rayzor.Bytes`, follow the target type
                    self.resolve_type_to_class_symbol_inner(type_table, *target_type)
                }
                _ => None,
            }
        } else {
            None
        }
    }

    /// Find a field in a class by symbol
    fn find_field_in_class(
        &self,
        class_symbol: &SymbolId,
        field_symbol: SymbolId,
    ) -> Option<(InternedString, TypeId, bool)> {
        if let Some(fields) = self.class_fields.get(class_symbol) {
            fields
                .iter()
                .find(|(_, symbol, _)| *symbol == field_symbol)
                .map(|(name, field_symbol, is_static)| {
                    let field_type = if let Some(field_sym) =
                        self.context.symbol_table.get_symbol(*field_symbol)
                    {
                        field_sym.type_id
                    } else {
                        self.context.type_table.borrow().dynamic_type()
                    };
                    (*name, field_type, *is_static)
                })
        } else {
            None
        }
    }

    /// Copy parent class fields to child class for field resolution
    /// This ensures that inherited fields can be resolved correctly in the child class
    /// Call this BEFORE processing child's members so fields are available in constructors
    fn copy_parent_fields(&mut self, parent_type_id: TypeId, child_symbol: SymbolId) {
        // Get the parent class symbol from the type
        if let Some(parent_symbol) = self.resolve_type_to_class_symbol(parent_type_id) {
            // Copy this parent's fields
            // Note: If the parent itself inherits from a grandparent, its class_fields
            // will already contain the grandparent's fields (since we process classes in order)
            if let Some(parent_fields) = self.class_fields.get(&parent_symbol).cloned() {
                // Clone to avoid borrow conflicts
                if let Some(child_fields) = self.class_fields.get_mut(&child_symbol) {
                    // Add parent fields to the beginning of child's field list
                    // This maintains the correct field order: parent fields first, then child fields
                    for parent_field in parent_fields.iter().rev() {
                        // Insert at beginning to maintain order
                        child_fields.insert(0, *parent_field);
                    }
                }
            }
        }
    }

    /// Copy parent class methods to child class for method resolution
    /// This enables method inheritance and overriding
    /// Call this AFTER processing child's members so child methods come first for overriding
    fn copy_parent_methods(&mut self, parent_type_id: TypeId, child_symbol: SymbolId) {
        // Get the parent class symbol from the type
        if let Some(parent_symbol) = self.resolve_type_to_class_symbol(parent_type_id) {
            // Copy this parent's methods
            // Parent methods are added to the child's method list before child methods are processed
            // When child methods are added, they will replace parent methods with the same name (override)
            if let Some(parent_methods) = self.class_methods.get(&parent_symbol).cloned() {
                // Clone to avoid borrow conflicts
                if let Some(child_methods) = self.class_methods.get_mut(&child_symbol) {
                    // Add parent methods to child's method list
                    // Child methods will override these when they're processed
                    for parent_method in parent_methods.iter() {
                        child_methods.push(*parent_method);
                    }
                }
            }
        }
    }

    /// Resolve a method symbol for a given receiver and method name
    fn resolve_method_symbol(
        &mut self,
        receiver: &TypedExpression,
        method_name: InternedString,
    ) -> SymbolId {
        // Try to resolve method from receiver's type
        match &receiver.kind {
            TypedExpressionKind::This { this_type: _ } => {
                // If calling method on 'this', look in current class
                if let Some(class_symbol) = self.context.class_context_stack.last() {
                    if let Some(methods) = self.class_methods.get(class_symbol) {
                        if let Some((_, method_symbol, _)) =
                            methods.iter().find(|(name, _, _)| *name == method_name)
                        {
                            return *method_symbol;
                        }
                    }
                }
            }
            TypedExpressionKind::Variable { symbol_id } => {
                // Try to resolve method from variable's type
                if let Some(symbol) = self.context.symbol_table.get_symbol(*symbol_id) {
                    if let Some(class_symbol) = self.resolve_type_to_class_symbol(symbol.type_id) {
                        // First check local class_methods (for classes lowered in this compilation unit)
                        if let Some(methods) = self.class_methods.get(&class_symbol) {
                            if let Some((_, method_symbol, _)) =
                                methods.iter().find(|(name, _, _)| *name == method_name)
                            {
                                return *method_symbol;
                            }
                        }
                        // Fallback: check the shared symbol table's class scope
                        // (for extern classes compiled in a different compilation unit)
                        if let Some(class_sym) = self.context.symbol_table.get_symbol(class_symbol)
                        {
                            if let Some(method_sym) = self
                                .context
                                .symbol_table
                                .lookup_symbol(class_sym.scope_id, method_name)
                            {
                                if method_sym.kind == crate::tast::symbols::SymbolKind::Function {
                                    return method_sym.id;
                                }
                            }
                        }
                    }
                }
            }
            TypedExpressionKind::MethodCall { .. } | TypedExpressionKind::FunctionCall { .. } => {
                // For method chains like z.mul(z).add(c), the receiver is a MethodCall.
                // We need to infer the type of that expression and resolve the method on it.
                if let Ok(receiver_type) = self.infer_expression_type(&receiver.kind) {
                    if let Some(class_symbol) = self.resolve_type_to_class_symbol(receiver_type) {
                        if let Some(methods) = self.class_methods.get(&class_symbol) {
                            if let Some((_, method_symbol, _)) =
                                methods.iter().find(|(name, _, _)| *name == method_name)
                            {
                                return *method_symbol;
                            }
                        }
                    }
                }
            }
            TypedExpressionKind::New { class_type, .. } => {
                // For `new Complex().method()`, resolve method on the class type
                if let Some(class_symbol) = self.resolve_type_to_class_symbol(*class_type) {
                    if let Some(methods) = self.class_methods.get(&class_symbol) {
                        if let Some((_, method_symbol, _)) =
                            methods.iter().find(|(name, _, _)| *name == method_name)
                        {
                            return *method_symbol;
                        }
                    }
                }
            }
            _ => {
                // For other receiver types, try general type inference
                if let Ok(receiver_type) = self.infer_expression_type(&receiver.kind) {
                    if let Some(class_symbol) = self.resolve_type_to_class_symbol(receiver_type) {
                        if let Some(methods) = self.class_methods.get(&class_symbol) {
                            if let Some((_, method_symbol, _)) =
                                methods.iter().find(|(name, _, _)| *name == method_name)
                            {
                                return *method_symbol;
                            }
                        }
                    }
                }
            }
        }

        // Fallback: Try to resolve using the receiver expression's type
        // (may differ from the variable symbol's type for extern classes)
        if let Some(class_symbol) = self.resolve_type_to_class_symbol(receiver.expr_type) {
            if let Some(methods) = self.class_methods.get(&class_symbol) {
                if let Some((_, method_symbol, _)) =
                    methods.iter().find(|(name, _, _)| *name == method_name)
                {
                    return *method_symbol;
                }
            }
        }

        // Last resort: scan ALL class_methods for a unique match by name.
        // This handles extern classes where TypeId is invalid but the method
        // definition symbols have full metadata (qualified_name, native_name, etc.).
        {
            let mut found: Option<SymbolId> = None;
            let mut ambiguous = false;
            for (_class_sym, methods) in &self.class_methods {
                if let Some((_, method_symbol, _)) =
                    methods.iter().find(|(name, _, _)| *name == method_name)
                {
                    if found.is_some() {
                        ambiguous = true;
                        break;
                    }
                    found = Some(*method_symbol);
                }
            }
            if let Some(method_symbol) = found {
                if !ambiguous {
                    return method_symbol;
                }
            }
        }

        // Create a method symbol placeholder if we can't resolve it
        // The type checker will resolve this properly during type checking
        self.context.symbol_table.create_function(method_name)
    }

    /// Try to find a static extension method in using modules
    /// Returns (class_symbol, method_symbol) if found
    fn find_static_extension_method(
        &self,
        method_name: InternedString,
        _receiver_type: TypeId,
    ) -> Option<(SymbolId, SymbolId)> {
        // Check each using module for a static method with this name
        for (_class_name, class_symbol) in &self.using_modules {
            // First, check local class_methods (for classes lowered in this instance)
            if let Some(methods) = self.class_methods.get(class_symbol) {
                for (meth_name, meth_symbol, is_static) in methods {
                    if *meth_name == method_name && *is_static {
                        return Some((*class_symbol, *meth_symbol));
                    }
                }
            }

            // Then, check the shared symbol table for methods registered by other lowering passes
            // Look up the class symbol to get its scope, then search for the method
            if let Some(class_sym) = self.context.symbol_table.get_symbol(*class_symbol) {
                // The class should have a scope ID where its members are registered
                // Try to find a method with the given name in that scope
                if let Some(method_sym) = self
                    .context
                    .symbol_table
                    .lookup_symbol(class_sym.scope_id, method_name)
                {
                    // Check if it's a static method by looking at its modifiers or kind
                    if method_sym.kind == crate::tast::symbols::SymbolKind::Function {
                        return Some((*class_symbol, method_sym.id));
                    }
                }
            }
        }
        None
    }

    /// Lower an expression as a statement
    fn lower_expression_as_statement(&mut self, expr: &Expr) -> LoweringResult<TypedStatement> {
        let typed_expr = self.lower_expression(expr)?;
        Ok(TypedStatement::Expression {
            expression: typed_expr,
            source_location: self.context.create_location(),
        })
    }

    /// Lower a statement (placeholder - not used with new parser)
    fn lower_statement(&mut self, _statement: &str) -> LoweringResult<TypedStatement> {
        let location = self.context.create_location();

        // Placeholder implementation
        Ok(TypedStatement::Expression {
            expression: TypedExpression {
                expr_type: self.context.type_table.borrow().void_type(),
                kind: TypedExpressionKind::Null,
                usage: VariableUsage::Copy,
                lifetime_id: crate::tast::LifetimeId::first(),
                source_location: location,
                metadata: ExpressionMetadata::default(),
            },
            source_location: location,
        })
    }

    /// Placeholder for old statement lowering - not used with new parser
    fn _old_statement_lowering_placeholder(&mut self) {
        // This was the old statement lowering implementation
        // Not used with the new parser interface
    }

    /// Lower an expression
    fn lower_expression(&mut self, expression: &Expr) -> LoweringResult<TypedExpression> {
        let kind = match &expression.kind {
            ExprKind::Int(value) => TypedExpressionKind::Literal {
                value: LiteralValue::Int(*value),
            },
            ExprKind::Float(value) => TypedExpressionKind::Literal {
                value: LiteralValue::Float(*value),
            },
            ExprKind::String(value) => TypedExpressionKind::Literal {
                value: LiteralValue::String(value.clone()),
            },
            ExprKind::Bool(value) => TypedExpressionKind::Literal {
                value: LiteralValue::Bool(*value),
            },
            ExprKind::Null => TypedExpressionKind::Null,
            ExprKind::Regex { pattern, flags } => TypedExpressionKind::Literal {
                value: LiteralValue::RegexWithFlags {
                    pattern: pattern.clone(),
                    flags: flags.clone(),
                },
            },
            ExprKind::Ident(name) => {
                let id_name = self.context.intern_string(name);

                // Need to resolve symbol by walking up the scope hierarchy
                let symbol_id =
                    self.resolve_symbol_in_scope_hierarchy(id_name)
                        .ok_or_else(|| LoweringError::UnresolvedSymbol {
                            name: name.clone(),
                            location: self.context.create_location_from_span(expression.span),
                        })?;

                // Check if this symbol is an instance VAR field of the current class
                // (not a method). If so, we need to create a FieldAccess with implicit
                // `this` receiver: `i = value` → `this.i = value`.
                let is_instance_field =
                    if let Some(class_symbol) = self.context.class_context_stack.last() {
                        let is_in_fields = self
                            .class_fields
                            .get(class_symbol)
                            .map(|fields| {
                                fields.iter().any(|(_, field_sym, is_static)| {
                                    *field_sym == symbol_id && !is_static
                                })
                            })
                            .unwrap_or(false);
                        // Exclude methods — they are handled by the call resolution path
                        let is_method = self
                            .class_methods
                            .get(class_symbol)
                            .map(|methods| {
                                methods
                                    .iter()
                                    .any(|(_, method_sym, _)| *method_sym == symbol_id)
                            })
                            .unwrap_or(false);
                        is_in_fields && !is_method
                    } else {
                        false
                    };

                if is_instance_field && !self.in_static_method {
                    // Create implicit `this` receiver for instance field access
                    // in non-static methods/constructors.
                    let this_name = self.context.intern_string("this");
                    let this_symbol = self
                        .resolve_symbol_in_scope_hierarchy(this_name)
                        .unwrap_or_else(|| self.context.symbol_table.create_variable(this_name));
                    let this_type = self
                        .context
                        .class_context_stack
                        .last()
                        .and_then(|cs| self.context.symbol_table.get_symbol(*cs))
                        .map(|s| s.type_id)
                        .unwrap_or_else(|| self.context.type_table.borrow().dynamic_type());
                    let receiver = TypedExpression {
                        expr_type: this_type,
                        kind: TypedExpressionKind::Variable {
                            symbol_id: this_symbol,
                        },
                        usage: VariableUsage::Copy,
                        lifetime_id: crate::tast::LifetimeId::first(),
                        source_location: self.context.create_location(),
                        metadata: ExpressionMetadata::default(),
                    };
                    TypedExpressionKind::FieldAccess {
                        object: Box::new(receiver),
                        field_symbol: symbol_id,
                        is_optional: false,
                    }
                } else {
                    TypedExpressionKind::Variable { symbol_id }
                }
            }
            ExprKind::Binary { left, op, right } => {
                // Special handling for `is` operator: `expr is Type`
                if matches!(op, BinaryOp::Is) {
                    let left_expr = self.lower_expression(left)?;
                    // The right side is a type name parsed as an expression (Ident)
                    // Extract the type name and resolve it
                    // Build a TypePath from the right-hand expression
                    let type_path = match &right.kind {
                        ExprKind::Ident(name) => parser::TypePath {
                            package: vec![],
                            name: name.clone(),
                            sub: None,
                        },
                        ExprKind::Field { expr, field, .. } => {
                            // Handle qualified names like `pack.Type`
                            let mut parts = Vec::new();
                            fn collect_parts(e: &Expr, parts: &mut Vec<String>) {
                                match &e.kind {
                                    ExprKind::Ident(n) => parts.push(n.clone()),
                                    ExprKind::Field { expr, field, .. } => {
                                        collect_parts(expr, parts);
                                        parts.push(field.clone());
                                    }
                                    _ => {}
                                }
                            }
                            collect_parts(expr, &mut parts);
                            parser::TypePath {
                                package: parts,
                                name: field.clone(),
                                sub: None,
                            }
                        }
                        _ => {
                            return Err(LoweringError::UnresolvedType {
                                type_name: format!("{:?}", right.kind),
                                location: self.context.create_location_from_span(expression.span),
                            });
                        }
                    };
                    // Resolve via full type resolution (handles user classes, imports, namespaces)
                    let check_type = self.resolve_type_path(&type_path)?;
                    TypedExpressionKind::Is {
                        expression: Box::new(left_expr),
                        check_type,
                    }
                } else {
                    let left_expr = self.lower_expression(left)?;
                    let right_expr = self.lower_expression(right)?;
                    let typed_op = self.lower_binary_operator(op)?;

                    TypedExpressionKind::BinaryOp {
                        left: Box::new(left_expr),
                        operator: typed_op,
                        right: Box::new(right_expr),
                    }
                }
            }
            ExprKind::Unary { op, expr } => {
                let operand_expr = self.lower_expression(expr)?;
                let typed_op = self.lower_unary_operator(op)?;

                TypedExpressionKind::UnaryOp {
                    operator: typed_op,
                    operand: Box::new(operand_expr),
                }
            }
            ExprKind::Call { expr, args } => {
                return self.lower_call_expression(expression, expr, args);
            }
            ExprKind::Field { expr, field, is_optional } => {
                return self.lower_field_expression(expression, expr, field, *is_optional);
            }
            ExprKind::Index { expr, index } => {
                let array_expr = self.lower_expression(expr)?;
                let index_expr = self.lower_expression(index)?;

                TypedExpressionKind::ArrayAccess {
                    array: Box::new(array_expr),
                    index: Box::new(index_expr),
                }
            }
            ExprKind::Assign { left, op, right } => {
                let target_expr = self.lower_expression(left)?;
                let value_expr = self.lower_expression(right)?;

                match op {
                    parser::AssignOp::Assign => {
                        // Simple assignment: target = value
                        TypedExpressionKind::BinaryOp {
                            left: Box::new(target_expr),
                            operator: BinaryOperator::Assign,
                            right: Box::new(value_expr),
                        }
                    }
                    _ => {
                        // Compound assignment: target op= value
                        // This needs to be: target = target op value
                        let target_clone = target_expr.clone();

                        // Map compound assignment operators to their corresponding binary operators
                        let binary_op = match op {
                            parser::AssignOp::AddAssign => BinaryOperator::Add,
                            parser::AssignOp::SubAssign => BinaryOperator::Sub,
                            parser::AssignOp::MulAssign => BinaryOperator::Mul,
                            parser::AssignOp::DivAssign => BinaryOperator::Div,
                            parser::AssignOp::ModAssign => BinaryOperator::Mod,
                            parser::AssignOp::AndAssign => BinaryOperator::BitAnd,
                            parser::AssignOp::OrAssign => BinaryOperator::BitOr,
                            parser::AssignOp::XorAssign => BinaryOperator::BitXor,
                            parser::AssignOp::ShlAssign => BinaryOperator::Shl,
                            parser::AssignOp::ShrAssign => BinaryOperator::Shr,
                            parser::AssignOp::UshrAssign => BinaryOperator::Shr, // UShr maps to Shr
                            parser::AssignOp::Assign => unreachable!(),          // Handled above
                        };

                        // Create the binary operation: target op value
                        let binary_expr = TypedExpression {
                            expr_type: target_expr.expr_type,
                            kind: TypedExpressionKind::BinaryOp {
                                left: Box::new(target_clone),
                                operator: binary_op,
                                right: Box::new(value_expr),
                            },
                            usage: VariableUsage::Copy,
                            lifetime_id: crate::tast::LifetimeId::first(),
                            source_location: self.context.create_location(),
                            metadata: ExpressionMetadata::default(),
                        };

                        // Now assign the result back to target: target = (target op value)
                        TypedExpressionKind::BinaryOp {
                            left: Box::new(target_expr),
                            operator: BinaryOperator::Assign,
                            right: Box::new(binary_expr),
                        }
                    }
                }
            }
            ExprKind::New {
                type_path,
                params,
                args,
            } => {
                // Resolve the base class type from type_path
                let base_class_type_id = self.resolve_type_path(type_path)?;

                // Lower constructor arguments
                let arg_exprs = args
                    .iter()
                    .map(|arg| self.lower_expression(arg))
                    .collect::<Result<Vec<_>, _>>()?;

                // Lower type arguments from params
                let type_args = params
                    .iter()
                    .map(|param| self.lower_type(param))
                    .collect::<Result<Vec<_>, _>>()?;

                // If type arguments are provided, create an instantiated type
                // e.g., new Array<Thread<Int>>() should have type Array<Thread<Int>>, not just Array
                let actual_class_type = if !type_args.is_empty() {
                    let symbol_id_opt = {
                        let type_table = self.context.type_table.borrow();
                        if let Some(base_type_info) = type_table.get(base_class_type_id) {
                            match &base_type_info.kind {
                                crate::tast::core::TypeKind::Class { symbol_id, .. } => {
                                    Some((*symbol_id, false)) // (symbol_id, is_array)
                                }
                                crate::tast::core::TypeKind::Array { .. } => {
                                    Some((SymbolId::invalid(), true)) // Mark as array type
                                }
                                _ => None,
                            }
                        } else {
                            None
                        }
                    };

                    if let Some((symbol_id, is_array)) = symbol_id_opt {
                        if is_array && type_args.len() == 1 {
                            self.context
                                .type_table
                                .borrow_mut()
                                .create_array_type(type_args[0])
                        } else if !is_array {
                            self.context
                                .type_table
                                .borrow_mut()
                                .create_class_type(symbol_id, type_args.clone())
                        } else {
                            base_class_type_id
                        }
                    } else {
                        base_class_type_id
                    }
                } else {
                    // No explicit type args — try to infer from constructor argument types
                    self.infer_type_args_from_constructor(base_class_type_id, &arg_exprs)
                        .unwrap_or(base_class_type_id)
                };

                // Extract and intern the class name for extern stdlib classes
                // The type_path.name contains the simple class name (e.g., "Channel" from "rayzor.concurrent.Channel")
                let class_name_str = &type_path.name;
                let interned_class_name = self.context.string_interner.intern(class_name_str);

                TypedExpressionKind::New {
                    class_type: actual_class_type,
                    arguments: arg_exprs,
                    type_arguments: type_args,
                    class_name: Some(interned_class_name),
                }
            }
            // Cast doesn't exist in ExprKind, remove this variant
            ExprKind::Ternary {
                cond,
                then_expr,
                else_expr,
            } => {
                let cond_expr = self.lower_expression(cond)?;
                let then_expression = self.lower_expression(then_expr)?;
                let else_expression = Some(Box::new(self.lower_expression(else_expr)?));

                TypedExpressionKind::Conditional {
                    condition: Box::new(cond_expr),
                    then_expr: Box::new(then_expression),
                    else_expr: else_expression,
                }
            }
            ExprKind::Block(block_elements) => {
                // Handle block expressions with error recovery
                let mut statements = Vec::new();
                let block_scope = self.context.enter_scope(ScopeKind::Block);

                for elem in block_elements {
                    match elem {
                        parser::BlockElement::Expr(expr) => {
                            // Check if this is a variable declaration expression
                            match &expr.kind {
                                parser::ExprKind::Var { .. } | parser::ExprKind::Final { .. } => {
                                    // Variable declaration - lower as expression and convert to statement
                                    match self.lower_expression(expr) {
                                        Ok(typed_expr) => {
                                            // Extract the declaration info to create a proper statement
                                            if let TypedExpressionKind::VarDeclarationExpr {
                                                symbol_id,
                                                var_type,
                                                initializer,
                                            } = typed_expr.kind
                                            {
                                                statements.push(TypedStatement::VarDeclaration {
                                                    symbol_id,
                                                    var_type,
                                                    initializer: Some(*initializer),
                                                    mutability: crate::tast::symbols::Mutability::Mutable,
                                                    source_location: self
                                                        .context
                                                        .span_to_location(&expr.span),
                                                });
                                            } else if let TypedExpressionKind::FinalDeclarationExpr {
                                                symbol_id,
                                                var_type,
                                                initializer,
                                            } = typed_expr.kind
                                            {
                                                statements.push(TypedStatement::VarDeclaration {
                                                    symbol_id,
                                                    var_type,
                                                    initializer: Some(*initializer),
                                                    mutability: crate::tast::symbols::Mutability::Immutable,
                                                    source_location: self
                                                        .context
                                                        .span_to_location(&expr.span),
                                                });
                                            }
                                        }
                                        Err(e) => {
                                            // Collect error and continue processing other statements
                                            self.collected_errors.push(e);
                                        }
                                    }
                                }
                                parser::ExprKind::Return(_) => {
                                    // Return expression - convert to Return statement
                                    match self.lower_expression(expr) {
                                        Ok(typed_expr) => {
                                            if let TypedExpressionKind::Return { value } =
                                                typed_expr.kind
                                            {
                                                statements.push(TypedStatement::Return {
                                                    value: value.map(|v| *v),
                                                    source_location: self
                                                        .context
                                                        .span_to_location(&expr.span),
                                                });
                                            } else {
                                                // Fallback: wrap as expression statement
                                                statements.push(TypedStatement::Expression {
                                                    expression: typed_expr,
                                                    source_location: self
                                                        .context
                                                        .span_to_location(&expr.span),
                                                });
                                            }
                                        }
                                        Err(e) => {
                                            // Collect error and continue processing other statements
                                            self.collected_errors.push(e);
                                        }
                                    }
                                }
                                _ => {
                                    // Regular expression - lower and wrap in statement
                                    match self.lower_expression(expr) {
                                        Ok(typed_expr) => {
                                            statements.push(TypedStatement::Expression {
                                                expression: typed_expr,
                                                source_location: self
                                                    .context
                                                    .span_to_location(&expr.span),
                                            });
                                        }
                                        Err(e) => {
                                            // Collect error and continue processing other statements
                                            self.collected_errors.push(e);
                                        }
                                    }
                                }
                            }
                        }
                        parser::BlockElement::Import(_)
                        | parser::BlockElement::Using(_)
                        | parser::BlockElement::Conditional(_) => {
                            // Skip imports, using statements, and conditional compilation for now
                            // These should be handled at the module level
                        }
                    }
                }

                // Leave the block scope
                let parent_scope = self
                    .context
                    .scope_tree
                    .get_scope(block_scope)
                    .and_then(|scope| scope.parent_id)
                    .unwrap_or(ScopeId::first());
                self.context.current_scope = parent_scope;

                TypedExpressionKind::Block {
                    statements,
                    scope_id: block_scope,
                }
            }
            ExprKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let cond_expr = self.lower_expression(cond)?;
                let then_expr = self.lower_expression(then_branch)?;
                let else_expr = if let Some(else_branch) = else_branch {
                    Some(Box::new(self.lower_expression(else_branch)?))
                } else {
                    None
                };

                TypedExpressionKind::Conditional {
                    condition: Box::new(cond_expr),
                    then_expr: Box::new(then_expr),
                    else_expr,
                }
            }
            ExprKind::While { cond, body } => {
                // Convert while expressions to statement form for proper CFG handling
                let cond_expr = self.lower_expression(cond)?;
                let body_stmt = self.convert_expression_to_statement(body)?;

                // Create a while statement and wrap it in a block expression
                let while_stmt = TypedStatement::While {
                    condition: cond_expr,
                    body: Box::new(body_stmt),
                    source_location: SourceLocation::unknown(),
                };

                // Return block expression containing the while statement
                TypedExpressionKind::Block {
                    statements: vec![while_stmt],
                    scope_id: ScopeId::from_raw(self.context.next_scope_id()),
                }
            }
            ExprKind::DoWhile { body, cond } => {
                // Convert do-while expressions to statement form
                let body_stmt = self.convert_expression_to_statement(body)?;
                let cond_expr = self.lower_expression(cond)?;

                // Create a do-while statement (add to TAST if missing)
                // Convert do-while to equivalent control flow:
                // { body; while(cond) { body } }
                let body_block = TypedStatement::Block {
                    statements: vec![body_stmt.clone()],
                    scope_id: ScopeId::from_raw(self.context.next_scope_id()),
                    source_location: SourceLocation::unknown(),
                };

                let while_stmt = TypedStatement::While {
                    condition: cond_expr,
                    body: Box::new(body_stmt),
                    source_location: SourceLocation::unknown(),
                };

                // Return block that executes body once, then while loop
                TypedExpressionKind::Block {
                    statements: vec![body_block, while_stmt],
                    scope_id: ScopeId::from_raw(self.context.next_scope_id()),
                }
            }
            ExprKind::For {
                var,
                key_var,
                iter,
                body,
            } => {
                return self.lower_for_expression(expression, var, key_var.as_deref(), iter, body);
            }
            ExprKind::Array(elements) => {
                let element_exprs = elements
                    .iter()
                    .map(|elem| self.lower_expression(elem))
                    .collect::<Result<Vec<_>, _>>()?;

                TypedExpressionKind::ArrayLiteral {
                    elements: element_exprs,
                }
            }
            ExprKind::Return(expr) => {
                let return_expr = if let Some(expr) = expr {
                    Some(Box::new(self.lower_expression(expr)?))
                } else {
                    None
                };

                TypedExpressionKind::Return { value: return_expr }
            }
            ExprKind::Break => TypedExpressionKind::Break,
            ExprKind::Continue => TypedExpressionKind::Continue,
            // Is doesn't exist in ExprKind, remove this variant
            ExprKind::Throw(expr) => {
                let expression = self.lower_expression(expr)?;
                TypedExpressionKind::Throw {
                    expression: Box::new(expression),
                }
            }
            ExprKind::Switch {
                expr,
                cases,
                default,
            } => {
                // Lower the discriminant expression
                let discriminant = Box::new(self.lower_expression(expr)?);

                // Store the discriminant type for use in pattern matching
                // This allows resolving enum constructor names like "Some" to "Option.Some"
                let prev_switch_type = self.context.switch_discriminant_type;
                self.context.switch_discriminant_type = Some(discriminant.expr_type);

                // Check if this is a switch expression or switch statement
                // In a switch expression, all cases must have expression values
                // In a switch statement, cases contain statements (like return)
                let is_expression = cases.iter().all(|case| {
                    // Check if the case body is a simple expression (not a block with statements)
                    !matches!(
                        &case.body.kind,
                        ExprKind::Block(_)
                            | ExprKind::Return(_)
                            | ExprKind::Break
                            | ExprKind::Continue
                            | ExprKind::Throw(_)
                    )
                });

                let result = if is_expression {
                    // Lower as switch expression
                    let mut typed_cases = Vec::with_capacity(cases.len());
                    for case in cases {
                        let typed_case = self.lower_switch_case_expression(case)?;
                        typed_cases.push(typed_case);
                    }

                    // Lower the default case if present
                    let default_case = if let Some(default_expr) = default {
                        Some(Box::new(self.lower_expression(default_expr)?))
                    } else {
                        None
                    };

                    TypedExpressionKind::Switch {
                        discriminant,
                        cases: typed_cases,
                        default_case,
                    }
                } else {
                    // Switch statement - lower as a block with switch statement
                    // For now, we'll lower it as a switch expression but mark it as void type
                    let mut typed_cases = Vec::with_capacity(cases.len());
                    for case in cases {
                        let typed_case = self.lower_switch_case(case)?;
                        typed_cases.push(typed_case);
                    }

                    // Lower the default case if present
                    let default_case = if let Some(default_expr) = default {
                        Some(Box::new(self.lower_expression(default_expr)?))
                    } else {
                        None
                    };

                    // Create a switch that returns void
                    TypedExpressionKind::Switch {
                        discriminant,
                        cases: typed_cases,
                        default_case,
                    }
                };

                // Restore previous switch type
                self.context.switch_discriminant_type = prev_switch_type;

                result
            }
            ExprKind::Try {
                expr,
                catches,
                finally_block,
            } => {
                // Lower the try expression
                let try_expr = Box::new(self.lower_expression(expr)?);

                // Lower catch clauses
                let mut catch_clauses = Vec::new();
                for catch in catches {
                    let typed_catch = self.lower_catch_clause(catch)?;
                    catch_clauses.push(typed_catch);
                }

                // Lower finally block if present
                let typed_finally = if let Some(finally_expr) = finally_block {
                    Some(Box::new(self.lower_expression(finally_expr)?))
                } else {
                    None
                };

                TypedExpressionKind::Try {
                    try_expr,
                    catch_clauses,
                    finally_block: typed_finally,
                }
            }
            ExprKind::This => {
                // Find current class context
                let this_type = if let Some(current_class) = self.context.class_context_stack.last()
                {
                    type_resolution::resolve_this_type(
                        &self.context.type_table,
                        self.context.symbol_table,
                        Some(*current_class),
                    )
                } else {
                    self.context.type_table.borrow().dynamic_type()
                };
                TypedExpressionKind::This { this_type }
            }
            ExprKind::Super => {
                // Find current class context and get super type
                let super_type =
                    if let Some(current_class) = self.context.class_context_stack.last() {
                        type_resolution::resolve_super_type(
                            &self.context.type_table,
                            self.context.symbol_table,
                            Some(*current_class),
                        )
                    } else {
                        self.context.type_table.borrow().dynamic_type()
                    };
                TypedExpressionKind::Super { super_type }
            }
            ExprKind::Map(entries) => {
                // Map literal: ["key1" => value1, "key2" => value2]
                let mut typed_entries = Vec::with_capacity(entries.len());
                for (key_expr, value_expr) in entries {
                    let key = self.lower_expression(key_expr)?;
                    let value = self.lower_expression(value_expr)?;
                    typed_entries.push(TypedMapEntry {
                        key,
                        value,
                        source_location: self.context.create_location(),
                    });
                }
                TypedExpressionKind::MapLiteral {
                    entries: typed_entries,
                }
            }
            ExprKind::Object(fields) => {
                // Object literal
                let mut typed_fields = Vec::with_capacity(fields.len());
                for field in fields {
                    let value = self.lower_expression(&field.expr)?;
                    let field_name = self.context.intern_string(&field.name);
                    typed_fields.push(TypedObjectField {
                        name: field_name,
                        value,
                        source_location: self.context.create_location(),
                    });
                }
                TypedExpressionKind::ObjectLiteral {
                    fields: typed_fields,
                }
            }
            ExprKind::StringInterpolation(parts) => {
                let mut typed_parts = Vec::new();
                for part in parts {
                    match part {
                        parser::StringPart::Literal(text) => {
                            typed_parts.push(StringInterpolationPart::String(text.clone()));
                        }
                        parser::StringPart::Interpolation(expr) => {
                            let typed_expr = self.lower_expression(expr)?;
                            typed_parts.push(StringInterpolationPart::Expression(typed_expr));
                        }
                    }
                }
                TypedExpressionKind::StringInterpolation { parts: typed_parts }
            }
            ExprKind::Paren(expr) => {
                // Parentheses just pass through the inner expression
                return self.lower_expression(expr);
            }
            ExprKind::Tuple(elements) => {
                // Standalone tuple without a known target type — desugar to array literal.
                // e.g., (1, 2, 3) becomes [1, 2, 3]
                let array_expr = parser::Expr {
                    kind: parser::ExprKind::Array(elements.clone()),
                    span: expression.span,
                };
                return self.lower_expression(&array_expr);
            }
            ExprKind::Cast { expr, type_hint } => {
                let typed_expr = self.lower_expression(expr)?;
                let (target_type, cast_kind) = if let Some(hint) = type_hint {
                    // cast(expr, Type) — safe/explicit cast
                    (self.lower_type(hint)?, CastKind::Explicit)
                } else {
                    // cast expr — unsafe cast (no type check, reinterpret)
                    (
                        self.context.type_table.borrow().dynamic_type(),
                        CastKind::Unsafe,
                    )
                };
                TypedExpressionKind::Cast {
                    expression: Box::new(typed_expr),
                    target_type,
                    cast_kind,
                }
            }
            ExprKind::TypeCheck { expr, type_hint } => {
                // (expr : Type) is a type check hint — returns the value (not a boolean).
                // It asserts at compile time that expr is compatible with Type.
                // At runtime, it acts as an implicit cast (identity for same type, coercion otherwise).
                let typed_expr = self.lower_expression(expr)?;
                let target_type = self.lower_type(type_hint)?;

                TypedExpressionKind::Cast {
                    expression: Box::new(typed_expr),
                    target_type,
                    cast_kind: CastKind::Checked,
                }
            }
            ExprKind::Function(func) => {
                // Function expression/lambda - create a new scope for the function body
                let function_scope = self.context.enter_scope(ScopeKind::Function);

                // Lower parameters - they will be automatically registered in the function scope
                let mut parameters = Vec::new();
                for param in &func.params {
                    let param_result = self.lower_function_param(param)?;
                    parameters.push(param_result);
                }

                // Lower function body in the new scope
                let body = if let Some(body_expr) = &func.body {
                    self.lower_function_body(body_expr)?
                } else {
                    Vec::new()
                };

                // Determine return type
                let return_type = if let Some(ret_type) = &func.return_type {
                    self.lower_type(ret_type)?
                } else {
                    self.context.type_table.borrow().dynamic_type()
                };

                // Exit the function scope
                self.context.exit_scope();

                TypedExpressionKind::FunctionLiteral {
                    parameters,
                    body,
                    return_type,
                }
            }
            ExprKind::Arrow { params, expr } => {
                // Arrow function: x -> x * 2 or (x:Int) -> x * 2
                let function_scope = self.context.enter_scope(ScopeKind::Function);

                let mut typed_params = Vec::new();
                for param in params {
                    let param_interned = self.context.string_interner.intern(&param.name);

                    // Use type annotation if present, otherwise fall back to dynamic
                    let param_type = if let Some(ref type_hint) = param.type_hint {
                        self.lower_type(type_hint)?
                    } else {
                        self.context.type_table.borrow().dynamic_type()
                    };

                    // Create symbol WITH the correct type so body expressions
                    // (like x * 2) resolve the variable to the right type
                    let param_symbol = self.context.symbol_table.create_variable_with_type(
                        param_interned,
                        self.context.current_scope,
                        param_type,
                    );

                    typed_params.push(TypedParameter {
                        symbol_id: param_symbol,
                        name: param_interned,
                        param_type,
                        is_optional: false,
                        default_value: None,
                        mutability: crate::tast::symbols::Mutability::Immutable,
                        source_location: self.context.span_to_location(&expression.span),
                    });
                }

                // Lower arrow body (single expression) in the new scope
                let body_expr = self.lower_expression(expr)?;
                let body = vec![TypedStatement::Expression {
                    expression: body_expr.clone(),
                    source_location: body_expr.source_location,
                }];

                // Exit the function scope
                self.context.exit_scope();

                TypedExpressionKind::FunctionLiteral {
                    parameters: typed_params,
                    body,
                    return_type: body_expr.expr_type,
                }
            }
            ExprKind::Var {
                name,
                type_hint,
                expr,
            } => {
                // Variable declaration as expression: `var x = 5` returns 5
                let var_name = self.context.intern_string(name);

                // Resolve target type FIRST for type-directed desugaring (e.g., tuples)
                let declared_type = if let Some(th) = type_hint {
                    Some(self.lower_type(th)?)
                } else {
                    None
                };

                // Check for tuple → SIMD4f.make() desugaring
                if let (Some(init_expr), Some(target_ty)) = (expr.as_ref(), declared_type) {
                    if let ExprKind::Tuple(elements) = &init_expr.kind {
                        if let Some(desugared) =
                            self.try_desugar_tuple_to_make(elements, target_ty, expression)?
                        {
                            // Successfully desugared tuple to a static method call.
                            // Wrap in VarDeclarationExpr.
                            let var_symbol = self.context.symbol_table.create_variable_with_type(
                                var_name,
                                self.context.current_scope,
                                target_ty,
                            );
                            if let Some(scope) = self
                                .context
                                .scope_tree
                                .get_scope_mut(self.context.current_scope)
                            {
                                scope.add_symbol(var_symbol, var_name);
                            }
                            return Ok(TypedExpression {
                                kind: TypedExpressionKind::VarDeclarationExpr {
                                    symbol_id: var_symbol,
                                    var_type: target_ty,
                                    initializer: Box::new(desugared),
                                },
                                expr_type: target_ty,
                                usage: VariableUsage::Copy,
                                lifetime_id: LifetimeId::from_raw(1),
                                source_location: self.context.span_to_location(&expression.span),
                                metadata: ExpressionMetadata::default(),
                            });
                        }
                    }
                }

                // Check for implicit @:from conversion (e.g., array literal → abstract type)
                // Array/object literals assigned to abstract types need an explicit Cast node
                // so the MIR Cast handler can look up and call the @:from conversion function.
                // Simple literals (int, float, string) and variables are handled by the
                // MIR Let handler's maybe_abstract_from_convert() instead.
                if let (Some(init_expr), Some(target_ty)) = (expr.as_ref(), declared_type) {
                    let needs_cast = self.is_abstract_type(target_ty)
                        && matches!(&init_expr.kind, ExprKind::Array(_) | ExprKind::Object(_));
                    if needs_cast {
                        // Lower the initializer, then wrap in an implicit cast to the abstract type
                        let array_expr = self.lower_expression(init_expr)?;
                        let cast_expr = TypedExpression {
                            kind: TypedExpressionKind::Cast {
                                expression: Box::new(array_expr),
                                target_type: target_ty,
                                cast_kind: crate::tast::node::CastKind::Implicit,
                            },
                            expr_type: target_ty,
                            usage: VariableUsage::Copy,
                            lifetime_id: LifetimeId::from_raw(1),
                            source_location: self.context.span_to_location(&expression.span),
                            metadata: ExpressionMetadata::default(),
                        };
                        let var_symbol = self.context.symbol_table.create_variable_with_type(
                            var_name,
                            self.context.current_scope,
                            target_ty,
                        );
                        if let Some(scope) = self
                            .context
                            .scope_tree
                            .get_scope_mut(self.context.current_scope)
                        {
                            scope.add_symbol(var_symbol, var_name);
                        }
                        return Ok(TypedExpression {
                            kind: TypedExpressionKind::VarDeclarationExpr {
                                symbol_id: var_symbol,
                                var_type: target_ty,
                                initializer: Box::new(cast_expr),
                            },
                            expr_type: target_ty,
                            usage: VariableUsage::Copy,
                            lifetime_id: LifetimeId::from_raw(1),
                            source_location: self.context.span_to_location(&expression.span),
                            metadata: ExpressionMetadata::default(),
                        });
                    }
                }

                // Lower initializer expression first if it exists
                let initializer = if let Some(init_expr) = expr {
                    self.lower_expression(init_expr)?
                } else {
                    // Default to null if no initializer
                    TypedExpression {
                        kind: TypedExpressionKind::Null,
                        expr_type: self.context.type_table.borrow().dynamic_type(),
                        usage: VariableUsage::Copy,
                        lifetime_id: LifetimeId::from_raw(1),
                        source_location: self.context.span_to_location(&expression.span),
                        metadata: ExpressionMetadata::default(),
                    }
                };

                // Determine variable type (use already-resolved declared_type if available)
                let var_type = if let Some(dt) = declared_type {
                    dt
                } else {
                    initializer.expr_type
                };

                // Create the variable symbol with the correct type
                let var_symbol = self.context.symbol_table.create_variable_with_type(
                    var_name,
                    self.context.current_scope,
                    var_type,
                );

                // Add the variable to the current scope so it can be resolved later
                if let Some(scope) = self
                    .context
                    .scope_tree
                    .get_scope_mut(self.context.current_scope)
                {
                    scope.add_symbol(var_symbol, var_name);
                }

                TypedExpressionKind::VarDeclarationExpr {
                    symbol_id: var_symbol,
                    var_type,
                    initializer: Box::new(initializer),
                }
            }
            ExprKind::Final {
                name,
                type_hint,
                expr,
            } => {
                // Final declaration as expression: `final x = 5` returns 5
                let var_name = self.context.intern_string(name);

                // Resolve target type FIRST for type-directed desugaring
                let declared_type = if let Some(th) = type_hint {
                    Some(self.lower_type(th)?)
                } else {
                    None
                };

                // Check for tuple → SIMD4f.make() desugaring
                if let (Some(init_expr), Some(target_ty)) = (expr.as_ref(), declared_type) {
                    if let ExprKind::Tuple(elements) = &init_expr.kind {
                        if let Some(desugared) =
                            self.try_desugar_tuple_to_make(elements, target_ty, expression)?
                        {
                            let var_symbol = self.context.symbol_table.create_variable_with_type(
                                var_name,
                                self.context.current_scope,
                                target_ty,
                            );
                            if let Some(scope) = self
                                .context
                                .scope_tree
                                .get_scope_mut(self.context.current_scope)
                            {
                                scope.add_symbol(var_symbol, var_name);
                            }
                            return Ok(TypedExpression {
                                kind: TypedExpressionKind::FinalDeclarationExpr {
                                    symbol_id: var_symbol,
                                    var_type: target_ty,
                                    initializer: Box::new(desugared),
                                },
                                expr_type: target_ty,
                                usage: VariableUsage::Copy,
                                lifetime_id: LifetimeId::from_raw(1),
                                source_location: self.context.span_to_location(&expression.span),
                                metadata: ExpressionMetadata::default(),
                            });
                        }
                    }
                }

                // Final variables must have an initializer
                let initializer = if let Some(init_expr) = expr {
                    self.lower_expression(init_expr)?
                } else {
                    return Err(LoweringError::IncompleteImplementation {
                        feature: "Final declaration without initializer".to_string(),
                        location: self.context.span_to_location(&expression.span),
                    });
                };

                // Determine variable type
                let var_type = if let Some(dt) = declared_type {
                    dt
                } else {
                    // Infer type from initializer
                    initializer.expr_type
                };

                // Create the variable symbol with the correct type
                let var_symbol = self.context.symbol_table.create_variable_with_type(
                    var_name,
                    self.context.current_scope,
                    var_type,
                );

                // Add the variable to the current scope so it can be resolved later
                if let Some(scope) = self
                    .context
                    .scope_tree
                    .get_scope_mut(self.context.current_scope)
                {
                    scope.add_symbol(var_symbol, var_name);
                }

                TypedExpressionKind::FinalDeclarationExpr {
                    symbol_id: var_symbol,
                    var_type,
                    initializer: Box::new(initializer),
                }
            }
            ExprKind::Meta { meta, expr } => {
                // Metadata annotation: @:meta expr
                let inner_expr = self.lower_expression(expr)?;

                // Convert parser metadata to typed metadata
                let typed_meta = TypedMetadata {
                    name: self.context.intern_string(&meta.name),
                    params: meta
                        .params
                        .iter()
                        .map(|param_expr| self.lower_expression(param_expr))
                        .collect::<Result<Vec<_>, _>>()?,
                    source_location: self.context.span_to_location(&meta.span),
                };

                TypedExpressionKind::Meta {
                    metadata: vec![typed_meta],
                    expression: Box::new(inner_expr),
                }
            }
            ExprKind::DollarIdent { name, arg } => {
                // Dollar identifier: $type, $v{...}, $i{...}, etc.
                let arg_expr = if let Some(arg_expr) = arg {
                    Some(Box::new(self.lower_expression(arg_expr)?))
                } else {
                    None
                };

                TypedExpressionKind::DollarIdent {
                    name: self.context.intern_string(name),
                    arg: arg_expr,
                }
            }
            ExprKind::Untyped(expr) => {
                // Untyped expression: untyped expr
                // Just lower the inner expression, the "untyped" is more of a compiler hint
                self.lower_expression(expr)?.kind
            }
            ExprKind::Macro(expr) => {
                // Macro expression: macro expr
                // Lower as macro expression in TAST
                let inner_expr = self.lower_expression(expr)?;
                let macro_name = self.context.intern_string("macro");
                let macro_symbol = self.context.symbol_table.create_variable(macro_name);
                TypedExpressionKind::MacroExpression {
                    macro_symbol,
                    arguments: vec![inner_expr],
                }
            }
            ExprKind::Inline(expr) => {
                // Inline expression: inline expr
                // The 'inline' modifier is a hint to the compiler to inline the call
                // For now, just lower the inner expression (inlining would happen in a later pass)
                self.lower_expression(expr)?.kind
            }
            ExprKind::Reify(expr) => {
                // Macro reification: $expr
                // This is similar to DollarIdent but for expressions
                let inner_expr = self.lower_expression(expr)?;
                TypedExpressionKind::DollarIdent {
                    name: self.context.intern_string("reify"),
                    arg: Some(Box::new(inner_expr)),
                }
            }
            ExprKind::ArrayComprehension { for_parts, expr } => {
                // Array comprehension: [for (i in 0...10) i * 2]
                let expr_location = self.context.span_to_location(&expression.span);
                let comprehension =
                    self.lower_array_comprehension(for_parts, expr, &expr_location)?;
                comprehension.kind
            }
            ExprKind::MapComprehension {
                for_parts,
                key,
                value,
            } => {
                // Map comprehension: [for (i in 0...10) i => i * 2]
                let expr_location = self.context.span_to_location(&expression.span);
                let comprehension =
                    self.lower_map_comprehension(for_parts, key, value, &expr_location)?;
                comprehension.kind
            }
            ExprKind::CompilerSpecific { target, code, args } => {
                // Compiler-specific code: __c__("code {0}", arg0)
                let code_expr = self.lower_expression(code)?;
                let lowered_args = args
                    .iter()
                    .filter_map(|a| self.lower_expression(a).ok())
                    .collect();
                TypedExpressionKind::CompilerSpecific {
                    target: self.context.intern_string(target),
                    code: Box::new(code_expr),
                    args: lowered_args,
                }
            }
            // For now, handle remaining expression types with placeholders
            _ => {
                // Return a placeholder expression for unhandled cases
                TypedExpressionKind::Literal {
                    value: LiteralValue::String("unhandled_expression".to_string()),
                }
            }
        };

        // Determine expression type based on kind
        let expr_type = self.infer_expression_type(&kind)?;

        // Determine ownership usage based on expression kind
        let usage = self.determine_variable_usage(&kind);

        // Assign lifetime based on expression scope and type
        let lifetime_id = self.assign_lifetime(&kind, &expr_type);

        // Analyze expression metadata
        let metadata = self.analyze_expression_metadata(&kind);

        let typed_expr = TypedExpression {
            expr_type,
            kind,
            usage,
            lifetime_id,
            source_location: self.context.span_to_location(&expression.span),
            metadata,
        };

        // // Debug switch expressions
        // match &typed_expr.kind {
        //     TypedExpressionKind::Switch { .. } => {
        //         eprintln!(
        //             "DEBUG: Created switch expression with type: {:?}",
        //             typed_expr.expr_type
        //         );
        //     }
        //     _ => {}
        // }

        Ok(typed_expr)
    }

    /// Lower a function call expression (ExprKind::Call).
    /// Extracted from lower_expression to reduce stack frame size.
    #[inline(never)]
    fn lower_call_expression(
        &mut self,
        expression: &Expr,
        expr: &Expr,
        args: &[Expr],
    ) -> LoweringResult<TypedExpression> {
        let arg_exprs = args
            .iter()
            .map(|arg| self.lower_expression(arg))
            .collect::<Result<Vec<_>, _>>()?;

        // Check if this is a method call (field access being called)
        let kind = match &expr.kind {
            ExprKind::Field {
                expr: obj_expr,
                field,
                is_optional: field_is_optional,
            } => {
                let is_optional_call = *field_is_optional;
                // Check if this is a static method call (Class.method)
                if let ExprKind::Ident(class_name) = &obj_expr.kind {
                    let class_name_interned = self.context.intern_string(class_name);

                    // Try to resolve as a class symbol
                    if let Some(symbol_id) =
                        self.resolve_symbol_in_scope_hierarchy(class_name_interned)
                    {
                        if let Some(symbol) = self.context.symbol_table.get_symbol(symbol_id) {
                            // Check if this symbol represents a class declaration (not just a variable of class type)
                            if symbol.kind == crate::tast::symbols::SymbolKind::Class {
                                // This is a class name, so this is a static method call
                                //
                                // For extern classes (Std, Math, Sys, etc.), the type_id may be invalid
                                // because they don't have concrete type definitions in the type table.
                                // In that case, use the symbol_id directly as the class_symbol.
                                let class_symbol = if symbol.type_id == TypeId::invalid() {
                                    // Extern class - use the symbol_id directly
                                    symbol_id
                                } else if let Ok(type_table) = self.context.type_table.try_borrow()
                                {
                                    // Try to get the class symbol from the type table
                                    if let Some(type_info) = type_table.get(symbol.type_id) {
                                        if let crate::tast::core::TypeKind::Class {
                                            symbol_id: ts_symbol,
                                            ..
                                        } = &type_info.kind
                                        {
                                            *ts_symbol
                                        } else {
                                            // Type exists but isn't a Class - use symbol_id as fallback
                                            symbol_id
                                        }
                                    } else {
                                        // Type not in table - use symbol_id as fallback
                                        symbol_id
                                    }
                                } else {
                                    // Can't borrow type table - use symbol_id as fallback
                                    symbol_id
                                };

                                // This is a static method call
                                let method_name = self.context.intern_string(field);

                                // Look for the method in this class
                                // 1. Check local class_methods (for classes lowered in this compilation unit)
                                // 2. Check shared symbol table's class scope (for extern classes from other units)
                                // 3. Create placeholder as last resort
                                let method_symbol = {
                                    // Strategy 1: local class_methods
                                    let from_local =
                                        self.class_methods.get(&class_symbol).and_then(|methods| {
                                            methods
                                                .iter()
                                                .find(|(name, _, _)| *name == method_name)
                                                .map(|(_, symbol, _)| *symbol)
                                        });

                                    if let Some(sym) = from_local {
                                        sym
                                    } else {
                                        // Strategy 2: shared symbol table scope lookup
                                        let from_scope = self
                                            .context
                                            .symbol_table
                                            .get_symbol(class_symbol)
                                            .map(|s| s.scope_id)
                                            .and_then(|scope_id| {
                                                self.context
                                                    .symbol_table
                                                    .lookup_symbol(scope_id, method_name)
                                                    .map(|sym| sym.id)
                                            });

                                        if let Some(sym) = from_scope {
                                            sym
                                        } else {
                                            // Strategy 3: create placeholder with qualified name
                                            let new_symbol = self
                                                .context
                                                .symbol_table
                                                .create_function(method_name);
                                            if let Some(class_sym) =
                                                self.context.symbol_table.get_symbol(class_symbol)
                                            {
                                                if let Some(class_qname) =
                                                    class_sym.qualified_name.and_then(|qn| {
                                                        self.context.string_interner.get(qn)
                                                    })
                                                {
                                                    let method_qname = format!(
                                                        "{}.{}",
                                                        class_qname,
                                                        self.context
                                                            .string_interner
                                                            .get(method_name)
                                                            .unwrap_or("")
                                                    );
                                                    let method_qname_interned =
                                                        self.context.intern_string(&method_qname);
                                                    if let Some(sym_mut) = self
                                                        .context
                                                        .symbol_table
                                                        .get_symbol_mut(new_symbol)
                                                    {
                                                        sym_mut.qualified_name =
                                                            Some(method_qname_interned);
                                                    }
                                                }
                                            }
                                            new_symbol
                                        }
                                    }
                                };

                                // Get method return type by extracting it from the Function type
                                // (must be done before arg_exprs is moved into StaticMethodCall)
                                let expr_type = if let Some(symbol) =
                                    self.context.symbol_table.get_symbol(method_symbol)
                                {
                                    let type_table = self.context.type_table.borrow();
                                    if let Some(method_type) = type_table.get(symbol.type_id) {
                                        match &method_type.kind {
                                            crate::tast::core::TypeKind::Function {
                                                params,
                                                return_type,
                                                ..
                                            } => {
                                                let ret = *return_type;
                                                let params_owned = params.clone();
                                                // If return type is a TypeParameter, infer from arguments
                                                if type_table.is_type_parameter(ret) {
                                                    let mut inferred = ret;
                                                    for (i, param_ty) in
                                                        params_owned.iter().enumerate()
                                                    {
                                                        if *param_ty == ret && i < arg_exprs.len() {
                                                            inferred = arg_exprs[i].expr_type;
                                                            break;
                                                        }
                                                    }
                                                    inferred
                                                } else if let Some(ret_info) = type_table.get(ret) {
                                                    // Check if return type is a GenericInstance with TypeParameter args
                                                    // e.g., Thread<T> from spawn<T>(fn: Void -> T): Thread<T>
                                                    if let crate::tast::core::TypeKind::GenericInstance {
                                                        base_type,
                                                        type_args: ret_type_args,
                                                        ..
                                                    } = &ret_info.kind
                                                    {
                                                        let mut subs: Vec<(TypeId, TypeId)> = Vec::new();
                                                        for ret_ta in ret_type_args.iter() {
                                                            if let Some(ta_info) = type_table.get(*ret_ta) {
                                                                if let crate::tast::core::TypeKind::TypeParameter {
                                                                    symbol_id: tp_sym,
                                                                    ..
                                                                } = &ta_info.kind
                                                                {
                                                                    for (pi, param_ty) in params_owned.iter().enumerate() {
                                                                        if pi >= arg_exprs.len() {
                                                                            continue;
                                                                        }
                                                                        if let Some(concrete) =
                                                                            Self::match_type_param_in_types(
                                                                                *tp_sym,
                                                                                *param_ty,
                                                                                arg_exprs[pi].expr_type,
                                                                                &type_table,
                                                                            )
                                                                        {
                                                                            subs.push((*ret_ta, concrete));
                                                                            break;
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }

                                                        if !subs.is_empty() {
                                                            let base_type_val = *base_type;
                                                            let new_args: Vec<TypeId> = ret_type_args
                                                                .iter()
                                                                .map(|ta| {
                                                                    subs.iter()
                                                                        .find(|(old, _)| old == ta)
                                                                        .map(|(_, new)| *new)
                                                                        .unwrap_or(*ta)
                                                                })
                                                                .collect();
                                                            drop(type_table);
                                                            self.context
                                                                .type_table
                                                                .borrow_mut()
                                                                .create_generic_instance(
                                                                    base_type_val,
                                                                    new_args,
                                                                )
                                                        } else {
                                                            ret
                                                        }
                                                    } else {
                                                        ret
                                                    }
                                                } else {
                                                    ret
                                                }
                                            }
                                            _ => symbol.type_id,
                                        }
                                    } else {
                                        symbol.type_id
                                    }
                                } else {
                                    self.context.type_table.borrow().dynamic_type()
                                };

                                let kind = TypedExpressionKind::StaticMethodCall {
                                    class_symbol,
                                    method_symbol,
                                    arguments: arg_exprs,
                                    type_arguments: Vec::new(),
                                };

                                let usage = VariableUsage::Copy;
                                let lifetime_id = self.assign_lifetime(&kind, &expr_type);
                                let metadata = self.analyze_expression_metadata(&kind);

                                // Calculate the span for the field name specifically
                                // The field appears after the object expression and a dot
                                let field_span = parser::haxe_ast::Span::new(
                                    obj_expr.span.end + 1, // +1 for the dot
                                    obj_expr.span.end + 1 + field.len(),
                                );

                                return Ok(TypedExpression {
                                    expr_type,
                                    kind,
                                    usage,
                                    lifetime_id,
                                    source_location: self.context.span_to_location(&field_span),
                                    metadata,
                                });
                            }

                            // Check if this is an enum constructor call like MyResult.Ok(42)
                            if symbol.kind == crate::tast::symbols::SymbolKind::Enum {
                                let enum_symbol = symbol_id;
                                let variant_name = self.context.intern_string(field);

                                // Look up the enum variant
                                if let Some(variants) =
                                    self.context.symbol_table.get_enum_variants(enum_symbol)
                                {
                                    for &variant_id in variants {
                                        if let Some(variant_sym) =
                                            self.context.symbol_table.get_symbol(variant_id)
                                        {
                                            if variant_sym.name == variant_name {
                                                // This is an enum constructor call
                                                // Create a func_expr representing the enum variant
                                                let mut func_expr = TypedExpression {
                                                    expr_type: variant_sym.type_id,
                                                    kind: TypedExpressionKind::Variable {
                                                        symbol_id: variant_id,
                                                    },
                                                    usage: VariableUsage::Borrow,
                                                    lifetime_id: crate::tast::LifetimeId::first(),
                                                    source_location: self.context.create_location(),
                                                    metadata: ExpressionMetadata::default(),
                                                };

                                                // Instantiate the enum constructor type for proper return type
                                                func_expr = self
                                                    .instantiate_enum_constructor_type(
                                                        variant_id, &arg_exprs, func_expr,
                                                    )?;

                                                let kind = TypedExpressionKind::FunctionCall {
                                                    function: Box::new(func_expr),
                                                    arguments: arg_exprs,
                                                    type_arguments: Vec::new(),
                                                };

                                                let expr_type =
                                                    self.infer_expression_type(&kind)?;
                                                let usage = self.determine_variable_usage(&kind);
                                                let lifetime_id =
                                                    self.assign_lifetime(&kind, &expr_type);
                                                let metadata =
                                                    self.analyze_expression_metadata(&kind);

                                                return Ok(TypedExpression {
                                                    expr_type,
                                                    kind,
                                                    usage,
                                                    lifetime_id,
                                                    source_location: self
                                                        .context
                                                        .span_to_location(&expression.span),
                                                    metadata,
                                                });
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Not a static call, proceed with instance method call
                let receiver_expr = self.lower_expression(obj_expr)?;
                let method_name = self.context.intern_string(field);

                // First, try to resolve as a regular method on the receiver
                let method_symbol = self.resolve_method_symbol(&receiver_expr, method_name);

                // Check if the resolved symbol is a placeholder (newly created function)
                // If so, try to find a static extension method from 'using' modules
                let is_placeholder = self
                    .context
                    .symbol_table
                    .get_symbol(method_symbol)
                    .map(|s| s.kind == crate::tast::symbols::SymbolKind::Function)
                    .unwrap_or(false);

                if is_placeholder {
                    // Try to find a static extension method
                    if let Some((class_symbol, static_method_symbol)) =
                        self.find_static_extension_method(method_name, receiver_expr.expr_type)
                    {
                        // Found a static extension! Convert to static method call
                        // with receiver as first argument
                        let mut new_args = vec![receiver_expr];
                        new_args.extend(arg_exprs);

                        TypedExpressionKind::StaticMethodCall {
                            class_symbol,
                            method_symbol: static_method_symbol,
                            arguments: new_args,
                            type_arguments: Vec::new(),
                        }
                    } else {
                        // No static extension found, use regular method call
                        TypedExpressionKind::MethodCall {
                            receiver: Box::new(receiver_expr),
                            method_symbol,
                            arguments: arg_exprs,
                            type_arguments: Vec::new(),
                            is_optional: is_optional_call,
                        }
                    }
                } else {
                    // Method was found on the receiver, use it
                    TypedExpressionKind::MethodCall {
                        receiver: Box::new(receiver_expr),
                        method_symbol,
                        arguments: arg_exprs,
                        type_arguments: Vec::new(),
                        is_optional: is_optional_call,
                    }
                }
            }
            _ => {
                // Regular function call
                let mut func_expr = self.lower_expression(expr)?;

                // Check if this is an enum constructor call and instantiate its type
                if let TypedExpressionKind::Variable { symbol_id } = &func_expr.kind {
                    if let Some(symbol) = self.context.symbol_table.get_symbol(*symbol_id) {
                        if symbol.kind == crate::tast::symbols::SymbolKind::EnumVariant {
                            // This is an enum constructor - instantiate its function type
                            func_expr = self.instantiate_enum_constructor_type(
                                *symbol_id, &arg_exprs, func_expr,
                            )?;
                        }
                    }
                }

                // Check if this is an unqualified call to a method on the current class.
                // In Haxe, `calculate(10, 20)` inside a class method is `this.calculate(10, 20)`,
                // and `staticMethod()` is `ClassName.staticMethod()`.
                if let TypedExpressionKind::Variable { symbol_id } = &func_expr.kind {
                    let method_info =
                        self.context
                            .class_context_stack
                            .last()
                            .and_then(|class_sym| {
                                self.class_methods.get(class_sym).and_then(|methods| {
                                    methods
                                        .iter()
                                        .find(|(_, sym, _)| *sym == *symbol_id)
                                        .map(|(_, _, is_static)| (*class_sym, *is_static))
                                })
                            });

                    if let Some((class_symbol, is_static)) = method_info {
                        let method_symbol = *symbol_id;
                        // Get the return type of the method
                        let return_type = {
                            let sym = self.context.symbol_table.get_symbol(method_symbol);
                            if let Some(sym) = sym {
                                let type_table = self.context.type_table.borrow();
                                if let Some(method_type) = type_table.get(sym.type_id) {
                                    match &method_type.kind {
                                        crate::tast::core::TypeKind::Function {
                                            params,
                                            return_type,
                                            ..
                                        } => {
                                            let ret = *return_type;
                                            // If return type is a TypeParameter, infer from arguments
                                            if type_table.is_type_parameter(ret) {
                                                let mut inferred = ret;
                                                for (i, param_ty) in params.iter().enumerate() {
                                                    if *param_ty == ret && i < arg_exprs.len() {
                                                        inferred = arg_exprs[i].expr_type;
                                                        break;
                                                    }
                                                }
                                                inferred
                                            } else {
                                                ret
                                            }
                                        }
                                        _ => sym.type_id,
                                    }
                                } else {
                                    sym.type_id
                                }
                            } else {
                                func_expr.expr_type
                            }
                        };

                        let kind = if is_static {
                            // Static methods: create StaticMethodCall with the class symbol
                            TypedExpressionKind::StaticMethodCall {
                                class_symbol,
                                method_symbol,
                                arguments: arg_exprs,
                                type_arguments: Vec::new(),
                            }
                        } else {
                            // Instance methods: create MethodCall with implicit `this` receiver
                            let this_name = self.context.intern_string("this");
                            let this_symbol = self
                                .resolve_symbol_in_scope_hierarchy(this_name)
                                .unwrap_or_else(|| {
                                    self.context.symbol_table.create_variable(this_name)
                                });
                            let this_type = self
                                .context
                                .class_context_stack
                                .last()
                                .and_then(|cs| self.context.symbol_table.get_symbol(*cs))
                                .map(|s| s.type_id)
                                .unwrap_or_else(|| self.context.type_table.borrow().dynamic_type());
                            let receiver = TypedExpression {
                                expr_type: this_type,
                                kind: TypedExpressionKind::Variable {
                                    symbol_id: this_symbol,
                                },
                                usage: VariableUsage::Copy,
                                lifetime_id: crate::tast::LifetimeId::first(),
                                source_location: self.context.create_location(),
                                metadata: ExpressionMetadata::default(),
                            };
                            TypedExpressionKind::MethodCall {
                                receiver: Box::new(receiver),
                                method_symbol,
                                arguments: arg_exprs,
                                type_arguments: Vec::new(),
                                is_optional: false,
                            }
                        };

                        let usage = VariableUsage::Copy;
                        let lifetime_id = self.assign_lifetime(&kind, &return_type);
                        let metadata = self.analyze_expression_metadata(&kind);
                        return Ok(TypedExpression {
                            expr_type: return_type,
                            kind,
                            usage,
                            lifetime_id,
                            source_location: self.context.create_location(),
                            metadata,
                        });
                    }
                }

                TypedExpressionKind::FunctionCall {
                    function: Box::new(func_expr),
                    arguments: arg_exprs,
                    type_arguments: Vec::new(),
                }
            }
        };

        // Build the TypedExpression for the non-early-return paths
        let expr_type = self.infer_expression_type(&kind)?;
        let usage = self.determine_variable_usage(&kind);
        let lifetime_id = self.assign_lifetime(&kind, &expr_type);
        let metadata = self.analyze_expression_metadata(&kind);

        Ok(TypedExpression {
            expr_type,
            kind,
            usage,
            lifetime_id,
            source_location: self.context.span_to_location(&expression.span),
            metadata,
        })
    }

    /// Lower a field access expression (ExprKind::Field).
    /// Extracted from lower_expression to reduce stack frame size.
    #[inline(never)]
    fn lower_field_expression(
        &mut self,
        expression: &Expr,
        expr: &Expr,
        field: &str,
        is_optional: bool,
    ) -> LoweringResult<TypedExpression> {
        // Helper function to extract a fully qualified path from nested Field expressions
        // For example: rayzor.concurrent.Thread -> vec!["rayzor", "concurrent", "Thread"]
        fn extract_qualified_path(expr: &parser::Expr) -> Option<Vec<String>> {
            match &expr.kind {
                ExprKind::Ident(name) => Some(vec![name.clone()]),
                ExprKind::Field {
                    expr: inner_expr,
                    field,
                    ..
                } => {
                    let mut path = extract_qualified_path(inner_expr)?;
                    path.push(field.clone());
                    Some(path)
                }
                _ => None, // Not a qualified path
            }
        }

        // Try to extract a fully qualified path (e.g., rayzor.concurrent.Thread)
        if let Some(mut path) = extract_qualified_path(expr) {
            path.push(field.to_string()); // Add the final field (e.g., "spawn")

            // Before attempting qualified type/package resolution, check if the base
            // identifier is a local variable or parameter. If so, this is a field
            // access chain (a.b.c.process()), NOT a qualified type path.
            let base_name_interned = self.context.intern_string(&path[0]);
            let base_is_local_var = self
                .resolve_symbol_in_scope_hierarchy(base_name_interned)
                .and_then(|id| self.context.symbol_table.get_symbol(id))
                .map(|sym| {
                    matches!(
                        sym.kind,
                        crate::tast::symbols::SymbolKind::Variable
                            | crate::tast::symbols::SymbolKind::Parameter
                            | crate::tast::symbols::SymbolKind::Field
                    )
                })
                .unwrap_or(false);

            // Try to resolve this as a package.Class.staticMethod pattern
            // Start from the full path and work backwards to find the class
            // Skip this if the base is a local variable (field access chain)
            for split_point in (1..if base_is_local_var { 1 } else { path.len() }).rev() {
                let package_and_class = &path[..split_point];
                let remaining = &path[split_point..];

                // Try to resolve the package+class part as a symbol
                // For "rayzor.concurrent.Thread.spawn", try:
                // - "rayzor.concurrent.Thread" (class) with "spawn" (method)
                // - "rayzor.concurrent" (class) with "Thread.spawn" (not valid, skip)
                // - "rayzor" (class) with "concurrent.Thread.spawn" (not valid, skip)

                // For static field access like rayzor.concurrent.Thread.spawn:
                // - path = ["rayzor", "concurrent", "Thread", "spawn"]
                // - When split at 2: package_and_class=["rayzor", "concurrent"], remaining=["Thread", "spawn"]
                // - Package = ["rayzor", "concurrent"]
                // - Class = remaining[0] = "Thread"
                // - Field = remaining[1] = "spawn"
                //
                // For class name access like rayzor.concurrent.Thread:
                // - path = ["rayzor", "concurrent", "Thread"]
                // - When split at 2: package_and_class=["rayzor", "concurrent"], remaining=["Thread"]
                // - Package = ["rayzor", "concurrent"]
                // - Class = remaining[0] = "Thread"
                // - Field = None (just accessing the class itself)
                if remaining.len() == 1 {
                    // Just accessing a class name (e.g., rayzor.concurrent.Thread)
                    let package_parts = package_and_class;
                    let class_name = &remaining[0];

                    let class_name_interned = self.context.intern_string(class_name);

                    // Build fully qualified class name
                    let qualified_class_name = if package_parts.is_empty() {
                        class_name.clone()
                    } else {
                        format!("{}.{}", package_parts.join("."), class_name)
                    };
                    let qualified_class_interned =
                        self.context.intern_string(&qualified_class_name);

                    // Construct QualifiedPath for namespace resolver
                    let qualified_path = {
                        let package_interned: Vec<_> = package_parts
                            .iter()
                            .map(|p| self.context.intern_string(p))
                            .collect();
                        crate::tast::namespace::QualifiedPath::new(
                            package_interned,
                            class_name_interned,
                        )
                    };

                    // Try to resolve the class
                    let symbol_id_opt = self
                        .context
                        .namespace_resolver
                        .lookup_symbol(&qualified_path)
                        .or_else(|| {
                            self.context
                                .symbol_table
                                .lookup_symbol(
                                    crate::tast::ScopeId::first(),
                                    qualified_class_interned,
                                )
                                .map(|s| s.id)
                        })
                        .or_else(|| {
                            self.resolve_symbol_in_scope_hierarchy(qualified_class_interned)
                        })
                        .or_else(|| self.resolve_symbol_in_scope_hierarchy(class_name_interned));

                    if let Some(symbol_id) = symbol_id_opt {
                        if let Some(symbol) = self.context.symbol_table.get_symbol(symbol_id) {
                            if symbol.kind == crate::tast::symbols::SymbolKind::Class
                                || symbol.kind == crate::tast::symbols::SymbolKind::Enum
                            {
                                // Return a reference to the class/enum itself
                                let class_type = symbol.type_id;
                                return Ok(TypedExpression {
                                    expr_type: class_type,
                                    kind: TypedExpressionKind::Variable { symbol_id },
                                    usage: VariableUsage::Borrow,
                                    lifetime_id: crate::tast::LifetimeId::first(),
                                    source_location: self.context.create_location(),
                                    metadata: ExpressionMetadata::default(),
                                });
                            }
                        }
                    } else if package_parts.len() >= 2
                        || (!package_parts.is_empty()
                            && matches!(
                                package_parts[0].as_str(),
                                "haxe"
                                    | "rayzor"
                                    | "sys"
                                    | "cpp"
                                    | "cs"
                                    | "java"
                                    | "python"
                                    | "lua"
                                    | "eval"
                                    | "neko"
                                    | "hl"
                                    | "flash"
                            ))
                    {
                        // Qualified class not found AND looks like a package path
                        // Either has 2+ package components OR starts with known stdlib/project package
                        // This indicates a package path like rayzor.concurrent.Thread or haxe.ds.StringMap
                        // Return UnresolvedType to trigger on-demand loading
                        return Err(LoweringError::UnresolvedType {
                            type_name: qualified_class_name.clone(),
                            location: self.context.create_location_from_span(expression.span),
                        });
                    }
                } else if remaining.len() == 2 {
                    let package_parts = package_and_class; // Full package path
                    let class_name = &remaining[0]; // Class is first element of remaining
                    let field_name = &remaining[1]; // Field is second element of remaining

                    let class_name_interned = self.context.intern_string(class_name);
                    let field_name_interned = self.context.intern_string(field_name);

                    // Build fully qualified class name for fallback lookup
                    let qualified_class_name = if package_parts.is_empty() {
                        class_name.clone()
                    } else {
                        format!("{}.{}", package_parts.join("."), class_name)
                    };
                    let qualified_class_interned =
                        self.context.intern_string(&qualified_class_name);

                    // Construct QualifiedPath for namespace resolver
                    let qualified_path = {
                        let package_interned: Vec<_> = package_parts
                            .iter()
                            .map(|p| self.context.intern_string(p))
                            .collect();
                        crate::tast::namespace::QualifiedPath::new(
                            package_interned,
                            class_name_interned,
                        )
                    };

                    // Try to resolve the class using the namespace resolver

                    let symbol_id_opt = self
                        .context
                        .namespace_resolver
                        .lookup_symbol(&qualified_path)
                        .or_else(|| {
                            // Fallback: Try to look up in root scope using full path string
                            self.context
                                .symbol_table
                                .lookup_symbol(
                                    crate::tast::ScopeId::first(), // Root scope
                                    qualified_class_interned,
                                )
                                .map(|s| s.id)
                        })
                        .or_else(|| {
                            self.resolve_symbol_in_scope_hierarchy(qualified_class_interned)
                        })
                        .or_else(|| self.resolve_symbol_in_scope_hierarchy(class_name_interned));

                    if let Some(symbol_id) = symbol_id_opt {
                        if let Some(symbol) = self.context.symbol_table.get_symbol(symbol_id) {
                            if symbol.kind == crate::tast::symbols::SymbolKind::Class {
                                // Found the class! Now look up the static field
                                {
                                    let field_info =
                                        if let Some(fields) = self.class_fields.get(&symbol_id) {
                                            fields
                                                .iter()
                                                .find(|(name, _, _)| *name == field_name_interned)
                                                .map(|(_, symbol, is_static)| (*symbol, *is_static))
                                        } else {
                                            None
                                        };

                                    if let Some((field_symbol, _is_static)) = field_info {
                                        let expr_type = if let Some(field) =
                                            self.find_field_in_class(&symbol_id, field_symbol)
                                        {
                                            field.1 // field type
                                        } else {
                                            self.context.type_table.borrow().dynamic_type()
                                        };

                                        let kind = TypedExpressionKind::StaticFieldAccess {
                                            class_symbol: symbol_id,
                                            field_symbol,
                                        };

                                        let usage = VariableUsage::Copy;
                                        let lifetime_id = self.assign_lifetime(&kind, &expr_type);
                                        let metadata = self.analyze_expression_metadata(&kind);

                                        return Ok(TypedExpression {
                                            expr_type,
                                            kind,
                                            usage,
                                            lifetime_id,
                                            source_location: self.context.create_location(),
                                            metadata,
                                        });
                                    }
                                }
                            } else if symbol.kind == crate::tast::symbols::SymbolKind::Enum {
                                // Found an enum! Look up the variant by field name
                                if let Some(variants) =
                                    self.context.symbol_table.get_enum_variants(symbol_id)
                                {
                                    for &variant_id in variants {
                                        if let Some(variant_sym) =
                                            self.context.symbol_table.get_symbol(variant_id)
                                        {
                                            if variant_sym.name == field_name_interned {
                                                let variant_type = variant_sym.type_id;
                                                let kind = TypedExpressionKind::Variable {
                                                    symbol_id: variant_id,
                                                };
                                                let usage = VariableUsage::Borrow;
                                                let lifetime_id =
                                                    self.assign_lifetime(&kind, &variant_type);
                                                let metadata =
                                                    self.analyze_expression_metadata(&kind);

                                                return Ok(TypedExpression {
                                                    expr_type: variant_type,
                                                    kind,
                                                    usage,
                                                    lifetime_id,
                                                    source_location: self
                                                        .context
                                                        .create_location_from_span(expression.span),
                                                    metadata,
                                                });
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else if package_parts.len() >= 2
                        || (!package_parts.is_empty()
                            && matches!(
                                package_parts[0].as_str(),
                                "haxe"
                                    | "rayzor"
                                    | "sys"
                                    | "cpp"
                                    | "cs"
                                    | "java"
                                    | "python"
                                    | "lua"
                                    | "eval"
                                    | "neko"
                                    | "hl"
                                    | "flash"
                            ))
                    {
                        // Qualified class not found AND looks like a package path
                        // Either has 2+ package components OR starts with known stdlib/project package
                        // This indicates a package path like rayzor.concurrent.Thread or haxe.ds.StringMap
                        // Return UnresolvedType to trigger on-demand loading
                        return Err(LoweringError::UnresolvedType {
                            type_name: qualified_class_name.clone(),
                            location: self.context.create_location_from_span(expression.span),
                        });
                    }
                }
            }
        }

        // Check if the expression is an identifier that refers to a class (static access)
        if let ExprKind::Ident(class_name) = &expr.kind {
            let class_name_interned = self.context.intern_string(class_name);

            // Try to resolve as a class or enum symbol
            if let Some(symbol_id) = self.resolve_symbol_in_scope_hierarchy(class_name_interned) {
                // Extract symbol kind to release the borrow before calling intern_string
                let symbol_kind = self
                    .context
                    .symbol_table
                    .get_symbol(symbol_id)
                    .map(|s| s.kind);

                // Check if this symbol represents a class declaration (not just a variable of class type)
                if symbol_kind == Some(crate::tast::symbols::SymbolKind::Class) {
                    // This is a class name, so this is static field access
                    let class_symbol = symbol_id;
                    let field_name = self.context.intern_string(field);

                    // Look for the field in this class and check if it's static
                    let field_info = if let Some(fields) = self.class_fields.get(&class_symbol) {
                        fields
                            .iter()
                            .find(|(name, _, _)| *name == field_name)
                            .map(|(_, symbol, is_static)| (*symbol, *is_static))
                    } else {
                        None
                    };

                    if let Some((field_symbol, _is_static)) = field_info {
                        // Create StaticFieldAccess for any Class.field syntax
                        // The type checker will validate if it's allowed
                        let expr_type = if let Some(field) =
                            self.find_field_in_class(&class_symbol, field_symbol)
                        {
                            field.1 // field type
                        } else {
                            self.context.type_table.borrow().dynamic_type()
                        };

                        let kind = TypedExpressionKind::StaticFieldAccess {
                            class_symbol,
                            field_symbol,
                        };

                        let usage = VariableUsage::Copy;
                        let lifetime_id = self.assign_lifetime(&kind, &expr_type);
                        let metadata = self.analyze_expression_metadata(&kind);

                        // Calculate the span for the field name specifically
                        // The field appears after the object expression and a dot
                        let field_span = parser::haxe_ast::Span::new(
                            expr.span.end + 1, // +1 for the dot
                            expr.span.end + 1 + field.len(),
                        );

                        return Ok(TypedExpression {
                            expr_type,
                            kind,
                            usage,
                            lifetime_id,
                            source_location: self.context.span_to_location(&field_span),
                            metadata,
                        });
                    }
                }

                // Check if this is an enum and the field is a variant
                if symbol_kind == Some(crate::tast::symbols::SymbolKind::Enum) {
                    let enum_symbol = symbol_id;
                    let variant_name = self.context.intern_string(field);

                    // Look up enum variants
                    if let Some(variants) = self.context.symbol_table.get_enum_variants(enum_symbol)
                    {
                        for &variant_id in variants {
                            if let Some(variant_sym) =
                                self.context.symbol_table.get_symbol(variant_id)
                            {
                                if variant_sym.name == variant_name {
                                    let variant_type = variant_sym.type_id;
                                    let kind = TypedExpressionKind::Variable {
                                        symbol_id: variant_id,
                                    };
                                    let usage = VariableUsage::Borrow;
                                    let lifetime_id = self.assign_lifetime(&kind, &variant_type);
                                    let metadata = self.analyze_expression_metadata(&kind);

                                    return Ok(TypedExpression {
                                        expr_type: variant_type,
                                        kind,
                                        usage,
                                        lifetime_id,
                                        source_location: self
                                            .context
                                            .create_location_from_span(expression.span),
                                        metadata,
                                    });
                                }
                            }
                        }
                    }
                }

                // Check if this is an abstract (enum abstract) and the field is a static value
                if symbol_kind == Some(crate::tast::symbols::SymbolKind::Abstract) {
                    let abstract_symbol = symbol_id;
                    let field_name = self.context.intern_string(field);

                    if let Some(fields) = self.class_fields.get(&abstract_symbol) {
                        if let Some((_, field_symbol, _)) =
                            fields.iter().find(|(name, _, _)| *name == field_name)
                        {
                            let field_symbol = *field_symbol;
                            let expr_type = self
                                .context
                                .symbol_table
                                .get_symbol(field_symbol)
                                .map(|s| s.type_id)
                                .unwrap_or_else(|| self.context.type_table.borrow().dynamic_type());

                            let kind = TypedExpressionKind::StaticFieldAccess {
                                class_symbol: abstract_symbol,
                                field_symbol,
                            };

                            let usage = VariableUsage::Copy;
                            let lifetime_id = self.assign_lifetime(&kind, &expr_type);
                            let metadata = self.analyze_expression_metadata(&kind);

                            return Ok(TypedExpression {
                                expr_type,
                                kind,
                                usage,
                                lifetime_id,
                                source_location: self
                                    .context
                                    .create_location_from_span(expression.span),
                                metadata,
                            });
                        }
                    }
                }
            }
        }

        // Not a static access, proceed with instance field access
        let obj_expr = self.lower_expression(expr)?;
        let field_name = self.context.intern_string(field);

        // Helper: look up a field or method by name in a class, checking both
        // class_fields and class_methods. Methods are tracked separately from fields,
        // so we must check both to resolve instance method calls like `obj.lock()`.
        let resolve_in_class =
            |this: &Self, class_sym: &SymbolId, name: InternedString| -> Option<SymbolId> {
                if let Some(fields) = this.class_fields.get(class_sym) {
                    if let Some((_, sym, _)) = fields.iter().find(|(n, _, _)| *n == name) {
                        return Some(*sym);
                    }
                }
                if let Some(methods) = this.class_methods.get(class_sym) {
                    if let Some((_, sym, _)) = methods.iter().find(|(n, _, _)| *n == name) {
                        return Some(*sym);
                    }
                }
                None
            };

        // For field access, we need to look up the field symbol from the object's type
        // Create type parameter with deferred constraint resolution
        // But we can try to resolve it if the object is 'this'
        let field_symbol = match &obj_expr.kind {
            TypedExpressionKind::This { this_type: _ } => {
                // If accessing field on 'this', try to find it in current class
                if let Some(class_symbol) = self.context.class_context_stack.last() {
                    resolve_in_class(self, class_symbol, field_name)
                        .unwrap_or_else(|| self.context.symbol_table.create_field(field_name))
                } else {
                    self.context.symbol_table.create_field(field_name)
                }
            }
            TypedExpressionKind::Variable { symbol_id } => {
                // If accessing field on a variable/parameter, try to resolve from its type
                if let Some(symbol) = self.context.symbol_table.get_symbol(*symbol_id) {
                    if let Some(class_symbol) = self.resolve_type_to_class_symbol(symbol.type_id) {
                        resolve_in_class(self, &class_symbol, field_name)
                            .unwrap_or_else(|| self.context.symbol_table.create_field(field_name))
                    } else {
                        // Can't resolve object type to class, create placeholder
                        self.context.symbol_table.create_field(field_name)
                    }
                } else {
                    // Object symbol not found, create placeholder
                    self.context.symbol_table.create_field(field_name)
                }
            }
            _ => {
                // For other expression kinds (chained calls, etc.), try to resolve
                // from the expression's type to find methods/fields
                let obj_type = obj_expr.expr_type;
                if let Some(class_symbol) = self.resolve_type_to_class_symbol(obj_type) {
                    resolve_in_class(self, &class_symbol, field_name)
                        .unwrap_or_else(|| self.context.symbol_table.create_field(field_name))
                } else {
                    self.context.symbol_table.create_field(field_name)
                }
            }
        };

        let kind = TypedExpressionKind::FieldAccess {
            object: Box::new(obj_expr),
            field_symbol,
            is_optional,
        };

        // Build the TypedExpression for the non-early-return path
        let expr_type = self.infer_expression_type(&kind)?;
        let usage = self.determine_variable_usage(&kind);
        let lifetime_id = self.assign_lifetime(&kind, &expr_type);
        let metadata = self.analyze_expression_metadata(&kind);

        Ok(TypedExpression {
            expr_type,
            kind,
            usage,
            lifetime_id,
            source_location: self.context.span_to_location(&expression.span),
            metadata,
        })
    }

    /// Lower a for-in loop expression (ExprKind::For).
    /// Extracted from lower_expression to reduce stack frame size.
    #[inline(never)]
    fn lower_for_expression(
        &mut self,
        expression: &Expr,
        var: &str,
        key_var: Option<&str>,
        iter: &Expr,
        body: &Expr,
    ) -> LoweringResult<TypedExpression> {
        // Check if the iterator is a range expression (0...len)
        // If so, desugar it to a while loop instead of trying to lower as iterable
        if let ExprKind::Binary {
            op: BinaryOp::Range,
            left,
            right,
        } = &iter.kind
        {
            // Desugar: for (i in start...end) { body }
            // Into: var i = start; while (i < end) { body; i++; }

            let start_expr = self.lower_expression(left)?;
            let end_expr = self.lower_expression(right)?;

            // Create the loop body scope
            let loop_body_scope_id = ScopeId::from_raw(self.context.next_scope_id());

            // Create the loop variable
            let var_name = self.context.intern_string(var);
            let int_type = self.context.type_table.borrow().int_type();
            let var_symbol = self.context.symbol_table.create_variable_with_type(
                var_name,
                loop_body_scope_id,
                int_type,
            );

            // Enter the loop body scope
            let old_scope = self.context.current_scope;
            self.context.current_scope = loop_body_scope_id;

            let body_stmt = self.convert_expression_to_statement(body)?;

            // Restore the previous scope
            self.context.current_scope = old_scope;

            // Create: var i = start
            let init_stmt = TypedStatement::VarDeclaration {
                symbol_id: var_symbol,
                var_type: int_type,
                initializer: Some(start_expr),
                source_location: SourceLocation::unknown(),
                mutability: crate::tast::Mutability::Mutable,
            };

            // Create: i < end
            let var_ref = TypedExpression {
                expr_type: int_type,
                kind: TypedExpressionKind::Variable {
                    symbol_id: var_symbol,
                },
                usage: VariableUsage::Copy,
                lifetime_id: LifetimeId::static_lifetime(),
                source_location: SourceLocation::unknown(),
                metadata: ExpressionMetadata::default(),
            };

            let condition = TypedExpression {
                expr_type: self.context.type_table.borrow().bool_type(),
                kind: TypedExpressionKind::BinaryOp {
                    left: Box::new(var_ref.clone()),
                    operator: BinaryOperator::Lt,
                    right: Box::new(end_expr),
                },
                usage: VariableUsage::Copy,
                lifetime_id: LifetimeId::static_lifetime(),
                source_location: SourceLocation::unknown(),
                metadata: ExpressionMetadata::default(),
            };

            // Create: i++
            let one_literal = TypedExpression {
                expr_type: int_type,
                kind: TypedExpressionKind::Literal {
                    value: LiteralValue::Int(1),
                },
                usage: VariableUsage::Copy,
                lifetime_id: LifetimeId::static_lifetime(),
                source_location: SourceLocation::unknown(),
                metadata: ExpressionMetadata::default(),
            };

            let increment = TypedExpression {
                expr_type: int_type,
                kind: TypedExpressionKind::BinaryOp {
                    left: Box::new(var_ref),
                    operator: BinaryOperator::AddAssign,
                    right: Box::new(one_literal),
                },
                usage: VariableUsage::Copy,
                lifetime_id: LifetimeId::static_lifetime(),
                source_location: SourceLocation::unknown(),
                metadata: ExpressionMetadata::default(),
            };

            // Create: for (var i = start; i < end; i++) { body }
            // Using TypedStatement::For separates the update from the body,
            // so `continue` properly executes i++ before jumping to condition.
            let for_stmt = TypedStatement::For {
                init: Some(Box::new(init_stmt)),
                condition: Some(condition),
                update: Some(increment),
                body: Box::new(body_stmt),
                source_location: SourceLocation::unknown(),
            };

            // Return block: { for (...) { body } }
            return Ok(TypedExpression {
                expr_type: self.context.type_table.borrow().void_type(),
                kind: TypedExpressionKind::Block {
                    statements: vec![for_stmt],
                    scope_id: ScopeId::from_raw(self.context.next_scope_id()),
                },
                usage: VariableUsage::Move,
                lifetime_id: LifetimeId::static_lifetime(),
                source_location: SourceLocation::unknown(),
                metadata: ExpressionMetadata::default(),
            });
        }

        // Not a range - check if it's an Array (we can inline the iterator)

        // Lower the iterable expression first
        let iterable_expr = self.lower_expression(iter)?;

        // Check if the iterable is an Array type - if so, we inline the iterator pattern
        // to avoid needing to compile ArrayIterator with its generic type parameters
        let is_array = {
            let type_table = self.context.type_table.borrow();
            if let Some(actual_type) = type_table.get(iterable_expr.expr_type) {
                matches!(&actual_type.kind, TypeKind::Array { .. })
            } else {
                false
            }
        };

        if is_array && key_var.is_none() {
            // INLINE ARRAY ITERATOR PATTERN:
            // for (x in arr) becomes:
            // {
            //     var _i = 0;
            //     var _len = arr.length;
            //     while (_i < _len) {
            //         var x = arr[_i];
            //         body;
            //         _i++;
            //     }
            // }

            let loop_body_scope_id = ScopeId::from_raw(self.context.next_scope_id());
            let source_location = self.context.create_location_from_span(expression.span);
            let int_type = self.context.type_table.borrow().int_type();
            let bool_type = self.context.type_table.borrow().bool_type();
            let element_type = self.infer_element_type_from_iterable(&iterable_expr);

            // Create loop variable
            let var_name = self.context.intern_string(var);
            let var_symbol = self.context.symbol_table.create_variable_with_type(
                var_name,
                loop_body_scope_id,
                element_type,
            );

            // Create internal _i counter
            let counter_name = self.context.intern_string("_i");
            let counter_symbol = self.context.symbol_table.create_variable_with_type(
                counter_name,
                loop_body_scope_id,
                int_type,
            );

            // Create internal _len variable
            let len_name = self.context.intern_string("_len");
            let len_symbol = self.context.symbol_table.create_variable_with_type(
                len_name,
                loop_body_scope_id,
                int_type,
            );

            // var _i = 0
            let zero_literal = TypedExpression {
                expr_type: int_type,
                kind: TypedExpressionKind::Literal {
                    value: LiteralValue::Int(0),
                },
                usage: VariableUsage::Copy,
                lifetime_id: LifetimeId::static_lifetime(),
                source_location,
                metadata: ExpressionMetadata::default(),
            };
            let counter_init = TypedStatement::VarDeclaration {
                symbol_id: counter_symbol,
                var_type: int_type,
                initializer: Some(zero_literal),
                source_location,
                mutability: crate::tast::Mutability::Mutable,
            };

            // var _len = arr.length (field access)
            // Create a symbol for the field access. The name "length" will be used during
            // HIR->MIR lowering to look up the stdlib runtime function (haxe_array_length)
            let length_name = self.context.intern_string("length");
            let length_symbol = self.context.symbol_table.create_variable(length_name);

            let length_access = TypedExpression {
                expr_type: int_type,
                kind: TypedExpressionKind::FieldAccess {
                    object: Box::new(iterable_expr.clone()),
                    field_symbol: length_symbol,
                    is_optional: false,
                },
                usage: VariableUsage::Copy,
                lifetime_id: LifetimeId::static_lifetime(),
                source_location,
                metadata: ExpressionMetadata::default(),
            };
            let len_init = TypedStatement::VarDeclaration {
                symbol_id: len_symbol,
                var_type: int_type,
                initializer: Some(length_access),
                source_location,
                mutability: crate::tast::Mutability::Immutable,
            };

            // _i < _len
            let counter_ref = TypedExpression {
                expr_type: int_type,
                kind: TypedExpressionKind::Variable {
                    symbol_id: counter_symbol,
                },
                usage: VariableUsage::Copy,
                lifetime_id: LifetimeId::static_lifetime(),
                source_location,
                metadata: ExpressionMetadata::default(),
            };
            let len_ref = TypedExpression {
                expr_type: int_type,
                kind: TypedExpressionKind::Variable {
                    symbol_id: len_symbol,
                },
                usage: VariableUsage::Copy,
                lifetime_id: LifetimeId::static_lifetime(),
                source_location,
                metadata: ExpressionMetadata::default(),
            };
            let condition = TypedExpression {
                expr_type: bool_type,
                kind: TypedExpressionKind::BinaryOp {
                    left: Box::new(counter_ref.clone()),
                    operator: BinaryOperator::Lt,
                    right: Box::new(len_ref),
                },
                usage: VariableUsage::Copy,
                lifetime_id: LifetimeId::static_lifetime(),
                source_location,
                metadata: ExpressionMetadata::default(),
            };

            // var x = arr[_i]
            let array_access = TypedExpression {
                expr_type: element_type,
                kind: TypedExpressionKind::ArrayAccess {
                    array: Box::new(iterable_expr),
                    index: Box::new(counter_ref.clone()),
                },
                usage: VariableUsage::Copy,
                lifetime_id: LifetimeId::static_lifetime(),
                source_location,
                metadata: ExpressionMetadata::default(),
            };
            let var_decl = TypedStatement::VarDeclaration {
                symbol_id: var_symbol,
                var_type: element_type,
                initializer: Some(array_access),
                source_location,
                mutability: crate::tast::Mutability::Immutable,
            };

            // Convert body
            let old_scope = self.context.current_scope;
            self.context.current_scope = loop_body_scope_id;
            let body_stmt = self.convert_expression_to_statement(body)?;
            self.context.current_scope = old_scope;

            // _i++
            let one_literal = TypedExpression {
                expr_type: int_type,
                kind: TypedExpressionKind::Literal {
                    value: LiteralValue::Int(1),
                },
                usage: VariableUsage::Copy,
                lifetime_id: LifetimeId::static_lifetime(),
                source_location,
                metadata: ExpressionMetadata::default(),
            };
            let increment = TypedExpression {
                expr_type: int_type,
                kind: TypedExpressionKind::BinaryOp {
                    left: Box::new(counter_ref),
                    operator: BinaryOperator::AddAssign,
                    right: Box::new(one_literal),
                },
                usage: VariableUsage::Copy,
                lifetime_id: LifetimeId::static_lifetime(),
                source_location,
                metadata: ExpressionMetadata::default(),
            };
            let increment_stmt = TypedStatement::Expression {
                expression: increment,
                source_location,
            };

            // while body: { var x = arr[_i]; body; _i++ }
            let while_body = TypedStatement::Block {
                statements: vec![var_decl, body_stmt, increment_stmt],
                scope_id: loop_body_scope_id,
                source_location,
            };

            // while (_i < _len) { ... }
            let while_stmt = TypedStatement::While {
                condition,
                body: Box::new(while_body),
                source_location,
            };

            // Return block: { var _i = 0; var _len = arr.length; while (...) }
            return Ok(TypedExpression {
                expr_type: self.context.type_table.borrow().void_type(),
                kind: TypedExpressionKind::Block {
                    statements: vec![counter_init, len_init, while_stmt],
                    scope_id: ScopeId::from_raw(self.context.next_scope_id()),
                },
                usage: VariableUsage::Move,
                lifetime_id: LifetimeId::static_lifetime(),
                source_location,
                metadata: ExpressionMetadata::default(),
            });
        }

        // For Map types, emit a ForIn statement that passes through to MIR level
        // where keys_to_array + array iteration handles it
        let is_map = {
            let type_table = self.context.type_table.borrow();
            if let Some(actual_type) = type_table.get(iterable_expr.expr_type) {
                matches!(&actual_type.kind, TypeKind::Map { .. })
            } else {
                false
            }
        };

        if is_map && key_var.is_none() {
            let loop_body_scope_id = ScopeId::from_raw(self.context.next_scope_id());
            let source_location = self.context.create_location_from_span(expression.span);
            // For Map for-in, iterate over keys (not values)
            let element_type = {
                let type_table = self.context.type_table.borrow();
                if let Some(actual_type) = type_table.get(iterable_expr.expr_type) {
                    if let TypeKind::Map { key_type, .. } = &actual_type.kind {
                        *key_type
                    } else {
                        self.infer_element_type_from_iterable(&iterable_expr)
                    }
                } else {
                    self.infer_element_type_from_iterable(&iterable_expr)
                }
            };
            let var_name = self.context.intern_string(var);
            let var_symbol = self.context.symbol_table.create_variable_with_type(
                var_name,
                loop_body_scope_id,
                element_type,
            );

            let old_scope = self.context.current_scope;
            self.context.current_scope = loop_body_scope_id;
            let body_stmt = self.convert_expression_to_statement(body)?;
            self.context.current_scope = old_scope;

            let for_in_stmt = TypedStatement::ForIn {
                value_var: var_symbol,
                key_var: None,
                iterable: iterable_expr,
                body: Box::new(body_stmt),
                source_location,
            };

            return Ok(TypedExpression {
                expr_type: self.context.type_table.borrow().void_type(),
                kind: TypedExpressionKind::Block {
                    statements: vec![for_in_stmt],
                    scope_id: loop_body_scope_id,
                },
                usage: VariableUsage::Move,
                lifetime_id: LifetimeId::static_lifetime(),
                source_location,
                metadata: ExpressionMetadata::default(),
            });
        }

        // For non-Array, non-Map types, use the generic iterator pattern (requires iterator classes)
        let _iterator_type_name = self.infer_iterator_type_name(&iterable_expr.expr_type);

        // Validate that the iterable type is actually iterable
        {
            let type_table = self.context.type_table.borrow();
            if let Some(actual_type) = type_table.get(iterable_expr.expr_type) {
                use crate::tast::core::TypeKind;
                let is_iterable = matches!(
                    &actual_type.kind,
                    TypeKind::Array { .. }
                        | TypeKind::Map { .. }
                        | TypeKind::String
                        | TypeKind::Dynamic
                        | TypeKind::Class { .. }
                        | TypeKind::Interface { .. }
                );
                if !is_iterable {
                    drop(type_table);
                    self.context.add_error(LoweringError::TypeInferenceError {
                        expression: "Type is not iterable".to_string(),
                        location: iterable_expr.source_location,
                    });
                }
            }
        }

        // Create the loop body scope
        let loop_body_scope_id = ScopeId::from_raw(self.context.next_scope_id());

        // Create the loop variable
        let var_name = self.context.intern_string(var);
        let element_type = self.infer_element_type_from_iterable(&iterable_expr);
        let var_symbol = self.context.symbol_table.create_variable_with_type(
            var_name,
            loop_body_scope_id,
            element_type,
        );

        // Handle optional key variable for key-value iteration
        let key_symbol = if let Some(key) = key_var {
            let key_name = self.context.intern_string(key);
            let int_type = self.context.type_table.borrow().int_type();
            Some(self.context.symbol_table.create_variable_with_type(
                key_name,
                loop_body_scope_id,
                int_type,
            ))
        } else {
            None
        };

        // Enter the loop body scope
        let old_scope = self.context.current_scope;
        self.context.current_scope = loop_body_scope_id;

        let body_stmt = self.convert_expression_to_statement(body)?;

        // Restore the previous scope
        self.context.current_scope = old_scope;

        // Use the existing convert_for_in_to_c_style_for function for generic iterators
        let for_stmt = self.convert_for_in_to_c_style_for(
            var_symbol,
            key_symbol,
            iterable_expr,
            body_stmt,
            loop_body_scope_id,
            self.context.create_location_from_span(expression.span),
        )?;

        // Wrap in an expression
        Ok(TypedExpression {
            expr_type: self.context.type_table.borrow().void_type(),
            kind: TypedExpressionKind::Block {
                statements: vec![for_stmt],
                scope_id: ScopeId::from_raw(self.context.next_scope_id()),
            },
            usage: VariableUsage::Move,
            lifetime_id: LifetimeId::static_lifetime(),
            source_location: self.context.create_location_from_span(expression.span),
            metadata: ExpressionMetadata::default(),
        })
    }

    /// Lower array comprehension: [for (i in 0...10) i * 2]
    fn lower_array_comprehension(
        &mut self,
        for_parts: &[parser::ComprehensionFor],
        expr: &Expr,
        location: &SourceLocation,
    ) -> LoweringResult<TypedExpression> {
        // Create a new scope for the comprehension
        let comprehension_scope = self
            .context
            .scope_tree
            .create_scope(Some(self.context.current_scope));

        let previous_scope = self.context.current_scope;
        self.context.current_scope = comprehension_scope;

        // Lower all for parts
        let mut typed_for_parts = Vec::new();

        for for_part in for_parts {
            // Lower the iterator expression first
            let typed_iterator = self.lower_expression(&for_part.iter)?;

            // Determine the element type from the iterator
            let (element_type, key_type) = self.infer_iterator_types(&typed_iterator)?;

            // Create symbol for the loop variable
            let var_name = self.context.intern_string(&for_part.var);
            let var_symbol = self.context.symbol_table.create_variable_with_type(
                var_name,
                comprehension_scope,
                element_type,
            );

            // Handle optional key variable for key-value iteration
            let key_var_symbol = if let Some(key_var) = &for_part.key_var {
                let key_type =
                    key_type.unwrap_or_else(|| self.context.type_table.borrow().int_type());
                let key_name = self.context.intern_string(key_var);
                let symbol = self.context.symbol_table.create_variable_with_type(
                    key_name,
                    comprehension_scope,
                    key_type,
                );
                Some(symbol)
            } else {
                None
            };

            typed_for_parts.push(TypedComprehensionFor {
                var_symbol,
                key_var_symbol,
                iterator: typed_iterator,
                var_type: element_type,
                key_type,
                scope_id: comprehension_scope,
                source_location: location.clone(),
            });
        }

        // Lower the expression in the comprehension scope
        let typed_expr = self.lower_expression(expr)?;
        let element_type = typed_expr.expr_type;

        // Restore the previous scope
        self.context.current_scope = previous_scope;

        // Create the array type
        let array_type = self
            .context
            .type_table
            .borrow_mut()
            .create_array_type(element_type);

        Ok(TypedExpression {
            expr_type: array_type,
            kind: TypedExpressionKind::ArrayComprehension {
                for_parts: typed_for_parts,
                expression: Box::new(typed_expr),
                element_type,
            },
            usage: VariableUsage::Copy,
            lifetime_id: crate::tast::LifetimeId::first(),
            source_location: location.clone(),
            metadata: ExpressionMetadata::default(),
        })
    }

    /// Lower map comprehension: [for (i in 0...10) i => i * 2]
    fn lower_map_comprehension(
        &mut self,
        for_parts: &[parser::ComprehensionFor],
        key: &Expr,
        value: &Expr,
        location: &SourceLocation,
    ) -> LoweringResult<TypedExpression> {
        // Create a new scope for the comprehension
        let comprehension_scope = self
            .context
            .scope_tree
            .create_scope(Some(self.context.current_scope));

        let previous_scope = self.context.current_scope;
        self.context.current_scope = comprehension_scope;

        // Lower all for parts
        let mut typed_for_parts = Vec::new();

        for for_part in for_parts {
            // Lower the iterator expression first
            let typed_iterator = self.lower_expression(&for_part.iter)?;

            // Determine the element type from the iterator
            let (element_type, key_type) = self.infer_iterator_types(&typed_iterator)?;

            // Create symbol for the loop variable
            let var_name = self.context.intern_string(&for_part.var);
            let var_symbol = self.context.symbol_table.create_variable_with_type(
                var_name,
                comprehension_scope,
                element_type,
            );

            // Handle optional key variable for key-value iteration
            let key_var_symbol = if let Some(key_var) = &for_part.key_var {
                let key_type =
                    key_type.unwrap_or_else(|| self.context.type_table.borrow().int_type());
                let key_name = self.context.intern_string(key_var);
                let symbol = self.context.symbol_table.create_variable_with_type(
                    key_name,
                    comprehension_scope,
                    key_type,
                );
                Some(symbol)
            } else {
                None
            };

            typed_for_parts.push(TypedComprehensionFor {
                var_symbol,
                key_var_symbol,
                iterator: typed_iterator,
                var_type: element_type,
                key_type,
                scope_id: comprehension_scope,
                source_location: location.clone(),
            });
        }

        // Lower the key and value expressions in the comprehension scope
        let typed_key = self.lower_expression(key)?;
        let typed_value = self.lower_expression(value)?;

        let key_type = typed_key.expr_type;
        let value_type = typed_value.expr_type;

        // Restore the previous scope
        self.context.current_scope = previous_scope;

        // Create the map type
        let map_type = self
            .context
            .type_table
            .borrow_mut()
            .create_map_type(key_type, value_type);

        Ok(TypedExpression {
            expr_type: map_type,
            kind: TypedExpressionKind::MapComprehension {
                for_parts: typed_for_parts,
                key_expr: Box::new(typed_key),
                value_expr: Box::new(typed_value),
                key_type,
                value_type,
            },
            usage: VariableUsage::Copy,
            lifetime_id: crate::tast::LifetimeId::first(),
            source_location: location.clone(),
            metadata: ExpressionMetadata::default(),
        })
    }

    /// Infer the element and optional key types from an iterator expression
    fn infer_iterator_types(
        &mut self,
        iterator: &TypedExpression,
    ) -> LoweringResult<(TypeId, Option<TypeId>)> {
        let type_table = self.context.type_table.borrow();

        match type_table.get(iterator.expr_type) {
            Some(iter_type) => match &iter_type.kind {
                // Array<T> -> element type T, no key type
                crate::tast::core::TypeKind::Array { element_type } => Ok((*element_type, None)),
                // Map<K, V> -> value type V, key type K
                crate::tast::core::TypeKind::Map {
                    key_type,
                    value_type,
                } => Ok((*value_type, Some(*key_type))),
                // String -> Char elements, Int keys
                _ if iter_type.kind == crate::tast::core::TypeKind::String => {
                    // In Haxe, iterating over strings yields characters
                    let int_type = type_table.int_type();
                    drop(type_table);
                    let char_type = self
                        .context
                        .type_table
                        .borrow_mut()
                        .create_type(crate::tast::core::TypeKind::Char);
                    Ok((char_type, Some(int_type)))
                }
                // IntIterator (from range expressions like 0...10)
                _ if self.is_int_iterator_type(iterator.expr_type) => {
                    Ok((type_table.int_type(), None))
                }
                // Dynamic or unknown - default to dynamic element type
                _ => Ok((type_table.dynamic_type(), None)),
            },
            None => Ok((type_table.dynamic_type(), None)),
        }
    }

    /// Check if a type is an integer iterator (from range expressions)
    fn is_int_iterator_type(&self, type_id: TypeId) -> bool {
        // Check if this is an IntIterator type
        if let Some(type_info) = self.context.type_table.borrow().get(type_id) {
            match &type_info.kind {
                TypeKind::Class { symbol_id, .. } => {
                    // Check if the class is IntIterator
                    if let Some(class_symbol) = self.context.symbol_table.get_symbol(*symbol_id) {
                        let int_iterator_name = self.context.string_interner.intern("IntIterator");
                        return class_symbol.name == int_iterator_name;
                    }
                }
                TypeKind::Dynamic => {
                    // Dynamic type could be an iterator at runtime
                    return true;
                }
                _ => {}
            }
        }
        false
    }

    /// Lower a literal
    fn lower_literal(&mut self, literal: &parser::StringPart) -> LoweringResult<LiteralValue> {
        match literal {
            parser::StringPart::Literal(text) => Ok(LiteralValue::String(text.clone())),
            parser::StringPart::Interpolation(expr) => {
                // String interpolation expressions should not be converted to literals
                // They should be handled as part of StringInterpolation expression
                Err(LoweringError::InternalError {
                    message: "String interpolation part cannot be converted to literal value"
                        .to_string(),
                    location: self.context.create_location_from_span(expr.span),
                })
            }
        }
    }

    /// Lower a binary operator
    fn lower_binary_operator(&mut self, operator: &BinaryOp) -> LoweringResult<BinaryOperator> {
        match operator {
            BinaryOp::Add => Ok(BinaryOperator::Add),
            BinaryOp::Sub => Ok(BinaryOperator::Sub),
            BinaryOp::Mul => Ok(BinaryOperator::Mul),
            BinaryOp::Div => Ok(BinaryOperator::Div),
            BinaryOp::Mod => Ok(BinaryOperator::Mod),
            BinaryOp::Eq => Ok(BinaryOperator::Eq),
            BinaryOp::NotEq => Ok(BinaryOperator::Ne),
            BinaryOp::Lt => Ok(BinaryOperator::Lt),
            BinaryOp::Le => Ok(BinaryOperator::Le),
            BinaryOp::Gt => Ok(BinaryOperator::Gt),
            BinaryOp::Ge => Ok(BinaryOperator::Ge),
            BinaryOp::And => Ok(BinaryOperator::And),
            BinaryOp::Or => Ok(BinaryOperator::Or),
            BinaryOp::BitAnd => Ok(BinaryOperator::BitAnd),
            BinaryOp::BitOr => Ok(BinaryOperator::BitOr),
            BinaryOp::BitXor => Ok(BinaryOperator::BitXor),
            BinaryOp::Shl => Ok(BinaryOperator::Shl),
            BinaryOp::Shr => Ok(BinaryOperator::Shr),
            BinaryOp::Ushr => Ok(BinaryOperator::Ushr),
            BinaryOp::Range => Ok(BinaryOperator::Range),
            BinaryOp::Arrow => Ok(BinaryOperator::Arrow),
            BinaryOp::Is => {
                // 'is' operator needs runtime type checking support
                // For now, lower as a comparison (downstream passes handle it)
                Ok(BinaryOperator::Eq)
            }
            BinaryOp::NullCoal => Ok(BinaryOperator::NullCoal),
        }
    }

    /// Lower a unary operator
    fn lower_unary_operator(&mut self, operator: &UnaryOp) -> LoweringResult<UnaryOperator> {
        match operator {
            UnaryOp::Neg => Ok(UnaryOperator::Neg),
            UnaryOp::Not => Ok(UnaryOperator::Not),
            UnaryOp::BitNot => Ok(UnaryOperator::BitNot),
            UnaryOp::PreIncr => Ok(UnaryOperator::PreInc),
            UnaryOp::PostIncr => Ok(UnaryOperator::PostInc),
            UnaryOp::PreDecr => Ok(UnaryOperator::PreDec),
            UnaryOp::PostDecr => Ok(UnaryOperator::PostDec),
        }
    }

    // Property access handling removed for simplicity

    /// Get the errors collected during lowering
    pub fn get_errors(&self) -> &[LoweringError] {
        &self.context.errors
    }

    /// Check if a type might reference type parameters that aren't defined yet
    fn type_might_reference_undefined_params(&mut self, type_annotation: &Type) -> bool {
        match type_annotation {
            Type::Path { path, params, .. } => {
                let name = if path.package.is_empty() {
                    &path.name
                } else {
                    return false; // Qualified paths are not type parameters
                };

                // Check if any type arguments contain references to type parameters
                // For example, in Sortable<T>, the T might not be defined yet
                for param in params {
                    if self.type_might_reference_undefined_params(param) {
                        return true;
                    }

                    // Check if this is a simple type parameter reference
                    if let Type::Path {
                        path: param_path,
                        params: param_params,
                        ..
                    } = param
                    {
                        if param_path.package.is_empty() && param_params.is_empty() {
                            // This looks like a type parameter reference (e.g., T)
                            // Check if it's NOT a built-in type
                            if self.resolve_builtin_type(&param_path.name).is_none() {
                                // It's not a built-in, so it might be a type parameter
                                // Check if it's already defined
                                let interned_param_name =
                                    self.context.intern_string(&param_path.name);
                                if self
                                    .context
                                    .resolve_type_parameter(interned_param_name)
                                    .is_none()
                                {
                                    return true;
                                }
                            }
                        }
                    }
                }

                // Check if the base type exists and can be resolved
                // Only defer if we can't resolve the base type or if it has arity issues
                if !params.is_empty() {
                    // Try to resolve the base type to see if it exists
                    let base_type_name = if path.package.is_empty() {
                        &path.name
                    } else {
                        // Qualified names should be resolvable - don't defer
                        return false;
                    };

                    // Check if this is a known interface/class that can accept type parameters
                    // Try to find the type in the symbol table without interning a new string
                    // We can check for common interface names directly
                    match base_type_name.as_str() {
                        "Comparable" | "Iterable" | "Iterator" | "Array" | "Map" => {
                            // These are well-known generic types, don't defer
                            return false;
                        }
                        _ => {
                            // For other types, be conservative and defer for now
                            // This could be improved with better symbol resolution
                            return true;
                        }
                    }
                }

                false
            }
            Type::Function { params, ret, .. } => {
                // Check function parameter and return types
                params
                    .iter()
                    .any(|p| self.type_might_reference_undefined_params(p))
                    || self.type_might_reference_undefined_params(ret)
            }
            Type::Anonymous { fields, .. } => {
                // Check anonymous type fields
                fields
                    .iter()
                    .any(|f| self.type_might_reference_undefined_params(&f.type_hint))
            }
            Type::Optional { inner, .. } => self.type_might_reference_undefined_params(inner),
            Type::Parenthesis { inner, .. } => self.type_might_reference_undefined_params(inner),
            Type::Intersection { left, right, .. } => {
                self.type_might_reference_undefined_params(left)
                    || self.type_might_reference_undefined_params(right)
            }
            Type::Wildcard { .. } => false,
        }
    }

    /// Convert an expression to a statement for proper CFG handling
    fn convert_expression_to_statement(&mut self, expr: &Expr) -> LoweringResult<TypedStatement> {
        let typed_expr = self.lower_expression(expr)?;

        // Wrap expression in an expression statement
        Ok(TypedStatement::Expression {
            expression: typed_expr,
            source_location: SourceLocation::unknown(),
        })
    }

    /// Convert for-in loop to iterator-based while loop for TAST compatibility
    ///
    /// Haxe for-in loops use the iterator pattern:
    /// ```haxe
    /// for (x in iterable) { body }
    /// // becomes:
    /// var __iter = iterable.iterator();
    /// while (__iter.hasNext()) {
    ///     var x = __iter.next();
    ///     body;
    /// }
    /// ```
    fn convert_for_in_to_c_style_for(
        &mut self,
        variable: SymbolId,
        key_variable: Option<SymbolId>,
        iterable: TypedExpression,
        body: TypedStatement,
        loop_body_scope_id: ScopeId,
        source_location: SourceLocation,
    ) -> LoweringResult<TypedStatement> {
        // Infer element type from the iterable
        let element_type = self.infer_element_type_from_iterable(&iterable);
        let bool_type = self.context.type_table.borrow().bool_type();
        let int_type = self.context.type_table.borrow().int_type();

        // Create iterator type (Dynamic for now, should be Iterator<T> or KeyValueIterator<K,V>)
        let iter_type = self.context.type_table.borrow().dynamic_type();

        // Create: var __iter = iterable.iterator() or iterable.keyValueIterator()
        let iter_str = self.context.intern_string("__iter");
        let iterator_symbol = self.context.symbol_table.create_variable(iter_str);

        // Choose iterator method based on whether we have key-value iteration
        let iterator_method_name = if key_variable.is_some() {
            self.context.intern_string("keyValueIterator")
        } else {
            self.context.intern_string("iterator")
        };
        let iterator_method_symbol = self
            .context
            .symbol_table
            .create_variable(iterator_method_name);

        let iterator_call = TypedExpression {
            expr_type: iter_type,
            kind: TypedExpressionKind::MethodCall {
                receiver: Box::new(iterable),
                method_symbol: iterator_method_symbol,
                arguments: vec![],
                type_arguments: vec![],
                is_optional: false,
            },
            usage: VariableUsage::Move,
            lifetime_id: LifetimeId::static_lifetime(),
            source_location,
            metadata: ExpressionMetadata::default(),
        };

        let init_stmt = TypedStatement::VarDeclaration {
            symbol_id: iterator_symbol,
            var_type: iter_type,
            initializer: Some(iterator_call),
            source_location,
            mutability: crate::tast::Mutability::Mutable,
        };

        // Create iterator variable reference
        let iterator_var = TypedExpression {
            expr_type: iter_type,
            kind: TypedExpressionKind::Variable {
                symbol_id: iterator_symbol,
            },
            usage: VariableUsage::Copy,
            lifetime_id: LifetimeId::static_lifetime(),
            source_location,
            metadata: ExpressionMetadata::default(),
        };

        // Create condition: __iter.hasNext()
        let has_next_name = self.context.intern_string("hasNext");
        let has_next_symbol = self.context.symbol_table.create_variable(has_next_name);
        let condition = TypedExpression {
            expr_type: bool_type,
            kind: TypedExpressionKind::MethodCall {
                receiver: Box::new(iterator_var.clone()),
                method_symbol: has_next_symbol,
                arguments: vec![],
                type_arguments: vec![],
                is_optional: false,
            },
            usage: VariableUsage::Copy,
            lifetime_id: LifetimeId::static_lifetime(),
            source_location,
            metadata: ExpressionMetadata::default(),
        };

        // Create loop body statements
        let mut loop_statements = Vec::new();

        if let Some(key_sym) = key_variable {
            // Key-value iteration: for (key => value in iterable)
            // var __pair = __iter.next();
            // var key = __pair.key;
            // var value = __pair.value;

            let pair_type = self.context.type_table.borrow().dynamic_type();
            let pair_str = self.context.intern_string("__pair");
            let pair_symbol = self.context.symbol_table.create_variable(pair_str);

            let next_name = self.context.intern_string("next");
            let next_symbol = self.context.symbol_table.create_variable(next_name);
            let next_call = TypedExpression {
                expr_type: pair_type,
                kind: TypedExpressionKind::MethodCall {
                    receiver: Box::new(iterator_var),
                    method_symbol: next_symbol,
                    arguments: vec![],
                    type_arguments: vec![],
                    is_optional: false,
                },
                usage: VariableUsage::Move,
                lifetime_id: LifetimeId::static_lifetime(),
                source_location,
                metadata: ExpressionMetadata::default(),
            };

            let pair_decl = TypedStatement::VarDeclaration {
                symbol_id: pair_symbol,
                var_type: pair_type,
                initializer: Some(next_call),
                source_location,
                mutability: crate::tast::Mutability::Mutable,
            };
            loop_statements.push(pair_decl);

            // Create pair variable reference
            let pair_var = TypedExpression {
                expr_type: pair_type,
                kind: TypedExpressionKind::Variable {
                    symbol_id: pair_symbol,
                },
                usage: VariableUsage::Copy,
                lifetime_id: LifetimeId::static_lifetime(),
                source_location,
                metadata: ExpressionMetadata::default(),
            };

            // var key = __pair.key
            let key_field_name = self.context.intern_string("key");
            let key_field_symbol = self.context.symbol_table.create_variable(key_field_name);
            let key_access = TypedExpression {
                expr_type: int_type, // Keys are typically Int for arrays, could be other types for maps
                kind: TypedExpressionKind::FieldAccess {
                    object: Box::new(pair_var.clone()),
                    field_symbol: key_field_symbol,
                    is_optional: false,
                },
                usage: VariableUsage::Copy,
                lifetime_id: LifetimeId::static_lifetime(),
                source_location,
                metadata: ExpressionMetadata::default(),
            };

            let key_decl = TypedStatement::VarDeclaration {
                symbol_id: key_sym,
                var_type: int_type,
                initializer: Some(key_access),
                source_location,
                mutability: crate::tast::Mutability::Mutable,
            };
            loop_statements.push(key_decl);

            // var value = __pair.value
            let value_field_name = self.context.intern_string("value");
            let value_field_symbol = self.context.symbol_table.create_variable(value_field_name);
            let value_access = TypedExpression {
                expr_type: element_type,
                kind: TypedExpressionKind::FieldAccess {
                    object: Box::new(pair_var),
                    field_symbol: value_field_symbol,
                    is_optional: false,
                },
                usage: VariableUsage::Copy,
                lifetime_id: LifetimeId::static_lifetime(),
                source_location,
                metadata: ExpressionMetadata::default(),
            };

            let value_decl = TypedStatement::VarDeclaration {
                symbol_id: variable,
                var_type: element_type,
                initializer: Some(value_access),
                source_location,
                mutability: crate::tast::Mutability::Mutable,
            };
            loop_statements.push(value_decl);
        } else {
            // Simple iteration: for (value in iterable)
            // var value = __iter.next()
            let next_name = self.context.intern_string("next");
            let next_symbol = self.context.symbol_table.create_variable(next_name);
            let next_call = TypedExpression {
                expr_type: element_type,
                kind: TypedExpressionKind::MethodCall {
                    receiver: Box::new(iterator_var),
                    method_symbol: next_symbol,
                    arguments: vec![],
                    type_arguments: vec![],
                    is_optional: false,
                },
                usage: VariableUsage::Move,
                lifetime_id: LifetimeId::static_lifetime(),
                source_location,
                metadata: ExpressionMetadata::default(),
            };

            let value_decl = TypedStatement::VarDeclaration {
                symbol_id: variable,
                var_type: element_type,
                initializer: Some(next_call),
                source_location,
                mutability: crate::tast::Mutability::Mutable,
            };
            loop_statements.push(value_decl);
        }

        // Add the loop body
        loop_statements.push(body);

        // Create while body block
        let while_body = TypedStatement::Block {
            statements: loop_statements,
            scope_id: loop_body_scope_id,
            source_location,
        };

        // Create while loop: while (__iter.hasNext()) { ... }
        let while_stmt = TypedStatement::While {
            condition,
            body: Box::new(while_body),
            source_location,
        };

        // Return block: { var __iter = iterable.iterator(); while (__iter.hasNext()) { ... } }
        Ok(TypedStatement::Block {
            statements: vec![init_stmt, while_stmt],
            scope_id: ScopeId::from_raw(self.context.next_scope_id()),
            source_location,
        })
    }

    /// Infer the element type from an iterable expression (array, map, etc.)
    fn infer_element_type_from_iterable(&self, iterable: &TypedExpression) -> TypeId {
        let type_table = self.context.type_table.borrow();

        // Get the actual type of the iterable
        if let Some(actual_type) = type_table.get(iterable.expr_type) {
            match &actual_type.kind {
                TypeKind::Array { element_type } => *element_type,
                TypeKind::Map { value_type, .. } => *value_type,
                TypeKind::Dynamic => type_table.dynamic_type(),
                _ => {
                    // Check if type implements Iterable<T> interface
                    if let TypeKind::Class { symbol_id, .. } = &actual_type.kind {
                        // Look for Iterable interface implementation
                        if let Some(class_symbol) = self.context.symbol_table.get_symbol(*symbol_id)
                        {
                            if let Some(hierarchy) =
                                self.context.symbol_table.get_class_hierarchy(*symbol_id)
                            {
                                // Check implemented interfaces for Iterable<T>
                                for interface_type in &hierarchy.interfaces {
                                    if let Some(interface_info) = type_table.get(*interface_type) {
                                        if let TypeKind::Interface {
                                            symbol_id: interface_symbol,
                                            ..
                                        } = &interface_info.kind
                                        {
                                            // Check if this is the Iterable interface
                                            if let Some(interface) = self
                                                .context
                                                .symbol_table
                                                .get_symbol(*interface_symbol)
                                            {
                                                let iterable_name =
                                                    self.context.string_interner.intern("Iterable");
                                                if interface.name == iterable_name {
                                                    // For generic interfaces, extract the type argument
                                                    // This would require tracking instantiated type arguments
                                                    return type_table.dynamic_type();
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    // Not iterable or can't determine element type
                    type_table.dynamic_type()
                }
            }
        } else {
            // If we can't determine the type, use dynamic
            type_table.dynamic_type()
        }
    }

    /// Infer the iterator type name to load based on the iterable expression type
    /// Returns the qualified name of the iterator class to load (e.g., "haxe.iterators.ArrayIterator")
    fn infer_iterator_type_name(&self, type_id: &TypeId) -> Option<String> {
        let type_table = self.context.type_table.borrow();

        if let Some(actual_type) = type_table.get(*type_id) {
            match &actual_type.kind {
                TypeKind::Array { .. } => {
                    // Arrays use ArrayIterator from haxe.iterators
                    Some("haxe.iterators.ArrayIterator".to_string())
                }
                TypeKind::Map { .. } => {
                    // Maps might use a MapIterator (implementation dependent)
                    Some("haxe.iterators.MapIterator".to_string())
                }
                TypeKind::Class { symbol_id, .. } => {
                    // Check the class name to determine iterator type
                    if let Some(class_symbol) = self.context.symbol_table.get_symbol(*symbol_id) {
                        if let Some(class_name) =
                            self.context.string_interner.get(class_symbol.name)
                        {
                            match class_name {
                                "Array" => Some("haxe.iterators.ArrayIterator".to_string()),
                                "String" => Some("haxe.iterators.StringIterator".to_string()),
                                "IntIterator" => Some("haxe.iterators.IntIterator".to_string()),
                                _ => {
                                    // For other classes, try to infer from package
                                    // Could also look at implemented interfaces
                                    None
                                }
                            }
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                }
                _ => None,
            }
        } else {
            None
        }
    }

    /// Infer the type of an expression based on its kind
    fn infer_expression_type(&mut self, kind: &TypedExpressionKind) -> LoweringResult<TypeId> {
        match kind {
            TypedExpressionKind::Literal { value } => {
                let type_table = self.context.type_table.borrow();
                match value {
                    LiteralValue::Bool(_) => Ok(type_table.bool_type()),
                    LiteralValue::Int(_) => Ok(type_table.int_type()),
                    LiteralValue::Float(_) => Ok(type_table.float_type()),
                    LiteralValue::String(_) => Ok(type_table.string_type()),
                    LiteralValue::Char(_) => Ok(type_table.string_type()), // Haxe treats char as string
                    LiteralValue::Regex(_) | LiteralValue::RegexWithFlags { .. } => {
                        // EReg type in Haxe — resolve as proper class type
                        drop(type_table);
                        let ereg_name = self.context.string_interner.intern("EReg");
                        if let Some(symbol_id) = self.resolve_symbol_in_scope_hierarchy(ereg_name) {
                            if let Some(symbol) = self.context.symbol_table.get_symbol(symbol_id) {
                                return Ok(symbol.type_id);
                            }
                        }
                        // Fallback to structural type
                        Ok(type_resolution::get_regex_type(
                            &self.context.type_table,
                            self.context.string_interner,
                        ))
                    }
                }
            }
            TypedExpressionKind::Variable { symbol_id } => {
                // Look up the symbol's type
                if let Some(symbol) = self.context.symbol_table.get_symbol(*symbol_id) {
                    Ok(symbol.type_id)
                } else {
                    Ok(self.context.type_table.borrow().dynamic_type())
                }
            }
            TypedExpressionKind::BinaryOp {
                left,
                operator,
                right,
            } => {
                let type_table = self.context.type_table.borrow();
                match operator {
                    BinaryOperator::Add => {
                        // Add can be either string concatenation or numeric addition
                        let left_type = left.expr_type;
                        let right_type = right.expr_type;
                        let dynamic_type = type_table.dynamic_type();
                        let string_type = type_table.string_type();
                        let int_type = type_table.int_type();
                        let float_type = type_table.float_type();

                        // If either operand is Dynamic, result is Dynamic
                        if left_type == dynamic_type || right_type == dynamic_type {
                            Ok(dynamic_type)
                        }
                        // If either operand is string, result is string (concatenation)
                        else if left_type == string_type || right_type == string_type {
                            Ok(string_type)
                        }
                        // If either operand is Float, result is Float
                        else if left_type == float_type || right_type == float_type {
                            Ok(float_type)
                        }
                        // If both are Int, result is Int
                        else if left_type == int_type && right_type == int_type {
                            Ok(int_type)
                        }
                        // Default to Float for safety
                        else {
                            Ok(float_type)
                        }
                    }
                    BinaryOperator::Sub
                    | BinaryOperator::Mul
                    | BinaryOperator::Div
                    | BinaryOperator::Mod => {
                        // Purely numeric operations
                        let left_type = left.expr_type;
                        let right_type = right.expr_type;
                        let dynamic_type = type_table.dynamic_type();

                        // If either operand is Dynamic, result is Dynamic
                        if left_type == dynamic_type || right_type == dynamic_type {
                            Ok(dynamic_type)
                        }
                        // If either operand is Float, result is Float
                        else if left_type == type_table.float_type()
                            || right_type == type_table.float_type()
                        {
                            Ok(type_table.float_type())
                        }
                        // If both are Int, result is Int (except for division which returns Float)
                        else if left_type == type_table.int_type()
                            && right_type == type_table.int_type()
                        {
                            match operator {
                                BinaryOperator::Div => Ok(type_table.float_type()), // Division always returns Float in Haxe
                                _ => Ok(type_table.int_type()),
                            }
                        }
                        // Default to Float for safety
                        else {
                            Ok(type_table.float_type())
                        }
                    }
                    BinaryOperator::Eq
                    | BinaryOperator::Ne
                    | BinaryOperator::Lt
                    | BinaryOperator::Le
                    | BinaryOperator::Gt
                    | BinaryOperator::Ge => Ok(type_table.bool_type()),
                    BinaryOperator::And | BinaryOperator::Or => Ok(type_table.bool_type()),
                    BinaryOperator::Assign
                    | BinaryOperator::AddAssign
                    | BinaryOperator::SubAssign
                    | BinaryOperator::MulAssign
                    | BinaryOperator::DivAssign
                    | BinaryOperator::ModAssign => {
                        // Assignment returns the type of the left operand
                        Ok(left.expr_type)
                    }
                    BinaryOperator::NullCoal => {
                        // Null coalescing: result type is the LHS type (non-null version)
                        Ok(left.expr_type)
                    }
                    _ => Ok(type_table.dynamic_type()),
                }
            }
            TypedExpressionKind::FunctionCall {
                function,
                arguments,
                ..
            } => {
                // Extract return type from function signature
                let type_table = self.context.type_table.borrow();
                match type_table.get(function.expr_type) {
                    Some(function_type) => match &function_type.kind {
                        crate::tast::core::TypeKind::Function {
                            params,
                            return_type,
                            ..
                        } => {
                            let ret = *return_type;
                            // If return type is a TypeParameter, infer concrete type from arguments
                            if type_table.is_type_parameter(ret) {
                                for (i, param_ty) in params.iter().enumerate() {
                                    if *param_ty == ret && i < arguments.len() {
                                        return Ok(arguments[i].expr_type);
                                    }
                                }
                            }
                            Ok(ret)
                        }
                        _ => Ok(type_table.dynamic_type()),
                    },
                    None => Ok(type_table.dynamic_type()),
                }
            }
            TypedExpressionKind::New { class_type, .. } => Ok(*class_type),
            TypedExpressionKind::ArrayLiteral { elements } => {
                if let Some(first_element) = elements.first() {
                    let element_type = first_element.expr_type;
                    Ok(self
                        .context
                        .type_table
                        .borrow_mut()
                        .create_array_type(element_type))
                } else {
                    let dyn_type = self.context.type_table.borrow().dynamic_type();
                    Ok(self
                        .context
                        .type_table
                        .borrow_mut()
                        .create_array_type(dyn_type))
                }
            }
            TypedExpressionKind::Null => {
                Ok(type_resolution::get_null_type(&self.context.type_table))
            }
            TypedExpressionKind::This { this_type } => Ok(*this_type),
            TypedExpressionKind::Super { super_type } => Ok(*super_type),
            TypedExpressionKind::ObjectLiteral { fields } => {
                // For anonymous objects, infer type from fields
                let field_types: Vec<(InternedString, TypeId)> =
                    fields.iter().map(|f| (f.name, f.value.expr_type)).collect();
                Ok(type_resolution::create_anonymous_object_type(
                    &self.context.type_table,
                    field_types,
                ))
            }
            TypedExpressionKind::StringInterpolation { .. } => {
                Ok(self.context.type_table.borrow().string_type())
            }
            TypedExpressionKind::Cast { target_type, .. } => Ok(*target_type),
            TypedExpressionKind::Is { .. } => Ok(self.context.type_table.borrow().bool_type()),
            TypedExpressionKind::FieldAccess {
                object,
                field_symbol,
                ..
            } => {
                // Look up field type from symbol table
                if let Some(symbol) = self.context.symbol_table.get_symbol(*field_symbol) {
                    // Check if this is a valid typed symbol
                    if symbol.type_id.is_valid() {
                        Ok(symbol.type_id)
                    } else {
                        // Handle built-in method access for Array, String, etc.
                        self.infer_builtin_method_type(object.expr_type, *field_symbol)
                    }
                } else {
                    // Handle built-in method access for Array, String, etc.
                    self.infer_builtin_method_type(object.expr_type, *field_symbol)
                }
            }
            TypedExpressionKind::ArrayAccess { array, .. } => {
                // Extract element type from array type
                let type_table = self.context.type_table.borrow();
                let result = match type_table.get(array.expr_type) {
                    Some(array_type) => match &array_type.kind {
                        crate::tast::core::TypeKind::Array { element_type } => Ok(*element_type),
                        _other => Ok(type_table.dynamic_type()),
                    },
                    None => Ok(type_table.dynamic_type()),
                };
                result
            }
            TypedExpressionKind::MethodCall {
                receiver,
                method_symbol,
                ..
            } => {
                // Extract return type from method signature and substitute type parameters
                self.infer_method_call_return_type(*method_symbol, receiver.expr_type)
            }
            TypedExpressionKind::StaticMethodCall {
                method_symbol,
                arguments,
                ..
            } => {
                // Extract return type from static method signature
                if let Some(symbol) = self.context.symbol_table.get_symbol(*method_symbol) {
                    let type_table = self.context.type_table.borrow();
                    if let Some(method_type) = type_table.get(symbol.type_id) {
                        match &method_type.kind {
                            crate::tast::core::TypeKind::Function {
                                params,
                                return_type,
                                ..
                            } => {
                                let ret = *return_type;
                                // If return type is a TypeParameter, infer from arguments
                                if type_table.is_type_parameter(ret) {
                                    for (i, param_ty) in params.iter().enumerate() {
                                        if *param_ty == ret && i < arguments.len() {
                                            return Ok(arguments[i].expr_type);
                                        }
                                    }
                                }
                                Ok(ret)
                            }
                            _ => Ok(type_table.dynamic_type()),
                        }
                    } else {
                        Ok(type_table.dynamic_type())
                    }
                } else {
                    Ok(self.context.type_table.borrow().dynamic_type())
                }
            }
            TypedExpressionKind::UnaryOp { operator, operand } => {
                let type_table = self.context.type_table.borrow();
                match operator {
                    UnaryOperator::Not => Ok(type_table.bool_type()),
                    UnaryOperator::Neg | UnaryOperator::BitNot => Ok(operand.expr_type),
                    UnaryOperator::PreInc
                    | UnaryOperator::PostInc
                    | UnaryOperator::PreDec
                    | UnaryOperator::PostDec => Ok(operand.expr_type),
                }
            }
            TypedExpressionKind::Conditional {
                then_expr,
                else_expr,
                ..
            } => {
                // Type unification handled by type checker
                Ok(then_expr.expr_type)
            }
            TypedExpressionKind::While { .. }
            | TypedExpressionKind::For { .. }
            | TypedExpressionKind::ForIn { .. } => Ok(self.context.type_table.borrow().void_type()),
            TypedExpressionKind::FunctionLiteral {
                parameters,
                return_type,
                ..
            } => {
                // Create a function type: (param1_type, param2_type, ...) -> return_type
                let param_types: Vec<TypeId> = parameters.iter().map(|p| p.param_type).collect();
                Ok(self
                    .context
                    .type_table
                    .borrow_mut()
                    .create_function_type(param_types, *return_type))
            }
            TypedExpressionKind::Return { .. } => Ok(self.context.type_table.borrow().void_type()),
            TypedExpressionKind::Throw { .. } => Ok(self.context.type_table.borrow().void_type()),
            TypedExpressionKind::Break | TypedExpressionKind::Continue => {
                Ok(self.context.type_table.borrow().void_type())
            }
            TypedExpressionKind::Block { statements, .. } => {
                // Block type is the type of the last expression
                let type_table = self.context.type_table.borrow();
                if let Some(last_stmt) = statements.last() {
                    // Extract type from last statement
                    match last_stmt {
                        TypedStatement::Expression { expression, .. } => Ok(expression.expr_type),
                        _ => {
                            // Non-expression statements result in void type
                            Ok(type_table.void_type())
                        }
                    }
                } else {
                    // Empty block has void type
                    Ok(type_table.void_type())
                }
            }
            TypedExpressionKind::Meta { expression, .. } => Ok(expression.expr_type),
            TypedExpressionKind::DollarIdent { .. } => {
                Ok(self.context.type_table.borrow().dynamic_type()) // Macro-related
            }
            TypedExpressionKind::CompilerSpecific { .. } => {
                Ok(self.context.type_table.borrow().dynamic_type())
            }
            TypedExpressionKind::Switch {
                cases,
                default_case,
                ..
            } => {
                // Switch expression type is inferred from the branches
                // All branches should have compatible types
                let mut branch_types = Vec::new();

                // Collect types from case branches
                for case in cases {
                    // Extract expression type from the case body statement
                    match &case.body {
                        TypedStatement::Expression { expression, .. } => {
                            branch_types.push(expression.expr_type);
                            // eprintln!(
                            //     "DEBUG: Switch case expression type: {:?}",
                            //     expression.expr_type
                            // );

                            // For switch expressions (not statements), check for void type
                            // But only if this is truly a switch expression context
                            // TODO: Properly distinguish between switch expressions and statements
                        }
                        _ => {
                            // Non-expression statements in switch expression context
                            // This shouldn't happen for valid switch expressions
                            self.context.errors.push(LoweringError::InternalError {
                                message: "Switch expression case must be an expression".to_string(),
                                location: case.source_location,
                            });
                        }
                    }
                }

                // Add default case type if present
                if let Some(default) = default_case {
                    branch_types.push(default.expr_type);
                    // eprintln!(
                    //     "DEBUG: Switch default expression type: {:?}",
                    //     default.expr_type
                    // );

                    // For switch expressions (not statements), check for void type in default
                    // But only if this is truly a switch expression context
                    // TODO: Properly distinguish between switch expressions and statements
                }

                // If no branches have expressions, it's a void switch
                if branch_types.is_empty() {
                    return Ok(self.context.type_table.borrow().void_type());
                }

                // Filter out void types and use the first non-void type for expression result
                let void_type = self.context.type_table.borrow().void_type();
                let non_void_types: Vec<TypeId> = branch_types
                    .iter()
                    .filter(|&&t| t != void_type)
                    .copied()
                    .collect();

                if non_void_types.is_empty() {
                    // All branches are void, return void (but errors should have been generated above)
                    return Ok(void_type);
                }

                // For now, use the first non-void branch type
                // Type unification deferred to type checker
                // eprintln!(
                //     "DEBUG: Switch expression inferred type: {:?}",
                //     non_void_types[0]
                // );
                Ok(non_void_types[0])
            }
            TypedExpressionKind::Try { try_expr, .. } => {
                // Try expression type is the type of the try block
                Ok(try_expr.expr_type)
            }
            TypedExpressionKind::VarDeclarationExpr { var_type, .. } => Ok(*var_type),
            TypedExpressionKind::FinalDeclarationExpr { var_type, .. } => Ok(*var_type),
            TypedExpressionKind::MapLiteral { entries } => {
                // Infer key and value types from initial values
                let (key_type, value_type) = if entries.is_empty() {
                    let dyn_type = self.context.type_table.borrow().dynamic_type();
                    (dyn_type, dyn_type)
                } else {
                    // Use first entry to infer types
                    let first = &entries[0];
                    (first.key.expr_type, first.value.expr_type)
                };
                Ok(type_resolution::create_map_type(
                    &self.context.type_table,
                    key_type,
                    value_type,
                ))
            }
            TypedExpressionKind::MacroExpression { .. } => {
                Ok(self.context.type_table.borrow().dynamic_type()) // Macro result type
            }
            TypedExpressionKind::ArrayComprehension { element_type, .. } => {
                // Array comprehension creates an Array<T> type
                Ok(self
                    .context
                    .type_table
                    .borrow_mut()
                    .create_array_type(*element_type))
            }
            TypedExpressionKind::MapComprehension {
                key_type,
                value_type,
                ..
            } => {
                // Map comprehension creates a Map<K, V> type
                Ok(type_resolution::create_map_type(
                    &self.context.type_table,
                    *key_type,
                    *value_type,
                ))
            }
            TypedExpressionKind::FunctionCall { function, .. } => {
                // Extract return type from the function's type
                // For enum constructors, the function has a Function type where return_type is the enum type
                let type_table = self.context.type_table.borrow();
                if let Some(func_type) = type_table.get(function.expr_type) {
                    match &func_type.kind {
                        crate::tast::core::TypeKind::Function { return_type, .. } => {
                            Ok(*return_type)
                        }
                        // If it's already an enum type (for simple enum variants), use it directly
                        crate::tast::core::TypeKind::Enum { .. } => Ok(function.expr_type),
                        _ => Ok(type_table.dynamic_type()),
                    }
                } else {
                    Ok(type_table.dynamic_type())
                }
            }
            _ => {
                // Log unhandled case as warning but continue with dynamic type
                self.context
                    .add_error(LoweringError::IncompleteImplementation {
                        feature: format!("Type inference for expression kind: {:?}", kind),
                        location: self.context.create_location(),
                    });
                Ok(self.context.type_table.borrow().dynamic_type())
            }
        }
    }

    /// Infer the return type of a method call, substituting type parameters from the receiver.
    fn infer_method_call_return_type(
        &mut self,
        method_symbol: SymbolId,
        receiver_type: TypeId,
    ) -> LoweringResult<TypeId> {
        // Phase 1: Collect all necessary information with immutable borrow
        let substitution_result = {
            let type_table = self.context.type_table.borrow();

            // Get the method symbol
            let method_type_id = match self.context.symbol_table.get_symbol(method_symbol) {
                Some(symbol) if symbol.type_id.is_valid() => symbol.type_id,
                _ => {
                    // Method symbol has no type info (placeholder for built-in methods).
                    // Use infer_builtin_method_type to get the method's function type,
                    // then extract the return type from it.
                    drop(type_table);
                    let method_func_type =
                        self.infer_builtin_method_type(receiver_type, method_symbol)?;
                    let type_table = self.context.type_table.borrow();
                    return match type_table.get(method_func_type) {
                        Some(info) => match &info.kind {
                            crate::tast::core::TypeKind::Function { return_type, .. } => {
                                Ok(*return_type)
                            }
                            _ => Ok(method_func_type), // Not a function type — it's the type itself (e.g., length: Int)
                        },
                        None => Ok(type_table.dynamic_type()),
                    };
                }
            };

            // Get the method's function type
            let return_type = match type_table.get(method_type_id) {
                Some(method_type) => match &method_type.kind {
                    crate::tast::core::TypeKind::Function { return_type, .. } => *return_type,
                    _ => return Ok(type_table.dynamic_type()),
                },
                None => return Ok(type_table.dynamic_type()),
            };

            // Compute the substitution
            self.compute_type_substitution(return_type, receiver_type, &type_table)
        };

        // Phase 2: Create new type if needed (with mutable borrow)
        match substitution_result {
            TypeSubstitutionResult::NoChange(type_id) => Ok(type_id),
            TypeSubstitutionResult::DirectSubstitution(type_id) => Ok(type_id),
            TypeSubstitutionResult::NeedGenericInstance {
                base_type,
                type_args,
            } => Ok(self
                .context
                .type_table
                .borrow_mut()
                .create_generic_instance(base_type, type_args)),
        }
    }

    /// Compute what substitution is needed (without creating new types)
    fn compute_type_substitution(
        &self,
        return_type: TypeId,
        receiver_type: TypeId,
        type_table: &std::cell::Ref<'_, crate::tast::TypeTable>,
    ) -> TypeSubstitutionResult {
        // Get receiver's substitution info
        let receiver_type_info = match type_table.get(receiver_type) {
            Some(info) => info,
            None => return TypeSubstitutionResult::NoChange(return_type),
        };

        // Extract base type parameters and type arguments from receiver
        let (base_type_params, type_args) = match &receiver_type_info.kind {
            crate::tast::core::TypeKind::GenericInstance {
                base_type,
                type_args,
                ..
            } => {
                // Get the base class's type parameters
                if let Some(base_info) = type_table.get(*base_type) {
                    match &base_info.kind {
                        crate::tast::core::TypeKind::Class {
                            type_args: params, ..
                        }
                        | crate::tast::core::TypeKind::Interface {
                            type_args: params, ..
                        } => (params.clone(), type_args.clone()),
                        _ => return TypeSubstitutionResult::NoChange(return_type),
                    }
                } else {
                    return TypeSubstitutionResult::NoChange(return_type);
                }
            }
            _ => return TypeSubstitutionResult::NoChange(return_type),
        };

        // Get the return type info
        let return_type_info = match type_table.get(return_type) {
            Some(info) => info,
            None => return TypeSubstitutionResult::NoChange(return_type),
        };

        match &return_type_info.kind {
            crate::tast::core::TypeKind::TypeParameter { symbol_id, .. } => {
                // Direct type parameter - find and substitute
                for (i, param_type_id) in base_type_params.iter().enumerate() {
                    if let Some(param_info) = type_table.get(*param_type_id) {
                        if let crate::tast::core::TypeKind::TypeParameter {
                            symbol_id: param_sym,
                            ..
                        } = &param_info.kind
                        {
                            if param_sym == symbol_id {
                                if i < type_args.len() {
                                    return TypeSubstitutionResult::DirectSubstitution(
                                        type_args[i],
                                    );
                                }
                            }
                        }
                    }
                }
                TypeSubstitutionResult::NoChange(return_type)
            }
            crate::tast::core::TypeKind::GenericInstance {
                base_type,
                type_args: ret_type_args,
                ..
            } => {
                // Generic return type - need to substitute type args recursively
                let mut new_type_args = Vec::with_capacity(ret_type_args.len());
                let mut changed = false;

                for arg in ret_type_args {
                    match self.compute_type_substitution(*arg, receiver_type, type_table) {
                        TypeSubstitutionResult::NoChange(_) => new_type_args.push(*arg),
                        TypeSubstitutionResult::DirectSubstitution(new_arg) => {
                            new_type_args.push(new_arg);
                            changed = true;
                        }
                        TypeSubstitutionResult::NeedGenericInstance {
                            base_type,
                            type_args,
                        } => {
                            // Would need to create nested type - for now just use the original
                            // This is a limitation, but handles most common cases
                            new_type_args.push(*arg);
                        }
                    }
                }

                if changed {
                    // Check if this exact type already exists
                    if let Some(existing) =
                        self.find_existing_generic_instance(*base_type, &new_type_args, type_table)
                    {
                        return TypeSubstitutionResult::DirectSubstitution(existing);
                    }
                    return TypeSubstitutionResult::NeedGenericInstance {
                        base_type: *base_type,
                        type_args: new_type_args,
                    };
                }
                TypeSubstitutionResult::NoChange(return_type)
            }
            _ => TypeSubstitutionResult::NoChange(return_type),
        }
    }

    /// Substitute type parameters in a type with actual type arguments from a receiver type.
    ///
    /// For example, if we have:
    /// - return_type = T (a TypeParameter)
    /// - receiver_type = Arc<Channel<Int>> (a GenericInstance)
    ///
    /// This function will substitute T with Channel<Int>.
    fn substitute_type_params_in_type(
        &self,
        return_type: TypeId,
        receiver_type: TypeId,
        type_table: &std::cell::Ref<'_, crate::tast::TypeTable>,
    ) -> TypeId {
        // Collect all necessary info in one pass, then we can release the borrow if needed

        // Get the receiver's substitution info (base_type_params and type_args)
        let substitution_info: Option<(Vec<TypeId>, Vec<TypeId>)> = {
            let receiver_type_info = match type_table.get(receiver_type) {
                Some(info) => info,
                None => return return_type,
            };

            match &receiver_type_info.kind {
                crate::tast::core::TypeKind::GenericInstance {
                    base_type,
                    type_args,
                    ..
                } => {
                    // Get the base class's type parameters
                    if let Some(base_info) = type_table.get(*base_type) {
                        match &base_info.kind {
                            crate::tast::core::TypeKind::Class {
                                type_args: params, ..
                            }
                            | crate::tast::core::TypeKind::Interface {
                                type_args: params, ..
                            } => Some((params.clone(), type_args.clone())),
                            _ => None,
                        }
                    } else {
                        None
                    }
                }
                _ => None,
            }
        };

        let (base_type_params, type_args) = match substitution_info {
            Some(info) => info,
            None => return return_type,
        };

        // Get the return type info
        let return_type_info = match type_table.get(return_type) {
            Some(info) => info,
            None => return return_type,
        };

        // Now substitute based on return type kind
        match &return_type_info.kind {
            crate::tast::core::TypeKind::TypeParameter { symbol_id, .. } => {
                // Find which type parameter this is and substitute
                for (i, param_type_id) in base_type_params.iter().enumerate() {
                    if let Some(param_info) = type_table.get(*param_type_id) {
                        if let crate::tast::core::TypeKind::TypeParameter {
                            symbol_id: param_sym,
                            ..
                        } = &param_info.kind
                        {
                            if param_sym == symbol_id {
                                // Found matching type parameter, substitute with type argument
                                if i < type_args.len() {
                                    return type_args[i];
                                }
                            }
                        }
                    }
                }
                return_type
            }
            crate::tast::core::TypeKind::GenericInstance {
                base_type,
                type_args: ret_type_args,
                ..
            } => {
                // Recursively substitute type parameters in the type arguments
                // E.g., Arc<T>.clone() returns Arc<T>, substitute T in the Arc<T>
                // First, collect all info we need
                let base = *base_type;
                let ret_args: Vec<TypeId> = ret_type_args.clone();

                let mut new_type_args = Vec::with_capacity(ret_args.len());
                let mut changed = false;
                for arg in &ret_args {
                    let substituted =
                        self.substitute_type_params_in_type(*arg, receiver_type, type_table);
                    if substituted != *arg {
                        changed = true;
                    }
                    new_type_args.push(substituted);
                }
                if changed {
                    // Need to create a new type - but we can't mutably borrow here
                    // Return a signal that we need to create a new type
                    // For now, we'll use an existing type if it matches, or return the original
                    // Actually, let's check if the type already exists
                    // This is a limitation - we may need to refactor more significantly
                    // For now, try to find if the substituted type exists
                    if let Some(existing) =
                        self.find_existing_generic_instance(base, &new_type_args, type_table)
                    {
                        return existing;
                    }
                    // Fallback: return original (the substitution will need to happen elsewhere)
                    return return_type;
                }
                return_type
            }
            crate::tast::core::TypeKind::Class {
                symbol_id: _sym_id,
                type_args: class_type_args,
            } if !class_type_args.is_empty() => {
                // For class types with type args that are type parameters, substitute them
                let class_args: Vec<TypeId> = class_type_args.clone();

                let mut new_type_args = Vec::with_capacity(class_args.len());
                let mut changed = false;
                for arg in &class_args {
                    let substituted =
                        self.substitute_type_params_in_type(*arg, receiver_type, type_table);
                    if substituted != *arg {
                        changed = true;
                    }
                    new_type_args.push(substituted);
                }
                if changed {
                    // Same limitation as above
                    return return_type;
                }
                return_type
            }
            _ => return_type,
        }
    }

    /// Try to find an existing GenericInstance with the given base type and type args
    fn find_existing_generic_instance(
        &self,
        base_type: TypeId,
        type_args: &[TypeId],
        type_table: &std::cell::Ref<'_, crate::tast::TypeTable>,
    ) -> Option<TypeId> {
        // Search through existing types to find a matching GenericInstance
        // This is O(n) but avoids the borrow conflict
        for (type_id, type_info) in type_table.iter() {
            if let crate::tast::core::TypeKind::GenericInstance {
                base_type: existing_base,
                type_args: existing_args,
                ..
            } = &type_info.kind
            {
                if *existing_base == base_type && existing_args == type_args {
                    return Some(type_id);
                }
            }
        }
        None
    }

    /// Recursively match a parameter type against an argument type to find where
    /// a specific TypeParameter appears, and extract the concrete type from the
    /// argument at the same structural position.
    ///
    /// For example, if param_ty is `Function { return_type: T }` and arg_ty is
    /// `Function { return_type: Int }`, this returns `Some(Int)` for target T.
    fn match_type_param_in_types(
        target_sym: SymbolId,
        param_ty: TypeId,
        arg_ty: TypeId,
        type_table: &std::cell::Ref<'_, crate::tast::TypeTable>,
    ) -> Option<TypeId> {
        let param_info = type_table.get(param_ty)?;
        match &param_info.kind {
            crate::tast::core::TypeKind::TypeParameter { symbol_id, .. } => {
                if *symbol_id == target_sym {
                    Some(arg_ty)
                } else {
                    None
                }
            }
            crate::tast::core::TypeKind::Function {
                params: fn_params,
                return_type: fn_ret,
                ..
            } => {
                let arg_info = type_table.get(arg_ty)?;
                if let crate::tast::core::TypeKind::Function {
                    params: arg_fn_params,
                    return_type: arg_fn_ret,
                    ..
                } = &arg_info.kind
                {
                    // Check return type
                    if let Some(result) = Self::match_type_param_in_types(
                        target_sym,
                        *fn_ret,
                        *arg_fn_ret,
                        type_table,
                    ) {
                        return Some(result);
                    }
                    // Check function parameters
                    for (fp, ap) in fn_params.iter().zip(arg_fn_params.iter()) {
                        if let Some(result) =
                            Self::match_type_param_in_types(target_sym, *fp, *ap, type_table)
                        {
                            return Some(result);
                        }
                    }
                }
                None
            }
            crate::tast::core::TypeKind::GenericInstance { type_args, .. } => {
                let arg_info = type_table.get(arg_ty)?;
                if let crate::tast::core::TypeKind::GenericInstance {
                    type_args: arg_type_args,
                    ..
                } = &arg_info.kind
                {
                    for (ta, ata) in type_args.iter().zip(arg_type_args.iter()) {
                        if let Some(result) =
                            Self::match_type_param_in_types(target_sym, *ta, *ata, type_table)
                        {
                            return Some(result);
                        }
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Determine variable usage based on expression kind (simplified for TAST)
    fn determine_variable_usage(&self, kind: &TypedExpressionKind) -> VariableUsage {
        match kind {
            TypedExpressionKind::Literal { .. } => {
                // Literals are always copyable
                VariableUsage::Copy
            }
            _ => {
                // Default to copy for TAST - ownership analysis happens in semantic graph
                VariableUsage::Copy
            }
        }
    }

    /// Assign lifetime based on expression scope and type (simplified for TAST)
    fn assign_lifetime(&self, kind: &TypedExpressionKind, expr_type: &TypeId) -> LifetimeId {
        match kind {
            TypedExpressionKind::Literal { .. } => {
                // Literals have static lifetime
                LifetimeId::static_lifetime()
            }
            TypedExpressionKind::Variable { symbol_id } => {
                // Variables have lifetime tied to their declaring scope
                if let Some(symbol) = self.context.symbol_table.get_symbol(*symbol_id) {
                    symbol.lifetime_id
                } else {
                    LifetimeId::first() // Default lifetime for TAST
                }
            }
            _ => {
                // Default to current scope lifetime - detailed analysis in semantic graph
                LifetimeId::from_raw(1) // TODO: Use proper lifetime ID generation
            }
        }
    }

    /// Analyze expression metadata for optimization and error reporting
    fn analyze_expression_metadata(&self, kind: &TypedExpressionKind) -> ExpressionMetadata {
        let mut metadata = ExpressionMetadata::default();

        match kind {
            TypedExpressionKind::Literal { .. } => {
                metadata.is_constant = true;
                metadata.has_side_effects = false;
                metadata.can_throw = false;
                metadata.complexity_score = 1;
            }
            TypedExpressionKind::Variable { .. } => {
                metadata.is_constant = false;
                metadata.has_side_effects = false;
                metadata.can_throw = false;
                metadata.complexity_score = 1;
            }
            TypedExpressionKind::FunctionCall { .. } => {
                metadata.is_constant = false;
                metadata.has_side_effects = true; // Assume function calls have side effects
                metadata.can_throw = true; // Assume function calls can throw
                metadata.complexity_score = 10;
            }
            TypedExpressionKind::BinaryOp { operator, .. } => {
                metadata.is_constant = false;
                metadata.complexity_score = 2;

                match operator {
                    BinaryOperator::Assign
                    | BinaryOperator::AddAssign
                    | BinaryOperator::SubAssign
                    | BinaryOperator::MulAssign
                    | BinaryOperator::DivAssign
                    | BinaryOperator::ModAssign => {
                        metadata.has_side_effects = true;
                        metadata.can_throw = false;
                    }
                    BinaryOperator::Div | BinaryOperator::Mod => {
                        metadata.has_side_effects = false;
                        metadata.can_throw = true; // Division by zero
                    }
                    _ => {
                        metadata.has_side_effects = false;
                        metadata.can_throw = false;
                    }
                }
            }
            TypedExpressionKind::New { .. } => {
                metadata.is_constant = false;
                metadata.has_side_effects = true; // Memory allocation
                metadata.can_throw = true; // Constructor can throw
                metadata.complexity_score = 5;
            }
            _ => {
                metadata.is_constant = false;
                metadata.has_side_effects = false;
                metadata.can_throw = false;
                metadata.complexity_score = 1;
            }
        }

        metadata
    }

    /// Validate that the lowered TAST contains all necessary information for memory safety analysis
    pub fn validate_tast(&self, typed_file: &TypedFile) -> Vec<LoweringError> {
        let mut errors = Vec::new();

        // Validate functions have proper lifetime and ownership information
        for function in &typed_file.functions {
            if function
                .parameters
                .iter()
                .any(|p| p.symbol_id == SymbolId::invalid())
            {
                errors.push(LoweringError::IncompleteImplementation {
                    feature: format!("Function parameter symbol resolution for {}", function.name),
                    location: function.source_location,
                });
            }

            // Validate expressions in function bodies
            for statement in &function.body {
                self.validate_statement(statement, &mut errors);
            }
        }

        // Validate classes have proper field information
        for class in &typed_file.classes {
            if class.symbol_id == SymbolId::invalid() {
                errors.push(LoweringError::IncompleteImplementation {
                    feature: format!("Class symbol resolution for {}", class.name),
                    location: class.source_location,
                });
            }

            for field in &class.fields {
                if field.symbol_id == SymbolId::invalid() {
                    errors.push(LoweringError::IncompleteImplementation {
                        feature: format!("Field symbol resolution for {}", field.name),
                        location: field.source_location,
                    });
                }
            }
        }

        errors
    }

    /// Validate a statement recursively
    fn validate_statement(&self, statement: &TypedStatement, errors: &mut Vec<LoweringError>) {
        match statement {
            TypedStatement::Expression {
                expression,
                source_location,
            } => {
                self.validate_expression(expression, errors);
            }
            TypedStatement::VarDeclaration {
                symbol_id,
                source_location,
                ..
            } => {
                if *symbol_id == SymbolId::invalid() {
                    errors.push(LoweringError::IncompleteImplementation {
                        feature: "Variable declaration symbol resolution".to_string(),
                        location: *source_location,
                    });
                }
            }
            TypedStatement::If {
                condition,
                then_branch,
                else_branch,
                ..
            } => {
                self.validate_expression(condition, errors);
                self.validate_statement(then_branch, errors);
                if let Some(else_stmt) = else_branch {
                    self.validate_statement(else_stmt, errors);
                }
            }
            TypedStatement::While {
                condition, body, ..
            } => {
                self.validate_expression(condition, errors);
                self.validate_statement(body, errors);
            }
            TypedStatement::For {
                init,
                condition,
                update,
                body,
                ..
            } => {
                if let Some(init_stmt) = init {
                    self.validate_statement(init_stmt, errors);
                }
                if let Some(cond_expr) = condition {
                    self.validate_expression(cond_expr, errors);
                }
                if let Some(update_expr) = update {
                    self.validate_expression(update_expr, errors);
                }
                self.validate_statement(body, errors);
            }
            TypedStatement::Block { statements, .. } => {
                for stmt in statements {
                    self.validate_statement(stmt, errors);
                }
            }
            _ => {
                // Other statement types - validate as needed
            }
        }
    }

    /// Validate an expression recursively
    fn validate_expression(&self, expression: &TypedExpression, errors: &mut Vec<LoweringError>) {
        // Check for invalid type IDs
        if expression.expr_type == TypeId::invalid() {
            errors.push(LoweringError::TypeInferenceError {
                expression: format!("{:?}", expression.kind),
                location: expression.source_location,
            });
        }

        // Check for invalid lifetime IDs
        if expression.lifetime_id == LifetimeId::invalid() {
            errors.push(LoweringError::LifetimeError {
                message: format!("Invalid lifetime for expression: {:?}", expression.kind),
                location: expression.source_location,
            });
        }

        // Validate variable references have proper symbols
        match &expression.kind {
            TypedExpressionKind::Variable { symbol_id } => {
                if *symbol_id == SymbolId::invalid() {
                    errors.push(LoweringError::UnresolvedSymbol {
                        name: "unknown_variable".to_string(),
                        location: expression.source_location,
                    });
                }
            }
            TypedExpressionKind::FieldAccess {
                object,
                field_symbol,
                ..
            } => {
                self.validate_expression(object, errors);
                if *field_symbol == SymbolId::invalid() {
                    errors.push(LoweringError::UnresolvedSymbol {
                        name: "unknown_field".to_string(),
                        location: expression.source_location,
                    });
                }
            }
            TypedExpressionKind::BinaryOp { left, right, .. } => {
                self.validate_expression(left, errors);
                self.validate_expression(right, errors);
            }
            TypedExpressionKind::UnaryOp { operand, .. } => {
                self.validate_expression(operand, errors);
            }
            TypedExpressionKind::FunctionCall {
                function,
                arguments,
                ..
            } => {
                self.validate_expression(function, errors);
                for arg in arguments {
                    self.validate_expression(arg, errors);
                }
            }
            TypedExpressionKind::ArrayAccess { array, index } => {
                self.validate_expression(array, errors);
                self.validate_expression(index, errors);
            }
            TypedExpressionKind::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
                self.validate_expression(condition, errors);
                self.validate_expression(then_expr, errors);
                if let Some(else_e) = else_expr {
                    self.validate_expression(else_e, errors);
                }
            }
            TypedExpressionKind::Block { statements, .. } => {
                for stmt in statements {
                    self.validate_statement(stmt, errors);
                }
            }
            _ => {
                // Other expression types - validate as needed
            }
        }
    }

    /// Resolve deferred type references (second pass)
    fn resolve_deferred_types(&mut self) -> LoweringResult<()> {
        let deferred = std::mem::take(&mut self.resolution_state.deferred_resolutions);

        for deferred_type in deferred {
            // Try to resolve the type now that all declarations have been processed
            // Parse the qualified type name into package path and simple name
            let parts: Vec<&str> = deferred_type.type_name.split('.').collect();

            let symbol = if parts.len() > 1 {
                // Qualified name like "haxe.iterators.ArrayIterator"
                let (package_parts, name) = parts.split_at(parts.len() - 1);
                let package: Vec<InternedString> = package_parts
                    .iter()
                    .map(|s| self.context.intern_string(s))
                    .collect();
                let name = self.context.intern_string(name[0]);

                let qualified_path =
                    crate::tast::namespace::QualifiedPath::new(package.clone(), name);

                // First try namespace resolver for qualified path lookup
                if let Some(symbol_id) = self
                    .context
                    .namespace_resolver
                    .lookup_symbol(&qualified_path)
                {
                    self.context.symbol_table.get_symbol(symbol_id)
                } else {
                    // Namespace lookup failed, try root scope lookup for backward compatibility
                    let interned_name = self.context.intern_string(&deferred_type.type_name);
                    self.context
                        .symbol_table
                        .lookup_symbol(ScopeId::first(), interned_name)
                }
            } else {
                // Simple name - look up in symbol table root scope
                let interned_name = self.context.intern_string(&deferred_type.type_name);
                self.context
                    .symbol_table
                    .lookup_symbol(ScopeId::first(), interned_name)
            };

            if let Some(symbol) = symbol {
                // Create the actual type based on the symbol kind
                let resolved_type = match symbol.kind {
                    crate::tast::SymbolKind::Class => self
                        .context
                        .type_table
                        .borrow_mut()
                        .create_class_type(symbol.id, Vec::new()),
                    crate::tast::SymbolKind::Interface => self
                        .context
                        .type_table
                        .borrow_mut()
                        .create_interface_type(symbol.id, Vec::new()),
                    crate::tast::SymbolKind::Enum => self
                        .context
                        .type_table
                        .borrow_mut()
                        .create_enum_type(symbol.id, Vec::new()),
                    crate::tast::SymbolKind::Abstract => {
                        self.context.type_table.borrow_mut().create_type(
                            crate::tast::core::TypeKind::Abstract {
                                symbol_id: symbol.id,
                                underlying: None,
                                type_args: Vec::new(),
                            },
                        )
                    }
                    crate::tast::SymbolKind::TypeAlias => {
                        let target_type = self.context.type_table.borrow().dynamic_type();
                        self.context.type_table.borrow_mut().create_type(
                            crate::tast::core::TypeKind::TypeAlias {
                                symbol_id: symbol.id,
                                target_type,
                                type_args: Vec::new(),
                            },
                        )
                    }
                    _ => {
                        // For other symbol kinds, keep as placeholder
                        continue;
                    }
                };

                // Record the mapping from placeholder to resolved type
                self.resolution_state
                    .placeholder_to_real
                    .insert(deferred_type.target_type_id, resolved_type);

                // TODO: Update all references to the placeholder type
                // For now, we've recorded the mapping which can be used later
            } else {
                // Still unresolved - only report errors for user-authored types.
                // Internal/synthetic stdlib references (file_id = u32::MAX) are
                // from lazy-loaded stdlib files whose transitive dependencies
                // may not be loaded yet. These are not user errors.
                let is_internal = deferred_type.location.file_id == u32::MAX
                    && deferred_type.location.byte_offset == 0;
                if !is_internal {
                    self.context.errors.push(LoweringError::UnresolvedType {
                        type_name: deferred_type.type_name,
                        location: deferred_type.location,
                    });
                }
                // Continue processing other deferred types
            }
        }

        Ok(())
    }

    /// Lower a switch case
    /// Lower a switch case for expression context (where case body is an expression)
    fn lower_switch_case_expression(
        &mut self,
        case: &parser::Case,
    ) -> Result<TypedSwitchCase, LoweringError> {
        // For switch expressions, the case body should be an expression
        let case_value = if let Some(first_pattern) = case.patterns.first() {
            // Check if this is a complex pattern that requires variable binding
            if self.pattern_has_variables(first_pattern) {
                // Create a new scope for this case to bind pattern variables
                let case_scope = self
                    .context
                    .scope_tree
                    .create_scope(Some(self.context.current_scope));
                let prev_scope = self.context.current_scope;
                self.context.current_scope = case_scope;

                // Bind pattern variables in the new scope
                let var_bindings = self.bind_pattern_variables(first_pattern)?;

                // For constructor patterns, create the constructor expression
                let case_expr =
                    self.create_constructor_expression_with_bindings(first_pattern, var_bindings)?;

                // Lower case body as expression in the new scope with bound variables
                let body_expr = self.lower_expression(&case.body)?;

                // Restore previous scope
                self.context.current_scope = prev_scope;

                let body = TypedStatement::Expression {
                    expression: body_expr,
                    source_location: self.context.span_to_location(&case.span),
                };

                return Ok(TypedSwitchCase {
                    case_value: case_expr,
                    body,
                    source_location: self.context.span_to_location(&case.span),
                });
            } else {
                self.lower_pattern_to_expression(first_pattern)?
            }
        } else {
            return Err(LoweringError::IncompleteImplementation {
                feature: "Empty switch case patterns".to_string(),
                location: self.context.span_to_location(&case.span),
            });
        };

        // Lower case body as expression
        let body_expr = self.lower_expression(&case.body)?;

        // Convert to statement for compatibility
        let body = TypedStatement::Expression {
            expression: body_expr,
            source_location: self.context.span_to_location(&case.span),
        };

        Ok(TypedSwitchCase {
            case_value,
            body,
            source_location: self.context.span_to_location(&case.span),
        })
    }

    fn lower_switch_case(&mut self, case: &parser::Case) -> Result<TypedSwitchCase, LoweringError> {
        // For now, use the first pattern as the case value
        // TODO: Handle multiple patterns and guards properly
        let case_value = if let Some(first_pattern) = case.patterns.first() {
            // Check if this is a complex pattern that requires variable binding
            if self.pattern_has_variables(first_pattern) {
                // Create a new scope for this case to bind pattern variables
                let case_scope = self
                    .context
                    .scope_tree
                    .create_scope(Some(self.context.current_scope));
                let prev_scope = self.context.current_scope;
                self.context.current_scope = case_scope;

                // Bind pattern variables in the new scope
                let var_bindings = self.bind_pattern_variables(first_pattern)?;

                // For constructor patterns, create the constructor expression
                let case_expr =
                    self.create_constructor_expression_with_bindings(first_pattern, var_bindings)?;

                // Lower case body in the new scope with bound variables
                let body = self.lower_expression_to_statement(&case.body)?;

                // Restore previous scope
                self.context.current_scope = prev_scope;

                return Ok(TypedSwitchCase {
                    case_value: case_expr,
                    body,
                    source_location: self.context.span_to_location(&case.span),
                });
            } else {
                // Simple patterns can be converted to expressions directly
                self.lower_pattern_to_expression(first_pattern)?
            }
        } else {
            return Err(LoweringError::IncompleteImplementation {
                feature: "Empty switch case patterns".to_string(),
                location: self.context.span_to_location(&case.span),
            });
        };

        // Lower case body as statement
        let body = self.lower_expression_to_statement(&case.body)?;

        Ok(TypedSwitchCase {
            case_value,
            body,
            source_location: self.context.span_to_location(&case.span),
        })
    }

    /// Check if a pattern contains variables that need binding
    fn pattern_has_variables(&self, pattern: &parser::Pattern) -> bool {
        use parser::Pattern;
        match pattern {
            Pattern::Var(_) => true,
            Pattern::Constructor { params, .. } => {
                // If the constructor has parameters, check if any are variables
                params.iter().any(|p| self.pattern_has_variables(p))
            }
            Pattern::Array(patterns) => {
                // Check if any array element patterns have variables
                patterns.iter().any(|p| self.pattern_has_variables(p))
            }
            Pattern::ArrayRest { elements, rest, .. } => {
                // Check elements and rest variable
                rest.is_some() || elements.iter().any(|p| self.pattern_has_variables(p))
            }
            Pattern::Object { fields, .. } => {
                // Check if any field patterns have variables
                fields.iter().any(|(_, p)| self.pattern_has_variables(p))
            }
            Pattern::Type { var, .. } => {
                // Type patterns always bind a variable
                true
            }
            Pattern::Or(patterns) => {
                // Or patterns can have variables in any branch
                patterns.iter().any(|p| self.pattern_has_variables(p))
            }
            Pattern::Const(_) | Pattern::Null | Pattern::Underscore => false,
            Pattern::Extractor { .. } => {
                // Extractors might bind variables - for now assume they do
                true
            }
        }
    }

    /// Bind pattern variables in the current scope
    fn bind_pattern_variables(
        &mut self,
        pattern: &parser::Pattern,
    ) -> Result<Vec<(InternedString, SymbolId)>, LoweringError> {
        use parser::Pattern;
        match pattern {
            Pattern::Var(var_name) => {
                // Create a variable symbol for the pattern variable
                let interned_name = self.context.intern_string(var_name);
                let var_symbol = self
                    .context
                    .symbol_table
                    .create_variable_in_scope(interned_name, self.context.current_scope);

                // Add to current scope
                self.context
                    .scope_tree
                    .get_scope_mut(self.context.current_scope)
                    .expect("Current scope should exist")
                    .add_symbol(var_symbol, interned_name);

                Ok(vec![(interned_name, var_symbol)])
            }
            Pattern::Constructor { params, .. } => {
                // Recursively bind variables in constructor parameters
                let mut bindings = Vec::new();
                for param in params {
                    bindings.extend(self.bind_pattern_variables(param)?);
                }
                Ok(bindings)
            }
            Pattern::Array(patterns) => {
                // Bind variables in each array element pattern
                let mut bindings = Vec::new();
                for pattern in patterns {
                    bindings.extend(self.bind_pattern_variables(pattern)?);
                }
                Ok(bindings)
            }
            Pattern::ArrayRest { elements, rest, .. } => {
                // Bind variables in element patterns
                let mut bindings = Vec::new();
                for pattern in elements {
                    bindings.extend(self.bind_pattern_variables(pattern)?);
                }

                // Bind the rest variable if present
                if let Some(rest_var) = rest {
                    let interned_name = self.context.intern_string(rest_var);
                    let var_symbol = self
                        .context
                        .symbol_table
                        .create_variable_in_scope(interned_name, self.context.current_scope);
                    self.context
                        .scope_tree
                        .get_scope_mut(self.context.current_scope)
                        .expect("Current scope should exist")
                        .add_symbol(var_symbol, interned_name);
                    bindings.push((interned_name, var_symbol));
                }
                Ok(bindings)
            }
            Pattern::Object { fields, .. } => {
                // Bind variables in each field pattern
                let mut bindings = Vec::new();
                for (_, field_pattern) in fields {
                    bindings.extend(self.bind_pattern_variables(field_pattern)?);
                }
                Ok(bindings)
            }
            Pattern::Type { var, .. } => {
                // Bind the typed variable
                let interned_name = self.context.intern_string(var);
                let var_symbol = self
                    .context
                    .symbol_table
                    .create_variable_in_scope(interned_name, self.context.current_scope);
                self.context
                    .scope_tree
                    .get_scope_mut(self.context.current_scope)
                    .expect("Current scope should exist")
                    .add_symbol(var_symbol, interned_name);
                Ok(vec![(interned_name, var_symbol)])
            }
            Pattern::Or(patterns) => {
                // For OR patterns, all branches must bind the same variables
                // This is a complex case - for now just bind variables from the first branch
                let mut bindings = Vec::new();
                if let Some(first_pattern) = patterns.first() {
                    bindings = self.bind_pattern_variables(first_pattern)?;
                }
                // TODO: Validate that all branches bind the same variables
                Ok(bindings)
            }
            Pattern::Const(_) | Pattern::Null | Pattern::Underscore => {
                // These patterns don't bind variables
                Ok(vec![])
            }
            Pattern::Extractor { .. } => {
                // Extractors are complex - for now skip binding
                // TODO: Implement extractor pattern variable binding
                Ok(vec![])
            }
        }
    }

    /// Try to resolve an enum constructor using the switch discriminant type
    /// This is needed for Haxe pattern matching where `case Some(v):` needs to
    /// be resolved as `Option.Some` based on the switch expression's type
    fn resolve_enum_constructor_from_discriminant(
        &self,
        constructor_name: InternedString,
    ) -> Option<SymbolId> {
        // Get the current switch discriminant type
        let discriminant_type = self.context.switch_discriminant_type?;

        // Get the type to find the enum symbol
        let type_table = self.context.type_table.borrow();

        // Recursively unwrap GenericInstance to find the base enum
        // Handles nested generics like Option<Option<Int>>
        let mut current_type_id = discriminant_type;
        let enum_symbol = loop {
            let ty = type_table.get(current_type_id)?;
            match &ty.kind {
                crate::tast::core::TypeKind::Enum { symbol_id, .. } => break *symbol_id,
                crate::tast::core::TypeKind::GenericInstance { base_type, .. } => {
                    // Continue unwrapping to find the base enum
                    current_type_id = *base_type;
                }
                _ => return None,
            }
        };
        drop(type_table);

        // Look up the enum's variants
        let variants = self.context.symbol_table.get_enum_variants(enum_symbol)?;

        // Find the variant with the matching name
        for &variant_id in variants {
            if let Some(variant_symbol) = self.context.symbol_table.get_symbol(variant_id) {
                if variant_symbol.name == constructor_name {
                    return Some(variant_id);
                }
            }
        }

        None
    }

    /// Create a constructor expression for pattern matching
    fn create_constructor_expression(
        &mut self,
        pattern: &parser::Pattern,
    ) -> Result<TypedExpression, LoweringError> {
        self.create_constructor_expression_with_bindings(pattern, vec![])
    }

    /// Create a constructor expression with pre-resolved variable bindings
    fn create_constructor_expression_with_bindings(
        &mut self,
        pattern: &parser::Pattern,
        variable_bindings: Vec<(InternedString, SymbolId)>,
    ) -> Result<TypedExpression, LoweringError> {
        use parser::Pattern;
        match pattern {
            Pattern::Constructor { path, params } => {
                // Resolve the constructor symbol
                let constructor_name = self.context.intern_string(&path.name);

                // First try to resolve from switch discriminant type (for enum pattern matching)
                // Then fall back to scope hierarchy lookup
                let constructor_symbol = self
                    .resolve_enum_constructor_from_discriminant(constructor_name)
                    .or_else(|| self.resolve_symbol_in_scope_hierarchy(constructor_name))
                    .ok_or_else(|| LoweringError::UnresolvedSymbol {
                        name: path.name.clone(),
                        location: SourceLocation::new(0, 0, 0, 0),
                    })?;

                if params.is_empty() {
                    // Simple constructor like Red, Green, Blue
                    let constructor_type = if let Some(symbol) =
                        self.context.symbol_table.get_symbol(constructor_symbol)
                    {
                        symbol.type_id
                    } else {
                        self.context.type_table.borrow().dynamic_type()
                    };

                    Ok(TypedExpression {
                        kind: TypedExpressionKind::Variable {
                            symbol_id: constructor_symbol,
                        },
                        expr_type: constructor_type,
                        usage: VariableUsage::Borrow,
                        lifetime_id: LifetimeId::from_raw(1),
                        source_location: SourceLocation::new(0, 0, 0, 0),
                        metadata: ExpressionMetadata::default(),
                    })
                } else {
                    // Constructor with parameters - use pattern placeholder for complex patterns
                    self.create_pattern_placeholder_with_bindings(pattern, variable_bindings)
                }
            }
            Pattern::ArrayRest { .. } | Pattern::Object { .. } | Pattern::Type { .. } => {
                // Complex patterns that need later compilation
                self.create_pattern_placeholder_with_bindings(pattern, variable_bindings)
            }
            Pattern::Var(name) => {
                // Check if this identifier is actually an enum variant (e.g., "None")
                // The parser produces Var for bare identifiers like `case None:`
                let interned_name = self.context.intern_string(name);
                if let Some(variant_sym_id) =
                    self.resolve_enum_constructor_from_discriminant(interned_name)
                {
                    let ct = self
                        .context
                        .symbol_table
                        .get_symbol(variant_sym_id)
                        .map(|s| s.type_id)
                        .unwrap_or_else(|| self.context.type_table.borrow().dynamic_type());
                    Ok(TypedExpression {
                        kind: TypedExpressionKind::Variable {
                            symbol_id: variant_sym_id,
                        },
                        expr_type: ct,
                        usage: VariableUsage::Borrow,
                        lifetime_id: LifetimeId::from_raw(1),
                        source_location: SourceLocation::new(0, 0, 0, 0),
                        metadata: ExpressionMetadata::default(),
                    })
                } else {
                    self.lower_pattern_to_expression(pattern)
                }
            }
            _ => {
                // For simple patterns, fall back to regular pattern conversion
                self.lower_pattern_to_expression(pattern)
            }
        }
    }

    /// Create a pattern placeholder for complex patterns that need later compilation
    fn create_pattern_placeholder(
        &mut self,
        pattern: &parser::Pattern,
    ) -> Result<TypedExpression, LoweringError> {
        self.create_pattern_placeholder_with_bindings(pattern, vec![])
    }

    /// Create a pattern placeholder with pre-resolved variable bindings
    fn create_pattern_placeholder_with_bindings(
        &mut self,
        pattern: &parser::Pattern,
        variable_bindings: Vec<(InternedString, SymbolId)>,
    ) -> Result<TypedExpression, LoweringError> {
        let source_location = SourceLocation::new(0, 0, 0, 0); // TODO: get actual pattern location
        Ok(TypedExpression {
            kind: TypedExpressionKind::PatternPlaceholder {
                pattern: pattern.clone(),
                source_location,
                variable_bindings,
            },
            expr_type: self.context.type_table.borrow().dynamic_type(),
            usage: VariableUsage::Borrow,
            lifetime_id: LifetimeId::from_raw(1),
            source_location,
            metadata: ExpressionMetadata::default(),
        })
    }

    /// Convert a pattern to an expression for case values
    fn lower_pattern_to_expression(
        &mut self,
        pattern: &parser::Pattern,
    ) -> Result<TypedExpression, LoweringError> {
        use parser::Pattern;

        match pattern {
            Pattern::Const(expr) => {
                // Convert constant expression directly
                self.lower_expression(expr)
            }
            Pattern::Var(name) => {
                // Check if this identifier is actually an enum variant (e.g., "None")
                let interned_name = self.context.intern_string(name);
                if let Some(variant_sym_id) =
                    self.resolve_enum_constructor_from_discriminant(interned_name)
                {
                    let ct = self
                        .context
                        .symbol_table
                        .get_symbol(variant_sym_id)
                        .map(|s| s.type_id)
                        .unwrap_or_else(|| self.context.type_table.borrow().dynamic_type());
                    return Ok(TypedExpression {
                        kind: TypedExpressionKind::Variable {
                            symbol_id: variant_sym_id,
                        },
                        expr_type: ct,
                        usage: VariableUsage::Borrow,
                        lifetime_id: LifetimeId::from_raw(1),
                        source_location: SourceLocation::new(0, 0, 0, 0),
                        metadata: ExpressionMetadata::default(),
                    });
                }

                // Variable patterns bind a new variable in the case body
                let var_symbol = self.context.symbol_table.create_variable(interned_name);

                // Register in current scope for the case body
                let current_scope = self.context.current_scope;
                if let Some(scope) = self.context.scope_tree.get_scope_mut(current_scope) {
                    scope.add_symbol(var_symbol, interned_name);
                }

                // Return a wildcard pattern expression
                Ok(TypedExpression {
                    kind: TypedExpressionKind::Null, // Placeholder for wildcard
                    expr_type: self.context.type_table.borrow().dynamic_type(),
                    usage: VariableUsage::Borrow,
                    lifetime_id: LifetimeId::first(),
                    source_location: self.context.create_location(),
                    metadata: ExpressionMetadata::default(),
                })
            }
            Pattern::Constructor { path, params } => {
                // Resolve the constructor symbol
                let constructor_name = self.context.intern_string(&path.name);

                // First try to resolve from switch discriminant type (for enum pattern matching)
                // Then fall back to scope hierarchy lookup
                let constructor_symbol = self
                    .resolve_enum_constructor_from_discriminant(constructor_name)
                    .or_else(|| self.resolve_symbol_in_scope_hierarchy(constructor_name))
                    .ok_or_else(|| LoweringError::UnresolvedSymbol {
                        name: path.name.clone(),
                        location: SourceLocation::new(0, 0, 0, 0),
                    })?;

                if params.is_empty() {
                    // Simple constructor like Red, Green, Blue
                    let constructor_var = TypedExpressionKind::Variable {
                        symbol_id: constructor_symbol,
                    };

                    // Get the constructor's type
                    let constructor_type = if let Some(symbol) =
                        self.context.symbol_table.get_symbol(constructor_symbol)
                    {
                        symbol.type_id
                    } else {
                        self.context.type_table.borrow().dynamic_type()
                    };

                    Ok(TypedExpression {
                        kind: constructor_var,
                        expr_type: constructor_type,
                        usage: VariableUsage::Borrow,
                        lifetime_id: LifetimeId::from_raw(1),
                        source_location: SourceLocation::new(0, 0, 0, 0),
                        metadata: ExpressionMetadata::default(),
                    })
                } else {
                    // Constructor with parameters like RGB(255, 0, 0)
                    let mut arg_exprs = Vec::new();
                    for param_pattern in params {
                        let arg_expr = self.lower_pattern_to_expression(param_pattern)?;
                        arg_exprs.push(arg_expr);
                    }

                    // Get the constructor's type
                    let constructor_type = if let Some(symbol) =
                        self.context.symbol_table.get_symbol(constructor_symbol)
                    {
                        symbol.type_id
                    } else {
                        self.context.type_table.borrow().dynamic_type()
                    };

                    // Create the constructor variable expression
                    let mut constructor_expr = TypedExpression {
                        kind: TypedExpressionKind::Variable {
                            symbol_id: constructor_symbol,
                        },
                        expr_type: constructor_type,
                        usage: VariableUsage::Borrow,
                        lifetime_id: LifetimeId::from_raw(1),
                        source_location: SourceLocation::new(0, 0, 0, 0),
                        metadata: ExpressionMetadata::default(),
                    };

                    // Check if this is a generic enum constructor and instantiate its type
                    if let Some(symbol) = self.context.symbol_table.get_symbol(constructor_symbol) {
                        if symbol.kind == crate::tast::symbols::SymbolKind::EnumVariant {
                            constructor_expr = self.instantiate_enum_constructor_type(
                                constructor_symbol,
                                &arg_exprs,
                                constructor_expr,
                            )?;
                        }
                    }

                    Ok(TypedExpression {
                        kind: TypedExpressionKind::FunctionCall {
                            function: Box::new(constructor_expr),
                            arguments: arg_exprs,
                            type_arguments: Vec::new(),
                        },
                        expr_type: self.context.type_table.borrow().dynamic_type(), // Will be updated by type inference
                        usage: VariableUsage::Borrow,
                        lifetime_id: LifetimeId::from_raw(1),
                        source_location: SourceLocation::new(0, 0, 0, 0),
                        metadata: ExpressionMetadata::default(),
                    })
                }
            }
            Pattern::Array(patterns) => {
                // Array patterns like [1, 2, 3]
                let mut elements = Vec::new();
                for pattern in patterns {
                    elements.push(self.lower_pattern_to_expression(pattern)?);
                }

                Ok(TypedExpression {
                    kind: TypedExpressionKind::ArrayLiteral { elements },
                    expr_type: self.context.type_table.borrow().dynamic_type(),
                    usage: VariableUsage::Borrow,
                    lifetime_id: LifetimeId::from_raw(1),
                    source_location: SourceLocation::new(0, 0, 0, 0),
                    metadata: ExpressionMetadata::default(),
                })
            }
            Pattern::Null => {
                // Null pattern
                Ok(TypedExpression {
                    kind: TypedExpressionKind::Null,
                    expr_type: self.context.type_table.borrow().dynamic_type(),
                    usage: VariableUsage::Borrow,
                    lifetime_id: LifetimeId::from_raw(1),
                    source_location: SourceLocation::new(0, 0, 0, 0),
                    metadata: ExpressionMetadata::default(),
                })
            }
            Pattern::Underscore => {
                // Wildcard pattern - for now create a special marker
                // In a full implementation, this would need special handling in switch
                Ok(TypedExpression {
                    kind: TypedExpressionKind::Literal {
                        value: LiteralValue::Bool(true), // Placeholder for wildcard
                    },
                    expr_type: self.context.type_table.borrow().dynamic_type(),
                    usage: VariableUsage::Borrow,
                    lifetime_id: LifetimeId::from_raw(1),
                    source_location: SourceLocation::new(0, 0, 0, 0),
                    metadata: ExpressionMetadata::default(),
                })
            }
            Pattern::Or(patterns) => {
                // Or patterns like 1 | 2 | 3
                // For now, just use the first pattern
                // TODO: Proper OR pattern handling requires different switch compilation
                if let Some(first) = patterns.first() {
                    self.lower_pattern_to_expression(first)
                } else {
                    Err(LoweringError::IncompleteImplementation {
                        feature: "Empty OR pattern".to_string(),
                        location: SourceLocation::new(0, 0, 0, 0),
                    })
                }
            }
            Pattern::Object { fields } => {
                // Object pattern: {x: 42, y: "hello"}
                // Convert to object literal expression
                let mut typed_fields = Vec::new();

                for (field_name, field_pattern) in fields {
                    // Recursively convert the field pattern to expression
                    let field_expr = self.lower_pattern_to_expression(field_pattern)?;
                    let interned_name = self.context.intern_string(field_name);

                    typed_fields.push(TypedObjectField {
                        name: interned_name,
                        value: field_expr,
                        source_location: SourceLocation::new(0, 0, 0, 0),
                    });
                }

                let field_types: Vec<(InternedString, TypeId)> = typed_fields
                    .iter()
                    .map(|f| (f.name, f.value.expr_type))
                    .collect();

                let kind = TypedExpressionKind::ObjectLiteral {
                    fields: typed_fields,
                };

                Ok(TypedExpression {
                    kind,
                    expr_type: {
                        // Extract field types for type inference

                        type_resolution::infer_object_literal_type(
                            &self.context.type_table,
                            &field_types,
                        )
                    },
                    usage: VariableUsage::Borrow,
                    lifetime_id: LifetimeId::from_raw(1),
                    source_location: SourceLocation::new(0, 0, 0, 0),
                    metadata: ExpressionMetadata::default(),
                })
            }

            Pattern::ArrayRest { elements, rest } => {
                // Array rest pattern: [first, ...rest]
                // Convert elements to expressions
                let mut typed_elements = Vec::new();

                for element_pattern in elements {
                    let element_expr = self.lower_pattern_to_expression(element_pattern)?;
                    typed_elements.push(element_expr);
                }

                // Handle the rest variable if present
                if let Some(rest_name) = rest {
                    // Create a variable expression for the rest binding
                    let rest_interned = self.context.intern_string(rest_name);

                    // Look up or create symbol for rest variable
                    let rest_symbol = if let Some(symbol_id) =
                        self.resolve_symbol_in_scope_hierarchy(rest_interned)
                    {
                        symbol_id
                    } else {
                        // Create new variable symbol in current scope
                        let symbol_id = self
                            .context
                            .symbol_table
                            .create_variable_in_scope(rest_interned, self.context.current_scope);

                        self.context
                            .scope_tree
                            .get_scope_mut(self.context.current_scope)
                            .expect("Current scope should exist")
                            .add_symbol(symbol_id, rest_interned);

                        symbol_id
                    };

                    let rest_expr = TypedExpression {
                        kind: TypedExpressionKind::Variable {
                            symbol_id: rest_symbol,
                        },
                        expr_type: self
                            .context
                            .type_table
                            .borrow_mut()
                            .create_array_type(self.context.type_table.borrow().dynamic_type()), // Array type with dynamic elements
                        usage: VariableUsage::Borrow,
                        lifetime_id: LifetimeId::from_raw(1),
                        source_location: SourceLocation::new(0, 0, 0, 0),
                        metadata: ExpressionMetadata::default(),
                    };

                    // Add rest to the elements
                    typed_elements.push(rest_expr);
                }

                let kind = TypedExpressionKind::ArrayLiteral {
                    elements: typed_elements,
                };

                Ok(TypedExpression {
                    kind,
                    expr_type: self
                        .context
                        .type_table
                        .borrow_mut()
                        .create_array_type(self.context.type_table.borrow().dynamic_type()), // Array type with dynamic elements
                    usage: VariableUsage::Borrow,
                    lifetime_id: LifetimeId::from_raw(1),
                    source_location: SourceLocation::new(0, 0, 0, 0),
                    metadata: ExpressionMetadata::default(),
                })
            }

            Pattern::Type { var, type_hint } => {
                // Type pattern: (s:String)
                // Create a variable expression with type constraint
                let var_interned = self.context.intern_string(var);

                // Look up or create symbol for the typed variable
                let var_symbol =
                    if let Some(symbol_id) = self.resolve_symbol_in_scope_hierarchy(var_interned) {
                        symbol_id
                    } else {
                        // Create new variable symbol in current scope
                        let symbol_id = self
                            .context
                            .symbol_table
                            .create_variable_in_scope(var_interned, self.context.current_scope);

                        self.context
                            .scope_tree
                            .get_scope_mut(self.context.current_scope)
                            .expect("Current scope should exist")
                            .add_symbol(symbol_id, var_interned);

                        symbol_id
                    };

                // Resolve the type hint to get the proper type
                let resolved_type = self.lower_type(type_hint)?;

                let kind = TypedExpressionKind::Variable {
                    symbol_id: var_symbol,
                };

                Ok(TypedExpression {
                    kind,
                    expr_type: resolved_type, // Use the type constraint from the pattern
                    usage: VariableUsage::Borrow,
                    lifetime_id: LifetimeId::from_raw(1),
                    source_location: SourceLocation::new(0, 0, 0, 0),
                    metadata: ExpressionMetadata::default(),
                })
            }

            Pattern::Extractor { .. } => {
                // Extractor patterns require runtime evaluation - not implemented
                Err(LoweringError::IncompleteImplementation {
                    feature: format!("Extractor pattern to expression conversion: {:?}", pattern),
                    location: SourceLocation::new(0, 0, 0, 0),
                })
            }
        }
    }

    /// Convert an expression to a statement
    fn lower_expression_to_statement(
        &mut self,
        expr: &parser::Expr,
    ) -> Result<TypedStatement, LoweringError> {
        let typed_expr = self.lower_expression(expr)?;
        Ok(TypedStatement::Expression {
            expression: typed_expr,
            source_location: self.context.span_to_location(&expr.span),
        })
    }

    /// Lower a catch clause
    fn lower_catch_clause(
        &mut self,
        catch: &parser::Catch,
    ) -> Result<TypedCatchClause, LoweringError> {
        // Create a new scope for the catch block
        let catch_scope = self.context.enter_scope(ScopeKind::Block);

        // Create symbol for exception variable in the catch scope
        let var_name = self.context.string_interner.intern(&catch.var);
        let var_symbol = self
            .context
            .symbol_table
            .create_variable_in_scope(var_name, self.context.current_scope);

        // Resolve exception type
        let exception_type = if let Some(type_hint) = &catch.type_hint {
            self.lower_type(type_hint)?
        } else {
            // Default to dynamic type if no type specified
            self.context.type_table.borrow().dynamic_type()
        };

        // Set the catch variable's type so field accesses (e.g., e.length) resolve correctly
        self.context
            .symbol_table
            .update_symbol_type(var_symbol, exception_type);

        // Lower filter condition if present (in the catch scope where the exception var is available)
        let filter = if let Some(filter_expr) = &catch.filter {
            Some(self.lower_expression(filter_expr)?)
        } else {
            None
        };

        // Lower catch handler body (in the catch scope where the exception var is available)
        let handler = self.lower_expression(&catch.body)?;

        // Exit the catch scope
        self.context.exit_scope();

        Ok(TypedCatchClause {
            exception_variable: var_symbol,
            exception_type,
            filter,
            body: TypedStatement::Expression {
                expression: handler,
                source_location: self.context.create_location(),
            },
            source_location: self.context.create_location(),
        })
    }

    /// Lower a function parameter
    fn lower_function_param(
        &mut self,
        param: &parser::FunctionParam,
    ) -> Result<TypedParameter, LoweringError> {
        // Create symbol for parameter in the current scope
        let param_name = self.context.string_interner.intern(&param.name);
        let param_symbol = self
            .context
            .symbol_table
            .create_variable_in_scope(param_name, self.context.current_scope);

        // Resolve parameter type
        let param_type = if let Some(type_hint) = &param.type_hint {
            self.lower_type(type_hint)?
        } else {
            self.context.type_table.borrow().dynamic_type()
        };

        // Update the symbol with its type
        self.context
            .symbol_table
            .update_symbol_type(param_symbol, param_type);

        // Lower default value if present
        let default_value = if let Some(default_expr) = &param.default_value {
            Some(self.lower_expression(default_expr)?)
        } else {
            None
        };

        Ok(TypedParameter {
            symbol_id: param_symbol,
            name: self.context.string_interner.intern(&param.name),
            param_type,
            is_optional: param.optional,
            default_value,
            mutability: crate::tast::symbols::Mutability::Immutable, // Function parameters are immutable by default in Haxe
            source_location: self.context.span_to_location(&param.span),
        })
    }

    /// Infer the type of built-in methods like Array.push, String.charAt, etc.
    fn infer_builtin_method_type(
        &mut self,
        receiver_type: TypeId,
        field_symbol: SymbolId,
    ) -> LoweringResult<TypeId> {
        // Get the field name from the symbol
        let field_name = if let Some(symbol) = self.context.symbol_table.get_symbol(field_symbol) {
            self.context
                .string_interner
                .get(symbol.name)
                .unwrap_or("<unknown>")
                .to_string()
        } else {
            return Ok(self.context.type_table.borrow().dynamic_type());
        };

        // Check the object type to see if it's a built-in type with known methods
        let type_table = self.context.type_table.borrow();
        if let Some(object_type_info) = type_table.get(receiver_type) {
            match &object_type_info.kind {
                crate::tast::core::TypeKind::Array { element_type } => {
                    match field_name.as_str() {
                        "push" => {
                            // push(item: T): Void
                            let void_type = type_table.void_type();
                            let element_type_copy = *element_type;
                            drop(type_table);
                            Ok(self
                                .context
                                .type_table
                                .borrow_mut()
                                .create_function_type(vec![element_type_copy], void_type))
                        }
                        "pop" => {
                            // pop(): T
                            Ok(*element_type)
                        }
                        "length" => {
                            // length: Int
                            Ok(type_table.int_type())
                        }
                        "map" => {
                            // map(f: (T) -> S): Array<S>
                            // For now, return Array<T> (same element type as input)
                            // The actual return element type depends on the callback,
                            // but we preserve the array type so trace/dispatch works.
                            let elem = *element_type;
                            drop(type_table);
                            let arr_type =
                                self.context.type_table.borrow_mut().create_array_type(elem);
                            let func_type = {
                                let tt = self.context.type_table.borrow();
                                let callback_type = tt.dynamic_type();
                                drop(tt);
                                self.context
                                    .type_table
                                    .borrow_mut()
                                    .create_function_type(vec![callback_type], arr_type)
                            };
                            Ok(func_type)
                        }
                        "filter" => {
                            // filter(f: (T) -> Bool): Array<T>
                            let elem = *element_type;
                            drop(type_table);
                            let arr_type =
                                self.context.type_table.borrow_mut().create_array_type(elem);
                            let func_type = {
                                let tt = self.context.type_table.borrow();
                                let callback_type = tt.dynamic_type();
                                drop(tt);
                                self.context
                                    .type_table
                                    .borrow_mut()
                                    .create_function_type(vec![callback_type], arr_type)
                            };
                            Ok(func_type)
                        }
                        "sort" => {
                            // sort(f: (T, T) -> Int): Void
                            let void_type = type_table.void_type();
                            let callback_type = type_table.dynamic_type();
                            drop(type_table);
                            Ok(self
                                .context
                                .type_table
                                .borrow_mut()
                                .create_function_type(vec![callback_type], void_type))
                        }
                        "indexOf" | "lastIndexOf" => {
                            // indexOf(x: T, ?fromIndex: Int): Int
                            Ok(type_table.int_type())
                        }
                        "contains" => {
                            // contains(x: T): Bool
                            Ok(type_table.bool_type())
                        }
                        "join" => {
                            // join(sep: String): String
                            Ok(type_table.string_type())
                        }
                        "slice" | "splice" | "concat" | "copy" | "reverse" => {
                            // Returns Array<T>
                            let elem = *element_type;
                            drop(type_table);
                            Ok(self.context.type_table.borrow_mut().create_array_type(elem))
                        }
                        "remove" => {
                            // remove(x: T): Bool
                            Ok(type_table.bool_type())
                        }
                        "insert" | "unshift" => {
                            // insert(pos: Int, x: T): Void
                            Ok(type_table.void_type())
                        }
                        "toString" => Ok(type_table.string_type()),
                        "iterator" | "keyValueIterator" => Ok(type_table.dynamic_type()),
                        _ => Ok(type_table.dynamic_type()),
                    }
                }
                crate::tast::core::TypeKind::String => {
                    match field_name.as_str() {
                        "length" => Ok(type_table.int_type()),
                        "charAt" => {
                            // charAt(index: Int): String
                            let string_type = type_table.string_type();
                            let int_type = type_table.int_type();
                            drop(type_table);
                            Ok(self
                                .context
                                .type_table
                                .borrow_mut()
                                .create_function_type(vec![int_type], string_type))
                        }
                        "toUpperCase" | "toLowerCase" | "toString" | "trim" => {
                            // toUpperCase(): String, toLowerCase(): String, toString(): String, trim(): String
                            let string_type = type_table.string_type();
                            drop(type_table);
                            Ok(self
                                .context
                                .type_table
                                .borrow_mut()
                                .create_function_type(vec![], string_type))
                        }
                        "substring" | "substr" => {
                            // substring(startIndex: Int, ?endIndex: Int): String
                            let string_type = type_table.string_type();
                            let int_type = type_table.int_type();
                            drop(type_table);
                            // For simplicity, we'll create a function that takes two Int parameters
                            Ok(self
                                .context
                                .type_table
                                .borrow_mut()
                                .create_function_type(vec![int_type, int_type], string_type))
                        }
                        "indexOf" | "lastIndexOf" => {
                            // indexOf(str: String, ?startIndex: Int): Int
                            let string_type = type_table.string_type();
                            let int_type = type_table.int_type();
                            drop(type_table);
                            Ok(self
                                .context
                                .type_table
                                .borrow_mut()
                                .create_function_type(vec![string_type], int_type))
                        }
                        "split" => {
                            // split(delimiter: String): Array<String>
                            let string_type = type_table.string_type();
                            drop(type_table);
                            let array_of_strings = self
                                .context
                                .type_table
                                .borrow_mut()
                                .create_array_type(string_type);
                            Ok(self
                                .context
                                .type_table
                                .borrow_mut()
                                .create_function_type(vec![string_type], array_of_strings))
                        }
                        _ => Ok(type_table.dynamic_type()),
                    }
                }
                crate::tast::core::TypeKind::Abstract {
                    symbol_id,
                    underlying,
                    ..
                } => {
                    // For abstracts (including @:forward), resolve through underlying type
                    let resolved_underlying =
                        underlying.or_else(|| type_table.resolve_abstract_underlying(*symbol_id));
                    if let Some(underlying_type) = resolved_underlying {
                        drop(type_table);
                        self.infer_builtin_method_type(underlying_type, field_symbol)
                    } else {
                        Ok(type_table.dynamic_type())
                    }
                }
                crate::tast::core::TypeKind::GenericInstance { base_type, .. } => {
                    // For generic instances, resolve through base type
                    let base = *base_type;
                    drop(type_table);
                    self.infer_builtin_method_type(base, field_symbol)
                }
                _ => Ok(type_table.dynamic_type()),
            }
        } else {
            Ok(type_table.dynamic_type())
        }
    }

    /// Lower a function body
    fn lower_function_body(
        &mut self,
        body: &parser::Expr,
    ) -> Result<Vec<TypedStatement>, LoweringError> {
        match &body.kind {
            parser::ExprKind::Block(elements) => {
                // Function body is a block - lower all elements with error recovery
                let mut statements = Vec::new();
                for element in elements {
                    match element {
                        parser::BlockElement::Expr(expr) => {
                            // Check if this is a variable declaration expression
                            match &expr.kind {
                                parser::ExprKind::Var { .. } | parser::ExprKind::Final { .. } => {
                                    // Variable declaration - lower as expression and convert to statement
                                    match self.lower_expression(expr) {
                                        Ok(typed_expr) => {
                                            // Extract the declaration info to create a proper statement
                                            if let TypedExpressionKind::VarDeclarationExpr {
                                                symbol_id,
                                                var_type,
                                                initializer,
                                            } = typed_expr.kind
                                            {
                                                statements.push(TypedStatement::VarDeclaration {
                                                    symbol_id,
                                                    var_type,
                                                    initializer: Some(*initializer),
                                                    mutability: crate::tast::symbols::Mutability::Mutable,
                                                    source_location: self
                                                        .context
                                                        .span_to_location(&expr.span),
                                                });
                                            } else if let TypedExpressionKind::FinalDeclarationExpr {
                                                symbol_id,
                                                var_type,
                                                initializer,
                                            } = typed_expr.kind
                                            {
                                                statements.push(TypedStatement::VarDeclaration {
                                                    symbol_id,
                                                    var_type,
                                                    initializer: Some(*initializer),
                                                    mutability: crate::tast::symbols::Mutability::Immutable,
                                                    source_location: self
                                                        .context
                                                        .span_to_location(&expr.span),
                                                });
                                            }
                                        }
                                        Err(e) => {
                                            // Collect error and continue processing other statements
                                            self.collected_errors.push(e);
                                        }
                                    }
                                }
                                _ => {
                                    // Regular expression - lower and wrap in statement
                                    match self.lower_expression(expr) {
                                        Ok(typed_expr) => {
                                            statements.push(TypedStatement::Expression {
                                                expression: typed_expr,
                                                source_location: self
                                                    .context
                                                    .span_to_location(&expr.span),
                                            });
                                        }
                                        Err(e) => {
                                            // Collect error and continue processing other statements
                                            self.collected_errors.push(e);
                                        }
                                    }
                                }
                            }
                        }
                        parser::BlockElement::Import(_)
                        | parser::BlockElement::Using(_)
                        | parser::BlockElement::Conditional(_) => {
                            // Skip imports, using statements, and conditional compilation for now
                            // These should be handled at the module level
                        }
                    }
                }
                Ok(statements)
            }
            _ => {
                // Single expression body - wrap in expression statement
                match self.lower_expression(body) {
                    Ok(typed_expr) => Ok(vec![TypedStatement::Expression {
                        expression: typed_expr,
                        source_location: self.context.span_to_location(&body.span),
                    }]),
                    Err(e) => {
                        // Collect error and return empty statement list
                        self.collected_errors.push(e);
                        Ok(vec![])
                    }
                }
            }
        }
    }

    /// Process @:overload metadata to extract method overload signatures
    fn process_overload_metadata(
        &mut self,
        metadata: &[parser::Metadata],
    ) -> LoweringResult<Vec<MethodOverload>> {
        let mut overload_signatures = Vec::new();

        for meta in metadata {
            if meta.name == "overload" {
                // @:overload(param1:Type1, param2:Type2 -> ReturnType)
                // For now, we'll implement a simplified version that parses function signature strings
                if meta.params.len() == 1 {
                    if let parser::ExprKind::String(signature_str) = &meta.params[0].kind {
                        // Parse the signature string to extract types
                        // This is a simplified implementation - a full parser would be more robust
                        if let Some(overload) =
                            self.parse_overload_signature(signature_str, &meta.span)?
                        {
                            overload_signatures.push(overload);
                        }
                    }
                }
            }
        }

        Ok(overload_signatures)
    }

    /// Process @:op metadata for operator overloading
    /// Extracts operator expressions like "A + B", "A * B", etc.
    fn process_operator_metadata(
        &mut self,
        metadata: &[parser::Metadata],
    ) -> LoweringResult<Vec<(String, Vec<String>)>> {
        let mut operator_metadata = Vec::new();

        for meta in metadata {
            if meta.name == "op" {
                // @:op(A + B) - operator expression is the first parameter
                if !meta.params.is_empty() {
                    // Extract the operator expression as a string
                    let operator_expr = self.expr_to_string(&meta.params[0]);

                    // Store the operator expression and any additional parameters
                    let additional_params: Vec<String> = meta.params[1..]
                        .iter()
                        .map(|e| self.expr_to_string(e))
                        .collect();

                    operator_metadata.push((operator_expr, additional_params));
                }
            }
        }

        Ok(operator_metadata)
    }

    /// Check if function has @:arrayAccess metadata
    fn has_array_access_metadata(&self, metadata: &[parser::Metadata]) -> bool {
        metadata.iter().any(|m| m.name == "arrayAccess")
    }

    /// Check if a type is SIMD4f (by native_name or symbol name).
    fn is_simd4f_type(&self, ty: crate::tast::TypeId) -> bool {
        use crate::tast::core::TypeKind;
        let type_table = self.context.type_table.borrow();
        let sym_id = type_table.get(ty).and_then(|ti| match &ti.kind {
            TypeKind::Abstract { symbol_id, .. } | TypeKind::Class { symbol_id, .. } => {
                Some(*symbol_id)
            }
            _ => None,
        });
        if let Some(sid) = sym_id {
            self.context
                .symbol_table
                .get_symbol(sid)
                .map(|s| {
                    let by_native = s
                        .native_name
                        .and_then(|nn| self.context.string_interner.get(nn))
                        .map(|n| n == "rayzor::SIMD4f")
                        .unwrap_or(false);
                    let by_name = self
                        .context
                        .string_interner
                        .get(s.name)
                        .map(|n| n == "SIMD4f")
                        .unwrap_or(false);
                    by_native || by_name
                })
                .unwrap_or(false)
        } else {
            false
        }
    }

    /// Check if a type is an abstract type (any abstract, not just SIMD4f).
    fn is_abstract_type(&self, ty: crate::tast::TypeId) -> bool {
        use crate::tast::core::TypeKind;
        let type_table = self.context.type_table.borrow();
        matches!(
            type_table.get(ty).map(|ti| &ti.kind),
            Some(TypeKind::Abstract { .. })
        )
    }

    /// Try to desugar a tuple literal to a static method call (e.g., SIMD4f.make()).
    /// Returns Ok(Some(expr)) if desugared, Ok(None) if the target type doesn't support tuple construction.
    fn try_desugar_tuple_to_make(
        &mut self,
        elements: &[parser::Expr],
        target_ty: crate::tast::TypeId,
        original_expr: &parser::Expr,
    ) -> LoweringResult<Option<TypedExpression>> {
        use crate::tast::core::TypeKind;

        // Check if target type is an abstract or class with a known native name
        let (class_symbol_id, native_name) = {
            let type_table = self.context.type_table.borrow();
            if let Some(type_info) = type_table.get(target_ty) {
                let sym_id = match &type_info.kind {
                    TypeKind::Abstract { symbol_id, .. } => Some(*symbol_id),
                    TypeKind::Class { symbol_id, .. } => Some(*symbol_id),
                    _ => None,
                };
                if let Some(sid) = sym_id {
                    let nn = self.context.symbol_table.get_symbol(sid).and_then(|s| {
                        s.native_name
                            .and_then(|nn| self.context.string_interner.get(nn))
                            .map(|s| s.to_string())
                    });
                    (Some(sid), nn)
                } else {
                    (None, None)
                }
            } else {
                (None, None)
            }
        };

        // Currently only SIMD4f supports tuple construction
        // Check by native_name or symbol name
        let is_simd4f = match (class_symbol_id, native_name.as_deref()) {
            (Some(_), Some("rayzor::SIMD4f")) => true,
            (Some(sid), _) => {
                // Fallback: check symbol name directly
                self.context
                    .symbol_table
                    .get_symbol(sid)
                    .and_then(|s| self.context.string_interner.get(s.name))
                    .map(|n| n == "SIMD4f")
                    .unwrap_or(false)
            }
            _ => false,
        };
        let class_symbol_id = match (is_simd4f, class_symbol_id) {
            (true, Some(sid)) => sid,
            _ => return Ok(None),
        };

        // Validate element count
        if elements.len() != 4 {
            return Err(LoweringError::InternalError {
                message: format!(
                    "SIMD4f tuple literal requires exactly 4 elements, got {}",
                    elements.len()
                ),
                location: self.context.span_to_location(&original_expr.span),
            });
        }

        // AST-level rewriting: construct a synthetic `SIMD4f.make(e1, e2, e3, e4)` expression
        // and lower it through the normal path, which handles static method resolution.
        let span = original_expr.span;
        let simd_ident = parser::Expr {
            kind: parser::ExprKind::Ident("SIMD4f".to_string()),
            span,
        };
        let field_access = parser::Expr {
            kind: parser::ExprKind::Field {
                expr: Box::new(simd_ident),
                field: "make".to_string(),
                is_optional: false,
            },
            span,
        };
        let args: Vec<parser::Expr> = elements.to_vec();
        let make_call = parser::Expr {
            kind: parser::ExprKind::Call {
                expr: Box::new(field_access),
                args,
            },
            span,
        };

        let lowered = self.lower_expression(&make_call)?;
        Ok(Some(lowered))
    }

    /// Convert a parser expression to a string representation
    /// Used for extracting metadata parameter values
    fn expr_to_string(&self, expr: &parser::Expr) -> String {
        match &expr.kind {
            parser::ExprKind::String(s) => s.clone(),
            parser::ExprKind::Ident(id) => id.clone(),
            parser::ExprKind::Int(n) => n.to_string(),
            parser::ExprKind::Float(f) => f.to_string(),
            parser::ExprKind::Bool(b) => b.to_string(),
            parser::ExprKind::Binary { left, op, right } => {
                format!(
                    "{} {:?} {}",
                    self.expr_to_string(left),
                    op,
                    self.expr_to_string(right)
                )
            }
            parser::ExprKind::Unary { op, expr: operand } => {
                format!("{:?}{}", op, self.expr_to_string(operand))
            }
            parser::ExprKind::Paren(inner) => {
                format!("({})", self.expr_to_string(inner))
            }
            parser::ExprKind::Tuple(elements) => {
                let parts: Vec<_> = elements.iter().map(|e| self.expr_to_string(e)).collect();
                format!("({})", parts.join(", "))
            }
            _ => format!("{:?}", expr.kind), // Fallback for complex expressions
        }
    }

    /// Parse a function signature string from @:overload metadata
    fn parse_overload_signature(
        &mut self,
        signature: &str,
        span: &parser::Span,
    ) -> LoweringResult<Option<MethodOverload>> {
        use crate::tast::node::MethodOverload;

        // Simple signature parsing: "param1:Type1, param2:Type2 -> ReturnType"
        // Split on "->" to separate parameters from return type
        if let Some(arrow_pos) = signature.find("->") {
            let params_part = signature[..arrow_pos].trim();
            let return_part = signature[arrow_pos + 2..].trim();

            // Parse parameter types
            let mut parameter_types = Vec::new();
            if !params_part.is_empty() {
                for param in params_part.split(',') {
                    let param = param.trim();
                    if let Some(colon_pos) = param.find(':') {
                        let type_part = param[colon_pos + 1..].trim();
                        // Convert string type name to TypeId
                        if let Ok(type_id) = self.resolve_type_by_name(type_part) {
                            parameter_types.push(type_id);
                        } else {
                            // Use Dynamic as fallback for unresolved types
                            parameter_types.push(self.context.type_table.borrow().dynamic_type());
                        }
                    }
                }
            }

            // Parse return type
            let return_type = if let Ok(type_id) = self.resolve_type_by_name(return_part) {
                type_id
            } else {
                self.context.type_table.borrow().dynamic_type()
            };

            Ok(Some(MethodOverload {
                parameter_types,
                return_type,
                source_location: self.context.create_location_from_span(*span),
            }))
        } else {
            // No return type specified, treat as function with no parameters returning Void
            Ok(Some(MethodOverload {
                parameter_types: Vec::new(),
                return_type: self.context.type_table.borrow().void_type(),
                source_location: self.context.create_location_from_span(*span),
            }))
        }
    }

    /// Resolve a type by name (helper for overload parsing)
    fn resolve_type_by_name(&mut self, type_name: &str) -> Result<TypeId, LoweringError> {
        match type_name {
            "Void" => Ok(self.context.type_table.borrow().void_type()),
            "Int" => Ok(self.context.type_table.borrow().int_type()),
            "Float" => Ok(self.context.type_table.borrow().float_type()),
            "Bool" => Ok(self.context.type_table.borrow().bool_type()),
            "String" => Ok(self.context.type_table.borrow().string_type()),
            "Dynamic" => Ok(self.context.type_table.borrow().dynamic_type()),
            _ => {
                // Try to resolve as a class/interface name
                let interned_name = self.context.intern_string(type_name);
                if let Some(symbol) = self
                    .context
                    .symbol_table
                    .lookup_symbol(self.context.current_scope, interned_name)
                {
                    Ok(symbol.type_id)
                } else {
                    Err(LoweringError::UnresolvedType {
                        type_name: type_name.to_string(),
                        location: SourceLocation::unknown(),
                    })
                }
            }
        }
    }

    /// Infer return type from function body by looking at return statements
    fn infer_return_type_from_body(&self, body: &[TypedStatement]) -> TypeId {
        // Look for return statements in the body
        for stmt in body {
            if let Some(return_type) = self.find_return_type_in_statement(stmt) {
                return return_type;
            }
        }
        // No return statements found, assume void
        self.context.type_table.borrow().void_type()
    }

    /// Find return type from a statement (recursively search nested blocks)
    fn find_return_type_in_statement(&self, stmt: &TypedStatement) -> Option<TypeId> {
        match stmt {
            TypedStatement::Return { value, .. } => {
                if let Some(expr) = value {
                    Some(expr.expr_type)
                } else {
                    Some(self.context.type_table.borrow().void_type())
                }
            }
            TypedStatement::Block { statements, .. } => {
                for s in statements {
                    if let Some(ret_type) = self.find_return_type_in_statement(s) {
                        return Some(ret_type);
                    }
                }
                None
            }
            TypedStatement::If {
                then_branch,
                else_branch,
                ..
            } => {
                // Check then branch
                if let Some(ret_type) = self.find_return_type_in_statement(then_branch.as_ref()) {
                    return Some(ret_type);
                }
                // Check else branch
                if let Some(else_stmt) = else_branch {
                    if let Some(ret_type) = self.find_return_type_in_statement(else_stmt.as_ref()) {
                        return Some(ret_type);
                    }
                }
                None
            }
            TypedStatement::While { body, .. }
            | TypedStatement::For { body, .. }
            | TypedStatement::ForIn { body, .. } => {
                self.find_return_type_in_statement(body.as_ref())
            }
            TypedStatement::Switch {
                cases,
                default_case,
                ..
            } => {
                for case in cases {
                    if let Some(ret_type) = self.find_return_type_in_statement(&case.body) {
                        return Some(ret_type);
                    }
                }
                if let Some(default) = default_case {
                    if let Some(ret_type) = self.find_return_type_in_statement(default.as_ref()) {
                        return Some(ret_type);
                    }
                }
                None
            }
            TypedStatement::Try {
                body,
                catch_clauses,
                finally_block,
                ..
            } => {
                if let Some(ret_type) = self.find_return_type_in_statement(body.as_ref()) {
                    return Some(ret_type);
                }
                for catch in catch_clauses {
                    if let Some(ret_type) = self.find_return_type_in_statement(&catch.body) {
                        return Some(ret_type);
                    }
                }
                if let Some(finally_stmt) = finally_block {
                    if let Some(ret_type) =
                        self.find_return_type_in_statement(finally_stmt.as_ref())
                    {
                        return Some(ret_type);
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Analyze if function body can throw exceptions
    fn analyze_can_throw(&self, body: &Option<Box<parser::Expr>>) -> bool {
        if let Some(body_expr) = body {
            self.expr_can_throw(body_expr)
        } else {
            false
        }
    }

    /// Check if an expression can throw
    fn expr_can_throw(&self, expr: &parser::Expr) -> bool {
        match &expr.kind {
            parser::ExprKind::Throw(_) => true,
            parser::ExprKind::Block(elements) => elements.iter().any(|elem| {
                if let parser::BlockElement::Expr(e) = elem {
                    self.expr_can_throw(e)
                } else {
                    false
                }
            }),
            parser::ExprKind::If {
                then_branch,
                else_branch,
                ..
            } => {
                self.expr_can_throw(then_branch)
                    || else_branch
                        .as_ref()
                        .map_or(false, |e| self.expr_can_throw(e))
            }
            parser::ExprKind::Switch { cases, default, .. } => {
                cases.iter().any(|c| self.expr_can_throw(&c.body))
                    || default.as_ref().map_or(false, |d| self.expr_can_throw(d))
            }
            parser::ExprKind::Try { .. } => false, // Try blocks handle exceptions
            parser::ExprKind::While { body, .. }
            | parser::ExprKind::DoWhile { body, .. }
            | parser::ExprKind::For { body, .. } => self.expr_can_throw(body),
            _ => false,
        }
    }

    /// Detect if function is async
    fn detect_async_kind(&self, _func: &parser::Function) -> AsyncKind {
        // Haxe doesn't have native async/await, but might use promises/futures
        // For now, return Sync
        AsyncKind::Sync
    }

    /// Analyze if function is pure (no side effects)
    fn analyze_is_pure(&self, body: &Option<Box<parser::Expr>>) -> bool {
        if let Some(body_expr) = body {
            self.expr_is_pure(body_expr)
        } else {
            true // No body means pure
        }
    }

    /// Check if an expression is pure
    fn expr_is_pure(&self, expr: &parser::Expr) -> bool {
        match &expr.kind {
            // Pure expressions
            parser::ExprKind::Int(_)
            | parser::ExprKind::Float(_)
            | parser::ExprKind::String(_)
            | parser::ExprKind::Bool(_)
            | parser::ExprKind::Null
            | parser::ExprKind::Ident(_) => true,

            // Assignments and mutations are impure
            parser::ExprKind::Assign { .. } => false,
            parser::ExprKind::Unary { op, .. } => !matches!(
                op,
                parser::UnaryOp::PreIncr
                    | parser::UnaryOp::PreDecr
                    | parser::UnaryOp::PostIncr
                    | parser::UnaryOp::PostDecr
            ),

            // Function calls might have side effects
            parser::ExprKind::Call { .. } | parser::ExprKind::New { .. } => false,

            // Recursively check compound expressions
            parser::ExprKind::Binary { left, right, .. } => {
                self.expr_is_pure(left) && self.expr_is_pure(right)
            }
            parser::ExprKind::Block(elements) => elements.iter().all(|elem| {
                if let parser::BlockElement::Expr(e) = elem {
                    self.expr_is_pure(e)
                } else {
                    true
                }
            }),

            _ => false, // Conservative: assume impure
        }
    }

    /// Calculate cyclomatic complexity of function body
    fn calculate_complexity(&self, body: &Option<Box<parser::Expr>>) -> u32 {
        if let Some(body_expr) = body {
            1 + self.expr_complexity(body_expr)
        } else {
            1
        }
    }

    /// Calculate expression complexity
    fn expr_complexity(&self, expr: &parser::Expr) -> u32 {
        match &expr.kind {
            parser::ExprKind::If {
                then_branch,
                else_branch,
                ..
            } => {
                1 + self.expr_complexity(then_branch)
                    + else_branch.as_ref().map_or(0, |e| self.expr_complexity(e))
            }
            parser::ExprKind::Switch { cases, default, .. } => {
                cases.len() as u32
                    + cases
                        .iter()
                        .map(|c| self.expr_complexity(&c.body))
                        .sum::<u32>()
                    + default.as_ref().map_or(0, |d| self.expr_complexity(d))
            }
            parser::ExprKind::While { body, .. }
            | parser::ExprKind::DoWhile { body, .. }
            | parser::ExprKind::For { body, .. } => 1 + self.expr_complexity(body),
            parser::ExprKind::Binary { op, left, right } => {
                let base = match op {
                    parser::BinaryOp::And | parser::BinaryOp::Or => 1,
                    _ => 0,
                };
                base + self.expr_complexity(left) + self.expr_complexity(right)
            }
            parser::ExprKind::Block(elements) => elements
                .iter()
                .map(|elem| {
                    if let parser::BlockElement::Expr(e) = elem {
                        self.expr_complexity(e)
                    } else {
                        0
                    }
                })
                .sum(),
            parser::ExprKind::Try { catches, .. } => catches.len() as u32,
            _ => 0,
        }
    }
}

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
