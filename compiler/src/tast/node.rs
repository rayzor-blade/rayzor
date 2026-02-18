//! Typed AST (TAST) node implementations
//!
//! This module contains the core data structures for the Typed AST system,
//! which adds type information, symbol resolution, and ownership tracking
//! to the syntax tree for advanced static analysis.

use crate::tast::symbols::Mutability;
use crate::tast::{
    InternedString, ScopeId, SourceLocation, StringInterner, SymbolId, TypeId, Visibility,
};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

/// Ownership and usage information for variables and expressions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VariableUsage {
    /// Transfer ownership (move semantics)
    Move,
    /// Immutable reference/borrow
    Borrow,
    /// Mutable reference/borrow
    BorrowMut,
    /// Copy for Copy types (primitives)
    Copy,
}

/// A complete typed file (compilation unit)
#[derive(Debug, Clone)]
pub struct TypedFile {
    /// Functions defined in this file
    pub functions: Vec<TypedFunction>,

    /// Classes defined in this file
    pub classes: Vec<TypedClass>,

    /// Interfaces defined in this file
    pub interfaces: Vec<TypedInterface>,

    /// Enums defined in this file
    pub enums: Vec<TypedEnum>,

    /// Type aliases defined in this file
    pub type_aliases: Vec<TypedTypeAlias>,

    /// Abstract types defined in this file
    pub abstracts: Vec<TypedAbstract>,

    /// Module-level fields (variables and functions at module level)
    pub module_fields: Vec<TypedModuleField>,

    /// Import statements
    pub imports: Vec<TypedImport>,

    /// Using statements
    pub using_statements: Vec<TypedUsing>,

    /// File-level metadata
    pub metadata: FileMetadata,

    /// Program-level safety mode (determined by main class annotation)
    /// None means runtime-managed memory (default), Some(mode) means manual memory management
    pub program_safety_mode: Option<SafetyMode>,

    /// String interner for efficient string management throughout compilation
    pub string_interner: Rc<RefCell<StringInterner>>,
}

/// Metadata about a typed file
#[derive(Debug, Clone, Default)]
pub struct FileMetadata {
    /// File path
    pub file_path: String,

    /// File name (interned)
    pub file_name: Option<InternedString>,

    /// Package/module name
    pub package_name: Option<String>,

    /// Total lines of code
    pub loc: usize,

    /// Compilation timestamp
    pub timestamp: u64,
}

impl TypedFile {
    /// Create a new TypedFile with the given string interner
    pub fn new(string_interner: Rc<RefCell<StringInterner>>) -> Self {
        Self {
            functions: Vec::new(),
            classes: Vec::new(),
            interfaces: Vec::new(),
            enums: Vec::new(),
            type_aliases: Vec::new(),
            abstracts: Vec::new(),
            module_fields: Vec::new(),
            imports: Vec::new(),
            using_statements: Vec::new(),
            metadata: FileMetadata::default(),
            string_interner,
            program_safety_mode: None, // Will be determined during analysis
        }
    }

    /// Get a shared reference to the string interner
    pub fn string_interner(&self) -> Rc<RefCell<StringInterner>> {
        Rc::clone(&self.string_interner)
    }

    /// Intern a string using the file's string interner
    pub fn intern_string(&self, s: &str) -> InternedString {
        self.string_interner.borrow_mut().intern(s)
    }

    /// Get a string from an interned string using the file's string interner
    pub fn get_string(&self, interned: InternedString) -> Option<String> {
        self.string_interner
            .borrow()
            .get(interned)
            .map(|s| s.to_string())
    }

    /// Detect and set the program-level safety mode by checking for a Main class with @:safety annotation
    /// Returns the detected safety mode (None for default/runtime-managed, Some(mode) for manual memory)
    pub fn detect_program_safety_mode(&mut self) -> Option<SafetyMode> {
        // SIMPLIFIED APPROACH: Just find the first class with @:safety annotation
        // In practice, this should only be on the Main class anyway
        let safety_mode = self.classes.iter().find_map(|c| {
            c.memory_annotations
                .iter()
                .find_map(|annotation| annotation.safety_mode())
        });

        self.program_safety_mode = safety_mode;
        safety_mode
    }

    /// Get the current program safety mode
    pub fn get_program_safety_mode(&self) -> Option<SafetyMode> {
        self.program_safety_mode
    }

    /// Check if the program uses manual memory management
    pub fn uses_manual_memory(&self) -> bool {
        self.program_safety_mode.is_some()
    }
}

/// A typed function definition
#[derive(Debug, Clone)]
pub struct TypedFunction {
    /// Symbol ID for this function
    pub symbol_id: SymbolId,

    /// Function name
    pub name: InternedString,

    /// Function parameters
    pub parameters: Vec<TypedParameter>,

    /// Return type
    pub return_type: TypeId,

    /// Function body statements
    pub body: Vec<TypedStatement>,

    /// Function visibility
    pub visibility: Visibility,

    /// Function effects (can throw, async, etc.)
    pub effects: FunctionEffects,

    /// Generic type parameters
    pub type_parameters: Vec<TypedTypeParameter>,

    /// Whether function is static
    pub is_static: bool,

    /// Source location
    pub source_location: SourceLocation,

    /// Function metadata
    pub metadata: FunctionMetadata,
}

/// Function parameter
#[derive(Debug, Clone)]
pub struct TypedParameter {
    /// Symbol ID for this parameter
    pub symbol_id: SymbolId,

    /// Parameter name
    pub name: InternedString,

    /// Parameter type
    pub param_type: TypeId,

    /// Whether parameter is optional
    pub is_optional: bool,

    /// Default value if optional
    pub default_value: Option<TypedExpression>,

    /// Parameter mutability
    pub mutability: Mutability,

    /// Source location
    pub source_location: SourceLocation,
}

/// Function effects information
#[derive(Debug, Clone, Default)]
pub struct FunctionEffects {
    /// Can throw exceptions
    pub can_throw: bool,

    /// Async nature of the function
    pub async_kind: AsyncKind,

    /// Is pure function (no side effects)
    pub is_pure: bool,

    /// Is inline function
    pub is_inline: bool,

    /// Types of exceptions that can be thrown (if any)
    pub exception_types: Vec<TypeId>,

    /// Memory effects (mutations, borrows, etc.)
    pub memory_effects: MemoryEffects,

    /// Resource effects (I/O, network, etc.)
    pub resource_effects: ResourceEffects,
}

/// Async function classification
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum AsyncKind {
    /// Synchronous function
    #[default]
    Sync,

    /// Async function (returns Promise/Future)
    Async,

    /// Generator function
    Generator,

    /// Async generator function
    AsyncGenerator,
}

/// Memory effects tracking
#[derive(Debug, Clone, Default)]
pub struct MemoryEffects {
    /// Variables that are mutated
    pub mutations: Vec<SymbolId>,

    /// Objects that are moved/consumed
    pub moves: Vec<SymbolId>,

    /// Whether the function can escape references
    pub escapes_references: bool,

    /// Global state access
    pub accesses_global_state: bool,
}

/// Resource effects tracking
#[derive(Debug, Clone, Default)]
pub struct ResourceEffects {
    /// File I/O operations
    pub performs_file_io: bool,

    /// Network I/O operations
    pub performs_network_io: bool,

    /// Database operations
    pub performs_database_ops: bool,

    /// System calls
    pub performs_system_calls: bool,

    /// Other I/O operations
    pub performs_other_io: bool,
}

/// Method overload signature information
#[derive(Debug, Clone)]
pub struct MethodOverload {
    /// Parameter types for this overload
    pub parameter_types: Vec<TypeId>,
    /// Return type for this overload
    pub return_type: TypeId,
    /// Source location of the overload metadata
    pub source_location: SourceLocation,
}

/// Safety mode for @:safety annotation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafetyMode {
    /// Strict mode - all classes must have @:safety annotation
    /// Compilation fails if any class lacks manual memory annotations
    Strict,

    /// Non-strict mode - unannotated classes are auto-wrapped in Rc
    /// Allows mixing manual memory classes with legacy code
    NonStrict,
}

impl Default for SafetyMode {
    fn default() -> Self {
        SafetyMode::NonStrict
    }
}

/// Memory safety annotations for types, functions, and variables
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryAnnotation {
    /// @:safety or @:safety(strict=true/false) - Opt-in to manual memory management
    /// When applied to main class, enables manual memory management for entire program
    /// - strict=true: All classes must have @:safety (compilation error otherwise)
    /// - strict=false (default): Unannotated classes auto-wrapped in Rc
    SafetyWithMode(SafetyMode),

    /// @:managed - Explicitly mark class as runtime-managed (this is the default)
    /// Use this to be explicit about memory management strategy
    Managed,

    /// @:move - Type uses move semantics (ownership transferred on assignment)
    /// Requires @:safety on the class
    Move,

    /// @:unique - Type must have unique ownership (no aliasing allowed)
    /// Requires @:safety on the class
    Unique,

    /// @:borrow - Parameter is borrowed (reference semantics, not owned)
    /// Used with @:safety classes
    Borrow,

    /// @:owned - Parameter takes ownership (consumes the argument)
    /// Used with @:safety classes
    Owned,

    /// @:linear - Type must be used exactly once (linear types)
    /// Requires @:safety on the class
    Linear,

    /// @:affine - Type can be used at most once (affine types)
    /// Requires @:safety on the class
    Affine,

    /// @:box - Type is heap-allocated with unique ownership (like Rust's Box)
    /// Requires @:safety on the class
    Box,

    /// @:arc - Type is heap-allocated with shared ownership via atomic reference counting
    /// Can be used with or without @:safety
    Arc,

    /// @:atomic - Type supports atomic operations (thread-safe shared mutable state)
    /// Can be used with or without @:safety
    Atomic,

    /// @:rc - Type is heap-allocated with shared ownership via reference counting (non-atomic)
    /// Can be used with or without @:safety
    Rc,
}

impl MemoryAnnotation {
    /// Parse annotation from metadata name
    /// For @:safety without parameters, defaults to non-strict mode
    pub fn from_metadata_name(name: &str) -> Option<Self> {
        match name {
            "safety" => Some(MemoryAnnotation::SafetyWithMode(SafetyMode::NonStrict)),
            "managed" => Some(MemoryAnnotation::Managed),
            "move" => Some(MemoryAnnotation::Move),
            "unique" => Some(MemoryAnnotation::Unique),
            "borrow" => Some(MemoryAnnotation::Borrow),
            "owned" => Some(MemoryAnnotation::Owned),
            "linear" => Some(MemoryAnnotation::Linear),
            "affine" => Some(MemoryAnnotation::Affine),
            "box" => Some(MemoryAnnotation::Box),
            "arc" => Some(MemoryAnnotation::Arc),
            "atomic" => Some(MemoryAnnotation::Atomic),
            "rc" => Some(MemoryAnnotation::Rc),
            _ => None,
        }
    }

    /// Parse @:safety with mode parameter (positional)
    /// e.g., @:safety(true) for strict mode, @:safety(false) for non-strict
    pub fn from_metadata_with_params(name: &str, params: &[String]) -> Option<Self> {
        if name == "safety" {
            // First positional parameter indicates strict mode
            let mode = params
                .first()
                .and_then(|value| match value.as_str() {
                    "true" => Some(SafetyMode::Strict),
                    "false" => Some(SafetyMode::NonStrict),
                    _ => None,
                })
                .unwrap_or(SafetyMode::NonStrict); // Default to non-strict if no params

            Some(MemoryAnnotation::SafetyWithMode(mode))
        } else {
            Self::from_metadata_name(name)
        }
    }

    /// Returns true if this annotation opts into manual memory management
    pub fn is_manual_memory_management(&self) -> bool {
        matches!(self, MemoryAnnotation::SafetyWithMode(_))
    }

    /// Get the safety mode if this is a Safety annotation
    pub fn safety_mode(&self) -> Option<SafetyMode> {
        match self {
            MemoryAnnotation::SafetyWithMode(mode) => Some(*mode),
            _ => None,
        }
    }

    /// Returns true if this annotation implies move semantics
    pub fn implies_move_semantics(&self) -> bool {
        matches!(
            self,
            MemoryAnnotation::Move
                | MemoryAnnotation::Unique
                | MemoryAnnotation::Box
                | MemoryAnnotation::Linear
                | MemoryAnnotation::Affine
        )
    }

    /// Returns true if this annotation implies shared ownership
    pub fn implies_shared_ownership(&self) -> bool {
        matches!(self, MemoryAnnotation::Arc | MemoryAnnotation::Rc)
    }

    /// Returns true if this annotation requires atomic operations
    pub fn requires_atomic(&self) -> bool {
        matches!(self, MemoryAnnotation::Arc | MemoryAnnotation::Atomic)
    }
}

/// Derived traits from @:derive([Clone, Copy, ...])
/// Similar to Rust's #[derive(...)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DerivedTrait {
    /// @:derive(Clone) - Type can be explicitly cloned via .clone()
    /// Automatically generates a clone() method that performs deep copy
    Clone,

    /// @:derive(Copy) - Type can be implicitly copied (bitwise copy is safe)
    /// Only valid for types with all Copy fields (primitives, etc.)
    /// Copy types don't move, they are duplicated on assignment
    Copy,

    /// @:derive(Debug) - Type can be formatted for debugging
    /// Generates toString() implementation
    Debug,

    /// @:derive(Default) - Type has a default value
    /// Generates a static default() method
    Default,

    /// @:derive(PartialEq) - Type can be compared for equality
    /// Generates == and != operators
    PartialEq,

    /// @:derive(Eq) - Type has full equivalence relation (reflexive, symmetric, transitive)
    /// Implies PartialEq
    Eq,

    /// @:derive(PartialOrd) - Type can be ordered (partial ordering)
    /// Generates <, <=, >, >= operators
    PartialOrd,

    /// @:derive(Ord) - Type has total ordering
    /// Implies PartialOrd and Eq
    Ord,

    /// @:derive(Hash) - Type can be hashed
    /// Generates hash() method for use in HashMap
    Hash,

    /// @:derive(Send) - Type can be transferred between threads
    /// Required for Thread.spawn() captures and Channel<T> element types
    /// Auto-derived if all fields are Send
    Send,

    /// @:derive(Sync) - Type can be safely shared between threads
    /// Required for Arc<T> element types
    /// Auto-derived if all fields are Sync
    Sync,
}

impl DerivedTrait {
    /// Parse trait from string (case-insensitive)
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "clone" => Some(DerivedTrait::Clone),
            "copy" => Some(DerivedTrait::Copy),
            "debug" => Some(DerivedTrait::Debug),
            "default" => Some(DerivedTrait::Default),
            "partialeq" => Some(DerivedTrait::PartialEq),
            "eq" => Some(DerivedTrait::Eq),
            "partialord" => Some(DerivedTrait::PartialOrd),
            "ord" => Some(DerivedTrait::Ord),
            "hash" => Some(DerivedTrait::Hash),
            "send" => Some(DerivedTrait::Send),
            "sync" => Some(DerivedTrait::Sync),
            _ => None,
        }
    }

    /// Get the trait name as a string
    pub fn as_str(&self) -> &'static str {
        match self {
            DerivedTrait::Clone => "Clone",
            DerivedTrait::Copy => "Copy",
            DerivedTrait::Debug => "Debug",
            DerivedTrait::Default => "Default",
            DerivedTrait::PartialEq => "PartialEq",
            DerivedTrait::Eq => "Eq",
            DerivedTrait::PartialOrd => "PartialOrd",
            DerivedTrait::Ord => "Ord",
            DerivedTrait::Hash => "Hash",
            DerivedTrait::Send => "Send",
            DerivedTrait::Sync => "Sync",
        }
    }

    /// Check if this trait requires another trait to be implemented
    pub fn requires(&self) -> Vec<DerivedTrait> {
        match self {
            DerivedTrait::Eq => vec![DerivedTrait::PartialEq],
            DerivedTrait::Ord => vec![DerivedTrait::PartialOrd, DerivedTrait::Eq],
            DerivedTrait::PartialOrd => vec![DerivedTrait::PartialEq],
            _ => vec![],
        }
    }

    /// Check if this trait is valid for types with the given properties
    pub fn is_valid_for_type(
        &self,
        has_non_copy_fields: bool,
        has_custom_implementation: bool,
    ) -> Result<(), String> {
        match self {
            DerivedTrait::Copy => {
                if has_non_copy_fields {
                    return Err("Cannot derive Copy for types with non-Copy fields".to_string());
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }
}

/// Function metadata
#[derive(Debug, Clone, Default)]
pub struct FunctionMetadata {
    /// Estimated complexity score
    pub complexity_score: u32,

    /// Number of statements in body
    pub statement_count: usize,

    /// Whether function is recursive
    pub is_recursive: bool,

    /// Call count (for optimization)
    pub call_count: u32,

    /// Whether function is marked with override modifier
    pub is_override: bool,

    /// Method overload signatures from @:overload metadata
    pub overload_signatures: Vec<MethodOverload>,

    /// Operator metadata from @:op(A + B), etc.
    /// Stored as (operator_string, params) e.g. ("A + B", [])
    pub operator_metadata: Vec<(String, Vec<String>)>,

    /// Whether this function is marked with @:arrayAccess
    pub is_array_access: bool,

    /// Whether this function is marked with @:from (abstract implicit conversion)
    pub is_from_conversion: bool,

    /// Whether this function is marked with @:to (abstract implicit conversion)
    pub is_to_conversion: bool,

    /// Memory safety annotations
    pub memory_annotations: Vec<MemoryAnnotation>,
}

/// Generic type parameter with variance support
#[derive(Debug, Clone)]
pub struct TypedTypeParameter {
    /// Symbol ID for this type parameter
    pub symbol_id: SymbolId,

    /// Parameter name (T, U, etc.)
    pub name: InternedString,

    /// Type constraints
    pub constraints: Vec<TypeId>,

    /// Variance annotation (covariant +, contravariant -, invariant)
    pub variance: TypeVariance,

    /// Source location
    pub source_location: SourceLocation,
}

/// Type variance for generic parameters
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeVariance {
    /// Invariant (no variance)
    Invariant,
    /// Covariant (+T)
    Covariant,
    /// Contravariant (-T)
    Contravariant,
}

/// A typed statement
#[derive(Debug, Clone)]
pub enum TypedStatement {
    /// Expression statement
    Expression {
        expression: TypedExpression,
        source_location: SourceLocation,
    },

    /// Variable declaration
    VarDeclaration {
        symbol_id: SymbolId,
        var_type: TypeId,
        initializer: Option<TypedExpression>,
        mutability: Mutability,
        source_location: SourceLocation,
    },

    /// Assignment statement
    Assignment {
        target: TypedExpression,
        value: TypedExpression,
        source_location: SourceLocation,
    },

    /// If statement
    If {
        condition: TypedExpression,
        then_branch: Box<TypedStatement>,
        else_branch: Option<Box<TypedStatement>>,
        source_location: SourceLocation,
    },

    /// While loop
    While {
        condition: TypedExpression,
        body: Box<TypedStatement>,
        source_location: SourceLocation,
    },

    /// For loop
    For {
        init: Option<Box<TypedStatement>>,
        condition: Option<TypedExpression>,
        update: Option<TypedExpression>,
        body: Box<TypedStatement>,
        source_location: SourceLocation,
    },

    /// For-in loop with optional key-value iteration: `for (item in iterable)` or `for (key => value in map)`
    ForIn {
        /// Variable to bind the value (or just the item for simple iteration)
        value_var: SymbolId,
        /// Optional variable to bind the key (for key-value iteration)
        key_var: Option<SymbolId>,
        /// Expression to iterate over
        iterable: TypedExpression,
        /// Loop body
        body: Box<TypedStatement>,
        source_location: SourceLocation,
    },

    /// Return statement
    Return {
        value: Option<TypedExpression>,
        source_location: SourceLocation,
    },

    /// Throw statement
    Throw {
        exception: TypedExpression,
        source_location: SourceLocation,
    },

    /// Try-catch-finally statement
    Try {
        body: Box<TypedStatement>,
        catch_clauses: Vec<TypedCatchClause>,
        finally_block: Option<Box<TypedStatement>>,
        source_location: SourceLocation,
    },

    /// Switch statement
    Switch {
        discriminant: TypedExpression,
        cases: Vec<TypedSwitchCase>,
        default_case: Option<Box<TypedStatement>>,
        source_location: SourceLocation,
    },

    /// Break statement
    Break {
        target_loop: Option<SymbolId>,
        source_location: SourceLocation,
    },

    /// Continue statement
    Continue {
        target_loop: Option<SymbolId>,
        source_location: SourceLocation,
    },

    /// Block statement
    Block {
        statements: Vec<TypedStatement>,
        scope_id: ScopeId,
        source_location: SourceLocation,
    },

    // Haxe-specific statements
    /// Pattern matching statement
    PatternMatch {
        value: TypedExpression,
        patterns: Vec<TypedPatternCase>,
        source_location: SourceLocation,
    },

    /// Macro expansion
    MacroExpansion {
        expansion_info: MacroExpansionInfo,
        expanded_statements: Vec<TypedStatement>,
        source_location: SourceLocation,
    },
}

/// Catch clause in try-catch with optional filter
#[derive(Debug, Clone)]
pub struct TypedCatchClause {
    /// Exception type to catch
    pub exception_type: TypeId,

    /// Variable to bind exception to
    pub exception_variable: SymbolId,

    /// Optional filter expression for conditional catching
    pub filter: Option<TypedExpression>,

    /// Catch block body
    pub body: TypedStatement,

    /// Source location
    pub source_location: SourceLocation,
}

/// Switch case
#[derive(Debug, Clone)]
pub struct TypedSwitchCase {
    /// Case value (constant expression)
    pub case_value: TypedExpression,

    /// Optional guard expression (`case v if v > 0:`)
    pub guard: Option<TypedExpression>,

    /// Case body
    pub body: TypedStatement,

    /// Source location
    pub source_location: SourceLocation,
}

/// Pattern case for pattern matching
#[derive(Debug, Clone)]
pub struct TypedPatternCase {
    /// Pattern to match
    pub pattern: TypedPattern,

    /// Guard condition (optional)
    pub guard: Option<TypedExpression>,

    /// Pattern body
    pub body: TypedStatement,

    /// Variables bound by this pattern
    pub bound_variables: Vec<SymbolId>,

    /// Source location
    pub source_location: SourceLocation,
}

/// Haxe pattern for pattern matching
#[derive(Debug, Clone)]
pub enum TypedPattern {
    /// Wildcard pattern (_)
    Wildcard { source_location: SourceLocation },

    /// Variable binding pattern
    Variable {
        symbol_id: SymbolId,
        pattern_type: TypeId,
        source_location: SourceLocation,
    },

    /// Literal value pattern
    Literal {
        value: TypedExpression,
        source_location: SourceLocation,
    },

    /// Constructor pattern (enum matching)
    Constructor {
        constructor: SymbolId,
        args: Vec<TypedPattern>,
        pattern_type: TypeId,
        source_location: SourceLocation,
    },

    /// Array pattern
    Array {
        elements: Vec<TypedPattern>,
        rest: Option<Box<TypedPattern>>,
        pattern_type: TypeId,
        source_location: SourceLocation,
    },

    /// Object pattern
    Object {
        fields: Vec<TypedFieldPattern>,
        pattern_type: TypeId,
        source_location: SourceLocation,
    },

    /// Guard pattern (pattern with additional condition)
    Guard {
        pattern: Box<TypedPattern>,
        guard: TypedExpression,
    },

    /// Extractor pattern: `expression => value`
    Extractor {
        /// Expression to extract from
        extractor_expr: TypedExpression,
        /// Value to extract/match
        value_expr: TypedExpression,
        /// Pattern type
        pattern_type: TypeId,
        source_location: SourceLocation,
    },
}

/// Field pattern for object matching
#[derive(Debug, Clone)]
pub struct TypedFieldPattern {
    /// Field name
    pub field_name: String,

    /// Field pattern
    pub pattern: TypedPattern,

    /// Source location
    pub source_location: SourceLocation,
}

/// Macro expansion information
#[derive(Debug, Clone)]
pub struct MacroExpansionInfo {
    /// The macro being expanded
    pub macro_symbol: SymbolId,

    /// Original source location before expansion
    pub original_location: SourceLocation,

    /// Expansion context
    pub expansion_context: String,

    /// Macro arguments
    pub macro_args: Vec<TypedExpression>,
}

/// Metadata annotation for expressions and declarations
#[derive(Debug, Clone)]
pub struct TypedMetadata {
    /// Metadata name (e.g., "native", "inline", etc.)
    pub name: InternedString,
    /// Optional parameters for the metadata
    pub params: Vec<TypedExpression>,
    /// Source location
    pub source_location: SourceLocation,
}

/// Module-level field (variable or function declared at module level)
#[derive(Debug, Clone)]
pub struct TypedModuleField {
    /// Field symbol ID
    pub symbol_id: SymbolId,
    /// Field name
    pub name: InternedString,
    /// Field kind (variable, final, or function)
    pub kind: TypedModuleFieldKind,
    /// Visibility
    pub visibility: Visibility,
    /// Source location
    pub source_location: SourceLocation,
}

/// Module-level field kind
#[derive(Debug, Clone)]
pub enum TypedModuleFieldKind {
    /// Variable field: `var x:Int = 10;`
    Var {
        field_type: TypeId,
        initializer: Option<TypedExpression>,
        mutability: Mutability,
    },
    /// Final field: `final x:Int = 10;`
    Final {
        field_type: TypeId,
        initializer: Option<TypedExpression>,
    },
    /// Function: `function foo():Void {}`
    Function(TypedFunction),
}

/// Using statement for extension methods
#[derive(Debug, Clone)]
pub struct TypedUsing {
    /// Module path being used (interned for efficiency)
    pub module_path: InternedString,
    /// Target type for using (if specified)
    pub target_type: Option<TypeId>,
    /// Source location
    pub source_location: SourceLocation,
}

/// Abstract type with full support for from/to conversions and special metadata
#[derive(Debug, Clone)]
pub struct TypedAbstract {
    /// Symbol ID
    pub symbol_id: SymbolId,
    /// Abstract name
    pub name: InternedString,
    /// Underlying type
    pub underlying_type: Option<TypeId>,
    /// Type parameters
    pub type_parameters: Vec<TypedTypeParameter>,
    /// Fields and methods
    pub fields: Vec<TypedField>,
    /// Methods
    pub methods: Vec<TypedFunction>,
    /// Constructors
    pub constructors: Vec<TypedFunction>,
    /// From conversion types
    pub from_types: Vec<TypeId>,
    /// To conversion types
    pub to_types: Vec<TypeId>,
    /// @:forward field/method names (empty = forward all)
    pub forward_fields: Vec<InternedString>,
    /// Whether this is an enum abstract
    pub is_enum_abstract: bool,
    /// Visibility
    pub visibility: Visibility,
    /// Source location
    pub source_location: SourceLocation,
}

/// A typed expression
#[derive(Debug, Clone)]
pub struct TypedExpression {
    /// Expression type
    pub expr_type: TypeId,

    /// Expression kind
    pub kind: TypedExpressionKind,

    /// Variable usage information
    pub usage: VariableUsage,

    /// Lifetime information
    pub lifetime_id: crate::tast::LifetimeId,

    /// Source location
    pub source_location: SourceLocation,

    /// Expression metadata
    pub metadata: ExpressionMetadata,
}

/// Expression metadata
#[derive(Debug, Clone, Default)]
pub struct ExpressionMetadata {
    /// Whether this expression is a compile-time constant
    pub is_constant: bool,

    /// Whether this expression has side effects
    pub has_side_effects: bool,

    /// Whether this expression can throw
    pub can_throw: bool,

    /// Estimated complexity score
    pub complexity_score: u32,
}

/// Typed expression kinds
#[derive(Debug, Clone)]
pub enum TypedExpressionKind {
    /// Literal values
    Literal {
        value: LiteralValue,
    },

    /// Variable reference
    Variable {
        symbol_id: SymbolId,
    },

    /// Field access: obj.field or obj?.field
    FieldAccess {
        object: Box<TypedExpression>,
        field_symbol: SymbolId,
        /// True for optional chaining: obj?.field
        is_optional: bool,
    },

    /// Static field access: Class.field
    StaticFieldAccess {
        class_symbol: SymbolId,
        field_symbol: SymbolId,
    },

    /// Array access: arr[index]
    ArrayAccess {
        array: Box<TypedExpression>,
        index: Box<TypedExpression>,
    },

    /// Function call
    FunctionCall {
        function: Box<TypedExpression>,
        arguments: Vec<TypedExpression>,
        type_arguments: Vec<TypeId>,
    },

    /// Method call: obj.method() or obj?.method()
    MethodCall {
        receiver: Box<TypedExpression>,
        method_symbol: SymbolId,
        arguments: Vec<TypedExpression>,
        type_arguments: Vec<TypeId>,
        /// True for optional chaining: obj?.method()
        is_optional: bool,
    },

    /// Static method call
    StaticMethodCall {
        class_symbol: SymbolId,
        method_symbol: SymbolId,
        arguments: Vec<TypedExpression>,
        type_arguments: Vec<TypeId>,
    },

    /// Binary operation
    BinaryOp {
        left: Box<TypedExpression>,
        operator: BinaryOperator,
        right: Box<TypedExpression>,
    },

    /// Unary operation
    UnaryOp {
        operator: UnaryOperator,
        operand: Box<TypedExpression>,
    },

    /// Conditional expression: condition ? then : else
    Conditional {
        condition: Box<TypedExpression>,
        then_expr: Box<TypedExpression>,
        else_expr: Option<Box<TypedExpression>>,
    },

    While {
        condition: Box<TypedExpression>,
        then_expr: Box<TypedExpression>,
    },

    For {
        variable: SymbolId,
        iterable: Box<TypedExpression>,
        body: Box<TypedExpression>,
    },

    /// For-in expression with optional key-value iteration
    ForIn {
        /// Variable to bind the value
        value_var: SymbolId,
        /// Optional variable to bind the key
        key_var: Option<SymbolId>,
        /// Expression to iterate over
        iterable: Box<TypedExpression>,
        /// Expression body
        body: Box<TypedExpression>,
    },

    /// Array literal: [1, 2, 3]
    ArrayLiteral {
        elements: Vec<TypedExpression>,
    },

    /// Map literal: ["key1" => value1, "key2" => value2]
    MapLiteral {
        entries: Vec<TypedMapEntry>,
    },

    /// Object literal: { field1: value1, field2: value2 }
    ObjectLiteral {
        fields: Vec<TypedObjectField>,
    },

    /// Function literal/lambda
    FunctionLiteral {
        parameters: Vec<TypedParameter>,
        body: Vec<TypedStatement>,
        return_type: TypeId,
    },

    /// Type cast: (Type) expression
    Cast {
        expression: Box<TypedExpression>,
        target_type: TypeId,
        cast_kind: CastKind,
    },

    /// Instance creation: new Class()
    New {
        class_type: TypeId,
        arguments: Vec<TypedExpression>,
        type_arguments: Vec<TypeId>,
        /// Original class name from source code (preserved for extern stdlib classes where TypeId may be invalid)
        class_name: Option<InternedString>,
    },

    /// This reference
    This {
        this_type: TypeId,
    },

    /// Super reference
    Super {
        super_type: TypeId,
    },

    Is {
        expression: Box<TypedExpression>,
        check_type: TypeId,
    },

    /// Null literal
    Null,

    Return {
        value: Option<Box<TypedExpression>>,
    },

    Throw {
        expression: Box<TypedExpression>,
    },

    Break,
    Continue,

    /// Variable declaration as expression: `var x = 5` (returns 5)
    VarDeclarationExpr {
        symbol_id: SymbolId,
        var_type: TypeId,
        initializer: Box<TypedExpression>,
    },

    /// Final declaration as expression: `final x = 5` (returns 5)
    FinalDeclarationExpr {
        symbol_id: SymbolId,
        var_type: TypeId,
        initializer: Box<TypedExpression>,
    },

    // Haxe-specific expressions
    /// String interpolation
    StringInterpolation {
        parts: Vec<StringInterpolationPart>,
    },

    /// Macro expression
    MacroExpression {
        macro_symbol: SymbolId,
        arguments: Vec<TypedExpression>,
    },

    Block {
        statements: Vec<TypedStatement>,
        scope_id: ScopeId,
    },

    /// Metadata annotation on expression: `@:meta expr`
    Meta {
        metadata: Vec<TypedMetadata>,
        expression: Box<TypedExpression>,
    },

    /// Dollar identifier: `$type`, `$v{...}`, `$i{...}`, etc.
    DollarIdent {
        name: InternedString,
        arg: Option<Box<TypedExpression>>,
    },

    /// Compiler-specific code expression: `__c__("code {0}", arg0)`
    CompilerSpecific {
        target: InternedString,
        code: Box<TypedExpression>,
        args: Vec<TypedExpression>,
    },

    /// Switch expression: `switch (expr) { case pattern: body; default: defaultBody; }`
    Switch {
        discriminant: Box<TypedExpression>,
        cases: Vec<TypedSwitchCase>,
        default_case: Option<Box<TypedExpression>>,
    },

    /// Try-catch-finally expression: `try expr catch (e:Type) handler finally expr`
    Try {
        try_expr: Box<TypedExpression>,
        catch_clauses: Vec<TypedCatchClause>,
        finally_block: Option<Box<TypedExpression>>,
    },

    /// Pattern placeholder for complex patterns that need later compilation
    PatternPlaceholder {
        pattern: parser::Pattern,
        source_location: SourceLocation,
        /// Variable bindings created by bind_pattern_variables in ast_lowering.
        /// Maps pattern variable names to their SymbolIds so TAST→HIR can use
        /// the same SymbolIds as the case body (avoiding SymbolId mismatch).
        variable_bindings: Vec<(InternedString, SymbolId)>,
    },

    /// Array comprehension: [for (i in 0...10) i * 2]
    ArrayComprehension {
        for_parts: Vec<TypedComprehensionFor>,
        expression: Box<TypedExpression>,
        element_type: TypeId,
    },

    /// Map comprehension: [for (i in 0...10) i => i * 2]
    MapComprehension {
        for_parts: Vec<TypedComprehensionFor>,
        key_expr: Box<TypedExpression>,
        value_expr: Box<TypedExpression>,
        key_type: TypeId,
        value_type: TypeId,
    },

    /// Await expression: `await someAsyncFunction()`
    Await {
        expression: Box<TypedExpression>,
        await_type: TypeId,
    },
}

/// Literal values
#[derive(Debug, Clone)]
pub enum LiteralValue {
    /// Boolean literal
    Bool(bool),

    /// Integer literal
    Int(i64),

    /// Floating point literal
    Float(f64),

    /// String literal
    String(String),

    /// Character literal
    Char(char),

    /// Regular expression literal
    Regex(String),

    /// Regex literal with flags
    RegexWithFlags { pattern: String, flags: String },
}

/// Object field in object literal
#[derive(Debug, Clone)]
pub struct TypedObjectField {
    /// Field name
    pub name: InternedString,

    /// Field value
    pub value: TypedExpression,

    /// Source location
    pub source_location: SourceLocation,
}

/// Map entry in map literal
#[derive(Debug, Clone)]
pub struct TypedMapEntry {
    /// Key expression
    pub key: TypedExpression,

    /// Value expression
    pub value: TypedExpression,

    /// Source location
    pub source_location: SourceLocation,
}

/// Comprehension for clause (for array/map comprehensions)
#[derive(Debug, Clone)]
pub struct TypedComprehensionFor {
    /// Loop variable binding
    pub var_symbol: SymbolId,

    /// Optional key variable (for key => value iteration)
    pub key_var_symbol: Option<SymbolId>,

    /// Iterator expression (what we're iterating over)
    pub iterator: TypedExpression,

    /// Type of the loop variable
    pub var_type: TypeId,

    /// Type of the key variable (if present)
    pub key_type: Option<TypeId>,

    /// Scope for the comprehension variables
    pub scope_id: ScopeId,

    /// Source location
    pub source_location: SourceLocation,
}

/// Binary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOperator {
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Mod,

    // Comparison
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,

    // Logical
    And,
    Or,

    // Bitwise
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,

    // Assignment
    Assign,
    AddAssign,
    SubAssign,
    ModAssign,
    MulAssign,
    DivAssign,

    // Range
    /// Range operator: 0...10 (creates IntIterator)
    Range,

    /// Null coalescing: a ?? b (returns a if non-null, else b)
    NullCoal,

    /// Arrow operator: key => value (for map comprehensions)
    Arrow,

    /// Unsigned shift right: a >>> b
    Ushr,
}

/// Unary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOperator {
    /// Negation: -x
    Neg,

    /// Logical not: !x
    Not,

    /// Bitwise not: ~x
    BitNot,

    /// Pre-increment: ++x
    PreInc,

    /// Post-increment: x++
    PostInc,

    /// Pre-decrement: --x
    PreDec,

    /// Post-decrement: x--
    PostDec,
}

/// Type cast kinds
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CastKind {
    /// Safe implicit cast
    Implicit,

    /// Explicit cast that may fail
    Explicit,

    /// Unsafe cast (compiler trusts programmer)
    Unsafe,

    /// Runtime type check cast
    Checked,
}

/// Parts of string interpolation
#[derive(Debug, Clone)]
pub enum StringInterpolationPart {
    /// Static string part
    String(String),

    /// Expression to interpolate
    Expression(TypedExpression),
}

/// Class definition
#[derive(Debug, Clone)]
pub struct TypedClass {
    /// Symbol ID
    pub symbol_id: SymbolId,

    /// Class name
    pub name: InternedString,

    /// Super class
    pub super_class: Option<TypeId>,

    /// Implemented interfaces
    pub interfaces: Vec<TypeId>,

    /// Fields
    pub fields: Vec<TypedField>,

    /// Methods
    pub methods: Vec<TypedFunction>,

    /// Constructors
    pub constructors: Vec<TypedFunction>,

    /// Generic type parameters
    pub type_parameters: Vec<TypedTypeParameter>,

    /// Visibility
    pub visibility: Visibility,

    /// Source location
    pub source_location: SourceLocation,

    /// Memory safety annotations (@:move, @:unique, @:box, @:arc, etc.)
    pub memory_annotations: Vec<MemoryAnnotation>,

    /// Derived traits from @:derive([Clone, Copy, ...])
    pub derived_traits: Vec<DerivedTrait>,
}

impl TypedClass {
    /// Check if this class has @:safety annotation (opts into manual memory management)
    pub fn has_safety_annotation(&self) -> bool {
        self.memory_annotations
            .iter()
            .any(|a| a.is_manual_memory_management())
    }

    /// Check if this class is explicitly marked as @:managed (runtime-managed memory)
    pub fn is_managed(&self) -> bool {
        self.memory_annotations.contains(&MemoryAnnotation::Managed)
    }

    /// Check if this class uses manual memory management
    /// Returns true if @:safety is present, false if @:managed or no annotation (default is runtime-managed)
    pub fn uses_manual_memory(&self) -> bool {
        self.has_safety_annotation()
    }

    /// Check if class derives a specific trait
    /// Example: class.derives(DerivedTrait::Clone)
    pub fn derives(&self, trait_: DerivedTrait) -> bool {
        self.derived_traits.contains(&trait_)
    }

    /// Check if class is Copy (can be implicitly copied)
    pub fn is_copy(&self) -> bool {
        self.derives(DerivedTrait::Copy)
    }

    /// Check if class is Clone (can be explicitly cloned via .clone())
    pub fn is_clone(&self) -> bool {
        self.derives(DerivedTrait::Clone)
    }

    /// Get all derived traits
    pub fn get_derived_traits(&self) -> &[DerivedTrait] {
        &self.derived_traits
    }

    /// Check if all required traits are derived
    /// For example, Ord requires PartialOrd and Eq
    pub fn has_required_traits_for(&self, trait_: DerivedTrait) -> bool {
        trait_.requires().iter().all(|req| self.derives(*req))
    }
}

/// Interface definition
#[derive(Debug, Clone)]
pub struct TypedInterface {
    /// Symbol ID
    pub symbol_id: SymbolId,

    /// Interface name
    pub name: InternedString,

    /// Extended interfaces
    pub extends: Vec<TypeId>,

    /// Method signatures
    pub methods: Vec<TypedMethodSignature>,

    /// Generic type parameters
    pub type_parameters: Vec<TypedTypeParameter>,

    /// Visibility
    pub visibility: Visibility,

    /// Source location
    pub source_location: SourceLocation,
}

/// Method signature (for interfaces)
#[derive(Debug, Clone)]
pub struct TypedMethodSignature {
    /// Method name
    pub name: InternedString,

    /// Parameters
    pub parameters: Vec<TypedParameter>,

    /// Return type
    pub return_type: TypeId,

    /// Effects
    pub effects: FunctionEffects,

    /// Source location
    pub source_location: SourceLocation,
}

/// Enum definition
#[derive(Debug, Clone)]
pub struct TypedEnum {
    /// Symbol ID
    pub symbol_id: SymbolId,

    /// Enum name
    pub name: InternedString,

    /// Enum variants
    pub variants: Vec<TypedEnumVariant>,

    /// Generic type parameters
    pub type_parameters: Vec<TypedTypeParameter>,

    /// Visibility
    pub visibility: Visibility,

    /// Source location
    pub source_location: SourceLocation,
}

/// Enum variant
#[derive(Debug, Clone)]
pub struct TypedEnumVariant {
    /// Variant name
    pub name: InternedString,

    /// Variant parameters (for complex enums)
    pub parameters: Vec<TypedParameter>,

    /// Source location
    pub source_location: SourceLocation,
}

/// Property accessor information for Haxe properties
///
/// Haxe properties can have custom getter/setter methods:
/// ```haxe
/// var x(get, set):Int;
/// function get_x():Int { return _x; }
/// function set_x(v:Int):Int { _x = v; return v; }
/// ```
#[derive(Debug, Clone)]
pub struct PropertyAccessInfo {
    /// Getter accessor
    pub getter: PropertyAccessor,
    /// Setter accessor
    pub setter: PropertyAccessor,
}

/// Property accessor mode
#[derive(Debug, Clone)]
pub enum PropertyAccessor {
    /// Default - direct field access
    Default,
    /// Null - no access allowed
    Null,
    /// Never - never allow access
    Never,
    /// Dynamic - dynamic access (not statically checked)
    Dynamic,
    /// Method - call specific getter/setter method by name
    /// The name is stored as InternedString and resolved to SymbolId during MIR lowering
    Method(InternedString),
}

/// Field definition
#[derive(Debug, Clone)]
pub struct TypedField {
    /// Symbol ID
    pub symbol_id: SymbolId,

    /// Field name
    pub name: InternedString,

    /// Field type
    pub field_type: TypeId,

    /// Initial value
    pub initializer: Option<TypedExpression>,

    /// Mutability
    pub mutability: Mutability,

    /// Visibility
    pub visibility: Visibility,

    /// Whether field is static
    pub is_static: bool,

    /// Property accessor info (Some for properties, None for regular fields)
    pub property_access: Option<PropertyAccessInfo>,

    /// Source location
    pub source_location: SourceLocation,
}

/// Type alias definition
#[derive(Debug, Clone)]
pub struct TypedTypeAlias {
    /// Symbol ID
    pub symbol_id: SymbolId,

    /// Alias name
    pub name: InternedString,

    /// Target type
    pub target_type: TypeId,

    /// Generic type parameters
    pub type_parameters: Vec<TypedTypeParameter>,

    /// Visibility
    pub visibility: Visibility,

    /// Source location
    pub source_location: SourceLocation,
}

/// Import statement
#[derive(Debug, Clone)]
pub struct TypedImport {
    /// Imported module path (interned for efficiency)
    pub module_path: InternedString,

    /// Imported symbols (None = import all, interned for efficiency)
    pub imported_symbols: Option<Vec<InternedString>>,

    /// Alias for import (interned for efficiency)
    pub alias: Option<InternedString>,

    /// Source location
    pub source_location: SourceLocation,
}

// Helper trait to get source location from any TAST node
pub trait HasSourceLocation {
    fn source_location(&self) -> SourceLocation;
}

impl HasSourceLocation for TypedStatement {
    fn source_location(&self) -> SourceLocation {
        match self {
            TypedStatement::Expression {
                source_location, ..
            } => *source_location,
            TypedStatement::VarDeclaration {
                source_location, ..
            } => *source_location,
            TypedStatement::Assignment {
                source_location, ..
            } => *source_location,
            TypedStatement::If {
                source_location, ..
            } => *source_location,
            TypedStatement::While {
                source_location, ..
            } => *source_location,
            TypedStatement::For {
                source_location, ..
            } => *source_location,
            TypedStatement::Return {
                source_location, ..
            } => *source_location,
            TypedStatement::Throw {
                source_location, ..
            } => *source_location,
            TypedStatement::Try {
                source_location, ..
            } => *source_location,
            TypedStatement::Switch {
                source_location, ..
            } => *source_location,
            TypedStatement::Break {
                source_location, ..
            } => *source_location,
            TypedStatement::Continue {
                source_location, ..
            } => *source_location,
            TypedStatement::Block {
                source_location, ..
            } => *source_location,
            TypedStatement::PatternMatch {
                source_location, ..
            } => *source_location,
            TypedStatement::MacroExpansion {
                source_location, ..
            } => *source_location,
            TypedStatement::ForIn {
                source_location, ..
            } => *source_location,
        }
    }
}

impl HasSourceLocation for TypedExpression {
    fn source_location(&self) -> SourceLocation {
        self.source_location
    }
}

#[cfg(test)]
mod tests {
    use crate::tast::ExpressionId;

    use super::*;

    #[test]
    fn test_expression_id() {
        let id = ExpressionId::from_raw(42);
        assert_eq!(id.as_raw(), 42);
        assert!(id.is_valid());

        let invalid = ExpressionId::invalid();
        assert!(!invalid.is_valid());
    }

    #[test]
    fn test_variable_usage() {
        let usage = VariableUsage::Move;
        assert_eq!(usage, VariableUsage::Move);

        let borrow = VariableUsage::Borrow;
        assert_ne!(usage, borrow);
    }

    #[test]
    fn test_typed_expression_creation() {
        let expr = TypedExpression {
            expr_type: TypeId::from_raw(1),
            kind: TypedExpressionKind::Literal {
                value: LiteralValue::Int(42),
            },
            usage: VariableUsage::Copy,
            lifetime_id: crate::tast::LifetimeId::first(),
            source_location: SourceLocation::unknown(),
            metadata: ExpressionMetadata::default(),
        };

        assert!(matches!(expr.kind, TypedExpressionKind::Literal { .. }));
        assert_eq!(expr.usage, VariableUsage::Copy);
    }

    #[test]
    fn test_function_effects() {
        let effects = FunctionEffects {
            can_throw: true,
            async_kind: AsyncKind::Async,
            is_pure: false,
            is_inline: true,
            exception_types: vec![],
            memory_effects: MemoryEffects::default(),
            resource_effects: ResourceEffects::default(),
        };

        assert!(effects.can_throw);
        assert!(!matches!(effects.async_kind, AsyncKind::Sync));
        assert!(effects.is_inline);
    }
}
