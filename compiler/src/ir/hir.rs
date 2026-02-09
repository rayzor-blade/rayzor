//! High-Level Intermediate Representation (HIR)
//!
//! HIR is the first IR level, close to Haxe source syntax but with:
//! - Fully resolved types
//! - Desugared syntax (e.g., for-in loops to iterators)
//! - Ownership and lifetime information attached
//! - Metadata preserved for optimization hints
//!
//! This matches the architecture plan where HIR preserves high-level
//! language features before lowering to MIR (SSA form).

use crate::tast::{InternedString, LifetimeId, ScopeId, SourceLocation, SymbolId, TypeId};
use indexmap::IndexMap;
use std::collections::HashMap;

/// HIR Module - top-level container
#[derive(Debug, Clone)]
pub struct HirModule {
    pub name: String,
    pub imports: Vec<HirImport>,
    pub types: IndexMap<TypeId, HirTypeDecl>, // IndexMap for deterministic ordering
    pub functions: IndexMap<SymbolId, HirFunction>, // IndexMap for deterministic ordering
    pub globals: HashMap<SymbolId, HirGlobal>,
    pub metadata: HirMetadata,
}

/// Import declaration with resolved symbols
#[derive(Debug, Clone)]
pub struct HirImport {
    pub module_path: Vec<String>,
    pub imported_symbols: Vec<SymbolId>,
    pub alias: Option<String>,
    pub is_static_extension: bool, // for 'using' imports
}

/// Type declaration in HIR
#[derive(Debug, Clone)]
pub enum HirTypeDecl {
    Class(HirClass),
    Interface(HirInterface),
    Enum(HirEnum),
    Abstract(HirAbstract),
    TypeAlias(HirTypeAlias),
}

/// Class declaration with resolved members
#[derive(Debug, Clone)]
pub struct HirClass {
    pub symbol_id: SymbolId,
    pub name: InternedString,
    pub type_params: Vec<HirTypeParam>,
    pub extends: Option<TypeId>,
    pub implements: Vec<TypeId>,
    pub fields: Vec<HirClassField>,
    pub methods: Vec<HirMethod>,
    pub constructor: Option<HirConstructor>,
    pub metadata: Vec<HirAttribute>,
    pub is_final: bool,
    pub is_abstract: bool,
    pub is_extern: bool,
}

/// Interface declaration
#[derive(Debug, Clone)]
pub struct HirInterface {
    pub symbol_id: SymbolId,
    pub name: InternedString,
    pub type_params: Vec<HirTypeParam>,
    pub extends: Vec<TypeId>,
    pub fields: Vec<HirInterfaceField>,
    pub methods: Vec<HirInterfaceMethod>,
    pub metadata: Vec<HirAttribute>,
}

/// Enum (algebraic data type) declaration
#[derive(Debug, Clone)]
pub struct HirEnum {
    pub symbol_id: SymbolId,
    pub name: InternedString,
    pub type_params: Vec<HirTypeParam>,
    pub variants: Vec<HirEnumVariant>,
    pub metadata: Vec<HirAttribute>,
}

/// Enum variant (constructor)
#[derive(Debug, Clone)]
pub struct HirEnumVariant {
    pub name: InternedString,
    pub fields: Vec<HirEnumField>,
    pub discriminant: Option<i32>,
}

/// Abstract type declaration
#[derive(Debug, Clone)]
pub struct HirAbstract {
    pub symbol_id: SymbolId,
    pub name: InternedString,
    pub type_params: Vec<HirTypeParam>,
    pub underlying: TypeId,
    pub from_rules: Vec<HirCastRule>,
    pub to_rules: Vec<HirCastRule>,
    pub operators: Vec<HirOperatorOverload>,
    pub fields: Vec<HirAbstractField>,
    pub metadata: Vec<HirAttribute>,
}

/// Cast rule for abstract types
#[derive(Debug, Clone)]
pub struct HirCastRule {
    pub from_type: TypeId,
    pub to_type: TypeId,
    pub is_implicit: bool,
    pub cast_function: Option<SymbolId>,
}

/// Operator overload for abstract types
#[derive(Debug, Clone)]
pub struct HirOperatorOverload {
    pub operator: HirBinaryOp,
    pub implementation: SymbolId,
}

/// Type alias declaration
#[derive(Debug, Clone)]
pub struct HirTypeAlias {
    pub symbol_id: SymbolId,
    pub name: InternedString,
    pub type_params: Vec<HirTypeParam>,
    pub aliased_type: TypeId,
}

/// Function declaration
#[derive(Debug, Clone)]
pub struct HirFunction {
    pub symbol_id: SymbolId,
    pub name: InternedString,
    /// Fully qualified name (e.g., "com.example.MyClass.myMethod")
    pub qualified_name: Option<InternedString>,
    pub type_params: Vec<HirTypeParam>,
    pub params: Vec<HirParam>,
    pub return_type: TypeId,
    pub body: Option<HirBlock>,
    pub metadata: Vec<HirAttribute>,
    pub is_inline: bool,
    pub is_macro: bool,
    pub is_extern: bool,
    pub calling_convention: HirCallingConvention,
    /// Flag to indicate if this is the main entry point
    pub is_main: bool,
}

impl HirFunction {
    /// Check if this function is an entry point (main or extern)
    pub fn is_entry_point(&self) -> bool {
        self.is_extern || self.is_main
    }
}

/// Constructor
#[derive(Debug, Clone)]
pub struct HirConstructor {
    pub params: Vec<HirParam>,
    pub super_call: Option<HirSuperCall>,
    pub field_inits: Vec<HirFieldInit>,
    pub body: HirBlock,
}

/// HIR Statement
#[derive(Debug, Clone)]
pub enum HirStatement {
    /// Variable declaration with pattern
    Let {
        pattern: HirPattern,
        type_hint: Option<TypeId>,
        init: Option<HirExpr>,
        is_mutable: bool,
    },

    /// Expression statement
    Expr(HirExpr),

    /// Assignment
    Assign {
        lhs: HirLValue,
        rhs: HirExpr,
        op: Option<HirBinaryOp>, // For compound assignments
    },

    /// Return statement
    Return(Option<HirExpr>),

    /// Break statement with optional target loop symbol
    Break(Option<SymbolId>), // Optional loop symbol

    /// Continue statement with optional target loop symbol
    Continue(Option<SymbolId>), // Optional loop symbol

    /// Throw statement
    Throw(HirExpr),

    /// If statement
    If {
        condition: HirExpr,
        then_branch: HirBlock,
        else_branch: Option<HirBlock>,
    },

    /// Switch statement with pattern matching
    Switch {
        scrutinee: HirExpr,
        cases: Vec<HirMatchCase>,
    },

    /// While loop
    While {
        label: Option<SymbolId>,
        condition: HirExpr,
        body: HirBlock,
        /// Optional update block executed on continue and at end of body.
        /// Used for C-style for loops and range iteration to ensure the
        /// loop counter increment runs even on `continue`.
        continue_update: Option<HirBlock>,
    },

    /// Do-while loop
    DoWhile {
        label: Option<SymbolId>,
        body: HirBlock,
        condition: HirExpr,
    },

    /// For-in loop (desugared to iterator)
    ForIn {
        label: Option<SymbolId>,
        pattern: HirPattern,
        iterator: HirExpr,
        body: HirBlock,
    },

    /// Try-catch statement
    TryCatch {
        try_block: HirBlock,
        catches: Vec<HirCatchClause>,
        finally_block: Option<HirBlock>,
    },

    /// Labeled block
    Label { symbol: SymbolId, block: HirBlock },
}

/// HIR Expression
#[derive(Debug, Clone)]
pub struct HirExpr {
    pub kind: HirExprKind,
    pub ty: TypeId,
    pub lifetime: LifetimeId,
    pub source_location: SourceLocation,
}

/// HIR Expression kinds
#[derive(Debug, Clone)]
pub enum HirExprKind {
    // === Literals ===
    Literal(HirLiteral),

    // === Variables ===
    Variable {
        symbol: SymbolId,
        capture_mode: Option<HirCaptureMode>,
    },

    // === Member access ===
    Field {
        object: Box<HirExpr>,
        field: SymbolId,
    },

    // === Array/Map access ===
    Index {
        object: Box<HirExpr>,
        index: Box<HirExpr>,
    },

    // === Function call ===
    Call {
        callee: Box<HirExpr>,
        type_args: Vec<TypeId>,
        args: Vec<HirExpr>,
        is_method: bool,
    },

    // === Constructor call ===
    New {
        class_type: TypeId,
        type_args: Vec<TypeId>,
        args: Vec<HirExpr>,
        /// Optional class name for cases where TypeId is invalid (e.g., extern stdlib classes)
        /// This preserves the class name for proper constructor resolution during MIR lowering
        class_name: Option<InternedString>,
    },

    // === Operators ===
    Unary {
        op: HirUnaryOp,
        operand: Box<HirExpr>,
    },

    Binary {
        op: HirBinaryOp,
        lhs: Box<HirExpr>,
        rhs: Box<HirExpr>,
    },

    // === Type operations ===
    Cast {
        expr: Box<HirExpr>,
        target: TypeId,
        is_safe: bool,
    },

    TypeCheck {
        expr: Box<HirExpr>,
        expected: TypeId,
    },

    // === Control flow ===
    If {
        condition: Box<HirExpr>,
        then_expr: Box<HirExpr>,
        else_expr: Box<HirExpr>,
    },

    Block(HirBlock),

    // === Closures ===
    Lambda {
        params: Vec<HirParam>,
        body: Box<HirExpr>,
        captures: Vec<HirCapture>,
    },

    // === Arrays ===
    Array {
        elements: Vec<HirExpr>,
    },

    // === Maps/Objects ===
    Map {
        entries: Vec<(HirExpr, HirExpr)>,
    },

    ObjectLiteral {
        fields: Vec<(InternedString, HirExpr)>,
    },

    // === Comprehensions ===
    ArrayComprehension {
        element: Box<HirExpr>,
        iterators: Vec<HirComprehensionIterator>,
        filter: Option<Box<HirExpr>>,
    },

    MapComprehension {
        key: Box<HirExpr>,
        value: Box<HirExpr>,
        iterators: Vec<HirComprehensionIterator>,
        filter: Option<Box<HirExpr>>,
    },

    // === String interpolation ===
    StringInterpolation {
        parts: Vec<HirStringPart>,
    },

    // === Macro expressions ===
    MacroExpansion {
        macro_name: SymbolId,
        args: Vec<HirExpr>,
    },

    Reification {
        expr: Box<HirExpr>,
    },

    // === Special forms ===
    This,
    Super,
    Null,

    // === Unsafe/Untyped ===
    Untyped(Box<HirExpr>),

    // === Inline code ===
    InlineCode {
        target: String,     // c, js, cpp, etc.
        code: Box<HirExpr>, // code expression (string literal or concat)
        args: Vec<HirExpr>, // positional arguments for {0}, {1}, etc.
    },

    // === Exception handling ===
    TryCatch {
        try_expr: Box<HirExpr>,
        catch_handlers: Vec<HirCatchHandler>,
        finally_expr: Option<Box<HirExpr>>,
    },
}

/// Catch handler for try-catch expressions
#[derive(Debug, Clone)]
pub struct HirCatchHandler {
    pub exception_var: SymbolId,
    pub exception_type: TypeId,
    pub guard: Option<Box<HirExpr>>,
    pub body: Box<HirExpr>,
}

/// Pattern for destructuring and matching
#[derive(Debug, Clone)]
pub enum HirPattern {
    /// Variable binding: `x`
    Variable {
        name: InternedString,
        symbol: SymbolId,
    },

    /// Wildcard: `_`
    Wildcard,

    /// Literal pattern: `42`, `"hello"`
    Literal(HirLiteral),

    /// Constructor pattern: `Some(x)`, `Point(x, y)`
    Constructor {
        enum_type: TypeId,
        variant: InternedString,
        fields: Vec<HirPattern>,
    },

    /// Tuple pattern: `(x, y, z)`
    Tuple(Vec<HirPattern>),

    /// Array pattern: `[head, ...tail]`
    Array {
        elements: Vec<HirPattern>,
        rest: Option<Box<HirPattern>>,
    },

    /// Object pattern: `{x: px, y: py}`
    Object {
        fields: Vec<(InternedString, HirPattern)>,
        rest: bool, // allows additional fields
    },

    /// Type annotation: `(x: String)`
    Typed {
        pattern: Box<HirPattern>,
        ty: TypeId,
    },

    /// Or pattern: `1 | 2 | 3`
    Or(Vec<HirPattern>),

    /// Guard pattern: `x if x > 0`
    Guard {
        pattern: Box<HirPattern>,
        condition: HirExpr,
    },
}

/// Match case for switch statements
#[derive(Debug, Clone)]
pub struct HirMatchCase {
    pub patterns: Vec<HirPattern>,
    pub guard: Option<HirExpr>,
    pub body: HirBlock,
}

/// Catch clause
#[derive(Debug, Clone)]
pub struct HirCatchClause {
    pub exception_type: TypeId,
    pub exception_var: SymbolId,
    pub body: HirBlock,
}

/// L-value for assignments
#[derive(Debug, Clone)]
pub enum HirLValue {
    Variable(SymbolId),
    Field {
        object: Box<HirExpr>,
        field: SymbolId,
    },
    Index {
        object: Box<HirExpr>,
        index: Box<HirExpr>,
    },
}

/// Literals
#[derive(Debug, Clone)]
pub enum HirLiteral {
    Int(i64),
    Float(f64),
    String(InternedString),
    Bool(bool),
    Regex {
        pattern: InternedString,
        flags: InternedString,
    },
}

/// Unary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HirUnaryOp {
    Not,      // !
    Neg,      // -
    BitNot,   // ~
    PreIncr,  // ++x
    PreDecr,  // --x
    PostIncr, // x++
    PostDecr, // x--
}

/// Binary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HirBinaryOp {
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

    // Range
    Range,     // ...
    RangeExcl, // ..

    // Null coalescing
    NullCoalesce, // ??
}

/// Block of statements
#[derive(Debug, Clone)]
pub struct HirBlock {
    pub statements: Vec<HirStatement>,
    pub expr: Option<Box<HirExpr>>, // Optional trailing expression
    pub scope: ScopeId,
}

/// Metadata/Attributes
#[derive(Debug, Clone)]
pub struct HirAttribute {
    pub name: InternedString,
    pub args: Vec<HirAttributeArg>,
}

#[derive(Debug, Clone)]
pub enum HirAttributeArg {
    Literal(HirLiteral),
    Named(InternedString, HirLiteral),
}

/// Additional HIR types for completeness

#[derive(Debug, Clone)]
pub struct HirTypeParam {
    pub name: InternedString,
    pub bounds: Vec<TypeId>,
    pub default: Option<TypeId>,
}

#[derive(Debug, Clone)]
pub struct HirParam {
    pub symbol_id: SymbolId, // Symbol ID from TAST (needed for variable lookup in MIR)
    pub name: InternedString,
    pub ty: TypeId,
    pub default: Option<HirExpr>,
    pub is_optional: bool,
    pub is_rest: bool, // For varargs
}

#[derive(Debug, Clone)]
pub struct HirClassField {
    pub symbol_id: SymbolId, // Symbol ID from TAST (needed for field access lowering)
    pub name: InternedString,
    pub ty: TypeId,
    pub init: Option<HirExpr>,
    pub visibility: HirVisibility,
    pub is_static: bool,
    pub is_final: bool,
    pub property_access: Option<crate::tast::PropertyAccessInfo>, // Property accessor info from TAST
}

#[derive(Debug, Clone)]
pub struct HirMethod {
    pub function: HirFunction,
    pub visibility: HirVisibility,
    pub is_static: bool,
    pub is_override: bool,
    pub is_abstract: bool,
}

#[derive(Debug, Clone)]
pub struct HirInterfaceField {
    pub name: InternedString,
    pub ty: TypeId,
    pub getter: bool,
    pub setter: bool,
}

#[derive(Debug, Clone)]
pub struct HirInterfaceMethod {
    pub name: InternedString,
    pub type_params: Vec<HirTypeParam>,
    pub params: Vec<HirParam>,
    pub return_type: TypeId,
}

#[derive(Debug, Clone)]
pub struct HirEnumField {
    pub name: InternedString,
    pub ty: TypeId,
}

#[derive(Debug, Clone)]
pub struct HirAbstractField {
    pub name: InternedString,
    pub ty: TypeId,
    pub getter: Option<SymbolId>,
    pub setter: Option<SymbolId>,
}

#[derive(Debug, Clone)]
pub struct HirGlobal {
    pub symbol_id: SymbolId,
    pub ty: TypeId,
    pub init: Option<HirExpr>,
    pub is_const: bool,
}

#[derive(Debug, Clone)]
pub struct HirSuperCall {
    pub args: Vec<HirExpr>,
}

#[derive(Debug, Clone)]
pub struct HirFieldInit {
    pub field: SymbolId,
    pub value: HirExpr,
}

#[derive(Debug, Clone)]
pub struct HirCapture {
    pub symbol: SymbolId,
    pub mode: HirCaptureMode,
    pub ty: TypeId,
}

#[derive(Debug, Clone, Copy)]
pub enum HirCaptureMode {
    ByValue,
    ByRef,
    ByMutableRef,
}

#[derive(Debug, Clone)]
pub struct HirComprehensionIterator {
    pub pattern: HirPattern,
    pub iterator: HirExpr,
}

#[derive(Debug, Clone)]
pub enum HirStringPart {
    Literal(InternedString),
    Interpolation(HirExpr),
}

#[derive(Debug, Clone, Copy)]
pub enum HirVisibility {
    Public,
    Private,
    Protected,
    Internal,
}

#[derive(Debug, Clone, Copy)]
pub enum HirCallingConvention {
    Haxe,
    C,
    Stdcall,
    Fastcall,
}

#[derive(Debug, Clone)]
pub struct HirMetadata {
    pub source_file: String,
    pub language_version: String,
    pub target_platforms: Vec<String>,
    pub optimization_hints: Vec<HirOptimizationHint>,
}

#[derive(Debug, Clone)]
pub enum HirOptimizationHint {
    Inline(SymbolId),
    NoInline(SymbolId),
    PureFunction(SymbolId),
    HotPath(Vec<SymbolId>),
    ColdPath(Vec<SymbolId>),
}

impl HirExpr {
    /// Create a new HIR expression
    pub fn new(kind: HirExprKind, ty: TypeId, lifetime: LifetimeId, loc: SourceLocation) -> Self {
        Self {
            kind,
            ty,
            lifetime,
            source_location: loc,
        }
    }
}

impl HirBlock {
    /// Create a new block
    pub fn new(statements: Vec<HirStatement>, scope: ScopeId) -> Self {
        Self {
            statements,
            expr: None,
            scope,
        }
    }

    /// Create a block with a trailing expression
    pub fn with_expr(statements: Vec<HirStatement>, expr: HirExpr, scope: ScopeId) -> Self {
        Self {
            statements,
            expr: Some(Box::new(expr)),
            scope,
        }
    }
}
