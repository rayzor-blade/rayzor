//! Haxe AST with full span tracking
//!
//! This AST covers 100% of Haxe language features as specified in https://haxe.org/manual

/// Source location information
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Span {
    /// Byte offset of the start (inclusive)
    pub start: usize,
    /// Byte offset of the end (exclusive)
    pub end: usize,
}

impl Span {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    pub fn merge(self, other: Span) -> Span {
        Span::new(self.start.min(other.start), self.end.max(other.end))
    }
}

/// A complete Haxe source file
#[derive(Debug, Clone, PartialEq)]
pub struct HaxeFile {
    pub filename: String,
    /// Source input preserved only when debug flag is enabled to minimize memory usage
    pub input: Option<String>,
    pub package: Option<Package>,
    pub imports: Vec<Import>,
    pub using: Vec<Using>,
    pub module_fields: Vec<ModuleField>,
    pub declarations: Vec<TypeDeclaration>,
    pub span: Span,
}

/// Package declaration: `package com.example;`
#[derive(Debug, Clone, PartialEq)]
pub struct Package {
    pub path: Vec<String>,
    pub span: Span,
}

/// Import declaration
#[derive(Debug, Clone, PartialEq)]
pub struct Import {
    pub path: Vec<String>,
    pub mode: ImportMode,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ImportMode {
    /// `import com.example.Class;`
    Normal,
    /// `import com.example.Class as Alias;`
    Alias(String),
    /// `import com.example.Class.field;`
    Field(String),
    /// `import com.example.*;`
    Wildcard,
    /// `import com.example.* except SomeClass, AnotherClass;`
    WildcardWithExclusions(Vec<String>),
}

/// Using declaration: `using Lambda;`
#[derive(Debug, Clone, PartialEq)]
pub struct Using {
    pub path: Vec<String>,
    pub span: Span,
}

/// Module-level field (variable or function declared at module level)
#[derive(Debug, Clone, PartialEq)]
pub struct ModuleField {
    pub meta: Vec<Metadata>,
    pub access: Option<Access>,
    pub modifiers: Vec<Modifier>,
    pub kind: ModuleFieldKind,
    pub span: Span,
}

/// Module-level field kind
#[derive(Debug, Clone, PartialEq)]
pub enum ModuleFieldKind {
    /// Variable field: `var x:Int = 10;`
    Var {
        name: String,
        type_hint: Option<Type>,
        expr: Option<Expr>,
    },
    /// Final field: `final x:Int = 10;`
    Final {
        name: String,
        type_hint: Option<Type>,
        expr: Option<Expr>,
    },
    /// Function: `function foo():Void {}`
    Function(Function),
}

impl ModuleField {
    pub fn span(&self) -> Span {
        self.span
    }
}

/// Conditional compilation block
#[derive(Debug, Clone, PartialEq)]
pub struct ConditionalBlock<T> {
    pub condition: ConditionalExpr,
    pub content: T,
    pub span: Span,
}

/// Conditional compilation expression
#[derive(Debug, Clone, PartialEq)]
pub enum ConditionalExpr {
    /// Simple identifier: `debug`
    Ident(String),
    /// Negation: `!debug`
    Not(Box<ConditionalExpr>),
    /// And: `debug && test`
    And(Box<ConditionalExpr>, Box<ConditionalExpr>),
    /// Or: `debug || test`
    Or(Box<ConditionalExpr>, Box<ConditionalExpr>),
    /// Parentheses: `(debug && test)`
    Paren(Box<ConditionalExpr>),
}

/// Conditional compilation directive
#[derive(Debug, Clone, PartialEq)]
pub struct ConditionalCompilation<T> {
    /// #if branches
    pub if_branch: ConditionalBlock<Vec<T>>,
    /// #elseif branches
    pub elseif_branches: Vec<ConditionalBlock<Vec<T>>>,
    /// #else branch
    pub else_branch: Option<Vec<T>>,
    pub span: Span,
}

/// Top-level type declarations
#[derive(Debug, Clone, PartialEq)]
pub enum TypeDeclaration {
    Class(ClassDecl),
    Interface(InterfaceDecl),
    Enum(EnumDecl),
    Typedef(TypedefDecl),
    Abstract(AbstractDecl),
    /// Conditional compilation block
    Conditional(ConditionalCompilation<TypeDeclaration>),
}

impl TypeDeclaration {
    pub fn span(&self) -> Span {
        match self {
            TypeDeclaration::Class(c) => c.span,
            TypeDeclaration::Interface(i) => i.span,
            TypeDeclaration::Enum(e) => e.span,
            TypeDeclaration::Typedef(t) => t.span,
            TypeDeclaration::Abstract(a) => a.span,
            TypeDeclaration::Conditional(c) => c.span,
        }
    }
}

/// Access modifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    Public,
    Private,
}

/// Function modifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Modifier {
    Static,
    Inline,
    Macro,
    Dynamic,
    Override,
    Final,
    Extern,
}

/// Metadata/Attributes: `@:meta`, `@:native("name")`
#[derive(Debug, Clone, PartialEq)]
pub struct Metadata {
    pub name: String,
    pub params: Vec<Expr>,
    pub span: Span,
}

/// Class declaration
#[derive(Debug, Clone, PartialEq)]
pub struct ClassDecl {
    pub meta: Vec<Metadata>,
    pub access: Option<Access>,
    pub modifiers: Vec<Modifier>,
    pub name: String,
    pub type_params: Vec<TypeParam>,
    pub extends: Option<Type>,
    pub implements: Vec<Type>,
    pub fields: Vec<ClassField>,
    pub span: Span,
}

impl ClassDecl {
    /// Check if the class has any constructors
    pub fn has_constructor(&self) -> bool {
        self.fields.iter().any(|field| match &field.kind {
            ClassFieldKind::Function(func) => func.name == "new",
            _ => false,
        })
    }

    /// Get all constructors (can be multiple with @:overload)
    pub fn get_constructors(&self) -> impl Iterator<Item = &ClassField> {
        self.fields.iter().filter(|field| match &field.kind {
            ClassFieldKind::Function(func) => func.name == "new",
            _ => false,
        })
    }

    /// Get the primary constructor (first one found)
    pub fn get_primary_constructor(&self) -> Option<&Function> {
        self.fields.iter().find_map(|field| match &field.kind {
            ClassFieldKind::Function(func) if func.name == "new" => Some(func),
            _ => None,
        })
    }

    /// Get all non-constructor fields
    pub fn get_non_constructor_fields(&self) -> impl Iterator<Item = &ClassField> {
        self.fields.iter().filter(|field| match &field.kind {
            ClassFieldKind::Function(func) => func.name != "new",
            _ => true,
        })
    }

    /// Get all methods (excluding constructors)
    pub fn get_methods(&self) -> impl Iterator<Item = &ClassField> {
        self.fields.iter().filter(|field| match &field.kind {
            ClassFieldKind::Function(func) => func.name != "new",
            _ => false,
        })
    }

    /// Get all variables and properties
    pub fn get_vars_and_properties(&self) -> impl Iterator<Item = &ClassField> {
        self.fields.iter().filter(|field| {
            matches!(
                &field.kind,
                ClassFieldKind::Var { .. }
                    | ClassFieldKind::Final { .. }
                    | ClassFieldKind::Property { .. }
            )
        })
    }

    /// Check if class has @:derive metadata with a specific trait
    /// Example: @:derive([Clone, Copy])
    pub fn has_derive_trait(&self, trait_name: &str) -> bool {
        self.get_derive_traits().iter().any(|t| t == trait_name)
    }

    /// Get all traits from @:derive metadata
    /// Example: @:derive([Clone, Copy]) returns vec!["Clone", "Copy"]
    pub fn get_derive_traits(&self) -> Vec<String> {
        self.meta
            .iter()
            .filter(|m| m.name == "derive")
            .flat_map(|m| {
                m.params
                    .iter()
                    .filter_map(|param| {
                        // Extract trait names from array expression
                        match &param.kind {
                            ExprKind::Array(items) => Some(
                                items
                                    .iter()
                                    .filter_map(|item| match &item.kind {
                                        ExprKind::Ident(name) => Some(name.clone()),
                                        _ => None,
                                    })
                                    .collect::<Vec<_>>(),
                            ),
                            ExprKind::Ident(name) => {
                                // Also support @:derive(Clone) without array
                                Some(vec![name.clone()])
                            }
                            _ => None,
                        }
                    })
                    .flatten()
            })
            .collect()
    }

    /// Check if metadata exists by name
    pub fn has_metadata(&self, name: &str) -> bool {
        self.meta.iter().any(|m| m.name == name)
    }

    /// Get metadata by name
    pub fn get_metadata(&self, name: &str) -> Option<&Metadata> {
        self.meta.iter().find(|m| m.name == name)
    }
}

/// Interface declaration
#[derive(Debug, Clone, PartialEq)]
pub struct InterfaceDecl {
    pub meta: Vec<Metadata>,
    pub access: Option<Access>,
    pub modifiers: Vec<Modifier>,
    pub name: String,
    pub type_params: Vec<TypeParam>,
    pub extends: Vec<Type>,
    pub fields: Vec<ClassField>,
    pub span: Span,
}

/// Enum declaration
#[derive(Debug, Clone, PartialEq)]
pub struct EnumDecl {
    pub meta: Vec<Metadata>,
    pub access: Option<Access>,
    pub name: String,
    pub type_params: Vec<TypeParam>,
    pub constructors: Vec<EnumConstructor>,
    pub span: Span,
}

/// Enum constructor
#[derive(Debug, Clone, PartialEq)]
pub struct EnumConstructor {
    pub meta: Vec<Metadata>,
    pub name: String,
    pub params: Vec<FunctionParam>,
    pub span: Span,
}

/// Typedef declaration
#[derive(Debug, Clone, PartialEq)]
pub struct TypedefDecl {
    pub meta: Vec<Metadata>,
    pub access: Option<Access>,
    pub name: String,
    pub type_params: Vec<TypeParam>,
    pub type_def: Type,
    pub span: Span,
}

/// Abstract declaration
#[derive(Debug, Clone, PartialEq)]
pub struct AbstractDecl {
    pub meta: Vec<Metadata>,
    pub access: Option<Access>,
    pub modifiers: Vec<Modifier>,
    pub name: String,
    pub type_params: Vec<TypeParam>,
    pub underlying: Option<Type>,
    pub from: Vec<Type>,
    pub to: Vec<Type>,
    pub fields: Vec<ClassField>,
    pub is_enum_abstract: bool,
    pub span: Span,
}

impl AbstractDecl {
    /// Check if the abstract has any constructors
    pub fn has_constructor(&self) -> bool {
        self.fields.iter().any(|field| match &field.kind {
            ClassFieldKind::Function(func) => func.name == "new",
            _ => false,
        })
    }

    /// Get all constructors
    pub fn get_constructors(&self) -> impl Iterator<Item = &ClassField> {
        self.fields.iter().filter(|field| match &field.kind {
            ClassFieldKind::Function(func) => func.name == "new",
            _ => false,
        })
    }

    /// Get the primary constructor
    pub fn get_primary_constructor(&self) -> Option<&Function> {
        self.fields.iter().find_map(|field| match &field.kind {
            ClassFieldKind::Function(func) if func.name == "new" => Some(func),
            _ => None,
        })
    }
}

/// Type variance annotation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variance {
    /// Invariant (no annotation)
    Invariant,
    /// Covariant (+ or out)
    Covariant,
    /// Contravariant (- or in)
    Contravariant,
}

/// Type parameter with constraints
#[derive(Debug, Clone, PartialEq)]
pub struct TypeParam {
    pub name: String,
    pub constraints: Vec<Type>,
    pub variance: Variance,
    pub span: Span,
}

/// Class field (variable, property, or function)
#[derive(Debug, Clone, PartialEq)]
pub struct ClassField {
    pub meta: Vec<Metadata>,
    pub access: Option<Access>,
    pub modifiers: Vec<Modifier>,
    pub kind: ClassFieldKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ClassFieldKind {
    /// Variable field: `var x:Int = 10;`
    Var {
        name: String,
        type_hint: Option<Type>,
        expr: Option<Expr>,
    },
    /// Final field: `final x:Int = 10;`
    Final {
        name: String,
        type_hint: Option<Type>,
        expr: Option<Expr>,
    },
    /// Property: `var x(get, set):Int;`
    Property {
        name: String,
        type_hint: Option<Type>,
        getter: PropertyAccess,
        setter: PropertyAccess,
    },
    /// Function: `function foo():Void {}`
    Function(Function),
}

/// Property access mode
#[derive(Debug, Clone, PartialEq)]
pub enum PropertyAccess {
    /// Default getter/setter
    Default,
    /// null access
    Null,
    /// Never allow access
    Never,
    /// Dynamic access
    Dynamic,
    /// Custom getter/setter name
    Custom(String),
}

/// Function declaration
#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    pub name: String,
    pub type_params: Vec<TypeParam>,
    pub params: Vec<FunctionParam>,
    pub return_type: Option<Type>,
    pub body: Option<Box<Expr>>,
    pub span: Span,
}

/// Function parameter
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionParam {
    pub meta: Vec<Metadata>,
    pub name: String,
    pub type_hint: Option<Type>,
    pub optional: bool,
    pub rest: bool,
    pub default_value: Option<Box<Expr>>,
    pub span: Span,
}

/// Arrow function parameter (supports optional type annotations)
#[derive(Debug, Clone, PartialEq)]
pub struct ArrowParam {
    pub name: String,
    pub type_hint: Option<Type>,
}

/// Type expressions
#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    /// Simple path: `Int`, `String`, `com.example.MyClass`
    Path {
        path: TypePath,
        params: Vec<Type>,
        span: Span,
    },
    /// Function type: `Int -> String -> Void`
    Function {
        params: Vec<Type>,
        ret: Box<Type>,
        span: Span,
    },
    /// Anonymous structure: `{ x:Int, y:String }`
    Anonymous { fields: Vec<AnonField>, span: Span },
    /// Optional type: `?Int`
    Optional { inner: Box<Type>, span: Span },
    /// Parenthesized type: `(Int)`
    Parenthesis { inner: Box<Type>, span: Span },
    /// Intersection type: `Type & { extraField: Int }`
    Intersection {
        left: Box<Type>,
        right: Box<Type>,
        span: Span,
    },
    /// Wildcard type: `?` (used in type parameters)
    Wildcard { span: Span },
}

/// Type path
#[derive(Debug, Clone, PartialEq)]
pub struct TypePath {
    pub package: Vec<String>,
    pub name: String,
    pub sub: Option<String>,
}

/// Anonymous structure field
#[derive(Debug, Clone, PartialEq)]
pub struct AnonField {
    pub name: String,
    pub optional: bool,
    pub type_hint: Type,
    pub span: Span,
}

/// Expressions
#[derive(Debug, Clone, PartialEq)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExprKind {
    /// Literals
    Int(i64),
    Float(f64),
    String(String),
    Bool(bool),
    Null,
    This,
    Super,

    /// Regex literal: ~/pattern/flags
    Regex {
        pattern: String,
        flags: String,
    },

    /// Identifiers and member access
    Ident(String),
    Field {
        expr: Box<Expr>,
        field: String,
        /// True for optional chaining: `obj?.field`
        is_optional: bool,
    },

    /// Array/Map access: `arr[0]`, `map["key"]`
    Index {
        expr: Box<Expr>,
        index: Box<Expr>,
    },

    /// Function call: `foo(1, 2)`
    Call {
        expr: Box<Expr>,
        args: Vec<Expr>,
    },

    /// Constructor call: `new MyClass()`
    New {
        type_path: TypePath,
        params: Vec<Type>,
        args: Vec<Expr>,
    },

    /// Unary operators
    Unary {
        op: UnaryOp,
        expr: Box<Expr>,
    },

    /// Binary operators
    Binary {
        left: Box<Expr>,
        op: BinaryOp,
        right: Box<Expr>,
    },

    /// Assignment
    Assign {
        left: Box<Expr>,
        op: AssignOp,
        right: Box<Expr>,
    },

    /// Ternary: `cond ? then : else`
    Ternary {
        cond: Box<Expr>,
        then_expr: Box<Expr>,
        else_expr: Box<Expr>,
    },

    /// Array literal: `[1, 2, 3]`
    Array(Vec<Expr>),

    /// Map literal: `["a" => 1, "b" => 2]`
    Map(Vec<(Expr, Expr)>),

    /// Object literal: `{x: 1, y: 2}`
    Object(Vec<ObjectField>),

    /// String interpolation: `'Hello $name'`
    StringInterpolation(Vec<StringPart>),

    /// Block expression: `{ statements; }`
    Block(Vec<BlockElement>),

    /// Variable declaration: `var x = 10`
    Var {
        name: String,
        type_hint: Option<Type>,
        expr: Option<Box<Expr>>,
    },

    /// Final declaration: `final x = 10`
    Final {
        name: String,
        type_hint: Option<Type>,
        expr: Option<Box<Expr>>,
    },

    /// Function expression: `function(x) return x * 2`
    Function(Function),

    /// Arrow function: `x -> x * 2` or `(x:Int) -> x * 2`
    Arrow {
        params: Vec<ArrowParam>,
        expr: Box<Expr>,
    },

    /// Return: `return expr`
    Return(Option<Box<Expr>>),

    /// Break
    Break,

    /// Continue
    Continue,

    /// Throw: `throw expr`
    Throw(Box<Expr>),

    /// If: `if (cond) then else else`
    If {
        cond: Box<Expr>,
        then_branch: Box<Expr>,
        else_branch: Option<Box<Expr>>,
    },

    /// Switch
    Switch {
        expr: Box<Expr>,
        cases: Vec<Case>,
        default: Option<Box<Expr>>,
    },

    /// For loop: `for (i in 0...10)` or `for (key => value in map)`
    For {
        var: String,
        key_var: Option<String>, // For key => value syntax
        iter: Box<Expr>,
        body: Box<Expr>,
    },

    /// While loop: `while (cond) body`
    While {
        cond: Box<Expr>,
        body: Box<Expr>,
    },

    /// Do-while: `do body while (cond)`
    DoWhile {
        body: Box<Expr>,
        cond: Box<Expr>,
    },

    /// Try-catch-finally
    Try {
        expr: Box<Expr>,
        catches: Vec<Catch>,
        finally_block: Option<Box<Expr>>,
    },

    /// Cast: `cast expr` or `cast(expr, Type)`
    Cast {
        expr: Box<Expr>,
        type_hint: Option<Type>,
    },

    /// Type check: `(expr : Type)`
    TypeCheck {
        expr: Box<Expr>,
        type_hint: Type,
    },

    /// Untyped expression: `untyped expr`
    Untyped(Box<Expr>),

    /// Metadata: `@:meta expr`
    Meta {
        meta: Metadata,
        expr: Box<Expr>,
    },

    /// Parentheses: `(expr)`
    Paren(Box<Expr>),

    /// Tuple literal: `(expr, expr, ...)`
    Tuple(Vec<Expr>),

    /// Macro expression: `macro expr`
    Macro(Box<Expr>),

    /// Inline expression: `inline expr` - forces inlining at call site
    Inline(Box<Expr>),

    /// Macro reification: `$expr`
    Reify(Box<Expr>),

    /// Dollar identifier: `$type`, `$v{...}`, `$i{...}`, etc.
    DollarIdent {
        name: String,
        arg: Option<Box<Expr>>,
    },

    /// Array comprehension: `[for (i in 0...10) i * 2]`
    ArrayComprehension {
        for_parts: Vec<ComprehensionFor>,
        expr: Box<Expr>,
    },

    /// Map comprehension: `[for (i in 0...10) i => i * 2]`
    MapComprehension {
        for_parts: Vec<ComprehensionFor>,
        key: Box<Expr>,
        value: Box<Expr>,
    },

    /// Compiler-specific code block: `__c__("code {0} {1}", arg0, arg1)`
    CompilerSpecific {
        target: String,
        code: Box<Expr>,
        args: Vec<Expr>,
    },
}

/// Object field in object literal
#[derive(Debug, Clone, PartialEq)]
pub struct ObjectField {
    pub name: String,
    pub expr: Expr,
    pub span: Span,
}

/// String interpolation part
#[derive(Debug, Clone, PartialEq)]
pub enum StringPart {
    Literal(String),
    Interpolation(Expr),
}

/// Block element (either a statement or expression)
#[derive(Debug, Clone, PartialEq)]
pub enum BlockElement {
    Expr(Expr),
    /// Import/using inside a block
    Import(Import),
    Using(Using),
    /// Conditional compilation inside a block
    Conditional(ConditionalCompilation<BlockElement>),
}

/// Switch case
#[derive(Debug, Clone, PartialEq)]
pub struct Case {
    pub patterns: Vec<Pattern>,
    pub guard: Option<Expr>,
    pub body: Expr,
    pub span: Span,
}

/// Pattern in switch case
#[derive(Debug, Clone, PartialEq)]
pub enum Pattern {
    /// Constant pattern: `1`, `"hello"`, `true`
    Const(Expr),
    /// Variable capture: `x`
    Var(String),
    /// Constructor pattern: `Some(x)`, `RGB(r, g, b)`
    Constructor {
        path: TypePath,
        params: Vec<Pattern>,
    },
    /// Array pattern: `[first, second]`
    Array(Vec<Pattern>),
    /// Array pattern with rest: `[first, ...rest]`
    ArrayRest {
        elements: Vec<Pattern>,
        rest: Option<String>,
    },
    /// Object pattern: `{x: 0, y: 0}`
    Object { fields: Vec<(String, Pattern)> },
    /// Type pattern: `(s:String)`
    Type { var: String, type_hint: Type },
    /// Null pattern
    Null,
    /// Or pattern: `1 | 2 | 3`
    Or(Vec<Pattern>),
    /// Underscore pattern: `_`
    Underscore,
    /// Extractor pattern: `_.method() => value` or `~/regex/.match(_) => true`
    Extractor { expr: Box<Expr>, value: Box<Expr> },
}

/// Catch clause
#[derive(Debug, Clone, PartialEq)]
pub struct Catch {
    pub var: String,
    pub type_hint: Option<Type>,
    pub filter: Option<Expr>,
    pub body: Expr,
    pub span: Span,
}

/// Comprehension for clause
#[derive(Debug, Clone, PartialEq)]
pub struct ComprehensionFor {
    pub var: String,
    pub key_var: Option<String>, // For key => value syntax
    pub iter: Expr,
    pub span: Span,
}

/// Unary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    /// `!expr`
    Not,
    /// `-expr`
    Neg,
    /// `~expr`
    BitNot,
    /// `++expr`
    PreIncr,
    /// `--expr`
    PreDecr,
    /// `expr++`
    PostIncr,
    /// `expr--`
    PostDecr,
}

/// Binary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Mod,

    // Comparison
    Eq,
    NotEq,
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
    Ushr,

    // Special
    Range,    // `...`
    Arrow,    // `=>`
    Is,       // `is` type check operator
    NullCoal, // `??`
}

/// Assignment operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssignOp {
    Assign,
    AddAssign,
    SubAssign,
    MulAssign,
    DivAssign,
    ModAssign,
    AndAssign,
    OrAssign,
    XorAssign,
    ShlAssign,
    ShrAssign,
    UshrAssign,
}
