//! Context API for Haxe Macro System
//!
//! Implements `haxe.macro.Context` methods that bridge macro execution
//! with the compiler's internal state (SymbolTable, TypeTable, ScopeTree, etc.).
//!
//! The `MacroContext` struct holds the current expansion context and provides
//! methods that macros can call during evaluation.

use super::errors::{MacroDiagnostic, MacroError, MacroSeverity};
use super::value::MacroValue;
use crate::tast::{SourceLocation, SymbolId, SymbolTable, TypeId, TypeTable};
use parser::HaxeFile;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;
use std::sync::Arc;

/// The expansion context available to macro functions.
///
/// Holds references to compiler state and tracks the current expansion
/// point (which class, which method, what position, etc.).
/// Typing services a macro body may call, answered by the live typer.
///
/// Macro expansion runs on the raw AST, before any typing exists, so a macro
/// that asks for a type (`Context.typeof`, `typeExpr`, …) cannot be answered
/// there. Such calls are DEFERRED: the call site is left in the AST and
/// re-expanded during lowering, where `AstLowering` implements this trait and
/// can type an expression in the scope the call sits in.
pub trait MacroTyper {
    /// Type `expr` in the scope enclosing the macro call. The typed result is
    /// discarded; only the type survives. An failure returns the message the
    /// typer would have reported.
    fn type_expr_in_scope(&mut self, expr: &parser::Expr) -> Result<TypeId, String>;
    /// `haxe.macro.TypeTools.toString` — the source-level spelling ("Null<Int>").
    fn type_display(&mut self, id: TypeId) -> String;
    /// `Std.string` on a `haxe.macro.Type` value — the constructor-shaped form.
    /// Only self-consistency is load-bearing (tests compare two of these), so
    /// it may share an implementation with `type_display`.
    fn type_std_string(&mut self, id: TypeId) -> String;

    /// `Context.getType` — resolve a dotted or bare type name in the scope of
    /// the module being lowered (private module types included).
    fn resolve_type_by_name(&mut self, name: &str) -> Result<TypeId, String>;

    /// `Context.makeMonomorph` — a fresh, unbound type variable. Each call
    /// must return a DISTINCT id; bindings are the context's to keep.
    fn fresh_monomorph(&mut self) -> TypeId;

    /// `Context.unify` — may bind monomorphs from `monomorphs` into
    /// `bindings`. True when the two types unify under those bindings.
    fn unify_types(
        &mut self,
        a: TypeId,
        b: TypeId,
        monomorphs: &std::collections::BTreeSet<TypeId>,
        bindings: &mut BTreeMap<TypeId, TypeId>,
    ) -> bool;

    /// The `haxe.macro.Type` constructor a type value presents as when a
    /// macro pattern-matches it: the variant name plus its positional
    /// payload. The def-shaped payload slot carries the type's own id as the
    /// handle — enough for the round trip `case TType(td, _)` then
    /// `TType(td, [args])`.
    fn type_adt_view(&mut self, id: TypeId) -> Option<(String, Vec<super::value::MacroValue>)>;

    /// Rebuild a typedef instance from its handle and fresh arguments —
    /// the reconstruction half of the round trip above.
    fn instantiate_alias(&mut self, def: TypeId, args: Vec<TypeId>) -> Option<TypeId>;
}

/// A scoped, non-owning handle to the live typer.
///
/// The interpreter owns its `MacroContext`, so the borrowed typer cannot be a
/// reference field. `expand_deferred_call` installs the pointer for exactly
/// one call and clears it before returning; the same discipline
/// `SymbolTableRef` above relies on.
#[derive(Clone, Copy)]
pub struct TyperRef {
    ptr: *mut (dyn MacroTyper + 'static),
}

impl TyperRef {
    /// # Safety
    /// The typer must outlive every dispatch made while this ref is installed.
    /// The lifetime is erased here for storage; the install/clear pairing in
    /// `expand_deferred_call` is what makes that sound.
    pub unsafe fn new(typer: &mut (dyn MacroTyper + '_)) -> Self {
        let ptr: *mut (dyn MacroTyper + '_) = typer;
        Self {
            ptr: std::mem::transmute::<*mut (dyn MacroTyper + '_), *mut (dyn MacroTyper + 'static)>(
                ptr,
            ),
        }
    }

    fn get(&mut self) -> &mut dyn MacroTyper {
        unsafe { &mut *self.ptr }
    }
}

pub struct MacroContext {
    /// Live typer, installed only for the span of a deferred re-expansion.
    typer: Option<TyperRef>,
    /// Monomorphs handed out by `makeMonomorph` during the current deferred
    /// call, and what `unify` bound them to. Cleared with the typer.
    monomorphs: std::collections::BTreeSet<TypeId>,
    mono_bindings: BTreeMap<TypeId, TypeId>,
    // --- Compiler state references ---
    /// Symbol table for resolving names and looking up symbols
    symbol_table: Option<SymbolTableRef>,

    /// Type table for resolving and creating types
    type_table: Option<Rc<RefCell<TypeTable>>>,

    // --- Current expansion context ---
    /// Position where the macro was invoked
    pub call_position: SourceLocation,

    /// Class being built (for @:build macros)
    pub build_class: Option<BuildClassContext>,

    /// Current module path (e.g., "com.example.MyModule")
    pub current_module: Option<String>,

    /// Current method name (if macro is called from within a method)
    pub current_method: Option<String>,

    /// Current class name (if macro is called from within a class)
    pub current_class: Option<String>,

    // --- Conditional compilation ---
    /// Conditional compilation defines (-D flags)
    pub defines: BTreeMap<String, String>,

    // --- Output collectors ---
    /// Diagnostics emitted by the macro
    pub diagnostics: Vec<MacroDiagnostic>,

    /// Types defined by the macro via defineType()
    pub defined_types: Vec<DefinedType>,

    /// Fields modified/added by @:build macros
    pub build_fields: Option<Vec<BuildField>>,
}

/// Reference to the symbol table (allows read-only access)
pub struct SymbolTableRef {
    ptr: *const SymbolTable,
}

impl SymbolTableRef {
    /// Create a new reference to a symbol table
    ///
    /// # Safety
    /// The caller must ensure the SymbolTable outlives this reference.
    pub unsafe fn new(table: &SymbolTable) -> Self {
        Self {
            ptr: table as *const SymbolTable,
        }
    }

    fn get(&self) -> &SymbolTable {
        unsafe { &*self.ptr }
    }
}

/// Context for @:build macro expansion on a class
#[derive(Debug, Clone)]
pub struct BuildClassContext {
    /// The class name
    pub class_name: String,
    /// Fully qualified name
    pub qualified_name: String,
    /// Symbol ID of the class (if resolved)
    pub symbol_id: Option<SymbolId>,
    /// Fields of the class before macro modification
    pub fields: Vec<BuildField>,
}

/// A field representation for build macros
///
/// Corresponds to `haxe.macro.Expr.Field` from the stdlib.
#[derive(Debug, Clone)]
pub struct BuildField {
    /// Field name
    pub name: String,
    /// Field kind
    pub kind: BuildFieldKind,
    /// Access modifiers
    pub access: Vec<FieldAccess>,
    /// Position/location
    pub pos: SourceLocation,
    /// Documentation string
    pub doc: Option<String>,
    /// Metadata annotations
    pub meta: Vec<FieldMeta>,
}

/// Kind of a build field
#[derive(Debug, Clone)]
pub enum BuildFieldKind {
    /// Variable field: `var x:Type = value;`
    Var {
        type_hint: Option<String>,
        expr: Option<Box<parser::Expr>>,
    },
    /// Function field: `function foo() {}`
    Function {
        params: Vec<String>,
        return_type: Option<String>,
        body: Option<Box<parser::Expr>>,
    },
    /// Property field: `var x(get, set):Type;`
    Property {
        get: String,
        set: String,
        type_hint: Option<String>,
    },
}

/// Access modifiers for build fields
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FieldAccess {
    Public,
    Private,
    Static,
    Override,
    Inline,
    Dynamic,
    Final,
    Extern,
    Abstract,
    Overload,
}

/// Metadata on a build field
#[derive(Debug, Clone)]
pub struct FieldMeta {
    /// Metadata name (e.g., "deprecated", "noCompletion")
    pub name: String,
    /// Optional arguments
    pub params: Vec<MacroValue>,
    /// Position
    pub pos: SourceLocation,
}

/// A type defined at compile time via Context.defineType()
#[derive(Debug, Clone)]
pub struct DefinedType {
    /// Package path (e.g., vec!["com", "example"])
    pub pack: Vec<String>,
    /// Type name
    pub name: String,
    /// Kind of type (class, interface, enum, typedef)
    pub kind: DefinedTypeKind,
    /// Fields for class/interface types
    pub fields: Vec<BuildField>,
    /// Position where defineType was called
    pub pos: SourceLocation,
}

/// Kind of a compile-time defined type
#[derive(Debug, Clone)]
pub enum DefinedTypeKind {
    Class,
    Interface,
    Enum,
    TypeAlias { target: String },
}

// === MacroContext Implementation ===

impl MacroContext {
    /// Create a new empty context (no compiler state attached)
    pub fn new() -> Self {
        Self {
            symbol_table: None,
            type_table: None,
            call_position: SourceLocation::unknown(),
            build_class: None,
            current_module: None,
            current_method: None,
            current_class: None,
            defines: BTreeMap::new(),
            typer: None,
            monomorphs: std::collections::BTreeSet::new(),
            mono_bindings: BTreeMap::new(),
            diagnostics: Vec::new(),
            defined_types: Vec::new(),
            build_fields: None,
        }
    }

    /// Create a context with compiler state attached
    ///
    /// # Safety
    /// The caller must ensure the SymbolTable outlives this context.
    pub unsafe fn with_compiler_state(
        symbol_table: &SymbolTable,
        type_table: Rc<RefCell<TypeTable>>,
    ) -> Self {
        Self {
            symbol_table: Some(SymbolTableRef::new(symbol_table)),
            type_table: Some(type_table),
            call_position: SourceLocation::unknown(),
            build_class: None,
            current_module: None,
            current_method: None,
            current_class: None,
            defines: BTreeMap::new(),
            typer: None,
            monomorphs: std::collections::BTreeSet::new(),
            mono_bindings: BTreeMap::new(),
            diagnostics: Vec::new(),
            defined_types: Vec::new(),
            build_fields: None,
        }
    }

    /// Set the macro call position
    pub fn set_call_position(&mut self, pos: SourceLocation) {
        self.call_position = pos;
    }

    /// Install the live typer for the span of one deferred re-expansion.
    ///
    /// # Safety
    /// The typer must outlive every dispatch until `clear_typer` runs. The
    /// caller pairs the two around a single `expand_macro_call`.
    pub unsafe fn set_typer(&mut self, typer: &mut dyn MacroTyper) {
        self.typer = Some(TyperRef::new(typer));
    }

    /// Remove the installed typer and the call-scoped monomorph state.
    /// Idempotent.
    pub fn clear_typer(&mut self) {
        self.typer = None;
        self.monomorphs.clear();
        self.mono_bindings.clear();
    }

    /// The ADT constructor view of a Type value, for pattern matching.
    pub fn project_type(&mut self, id: TypeId) -> Option<(String, Vec<MacroValue>)> {
        let t = self.typer.as_mut()?;
        t.get().type_adt_view(id)
    }

    /// A `haxe.macro.Type` value in whatever shape a macro produced it,
    /// resolved back to a TypeId: an opaque handle as-is, a constructed
    /// `TType(def, [args])` re-instantiated, any other constructor by its
    /// def-handle payload slot.
    pub fn coerce_type_value(&mut self, value: &MacroValue) -> Option<TypeId> {
        match value {
            MacroValue::Type(id) => Some(*id),
            MacroValue::Enum(_, variant, payload) => {
                let def = match payload.first() {
                    Some(MacroValue::Type(id)) => *id,
                    _ => return None,
                };
                if &**variant == "TType" {
                    if let Some(MacroValue::Array(items)) = payload.get(1) {
                        let args: Option<Vec<TypeId>> = items
                            .iter()
                            .map(|v| match v {
                                MacroValue::Type(id) => Some(*id),
                                _ => None,
                            })
                            .collect();
                        if let Some(args) = args {
                            let t = self.typer.as_mut()?;
                            return t.get().instantiate_alias(def, args);
                        }
                    }
                }
                Some(def)
            }
            _ => None,
        }
    }

    /// Copy another context's installed typer into this one. The interpreter
    /// builds its own context per expansion; this is how the typer installed
    /// on the expander's context reaches it. Sound for the same reason and
    /// span as `set_typer`: both copies die before `clear_typer` runs.
    pub fn adopt_typer_from(&mut self, other: &MacroContext) {
        self.typer = other.typer;
    }

    /// Whether a live typer is installed (deferred re-expansion in progress).
    pub fn has_typer(&self) -> bool {
        self.typer.is_some()
    }

    /// `TypeTools.toString` / `Std.string` rendering for a `Type` value, when
    /// the live typer is installed.
    pub fn format_type(&mut self, id: TypeId, constructor_form: bool) -> Option<String> {
        // A bound monomorph prints as what it was bound to.
        let mut id = id;
        let mut hops = 0;
        while let Some(&bound) = self.mono_bindings.get(&id) {
            id = bound;
            hops += 1;
            if hops > 32 {
                break;
            }
        }
        let t = self.typer.as_mut()?;
        Some(if constructor_form {
            t.get().type_std_string(id)
        } else {
            t.get().type_display(id)
        })
    }

    /// Set the build class context (for @:build macros)
    pub fn set_build_class(&mut self, ctx: BuildClassContext) {
        // Initialize build_fields from the class fields
        self.build_fields = Some(ctx.fields.clone());
        self.build_class = Some(ctx);
    }

    // ==========================================================
    // Context API methods (matching haxe.macro.Context)
    // ==========================================================

    /// `Context.error(msg, pos)` — Emit a compilation error
    ///
    /// Adds an error diagnostic and returns a MacroError to abort the macro.
    pub fn error(&mut self, msg: &str, pos: SourceLocation) -> MacroError {
        self.diagnostics.push(MacroDiagnostic::error(msg, pos));
        MacroError::ContextError {
            method: "error".to_string(),
            message: msg.to_string(),
            location: pos,
        }
    }

    /// `Context.fatalError(msg, pos)` — Emit a fatal compilation error
    pub fn fatal_error(&mut self, msg: &str, pos: SourceLocation) -> MacroError {
        self.diagnostics
            .push(MacroDiagnostic::error(format!("[fatal] {}", msg), pos));
        MacroError::ContextError {
            method: "fatalError".to_string(),
            message: msg.to_string(),
            location: pos,
        }
    }

    /// `Context.reportError(msg, pos)` — Emit error without aborting
    pub fn report_error(&mut self, msg: &str, pos: SourceLocation) {
        self.diagnostics.push(MacroDiagnostic::error(msg, pos));
    }

    /// `Context.warning(msg, pos)` — Emit a compilation warning
    pub fn warning(&mut self, msg: &str, pos: SourceLocation) {
        self.diagnostics.push(MacroDiagnostic::warning(msg, pos));
    }

    /// `Context.info(msg, pos)` — Emit a compilation info message
    pub fn info(&mut self, msg: &str, pos: SourceLocation) {
        self.diagnostics.push(MacroDiagnostic::info(msg, pos));
    }

    /// `Context.currentPos()` — Get the position where the macro was called
    pub fn current_pos(&self) -> MacroValue {
        MacroValue::Position(self.call_position)
    }

    /// `Context.getLocalClass()` — Get a `Ref<ClassType>`-like value for
    /// the current class. In Haxe's macro API the result is a `Null<Ref<T>>`
    /// that you call `.get()` on to dereference. We model that with an
    /// Object that carries the class info directly; `.get()` on objects is
    /// a passthrough (see `method_call` / `field_access`) so both
    /// `cls.get().name` and `cls.name` resolve to the class name.
    pub fn get_local_class(&self) -> MacroValue {
        match &self.current_class {
            Some(name) => {
                let mut obj = std::collections::BTreeMap::new();
                obj.insert(
                    "name".to_string(),
                    MacroValue::String(Arc::from(name.as_str())),
                );
                // Mirror tink-style class metadata where reasonable.
                if let Some(build) = &self.build_class {
                    obj.insert("pack".to_string(), MacroValue::Array(Arc::new(Vec::new())));
                    obj.insert(
                        "module".to_string(),
                        MacroValue::String(Arc::from(build.qualified_name.as_str())),
                    );
                }
                MacroValue::Object(Arc::new(obj))
            }
            None => MacroValue::Null,
        }
    }

    /// `Context.getLocalModule()` — Get the current module path
    pub fn get_local_module(&self) -> MacroValue {
        match &self.current_module {
            Some(path) => MacroValue::String(Arc::from(path.as_str())),
            None => MacroValue::Null,
        }
    }

    /// `Context.getLocalMethod()` — Get the current method name (or Null)
    pub fn get_local_method(&self) -> MacroValue {
        match &self.current_method {
            Some(name) => MacroValue::String(Arc::from(name.as_str())),
            None => MacroValue::Null,
        }
    }

    /// `Context.defined(flag)` — Check if a conditional compilation flag is set
    pub fn defined(&self, flag: &str) -> MacroValue {
        MacroValue::Bool(self.defines.contains_key(flag))
    }

    /// `Context.definedValue(key)` — Get value of a conditional compilation flag
    pub fn defined_value(&self, key: &str) -> MacroValue {
        match self.defines.get(key) {
            Some(val) => MacroValue::String(Arc::from(val.as_str())),
            None => MacroValue::Null,
        }
    }

    /// `Context.getDefines()` — Get all conditional compilation defines
    pub fn get_defines(&self) -> MacroValue {
        let mut obj = BTreeMap::new();
        for (k, v) in &self.defines {
            obj.insert(k.clone(), MacroValue::String(Arc::from(v.as_str())));
        }
        MacroValue::Object(Arc::new(obj))
    }

    /// `Context.getType(name)` — Resolve a type by its qualified name
    ///
    /// Returns a MacroValue::Type(TypeId) or an error if not found.
    pub fn get_type(
        &mut self,
        name: &str,
        location: SourceLocation,
    ) -> Result<MacroValue, MacroError> {
        let Some(typer) = self.typer.as_mut() else {
            return Err(MacroError::NeedsTyper { location });
        };
        match typer.get().resolve_type_by_name(name) {
            Ok(id) => Ok(MacroValue::Type(id)),
            Err(message) => Err(MacroError::ContextError {
                method: "getType".to_string(),
                message,
                location,
            }),
        }
    }

    /// `Context.typeof(expr)` — Get the type of an expression.
    ///
    /// Answerable only while the live typer is installed (deferred
    /// re-expansion during lowering). Before that, the error below is the
    /// signal the expander reads to defer the whole macro call.
    pub fn typeof_expr(
        &mut self,
        expr: &parser::Expr,
        location: SourceLocation,
    ) -> Result<MacroValue, MacroError> {
        let Some(typer) = self.typer.as_mut() else {
            return Err(MacroError::NeedsTyper { location });
        };
        match typer.get().type_expr_in_scope(expr) {
            Ok(id) => Ok(MacroValue::Type(id)),
            Err(message) => Err(MacroError::ContextError {
                method: "typeof".to_string(),
                message,
                location,
            }),
        }
    }

    /// `Context.typeExpr(expr)` — type an expression, returning a TypedExpr
    /// view. Only `t` carries a real value; the macros in the wild read `.t`
    /// and hand it to `TypeTools.toString`.
    pub fn type_expr(
        &mut self,
        expr: &parser::Expr,
        location: SourceLocation,
    ) -> Result<MacroValue, MacroError> {
        let typed = self.typeof_expr(expr, location).map_err(|e| match e {
            MacroError::ContextError {
                message, location, ..
            } => MacroError::ContextError {
                method: "typeExpr".to_string(),
                message,
                location,
            },
            other => other,
        })?;
        let mut obj = BTreeMap::new();
        obj.insert("t".to_string(), typed);
        obj.insert("pos".to_string(), MacroValue::Position(location));
        // The expression itself, so `Context.storeTypedExpr` can give back what
        // was typed. A real typed AST is not kept, and this round trip is what
        // the `storeTypedExpr(typeExpr(e))` idiom actually needs.
        obj.insert("expr".to_string(), MacroValue::Expr(Arc::new(expr.clone())));
        Ok(MacroValue::Object(Arc::new(obj)))
    }

    /// `Context.parse(expr, pos)` — Parse a string as Haxe code
    ///
    /// Returns the parsed expression as a MacroValue::Expr.
    pub fn parse(&self, code: &str, location: SourceLocation) -> Result<MacroValue, MacroError> {
        // Wrap in a class/function context so the parser can handle it
        let wrapper = format!(
            "class __MacroParse__ {{ static function __parse__() {{ {}; }} }}",
            code
        );

        let file = parser::parse_haxe_file("__macro_parse__", &wrapper, false).map_err(|e| {
            MacroError::ContextError {
                method: "parse".to_string(),
                message: format!("parse error: {:?}", e),
                location,
            }
        })?;

        // Extract the expression from the parsed wrapper
        if let Some(expr) = extract_body_expr(&file) {
            Ok(MacroValue::Expr(Arc::new(expr)))
        } else {
            Err(MacroError::ContextError {
                method: "parse".to_string(),
                message: "failed to extract expression from parsed code".to_string(),
                location,
            })
        }
    }

    /// `Context.makeExpr(value, pos)` — Build an AST expression from a value
    pub fn make_expr(
        &self,
        value: &MacroValue,
        _location: SourceLocation,
    ) -> Result<MacroValue, MacroError> {
        let expr = super::ast_bridge::value_to_expr(value);
        Ok(MacroValue::Expr(Arc::new(expr)))
    }

    /// `Context.getBuildFields()` — Get fields of the class being built
    ///
    /// Only available in @:build macro context.
    pub fn get_build_fields(&self, location: SourceLocation) -> Result<MacroValue, MacroError> {
        let fields = self
            .build_fields
            .as_ref()
            .ok_or_else(|| MacroError::ContextError {
                method: "getBuildFields".to_string(),
                message: "getBuildFields() is only available in @:build macro context".to_string(),
                location,
            })?;

        // Convert build fields to an array of objects
        let field_values: Vec<MacroValue> = fields.iter().map(build_field_to_value).collect();

        Ok(MacroValue::Array(Arc::new(field_values)))
    }

    /// `Context.defineType(typeDefinition)` — Define a new type at compile time
    pub fn define_type(
        &mut self,
        type_def: DefinedType,
        location: SourceLocation,
    ) -> Result<MacroValue, MacroError> {
        // Validate the type definition
        if type_def.name.is_empty() {
            return Err(MacroError::ContextError {
                method: "defineType".to_string(),
                message: "type name cannot be empty".to_string(),
                location,
            });
        }

        // Check for duplicate definitions
        if self
            .defined_types
            .iter()
            .any(|t| t.name == type_def.name && t.pack == type_def.pack)
        {
            return Err(MacroError::ContextError {
                method: "defineType".to_string(),
                message: format!(
                    "type '{}.{}' already defined",
                    type_def.pack.join("."),
                    type_def.name
                ),
                location,
            });
        }

        self.defined_types.push(type_def);
        Ok(MacroValue::Null)
    }

    /// `Context.getPosInfos(pos)` — Get position information
    pub fn get_pos_infos(&self, pos: &SourceLocation) -> MacroValue {
        let mut obj = BTreeMap::new();
        obj.insert("file".to_string(), MacroValue::Int(pos.file_id as i64));
        obj.insert("min".to_string(), MacroValue::Int(pos.byte_offset as i64));
        obj.insert("max".to_string(), MacroValue::Int(pos.byte_offset as i64));
        MacroValue::Object(Arc::new(obj))
    }

    /// `Context.makePosition(inf)` — Build a position from info object
    pub fn make_position(&self, info: &MacroValue) -> Result<MacroValue, MacroError> {
        if let MacroValue::Object(obj) = info {
            let file_id = obj.get("file").and_then(|v| v.as_int()).unwrap_or(0) as u32;
            let min = obj.get("min").and_then(|v| v.as_int()).unwrap_or(0) as u32;
            let _max = obj.get("max").and_then(|v| v.as_int()).unwrap_or(0);

            Ok(MacroValue::Position(SourceLocation::new(
                file_id, 0, 0, min,
            )))
        } else {
            Err(MacroError::TypeError {
                message: "makePosition expects an object with {file, min, max}".to_string(),
                location: SourceLocation::unknown(),
            })
        }
    }

    // ==========================================================
    // Method dispatch — called from the interpreter
    // ==========================================================

    /// Dispatch a Context method call from the interpreter.
    ///
    /// Maps `Context.methodName(args)` calls to the appropriate method.
    pub fn dispatch(
        &mut self,
        method: &str,
        args: &[MacroValue],
        location: SourceLocation,
    ) -> Result<MacroValue, MacroError> {
        match method {
            "error" => {
                let msg = arg_as_string(args, 0, "error", location)?;
                let pos = arg_as_position(args, 1).unwrap_or(location);
                Err(self.error(&msg, pos))
            }
            "fatalError" => {
                let msg = arg_as_string(args, 0, "fatalError", location)?;
                let pos = arg_as_position(args, 1).unwrap_or(location);
                Err(self.fatal_error(&msg, pos))
            }
            "reportError" => {
                let msg = arg_as_string(args, 0, "reportError", location)?;
                let pos = arg_as_position(args, 1).unwrap_or(location);
                self.report_error(&msg, pos);
                Ok(MacroValue::Null)
            }
            "warning" => {
                let msg = arg_as_string(args, 0, "warning", location)?;
                let pos = arg_as_position(args, 1).unwrap_or(location);
                self.warning(&msg, pos);
                Ok(MacroValue::Null)
            }
            "info" => {
                let msg = arg_as_string(args, 0, "info", location)?;
                let pos = arg_as_position(args, 1).unwrap_or(location);
                self.info(&msg, pos);
                Ok(MacroValue::Null)
            }
            "currentPos" => Ok(self.current_pos()),
            "getLocalClass" => Ok(self.get_local_class()),
            "getLocalModule" => Ok(self.get_local_module()),
            "getLocalMethod" => Ok(self.get_local_method()),
            "defined" => {
                let flag = arg_as_string(args, 0, "defined", location)?;
                Ok(self.defined(&flag))
            }
            "definedValue" => {
                let key = arg_as_string(args, 0, "definedValue", location)?;
                Ok(self.defined_value(&key))
            }
            "getDefines" => Ok(self.get_defines()),
            // `Context.getExpectedType()` — Haxe types this `Null<Type>` and
            // null is the honest answer: the expected type at a macro call
            // site is not tracked. A macro that branches on it takes its
            // unknown path instead of failing to expand at all.
            "getExpectedType" => Ok(MacroValue::Null),
            // `Context.toComplexType(t)` — the syntax form of a type. Built
            // from the type's source spelling, so a plain path converts and
            // anything with parameters, a function arrow or a structure gives
            // null, which the signature already allows.
            "toComplexType" => {
                let Some(MacroValue::Type(id)) = args.first().cloned() else {
                    return Err(MacroError::ContextError {
                        method: "toComplexType".to_string(),
                        message: "toComplexType expects a Type argument".to_string(),
                        location,
                    });
                };
                let Some(typer) = self.typer.as_mut() else {
                    return Err(MacroError::NeedsTyper { location });
                };
                let spelling = typer.get().type_display(id);
                if spelling.is_empty()
                    || spelling.contains(['<', '{', '(', '>', '-'])
                {
                    return Ok(MacroValue::Null);
                }
                let mut pack: Vec<&str> = spelling.split('.').collect();
                let name = pack.pop().unwrap_or("").to_string();
                let mut obj = BTreeMap::new();
                obj.insert("kind".to_string(), MacroValue::String("TPath".into()));
                obj.insert(
                    "pack".to_string(),
                    MacroValue::Array(Arc::new(
                        pack.into_iter()
                            .map(|p| MacroValue::String(p.into()))
                            .collect(),
                    )),
                );
                obj.insert("name".to_string(), MacroValue::String(name.into()));
                obj.insert("params".to_string(), MacroValue::Array(Arc::new(Vec::new())));
                obj.insert("sub".to_string(), MacroValue::Null);
                Ok(MacroValue::Object(Arc::new(obj)))
            }
            // `Context.storeTypedExpr(t)` — hand back the expression `typeExpr`
            // was given. Haxe stores a typed AST and returns a reference to it;
            // re-expanding the source expression is the same thing to every
            // caller that does not inspect the result.
            "storeTypedExpr" => match args.first() {
                Some(MacroValue::Object(fields)) => match fields.get("expr") {
                    Some(expr @ MacroValue::Expr(_)) => Ok(expr.clone()),
                    _ => Err(MacroError::ContextError {
                        method: "storeTypedExpr".to_string(),
                        message: "storeTypedExpr expects the result of Context.typeExpr"
                            .to_string(),
                        location,
                    }),
                },
                Some(expr @ MacroValue::Expr(_)) => Ok(expr.clone()),
                _ => Err(MacroError::ContextError {
                    method: "storeTypedExpr".to_string(),
                    message: "storeTypedExpr expects a typed expression".to_string(),
                    location,
                }),
            },
            "getType" => {
                let name = arg_as_string(args, 0, "getType", location)?;
                self.get_type(&name, location)
            }
            "typeof" => {
                if let Some(MacroValue::Expr(expr)) = args.first() {
                    let expr = expr.clone();
                    self.typeof_expr(&expr, location)
                } else {
                    Err(MacroError::ContextError {
                        method: "typeof".to_string(),
                        message: "typeof expects an Expr argument".to_string(),
                        location,
                    })
                }
            }
            "typeExpr" => {
                if let Some(MacroValue::Expr(expr)) = args.first() {
                    let expr = expr.clone();
                    self.type_expr(&expr, location)
                } else {
                    Err(MacroError::ContextError {
                        method: "typeExpr".to_string(),
                        message: "typeExpr expects an Expr argument".to_string(),
                        location,
                    })
                }
            }
            "makeMonomorph" => {
                let Some(typer) = self.typer.as_mut() else {
                    return Err(MacroError::NeedsTyper { location });
                };
                let id = typer.get().fresh_monomorph();
                self.monomorphs.insert(id);
                Ok(MacroValue::Type(id))
            }
            "unify" => {
                if self.typer.is_none() {
                    return Err(MacroError::NeedsTyper { location });
                }
                let (Some(first), Some(second)) = (args.first().cloned(), args.get(1).cloned())
                else {
                    return Err(MacroError::ContextError {
                        method: "unify".to_string(),
                        message: "unify expects two Type arguments".to_string(),
                        location,
                    });
                };
                let (Some(a), Some(b)) = (
                    self.coerce_type_value(&first),
                    self.coerce_type_value(&second),
                ) else {
                    return Err(MacroError::ContextError {
                        method: "unify".to_string(),
                        message: "unify expects two Type arguments".to_string(),
                        location,
                    });
                };
                let Some(typer) = self.typer.as_mut() else {
                    return Err(MacroError::NeedsTyper { location });
                };
                let ok = typer
                    .get()
                    .unify_types(a, b, &self.monomorphs, &mut self.mono_bindings);
                Ok(MacroValue::Bool(ok))
            }
            "parse" => {
                let code = arg_as_string(args, 0, "parse", location)?;
                self.parse(&code, location)
            }
            "makeExpr" => {
                if let Some(val) = args.first() {
                    self.make_expr(val, location)
                } else {
                    Ok(MacroValue::Null)
                }
            }
            "getBuildFields" => self.get_build_fields(location),
            "defineType" => {
                // Extract type definition from MacroValue::Object
                let type_def = value_to_defined_type(args.first(), location)?;
                self.define_type(type_def, location)
            }
            "getPosInfos" => {
                if let Some(MacroValue::Position(pos)) = args.first() {
                    Ok(self.get_pos_infos(pos))
                } else {
                    Ok(self.get_pos_infos(&SourceLocation::unknown()))
                }
            }
            "makePosition" => {
                if let Some(val) = args.first() {
                    self.make_position(val)
                } else {
                    Ok(MacroValue::Position(SourceLocation::unknown()))
                }
            }
            _ => Err(MacroError::ContextError {
                method: method.to_string(),
                message: format!("unknown Context method: '{}'", method),
                location,
            }),
        }
    }

    /// Take all collected diagnostics (draining the internal list)
    pub fn take_diagnostics(&mut self) -> Vec<MacroDiagnostic> {
        std::mem::take(&mut self.diagnostics)
    }

    /// Take all defined types (draining the internal list)
    pub fn take_defined_types(&mut self) -> Vec<DefinedType> {
        std::mem::take(&mut self.defined_types)
    }

    /// Take the modified build fields (if any)
    pub fn take_build_fields(&mut self) -> Option<Vec<BuildField>> {
        self.build_fields.take()
    }
}

impl Default for MacroContext {
    fn default() -> Self {
        Self::new()
    }
}

// ==========================================================
// Helper functions
// ==========================================================

/// Extract a string argument from the args list
fn arg_as_string(
    args: &[MacroValue],
    index: usize,
    method: &str,
    location: SourceLocation,
) -> Result<String, MacroError> {
    match args.get(index) {
        Some(MacroValue::String(s)) => Ok(s.to_string()),
        Some(other) => Ok(other.to_display_string()),
        None => Err(MacroError::ContextError {
            method: method.to_string(),
            message: format!("missing argument {} for Context.{}()", index, method),
            location,
        }),
    }
}

/// Extract a position argument from the args list
fn arg_as_position(args: &[MacroValue], index: usize) -> Option<SourceLocation> {
    match args.get(index) {
        Some(MacroValue::Position(pos)) => Some(*pos),
        _ => None,
    }
}

/// Convert a BuildField to a MacroValue (Object representation)
///
/// Matches the `haxe.macro.Expr.Field` structure.
fn build_field_to_value(field: &BuildField) -> MacroValue {
    let mut obj = BTreeMap::new();
    obj.insert(
        "name".to_string(),
        MacroValue::String(Arc::from(field.name.as_str())),
    );
    obj.insert("pos".to_string(), MacroValue::Position(field.pos));

    // Access modifiers
    let access: Vec<MacroValue> = field
        .access
        .iter()
        .map(|a| MacroValue::String(Arc::from(format!("{:?}", a).as_str())))
        .collect();
    obj.insert("access".to_string(), MacroValue::Array(Arc::new(access)));

    // Kind
    let kind_value = match &field.kind {
        BuildFieldKind::Var { type_hint, expr } => {
            let mut kind_obj = BTreeMap::new();
            kind_obj.insert("kind".to_string(), MacroValue::String(Arc::from("FVar")));
            let type_val = match type_hint {
                Some(t) => MacroValue::String(Arc::from(t.as_str())),
                None => MacroValue::Null,
            };
            let expr_val = match expr {
                Some(e) => MacroValue::Expr(Arc::from(e.as_ref().clone())),
                None => MacroValue::Null,
            };
            kind_obj.insert("type".to_string(), type_val.clone());
            kind_obj.insert("expr".to_string(), expr_val.clone());
            // Positional `__args__` array so enum-pattern matchers
            // (`switch f.kind { case FVar(t, _): ... }`) can bind the
            // constructor's positional args by index. Mirrors
            // `haxe.macro.Expr.FieldType.FVar(t:ComplexType, e:Expr)`.
            kind_obj.insert(
                "__args__".to_string(),
                MacroValue::Array(Arc::new(vec![type_val, expr_val])),
            );
            MacroValue::Object(Arc::new(kind_obj))
        }
        BuildFieldKind::Function {
            params,
            return_type,
            body,
        } => {
            let mut kind_obj = BTreeMap::new();
            kind_obj.insert("kind".to_string(), MacroValue::String(Arc::from("FFun")));
            kind_obj.insert(
                "args".to_string(),
                MacroValue::Array(Arc::new(
                    params
                        .iter()
                        .map(|p| MacroValue::String(Arc::from(p.as_str())))
                        .collect(),
                )),
            );
            if let Some(rt) = return_type {
                kind_obj.insert(
                    "ret".to_string(),
                    MacroValue::String(Arc::from(rt.as_str())),
                );
            }
            if let Some(b) = body {
                kind_obj.insert(
                    "expr".to_string(),
                    MacroValue::Expr(Arc::from(b.as_ref().clone())),
                );
            }
            MacroValue::Object(Arc::new(kind_obj))
        }
        BuildFieldKind::Property {
            get,
            set,
            type_hint,
        } => {
            let mut kind_obj = BTreeMap::new();
            kind_obj.insert("kind".to_string(), MacroValue::String(Arc::from("FProp")));
            kind_obj.insert(
                "get".to_string(),
                MacroValue::String(Arc::from(get.as_str())),
            );
            kind_obj.insert(
                "set".to_string(),
                MacroValue::String(Arc::from(set.as_str())),
            );
            if let Some(t) = type_hint {
                kind_obj.insert(
                    "type".to_string(),
                    MacroValue::String(Arc::from(t.as_str())),
                );
            }
            MacroValue::Object(Arc::new(kind_obj))
        }
    };
    obj.insert("kind".to_string(), kind_value);

    // Documentation
    if let Some(doc) = &field.doc {
        obj.insert(
            "doc".to_string(),
            MacroValue::String(Arc::from(doc.as_str())),
        );
    }

    // Metadata
    let meta: Vec<MacroValue> = field
        .meta
        .iter()
        .map(|m| {
            let mut meta_obj = BTreeMap::new();
            meta_obj.insert(
                "name".to_string(),
                MacroValue::String(Arc::from(m.name.as_str())),
            );
            meta_obj.insert(
                "params".to_string(),
                MacroValue::Array(Arc::new(m.params.clone())),
            );
            meta_obj.insert("pos".to_string(), MacroValue::Position(m.pos));
            MacroValue::Object(Arc::new(meta_obj))
        })
        .collect();
    obj.insert("meta".to_string(), MacroValue::Array(Arc::new(meta)));

    MacroValue::Object(Arc::new(obj))
}

/// Extract the body expression from a parsed wrapper file
fn extract_body_expr(file: &HaxeFile) -> Option<parser::Expr> {
    if let Some(decl) = file.declarations.first() {
        if let parser::TypeDeclaration::Class(class) = decl {
            for field in &class.fields {
                if let parser::ClassFieldKind::Function(func) = &field.kind {
                    if let Some(body) = &func.body {
                        // The body is a Block; extract the first statement's expression
                        if let parser::ExprKind::Block(stmts) = &body.kind {
                            if let Some(parser::BlockElement::Expr(first)) = stmts.first() {
                                return Some(first.clone());
                            }
                        }
                        return Some((**body).clone());
                    }
                }
            }
        }
    }
    None
}

/// Convert a MacroValue (Object) to a DefinedType
fn value_to_defined_type(
    value: Option<&MacroValue>,
    location: SourceLocation,
) -> Result<DefinedType, MacroError> {
    let obj = match value {
        Some(MacroValue::Object(o)) => o,
        _ => {
            return Err(MacroError::ContextError {
                method: "defineType".to_string(),
                message: "defineType expects an object argument".to_string(),
                location,
            });
        }
    };

    let name = obj
        .get("name")
        .and_then(|v| v.as_string())
        .unwrap_or("")
        .to_string();

    let pack = match obj.get("pack") {
        Some(MacroValue::Array(arr)) => arr
            .iter()
            .filter_map(|v| v.as_string().map(String::from))
            .collect(),
        _ => Vec::new(),
    };

    let kind_str = obj
        .get("kind")
        .and_then(|v| v.as_string())
        .unwrap_or("class");

    let kind = match kind_str {
        "interface" => DefinedTypeKind::Interface,
        "enum" => DefinedTypeKind::Enum,
        "typedef" | "alias" => {
            let target = obj
                .get("target")
                .and_then(|v| v.as_string())
                .unwrap_or("")
                .to_string();
            DefinedTypeKind::TypeAlias { target }
        }
        _ => DefinedTypeKind::Class,
    };

    // Extract fields (simplified — full implementation in Phase 6)
    let fields = match obj.get("fields") {
        Some(MacroValue::Array(arr)) => {
            arr.iter().filter_map(|v| value_to_build_field(v)).collect()
        }
        _ => Vec::new(),
    };

    Ok(DefinedType {
        pack,
        name,
        kind,
        fields,
        pos: location,
    })
}

/// Convert a MacroValue to a BuildField (simplified)
fn value_to_build_field(value: &MacroValue) -> Option<BuildField> {
    let obj = match value {
        MacroValue::Object(o) => o,
        _ => return None,
    };

    let name = obj.get("name")?.as_string()?.to_string();

    Some(BuildField {
        name,
        kind: BuildFieldKind::Var {
            type_hint: obj
                .get("type")
                .and_then(|v| v.as_string())
                .map(String::from),
            expr: None,
        },
        access: Vec::new(),
        pos: SourceLocation::unknown(),
        doc: obj.get("doc").and_then(|v| v.as_string()).map(String::from),
        meta: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_emits_diagnostic() {
        let mut ctx = MacroContext::new();
        let err = ctx.error("something went wrong", SourceLocation::unknown());
        assert!(matches!(err, MacroError::ContextError { .. }));
        assert_eq!(ctx.diagnostics.len(), 1);
        assert_eq!(ctx.diagnostics[0].severity, MacroSeverity::Error);
        assert_eq!(ctx.diagnostics[0].message, "something went wrong");
    }

    #[test]
    fn test_warning_emits_diagnostic() {
        let mut ctx = MacroContext::new();
        ctx.warning("be careful", SourceLocation::unknown());
        assert_eq!(ctx.diagnostics.len(), 1);
        assert_eq!(ctx.diagnostics[0].severity, MacroSeverity::Warning);
    }

    #[test]
    fn test_info_emits_diagnostic() {
        let mut ctx = MacroContext::new();
        ctx.info("just so you know", SourceLocation::unknown());
        assert_eq!(ctx.diagnostics.len(), 1);
        assert_eq!(ctx.diagnostics[0].severity, MacroSeverity::Info);
    }

    #[test]
    fn test_current_pos() {
        let mut ctx = MacroContext::new();
        let pos = SourceLocation::new(1, 10, 5, 100);
        ctx.set_call_position(pos);
        let result = ctx.current_pos();
        assert!(matches!(result, MacroValue::Position(p) if p == pos));
    }

    #[test]
    fn test_get_local_class() {
        let mut ctx = MacroContext::new();
        assert_eq!(ctx.get_local_class(), MacroValue::Null);

        ctx.current_class = Some("MyClass".to_string());
        // Phase 5.5: getLocalClass now returns a Ref<ClassType>-style
        // Object with a `name` field, instead of a bare String, so build
        // macros can do `Context.getLocalClass().get().name`.
        let cls = ctx.get_local_class();
        match cls {
            MacroValue::Object(obj) => {
                assert_eq!(
                    obj.get("name"),
                    Some(&MacroValue::String(Arc::from("MyClass")))
                );
            }
            other => panic!("expected Object, got {:?}", other),
        }
    }

    #[test]
    fn test_defined() {
        let mut ctx = MacroContext::new();
        ctx.defines.insert("debug".to_string(), "1".to_string());

        assert_eq!(ctx.defined("debug"), MacroValue::Bool(true));
        assert_eq!(ctx.defined("release"), MacroValue::Bool(false));
        assert_eq!(
            ctx.defined_value("debug"),
            MacroValue::String(Arc::from("1"))
        );
        assert_eq!(ctx.defined_value("release"), MacroValue::Null);
    }

    #[test]
    fn test_parse() {
        let ctx = MacroContext::new();
        let result = ctx.parse("1 + 2", SourceLocation::unknown());
        assert!(result.is_ok());
        let val = result.unwrap();
        assert!(matches!(val, MacroValue::Expr(_)));
    }

    #[test]
    fn test_make_expr() {
        let ctx = MacroContext::new();
        let result = ctx.make_expr(&MacroValue::Int(42), SourceLocation::unknown());
        assert!(result.is_ok());
        assert!(matches!(result.unwrap(), MacroValue::Expr(_)));
    }

    #[test]
    fn test_get_build_fields_no_context() {
        let ctx = MacroContext::new();
        let result = ctx.get_build_fields(SourceLocation::unknown());
        assert!(result.is_err());
    }

    #[test]
    fn test_get_build_fields_with_context() {
        let mut ctx = MacroContext::new();
        ctx.set_build_class(BuildClassContext {
            class_name: "Test".to_string(),
            qualified_name: "com.Test".to_string(),
            symbol_id: None,
            fields: vec![BuildField {
                name: "x".to_string(),
                kind: BuildFieldKind::Var {
                    type_hint: Some("Int".to_string()),
                    expr: None,
                },
                access: vec![FieldAccess::Public],
                pos: SourceLocation::unknown(),
                doc: None,
                meta: Vec::new(),
            }],
        });

        let result = ctx.get_build_fields(SourceLocation::unknown());
        assert!(result.is_ok());
        if let MacroValue::Array(fields) = result.unwrap() {
            assert_eq!(fields.len(), 1);
        } else {
            panic!("expected array");
        }
    }

    #[test]
    fn test_define_type() {
        let mut ctx = MacroContext::new();
        let td = DefinedType {
            pack: vec!["com".to_string(), "test".to_string()],
            name: "Generated".to_string(),
            kind: DefinedTypeKind::Class,
            fields: Vec::new(),
            pos: SourceLocation::unknown(),
        };
        let result = ctx.define_type(td, SourceLocation::unknown());
        assert!(result.is_ok());
        assert_eq!(ctx.defined_types.len(), 1);
    }

    #[test]
    fn test_define_type_duplicate() {
        let mut ctx = MacroContext::new();
        let td = DefinedType {
            pack: vec!["com".to_string()],
            name: "Foo".to_string(),
            kind: DefinedTypeKind::Class,
            fields: Vec::new(),
            pos: SourceLocation::unknown(),
        };
        ctx.define_type(td.clone(), SourceLocation::unknown())
            .unwrap();
        let result = ctx.define_type(td, SourceLocation::unknown());
        assert!(result.is_err());
    }

    #[test]
    fn test_dispatch_error() {
        let mut ctx = MacroContext::new();
        let result = ctx.dispatch(
            "error",
            &[MacroValue::String(Arc::from("test error"))],
            SourceLocation::unknown(),
        );
        assert!(result.is_err());
        assert_eq!(ctx.diagnostics.len(), 1);
    }

    #[test]
    fn test_dispatch_warning() {
        let mut ctx = MacroContext::new();
        let result = ctx.dispatch(
            "warning",
            &[MacroValue::String(Arc::from("test warning"))],
            SourceLocation::unknown(),
        );
        assert!(result.is_ok());
        assert_eq!(ctx.diagnostics.len(), 1);
    }

    #[test]
    fn test_dispatch_defined() {
        let mut ctx = MacroContext::new();
        ctx.defines.insert("debug".to_string(), "1".to_string());
        let result = ctx.dispatch(
            "defined",
            &[MacroValue::String(Arc::from("debug"))],
            SourceLocation::unknown(),
        );
        assert_eq!(result.unwrap(), MacroValue::Bool(true));
    }

    #[test]
    fn test_dispatch_unknown_method() {
        let mut ctx = MacroContext::new();
        let result = ctx.dispatch("nonexistent", &[], SourceLocation::unknown());
        assert!(result.is_err());
    }

    #[test]
    fn test_take_diagnostics() {
        let mut ctx = MacroContext::new();
        ctx.warning("a", SourceLocation::unknown());
        ctx.info("b", SourceLocation::unknown());
        let diags = ctx.take_diagnostics();
        assert_eq!(diags.len(), 2);
        assert!(ctx.diagnostics.is_empty());
    }

    #[test]
    fn test_get_pos_infos() {
        let ctx = MacroContext::new();
        let pos = SourceLocation::new(5, 10, 3, 42);
        let result = ctx.get_pos_infos(&pos);
        if let MacroValue::Object(obj) = result {
            assert_eq!(obj.get("file"), Some(&MacroValue::Int(5)));
            assert_eq!(obj.get("min"), Some(&MacroValue::Int(42)));
        } else {
            panic!("expected object");
        }
    }

    #[test]
    fn test_make_position() {
        let ctx = MacroContext::new();
        let mut info = BTreeMap::new();
        info.insert("file".to_string(), MacroValue::Int(3));
        info.insert("min".to_string(), MacroValue::Int(100));
        info.insert("max".to_string(), MacroValue::Int(200));
        let result = ctx.make_position(&MacroValue::Object(Arc::new(info)));
        assert!(result.is_ok());
        if let MacroValue::Position(pos) = result.unwrap() {
            assert_eq!(pos.file_id, 3);
            assert_eq!(pos.byte_offset, 100);
        } else {
            panic!("expected position");
        }
    }
}
