use crate::tast::{SourceLocation, TypeId};
use parser::{Expr, FunctionParam};
use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, Mutex};

/// A callable function in the macro interpreter
#[derive(Debug, Clone)]
pub struct MacroFunction {
    /// Function name (may be anonymous)
    pub name: String,
    /// Parameter names (for binding arguments)
    pub params: Vec<MacroParam>,
    /// The function body AST (interpreted at call time)
    pub body: Arc<Expr>,
    /// Shared lexical variables retained by the closure.
    pub captures: BTreeMap<String, super::environment::VariableBinding>,
    pub context: Option<MacroFunctionContext>,
    pub return_type: Option<parser::Type>,
    /// A static method reference forwards the caller's actual argument list.
    pub static_target: Option<(String, String)>,
}

#[derive(Debug, Clone)]
pub struct MacroFunctionContext {
    pub imports: Arc<BTreeMap<String, String>>,
    pub source_file: Option<String>,
    pub owner: Option<String>,
}

/// A parameter in a macro function definition
#[derive(Debug, Clone)]
pub struct MacroParam {
    /// Parameter name
    pub name: String,
    /// Whether this parameter is optional
    pub optional: bool,
    /// Whether this parameter is a rest parameter
    pub rest: bool,
    /// Default value expression (if optional)
    pub default_value: Option<Box<Expr>>,
    pub type_hint: Option<parser::Type>,
}

impl MacroParam {
    pub fn from_function_param(param: &FunctionParam) -> Self {
        Self {
            name: param.name.clone(),
            optional: param.optional,
            rest: param.rest,
            default_value: param.default_value.clone(),
            type_hint: param.type_hint.clone(),
        }
    }
}

/// Storage shared by every alias of a Haxe object, array or map.
///
/// Reads take a snapshot of the current contents; a write changes the
/// contents in place, so every alias observes it. A write copies the contents
/// only while a snapshot of them is still held.
pub struct Shared<T>(Arc<Mutex<Arc<T>>>);

impl<T> Shared<T> {
    pub fn new(value: T) -> Self {
        Self(Arc::new(Mutex::new(Arc::new(value))))
    }

    fn cell(&self) -> std::sync::MutexGuard<'_, Arc<T>> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// The contents at this moment.
    pub fn read(&self) -> Arc<T> {
        self.cell().clone()
    }

    pub fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    pub(crate) fn address(&self) -> usize {
        Arc::as_ptr(&self.0) as *const () as usize
    }
}

impl<T: Clone> Shared<T> {
    /// Change the contents in place. `change` must not read this container.
    pub fn write<R>(&self, change: impl FnOnce(&mut T) -> R) -> R {
        change(Arc::make_mut(&mut self.cell()))
    }

    /// Replace the contents.
    pub fn set(&self, value: T) {
        *self.cell() = Arc::new(value);
    }

    /// A new container holding the same contents.
    pub fn copy(&self) -> Self {
        Self(Arc::new(Mutex::new(self.read())))
    }
}

impl<T> Clone for Shared<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T: fmt::Debug> fmt::Debug for Shared<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.read().fmt(f)
    }
}

impl<T: PartialEq> PartialEq for Shared<T> {
    fn eq(&self, other: &Self) -> bool {
        self.ptr_eq(other) || *self.read() == *other.read()
    }
}

pub type ObjectFields = BTreeMap<String, MacroValue>;
pub type MapEntries = Vec<(MacroValue, MacroValue)>;

/// Runtime values used during macro interpretation.
///
/// These represent all the value types that can exist during
/// compile-time macro evaluation.
///
/// Objects, arrays and maps are shared mutable storage, as in Haxe. The other
/// composites are immutable and use Arc for O(1) clone.
#[derive(Debug, Clone)]
pub enum MacroValue {
    /// Null value
    Null,

    /// A macro-time abstract retains its declaration for conversions and operators.
    Abstract(Arc<str>, Arc<MacroValue>),

    /// Boolean value
    Bool(bool),

    /// Integer value (64-bit for macro evaluation precision)
    Int(i64),

    /// Float value
    Float(f64),

    /// String value (Arc for O(1) clone)
    String(Arc<str>),

    /// Array of values
    Array(Shared<Vec<MacroValue>>),

    /// Map entries retain typed keys, including object identity.
    Map(Shared<MapEntries>),

    /// Anonymous object / struct
    Object(Shared<ObjectFields>),

    /// Enum value: (enum_name, variant_name, args)
    Enum(Arc<str>, Arc<str>, Arc<Vec<MacroValue>>),

    /// A reified AST expression node, copied on write
    Expr(Arc<Expr>),

    /// A reference to a resolved type in the compiler
    Type(TypeId),

    /// A runtime class/type value, distinct from haxe.macro.Type descriptors.
    Class(Arc<str>),

    /// A callable function value (Arc for O(1) clone)
    Function(Arc<MacroFunction>),

    /// A source position value
    Position(SourceLocation),
}

/// Repoints aliases of an expression whose field was written: expressions are
/// copied on write, so the containers and variables holding the old copy are
/// updated to the new one. Each container and variable is visited once.
#[derive(Default)]
pub(crate) struct ReferenceUpdates {
    visited: std::collections::HashSet<usize>,
    // Holds every visited container so its address is not reused mid-walk.
    retained: Vec<MacroValue>,
}

impl ReferenceUpdates {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn replace_binding(
        &mut self,
        binding: &super::environment::VariableBinding,
        old: &MacroValue,
        new: &MacroValue,
    ) {
        if !self
            .visited
            .insert(Arc::as_ptr(binding) as *const () as usize)
        {
            return;
        }
        // Captures can refer back to their own function's binding.
        let mut value = binding
            .lock()
            .expect("macro variable lock poisoned")
            .clone();
        if value.replace_reference(old, new, self) {
            *binding.lock().expect("macro variable lock poisoned") = value;
        }
    }
}

#[inline]
fn scan_checkpoint(index: usize) {
    if index.is_multiple_of(128) {
        crate::cooperative::poll();
    }
}

impl MacroValue {
    /// Whether this value is, or carries inside an enum or abstract, a container
    /// that other references can see changes to.
    /// Haxe `==`: arrays, maps and objects by identity, everything else by value.
    pub fn haxe_equals(&self, other: &Self) -> bool {
        match (self.underlying(), other.underlying()) {
            (MacroValue::Array(a), MacroValue::Array(b)) => a.ptr_eq(b),
            (MacroValue::Map(a), MacroValue::Map(b)) => a.ptr_eq(b),
            (MacroValue::Object(a), MacroValue::Object(b)) => a.ptr_eq(b),
            (left, right) => left == right,
        }
    }

    pub fn holds_shared(&self) -> bool {
        match self {
            MacroValue::Array(_) | MacroValue::Map(_) | MacroValue::Object(_) => true,
            MacroValue::Enum(_, _, arguments) => arguments.iter().any(MacroValue::holds_shared),
            MacroValue::Abstract(_, inner) => inner.holds_shared(),
            _ => false,
        }
    }

    pub fn array(values: Vec<MacroValue>) -> Self {
        Self::Array(Shared::new(values))
    }

    pub fn object(fields: ObjectFields) -> Self {
        Self::Object(Shared::new(fields))
    }

    pub fn map(entries: MapEntries) -> Self {
        Self::Map(Shared::new(entries))
    }

    pub(crate) fn underlying(&self) -> &Self {
        let mut value = self;
        while let Self::Abstract(_, inner) = value {
            value = inner;
        }
        value
    }

    pub(crate) fn same_reference(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Object(a), Self::Object(b)) => a.ptr_eq(b),
            (Self::Array(a), Self::Array(b)) => a.ptr_eq(b),
            (Self::Map(a), Self::Map(b)) => a.ptr_eq(b),
            (Self::Expr(a), Self::Expr(b)) => Arc::ptr_eq(a, b),
            (Self::Function(a), Self::Function(b)) => Arc::ptr_eq(a, b),
            (Self::Abstract(owner, a), Self::Abstract(other, b)) => {
                owner == other && Arc::ptr_eq(a, b)
            }
            (Self::Enum(owner, tag, a), Self::Enum(other, variant, b)) => {
                owner == other && tag == variant && Arc::ptr_eq(a, b)
            }
            _ => false,
        }
    }

    pub(crate) fn map_key_eq(&self, other: &Self) -> bool {
        let (left, right) = (self.underlying(), other.underlying());
        left.same_reference(right)
            || match (left, right) {
                (
                    Self::Object(_)
                    | Self::Array(_)
                    | Self::Map(_)
                    | Self::Expr(_)
                    | Self::Function(_),
                    _,
                ) => false,
                _ => left == right,
            }
    }

    pub(crate) fn insert_map_entry(entries: &mut MapEntries, key: Self, value: Self) {
        if let Some((_, stored)) = entries
            .iter_mut()
            .find(|(stored, _)| stored.map_key_eq(&key))
        {
            *stored = value;
        } else {
            entries.push((key, value));
        }
    }

    pub(crate) fn map_get(entries: &[(Self, Self)], key: &Self) -> Self {
        entries
            .iter()
            .find(|(stored, _)| stored.map_key_eq(key))
            .map(|(_, value)| value.clone())
            .unwrap_or(Self::Null)
    }

    pub(crate) fn map_method(&self, method: &str, args: &[Self]) -> Option<Self> {
        let Self::Map(map) = self else {
            return None;
        };
        let entries = map.read();
        Some(match method {
            "get" => Self::map_get(&entries, args.first()?),
            "exists" => Self::Bool(
                entries
                    .iter()
                    .any(|(key, _)| key.map_key_eq(args.first().unwrap_or(&Self::Null))),
            ),
            "keys" => Self::array_iterator(Self::array(
                entries.iter().map(|(key, _)| key.clone()).collect(),
            )),
            "iterator" => Self::array_iterator(Self::array(
                entries.iter().map(|(_, value)| value.clone()).collect(),
            )),
            "keyValueIterator" => Self::array_iterator(Self::array(
                entries
                    .iter()
                    .map(|(key, value)| {
                        Self::object(BTreeMap::from([
                            ("key".to_string(), key.clone()),
                            ("value".to_string(), value.clone()),
                        ]))
                    })
                    .collect(),
            )),
            "copy" => Self::Map(map.copy()),
            "toString" => Self::from_string(self.to_display_string()),
            _ => return None,
        })
    }

    /// An iterator over `array`, which it reads as it advances.
    pub(crate) fn array_iterator(array: Self) -> Self {
        use parser::{BinaryOp, ExprKind, Span, UnaryOp};
        let expression = |kind| Expr {
            kind,
            span: Span::default(),
        };
        let array_ident = || expression(ExprKind::Ident("array".to_string()));
        let current = || expression(ExprKind::Ident("current".to_string()));
        let captures = BTreeMap::from([
            ("array".to_string(), Arc::new(Mutex::new(array))),
            ("current".to_string(), Arc::new(Mutex::new(Self::Int(0)))),
        ]);
        let method = |name: &str, body| {
            Self::Function(Arc::new(MacroFunction {
                name: format!("<iterator.{name}>"),
                params: Vec::new(),
                body: Arc::new(expression(body)),
                captures: captures.clone(),
                context: None,
                return_type: None,
                static_target: None,
            }))
        };
        Self::object(BTreeMap::from([
            (
                "__type__".to_string(),
                Self::from_str("haxe.iterators.ArrayIterator"),
            ),
            (
                "hasNext".to_string(),
                method(
                    "hasNext",
                    ExprKind::Binary {
                        left: Box::new(current()),
                        op: BinaryOp::Lt,
                        right: Box::new(expression(ExprKind::Field {
                            expr: Box::new(array_ident()),
                            field: "length".to_string(),
                            is_optional: false,
                        })),
                    },
                ),
            ),
            (
                "next".to_string(),
                method(
                    "next",
                    ExprKind::Index {
                        expr: Box::new(array_ident()),
                        index: Box::new(expression(ExprKind::Unary {
                            op: UnaryOp::PostIncr,
                            expr: Box::new(current()),
                        })),
                    },
                ),
            ),
        ]))
    }

    /// Mutate a map in place and return the Haxe method's result.
    pub(crate) fn mutate_map(&self, method: &str, args: &[Self]) -> Option<Self> {
        let Self::Map(entries) = self else {
            return None;
        };
        Some(match method {
            "set" => {
                let (key, value) = (args.first()?.clone(), args.get(1)?.clone());
                entries.write(|entries| Self::insert_map_entry(entries, key, value));
                Self::Null
            }
            "remove" => {
                let key = args.first()?;
                Self::Bool(entries.write(|entries| {
                    let index = entries
                        .iter()
                        .position(|(stored, _)| stored.map_key_eq(key));
                    index.map(|index| entries.remove(index)).is_some()
                }))
            }
            "clear" => {
                entries.write(|entries| entries.clear());
                Self::Null
            }
            _ => return None,
        })
    }

    /// Repoint aliases of the expression `old` at `new`. Containers are updated
    /// in place; returns whether `self` itself must be replaced.
    pub(crate) fn replace_reference(
        &mut self,
        old: &Self,
        new: &Self,
        updates: &mut ReferenceUpdates,
    ) -> bool {
        if self.same_reference(old) {
            *self = new.clone();
            return true;
        }
        match self {
            Self::Abstract(_, inner) => {
                let mut value = (**inner).clone();
                if value.replace_reference(old, new, updates) {
                    *inner = Arc::new(value);
                    return true;
                }
                false
            }
            Self::Function(function) => {
                for (i, binding) in function.captures.values().enumerate() {
                    scan_checkpoint(i);
                    updates.replace_binding(binding, old, new);
                }
                false
            }
            Self::Enum(_, _, values) => {
                let mut changed = false;
                let mut replacement = values.as_ref().clone();
                for (i, value) in replacement.iter_mut().enumerate() {
                    scan_checkpoint(i);
                    changed |= value.replace_reference(old, new, updates);
                }
                if changed {
                    *values = Arc::new(replacement);
                }
                changed
            }
            Self::Object(fields) => {
                if updates.visited.insert(fields.address()) {
                    updates.retained.push(Self::Object(fields.clone()));
                    let changes: Vec<_> = fields
                        .read()
                        .iter()
                        .enumerate()
                        .filter_map(|(i, (key, value))| {
                            scan_checkpoint(i);
                            let mut value = value.clone();
                            value
                                .replace_reference(old, new, updates)
                                .then(|| (key.clone(), value))
                        })
                        .collect();
                    if !changes.is_empty() {
                        fields.write(|fields| fields.extend(changes));
                    }
                }
                false
            }
            Self::Array(values) => {
                if updates.visited.insert(values.address()) {
                    updates.retained.push(Self::Array(values.clone()));
                    let changes: Vec<_> = values
                        .read()
                        .iter()
                        .enumerate()
                        .filter_map(|(i, value)| {
                            scan_checkpoint(i);
                            let mut value = value.clone();
                            value
                                .replace_reference(old, new, updates)
                                .then_some((i, value))
                        })
                        .collect();
                    if !changes.is_empty() {
                        values.write(|values| {
                            for (i, value) in changes {
                                values[i] = value;
                            }
                        });
                    }
                }
                false
            }
            Self::Map(entries) => {
                if updates.visited.insert(entries.address()) {
                    updates.retained.push(Self::Map(entries.clone()));
                    let mut replacement = entries.read().as_ref().clone();
                    let mut changed = false;
                    for (i, (key, value)) in replacement.iter_mut().enumerate() {
                        scan_checkpoint(i);
                        changed |= key.replace_reference(old, new, updates);
                        changed |= value.replace_reference(old, new, updates);
                    }
                    if changed {
                        entries.set(replacement);
                    }
                }
                false
            }
            _ => false,
        }
    }

    /// Write a field of an object, in place, or of a reified expression, which
    /// yields a new expression the caller propagates.
    pub(crate) fn set_field(&mut self, field: &str, value: Self) -> bool {
        match self {
            Self::Object(fields) => {
                fields.write(|fields| fields.insert(field.to_owned(), value));
                true
            }
            Self::Expr(expression) => match (field, value) {
                ("expr", value) => {
                    let Some(replacement) = super::expr_adt::try_expr_of(&value, expression.span)
                    else {
                        return false;
                    };
                    *Arc::make_mut(expression) = replacement;
                    true
                }
                ("pos", Self::Position(position)) => {
                    let expression = Arc::make_mut(expression);
                    let length = expression.span.end.saturating_sub(expression.span.start);
                    expression.span = parser::Span::new(
                        position.byte_offset as usize,
                        position.byte_offset as usize + length,
                    );
                    true
                }
                _ => false,
            },
            _ => false,
        }
    }

    /// Returns the type name of this value for error messages
    pub fn type_name(&self) -> &'static str {
        match self {
            MacroValue::Abstract(_, inner) => inner.type_name(),
            MacroValue::Null => "Null",
            MacroValue::Bool(_) => "Bool",
            MacroValue::Int(_) => "Int",
            MacroValue::Float(_) => "Float",
            MacroValue::String(_) => "String",
            MacroValue::Array(_) => "Array",
            MacroValue::Map(_) => "Map",
            MacroValue::Object(_) => "Object",
            MacroValue::Enum(_, _, _) => "Enum",
            MacroValue::Expr(_) => "Expr",
            MacroValue::Type(_) => "Type",
            MacroValue::Class(_) => "Class",
            MacroValue::Function(_) => "Function",
            MacroValue::Position(_) => "Position",
        }
    }

    /// Check if this value is truthy
    pub fn is_truthy(&self) -> bool {
        match self {
            MacroValue::Abstract(_, inner) => inner.is_truthy(),
            MacroValue::Null => false,
            MacroValue::Bool(b) => *b,
            MacroValue::Int(i) => *i != 0,
            MacroValue::Float(f) => *f != 0.0,
            MacroValue::String(s) => !s.is_empty(),
            MacroValue::Array(a) => !a.read().is_empty(),
            MacroValue::Map(_) => true,
            MacroValue::Object(_) => true,
            MacroValue::Enum(_, _, _) => true,
            MacroValue::Expr(_expr) => {
                // Try to unwrap the Expr to check truthiness of the underlying value
                use super::ast_bridge::unwrap_expr_value;
                let unwrapped = unwrap_expr_value(self);
                if !matches!(unwrapped, MacroValue::Expr(_)) {
                    unwrapped.is_truthy()
                } else {
                    true // Non-literal expressions are considered truthy
                }
            }
            MacroValue::Type(_) => true,
            MacroValue::Class(_) => true,
            MacroValue::Function(_) => true,
            MacroValue::Position(_) => true,
        }
    }

    /// Try to convert to an integer
    pub fn as_int(&self) -> Option<i64> {
        match self {
            MacroValue::Abstract(_, inner) => inner.as_int(),
            MacroValue::Int(i) => Some(*i),
            MacroValue::Float(f) => Some(*f as i64),
            MacroValue::Bool(b) => Some(if *b { 1 } else { 0 }),
            MacroValue::Expr(_) => {
                // Try to unwrap the Expr to extract an integer
                use super::ast_bridge::unwrap_expr_value;
                let unwrapped = unwrap_expr_value(self);
                if !matches!(unwrapped, MacroValue::Expr(_)) {
                    unwrapped.as_int()
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Try to convert to a float
    pub fn as_float(&self) -> Option<f64> {
        match self {
            MacroValue::Abstract(_, inner) => inner.as_float(),
            MacroValue::Float(f) => Some(*f),
            MacroValue::Int(i) => Some(*i as f64),
            MacroValue::Expr(_) => {
                use super::ast_bridge::unwrap_expr_value;
                let unwrapped = unwrap_expr_value(self);
                if !matches!(unwrapped, MacroValue::Expr(_)) {
                    unwrapped.as_float()
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Try to convert to a string
    pub fn as_string(&self) -> Option<&str> {
        match self {
            MacroValue::Abstract(_, inner) => inner.as_string(),
            MacroValue::String(s) => Some(s),
            _ => None,
        }
    }

    /// Try to convert to a boolean
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            MacroValue::Abstract(_, inner) => inner.as_bool(),
            MacroValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// The array's contents at this moment.
    pub fn as_array(&self) -> Option<Arc<Vec<MacroValue>>> {
        match self {
            MacroValue::Abstract(_, inner) => inner.as_array(),
            MacroValue::Array(a) => Some(a.read()),
            _ => None,
        }
    }

    /// The object's fields at this moment.
    pub fn as_object(&self) -> Option<Arc<ObjectFields>> {
        match self {
            MacroValue::Abstract(_, inner) => inner.as_object(),
            MacroValue::Object(o) => Some(o.read()),
            _ => None,
        }
    }

    /// Try to get as an Expr reference
    pub fn as_expr(&self) -> Option<&Expr> {
        match self {
            MacroValue::Expr(e) => Some(e),
            _ => None,
        }
    }

    /// Convert value to a display string (like Haxe's Std.string())
    pub fn to_display_string(&self) -> String {
        match self {
            MacroValue::Abstract(_, inner) => inner.to_display_string(),
            MacroValue::Null => "null".to_string(),
            MacroValue::Bool(b) => b.to_string(),
            MacroValue::Int(i) => i.to_string(),
            MacroValue::Float(f) => {
                if f.fract() == 0.0 {
                    format!("{:.1}", f)
                } else {
                    f.to_string()
                }
            }
            MacroValue::String(s) => s.to_string(),
            MacroValue::Array(items) => {
                let parts: Vec<String> =
                    items.read().iter().map(|v| v.to_display_string()).collect();
                format!("[{}]", parts.join(","))
            }
            MacroValue::Map(entries) => {
                let parts: Vec<_> = entries
                    .read()
                    .iter()
                    .map(|(key, value)| {
                        format!(
                            "{} => {}",
                            key.to_display_string(),
                            value.to_display_string()
                        )
                    })
                    .collect();
                format!("[{}]", parts.join(", "))
            }
            MacroValue::Object(fields) if matches!(fields.read().get("__type__"), Some(MacroValue::String(t)) if &**t == "Date") => {
                super::interpreter::host_std::date_string(
                    fields
                        .read()
                        .get("__time__")
                        .and_then(MacroValue::as_float)
                        .unwrap_or(0.0),
                )
            }
            // A caught compiler error prints as its message, as haxe.macro.Error does.
            MacroValue::Object(fields) if matches!(fields.read().get("__type__"), Some(MacroValue::String(t)) if &**t == "haxe.macro.Error") => {
                fields
                    .read()
                    .get("message")
                    .map(|m| m.to_display_string())
                    .unwrap_or_default()
            }
            MacroValue::Object(fields) => {
                let parts: Vec<String> = fields
                    .read()
                    .iter()
                    .map(|(k, v)| format!("{}: {}", k, v.to_display_string()))
                    .collect();
                format!("{{{}}}", parts.join(", "))
            }
            // `Ctor` or `Ctor(a,b)`, as Std.string spells an enum value.
            MacroValue::Enum(_, variant, args) => {
                if args.is_empty() {
                    variant.to_string()
                } else {
                    let parts: Vec<String> = args.iter().map(|v| v.to_display_string()).collect();
                    format!("{}({})", variant, parts.join(","))
                }
            }
            MacroValue::Expr(_) => {
                use super::ast_bridge::unwrap_expr_value;
                let unwrapped = unwrap_expr_value(self);
                if !matches!(unwrapped, MacroValue::Expr(_)) {
                    unwrapped.to_display_string()
                } else {
                    "<expr>".to_string()
                }
            }
            MacroValue::Type(_) => "<type>".to_string(),
            MacroValue::Class(name) => format!("Class<{name}>"),
            MacroValue::Function(f) => format!("<function:{}>", f.name),
            MacroValue::Position(loc) => super::context_api::named_position_string(loc)
                .unwrap_or_else(|| format!("{}:{}", loc.file_id, loc.line)),
        }
    }

    /// Convenience: create a String value from &str
    pub fn from_str(s: &str) -> Self {
        MacroValue::String(Arc::from(s))
    }

    /// Convenience: create a String value from String
    pub fn from_string(s: String) -> Self {
        MacroValue::String(Arc::from(s.as_str()))
    }
}

impl fmt::Display for MacroValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_display_string())
    }
}

impl PartialEq for MacroValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (MacroValue::Abstract(_, inner), other) => inner.as_ref() == other,
            (other, MacroValue::Abstract(_, inner)) => other == inner.as_ref(),

            (MacroValue::Class(a), MacroValue::Class(b)) => a == b,
            (MacroValue::Null, MacroValue::Null) => true,
            (MacroValue::Bool(a), MacroValue::Bool(b)) => a == b,
            (MacroValue::Int(a), MacroValue::Int(b)) => a == b,
            (MacroValue::Float(a), MacroValue::Float(b)) => a == b,
            (MacroValue::String(a), MacroValue::String(b)) => a == b,
            (MacroValue::Int(a), MacroValue::Float(b)) => (*a as f64) == *b,
            (MacroValue::Float(a), MacroValue::Int(b)) => *a == (*b as f64),
            (MacroValue::Array(a), MacroValue::Array(b)) => a == b,
            (MacroValue::Map(a), MacroValue::Map(b)) => a == b,
            (MacroValue::Object(a), MacroValue::Object(b)) => a == b,
            (MacroValue::Enum(n1, v1, a1), MacroValue::Enum(n2, v2, a2)) => {
                n1 == n2 && v1 == v2 && a1 == a2
            }
            _ => false,
        }
    }
}
