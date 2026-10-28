use crate::tast::{SourceLocation, TypeId};
use parser::{Expr, FunctionParam};
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

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
}

#[derive(Debug, Clone)]
pub struct MacroFunctionContext {
    pub imports: BTreeMap<String, String>,
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

/// Runtime values used during macro interpretation.
///
/// These represent all the value types that can exist during
/// compile-time macro evaluation.
///
/// Performance: String, Array, Object, Expr, and Function use Arc<> for
/// O(1) clone via reference counting. Only mutating operations (array push,
/// object field set) trigger COW copies via Arc::make_mut().
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

    /// Array of values (Arc for O(1) clone, COW on mutation)
    Array(Arc<Vec<MacroValue>>),

    /// Map entries retain typed keys, including object identity.
    Map(Arc<Vec<(MacroValue, MacroValue)>>),

    /// Anonymous object / struct (Arc for O(1) clone, COW on mutation)
    Object(Arc<BTreeMap<String, MacroValue>>),

    /// Enum value: (enum_name, variant_name, args)
    Enum(Arc<str>, Arc<str>, Arc<Vec<MacroValue>>),

    /// A reified AST expression node (Arc for O(1) clone)
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

enum ReferenceOwner {
    Object(std::sync::Weak<BTreeMap<String, MacroValue>>),
    Values(std::sync::Weak<Vec<MacroValue>>),
    Map(std::sync::Weak<Vec<(MacroValue, MacroValue)>>),
    Expr(std::sync::Weak<Expr>),
    Function(std::sync::Weak<MacroFunction>),
    Abstract(std::sync::Weak<MacroValue>),
}

impl ReferenceOwner {
    fn alive(&self) -> bool {
        match self {
            Self::Object(value) => value.strong_count() != 0,
            Self::Values(value) => value.strong_count() != 0,
            Self::Map(value) => value.strong_count() != 0,
            Self::Expr(value) => value.strong_count() != 0,
            Self::Function(value) => value.strong_count() != 0,
            Self::Abstract(value) => value.strong_count() != 0,
        }
    }
}

struct ReferenceSummary {
    owner: ReferenceOwner,
    serial: u64,
    maximum: u64,
}

/// Closed backing stores cache the greatest identity they contain. Captured
/// variable cells remain unbounded because they can acquire later values.
#[derive(Default)]
pub(crate) struct ReferenceIndex {
    values: std::collections::HashMap<(u8, usize), ReferenceSummary>,
    serial: u64,
    mutations: u64,
    #[cfg(test)]
    summarized: usize,
}

impl ReferenceIndex {
    fn summarize(&mut self, value: &MacroValue) -> (u64, u64) {
        let Some(key) = value.reference_key() else {
            return (0, 0);
        };
        if let Some(summary) = self.values.get(&key) {
            return (summary.serial, summary.maximum);
        }
        let owner = match value {
            MacroValue::Object(value) => ReferenceOwner::Object(Arc::downgrade(value)),
            MacroValue::Array(value) | MacroValue::Enum(_, _, value) => {
                ReferenceOwner::Values(Arc::downgrade(value))
            }
            MacroValue::Map(value) => ReferenceOwner::Map(Arc::downgrade(value)),
            MacroValue::Expr(value) => ReferenceOwner::Expr(Arc::downgrade(value)),
            MacroValue::Function(value) => ReferenceOwner::Function(Arc::downgrade(value)),
            MacroValue::Abstract(_, value) => ReferenceOwner::Abstract(Arc::downgrade(value)),
            _ => unreachable!("reference identity"),
        };
        self.serial = self.serial.saturating_add(1);
        let serial = self.serial;
        // Weak ownership prevents address reuse and makes a cached store's next
        // COW write acquire a new identity, even with one strong owner.
        self.values.insert(
            key,
            ReferenceSummary {
                owner,
                serial,
                maximum: u64::MAX,
            },
        );
        #[cfg(test)]
        {
            self.summarized += 1;
        }
        let mut maximum = serial;
        match value {
            MacroValue::Object(values) => {
                for value in values.values() {
                    maximum = maximum.max(self.summarize(value).1);
                }
            }
            MacroValue::Array(values) | MacroValue::Enum(_, _, values) => {
                for value in values.iter() {
                    maximum = maximum.max(self.summarize(value).1);
                }
            }
            MacroValue::Map(values) => {
                for (key, value) in values.iter() {
                    maximum = maximum
                        .max(self.summarize(key).1)
                        .max(self.summarize(value).1);
                }
            }
            MacroValue::Abstract(_, value) => maximum = maximum.max(self.summarize(value).1),
            MacroValue::Function(function) if !function.captures.is_empty() => maximum = u64::MAX,
            _ => {}
        }
        self.values.get_mut(&key).unwrap().maximum = maximum;
        (serial, maximum)
    }

    fn target(&mut self, old: &MacroValue) -> u64 {
        self.mutations += 1;
        if self.values.len() > 4096 && self.mutations.is_multiple_of(128) {
            self.values.retain(|_, summary| summary.owner.alive());
        }
        self.summarize(old).0
    }
}

#[derive(Default)]
pub(crate) struct ReferenceUpdates<'a> {
    values: BTreeMap<(u8, usize), (MacroValue, MacroValue, bool)>,
    bindings: std::collections::BTreeSet<usize>,
    index: Option<&'a mut ReferenceIndex>,
    target: u64,
}

impl<'a> ReferenceUpdates<'a> {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn indexed(old: &MacroValue, index: &'a mut ReferenceIndex) -> Self {
        let target = index.target(old);
        Self {
            index: Some(index),
            target,
            ..Default::default()
        }
    }

    fn excludes(&mut self, value: &MacroValue) -> bool {
        self.index
            .as_deref_mut()
            .is_some_and(|index| index.summarize(value).1 < self.target)
    }

    pub(crate) fn replace_binding(
        &mut self,
        binding: &super::environment::VariableBinding,
        old: &MacroValue,
        new: &MacroValue,
    ) {
        if !self.bindings.insert(Arc::as_ptr(binding) as usize) {
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

impl MacroValue {
    pub(crate) fn underlying(&self) -> &Self {
        let mut value = self;
        while let Self::Abstract(_, inner) = value {
            value = inner;
        }
        value
    }

    pub(crate) fn reference_key(&self) -> Option<(u8, usize)> {
        Some(match self {
            Self::Object(value) => (0, Arc::as_ptr(value) as usize),
            Self::Array(value) => (1, Arc::as_ptr(value) as usize),
            Self::Enum(_, _, value) => (2, Arc::as_ptr(value) as usize),
            Self::Map(value) => (3, Arc::as_ptr(value) as usize),
            Self::Expr(value) => (4, Arc::as_ptr(value) as usize),
            Self::Function(value) => (5, Arc::as_ptr(value) as usize),
            Self::Abstract(_, value) => (6, Arc::as_ptr(value) as usize),
            _ => return None,
        })
    }

    pub(crate) fn same_reference(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Object(a), Self::Object(b)) => Arc::ptr_eq(a, b),
            (Self::Array(a), Self::Array(b)) => Arc::ptr_eq(a, b),
            (Self::Map(a), Self::Map(b)) => Arc::ptr_eq(a, b),
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

    pub(crate) fn insert_map_entry(entries: &mut Vec<(Self, Self)>, key: Self, value: Self) {
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
        let Self::Map(entries) = self else {
            return None;
        };
        Some(match method {
            "get" => Self::map_get(entries, args.first()?),
            "exists" => Self::Bool(
                entries
                    .iter()
                    .any(|(key, _)| key.map_key_eq(args.first().unwrap_or(&Self::Null))),
            ),
            "keys" => Self::array_iterator(Arc::new(
                entries.iter().map(|(key, _)| key.clone()).collect(),
            )),
            "iterator" => Self::array_iterator(Arc::new(
                entries.iter().map(|(_, value)| value.clone()).collect(),
            )),
            "keyValueIterator" => Self::array_iterator(Arc::new(
                entries
                    .iter()
                    .map(|(key, value)| {
                        Self::Object(Arc::new(BTreeMap::from([
                            ("key".to_string(), key.clone()),
                            ("value".to_string(), value.clone()),
                        ])))
                    })
                    .collect(),
            )),
            "copy" => Self::Map(Arc::new(entries.as_ref().clone())),
            "toString" => Self::from_string(self.to_display_string()),
            _ => return None,
        })
    }

    pub(crate) fn array_iterator(values: Arc<Vec<Self>>) -> Self {
        use parser::{BinaryOp, ExprKind, Span, UnaryOp};
        let expression = |kind| Expr {
            kind,
            span: Span::default(),
        };
        let array = || expression(ExprKind::Ident("array".to_string()));
        let current = || expression(ExprKind::Ident("current".to_string()));
        let captures = BTreeMap::from([
            (
                "array".to_string(),
                Arc::new(std::sync::Mutex::new(Self::Array(values))),
            ),
            (
                "current".to_string(),
                Arc::new(std::sync::Mutex::new(Self::Int(0))),
            ),
        ]);
        let method = |name: &str, body| {
            Self::Function(Arc::new(MacroFunction {
                name: format!("<iterator.{name}>"),
                params: Vec::new(),
                body: Arc::new(expression(body)),
                captures: captures.clone(),
                context: None,
                return_type: None,
            }))
        };
        Self::Object(Arc::new(BTreeMap::from([
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
                            expr: Box::new(array()),
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
                        expr: Box::new(array()),
                        index: Box::new(expression(ExprKind::Unary {
                            op: UnaryOp::PostIncr,
                            expr: Box::new(current()),
                        })),
                    },
                ),
            ),
        ])))
    }

    /// Mutate a map and return the Haxe method's result separately from its storage.
    pub(crate) fn mutate_map(&mut self, method: &str, args: &[Self]) -> Option<Self> {
        let Self::Map(entries) = self else {
            return None;
        };
        Some(match method {
            "set" => {
                let (key, value) = (args.first()?, args.get(1)?);
                Self::insert_map_entry(Arc::make_mut(entries), key.clone(), value.clone());
                Self::Null
            }
            "remove" => {
                let key = args.first()?;
                let index = entries
                    .iter()
                    .position(|(stored, _)| stored.map_key_eq(key));
                if let Some(index) = index {
                    Arc::make_mut(entries).remove(index);
                }
                Self::Bool(index.is_some())
            }
            "clear" => {
                Arc::make_mut(entries).clear();
                Self::Null
            }
            _ => return None,
        })
    }

    /// Preserve shared value identity when a COW mutation replaces its storage.
    pub(crate) fn replace_reference(
        &mut self,
        old: &Self,
        new: &Self,
        updates: &mut ReferenceUpdates<'_>,
    ) -> bool {
        let same = self.same_reference(old);
        if same {
            *self = new.clone();
            return true;
        }
        if updates.excludes(self) {
            return false;
        }
        if let Self::Abstract(_, inner) = self {
            let mut value = (**inner).clone();
            if value.replace_reference(old, new, updates) {
                *inner = Arc::new(value);
                return true;
            }
            return false;
        }
        if let Self::Function(function) = self {
            for binding in function.captures.values() {
                updates.replace_binding(binding, old, new);
            }
            return false;
        }
        let key = match self {
            Self::Object(value) => (0, Arc::as_ptr(value) as usize),
            Self::Array(value) => (1, Arc::as_ptr(value) as usize),
            Self::Enum(_, _, value) => (2, Arc::as_ptr(value) as usize),
            Self::Map(value) => (3, Arc::as_ptr(value) as usize),
            _ => return false,
        };
        if let Some((_, value, changed)) = updates.values.get(&key) {
            if *changed {
                *self = value.clone();
            }
            return *changed;
        }
        // Retain the original allocation so its address cannot be reused while
        // the memo is shared across bindings. Parents retain their aliases too.
        let original = self.clone();
        let changed = match self {
            Self::Object(fields) => {
                let changes: Vec<_> = fields
                    .iter()
                    .filter_map(|(key, value)| {
                        let mut value = value.clone();
                        value
                            .replace_reference(old, new, updates)
                            .then(|| (key.clone(), value))
                    })
                    .collect();
                let changed = !changes.is_empty();
                if changed {
                    Arc::make_mut(fields).extend(changes);
                }
                changed
            }
            Self::Array(values) | Self::Enum(_, _, values) => {
                let changes: Vec<_> = values
                    .iter()
                    .enumerate()
                    .filter_map(|(i, value)| {
                        let mut value = value.clone();
                        value
                            .replace_reference(old, new, updates)
                            .then_some((i, value))
                    })
                    .collect();
                let changed = !changes.is_empty();
                for (i, value) in changes {
                    Arc::make_mut(values)[i] = value;
                }
                changed
            }
            Self::Map(entries) => {
                let mut replacement = entries.as_ref().clone();
                let mut changed = false;
                for (key, value) in &mut replacement {
                    changed |= key.replace_reference(old, new, updates);
                    changed |= value.replace_reference(old, new, updates);
                }
                if changed {
                    *entries = Arc::new(replacement);
                }
                changed
            }
            _ => false,
        };
        updates
            .values
            .insert(key, (original, self.clone(), changed));
        changed
    }

    /// Mutate the fields presented by objects and reified expressions.
    pub(crate) fn set_field(&mut self, field: &str, value: Self) -> bool {
        match self {
            Self::Object(fields) => {
                Arc::make_mut(fields).insert(field.to_owned(), value);
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
            MacroValue::Array(a) => !a.is_empty(),
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

    /// Try to get as an array reference
    pub fn as_array(&self) -> Option<&Vec<MacroValue>> {
        match self {
            MacroValue::Abstract(_, inner) => inner.as_array(),
            MacroValue::Array(a) => Some(a),
            _ => None,
        }
    }

    /// Try to get as a mutable array reference (COW via Arc::make_mut)
    pub fn as_array_mut(&mut self) -> Option<&mut Vec<MacroValue>> {
        match self {
            MacroValue::Array(a) => Some(Arc::make_mut(a)),
            _ => None,
        }
    }

    /// Try to get as an object reference
    pub fn as_object(&self) -> Option<&BTreeMap<String, MacroValue>> {
        match self {
            MacroValue::Abstract(_, inner) => inner.as_object(),
            MacroValue::Object(o) => Some(o),
            _ => None,
        }
    }

    /// Try to get as a mutable object reference (COW via Arc::make_mut)
    pub fn as_object_mut(&mut self) -> Option<&mut BTreeMap<String, MacroValue>> {
        match self {
            MacroValue::Object(o) => Some(Arc::make_mut(o)),
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
                let parts: Vec<String> = items.iter().map(|v| v.to_display_string()).collect();
                format!("[{}]", parts.join(","))
            }
            MacroValue::Map(entries) => {
                let parts: Vec<_> = entries
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
            // A caught compiler error prints as its message, as haxe.macro.Error does.
            MacroValue::Object(fields) if matches!(fields.get("__type__"), Some(MacroValue::String(t)) if &**t == "haxe.macro.Error") => {
                fields
                    .get("message")
                    .map(|m| m.to_display_string())
                    .unwrap_or_default()
            }
            MacroValue::Object(fields) => {
                let parts: Vec<String> = fields
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
            (MacroValue::Enum(n1, v1, a1), MacroValue::Enum(n2, v2, a2)) => {
                n1 == n2 && v1 == v2 && a1 == a2
            }
            _ => false,
        }
    }
}

#[cfg(test)]
mod reference_index_tests {
    use super::*;
    use std::sync::Mutex;

    fn object(value: i64) -> MacroValue {
        MacroValue::Object(Arc::new(BTreeMap::from([(
            "value".into(),
            MacroValue::Int(value),
        )])))
    }

    #[test]
    fn unrelated_catalogue_is_not_walked_again_for_local_mutations() {
        let mut catalogue = MacroValue::Array(Arc::new((0..1000).map(object).collect()));
        let mut index = ReferenceIndex::default();
        index.summarize(&catalogue);
        let initial = index.summarized;
        for value in 0..40 {
            let old = object(value);
            let new = object(value + 1);
            let mut updates = ReferenceUpdates::indexed(&old, &mut index);
            assert!(!catalogue.replace_reference(&old, &new, &mut updates));
            assert!(updates.values.is_empty());
        }
        assert_eq!(index.summarized, initial + 40);
    }

    #[test]
    fn captured_cells_can_acquire_references_after_a_summary() {
        let binding = Arc::new(Mutex::new(MacroValue::Null));
        let function = MacroFunction {
            name: "capture".into(),
            params: Vec::new(),
            body: Arc::new(Expr {
                kind: parser::ExprKind::Null,
                span: parser::Span::new(0, 0),
            }),
            captures: BTreeMap::from([("cell".into(), binding.clone())]),
            context: None,
            return_type: None,
        };
        let mut parent =
            MacroValue::Array(Arc::new(vec![MacroValue::Function(Arc::new(function))]));
        let mut index = ReferenceIndex::default();
        index.summarize(&parent);
        let old = object(1);
        let new = object(2);
        *binding.lock().unwrap() = old.clone();
        parent.replace_reference(&old, &new, &mut ReferenceUpdates::indexed(&old, &mut index));
        assert!(binding.lock().unwrap().same_reference(&new));
    }

    #[test]
    fn a_cow_write_invalidates_a_cached_closed_store() {
        let mut parent = MacroValue::Array(Arc::new(vec![MacroValue::Int(0)]));
        let mut index = ReferenceIndex::default();
        index.summarize(&parent);
        let old = object(1);
        let new = object(2);
        let MacroValue::Array(values) = &mut parent else {
            unreachable!()
        };
        Arc::make_mut(values).push(old.clone());
        assert!(parent.replace_reference(
            &old,
            &new,
            &mut ReferenceUpdates::indexed(&old, &mut index)
        ));
        let MacroValue::Array(values) = parent else {
            unreachable!()
        };
        assert!(values[1].same_reference(&new));
    }

    #[test]
    fn dead_stores_do_not_accumulate_in_the_index() {
        let mut index = ReferenceIndex::default();
        for value in 0..10_000 {
            let target = object(value);
            index.target(&target);
        }
        assert!(index.values.len() < 4096 + 128);
    }
}
