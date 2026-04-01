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
    /// Captured environment variables (for closures)
    pub captures: BTreeMap<String, MacroValue>,
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
}

impl MacroParam {
    pub fn from_function_param(param: &FunctionParam) -> Self {
        Self {
            name: param.name.clone(),
            optional: param.optional,
            rest: param.rest,
            default_value: param.default_value.clone(),
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

    /// Anonymous object / struct (Arc for O(1) clone, COW on mutation)
    Object(Arc<BTreeMap<String, MacroValue>>),

    /// Enum value: (enum_name, variant_name, args)
    Enum(Arc<str>, Arc<str>, Arc<Vec<MacroValue>>),

    /// A reified AST expression node (Arc for O(1) clone)
    Expr(Arc<Expr>),

    /// A reference to a resolved type in the compiler
    Type(TypeId),

    /// A callable function value (Arc for O(1) clone)
    Function(Arc<MacroFunction>),

    /// A source position value
    Position(SourceLocation),
}

impl MacroValue {
    /// Returns the type name of this value for error messages
    pub fn type_name(&self) -> &'static str {
        match self {
            MacroValue::Null => "Null",
            MacroValue::Bool(_) => "Bool",
            MacroValue::Int(_) => "Int",
            MacroValue::Float(_) => "Float",
            MacroValue::String(_) => "String",
            MacroValue::Array(_) => "Array",
            MacroValue::Object(_) => "Object",
            MacroValue::Enum(_, _, _) => "Enum",
            MacroValue::Expr(_) => "Expr",
            MacroValue::Type(_) => "Type",
            MacroValue::Function(_) => "Function",
            MacroValue::Position(_) => "Position",
        }
    }

    /// Check if this value is truthy
    pub fn is_truthy(&self) -> bool {
        match self {
            MacroValue::Null => false,
            MacroValue::Bool(b) => *b,
            MacroValue::Int(i) => *i != 0,
            MacroValue::Float(f) => *f != 0.0,
            MacroValue::String(s) => !s.is_empty(),
            MacroValue::Array(a) => !a.is_empty(),
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
            MacroValue::Function(_) => true,
            MacroValue::Position(_) => true,
        }
    }

    /// Try to convert to an integer
    pub fn as_int(&self) -> Option<i64> {
        match self {
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
            MacroValue::String(s) => Some(s),
            _ => None,
        }
    }

    /// Try to convert to a boolean
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            MacroValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Try to get as an array reference
    pub fn as_array(&self) -> Option<&Vec<MacroValue>> {
        match self {
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
            MacroValue::Object(fields) => {
                let parts: Vec<String> = fields
                    .iter()
                    .map(|(k, v)| format!("{}: {}", k, v.to_display_string()))
                    .collect();
                format!("{{{}}}", parts.join(", "))
            }
            MacroValue::Enum(enum_name, variant, args) => {
                if args.is_empty() {
                    format!("{}.{}", enum_name, variant)
                } else {
                    let parts: Vec<String> = args.iter().map(|v| v.to_display_string()).collect();
                    format!("{}.{}({})", enum_name, variant, parts.join(", "))
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
            MacroValue::Function(f) => format!("<function:{}>", f.name),
            MacroValue::Position(loc) => {
                format!("{}:{}", loc.file_id, loc.line)
            }
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
