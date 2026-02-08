//! IR Type System
//!
//! Defines the type system for the intermediate representation.
//! IR types are lower-level than TAST types and map more directly to runtime representations.

use super::IrId;
use serde::{Deserialize, Serialize};
use std::fmt;

/// IR type representation
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IrType {
    /// Void type (no value)
    Void,

    /// Boolean type
    Bool,

    /// Integer types
    I8,
    I16,
    I32,
    I64,

    /// Unsigned integer types
    U8,
    U16,
    U32,
    U64,

    /// Floating point types
    F32,
    F64,

    /// Pointer type
    Ptr(Box<IrType>),

    /// Reference type (managed pointer)
    Ref(Box<IrType>),

    /// Array type with known size
    Array(Box<IrType>, usize),

    /// Dynamic array (slice) type
    Slice(Box<IrType>),

    /// SIMD vector type (fixed-size, homogeneous)
    /// Used for auto-vectorization and explicit SIMD operations
    Vector {
        /// Element type (must be numeric: I8-I64, U8-U64, F32, F64)
        element: Box<IrType>,
        /// Number of elements (typically 2, 4, 8, or 16)
        count: usize,
    },

    /// String type (UTF-8)
    String,

    /// Function type
    Function {
        params: Vec<IrType>,
        return_type: Box<IrType>,
        varargs: bool,
    },

    /// Structure type
    Struct {
        name: String,
        fields: Vec<StructField>,
    },

    /// Union type (sum type)
    Union {
        name: String,
        variants: Vec<UnionVariant>,
    },

    /// Opaque type (for external types)
    Opaque {
        name: String,
        size: usize,
        align: usize,
    },

    /// Type variable / type parameter (e.g., "T" in Container<T>)
    /// This represents a generic type parameter before monomorphization.
    /// Also aliased as TypeParam for clarity in generic contexts.
    TypeVar(String),

    /// Generic type instantiation (e.g., Container<Int>, Array<String>)
    /// Used to represent a concrete instantiation of a generic type.
    Generic {
        /// The base generic type (e.g., the IrType for Container)
        base: Box<IrType>,
        /// The concrete type arguments (e.g., [Int] for Container<Int>)
        type_args: Vec<IrType>,
    },

    /// Any type (dynamic)
    Any,
}

impl IrType {
    /// Create a new type parameter (alias for TypeVar)
    pub fn type_param(name: impl Into<String>) -> Self {
        IrType::TypeVar(name.into())
    }

    /// Create a new generic instantiation
    pub fn generic(base: IrType, type_args: Vec<IrType>) -> Self {
        IrType::Generic {
            base: Box::new(base),
            type_args,
        }
    }

    /// Check if this is a type parameter
    pub fn is_type_param(&self) -> bool {
        matches!(self, IrType::TypeVar(_))
    }

    /// Check if this is a generic instantiation
    pub fn is_generic_instance(&self) -> bool {
        matches!(self, IrType::Generic { .. })
    }

    /// Get the type parameter name if this is a TypeVar
    pub fn type_param_name(&self) -> Option<&str> {
        match self {
            IrType::TypeVar(name) => Some(name),
            _ => None,
        }
    }

    /// Get the base type and type args if this is a Generic
    pub fn generic_parts(&self) -> Option<(&IrType, &[IrType])> {
        match self {
            IrType::Generic { base, type_args } => Some((base, type_args)),
            _ => None,
        }
    }
}

/// Structure field
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StructField {
    pub name: String,
    pub ty: IrType,
    pub offset: usize,
}

/// Union variant
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct UnionVariant {
    pub name: String,
    pub tag: u32,
    pub fields: Vec<IrType>,
}

impl IrType {
    /// Get the size of the type in bytes
    pub fn size(&self) -> usize {
        match self {
            IrType::Void => 0,
            IrType::Bool | IrType::I8 | IrType::U8 => 1,
            IrType::I16 | IrType::U16 => 2,
            IrType::I32 | IrType::U32 | IrType::F32 => 4,
            IrType::I64 | IrType::U64 | IrType::F64 => 8,
            IrType::Ptr(_) | IrType::Ref(_) => std::mem::size_of::<usize>(),
            IrType::Array(elem_ty, count) => elem_ty.size() * count,
            IrType::Slice(_) => std::mem::size_of::<usize>() * 2, // ptr + len
            IrType::String => std::mem::size_of::<usize>() * 3,   // ptr + len + capacity
            IrType::Function { .. } => std::mem::size_of::<usize>(), // function pointer
            IrType::Struct { fields, .. } => fields.iter().map(|f| f.ty.size()).sum(),
            IrType::Union { variants, .. } => {
                // Tag size + largest variant
                let tag_size = 4;
                let max_variant_size = variants
                    .iter()
                    .map(|v| v.fields.iter().map(|f| f.size()).sum::<usize>())
                    .max()
                    .unwrap_or(0);
                tag_size + max_variant_size
            }
            IrType::Opaque { size, .. } => *size,
            IrType::Vector { element, count } => element.size() * count,
            IrType::TypeVar(_) => 8, // Safety net: pointer-sized if TypeVar leaks through
            IrType::Generic { .. } => {
                panic!("Cannot get size of generic type before monomorphization")
            }
            IrType::Any => std::mem::size_of::<usize>() * 2, // type_id + value_ptr
        }
    }

    /// Get the alignment requirement of the type
    pub fn align(&self) -> usize {
        match self {
            IrType::Void => 1,
            IrType::Bool | IrType::I8 | IrType::U8 => 1,
            IrType::I16 | IrType::U16 => 2,
            IrType::I32 | IrType::U32 | IrType::F32 => 4,
            IrType::I64 | IrType::U64 | IrType::F64 => 8,
            IrType::Ptr(_) | IrType::Ref(_) => std::mem::align_of::<usize>(),
            IrType::Array(elem_ty, _) => elem_ty.align(),
            IrType::Slice(_) | IrType::String | IrType::Any => std::mem::align_of::<usize>(),
            IrType::Function { .. } => std::mem::align_of::<usize>(),
            IrType::Struct { fields, .. } => fields.iter().map(|f| f.ty.align()).max().unwrap_or(1),
            IrType::Union { .. } => 4, // Assume 4-byte alignment for tag
            IrType::Opaque { align, .. } => *align,
            // SIMD vectors require alignment equal to their size for optimal performance
            IrType::Vector { element, count } => (element.size() * count).max(element.align()),
            IrType::TypeVar(_) => 8, // Safety net: pointer-aligned if TypeVar leaks through
            IrType::Generic { .. } => {
                panic!("Cannot get alignment of generic type before monomorphization")
            }
        }
    }

    /// Check if this is a primitive type
    pub fn is_primitive(&self) -> bool {
        matches!(
            self,
            IrType::Void
                | IrType::Bool
                | IrType::I8
                | IrType::I16
                | IrType::I32
                | IrType::I64
                | IrType::U8
                | IrType::U16
                | IrType::U32
                | IrType::U64
                | IrType::F32
                | IrType::F64
        )
    }

    /// Check if this is a pointer type
    pub fn is_pointer(&self) -> bool {
        matches!(self, IrType::Ptr(_) | IrType::Ref(_))
    }

    /// Check if this is an integer type
    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            IrType::I8
                | IrType::I16
                | IrType::I32
                | IrType::I64
                | IrType::U8
                | IrType::U16
                | IrType::U32
                | IrType::U64
        )
    }

    /// Check if this is a floating point type
    pub fn is_float(&self) -> bool {
        matches!(self, IrType::F32 | IrType::F64)
    }

    /// Check if this is a signed integer type
    pub fn is_signed_integer(&self) -> bool {
        matches!(self, IrType::I8 | IrType::I16 | IrType::I32 | IrType::I64)
    }

    /// Check if this is a SIMD vector type
    pub fn is_vector(&self) -> bool {
        matches!(self, IrType::Vector { .. })
    }

    /// Create a SIMD vector type from element type and count
    pub fn vector(element: IrType, count: usize) -> Self {
        IrType::Vector {
            element: Box::new(element),
            count,
        }
    }

    /// Get the element type if this is a vector
    pub fn vector_element(&self) -> Option<&IrType> {
        match self {
            IrType::Vector { element, .. } => Some(element),
            _ => None,
        }
    }

    /// Get the element count if this is a vector
    pub fn vector_count(&self) -> Option<usize> {
        match self {
            IrType::Vector { count, .. } => Some(*count),
            _ => None,
        }
    }

    /// Check if this type can be vectorized (is a numeric scalar)
    pub fn is_vectorizable(&self) -> bool {
        matches!(
            self,
            IrType::I8
                | IrType::I16
                | IrType::I32
                | IrType::I64
                | IrType::U8
                | IrType::U16
                | IrType::U32
                | IrType::U64
                | IrType::F32
                | IrType::F64
        )
    }

    /// Get the default value for this type
    pub fn default_value(&self) -> IrValue {
        match self {
            IrType::Void => IrValue::Void,
            IrType::Bool => IrValue::Bool(false),
            IrType::I8 => IrValue::I8(0),
            IrType::I16 => IrValue::I16(0),
            IrType::I32 => IrValue::I32(0),
            IrType::I64 => IrValue::I64(0),
            IrType::U8 => IrValue::U8(0),
            IrType::U16 => IrValue::U16(0),
            IrType::U32 => IrValue::U32(0),
            IrType::U64 => IrValue::U64(0),
            IrType::F32 => IrValue::F32(0.0),
            IrType::F64 => IrValue::F64(0.0),
            IrType::Ptr(_) | IrType::Ref(_) => IrValue::Null,
            IrType::String => IrValue::String(String::new()),
            _ => IrValue::Undef,
        }
    }
}

/// IR constant value
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum IrValue {
    /// No value
    Void,
    /// Undefined value
    Undef,
    /// Null pointer
    Null,
    /// Boolean value
    Bool(bool),
    /// Integer values
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    /// Floating point values
    F32(f32),
    F64(f64),
    /// String value
    String(String),
    /// Array value
    Array(Vec<IrValue>),
    /// Struct value
    Struct(Vec<IrValue>),
    /// Function pointer (reference to a function by ID)
    Function(super::IrFunctionId),
    /// Closure value (function pointer + environment)
    Closure {
        function: super::IrFunctionId,
        environment: Box<IrValue>, // Struct containing captured variables
    },
}

impl fmt::Display for IrType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IrType::Void => write!(f, "void"),
            IrType::Bool => write!(f, "bool"),
            IrType::I8 => write!(f, "i8"),
            IrType::I16 => write!(f, "i16"),
            IrType::I32 => write!(f, "i32"),
            IrType::I64 => write!(f, "i64"),
            IrType::U8 => write!(f, "u8"),
            IrType::U16 => write!(f, "u16"),
            IrType::U32 => write!(f, "u32"),
            IrType::U64 => write!(f, "u64"),
            IrType::F32 => write!(f, "f32"),
            IrType::F64 => write!(f, "f64"),
            IrType::Ptr(ty) => write!(f, "*{}", ty),
            IrType::Ref(ty) => write!(f, "&{}", ty),
            IrType::Array(ty, size) => write!(f, "[{}; {}]", ty, size),
            IrType::Slice(ty) => write!(f, "[{}]", ty),
            IrType::String => write!(f, "string"),
            IrType::Function {
                params,
                return_type,
                varargs,
            } => {
                write!(f, "fn(")?;
                for (i, param) in params.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", param)?;
                }
                if *varargs {
                    write!(f, ", ...")?;
                }
                write!(f, ") -> {}", return_type)
            }
            IrType::Struct { name, .. } => write!(f, "struct {}", name),
            IrType::Union { name, .. } => write!(f, "union {}", name),
            IrType::Opaque { name, .. } => write!(f, "opaque {}", name),
            IrType::TypeVar(name) => write!(f, "${}", name),
            IrType::Generic { base, type_args } => {
                write!(f, "{}<", base)?;
                for (i, arg) in type_args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ">")
            }
            IrType::Any => write!(f, "any"),
            IrType::Vector { element, count } => write!(f, "vec<{}; {}>", element, count),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_size() {
        assert_eq!(IrType::I32.size(), 4);
        assert_eq!(IrType::Bool.size(), 1);
        assert_eq!(IrType::F64.size(), 8);
        assert_eq!(IrType::Array(Box::new(IrType::I32), 10).size(), 40);
    }

    #[test]
    fn test_type_display() {
        assert_eq!(format!("{}", IrType::I32), "i32");
        assert_eq!(format!("{}", IrType::Ptr(Box::new(IrType::I32))), "*i32");
        assert_eq!(
            format!("{}", IrType::Array(Box::new(IrType::U8), 16)),
            "[u8; 16]"
        );
    }

    #[test]
    fn test_type_properties() {
        assert!(IrType::I32.is_primitive());
        assert!(IrType::I32.is_integer());
        assert!(IrType::I32.is_signed_integer());
        assert!(!IrType::U32.is_signed_integer());
        assert!(IrType::F32.is_float());
        assert!(IrType::Ptr(Box::new(IrType::I32)).is_pointer());
    }
}
