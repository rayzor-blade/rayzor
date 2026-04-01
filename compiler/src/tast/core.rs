//! Core Type System for TAST (Typed AST)
//!
//! This module provides the foundation for type representation, storage, and management
//! in the Typed AST system. Features:
//! - Complete Haxe type system representation
//! - High-performance arena-allocated storage
//! - Efficient type lookup and caching
//! - Generic type instantiation support
//! - Integration with Symbol and Scope systems

use crate::tast::SourceLocation;

use super::{
    collections::{new_id_map, IdMap, IdSet},
    type_cache::{TypeCache, TypeCacheKey},
    InternedString, LifetimeId, ScopeId, StringInterner, SymbolId, TypeId, TypedArena,
};
use std::collections::BTreeMap;
use std::fmt;
use std::rc::Rc;

/// Core type kinds representing all Haxe types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeKind {
    // === Primitive Types ===
    /// void type (no value)
    Void,
    /// Boolean type (true/false)
    Bool,
    /// 32-bit signed integer
    Int,
    /// 64-bit floating point number
    Float,
    /// UTF-8 string
    String,
    Char,

    // === Named Types ===
    /// Class type with optional generic type arguments
    Class {
        symbol_id: SymbolId,
        type_args: Vec<TypeId>,
    },

    /// Interface type with optional generic type arguments
    Interface {
        symbol_id: SymbolId,
        type_args: Vec<TypeId>,
    },

    /// Enum type with optional generic type arguments
    Enum {
        symbol_id: SymbolId,
        type_args: Vec<TypeId>,
    },

    /// Abstract type with optional underlying type
    Abstract {
        symbol_id: SymbolId,
        underlying: Option<TypeId>,
        type_args: Vec<TypeId>,
    },

    /// Type alias/typedef
    TypeAlias {
        symbol_id: SymbolId,
        target_type: TypeId,
        type_args: Vec<TypeId>,
    },

    // === Function Types ===
    /// Function type: (param_types...) -> return_type
    Function {
        params: Vec<TypeId>,
        return_type: TypeId,
        /// Function effects (throw, async, etc.)
        effects: FunctionEffects,
    },

    // === Collection Types ===
    /// Array<T> type
    Array {
        element_type: TypeId,
    },

    /// Map<K,V> type
    Map {
        key_type: TypeId,
        value_type: TypeId,
    },

    // === Special Types ===
    /// Nullable type: T?
    Optional {
        inner_type: TypeId,
    },

    /// Dynamic type (no compile-time type checking)
    Dynamic,

    /// Unknown type (for type inference)
    Unknown,

    /// Error type (for error recovery during type checking)
    Error,

    /// Placeholder type (for forward references during AST lowering)
    Placeholder {
        name: InternedString,
    },

    // === Generic Types ===
    /// Type parameter: T, U, etc.
    TypeParameter {
        symbol_id: SymbolId,
        constraints: Vec<TypeId>,
        variance: Variance,
    },

    /// Generic instance: Array<String>, Map<Int, Bool>, etc.
    GenericInstance {
        base_type: TypeId,
        type_args: Vec<TypeId>,
        /// Cached instantiation for performance
        instantiation_cache_id: Option<u32>,
    },

    // === Advanced Types ===
    /// Anonymous structure type: { field1: Type1, field2: Type2 }
    Anonymous {
        fields: Vec<AnonymousField>,
    },

    /// Union type: Type1 | Type2 | Type3
    Union {
        types: Vec<TypeId>,
    },

    /// Intersection type: Type1 & Type2 & Type3
    Intersection {
        types: Vec<TypeId>,
    },

    /// Reference type for ownership analysis
    Reference {
        target_type: TypeId,
        mutability: Mutability,
        lifetime_id: LifetimeId,
    },
}

/// Function effect annotations
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FunctionEffects {
    /// Can throw exceptions
    pub can_throw: bool,
    /// Is async/awaitable
    pub is_async: bool,
    /// Is pure (no side effects)
    pub is_pure: bool,
    /// Memory effects for optimization
    pub memory_effects: MemoryEffects,
}

/// Memory effect annotations for optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryEffects {
    /// No memory effects (pure)
    None,
    /// Only reads memory
    ReadOnly,
    /// May modify memory
    ReadWrite,
    /// Unknown memory effects
    Unknown,
}

impl Default for MemoryEffects {
    fn default() -> Self {
        Self::Unknown
    }
}

/// Type parameter variance for generic types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Variance {
    /// T is covariant (can be more specific)
    Covariant,
    /// T is contravariant (can be more general)
    Contravariant,
    /// T is invariant (must be exact)
    Invariant,
}

impl Default for Variance {
    fn default() -> Self {
        Self::Invariant
    }
}

/// Mutability for references and variables
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Mutability {
    /// Immutable (final/const)
    Immutable,
    /// Mutable (var)
    Mutable,
}

impl Default for Mutability {
    fn default() -> Self {
        Self::Immutable
    }
}

/// Anonymous type field
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnonymousField {
    pub name: InternedString,
    pub type_id: TypeId,
    pub is_public: bool,
    pub optional: bool,
}

/// Complete type representation
#[derive(Debug, Clone)]
pub struct Type {
    /// Unique identifier for this type
    pub id: TypeId,

    /// The kind of type this represents
    pub kind: TypeKind,

    /// Source location where this type was defined/inferred
    pub source_location: SourceLocation,

    /// Flags for type properties
    pub flags: TypeFlags,

    /// Size hint for optimization (bytes, if known)
    pub size_hint: Option<u32>,

    /// Alignment hint for optimization (bytes, if known)
    pub alignment_hint: Option<u32>,
}

/// Type flags for various properties
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TypeFlags {
    /// Type is complete (fully defined)
    pub is_complete: bool,

    /// Type is recursive (contains itself)
    pub is_recursive: bool,

    /// Type is extern (defined outside Haxe)
    pub is_extern: bool,

    /// Type is abstract (cannot be instantiated directly)
    pub is_abstract: bool,

    /// Type is final (cannot be extended)
    pub is_final: bool,

    /// Type implements Copy semantics
    pub is_copy: bool,

    /// Type needs drop/cleanup
    pub needs_drop: bool,

    /// Type is zero-sized
    pub is_zero_sized: bool,

    /// Type is guaranteed non-null (@:notNull)
    pub is_non_null: bool,
}

impl Type {
    /// Create a new type with the given ID and kind
    pub fn new(id: TypeId, kind: TypeKind) -> Self {
        Self {
            id,
            kind,
            source_location: SourceLocation::unknown(),
            flags: TypeFlags::default(),
            size_hint: None,
            alignment_hint: None,
        }
    }

    /// Create a new type with source location
    pub fn with_location(id: TypeId, kind: TypeKind, location: SourceLocation) -> Self {
        Self {
            id,
            kind,
            source_location: location,
            flags: TypeFlags::default(),
            size_hint: None,
            alignment_hint: None,
        }
    }

    /// Check if this type is a primitive type
    pub fn is_primitive(&self) -> bool {
        matches!(
            self.kind,
            TypeKind::Void | TypeKind::Bool | TypeKind::Int | TypeKind::Float | TypeKind::String
        )
    }

    /// Check if this type is a named type (class, interface, enum, etc.)
    pub fn is_named_type(&self) -> bool {
        matches!(
            self.kind,
            TypeKind::Class { .. }
                | TypeKind::Interface { .. }
                | TypeKind::Enum { .. }
                | TypeKind::Abstract { .. }
                | TypeKind::TypeAlias { .. }
        )
    }

    /// Check if this type is a generic type (has type parameters)
    pub fn is_generic(&self) -> bool {
        match &self.kind {
            TypeKind::Class { type_args, .. }
            | TypeKind::Interface { type_args, .. }
            | TypeKind::Enum { type_args, .. }
            | TypeKind::Abstract { type_args, .. }
            | TypeKind::TypeAlias { type_args, .. } => !type_args.is_empty(),
            TypeKind::GenericInstance { .. } | TypeKind::TypeParameter { .. } => true,
            _ => false,
        }
    }

    /// Check if this type is a function type
    pub fn is_function(&self) -> bool {
        matches!(self.kind, TypeKind::Function { .. })
    }

    /// Check if this type is nullable/optional
    pub fn is_nullable(&self) -> bool {
        matches!(self.kind, TypeKind::Optional { .. })
    }

    /// Get the symbol ID if this is a named type
    pub fn symbol_id(&self) -> Option<SymbolId> {
        match &self.kind {
            TypeKind::Class { symbol_id, .. }
            | TypeKind::Interface { symbol_id, .. }
            | TypeKind::Enum { symbol_id, .. }
            | TypeKind::Abstract { symbol_id, .. }
            | TypeKind::TypeAlias { symbol_id, .. }
            | TypeKind::TypeParameter { symbol_id, .. } => Some(*symbol_id),
            _ => None,
        }
    }

    /// Get all type arguments for generic types
    pub fn type_args(&self) -> &[TypeId] {
        match &self.kind {
            TypeKind::Class { type_args, .. }
            | TypeKind::Interface { type_args, .. }
            | TypeKind::Enum { type_args, .. }
            | TypeKind::Abstract { type_args, .. }
            | TypeKind::TypeAlias { type_args, .. }
            | TypeKind::GenericInstance { type_args, .. } => type_args,
            _ => &[],
        }
    }

    /// Get the underlying type for wrappers (Optional, Abstract, etc.)
    pub fn underlying_type(&self) -> Option<TypeId> {
        match &self.kind {
            TypeKind::Optional { inner_type } => Some(*inner_type),
            TypeKind::Abstract { underlying, .. } => *underlying,
            TypeKind::TypeAlias { target_type, .. } => Some(*target_type),
            TypeKind::Array { element_type } => Some(*element_type),
            TypeKind::Reference { target_type, .. } => Some(*target_type),
            _ => None,
        }
    }
}

/// High-performance type table for efficient storage and lookup
pub struct TypeTable {
    /// Arena for allocating types
    arena: TypedArena<Type>,

    /// All types indexed by ID
    types: Vec<&'static Type>,

    /// Reverse lookup: type kind hash -> type IDs (for deduplication)
    kind_index: BTreeMap<u64, Vec<TypeId>>,

    /// Named types by symbol ID
    symbol_index: IdMap<SymbolId, Vec<TypeId>>,

    /// Generic instances cache for performance
    generic_cache: BTreeMap<(TypeId, Vec<TypeId>), TypeId>,

    /// Type usage tracking for analysis
    usage_stats: IdMap<TypeId, TypeUsageStats>,

    /// Common types cache (primitives, etc.)
    common_types: CommonTypesCache,

    /// String interner for efficient string management
    string_interner: StringInterner,

    /// Multi-level type cache for frequently accessed types
    type_cache: Rc<TypeCache>,
}

/// Usage statistics for types
#[derive(Debug, Clone, Default)]
pub struct TypeUsageStats {
    /// Number of times this type is referenced
    pub reference_count: u32,
    /// Number of times this type is instantiated
    pub instantiation_count: u32,
    /// Scopes where this type is used
    pub used_in_scopes: IdSet<ScopeId>,
}

/// Cache for commonly used types
#[derive(Debug)]
struct CommonTypesCache {
    void_type: TypeId,
    bool_type: TypeId,
    int_type: TypeId,
    float_type: TypeId,
    string_type: TypeId,
    dynamic_type: TypeId,
    unknown_type: TypeId,
    error_type: TypeId,
}

impl TypeTable {
    /// Create a new type table with common types pre-allocated
    pub fn new() -> Self {
        let arena = TypedArena::new();
        let types = Vec::new();
        let kind_index = BTreeMap::new();
        let symbol_index = new_id_map();
        let generic_cache = BTreeMap::new();
        let usage_stats = new_id_map();

        let mut table = Self {
            arena,
            types,
            kind_index,
            symbol_index,
            generic_cache,
            usage_stats,
            string_interner: StringInterner::new(),
            type_cache: Rc::new(TypeCache::new()),
            // Temporary placeholder - will be filled below
            common_types: CommonTypesCache {
                void_type: TypeId::invalid(),
                bool_type: TypeId::invalid(),
                int_type: TypeId::invalid(),
                float_type: TypeId::invalid(),
                string_type: TypeId::invalid(),
                dynamic_type: TypeId::invalid(),
                unknown_type: TypeId::invalid(),
                error_type: TypeId::invalid(),
            },
        };

        // Pre-allocate common primitive types
        table.common_types = CommonTypesCache {
            void_type: table.intern_type(TypeKind::Void),
            bool_type: table.intern_type(TypeKind::Bool),
            int_type: table.intern_type(TypeKind::Int),
            float_type: table.intern_type(TypeKind::Float),
            string_type: table.intern_type(TypeKind::String),
            dynamic_type: table.intern_type(TypeKind::Dynamic),
            unknown_type: table.intern_type(TypeKind::Unknown),
            error_type: table.intern_type(TypeKind::Error),
        };

        table
    }

    /// Create a type table with custom arena configuration
    pub fn with_capacity(capacity: usize) -> Self {
        let arena = TypedArena::with_capacity(capacity);
        let types = Vec::with_capacity(capacity);
        let kind_index = BTreeMap::new();
        let symbol_index = new_id_map();
        let generic_cache = BTreeMap::new();
        let usage_stats = new_id_map();

        let mut table = Self {
            arena,
            types,
            kind_index,
            symbol_index,
            generic_cache,
            usage_stats,
            string_interner: StringInterner::new(),
            type_cache: Rc::new(TypeCache::new()),
            common_types: CommonTypesCache {
                void_type: TypeId::invalid(),
                bool_type: TypeId::invalid(),
                int_type: TypeId::invalid(),
                float_type: TypeId::invalid(),
                string_type: TypeId::invalid(),
                dynamic_type: TypeId::invalid(),
                unknown_type: TypeId::invalid(),
                error_type: TypeId::invalid(),
            },
        };

        // Pre-allocate common types
        table.common_types = CommonTypesCache {
            void_type: table.intern_type(TypeKind::Void),
            bool_type: table.intern_type(TypeKind::Bool),
            int_type: table.intern_type(TypeKind::Int),
            float_type: table.intern_type(TypeKind::Float),
            string_type: table.intern_type(TypeKind::String),
            dynamic_type: table.intern_type(TypeKind::Dynamic),
            unknown_type: table.intern_type(TypeKind::Unknown),
            error_type: table.intern_type(TypeKind::Error),
        };

        table
    }

    /// Get a type by its ID
    pub fn get(&self, id: TypeId) -> Option<&Type> {
        if !id.is_valid() {
            return None;
        }

        self.types.get(id.as_raw() as usize).copied()
    }

    /// Iterate over all types in the table
    /// Returns an iterator of (TypeId, &Type) pairs
    pub fn iter(&self) -> impl Iterator<Item = (TypeId, &Type)> {
        self.types
            .iter()
            .enumerate()
            .map(|(idx, ty)| (TypeId::from_raw(idx as u32), *ty))
    }

    /// Get a type by its ID (unchecked)
    ///
    /// # Safety
    /// The caller must ensure the ID is valid and within bounds.
    pub unsafe fn get_unchecked(&self, id: TypeId) -> &Type {
        self.types.get_unchecked(id.as_raw() as usize)
    }

    /// Create a new type with automatic ID assignment
    pub fn create_type(&mut self, kind: TypeKind) -> TypeId {
        self.intern_type(kind)
    }

    /// Create a new type with automatic ID assignment
    pub fn create_union_type(&mut self, mut types: Vec<TypeId>) -> TypeId {
        // Sort types for consistent caching
        types.sort();
        types.dedup();

        let cache_key = TypeCacheKey::UnionType(types.clone());

        // Check cache first
        if let Some(cached_id) = self.type_cache.get(&cache_key) {
            return cached_id;
        }

        let type_id = self.intern_type(TypeKind::Union { types });
        self.type_cache.insert(cache_key, type_id);
        type_id
    }

    /// Create a new type with source location
    pub fn create_type_with_location(
        &mut self,
        kind: TypeKind,
        location: SourceLocation,
    ) -> TypeId {
        self.intern_type_with_location(kind, location)
    }

    /// Intern a type, returning existing ID if the type already exists
    fn intern_type(&mut self, kind: TypeKind) -> TypeId {
        self.intern_type_with_location(kind, SourceLocation::unknown())
    }

    /// Intern a type with source location
    fn intern_type_with_location(&mut self, kind: TypeKind, location: SourceLocation) -> TypeId {
        // Check if this type already exists
        let kind_hash = self.hash_type_kind(&kind);
        if let Some(existing_ids) = self.kind_index.get(&kind_hash) {
            for &existing_id in existing_ids {
                if let Some(existing_type) = self.get(existing_id) {
                    if existing_type.kind == kind {
                        return existing_id;
                    }
                }
            }
        }

        // Create new type
        let type_id = TypeId::from_raw(self.types.len() as u32);
        let new_type = self
            .arena
            .alloc(Type::with_location(type_id, kind, location));

        // SAFETY: Arena ensures this reference is valid for the lifetime of TypeTable
        let static_type_ref: &'static Type = unsafe { std::mem::transmute(new_type) };

        self.types.push(static_type_ref);

        // Update indices
        self.kind_index
            .entry(kind_hash)
            .or_insert_with(Vec::new)
            .push(type_id);

        // Index by symbol if it's a named type
        if let Some(symbol_id) = static_type_ref.symbol_id() {
            self.symbol_index
                .entry(symbol_id)
                .or_insert_with(Vec::new)
                .push(type_id);
        }

        // Initialize usage stats
        self.usage_stats.insert(type_id, TypeUsageStats::default());

        type_id
    }

    /// Create a simple hash of a type kind for deduplication
    fn hash_type_kind(&self, kind: &TypeKind) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        std::mem::discriminant(kind).hash(&mut hasher);

        match kind {
            TypeKind::Void
            | TypeKind::Bool
            | TypeKind::Int
            | TypeKind::Char
            | TypeKind::Float
            | TypeKind::String
            | TypeKind::Dynamic
            | TypeKind::Unknown
            | TypeKind::Error => {
                // No additional data to hash
            }
            TypeKind::Placeholder { name } => {
                name.hash(&mut hasher);
            }
            TypeKind::Class {
                symbol_id,
                type_args,
            }
            | TypeKind::Interface {
                symbol_id,
                type_args,
            }
            | TypeKind::Enum {
                symbol_id,
                type_args,
            } => {
                symbol_id.hash(&mut hasher);
                type_args.hash(&mut hasher);
            }
            TypeKind::Abstract {
                symbol_id,
                underlying,
                type_args,
            } => {
                symbol_id.hash(&mut hasher);
                underlying.hash(&mut hasher);
                type_args.hash(&mut hasher);
            }
            TypeKind::TypeAlias {
                symbol_id,
                target_type,
                type_args,
            } => {
                symbol_id.hash(&mut hasher);
                target_type.hash(&mut hasher);
                type_args.hash(&mut hasher);
            }
            TypeKind::Function {
                params,
                return_type,
                effects,
            } => {
                params.hash(&mut hasher);
                return_type.hash(&mut hasher);
                effects.hash(&mut hasher);
            }
            TypeKind::Array { element_type } => {
                element_type.hash(&mut hasher);
            }
            TypeKind::Map {
                key_type,
                value_type,
            } => {
                key_type.hash(&mut hasher);
                value_type.hash(&mut hasher);
            }
            TypeKind::Optional { inner_type } => {
                inner_type.hash(&mut hasher);
            }
            TypeKind::TypeParameter {
                symbol_id,
                constraints,
                variance,
            } => {
                symbol_id.hash(&mut hasher);
                constraints.hash(&mut hasher);
                variance.hash(&mut hasher);
            }
            TypeKind::GenericInstance {
                base_type,
                type_args,
                ..
            } => {
                base_type.hash(&mut hasher);
                type_args.hash(&mut hasher);
            }
            TypeKind::Anonymous { fields } => {
                fields.hash(&mut hasher);
            }
            TypeKind::Union { types } | TypeKind::Intersection { types } => {
                types.hash(&mut hasher);
            }
            TypeKind::Reference {
                target_type,
                mutability,
                lifetime_id,
            } => {
                target_type.hash(&mut hasher);
                mutability.hash(&mut hasher);
                lifetime_id.hash(&mut hasher);
            }
        }

        hasher.finish()
    }

    // === Common Type Getters ===

    /// Get the void type
    pub fn void_type(&self) -> TypeId {
        self.common_types.void_type
    }

    /// Get the bool type
    pub fn bool_type(&self) -> TypeId {
        self.common_types.bool_type
    }

    /// Get the int type
    pub fn int_type(&self) -> TypeId {
        self.common_types.int_type
    }

    /// Get the float type
    pub fn float_type(&self) -> TypeId {
        self.common_types.float_type
    }

    /// Get the string type
    pub fn string_type(&self) -> TypeId {
        self.common_types.string_type
    }

    /// Get the dynamic type
    pub fn dynamic_type(&self) -> TypeId {
        self.common_types.dynamic_type
    }

    /// Get the unknown type (for inference)
    pub fn unknown_type(&self) -> TypeId {
        self.common_types.unknown_type
    }

    /// Get the error type (for error recovery)
    pub fn error_type(&self) -> TypeId {
        self.common_types.error_type
    }

    // === Type Construction Helpers ===

    /// Create an optional/nullable type: T?
    pub fn create_optional_type(&mut self, inner_type: TypeId) -> TypeId {
        let cache_key = TypeCacheKey::OptionalType(inner_type);

        // Check cache first
        if let Some(cached_id) = self.type_cache.get(&cache_key) {
            return cached_id;
        }

        let type_id = self.intern_type(TypeKind::Optional { inner_type });
        self.type_cache.insert(cache_key, type_id);
        type_id
    }

    /// Create an array type: Array<T>
    pub fn create_array_type(&mut self, element_type: TypeId) -> TypeId {
        let cache_key = TypeCacheKey::ArrayType(element_type);

        // Check cache first
        if let Some(cached_id) = self.type_cache.get(&cache_key) {
            return cached_id;
        }

        let type_id = self.intern_type(TypeKind::Array { element_type });
        self.type_cache.insert(cache_key, type_id);
        type_id
    }

    /// Create a map type: Map<K, V>
    pub fn create_map_type(&mut self, key_type: TypeId, value_type: TypeId) -> TypeId {
        let cache_key = TypeCacheKey::MapType(key_type, value_type);

        // Check cache first
        if let Some(cached_id) = self.type_cache.get(&cache_key) {
            return cached_id;
        }

        let type_id = self.intern_type(TypeKind::Map {
            key_type,
            value_type,
        });
        self.type_cache.insert(cache_key, type_id);
        type_id
    }

    /// Create a function type: (params...) -> return_type
    pub fn create_function_type(&mut self, params: Vec<TypeId>, return_type: TypeId) -> TypeId {
        let cache_key = TypeCacheKey::FunctionType(params.clone(), return_type);

        // Check cache first
        if let Some(cached_id) = self.type_cache.get(&cache_key) {
            return cached_id;
        }

        let type_id = self.intern_type(TypeKind::Function {
            params,
            return_type,
            effects: FunctionEffects::default(),
        });
        self.type_cache.insert(cache_key, type_id);
        type_id
    }

    /// Create a function type with effects
    pub fn create_function_type_with_effects(
        &mut self,
        params: Vec<TypeId>,
        return_type: TypeId,
        effects: FunctionEffects,
    ) -> TypeId {
        self.intern_type(TypeKind::Function {
            params,
            return_type,
            effects,
        })
    }

    /// Create a class type
    pub fn create_class_type(&mut self, symbol_id: SymbolId, type_args: Vec<TypeId>) -> TypeId {
        let cache_key = if type_args.is_empty() {
            TypeCacheKey::NamedType(symbol_id, 0)
        } else {
            TypeCacheKey::GenericType(symbol_id, type_args.clone(), 0)
        };

        // Check cache first
        if let Some(cached_id) = self.type_cache.get(&cache_key) {
            return cached_id;
        }

        let type_id = self.intern_type(TypeKind::Class {
            symbol_id,
            type_args,
        });
        self.type_cache.insert(cache_key, type_id);
        type_id
    }

    /// Create an interface type
    pub fn create_interface_type(&mut self, symbol_id: SymbolId, type_args: Vec<TypeId>) -> TypeId {
        let cache_key = if type_args.is_empty() {
            TypeCacheKey::NamedType(symbol_id, 1)
        } else {
            TypeCacheKey::GenericType(symbol_id, type_args.clone(), 1)
        };

        // Check cache first
        if let Some(cached_id) = self.type_cache.get(&cache_key) {
            return cached_id;
        }

        let type_id = self.intern_type(TypeKind::Interface {
            symbol_id,
            type_args,
        });
        self.type_cache.insert(cache_key, type_id);
        type_id
    }

    /// Create an enum type
    pub fn create_enum_type(&mut self, symbol_id: SymbolId, type_args: Vec<TypeId>) -> TypeId {
        let cache_key = if type_args.is_empty() {
            TypeCacheKey::NamedType(symbol_id, 2)
        } else {
            TypeCacheKey::GenericType(symbol_id, type_args.clone(), 2)
        };

        // Check cache first
        if let Some(cached_id) = self.type_cache.get(&cache_key) {
            return cached_id;
        }

        let type_id = self.intern_type(TypeKind::Enum {
            symbol_id,
            type_args,
        });
        self.type_cache.insert(cache_key, type_id);
        type_id
    }

    /// Create an abstract type
    pub fn create_abstract_type(
        &mut self,
        symbol_id: SymbolId,
        underlying: Option<TypeId>,
        type_args: Vec<TypeId>,
    ) -> TypeId {
        self.intern_type(TypeKind::Abstract {
            symbol_id,
            underlying,
            type_args,
        })
    }

    /// Create a generic instance with caching
    pub fn create_generic_instance(&mut self, base_type: TypeId, type_args: Vec<TypeId>) -> TypeId {
        // Use a temporary key for lookup to avoid cloning unless necessary
        let _temp_key = (base_type, &type_args);

        // Check if this combination already exists
        for ((cached_base, cached_args), &cached_id) in &self.generic_cache {
            if *cached_base == base_type && cached_args == &type_args {
                return cached_id;
            }
        }

        // Not in cache, create new instance
        // Clone type_args only once for both the TypeKind and the cache
        let type_args_clone = type_args.clone();

        let instance_id = self.intern_type(TypeKind::GenericInstance {
            base_type,
            type_args,
            instantiation_cache_id: None,
        });

        // Store in cache using the already cloned args
        self.generic_cache
            .insert((base_type, type_args_clone), instance_id);
        instance_id
    }

    /// Create a type parameter
    pub fn create_type_parameter(
        &mut self,
        symbol_id: SymbolId,
        constraints: Vec<TypeId>,
        variance: Variance,
    ) -> TypeId {
        self.intern_type(TypeKind::TypeParameter {
            symbol_id,
            constraints,
            variance,
        })
    }

    // === Lookup Operations ===

    /// Find all types associated with a symbol
    pub fn types_for_symbol(&self, symbol_id: SymbolId) -> Option<&[TypeId]> {
        self.symbol_index.get(&symbol_id).map(|v| v.as_slice())
    }

    /// Check if a type exists with the given kind
    pub fn find_type_with_kind(&self, kind: &TypeKind) -> Option<TypeId> {
        let kind_hash = self.hash_type_kind(kind);
        if let Some(existing_ids) = self.kind_index.get(&kind_hash) {
            for &existing_id in existing_ids {
                if let Some(existing_type) = self.get(existing_id) {
                    if existing_type.kind == *kind {
                        return Some(existing_id);
                    }
                }
            }
        }
        None
    }

    // === Usage Tracking ===

    /// Record that a type is being used/referenced
    pub fn record_type_usage(&mut self, type_id: TypeId, scope_id: ScopeId) {
        if let Some(stats) = self.usage_stats.get_mut(&type_id) {
            stats.reference_count += 1;
            stats.used_in_scopes.insert(scope_id);
        }
    }

    /// Record that a type is being instantiated
    pub fn record_type_instantiation(&mut self, type_id: TypeId) {
        if let Some(stats) = self.usage_stats.get_mut(&type_id) {
            stats.instantiation_count += 1;
        }
    }

    /// Get usage statistics for a type
    pub fn get_usage_stats(&self, type_id: TypeId) -> Option<&TypeUsageStats> {
        self.usage_stats.get(&type_id)
    }

    // === Statistics and Analysis ===

    /// Get comprehensive statistics about the type table
    pub fn stats(&self) -> TypeTableStats {
        let arena_stats = self.arena.stats();

        let mut primitive_count = 0;
        let mut named_type_count = 0;
        let mut generic_count = 0;
        let mut function_count = 0;

        for type_ref in &self.types {
            match &type_ref.kind {
                TypeKind::Void
                | TypeKind::Bool
                | TypeKind::Int
                | TypeKind::Float
                | TypeKind::String => primitive_count += 1,

                TypeKind::Class { .. }
                | TypeKind::Interface { .. }
                | TypeKind::Enum { .. }
                | TypeKind::Abstract { .. }
                | TypeKind::TypeAlias { .. } => named_type_count += 1,

                TypeKind::TypeParameter { .. } | TypeKind::GenericInstance { .. } => {
                    generic_count += 1
                }

                TypeKind::Function { .. } => function_count += 1,

                _ => {}
            }
        }

        TypeTableStats {
            total_types: self.types.len(),
            primitive_types: primitive_count,
            named_types: named_type_count,
            generic_types: generic_count,
            function_types: function_count,
            unique_type_kinds: self.kind_index.len(),
            generic_cache_size: self.generic_cache.len(),
            total_memory_bytes: arena_stats.total_bytes_allocated,
            arena_chunks: arena_stats.chunk_count,
            average_types_per_symbol: if self.symbol_index.is_empty() {
                0.0
            } else {
                self.types.len() as f64 / self.symbol_index.len() as f64
            },
        }
    }

    /// Get the number of types in the table
    pub fn len(&self) -> usize {
        self.types.len()
    }

    /// Check if the type table is empty
    pub fn is_empty(&self) -> bool {
        self.types.is_empty()
    }

    /// Check if a type is a type parameter
    pub fn is_type_parameter(&self, type_id: TypeId) -> bool {
        if let Some(type_obj) = self.get(type_id) {
            matches!(type_obj.kind, TypeKind::TypeParameter { .. })
        } else {
            false
        }
    }

    /// Intern a string into the string interner
    pub fn intern_string(&mut self, s: &str) -> InternedString {
        self.string_interner.intern(s)
    }

    /// Get a string from an interned string
    pub fn get_string(&self, interned: InternedString) -> Option<&str> {
        self.string_interner.get(interned)
    }

    /// Find unused types (never referenced)
    pub fn find_unused_types(&self) -> Vec<TypeId> {
        let mut unused = Vec::new();

        for (type_id, stats) in &self.usage_stats {
            if stats.reference_count == 0 && stats.instantiation_count == 0 {
                unused.push(*type_id);
            }
        }

        unused
    }

    /// Get the most frequently used types
    pub fn most_used_types(&self, limit: usize) -> Vec<(TypeId, u32)> {
        let mut usage_pairs: Vec<(TypeId, u32)> = self
            .usage_stats
            .iter()
            .map(|(id, stats)| (*id, stats.reference_count + stats.instantiation_count))
            .collect();

        usage_pairs.sort_by(|a, b| b.1.cmp(&a.1));
        usage_pairs.truncate(limit);
        usage_pairs
    }

    pub fn get_type_symbol(&self, ty: TypeId) -> Option<SymbolId> {
        self.get(ty)
            .map(|t| t.symbol_id().unwrap_or(SymbolId::invalid()))
    }

    // === Cache Management Methods ===

    /// Preload common types into the cache for better performance
    pub fn preload_common_types_to_cache(&self) {
        let common_types = vec![
            (
                TypeCacheKey::ArrayType(self.dynamic_type()),
                self.create_array_type_uncached(self.dynamic_type()),
            ),
            (
                TypeCacheKey::ArrayType(self.string_type()),
                self.create_array_type_uncached(self.string_type()),
            ),
            (
                TypeCacheKey::ArrayType(self.int_type()),
                self.create_array_type_uncached(self.int_type()),
            ),
            (
                TypeCacheKey::OptionalType(self.string_type()),
                self.create_optional_type_uncached(self.string_type()),
            ),
            (
                TypeCacheKey::OptionalType(self.int_type()),
                self.create_optional_type_uncached(self.int_type()),
            ),
            (
                TypeCacheKey::OptionalType(self.bool_type()),
                self.create_optional_type_uncached(self.bool_type()),
            ),
        ];

        self.type_cache.preload_common_types(common_types);
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> super::type_cache::CacheStats {
        self.type_cache.stats()
    }

    /// Clear the type cache
    pub fn clear_cache(&self) {
        self.type_cache.clear();
    }

    /// Get current cache sizes (L1, L2)
    pub fn cache_sizes(&self) -> (usize, usize) {
        self.type_cache.sizes()
    }

    // Uncached versions for preloading
    fn create_array_type_uncached(&self, _element_type: TypeId) -> TypeId {
        // This would need access to intern_type, but since we're in an immutable context,
        // we'll use a placeholder. In practice, these would be created during initialization.
        self.dynamic_type() // Placeholder
    }

    fn create_optional_type_uncached(&self, _inner_type: TypeId) -> TypeId {
        // This would need access to intern_type, but since we're in an immutable context,
        // we'll use a placeholder. In practice, these would be created during initialization.
        self.dynamic_type() // Placeholder
    }

    /// Resolve a type alias to its target type with caching
    pub fn resolve_type_alias(&self, symbol_id: SymbolId) -> TypeId {
        let cache_key = TypeCacheKey::TypeAlias(symbol_id);

        // Check cache first
        if let Some(cached_id) = self.type_cache.get(&cache_key) {
            return cached_id;
        }

        // Look up the type alias and follow the chain
        if let Some(types) = self.symbol_index.get(&symbol_id) {
            for &type_id in types {
                if let Some(ty) = self.get(type_id) {
                    if let TypeKind::TypeAlias { target_type, .. } = &ty.kind {
                        self.type_cache.insert(cache_key, *target_type);
                        return *target_type;
                    }
                }
            }
        }

        // Not found, return dynamic type
        let dynamic = self.dynamic_type();
        self.type_cache.insert(cache_key, dynamic);
        dynamic
    }

    /// Resolve abstract type underlying with caching
    pub fn resolve_abstract_underlying(&self, symbol_id: SymbolId) -> Option<TypeId> {
        let cache_key = TypeCacheKey::AbstractUnderlying(symbol_id);

        // Check cache first - use dynamic_type as sentinel for None
        if let Some(cached_id) = self.type_cache.get(&cache_key) {
            return if cached_id == self.dynamic_type() {
                None
            } else {
                Some(cached_id)
            };
        }

        // Look up the abstract type
        if let Some(types) = self.symbol_index.get(&symbol_id) {
            for &type_id in types {
                if let Some(ty) = self.get(type_id) {
                    if let TypeKind::Abstract { underlying, .. } = &ty.kind {
                        if let Some(underlying_type) = underlying {
                            self.type_cache.insert(cache_key, *underlying_type);
                            return Some(*underlying_type);
                        }
                    }
                }
            }
        }

        // Not found, cache as dynamic (sentinel for None)
        self.type_cache.insert(cache_key, self.dynamic_type());
        None
    }
}

impl Default for TypeTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about type table usage and performance
#[derive(Debug, Clone)]
pub struct TypeTableStats {
    /// Total number of types
    pub total_types: usize,
    /// Number of primitive types
    pub primitive_types: usize,
    /// Number of named types (classes, interfaces, etc.)
    pub named_types: usize,
    /// Number of generic types
    pub generic_types: usize,
    /// Number of function types
    pub function_types: usize,
    /// Number of unique type kinds
    pub unique_type_kinds: usize,
    /// Generic instantiation cache size
    pub generic_cache_size: usize,
    /// Total memory used in bytes
    pub total_memory_bytes: usize,
    /// Number of arena chunks
    pub arena_chunks: usize,
    /// Average types per symbol
    pub average_types_per_symbol: f64,
}

impl TypeTableStats {
    /// Get memory usage per type in bytes
    pub fn memory_per_type(&self) -> f64 {
        if self.total_types == 0 {
            0.0
        } else {
            self.total_memory_bytes as f64 / self.total_types as f64
        }
    }

    /// Get cache hit ratio for generic instantiations
    pub fn generic_cache_efficiency(&self) -> f64 {
        if self.generic_types == 0 {
            0.0
        } else {
            self.generic_cache_size as f64 / self.generic_types as f64
        }
    }
}

// === Hash implementations for type kinds ===

impl std::hash::Hash for FunctionEffects {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.can_throw.hash(state);
        self.is_async.hash(state);
        self.is_pure.hash(state);
        std::mem::discriminant(&self.memory_effects).hash(state);
    }
}

impl std::hash::Hash for AnonymousField {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        self.type_id.hash(state);
        self.optional.hash(state);
    }
}

// === Display implementations ===

impl fmt::Display for TypeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TypeKind::Void => write!(f, "Void"),
            TypeKind::Bool => write!(f, "Bool"),
            TypeKind::Int => write!(f, "Int"),
            TypeKind::Float => write!(f, "Float"),
            TypeKind::String => write!(f, "String"),
            TypeKind::Dynamic => write!(f, "Dynamic"),
            TypeKind::Unknown => write!(f, "?"),
            TypeKind::Error => write!(f, "<error>"),
            TypeKind::Placeholder { name } => write!(f, "<placeholder:{:?}>", name),

            TypeKind::Class {
                symbol_id,
                type_args,
            } => {
                write!(f, "Class({:?}", symbol_id)?;
                if !type_args.is_empty() {
                    write!(f, "<{:?}>", type_args)?;
                }
                write!(f, ")")
            }

            TypeKind::Interface {
                symbol_id,
                type_args,
            } => {
                write!(f, "Interface({:?}", symbol_id)?;
                if !type_args.is_empty() {
                    write!(f, "<{:?}>", type_args)?;
                }
                write!(f, ")")
            }

            TypeKind::Function {
                params,
                return_type,
                ..
            } => {
                write!(f, "({:?}) -> {:?}", params, return_type)
            }

            TypeKind::Array { element_type } => {
                write!(f, "Array<{:?}>", element_type)
            }

            TypeKind::Optional { inner_type } => {
                write!(f, "{:?}?", inner_type)
            }

            _ => write!(f, "{:?}", self), // Fallback to debug for complex types
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.kind)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_table_creation() {
        let table = TypeTable::new();

        // Should have common types pre-allocated
        assert!(table.void_type().is_valid());
        assert!(table.bool_type().is_valid());
        assert!(table.int_type().is_valid());
        assert!(table.float_type().is_valid());
        assert!(table.string_type().is_valid());
        assert!(table.dynamic_type().is_valid());

        assert!(table.len() >= 6); // At least the primitive types
    }

    #[test]
    fn test_primitive_types() {
        let table = TypeTable::new();

        let void_type = table.get(table.void_type()).unwrap();
        assert!(matches!(void_type.kind, TypeKind::Void));
        assert!(void_type.is_primitive());

        let int_type = table.get(table.int_type()).unwrap();
        assert!(matches!(int_type.kind, TypeKind::Int));
        assert!(int_type.is_primitive());
    }

    #[test]
    fn test_type_deduplication() {
        let mut table = TypeTable::new();

        // Creating the same type twice should return the same ID
        let array1 = table.create_array_type(table.int_type());
        let array2 = table.create_array_type(table.int_type());

        assert_eq!(array1, array2);
    }

    #[test]
    fn test_optional_types() {
        let mut table = TypeTable::new();

        let optional_int = table.create_optional_type(table.int_type());
        let optional_type = table.get(optional_int).unwrap();

        assert!(matches!(optional_type.kind, TypeKind::Optional { .. }));
        assert!(optional_type.is_nullable());
        assert_eq!(optional_type.underlying_type(), Some(table.int_type()));
    }

    #[test]
    fn test_function_types() {
        let mut table = TypeTable::new();

        let func_type = table.create_function_type(
            vec![table.int_type(), table.string_type()],
            table.bool_type(),
        );

        let function = table.get(func_type).unwrap();
        assert!(function.is_function());

        if let TypeKind::Function {
            params,
            return_type,
            ..
        } = &function.kind
        {
            assert_eq!(params.len(), 2);
            assert_eq!(params[0], table.int_type());
            assert_eq!(params[1], table.string_type());
            assert_eq!(*return_type, table.bool_type());
        }
    }

    #[test]
    fn test_generic_types() {
        let mut table = TypeTable::new();

        let symbol_id = SymbolId::from_raw(1);
        let class_type = table.create_class_type(symbol_id, vec![table.string_type()]);

        let class = table.get(class_type).unwrap();
        assert!(class.is_generic());
        assert!(class.is_named_type());
        assert_eq!(class.symbol_id(), Some(symbol_id));
        assert_eq!(class.type_args(), &[table.string_type()]);
    }

    #[test]
    fn test_generic_instance_caching() {
        let mut table = TypeTable::new();

        let base_type = table.create_class_type(SymbolId::from_raw(1), vec![]);
        let type_args = vec![table.int_type(), table.string_type()];

        let instance1 = table.create_generic_instance(base_type, type_args.clone());
        let instance2 = table.create_generic_instance(base_type, type_args);

        assert_eq!(instance1, instance2);
    }

    #[test]
    fn test_usage_tracking() {
        let mut table = TypeTable::new();
        let scope_id = ScopeId::from_raw(1);

        let int_type = table.int_type();

        // Record usage
        table.record_type_usage(int_type, scope_id);
        table.record_type_instantiation(int_type);

        let stats = table.get_usage_stats(int_type).unwrap();
        assert_eq!(stats.reference_count, 1);
        assert_eq!(stats.instantiation_count, 1);
        assert!(stats.used_in_scopes.contains(&scope_id));
    }

    #[test]
    fn test_symbol_lookup() {
        let mut table = TypeTable::new();
        let symbol_id = SymbolId::from_raw(42);

        let class_type = table.create_class_type(symbol_id, vec![]);
        let interface_type = table.create_interface_type(symbol_id, vec![]);

        let types = table.types_for_symbol(symbol_id).unwrap();
        assert_eq!(types.len(), 2);
        assert!(types.contains(&class_type));
        assert!(types.contains(&interface_type));
    }

    #[test]
    fn test_type_table_stats() {
        let mut table = TypeTable::new();

        // Create various types
        table.create_array_type(table.int_type());
        table.create_function_type(vec![table.bool_type()], table.void_type());
        table.create_class_type(SymbolId::from_raw(1), vec![]);

        let stats = table.stats();
        assert!(stats.total_types > 0);
        assert!(stats.primitive_types >= 5); // Void, Bool, Int, Float, String
        assert!(stats.function_types >= 1);
        assert!(stats.named_types >= 1);
        assert!(stats.total_memory_bytes > 0);
    }

    #[test]
    fn test_type_flags_and_properties() {
        let mut table = TypeTable::new();

        let array_type = table.create_array_type(table.int_type());
        let array = table.get(array_type).unwrap();

        // Test type property queries
        assert!(!array.is_primitive());
        assert!(!array.is_function());
        assert!(!array.is_nullable());
        assert_eq!(array.underlying_type(), Some(table.int_type()));
    }

    #[test]
    fn test_complex_nested_types() {
        let mut table = TypeTable::new();

        // Create Array<Map<String, Int>>
        let map_type = table.create_map_type(table.string_type(), table.int_type());
        let array_of_maps = table.create_array_type(map_type);

        // Create function: (Array<Map<String, Int>>) -> Bool
        let func_type = table.create_function_type(vec![array_of_maps], table.bool_type());

        let function = table.get(func_type).unwrap();
        assert!(function.is_function());

        // Test that we can traverse the type structure
        if let TypeKind::Function { params, .. } = &function.kind {
            let param_type = table.get(params[0]).unwrap();
            if let TypeKind::Array { element_type } = &param_type.kind {
                let map_type = table.get(*element_type).unwrap();
                assert!(matches!(map_type.kind, TypeKind::Map { .. }));
            }
        }
    }

    #[test]
    fn test_type_display() {
        let table = TypeTable::new();

        let void_type = table.get(table.void_type()).unwrap();
        assert_eq!(format!("{}", void_type), "Void");

        let int_type = table.get(table.int_type()).unwrap();
        assert_eq!(format!("{}", int_type), "Int");
    }

    #[test]
    fn test_source_location() {
        let mut table = TypeTable::new();
        let location = SourceLocation::new(1, 10, 5, 100);

        // Use a unique class type that won't be deduplicated
        let unique_symbol = SymbolId::from_raw(9999);
        let class_type_kind = TypeKind::Class {
            symbol_id: unique_symbol,
            type_args: vec![],
        };
        let type_id = table.create_type_with_location(class_type_kind, location);
        let type_obj = table.get(type_id).unwrap();

        assert_eq!(type_obj.source_location, location);
        assert!(type_obj.source_location.is_valid());
    }
}
