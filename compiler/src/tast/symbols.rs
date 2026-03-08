//! High-Performance Symbol System for TAST
//!
//! This module provides efficient symbol storage, lookup, and management for the
//! Typed AST system. Features:
//! - Arena-allocated symbols for cache efficiency
//! - Fast lookup by name and ID
//! - Complete symbol metadata tracking
//! - Integration with string interning and ID systems
//! - Support for nested scopes and shadowing

use crate::tast::type_checker::{ClassHierarchyInfo, ClassHierarchyRegistry};

use super::{
    symbol_cache::SymbolResolutionCache, InternedString, LifetimeId, ScopeId, StringInterner,
    SymbolId, TypeId, TypedArena,
};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::rc::Rc;

/// The kind of symbol (variable, function, class, etc.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SymbolKind {
    /// Local variable or parameter
    Variable,
    /// Function parameter specifically
    Parameter,
    /// Function or method
    Function,
    /// Class definition
    Class,
    /// Interface definition
    Interface,
    /// Enum definition
    Enum,
    /// Type alias (typedef)
    TypeAlias,
    /// Abstract type
    Abstract,
    /// Field within a class/interface
    Field,
    /// Property (with get/set)
    Property,
    /// Enum variant
    EnumVariant,
    /// Module/package
    Module,
    /// Import alias
    ImportAlias,
    /// Type parameter
    TypeParameter,
    /// Macro definition
    Macro,
    /// Metadata definition
    Metadata,
}

impl SymbolKind {
    /// Check if this symbol kind represents a type
    pub fn is_type(self) -> bool {
        matches!(
            self,
            SymbolKind::Class
                | SymbolKind::Interface
                | SymbolKind::Enum
                | SymbolKind::TypeAlias
                | SymbolKind::Abstract
                | SymbolKind::TypeParameter
        )
    }

    /// Check if this symbol kind represents a value
    pub fn is_value(self) -> bool {
        matches!(
            self,
            SymbolKind::Variable
                | SymbolKind::Parameter
                | SymbolKind::Function
                | SymbolKind::Field
                | SymbolKind::Property
                | SymbolKind::EnumVariant
        )
    }

    /// Check if this symbol can be shadowed by another symbol
    pub fn can_be_shadowed(self) -> bool {
        matches!(
            self,
            SymbolKind::Variable
                | SymbolKind::Parameter
                | SymbolKind::ImportAlias
                | SymbolKind::TypeParameter
        )
    }

    /// Check if this symbol requires unique names in its scope
    pub fn requires_unique_name(self) -> bool {
        matches!(
            self,
            SymbolKind::Function
                | SymbolKind::Class
                | SymbolKind::Interface
                | SymbolKind::Enum
                | SymbolKind::TypeAlias
                | SymbolKind::Abstract
                | SymbolKind::Field
                | SymbolKind::Property
                | SymbolKind::EnumVariant
                | SymbolKind::Module
        )
    }
}

impl fmt::Display for SymbolKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            SymbolKind::Variable => "variable",
            SymbolKind::Parameter => "parameter",
            SymbolKind::Function => "function",
            SymbolKind::Class => "class",
            SymbolKind::Interface => "interface",
            SymbolKind::Enum => "enum",
            SymbolKind::TypeAlias => "typedef",
            SymbolKind::Abstract => "abstract",
            SymbolKind::Field => "field",
            SymbolKind::Property => "property",
            SymbolKind::EnumVariant => "enum variant",
            SymbolKind::Module => "module",
            SymbolKind::ImportAlias => "import",
            SymbolKind::TypeParameter => "type parameter",
            SymbolKind::Macro => "macro",
            SymbolKind::Metadata => "metadata",
        };
        write!(f, "{}", name)
    }
}

/// Visibility level for symbols
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Visibility {
    /// public - visible everywhere
    Public,
    /// private - visible only within the same class/module
    Private,
    /// internal - visible within the same package
    Internal,
    /// protected - visible within subclasses (if applicable)
    Protected,
}

impl Default for Visibility {
    fn default() -> Self {
        Visibility::Private
    }
}

impl fmt::Display for Visibility {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Visibility::Public => "public",
            Visibility::Private => "private",
            Visibility::Internal => "internal",
            Visibility::Protected => "protected",
        };
        write!(f, "{}", name)
    }
}

/// Mutability of a symbol
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Mutability {
    /// Immutable (final in Haxe)
    Immutable,
    /// Mutable (var in Haxe)
    Mutable,
    /// Unknown/inferred mutability
    Unknown,
}

impl Default for Mutability {
    fn default() -> Self {
        Mutability::Unknown
    }
}

impl fmt::Display for Mutability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Mutability::Immutable => "final",
            Mutability::Mutable => "var",
            Mutability::Unknown => "unknown",
        };
        write!(f, "{}", name)
    }
}

/// Source location information for a symbol
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SourceLocation {
    /// File ID where this symbol is defined
    pub file_id: u32,
    /// Line number (1-based)
    pub line: u32,
    /// Column number (1-based)
    pub column: u32,
    /// Byte offset in file
    pub byte_offset: u32,
}

impl SourceLocation {
    pub const fn new(file_id: u32, line: u32, column: u32, byte_offset: u32) -> Self {
        Self {
            file_id,
            line,
            column,
            byte_offset,
        }
    }

    pub const fn unknown() -> Self {
        Self::new(u32::MAX, 0, 0, 0)
    }

    pub const fn is_valid(self) -> bool {
        self.file_id != u32::MAX
    }
}

impl Default for SourceLocation {
    fn default() -> Self {
        Self::unknown()
    }
}

impl fmt::Display for SourceLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_valid() {
            write!(f, "{}:{}:{}", self.file_id, self.line, self.column)
        } else {
            write!(f, "<unknown>")
        }
    }
}

/// Complete symbol information
#[derive(Debug, Clone)]
pub struct Symbol {
    /// Unique identifier for this symbol
    pub id: SymbolId,
    /// Symbol name (interned string)
    pub name: InternedString,
    /// What kind of symbol this is
    pub kind: SymbolKind,
    /// Resolved type of this symbol
    pub type_id: TypeId,
    /// Scope where this symbol is defined
    pub scope_id: ScopeId,
    /// Lifetime associated with this symbol
    pub lifetime_id: LifetimeId,
    /// Visibility level
    pub visibility: Visibility,
    /// Mutability
    pub mutability: Mutability,
    /// Where this symbol is defined in source code
    pub definition_location: SourceLocation,
    /// Whether this symbol has been used
    pub is_used: bool,
    /// Whether this symbol is part of an export
    pub is_exported: bool,
    /// Optional documentation string
    pub documentation: Option<InternedString>,
    /// Additional flags for symbol properties
    pub flags: SymbolFlags,
    /// Package ID this symbol belongs to
    pub package_id: Option<super::namespace::PackageId>,
    /// Full qualified name (e.g., "com.example.MyClass")
    pub qualified_name: Option<InternedString>,
    /// Native name from @:native metadata (e.g., "rayzor::concurrent::Arc")
    /// Used for runtime mapping lookup. Lowered form replaces :: with _
    pub native_name: Option<InternedString>,
    /// Framework names from @:frameworks(["Accelerate", "CoreFoundation"]) metadata
    /// Auto-loaded into TCC context when using __c__() inline code
    pub frameworks: Option<Vec<InternedString>>,
    /// Include paths from @:cInclude(["vendor/stb"]) metadata
    /// Auto-added to TCC include search path for __c__() inline code
    pub c_includes: Option<Vec<InternedString>>,
    /// C source files from @:cSource(["vendor/stb_image.c"]) metadata
    /// Auto-compiled into TCC context for __c__() inline code
    pub c_sources: Option<Vec<InternedString>>,
    /// System libraries from @:clib(["sqlite3"]) metadata
    /// Discovered via pkg-config and loaded into TCC context
    pub c_libs: Option<Vec<InternedString>>,
}

/// Bitflags for various symbol properties
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SymbolFlags(u32);

impl SymbolFlags {
    pub const NONE: Self = Self(0);
    pub const STATIC: Self = Self(1 << 0);
    pub const INLINE: Self = Self(1 << 1);
    pub const OVERRIDE: Self = Self(1 << 2);
    pub const ABSTRACT: Self = Self(1 << 3);
    pub const FINAL: Self = Self(1 << 4);
    pub const EXTERN: Self = Self(1 << 5);
    pub const MACRO: Self = Self(1 << 6);
    pub const DYNAMIC: Self = Self(1 << 7);
    pub const OPTIONAL: Self = Self(1 << 8);
    pub const DEPRECATED: Self = Self(1 << 9);
    pub const COMPILER_GENERATED: Self = Self(1 << 10);
    /// @:generic - indicates the type should be monomorphized rather than type-erased
    pub const GENERIC: Self = Self(1 << 11);
    /// @:forward - forwards field access to underlying type (for abstracts)
    pub const FORWARD: Self = Self(1 << 12);
    /// @:native - has native runtime implementation
    pub const NATIVE: Self = Self(1 << 13);
    /// @:cstruct - C-compatible flat struct layout (no object header)
    pub const CSTRUCT: Self = Self(1 << 14);
    /// @:no_mangle or @:cstruct(NoMangle) - use unmangled name in C typedef
    pub const NO_MANGLE: Self = Self(1 << 15);
    /// @:gpuStruct - GPU-compatible flat struct layout (4-byte floats, no object header)
    pub const GPU_STRUCT: Self = Self(1 << 16);
    /// @:keep - preserve symbol even if seemingly unreachable (skip DCE)
    pub const KEEP: Self = Self(1 << 17);

    pub const fn empty() -> Self {
        Self::NONE
    }

    pub const fn contains(self, flag: Self) -> bool {
        (self.0 & flag.0) != 0
    }

    pub const fn insert(&mut self, flag: Self) {
        self.0 |= flag.0;
    }

    pub const fn remove(&mut self, flag: Self) {
        self.0 &= !flag.0;
    }

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub const fn intersection(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Check if this symbol requires monomorphization (@:generic)
    pub const fn is_generic(self) -> bool {
        self.contains(Self::GENERIC)
    }

    /// Check if this symbol has @:forward metadata
    pub const fn is_forward(self) -> bool {
        self.contains(Self::FORWARD)
    }

    /// Check if this symbol has @:native metadata
    pub const fn is_native(self) -> bool {
        self.contains(Self::NATIVE)
    }

    /// Check if this symbol has @:cstruct metadata
    pub const fn is_cstruct(self) -> bool {
        self.contains(Self::CSTRUCT)
    }

    /// Check if this symbol has @:no_mangle metadata
    pub const fn is_no_mangle(self) -> bool {
        self.contains(Self::NO_MANGLE)
    }

    /// Check if this symbol has @:gpuStruct metadata
    pub const fn is_gpu_struct(self) -> bool {
        self.contains(Self::GPU_STRUCT)
    }

    /// Check if this symbol has @:keep metadata
    pub const fn is_keep(self) -> bool {
        self.contains(Self::KEEP)
    }
}

impl Default for SymbolFlags {
    fn default() -> Self {
        Self::NONE
    }
}

impl Symbol {
    /// Create a new symbol with the given parameters
    pub fn new(
        id: SymbolId,
        name: InternedString,
        kind: SymbolKind,
        type_id: TypeId,
        scope_id: ScopeId,
        definition_location: SourceLocation,
    ) -> Self {
        Self {
            id,
            name,
            kind,
            type_id,
            scope_id,
            lifetime_id: LifetimeId::invalid(),
            visibility: Visibility::default(),
            mutability: Mutability::default(),
            definition_location,
            is_used: false,
            is_exported: false,
            documentation: None,
            flags: SymbolFlags::default(),
            package_id: None,
            qualified_name: None,
            native_name: None,
            frameworks: None,
            c_includes: None,
            c_sources: None,
            c_libs: None,
        }
    }

    /// Create a variable symbol
    pub fn variable(
        id: SymbolId,
        name: InternedString,
        type_id: TypeId,
        scope_id: ScopeId,
        mutability: Mutability,
        location: SourceLocation,
    ) -> Self {
        Self {
            kind: SymbolKind::Variable,
            mutability,
            ..Self::new(id, name, SymbolKind::Variable, type_id, scope_id, location)
        }
    }

    /// Create a function symbol
    pub fn function(
        id: SymbolId,
        name: InternedString,
        type_id: TypeId,
        scope_id: ScopeId,
        visibility: Visibility,
        location: SourceLocation,
    ) -> Self {
        Self {
            kind: SymbolKind::Function,
            visibility,
            ..Self::new(id, name, SymbolKind::Function, type_id, scope_id, location)
        }
    }

    /// Create a class symbol
    pub fn class(
        id: SymbolId,
        name: InternedString,
        type_id: TypeId,
        scope_id: ScopeId,
        visibility: Visibility,
        location: SourceLocation,
    ) -> Self {
        Self {
            kind: SymbolKind::Class,
            visibility,
            ..Self::new(id, name, SymbolKind::Class, type_id, scope_id, location)
        }
    }

    /// Mark this symbol as used
    pub fn mark_used(&mut self) {
        self.is_used = true;
    }

    /// Check if this symbol is static
    pub fn is_static(&self) -> bool {
        self.flags.contains(SymbolFlags::STATIC)
    }

    /// Check if this symbol is inline
    pub fn is_inline(&self) -> bool {
        self.flags.contains(SymbolFlags::INLINE)
    }

    /// Check if this symbol is abstract
    pub fn is_abstract(&self) -> bool {
        self.flags.contains(SymbolFlags::ABSTRACT)
    }

    /// Check if this symbol is final
    pub fn is_final(&self) -> bool {
        self.flags.contains(SymbolFlags::FINAL)
    }

    /// Check if this symbol has @:keep metadata
    pub fn is_keep(&self) -> bool {
        self.flags.contains(SymbolFlags::KEEP)
    }

    /// Get a human-readable description of this symbol
    pub fn description(&self, interner: &StringInterner) -> String {
        let name = interner.get(self.name).unwrap_or("<unknown>");
        format!("{} {} ({})", self.visibility, self.kind, name)
    }
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {} #{}", self.visibility, self.kind, self.id)
    }
}

/// Efficient storage and lookup for symbols
#[derive(Debug)]
pub struct SymbolTable {
    /// Arena for symbol storage
    symbols_arena: TypedArena<Symbol>,
    /// Map from symbol ID to symbol reference
    symbols_by_id: HashMap<SymbolId, &'static Symbol>,
    /// Map from (scope, name) to symbol ID for fast name lookup
    symbols_by_name: HashMap<(ScopeId, InternedString), SymbolId>,
    /// Symbols grouped by scope for iteration
    symbols_by_scope: HashMap<ScopeId, Vec<SymbolId>>,
    /// Symbols grouped by kind
    symbols_by_kind: HashMap<SymbolKind, Vec<SymbolId>>,
    /// Separate usage tracking (more cache-friendly than interior mutability)
    used_symbols: HashSet<SymbolId>,
    /// Statistics
    total_symbols: usize,
    /// Class hierarchy information indexed by symbol ID
    pub(crate) class_hierarchies: HashMap<SymbolId, ClassHierarchyInfo>,
    /// Type ID to Symbol ID mapping for hierarchy lookups
    type_to_symbol: HashMap<TypeId, SymbolId>,
    /// Symbol ID to Type ID reverse mapping for fast lookups
    symbol_to_type: HashMap<SymbolId, TypeId>,
    /// Cache for computed hierarchy queries
    supertype_cache: HashMap<SymbolId, HashSet<TypeId>>,
    /// Map from enum symbol to its variants
    enum_variants: HashMap<SymbolId, Vec<SymbolId>>,
    /// Enhanced symbol resolution cache
    symbol_cache: Rc<SymbolResolutionCache>,
}

impl SymbolTable {
    /// Create a new symbol table
    pub fn new() -> Self {
        Self {
            symbols_arena: TypedArena::new(),
            symbols_by_id: HashMap::new(),
            symbols_by_name: HashMap::new(),
            symbols_by_scope: HashMap::new(),
            symbols_by_kind: HashMap::new(),
            used_symbols: HashSet::new(),
            total_symbols: 0,
            class_hierarchies: HashMap::new(),
            type_to_symbol: HashMap::new(),
            symbol_to_type: HashMap::new(),
            supertype_cache: HashMap::new(),
            enum_variants: HashMap::new(),
            symbol_cache: Rc::new(SymbolResolutionCache::new(1000)),
        }
    }

    /// Create a new symbol table with estimated capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            symbols_arena: TypedArena::with_capacity(capacity),
            symbols_by_id: HashMap::with_capacity(capacity),
            symbols_by_name: HashMap::with_capacity(capacity),
            symbols_by_scope: HashMap::with_capacity(capacity / 10), // Estimate 10 symbols per scope
            symbols_by_kind: HashMap::with_capacity(SymbolKind::Metadata as usize + 1),
            used_symbols: HashSet::with_capacity(capacity / 2), // Estimate half of symbols get used
            total_symbols: 0,
            class_hierarchies: HashMap::with_capacity(capacity),
            type_to_symbol: HashMap::with_capacity(capacity),
            symbol_to_type: HashMap::with_capacity(capacity),
            supertype_cache: HashMap::with_capacity(capacity),
            enum_variants: HashMap::with_capacity(capacity / 20), // Estimate fewer enums
            symbol_cache: Rc::new(SymbolResolutionCache::with_sizes(capacity, capacity / 2)),
        }
    }

    /// Add a new symbol to the table
    pub fn add_symbol(&mut self, symbol: Symbol) -> &Symbol {
        let id = symbol.id;
        let scope_id = symbol.scope_id;
        let name = symbol.name;
        let kind = symbol.kind;

        // Store symbol in arena
        let symbol_ref = self.symbols_arena.alloc(symbol);

        // SAFETY: Arena lifetime extends beyond all references
        let static_ref: &'static Symbol = unsafe { std::mem::transmute(symbol_ref) };

        // Update indices
        self.symbols_by_id.insert(id, static_ref);
        self.symbols_by_name.insert((scope_id, name), id);

        self.symbols_by_scope
            .entry(scope_id)
            .or_insert_with(Vec::new)
            .push(id);

        self.symbols_by_kind
            .entry(kind)
            .or_insert_with(Vec::new)
            .push(id);

        self.total_symbols += 1;

        // Invalidate relevant cache entries
        self.symbol_cache.invalidate_scope(scope_id);
        self.symbol_cache.invalidate_name(name);

        // Invalidate cache entries for this symbol kind
        use super::symbol_cache::SymbolCacheKey;
        let kind_key = SymbolCacheKey::SymbolsByKind(kind);
        // Note: We could implement specific invalidation for single keys, but clearing is simpler for now

        static_ref
    }

    /// Register a type ID to symbol ID mapping
    pub fn register_type_symbol_mapping(&mut self, type_id: TypeId, symbol_id: SymbolId) {
        self.type_to_symbol.insert(type_id, symbol_id);
        self.symbol_to_type.insert(symbol_id, type_id);
    }

    /// Get symbol ID from a type ID
    pub fn get_symbol_from_type(&self, type_id: TypeId) -> Option<SymbolId> {
        self.type_to_symbol.get(&type_id).copied()
    }

    /// Get a symbol by its ID
    pub fn get_symbol(&self, id: SymbolId) -> Option<&Symbol> {
        self.symbols_by_id.get(&id).copied()
    }

    /// Get a mutable symbol by its ID
    pub fn get_symbol_mut(&mut self, id: SymbolId) -> Option<&mut Symbol> {
        self.symbols_by_id.get(&id).and_then(|&symbol_ptr| {
            // SAFETY: We maintain exclusive ownership of symbols in the arena
            // and this method has &mut self, ensuring exclusive access
            unsafe {
                let mutable_ptr = symbol_ptr as *const Symbol as *mut Symbol;
                mutable_ptr.as_mut()
            }
        })
    }

    pub fn update_symbol_type(&mut self, id: SymbolId, type_id: TypeId) -> bool {
        // We need to find the symbol and update it
        // Since we use an arena, we can't directly mutate, but we can use unsafe code
        // to update the type_id field which doesn't affect memory layout

        if let Some(&symbol_ref) = self.symbols_by_id.get(&id) {
            unsafe {
                // SAFETY: We're only modifying the type_id field which doesn't affect
                // the memory layout or any references. The symbol's identity remains the same.
                let symbol_ptr = symbol_ref as *const Symbol as *mut Symbol;
                (*symbol_ptr).type_id = type_id;
            }
            return true;
        }
        false
    }

    /// Add flags to a symbol
    pub fn add_symbol_flags(&mut self, id: SymbolId, flags: SymbolFlags) -> bool {
        if let Some(&symbol_ref) = self.symbols_by_id.get(&id) {
            unsafe {
                // SAFETY: We're only modifying the flags field which doesn't affect
                // the memory layout or any references.
                let symbol_ptr = symbol_ref as *const Symbol as *mut Symbol;
                (*symbol_ptr).flags = (*symbol_ptr).flags.union(flags);
            }
            return true;
        }
        false
    }

    /// Look up a symbol by name in a specific scope (with caching)
    pub fn lookup_symbol(&self, scope_id: ScopeId, name: InternedString) -> Option<&Symbol> {
        // Check cache first
        if let Some(cached_result) = self.symbol_cache.get(scope_id, name) {
            return cached_result.and_then(|id| self.get_symbol(id));
        }

        // Cache miss, perform lookup
        let symbol_id = self.symbols_by_name.get(&(scope_id, name)).copied();

        // Store result in cache (including None results to avoid repeated misses)
        self.symbol_cache.insert(scope_id, name, symbol_id);

        symbol_id.and_then(|id| self.get_symbol(id))
    }

    /// Remap a name in a scope to a different symbol.
    /// Used when import resolution discovers the correct symbol after initial registration.
    pub fn remap_symbol_in_scope(
        &mut self,
        scope_id: ScopeId,
        name: InternedString,
        new_symbol_id: SymbolId,
    ) {
        self.symbols_by_name.insert((scope_id, name), new_symbol_id);
        // Invalidate cache for this lookup
        self.symbol_cache.invalidate_scope(scope_id);
    }

    /// Get all symbols in a scope (with caching)
    pub fn symbols_in_scope(&self, scope_id: ScopeId) -> Vec<&Symbol> {
        use super::symbol_cache::SymbolCacheKey;

        let cache_key = SymbolCacheKey::SymbolsInScope(scope_id);

        // Check cache first
        if let Some(cached_ids) = self.symbol_cache.get_symbols(&cache_key) {
            return cached_ids
                .iter()
                .filter_map(|&id| self.get_symbol(id))
                .collect();
        }

        // Cache miss, perform lookup
        let symbol_ids = self
            .symbols_by_scope
            .get(&scope_id)
            .cloned()
            .unwrap_or_default();

        // Store result in cache
        self.symbol_cache
            .insert_symbols(cache_key, symbol_ids.clone());

        symbol_ids
            .iter()
            .filter_map(|&id| self.get_symbol(id))
            .collect()
    }

    /// Get all symbols of a specific kind (with caching)
    pub fn symbols_of_kind(&self, kind: SymbolKind) -> Vec<&Symbol> {
        use super::symbol_cache::SymbolCacheKey;

        let cache_key = SymbolCacheKey::SymbolsByKind(kind);

        // Check cache first
        if let Some(cached_ids) = self.symbol_cache.get_symbols(&cache_key) {
            return cached_ids
                .iter()
                .filter_map(|&id| self.get_symbol(id))
                .collect();
        }

        // Cache miss, perform lookup
        let symbol_ids = self.symbols_by_kind.get(&kind).cloned().unwrap_or_default();

        // Store result in cache
        self.symbol_cache
            .insert_symbols(cache_key, symbol_ids.clone());

        symbol_ids
            .iter()
            .filter_map(|&id| self.get_symbol(id))
            .collect()
    }

    /// Check if a name is already used in a scope
    pub fn is_name_used(&self, scope_id: ScopeId, name: InternedString) -> bool {
        self.symbols_by_name.contains_key(&(scope_id, name))
    }

    /// Add an alias name for an existing symbol in the same scope.
    /// This allows a symbol to be looked up by multiple names (e.g., both short and qualified names).
    pub fn add_symbol_alias(
        &mut self,
        symbol_id: SymbolId,
        scope_id: ScopeId,
        alias_name: InternedString,
    ) {
        // Only add if not already present
        if !self.symbols_by_name.contains_key(&(scope_id, alias_name)) {
            self.symbols_by_name
                .insert((scope_id, alias_name), symbol_id);
            // Invalidate cache for this name
            self.symbol_cache.clear();
        }
    }

    /// Get all symbols (for iteration)
    pub fn all_symbols(&self) -> impl Iterator<Item = &Symbol> {
        self.symbols_by_id.values().copied()
    }

    /// Get symbols matching a predicate
    pub fn find_symbols<F>(&self, predicate: F) -> Vec<&Symbol>
    where
        F: Fn(&Symbol) -> bool,
    {
        self.all_symbols()
            .filter(|symbol| predicate(symbol))
            .collect()
    }

    /// Get unused symbols
    pub fn unused_symbols(&self) -> Vec<&Symbol> {
        self.all_symbols()
            .filter(|symbol| !self.used_symbols.contains(&symbol.id))
            .collect()
    }

    /// Get exported symbols
    pub fn exported_symbols(&self) -> Vec<&Symbol> {
        self.find_symbols(|symbol| symbol.is_exported)
    }

    /// Get public symbols
    pub fn public_symbols(&self) -> Vec<&Symbol> {
        self.find_symbols(|symbol| symbol.visibility == Visibility::Public)
    }

    /// Mark a symbol as used
    pub fn mark_symbol_used(&mut self, id: SymbolId) {
        self.used_symbols.insert(id);
    }

    /// Check if a symbol is used
    pub fn is_symbol_used(&self, id: SymbolId) -> bool {
        self.used_symbols.contains(&id)
    }

    /// Mark multiple symbols as used at once
    pub fn mark_symbols_used(&mut self, ids: &[SymbolId]) {
        for &id in ids {
            self.used_symbols.insert(id);
        }
    }

    /// Check if a symbol represents a class
    pub fn is_class(&self, symbol_id: SymbolId) -> bool {
        self.get_symbol(symbol_id)
            .map(|s| matches!(s.kind, SymbolKind::Class))
            .unwrap_or(false)
    }

    /// Check if a symbol represents an interface
    pub fn is_interface(&self, symbol_id: SymbolId) -> bool {
        self.get_symbol(symbol_id)
            .map(|s| matches!(s.kind, SymbolKind::Interface))
            .unwrap_or(false)
    }

    /// Create a type parameter symbol with the given constraints
    pub fn create_type_parameter(
        &mut self,
        name: InternedString,
        constraints: Vec<super::type_checker::ConstraintKind>,
    ) -> SymbolId {
        // Generate a new symbol ID
        let symbol_id = SymbolId::from_raw(self.total_symbols as u32);

        // Create the type parameter symbol
        // We'll use an invalid TypeId for now since type parameters get resolved later
        let symbol = Symbol {
            id: symbol_id,
            name,
            kind: SymbolKind::TypeParameter,
            type_id: TypeId::invalid(), // Type parameters get their actual type during instantiation
            scope_id: ScopeId::invalid(), // Will be set when added to scope
            lifetime_id: LifetimeId::invalid(),
            visibility: Visibility::Private, // Type parameters are typically private to their scope
            mutability: Mutability::Immutable, // Type parameters are immutable
            definition_location: SourceLocation::unknown(), // Would be set by caller in full implementation
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

        // Add the symbol to the table
        self.add_symbol(symbol);

        symbol_id
    }

    /// Create a class symbol
    pub fn create_class(&mut self, name: InternedString) -> SymbolId {
        self.create_class_in_scope(name, ScopeId::invalid())
    }

    pub fn create_class_in_scope(&mut self, name: InternedString, scope_id: ScopeId) -> SymbolId {
        // Generate a new symbol ID
        let symbol_id = SymbolId::from_raw(self.total_symbols as u32);

        // Create the class symbol
        let symbol = Symbol {
            id: symbol_id,
            name,
            kind: SymbolKind::Class,
            type_id: TypeId::invalid(), // Will be set when type is created
            scope_id: scope_id,         // Set to the correct scope
            lifetime_id: LifetimeId::invalid(),
            visibility: Visibility::Public, // Classes are typically public by default
            mutability: Mutability::Immutable, // Class definitions are immutable
            definition_location: SourceLocation::unknown(), // Would be set by caller in full implementation
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

        // Add the symbol to the table
        self.add_symbol(symbol);

        symbol_id
    }

    /// Create a interface symbol
    pub fn create_interface(&mut self, name: InternedString) -> SymbolId {
        self.create_interface_in_scope(name, ScopeId::invalid())
    }

    pub fn create_interface_in_scope(
        &mut self,
        name: InternedString,
        scope_id: ScopeId,
    ) -> SymbolId {
        // Generate a new symbol ID
        let symbol_id = SymbolId::from_raw(self.total_symbols as u32);

        // Create the Interface symbol
        let symbol = Symbol {
            id: symbol_id,
            name,
            kind: SymbolKind::Interface,
            type_id: TypeId::invalid(), // Will be set when type is created
            scope_id: scope_id,         // Set to the correct scope
            lifetime_id: LifetimeId::invalid(),
            visibility: Visibility::Public,
            mutability: Mutability::Immutable,
            definition_location: SourceLocation::unknown(), // Would be set by caller in full implementation
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

        // Add the symbol to the table
        self.add_symbol(symbol);

        symbol_id
    }

    /// Create a type alias (typedef) symbol
    pub fn create_type_alias(&mut self, name: InternedString) -> SymbolId {
        self.create_type_alias_in_scope(name, ScopeId::invalid())
    }

    /// Create a type alias (typedef) symbol in a specific scope
    pub fn create_type_alias_in_scope(
        &mut self,
        name: InternedString,
        scope_id: ScopeId,
    ) -> SymbolId {
        let symbol_id = SymbolId::from_raw(self.total_symbols as u32);

        let symbol = Symbol {
            id: symbol_id,
            name,
            kind: SymbolKind::TypeAlias,
            type_id: TypeId::invalid(),
            scope_id,
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

        self.add_symbol(symbol);
        symbol_id
    }

    /// Create a function symbol
    pub fn create_function(&mut self, name: InternedString) -> SymbolId {
        self.create_function_in_scope(name, ScopeId::invalid())
    }

    /// Create a function symbol in a specific scope
    pub fn create_function_in_scope(
        &mut self,
        name: InternedString,
        scope_id: ScopeId,
    ) -> SymbolId {
        let symbol_id = SymbolId::from_raw(self.total_symbols as u32);

        let symbol = Symbol {
            id: symbol_id,
            name,
            kind: SymbolKind::Function,
            type_id: TypeId::invalid(),
            scope_id,
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

        self.add_symbol(symbol);
        symbol_id
    }

    /// Create a variable symbol
    pub fn create_variable(&mut self, name: InternedString) -> SymbolId {
        self.create_variable_in_scope(name, ScopeId::invalid())
    }

    /// Create a variable symbol in a specific scope
    pub fn create_variable_in_scope(
        &mut self,
        name: InternedString,
        scope_id: ScopeId,
    ) -> SymbolId {
        self.create_variable_with_type(name, scope_id, TypeId::invalid())
    }

    /// Create a variable symbol with a specific type
    pub fn create_variable_with_type(
        &mut self,
        name: InternedString,
        scope_id: ScopeId,
        type_id: TypeId,
    ) -> SymbolId {
        let symbol_id = SymbolId::from_raw(self.total_symbols as u32);

        let symbol = Symbol {
            id: symbol_id,
            name,
            kind: SymbolKind::Variable,
            type_id,
            scope_id,
            lifetime_id: LifetimeId::invalid(),
            visibility: Visibility::Public,
            mutability: Mutability::Mutable,
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

        self.add_symbol(symbol);
        symbol_id
    }

    /// Create a field symbol
    pub fn create_field(&mut self, name: InternedString) -> SymbolId {
        let symbol_id = SymbolId::from_raw(self.total_symbols as u32);

        let symbol = Symbol {
            id: symbol_id,
            name,
            kind: SymbolKind::Field,
            type_id: TypeId::invalid(),
            scope_id: ScopeId::invalid(),
            lifetime_id: LifetimeId::invalid(),
            visibility: Visibility::Public,
            mutability: Mutability::Mutable,
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

        self.add_symbol(symbol);
        symbol_id
    }

    /// Create an enum symbol
    pub fn create_enum(&mut self, name: InternedString) -> SymbolId {
        self.create_enum_in_scope(name, ScopeId::invalid())
    }

    pub fn create_enum_in_scope(&mut self, name: InternedString, scope_id: ScopeId) -> SymbolId {
        let symbol_id = SymbolId::from_raw(self.total_symbols as u32);

        let symbol = Symbol {
            id: symbol_id,
            name,
            kind: SymbolKind::Enum,
            type_id: TypeId::invalid(),
            scope_id: scope_id, // Set to the correct scope
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

        self.add_symbol(symbol);
        symbol_id
    }

    /// Create an abstract type symbol in a specific scope
    pub fn create_abstract_in_scope(
        &mut self,
        name: InternedString,
        scope_id: ScopeId,
    ) -> SymbolId {
        let symbol_id = SymbolId::from_raw(self.total_symbols as u32);

        let symbol = Symbol {
            id: symbol_id,
            name,
            kind: SymbolKind::Abstract,
            type_id: TypeId::invalid(),
            scope_id: scope_id, // Set to the correct scope
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

        self.add_symbol(symbol);
        symbol_id
    }

    /// Create an enum variant symbol linked to its parent enum
    pub fn create_enum_variant_in_scope(
        &mut self,
        name: InternedString,
        scope_id: ScopeId,
        parent_enum: SymbolId,
    ) -> SymbolId {
        let symbol_id = SymbolId::from_raw(self.total_symbols as u32);

        let symbol = Symbol {
            id: symbol_id,
            name,
            kind: SymbolKind::EnumVariant,
            type_id: TypeId::invalid(),
            scope_id: scope_id,
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

        self.add_symbol(symbol);

        // Store the relationship between variant and parent enum
        self.enum_variants
            .entry(parent_enum)
            .or_insert_with(Vec::new)
            .push(symbol_id);

        symbol_id
    }

    /// Find the parent enum symbol for a given enum constructor
    pub fn find_parent_enum_for_constructor(
        &self,
        constructor_symbol: SymbolId,
    ) -> Option<SymbolId> {
        for (enum_symbol, variants) in &self.enum_variants {
            if variants.contains(&constructor_symbol) {
                return Some(*enum_symbol);
            }
        }
        None
    }

    /// Get the variants of an enum
    pub fn get_enum_variants(&self, enum_symbol: SymbolId) -> Option<&Vec<SymbolId>> {
        self.enum_variants.get(&enum_symbol)
    }

    /// Get all used symbols
    pub fn used_symbols(&self) -> Vec<&Symbol> {
        self.all_symbols()
            .filter(|symbol| self.used_symbols.contains(&symbol.id))
            .collect()
    }

    /// Get symbol table statistics
    pub fn stats(&self) -> SymbolTableStats {
        let arena_stats = self.symbols_arena.stats();
        let total_bytes_allocated = arena_stats.total_bytes_allocated;
        SymbolTableStats {
            total_symbols: self.total_symbols,
            used_symbols: self.used_symbols.len(),
            unused_symbols: self.total_symbols - self.used_symbols.len(),
            symbols_by_kind: self
                .symbols_by_kind
                .iter()
                .map(|(&kind, symbols)| (kind, symbols.len()))
                .collect(),
            scopes_with_symbols: self.symbols_by_scope.len(),
            arena_stats,
            memory_usage: total_bytes_allocated
                + self.symbols_by_id.len() * std::mem::size_of::<(SymbolId, &Symbol)>()
                + self.symbols_by_name.len()
                    * std::mem::size_of::<((ScopeId, InternedString), SymbolId)>()
                + self.used_symbols.len() * std::mem::size_of::<SymbolId>(),
        }
    }

    /// Get the number of symbols
    pub fn len(&self) -> usize {
        self.total_symbols
    }

    /// Check if the symbol table is empty
    pub fn is_empty(&self) -> bool {
        self.total_symbols == 0
    }

    /// Get all direct subclasses of a class
    pub fn get_direct_subclasses(&self, class_id: SymbolId) -> Vec<SymbolId> {
        let mut subclasses = Vec::with_capacity(4); // Most classes have few direct subclasses

        // Get the type ID for this class
        let class_type = self.find_type_for_symbol(class_id);

        // Iterate through all class hierarchies to find subclasses
        for (&symbol_id, hierarchy) in &self.class_hierarchies {
            if let Some(superclass) = hierarchy.superclass {
                if Some(superclass) == class_type {
                    subclasses.push(symbol_id);
                }
            }
        }

        subclasses
    }

    /// Find type ID for a symbol (helper method)
    fn find_type_for_symbol(&self, symbol_id: SymbolId) -> Option<TypeId> {
        // Use O(1) reverse lookup
        self.symbol_to_type.get(&symbol_id).copied()
    }

    /// Get all interfaces implemented by a class (including inherited)
    pub fn get_all_interfaces(&self, class_id: SymbolId) -> Vec<SymbolId> {
        let mut interfaces = Vec::with_capacity(8); // Estimate for interface collection
        let mut seen = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(class_id);

        while let Some(current) = queue.pop_front() {
            if !seen.insert(current) {
                continue;
            }

            if let Some(hierarchy) = self.get_class_hierarchy(current) {
                // Add direct interfaces
                for &interface_type in &hierarchy.interfaces {
                    if let Some(interface_sym) = self.get_symbol_from_type(interface_type) {
                        if self.is_interface(interface_sym) {
                            interfaces.push(interface_sym);
                        }
                    }
                }

                // Process superclass
                if let Some(superclass) = hierarchy.superclass {
                    if let Some(super_sym) = self.get_symbol_from_type(superclass) {
                        queue.push_back(super_sym);
                    }
                }
            }
        }

        interfaces
    }

    /// Check if source class is subtype of target class
    pub fn is_class_subtype_of(&self, source: SymbolId, target: SymbolId) -> bool {
        if source == target {
            return true;
        }

        // Get target's type ID
        if let Some(target_type) = self.find_type_for_symbol(target) {
            let supertypes = self.compute_all_supertypes(source);
            return supertypes.contains(&target_type);
        }

        false
    }

    /// Check if a class implements an interface
    pub fn implements_interface(&self, class_id: SymbolId, interface_id: SymbolId) -> bool {
        if let Some(interface_type) = self.find_type_for_symbol(interface_id) {
            let supertypes = self.compute_all_supertypes(class_id);
            return supertypes.contains(&interface_type);
        }

        false
    }

    /// Validate that there are no cycles in the class hierarchy
    pub fn validate_no_inheritance_cycles(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::with_capacity(4); // Most checks produce few errors

        for &class_id in self.class_hierarchies.keys() {
            if self.has_inheritance_cycle(class_id) {
                if let Some(symbol) = self.get_symbol(class_id) {
                    errors.push(format!(
                        "Circular inheritance detected for class '{:?}'",
                        symbol.name
                    ));
                }
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Check if there's an inheritance cycle starting from the given class
    fn has_inheritance_cycle(&self, start: SymbolId) -> bool {
        let mut visited = HashSet::new();
        let mut current = start;

        loop {
            if !visited.insert(current) {
                // We've seen this before - cycle detected
                return true;
            }

            // Get superclass
            if let Some(hierarchy) = self.get_class_hierarchy(current) {
                if let Some(superclass) = hierarchy.superclass {
                    if let Some(super_sym) = self.get_symbol_from_type(superclass) {
                        current = super_sym;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            } else {
                break;
            }

            // Safety check
            if visited.len() > 1000 {
                return true;
            }
        }

        false
    }

    /// Get class hierarchy info (crate-visible for HIR lowering)
    pub(crate) fn get_class_hierarchy(&self, class_id: SymbolId) -> Option<&ClassHierarchyInfo> {
        self.class_hierarchies.get(&class_id)
    }

    /// Get class hierarchy info for TypeChecker
    pub fn get_class_info_for_type_checking(&self, class_id: SymbolId) -> Option<ClassTypeInfo> {
        let hierarchy = self.get_class_hierarchy(class_id)?;
        let symbol = self.get_symbol(class_id)?;

        Some(ClassTypeInfo {
            symbol_id: class_id,
            name: symbol.name,
            kind: symbol.kind.clone(),
            superclass: hierarchy.superclass,
            interfaces: hierarchy.interfaces.clone(),
            all_supertypes: self.compute_all_supertypes(class_id),
            depth: hierarchy.depth,
        })
    }

    // === Cache Management Methods ===

    /// Get symbol cache statistics
    pub fn cache_stats(&self) -> super::symbol_cache::CacheStats {
        self.symbol_cache.stats()
    }

    /// Get current cache sizes (symbol cache, multi-symbol cache)
    pub fn cache_sizes(&self) -> (usize, usize) {
        self.symbol_cache.sizes()
    }

    /// Clear the symbol cache (useful for testing or memory management)
    pub fn clear_cache(&self) {
        self.symbol_cache.clear();
    }

    /// Invalidate cache entries for a specific scope
    pub fn invalidate_scope_cache(&self, scope_id: ScopeId) {
        self.symbol_cache.invalidate_scope(scope_id);
    }

    /// Invalidate cache entries for a specific symbol name
    pub fn invalidate_name_cache(&self, name: InternedString) {
        self.symbol_cache.invalidate_name(name);
    }

    // === Qualified Name Methods ===

    /// Build a qualified name for a symbol by walking up the scope chain
    ///
    /// Returns a fully qualified name like "com.example.MyClass.myMethod"
    ///
    /// # Arguments
    /// * `symbol_id` - The symbol to build a qualified name for
    /// * `scope_tree` - The scope tree to walk for parent scopes
    /// * `interner` - String interner for resolving names
    ///
    /// # Returns
    /// The qualified name, or just the symbol name if no path can be built
    pub fn build_qualified_name(
        &self,
        symbol_id: SymbolId,
        scope_tree: &super::scopes::ScopeTree,
        interner: &super::StringInterner,
    ) -> InternedString {
        let symbol = match self.get_symbol(symbol_id) {
            Some(s) => s,
            None => return interner.intern("<unknown>"),
        };

        let mut path_parts = Vec::new();
        let mut current_scope_id = symbol.scope_id;

        // Walk up the scope chain collecting names
        while let Some(scope) = scope_tree.get_scope(current_scope_id) {
            // Include Package, Class, Interface, Enum, and Abstract scopes in the path
            match scope.kind {
                super::scopes::ScopeKind::Package
                | super::scopes::ScopeKind::Class
                | super::scopes::ScopeKind::Interface
                | super::scopes::ScopeKind::Enum
                | super::scopes::ScopeKind::Abstract => {
                    if let Some(name) = scope.name {
                        path_parts.push(name);
                    }
                }
                super::scopes::ScopeKind::Global => break,
                _ => {} // Skip function, block, loop scopes
            }

            // Move to parent scope
            match scope.parent_id {
                Some(parent_id) => current_scope_id = parent_id,
                None => break,
            }
        }

        // Reverse to get package.class.name order
        path_parts.reverse();

        // Add the symbol name itself
        path_parts.push(symbol.name);

        // Join with dots
        let qualified_name = path_parts
            .iter()
            .filter_map(|name| interner.get(*name))
            .collect::<Vec<_>>()
            .join(".");

        interner.intern(&qualified_name)
    }

    /// Update a symbol's qualified name field
    ///
    /// This should be called after creating a symbol to populate its qualified_name
    pub fn update_qualified_name(
        &mut self,
        symbol_id: SymbolId,
        scope_tree: &super::scopes::ScopeTree,
        interner: &super::StringInterner,
    ) -> bool {
        let qualified_name = self.build_qualified_name(symbol_id, scope_tree, interner);

        // Update the symbol's qualified_name field
        if let Some(&symbol_ref) = self.symbols_by_id.get(&symbol_id) {
            unsafe {
                // SAFETY: We're only modifying the qualified_name field which doesn't affect
                // memory layout or any references. The symbol's identity remains the same.
                let symbol_ptr = symbol_ref as *const Symbol as *mut Symbol;
                (*symbol_ptr).qualified_name = Some(qualified_name);
            }
            return true;
        }
        false
    }

    /// Lookup a symbol by its qualified name
    ///
    /// This performs a linear search, so consider caching if called frequently
    pub fn resolve_qualified_name(&self, qualified_name: InternedString) -> Option<SymbolId> {
        self.all_symbols()
            .find(|symbol| symbol.qualified_name == Some(qualified_name))
            .map(|symbol| symbol.id)
    }

    /// Get all symbols in a given package (by package name prefix)
    pub fn symbols_in_package(
        &self,
        package_prefix: InternedString,
        interner: &super::StringInterner,
    ) -> Vec<SymbolId> {
        let Some(package_str) = interner.get(package_prefix) else {
            return Vec::new();
        };
        let package_prefix_with_dot = format!("{}.", package_str);

        self.all_symbols()
            .filter(|symbol| {
                if let Some(qname) = symbol.qualified_name {
                    if let Some(qname_str) = interner.get(qname) {
                        return qname_str.starts_with(&package_prefix_with_dot)
                            || qname_str == package_str;
                    }
                }
                false
            })
            .map(|symbol| symbol.id)
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct ClassTypeInfo {
    pub symbol_id: SymbolId,
    pub name: InternedString,
    pub kind: SymbolKind,
    pub superclass: Option<TypeId>,
    pub interfaces: Vec<TypeId>,
    pub all_supertypes: HashSet<TypeId>,
    pub depth: usize,
}

impl Default for SymbolTable {
    fn default() -> Self {
        Self::new()
    }
}

impl ClassHierarchyRegistry for SymbolTable {
    fn register_class_hierarchy(&mut self, class_id: SymbolId, info: ClassHierarchyInfo) {
        // Clear any cached data for this class
        self.supertype_cache.remove(&class_id);

        // Store the hierarchy information
        self.class_hierarchies.insert(class_id, info);

        // Note: Since symbols are immutable in the arena, we can't update is_type
        // This should be set when the symbol is created
    }

    fn get_class_hierarchy(&self, class_id: SymbolId) -> Option<&ClassHierarchyInfo> {
        self.class_hierarchies.get(&class_id)
    }

    fn compute_all_supertypes(&self, class_id: SymbolId) -> HashSet<TypeId> {
        // Check cache first
        if let Some(cached) = self.supertype_cache.get(&class_id) {
            return cached.clone();
        }

        // Compute supertypes using BFS
        let mut supertypes = HashSet::new();
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();

        // Start with direct superclass and interfaces
        if let Some(hierarchy) = self.get_class_hierarchy(class_id) {
            if let Some(superclass) = hierarchy.superclass {
                queue.push_back(superclass);
                supertypes.insert(superclass);
            }

            for &interface in &hierarchy.interfaces {
                queue.push_back(interface);
                supertypes.insert(interface);
            }
        }

        // Process queue to find transitive supertypes
        while let Some(current_type) = queue.pop_front() {
            if !visited.insert(current_type) {
                continue; // Already processed
            }

            // Get symbol ID from type
            if let Some(symbol_id) = self.get_symbol_from_type(current_type) {
                if let Some(hierarchy) = self.get_class_hierarchy(symbol_id) {
                    // Add superclass
                    if let Some(superclass) = hierarchy.superclass {
                        if supertypes.insert(superclass) {
                            queue.push_back(superclass);
                        }
                    }

                    // Add interfaces
                    for &interface in &hierarchy.interfaces {
                        if supertypes.insert(interface) {
                            queue.push_back(interface);
                        }
                    }
                }
            }
        }

        // Note: Can't cache due to immutability, but could use interior mutability if needed
        supertypes
    }
}

/// Statistics about symbol table usage
#[derive(Debug, Clone)]
pub struct SymbolTableStats {
    pub total_symbols: usize,
    pub used_symbols: usize,
    pub unused_symbols: usize,
    pub symbols_by_kind: HashMap<SymbolKind, usize>,
    pub scopes_with_symbols: usize,
    pub arena_stats: super::ArenaStats,
    pub memory_usage: usize,
}

impl SymbolTableStats {
    /// Get the most common symbol kind
    pub fn most_common_kind(&self) -> Option<(SymbolKind, usize)> {
        self.symbols_by_kind
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(&kind, &count)| (kind, count))
    }

    /// Get memory usage per symbol
    pub fn memory_per_symbol(&self) -> f64 {
        if self.total_symbols > 0 {
            self.memory_usage as f64 / self.total_symbols as f64
        } else {
            0.0
        }
    }

    /// Get the percentage of symbols that are used
    pub fn usage_percentage(&self) -> f64 {
        if self.total_symbols > 0 {
            (self.used_symbols as f64 / self.total_symbols as f64) * 100.0
        } else {
            0.0
        }
    }

    /// Check if there are many unused symbols (potential dead code)
    pub fn has_excessive_unused_symbols(&self) -> bool {
        self.usage_percentage() < 70.0 && self.unused_symbols > 10
    }
}

#[cfg(test)]
mod tests {
    use crate::tast::{Scope, ScopeKind, ScopeLocation};

    use super::*;

    fn create_test_interner() -> StringInterner {
        StringInterner::new()
    }

    #[test]
    fn test_symbol_kind_properties() {
        assert!(SymbolKind::Class.is_type());
        assert!(SymbolKind::Variable.is_value());
        assert!(!SymbolKind::Class.is_value());
        assert!(!SymbolKind::Variable.is_type());

        assert!(SymbolKind::Variable.can_be_shadowed());
        assert!(!SymbolKind::Class.can_be_shadowed());

        assert!(SymbolKind::Function.requires_unique_name());
        assert!(!SymbolKind::Variable.requires_unique_name());
    }

    #[test]
    fn test_symbol_flags() {
        let mut flags = SymbolFlags::empty();
        assert!(flags.is_empty());

        flags.insert(SymbolFlags::STATIC);
        assert!(flags.contains(SymbolFlags::STATIC));
        assert!(!flags.contains(SymbolFlags::INLINE));

        flags.insert(SymbolFlags::INLINE);
        assert!(flags.contains(SymbolFlags::STATIC));
        assert!(flags.contains(SymbolFlags::INLINE));

        flags.remove(SymbolFlags::STATIC);
        assert!(!flags.contains(SymbolFlags::STATIC));
        assert!(flags.contains(SymbolFlags::INLINE));

        let combined = SymbolFlags::STATIC.union(SymbolFlags::FINAL);
        assert!(combined.contains(SymbolFlags::STATIC));
        assert!(combined.contains(SymbolFlags::FINAL));
    }

    #[test]
    fn test_source_location() {
        let loc = SourceLocation::new(1, 10, 5, 150);
        assert!(loc.is_valid());
        assert_eq!(loc.line, 10);
        assert_eq!(loc.column, 5);

        let unknown = SourceLocation::unknown();
        assert!(!unknown.is_valid());

        let default_loc = SourceLocation::default();
        assert!(!default_loc.is_valid());
    }

    #[test]
    fn test_symbol_creation() {
        let interner = create_test_interner();
        let name = interner.intern("test_var");

        let symbol = Symbol::variable(
            SymbolId::from_raw(1),
            name,
            TypeId::from_raw(1),
            ScopeId::from_raw(1),
            Mutability::Mutable,
            SourceLocation::new(1, 5, 10, 100),
        );

        assert_eq!(symbol.kind, SymbolKind::Variable);
        assert_eq!(symbol.mutability, Mutability::Mutable);
        assert_eq!(symbol.name, name);
        assert!(!symbol.is_used);
    }

    #[test]
    fn test_symbol_table_basic_operations() {
        let mut table = SymbolTable::new();
        let interner = create_test_interner();

        let name = interner.intern("test_function");
        let symbol = Symbol::function(
            SymbolId::from_raw(1),
            name,
            TypeId::from_raw(1),
            ScopeId::from_raw(1),
            Visibility::Public,
            SourceLocation::new(1, 10, 1, 200),
        );

        let added_symbol = table.add_symbol(symbol);
        assert_eq!(added_symbol.kind, SymbolKind::Function);
        assert_eq!(added_symbol.visibility, Visibility::Public);

        // Test lookup by ID
        let found = table.get_symbol(SymbolId::from_raw(1));
        assert!(found.is_some());
        assert_eq!(found.unwrap().kind, SymbolKind::Function);

        // Test lookup by name
        let found_by_name = table.lookup_symbol(ScopeId::from_raw(1), name);
        assert!(found_by_name.is_some());
        assert_eq!(found_by_name.unwrap().id, SymbolId::from_raw(1));

        assert_eq!(table.len(), 1);
        assert!(!table.is_empty());
    }

    #[test]
    fn test_symbol_table_scopes() {
        let mut table = SymbolTable::new();
        let interner = create_test_interner();

        let scope1 = ScopeId::from_raw(1);
        let scope2 = ScopeId::from_raw(2);

        // Add symbols to different scopes
        let name = interner.intern("x");

        let symbol1 = Symbol::variable(
            SymbolId::from_raw(1),
            name,
            TypeId::from_raw(1),
            scope1,
            Mutability::Mutable,
            SourceLocation::new(1, 5, 1, 50),
        );

        let symbol2 = Symbol::variable(
            SymbolId::from_raw(2),
            name,
            TypeId::from_raw(2),
            scope2,
            Mutability::Immutable,
            SourceLocation::new(1, 10, 1, 100),
        );

        table.add_symbol(symbol1);
        table.add_symbol(symbol2);

        // Same name in different scopes should be allowed
        let symbols_scope1 = table.symbols_in_scope(scope1);
        let symbols_scope2 = table.symbols_in_scope(scope2);

        assert_eq!(symbols_scope1.len(), 1);
        assert_eq!(symbols_scope2.len(), 1);

        assert_eq!(symbols_scope1[0].mutability, Mutability::Mutable);
        assert_eq!(symbols_scope2[0].mutability, Mutability::Immutable);
    }

    #[test]
    fn test_symbol_table_by_kind() {
        let mut table = SymbolTable::new();
        let interner = create_test_interner();

        // Add various symbol kinds
        let function_name = interner.intern("func");
        let class_name = interner.intern("MyClass");
        let var_name = interner.intern("x");

        let function_symbol = Symbol::function(
            SymbolId::from_raw(1),
            function_name,
            TypeId::from_raw(1),
            ScopeId::from_raw(1),
            Visibility::Public,
            SourceLocation::new(1, 1, 1, 1),
        );

        let class_symbol = Symbol::class(
            SymbolId::from_raw(2),
            class_name,
            TypeId::from_raw(2),
            ScopeId::from_raw(1),
            Visibility::Public,
            SourceLocation::new(1, 5, 1, 50),
        );

        let var_symbol = Symbol::variable(
            SymbolId::from_raw(3),
            var_name,
            TypeId::from_raw(3),
            ScopeId::from_raw(1),
            Mutability::Mutable,
            SourceLocation::new(1, 10, 1, 100),
        );

        table.add_symbol(function_symbol);
        table.add_symbol(class_symbol);
        table.add_symbol(var_symbol);

        let functions = table.symbols_of_kind(SymbolKind::Function);
        let classes = table.symbols_of_kind(SymbolKind::Class);
        let variables = table.symbols_of_kind(SymbolKind::Variable);

        assert_eq!(functions.len(), 1);
        assert_eq!(classes.len(), 1);
        assert_eq!(variables.len(), 1);

        assert_eq!(functions[0].name, function_name);
        assert_eq!(classes[0].name, class_name);
        assert_eq!(variables[0].name, var_name);
    }

    #[test]
    fn test_symbol_usage_tracking() {
        let mut table = SymbolTable::new();
        let interner = create_test_interner();

        let name = interner.intern("test_var");
        let symbol = Symbol::variable(
            SymbolId::from_raw(1),
            name,
            TypeId::from_raw(1),
            ScopeId::from_raw(1),
            Mutability::Mutable,
            SourceLocation::new(1, 5, 1, 50),
        );

        table.add_symbol(symbol);

        // Initially unused
        let unused = table.unused_symbols();
        assert_eq!(unused.len(), 1);
        assert!(!table.is_symbol_used(SymbolId::from_raw(1)));

        // Mark as used
        table.mark_symbol_used(SymbolId::from_raw(1));

        let unused_after = table.unused_symbols();
        assert_eq!(unused_after.len(), 0);
        assert!(table.is_symbol_used(SymbolId::from_raw(1)));

        let used_symbols = table.used_symbols();
        assert_eq!(used_symbols.len(), 1);
        assert_eq!(used_symbols[0].id, SymbolId::from_raw(1));
    }

    #[test]
    fn test_batch_symbol_usage() {
        let mut table = SymbolTable::new();
        let interner = create_test_interner();

        // Add multiple symbols
        for i in 1..=5 {
            let name = interner.intern(&format!("var_{}", i));
            let symbol = Symbol::variable(
                SymbolId::from_raw(i),
                name,
                TypeId::from_raw(i),
                ScopeId::from_raw(1),
                Mutability::Mutable,
                SourceLocation::new(1, i as u32, 1, i as u32 * 10),
            );
            table.add_symbol(symbol);
        }

        // Mark multiple symbols as used
        let ids_to_mark = [
            SymbolId::from_raw(1),
            SymbolId::from_raw(3),
            SymbolId::from_raw(5),
        ];
        table.mark_symbols_used(&ids_to_mark);

        // Check usage
        assert!(table.is_symbol_used(SymbolId::from_raw(1)));
        assert!(!table.is_symbol_used(SymbolId::from_raw(2)));
        assert!(table.is_symbol_used(SymbolId::from_raw(3)));
        assert!(!table.is_symbol_used(SymbolId::from_raw(4)));
        assert!(table.is_symbol_used(SymbolId::from_raw(5)));

        let used = table.used_symbols();
        let unused = table.unused_symbols();

        assert_eq!(used.len(), 3);
        assert_eq!(unused.len(), 2);
    }

    #[test]
    fn test_symbol_table_name_conflicts() {
        let mut table = SymbolTable::new();
        let interner = create_test_interner();

        let name = interner.intern("conflict");
        let scope = ScopeId::from_raw(1);

        // Add first symbol
        let symbol1 = Symbol::variable(
            SymbolId::from_raw(1),
            name,
            TypeId::from_raw(1),
            scope,
            Mutability::Mutable,
            SourceLocation::new(1, 5, 1, 50),
        );

        table.add_symbol(symbol1);
        assert!(table.is_name_used(scope, name));

        // Check that we can detect conflicts
        assert!(table.is_name_used(scope, name));

        // Different scope should be fine
        let different_scope = ScopeId::from_raw(2);
        assert!(!table.is_name_used(different_scope, name));
    }

    #[test]
    fn test_symbol_table_stats() {
        let mut table = SymbolTable::with_capacity(100);
        let interner = create_test_interner();

        // Add various symbols
        for i in 0..50 {
            let name = interner.intern(&format!("symbol_{}", i));
            let kind = if i % 3 == 0 {
                SymbolKind::Function
            } else if i % 3 == 1 {
                SymbolKind::Variable
            } else {
                SymbolKind::Class
            };

            let symbol = Symbol::new(
                SymbolId::from_raw(i),
                name,
                kind,
                TypeId::from_raw(i),
                ScopeId::from_raw(i / 10),
                SourceLocation::new(1, i as u32 + 1, 1, i as u32 * 10),
            );

            table.add_symbol(symbol);
        }

        // Mark some symbols as used
        for i in 0..25 {
            table.mark_symbol_used(SymbolId::from_raw(i));
        }

        let stats = table.stats();
        assert_eq!(stats.total_symbols, 50);
        assert_eq!(stats.used_symbols, 25);
        assert_eq!(stats.unused_symbols, 25);
        assert!(stats.symbols_by_kind.len() >= 3);
        assert!(stats.memory_usage > 0);

        let most_common = stats.most_common_kind();
        assert!(most_common.is_some());

        assert!(stats.memory_per_symbol() > 0.0);
        assert_eq!(stats.usage_percentage(), 50.0);
        assert!(stats.has_excessive_unused_symbols()); // 50% usage is below 70% threshold
    }

    #[test]
    fn test_symbol_predicates() {
        let interner = create_test_interner();
        let name = interner.intern("test_symbol");

        let mut symbol = Symbol::function(
            SymbolId::from_raw(1),
            name,
            TypeId::from_raw(1),
            ScopeId::from_raw(1),
            Visibility::Public,
            SourceLocation::new(1, 5, 1, 50),
        );

        // Test flag operations
        symbol.flags.insert(SymbolFlags::STATIC);
        assert!(symbol.is_static());
        assert!(!symbol.is_inline());

        symbol.flags.insert(SymbolFlags::FINAL);
        assert!(symbol.is_final());

        symbol.flags.insert(SymbolFlags::ABSTRACT);
        assert!(symbol.is_abstract());
    }

    #[test]
    fn test_symbol_display() {
        let interner = create_test_interner();
        let name = interner.intern("display_test");

        let symbol = Symbol::function(
            SymbolId::from_raw(42),
            name,
            TypeId::from_raw(1),
            ScopeId::from_raw(1),
            Visibility::Public,
            SourceLocation::new(1, 10, 5, 100),
        );

        let display_str = symbol.to_string();
        assert!(display_str.contains("public"));
        assert!(display_str.contains("function"));
        assert!(display_str.contains("42"));

        let description = symbol.description(&interner);
        assert!(description.contains("display_test"));
        assert!(description.contains("public"));
        assert!(description.contains("function"));
    }

    #[test]
    fn test_symbol_table_with_hierarchy() {
        let scope_id = ScopeId::from_raw(1);
        let mut scope = Scope::new(
            scope_id,
            ScopeKind::Global,
            None,
            ScopeLocation::unknown(),
            0,
        );
        let mut symbol_table = SymbolTable::new();
        let interner = StringInterner::new();

        // Create class symbols
        let object_name = interner.intern("Object");
        let animal_name = interner.intern("Animal");
        let dog_name = interner.intern("Dog");

        let object_sym = symbol_table.create_class(object_name);
        scope.add_symbol(object_sym, object_name);

        let animal_sym = symbol_table.create_class(animal_name);
        scope.add_symbol(animal_sym, object_name);
        let dog_sym = symbol_table.create_class(dog_name);
        scope.add_symbol(dog_sym, object_name);

        // Create type IDs (in real usage, these would come from TypeTable)
        let object_type = TypeId::from_raw(1);
        let animal_type = TypeId::from_raw(2);
        let dog_type = TypeId::from_raw(3);

        // Register type mappings
        symbol_table.register_type_symbol_mapping(object_type, object_sym);
        symbol_table.register_type_symbol_mapping(animal_type, animal_sym);
        symbol_table.register_type_symbol_mapping(dog_type, dog_sym);

        // Register hierarchies
        symbol_table.register_class_hierarchy(
            object_sym,
            ClassHierarchyInfo {
                superclass: None,
                interfaces: vec![],
                all_supertypes: HashSet::new(),
                depth: 0,
                is_final: false,
                is_abstract: false,
                is_extern: false,
                is_interface: false,
                sealed_to: None,
            },
        );

        symbol_table.register_class_hierarchy(
            animal_sym,
            ClassHierarchyInfo {
                superclass: Some(object_type),
                interfaces: vec![],
                all_supertypes: [object_type].into_iter().collect(),
                depth: 1,
                is_final: false,
                is_abstract: false,
                is_extern: false,
                is_interface: false,
                sealed_to: None,
            },
        );

        symbol_table.register_class_hierarchy(
            dog_sym,
            ClassHierarchyInfo {
                superclass: Some(animal_type),
                interfaces: vec![],
                all_supertypes: [object_type, animal_type].into_iter().collect(),
                depth: 2,
                is_final: false,
                is_abstract: false,
                is_extern: false,
                is_interface: false,
                sealed_to: None,
            },
        );

        // Test hierarchy queries
        assert!(symbol_table.is_class(dog_sym));
        assert!(!symbol_table.is_interface(dog_sym));

        let dog_supertypes = symbol_table.compute_all_supertypes(dog_sym);
        assert!(dog_supertypes.contains(&animal_type));
        assert!(dog_supertypes.contains(&object_type));

        let animal_subclasses = symbol_table.get_direct_subclasses(animal_sym);
        assert_eq!(animal_subclasses, vec![dog_sym]);

        // Test cycle detection
        assert!(symbol_table.validate_no_inheritance_cycles().is_ok());
    }
}
