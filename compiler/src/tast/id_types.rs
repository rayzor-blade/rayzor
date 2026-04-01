//! Core ID Types for TAST System
//!
//! This module provides type-safe, efficient identifier types used throughout
//! the Typed AST system. Each ID type is a lightweight wrapper around u32
//! that prevents mixing up different kinds of identifiers.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::num::NonZeroU32;

/// Trait for ID types that can be created and validated
pub trait IdType: Copy + Clone + PartialEq + Eq + std::hash::Hash + fmt::Debug {
    /// Create a new ID from a raw u32 value
    fn from_raw(raw: u32) -> Self;

    /// Get the raw u32 value of this ID
    fn as_raw(self) -> u32;

    /// Check if this ID is valid (not a sentinel value)
    fn is_valid(self) -> bool;

    /// Get an invalid/null sentinel value
    fn invalid() -> Self;

    /// Create the first valid ID (typically used for ID generators)
    fn first() -> Self {
        Self::from_raw(0)
    }

    /// Get the next ID in sequence
    fn next(self) -> Self {
        Self::from_raw(self.as_raw().wrapping_add(1))
    }
}

/// Macro to define ID types with consistent behavior
macro_rules! define_id_type {
    (
        $(#[$meta:meta])*
        $name:ident
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
        pub struct $name(pub (crate) u32);

        impl $name {
            /// Create a new ID from a raw u32 value
            pub const fn from_raw(raw: u32) -> Self {
                Self(raw)
            }

            /// Get the raw u32 value of this ID
            pub const fn as_raw(self) -> u32 {
                self.0
            }

            /// Check if this ID is valid (not the sentinel value)
            pub const fn is_valid(self) -> bool {
                self.0 != u32::MAX
            }

            /// Get an invalid/null sentinel value
            pub const fn invalid() -> Self {
                Self(u32::MAX)
            }

            /// Create the first valid ID
            pub const fn first() -> Self {
                Self(0)
            }

            /// Get the next ID in sequence
            pub const fn next(self) -> Self {
                Self(self.0.wrapping_add(1))
            }

            /// Create from NonZeroU32 for guaranteed valid IDs
            pub const fn from_non_zero(id: NonZeroU32) -> Self {
                // NonZeroU32 guarantees the value is not 0, and we use u32::MAX as invalid
                // So we need to map NonZeroU32 values to valid range [0, u32::MAX)
                Self(id.get().wrapping_sub(1))
            }

            /// Convert to NonZeroU32 if valid
            pub const fn to_non_zero(self) -> Option<NonZeroU32> {
                if self.is_valid() {
                    // Map valid range [0, u32::MAX) to [1, u32::MAX]
                    NonZeroU32::new(self.0.wrapping_add(1))
                } else {
                    None
                }
            }
        }

        impl IdType for $name {
            fn from_raw(raw: u32) -> Self {
                Self::from_raw(raw)
            }

            fn as_raw(self) -> u32 {
                self.as_raw()
            }

            fn is_valid(self) -> bool {
                self.is_valid()
            }

            fn invalid() -> Self {
                Self::invalid()
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::invalid()
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                if self.is_valid() {
                    write!(f, "{}({})", stringify!($name), self.0)
                } else {
                    write!(f, "{}(<invalid>)", stringify!($name))
                }
            }
        }

        impl From<u32> for $name {
            fn from(raw: u32) -> Self {
                Self::from_raw(raw)
            }
        }

        impl From<$name> for u32 {
            fn from(id: $name) -> u32 {
                id.as_raw()
            }
        }
    };
}

// Define all our core ID types
define_id_type! {
    /// Unique identifier for symbols (variables, functions, classes, etc.)
    ///
    /// Used throughout the symbol table and name resolution system to
    /// efficiently reference symbols without string comparisons.
    SymbolId
}

define_id_type! {
    /// Unique identifier for types in the type system
    ///
    /// Used for type checking, generic instantiation, and type-related
    /// analysis throughout the compilation pipeline.
    TypeId
}

define_id_type! {
    /// Unique identifier for scopes in the scope tree
    ///
    /// Used for lexical scoping, name resolution, and lifetime analysis.
    /// Each scope represents a region where names can be declared.
    ScopeId
}

define_id_type! {
    /// Unique identifier for lifetimes in ownership analysis
    ///
    /// Used for tracking object lifetimes, borrow checking, and ensuring
    /// memory safety in the ownership system.
    LifetimeId
}

define_id_type! {
    /// Unique identifier for expressions in the AST
    ///
    /// Used for cross-referencing expressions, building control/data flow
    /// graphs, and optimization analysis.
    ExpressionId
}

define_id_type! {
    /// Unique identifier for statements in the AST
    ///
    /// Used for control flow analysis and statement-level optimizations.
    StatementId
}

define_id_type! {
    /// Unique identifier for basic blocks in control flow graphs
    ///
    /// Used during HIR/MIR lowering for control flow analysis and optimization.
    BlockId
}

define_id_type! {
    /// Unique identifier for data flow nodes in DFG
    DataFlowNodeId
}

define_id_type! {
    /// Unique identifier for call sites in call graph
    CallSiteId
}

define_id_type! {
    /// Unique identifier for SSA variables
    SsaVariableId
}

define_id_type! {
    /// Unique identifier for borrow edges
    BorrowEdgeId
}

define_id_type! {
    /// Unique identifier for move edges
   MoveEdgeId
}

define_id_type! {
    SemanticGraphId
}

/// Generator for creating unique IDs of a specific type
///
/// Provides thread-safe ID generation with overflow protection.
#[derive(Debug)]
pub struct IdGenerator<T: IdType> {
    next_id: std::sync::atomic::AtomicU32,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: IdType> IdGenerator<T> {
    /// Create a new ID generator starting from the first valid ID
    pub const fn new() -> Self {
        Self {
            next_id: std::sync::atomic::AtomicU32::new(0),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create a new ID generator starting from a specific ID
    pub const fn with_start(start_id: u32) -> Self {
        Self {
            next_id: std::sync::atomic::AtomicU32::new(start_id),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Generate the next unique ID
    ///
    /// This is thread-safe and can be called from multiple threads concurrently.
    /// Panics if we run out of valid IDs (after 2^32 - 2 allocations).
    pub fn next(&self) -> T {
        let raw_id = self
            .next_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Check for overflow (we reserve u32::MAX as invalid sentinel)
        if raw_id == u32::MAX {
            panic!(
                "ID generator overflow: exhausted all valid IDs for {}",
                std::any::type_name::<T>()
            );
        }

        T::from_raw(raw_id)
    }

    /// Peek at the next ID that would be generated without consuming it
    pub fn peek_next(&self) -> T {
        let raw_id = self.next_id.load(std::sync::atomic::Ordering::Relaxed);
        T::from_raw(raw_id)
    }

    /// Get the number of IDs generated so far
    pub fn count(&self) -> u32 {
        self.next_id.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Reset the generator to start from the beginning
    ///
    /// # Safety
    /// This should only be called when you're sure no existing IDs are in use.
    pub unsafe fn reset(&self) {
        self.next_id.store(0, std::sync::atomic::Ordering::Relaxed);
    }

    /// Reserve a block of IDs and return the starting ID
    ///
    /// Useful for allocating multiple related IDs atomically.
    pub fn reserve_block(&self, count: u32) -> T {
        if count == 0 {
            return self.peek_next();
        }

        let start_id = self
            .next_id
            .fetch_add(count, std::sync::atomic::Ordering::Relaxed);

        // Check for overflow
        if start_id > u32::MAX - count {
            panic!("ID generator overflow: cannot reserve {} IDs", count);
        }

        T::from_raw(start_id)
    }
}

impl<T: IdType> Default for IdGenerator<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Collection of all ID generators used in the TAST system
///
/// Provides centralized ID generation for consistent numbering across
/// the entire compilation process.
#[derive(Debug)]
pub struct TastIdGenerators {
    pub symbols: IdGenerator<SymbolId>,
    pub types: IdGenerator<TypeId>,
    pub scopes: IdGenerator<ScopeId>,
    pub lifetimes: IdGenerator<LifetimeId>,
    pub expressions: IdGenerator<ExpressionId>,
    pub statements: IdGenerator<StatementId>,
    pub blocks: IdGenerator<BlockId>,
}

impl TastIdGenerators {
    /// Create a new set of ID generators
    pub const fn new() -> Self {
        Self {
            symbols: IdGenerator::new(),
            types: IdGenerator::new(),
            scopes: IdGenerator::new(),
            lifetimes: IdGenerator::new(),
            expressions: IdGenerator::new(),
            statements: IdGenerator::new(),
            blocks: IdGenerator::new(),
        }
    }

    /// Get statistics about ID usage across all generators
    pub fn stats(&self) -> IdGeneratorStats {
        IdGeneratorStats {
            symbols_generated: self.symbols.count(),
            types_generated: self.types.count(),
            scopes_generated: self.scopes.count(),
            lifetimes_generated: self.lifetimes.count(),
            expressions_generated: self.expressions.count(),
            statements_generated: self.statements.count(),
            blocks_generated: self.blocks.count(),
        }
    }

    /// Reset all generators (unsafe - see IdGenerator::reset)
    pub unsafe fn reset_all(&self) {
        self.symbols.reset();
        self.types.reset();
        self.scopes.reset();
        self.lifetimes.reset();
        self.expressions.reset();
        self.statements.reset();
        self.blocks.reset();
    }
}

impl Default for TastIdGenerators {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about ID generation usage
#[derive(Debug, Clone)]
pub struct IdGeneratorStats {
    pub symbols_generated: u32,
    pub types_generated: u32,
    pub scopes_generated: u32,
    pub lifetimes_generated: u32,
    pub expressions_generated: u32,
    pub statements_generated: u32,
    pub blocks_generated: u32,
}

impl IdGeneratorStats {
    /// Get the total number of IDs generated across all types
    pub fn total_ids(&self) -> u64 {
        self.symbols_generated as u64
            + self.types_generated as u64
            + self.scopes_generated as u64
            + self.lifetimes_generated as u64
            + self.expressions_generated as u64
            + self.statements_generated as u64
            + self.blocks_generated as u64
    }

    /// Check if any generator is getting close to overflow
    pub fn check_overflow_risk(&self) -> Vec<String> {
        let mut warnings = Vec::new();
        let threshold = u32::MAX - 1000; // Warn when within 1000 of overflow

        if self.symbols_generated > threshold {
            warnings.push("SymbolId generator near overflow".to_string());
        }
        if self.types_generated > threshold {
            warnings.push("TypeId generator near overflow".to_string());
        }
        if self.scopes_generated > threshold {
            warnings.push("ScopeId generator near overflow".to_string());
        }
        if self.lifetimes_generated > threshold {
            warnings.push("LifetimeId generator near overflow".to_string());
        }
        if self.expressions_generated > threshold {
            warnings.push("ExpressionId generator near overflow".to_string());
        }
        if self.statements_generated > threshold {
            warnings.push("StatementId generator near overflow".to_string());
        }
        if self.blocks_generated > threshold {
            warnings.push("BlockId generator near overflow".to_string());
        }

        warnings
    }
}

/// Convenience functions for working with ID collections
pub mod collections {
    use super::*;
    use std::collections::{BTreeMap, BTreeSet};

    /// Fast hash map specialized for ID keys
    pub type IdMap<K, V> = BTreeMap<K, V>;

    /// Fast hash set specialized for ID values
    pub type IdSet<T> = BTreeSet<T>;

    /// Create a new ID map with reasonable default capacity
    pub fn new_id_map<K: IdType, V>() -> IdMap<K, V> {
        BTreeMap::new()
    }

    /// Create a new ID set with reasonable default capacity
    pub fn new_id_set<T: IdType>() -> IdSet<T> {
        BTreeSet::new()
    }

    /// Create a new ID map with specific capacity
    pub fn new_id_map_with_capacity<K: IdType, V>(capacity: usize) -> IdMap<K, V> {
        BTreeMap::new()
    }

    /// Create a new ID set with specific capacity
    pub fn new_id_set_with_capacity<T: IdType>(capacity: usize) -> IdSet<T> {
        BTreeSet::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_id_basic_operations() {
        let id1 = SymbolId::from_raw(42);
        let id2 = SymbolId::from_raw(42);
        let id3 = SymbolId::from_raw(43);

        // Basic equality and inequality
        assert_eq!(id1, id2);
        assert_ne!(id1, id3);

        // Raw value access
        assert_eq!(id1.as_raw(), 42);
        assert_eq!(id3.as_raw(), 43);

        // Validity checks
        assert!(id1.is_valid());
        assert!(id2.is_valid());
        assert!(id3.is_valid());

        let invalid = SymbolId::invalid();
        assert!(!invalid.is_valid());
        assert_eq!(invalid.as_raw(), u32::MAX);
    }

    #[test]
    fn test_id_ordering() {
        let id1 = TypeId::from_raw(1);
        let id2 = TypeId::from_raw(2);
        let id3 = TypeId::from_raw(3);

        assert!(id1 < id2);
        assert!(id2 < id3);
        assert!(id1 < id3);

        // Test sorting
        let mut ids = vec![id3, id1, id2];
        ids.sort();
        assert_eq!(ids, vec![id1, id2, id3]);
    }

    #[test]
    fn test_id_hashing() {
        let id1 = ScopeId::from_raw(100);
        let id2 = ScopeId::from_raw(100);
        let id3 = ScopeId::from_raw(101);

        let mut set = BTreeSet::new();
        set.insert(id1);
        set.insert(id2); // Should not increase size (same as id1)
        set.insert(id3);

        assert_eq!(set.len(), 2);
        assert!(set.contains(&id1));
        assert!(set.contains(&id2));
        assert!(set.contains(&id3));
    }

    #[test]
    fn test_id_display() {
        let valid_id = LifetimeId::from_raw(42);
        let invalid_id = LifetimeId::invalid();

        assert_eq!(format!("{}", valid_id), "LifetimeId(42)");
        assert_eq!(format!("{}", invalid_id), "LifetimeId(<invalid>)");
        assert_eq!(format!("{:?}", valid_id), "LifetimeId(42)");
    }

    #[test]
    fn test_id_conversions() {
        let raw_value = 123u32;
        let id = ExpressionId::from(raw_value);
        let converted_back: u32 = id.into();

        assert_eq!(converted_back, raw_value);
        assert_eq!(id.as_raw(), raw_value);
    }

    #[test]
    fn test_id_sequence_operations() {
        let first = StatementId::first();
        assert_eq!(first.as_raw(), 0);

        let second = first.next();
        assert_eq!(second.as_raw(), 1);

        let third = second.next();
        assert_eq!(third.as_raw(), 2);
    }

    #[test]
    fn test_non_zero_conversions() {
        // Valid ID conversion
        let id = BlockId::from_raw(42);
        let non_zero = id.to_non_zero().unwrap();
        let round_trip = BlockId::from_non_zero(non_zero);
        assert_eq!(id, round_trip);

        // Invalid ID conversion
        let invalid = BlockId::invalid();
        assert_eq!(invalid.to_non_zero(), None);

        // First ID special case
        let first = BlockId::first();
        let non_zero_first = first.to_non_zero().unwrap();
        assert_eq!(non_zero_first.get(), 1);
        let round_trip_first = BlockId::from_non_zero(non_zero_first);
        assert_eq!(first, round_trip_first);
    }

    #[test]
    fn test_id_generator_basic() {
        let generator = IdGenerator::<SymbolId>::new();

        let id1 = generator.next();
        let id2 = generator.next();
        let id3 = generator.next();

        assert_eq!(id1.as_raw(), 0);
        assert_eq!(id2.as_raw(), 1);
        assert_eq!(id3.as_raw(), 2);

        assert_eq!(generator.count(), 3);

        let next_peek = generator.peek_next();
        assert_eq!(next_peek.as_raw(), 3);

        // Peek shouldn't consume
        let id4 = generator.next();
        assert_eq!(id4.as_raw(), 3);
    }

    #[test]
    fn test_id_generator_with_start() {
        let generator = IdGenerator::<TypeId>::with_start(100);

        let id1 = generator.next();
        let id2 = generator.next();

        assert_eq!(id1.as_raw(), 100);
        assert_eq!(id2.as_raw(), 101);
        assert_eq!(generator.count(), 102);
    }

    #[test]
    fn test_id_generator_block_reservation() {
        let generator = IdGenerator::<ScopeId>::new();

        // Reserve a block of 5 IDs
        let start_id = generator.reserve_block(5);
        assert_eq!(start_id.as_raw(), 0);
        assert_eq!(generator.count(), 5);

        // Next single ID should come after the block
        let next_id = generator.next();
        assert_eq!(next_id.as_raw(), 5);

        // Reserve empty block (should be no-op except returning current next)
        let empty_start = generator.reserve_block(0);
        assert_eq!(empty_start.as_raw(), 6);
        assert_eq!(generator.count(), 6); // Should not change
    }

    #[test]
    fn test_id_generator_thread_safety() {
        let generator = Arc::new(IdGenerator::<LifetimeId>::new());
        let mut handles = vec![];

        // Multiple threads generating IDs concurrently
        for _ in 0..4 {
            let gen = Arc::clone(&generator);
            let handle = thread::spawn(move || {
                let mut ids = Vec::new();
                for _ in 0..100 {
                    ids.push(gen.next());
                }
                ids
            });
            handles.push(handle);
        }

        let mut all_ids = Vec::new();
        for handle in handles {
            let thread_ids = handle.join().unwrap();
            all_ids.extend(thread_ids);
        }

        // Verify all IDs are unique
        let mut raw_ids: Vec<u32> = all_ids.iter().map(|id| id.as_raw()).collect();
        raw_ids.sort();

        for i in 0..raw_ids.len() {
            assert_eq!(raw_ids[i], i as u32);
        }

        assert_eq!(generator.count(), 400);
    }

    #[test]
    fn test_tast_id_generators() {
        let generators = TastIdGenerators::new();

        let symbol = generators.symbols.next();
        let type_id = generators.types.next();
        let scope = generators.scopes.next();
        let lifetime = generators.lifetimes.next();

        assert_eq!(symbol.as_raw(), 0);
        assert_eq!(type_id.as_raw(), 0);
        assert_eq!(scope.as_raw(), 0);
        assert_eq!(lifetime.as_raw(), 0);

        let stats = generators.stats();
        assert_eq!(stats.symbols_generated, 1);
        assert_eq!(stats.types_generated, 1);
        assert_eq!(stats.scopes_generated, 1);
        assert_eq!(stats.lifetimes_generated, 1);
        assert_eq!(stats.total_ids(), 4);
    }

    #[test]
    fn test_id_generator_stats() {
        let generators = TastIdGenerators::new();

        // Generate some IDs
        for _ in 0..10 {
            generators.symbols.next();
            generators.types.next();
        }

        let stats = generators.stats();
        assert_eq!(stats.symbols_generated, 10);
        assert_eq!(stats.types_generated, 10);
        assert_eq!(stats.total_ids(), 20);

        // Test overflow warnings
        let warnings = stats.check_overflow_risk();
        assert!(warnings.is_empty()); // Should be no warnings with small numbers
    }

    #[test]
    fn test_different_id_types_are_distinct() {
        // This test ensures we can't accidentally mix up different ID types
        let symbol_id = SymbolId::from_raw(42);
        let type_id = TypeId::from_raw(42);

        // These should be different types even with same raw value
        // This test mainly verifies the type system prevents mixing
        assert_eq!(symbol_id.as_raw(), type_id.as_raw()); // Same raw value
                                                          // But they're different types, so this wouldn't compile:
                                                          // assert_eq!(symbol_id, type_id); // Compile error - good!
    }

    #[test]
    fn test_id_default_values() {
        let symbol_id: SymbolId = Default::default();
        let type_id: TypeId = Default::default();

        assert!(!symbol_id.is_valid());
        assert!(!type_id.is_valid());
        assert_eq!(symbol_id, SymbolId::invalid());
        assert_eq!(type_id, TypeId::invalid());
    }

    #[test]
    fn test_collections_helpers() {
        let mut symbol_map = collections::new_id_map::<SymbolId, String>();
        let mut type_set = collections::new_id_set::<TypeId>();

        let sym1 = SymbolId::from_raw(1);
        let sym2 = SymbolId::from_raw(2);
        let type1 = TypeId::from_raw(1);

        symbol_map.insert(sym1, "symbol1".to_string());
        symbol_map.insert(sym2, "symbol2".to_string());

        type_set.insert(type1);

        assert_eq!(symbol_map.get(&sym1), Some(&"symbol1".to_string()));
        assert_eq!(symbol_map.len(), 2);
        assert!(type_set.contains(&type1));
        assert_eq!(type_set.len(), 1);
    }
}
