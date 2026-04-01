//! Symbol resolution cache implementation for performance optimization
//!
//! This module provides a caching layer for symbol resolution to avoid
//! repeated lookups through the scope hierarchy.

use crate::tast::{InternedString, ScopeId, SymbolId, SymbolKind, TypeId};
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::rc::Rc;

/// Cache key for symbol resolution
#[derive(Hash, Eq, PartialEq, PartialOrd, Ord, Clone, Debug)]
pub enum SymbolCacheKey {
    /// Symbol lookup by name in a specific scope
    NameInScope(ScopeId, InternedString),
    /// Symbol lookup by name globally (searches all scopes)
    GlobalName(InternedString),
    /// Symbols by kind (e.g., all classes, all functions)
    SymbolsByKind(SymbolKind),
    /// Symbols in a specific scope
    SymbolsInScope(ScopeId),
    /// Type ID to Symbol ID mapping
    TypeToSymbol(TypeId),
    /// Supertype hierarchy for a symbol
    Supertypes(SymbolId),
    /// Enum variants for an enum symbol
    EnumVariants(SymbolId),
    /// Fully qualified name resolution
    QualifiedName(InternedString),
}

/// Statistics for cache performance monitoring
#[derive(Default, Debug, Clone)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub invalidations: u64,
    pub evictions: u64,
    pub total_lookups: u64,
}

impl CacheStats {
    pub fn hit_rate(&self) -> f64 {
        if self.total_lookups == 0 {
            0.0
        } else {
            (self.hits as f64 / self.total_lookups as f64) * 100.0
        }
    }
}

/// Entry in the symbol cache with access tracking
#[derive(Debug)]
struct SymbolCacheEntry<T> {
    value: T,
    access_count: Cell<u32>,
    last_access: Cell<u64>,
}

/// Enhanced symbol resolution cache with multi-level caching
#[derive(Debug)]
pub struct SymbolResolutionCache {
    /// Primary cache for single symbol lookups
    symbol_cache: RefCell<BTreeMap<SymbolCacheKey, SymbolCacheEntry<Option<SymbolId>>>>,

    /// Cache for multi-symbol results (like all symbols in a scope)
    multi_symbol_cache: RefCell<BTreeMap<SymbolCacheKey, SymbolCacheEntry<Vec<SymbolId>>>>,

    /// Cache statistics
    stats: RefCell<CacheStats>,

    /// Access counter for LRU tracking
    access_counter: Cell<u64>,

    /// Maximum cache sizes before eviction
    symbol_cache_max_size: usize,
    multi_symbol_cache_max_size: usize,
}

impl SymbolResolutionCache {
    /// Create a new symbol resolution cache
    pub fn new(max_size: usize) -> Self {
        Self {
            symbol_cache: RefCell::new(BTreeMap::new()),
            multi_symbol_cache: RefCell::new(BTreeMap::new()),
            stats: RefCell::new(CacheStats::default()),
            access_counter: Cell::new(0),
            symbol_cache_max_size: max_size,
            multi_symbol_cache_max_size: max_size / 2,
        }
    }

    /// Create a symbol resolution cache with custom sizes
    pub fn with_sizes(symbol_cache_size: usize, multi_symbol_cache_size: usize) -> Self {
        Self {
            symbol_cache: RefCell::new(BTreeMap::new()),
            multi_symbol_cache: RefCell::new(BTreeMap::new()),
            stats: RefCell::new(CacheStats::default()),
            access_counter: Cell::new(0),
            symbol_cache_max_size: symbol_cache_size,
            multi_symbol_cache_max_size: multi_symbol_cache_size,
        }
    }

    /// Look up a symbol in the cache using legacy interface
    pub fn get(&self, scope: ScopeId, name: InternedString) -> Option<Option<SymbolId>> {
        self.get_symbol(&SymbolCacheKey::NameInScope(scope, name))
    }

    /// Look up a single symbol in the cache
    pub fn get_symbol(&self, key: &SymbolCacheKey) -> Option<Option<SymbolId>> {
        let current_access = self.access_counter.get();
        self.access_counter.set(current_access + 1);

        let mut stats = self.stats.borrow_mut();
        stats.total_lookups += 1;

        if let Some(entry) = self.symbol_cache.borrow().get(key) {
            entry.access_count.set(entry.access_count.get() + 1);
            entry.last_access.set(current_access);
            stats.hits += 1;
            Some(entry.value)
        } else {
            stats.misses += 1;
            None
        }
    }

    /// Look up multiple symbols in the cache
    pub fn get_symbols(&self, key: &SymbolCacheKey) -> Option<Vec<SymbolId>> {
        let current_access = self.access_counter.get();
        self.access_counter.set(current_access + 1);

        let mut stats = self.stats.borrow_mut();
        stats.total_lookups += 1;

        if let Some(entry) = self.multi_symbol_cache.borrow().get(key) {
            entry.access_count.set(entry.access_count.get() + 1);
            entry.last_access.set(current_access);
            stats.hits += 1;
            Some(entry.value.clone())
        } else {
            stats.misses += 1;
            None
        }
    }

    /// Insert a symbol resolution result into the cache using legacy interface
    pub fn insert(&self, scope: ScopeId, name: InternedString, symbol: Option<SymbolId>) {
        self.insert_symbol(SymbolCacheKey::NameInScope(scope, name), symbol);
    }

    /// Insert a single symbol into the cache
    pub fn insert_symbol(&self, key: SymbolCacheKey, symbol: Option<SymbolId>) {
        let current_access = self.access_counter.get();

        let entry = SymbolCacheEntry {
            value: symbol,
            access_count: Cell::new(1),
            last_access: Cell::new(current_access),
        };

        let mut cache = self.symbol_cache.borrow_mut();

        // Check if we need to evict entries
        if cache.len() >= self.symbol_cache_max_size {
            self.evict_lru_symbol(&mut cache);
        }

        cache.insert(key, entry);
    }

    /// Insert multiple symbols into the cache
    pub fn insert_symbols(&self, key: SymbolCacheKey, symbols: Vec<SymbolId>) {
        let current_access = self.access_counter.get();

        let entry = SymbolCacheEntry {
            value: symbols,
            access_count: Cell::new(1),
            last_access: Cell::new(current_access),
        };

        let mut cache = self.multi_symbol_cache.borrow_mut();

        // Check if we need to evict entries
        if cache.len() >= self.multi_symbol_cache_max_size {
            self.evict_lru_multi_symbol(&mut cache);
        }

        cache.insert(key, entry);
    }

    /// Invalidate all cached entries for a specific scope
    pub fn invalidate_scope(&self, scope: ScopeId) {
        let mut symbol_cache = self.symbol_cache.borrow_mut();
        let mut multi_symbol_cache = self.multi_symbol_cache.borrow_mut();
        let mut stats = self.stats.borrow_mut();

        // Remove all entries for this scope
        symbol_cache.retain(|key, _| {
            let should_remove = matches!(key,
                SymbolCacheKey::NameInScope(sid, _) |
                SymbolCacheKey::SymbolsInScope(sid) if *sid == scope
            );
            if should_remove {
                stats.invalidations += 1;
            }
            !should_remove
        });

        multi_symbol_cache.retain(|key, _| {
            let should_remove = matches!(key,
                SymbolCacheKey::NameInScope(sid, _) |
                SymbolCacheKey::SymbolsInScope(sid) if *sid == scope
            );
            if should_remove {
                stats.invalidations += 1;
            }
            !should_remove
        });
    }

    /// Invalidate all cached entries for a specific symbol name
    pub fn invalidate_name(&self, name: InternedString) {
        let mut symbol_cache = self.symbol_cache.borrow_mut();
        let mut multi_symbol_cache = self.multi_symbol_cache.borrow_mut();
        let mut stats = self.stats.borrow_mut();

        // Remove all entries for this name
        symbol_cache.retain(|key, _| {
            let should_remove = matches!(key,
                SymbolCacheKey::NameInScope(_, n) |
                SymbolCacheKey::GlobalName(n) |
                SymbolCacheKey::QualifiedName(n) if *n == name
            );
            if should_remove {
                stats.invalidations += 1;
            }
            !should_remove
        });

        multi_symbol_cache.retain(|key, _| {
            let should_remove = matches!(key,
                SymbolCacheKey::NameInScope(_, n) |
                SymbolCacheKey::GlobalName(n) |
                SymbolCacheKey::QualifiedName(n) if *n == name
            );
            if should_remove {
                stats.invalidations += 1;
            }
            !should_remove
        });
    }

    /// Clear the entire cache
    pub fn clear(&self) {
        let mut symbol_cache = self.symbol_cache.borrow_mut();
        let mut multi_symbol_cache = self.multi_symbol_cache.borrow_mut();
        let mut stats = self.stats.borrow_mut();

        stats.invalidations += symbol_cache.len() as u64;
        stats.invalidations += multi_symbol_cache.len() as u64;

        symbol_cache.clear();
        multi_symbol_cache.clear();
        self.access_counter.set(0);
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        self.stats.borrow().clone()
    }

    /// Get current cache sizes
    pub fn sizes(&self) -> (usize, usize) {
        (
            self.symbol_cache.borrow().len(),
            self.multi_symbol_cache.borrow().len(),
        )
    }

    /// Evict least recently used entry from symbol cache
    fn evict_lru_symbol(
        &self,
        cache: &mut BTreeMap<SymbolCacheKey, SymbolCacheEntry<Option<SymbolId>>>,
    ) {
        if let Some(lru_key) = cache
            .iter()
            .min_by_key(|(_, entry)| entry.last_access.get())
            .map(|(k, _)| k.clone())
        {
            cache.remove(&lru_key);
            self.stats.borrow_mut().evictions += 1;
        }
    }

    /// Evict least recently used entry from multi-symbol cache
    fn evict_lru_multi_symbol(
        &self,
        cache: &mut BTreeMap<SymbolCacheKey, SymbolCacheEntry<Vec<SymbolId>>>,
    ) {
        if let Some(lru_key) = cache
            .iter()
            .min_by_key(|(_, entry)| entry.last_access.get())
            .map(|(k, _)| k.clone())
        {
            cache.remove(&lru_key);
            self.stats.borrow_mut().evictions += 1;
        }
    }
}

/// Bloom filter for quick negative lookups
pub struct SymbolBloomFilter {
    bits: Vec<u64>,
    num_hashes: usize,
}

impl SymbolBloomFilter {
    /// Create a new bloom filter with the specified size
    pub fn new(expected_items: usize) -> Self {
        // Calculate optimal size (10 bits per item for ~1% false positive rate)
        let num_bits = expected_items * 10;
        let num_words = (num_bits + 63) / 64;

        Self {
            bits: vec![0; num_words],
            num_hashes: 7, // Optimal for 10 bits per item
        }
    }

    /// Add a symbol name to the bloom filter
    pub fn insert(&mut self, name: InternedString) {
        let hash = name.as_raw() as u64;

        for i in 0..self.num_hashes {
            let bit_pos = self.hash_to_bit(hash, i);
            let word_idx = bit_pos / 64;
            let bit_idx = bit_pos % 64;

            if word_idx < self.bits.len() {
                self.bits[word_idx] |= 1u64 << bit_idx;
            }
        }
    }

    /// Check if a symbol name might be in the set
    pub fn might_contain(&self, name: InternedString) -> bool {
        let hash = name.as_raw() as u64;

        for i in 0..self.num_hashes {
            let bit_pos = self.hash_to_bit(hash, i);
            let word_idx = bit_pos / 64;
            let bit_idx = bit_pos % 64;

            if word_idx >= self.bits.len() {
                return false;
            }

            if (self.bits[word_idx] & (1u64 << bit_idx)) == 0 {
                return false;
            }
        }

        true
    }

    /// Clear the bloom filter
    pub fn clear(&mut self) {
        self.bits.fill(0);
    }

    /// Generate a bit position from a hash and index
    fn hash_to_bit(&self, hash: u64, index: usize) -> usize {
        // Use different parts of the hash for each index
        let shifted = hash
            .wrapping_add(index as u64)
            .wrapping_mul(0x517cc1b727220a95);
        (shifted as usize) % (self.bits.len() * 64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tast::StringInterner;

    // #[test]
    // fn test_symbol_cache_basic() {
    //     let cache = SymbolResolutionCache::new(100);
    //     let mut interner = StringInterner::new();

    //     let scope1 = new_scope_id();
    //     let name1 = interner.intern("test_symbol");
    //     let symbol1 = new_symbol_id();

    //     // Cache miss
    //     assert_eq!(cache.get(scope1, name1), None);
    //     assert_eq!(cache.stats().misses, 1);

    //     // Insert
    //     cache.insert(scope1, name1, Some(symbol1));

    //     // Cache hit
    //     assert_eq!(cache.get(scope1, name1), Some(Some(symbol1)));
    //     assert_eq!(cache.stats().hits, 1);

    //     // Different scope = cache miss
    //     let scope2 = new_scope_id();
    //     assert_eq!(cache.get(scope2, name1), None);
    //     assert_eq!(cache.stats().misses, 2);
    // }

    #[test]
    fn test_bloom_filter() {
        let mut bloom = SymbolBloomFilter::new(1000);
        let mut interner = StringInterner::new();

        let name1 = interner.intern("test1");
        let name2 = interner.intern("test2");
        let name3 = interner.intern("test3");

        // Insert some names
        bloom.insert(name1);
        bloom.insert(name2);

        // Check containment
        assert!(bloom.might_contain(name1));
        assert!(bloom.might_contain(name2));

        // Name3 not inserted - might still return true (false positive)
        // but should return false most of the time
        let might_contain_name3 = bloom.might_contain(name3);

        // Clear and recheck
        bloom.clear();
        assert!(!bloom.might_contain(name1));
        assert!(!bloom.might_contain(name2));
    }
}
