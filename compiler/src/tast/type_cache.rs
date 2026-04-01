//! Type Cache System for Frequently Accessed Types
//!
//! This module provides a multi-level caching system for type lookups
//! to improve performance in the type system.

use super::{InternedString, SymbolId, TypeId};
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::rc::Rc;

/// Statistics for cache performance monitoring
#[derive(Debug, Default, Clone)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub total_lookups: u64,
}

impl CacheStats {
    pub fn hit_rate(&self) -> f64 {
        if self.total_lookups == 0 {
            0.0
        } else {
            self.hits as f64 / self.total_lookups as f64
        }
    }
}

/// Cache key for type lookups
#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum TypeCacheKey {
    /// Named type lookup by symbol + kind tag (0=class, 1=interface, 2=enum)
    NamedType(SymbolId, u8),
    /// Named type with generic arguments + kind tag
    GenericType(SymbolId, Vec<TypeId>, u8),
    /// Array type lookup
    ArrayType(TypeId),
    /// Map type lookup
    MapType(TypeId, TypeId),
    /// Optional type lookup
    OptionalType(TypeId),
    /// Function type lookup
    FunctionType(Vec<TypeId>, TypeId),
    /// Union type lookup
    UnionType(Vec<TypeId>),
    /// Anonymous object type by field names
    AnonymousType(Vec<InternedString>),
    /// Type alias resolution
    TypeAlias(SymbolId),
    /// Abstract type underlying
    AbstractUnderlying(SymbolId),
}

/// Entry in the type cache with access tracking
#[derive(Debug)]
struct CacheEntry {
    type_id: TypeId,
    access_count: Cell<u32>,
    last_access: Cell<u64>,
}

/// Multi-level type cache system
pub struct TypeCache {
    /// L1 cache: Small, fast cache for most frequently accessed types
    l1_cache: RefCell<BTreeMap<TypeCacheKey, CacheEntry>>,
    l1_max_size: usize,

    /// L2 cache: Larger cache for moderately accessed types
    l2_cache: RefCell<BTreeMap<TypeCacheKey, CacheEntry>>,
    l2_max_size: usize,

    /// Access counter for LRU tracking
    access_counter: Cell<u64>,

    /// Cache statistics
    stats: RefCell<CacheStats>,

    /// Enable statistics collection
    collect_stats: bool,
}

impl TypeCache {
    /// Create a new type cache with default sizes
    pub fn new() -> Self {
        Self::with_sizes(64, 512, true)
    }

    /// Create a type cache with custom sizes
    pub fn with_sizes(l1_size: usize, l2_size: usize, collect_stats: bool) -> Self {
        TypeCache {
            l1_cache: RefCell::new(BTreeMap::new()),
            l1_max_size: l1_size,
            l2_cache: RefCell::new(BTreeMap::new()),
            l2_max_size: l2_size,
            access_counter: Cell::new(0),
            stats: RefCell::new(CacheStats::default()),
            collect_stats,
        }
    }

    /// Look up a type in the cache
    pub fn get(&self, key: &TypeCacheKey) -> Option<TypeId> {
        let current_access = self.access_counter.get();
        self.access_counter.set(current_access + 1);

        if self.collect_stats {
            let mut stats = self.stats.borrow_mut();
            stats.total_lookups += 1;
        }

        // Check L1 cache first
        if let Some(entry) = self.l1_cache.borrow().get(key) {
            entry.access_count.set(entry.access_count.get() + 1);
            entry.last_access.set(current_access);

            if self.collect_stats {
                self.stats.borrow_mut().hits += 1;
            }

            return Some(entry.type_id);
        }

        // Check L2 cache
        let promote_info = {
            let l2_cache = self.l2_cache.borrow();
            if let Some(entry) = l2_cache.get(key) {
                entry.access_count.set(entry.access_count.get() + 1);
                entry.last_access.set(current_access);

                if self.collect_stats {
                    self.stats.borrow_mut().hits += 1;
                }

                let type_id = entry.type_id;
                let should_promote = entry.access_count.get() > 3;

                Some((type_id, should_promote))
            } else {
                None
            }
        }; // Drop l2_cache borrow here

        if let Some((type_id, should_promote)) = promote_info {
            // Promote to L1 if accessed frequently
            if should_promote {
                self.promote_to_l1(key.clone(), type_id);
            }

            return Some(type_id);
        }

        if self.collect_stats {
            self.stats.borrow_mut().misses += 1;
        }

        None
    }

    /// Insert a type into the cache
    pub fn insert(&self, key: TypeCacheKey, type_id: TypeId) {
        let current_access = self.access_counter.get();

        let entry = CacheEntry {
            type_id,
            access_count: Cell::new(1),
            last_access: Cell::new(current_access),
        };

        // Try to insert into L1 first
        let mut l1_cache = self.l1_cache.borrow_mut();
        if l1_cache.len() < self.l1_max_size {
            l1_cache.insert(key, entry);
        } else {
            // L1 is full, insert into L2
            drop(l1_cache);
            self.insert_into_l2(key, entry);
        }
    }

    /// Promote an entry from L2 to L1
    fn promote_to_l1(&self, key: TypeCacheKey, type_id: TypeId) {
        let mut l1_cache = self.l1_cache.borrow_mut();

        // If L1 is full, evict LRU entry
        if l1_cache.len() >= self.l1_max_size {
            self.evict_lru_from_l1(&mut l1_cache);
        }

        let current_access = self.access_counter.get();
        let entry = CacheEntry {
            type_id,
            access_count: Cell::new(1),
            last_access: Cell::new(current_access),
        };

        l1_cache.insert(key.clone(), entry);

        // Remove from L2
        self.l2_cache.borrow_mut().remove(&key);
    }

    /// Insert into L2 cache
    fn insert_into_l2(&self, key: TypeCacheKey, entry: CacheEntry) {
        let mut l2_cache = self.l2_cache.borrow_mut();

        // If L2 is full, evict LRU entry
        if l2_cache.len() >= self.l2_max_size {
            self.evict_lru_from_l2(&mut l2_cache);
        }

        l2_cache.insert(key, entry);
    }

    /// Evict least recently used entry from L1
    fn evict_lru_from_l1(&self, l1_cache: &mut BTreeMap<TypeCacheKey, CacheEntry>) {
        if let Some(lru_key) = l1_cache
            .iter()
            .min_by_key(|(_, entry)| entry.last_access.get())
            .map(|(k, _)| k.clone())
        {
            if let Some(evicted) = l1_cache.remove(&lru_key) {
                // Move evicted entry to L2
                self.insert_into_l2(lru_key, evicted);

                if self.collect_stats {
                    self.stats.borrow_mut().evictions += 1;
                }
            }
        }
    }

    /// Evict least recently used entry from L2
    fn evict_lru_from_l2(&self, l2_cache: &mut BTreeMap<TypeCacheKey, CacheEntry>) {
        if let Some(lru_key) = l2_cache
            .iter()
            .min_by_key(|(_, entry)| entry.last_access.get())
            .map(|(k, _)| k.clone())
        {
            l2_cache.remove(&lru_key);

            if self.collect_stats {
                self.stats.borrow_mut().evictions += 1;
            }
        }
    }

    /// Clear all caches
    pub fn clear(&self) {
        self.l1_cache.borrow_mut().clear();
        self.l2_cache.borrow_mut().clear();
        self.access_counter.set(0);
        *self.stats.borrow_mut() = CacheStats::default();
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        self.stats.borrow().clone()
    }

    /// Get current cache sizes
    pub fn sizes(&self) -> (usize, usize) {
        (self.l1_cache.borrow().len(), self.l2_cache.borrow().len())
    }

    /// Preload common types into cache
    pub fn preload_common_types(&self, common_types: Vec<(TypeCacheKey, TypeId)>) {
        for (key, type_id) in common_types {
            self.insert(key, type_id);
        }
    }
}

/// Thread-local type cache for single-threaded compiler
thread_local! {
    static TYPE_CACHE: Rc<TypeCache> = Rc::new(TypeCache::new());
}

/// Get the thread-local type cache
pub fn get_type_cache() -> Rc<TypeCache> {
    TYPE_CACHE.with(|cache| cache.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_basic_operations() {
        let cache = TypeCache::new();
        let key = TypeCacheKey::NamedType(SymbolId::from_raw(1), 0);
        let type_id = TypeId::from_raw(100);

        // Test miss
        assert_eq!(cache.get(&key), None);

        // Test insert and hit
        cache.insert(key.clone(), type_id);
        assert_eq!(cache.get(&key), Some(type_id));

        // Test stats
        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.total_lookups, 2);
    }

    #[test]
    fn test_cache_promotion() {
        let cache = TypeCache::with_sizes(2, 4, true);

        // Fill L1
        cache.insert(
            TypeCacheKey::NamedType(SymbolId::from_raw(1), 0),
            TypeId::from_raw(1),
        );
        cache.insert(
            TypeCacheKey::NamedType(SymbolId::from_raw(2), 0),
            TypeId::from_raw(2),
        );

        // This should go to L2
        cache.insert(
            TypeCacheKey::NamedType(SymbolId::from_raw(3), 0),
            TypeId::from_raw(3),
        );

        let (l1_size, l2_size) = cache.sizes();
        assert_eq!(l1_size, 2);
        assert_eq!(l2_size, 1);

        // Access L2 entry multiple times to trigger promotion
        let key3 = TypeCacheKey::NamedType(SymbolId::from_raw(3), 0);
        for _ in 0..4 {
            cache.get(&key3);
        }

        // Should now be promoted to L1
        let (l1_size, l2_size) = cache.sizes();
        assert_eq!(l1_size, 2);
        assert_eq!(l2_size, 1); // One entry was evicted from L1 to L2
    }
}
