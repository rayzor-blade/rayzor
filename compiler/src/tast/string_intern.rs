//! High-Performance String Interning System
//!
//! This module provides efficient string interning for compiler symbols, type names,
//! and identifiers. Features:
//! - Arena-based string storage for cache efficiency
//! - O(1) string comparison via ID comparison
//! - Thread-safe concurrent interning
//! - Deduplication of identical strings
//! - Fast reverse lookup for debugging

use super::TypedArena;
use std::collections::HashMap;
use std::hash::{BuildHasher, Hasher};
use std::sync::{
    atomic::{AtomicU32, Ordering},
    Mutex,
};

/// A high-performance hasher optimized for string keys
///
/// Uses the same hasher as rustc for maximum performance on string workloads.
/// FxHash is faster than the default hasher for string keys.
#[derive(Default)]
pub struct FxHasher {
    hash: usize,
}

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.hash = self.hash.rotate_left(5).wrapping_add(byte as usize);
        }
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.hash as u64
    }
}

/// Build hasher for FxHasher
#[derive(Default)]
pub struct FxBuildHasher;

impl BuildHasher for FxBuildHasher {
    type Hasher = FxHasher;

    fn build_hasher(&self) -> Self::Hasher {
        FxHasher::default()
    }
}

/// Fast hash map type optimized for string keys
type FxHashMap<K, V> = HashMap<K, V, FxBuildHasher>;

/// An interned string represented as a unique ID
///
/// Two InternedString values are equal if and only if they represent
/// the same string content. Comparison is O(1) via ID comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct InternedString(u32);

impl InternedString {
    /// Create an InternedString from a raw ID (for testing/debugging)
    ///
    /// # Safety
    /// The caller must ensure the ID is valid within the interner context.
    pub const unsafe fn from_raw(id: u32) -> Self {
        Self(id)
    }

    /// Get the raw ID of this interned string
    pub const fn as_raw(self) -> u32 {
        self.0
    }

    /// Check if this is a valid (non-null) interned string
    pub const fn is_valid(self) -> bool {
        self.0 != u32::MAX
    }
}

impl Default for InternedString {
    fn default() -> Self {
        // Use max value as "null" sentinel
        Self(u32::MAX)
    }
}

/// High-performance string interner with arena-based storage
///
/// Provides fast string interning with deduplication and efficient lookup.
/// Thread-safe for concurrent access from multiple threads.
pub struct StringInterner {
    /// Arena for storing string bytes
    arena: TypedArena<u8>,

    /// Map from string content to interned ID
    /// Protected by mutex for thread safety
    intern_map: Mutex<FxHashMap<&'static str, InternedString>>,

    /// Reverse map from ID to string (for debugging and display)
    /// Protected by mutex for thread safety
    reverse_map: Mutex<Vec<&'static str>>,

    /// Next available string ID (thread-safe atomic)
    next_id: AtomicU32,
}

impl StringInterner {
    /// Create a new string interner
    pub fn new() -> Self {
        Self {
            arena: TypedArena::new(),
            intern_map: Mutex::new(FxHashMap::default()),
            reverse_map: Mutex::new(Vec::new()),
            next_id: AtomicU32::new(0),
        }
    }

    /// Create a new string interner with pre-allocated capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            arena: TypedArena::with_capacity(capacity * 16), // Estimate ~16 bytes per string
            intern_map: Mutex::new(FxHashMap::with_capacity_and_hasher(capacity, FxBuildHasher)),
            reverse_map: Mutex::new(Vec::with_capacity(capacity)),
            next_id: AtomicU32::new(0),
        }
    }

    /// Intern a string, returning its unique ID
    ///
    /// If the string has been interned before, returns the existing ID.
    /// Otherwise, allocates new storage and returns a new ID.
    ///
    /// This operation is thread-safe and can be called concurrently.
    pub fn intern(&self, s: &str) -> InternedString {
        // Fast path: check if already interned (read-only lock)
        {
            let intern_map = self.intern_map.lock().unwrap();
            if let Some(&existing_id) = intern_map.get(s) {
                return existing_id;
            }
        }

        // Slow path: need to intern new string
        self.intern_new_string(s)
    }

    /// Intern a new string (slow path when string not found)
    fn intern_new_string(&self, s: &str) -> InternedString {
        // Allocate space in arena for the string bytes
        let string_bytes = self.arena.alloc_slice_copy(s.as_bytes());

        // SAFETY: We just allocated valid UTF-8 bytes in the arena
        // The arena's lifetime encompasses the entire StringInterner lifetime
        let interned_str: &'static str = unsafe {
            let ptr = string_bytes.as_ptr();
            let len = string_bytes.len();
            let slice = std::slice::from_raw_parts(ptr, len);
            std::str::from_utf8_unchecked(slice)
        };

        // Extend the lifetime to 'static since arena outlives all references
        let static_str: &'static str = unsafe { std::mem::transmute(interned_str) };

        // Now we need to add it to both maps with proper synchronization
        let mut intern_map = self.intern_map.lock().unwrap();
        let mut reverse_map = self.reverse_map.lock().unwrap();

        // Double-check: another thread might have interned it while we were allocating
        if let Some(&existing_id) = intern_map.get(s) {
            // Another thread beat us to it - return the existing ID
            // The arena allocation above becomes waste, but that's rare and acceptable
            return existing_id;
        }

        // Create new ID and add to both maps
        let id = InternedString(self.next_id.fetch_add(1, Ordering::Relaxed));

        intern_map.insert(static_str, id);
        reverse_map.push(static_str);

        debug_assert_eq!(reverse_map.len() - 1, id.0 as usize);

        id
    }

    /// Get the string content for an interned string ID
    ///
    /// Returns None if the ID is invalid.
    pub fn get(&self, id: InternedString) -> Option<&str> {
        if !id.is_valid() {
            return None;
        }

        let reverse_map = self.reverse_map.lock().unwrap();
        reverse_map.get(id.0 as usize).copied()
    }

    /// Get the string content for an interned string ID (unchecked)
    ///
    /// # Safety
    /// The caller must ensure the ID is valid.
    pub unsafe fn get_unchecked(&self, id: InternedString) -> &str {
        let reverse_map = self.reverse_map.lock().unwrap();
        reverse_map.get_unchecked(id.0 as usize)
    }

    /// Get the number of unique strings interned
    pub fn len(&self) -> usize {
        self.next_id.load(Ordering::Relaxed) as usize
    }

    /// Check if the interner is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get statistics about the interner's memory usage
    pub fn stats(&self) -> InternerStats {
        let reverse_map = self.reverse_map.lock().unwrap();
        let arena_stats = self.arena.stats();

        let total_string_length: usize = reverse_map.iter().map(|s| s.len()).sum();
        let average_string_length = if reverse_map.is_empty() {
            0.0
        } else {
            total_string_length as f64 / reverse_map.len() as f64
        };

        InternerStats {
            unique_strings: reverse_map.len(),
            total_string_bytes: total_string_length,
            arena_bytes_allocated: arena_stats.total_bytes_allocated,
            arena_bytes_capacity: arena_stats.total_bytes_capacity,
            arena_chunks: arena_stats.chunk_count,
            average_string_length,
            memory_efficiency: if arena_stats.total_bytes_capacity > 0 {
                total_string_length as f64 / arena_stats.total_bytes_capacity as f64
            } else {
                0.0
            },
        }
    }

    /// Iterate over all interned strings
    pub fn iter(&self) -> InternerIterator {
        let reverse_map = self.reverse_map.lock().unwrap();
        let strings: Vec<&str> = reverse_map.iter().copied().collect();

        InternerIterator { strings, index: 0 }
    }

    /// Check if a string has been interned without interning it
    pub fn contains(&self, s: &str) -> bool {
        let intern_map = self.intern_map.lock().unwrap();
        intern_map.contains_key(s)
    }

    /// Get the ID of an already-interned string without interning it
    ///
    /// Returns None if the string hasn't been interned yet.
    pub fn get_id(&self, s: &str) -> Option<InternedString> {
        let intern_map = self.intern_map.lock().unwrap();
        intern_map.get(s).copied()
    }
}

impl Default for StringInterner {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for StringInterner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stats = self.stats();
        f.debug_struct("StringInterner")
            .field("unique_strings", &stats.unique_strings)
            .field("total_bytes", &stats.total_string_bytes)
            .field("arena_bytes", &stats.arena_bytes_allocated)
            .finish()
    }
}

/// Statistics about string interner memory usage and efficiency
#[derive(Debug, Clone)]
pub struct InternerStats {
    /// Number of unique strings interned
    pub unique_strings: usize,
    /// Total bytes used by string content
    pub total_string_bytes: usize,
    /// Bytes allocated in the arena
    pub arena_bytes_allocated: usize,
    /// Total arena capacity in bytes
    pub arena_bytes_capacity: usize,
    /// Number of arena chunks
    pub arena_chunks: usize,
    /// Average length of interned strings
    pub average_string_length: f64,
    /// Efficiency ratio (string bytes / arena capacity)
    pub memory_efficiency: f64,
}

impl InternerStats {
    /// Get the memory overhead as a percentage
    pub fn overhead_percent(&self) -> f64 {
        if self.total_string_bytes == 0 {
            0.0
        } else {
            let overhead = self
                .arena_bytes_allocated
                .saturating_sub(self.total_string_bytes);
            (overhead as f64 / self.total_string_bytes as f64) * 100.0
        }
    }

    /// Get the arena utilization as a percentage
    pub fn utilization_percent(&self) -> f64 {
        if self.arena_bytes_capacity == 0 {
            0.0
        } else {
            (self.arena_bytes_allocated as f64 / self.arena_bytes_capacity as f64) * 100.0
        }
    }
}

/// Iterator over all interned strings
pub struct InternerIterator {
    strings: Vec<&'static str>,
    index: usize,
}

impl Iterator for InternerIterator {
    type Item = (InternedString, &'static str);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.strings.len() {
            let string = self.strings[self.index];
            let id = InternedString(self.index as u32);
            self.index += 1;
            Some((id, string))
        } else {
            None
        }
    }
}

impl ExactSizeIterator for InternerIterator {
    fn len(&self) -> usize {
        self.strings.len() - self.index
    }
}

// `InternedString` deliberately has no `Display`. It is an index into a
// per-process interner, so it carries no text of its own; a `Display` that
// printed the index would let `{}` or `.to_string()` produce a name-shaped
// string that matches nothing and varies between runs. Resolve through the
// interner that produced the id — `StringInterner::get` — and use `{:?}` only
// where an id is genuinely what is wanted.

/// Convenience functions for common string interning patterns
impl StringInterner {
    /// Intern multiple strings at once
    pub fn intern_all<'a, I>(&self, strings: I) -> Vec<InternedString>
    where
        I: IntoIterator<Item = &'a str>,
    {
        strings.into_iter().map(|s| self.intern(s)).collect()
    }

    /// Intern a string slice and return both the ID and the interned string
    pub fn intern_with_string(&self, s: &str) -> (InternedString, &str) {
        let id = self.intern(s);
        let interned = self.get(id).unwrap();
        (id, interned)
    }

    /// Create a formatted string and intern it
    pub fn intern_formatted(&self, args: std::fmt::Arguments) -> InternedString {
        let formatted = args.to_string();
        self.intern(&formatted)
    }
}

/// Macro for convenient formatted string interning
#[macro_export]
macro_rules! intern_format {
    ($interner:expr, $($arg:tt)*) => {
        $interner.intern_formatted(format_args!($($arg)*))
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_basic_interning() {
        let interner = StringInterner::new();

        let hello1 = interner.intern("hello");
        let hello2 = interner.intern("hello");
        let world = interner.intern("world");

        // Same string should get same ID
        assert_eq!(hello1, hello2);
        assert_ne!(hello1, world);

        // Can retrieve strings
        assert_eq!(interner.get(hello1).unwrap(), "hello");
        assert_eq!(interner.get(world).unwrap(), "world");
    }

    #[test]
    fn test_empty_and_special_strings() {
        let interner = StringInterner::new();

        let empty = interner.intern("");
        let space = interner.intern(" ");
        let newline = interner.intern("\n");
        let unicode = interner.intern("🦀");

        assert_eq!(interner.get(empty).unwrap(), "");
        assert_eq!(interner.get(space).unwrap(), " ");
        assert_eq!(interner.get(newline).unwrap(), "\n");
        assert_eq!(interner.get(unicode).unwrap(), "🦀");

        // All should be different
        assert_ne!(empty, space);
        assert_ne!(space, newline);
        assert_ne!(newline, unicode);
    }

    #[test]
    fn test_long_strings() {
        let interner = StringInterner::new();

        let long_string = "a".repeat(10000);
        let id1 = interner.intern(&long_string);
        let id2 = interner.intern(&long_string);

        assert_eq!(id1, id2);
        assert_eq!(interner.get(id1).unwrap(), long_string);
    }

    #[test]
    fn test_many_strings() {
        let interner = StringInterner::new();
        let mut ids = Vec::new();

        // Intern many unique strings
        for i in 0..1000 {
            let s = format!("string_{}", i);
            let id = interner.intern(&s);
            ids.push((id, s));
        }

        // Verify all are different and retrievable
        for (id, original) in &ids {
            assert_eq!(interner.get(*id).unwrap(), original);
        }

        // Verify all IDs are unique
        for i in 0..ids.len() {
            for j in i + 1..ids.len() {
                assert_ne!(ids[i].0, ids[j].0);
            }
        }

        assert_eq!(interner.len(), 1000);
    }

    #[test]
    fn test_thread_safety() {
        let interner = Arc::new(StringInterner::new());
        let mut handles = vec![];

        // Multiple threads interning strings concurrently
        for thread_id in 0..4 {
            let interner_clone = Arc::clone(&interner);
            let handle = thread::spawn(move || {
                let mut local_ids = Vec::new();

                for i in 0..100 {
                    let s = format!("thread_{}_{}", thread_id, i);
                    let id = interner_clone.intern(&s);
                    local_ids.push((id, s));
                }

                // Also intern some common strings
                for _ in 0..10 {
                    interner_clone.intern("common");
                    interner_clone.intern("shared");
                }

                local_ids
            });
            handles.push(handle);
        }

        let mut all_ids = Vec::new();
        for handle in handles {
            let thread_ids = handle.join().unwrap();
            all_ids.extend(thread_ids);
        }

        // Verify all strings are retrievable
        for (id, original) in &all_ids {
            assert_eq!(interner.get(*id).unwrap(), original);
        }

        // Common strings should have same ID
        let common1 = interner.intern("common");
        let common2 = interner.intern("common");
        assert_eq!(common1, common2);
    }

    #[test]
    fn test_stats_and_efficiency() {
        let interner = StringInterner::with_capacity(100);

        let initial_stats = interner.stats();
        assert_eq!(initial_stats.unique_strings, 0);
        assert_eq!(initial_stats.total_string_bytes, 0);

        // Intern some strings
        for i in 0..50 {
            interner.intern(&format!("test_string_{}", i));
        }

        let stats = interner.stats();
        assert_eq!(stats.unique_strings, 50);
        assert!(stats.total_string_bytes > 0);
        assert!(stats.average_string_length > 0.0);
        assert!(stats.memory_efficiency > 0.0);
        assert!(stats.utilization_percent() > 0.0);
    }

    #[test]
    fn test_iterator() {
        let interner = StringInterner::new();

        let strings = vec!["alpha", "beta", "gamma"];
        let mut ids = Vec::new();

        for s in &strings {
            ids.push(interner.intern(s));
        }

        // Test iterator
        let mut collected: Vec<(InternedString, &str)> = interner.iter().collect();
        collected.sort_by_key(|(id, _)| id.as_raw());

        assert_eq!(collected.len(), 3);
        for (i, (id, string)) in collected.iter().enumerate() {
            assert_eq!(id.as_raw(), i as u32);
            assert!(strings.contains(string));
        }
    }

    #[test]
    fn test_contains_and_get_id() {
        let interner = StringInterner::new();

        assert!(!interner.contains("test"));
        assert_eq!(interner.get_id("test"), None);

        let id = interner.intern("test");

        assert!(interner.contains("test"));
        assert_eq!(interner.get_id("test"), Some(id));

        assert!(!interner.contains("other"));
        assert_eq!(interner.get_id("other"), None);
    }

    #[test]
    fn test_convenience_methods() {
        let interner = StringInterner::new();

        // Test intern_all
        let strings = vec!["one", "two", "three"];
        let ids = interner.intern_all(strings.iter().copied());
        assert_eq!(ids.len(), 3);

        for (i, &string) in strings.iter().enumerate() {
            assert_eq!(interner.get(ids[i]).unwrap(), string);
        }

        // Test intern_with_string
        let (id, interned) = interner.intern_with_string("test");
        assert_eq!(interned, "test");
        assert_eq!(interner.get(id).unwrap(), "test");
    }

    #[test]
    fn test_invalid_ids() {
        let interner = StringInterner::new();

        let invalid_id = InternedString::default();
        assert!(!invalid_id.is_valid());
        assert_eq!(interner.get(invalid_id), None);

        // Test out of range ID
        let out_of_range = unsafe { InternedString::from_raw(999999) };
        assert_eq!(interner.get(out_of_range), None);
    }

    #[test]
    fn test_unicode_strings() {
        let interner = StringInterner::new();

        let unicode_strings = vec![
            "Hello, 世界",
            "🦀 Rust 🦀",
            "Здравствуй, мир!",
            "नमस्ते दुनिया",
            "مرحبا بالعالم",
        ];

        let mut ids = Vec::new();
        for s in &unicode_strings {
            let id = interner.intern(s);
            ids.push(id);
        }

        // Verify all unicode strings are handled correctly
        for (i, &id) in ids.iter().enumerate() {
            assert_eq!(interner.get(id).unwrap(), unicode_strings[i]);
        }
    }

    #[test]
    fn test_interned_string_ordering() {
        let interner = StringInterner::new();

        let id1 = interner.intern("a");
        let id2 = interner.intern("b");
        let id3 = interner.intern("c");

        // IDs should be ordered by creation order
        assert!(id1 < id2);
        assert!(id2 < id3);
        assert!(id1 < id3);

        // Test PartialOrd/Ord implementation
        let mut ids = vec![id3, id1, id2];
        ids.sort();
        assert_eq!(ids, vec![id1, id2, id3]);
    }
}
