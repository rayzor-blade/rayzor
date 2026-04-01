//! High-Performance Scope System for TAST
//!
//! This module provides efficient scope management for lexical scoping, name resolution,
//! and lifetime tracking. Features:
//! - Hierarchical scope tree with fast parent/child traversal
//! - Integration with Symbol System for name resolution
//! - Lifetime boundary tracking for ownership analysis
//! - Support for Haxe's scoping rules (classes, functions, blocks, etc.)
//! - Cache-friendly data structures with arena allocation

use super::{
    InternedString, LifetimeId, ScopeId, StringInterner, Symbol, SymbolId, SymbolTable, TypedArena,
};
use std::collections::BTreeMap;
use std::fmt;

/// The kind of scope (class, function, block, etc.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ScopeKind {
    /// Global/module scope
    Global,
    /// Package scope
    Package,
    /// Class body scope
    Class,
    /// Interface body scope
    Interface,
    /// Enum body scope
    Enum,
    /// Abstract type body scope
    Abstract,
    /// Function body scope
    Function,
    /// Block scope (including control flow blocks)
    Block,
    /// Loop scope (for loop variables)
    Loop,
    /// Try/catch scope
    TryCatch,
    /// Switch case scope
    SwitchCase,
    /// Anonymous function scope
    AnonymousFunction,
    /// Type parameter scope (for generic constraints)
    TypeParameter,
    /// Import scope (for import aliases)
    Import,
}

impl ScopeKind {
    /// Check if this scope kind can contain type declarations
    pub fn can_declare_types(self) -> bool {
        matches!(
            self,
            ScopeKind::Global
                | ScopeKind::Package
                | ScopeKind::Class
                | ScopeKind::Interface
                | ScopeKind::Enum
                | ScopeKind::Abstract
        )
    }

    /// Check if this scope kind can contain function declarations
    pub fn can_declare_functions(self) -> bool {
        matches!(
            self,
            ScopeKind::Global
                | ScopeKind::Package
                | ScopeKind::Class
                | ScopeKind::Interface
                | ScopeKind::Abstract
        )
    }

    /// Check if this scope kind supports variable shadowing
    pub fn allows_shadowing(self) -> bool {
        matches!(
            self,
            ScopeKind::Function
                | ScopeKind::Block
                | ScopeKind::Loop
                | ScopeKind::TryCatch
                | ScopeKind::SwitchCase
                | ScopeKind::AnonymousFunction
        )
    }

    /// Check if this scope creates a new lifetime region
    pub fn creates_lifetime_boundary(self) -> bool {
        matches!(
            self,
            ScopeKind::Function
                | ScopeKind::Block
                | ScopeKind::Loop
                | ScopeKind::TryCatch
                | ScopeKind::AnonymousFunction
        )
    }

    /// Check if variables in this scope can escape to parent scopes
    pub fn allows_variable_escape(self) -> bool {
        matches!(
            self,
            ScopeKind::Global
                | ScopeKind::Package
                | ScopeKind::Class
                | ScopeKind::Interface
                | ScopeKind::Enum
                | ScopeKind::Abstract
        )
    }
}

impl fmt::Display for ScopeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            ScopeKind::Global => "global",
            ScopeKind::Package => "package",
            ScopeKind::Class => "class",
            ScopeKind::Interface => "interface",
            ScopeKind::Enum => "enum",
            ScopeKind::Abstract => "abstract",
            ScopeKind::Function => "function",
            ScopeKind::Block => "block",
            ScopeKind::Loop => "loop",
            ScopeKind::TryCatch => "try-catch",
            ScopeKind::SwitchCase => "switch-case",
            ScopeKind::AnonymousFunction => "anonymous function",
            ScopeKind::TypeParameter => "type parameter",
            ScopeKind::Import => "import",
        };
        write!(f, "{}", name)
    }
}

/// Source location information for a scope
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScopeLocation {
    /// File ID where this scope begins
    pub file_id: u32,
    /// Line where scope opens
    pub start_line: u32,
    /// Line where scope closes (0 if unknown/open)
    pub end_line: u32,
    /// Column where scope opens
    pub start_column: u32,
    /// Column where scope closes (0 if unknown/open)
    pub end_column: u32,
}

impl ScopeLocation {
    pub const fn new(file_id: u32, start_line: u32, start_column: u32) -> Self {
        Self {
            file_id,
            start_line,
            end_line: 0,
            start_column,
            end_column: 0,
        }
    }

    pub const fn with_end(
        file_id: u32,
        start_line: u32,
        start_column: u32,
        end_line: u32,
        end_column: u32,
    ) -> Self {
        Self {
            file_id,
            start_line,
            end_line,
            start_column,
            end_column,
        }
    }

    pub const fn unknown() -> Self {
        Self {
            file_id: u32::MAX,
            start_line: 0,
            end_line: 0,
            start_column: 0,
            end_column: 0,
        }
    }

    pub const fn is_valid(self) -> bool {
        self.file_id != u32::MAX
    }

    pub fn set_end(&mut self, end_line: u32, end_column: u32) {
        self.end_line = end_line;
        self.end_column = end_column;
    }
}

impl Default for ScopeLocation {
    fn default() -> Self {
        Self::unknown()
    }
}

impl fmt::Display for ScopeLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_valid() {
            if self.end_line > 0 {
                write!(
                    f,
                    "{}:{}:{}-{}:{}",
                    self.file_id,
                    self.start_line,
                    self.start_column,
                    self.end_line,
                    self.end_column
                )
            } else {
                write!(
                    f,
                    "{}:{}:{}",
                    self.file_id, self.start_line, self.start_column
                )
            }
        } else {
            write!(f, "<unknown>")
        }
    }
}

/// Complete scope information
#[derive(Debug, Clone)]
pub struct Scope {
    /// Unique identifier for this scope
    pub id: ScopeId,
    /// Optional name for named scopes (classes, functions, etc.)
    pub name: Option<InternedString>,
    /// What kind of scope this is
    pub kind: ScopeKind,
    /// Parent scope (None for global scope)
    pub parent_id: Option<ScopeId>,
    /// Direct child scopes
    pub children: Vec<ScopeId>,
    /// Symbols declared directly in this scope
    pub symbols: Vec<SymbolId>,
    /// Lifetime associated with this scope
    pub lifetime_id: LifetimeId,
    /// Source location of this scope
    pub location: ScopeLocation,
    /// Depth in the scope hierarchy (0 for global)
    pub depth: u32,
    /// Whether this scope is currently active/open
    pub is_active: bool,
    /// Cached symbol lookup for performance
    symbol_lookup_cache: BTreeMap<InternedString, SymbolId>,
}

impl Scope {
    /// Create a new scope
    pub fn new(
        id: ScopeId,
        kind: ScopeKind,
        parent_id: Option<ScopeId>,
        location: ScopeLocation,
        depth: u32,
    ) -> Self {
        Self {
            id,
            name: None,
            kind,
            parent_id,
            children: Vec::new(),
            symbols: Vec::new(),
            lifetime_id: LifetimeId::invalid(),
            location,
            depth,
            is_active: true,
            symbol_lookup_cache: BTreeMap::new(),
        }
    }

    /// Create a named scope (for classes, functions, etc.)
    pub fn named(
        id: ScopeId,
        name: InternedString,
        kind: ScopeKind,
        parent_id: Option<ScopeId>,
        location: ScopeLocation,
        depth: u32,
    ) -> Self {
        Self {
            name: Some(name),
            ..Self::new(id, kind, parent_id, location, depth)
        }
    }

    /// Add a child scope
    pub fn add_child(&mut self, child_id: ScopeId) {
        if !self.children.contains(&child_id) {
            self.children.push(child_id);
        }
    }

    /// Add a symbol to this scope
    pub fn add_symbol(&mut self, symbol_id: SymbolId, symbol_name: InternedString) {
        self.symbols.push(symbol_id);
        self.symbol_lookup_cache.insert(symbol_name, symbol_id);
    }

    /// Remove a symbol from this scope (rarely used, but needed for error recovery)
    pub fn remove_symbol(&mut self, symbol_id: SymbolId, symbol_name: InternedString) {
        self.symbols.retain(|&id| id != symbol_id);
        self.symbol_lookup_cache.remove(&symbol_name);
    }

    /// Check if a name is already declared in this scope
    pub fn has_symbol(&self, name: InternedString) -> bool {
        self.symbol_lookup_cache.contains_key(&name)
    }

    /// Get a symbol by name in this scope (not including parent scopes)
    pub fn get_symbol(&self, name: InternedString) -> Option<SymbolId> {
        self.symbol_lookup_cache.get(&name).copied()
    }

    /// Close this scope (mark as inactive and set end location)
    pub fn close(&mut self, end_line: u32, end_column: u32) {
        self.is_active = false;
        self.location.set_end(end_line, end_column);
    }

    /// Check if this scope can declare a symbol of the given kind
    pub fn can_declare_symbol(&self, symbol_kind: super::SymbolKind) -> bool {
        match symbol_kind {
            super::SymbolKind::Class
            | super::SymbolKind::Interface
            | super::SymbolKind::Enum
            | super::SymbolKind::TypeAlias
            | super::SymbolKind::Abstract => self.kind.can_declare_types(),

            super::SymbolKind::Function => self.kind.can_declare_functions(),

            super::SymbolKind::Variable | super::SymbolKind::Parameter => true, // Variables can be declared in most scopes

            super::SymbolKind::Field | super::SymbolKind::Property => matches!(
                self.kind,
                ScopeKind::Class | ScopeKind::Interface | ScopeKind::Abstract
            ),

            super::SymbolKind::EnumVariant => matches!(self.kind, ScopeKind::Enum),

            super::SymbolKind::TypeParameter => matches!(
                self.kind,
                ScopeKind::Class
                    | ScopeKind::Interface
                    | ScopeKind::Function
                    | ScopeKind::Abstract
                    | ScopeKind::TypeParameter
            ),

            _ => true,
        }
    }

    /// Get a human-readable description of this scope
    pub fn description(&self, interner: &StringInterner) -> String {
        if let Some(name) = self.name {
            let name_str = interner.get(name).unwrap_or("<unknown>");
            format!("{} {} (depth {})", self.kind, name_str, self.depth)
        } else {
            format!("{} scope (depth {})", self.kind, self.depth)
        }
    }
}

impl fmt::Display for Scope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} scope #{} (depth {})", self.kind, self.id, self.depth)
    }
}

/// High-performance scope tree for managing hierarchical scopes
#[derive(Debug)]
pub struct ScopeTree {
    /// Arena for scope storage
    scopes_arena: TypedArena<Scope>,
    /// Map from scope ID to scope reference
    scopes_by_id: BTreeMap<ScopeId, &'static mut Scope>,
    /// Root scope (global scope)
    root_scope_id: ScopeId,
    /// Currently active scope (for parsing)
    current_scope_id: ScopeId,
    /// Fast lookup: scope -> all ancestor scopes (cached for performance)
    ancestor_cache: BTreeMap<ScopeId, Vec<ScopeId>>,
    /// Statistics
    total_scopes: usize,
    max_depth: u32,
}

impl ScopeTree {
    /// Create a new scope tree with a global scope
    pub fn new(global_scope_id: ScopeId) -> Self {
        let mut tree = Self {
            scopes_arena: TypedArena::new(),
            scopes_by_id: BTreeMap::new(),
            root_scope_id: global_scope_id,
            current_scope_id: global_scope_id,
            ancestor_cache: BTreeMap::new(),
            total_scopes: 0,
            max_depth: 0,
        };

        // Create the global scope
        let global_scope = Scope::new(
            global_scope_id,
            ScopeKind::Global,
            None,
            ScopeLocation::unknown(),
            0,
        );

        tree.add_scope(global_scope);
        tree
    }

    /// Add a new scope to the tree
    pub fn add_scope(&mut self, scope: Scope) -> ScopeId {
        let scope_id = scope.id;
        let parent_id = scope.parent_id;
        let depth = scope.depth;

        // Store scope in arena
        let scope_ref = self.scopes_arena.alloc(scope);

        // SAFETY: Arena lifetime extends beyond all references, and we have exclusive access
        let static_ref: &'static mut Scope = unsafe { std::mem::transmute(scope_ref) };

        // Update indices
        self.scopes_by_id.insert(scope_id, static_ref);

        // Update parent-child relationships
        if let Some(parent_id) = parent_id {
            if let Some(parent_scope) = self.scopes_by_id.get_mut(&parent_id) {
                parent_scope.add_child(scope_id);
            }
        }

        // Invalidate ancestor cache for this scope and descendants
        self.invalidate_ancestor_cache(scope_id);

        self.total_scopes += 1;
        self.max_depth = self.max_depth.max(depth);

        scope_id
    }

    /// Create a new scope with auto-generated ID
    pub fn create_scope(&mut self, parent_id: Option<ScopeId>) -> ScopeId {
        let scope_id = ScopeId::from_raw(self.total_scopes as u32);
        let depth = if let Some(parent_id) = parent_id {
            self.get_scope(parent_id).map(|p| p.depth + 1).unwrap_or(0)
        } else {
            0
        };

        let scope = Scope::new(
            scope_id,
            ScopeKind::Block, // Default scope kind
            parent_id,
            ScopeLocation::unknown(),
            depth,
        );

        self.add_scope(scope)
    }

    /// Get a scope by its ID
    pub fn get_scope(&self, id: ScopeId) -> Option<&Scope> {
        self.scopes_by_id.get(&id).map(|scope_ref| &**scope_ref)
    }

    /// Get a mutable scope by its ID
    pub fn get_scope_mut(&mut self, id: ScopeId) -> Option<&mut Scope> {
        self.scopes_by_id
            .get_mut(&id)
            .map(|scope_ref| &mut **scope_ref)
    }

    /// Get the root (global) scope
    pub fn root_scope(&self) -> &Scope {
        self.get_scope(self.root_scope_id)
            .expect("Root scope should always exist")
    }

    /// Get the current active scope
    pub fn current_scope(&self) -> &Scope {
        self.get_scope(self.current_scope_id)
            .expect("Current scope should always exist")
    }

    /// Get the current active scope mutably
    pub fn current_scope_mut(&mut self) -> &mut Scope {
        self.get_scope_mut(self.current_scope_id)
            .expect("Current scope should always exist")
    }

    /// Set the current active scope
    pub fn set_current_scope(&mut self, scope_id: ScopeId) {
        if self.scopes_by_id.contains_key(&scope_id) {
            self.current_scope_id = scope_id;
        }
    }

    /// Enter a new scope (create and set as current)
    pub fn enter_scope(
        &mut self,
        kind: ScopeKind,
        name: Option<InternedString>,
        location: ScopeLocation,
        scope_id: ScopeId,
    ) -> &mut Scope {
        let parent_id = Some(self.current_scope_id);
        let depth = self.current_scope().depth + 1;

        let scope = if let Some(name) = name {
            Scope::named(scope_id, name, kind, parent_id, location, depth)
        } else {
            Scope::new(scope_id, kind, parent_id, location, depth)
        };

        self.add_scope(scope);
        self.current_scope_id = scope_id;
        self.get_scope_mut(scope_id)
            .expect("Scope should exist after adding")
    }

    /// Exit the current scope and return to parent
    pub fn exit_scope(&mut self, end_line: u32, end_column: u32) -> Option<ScopeId> {
        let current_scope = self.current_scope_mut();
        current_scope.close(end_line, end_column);

        let parent_id = current_scope.parent_id;

        if let Some(parent_id) = parent_id {
            self.current_scope_id = parent_id;
        }

        parent_id
    }

    /// Get all ancestor scopes of a given scope (cached for performance)
    pub fn get_ancestors(&mut self, scope_id: ScopeId) -> Vec<ScopeId> {
        if let Some(cached) = self.ancestor_cache.get(&scope_id) {
            return cached.clone();
        }

        let mut ancestors = Vec::new();
        let mut current_id = scope_id;

        while let Some(scope) = self.get_scope(current_id) {
            if let Some(parent_id) = scope.parent_id {
                ancestors.push(parent_id);
                current_id = parent_id;
            } else {
                break;
            }
        }

        self.ancestor_cache.insert(scope_id, ancestors.clone());
        ancestors
    }

    /// Resolve a name starting from a specific scope, walking up the scope chain
    pub fn resolve_name(
        &mut self,
        start_scope_id: ScopeId,
        name: InternedString,
    ) -> Option<SymbolId> {
        // First check the starting scope
        if let Some(scope) = self.get_scope(start_scope_id) {
            if let Some(symbol_id) = scope.get_symbol(name) {
                return Some(symbol_id);
            }
        }

        // Then walk up the ancestor chain
        let ancestors = self.get_ancestors(start_scope_id);
        for ancestor_id in ancestors {
            if let Some(scope) = self.get_scope(ancestor_id) {
                if let Some(symbol_id) = scope.get_symbol(name) {
                    return Some(symbol_id);
                }
            }
        }

        None
    }

    /// Resolve a name from the current scope
    pub fn resolve_name_current(&mut self, name: InternedString) -> Option<SymbolId> {
        self.resolve_name(self.current_scope_id, name)
    }

    /// Check if a name would shadow an existing symbol
    pub fn would_shadow(
        &mut self,
        scope_id: ScopeId,
        name: InternedString,
    ) -> Option<(SymbolId, ScopeId)> {
        let ancestors = self.get_ancestors(scope_id);
        for ancestor_id in ancestors {
            if let Some(scope) = self.get_scope(ancestor_id) {
                if let Some(symbol_id) = scope.get_symbol(name) {
                    return Some((symbol_id, ancestor_id));
                }
            }
        }
        None
    }

    /// Get all scopes at a specific depth
    pub fn scopes_at_depth(&self, depth: u32) -> Vec<&Scope> {
        self.scopes_by_id
            .values()
            .filter_map(|scope_ref| {
                let scope: &Scope = scope_ref;
                if scope.depth == depth {
                    Some(scope)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get all child scopes of a given scope
    pub fn get_children(&self, scope_id: ScopeId) -> Vec<&Scope> {
        if let Some(scope) = self.get_scope(scope_id) {
            scope
                .children
                .iter()
                .filter_map(|&child_id| self.get_scope(child_id))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get all descendant scopes (children, grandchildren, etc.)
    pub fn get_descendants(&self, scope_id: ScopeId) -> Vec<&Scope> {
        let mut descendants = Vec::new();
        let mut to_visit = vec![scope_id];

        while let Some(current_id) = to_visit.pop() {
            let children = self.get_children(current_id);
            for child in children {
                descendants.push(child);
                to_visit.push(child.id);
            }
        }

        descendants
    }

    /// Find scopes matching a predicate
    pub fn find_scopes<F>(&self, predicate: F) -> Vec<&Scope>
    where
        F: Fn(&Scope) -> bool,
    {
        self.scopes_by_id
            .values()
            .filter_map(|scope_ref| {
                let scope: &Scope = scope_ref;
                if predicate(scope) {
                    Some(scope)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get all active (unclosed) scopes
    pub fn active_scopes(&self) -> Vec<&Scope> {
        self.find_scopes(|scope| scope.is_active)
    }

    /// Invalidate ancestor cache for a scope and its descendants
    fn invalidate_ancestor_cache(&mut self, scope_id: ScopeId) {
        self.ancestor_cache.remove(&scope_id);

        // Collect descendant IDs first to avoid borrowing conflicts
        let descendant_ids: Vec<ScopeId> = self
            .get_descendants(scope_id)
            .into_iter()
            .map(|scope| scope.id)
            .collect();

        // Then invalidate cache for all descendants
        for descendant_id in descendant_ids {
            self.ancestor_cache.remove(&descendant_id);
        }
    }

    /// Get scope tree statistics
    pub fn stats(&self) -> ScopeTreeStats {
        let arena_stats = self.scopes_arena.stats();
        let mut scopes_by_kind = BTreeMap::new();
        let mut scopes_by_depth = BTreeMap::new();
        let mut total_symbols = 0;

        for scope_ref in self.scopes_by_id.values() {
            let scope: &Scope = scope_ref;
            *scopes_by_kind.entry(scope.kind).or_insert(0) += 1;
            *scopes_by_depth.entry(scope.depth).or_insert(0) += 1;
            total_symbols += scope.symbols.len();
        }

        let memory_usage = arena_stats.total_bytes_allocated
            + self.scopes_by_id.len() * std::mem::size_of::<(ScopeId, &Scope)>()
            + self.ancestor_cache.len() * std::mem::size_of::<(ScopeId, Vec<ScopeId>)>();

        ScopeTreeStats {
            total_scopes: self.total_scopes,
            max_depth: self.max_depth,
            scopes_by_kind,
            scopes_by_depth,
            total_symbols,
            arena_stats,
            cache_entries: self.ancestor_cache.len(),
            memory_usage,
        }
    }

    /// Get the number of scopes
    pub fn len(&self) -> usize {
        self.total_scopes
    }

    /// Check if the scope tree is empty (only global scope)
    pub fn is_empty(&self) -> bool {
        self.total_scopes <= 1
    }

    /// Add an existing symbol to a specific scope
    pub fn add_symbol_to_scope(
        &mut self,
        scope_id: ScopeId,
        symbol_id: SymbolId,
    ) -> Result<(), ScopeError> {
        if let Some(scope) = self.get_scope_mut(scope_id) {
            if !scope.symbols.contains(&symbol_id) {
                scope.symbols.push(symbol_id);
            }
            Ok(())
        } else {
            Err(ScopeError::ScopeNotFound { scope_id })
        }
    }

    /// Look up a symbol by name in the scope chain, walking up from the given scope
    pub fn lookup_symbol_in_scope_chain(
        &mut self,
        start_scope_id: ScopeId,
        name: InternedString,
        symbol_table: &super::SymbolTable,
    ) -> Option<SymbolId> {
        // Start with the given scope
        let mut current_scope_id = start_scope_id;

        loop {
            // Check current scope for the symbol by name
            if let Some(scope) = self.get_scope(current_scope_id) {
                // Look through symbols in this scope to find one with the matching name
                for &symbol_id in &scope.symbols {
                    if let Some(symbol) = symbol_table.get_symbol(symbol_id) {
                        if symbol.name == name {
                            return Some(symbol_id);
                        }
                    }
                }

                // Move to parent scope
                if let Some(parent_id) = scope.parent_id {
                    current_scope_id = parent_id;
                } else {
                    // Reached root scope, symbol not found
                    break;
                }
            } else {
                // Invalid scope ID
                break;
            }
        }

        None
    }
}

/// Statistics about scope tree usage
#[derive(Debug, Clone)]
pub struct ScopeTreeStats {
    pub total_scopes: usize,
    pub max_depth: u32,
    pub scopes_by_kind: BTreeMap<ScopeKind, usize>,
    pub scopes_by_depth: BTreeMap<u32, usize>,
    pub total_symbols: usize,
    pub arena_stats: super::ArenaStats,
    pub cache_entries: usize,
    pub memory_usage: usize,
}

impl ScopeTreeStats {
    /// Get the most common scope kind
    pub fn most_common_scope_kind(&self) -> Option<(ScopeKind, usize)> {
        self.scopes_by_kind
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(&kind, &count)| (kind, count))
    }

    /// Get the average number of symbols per scope
    pub fn average_symbols_per_scope(&self) -> f64 {
        if self.total_scopes > 0 {
            self.total_symbols as f64 / self.total_scopes as f64
        } else {
            0.0
        }
    }

    /// Get memory usage per scope
    pub fn memory_per_scope(&self) -> f64 {
        if self.total_scopes > 0 {
            self.memory_usage as f64 / self.total_scopes as f64
        } else {
            0.0
        }
    }

    /// Check if the scope tree is deeply nested (potential performance concern)
    pub fn is_deeply_nested(&self) -> bool {
        self.max_depth > 20
    }

    /// Get cache hit ratio (if we track cache misses in the future)
    pub fn cache_utilization(&self) -> f64 {
        if self.total_scopes > 0 {
            self.cache_entries as f64 / self.total_scopes as f64
        } else {
            0.0
        }
    }
}

/// Integration with Symbol System for complete name resolution
pub struct NameResolver<'a> {
    scope_tree: &'a mut ScopeTree,
    symbol_table: &'a SymbolTable,
}

impl<'a> NameResolver<'a> {
    /// Create a new name resolver
    pub fn new(scope_tree: &'a mut ScopeTree, symbol_table: &'a SymbolTable) -> Self {
        Self {
            scope_tree,
            symbol_table,
        }
    }

    /// Resolve a name to a symbol, returning the symbol and its scope
    pub fn resolve_symbol(&mut self, name: InternedString) -> Option<(&Symbol, ScopeId)> {
        self.resolve_symbol_from_scope(self.scope_tree.current_scope_id, name)
    }

    /// Resolve a name from a specific scope
    pub fn resolve_symbol_from_scope(
        &mut self,
        start_scope: ScopeId,
        name: InternedString,
    ) -> Option<(&Symbol, ScopeId)> {
        // Try the starting scope first
        if let Some(scope) = self.scope_tree.get_scope(start_scope) {
            if let Some(symbol_id) = scope.get_symbol(name) {
                if let Some(symbol) = self.symbol_table.get_symbol(symbol_id) {
                    return Some((symbol, start_scope));
                }
            }
        }

        // Walk up the ancestor chain
        let ancestors = self.scope_tree.get_ancestors(start_scope);
        for ancestor_id in ancestors {
            if let Some(scope) = self.scope_tree.get_scope(ancestor_id) {
                if let Some(symbol_id) = scope.get_symbol(name) {
                    if let Some(symbol) = self.symbol_table.get_symbol(symbol_id) {
                        return Some((symbol, ancestor_id));
                    }
                }
            }
        }

        None
    }

    /// Check for shadowing conflicts when declaring a new symbol
    pub fn check_shadowing(&mut self, scope_id: ScopeId, name: InternedString) -> ShadowingResult {
        // Check if name already exists in the current scope
        if let Some(scope) = self.scope_tree.get_scope(scope_id) {
            if scope.has_symbol(name) {
                return ShadowingResult::Conflict;
            }
        }

        // Get scope kind first to avoid borrowing conflicts
        let allows_shadowing = self
            .scope_tree
            .get_scope(scope_id)
            .map(|scope| scope.kind.allows_shadowing())
            .unwrap_or(false);

        // Check if it would shadow a symbol in parent scopes
        if let Some((symbol, shadowed_scope)) = self.resolve_symbol_from_scope(scope_id, name) {
            if allows_shadowing {
                ShadowingResult::Shadows {
                    symbol_id: symbol.id,
                    scope_id: shadowed_scope,
                }
            } else {
                ShadowingResult::Forbidden
            }
        } else {
            ShadowingResult::Ok
        }
    }
}

/// Result of checking for shadowing conflicts
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShadowingResult {
    /// No conflict, safe to declare
    Ok,
    /// Name already exists in this scope
    Conflict,
    /// Would shadow a symbol (but is allowed)
    Shadows {
        symbol_id: SymbolId,
        scope_id: ScopeId,
    },
    /// Shadowing is forbidden in this scope kind
    Forbidden,
}

/// Errors that can occur during scope operations
#[derive(Debug, Clone)]
pub enum ScopeError {
    ScopeNotFound {
        scope_id: ScopeId,
    },
    SymbolNotFound {
        symbol_id: SymbolId,
    },
    SymbolAlreadyExists {
        symbol_id: SymbolId,
        scope_id: ScopeId,
    },
}

impl fmt::Display for ScopeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScopeError::ScopeNotFound { scope_id } => {
                write!(f, "Scope {} not found", scope_id)
            }
            ScopeError::SymbolNotFound { symbol_id } => {
                write!(f, "Symbol {} not found", symbol_id)
            }
            ScopeError::SymbolAlreadyExists {
                symbol_id,
                scope_id,
            } => {
                write!(
                    f,
                    "Symbol {} already exists in scope {}",
                    symbol_id, scope_id
                )
            }
        }
    }
}

impl std::error::Error for ScopeError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tast::{
        Mutability, SourceLocation, StringInterner, Symbol, SymbolKind, SymbolTable, TypeId,
    };

    fn create_test_setup() -> (StringInterner, ScopeTree, SymbolTable) {
        let interner = StringInterner::new();
        let scope_tree = ScopeTree::new(ScopeId::from_raw(0));
        let symbol_table = SymbolTable::new();
        (interner, scope_tree, symbol_table)
    }

    #[test]
    fn test_scope_kind_properties() {
        assert!(ScopeKind::Class.can_declare_types());
        assert!(ScopeKind::Class.can_declare_functions());
        assert!(!ScopeKind::Block.can_declare_types());
        assert!(!ScopeKind::Block.can_declare_functions());

        assert!(ScopeKind::Function.allows_shadowing());
        assert!(!ScopeKind::Class.allows_shadowing());

        assert!(ScopeKind::Function.creates_lifetime_boundary());
        assert!(!ScopeKind::Global.creates_lifetime_boundary());

        assert!(ScopeKind::Global.allows_variable_escape());
        assert!(!ScopeKind::Function.allows_variable_escape());
    }

    #[test]
    fn test_scope_location() {
        let loc = ScopeLocation::new(1, 10, 5);
        assert!(loc.is_valid());
        assert_eq!(loc.start_line, 10);
        assert_eq!(loc.start_column, 5);
        assert_eq!(loc.end_line, 0);

        let mut loc = ScopeLocation::new(1, 10, 5);
        loc.set_end(20, 15);
        assert_eq!(loc.end_line, 20);
        assert_eq!(loc.end_column, 15);

        let unknown = ScopeLocation::unknown();
        assert!(!unknown.is_valid());
    }

    #[test]
    fn test_scope_creation() {
        let (interner, mut _scope_tree, _symbol_table) = create_test_setup();
        let name = interner.intern("TestClass");

        let scope = Scope::named(
            ScopeId::from_raw(1),
            name,
            ScopeKind::Class,
            Some(ScopeId::from_raw(0)),
            ScopeLocation::new(1, 10, 1),
            1,
        );

        assert_eq!(scope.kind, ScopeKind::Class);
        assert_eq!(scope.name, Some(name));
        assert_eq!(scope.parent_id, Some(ScopeId::from_raw(0)));
        assert_eq!(scope.depth, 1);
        assert!(scope.is_active);
    }

    #[test]
    fn test_scope_tree_basic_operations() {
        let (_interner, scope_tree, _symbol_table) = create_test_setup();

        // Test root scope
        let root = scope_tree.root_scope();
        assert_eq!(root.kind, ScopeKind::Global);
        assert_eq!(root.depth, 0);
        assert!(root.parent_id.is_none());

        // Test current scope
        let current = scope_tree.current_scope();
        assert_eq!(current.id, root.id);

        // Test scope count
        assert_eq!(scope_tree.len(), 1);
        assert!(scope_tree.is_empty()); // Only global scope
    }

    #[test]
    fn test_scope_tree_nesting() {
        let (interner, mut scope_tree, _symbol_table) = create_test_setup();

        let class_name = interner.intern("MyClass");
        let function_name = interner.intern("myMethod");

        // Enter class scope
        {
            let class_scope = scope_tree.enter_scope(
                ScopeKind::Class,
                Some(class_name),
                ScopeLocation::new(1, 5, 1),
                ScopeId::from_raw(1),
            );
            assert_eq!(class_scope.kind, ScopeKind::Class);
            assert_eq!(class_scope.depth, 1);
        }

        // Enter function scope
        {
            let function_scope = scope_tree.enter_scope(
                ScopeKind::Function,
                Some(function_name),
                ScopeLocation::new(1, 10, 5),
                ScopeId::from_raw(2),
            );
            assert_eq!(function_scope.kind, ScopeKind::Function);
            assert_eq!(function_scope.depth, 2);
        }

        // Check current scope
        let current = scope_tree.current_scope();
        assert_eq!(current.id, ScopeId::from_raw(2));

        // Check parent-child relationships
        let function_scope = scope_tree.get_scope(ScopeId::from_raw(2)).unwrap();
        let class_scope = scope_tree.get_scope(ScopeId::from_raw(1)).unwrap();
        assert_eq!(function_scope.parent_id, Some(ScopeId::from_raw(1)));
        assert!(class_scope.children.contains(&ScopeId::from_raw(2)));

        // Exit scopes
        scope_tree.exit_scope(15, 5);
        assert_eq!(scope_tree.current_scope().id, ScopeId::from_raw(1));

        scope_tree.exit_scope(20, 1);
        assert_eq!(scope_tree.current_scope().id, ScopeId::from_raw(0));

        assert_eq!(scope_tree.len(), 3); // Global + Class + Function
    }

    #[test]
    fn test_scope_symbol_management() {
        let (interner, mut scope_tree, _symbol_table) = create_test_setup();

        let var_name = interner.intern("x");
        let symbol_id = SymbolId::from_raw(1);

        // Add symbol to current scope
        let current_scope = scope_tree.current_scope_mut();
        current_scope.add_symbol(symbol_id, var_name);

        // Check symbol presence
        assert!(current_scope.has_symbol(var_name));
        assert_eq!(current_scope.get_symbol(var_name), Some(symbol_id));
        assert_eq!(current_scope.symbols.len(), 1);

        // Test symbol removal
        current_scope.remove_symbol(symbol_id, var_name);
        assert!(!current_scope.has_symbol(var_name));
        assert_eq!(current_scope.symbols.len(), 0);
    }

    #[test]
    fn test_scope_tree_ancestors() {
        let (interner, mut scope_tree, _symbol_table) = create_test_setup();

        // Create nested scopes: Global -> Class -> Function -> Block
        scope_tree.enter_scope(
            ScopeKind::Class,
            Some(interner.intern("MyClass")),
            ScopeLocation::new(1, 1, 1),
            ScopeId::from_raw(1),
        );

        scope_tree.enter_scope(
            ScopeKind::Function,
            Some(interner.intern("myMethod")),
            ScopeLocation::new(1, 5, 5),
            ScopeId::from_raw(2),
        );

        scope_tree.enter_scope(
            ScopeKind::Block,
            None,
            ScopeLocation::new(1, 10, 9),
            ScopeId::from_raw(3),
        );

        // Check ancestors from deepest scope
        let ancestors = scope_tree.get_ancestors(ScopeId::from_raw(3));
        assert_eq!(ancestors.len(), 3);
        assert_eq!(ancestors[0], ScopeId::from_raw(2)); // Function
        assert_eq!(ancestors[1], ScopeId::from_raw(1)); // Class
        assert_eq!(ancestors[2], ScopeId::from_raw(0)); // Global

        // Check ancestors from function scope
        let ancestors = scope_tree.get_ancestors(ScopeId::from_raw(2));
        assert_eq!(ancestors.len(), 2);
        assert_eq!(ancestors[0], ScopeId::from_raw(1)); // Class
        assert_eq!(ancestors[1], ScopeId::from_raw(0)); // Global

        // Global scope has no ancestors
        let ancestors = scope_tree.get_ancestors(ScopeId::from_raw(0));
        assert_eq!(ancestors.len(), 0);
    }

    #[test]
    fn test_scope_name_resolution() {
        let (interner, mut scope_tree, _symbol_table) = create_test_setup();

        let var_name = interner.intern("x");
        let global_symbol = SymbolId::from_raw(1);
        let local_symbol = SymbolId::from_raw(2);

        // Add symbol to global scope
        let global_scope = scope_tree.get_scope_mut(ScopeId::from_raw(0)).unwrap();
        global_scope.add_symbol(global_symbol, var_name);

        // Enter function scope
        scope_tree.enter_scope(
            ScopeKind::Function,
            Some(interner.intern("test")),
            ScopeLocation::new(1, 5, 1),
            ScopeId::from_raw(1),
        );

        // Should resolve to global symbol
        let resolved = scope_tree.resolve_name_current(var_name);
        assert_eq!(resolved, Some(global_symbol));

        // Add local symbol with same name
        let function_scope = scope_tree.current_scope_mut();
        function_scope.add_symbol(local_symbol, var_name);

        // Should now resolve to local symbol (shadowing)
        let resolved = scope_tree.resolve_name_current(var_name);
        assert_eq!(resolved, Some(local_symbol));

        // Should still find global symbol from global scope
        let resolved_global = scope_tree.resolve_name(ScopeId::from_raw(0), var_name);
        assert_eq!(resolved_global, Some(global_symbol));
    }

    #[test]
    fn test_scope_tree_children_and_descendants() {
        let (interner, mut scope_tree, _symbol_table) = create_test_setup();

        // Create tree: Global -> Class -> (Function1, Function2) -> Block
        scope_tree.enter_scope(
            ScopeKind::Class,
            Some(interner.intern("MyClass")),
            ScopeLocation::new(1, 1, 1),
            ScopeId::from_raw(1),
        );

        scope_tree.enter_scope(
            ScopeKind::Function,
            Some(interner.intern("method1")),
            ScopeLocation::new(1, 5, 5),
            ScopeId::from_raw(2),
        );

        scope_tree.enter_scope(
            ScopeKind::Block,
            None,
            ScopeLocation::new(1, 10, 9),
            ScopeId::from_raw(3),
        );

        scope_tree.exit_scope(15, 9);
        scope_tree.exit_scope(20, 5);

        scope_tree.enter_scope(
            ScopeKind::Function,
            Some(interner.intern("method2")),
            ScopeLocation::new(1, 25, 5),
            ScopeId::from_raw(4),
        );

        // Test children of class scope
        let class_children = scope_tree.get_children(ScopeId::from_raw(1));
        assert_eq!(class_children.len(), 2);
        assert!(class_children.iter().any(|s| s.id == ScopeId::from_raw(2)));
        assert!(class_children.iter().any(|s| s.id == ScopeId::from_raw(4)));

        // Test descendants of class scope
        let class_descendants = scope_tree.get_descendants(ScopeId::from_raw(1));
        assert_eq!(class_descendants.len(), 3); // Function1, Function2, Block

        // Test children of global scope
        let global_children = scope_tree.get_children(ScopeId::from_raw(0));
        assert_eq!(global_children.len(), 1);
        assert_eq!(global_children[0].id, ScopeId::from_raw(1));
    }

    #[test]
    fn test_name_resolver() {
        let (interner, mut scope_tree, mut symbol_table) = create_test_setup();

        let var_name = interner.intern("testVar");

        // Create a symbol and add to symbol table
        let symbol = Symbol::variable(
            SymbolId::from_raw(1),
            var_name,
            TypeId::from_raw(1),
            ScopeId::from_raw(0),
            Mutability::Mutable,
            SourceLocation::new(1, 5, 1, 50),
        );
        symbol_table.add_symbol(symbol);

        // Add symbol to global scope
        let global_scope = scope_tree.get_scope_mut(ScopeId::from_raw(0)).unwrap();
        global_scope.add_symbol(SymbolId::from_raw(1), var_name);

        // Test resolution in its own scope
        {
            let mut resolver = NameResolver::new(&mut scope_tree, &symbol_table);

            // Test resolution
            let resolved = resolver.resolve_symbol(var_name);
            assert!(resolved.is_some());

            let (symbol, scope_id) = resolved.unwrap();
            assert_eq!(symbol.id, SymbolId::from_raw(1));
            assert_eq!(scope_id, ScopeId::from_raw(0));

            // Test shadowing check
            let shadowing_result = resolver.check_shadowing(ScopeId::from_raw(0), var_name);
            assert_eq!(shadowing_result, ShadowingResult::Conflict);
        }

        // Enter function scope and test shadowing
        scope_tree.enter_scope(
            ScopeKind::Function,
            Some(interner.intern("test")),
            ScopeLocation::new(1, 10, 1),
            ScopeId::from_raw(1),
        );

        // Test shadowing in new scope
        {
            let mut resolver = NameResolver::new(&mut scope_tree, &symbol_table);
            let shadowing_result = resolver.check_shadowing(ScopeId::from_raw(1), var_name);
            assert!(matches!(shadowing_result, ShadowingResult::Shadows { .. }));
        }
    }

    #[test]
    fn test_scope_tree_stats() {
        let (interner, mut scope_tree, _symbol_table) = create_test_setup();

        // Create various scope types
        scope_tree.enter_scope(
            ScopeKind::Class,
            Some(interner.intern("Class1")),
            ScopeLocation::new(1, 1, 1),
            ScopeId::from_raw(1),
        );

        scope_tree.enter_scope(
            ScopeKind::Function,
            Some(interner.intern("method1")),
            ScopeLocation::new(1, 5, 5),
            ScopeId::from_raw(2),
        );

        scope_tree.enter_scope(
            ScopeKind::Block,
            None,
            ScopeLocation::new(1, 10, 9),
            ScopeId::from_raw(3),
        );

        let stats = scope_tree.stats();
        assert_eq!(stats.total_scopes, 4); // Global + Class + Function + Block
        assert_eq!(stats.max_depth, 3);
        assert!(stats.scopes_by_kind.len() >= 3);
        assert!(stats.memory_usage > 0);

        let most_common = stats.most_common_scope_kind();
        assert!(most_common.is_some());

        assert!(stats.memory_per_scope() > 0.0);
        assert!(!stats.is_deeply_nested());
    }

    #[test]
    fn test_scope_symbol_declaration_rules() {
        let (_interner, scope_tree, _symbol_table) = create_test_setup();

        let global_scope = scope_tree.root_scope();
        assert!(global_scope.can_declare_symbol(SymbolKind::Class));
        assert!(global_scope.can_declare_symbol(SymbolKind::Function));
        assert!(global_scope.can_declare_symbol(SymbolKind::Variable));
        assert!(!global_scope.can_declare_symbol(SymbolKind::Field));

        // Test class scope rules
        let class_scope = Scope::new(
            ScopeId::from_raw(1),
            ScopeKind::Class,
            Some(ScopeId::from_raw(0)),
            ScopeLocation::new(1, 5, 1),
            1,
        );

        assert!(class_scope.can_declare_symbol(SymbolKind::Field));
        assert!(class_scope.can_declare_symbol(SymbolKind::Property));
        assert!(class_scope.can_declare_symbol(SymbolKind::Function));
        assert!(!class_scope.can_declare_symbol(SymbolKind::EnumVariant));

        // Test enum scope rules
        let enum_scope = Scope::new(
            ScopeId::from_raw(2),
            ScopeKind::Enum,
            Some(ScopeId::from_raw(0)),
            ScopeLocation::new(1, 10, 1),
            1,
        );

        assert!(enum_scope.can_declare_symbol(SymbolKind::EnumVariant));
        assert!(!enum_scope.can_declare_symbol(SymbolKind::Field));
    }
}
