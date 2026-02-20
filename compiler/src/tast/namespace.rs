//! Namespace and Package Management
//!
//! This module provides infrastructure for managing package hierarchies,
//! namespace resolution, and import tracking for proper type path resolution.

use super::{InternedString, ScopeId, StringInterner, SymbolId};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

/// Unique identifier for a package
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PackageId(u32);

impl PackageId {
    /// Create a new package ID
    pub fn new(id: u32) -> Self {
        PackageId(id)
    }

    /// Get the raw ID value
    pub fn as_raw(&self) -> u32 {
        self.0
    }

    /// Root package ID (no package)
    pub fn root() -> Self {
        PackageId(0)
    }
}

/// Information about a package in the hierarchy
#[derive(Debug, Clone)]
pub struct PackageInfo {
    /// Full package path (e.g., ["com", "example", "game"])
    pub full_path: Vec<InternedString>,

    /// Parent package (None for root)
    pub parent: Option<PackageId>,

    /// Symbols defined directly in this package
    pub symbols: HashMap<InternedString, SymbolId>,

    /// Sub-packages
    pub sub_packages: HashMap<InternedString, PackageId>,

    /// Package visibility (for internal packages)
    pub is_internal: bool,
}

impl PackageInfo {
    /// Create a new package info
    pub fn new(full_path: Vec<InternedString>, parent: Option<PackageId>) -> Self {
        PackageInfo {
            full_path,
            parent,
            symbols: HashMap::new(),
            sub_packages: HashMap::new(),
            is_internal: false,
        }
    }

    /// Get the package name (last segment of path)
    pub fn name(&self) -> Option<InternedString> {
        self.full_path.last().copied()
    }

    /// Get the full qualified path as a string
    pub fn qualified_path(&self, interner: &StringInterner) -> String {
        self.full_path
            .iter()
            .map(|&s| interner.get(s).unwrap_or("<unknown>"))
            .collect::<Vec<_>>()
            .join(".")
    }
}

/// Qualified path representing a type reference
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QualifiedPath {
    /// Package path segments (e.g., ["com", "example"])
    pub package: Vec<InternedString>,

    /// Type name
    pub name: InternedString,
}

impl QualifiedPath {
    /// Create a new qualified path
    pub fn new(package: Vec<InternedString>, name: InternedString) -> Self {
        QualifiedPath { package, name }
    }

    /// Create from a simple name (no package)
    pub fn simple(name: InternedString) -> Self {
        QualifiedPath {
            package: Vec::new(),
            name,
        }
    }

    /// Get the full qualified name
    pub fn to_string(&self, interner: &StringInterner) -> String {
        if self.package.is_empty() {
            interner.get(self.name).unwrap_or("<unknown>").to_string()
        } else {
            format!(
                "{}.{}",
                self.package
                    .iter()
                    .map(|&s| interner.get(s).unwrap_or("<unknown>"))
                    .collect::<Vec<_>>()
                    .join("."),
                interner.get(self.name).unwrap_or("<unknown>")
            )
        }
    }
}

/// Import entry representing a single import statement
#[derive(Debug, Clone)]
pub struct ImportEntry {
    /// Full package path being imported
    pub package_path: QualifiedPath,

    /// Alias for the import (if any)
    pub alias: Option<InternedString>,

    /// For wildcard imports, specific exclusions
    pub exclusions: Vec<InternedString>,

    /// Whether this is a wildcard import (*)
    pub is_wildcard: bool,

    /// Source location for error reporting
    pub location: super::SourceLocation,
}

/// Manages package hierarchy and namespace resolution
pub struct NamespaceResolver {
    /// All packages by ID
    packages: HashMap<PackageId, PackageInfo>,

    /// Package lookup by full path
    package_paths: HashMap<Vec<InternedString>, PackageId>,

    /// Current package context
    current_package: Option<PackageId>,

    /// Next package ID
    next_package_id: u32,

    /// Files that have been loaded (to avoid reloading)
    loaded_files: HashSet<PathBuf>,

    /// Source paths for user code (checked first)
    source_paths: Vec<PathBuf>,

    /// Standard library paths (checked after source paths)
    stdlib_paths: Vec<PathBuf>,
}

impl NamespaceResolver {
    /// Create a new namespace resolver
    pub fn new() -> Self {
        let mut resolver = NamespaceResolver {
            packages: HashMap::new(),
            package_paths: HashMap::new(),
            current_package: None,
            next_package_id: 1, // 0 is reserved for root
            loaded_files: HashSet::new(),
            source_paths: Vec::new(),
            stdlib_paths: Vec::new(),
        };

        // Create root package
        let root = PackageInfo::new(Vec::new(), None);
        resolver.packages.insert(PackageId::root(), root);
        resolver.package_paths.insert(Vec::new(), PackageId::root());

        resolver
    }

    /// Set source paths for user code (checked first during import resolution)
    pub fn set_source_paths(&mut self, paths: Vec<PathBuf>) {
        self.source_paths = paths;
    }

    /// Add an additional source path (e.g. from an rpkg package).
    pub fn add_source_path(&mut self, path: PathBuf) {
        if !self.source_paths.contains(&path) {
            self.source_paths.push(path);
        }
    }

    /// Set standard library paths (checked after source paths)
    pub fn set_stdlib_paths(&mut self, paths: Vec<PathBuf>) {
        self.stdlib_paths = paths;
    }

    /// Mark a file as loaded
    pub fn mark_file_loaded(&mut self, path: PathBuf) {
        self.loaded_files.insert(path);
    }

    /// Check if a file has been loaded
    pub fn is_file_loaded(&self, path: &PathBuf) -> bool {
        self.loaded_files.contains(path)
    }

    /// Get all loaded files
    pub fn loaded_files(&self) -> &HashSet<PathBuf> {
        &self.loaded_files
    }

    /// Resolve a qualified path to a filesystem path
    /// Example: "haxe.iterators.ArrayIterator" -> "haxe/iterators/ArrayIterator.hx"
    ///
    /// Lookup order:
    /// 1. Source paths (project workspace)
    /// 2. Standard library paths
    ///
    /// Returns the first existing file path found, or None if not found
    /// Check if a qualified path refers to a file that is already loaded
    pub fn is_qualified_path_loaded(&self, qualified_path: &str) -> bool {
        let file_path = qualified_path.replace('.', "/") + ".hx";
        for source_path in &self.source_paths {
            let full_path = source_path.join(&file_path);
            if full_path.exists() && self.is_file_loaded(&full_path) {
                return true;
            }
        }
        for stdlib_path in &self.stdlib_paths {
            let full_path = stdlib_path.join(&file_path);
            if full_path.exists() && self.is_file_loaded(&full_path) {
                return true;
            }
        }
        false
    }

    pub fn resolve_qualified_path_to_file(&self, qualified_path: &str) -> Option<PathBuf> {
        self.resolve_qualified_path_impl(qualified_path, true)
    }

    /// Resolve a qualified path ignoring the loaded-files check.
    /// Used by import loading which needs to compile files even if BLADE cache
    /// has pre-registered their symbols (BLADE doesn't preserve full TAST state).
    pub fn resolve_qualified_path_to_file_force(&self, qualified_path: &str) -> Option<PathBuf> {
        self.resolve_qualified_path_impl(qualified_path, false)
    }

    fn resolve_qualified_path_impl(
        &self,
        qualified_path: &str,
        check_loaded: bool,
    ) -> Option<PathBuf> {
        // Convert qualified path to file path
        // "haxe.iterators.ArrayIterator" -> "haxe/iterators/ArrayIterator.hx"
        let file_path = qualified_path.replace('.', "/") + ".hx";

        // First check source paths (user workspace)
        for source_path in &self.source_paths {
            let full_path = source_path.join(&file_path);
            if full_path.exists() {
                if check_loaded && self.is_file_loaded(&full_path) {
                    // Already loaded from cache - skip
                    return None;
                }
                return Some(full_path);
            }
        }

        // Then check stdlib paths
        for stdlib_path in &self.stdlib_paths {
            let full_path = stdlib_path.join(&file_path);
            if full_path.exists() {
                if check_loaded && self.is_file_loaded(&full_path) {
                    // Already loaded from cache - skip
                    return None;
                }
                return Some(full_path);
            }
        }

        None
    }

    /// Resolve a QualifiedPath to a filesystem path
    pub fn resolve_to_file(
        &self,
        path: &QualifiedPath,
        interner: &StringInterner,
    ) -> Option<PathBuf> {
        let qualified_str = path.to_string(interner);
        self.resolve_qualified_path_to_file(&qualified_str)
    }

    /// Get or create a package by path
    pub fn get_or_create_package(&mut self, path: Vec<InternedString>) -> PackageId {
        if let Some(&id) = self.package_paths.get(&path) {
            return id;
        }

        // Create parent packages if needed
        let parent_id = if path.is_empty() {
            None
        } else {
            let parent_path = path[..path.len() - 1].to_vec();
            Some(self.get_or_create_package(parent_path))
        };

        // Create new package
        let id = PackageId::new(self.next_package_id);
        self.next_package_id += 1;

        let mut package = PackageInfo::new(path.clone(), parent_id);

        // Register in parent
        if let Some(parent_id) = parent_id {
            if let Some(parent) = self.packages.get_mut(&parent_id) {
                if let Some(name) = path.last() {
                    parent.sub_packages.insert(*name, id);
                }
            }
        }

        self.packages.insert(id, package);
        self.package_paths.insert(path, id);

        id
    }

    /// Set the current package context
    pub fn set_current_package(&mut self, package_id: Option<PackageId>) {
        self.current_package = package_id;
    }

    /// Get the current package
    pub fn current_package(&self) -> Option<PackageId> {
        self.current_package
    }

    /// Register a symbol in a package
    pub fn register_symbol(
        &mut self,
        package_id: PackageId,
        name: InternedString,
        symbol_id: SymbolId,
    ) {
        if let Some(package) = self.packages.get_mut(&package_id) {
            package.symbols.insert(name, symbol_id);
        }
    }

    /// Look up a symbol by qualified path
    pub fn lookup_symbol(&self, path: &QualifiedPath) -> Option<SymbolId> {
        if let Some(&package_id) = self.package_paths.get(&path.package) {
            if let Some(package) = self.packages.get(&package_id) {
                return package.symbols.get(&path.name).copied();
            }
        }
        None
    }

    /// Debug: list all packages
    pub fn debug_list_packages(&self) -> Vec<Vec<InternedString>> {
        self.package_paths.keys().cloned().collect()
    }

    /// Find all symbols matching a name in the package hierarchy
    pub fn find_symbols_by_name(
        &self,
        name: InternedString,
        from_package: PackageId,
    ) -> Vec<(PackageId, SymbolId)> {
        let mut results = Vec::new();

        // Check current package and parents
        let mut current = Some(from_package);
        while let Some(package_id) = current {
            if let Some(package) = self.packages.get(&package_id) {
                if let Some(&symbol_id) = package.symbols.get(&name) {
                    results.push((package_id, symbol_id));
                }
                current = package.parent;
            } else {
                break;
            }
        }

        results
    }

    /// Get package info by ID
    pub fn get_package(&self, id: PackageId) -> Option<&PackageInfo> {
        self.packages.get(&id)
    }

    /// Get all sub-packages of a package
    pub fn get_sub_packages(&self, package_id: PackageId) -> Vec<(InternedString, PackageId)> {
        if let Some(package) = self.packages.get(&package_id) {
            package.sub_packages.iter().map(|(&k, &v)| (k, v)).collect()
        } else {
            Vec::new()
        }
    }

    /// Find the package containing a symbol
    pub fn find_package_by_symbol(&self, symbol_id: SymbolId) -> Option<PackageId> {
        for (&package_id, package) in &self.packages {
            if package.symbols.values().any(|&s| s == symbol_id) {
                return Some(package_id);
            }
        }
        None
    }

    /// Mark a package as internal
    pub fn set_package_internal(&mut self, package_id: PackageId, is_internal: bool) {
        if let Some(package) = self.packages.get_mut(&package_id) {
            package.is_internal = is_internal;
        }
    }

    /// Check if a package path exists
    pub fn has_package(&self, path: &[InternedString]) -> bool {
        self.package_paths.contains_key(path)
    }

    /// Get package ID by path
    pub fn get_package_by_path(&self, path: &[InternedString]) -> Option<PackageId> {
        self.package_paths.get(path).copied()
    }
}

/// Import resolver for managing import visibility and precedence
pub struct ImportResolver {
    /// Imports organized by scope
    imports_by_scope: HashMap<ScopeId, Vec<ImportEntry>>,

    /// Type aliases by scope and name
    aliases: HashMap<(ScopeId, InternedString), QualifiedPath>,
}

impl ImportResolver {
    /// Create a new import resolver
    pub fn new() -> Self {
        ImportResolver {
            imports_by_scope: HashMap::new(),
            aliases: HashMap::new(),
        }
    }

    /// Add an import to a scope
    pub fn add_import(&mut self, scope: ScopeId, import: ImportEntry) {
        // Register alias if present
        if let Some(alias) = import.alias {
            self.aliases
                .insert((scope, alias), import.package_path.clone());
        }

        self.imports_by_scope
            .entry(scope)
            .or_insert_with(Vec::new)
            .push(import);
    }

    /// Resolve a type name in the context of imports
    pub fn resolve_type(
        &self,
        name: InternedString,
        scope: ScopeId,
        ns: &NamespaceResolver,
    ) -> Vec<QualifiedPath> {
        let mut candidates = Vec::new();

        // Check aliases first (check current scope and root scope)
        if let Some(path) = self.aliases.get(&(scope, name)) {
            candidates.push(path.clone());
        }
        // Also check root scope for aliases
        if scope != ScopeId::first() {
            if let Some(path) = self.aliases.get(&(ScopeId::first(), name)) {
                candidates.push(path.clone());
            }
        }

        // Check current package (types in same package are automatically visible)
        if let Some(pkg_id) = ns.current_package() {
            if let Some(package) = ns.get_package(pkg_id) {
                let path = QualifiedPath::new(package.full_path.clone(), name);
                if ns.lookup_symbol(&path).is_some() {
                    candidates.push(path);
                }
            }
        }

        // Check explicit imports in current scope
        if let Some(imports) = self.imports_by_scope.get(&scope) {
            for import in imports {
                if !import.is_wildcard && import.package_path.name == name {
                    // Direct import match
                    candidates.push(import.package_path.clone());
                } else if import.is_wildcard && !import.exclusions.contains(&name) {
                    // Wildcard import - construct full path
                    let wildcard_path = import.package_path.package.clone();
                    candidates.push(QualifiedPath::new(wildcard_path, name));
                }
            }
        }

        // Also check root scope imports (imports are typically at module level)
        if scope != ScopeId::first() {
            if let Some(imports) = self.imports_by_scope.get(&ScopeId::first()) {
                for import in imports {
                    if !import.is_wildcard && import.package_path.name == name {
                        // Direct import match
                        if !candidates.iter().any(|c| c == &import.package_path) {
                            candidates.push(import.package_path.clone());
                        }
                    } else if import.is_wildcard && !import.exclusions.contains(&name) {
                        // Wildcard import - construct full path
                        let wildcard_path = import.package_path.package.clone();
                        let path = QualifiedPath::new(wildcard_path, name);
                        if !candidates.iter().any(|c| c == &path) {
                            candidates.push(path);
                        }
                    }
                }
            }
        }

        candidates
    }

    /// Get all imports for a scope
    pub fn get_imports(&self, scope: ScopeId) -> &[ImportEntry] {
        self.imports_by_scope
            .get(&scope)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tast::StringInterner;

    #[test]
    fn test_package_hierarchy() {
        let mut interner = StringInterner::new();
        let mut resolver = NamespaceResolver::new();

        let com = interner.intern("com");
        let example = interner.intern("example");
        let game = interner.intern("game");

        let package_id = resolver.get_or_create_package(vec![com, example, game]);

        let package = resolver.get_package(package_id).unwrap();
        assert_eq!(package.full_path.len(), 3);
        assert_eq!(package.qualified_path(&interner), "com.example.game");
    }

    #[test]
    fn test_symbol_registration() {
        let mut interner = StringInterner::new();
        let mut resolver = NamespaceResolver::new();

        let com = interner.intern("com");
        let example = interner.intern("example");
        let player = interner.intern("Player");

        let package_id = resolver.get_or_create_package(vec![com, example]);
        let symbol_id = SymbolId::from_raw(42);

        resolver.register_symbol(package_id, player, symbol_id);

        let path = QualifiedPath::new(vec![com, example], player);
        let found = resolver.lookup_symbol(&path);
        assert_eq!(found, Some(symbol_id));
    }
}
