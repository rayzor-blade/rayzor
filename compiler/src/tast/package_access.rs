//! Package-level Access Control System
//!
//! This module provides comprehensive package visibility enforcement for the type system.
//! It handles cross-package access validation, internal visibility checks, and import
//! permission validation to ensure proper encapsulation between compilation units.

use super::{
    AccessLevel, InternedString, NamespaceResolver, PackageId, SourceLocation, StringInterner,
    SymbolId, SymbolTable, TypeCheckError, TypeErrorKind, TypeId, Visibility,
};
use std::collections::{BTreeMap, BTreeSet};

/// Package access context for tracking current compilation unit
#[derive(Debug, Clone)]
pub struct PackageAccessContext {
    /// Current package being compiled
    pub current_package: Option<PackageId>,

    /// Current file path for multi-file compilation
    pub current_file: Option<InternedString>,

    /// Symbols visible from current context
    pub visible_symbols: BTreeSet<SymbolId>,

    /// Import permissions for current context
    pub import_permissions: BTreeMap<PackageId, AccessPermission>,
}

impl PackageAccessContext {
    /// Create a new package access context
    pub fn new() -> Self {
        PackageAccessContext {
            current_package: None,
            current_file: None,
            visible_symbols: BTreeSet::new(),
            import_permissions: BTreeMap::new(),
        }
    }

    /// Set the current package context
    pub fn set_package(&mut self, package: Option<PackageId>) {
        self.current_package = package;
        self.visible_symbols.clear();
        self.import_permissions.clear();
    }

    /// Set the current file context
    pub fn set_file(&mut self, file_path: Option<InternedString>) {
        self.current_file = file_path;
    }

    /// Add import permission for a package
    pub fn add_import_permission(&mut self, package: PackageId, permission: AccessPermission) {
        self.import_permissions.insert(package, permission);
    }

    /// Check if a symbol is visible in current context
    pub fn is_symbol_visible(&self, symbol: SymbolId) -> bool {
        self.visible_symbols.contains(&symbol)
    }

    /// Add visible symbol to context
    pub fn add_visible_symbol(&mut self, symbol: SymbolId) {
        self.visible_symbols.insert(symbol);
    }
}

/// Access permission levels for package imports
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessPermission {
    /// Full access to all public members
    Public,

    /// Access to public and internal members (same package)
    Internal,

    /// No access
    None,
}

/// Package access validator for enforcing visibility rules
pub struct PackageAccessValidator<'a> {
    /// Symbol table reference
    symbol_table: &'a SymbolTable,

    /// Namespace resolver reference
    namespace_resolver: &'a NamespaceResolver,

    /// String interner reference
    string_interner: &'a StringInterner,

    /// Current access context
    context: PackageAccessContext,

    /// Cache of symbol-to-package mappings
    symbol_packages: BTreeMap<SymbolId, PackageId>,

    /// Cache of package relationships
    package_cache: PackageRelationCache,
}

impl<'a> PackageAccessValidator<'a> {
    /// Create a new package access validator
    pub fn new(
        symbol_table: &'a SymbolTable,
        namespace_resolver: &'a NamespaceResolver,
        string_interner: &'a StringInterner,
    ) -> Self {
        PackageAccessValidator {
            symbol_table,
            namespace_resolver,
            string_interner,
            context: PackageAccessContext::new(),
            symbol_packages: BTreeMap::new(),
            package_cache: PackageRelationCache::new(),
        }
    }

    /// Set the current package context
    pub fn set_context(&mut self, package: Option<PackageId>, file: Option<InternedString>) {
        self.context.set_package(package);
        self.context.set_file(file);

        // Populate visible symbols for the package
        if let Some(pkg_id) = package {
            self.populate_visible_symbols(pkg_id);
        }
    }

    /// Populate visible symbols for a package
    fn populate_visible_symbols(&mut self, package_id: PackageId) {
        // Add all symbols from current package
        if let Some(package) = self.namespace_resolver.get_package(package_id) {
            for &symbol_id in package.symbols.values() {
                self.context.add_visible_symbol(symbol_id);
                self.symbol_packages.insert(symbol_id, package_id);
            }
        }

        // Add symbols from parent packages
        let mut current_package = Some(package_id);
        while let Some(pkg_id) = current_package {
            if let Some(package) = self.namespace_resolver.get_package(pkg_id) {
                current_package = package.parent;

                // Add public symbols from parent
                for &symbol_id in package.symbols.values() {
                    if let Some(symbol) = self.symbol_table.get_symbol(symbol_id) {
                        if symbol.visibility == Visibility::Public {
                            self.context.add_visible_symbol(symbol_id);
                        }
                    }
                }
            } else {
                break;
            }
        }
    }

    /// Validate access to a symbol from current context
    pub fn validate_symbol_access(
        &mut self,
        target_symbol: SymbolId,
        location: SourceLocation,
    ) -> Result<(), TypeCheckError> {
        // Get symbol information
        let symbol_info = self
            .symbol_table
            .get_symbol(target_symbol)
            .ok_or_else(|| self.create_unknown_symbol_error(target_symbol, location))?;

        // Check visibility
        match symbol_info.visibility {
            Visibility::Public => {
                // Public symbols are always accessible
                Ok(())
            }
            Visibility::Internal => {
                // Internal symbols require same-package access
                self.validate_internal_access(target_symbol, location)
            }
            Visibility::Private | Visibility::Protected => {
                // These are handled by class-level access checking
                Ok(())
            }
        }
    }

    /// Validate internal access to a symbol
    fn validate_internal_access(
        &mut self,
        target_symbol: SymbolId,
        location: SourceLocation,
    ) -> Result<(), TypeCheckError> {
        // Get target symbol's package
        let target_package = self.get_symbol_package(target_symbol)?;

        // Check if we're in the same package
        if let Some(current_package) = self.context.current_package {
            if self.are_packages_same(current_package, target_package) {
                Ok(())
            } else {
                Err(self.create_package_access_error(
                    target_symbol,
                    target_package,
                    current_package,
                    location,
                ))
            }
        } else {
            // No current package (default package)
            if target_package == PackageId::root() {
                Ok(())
            } else {
                Err(self.create_package_access_error(
                    target_symbol,
                    target_package,
                    PackageId::root(),
                    location,
                ))
            }
        }
    }

    /// Get the package a symbol belongs to
    fn get_symbol_package(&mut self, symbol_id: SymbolId) -> Result<PackageId, TypeCheckError> {
        // Check cache first
        if let Some(&package_id) = self.symbol_packages.get(&symbol_id) {
            return Ok(package_id);
        }

        // Find package by searching namespace
        if let Some(package_id) = self.namespace_resolver.find_package_by_symbol(symbol_id) {
            self.symbol_packages.insert(symbol_id, package_id);
            Ok(package_id)
        } else {
            // Default to root package if not found
            Ok(PackageId::root())
        }
    }

    /// Check if two packages are the same (considering sub-packages)
    fn are_packages_same(&mut self, pkg1: PackageId, pkg2: PackageId) -> bool {
        if pkg1 == pkg2 {
            return true;
        }

        // Check cache
        let key = if pkg1.as_raw() < pkg2.as_raw() {
            (pkg1, pkg2)
        } else {
            (pkg2, pkg1)
        };

        if let Some(&result) = self.package_cache.same_package.get(&key) {
            return result;
        }

        // Check if one is a sub-package of the other
        let result = self.is_sub_package(pkg1, pkg2) || self.is_sub_package(pkg2, pkg1);

        self.package_cache.same_package.insert(key, result);
        result
    }

    /// Check if pkg1 is a sub-package of pkg2
    fn is_sub_package(&self, pkg1: PackageId, pkg2: PackageId) -> bool {
        let mut current = Some(pkg1);
        while let Some(pkg_id) = current {
            if pkg_id == pkg2 {
                return true;
            }
            current = self
                .namespace_resolver
                .get_package(pkg_id)
                .and_then(|p| p.parent);
        }
        false
    }

    /// Validate import statement
    pub fn validate_import(
        &mut self,
        import_path: &[InternedString],
        location: SourceLocation,
    ) -> Result<(), TypeCheckError> {
        // Check if the imported package exists
        if let Some(package_id) = self.namespace_resolver.get_package_by_path(import_path) {
            // Check if we have permission to import from this package
            if let Some(current_package) = self.context.current_package {
                let permission = self.get_import_permission(current_package, package_id);

                if permission == AccessPermission::None {
                    return Err(self.create_import_permission_error(
                        package_id,
                        current_package,
                        location,
                    ));
                }

                // Add import permission to context
                self.context.add_import_permission(package_id, permission);
            }

            Ok(())
        } else {
            Err(self.create_unknown_package_error(import_path, location))
        }
    }

    /// Get import permission between packages
    fn get_import_permission(
        &self,
        from_package: PackageId,
        to_package: PackageId,
    ) -> AccessPermission {
        if from_package == to_package {
            AccessPermission::Internal
        } else if self.is_sub_package(from_package, to_package)
            || self.is_sub_package(to_package, from_package)
        {
            AccessPermission::Internal
        } else {
            AccessPermission::Public
        }
    }

    /// Create error for unknown symbol
    fn create_unknown_symbol_error(
        &self,
        symbol: SymbolId,
        location: SourceLocation,
    ) -> TypeCheckError {
        TypeCheckError {
            kind: TypeErrorKind::UnknownSymbol {
                name: format!("Symbol#{}", symbol.as_raw()),
            },
            location,
            context: "Unknown symbol reference".to_string(),
            suggestion: None,
        }
    }

    /// Create error for package access violation
    fn create_package_access_error(
        &self,
        symbol: SymbolId,
        target_package: PackageId,
        current_package: PackageId,
        location: SourceLocation,
    ) -> TypeCheckError {
        let symbol_name = self
            .symbol_table
            .get_symbol(symbol)
            .map(|s| self.string_interner.get(s.name).unwrap_or("<unknown>"))
            .unwrap_or("<unknown>");

        let target_pkg_name = self
            .namespace_resolver
            .get_package(target_package)
            .map(|p| p.qualified_path(self.string_interner))
            .unwrap_or_else(|| "<unknown>".to_string());

        let current_pkg_name = self
            .namespace_resolver
            .get_package(current_package)
            .map(|p| p.qualified_path(self.string_interner))
            .unwrap_or_else(|| "<default>".to_string());

        TypeCheckError {
            kind: TypeErrorKind::AccessViolation {
                symbol_id: symbol,
                required_access: AccessLevel::Internal,
            },
            location,
            context: format!("Internal symbol '{}' in package '{}' cannot be accessed from package '{}'",
                symbol_name, target_pkg_name, current_pkg_name),
            suggestion: Some("Make the symbol public to access from different packages, or move the accessing code to the same package".to_string()),
        }
    }

    /// Create error for import permission violation
    fn create_import_permission_error(
        &self,
        target_package: PackageId,
        current_package: PackageId,
        location: SourceLocation,
    ) -> TypeCheckError {
        let target_pkg_name = self
            .namespace_resolver
            .get_package(target_package)
            .map(|p| p.qualified_path(self.string_interner))
            .unwrap_or_else(|| "<unknown>".to_string());

        let current_pkg_name = self
            .namespace_resolver
            .get_package(current_package)
            .map(|p| p.qualified_path(self.string_interner))
            .unwrap_or_else(|| "<default>".to_string());

        TypeCheckError {
            kind: TypeErrorKind::ImportError {
                message: format!(
                    "Cannot import package '{}' from '{}'",
                    target_pkg_name, current_pkg_name
                ),
            },
            location,
            context: format!(
                "Import restriction: package '{}' cannot be imported from '{}'",
                target_pkg_name, current_pkg_name
            ),
            suggestion: Some(
                "Check package visibility settings and import permissions".to_string(),
            ),
        }
    }

    /// Create error for unknown package
    fn create_unknown_package_error(
        &self,
        import_path: &[InternedString],
        location: SourceLocation,
    ) -> TypeCheckError {
        let path_str = import_path
            .iter()
            .map(|&s| self.string_interner.get(s).unwrap_or("<unknown>"))
            .collect::<Vec<_>>()
            .join(".");

        TypeCheckError {
            kind: TypeErrorKind::ImportError {
                message: format!("Unknown package '{}'", path_str),
            },
            location,
            context: format!("Package '{}' not found", path_str),
            suggestion: Some(
                "Check that the package path is correct and the package exists".to_string(),
            ),
        }
    }

    /// Get current package context
    pub fn current_context(&self) -> &PackageAccessContext {
        &self.context
    }

    /// Check if a type is accessible from current context
    pub fn is_type_accessible(&mut self, type_id: TypeId) -> bool {
        // This would integrate with the type table to check if all symbols
        // referenced by the type are accessible
        true // Placeholder
    }
}

/// Cache for package relationship queries
#[derive(Debug)]
struct PackageRelationCache {
    /// Cache for same-package checks
    same_package: BTreeMap<(PackageId, PackageId), bool>,
}

impl PackageRelationCache {
    fn new() -> Self {
        PackageRelationCache {
            same_package: BTreeMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tast::{NamespaceResolver, StringInterner, SymbolTable};

    #[test]
    fn test_package_access_validation() {
        let mut interner = StringInterner::new();
        let mut symbol_table = SymbolTable::new();
        let mut namespace = NamespaceResolver::new();

        // Create packages
        let com = interner.intern("com");
        let example = interner.intern("example");
        let pkg1 = namespace.get_or_create_package(vec![com, example]);

        let utils = interner.intern("utils");
        let pkg2 = namespace.get_or_create_package(vec![utils]);

        // Create validator
        let mut validator = PackageAccessValidator::new(&symbol_table, &namespace, &interner);

        // Set context to pkg1
        validator.set_context(Some(pkg1), None);

        // Test same-package access
        assert!(validator.are_packages_same(pkg1, pkg1));
        assert!(!validator.are_packages_same(pkg1, pkg2));
    }

    #[test]
    fn test_sub_package_detection() {
        let mut interner = StringInterner::new();
        let symbol_table = SymbolTable::new();
        let mut namespace = NamespaceResolver::new();

        let com = interner.intern("com");
        let example = interner.intern("example");
        let game = interner.intern("game");

        let parent = namespace.get_or_create_package(vec![com, example]);
        let child = namespace.get_or_create_package(vec![com, example, game]);

        let validator = PackageAccessValidator::new(&symbol_table, &namespace, &interner);

        assert!(validator.is_sub_package(child, parent));
        assert!(!validator.is_sub_package(parent, child));
    }
}
