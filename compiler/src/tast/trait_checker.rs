//! Trait checking system for derived traits
//!
//! This module provides the `TraitChecker` which queries whether types
//! implement specific traits (Send, Sync, Clone, etc.). This is used during
//! extern call lowering to validate thread safety and other constraints.

use crate::tast::{
    core::{Mutability, TypeKind},
    core_types::CoreTypeChecker,
    node::{DerivedTrait, TypedClass},
    StringInterner, SymbolId, SymbolTable, TypeId, TypeTable,
};
use std::cell::RefCell;

/// Trait checker for querying trait implementations
pub struct TraitChecker<'a> {
    type_table: &'a RefCell<TypeTable>,
    symbol_table: &'a SymbolTable,
    /// Map from SymbolId to TypedClass for quick lookup
    class_map: std::collections::BTreeMap<SymbolId, &'a TypedClass>,
    /// Core type checker for identifying stdlib concurrent types
    core_checker: CoreTypeChecker<'a>,
}

impl<'a> TraitChecker<'a> {
    /// Create a new trait checker
    pub fn new(
        type_table: &'a RefCell<TypeTable>,
        symbol_table: &'a SymbolTable,
        string_interner: &'a StringInterner,
        classes: &'a [TypedClass],
    ) -> Self {
        // Build a map from SymbolId to TypedClass
        let mut class_map = std::collections::BTreeMap::new();
        for class in classes {
            class_map.insert(class.symbol_id, class);
        }

        Self {
            type_table,
            symbol_table,
            class_map,
            core_checker: CoreTypeChecker::new(type_table, symbol_table, string_interner),
        }
    }

    /// Extend the class lookup map with another batch of classes (e.g. stdlib
    /// classes from `loaded_stdlib_typed_files`). Returns `self` for chaining.
    ///
    /// Without this, `requires_strict_move`, `is_copy`, `auto_derive_*` etc.
    /// only see classes from a single `TypedFile` — cross-file `@:move`-annotated
    /// stdlib types like `rayzor.ds.Tensor` / `rayzor.ds.QTensor` are silently
    /// invisible and the move semantics never fire at user-call sites.
    /// Existing entries take precedence; later additions don't shadow them.
    pub fn extend_classes(mut self, classes: &'a [TypedClass]) -> Self {
        for class in classes {
            self.class_map.entry(class.symbol_id).or_insert(class);
        }
        self
    }

    /// Check if a type implements the Send trait
    ///
    /// Send types can be transferred between threads safely.
    /// Required for Thread.spawn() captures and Channel<T> element types.
    pub fn is_send(&self, type_id: TypeId) -> bool {
        self.implements_trait(type_id, DerivedTrait::Send)
    }

    /// Check if a type implements the Sync trait
    ///
    /// Sync types can be safely shared between threads (via references).
    /// Required for Arc<T> element types.
    pub fn is_sync(&self, type_id: TypeId) -> bool {
        self.implements_trait(type_id, DerivedTrait::Sync)
    }

    /// Check if a type implements the Clone trait
    pub fn is_clone(&self, type_id: TypeId) -> bool {
        self.implements_trait(type_id, DerivedTrait::Clone)
    }

    /// Check if a type implements the Copy trait
    pub fn is_copy(&self, type_id: TypeId) -> bool {
        self.implements_trait(type_id, DerivedTrait::Copy)
    }

    /// Check if a type is annotated with `@:move` — meaning aliasing it after
    /// a move must be a hard compile error rather than a soft warning.
    ///
    /// Mirrors `is_copy`'s shape: resolve `TypeId → TypeKind::Class { symbol_id }`,
    /// look the class up, and read its `has_move_annotation()` flag. Any other
    /// type kind (primitives, arrays, references…) is never strictly-moved.
    pub fn requires_strict_move(&self, type_id: TypeId) -> bool {
        let type_kind = {
            let type_table = self.type_table.borrow();
            match type_table.get(type_id) {
                Some(info) => info.kind.clone(),
                None => return false,
            }
        };

        match &type_kind {
            TypeKind::Class { symbol_id, .. } => self
                .find_class(*symbol_id)
                .map(|c| c.has_move_annotation())
                .unwrap_or(false),
            _ => false,
        }
    }

    /// Check if a type is annotated with `@:shared` — meaning `.clone()` is
    /// an atomic-refcount increment (runtime-safe aliasing) and the compiler
    /// must NOT emit `MarkMoved` / `CheckLive` guards for bindings of this
    /// class. Mirrors `requires_strict_move`'s shape.
    pub fn requires_shared(&self, type_id: TypeId) -> bool {
        let type_kind = {
            let type_table = self.type_table.borrow();
            match type_table.get(type_id) {
                Some(info) => info.kind.clone(),
                None => return false,
            }
        };

        match &type_kind {
            TypeKind::Class { symbol_id, .. } => self
                .find_class(*symbol_id)
                .map(|c| c.has_shared_annotation())
                .unwrap_or(false),
            _ => false,
        }
    }

    /// Generic trait checking
    pub fn implements_trait(&self, type_id: TypeId, trait_: DerivedTrait) -> bool {
        // Extract the kind from the type table, then drop the borrow
        let type_kind = {
            let type_table = self.type_table.borrow();
            match type_table.get(type_id) {
                Some(info) => info.kind.clone(),
                None => return false,
            }
        }; // type_table borrow is dropped here

        match &type_kind {
            // Primitives are always Send + Sync
            TypeKind::Int | TypeKind::Float | TypeKind::Bool => true,

            // String is Send but NOT Sync (has interior mutability in our impl)
            TypeKind::String => matches!(trait_, DerivedTrait::Send | DerivedTrait::Clone),

            // Void is Send + Sync (it's empty)
            TypeKind::Void => true,

            // Check class for derived traits
            // Interfaces share the class path: `class_implements_trait` falls
            // through `find_class` (an interface isn't a class → None) to the
            // symbol's derive flags, which the interface lowering mirrors on.
            // So `@:derive([Send])` on an interface makes an interface-typed
            // value Send at a Thread.spawn capture site.
            TypeKind::Class { symbol_id, .. } | TypeKind::Interface { symbol_id, .. } => {
                // Stdlib concurrent types are inherently Send + Sync
                if matches!(trait_, DerivedTrait::Send | DerivedTrait::Sync) {
                    if self.core_checker.is_arc(type_id)
                        || self.core_checker.is_mutex(type_id)
                        || self.core_checker.is_channel(type_id)
                        || self.core_checker.is_thread(type_id)
                    {
                        return true;
                    }
                }
                self.class_implements_trait(*symbol_id, trait_)
            }

            // Function types: Send-ness is tracked on `FunctionEffects.is_send`.
            // Bare function refs / lambdas-with-Send-captures default to true;
            // closures that capture non-Send state get `is_send = false` at
            // FunctionLiteral TAST lowering time. Sync follows the same flag —
            // a Send-safe function is also safe to share (it has no mutable
            // state of its own).
            TypeKind::Function { effects, .. } => match trait_ {
                DerivedTrait::Send | DerivedTrait::Sync => effects.is_send,
                // Copy / Clone for function types are pre-existing trivia:
                // function values are bit-pattern Copy (they're code pointers
                // + a possibly-aliased env pointer). Keep as `true` to match
                // the prior implicit behaviour for non-Send traits.
                _ => true,
            },

            // Arrays: Send/Sync if element is Send/Sync
            TypeKind::Array { element_type } => self.implements_trait(*element_type, trait_),

            // References: &T is Send if T is Sync, &mut T is Send if T is Send
            TypeKind::Reference {
                target_type,
                mutability,
                ..
            } => {
                if *mutability == Mutability::Mutable {
                    // &mut T is Send if T is Send, Sync if T is Sync
                    self.implements_trait(*target_type, trait_)
                } else {
                    // &T is Send if T is Sync (shared reference)
                    match trait_ {
                        DerivedTrait::Send => self.is_sync(*target_type),
                        DerivedTrait::Sync => self.is_sync(*target_type),
                        _ => self.implements_trait(*target_type, trait_),
                    }
                }
            }

            // Abstract types: an explicit @:derive wins; otherwise delegate to
            // the underlying representation type; a @:coreType value-type with
            // no underlying (SIMD4f / Ptr / Usize / Atomic — register- or
            // pointer-sized opaque values) is Send/Sync/Copy/Clone like a
            // primitive. (Sharing such a value across threads is exactly what
            // rayzor.Atomic exists for.)
            TypeKind::Abstract {
                symbol_id,
                underlying,
                ..
            } => {
                if self.class_implements_trait(*symbol_id, trait_) {
                    true
                } else if let Some(u) = underlying {
                    self.implements_trait(*u, trait_)
                } else {
                    // Only Send/Sync for a bare @:coreType (a shared address /
                    // register value is safe to move/share across threads).
                    // Copy/Clone are deliberately NOT auto-granted here — that
                    // changes value-identity semantics (test_copy_clone_identity).
                    matches!(trait_, DerivedTrait::Send | DerivedTrait::Sync)
                }
            }

            // Dynamic type: unknown, assume NOT Send/Sync
            TypeKind::Dynamic => false,

            // Unknown types
            _ => false,
        }
    }

    /// Check if a class implements a trait
    fn class_implements_trait(&self, symbol_id: SymbolId, trait_: DerivedTrait) -> bool {
        // Find the class
        let class = match self.find_class(symbol_id) {
            Some(c) => c,
            None => {
                // Cross-file / extern class: its TypedClass isn't in this file's
                // `classes` list, but the concurrency derives were mirrored onto
                // the symbol at lowering. Read them from the shared symbol table
                // (e.g. a `@:derive([Send])` sys.net.SocketOutput used at a
                // Thread.spawn capture site in another module).
                if let Some(sym) = self.symbol_table.get_symbol(symbol_id) {
                    return match trait_ {
                        DerivedTrait::Send => sym.flags.derives_send(),
                        DerivedTrait::Sync => sym.flags.derives_sync(),
                        _ => false,
                    };
                }
                return false;
            }
        };

        // 1. Check if explicitly derived
        if class.derives(trait_) {
            return true;
        }

        // 2. Auto-derive rules (like Rust)
        match trait_ {
            DerivedTrait::Send => self.auto_derive_send(&class),
            DerivedTrait::Sync => self.auto_derive_sync(&class),
            DerivedTrait::Clone => self.auto_derive_clone(&class),
            DerivedTrait::Copy => self.auto_derive_copy(&class),
            _ => false,
        }
    }

    /// Check if all fields are Send (auto-derive)
    fn auto_derive_send(&self, class: &TypedClass) -> bool {
        for field in &class.fields {
            if !self.is_send(field.field_type) {
                return false;
            }
        }
        true
    }

    /// Check if all fields are Sync (auto-derive)
    fn auto_derive_sync(&self, class: &TypedClass) -> bool {
        for field in &class.fields {
            if !self.is_sync(field.field_type) {
                return false;
            }
        }
        true
    }

    /// Check if all fields are Clone (auto-derive)
    fn auto_derive_clone(&self, class: &TypedClass) -> bool {
        for field in &class.fields {
            if !self.is_clone(field.field_type) {
                return false;
            }
        }
        true
    }

    /// Check if all fields are Copy (auto-derive)
    fn auto_derive_copy(&self, class: &TypedClass) -> bool {
        for field in &class.fields {
            if !self.is_copy(field.field_type) {
                return false;
            }
        }
        true
    }

    /// Find a class by symbol ID
    fn find_class(&self, symbol_id: SymbolId) -> Option<&'a TypedClass> {
        self.class_map.get(&symbol_id).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_context() -> (RefCell<TypeTable>, SymbolTable, StringInterner) {
        let type_table = RefCell::new(TypeTable::new());
        let symbol_table = SymbolTable::new();
        let string_interner = StringInterner::new();
        (type_table, symbol_table, string_interner)
    }

    #[test]
    fn test_primitives_are_send_sync() {
        let (type_table, symbol_table, string_interner) = create_test_context();
        let classes: Vec<TypedClass> = vec![];
        let checker = TraitChecker::new(&type_table, &symbol_table, &string_interner, &classes);

        // Use common types from type table
        let int_type = type_table.borrow().int_type();
        let float_type = type_table.borrow().float_type();
        let bool_type = type_table.borrow().bool_type();

        // All primitives should be Send + Sync
        assert!(checker.is_send(int_type));
        assert!(checker.is_sync(int_type));

        assert!(checker.is_send(float_type));
        assert!(checker.is_sync(float_type));

        assert!(checker.is_send(bool_type));
        assert!(checker.is_sync(bool_type));
    }

    #[test]
    fn test_string_is_send_not_sync() {
        let (type_table, symbol_table, string_interner) = create_test_context();
        let classes: Vec<TypedClass> = vec![];
        let checker = TraitChecker::new(&type_table, &symbol_table, &string_interner, &classes);

        let string_type = type_table.borrow().string_type();

        // String is Send but NOT Sync (due to interior mutability concerns)
        assert!(checker.is_send(string_type));
        assert!(!checker.is_sync(string_type));
    }

    #[test]
    fn test_default_function_types_are_send_sync() {
        // Send-ness is tracked on `FunctionEffects.is_send`. The default
        // factory builds function types with `is_send = true` — matching
        // the common case of bare named function references (extern stdlib
        // functions, named class methods, intrinsics) which have no
        // captures and are bit-pattern Send. Closures that capture non-Send
        // state get `is_send = false` set explicitly at FunctionLiteral
        // TAST lowering time; see ast_lowering.rs.
        let (type_table, symbol_table, string_interner) = create_test_context();
        let classes: Vec<TypedClass> = vec![];
        let checker = TraitChecker::new(&type_table, &symbol_table, &string_interner, &classes);

        let void_type = type_table.borrow().void_type();
        let func_type = type_table
            .borrow_mut()
            .create_function_type(vec![], void_type);

        assert!(checker.is_send(func_type));
        assert!(checker.is_sync(func_type));
    }

    #[test]
    fn test_non_send_closure_is_rejected() {
        // Explicit `is_send = false` via `create_function_type_with_effects`
        // is honoured — represents a closure that captures non-Send state.
        use crate::tast::core::FunctionEffects;
        let (type_table, symbol_table, string_interner) = create_test_context();
        let classes: Vec<TypedClass> = vec![];
        let checker = TraitChecker::new(&type_table, &symbol_table, &string_interner, &classes);

        let void_type = type_table.borrow().void_type();
        let effects = FunctionEffects {
            is_send: false,
            ..Default::default()
        };
        let closure_type =
            type_table
                .borrow_mut()
                .create_function_type_with_effects(vec![], void_type, effects);

        assert!(!checker.is_send(closure_type));
        assert!(!checker.is_sync(closure_type));
    }

    // TODO: Add comprehensive tests with actual TypedClass instances
    // The tests below would require properly constructing TypedClass with all fields,
    // which is complex. These will be added after the class structure is finalized.
    //
    // Test scenarios to add:
    // 1. Class with explicit @:derive([Send, Sync])
    // 2. Class with auto-derived Send/Sync (all fields are Send/Sync)
    // 3. Class with String field (Send but NOT Sync)
    // 4. Nested classes (class with field of another class type)
}
