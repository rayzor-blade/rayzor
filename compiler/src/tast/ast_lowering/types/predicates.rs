//! Questions asked about a type: copyable, hashable, ordered.

use super::*;
use crate::tast::node::HasSourceLocation;
use crate::tast::{core::*, node::MemoryEffects, node::*, type_resolution, *};
use parser::{
    AbstractDecl, BinaryOp, BlockElement, ClassDecl, ClassField, ClassFieldKind, EnumConstructor,
    EnumDecl, Expr, ExprKind, Function, FunctionParam, HaxeFile, Import, InterfaceDecl, Metadata,
    Modifier, ModuleField, Package, Type, TypeDeclaration, TypeParam, TypedefDecl, UnaryOp, Using,
};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fmt;
use std::rc::Rc;
use tracing::warn;

impl<'a> AstLowering<'a> {
    /// Check if a type implements Clone
    pub(crate) fn is_type_clone(&self, type_id: TypeId) -> bool {
        let type_table = self.context.type_table.borrow();

        if let Some(type_info) = type_table.get(type_id) {
            match &type_info.kind {
                // Primitive types are implicitly Copy (and thus Clone)
                crate::tast::core::TypeKind::Int
                | crate::tast::core::TypeKind::Float
                | crate::tast::core::TypeKind::Bool
                | crate::tast::core::TypeKind::Void => true,

                // String is Clone but not Copy
                crate::tast::core::TypeKind::String => true,

                // Class types: check if they derive Clone
                crate::tast::core::TypeKind::Class { symbol_id, .. } => {
                    // TODO: Look up class and check derived_traits
                    // For now, assume classes are Clone (conservative)
                    true
                }

                // Arrays and Maps are Clone if their element types are Clone
                crate::tast::core::TypeKind::Array { element_type } => {
                    self.is_type_clone(*element_type)
                }

                // Other types default to not Clone
                _ => false,
            }
        } else {
            false
        }
    }

    /// Check if a type implements Copy
    pub(crate) fn is_type_copy(&self, type_id: TypeId) -> bool {
        let type_table = self.context.type_table.borrow();

        if let Some(type_info) = type_table.get(type_id) {
            match &type_info.kind {
                // Only primitive types are Copy
                crate::tast::core::TypeKind::Int
                | crate::tast::core::TypeKind::Float
                | crate::tast::core::TypeKind::Bool => true,

                // Class types: check if they derive Copy
                crate::tast::core::TypeKind::Class { symbol_id, .. } => {
                    // TODO: Look up class and check if it derives Copy
                    // For now, assume classes are NOT Copy (safe default)
                    false
                }

                // String, Arrays, and other heap types are NOT Copy
                _ => false,
            }
        } else {
            false
        }
    }

    /// Check if a type supports equality comparison (for @:derive(PartialEq))
    pub(crate) fn is_type_equatable(&self, type_id: TypeId) -> bool {
        let type_table = self.context.type_table.borrow();
        if let Some(type_info) = type_table.get(type_id) {
            match &type_info.kind {
                crate::tast::core::TypeKind::Int
                | crate::tast::core::TypeKind::Float
                | crate::tast::core::TypeKind::Bool
                | crate::tast::core::TypeKind::Void
                | crate::tast::core::TypeKind::String => true,

                crate::tast::core::TypeKind::Class { .. } => {
                    // Classes are equatable if they derive PartialEq (checked at codegen)
                    // or are compared by pointer (fallback). Accept for now.
                    true
                }

                crate::tast::core::TypeKind::Enum { .. } => true,
                crate::tast::core::TypeKind::Array { element_type } => {
                    self.is_type_equatable(*element_type)
                }

                // Function types and Dynamic are not equatable
                crate::tast::core::TypeKind::Function { .. } => false,
                crate::tast::core::TypeKind::Dynamic => false,

                _ => true, // Conservative: allow other types
            }
        } else {
            false
        }
    }

    /// Check if a type supports ordering (for @:derive(PartialOrd))
    pub(crate) fn is_type_orderable(&self, type_id: TypeId) -> bool {
        let type_table = self.context.type_table.borrow();
        if let Some(type_info) = type_table.get(type_id) {
            match &type_info.kind {
                crate::tast::core::TypeKind::Int
                | crate::tast::core::TypeKind::Float
                | crate::tast::core::TypeKind::Bool
                | crate::tast::core::TypeKind::String => true,

                crate::tast::core::TypeKind::Class { .. } => true,
                crate::tast::core::TypeKind::Enum { .. } => true,

                crate::tast::core::TypeKind::Function { .. } => false,
                crate::tast::core::TypeKind::Dynamic => false,

                _ => true,
            }
        } else {
            false
        }
    }

    /// Check if a type is hashable (for @:derive(Hash))
    pub(crate) fn is_type_hashable(&self, type_id: TypeId) -> bool {
        let type_table = self.context.type_table.borrow();
        if let Some(type_info) = type_table.get(type_id) {
            match &type_info.kind {
                crate::tast::core::TypeKind::Int
                | crate::tast::core::TypeKind::Bool
                | crate::tast::core::TypeKind::String => true,

                // Float is technically hashable but fragile (NaN != NaN)
                crate::tast::core::TypeKind::Float => true,

                crate::tast::core::TypeKind::Class { .. } => true,
                crate::tast::core::TypeKind::Enum { .. } => true,

                crate::tast::core::TypeKind::Function { .. } => false,
                crate::tast::core::TypeKind::Dynamic => false,

                _ => false,
            }
        } else {
            false
        }
    }

    /// Detect "auto-deref wrapper" classes — types where field access
    /// transparently forwards to their inner value via a `get()` method.
    ///
    /// A class qualifies for auto-deref when it carries the
    /// `@:autoDeref` metadata (parsed into [`SymbolFlags::AUTO_DEREF`]).
    /// The built-in `rayzor.concurrent.Arc` / `MutexGuard` extern
    /// classes are annotated with `@:autoDeref` in their `.hx`
    /// definitions; the previously-hardcoded qualified-name list is
    /// retained below as a fallback only for the brief window between
    /// cache load and metadata propagation (some pre-blade-cache
    /// extern symbols may temporarily lack the flag).
    pub(crate) fn is_auto_deref_wrapper_class(&self, class_sym: SymbolId) -> bool {
        let sym = match self.context.symbol_table.get_symbol(class_sym) {
            Some(s) => s,
            None => return false,
        };
        if sym.flags.is_auto_deref() {
            return true;
        }
        // Backward-compat fallback: stdlib Arc / MutexGuard cached
        // before the @:autoDeref annotation landed. Once a clean
        // blade cache regenerates with the new metadata flag, this
        // branch becomes dead and can be removed.
        let qn = sym
            .qualified_name
            .and_then(|q| self.context.string_interner.get(q))
            .unwrap_or("");
        matches!(qn, "rayzor.concurrent.Arc" | "rayzor.concurrent.MutexGuard")
    }

    /// Check if a type is an integer iterator (from range expressions)
    pub(crate) fn is_int_iterator_type(&self, type_id: TypeId) -> bool {
        // Check if this is an IntIterator type
        if let Some(type_info) = self.context.type_table.borrow().get(type_id) {
            match &type_info.kind {
                TypeKind::Class { symbol_id, .. } => {
                    // Check if the class is IntIterator
                    if let Some(class_symbol) = self.context.symbol_table.get_symbol(*symbol_id) {
                        let int_iterator_name = self.context.string_interner.intern("IntIterator");
                        return class_symbol.name == int_iterator_name;
                    }
                }
                TypeKind::Dynamic => {
                    // Dynamic type could be an iterator at runtime
                    return true;
                }
                _ => {}
            }
        }
        false
    }

    /// Check if a type is SIMD4f (by native_name or symbol name).
    fn is_simd4f_type(&self, ty: crate::tast::TypeId) -> bool {
        use crate::tast::core::TypeKind;
        let type_table = self.context.type_table.borrow();
        let sym_id = type_table.get(ty).and_then(|ti| match &ti.kind {
            TypeKind::Abstract { symbol_id, .. } | TypeKind::Class { symbol_id, .. } => {
                Some(*symbol_id)
            }
            _ => None,
        });
        if let Some(sid) = sym_id {
            self.context
                .symbol_table
                .get_symbol(sid)
                .map(|s| {
                    let by_native = s
                        .native_name
                        .and_then(|nn| self.context.string_interner.get(nn))
                        .map(|n| n == "rayzor::SIMD4f")
                        .unwrap_or(false);
                    let by_name = self
                        .context
                        .string_interner
                        .get(s.name)
                        .map(|n| n == "SIMD4f")
                        .unwrap_or(false);
                    by_native || by_name
                })
                .unwrap_or(false)
        } else {
            false
        }
    }

    /// Check if a type is an abstract type (any abstract, not just SIMD4f).
    pub(crate) fn is_abstract_type(&self, ty: crate::tast::TypeId) -> bool {
        use crate::tast::core::TypeKind;
        let type_table = self.context.type_table.borrow();
        matches!(
            type_table.get(ty).map(|ti| &ti.kind),
            Some(TypeKind::Abstract { .. })
        )
    }
}
