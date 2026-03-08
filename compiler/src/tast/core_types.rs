//! Core type identification for Rayzor stdlib
//!
//! This module provides utilities for identifying Rayzor standard library types
//! by their fully qualified paths. This is essential for applying special
//! validation rules (e.g., Send/Sync constraints on Thread::spawn).

use crate::tast::core::TypeKind;
use crate::tast::{StringInterner, SymbolId, SymbolTable, TypeId, TypeTable};
use std::cell::RefCell;
use std::rc::Rc;

/// Fully qualified paths for Rayzor core types
pub struct CoreTypePaths {
    pub thread: &'static str,
    pub channel: &'static str,
    pub mutex: &'static str,
    pub arc: &'static str,
}

impl CoreTypePaths {
    pub fn standard() -> Self {
        Self {
            thread: "rayzor.concurrent.Thread",
            channel: "rayzor.concurrent.Channel",
            mutex: "rayzor.concurrent.Mutex",
            arc: "rayzor.concurrent.Arc",
        }
    }
}

/// Core type identifier for Rayzor stdlib types
pub struct CoreTypeChecker<'a> {
    type_table: &'a Rc<RefCell<TypeTable>>,
    symbol_table: &'a SymbolTable,
    string_interner: &'a StringInterner,
    paths: CoreTypePaths,
}

impl<'a> CoreTypeChecker<'a> {
    /// Create a new core type checker
    pub fn new(
        type_table: &'a Rc<RefCell<TypeTable>>,
        symbol_table: &'a SymbolTable,
        string_interner: &'a StringInterner,
    ) -> Self {
        Self {
            type_table,
            symbol_table,
            string_interner,
            paths: CoreTypePaths::standard(),
        }
    }

    pub fn is_thread(&self, type_id: TypeId) -> bool {
        self.is_core_type(type_id, self.paths.thread)
    }

    pub fn is_channel(&self, type_id: TypeId) -> bool {
        self.is_core_type(type_id, self.paths.channel)
    }

    pub fn is_arc(&self, type_id: TypeId) -> bool {
        self.is_core_type(type_id, self.paths.arc)
    }

    pub fn is_mutex(&self, type_id: TypeId) -> bool {
        self.is_core_type(type_id, self.paths.mutex)
    }

    /// Get the type argument from a generic type (e.g., T from Thread<T>)
    ///
    /// Returns None if the type is not generic or has no type arguments.
    pub fn get_type_argument(&self, type_id: TypeId) -> Option<TypeId> {
        let type_table = self.type_table.borrow();
        let type_info = type_table.get(type_id)?;

        match &type_info.kind {
            TypeKind::Class { type_args, .. }
            | TypeKind::Interface { type_args, .. }
            | TypeKind::Enum { type_args, .. } => type_args.first().copied(),
            TypeKind::GenericInstance { type_args, .. } => type_args.first().copied(),
            _ => None,
        }
    }

    /// Check if a type matches a fully qualified path
    fn is_core_type(&self, type_id: TypeId, expected_path: &str) -> bool {
        let type_table = self.type_table.borrow();
        let type_info = match type_table.get(type_id) {
            Some(t) => t,
            None => return false,
        };

        // Get the symbol ID from the type
        let symbol_id = match &type_info.kind {
            TypeKind::Class { symbol_id, .. }
            | TypeKind::Interface { symbol_id, .. }
            | TypeKind::Enum { symbol_id, .. }
            | TypeKind::Abstract { symbol_id, .. } => *symbol_id,
            TypeKind::GenericInstance { base_type, .. } => {
                // For generic instances, check the base type
                return self.is_core_type(*base_type, expected_path);
            }
            _ => return false,
        };

        // Get the fully qualified name from the symbol
        self.check_symbol_path(symbol_id, expected_path)
    }

    /// Check if a symbol's fully qualified path matches the expected path
    fn check_symbol_path(&self, symbol_id: SymbolId, expected_path: &str) -> bool {
        let symbol = match self.symbol_table.get_symbol(symbol_id) {
            Some(s) => s,
            None => return false,
        };

        // Try qualified_name first (most precise)
        if let Some(qn) = symbol.qualified_name {
            let fqn_str = self
                .string_interner
                .get(qn)
                .map(|s| s.to_string())
                .or_else(|| {
                    let type_table = self.type_table.borrow();
                    type_table.get_string(qn).map(|s| s.to_string())
                });

            if let Some(fqn) = fqn_str {
                let normalized_fqn = fqn.replace("::", ".");
                let normalized_expected = expected_path.replace("::", ".");
                return normalized_fqn == normalized_expected;
            }
        }

        // Fallback: match by bare name against last component of expected path
        let bare_name = self
            .string_interner
            .get(symbol.name)
            .map(|s| s.to_string())
            .or_else(|| {
                let type_table = self.type_table.borrow();
                type_table.get_string(symbol.name).map(|s| s.to_string())
            });

        if let Some(name) = bare_name {
            let expected_short = expected_path.rsplit('.').next().unwrap_or(expected_path);
            return name == expected_short;
        }

        false
    }

    /// Validate Thread::spawn - all captured variables must be Send
    ///
    /// Returns the closure type ID if this is a Thread::spawn call
    pub fn get_thread_spawn_closure(
        &self,
        call_expr: &crate::tast::node::TypedExpression,
    ) -> Option<TypeId> {
        use crate::tast::node::TypedExpressionKind;

        // Check if this is a static method call
        if let TypedExpressionKind::StaticMethodCall {
            class_symbol,
            method_symbol,
            arguments,
            ..
        } = &call_expr.kind
        {
            // Check if the class is Thread
            if !self.check_symbol_path(*class_symbol, self.paths.thread) {
                return None;
            }

            // Check if the method is "spawn"
            let method_sym = self.symbol_table.get_symbol(*method_symbol)?;
            let method_name_str = self
                .string_interner
                .get(method_sym.name)
                .map(|s| s.to_string())
                .or_else(|| {
                    let tt = self.type_table.borrow();
                    tt.get_string(method_sym.name).map(|s| s.to_string())
                })?;
            if method_name_str != "spawn" {
                return None;
            }

            // Get the first argument (the closure)
            let closure_arg = arguments.first()?;
            Some(closure_arg.expr_type)
        } else {
            None
        }
    }

    /// Validate Channel::new - T must be Send
    ///
    /// Returns the channel element type if this is a Channel::new call
    pub fn get_channel_element_type(&self, type_id: TypeId) -> Option<TypeId> {
        if self.is_channel(type_id) {
            self.get_type_argument(type_id)
        } else {
            None
        }
    }

    /// Validate Arc::new - T must be Send + Sync
    ///
    /// Returns the Arc element type if this is an Arc type
    pub fn get_arc_element_type(&self, type_id: TypeId) -> Option<TypeId> {
        if self.is_arc(type_id) {
            self.get_type_argument(type_id)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_core_paths() {
        let paths = CoreTypePaths::standard();
        assert_eq!(paths.thread, "rayzor.concurrent.Thread");
        assert_eq!(paths.channel, "rayzor.concurrent.Channel");
        assert_eq!(paths.arc, "rayzor.concurrent.Arc");
    }

    // TODO: Add integration tests with actual TypeTable and SymbolTable
}
