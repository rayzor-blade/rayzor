//! Name-keyed lookups into the symbol, type and stdlib tables.
//! FQN-first: a bare name is only ever a fallback.

use super::*;
use crate::ir::drop_analysis::{DropBehavior, DropPointAnalyzer, DropPoints};
use crate::ir::hir::*;
use crate::ir::{
    BinaryOp, CallingConvention, CompareOp, EnvironmentLayout, FunctionKind,
    FunctionSignatureBuilder, IrBasicBlock, IrBlockId, IrBuilder, IrEnumVariant, IrField,
    IrFunction, IrFunctionId, IrFunctionSignature, IrGlobal, IrGlobalId, IrId, IrInstruction,
    IrLocal, IrModule, IrParameter, IrPhiNode, IrSourceLocation, IrTerminator, IrType, IrTypeDef,
    IrTypeDefId, IrTypeDefinition, IrValue, Linkage, UnaryOp,
};
use crate::stdlib::{IrTypeDescriptor, MethodSignature, StdlibMapping};
use crate::tast::symbols::SymbolFlags;
use crate::tast::{
    InternedString, SourceLocation, StringInterner, SymbolId, SymbolTable, TypeId, TypeKind,
    TypeTable,
};
use log::{debug, trace, warn};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

impl<'a> HirToMirContext<'a> {
    /// Find the class TypeId whose symbol carries `hint` as its name.
    /// Qualified match first; a bare name resolves ONLY when exactly one class
    /// answers to it. Several same-named classes in different packages are
    /// ambiguous, and picking the numerically-first entry would silently select
    /// a foreign class's layout for field access.
    pub(crate) fn find_class_type_by_name(&self, hint: &str) -> Option<TypeId> {
        let bare = {
            let h = hint.rsplit(':').next().unwrap_or(hint);
            h.rsplit('.').next().unwrap_or(h)
        };
        let mut bare_hits: Vec<TypeId> = Vec::new();
        for (ty, sym) in &self.class_type_to_symbol {
            let Some(symbol) = self.symbol_table.get_symbol(*sym) else {
                continue;
            };
            if let Some(qname) = symbol
                .qualified_name
                .and_then(|n| self.string_interner.get(n))
            {
                if qname == hint {
                    return Some(*ty);
                }
            }
            if self
                .string_interner
                .get(symbol.name)
                .is_some_and(|n| n == hint || n == bare)
            {
                bare_hits.push(*ty);
            }
        }
        bare_hits.dedup();
        match bare_hits.as_slice() {
            [only] => Some(*only),
            _ => None,
        }
    }

    /// Find the class name that owns a given method symbol by scanning HIR type declarations.
    /// Returns the normalized class name (e.g., "rayzor_gpu_GPUCompute") if found.
    /// Extract the normalized class name from an HirExpr's type by looking up
    /// the receiver in HIR type declarations. Handles extern classes whose TypeId
    /// may be invalid by checking all HIR types for a class with the method symbol.
    pub(crate) fn find_receiver_class_name(
        &self,
        receiver: &crate::ir::hir::HirExpr,
    ) -> Option<String> {
        // Strategy 1: Look up receiver.ty directly in current_hir_types
        if let Some(decl) = self.current_hir_types.get(&receiver.ty) {
            if let Some(name) = self.extract_class_native_name(decl) {
                return Some(name);
            }
        }

        // Strategy 2: If the receiver is a variable, get its symbol's type_id
        // and look that up. Extern class variables may have a different TypeId
        // than the class declaration.
        if let crate::ir::hir::HirExprKind::Variable { symbol, .. } = &receiver.kind {
            if let Some(sym) = self.symbol_table.get_symbol(*symbol) {
                // Try the symbol's type_id
                if let Some(decl) = self.current_hir_types.get(&sym.type_id) {
                    if let Some(name) = self.extract_class_native_name(decl) {
                        return Some(name);
                    }
                }
                // Try the symbol's native_name → look for matching class
                if let Some(native) = sym.native_name {
                    if let Some(native_str) = self.string_interner.get(native) {
                        return Some(native_str.replace("::", "_"));
                    }
                }
                // Try qualified_name (e.g. "sys.net.Host" → "sys_net_Host")
                // This covers extern classes where native_name wasn't propagated
                // across compilation contexts but qualified_name was set by the import resolver.
                if let Some(qn) = sym.qualified_name {
                    if let Some(qs) = self.string_interner.get(qn) {
                        // Only use qualified name if it contains a package separator
                        // (avoid turning "Host" into "Host" redundantly)
                        if qs.contains('.') {
                            return Some(qs.replace('.', "_"));
                        }
                    }
                }
            }

            // Strategy 3: Check register_class_hints via symbol_map.
            // This handles runtime-backed types whose stdlib class isn't loaded as HIR
            // (e.g., ArrayIterator from arr.iterator(), ArrayKeyValueIterator).
            if let Some(reg) = self.symbol_map.get(symbol) {
                if let Some(hint) = self.register_class_hints.get(reg) {
                    return Some(hint.clone());
                }
            }
        }

        None
    }

    pub(crate) fn find_typedef_anonymous_target_by_name(&self, qname: &str) -> Option<TypeId> {
        let type_table = self.type_table;
        for (_tid, ti) in type_table.iter() {
            if let TypeKind::TypeAlias {
                symbol_id,
                target_type,
                ..
            } = &ti.kind
            {
                let matches = self
                    .symbol_table
                    .get_symbol(*symbol_id)
                    .and_then(|s| s.qualified_name.and_then(|n| self.string_interner.get(n)))
                    == Some(qname);
                if matches {
                    let resolved = self.resolve_through_aliases(*target_type);
                    if matches!(
                        type_table.get(resolved).map(|t| &t.kind),
                        Some(TypeKind::Anonymous { .. })
                    ) {
                        return Some(resolved);
                    }
                }
            }
        }
        None
    }

    /// Check if a type is a class type and return its SymbolId
    pub(crate) fn get_class_symbol(&self, type_id: TypeId) -> Option<SymbolId> {
        self.resolve_receiver_class_symbol(type_id)
    }

    /// Check if a type is a @:derive(Copy) class, returning its SymbolId.
    pub(crate) fn get_copy_class_symbol(&self, type_id: TypeId) -> Option<SymbolId> {
        let (kind_class_sym, generic_base) = {
            let type_table = self.type_table;
            let type_ref = type_table.get(type_id)?;
            match &type_ref.kind {
                TypeKind::Class { symbol_id, .. } => (Some(*symbol_id), None),
                TypeKind::GenericInstance { base_type, .. } => (None, Some(*base_type)),
                _ => (None, None),
            }
        };
        if let Some(sym) = kind_class_sym {
            if self.derive_copy_classes.contains(&sym) {
                return Some(sym);
            }
        }
        if let Some(base) = generic_base {
            return self.get_copy_class_symbol(base);
        }
        None
    }

    /// Determine the drop behavior for a type using the Drop trait system.
    ///
    /// Uses the type's `is_extern` flag to identify runtime-managed types.
    /// - Extern classes (Thread, Channel, Arc, etc.) → RuntimeManaged
    /// - User-defined classes → AutoDrop
    /// - Primitives, arrays, Dynamic → NoDrop
    pub(crate) fn get_drop_behavior(&self, type_id: TypeId) -> DropBehavior {
        let type_table = self.type_table;
        if let Some(type_info) = type_table.get(type_id) {
            match &type_info.kind {
                // GenericInstance: Check if the base type is extern (e.g., Arc<T>, Channel<T>)
                TypeKind::GenericInstance { base_type, .. } => {
                    if let Some(base_info) = type_table.get(*base_type) {
                        if let Some(symbol_id) = base_info.symbol_id() {
                            if let Some(symbol) = self.symbol_table.get_symbol(symbol_id) {
                                // Use SymbolFlags::EXTERN from the TypedAST
                                if symbol.flags.contains(SymbolFlags::EXTERN) {
                                    return DropBehavior::RuntimeManaged;
                                }
                                // Fallback: Check stdlib class by native name or simple name
                                if self.is_stdlib_class_by_symbol(symbol) {
                                    return DropBehavior::RuntimeManaged;
                                }
                            }
                        }
                        // Base type is a non-extern Class
                        if matches!(&base_info.kind, TypeKind::Class { .. }) {
                            return DropBehavior::AutoDrop;
                        }
                    }
                    DropBehavior::NoDrop
                }
                TypeKind::Class { symbol_id, .. } => {
                    // Use SymbolFlags from the TypedAST
                    if let Some(symbol) = self.symbol_table.get_symbol(*symbol_id) {
                        // @:cstruct classes are user-managed (no AutoDrop header)
                        if symbol.flags.is_cstruct() {
                            return DropBehavior::RuntimeManaged;
                        }
                        if symbol.flags.contains(SymbolFlags::EXTERN) {
                            return DropBehavior::RuntimeManaged;
                        }
                        // Fallback: Check if it's a known stdlib class by native name or simple name.
                        // This handles cases where the EXTERN flag wasn't propagated
                        // through the symbol table (e.g., stdlib types loaded via imports
                        // where the local symbol doesn't carry the extern flag).
                        if self.is_stdlib_class_by_symbol(symbol) {
                            return DropBehavior::RuntimeManaged;
                        }
                    }
                    // Check for @:derive(Drop) — user-defined destructor
                    if self.derive_drop_classes.contains(symbol_id) {
                        return DropBehavior::AutoDropWithDtor;
                    }
                    // Check for @:manualDrop — user manages lifetime
                    if self.manual_drop_classes.contains(symbol_id) {
                        return DropBehavior::ManualDrop;
                    }
                    // User-defined (non-extern) classes need AutoDrop
                    DropBehavior::AutoDrop
                }
                // Arrays: runtime-managed buffer, no Free needed
                TypeKind::Array { .. } => DropBehavior::NoDrop,
                // Dynamic: unknown ownership at compile time, no Free
                TypeKind::Dynamic => DropBehavior::NoDrop,
                // Other types don't need drop
                _ => DropBehavior::NoDrop,
            }
        } else {
            DropBehavior::NoDrop
        }
    }

    /// @:derive(Drop) — get the class SymbolId for an IrId if it's a Drop class.
    /// Uses register_class_hints to map IrId → class name, then resolves to SymbolId.
    pub(crate) fn get_drop_class_for_ir(&self, ir_id: IrId) -> Option<SymbolId> {
        if self.derive_drop_classes.is_empty() {
            return None;
        }
        // First check the direct mapping
        if let Some(&class_sym) = self.ir_to_drop_class.get(&ir_id) {
            return Some(class_sym);
        }
        // Fallback: use register_class_hints to find the class name, then resolve
        let class_name = self.register_class_hints.get(&ir_id)?;
        for &class_sym in &self.derive_drop_classes {
            if let Some(symbol) = self.symbol_table.get_symbol(class_sym) {
                let sym_name = self.string_interner.get(symbol.name);
                if sym_name == Some(class_name.as_str()) {
                    return Some(class_sym);
                }
            }
        }
        None
    }

    /// Get collected enum RTTI data for registration
    pub fn get_enums_for_registration(&self) -> &BTreeMap<SymbolId, (u32, String, Vec<String>)> {
        &self.enums_for_registration
    }

    pub(crate) fn get_extern_function_signature(
        &self,
        name: &str,
    ) -> Option<(Vec<IrType>, IrType)> {
        // Query the runtime_mapping.rs type registry for explicit type information
        self.stdlib_mapping.get_function_signature(name)
    }

    /// Check if a type is an interface type and return its SymbolId.
    /// Also handles TypeParameters with interface constraints (T:Printable).
    pub(crate) fn get_interface_symbol(&self, type_id: TypeId) -> Option<SymbolId> {
        let type_table = self.type_table;
        let type_ref = type_table.get(type_id)?;
        match &type_ref.kind {
            TypeKind::Interface { symbol_id, .. } => Some(*symbol_id),
            TypeKind::TypeParameter { constraints, .. } => {
                // For constrained type params, find the first interface constraint
                for constraint_id in constraints {
                    if let Some(constraint_type) = type_table.get(*constraint_id) {
                        if let TypeKind::Interface { symbol_id, .. } = &constraint_type.kind {
                            return Some(*symbol_id);
                        }
                    }
                }
                None
            }
            // Cross-file `Array<I>` where I is declared in another file
            // sometimes lands as a Placeholder during this file's TAST
            // lowering. The resolved Interface symbol IS present in the
            // shared `interface_method_names` map — look the name up
            // there to recover the proper SymbolId so element-wrap and
            // virtual-dispatch paths fire correctly.
            // Cross-file `Array<I>` where I is declared in another file
            // sometimes lands as a Placeholder during this file's TAST
            // lowering. Resolve via:
            //   (1) already-loaded interface_method_names (fast path), or
            //   (2) symbol-table walk for a symbol with that name whose
            //       kind is Interface (handles the case where the file
            //       declaring `I` hasn't run register_interface_metadata
            //       in *this* context yet — see also the lazy vtable
            //       construction in `wrap_in_interface_fat_ptr`).
            TypeKind::Placeholder { name } => {
                if let Some(&sym) = self.interface_method_names.keys().find(|sym_id| {
                    self.symbol_table
                        .get_symbol(**sym_id)
                        .map(|s| s.name == *name)
                        .unwrap_or(false)
                }) {
                    return Some(sym);
                }
                use crate::tast::SymbolKind;
                self.symbol_table
                    .all_symbols()
                    .find(|s| s.name == *name && matches!(s.kind, SymbolKind::Interface))
                    .map(|s| s.id)
            }
            _ => None,
        }
    }

    /// Check if a qualified name + method belongs to rayzor stdlib and return the runtime function name
    ///
    /// For static methods like Thread.spawn, Channel.init, etc.
    /// Uses StdlibMapping as single source of truth - no hardcoded mappings!
    pub(crate) fn get_static_stdlib_runtime_func(
        &self,
        qualified_name: &str,
        method_name: &str,
    ) -> Option<&'static str> {
        self.get_static_stdlib_runtime_func_with_params_internal(qualified_name, method_name, None)
    }

    /// Param-count aware static stdlib lookup.
    ///
    /// Prefer this where call argument count is known to avoid collisions such as
    /// `Std.random(Int)` vs `Math.random()`.
    pub(crate) fn get_static_stdlib_runtime_func_with_params(
        &self,
        qualified_name: &str,
        method_name: &str,
        param_count: usize,
    ) -> Option<&'static str> {
        self.get_static_stdlib_runtime_func_with_params_internal(
            qualified_name,
            method_name,
            Some(param_count),
        )
    }

    pub(crate) fn get_static_stdlib_runtime_func_with_params_internal(
        &self,
        qualified_name: &str,
        method_name: &str,
        param_count: Option<usize>,
    ) -> Option<&'static str> {
        // Parse qualified name to extract class name
        // Patterns: "rayzor.concurrent.Thread.spawn", "test.Thread.spawn", "StringTools.startsWith"
        let parts: Vec<&str> = qualified_name.split('.').collect();
        let class_name = parts.iter().rev().nth(1)?; // Second-to-last part is class name

        // Use StdlibMapping to find the runtime function
        // This is the ONLY source of truth - all mappings come from the actual Rust implementations

        // PRIORITY: First try qualified class name with underscore format (e.g., "sys_thread_Thread")
        // This handles sys.thread.Thread -> sys_thread_Thread mappings and prevents
        // conflicts between sys.thread.Thread and rayzor.concurrent.Thread
        if parts.len() >= 2 {
            // Build qualified class name: all parts except the last (method name), joined with underscore
            let qualified_class_name = parts[..parts.len() - 1].join("_");
            if let Some(count) = param_count {
                if let Some((_sig, mapping)) = self.stdlib_mapping.find_by_name_and_params(
                    &qualified_class_name,
                    method_name,
                    count,
                ) {
                    return Some(mapping.runtime_name);
                }
            }
            if let Some((_sig, mapping)) = self
                .stdlib_mapping
                .find_by_name(&qualified_class_name, method_name)
            {
                return Some(mapping.runtime_name);
            }
        }

        // FALLBACK: Try simple class name (e.g., "Thread", "Json") ONLY when the
        // qualified name is in a stdlib namespace. Without this guard, user
        // classes like `tink.Json` would hijack stdlib method dispatch via
        // simple-name matching, causing `tink.Json.parse` to silently route
        // to `haxe.Json.parse` (haxe_json_parse).
        let in_stdlib_ns = parts.len() == 1
            || qualified_name.starts_with("haxe.")
            || qualified_name.starts_with("sys.")
            || qualified_name.starts_with("rayzor.")
            || qualified_name.starts_with("StringTools")
            || qualified_name.starts_with("Std.")
            || qualified_name.starts_with("Math.")
            || qualified_name.starts_with("Type.")
            || qualified_name.starts_with("Reflect.");
        if !in_stdlib_ns {
            return None;
        }
        if let Some(count) = param_count {
            if let Some((_sig, mapping)) =
                self.stdlib_mapping
                    .find_by_name_and_params(class_name, method_name, count)
            {
                return Some(mapping.runtime_name);
            }
            // With a known arg count, avoid arity-blind fallback here.
            // Callers can decide whether to do a global static fallback by (name, arity).
            return None;
        }
        if let Some((_sig, mapping)) = self.stdlib_mapping.find_by_name(class_name, method_name) {
            return Some(mapping.runtime_name);
        }

        None
    }

    /// Get the correct signature for a stdlib MIR wrapper function.
    /// These signatures MUST match what's defined in compiler/src/stdlib/{thread,channel,sync}.rs
    /// Returns (param_types, return_type) or None if not a known MIR wrapper.
    pub(crate) fn get_stdlib_mir_wrapper_signature(
        &self,
        name: &str,
    ) -> Option<(Vec<IrType>, IrType)> {
        // Query the runtime_mapping.rs type registry for explicit type information
        self.stdlib_mapping.get_function_signature(name)
    }

    pub(crate) fn get_stdlib_runtime_info(
        &self,
        method_symbol: SymbolId,
        receiver_type: TypeId,
        param_count: Option<usize>,
        receiver_class_hint: Option<&str>,
    ) -> Option<(
        &'static str,
        &'static str,
        &crate::stdlib::RuntimeFunctionCall,
    )> {
        let found = self.get_stdlib_runtime_info_inner(
            method_symbol,
            receiver_type,
            param_count,
            receiver_class_hint,
        )?;
        if let Some(want) = param_count {
            let got = found.2.param_count;
            if got != want && std::env::var_os("RAYZOR_STDLIB_ARITY_WARN").is_some() {
                eprintln!(
                    "[stdlib-arity] {}.{} called with {} arg(s) but runtime mapping '{}' takes {} \
                     — the call would be emitted against a mismatched native signature. \
                     Either the .hx declaration promises parameters the runtime does not \
                     implement, or the mapping's `params:` is wrong.",
                    found.0, found.1, want, found.2.runtime_name, got
                );
            }
            if got != want && std::env::var_os("RAYZOR_STRICT_STDLIB_ARITY").is_some() {
                return None;
            }
        }
        Some(found)
    }

    pub(crate) fn get_stdlib_runtime_info_inner(
        &self,
        method_symbol: SymbolId,
        receiver_type: TypeId,
        param_count: Option<usize>,
        receiver_class_hint: Option<&str>,
    ) -> Option<(
        &'static str,
        &'static str,
        &crate::stdlib::RuntimeFunctionCall,
    )> {
        // Get the method name and optional qualified name from the symbol table
        // Prefer @:native name over Haxe name for runtime mapping lookup
        let (method_name, qualified_name, method_is_extern) =
            if let Some(symbol) = self.symbol_table.get_symbol(method_symbol) {
                let name = self.string_interner.get(symbol.name)?;
                let qname = symbol
                    .qualified_name
                    .and_then(|qn| self.string_interner.get(qn));
                (name, qname, symbol.flags.contains(SymbolFlags::EXTERN))
            } else {
                return None;
            };

        let allow_generic_monomorphization = self
            .resolve_receiver_class_symbol(receiver_type)
            .and_then(|symbol_id| self.symbol_table.get_symbol(symbol_id))
            .map_or(false, |symbol| {
                if !symbol.flags.contains(SymbolFlags::EXTERN) {
                    return false;
                }
                let class_name = match self.string_interner.get(symbol.name) {
                    Some(name) => name,
                    None => return false,
                };
                let has_monomorphized_variants = !self
                    .stdlib_mapping
                    .get_monomorphized_variants(class_name)
                    .is_empty();
                if !has_monomorphized_variants {
                    return false;
                }

                let in_stdlib_namespace = symbol
                    .qualified_name
                    .and_then(|qn| self.string_interner.get(qn))
                    .map_or(false, |qn| {
                        qn.starts_with("rayzor.")
                            || qn.starts_with("sys.")
                            || qn.starts_with("haxe.")
                    });
                in_stdlib_namespace || self.is_stdlib_class_by_symbol(symbol)
            });

        // Get the class name and type args from the receiver type
        let type_table = self.type_table;
        let type_info = type_table.get(receiver_type);

        // GUARD: If receiver is a user-defined class (not a known stdlib/extern class),
        // return None immediately. User class methods are resolved by the method
        // resolution code, not through stdlib runtime mappings. Without this guard,
        // common method names like "add", "get", "set", "length" would incorrectly
        // match stdlib methods (e.g., Point2D.add → sys_deque_add).
        if let Some(ti) = type_info {
            match &ti.kind {
                crate::tast::core::TypeKind::Class { symbol_id, .. } => {
                    let is_known_stdlib = self
                        .symbol_table
                        .get_symbol(*symbol_id)
                        .map(|s| {
                            let name = self.string_interner.get(s.name).unwrap_or("");
                            let is_extern = s.flags.contains(SymbolFlags::EXTERN);
                            let is_wrapper = self.stdlib_mapping.is_mir_wrapper_class(name);
                            // Check if class is in stdlib namespace via qualified name
                            let in_stdlib_ns = s
                                .qualified_name
                                .and_then(|qn| self.string_interner.get(qn))
                                .map(|qn| {
                                    qn.starts_with("haxe.")
                                        || qn.starts_with("sys.")
                                        || qn.starts_with("rayzor.")
                                })
                                .unwrap_or(false);
                            // Check if the stdlib mapping has ANY method for this class.
                            // IMPORTANT: this must be combined with in_stdlib_ns because
                            // class_has_any_method matches by simple name (e.g., "Json"),
                            // and user classes like `tink.Json` have the same simple name
                            // as stdlib `haxe.Json`. Without the namespace check, user
                            // macros would be hijacked by stdlib method mappings.
                            let has_stdlib_methods_in_ns =
                                self.stdlib_mapping.class_has_any_method(name) && in_stdlib_ns;
                            is_extern || is_wrapper || in_stdlib_ns || has_stdlib_methods_in_ns
                        })
                        .unwrap_or(false);
                    if !is_known_stdlib {
                        return None;
                    }
                }
                crate::tast::core::TypeKind::Placeholder { name } => {
                    // Placeholder types arise from unresolved cross-module types.
                    // Check if the placeholder name maps to a known stdlib class.
                    // If not, it's a user class and we should not match stdlib methods.
                    let placeholder_name = self.string_interner.get(*name);
                    let is_known_stdlib = placeholder_name
                        .map(|n| {
                            self.stdlib_mapping.is_mir_wrapper_class(n)
                                || self.stdlib_mapping.class_has_any_method(n)
                                || n.starts_with("haxe.")
                                || n.starts_with("sys.")
                        })
                        .unwrap_or(false);
                    if !is_known_stdlib {
                        return None;
                    }
                }
                _ => {}
            }
        }

        debug!(
            "[get_stdlib_runtime_info] method={}, receiver_type={:?}, type_info_exists={}, qualified_name={:?}",
            method_name,
            receiver_type,
            type_info.is_some(),
            qualified_name
        );

        // FALLBACK: If receiver_type is invalid (extern classes like Vec), try to detect class from qualified name
        if type_info.is_none() {
            debug!(
                "[get_stdlib_runtime_info] receiver_type {:?} not in type_table, qualified_name={:?}",
                receiver_type, qualified_name
            );
            if let Some(qname) = qualified_name {
                // Pattern: "rayzor.Vec.get" or "rayzor.concurrent.MutexGuard.get"
                let parts: Vec<&str> = qname.split('.').collect();
                if parts.len() >= 2 {
                    let class_parts = &parts[..parts.len() - 1];
                    let class_name = class_parts[class_parts.len() - 1];
                    let qualified_class_name = class_parts.join("_");

                    if let Some(count) = param_count {
                        if let Some((sig, mapping)) = self.stdlib_mapping.find_by_name_and_params(
                            &qualified_class_name,
                            method_name,
                            count,
                        ) {
                            return Some((sig.class, sig.method, mapping));
                        }
                    }
                    if let Some((sig, mapping)) = self
                        .stdlib_mapping
                        .find_by_name(&qualified_class_name, method_name)
                    {
                        return Some((sig.class, sig.method, mapping));
                    }

                    // For unresolved extern classes in stdlib namespaces, allow a
                    // simple-class fallback as a compatibility path.
                    // Keep this disabled for non-stdlib namespaces to avoid user
                    // class collisions (e.g., test.Vec, test.Thread, ...).
                    let allow_simple_fallback = class_parts.len() == 1
                        || qualified_class_name.starts_with("rayzor_")
                        || qualified_class_name.starts_with("sys_")
                        || qualified_class_name.starts_with("haxe_");
                    if method_is_extern && allow_simple_fallback {
                        if let Some(count) = param_count {
                            if let Some((sig, mapping)) = self
                                .stdlib_mapping
                                .find_by_name_and_params(class_name, method_name, count)
                            {
                                return Some((sig.class, sig.method, mapping));
                            }
                        }
                        if let Some((sig, mapping)) =
                            self.stdlib_mapping.find_by_name(class_name, method_name)
                        {
                            return Some((sig.class, sig.method, mapping));
                        }

                        let variants = self.stdlib_mapping.get_monomorphized_variants(class_name);
                        if !variants.is_empty() {
                            for variant in variants {
                                if let Some(count) = param_count {
                                    if let Some((sig, mapping)) = self
                                        .stdlib_mapping
                                        .find_by_name_and_params(variant, method_name, count)
                                    {
                                        return Some((sig.class, sig.method, mapping));
                                    }
                                }
                                if let Some((sig, mapping)) =
                                    self.stdlib_mapping.find_by_name(variant, method_name)
                                {
                                    return Some((sig.class, sig.method, mapping));
                                }
                            }
                        }
                    }
                }
            }

            // FALLBACK 1.5: Try to get class name from HIR type declarations.
            // This handles extern classes (like Tensor) whose receiver_type isn't in
            // the type_table but IS in current_hir_types.
            if let Some(type_decl) = self.current_hir_types.get(&receiver_type) {
                if let HirTypeDecl::Class(class) = type_decl {
                    let class_name_str = self.string_interner.get(class.name).unwrap_or("");
                    // Try with class name directly (e.g., "Tensor")
                    if let Some((sig, mapping)) = self
                        .stdlib_mapping
                        .find_by_name(class_name_str, method_name)
                    {
                        debug!(
                            "[FALLBACK1.5] Found {}.{} via HIR type decl",
                            class_name_str, method_name
                        );
                        return Some((sig.class, sig.method, mapping));
                    }
                    // Try with native name metadata (e.g., "rayzor::ds::Tensor" -> "rayzor_ds_Tensor")
                    for attr in &class.metadata {
                        let attr_name = self.string_interner.get(attr.name).unwrap_or("");
                        if attr_name == "native" {
                            if let Some(first_arg) = attr.args.first() {
                                if let crate::ir::hir::HirAttributeArg::Literal(
                                    crate::ir::hir::HirLiteral::String(s),
                                ) = first_arg
                                {
                                    let native_str = self.string_interner.get(*s).unwrap_or("");
                                    // Convert "rayzor::ds::Tensor" to "rayzor_ds_Tensor"
                                    let normalized = native_str.replace("::", "_");
                                    if let Some((sig, mapping)) =
                                        self.stdlib_mapping.find_by_name(&normalized, method_name)
                                    {
                                        debug!(
                                            "[FALLBACK1.5] Found {}.{} via @:native class name",
                                            normalized, method_name
                                        );
                                        return Some((sig.class, sig.method, mapping));
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // FALLBACK: Use receiver_class_hint if available.
            // This handles monomorphized extern classes like Vec<Int> -> VecI32 where
            // the receiver TypeId is not in type_table but the New handler stored a
            // class hint on the receiver register.
            if let Some(hint) = receiver_class_hint {
                if let Some(count) = param_count {
                    if let Some((sig, mapping)) =
                        self.stdlib_mapping
                            .find_by_name_and_params(hint, method_name, count)
                    {
                        return Some((sig.class, sig.method, mapping));
                    }
                }
                if let Some((sig, mapping)) = self.stdlib_mapping.find_by_name(hint, method_name) {
                    return Some((sig.class, sig.method, mapping));
                }
            }

            // No qualified name and no valid receiver type — cannot resolve stdlib method.
            // Do NOT brute-force search all stdlib classes by bare method name, as this
            // causes false positives (e.g., user field "current" matching Thread.current).
            return None;
        }

        let type_info = type_info.unwrap();

        // Extract class info from the type, following TypeAlias if needed
        let (base_class_name, qualified_class_name, type_args) = match &type_info.kind {
            TypeKind::String => (Some("String"), None, Vec::new()),
            TypeKind::Array { .. } => (Some("Array"), None, Vec::new()),
            TypeKind::Enum { .. } => (Some("Enum"), None, Vec::new()),
            TypeKind::Map { key_type, .. } => {
                // Map types route to IntMap, StringMap, or ObjectMap based on key type
                let key_kind = type_table.get(*key_type).map(|t| t.kind.clone());
                let class = match key_kind {
                    Some(TypeKind::Int) | Some(TypeKind::Bool) => "IntMap",
                    Some(TypeKind::String) => "StringMap",
                    _ => "ObjectMap",
                };
                (Some(class), None, Vec::new())
            }
            TypeKind::Class {
                symbol_id,
                type_args,
                ..
            } => {
                // Get class name and qualified name from symbol
                let (name, qname) =
                    if let Some(class_info) = self.symbol_table.get_symbol(*symbol_id) {
                        let n = self.string_interner.get(class_info.name);
                        // Prefer lowered @:native name, fall back to qualified name
                        let qn = if let Some(native) = class_info.native_name {
                            self.string_interner
                                .get(native)
                                .map(|n| n.replace("::", "_"))
                        } else {
                            class_info
                                .qualified_name
                                .and_then(|qn| self.string_interner.get(qn))
                                .map(|qn| qn.replace(".", "_"))
                        };
                        (n, qn)
                    } else {
                        (None, None)
                    };
                (name, qname, type_args.clone())
            }
            TypeKind::TypeAlias { target_type, .. } => {
                // For type aliases like `typedef Bytes = rayzor.Bytes`, follow the target type
                if let Some(target_info) = type_table.get(*target_type) {
                    match &target_info.kind {
                        TypeKind::Class {
                            symbol_id,
                            type_args,
                            ..
                        } => {
                            let (name, qname) = if let Some(class_info) =
                                self.symbol_table.get_symbol(*symbol_id)
                            {
                                let n = self.string_interner.get(class_info.name);
                                let qn = if let Some(native) = class_info.native_name {
                                    self.string_interner
                                        .get(native)
                                        .map(|n| n.replace("::", "_"))
                                } else {
                                    class_info
                                        .qualified_name
                                        .and_then(|qn| self.string_interner.get(qn))
                                        .map(|qn| qn.replace(".", "_"))
                                };
                                (n, qn)
                            } else {
                                (None, None)
                            };
                            (name, qname, type_args.clone())
                        }
                        TypeKind::Placeholder {
                            name: placeholder_name,
                        } => {
                            // The typedef target wasn't resolved at compile time - try to look it up by name
                            // This handles cases like `typedef Bytes = rayzor.Bytes` where the target was loaded
                            // after the typedef was initially compiled
                            if let Some(target_name) = self.string_interner.get(*placeholder_name) {
                                // Convert "rayzor.Bytes" to "rayzor_Bytes" for stdlib mapping lookup
                                let qualified_name = target_name.replace(".", "_");
                                if let Some((_sig, mapping)) = self
                                    .stdlib_mapping
                                    .find_by_name(&qualified_name, method_name)
                                {
                                    // Early return with the mapping
                                    return Some((_sig.class, _sig.method, mapping));
                                }
                            }
                            (None, None, Vec::new())
                        }
                        _ => (None, None, Vec::new()),
                    }
                } else {
                    (None, None, Vec::new())
                }
            }
            TypeKind::Abstract {
                symbol_id,
                type_args,
                ..
            } => {
                // For abstract types like Ptr<T>, Ref<T>, Box<T>, Usize
                // These are zero-cost abstracts over Int with methods in stdlib_mapping
                let (name, qname) =
                    if let Some(class_info) = self.symbol_table.get_symbol(*symbol_id) {
                        let n = self.string_interner.get(class_info.name);
                        let qn = if let Some(native) = class_info.native_name {
                            self.string_interner
                                .get(native)
                                .map(|n| n.replace("::", "_"))
                        } else {
                            class_info
                                .qualified_name
                                .and_then(|qn| self.string_interner.get(qn))
                                .map(|qn| qn.replace(".", "_"))
                        };
                        (n, qn)
                    } else {
                        (None, None)
                    };
                (name, qname, type_args.clone())
            }
            TypeKind::GenericInstance {
                base_type,
                type_args,
                ..
            } => {
                // For generic instances like Deque<Int>, Channel<String>, Arc<T>
                // Get the base type's class name (e.g., "Deque" from Deque<Int>)
                if let Some(base_info) = type_table.get(*base_type) {
                    // Generic enums (e.g., Option<Int>) resolve to "Enum" class
                    if matches!(&base_info.kind, TypeKind::Enum { .. }) {
                        return self
                            .stdlib_mapping
                            .find_by_name("Enum", method_name)
                            .map(|(sig, mapping)| (sig.class, sig.method, mapping));
                    }
                    let sym_id = match &base_info.kind {
                        TypeKind::Class { symbol_id, .. } => Some(*symbol_id),
                        TypeKind::Abstract { symbol_id, .. } => Some(*symbol_id),
                        _ => None,
                    };
                    if let Some(symbol_id) = sym_id {
                        let (name, qname) =
                            if let Some(class_info) = self.symbol_table.get_symbol(symbol_id) {
                                let n = self.string_interner.get(class_info.name);
                                let qn = if let Some(native) = class_info.native_name {
                                    self.string_interner
                                        .get(native)
                                        .map(|n| n.replace("::", "_"))
                                } else {
                                    class_info
                                        .qualified_name
                                        .and_then(|qn| self.string_interner.get(qn))
                                        .map(|qn| qn.replace(".", "_"))
                                };
                                (n, qn)
                            } else {
                                (None, None)
                            };
                        (name, qname, type_args.clone())
                    } else {
                        (None, None, Vec::new())
                    }
                } else {
                    (None, None, Vec::new())
                }
            }
            // TypeParameter / Dynamic / Placeholder: the receiver type doesn't tell us the class.
            // Return None to let the caller's fallback chain handle resolution.
            // The caller has context-specific handlers (Dynamic method handler, function_map, etc.)
            // that work better than brute-force stdlib search for these cases.
            TypeKind::Placeholder {
                name: placeholder_name,
            } => {
                // Use the Placeholder name to derive the class for stdlib lookup.
                // e.g., "rayzor.Bytes" → try "rayzor_Bytes", "Bytes"
                let ph_name = self
                    .string_interner
                    .get(*placeholder_name)
                    .map(|s| s.to_string());
                if let Some(ref ph) = ph_name {
                    let underscore_name = ph.replace(".", "_");
                    let bare_name = ph.rsplit('.').next().unwrap_or(ph);
                    // Try qualified underscore name (e.g., "rayzor_Bytes")
                    if let Some(count) = param_count {
                        if let Some((sig, mapping)) = self.stdlib_mapping.find_by_name_and_params(
                            &underscore_name,
                            method_name,
                            count,
                        ) {
                            return Some((sig.class, sig.method, mapping));
                        }
                        if let Some((sig, mapping)) = self.stdlib_mapping.find_by_name_and_params(
                            bare_name,
                            method_name,
                            count,
                        ) {
                            return Some((sig.class, sig.method, mapping));
                        }
                    }
                    if let Some((sig, mapping)) = self
                        .stdlib_mapping
                        .find_by_name(&underscore_name, method_name)
                    {
                        return Some((sig.class, sig.method, mapping));
                    }
                    if let Some((sig, mapping)) =
                        self.stdlib_mapping.find_by_name(bare_name, method_name)
                    {
                        return Some((sig.class, sig.method, mapping));
                    }
                }
                // Fall through to qualified_name and hint lookups
                if let Some(qname) = qualified_name {
                    let parts: Vec<&str> = qname.split('.').collect();
                    if parts.len() >= 2 {
                        let class_parts = &parts[..parts.len() - 1];
                        let underscore_class = class_parts.join("_");
                        if let Some((sig, mapping)) = self
                            .stdlib_mapping
                            .find_by_name(&underscore_class, method_name)
                        {
                            return Some((sig.class, sig.method, mapping));
                        }
                    }
                }
                if let Some(hint) = receiver_class_hint {
                    if let Some((sig, mapping)) =
                        self.stdlib_mapping.find_by_name(hint, method_name)
                    {
                        return Some((sig.class, sig.method, mapping));
                    }
                }
                return None;
            }
            TypeKind::TypeParameter { .. } | TypeKind::Dynamic | TypeKind::Unknown => {
                // Try qualified_name if available (e.g., for user-class methods like "test.Counter.increment")
                if let Some(qname) = qualified_name {
                    let parts: Vec<&str> = qname.split('.').collect();
                    if parts.len() >= 2 {
                        let class_parts = &parts[..parts.len() - 1];
                        let underscore_class = class_parts.join("_");
                        if let Some((sig, mapping)) = self
                            .stdlib_mapping
                            .find_by_name(&underscore_class, method_name)
                        {
                            return Some((sig.class, sig.method, mapping));
                        }
                        if let Some(&class_name) = parts.iter().rev().nth(1) {
                            if let Some((sig, mapping)) =
                                self.stdlib_mapping.find_by_name(class_name, method_name)
                            {
                                return Some((sig.class, sig.method, mapping));
                            }
                        }
                    }
                }
                // Try receiver_class_hint for disambiguation (e.g., guard.get() where guard
                // is typed as Dynamic but hint says "rayzor_concurrent_MutexGuard")
                if let Some(hint) = receiver_class_hint {
                    if let Some((sig, mapping)) =
                        self.stdlib_mapping.find_by_name(hint, method_name)
                    {
                        return Some((sig.class, sig.method, mapping));
                    }
                }
                return None;
            }
            _ => (None, None, Vec::new()),
        };

        let base_class_name = base_class_name?;

        // MONOMORPHIZATION: For generic extern classes like Vec<T>, monomorphize the class name
        // based on type arguments. Vec<Int> -> VecI32, Vec<Float> -> VecF64, etc.
        let monomorphized_class_name: Option<String> =
            if allow_generic_monomorphization && !type_args.is_empty() {
                let first_arg = type_args[0];
                let suffix = if let Some(arg_type) = type_table.get(first_arg) {
                    match &arg_type.kind {
                        TypeKind::Int => Some("I32"),
                        TypeKind::Float => Some("F64"),
                        TypeKind::Bool => Some("Bool"),
                        TypeKind::String => Some("Ptr"), // Strings are reference types
                        TypeKind::Class { symbol_id, .. } => {
                            // Check if it's Int64 (a class type representing 64-bit int)
                            if let Some(class_info) = self.symbol_table.get_symbol(*symbol_id) {
                                if let Some(name) = self.string_interner.get(class_info.name) {
                                    if name == "Int64" {
                                        Some("I64")
                                    } else {
                                        Some("Ptr") // Other classes are reference types
                                    }
                                } else {
                                    Some("Ptr")
                                }
                            } else {
                                Some("Ptr")
                            }
                        }
                        _ => Some("Ptr"), // Default to pointer for other types
                    }
                } else {
                    Some("Ptr") // Unknown type, use pointer
                };
                suffix.and_then(|s| {
                    let candidate = format!("{}{}", base_class_name, s);
                    let variants = self
                        .stdlib_mapping
                        .get_monomorphized_variants(base_class_name);
                    if variants.iter().any(|v| *v == candidate) {
                        Some(candidate)
                    } else {
                        None
                    }
                })
            } else {
                None
            };

        // Use monomorphized name if available, then receiver_class_hint, then base name.
        // receiver_class_hint comes from the New handler (e.g., Vec<Int> -> "VecI32") and is
        // needed when allow_generic_monomorphization is false (e.g., qualified_name is "Vec"
        // not "rayzor.Vec" so the stdlib namespace check fails).
        let class_name = monomorphized_class_name
            .as_deref()
            .or(receiver_class_hint)
            .unwrap_or(base_class_name);

        // PRIORITY: If receiver_class_hint is a fully qualified name (contains '.'),
        // normalize it and try exact lookup FIRST. This handles subclass method dispatch
        // (e.g., sys.ssl.Socket.close should call rayzor_ssl_socket_close, not sys_net_Socket_close).
        if let Some(hint) = receiver_class_hint {
            if hint.contains('.') {
                let normalized_hint = hint.replace(".", "_");
                if let Some(count) = param_count {
                    if let Some((sig, mapping)) = self.stdlib_mapping.find_by_name_and_params(
                        &normalized_hint,
                        method_name,
                        count,
                    ) {
                        return Some((sig.class, sig.method, mapping));
                    }
                }
                if let Some((sig, mapping)) = self
                    .stdlib_mapping
                    .find_by_name(&normalized_hint, method_name)
                {
                    return Some((sig.class, sig.method, mapping));
                }
            }
        }

        // Try to find this method in the stdlib mapping
        // First try qualified name (e.g., "rayzor_Bytes"), then fall back to simple name
        if let Some(ref qn) = qualified_class_name {
            // Try with param_count first if available (to disambiguate overloads)
            if let Some(count) = param_count {
                if let Some((sig, mapping)) =
                    self.stdlib_mapping
                        .find_by_name_and_params(qn, method_name, count)
                {
                    debug!(
                        "[get_stdlib_runtime_info] Found {}.{} ({} params) via qualified name -> {}",
                        qn, method_name, count, mapping.runtime_name
                    );
                    debug!(
                        "[get_stdlib_runtime_info] Found {}.{} ({} params) via qualified name -> {}",
                        qn, method_name, count, mapping.runtime_name
                    );
                    return Some((sig.class, sig.method, mapping));
                }
            }
            // Fallback to find_by_name without param count
            if let Some((sig, mapping)) = self.stdlib_mapping.find_by_name(qn, method_name) {
                debug!(
                    "[get_stdlib_runtime_info] Found {}.{} via qualified name -> {}",
                    qn, method_name, mapping.runtime_name
                );
                return Some((sig.class, sig.method, mapping));
            }
        }
        // Try with param_count first if available
        if let Some(count) = param_count {
            if let Some((sig, mapping)) =
                self.stdlib_mapping
                    .find_by_name_and_params(class_name, method_name, count)
            {
                debug!(
                    "[get_stdlib_runtime_info] Found {}.{} ({} params) -> {}",
                    class_name, method_name, count, mapping.runtime_name
                );
                return Some((sig.class, sig.method, mapping));
            }
        }
        // Fallback without param count
        if let Some((sig, mapping)) = self.stdlib_mapping.find_by_name(class_name, method_name) {
            return Some((sig.class, sig.method, mapping));
        }

        // Bare-name fallback: if class_name is qualified (e.g., "haxe.ds.ObjectMap"),
        // try the bare class name (e.g., "ObjectMap") for methods registered with simple names.
        if class_name.contains('.') {
            let bare_name = class_name.rsplit('.').next().unwrap_or(class_name);
            if let Some(count) = param_count {
                if let Some((sig, mapping)) =
                    self.stdlib_mapping
                        .find_by_name_and_params(bare_name, method_name, count)
                {
                    return Some((sig.class, sig.method, mapping));
                }
            }
            if let Some((sig, mapping)) = self.stdlib_mapping.find_by_name(bare_name, method_name) {
                return Some((sig.class, sig.method, mapping));
            }
        }

        // @:forward fallback: if the abstract's own name didn't match, try underlying type
        {
            let type_table = self.type_table;
            if let Some(type_info) = type_table.get(receiver_type) {
                if let TypeKind::Abstract { symbol_id, .. } = &type_info.kind {
                    if let Some((underlying_type, forward_list)) =
                        self.abstract_forward_rules.get(symbol_id)
                    {
                        let method_interned =
                            self.symbol_table.get_symbol(method_symbol).map(|s| s.name);
                        let is_forwarded = forward_list.is_empty()
                            || method_interned.map_or(false, |n| forward_list.contains(&n));
                        if is_forwarded {
                            let underlying = *underlying_type;
                            return self.get_stdlib_runtime_info(
                                method_symbol,
                                underlying,
                                param_count,
                                None,
                            );
                        }
                    }
                }
            }
        }

        None
    }

    pub(crate) fn lookup_class_symbol_by_name(&self, name: &str) -> Option<SymbolId> {
        // Accept a bare or qualified name; compare against both the symbol's
        // qualified name and its bare name, and also the trailing segment of a
        // qualified hint (e.g. hint "nue.arch.LlamaArch" vs symbol "LlamaArch").
        let bare = name.rsplit('.').next().unwrap_or(name);
        for i in 0..self.symbol_table.len() {
            let sid = SymbolId::from_raw(i as u32);
            let Some(sym) = self.symbol_table.get_symbol(sid) else {
                continue;
            };
            if !matches!(sym.kind, crate::tast::SymbolKind::Class) {
                continue;
            }
            let qn_match = sym
                .qualified_name
                .and_then(|n| self.string_interner.get(n))
                .map(|qn| qn == name || qn == bare)
                .unwrap_or(false);
            let bare_match = self
                .string_interner
                .get(sym.name)
                .map(|n| n == name || n == bare)
                .unwrap_or(false);
            if qn_match || bare_match {
                return Some(sid);
            }
        }
        None
    }

    /// Find the current-context Interface SymbolId for a qualified name
    /// (e.g. `"nue.Module"` or bare `"Module"`). Used as the lookup half
    /// of the cross-file Path 3 fallback in `maybe_materialize_for_call`
    /// — see the call site comment for the why.
    pub(crate) fn lookup_interface_symbol_by_qualified_name(&self, name: &str) -> Option<SymbolId> {
        for i in 0..self.symbol_table.len() {
            let sid = SymbolId::from_raw(i as u32);
            let Some(sym) = self.symbol_table.get_symbol(sid) else {
                continue;
            };
            if !matches!(sym.kind, crate::tast::SymbolKind::Interface) {
                continue;
            }
            let matches = sym
                .qualified_name
                .and_then(|n| self.string_interner.get(n))
                .map(|qn| qn == name)
                .unwrap_or(false)
                || self
                    .string_interner
                    .get(sym.name)
                    .map(|n| n == name)
                    .unwrap_or(false);
            if matches {
                return Some(sid);
            }
        }
        None
    }

    /// Resolve an abstract @:from/@:to conversion function to an IrFunctionId.
    /// Tries function_map (user-defined), external_function_map, then stdlib_mapping.
    pub(crate) fn resolve_abstract_conversion_function(
        &mut self,
        symbol: SymbolId,
        target_type: TypeId,
    ) -> Option<IrFunctionId> {
        // 1. Check function_map (user-defined abstract methods)
        if let Some(&func_id) = self.function_map.get(&symbol) {
            return Some(func_id);
        }
        // 2. Check external_function_map
        if let Some(&func_id) = self.external_function_map.get(&symbol) {
            return Some(func_id);
        }
        // 3. Fallback: look up by qualified name
        if let Some(sym_info) = self.symbol_table.get_symbol(symbol) {
            if let Some(qn) = sym_info.qualified_name {
                if let Some(qualified) = self.string_interner.get(qn) {
                    if let Some(&func_id) = self.external_function_name_map.get(qualified) {
                        return Some(func_id);
                    }
                }
            }
            // 4. Try stdlib mapping by method name for @:coreType abstracts
            if let Some(method_name) = self.string_interner.get(sym_info.name) {
                // Check if this is a stdlib abstract (e.g., SIMD4f)
                let type_table = self.type_table;
                if let Some(ti) = type_table.get(target_type) {
                    if let TypeKind::Abstract { symbol_id, .. } = &ti.kind {
                        if let Some(abs_sym) = self.symbol_table.get_symbol(*symbol_id) {
                            let class_name = abs_sym
                                .native_name
                                .and_then(|nn| self.string_interner.get(nn))
                                .map(|n| n.replace("::", "_"))
                                .or_else(|| {
                                    self.string_interner
                                        .get(abs_sym.name)
                                        .map(|n| format!("rayzor_{}", n))
                                });
                            if let Some(class) = class_name {
                                // Try stdlib mapping (e.g., rayzor_SIMD4f + fromArray)
                                if let Some((_, mapping)) =
                                    self.stdlib_mapping.find_by_name(&class, method_name)
                                {
                                    let runtime_func = mapping.runtime_name;
                                    let result_type = self.convert_type(target_type);
                                    if mapping.is_mir_wrapper {
                                        // MIR wrapper — register as forward ref
                                        let func_id = self.register_stdlib_mir_forward_ref(
                                            runtime_func,
                                            vec![IrType::Ptr(Box::new(IrType::Void))],
                                            result_type,
                                        );
                                        return Some(func_id);
                                    } else {
                                        let func_id = self.get_or_register_extern_function(
                                            runtime_func,
                                            vec![IrType::Ptr(Box::new(IrType::Void))],
                                            result_type,
                                        );
                                        return Some(func_id);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Resolve a TypeId to the abstract's qualified name, for looking up @:from/@:to rules.
    /// Uses qualified_name for user-defined abstracts, native_name for extern abstracts (e.g., SIMD4f).
    /// Returns None if the TypeId is not an abstract type.
    pub(crate) fn resolve_abstract_name(&self, type_id: TypeId) -> Option<InternedString> {
        let type_table = self.type_table;
        if let Some(ti) = type_table.get(type_id) {
            if let TypeKind::Abstract { symbol_id, .. } = &ti.kind {
                if let Some(sym) = self.symbol_table.get_symbol(*symbol_id) {
                    return sym.qualified_name.or(sym.native_name).or(Some(sym.name));
                }
            }
        }
        None
    }

    pub(crate) fn resolve_cross_module_constructor_by_name(
        &self,
        actual_symbol_id: Option<SymbolId>,
        final_class_name: Option<&str>,
    ) -> Option<IrFunctionId> {
        let key = self.cross_module_constructor_fqn_key(actual_symbol_id, final_class_name)?;
        self.external_function_name_map.get(&key).copied()
    }

    /// Get the effective IR type for an expression, checking symbol_ir_types for pattern-bound variables.
    pub(crate) fn resolve_expr_ir_type(&self, expr: &HirExpr, default_type: IrType) -> IrType {
        if let HirExprKind::Variable { symbol, .. } = &expr.kind {
            if let Some(resolved) = self.symbol_ir_types.get(symbol) {
                return resolved.clone();
            }
            // Erased default (`Dynamic` converts to Ptr(Void)): the
            // variable's bound register carries the type its producer
            // actually emitted (e.g. `string_concat` → Ptr(String)).
            // Register evidence beats the erased HIR type; keep the
            // default when the register is itself untyped/void.
            if matches!(&default_type, IrType::Ptr(inner) if matches!(inner.as_ref(), IrType::Void))
            {
                if let Some(&reg) = self.symbol_map.get(symbol) {
                    if let Some(reg_ty) = self.builder.get_register_type(reg) {
                        if !matches!(&reg_ty, IrType::Ptr(inner) if matches!(inner.as_ref(), IrType::Void))
                        {
                            return reg_ty;
                        }
                    }
                }
            }
        }
        default_type
    }

    /// Get the effective TypeId for an expression, checking symbol_type_ids for pattern-bound variables.
    pub(crate) fn resolve_expr_type_id(&self, expr: &HirExpr) -> TypeId {
        if let HirExprKind::Variable { symbol, .. } = &expr.kind {
            if let Some(resolved) = self.symbol_type_ids.get(symbol) {
                return *resolved;
            }
        }
        expr.ty
    }

    /// Resolve a field index by name, disambiguating by receiver type when multiple classes
    /// have the same field name (e.g., both StringBuf.length and List.length).
    /// Returns (class_type_id, field_index) or None if not found.
    pub(crate) fn resolve_field_index_by_name(
        &mut self,
        field_name: InternedString,
        receiver_ty: TypeId,
    ) -> Option<(TypeId, u32)> {
        match self.resolve_field_index_candidates(field_name, receiver_ty) {
            FieldIndexResolution::None => None,
            FieldIndexResolution::Unique(class_ty, idx) => Some((class_ty, idx)),
            FieldIndexResolution::Ambiguous(matches) => {
                // Candidates disagree on the slot and the receiver's class is
                // unresolvable — any pick would silently read the wrong slot.
                let target_name_str = self
                    .string_interner
                    .get(field_name)
                    .unwrap_or("<unknown>")
                    .to_string();
                let owners = matches
                    .iter()
                    .map(|&(class_ty, idx)| {
                        let type_table = self.type_table;
                        let name = self
                            .resolve_type_class_name_with(&type_table, class_ty)
                            .unwrap_or_else(|| "<unknown class>".to_string());
                        format!("{} (slot {})", name, idx)
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                let loc = self
                    .symbol_table
                    .find_symbols(|s| s.name == field_name)
                    .first()
                    .map(|s| s.definition_location)
                    .unwrap_or(SourceLocation::unknown());
                self.add_error(
                    &format!(
                        "E0803: ambiguous field access: `{}` exists on multiple classes ({}) and the receiver's class could not be resolved. Annotate the receiver's type",
                        target_name_str, owners
                    ),
                    loc,
                );
                None
            }
        }
    }

    /// Candidate resolution behind `resolve_field_index_by_name`, with no
    /// error reporting — usable as a pure existence PROBE (does any class
    /// declare this field?) where ambiguity must not fail the compile.
    pub(crate) fn resolve_field_index_candidates(
        &self,
        field_name: InternedString,
        receiver_ty: TypeId,
    ) -> FieldIndexResolution {
        let mut all_matches: Vec<(TypeId, u32)> = Vec::new();
        let mut match_owners: Vec<Option<&str>> = Vec::new();
        let target_name_str = self
            .string_interner
            .get(field_name)
            .unwrap_or("<unknown>")
            .to_string();
        for (_sym, &(class_ty, idx)) in &self.field_index_map {
            if let Some(sym_info) = self.symbol_table.get_symbol(*_sym) {
                // Compare by InternedString ID first (fast), then by string content (handles
                // cross-context InternedString mismatches where same string gets different IDs)
                let id_match = sym_info.name == field_name;
                let str_match = {
                    let a = self.string_interner.get(sym_info.name);
                    let b = self.string_interner.get(field_name);
                    a.is_some() && a == b
                };
                if id_match || str_match {
                    all_matches.push((class_ty, idx));
                    match_owners.push(self.field_class_names.get(_sym).map(|s| s.as_str()));
                }
            } else {
                debug!(
                    "[RESOLVE_FIELD] symbol_table.get_symbol({:?}) returned None (looking for '{}')",
                    _sym, target_name_str
                );
            }
        }

        if all_matches.is_empty() {
            debug!(
                "[RESOLVE_FIELD] No matches found for '{}' in {} field_index_map entries",
                target_name_str,
                self.field_index_map.len()
            );
            return FieldIndexResolution::None;
        }

        if all_matches.len() == 1 {
            // Even with a single match, verify it belongs to the receiver's class.
            // Without this check, String.length could incorrectly match StringBuf.length.
            let (match_class_ty, match_idx) = all_matches[0];
            if match_class_ty == receiver_ty {
                return FieldIndexResolution::Unique(match_class_ty, match_idx);
            }
            // Fall through to disambiguation logic which checks type chains
        }

        // Multiple classes have this field name — disambiguate by receiver type.
        //
        // Strategy 0: owner-symbol match. Candidate class TypeIds come from the
        // MODULE THAT DECLARED the field, and TypeIds are context-local — a
        // candidate's class_ty can numerically equal an unrelated class's
        // TypeId in THIS context (`loop.finishReason` on a GenerationLoop
        // receiver matched ChatResponse's entry that way and read the eos Int
        // slot as a String). Field OWNER names are recorded per declaration
        // symbol, which is session-global, so a single candidate whose owner
        // is the receiver's class is the declaration itself and beats any
        // TypeId comparison. Positive selection only: with zero or several
        // owner matches (inherited fields carry the PARENT's owner name and
        // must keep resolving through the type chain), fall through unchanged.
        let recv_bare = {
            let type_table = self.type_table;
            let mut resolved = self.resolve_through_aliases(receiver_ty);
            let mut visited = BTreeSet::new();
            loop {
                if !visited.insert(resolved) {
                    break;
                }
                match type_table.get(resolved).map(|ti| &ti.kind) {
                    Some(TypeKind::GenericInstance { base_type, .. }) => resolved = *base_type,
                    _ => break,
                }
            }
            type_table
                .get(resolved)
                .and_then(|ti| match &ti.kind {
                    TypeKind::Class { symbol_id, .. } => Some(*symbol_id),
                    _ => None,
                })
                .and_then(|sym| self.symbol_table.get_symbol(sym))
                .and_then(|si| self.string_interner.get(si.name))
        };
        {
            if let Some(recv_bare) = recv_bare {
                let mut owned: Vec<(TypeId, u32)> = all_matches
                    .iter()
                    .zip(match_owners.iter())
                    .filter(|(_, owner)| {
                        owner.is_some_and(|o| Self::class_names_match(o, recv_bare))
                    })
                    .map(|(&m, _)| m)
                    .collect();
                owned.sort_unstable_by_key(|&(t, i)| (t.as_raw(), i));
                owned.dedup();
                if owned.len() == 1 {
                    return FieldIndexResolution::Unique(owned[0].0, owned[0].1);
                }
            }
        }

        // Strategy 1: Resolve receiver_ty through TypeAlias/GenericInstance chains to find
        // the underlying class TypeId, then match directly against candidates' class_ty.
        {
            // Resolve receiver_ty through TypeAlias/GenericInstance/Placeholder chains
            // to find the underlying Class type and its symbol_id
            let mut resolved = self.resolve_through_aliases(receiver_ty);
            let type_table = self.type_table;
            // Also follow GenericInstance to base type
            let mut visited = BTreeSet::new();
            loop {
                if !visited.insert(resolved) {
                    break;
                }
                if let Some(ti) = type_table.get(resolved) {
                    match &ti.kind {
                        TypeKind::GenericInstance { base_type, .. } => resolved = *base_type,
                        _ => break,
                    }
                } else {
                    break;
                }
            }

            // Check if resolved type directly matches a candidate's class_type_id
            for &(class_ty, idx) in &all_matches {
                if class_ty == resolved {
                    return FieldIndexResolution::Unique(class_ty, idx);
                }
            }

            // Extract class symbol_id from resolved type (handles Class with type_args)
            let receiver_symbol = type_table.get(resolved).and_then(|ti| match &ti.kind {
                TypeKind::Class { symbol_id, .. } => Some(*symbol_id),
                _ => None,
            });

            // Match by class symbol_id using the class_type_to_symbol mapping.
            // This survives type_table overwrites where class TypeIds get reused —
            // but the map is keyed by context-local TypeIds and also holds entries
            // punned from SymbolIds, so a hit can be a raw-id coincidence. Where
            // the candidate records its declaring class, require that NAME to
            // agree; a disagreement means we hit another class's entry, and
            // returning here would preempt the name-based Strategy 2 below with a
            // wrong GEP index.
            if let Some(recv_sym) = receiver_symbol {
                for (&(class_ty, idx), owner) in all_matches.iter().zip(match_owners.iter()) {
                    let candidate_sym = self.class_type_to_symbol.get(&class_ty);
                    if let Some(&csym) = candidate_sym {
                        if csym != recv_sym {
                            continue;
                        }
                        if let (Some(owner), Some(recv_name)) = (owner, recv_bare) {
                            if !Self::class_names_match(owner, recv_name) {
                                continue;
                            }
                        }
                        return FieldIndexResolution::Unique(class_ty, idx);
                    }
                }
            }
        }

        // Strategy 2: Resolve receiver to class name, match against candidate class names
        let receiver_class_name = {
            let type_table = self.type_table;
            self.resolve_type_class_name_with(&type_table, receiver_ty)
        };

        if let Some(ref recv_name) = receiver_class_name {
            for &(class_ty, idx) in &all_matches {
                let candidate_name = {
                    let type_table = self.type_table;
                    self.resolve_type_class_name_with(&type_table, class_ty)
                };
                if candidate_name.as_ref() == Some(recv_name) {
                    return FieldIndexResolution::Unique(class_ty, idx);
                }
            }
        }

        // No verified match. We can't blindly return all_matches[0] — that
        // path lets stdlib types (e.g. `Array.length`) resolve to an
        // unrelated entry like `List.length` (registered when @:build pulls
        // List into the compilation), routing through GEP-based field
        // access and producing nonsensical MIR. But we also can't always
        // return None — generic returns (e.g. `Arc<T>::get()` returning T)
        // give a receiver_ty that doesn't match any concrete class, and
        // the legitimate user field would then be unreachable.
        //
        // Compromise: return None when the receiver is a known stdlib
        // type (Array/String/Map/etc.) where the caller will reach the
        // proper stdlib runtime dispatch. Otherwise fall back to the
        // first match — this preserves the old behaviour for user-class
        // field access via generic returns.
        let receiver_is_stdlib_property_target = self
            .type_table
            .get(receiver_ty)
            .map(|ti| {
                matches!(
                    ti.kind,
                    TypeKind::Array { .. } | TypeKind::String | TypeKind::Map { .. }
                )
            })
            .unwrap_or(false);
        if receiver_is_stdlib_property_target {
            return FieldIndexResolution::None;
        }

        // If the receiver resolves to an EXTERN class (registered via @:native
        // or an `extern class String { ... }` declaration), return None so the
        // caller falls through to stdlib runtime dispatch instead of
        // pretending a same-named user-class field is the right answer.
        // Without this, a cached `StringBuf.length` entry in field_index_map
        // (loaded from BLADE cache by a prior test) became the wrong-match
        // answer for `s.length` where s is `String` (extern Class), and the
        // GEP-based field access then errored with "Cannot access field
        // 'length': class not registered or field does not exist".
        let receiver_is_extern_class = self
            .type_table
            .get(receiver_ty)
            .and_then(|ti| match &ti.kind {
                TypeKind::Class { symbol_id, .. } => Some(*symbol_id),
                _ => None,
            })
            .and_then(|sym| self.symbol_table.get_symbol(sym))
            .map(|s| s.flags.contains(SymbolFlags::EXTERN))
            .unwrap_or(false);
        if receiver_is_extern_class {
            return FieldIndexResolution::None;
        }

        // A single surviving candidate may dispatch (generic returns
        // legitimately land here with a receiver_ty matching no concrete
        // class), and candidates that AGREE on the field index are the same
        // declaration re-registered across contexts (e.g. a typedef'd
        // anonymous struct — one anon TypeId per importing module). Only
        // candidates with DIFFERENT indexes are a bare-name guess across
        // unrelated classes — Ambiguous, never first-match (the pick would
        // silently read the wrong slot).
        let mut distinct_indexes: Vec<u32> = all_matches.iter().map(|&(_, idx)| idx).collect();
        distinct_indexes.sort_unstable();
        distinct_indexes.dedup();
        if distinct_indexes.len() > 1 {
            return FieldIndexResolution::Ambiguous(all_matches);
        }

        let (class_ty, idx) = all_matches[0];
        FieldIndexResolution::Unique(class_ty, idx)
    }

    pub(crate) fn resolve_interface_method_names(
        &self,
        interface_symbol: SymbolId,
    ) -> Option<Vec<InternedString>> {
        if let Some(v) = self.interface_method_names.get(&interface_symbol) {
            return Some(v.clone());
        }
        let name = self.symbol_table.get_symbol(interface_symbol)?.name;
        self.interface_method_names
            .iter()
            .find(|(k, _)| {
                self.symbol_table
                    .get_symbol(**k)
                    .map(|s| s.name == name)
                    .unwrap_or(false)
            })
            .map(|(_, v)| v.clone())
    }

    /// Resolve the return type for an interface method call.
    /// If the method symbol has a Function type in the type table, extract the return type.
    /// Otherwise fall back to convert_type(expr_ty).
    pub(crate) fn resolve_interface_method_return_type(
        &self,
        method_symbol: SymbolId,
        expr_ty: TypeId,
    ) -> IrType {
        self.resolve_interface_method_return_type_full(method_symbol, expr_ty)
            .0
    }

    /// Variant of [`resolve_interface_method_return_type`] that also
    /// returns the resolved concrete `TypeId` when re-resolution
    /// succeeded. The `TypeId` is `Some` iff we replaced a Ptr fallback
    /// with a concrete entry from `interface_method_return_types`. Used
    /// by call-site tracking to propagate the concrete type through
    /// subsequent `Let` bindings so downstream receivers don't fall
    /// into the Dynamic-dispatch path.
    pub(crate) fn resolve_interface_method_return_type_full(
        &self,
        method_symbol: SymbolId,
        expr_ty: TypeId,
    ) -> (IrType, Option<TypeId>) {
        use crate::tast::TypeKind;
        let fallback = self.convert_type(expr_ty);
        // Compare TypeKinds, not IrTypes — `Array<Int>`, `Class`, and
        // `Dynamic` all convert to `Ptr(Void)` so an IrType test would
        // discard every reference-typed iface return. Look at the
        // call-site `expr_ty`'s TypeKind: if it's already concrete
        // (not Dynamic / Placeholder / Unknown), the type checker
        // already knows what it is — return the fallback and skip
        // the map lookup.
        let expr_kind = self.type_table.get(expr_ty).map(|t| t.kind.clone());
        let needs_resolve = matches!(
            expr_kind,
            Some(TypeKind::Dynamic) | Some(TypeKind::Placeholder { .. }) | Some(TypeKind::Unknown)
        );
        if !needs_resolve {
            return (fallback, None);
        }
        // String-compare method names — entries loaded from a BLADE
        // cache may have been interned in a different session, so raw
        // InternedString ids won't match even when the names do.
        let method_name_str = self
            .symbol_table
            .get_symbol(method_symbol)
            .and_then(|s| self.string_interner.get(s.name));
        if let Some(method_name) = method_name_str {
            for ((_, name), &ret_type_id) in &self.interface_method_return_types {
                let entry_name = self.string_interner.get(*name);
                if entry_name != Some(method_name) {
                    continue;
                }
                // Accept the map entry whenever its TypeKind is more
                // specific than `Dynamic`. Map is authoritative for
                // cross-file return types; the only "no-op" case is
                // when the entry is itself Dynamic-shaped.
                let entry_kind = self.type_table.get(ret_type_id).map(|t| t.kind.clone());
                let is_concrete = !matches!(
                    entry_kind,
                    Some(TypeKind::Dynamic)
                        | Some(TypeKind::Placeholder { .. })
                        | Some(TypeKind::Unknown)
                        | None
                );
                if is_concrete {
                    return (self.convert_type(ret_type_id), Some(ret_type_id));
                }
            }
        }
        (fallback, None)
    }

    pub(crate) fn resolve_through_aliases(&self, type_id: TypeId) -> TypeId {
        let type_table = self.type_table;
        let mut current = type_id;
        let mut visited = BTreeSet::new();
        loop {
            if !visited.insert(current) {
                break;
            }
            match type_table.get(current).map(|t| &t.kind) {
                Some(TypeKind::TypeAlias { target_type, .. }) => current = *target_type,
                Some(TypeKind::Placeholder { name }) => {
                    // Try to resolve Placeholder by searching for a class OR a
                    // typedef with a matching name. A cross-module structural
                    // typedef (`typedef Loaded = { var m:Base; ... }`) decays
                    // to a Placeholder just like a class does; without the
                    // TypeAlias arm this loop only ever finds a same-named
                    // Class (if one coincidentally exists) or gives up,
                    // leaving field access on the typedef unable to recognise
                    // it as Anonymous and falling through to class-field GEP.
                    if let Some(name_str) = self.string_interner.get(*name) {
                        // Search for "rayzor.Bytes" → match class "Bytes" with qualified "rayzor.Bytes"
                        // Also try bare name: "rayzor.Bytes" → "Bytes"
                        let bare_name = name_str.rsplit('.').next().unwrap_or(name_str);
                        let mut found = None;
                        let mut found_alias_target = None;
                        for (tid, ti) in type_table.iter() {
                            match &ti.kind {
                                TypeKind::Class { symbol_id, .. } => {
                                    if let Some(sym) = self.symbol_table.get_symbol(*symbol_id) {
                                        let sym_name =
                                            self.string_interner.get(sym.name).unwrap_or("");
                                        let sym_qname = sym
                                            .qualified_name
                                            .and_then(|qn| self.string_interner.get(qn));
                                        let matches = sym_qname.map_or(false, |qn| qn == name_str)
                                            || sym_name == name_str
                                            || sym_name == bare_name
                                            || sym_qname.map_or(false, |qn| qn == bare_name);
                                        if matches {
                                            found = Some(tid);
                                            break;
                                        }
                                    }
                                }
                                TypeKind::TypeAlias {
                                    symbol_id,
                                    target_type,
                                    ..
                                } => {
                                    if let Some(sym) = self.symbol_table.get_symbol(*symbol_id) {
                                        let sym_name =
                                            self.string_interner.get(sym.name).unwrap_or("");
                                        let sym_qname = sym
                                            .qualified_name
                                            .and_then(|qn| self.string_interner.get(qn));
                                        let matches = sym_qname.map_or(false, |qn| qn == name_str)
                                            || sym_name == name_str
                                            || sym_name == bare_name
                                            || sym_qname.map_or(false, |qn| qn == bare_name);
                                        if matches && found_alias_target.is_none() {
                                            found_alias_target = Some(*target_type);
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                        if let Some(resolved_tid) = found {
                            current = resolved_tid;
                        } else if let Some(target_tid) = found_alias_target {
                            // Continue resolving through the typedef's target
                            // (e.g. straight to the Anonymous struct type).
                            current = target_tid;
                        } else {
                            break;
                        }
                    } else {
                        break;
                    }
                }
                _ => break,
            }
        }
        current
    }

    /// Resolve a TypeId to its stdlib class name (e.g., "Array", "String").
    /// Takes a borrowed type_table to avoid double-borrow issues.
    pub(crate) fn resolve_type_class_name_with(
        &self,
        type_table: &TypeTable,
        type_id: TypeId,
    ) -> Option<String> {
        let type_info = type_table.get(type_id)?;
        match &type_info.kind {
            TypeKind::Class { symbol_id, .. } | TypeKind::Abstract { symbol_id, .. } => {
                self.symbol_table.get_symbol(*symbol_id).and_then(|s| {
                    s.qualified_name
                        .or(Some(s.name))
                        .and_then(|n| self.string_interner.get(n))
                        .map(|s| s.replace(".", "_"))
                })
            }
            TypeKind::GenericInstance { base_type, .. } => {
                self.resolve_type_class_name_with(type_table, *base_type)
            }
            TypeKind::TypeAlias { target_type, .. } => {
                self.resolve_type_class_name_with(type_table, *target_type)
            }
            _ => None,
        }
    }

    /// Resolve a TypeParameter TypeId to its concrete type using the receiver's type arguments.
    /// For example, if expr.ty is TypeParameter(T) and receiver is Container<Float>,
    /// returns Some(Float_TypeId).
    pub(crate) fn resolve_type_param_from_receiver(
        &self,
        type_param_id: TypeId,
        receiver_type_id: TypeId,
    ) -> Option<TypeId> {
        let type_table = self.type_table;

        // Check if type_param_id is actually a TypeParameter
        let param_info = type_table.get(type_param_id)?;
        let param_symbol = match &param_info.kind {
            crate::tast::TypeKind::TypeParameter { symbol_id, .. } => *symbol_id,
            _ => return None,
        };

        // Get receiver's class info with concrete type_args
        let recv_info = type_table.get(receiver_type_id)?;
        let (class_symbol, concrete_args) = match &recv_info.kind {
            crate::tast::TypeKind::Class {
                symbol_id,
                type_args,
            } => (*symbol_id, type_args.clone()),
            crate::tast::TypeKind::GenericInstance {
                base_type,
                type_args,
                ..
            } => {
                // Unwrap GenericInstance to get the base class's symbol_id
                if let Some(base_info) = type_table.get(*base_type) {
                    match &base_info.kind {
                        crate::tast::TypeKind::Class { symbol_id, .. } => {
                            (*symbol_id, type_args.clone())
                        }
                        _ => return None,
                    }
                } else {
                    return None;
                }
            }
            _ => return None,
        };

        // Find the type parameter's name
        let param_name = self.symbol_table.get_symbol(param_symbol)?.name;

        // Find the class declaration in HIR to get type parameter order
        for (_tid, decl) in self.current_hir_types.iter() {
            if let crate::ir::hir::HirTypeDecl::Class(c) = decl {
                if c.symbol_id == class_symbol {
                    for (i, tp) in c.type_params.iter().enumerate() {
                        if tp.name == param_name {
                            return concrete_args.get(i).copied();
                        }
                    }
                }
            }
        }
        None
    }
}
