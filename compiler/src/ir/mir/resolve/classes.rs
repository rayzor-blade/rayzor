//! Resolves receiver classes, their methods, and runtime type ids.

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
    pub(crate) fn resolve_receiver_class_symbol(&self, type_id: TypeId) -> Option<SymbolId> {
        let mut current = type_id;
        {
            let type_table = self.type_table;
            let mut visited = BTreeSet::new();
            loop {
                if !visited.insert(current) {
                    break;
                }
                match type_table.get(current).map(|ti| &ti.kind) {
                    Some(TypeKind::Class { symbol_id, .. }) => return Some(*symbol_id),
                    Some(TypeKind::TypeAlias { target_type, .. }) => {
                        current = *target_type;
                    }
                    Some(TypeKind::GenericInstance { base_type, .. }) => {
                        current = *base_type;
                    }
                    _ => break,
                }
            }
        }

        self.class_type_to_symbol
            .get(&current)
            .copied()
            .or_else(|| self.class_type_to_symbol.get(&type_id).copied())
    }

    /// Resolve the canonical runtime class TypeId used by MIR type metadata.
    /// TAST class TypeIds can differ from HIR/MIR TypeIds; runtime RTTI registration
    /// is keyed by MIR TypeId, so we map through class_symbol when needed.
    pub(crate) fn resolve_runtime_class_type_id(
        &self,
        tast_type_id: TypeId,
        class_symbol: SymbolId,
    ) -> TypeId {
        let has_mir_type = |tid: TypeId| {
            self.builder
                .module
                .types
                .values()
                .any(|typedef| typedef.type_id == tid)
        };

        if self.class_type_to_symbol.get(&tast_type_id) == Some(&class_symbol)
            && has_mir_type(tast_type_id)
        {
            return tast_type_id;
        }

        let mut best: Option<(u8, TypeId)> = None;
        for (candidate_type_id, sym) in &self.class_type_to_symbol {
            if *sym != class_symbol {
                continue;
            }

            // Prefer TypeIds that are known canonical class IDs:
            // 1) Present as a typedef in the current MIR module
            // 2) Registered via register_class_metadata (every entry in
            //    class_type_to_symbol is — this loop iterates that map, so the
            //    check is constant; kept as a distinct tier for the tie-break).
            let score = if has_mir_type(*candidate_type_id) {
                3
            } else {
                2
            };

            match best {
                Some((best_score, best_tid)) => {
                    if score > best_score
                        || (score == best_score && candidate_type_id.as_raw() < best_tid.as_raw())
                    {
                        best = Some((score, *candidate_type_id));
                    }
                }
                None => best = Some((score, *candidate_type_id)),
            }
        }

        if let Some((_, tid)) = best {
            return tid;
        }

        // Last-resort fallback: match class short name to MIR typedef name.
        // This handles symbol-id drift between TAST/HIR when class_type_to_symbol
        // has not been populated for the current symbol-id variant.
        if let Some(class_name) = self
            .symbol_table
            .get_symbol(class_symbol)
            .and_then(|s| self.string_interner.get(s.name))
        {
            let mut by_name: Option<TypeId> = None;
            for typedef in self.builder.module.types.values() {
                if typedef.name == class_name {
                    by_name = Some(typedef.type_id);
                    break;
                }
            }
            if let Some(tid) = by_name {
                return tid;
            }
        }

        tast_type_id
    }

    /// Resolve a method symbol by (class, method_name), walking parent classes.
    pub(crate) fn resolve_class_method_symbol(
        &self,
        class_symbol: SymbolId,
        method_name: InternedString,
    ) -> Option<SymbolId> {
        let mut current = Some(class_symbol);
        let mut visited = BTreeSet::new();
        while let Some(cls) = current {
            if !visited.insert(cls) {
                break;
            }
            if let Some(method_symbol) = self
                .class_method_symbols
                .get(&(cls, method_name))
                .copied()
                .or_else(|| self.class_method_by_name.get(&(cls, method_name)).copied())
            {
                return Some(method_symbol);
            }
            current = self.class_parent_map.get(&cls).copied();
        }
        None
    }

    pub(crate) fn get_return_class_hint<'b>(dispatching_class: &'b str, method: &str) -> &'b str {
        match (dispatching_class, method) {
            // Mutex.lock() and Mutex.tryLock() return MutexGuard
            (c, "lock" | "tryLock") if c.contains("Mutex") && !c.contains("MutexGuard") => {
                "rayzor_concurrent_MutexGuard"
            }
            // Array.iterator() returns ArrayIterator
            ("Array", "iterator") => "ArrayIterator",
            // Array.keyValueIterator() returns ArrayKeyValueIterator
            ("Array", "keyValueIterator") => "ArrayKeyValueIterator",
            // Default: return value is associated with the same class
            _ => dispatching_class,
        }
    }
}
