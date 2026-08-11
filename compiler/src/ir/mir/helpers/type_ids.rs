//! Deterministic runtime type ids for classes, interfaces and enums.

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
    /// Map a compiler TypeId to the runtime's type_id constant.
    /// Used for anonymous object shape descriptors.
    /// Runtime type IDs: 0=Void, 1=Null, 2=Bool, 3=Int, 4=Float, 5=String
    pub(crate) fn runtime_type_id(&self, type_id: TypeId) -> u32 {
        use crate::tast::TypeKind;
        let type_table = self.type_table;
        match type_table.get(type_id).map(|t| &t.kind) {
            Some(TypeKind::Void) => 0,
            Some(TypeKind::Bool) => 2,
            Some(TypeKind::Int) => 3,
            Some(TypeKind::Float) => 4,
            Some(TypeKind::String) => 5,
            Some(TypeKind::Dynamic) => 5, // Dynamic matches anything
            Some(TypeKind::Class { symbol_id, .. }) => {
                // Class runtime IDs are derived from the qualified class
                // name so they're identical across compilation sessions
                // regardless of import order. Without this, `symbol_id + 1000`
                // was the type_id — but symbol_ids depend on which file gets
                // loaded first, so the cached MIR for StringTools (with
                // StringBuf's type_id baked in as `Const I64(_)`) would have
                // a different value depending on whether test_file_readall
                // or test_stringbuf_stringtools compiled StringBuf first.
                // Loading that cached MIR in a different session left the
                // wrong type_id in object headers → instanceof/cast/vtable
                // lookups missed → SIGILL.
                self.deterministic_class_type_id(*symbol_id)
                    .unwrap_or_else(|| {
                        // Fallback for classes without a resolvable name —
                        // legacy raw-id behaviour. NOTE: not stable across
                        // runs/contexts; a location-hash replacement was
                        // attempted 2026-06-13 but broke clone-identity
                        // (ids disagreed with another id source) — needs a
                        // unified fix with resolve_runtime_class_type_id.
                        self.resolve_runtime_class_type_id(type_id, *symbol_id)
                            .as_raw()
                            + 1000
                    })
            }
            Some(TypeKind::Interface { symbol_id, .. }) => self
                .deterministic_iface_or_enum_type_id(*symbol_id, "iface")
                .unwrap_or_else(|| type_id.as_raw()),
            Some(TypeKind::Enum { symbol_id, .. }) => self
                .deterministic_iface_or_enum_type_id(*symbol_id, "enum")
                .unwrap_or_else(|| type_id.as_raw() + 1000),
            Some(TypeKind::TypeAlias { target_type, .. }) => {
                let target = *target_type;
                self.runtime_type_id(target)
            }
            _ => 0, // default to void/unknown
        }
    }

    /// FNV-1a 32-bit hash over the qualified name of a class symbol,
    /// biased into [1000, u32::MAX) so it never collides with the
    /// primitive `runtime_type_id` slots (0=Void, 2=Bool, 3=Int, 4=Float,
    /// 5=String). Returns None when the symbol has no resolvable name.
    pub(crate) fn deterministic_class_type_id(&self, symbol_id: SymbolId) -> Option<u32> {
        let sym = self.symbol_table.get_symbol(symbol_id)?;
        // Prefer qualified_name; fall back to bare name. The bare-name
        // fallback may collide across packages — acceptable in practice
        // since the legacy `symbol_id + 1000` also lacked package isolation.
        let qname = sym
            .qualified_name
            .and_then(|n| self.string_interner.get(n))
            .or_else(|| self.string_interner.get(sym.name))?;
        Some(Self::fnv1a_class_type_id(qname))
    }

    pub(crate) fn deterministic_iface_or_enum_type_id(
        &self,
        symbol_id: SymbolId,
        kind_tag: &str,
    ) -> Option<u32> {
        let sym = self.symbol_table.get_symbol(symbol_id)?;
        let qname = sym
            .qualified_name
            .and_then(|n| self.string_interner.get(n))
            .or_else(|| self.string_interner.get(sym.name))?;
        // Tag with kind so a class and an enum with the same qualified
        // name can't collide. Output lives in the same biased range as
        // class ids.
        let key = format!("{}:{}", kind_tag, qname);
        Some(Self::fnv1a_class_type_id(&key))
    }

    pub(crate) fn fnv1a_class_type_id(s: &str) -> u32 {
        // FNV-1a 32-bit.
        let mut hash: u32 = 0x811c9dc5;
        for b in s.as_bytes() {
            hash ^= *b as u32;
            hash = hash.wrapping_mul(0x01000193);
        }
        // Bias above primitive type_ids (0..=5) and the legacy +1000 floor
        // so cached MIR that does `runtime_type_id - 1000` and stores the
        // result in the object header still gets a positive distinguishable
        // value. Lower 12 bits get reserved for the bias floor.
        const BIAS: u32 = 0x1000_0000;
        BIAS | (hash & 0x0fff_ffff)
    }

    /// Deterministic fallback when the symbol has no resolvable name:
    /// hash the definition location instead. line/column/byte_offset are
    /// stable across runs (file_id is assignment-order-dependent, so it's
    /// excluded). The legacy raw-id fallback varied per run — observed as
    /// per-run `__reflect_ctor_wrap_<N>` names and run-to-run wasm heap
    /// layout shifts.
    #[allow(dead_code)]
    pub(crate) fn location_fallback_type_id(
        &self,
        symbol_id: SymbolId,
        kind_tag: &str,
    ) -> Option<u32> {
        let sym = self.symbol_table.get_symbol(symbol_id)?;
        let loc = sym.definition_location;
        if loc.line == 0 && loc.column == 0 && loc.byte_offset == 0 {
            return None;
        }
        let key = format!(
            "{}@{}:{}:{}:{}",
            kind_tag, loc.file_id, loc.line, loc.column, loc.byte_offset
        );
        Some(Self::fnv1a_class_type_id(&key))
    }

    pub(crate) fn is_value_type_type_id(&self, type_id: TypeId) -> bool {
        let resolved = self.resolve_through_aliases(type_id);
        let type_table = self.type_table;
        let Some(type_info) = type_table.get(resolved) else {
            return false;
        };
        let TypeKind::Enum { symbol_id, .. } = type_info.kind else {
            return false;
        };

        let Some(sym) = self.symbol_table.get_symbol(symbol_id) else {
            return false;
        };
        let name = self.string_interner.get(sym.name);
        if name == Some("ValueType") {
            return true;
        }
        if let Some(qn) = sym
            .qualified_name
            .and_then(|qn| self.string_interner.get(qn))
        {
            return qn == "ValueType" || qn.ends_with(".ValueType");
        }
        false
    }

    pub(crate) fn expr_is_value_type_expr(&self, expr: &HirExpr) -> bool {
        if self.trace_typeof_inner_arg(expr).is_some() {
            return true;
        }

        match &expr.kind {
            HirExprKind::Cast { expr: inner, .. } => self.expr_is_value_type_expr(inner),
            HirExprKind::Variable { symbol, .. } => {
                self.value_type_symbols.contains(symbol) || self.is_value_type_type_id(expr.ty)
            }
            _ => self.is_value_type_type_id(expr.ty),
        }
    }
}
