//! Heap allocation: `malloc`/`free` declarations and object sizing.

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
    /// Declare malloc function so we can call it during lowering for heap allocations
    pub(crate) fn declare_malloc(&mut self) {
        use crate::ir::modules::IrExternFunction;
        use crate::ir::{CallingConvention, IrFunctionSignature, IrId, IrParameter, IrType};
        use crate::tast::SymbolId;

        // Create malloc signature: fn malloc(size: u64) -> *u8
        let signature = IrFunctionSignature {
            parameters: vec![IrParameter {
                name: "size".to_string(),
                ty: IrType::U64,
                reg: IrId::new(0),
                by_ref: false,
            }],
            return_type: IrType::Ptr(Box::new(IrType::U8)),
            calling_convention: CallingConvention::C,
            can_throw: false,
            type_params: vec![],
            uses_sret: false,
        };

        // Allocate a function ID
        let func_id = self.builder.module.alloc_function_id();

        // Create extern function declaration (NOT a regular function!)
        // This is critical - extern functions go in extern_functions map
        // so Cranelift backend will link them to runtime symbols
        let extern_func = IrExternFunction {
            id: func_id,
            name: "malloc".to_string(),
            symbol_id: SymbolId::from_raw(u32::MAX - 1), // Dummy symbol ID for extern
            signature,
            source: "libc".to_string(),
        };

        self.builder
            .module
            .extern_functions
            .insert(func_id, extern_func);
    }

    /// Declare free function so we can call it during lowering for drop semantics
    pub(crate) fn declare_free(&mut self) {
        use crate::ir::modules::IrExternFunction;
        use crate::ir::{CallingConvention, IrFunctionSignature, IrId, IrParameter, IrType};
        use crate::tast::SymbolId;

        // Create free signature: fn free(ptr: *u8) -> void
        let signature = IrFunctionSignature {
            parameters: vec![IrParameter {
                name: "ptr".to_string(),
                ty: IrType::Ptr(Box::new(IrType::U8)),
                reg: IrId::new(0),
                by_ref: false,
            }],
            return_type: IrType::Void,
            calling_convention: CallingConvention::C,
            can_throw: false,
            type_params: vec![],
            uses_sret: false,
        };

        // Allocate a function ID
        let func_id = self.builder.module.alloc_function_id();

        // Create extern function declaration
        let extern_func = IrExternFunction {
            id: func_id,
            name: "free".to_string(),
            symbol_id: SymbolId::from_raw(u32::MAX - 2), // Dummy symbol ID for extern
            signature,
            source: "libc".to_string(),
        };

        self.builder
            .module
            .extern_functions
            .insert(func_id, extern_func);
    }

    /// Build a heap allocation (via malloc) for class instances
    /// This is used for class instances that may escape the current function
    pub(crate) fn build_heap_alloc(&mut self, size: u64) -> Option<IrId> {
        // Look up malloc in extern_functions (where declare_malloc adds it)
        let malloc_id = self
            .builder
            .module
            .extern_functions
            .iter()
            .find(|(_, f)| f.name == "malloc")
            .map(|(id, _)| *id)
            .or_else(|| {
                // Fallback: check external_function_name_map
                self.external_function_name_map.get("malloc").copied()
            })?;

        // Create size constant
        let size_reg = self.builder.build_const(IrValue::U64(size))?;

        // Call malloc - returns Ptr(U8)
        let ptr_u8_ty = IrType::Ptr(Box::new(IrType::U8));
        let obj_ptr =
            self.builder
                .build_call_direct(malloc_id, vec![size_reg], ptr_u8_ty.clone())?;

        // Zero-initialize: Haxe guarantees unassigned fields read as null,
        // but malloc returns dirty memory. Fields a constructor never
        // assigns (e.g. `public var registry:ArchRegistry;` with an empty
        // ctor) otherwise hold heap garbage that passes `!= null` checks —
        // latent on native (fresh pages happen to be zero), observed as an
        // OOB fault on wasm once startup allocations shifted the heap.
        // Use i64 GEPs (not PtrAdd) so ScalarReplacement keeps tracking the
        // allocation — SROA promotes through GEP/Copy/Cast only, and a
        // PtrAdd use would dangle after the malloc is elided.
        let zero = self.builder.build_const(IrValue::I64(0))?;
        let lanes = size / 8;
        for lane in 0..lanes {
            let idx = self.builder.build_const(IrValue::I32(lane as i32))?;
            let slot = self.builder.build_gep(obj_ptr, vec![idx], IrType::I64)?;
            self.builder.build_store(slot, zero)?;
        }
        Some(obj_ptr)
    }

    /// Expand a class instance's allocation size to cover its full inheritance
    /// chain. A subclass contains every field of its parent at the same slot
    /// indices, so its allocation must be at least as large as the parent's.
    ///
    /// `register_class_metadata` computes a class's own size from the fields it
    /// can see, but for an IMPORTED parent (e.g. `class MyException extends
    /// haxe.Exception`) `collect_inherited_fields` may not have the parent's
    /// fields available at registration time, leaving the subclass undersized.
    /// Inherited-field writes (notably the `stack` field populated at `throw`,
    /// resolved via `resolve_field_index_by_name` which DOES walk the parent
    /// chain) then write past the allocation and corrupt the heap. Evaluated at
    /// the `new` site — after all classes are registered — the parent's size is
    /// reliably available, so we take the max over the chain. A larger
    /// allocation is always safe (field indices are unchanged).
    pub(crate) fn alloc_size_with_inheritance(
        &self,
        class_type: TypeId,
        class_symbol: Option<SymbolId>,
        own_size: u64,
    ) -> u64 {
        let mut size = own_size;
        let mut cur_type = class_type;
        let mut cur_sym = class_symbol;
        let mut visited: BTreeSet<TypeId> = BTreeSet::new();
        visited.insert(cur_type);
        loop {
            // Find the parent (`extends`) of the current class via its HIR decl,
            // by type id first, then by symbol id. Carry the declaration's
            // `extends_symbol` alongside — it names the parent directly.
            let extends: Option<(Option<TypeId>, Option<SymbolId>)> = self
                .current_hir_types
                .get(&cur_type)
                .and_then(|d| match d {
                    HirTypeDecl::Class(c) => Some((c.extends, c.extends_symbol)),
                    _ => None,
                })
                .or_else(|| {
                    cur_sym.and_then(|s| {
                        self.current_hir_types.values().find_map(|d| match d {
                            HirTypeDecl::Class(c) if c.symbol_id == s => {
                                Some((c.extends, c.extends_symbol))
                            }
                            _ => None,
                        })
                    })
                });
            let Some((parent_type_opt, extends_symbol)) = extends else {
                break;
            };
            let Some(parent_type) = parent_type_opt else {
                break;
            };
            if !visited.insert(parent_type) {
                break;
            }
            let parent_sym = extends_symbol.or_else(|| {
                self.type_table
                    .get(parent_type)
                    .and_then(|t| match &t.kind {
                        crate::tast::TypeKind::Class { symbol_id, .. } => Some(*symbol_id),
                        _ => None,
                    })
            });
            // Parent's registered alloc size: by declaring SymbolId, then by
            // qualified/simple name. Both are stable across compilation
            // contexts; context-local TypeIds must never key this lookup.
            let parent_size = parent_sym
                .and_then(|ps| self.class_alloc_sizes.get(&ps).copied())
                .or_else(|| {
                    parent_sym
                        .and_then(|ps| self.symbol_table.get_symbol(ps))
                        .and_then(|s| {
                            s.qualified_name
                                .and_then(|n| self.string_interner.get(n))
                                .or_else(|| self.string_interner.get(s.name))
                        })
                        .and_then(|name| self.class_alloc_sizes_by_name.get(name).copied())
                });
            if let Some(ps) = parent_size {
                size = size.max(ps);
            }
            cur_type = parent_type;
            cur_sym = parent_sym;
        }
        size
    }
}
