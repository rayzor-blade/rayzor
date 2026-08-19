//! Vtable construction for classes.

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
    /// Build the vtable for a single class — inherits parent's slots and uses
    /// the most-derived implementation for each slot.
    pub(crate) fn build_vtable_for_class(&mut self, class_sym: SymbolId) {
        // Inherit parent's virtual slots
        let parent_slots = self
            .class_parent_map
            .get(&class_sym)
            .and_then(|parent| self.class_virtual_slots.get(parent))
            .cloned()
            .unwrap_or_default();

        let mut slots = parent_slots;

        // Add any new virtual slots defined at this class level
        if let Some(own_slots) = self.class_virtual_slots.get(&class_sym).cloned() {
            for (name, _) in own_slots {
                if !slots.iter().any(|(n, _)| *n == name) {
                    let idx = slots.len() as u32;
                    slots.push((name, idx));
                }
            }
        }

        if slots.is_empty() {
            return;
        }

        let mut vtable = Vec::new();
        for (method_name, _) in &slots {
            let method_sym = self
                .class_method_by_name
                .get(&(class_sym, *method_name))
                .copied()
                .or_else(|| {
                    // Walk parent chain to find inherited implementation
                    self.parent_chain(class_sym).into_iter().find_map(|parent| {
                        self.class_method_by_name
                            .get(&(parent, *method_name))
                            .copied()
                    })
                });

            if let Some(sym) = method_sym {
                vtable.push(sym);
            }
        }

        if !vtable.is_empty() {
            self.class_virtual_slots.insert(class_sym, slots);
            self.class_vtables.insert(class_sym, vtable);
        }
    }

    pub(crate) fn build_class_vtables(&mut self) {
        if self.override_methods.is_empty() {
            return;
        }

        // Step 1: For each override, trace up the parent chain to find the base class
        // that originally defines the method. Mark that base method as needing a virtual slot.
        let mut base_virtual_methods: BTreeSet<(SymbolId, InternedString)> = BTreeSet::new();

        let override_methods_snapshot: Vec<_> = self
            .override_methods
            .iter()
            .map(|(s, n)| (*s, *n))
            .collect();
        for (child_sym, method_name) in &override_methods_snapshot {
            let method_name = *method_name;
            let child_sym = *child_sym;
            let mut defining_class = None;
            for parent in self.parent_chain(child_sym) {
                if self
                    .class_method_by_name
                    .contains_key(&(parent, method_name))
                {
                    defining_class = Some(parent);
                }
            }
            if let Some(base) = defining_class {
                base_virtual_methods.insert((base, method_name));
            }
        }

        // Step 2: Assign virtual slots for each base class
        for (base_class, method_name) in &base_virtual_methods {
            let slots = self.class_virtual_slots.entry(*base_class).or_default();
            if !slots.iter().any(|(n, _)| *n == *method_name) {
                let slot_idx = slots.len() as u32;
                slots.push((*method_name, slot_idx));
            }
        }

        // Step 3: Build vtables for all classes in hierarchies with virtual methods.
        // Process topologically (parents before children).
        let mut classes_to_process: BTreeSet<SymbolId> = BTreeSet::new();
        for (base_class, _) in &base_virtual_methods {
            classes_to_process.insert(*base_class);
        }
        for (child_sym, _) in &override_methods_snapshot {
            classes_to_process.insert(*child_sym);
        }
        // Also include intermediate classes in the hierarchy
        for &(child, _) in &override_methods_snapshot {
            for parent in self.parent_chain(child) {
                classes_to_process.insert(parent);
            }
        }
        // Every subclass of a class with a vtable is included: a leaf that
        // inherits (but does not override) virtual methods still needs entries
        // for haxe_vtable_lookup to find it by type_id at runtime.
        let parent_map_snapshot: Vec<_> = self
            .class_parent_map
            .iter()
            .map(|(&c, &p)| (c, p))
            .collect();
        let mut changed = true;
        while changed {
            changed = false;
            for &(child, parent) in &parent_map_snapshot {
                if classes_to_process.contains(&parent) && !classes_to_process.contains(&child) {
                    classes_to_process.insert(child);
                    changed = true;
                }
            }
        }

        let mut processed: BTreeSet<SymbolId> = BTreeSet::new();
        let mut remaining: Vec<SymbolId> = classes_to_process.into_iter().collect();
        let max_iters = remaining.len() + 1;
        for _ in 0..max_iters {
            if remaining.is_empty() {
                break;
            }
            let mut next_remaining = Vec::new();
            for cls in remaining {
                let parent_ready = self
                    .class_parent_map
                    .get(&cls)
                    .map_or(true, |p| processed.contains(p));
                if parent_ready {
                    self.build_vtable_for_class(cls);
                    processed.insert(cls);
                } else {
                    next_remaining.push(cls);
                }
            }
            remaining = next_remaining;
        }
        for cls in remaining {
            self.build_vtable_for_class(cls);
        }

        // Step 4: Populate virtual_dispatch_info for call sites. Every method
        // SymbolId in a virtual hierarchy (base and overrides) maps to its slot,
        // so calls through base and derived types both dispatch via the vtable.
        let virtual_slots_snapshot: Vec<_> = self
            .class_virtual_slots
            .iter()
            .map(|(&s, v)| (s, v.clone()))
            .collect();
        for (class_sym, slots) in &virtual_slots_snapshot {
            for (method_name, slot_idx) in slots {
                if let Some(&method_sym) =
                    self.class_method_by_name.get(&(*class_sym, *method_name))
                {
                    self.virtual_dispatch_info
                        .insert(method_sym, (*slot_idx, *class_sym));
                }
            }
        }
    }

    /// Register (or reuse) a name-keyed forward reference to a vtable dispatch
    /// thunk, given the target method's fully-qualified name. Used when the
    /// method's class is imported and absent from this lowering context (no
    /// SymbolId, no compiled function yet — e.g. a `new C()` in a sibling file
    /// compiled before `C`), so neither the SymbolId-keyed maps nor
    /// `external_function_name_map` can resolve it. The thunk name is
    /// deterministic (`__vtable_dispatch_thunk__<sanitized_fqn>`) and identical
    /// to what `C`'s own context emits, so the empty stub dedupes with the real
    /// thunk at the LLVM declare/merge pass and the fat-pointer slot points at
    /// the real dispatcher regardless of compile order.
    pub(crate) fn forward_ref_dispatch_thunk_by_name(
        &mut self,
        method_fqn: &str,
    ) -> Option<IrFunctionId> {
        let sanitized: String = method_fqn
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { '_' })
            .collect();
        let thunk_name = format!("__vtable_dispatch_thunk__{}", sanitized);
        // Reuse if this context already has the (real or stub) thunk by name.
        if let Some(&existing) = self.external_function_name_map.get(&thunk_name) {
            return Some(existing);
        }
        for (func_id, func) in &self.builder.module.functions {
            if func.name == thunk_name {
                return Some(*func_id);
            }
        }
        // Minimal empty stub named exactly like the real thunk. Only its name
        // matters for the fat-ptr slot (which stores the function's address);
        // the real dispatcher's body and signature take over at merge dedup.
        // The permissive `(env, this) -> i64` Haxe-CC signature keeps the stub
        // self-consistent if it is ever called before the merge.
        let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
        let ptr_void = IrType::Ptr(Box::new(IrType::Void));
        let sig = FunctionSignatureBuilder::new()
            .param("env".to_string(), ptr_u8)
            .param("this".to_string(), ptr_void)
            .returns(IrType::I64)
            .calling_convention(CallingConvention::Haxe)
            .build();
        let stub_symbol = SymbolId::from_raw(u32::MAX - 3000 - self.next_wrapper_id);
        self.next_wrapper_id += 1;
        let saved_current_function = self.builder.current_function;
        let saved_current_block = self.builder.current_block;
        let thunk_id = self
            .builder
            .start_function(stub_symbol, thunk_name.clone(), sig);
        // The body stays empty (forward declaration); merge supplies the real one.
        self.check_move_flow();
        self.builder.finish_function();
        self.builder.current_function = saved_current_function;
        self.builder.current_block = saved_current_block;
        // Record by name so subsequent slots reuse this stub.
        self.external_function_name_map.insert(thunk_name, thunk_id);
        Some(thunk_id)
    }
}
