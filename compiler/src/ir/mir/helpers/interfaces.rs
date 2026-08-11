//! Interface fat pointers: wrapping, cloning, constraint queries.

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
    /// Wrap a class instance in an interface fat pointer.
    /// Fat pointer layout: { object_ptr: i64, fn_ptr_0: i64, fn_ptr_1: i64, ... }
    /// One slot per interface method, in the interface's method order.
    pub(crate) fn wrap_in_interface_fat_ptr(
        &mut self,
        obj_reg: IrId,
        class_symbol: SymbolId,
        interface_symbol: SymbolId,
    ) -> Option<IrId> {
        // Resolve the method SymbolId vtable. Three tiers:
        //  (1) eager (class, iface) vtable from register_class_metadata;
        //  (2) lazy build from `class_method_symbols`/`class_method_by_name`;
        //  (3) name-only: when neither the vtable nor the class's method
        //      symbols are in THIS context (a fully-imported class the
        //      importer never lowered — e.g. `new LlamaArch()` in
        //      `ArchRegistry.withDefaults`), fall back to just the interface
        //      method NAMES. We can't fill method-symbol slots, but the slot
        //      loop resolves each slot's function by `<class_fqn>.<method>`
        //      via the stable name-keyed `external_function_name_map`, so the
        //      symbols aren't actually needed.
        // Length the call sites will index against (drift-tolerant). A cached
        // eager vtable that is SHORTER than this (a method compacted out in
        // register_class_metadata, or built before a base interface's methods
        // were merged in) would place every slot at the wrong offset — the call
        // site reads past the fat pointer. Reject it and rebuild position-
        // preserving below.
        let expected_method_count = self
            .resolve_interface_method_names(interface_symbol)
            .map(|n| n.len());
        let vtable: Vec<Option<SymbolId>> = self
            .interface_vtables
            .get(&(class_symbol, interface_symbol))
            .cloned()
            .filter(|syms| expected_method_count.map_or(true, |n| syms.len() == n))
            .map(|syms| syms.into_iter().map(Some).collect::<Vec<_>>())
            .or_else(|| {
                // Tier 2/3: driven by interface_method_names (forwarded and
                // stable). Resolve each method's SymbolId if available; leave
                // None otherwise (name-based resolution handles those slots).
                let method_names = self.resolve_interface_method_names(interface_symbol)?;
                let mut all_resolved = true;
                let entries: Vec<Option<SymbolId>> = method_names
                    .iter()
                    .map(|mname| {
                        let s = self
                            .class_method_symbols
                            .get(&(class_symbol, *mname))
                            .copied()
                            .or_else(|| {
                                self.class_method_by_name
                                    .get(&(class_symbol, *mname))
                                    .copied()
                            });
                        if s.is_none() {
                            all_resolved = false;
                        }
                        s
                    })
                    .collect();
                // Cache only fully-resolved vtables (partial ones would poison
                // the O(1) fast path with wrong slots).
                if all_resolved {
                    self.interface_vtables.insert(
                        (class_symbol, interface_symbol),
                        entries.iter().map(|e| e.unwrap()).collect(),
                    );
                }
                Some(entries)
            })?;
        let method_count = vtable.len();
        let fat_ptr_size = ((1 + method_count) * 8) as u64; // object_ptr + N function pointers
                                                            // Allocate fat pointer with malloc so IrInstruction::Free
                                                            // (lowered to libc free) uses matching allocator semantics.
        let malloc_fn = self.get_or_register_extern_function(
            "malloc",
            vec![IrType::U64],
            IrType::Ptr(Box::new(IrType::U8)),
        );
        let size_reg = self.builder.build_const(IrValue::U64(fat_ptr_size))?;
        let fat_ptr = self.builder.build_call_direct(
            malloc_fn,
            vec![size_reg],
            IrType::Ptr(Box::new(IrType::U8)),
        )?;

        // Store object pointer at offset 0
        // Need to bitcast obj_reg to i64 if it's a pointer
        let obj_as_i64 = {
            let obj_ty = self
                .builder
                .get_register_type(obj_reg)
                .unwrap_or(IrType::I64);
            if matches!(obj_ty, IrType::Ptr(_)) {
                self.builder.build_bitcast(obj_reg, IrType::I64)?
            } else {
                obj_reg
            }
        };
        self.builder.build_store(fat_ptr, obj_as_i64);

        // Class FQN (used for name-based cross-module method resolution below).
        let class_fqn: Option<String> = self.symbol_table.get_symbol(class_symbol).and_then(|s| {
            s.qualified_name
                .and_then(|n| self.string_interner.get(n))
                .or_else(|| self.string_interner.get(s.name))
                .map(|s| s.to_string())
        });
        // Interface method NAMES in vtable order — the stable, drift-proof
        // handle for name-based resolution (SymbolIds drift across contexts).
        let iface_method_names: Vec<Option<String>> = self
            .resolve_interface_method_names(interface_symbol)
            .map(|names| {
                names
                    .iter()
                    .map(|n| self.string_interner.get(*n).map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();
        // Store function pointers for each interface method.
        // Cross-file vtables (the class lives in one file, the constructor
        // call lives in another) need to find the method's IrFunctionId in
        // `external_function_map` — the per-context `function_map` only has
        // functions lowered in *this* file.
        for (i, method_sym_opt) in vtable.iter().enumerate() {
            // Resolve this slot's function IrFunctionId. Prefer the SymbolId
            // path (when the method symbol resolved), then fall back to the
            // FQN-keyed name lookup for imported classes whose method SymbolIds
            // aren't present in this context.
            let func_id_opt = method_sym_opt
                .and_then(|method_sym| {
                    self.function_map
                        .get(&method_sym)
                        .copied()
                        .or_else(|| self.external_function_map.get(&method_sym).copied())
                })
                // FQN-keyed fallback: cross-module, the method SymbolId drifts
                // or is absent, so `function_map`/`external_function_map`
                // (SymbolId-keyed) miss an imported class's method. Resolve by
                // constructing `<class_fqn>.<method_name>` and looking it up in
                // the stable name-keyed `external_function_name_map`.
                .or_else(|| {
                    let fqn = class_fqn.as_ref()?;
                    let mname = iface_method_names.get(i)?.as_ref()?;
                    let key = format!("{}.{}", fqn, mname);
                    self.external_function_name_map.get(&key).copied()
                });
            if std::env::var_os("RAYZOR_IFACE_DIAG").is_some() {
                let cn = class_fqn.clone().unwrap_or_default();
                let mn = iface_method_names
                    .get(i)
                    .and_then(|o| o.clone())
                    .unwrap_or_default();
                eprintln!(
                    "[fatptr] class={} slots={} slot={} method={} resolved={}",
                    cn,
                    method_count,
                    i,
                    mn,
                    func_id_opt.is_some()
                );
            }
            let dispatch_func_id = match func_id_opt {
                // Slots MUST hold dispatch thunks, never raw methods: the
                // CallIndirect lowering on every backend uses the closure
                // ABI (env prepended), so a raw `(this, args)` method in a
                // slot receives `(this=env, args=this, …)` — shifted args.
                // Natively this was masked by devirtualization; on wasm it
                // surfaced as iface methods silently reading garbage
                // (Tokenizer.encode returning [] in llama-chat).
                Some(func_id) => self
                    .ensure_vtable_dispatch_thunk(func_id)
                    .or_else(|| {
                        method_sym_opt
                            .and_then(|ms| self.ensure_cross_module_dispatch_thunk(ms, func_id))
                    })
                    .unwrap_or(func_id),
                None => {
                    let key = match (
                        class_fqn.as_ref(),
                        iface_method_names.get(i).and_then(|o| o.as_ref()),
                    ) {
                        (Some(fqn), Some(mname)) => format!("{}.{}", fqn, mname),
                        // No FQN and no symbol: nothing addresses this method.
                        // Refuse the wrapper rather than booby-trap the slot.
                        _ => return None,
                    };
                    self.forward_ref_dispatch_thunk_by_name(&key)?
                }
            };
            let fn_ref = self.builder.build_function_ref(dispatch_func_id)?;
            let offset_val = self
                .builder
                .build_const(IrValue::I64(((i + 1) * 8) as i64))?;
            let slot_ptr = self.builder.build_ptr_add(
                fat_ptr,
                offset_val,
                IrType::Ptr(Box::new(IrType::U8)),
            )?;
            self.builder.build_store(slot_ptr, fn_ref);
        }

        Some(fat_ptr)
    }

    /// Clone an interface fat pointer when assigning interface -> interface.
    /// This prevents multiple variables from aliasing the same wrapper allocation,
    /// so source reassignment can free only the source wrapper without invalidating
    /// other interface variables.
    pub(crate) fn clone_interface_fat_ptr(
        &mut self,
        fat_ptr_reg: IrId,
        source_interface: SymbolId,
        target_interface: SymbolId,
    ) -> Option<IrId> {
        let source_method_count = self.interface_method_names.get(&source_interface)?.len();
        let target_method_count = self.interface_method_names.get(&target_interface)?.len();

        // Reject downcast-like layout growth to avoid reading past source wrapper.
        if source_method_count < target_method_count {
            return None;
        }

        let slot_count = 1 + target_method_count; // object ptr + method slots

        // NULL GUARD. A null interface value is legitimate — `get()` returning
        // "not found", an unset field, a failed cast — and it must survive an
        // assignment as null so the caller's `x == null` check can see it.
        // The clone below DEREFERENCES the source wrapper to copy its slots,
        // so without this branch `var a:Iface = returnsNull()` segfaults on the
        // BIND, before any null check the user wrote can run. That crash lands
        // in JIT'd code with no symbol and no message.
        //
        //   if src == 0 { result = 0 } else { result = <clone> }
        let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
        let src_as_int = self
            .builder
            .build_bitcast(fat_ptr_reg, IrType::I64)
            .unwrap_or(fat_ptr_reg);
        let zero = self.builder.build_const(IrValue::I64(0))?;
        let is_null = self.builder.build_cmp(CompareOp::Eq, src_as_int, zero)?;
        let null_block = self.builder.create_block()?;
        let clone_block = self.builder.create_block()?;
        let join_block = self.builder.create_block()?;
        self.builder
            .build_cond_branch(is_null, null_block, clone_block)?;

        // null path: propagate a null interface value untouched.
        self.builder.switch_to_block(null_block);
        let null_ptr = self.builder.build_const(IrValue::I64(0))?;
        let null_ptr = self
            .builder
            .build_bitcast(null_ptr, ptr_u8.clone())
            .unwrap_or(null_ptr);
        self.builder.build_branch(join_block)?;

        // clone path (the original body).
        self.builder.switch_to_block(clone_block);
        let fat_ptr_size = (slot_count * 8) as u64;
        let malloc_fn = self.get_or_register_extern_function(
            "malloc",
            vec![IrType::U64],
            IrType::Ptr(Box::new(IrType::U8)),
        );
        let size_reg = self.builder.build_const(IrValue::U64(fat_ptr_size))?;
        let cloned_ptr = self.builder.build_call_direct(
            malloc_fn,
            vec![size_reg],
            IrType::Ptr(Box::new(IrType::U8)),
        )?;

        let src_ptr = {
            let src_ty = self
                .builder
                .get_register_type(fat_ptr_reg)
                .unwrap_or(IrType::I64);
            if matches!(src_ty, IrType::Ptr(_)) {
                fat_ptr_reg
            } else {
                self.builder
                    .build_bitcast(fat_ptr_reg, IrType::Ptr(Box::new(IrType::U8)))?
            }
        };

        // Copy object slot + required method slots.
        let obj_ptr = self.builder.build_load(src_ptr, IrType::I64)?;
        self.builder.build_store(cloned_ptr, obj_ptr);
        for i in 1..slot_count {
            let offset_reg = self.builder.build_const(IrValue::I64((i * 8) as i64))?;
            let src_slot_ptr = self.builder.build_ptr_add(
                src_ptr,
                offset_reg,
                IrType::Ptr(Box::new(IrType::U8)),
            )?;
            let slot_val = self.builder.build_load(src_slot_ptr, IrType::I64)?;

            let dst_slot_ptr = self.builder.build_ptr_add(
                cloned_ptr,
                offset_reg,
                IrType::Ptr(Box::new(IrType::U8)),
            )?;
            self.builder.build_store(dst_slot_ptr, slot_val);
        }
        self.builder.build_branch(join_block)?;

        // join: pick whichever path ran.
        self.builder.switch_to_block(join_block);
        let result = self.builder.build_phi(join_block, ptr_u8)?;
        self.builder
            .add_phi_incoming(join_block, result, null_block, null_ptr);
        self.builder
            .add_phi_incoming(join_block, result, clone_block, cloned_ptr);
        Some(result)
    }

    /// If value_type is a class and target_type is an interface the class implements,
    /// wrap the value in a fat pointer. Otherwise return the value unchanged.
    pub(crate) fn maybe_wrap_for_interface(
        &mut self,
        value_reg: IrId,
        value_type: TypeId,
        target_type: TypeId,
    ) -> (IrId, bool) {
        let iface_sym = match self.get_interface_symbol(target_type) {
            Some(s) => s,
            None => return (value_reg, false),
        };

        // Class -> interface: build a fresh fat pointer wrapper from class vtable entries.
        if let Some(class_sym) = self.get_class_symbol(value_type) {
            // Already an interface fat pointer (e.g. a conditional result whose
            // branches were wrapped per-branch below, but whose static type is
            // still one branch's class): never class-wrap again — nesting the
            // fat pointer stores the wrapper where the receiver belongs and
            // dispatch reads garbage.
            if self.interface_wrapped_args.contains(&value_reg) {
                return (value_reg, false);
            }
            // Check if we have a vtable for this (class, interface) pair
            if !self.interface_vtables.contains_key(&(class_sym, iface_sym)) {
                return (value_reg, false);
            }

            return match self.wrap_in_interface_fat_ptr(value_reg, class_sym, iface_sym) {
                Some(wrapped) => (wrapped, true),
                None => (value_reg, false),
            };
        }

        // Interface -> same or different interface: clone the fat pointer so destination
        // does not alias source wrapper. This prevents use-after-free when one variable
        // is later reassigned (freeing its old fat pointer while the other still uses it).
        if let Some(src_iface_sym) = self.get_interface_symbol(value_type) {
            return match self.clone_interface_fat_ptr(value_reg, src_iface_sym, iface_sym) {
                Some(cloned) => (cloned, true),
                None => (value_reg, false),
            };
        }

        (value_reg, false)
    }

    /// Wrap a raw class object as an interface fat pointer when the class is
    /// NOT resolvable in this lowering context (fully imported, compiled after
    /// this file, so it has no SymbolId / vtable / compiled methods here). We
    /// know only the class's fully-qualified name and the interface. Build the
    /// `{obj, thunk…}` fat pointer, resolving each method slot to a NAME-keyed
    /// forward-ref dispatch thunk (`<class_fqn>.<method>`) that dedupes with
    /// the real thunk at merge. This is the order-independent counterpart to
    /// `wrap_in_interface_fat_ptr` for the cross-module `new C()` case.
    pub(crate) fn wrap_new_class_as_interface_by_name(
        &mut self,
        obj_reg: IrId,
        class_fqn: &str,
        interface_symbol: SymbolId,
    ) -> Option<IrId> {
        let method_names: Vec<String> = self
            .resolve_interface_method_names(interface_symbol)?
            .iter()
            .filter_map(|n| self.string_interner.get(*n).map(|s| s.to_string()))
            .collect();
        if method_names.is_empty() {
            return None;
        }
        let method_count = method_names.len();
        let fat_ptr_size = ((1 + method_count) * 8) as u64;
        let malloc_fn = self.get_or_register_extern_function(
            "malloc",
            vec![IrType::U64],
            IrType::Ptr(Box::new(IrType::U8)),
        );
        let size_reg = self.builder.build_const(IrValue::U64(fat_ptr_size))?;
        let fat_ptr = self.builder.build_call_direct(
            malloc_fn,
            vec![size_reg],
            IrType::Ptr(Box::new(IrType::U8)),
        )?;
        // obj at slot 0
        let obj_as_i64 = {
            let obj_ty = self
                .builder
                .get_register_type(obj_reg)
                .unwrap_or(IrType::I64);
            if matches!(obj_ty, IrType::Ptr(_)) {
                self.builder.build_bitcast(obj_reg, IrType::I64)?
            } else {
                obj_reg
            }
        };
        self.builder.build_store(fat_ptr, obj_as_i64);
        // method thunks at slots 1..N
        for (i, mname) in method_names.iter().enumerate() {
            let method_fqn = format!("{}.{}", class_fqn, mname);
            // Prefer a real thunk/function already present by name; else a
            // name-keyed forward-ref stub that dedupes at merge.
            let thunk_id = self
                .external_function_name_map
                .get(&format!(
                    "__vtable_dispatch_thunk__{}",
                    method_fqn
                        .chars()
                        .map(|c| if c.is_alphanumeric() { c } else { '_' })
                        .collect::<String>()
                ))
                .copied()
                .or_else(|| self.forward_ref_dispatch_thunk_by_name(&method_fqn))?;
            let fn_ref = self.builder.build_function_ref(thunk_id)?;
            let offset_val = self
                .builder
                .build_const(IrValue::I64(((i + 1) * 8) as i64))?;
            let slot_ptr = self.builder.build_ptr_add(
                fat_ptr,
                offset_val,
                IrType::Ptr(Box::new(IrType::U8)),
            )?;
            self.builder.build_store(slot_ptr, fn_ref);
        }
        Some(fat_ptr)
    }

    /// Check if a list of constraint TypeIds contains at least one interface constraint
    pub(crate) fn has_interface_constraint(&self, constraints: &[TypeId]) -> bool {
        let type_table = self.type_table;
        constraints.iter().any(|c| {
            type_table
                .get(*c)
                .map_or(false, |t| matches!(t.kind, TypeKind::Interface { .. }))
        })
    }

    pub(crate) fn shared_interface_for(&self, a: SymbolId, b: SymbolId) -> Option<SymbolId> {
        let a_ifaces: std::collections::BTreeSet<SymbolId> = self
            .interface_vtables
            .keys()
            .filter(|(c, _)| *c == a)
            .map(|(_, i)| *i)
            .collect();
        let mut best: Option<(SymbolId, usize)> = None;
        for (c, i) in self.interface_vtables.keys() {
            if *c == b && a_ifaces.contains(i) {
                let n = self
                    .resolve_interface_method_names(*i)
                    .map(|v| v.len())
                    .unwrap_or(0);
                if best.map_or(true, |(_, bn)| n > bn) {
                    best = Some((*i, n));
                }
            }
        }
        best.map(|(i, _)| i)
    }

    /// Emit a diagnostic at an interface method call site when the
    /// call-side HIR type was `Dynamic`-shaped (a Ptr after
    /// `convert_type`). Two cases:
    ///
    /// - `resolved` is `Some` → we recovered the concrete return type
    ///   from `interface_method_return_types`. Emit a `Hint` so the
    ///   user knows an annotation at the binding site would silence
    ///   the recovery layer and the runtime extra work it implies.
    /// - `resolved` is `None` → we did NOT find a matching iface
    ///   method return type. The value will flow through downstream
    ///   code as `Dynamic`, which is a footgun (boxed eq, reflect
    ///   field reads, SIGSEGV-prone method dispatch). Emit a
    ///   `Warning` recommending an explicit `:T` annotation.
    ///
    /// Both diagnostics are no-ops when the HIR type is already
    /// concrete (the common path, fired millions of times per
    /// compile) so the cost stays in the cross-context case.
    pub(crate) fn emit_iface_return_diagnostic(
        &mut self,
        method_symbol: SymbolId,
        expr_ty: TypeId,
        resolved: Option<TypeId>,
        location: SourceLocation,
    ) {
        use crate::tast::TypeKind;
        // Only diagnose call sites whose HIR-level type kind is the
        // erased fallback (Dynamic / Placeholder / Unknown). A concrete
        // TAST type means the type checker already resolved it; no
        // user-actionable advice to offer.
        let expr_kind = self.type_table.get(expr_ty).map(|t| t.kind.clone());
        let is_erased = matches!(
            expr_kind,
            Some(TypeKind::Dynamic) | Some(TypeKind::Placeholder { .. }) | Some(TypeKind::Unknown)
        );
        if !is_erased {
            return;
        }
        // Suppress diagnostics for Void-returning methods: there's no
        // value flowing downstream, so there's no annotation to add
        // and no misdispatch risk. (Common for things like
        // `cache.reset()` on an iface field.)
        if let Some(real_ty) = resolved {
            if matches!(
                self.type_table.get(real_ty).map(|t| &t.kind),
                Some(TypeKind::Void)
            ) {
                return;
            }
        }
        let method_name = self
            .symbol_table
            .get_symbol(method_symbol)
            .and_then(|s| self.string_interner.get(s.name))
            .unwrap_or("<unknown>")
            .to_string();
        // Deduplicate: same (method, source line/col) fires once per
        // compile. Without this, generic helpers called from many
        // sites would dump dozens of identical warnings.
        let dedup_key = (
            method_name.clone(),
            location.file_id,
            location.line,
            location.column,
        );
        if !self.iface_diag_seen.insert(dedup_key) {
            return;
        }
        let span = Self::source_location_to_span(&location);
        let diag = match resolved {
            Some(real_ty) => {
                let ty_name = self.format_type_for_hint(real_ty);
                diagnostics::DiagnosticBuilder::hint(
                    format!(
                        "interface method `{}()` return type recovered as `{}` at MIR; \
                         annotating the binding site (e.g. `var x:{} = …`) makes the \
                         concrete type visible to earlier compiler passes too",
                        method_name, ty_name, ty_name
                    ),
                    span.clone(),
                )
                .code("W0014")
                .label(span, "cross-context iface return resolved late")
                .build()
            }
            None => diagnostics::DiagnosticBuilder::warning(
                format!(
                    "cannot resolve return type of interface method `{}()` across \
                     compilation contexts — value will flow as `Dynamic` and \
                     downstream `.method()` / `==` may misdispatch or crash; \
                     annotate the binding site (e.g. `var x:T = …`) to silence",
                    method_name
                ),
                span.clone(),
            )
            .code("W0015")
            .label(span, "iface return erased to Dynamic")
            .build(),
        };
        self.diagnostics.push(diag);
    }
}
