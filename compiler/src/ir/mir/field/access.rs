//! Field and index reads, including the dynamic and anonymous-structure fallbacks.

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
    pub(crate) fn lower_field_access(
        &mut self,
        obj: IrId,
        field: SymbolId,
        receiver_ty: TypeId,
        field_ty: TypeId,
    ) -> Option<IrId> {
        // Type erasure: if field_ty is a TypeParameter, resolve to concrete type
        // using the receiver's type arguments (e.g., Container<Float>.value → Float)
        let resolved_field_ty = self.resolve_type_param_from_receiver(field_ty, receiver_ty);
        let field_ty = resolved_field_ty.unwrap_or(field_ty);

        // SPECIAL CASE: Auto-unbox Dynamic for field access
        // If receiver is Dynamic, automatically unbox to get the actual object pointer
        let (obj, receiver_ty) = {
            let type_table = self.type_table;
            let obj_ir_type = self.builder.get_register_type(obj);
            if let Some(ty) = type_table.get(receiver_ty) {
                if matches!(ty.kind, TypeKind::Dynamic) {
                    // Cross-context iface dispatch fallback: when the receiver's
                    // HIR type is `Dynamic` (interface method return type wasn't
                    // resolved cross-file) but the MIR value is already a raw
                    // pointer to a stdlib reference type (HaxeArray, HaxeString,
                    // Bytes), the unbox-then-class-or-reflect path below treats
                    // those bytes as a `DynamicValue` header and SIGBUSes.
                    //
                    // Check the stdlib mapping by field name against the common
                    // reference classes first. If `length` matches `Array.length`,
                    // call `array_length` directly with the raw pointer — the
                    // runtime reads `HaxeArray.len` at a fixed offset and works.
                    //
                    // Class instances retain their `__type_id` header; for those,
                    // `field_exists_in_any_class` further down handles dispatch
                    // correctly after unboxing.
                    //
                    // A receiver register already typed String is a RAW
                    // HaxeString that flowed out of a typed producer (e.g.
                    // `string_concat`) under an erased HIR type — not a
                    // DynamicValue box. The unbox+reflect path below walks
                    // garbage; dispatch the known property directly.
                    if matches!(&obj_ir_type, Some(IrType::String))
                        || matches!(&obj_ir_type, Some(IrType::Ptr(inner)) if matches!(inner.as_ref(), IrType::String))
                    {
                        let fname = self
                            .symbol_table
                            .get_symbol(field)
                            .and_then(|s| self.string_interner.get(s.name));
                        if fname == Some("length") {
                            let len_id = self.get_or_register_extern_function(
                                "haxe_string_length",
                                vec![IrType::String],
                                IrType::I64,
                            );
                            let raw =
                                self.builder
                                    .build_call_direct(len_id, vec![obj], IrType::I64)?;
                            // Erased field type ⇒ downstream consumers speak
                            // the Dynamic box protocol; concrete Int callers
                            // take the raw value.
                            let field_erased = type_table.get(field_ty).map(|t| {
                                matches!(
                                    t.kind,
                                    TypeKind::Dynamic
                                        | TypeKind::Placeholder { .. }
                                        | TypeKind::Unknown
                                )
                            });
                            if field_erased == Some(true) {
                                return self.box_primitive_as_dynamic(
                                    raw,
                                    IrType::I64,
                                    PrimBoxKind::Int,
                                );
                            }
                            return Some(raw);
                        }
                    }
                    if let Some(IrType::Ptr(inner)) = &obj_ir_type {
                        if matches!(**inner, IrType::Void) {
                            // Try the stdlib name-based dispatch first. Only
                            // covers `Array.length`-style zero-arg properties
                            // returning a primitive; boxes the result so
                            // downstream Dynamic-aware code (string concat,
                            // trace) gets a `DynamicValue*` rather than a
                            // raw i32 it would later deref as a pointer.
                            if let Some(boxed) =
                                self.try_dynamic_stdlib_property_dispatch(obj, field)
                            {
                                return Some(boxed);
                            }

                            let field_in_class = self.field_exists_in_any_class(field);

                            // Unbox to get the actual object pointer from DynamicValue*
                            let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                            let unbox_func = self.get_or_register_extern_function(
                                "haxe_unbox_reference_ptr",
                                vec![ptr_u8.clone()],
                                ptr_u8.clone(),
                            );
                            let unboxed_obj =
                                self.builder
                                    .build_call_direct(unbox_func, vec![obj], ptr_u8)?;

                            if field_in_class {
                                return self.lower_field_access_for_class(
                                    unboxed_obj,
                                    field,
                                    field_ty,
                                );
                            }
                            // No class has this field — use Reflect API (anonymous object).
                            // Already unboxed above, so use raw_anon path (no double-unbox).
                            return self.raw_anon_reflect_field_read(unboxed_obj, field, field_ty);
                        }
                    }
                    // Also check for I64 - this is a raw pointer from Array element access
                    if matches!(&obj_ir_type, Some(IrType::I64)) {
                        let field_in_class = self.field_exists_in_any_class(field);
                        if field_in_class {
                            return self.lower_field_access_for_class(obj, field, field_ty);
                        }
                        return self.dynamic_reflect_field_read(obj, field, field_ty);
                    }
                    // Ptr(U8) from stdlib method returns (e.g., MutexGuard_get) is a raw
                    // class pointer, NOT a boxed DynamicValue. Check class fields first.
                    if matches!(&obj_ir_type, Some(IrType::Ptr(inner)) if matches!(**inner, IrType::U8))
                    {
                        let field_in_class = self.field_exists_in_any_class(field);
                        if field_in_class {
                            return self.lower_field_access_for_class(obj, field, field_ty);
                        }
                        // Raw Ptr(U8) is NOT boxed — call haxe_reflect_field directly
                        // without the haxe_unbox_reference_ptr step that
                        // dynamic_reflect_field_read would apply.
                        return self.raw_anon_reflect_field_read(obj, field, field_ty);
                    }

                    // Unbox to get the actual object pointer
                    let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                    let unbox_func_id = self.get_or_register_extern_function(
                        "haxe_unbox_reference_ptr",
                        vec![ptr_u8.clone()],
                        ptr_u8.clone(),
                    );
                    let unboxed_obj =
                        self.builder
                            .build_call_direct(unbox_func_id, vec![obj], ptr_u8.clone())?;

                    // Get the actual class type from the field's class
                    // The field_index_map tells us which class this field belongs to
                    // For Dynamic types, the field symbol may be a newly created placeholder,
                    // so we need to look up by field name instead
                    let (actual_type, _resolved_field) = if let Some(&(class_type_id, _field_idx)) =
                        self.field_index_map.get(&field)
                    {
                        (class_type_id, field)
                    } else {
                        // Field not found by SymbolId - try looking up by name
                        // This handles Dynamic field access where a new symbol was created
                        let field_name = self.symbol_table.get_symbol(field).map(|s| s.name);

                        if let Some(name) = field_name {
                            // Search for any field with the same name in field_index_map
                            let mut found = None;
                            for (sym, &(class_ty, _idx)) in &self.field_index_map {
                                if let Some(sym_info) = self.symbol_table.get_symbol(*sym) {
                                    if sym_info.name == name {
                                        // Get the field's actual type from the symbol
                                        let resolved_field_ty = sym_info.type_id;
                                        found = Some((class_ty, *sym, resolved_field_ty));
                                        break;
                                    }
                                }
                            }

                            if let Some((class_ty, resolved_sym, resolved_field_ty)) = found {
                                // Early return with the correct field symbol AND correct field type
                                return self.lower_field_access(
                                    unboxed_obj,
                                    resolved_sym,
                                    class_ty,
                                    resolved_field_ty,
                                );
                            } else {
                                // Dynamic field READ fallback via Reflect API.
                                // Safe: field_index_map name-match failed, so no class has this field.
                                // Only anonymous objects (typed as Dynamic) reach here.
                                return self.dynamic_reflect_field_read(
                                    unboxed_obj,
                                    field,
                                    field_ty,
                                );
                            }
                        } else {
                            (receiver_ty, field)
                        }
                    };

                    // If we reach here, we couldn't resolve the field - fall through to normal handling
                    // This shouldn't happen for valid Dynamic field access, but provides a fallback
                    (unboxed_obj, actual_type)
                } else {
                    (obj, receiver_ty)
                }
            } else {
                (obj, receiver_ty)
            }
        };

        // SPECIAL CASE: Check if this is a property access on a @:coreType extern class
        // For example, Array.length should map to haxe_array_length() runtime call
        // These classes have no actual fields - all access must go through runtime functions
        //
        // IMPORTANT: Skip this check if the field is a known user class field.
        // Without this guard, brute-force fallback in get_stdlib_runtime_info can match
        // user fields to unrelated stdlib methods with the same bare name (e.g.,
        // ArrayIterator.current matched to Thread.current → sys_thread_current).
        // When receiver type is Placeholder (unresolved extern class like rayzor.Bytes),
        // never consider it a known user field — it must go through stdlib dispatch.
        let receiver_is_placeholder = {
            let type_table = self.type_table;
            type_table
                .get(receiver_ty)
                .map_or(false, |t| matches!(t.kind, TypeKind::Placeholder { .. }))
        };
        // Existence PROBE only (gates stdlib property dispatch): ambiguity
        // still means "a user field of this name exists" — the erroring
        // resolver is reserved for callers that consume the slot.
        let is_known_user_field = !receiver_is_placeholder
            && (self.field_index_map.contains_key(&field)
                || !matches!(
                    self.resolve_field_index_candidates(
                        self.symbol_table
                            .get_symbol(field)
                            .map(|s| s.name)
                            .unwrap_or_default(),
                        receiver_ty,
                    ),
                    FieldIndexResolution::None
                ));

        let field_name_debug = self
            .symbol_table
            .get_symbol(field)
            .and_then(|s| self.string_interner.get(s.name))
            .unwrap_or("<unknown>");

        if !is_known_user_field {
            if let Some((_class_match, _method, runtime_call)) =
                self.get_stdlib_runtime_info(field, receiver_ty, Some(0), None)
            {
                let runtime_func = runtime_call.runtime_name;
                debug!(
                    "[lower_field_access] Found stdlib property! runtime_func={}",
                    runtime_func
                );

                // Determine result type based on whether it returns a primitive or complex type
                // If needs_out_param is false and has_return is true, it returns a primitive (i32/i64/f64)
                // Otherwise it returns a complex type (Ptr) or void
                let result_type = if !runtime_call.needs_out_param && runtime_call.has_return {
                    // Returns a primitive - get the actual primitive type from field_ty
                    let field_kind = {
                        let type_table = self.type_table;
                        type_table.get(field_ty).map(|t| t.kind.clone())
                    };

                    // Map TAST primitive types to IR types correctly
                    match field_kind {
                        Some(crate::tast::TypeKind::Int) => IrType::I32,
                        Some(crate::tast::TypeKind::Float) => IrType::F64,
                        Some(crate::tast::TypeKind::Bool) => IrType::Bool,
                        _ => {
                            warn!(
                                "Unexpected field kind {:?} for primitive-returning function {}",
                                field_kind, runtime_func
                            );
                            self.convert_type(field_ty)
                        }
                    }
                } else {
                    // Returns a complex type or void
                    self.convert_type(field_ty)
                };

                debug!(
                    "[lower_field_access] result_type for {} = {:?} (needs_out_param={}, has_return={})",
                    runtime_func,
                    result_type,
                    runtime_call.needs_out_param,
                    runtime_call.has_return
                );

                // Generate a call to the runtime property getter
                // Property getters take the object as the only parameter
                // Use explicit Ptr(Void) type for opaque stdlib objects (Array, String, etc.)
                let param_types = vec![IrType::Ptr(Box::new(IrType::Void))];
                let runtime_func_id = self.get_or_register_extern_function(
                    &runtime_func,
                    param_types,
                    result_type.clone(),
                );

                // Call the property getter with just the object
                let result_reg =
                    self.builder
                        .build_call_direct(runtime_func_id, vec![obj], result_type.clone());

                // DEBUG: Check actual type of result register
                if let Some(reg) = result_reg {
                    if let Some(reg_type) = self.builder.get_register_type(reg) {
                        debug!(
                            "[lower_field_access] result_reg={}, register_type={:?}",
                            reg, reg_type
                        );
                    } else {
                        debug!(
                            "[lower_field_access] result_reg={} has no type in builder",
                            reg
                        );
                    }
                }

                return result_reg;
            } else {
                debug!(
                    "[lower_field_access] get_stdlib_runtime_info returned None for field='{}' ({:?}), receiver_ty={:?}",
                    field_name_debug, field, receiver_ty
                );
            }
        } // end if !is_known_user_field

        // Check if this is a property with a custom getter
        // Try direct SymbolId lookup first, then fall back to name-based matching
        let mut property_info_owned = self.property_access_map.get(&field).cloned();
        if property_info_owned.is_none() {
            // Name-based fallback: SymbolIds may differ between import and user modules.
            // Prefer entries with `Method(...)` getters over `Default` — orphan entries
            // from prior BLADE cache loads (with empty class_name and Default getter)
            // can shadow the real definition when iteration order surfaces them first.
            // See bugs_known.md for the StringBuf.length cross-test contamination case.
            if let Some(field_name) = self.symbol_table.get_symbol(field).map(|s| s.name) {
                // Keep the name-based match from crossing class boundaries: a
                // candidate whose owner class is known to differ from the receiver's
                // belongs to an unrelated class. Owner from `field_class_names`
                // (forwarded from imports); both unknown falls through to the
                // ambiguity check below — a UNIQUE getter target may still
                // dispatch, but multiple distinct targets with unconfirmed
                // ownership is a guess and hard-fails (E0802).
                let receiver_class_name = self.receiver_type_class_name(receiver_ty);

                let mut method_matches: Vec<(Option<String>, crate::tast::PropertyAccessInfo)> =
                    Vec::new();
                let mut default_match: Option<crate::tast::PropertyAccessInfo> = None;
                for (sym_id, info) in &self.property_access_map {
                    let sym_name = match self.symbol_table.get_symbol(*sym_id) {
                        Some(s) => s.name,
                        None => continue,
                    };
                    if sym_name != field_name {
                        continue;
                    }
                    if let (Some(owner), Some(recv)) = (
                        self.field_class_names.get(sym_id),
                        receiver_class_name.as_ref(),
                    ) {
                        if !Self::class_names_match(owner, recv) {
                            continue;
                        }
                    }
                    match info.getter {
                        crate::tast::PropertyAccessor::Method(_) => {
                            method_matches
                                .push((self.field_class_names.get(sym_id).cloned(), info.clone()));
                        }
                        _ => {
                            if default_match.is_none() {
                                default_match = Some(info.clone());
                            }
                        }
                    }
                }
                // A candidate whose owner is POSITIVELY confirmed to be the
                // receiver's class beats unconfirmed ones.
                if let Some(recv) = receiver_class_name.as_ref() {
                    let confirmed: Vec<_> = method_matches
                        .iter()
                        .filter(|(owner, _)| {
                            owner
                                .as_ref()
                                .is_some_and(|o| Self::class_names_match(o, recv))
                        })
                        .cloned()
                        .collect();
                    if !confirmed.is_empty() {
                        method_matches = confirmed;
                    }
                }
                // Same getter method name = same target; only distinct targets
                // are ambiguous.
                let mut getter_names: Vec<_> = method_matches
                    .iter()
                    .filter_map(|(_, info)| match &info.getter {
                        crate::tast::PropertyAccessor::Method(g) => Some(*g),
                        _ => None,
                    })
                    .collect();
                getter_names.sort_unstable();
                getter_names.dedup();
                if getter_names.len() > 1 {
                    let field_str = self
                        .string_interner
                        .get(field_name)
                        .unwrap_or("<unknown>")
                        .to_string();
                    let owners = method_matches
                        .iter()
                        .map(|(owner, _)| owner.as_deref().unwrap_or("<unknown class>"))
                        .collect::<Vec<_>>()
                        .join(", ");
                    let loc = self
                        .symbol_table
                        .get_symbol(field)
                        .map(|s| s.definition_location)
                        .unwrap_or(SourceLocation::unknown());
                    self.add_error(
                        &format!(
                            "E0802: ambiguous property access: `{}` matches getters on multiple classes ({}) and the receiver's class could not be confirmed. Annotate the receiver's type",
                            field_str, owners
                        ),
                        loc,
                    );
                    return None;
                }
                property_info_owned = method_matches
                    .into_iter()
                    .next()
                    .map(|(_, info)| info)
                    .or(default_match);
            }
        }
        if let Some(property_info) = property_info_owned.as_ref() {
            match &property_info.getter {
                crate::tast::PropertyAccessor::Method(getter_method_name) => {
                    // Look up the getter method by name in function_map and external_function_map
                    let getter_func_id = self
                        .function_map
                        .iter()
                        .find(|(sym_id, _)| {
                            if let Some(symbol) = self.symbol_table.get_symbol(**sym_id) {
                                symbol.name == *getter_method_name
                            } else {
                                false
                            }
                        })
                        .map(|(_, func_id)| *func_id)
                        .or_else(|| {
                            self.external_function_map
                                .iter()
                                .find(|(sym_id, _)| {
                                    if let Some(symbol) = self.symbol_table.get_symbol(**sym_id) {
                                        symbol.name == *getter_method_name
                                    } else {
                                        false
                                    }
                                })
                                .map(|(_, func_id)| *func_id)
                        });

                    if let Some(func_id) = getter_func_id {
                        // Determine the result type: check the function signature first,
                        // then try the definition-site field symbol (from property_access_map),
                        // finally fall back to the expression-level field_ty.
                        let result_type = self
                            .builder
                            .module
                            .functions
                            .get(&func_id)
                            .map(|f| f.signature.return_type.clone())
                            .unwrap_or_else(|| {
                                // Function not in module (forward ref from import).
                                // The access-site field symbol may have unresolved type.
                                // Try the definition-site symbol from property_access_map
                                // which was populated from the defining module.
                                let def_sym_type =
                                    self.property_access_map.iter().find_map(|(def_sym, _)| {
                                        let sym = self.symbol_table.get_symbol(*def_sym)?;
                                        let name = self.symbol_table.get_symbol(field)?.name;
                                        if sym.name == name && sym.type_id.as_raw() != u32::MAX {
                                            Some(sym.type_id)
                                        } else {
                                            None
                                        }
                                    });
                                let ty = def_sym_type.unwrap_or(field_ty);
                                self.convert_type(ty)
                            });
                        return self
                            .builder
                            .build_call_direct(func_id, vec![obj], result_type);
                    }

                    // Fallback: extern-class accessor — try the stdlib mapping
                    // (e.g. sys.thread.Tls.get_value → sys_tls_get_value).
                    let getter_name = *getter_method_name;
                    let result_type = self.convert_type(field_ty);
                    if let Some(result) = self.try_property_call_via_stdlib(
                        receiver_ty,
                        getter_name,
                        vec![obj],
                        result_type,
                    ) {
                        return Some(result);
                    }
                    // Getter not found — fall through to other paths (stdlib dispatch, GEP, etc.)
                }
                crate::tast::PropertyAccessor::Null | crate::tast::PropertyAccessor::Never => {
                    self.add_error(
                        "Cannot read from write-only property (Null or Never getter)",
                        SourceLocation::unknown(),
                    );
                    return None;
                }
                crate::tast::PropertyAccessor::Default | crate::tast::PropertyAccessor::Dynamic => {
                    // Fall through to direct field access
                }
            }
        }

        // ANONYMOUS OBJECT FIELD ACCESS
        // If receiver is an anonymous type (or a typedef alias to one),
        // use rayzor_anon_get_field_by_index
        {
            let mut resolved_receiver_ty = self.resolve_through_aliases(receiver_ty);
            let type_table = self.type_table;
            let mut is_anon = matches!(
                type_table.get(resolved_receiver_ty).map(|t| &t.kind),
                Some(TypeKind::Anonymous { .. })
            );
            // Cross-module: a structural typedef return (e.g.
            // `function load():LoadedModel` where
            // `typedef LoadedModel = {model:..., tokenizer:..., metadata:...}`)
            // can decay to a synthetic Class carrying the typedef's qualified
            // name instead of resolving to its real Anonymous target — no
            // Placeholder involved, so `resolve_through_aliases` never sees
            // it. Detect that pattern (a Class with no fields of its own) and
            // recover the target by finding a same-named TypeAlias whose
            // target IS Anonymous elsewhere in the shared type table.
            if !is_anon {
                if let Some(TypeKind::Class { symbol_id, .. }) =
                    type_table.get(resolved_receiver_ty).map(|t| &t.kind)
                {
                    let has_own_fields = self
                        .field_index_map
                        .values()
                        .any(|(cty, _)| *cty == resolved_receiver_ty);
                    if !has_own_fields {
                        if let Some(qname) =
                            self.symbol_table.get_symbol(*symbol_id).and_then(|s| {
                                s.qualified_name.and_then(|n| self.string_interner.get(n))
                            })
                        {
                            if let Some(anon_target) =
                                self.find_typedef_anonymous_target_by_name(qname)
                            {
                                resolved_receiver_ty = anon_target;
                                is_anon = true;
                            }
                        }
                    }
                }
            }
            if is_anon {
                // Get field name and find its sorted index + actual type from the anonymous struct
                let field_name = self
                    .symbol_table
                    .get_symbol(field)
                    .and_then(|s| self.string_interner.get(s.name))
                    .map(|s| s.to_string());

                if let Some(field_name) = field_name {
                    // Get all anonymous fields, sort them, and find both the sorted index
                    // AND the actual field type (not the possibly-Dynamic expr type)
                    let sorted_result = if let Some(ty_info) = type_table.get(resolved_receiver_ty)
                    {
                        if let TypeKind::Anonymous {
                            fields: anon_fields,
                        } = &ty_info.kind
                        {
                            // Build (name, type_id) pairs and sort by name
                            let mut named_fields: Vec<(String, TypeId)> = anon_fields
                                .iter()
                                .filter_map(|f| {
                                    self.string_interner
                                        .get(f.name)
                                        .map(|s| (s.to_string(), f.type_id))
                                })
                                .collect();
                            named_fields.sort_by(|a, b| a.0.cmp(&b.0));
                            named_fields
                                .iter()
                                .enumerate()
                                .find(|(_, (n, _))| *n == field_name)
                                .map(|(idx, (_, ty))| (idx, *ty))
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    if let Some((sorted_idx, actual_field_ty)) = sorted_result {
                        // Emit: rayzor_anon_get_field_by_index(handle, sorted_idx) -> u64
                        let anon_get_id = self.get_or_register_extern_function(
                            "rayzor_anon_get_field_by_index",
                            vec![IrType::Ptr(Box::new(IrType::U8)), IrType::I32],
                            IrType::I64,
                        );

                        let idx_val = self.builder.build_const(IrValue::I32(sorted_idx as i32))?;
                        let raw_val = self.builder.build_call_direct(
                            anon_get_id,
                            vec![obj, idx_val],
                            IrType::I64,
                        )?;

                        // Convert the raw u64 back to the field's actual type
                        // Use the anonymous struct's field type (not the expression-level type
                        // which may be Dynamic)
                        return self.coerce_from_i64(raw_val, actual_field_ty);
                    }
                }
            }
        }

        // Look up the field index from our field_index_map
        // Also capture the actual field type from the symbol table for correct IR type resolution.
        // The passed-in field_ty may be Dynamic when the TAST couldn't resolve the field type
        // (e.g., forward-referenced classes in the same file).
        let (class_type_id, field_index, actual_field_ty) = match self
            .field_index_map
            .get(&field)
            .copied()
        {
            Some((class_ty, idx)) => {
                // The SymbolId is a GLOBAL monotonic counter, so any stdlib
                // member added anywhere shifts every later id; an imported
                // receiver's field id can then ALIAS a different class's
                // field_index_map entry and GEP the wrong offset (SIGSEGV in
                // cross-module dispatch). When the looked-up owner class does
                // not equal the receiver, it is either an inherited field
                // (fine) or a cross-context collision (wrong). Cross-check with
                // the drift-proof name+receiver-type probe (no E0803) and
                // override only on a UNIQUE disagreement — a collision. For
                // inherited fields the probe yields None/Ambiguous, so the
                // SymbolId result stands.
                // A drifted id can alias a DIFFERENT class's entry OR a
                // different field of the SAME class (wrong index but matching
                // class), so gating on class mismatch is insufficient. The
                // name+receiver probe is the authoritative slot (by field name,
                // disambiguated by the stable receiver type); prefer its unique
                // result whenever it resolves, and fall back to the SymbolId
                // fast path only when the probe is None/Ambiguous (e.g.
                // inherited fields the probe does not attribute to the
                // receiver's own class). The probe never emits E0803.
                let mut chosen = (class_ty, idx);
                let mut probe_tag = "no-name";
                if let Some(fname) = self.symbol_table.get_symbol(field).map(|s| s.name) {
                    match self.resolve_field_index_candidates(fname, receiver_ty) {
                        FieldIndexResolution::Unique(nct, nidx) => {
                            chosen = (nct, nidx);
                            probe_tag = "unique";
                        }
                        FieldIndexResolution::Ambiguous(_) => probe_tag = "ambiguous",
                        FieldIndexResolution::None => probe_tag = "none",
                    }
                }
                if std::env::var("RAYZOR_FIELD_DEBUG").is_ok() {
                    let fname = self
                        .symbol_table
                        .get_symbol(field)
                        .and_then(|si| self.string_interner.get(si.name))
                        .unwrap_or("?");
                    if fname == "finishReason" {
                        let recv_kind = {
                            let tt = self.type_table;
                            format!("{:?}", tt.get(receiver_ty).map(|t| &t.kind))
                        };
                        eprintln!(
                            "[FLD] {} sym={:?} direct=({:?},{}) probe={} chosen=({:?},{}) recv={:?} {}",
                            fname,
                            field,
                            class_ty,
                            idx,
                            probe_tag,
                            chosen.0,
                            chosen.1,
                            receiver_ty,
                            &recv_kind[..recv_kind.len().min(90)]
                        );
                    }
                }
                let sym_type = if chosen == (class_ty, idx) {
                    self.symbol_table
                        .get_symbol(field)
                        .map(|s| s.type_id)
                        .unwrap_or(field_ty)
                } else {
                    let (cty, cidx) = chosen;
                    let mut st = field_ty;
                    for (sym, &(mcty, mcidx)) in &self.field_index_map {
                        if mcty == cty && mcidx == cidx {
                            if let Some(si) = self.symbol_table.get_symbol(*sym) {
                                st = si.type_id;
                            }
                            break;
                        }
                    }
                    st
                };
                (chosen.0, chosen.1, sym_type)
            }
            None => {
                // Fallback: Try to find field by name instead of SymbolId
                // This handles cases where the same field has different SymbolIds in different scopes
                let field_name = self
                    .symbol_table
                    .get_symbol(field)
                    .map(|s| s.name)
                    .or_else(|| None)?;

                let field_name_str = self.string_interner.get(field_name).unwrap_or("<unknown>");

                // Use helper that disambiguates by receiver type when multiple classes
                // have the same field name (e.g., StringBuf.length vs List.length)
                let found_result = self
                    .resolve_field_index_by_name(field_name, receiver_ty)
                    .map(|(class_ty, idx)| {
                        let sym_type = {
                            // Find the symbol in field_index_map that matches this class_ty + idx
                            let mut st = field_ty;
                            for (sym, &(cty, cidx)) in &self.field_index_map {
                                if cty == class_ty && cidx == idx {
                                    if let Some(si) = self.symbol_table.get_symbol(*sym) {
                                        st = si.type_id;
                                    }
                                    break;
                                }
                            }
                            st
                        };
                        (class_ty, idx, sym_type)
                    });

                // Fallback: class_instance_fields lookup via register_class_hints
                // This handles derive trait returns (e.g. clone()) where the receiver's
                // HIR type is Dynamic but the register has a class hint set.
                let found_result = found_result.or_else(|| {
                    let class_hint = self.register_class_hints.get(&obj)?;
                    // Find the class SymbolId matching this hint name
                    for (sym_id, fields) in &self.class_instance_fields {
                        let sym_name = self
                            .symbol_table
                            .get_symbol(*sym_id)
                            .and_then(|s| self.string_interner.get(s.name));
                        if sym_name == Some(class_hint.as_str()) {
                            // Search for our field by name in this class's fields
                            for &(f_sym, f_type, gep_idx) in fields {
                                let f_name = self
                                    .symbol_table
                                    .get_symbol(f_sym)
                                    .and_then(|s| self.string_interner.get(s.name));
                                if f_name == Some(field_name_str) {
                                    return Some((receiver_ty, gep_idx, f_type));
                                }
                            }
                        }
                    }
                    None
                });

                let found_result = found_result.or_else(|| {
                    let class_sym = match self.type_table.get(receiver_ty).map(|ti| &ti.kind) {
                        Some(TypeKind::Class { symbol_id, .. }) => *symbol_id,
                        _ => return None,
                    };
                    let sym = self.symbol_table.get_symbol(class_sym)?;
                    // Qualified name first (exact, collision-proof); bare name
                    // second (routes through the alias map, which refuses
                    // ambiguous bare names rather than guessing).
                    let gep_idx = sym
                        .qualified_name
                        .and_then(|q| self.string_interner.get(q))
                        .and_then(|qn| lookup_class_field_gep(qn, field_name_str))
                        .or_else(|| {
                            self.string_interner
                                .get(sym.name)
                                .and_then(|cn| lookup_class_field_gep(cn, field_name_str))
                        })?;
                    Some((receiver_ty, gep_idx, field_ty))
                });

                match found_result {
                    Some(result) => result,
                    None => {
                        // Try typedef_field_map for anonymous struct fields (like FileStat)
                        // This handles cases where the typedef's anonymous struct fields
                        // are accessed with newly created symbols at the access site
                        //
                        // receiver_ty might be the typedef's TypeId OR the aliased anonymous struct TypeId
                        // Try both and also search all registered typedefs for this field name
                        let mut typedef_lookup = self
                            .typedef_field_map
                            .get(&(receiver_ty, field_name))
                            .copied();

                        // If not found with receiver_ty, search all typedefs for this field
                        if typedef_lookup.is_none() {
                            for ((typedef_ty, fname), &idx) in &self.typedef_field_map {
                                if *fname == field_name {
                                    typedef_lookup = Some(idx);
                                    // Use the typedef's type id for the result
                                    return Some(self.lower_typedef_field_access(
                                        obj,
                                        *typedef_ty,
                                        idx,
                                        field_ty,
                                    )?);
                                }
                            }
                        }

                        if let Some(typedef_field_idx) = typedef_lookup {
                            // Found in typedef_field_map - return (receiver_type, field_index, field_ty)
                            (receiver_ty, typedef_field_idx, field_ty)
                        } else {
                            // Last resort: look up the field by name in the type_table for anonymous structs
                            // This handles cross-module typedef field access where the typedef was
                            // registered in a different HIR->MIR pass
                            let type_table = self.type_table;

                            // Get the field name string for lookup
                            let field_name_str = self
                                .string_interner
                                .get(field_name)
                                .map(|s| s.to_string())
                                .unwrap_or_default();

                            // Search all types for an anonymous struct with this field name
                            let mut found_field = None;
                            for (type_id, type_info) in type_table.iter() {
                                if let TypeKind::Anonymous { fields } = &type_info.kind {
                                    for (idx, anon_field) in fields.iter().enumerate() {
                                        let anon_field_name = self
                                            .string_interner
                                            .get(anon_field.name)
                                            .map(|s| s.to_string())
                                            .unwrap_or_default();
                                        if anon_field_name == field_name_str {
                                            found_field = Some((type_id, idx as u32));
                                            break;
                                        }
                                    }
                                    if found_field.is_some() {
                                        break;
                                    }
                                }
                            }

                            if let Some((found_type_id, field_idx)) = found_field {
                                // Get the actual field type from the type_table
                                let actual_field_ty = {
                                    let type_table = self.type_table;
                                    if let Some(type_info) = type_table.get(found_type_id) {
                                        if let TypeKind::Anonymous { fields } = &type_info.kind {
                                            if let Some(field) = fields.get(field_idx as usize) {
                                                field.type_id
                                            } else {
                                                field_ty
                                            }
                                        } else {
                                            field_ty
                                        }
                                    } else {
                                        field_ty
                                    }
                                };
                                return Some(self.lower_typedef_field_access(
                                    obj,
                                    receiver_ty,
                                    field_idx,
                                    actual_field_ty,
                                )?);
                            }

                            // Check if this is actually a static field accessed as instance field
                            let global_lookup =
                                self.global_symbol_map.get(&field).copied().or_else(|| {
                                    self.builder
                                        .module
                                        .globals
                                        .values()
                                        .find(|g| g.name.ends_with(&format!(".{}", field_name_str)))
                                        .map(|g| g.id)
                                });
                            if let Some(global_id) = global_lookup {
                                let global_type = self
                                    .builder
                                    .module
                                    .globals
                                    .get(&global_id)
                                    .map(|g| g.ty.clone())
                                    .unwrap_or(IrType::Any);
                                return self.builder.build_load_global(global_id, global_type);
                            }

                            if std::env::var_os("RAYZOR_E0100_DEBUG").is_some() {
                                let recv_kind = self
                                    .type_table
                                    .get(receiver_ty)
                                    .map(|ti| format!("{:?}", ti.kind));
                                eprintln!(
                                    "[E0100] field='{}' sym={:?} receiver_ty={:?} kind={:?}",
                                    field_name_str, field, receiver_ty, recv_kind
                                );
                                // Is the receiver's class present in
                                // class_instance_fields AT ALL in this context?
                                // Two name/id resolution strategies both failed
                                // here, which points at the field table being
                                // ABSENT (never forwarded across the module
                                // boundary) rather than merely keyed differently.
                                // Print the evidence instead of guessing again.
                                let class_sym =
                                    match self.type_table.get(receiver_ty).map(|ti| &ti.kind) {
                                        Some(TypeKind::Class { symbol_id, .. }) => Some(*symbol_id),
                                        _ => None,
                                    };
                                let class_name = class_sym.and_then(|cs| {
                                    self.symbol_table
                                        .get_symbol(cs)
                                        .and_then(|s| self.string_interner.get(s.name))
                                });
                                let by_id = class_sym
                                    .map(|cs| self.class_instance_fields.contains_key(&cs))
                                    .unwrap_or(false);
                                let by_name = class_name
                                    .map(|cn| {
                                        self.class_instance_fields.keys().any(|k| {
                                            self.symbol_table
                                                .get_symbol(*k)
                                                .and_then(|s| self.string_interner.get(s.name))
                                                == Some(cn)
                                        })
                                    })
                                    .unwrap_or(false);
                                if let Ok(reg) = class_field_layouts().lock() {
                                    let mut keys: Vec<&String> = reg.keys().collect();
                                    keys.sort();
                                    eprintln!(
                                        "[E0100]   GLOBAL registry: {} classes, has_this={}, keys={:?}",
                                        reg.len(),
                                        class_name.map(|c| reg.contains_key(c)).unwrap_or(false),
                                        keys.iter().take(12).collect::<Vec<_>>()
                                    );
                                }
                                eprintln!(
                                    "[E0100]   class_name={:?} in_cif_by_id={} in_cif_by_name={} cif_len={} type_meta_calls={} cur_fn={:?}",
                                    class_name,
                                    by_id,
                                    by_name,
                                    self.class_instance_fields.len(),
                                    self.dbg_type_meta_calls,
                                    self.builder.current_function
                                );
                            }
                            self.add_error(
                                &format!(
                                    "Cannot access field '{}': class not registered or field does not exist",
                                    field_name_str,
                                ),
                                SourceLocation::unknown(),
                            );
                            return None;
                        }
                    }
                }
            }
        };

        // Get the type of the field — use actual_field_ty from the class metadata
        // (the symbol table's type), not field_ty (the expression-level type which
        // may be Dynamic when the TAST couldn't resolve forward-referenced class fields)
        let field_ir_ty = self.convert_type(actual_field_ty);

        // Type erasure: if the field's declared type is TypeParameter, use I64 for GEP stride
        // (all type-erased fields are stored as I64 regardless of concrete type)
        let field_is_type_param = {
            let declared_type_id = self.symbol_table.get_symbol(field).map(|s| s.type_id);
            if let Some(decl_id) = declared_type_id {
                let type_table = self.type_table;
                type_table.get(decl_id).map_or(false, |ti| {
                    matches!(ti.kind, crate::tast::TypeKind::TypeParameter { .. })
                })
            } else {
                false
            }
        };
        let gep_element_ty = if field_is_type_param {
            IrType::I64
        } else {
            field_ir_ty.clone()
        };

        // @:cstruct: use byte-offset PtrAdd instead of index-based GEP
        // Check both class_type_id (from field_index_map) and receiver_ty (expression type)
        let cstruct_type = if self.is_cstruct_class(class_type_id) {
            Some(class_type_id)
        } else if self.is_cstruct_class(receiver_ty) {
            Some(receiver_ty)
        } else {
            None
        };
        if let Some(cstruct_ty) = cstruct_type {
            if let Some(layout) = self.get_or_compute_cstruct_layout(cstruct_ty) {
                if let Some(field_layout) = layout.fields.iter().find(|f| f.symbol_id == field) {
                    let offset_const = self
                        .builder
                        .build_const(IrValue::I64(field_layout.byte_offset as i64))?;
                    let byte_ptr_ty = IrType::Ptr(Box::new(IrType::U8));
                    let field_ptr = self.builder.build_ptr_add(obj, offset_const, byte_ptr_ty)?;
                    let field_value = self.builder.build_load(field_ptr, field_ir_ty.clone())?;
                    self.builder.set_register_type(field_value, field_ir_ty);
                    return Some(field_value);
                }
                // Field not found by symbol_id — try by name
                let field_name = self.symbol_table.get_symbol(field).map(|s| s.name);
                if let Some(fname) = field_name {
                    let fname_str = self.string_interner.get(fname).unwrap_or("");
                    if let Some(field_layout) = layout.fields.iter().find(|f| f.name == fname_str) {
                        let offset_const = self
                            .builder
                            .build_const(IrValue::I64(field_layout.byte_offset as i64))?;
                        let byte_ptr_ty = IrType::Ptr(Box::new(IrType::U8));
                        let field_ptr =
                            self.builder.build_ptr_add(obj, offset_const, byte_ptr_ty)?;
                        let field_value =
                            self.builder.build_load(field_ptr, field_ir_ty.clone())?;
                        self.builder.set_register_type(field_value, field_ir_ty);
                        return Some(field_value);
                    }
                }
            }
        }

        // @:gpuStruct: byte-offset PtrAdd with f32→f64 promotion on read
        let gpu_struct_type = if self.is_gpu_struct_class(class_type_id) {
            Some(class_type_id)
        } else if self.is_gpu_struct_class(receiver_ty) {
            Some(receiver_ty)
        } else {
            None
        };
        if let Some(gpu_ty) = gpu_struct_type {
            if let Some(layout) = self.get_or_compute_gpu_struct_layout(gpu_ty) {
                let find_field = |layout: &GpuStructLayout| -> Option<GpuStructFieldLayout> {
                    layout
                        .fields
                        .iter()
                        .find(|f| f.symbol_id == field)
                        .cloned()
                        .or_else(|| {
                            let fname = self
                                .symbol_table
                                .get_symbol(field)
                                .and_then(|s| self.string_interner.get(s.name));
                            fname.and_then(|n| layout.fields.iter().find(|f| f.name == n).cloned())
                        })
                };
                if let Some(fl) = find_field(&layout) {
                    let offset_const = self
                        .builder
                        .build_const(IrValue::I64(fl.byte_offset as i64))?;
                    let byte_ptr_ty = IrType::Ptr(Box::new(IrType::U8));
                    let field_ptr = self.builder.build_ptr_add(obj, offset_const, byte_ptr_ty)?;
                    // Load as GPU type (f32/i32)
                    let raw_value = self.builder.build_load(field_ptr, fl.ir_type.clone())?;
                    // Promote f32→f64 so Haxe Float semantics work on CPU
                    let promoted = if fl.ir_type == IrType::F32 {
                        let cast = self.builder.build_cast(raw_value, IrType::F32, IrType::F64);
                        cast.unwrap_or(raw_value)
                    } else if fl.ir_type == IrType::I32 && field_ir_ty == IrType::I64 {
                        let cast = self.builder.build_cast(raw_value, IrType::I32, IrType::I64);
                        cast.unwrap_or(raw_value)
                    } else {
                        raw_value
                    };
                    self.builder.set_register_type(promoted, field_ir_ty);
                    return Some(promoted);
                }
            }
        }

        // Create constant for field index
        let index_const = self.builder.build_const(IrValue::I32(field_index as i32))?;

        // Use GetElementPtr to get pointer to the field
        // obj is a pointer to the struct, indices are [field_index]
        let field_ptr = self
            .builder
            .build_gep(obj, vec![index_const], gep_element_ty.clone())?;

        // Load the value from the field pointer
        let field_value = self.builder.build_load(field_ptr, gep_element_ty.clone())?;

        // Type erasure coercion: if field was loaded as I64 (erased type param),
        // coerce to the concrete type expected by the caller.
        // Note: field_ir_ty is always I64 for type params (from convert_type(TypeParameter)),
        // so we must check the RESOLVED type (field_ty) to detect when coercion is needed.
        let expected_ir_ty = self.convert_type(field_ty);
        if field_is_type_param && expected_ir_ty != IrType::I64 {
            let coerced = self.coerce_from_i64(field_value, field_ty)?;
            self.builder.set_register_type(coerced, expected_ir_ty);
            return Some(coerced);
        }

        // Register the type of the loaded value for use in later instructions (e.g., Cmp)
        self.builder.set_register_type(field_value, field_ir_ty);

        Some(field_value)
    }

    pub(crate) fn lower_index_access(&mut self, obj: IrId, idx: IrId, ty: TypeId) -> Option<IrId> {
        // Array element address, computed INLINE rather than via an
        // `haxe_array_get_ptr` runtime call. A non-inlinable call per element is an
        // optimization barrier: it blocks bounds-check hoisting and loop
        // vectorization (a `for (x in a) s += x` reduction can't become a SIMD
        // addv, and on wasm it's a host-import boundary). HaxeArray is #[repr(C)]
        // { ptr@0, len@8, cap@16, elem_size@24 } with uniform 8-byte element slots,
        // so the address is `*(arr.ptr) + idx*8` — the same i64-slot GEP used for
        // object fields (build_gep with IrType::I64 scales the index by 8).
        //
        // OOB/null behavior is unchanged: the old call returned null on OOB and the
        // caller then loaded from it (a crash), so an unchecked inline GEP is
        // behavior-equivalent for valid in-bounds access while staying branch-free
        // and vectorizable. (A bounds-check branch here would defeat the point —
        // BCE only runs in the AOT/pipeline PassManager, not the JIT path.)
        //
        // `obj` is the HaxeArray struct pointer; `arr.ptr` (the data buffer) is the
        // first field at offset 0, so a plain load reads it.
        let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
        let data_ptr = self.builder.build_load(obj, ptr_u8)?;
        // elem_ptr = data_ptr + idx * 8 (uniform 8-byte slots = the i64 stride).
        let elem_ptr = self.builder.build_gep(data_ptr, vec![idx], IrType::I64)?;

        // Determine the correct load type based on the element type.
        // Array slots are always 8 bytes, so we load as the storage type first.
        // For types smaller than 8 bytes (Int→I32, Bool), we need to cast after loading.
        let (load_type, target_type) = {
            let type_table = self.type_table;
            match type_table.get(ty).map(|ti| &ti.kind) {
                Some(crate::tast::TypeKind::String) => {
                    let t = IrType::Ptr(Box::new(IrType::String));
                    (t.clone(), t)
                }
                Some(crate::tast::TypeKind::Float) => (IrType::F64, IrType::F64),
                Some(crate::tast::TypeKind::Int) => (IrType::I64, IrType::I32),
                Some(crate::tast::TypeKind::Bool) => (IrType::I64, IrType::Bool),
                _ => {
                    let t = IrType::I64;
                    (t.clone(), t)
                }
            }
        };

        let loaded = self.builder.build_load(elem_ptr, load_type.clone())?;

        // Cast to target type if different from load type (e.g., I64 → I32 for Int elements)
        if load_type != target_type {
            self.builder.build_cast(loaded, load_type, target_type)
        } else {
            Some(loaded)
        }
    }

    pub(crate) fn lower_field_expr(&mut self, expr: &HirExpr) -> Option<IrId> {
        let HirExprKind::Field { object, field } = &expr.kind else {
            unreachable!("lower_field_expr on a non-Field expression")
        };
        // Check if this is an enum variant access (e.g., Color.Red)
        // In that case, the object is an Enum type symbol, not a value
        if let HirExprKind::Variable { symbol, .. } = &object.kind {
            if let Some(sym) = self.symbol_table.get_symbol(*symbol) {
                use crate::tast::SymbolKind;
                if sym.kind == SymbolKind::Enum {
                    // This is an enum variant access - get the variant discriminant
                    let enum_name = self.string_interner.get(sym.name).unwrap_or("<unknown>");
                    let field_sym = self.symbol_table.get_symbol(*field);
                    let field_name = field_sym
                        .and_then(|s| self.string_interner.get(s.name))
                        .unwrap_or("<unknown>");

                    if let Some(variants) = self.symbol_table.get_enum_variants(*symbol) {
                        for (idx, variant_id) in variants.iter().enumerate() {
                            let variant_sym = self.symbol_table.get_symbol(*variant_id);
                            let variant_name = variant_sym
                                .and_then(|s| self.string_interner.get(s.name))
                                .unwrap_or("<unknown>");
                            // Compare by name since the field symbol might be different from the variant symbol
                            if *variant_id == *field || variant_name == field_name {
                                // If enum has parameterized variants, all variants must be boxed
                                if self.enum_is_boxed(*symbol) {
                                    return self.build_boxed_enum_tag_only(idx as i32);
                                }
                                return self.builder.build_const(IrValue::I64(idx as i64));
                            }
                        }
                    }
                    // If field is not a variant, fall through to regular field access
                }
            }
        }

        // Regular field access
        debug!("[Field expression] About to lower object");
        let obj_reg = self.lower_expression(object)?;
        debug!(
            "[Field expression] Object lowered to reg={}, now calling lower_field_access",
            obj_reg
        );

        // @:move strict-move tracking: prepend a CheckLive guard if
        // the field's receiver register is a strict-move local. The
        // inner Variable arm may have already emitted one when the
        // object expression is a bare local; this covers paths that
        // bypass that arm (e.g. `this`-redirects, anon views).
        if self.strict_move_locals.contains(&obj_reg) {
            let loc = self.convert_source_location(&expr.source_location);
            let _ = self.builder.build_check_live(obj_reg, loc);
        }

        // Track object as temp if it's an OWNED heap-allocated value
        // This includes:
        // 1. Direct `new` expressions: `new Complex(...)`
        // 2. Method calls that return class instances: `z.mul(z)` returns new Complex
        //
        // We check if the return type is a Class (heap-allocated via malloc).
        // Runtime/extern functions typically return primitives, strings, or Dynamic,
        // not Class instances, so this heuristic is safe.
        let is_owned_heap_value = matches!(
            &object.kind,
            HirExprKind::New { .. } | HirExprKind::Call { .. }
        ) && self.get_drop_behavior(object.ty) == DropBehavior::AutoDrop;

        // Only register NEW expressions as temporaries, not method Call results.
        // Method calls (getObj(), input(), etc.) often return references to existing
        // objects — freeing these would corrupt the heap. Only `new Foo(...)` creates
        // a genuinely owned temporary that must be freed after the field access chain.
        let is_new_expr = matches!(&object.kind, HirExprKind::New { .. });
        if is_owned_heap_value && is_new_expr {
            self.temp_heap_values.push(obj_reg);
        }

        let receiver_ty = object.ty; // The type of the object being accessed

        // Structural subtyping: if object is a variable with an anon view,
        // redirect field access to the backing representation
        if let HirExprKind::Variable {
            symbol: obj_sym, ..
        } = &object.kind
        {
            if let Some(backing) = self.anon_views.get(obj_sym).cloned() {
                let field_name = self
                    .symbol_table
                    .get_symbol(*field)
                    .and_then(|s| self.string_interner.get(s.name))
                    .map(|s| s.to_string());

                if let Some(field_name) = field_name {
                    match &backing {
                        AnonBacking::Class { field_map, .. } => {
                            if let Some((_, gep_idx, field_type_id)) =
                                field_map.iter().find(|(n, ..)| *n == field_name)
                            {
                                let field_ir_ty = self.convert_type(*field_type_id);
                                let idx_const =
                                    self.builder.build_const(IrValue::I64(*gep_idx as i64))?;
                                let field_ptr = self.builder.build_gep(
                                    obj_reg,
                                    vec![idx_const],
                                    field_ir_ty.clone(),
                                )?;
                                return Some(self.builder.build_load(field_ptr, field_ir_ty)?);
                            }
                        }
                        AnonBacking::WiderAnon { field_map, .. } => {
                            if let Some((_, src_idx, field_type_id)) =
                                field_map.iter().find(|(n, ..)| *n == field_name)
                            {
                                let anon_get_id = self.get_or_register_extern_function(
                                    "rayzor_anon_get_field_by_index",
                                    vec![IrType::Ptr(Box::new(IrType::U8)), IrType::I32],
                                    IrType::I64,
                                );
                                let idx_val =
                                    self.builder.build_const(IrValue::I32(*src_idx as i32))?;
                                let raw_val = self.builder.build_call_direct(
                                    anon_get_id,
                                    vec![obj_reg, idx_val],
                                    IrType::I64,
                                )?;
                                return self.coerce_from_i64(raw_val, *field_type_id);
                            }
                        }
                    }
                }
            }
        }

        // Raw anonymous object handle: use haxe_reflect_field directly
        // without haxe_unbox_reference_ptr (the handle is NOT a boxed DynamicValue*)
        if let HirExprKind::Variable {
            symbol: obj_sym, ..
        } = &object.kind
        {
            // Skip raw_anon path if variable has a class hint (e.g., from @:derive(Clone))
            if self.raw_anon_symbols.contains(obj_sym)
                && !self.register_class_hints.contains_key(&obj_reg)
                && !self.monomorphized_var_types.contains_key(obj_sym)
            {
                return self.raw_anon_reflect_field_read(obj_reg, *field, expr.ty);
            }
        }

        // Recover an invalid receiver type (cross-module the typechecker
        // can leave a local unresolved, e.g. `var b = Bytes.ofString(s)`)
        // from the object variable's symbol type, then a class hint —
        // an unresolved receiver has no class handle for field dispatch.
        let receiver_ty = if receiver_ty == TypeId::invalid() {
            let from_symbol = if let HirExprKind::Variable { symbol, .. } = &object.kind {
                self.symbol_table
                    .get_symbol(*symbol)
                    .map(|s| s.type_id)
                    .filter(|t| *t != TypeId::invalid())
            } else {
                None
            };
            // Class name from either a register hint or the object
            // variable's tracked stdlib class (`monomorphized_var_types`,
            // set by `detect_stdlib_class_from_call` for factory results
            // like `Bytes.ofString`).
            let hint_name: Option<String> = self
                .register_class_hints
                .get(&obj_reg)
                .cloned()
                .or_else(|| {
                    if let HirExprKind::Variable { symbol, .. } = &object.kind {
                        self.monomorphized_var_types.get(symbol).cloned()
                    } else {
                        None
                    }
                });
            from_symbol
                .or_else(|| self.find_class_type_by_name(hint_name.as_deref()?))
                .unwrap_or(receiver_ty)
        } else {
            receiver_ty
        };

        // When receiver is Dynamic but has a class hint (e.g., from @:derive(Clone)),
        // resolve receiver_ty to the actual class type for correct GEP field access.
        let receiver_ty = {
            let is_dynamic = {
                let type_table = self.type_table;
                type_table
                    .get(receiver_ty)
                    .map_or(false, |t| matches!(t.kind, TypeKind::Dynamic))
            };
            if is_dynamic {
                if let Some(class_hint) = self.register_class_hints.get(&obj_reg).cloned() {
                    // Find the class type by name
                    let class_type = self.find_class_type_by_name(&class_hint);
                    class_type.unwrap_or(receiver_ty)
                } else {
                    receiver_ty
                }
            } else {
                receiver_ty
            }
        };

        // Reference-class property on an unresolved receiver: when the
        // local's type stayed invalid cross-module but a class hint
        // survives (e.g. `Bytes.ofString(...)` result), resolve the
        // property directly against the stdlib mapping by class name so
        // it doesn't fall through to a same-named getter on an
        // unrelated class.
        if receiver_ty == TypeId::invalid() {
            if let Some(result) =
                self.try_stdlib_property_by_hint(obj_reg, &object.kind, *field, expr.ty)
            {
                return Some(result);
            }
        }

        let result = self.lower_field_access(obj_reg, *field, receiver_ty, expr.ty);
        debug!(
            "[Field expression] lower_field_access returned {:?}",
            result
        );
        result
    }

    pub(crate) fn lower_index_expr(&mut self, expr: &HirExpr) -> Option<IrId> {
        let HirExprKind::Index { object, index } = &expr.kind else {
            unreachable!("lower_index_expr on a non-Index expression")
        };
        let obj_reg = self.lower_expression(object)?;

        // SIMD vector lane extraction: detect when the object is a Vector type
        // (e.g. SIMD4f) and emit a direct VectorExtract instead of going through
        // the heap-array `haxe_array_get_ptr` path. This is required because
        // SIMD4f abstracts lower to v128 register values, not heap pointers.
        if let Some(IrType::Vector { element, count }) = self.builder.get_register_type(obj_reg) {
            let elem_ty = (*element).clone();
            let lane_count = count;
            // Constant lane index: emit VectorExtract directly with that lane.
            if let HirExprKind::Literal(HirLiteral::Int(lane_val)) = &index.kind {
                if *lane_val >= 0 && (*lane_val as usize) < lane_count as usize {
                    let lane = *lane_val as u8;
                    let extracted =
                        self.builder
                            .build_vector_extract(obj_reg, lane, elem_ty.clone())?;
                    // Haxe `Float` is f64. F32 lanes always widen to F64 so downstream
                    // consumers (string concat, float_to_string, arithmetic) see the
                    // expected Float type. The HIR expr.ty is often Dynamic (@:coreType
                    // abstracts erase to Dynamic), so we cannot rely on it.
                    if matches!(elem_ty, IrType::F32) {
                        return self.builder.build_cast(extracted, IrType::F32, IrType::F64);
                    }
                    // Same for narrow int lanes: Haxe `Int` is i32, so
                    // widen with the accessor's SIGNED contract (the MIR
                    // int cast zero-extends on Cranelift; `(x<<s)>>s`
                    // arithmetic recovers the sign, and folds away
                    // under a following `& 0xFF`). A raw i8 register
                    // here otherwise degrades downstream typing.
                    if matches!(elem_ty, IrType::I8 | IrType::I16) {
                        let shift = if matches!(elem_ty, IrType::I8) {
                            24
                        } else {
                            16
                        };
                        let widened =
                            self.builder
                                .build_cast(extracted, elem_ty.clone(), IrType::I32)?;
                        let sh = self.builder.build_const(IrValue::I32(shift))?;
                        let shl = self.builder.build_binop(BinaryOp::Shl, widened, sh)?;
                        return self.builder.build_binop(BinaryOp::Shr, shl, sh);
                    }
                    return Some(extracted);
                }
            }
            // Non-constant lane on a vector: not yet supported via direct
            // VectorExtract (which requires a constant lane). Fall through to
            // the runtime SIMD4f_extract MIR wrapper path below — but that
            // path also currently only supports lane 0. For now we emit a
            // diagnostic-friendly fallback by lowering the lane and routing
            // through the wrapper, which will be improved when the wrapper
            // gains a runtime lane switch.
            let _ = lane_count;
        }

        let idx_reg = self.lower_expression(index)?;
        self.lower_index_access(obj_reg, idx_reg, expr.ty)
    }
}
