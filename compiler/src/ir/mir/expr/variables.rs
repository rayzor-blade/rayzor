//! Variable references: locals, statics, and a class or enum name used as a value.

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
    pub(crate) fn lower_variable_expr(&mut self, expr: &HirExpr) -> Option<IrId> {
        let HirExprKind::Variable { symbol, .. } = &expr.kind else {
            unreachable!("lower_variable_expr on a non-Variable expression")
        };
        if std::env::var_os("RAYZOR_GLOBALS_DEBUG").is_some() {
            if let Some(n) = self
                .symbol_table
                .get_symbol(*symbol)
                .and_then(|sy| self.string_interner.get(sy.name))
            {
                if n.starts_with("PLAIN") {
                    eprintln!("[globals] VAR-ARM entry: {}", n);
                }
            }
        }
        // Check if this is a class/enum used as a value (e.g., Type.getClassName(Animal))
        // Must be before function reference check since class symbols may also map to constructors
        if let Some(sym) = self.symbol_table.get_symbol(*symbol) {
            use crate::tast::SymbolKind;
            if sym.kind == SymbolKind::Class {
                // Class runtime type_id = SymbolId.as_raw() (matches TypeId::from_raw in tast_to_hir)
                let runtime_type_id = sym.id.as_raw() as i64;
                return self.builder.build_const(IrValue::I64(runtime_type_id));
            }
            if sym.kind == SymbolKind::Enum {
                let runtime_type_id = self.enum_runtime_id(sym.id) as i64;
                return self.builder.build_const(IrValue::I64(runtime_type_id));
            }
        }

        // Check if this symbol is a function reference (local or external)
        if let Some(func_id) = self.get_function_id(symbol) {
            // Use FunctionRef (not build_function_ptr / IrValue::Function).
            // The cranelift backend's CallIndirect path unifies on
            // closure objects: it loads fn_ptr from *(reg + 0) and
            // env_ptr from *(reg + 8). FunctionRef wraps the bare
            // function in exactly that {fn_ptr, env_ptr=null}
            // layout, so an indirect call against `var f = static_fn`
            // dereferences valid pointers. build_function_ptr only
            // produces the raw fn_addr — an indirect call then
            // reads the first 16 bytes of the function's code as
            // "fn_ptr, env_ptr" and SIGSEGVs.
            return self.builder.build_function_ref(func_id);
        }

        // IMPORTANT: If we're inside a lambda and this is a captured variable,
        // we must RELOAD from the environment on each access. This ensures that:
        // 1. Updates from other threads (e.g., main thread setting `ready = true`)
        //    are visible to the lambda (thread reading `ready` in a while loop)
        // 2. Mutable captured variables have proper by-reference semantics
        //
        // Without this reload, the captured variable would be cached in an SSA register
        // at lambda entry and never refreshed, causing hangs in condition variable patterns.
        if let Some(ref env_layout) = self.current_env_layout {
            if let Some(_field) = env_layout.find_field(*symbol) {
                // This is a captured variable - reload from environment
                let env_ptr = IrId::new(0); // First parameter in lambda is environment pointer
                let loaded = env_layout.load_field(&mut self.builder, env_ptr, *symbol)?;
                debug!(
                    "Reloading captured variable {:?} from environment, got {:?}",
                    symbol, loaded
                );
                // Update symbol_map so subsequent operations use the new value
                self.symbol_map.insert(*symbol, loaded);
                return Some(loaded);
            }
        }

        // Try to get from symbol_map first (local variables, parameters)
        // SPECIAL CASE: If this is a "this" variable by name but not by the synthetic SymbolId(0),
        // redirect the lookup to use the synthetic `this` symbol. This handles implicit `this`
        // references created during AST lowering for field access in class methods/constructors.
        let lookup_symbol = if let Some(sym_info) = self.symbol_table.get_symbol(*symbol) {
            if let Some(name) = self.string_interner.get(sym_info.name) {
                if name == "this" && *symbol != SymbolId::from_raw(0) {
                    // Redirect to synthetic `this` symbol
                    SymbolId::from_raw(0)
                } else {
                    *symbol
                }
            } else {
                *symbol
            }
        } else {
            *symbol
        };

        // Check direct SymbolId match against global_symbol_map first
        if let Some(&gid) = self.global_symbol_map.get(&lookup_symbol) {
            let global_type = self
                .builder
                .module
                .globals
                .get(&gid)
                .map(|g| g.ty.clone())
                .unwrap_or(IrType::Any);
            return self.builder.build_load_global(gid, global_type);
        }

        if let Some(&reg) = self.symbol_map.get(&lookup_symbol) {
            {
                let _n = self
                    .symbol_table
                    .get_symbol(lookup_symbol)
                    .and_then(|s| self.string_interner.get(s.name));
                let _ = _n; // symbol name used for debugging
            }
            // @:move strict-move tracking: prepend a CheckLive guard
            // before every read of a strict-move local so backends /
            // ownership analysis can diagnose use-after-move.
            if self.strict_move_locals.contains(&reg) {
                let loc = self.convert_source_location(&expr.source_location);
                let _ = self.builder.build_check_live(reg, loc);
            }
            // Check if we need to convert the type
            // This handles cases where captured variables are stored as i64 in closure environment
            // but need to be used as their original type (e.g., i32)
            if let Some(actual_type) = self.builder.get_register_type(reg) {
                let expected_type = self.convert_type(expr.ty);

                // If types don't match, consider adding a cast instruction
                // CRITICAL: Do NOT cast from Ptr to smaller types (I32, etc.)
                // This can happen when generic type resolution fails (e.g., Thread<T> where T is unresolved)
                // and the type system incorrectly infers I32 for a class instance pointer
                if actual_type != expected_type {
                    // Skip casts in these cases to preserve actual type:
                    // 1. Actual is pointer, expected is scalar (would truncate pointer)
                    // 2. Actual is String, expected is Ptr(Void) (would lose string data)
                    // 3. Actual is more specific than Ptr(Void)
                    // 4. Actual is Ptr(String), expected is Ptr(Void) - preserve String type info for trace
                    let actual_is_ptr = matches!(&actual_type, IrType::Ptr(_));
                    let expected_is_ptr = matches!(&expected_type, IrType::Ptr(_));
                    let expected_is_void_ptr = matches!(&expected_type, IrType::Ptr(inner) if matches!(**inner, IrType::Void));
                    let actual_is_specific = matches!(
                        &actual_type,
                        IrType::String
                            | IrType::I32
                            | IrType::I64
                            | IrType::F32
                            | IrType::F64
                            | IrType::Bool
                            | IrType::Function { .. }
                    );
                    // Only skip cast for Ptr(String) specifically - NOT for other pointer types like Ptr(U8)
                    // which are used by concurrency primitives (Mutex, Thread, Channel, etc.)
                    let actual_is_string_ptr = matches!(&actual_type, IrType::Ptr(inner) if matches!(**inner, IrType::String));

                    let actual_is_vector = actual_type.is_vector();

                    // Skip casts that lose semantic type info due to TypeParameter erasure.
                    // TypeParameter converts to I64 via convert_type(), but the Let handler
                    // may have resolved the concrete type (String, F64, F32) from the
                    // GenericInstance return type. Casting back to I64 destroys that info,
                    // which is needed by trace dispatch and other type-aware operations.
                    let actual_loses_info_to_i64 =
                        matches!(&actual_type, IrType::String | IrType::F64 | IrType::F32)
                            && expected_type == IrType::I64;

                    // I64→I32 narrowing would truncate high bits (e.g., function pointers
                    // from CC.getSymbol on 64-bit platforms). Extern functions return I64
                    // but Haxe `Int` maps to I32; preserve the full I64 value.
                    let actual_i64_expected_i32 =
                        actual_type == IrType::I64 && expected_type == IrType::I32;

                    let should_skip_cast = (actual_is_ptr && !expected_is_ptr)  // pointer to scalar
                        || (actual_is_specific && expected_is_void_ptr)          // specific type to void pointer
                        || (actual_is_string_ptr && expected_is_void_ptr)        // Ptr(String) to Ptr(Void)
                        || actual_is_vector // vector types (SIMD) should never be cast
                        || actual_loses_info_to_i64 // TypeParameter erasure would lose concrete type
                        || actual_i64_expected_i32; // I64→I32 truncation would lose high bits

                    if should_skip_cast {
                        debug!(
                            "Variable type mismatch - symbol={:?}, actual: {:?}, expected: {:?}, SKIPPING cast (would lose type info)",
                            symbol, actual_type, expected_type
                        );
                        Some(reg)
                    } else {
                        debug!(
                            "Variable type mismatch - symbol={:?}, actual: {:?}, expected: {:?}, inserting cast",
                            symbol, actual_type, expected_type
                        );
                        let cast_reg = self.builder.build_cast(reg, actual_type, expected_type);
                        // Propagate class hint from source to cast register
                        if let Some(cast_id) = cast_reg {
                            if let Some(hint) = self.register_class_hints.get(&reg).cloned() {
                                self.register_class_hints.insert(cast_id, hint);
                            }
                        }
                        cast_reg
                    }
                } else {
                    Some(reg)
                }
            } else {
                // No type info, return as-is
                Some(reg)
            }
        } else {
            // Symbol not in local scope - check if it's a class field
            // If so, we need to access it via 'this' pointer

            // First check field_index_map - this is more reliable than SymbolKind::Field
            // because field symbols may be registered with SymbolKind::Variable
            let field_entry = self.field_index_map.get(symbol).copied().or_else(|| {
                // Name-based fallback: SymbolIds differ between compilation contexts
                // (e.g., ArrayIterator.current in stdlib vs user code)
                if let Some(sym_info) = self.symbol_table.get_symbol(*symbol) {
                    if let Some(this_type) = self.current_this_type {
                        let field_name = sym_info.name;
                        self.resolve_field_index_by_name(field_name, this_type)
                    } else {
                        None
                    }
                } else {
                    None
                }
            });
            if let Some((field_class_type, _field_idx)) = field_entry {
                // Get 'this' pointer (SymbolId(0) is the special 'this' mapping)
                if let Some(&this_reg) = self.symbol_map.get(&SymbolId::from_raw(0)) {
                    // Use current_this_type if available, otherwise use field_class_type
                    let owner_type = self.current_this_type.unwrap_or(field_class_type);
                    return self.lower_field_access(this_reg, *symbol, owner_type, expr.ty);
                }
            }

            // Fallback: check if symbol table says it's a field or enum variant
            if let Some(sym) = self.symbol_table.get_symbol(*symbol) {
                use crate::tast::SymbolKind;
                if sym.kind == SymbolKind::Field {
                    if let Some(&this_reg) = self.symbol_map.get(&SymbolId::from_raw(0)) {
                        if let Some(owner_type) = self.current_this_type {
                            return self.lower_field_access(this_reg, *symbol, owner_type, expr.ty);
                        }
                    }
                } else if sym.kind == SymbolKind::EnumVariant {
                    // Enum variant without parameters - return its discriminant value.
                    // Use expr.ty to resolve the parent enum — symbol-based lookup
                    // can match the wrong parent when multiple enums share variant names
                    // (e.g., Option.None vs DisplayMode.None).
                    let parent_enum_id = self
                        .resolve_enum_symbol(expr.ty)
                        .or_else(|| self.symbol_table.find_parent_enum_for_constructor(*symbol));
                    if let Some(parent_enum_id) = parent_enum_id {
                        if let Some(variants) = self.symbol_table.get_enum_variants(parent_enum_id)
                        {
                            // Find the index of this variant by SymbolId or by name
                            // (generic instantiation may create different SymbolIds)
                            let variant_name = self.string_interner.get(sym.name).unwrap_or("");
                            for (idx, variant_id) in variants.iter().enumerate() {
                                let id_match = *variant_id == *symbol;
                                let name_match = !id_match
                                    && self
                                        .symbol_table
                                        .get_symbol(*variant_id)
                                        .and_then(|vs| self.string_interner.get(vs.name))
                                        .map_or(false, |vn| vn == variant_name);
                                if id_match || name_match {
                                    // If enum has parameterized variants, all variants must be boxed
                                    if self.enum_is_boxed(parent_enum_id) {
                                        return self.build_boxed_enum_tag_only(idx as i32);
                                    }
                                    return self.builder.build_const(IrValue::I64(idx as i64));
                                }
                            }
                        }
                    }
                    // If we can't find the variant info, try to get the discriminant from the type
                    debug!("EnumVariant {:?} - could not find discriminant", symbol);
                }
            }

            if std::env::var_os("RAYZOR_GLOBALS_DEBUG").is_some() {
                if let Some(n) = self
                    .symbol_table
                    .get_symbol(*symbol)
                    .and_then(|sy| self.string_interner.get(sy.name))
                {
                    if n.starts_with("PLAIN") {
                        eprintln!("[globals] VAR-ARM reached global check: {}", n);
                    }
                }
            }
            // Check if this is a global variable (static class field, module-level var)
            if let Some(&global_id) = self.global_symbol_map.get(symbol) {
                debug!(
                    "[GLOBAL ACCESS] Found global {:?} -> {:?}",
                    symbol, global_id
                );
                // Which global id does this READ target? Compare against the
                // id the owning module's __init__ STORES to: if they differ,
                // the read and the initialiser are talking about different
                // slots and the value will come back empty/zero.
                if std::env::var_os("RAYZOR_GLOBALS_DEBUG").is_some() {
                    let nm = self
                        .symbol_table
                        .get_symbol(*symbol)
                        .and_then(|sy| self.string_interner.get(sy.name))
                        .unwrap_or("<?>");
                    let gname = self
                        .builder
                        .module
                        .globals
                        .get(&global_id)
                        .map(|g| g.name.clone())
                        .unwrap_or_else(|| "<not-in-table>".to_string());
                    eprintln!("[globals] READ {} -> @g{} ({})", nm, global_id.0, gname);
                }
                // Load the global variable's value
                // First get the global's type from the module
                let global_type = self
                    .builder
                    .module
                    .globals
                    .get(&global_id)
                    .map(|g| g.ty.clone())
                    .unwrap_or(IrType::Any);
                return self.builder.build_load_global(global_id, global_type);
            }

            // If we get here, we couldn't resolve the variable
            let sym_name = self
                .symbol_table
                .get_symbol(*symbol)
                .and_then(|s| self.string_interner.get(s.name));
            // FQN-FIRST. `global_symbol_map` is per-module and never
            // seeded from imports, so a static declared in another module
            // always misses the exact-SymbolId lookup above and falls into
            // the name scans below — which only ever search THIS module's
            // globals, miss too, and return None. The caller then yields an
            // empty value with no diagnostic: `Consts.PLAIN_STR` from
            // another module reads as "".
            //
            // Imported globals ARE merged into `module.globals` (see the
            // renumbering in compilation.rs), and `IrGlobal::name` carries
            // the qualified form, so an exact qualified-name match resolves
            // them without widening to bare names. See feedback: FQN-first,
            // bare-name lookup IS the defect.
            let qual = self
                .symbol_table
                .get_symbol(*symbol)
                .and_then(|sy| sy.qualified_name)
                .and_then(|q| self.string_interner.get(q));
            if let Some(qn) = qual {
                for global in self.builder.module.globals.values() {
                    if global.name == qn {
                        if std::env::var_os("RAYZOR_GLOBALS_DEBUG").is_some() {
                            eprintln!("[globals] RESOLVED-FQN {} -> @g{}", qn, global.id.0);
                        }
                        return self.builder.build_load_global(global.id, global.ty.clone());
                    }
                }
                // Not in THIS module's table — try modules already lowered.
                if let Some((gid, gty)) = self.external_globals.get(qn).cloned() {
                    // Type comes from the DEFINING module: this module's
                    // table does not contain the global, so looking it up
                    // locally yields IrType::Any and a String loaded as an
                    // untyped slot prints as "<unknown type N>" and then
                    // segfaults.
                    if std::env::var_os("RAYZOR_GLOBALS_DEBUG").is_some() {
                        eprintln!(
                            "[globals] RESOLVED-EXTERNAL {} -> @g{} : {:?}",
                            qn, gid.0, gty
                        );
                    }
                    return self.builder.build_load_global(gid, gty);
                }
                if std::env::var_os("RAYZOR_GLOBALS_DEBUG").is_some() {
                    let have: Vec<String> = self
                        .builder
                        .module
                        .globals
                        .values()
                        .map(|g| format!("@g{}={}", g.id.0, g.name))
                        .collect();
                    eprintln!(
                        "[globals] FQN-MISS qualified={:?} — table has {} entry(ies): {:?}",
                        qn,
                        have.len(),
                        have
                    );
                }
            }
            // Try name-based global lookup as fallback (SymbolIds may differ)
            if let Some(name_str) = sym_name {
                for (&gsym, &gid) in &self.global_symbol_map {
                    if let Some(gsym_info) = self.symbol_table.get_symbol(gsym) {
                        if let Some(gname) = self.string_interner.get(gsym_info.name) {
                            if gname == name_str {
                                if std::env::var_os("RAYZOR_GLOBALS_DEBUG").is_some() {
                                    eprintln!(
                                        "[globals] RESOLVED-BARENAME {} -> @g{}",
                                        name_str, gid.0
                                    );
                                }
                                let global_type = self
                                    .builder
                                    .module
                                    .globals
                                    .get(&gid)
                                    .map(|g| g.ty.clone())
                                    .unwrap_or(IrType::Any);
                                return self.builder.build_load_global(gid, global_type);
                            }
                        }
                    }
                }
                // Suffix matching is a LOOSENING step: `.foo` matches any
                // class's `foo`. Kept as a bridge over SymbolId drift, but
                // announced — if this is what resolved the read, the
                // qualified name was unavailable and the result is a guess.
                for global in self.builder.module.globals.values() {
                    if global.name.ends_with(&format!(".{}", name_str)) || global.name == name_str {
                        if std::env::var_os("RAYZOR_GLOBALS_DEBUG").is_some() {
                            eprintln!(
                                "[globals] RESOLVED-SUFFIX {} -> @g{} ({})",
                                name_str, global.id.0, global.name
                            );
                        }
                        return self.builder.build_load_global(global.id, global.ty.clone());
                    }
                }
            }
            // TOTAL MISS. Previously a `debug!` (off by default) and a
            // silent None, which the caller turns into an empty value —
            // the read succeeds and yields "" or 0. Say so: a cross-module
            // static that resolves to nothing is a defect, not a default.
            // Gated: this also fires for benign stdlib internals (`i64`,
            // `base`, ...) that are resolved by other means downstream, so
            // always-on would bury the real cases in noise. It is the tool
            // to reach for when a cross-module read yields "" or 0.
            if std::env::var_os("RAYZOR_GLOBALS_DEBUG").is_some() {
                eprintln!(
                    "[globals] unresolved variable {:?} (name={:?}, qualified={:?}) \
                     — this read will produce an empty/zero value. If it names a \
                     static in another module, that module's globals were not \
                     visible here.",
                    symbol, sym_name, qual
                );
            }
            None
        }
    }
}
