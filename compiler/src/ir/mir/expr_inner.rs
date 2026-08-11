//! Per-kind expression lowering.

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
    pub(crate) fn lower_expression_inner(&mut self, expr: &HirExpr) -> Option<IrId> {
        // DEBUG: Check if this is Field expression being lowered
        if matches!(&expr.kind, HirExprKind::Field { .. }) {
            debug!("[lower_expression] START - Field expression");
        }

        // Set source location for debugging
        self.builder
            .set_source_location(self.convert_source_location(&expr.source_location));

        let result = match &expr.kind {
            HirExprKind::Literal(lit) => self.lower_literal(lit, expr.ty),

            HirExprKind::Variable { symbol, .. } => {
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
                                let cast_reg =
                                    self.builder.build_cast(reg, actual_type, expected_type);
                                // Propagate class hint from source to cast register
                                if let Some(cast_id) = cast_reg {
                                    if let Some(hint) = self.register_class_hints.get(&reg).cloned()
                                    {
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
                                    return self.lower_field_access(
                                        this_reg, *symbol, owner_type, expr.ty,
                                    );
                                }
                            }
                        } else if sym.kind == SymbolKind::EnumVariant {
                            // Enum variant without parameters - return its discriminant value.
                            // Use expr.ty to resolve the parent enum — symbol-based lookup
                            // can match the wrong parent when multiple enums share variant names
                            // (e.g., Option.None vs DisplayMode.None).
                            let parent_enum_id = self.resolve_enum_symbol(expr.ty).or_else(|| {
                                self.symbol_table.find_parent_enum_for_constructor(*symbol)
                            });
                            if let Some(parent_enum_id) = parent_enum_id {
                                if let Some(variants) =
                                    self.symbol_table.get_enum_variants(parent_enum_id)
                                {
                                    // Find the index of this variant by SymbolId or by name
                                    // (generic instantiation may create different SymbolIds)
                                    let variant_name =
                                        self.string_interner.get(sym.name).unwrap_or("");
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
                                            return self
                                                .builder
                                                .build_const(IrValue::I64(idx as i64));
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
                                return self
                                    .builder
                                    .build_load_global(global.id, global.ty.clone());
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
                            if global.name.ends_with(&format!(".{}", name_str))
                                || global.name == name_str
                            {
                                if std::env::var_os("RAYZOR_GLOBALS_DEBUG").is_some() {
                                    eprintln!(
                                        "[globals] RESOLVED-SUFFIX {} -> @g{} ({})",
                                        name_str, global.id.0, global.name
                                    );
                                }
                                return self
                                    .builder
                                    .build_load_global(global.id, global.ty.clone());
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

            HirExprKind::Field { object, field } => {
                // Check if this is an enum variant access (e.g., Color.Red)
                // In that case, the object is an Enum type symbol, not a value
                if let HirExprKind::Variable { symbol, .. } = &object.kind {
                    if let Some(sym) = self.symbol_table.get_symbol(*symbol) {
                        use crate::tast::SymbolKind;
                        if sym.kind == SymbolKind::Enum {
                            // This is an enum variant access - get the variant discriminant
                            let enum_name =
                                self.string_interner.get(sym.name).unwrap_or("<unknown>");
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
                ) && self.get_drop_behavior(object.ty)
                    == DropBehavior::AutoDrop;

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
                                        let idx_const = self
                                            .builder
                                            .build_const(IrValue::I64(*gep_idx as i64))?;
                                        let field_ptr = self.builder.build_gep(
                                            obj_reg,
                                            vec![idx_const],
                                            field_ir_ty.clone(),
                                        )?;
                                        return Some(
                                            self.builder.build_load(field_ptr, field_ir_ty)?,
                                        );
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
                                        let idx_val = self
                                            .builder
                                            .build_const(IrValue::I32(*src_idx as i32))?;
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

            HirExprKind::Index { object, index } => {
                let obj_reg = self.lower_expression(object)?;

                // SIMD vector lane extraction: detect when the object is a Vector type
                // (e.g. SIMD4f) and emit a direct VectorExtract instead of going through
                // the heap-array `haxe_array_get_ptr` path. This is required because
                // SIMD4f abstracts lower to v128 register values, not heap pointers.
                if let Some(IrType::Vector { element, count }) =
                    self.builder.get_register_type(obj_reg)
                {
                    let elem_ty = (*element).clone();
                    let lane_count = count;
                    // Constant lane index: emit VectorExtract directly with that lane.
                    if let HirExprKind::Literal(HirLiteral::Int(lane_val)) = &index.kind {
                        if *lane_val >= 0 && (*lane_val as usize) < lane_count as usize {
                            let lane = *lane_val as u8;
                            let extracted = self.builder.build_vector_extract(
                                obj_reg,
                                lane,
                                elem_ty.clone(),
                            )?;
                            // Haxe `Float` is f64. F32 lanes always widen to F64 so downstream
                            // consumers (string concat, float_to_string, arithmetic) see the
                            // expected Float type. The HIR expr.ty is often Dynamic (@:coreType
                            // abstracts erase to Dynamic), so we cannot rely on it.
                            if matches!(elem_ty, IrType::F32) {
                                return self.builder.build_cast(
                                    extracted,
                                    IrType::F32,
                                    IrType::F64,
                                );
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
                                let widened = self.builder.build_cast(
                                    extracted,
                                    elem_ty.clone(),
                                    IrType::I32,
                                )?;
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

            HirExprKind::Call {
                callee,
                args,
                is_method,
                type_args: hir_type_args,
                // Carried from TAST; the shape probes below still derive the
                // target themselves until they are replaced by a match on this.
                target: _resolved_target,
            } => {
                // RAYZOR_PROBE_CALLTARGET=1 tabulates (target, callee shape) so
                // the carried target can be checked against what the shape
                // probes below discriminate on, before anything dispatches on it.
                if std::env::var_os("RAYZOR_PROBE_CALLTARGET").is_some() {
                    let t = match _resolved_target {
                        crate::ir::hir::CallTarget::Function => "Function",
                        crate::ir::hir::CallTarget::Method { .. } => "Method",
                        crate::ir::hir::CallTarget::Static { .. } => "Static",
                    };
                    let shape = match &callee.kind {
                        HirExprKind::Field { object, .. } => match &object.kind {
                            HirExprKind::Variable { .. } => "Field(Variable)",
                            _ => "Field(other)",
                        },
                        HirExprKind::Variable { .. } => "Variable",
                        HirExprKind::Super => "Super",
                        _ => "other",
                    };
                    eprintln!("[calltarget] {} {} is_method={}", t, shape, is_method);
                }
                // @:shader wgsl() — intercept at Call entry point
                if let HirExprKind::Field { object, field } = &callee.kind {
                    let field_name_check = self
                        .symbol_table
                        .get_symbol(*field)
                        .and_then(|s| self.string_interner.get(s.name));
                    let is_wgsl = field_name_check == Some("wgsl");
                    if is_wgsl {
                        if let HirExprKind::Variable {
                            symbol: class_sym, ..
                        } = &object.kind
                        {
                            let is_shader = self
                                .symbol_table
                                .get_symbol(*class_sym)
                                .map(|s| s.flags.is_shader())
                                .unwrap_or(false);
                            if is_shader {
                                for (_tid, decl) in self.current_hir_types.iter() {
                                    if let crate::ir::hir::HirTypeDecl::Class(c) = decl {
                                        if c.symbol_id == *class_sym {
                                            let tt = self.type_table;
                                            match crate::codegen::wgsl_transpiler::transpile_shader_from_hir(
                                                c, self.symbol_table, tt, self.string_interner, self.current_hir_types,
                                            ) {
                                                Ok(wgsl) => return self.builder.build_const(IrValue::String(wgsl)),
                                                Err(e) => return self.builder.build_const(IrValue::String(format!("/* WGSL error: {} */", e))),
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Reset call_label for tracing which path generates the call
                self.builder.call_label = Some("CALL_START".to_string());
                let result_type = self.convert_type(expr.ty);

                // Update the caller's shadow-stack frame to this call-site line/col so
                // the trace shows WHERE the call was made, not the function definition line.
                let call_loc = expr.source_location;
                if call_loc.is_valid() && call_loc.line > 0 {
                    let update_loc_fn = self.get_or_register_extern_function(
                        "rayzor_update_call_frame_location",
                        vec![IrType::I32, IrType::I32],
                        IrType::Void,
                    );
                    if let (Some(line_c), Some(col_c)) = (
                        self.builder.build_const(IrValue::I32(call_loc.line as i32)),
                        self.builder
                            .build_const(IrValue::I32(call_loc.column as i32)),
                    ) {
                        self.builder.build_call_direct(
                            update_loc_fn,
                            vec![line_c, col_c],
                            IrType::Void,
                        );
                    }
                }

                // Convert HIR type_args to IrType for use in CallDirect
                let converted_hir_type_args: Vec<IrType> = hir_type_args
                    .iter()
                    .map(|&ty_id| self.convert_type(ty_id))
                    .collect();

                debug!(
                    "[CALL] expr.ty={:?}, result_type={:?}, is_method={}",
                    expr.ty, result_type, is_method
                );

                // @:async method dispatch: .await(), .poll(), .isReady()
                // on registers known to hold Future handles from async function calls.
                // MethodCall pattern: callee = Variable(method_symbol), args[0] = receiver
                if *is_method {
                    // Get method name from callee symbol
                    let async_method_sym = match &callee.kind {
                        HirExprKind::Variable { symbol, .. } => Some(*symbol),
                        HirExprKind::Field { field, .. } => Some(*field),
                        _ => None,
                    };
                    // Get receiver symbol from first arg (MethodCall puts receiver as args[0])
                    let receiver_sym_from_args = args.first().and_then(|a| {
                        if let HirExprKind::Variable { symbol, .. } = &a.kind {
                            Some(*symbol)
                        } else {
                            None
                        }
                    });
                    if let (Some(method_sym), Some(recv_sym)) =
                        (async_method_sym, receiver_sym_from_args)
                    {
                        let receiver_reg = self.symbol_map.get(&recv_sym).copied();
                        if let Some(recv_reg) = receiver_reg {
                            if self.async_result_registers.contains(&recv_reg) {
                                let method_name = self
                                    .symbol_table
                                    .get_symbol(method_sym)
                                    .and_then(|s| self.string_interner.get(s.name));
                                if let Some(method) = method_name {
                                    let ext_name = match method {
                                        "await" => Some("rayzor_future_await"),
                                        "poll" => Some("rayzor_future_poll"),
                                        "isReady" => Some("rayzor_future_is_ready"),
                                        _ => None,
                                    };
                                    if let Some(extern_name) = ext_name {
                                        // Look up the extern directly (declared by ensure_future_externs)
                                        let func_id = self
                                            .builder
                                            .module
                                            .extern_functions
                                            .iter()
                                            .find(|(_, f)| f.name == extern_name)
                                            .map(|(id, _)| *id);
                                        if let Some(func_id) = func_id {
                                            let ret_ty = if method == "isReady" {
                                                IrType::Bool
                                            } else {
                                                IrType::Ptr(Box::new(IrType::U8))
                                            };
                                            return self.builder.build_call_direct(
                                                func_id,
                                                vec![recv_reg],
                                                ret_ty,
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Static synthetic calls resolved as Variable — find parent class
                if let HirExprKind::Variable { symbol, .. } = &callee.kind {
                    let callee_name = self
                        .symbol_table
                        .get_symbol(*symbol)
                        .and_then(|s| self.string_interner.get(s.name))
                        .map(|s| s.to_string());
                    // @:gpuStruct synthetic static methods: gpuDef/gpuSize/gpuAlignment
                    if matches!(
                        callee_name.as_deref(),
                        Some("gpuDef")
                            | Some("gpuSize")
                            | Some("gpuAlignment")
                            | Some("gpuVertexLayout")
                            | Some("wgsl")
                    ) {
                        for (tid, decl) in self.current_hir_types.iter() {
                            if let crate::ir::hir::HirTypeDecl::Class(c) = decl {
                                let sym_flags = self
                                    .symbol_table
                                    .get_symbol(c.symbol_id)
                                    .map(|s| s.flags)
                                    .unwrap_or(SymbolFlags::NONE);
                                let is_gpu_struct = sym_flags.is_gpu_struct();
                                let is_shader = sym_flags.is_shader();
                                if !is_gpu_struct && !is_shader {
                                    continue;
                                }
                                // @:shader wgsl() — handle before has_method check
                                // (synthetic wgsl() may not be in HIR methods list)
                                if is_shader && callee_name.as_deref() == Some("wgsl") {
                                    let type_table = self.type_table;
                                    match crate::codegen::wgsl_transpiler::transpile_shader_from_hir(
                                        c,
                                        self.symbol_table,
                                        type_table,
                                        self.string_interner,
                                        self.current_hir_types,
                                    ) {
                                        Ok(wgsl_source) => {
                                            return self
                                                .builder
                                                .build_const(IrValue::String(wgsl_source));
                                        }
                                        Err(e) => {
                                            return self.builder.build_const(IrValue::String(
                                                format!("/* WGSL error: {} */", e),
                                            ));
                                        }
                                    }
                                }
                                let has_method =
                                    c.methods.iter().any(|m| m.function.symbol_id == *symbol);
                                if has_method {
                                    // Find canonical TypeId
                                    let canonical_tid = {
                                        let type_table = self.type_table;
                                        type_table.get(*tid).and_then(|_| Some(*tid)).or_else(
                                            || {
                                                type_table.iter().find_map(|(_, t)| {
                                                    if let crate::tast::core::TypeKind::Class {
                                                        symbol_id: sid,
                                                        ..
                                                    } = &t.kind
                                                    {
                                                        if *sid == c.symbol_id {
                                                            return Some(t.id);
                                                        }
                                                    }
                                                    None
                                                })
                                            },
                                        )
                                    };
                                    // Handle wgsl() on @:shader classes BEFORE layout check
                                    if is_shader && callee_name.as_deref() == Some("wgsl") {
                                        let type_table = self.type_table;
                                        match crate::codegen::wgsl_transpiler::transpile_shader_from_hir(
                                            c,
                                            self.symbol_table,
                                            type_table,
                                            self.string_interner,
                                            self.current_hir_types,
                                        ) {
                                            Ok(wgsl_source) => {
                                                return self.builder.build_const(
                                                    IrValue::String(wgsl_source),
                                                );
                                            }
                                            Err(e) => {
                                                return self.builder.build_const(
                                                    IrValue::String(format!("/* WGSL error: {} */", e)),
                                                );
                                            }
                                        }
                                    }

                                    if let Some(real_tid) = canonical_tid {
                                        if let Some(layout) =
                                            self.get_or_compute_gpu_struct_layout(real_tid)
                                        {
                                            match callee_name.as_deref().unwrap() {
                                                "gpuDef" => {
                                                    let mut full = String::new();
                                                    for dep in &layout.dep_typedefs {
                                                        full.push_str(dep);
                                                    }
                                                    full.push_str(&layout.msl_typedef);
                                                    return self
                                                        .builder
                                                        .build_const(IrValue::String(full));
                                                }
                                                "gpuSize" => {
                                                    return self.builder.build_const(IrValue::I32(
                                                        layout.total_size as i32,
                                                    ));
                                                }
                                                "gpuAlignment" => {
                                                    return self.builder.build_const(IrValue::I32(
                                                        layout.alignment as i32,
                                                    ));
                                                }
                                                "gpuVertexLayout" => {
                                                    // Return "stride:offset1,fmt1,loc1;offset2,fmt2,loc2;..."
                                                    // Parsed by pure Haxe VertexLayout class
                                                    let mut parts = Vec::new();
                                                    parts.push(format!("{}", layout.total_size));
                                                    for (i, f) in layout.fields.iter().enumerate() {
                                                        parts.push(format!(
                                                            "{},{},{}",
                                                            f.byte_offset, f.vertex_format, i
                                                        ));
                                                    }
                                                    let encoded = parts.join(";");
                                                    return self
                                                        .builder
                                                        .build_const(IrValue::String(encoded));
                                                }
                                                "wgsl" => {
                                                    // @:shader class — transpile HIR to WGSL
                                                    let type_table = self.type_table;
                                                    match crate::codegen::wgsl_transpiler::transpile_shader_from_hir(
                                                        c,
                                                        self.symbol_table,
                                                        type_table,
                                                        self.string_interner,
                                                        self.current_hir_types,
                                                    ) {
                                                        Ok(wgsl_source) => {
                                                            return self.builder.build_const(
                                                                IrValue::String(wgsl_source),
                                                            );
                                                        }
                                                        Err(e) => {
                                                            return self.builder.build_const(
                                                                IrValue::String(format!("/* WGSL error: {} */", e)),
                                                            );
                                                        }
                                                    }
                                                }
                                                _ => {}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if callee_name.as_deref() == Some("cdef") {
                        // Find the @:cstruct class that has this cdef method
                        for (tid, decl) in self.current_hir_types.iter() {
                            if let crate::ir::hir::HirTypeDecl::Class(c) = decl {
                                let is_cstruct = self
                                    .symbol_table
                                    .get_symbol(c.symbol_id)
                                    .map(|s| s.flags.is_cstruct())
                                    .unwrap_or(false);
                                if !is_cstruct {
                                    continue;
                                }
                                // Check if this class has a method with our cdef symbol
                                let has_cdef =
                                    c.methods.iter().any(|m| m.function.symbol_id == *symbol);
                                if has_cdef {
                                    // HIR TypeId may not be in type_table — find canonical TypeId by symbol
                                    let canonical_tid = {
                                        let type_table = self.type_table;
                                        type_table.get(*tid).and_then(|_| Some(*tid)).or_else(
                                            || {
                                                // Scan type_table for a Class with matching symbol_id
                                                type_table.iter().find_map(|(_, t)| {
                                                    if let crate::tast::core::TypeKind::Class {
                                                        symbol_id: sid,
                                                        ..
                                                    } = &t.kind
                                                    {
                                                        if *sid == c.symbol_id {
                                                            return Some(t.id);
                                                        }
                                                    }
                                                    None
                                                })
                                            },
                                        )
                                    };
                                    if let Some(real_tid) = canonical_tid {
                                        if let Some(layout) =
                                            self.get_or_compute_cstruct_layout(real_tid)
                                        {
                                            return self
                                                .builder
                                                .build_const(IrValue::String(layout.cdef_string));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                if let HirExprKind::Variable { symbol, .. } = &callee.kind {
                    let vname = self
                        .symbol_table
                        .get_symbol(*symbol)
                        .and_then(|s| self.string_interner.get(s.name))
                        .unwrap_or("?");
                    debug!(
                        "[CALL-VAR] callee='{}', is_method={}, args.len()={}",
                        vname,
                        is_method,
                        args.len()
                    );

                    // @:derive(Hash) synthetic hashCode() — intercept Variable callee path
                    // Instance method calls are desugared to Variable callee with receiver as first arg
                    if vname == "hashCode" && *is_method && args.len() == 1 {
                        let receiver = &args[0];
                        let class_sym = {
                            let type_table = self.type_table;
                            type_table.get(receiver.ty).and_then(|t| {
                                if let TypeKind::Class { symbol_id, .. } = &t.kind {
                                    if self.derive_hash_classes.contains(symbol_id) {
                                        Some(*symbol_id)
                                    } else {
                                        None
                                    }
                                } else {
                                    None
                                }
                            })
                        };
                        if let Some(sym) = class_sym {
                            return self.lower_derived_hash_code(receiver, sym);
                        }
                    }

                    // @:derive(Clone) synthetic clone() — deep copy
                    if vname == "clone" && *is_method && args.len() == 1 {
                        let receiver = &args[0];
                        let class_sym = {
                            let type_table = self.type_table;
                            type_table.get(receiver.ty).and_then(|t| {
                                if let TypeKind::Class { symbol_id, .. } = &t.kind {
                                    if self.derive_clone_classes.contains(symbol_id) {
                                        Some(*symbol_id)
                                    } else {
                                        None
                                    }
                                } else {
                                    None
                                }
                            })
                        };
                        if let Some(sym) = class_sym {
                            return self.lower_derived_clone(receiver, sym);
                        }
                    }

                    // @:derive(Debug) synthetic toString()
                    if vname == "toString" && *is_method && args.len() == 1 {
                        let receiver = &args[0];
                        let class_sym = {
                            let type_table = self.type_table;
                            type_table.get(receiver.ty).and_then(|t| {
                                if let TypeKind::Class { symbol_id, .. } = &t.kind {
                                    if self.derive_debug_classes.contains(symbol_id) {
                                        Some(*symbol_id)
                                    } else {
                                        None
                                    }
                                } else {
                                    None
                                }
                            })
                        };
                        if let Some(sym) = class_sym {
                            return self.lower_derived_to_string(receiver, sym);
                        }
                    }

                    // Array<Float>.push on WASM32 (Variable-callee shape, where the
                    // receiver is desugared to args[0] and the value to args[1]).
                    // The generic `array_push` MIR wrapper takes an I64 value param,
                    // but IrType::I64 lowers to a WASM `i32`, so a Float value (the
                    // f64 bit-pattern bitcast to i64) loses its high 32 bits and reads
                    // back as 0/garbage. Route Float-element pushes through the f64
                    // runtime entry — value stays F64 (→ WASM f64, full 8 bytes). This
                    // matches the array-literal lowering and is bit-identical on native.
                    if vname == "push" && *is_method && args.len() == 2 {
                        let elem_is_f64 = {
                            let type_table = self.type_table;
                            type_table
                                .get(args[0].ty)
                                .and_then(|t| {
                                    if let TypeKind::Array { element_type } = &t.kind {
                                        Some(*element_type)
                                    } else {
                                        None
                                    }
                                })
                                .map(|et| self.convert_type(et) == IrType::F64)
                                .unwrap_or(false)
                        };
                        if elem_is_f64 {
                            if let (Some(arr_reg), Some(val_reg)) = (
                                self.lower_expression(&args[0]),
                                self.lower_expression(&args[1]),
                            ) {
                                let val_ty = self
                                    .builder
                                    .get_register_type(val_reg)
                                    .unwrap_or(IrType::F64);
                                let val_f64 = if val_ty == IrType::F64 {
                                    val_reg
                                } else {
                                    self.builder
                                        .build_cast(val_reg, val_ty, IrType::F64)
                                        .unwrap_or(val_reg)
                                };
                                let push_fn = self.get_or_register_extern_function(
                                    "haxe_array_push_f64",
                                    vec![IrType::Ptr(Box::new(IrType::I64)), IrType::F64],
                                    IrType::Void,
                                );
                                return self.builder.build_call_direct(
                                    push_fn,
                                    vec![arr_reg, val_f64],
                                    IrType::Void,
                                );
                            }
                        }
                    }

                    // Array.join: the generic array_join runtime treats every
                    // element as a HaxeString pointer, which SIGSEGVs for non-
                    // String element types. Route through haxe_array_join_typed
                    // with the element's type tag so each element is converted
                    // via Std.string first (1=Int 2=Bool 4=Float 5=String 6=Ref).
                    if vname == "join" && *is_method && args.len() == 2 {
                        let elem_tag: i32 = {
                            let type_table = self.type_table;
                            type_table
                                .get(args[0].ty)
                                .and_then(|t| {
                                    if let TypeKind::Array { element_type } = &t.kind {
                                        Some(*element_type)
                                    } else {
                                        None
                                    }
                                })
                                .and_then(|et| type_table.get(et).map(|t| t.kind.clone()))
                                .map(|k| match k {
                                    TypeKind::Int => 1,
                                    TypeKind::Bool => 2,
                                    TypeKind::Float => 4,
                                    TypeKind::String => 5,
                                    _ => 6,
                                })
                                .unwrap_or(5)
                        };
                        if let (Some(arr_reg), Some(sep_reg)) = (
                            self.lower_expression(&args[0]),
                            self.lower_expression(&args[1]),
                        ) {
                            let ptr_void = IrType::Ptr(Box::new(IrType::Void));
                            let tag_reg = self.builder.build_const(IrValue::I32(elem_tag))?;
                            let join_fn = self.get_or_register_extern_function(
                                "haxe_array_join_typed",
                                vec![ptr_void.clone(), ptr_void.clone(), IrType::I32],
                                ptr_void.clone(),
                            );
                            return self.builder.build_call_direct(
                                join_fn,
                                vec![arr_reg, sep_reg, tag_reg],
                                ptr_void,
                            );
                        }
                    }
                }
                {
                    // DEBUG: check callee kind for localhost
                }
                if let HirExprKind::Field { object, field } = &callee.kind {
                    // This is a method call: object.method(args)
                    // The method symbol should be in our function_map (local or external)
                    let method_name_interned = self.symbol_table.get_symbol(*field).map(|s| s.name);
                    let method_name =
                        method_name_interned.and_then(|name| self.string_interner.get(name));
                    let in_local = self.function_map.contains_key(field);
                    let in_external = self.external_function_map.contains_key(field);
                    debug!(
                        "[Method call] method={:?}, field={:?}, in_local={}, in_external={}",
                        method_name, field, in_local, in_external
                    );

                    // @:cstruct synthetic cdef() method — return C typedef string
                    if method_name == Some("cdef") {
                        let obj_type = object.ty;
                        if self.is_cstruct_class(obj_type) {
                            if let Some(layout) = self.get_or_compute_cstruct_layout(obj_type) {
                                return self
                                    .builder
                                    .build_const(IrValue::String(layout.cdef_string));
                            }
                        }
                        // Fallback: for static calls, obj_type may differ from cached TypeId.
                        // Extract symbol_id from obj_type, find matching layout.
                        let obj_sym_id = {
                            let type_table = self.type_table;
                            type_table.get(obj_type).and_then(|t| {
                                if let crate::tast::core::TypeKind::Class { symbol_id, .. } =
                                    &t.kind
                                {
                                    Some(*symbol_id)
                                } else {
                                    None
                                }
                            })
                        };
                        if let Some(sym_id) = obj_sym_id {
                            // Find the cached layout whose class has this symbol_id
                            let cdef_str = self.cstruct_layouts.iter().find_map(|(tid, layout)| {
                                // Check if this type_id's class matches our symbol
                                let type_table = self.type_table;
                                if let Some(t) = type_table.get(*tid) {
                                    if let crate::tast::core::TypeKind::Class {
                                        symbol_id, ..
                                    } = &t.kind
                                    {
                                        if *symbol_id == sym_id {
                                            return Some(layout.cdef_string.clone());
                                        }
                                    }
                                }
                                // Also check via HirTypeDecl
                                for (htid, decl) in self.current_hir_types.iter() {
                                    if *htid == *tid {
                                        if let crate::ir::hir::HirTypeDecl::Class(c) = decl {
                                            if c.symbol_id == sym_id {
                                                return Some(layout.cdef_string.clone());
                                            }
                                        }
                                    }
                                }
                                None
                            });
                            if let Some(cdef) = cdef_str {
                                return self.builder.build_const(IrValue::String(cdef));
                            }
                        }
                    }

                    // Clone/Debug interceptions are in the Variable callee path above

                    // @:derive(Hash) synthetic hashCode() — inline field-based hash computation
                    if method_name == Some("hashCode") && args.is_empty() {
                        let class_sym = {
                            let type_table = self.type_table;
                            type_table.get(object.ty).and_then(|t| {
                                if let TypeKind::Class { symbol_id, .. } = &t.kind {
                                    if self.derive_hash_classes.contains(symbol_id) {
                                        Some(*symbol_id)
                                    } else {
                                        None
                                    }
                                } else {
                                    None
                                }
                            })
                        };
                        if let Some(sym) = class_sym {
                            return self.lower_derived_hash_code(object, sym);
                        }
                    }

                    // @:shader synthetic method — wgsl()
                    if method_name == Some("wgsl") {
                        // Find the @:shader class from the object type
                        let obj_sym = {
                            let type_table = self.type_table;
                            type_table.get(object.ty).and_then(|t| {
                                if let crate::tast::core::TypeKind::Class { symbol_id, .. } =
                                    &t.kind
                                {
                                    Some(*symbol_id)
                                } else {
                                    None
                                }
                            })
                        };
                        let is_shader = obj_sym
                            .and_then(|sid| self.symbol_table.get_symbol(sid))
                            .map(|s| s.flags.is_shader())
                            .unwrap_or(false);
                        if is_shader {
                            // Find the HIR class and transpile
                            for (_tid, decl) in self.current_hir_types.iter() {
                                if let crate::ir::hir::HirTypeDecl::Class(c) = decl {
                                    if Some(c.symbol_id) == obj_sym {
                                        let type_table = self.type_table;
                                        match crate::codegen::wgsl_transpiler::transpile_shader_from_hir(
                                            c, self.symbol_table, type_table, self.string_interner, self.current_hir_types,
                                        ) {
                                            Ok(wgsl) => return self.builder.build_const(IrValue::String(wgsl)),
                                            Err(e) => return self.builder.build_const(IrValue::String(format!("/* WGSL error: {} */", e))),
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // @:gpuStruct synthetic methods — gpuDef/gpuSize/gpuAlignment
                    if matches!(
                        method_name,
                        Some("gpuDef") | Some("gpuSize") | Some("gpuAlignment")
                    ) {
                        let obj_type = object.ty;
                        // Try direct type check first, then fallback via symbol_id
                        let gpu_layout = if self.is_gpu_struct_class(obj_type) {
                            self.get_or_compute_gpu_struct_layout(obj_type)
                        } else {
                            // Static call: obj_type may differ from cached TypeId
                            let obj_sym_id = {
                                let type_table = self.type_table;
                                type_table.get(obj_type).and_then(|t| {
                                    if let crate::tast::core::TypeKind::Class {
                                        symbol_id, ..
                                    } = &t.kind
                                    {
                                        Some(*symbol_id)
                                    } else {
                                        None
                                    }
                                })
                            };
                            obj_sym_id.and_then(|sym_id| {
                                self.gpu_struct_layouts.iter().find_map(|(tid, layout)| {
                                    let type_table = self.type_table;
                                    if let Some(t) = type_table.get(*tid) {
                                        if let crate::tast::core::TypeKind::Class {
                                            symbol_id,
                                            ..
                                        } = &t.kind
                                        {
                                            if *symbol_id == sym_id {
                                                return Some(layout.clone());
                                            }
                                        }
                                    }
                                    for (htid, decl) in self.current_hir_types.iter() {
                                        if *htid == *tid {
                                            if let crate::ir::hir::HirTypeDecl::Class(c) = decl {
                                                if c.symbol_id == sym_id {
                                                    return Some(layout.clone());
                                                }
                                            }
                                        }
                                    }
                                    None
                                })
                            })
                        };
                        if let Some(layout) = gpu_layout {
                            match method_name.unwrap() {
                                "gpuDef" => {
                                    // Return full MSL typedef (deps + own)
                                    let mut full = String::new();
                                    for dep in &layout.dep_typedefs {
                                        full.push_str(dep);
                                    }
                                    full.push_str(&layout.msl_typedef);
                                    return self.builder.build_const(IrValue::String(full));
                                }
                                "gpuSize" => {
                                    return self
                                        .builder
                                        .build_const(IrValue::I32(layout.total_size as i32));
                                }
                                "gpuAlignment" => {
                                    return self
                                        .builder
                                        .build_const(IrValue::I32(layout.alignment as i32));
                                }
                                _ => {}
                            }
                        }
                    }

                    // Check for interface dispatch: if object has interface type,
                    // load the method function pointer from the fat pointer and call indirectly
                    if let Some(iface_sym) = self.get_interface_symbol(object.ty) {
                        let method_name_interned =
                            self.symbol_table.get_symbol(*field).map(|s| s.name);

                        if let Some(method_name_i) = method_name_interned {
                            // Find the method's index in the interface. Resolve
                            // by name (drift-tolerant): the fat-pointer builder
                            // uses the same resolver, so cross-module SymbolId
                            // drift can't leave the call site indexing a
                            // different (truncated) method list than the layout.
                            let method_index = self
                                .resolve_interface_method_names(iface_sym)
                                .and_then(|names| names.iter().position(|n| *n == method_name_i));
                            if std::env::var_os("RAYZOR_IFACE_DEBUG").is_some() {
                                let mn = self
                                    .string_interner
                                    .get(method_name_i)
                                    .unwrap_or("?")
                                    .to_string();
                                let iname = self
                                    .symbol_table
                                    .get_symbol(iface_sym)
                                    .and_then(|s| s.qualified_name)
                                    .and_then(|n| self.string_interner.get(n))
                                    .unwrap_or("?")
                                    .to_string();
                                let names_list =
                                    self.interface_method_names.get(&iface_sym).map(|v| {
                                        v.iter()
                                            .filter_map(|n| self.string_interner.get(*n))
                                            .collect::<Vec<_>>()
                                            .join(",")
                                    });
                                eprintln!(
                                    "[disp] mod={} iface={} method={} idx={:?} names={:?}",
                                    self.builder.module.name, iname, mn, method_index, names_list
                                );
                            }

                            if let Some(idx) = method_index {
                                // Lower the object (fat pointer)
                                let fat_ptr = self.lower_expression(object)?;

                                // Lower arguments
                                let arg_regs: Vec<_> = args
                                    .iter()
                                    .filter_map(|a| self.lower_expression(a))
                                    .collect();

                                // Load object pointer from fat_ptr[0]
                                let obj_ptr = self.builder.build_load(fat_ptr, IrType::I64)?;

                                // Load function pointer from fat_ptr[(idx+1)*8]
                                let fn_offset = self
                                    .builder
                                    .build_const(IrValue::I64(((idx + 1) * 8) as i64))?;
                                let fn_slot = self.builder.build_ptr_add(
                                    fat_ptr,
                                    fn_offset,
                                    IrType::Ptr(Box::new(IrType::U8)),
                                )?;
                                let fn_ptr = self.builder.build_load(fn_slot, IrType::I64)?;

                                // Build call args: self (obj_ptr) + user args
                                let mut call_args = vec![obj_ptr];
                                call_args.extend(arg_regs);

                                // Build signature: (self: Ptr, args...) -> return_type
                                let param_types = {
                                    let mut types = vec![IrType::Ptr(Box::new(IrType::Void))]; // self
                                    for arg in args {
                                        types.push(self.convert_type(arg.ty));
                                    }
                                    types
                                };
                                // Resolve return type from the method's symbol type,
                                // not expr.ty (which may be the interface type instead
                                // of the method's return type in some TAST configurations)
                                let (return_ir_type, resolved_ret_type_id) =
                                    self.resolve_interface_method_return_type_full(*field, expr.ty);
                                // If the HIR `expr.ty` was Dynamic-shaped (a Ptr
                                // after `convert_type`) but the cross-context
                                // map resolved a concrete return type, emit
                                // either a hint (resolved) or a warning
                                // (still missing). The warning fires when the
                                // user would benefit from an explicit
                                // annotation at the binding site.
                                self.emit_iface_return_diagnostic(
                                    *field,
                                    expr.ty,
                                    resolved_ret_type_id,
                                    expr.source_location,
                                );
                                let return_type = Box::new(return_ir_type);
                                let func_signature = IrType::Function {
                                    params: param_types,
                                    return_type,
                                    varargs: false,
                                };

                                let call_result = self.builder.build_call_indirect(
                                    fn_ptr,
                                    call_args,
                                    func_signature,
                                )?;
                                // Track for cross-Let type propagation: when
                                // the iface call result's HIR type was
                                // Dynamic but we re-resolved a concrete
                                // TypeId, store the (register → TypeId) so
                                // the binding site can override the
                                // variable's effective type.
                                if let Some(real_ty) = resolved_ret_type_id {
                                    self.interface_call_result_types
                                        .insert(call_result, real_ty);
                                }
                                return Some(call_result);
                            }
                        }
                    }

                    let maybe_func_id = self
                        .resolve_function_id_with_qualified_fallback(*field)
                        .or_else(|| {
                            method_name_interned
                                .and_then(|name| self.resolve_method_function_id(object.ty, name))
                        })
                        .or_else(|| {
                            // Cross-module STATIC call whose method SymbolId
                            // drifted AND whose forwarded stub carries no
                            // qualified_name (the resolver above bails on its
                            // `?`), in a context where the class's methods were
                            // never registered (so the receiver-type path also
                            // misses). Construct the EXACT fully-qualified name
                            // from the CLASS symbol on the Field's object and
                            // resolve by name — the same mechanism the
                            // interface fat-ptr slots use. This is exact-FQN
                            // construction, not suffix matching: an unresolved
                            // name still errors loudly.
                            //
                            // Statics WITH args already survived through the
                            // param-count name paths; the zero-arg form had no
                            // name-based route at all (Q4Matmul.dumpCensus()
                            // from the llama-chat entry module, E0100).
                            let mname = method_name?;
                            let class_sym = match &object.kind {
                                HirExprKind::Variable { symbol, .. } => *symbol,
                                _ => return None,
                            };
                            let csym = self.symbol_table.get_symbol(class_sym)?;
                            let cqual = csym
                                .qualified_name
                                .and_then(|q| self.string_interner.get(q));
                            let cbare = self.string_interner.get(csym.name);
                            for cname in [cqual, cbare].into_iter().flatten() {
                                let key = format!("{}.{}", cname, mname);
                                if let Some(&fid) = self.external_function_name_map.get(&key) {
                                    return Some(fid);
                                }
                                for (ext_sym, &fid) in &self.external_function_map {
                                    let ext_qual = self
                                        .symbol_table
                                        .get_symbol(*ext_sym)
                                        .and_then(|sy| sy.qualified_name)
                                        .and_then(|q| self.string_interner.get(q));
                                    if ext_qual == Some(key.as_str()) {
                                        return Some(fid);
                                    }
                                }
                            }
                            None
                        });
                    if let Some(func_id) = maybe_func_id {
                        // If the resolved function is from an import module (renumbered to
                        // 100_000+), it's a real user-defined or compiled stdlib function.
                        // Skip the stdlib runtime mapping check — it would incorrectly redirect
                        // user methods like "add" to stdlib functions (e.g., sys_deque_add).
                        // FIRST: Try to route through runtime mapping for extern class methods
                        // Check if there's a runtime mapping using the standard approach
                        // BUT: for String methods with optional params, use param-count-aware lookup
                        // Note: get_stdlib_runtime_info has an internal guard that returns None
                        // for user-defined class receivers, preventing name collisions.
                        let stdlib_info = {
                            let method_name_str = self
                                .symbol_table
                                .get_symbol(*field)
                                .and_then(|s| self.string_interner.get(s.name));

                            // Check if this is a method with optional params that needs param-count-aware lookup
                            if let Some(mn) = method_name_str {
                                if mn == "indexOf" || mn == "lastIndexOf" || mn == "substr" {
                                    // Use param-count-aware lookup for overloaded String methods.
                                    // `substr` has 1-arg (default len) and 2-arg forms registered
                                    // as separate mappings; without param-count dispatch the
                                    // generic name lookup matches the wrong arity and the call
                                    // is lowered against a mismatched signature.
                                    let arg_count = args.len();
                                    debug!(
                                        "[String overload lookup] method={}, arg_count={}",
                                        mn, arg_count
                                    );
                                    self.stdlib_mapping
                                        .find_by_name_and_params("String", mn, arg_count)
                                        .map(|(sig, mapping)| (sig.class, sig.method, mapping))
                                } else if mn == "wait" {
                                    // Lock.wait() has overloads: 0 params (blocking) vs 1 param (with timeout)
                                    let arg_count = args.len();
                                    debug!("[wait lookup] method={}, arg_count={}", mn, arg_count);
                                    self.stdlib_mapping
                                        .find_by_name_and_params("sys_thread_Lock", mn, arg_count)
                                        .or_else(|| {
                                            self.stdlib_mapping.find_by_name_and_params(
                                                "sys_thread_Condition",
                                                mn,
                                                arg_count,
                                            )
                                        })
                                        .map(|(sig, mapping)| (sig.class, sig.method, mapping))
                                        .or_else(|| {
                                            self.get_stdlib_runtime_info(
                                                *field,
                                                object.ty,
                                                Some(arg_count),
                                                None,
                                            )
                                        })
                                } else if mn == "tryAcquire" {
                                    // Semaphore.tryAcquire() has overloads: 0 params vs 1 param (with timeout)
                                    let arg_count = args.len();
                                    debug!(
                                        "[tryAcquire lookup] method={}, arg_count={}",
                                        mn, arg_count
                                    );
                                    self.stdlib_mapping
                                        .find_by_name_and_params(
                                            "sys_thread_Semaphore",
                                            mn,
                                            arg_count,
                                        )
                                        .map(|(sig, mapping)| (sig.class, sig.method, mapping))
                                        .or_else(|| {
                                            self.get_stdlib_runtime_info(
                                                *field,
                                                object.ty,
                                                Some(arg_count),
                                                None,
                                            )
                                        })
                                } else {
                                    // Extract receiver class hint by finding which class owns this method symbol
                                    let receiver_hint: Option<String> =
                                        self.find_receiver_class_name(object);
                                    let hint_ref = receiver_hint.as_deref();
                                    self.get_stdlib_runtime_info(
                                        *field,
                                        object.ty,
                                        Some(args.len()),
                                        hint_ref,
                                    )
                                }
                            } else {
                                self.get_stdlib_runtime_info(
                                    *field,
                                    object.ty,
                                    Some(args.len()),
                                    None,
                                )
                            }
                        };

                        if let Some((class_name, method_name, runtime_call)) = stdlib_info {
                            let mut runtime_func_owned = runtime_call.runtime_name.to_string();
                            let is_mir_wrapper = runtime_call.is_mir_wrapper;
                            let raw_value_params = runtime_call.raw_value_params;
                            let extend_to_i64_params = runtime_call.extend_to_i64_params;
                            let returns_raw_value = runtime_call.returns_raw_value;
                            let has_return = runtime_call.has_return;
                            let explicit_return_type =
                                runtime_call.return_type.map(|rt| rt.to_ir_type());
                            let has_self_param = runtime_call.has_self_param;

                            // L3: size-correct Ptr<T> wrappers. The default
                            // Ptr_offset/deref/write are size-erased (treat T as
                            // i64=8B). When the receiver's pointee is a narrow value
                            // type, redirect to the sized variant registered in
                            // systems.rs. Unknown/generic/>=8-byte pointee keeps the
                            // default name -> byte-identical to pre-L3 codegen.
                            if matches!(
                                runtime_func_owned.as_str(),
                                "Ptr_offset" | "Ptr_deref" | "Ptr_write"
                            ) {
                                let pointee = {
                                    let type_table = self.type_table;
                                    type_table.get(object.ty).and_then(|ti| match &ti.kind {
                                        crate::tast::TypeKind::Class { type_args, .. }
                                        | crate::tast::TypeKind::GenericInstance {
                                            type_args,
                                            ..
                                        } => {
                                            if !type_args.is_empty() {
                                                Some(self.convert_type(type_args[0]))
                                            } else {
                                                None
                                            }
                                        }
                                        _ => None,
                                    })
                                };
                                if let Some(pointee) = pointee {
                                    let suffix = match &pointee {
                                        IrType::F32 => "_4f",
                                        IrType::I32 | IrType::U32 => "_4",
                                        IrType::U8 | IrType::I8 | IrType::Bool => "_1",
                                        _ => "",
                                    };
                                    if !suffix.is_empty() {
                                        runtime_func_owned.push_str(suffix);
                                    }
                                }
                            }
                            let runtime_func: &str = &runtime_func_owned;

                            // Try special runtime calls that need custom MIR lowering
                            // (e.g., Type.typeof needs to return boxed ValueType enum)
                            if let Some(special_result) = self.try_lower_special_runtime_call(
                                runtime_func,
                                args,
                                result_type.clone(),
                                expr.source_location,
                            ) {
                                return special_result;
                            }
                            // Method redirected via runtime mapping

                            // Reflect.compare: redirect to haxe_reflect_compare_typed with type tag
                            // Must be done before the generic arg boxing loop below.
                            if runtime_func == "haxe_reflect_compare" && args.len() >= 2 {
                                let type_info = self.infer_reflect_compare_type_info(args);
                                if let Some(info) = type_info {
                                    let mut typed_args = Vec::new();
                                    for arg in args.iter() {
                                        if let Some(reg) = self.lower_expression(arg) {
                                            let reg_ty = self
                                                .builder
                                                .get_register_type(reg)
                                                .unwrap_or(IrType::I64);
                                            let final_reg = if reg_ty != IrType::I64 {
                                                self.builder
                                                    .build_cast(reg, reg_ty, IrType::I64)
                                                    .unwrap_or(reg)
                                            } else {
                                                reg
                                            };
                                            typed_args.push(final_reg);
                                        }
                                    }
                                    let tag_reg = match info {
                                        Ok(tag_value) => {
                                            self.builder.build_const(IrValue::I32(tag_value))?
                                        }
                                        Err(type_param_name) => {
                                            let tag = self.builder.build_const(IrValue::I32(0))?;
                                            if let Some(func) = self.builder.current_function_mut()
                                            {
                                                func.type_param_tag_fixups
                                                    .push((tag, type_param_name));
                                            }
                                            tag
                                        }
                                    };
                                    typed_args.push(tag_reg);
                                    let extern_func_id = self.get_or_register_extern_function(
                                        "haxe_reflect_compare_typed",
                                        vec![IrType::I64, IrType::I64, IrType::I32],
                                        IrType::I64,
                                    );
                                    let call_result = self.builder.build_call_direct(
                                        extern_func_id,
                                        typed_args,
                                        IrType::I64,
                                    )?;
                                    if result_type == IrType::I32 {
                                        return self.builder.build_cast(
                                            call_result,
                                            IrType::I64,
                                            IrType::I32,
                                        );
                                    }
                                    return Some(call_result);
                                }
                            }

                            // Get expected parameter types from the extern function signature
                            // This is critical for generic classes like Deque<T> where the runtime
                            // expects boxed pointers but HIR types may be primitives
                            let (expected_param_types, actual_return_type) = self
                                .get_stdlib_mir_wrapper_signature(runtime_func)
                                .map(|(params, ret)| (params, ret))
                                .unwrap_or_else(|| {
                                    // Fallback: derive from arguments, using stdlib mapping hints
                                    let mut params = vec![IrType::Ptr(Box::new(IrType::U8))];
                                    for (i, arg) in args.iter().enumerate() {
                                        // Check raw_value_params bitmask: bit (i+1) means param i+1
                                        // (bit 0 is self, bit 1 is first user arg, etc.)
                                        let param_bit = 1u32 << (i + 1);
                                        if raw_value_params & param_bit != 0 {
                                            params.push(IrType::U64);
                                        } else if extend_to_i64_params & param_bit != 0 {
                                            params.push(IrType::I64);
                                        } else {
                                            params.push(self.convert_type(arg.ty));
                                        }
                                    }
                                    // Use explicit return type from types: descriptor when available,
                                    // otherwise fall back to legacy inference
                                    let ret_type = if let Some(ref rt) = explicit_return_type {
                                        rt.clone()
                                    } else if returns_raw_value {
                                        IrType::U64
                                    } else if has_return {
                                        result_type.clone()
                                    } else {
                                        IrType::Void
                                    };
                                    (params, ret_type)
                                });
                            debug!(
                                "[Extern method redirect] expected params: {:?}, return type: {:?}",
                                expected_param_types, actual_return_type
                            );

                            // For static stdlib methods (e.g., StringTools.startsWith via `using`),
                            // the object is a class reference, NOT an instance receiver.
                            // The args already include the real receiver (first arg from `using` desugaring).
                            // Don't prepend the class reference as 'this'.
                            let is_static_stdlib = !has_self_param;

                            let mut arg_regs = if is_static_stdlib {
                                Vec::new()
                            } else {
                                // Lower the object (this will be the first parameter)
                                let obj_reg = self.lower_expression(object)?;
                                vec![obj_reg] // 'this' as first arg
                            };
                            for (i, arg) in args.iter().enumerate() {
                                let arg_reg = self.lower_expression(arg)?;
                                let actual_ty = self.convert_type(arg.ty);

                                // Get expected type for this argument
                                // For instance methods, offset by 1 for 'this'
                                let param_idx = if is_static_stdlib { i } else { i + 1 };
                                let expected_ty = expected_param_types
                                    .get(param_idx)
                                    .cloned()
                                    .unwrap_or_else(|| actual_ty.clone());

                                // Auto-box if needed (e.g., Int -> Ptr(U8) for Deque<Int>.add())
                                let final_reg = self.maybe_box_for_extern_call(
                                    arg_reg,
                                    &actual_ty,
                                    &expected_ty,
                                )?;
                                arg_regs.push(final_reg);
                            }

                            // Inject hidden enum type_id arg for runtime enum helpers
                            // (enumEq, enumConstructor, enumParameters, getEnum)
                            self.inject_hidden_enum_type_id_arg(runtime_func, args, &mut arg_regs);

                            // Use the expected parameter types for the extern function registration
                            // This ensures the signature matches what the runtime expects
                            let param_types = if expected_param_types.len() == arg_regs.len() {
                                expected_param_types.clone()
                            } else {
                                // Fallback if lengths don't match
                                let mut params = if is_static_stdlib {
                                    Vec::new()
                                } else {
                                    vec![IrType::Ptr(Box::new(IrType::U8))]
                                };
                                for arg in args {
                                    params.push(self.convert_type(arg.ty));
                                }
                                params
                            };

                            // Register and call the function (MIR wrapper or extern)
                            let call_result = if is_mir_wrapper {
                                let mir_func_id = self.register_stdlib_mir_forward_ref(
                                    runtime_func,
                                    param_types,
                                    actual_return_type.clone(),
                                );
                                self.builder.build_call_direct(
                                    mir_func_id,
                                    arg_regs,
                                    actual_return_type.clone(),
                                )?
                            } else {
                                let extern_func_id = self.get_or_register_extern_function(
                                    runtime_func,
                                    param_types,
                                    actual_return_type.clone(),
                                );
                                self.builder.build_call_direct(
                                    extern_func_id,
                                    arg_regs,
                                    actual_return_type.clone(),
                                )?
                            };

                            // Auto-unbox if runtime returns Ptr(U8) but HIR expects primitive
                            // (e.g., Deque<Int>.pop() returns boxed int that needs unboxing)
                            // For generic classes like Channel<Int>, resolve T from receiver type args
                            let resolved_expected = {
                                let needs_resolve = result_type == IrType::Any
                                    || matches!(&result_type, IrType::Ptr(inner) if matches!(inner.as_ref(), IrType::Void))
                                    || result_type == IrType::I64;
                                if needs_resolve {
                                    let type_table = self.type_table;
                                    type_table
                                        .get(object.ty)
                                        .and_then(|ti| match &ti.kind {
                                            crate::tast::TypeKind::Class { type_args, .. }
                                            | crate::tast::TypeKind::GenericInstance {
                                                type_args,
                                                ..
                                            } => {
                                                if !type_args.is_empty() {
                                                    Some(self.convert_type(type_args[0]))
                                                } else {
                                                    None
                                                }
                                            }
                                            _ => None,
                                        })
                                        .unwrap_or_else(|| result_type.clone())
                                } else {
                                    result_type.clone()
                                }
                            };
                            // Store class hint on result register for subsequent method dispatch.
                            // E.g., Array.iterator() returns ArrayIterator — tag the result so
                            // it.hasNext()/it.next() can find the correct stdlib mapping.
                            //
                            // MIR wrappers return values in their declared type directly —
                            // they don't return boxed DynamicValue* pointers. Skip unboxing
                            // for MIR wrapper calls to avoid spurious dereferences
                            // (e.g., Host.localhost() returns a raw string pointer, not a boxed value).
                            let final_result = if is_mir_wrapper {
                                // MIR wrappers return values in their declared type directly.
                                // array_pop returns raw I64 — cast to Ptr(Void) for class types.
                                let expects_class_ptr = matches!(&resolved_expected, IrType::Ptr(inner) if matches!(inner.as_ref(), IrType::Void));
                                if actual_return_type == IrType::I64 && expects_class_ptr {
                                    // Raw I64 → Ptr(Void) cast
                                    self.builder.build_cast(
                                        call_result,
                                        IrType::I64,
                                        IrType::Ptr(Box::new(IrType::Void)),
                                    )
                                } else if actual_return_type == IrType::I64
                                    && result_type == IrType::I32
                                {
                                    // Truncate I64 → I32 for Int-returning methods
                                    // (stdlib returns usize/i64 but Haxe Int is i32)
                                    self.builder
                                        .build_cast(call_result, IrType::I64, IrType::I32)
                                } else {
                                    Some(call_result)
                                }
                            } else {
                                self.maybe_unbox_for_extern_return(
                                    call_result,
                                    &actual_return_type,
                                    &resolved_expected,
                                )
                            };
                            if let Some(result_reg) = final_result {
                                let return_class =
                                    Self::get_return_class_hint(class_name, method_name);
                                self.register_class_hints
                                    .insert(result_reg, return_class.to_string());
                            }
                            return final_result;
                        }

                        // FALLBACK: For extern classes not in type_table (like rayzor.Bytes),
                        // try to extract class name from the MIR function's qualified_name
                        debug!(
                            "[FALLBACK check] func_id={:?}, in module={}",
                            func_id,
                            self.builder.module.functions.contains_key(&func_id)
                        );
                        if let Some(func) = self.builder.module.functions.get(&func_id) {
                            debug!(
                                "[FALLBACK] MIR function '{}' has qualified_name: {:?}",
                                func.name, func.qualified_name
                            );
                            if let Some(ref qn) = func.qualified_name {
                                // Pattern: "rayzor.Bytes.set" -> class="rayzor_Bytes", method="set"
                                let parts: Vec<&str> = qn.split('.').collect();
                                if parts.len() >= 2 {
                                    // Get method name (last part) and class name (all but last, joined with underscore)
                                    let mir_method_name = *parts.last().unwrap();
                                    let class_parts = &parts[..parts.len() - 1];
                                    let qualified_class = class_parts.join("_");

                                    // Try to find in stdlib mapping
                                    if let Some((_sig, mapping)) = self
                                        .stdlib_mapping
                                        .find_by_name(&qualified_class, mir_method_name)
                                    {
                                        let runtime_func = mapping.runtime_name;
                                        debug!(
                                            "[Extern method redirect via qualified_name] {}.{} -> {}",
                                            qualified_class, mir_method_name, runtime_func
                                        );

                                        // Get expected parameter types from the extern function signature
                                        let (expected_param_types, actual_return_type) = self
                                            .get_stdlib_mir_wrapper_signature(runtime_func)
                                            .map(|(params, ret)| (params, ret))
                                            .unwrap_or_else(|| {
                                                let mut params =
                                                    vec![IrType::Ptr(Box::new(IrType::U8))];
                                                for arg in args {
                                                    params.push(self.convert_type(arg.ty));
                                                }
                                                // Use explicit return type from types: descriptor when available
                                                let ret_type =
                                                    if let Some(ref rt) = mapping.return_type {
                                                        rt.to_ir_type()
                                                    } else if mapping.has_return {
                                                        result_type.clone()
                                                    } else {
                                                        IrType::Void
                                                    };
                                                (params, ret_type)
                                            });

                                        // Lower the object (this will be the first parameter)
                                        let obj_reg = self.lower_expression(object)?;

                                        // Lower the arguments and auto-box if needed
                                        let mut arg_regs = vec![obj_reg];
                                        for (i, arg) in args.iter().enumerate() {
                                            let arg_reg = self.lower_expression(arg)?;
                                            let actual_ty = self.convert_type(arg.ty);
                                            let expected_ty = expected_param_types
                                                .get(i + 1)
                                                .cloned()
                                                .unwrap_or_else(|| actual_ty.clone());
                                            let final_reg = self.maybe_box_for_extern_call(
                                                arg_reg,
                                                &actual_ty,
                                                &expected_ty,
                                            )?;
                                            arg_regs.push(final_reg);
                                        }

                                        // Use expected parameter types for registration
                                        let param_types =
                                            if expected_param_types.len() == arg_regs.len() {
                                                expected_param_types.clone()
                                            } else {
                                                let mut params =
                                                    vec![IrType::Ptr(Box::new(IrType::U8))];
                                                for arg in args {
                                                    params.push(self.convert_type(arg.ty));
                                                }
                                                params
                                            };

                                        // Register the extern function
                                        let extern_func_id = self.get_or_register_extern_function(
                                            runtime_func,
                                            param_types,
                                            actual_return_type.clone(),
                                        );

                                        // Call the extern function
                                        let call_result = self.builder.build_call_direct(
                                            extern_func_id,
                                            arg_regs,
                                            actual_return_type.clone(),
                                        )?;

                                        // Auto-unbox if runtime returns Ptr(U8) but HIR expects primitive
                                        return self.maybe_unbox_for_extern_return(
                                            call_result,
                                            &actual_return_type,
                                            &result_type,
                                        );
                                    }
                                }
                            }
                        }

                        // Check for virtual dispatch: if the method is in a class hierarchy
                        // with overrides, dispatch through the vtable instead of calling directly.
                        // Skip vtable for super.method() — must call parent directly.
                        let object_is_super = matches!(object.kind, HirExprKind::Super);
                        let vtable_lookup = if object_is_super {
                            None
                        } else {
                            self.virtual_dispatch_info.get(field).copied().or_else(|| {
                                let method_name =
                                    self.symbol_table.get_symbol(*field).map(|s| s.name);
                                if let Some(method_name) = method_name {
                                    let receiver_class_sym = {
                                        let type_table = self.type_table;
                                        type_table.get(object.ty).and_then(|t| match &t.kind {
                                            TypeKind::Class { symbol_id, .. } => Some(*symbol_id),
                                            _ => None,
                                        })
                                    };
                                    if let Some(class_sym) = receiver_class_sym {
                                        if let Some(&method_sym) =
                                            self.class_method_by_name.get(&(class_sym, method_name))
                                        {
                                            return self
                                                .virtual_dispatch_info
                                                .get(&method_sym)
                                                .copied();
                                        }
                                        let mut current = class_sym;
                                        while let Some(&parent) =
                                            self.class_parent_map.get(&current)
                                        {
                                            if let Some(&method_sym) = self
                                                .class_method_by_name
                                                .get(&(parent, method_name))
                                            {
                                                return self
                                                    .virtual_dispatch_info
                                                    .get(&method_sym)
                                                    .copied();
                                            }
                                            current = parent;
                                        }
                                    }
                                }
                                None
                            })
                        };
                        if let Some((slot_index, _)) = vtable_lookup {
                            let obj_reg = self.lower_expression(object)?;

                            // If Dynamic-typed, unbox to get raw object pointer
                            let obj_reg = {
                                let is_dynamic = {
                                    let type_table = self.type_table;
                                    type_table
                                        .get(object.ty)
                                        .map(|t| matches!(t.kind, TypeKind::Dynamic))
                                        .unwrap_or(false)
                                };
                                if is_dynamic {
                                    let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                                    let unbox_func_id = self.get_or_register_extern_function(
                                        "haxe_unbox_reference_ptr",
                                        vec![ptr_u8.clone()],
                                        ptr_u8.clone(),
                                    );
                                    self.builder.build_call_direct(
                                        unbox_func_id,
                                        vec![obj_reg],
                                        ptr_u8,
                                    )?
                                } else {
                                    obj_reg
                                }
                            };

                            // Lower arguments
                            let mut call_args = vec![obj_reg];
                            for arg in args.iter() {
                                if let Some(reg) = self.lower_expression(arg) {
                                    call_args.push(reg);
                                }
                            }

                            // haxe_vtable_lookup(obj, slot) -> closure_ptr (i64)
                            let lookup_fn = self.get_or_register_extern_function(
                                "haxe_vtable_lookup",
                                vec![IrType::Ptr(Box::new(IrType::U8)), IrType::I32],
                                IrType::I64,
                            );
                            let slot_reg =
                                self.builder.build_const(IrValue::I32(slot_index as i32))?;
                            let closure_ptr = self.builder.build_call_direct(
                                lookup_fn,
                                vec![obj_reg, slot_reg],
                                IrType::I64,
                            )?;

                            // Build the function signature for the indirect call
                            let mut param_types = vec![IrType::Ptr(Box::new(IrType::Void))]; // self
                            for arg in args {
                                param_types.push(self.convert_type(arg.ty));
                            }
                            let return_type = Box::new(self.convert_type(expr.ty));
                            let func_signature = IrType::Function {
                                params: param_types,
                                return_type,
                                varargs: false,
                            };

                            return self.builder.build_call_indirect(
                                closure_ptr,
                                call_args,
                                func_signature,
                            );
                        }

                        // super.method() — resolve to parent class method directly
                        if object_is_super {
                            if let Some(method_name_i) = method_name_interned {
                                // Find current class → parent via class_parent_map
                                let current_class = self.builder.current_function().and_then(|f| {
                                    self.class_method_by_name
                                        .iter()
                                        .find(|(_, &method_sym)| {
                                            self.function_map.get(&method_sym) == Some(&f.id)
                                        })
                                        .map(|((class_sym, _), _)| *class_sym)
                                });
                                let parent_class = current_class
                                    .and_then(|cls| self.class_parent_map.get(&cls).copied());
                                let parent_method_func_id = parent_class
                                    .and_then(|pc| {
                                        self.class_method_by_name
                                            .get(&(pc, method_name_i))
                                            .and_then(|&sym| {
                                                self.resolve_function_id_with_qualified_fallback(
                                                    sym,
                                                )
                                            })
                                    })
                                    .or_else(|| {
                                        self.resolve_function_id_with_qualified_fallback(*field)
                                    });
                                if let Some(func_id) = parent_method_func_id {
                                    let obj_reg = self.lower_expression(object)?;
                                    let mut call_args = vec![obj_reg];
                                    for arg in args {
                                        if let Some(reg) = self.lower_expression(arg) {
                                            call_args.push(reg);
                                        }
                                    }
                                    let ret = self.convert_type(expr.ty);
                                    return self.builder.build_call_direct(func_id, call_args, ret);
                                }
                            }
                        }

                        // Regular method call (not extern or no runtime mapping)
                        // Detect static calls: if the object is a class/abstract symbol
                        // reference (not an instance), this is a static method call and
                        // the object should NOT be passed as 'this'.
                        let is_static_class_call = if let HirExprKind::Variable {
                            symbol: obj_sym,
                            ..
                        } = &object.kind
                        {
                            let kind = self.symbol_table.get_symbol(*obj_sym).map(|s| s.kind);
                            kind.map(|k| {
                                matches!(
                                    k,
                                    crate::tast::symbols::SymbolKind::Class
                                        | crate::tast::symbols::SymbolKind::Abstract
                                        | crate::tast::symbols::SymbolKind::TypeAlias
                                )
                            })
                            .unwrap_or(false)
                        } else {
                            false
                        };

                        // @:derive(Default) synthetic static createDefault() — zero-initialized instance
                        if is_static_class_call
                            && method_name == Some("createDefault")
                            && args.is_empty()
                        {
                            if let HirExprKind::Variable {
                                symbol: obj_sym, ..
                            } = &object.kind
                            {
                                // For static calls, obj_sym is the class symbol itself.
                                // Also try resolving through type_id → TypeKind::Class → symbol_id.
                                let class_sym = self
                                    .symbol_table
                                    .get_symbol(*obj_sym)
                                    .and_then(|s| {
                                        let type_table = self.type_table;
                                        type_table.get(s.type_id).and_then(|t| {
                                            if let TypeKind::Class { symbol_id, .. } = &t.kind {
                                                Some(*symbol_id)
                                            } else {
                                                None
                                            }
                                        })
                                    })
                                    .or(Some(*obj_sym))
                                    .filter(|sym| self.derive_default_classes.contains(sym));
                                if let Some(sym) = class_sym {
                                    return self.lower_derived_default(sym);
                                }
                            }
                        }

                        if is_static_class_call {
                            // Check if this static call has a stdlib runtime mapping.
                            // Static method calls (e.g., Reflect.hasField, Type.typeof, Std.string)
                            // arrive here as Field { object: ClassVar, field: method } and the
                            // stdlib_info lookup above may have failed because the class variable's
                            // TypeId isn't in type_table. Route through the extern function.
                            let static_class_name = self.find_receiver_class_name(object);
                            let static_method_name = self
                                .symbol_table
                                .get_symbol(*field)
                                .and_then(|s| self.string_interner.get(s.name))
                                .map(|s| s.to_string());

                            // Also try to get class name from the object symbol directly.
                            // Prefer native_name or qualified_name (fully qualified) over
                            // bare name so that extern classes like sys.net.Host resolve to
                            // "sys_net_Host" instead of just "Host".
                            let static_class_name = static_class_name.or_else(|| {
                                if let HirExprKind::Variable {
                                    symbol: obj_sym, ..
                                } = &object.kind
                                {
                                    let sym = self.symbol_table.get_symbol(*obj_sym)?;
                                    // 1) native_name  (e.g. "sys::net::Host" → "sys_net_Host")
                                    if let Some(native) = sym.native_name {
                                        if let Some(ns) = self.string_interner.get(native) {
                                            // "rayzor::runtime::CC" → "rayzor_runtime_CC"
                                            return Some(
                                                ns.split("::").collect::<Vec<_>>().join("_"),
                                            );
                                        }
                                    }
                                    // 2) qualified_name  (e.g. "sys.net.Host" → "sys_net_Host")
                                    if let Some(qn) = sym.qualified_name {
                                        if let Some(qs) = self.string_interner.get(qn) {
                                            return Some(qs.replace('.', "_"));
                                        }
                                    }
                                    // 3) bare name fallback
                                    self.string_interner.get(sym.name).map(|s| s.to_string())
                                } else {
                                    None
                                }
                            });

                            if let (Some(ref cls), Some(ref mn)) =
                                (&static_class_name, &static_method_name)
                            {
                                let static_stdlib_info = self
                                    .stdlib_mapping
                                    .find_by_name_and_params(cls, mn, args.len())
                                    .or_else(|| self.stdlib_mapping.find_by_name(cls, mn))
                                    .map(|(sig, mapping)| (sig.class, sig.method, mapping));

                                if let Some((sc_class_name, sc_method_name, runtime_call)) =
                                    static_stdlib_info
                                {
                                    let runtime_func = runtime_call.runtime_name;
                                    let is_mir_wrapper = runtime_call.is_mir_wrapper;
                                    let returns_raw_value = runtime_call.returns_raw_value;
                                    let has_return = runtime_call.has_return;
                                    let raw_value_params = runtime_call.raw_value_params;
                                    let extend_to_i64_params = runtime_call.extend_to_i64_params;
                                    let explicit_return_type =
                                        runtime_call.return_type.map(|rt| rt.to_ir_type());

                                    // First, try special runtime calls that need custom MIR lowering
                                    // (e.g., Reflect.callMethod, Reflect.makeVarArgs, Type.typeof)
                                    if let Some(special_result) = self
                                        .try_lower_special_runtime_call(
                                            runtime_func,
                                            args,
                                            result_type.clone(),
                                            expr.source_location,
                                        )
                                    {
                                        return special_result;
                                    }

                                    // Get expected parameter types from the mapping
                                    let (expected_param_types, actual_return_type) = self
                                        .get_stdlib_mir_wrapper_signature(runtime_func)
                                        .unwrap_or_else(|| {
                                            let mut params = Vec::new();
                                            for (i, arg) in args.iter().enumerate() {
                                                let param_bit = 1u32 << i;
                                                if raw_value_params & param_bit != 0 {
                                                    params.push(IrType::U64);
                                                } else if extend_to_i64_params & param_bit != 0 {
                                                    params.push(IrType::I64);
                                                } else {
                                                    params.push(self.convert_type(arg.ty));
                                                }
                                            }
                                            let ret_type =
                                                if let Some(ref ert) = explicit_return_type {
                                                    ert.clone()
                                                } else if returns_raw_value {
                                                    IrType::U64
                                                } else if has_return {
                                                    result_type.clone()
                                                } else {
                                                    IrType::Void
                                                };
                                            (params, ret_type)
                                        });

                                    // Lower arguments (no 'this' for static methods)
                                    let mut arg_regs = Vec::new();
                                    for (i, arg) in args.iter().enumerate() {
                                        let arg_reg = self.lower_expression(arg)?;
                                        let actual_ty = self.convert_type(arg.ty);
                                        let expected_ty = expected_param_types
                                            .get(i)
                                            .cloned()
                                            .unwrap_or_else(|| actual_ty.clone());
                                        let final_reg = self.maybe_box_for_extern_call(
                                            arg_reg,
                                            &actual_ty,
                                            &expected_ty,
                                        )?;
                                        arg_regs.push(final_reg);
                                    }

                                    if is_mir_wrapper {
                                        let param_types: Vec<_> = arg_regs
                                            .iter()
                                            .map(|r| {
                                                self.builder
                                                    .get_register_type(*r)
                                                    .unwrap_or(IrType::I64)
                                            })
                                            .collect();
                                        let mir_func_id = self.register_stdlib_mir_forward_ref(
                                            runtime_func,
                                            param_types,
                                            actual_return_type.clone(),
                                        );
                                        return self.builder.build_call_direct(
                                            mir_func_id,
                                            arg_regs,
                                            actual_return_type,
                                        );
                                    }

                                    // Inject hidden enum type_id arg for runtime enum helpers
                                    // (enumEq, enumConstructor, enumParameters, getEnum)
                                    self.inject_hidden_enum_type_id_arg(
                                        runtime_func,
                                        args,
                                        &mut arg_regs,
                                    );

                                    let param_types: Vec<_> = arg_regs
                                        .iter()
                                        .map(|r| {
                                            self.builder
                                                .get_register_type(*r)
                                                .unwrap_or(IrType::I64)
                                        })
                                        .collect();
                                    let extern_func_id = self.get_or_register_extern_function(
                                        runtime_func,
                                        param_types,
                                        actual_return_type.clone(),
                                    );

                                    let call_result = self.builder.build_call_direct(
                                        extern_func_id,
                                        arg_regs,
                                        actual_return_type.clone(),
                                    );

                                    // Tag a static factory result with its class so a
                                    // cross-module `var b = Bytes.ofString(s)` keeps a
                                    // class handle when the local's own type stays
                                    // unresolved. Applies when the factory returns the
                                    // same reference class (a `PtrVoid`/`Ptr` handle).
                                    if let Some(reg) = call_result {
                                        if matches!(
                                            self.builder.get_register_type(reg),
                                            Some(IrType::Ptr(_))
                                        ) {
                                            if let Some(hint) = self.static_factory_return_class(
                                                sc_class_name,
                                                sc_method_name,
                                            ) {
                                                self.register_class_hints.insert(reg, hint);
                                            }
                                        }
                                    }

                                    // Handle returns_raw_value: cast raw U64 to appropriate type
                                    if returns_raw_value {
                                        if let Some(raw_reg) = call_result {
                                            return match &result_type {
                                                IrType::I32 => self.builder.build_cast(
                                                    raw_reg,
                                                    IrType::U64,
                                                    IrType::I32,
                                                ),
                                                IrType::Bool => self.builder.build_cast(
                                                    raw_reg,
                                                    IrType::U64,
                                                    IrType::Bool,
                                                ),
                                                // F64/F32: bitcast raw u64 bits back to float.
                                                // Map<K,Float>.get stores f64 bits as u64; this
                                                // reverses the set-side bitcast so reads return
                                                // the original f64 value (was returning u64 bits
                                                // mis-interpreted as a giant int).
                                                IrType::F64 => {
                                                    self.builder.build_bitcast(raw_reg, IrType::F64)
                                                }
                                                IrType::F32 => {
                                                    let f64v = self
                                                        .builder
                                                        .build_bitcast(raw_reg, IrType::F64)?;
                                                    self.builder.build_cast(
                                                        f64v,
                                                        IrType::F64,
                                                        IrType::F32,
                                                    )
                                                }
                                                _ => Some(raw_reg),
                                            };
                                        }
                                    }

                                    return call_result;
                                }
                            }

                            // Static call: do NOT include the class reference as 'this'
                            let callee_is_user_defined = self
                                .builder
                                .module
                                .functions
                                .get(&func_id)
                                .map(|f| f.kind == crate::ir::functions::FunctionKind::UserDefined)
                                .unwrap_or(false);

                            let mut arg_regs = Vec::new();
                            for (param_idx, arg) in args.iter().enumerate() {
                                if let Some(reg) = self.lower_expression(arg) {
                                    // Materialize anon-backed variables at call boundary
                                    let reg = self.maybe_materialize_for_call(
                                        arg,
                                        reg,
                                        Some(func_id),
                                        param_idx,
                                    );
                                    // @:derive(Copy): copy variable args at call boundary
                                    let reg = if let HirExprKind::Variable { .. } = &arg.kind {
                                        if let Some(class_sym) = self.get_copy_class_symbol(arg.ty)
                                        {
                                            self.emit_shallow_copy(reg, class_sym).unwrap_or(reg)
                                        } else {
                                            reg
                                        }
                                    } else {
                                        reg
                                    };
                                    if callee_is_user_defined {
                                        let is_heap_intermediate = matches!(
                                            &arg.kind,
                                            HirExprKind::New { .. } | HirExprKind::Call { .. }
                                        ) && self
                                            .get_drop_behavior(arg.ty)
                                            == DropBehavior::AutoDrop
                                            && !self.interface_wrapped_args.contains(&reg);
                                        if is_heap_intermediate {
                                            self.temp_heap_values.push(reg);
                                        }
                                    }
                                    arg_regs.push(reg);
                                }
                            }

                            self.coerce_args_for_cross_module_call(func_id, &mut arg_regs, false);
                            self.fill_default_args(func_id, &mut arg_regs, false);

                            let actual_return_type =
                                if let Some(func) = self.builder.module.functions.get(&func_id) {
                                    func.signature.return_type.clone()
                                } else {
                                    result_type.clone()
                                };

                            let result = self.builder.build_call_direct(
                                func_id,
                                arg_regs,
                                actual_return_type,
                            );
                            // Set class hint on result for cross-module method dispatch
                            if let Some(reg) = result {
                                self.set_class_hint_for_return(reg, expr.ty);
                            }
                            return result;
                        }

                        // Lower the object (this will be the first parameter)
                        let obj_reg = self.lower_expression(object)?;

                        // If the object is Dynamic-typed, unbox it to get the raw object pointer.
                        // Dynamic variables store a boxed DynamicValue*, but the method expects
                        // a raw class pointer as 'this'.
                        let obj_reg = {
                            let is_dynamic = {
                                let type_table = self.type_table;
                                type_table
                                    .get(object.ty)
                                    .map(|t| matches!(t.kind, TypeKind::Dynamic))
                                    .unwrap_or(false)
                            };
                            if is_dynamic {
                                let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                                let unbox_func_id = self.get_or_register_extern_function(
                                    "haxe_unbox_reference_ptr",
                                    vec![ptr_u8.clone()],
                                    ptr_u8.clone(),
                                );
                                self.builder.build_call_direct(
                                    unbox_func_id,
                                    vec![obj_reg],
                                    ptr_u8,
                                )?
                            } else {
                                obj_reg
                            }
                        };

                        // Track NEW expressions as temps (not Call results — they may be references)
                        let is_new_temp = matches!(&object.kind, HirExprKind::New { .. })
                            && self.get_drop_behavior(object.ty) == DropBehavior::AutoDrop;
                        if is_new_temp {
                            self.temp_heap_values.push(obj_reg);
                        }

                        // Lower the arguments — track heap intermediates only for user-defined callees
                        let callee_is_user_defined = self
                            .builder
                            .module
                            .functions
                            .get(&func_id)
                            .map(|f| f.kind == crate::ir::functions::FunctionKind::UserDefined)
                            .unwrap_or(false);

                        let mut method_arg_regs = vec![obj_reg]; // 'this' as first arg
                        for arg in args.iter() {
                            if let Some(reg) = self.lower_expression(arg) {
                                if callee_is_user_defined {
                                    let is_heap_intermediate = matches!(
                                        &arg.kind,
                                        HirExprKind::New { .. } | HirExprKind::Call { .. }
                                    ) && self.get_drop_behavior(arg.ty)
                                        == DropBehavior::AutoDrop;
                                    if is_heap_intermediate {
                                        self.temp_heap_values.push(reg);
                                    }
                                }
                                method_arg_regs.push(reg);
                            }
                        }
                        let arg_regs = method_arg_regs;

                        // IMPORTANT: Use the function's actual return type, not expr.ty
                        // expr.ty can be incorrect (e.g., unresolved TypeParameter or wrong type)
                        let actual_return_type = if let Some(func) =
                            self.builder.module.functions.get(&func_id)
                        {
                            debug!(
                                "[Field method] Using actual return type {:?} for function {:?}",
                                func.signature.return_type, func.name
                            );
                            func.signature.return_type.clone()
                        } else {
                            debug!(
                                "[Field method] Function not found in module, using expr return type {:?}",
                                result_type
                            );
                            result_type.clone()
                        };

                        // debug!("Calling method with {} args (including this)", arg_regs.len());
                        let call_result = self.builder.build_call_direct(
                            func_id,
                            arg_regs,
                            actual_return_type.clone(),
                        )?;

                        // Auto-unbox for generic stdlib methods (e.g., Channel<Int>.tryReceive())
                        // When the function returns Ptr(U8) but the caller expects a resolved generic type,
                        // we need to unbox. Resolve T from the receiver object's type arguments.
                        let actual_is_ptr = matches!(&actual_return_type, IrType::Ptr(inner) if matches!(inner.as_ref(), IrType::U8 | IrType::Void));
                        if actual_is_ptr && actual_return_type != IrType::Void {
                            let resolved_type = {
                                let type_table = self.type_table;
                                type_table.get(object.ty).and_then(|ti| {
                                    match &ti.kind {
                                        crate::tast::TypeKind::Class { type_args, .. }
                                        | crate::tast::TypeKind::GenericInstance {
                                            type_args,
                                            ..
                                        } => {
                                            if !type_args.is_empty() {
                                                let t = self.convert_type(type_args[0]);
                                                // Only unbox if resolved to a concrete primitive
                                                if matches!(
                                                    t,
                                                    IrType::I32
                                                        | IrType::I64
                                                        | IrType::F32
                                                        | IrType::F64
                                                        | IrType::Bool
                                                ) {
                                                    Some(t)
                                                } else {
                                                    None
                                                }
                                            } else {
                                                None
                                            }
                                        }
                                        _ => None,
                                    }
                                })
                            };
                            if let Some(ref resolved) = resolved_type {
                                return self.maybe_unbox_for_extern_return(
                                    call_result,
                                    &actual_return_type,
                                    resolved,
                                );
                            }
                        }
                        return Some(call_result);
                    } else {
                        // Method not found by direct symbol lookup.
                        // First: try rpkg/plugin extern dispatch via direct mapping lookup.
                        // This bypasses get_stdlib_runtime_info's guard which rejects rpkg classes.
                        {
                            let method_name_str = self
                                .symbol_table
                                .get_symbol(*field)
                                .and_then(|s| self.string_interner.get(s.name))
                                .map(|s| s.to_string());
                            let receiver_class = self.find_receiver_class_name(object);

                            if let (Some(ref cls), Some(ref mn)) =
                                (&receiver_class, &method_name_str)
                            {
                                let plugin_match = self
                                    .stdlib_mapping
                                    .find_by_name_and_params(cls, mn, args.len())
                                    .or_else(|| self.stdlib_mapping.find_by_name(cls, mn));

                                if let Some((sig, runtime_call)) = plugin_match {
                                    let runtime_func = runtime_call.runtime_name;
                                    let is_mir_wrapper = runtime_call.is_mir_wrapper;
                                    let explicit_return_type =
                                        runtime_call.return_type.map(|rt| rt.to_ir_type());
                                    let is_static_call = sig.is_static;

                                    // A plugin mapping carries the exact ABI signature it
                                    // declared (param_types includes self for instance
                                    // methods). It is authoritative — and free of the TAST
                                    // param-type drift that otherwise mis-boxes scalar args
                                    // on an imported extern class. Prefer it over the
                                    // name-keyed wrapper registry (which has no entry for a
                                    // pure plugin symbol and so falls back to Ptr).
                                    let (expected_param_types, actual_return_type) =
                                        if let Some(descs) = runtime_call.param_types {
                                            let params: Vec<IrType> =
                                                descs.iter().map(|d| d.to_ir_type()).collect();
                                            let ret = runtime_call
                                                .return_type
                                                .map(|d| d.to_ir_type())
                                                .or_else(|| explicit_return_type.clone())
                                                .unwrap_or_else(|| self.convert_type(expr.ty));
                                            (params, ret)
                                        } else {
                                            self.get_stdlib_mir_wrapper_signature(runtime_func)
                                                .unwrap_or_else(|| {
                                                    let mut params = if is_static_call {
                                                        Vec::new()
                                                    } else {
                                                        vec![IrType::Ptr(Box::new(IrType::U8))]
                                                    };
                                                    for arg in args {
                                                        params.push(self.convert_type(arg.ty));
                                                    }
                                                    let ret = explicit_return_type
                                                        .clone()
                                                        .unwrap_or_else(|| {
                                                            self.convert_type(expr.ty)
                                                        });
                                                    (params, ret)
                                                })
                                        };

                                    let mut arg_regs = if is_static_call {
                                        Vec::new()
                                    } else {
                                        let obj_reg = self.lower_expression(object)?;
                                        vec![obj_reg]
                                    };
                                    // Coerce each user arg to the wrapper's DECLARED param type.
                                    // Critical for SIMD4f static methods (splat/make/load): they
                                    // declare F32 lane params, but Haxe `Float` args are F64 —
                                    // without the demote the raw f64 bit-pattern lands in the f32
                                    // lanes as garbage (splat(4.0) -> nonsense). param_offset skips
                                    // the leading self slot on instance wrappers. Previously args
                                    // were pushed raw, so any wrapper whose signature param type
                                    // differed from the arg type (F32 vs F64 here) silently
                                    // mismatched at the call boundary.
                                    let param_offset = if is_static_call { 0 } else { 1 };
                                    for (i, arg) in args.iter().enumerate() {
                                        if let Some(reg) = self.lower_expression(arg) {
                                            let actual_ty = self.convert_type(arg.ty);
                                            let expected_ty = expected_param_types
                                                .get(i + param_offset)
                                                .cloned()
                                                .unwrap_or_else(|| actual_ty.clone());
                                            let final_reg = self.maybe_box_for_extern_call(
                                                reg,
                                                &actual_ty,
                                                &expected_ty,
                                            )?;
                                            arg_regs.push(final_reg);
                                        }
                                    }

                                    let call_result = if is_mir_wrapper {
                                        let fid = self.register_stdlib_mir_forward_ref(
                                            runtime_func,
                                            expected_param_types,
                                            actual_return_type.clone(),
                                        );
                                        self.builder.build_call_direct(
                                            fid,
                                            arg_regs,
                                            actual_return_type.clone(),
                                        )?
                                    } else {
                                        let fid = self.get_or_register_extern_function(
                                            runtime_func,
                                            expected_param_types,
                                            actual_return_type.clone(),
                                        );
                                        self.builder.build_call_direct(
                                            fid,
                                            arg_regs,
                                            actual_return_type.clone(),
                                        )?
                                    };
                                    // Reconcile the descriptor's NATIVE return type with the
                                    // Haxe-declared type at this callsite.
                                    let declared_ir = self.convert_type(expr.ty);
                                    return Some(self.reconcile_extern_return(
                                        call_result,
                                        &actual_return_type,
                                        &declared_ir,
                                    ));
                                }
                            }
                        }

                        // Fallback: Dynamic method call or stdlib method
                        let object_type = object.ty;

                        // Check if the object is a stdlib class (including extern abstracts like Ptr, Ref, Box, Usize)
                        // These should resolve via stdlib_mapping without any Dynamic unboxing
                        debug!(
                            "[FIELDACCESS] Entering stdlib class check for object_type={:?}",
                            object_type
                        );
                        {
                            let type_table = self.type_table;
                            let class_symbol_id =
                                if let Some(type_info) = type_table.get(object_type) {
                                    match &type_info.kind {
                                        TypeKind::Class { symbol_id, .. } => Some(*symbol_id),
                                        TypeKind::Abstract { symbol_id, .. } => Some(*symbol_id),
                                        TypeKind::GenericInstance { base_type, .. } => {
                                            // For GenericInstance like ObjectMap<Point, Int>,
                                            // resolve the base class/abstract symbol
                                            if let Some(base_info) = type_table.get(*base_type) {
                                                match &base_info.kind {
                                                    TypeKind::Class { symbol_id, .. } => {
                                                        Some(*symbol_id)
                                                    }
                                                    TypeKind::Abstract { symbol_id, .. } => {
                                                        Some(*symbol_id)
                                                    }
                                                    _ => None,
                                                }
                                            } else {
                                                None
                                            }
                                        }
                                        _ => None,
                                    }
                                } else {
                                    None
                                };

                            // Also try HIR type declarations for extern classes not in type_table
                            let class_symbol_id = class_symbol_id.or_else(|| {
                                if let Some(type_decl) = self.current_hir_types.get(&object_type) {
                                    if let HirTypeDecl::Class(class) = type_decl {
                                        return Some(class.symbol_id);
                                    }
                                }
                                None
                            });

                            // For static class calls where object.ty is invalid (TypeId::MAX),
                            // extract the class symbol directly from the object Variable expression
                            let class_symbol_id = class_symbol_id.or_else(|| {
                                if let HirExprKind::Variable {
                                    symbol: obj_sym, ..
                                } = &object.kind
                                {
                                    let sym = self.symbol_table.get_symbol(*obj_sym)?;
                                    if matches!(
                                        sym.kind,
                                        crate::tast::SymbolKind::Class
                                            | crate::tast::SymbolKind::Abstract
                                            | crate::tast::SymbolKind::TypeAlias
                                    ) {
                                        return Some(*obj_sym);
                                    }
                                }
                                None
                            });

                            if let Some(sym_id) = class_symbol_id {
                                // Get the class name from qualified_name or native_name
                                let class_name_from_obj =
                                    self.symbol_table.get_symbol(sym_id).and_then(|s| {
                                        // Prefer native_name (from @:native annotation)
                                        s.native_name
                                            .and_then(|nn| self.string_interner.get(nn))
                                            .map(|ns| ns.replace("::", "_"))
                                            .or_else(|| {
                                                s.qualified_name
                                                    .and_then(|qn| self.string_interner.get(qn))
                                                    .map(|s| s.replace(".", "_"))
                                            })
                                    });

                                let method_name_opt = self
                                    .symbol_table
                                    .get_symbol(*field)
                                    .and_then(|s| self.string_interner.get(s.name));

                                // PRIORITY: Use the field's qualified name to derive the class,
                                // since the field symbol knows which package it belongs to.
                                // This prevents sys.thread.Thread.sleep from being resolved
                                // via rayzor.concurrent.Thread when the class symbol is shared.
                                let class_name_from_field = self
                                    .symbol_table
                                    .get_symbol(*field)
                                    .and_then(|s| s.qualified_name)
                                    .and_then(|qn| self.string_interner.get(qn))
                                    .and_then(|qn_str| {
                                        let parts: Vec<&str> = qn_str.split('.').collect();
                                        if parts.len() >= 2 {
                                            Some(parts[..parts.len() - 1].join("_"))
                                        } else {
                                            None
                                        }
                                    });

                                let class_name_opt = class_name_from_field.or(class_name_from_obj);

                                // Also try bare class name as final fallback
                                let class_name_opt = class_name_opt.or_else(|| {
                                    self.symbol_table
                                        .get_symbol(sym_id)
                                        .and_then(|s| self.string_interner.get(s.name))
                                        .map(|s| s.to_string())
                                });

                                if let (Some(class_name), Some(method_name)) =
                                    (class_name_opt, method_name_opt)
                                {
                                    // Look up in stdlib_mapping (try abstract's own name first, then @:forward underlying)
                                    let stdlib_result = self
                                        .stdlib_mapping
                                        .find_by_name(&class_name, method_name)
                                        .map(|(sig, m)| (sig.clone(), m.clone()))
                                        .or_else(|| {
                                            // @:forward fallback: check if method should be forwarded to underlying type
                                            let (underlying_type, forward_list) =
                                                self.abstract_forward_rules.get(&sym_id)?;
                                            // Check if this method is in the forward list (empty = forward all)
                                            let method_interned = self
                                                .symbol_table
                                                .get_symbol(*field)
                                                .map(|s| s.name);
                                            let is_forwarded = forward_list.is_empty()
                                                || method_interned
                                                    .map_or(false, |n| forward_list.contains(&n));
                                            if !is_forwarded {
                                                return None;
                                            }
                                            // Resolve underlying type's class name
                                            let underlying_class = self
                                                .resolve_type_class_name_with(
                                                    &type_table,
                                                    *underlying_type,
                                                )?;
                                            self.stdlib_mapping
                                                .find_by_name(&underlying_class, method_name)
                                                .map(|(sig, m)| (sig.clone(), m.clone()))
                                        });

                                    if let Some((sig, mapping)) = stdlib_result {
                                        // Extract data before dropping borrows
                                        let is_mir_wrapper = mapping.is_mir_wrapper;
                                        let runtime_name = mapping.runtime_name.to_string();
                                        let has_return = mapping.has_return;
                                        let returns_raw_value = mapping.returns_raw_value;
                                        let raw_value_params = mapping.raw_value_params;
                                        let extend_to_i64_params = mapping.extend_to_i64_params;
                                        let explicit_return_type =
                                            mapping.return_type.map(|rt| rt.to_ir_type());
                                        let mapping_is_static = sig.is_static;

                                        // First, try special runtime calls that need custom MIR lowering
                                        // (e.g., Reflect.callMethod, Reflect.makeVarArgs, Type.typeof)
                                        if let Some(special_result) = self
                                            .try_lower_special_runtime_call(
                                                &runtime_name,
                                                args,
                                                result_type.clone(),
                                                expr.source_location,
                                            )
                                        {
                                            return special_result;
                                        }

                                        // Std.string on ValueType enum values needs special routing
                                        // (same check that trace/interpolation paths do)
                                        if runtime_name == "haxe_std_string_ptr" && args.len() == 1
                                        {
                                            if self.expr_is_value_type_expr(&args[0]) {
                                                let arg_reg = self.lower_expression(&args[0])?;
                                                return self.convert_value_type_to_string(arg_reg);
                                            }
                                        }

                                        // Reflect.compare: detect arg types from expressions and
                                        // route to haxe_reflect_compare_typed. This avoids boxing
                                        // and ensures string comparison works correctly.
                                        // Must be done before the arg boxing loop below.
                                        if runtime_name == "haxe_reflect_compare" && args.len() >= 2
                                        {
                                            let type_info =
                                                self.infer_reflect_compare_type_info(args);
                                            if let Some(info) = type_info {
                                                let mut arg_regs = Vec::new();
                                                for arg in args.iter() {
                                                    if let Some(reg) = self.lower_expression(arg) {
                                                        let reg_ty = self
                                                            .builder
                                                            .get_register_type(reg)
                                                            .unwrap_or(IrType::I64);
                                                        let final_reg = if reg_ty != IrType::I64 {
                                                            self.builder
                                                                .build_cast(
                                                                    reg,
                                                                    reg_ty,
                                                                    IrType::I64,
                                                                )
                                                                .unwrap_or(reg)
                                                        } else {
                                                            reg
                                                        };
                                                        arg_regs.push(final_reg);
                                                    }
                                                }
                                                let tag_reg = match info {
                                                    Ok(tag_value) => self
                                                        .builder
                                                        .build_const(IrValue::I32(tag_value))?,
                                                    Err(type_param_name) => {
                                                        // Generic: placeholder tag with fixup
                                                        let tag = self
                                                            .builder
                                                            .build_const(IrValue::I32(0))?;
                                                        if let Some(func) =
                                                            self.builder.current_function_mut()
                                                        {
                                                            func.type_param_tag_fixups
                                                                .push((tag, type_param_name));
                                                        }
                                                        tag
                                                    }
                                                };
                                                arg_regs.push(tag_reg);
                                                let extern_func_id = self
                                                    .get_or_register_extern_function(
                                                        "haxe_reflect_compare_typed",
                                                        vec![IrType::I64, IrType::I64, IrType::I32],
                                                        IrType::I64,
                                                    );
                                                let call_result = self.builder.build_call_direct(
                                                    extern_func_id,
                                                    arg_regs,
                                                    IrType::I64,
                                                )?;
                                                if result_type == IrType::I32 {
                                                    return self.builder.build_cast(
                                                        call_result,
                                                        IrType::I64,
                                                        IrType::I32,
                                                    );
                                                }
                                                return Some(call_result);
                                            }
                                        }

                                        // Lower args, auto-boxing primitives when the MIR wrapper
                                        // expects Ptr(U8) (e.g., Channel<Int>.send(42)).
                                        // For instance methods, prepend object as receiver (param 0).
                                        // For static methods, skip the object — it's just a class ref.
                                        let mir_wrapper_sig =
                                            self.get_stdlib_mir_wrapper_signature(&runtime_name);
                                        let is_static_call = mapping_is_static || !*is_method;
                                        let mut arg_regs = Vec::new();
                                        if !is_static_call {
                                            let obj_reg = self.lower_expression(object)?;
                                            arg_regs.push(obj_reg);
                                        }
                                        for (i, arg) in args.iter().enumerate() {
                                            if let Some(reg) = self.lower_expression(arg) {
                                                let actual_ty = self.convert_type(arg.ty);
                                                // For instance methods, param 0 = receiver, user args start at i+1
                                                // For static methods, no receiver, user args start at i
                                                let param_idx =
                                                    if is_static_call { i } else { i + 1 };
                                                let expected_ty = mir_wrapper_sig
                                                    .as_ref()
                                                    .and_then(|(params, _)| {
                                                        params.get(param_idx).cloned()
                                                    })
                                                    .unwrap_or_else(|| actual_ty.clone());
                                                let final_reg = self.maybe_box_for_extern_call(
                                                    reg,
                                                    &actual_ty,
                                                    &expected_ty,
                                                )?;
                                                arg_regs.push(final_reg);
                                            }
                                        }

                                        if is_mir_wrapper {
                                            let param_types: Vec<_> = mir_wrapper_sig
                                                .as_ref()
                                                .map(|(params, _)| params.clone())
                                                .unwrap_or_else(|| {
                                                    arg_regs
                                                        .iter()
                                                        .map(|r| {
                                                            self.builder
                                                                .get_register_type(*r)
                                                                .unwrap_or(IrType::I64)
                                                        })
                                                        .collect()
                                                });

                                            let mir_return_type = mir_wrapper_sig
                                                .as_ref()
                                                .map(|(_, ret)| ret.clone())
                                                .unwrap_or_else(|| result_type.clone());

                                            let mir_func_id = self.register_stdlib_mir_forward_ref(
                                                &runtime_name,
                                                param_types,
                                                mir_return_type.clone(),
                                            );

                                            let call_result = self.builder.build_call_direct(
                                                mir_func_id,
                                                arg_regs,
                                                mir_return_type.clone(),
                                            )?;

                                            // Resolve generic return type from receiver's type arguments
                                            // For Channel<Int>.tryReceive() -> Null<Int>, result_type may be
                                            // Ptr(Void) (Dynamic) because the generic T is not resolved.
                                            // We resolve T from the receiver object's actual type args.
                                            let resolved_result = {
                                                let needs_resolve = result_type == IrType::Any
                                                    || matches!(&result_type, IrType::Ptr(inner) if matches!(inner.as_ref(), IrType::Void))
                                                    || result_type == IrType::I64;
                                                if needs_resolve {
                                                    let type_table = self.type_table;
                                                    type_table.get(object.ty).and_then(|ti| {
                                                        if let crate::tast::TypeKind::Class { type_args, .. } = &ti.kind {
                                                            if !type_args.is_empty() {
                                                                Some(self.convert_type(type_args[0]))
                                                            } else {
                                                                None
                                                            }
                                                        } else if let crate::tast::TypeKind::GenericInstance { type_args, .. } = &ti.kind {
                                                            if !type_args.is_empty() {
                                                                Some(self.convert_type(type_args[0]))
                                                            } else {
                                                                None
                                                            }
                                                        } else {
                                                            None
                                                        }
                                                    }).unwrap_or_else(|| result_type.clone())
                                                } else {
                                                    result_type.clone()
                                                }
                                            };

                                            // MIR wrappers return values in their declared type directly —
                                            // no unboxing needed (they don't return boxed DynamicValue*).
                                            return Some(call_result);
                                        } else {
                                            // Inject hidden enum type_id arg for runtime enum helpers
                                            let pre = arg_regs.len();
                                            self.inject_hidden_enum_type_id_arg(
                                                &runtime_name,
                                                args,
                                                &mut arg_regs,
                                            );

                                            // Use explicit types from the types: descriptor when available
                                            let (param_types, return_type) = self
                                                .get_stdlib_mir_wrapper_signature(&runtime_name)
                                                .unwrap_or_else(|| {
                                                    let params: Vec<_> = arg_regs
                                                        .iter()
                                                        .enumerate()
                                                        .map(|(i, r)| {
                                                            let param_bit = 1u32 << i;
                                                            if raw_value_params & param_bit != 0 {
                                                                IrType::U64
                                                            } else if extend_to_i64_params
                                                                & param_bit
                                                                != 0
                                                            {
                                                                IrType::I64
                                                            } else {
                                                                self.builder
                                                                    .get_register_type(*r)
                                                                    .unwrap_or(IrType::I64)
                                                            }
                                                        })
                                                        .collect();
                                                    let ret = if let Some(ref ert) =
                                                        explicit_return_type
                                                    {
                                                        ert.clone()
                                                    } else if returns_raw_value {
                                                        IrType::U64
                                                    } else if has_return {
                                                        result_type.clone()
                                                    } else {
                                                        IrType::Void
                                                    };
                                                    (params, ret)
                                                });

                                            let extern_func_id = self
                                                .get_or_register_extern_function(
                                                    &runtime_name,
                                                    param_types,
                                                    return_type.clone(),
                                                );

                                            let call_result = self.builder.build_call_direct(
                                                extern_func_id,
                                                arg_regs,
                                                return_type.clone(),
                                            );

                                            // Auto-unbox if runtime returns Ptr(U8) but HIR expects primitive
                                            if let Some(call_reg) = call_result {
                                                return self.maybe_unbox_for_extern_return(
                                                    call_reg,
                                                    &return_type,
                                                    &result_type,
                                                );
                                            }
                                            return call_result;
                                        }
                                    }
                                }
                            }
                        }

                        // First check if the object is Dynamic - handle auto-unbox for method calls
                        let type_table = self.type_table;
                        if let Some(type_info) = type_table.get(object_type) {
                            if matches!(type_info.kind, TypeKind::Dynamic) {
                                // Dynamic method call - need to resolve method by name.
                                //
                                // EXCLUDE the currently-compiling function from
                                // candidates. Otherwise a method whose name
                                // matches the enclosing function (e.g.,
                                // `ArchRegistry.build` calling
                                // `(arch:Dynamic).build(...)` on an
                                // `ArchBuilder` instance) silently resolves
                                // back to the enclosing function → infinite
                                // recursion → stack overflow.
                                let method_name =
                                    self.symbol_table.get_symbol(*field).map(|s| s.name);
                                let caller_func_id = self.builder.current_function;
                                if let Some(name) = method_name {
                                    // Look up function by name in function_map.
                                    // Tighten by arity to avoid grabbing
                                    // same-named methods on unrelated classes.
                                    let target_argc = args.len() + 1; // +1 for receiver
                                    let mut found_func = None;
                                    for (sym, &func_id) in &self.function_map {
                                        if Some(func_id) == caller_func_id {
                                            continue;
                                        }
                                        if let Some(sym_info) = self.symbol_table.get_symbol(*sym) {
                                            if sym_info.name != name {
                                                continue;
                                            }
                                        } else {
                                            continue;
                                        }
                                        if let Some(func) =
                                            self.builder.module.functions.get(&func_id)
                                        {
                                            if func.signature.parameters.len() != target_argc {
                                                continue;
                                            }
                                        }
                                        found_func = Some(func_id);
                                        break;
                                    }

                                    if let Some(func_id) = found_func {
                                        // Lower the object and unbox it
                                        let obj_reg = self.lower_expression(object)?;

                                        // Unbox the Dynamic to get the actual object pointer
                                        let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                                        let unbox_func_id = self.get_or_register_extern_function(
                                            "haxe_unbox_reference_ptr",
                                            vec![ptr_u8.clone()],
                                            ptr_u8.clone(),
                                        );
                                        let unboxed_obj = self.builder.build_call_direct(
                                            unbox_func_id,
                                            vec![obj_reg],
                                            ptr_u8,
                                        )?;

                                        // Lower the arguments
                                        let arg_regs: Vec<_> =
                                            std::iter::once(unboxed_obj) // Add unboxed 'this' as first arg
                                                .chain(
                                                    args.iter()
                                                        .filter_map(|a| self.lower_expression(a)),
                                                )
                                                .collect();

                                        // Get the function's actual return type
                                        let actual_return_type = if let Some(func) =
                                            self.builder.module.functions.get(&func_id)
                                        {
                                            func.signature.return_type.clone()
                                        } else {
                                            result_type.clone()
                                        };

                                        return self.builder.build_call_direct(
                                            func_id,
                                            arg_regs,
                                            actual_return_type,
                                        );
                                    }
                                }
                            }
                        }

                        // Check if the object type is a String - handle String method calls
                        {
                            let type_table = self.type_table;
                            if let Some(type_info) = type_table.get(object_type) {
                                debug!(
                                    "[CHECK STRING] object_type={:?}, kind={:?}",
                                    object_type, type_info.kind
                                );
                                if matches!(type_info.kind, TypeKind::String) {
                                    // Get the method name from the field symbol
                                    let method_name = self
                                        .symbol_table
                                        .get_symbol(*field)
                                        .and_then(|s| self.string_interner.get(s.name));

                                    if let Some(method_name) = method_name {
                                        // For String methods with optional params (indexOf, lastIndexOf),
                                        // look up the mapping by param count to get the right variant
                                        let arg_count = args.len();
                                        let mapping_opt = self
                                            .stdlib_mapping
                                            .find_by_name_and_params(
                                                "String",
                                                method_name,
                                                arg_count,
                                            )
                                            .or_else(|| {
                                                self.stdlib_mapping
                                                    .find_by_name("String", method_name)
                                            });

                                        // Look up the runtime function for this String method
                                        if let Some((_sig, mapping)) = mapping_opt {
                                            let runtime_func = mapping.runtime_name;

                                            debug!(
                                                "[STRING METHOD] Found String.{} with {} args -> {}",
                                                method_name, arg_count, runtime_func
                                            );

                                            // Lower the object (the String pointer)
                                            let obj_reg = self.lower_expression(object)?;

                                            // Lower the method arguments
                                            let method_arg_regs: Vec<_> = args
                                                .iter()
                                                .filter_map(|a| self.lower_expression(a))
                                                .collect();

                                            // Build param types: string_ptr, ...args
                                            let string_ptr_ty =
                                                IrType::Ptr(Box::new(IrType::String));
                                            let mut param_types = vec![string_ptr_ty.clone()];
                                            for arg in &method_arg_regs {
                                                // Haxe Int is i32, default to I32 for integer args
                                                let arg_ty = self
                                                    .builder
                                                    .get_register_type(*arg)
                                                    .unwrap_or(IrType::I32);
                                                param_types.push(arg_ty);
                                            }

                                            // Determine return type - for String methods returning String,
                                            // they return a pointer to HaxeString
                                            let return_type = if result_type == IrType::String {
                                                string_ptr_ty.clone()
                                            } else {
                                                result_type.clone()
                                            };

                                            let runtime_func_id = self
                                                .get_or_register_extern_function(
                                                    runtime_func,
                                                    param_types,
                                                    return_type.clone(),
                                                );

                                            let mut call_args = vec![obj_reg];
                                            call_args.extend(method_arg_regs);

                                            return self.builder.build_call_direct(
                                                runtime_func_id,
                                                call_args,
                                                return_type,
                                            );
                                        }
                                    }
                                }
                            }
                        }

                        // Check if the object type is a rayzor stdlib class (or GenericInstance like Deque<Int>)
                        let type_table = self.type_table;
                        let mut class_symbol_id = if let Some(type_info) =
                            type_table.get(object_type)
                        {
                            match &type_info.kind {
                                TypeKind::Class { symbol_id, .. } => Some(*symbol_id),
                                TypeKind::Abstract { symbol_id, .. } => Some(*symbol_id),
                                TypeKind::GenericInstance { base_type, .. } => {
                                    // For GenericInstance like Deque<Int>, get the base class/abstract symbol
                                    if let Some(base_info) = type_table.get(*base_type) {
                                        match &base_info.kind {
                                            TypeKind::Class { symbol_id, .. } => {
                                                debug!(
                                                    "[STDLIB FALLBACK] GenericInstance base class symbol_id={:?}",
                                                    symbol_id
                                                );
                                                Some(*symbol_id)
                                            }
                                            TypeKind::Abstract { symbol_id, .. } => {
                                                debug!(
                                                    "[STDLIB FALLBACK] GenericInstance base abstract symbol_id={:?}",
                                                    symbol_id
                                                );
                                                Some(*symbol_id)
                                            }
                                            _ => None,
                                        }
                                    } else {
                                        None
                                    }
                                }
                                _ => None,
                            }
                        } else {
                            None
                        };

                        // Fallback for static class references where object_type is not concrete
                        // (e.g., extern class identifiers like Std/Math).
                        if class_symbol_id.is_none() {
                            if let HirExprKind::Variable {
                                symbol: object_symbol,
                                ..
                            } = &object.kind
                            {
                                if let Some(sym) = self.symbol_table.get_symbol(*object_symbol) {
                                    if matches!(
                                        sym.kind,
                                        crate::tast::symbols::SymbolKind::Class
                                            | crate::tast::symbols::SymbolKind::Abstract
                                            | crate::tast::symbols::SymbolKind::TypeAlias
                                    ) {
                                        class_symbol_id = Some(sym.id);
                                    }
                                }
                            }
                        }

                        if let Some(symbol_id) = class_symbol_id {
                            if let Some(class_symbol) = self.symbol_table.get_symbol(symbol_id) {
                                if let Some(class_name) =
                                    self.string_interner.get(class_symbol.name)
                                {
                                    debug!(
                                        "[STDLIB FALLBACK] Found class '{}', checking for stdlib method",
                                        class_name
                                    );

                                    // Check if it's a rayzor stdlib class by using native name or qualified name
                                    let qualified_name_opt = class_symbol
                                        .native_name
                                        .and_then(|nn| self.string_interner.get(nn))
                                        .map(|n| n.replace("::", "_"))
                                        .or_else(|| {
                                            class_symbol
                                                .qualified_name
                                                .and_then(|qn| self.string_interner.get(qn))
                                                .map(|s| s.to_string())
                                        });

                                    // Try to get the method name from the field symbol
                                    let method_name = if let Some(field_sym) =
                                        self.symbol_table.get_symbol(*field)
                                    {
                                        self.string_interner.get(field_sym.name)
                                    } else {
                                        None
                                    };

                                    if let Some(method_name) = method_name {
                                        let static_args = self.effective_static_call_args(args);
                                        let object_qualified_name_opt =
                                            if let HirExprKind::Variable {
                                                symbol: object_symbol,
                                                ..
                                            } = &object.kind
                                            {
                                                self.symbol_table
                                                    .get_symbol(*object_symbol)
                                                    .and_then(|s| s.qualified_name)
                                                    .and_then(|qn| self.string_interner.get(qn))
                                                    .map(|s| s.to_string())
                                            } else {
                                                None
                                            };

                                        // Prefer class-qualified lookup when a qualified class name
                                        // exists, but always keep a global static fallback.
                                        // Some extern classes (e.g. Math) may not carry
                                        // qualified/native names on the symbol.
                                        let runtime_func_opt = qualified_name_opt
                                            .as_deref()
                                            .and_then(|class_qualified_name| {
                                                let lookup =
                                                    format!("{}.{}", class_qualified_name, method_name);
                                                self.get_static_stdlib_runtime_func_with_params(
                                                    &lookup,
                                                    method_name,
                                                    static_args.len(),
                                                )
                                            })
                                            .or_else(|| {
                                                object_qualified_name_opt
                                                    .as_deref()
                                                    .and_then(|class_qualified_name| {
                                                        let lookup = format!(
                                                            "{}.{}",
                                                            class_qualified_name, method_name
                                                        );
                                                        self.get_static_stdlib_runtime_func_with_params(
                                                            &lookup,
                                                            method_name,
                                                            static_args.len(),
                                                        )
                                                    })
                                            })
                                            .or_else(|| {
                                                let lookup =
                                                    format!("{}.{}", class_name, method_name);
                                                self.get_static_stdlib_runtime_func_with_params(
                                                    &lookup,
                                                    method_name,
                                                    static_args.len(),
                                                )
                                            })
                                            .or_else(|| {
                                                self.stdlib_mapping
                                                    .find_static_method_by_name_and_params(
                                                        method_name,
                                                        static_args.len(),
                                                    )
                                                    .map(|(_, mapping)| mapping.runtime_name)
                                            });

                                        if let Some(runtime_func) = runtime_func_opt {
                                            // println!("✅ Generating runtime call to {} for {}.{}", runtime_func, class_name, method_name);

                                            // Lower all arguments (don't include object for static methods like spawn)
                                            let arg_regs: Vec<_> = static_args
                                                .iter()
                                                .filter_map(|a| self.lower_expression(a))
                                                .collect();
                                            debug!(
                                                "[FIELD-PATH STATIC] Dispatching {}.{} -> {}, arg_count={}",
                                                class_name,
                                                method_name,
                                                runtime_func,
                                                arg_regs.len()
                                            );

                                            // Use the function signature from the mapping (hlp_* introspection)
                                            // if available; this is the authoritative source of type info.
                                            let (expected_param_types, expected_return_type) = self
                                                .get_extern_function_signature(&runtime_func)
                                                .unwrap_or_else(|| {
                                                    let param_types: Vec<IrType> = arg_regs
                                                        .iter()
                                                        .map(|reg| {
                                                            self.builder
                                                                .get_register_type(*reg)
                                                                .unwrap_or(IrType::Any)
                                                        })
                                                        .collect();
                                                    (param_types, result_type.clone())
                                                });

                                            // Cast/box arguments to expected types
                                            let final_arg_regs: Vec<_> = arg_regs.iter().enumerate()
                                                    .map(|(i, &reg)| {
                                                        if let (Some(expected_ty), Some(actual_ty)) = (
                                                            expected_param_types.get(i),
                                                            self.builder.get_register_type(reg)
                                                        ) {
                                                            if *expected_ty != actual_ty {
                                                                let is_ptr_u8 = matches!(expected_ty, IrType::Ptr(inner) if matches!(inner.as_ref(), IrType::U8));
                                                                if is_ptr_u8 && i < args.len() {
                                                                    if let Some(boxed) = self.box_value_for_dynamic(reg, args[i].ty) {
                                                                        return boxed;
                                                                    }
                                                                }
                                                                if let Some(casted) = self.builder.build_cast(reg, actual_ty.clone(), expected_ty.clone()) {
                                                                    return casted;
                                                                }
                                                            }
                                                        }
                                                        reg
                                                    })
                                                    .collect();

                                            let runtime_func_id = self
                                                .get_or_register_extern_function(
                                                    &runtime_func,
                                                    expected_param_types,
                                                    expected_return_type.clone(),
                                                );

                                            // Generate the call to the runtime function
                                            let call_result = self.builder.build_call_direct(
                                                runtime_func_id,
                                                final_arg_regs,
                                                expected_return_type.clone(),
                                            );
                                            return call_result;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Enum constructors can arrive as field callees for imported
                // modules, e.g. `ForeignMetaish.U32(2048)`. Lower those here
                // before the callee expression itself turns `Enum.Variant`
                // into a tag-only value and drops the payload arguments.
                if let HirExprKind::Field { object, field } = &callee.kind {
                    if let HirExprKind::Variable {
                        symbol: enum_symbol,
                        ..
                    } = &object.kind
                    {
                        if let Some(enum_sym) = self.symbol_table.get_symbol(*enum_symbol) {
                            if enum_sym.kind == crate::tast::SymbolKind::Enum {
                                let field_sym = self.symbol_table.get_symbol(*field);
                                let field_name = field_sym
                                    .and_then(|s| self.string_interner.get(s.name))
                                    .unwrap_or("");

                                if let Some(variants) =
                                    self.symbol_table.get_enum_variants(*enum_symbol)
                                {
                                    for (idx, variant_id) in variants.iter().enumerate() {
                                        let variant_sym = self.symbol_table.get_symbol(*variant_id);
                                        let variant_name = variant_sym
                                            .and_then(|s| self.string_interner.get(s.name))
                                            .unwrap_or("");
                                        let id_match = *variant_id == *field;
                                        let name_match = !id_match && variant_name == field_name;

                                        if id_match || name_match {
                                            let field_count = self
                                                .get_enum_variant_field_count(*enum_symbol, idx);
                                            if field_count == 0 {
                                                if self.enum_is_boxed(*enum_symbol) {
                                                    return self
                                                        .build_boxed_enum_tag_only(idx as i32);
                                                }
                                                return self
                                                    .builder
                                                    .build_const(IrValue::I64(idx as i64));
                                            }

                                            let constructor_args = if *is_method
                                                && !args.is_empty()
                                                && self.is_enum_symbol_expr(&args[0])
                                            {
                                                &args[1..]
                                            } else {
                                                args
                                            };
                                            return self.build_boxed_enum_with_fields(
                                                idx as i32,
                                                field_count,
                                                constructor_args,
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Check if callee is an enum constructor (EnumVariant symbol kind)
                // Handle enum constructors with parameters like MyResult.Ok(42)
                if let HirExprKind::Variable { symbol, .. } = &callee.kind {
                    if let Some(sym) = self.symbol_table.get_symbol(*symbol) {
                        use crate::tast::SymbolKind;
                        if sym.kind == SymbolKind::EnumVariant
                            || (sym.kind == SymbolKind::Function
                                && self
                                    .symbol_table
                                    .find_parent_enum_for_constructor(*symbol)
                                    .is_some())
                        {
                            // Find the parent enum and variant index
                            if let Some(parent_enum_id) =
                                self.symbol_table.find_parent_enum_for_constructor(*symbol)
                            {
                                if let Some(variants) =
                                    self.symbol_table.get_enum_variants(parent_enum_id)
                                {
                                    for (idx, variant_id) in variants.iter().enumerate() {
                                        if *variant_id == *symbol {
                                            // Get variant field count from HIR
                                            let field_count = self
                                                .get_enum_variant_field_count(parent_enum_id, idx);

                                            if field_count == 0 {
                                                // If enum has parameterized variants, all variants must be boxed
                                                if self.enum_is_boxed(parent_enum_id) {
                                                    return self
                                                        .build_boxed_enum_tag_only(idx as i32);
                                                }
                                                // Pure discriminant enum - return index directly
                                                return self
                                                    .builder
                                                    .build_const(IrValue::I64(idx as i64));
                                            }

                                            // Has parameters - allocate boxed enum struct
                                            // Layout: [tag:i32][pad:i32][field0:i64][field1:i64]...
                                            let struct_size = 8 + 8 * field_count; // 8 for tag+pad, 8 per field

                                            // Allocate memory
                                            let size_const = self
                                                .builder
                                                .build_const(IrValue::I64(struct_size as i64))?;
                                            let alloc_func = self.get_or_register_extern_function(
                                                "malloc",
                                                vec![IrType::I64],
                                                IrType::Ptr(Box::new(IrType::I8)),
                                            );
                                            let ptr = self.builder.build_call_direct(
                                                alloc_func,
                                                vec![size_const],
                                                IrType::Ptr(Box::new(IrType::I8)),
                                            )?;

                                            // Store tag at offset 0 (as i32)
                                            // Note: GEP multiplies index by element size, so we use I8 elements
                                            // for byte-based addressing, then bitcast to the target type
                                            let zero_offset =
                                                self.builder.build_const(IrValue::I64(0))?;
                                            let tag_ptr = self.builder.build_gep(
                                                ptr,
                                                vec![zero_offset],
                                                IrType::Ptr(Box::new(IrType::I8)), // Byte-based
                                            )?;
                                            let tag_ptr_i32 = self.builder.build_bitcast(
                                                tag_ptr,
                                                IrType::Ptr(Box::new(IrType::I32)),
                                            )?;
                                            let tag_val = self
                                                .builder
                                                .build_const(IrValue::I32(idx as i32))?;
                                            self.builder.build_store(tag_ptr_i32, tag_val)?;

                                            // Store each parameter at byte offset 8 + i*8
                                            // When is_method=true, args[0] is the enum class reference
                                            // (receiver), not a constructor field. Skip it.
                                            let constructor_args: &[HirExpr] = if *is_method {
                                                if args.len() > 1 {
                                                    &args[1..]
                                                } else {
                                                    &[]
                                                }
                                            } else {
                                                args
                                            };
                                            for (i, arg) in constructor_args.iter().enumerate() {
                                                let arg_reg = self.lower_expression(arg)?;
                                                let field_offset = self.builder.build_const(
                                                    IrValue::I64((8 + i * 8) as i64),
                                                )?;
                                                // Use I8 element type for byte-based addressing
                                                let field_ptr = self.builder.build_gep(
                                                    ptr,
                                                    vec![field_offset],
                                                    IrType::Ptr(Box::new(IrType::I8)),
                                                )?;
                                                // Bitcast to i64 ptr for the store
                                                let field_ptr_i64 = self.builder.build_bitcast(
                                                    field_ptr,
                                                    IrType::Ptr(Box::new(IrType::I64)),
                                                )?;
                                                self.builder.build_store(field_ptr_i64, arg_reg)?;
                                            }

                                            // Return pointer as i64 for uniform handling
                                            // (bitcast pointer to i64)
                                            return self.builder.build_bitcast(ptr, IrType::I64);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Check if callee is a direct function reference
                if let HirExprKind::Variable { symbol, .. } = &callee.kind {
                    // Virtual dispatch for instance method calls (is_method=true):
                    // Skip vtable dispatch for super.method() calls — these must call
                    // the parent's implementation directly, not the overridden version.
                    let receiver_is_super =
                        !args.is_empty() && matches!(args[0].kind, HirExprKind::Super);
                    // super.method() — bypass vtable AND resolve to parent's implementation.
                    if receiver_is_super {
                        let method_name = self.symbol_table.get_symbol(*symbol).map(|s| s.name);
                        if let Some(method_name) = method_name {
                            // Find parent class: determine which class the current function
                            // belongs to, then look up its parent via class_parent_map.
                            let current_class = self.builder.current_function().and_then(|f| {
                                // Find the class this function is a method of
                                self.class_method_by_name
                                    .iter()
                                    .find(|(_, &method_sym)| {
                                        self.function_map.get(&method_sym) == Some(&f.id)
                                    })
                                    .map(|((class_sym, _), _)| *class_sym)
                            });
                            let parent_class = current_class
                                .and_then(|cls| self.class_parent_map.get(&cls).copied());
                            // Resolve parent's method by name
                            let super_func_id = parent_class
                                .and_then(|pc| {
                                    self.class_method_by_name.get(&(pc, method_name)).and_then(
                                        |&sym| {
                                            self.resolve_function_id_with_qualified_fallback(sym)
                                        },
                                    )
                                })
                                .or_else(|| {
                                    // Fallback: direct symbol resolution
                                    self.resolve_function_id_with_qualified_fallback(*symbol)
                                });
                            if let Some(func_id) = super_func_id {
                                let obj_reg = self.lower_expression(&args[0])?;
                                let mut call_args = vec![obj_reg];
                                for arg in args.iter().skip(1) {
                                    if let Some(reg) = self.lower_expression(arg) {
                                        call_args.push(reg);
                                    }
                                }
                                let ret_type = self.convert_type(expr.ty);
                                return self
                                    .builder
                                    .build_call_direct(func_id, call_args, ret_type);
                            }
                        }
                    }
                    if *is_method && !args.is_empty() && !receiver_is_super {
                        let vtable_slot =
                            self.virtual_dispatch_info.get(symbol).copied().or_else(|| {
                                let method_name =
                                    self.symbol_table.get_symbol(*symbol).map(|s| s.name)?;
                                let receiver_type = self.resolve_through_aliases(args[0].ty);
                                let type_table = self.type_table;
                                let class_sym = match &type_table.get(receiver_type)?.kind {
                                    TypeKind::Class { symbol_id, .. } => Some(*symbol_id),
                                    _ => None,
                                }?;
                                let mut current = Some(class_sym);
                                while let Some(cls) = current {
                                    if let Some(&method_sym) =
                                        self.class_method_by_name.get(&(cls, method_name))
                                    {
                                        if let Some(info) =
                                            self.virtual_dispatch_info.get(&method_sym)
                                        {
                                            return Some(*info);
                                        }
                                    }
                                    current = self.class_parent_map.get(&cls).copied();
                                }
                                None
                            });

                        if let Some((slot_index, _defining_class)) = vtable_slot {
                            let obj_reg = self.lower_expression(&args[0])?;
                            let mut call_args = vec![obj_reg];
                            for arg in args.iter().skip(1) {
                                if let Some(reg) = self.lower_expression(arg) {
                                    call_args.push(reg);
                                }
                            }
                            let lookup_fn = self.get_or_register_extern_function(
                                "haxe_vtable_lookup",
                                vec![IrType::Ptr(Box::new(IrType::U8)), IrType::I32],
                                IrType::I64,
                            );
                            let slot_reg =
                                self.builder.build_const(IrValue::I32(slot_index as i32))?;
                            let fn_ptr = self.builder.build_call_direct(
                                lookup_fn,
                                vec![obj_reg, slot_reg],
                                IrType::I64,
                            )?;
                            let mut param_types = vec![IrType::Ptr(Box::new(IrType::Void))];
                            for arg in args.iter().skip(1) {
                                param_types.push(self.convert_type(arg.ty));
                            }
                            let return_type = Box::new(self.convert_type(expr.ty));
                            let func_signature = IrType::Function {
                                params: param_types,
                                return_type,
                                varargs: false,
                            };
                            return self.builder.build_call_indirect(
                                fn_ptr,
                                call_args,
                                func_signature,
                            );
                        }
                    }

                    let symbol_name = self
                        .symbol_table
                        .get_symbol(*symbol)
                        .and_then(|s| self.string_interner.get(s.name))
                        .unwrap_or("<unknown>");
                    debug!(
                        "DEBUG: Callee is Variable, symbol={:?} ({}), is_method={}, args.len()={}",
                        symbol,
                        symbol_name,
                        is_method,
                        args.len()
                    );

                    // DIRECT SYMBOL RESOLUTION:
                    // For static extension methods (using IntTools; → x.add(3)) and
                    // other user-defined method calls, try resolving the function by symbol ID first.
                    // This avoids bare-name collisions (e.g., user "add" vs "rayzor_ssl_cert_add").
                    // Only intercept for user-defined functions — extern/stdlib methods need the
                    // more specific handlers below (auto-boxing, runtime mapping, etc.).
                    //
                    // IMPORTANT: Skip this fast path when the receiver is Dynamic or Interface-typed,
                    // because those need special dispatch (unboxing / fat pointer extraction) handled below.
                    if let Some(func_id) = self.resolve_function_id_with_qualified_fallback(*symbol)
                    {
                        let is_user_defined = self
                            .builder
                            .module
                            .functions
                            .get(&func_id)
                            .map(|f| f.kind == crate::ir::functions::FunctionKind::UserDefined)
                            .unwrap_or(false);

                        // Check if receiver needs special dispatch (Dynamic unbox or Interface fat pointer)
                        let receiver_needs_special_dispatch = if *is_method && !args.is_empty() {
                            let receiver_type = self.resolve_through_aliases(args[0].ty);
                            let type_table = self.type_table;
                            type_table
                                .get(receiver_type)
                                .map(|t| {
                                    matches!(
                                        t.kind,
                                        TypeKind::Dynamic
                                            | TypeKind::Interface { .. }
                                            | TypeKind::TypeParameter { .. }
                                            | TypeKind::Placeholder { .. }
                                            | TypeKind::Unknown
                                    )
                                })
                                .unwrap_or(false)
                        } else {
                            false
                        };

                        // A generic instance method imported from another module
                        // resolves here to a cross-module forward-ref stub whose
                        // FunctionKind is not UserDefined (the real impl arrives via
                        // merge + fixup later). The is_user_defined gate would route
                        // it to the fallback path below, which does NOT attach the
                        // receiver's concrete type_args — so the monomorphizer can
                        // never specialize the imported generic method, and its whole
                        // call chain (e.g. an imported haxe.ds.BalancedTree.set ->
                        // setLoop -> balance -> compare) reaches codegen as generic
                        // trap stubs and SIGILLs. Route generic instance-method calls
                        // through the type_args-aware block regardless of kind, as
                        // long as the callee is not a genuine extern/intrinsic.
                        let (callee_has_type_params, callee_is_externish) = self
                            .builder
                            .module
                            .functions
                            .get(&func_id)
                            .map(|f| {
                                (
                                    !f.signature.type_params.is_empty(),
                                    matches!(
                                        f.kind,
                                        crate::ir::functions::FunctionKind::ExternC
                                            | crate::ir::functions::FunctionKind::Intrinsic
                                    ),
                                )
                            })
                            .unwrap_or((false, false));
                        // A callee registered as an extern (e.g. the iterator-protocol
                        // methods List.iterator / .keys, which lower to extern Imports
                        // resolved at link time) must NOT take the receiver-type_args
                        // route: attaching type_args would ask the monomorphizer to
                        // specialize an extern that has no body, producing an
                        // unresolvable `Import` symbol that makes finalize panic on the
                        // whole module. Real cross-module methods (BalancedTree.set) are
                        // not in extern_functions, so they still route correctly.
                        let callee_is_externish = callee_is_externish
                            || self.builder.module.extern_functions.contains_key(&func_id);
                        // The callee may be a cross-module function not yet present
                        // in this module (resolved to its eventual id but merged
                        // later), so callee_has_type_params is unreliable here. Detect
                        // genericity from the RECEIVER instead: a concrete generic
                        // class instance (Class/GenericInstance carrying type_args).
                        let receiver_is_generic_instance = *is_method && !args.is_empty() && {
                            let rt = self.resolve_through_aliases(args[0].ty);
                            self.type_table
                                .get(rt)
                                .map(|t| match &t.kind {
                                    TypeKind::GenericInstance { type_args, .. }
                                    | TypeKind::Class { type_args, .. } => !type_args.is_empty(),
                                    _ => false,
                                })
                                .unwrap_or(false)
                        };
                        let route_as_generic_method = *is_method
                            && !callee_is_externish
                            && (callee_has_type_params || receiver_is_generic_instance);

                        if (is_user_defined || route_as_generic_method)
                            && !receiver_needs_special_dispatch
                        {
                            // Lower args and, for instance method
                            // calls, apply call-boundary materialization
                            // (class→iface fat-ptr wrap, anon coercion).
                            // HIR params don't include `this`, so the
                            // user arg at index `i` corresponds to HIR
                            // param index `i - 1` when `is_method=true`.
                            // Without this wrap, passing a raw class
                            // instance to a method whose param is
                            // interface-typed stores a non-fat-ptr in
                            // any iface field the callee assigns to →
                            // later virtual dispatch on that field
                            // SIGSEGVs (e.g. `reg.register("llama",
                            // new LlamaArch())` in nue.arch).
                            // Static calls are left untouched: they
                            // already go through the static-call path
                            // earlier in this handler when receiver is
                            // missing, and routing them through
                            // `maybe_materialize_for_call` here has
                            // triggered Cranelift symbol-clash on
                            // stdlib MIR wrappers like `array_length`.
                            let arg_regs: Vec<_> = if *is_method {
                                args.iter()
                                    .enumerate()
                                    .filter_map(|(i, a)| {
                                        let reg = self.lower_expression(a)?;
                                        if i == 0 {
                                            Some(reg)
                                        } else {
                                            Some(self.maybe_materialize_for_call(
                                                a,
                                                reg,
                                                Some(func_id),
                                                i - 1,
                                            ))
                                        }
                                    })
                                    .collect()
                            } else {
                                args.iter()
                                    .filter_map(|a| self.lower_expression(a))
                                    .collect()
                            };

                            let actual_return_type =
                                if let Some(func) = self.builder.module.functions.get(&func_id) {
                                    func.signature.return_type.clone()
                                } else {
                                    result_type.clone()
                                };

                            // For generic class method calls, extract concrete type args
                            // from the receiver's type (e.g., Container<String>.get() → type_args=[String]).
                            // This enables the monomorphizer to specialize the function.
                            //
                            // The callee may be a cross-module function not yet merged
                            // into this module, so its type_params are invisible here
                            // (has_type_params is false). Fall back to the receiver
                            // signal: a concrete generic-class instance receiver means
                            // we still want to attach the receiver's type_args so the
                            // monomorphizer can specialize the imported generic method.
                            let has_type_params = self
                                .builder
                                .module
                                .functions
                                .get(&func_id)
                                .map(|f| !f.signature.type_params.is_empty())
                                .unwrap_or(false)
                                || (receiver_is_generic_instance && !callee_is_externish);

                            // Gather type_args: first from HIR call-site type_args, then from receiver's
                            // generic instance type_args, then from the converted HIR type_args computed
                            // earlier in the Call handler.
                            let call_type_args = if has_type_params {
                                if !converted_hir_type_args.is_empty() {
                                    converted_hir_type_args.clone()
                                } else if *is_method && !args.is_empty() {
                                    // Extract from receiver's GenericInstance / Class type_args
                                    let receiver_type = self.resolve_through_aliases(args[0].ty);
                                    let type_table = self.type_table;
                                    type_table
                                        .get(receiver_type)
                                        .and_then(|t| match &t.kind {
                                            TypeKind::GenericInstance { type_args, .. }
                                            | TypeKind::Class { type_args, .. } => {
                                                if type_args.is_empty() {
                                                    None
                                                } else {
                                                    Some(
                                                        type_args
                                                            .iter()
                                                            .map(|&ta| self.convert_type(ta))
                                                            .collect::<Vec<_>>(),
                                                    )
                                                }
                                            }
                                            _ => None,
                                        })
                                        .unwrap_or_default()
                                } else {
                                    vec![]
                                }
                            } else {
                                vec![]
                            };

                            if !call_type_args.is_empty() {
                                let result = self.builder.build_call_direct_with_type_args(
                                    func_id,
                                    arg_regs,
                                    actual_return_type,
                                    call_type_args,
                                );
                                // The function body uses type-erased I64 for all type param values.
                                // When the resolved concrete type differs (F64, I32, String, etc.),
                                // the caller's register has the right TYPE but the value is still
                                // the I64 bit pattern. After inlining + SRA, this becomes visible.
                                // Insert a bitcast for float types where i64→f64 reinterpretation
                                // is needed at the calling convention level.
                                if let Some(reg) = result {
                                    let reg_type =
                                        self.builder.get_register_type(reg).unwrap_or(IrType::I64);
                                    if matches!(reg_type, IrType::F64 | IrType::F32) {
                                        // Bitcast from i64 to f64 to ensure calling convention
                                        // uses the float register file
                                        return self.builder.build_bitcast(reg, reg_type);
                                    }
                                }
                                return result;
                            }
                            return self.builder.build_call_direct(
                                func_id,
                                arg_regs,
                                actual_return_type,
                            );
                        }
                    }

                    // INTERFACE DISPATCH:
                    // When is_method=true, args[0] is the receiver. If the receiver has
                    // an interface type, dispatch through the fat pointer vtable.
                    if *is_method && !args.is_empty() {
                        let receiver = &args[0];
                        let receiver_type = receiver.ty;

                        if let Some(iface_sym) = self.get_interface_symbol(receiver_type) {
                            let method_name_interned =
                                self.symbol_table.get_symbol(*symbol).map(|s| s.name);

                            if let Some(method_name_i) = method_name_interned {
                                let method_index =
                                    self.resolve_interface_method_names(iface_sym).and_then(
                                        |names| names.iter().position(|n| *n == method_name_i),
                                    );

                                if let Some(idx) = method_index {
                                    // Lower the receiver (fat pointer)
                                    let fat_ptr_raw = self.lower_expression(receiver)?;

                                    // The fat pointer may be stored as I64 - bitcast to Ptr if needed
                                    let fat_ptr_ty = self
                                        .builder
                                        .get_register_type(fat_ptr_raw)
                                        .unwrap_or(IrType::I64);
                                    let fat_ptr = if !matches!(fat_ptr_ty, IrType::Ptr(_)) {
                                        self.builder.build_bitcast(
                                            fat_ptr_raw,
                                            IrType::Ptr(Box::new(IrType::I64)),
                                        )?
                                    } else {
                                        fat_ptr_raw
                                    };

                                    // Lower the actual arguments (skip args[0] which is receiver)
                                    let arg_regs: Vec<_> = args[1..]
                                        .iter()
                                        .filter_map(|a| self.lower_expression(a))
                                        .collect();

                                    // Load object pointer from fat_ptr[0]
                                    let obj_ptr = self.builder.build_load(fat_ptr, IrType::I64)?;

                                    // Load function pointer from fat_ptr[(idx+1)*8]
                                    let fn_offset = self
                                        .builder
                                        .build_const(IrValue::I64(((idx + 1) * 8) as i64))?;
                                    let fn_slot = self.builder.build_ptr_add(
                                        fat_ptr,
                                        fn_offset,
                                        IrType::Ptr(Box::new(IrType::U8)),
                                    )?;
                                    let fn_ptr = self.builder.build_load(fn_slot, IrType::I64)?;

                                    // Build call args: self (obj_ptr) + user args
                                    let mut call_args = vec![obj_ptr];
                                    call_args.extend(arg_regs);

                                    // Build signature: (self: Ptr, args...) -> return_type
                                    let param_types = {
                                        let mut types = vec![IrType::Ptr(Box::new(IrType::Void))]; // self
                                        for arg in args[1..].iter() {
                                            types.push(self.convert_type(arg.ty));
                                        }
                                        types
                                    };
                                    // Resolve return type from the method's symbol type,
                                    // not expr.ty (which may be the interface type instead
                                    // of the method's return type in some TAST configurations)
                                    let (return_ir_type, resolved_ret_type_id) = self
                                        .resolve_interface_method_return_type_full(
                                            *symbol, expr.ty,
                                        );
                                    self.emit_iface_return_diagnostic(
                                        *symbol,
                                        expr.ty,
                                        resolved_ret_type_id,
                                        expr.source_location,
                                    );
                                    let return_type = Box::new(return_ir_type);
                                    let func_signature = IrType::Function {
                                        params: param_types,
                                        return_type,
                                        varargs: false,
                                    };

                                    let call_result = self.builder.build_call_indirect(
                                        fn_ptr,
                                        call_args,
                                        func_signature,
                                    )?;
                                    if let Some(real_ty) = resolved_ret_type_id {
                                        self.interface_call_result_types
                                            .insert(call_result, real_ty);
                                    }
                                    return Some(call_result);
                                }
                            }
                        }
                    }

                    // ENUM INSTANCE METHOD DISPATCH:
                    // Delegates to runtime functions registered in runtime_mapping.rs.
                    // Injects compile-time constants (type_id, is_boxed) as extra params.
                    if *is_method && !args.is_empty() {
                        if let Some(Some(result)) = self.try_dispatch_enum_method(*symbol, args) {
                            return Some(result);
                        }
                    }

                    // EARLY RESOLUTION: For typed instance method calls on USER classes,
                    // resolve to the import function BEFORE the extern class method dispatch.
                    // This prevents user methods like Point2D.add from being incorrectly
                    // matched to stdlib methods (sys_deque_add).
                    // Skip for classes that have runtime mappings (e.g., EReg) — those
                    // must go through get_stdlib_runtime_info for proper dispatch.
                    if *is_method && !args.is_empty() {
                        let method_name_i = self.symbol_table.get_symbol(*symbol).map(|s| s.name);
                        // Check if receiver class has runtime mappings (skip early resolution if so)
                        let receiver_has_runtime_mapping = {
                            let receiver_type = self.resolve_through_aliases(args[0].ty);
                            let type_table = self.type_table;
                            type_table
                                .get(receiver_type)
                                .and_then(|ti| {
                                    if let crate::tast::core::TypeKind::Class {
                                        symbol_id, ..
                                    } = &ti.kind
                                    {
                                        self.symbol_table
                                            .get_symbol(*symbol_id)
                                            .and_then(|sym| self.string_interner.get(sym.name))
                                            .map(|name| {
                                                self.stdlib_mapping.class_has_any_method(name)
                                            })
                                    } else {
                                        None
                                    }
                                })
                                .unwrap_or(false)
                        };
                        if let Some(mn) = method_name_i {
                            let resolved = if receiver_has_runtime_mapping {
                                None // Let runtime mapping handle it
                            } else {
                                self.resolve_method_function_id(args[0].ty, mn)
                            };
                            if let Some(func_id) = resolved {
                                if func_id.0 >= 100_000 {
                                    // Resolved to an import function — use it directly
                                    let mut arg_regs = Vec::new();
                                    for arg in args.iter() {
                                        if let Some(reg) = self.lower_expression(arg) {
                                            arg_regs.push(reg);
                                        }
                                    }
                                    self.coerce_args_for_cross_module_call(
                                        func_id,
                                        &mut arg_regs,
                                        false,
                                    );
                                    return self.builder.build_call_direct(
                                        func_id,
                                        arg_regs,
                                        result_type,
                                    );
                                }
                            }
                        }
                    }

                    // EXTERN CLASS METHOD HANDLING:
                    // When MethodCall is desugared to Call with Variable callee,
                    // is_method=true and args[0] is the receiver (for instance methods).
                    // For static methods, there is no receiver - all args are actual arguments.
                    // We need to check if this is an extern class method and redirect to runtime.
                    if *is_method && !args.is_empty() {
                        let receiver = &args[0];
                        // Resolve TypeAlias to get the actual receiver type
                        // (e.g., List<Int> may be wrapped in TypeAlias)
                        let receiver_type = self.resolve_through_aliases(receiver.ty);

                        // Extract receiver class hint for disambiguation (e.g., "rayzor_ds_Tensor")
                        let receiver_class_hint_owned: Option<String> =
                            if let HirExprKind::Variable {
                                symbol: recv_sym, ..
                            } = &receiver.kind
                            {
                                self.monomorphized_var_types
                                    .get(recv_sym)
                                    .map(|s| s.to_string())
                            } else {
                                None
                            };
                        // Fallback: use find_receiver_class_name if monomorphized_var_types didn't have it
                        let receiver_class_hint_owned = receiver_class_hint_owned
                            .or_else(|| self.find_receiver_class_name(receiver))
                            .or_else(|| {
                                // Fallback: check register_class_hints for the receiver's MIR register.
                                // This resolves class names for variables assigned from extern class
                                // constructors (e.g., `var map = new ObjectMap<K,V>()`), where the
                                // New handler stored a class hint on the result register.
                                if let HirExprKind::Variable {
                                    symbol: recv_sym, ..
                                } = &receiver.kind
                                {
                                    self.symbol_map
                                        .get(recv_sym)
                                        .and_then(|reg| self.register_class_hints.get(reg).cloned())
                                } else {
                                    None
                                }
                            });

                        // SIMD4f detection: the DECLARED (TAST) receiver type is the SOLE
                        // authority for SIMD-vector classification. Two independent hazards,
                        // both caused by SSA register-id REUSE corrupting the name/register
                        // class hint computed above, are corrected here:
                        //
                        //  1. A loop-carried phi accumulator (`vacc = vacc.add(..)`) inherits
                        //     a stale hint ("rayzor_Usize") from a register a prior Usize
                        //     value held — masking the real SIMD4f type and routing `.add()`
                        //     to Usize_add (scalar integer add on a vector — garbage).
                        //  2. Conversely, a Usize/Bytes receiver (plain address arithmetic)
                        //     inherits a stale SIMD hint ("rayzor_SIMD4f") from a register a
                        //     prior SIMD value held — mis-routing it into the SIMD4f arith
                        //     interception below, which builds a VectorBinOp fed i64 operands
                        //     that the LLVM tier rejects (panic / Cranelift-only fallback).
                        //
                        // Resolution: convert_type(receiver_type) decides. If it IS a vector,
                        // that class wins unconditionally (hazard 1). If it is NOT, any SIMD
                        // class named by the hint is stale and is discarded (hazard 2); a
                        // genuine chained SIMD receiver whose HIR type is opaque (Dynamic) is
                        // still recovered from the receiver register's type.
                        let ir_ty = self.convert_type(receiver_type);
                        let receiver_class_hint_owned = if ir_ty.is_vector() {
                            // Distinguish the integer companion SIMD4i32 (i32x4)
                            // from SIMD4f (f32x4) — both are vectors, but their
                            // instance methods (sum/get/set) map to different
                            // wrappers. Without this, SIMD4i32.sum() dispatched to
                            // SIMD4f_sum (f32 reduce) — masked on native (Cranelift
                            // reduces by SSA value type) but wrong on wasm.
                            Some(simd_vector_class(&ir_ty).to_string())
                        } else {
                            // receiver_type is NOT a vector: a SIMD class named by the
                            // name/register hint is stale (hazard 2) and MUST be rejected
                            // before it reaches the arith interception. A non-SIMD hint
                            // (e.g. a genuine Bytes/Usize) is left intact.
                            let non_simd_hint = receiver_class_hint_owned
                                .filter(|h| h != "rayzor_SIMD4f" && h != "rayzor_SIMD4i32");
                            if non_simd_hint.is_some() {
                                non_simd_hint
                            } else if self.type_is_native_named(receiver_type, "rayzor::Atomic") {
                                // Atomic's type-map returns Ptr<I32> (not a vector), so is_vector()
                                // never fires; resolve it by the abstract's @:native name instead.
                                Some("rayzor_Atomic".to_string())
                            } else if let crate::ir::hir::HirExprKind::Variable {
                                symbol: recv_sym,
                                ..
                            } = &receiver.kind
                            {
                                // Chained-call recovery: receiver's HIR type is Dynamic but its
                                // register was typed VecF32x4 by a previous SIMD4f call (e.g.
                                // b.sum() where b = a.sqrt()). Only trust the register type when
                                // it is genuinely a vector — a plain address register is I64.
                                self.symbol_map
                                    .get(recv_sym)
                                    .and_then(|reg| self.builder.get_register_type(*reg))
                                    .filter(|ty| ty.is_vector())
                                    .map(|ty| simd_vector_class(&ty).to_string())
                            } else {
                                None
                            }
                        };
                        let receiver_class_hint = receiver_class_hint_owned.as_deref();

                        // SIMD4f arithmetic METHODS (`a.add(b)` etc.) must compile
                        // to the same single vector instruction as the OPERATORS
                        // (`a + b`, lowered to VectorBinOp at ~19541). The default
                        // method-call path routes them to a MIR wrapper that
                        // mishandles the vector ABI and returns garbage (a SIMD4f
                        // value carried as I64/Ptr(Void)). Emit VectorBinOp
                        // directly. Restricted to rayzor_SIMD4f (f32x4); the i32x4
                        // companion is excluded because integer VectorBinOp
                        // miscompiles on the wasm backend.
                        if receiver_class_hint == Some("rayzor_SIMD4f") && args.len() == 2 {
                            let mname = self
                                .symbol_table
                                .get_symbol(*symbol)
                                .and_then(|s| self.string_interner.get(s.name));
                            let vbop = match mname {
                                Some("add") => Some(BinaryOp::Add),
                                Some("sub") => Some(BinaryOp::Sub),
                                Some("mul") => Some(BinaryOp::Mul),
                                Some("div") => Some(BinaryOp::Div),
                                _ => None,
                            };
                            // Defense-in-depth: the receiver operand's own DECLARED type
                            // must itself classify as f32x4. The hint is no longer trusted
                            // in isolation — this refuses to build a VectorBinOp over a
                            // non-vector operand (the failure mode a stale SIMD hint on a
                            // reused register would otherwise cause: VectorBinOp fed i64).
                            let operands_are_simd4f = {
                                let t = self.convert_type(args[0].ty);
                                t.is_vector() && simd_vector_class(&t) == "rayzor_SIMD4f"
                            };
                            if let Some(bin_op) = vbop.filter(|_| operands_are_simd4f) {
                                let lhs_reg = self.lower_expression(&args[0])?;
                                let rhs_reg = self.lower_expression(&args[1])?;
                                // vec_ty must ALWAYS be a vector: fall through both operand
                                // register types (a scalar-typed operand register is a bug)
                                // to the f32x4 default rather than emitting VectorBinOp{I64}.
                                let vec_ty = self
                                    .builder
                                    .get_register_type(lhs_reg)
                                    .filter(|t| matches!(t, IrType::Vector { .. }))
                                    .or_else(|| {
                                        self.builder
                                            .get_register_type(rhs_reg)
                                            .filter(|t| matches!(t, IrType::Vector { .. }))
                                    })
                                    .unwrap_or(IrType::Vector {
                                        element: Box::new(IrType::F32),
                                        count: 4,
                                    });
                                return self
                                    .builder
                                    .build_vector_binop(bin_op, lhs_reg, rhs_reg, vec_ty);
                            }
                        }

                        // Calculate actual param count (excluding the receiver) for overload disambiguation
                        // e.g., s.indexOf("World", 0) has args=[s, "World", 0], param_count=2
                        let param_count = args.len().saturating_sub(1);

                        // SIMD4f direct lookup: When receiver is known to be SIMD4f, bypass
                        // get_stdlib_runtime_info (whose FALLBACK2 excludes SIMD matches).
                        let runtime_info = if matches!(
                            receiver_class_hint,
                            Some("rayzor_SIMD4f") | Some("rayzor_SIMD4i32")
                        ) {
                            let simd_cls = receiver_class_hint.unwrap();
                            let method_name_str = self
                                .symbol_table
                                .get_symbol(*symbol)
                                .and_then(|s| self.string_interner.get(s.name));
                            if let Some(mn) = method_name_str {
                                self.stdlib_mapping
                                    .find_by_name_and_params(simd_cls, mn, param_count)
                                    .or_else(|| self.stdlib_mapping.find_by_name(simd_cls, mn))
                                    .map(|(sig, mapping)| (sig.class, sig.method, mapping))
                            } else {
                                None
                            }
                        } else if receiver_class_hint == Some("rayzor_Atomic") {
                            // Atomic direct lookup: bypass FALLBACK2 (mirror of SIMD4f).
                            let method_name_str = self
                                .symbol_table
                                .get_symbol(*symbol)
                                .and_then(|s| self.string_interner.get(s.name));
                            method_name_str.and_then(|mn| {
                                self.stdlib_mapping
                                    .find_by_name_and_params("rayzor_Atomic", mn, param_count)
                                    .or_else(|| {
                                        self.stdlib_mapping.find_by_name("rayzor_Atomic", mn)
                                    })
                                    .map(|(sig, mapping)| (sig.class, sig.method, mapping))
                            })
                        } else {
                            // Try to find stdlib runtime mapping for this method
                            self.get_stdlib_runtime_info(
                                *symbol,
                                receiver_type,
                                Some(param_count),
                                receiver_class_hint,
                            )
                        };
                        if let Some((class_name, method_name, runtime_call)) = runtime_info {
                            // Skip methods that need ptr_conversion - let them fall through to
                            // the existing handler which properly handles params_need_ptr_conversion
                            if runtime_call.params_need_ptr_conversion != 0 {
                                debug!(
                                    "[EXTERN METHOD VAR] Skipping {} - has ptr_conversion, using fallback path",
                                    runtime_call.runtime_name
                                );
                            } else {
                                let runtime_func = runtime_call.runtime_name;
                                let is_instance_method = runtime_call.has_self_param;
                                let is_mir_wrapper = runtime_call.is_mir_wrapper;
                                let returns_raw_value = runtime_call.returns_raw_value;
                                let raw_value_params = runtime_call.raw_value_params;
                                let extend_to_i64_params = runtime_call.extend_to_i64_params;
                                let has_return = runtime_call.has_return;
                                let explicit_return_type =
                                    runtime_call.return_type.map(|t| t.to_ir_type());
                                if std::env::var_os("RAYZOR_TRACE_STDLIB_DISPATCH").is_some() {
                                    eprintln!(
                                        "[EXTERN METHOD VAR] Redirecting {}.{} -> {} (instance={}, mir_wrapper={})",
                                        class_name, method_name, runtime_func, is_instance_method, is_mir_wrapper
                                    );
                                }
                                debug!(
                                    "[EXTERN METHOD VAR] Redirecting {}.{} -> {} (instance={}, mir_wrapper={})",
                                    class_name,
                                    method_name,
                                    runtime_func,
                                    is_instance_method,
                                    is_mir_wrapper
                                );

                                // MIR wrapper path: use register_stdlib_mir_forward_ref
                                // MIR wrappers (SIMD4f, Thread, Channel, etc.) are compiled by
                                // Cranelift alongside user code. They must NOT be registered as
                                // extern C functions.
                                if is_mir_wrapper {
                                    self.builder.call_label =
                                        Some(format!("MIR_WRAPPER:{}", runtime_func));

                                    // Get the MIR wrapper's expected signature for auto-boxing/unboxing
                                    let mir_wrapper_sig =
                                        self.get_stdlib_mir_wrapper_signature(runtime_func);

                                    // Lower receiver + args with auto-boxing
                                    // When MIR wrapper expects Ptr(U8) but arg is a concrete primitive
                                    // (I32, F64, Bool from Channel<Int>.send(42)), box the value.
                                    // But for type-erased pointers (I64 from TypeParameter), just cast.
                                    let mut arg_regs = Vec::new();
                                    let mut param_types = Vec::new();
                                    for (i, arg) in args.iter().enumerate() {
                                        if i == 0 && !is_instance_method {
                                            continue; // Skip class receiver for static methods
                                        }
                                        if let Some(reg) = self.lower_expression(arg) {
                                            let actual_ty = self
                                                .builder
                                                .get_register_type(reg)
                                                .unwrap_or(IrType::I64);
                                            let param_idx =
                                                if is_instance_method { i } else { i - 1 };
                                            let expected_ty = mir_wrapper_sig
                                                .as_ref()
                                                .and_then(|(params, _)| {
                                                    params.get(param_idx).cloned()
                                                })
                                                .unwrap_or_else(|| actual_ty.clone());

                                            // Check if this arg is a type-erased pointer (I64 from
                                            // TypeParameter/class/GenericInstance/Array) vs a concrete
                                            // primitive (I32/F64/Bool). Type-erased pointers should
                                            // be CAST to Ptr(U8), not BOXED as integers.
                                            let is_type_erased_ptr =
                                                matches!(actual_ty, IrType::I64) && {
                                                    let type_table = self.type_table;
                                                    type_table
                                                        .get(arg.ty)
                                                        .map(|ti| {
                                                            matches!(
                                                    ti.kind,
                                                    crate::tast::TypeKind::TypeParameter { .. }
                                                    | crate::tast::TypeKind::Class { .. }
                                                    | crate::tast::TypeKind::GenericInstance { .. }
                                                    | crate::tast::TypeKind::Interface { .. }
                                                    | crate::tast::TypeKind::Dynamic
                                                    | crate::tast::TypeKind::Placeholder { .. }
                                                    | crate::tast::TypeKind::Array { .. }
                                                    | crate::tast::TypeKind::Abstract { .. }
                                                    | crate::tast::TypeKind::Function { .. }
                                                )
                                                        })
                                                        .unwrap_or(false)
                                                };

                                            let final_reg = if (runtime_func == "Channel_send"
                                                || runtime_func == "Channel_trySend")
                                                && i >= 1
                                            {
                                                // Uniformly box Channel payloads (refs too) so the
                                                // erased receive can tag-dispatch. i==0 is the channel
                                                // handle — never box it.
                                                self.box_channel_payload(
                                                    reg,
                                                    arg.ty,
                                                    &actual_ty,
                                                    &expected_ty,
                                                )?
                                            } else if is_type_erased_ptr
                                                && matches!(&expected_ty, IrType::Ptr(_))
                                            {
                                                // Cast I64 → Ptr(U8) for type-erased pointers
                                                self.builder
                                                    .build_cast(
                                                        reg,
                                                        IrType::I64,
                                                        expected_ty.clone(),
                                                    )
                                                    .unwrap_or(reg)
                                            } else {
                                                self.maybe_box_for_extern_call(
                                                    reg,
                                                    &actual_ty,
                                                    &expected_ty,
                                                )?
                                            };
                                            arg_regs.push(final_reg);
                                            param_types.push(expected_ty);
                                        }
                                    }

                                    // Use the MIR wrapper's actual return type instead of the
                                    // erased HIR type. For Dynamic/TypeParameter returns, the HIR
                                    // type erases to Ptr(Void) or I64, but the MIR wrapper returns
                                    // a concrete type (e.g., Ptr(U8)). Using the concrete type
                                    // prevents spurious unboxing in downstream field access.
                                    let mir_return_type = mir_wrapper_sig
                                        .map(|(_, ret)| ret)
                                        .unwrap_or_else(|| result_type.clone());

                                    let mir_func_id = self.register_stdlib_mir_forward_ref(
                                        runtime_func,
                                        param_types,
                                        mir_return_type.clone(),
                                    );

                                    let call_result = self.builder.build_call_direct(
                                        mir_func_id,
                                        arg_regs,
                                        mir_return_type.clone(),
                                    )?;

                                    // Store class hint for result to enable disambiguation
                                    // of subsequent method calls on TypeParameter receivers
                                    {
                                        let return_class =
                                            Self::get_return_class_hint(class_name, method_name);
                                        self.register_class_hints
                                            .insert(call_result, return_class.to_string());
                                    }

                                    // Auto-unbox if MIR wrapper returns Ptr(U8) but HIR expects primitive
                                    // (e.g., Thread<Int>.join() returns boxed int, Channel<Int>.tryReceive()
                                    // returns boxed int). Resolve T from receiver's generic type_args.
                                    let resolved_expected = {
                                        let needs_resolve = result_type == IrType::Any
                                            || matches!(&result_type, IrType::Ptr(inner) if matches!(inner.as_ref(), IrType::Void))
                                            || result_type == IrType::I64;
                                        if needs_resolve {
                                            let type_table = self.type_table;
                                            type_table
                                                .get(receiver_type)
                                                .and_then(|ti| match &ti.kind {
                                                    crate::tast::TypeKind::Class {
                                                        type_args,
                                                        ..
                                                    }
                                                    | crate::tast::TypeKind::GenericInstance {
                                                        type_args,
                                                        ..
                                                    } => {
                                                        if !type_args.is_empty() {
                                                            Some(self.convert_type(type_args[0]))
                                                        } else {
                                                            None
                                                        }
                                                    }
                                                    _ => None,
                                                })
                                                .unwrap_or_else(|| result_type.clone())
                                        } else {
                                            result_type.clone()
                                        }
                                    };

                                    // Thread<T>.join() boxes its result via haxe_box_int_ptr
                                    // (rayzor_thread_join), so for a concrete HEAP T the boxed
                                    // i64 payload IS the object pointer. maybe_unbox's raw
                                    // passthrough arm — shared with methods that return an
                                    // UN-boxed handle (e.g. Arc.get) — would skip the unbox and
                                    // hand back the box address (garbage). Unbox inline, keyed on
                                    // the wrapper so only the boxing method changes behavior.
                                    // Excludes Ptr(primitive) (Null<Int>), handled above.
                                    let resolved_is_heap_ptr = matches!(&resolved_expected, IrType::Ptr(inner) if !matches!(
                                        inner.as_ref(),
                                        IrType::I32
                                            | IrType::I64
                                            | IrType::F32
                                            | IrType::F64
                                            | IrType::Bool
                                    )) || matches!(
                                        resolved_expected,
                                        IrType::String
                                    );
                                    let mir_ret_is_ptr_u8 = matches!(&mir_return_type, IrType::Ptr(inner) if matches!(inner.as_ref(), IrType::U8 | IrType::Void));
                                    if runtime_func == "Thread_join"
                                        && resolved_is_heap_ptr
                                        && mir_ret_is_ptr_u8
                                    {
                                        let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                                        let unbox_id = self.get_or_register_extern_function(
                                            "haxe_unbox_int_ptr",
                                            vec![ptr_u8.clone()],
                                            IrType::I64,
                                        );
                                        let i64v = self.builder.build_call_direct(
                                            unbox_id,
                                            vec![call_result],
                                            IrType::I64,
                                        )?;
                                        return self.builder.build_cast(i64v, IrType::I64, ptr_u8);
                                    }

                                    // Channel payloads are uniformly boxed DynamicValues; route the
                                    // return through the tag-aware unbox (fixes erased prim receive,
                                    // recovers boxed refs) instead of the raw shared path.
                                    if runtime_func == "Channel_receive"
                                        || runtime_func == "Channel_tryReceive"
                                    {
                                        let is_try = runtime_func == "Channel_tryReceive";
                                        // Inferred channels erase T to I64, whose unbox
                                        // int-truncates a Float payload (4.75 -> 4). The
                                        // enclosing `var x:Float = ...` declared type is the
                                        // ground truth — refine to the float arm. Scoped to
                                        // floats + non-try (tryReceive keeps the tag-driven
                                        // unbox so nullables stay correct).
                                        let refined = if !is_try
                                            && matches!(resolved_expected, IrType::I64)
                                        {
                                            self.let_target_type_hint
                                                .map(|t| self.convert_type(t))
                                                .filter(|t| matches!(t, IrType::F64 | IrType::F32))
                                        } else {
                                            None
                                        };
                                        return self.unbox_channel_return(
                                            call_result,
                                            refined.as_ref().unwrap_or(&resolved_expected),
                                            is_try,
                                        );
                                    }

                                    return self.maybe_unbox_for_extern_return(
                                        call_result,
                                        &mir_return_type,
                                        &resolved_expected,
                                    );
                                }

                                // Extern C path: register as extern function
                                // Get expected parameter types from the extern function signature
                                let (expected_param_types, actual_return_type) = self
                                    .get_stdlib_mir_wrapper_signature(runtime_func)
                                    .map(|(params, ret)| (params, ret))
                                    .unwrap_or_else(|| {
                                        // When is_method=true, args[0] is always receiver/class - skip it
                                        // For instance methods, add self param first
                                        let mut params = if is_instance_method {
                                            vec![IrType::Ptr(Box::new(IrType::U8))]
                                        } else {
                                            vec![]
                                        };
                                        // Always skip args[0] since is_method=true
                                        // Use stdlib mapping hints for param types
                                        for (i, arg) in args.iter().skip(1).enumerate() {
                                            // raw_value_params: bit 0 = self, bit 1 = first user param, etc.
                                            let user_bit = 1u32 << (i + 1);
                                            if raw_value_params & user_bit != 0 {
                                                params.push(IrType::U64);
                                            } else if extend_to_i64_params & user_bit != 0 {
                                                params.push(IrType::I64);
                                            } else {
                                                params.push(self.convert_type(arg.ty));
                                            }
                                        }
                                        // Use explicit return type from mapping if available,
                                        // otherwise fall back to HIR-inferred result_type
                                        let ret_type = if returns_raw_value {
                                            IrType::U64
                                        } else if let Some(ref ert) = explicit_return_type {
                                            ert.clone()
                                        } else if has_return {
                                            result_type.clone()
                                        } else {
                                            IrType::Void
                                        };
                                        (params, ret_type)
                                    });

                                self.builder.call_label =
                                    Some(format!("EXTERN_C:{}", runtime_func));
                                // Build argument list based on whether this is instance or static method
                                let mut arg_regs = Vec::new();
                                let args_to_process: &[HirExpr] = if is_instance_method {
                                    let receiver_reg = self.lower_expression(receiver)?;
                                    arg_regs.push(receiver_reg);
                                    &args[1..]
                                } else {
                                    &args[1..]
                                };

                                // Lower the arguments and auto-box if needed
                                let param_offset = if is_instance_method { 1 } else { 0 };
                                for (i, arg) in args_to_process.iter().enumerate() {
                                    let arg_reg = self.lower_expression(arg)?;
                                    let actual_ty = self.convert_type(arg.ty);
                                    let expected_ty = expected_param_types
                                        .get(i + param_offset)
                                        .cloned()
                                        .unwrap_or_else(|| actual_ty.clone());
                                    let final_reg = self.maybe_box_for_extern_call(
                                        arg_reg,
                                        &actual_ty,
                                        &expected_ty,
                                    )?;
                                    arg_regs.push(final_reg);
                                }

                                // Use expected parameter types for registration
                                let param_types = if expected_param_types.len() == arg_regs.len() {
                                    expected_param_types.clone()
                                } else {
                                    let mut params = if is_instance_method {
                                        vec![IrType::Ptr(Box::new(IrType::U8))]
                                    } else {
                                        vec![]
                                    };
                                    for arg in args.iter().skip(1) {
                                        params.push(self.convert_type(arg.ty));
                                    }
                                    params
                                };

                                let extern_func_id = self.get_or_register_extern_function(
                                    runtime_func,
                                    param_types,
                                    actual_return_type.clone(),
                                );

                                let call_result = self.builder.build_call_direct(
                                    extern_func_id,
                                    arg_regs,
                                    actual_return_type.clone(),
                                )?;

                                // Handle returns_raw_value: cast raw U64 to the appropriate type
                                if returns_raw_value {
                                    // Compute the actual element type T from the receiver's
                                    // generic args. When T resolves (e.g. `Map<String, Tensor>`
                                    // → T = Tensor), the u64 returned by the runtime is a
                                    // pointer that needs bit-reinterpret to the right Ptr
                                    // type. When T is unresolved (`new StringMap()` with no
                                    // type args), the value is a primitive stored as raw
                                    // bits — keep on the U64→I64 cast path for downstream
                                    // unboxing.
                                    let resolved_value_ty: Option<IrType> = {
                                        let type_table = self.type_table;
                                        type_table.get(receiver_type).and_then(|ti| {
                                            match &ti.kind {
                                                crate::tast::TypeKind::Class {
                                                    type_args, ..
                                                }
                                                | crate::tast::TypeKind::GenericInstance {
                                                    type_args,
                                                    ..
                                                } => {
                                                    // Value T is the LAST type arg: StringMap<T> /
                                                    // IntMap<T> have type_args=[T]; ObjectMap<K, V>
                                                    // has type_args=[K, V] — the value lives at the
                                                    // tail in both cases.
                                                    type_args
                                                        .last()
                                                        .map(|ta| self.convert_type(*ta))
                                                }
                                                _ => None,
                                            }
                                        })
                                    };
                                    // When the HIR result_type is opaque (Any/I64/Ptr<U8>) because
                                    // the Haxe surface declares `Null<V>` for Map.get, the *real*
                                    // value type is `resolved_value_ty` (V from receiver's type
                                    // args). For F64/F32 V, bitcast the raw u64 bits directly to
                                    // float rather than try to unbox a DynamicValue (the runtime
                                    // stores raw bits, not a heap-boxed value, so the unbox path
                                    // would mis-interpret the bytes).
                                    let result_is_opaque = match &result_type {
                                        IrType::Any | IrType::I64 => true,
                                        IrType::Ptr(inner) => {
                                            matches!(**inner, IrType::U8 | IrType::Void)
                                        }
                                        _ => false,
                                    };
                                    let effective_ty = match resolved_value_ty.as_ref() {
                                        Some(rty @ (IrType::F64 | IrType::F32))
                                            if result_is_opaque =>
                                        {
                                            rty.clone()
                                        }
                                        _ => result_type.clone(),
                                    };
                                    let final_result = match &effective_ty {
                                        IrType::I32 => self.builder.build_cast(
                                            call_result,
                                            IrType::U64,
                                            IrType::I32,
                                        ),
                                        IrType::I64 => self.builder.build_cast(
                                            call_result,
                                            IrType::U64,
                                            IrType::I64,
                                        ),
                                        IrType::F64 => {
                                            self.builder.build_bitcast(call_result, IrType::F64)
                                        }
                                        IrType::F32 => {
                                            if let Some(f64_reg) =
                                                self.builder.build_bitcast(call_result, IrType::F64)
                                            {
                                                self.builder.build_cast(
                                                    f64_reg,
                                                    IrType::F64,
                                                    IrType::F32,
                                                )
                                            } else {
                                                None
                                            }
                                        }
                                        IrType::Bool => self.builder.build_cast(
                                            call_result,
                                            IrType::U64,
                                            IrType::Bool,
                                        ),
                                        IrType::Ptr(ref inner)
                                            if matches!(inner.as_ref(), IrType::String) =>
                                        {
                                            // Concrete pointer type (e.g., Ptr(String))
                                            self.builder.build_cast(
                                                call_result,
                                                IrType::U64,
                                                result_type.clone(),
                                            )
                                        }
                                        IrType::Ptr(_)
                                            if matches!(
                                                resolved_value_ty.as_ref(),
                                                Some(IrType::Ptr(_))
                                            ) =>
                                        {
                                            // Receiver parameterised with a concrete
                                            // pointer-typed T (extern class, user class,
                                            // array). Bit-reinterpret the runtime u64 as
                                            // the resolved pointer type.
                                            self.builder.build_bitcast(
                                                call_result,
                                                resolved_value_ty.unwrap(),
                                            )
                                        }
                                        _ => {
                                            // Unresolved T or Dynamic, or T resolved to a
                                            // primitive that arrived here boxed (result_type
                                            // = Ptr<U8> from `Null<Int>`): keep as I64 so
                                            // the downstream unbox path can extract the
                                            // primitive value. Bitcasting U64 → I32 here
                                            // would skip that boxing and produce values
                                            // that don't match the consumer's expected
                                            // register type.
                                            self.builder.build_cast(
                                                call_result,
                                                IrType::U64,
                                                IrType::I64,
                                            )
                                        }
                                    };
                                    return final_result;
                                }

                                // Auto-unbox if runtime returns Ptr(U8) but HIR expects primitive
                                let unboxed = self.maybe_unbox_for_extern_return(
                                    call_result,
                                    &actual_return_type,
                                    &result_type,
                                );
                                return unboxed;
                            } // end else (no ptr_conversion needed)
                        }
                    }
                    // SPECIAL CASE: Handle global trace() function
                    // Route to type-specific trace functions based on argument type
                    if symbol_name == "trace" && args.len() == 1 {
                        let arg = &args[0];

                        // Route trace(Type.typeof(x)) to enum tracing directly.
                        // This preserves parity even when the call-site type was widened.
                        if let Some(typeof_arg) = self.trace_typeof_inner_arg(arg) {
                            let value_reg = self.lower_type_typeof_call(
                                std::slice::from_ref(typeof_arg),
                                IrType::I64,
                            )?;
                            let trace_typeof_id = self.get_or_register_extern_function(
                                "haxe_trace_value_type",
                                vec![IrType::I64],
                                IrType::Void,
                            );
                            return self.builder.build_call_direct(
                                trace_typeof_id,
                                vec![value_reg],
                                IrType::Void,
                            );
                        }

                        // Handle ValueType values that were previously produced/stored.
                        if self.expr_is_value_type_expr(arg) {
                            let arg_reg = self.lower_expression(arg)?;
                            let trace_typeof_id = self.get_or_register_extern_function(
                                "haxe_trace_value_type",
                                vec![IrType::I64],
                                IrType::Void,
                            );
                            return self.builder.build_call_direct(
                                trace_typeof_id,
                                vec![arg_reg],
                                IrType::Void,
                            );
                        }

                        // Check if arg is a class or enum type
                        // For classes: try to call toString() method
                        // For enums: for now, fall through to traceAny (enum toString not yet implemented)
                        let type_table = self.type_table;
                        let type_kind = type_table.get(arg.ty).map(|ti| ti.kind.clone());

                        debug!(
                            "[TRACE ARG TYPE] arg.ty={:?}, type_kind={:?}",
                            arg.ty, type_kind
                        );

                        let class_info =
                            if let Some(crate::tast::core::TypeKind::Class { symbol_id, .. }) =
                                &type_kind
                            {
                                // Skip extern abstracts (CString, Usize, Ptr, etc.)
                                // — they appear as Class in the type table but don't have toString()
                                // Get class name for stdlib lookup
                                let class_name_str = self
                                    .symbol_table
                                    .get_symbol(*symbol_id)
                                    .and_then(|s| self.string_interner.get(s.name))
                                    .unwrap_or("");

                                let is_extern = self
                                    .symbol_table
                                    .get_symbol(*symbol_id)
                                    .map(|s| {
                                        s.flags.contains(crate::tast::symbols::SymbolFlags::EXTERN)
                                    })
                                    .unwrap_or(false);

                                // Skip extern classes UNLESS they have a toString in stdlib_mapping
                                // (e.g., StringMap, IntMap, Date have stdlib toString methods)
                                let has_stdlib_tostring = self
                                    .stdlib_mapping
                                    .find_by_name(class_name_str, "toString")
                                    .is_some();

                                if is_extern && !has_stdlib_tostring {
                                    None
                                } else {
                                    Some(class_name_str.to_string())
                                }
                            } else {
                                None
                            };

                        // Check if the trace argument is an enum variant expression (e.g., Color.Red)
                        // If so, we can print the variant name directly
                        if let HirExprKind::Field { object, field } = &arg.kind {
                            if let HirExprKind::Variable {
                                symbol: enum_symbol,
                                ..
                            } = &object.kind
                            {
                                if let Some(enum_sym) = self.symbol_table.get_symbol(*enum_symbol) {
                                    use crate::tast::SymbolKind;
                                    if enum_sym.kind == SymbolKind::Enum {
                                        // Get the variant name
                                        let field_sym = self.symbol_table.get_symbol(*field);
                                        if let Some(variant_name) =
                                            field_sym.and_then(|s| self.string_interner.get(s.name))
                                        {
                                            // Create a string constant with the variant name
                                            // IrValue::String will be converted by Cranelift to call haxe_string_literal
                                            // which returns a *mut HaxeString pointer
                                            let variant_name_str = variant_name.to_string();
                                            let string_ptr = self
                                                .builder
                                                .build_const(IrValue::String(variant_name_str))?;

                                            // Get or create the string trace function
                                            let string_ptr_ty =
                                                IrType::Ptr(Box::new(IrType::String));
                                            let string_trace_id = self
                                                .get_or_register_extern_function(
                                                    "haxe_trace_string_struct",
                                                    vec![string_ptr_ty],
                                                    IrType::Void,
                                                );

                                            // Trace the string
                                            return self.builder.build_call_direct(
                                                string_trace_id,
                                                vec![string_ptr],
                                                IrType::Void,
                                            );
                                        }
                                    }
                                }
                            }
                        }

                        // Check if it's an enum variable - print discriminant for now
                        // Full variant name lookup for variables would require runtime RTTI
                        // Direct enum variant expressions (Color.Red) are handled above

                        // If this is a class type, try to call toString() on it
                        if class_info.is_some() {
                            let obj_reg = self.lower_expression(arg)?;
                            if let Some(string_reg) = self.try_call_tostring(obj_reg, arg.ty)? {
                                let string_ptr_ty = IrType::Ptr(Box::new(IrType::String));
                                let string_trace_id = self.get_or_register_extern_function(
                                    "haxe_trace_string_struct",
                                    vec![string_ptr_ty],
                                    IrType::Void,
                                );
                                return self.builder.build_call_direct(
                                    string_trace_id,
                                    vec![string_reg],
                                    IrType::Void,
                                );
                            }
                        }

                        // Lower the argument first to get the actual MIR register
                        // Check if this is a field access
                        let is_field = matches!(&arg.kind, HirExprKind::Field { .. });
                        if is_field {
                            if let HirExprKind::Field { object, field } = &arg.kind {
                                let field_sym = self.symbol_table.get_symbol(*field);
                                let field_name = field_sym
                                    .and_then(|s| self.string_interner.get(s.name))
                                    .unwrap_or("<unknown>");
                                debug!("[TRACE] Argument is Field access: field={}", field_name);

                                // Check what the object is
                                if let HirExprKind::Variable { symbol, .. } = &object.kind {
                                    let var_sym = self.symbol_table.get_symbol(*symbol);
                                    let var_name = var_sym
                                        .and_then(|s| self.string_interner.get(s.name))
                                        .unwrap_or("<unknown>");
                                    debug!("[TRACE] Field object is Variable: {}", var_name);
                                }
                            }
                        }
                        let arg_reg = self.lower_expression(arg)?;
                        debug!(
                            "[TRACE] After lowering, arg_reg={}, checking type...",
                            arg_reg
                        );
                        if let Some(ty) = self.builder.get_register_type(arg_reg) {
                            debug!("[TRACE] arg_reg type from builder: {:?}", ty);
                        }

                        // Check if the HIR type is an enum
                        // Also check if the arg is a variable and look up its declared type
                        // (trace() takes Dynamic, so arg.ty might be Dynamic even if the variable is an enum)
                        let type_table = self.type_table;
                        let mut hir_type_kind = type_table.get(arg.ty).map(|ti| ti.kind.clone());

                        // If arg.ty is Dynamic but the argument is a variable, look up the variable's declared type
                        // This is needed because trace() accepts Dynamic, so the expression type might be Dynamic
                        // even when the underlying variable has a more specific type (like an enum)
                        if matches!(
                            &hir_type_kind,
                            Some(crate::tast::core::TypeKind::Dynamic) | None
                        ) {
                            if let HirExprKind::Variable { symbol, .. } = &arg.kind {
                                if let Some(sym) = self.symbol_table.get_symbol(*symbol) {
                                    let var_type_kind =
                                        type_table.get(sym.type_id).map(|ti| ti.kind.clone());
                                    if var_type_kind.is_some() {
                                        hir_type_kind = var_type_kind;
                                    }
                                }
                            }
                        }

                        // Handle enum variables - use RTTI-based trace with compile-time type_id
                        // Direct enum variant expressions (Color.Red) are handled above and print variant names
                        if let Some(crate::tast::core::TypeKind::Enum {
                            symbol_id,
                            ref type_args,
                        }) = hir_type_kind
                        {
                            if self.symbol_table.get_symbol(symbol_id).is_some() {
                                let enum_type_id = self.enum_runtime_id(symbol_id);

                                // Build type_id constant (u32)
                                let type_id_const = self
                                    .builder
                                    .build_const(IrValue::I32(enum_type_id as i32))?;

                                // Check if enum is boxed (has parameterized variants)
                                // Boxed enums store a pointer to heap-allocated struct
                                // Unboxed enums store just the discriminant as i64
                                if self.enum_is_boxed(symbol_id) {
                                    // Resolve concrete param types from type_args (type inference)
                                    // type_args maps type parameters to concrete types
                                    let concrete_type_args: Vec<u8> = {
                                        let type_table = self.type_table;
                                        type_args.iter().map(|&ta| {
                                            match type_table.get(ta).map(|ti| &ti.kind) {
                                                Some(crate::tast::core::TypeKind::Int) => 0u8,
                                                Some(crate::tast::core::TypeKind::Float) => 1u8,
                                                Some(crate::tast::core::TypeKind::Bool) => 2u8,
                                                Some(crate::tast::core::TypeKind::String) => 3u8,
                                                Some(crate::tast::core::TypeKind::TypeParameter { .. }) => 5u8,
                                                Some(crate::tast::core::TypeKind::Dynamic) => 5u8,
                                                _ => 4u8,
                                            }
                                        }).collect()
                                    };

                                    // If we have concrete type args, use the typed trace
                                    if !concrete_type_args.is_empty()
                                        && concrete_type_args.iter().any(|&t| t != 5)
                                    {
                                        let trace_typed_id = self.get_or_register_extern_function(
                                            "haxe_trace_enum_boxed_typed",
                                            vec![
                                                IrType::I32,
                                                IrType::Ptr(Box::new(IrType::I8)),
                                                IrType::Ptr(Box::new(IrType::I8)),
                                                IrType::I64,
                                            ],
                                            IrType::Void,
                                        );

                                        let ptr_reg = self.builder.build_bitcast(
                                            arg_reg,
                                            IrType::Ptr(Box::new(IrType::I8)),
                                        )?;

                                        // Build param types data via heap alloc + stores
                                        let alloc_size = self.builder.build_const(IrValue::I64(
                                            concrete_type_args.len() as i64,
                                        ))?;
                                        let alloc_func = self.get_or_register_extern_function(
                                            "malloc",
                                            vec![IrType::I64],
                                            IrType::Ptr(Box::new(IrType::I8)),
                                        );
                                        let param_types_data = self.builder.build_call_direct(
                                            alloc_func,
                                            vec![alloc_size],
                                            IrType::Ptr(Box::new(IrType::I8)),
                                        )?;
                                        for (i, &ptype) in concrete_type_args.iter().enumerate() {
                                            let offset =
                                                self.builder.build_const(IrValue::I64(i as i64))?;
                                            let elem_ptr = self.builder.build_gep(
                                                param_types_data,
                                                vec![offset],
                                                IrType::Ptr(Box::new(IrType::I8)),
                                            )?;
                                            let val = self
                                                .builder
                                                .build_const(IrValue::I8(ptype as i8))?;
                                            self.builder.build_store(elem_ptr, val);
                                        }
                                        let param_count = self.builder.build_const(
                                            IrValue::I64(concrete_type_args.len() as i64),
                                        )?;

                                        return self.builder.build_call_direct(
                                            trace_typed_id,
                                            vec![
                                                type_id_const,
                                                ptr_reg,
                                                param_types_data,
                                                param_count,
                                            ],
                                            IrType::Void,
                                        );
                                    }

                                    // Fallback: use untyped boxed trace
                                    let trace_enum_boxed_id = self.get_or_register_extern_function(
                                        "haxe_trace_enum_boxed",
                                        vec![IrType::I32, IrType::Ptr(Box::new(IrType::I8))],
                                        IrType::Void,
                                    );

                                    let ptr_reg = self.builder.build_bitcast(
                                        arg_reg,
                                        IrType::Ptr(Box::new(IrType::I8)),
                                    )?;

                                    return self.builder.build_call_direct(
                                        trace_enum_boxed_id,
                                        vec![type_id_const, ptr_reg],
                                        IrType::Void,
                                    );
                                } else {
                                    // Unboxed enum: arg_reg holds the discriminant (i64)
                                    // Call haxe_trace_enum(type_id: u32, discriminant: i64)
                                    let trace_enum_id = self.get_or_register_extern_function(
                                        "haxe_trace_enum",
                                        vec![IrType::I32, IrType::I64],
                                        IrType::Void,
                                    );

                                    return self.builder.build_call_direct(
                                        trace_enum_id,
                                        vec![type_id_const, arg_reg],
                                        IrType::Void,
                                    );
                                }
                            }
                        }

                        // Get the actual MIR type from the register (not the HIR type)
                        // This is important because HIR types may be vague (Ptr(Void)) but
                        // MIR registers have the actual type (String, etc.)
                        let actual_reg_type = self
                            .builder
                            .get_register_type(arg_reg)
                            .unwrap_or_else(|| self.convert_type(arg.ty));

                        let mut arg_type = actual_reg_type.clone();
                        // If the MIR type is Ptr(Void) but we have better type info from the symbol,
                        // use the symbol's type instead. This handles cases like trace(t) where t is
                        // a float from Sys.time() but the trace() signature says Dynamic.
                        // BUT: don't override Ptr(U8) — that means a boxed DynamicValue* (e.g., from
                        // Array.pop() returning Null<T>), which traceAny can properly unbox.
                        let is_boxed_dynamic = matches!(&arg_type, IrType::Ptr(inner) if matches!(inner.as_ref(), IrType::U8));
                        if matches!(arg_type, IrType::Ptr(_)) && !is_boxed_dynamic {
                            if let Some(ref type_kind) = hir_type_kind {
                                let better_type = match type_kind {
                                    crate::tast::core::TypeKind::Float => Some(IrType::F64),
                                    crate::tast::core::TypeKind::Int => Some(IrType::I64),
                                    crate::tast::core::TypeKind::Bool => Some(IrType::Bool),
                                    crate::tast::core::TypeKind::String => Some(IrType::String),
                                    _ => None,
                                };
                                if let Some(better) = better_type {
                                    arg_type = better;
                                }
                            }
                        }

                        // Check if this is an Array type from HIR type info
                        let is_array_type = matches!(
                            &hir_type_kind,
                            Some(crate::tast::core::TypeKind::Array { .. })
                        );

                        // For Array types, call haxe_trace_array directly
                        if is_array_type {
                            let trace_array_id = self.get_or_register_extern_function(
                                "haxe_trace_array",
                                vec![IrType::Ptr(Box::new(IrType::Void))],
                                IrType::Void,
                            );
                            return self.builder.build_call_direct(
                                trace_array_id,
                                vec![arg_reg],
                                IrType::Void,
                            );
                        }

                        // Handle TypeParameter types that are still type-erased (I64).
                        // Only activate when the register type didn't reveal the concrete type.
                        // Inside generic functions, emit a fixup for the monomorphize pass.
                        // Outside generic functions (shouldn't normally happen with proper type
                        // resolution), fall through to traceInt.
                        if matches!(arg_type, IrType::I64 | IrType::I32)
                            && matches!(
                                &hir_type_kind,
                                Some(crate::tast::core::TypeKind::TypeParameter { .. })
                            )
                        {
                            if let Some(crate::tast::core::TypeKind::TypeParameter {
                                symbol_id,
                                ..
                            }) = &hir_type_kind
                            {
                                let type_param_name = self
                                    .symbol_table
                                    .get_symbol(*symbol_id)
                                    .and_then(|sym| self.string_interner.get(sym.name))
                                    .map(|s| s.to_string());

                                if let Some(ref tp_name) = type_param_name {
                                    // Only emit a tag fixup if the current function actually
                                    // has this type parameter (i.e., we're inside a generic function).
                                    // If not, the fixup would never be resolved, so fall through
                                    // to normal trace dispatch instead.
                                    let current_func_has_param = self
                                        .builder
                                        .current_function()
                                        .map(|f| {
                                            f.signature
                                                .type_params
                                                .iter()
                                                .any(|tp| tp.name == *tp_name)
                                        })
                                        .unwrap_or(false);

                                    if current_func_has_param {
                                        let tag_reg = self.builder.build_const(IrValue::I32(0))?;
                                        if let Some(func) = self.builder.current_function_mut() {
                                            func.type_param_tag_fixups
                                                .push((tag_reg, tp_name.clone()));
                                        }

                                        let trace_typed_id = self.get_or_register_extern_function(
                                            "haxe_trace_typed",
                                            vec![IrType::I64, IrType::I32],
                                            IrType::Void,
                                        );

                                        return self.builder.build_call_direct(
                                            trace_typed_id,
                                            vec![arg_reg, tag_reg],
                                            IrType::Void,
                                        );
                                    }
                                    // If not in a generic function, fall through to normal dispatch.
                                }
                            }
                        }

                        // Special case: Optional<primitive> returned from MIR wrappers (e.g. array pop/shift)
                        // MIR wrappers cast DynamicValue* to IrType::Any (I64), but the value is still a boxed pointer.
                        // Detect via hir_type_kind and route to traceAny for proper unboxing.
                        // BUT: extern functions with returns_raw_value (e.g., StringMap.get) return the
                        // actual value bits as I64, NOT a boxed pointer. Nested Call expressions produce
                        // these raw values, so skip is_optional_boxed for Call args.
                        let is_optional_boxed = matches!(&arg_type, IrType::I64 | IrType::I32)
                            && matches!(
                                &hir_type_kind,
                                Some(crate::tast::core::TypeKind::Optional { .. })
                            )
                            && !matches!(&arg.kind, HirExprKind::Call { .. });

                        let trace_method = {
                            match &arg_type {
                                IrType::I32 | IrType::I64 | IrType::U64 if is_optional_boxed => {
                                    "traceAny"
                                }
                                IrType::I32 | IrType::I64 | IrType::U64 => "traceInt",
                                IrType::F32 | IrType::F64 => "traceFloat",
                                IrType::Bool => "traceBool",
                                IrType::String => "traceString", // String is ptr+len struct
                                // Also handle Ptr(String) - returned by String methods like toUpperCase()
                                IrType::Ptr(inner) if matches!(inner.as_ref(), IrType::String) => {
                                    "traceString"
                                }
                                // Ptr(U8) when HIR type is String — from MIR wrappers returning raw string pointers
                                IrType::Ptr(inner) if matches!(inner.as_ref(), IrType::U8) => {
                                    let is_hir_string = matches!(
                                        &hir_type_kind,
                                        Some(crate::tast::core::TypeKind::String)
                                    );
                                    if is_hir_string {
                                        "traceString"
                                    } else {
                                        "traceAny"
                                    }
                                }
                                IrType::TypeVar(_) => "traceTypedGeneric", // tag-based dispatch
                                _ => "traceAny", // Fallback for Dynamic or unknown types
                            }
                        };

                        // Debug: Print trace method selection
                        debug!(
                            "[DEBUG trace] arg_reg={}, arg_type={:?}, trace_method={}",
                            arg_reg, arg_type, trace_method
                        );

                        // Build the qualified name for the trace function
                        let trace_func_name = format!("rayzor.Trace.{}", trace_method);

                        // Look up the runtime function name
                        // For now, manually map to the runtime function
                        let runtime_func = match trace_method {
                            "traceInt" => "haxe_trace_int",
                            "traceFloat" => "haxe_trace_float",
                            "traceBool" => "haxe_trace_bool",
                            "traceString" => "haxe_trace_string",
                            "traceAny" => "haxe_trace_any",
                            _ => "haxe_trace_any",
                        };

                        // Special handling for String: use haxe_trace_string_struct that takes a pointer
                        if trace_method == "traceString" {
                            // String is represented as a pointer to HaxeString struct
                            let param_types = vec![IrType::Ptr(Box::new(IrType::String))];
                            let string_trace_id = self.get_or_register_extern_function(
                                "haxe_trace_string_struct",
                                param_types,
                                IrType::Void,
                            );
                            return self.builder.build_call_direct(
                                string_trace_id,
                                vec![arg_reg],
                                IrType::Void,
                            );
                        }

                        // TypeVar trace: use haxe_trace_typed with tag fixup.
                        // Bitcast value to I64 (TypeVar is pointer-sized) to avoid
                        // Cranelift type mismatch when inlining resolves to F64.
                        if trace_method == "traceTypedGeneric" {
                            let tag_reg = self.builder.build_const(IrValue::I32(0))?;
                            if let IrType::TypeVar(ref name) = arg_type {
                                if let Some(func) = self.builder.current_function_mut() {
                                    func.type_param_tag_fixups.push((tag_reg, name.clone()));
                                }
                            }
                            let val_as_i64 = self
                                .builder
                                .build_bitcast(arg_reg, IrType::I64)
                                .unwrap_or(arg_reg);
                            let trace_typed_id = self.get_or_register_extern_function(
                                "haxe_trace_typed",
                                vec![IrType::I64, IrType::I32],
                                IrType::Void,
                            );
                            return self.builder.build_call_direct(
                                trace_typed_id,
                                vec![val_as_i64, tag_reg],
                                IrType::Void,
                            );
                        }

                        // Get or register the extern runtime function
                        // Note: Runtime trace functions expect specific types:
                        // - haxe_trace_int expects i64
                        // - haxe_trace_float expects f64
                        // We need to cast arguments to match
                        // Note: We don't need to cast arguments here - the Cranelift backend
                        // handles signature-aware type conversion automatically (see cranelift_backend.rs:1487-1491)
                        // It will insert sextend for i32->i64, fcvt for f32->f64, etc.
                        let param_types = match trace_method {
                            "traceInt" => vec![IrType::I64],
                            "traceFloat" => vec![IrType::F64],
                            "traceBool" => vec![IrType::Bool],
                            _ => vec![arg_type.clone()],
                        };

                        // If Optional boxed value routed to traceAny, cast I64 back to pointer
                        let final_arg_reg = if is_optional_boxed && trace_method == "traceAny" {
                            let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                            self.builder.build_cast(arg_reg, IrType::I64, ptr_u8)?
                        } else {
                            arg_reg
                        };

                        let final_param_types = if is_optional_boxed && trace_method == "traceAny" {
                            vec![IrType::Ptr(Box::new(IrType::U8))]
                        } else {
                            param_types
                        };

                        let runtime_func_id = self.get_or_register_extern_function(
                            runtime_func,
                            final_param_types,
                            IrType::Void,
                        );

                        // Generate the call
                        return self.builder.build_call_direct(
                            runtime_func_id,
                            vec![final_arg_reg],
                            IrType::Void,
                        );
                    }

                    // SPECIAL CASE: Handle Std.string() function
                    // Route to type-specific string conversion functions based on argument type
                    // Note: Std.string() comes as a static method call with 2 args (Std class + actual arg)
                    if symbol_name == "string"
                        && (args.len() == 1 || (args.len() == 2 && *is_method))
                    {
                        debug!(
                            "[STD.STRING CHECK] Found 'string' call, is_method={}, args.len()={}",
                            is_method,
                            args.len()
                        );

                        // For static method calls, the actual argument is the second one (skip Std class)
                        let arg = if *is_method && args.len() == 2 {
                            &args[1]
                        } else {
                            &args[0]
                        };
                        let arg_is_value_type = self.expr_is_value_type_expr(arg);

                        // ValueType pretty-print parity path
                        if arg_is_value_type {
                            let arg_reg = self.lower_expression(arg)?;
                            return self.convert_value_type_to_string(arg_reg);
                        }

                        let arg_type = self.convert_type(arg.ty);

                        // Check HIR type for Array (TypeKind::Array maps to Ptr in MIR)
                        let hir_type_kind = {
                            let tt = self.type_table;
                            tt.get(arg.ty).map(|ti| ti.kind.clone())
                        };
                        let is_array =
                            matches!(hir_type_kind.as_ref(), Some(TypeKind::Array { .. }));

                        if is_array {
                            let arg_reg = self.lower_expression(arg)?;
                            let conv_fn = self.get_or_register_extern_function(
                                "haxe_array_to_string",
                                vec![IrType::Ptr(Box::new(IrType::Void))],
                                IrType::Ptr(Box::new(IrType::String)),
                            );
                            return self.builder.build_call_direct(
                                conv_fn,
                                vec![arg_reg],
                                IrType::Ptr(Box::new(IrType::String)),
                            );
                        }

                        // Determine which MIR wrapper function to call based on type
                        // These wrappers call the extern runtime functions
                        let mir_wrapper = match arg_type {
                            IrType::I32 | IrType::I64 => "int_to_string",
                            IrType::F32 | IrType::F64 => "float_to_string",
                            IrType::Bool => "bool_to_string",
                            IrType::String => "string_to_string",
                            _ => "int_to_string",
                        };

                        debug!(
                            "[STD.STRING] Routing Std.string() call to {} for type {:?}",
                            mir_wrapper, arg_type
                        );

                        // Lower the argument
                        let arg_reg = self.lower_expression(arg)?;

                        // Get or register the MIR wrapper function
                        // These return String (a struct with ptr + len)
                        let param_types = vec![arg_type.clone()];
                        let return_type = IrType::String; // String is represented as ptr+len
                        let mir_wrapper_id = self.get_or_register_extern_function(
                            mir_wrapper,
                            param_types,
                            return_type.clone(),
                        );

                        // Generate the call to MIR wrapper
                        return self.builder.build_call_direct(
                            mir_wrapper_id,
                            vec![arg_reg],
                            return_type,
                        );
                    }

                    // For instance method calls, check if this is a stdlib method or Dynamic method
                    // Note: Static methods like Thread.spawn() can also come through here with is_method=true
                    if *is_method && !args.is_empty() {
                        // The first arg is the receiver for instance method calls
                        // Resolve TypeAlias to get the actual receiver type.
                        //
                        // Cross-context override: when the receiver
                        // expression is a Variable whose binding was
                        // populated by an iface call whose return type
                        // we re-resolved at MIR time (Dynamic →
                        // concrete), use that override instead of the
                        // poisoned `args[0].ty`. Without this, the
                        // dispatch falls into the Dynamic-receiver path
                        // and the downstream MIR-wrapper boxing
                        // produces a malformed call (SIGSEGVs on
                        // e.g. `Array.push`).
                        let receiver_type = self
                            .effective_receiver_type(&args[0])
                            .map(|tid| self.resolve_through_aliases(tid))
                            .unwrap_or_else(|| self.resolve_through_aliases(args[0].ty));

                        {
                            let type_table = self.type_table;
                            if let Some(type_info) = type_table.get(receiver_type) {
                                debug!(
                                    "[METHOD CALL] receiver_type={:?}, kind={:?}",
                                    receiver_type, type_info.kind
                                );
                            } else {
                                // Print method name for calls with invalid receiver type
                                let method_name = self
                                    .symbol_table
                                    .get_symbol(*symbol)
                                    .map(|s| self.string_interner.get(s.name));
                                debug!(
                                    "[METHOD CALL] receiver_type={:?} NOT IN TYPE TABLE, method={:?}",
                                    receiver_type, method_name
                                );
                            }
                        }

                        // SPECIAL CASE: Handle Dynamic and TypeParameter method calls
                        // When receiver is Dynamic or TypeParameter (unresolved generic), resolve method by name
                        // TypeParameter arises from chained calls on generic types like Arc<T>.get().lock()
                        // where the return type of get() is TypeParameter T
                        {
                            let type_table = self.type_table;
                            if let Some(type_info) = type_table.get(receiver_type) {
                                if matches!(
                                    type_info.kind,
                                    TypeKind::Dynamic
                                        | TypeKind::TypeParameter { .. }
                                        | TypeKind::Placeholder { .. }
                                        | TypeKind::Unknown
                                ) {
                                    // First, check if this might be a stdlib method call
                                    // by checking if the receiver expression comes from a stdlib function
                                    // (i.e., its result type would be Ptr(Void) for MIR wrappers)
                                    let method_name_str = self
                                        .symbol_table
                                        .get_symbol(*symbol)
                                        .and_then(|s| self.string_interner.get(s.name));

                                    // Check if any stdlib class has this method - use the mapping dynamically
                                    // instead of hardcoding method names. This handles cases like:
                                    // - MutexGuard.get() vs Arc.get() - both have "get" but are different
                                    // - Mutex.lock() returning Dynamic typed as MutexGuard
                                    // For Dynamic receivers, check user-defined methods FIRST.
                                    // Stdlib has common names like "sum", "get", "set" that
                                    // collide with user methods on Dynamic-typed objects.
                                    let receiver_is_dynamic = {
                                        let type_table = self.type_table;
                                        type_table
                                            .get(receiver_type)
                                            .map(|t| matches!(t.kind, TypeKind::Dynamic))
                                            .unwrap_or(false)
                                    };
                                    let user_func_for_dynamic = if receiver_is_dynamic {
                                        let method_name_is =
                                            self.symbol_table.get_symbol(*symbol).map(|s| s.name);
                                        if let Some(name) = method_name_is {
                                            let mut found = None;
                                            for (sym, &fid) in &self.function_map {
                                                if let Some(si) = self.symbol_table.get_symbol(*sym)
                                                {
                                                    if si.name == name {
                                                        found = Some(fid);
                                                        break;
                                                    }
                                                }
                                            }
                                            found
                                        } else {
                                            None
                                        }
                                    } else {
                                        None
                                    };

                                    if let Some(func_id) = user_func_for_dynamic {
                                        // User-defined method found for Dynamic receiver — use it with unboxing
                                        let receiver_reg = self.lower_expression(&args[0])?;

                                        // Dynamic receivers are always boxed (from haxe_box_reference_ptr),
                                        // even if the MIR register type shows Ptr(Void) due to cast.
                                        // Always unbox unless receiver has a class hint (stdlib container).
                                        let has_class_hint =
                                            self.register_class_hints.contains_key(&receiver_reg);
                                        let should_unbox = !has_class_hint;
                                        let actual_receiver = if should_unbox {
                                            let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                                            let unbox_func_id = self
                                                .get_or_register_extern_function(
                                                    "haxe_unbox_reference_ptr",
                                                    vec![ptr_u8.clone()],
                                                    ptr_u8.clone(),
                                                );
                                            self.builder.build_call_direct(
                                                unbox_func_id,
                                                vec![receiver_reg],
                                                ptr_u8,
                                            )?
                                        } else {
                                            receiver_reg
                                        };

                                        // Lower remaining args
                                        let arg_regs: Vec<_> = std::iter::once(actual_receiver)
                                            .chain(
                                                args[1..]
                                                    .iter()
                                                    .filter_map(|a| self.lower_expression(a)),
                                            )
                                            .collect();

                                        let actual_return_type = if let Some(func) =
                                            self.builder.module.functions.get(&func_id)
                                        {
                                            func.signature.return_type.clone()
                                        } else {
                                            result_type.clone()
                                        };

                                        return self.builder.build_call_direct(
                                            func_id,
                                            arg_regs,
                                            actual_return_type,
                                        );
                                    }

                                    let is_stdlib_method = method_name_str
                                        .map(|m| self.stdlib_mapping.any_class_has_method(m))
                                        .unwrap_or(false);
                                    if is_stdlib_method {
                                        let method_name = method_name_str.unwrap();
                                        // Calculate actual param count (exclude receiver for instance methods)
                                        let actual_param_count = args.len().saturating_sub(1);
                                        debug!(
                                            "[DYNAMIC METHOD] Found stdlib method '{}' in mapping, param_count={}",
                                            method_name, actual_param_count
                                        );

                                        // Query the stdlib mapping for all classes that have this method.
                                        // Results are sorted by priority (MutexGuard before Arc, etc.)
                                        let matching_classes = self
                                            .stdlib_mapping
                                            .find_classes_with_method(method_name);
                                        debug!(
                                            "[DYNAMIC STDLIB] {} classes have method '{}' (before param count filter)",
                                            matching_classes.len(),
                                            method_name
                                        );

                                        // Filter by param count to disambiguate overloaded methods
                                        // e.g., Array.join(sep) with 1 param vs Thread.join() with 0 params
                                        let mut filtered_classes: Vec<_> = matching_classes
                                            .into_iter()
                                            .filter(|(_, _, call)| {
                                                call.param_count == actual_param_count
                                            })
                                            .collect();
                                        debug!(
                                            "[DYNAMIC STDLIB] {} classes after param count filter",
                                            filtered_classes.len()
                                        );

                                        // Disambiguate using class hints when multiple classes match
                                        // (e.g., Arc.get vs MutexGuard.get — both have 0 params)
                                        if filtered_classes.len() > 1 {
                                            // Check if the receiver variable has a class hint
                                            let receiver_hint = if let HirExprKind::Variable {
                                                symbol: recv_sym,
                                                ..
                                            } = &args[0].kind
                                            {
                                                self.monomorphized_var_types.get(&recv_sym).cloned()
                                            } else {
                                                None
                                            };

                                            if let Some(hint) = &receiver_hint {
                                                let hinted: Vec<_> = filtered_classes
                                                    .iter()
                                                    .filter(|(class, _, _)| {
                                                        *class == hint.as_str()
                                                            || class
                                                                .ends_with(&format!("_{}", hint))
                                                            || hint
                                                                .ends_with(&format!("_{}", class))
                                                    })
                                                    .copied()
                                                    .collect();
                                                if !hinted.is_empty() {
                                                    debug!(
                                                        "[DYNAMIC STDLIB] Disambiguated by class hint '{}': {} -> {} matches",
                                                        hint,
                                                        filtered_classes.len(),
                                                        hinted.len()
                                                    );
                                                    filtered_classes = hinted;
                                                }
                                            }
                                        }

                                        // No priority guessing: if the candidates still name more
                                        // than one distinct runtime function (same-target aliases
                                        // like rayzor_Bytes.get / haxe_io_Bytes.get are NOT
                                        // ambiguous), the receiver's type is unresolved and any
                                        // pick would silently call an unrelated class's method.
                                        {
                                            let mut distinct: Vec<&str> = filtered_classes
                                                .iter()
                                                .map(|(_, _, call)| call.runtime_name)
                                                .collect();
                                            distinct.sort_unstable();
                                            distinct.dedup();
                                            if distinct.len() > 1 {
                                                let candidates = filtered_classes
                                                    .iter()
                                                    .map(|(class, _, _)| *class)
                                                    .collect::<Vec<_>>()
                                                    .join(", ");
                                                self.add_error(
                                                    &format!(
                                                        "E0801: ambiguous dynamic method dispatch: `{}` with {} argument(s) matches multiple stdlib classes ({}) and the receiver's type is unresolved. Annotate the receiver's type so the call resolves to one class",
                                                        method_name, actual_param_count, candidates
                                                    ),
                                                    expr.source_location,
                                                );
                                                return None;
                                            }
                                        }

                                        // A unique runtime target remains — dispatch to it.
                                        if let Some(&(class_name, _sig, runtime_call)) =
                                            filtered_classes.first()
                                        {
                                            debug!(
                                                "[DYNAMIC STDLIB] Using {}.{} -> {}",
                                                class_name, method_name, runtime_call.runtime_name
                                            );
                                            let runtime_func = runtime_call.runtime_name;

                                            // Check if this is a MIR wrapper class
                                            if self.stdlib_mapping.is_mir_wrapper_class(class_name)
                                            {
                                                // Use runtime_name directly as the MIR wrapper function name
                                                // (e.g., "Arc_init" not "rayzor_concurrent_Arc_init")
                                                let mir_func_name = runtime_func.to_string();
                                                debug!(
                                                    "[DYNAMIC STDLIB MIR] Using MIR wrapper: {}",
                                                    mir_func_name
                                                );

                                                // Lower all arguments with auto-boxing
                                                // CRITICAL: If the receiver (args[0]) can't be lowered, skip this handler
                                                // to prevent generating 0-arg calls for instance methods that expect self.
                                                let mir_wrapper_sig = self
                                                    .get_stdlib_mir_wrapper_signature(
                                                        &mir_func_name,
                                                    );
                                                let mut arg_regs = Vec::new();
                                                let mut param_types = Vec::new();
                                                let mut receiver_failed = false;
                                                for (i, arg) in args.iter().enumerate() {
                                                    if let Some(reg) = self.lower_expression(arg) {
                                                        let actual_ty = self.convert_type(arg.ty);
                                                        let expected_ty = mir_wrapper_sig
                                                            .as_ref()
                                                            .and_then(|(params, _)| {
                                                                params.get(i).cloned()
                                                            })
                                                            .unwrap_or_else(|| actual_ty.clone());

                                                        // TypeParameter/Dynamic/Placeholder args erased to I64
                                                        // should be CAST to Ptr(U8), not BOXED — but ONLY
                                                        // when the actual register value is a pointer (I64).
                                                        // For concrete primitives (I32, F64, Bool from Channel<Int>),
                                                        // the value must be BOXED, not cast.
                                                        let is_erased_type_param = {
                                                            let type_table = self.type_table;
                                                            type_table
                                                                .get(arg.ty)
                                                                .map(|ti| {
                                                                    matches!(
                                                                ti.kind,
                                                                TypeKind::TypeParameter { .. }
                                                                | TypeKind::Dynamic
                                                                | TypeKind::Placeholder { .. }
                                                            )
                                                                })
                                                                .unwrap_or(false)
                                                        };
                                                        // Check if register holds a concrete primitive
                                                        // (e.g., Channel<Int>.send(42) → reg is I32, not a pointer)
                                                        let reg_ir_type =
                                                            self.builder.get_register_type(reg);
                                                        let is_concrete_primitive = matches!(
                                                            reg_ir_type,
                                                            Some(IrType::I32)
                                                                | Some(IrType::F32)
                                                                | Some(IrType::F64)
                                                                | Some(IrType::Bool)
                                                        );
                                                        let final_reg = if (mir_func_name
                                                            == "Channel_send"
                                                            || mir_func_name == "Channel_trySend")
                                                            && i >= 1
                                                        {
                                                            // Uniformly box Channel payloads (refs too)
                                                            // so the erased receive can tag-dispatch.
                                                            // i==0 is the channel handle — never box it.
                                                            self.box_channel_payload(
                                                                reg,
                                                                arg.ty,
                                                                &actual_ty,
                                                                &expected_ty,
                                                            )?
                                                        } else if is_erased_type_param
                                                            && matches!(actual_ty, IrType::I64)
                                                            && matches!(
                                                                &expected_ty,
                                                                IrType::Ptr(_)
                                                            )
                                                            && !is_concrete_primitive
                                                        {
                                                            // Cast I64 → Ptr(U8) — the I64 is actually a pointer
                                                            self.builder
                                                                .build_cast(
                                                                    reg,
                                                                    IrType::I64,
                                                                    expected_ty.clone(),
                                                                )
                                                                .unwrap_or(reg)
                                                        } else if is_concrete_primitive
                                                            && matches!(
                                                                &expected_ty,
                                                                IrType::Ptr(_)
                                                            )
                                                        {
                                                            // Box concrete primitive for generic param
                                                            let box_ty = reg_ir_type.unwrap();
                                                            self.maybe_box_for_extern_call(
                                                                reg,
                                                                &box_ty,
                                                                &expected_ty,
                                                            )?
                                                        } else {
                                                            self.maybe_box_for_extern_call(
                                                                reg,
                                                                &actual_ty,
                                                                &expected_ty,
                                                            )?
                                                        };
                                                        arg_regs.push(final_reg);
                                                        param_types.push(expected_ty);
                                                    } else if i == 0 {
                                                        // Receiver failed to lower — can't call instance method
                                                        receiver_failed = true;
                                                        break;
                                                    }
                                                }

                                                // If receiver failed to lower, skip this handler
                                                // and let the general fallback chain handle it
                                                if receiver_failed {
                                                    // Don't generate a broken call; fall through
                                                } else {
                                                    // Get MIR wrapper return type
                                                    let mir_return_type = mir_wrapper_sig
                                                        .as_ref()
                                                        .map(|(_, ret)| ret.clone())
                                                        .unwrap_or_else(|| result_type.clone());

                                                    // Register forward reference
                                                    let mir_func_id = self
                                                        .register_stdlib_mir_forward_ref(
                                                            &mir_func_name,
                                                            param_types,
                                                            mir_return_type.clone(),
                                                        );

                                                    let call_result =
                                                        self.builder.build_call_direct(
                                                            mir_func_id,
                                                            arg_regs,
                                                            mir_return_type.clone(),
                                                        )?;

                                                    // Auto-unbox: resolve generic T from receiver type args
                                                    // e.g., Channel<Int>.tryReceive() returns Ptr(U8) but should produce I32
                                                    let resolved_expected = {
                                                        let type_table = self.type_table;
                                                        // The receiver is args[0] - check its type for generic args
                                                        let from_receiver = if !args.is_empty() {
                                                            type_table.get(args[0].ty).and_then(|ti| {
                                                            match &ti.kind {
                                                                crate::tast::TypeKind::Class { type_args, .. }
                                                                | crate::tast::TypeKind::GenericInstance { type_args, .. } => {
                                                                    if !type_args.is_empty() {
                                                                        let t = self.convert_type(type_args[0]);
                                                                        if matches!(t, IrType::I32 | IrType::I64 | IrType::F32 | IrType::F64 | IrType::Bool) {
                                                                            Some(t)
                                                                        } else {
                                                                            None
                                                                        }
                                                                    } else { None }
                                                                }
                                                                _ => None,
                                                            }
                                                        })
                                                        } else {
                                                            None
                                                        };
                                                        // Also check if return type is Optional{primitive} (Null<T>)
                                                        // and resolve to the inner primitive for unboxing
                                                        let from_optional = type_table.get(expr.ty).and_then(|ti| {
                                                            if let crate::tast::TypeKind::Optional { inner_type } = &ti.kind {
                                                                let t = self.convert_type(*inner_type);
                                                                if matches!(t, IrType::I32 | IrType::I64 | IrType::F32 | IrType::F64 | IrType::Bool) {
                                                                    Some(t)
                                                                } else {
                                                                    None
                                                                }
                                                            } else {
                                                                None
                                                            }
                                                        });
                                                        from_receiver
                                                            .or(from_optional)
                                                            .unwrap_or_else(|| result_type.clone())
                                                    };
                                                    let final_result = if mir_func_name
                                                        == "Channel_receive"
                                                        || mir_func_name == "Channel_tryReceive"
                                                    {
                                                        self.unbox_channel_return(
                                                            call_result,
                                                            &resolved_expected,
                                                            mir_func_name == "Channel_tryReceive",
                                                        )
                                                    } else {
                                                        self.maybe_unbox_for_extern_return(
                                                            call_result,
                                                            &mir_return_type,
                                                            &resolved_expected,
                                                        )
                                                    };

                                                    // Store class hint for the result register to enable
                                                    // disambiguation of subsequent method calls on this value.
                                                    // E.g., Mutex.lock() returns MutexGuard, so the result
                                                    // should be tagged as MutexGuard for .get()/.unlock() dispatch.
                                                    if let Some(result_reg) = final_result {
                                                        let return_class =
                                                            Self::get_return_class_hint(
                                                                class_name,
                                                                method_name,
                                                            );
                                                        self.register_class_hints.insert(
                                                            result_reg,
                                                            return_class.to_string(),
                                                        );
                                                    }

                                                    return final_result;
                                                } // end else !receiver_failed
                                            } else {
                                                // Direct extern call
                                                let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));

                                                // Extract runtime_call data before borrowing self mutably
                                                let has_return = runtime_call.has_return;

                                                // Lower all arguments using a for loop (not a closure)
                                                // to avoid borrow conflict with stdlib_mapping
                                                let mut arg_regs = Vec::new();
                                                for arg in args {
                                                    if let Some(reg) = self.lower_expression(arg) {
                                                        arg_regs.push(reg);
                                                    }
                                                }

                                                // Build param types
                                                let param_types: Vec<_> = arg_regs
                                                    .iter()
                                                    .map(|_| ptr_u8.clone())
                                                    .collect();

                                                // Determine return type: Void if function doesn't return, otherwise ptr
                                                let return_type = if has_return {
                                                    ptr_u8.clone()
                                                } else {
                                                    IrType::Void
                                                };

                                                let extern_func_id = self
                                                    .get_or_register_extern_function(
                                                        runtime_func,
                                                        param_types,
                                                        return_type.clone(),
                                                    );

                                                return self.builder.build_call_direct(
                                                    extern_func_id,
                                                    arg_regs,
                                                    return_type,
                                                );
                                            }
                                        }
                                        // If no mapping found, fall through to regular dispatch
                                    } else {
                                        // Look up method by name in function_map (generic Dynamic dispatch)
                                        let method_name =
                                            self.symbol_table.get_symbol(*symbol).map(|s| s.name);
                                        if let Some(name) = method_name {
                                            let mut found_func = None;
                                            for (sym, &func_id) in &self.function_map {
                                                if let Some(sym_info) =
                                                    self.symbol_table.get_symbol(*sym)
                                                {
                                                    if sym_info.name == name {
                                                        found_func = Some(func_id);
                                                        break;
                                                    }
                                                }
                                            }

                                            if let Some(func_id) = found_func {
                                                // Lower the receiver
                                                let receiver_reg =
                                                    self.lower_expression(&args[0])?;

                                                // Check if the receiver was boxed by examining its MIR register type.
                                                // Boxing creates a Ptr(U8) value. If the receiver has a different
                                                // pointer type (like Ptr(Void) from a stdlib function return),
                                                // it wasn't boxed and shouldn't be unboxed.
                                                //
                                                // IMPORTANT: If the receiver has a class hint (set by stdlib MIR
                                                // wrapper dispatch), it's a raw class pointer from a method like
                                                // MutexGuard_get — NOT a boxed DynamicValue. Don't unbox it.
                                                let has_class_hint = self
                                                    .register_class_hints
                                                    .contains_key(&receiver_reg);
                                                let receiver_mir_type =
                                                    self.builder.get_register_type(receiver_reg);
                                                // Dynamic receivers are always boxed (from haxe_box_reference_ptr),
                                                // even if MIR register type shows Ptr(Void) due to cast.
                                                // Always unbox for Dynamic unless it has a class hint.
                                                let should_unbox = !has_class_hint;

                                                let actual_receiver = if should_unbox {
                                                    // Unbox the Dynamic to get the actual object pointer
                                                    let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                                                    let unbox_func_id = self
                                                        .get_or_register_extern_function(
                                                            "haxe_unbox_reference_ptr",
                                                            vec![ptr_u8.clone()],
                                                            ptr_u8.clone(),
                                                        );
                                                    self.builder.build_call_direct(
                                                        unbox_func_id,
                                                        vec![receiver_reg],
                                                        ptr_u8,
                                                    )?
                                                } else {
                                                    debug!(
                                                        "[DYNAMIC METHOD] Skipping unbox - stdlib container method"
                                                    );
                                                    receiver_reg
                                                };

                                                // Lower the rest of arguments (skip receiver at index 0)
                                                let arg_regs: Vec<_> =
                                                    std::iter::once(actual_receiver)
                                                        .chain(args[1..].iter().filter_map(|a| {
                                                            self.lower_expression(a)
                                                        }))
                                                        .collect();

                                                // Get the function's actual return type
                                                let actual_return_type = if let Some(func) =
                                                    self.builder.module.functions.get(&func_id)
                                                {
                                                    func.signature.return_type.clone()
                                                } else {
                                                    result_type.clone()
                                                };

                                                return self.builder.build_call_direct(
                                                    func_id,
                                                    arg_regs,
                                                    actual_return_type,
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // NOTE: MutexGuard method calls are handled through the general stdlib mechanism:
                        // 1. Dynamic dispatch uses find_classes_with_method() with dynamic priority
                        // 2. MutexGuard is prioritized (return-only type with no constructor)
                        // 3. MutexGuard_get MIR wrapper is called via stdlib_mapping

                        // NOTE: String method calls are handled through the general stdlib mechanism:
                        // 1. get_stdlib_runtime_info() maps TypeKind::String to class name "String"
                        // 2. stdlib_mapping lookup finds the correct runtime function
                        // 3. The general path handles param types and return types

                        // PRIORITY CHECK: For extern generic classes like Vec<T>, the receiver type
                        // may be TypeId::MAX (invalid). In this case, try to use the tracked
                        // monomorphized class from variable assignment.
                        if receiver_type == TypeId::from_raw(u32::MAX) {
                            debug!(
                                "[MONO VAR CHECK] receiver_type is MAX, checking monomorphized_var_types ({} entries)",
                                self.monomorphized_var_types.len()
                            );

                            // Try to extract the SymbolId from the receiver expression
                            // The receiver (args[0]) should be a variable reference like HirExprKind::Variable
                            let receiver_symbol = match &args[0].kind {
                                HirExprKind::Variable { symbol, .. } => Some(*symbol),
                                HirExprKind::Field { field, .. } => Some(*field),
                                _ => None,
                            };
                            debug!(
                                "[MONO VAR CHECK] Receiver expression symbol: {:?}",
                                receiver_symbol
                            );

                            if let Some(var_symbol) = receiver_symbol {
                                // Check if this variable has a tracked monomorphized class
                                if let Some(mono_class) =
                                    self.monomorphized_var_types.get(&var_symbol).cloned()
                                {
                                    // Get the method name
                                    if let Some(method_sym) = self.symbol_table.get_symbol(*symbol)
                                    {
                                        if let Some(method_name) =
                                            self.string_interner.get(method_sym.name)
                                        {
                                            debug!(
                                                "[MONO VAR DISPATCH] Found tracked class '{}' for variable {:?}, method '{}'",
                                                mono_class, var_symbol, method_name
                                            );

                                            // Build the MIR wrapper function name: VecI32_push, VecF64_get, etc.
                                            let mir_func_name =
                                                format!("{}_{}", mono_class, method_name);

                                            // Get the signature from get_stdlib_mir_wrapper_signature
                                            if let Some((mir_param_types, mir_return_type)) = self
                                                .get_stdlib_mir_wrapper_signature(&mir_func_name)
                                            {
                                                debug!(
                                                    "[MONO VAR DISPATCH] Using MIR wrapper: {}",
                                                    mir_func_name
                                                );

                                                // Lower all arguments (including receiver)
                                                let mut arg_regs = Vec::new();
                                                for arg in args {
                                                    if let Some(reg) = self.lower_expression(arg) {
                                                        arg_regs.push(reg);
                                                    }
                                                }

                                                // Register forward reference
                                                let mir_func_id = self
                                                    .register_stdlib_mir_forward_ref(
                                                        &mir_func_name,
                                                        mir_param_types.clone(),
                                                        mir_return_type.clone(),
                                                    );

                                                debug!(
                                                    "[MONO VAR DISPATCH] Registered forward ref to {} with ID {:?}",
                                                    mir_func_name, mir_func_id
                                                );

                                                // Generate the call
                                                let result = self.builder.build_call_direct(
                                                    mir_func_id,
                                                    arg_regs,
                                                    mir_return_type,
                                                );
                                                debug!(
                                                    "[MONO VAR DISPATCH] Generated call, result: {:?}",
                                                    result
                                                );
                                                return result;
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // GUARD: Skip instance method handling if receiver is a Class type itself
                        // This can happen when static method calls come through with is_method=true
                        // e.g., Thread.spawn(closure) might be seen as Thread(receiver).spawn(closure)
                        let receiver_is_class_type = {
                            let type_table = self.type_table;
                            type_table.get(receiver_type)
                                .map(|ti| {
                                    // Check if the type is a class AND matches one of our MIR wrapper classes
                                    if let crate::tast::core::TypeKind::Class { symbol_id, .. } = &ti.kind {
                                        self.symbol_table.get_symbol(*symbol_id)
                                            .and_then(|s| self.string_interner.get(s.name))
                                            .map(|name| {
                                                // Use dynamic check via stdlib_mapping instead of hardcoded list
                                                let is_mir_wrapper = self.stdlib_mapping.is_mir_wrapper_class(name);
                                                if is_mir_wrapper {
                                                    debug!("[GUARD] Receiver type is {} class (MIR wrapper), skipping instance method path", name);
                                                }
                                                is_mir_wrapper
                                            })
                                            .unwrap_or(false)
                                    } else {
                                        false
                                    }
                                })
                                .unwrap_or(false)
                        };

                        // Try the receiver type path first (for true instance methods)
                        // Skip if receiver is a MIR wrapper class type (those are static methods)
                        {
                            let sym_name = self
                                .symbol_table
                                .get_symbol(*symbol)
                                .and_then(|s| self.string_interner.get(s.name))
                                .unwrap_or("?");
                            if matches!(
                                sym_name,
                                "balance"
                                    | "setLoop"
                                    | "compare"
                                    | "merge"
                                    | "minBinding"
                                    | "removeMinBinding"
                            ) {
                                debug!(
                                    "[DISPATCH_TRACE] '{}' receiver_is_class_type={}, receiver_type={:?}",
                                    sym_name, receiver_is_class_type, receiver_type
                                );
                            }
                        }
                        if !receiver_is_class_type {
                            // Calculate param_count for overload disambiguation: args[0] is receiver, rest are params
                            let method_param_count =
                                if args.len() > 1 { args.len() - 1 } else { 0 };
                            {
                                if let Some((class_name, method_name, runtime_call)) = self
                                    .get_stdlib_runtime_info(
                                        *symbol,
                                        receiver_type,
                                        Some(method_param_count),
                                        None,
                                    )
                                {
                                    let runtime_func = runtime_call.runtime_name;
                                    let ptr_conversion_mask =
                                        runtime_call.params_need_ptr_conversion;
                                    let raw_value_mask = runtime_call.raw_value_params;
                                    let returns_raw_value = runtime_call.returns_raw_value;
                                    let extend_i64_mask = runtime_call.extend_to_i64_params;
                                    let needs_out_param = runtime_call.needs_out_param;
                                    let has_return = runtime_call.has_return; // Copy for use in fallback closure

                                    // SPECIAL CASE: Instance methods that need out parameter (like Array.slice, String.split)
                                    // These have void return but write result to first out parameter
                                    // Generate inline wrapper: allocate + call runtime + return pointer
                                    if needs_out_param {
                                        debug!(
                                        "[OUT PARAM] Instance method {}.{} needs out param inline wrapper",
                                        class_name, method_name
                                    );

                                        // Lower all arguments (receiver + method args)
                                        let mut call_arg_regs = Vec::new();
                                        for arg in args {
                                            if let Some(reg) = self.lower_expression(arg) {
                                                call_arg_regs.push(reg);
                                            }
                                        }

                                        // Allocate space for the result object
                                        // For arrays/strings, allocate an opaque pointer-sized value
                                        let out_ptr_ty = IrType::Ptr(Box::new(IrType::Void));
                                        let out_ptr =
                                            self.builder.build_alloc(out_ptr_ty.clone(), None)?;

                                        // Register the extern runtime function
                                        // Signature: void runtime_func(out: *Ptr(Void), receiver: Ptr(Void), ...params)
                                        let mut extern_param_types = vec![out_ptr_ty.clone()]; // out parameter
                                        for arg in args {
                                            extern_param_types.push(self.convert_type(arg.ty));
                                        }

                                        let extern_func_id = self.get_or_register_extern_function(
                                            runtime_func,
                                            extern_param_types,
                                            IrType::Void,
                                        );

                                        // Call runtime function: runtime_func(out_ptr, receiver, ...args)
                                        let mut runtime_args = vec![out_ptr];
                                        runtime_args.extend(call_arg_regs);

                                        self.builder.build_call_direct(
                                            extern_func_id,
                                            runtime_args,
                                            IrType::Void,
                                        );

                                        // Load the result pointer from the out parameter
                                        let result_ptr =
                                            self.builder.build_load(out_ptr, out_ptr_ty)?;

                                        debug!(
                                        "[OUT PARAM] Generated inline wrapper for {}, result_ptr: {:?}",
                                        runtime_func, result_ptr
                                    );

                                        return Some(result_ptr);
                                    }

                                    // SPECIAL CASE: Check if this is a stdlib MIR wrapper function
                                    // MIR wrappers are functions that forward to extern runtime functions.
                                    // The wrappers handle calling convention differences and provide default arguments.
                                    // NOTE: We check runtime_call.is_mir_wrapper, not just is_mir_wrapper_class(),
                                    // because some methods on MIR wrapper classes (e.g., String.split) are
                                    // direct extern calls without wrappers.
                                    if runtime_call.is_mir_wrapper {
                                        // Use the runtime function name from the mapping to handle overloaded methods
                                        // For example, String.indexOf can map to String_indexOf (1-arg) or String_indexOf_2 (2-arg)
                                        let mir_func_name = runtime_func.to_string();
                                        debug!(
                                        "[STDLIB MIR] Detected stdlib MIR wrapper function (instance): {}",
                                        mir_func_name
                                    );

                                        // Lower all arguments and collect their types
                                        // Auto-box primitive args when MIR wrapper expects Ptr(U8)
                                        // (e.g., Channel<Int>.send(42) needs to box the Int)
                                        let mir_wrapper_params = self
                                            .get_stdlib_mir_wrapper_signature(&mir_func_name)
                                            .map(|(params, _)| params);
                                        let mut arg_regs = Vec::new();
                                        let mut param_types = Vec::new();
                                        for (i, arg) in args.iter().enumerate() {
                                            if let Some(reg) = self.lower_expression(arg) {
                                                let actual_ty = self.convert_type(arg.ty);
                                                // Check if MIR wrapper expects a different type (e.g., Ptr(U8) for boxed value)
                                                let expected_ty = mir_wrapper_params
                                                    .as_ref()
                                                    .and_then(|params| params.get(i).cloned())
                                                    .unwrap_or_else(|| actual_ty.clone());
                                                // Channel payloads are uniformly boxed (refs too) so
                                                // the erased receive arm can tag-dispatch. i==0 is the
                                                // channel handle — never box it.
                                                let final_reg = if (mir_func_name == "Channel_send"
                                                    || mir_func_name == "Channel_trySend")
                                                    && i >= 1
                                                {
                                                    self.box_channel_payload(
                                                        reg,
                                                        arg.ty,
                                                        &actual_ty,
                                                        &expected_ty,
                                                    )?
                                                } else {
                                                    self.maybe_box_for_extern_call(
                                                        reg,
                                                        &actual_ty,
                                                        &expected_ty,
                                                    )?
                                                };
                                                arg_regs.push(final_reg);
                                                param_types.push(expected_ty);
                                            }
                                        }

                                        // SPECIAL: For generic methods that return T (like Thread<T>.join() -> T,
                                        // Channel<T>.tryReceive() -> Null<T>), we need to resolve the type parameter
                                        // from the receiver's generic arguments.
                                        // Also resolve when result_type is Ptr(Void) which comes from Dynamic/unresolved generics.
                                        let needs_generic_resolve = result_type == IrType::Any
                                            || matches!(&result_type, IrType::Ptr(inner) if matches!(inner.as_ref(), IrType::Void))
                                            || result_type == IrType::I64;
                                        let resolved_result_type = if needs_generic_resolve {
                                            // Check if the receiver is a generic class with type parameters
                                            let type_table = self.type_table;
                                            if let Some(receiver_info) =
                                                type_table.get(receiver_type)
                                            {
                                                if let crate::tast::TypeKind::Class {
                                                    type_args,
                                                    ..
                                                } = &receiver_info.kind
                                                {
                                                    // For Thread<T>.join(), type_args[0] is T
                                                    if !type_args.is_empty() {
                                                        let concrete_type =
                                                            self.convert_type(type_args[0]);
                                                        debug!(
                                                        "[GENERIC RESOLVE] Resolved return type from {:?} to {:?}",
                                                        result_type, concrete_type
                                                    );
                                                        concrete_type
                                                    } else {
                                                        result_type.clone()
                                                    }
                                                } else {
                                                    result_type.clone()
                                                }
                                            } else {
                                                result_type.clone()
                                            }
                                        } else {
                                            result_type.clone()
                                        };

                                        // Register forward reference - will be provided by merged stdlib module
                                        let mir_func_id = self.register_stdlib_mir_forward_ref(
                                            &mir_func_name,
                                            param_types,
                                            resolved_result_type.clone(),
                                        );

                                        // IMPORTANT: For Void-returning functions, use the function's ACTUAL return type.
                                        // For non-void functions, trust resolved_result_type (which handles generics correctly).
                                        // This fixes the bug where void functions like Channel.send incorrectly get dest registers.
                                        let final_return_type = if let Some(func) =
                                            self.builder.module.functions.get(&mir_func_id)
                                        {
                                            if func.signature.return_type == IrType::Void {
                                                debug!(
                                                "[STDLIB MIR] Function {} returns Void, using actual signature",
                                                mir_func_name
                                            );
                                                IrType::Void
                                            } else if resolved_result_type == IrType::Any
                                                || matches!(resolved_result_type, IrType::Ptr(ref inner) if **inner == IrType::Void)
                                            {
                                                debug!(
                                                "[STDLIB MIR] resolved_result_type is Any/Ptr(Void), using function signature {:?}",
                                                func.signature.return_type
                                            );
                                                func.signature.return_type.clone()
                                            } else {
                                                debug!(
                                                "[STDLIB MIR] Using resolved_result_type {:?} (handles generics)",
                                                resolved_result_type
                                            );
                                                resolved_result_type.clone()
                                            }
                                        } else {
                                            resolved_result_type.clone()
                                        };

                                        debug!(
                                        "[STDLIB MIR] Registered forward ref (instance) to {} with ID {:?}, final return type: {:?}",
                                        mir_func_name, mir_func_id, final_return_type
                                    );

                                        // Generate the call with the MIR wrapper's actual return type
                                        // (which may be Ptr(U8) for generic methods returning T)
                                        let mir_actual_return = self
                                            .get_stdlib_mir_wrapper_signature(&mir_func_name)
                                            .map(|(_, ret)| ret)
                                            .unwrap_or_else(|| final_return_type.clone());
                                        let call_result = self.builder.build_call_direct(
                                            mir_func_id,
                                            arg_regs,
                                            mir_actual_return.clone(),
                                        )?;

                                        // Auto-unbox if MIR wrapper returns Ptr(U8) but caller expects primitive
                                        // (e.g., Channel<Int>.tryReceive() returns boxed int that needs unboxing)
                                        let final_result = if mir_func_name == "Channel_receive"
                                            || mir_func_name == "Channel_tryReceive"
                                        {
                                            self.unbox_channel_return(
                                                call_result,
                                                &resolved_result_type,
                                                mir_func_name == "Channel_tryReceive",
                                            )
                                        } else {
                                            self.maybe_unbox_for_extern_return(
                                                call_result,
                                                &mir_actual_return,
                                                &resolved_result_type,
                                            )
                                        };

                                        // Set class hint on the FINAL result register (after potential unboxing)
                                        // to enable disambiguation of subsequent method calls.
                                        // E.g., Array.iterator() returns ArrayIterator, so subsequent
                                        // .hasNext()/.next() calls dispatch to ArrayIterator methods.
                                        if let Some(result_reg) = final_result {
                                            let return_class = Self::get_return_class_hint(
                                                class_name,
                                                method_name,
                                            );
                                            self.register_class_hints
                                                .insert(result_reg, return_class.to_string());
                                        }

                                        return final_result;
                                    }

                                    // println!(
                                    //     "✅ Generating runtime call to {} (receiver type path)",
                                    //     runtime_func
                                    // );

                                    // Lower all arguments
                                    let arg_regs: Vec<_> = args
                                        .iter()
                                        .filter_map(|a| self.lower_expression(a))
                                        .collect();

                                    // Apply raw value conversion for high-performance inline storage (StringMap, IntMap)
                                    // Values are cast to u64 raw bits - no boxing, no heap allocation
                                    let mut final_arg_regs = arg_regs.clone();
                                    if raw_value_mask != 0 {
                                        for i in 0..arg_regs.len() {
                                            if (raw_value_mask & (1 << i)) != 0 {
                                                let arg_reg = arg_regs[i];
                                                let arg_type = self
                                                    .builder
                                                    .get_register_type(arg_reg)
                                                    .unwrap_or(IrType::I64);

                                                // Cast value to U64 raw bits - zero-cost for same-size types
                                                let raw_reg = match &arg_type {
                                                    IrType::I32 => {
                                                        // Zero-extend i32 to u64
                                                        self.builder.build_cast(
                                                            arg_reg,
                                                            IrType::I32,
                                                            IrType::U64,
                                                        )
                                                    }
                                                    IrType::I64 => {
                                                        // Reinterpret i64 as u64 (same bits) - use cast
                                                        self.builder.build_cast(
                                                            arg_reg,
                                                            IrType::I64,
                                                            IrType::U64,
                                                        )
                                                    }
                                                    IrType::F64 => {
                                                        // Reinterpret f64 bits as u64 - use BitCast instruction
                                                        self.builder
                                                            .build_bitcast(arg_reg, IrType::U64)
                                                    }
                                                    IrType::F32 => {
                                                        // Extend f32 to f64, then reinterpret as u64
                                                        let f64_reg = self
                                                            .builder
                                                            .build_cast(
                                                                arg_reg,
                                                                IrType::F32,
                                                                IrType::F64,
                                                            )
                                                            .unwrap_or(arg_reg);
                                                        self.builder
                                                            .build_bitcast(f64_reg, IrType::U64)
                                                    }
                                                    IrType::Bool => {
                                                        // Zero-extend bool to u64
                                                        self.builder.build_cast(
                                                            arg_reg,
                                                            IrType::Bool,
                                                            IrType::U64,
                                                        )
                                                    }
                                                    IrType::Ptr(_) => {
                                                        // Pointer to u64 (address as integer)
                                                        self.builder.build_cast(
                                                            arg_reg,
                                                            arg_type.clone(),
                                                            IrType::U64,
                                                        )
                                                    }
                                                    _ => {
                                                        // For other types, try direct cast to U64
                                                        self.builder.build_cast(
                                                            arg_reg,
                                                            arg_type.clone(),
                                                            IrType::U64,
                                                        )
                                                    }
                                                };

                                                if let Some(raw) = raw_reg {
                                                    final_arg_regs[i] = raw;
                                                }
                                            }
                                        }
                                    }
                                    // Apply pointer conversion for parameters that need it (DEPRECATED - use raw_value_params)
                                    // This creates boxed Dynamic values for legacy runtime functions.
                                    else if ptr_conversion_mask != 0 {
                                        for i in 0..arg_regs.len() {
                                            // Check if bit i is set in the mask
                                            if (ptr_conversion_mask & (1 << i)) != 0 {
                                                let arg_reg = arg_regs[i];
                                                let arg_type = self
                                                    .builder
                                                    .get_register_type(arg_reg)
                                                    .unwrap_or(IrType::I64);

                                                // Use proper Dynamic boxing based on the argument type
                                                // This creates a tagged Dynamic value that can be unboxed later
                                                // Use the haxe_box_*_ptr wrapper functions which handle type conversion internally
                                                let boxed_reg = match &arg_type {
                                                    IrType::I32 => {
                                                        // Box int using haxe_box_int_ptr wrapper (which handles i32->i64 cast)
                                                        let box_func = self
                                                            .get_or_register_extern_function(
                                                                "haxe_box_int_ptr",
                                                                vec![IrType::I32],
                                                                IrType::Ptr(Box::new(IrType::U8)),
                                                            );
                                                        self.builder.build_call_direct(
                                                            box_func,
                                                            vec![arg_reg],
                                                            IrType::Ptr(Box::new(IrType::U8)),
                                                        )
                                                    }
                                                    IrType::I64 => {
                                                        // Box int64 - truncate to i32 and use haxe_box_int_ptr wrapper
                                                        let truncated = self
                                                            .builder
                                                            .build_cast(
                                                                arg_reg,
                                                                IrType::I64,
                                                                IrType::I32,
                                                            )
                                                            .unwrap_or(arg_reg);
                                                        let box_func = self
                                                            .get_or_register_extern_function(
                                                                "haxe_box_int_ptr",
                                                                vec![IrType::I32],
                                                                IrType::Ptr(Box::new(IrType::U8)),
                                                            );
                                                        self.builder.build_call_direct(
                                                            box_func,
                                                            vec![truncated],
                                                            IrType::Ptr(Box::new(IrType::U8)),
                                                        )
                                                    }
                                                    IrType::F32 | IrType::F64 => {
                                                        // Box float using haxe_box_float_ptr wrapper
                                                        let float_val = if arg_type == IrType::F32 {
                                                            self.builder
                                                                .build_cast(
                                                                    arg_reg,
                                                                    IrType::F32,
                                                                    IrType::F64,
                                                                )
                                                                .unwrap_or(arg_reg)
                                                        } else {
                                                            arg_reg
                                                        };
                                                        let box_func = self
                                                            .get_or_register_extern_function(
                                                                "haxe_box_float_ptr",
                                                                vec![IrType::F64],
                                                                IrType::Ptr(Box::new(IrType::U8)),
                                                            );
                                                        self.builder.build_call_direct(
                                                            box_func,
                                                            vec![float_val],
                                                            IrType::Ptr(Box::new(IrType::U8)),
                                                        )
                                                    }
                                                    IrType::Bool => {
                                                        // Box bool using haxe_box_bool_ptr wrapper
                                                        let box_func = self
                                                            .get_or_register_extern_function(
                                                                "haxe_box_bool_ptr",
                                                                vec![IrType::Bool],
                                                                IrType::Ptr(Box::new(IrType::U8)),
                                                            );
                                                        self.builder.build_call_direct(
                                                            box_func,
                                                            vec![arg_reg],
                                                            IrType::Ptr(Box::new(IrType::U8)),
                                                        )
                                                    }
                                                    IrType::Ptr(_) | IrType::Struct { .. } => {
                                                        // Pointer/reference types still need stack allocation for ptr_params
                                                        // because the runtime function expects a pointer TO the value,
                                                        // and the value itself is a pointer we need to pass BY REFERENCE.
                                                        // Example: haxe_array_push(arr, data) where data = &value
                                                        // For Array<Thread>, value is a pointer, so data = &pointer
                                                        if let Some(stack_slot) = self
                                                            .builder
                                                            .build_alloc(arg_type.clone(), None)
                                                        {
                                                            self.builder
                                                                .build_store(stack_slot, arg_reg);
                                                            Some(stack_slot)
                                                        } else {
                                                            Some(arg_reg)
                                                        }
                                                    }
                                                    _ => {
                                                        // For other types, fallback to stack allocation
                                                        // (This preserves the old behavior for edge cases)
                                                        if let Some(stack_slot) = self
                                                            .builder
                                                            .build_alloc(arg_type.clone(), None)
                                                        {
                                                            self.builder
                                                                .build_store(stack_slot, arg_reg);
                                                            Some(stack_slot)
                                                        } else {
                                                            Some(arg_reg)
                                                        }
                                                    }
                                                };

                                                if let Some(boxed) = boxed_reg {
                                                    final_arg_regs[i] = boxed;
                                                }
                                            }
                                        }
                                    }

                                    // Apply i32 -> i64 extension for IntMap key parameters
                                    // This is needed because Haxe Int is 32-bit but the runtime uses 64-bit keys
                                    if extend_i64_mask != 0 {
                                        for i in 0..final_arg_regs.len() {
                                            if (extend_i64_mask & (1 << i)) != 0 {
                                                let arg_reg = final_arg_regs[i];
                                                let arg_type = self
                                                    .builder
                                                    .get_register_type(arg_reg)
                                                    .unwrap_or(IrType::I32);

                                                // Only extend i32 to i64, skip if already i64
                                                if arg_type == IrType::I32 {
                                                    if let Some(extended) = self.builder.build_cast(
                                                        arg_reg,
                                                        IrType::I32,
                                                        IrType::I64,
                                                    ) {
                                                        final_arg_regs[i] = extended;
                                                    }
                                                }
                                            }
                                        }
                                    }

                                    // Get or register the extern runtime function
                                    // Use actual argument types from TAST, applying type conversion where needed
                                    let param_types: Vec<IrType> = args
                                        .iter()
                                        .enumerate()
                                        .map(|(i, arg)| {
                                            // Raw value params are passed as U64 (high-performance inline storage)
                                            if raw_value_mask != 0
                                                && (raw_value_mask & (1 << i)) != 0
                                            {
                                                IrType::U64
                                            }
                                            // Extended i64 params need i64 type in signature
                                            else if extend_i64_mask != 0
                                                && (extend_i64_mask & (1 << i)) != 0
                                            {
                                                IrType::I64
                                            }
                                            // Legacy ptr_conversion params are passed as Ptr (boxed Dynamic)
                                            else if ptr_conversion_mask != 0
                                                && (ptr_conversion_mask & (1 << i)) != 0
                                            {
                                                IrType::Ptr(Box::new(IrType::U8))
                                            } else {
                                                self.convert_type(arg.ty)
                                            }
                                        })
                                        .collect();

                                    // For functions that return raw values (u64), we need to:
                                    // 1. Resolve the actual type parameter T from the receiver's generic args
                                    // 2. Call with U64 return type
                                    // 3. Cast the result to the resolved type
                                    //
                                    // `resolved_from_type_args` records whether the resolution
                                    // came from a real receiver type_arg substitution (true) or
                                    // fell through to `result_type` because there were no type
                                    // args. The U64 → Ptr post-cast uses this to decide whether
                                    // bitcasting a Ptr-typed result is safe — bitcasting a Ptr
                                    // when the source u64 actually holds an Int (because T was
                                    // unresolved) would silently produce a garbage address.
                                    let (resolved_return_type, resolved_from_type_args) =
                                        if returns_raw_value {
                                            // Resolve value T from receiver's type args. The
                                            // LAST type arg is V — covers both single-param
                                            // (StringMap<T>, IntMap<T>) and two-param
                                            // (ObjectMap<K, V>) container shapes.
                                            let type_table = self.type_table;
                                            let resolved =
                                                if let Some(receiver_info) =
                                                    type_table.get(receiver_type)
                                                {
                                                    match &receiver_info.kind {
                                                crate::tast::TypeKind::Class { type_args, .. }
                                                | crate::tast::TypeKind::GenericInstance {
                                                    type_args, ..
                                                } => type_args
                                                    .last()
                                                    .map(|ta| self.convert_type(*ta)),
                                                _ => None,
                                            }
                                                } else {
                                                    None
                                                };
                                            match resolved {
                                                Some(t) => (t, true),
                                                None => (result_type.clone(), false),
                                            }
                                        } else {
                                            // IMPORTANT: For MIR wrappers, use their actual return type instead of HIR type
                                            // HIR type may be Dynamic/Ptr(Void) but the wrapper returns a concrete type (e.g., Bool)
                                            let ret = self
                                                .get_stdlib_mir_wrapper_signature(&runtime_func)
                                                .map(|(_, ret_ty)| ret_ty)
                                                .unwrap_or_else(|| {
                                                    if has_return {
                                                        result_type.clone()
                                                    } else {
                                                        IrType::Void
                                                    }
                                                });
                                            (ret, false)
                                        };
                                    debug!(
                                    "[RESOLVED RETURN TYPE] runtime_func={}, result_type={:?}, resolved={:?}",
                                    runtime_func, result_type, resolved_return_type
                                );

                                    let call_return_type = if returns_raw_value {
                                        IrType::U64
                                    } else {
                                        resolved_return_type.clone()
                                    };

                                    let runtime_func_id = self.get_or_register_extern_function(
                                        &runtime_func,
                                        param_types,
                                        call_return_type.clone(),
                                    );

                                    // Generate the call to the runtime function
                                    let call_result = self.builder.build_call_direct(
                                        runtime_func_id,
                                        final_arg_regs,
                                        call_return_type,
                                    );

                                    // If this returns raw value, cast U64 back to the resolved type parameter
                                    if returns_raw_value {
                                        if let Some(raw_reg) = call_result {
                                            // Cast U64 to the resolved type parameter
                                            let final_result = match &resolved_return_type {
                                                IrType::I32 => self.builder.build_cast(
                                                    raw_reg,
                                                    IrType::U64,
                                                    IrType::I32,
                                                ),
                                                IrType::I64 => self.builder.build_cast(
                                                    raw_reg,
                                                    IrType::U64,
                                                    IrType::I64,
                                                ),
                                                IrType::F64 => {
                                                    self.builder.build_bitcast(raw_reg, IrType::F64)
                                                }
                                                IrType::F32 => {
                                                    // Bitcast to F64, then convert to F32
                                                    if let Some(f64_reg) = self
                                                        .builder
                                                        .build_bitcast(raw_reg, IrType::F64)
                                                    {
                                                        self.builder.build_cast(
                                                            f64_reg,
                                                            IrType::F64,
                                                            IrType::F32,
                                                        )
                                                    } else {
                                                        None
                                                    }
                                                }
                                                IrType::Bool => self.builder.build_cast(
                                                    raw_reg,
                                                    IrType::U64,
                                                    IrType::Bool,
                                                ),
                                                IrType::Ptr(_) => {
                                                    // Pointer type — bit-reinterpret u64 → ptr,
                                                    // BUT only when we actually resolved T from
                                                    // the receiver's type_args. When T was
                                                    // unresolved (e.g. `new StringMap()` with
                                                    // no type args), the user is storing
                                                    // primitives as raw bits — bitcasting to
                                                    // Ptr would silently mangle them. The
                                                    // legacy U64→I64 cast keeps the bits as
                                                    // an integer for downstream unboxing.
                                                    if resolved_from_type_args {
                                                        self.builder.build_bitcast(
                                                            raw_reg,
                                                            resolved_return_type.clone(),
                                                        )
                                                    } else {
                                                        self.builder.build_cast(
                                                            raw_reg,
                                                            IrType::U64,
                                                            IrType::I64,
                                                        )
                                                    }
                                                }
                                                _ => {
                                                    // Truly unresolved T (Dynamic, type parameter)
                                                    // — keep as I64 so the raw value isn't
                                                    // misinterpreted as anything else.
                                                    self.builder.build_cast(
                                                        raw_reg,
                                                        IrType::U64,
                                                        IrType::I64,
                                                    )
                                                }
                                            };
                                            return final_result;
                                        }
                                    }

                                    return call_result;
                                }

                                // GUARD: Check if receiver is a user-defined class (not stdlib)
                                // If so, skip all stdlib fallbacks - they would incorrectly match stdlib methods
                                let receiver_is_user_class = {
                                    let type_table = self.type_table;
                                    type_table
                                        .get(receiver_type)
                                        .map(|ti| {
                                            match &ti.kind {
                                                crate::tast::core::TypeKind::Class {
                                                    symbol_id,
                                                    ..
                                                } => {
                                                    // Check if this is a stdlib class
                                                    self.symbol_table
                                                        .get_symbol(*symbol_id)
                                                        .map(|s| !self.is_stdlib_class_by_symbol(s))
                                                        .unwrap_or(false)
                                                }
                                                // TypeParameter receivers always come from user-defined generics.
                                                // Method calls on T should resolve through function_map, not stdlib.
                                                // (Constrained T:Interface is handled earlier by interface dispatch.)
                                                crate::tast::core::TypeKind::TypeParameter {
                                                    ..
                                                } => true,
                                                // GenericInstance: check if the base type is a user class
                                                crate::tast::core::TypeKind::GenericInstance {
                                                    base_type,
                                                    ..
                                                } => type_table
                                                    .get(*base_type)
                                                    .map(|bt| {
                                                        if let crate::tast::core::TypeKind::Class {
                                                        symbol_id,
                                                        ..
                                                    } = &bt.kind
                                                    {
                                                        self.symbol_table
                                                            .get_symbol(*symbol_id)
                                                            .map(|s| {
                                                                !self.is_stdlib_class_by_symbol(s)
                                                            })
                                                            .unwrap_or(false)
                                                    } else {
                                                        false
                                                    }
                                                    })
                                                    .unwrap_or(false),
                                                // Abstract types with user-defined methods
                                                crate::tast::core::TypeKind::Abstract {
                                                    symbol_id,
                                                    ..
                                                } => self
                                                    .symbol_table
                                                    .get_symbol(*symbol_id)
                                                    .map(|s| !self.is_stdlib_class_by_symbol(s))
                                                    .unwrap_or(false),
                                                _ => false,
                                            }
                                        })
                                        .unwrap_or(false)
                                };

                                // Skip stdlib fallbacks for user-defined classes
                                if receiver_is_user_class {
                                    // For user-defined classes, the method should be in function_map
                                    // Don't try to match stdlib methods
                                } else {
                                    // Fallback: Use stdlib mapping to try all possible class/method combinations
                                    // This is necessary when qualified names aren't set properly
                                    if let Some(method_sym) = self.symbol_table.get_symbol(*symbol)
                                    {
                                        if let Some(method_name) =
                                            self.string_interner.get(method_sym.name)
                                        {
                                            let static_args = self.effective_static_call_args(args);
                                            // First try to use the qualified name if available
                                            if let Some(qual_name) = method_sym
                                                .qualified_name
                                                .and_then(|qn| self.string_interner.get(qn))
                                            {
                                                if let Some(runtime_func) = self
                                                    .get_static_stdlib_runtime_func_with_params(
                                                        qual_name,
                                                        method_name,
                                                        static_args.len(),
                                                    )
                                                {
                                                    // CHECK: Is this a MIR wrapper function or a true extern?
                                                    // The mapping's `is_mir_wrapper` flag decides — having
                                                    // explicit type info does NOT (typed extern intrinsics
                                                    // like `haxe_bytes_get` carry signatures too; routing
                                                    // them here creates a body-less forward-ref stub that
                                                    // traps at runtime).
                                                    if let Some((
                                                        _mir_param_types,
                                                        _mir_return_type,
                                                    )) = self
                                                        .get_stdlib_mir_wrapper_signature(
                                                            runtime_func,
                                                        )
                                                        .filter(|_| {
                                                            self.stdlib_mapping
                                                                .is_mir_wrapper_function(
                                                                    runtime_func,
                                                                )
                                                        })
                                                    {
                                                        debug!(
                                                        "[QUALIFIED NAME PATH] Detected MIR wrapper: {}",
                                                        runtime_func
                                                    );

                                                        // Lower all arguments and collect their types
                                                        let mut arg_regs = Vec::new();
                                                        let mut param_types = Vec::new();
                                                        for arg in static_args {
                                                            if let Some(reg) =
                                                                self.lower_expression(arg)
                                                            {
                                                                arg_regs.push(reg);
                                                                param_types.push(
                                                                    self.convert_type(arg.ty),
                                                                );
                                                            }
                                                        }

                                                        // Register forward reference - will be provided by merged stdlib module
                                                        let mir_func_id = self
                                                            .register_stdlib_mir_forward_ref(
                                                                runtime_func,
                                                                param_types,
                                                                result_type.clone(),
                                                            );

                                                        debug!(
                                                        "[QUALIFIED NAME PATH] Registered forward ref to {} with ID {:?}",
                                                        runtime_func, mir_func_id
                                                    );

                                                        // Generate the call
                                                        let result =
                                                            self.builder.build_call_direct(
                                                                mir_func_id,
                                                                arg_regs,
                                                                result_type,
                                                            );
                                                        debug!(
                                                        "[QUALIFIED NAME PATH] Generated call, result: {:?}",
                                                        result
                                                    );
                                                        return result;
                                                    }

                                                    // Lower all arguments
                                                    let arg_regs: Vec<_> = static_args
                                                        .iter()
                                                        .filter_map(|a| self.lower_expression(a))
                                                        .collect();

                                                    // Get expected types FIRST so we can auto-box before ptr_conversion
                                                    let (
                                                        expected_param_types_qn,
                                                        expected_return_type_qn,
                                                    ) = self
                                                        .get_extern_function_signature(
                                                            &runtime_func,
                                                        )
                                                        .unwrap_or_else(|| {
                                                            let param_types: Vec<IrType> =
                                                                static_args
                                                                    .iter()
                                                                    .map(|arg| {
                                                                        self.convert_type(arg.ty)
                                                                    })
                                                                    .collect();
                                                            (param_types, result_type.clone())
                                                        });

                                                    // Auto-box arguments when expected type is Ptr(U8) (Dynamic)
                                                    let mut final_arg_regs: Vec<_> = arg_regs.iter().enumerate()
                                                    .map(|(i, &reg)| {
                                                        if let (Some(expected_ty), Some(actual_ty)) = (
                                                            expected_param_types_qn.get(i),
                                                            self.builder.get_register_type(reg)
                                                        ) {
                                                            if *expected_ty != actual_ty {
                                                                let is_ptr_u8 = matches!(expected_ty, IrType::Ptr(inner) if matches!(inner.as_ref(), IrType::U8));
                                                                if is_ptr_u8 && i < static_args.len() {
                                                                    if let Some(boxed) = self.box_value_for_dynamic(reg, static_args[i].ty) {
                                                                        return boxed;
                                                                    }
                                                                }
                                                                if let Some(casted) = self.builder.build_cast(reg, actual_ty.clone(), expected_ty.clone()) {
                                                                    return casted;
                                                                }
                                                            }
                                                        }
                                                        reg
                                                    })
                                                    .collect();

                                                    let runtime_func_id_qn = self
                                                        .get_or_register_extern_function(
                                                            &runtime_func,
                                                            expected_param_types_qn,
                                                            expected_return_type_qn.clone(),
                                                        );

                                                    // Generate the call to the runtime function
                                                    let call_result_qn =
                                                        self.builder.build_call_direct(
                                                            runtime_func_id_qn,
                                                            final_arg_regs,
                                                            expected_return_type_qn.clone(),
                                                        )?;
                                                    return Some(self.reconcile_extern_return(
                                                        call_result_qn,
                                                        &expected_return_type_qn,
                                                        &result_type,
                                                    ));

                                                    // DEAD CODE below (kept for reference): old ptr_conversion path
                                                    #[allow(unreachable_code)]
                                                    let _unused_final_arg_regs = arg_regs.clone();
                                                    #[allow(unreachable_code)]
                                                    let mut final_arg_regs = arg_regs.clone();
                                                    let ptr_conversion_mask = self
                                                        .stdlib_mapping
                                                        .find_by_runtime_name(&runtime_func)
                                                        .map(|m| m.params_need_ptr_conversion)
                                                        .unwrap_or(0);
                                                    if ptr_conversion_mask != 0 {
                                                        for i in 0..arg_regs.len() {
                                                            // Check if bit i is set in the mask
                                                            if (ptr_conversion_mask & (1 << i)) != 0
                                                            {
                                                                let arg_reg = arg_regs[i];
                                                                // Default to I64 (pointer-sized) if type is unknown.
                                                                // This is safer than I32 since pointers and most values are 64-bit.
                                                                let arg_type = self
                                                                    .builder
                                                                    .get_register_type(arg_reg)
                                                                    .unwrap_or(IrType::I64);

                                                                // For array operations, always allocate 8 bytes (elem_size is always 8)
                                                                // and extend smaller values to 64-bit
                                                                let (alloc_type, value_to_store) =
                                                                    match arg_type {
                                                                        IrType::I32 => {
                                                                            let ext_val = self
                                                                                .builder
                                                                                .build_cast(
                                                                                    arg_reg,
                                                                                    IrType::I32,
                                                                                    IrType::I64,
                                                                                );
                                                                            (
                                                                                IrType::I64,
                                                                                ext_val.unwrap_or(
                                                                                    arg_reg,
                                                                                ),
                                                                            )
                                                                        }
                                                                        IrType::F32 => {
                                                                            let ext_val = self
                                                                                .builder
                                                                                .build_cast(
                                                                                    arg_reg,
                                                                                    IrType::F32,
                                                                                    IrType::F64,
                                                                                );
                                                                            (
                                                                                IrType::F64,
                                                                                ext_val.unwrap_or(
                                                                                    arg_reg,
                                                                                ),
                                                                            )
                                                                        }
                                                                        _ => (
                                                                            arg_type.clone(),
                                                                            arg_reg,
                                                                        ),
                                                                    };

                                                                // Allocate stack space and pass a pointer to the value.
                                                                if let Some(stack_slot) =
                                                                    self.builder.build_alloc(
                                                                        alloc_type.clone(),
                                                                        None,
                                                                    )
                                                                {
                                                                    // Store the value into the stack slot
                                                                    self.builder.build_store(
                                                                        stack_slot,
                                                                        value_to_store,
                                                                    );
                                                                    // Use the pointer for the call
                                                                    final_arg_regs[i] = stack_slot;
                                                                }
                                                            }
                                                        }
                                                    }

                                                    // Use the function signature from the mapping (hlp_* introspection)
                                                    // if available; this is the authoritative source of type info.
                                                    let (
                                                        expected_param_types,
                                                        expected_return_type,
                                                    ) = self
                                                        .get_extern_function_signature(
                                                            &runtime_func,
                                                        )
                                                        .unwrap_or_else(|| {
                                                            let param_types: Vec<IrType> = args
                                                                .iter()
                                                                .enumerate()
                                                                .map(|(i, arg)| {
                                                                    if ptr_conversion_mask != 0
                                                                        && (ptr_conversion_mask
                                                                            & (1 << i))
                                                                            != 0
                                                                    {
                                                                        IrType::Ptr(Box::new(
                                                                            IrType::U8,
                                                                        ))
                                                                    } else {
                                                                        self.convert_type(arg.ty)
                                                                    }
                                                                })
                                                                .collect();
                                                            (param_types, result_type.clone())
                                                        });
                                                    let runtime_func_id = self
                                                        .get_or_register_extern_function(
                                                            &runtime_func,
                                                            expected_param_types,
                                                            expected_return_type.clone(),
                                                        );

                                                    // Generate the call to the runtime function
                                                    return self.builder.build_call_direct(
                                                        runtime_func_id,
                                                        final_arg_regs,
                                                        expected_return_type,
                                                    );
                                                }
                                            }

                                            // Fallback: try each possible stdlib class (only if qualified name didn't work)
                                            // For static methods like Arc.init, Mutex.init, etc, try to infer the class from the return type
                                            // debug!("Qualified name not available, trying to infer class from return type={:?}", expr.ty);

                                            let inferred_class = {
                                                let type_table = self.type_table;
                                                debug!(
                                                "[INFER CLASS] Checking return type expr.ty={:?}",
                                                expr.ty
                                            );
                                                if let Some(type_info) = type_table.get(expr.ty) {
                                                    debug!(
                                                        "[INFER CLASS] Return type kind={:?}",
                                                        type_info.kind
                                                    );
                                                    if let TypeKind::Class { symbol_id, .. } =
                                                        &type_info.kind
                                                    {
                                                        if let Some(class_sym) =
                                                            self.symbol_table.get_symbol(*symbol_id)
                                                        {
                                                            let class_name = self
                                                                .string_interner
                                                                .get(class_sym.name);
                                                            debug!(
                                                            "[INFER CLASS] Inferred class from return type: {:?}",
                                                            class_name
                                                        );
                                                            class_name
                                                        } else {
                                                            debug!(
                                                            "[INFER CLASS] Class symbol not found"
                                                        );
                                                            None
                                                        }
                                                    } else {
                                                        debug!(
                                                        "[INFER CLASS] Return type is not a class"
                                                    );
                                                        None
                                                    }
                                                } else {
                                                    debug!(
                                                    "[INFER CLASS] Type info not found for expr.ty={:?}",
                                                    expr.ty
                                                );
                                                    None
                                                }
                                            };

                                            if let Some(class_name) = inferred_class {
                                                // SPECIAL CASE: Check if this is a stdlib MIR function
                                                if self
                                                    .stdlib_mapping
                                                    .is_mir_wrapper_class(class_name)
                                                {
                                                    // The mapping is the source of truth for the
                                                    // wrapper's name — synthesizing it by the
                                                    // `{class.lowercase()}_{method}` convention
                                                    // produces a body-less stub whenever the real
                                                    // entry differs (`QTensor_requantQ6KToQ4KM` vs
                                                    // `qtensor_requantQ6KToQ4KM` → trap at call).
                                                    // The class here was inferred from the RETURN
                                                    // type, which differs from the declaring class
                                                    // for non-factory methods (QTensor.gatherRowsQ6K
                                                    // returns Tensor) — a globally unique method
                                                    // name still identifies the entry.
                                                    let mir_func_name = self
                                                        .stdlib_mapping
                                                        .find_by_name(class_name, method_name)
                                                        .or_else(|| {
                                                            self.stdlib_mapping
                                                                .find_unique_by_method(method_name)
                                                        })
                                                        .map(|(_, call)| {
                                                            call.runtime_name.to_string()
                                                        })
                                                        .unwrap_or_else(|| {
                                                            format!(
                                                                "{}_{}",
                                                                class_name.to_lowercase(),
                                                                method_name
                                                            )
                                                        });
                                                    debug!(
                                                    "[STDLIB MIR] Detected stdlib MIR function: {}",
                                                    mir_func_name
                                                );

                                                    // Lower all arguments and collect their types
                                                    let mut arg_regs = Vec::new();
                                                    let mut param_types = Vec::new();
                                                    for arg in static_args {
                                                        if let Some(reg) =
                                                            self.lower_expression(arg)
                                                        {
                                                            arg_regs.push(reg);
                                                            param_types
                                                                .push(self.convert_type(arg.ty));
                                                        }
                                                    }

                                                    // Register forward reference - will be provided by merged stdlib module
                                                    let mir_func_id = self
                                                        .register_stdlib_mir_forward_ref(
                                                            &mir_func_name,
                                                            param_types,
                                                            result_type.clone(),
                                                        );

                                                    debug!(
                                                    "[STDLIB MIR] Registered forward ref to {} with ID {:?}",
                                                    mir_func_name, mir_func_id
                                                );

                                                    // Generate the call
                                                    let result = self.builder.build_call_direct(
                                                        mir_func_id,
                                                        arg_regs,
                                                        result_type,
                                                    );
                                                    debug!(
                                                        "[STDLIB MIR] Generated call, result: {:?}",
                                                        result
                                                    );
                                                    return result;
                                                }

                                                // Try the inferred class first
                                                let fake_qual_name = format!(
                                                    "rayzor.concurrent.{}.{}",
                                                    class_name, method_name
                                                );
                                                if let Some(runtime_func) = self
                                                    .get_static_stdlib_runtime_func_with_params(
                                                        &fake_qual_name,
                                                        method_name,
                                                        static_args.len(),
                                                    )
                                                {
                                                    debug!(
                                                    "[INFERRED CLASS PATH] Got runtime_func='{}' for class={}, method={}",
                                                    runtime_func, class_name, method_name
                                                );
                                                    // println!("✅ Generating runtime call to {} for {}.{} (inferred from return type)", runtime_func, class_name, method_name);

                                                    // Lower all arguments
                                                    let arg_regs: Vec<_> = static_args
                                                        .iter()
                                                        .filter_map(|a| self.lower_expression(a))
                                                        .collect();

                                                    // Apply pointer conversion for parameters that need it (metadata-driven)
                                                    // Look up the RuntimeFunctionCall metadata by runtime function name
                                                    // This means the runtime function expects a POINTER TO the value, not the value directly.
                                                    let mut final_arg_regs = arg_regs.clone();
                                                    let ptr_conversion_mask = self
                                                        .stdlib_mapping
                                                        .find_by_runtime_name(&runtime_func)
                                                        .map(|m| m.params_need_ptr_conversion)
                                                        .unwrap_or(0);
                                                    if ptr_conversion_mask != 0 {
                                                        for i in 0..arg_regs.len() {
                                                            // Check if bit i is set in the mask
                                                            if (ptr_conversion_mask & (1 << i)) != 0
                                                            {
                                                                let arg_reg = arg_regs[i];
                                                                // Default to I64 (pointer-sized) if type is unknown.
                                                                // This is safer than I32 since pointers and most values are 64-bit.
                                                                let arg_type = self
                                                                    .builder
                                                                    .get_register_type(arg_reg)
                                                                    .unwrap_or(IrType::I64);

                                                                // For array operations, always allocate 8 bytes (elem_size is always 8)
                                                                // and extend smaller values to 64-bit
                                                                let (alloc_type, value_to_store) =
                                                                    match arg_type {
                                                                        IrType::I32 => {
                                                                            let ext_val = self
                                                                                .builder
                                                                                .build_cast(
                                                                                    arg_reg,
                                                                                    IrType::I32,
                                                                                    IrType::I64,
                                                                                );
                                                                            (
                                                                                IrType::I64,
                                                                                ext_val.unwrap_or(
                                                                                    arg_reg,
                                                                                ),
                                                                            )
                                                                        }
                                                                        IrType::F32 => {
                                                                            let ext_val = self
                                                                                .builder
                                                                                .build_cast(
                                                                                    arg_reg,
                                                                                    IrType::F32,
                                                                                    IrType::F64,
                                                                                );
                                                                            (
                                                                                IrType::F64,
                                                                                ext_val.unwrap_or(
                                                                                    arg_reg,
                                                                                ),
                                                                            )
                                                                        }
                                                                        _ => (
                                                                            arg_type.clone(),
                                                                            arg_reg,
                                                                        ),
                                                                    };

                                                                // Allocate stack space and pass a pointer to the value.
                                                                if let Some(stack_slot) =
                                                                    self.builder.build_alloc(
                                                                        alloc_type.clone(),
                                                                        None,
                                                                    )
                                                                {
                                                                    // Store the value into the stack slot
                                                                    self.builder.build_store(
                                                                        stack_slot,
                                                                        value_to_store,
                                                                    );
                                                                    // Use the pointer for the call
                                                                    final_arg_regs[i] = stack_slot;
                                                                }
                                                            }
                                                        }
                                                    }

                                                    // Use the function signature from the mapping (hlp_* introspection)
                                                    // if available; this is the authoritative source of type info.
                                                    let (
                                                        expected_param_types,
                                                        expected_return_type,
                                                    ) = self
                                                        .get_extern_function_signature(
                                                            &runtime_func,
                                                        )
                                                        .unwrap_or_else(|| {
                                                            let param_types: Vec<IrType> = args
                                                                .iter()
                                                                .enumerate()
                                                                .map(|(i, arg)| {
                                                                    if ptr_conversion_mask != 0
                                                                        && (ptr_conversion_mask
                                                                            & (1 << i))
                                                                            != 0
                                                                    {
                                                                        IrType::Ptr(Box::new(
                                                                            IrType::U8,
                                                                        ))
                                                                    } else {
                                                                        self.convert_type(arg.ty)
                                                                    }
                                                                })
                                                                .collect();
                                                            (param_types, result_type.clone())
                                                        });
                                                    let runtime_func_id = self
                                                        .get_or_register_extern_function(
                                                            &runtime_func,
                                                            expected_param_types,
                                                            expected_return_type.clone(),
                                                        );

                                                    // Generate the call to the runtime function
                                                    return self.builder.build_call_direct(
                                                        runtime_func_id,
                                                        final_arg_regs,
                                                        expected_return_type,
                                                    );
                                                }
                                            }

                                            // Last resort: try all stdlib classes with param count matching
                                            // NOTE: We must match by param count to disambiguate overloaded methods
                                            // (e.g., Array.join(sep) with 1 param vs Thread.join() with 0 params)
                                            let actual_arg_count = args.len().saturating_sub(1); // Subtract 1 for receiver (self)
                                            debug!(
                                            "[LAST RESORT] Could not infer class for method '{}' with {} args, trying all stdlib classes",
                                            method_name, actual_arg_count
                                        );
                                            // Get all stdlib classes dynamically from the mapping
                                            // NOTE: We do NOT add stdlib MIR detection here because we don't know which class
                                            // to use - the fallback tries all classes and would match the wrong one
                                            let stdlib_classes =
                                                self.stdlib_mapping.get_all_classes();
                                            for class_name in &stdlib_classes {
                                                // Use find_by_name_and_params to ensure param count matches
                                                // This prevents Array.join(1 param) from matching Thread.join(0 params)
                                                if let Some((sig, mapping)) =
                                                    self.stdlib_mapping.find_by_name_and_params(
                                                        class_name,
                                                        method_name,
                                                        actual_arg_count,
                                                    )
                                                {
                                                    let runtime_func = mapping.runtime_name;

                                                    // CHECK: Is this a MIR wrapper or an extern?
                                                    // Gate on the mapping's `is_mir_wrapper` flag —
                                                    // typed extern intrinsics carry signatures too,
                                                    // and a forward-ref stub for one never gets a
                                                    // body (traps at runtime).
                                                    if let Some((
                                                        mir_param_types,
                                                        mir_return_type,
                                                    )) = self
                                                        .get_stdlib_mir_wrapper_signature(
                                                            &runtime_func,
                                                        )
                                                        .filter(|_| mapping.is_mir_wrapper)
                                                    {
                                                        debug!(
                                                        "[FALLBACK PATH] Detected MIR wrapper: {}",
                                                        runtime_func
                                                    );

                                                        // Lower all arguments
                                                        let mut arg_regs = Vec::new();
                                                        for arg in args {
                                                            if let Some(reg) =
                                                                self.lower_expression(arg)
                                                            {
                                                                arg_regs.push(reg);
                                                            }
                                                        }

                                                        // Register forward reference - signature comes from get_stdlib_mir_wrapper_signature
                                                        let mir_func_id = self
                                                            .register_stdlib_mir_forward_ref(
                                                                &runtime_func,
                                                                mir_param_types,
                                                                mir_return_type,
                                                            );

                                                        debug!(
                                                        "[FALLBACK PATH] Registered forward ref to {} with ID {:?}",
                                                        runtime_func, mir_func_id
                                                    );

                                                        // Generate the call
                                                        let result =
                                                            self.builder.build_call_direct(
                                                                mir_func_id,
                                                                arg_regs,
                                                                result_type,
                                                            );
                                                        debug!(
                                                        "[FALLBACK PATH] Generated call, result: {:?}",
                                                        result
                                                    );
                                                        return result;
                                                    }

                                                    // Lower all arguments
                                                    let arg_regs: Vec<_> = args
                                                        .iter()
                                                        .filter_map(|a| self.lower_expression(a))
                                                        .collect();

                                                    // Apply pointer conversion for parameters that need it (metadata-driven)
                                                    // Look up the RuntimeFunctionCall metadata by runtime function name
                                                    // This means the runtime function expects a POINTER TO the value, not the value directly.
                                                    let mut final_arg_regs = arg_regs.clone();
                                                    let ptr_conversion_mask = self
                                                        .stdlib_mapping
                                                        .find_by_runtime_name(&runtime_func)
                                                        .map(|m| m.params_need_ptr_conversion)
                                                        .unwrap_or(0);
                                                    if ptr_conversion_mask != 0 {
                                                        for i in 0..arg_regs.len() {
                                                            // Check if bit i is set in the mask
                                                            if (ptr_conversion_mask & (1 << i)) != 0
                                                            {
                                                                let arg_reg = arg_regs[i];
                                                                // Default to I64 (pointer-sized) if type is unknown.
                                                                // This is safer than I32 since pointers and most values are 64-bit.
                                                                let arg_type = self
                                                                    .builder
                                                                    .get_register_type(arg_reg)
                                                                    .unwrap_or(IrType::I64);

                                                                // For array operations, always allocate 8 bytes (elem_size is always 8)
                                                                // and extend smaller values to 64-bit
                                                                let (alloc_type, value_to_store) =
                                                                    match arg_type {
                                                                        IrType::I32 => {
                                                                            let ext_val = self
                                                                                .builder
                                                                                .build_cast(
                                                                                    arg_reg,
                                                                                    IrType::I32,
                                                                                    IrType::I64,
                                                                                );
                                                                            (
                                                                                IrType::I64,
                                                                                ext_val.unwrap_or(
                                                                                    arg_reg,
                                                                                ),
                                                                            )
                                                                        }
                                                                        IrType::F32 => {
                                                                            let ext_val = self
                                                                                .builder
                                                                                .build_cast(
                                                                                    arg_reg,
                                                                                    IrType::F32,
                                                                                    IrType::F64,
                                                                                );
                                                                            (
                                                                                IrType::F64,
                                                                                ext_val.unwrap_or(
                                                                                    arg_reg,
                                                                                ),
                                                                            )
                                                                        }
                                                                        _ => (
                                                                            arg_type.clone(),
                                                                            arg_reg,
                                                                        ),
                                                                    };

                                                                // Allocate stack space and pass a pointer to the value.
                                                                if let Some(stack_slot) =
                                                                    self.builder.build_alloc(
                                                                        alloc_type.clone(),
                                                                        None,
                                                                    )
                                                                {
                                                                    // Store the value into the stack slot
                                                                    self.builder.build_store(
                                                                        stack_slot,
                                                                        value_to_store,
                                                                    );
                                                                    // Use the pointer for the call
                                                                    final_arg_regs[i] = stack_slot;
                                                                }
                                                            }
                                                        }
                                                    }

                                                    // Get or register the extern runtime function
                                                    // Use actual argument types from TAST, applying ptr conversion where needed
                                                    let param_types: Vec<IrType> = args
                                                        .iter()
                                                        .enumerate()
                                                        .map(|(i, arg)| {
                                                            // If this param was converted to a pointer, the type is Ptr
                                                            if ptr_conversion_mask != 0
                                                                && (ptr_conversion_mask & (1 << i))
                                                                    != 0
                                                            {
                                                                IrType::Ptr(Box::new(IrType::U8))
                                                            } else {
                                                                self.convert_type(arg.ty)
                                                            }
                                                        })
                                                        .collect();
                                                    let runtime_func_id = self
                                                        .get_or_register_extern_function(
                                                            &runtime_func,
                                                            param_types,
                                                            result_type.clone(),
                                                        );

                                                    // Generate the call to the runtime function
                                                    return self.builder.build_call_direct(
                                                        runtime_func_id,
                                                        final_arg_regs,
                                                        result_type,
                                                    );
                                                }
                                            }
                                        }
                                    }
                                } // end of else block for receiver_is_user_class
                            }
                        } else {
                            // receiver_is_class_type == true
                            // This is an instance method call on a MIR wrapper class (Thread, Channel, etc.)
                            // Route to the MIR wrapper function (Thread_join, Channel_send, etc.)
                            let receiver_is_synthetic_class = args
                                .first()
                                .map(|arg| self.is_class_symbol_expr(arg))
                                .unwrap_or(false);
                            if !receiver_is_synthetic_class {
                                if let Some(sym_info) = self.symbol_table.get_symbol(*symbol) {
                                    if let Some(method_name) =
                                        self.string_interner.get(sym_info.name)
                                    {
                                        // Get the class name from the receiver type
                                        let class_name = {
                                            let type_table = self.type_table;
                                            type_table.get(receiver_type).and_then(|ti| {
                                                if let crate::tast::core::TypeKind::Class {
                                                    symbol_id,
                                                    ..
                                                } = &ti.kind
                                                {
                                                    self.symbol_table
                                                        .get_symbol(*symbol_id)
                                                        .and_then(|s| {
                                                            self.string_interner.get(s.name)
                                                        })
                                                        .map(|s| s.to_string())
                                                } else {
                                                    None
                                                }
                                            })
                                        };

                                        if let Some(class_name) = class_name {
                                            // Build MIR wrapper function name: Thread_join, Channel_send, etc.
                                            let mir_func_name =
                                                format!("{}_{}", class_name, method_name);
                                            debug!(
                                                "[MIR WRAPPER INSTANCE] Routing {}.{} to {}",
                                                class_name, method_name, mir_func_name
                                            );

                                            // Get the registered signature for this MIR wrapper
                                            if let Some((mir_param_types, mir_return_type)) = self
                                                .get_stdlib_mir_wrapper_signature(&mir_func_name)
                                            {
                                                let call_args = if args.len()
                                                    == mir_param_types.len() + 1
                                                    && !args.is_empty()
                                                {
                                                    &args[1..]
                                                } else {
                                                    &args[..]
                                                };
                                                // Lower all arguments (first arg is receiver/self)
                                                // Auto-box primitive args when MIR wrapper expects Ptr(U8)
                                                // (e.g., Channel<Int>.send(42) needs to box the Int)
                                                let mut arg_regs = Vec::new();
                                                for (i, arg) in call_args.iter().enumerate() {
                                                    if let Some(reg) = self.lower_expression(arg) {
                                                        let actual_ty = self.convert_type(arg.ty);
                                                        let expected_ty = mir_param_types
                                                            .get(i)
                                                            .cloned()
                                                            .unwrap_or_else(|| actual_ty.clone());

                                                        // Auto-box if MIR wrapper expects Ptr(U8) but arg is primitive.
                                                        // Channel payloads box uniformly (refs too); i==0 is the
                                                        // channel handle/self (a reference) — never box it.
                                                        let final_reg = if (mir_func_name
                                                            == "Channel_send"
                                                            || mir_func_name == "Channel_trySend")
                                                            && i >= 1
                                                        {
                                                            self.box_channel_payload(
                                                                reg,
                                                                arg.ty,
                                                                &actual_ty,
                                                                &expected_ty,
                                                            )?
                                                        } else {
                                                            self.maybe_box_for_extern_call(
                                                                reg,
                                                                &actual_ty,
                                                                &expected_ty,
                                                            )?
                                                        };
                                                        arg_regs.push(final_reg);
                                                    }
                                                }

                                                // Register forward reference to MIR wrapper
                                                let mir_func_id = self
                                                    .register_stdlib_mir_forward_ref(
                                                        &mir_func_name,
                                                        mir_param_types,
                                                        mir_return_type.clone(),
                                                    );

                                                debug!(
                                                "[MIR WRAPPER INSTANCE] Registered forward ref to {} with ID {:?}",
                                                mir_func_name, mir_func_id
                                            );

                                                // Generate the call with the MIR wrapper's return type
                                                let call_result = self.builder.build_call_direct(
                                                    mir_func_id,
                                                    arg_regs,
                                                    mir_return_type.clone(),
                                                )?;

                                                // Auto-unbox if MIR wrapper returns Ptr(U8) but HIR expects primitive
                                                // (e.g., Channel<Int>.tryReceive() returns boxed int)
                                                debug!(
                                                "[MIR WRAPPER INSTANCE] call_result={:?}, mir_return_type={:?}, result_type={:?}",
                                                call_result, mir_return_type, result_type
                                            );
                                                if mir_func_name == "Channel_receive"
                                                    || mir_func_name == "Channel_tryReceive"
                                                {
                                                    return self.unbox_channel_return(
                                                        call_result,
                                                        &result_type,
                                                        mir_func_name == "Channel_tryReceive",
                                                    );
                                                }
                                                return self.maybe_unbox_for_extern_return(
                                                    call_result,
                                                    &mir_return_type,
                                                    &result_type,
                                                );
                                            } else {
                                                debug!(
                                                "[MIR WRAPPER INSTANCE] No signature found for {}, falling through",
                                                mir_func_name
                                            );
                                            }
                                        }
                                    }
                                }
                            } else {
                                debug!(
                                    "[MIR WRAPPER INSTANCE] Skipping instance-wrapper dispatch for synthetic class receiver"
                                );
                            }
                        } // end if receiver_is_class_type else block
                    }
                    // For static methods, check if it's a stdlib static method
                    if !*is_method || self.effective_static_call_args(args).len() != args.len() {
                        if let Some(sym_info) = self.symbol_table.get_symbol(*symbol) {
                            if let Some(method_name) = self.string_interner.get(sym_info.name) {
                                let static_args = self.effective_static_call_args(args);
                                debug!(
                                    "[STATIC-PATH] method_name='{}', symbol={:?}, has_qualified_name={}",
                                    method_name,
                                    symbol,
                                    sym_info.qualified_name.is_some()
                                );

                                // Try to get the qualified name to determine the class
                                if let Some(qual_name) = sym_info.qualified_name {
                                    if let Some(qual_name_str) = self.string_interner.get(qual_name)
                                    {
                                        debug!("[PRE-CHECK] Qualified name: '{}'", qual_name_str);

                                        // SPECIAL CASE: Thread/Channel/Mutex/Arc methods are MIR wrappers, not runtime_mapping
                                        // These are implemented in stdlib MIR (thread.rs, channel.rs, etc.)
                                        // Pattern: "rayzor.concurrent.Thread.spawn" -> "Thread_spawn"
                                        // NOTE: This only applies to rayzor.concurrent.*, NOT sys.thread.*
                                        let parts: Vec<&str> = qual_name_str.split('.').collect();
                                        if parts.len() >= 2 {
                                            let class_name = parts[parts.len() - 2];

                                            // Check if this is a rayzor.concurrent.* class (NOT sys.thread.*)
                                            // sys.thread.Thread uses runtime mapping directly, not MIR wrappers
                                            // Use dynamic check via stdlib_mapping instead of hardcoded list
                                            let is_rayzor_concurrent =
                                                qual_name_str.starts_with("rayzor.concurrent.");
                                            if is_rayzor_concurrent
                                                && self
                                                    .stdlib_mapping
                                                    .is_mir_wrapper_class(class_name)
                                            {
                                                // Use capitalized class names for rayzor.concurrent (Thread, Channel, etc.)
                                                let mir_func_name =
                                                    format!("{}_{}", class_name, method_name);
                                                debug!(
                                                    "[STDLIB MIR] Detected stdlib MIR function: {}, args.len()={}",
                                                    mir_func_name,
                                                    static_args.len()
                                                );
                                                for (idx, arg) in static_args.iter().enumerate() {
                                                    debug!(
                                                        "[STDLIB MIR PRE] arg[{}] kind={:?}, ty={:?}",
                                                        idx,
                                                        std::mem::discriminant(&arg.kind),
                                                        arg.ty
                                                    );
                                                }

                                                // WORKAROUND: static calls may carry a synthetic
                                                // class receiver argument. Prefer the mapping
                                                // signature arity to trim that argument.
                                                let mut actual_args = static_args;
                                                if let Some((expected_params, _)) = self
                                                    .get_stdlib_mir_wrapper_signature(
                                                        &mir_func_name,
                                                    )
                                                {
                                                    if actual_args.len() != expected_params.len()
                                                        && static_args.len()
                                                            == expected_params.len() + 1
                                                        && !static_args.is_empty()
                                                    {
                                                        debug!(
                                                            "[STDLIB MIR FIX] Arity-based static receiver trim for {}: {} -> {} args",
                                                            mir_func_name,
                                                            static_args.len(),
                                                            expected_params.len()
                                                        );
                                                        actual_args = &static_args[1..];
                                                    }
                                                }

                                                // Lower all arguments and collect their types
                                                let mut arg_regs = Vec::new();
                                                let mut param_types = Vec::new();
                                                for (idx, arg) in actual_args.iter().enumerate() {
                                                    debug!(
                                                        "[STDLIB MIR] arg[{}] ty={:?}",
                                                        idx, arg.ty
                                                    );
                                                    if let Some(reg) = self.lower_expression(arg) {
                                                        arg_regs.push(reg);
                                                        param_types.push(self.convert_type(arg.ty));
                                                    }
                                                }
                                                // Register forward reference - will be provided by merged stdlib module
                                                // We infer the signature from the call site arguments
                                                let mir_func_id = self
                                                    .register_stdlib_mir_forward_ref(
                                                        &mir_func_name,
                                                        param_types,
                                                        result_type.clone(),
                                                    );

                                                debug!(
                                                    "[STDLIB MIR] Registered forward ref to {} with ID {:?}",
                                                    mir_func_name, mir_func_id
                                                );

                                                // Generate the call
                                                let result = self.builder.build_call_direct(
                                                    mir_func_id,
                                                    arg_regs,
                                                    result_type,
                                                );
                                                debug!(
                                                    "[STDLIB MIR] Generated call, result: {:?}",
                                                    result
                                                );
                                                return result;
                                            }
                                        }

                                        // Check if this is a stdlib class method by looking at qualified name
                                        // e.g., "rayzor.concurrent.Thread.spawn" or "test.Thread.spawn"
                                        let lookup_result = self
                                            .get_static_stdlib_runtime_func_with_params(
                                                qual_name_str,
                                                method_name,
                                                static_args.len(),
                                            );
                                        debug!(
                                            "[PRE-CHECK] get_static_stdlib_runtime_func returned: {:?}",
                                            lookup_result
                                        );

                                        if let Some(runtime_func) = lookup_result {
                                            debug!(
                                                "[STATIC METHOD] Found stdlib runtime func: {}.{} -> {}, args.len()={}",
                                                qual_name_str,
                                                method_name,
                                                runtime_func,
                                                static_args.len()
                                            );

                                            if let Some(result) = self
                                                .try_lower_special_runtime_call(
                                                    &runtime_func,
                                                    static_args,
                                                    result_type.clone(),
                                                    expr.source_location.clone(),
                                                )
                                            {
                                                return result;
                                            }

                                            // Get the expected signature from our registered extern functions
                                            // This ensures we use the correct types (e.g., I64 for Std.random)
                                            let (expected_param_types, expected_return_type) = self
                                                .get_extern_function_signature(&runtime_func)
                                                .unwrap_or_else(|| {
                                                    // Fall back to inferred types from TAST
                                                    let string_ptr_ty =
                                                        IrType::Ptr(Box::new(IrType::String));
                                                    let param_types: Vec<IrType> = static_args
                                                        .iter()
                                                        .map(|a| {
                                                            let arg_ty = self.convert_type(a.ty);
                                                            if arg_ty == IrType::String {
                                                                string_ptr_ty.clone()
                                                            } else {
                                                                arg_ty
                                                            }
                                                        })
                                                        .collect();
                                                    (param_types, result_type.clone())
                                                });

                                            let runtime_call_args = if static_args.len()
                                                == expected_param_types.len() + 1
                                                && !static_args.is_empty()
                                            {
                                                &static_args[1..]
                                            } else {
                                                static_args
                                            };

                                            // Lower all arguments
                                            let arg_regs: Vec<_> = runtime_call_args
                                                .iter()
                                                .filter_map(|a| self.lower_expression(a))
                                                .collect();

                                            debug!(
                                                "[STATIC METHOD] Lowered {} args: {:?}",
                                                arg_regs.len(),
                                                arg_regs
                                            );

                                            // Cast/box arguments to expected types if needed
                                            let final_arg_regs: Vec<_> = arg_regs.iter().enumerate()
                                                .map(|(i, &reg)| {
                                                    if let (Some(expected_ty), Some(actual_ty)) = (
                                                        expected_param_types.get(i),
                                                        self.builder.get_register_type(reg)
                                                    ) {
                                                        // If types differ, insert a cast or box
                                                        if *expected_ty != actual_ty {
                                                            // When expected is Ptr(U8) (Dynamic/boxed), auto-box the value
                                                            let is_ptr_u8 = matches!(expected_ty, IrType::Ptr(inner) if matches!(inner.as_ref(), IrType::U8));
                                                            if is_ptr_u8 && i < runtime_call_args.len() {
                                                                debug!("[STATIC METHOD BOX] Attempting auto-box for arg {} with type {:?}", i, runtime_call_args[i].ty);
                                                                // Use box_value_for_dynamic to properly box based on HIR type
                                                                if let Some(boxed) = self.box_value_for_dynamic(reg, runtime_call_args[i].ty) {
                                                                    debug!("[STATIC METHOD BOX] Auto-boxed arg {} for Dynamic param", i);
                                                                    return boxed;
                                                                }
                                                                debug!("[STATIC METHOD BOX] box_value_for_dynamic returned None for arg {}", i);
                                                            }
                                                            debug!("[STATIC METHOD] Casting arg {} from {:?} to {:?}", i, actual_ty, expected_ty);
                                                            if let Some(casted) = self.builder.build_cast(reg, actual_ty.clone(), expected_ty.clone()) {
                                                                return casted;
                                                            }
                                                        }
                                                    }
                                                    reg
                                                })
                                                .collect();

                                            // Inject hidden enum type_id for enum helper runtime calls.
                                            let mut final_arg_regs = final_arg_regs;
                                            self.inject_hidden_enum_type_id_arg(
                                                &runtime_func,
                                                runtime_call_args,
                                                &mut final_arg_regs,
                                            );

                                            let runtime_func_id = self
                                                .get_or_register_extern_function(
                                                    &runtime_func,
                                                    expected_param_types,
                                                    expected_return_type.clone(),
                                                );

                                            debug!(
                                                "[STATIC METHOD] Registered runtime func {} with ID {:?}",
                                                runtime_func, runtime_func_id
                                            );

                                            // Generate the call to the runtime function
                                            let result = self.builder.build_call_direct(
                                                runtime_func_id,
                                                final_arg_regs,
                                                expected_return_type,
                                            );
                                            debug!(
                                                "[STATIC METHOD] Generated call, result: {:?}",
                                                result
                                            );
                                            return result;
                                        }
                                    }
                                }

                                // Fallback: still inside method_name scope.
                                // If qualified_name is not set (e.g., Reflect.compare from import files),
                                // try to find a matching static stdlib method by scanning all known classes.
                                // Only match static methods to avoid false positives.
                                // If qualified_name is available, prefer class-qualified lookup
                                // before doing a global static-name fallback.
                                let mut static_fallback = None;
                                if let Some(qual_name_str) = sym_info
                                    .qualified_name
                                    .and_then(|q| self.string_interner.get(q))
                                {
                                    let parts: Vec<&str> = qual_name_str.split('.').collect();
                                    if parts.len() >= 2 {
                                        let mut class_candidates: Vec<String> = Vec::new();
                                        // Fully-qualified class form used in runtime mapping
                                        class_candidates.push(parts[..parts.len() - 1].join("_"));
                                        // Simple class name fallback
                                        class_candidates.push(parts[parts.len() - 2].to_string());

                                        for class_name in class_candidates {
                                            if let Some(found) =
                                                self.stdlib_mapping.find_by_name_and_params(
                                                    &class_name,
                                                    method_name,
                                                    static_args.len(),
                                                )
                                            {
                                                static_fallback = Some(found);
                                                break;
                                            }
                                        }
                                    }
                                }

                                if static_fallback.is_none() {
                                    debug!(
                                        "[STATIC-FALLBACK] Trying global find_static_method_by_name_and_params('{}', {})...",
                                        method_name,
                                        static_args.len()
                                    );
                                    static_fallback =
                                        self.stdlib_mapping.find_static_method_by_name_and_params(
                                            method_name,
                                            static_args.len(),
                                        );
                                }

                                if let Some((_sig, mapping)) = static_fallback {
                                    let runtime_func_name = mapping.runtime_name.to_string();
                                    debug!(
                                        "[STATIC FALLBACK] Found static {}.{} -> {} via name scan",
                                        _sig.class, method_name, runtime_func_name
                                    );

                                    if let Some(result) = self.try_lower_special_runtime_call(
                                        &runtime_func_name,
                                        static_args,
                                        result_type.clone(),
                                        expr.source_location.clone(),
                                    ) {
                                        return result;
                                    }

                                    // Lower all arguments first
                                    let mut arg_regs: Vec<IrId> = Vec::new();
                                    let mut arg_types: Vec<IrType> = Vec::new();
                                    for arg in static_args.iter() {
                                        if let Some(reg) = self.lower_expression(arg) {
                                            arg_regs.push(reg);
                                            arg_types.push(self.convert_type(arg.ty));
                                        }
                                    }

                                    // Special case: Reflect.compare → haxe_reflect_compare_typed
                                    // Same logic as the qualified-name path: detect argument type and
                                    // append a type_tag parameter to avoid boxing.
                                    if runtime_func_name == "haxe_reflect_compare" {
                                        let mut known_type_tag: Option<i32> = None;
                                        let mut type_param_name: Option<String> = None;
                                        let mut use_typed = false;

                                        if let Some(first_arg) = static_args.first() {
                                            let type_table = self.type_table;
                                            if let Some(ti) = type_table.get(first_arg.ty) {
                                                use crate::tast::core::TypeKind;
                                                match &ti.kind {
                                                    TypeKind::TypeParameter {
                                                        symbol_id, ..
                                                    } => {
                                                        use_typed = true;
                                                        if let Some(sym) =
                                                            self.symbol_table.get_symbol(*symbol_id)
                                                        {
                                                            if let Some(name_str) =
                                                                self.string_interner.get(sym.name)
                                                            {
                                                                type_param_name =
                                                                    Some(name_str.to_string());
                                                            }
                                                        }
                                                    }
                                                    TypeKind::Int => {
                                                        use_typed = true;
                                                        known_type_tag = Some(1);
                                                    }
                                                    TypeKind::Float => {
                                                        use_typed = true;
                                                        known_type_tag = Some(4);
                                                    }
                                                    TypeKind::Bool => {
                                                        use_typed = true;
                                                        known_type_tag = Some(2);
                                                    }
                                                    TypeKind::String => {
                                                        use_typed = true;
                                                        known_type_tag = Some(5);
                                                    }
                                                    _ => {}
                                                }
                                            }
                                        }

                                        if use_typed {
                                            // Cast value args to I64 — haxe_reflect_compare_typed
                                            // takes type-erased i64 values, not typed structs
                                            for i in 0..arg_regs.len().min(2) {
                                                let reg_ty = self
                                                    .builder
                                                    .get_register_type(arg_regs[i])
                                                    .unwrap_or(IrType::I64);
                                                if reg_ty != IrType::I64 {
                                                    if let Some(cast) = self.builder.build_cast(
                                                        arg_regs[i],
                                                        reg_ty,
                                                        IrType::I64,
                                                    ) {
                                                        arg_regs[i] = cast;
                                                    }
                                                }
                                                arg_types[i] = IrType::I64;
                                            }

                                            let tag_reg = if let Some(tp_name) = type_param_name {
                                                let tag =
                                                    self.builder.build_const(IrValue::I32(0))?;
                                                if let Some(func) =
                                                    self.builder.current_function_mut()
                                                {
                                                    func.type_param_tag_fixups.push((tag, tp_name));
                                                }
                                                tag
                                            } else {
                                                self.builder.build_const(IrValue::I32(
                                                    known_type_tag.unwrap_or(1),
                                                ))?
                                            };
                                            arg_regs.push(tag_reg);
                                            arg_types.push(IrType::I32);

                                            let extern_func_id = self
                                                .get_or_register_extern_function(
                                                    "haxe_reflect_compare_typed",
                                                    arg_types,
                                                    result_type.clone(),
                                                );
                                            return self.builder.build_call_direct(
                                                extern_func_id,
                                                arg_regs,
                                                result_type,
                                            );
                                        }
                                    }

                                    // General case: call the runtime function directly
                                    let (expected_param_types, expected_return_type) = self
                                        .get_extern_function_signature(&runtime_func_name)
                                        .unwrap_or_else(|| (arg_types, result_type.clone()));

                                    let final_arg_regs: Vec<_> = arg_regs
                                        .iter()
                                        .enumerate()
                                        .map(|(i, &reg)| {
                                            if let (Some(expected_ty), Some(actual_ty)) = (
                                                expected_param_types.get(i),
                                                self.builder.get_register_type(reg),
                                            ) {
                                                if *expected_ty != actual_ty {
                                                    if let Some(casted) = self.builder.build_cast(
                                                        reg,
                                                        actual_ty.clone(),
                                                        expected_ty.clone(),
                                                    ) {
                                                        return casted;
                                                    }
                                                }
                                            }
                                            reg
                                        })
                                        .collect();

                                    // Inject hidden enum type_id for enum helper runtime calls.
                                    let mut final_arg_regs = final_arg_regs;
                                    self.inject_hidden_enum_type_id_arg(
                                        &runtime_func_name,
                                        args,
                                        &mut final_arg_regs,
                                    );

                                    let runtime_func_id = self.get_or_register_extern_function(
                                        &runtime_func_name,
                                        expected_param_types,
                                        expected_return_type.clone(),
                                    );

                                    return self.builder.build_call_direct(
                                        runtime_func_id,
                                        final_arg_regs,
                                        expected_return_type,
                                    );
                                }
                            } // end of if let Some(method_name)
                        }
                    }

                    // Check if this symbol is a function (local or external)
                    // First try direct symbol ID lookup
                    let method_name_interned =
                        self.symbol_table.get_symbol(*symbol).map(|s| s.name);
                    let mut func_id_opt = self.get_function_id(symbol);

                    // Intercept @:shader wgsl() calls — the stub function has
                    // an empty body; redirect to the transpiler output.
                    if func_id_opt.is_some() {
                        let callee_is_wgsl = self
                            .symbol_table
                            .get_symbol(*symbol)
                            .and_then(|s| self.string_interner.get(s.name))
                            .map(|n| n == "wgsl")
                            .unwrap_or(false);
                        if callee_is_wgsl {
                            // Find the @:shader class in current_hir_types
                            for (_tid, decl) in self.current_hir_types.iter() {
                                if let crate::ir::hir::HirTypeDecl::Class(c) = decl {
                                    let is_shader = self
                                        .symbol_table
                                        .get_symbol(c.symbol_id)
                                        .map(|s| s.flags.is_shader())
                                        .unwrap_or(false);
                                    if is_shader {
                                        let type_table = self.type_table;
                                        match crate::codegen::wgsl_transpiler::transpile_shader_from_hir(
                                            c, self.symbol_table, type_table, self.string_interner, self.current_hir_types,
                                        ) {
                                            Ok(wgsl) => return self.builder.build_const(IrValue::String(wgsl)),
                                            Err(e) => return self.builder.build_const(IrValue::String(format!("/* WGSL error: {} */", e))),
                                        }
                                    }
                                }
                            }
                        }
                    }

                    let has_synthetic_static_receiver =
                        *is_method && self.effective_static_call_args(args).len() != args.len();

                    if func_id_opt.is_none()
                        && *is_method
                        && !has_synthetic_static_receiver
                        && !args.is_empty()
                    {
                        if let Some(method_name) = method_name_interned {
                            func_id_opt = self.resolve_method_function_id(args[0].ty, method_name);
                        }
                    }

                    // If not found by symbol ID, try lookup by qualified name
                    // This handles cross-module calls where symbol IDs differ between modules,
                    // and also intra-module static method calls where the call site symbol
                    // differs from the method definition symbol (e.g., Body.Sun() in nbody)
                    if func_id_opt.is_none() {
                        if let Some(sym_info) = self.symbol_table.get_symbol(*symbol) {
                            if let Some(qual_name) = sym_info.qualified_name {
                                if let Some(qual_name_str) = self.string_interner.get(qual_name) {
                                    // Search local function_map by qualified name first
                                    for (local_sym, &local_func_id) in &self.function_map {
                                        if let Some(local_sym_info) =
                                            self.symbol_table.get_symbol(*local_sym)
                                        {
                                            if let Some(local_qual) = local_sym_info.qualified_name
                                            {
                                                if let Some(local_qual_str) =
                                                    self.string_interner.get(local_qual)
                                                {
                                                    if local_qual_str == qual_name_str {
                                                        debug!(
                                                            "[QUAL-NAME LOCAL] Found function by qualified name '{}': symbol {:?} -> func_id={:?}",
                                                            qual_name_str, local_sym, local_func_id
                                                        );
                                                        func_id_opt = Some(local_func_id);
                                                        break;
                                                    }
                                                }
                                            }
                                        }
                                    }

                                    // If not found locally, search external_function_map
                                    if func_id_opt.is_none() {
                                        for (ext_sym, &ext_func_id) in &self.external_function_map {
                                            if let Some(ext_sym_info) =
                                                self.symbol_table.get_symbol(*ext_sym)
                                            {
                                                if let Some(ext_qual) = ext_sym_info.qualified_name
                                                {
                                                    if let Some(ext_qual_str) =
                                                        self.string_interner.get(ext_qual)
                                                    {
                                                        if ext_qual_str == qual_name_str {
                                                            debug!(
                                                                "[CROSS-MODULE] Found function by qualified name '{}': symbol {:?} -> func_id={:?}",
                                                                qual_name_str, ext_sym, ext_func_id
                                                            );
                                                            func_id_opt = Some(ext_func_id);
                                                            break;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            } else {
                                // Call-site symbol has no qualified name (common for cross-class
                                // static method calls in multi-class files, e.g., Body.Jupiter()).
                                // Fall back to searching function_map by bare name.
                                // Never the currently-compiling function (a lone
                                // same-named local — e.g. `.forward` inside a
                                // forward — would self-bind into infinite
                                // recursion), and for method calls never a
                                // candidate whose class positively differs from
                                // the receiver's.
                                let recv_class_bare: Option<String> = if *is_method
                                    && !args.is_empty()
                                {
                                    let type_table = self.type_table;
                                    type_table
                                        .get(args[0].ty)
                                        .and_then(|ti| match &ti.kind {
                                            TypeKind::Class { symbol_id, .. } => Some(*symbol_id),
                                            _ => None,
                                        })
                                        .and_then(|sid| self.symbol_table.get_symbol(sid))
                                        .and_then(|s| self.string_interner.get(s.name))
                                        .map(|s| s.to_string())
                                } else {
                                    None
                                };
                                if let Some(func_name) = self.string_interner.get(sym_info.name) {
                                    for (func_sym, &func_id) in &self.function_map {
                                        if *func_sym == *symbol {
                                            continue;
                                        }
                                        if Some(func_id) == self.builder.current_function {
                                            continue;
                                        }
                                        if let Some(func_sym_info) =
                                            self.symbol_table.get_symbol(*func_sym)
                                        {
                                            if let (Some(rb), Some(qn)) = (
                                                recv_class_bare.as_deref(),
                                                func_sym_info
                                                    .qualified_name
                                                    .and_then(|q| self.string_interner.get(q)),
                                            ) {
                                                let parts: Vec<&str> = qn.split('.').collect();
                                                if parts.len() >= 2 && parts[parts.len() - 2] != rb
                                                {
                                                    continue;
                                                }
                                            }
                                            if let Some(fm_name) =
                                                self.string_interner.get(func_sym_info.name)
                                            {
                                                if fm_name == func_name {
                                                    debug!(
                                                        "[BARE-NAME LOCAL] Found function by bare name '{}': sym {:?} -> func_id={:?}",
                                                        func_name, func_sym, func_id
                                                    );
                                                    func_id_opt = Some(func_id);
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                    // NOTE: Removed bare-name search in external_function_map.
                                    // Bare-name matching across modules causes false positives
                                    // (e.g., ListNode.create() -> rayzor_tcc_create).
                                    // Cross-module calls must use qualified name matching.
                                }
                            }
                        }
                    }

                    // If still not found, try lookup by function name in function_map
                    // This handles cases where method calls use different symbol IDs than the definition
                    // (e.g., chained method calls like z.mul(z).add(c) where add has a different symbol)
                    //
                    // IMPORTANT: When matching by bare name, also verify the function belongs to
                    // the receiver's class (via qualified name). Without this, common names like
                    // "get", "set", "toString" could match wrong stdlib functions.
                    if func_id_opt.is_none() && *is_method && !has_synthetic_static_receiver {
                        if let Some(sym_info) = self.symbol_table.get_symbol(*symbol) {
                            if let Some(method_name) = self.string_interner.get(sym_info.name) {
                                debug!(
                                    "[NAME-FALLBACK] Searching for method '{}' sym={:?}",
                                    method_name, symbol
                                );
                                // Get receiver's class name for disambiguation
                                let receiver_class_name = if !args.is_empty() {
                                    let type_table = self.type_table;
                                    let class_sym =
                                        type_table.get(args[0].ty).and_then(|ti| match &ti.kind {
                                            TypeKind::Class { symbol_id, .. } => Some(*symbol_id),
                                            TypeKind::GenericInstance { base_type, .. } => {
                                                type_table.get(*base_type).and_then(|bt| match &bt
                                                    .kind
                                                {
                                                    TypeKind::Class { symbol_id, .. } => {
                                                        Some(*symbol_id)
                                                    }
                                                    _ => None,
                                                })
                                            }
                                            _ => None,
                                        });
                                    let name = class_sym.and_then(|sid| {
                                        self.symbol_table
                                            .get_symbol(sid)
                                            .and_then(|s| self.string_interner.get(s.name))
                                            .map(|s| s.to_string())
                                    });
                                    name
                                } else {
                                    None
                                };

                                debug!(
                                    "[NAME-FALLBACK] receiver_class_name={:?}",
                                    receiver_class_name
                                );
                                // Search function_map by name, preferring qualified name match
                                // Pass 1: strict class name matching
                                for (func_sym, &func_id) in &self.function_map {
                                    if let Some(func_sym_info) =
                                        self.symbol_table.get_symbol(*func_sym)
                                    {
                                        if let Some(func_name) =
                                            self.string_interner.get(func_sym_info.name)
                                        {
                                            if func_name == method_name {
                                                // If we know the receiver class, verify via qualified name
                                                if let Some(ref class_name) = receiver_class_name {
                                                    let qual_match = func_sym_info
                                                        .qualified_name
                                                        .and_then(|qn| self.string_interner.get(qn))
                                                        .map(|qn| qn.contains(class_name.as_str()))
                                                        .unwrap_or(false);
                                                    if !qual_match {
                                                        continue; // Skip — wrong class
                                                    }
                                                }
                                                debug!(
                                                    "[NAME FALLBACK] Found method '{}' by name: {:?} -> {:?}",
                                                    method_name, func_sym, func_id
                                                );
                                                func_id_opt = Some(func_id);
                                                break;
                                            }
                                        }
                                    }
                                }

                                // Pass 2: Search external_function_name_map by qualified name
                                // Try both "ClassName.method" and "pkg.ClassName.method" patterns
                                if func_id_opt.is_none() {
                                    if let Some(ref class_name) = receiver_class_name {
                                        let suffix = format!("{}.{}", class_name, method_name);
                                        // Direct match first
                                        if let Some(&fid) =
                                            self.external_function_name_map.get(&suffix)
                                        {
                                            func_id_opt = Some(fid);
                                        } else {
                                            // Suffix match: find "pkg.ClassName.method"
                                            for (name, &fid) in &self.external_function_name_map {
                                                if name.ends_with(&suffix) {
                                                    func_id_opt = Some(fid);
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Guard against short-name symbol collisions for static calls:
                    // if the resolved function signature arity does not match this call site,
                    // drop it so static runtime fallback can re-resolve by (name, arity).
                    if func_id_opt.is_some() && (!*is_method || has_synthetic_static_receiver) {
                        let static_arg_count = self.effective_static_call_args(args).len();
                        if let Some(func_id) = func_id_opt {
                            let expected_params_opt = self
                                .builder
                                .module
                                .functions
                                .get(&func_id)
                                .map(|func| func.signature.parameters.len())
                                .or_else(|| {
                                    self.symbol_table.get_symbol(*symbol).and_then(|sym| {
                                        let type_table = self.type_table;
                                        type_table.get(sym.type_id).and_then(|ti| {
                                            if let TypeKind::Function { params, .. } = &ti.kind {
                                                Some(params.len())
                                            } else {
                                                None
                                            }
                                        })
                                    })
                                });

                            if let Some(expected_params) = expected_params_opt {
                                if expected_params != static_arg_count {
                                    debug!(
                                        "[STATIC ARITY MISMATCH] symbol={:?} resolved func_id={:?} expected_params={} call_args={} -> fallback",
                                        symbol, func_id, expected_params, static_arg_count
                                    );
                                    func_id_opt = None;
                                }
                            }
                        }
                    }

                    // Fallback for extern class static methods (e.g. NativeStackTrace.exceptionStack()).
                    // StaticFieldAccess becomes Variable in HIR, bypassing the Field-callee stdlib
                    // dispatch. Prefer symbol qualified_name, then class-less name fallback.
                    if func_id_opt.is_none() {
                        if let Some(sym_info) = self.symbol_table.get_symbol(*symbol) {
                            if let Some(method_name) = self.string_interner.get(sym_info.name) {
                                let static_args = self.effective_static_call_args(args);
                                let mut runtime_func_name: Option<String> = None;

                                // Prefer class-qualified dispatch if present on the symbol.
                                if let Some(qual_name_str) = sym_info
                                    .qualified_name
                                    .and_then(|qn| self.string_interner.get(qn))
                                {
                                    if let Some(found) = self
                                        .get_static_stdlib_runtime_func_with_params(
                                            qual_name_str,
                                            method_name,
                                            static_args.len(),
                                        )
                                    {
                                        runtime_func_name = Some(found.to_string());
                                    }
                                }

                                // True last resort: class-less static method lookup.
                                if runtime_func_name.is_none() {
                                    if let Some((_sig, mapping)) =
                                        self.stdlib_mapping.find_static_method_by_name_and_params(
                                            method_name,
                                            static_args.len(),
                                        )
                                    {
                                        runtime_func_name = Some(mapping.runtime_name.to_string());
                                    }
                                }

                                if let Some(runtime_func_name) = runtime_func_name {
                                    let mut arg_regs: Vec<IrId> = Vec::new();
                                    let mut arg_types: Vec<IrType> = Vec::new();
                                    for arg in static_args.iter() {
                                        if let Some(reg) = self.lower_expression(arg) {
                                            arg_regs.push(reg);
                                            arg_types.push(self.convert_type(arg.ty));
                                        }
                                    }
                                    let (expected_param_types, expected_return_type) = self
                                        .get_extern_function_signature(&runtime_func_name)
                                        .unwrap_or_else(|| (arg_types, result_type.clone()));
                                    let runtime_func_id = self.get_or_register_extern_function(
                                        &runtime_func_name,
                                        expected_param_types,
                                        expected_return_type.clone(),
                                    );
                                    return self.builder.build_call_direct(
                                        runtime_func_id,
                                        arg_regs,
                                        expected_return_type,
                                    );
                                }
                            }
                        }
                    }

                    if let Some(func_id) = func_id_opt {
                        let sym_name = self
                            .symbol_table
                            .get_symbol(*symbol)
                            .and_then(|s| self.string_interner.get(s.name))
                            .unwrap_or("<unknown>");
                        self.builder.call_label = Some(format!("FUNC_MAP:{}", sym_name));
                        let qual_name = self
                            .symbol_table
                            .get_symbol(*symbol)
                            .and_then(|s| s.qualified_name)
                            .and_then(|qn| self.string_interner.get(qn))
                            .unwrap_or("<none>");
                        let is_external = self.external_function_map.contains_key(symbol);

                        debug!(
                            "[FUNCTION_MAP LOOKUP] Found symbol {:?} '{}' (qual: '{}') -> func_id={:?}, is_method={}, external={}",
                            symbol, sym_name, qual_name, func_id, is_method, is_external
                        );

                        // IMPORTANT: Use the function's actual return type, not expr.ty
                        // Check both functions (local) and extern_functions (forward refs to stdlib)
                        let actual_return_type = if let Some(func) =
                            self.builder.module.functions.get(&func_id)
                        {
                            debug!(
                                "[FUNCTION_MAP] Using actual return type {:?} for function {:?}",
                                func.signature.return_type, func.name
                            );
                            func.signature.return_type.clone()
                        } else if let Some(func) =
                            self.builder.module.extern_functions.get(&func_id)
                        {
                            debug!(
                                "[FUNCTION_MAP] Using extern_functions return type {:?} for {:?}",
                                func.signature.return_type, func_id
                            );
                            func.signature.return_type.clone()
                        } else {
                            // Function not in module yet (probably forward ref to stdlib MIR wrapper)
                            // Try to look up the correct signature by function name
                            debug!(
                                "[FUNCTION_MAP] Function {:?} not found in module, checking stdlib signatures",
                                func_id
                            );
                            if let Some((_params, ret_ty)) =
                                self.get_stdlib_mir_wrapper_signature(&sym_name)
                            {
                                debug!(
                                    "[FUNCTION_MAP] Found stdlib signature for '{}': returns {:?}",
                                    sym_name, ret_ty
                                );
                                ret_ty
                            } else {
                                debug!(
                                    "[FUNCTION_MAP] No stdlib signature found, using expr return type {:?}",
                                    result_type
                                );
                                result_type.clone()
                            }
                        };
                        let function_param_count = self
                            .builder
                            .module
                            .functions
                            .get(&func_id)
                            .map(|f| f.signature.parameters.len())
                            .or_else(|| {
                                self.builder
                                    .module
                                    .extern_functions
                                    .get(&func_id)
                                    .map(|f| f.signature.parameters.len())
                            });
                        let has_arity_static_receiver = *is_method
                            && function_param_count
                                .map(|param_count| args.len() == param_count + 1)
                                .unwrap_or(false);
                        let treat_as_static_call =
                            has_synthetic_static_receiver || has_arity_static_receiver;
                        if has_arity_static_receiver {
                            debug!(
                                "[STATIC-RECEIVER ARITY] Treating method call as static: symbol={:?}, func_id={:?}, args={}, params={:?}",
                                symbol,
                                func_id,
                                args.len(),
                                function_param_count
                            );
                        }

                        // Handle method calls where the object is passed as first argument
                        if *is_method && !treat_as_static_call {
                            // For method calls, args already includes the object as first arg.
                            // Track non-receiver args as temps ONLY if the callee is user-defined.
                            // Stdlib/runtime methods (e.g., Array.push) may store arguments.
                            let callee_is_user_defined = self
                                .builder
                                .module
                                .functions
                                .get(&func_id)
                                .map(|f| f.kind == crate::ir::functions::FunctionKind::UserDefined)
                                .unwrap_or(false);

                            let mut arg_regs = Vec::new();

                            // Check if receiver (args[0]) is Dynamic-typed — needs unboxing
                            let receiver_is_dynamic = if !args.is_empty() {
                                let type_table = self.type_table;
                                type_table
                                    .get(args[0].ty)
                                    .map(|t| matches!(t.kind, TypeKind::Dynamic))
                                    .unwrap_or(false)
                            } else {
                                false
                            };

                            for (i, arg) in args.iter().enumerate() {
                                if let Some(reg) = self.lower_expression(arg) {
                                    // Materialize anon-backed variables at call boundary (skip receiver)
                                    // For method calls, args[0] is receiver, args[1..] are params
                                    // HIR param_types don't include `this`, so param_index = i - 1
                                    let reg = if i > 0 {
                                        self.maybe_materialize_for_call(
                                            arg,
                                            reg,
                                            Some(func_id),
                                            i - 1,
                                        )
                                    } else {
                                        reg
                                    };
                                    if i == 0 && receiver_is_dynamic && callee_is_user_defined {
                                        // Dynamic receiver: unbox DynamicValue* to get raw object pointer
                                        let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                                        let unbox_func_id = self.get_or_register_extern_function(
                                            "haxe_unbox_reference_ptr",
                                            vec![ptr_u8.clone()],
                                            ptr_u8.clone(),
                                        );
                                        if let Some(unboxed) = self.builder.build_call_direct(
                                            unbox_func_id,
                                            vec![reg],
                                            ptr_u8,
                                        ) {
                                            arg_regs.push(unboxed);
                                        } else {
                                            arg_regs.push(reg);
                                        }
                                    } else {
                                        // @:derive(Copy): copy variable args at call boundary
                                        let reg = if i > 0 {
                                            if let HirExprKind::Variable { .. } = &arg.kind {
                                                if let Some(class_sym) =
                                                    self.get_copy_class_symbol(arg.ty)
                                                {
                                                    self.emit_shallow_copy(reg, class_sym)
                                                        .unwrap_or(reg)
                                                } else {
                                                    reg
                                                }
                                            } else {
                                                reg
                                            }
                                        } else {
                                            reg
                                        };
                                        if i > 0 && callee_is_user_defined {
                                            let is_heap_intermediate = matches!(
                                                &arg.kind,
                                                HirExprKind::New { .. } | HirExprKind::Call { .. }
                                            ) && self
                                                .get_drop_behavior(arg.ty)
                                                == DropBehavior::AutoDrop;
                                            if is_heap_intermediate {
                                                self.temp_heap_values.push(reg);
                                            }
                                        }
                                        arg_regs.push(reg);
                                    }
                                }
                            }

                            // Coerce Int→Float at cross-module call boundaries
                            self.coerce_args_for_cross_module_call(func_id, &mut arg_regs, true);
                            // Fill in default values for any missing optional parameters
                            self.fill_default_args(func_id, &mut arg_regs, true);

                            // Extract type_args for generic method calls.
                            // Priority: 1) HIR type_args, 2) class type_args, 3) infer from args
                            let ir_type_args = if !converted_hir_type_args.is_empty() {
                                // Method-level type args from HIR (e.g., explicitly specified)
                                converted_hir_type_args.clone()
                            } else if !args.is_empty() {
                                let receiver_type = args[0].ty;
                                let class_type_args = {
                                    let type_table = self.type_table;
                                    if let Some(receiver_info) = type_table.get(receiver_type) {
                                        if let crate::tast::TypeKind::Class { type_args, .. } =
                                            &receiver_info.kind
                                        {
                                            type_args.clone()
                                        } else {
                                            Vec::new()
                                        }
                                    } else {
                                        Vec::new()
                                    }
                                };
                                if !class_type_args.is_empty() {
                                    class_type_args
                                        .iter()
                                        .map(|&ty_id| self.convert_type(ty_id))
                                        .collect::<Vec<_>>()
                                } else {
                                    // Infer method's own type params from argument types
                                    // (e.g., add<T>(x:T) called with String → T=String)
                                    if let Some(func) = self.builder.module.functions.get(&func_id)
                                    {
                                        if !func.signature.type_params.is_empty() {
                                            let mut inferred: Vec<IrType> = Vec::new();
                                            for type_param in &func.signature.type_params {
                                                let mut found = false;
                                                for (i, sig_param) in
                                                    func.signature.parameters.iter().enumerate()
                                                {
                                                    if let IrType::TypeVar(ref name) = sig_param.ty
                                                    {
                                                        if name == &type_param.name
                                                            && i < args.len()
                                                        {
                                                            let arg_type =
                                                                self.convert_type(args[i].ty);
                                                            inferred.push(arg_type);
                                                            found = true;
                                                            break;
                                                        }
                                                    }
                                                }
                                                if !found
                                                    && func.signature.type_params.len() == 1
                                                    && args.len() > 1
                                                {
                                                    // Single type param, infer from first non-this arg
                                                    let arg_type = self.convert_type(args[1].ty);
                                                    inferred.push(arg_type);
                                                }
                                            }
                                            inferred
                                        } else {
                                            Vec::new()
                                        }
                                    } else {
                                        Vec::new()
                                    }
                                }
                            } else {
                                Vec::new()
                            };

                            debug!(
                                "[FUNCTION_MAP] Method call lowered {} args: {:?}, type_args: {:?}",
                                arg_regs.len(),
                                arg_regs,
                                ir_type_args
                            );

                            // Virtual dispatch: if this method is in a class hierarchy
                            // with overrides, dispatch through the vtable.
                            if let Some(&(slot_index, _)) = self.virtual_dispatch_info.get(symbol) {
                                if !arg_regs.is_empty() {
                                    let receiver_reg = arg_regs[0];
                                    let lookup_fn = self.get_or_register_extern_function(
                                        "haxe_vtable_lookup",
                                        vec![IrType::Ptr(Box::new(IrType::U8)), IrType::I32],
                                        IrType::I64,
                                    );
                                    let slot_reg =
                                        self.builder.build_const(IrValue::I32(slot_index as i32));
                                    if let Some(slot_r) = slot_reg {
                                        if let Some(closure_ptr) = self.builder.build_call_direct(
                                            lookup_fn,
                                            vec![receiver_reg, slot_r],
                                            IrType::I64,
                                        ) {
                                            let mut param_types =
                                                vec![IrType::Ptr(Box::new(IrType::Void))];
                                            for arg in args.iter().skip(1) {
                                                param_types.push(self.convert_type(arg.ty));
                                            }
                                            let return_type = Box::new(actual_return_type.clone());
                                            let func_signature = IrType::Function {
                                                params: param_types,
                                                return_type,
                                                varargs: false,
                                            };
                                            return self.builder.build_call_indirect(
                                                closure_ptr,
                                                arg_regs,
                                                func_signature,
                                            );
                                        }
                                    }
                                }
                            }

                            let result = if ir_type_args.is_empty() {
                                self.builder.build_call_direct(
                                    func_id,
                                    arg_regs,
                                    actual_return_type.clone(),
                                )
                            } else {
                                self.builder.build_call_direct_with_type_args(
                                    func_id,
                                    arg_regs,
                                    actual_return_type.clone(),
                                    ir_type_args,
                                )
                            };
                            // Set class hint on result for cross-module method dispatch
                            if let Some(reg) = result {
                                self.set_class_hint_for_return(reg, expr.ty);
                            }
                            debug!("[FUNCTION_MAP] Result: {:?}", result);

                            // Type erasure coercion for generic method returns:
                            // The function returns I64 (type-erased), but the concrete
                            // return type may differ. Only apply to methods on generic classes —
                            // non-generic classes (Thread, Bytes, etc.) must NOT be coerced.
                            let receiver_is_generic = if !args.is_empty() {
                                let type_table = self.type_table;
                                type_table
                                    .get(args[0].ty)
                                    .map(|ti| match &ti.kind {
                                        TypeKind::Class { type_args, .. } => !type_args.is_empty(),
                                        TypeKind::GenericInstance { .. } => true,
                                        TypeKind::TypeParameter { .. } => true,
                                        _ => false,
                                    })
                                    .unwrap_or(false)
                            } else {
                                false
                            };

                            if receiver_is_generic {
                                if let Some(call_result) = result {
                                    if actual_return_type == IrType::I64 {
                                        let expected_ir_type = self.convert_type(expr.ty);
                                        if expected_ir_type != IrType::I64 {
                                            // Path 1: AST resolved type (e.g., Box<Int> → Ptr)
                                            return self.coerce_from_i64(call_result, expr.ty);
                                        }
                                        // Path 2: expr.ty is still TypeParameter → resolve via receiver's type_args
                                        if !args.is_empty() {
                                            if let Some(concrete_ty_id) = self
                                                .resolve_type_param_from_receiver(
                                                    expr.ty, args[0].ty,
                                                )
                                            {
                                                let concrete_ir_type =
                                                    self.convert_type(concrete_ty_id);
                                                if concrete_ir_type != IrType::I64 {
                                                    return self.coerce_from_i64(
                                                        call_result,
                                                        concrete_ty_id,
                                                    );
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            return result;
                        } else {
                            // Direct function call (static method or free function)
                            // Track heap-allocated intermediates passed as arguments,
                            // but ONLY if the callee is a user-defined function.
                            // Stdlib/runtime functions (MirWrapper, ExternC) may store
                            // arguments (e.g., Array.push), so freeing would cause
                            // dangling pointers.
                            let callee_is_user_defined = self
                                .builder
                                .module
                                .functions
                                .get(&func_id)
                                .map(|f| f.kind == crate::ir::functions::FunctionKind::UserDefined)
                                .unwrap_or(false);

                            let mut call_args = self.effective_static_call_args(args);
                            if call_args.len() == args.len() {
                                if let Some(param_count) = function_param_count {
                                    if args.len() == param_count + 1 && !args.is_empty() {
                                        call_args = &args[1..];
                                    }
                                }
                            }
                            let mut arg_regs = Vec::new();
                            for (param_idx, arg) in call_args.iter().enumerate() {
                                if let Some(reg) = self.lower_expression(arg) {
                                    // Materialize anon-backed variables at call boundary
                                    let reg = self.maybe_materialize_for_call(
                                        arg,
                                        reg,
                                        Some(func_id),
                                        param_idx,
                                    );
                                    // @:derive(Copy): copy variable args at call boundary
                                    let reg = if let HirExprKind::Variable { .. } = &arg.kind {
                                        if let Some(class_sym) = self.get_copy_class_symbol(arg.ty)
                                        {
                                            self.emit_shallow_copy(reg, class_sym).unwrap_or(reg)
                                        } else {
                                            reg
                                        }
                                    } else {
                                        reg
                                    };
                                    if callee_is_user_defined {
                                        let is_heap_intermediate = matches!(
                                            &arg.kind,
                                            HirExprKind::New { .. } | HirExprKind::Call { .. }
                                        ) && self
                                            .get_drop_behavior(arg.ty)
                                            == DropBehavior::AutoDrop
                                            && !self.interface_wrapped_args.contains(&reg);
                                        if is_heap_intermediate {
                                            self.temp_heap_values.push(reg);
                                        }
                                    }
                                    arg_regs.push(reg);
                                }
                            }

                            self.coerce_args_for_cross_module_call(func_id, &mut arg_regs, false);
                            // Fill in default values for any missing optional parameters
                            self.fill_default_args(func_id, &mut arg_regs, false);

                            // Last-chance parity guard for static-call symbol collisions:
                            // if the resolved function arity still does not match the call site,
                            // re-resolve by (method_name, arity) through stdlib static mapping.
                            if let Some(expected_params) = self
                                .builder
                                .module
                                .functions
                                .get(&func_id)
                                .map(|f| f.signature.parameters.len())
                            {
                                if expected_params != arg_regs.len() {
                                    if let Some(sym_info) = self.symbol_table.get_symbol(*symbol) {
                                        if let Some(method_name) =
                                            self.string_interner.get(sym_info.name)
                                        {
                                            if let Some((_sig, mapping)) = self
                                                .stdlib_mapping
                                                .find_static_method_by_name_and_params(
                                                    method_name,
                                                    call_args.len(),
                                                )
                                            {
                                                let runtime_func_name =
                                                    mapping.runtime_name.to_string();
                                                let mut fallback_arg_regs: Vec<IrId> = Vec::new();
                                                let mut fallback_arg_types: Vec<IrType> =
                                                    Vec::new();
                                                for arg in call_args.iter() {
                                                    if let Some(reg) = self.lower_expression(arg) {
                                                        fallback_arg_regs.push(reg);
                                                        fallback_arg_types
                                                            .push(self.convert_type(arg.ty));
                                                    }
                                                }
                                                let (expected_param_types, expected_return_type) =
                                                    self.get_extern_function_signature(
                                                        &runtime_func_name,
                                                    )
                                                    .unwrap_or_else(|| {
                                                        (
                                                            fallback_arg_types,
                                                            actual_return_type.clone(),
                                                        )
                                                    });
                                                let runtime_func_id = self
                                                    .get_or_register_extern_function(
                                                        &runtime_func_name,
                                                        expected_param_types,
                                                        expected_return_type.clone(),
                                                    );
                                                return self.builder.build_call_direct(
                                                    runtime_func_id,
                                                    fallback_arg_regs,
                                                    expected_return_type,
                                                );
                                            }
                                        }
                                    }
                                }
                            }

                            // Auto-box arguments when expected type is Ptr(U8) but actual is primitive
                            // This handles cases like Type.enumIndex(Color.Red) where the enum discriminant
                            // (raw i64) needs to be boxed as DynamicValue* for the runtime function.
                            if let Some(func) = self.builder.module.functions.get(&func_id) {
                                let expected_types: Vec<IrType> = func
                                    .signature
                                    .parameters
                                    .iter()
                                    .map(|p| p.ty.clone())
                                    .collect();
                                for (i, expected_ty) in expected_types.iter().enumerate() {
                                    if i < arg_regs.len() {
                                        let is_ptr_u8 = matches!(expected_ty, IrType::Ptr(inner) if matches!(inner.as_ref(), IrType::U8));
                                        if is_ptr_u8 {
                                            if let Some(actual_ty) =
                                                self.builder.get_register_type(arg_regs[i])
                                            {
                                                if !matches!(actual_ty, IrType::Ptr(_))
                                                    && i < call_args.len()
                                                {
                                                    if let Some(boxed) = self.box_value_for_dynamic(
                                                        arg_regs[i],
                                                        call_args[i].ty,
                                                    ) {
                                                        arg_regs[i] = boxed;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            // Inject hidden enum type_id for enum helper runtime calls.
                            {
                                let func_name = self
                                    .builder
                                    .module
                                    .functions
                                    .get(&func_id)
                                    .map(|f| f.name.clone())
                                    .or_else(|| {
                                        self.builder
                                            .module
                                            .extern_functions
                                            .get(&func_id)
                                            .map(|f| f.name.clone())
                                    })
                                    .unwrap_or_default();
                                if let Some(result) = self.try_lower_special_runtime_call(
                                    &func_name,
                                    call_args,
                                    result_type.clone(),
                                    expr.source_location.clone(),
                                ) {
                                    return result;
                                }
                                self.inject_hidden_enum_type_id_arg(
                                    &func_name,
                                    call_args,
                                    &mut arg_regs,
                                );
                            }

                            // Infer type_args for static generic calls if not already provided
                            let final_type_args = if converted_hir_type_args.is_empty() {
                                // Check if the function has type parameters
                                if let Some(func) = self.builder.module.functions.get(&func_id) {
                                    if !func.signature.type_params.is_empty()
                                        && !call_args.is_empty()
                                    {
                                        // Try to infer type_args from argument types
                                        debug!(
                                            "[TYPE INFERENCE] Function {} has type_params: {:?}",
                                            func.name, func.signature.type_params
                                        );
                                        debug!(
                                            "[TYPE INFERENCE] Function params: {:?}",
                                            func.signature
                                                .parameters
                                                .iter()
                                                .map(|p| (&p.name, &p.ty))
                                                .collect::<Vec<_>>()
                                        );

                                        let mut inferred: Vec<IrType> = Vec::new();
                                        for (_param_idx, type_param) in
                                            func.signature.type_params.iter().enumerate()
                                        {
                                            // Look for a parameter using this type variable
                                            let mut found = false;
                                            for (i, sig_param) in
                                                func.signature.parameters.iter().enumerate()
                                            {
                                                debug!(
                                                    "[TYPE INFERENCE] Checking param {} type {:?} against type_param {}",
                                                    sig_param.name, sig_param.ty, type_param.name
                                                );
                                                if let IrType::TypeVar(ref name) = sig_param.ty {
                                                    if name == &type_param.name
                                                        && i < call_args.len()
                                                    {
                                                        // Use the concrete type of the corresponding argument
                                                        let arg_type =
                                                            self.convert_type(call_args[i].ty);
                                                        debug!(
                                                            "[TYPE INFERENCE] Inferred {}={:?} from arg {}",
                                                            type_param.name, arg_type, i
                                                        );
                                                        inferred.push(arg_type);
                                                        found = true;
                                                        break;
                                                    }
                                                }
                                            }
                                            if !found {
                                                // Couldn't infer this type param from signature params
                                                // Try using the first argument's type as a fallback for single type param
                                                if func.signature.type_params.len() == 1
                                                    && !call_args.is_empty()
                                                {
                                                    let arg_type =
                                                        self.convert_type(call_args[0].ty);
                                                    debug!(
                                                        "[TYPE INFERENCE] Fallback: Inferred {}={:?} from first arg",
                                                        type_param.name, arg_type
                                                    );
                                                    inferred.push(arg_type);
                                                } else {
                                                    debug!(
                                                        "[TYPE INFERENCE] Could not infer {}, using Any",
                                                        type_param.name
                                                    );
                                                    inferred.push(IrType::Any);
                                                }
                                            }
                                        }
                                        inferred
                                    } else {
                                        Vec::new()
                                    }
                                } else {
                                    Vec::new()
                                }
                            } else {
                                converted_hir_type_args.clone()
                            };

                            // Wrap arguments for constrained type parameters (T:Interface)
                            // If a parameter expects a fat pointer (constrained TypeParam),
                            // wrap the class argument in a fat pointer with the interface's vtable.
                            if let Some(constrained) =
                                self.constrained_param_interfaces.get(&func_id).cloned()
                            {
                                for (param_idx, iface_sym) in &constrained {
                                    if *param_idx < arg_regs.len() && *param_idx < call_args.len() {
                                        let arg_type = call_args[*param_idx].ty;
                                        if let Some(class_sym) = self.get_class_symbol(arg_type) {
                                            if self
                                                .interface_vtables
                                                .contains_key(&(class_sym, *iface_sym))
                                            {
                                                if let Some(wrapped) = self
                                                    .wrap_in_interface_fat_ptr(
                                                        arg_regs[*param_idx],
                                                        class_sym,
                                                        *iface_sym,
                                                    )
                                                {
                                                    arg_regs[*param_idx] = wrapped;
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            // Use HIR type_args or inferred type_args for static generic calls
                            debug!(
                                "[FUNCTION_MAP] Direct call lowered {} args: {:?}, final_type_args: {:?}",
                                arg_regs.len(),
                                arg_regs,
                                final_type_args
                            );
                            let result = if final_type_args.is_empty() {
                                self.builder.build_call_direct(
                                    func_id,
                                    arg_regs,
                                    actual_return_type,
                                )
                            } else {
                                self.builder.build_call_direct_with_type_args(
                                    func_id,
                                    arg_regs,
                                    actual_return_type,
                                    final_type_args,
                                )
                            };
                            debug!("[FUNCTION_MAP] Result: {:?}", result);
                            return result;
                        }
                    } else {
                        // Function not in function_map - might be an extern/stdlib function
                        // Check if it's a stdlib static method (like Math.sin, Sys.println)
                        if let Some(sym_info) = self.symbol_table.get_symbol(*symbol) {
                            if let Some(method_name) = self.string_interner.get(sym_info.name) {
                                let static_args = self.effective_static_call_args(args);
                                // Check if method name matches known Math/Sys methods
                                // Try to find this method in ANY stdlib class with static methods
                                // This replaces the hardcoded is_math_method and is_sys_method checks
                                let method_static: &'static str =
                                    Box::leak(method_name.to_string().into_boxed_str());

                                // Try all stdlib classes that have static methods
                                let mut found_mapping = None;
                                for class_name in self.stdlib_mapping.get_all_classes() {
                                    if self.stdlib_mapping.class_has_static_methods(class_name) {
                                        let sig = crate::stdlib::MethodSignature {
                                            class: class_name,
                                            method: method_static,
                                            is_static: true,
                                            is_constructor: false, // Normal static method, not constructor
                                            param_count: static_args.len(),
                                        };
                                        if let Some(mapping) = self.stdlib_mapping.get(&sig) {
                                            found_mapping = Some((class_name, mapping));
                                            break;
                                        }
                                    }
                                }

                                if let Some((class_name, mapping)) = found_mapping {
                                    self.builder.call_label =
                                        Some(format!("STATIC_SEARCH:{}", class_name));
                                    let runtime_name = mapping.runtime_name;
                                    // eprintln!(
                                    //     "INFO: {} static method detected: {} (runtime: {})",
                                    //     class_name, method_name, runtime_name
                                    // );

                                    // Lower arguments and get their types
                                    let mut arg_regs = Vec::new();
                                    let mut arg_types = Vec::new();
                                    for arg in static_args {
                                        if let Some(reg) = self.lower_expression(arg) {
                                            arg_regs.push(reg);
                                            arg_types.push(self.convert_type(arg.ty));
                                        }
                                    }

                                    // Reflect.compare: use haxe_reflect_compare_typed which accepts
                                    // raw type-erased i64 values + a type tag, avoiding boxing.
                                    // For generic code, the type tag is a placeholder resolved at
                                    // monomorphization time.
                                    if runtime_name == "haxe_reflect_compare" {
                                        let mut type_param_name: Option<String> = None;
                                        let mut known_type_tag: Option<i32> = None;
                                        let mut use_typed_compare = false;

                                        if let Some(first_arg) = static_args.first() {
                                            let type_table = self.type_table;
                                            if let Some(ti) = type_table.get(first_arg.ty) {
                                                match &ti.kind {
                                                    TypeKind::TypeParameter {
                                                        symbol_id, ..
                                                    } => {
                                                        use_typed_compare = true;
                                                        // Get type param name from symbol table
                                                        if let Some(sym) =
                                                            self.symbol_table.get_symbol(*symbol_id)
                                                        {
                                                            if let Some(name_str) =
                                                                self.string_interner.get(sym.name)
                                                            {
                                                                type_param_name =
                                                                    Some(name_str.to_string());
                                                            }
                                                        }
                                                    }
                                                    TypeKind::Int => {
                                                        use_typed_compare = true;
                                                        known_type_tag = Some(1);
                                                    }
                                                    TypeKind::Float => {
                                                        use_typed_compare = true;
                                                        known_type_tag = Some(4);
                                                    }
                                                    TypeKind::Bool => {
                                                        use_typed_compare = true;
                                                        known_type_tag = Some(2);
                                                    }
                                                    TypeKind::String => {
                                                        use_typed_compare = true;
                                                        known_type_tag = Some(5);
                                                    }
                                                    _ => {} // Dynamic/other: fall through to boxing path
                                                }
                                            }
                                        }

                                        if use_typed_compare {
                                            // Cast value args to I64 — haxe_reflect_compare_typed
                                            // takes type-erased i64 values, not typed structs
                                            for i in 0..arg_regs.len().min(2) {
                                                let reg_ty = self
                                                    .builder
                                                    .get_register_type(arg_regs[i])
                                                    .unwrap_or(IrType::I64);
                                                if reg_ty != IrType::I64 {
                                                    if let Some(cast) = self.builder.build_cast(
                                                        arg_regs[i],
                                                        reg_ty,
                                                        IrType::I64,
                                                    ) {
                                                        arg_regs[i] = cast;
                                                    }
                                                }
                                                arg_types[i] = IrType::I64;
                                            }

                                            // Emit type tag constant (placeholder 0 for generics, real value for concrete)
                                            let tag_reg = if let Some(tp_name) = type_param_name {
                                                let tag =
                                                    self.builder.build_const(IrValue::I32(0))?;
                                                // Record fixup for the monomorphize pass to resolve
                                                if let Some(func) =
                                                    self.builder.current_function_mut()
                                                {
                                                    func.type_param_tag_fixups.push((tag, tp_name));
                                                }
                                                tag
                                            } else {
                                                self.builder.build_const(IrValue::I32(
                                                    known_type_tag.unwrap_or(1),
                                                ))?
                                            };

                                            arg_regs.push(tag_reg);
                                            arg_types.push(IrType::I32);

                                            let extern_func_id = self
                                                .get_or_register_extern_function(
                                                    "haxe_reflect_compare_typed",
                                                    arg_types,
                                                    result_type.clone(),
                                                );

                                            return self.builder.build_call_direct(
                                                extern_func_id,
                                                arg_regs,
                                                result_type,
                                            );
                                        } else {
                                            // Dynamic case: box arguments for haxe_reflect_compare
                                            let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                                            for (i, arg) in static_args.iter().enumerate() {
                                                if i >= arg_regs.len() {
                                                    break;
                                                }
                                                if let Some(boxed) =
                                                    self.box_value_for_dynamic(arg_regs[i], arg.ty)
                                                {
                                                    arg_regs[i] = boxed;
                                                    arg_types[i] = ptr_u8.clone();
                                                }
                                            }
                                        }
                                    }

                                    // Register the external runtime function
                                    let extern_func_id = self.get_or_register_extern_function(
                                        runtime_name,
                                        arg_types,
                                        result_type.clone(),
                                    );

                                    // Generate call to external function
                                    return self.builder.build_call_direct(
                                        extern_func_id,
                                        arg_regs,
                                        result_type,
                                    );
                                }
                            }
                        }
                    }
                }

                // Before falling through to indirect call, try to look up by name or register a forward reference
                // for unresolved static method calls (cross-module dependencies during stdlib compilation)
                if let HirExprKind::Variable { symbol, .. } = &callee.kind {
                    let unresolved_call_args = if !*is_method {
                        self.effective_static_call_args(args)
                    } else {
                        args
                    };
                    if let Some(sym_info) = self.symbol_table.get_symbol(*symbol) {
                        if let Some(qual_name) = sym_info.qualified_name {
                            if let Some(qual_name_str) = self.string_interner.get(qual_name) {
                                let _method_name = self
                                    .string_interner
                                    .get(sym_info.name)
                                    .unwrap_or("<unknown>");
                                debug!("[PRE-CHECK] Qualified name: '{}'", qual_name_str);

                                // FIRST: Check if this function is already compiled and in the name map
                                if let Some(&existing_func_id) =
                                    self.external_function_name_map.get(qual_name_str)
                                {
                                    self.builder.call_label =
                                        Some(format!("NAME_MAP_HIT:{}", qual_name_str));
                                    debug!(
                                        "[NAME MAP HIT] Found '{}' in external_function_name_map -> {:?}",
                                        qual_name_str, existing_func_id
                                    );

                                    // Lower arguments
                                    let arg_regs: Vec<_> = unresolved_call_args
                                        .iter()
                                        .filter_map(|a| self.lower_expression(a))
                                        .collect();

                                    // Generate the call to the external function
                                    return self.builder.build_call_direct(
                                        existing_func_id,
                                        arg_regs,
                                        result_type,
                                    );
                                }

                                self.builder.call_label =
                                    Some(format!("FORWARD_REF:{}", qual_name_str));
                                debug!(
                                    "[FORWARD REF] Registering forward reference for unresolved call to '{}'",
                                    qual_name_str
                                );

                                // Lower arguments and collect their types
                                let mut arg_regs = Vec::new();
                                let mut param_types = Vec::new();
                                for arg in unresolved_call_args {
                                    if let Some(reg) = self.lower_expression(arg) {
                                        arg_regs.push(reg);
                                        param_types.push(self.convert_type(arg.ty));
                                    }
                                }

                                // Register as a forward reference using qualified name
                                // This will be resolved later during module linking
                                let forward_func_id = self.register_stdlib_mir_forward_ref(
                                    qual_name_str,
                                    param_types,
                                    result_type.clone(),
                                );

                                debug!(
                                    "[FORWARD REF] Registered forward ref to '{}' with ID {:?}",
                                    qual_name_str, forward_func_id
                                );

                                // Generate the call to the forward reference
                                return self.builder.build_call_direct(
                                    forward_func_id,
                                    arg_regs,
                                    result_type,
                                );
                            }
                        }
                    }
                }

                // Indirect function call (function pointer)
                // TODO: Get the full function signature from the callee's type
                // For now, we'll infer it from the arguments and return type
                // This is a temporary workaround until we pass type_table to HIR→MIR
                self.builder.call_label = Some("INDIRECT_CALL".to_string());

                debug!(
                    "Taking indirect function call path - callee kind={:?}, args.len()={}",
                    std::mem::discriminant(&callee.kind),
                    args.len()
                );

                // Lower arguments FIRST, before trying to lower the callee
                // This ensures lambdas in arguments get generated even if callee lowering fails
                debug!("About to lower {} indirect call arguments", args.len());
                for (i, a) in args.iter().enumerate() {
                    debug!("  arg[{}] kind={:?}", i, std::mem::discriminant(&a.kind));
                }
                let arg_regs: Vec<_> = args
                    .iter()
                    .filter_map(|a| {
                        debug!(
                            "NOW lowering arg with kind={:?}",
                            std::mem::discriminant(&a.kind)
                        );
                        self.lower_expression(a)
                    })
                    .collect();
                debug!(
                    "Lowered {} indirect call arguments successfully",
                    arg_regs.len()
                );

                // Now try to lower the callee - if this fails, the call won't be generated
                // but the lambda functions in arguments will have been created
                let func_ptr = self.lower_expression(callee)?;

                // Build function signature from callee type or argument types
                let param_types: Vec<IrType> = {
                    // Try to get param types from callee's function type
                    let type_table = self.type_table;
                    let callee_type = type_table.get(callee.ty);
                    if let Some(type_ref) = callee_type {
                        if let crate::tast::TypeKind::Function { params, .. } = &type_ref.kind {
                            // `Void -> T` is Haxe's spelling for "takes nothing",
                            // so a Void entry is notation rather than a slot. Left
                            // in, the signature has a parameter no call site can
                            // fill: LLVM rejects a Void parameter outright, and
                            // Cranelift asserts on the argument count once the
                            // closure is called with its environment.
                            params
                                .iter()
                                .map(|p| self.convert_type(*p))
                                .filter(|t| !matches!(t, IrType::Void))
                                .collect()
                        } else {
                            // Fallback: infer from actual argument types
                            args.iter().map(|a| self.convert_type(a.ty)).collect()
                        }
                    } else {
                        args.iter().map(|a| self.convert_type(a.ty)).collect()
                    }
                };
                let return_type = Box::new(self.convert_type(expr.ty));

                let func_signature = IrType::Function {
                    params: param_types,
                    return_type,
                    varargs: false,
                };

                self.builder
                    .build_call_indirect(func_ptr, arg_regs, func_signature)
            }

            HirExprKind::New {
                class_type,
                type_args: hir_type_args,
                args,
                class_name: hir_class_name,
                ..
            } => {
                let debug_class_name =
                    hir_class_name.and_then(|interned| self.string_interner.get(interned));
                debug!(
                    "[NEW EXPR]: class_type={:?}, args.len={}, hir_class_name={:?}, hir_type_args={:?}",
                    class_type,
                    args.len(),
                    debug_class_name,
                    hir_type_args
                );

                // Check if this is an abstract type
                let type_table = self.type_table;
                let (is_abstract, actual_symbol_id) = if let Some(type_ref) =
                    type_table.get(*class_type)
                {
                    let symbol_id = match &type_ref.kind {
                        crate::tast::TypeKind::Class { symbol_id, .. } => Some(*symbol_id),
                        crate::tast::TypeKind::Abstract { symbol_id, .. } => Some(*symbol_id),
                        crate::tast::TypeKind::GenericInstance { base_type, .. } => {
                            // Unwrap GenericInstance to find base class symbol_id
                            if let Some(base_info) = type_table.get(*base_type) {
                                match &base_info.kind {
                                    crate::tast::TypeKind::Class { symbol_id, .. } => {
                                        Some(*symbol_id)
                                    }
                                    _ => None,
                                }
                            } else {
                                None
                            }
                        }
                        crate::tast::TypeKind::TypeAlias {
                            symbol_id,
                            target_type,
                            ..
                        } => {
                            // Resolve TypeAlias to find the actual class symbol
                            // For user classes, the TypeAlias symbol_id IS the class symbol
                            let mut resolved = Some(*symbol_id);
                            // Also try resolving through target_type for deeper aliases
                            if let Some(target) = type_table.get(*target_type) {
                                if let crate::tast::TypeKind::Class {
                                    symbol_id: class_sym,
                                    ..
                                } = &target.kind
                                {
                                    resolved = Some(*class_sym);
                                }
                            }
                            resolved
                        }
                        _ => None,
                    };
                    let is_abs = matches!(type_ref.kind, crate::tast::TypeKind::Abstract { .. });
                    (is_abs, symbol_id)
                } else {
                    (false, None)
                };

                // Cross-module `new C()` where C is an imported user class often
                // arrives with `class_type` as a Placeholder (the class metadata
                // wasn't resolved in THIS lowering context), so `actual_symbol_id`
                // is None. That degrades the allocation to a bare 16-byte block
                // with a ZERO class-id header, NO constructor call, and NO
                // interface fat-ptr wrap — the exact shape that made
                // `ArchRegistry.withDefaults`'s `new LlamaArch()` produce a
                // headerless object whose later interface dispatch read garbage
                // (Void-tagged receiver → SIGSEGV). Recover the class SymbolId by
                // name (Placeholder name or the HIR class name), mirroring the
                // interface machinery's name-based lazy resolution. This restores
                // the proper alloc size, header id, ctor call, and iface wrap.
                let actual_symbol_id = actual_symbol_id.or_else(|| {
                    let placeholder_name = type_table.get(*class_type).and_then(|t| {
                        if let crate::tast::TypeKind::Placeholder { name } = &t.kind {
                            self.string_interner.get(*name)
                        } else {
                            None
                        }
                    });
                    let by_name = placeholder_name.or_else(|| {
                        hir_class_name.and_then(|interned| self.string_interner.get(interned))
                    });
                    by_name.and_then(|name| self.lookup_class_symbol_by_name(name))
                });

                // SPECIAL CASE: Abstract type constructors
                // If this is an abstract type, treat this as a simple value wrap (no allocation).
                if is_abstract {
                    // Before treating as abstract value-wrap, check if there's a user-defined
                    // class with the same name that has a constructor registered. User classes
                    // should override stdlib abstract types with the same short name
                    // (e.g., user's `class Box<T>` vs stdlib `rayzor.Box<T>` abstract).
                    let has_user_constructor = hir_class_name
                        .and_then(|interned| self.string_interner.get(interned))
                        .map(|name| self.constructor_name_map.contains_key(name))
                        .unwrap_or(false)
                        || actual_symbol_id
                            .and_then(|sid| self.symbol_table.get_symbol(sid))
                            .and_then(|s| s.qualified_name)
                            .and_then(|qn| self.string_interner.get(qn))
                            .map(|qn| self.constructor_name_map.contains_key(qn))
                            .unwrap_or(false);

                    if !has_user_constructor {
                        if args.len() == 1 {
                            return self.lower_expression(&args[0]);
                        } else if args.is_empty() {
                            return self.builder.build_const(IrValue::I32(0));
                        } else {
                            return self.lower_expression(&args[0]);
                        }
                    }
                    // User constructor exists - fall through to normal class construction
                }

                // SPECIAL CASE: Check if this is an extern stdlib class constructor BEFORE fallback
                // For extern stdlib classes (Channel, Thread, Arc, Mutex), we call the MIR wrapper
                // function (e.g., Channel_init) instead of allocating and calling a constructor.
                // This MUST come before the value-wrap fallback to prevent returning the argument value!

                // PRIORITY #1: Use class_name from HIR if available (preserves actual class name even when TypeId is invalid)
                let mut class_name =
                    hir_class_name.and_then(|interned| self.string_interner.get(interned));

                // FALLBACK #1: Try to get class name from TypeId if HIR didn't have it
                if class_name.is_none() {
                    let type_table = self.type_table;
                    class_name = if let Some(type_ref) = type_table.get(*class_type) {
                        match &type_ref.kind {
                            crate::tast::TypeKind::Class { symbol_id, .. } => self
                                .symbol_table
                                .get_symbol(*symbol_id)
                                .and_then(|sym| self.string_interner.get(sym.name)),
                            crate::tast::TypeKind::GenericInstance { base_type, .. } => {
                                // Unwrap GenericInstance to get base class name
                                if let Some(base_info) = type_table.get(*base_type) {
                                    if let crate::tast::TypeKind::Class { symbol_id, .. } =
                                        &base_info.kind
                                    {
                                        self.symbol_table
                                            .get_symbol(*symbol_id)
                                            .and_then(|sym| self.string_interner.get(sym.name))
                                    } else {
                                        None
                                    }
                                } else {
                                    None
                                }
                            }
                            _ => None,
                        }
                    } else {
                        None
                    };
                }

                // FALLBACK #2: If TypeId lookup failed (e.g., for extern stdlib classes that aren't
                // pre-registered because Channel.hx is skipped), try getting class name from the
                // actual_symbol_id which comes from the HIR New expression
                if class_name.is_none() {
                    if let Some(sym_id) = actual_symbol_id {
                        class_name = self
                            .symbol_table
                            .get_symbol(sym_id)
                            .and_then(|sym| self.string_interner.get(sym.name));
                    }
                }

                // FALLBACK #3: If still no class name and TypeId is invalid (u32::MAX),
                // try checking all stdlib registered class names to see if ANY constructor matches
                // This is a last resort for extern stdlib classes that weren't pre-registered
                if class_name.is_none() && *class_type == TypeId::from_raw(u32::MAX) {
                    // Get ALL classes that have registered constructors from the stdlib mapping
                    let constructor_classes = self.stdlib_mapping.get_constructor_classes();

                    // Try each registered constructor class
                    for potential_class in constructor_classes {
                        let method_sig = crate::stdlib::runtime_mapping::MethodSignature {
                            class: potential_class,
                            method: "new",
                            is_static: true,
                            is_constructor: true,
                            param_count: 0,
                        };
                        if self.stdlib_mapping.get(&method_sig).is_some() {
                            class_name = Some(potential_class);
                            break;
                        }
                    }
                }

                // MONOMORPHIZATION: For generic extern classes like Vec<T>, monomorphize the class name
                // based on type arguments. Vec<Int> -> VecI32, Vec<Float> -> VecF64, etc.
                // Use hir_type_args directly (from HIR) instead of type_table lookup (which may fail for extern classes)
                let monomorphized_class_name: Option<String> = if let Some(base_name) = class_name {
                    if base_name == "Vec" && !hir_type_args.is_empty() {
                        // Get the first type argument and determine the monomorphized suffix
                        let first_arg = hir_type_args[0];
                        let type_table = self.type_table;
                        let suffix = if let Some(arg_type) = type_table.get(first_arg) {
                            match &arg_type.kind {
                                crate::tast::TypeKind::Int => Some("I32"),
                                crate::tast::TypeKind::Float => Some("F64"),
                                crate::tast::TypeKind::Bool => Some("Bool"),
                                crate::tast::TypeKind::String => Some("Ptr"),
                                crate::tast::TypeKind::Class { symbol_id, .. } => {
                                    // Check if it's Int64 (a class type representing 64-bit int)
                                    if let Some(class_info) =
                                        self.symbol_table.get_symbol(*symbol_id)
                                    {
                                        if let Some(name) =
                                            self.string_interner.get(class_info.name)
                                        {
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
                                _ => Some("Ptr"),
                            }
                        } else {
                            Some("Ptr") // If type not found, default to Ptr variant
                        };
                        suffix.map(|s| format!("Vec{}", s))
                    } else {
                        None
                    }
                } else {
                    None
                };

                // Use monomorphized name if available, otherwise use original class name
                let final_class_name = monomorphized_class_name.as_deref().or(class_name);
                debug!(
                    "[NEW EXPR]: final_class_name={:?} (monomorphized from {:?})",
                    final_class_name, class_name
                );

                if let Some(class_name) = final_class_name {
                    // Check if this class has a "new" constructor registered in the runtime mapping
                    // Use find_constructor to look up the registered constructor mapping
                    // This returns both the MethodSignature and RuntimeFunctionCall from the registry
                    // PRIORITY: Try HIR class name first (may be fully qualified, e.g., "sys.ssl.Socket")
                    // This is critical for subclasses like sys.ssl.Socket that extend sys.net.Socket,
                    // because the symbol's qualified_name resolves to the parent class.
                    // Look up constructor: extract needed data (Copy types) to avoid holding
                    // a borrow on self.stdlib_mapping while calling self.lower_expression later.
                    let arg_count = args.len();
                    let constructor_info: Option<(&'static str, bool, bool)> = {
                        let mut found = None;

                        // Helper: try constructor lookup with param count first, then without.
                        // This ensures overloaded constructors (e.g., Uncompress with 0 or 1 args)
                        // select the correct overload.
                        macro_rules! try_find_ctor {
                            ($name:expr) => {
                                if found.is_none() {
                                    // Try param-count-aware lookup first
                                    if let Some((_, rc)) = self
                                        .stdlib_mapping
                                        .find_constructor_with_params($name, arg_count)
                                    {
                                        found = Some((
                                            rc.runtime_name,
                                            rc.needs_out_param,
                                            rc.is_mir_wrapper,
                                        ));
                                    }
                                    // Fall back to any-param lookup
                                    else if let Some((_, rc)) =
                                        self.stdlib_mapping.find_constructor($name)
                                    {
                                        found = Some((
                                            rc.runtime_name,
                                            rc.needs_out_param,
                                            rc.is_mir_wrapper,
                                        ));
                                    }
                                }
                            };
                        }

                        // PRIORITY #0: Try HIR class name as qualified (e.g., "sys.ssl.Socket" -> "sys_ssl_Socket")
                        // This handles subclass constructors before the symbol-based lookup (which may
                        // resolve to the parent class due to class inheritance).
                        if class_name.contains('.') {
                            let qualified_hir = class_name.replace(".", "_");
                            try_find_ctor!(&qualified_hir);
                        }
                        if found.is_none() {
                            if let Some(sym_id) = actual_symbol_id {
                                if let Some(sym) = self.symbol_table.get_symbol(sym_id) {
                                    // Try lowered @:native name first
                                    if let Some(native) = sym.native_name {
                                        if let Some(native_str) = self.string_interner.get(native) {
                                            let native_class_name = native_str.replace("::", "_");
                                            try_find_ctor!(&native_class_name);
                                        }
                                    }
                                    // Fall back to qualified name
                                    if found.is_none() {
                                        if let Some(qn) = sym.qualified_name {
                                            if let Some(qual_name) = self.string_interner.get(qn) {
                                                let qualified_class_name =
                                                    qual_name.replace(".", "_");
                                                try_find_ctor!(&qualified_class_name);
                                            }
                                        }
                                    }
                                }
                            }
                            // FALLBACK: Try simple class name
                            try_find_ctor!(class_name);

                            // FALLBACK #2: Try bare class name for qualified paths
                            // (e.g., "haxe.ds.ObjectMap" -> "ObjectMap")
                            if found.is_none() && class_name.contains('.') {
                                let bare_name = class_name.rsplit('.').next().unwrap_or(class_name);
                                try_find_ctor!(bare_name);
                            }
                        } // close PRIORITY #0 if found.is_none()
                        found
                    };
                    if let Some((wrapper_name, needs_out_param, is_mir_wrapper)) = constructor_info
                    {
                        // Lower arguments
                        let arg_regs: Vec<_> = args
                            .iter()
                            .filter_map(|a| self.lower_expression(a))
                            .collect();

                        // Register forward ref if not already present
                        let param_types: Vec<IrType> = arg_regs
                            .iter()
                            .map(|reg| self.builder.get_register_type(*reg).unwrap_or(IrType::Any))
                            .collect();

                        // For extern classes, the return type should be a pointer (opaque handle),
                        // not the class struct itself
                        let result_type = IrType::Ptr(Box::new(IrType::U8));

                        // MIR wrappers are compiled by Cranelift alongside user code -> use forward ref
                        // needs_out_param constructors also use MIR forward refs (legacy path)
                        // Direct extern calls are for non-wrapper, non-out-param constructors
                        let wrapper_func_id = if is_mir_wrapper || needs_out_param {
                            self.register_stdlib_mir_forward_ref(
                                wrapper_name,
                                param_types,
                                result_type.clone(),
                            )
                        } else {
                            // Simple constructors are direct extern calls
                            self.get_or_register_extern_function(
                                wrapper_name,
                                param_types,
                                result_type.clone(),
                            )
                        };

                        // Call the wrapper and return the result
                        let result =
                            self.builder
                                .build_call_direct(wrapper_func_id, arg_regs, result_type);
                        if let Some(reg) = result {
                            self.register_class_hints
                                .insert(reg, class_name.to_string());
                        }
                        return result;
                    }
                }

                // Check if constructor exists - try both TypeId and TypeId derived from SymbolId
                let mut constructor_type_id = *class_type;
                let mut has_constructor = self.constructor_map.contains_key(class_type);
                let mut ctor_path = if has_constructor { "typeid" } else { "none" };

                // If not found and we have a SymbolId, try TypeId derived from SymbolId as fallback.
                //
                // SymbolId and TypeId are DIFFERENT numbering spaces sharing this
                // map, so this lookup can land on another class's entry: `new
                // ChatResponse` picked up GenerationLoop's ctor (cached at
                // GenerationLoop's TypeId, numerically equal to ChatResponse's
                // SymbolId) and the Int passed in a String slot SIGSEGV'd at
                // runtime — host-dependent, since symbol numbering shifts with
                // module count. Accept the pun only when the candidate's owning
                // class matches the class being constructed; an unowned entry
                // passes (bridge for ctors registered before owners existed).
                if !has_constructor {
                    if let Some(sym_id) = actual_symbol_id {
                        let type_id_from_symbol = TypeId::from_raw(sym_id.as_raw());
                        if let Some(&cand) = self.constructor_map.get(&type_id_from_symbol) {
                            let owner_ok =
                                match (self.constructor_owner_map.get(&cand), debug_class_name) {
                                    (Some(owner), Some(target)) => {
                                        Self::class_names_match(owner, target)
                                    }
                                    _ => true,
                                };
                            if owner_ok {
                                constructor_type_id = type_id_from_symbol;
                                has_constructor = true;
                                ctor_path = "SYMBOL-PUN";
                            } else if std::env::var("RAYZOR_CTOR_DEBUG").is_ok() {
                                eprintln!(
                                    "[CTOR] class={:?} REFUSED pun candidate {:?} (owner={:?})",
                                    debug_class_name,
                                    cand,
                                    self.constructor_owner_map.get(&cand)
                                );
                            }
                        }
                    }
                }

                // If not found and this is a GenericInstance, try the base class TypeId
                if !has_constructor {
                    let type_table = self.type_table;
                    if let Some(type_info) = type_table.get(*class_type) {
                        if let crate::tast::TypeKind::GenericInstance { base_type, .. } =
                            &type_info.kind
                        {
                            if self.constructor_map.contains_key(base_type) {
                                constructor_type_id = *base_type;
                                has_constructor = true;
                                ctor_path = "generic-base";
                            }
                        }
                    }
                }

                // Try constructor_name_map as final fallback before value wrap
                // Check bare name first, then qualified name from symbol table
                if !has_constructor {
                    if let Some(class_name) = final_class_name {
                        if let Some(&func_id) = self.constructor_name_map.get(class_name) {
                            constructor_type_id = *class_type;
                            has_constructor = true;
                            ctor_path = "bare-name";
                            self.constructor_map.insert(*class_type, func_id);
                            let bare = class_name.rsplit('.').next().unwrap_or(class_name);
                            self.constructor_owner_map
                                .entry(func_id)
                                .or_insert_with(|| bare.to_string());
                        }
                    }
                    // Also try qualified name (e.g., "haxe.ds.List" vs bare "List")
                    if !has_constructor {
                        if let Some(sym_id) = actual_symbol_id {
                            if let Some(sym) = self.symbol_table.get_symbol(sym_id) {
                                if let Some(qn) = sym.qualified_name {
                                    if let Some(qual_name) = self.string_interner.get(qn) {
                                        if let Some(&func_id) =
                                            self.constructor_name_map.get(qual_name)
                                        {
                                            constructor_type_id = *class_type;
                                            has_constructor = true;
                                            ctor_path = "qual-name";
                                            self.constructor_map.insert(*class_type, func_id);
                                            let bare =
                                                qual_name.rsplit('.').next().unwrap_or(qual_name);
                                            self.constructor_owner_map
                                                .entry(func_id)
                                                .or_insert_with(|| bare.to_string());
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // If no constructor exists and we have exactly one argument, treat as value wrap
                // This handles abstract types that weren't properly detected above
                if !has_constructor && args.len() == 1 {
                    let result = self.lower_expression(&args[0]);
                    return result;
                }

                // SPECIAL CASE: Array constructor (@:coreType extern class)
                // Array needs special handling - call haxe_array_new() runtime function
                let type_table = self.type_table;
                let is_array = if let Some(type_ref) = type_table.get(*class_type) {
                    matches!(type_ref.kind, crate::tast::TypeKind::Array { .. })
                } else {
                    false
                };

                if is_array {
                    // Allocate HaxeArray struct on heap (32 bytes = 4 x 8 for ptr, len, cap, elem_size)
                    // Must be heap-allocated because array pointers can escape the creating function
                    // (e.g., stored in class fields, returned from functions)
                    let array_ptr = self.build_heap_alloc(32)?;

                    // Zero-initialize the HaxeArray struct (ptr=null, len=0, cap=0, elem_size=8)
                    // This represents an empty uninitialized array
                    if let Some(zero_i64) = self.builder.build_const(IrValue::I64(0)) {
                        // Zero out ptr field (offset 0)
                        if let Some(index_0) = self.builder.build_const(IrValue::I32(0)) {
                            if let Some(ptr_field) =
                                self.builder
                                    .build_gep(array_ptr, vec![index_0], IrType::I64)
                            {
                                self.builder.build_store(ptr_field, zero_i64);
                            }
                        }
                        // Zero out len field (offset 8)
                        if let Some(index_1) = self.builder.build_const(IrValue::I32(1)) {
                            if let Some(len_field) =
                                self.builder
                                    .build_gep(array_ptr, vec![index_1], IrType::I64)
                            {
                                self.builder.build_store(len_field, zero_i64);
                            }
                        }
                        // Zero out cap field (offset 16)
                        if let Some(index_2) = self.builder.build_const(IrValue::I32(2)) {
                            if let Some(cap_field) =
                                self.builder
                                    .build_gep(array_ptr, vec![index_2], IrType::I64)
                            {
                                self.builder.build_store(cap_field, zero_i64);
                            }
                        }
                        // Set elem_size field to 8 bytes (offset 24) - assume pointer size for now
                        if let Some(elem_size_val) = self.builder.build_const(IrValue::I64(8)) {
                            if let Some(index_3) = self.builder.build_const(IrValue::I32(3)) {
                                if let Some(elem_size_field) =
                                    self.builder
                                        .build_gep(array_ptr, vec![index_3], IrType::I64)
                                {
                                    self.builder.build_store(elem_size_field, elem_size_val);
                                }
                            }
                        }
                    }

                    // Return the zero-initialized array pointer
                    return Some(array_ptr);
                }

                // @:cstruct CLASS: flat C-compatible allocation (no object header)
                if self.is_cstruct_class(*class_type) {
                    if let Some(layout) = self.get_or_compute_cstruct_layout(*class_type) {
                        let obj_ptr = self.build_heap_alloc(layout.total_size as u64)?;

                        // Zero-initialize all bytes via first field store of 0
                        // (memset-style zero init would be better but this works)
                        if let Some(zero) = self.builder.build_const(IrValue::I64(0)) {
                            for field in &layout.fields {
                                let offset_const = self
                                    .builder
                                    .build_const(IrValue::I64(field.byte_offset as i64))?;
                                let field_ptr = self.builder.build_ptr_add(
                                    obj_ptr,
                                    offset_const,
                                    IrType::Ptr(Box::new(IrType::U8)),
                                )?;
                                let zero_val = match &field.ir_type {
                                    IrType::F64 => self.builder.build_const(IrValue::F64(0.0))?,
                                    IrType::I64 => self.builder.build_const(IrValue::I64(0))?,
                                    IrType::I32 => self.builder.build_const(IrValue::I32(0))?,
                                    IrType::Bool => {
                                        self.builder.build_const(IrValue::Bool(false))?
                                    }
                                    _ => self.builder.build_const(IrValue::I64(0))?,
                                };
                                self.builder.build_store(field_ptr, zero_val);
                            }
                        }

                        // Call constructor if exists
                        let constructor_func_id =
                            self.constructor_map.get(&constructor_type_id).copied();
                        if let Some(constructor_func_id) = constructor_func_id {
                            let arg_regs: Vec<_> = std::iter::once(obj_ptr)
                                .chain(args.iter().filter_map(|a| self.lower_expression(a)))
                                .collect();
                            self.builder.build_call_direct(
                                constructor_func_id,
                                arg_regs,
                                IrType::Void,
                            );
                        }

                        return Some(obj_ptr);
                    }
                }

                // @:gpuStruct CLASS: GPU-compatible flat allocation (no object header)
                // Uses 4-byte floats, 4-byte ints — matches Metal/CUDA struct layout
                if self.is_gpu_struct_class(*class_type) {
                    if let Some(layout) = self.get_or_compute_gpu_struct_layout(*class_type) {
                        let obj_ptr = self.build_heap_alloc(layout.total_size as u64)?;

                        // Zero-initialize all fields
                        if let Some(_zero) = self.builder.build_const(IrValue::I32(0)) {
                            for field in &layout.fields {
                                let offset_const = self
                                    .builder
                                    .build_const(IrValue::I64(field.byte_offset as i64))?;
                                let field_ptr = self.builder.build_ptr_add(
                                    obj_ptr,
                                    offset_const,
                                    IrType::Ptr(Box::new(IrType::U8)),
                                )?;
                                let zero_val = match &field.ir_type {
                                    IrType::F32 => self.builder.build_const(IrValue::F32(0.0))?,
                                    IrType::I32 => self.builder.build_const(IrValue::I32(0))?,
                                    _ => self.builder.build_const(IrValue::I32(0))?,
                                };
                                self.builder.build_store(field_ptr, zero_val);
                            }
                        }

                        // Call constructor if exists
                        let constructor_func_id =
                            self.constructor_map.get(&constructor_type_id).copied();
                        if let Some(constructor_func_id) = constructor_func_id {
                            let arg_regs: Vec<_> = std::iter::once(obj_ptr)
                                .chain(args.iter().filter_map(|a| self.lower_expression(a)))
                                .collect();
                            self.builder.build_call_direct(
                                constructor_func_id,
                                arg_regs,
                                IrType::Void,
                            );
                        }

                        return Some(obj_ptr);
                    }
                }

                // CLASS TYPE CONSTRUCTOR:
                // Allocate object on HEAP (not stack) since objects may escape the current function
                // When a method returns `new Foo()`, the object must outlive the callee's stack frame
                let _class_mir_type = self.convert_type(*class_type);

                // Look up pre-computed allocation size from register_class_metadata.
                // Key order: same-context SymbolId, then QUALIFIED NAME — the only
                // key stable across compilation contexts. Context-local TypeIds
                // (including accumulated cross-context TypeId→SymbolId maps) must
                // never be consulted: a shifted id resolves to another class's
                // size and silently under-allocates.
                let alloc_dbg = std::env::var_os("RAYZOR_ALLOC_DEBUG").is_some();
                let stage = std::cell::Cell::new("sym");
                let class_sym_for_name = actual_symbol_id.or_else(|| {
                    self.type_table
                        .get(*class_type)
                        .and_then(|t| match &t.kind {
                            crate::tast::TypeKind::Class { symbol_id, .. } => Some(*symbol_id),
                            _ => None,
                        })
                });
                let class_qname: Option<&str> = class_sym_for_name
                    .and_then(|sid| self.symbol_table.get_symbol(sid))
                    .and_then(|sym| {
                        sym.qualified_name
                            .and_then(|n| self.string_interner.get(n))
                            .or_else(|| self.string_interner.get(sym.name))
                    });
                let obj_size: u64 = actual_symbol_id
                    .and_then(|sid| self.class_alloc_sizes.get(&sid).copied())
                    .or_else(|| {
                        stage.set("name");
                        class_qname.and_then(|n| self.class_alloc_sizes_by_name.get(n).copied())
                    })
                    // Fallback: declared instance-field count from the parsed
                    // AST (class compiles later than this module — no layout
                    // registered anywhere yet).
                    .or_else(|| {
                        stage.set("ast");
                        let index = self.static_sig_index.as_ref()?.clone();
                        let sym = self.symbol_table.get_symbol(class_sym_for_name?)?;
                        let qname = sym.qualified_name.and_then(|n| self.string_interner.get(n));
                        let bare = self.string_interner.get(sym.name);
                        let mut index = index.borrow_mut();
                        let no_parse = |_: &str| -> Option<std::path::PathBuf> { None };
                        let count = qname
                            .and_then(|q| index.instance_field_count(q, &no_parse))
                            .or_else(|| {
                                bare.filter(|b| qname != Some(*b))
                                    .and_then(|b| index.instance_field_count(b, &no_parse))
                            })?;
                        Some((count as u64 + 1) * 8)
                    })
                    .unwrap_or_else(|| {
                        stage.set("argcount");
                        ((args.len() as u64 + 1) * 8).max(16)
                    });
                if alloc_dbg {
                    let cname = actual_symbol_id
                        .and_then(|sid| self.symbol_table.get_symbol(sid))
                        .and_then(|sym| {
                            sym.qualified_name
                                .and_then(|n| self.string_interner.get(n))
                                .or_else(|| self.string_interner.get(sym.name))
                        })
                        .unwrap_or("<?>");
                    eprintln!(
                        "[alloc-size] class={} sym={:?} type={:?} stage={} size={}",
                        cname,
                        actual_symbol_id,
                        class_type,
                        stage.get(),
                        obj_size
                    );
                }
                // Ensure the allocation covers inherited fields: a subclass of an
                // imported class (e.g. MyException extends haxe.Exception) can be
                // registered with an undersized `obj_size` because the parent's
                // fields weren't available at registration; inherited-field writes
                // (the `stack` field at `throw`) would then overflow the heap block.
                let obj_size =
                    self.alloc_size_with_inheritance(*class_type, actual_symbol_id, obj_size);
                // Use heap allocation (malloc) for class instances
                let obj_ptr = self.build_heap_alloc(obj_size);
                let obj_ptr = obj_ptr?;

                // Store object header: runtime type_id at GEP index 0.
                // Use the same class-id resolver as typed throw/catch (without +1000).
                {
                    // Store the runtime_type_id directly (no -1000 transform).
                    // The id is now a stable name-hash; cast/is checks
                    // compare with the same `runtime_type_id()` value, so
                    // any offset transformation here would just have to be
                    // mirrored on the comparison side. Keeping the id raw
                    // makes both sides agree by construction.
                    //
                    // If `class_type` is a Placeholder (cross-module import),
                    // `runtime_type_id` returns 0. Prefer the name-recovered
                    // `actual_symbol_id`'s deterministic id so the header
                    // carries the real class id (needed for is/cast/vtable and
                    // for the interface fat-ptr wrap to find the vtable).
                    let raw_type_id = self.runtime_type_id(*class_type);
                    let runtime_type_id = if raw_type_id != 0 {
                        raw_type_id
                    } else {
                        actual_symbol_id
                            .and_then(|sid| self.deterministic_class_type_id(sid))
                            .unwrap_or(raw_type_id)
                    } as i64;
                    if let Some(type_id_const) =
                        self.builder.build_const(IrValue::I64(runtime_type_id))
                    {
                        if let Some(index_0) = self.builder.build_const(IrValue::I32(0)) {
                            if let Some(header_ptr) =
                                self.builder.build_gep(obj_ptr, vec![index_0], IrType::I64)
                            {
                                self.builder.build_store(header_ptr, type_id_const);
                            }
                        }
                    }
                }

                // @:derive(Default): initialize all fields before constructor call.
                // Uses @:default(value) metadata if present, else type-based defaults.
                if let Some(class_sym) = actual_symbol_id {
                    if self.derive_default_classes.contains(&class_sym) {
                        if let Some(fields) = self.class_instance_fields.get(&class_sym).cloned() {
                            // Pre-collect field defaults to avoid borrow issues
                            let field_defaults: Vec<_> = fields
                                .iter()
                                .map(|&(field_sym, _, _)| {
                                    self.field_default_exprs.get(&field_sym).cloned()
                                })
                                .collect();

                            for (i, &(_field_sym, field_type_id, gep_idx)) in
                                fields.iter().enumerate()
                            {
                                let field_ir_type = self.convert_type(field_type_id);
                                let default_val = if let Some(ref default_expr) = field_defaults[i]
                                {
                                    self.lower_expression(default_expr)
                                        .or_else(|| self.build_type_default(field_type_id))
                                } else {
                                    self.build_type_default(field_type_id)
                                };
                                if let Some(val) = default_val {
                                    let idx =
                                        self.builder.build_const(IrValue::I32(gep_idx as i32));
                                    if let Some(idx) = idx {
                                        let ptr = self.builder.build_gep(
                                            obj_ptr,
                                            vec![idx],
                                            field_ir_type,
                                        );
                                        if let Some(ptr) = ptr {
                                            self.builder.build_store(ptr, val);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Look up constructor by TypeId - use the resolved constructor_type_id
                let constructor_func_id = self.constructor_map.get(&constructor_type_id).copied();
                if std::env::var("RAYZOR_CTOR_DEBUG").is_ok() {
                    let cname = debug_class_name.unwrap_or("?");
                    eprintln!(
                        "[CTOR] class={} path={} class_type={:?} resolved_type={:?} fid={:?}",
                        cname, ctor_path, class_type, constructor_type_id, constructor_func_id
                    );
                }
                if let Some(constructor_func_id) = constructor_func_id {
                    // Call constructor with object as first argument.
                    //
                    // Each user arg goes through `maybe_materialize_for_call`
                    // (handles anon-views, class→anon coercion, and the
                    // class→interface fat-pointer wrap). Without this,
                    // `new Holder(c)` where `Holder.new(it:I)` stores the
                    // raw class pointer into the interface-typed field and
                    // `h.slot.value()` SIGSEGVs on virtual dispatch.
                    let mut arg_regs: Vec<IrId> = Vec::with_capacity(args.len() + 1);
                    arg_regs.push(obj_ptr);
                    // HIR constructor params don't include `this`, so user arg
                    // `args[i]` lines up with HIR param index `i`. Don't shift
                    // by +1 the way Call-method dispatch does — that would
                    // bias the lookup off the end of `function_param_hir_types`
                    // and silently skip Path 3 (class→interface wrap).
                    for (i, a) in args.iter().enumerate() {
                        if let Some(reg) = self.lower_expression(a) {
                            let wrapped = self.maybe_materialize_for_call(
                                a,
                                reg,
                                Some(constructor_func_id),
                                i,
                            );
                            arg_regs.push(wrapped);
                        }
                    }

                    let pre_fill = arg_regs.len();
                    // Coerce Int→Float at cross-module call boundaries
                    self.coerce_args_for_cross_module_call(
                        constructor_func_id,
                        &mut arg_regs,
                        true,
                    );
                    // Fill in default values for any missing optional parameters
                    self.fill_default_args(constructor_func_id, &mut arg_regs, true);
                    let post_fill = arg_regs.len();
                    let sig_params = self
                        .builder
                        .module
                        .functions
                        .get(&constructor_func_id)
                        .map(|f| f.signature.parameters.len())
                        .unwrap_or(0);

                    // Generic constructor specialization: a `new Foo<T1,T2>(...)`
                    // call must carry the concrete type args so the monomorphizer
                    // can specialize Foo.new. Untyped constructor params lower to
                    // `*void`, which defeats the monomorphizer's TypeVar-based
                    // argument inference (extract_generic_call), so without explicit
                    // type_args the generic ctor template is left un-monomorphized
                    // and trap-stubbed → SIGILL at runtime. This bit haxe.ds.TreeNode
                    // (self-referential `left`/`right` + an in-ctor method call).
                    // Prefer the explicit `new Foo<...>` args; fall back to the
                    // class_type's own type_args (GenericInstance / Class).
                    let ctor_type_args: Vec<IrType> = {
                        let mut ta_ids: Vec<TypeId> = hir_type_args.to_vec();
                        if ta_ids.is_empty() {
                            if let Some(ti) = self.type_table.get(*class_type) {
                                ta_ids = match &ti.kind {
                                    crate::tast::TypeKind::GenericInstance {
                                        type_args, ..
                                    }
                                    | crate::tast::TypeKind::Class { type_args, .. } => {
                                        type_args.clone()
                                    }
                                    _ => Vec::new(),
                                };
                            }
                        }
                        let converted: Vec<IrType> =
                            ta_ids.iter().map(|&t| self.convert_type(t)).collect();
                        // Only request specialization when every arg is concrete; an
                        // unresolved TypeVar means we're still inside a generic
                        // context and the enclosing instantiation handles it.
                        if converted.is_empty()
                            || converted.iter().any(|t| matches!(t, IrType::TypeVar(_)))
                        {
                            Vec::new()
                        } else {
                            converted
                        }
                    };

                    // Constructor returns void, so we ignore the result
                    if ctor_type_args.is_empty() {
                        self.builder
                            .build_call_direct(constructor_func_id, arg_regs, IrType::Void);
                    } else {
                        self.builder.build_call_direct_with_type_args(
                            constructor_func_id,
                            arg_regs,
                            IrType::Void,
                            ctor_type_args,
                        );
                    }

                    // Transfer ownership: constructor args that are heap-allocated
                    // are now owned by the new object (stored in its fields).
                    // Remove them from drop tracking to prevent premature free.
                    for arg in args.iter() {
                        if let HirExprKind::Variable { symbol, .. } = &arg.kind {
                            if self.owned_heap_values.contains_key(symbol) {
                                self.owned_heap_values.remove(symbol);
                            }
                        }
                    }
                } else if let Some(ctor_fn) = self
                    .resolve_cross_module_constructor_by_name(actual_symbol_id, final_class_name)
                {
                    // Cross-module: the class's constructor isn't in this
                    // context's `constructor_map`/`constructor_name_map` but is
                    // resolvable by FQN (`<class>.new`) in the shared name map.
                    // Call it so the object is constructed rather than left with
                    // default-null fields.
                    let mut arg_regs: Vec<IrId> = Vec::with_capacity(args.len() + 1);
                    arg_regs.push(obj_ptr);
                    for a in args.iter() {
                        if let Some(reg) = self.lower_expression(a) {
                            arg_regs.push(reg);
                        }
                    }
                    self.builder
                        .build_call_direct(ctor_fn, arg_regs, IrType::Void);
                    for arg in args.iter() {
                        if let HirExprKind::Variable { symbol, .. } = &arg.kind {
                            self.owned_heap_values.remove(symbol);
                        }
                    }
                } else if let Some(ctor_key) =
                    self.cross_module_constructor_fqn_key(actual_symbol_id, final_class_name)
                {
                    // Cross-module, class compiles LATER in this import pass
                    // (cycle-breaker / retry ordering), so its constructor isn't
                    // resolvable by name yet either. Emit the call against a
                    // named forward-ref stub; `fixup_stale_cross_module_refs`
                    // redirects it to the real constructor by qualified name
                    // once every module is loaded. Silently emitting NO call
                    // here left the object zero-initialized.
                    let mut arg_regs: Vec<IrId> = Vec::with_capacity(args.len() + 1);
                    arg_regs.push(obj_ptr);
                    for a in args.iter() {
                        if let Some(reg) = self.lower_expression(a) {
                            arg_regs.push(reg);
                        }
                    }
                    let param_types: Vec<IrType> = arg_regs
                        .iter()
                        .map(|r| {
                            self.builder
                                .get_register_type(*r)
                                .unwrap_or(IrType::Ptr(Box::new(IrType::Void)))
                        })
                        .collect();
                    let stub_id =
                        self.register_stdlib_mir_forward_ref(&ctor_key, param_types, IrType::Void);
                    self.builder
                        .build_call_direct(stub_id, arg_regs, IrType::Void);
                    for arg in args.iter() {
                        if let HirExprKind::Variable { symbol, .. } = &arg.kind {
                            self.owned_heap_values.remove(symbol);
                        }
                    }
                }

                if let Some(class_name) = final_class_name {
                    self.register_class_hints
                        .insert(obj_ptr, class_name.to_string());
                }
                Some(obj_ptr)
            }

            HirExprKind::Unary { op, operand } => {
                // Handle increment/decrement operators specially
                match op {
                    HirUnaryOp::PostIncr
                    | HirUnaryOp::PreIncr
                    | HirUnaryOp::PostDecr
                    | HirUnaryOp::PreDecr => {
                        // For increment/decrement, we need to:
                        // 1. Load the current value
                        // 2. Compute new value (old ± 1)
                        // 3. Store the new value back
                        // 4. Return old value (post) or new value (pre)

                        let old_value = self.lower_expression(operand)?;
                        let one = self.builder.build_const(IrValue::I32(1))?;

                        let is_increment = matches!(op, HirUnaryOp::PostIncr | HirUnaryOp::PreIncr);
                        let new_value = if is_increment {
                            self.builder.build_binop(BinaryOp::Add, old_value, one)?
                        } else {
                            self.builder.build_binop(BinaryOp::Sub, old_value, one)?
                        };

                        // Register the new_value with its type
                        let result_type = self.convert_type(expr.ty);
                        let src_loc = self.convert_source_location(&expr.source_location);
                        if let Some(func) = self.builder.current_function_mut() {
                            func.locals.insert(
                                new_value,
                                crate::ir::IrLocal {
                                    name: format!("_incr{}", new_value.0),
                                    ty: result_type.clone(),
                                    mutable: false,
                                    source_location: src_loc.clone(),
                                    allocation: crate::ir::AllocationHint::Stack,
                                },
                            );
                        }

                        // Store the new value back to the operand
                        match &operand.kind {
                            HirExprKind::Variable { symbol, .. } => {
                                // Static field referenced bare: it lives in
                                // GLOBAL storage, not an SSA local — rebinding
                                // symbol_map silently dropped the write, so
                                // `staticCounter++` was a lost store (plain
                                // `x = x + 1` assignment already routed
                                // through build_store_global).
                                if let Some(&global_id) = self.global_symbol_map.get(symbol) {
                                    self.builder.build_store_global(global_id, new_value);
                                } else {
                                    // If we're inside a lambda with captured variables, also store back to environment
                                    if let Some(ref env_layout) = self.current_env_layout {
                                        if env_layout.find_field(*symbol).is_some() {
                                            // This is a captured variable - store it back to environment
                                            let env_ptr = IrId::new(0); // First parameter in lambda is environment pointer
                                            env_layout.store_field(
                                                &mut self.builder,
                                                env_ptr,
                                                *symbol,
                                                new_value,
                                            );
                                        }
                                    }

                                    self.symbol_map.insert(*symbol, new_value);
                                }
                            }
                            HirExprKind::Field { object, field } => {
                                // Field access (e.g., this.length++) — store new value back
                                // via GEP + Store, same as field assignment
                                if let Some(obj_reg) = self.lower_expression(object) {
                                    // Look up field index (with receiver-type disambiguation)
                                    let field_idx = self
                                        .field_index_map
                                        .get(field)
                                        .map(|&(_, idx)| idx)
                                        .or_else(|| {
                                            let field_name = self
                                                .symbol_table
                                                .get_symbol(*field)
                                                .map(|s| s.name)?;
                                            let receiver_ty = object.ty;
                                            self.resolve_field_index_by_name(
                                                field_name,
                                                receiver_ty,
                                            )
                                            .map(|(_, idx)| idx)
                                        });
                                    if let Some(idx) = field_idx {
                                        let idx_const =
                                            self.builder.build_const(IrValue::I32(idx as i32));
                                        if let Some(idx_reg) = idx_const {
                                            let field_ty = result_type.clone();
                                            let field_ptr = self.builder.build_gep(
                                                obj_reg,
                                                vec![idx_reg],
                                                field_ty.clone(),
                                            );
                                            if let Some(ptr) = field_ptr {
                                                self.builder.build_store(ptr, new_value);
                                            }
                                        }
                                    }
                                }
                            }
                            _ => {}
                        }

                        // Return appropriate value
                        let result_reg = match op {
                            HirUnaryOp::PostIncr | HirUnaryOp::PostDecr => old_value, // Post: return old value
                            HirUnaryOp::PreIncr | HirUnaryOp::PreDecr => new_value, // Pre: return new value
                            _ => unreachable!(),
                        };

                        Some(result_reg)
                    }
                    _ => {
                        // Handle other unary operators normally
                        let operand_reg = self.lower_expression(operand)?;
                        let result_reg = self
                            .builder
                            .build_unop(self.convert_unary_op(*op), operand_reg)?;

                        // Register the result with its type so Cranelift can find it
                        let result_type = self.convert_type(expr.ty);
                        let src_loc = self.convert_source_location(&expr.source_location);
                        if let Some(func) = self.builder.current_function_mut() {
                            func.locals.insert(
                                result_reg,
                                crate::ir::IrLocal {
                                    name: format!("_temp{}", result_reg.0),
                                    ty: result_type,
                                    mutable: false,
                                    source_location: src_loc,
                                    allocation: crate::ir::AllocationHint::Stack,
                                },
                            );
                        }

                        Some(result_reg)
                    }
                }
            }

            HirExprKind::Binary { op, lhs, rhs } => {
                // Handle short-circuit operators specially
                match op {
                    HirBinaryOp::And => return self.lower_logical_and(lhs, rhs),
                    HirBinaryOp::Or => return self.lower_logical_or(lhs, rhs),
                    HirBinaryOp::NullCoalesce => return self.lower_null_coalesce(lhs, rhs),
                    _ => {}
                }

                // Special handling for string concatenation with +
                if matches!(op, HirBinaryOp::Add) {
                    let lhs_type_raw = self.convert_type(lhs.ty);
                    let rhs_type_raw = self.convert_type(rhs.ty);

                    // Override types with resolved IR types for pattern-bound variables
                    let lhs_type = self.resolve_expr_ir_type(lhs, lhs_type_raw);
                    let rhs_type = self.resolve_expr_ir_type(rhs, rhs_type_raw);

                    let lhs_is_string = matches!(&lhs_type, IrType::String)
                        || matches!(&lhs_type, IrType::Ptr(inner) if matches!(inner.as_ref(), IrType::String));
                    let rhs_is_string = matches!(&rhs_type, IrType::String)
                        || matches!(&rhs_type, IrType::Ptr(inner) if matches!(inner.as_ref(), IrType::String));

                    // String concat chain detection: `prev + x` where `prev` is itself a
                    // Binary Add producing a string. The HIR type may erase to Dynamic, so we
                    // recursively inspect the LHS to detect string concat chains.
                    fn is_string_concat_chain(expr: &HirExpr) -> bool {
                        if let HirExprKind::Binary {
                            op: HirBinaryOp::Add,
                            lhs,
                            rhs,
                        } = &expr.kind
                        {
                            if matches!(&lhs.kind, HirExprKind::Literal(HirLiteral::String(_)))
                                || matches!(&rhs.kind, HirExprKind::Literal(HirLiteral::String(_)))
                            {
                                return true;
                            }
                            return is_string_concat_chain(lhs) || is_string_concat_chain(rhs);
                        }
                        false
                    }
                    let lhs_is_string = lhs_is_string || is_string_concat_chain(lhs);
                    let rhs_is_string = rhs_is_string || is_string_concat_chain(rhs);

                    if lhs_is_string || rhs_is_string {
                        // Lower both operands
                        let lhs_reg = self.lower_expression(lhs)?;
                        let rhs_reg = self.lower_expression(rhs)?;

                        // Use MIR register types (from runtime mapping) instead of HIR types,
                        // which may be unresolved generics (e.g. Ptr(Void) for Vec<Int>.length())
                        let lhs_mir_type = self
                            .builder
                            .get_register_type(lhs_reg)
                            .unwrap_or(lhs_type.clone());
                        let rhs_mir_type = self
                            .builder
                            .get_register_type(rhs_reg)
                            .unwrap_or(rhs_type.clone());

                        let lhs_is_string_mir = matches!(&lhs_mir_type, IrType::String)
                            || matches!(&lhs_mir_type, IrType::Ptr(inner) if matches!(inner.as_ref(), IrType::String))
                            // Defensive: if HIR type says String but MIR register is Ptr(Void)
                            // (e.g. extern/stdlib string returns), trust the HIR type
                            || (lhs_is_string && matches!(&lhs_mir_type, IrType::Ptr(inner) if matches!(inner.as_ref(), IrType::Void)));
                        let rhs_is_string_mir = matches!(&rhs_mir_type, IrType::String)
                            || matches!(&rhs_mir_type, IrType::Ptr(inner) if matches!(inner.as_ref(), IrType::String))
                            || (rhs_is_string
                                && matches!(&rhs_mir_type, IrType::Ptr(inner) if matches!(inner.as_ref(), IrType::Void)));

                        // Convert non-string operand to string if needed
                        // For class instances with toString(), call it directly at compile time
                        let lhs_str_val = if !lhs_is_string_mir {
                            if self.expr_is_value_type_expr(lhs) {
                                self.convert_value_type_to_string(lhs_reg)?
                            } else if let Some(reg) =
                                self.try_call_tostring(lhs_reg, self.resolve_expr_type_id(lhs))?
                            {
                                reg
                            } else {
                                self.convert_to_string_with_hint(
                                    lhs_reg,
                                    &lhs_mir_type,
                                    Some(lhs.ty),
                                )?
                            }
                        } else {
                            lhs_reg
                        };

                        let rhs_str_val = if !rhs_is_string_mir {
                            if self.expr_is_value_type_expr(rhs) {
                                self.convert_value_type_to_string(rhs_reg)?
                            } else if let Some(reg) =
                                self.try_call_tostring(rhs_reg, self.resolve_expr_type_id(rhs))?
                            {
                                reg
                            } else {
                                self.convert_to_string_with_hint(
                                    rhs_reg,
                                    &rhs_mir_type,
                                    Some(rhs.ty),
                                )?
                            }
                        } else {
                            rhs_reg
                        };

                        // String values are already pointers (*HaxeString):
                        // - string literals from haxe_string_literal return *mut HaxeString
                        // - conversion functions like int_to_string also return pointers
                        // Pass them directly to string_concat which expects *HaxeString args
                        let string_ptr_ty = IrType::Ptr(Box::new(IrType::String));
                        let concat_func_id = self.register_stdlib_mir_forward_ref(
                            "string_concat",
                            vec![string_ptr_ty.clone(), string_ptr_ty.clone()],
                            string_ptr_ty.clone(),
                        );

                        return self.builder.build_call_direct(
                            concat_func_id,
                            vec![lhs_str_val, rhs_str_val],
                            string_ptr_ty,
                        );
                    }
                }

                // String comparison: Eq/Ne/Lt/Le/Gt/Ge on strings need content comparison
                if matches!(
                    op,
                    HirBinaryOp::Eq
                        | HirBinaryOp::Ne
                        | HirBinaryOp::Lt
                        | HirBinaryOp::Le
                        | HirBinaryOp::Gt
                        | HirBinaryOp::Ge
                ) {
                    let lhs_type_raw = self.convert_type(lhs.ty);
                    let rhs_type_raw = self.convert_type(rhs.ty);
                    let lhs_type = self.resolve_expr_ir_type(lhs, lhs_type_raw);
                    let rhs_type = self.resolve_expr_ir_type(rhs, rhs_type_raw);

                    let lhs_is_string = matches!(&lhs_type, IrType::String)
                        || matches!(&lhs_type, IrType::Ptr(inner) if matches!(inner.as_ref(), IrType::String));
                    let rhs_is_string = matches!(&rhs_type, IrType::String)
                        || matches!(&rhs_type, IrType::Ptr(inner) if matches!(inner.as_ref(), IrType::String));

                    if lhs_is_string && rhs_is_string {
                        let lhs_reg = self.lower_expression(lhs)?;
                        let rhs_reg = self.lower_expression(rhs)?;
                        let string_ptr_ty = IrType::Ptr(Box::new(IrType::String));
                        let cmp_func = self.get_or_register_extern_function(
                            "haxe_string_compare",
                            vec![string_ptr_ty.clone(), string_ptr_ty.clone()],
                            IrType::I32,
                        );
                        let cmp_result = self.builder.build_call_direct(
                            cmp_func,
                            vec![lhs_reg, rhs_reg],
                            IrType::I32,
                        )?;
                        let zero = self.builder.build_const(IrValue::I32(0))?;
                        let cmp_op = match op {
                            HirBinaryOp::Eq => CompareOp::Eq,
                            HirBinaryOp::Ne => CompareOp::Ne,
                            HirBinaryOp::Lt => CompareOp::Lt,
                            HirBinaryOp::Le => CompareOp::Le,
                            HirBinaryOp::Gt => CompareOp::Gt,
                            HirBinaryOp::Ge => CompareOp::Ge,
                            _ => unreachable!(),
                        };
                        return self.builder.build_cmp(cmp_op, cmp_result, zero);
                    }
                }

                // @:derive(PartialEq) field-by-field equality for class instances
                if matches!(op, HirBinaryOp::Eq | HirBinaryOp::Ne) {
                    let class_sym = {
                        let type_table = self.type_table;
                        let lhs_sym = type_table.get(lhs.ty).and_then(|t| {
                            if let TypeKind::Class { symbol_id, .. } = &t.kind {
                                Some(*symbol_id)
                            } else {
                                None
                            }
                        });
                        let rhs_sym = type_table.get(rhs.ty).and_then(|t| {
                            if let TypeKind::Class { symbol_id, .. } = &t.kind {
                                Some(*symbol_id)
                            } else {
                                None
                            }
                        });
                        match (lhs_sym, rhs_sym) {
                            (Some(l), Some(r))
                                if l == r && self.derive_partial_eq_classes.contains(&l) =>
                            {
                                Some(l)
                            }
                            _ => None,
                        }
                    };
                    if let Some(sym) = class_sym {
                        return self.lower_derived_equality(op, lhs, rhs, sym);
                    }

                    // Null<prim> equality: route through null-guarded unboxing
                    // compares (the nullable is a box; a raw cmp compared the
                    // box pointer). `x == null` keeps the pointer compare.
                    if !matches!(&lhs.kind, HirExprKind::Null)
                        && !matches!(&rhs.kind, HirExprKind::Null)
                    {
                        if let Some(res) = self.lower_nullable_prim_eq(op, lhs, rhs) {
                            return Some(res);
                        }
                    }
                }

                // @:derive(PartialOrd) lexicographic ordering for class instances
                if matches!(
                    op,
                    HirBinaryOp::Lt | HirBinaryOp::Le | HirBinaryOp::Gt | HirBinaryOp::Ge
                ) {
                    let class_sym = {
                        let type_table = self.type_table;
                        let lhs_sym = type_table.get(lhs.ty).and_then(|t| {
                            if let TypeKind::Class { symbol_id, .. } = &t.kind {
                                Some(*symbol_id)
                            } else {
                                None
                            }
                        });
                        let rhs_sym = type_table.get(rhs.ty).and_then(|t| {
                            if let TypeKind::Class { symbol_id, .. } = &t.kind {
                                Some(*symbol_id)
                            } else {
                                None
                            }
                        });
                        match (lhs_sym, rhs_sym) {
                            (Some(l), Some(r))
                                if l == r && self.derive_partial_ord_classes.contains(&l) =>
                            {
                                Some(l)
                            }
                            _ => None,
                        }
                    };
                    if let Some(sym) = class_sym {
                        return self.lower_derived_ordering(op, lhs, rhs, sym);
                    }
                }

                // Dynamic arithmetic: when both HIR types are actually Dynamic, check actual
                // MIR register types after lowering to determine if values are truly boxed
                // DynamicValue pointers vs raw concrete values (e.g., class field access on
                // Dynamic-typed object, or integer arithmetic result with Dynamic HIR type).
                // NOTE: We check the actual HIR TypeKind, not just MIR Ptr(Void), because
                // class types also lower to Ptr(Void) but are NOT boxed DynamicValues.
                {
                    let (lhs_is_dyn, rhs_is_dyn) = {
                        let type_table = self.type_table;
                        let lhs_dyn = type_table
                            .get(lhs.ty)
                            .map(|t| matches!(t.kind, TypeKind::Dynamic))
                            .unwrap_or(false);
                        let rhs_dyn = type_table
                            .get(rhs.ty)
                            .map(|t| matches!(t.kind, TypeKind::Dynamic))
                            .unwrap_or(false);
                        (lhs_dyn, rhs_dyn)
                    };

                    if lhs_is_dyn && rhs_is_dyn {
                        let is_supported = matches!(
                            op,
                            HirBinaryOp::Add
                                | HirBinaryOp::Sub
                                | HirBinaryOp::Mul
                                | HirBinaryOp::Div
                                | HirBinaryOp::Mod
                                | HirBinaryOp::Eq
                                | HirBinaryOp::Lt
                                | HirBinaryOp::Gt
                                | HirBinaryOp::Le
                                | HirBinaryOp::Ge
                                | HirBinaryOp::Ne
                        );
                        if is_supported {
                            // Lower operands first, then check actual register types
                            let lhs_reg = self.lower_expression(lhs)?;
                            let rhs_reg = self.lower_expression(rhs)?;

                            // SIMD vector short-circuit: @:coreType abstracts like SIMD4f
                            // have HIR type Dynamic but lower to Vector register type.
                            // Emit VectorBinOp directly for vector+vector arithmetic.
                            {
                                let lhs_rty = self.builder.get_register_type(lhs_reg);
                                let rhs_rty = self.builder.get_register_type(rhs_reg);
                                let lhs_is_vec = matches!(&lhs_rty, Some(IrType::Vector { .. }));
                                let rhs_is_vec = matches!(&rhs_rty, Some(IrType::Vector { .. }));
                                if lhs_is_vec || rhs_is_vec {
                                    let vec_ty = if lhs_is_vec {
                                        lhs_rty.unwrap()
                                    } else {
                                        rhs_rty.unwrap()
                                    };
                                    let bin_op = match op {
                                        HirBinaryOp::Add => BinaryOp::Add,
                                        HirBinaryOp::Sub => BinaryOp::Sub,
                                        HirBinaryOp::Mul => BinaryOp::Mul,
                                        HirBinaryOp::Div => BinaryOp::Div,
                                        _ => {
                                            // Unsupported vector op in Dynamic path; fall
                                            // through to the regular Dynamic handling (will
                                            // likely produce invalid code — covered by other
                                            // type checks).
                                            return self.builder.build_vector_binop(
                                                BinaryOp::Add,
                                                lhs_reg,
                                                rhs_reg,
                                                vec_ty,
                                            );
                                        }
                                    };
                                    return self
                                        .builder
                                        .build_vector_binop(bin_op, lhs_reg, rhs_reg, vec_ty);
                                }
                            }

                            // Determine if each operand is actually a boxed DynamicValue:
                            // - Variables: use boxed_dynamic_symbols tracking (handles lambda
                            //   params that have Dynamic type but hold raw i64 values)
                            // - Other expressions: check actual MIR register type — concrete
                            //   types (I32, F64, etc.) are definitely not boxed pointers
                            let is_concrete_ir_type = |ty: &IrType| -> bool {
                                matches!(
                                    ty,
                                    IrType::I8
                                        | IrType::I16
                                        | IrType::I32
                                        | IrType::I64
                                        | IrType::U8
                                        | IrType::U16
                                        | IrType::U32
                                        | IrType::U64
                                        | IrType::F32
                                        | IrType::F64
                                        | IrType::Bool
                                )
                            };

                            let is_concrete = |reg| {
                                self.builder
                                    .get_register_type(reg)
                                    .as_ref()
                                    .map(|t| is_concrete_ir_type(t))
                                    .unwrap_or(false)
                            };
                            let lhs_boxed = match &lhs.kind {
                                HirExprKind::Variable { symbol, .. } => {
                                    self.boxed_dynamic_symbols.contains(symbol)
                                }
                                _ => {
                                    let ty = self.builder.get_register_type(lhs_reg);
                                    !ty.as_ref().map(|t| is_concrete_ir_type(t)).unwrap_or(false)
                                }
                            };
                            let rhs_boxed = match &rhs.kind {
                                HirExprKind::Variable { symbol, .. } => {
                                    self.boxed_dynamic_symbols.contains(symbol)
                                }
                                _ => {
                                    let ty = self.builder.get_register_type(rhs_reg);
                                    !ty.as_ref().map(|t| is_concrete_ir_type(t)).unwrap_or(false)
                                }
                            };
                            // Eq/Ne on two pointer-shaped Dynamic operands cannot be decided
                            // from the register type: a boxed value and a lambda parameter
                            // holding a raw i64 are BOTH Ptr(Void) here, and neither is
                            // tracked. Comparing them as pointers is wrong for boxes, and
                            // unboxing is wrong for raw values -- so hand both to the runtime,
                            // which validates each side before dereferencing it and falls back
                            // to the raw value when it is not a box.
                            if matches!(op, HirBinaryOp::Eq | HirBinaryOp::Ne)
                                && !is_concrete(lhs_reg)
                                && !is_concrete(rhs_reg)
                            {
                                let ptr_void = IrType::Ptr(Box::new(IrType::Void));
                                let eq_func = self.get_or_register_extern_function(
                                    "haxe_dynamic_equals",
                                    vec![ptr_void.clone(), ptr_void],
                                    IrType::Bool,
                                );
                                let eq = self.builder.build_call_direct(
                                    eq_func,
                                    vec![lhs_reg, rhs_reg],
                                    IrType::Bool,
                                )?;
                                if matches!(op, HirBinaryOp::Eq) {
                                    return Some(eq);
                                }
                                let f = self.builder.build_const(IrValue::Bool(false))?;
                                return self.builder.build_cmp(CompareOp::Eq, eq, f);
                            }

                            if lhs_boxed && rhs_boxed {
                                // Both are actually boxed DynamicValue pointers.
                                // Unbox to f64, operate, rebox.
                                let ptr_void = IrType::Ptr(Box::new(IrType::Void));
                                let unbox_func = self.get_or_register_extern_function(
                                    "haxe_unbox_float_ptr",
                                    vec![ptr_void.clone()],
                                    IrType::F64,
                                );
                                let lhs_f64 = self.builder.build_call_direct(
                                    unbox_func,
                                    vec![lhs_reg],
                                    IrType::F64,
                                )?;
                                let rhs_f64 = self.builder.build_call_direct(
                                    unbox_func,
                                    vec![rhs_reg],
                                    IrType::F64,
                                )?;

                                let is_comparison = matches!(
                                    op,
                                    HirBinaryOp::Eq
                                        | HirBinaryOp::Ne
                                        | HirBinaryOp::Lt
                                        | HirBinaryOp::Le
                                        | HirBinaryOp::Gt
                                        | HirBinaryOp::Ge
                                );

                                if is_comparison {
                                    // Eq/Ne go through the runtime, which dispatches on the
                                    // box tag: numeric across Int/Float, by value for Bool
                                    // and String, by reference otherwise. Unboxing to f64
                                    // would equate 1 and "1" and read a string as a double.
                                    // Ordering comparisons stay numeric.
                                    if matches!(op, HirBinaryOp::Eq | HirBinaryOp::Ne) {
                                        let ptr_void = IrType::Ptr(Box::new(IrType::Void));
                                        let eq_func = self.get_or_register_extern_function(
                                            "haxe_dynamic_equals",
                                            vec![ptr_void.clone(), ptr_void],
                                            IrType::Bool,
                                        );
                                        let eq = self.builder.build_call_direct(
                                            eq_func,
                                            vec![lhs_reg, rhs_reg],
                                            IrType::Bool,
                                        )?;
                                        if matches!(op, HirBinaryOp::Eq) {
                                            return Some(eq);
                                        }
                                        let f = self.builder.build_const(IrValue::Bool(false))?;
                                        return self.builder.build_cmp(CompareOp::Eq, eq, f);
                                    }
                                    let cmp_op = match op {
                                        HirBinaryOp::Lt => CompareOp::Lt,
                                        HirBinaryOp::Le => CompareOp::Le,
                                        HirBinaryOp::Gt => CompareOp::Gt,
                                        HirBinaryOp::Ge => CompareOp::Ge,
                                        _ => unreachable!(),
                                    };
                                    return self.builder.build_cmp(cmp_op, lhs_f64, rhs_f64);
                                } else {
                                    let bin_op = match op {
                                        HirBinaryOp::Add => BinaryOp::FAdd,
                                        HirBinaryOp::Sub => BinaryOp::FSub,
                                        HirBinaryOp::Mul => BinaryOp::FMul,
                                        HirBinaryOp::Div => BinaryOp::FDiv,
                                        HirBinaryOp::Mod => BinaryOp::FRem,
                                        _ => unreachable!(),
                                    };
                                    let result_f64 =
                                        self.builder.build_binop(bin_op, lhs_f64, rhs_f64)?;

                                    let box_func = self.get_or_register_extern_function(
                                        "haxe_box_float_ptr",
                                        vec![IrType::F64],
                                        IrType::Ptr(Box::new(IrType::U8)),
                                    );
                                    return self.builder.build_call_direct(
                                        box_func,
                                        vec![result_f64],
                                        IrType::Ptr(Box::new(IrType::U8)),
                                    );
                                }
                            }

                            // At least one operand is NOT boxed — it's a raw concrete value
                            // with Dynamic HIR type (e.g., class field access, integer arithmetic).
                            // Handle mixed boxed+concrete and fully concrete cases.
                            let mut effective_lhs = lhs_reg;
                            let mut effective_rhs = rhs_reg;

                            // Unbox boxed side if mixed (one boxed, one concrete)
                            if lhs_boxed && !rhs_boxed {
                                let rhs_ty = self
                                    .builder
                                    .get_register_type(rhs_reg)
                                    .unwrap_or(IrType::I64);
                                let ptr_void = IrType::Ptr(Box::new(IrType::Void));
                                if matches!(rhs_ty, IrType::F32 | IrType::F64) {
                                    let unbox = self.get_or_register_extern_function(
                                        "haxe_unbox_float_ptr",
                                        vec![ptr_void],
                                        IrType::F64,
                                    );
                                    effective_lhs = self.builder.build_call_direct(
                                        unbox,
                                        vec![lhs_reg],
                                        IrType::F64,
                                    )?;
                                } else {
                                    let unbox = self.get_or_register_extern_function(
                                        "haxe_unbox_int_ptr",
                                        vec![ptr_void],
                                        IrType::I64,
                                    );
                                    effective_lhs = self.builder.build_call_direct(
                                        unbox,
                                        vec![lhs_reg],
                                        IrType::I64,
                                    )?;
                                }
                            } else if !lhs_boxed && rhs_boxed {
                                let lhs_ty = self
                                    .builder
                                    .get_register_type(lhs_reg)
                                    .unwrap_or(IrType::I64);
                                let ptr_void = IrType::Ptr(Box::new(IrType::Void));
                                if matches!(lhs_ty, IrType::F32 | IrType::F64) {
                                    let unbox = self.get_or_register_extern_function(
                                        "haxe_unbox_float_ptr",
                                        vec![ptr_void],
                                        IrType::F64,
                                    );
                                    effective_rhs = self.builder.build_call_direct(
                                        unbox,
                                        vec![rhs_reg],
                                        IrType::F64,
                                    )?;
                                } else {
                                    let unbox = self.get_or_register_extern_function(
                                        "haxe_unbox_int_ptr",
                                        vec![ptr_void],
                                        IrType::I64,
                                    );
                                    effective_rhs = self.builder.build_call_direct(
                                        unbox,
                                        vec![rhs_reg],
                                        IrType::I64,
                                    )?;
                                }
                            }

                            // Use actual register types for type coercion
                            let eff_lhs_ty = self
                                .builder
                                .get_register_type(effective_lhs)
                                .unwrap_or(IrType::I64);
                            let eff_rhs_ty = self
                                .builder
                                .get_register_type(effective_rhs)
                                .unwrap_or(IrType::I64);

                            let l_is_int = matches!(
                                eff_lhs_ty,
                                IrType::I8
                                    | IrType::I16
                                    | IrType::I32
                                    | IrType::I64
                                    | IrType::U8
                                    | IrType::U16
                                    | IrType::U32
                                    | IrType::U64
                            );
                            let r_is_int = matches!(
                                eff_rhs_ty,
                                IrType::I8
                                    | IrType::I16
                                    | IrType::I32
                                    | IrType::I64
                                    | IrType::U8
                                    | IrType::U16
                                    | IrType::U32
                                    | IrType::U64
                            );
                            let l_is_float = matches!(eff_lhs_ty, IrType::F32 | IrType::F64);
                            let r_is_float = matches!(eff_rhs_ty, IrType::F32 | IrType::F64);

                            if l_is_int && r_is_float {
                                effective_lhs = self.builder.build_cast(
                                    effective_lhs,
                                    eff_lhs_ty.clone(),
                                    IrType::F64,
                                )?;
                            }
                            if r_is_int && l_is_float {
                                effective_rhs = self.builder.build_cast(
                                    effective_rhs,
                                    eff_rhs_ty.clone(),
                                    IrType::F64,
                                )?;
                            }
                            if matches!(op, HirBinaryOp::Div) && l_is_int && r_is_int {
                                effective_lhs = self.builder.build_cast(
                                    effective_lhs,
                                    eff_lhs_ty.clone(),
                                    IrType::F64,
                                )?;
                                effective_rhs = self.builder.build_cast(
                                    effective_rhs,
                                    eff_rhs_ty.clone(),
                                    IrType::F64,
                                )?;
                            }

                            let result_reg = match self.convert_binary_op_to_mir(*op) {
                                MirBinaryOp::Binary(bin_op) => self.builder.build_binop(
                                    bin_op,
                                    effective_lhs,
                                    effective_rhs,
                                )?,
                                MirBinaryOp::Compare(cmp_op) => {
                                    self.builder
                                        .build_cmp(cmp_op, effective_lhs, effective_rhs)?
                                }
                            };
                            return Some(result_reg);
                        }
                    }

                    // Mixed Dynamic + concrete arithmetic:
                    // One operand is Dynamic, the other is a concrete primitive.
                    // Distinguish boxed DynamicValue* (Ptr(U8)) from type-erased raw values
                    // (Ptr(Void)). Only unbox Ptr(U8); cast Ptr(Void) to integer.
                    if (lhs_is_dyn || rhs_is_dyn) && !(lhs_is_dyn && rhs_is_dyn) {
                        let is_arith = matches!(
                            op,
                            HirBinaryOp::Add
                                | HirBinaryOp::Sub
                                | HirBinaryOp::Mul
                                | HirBinaryOp::Div
                                | HirBinaryOp::Mod
                                | HirBinaryOp::Eq
                                | HirBinaryOp::Lt
                                | HirBinaryOp::Gt
                                | HirBinaryOp::Le
                                | HirBinaryOp::Ge
                                | HirBinaryOp::Ne
                        );
                        if is_arith {
                            let mut lhs_reg = self.lower_expression(lhs)?;
                            let mut rhs_reg = self.lower_expression(rhs)?;

                            // Determine concrete type from the non-Dynamic side
                            let concrete_ty = if lhs_is_dyn {
                                self.builder
                                    .get_register_type(rhs_reg)
                                    .unwrap_or(IrType::I32)
                            } else {
                                self.builder
                                    .get_register_type(lhs_reg)
                                    .unwrap_or(IrType::I32)
                            };

                            let is_float = matches!(concrete_ty, IrType::F32 | IrType::F64);

                            // Coerce the Dynamic-side register to match the concrete type.
                            // Uses haxe_coerce_dynamic_to_int/float which handles both
                            // boxed DynamicValue* and type-erased raw integers at runtime.
                            let coerce_dyn = |s: &mut Self, reg: IrId| -> Option<IrId> {
                                let reg_ty = s.builder.get_register_type(reg);
                                // Already concrete? No coercion needed.
                                if reg_ty
                                    .as_ref()
                                    .map(|t| {
                                        matches!(
                                            t,
                                            IrType::I32
                                                | IrType::I64
                                                | IrType::F32
                                                | IrType::F64
                                                | IrType::Bool
                                        )
                                    })
                                    .unwrap_or(false)
                                {
                                    return Some(reg);
                                }
                                let ptr_void = IrType::Ptr(Box::new(IrType::Void));
                                if is_float {
                                    let f = s.get_or_register_extern_function(
                                        "haxe_coerce_dynamic_to_float",
                                        vec![ptr_void],
                                        IrType::F64,
                                    );
                                    s.builder.build_call_direct(f, vec![reg], IrType::F64)
                                } else {
                                    let f = s.get_or_register_extern_function(
                                        "haxe_coerce_dynamic_to_int",
                                        vec![ptr_void],
                                        IrType::I64,
                                    );
                                    let v =
                                        s.builder.build_call_direct(f, vec![reg], IrType::I64)?;
                                    Some(
                                        s.builder
                                            .build_cast(v, IrType::I64, concrete_ty.clone())
                                            .unwrap_or(v),
                                    )
                                }
                            };

                            if lhs_is_dyn {
                                lhs_reg = coerce_dyn(self, lhs_reg)?;
                            } else {
                                rhs_reg = coerce_dyn(self, rhs_reg)?;
                            }

                            let mir_op = self.convert_binary_op_to_mir(*op);
                            let result_reg = match mir_op {
                                MirBinaryOp::Binary(arith_op) => {
                                    self.builder.build_binop(arith_op, lhs_reg, rhs_reg)?
                                }
                                MirBinaryOp::Compare(cmp_op) => {
                                    self.builder.build_cmp(cmp_op, lhs_reg, rhs_reg)?
                                }
                            };
                            return Some(result_reg);
                        }
                    }
                }

                let mut lhs_reg = self.lower_expression(lhs)?;
                let mut rhs_reg = self.lower_expression(rhs)?;

                // Auto-unbox Null<T> operands when the binop is arithmetic on
                // primitives. Without this, `Null<Int> + Null<Int>` (or
                // `Int + Null<Int>`) operates on raw DynamicValue* pointers,
                // producing garbage / crashes. Unbox using each operand's
                // OWN inner primitive type — not expr.ty, which may itself
                // still be Optional after typechecking unified to Null<Int>.
                if matches!(
                    op,
                    HirBinaryOp::Add
                        | HirBinaryOp::Sub
                        | HirBinaryOp::Mul
                        | HirBinaryOp::Div
                        | HirBinaryOp::Mod
                ) {
                    if self.is_optional_primitive(lhs.ty) {
                        if let Some(inner) = self.optional_inner_type(lhs.ty) {
                            if let Some(unboxed) = self.maybe_unbox_optional(lhs_reg, lhs.ty, inner)
                            {
                                lhs_reg = unboxed;
                            }
                        }
                    }
                    if self.is_optional_primitive(rhs.ty) {
                        if let Some(inner) = self.optional_inner_type(rhs.ty) {
                            if let Some(unboxed) = self.maybe_unbox_optional(rhs_reg, rhs.ty, inner)
                            {
                                rhs_reg = unboxed;
                            }
                        }
                    }
                }

                let lhs_type = self.convert_type(lhs.ty);
                let rhs_type = self.convert_type(rhs.ty);

                // Type coercion for mixed int/float operations
                // When one operand is float and the other is int, cast int to float
                let lhs_is_int = matches!(
                    lhs_type,
                    IrType::I8
                        | IrType::I16
                        | IrType::I32
                        | IrType::I64
                        | IrType::U8
                        | IrType::U16
                        | IrType::U32
                        | IrType::U64
                );
                let rhs_is_int = matches!(
                    rhs_type,
                    IrType::I8
                        | IrType::I16
                        | IrType::I32
                        | IrType::I64
                        | IrType::U8
                        | IrType::U16
                        | IrType::U32
                        | IrType::U64
                );
                let lhs_is_float = matches!(lhs_type, IrType::F32 | IrType::F64);
                let rhs_is_float = matches!(rhs_type, IrType::F32 | IrType::F64);

                // Cast int to float when mixing types (promotes to F64)
                if lhs_is_int && rhs_is_float {
                    lhs_reg = self
                        .builder
                        .build_cast(lhs_reg, lhs_type.clone(), IrType::F64)?;
                }
                if rhs_is_int && lhs_is_float {
                    rhs_reg = self
                        .builder
                        .build_cast(rhs_reg, rhs_type.clone(), IrType::F64)?;
                }

                // Special handling for division: Haxe always returns Float from division
                // If operands are integers, convert them to float first
                if matches!(op, HirBinaryOp::Div) && lhs_is_int && rhs_is_int {
                    lhs_reg = self
                        .builder
                        .build_cast(lhs_reg, lhs_type.clone(), IrType::F64)?;
                    rhs_reg = self
                        .builder
                        .build_cast(rhs_reg, rhs_type.clone(), IrType::F64)?;
                }

                // Check if result is a SIMD vector type — emit VectorBinOp directly (zero overhead)
                // We check operand types (from register_types) rather than convert_type(expr.ty)
                // because @:coreType abstracts may not resolve correctly through convert_type.
                let lhs_actual_type = self
                    .builder
                    .get_register_type(lhs_reg)
                    .unwrap_or(IrType::I64);
                let rhs_actual_type = self
                    .builder
                    .get_register_type(rhs_reg)
                    .unwrap_or(IrType::I64);
                let result_type = if lhs_actual_type.is_vector() {
                    lhs_actual_type.clone()
                } else if rhs_actual_type.is_vector() {
                    rhs_actual_type.clone()
                } else {
                    self.convert_type(expr.ty)
                };
                let result_reg = if result_type.is_vector() {
                    let bin_op = match op {
                        HirBinaryOp::Add => BinaryOp::Add,
                        HirBinaryOp::Sub => BinaryOp::Sub,
                        HirBinaryOp::Mul => BinaryOp::Mul,
                        HirBinaryOp::Div => BinaryOp::Div,
                        _ => {
                            debug!("Unsupported vector binary op: {:?}", op);
                            return None;
                        }
                    };
                    self.builder.build_vector_binop(
                        bin_op,
                        lhs_reg,
                        rhs_reg,
                        result_type.clone(),
                    )?
                } else {
                    match self.convert_binary_op_to_mir(*op) {
                        MirBinaryOp::Binary(bin_op) => {
                            self.builder.build_binop(bin_op, lhs_reg, rhs_reg)?
                        }
                        MirBinaryOp::Compare(cmp_op) => {
                            self.builder.build_cmp(cmp_op, lhs_reg, rhs_reg)?
                        }
                    }
                };
                let src_loc = self.convert_source_location(&expr.source_location);
                if let Some(func) = self.builder.current_function_mut() {
                    func.locals.insert(
                        result_reg,
                        crate::ir::IrLocal {
                            name: format!("_temp{}", result_reg.0),
                            ty: result_type,
                            mutable: false,
                            source_location: src_loc,
                            allocation: crate::ir::AllocationHint::Stack,
                        },
                    );
                }

                Some(result_reg)
            }

            HirExprKind::Cast {
                expr,
                target,
                is_safe,
            } => {
                // Collapse double-cast pattern from abstract methods:
                // Cast(safe, target=Int) { Cast(unsafe, target=Dynamic) { This/Variable } }
                // This pattern comes from `(cast this : Int)` syntax in abstract methods.
                // The inner `cast this` (→Dynamic) is a no-op for abstract types, and the
                // outer `(... : Int)` extracts the underlying value. Collapse to just the
                // innermost expression.
                if let HirExprKind::Cast {
                    expr: inner_expr,
                    target: inner_target,
                    is_safe: false,
                } = &expr.kind
                {
                    let inner_target_is_dynamic = {
                        let type_table = self.type_table;
                        type_table
                            .get(*inner_target)
                            .map(|t| matches!(t.kind, TypeKind::Dynamic))
                            .unwrap_or(false)
                    };
                    let inner_source_is_abstract = {
                        let type_table = self.type_table;
                        type_table
                            .get(inner_expr.ty)
                            .map(|t| matches!(t.kind, TypeKind::Abstract { .. }))
                            .unwrap_or(false)
                    };
                    if inner_target_is_dynamic && inner_source_is_abstract {
                        // The whole double-cast is an identity: abstract → dynamic → underlying
                        // Just lower the innermost expression directly
                        return self.lower_expression(inner_expr);
                    }
                }

                // Check if target is an abstract type with @:from conversion rules
                // This generalizes the SIMD4f path to work with any abstract
                if let Some(converted) = self.try_abstract_from_cast(expr, *target) {
                    return Some(converted);
                }

                let from_type = self.convert_type(expr.ty);
                let to_type = self.convert_type(*target);

                // For abstract types, the cast between abstract and underlying is a no-op
                // (they share the same MIR type).
                // BUT: for safe casts between class types, we must NOT skip — even though
                // both are Ptr(Void), the runtime needs to verify the class hierarchy.
                if from_type == to_type && !*is_safe {
                    return self.lower_expression(expr);
                }
                if from_type == to_type && *is_safe {
                    // Same MIR type — but for class/interface types, fall through to
                    // the type-kind handlers for runtime verification and fat pointer wrapping.
                    let type_table = self.type_table;
                    let src_kind = type_table.get(expr.ty).map(|t| &t.kind).cloned();
                    let tgt_kind = type_table.get(*target).map(|t| &t.kind).cloned();
                    let needs_runtime_check = matches!(
                        (&src_kind, &tgt_kind),
                        (Some(TypeKind::Class { .. }), Some(TypeKind::Class { .. }))
                            | (
                                Some(TypeKind::Class { .. }),
                                Some(TypeKind::Interface { .. })
                            )
                            | (
                                Some(TypeKind::Interface { .. }),
                                Some(TypeKind::Class { .. })
                            )
                            | (
                                Some(TypeKind::Interface { .. }),
                                Some(TypeKind::Interface { .. })
                            )
                    );
                    if !needs_runtime_check {
                        return self.lower_expression(expr);
                    }
                    // Fall through to Class→Class safe cast handler
                }

                // For `cast this` in abstract methods: the source is Abstract (resolves to
                // underlying e.g. I32) but target is Dynamic (Ptr(U8)). This is NOT a real
                // Dynamic boxing — it's just the Haxe syntax for extracting the underlying
                // value. Skip the cast to avoid reinterpreting integers as pointers.
                // Also handle the case where `cast this` target is Dynamic and the source
                // is a primitive type (the underlying type of the abstract).
                {
                    let type_table = self.type_table;
                    let source_kind = type_table.get(expr.ty).map(|t| t.kind.clone());
                    let target_kind = type_table.get(*target).map(|t| t.kind.clone());

                    let source_is_abstract =
                        matches!(&source_kind, Some(TypeKind::Abstract { .. }));
                    let target_is_dynamic = matches!(&target_kind, Some(TypeKind::Dynamic));

                    if source_is_abstract && target_is_dynamic {
                        return self.lower_expression(expr);
                    }

                    // Also handle: source is concrete primitive, target is Dynamic,
                    // inside an abstract method body where `this` was registered as
                    // the underlying type. `cast this` produces
                    // Cast(This(underlying_type) → Dynamic), which is a no-op.
                    if target_is_dynamic
                        && !*is_safe
                        && matches!(
                            &source_kind,
                            Some(TypeKind::Int)
                                | Some(TypeKind::Float)
                                | Some(TypeKind::Bool)
                                | Some(TypeKind::String)
                        )
                        && matches!(&expr.kind, HirExprKind::This)
                    {
                        return self.lower_expression(expr);
                    }
                }

                // For unsafe casts, emit a direct cast instruction (no runtime check).
                // For safe casts, we need to fall through to the type-specific
                // handling even when MIR types match (e.g., Class→Class are both
                // Ptr(U8) but need runtime hierarchy verification).
                if !*is_safe {
                    let value_reg = self.lower_expression(expr)?;
                    return self.builder.build_cast(value_reg, from_type, to_type);
                }

                // Safe cast: resolve at compile time based on source/target type kinds
                let source_kind = {
                    let type_table = self.type_table;
                    type_table.get(expr.ty).map(|ti| ti.kind.clone())
                };
                let target_kind = {
                    let type_table = self.type_table;
                    type_table.get(*target).map(|ti| ti.kind.clone())
                };

                match (&source_kind, &target_kind) {
                    // Primitive-to-primitive: always succeeds, emit static conversion
                    (Some(TypeKind::Int), Some(TypeKind::Float))
                    | (Some(TypeKind::Float), Some(TypeKind::Int))
                    | (Some(TypeKind::Int), Some(TypeKind::Bool))
                    | (Some(TypeKind::Bool), Some(TypeKind::Int))
                    | (Some(TypeKind::Float), Some(TypeKind::Bool))
                    | (Some(TypeKind::Bool), Some(TypeKind::Float)) => {
                        let value_reg = self.lower_expression(expr)?;
                        self.builder.build_cast(value_reg, from_type, to_type)
                    }

                    // Dynamic → primitive type: unbox the DynamicValue
                    // But skip unboxing if the source register is already a raw
                    // primitive (e.g., `cast this` in enum abstract methods where
                    // `this` is the underlying Int value, not a boxed DynamicValue*).
                    (Some(TypeKind::Dynamic), Some(TypeKind::Int)) => {
                        let value_reg = self.lower_expression(expr)?;
                        let reg_ty = self.builder.get_register_type(value_reg);
                        if matches!(
                            reg_ty,
                            Some(IrType::I32) | Some(IrType::I64) | Some(IrType::U32)
                        ) {
                            // Already a raw integer — cast to I32 if needed
                            if matches!(reg_ty, Some(IrType::I32)) {
                                Some(value_reg)
                            } else {
                                self.builder
                                    .build_cast(value_reg, reg_ty.unwrap(), IrType::I32)
                            }
                        } else {
                            let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                            let unbox_id = self.get_or_register_extern_function(
                                "haxe_unbox_int_ptr",
                                vec![ptr_u8],
                                IrType::I32,
                            );
                            self.builder
                                .build_call_direct(unbox_id, vec![value_reg], IrType::I32)
                        }
                    }
                    (Some(TypeKind::Dynamic), Some(TypeKind::Float)) => {
                        let value_reg = self.lower_expression(expr)?;
                        let reg_ty = self.builder.get_register_type(value_reg);
                        if matches!(reg_ty, Some(IrType::F32) | Some(IrType::F64)) {
                            if matches!(reg_ty, Some(IrType::F64)) {
                                Some(value_reg)
                            } else {
                                self.builder.build_cast(value_reg, IrType::F32, IrType::F64)
                            }
                        } else {
                            let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                            let unbox_id = self.get_or_register_extern_function(
                                "haxe_unbox_float_ptr",
                                vec![ptr_u8],
                                IrType::F64,
                            );
                            self.builder
                                .build_call_direct(unbox_id, vec![value_reg], IrType::F64)
                        }
                    }
                    (Some(TypeKind::Dynamic), Some(TypeKind::Bool)) => {
                        let value_reg = self.lower_expression(expr)?;
                        let reg_ty = self.builder.get_register_type(value_reg);
                        if matches!(reg_ty, Some(IrType::Bool)) {
                            Some(value_reg)
                        } else {
                            let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                            let unbox_id = self.get_or_register_extern_function(
                                "haxe_unbox_bool_ptr",
                                vec![ptr_u8],
                                IrType::Bool,
                            );
                            self.builder
                                .build_call_direct(unbox_id, vec![value_reg], IrType::Bool)
                        }
                    }
                    // Dynamic → non-primitive type: runtime downcast (null on failure)
                    (Some(TypeKind::Dynamic), _)
                        if !matches!(&target_kind, Some(TypeKind::Dynamic)) =>
                    {
                        let value_reg = self.lower_expression(expr)?;
                        // DynamicValue boxing uses TAST TypeId (value_ty.as_raw()),
                        // so downcast comparison must also use TAST TypeId for consistency.
                        let type_id_const = self
                            .builder
                            .build_const(IrValue::I64(target.as_raw() as i64))?;
                        let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                        let downcast_id = self.get_or_register_extern_function(
                            "haxe_std_downcast",
                            vec![ptr_u8.clone(), IrType::I64],
                            ptr_u8.clone(),
                        );
                        self.builder.build_call_direct(
                            downcast_id,
                            vec![value_reg, type_id_const],
                            ptr_u8,
                        )
                    }

                    // Concrete → Dynamic: box the value
                    (_, Some(TypeKind::Dynamic)) => {
                        let value_reg = self.lower_expression(expr)?;
                        self.maybe_box_value(value_reg, expr.ty, *target)
                            .or(Some(value_reg))
                    }

                    // Cross-module `(class : Interface)` where the SOURCE type
                    // didn't resolve to a Class in this MIR-lowering context
                    // (arrives Unknown/Placeholder — the class metadata wasn't
                    // imported here). The typechecker still promoted this to a
                    // Class→Interface cast (Bug #2's TAST promotion), and the
                    // value register IS a raw class object we can wrap: recover
                    // the source class SymbolId by NAME from the inner
                    // expression. Without this, the cast is a silent no-op and
                    // the raw pointer flows into an interface slot → later
                    // vtable dispatch reads garbage (the nue `new LlamaArch()` →
                    // ArchBuilder registration path).
                    (
                        src,
                        Some(TypeKind::Interface {
                            symbol_id: tgt_sym, ..
                        }),
                    ) if !matches!(
                        src,
                        Some(TypeKind::Class { .. }) | Some(TypeKind::Interface { .. })
                    ) && (self.recover_cast_src_class_by_name(expr).is_some()
                        || self.new_arg_class_fqn(expr).is_some()) =>
                    {
                        let tgt_sym = *tgt_sym;
                        let src_sym = self.recover_cast_src_class_by_name(expr);
                        let value_reg = self.lower_expression(expr)?;
                        if self.interface_wrapped_args.contains(&value_reg) {
                            Some(value_reg)
                        } else if let Some(src_sym) = src_sym.filter(|s| {
                            self.interface_vtables.contains_key(&(*s, tgt_sym))
                                || self.interface_method_names.contains_key(&tgt_sym)
                        }) {
                            // In-context (or forwardable) class symbol available.
                            match self.wrap_in_interface_fat_ptr(value_reg, src_sym, tgt_sym) {
                                Some(fat_ptr) => {
                                    self.interface_wrapped_args.insert(fat_ptr);
                                    Some(fat_ptr)
                                }
                                None => Some(value_reg),
                            }
                        } else if let Some(class_fqn) = self.new_arg_class_fqn(expr) {
                            // Fully-imported class not present here: wrap by NAME
                            // via forward-ref dispatch thunks (order-independent).
                            match self
                                .wrap_new_class_as_interface_by_name(value_reg, &class_fqn, tgt_sym)
                            {
                                Some(fat_ptr) => {
                                    self.interface_wrapped_args.insert(fat_ptr);
                                    Some(fat_ptr)
                                }
                                None => Some(value_reg),
                            }
                        } else {
                            Some(value_reg)
                        }
                    }

                    // Class → Class safe cast: runtime downcast via object header
                    (
                        Some(TypeKind::Class {
                            symbol_id: src_sym, ..
                        }),
                        Some(TypeKind::Class {
                            symbol_id: tgt_sym, ..
                        }),
                    ) => {
                        let src_sym = *src_sym;
                        let tgt_sym = *tgt_sym;
                        let value_reg = self.lower_expression(expr)?;

                        if src_sym == tgt_sym || self.is_subclass_of(expr.ty, *target) {
                            // Same class or upcast: always succeeds
                            self.builder.build_cast(value_reg, from_type, to_type)
                        } else {
                            // Downcast or unrelated: runtime check via object header.
                            // haxe_safe_downcast_class reads the type id from offset 0,
                            // walks the hierarchy, and returns obj_ptr or null.
                            // Use `runtime_type_id` directly so the value here
                            // matches the value the New handler stored in the
                            // header (both sides are now the deterministic
                            // name-hash, no transform on either end).
                            let _ = tgt_sym;
                            let target_type_id = self.runtime_type_id(*target) as i64;
                            let type_id_const =
                                self.builder.build_const(IrValue::I64(target_type_id))?;
                            let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                            let downcast_func = self.get_or_register_extern_function(
                                "haxe_safe_downcast_class",
                                vec![ptr_u8.clone(), IrType::I64],
                                ptr_u8.clone(),
                            );
                            self.builder.build_call_direct(
                                downcast_func,
                                vec![value_reg, type_id_const],
                                ptr_u8,
                            )
                        }
                    }

                    // Class → Interface safe cast: wrap in fat pointer if implements, null if not
                    (
                        Some(TypeKind::Class {
                            symbol_id: src_sym, ..
                        }),
                        Some(TypeKind::Interface {
                            symbol_id: tgt_sym, ..
                        }),
                    ) => {
                        let src_sym = *src_sym;
                        let tgt_sym = *tgt_sym;

                        // Wrap in fat pointer if the class implements the iface.
                        // Cannot defer to the Let handler because SafeCast's
                        // result type is the interface type, so Let would see
                        // Interface→Interface instead of Class→Interface.
                        //
                        // Presence in `interface_vtables` is the fast check, but
                        // cross-module the eager (class, iface) pair is often
                        // missing here (built lazily). `wrap_in_interface_fat_ptr`
                        // has a name-based lazy-build fallback, so ATTEMPT the
                        // wrap whenever the interface is known to have methods
                        // in this context; only fall back to null when we can't
                        // build a vtable at all (class genuinely doesn't
                        // implement it). Returning null on a valid cross-module
                        // implementer was the SIGSEGV root for a plain
                        // `param:Interface` arg — the raw class pointer flowed
                        // through and dispatch read past the object.
                        let known_implements =
                            self.interface_vtables.contains_key(&(src_sym, tgt_sym))
                                || self.interface_method_names.contains_key(&tgt_sym);
                        if known_implements {
                            let value_reg = self.lower_expression(expr)?;
                            match self.wrap_in_interface_fat_ptr(value_reg, src_sym, tgt_sym) {
                                Some(fat_ptr) => {
                                    self.interface_wrapped_args.insert(fat_ptr);
                                    Some(fat_ptr)
                                }
                                None => Some(value_reg),
                            }
                        } else {
                            // Not known to implement at compile time — return null
                            self.builder.build_const(IrValue::Null)
                        }
                    }

                    // Interface → Class safe cast: extract raw obj + downcast
                    (
                        Some(TypeKind::Interface { .. }),
                        Some(TypeKind::Class {
                            symbol_id: tgt_sym, ..
                        }),
                    ) => {
                        let tgt_sym = *tgt_sym;
                        let value_reg = self.lower_expression(expr)?;

                        // Load raw object pointer from fat pointer offset 0
                        let raw_obj = self
                            .builder
                            .build_load(value_reg, IrType::Ptr(Box::new(IrType::U8)))?;

                        // Downcast via object header type_id check.
                        // Match the New handler: object header stores the
                        // raw `runtime_type_id` (deterministic name-hash).
                        let _ = tgt_sym;
                        let target_type_id = self.runtime_type_id(*target) as i64;
                        let type_id_const =
                            self.builder.build_const(IrValue::I64(target_type_id))?;
                        let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                        let downcast_func = self.get_or_register_extern_function(
                            "haxe_safe_downcast_class",
                            vec![ptr_u8.clone(), IrType::I64],
                            ptr_u8.clone(),
                        );
                        self.builder.build_call_direct(
                            downcast_func,
                            vec![raw_obj, type_id_const],
                            ptr_u8,
                        )
                    }

                    // Interface → Interface cast.
                    //
                    // Three sub-cases:
                    //
                    // - Same iface or source has ≥ target's methods: the
                    //   existing fat pointer's vtable already covers the
                    //   target's slots (interface-method order is stable
                    //   per declared interface), so a pass-through cast is
                    //   safe.
                    //
                    // - Source has fewer methods than target (e.g.
                    //   `Module` → `CausalLanguageModel`): the source fat
                    //   pointer was built with only the source iface's
                    //   slots. Reading the target's extra slots from the
                    //   source fat pointer runs past the allocated method
                    //   slots into garbage → call-indirect to a bogus
                    //   address → SIGSEGV.
                    //
                    //   Rebuild via `haxe_iface_fat_ptr_build`: extract
                    //   the obj_ptr from the source fat pointer, read the
                    //   class type_id from the object header, look up the
                    //   per-(class, target_iface) vtable populated in
                    //   `__vtable_init__`, and allocate a fresh fat
                    //   pointer with the correct method slots.
                    (
                        Some(TypeKind::Interface {
                            symbol_id: src_iface_sym,
                            ..
                        }),
                        Some(TypeKind::Interface {
                            symbol_id: tgt_iface_sym,
                            ..
                        }),
                    ) => {
                        let src_iface_sym = *src_iface_sym;
                        let tgt_iface_sym = *tgt_iface_sym;
                        // Pass-through is only correct when the target's method
                        // slots are a PREFIX of the source's (same names, same
                        // order) — i.e. the source vtable already holds each
                        // target method at the same index. A method-COUNT test
                        // is wrong for a sub-interface with its OWN methods:
                        // `Module` (forward, parameters) → `CausalLanguageModel`
                        // (forwardIds, resetCache) has equal counts but disjoint
                        // slots, so pass-through would dispatch `resetCache`
                        // through `Module`'s `parameters` slot. Rebuild unless
                        // the names line up as a prefix.
                        let target_is_prefix = match (
                            self.interface_method_names.get(&src_iface_sym),
                            self.interface_method_names.get(&tgt_iface_sym),
                        ) {
                            (Some(src_names), Some(tgt_names)) => {
                                tgt_names.len() <= src_names.len()
                                    && tgt_names.iter().zip(src_names.iter()).all(|(t, s)| t == s)
                            }
                            // Same interface (src == tgt) is trivially a prefix.
                            _ => src_iface_sym == tgt_iface_sym,
                        };
                        if src_iface_sym == tgt_iface_sym || target_is_prefix {
                            // Source vtable covers the target's slots at the same
                            // indices — existing pass-through behaviour is fine.
                            let value_reg = self.lower_expression(expr)?;
                            self.builder.build_cast(value_reg, from_type, to_type)
                        } else {
                            // Target has methods source doesn't —
                            // rebuild the fat pointer with the target's
                            // method slots via the runtime registry.
                            let value_reg = self.lower_expression(expr)?;
                            let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                            let src_ptr = {
                                let src_ty = self
                                    .builder
                                    .get_register_type(value_reg)
                                    .unwrap_or(IrType::I64);
                                if matches!(src_ty, IrType::Ptr(_)) {
                                    value_reg
                                } else {
                                    self.builder.build_bitcast(value_reg, ptr_u8.clone())?
                                }
                            };
                            let obj_ptr = self.builder.build_load(src_ptr, ptr_u8.clone())?;
                            let target_iface_type_id = self
                                .deterministic_iface_or_enum_type_id(tgt_iface_sym, "iface")
                                .unwrap_or(tgt_iface_sym.as_raw())
                                as i32;
                            let iface_tid_reg = self
                                .builder
                                .build_const(IrValue::I32(target_iface_type_id))?;
                            let rebuild_fn = self.get_or_register_extern_function(
                                "haxe_iface_fat_ptr_build",
                                vec![ptr_u8.clone(), IrType::I32],
                                ptr_u8.clone(),
                            );
                            self.builder.build_call_direct(
                                rebuild_fn,
                                vec![obj_ptr, iface_tid_reg],
                                ptr_u8,
                            )
                        }
                    }

                    // Fallback: emit raw cast (same as unsafe)
                    _ => {
                        let value_reg = self.lower_expression(expr)?;
                        self.builder.build_cast(value_reg, from_type, to_type)
                    }
                }
            }

            HirExprKind::TypeCheck { expr, expected } => {
                // (expr is Type) — compile-time type check for statically-typed code
                let source_kind = {
                    let type_table = self.type_table;
                    type_table.get(expr.ty).map(|ti| ti.kind.clone())
                };
                let target_kind = {
                    let type_table = self.type_table;
                    type_table.get(*expected).map(|ti| ti.kind.clone())
                };

                // For statically-typed code, resolve at compile time
                let result = match (&source_kind, &target_kind) {
                    // Dynamic source: runtime type check via haxe_std_is
                    (Some(TypeKind::Dynamic), _) => {
                        let value_reg = self.lower_expression(expr)?;
                        let rt_type_id = self.runtime_type_id(*expected);
                        let type_id_const =
                            self.builder.build_const(IrValue::I64(rt_type_id as i64))?;
                        let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                        let is_func_id = self.get_or_register_extern_function(
                            "haxe_std_is",
                            vec![ptr_u8, IrType::I64],
                            IrType::Bool,
                        );
                        self.builder.build_call_direct(
                            is_func_id,
                            vec![value_reg, type_id_const],
                            IrType::Bool,
                        )
                    }
                    // Same type kind → always true (but null check needed for refs)
                    _ if expr.ty == *expected => {
                        // Exact same type → true
                        self.builder.build_const(IrValue::Bool(true))
                    }
                    // Primitive type checks
                    (Some(TypeKind::Int), Some(TypeKind::Int))
                    | (Some(TypeKind::Float), Some(TypeKind::Float))
                    | (Some(TypeKind::Bool), Some(TypeKind::Bool))
                    | (Some(TypeKind::String), Some(TypeKind::String)) => {
                        self.builder.build_const(IrValue::Bool(true))
                    }
                    // Cross-primitive: always false
                    (Some(TypeKind::Int), Some(TypeKind::Float))
                    | (Some(TypeKind::Float), Some(TypeKind::Int))
                    | (Some(TypeKind::Int), Some(TypeKind::String))
                    | (Some(TypeKind::String), Some(TypeKind::Int))
                    | (Some(TypeKind::Float), Some(TypeKind::String))
                    | (Some(TypeKind::String), Some(TypeKind::Float))
                    | (Some(TypeKind::Bool), Some(TypeKind::Int))
                    | (Some(TypeKind::Int), Some(TypeKind::Bool))
                    | (Some(TypeKind::Bool), Some(TypeKind::Float))
                    | (Some(TypeKind::Float), Some(TypeKind::Bool)) => {
                        self.builder.build_const(IrValue::Bool(false))
                    }
                    // Class-to-class: check class hierarchy at compile time
                    (
                        Some(TypeKind::Class {
                            symbol_id: src_sym, ..
                        }),
                        Some(TypeKind::Class {
                            symbol_id: tgt_sym, ..
                        }),
                    ) => {
                        let src_sym = *src_sym;
                        let tgt_sym = *tgt_sym;
                        if src_sym == tgt_sym {
                            // Same class → true
                            let _value = self.lower_expression(expr);
                            self.builder.build_const(IrValue::Bool(true))
                        } else if self.is_subclass_of(expr.ty, *expected) {
                            // Source is subclass of target (upcast) → always true
                            let _value = self.lower_expression(expr);
                            self.builder.build_const(IrValue::Bool(true))
                        } else if self.is_subclass_of(*expected, expr.ty) {
                            // Target is subclass of source (downcast) →
                            // runtime check via object header. Both the
                            // header (written by the New handler) and this
                            // comparison use `runtime_type_id` directly,
                            // which is the deterministic name-hash. No
                            // ±1000 transform on either side.
                            let value_reg = self.lower_expression(expr)?;
                            let target_type_id = self.runtime_type_id(*expected) as i64;
                            let type_id_const =
                                self.builder.build_const(IrValue::I64(target_type_id))?;
                            let ptr_void = IrType::Ptr(Box::new(IrType::Void));
                            let is_func = self.get_or_register_extern_function(
                                "haxe_object_is_instance",
                                vec![ptr_void, IrType::I64],
                                IrType::I64,
                            );
                            let result_i64 = self.builder.build_call_direct(
                                is_func,
                                vec![value_reg, type_id_const],
                                IrType::I64,
                            )?;
                            // Convert i64 (0/1) to Bool
                            let zero = self.builder.build_const(IrValue::I64(0))?;
                            self.builder.build_cmp(CompareOp::Ne, result_i64, zero)
                        } else {
                            // Unrelated classes → false
                            let _value = self.lower_expression(expr);
                            self.builder.build_const(IrValue::Bool(false))
                        }
                    }
                    // Class -> Interface: runtime check against registered interface impl map.
                    (Some(TypeKind::Class { .. }), Some(TypeKind::Interface { .. })) => {
                        let value_reg = self.lower_expression(expr)?;
                        let target_type_id = self.runtime_type_id(*expected) as i64;
                        let type_id_const =
                            self.builder.build_const(IrValue::I64(target_type_id))?;
                        let ptr_void = IrType::Ptr(Box::new(IrType::Void));
                        let is_func = self.get_or_register_extern_function(
                            "haxe_object_is_instance",
                            vec![ptr_void, IrType::I64],
                            IrType::I64,
                        );
                        let result_i64 = self.builder.build_call_direct(
                            is_func,
                            vec![value_reg, type_id_const],
                            IrType::I64,
                        )?;
                        let zero = self.builder.build_const(IrValue::I64(0))?;
                        self.builder.build_cmp(CompareOp::Ne, result_i64, zero)
                    }
                    // Class vs primitive or other unrelated → false
                    (Some(TypeKind::Class { .. }), Some(TypeKind::Int))
                    | (Some(TypeKind::Class { .. }), Some(TypeKind::Float))
                    | (Some(TypeKind::Class { .. }), Some(TypeKind::Bool))
                    | (Some(TypeKind::Class { .. }), Some(TypeKind::String))
                    | (Some(TypeKind::Int), Some(TypeKind::Class { .. }))
                    | (Some(TypeKind::Float), Some(TypeKind::Class { .. }))
                    | (Some(TypeKind::Bool), Some(TypeKind::Class { .. }))
                    | (Some(TypeKind::String), Some(TypeKind::Class { .. })) => {
                        let _value = self.lower_expression(expr);
                        self.builder.build_const(IrValue::Bool(false))
                    }
                    // Fallback: lower the expression (for side effects) and return true
                    // (trust static types — proper runtime checks need object headers)
                    _ => {
                        let _value = self.lower_expression(expr);
                        self.builder.build_const(IrValue::Bool(true))
                    }
                };
                result
            }

            HirExprKind::If {
                condition,
                then_expr,
                else_expr,
            } => self.lower_conditional_typed(condition, then_expr, else_expr, Some(expr.ty)),

            HirExprKind::Block(block) => self.lower_block_expr(block),

            HirExprKind::Lambda {
                params,
                body,
                captures,
            } => {
                debug!(
                    "Lowering lambda with {} params, {} captures",
                    params.len(),
                    captures.len()
                );
                self.lower_lambda(params, body, captures, expr.ty)
            }

            HirExprKind::MethodReference {
                receiver,
                method_symbol,
            } => self.lower_method_reference(receiver, *method_symbol),

            HirExprKind::Array { elements } => self.lower_array_literal(elements, expr.ty),

            HirExprKind::Map { entries } => self.lower_map_literal(entries),

            HirExprKind::ObjectLiteral { fields } => self.lower_object_literal(fields, expr.ty),

            HirExprKind::ArrayComprehension { .. } => {
                // Array comprehensions are desugared to loops
                self.add_error(
                    "Array comprehensions not yet implemented in MIR",
                    expr.source_location,
                );
                None
            }

            HirExprKind::StringInterpolation { parts } => self.lower_string_interpolation(parts),

            HirExprKind::This => {
                // 'this' is typically passed as first parameter
                self.symbol_map.get(&SymbolId::from_raw(0)).copied()
            }

            HirExprKind::Super => {
                // 'super' should only appear in constructor super calls, which are handled
                // specially in lower_constructor_body. If we reach here, it's likely being
                // used incorrectly (e.g., super.method() which isn't supported yet)
                // eprintln!("WARNING: HirExprKind::Super encountered in expression lowering");
                // eprintln!("  This might be super.field or super.method() which isn't implemented yet");
                // For now, treat it like 'this' (same object, but calling parent methods)
                self.symbol_map.get(&SymbolId::from_raw(0)).copied()
            }

            HirExprKind::Null => self.builder.build_null(),

            HirExprKind::Untyped(inner) => {
                // Untyped expressions bypass type checking
                self.lower_expression(inner)
            }

            HirExprKind::InlineCode { target, code, args } => {
                // Platform-specific inline code (__c__, __js__, etc.)
                self.lower_inline_code(target, code, args)
            }

            HirExprKind::TryCatch {
                try_expr,
                catch_handlers,
                finally_expr,
            } => {
                // Exception handling via setjmp/longjmp (expression form).
                // try { body } catch (e) { handler } finally { cleanup }

                // Snapshot the pre-try value of every variable the try/catch
                // bodies touch, so we can build merge phis at the continuation.
                // The catch runs INSTEAD of the try, so without a merge the
                // continuation keeps whichever branch lowered LAST (the catch),
                // and a var the catch modified leaks even when nothing threw
                // (`try { ok = true } catch { ok = false }` always saw `false`).
                // Mirrors lower_if_statement's then/else merge.
                let mut tc_tracked: std::collections::BTreeSet<SymbolId> =
                    std::collections::BTreeSet::new();
                self.collect_referenced_variables_in_expr(try_expr, &mut tc_tracked);
                for h in catch_handlers {
                    self.collect_referenced_variables_in_expr(&h.body, &mut tc_tracked);
                }
                let mut tc_pre: BTreeMap<SymbolId, (IrId, IrType)> = BTreeMap::new();
                for s in &tc_tracked {
                    if self.is_parameter_symbol(s) {
                        continue;
                    }
                    if let Some(&reg) = self.symbol_map.get(s) {
                        let ty = self
                            .builder
                            .current_function()
                            .and_then(|f| f.locals.get(&reg).map(|l| l.ty.clone()));
                        if let Some(ty) = ty {
                            tc_pre.insert(*s, (reg, ty));
                        }
                    }
                }
                // (exit_block, {var -> value}) captured at every path that reaches
                // the continuation, used to build the merge phis below.
                let mut tc_exits: Vec<(IrBlockId, BTreeMap<SymbolId, IrId>)> = Vec::new();

                let normal_path_block = self.builder.create_block()?;
                let landing_pad_block = self.builder.create_block()?;
                let continuation_block = self.builder.create_block()?;

                // --- Setup: push handler and call _setjmp ---
                let push_fn = self.get_or_register_extern_function(
                    "rayzor_exception_push_handler",
                    vec![],
                    IrType::Ptr(Box::new(IrType::Void)),
                );
                let jmp_buf = self.builder.build_call_direct(
                    push_fn,
                    vec![],
                    IrType::Ptr(Box::new(IrType::Void)),
                );

                let setjmp_fn = self.get_or_register_extern_function(
                    "_setjmp",
                    vec![IrType::Ptr(Box::new(IrType::Void))],
                    IrType::I32,
                );
                let jmp_buf_reg = jmp_buf
                    .unwrap_or_else(|| self.builder.build_const(IrValue::I64(0)).expect("const"));
                let setjmp_result =
                    self.builder
                        .build_call_direct(setjmp_fn, vec![jmp_buf_reg], IrType::I32);

                let zero = self.builder.build_const(IrValue::I32(0)).expect("const");
                let setjmp_reg = setjmp_result.unwrap_or(zero);
                let cmp = self.builder.build_cmp(CompareOp::Eq, setjmp_reg, zero)?;
                self.builder
                    .build_cond_branch(cmp, normal_path_block, landing_pad_block);

                // --- normal_path: execute try body ---
                self.builder.switch_to_block(normal_path_block);
                self.lower_expression(try_expr);

                let pop_fn = self.get_or_register_extern_function(
                    "rayzor_exception_pop_handler",
                    vec![],
                    IrType::Void,
                );
                self.builder.build_call_direct(pop_fn, vec![], IrType::Void);

                if let Some(finally_body) = &finally_expr {
                    self.lower_expression(finally_body);
                }

                // Capture the try-path's values before they leave for the merge.
                if !self.is_terminated() {
                    if let Some(blk) = self.builder.current_block() {
                        tc_exits.push((blk, self.capture_tracked_values(&tc_pre)));
                    }
                }
                self.builder.build_branch(continuation_block);

                // Catch bodies run INSTEAD of the try, so reset every tracked var
                // to its pre-try value before lowering them.
                for (s, (reg, _)) in &tc_pre {
                    self.symbol_map.insert(*s, *reg);
                }

                // --- landing_pad: exception was thrown ---
                self.builder.switch_to_block(landing_pad_block);

                let pop_fn2 = self.get_or_register_extern_function(
                    "rayzor_exception_pop_handler",
                    vec![],
                    IrType::Void,
                );
                self.builder
                    .build_call_direct(pop_fn2, vec![], IrType::Void);

                let get_exc_fn = self.get_or_register_extern_function(
                    "rayzor_get_exception",
                    vec![],
                    IrType::I64,
                );
                let exception_id = self
                    .builder
                    .build_call_direct(get_exc_fn, vec![], IrType::I64)
                    .unwrap_or_else(|| self.builder.build_const(IrValue::I64(0)).expect("const"));

                let get_exc_type_fn = self.get_or_register_extern_function(
                    "rayzor_get_exception_type_id",
                    vec![],
                    IrType::I32,
                );
                let exc_type_id = self
                    .builder
                    .build_call_direct(get_exc_type_fn, vec![], IrType::I32)
                    .unwrap_or_else(|| self.builder.build_const(IrValue::I32(0)).expect("const"));

                // Type-based dispatch across catch handlers
                if !catch_handlers.is_empty() {
                    let mut next_test_block: Option<IrBlockId> = None;

                    for (i, handler) in catch_handlers.iter().enumerate() {
                        let catch_type_kind = {
                            let type_table = self.type_table;
                            type_table
                                .get(handler.exception_type)
                                .map(|t| t.kind.clone())
                        };
                        let is_dynamic =
                            matches!(catch_type_kind, Some(crate::tast::TypeKind::Dynamic));

                        let catch_body_block = match self.builder.create_block() {
                            Some(b) => b,
                            None => return None,
                        };

                        if let Some(test_block) = next_test_block {
                            self.builder.switch_to_block(test_block);
                        }

                        if is_dynamic || i == catch_handlers.len() - 1 {
                            self.builder.build_branch(catch_body_block);
                            next_test_block = None;
                        } else {
                            let expected_type_id = self.runtime_type_id(handler.exception_type);
                            let expected_const = self
                                .builder
                                .build_const(IrValue::I32(expected_type_id as i32))
                                .expect("const");

                            // Use polymorphic matching for class types (walks inheritance),
                            // exact match for primitives
                            let is_class_type = matches!(
                                catch_type_kind,
                                Some(crate::tast::TypeKind::Class { .. })
                            );

                            let type_match = if is_class_type {
                                let match_fn = self.get_or_register_extern_function(
                                    "rayzor_exception_type_matches",
                                    vec![IrType::I32, IrType::I32],
                                    IrType::I32,
                                );
                                let result = self
                                    .builder
                                    .build_call_direct(
                                        match_fn,
                                        vec![exc_type_id, expected_const],
                                        IrType::I32,
                                    )
                                    .unwrap_or_else(|| {
                                        self.builder.build_const(IrValue::I32(0)).expect("const")
                                    });
                                let zero_val =
                                    self.builder.build_const(IrValue::I32(0)).expect("const");
                                self.builder
                                    .build_cmp(CompareOp::Ne, result, zero_val)
                                    .expect("cmp")
                            } else {
                                self.builder
                                    .build_cmp(CompareOp::Eq, exc_type_id, expected_const)
                                    .expect("cmp")
                            };

                            let next_block = self.builder.create_block().expect("create block");
                            self.builder.build_cond_branch(
                                type_match,
                                catch_body_block,
                                next_block,
                            );
                            next_test_block = Some(next_block);
                        }

                        // --- catch body ---
                        self.builder.switch_to_block(catch_body_block);
                        // Each catch is mutually exclusive and runs in place of
                        // the try — reset tracked vars to pre-try values first.
                        for (s, (reg, _)) in &tc_pre {
                            self.symbol_map.insert(*s, *reg);
                        }
                        self.symbol_map.insert(handler.exception_var, exception_id);
                        self.lower_expression(&handler.body);

                        if let Some(finally_body) = &finally_expr {
                            self.lower_expression(finally_body);
                        }
                        if !self.is_terminated() {
                            if let Some(blk) = self.builder.current_block() {
                                tc_exits.push((blk, self.capture_tracked_values(&tc_pre)));
                            }
                        }
                        self.builder.build_branch(continuation_block);
                    }

                    // Fallthrough if no catch matched (exception unhandled here):
                    // tracked vars keep their pre-try values.
                    if let Some(fallthrough_block) = next_test_block {
                        self.builder.switch_to_block(fallthrough_block);
                        for (s, (reg, _)) in &tc_pre {
                            self.symbol_map.insert(*s, *reg);
                        }
                        if let Some(finally_body) = &finally_expr {
                            self.lower_expression(finally_body);
                        }
                        if !self.is_terminated() {
                            if let Some(blk) = self.builder.current_block() {
                                tc_exits.push((blk, self.capture_tracked_values(&tc_pre)));
                            }
                        }
                        self.builder.build_branch(continuation_block);
                    }
                } else {
                    // No catch clauses: the landing pad keeps pre-try values.
                    for (s, (reg, _)) in &tc_pre {
                        self.symbol_map.insert(*s, *reg);
                    }
                    if let Some(finally_body) = &finally_expr {
                        self.lower_expression(finally_body);
                    }
                    if !self.is_terminated() {
                        if let Some(blk) = self.builder.current_block() {
                            tc_exits.push((blk, self.capture_tracked_values(&tc_pre)));
                        }
                    }
                    self.builder.build_branch(continuation_block);
                }

                // --- continuation: merge the tracked vars across all paths ---
                self.builder.switch_to_block(continuation_block);
                for (s, (pre_reg, ty)) in &tc_pre {
                    let mut incomings: Vec<(IrBlockId, IrId)> = Vec::new();
                    for (blk, vals) in &tc_exits {
                        incomings.push((*blk, vals.get(s).copied().unwrap_or(*pre_reg)));
                    }
                    if incomings.is_empty() {
                        continue;
                    }
                    let first = incomings[0].1;
                    if incomings.iter().all(|(_, v)| *v == first) {
                        // Every path carries the same value — no phi needed.
                        self.symbol_map.insert(*s, first);
                        continue;
                    }
                    if let Some(phi_reg) = self.builder.build_phi(continuation_block, ty.clone()) {
                        for (blk, val) in &incomings {
                            self.builder
                                .add_phi_incoming(continuation_block, phi_reg, *blk, *val);
                        }
                        if let Some(func) = self.builder.current_function_mut() {
                            if let Some(local) = func.locals.get(pre_reg).cloned() {
                                func.locals.insert(
                                    phi_reg,
                                    crate::ir::IrLocal {
                                        name: format!("{}_tcphi", local.name),
                                        ty: ty.clone(),
                                        mutable: true,
                                        source_location: local.source_location,
                                        allocation: crate::ir::AllocationHint::Register,
                                    },
                                );
                            }
                        }
                        self.symbol_map.insert(*s, phi_reg);
                    }
                }
                None // try/catch as statement has no return value
            }

            _ => {
                self.add_error("Unsupported expression type in MIR", expr.source_location);
                None
            }
        };

        // debug!("lower_expression result: {:?}", result);
        result
    }
}
