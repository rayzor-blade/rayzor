//! Field access: reads, assignment targets, and the class-typed fast path.

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

mod access;
mod lvalue;

impl<'a> HirToMirContext<'a> {
    /// Direct field access for class objects without Dynamic unboxing.
    /// Used when we know the object is a raw pointer (e.g., from StringMap<Point>.get())
    /// but TAST thinks it's Dynamic because the type parameter wasn't resolved.
    pub(crate) fn lower_field_access_for_class(
        &mut self,
        obj: IrId,
        field: SymbolId,
        field_ty: TypeId,
    ) -> Option<IrId> {
        let field_name = self
            .symbol_table
            .get_symbol(field)
            .and_then(|s| self.string_interner.get(s.name))
            .unwrap_or("<unknown>");

        debug!(
            "[lower_field_access_for_class] field_name='{}', field={:?}",
            field_name, field
        );

        // field_index_map is consulted FIRST: a user-defined class field must GEP
        // directly, or a name collision with the stdlib (a user `length` field
        // dispatched to array_length()) wins. Stdlib dispatch is the fallback.
        let field_found_in_class = self.field_index_map.contains_key(&field) || {
            // Also check by name.
            if let Some(sym) = self.symbol_table.get_symbol(field) {
                self.field_index_map.iter().any(|(s, _)| {
                    self.symbol_table.get_symbol(*s).map(|si| si.name) == Some(sym.name)
                })
            } else {
                false
            }
        };

        if !field_found_in_class {
            // Stdlib dispatch (e.g. Array.length on a Dynamic-typed Array): only
            // stdlib objects reach here, and their fields are runtime functions.
            debug!(
                "[lower_field_access_for_class] field '{}' not in field_index_map, trying stdlib",
                field_name
            );
            let common_stdlib_types = [
                crate::tast::TypeKind::Array {
                    element_type: TypeId::from_raw(0),
                },
                crate::tast::TypeKind::String,
            ];

            for ref_kind in &common_stdlib_types {
                let matching_type_id = {
                    let type_table = self.type_table;
                    let mut found = None;
                    for (type_id, type_info) in type_table.iter() {
                        let matches = match (&type_info.kind, ref_kind) {
                            (
                                crate::tast::TypeKind::Array { .. },
                                crate::tast::TypeKind::Array { .. },
                            ) => true,
                            (crate::tast::TypeKind::String, crate::tast::TypeKind::String) => true,
                            _ => false,
                        };
                        if matches {
                            found = Some(type_id);
                            break;
                        }
                    }
                    found
                };

                if let Some(class_ty) = matching_type_id {
                    if let Some((_class, _method, runtime_call)) =
                        self.get_stdlib_runtime_info(field, class_ty, None, None)
                    {
                        let runtime_func = runtime_call.runtime_name;
                        debug!(
                            "[lower_field_access_for_class] Found stdlib property! runtime_func={}",
                            runtime_func
                        );
                        let param_types = vec![IrType::Ptr(Box::new(IrType::Void))];

                        let result_type =
                            if !runtime_call.needs_out_param && runtime_call.has_return {
                                if let Some(mir_func) = self
                                    .builder
                                    .module
                                    .functions
                                    .iter()
                                    .find(|(_, f)| f.name == runtime_func)
                                    .map(|(_, f)| f)
                                {
                                    mir_func.signature.return_type.clone()
                                } else {
                                    let actual_field_type = self
                                        .symbol_table
                                        .get_symbol(field)
                                        .map(|s| s.type_id)
                                        .unwrap_or(field_ty);
                                    let field_kind = {
                                        let type_table = self.type_table;
                                        type_table.get(actual_field_type).map(|t| t.kind.clone())
                                    };
                                    match field_kind {
                                        Some(crate::tast::TypeKind::Int) => IrType::I64,
                                        Some(crate::tast::TypeKind::Float) => IrType::F64,
                                        Some(crate::tast::TypeKind::Bool) => IrType::Bool,
                                        _ => IrType::I64,
                                    }
                                }
                            } else {
                                self.convert_type(field_ty)
                            };

                        // A stdlib MIR wrapper (e.g. `array_length` wrapping
                        // `haxe_array_length`) must go through a MIR forward-ref:
                        // registering it as an extern import makes defining its
                        // body in the backend invalid.
                        let runtime_func_id = if runtime_call.is_mir_wrapper {
                            self.register_stdlib_mir_forward_ref(
                                &runtime_func,
                                param_types,
                                result_type.clone(),
                            )
                        } else {
                            self.get_or_register_extern_function(
                                &runtime_func,
                                param_types,
                                result_type.clone(),
                            )
                        };

                        return self.builder.build_call_direct(
                            runtime_func_id,
                            vec![obj],
                            result_type,
                        );
                    }
                }
            }
        }

        debug!(
            "[lower_field_access_for_class] Using field_index_map for field '{}'",
            field_name
        );

        // The field type comes from the symbol table, not the passed-in field_ty,
        // which may be Dynamic from unresolved type parameters.
        let (resolved_class_ty, field_index, actual_field_type) = match self
            .field_index_map
            .get(&field)
        {
            Some(&(class_ty, idx)) => {
                let sym_type = self
                    .symbol_table
                    .get_symbol(field)
                    .map(|s| s.type_id)
                    .unwrap_or(field_ty);
                (class_ty, idx, sym_type)
            }
            None => {
                // Fall back to a by-name lookup instead of the SymbolId.
                let field_name = self.symbol_table.get_symbol(field).map(|s| s.name)?;

                let field_name_str = self.string_interner.get(field_name).unwrap_or("<unknown>");
                debug!(
                    "[lower_field_access_for_class] Looking up field '{}' ({:?}) by name in field_index_map ({} entries)",
                    field_name_str,
                    field,
                    self.field_index_map.len()
                );

                let mut found = None;
                for (sym, &(class_ty, idx)) in &self.field_index_map {
                    if let Some(sym_info) = self.symbol_table.get_symbol(*sym) {
                        if sym_info.name == field_name {
                            debug!(
                                "[lower_field_access_for_class] Found field '{}' at index {} in class_ty={:?}",
                                field_name_str, idx, class_ty
                            );
                            found = Some((class_ty, idx, sym_info.type_id));
                            break;
                        }
                    }
                }

                match found {
                    Some(result) => result,
                    None => {
                        debug!(
                            "[lower_field_access_for_class] Field '{}' ({:?}) NOT FOUND in field_index_map!",
                            field_name_str, field
                        );
                        self.add_error(
                            &format!(
                                "Field '{}' ({:?}) index not found for raw pointer access",
                                field_name_str, field
                            ),
                            SourceLocation::unknown(),
                        );
                        return None;
                    }
                }
            }
        };

        let index_const = self.builder.build_const(IrValue::I32(field_index as i32))?;

        let field_ir_ty = self.convert_type(actual_field_type);

        // Struct context lets the backends type the field access.
        let struct_ctx = {
            let class_sym = self.class_type_to_symbol.get(&resolved_class_ty);
            let class_name = class_sym
                .and_then(|sym_id| self.symbol_table.get_symbol(*sym_id))
                .and_then(|sym| self.string_interner.get(sym.name))
                .map(|s| s.to_string());
            class_name.map(|cn| crate::ir::instructions::StructFieldRef {
                struct_name: cn,
                field_name: field_name.to_string(),
                field_index: field_index as u32,
            })
        };

        let field_ptr = self.builder.build_gep_with_context(
            obj,
            vec![index_const],
            field_ir_ty.clone(),
            struct_ctx,
        )?;

        let field_value = self.builder.build_load(field_ptr, field_ir_ty.clone())?;

        self.builder
            .set_register_type(field_value, field_ir_ty.clone());

        // Type erasure coercion: if field is I64 (erased type param) but expr expects concrete type
        let expected_ir_ty = self.convert_type(field_ty);
        if field_ir_ty == IrType::I64 && expected_ir_ty != IrType::I64 {
            return self.coerce_from_i64(field_value, field_ty);
        }

        Some(field_value)
    }
}
