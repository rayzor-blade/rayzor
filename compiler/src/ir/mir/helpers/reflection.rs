//! Dynamic field reads and detection of reflective call sites.

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
    pub(crate) fn dynamic_reflect_field_read(
        &mut self,
        obj: IrId,
        field: SymbolId,
        field_ty: TypeId,
    ) -> Option<IrId> {
        let field_name_str = self
            .symbol_table
            .get_symbol(field)
            .and_then(|s| self.string_interner.get(s.name))
            .map(|s| s.to_string())?;

        // Register evidence beats the erased HIR type: a receiver whose register is
        // already a concrete String came from a typed producer (e.g. `string_concat`)
        // under a Dynamic-erased HIR type. It is not a boxed Dynamic, so the
        // unbox_reference peeling below would walk garbage.
        let obj_reg_ty = self.builder.get_register_type(obj);
        if (matches!(&obj_reg_ty, Some(IrType::String))
            || matches!(&obj_reg_ty, Some(IrType::Ptr(inner)) if matches!(inner.as_ref(), IrType::String)))
            && field_name_str == "length"
        {
            let len_id = self.get_or_register_extern_function(
                "haxe_string_length",
                vec![IrType::String],
                IrType::I64,
            );
            return self
                .builder
                .build_call_direct(len_id, vec![obj], IrType::I64);
        }

        let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));

        // Unbox the Dynamic value to get the anonymous object handle.
        // The object may be a boxed DynamicValue (from haxe_box_reference_ptr).
        let unbox_ref_id = self.get_or_register_extern_function(
            "haxe_unbox_reference_ptr",
            vec![ptr_u8.clone()],
            ptr_u8.clone(),
        );
        let handle = self
            .builder
            .build_call_direct(unbox_ref_id, vec![obj], ptr_u8.clone())?;

        let field_name_reg = self.builder.build_const(IrValue::String(field_name_str))?;
        let reflect_field_id = self.get_or_register_extern_function(
            "haxe_reflect_field",
            vec![ptr_u8.clone(), ptr_u8.clone()],
            ptr_u8.clone(),
        );
        let dynamic_result = self.builder.build_call_direct(
            reflect_field_id,
            vec![handle, field_name_reg],
            ptr_u8.clone(),
        )?;

        // Unbox based on field_ty
        let type_table = self.type_table;
        let field_type_kind = type_table.get(field_ty).map(|t| t.kind.clone());

        let result = match field_type_kind.as_ref() {
            Some(TypeKind::Int) => {
                let unbox_id = self.get_or_register_extern_function(
                    "haxe_unbox_int_ptr",
                    vec![ptr_u8.clone()],
                    IrType::I64,
                );
                self.builder
                    .build_call_direct(unbox_id, vec![dynamic_result], IrType::I64)?
            }
            Some(TypeKind::Float) => {
                let unbox_id = self.get_or_register_extern_function(
                    "haxe_unbox_float_ptr",
                    vec![ptr_u8.clone()],
                    IrType::F64,
                );
                self.builder
                    .build_call_direct(unbox_id, vec![dynamic_result], IrType::F64)?
            }
            Some(TypeKind::Bool) => {
                let unbox_id = self.get_or_register_extern_function(
                    "haxe_unbox_bool_ptr",
                    vec![ptr_u8.clone()],
                    IrType::Bool,
                );
                self.builder
                    .build_call_direct(unbox_id, vec![dynamic_result], IrType::Bool)?
            }
            Some(TypeKind::String) => {
                let unbox_id = self.get_or_register_extern_function(
                    "haxe_unbox_reference_ptr",
                    vec![ptr_u8.clone()],
                    ptr_u8.clone(),
                );
                self.builder
                    .build_call_direct(unbox_id, vec![dynamic_result], ptr_u8)?
            }
            _ => {
                // Dynamic or unknown: return DynamicValue* as-is
                dynamic_result
            }
        };
        Some(result)
    }

    /// Read a field from a raw anonymous object handle via Reflect API.
    /// Unlike `dynamic_reflect_field_read`, this does NOT call
    /// `haxe_unbox_reference_ptr` first — the handle is already raw.
    pub(crate) fn raw_anon_reflect_field_read(
        &mut self,
        obj: IrId,
        field: SymbolId,
        field_ty: TypeId,
    ) -> Option<IrId> {
        let field_name_str = self
            .symbol_table
            .get_symbol(field)
            .and_then(|s| self.string_interner.get(s.name))
            .map(|s| s.to_string())?;

        let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));

        let field_name_reg = self.builder.build_const(IrValue::String(field_name_str))?;
        let reflect_field_id = self.get_or_register_extern_function(
            "haxe_reflect_field",
            vec![ptr_u8.clone(), ptr_u8.clone()],
            ptr_u8.clone(),
        );
        let dynamic_result = self.builder.build_call_direct(
            reflect_field_id,
            vec![obj, field_name_reg],
            ptr_u8.clone(),
        )?;

        // Unbox based on field_ty
        let type_table = self.type_table;
        let field_type_kind = type_table.get(field_ty).map(|t| t.kind.clone());

        let result = match field_type_kind.as_ref() {
            Some(TypeKind::Int) => {
                let unbox_id = self.get_or_register_extern_function(
                    "haxe_unbox_int_ptr",
                    vec![ptr_u8.clone()],
                    IrType::I64,
                );
                self.builder
                    .build_call_direct(unbox_id, vec![dynamic_result], IrType::I64)?
            }
            Some(TypeKind::Float) => {
                let unbox_id = self.get_or_register_extern_function(
                    "haxe_unbox_float_ptr",
                    vec![ptr_u8.clone()],
                    IrType::F64,
                );
                self.builder
                    .build_call_direct(unbox_id, vec![dynamic_result], IrType::F64)?
            }
            Some(TypeKind::Bool) => {
                let unbox_id = self.get_or_register_extern_function(
                    "haxe_unbox_bool_ptr",
                    vec![ptr_u8.clone()],
                    IrType::Bool,
                );
                self.builder
                    .build_call_direct(unbox_id, vec![dynamic_result], IrType::Bool)?
            }
            Some(TypeKind::String) => {
                let unbox_id = self.get_or_register_extern_function(
                    "haxe_unbox_reference_ptr",
                    vec![ptr_u8.clone()],
                    ptr_u8.clone(),
                );
                self.builder
                    .build_call_direct(unbox_id, vec![dynamic_result], ptr_u8)?
            }
            _ => {
                // Dynamic or unknown: return DynamicValue* as-is
                dynamic_result
            }
        };
        Some(result)
    }

    /// Dispatch a Dynamic-receiver field read to a stdlib runtime property getter
    /// (`Array.length`, `String.length`, …) before falling through to
    /// `haxe_unbox_reference_ptr` + reflect.
    ///
    /// Matches any zero-arg instance property of that name in the stdlib mapping.
    /// Primitive results are boxed as `DynamicValue*` so Dynamic-aware consumers
    /// (string concat, trace, `+`) don't dereference a raw integer as a pointer.
    pub(crate) fn try_dynamic_stdlib_property_dispatch(
        &mut self,
        obj: IrId,
        field: SymbolId,
    ) -> Option<IrId> {
        let field_name = self
            .symbol_table
            .get_symbol(field)
            .and_then(|s| self.string_interner.get(s.name))
            .map(|s| s.to_string())?;
        let (_, call) = self
            .stdlib_mapping
            .find_instance_property_by_name(&field_name)?;

        let (runtime_ret_ty, box_kind) =
            match call.return_type.as_ref().unwrap_or(&IrTypeDescriptor::I64) {
                IrTypeDescriptor::I32 => (IrType::I32, Some(PrimBoxKind::Int)),
                IrTypeDescriptor::I64 => (IrType::I64, Some(PrimBoxKind::Int)),
                IrTypeDescriptor::F64 | IrTypeDescriptor::F32 => {
                    (IrType::F64, Some(PrimBoxKind::Float))
                }
                IrTypeDescriptor::Bool => (IrType::Bool, Some(PrimBoxKind::Bool)),
                _ => (IrType::Ptr(Box::new(IrType::U8)), None),
            };
        let func_id = self.get_or_register_extern_function(
            call.runtime_name,
            vec![IrType::Ptr(Box::new(IrType::Void))],
            runtime_ret_ty.clone(),
        );
        let raw = self
            .builder
            .build_call_direct(func_id, vec![obj], runtime_ret_ty.clone())?;
        match box_kind {
            Some(kind) => self.box_primitive_as_dynamic(raw, runtime_ret_ty, kind),
            None => Some(raw),
        }
    }

    /// Infer type info for Reflect.compare args.
    /// Returns:
    /// - `Some(Ok(tag))` for concrete types (Int=1, Bool=2, Float=4, String=5)
    /// - `Some(Err(type_param_name))` for generic type parameters (needs fixup)
    /// - `None` for Dynamic/unknown (fall through to boxing path)
    ///
    /// One tag describes BOTH slots: `haxe_reflect_compare_typed` reads `a` and
    /// `b` the same way. Reading the tag off the first argument alone is a lie
    /// whenever the second has another representation -- `compare(5, aDynamic)`
    /// then compares 5 against a box ADDRESS, and `compare("s", aDynamic)`
    /// dereferences the box header as a HaxeString and segfaults. When the two
    /// operands disagree, answer `None` so the caller boxes both and uses
    /// `haxe_reflect_compare`, which carries each side's own type.
    pub(crate) fn infer_reflect_compare_type_info(
        &self,
        args: &[HirExpr],
    ) -> Option<Result<i32, String>> {
        let first = self.reflect_compare_operand_tag(&args[0])?;
        if let Some(second) = args.get(1) {
            if self.reflect_compare_operand_tag(second) != Some(first.clone()) {
                return None;
            }
        }
        Some(first)
    }

    /// The tag one Reflect.compare operand asks for, or `None` when its own
    /// representation is not known here.
    fn reflect_compare_operand_tag(&self, first: &HirExpr) -> Option<Result<i32, String>> {
        use crate::tast::core::TypeKind;
        let type_table = self.type_table;
        debug!(
            "[REFLECT_COMPARE_TYPE] arg ty={:?}, type_info={:?}",
            first.ty,
            type_table.get(first.ty).map(|ti| format!("{:?}", ti.kind))
        );
        if let Some(ti) = type_table.get(first.ty) {
            match &ti.kind {
                TypeKind::Int => return Some(Ok(1)),
                TypeKind::Float => return Some(Ok(4)),
                TypeKind::Bool => return Some(Ok(2)),
                TypeKind::String => return Some(Ok(5)),
                TypeKind::TypeParameter { symbol_id, .. } => {
                    if let Some(sym) = self.symbol_table.get_symbol(*symbol_id) {
                        if let Some(name_str) = self.string_interner.get(sym.name) {
                            return Some(Err(name_str.to_string()));
                        }
                    }
                    return Some(Err("T".to_string()));
                }
                TypeKind::Dynamic => {
                    // Params are Dynamic — infer from expression kind
                    if let HirExprKind::Literal(crate::ir::hir::HirLiteral::String(_)) = &first.kind
                    {
                        return Some(Ok(5));
                    }
                    if let HirExprKind::Variable { symbol, .. } = &first.kind {
                        if let Some(sym) = self.symbol_table.get_symbol(*symbol) {
                            if let Some(sym_ti) = type_table.get(sym.type_id) {
                                match &sym_ti.kind {
                                    TypeKind::Int => return Some(Ok(1)),
                                    TypeKind::Float => return Some(Ok(4)),
                                    TypeKind::Bool => return Some(Ok(2)),
                                    TypeKind::String => return Some(Ok(5)),
                                    _ => {}
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        None
    }

    /// Erase a typed Reflect.compare operand into the runtime's i64 slot.
    /// Floats must preserve their IEEE bits: a numeric F64 -> I64 cast turns
    /// both 0.1 and 0.2 into zero, so the runtime can no longer order them.
    pub(crate) fn erase_reflect_compare_arg(&mut self, reg: IrId) -> IrId {
        let reg_ty = self.builder.get_register_type(reg).unwrap_or(IrType::I64);
        match reg_ty {
            IrType::I64 => reg,
            IrType::F64 => self.builder.build_bitcast(reg, IrType::I64).unwrap_or(reg),
            IrType::F32 => self
                .builder
                .build_cast(reg, IrType::F32, IrType::F64)
                .and_then(|wide| self.builder.build_bitcast(wide, IrType::I64))
                .unwrap_or(reg),
            other => self
                .builder
                .build_cast(reg, other, IrType::I64)
                .unwrap_or(reg),
        }
    }

    /// Detect reflective Type allocation calls that return fresh class instances.
    pub(crate) fn is_reflective_type_alloc_call(&self, callee: &HirExpr, arg_count: usize) -> bool {
        let is_alloc_method = |name: &str| matches!(name, "createInstance" | "createEmptyInstance");

        match &callee.kind {
            HirExprKind::Variable { symbol, .. } => {
                if let Some((class_name, method_name, _)) =
                    self.get_stdlib_runtime_info(*symbol, TypeId::invalid(), Some(arg_count), None)
                {
                    if class_name == "Type" && is_alloc_method(method_name) {
                        return true;
                    }
                }

                let Some(sym) = self.symbol_table.get_symbol(*symbol) else {
                    return false;
                };
                let method_name = self.string_interner.get(sym.name).unwrap_or("");
                if !is_alloc_method(method_name) {
                    return false;
                }

                if let Some(qn) = sym.qualified_name.and_then(|q| self.string_interner.get(q)) {
                    return qn == "Type.createInstance"
                        || qn == "Type.createEmptyInstance"
                        || qn.ends_with(".Type.createInstance")
                        || qn.ends_with(".Type.createEmptyInstance");
                }
                false
            }
            HirExprKind::Field { object, field } => {
                let owner_is_type = if let HirExprKind::Variable { symbol, .. } = &object.kind {
                    self.symbol_table.get_symbol(*symbol).is_some_and(|sym| {
                        self.string_interner.get(sym.name) == Some("Type")
                            || sym
                                .qualified_name
                                .and_then(|q| self.string_interner.get(q))
                                .is_some_and(|qn| qn == "Type" || qn.ends_with(".Type"))
                    })
                } else {
                    false
                };
                if !owner_is_type {
                    return false;
                }

                self.symbol_table
                    .get_symbol(*field)
                    .and_then(|sym| self.string_interner.get(sym.name))
                    .is_some_and(is_alloc_method)
            }
            _ => false,
        }
    }

    pub(crate) fn is_type_typeof_callee_expr(&self, expr: &HirExpr) -> bool {
        match &expr.kind {
            HirExprKind::Variable { symbol, .. } => self.is_type_typeof_symbol(*symbol),
            HirExprKind::Field { field, .. } => self.is_type_typeof_symbol(*field),
            HirExprKind::Cast { expr, .. } => self.is_type_typeof_callee_expr(expr),
            _ => false,
        }
    }

    pub(crate) fn is_type_typeof_symbol(&self, symbol_id: SymbolId) -> bool {
        let Some(sym) = self.symbol_table.get_symbol(symbol_id) else {
            return false;
        };

        let name = self.string_interner.get(sym.name);
        if name == Some("haxe_type_typeof") {
            return true;
        }

        if let Some(native) = sym.native_name.and_then(|n| self.string_interner.get(n)) {
            if native == "haxe_type_typeof" {
                return true;
            }
        }

        if name != Some("typeof") {
            return false;
        }

        if let Some(qn) = sym
            .qualified_name
            .and_then(|qn| self.string_interner.get(qn))
        {
            if qn == "Type.typeof" || qn.ends_with(".Type.typeof") {
                return true;
            }
        }

        // Some lowering paths keep only the bare method name.
        true
    }

    pub(crate) fn trace_typeof_inner_arg<'b>(&self, expr: &'b HirExpr) -> Option<&'b HirExpr> {
        match &expr.kind {
            HirExprKind::Cast { expr: inner, .. } => self.trace_typeof_inner_arg(inner),
            HirExprKind::Call { callee, args, .. } => {
                if args.len() == 1 && self.is_type_typeof_callee_expr(callee) {
                    return args.first();
                }
                None
            }
            _ => None,
        }
    }
}
