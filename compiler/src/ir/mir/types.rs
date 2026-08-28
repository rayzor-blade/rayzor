//! HIR-to-MIR conversion of types and operators, plus the scalar coercions
//! and Dynamic boxing used at every call boundary.

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
    pub(crate) fn convert_value_type_to_string(&mut self, value: IrId) -> Option<IrId> {
        let string_ptr_ty = IrType::Ptr(Box::new(IrType::String));
        let conv_fn = self.get_or_register_extern_function(
            "haxe_string_from_value_type",
            vec![IrType::I64],
            string_ptr_ty.clone(),
        );
        self.builder
            .build_call_direct(conv_fn, vec![value], string_ptr_ty)
    }

    pub(crate) fn convert_binary_op(&self, op: HirBinaryOp) -> BinaryOp {
        match op {
            HirBinaryOp::Add => BinaryOp::Add,
            HirBinaryOp::Sub => BinaryOp::Sub,
            HirBinaryOp::Mul => BinaryOp::Mul,
            HirBinaryOp::Div => BinaryOp::Div,
            HirBinaryOp::Mod => BinaryOp::Rem,
            HirBinaryOp::BitAnd => BinaryOp::And,
            HirBinaryOp::BitOr => BinaryOp::Or,
            HirBinaryOp::BitXor => BinaryOp::Xor,
            HirBinaryOp::Shl => BinaryOp::Shl,
            HirBinaryOp::Shr => BinaryOp::Shr,
            HirBinaryOp::Ushr => BinaryOp::Ushr,
            _ => BinaryOp::Add, // Default fallback
        }
    }

    pub(crate) fn convert_binary_op_to_mir(&self, op: HirBinaryOp) -> MirBinaryOp {
        match op {
            HirBinaryOp::Add => MirBinaryOp::Binary(BinaryOp::Add),
            HirBinaryOp::Sub => MirBinaryOp::Binary(BinaryOp::Sub),
            HirBinaryOp::Mul => MirBinaryOp::Binary(BinaryOp::Mul),
            HirBinaryOp::Div => MirBinaryOp::Binary(BinaryOp::Div),
            HirBinaryOp::Mod => MirBinaryOp::Binary(BinaryOp::Rem),
            HirBinaryOp::Eq => MirBinaryOp::Compare(CompareOp::Eq),
            HirBinaryOp::Ne => MirBinaryOp::Compare(CompareOp::Ne),
            HirBinaryOp::Lt => MirBinaryOp::Compare(CompareOp::Lt),
            HirBinaryOp::Le => MirBinaryOp::Compare(CompareOp::Le),
            HirBinaryOp::Gt => MirBinaryOp::Compare(CompareOp::Gt),
            HirBinaryOp::Ge => MirBinaryOp::Compare(CompareOp::Ge),
            HirBinaryOp::BitAnd => MirBinaryOp::Binary(BinaryOp::And),
            HirBinaryOp::BitOr => MirBinaryOp::Binary(BinaryOp::Or),
            HirBinaryOp::BitXor => MirBinaryOp::Binary(BinaryOp::Xor),
            HirBinaryOp::Shl => MirBinaryOp::Binary(BinaryOp::Shl),
            HirBinaryOp::Shr => MirBinaryOp::Binary(BinaryOp::Shr),
            HirBinaryOp::Ushr => MirBinaryOp::Binary(BinaryOp::Ushr),
            _ => MirBinaryOp::Binary(BinaryOp::Add), // Default
        }
    }

    pub(crate) fn convert_unary_op(&self, op: HirUnaryOp) -> UnaryOp {
        match op {
            HirUnaryOp::Not => UnaryOp::Not,
            HirUnaryOp::Neg => UnaryOp::Neg,
            HirUnaryOp::BitNot => UnaryOp::Not, // Reuse Not for bit not
            _ => UnaryOp::Neg,                  // Default
        }
    }

    /// The type argument standing in for a generic abstract's underlying.
    ///
    /// `abstract Val<T>(T)` writes its underlying as its own type parameter, so
    /// converting that directly reports the erased representation rather than
    /// the instantiation's. `Val<Float>` carries Float in its type_args; this
    /// hands it back so the conversion sees an f64.
    ///
    /// Deliberately narrow: only a bare type parameter as the whole underlying,
    /// and only when the instantiation supplies exactly one argument. With more
    /// than one, matching a parameter to its argument needs the declaration's
    /// parameter ORDER, which is not recorded on the type. Returning None then
    /// keeps the previous behaviour rather than guessing a position.
    fn substitute_abstract_type_arg(
        &self,
        underlying: TypeId,
        type_args: &[TypeId],
    ) -> Option<TypeId> {
        if type_args.len() != 1 {
            return None;
        }
        match self.type_table.get(underlying).map(|t| &t.kind) {
            Some(TypeKind::TypeParameter { .. }) => Some(type_args[0]),
            _ => None,
        }
    }

    pub(crate) fn convert_type(&self, type_id: TypeId) -> IrType {
        use crate::tast::TypeKind;

        let type_table = self.type_table;
        let type_ref = type_table.get(type_id);

        match type_ref.as_ref().map(|t| &t.kind) {
            Some(TypeKind::Int) => IrType::I32,
            Some(TypeKind::Float) => IrType::F64,
            Some(TypeKind::Bool) => IrType::Bool,
            Some(TypeKind::Void) => IrType::Void,
            Some(TypeKind::String) => IrType::String,

            Some(TypeKind::Function {
                params,
                return_type,
                ..
            }) => {
                let param_types: Vec<IrType> =
                    params.iter().map(|p| self.convert_type(*p)).collect();

                let ret_type = Box::new(self.convert_type(*return_type));

                IrType::Function {
                    params: param_types,
                    return_type: ret_type,
                    varargs: false,
                }
            }

            Some(TypeKind::Class { .. }) => IrType::Ptr(Box::new(IrType::Void)),
            Some(TypeKind::Interface { .. }) => IrType::Ptr(Box::new(IrType::Void)),
            Some(TypeKind::Enum { .. }) => IrType::I64, // Enums as discriminant values (i64 to match Haxe Int)
            Some(TypeKind::Array { element_type, .. }) => {
                // HaxeArray is an opaque runtime structure, represented as Ptr(Void)
                // regardless of element type. Element type information is tracked at runtime.
                IrType::Ptr(Box::new(IrType::Void))
            }
            Some(TypeKind::Optional { inner_type }) => {
                // Null<T>: primitives need boxing to distinguish null from 0/0.0/false.
                // Reference types are already nullable pointers — no boxing needed.
                let inner_ir = self.convert_type(*inner_type);
                match inner_ir {
                    IrType::I32 | IrType::I64 | IrType::F64 | IrType::F32 | IrType::Bool => {
                        IrType::Ptr(Box::new(IrType::U8)) // Boxed DynamicValue*
                    }
                    _ => inner_ir,
                }
            }

            Some(TypeKind::Abstract {
                underlying,
                symbol_id,
                type_args,
            }) => {
                // Pointer-sized abstracts (Usize, Ptr, Ref, Box) are I64 regardless
                // of their declared underlying type: they carry machine addresses
                // and must never be truncated.
                let name_str = self
                    .symbol_table
                    .get_symbol(*symbol_id)
                    .and_then(|sym| self.string_interner.get(sym.name))
                    .unwrap_or("");
                if matches!(name_str, "Usize" | "Ptr" | "Ref" | "Box") {
                    return IrType::I64;
                }
                // Int64 is a real 64-bit integer here. Haxe declares it as an
                // abstract over a two-word object because some of its targets
                // have no 64-bit integer to lower to; rayzor does, so carrying
                // a high/low pair -- and an allocation per value -- buys
                // nothing. It was reaching the underlying-type fallback below
                // and coming out as I32, which does not merely cost
                // performance: every Int64 was being truncated to 32 bits.
                if matches!(name_str, "Int64" | "__Int64" | "___Int64") {
                    return IrType::I64;
                }
                // `Single`'s `to`/`from Float` are cast compatibility, not identity:
                // the representation is 32-bit. Decided here with the other
                // representation-bearing coreTypes, before the underlying/opaque
                // fallback, which would hand it an 8-byte slot holding f64 bits.
                if name_str == "Single" {
                    return IrType::F32;
                }

                // A SIMD coreType's representation follows from its IDENTITY, so it
                // is decided ahead of both the underlying type and the
                // `is_systems_type` check below — the latter claims them (SIMD* are
                // mir_wrapper classes) and returns I64, truncating the vector to a
                // 64-bit param, while the former hands back whatever nominal
                // underlying the declaration carries (a BLADE manifest records the
                // absent one as `Dynamic`, which lowers to a pointer). Match on
                // native_name, qualified_name or bare name: a coreType SIMD abstract
                // reached through a function parameter may not carry native_name in
                // this view.
                let type_names = self.symbol_table.get_symbol(*symbol_id).map(|sym| {
                    let native = sym
                        .native_name
                        .and_then(|nn| self.string_interner.get(nn))
                        .unwrap_or("");
                    let qualified = sym
                        .qualified_name
                        .and_then(|qn| self.string_interner.get(qn))
                        .unwrap_or("");
                    let bare = self.string_interner.get(sym.name).unwrap_or("");
                    (native, qualified, bare)
                });
                if let Some((native, qualified, bare)) = type_names {
                    let is = |simd: &str| {
                        native == format!("rayzor::{}", simd)
                            || qualified == format!("rayzor.{}", simd)
                            || bare == simd
                    };
                    if is("SIMD4f") {
                        return IrType::vector(IrType::F32, 4);
                    }
                    if is("SIMD4i32") {
                        return IrType::vector(IrType::I32, 4);
                    }
                    if is("SIMD16i8") {
                        return IrType::vector(IrType::I8, 16);
                    }
                    if is("SIMD8i32") {
                        return IrType::vector(IrType::I32, 8);
                    }
                    if is("SIMD32i8") {
                        return IrType::vector(IrType::I8, 32);
                    }
                    if native == "rayzor::Atomic" {
                        return IrType::Ptr(Box::new(IrType::I32));
                    }
                }

                if let Some(underlying_type) = underlying {
                    // A generic abstract's underlying is written in terms of its
                    // own type parameter -- `abstract Val<T>(T)` -- so converting
                    // it directly asks what representation `T` has, and gets the
                    // erased one. The instantiation already carries the answer in
                    // type_args: substitute before converting, or `Val<Float>`
                    // reports the erasure rather than an f64.
                    let resolved = self
                        .substitute_abstract_type_arg(*underlying_type, type_args)
                        .unwrap_or(*underlying_type);
                    self.convert_type(resolved)
                } else {
                    // Systems types (Ptr, Ref, Box, Usize) are pointer-sized abstracts.
                    let is_systems_type = self
                        .symbol_table
                        .get_symbol(*symbol_id)
                        .and_then(|sym| {
                            let qn = sym
                                .qualified_name
                                .and_then(|qn| self.string_interner.get(qn))
                                .map(|qn| qn.replace(".", "_"))
                                .or_else(|| {
                                    sym.native_name
                                        .and_then(|nn| self.string_interner.get(nn))
                                        .map(|nn| nn.replace("::", "_"))
                                });
                            qn.filter(|name| {
                                self.stdlib_mapping
                                    .class_key(name)
                                    .is_some_and(|k| self.stdlib_mapping.is_mir_wrapper_class(k))
                            })
                        })
                        .is_some();
                    if is_systems_type {
                        IrType::I64
                    } else {
                        IrType::I32
                    }
                }
            }

            // Type erasure: type params are pointer-sized so one struct layout serves
            // every instantiation; coercion happens at generic boundaries (field access,
            // calls). Constrained params (T:Interface) become Ptr(Void) for vtable-based
            // dispatch. Callers that need TypeVar for generic dispatch extract the type
            // param name separately via resolve_type_param_name().
            Some(TypeKind::TypeParameter { constraints, .. }) => {
                if !constraints.is_empty() && self.has_interface_constraint(&constraints) {
                    IrType::Ptr(Box::new(IrType::Void))
                } else {
                    IrType::I64
                }
            }
            Some(TypeKind::Dynamic) => {
                // Also the stdlib placeholder for unresolved generic type params. A
                // dynamic value can be an object/pointer, so it stays pointer-sized
                // to avoid truncation.
                IrType::Ptr(Box::new(IrType::Void))
            }

            Some(TypeKind::Unknown) | Some(TypeKind::Error) => {
                // May be unresolved generic class instances; pointer-sized to avoid
                // truncating the full 64-bit value.
                warn!("Unknown/Error type {:?}, defaulting to Ptr(Void)", type_id);
                IrType::Ptr(Box::new(IrType::Void))
            }

            Some(TypeKind::GenericInstance { .. }) => IrType::Ptr(Box::new(IrType::Void)),

            Some(TypeKind::Map { .. }) => IrType::Ptr(Box::new(IrType::Void)),

            Some(TypeKind::Anonymous { .. }) => IrType::Ptr(Box::new(IrType::Void)),

            Some(TypeKind::Union { .. }) => IrType::Ptr(Box::new(IrType::Void)),

            Some(TypeKind::Intersection { .. }) => IrType::Ptr(Box::new(IrType::Void)),

            Some(TypeKind::TypeAlias { target_type, .. }) => self.convert_type(*target_type),

            Some(TypeKind::Placeholder { .. }) => IrType::Ptr(Box::new(IrType::Void)),

            Some(TypeKind::Char) => IrType::I32,

            None => {
                // Usually an unresolved generic type parameter; Ptr(Void) avoids
                // truncating values that are actually pointers/objects.
                // TODO: resolve generic type parameters from instantiation context.
                warn!(
                    "Type {:?} not found in type table, defaulting to Ptr(Void)",
                    type_id
                );
                IrType::Ptr(Box::new(IrType::Void))
            }

            Some(other) => {
                warn!(
                    "Unhandled type kind for {:?}: {:?}, defaulting to Ptr(Void)",
                    type_id, other
                );
                IrType::Ptr(Box::new(IrType::Void))
            }
        }
    }

    pub(crate) fn convert_source_location(&self, loc: &SourceLocation) -> IrSourceLocation {
        IrSourceLocation {
            file_id: loc.file_id,
            line: loc.line,
            column: loc.column,
        }
    }

    /// Coerce args to match external function param types at cross-module call boundaries.
    /// Only casts I32→F64 and I64→F64 where the callee expects Float but caller passes Int.
    pub(crate) fn coerce_args_for_cross_module_call(
        &mut self,
        func_id: IrFunctionId,
        arg_regs: &mut [IrId],
        skip_first: bool, // true if first arg is implicit 'this'
    ) {
        // Only coerce for external (cross-module) functions
        let param_types = match self.external_function_param_types.get(&func_id) {
            Some(types) => types.clone(),
            None => return,
        };

        let start = if skip_first { 1 } else { 0 };
        for (i, arg_reg) in arg_regs.iter_mut().enumerate().skip(start) {
            if let Some(expected_ty) = param_types.get(i) {
                let actual_ty = self.builder.get_register_type(*arg_reg);
                // Cast I32/I64 → F64 when callee expects Float
                if matches!(expected_ty, IrType::F64)
                    && matches!(actual_ty, Some(IrType::I32) | Some(IrType::I64))
                {
                    if let Some(cast_reg) =
                        self.builder
                            .build_cast(*arg_reg, actual_ty.unwrap().clone(), IrType::F64)
                    {
                        *arg_reg = cast_reg;
                    }
                }
            }
        }
    }

    /// Like convert_type but returns TypeVar for TypeParameter types that match
    /// a known type param name. This preserves generic type info in function signatures
    /// so the monomorphizer can specialize them.
    pub(crate) fn convert_type_or_type_var(
        &self,
        type_id: crate::tast::TypeId,
        type_param_names: &[String],
    ) -> IrType {
        if !type_param_names.is_empty() {
            if let Some(name) = self.resolve_type_param_name(type_id) {
                if type_param_names.contains(&name) {
                    return IrType::TypeVar(name);
                }
            }
        }
        self.convert_type(type_id)
    }

    /// Convert a value to a string pointer
    /// Uses the appropriate *_to_string MIR wrapper based on the source type.
    /// If hir_type_id is provided and is a TypeParameter, uses tag-based generic dispatch.
    pub(crate) fn convert_to_string_with_hint(
        &mut self,
        value: IrId,
        from_type: &IrType,
        hir_type_id: Option<crate::tast::TypeId>,
    ) -> Option<IrId> {
        // Check HIR type for Array (maps to Ptr(Void) in MIR, indistinguishable from Dynamic)
        if let Some(type_id) = hir_type_id {
            let type_kind = self.type_table.get(type_id).map(|ti| ti.kind.clone());
            if matches!(type_kind.as_ref(), Some(TypeKind::Array { .. })) {
                let func_id = self.get_or_register_extern_function(
                    "haxe_array_to_string",
                    vec![IrType::Ptr(Box::new(IrType::Void))],
                    IrType::Ptr(Box::new(IrType::String)),
                );
                return self.builder.build_call_direct(
                    func_id,
                    vec![value],
                    IrType::Ptr(Box::new(IrType::String)),
                );
            }
        }
        // Check if the HIR type is a TypeParameter — if so, use tag-based dispatch
        // even though the MIR type is I64 (type-erased).
        if let Some(type_id) = hir_type_id {
            if let Some(type_param_name) = self.resolve_type_param_name(type_id) {
                let tag_reg = self.builder.build_const(IrValue::I32(0))?;
                if let Some(func) = self.builder.current_function_mut() {
                    func.type_param_tag_fixups.push((tag_reg, type_param_name));
                }
                let func_id = self.get_or_register_extern_function(
                    "haxe_value_to_string_by_tag",
                    vec![IrType::I64, IrType::I32],
                    IrType::Ptr(Box::new(IrType::String)),
                );
                let val_as_i64 = self
                    .builder
                    .build_bitcast(value, IrType::I64)
                    .unwrap_or(value);
                return self.builder.build_call_direct(
                    func_id,
                    vec![val_as_i64, tag_reg],
                    IrType::String,
                );
            }
        }
        self.convert_to_string(value, from_type)
    }

    /// Convert a value to a string pointer
    /// Uses the appropriate *_to_string MIR wrapper based on the source type
    pub(crate) fn convert_to_string(&mut self, value: IrId, from_type: &IrType) -> Option<IrId> {
        let mir_wrapper = match from_type {
            IrType::I32 | IrType::I64 => "int_to_string",
            IrType::F32 | IrType::F64 => "float_to_string",
            IrType::Bool => "bool_to_string",
            IrType::String => {
                return Some(value);
            }
            IrType::Ptr(inner) if matches!(inner.as_ref(), IrType::String) => {
                return Some(value);
            }
            IrType::Ptr(inner) if matches!(inner.as_ref(), IrType::Void) => {
                // Ptr(Void) could be Array, Class or DynBox; the class hint is
                // the only thing that identifies an Array here.
                let is_array = self
                    .register_class_hints
                    .get(&value)
                    .map(|h| h == "Array")
                    .unwrap_or(false);
                if is_array {
                    let func_id = self.get_or_register_extern_function(
                        "haxe_array_to_string",
                        vec![IrType::Ptr(Box::new(IrType::Void))],
                        IrType::Ptr(Box::new(IrType::String)),
                    );
                    return self.builder.build_call_direct(
                        func_id,
                        vec![value],
                        IrType::Ptr(Box::new(IrType::String)),
                    );
                }
                return self.convert_dynamic_to_string(value);
            }
            IrType::Ptr(_) => {
                return self.convert_dynamic_to_string(value);
            }
            IrType::TypeVar(ref type_param_name) => {
                // The tag is a placeholder (0) resolved during inlining or
                // monomorphization via type_param_tag_fixups.
                let tag_reg = self.builder.build_const(IrValue::I32(0))?;
                if let Some(func) = self.builder.current_function_mut() {
                    func.type_param_tag_fixups
                        .push((tag_reg, type_param_name.clone()));
                }
                let func_id = self.get_or_register_extern_function(
                    "haxe_value_to_string_by_tag",
                    vec![IrType::I64, IrType::I32],
                    IrType::Ptr(Box::new(IrType::String)),
                );
                let val_as_i64 = self
                    .builder
                    .build_bitcast(value, IrType::I64)
                    .unwrap_or(value);
                return self.builder.build_call_direct(
                    func_id,
                    vec![val_as_i64, tag_reg],
                    IrType::String,
                );
            }
            IrType::Any => {
                return self.convert_dynamic_to_string(value);
            }
            _ => "int_to_string", // Fallback
        };

        debug!(
            "[CONVERT TO STRING] Using {} for type {:?}",
            mir_wrapper, from_type
        );

        // Declare each wrapper with its own signature, never the caller's type:
        // wrappers are registered by name, so the first caller would otherwise pin
        // the parameter (one `Single` f32 argument would make every later `Float`
        // truncate on the way in). Arg-side coercion widens/narrows to these, and
        // the Cranelift C ABI auto-cast covers I32↔I64 promotion at the call site.
        let param_ty = match mir_wrapper {
            "int_to_string" => IrType::I64,
            "float_to_string" => IrType::F64,
            "bool_to_string" => IrType::Bool,
            _ => from_type.clone(),
        };
        let func_id =
            self.register_stdlib_mir_forward_ref(mir_wrapper, vec![param_ty], IrType::String);

        self.builder
            .build_call_direct(func_id, vec![value], IrType::String)
    }

    /// Convert a dynamic/unknown-type value to string using runtime dispatch.
    /// Calls haxe_std_string_ptr(ptr) which reads DynBox and dispatches to_string.
    pub(crate) fn convert_dynamic_to_string(&mut self, value: IrId) -> Option<IrId> {
        let func_id = self.get_or_register_extern_function(
            "haxe_std_string_ptr",
            vec![IrType::Ptr(Box::new(IrType::U8))],
            IrType::Ptr(Box::new(IrType::String)),
        );

        let ptr_val = self
            .builder
            .build_bitcast(value, IrType::Ptr(Box::new(IrType::U8)))
            .unwrap_or(value);

        self.builder.build_call_direct(
            func_id,
            vec![ptr_val],
            IrType::Ptr(Box::new(IrType::String)),
        )
    }

    /// Box a `Channel<T>` send payload into a self-describing DynamicValue so
    /// the receive side can tag-dispatch (see `haxe_channel_unbox_erased`).
    /// References carry their real class id (keeps `Std.is`/`Type.typeof`
    /// honest); String uses the HaxeString wrapper; primitives take the
    /// extern-boxing path. Only `Channel_send`/`Channel_trySend` route here.
    pub(crate) fn box_channel_payload(
        &mut self,
        value: IrId,
        value_ty: TypeId,
        actual_ty: &IrType,
        expected_ty: &IrType,
    ) -> Option<IrId> {
        use crate::tast::TypeKind;
        let value_kind = self.type_table.get(value_ty).map(|t| t.kind.clone());
        let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
        match &value_kind {
            Some(TypeKind::Class { .. })
            | Some(TypeKind::Enum { .. })
            | Some(TypeKind::Interface { .. })
            | Some(TypeKind::Anonymous { .. })
            | Some(TypeKind::Array { .. }) => {
                let stable_id = match self.runtime_type_id(value_ty) {
                    0 => value_ty.as_raw(),
                    stable => stable,
                };
                let type_id_const = self.builder.build_const(IrValue::U32(stable_id))?;
                let box_func_id = self.get_or_register_extern_function(
                    "haxe_box_reference_ptr",
                    vec![ptr_u8.clone(), IrType::U32],
                    ptr_u8.clone(),
                );
                self.builder
                    .build_call_direct(box_func_id, vec![value, type_id_const], ptr_u8)
            }
            Some(TypeKind::String) => {
                let value_as_ptr = self
                    .builder
                    .build_bitcast(value, ptr_u8.clone())
                    .unwrap_or(value);
                let box_func_id = self.get_or_register_extern_function(
                    "haxe_box_haxestring_ptr",
                    vec![ptr_u8.clone()],
                    ptr_u8.clone(),
                );
                self.builder
                    .build_call_direct(box_func_id, vec![value_as_ptr], ptr_u8)
            }
            Some(TypeKind::Function { .. }) => self.box_value_for_dynamic(value, value_ty),
            // Primitives (Int/Float/Bool) and anything else take the extern-boxing path.
            _ => self.maybe_box_for_extern_call(value, actual_ty, expected_ty),
        }
    }

    /// Tag-aware unbox for `Channel<T>` receive/tryReceive returns. The payload
    /// is always a self-describing DynamicValue (see `box_channel_payload`).
    /// Mirrors `maybe_unbox_for_extern_return` except in two arms: the erased
    /// I64 arm tag-dispatches via `haxe_channel_unbox_erased` to resolve the
    /// boxed-prim-vs-ref ambiguity, and the reference arm unboxes. A
    /// `tryReceive` reference payload (Ptr(U8) erased, Ptr(Void) concrete)
    /// unboxes null-guarded, so an empty channel (0) resolves to null and
    /// `!= null` keeps working; prim payloads take the typed arms below.
    pub(crate) fn unbox_channel_return(
        &mut self,
        value: IrId,
        resolved_expected: &IrType,
        is_try: bool,
    ) -> Option<IrId> {
        let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));

        if is_try {
            if let IrType::Ptr(inner) = resolved_expected {
                match inner.as_ref() {
                    // Ptr(U8) (erased/inferred) and Ptr(Void) (explicit class, and
                    // Null<prim>) are statically indistinguishable here: a class
                    // payload must unwrap to the raw object ptr, while a Null<prim>
                    // payload must keep its box. Only the DynamicValue tag can tell
                    // them apart, so dispatch at runtime (null-guarded).
                    IrType::Void | IrType::U8 => {
                        let f = self.get_or_register_extern_function(
                            "haxe_channel_unbox_try",
                            vec![ptr_u8.clone()],
                            resolved_expected.clone(),
                        );
                        return self.builder.build_call_direct(
                            f,
                            vec![value],
                            resolved_expected.clone(),
                        );
                    }
                    _ => {}
                }
            }
        }

        // Null<prim> nullable -> inner primitive (same unwrap as maybe_unbox).
        let target_type = match resolved_expected {
            IrType::Ptr(inner) => match inner.as_ref() {
                IrType::I32 | IrType::I64 | IrType::Bool | IrType::F32 | IrType::F64 => {
                    inner.as_ref()
                }
                _ => resolved_expected,
            },
            _ => resolved_expected,
        };

        match target_type {
            IrType::I32 => {
                let f = self.get_or_register_extern_function(
                    "haxe_unbox_int_ptr",
                    vec![ptr_u8],
                    IrType::I64,
                );
                let v = self
                    .builder
                    .build_call_direct(f, vec![value], IrType::I64)?;
                self.builder.build_cast(v, IrType::I64, IrType::I32)
            }
            IrType::I64 => {
                // Erased receive: the DynamicValue tag decides prim value vs raw ref ptr.
                let f = self.get_or_register_extern_function(
                    "haxe_channel_unbox_erased",
                    vec![ptr_u8],
                    IrType::I64,
                );
                self.builder.build_call_direct(f, vec![value], IrType::I64)
            }
            IrType::Bool => {
                let f = self.get_or_register_extern_function(
                    "haxe_unbox_bool_ptr",
                    vec![ptr_u8],
                    IrType::Bool,
                );
                self.builder.build_call_direct(f, vec![value], IrType::Bool)
            }
            IrType::F32 => {
                let f = self.get_or_register_extern_function(
                    "haxe_unbox_float_ptr",
                    vec![ptr_u8],
                    IrType::F64,
                );
                let v = self
                    .builder
                    .build_call_direct(f, vec![value], IrType::F64)?;
                self.builder.build_cast(v, IrType::F64, IrType::F32)
            }
            IrType::F64 => {
                let f = self.get_or_register_extern_function(
                    "haxe_unbox_float_ptr",
                    vec![ptr_u8],
                    IrType::F64,
                );
                self.builder.build_call_direct(f, vec![value], IrType::F64)
            }
            IrType::String => {
                let f = self.get_or_register_extern_function(
                    "haxe_unbox_reference_ptr",
                    vec![ptr_u8],
                    IrType::String,
                );
                self.builder
                    .build_call_direct(f, vec![value], IrType::String)
            }
            // Reference payload (class/enum/anon/array): recover the raw ref and
            // type it as the expected pointer for downstream access.
            other => {
                let ret_ty = if matches!(other, IrType::Ptr(_)) {
                    other.clone()
                } else {
                    ptr_u8.clone()
                };
                let f = self.get_or_register_extern_function(
                    "haxe_unbox_reference_ptr",
                    vec![ptr_u8],
                    ret_ty.clone(),
                );
                self.builder.build_call_direct(f, vec![value], ret_ty)
            }
        }
    }

    pub(crate) fn box_primitive_to_dynamic(&mut self, value: IrId, reg_ty: IrType) -> Option<IrId> {
        let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
        match reg_ty {
            IrType::I32 | IrType::I64 => {
                let v64 = if matches!(reg_ty, IrType::I32) {
                    self.builder.build_cast(value, IrType::I32, IrType::I64)?
                } else {
                    value
                };
                let box_func = self.get_or_register_extern_function(
                    "haxe_box_int_ptr",
                    vec![IrType::I64],
                    ptr_u8.clone(),
                );
                self.builder.build_call_direct(box_func, vec![v64], ptr_u8)
            }
            IrType::F64 | IrType::F32 => {
                let v64 = if matches!(reg_ty, IrType::F32) {
                    self.builder.build_cast(value, IrType::F32, IrType::F64)?
                } else {
                    value
                };
                let box_func = self.get_or_register_extern_function(
                    "haxe_box_float_ptr",
                    vec![IrType::F64],
                    ptr_u8.clone(),
                );
                self.builder.build_call_direct(box_func, vec![v64], ptr_u8)
            }
            IrType::Bool => {
                let v64 = self.builder.build_cast(value, IrType::Bool, IrType::I64)?;
                let box_func = self.get_or_register_extern_function(
                    "haxe_box_bool_ptr",
                    vec![IrType::I64],
                    ptr_u8.clone(),
                );
                self.builder.build_call_direct(box_func, vec![v64], ptr_u8)
            }
            _ => None,
        }
    }

    /// Box a primitive MIR value into a `DynamicValue*` via the
    /// corresponding `haxe_box_*_ptr` runtime helper. Centralised so
    /// the runtime-symbol names live in one place; widening (i32 → i64
    /// for the int boxer) is handled here.
    pub(crate) fn box_primitive_as_dynamic(
        &mut self,
        value: IrId,
        value_ty: IrType,
        kind: PrimBoxKind,
    ) -> Option<IrId> {
        let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
        let (box_fn, box_arg_ty) = match kind {
            PrimBoxKind::Int => ("haxe_box_int_ptr", IrType::I64),
            PrimBoxKind::Float => ("haxe_box_float_ptr", IrType::F64),
            PrimBoxKind::Bool => ("haxe_box_bool_ptr", IrType::Bool),
        };
        let widened = if matches!(kind, PrimBoxKind::Int) && matches!(value_ty, IrType::I32) {
            self.builder.build_cast(value, IrType::I32, IrType::I64)?
        } else {
            value
        };
        let box_id = self.get_or_register_extern_function(box_fn, vec![box_arg_ty], ptr_u8.clone());
        self.builder
            .build_call_direct(box_id, vec![widened], ptr_u8)
    }

    /// Box a raw value as a DynamicValue* for functions that expect Dynamic parameters.
    /// Resolves TypeParameter types through the current class context.
    /// Returns Some(boxed_reg) on success, None if value is already Dynamic or can't be resolved.
    pub(crate) fn box_value_for_dynamic(&mut self, value: IrId, type_id: TypeId) -> Option<IrId> {
        let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));

        // Resolve the concrete type — handles TypeParameter through class context
        let concrete_type_id = {
            let type_table = self.type_table;
            if let Some(ti) = type_table.get(type_id) {
                match &ti.kind {
                    crate::tast::TypeKind::TypeParameter { .. } => {
                        if let Some(this_ty) = self.current_this_type {
                            self.resolve_type_param_from_receiver(type_id, this_ty)
                                .unwrap_or(type_id)
                        } else {
                            type_id
                        }
                    }
                    crate::tast::TypeKind::Dynamic => {
                        // Already Dynamic (boxed) — no boxing needed
                        return None;
                    }
                    _ => type_id,
                }
            } else {
                type_id
            }
        };

        let ir_type = self.convert_type(concrete_type_id);
        let (is_string, is_enum, is_function, is_class_like) = {
            let type_table = self.type_table;
            let ti = type_table.get(concrete_type_id);
            (
                ti.map(|t| matches!(t.kind, crate::tast::TypeKind::String))
                    .unwrap_or(false),
                ti.map(|t| matches!(t.kind, crate::tast::TypeKind::Enum { .. }))
                    .unwrap_or(false),
                ti.map(|t| matches!(t.kind, crate::tast::TypeKind::Function { .. }))
                    .unwrap_or(false),
                // Class / Interface / Anonymous / Array share the
                // "pointer-with-type_id-header" runtime representation. Without
                // this branch the catch-all below boxes them via
                // `haxe_box_haxestring_ptr`, mis-tagging them as `TYPE_STRING`
                // and breaking `Type.typeof`/`Type.getClass` on the `Dynamic`.
                ti.map(|t| {
                    matches!(
                        t.kind,
                        crate::tast::TypeKind::Class { .. }
                            | crate::tast::TypeKind::Interface { .. }
                            | crate::tast::TypeKind::Anonymous { .. }
                            | crate::tast::TypeKind::Array { .. }
                    )
                })
                .unwrap_or(false),
            )
        };

        // Enum values need haxe_box_reference_ptr to preserve the enum's type_id.
        // Tag with the stable runtime id, not the context-local TypeId, so the
        // box agrees with RTTI registration and with `is`/reflection lookups.
        if is_enum {
            let type_id_u32 = self.runtime_type_id(concrete_type_id);
            let type_id_const = self.builder.build_const(IrValue::U32(type_id_u32))?;
            let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
            // Cast the raw i64 discriminant (or boxed enum ptr) to *u8
            let actual_reg_type = self
                .builder
                .get_register_type(value)
                .unwrap_or(ir_type.clone());
            let as_ptr = if matches!(&actual_reg_type, IrType::Ptr(_)) {
                value
            } else {
                self.builder
                    .build_cast(value, actual_reg_type, ptr_u8.clone())
                    .unwrap_or(value)
            };
            let box_func = self.get_or_register_extern_function(
                "haxe_box_reference_ptr",
                vec![ptr_u8.clone(), IrType::U32],
                ptr_u8.clone(),
            );
            return self
                .builder
                .build_call_direct(box_func, vec![as_ptr, type_id_const], ptr_u8);
        }

        if is_function {
            // Function values are represented as closure pointers at runtime.
            // Normalize to *u8 and tag with TYPE_FUNCTION for Reflect/Type parity.
            let actual_reg_type = self
                .builder
                .get_register_type(value)
                .unwrap_or(ir_type.clone());
            let as_ptr = if matches!(&actual_reg_type, IrType::Ptr(_)) {
                value
            } else if matches!(&actual_reg_type, IrType::Function { .. }) {
                let as_i64 = self
                    .builder
                    .build_bitcast(value, IrType::I64)
                    .unwrap_or(value);
                self.builder
                    .build_cast(as_i64, IrType::I64, ptr_u8.clone())
                    .unwrap_or(as_i64)
            } else {
                self.builder
                    .build_cast(value, actual_reg_type, ptr_u8.clone())
                    .unwrap_or(value)
            };
            let box_func = self.get_or_register_extern_function(
                "haxe_box_function_ptr",
                vec![ptr_u8.clone()],
                ptr_u8.clone(),
            );
            return self
                .builder
                .build_call_direct(box_func, vec![as_ptr], ptr_u8);
        }

        if is_class_like {
            // `haxe_box_class_instance` reads the runtime type_id from the
            // instance's `__type_id` header at offset 0 rather than taking a
            // TypeId argument: the TAST TypeId here need not match the IR TypeId
            // in the header (different id namespaces). Header-driven tagging is
            // what lets `Type.typeof`/`Type.getClass` recover the class identity.
            let actual_reg_type = self
                .builder
                .get_register_type(value)
                .unwrap_or(ir_type.clone());
            let as_ptr = if matches!(&actual_reg_type, IrType::Ptr(_)) {
                value
            } else {
                self.builder
                    .build_cast(value, actual_reg_type, ptr_u8.clone())
                    .unwrap_or(value)
            };
            let box_func = self.get_or_register_extern_function(
                "haxe_box_class_instance",
                vec![ptr_u8.clone()],
                ptr_u8.clone(),
            );
            self.builder
                .build_call_direct(box_func, vec![as_ptr], ptr_u8)
        } else if is_string || matches!(&ir_type, IrType::Ptr(_)) {
            // String, plus any `Ptr` fallback that isn't class/interface/anon/array
            // (e.g. type-erased opaque pointers from extern declarations), boxes as
            // a HaxeString DynamicValue. The register may be I64 (type-erased) even
            // though the resolved type is Ptr.
            let actual_reg_type = self
                .builder
                .get_register_type(value)
                .unwrap_or(ir_type.clone());
            let as_ptr = if matches!(&actual_reg_type, IrType::Ptr(_)) {
                value
            } else {
                // i64 → *u8 cast (for type-erased string pointers)
                self.builder
                    .build_cast(value, actual_reg_type, ptr_u8.clone())
                    .unwrap_or(value)
            };
            let box_func = self.get_or_register_extern_function(
                "haxe_box_haxestring_ptr",
                vec![ptr_u8.clone()],
                ptr_u8.clone(),
            );
            self.builder
                .build_call_direct(box_func, vec![as_ptr], ptr_u8)
        } else if matches!(&ir_type, IrType::I32) {
            let as_i64 = self
                .builder
                .build_cast(value, IrType::I32, IrType::I64)
                .unwrap_or(value);
            let box_func = self.get_or_register_extern_function(
                "haxe_box_int_ptr",
                vec![IrType::I64],
                ptr_u8.clone(),
            );
            self.builder
                .build_call_direct(box_func, vec![as_i64], ptr_u8)
        } else if matches!(&ir_type, IrType::I64) {
            // i64 — box as int (type-erased int or fallback)
            let box_func = self.get_or_register_extern_function(
                "haxe_box_int_ptr",
                vec![IrType::I64],
                ptr_u8.clone(),
            );
            self.builder
                .build_call_direct(box_func, vec![value], ptr_u8)
        } else if matches!(&ir_type, IrType::F64) {
            let box_func = self.get_or_register_extern_function(
                "haxe_box_float_ptr",
                vec![IrType::F64],
                ptr_u8.clone(),
            );
            self.builder
                .build_call_direct(box_func, vec![value], ptr_u8)
        } else if matches!(&ir_type, IrType::Bool) {
            let box_func = self.get_or_register_extern_function(
                "haxe_box_bool_ptr",
                vec![IrType::Bool],
                ptr_u8.clone(),
            );
            self.builder
                .build_call_direct(box_func, vec![value], ptr_u8)
        } else {
            // Unknown type — can't box
            None
        }
    }

    /// Coerce a value to i64 for anonymous object field storage.
    /// Ints pass through, floats are bitcast, pointers are cast to i64.
    pub(crate) fn coerce_to_i64(&mut self, value: IrId, type_id: TypeId) -> Option<IrId> {
        let ir_type = self.convert_type(type_id);
        match &ir_type {
            IrType::I64 => Some(value),
            IrType::I32 | IrType::U64 | IrType::Bool => {
                self.builder.build_cast(value, ir_type.clone(), IrType::I64)
            }
            IrType::F64 => {
                // Float: bitcast f64 to i64 (preserves bits)
                self.builder.build_bitcast(value, IrType::I64)
            }
            IrType::F32 => {
                let as_f64 = self.builder.build_cast(value, IrType::F32, IrType::F64)?;
                self.builder.build_bitcast(as_f64, IrType::I64)
            }
            IrType::Ptr(_) => self.builder.build_cast(value, ir_type.clone(), IrType::I64),
            _ => self.builder.build_cast(value, ir_type.clone(), IrType::I64),
        }
    }

    /// Reconcile an extern call's native return type with the Haxe-declared
    /// type at the callsite.
    ///
    /// A native `i64` count consumed as Haxe `Int` must narrow to `i32`, and an
    /// `i64`/`f64` mismatch must convert by value. Left at the native width, the
    /// downstream arg-adaptation in `IrBuilder::build_call_direct` treats an
    /// `I64 -> F64` pairing as generic type erasure and bitcasts it.
    ///
    /// Only numeric scalars are reconciled: `Any`/pointer/void declarations
    /// are genuine erased slots where the bits, not the value, are wanted.
    pub(crate) fn reconcile_extern_return(
        &mut self,
        value: IrId,
        native_ty: &IrType,
        declared_ty: &IrType,
    ) -> IrId {
        if native_ty == declared_ty {
            return value;
        }
        let is_numeric = |t: &IrType| {
            matches!(
                t,
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
            )
        };
        if !is_numeric(native_ty) || !is_numeric(declared_ty) {
            return value;
        }
        self.builder
            .build_cast(value, native_ty.clone(), declared_ty.clone())
            .unwrap_or(value)
    }

    /// Coerce a raw i64 value from anonymous object storage back to the target
    /// type. Inverse of `coerce_to_i64`.
    pub(crate) fn coerce_from_i64(&mut self, value: IrId, type_id: TypeId) -> Option<IrId> {
        let ir_type = self.convert_type(type_id);
        match &ir_type {
            IrType::I64 => Some(value),
            IrType::I32 | IrType::U64 | IrType::Bool => {
                self.builder.build_cast(value, IrType::I64, ir_type.clone())
            }
            IrType::F64 => {
                // i64 -> f64 bitcast (preserves bits)
                self.builder.build_bitcast(value, IrType::F64)
            }
            IrType::F32 => {
                let as_f64 = self.builder.build_bitcast(value, IrType::F64)?;
                self.builder.build_cast(as_f64, IrType::F64, IrType::F32)
            }
            IrType::Ptr(_) => self.builder.build_cast(value, IrType::I64, ir_type.clone()),
            _ => self.builder.build_cast(value, IrType::I64, ir_type.clone()),
        }
    }
}
