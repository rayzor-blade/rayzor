//! Resolves field types and array element types.

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
    /// Resolve a field's TypeId to an IrType, substituting generic type parameters
    /// if generic_info is provided (base_type + concrete type_args).
    pub(crate) fn resolve_field_type(
        &self,
        field_type_id: TypeId,
        generic_info: &Option<(TypeId, Vec<TypeId>)>,
    ) -> IrType {
        let type_table = self.type_table;

        // Check if this TypeId is a type parameter that needs substitution
        if let Some(type_info) = type_table.get(field_type_id) {
            if let crate::tast::TypeKind::TypeParameter { symbol_id, .. } = &type_info.kind {
                if let Some((base_type, concrete_args)) = generic_info {
                    // Find which index this type parameter corresponds to in the base enum's type_args
                    if let Some(base_info) = type_table.get(*base_type) {
                        let base_type_args = match &base_info.kind {
                            crate::tast::TypeKind::Enum { type_args, .. } => type_args,
                            _ => return IrType::I64,
                        };
                        for (idx, &param_type_id) in base_type_args.iter().enumerate() {
                            if let Some(param_info) = type_table.get(param_type_id) {
                                if let crate::tast::TypeKind::TypeParameter {
                                    symbol_id: param_sym,
                                    ..
                                } = &param_info.kind
                                {
                                    if param_sym == symbol_id {
                                        // Found match — substitute with concrete type
                                        if let Some(&concrete_type_id) = concrete_args.get(idx) {
                                            return self.convert_type(concrete_type_id);
                                        }
                                    }
                                }
                            }
                        }
                        // Positional lookup failed (concrete_args may be shorter than base type_args).
                        // If there's exactly 1 concrete arg, use it as the substitution.
                        if concrete_args.len() == 1 {
                            let concrete_id = concrete_args[0];
                            if let Some(ci) = type_table.get(concrete_id) {
                                if !matches!(ci.kind, crate::tast::TypeKind::TypeParameter { .. }) {
                                    return self.convert_type(concrete_id);
                                }
                            }
                        }
                    }
                }
                // Unresolved type parameter — treat as I64
                return IrType::I64;
            }
        }

        // Not a type parameter — convert directly
        self.convert_type(field_type_id)
    }

    /// Like resolve_field_type but returns the concrete TypeId instead of IrType.
    pub(crate) fn resolve_field_type_id(
        &self,
        field_type_id: TypeId,
        generic_info: &Option<(TypeId, Vec<TypeId>)>,
    ) -> TypeId {
        let type_table = self.type_table;
        if let Some(type_info) = type_table.get(field_type_id) {
            if let crate::tast::TypeKind::TypeParameter { symbol_id, .. } = &type_info.kind {
                if let Some((base_type, concrete_args)) = generic_info {
                    if let Some(base_info) = type_table.get(*base_type) {
                        let base_type_args = match &base_info.kind {
                            crate::tast::TypeKind::Enum { type_args, .. } => type_args,
                            _ => return field_type_id,
                        };
                        for (idx, &param_type_id) in base_type_args.iter().enumerate() {
                            if let Some(param_info) = type_table.get(param_type_id) {
                                if let crate::tast::TypeKind::TypeParameter {
                                    symbol_id: param_sym,
                                    ..
                                } = &param_info.kind
                                {
                                    if param_sym == symbol_id {
                                        if let Some(&concrete_type_id) = concrete_args.get(idx) {
                                            return concrete_type_id;
                                        }
                                    }
                                }
                            }
                        }
                        if concrete_args.len() == 1 {
                            let concrete_id = concrete_args[0];
                            if let Some(ci) = type_table.get(concrete_id) {
                                if !matches!(ci.kind, crate::tast::TypeKind::TypeParameter { .. }) {
                                    return concrete_id;
                                }
                            }
                        }
                    }
                }
            }
        }
        field_type_id
    }

    /// Extract the element type from an Array type.
    /// If the type is Array<T>, returns Some(T). Otherwise returns None.
    pub(crate) fn get_array_element_type(&self, type_id: TypeId) -> Option<TypeId> {
        use crate::tast::TypeKind;
        let type_table = self.type_table;
        let type_ref = type_table.get(type_id)?;
        match &type_ref.kind {
            TypeKind::Array { element_type, .. } => Some(*element_type),
            _ => None,
        }
    }
}
