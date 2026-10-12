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
        let resolved = self.resolve_field_type_id(field_type_id, generic_info);
        if matches!(
            self.type_table.get(resolved).map(|ty| &ty.kind),
            Some(TypeKind::TypeParameter { .. })
        ) {
            return IrType::I64;
        }
        self.convert_type(resolved)
    }

    /// Resolve payload parameters against the constructor's result type.
    pub(crate) fn resolve_field_type_id(
        &self,
        field_type_id: TypeId,
        generic_info: &Option<(TypeId, Vec<TypeId>)>,
    ) -> TypeId {
        let mut bindings = BTreeMap::new();
        if let Some((result, actual_args)) = generic_info {
            if let Some(TypeKind::Enum { type_args, .. }) =
                self.type_table.get(*result).map(|ty| &ty.kind)
            {
                for (&declared, &actual) in type_args.iter().zip(actual_args) {
                    self.bind_type_parameters(declared, actual, &mut bindings, 0);
                }
            }
        }
        self.resolve_payload_type_id(field_type_id, &bindings, 0)
    }

    pub(crate) fn bind_type_parameters(
        &self,
        declared: TypeId,
        actual: TypeId,
        bindings: &mut BTreeMap<SymbolId, TypeId>,
        depth: u32,
    ) {
        if depth > 32 {
            return;
        }
        let declared = self.resolve_through_aliases(declared);
        let actual = self.resolve_through_aliases(actual);
        let (Some(declared), Some(actual_info)) =
            (self.type_table.get(declared), self.type_table.get(actual))
        else {
            return;
        };
        match (&declared.kind, &actual_info.kind) {
            (TypeKind::TypeParameter { symbol_id, .. }, _) => {
                bindings.entry(*symbol_id).or_insert(actual);
            }
            (
                TypeKind::Array { element_type: left },
                TypeKind::Array {
                    element_type: right,
                },
            )
            | (TypeKind::Optional { inner_type: left }, TypeKind::Optional { inner_type: right }) => {
                self.bind_type_parameters(*left, *right, bindings, depth + 1)
            }
            (
                TypeKind::Class {
                    symbol_id: left,
                    type_args: declared,
                }
                | TypeKind::Interface {
                    symbol_id: left,
                    type_args: declared,
                }
                | TypeKind::Enum {
                    symbol_id: left,
                    type_args: declared,
                }
                | TypeKind::Abstract {
                    symbol_id: left,
                    type_args: declared,
                    ..
                },
                TypeKind::Class {
                    symbol_id: right,
                    type_args: actual,
                }
                | TypeKind::Interface {
                    symbol_id: right,
                    type_args: actual,
                }
                | TypeKind::Enum {
                    symbol_id: right,
                    type_args: actual,
                }
                | TypeKind::Abstract {
                    symbol_id: right,
                    type_args: actual,
                    ..
                },
            ) if left == right => {
                for (&left, &right) in declared.iter().zip(actual) {
                    self.bind_type_parameters(left, right, bindings, depth + 1);
                }
            }
            (
                TypeKind::GenericInstance {
                    base_type: left,
                    type_args: declared,
                    ..
                },
                TypeKind::GenericInstance {
                    base_type: right,
                    type_args: actual,
                    ..
                },
            ) if left == right => {
                for (&left, &right) in declared.iter().zip(actual) {
                    self.bind_type_parameters(left, right, bindings, depth + 1);
                }
            }
            _ => {}
        }
    }

    fn resolve_payload_type_id(
        &self,
        field_type_id: TypeId,
        bindings: &BTreeMap<SymbolId, TypeId>,
        depth: u32,
    ) -> TypeId {
        if depth > 32 {
            return field_type_id;
        }
        let Some(mut kind) = self.type_table.get(field_type_id).map(|t| t.kind.clone()) else {
            return field_type_id;
        };
        match &mut kind {
            TypeKind::TypeParameter { symbol_id, .. } => {
                return bindings.get(symbol_id).copied().unwrap_or(field_type_id);
            }
            TypeKind::Enum { type_args, .. }
            | TypeKind::Class { type_args, .. }
            | TypeKind::Interface { type_args, .. }
            | TypeKind::Abstract { type_args, .. }
            | TypeKind::TypeAlias { type_args, .. }
            | TypeKind::GenericInstance { type_args, .. } => {
                for arg in type_args {
                    *arg = self.resolve_payload_type_id(*arg, bindings, depth + 1);
                }
            }
            TypeKind::Array { element_type }
            | TypeKind::Optional {
                inner_type: element_type,
            } => {
                *element_type = self.resolve_payload_type_id(*element_type, bindings, depth + 1);
            }
            TypeKind::Function {
                params,
                return_type,
                ..
            } => {
                for param in params {
                    *param = self.resolve_payload_type_id(*param, bindings, depth + 1);
                }
                *return_type = self.resolve_payload_type_id(*return_type, bindings, depth + 1);
            }
            _ => return field_type_id,
        }
        // TAST substitution interns concrete composite payload types before MIR.
        self.type_table
            .find_type_with_kind(&kind)
            .unwrap_or(field_type_id)
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
