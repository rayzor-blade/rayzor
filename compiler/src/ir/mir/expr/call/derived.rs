//! `@:derive`d instance methods reached through a variable callee — the
//! receiver arrives desugared as the first argument.

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
    pub(crate) fn try_derived_instance_call(
        &mut self,
        expr: &HirExpr,
        fell_through: &mut bool,
    ) -> Option<IrId> {
        let HirExprKind::Call {
            callee,
            args,
            is_method,
            ..
        } = &expr.kind
        else {
            unreachable!("try_derived_instance_call on a non-Call expression")
        };
        let HirExprKind::Variable { symbol, .. } = &callee.kind else {
            *fell_through = true;
            return None;
        };
        let vname = self
            .symbol_table
            .get_symbol(*symbol)
            .and_then(|s| self.string_interner.get(s.name))
            .unwrap_or("?");
        // @:derive(Hash) synthetic hashCode(). Instance calls arrive with a
        // variable callee and the receiver desugared to args[0], so arity 1
        // means a receiver and no user arguments.
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
        *fell_through = true;
        None
    }
}
