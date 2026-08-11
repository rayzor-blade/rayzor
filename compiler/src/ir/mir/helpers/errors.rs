//! Lowering errors, and the spans and type hints attached to them.

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
    pub(crate) fn add_error(&mut self, msg: &str, location: SourceLocation) {
        self.errors.push(LoweringError {
            message: msg.to_string(),
            location,
        });
    }

    /// Best-effort short type-name for a TypeId, for hint messages.
    /// Falls back to a debug-ish form when the type table can't render
    /// a clean name (we don't ship a full pretty-printer here — the
    /// goal is to help the user write `var x:T = …`, not produce
    /// canonical source).
    pub(crate) fn format_type_for_hint(&self, type_id: TypeId) -> String {
        let type_table = self.type_table;
        let ti = match type_table.get(type_id) {
            Some(ti) => ti,
            None => return "<unknown>".to_string(),
        };
        let sym_name = |sid: SymbolId| -> String {
            self.symbol_table
                .get_symbol(sid)
                .and_then(|s| self.string_interner.get(s.name))
                .unwrap_or("<class>")
                .to_string()
        };
        let render_args = |args: &[TypeId]| -> String {
            args.iter()
                .map(|&t| self.format_type_for_hint(t))
                .collect::<Vec<_>>()
                .join(", ")
        };
        match &ti.kind {
            crate::tast::TypeKind::Int => "Int".to_string(),
            crate::tast::TypeKind::Float => "Float".to_string(),
            crate::tast::TypeKind::Bool => "Bool".to_string(),
            crate::tast::TypeKind::String => "String".to_string(),
            crate::tast::TypeKind::Void => "Void".to_string(),
            crate::tast::TypeKind::Dynamic => "Dynamic".to_string(),
            crate::tast::TypeKind::Array { element_type } => {
                format!("Array<{}>", self.format_type_for_hint(*element_type))
            }
            crate::tast::TypeKind::Class {
                symbol_id,
                type_args,
            }
            | crate::tast::TypeKind::Interface {
                symbol_id,
                type_args,
            }
            | crate::tast::TypeKind::Enum {
                symbol_id,
                type_args,
            } => {
                let base = sym_name(*symbol_id);
                if type_args.is_empty() {
                    base
                } else {
                    format!("{}<{}>", base, render_args(type_args))
                }
            }
            crate::tast::TypeKind::GenericInstance {
                base_type,
                type_args,
                ..
            } => {
                let base = self.format_type_for_hint(*base_type);
                if type_args.is_empty() {
                    base
                } else {
                    format!("{}<{}>", base, render_args(type_args))
                }
            }
            _ => format!("{:?}", ti.kind),
        }
    }

    /// Convert a compiler SourceLocation to a diagnostics SourceSpan.
    pub(crate) fn source_location_to_span(loc: &SourceLocation) -> diagnostics::SourceSpan {
        let start = diagnostics::SourcePosition::new(
            loc.line as usize,
            loc.column as usize,
            loc.byte_offset as usize,
        );
        let end = diagnostics::SourcePosition::new(
            loc.line as usize,
            (loc.column + 1) as usize,
            (loc.byte_offset + 1) as usize,
        );
        diagnostics::SourceSpan::new(start, end, diagnostics::FileId::new(loc.file_id as usize))
    }

    /// Qualified name of the symbol that maps to `func_id` in the local or
    /// external function maps. Lowering-order-independent fallback for class
    /// identification when the IrFunction's own qualified name is not yet
    /// populated (see `func_not_other_class`).
    pub(crate) fn symbol_qualified_name_for_func(&self, func_id: IrFunctionId) -> Option<String> {
        for (sym, &fid) in self
            .function_map
            .iter()
            .chain(self.external_function_map.iter())
        {
            if fid == func_id {
                if let Some(q) = self
                    .symbol_table
                    .get_symbol(*sym)
                    .and_then(|s| s.qualified_name)
                    .and_then(|qn| self.string_interner.get(qn))
                {
                    return Some(q.to_string());
                }
            }
        }
        None
    }
}
