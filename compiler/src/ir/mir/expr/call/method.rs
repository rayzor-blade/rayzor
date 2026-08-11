//! Method calls written as `receiver.method(args)`.

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
    pub(crate) fn try_method_call(
        &mut self,
        expr: &HirExpr,
        result_type: IrType,
        fell_through: &mut bool,
    ) -> Option<IrId> {
        let HirExprKind::Call {
            callee,
            args,
            is_method,
            ..
        } = &expr.kind
        else {
            unreachable!("try_method_call on a non-Call expression")
        };
        if let HirExprKind::Field { object, field } = &callee.kind {
            // This is a method call: object.method(args)
            // The method symbol should be in our function_map (local or external)
            let method_name_interned = self.symbol_table.get_symbol(*field).map(|s| s.name);
            let method_name = method_name_interned.and_then(|name| self.string_interner.get(name));
            let in_local = self.function_map.contains_key(field);
            let in_external = self.external_function_map.contains_key(field);
            debug!(
                "[Method call] method={:?}, field={:?}, in_local={}, in_external={}",
                method_name, field, in_local, in_external
            );

            // @:cstruct synthetic cdef() method — return C typedef string
            probe!(self.try_cdef_method_call(expr));

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
            probe!(self.try_wgsl_method_call(expr));

            // @:gpuStruct synthetic methods — gpuDef/gpuSize/gpuAlignment
            probe!(self.try_gpu_struct_method_call(expr));

            // Check for interface dispatch: if object has interface type,
            // load the method function pointer from the fat pointer and call indirectly
            probe!(self.try_interface_method_call(expr));

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
            probe!(self.try_resolved_method_call(expr, maybe_func_id, result_type.clone()));
        }
        *fell_through = true;
        None
    }
}
