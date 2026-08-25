//! Binds pattern variables and collects the variables a pattern introduces.

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
    pub(crate) fn bind_pattern(&mut self, pattern: &HirPattern, value: IrId) {
        self.bind_pattern_with_scrutinee_type(pattern, value, None);
    }

    /// Bind a pattern with type information (registers locals for Cranelift)
    pub(crate) fn bind_pattern_with_type(
        &mut self,
        pattern: &HirPattern,
        value: IrId,
        ty: Option<TypeId>,
        is_mutable: bool,
    ) {
        match pattern {
            HirPattern::Variable { name, symbol } => {
                self.symbol_map.insert(*symbol, value);

                // Register as local so Cranelift can find the type
                if let Some(type_id) = ty {
                    let var_type_from_hint = self.convert_type(type_id);

                    // The register type can be more specific than the hint: unresolved
                    // generic returns (Thread<T>), or a vague HIR Ptr(Void) over what is
                    // really a String.
                    let actual_reg_type = self.builder.get_register_type(value);
                    let var_type = if let Some(ref actual_type) = actual_reg_type {
                        let hint_is_void_ptr = matches!(&var_type_from_hint, IrType::Ptr(inner) if matches!(**inner, IrType::Void));
                        let actual_is_specific = !matches!(actual_type, IrType::Ptr(inner) if matches!(**inner, IrType::Void));

                        let actual_is_ptr = matches!(actual_type, IrType::Ptr(_));
                        let hint_is_scalar = matches!(
                            &var_type_from_hint,
                            IrType::I32 | IrType::I64 | IrType::Bool | IrType::F32 | IrType::F64
                        );

                        // Float-vs-integer disagreement: the register holds the actual
                        // value, so its type is authoritative for storage. A `Float` hint
                        // over an i64 (a `Usize` address whose HIR type decayed) would make
                        // every read coerce i64→f64→i64 at loop phis and call arguments,
                        // corrupting the carried pointer.
                        let float_int_mismatch = matches!(actual_type, IrType::F32 | IrType::F64)
                            != matches!(&var_type_from_hint, IrType::F32 | IrType::F64);

                        if (hint_is_void_ptr && actual_is_specific)
                            || (actual_is_ptr && hint_is_scalar)
                            || float_int_mismatch
                        {
                            actual_type.clone()
                        } else {
                            var_type_from_hint
                        }
                    } else {
                        var_type_from_hint
                    };

                    let local_name = self.interned_str(*name).to_string();
                    if let Some(func) = self.builder.current_function_mut() {
                        func.locals.insert(
                            value,
                            crate::ir::IrLocal {
                                name: local_name,
                                ty: var_type.clone(),
                                mutable: is_mutable,
                                source_location: IrSourceLocation::unknown(),
                                allocation: crate::ir::AllocationHint::Stack,
                            },
                        );
                    }
                }
                let _ = self.box_capture_binding(*symbol, value);
            }
            _ => {
                self.bind_pattern(pattern, value);
            }
        }
    }

    pub(crate) fn bind_pattern_with_scrutinee_type(
        &mut self,
        pattern: &HirPattern,
        value: IrId,
        scrutinee_type: Option<TypeId>,
    ) {
        match pattern {
            HirPattern::Variable { symbol, .. } => {
                self.symbol_map.insert(*symbol, value);
                let _ = self.box_capture_binding(*symbol, value);
            }
            HirPattern::Wildcard => {
                // Wildcard doesn't bind anything.
            }
            HirPattern::Tuple(patterns) => {
                for (i, p) in patterns.iter().enumerate() {
                    if let Some(elem) = self.builder.build_extract_value(value, vec![i as u32]) {
                        self.bind_pattern(p, elem);
                    }
                }
            }
            HirPattern::Literal(_) => {
                // Literals in patterns match, they don't bind.
            }
            HirPattern::Constructor {
                enum_type,
                variant,
                fields,
            } => {
                // Prefer the scrutinee type when it resolves to an enum — it carries the
                // concrete generic args; otherwise use the pattern's enum_type.
                let effective_type = scrutinee_type
                    .filter(|t| *t != TypeId::invalid())
                    .filter(|t| self.resolve_enum_symbol(*t).is_some())
                    .unwrap_or(*enum_type);
                let field_types = self.get_enum_variant_field_types(effective_type, *variant);

                // effective_type first, then the pattern's enum_type if it won't resolve.
                let enum_symbol = self
                    .resolve_enum_symbol(effective_type)
                    .or_else(|| self.resolve_enum_symbol(*enum_type));
                let is_boxed = enum_symbol.map_or(false, |s| self.enum_is_boxed(s));
                if !is_boxed || fields.is_empty() {
                    return;
                }

                // Boxed enum: bitcast value (i64) to pointer for GEP
                let enum_ptr = match self
                    .builder
                    .build_bitcast(value, IrType::Ptr(Box::new(IrType::I8)))
                {
                    Some(p) => p,
                    None => return,
                };

                for (i, field_pattern) in fields.iter().enumerate() {
                    if matches!(field_pattern, HirPattern::Wildcard) {
                        continue;
                    }
                    // Field at byte offset 8 + i*8 (tag is at offset 0, 8 bytes with padding)
                    let field_offset = (8 + i * 8) as i64;
                    if let Some(offset_val) = self.builder.build_int(field_offset, IrType::I64) {
                        if let Some(field_ptr) = self.builder.build_gep(
                            enum_ptr,
                            vec![offset_val],
                            IrType::Ptr(Box::new(IrType::I8)),
                        ) {
                            let (resolved_type, resolved_type_id) = field_types
                                .get(i)
                                .cloned()
                                .unwrap_or((IrType::I64, TypeId::invalid()));

                            let (load_type, needs_bitcast_after) = match &resolved_type {
                                // Pointer types: load as raw I64, then bitcast to pointer
                                IrType::Ptr(_) | IrType::String => {
                                    (IrType::I64, Some(resolved_type.clone()))
                                }
                                // Value types stored in 8 bytes
                                IrType::I32 | IrType::Bool => (IrType::I64, None),
                                IrType::I64 | IrType::F64 => (resolved_type.clone(), None),
                                // Unknown/Any: load as I64 for runtime dispatch
                                _ => (IrType::I64, None),
                            };

                            let field_ptr_typed = self
                                .builder
                                .build_bitcast(field_ptr, IrType::Ptr(Box::new(load_type.clone())));
                            if let Some(fpt) = field_ptr_typed {
                                if let Some(mut field_val) = self.builder.build_load(fpt, load_type)
                                {
                                    if let Some(target_type) = needs_bitcast_after {
                                        if let Some(cast_val) =
                                            self.builder.build_bitcast(field_val, target_type)
                                        {
                                            field_val = cast_val;
                                        }
                                    }
                                    if let HirPattern::Variable { symbol, .. } = field_pattern {
                                        self.symbol_ir_types.insert(*symbol, resolved_type.clone());
                                        if resolved_type_id != TypeId::invalid() {
                                            self.symbol_type_ids.insert(*symbol, resolved_type_id);
                                        }
                                    }
                                    self.bind_pattern(field_pattern, field_val);
                                }
                            }
                        }
                    }
                }
            }
            HirPattern::Array { .. } => {
                // Array patterns need runtime length checks
                self.add_error(
                    "Array patterns not yet supported in MIR lowering",
                    SourceLocation::unknown(),
                );
            }
            HirPattern::Object { .. } => {
                // Object patterns need field extraction
                self.add_error(
                    "Object patterns not yet supported in MIR lowering",
                    SourceLocation::unknown(),
                );
            }
            HirPattern::Typed { pattern, .. } => {
                // Type annotations in patterns don't affect binding
                self.bind_pattern(pattern, value);
            }
            HirPattern::Guard { pattern, .. } => {
                // Guards are conditions, not bindings
                self.bind_pattern(pattern, value);
            }
            HirPattern::Or(patterns) => {
                // Only the first alternative is bound; binding all of them is
                // not implemented.
                if let Some(first) = patterns.first() {
                    self.bind_pattern(first, value);
                }
            }
        }
    }

    /// Collect all variable symbols from a pattern
    pub(crate) fn collect_pattern_variables(
        &self,
        pattern: &HirPattern,
        variables: &mut std::collections::BTreeSet<SymbolId>,
    ) {
        match pattern {
            HirPattern::Variable { symbol, .. } => {
                variables.insert(*symbol);
            }
            HirPattern::Tuple(patterns) => {
                for p in patterns {
                    self.collect_pattern_variables(p, variables);
                }
            }
            HirPattern::Constructor { fields, .. } => {
                for p in fields {
                    self.collect_pattern_variables(p, variables);
                }
            }
            HirPattern::Array { elements, rest } => {
                for p in elements {
                    self.collect_pattern_variables(p, variables);
                }
                if let Some(rest_pat) = rest {
                    self.collect_pattern_variables(rest_pat, variables);
                }
            }
            _ => {}
        }
    }

    /// Recursively walk a pattern collecting covered enum variants.
    pub(crate) fn collect_variants_from_pattern(
        &self,
        pattern: &HirPattern,
        has_guard: bool,
        covered: &mut BTreeSet<InternedString>,
        has_wildcard: &mut bool,
    ) {
        match pattern {
            HirPattern::Constructor { variant, .. } => {
                if !has_guard {
                    covered.insert(*variant);
                }
            }
            HirPattern::Wildcard => {
                if !has_guard {
                    *has_wildcard = true;
                }
            }
            HirPattern::Variable { .. } => {
                if !has_guard {
                    *has_wildcard = true;
                }
            }
            HirPattern::Or(sub_patterns) => {
                for sub in sub_patterns {
                    self.collect_variants_from_pattern(sub, has_guard, covered, has_wildcard);
                }
            }
            HirPattern::Guard { .. } => {
                // Guard patterns don't count for exhaustiveness
            }
            HirPattern::Typed { pattern, .. } => {
                self.collect_variants_from_pattern(pattern, has_guard, covered, has_wildcard);
            }
            _ => {}
        }
    }
}
