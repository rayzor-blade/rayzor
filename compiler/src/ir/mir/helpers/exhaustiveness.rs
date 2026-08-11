//! Switch exhaustiveness over enums and `Bool`.

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
    /// Check exhaustiveness of a switch statement on enum or bool types.
    /// Emits diagnostics for non-exhaustive matches — does not affect code generation.
    pub(crate) fn check_switch_exhaustiveness(
        &mut self,
        scrutinee: &HirExpr,
        cases: &[HirMatchCase],
    ) {
        // Resolve scrutinee type to determine if it's an enum
        let type_table = self.type_table;
        let type_info = match type_table.get(scrutinee.ty) {
            Some(info) => info,
            None => return,
        };

        let enum_symbol = match &type_info.kind {
            crate::tast::TypeKind::Enum { symbol_id, .. } => *symbol_id,
            crate::tast::TypeKind::GenericInstance { base_type, .. } => {
                match type_table.get(*base_type) {
                    Some(base_info) => match &base_info.kind {
                        crate::tast::TypeKind::Enum { symbol_id, .. } => *symbol_id,
                        _ => return,
                    },
                    None => return,
                }
            }
            crate::tast::TypeKind::Bool => {
                self.check_bool_exhaustiveness(scrutinee, cases);
                return;
            }
            _ => return,
        };

        // Get all variant names from enum definition
        let variant_symbols = match self.symbol_table.get_enum_variants(enum_symbol) {
            Some(v) => v.clone(),
            None => return,
        };

        let all_variants: BTreeSet<InternedString> = variant_symbols
            .iter()
            .filter_map(|sid| self.symbol_table.get_symbol(*sid).map(|s| s.name))
            .collect();

        if all_variants.is_empty() {
            return;
        }

        // Collect covered variants from patterns
        let (covered, has_wildcard) = self.collect_covered_enum_variants(cases);

        if has_wildcard {
            return;
        }

        // Compute missing variants
        let missing: Vec<InternedString> = all_variants
            .iter()
            .filter(|v| !covered.contains(v))
            .copied()
            .collect();

        if missing.is_empty() {
            return;
        }

        // Emit warning
        let enum_name = self
            .symbol_table
            .get_symbol(enum_symbol)
            .and_then(|s| self.string_interner.get(s.name))
            .unwrap_or("?");
        let missing_names: Vec<&str> = missing
            .iter()
            .filter_map(|n| self.string_interner.get(*n))
            .collect();
        let span = Self::source_location_to_span(&scrutinee.source_location);
        let mut builder = diagnostics::DiagnosticBuilder::warning(
            format!("non-exhaustive switch on enum '{}'", enum_name),
            span.clone(),
        )
        .code("W0006")
        .label(
            span,
            format!("missing variants: {}", missing_names.join(", ")),
        );

        for name in &missing_names {
            builder = builder.note(format!("add case for `{}`", name));
        }
        builder = builder.help("consider adding a default case with `default:`");

        self.diagnostics.push(builder.build());
    }

    /// Check if a switch on a Bool type is exhaustive.
    pub(crate) fn check_bool_exhaustiveness(
        &mut self,
        scrutinee: &HirExpr,
        cases: &[HirMatchCase],
    ) {
        let mut has_true = false;
        let mut has_false = false;
        let mut has_wildcard = false;

        for case in cases {
            if case.guard.is_some() {
                continue;
            }
            for pattern in &case.patterns {
                match pattern {
                    HirPattern::Literal(HirLiteral::Bool(true)) => has_true = true,
                    HirPattern::Literal(HirLiteral::Bool(false)) => has_false = true,
                    HirPattern::Wildcard | HirPattern::Variable { .. } => has_wildcard = true,
                    HirPattern::Or(subs) => {
                        for sub in subs {
                            match sub {
                                HirPattern::Literal(HirLiteral::Bool(true)) => has_true = true,
                                HirPattern::Literal(HirLiteral::Bool(false)) => has_false = true,
                                HirPattern::Wildcard | HirPattern::Variable { .. } => {
                                    has_wildcard = true
                                }
                                _ => {}
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        if has_wildcard || (has_true && has_false) {
            return;
        }

        let missing = match (has_true, has_false) {
            (false, false) => "true, false",
            (true, false) => "false",
            (false, true) => "true",
            _ => return,
        };

        let span = Self::source_location_to_span(&scrutinee.source_location);
        let diagnostic = diagnostics::DiagnosticBuilder::warning(
            format!("non-exhaustive switch on Bool — missing: {}", missing),
            span.clone(),
        )
        .code("W0007")
        .label(span, "missing cases")
        .help("consider adding a default case with `default:`")
        .build();
        self.diagnostics.push(diagnostic);
    }

    /// Collect enum variant names covered by switch case patterns.
    /// Returns (covered_variants, has_wildcard).
    pub(crate) fn collect_covered_enum_variants(
        &self,
        cases: &[HirMatchCase],
    ) -> (BTreeSet<InternedString>, bool) {
        let mut covered = BTreeSet::new();
        let mut has_wildcard = false;

        for case in cases {
            // Cases with guards don't guarantee coverage (guard may fail at runtime)
            let case_has_guard = case.guard.is_some();

            for pattern in &case.patterns {
                self.collect_variants_from_pattern(
                    pattern,
                    case_has_guard,
                    &mut covered,
                    &mut has_wildcard,
                );
            }
        }

        (covered, has_wildcard)
    }
}
