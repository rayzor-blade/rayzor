//! Lambdas emitted as standalone IR functions.

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
    /// PASS 1: Create lambda skeleton with placeholder signature
    pub(crate) fn generate_lambda_skeleton(
        &mut self,
        params: &[HirParam],
        captures: &[HirCapture],
    ) -> LambdaContext {
        // Allocate function ID
        let func_id = self.builder.module.alloc_function_id();
        // Qualify the lambda name with its module. `lambda_counter` only
        // resets per lowering instance, so a bare `<lambda_N>` is unique
        // WITHIN a file but not across them — every file's first closure is
        // `<lambda_0>`. The backend binds functions by name, so two identically
        // named closures from different modules COLLAPSE: the SpinPool worker
        // spawn closure and a user file's first closure both being `<lambda_0>`
        // made a `Thread.spawn` capture the wrong body/env and dispatch a pool
        // worker into `rayzor_channel_receive`. Prefix with the sanitized module
        // name so cross-module closures never alias.
        let module_prefix: String = self
            .builder
            .module
            .name
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { '_' })
            .collect();
        let lambda_name = format!("<lambda_{}__{}>", module_prefix, self.lambda_counter);
        self.lambda_counter += 1;

        // Build environment layout if we have captures
        let env_layout = if !captures.is_empty() {
            Some(EnvironmentLayout::new(captures, |ty| self.convert_type(ty)))
        } else {
            None
        };

        // Build parameters: env* is ALWAYS first param (CallIndirect always prepends env_ptr)
        let mut func_params = Vec::new();
        let mut next_reg_id = 0u32;

        // Always add env_ptr as first parameter - even for capture-less lambdas.
        // Cranelift's CallIndirect always loads env_ptr from closure struct and
        // prepends it as the first argument, so the function signature must match.
        func_params.push(IrParameter {
            name: "env".to_string(),
            ty: IrType::Ptr(Box::new(IrType::Void)),
            reg: IrId::new(next_reg_id),
            by_ref: false,
        });
        next_reg_id += 1;

        for param in params {
            let param_type = self.convert_type(param.ty);
            let param_name = self
                .string_interner
                .get(param.name)
                .unwrap_or("<param>")
                .to_string();

            func_params.push(IrParameter {
                name: param_name,
                ty: param_type,
                reg: IrId::new(next_reg_id),
                by_ref: false,
            });
            next_reg_id += 1;
        }

        // Create PLACEHOLDER signature
        let signature = IrFunctionSignature {
            parameters: func_params,
            return_type: IrType::Any, // PLACEHOLDER - will be inferred
            calling_convention: CallingConvention::Haxe,
            can_throw: false,
            type_params: vec![],
            uses_sret: false,
        };

        // Create empty function
        let symbol_id = SymbolId::from_raw(1000000 + func_id.0);
        let lambda_function = IrFunction::new(func_id, symbol_id, lambda_name, signature);
        let entry_block = lambda_function.entry_block();

        // Add to module
        self.builder.module.add_function(lambda_function);

        LambdaContext {
            func_id,
            entry_block,
            param_offset: 1, // env_ptr is always first param
            env_layout,
        }
    }

    // ========================================================================
    // Lambda Generation - Now Using Two-Pass Architecture
    // ========================================================================

    /// Generate a lambda function using two-pass architecture
    ///
    /// Creates a new function that takes (env*, params...) as arguments,
    /// where env* is a pointer to a struct containing captured variables.
    ///
    /// Two-Pass Architecture:
    /// - Pass 1: Create skeleton with placeholder signature
    /// - Pass 2: Lower body, infer types from actual MIR, update signature
    pub(crate) fn generate_lambda_function(
        &mut self,
        params: &[HirParam],
        body: &HirExpr,
        captures: &[HirCapture],
        _lambda_type: TypeId, // No longer needed - type inferred from MIR
    ) -> Option<IrFunctionId> {
        // TWO-PASS LAMBDA LOWERING
        // Pass 1: Create skeleton with placeholder signature
        let context = self.generate_lambda_skeleton(params, captures);

        // Pass 2: Lower body and infer return type from actual MIR
        self.lower_lambda_body(context, params, body)
    }
}
