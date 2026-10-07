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
        let func_id = self.builder.module.alloc_function_id();
        // A MIR module's name is its package; sibling files need distinct
        // closure identities too. Escape punctuation so package separators
        // cannot collide with underscores in identifiers.
        let source_name = std::path::Path::new(&self.builder.module.source_file)
            .file_stem()
            .and_then(|name| name.to_str())
            .unwrap_or("module");
        let module_name = format!("{}.{}", self.builder.module.name, source_name);
        let module_prefix: String = module_name
            .bytes()
            .map(|byte| {
                if byte.is_ascii_alphanumeric() {
                    char::from(byte).to_string()
                } else {
                    format!("_{byte:02x}")
                }
            })
            .collect();
        let lambda_name = format!("<lambda_{}__{}>", module_prefix, self.lambda_counter);
        self.lambda_counter += 1;

        let type_params = self
            .builder
            .current_function
            .and_then(|id| self.builder.module.functions.get(&id))
            .map(|function| function.signature.type_params.clone())
            .unwrap_or_default();
        let type_param_names: Vec<_> = type_params.iter().map(|param| param.name.clone()).collect();
        // A closure created by a generic body specializes with that body.
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
            let param_type = self.convert_type_or_type_var(param.ty, &type_param_names);
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

        let signature = IrFunctionSignature {
            parameters: func_params,
            return_type: IrType::Any, // PLACEHOLDER - will be inferred
            calling_convention: CallingConvention::Haxe,
            can_throw: false,
            type_params,
            uses_sret: false,
        };

        let symbol_id = SymbolId::from_raw(1000000 + func_id.0);
        let lambda_function = IrFunction::new(func_id, symbol_id, lambda_name, signature);
        let entry_block = lambda_function.entry_block();

        self.builder.module.add_function(lambda_function);

        LambdaContext {
            func_id,
            entry_block,
            param_offset: 1, // env_ptr is always first param
            env_layout,
        }
    }

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
        lambda_type: TypeId,
    ) -> Option<IrFunctionId> {
        // Pass 1: Create skeleton with placeholder signature
        let context = self.generate_lambda_skeleton(params, captures);

        // Pass 2: Lower body and infer return type from actual MIR
        self.lower_lambda_body(context, params, body, lambda_type)
    }
}
