//! Devirtualization Pass
//!
//! Converts indirect function calls (CallIndirect) to direct calls (CallDirect)
//! when the target function can be statically determined.
//!
//! Primary targets:
//! - Interface method calls where the concrete type is known from a recent allocation
//!   or cast in the same function (store-to-load forwarding for function pointers)
//! - Closure calls where the function pointer is a known constant

use super::optimization::{OptimizationPass, OptimizationResult};
use super::{IrFunction, IrFunctionId, IrId, IrInstruction, IrModule, IrType, IrValue};
use std::collections::HashMap;

pub struct DevirtualizationPass;

impl DevirtualizationPass {
    pub fn new() -> Self {
        Self
    }
}

/// Tracks what a register holds — used for store-to-load forwarding.
#[derive(Clone, Debug)]
enum ValueOrigin {
    /// Register holds a function reference (known function ID)
    FunctionRef(IrFunctionId),
    /// Register holds a value loaded from base_ptr + offset
    LoadFrom { base: IrId, offset: i64 },
    /// Register is base_ptr + constant offset
    PtrOffset { base: IrId, offset: i64 },
}

impl OptimizationPass for DevirtualizationPass {
    fn name(&self) -> &'static str {
        "devirtualization"
    }

    fn run_on_module(&mut self, module: &mut IrModule) -> OptimizationResult {
        let mut result = OptimizationResult::unchanged();

        // Collect all function names → IrFunctionId for resolving function refs
        let func_name_to_id: HashMap<String, IrFunctionId> = module
            .functions
            .iter()
            .map(|(&id, f)| (f.name.clone(), id))
            .collect();

        let func_ids: Vec<IrFunctionId> = module.functions.keys().copied().collect();
        for func_id in func_ids {
            let r = devirtualize_function(
                module.functions.get_mut(&func_id).unwrap(),
                &func_name_to_id,
            );
            result = result.combine(r);
        }

        result
    }
}

fn devirtualize_function(
    function: &mut IrFunction,
    _func_name_to_id: &HashMap<String, IrFunctionId>,
) -> OptimizationResult {
    // Phase 1: Analyze — build value origin map across all blocks
    let mut origins: HashMap<IrId, ValueOrigin> = HashMap::new();
    // Track stores: (base_ptr, offset) → stored value register
    let mut stores: HashMap<(IrId, i64), IrId> = HashMap::new();

    let block_ids: Vec<_> = function.cfg.blocks.keys().copied().collect();

    for &block_id in &block_ids {
        let block = &function.cfg.blocks[&block_id];
        for inst in &block.instructions {
            match inst {
                // Track function references
                IrInstruction::FunctionRef { dest, func_id } => {
                    origins.insert(*dest, ValueOrigin::FunctionRef(*func_id));
                }

                // Track constant values (for pointer offsets)
                IrInstruction::Const { dest, value } => {
                    let v = match value {
                        IrValue::I32(n) => Some(*n as i64),
                        IrValue::I64(n) => Some(*n),
                        _ => None,
                    };
                    if let Some(v) = v {
                        origins.insert(
                            *dest,
                            ValueOrigin::PtrOffset {
                                base: *dest,
                                offset: v,
                            },
                        );
                    }
                }

                // Track PtrAdd: dest = base + offset_reg
                IrInstruction::PtrAdd {
                    dest, ptr, offset, ..
                } => {
                    // If offset is a known constant, track the pointer arithmetic
                    if let Some(ValueOrigin::PtrOffset {
                        offset: off_val, ..
                    }) = origins.get(offset)
                    {
                        origins.insert(
                            *dest,
                            ValueOrigin::PtrOffset {
                                base: *ptr,
                                offset: *off_val,
                            },
                        );
                    }
                }

                // Track stores: Store(ptr, value) at known base+offset
                IrInstruction::Store { ptr, value, .. } => {
                    // Direct store to base pointer (offset 0)
                    stores.insert((*ptr, 0), *value);

                    // Store via ptr that is base+offset
                    if let Some(ValueOrigin::PtrOffset { base, offset }) = origins.get(ptr) {
                        stores.insert((*base, *offset), *value);
                    }
                }

                // Track loads: dest = Load(ptr)
                IrInstruction::Load { dest, ptr, .. } => {
                    // Direct load from base (offset 0)
                    if let Some(&stored_val) = stores.get(&(*ptr, 0)) {
                        if let Some(origin) = origins.get(&stored_val) {
                            origins.insert(*dest, origin.clone());
                        }
                    }

                    // Load via ptr that is base+offset
                    if let Some(ValueOrigin::PtrOffset { base, offset }) = origins.get(ptr).cloned()
                    {
                        if let Some(&stored_val) = stores.get(&(base, offset)) {
                            if let Some(origin) = origins.get(&stored_val) {
                                origins.insert(*dest, origin.clone());
                            }
                        } else {
                            origins.insert(*dest, ValueOrigin::LoadFrom { base, offset });
                        }
                    }
                }

                // Track casts (propagate origin through type casts)
                IrInstruction::Cast {
                    dest, src: source, ..
                }
                | IrInstruction::BitCast {
                    dest, src: source, ..
                } => {
                    if let Some(origin) = origins.get(source).cloned() {
                        origins.insert(*dest, origin);
                    }
                }

                _ => {}
            }
        }
    }

    // Phase 2: Transform — replace CallIndirect with CallDirect where possible
    let mut devirtualized = 0;

    for &block_id in &block_ids {
        let block = function.cfg.blocks.get_mut(&block_id).unwrap();
        for inst in &mut block.instructions {
            if let IrInstruction::CallIndirect {
                dest,
                func_ptr,
                args,
                signature,
                arg_ownership,
                is_tail_call,
            } = inst
            {
                // Check if func_ptr resolves to a known function
                if let Some(ValueOrigin::FunctionRef(known_func_id)) = origins.get(func_ptr) {
                    let func_id = *known_func_id;
                    let dest_val = *dest;
                    let args_val = args.clone();
                    let ownership_val = arg_ownership.clone();
                    let type_args = Vec::new();
                    let tail = *is_tail_call;

                    *inst = IrInstruction::CallDirect {
                        dest: dest_val,
                        func_id: func_id,
                        args: args_val,
                        arg_ownership: ownership_val,
                        type_args,
                        is_tail_call: tail,
                    };

                    devirtualized += 1;
                }
            }
        }
    }

    if devirtualized > 0 {
        let mut result = OptimizationResult::changed();
        result
            .stats
            .insert("devirtualized_calls".to_string(), devirtualized);
        result
    } else {
        OptimizationResult::unchanged()
    }
}
