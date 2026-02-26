#![allow(
    unused_imports,
    unused_variables,
    dead_code,
    unreachable_patterns,
    unused_mut,
    unused_assignments,
    unused_parens
)]
#![allow(
    clippy::single_component_path_imports,
    clippy::for_kv_map,
    clippy::explicit_auto_deref
)]
#![allow(
    clippy::println_empty_string,
    clippy::len_zero,
    clippy::useless_vec,
    clippy::field_reassign_with_default
)]
#![allow(
    clippy::needless_borrow,
    clippy::redundant_closure,
    clippy::bool_assert_comparison
)]
#![allow(
    clippy::empty_line_after_doc_comments,
    clippy::useless_format,
    clippy::clone_on_copy
)]
//! Test MIR Interpreter execution with real code
//!
//! Demonstrates the 5-tier compilation system starting with the interpreter:
//! - Phase 0: Interpreter (instant startup, ~5-10x native)
//! - Phase 1-4: JIT compilation tiers
//!
//! Run with: cargo run --example test_mir_interpreter

use compiler::codegen::mir_interpreter::{InterpValue, MirInterpreter};
use compiler::codegen::profiling::ProfileConfig;
use compiler::codegen::tiered_backend::{OptimizationTier, TieredBackend, TieredConfig};
use compiler::ir::{
    BinaryOp, CallingConvention, CompareOp, IrBlockId, IrFunction, IrFunctionId,
    IrFunctionSignature, IrId, IrInstruction, IrModule, IrParameter, IrTerminator, IrType,
};
use compiler::tast::SymbolId;

fn main() {
    println!("=== MIR Interpreter Test ===\n");

    // Test 1: Direct interpreter execution
    test_direct_interpreter();

    // Test 2: Tiered backend with interpreter startup
    test_tiered_with_interpreter();

    // Test 3: Arithmetic operations
    test_arithmetic_operations();

    println!("\n=== All Interpreter Tests Passed! ===");
}

/// Test direct MIR interpreter execution without tiered backend
fn test_direct_interpreter() {
    println!("--- Test 1: Direct Interpreter Execution ---");

    // Create a simple add function: add(a, b) -> a + b
    let add_function = create_add_function();

    // Create module
    let mut module = IrModule::new("interpreter_test".to_string(), "test.hx".to_string());
    let func_id = IrFunctionId(0);
    module.functions.insert(func_id, add_function);

    // Create interpreter and execute
    let mut interpreter = MirInterpreter::new();

    // Execute: add(5, 3) = 8
    let result = interpreter.execute(
        &module,
        func_id,
        vec![InterpValue::I64(5), InterpValue::I64(3)],
    );

    match result {
        Ok(InterpValue::I64(n)) => {
            assert_eq!(n, 8, "Expected 5 + 3 = 8, got {}", n);
            println!("  add(5, 3) = {} [OK]", n);
        }
        Ok(other) => panic!("Expected I64, got {:?}", other),
        Err(e) => panic!("Interpreter error: {}", e),
    }

    // Execute: add(100, 200) = 300
    let result = interpreter.execute(
        &module,
        func_id,
        vec![InterpValue::I64(100), InterpValue::I64(200)],
    );

    match result {
        Ok(InterpValue::I64(n)) => {
            assert_eq!(n, 300, "Expected 100 + 200 = 300, got {}", n);
            println!("  add(100, 200) = {} [OK]", n);
        }
        Ok(other) => panic!("Expected I64, got {:?}", other),
        Err(e) => panic!("Interpreter error: {}", e),
    }

    println!("  Direct interpreter test PASSED\n");
}

/// Test tiered backend starting in interpreted mode
fn test_tiered_with_interpreter() {
    println!("--- Test 2: Tiered Backend with Interpreter Startup ---");

    // Create a multiply function: mul(a, b) -> a * b
    let mul_function = create_multiply_function();

    // Create module
    let mut module = IrModule::new("tiered_test".to_string(), "test.hx".to_string());
    let func_id = IrFunctionId(0);
    module.functions.insert(func_id, mul_function);

    // Configure tiered backend with interpreter startup
    let config = TieredConfig {
        profile_config: ProfileConfig {
            interpreter_threshold: 5, // JIT after 5 calls
            warm_threshold: 20,
            hot_threshold: 100,
            blazing_threshold: 500,
            sample_rate: 1,
        },
        enable_background_optimization: false, // Disable for deterministic test
        optimization_check_interval_ms: 50,
        max_parallel_optimizations: 1,
        verbosity: 1,
        start_interpreted: true, // Start in interpreter mode
        bailout_strategy: compiler::codegen::BailoutStrategy::Quick,
        max_tier_promotions: 0,
        enable_stack_traces: false,
    };

    // Create tiered backend
    let mut backend = TieredBackend::new(config).expect("Failed to create tiered backend");

    // Compile module (starts in interpreted mode)
    backend
        .compile_module(module)
        .expect("Failed to compile module");

    // Verify function starts at Interpreted tier
    let stats = backend.get_statistics();
    assert_eq!(
        stats.interpreted_functions, 1,
        "Expected 1 interpreted function"
    );
    println!(
        "  Function starts at Phase 0 (Interpreted): {} functions",
        stats.interpreted_functions
    );

    // Execute via interpreter
    for i in 1..=10 {
        let result =
            backend.execute_function(func_id, vec![InterpValue::I64(i), InterpValue::I64(2)]);

        match result {
            Ok(InterpValue::I64(n)) => {
                let expected = i * 2;
                assert_eq!(n, expected, "Expected {} * 2 = {}, got {}", i, expected, n);
                if i == 1 || i == 5 || i == 10 {
                    println!("  mul({}, 2) = {} [Call #{}]", i, n, i);
                }
            }
            Ok(other) => panic!("Expected I64, got {:?}", other),
            Err(e) => panic!("Execution error: {}", e),
        }
    }

    // Check promotion status
    let stats = backend.get_statistics();
    println!("  After 10 calls:");
    println!("    Interpreted (P0): {}", stats.interpreted_functions);
    println!("    Baseline (P1): {}", stats.baseline_functions);
    println!(
        "    Queued for optimization: {}",
        stats.queued_for_optimization
    );

    println!("  Tiered backend interpreter test PASSED\n");
}

/// Test various arithmetic operations in the interpreter
fn test_arithmetic_operations() {
    println!("--- Test 3: Arithmetic Operations ---");

    let mut interpreter = MirInterpreter::new();

    // Test subtraction
    let sub_func = create_subtract_function();
    let mut module = IrModule::new("arith_test".to_string(), "test.hx".to_string());
    let sub_id = IrFunctionId(0);
    module.functions.insert(sub_id, sub_func);

    let result = interpreter.execute(
        &module,
        sub_id,
        vec![InterpValue::I64(10), InterpValue::I64(3)],
    );

    match result {
        Ok(InterpValue::I64(n)) => {
            assert_eq!(n, 7, "Expected 10 - 3 = 7, got {}", n);
            println!("  sub(10, 3) = {} [OK]", n);
        }
        Ok(other) => panic!("Expected I64, got {:?}", other),
        Err(e) => panic!("Interpreter error: {}", e),
    }

    // Test with negative numbers
    let result = interpreter.execute(
        &module,
        sub_id,
        vec![InterpValue::I64(5), InterpValue::I64(10)],
    );

    match result {
        Ok(InterpValue::I64(n)) => {
            assert_eq!(n, -5, "Expected 5 - 10 = -5, got {}", n);
            println!("  sub(5, 10) = {} [OK]", n);
        }
        Ok(other) => panic!("Expected I64, got {:?}", other),
        Err(e) => panic!("Interpreter error: {}", e),
    }

    // Test conditional (max function)
    let max_func = create_max_function();
    let mut module2 = IrModule::new("max_test".to_string(), "test.hx".to_string());
    let max_id = IrFunctionId(0);
    module2.functions.insert(max_id, max_func);

    let result = interpreter.execute(
        &module2,
        max_id,
        vec![InterpValue::I64(7), InterpValue::I64(3)],
    );

    match result {
        Ok(InterpValue::I64(n)) => {
            assert_eq!(n, 7, "Expected max(7, 3) = 7, got {}", n);
            println!("  max(7, 3) = {} [OK]", n);
        }
        Ok(other) => panic!("Expected I64, got {:?}", other),
        Err(e) => panic!("Interpreter error: {}", e),
    }

    let result = interpreter.execute(
        &module2,
        max_id,
        vec![InterpValue::I64(2), InterpValue::I64(9)],
    );

    match result {
        Ok(InterpValue::I64(n)) => {
            assert_eq!(n, 9, "Expected max(2, 9) = 9, got {}", n);
            println!("  max(2, 9) = {} [OK]", n);
        }
        Ok(other) => panic!("Expected I64, got {:?}", other),
        Err(e) => panic!("Interpreter error: {}", e),
    }

    println!("  Arithmetic operations test PASSED\n");
}

/// Create add(a, b) -> a + b
fn create_add_function() -> IrFunction {
    let func_id = IrFunctionId(0);
    let symbol_id = SymbolId::from_raw(1);

    let signature = IrFunctionSignature {
        parameters: vec![
            IrParameter {
                name: "a".to_string(),
                ty: IrType::I64,
                reg: IrId::new(0),
                by_ref: false,
            },
            IrParameter {
                name: "b".to_string(),
                ty: IrType::I64,
                reg: IrId::new(1),
                by_ref: false,
            },
        ],
        return_type: IrType::I64,
        calling_convention: CallingConvention::Haxe,
        can_throw: false,
        type_params: Vec::new(),
        uses_sret: false,
    };

    let mut function = IrFunction::new(func_id, symbol_id, "add".to_string(), signature);

    let entry_block = function.entry_block();
    let a_reg = function.get_param_reg(0).unwrap();
    let b_reg = function.get_param_reg(1).unwrap();
    let result_reg = function.alloc_reg();

    if let Some(block) = function.cfg.get_block_mut(entry_block) {
        // result = a + b
        block.instructions.push(IrInstruction::BinOp {
            dest: result_reg,
            op: BinaryOp::Add,
            left: a_reg,
            right: b_reg,
        });
        block.set_terminator(IrTerminator::Return {
            value: Some(result_reg),
        });
    }

    function
}

/// Create mul(a, b) -> a * b
fn create_multiply_function() -> IrFunction {
    let func_id = IrFunctionId(0);
    let symbol_id = SymbolId::from_raw(2);

    let signature = IrFunctionSignature {
        parameters: vec![
            IrParameter {
                name: "a".to_string(),
                ty: IrType::I64,
                reg: IrId::new(0),
                by_ref: false,
            },
            IrParameter {
                name: "b".to_string(),
                ty: IrType::I64,
                reg: IrId::new(1),
                by_ref: false,
            },
        ],
        return_type: IrType::I64,
        calling_convention: CallingConvention::Haxe,
        can_throw: false,
        type_params: Vec::new(),
        uses_sret: false,
    };

    let mut function = IrFunction::new(func_id, symbol_id, "mul".to_string(), signature);

    let entry_block = function.entry_block();
    let a_reg = function.get_param_reg(0).unwrap();
    let b_reg = function.get_param_reg(1).unwrap();
    let result_reg = function.alloc_reg();

    if let Some(block) = function.cfg.get_block_mut(entry_block) {
        // result = a * b
        block.instructions.push(IrInstruction::BinOp {
            dest: result_reg,
            op: BinaryOp::Mul,
            left: a_reg,
            right: b_reg,
        });
        block.set_terminator(IrTerminator::Return {
            value: Some(result_reg),
        });
    }

    function
}

/// Create sub(a, b) -> a - b
fn create_subtract_function() -> IrFunction {
    let func_id = IrFunctionId(0);
    let symbol_id = SymbolId::from_raw(3);

    let signature = IrFunctionSignature {
        parameters: vec![
            IrParameter {
                name: "a".to_string(),
                ty: IrType::I64,
                reg: IrId::new(0),
                by_ref: false,
            },
            IrParameter {
                name: "b".to_string(),
                ty: IrType::I64,
                reg: IrId::new(1),
                by_ref: false,
            },
        ],
        return_type: IrType::I64,
        calling_convention: CallingConvention::Haxe,
        can_throw: false,
        type_params: Vec::new(),
        uses_sret: false,
    };

    let mut function = IrFunction::new(func_id, symbol_id, "sub".to_string(), signature);

    let entry_block = function.entry_block();
    let a_reg = function.get_param_reg(0).unwrap();
    let b_reg = function.get_param_reg(1).unwrap();
    let result_reg = function.alloc_reg();

    if let Some(block) = function.cfg.get_block_mut(entry_block) {
        // result = a - b
        block.instructions.push(IrInstruction::BinOp {
            dest: result_reg,
            op: BinaryOp::Sub,
            left: a_reg,
            right: b_reg,
        });
        block.set_terminator(IrTerminator::Return {
            value: Some(result_reg),
        });
    }

    function
}

/// Create max(a, b) -> if a > b then a else b
fn create_max_function() -> IrFunction {
    let func_id = IrFunctionId(0);
    let symbol_id = SymbolId::from_raw(4);

    let signature = IrFunctionSignature {
        parameters: vec![
            IrParameter {
                name: "a".to_string(),
                ty: IrType::I64,
                reg: IrId::new(0),
                by_ref: false,
            },
            IrParameter {
                name: "b".to_string(),
                ty: IrType::I64,
                reg: IrId::new(1),
                by_ref: false,
            },
        ],
        return_type: IrType::I64,
        calling_convention: CallingConvention::Haxe,
        can_throw: false,
        type_params: Vec::new(),
        uses_sret: false,
    };

    let mut function = IrFunction::new(func_id, symbol_id, "max".to_string(), signature);

    let entry_block = function.entry_block();
    let a_reg = function.get_param_reg(0).unwrap();
    let b_reg = function.get_param_reg(1).unwrap();
    let cmp_reg = function.alloc_reg();

    // Create blocks for conditional
    let then_block = function.cfg.create_block();
    let else_block = function.cfg.create_block();

    // Entry block: compare a > b
    if let Some(block) = function.cfg.get_block_mut(entry_block) {
        block.instructions.push(IrInstruction::Cmp {
            dest: cmp_reg,
            op: CompareOp::Gt,
            left: a_reg,
            right: b_reg,
        });
        block.set_terminator(IrTerminator::CondBranch {
            condition: cmp_reg,
            true_target: then_block,
            false_target: else_block,
        });
    }

    // Then block: return a
    if let Some(block) = function.cfg.get_block_mut(then_block) {
        block.set_terminator(IrTerminator::Return { value: Some(a_reg) });
    }

    // Else block: return b
    if let Some(block) = function.cfg.get_block_mut(else_block) {
        block.set_terminator(IrTerminator::Return { value: Some(b_reg) });
    }

    function
}
