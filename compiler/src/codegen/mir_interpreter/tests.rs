use super::*;
use crate::ir::instructions::OwnershipMode;
use crate::ir::mir_builder::MirBuilder;
use crate::ir::{
    BinaryOp, CallingConvention, CompareOp, IrBlockId, IrGlobalId, IrId, IrInstruction, IrPhiNode,
    IrTerminator, IrValue,
};

fn runtime() -> MirInterpreter {
    let plugin = rayzor_runtime::plugin_impl::get_plugin();
    MirInterpreter::with_symbols(&plugin.runtime_symbols())
}

fn push(module: &mut IrModule, f: IrFunctionId, block: IrBlockId, inst: IrInstruction) {
    module
        .functions
        .get_mut(&f)
        .unwrap()
        .cfg
        .blocks
        .get_mut(&block)
        .unwrap()
        .instructions
        .push(inst);
}

fn phi(module: &mut IrModule, f: IrFunctionId, block: IrBlockId, node: IrPhiNode) {
    module
        .functions
        .get_mut(&f)
        .unwrap()
        .cfg
        .blocks
        .get_mut(&block)
        .unwrap()
        .phi_nodes
        .push(node);
}

fn call(dest: Option<IrId>, func_id: IrFunctionId, args: Vec<IrId>) -> IrInstruction {
    IrInstruction::CallDirect {
        dest,
        func_id,
        arg_ownership: args.iter().map(|_| OwnershipMode::Move).collect(),
        args,
        type_args: Vec::new(),
        is_tail_call: false,
    }
}

fn extern_fn(b: &mut MirBuilder, name: &str, params: &[IrType], ret: IrType) -> IrFunctionId {
    let mut f = b.begin_function(name);
    for (i, p) in params.iter().enumerate() {
        f = f.param(format!("p{}", i), p.clone());
    }
    f.returns(ret)
        .calling_convention(CallingConvention::C)
        .extern_func()
        .build()
}

#[test]
fn i32_arithmetic_wraps_at_32_bits() {
    let mut b = MirBuilder::new("t");
    let f = b
        .begin_function("add")
        .param("a", IrType::I32)
        .param("b", IrType::I32)
        .returns(IrType::I32)
        .build();
    b.set_current_function(f);
    let entry = b.create_block("entry");
    b.set_insert_point(entry);
    let (x, y) = (b.get_param(0), b.get_param(1));
    let s = b.bin_op(BinaryOp::Add, x, y);
    b.ret(Some(s));
    let m = b.finish();
    let r = MirInterpreter::new()
        .execute(&m, f, vec![InterpValue::I32(i32::MAX), InterpValue::I32(1)])
        .unwrap();
    assert!(matches!(r, InterpValue::I32(i32::MIN)), "{:?}", r);
}

#[test]
fn i64_keeps_all_64_bits() {
    let mut b = MirBuilder::new("t");
    let f = b.begin_function("wide").returns(IrType::I64).build();
    b.set_current_function(f);
    let entry = b.create_block("entry");
    b.set_insert_point(entry);
    let c = b.const_i64(0x7fff_ffff_ffff_fff0);
    let one = b.const_i64(0x0f);
    let s = b.bin_op(BinaryOp::Add, c, one);
    b.ret(Some(s));
    let m = b.finish();
    let r = MirInterpreter::new().execute(&m, f, vec![]).unwrap();
    assert!(
        matches!(r, InterpValue::I64(0x7fff_ffff_ffff_ffff)),
        "{:?}",
        r
    );
}

#[test]
fn loop_with_phis_sums() {
    let mut b = MirBuilder::new("t");
    let f = b
        .begin_function("sum")
        .param("n", IrType::I64)
        .returns(IrType::I64)
        .build();
    b.set_current_function(f);
    let entry = b.create_block("entry");
    let header = b.create_block("header");
    let body = b.create_block("body");
    let exit = b.create_block("exit");
    let n = b.get_param(0);
    b.set_insert_point(entry);
    let zero = b.const_i64(0);
    b.br(header);
    let (i, acc) = (b.alloc_reg(), b.alloc_reg());
    b.set_insert_point(header);
    let more = b.cmp(CompareOp::Lt, i, n);
    b.cond_br(more, body, exit);
    b.set_insert_point(body);
    let one = b.const_i64(1);
    let acc2 = b.bin_op(BinaryOp::Add, acc, i);
    let i2 = b.bin_op(BinaryOp::Add, i, one);
    b.br(header);
    b.set_insert_point(exit);
    b.ret(Some(acc));
    let mut m = b.finish();
    for (dest, init, next) in [(i, zero, i2), (acc, zero, acc2)] {
        phi(
            &mut m,
            f,
            header,
            IrPhiNode {
                dest,
                incoming: vec![(entry, init), (body, next)],
                ty: IrType::I64,
            },
        );
    }
    let mut interp = MirInterpreter::new();
    interp.set_max_iterations(u64::MAX);
    let r = interp.execute(&m, f, vec![InterpValue::I64(100)]).unwrap();
    assert!(matches!(r, InterpValue::I64(4950)), "{:?}", r);
}

#[test]
fn calls_and_floats() {
    let mut b = MirBuilder::new("t");
    let g = b
        .begin_function("fma_ish")
        .param("a", IrType::F64)
        .param("b", IrType::F64)
        .returns(IrType::F64)
        .build();
    let f = b.begin_function("caller").returns(IrType::F64).build();
    b.set_current_function(g);
    let e = b.create_block("entry");
    b.set_insert_point(e);
    let (x, y) = (b.get_param(0), b.get_param(1));
    let p = b.bin_op(BinaryOp::FMul, x, y);
    let half = b.const_value(IrValue::F64(1.5));
    let s = b.bin_op(BinaryOp::FAdd, p, half);
    b.ret(Some(s));
    b.set_current_function(f);
    let e = b.create_block("entry");
    b.set_insert_point(e);
    let a = b.const_value(IrValue::F64(2.0));
    let c = b.const_value(IrValue::F64(3.25));
    let r = b.call(g, vec![a, c]).unwrap();
    b.ret(Some(r));
    let m = b.finish();
    let r = MirInterpreter::new().execute(&m, f, vec![]).unwrap();
    assert!(matches!(r, InterpValue::F64(v) if v == 8.0), "{:?}", r);
}

extern "C" fn many_args(
    a0: i64,
    a1: i64,
    a2: i64,
    a3: i64,
    a4: i64,
    a5: i64,
    a6: i64,
    a7: i64,
    a8: i64,
    a9: i64,
    x: f64,
    y: f64,
) -> f64 {
    (a0 + 2 * a1 + 3 * a2 + 4 * a3 + 5 * a4 + 6 * a5 + 7 * a6 + 8 * a7 + 9 * a8 + 10 * a9) as f64
        + x * 100.0
        + y * 1000.0
}

extern "C" fn widened(x: i64) -> i64 {
    x
}

#[test]
fn native_calls_take_stack_arguments_and_c_promotion() {
    let mut b = MirBuilder::new("t");
    let mut params = vec![IrType::I64; 10];
    params.extend([IrType::F64, IrType::F64]);
    let many = extern_fn(&mut b, "many_args", &params, IrType::F64);
    let wide = extern_fn(&mut b, "widened", &[IrType::I32], IrType::I64);
    let f = b.begin_function("caller").returns(IrType::F64).build();
    let g = b.begin_function("negative").returns(IrType::I64).build();
    b.set_current_function(f);
    let e = b.create_block("entry");
    b.set_insert_point(e);
    let mut args: Vec<IrId> = (1..=10).map(|v| b.const_i64(v)).collect();
    args.push(b.const_value(IrValue::F64(0.5)));
    args.push(b.const_value(IrValue::F64(0.25)));
    let r = b.call(many, args).unwrap();
    b.ret(Some(r));
    b.set_current_function(g);
    let e = b.create_block("entry");
    b.set_insert_point(e);
    let minus = b.const_i32(-5);
    let r = b.call(wide, vec![minus]).unwrap();
    b.ret(Some(r));
    let m = b.finish();

    let mut interp = MirInterpreter::new();
    interp.register_symbol("many_args", many_args as *const u8);
    interp.register_symbol("widened", widened as *const u8);
    let expected = (1..=10).map(|i| i * i).sum::<i64>() as f64 + 50.0 + 250.0;
    let r = interp.execute(&m, f, vec![]).unwrap();
    assert!(matches!(r, InterpValue::F64(v) if v == expected), "{:?}", r);
    let r = interp.execute(&m, g, vec![]).unwrap();
    assert!(matches!(r, InterpValue::I64(-5)), "{:?}", r);
}

/// Calls a Haxe closure object `{fn, env}` the way compiled code does.
extern "C" fn call_closure(closure: *const usize, x: i64) -> i64 {
    unsafe {
        let code: extern "C" fn(usize, i64) -> i64 = std::mem::transmute(*closure);
        code(*closure.add(1), x)
    }
}

#[test]
fn native_code_calls_interpreted_functions_through_trampolines() {
    let mut b = MirBuilder::new("t");
    let callback = extern_fn(
        &mut b,
        "call_closure",
        &[IrType::I64, IrType::I64],
        IrType::I64,
    );
    let twice = b
        .begin_function("twice")
        .param("x", IrType::I64)
        .returns(IrType::I64)
        .build();
    let f = b.begin_function("caller").returns(IrType::I64).build();
    b.set_current_function(twice);
    let e = b.create_block("entry");
    b.set_insert_point(e);
    let x = b.get_param(0);
    let two = b.const_i64(2);
    let r = b.bin_op(BinaryOp::Mul, x, two);
    b.ret(Some(r));
    b.set_current_function(f);
    let e = b.create_block("entry");
    b.set_insert_point(e);
    let closure = b.function_ref(twice);
    let arg = b.const_i64(21);
    let r = b.call(callback, vec![closure, arg]).unwrap();
    b.ret(Some(r));
    let m = b.finish();

    let mut interp = runtime();
    interp.register_symbol("call_closure", call_closure as *const u8);
    let r = interp.execute(&m, f, vec![]).unwrap();
    assert!(matches!(r, InterpValue::I64(42)), "{:?}", r);
}

#[test]
fn closures_capture_and_call_indirectly() {
    let mut b = MirBuilder::new("t");
    let lambda = b
        .begin_function("<lambda_0>")
        .param("env", IrType::I64)
        .param("x", IrType::I64)
        .returns(IrType::I64)
        .build();
    let f = b.begin_function("caller").returns(IrType::I64).build();
    b.set_current_function(lambda);
    let le = b.create_block("entry");
    b.set_insert_point(le);
    let env = b.get_param(0);
    let x = b.get_param(1);
    let captured = b.load(env, IrType::I64);
    let r = b.bin_op(BinaryOp::Add, captured, x);
    b.ret(Some(r));
    b.set_current_function(f);
    let e = b.create_block("entry");
    b.set_insert_point(e);
    let forty = b.const_i64(40);
    let two = b.const_i64(2);
    let (closure, result) = (b.alloc_reg(), b.alloc_reg());
    b.ret(Some(result));
    let mut m = b.finish();
    push(
        &mut m,
        f,
        e,
        IrInstruction::MakeClosure {
            dest: closure,
            func_id: lambda,
            captured_values: vec![forty],
        },
    );
    push(
        &mut m,
        f,
        e,
        IrInstruction::CallIndirect {
            dest: Some(result),
            func_ptr: closure,
            args: vec![two],
            signature: IrType::Function {
                params: vec![IrType::I64],
                return_type: Box::new(IrType::I64),
                varargs: false,
            },
            arg_ownership: vec![OwnershipMode::Move],
            is_tail_call: false,
        },
    );
    let r = runtime().execute(&m, f, vec![]).unwrap();
    assert!(matches!(r, InterpValue::I64(42)), "{:?}", r);
}

#[test]
fn string_constants_are_runtime_strings() {
    let mut b = MirBuilder::new("t");
    let f = b.begin_function("s").returns(IrType::String).build();
    b.set_current_function(f);
    let e = b.create_block("entry");
    b.set_insert_point(e);
    let s = b.const_string("hello");
    b.ret(Some(s));
    let m = b.finish();
    let r = runtime().execute(&m, f, vec![]).unwrap();
    let InterpValue::Ptr(p) = r else {
        panic!("{:?}", r)
    };
    let (data, len) = unsafe { (*(p as *const *const u8), *((p + 8) as *const usize)) };
    let text = unsafe { std::slice::from_raw_parts(data, len) };
    assert_eq!(text, b"hello");
}

/// An empty literal's data pointer is a real address, as in compiled code:
/// the runtime reads it to tell a string from a boxed value.
#[test]
fn empty_string_constants_point_at_memory() {
    let mut b = MirBuilder::new("t");
    let f = b.begin_function("s").returns(IrType::String).build();
    b.set_current_function(f);
    let e = b.create_block("entry");
    b.set_insert_point(e);
    let s = b.const_string("");
    b.ret(Some(s));
    let m = b.finish();
    let r = runtime().execute(&m, f, vec![]).unwrap();
    let InterpValue::Ptr(p) = r else {
        panic!("{:?}", r)
    };
    let (data, len) = unsafe { (*(p as *const usize), *((p + 8) as *const usize)) };
    assert_eq!(len, 0);
    assert!(data >= 4096, "data pointer {:#x}", data);
}

#[test]
fn globals_live_in_the_runtime_store() {
    let mut b = MirBuilder::new("t");
    let f = b.begin_function("g").returns(IrType::I64).build();
    b.set_current_function(f);
    let e = b.create_block("entry");
    b.set_insert_point(e);
    let v = b.const_i64(-123_456_789_012);
    let out = b.alloc_reg();
    b.ret(Some(out));
    let mut m = b.finish();
    let id = IrGlobalId(4_000_001);
    push(
        &mut m,
        f,
        e,
        IrInstruction::StoreGlobal {
            global_id: id,
            value: v,
        },
    );
    push(
        &mut m,
        f,
        e,
        IrInstruction::LoadGlobal {
            dest: out,
            global_id: id,
            ty: IrType::I64,
        },
    );
    let r = runtime().execute(&m, f, vec![]).unwrap();
    assert!(matches!(r, InterpValue::I64(-123_456_789_012)), "{:?}", r);
    let load: unsafe extern "C" fn(i64) -> i64 = rayzor_runtime::rayzor_global_load;
    assert_eq!(unsafe { load(4_000_001) }, -123_456_789_012);
}

/// `try { if (i % 2 == 0) throw i; } catch (_) caught++` over many iterations:
/// the try body runs inside a setjmp frame that must end with its handler.
#[test]
fn try_catch_in_a_loop() {
    let mut b = MirBuilder::new("t");
    let push_handler = extern_fn(&mut b, "rayzor_exception_push_handler", &[], IrType::I64);
    let pop_handler = extern_fn(&mut b, "rayzor_exception_pop_handler", &[], IrType::Void);
    let setjmp = extern_fn(&mut b, "_setjmp", &[IrType::I64], IrType::I32);
    let throw = extern_fn(&mut b, "rayzor_throw", &[IrType::I64], IrType::Void);
    let f = b.begin_function("catches").returns(IrType::I64).build();
    b.set_current_function(f);
    let entry = b.create_block("entry");
    let header = b.create_block("header");
    let try_entry = b.create_block("try");
    let body = b.create_block("body");
    let throws = b.create_block("throws");
    let quiet = b.create_block("quiet");
    let landing = b.create_block("landing");
    let latch = b.create_block("latch");
    let exit = b.create_block("exit");
    let (i, caught, caught_next) = (b.alloc_reg(), b.alloc_reg(), b.alloc_reg());

    b.set_insert_point(entry);
    let zero = b.const_i64(0);
    let n = b.const_i64(100_000);
    let one = b.const_i64(1);
    let two = b.const_i64(2);
    let zero32 = b.const_i32(0);
    b.br(header);

    b.set_insert_point(header);
    let more = b.cmp(CompareOp::Lt, i, n);
    b.cond_br(more, try_entry, exit);

    b.set_insert_point(try_entry);
    let buf = b.call(push_handler, vec![]).unwrap();
    let r = b.call(setjmp, vec![buf]).unwrap();
    let normal = b.cmp(CompareOp::Eq, r, zero32);
    b.cond_br(normal, body, landing);

    b.set_insert_point(body);
    let rem = b.bin_op(BinaryOp::Rem, i, two);
    let even = b.cmp(CompareOp::Eq, rem, zero);
    b.cond_br(even, throws, quiet);

    b.set_insert_point(throws);
    b.call(throw, vec![i]);
    b.ret(Some(zero));

    b.set_insert_point(quiet);
    b.call(pop_handler, vec![]);
    b.br(latch);

    b.set_insert_point(landing);
    b.call(pop_handler, vec![]);
    let bumped = b.bin_op(BinaryOp::Add, caught, one);
    b.br(latch);

    b.set_insert_point(latch);
    let i2 = b.bin_op(BinaryOp::Add, i, one);
    b.br(header);

    b.set_insert_point(exit);
    b.ret(Some(caught));
    let mut m = b.finish();
    phi(
        &mut m,
        f,
        header,
        IrPhiNode {
            dest: i,
            incoming: vec![(entry, zero), (latch, i2)],
            ty: IrType::I64,
        },
    );
    phi(
        &mut m,
        f,
        header,
        IrPhiNode {
            dest: caught,
            incoming: vec![(entry, zero), (latch, caught_next)],
            ty: IrType::I64,
        },
    );
    phi(
        &mut m,
        f,
        latch,
        IrPhiNode {
            dest: caught_next,
            incoming: vec![(quiet, caught), (landing, bumped)],
            ty: IrType::I64,
        },
    );
    let mut interp = runtime();
    interp.set_max_iterations(u64::MAX);
    let r = interp.execute(&m, f, vec![]).unwrap();
    assert!(matches!(r, InterpValue::I64(50_000)), "{:?}", r);
}

#[test]
fn struct_returns_through_the_callers_slot() {
    let fields = vec![
        crate::ir::StructField {
            name: "a".into(),
            ty: IrType::I64,
            offset: 0,
        },
        crate::ir::StructField {
            name: "b".into(),
            ty: IrType::I32,
            offset: 8,
        },
    ];
    let ty = IrType::Struct {
        name: "Pair".into(),
        fields,
    };
    let mut b = MirBuilder::new("t");
    let f = b.begin_function("pair").returns(ty.clone()).build();
    b.set_current_function(f);
    let e = b.create_block("entry");
    b.set_insert_point(e);
    let a = b.const_i64(1 << 40);
    let c = b.const_i32(-7);
    let s = b.create_struct(ty, vec![a, c]);
    b.ret(Some(s));
    let m = b.finish();
    let r = MirInterpreter::new().execute(&m, f, vec![]).unwrap();
    let InterpValue::Ptr(p) = r else {
        panic!("{:?}", r)
    };
    unsafe {
        assert_eq!(*(p as *const i64), 1 << 40);
        assert_eq!(*((p + 8) as *const i32), -7);
    }
}

#[test]
fn recursion_reaches_a_thousand_frames() {
    let mut b = MirBuilder::new("t");
    let f = b
        .begin_function("down")
        .param("n", IrType::I64)
        .returns(IrType::I64)
        .build();
    b.set_current_function(f);
    let entry = b.create_block("entry");
    let recurse = b.create_block("recurse");
    let done = b.create_block("done");
    b.set_insert_point(entry);
    let n = b.get_param(0);
    let zero = b.const_i64(0);
    let is_zero = b.cmp(CompareOp::Eq, n, zero);
    b.cond_br(is_zero, done, recurse);
    b.set_insert_point(recurse);
    let one = b.const_i64(1);
    let less = b.bin_op(BinaryOp::Sub, n, one);
    let r = b.call(f, vec![less]).unwrap();
    let r1 = b.bin_op(BinaryOp::Add, r, one);
    b.ret(Some(r1));
    b.set_insert_point(done);
    b.ret(Some(zero));
    let m = b.finish();
    // A main thread's stack, which is where the interpreter runs.
    let r = std::thread::Builder::new()
        .stack_size(8 << 20)
        .spawn(move || {
            let mut interp = MirInterpreter::new();
            interp.set_max_iterations(u64::MAX);
            interp.execute(&m, f, vec![InterpValue::I64(1000)]).unwrap()
        })
        .unwrap()
        .join()
        .unwrap();
    assert!(matches!(r, InterpValue::I64(1000)), "{:?}", r);
}

/// SIMD reaching the interpreter hands the function to compiled code instead
/// of computing something else.
#[test]
fn vector_op_triggers_jit_bailout() {
    let mut builder = MirBuilder::new("simd_test");
    let vec_ty = IrType::vector(IrType::F32, 4);
    let func_id = builder
        .begin_function("simd_add")
        .param("a", vec_ty.clone())
        .param("b", vec_ty.clone())
        .returns(vec_ty.clone())
        .build();
    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);
    let a = builder.get_param(0);
    let b = builder.get_param(1);
    let r = builder.vector_bin_op(BinaryOp::Add, a, b, vec_ty.clone());
    builder.ret(Some(r));
    let module = builder.finish();

    let result = MirInterpreter::new().execute(
        &module,
        func_id,
        vec![InterpValue::I64(0), InterpValue::I64(0)],
    );
    assert!(
        matches!(result, Err(InterpError::JitBailout(id)) if id == func_id),
        "{:?}",
        result
    );
}

#[test]
fn hot_loops_ask_for_compiled_code() {
    let mut b = MirBuilder::new("t");
    let f = b.begin_function("spin").returns(IrType::Void).build();
    b.set_current_function(f);
    let entry = b.create_block("entry");
    let header = b.create_block("header");
    b.set_insert_point(entry);
    b.br(header);
    b.set_insert_point(header);
    b.br(header);
    let m = b.finish();
    let mut interp = MirInterpreter::new();
    interp.set_max_iterations(100);
    let r = interp.execute(&m, f, vec![]);
    assert!(
        matches!(r, Err(InterpError::JitBailout(id)) if id == f),
        "{:?}",
        r
    );
}

static RESUME_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

extern "C" fn count_resume_call() {
    RESUME_CALLS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
}

/// A suspended stack picks up where it stopped: the side effect before the
/// loop happens once, and the sum is the whole sum.
#[test]
fn suspended_stacks_resume_without_replaying() {
    let mut b = MirBuilder::new("t");
    let bump = extern_fn(&mut b, "count_resume_call", &[], IrType::Void);
    let f = b
        .begin_function("sum_after_effect")
        .param("n", IrType::I64)
        .returns(IrType::I64)
        .build();
    let outer = b.begin_function("outer").returns(IrType::I64).build();
    b.set_current_function(f);
    let entry = b.create_block("entry");
    let header = b.create_block("header");
    let body = b.create_block("body");
    let exit = b.create_block("exit");
    let n = b.get_param(0);
    b.set_insert_point(entry);
    b.call(bump, vec![]);
    let zero = b.const_i64(0);
    b.br(header);
    let (i, acc) = (b.alloc_reg(), b.alloc_reg());
    b.set_insert_point(header);
    let more = b.cmp(CompareOp::Lt, i, n);
    b.cond_br(more, body, exit);
    b.set_insert_point(body);
    let one = b.const_i64(1);
    let acc2 = b.bin_op(BinaryOp::Add, acc, i);
    let i2 = b.bin_op(BinaryOp::Add, i, one);
    b.br(header);
    b.set_insert_point(exit);
    b.ret(Some(acc));
    b.set_current_function(outer);
    let e = b.create_block("entry");
    b.set_insert_point(e);
    let count = b.const_i64(1000);
    let r = b.call(f, vec![count]).unwrap();
    let ten = b.const_i64(10);
    let r10 = b.bin_op(BinaryOp::Add, r, ten);
    b.ret(Some(r10));
    let mut m = b.finish();
    for (dest, init, next) in [(i, zero, i2), (acc, zero, acc2)] {
        phi(
            &mut m,
            f,
            header,
            IrPhiNode {
                dest,
                incoming: vec![(entry, init), (body, next)],
                ty: IrType::I64,
            },
        );
    }

    RESUME_CALLS.store(0, std::sync::atomic::Ordering::SeqCst);
    let mut interp = MirInterpreter::new();
    interp.register_symbol("count_resume_call", count_resume_call as *const u8);
    interp.set_max_iterations(50);
    let r = interp.execute(&m, outer, vec![]);
    assert!(
        matches!(r, Err(InterpError::JitBailout(id)) if id == f),
        "{:?}",
        r
    );
    assert_eq!(interp.suspended_functions(), vec![outer, f]);
    interp.set_max_iterations(u64::MAX);
    let r = interp.resume().unwrap();
    assert!(matches!(r, InterpValue::I64(499_510)), "{:?}", r);
    assert_eq!(RESUME_CALLS.load(std::sync::atomic::Ordering::SeqCst), 1);
}
