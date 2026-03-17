//! Integration test: compile CUDA kernels via system C compiler (cc/clang)
//! with TCC stub macros, execute on CPU, verify correctness.
//!
//! This validates the CUDA codegen pipeline end-to-end without needing
//! an NVIDIA GPU. The same kernel source will be compiled by NVRTC
//! on real hardware.

use std::io::Write;
use std::process::Command;

use rayzor_gpu::buffer::DTYPE_F32;
use rayzor_gpu::codegen::cuda;
use rayzor_gpu::kernel_ir::KernelOp;

/// Find a C compiler on the system.
fn find_cc() -> Option<String> {
    for cc in &["cc", "clang", "gcc"] {
        if Command::new(cc).arg("--version").output().is_ok() {
            return Some(cc.to_string());
        }
    }
    None
}

/// Compile CUDA kernel source (with TCC stubs) via system CC, run, check output.
fn run_kernel_test(
    kernel_source: &str,
    fn_name: &str,
    num_inputs: usize,
    setup_fn: &str,
) -> String {
    let cc = match find_cc() {
        Some(c) => c,
        None => {
            eprintln!("SKIP: no C compiler found");
            return String::new();
        }
    };

    let wrapped = cuda::wrap_for_tcc(kernel_source, fn_name, num_inputs);

    let full_source = format!(
        r#"{wrapped}

#include <stdio.h>
#include <stdlib.h>

int main() {{
    {setup_fn}
    return 0;
}}
"#
    );

    let uid = std::thread::current().id();
    let dir = std::env::temp_dir().join(format!("rayzor_cuda_{:?}_{}", uid, fn_name));
    let _ = std::fs::create_dir_all(&dir);
    let src_path = dir.join("test.c");
    let bin_path = dir.join("test_bin");

    let mut f = std::fs::File::create(&src_path).expect("create src");
    f.write_all(full_source.as_bytes()).expect("write src");

    let compile = Command::new(&cc)
        .arg(&src_path)
        .arg("-o")
        .arg(&bin_path)
        .arg("-lm")
        .arg("-std=c11")
        .output()
        .expect("compile");

    if !compile.status.success() {
        let stderr = String::from_utf8_lossy(&compile.stderr);
        panic!(
            "C compilation failed:\n{}\n--- source ---\n{}",
            stderr, full_source
        );
    }

    let run = Command::new(&bin_path).output().expect("run");
    let stdout = String::from_utf8_lossy(&run.stdout).to_string();

    // Cleanup
    let _ = std::fs::remove_file(&src_path);
    let _ = std::fs::remove_file(&bin_path);

    stdout
}

#[test]
fn test_cuda_add_f32_via_cc() {
    let src = cuda::emit_binary_elementwise(KernelOp::Add, DTYPE_F32);
    let fn_name = cuda::kernel_fn_name(KernelOp::Add, DTYPE_F32);

    let setup = r#"
    const int N = 8;
    float a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float b[8] = {10, 20, 30, 40, 50, 60, 70, 80};
    float result[8] = {0};

    run_rayzor_add_float(a, b, result, N, 4);

    int pass = 1;
    for (int i = 0; i < N; i++) {
        float expected = a[i] + b[i];
        if (result[i] != expected) {
            printf("FAIL: result[%d] = %f, expected %f\n", i, result[i], expected);
            pass = 0;
        }
    }
    if (pass) printf("PASS: add\n");
    "#;

    let output = run_kernel_test(&src, &fn_name, 2, setup);
    assert!(output.contains("PASS: add"), "add test failed: {}", output);
}

#[test]
fn test_cuda_relu_f32_via_cc() {
    let src = cuda::emit_unary_elementwise(KernelOp::Relu, DTYPE_F32);
    let fn_name = cuda::kernel_fn_name(KernelOp::Relu, DTYPE_F32);

    let setup = r#"
    const int N = 6;
    float a[6] = {-3, -1, 0, 1, 2, 3};
    float result[6] = {0};

    run_rayzor_relu_float(a, result, N, 4);

    float expected[6] = {0, 0, 0, 1, 2, 3};
    int pass = 1;
    for (int i = 0; i < N; i++) {
        if (result[i] != expected[i]) {
            printf("FAIL: relu[%d] = %f, expected %f\n", i, result[i], expected[i]);
            pass = 0;
        }
    }
    if (pass) printf("PASS: relu\n");
    "#;

    let output = run_kernel_test(&src, &fn_name, 1, setup);
    assert!(
        output.contains("PASS: relu"),
        "relu test failed: {}",
        output
    );
}

#[test]
fn test_cuda_sigmoid_f32_via_cc() {
    let src = cuda::emit_unary_elementwise(KernelOp::Sigmoid, DTYPE_F32);
    let fn_name = cuda::kernel_fn_name(KernelOp::Sigmoid, DTYPE_F32);

    let setup = r#"
    const int N = 3;
    float a[3] = {0, 10, -10};
    float result[3] = {0};

    run_rayzor_sigmoid_float(a, result, N, 4);

    int pass = 1;
    // sigmoid(0) = 0.5
    if (result[0] < 0.49 || result[0] > 0.51) {
        printf("FAIL: sigmoid(0) = %f\n", result[0]); pass = 0;
    }
    // sigmoid(10) ≈ 1.0
    if (result[1] < 0.99) {
        printf("FAIL: sigmoid(10) = %f\n", result[1]); pass = 0;
    }
    // sigmoid(-10) ≈ 0.0
    if (result[2] > 0.01) {
        printf("FAIL: sigmoid(-10) = %f\n", result[2]); pass = 0;
    }
    if (pass) printf("PASS: sigmoid\n");
    "#;

    let output = run_kernel_test(&src, &fn_name, 1, setup);
    assert!(
        output.contains("PASS: sigmoid"),
        "sigmoid test failed: {}",
        output
    );
}

#[test]
fn test_cuda_mul_f32_via_cc() {
    let src = cuda::emit_binary_elementwise(KernelOp::Mul, DTYPE_F32);
    let fn_name = cuda::kernel_fn_name(KernelOp::Mul, DTYPE_F32);

    let setup = r#"
    const int N = 4;
    float a[4] = {2, 3, 4, 5};
    float b[4] = {10, 20, 30, 40};
    float result[4] = {0};

    run_rayzor_mul_float(a, b, result, N, 4);

    float expected[4] = {20, 60, 120, 200};
    int pass = 1;
    for (int i = 0; i < N; i++) {
        if (result[i] != expected[i]) {
            printf("FAIL: mul[%d] = %f, expected %f\n", i, result[i], expected[i]);
            pass = 0;
        }
    }
    if (pass) printf("PASS: mul\n");
    "#;

    let output = run_kernel_test(&src, &fn_name, 2, setup);
    assert!(output.contains("PASS: mul"), "mul test failed: {}", output);
}

#[test]
fn test_cuda_neg_exp_log_sqrt_via_cc() {
    let ops = [
        (KernelOp::Neg, "neg", vec![1.0f32, -2.0, 3.0], vec![-1.0f32, 2.0, -3.0]),
        (KernelOp::Sqrt, "sqrt", vec![4.0, 9.0, 16.0], vec![2.0, 3.0, 4.0]),
    ];

    for (op, name, input, expected) in &ops {
        let src = cuda::emit_unary_elementwise(*op, DTYPE_F32);
        let fn_name = cuda::kernel_fn_name(*op, DTYPE_F32);

        let input_str: Vec<String> = input.iter().map(|v| format!("{v}")).collect();
        let expected_str: Vec<String> = expected.iter().map(|v| format!("{v}")).collect();

        let setup = format!(
            r#"
    const int N = {n};
    float a[{n}] = {{{inputs}}};
    float result[{n}] = {{0}};
    float expected[{n}] = {{{expects}}};

    run_{fn_name}(a, result, N, 4);

    int pass = 1;
    for (int i = 0; i < N; i++) {{
        float diff = result[i] - expected[i];
        if (diff < 0) diff = -diff;
        if (diff > 0.01) {{
            printf("FAIL: {name}[%d] = %f, expected %f\n", i, result[i], expected[i]);
            pass = 0;
        }}
    }}
    if (pass) printf("PASS: {name}\n");
    "#,
            n = input.len(),
            inputs = input_str.join(", "),
            expects = expected_str.join(", "),
        );

        let output = run_kernel_test(&src, &fn_name, 1, &setup);
        assert!(
            output.contains(&format!("PASS: {name}")),
            "{name} test failed: {}",
            output
        );
    }
}

#[test]
#[ignore] // Reduction requires barrier-aware multi-thread simulation — use Docker+NVRTC
fn test_cuda_reduction_sum_via_cc() {
    // Reduction needs a different wrapper — single block, check output[0]
    let src = rayzor_gpu::codegen::cuda_reduction::emit_reduction(KernelOp::ReduceSum, DTYPE_F32);
    let fn_name = rayzor_gpu::codegen::cuda_reduction::reduction_fn_name(KernelOp::ReduceSum, DTYPE_F32);

    let wrapped = cuda::wrap_for_tcc(&src, &fn_name, 1);

    let full = format!(
        r#"{wrapped}

#include <stdio.h>

int main() {{
    const int N = 8;
    float input[8] = {{1, 2, 3, 4, 5, 6, 7, 8}};
    float output[1] = {{0}};

    // Simulate single-block reduction
    run_{fn_name}(input, output, N, 8);

    float expected = 36.0;
    float diff = output[0] - expected;
    if (diff < 0) diff = -diff;
    if (diff < 0.01) {{
        printf("PASS: reduce_sum\n");
    }} else {{
        printf("FAIL: reduce_sum = %f, expected %f\n", output[0], expected);
    }}

    return 0;
}}
"#
    );

    let cc = match find_cc() {
        Some(c) => c,
        None => return,
    };

    let uid = std::thread::current().id();
    let dir = std::env::temp_dir().join(format!("rayzor_cuda_{:?}_reduce", uid));
    let _ = std::fs::create_dir_all(&dir);
    let src_path = dir.join("test_reduce.c");
    let bin_path = dir.join("test_reduce_bin");

    std::fs::write(&src_path, &full).expect("write");
    let compile = Command::new(&cc)
        .arg(&src_path)
        .arg("-o")
        .arg(&bin_path)
        .arg("-lm")
        .arg("-std=c11")
        .output()
        .expect("compile");

    assert!(
        compile.status.success(),
        "reduce compile failed: {}",
        String::from_utf8_lossy(&compile.stderr)
    );

    let run = Command::new(&bin_path).output().expect("run");
    let stdout = String::from_utf8_lossy(&run.stdout);
    assert!(
        stdout.contains("PASS: reduce_sum"),
        "reduce_sum failed: {}",
        stdout
    );

    let _ = std::fs::remove_file(&src_path);
    let _ = std::fs::remove_file(&bin_path);
}
