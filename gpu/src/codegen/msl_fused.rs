//! Fused kernel MSL code generation.
//!
//! Translates a LazyOp expression tree into a single MSL kernel that performs
//! all operations in one dispatch. Input buffers are bound to consecutive
//! `[[buffer(N)]]` slots, and the result is written to the last buffer slot.

use std::collections::HashMap;
use std::rc::Rc;

use crate::kernel_ir::KernelOp;
use crate::lazy::LazyOp;

use super::msl::dtype_to_msl;

/// Result of fused kernel emission.
pub struct FusedKernelSource {
    /// MSL source code.
    pub source: String,
    /// Kernel function name.
    pub fn_name: String,
    /// Number of input buffer bindings (result is at index `num_inputs`).
    pub num_inputs: usize,
}

/// Generate MSL source for a fused elementwise kernel.
///
/// `ptr_to_idx` maps `Rc::as_ptr()` → buffer binding index.
pub fn emit_fused_kernel(
    op: &LazyOp,
    dtype: u8,
    ptr_to_idx: &HashMap<usize, usize>,
    num_inputs: usize,
) -> FusedKernelSource {
    let msl_type = dtype_to_msl(dtype);
    let mut counter: usize = 0;
    let mut body_lines: Vec<String> = Vec::new();

    let result_var = emit_op(op, msl_type, ptr_to_idx, &mut counter, &mut body_lines);

    // Build parameter list
    let mut params: Vec<String> = Vec::new();
    for i in 0..num_inputs {
        params.push(format!(
            "    device const {msl_type}* in{i} [[buffer({i})]],",
        ));
    }
    params.push(format!(
        "    device {msl_type}* result [[buffer({num_inputs})]],",
    ));
    params.push("    uint id [[thread_position_in_grid]]".to_string());

    let fn_name = format!("fused_{num_inputs}in_{counter}ops");

    let source = format!(
        "#include <metal_stdlib>\nusing namespace metal;\n\nkernel void {fn_name}(\n{params}\n) {{\n{body}\n    result[id] = {result_var};\n}}\n",
        params = params.join("\n"),
        body = body_lines.join("\n"),
    );

    FusedKernelSource {
        source,
        fn_name,
        num_inputs,
    }
}

/// Recursively emit MSL for a LazyOp node, returning the variable name
/// holding the result of this subtree.
fn emit_op(
    op: &LazyOp,
    msl_type: &str,
    ptr_to_idx: &HashMap<usize, usize>,
    counter: &mut usize,
    lines: &mut Vec<String>,
) -> String {
    match op {
        LazyOp::Input(buf) => {
            let ptr = Rc::as_ptr(buf) as usize;
            let idx = ptr_to_idx[&ptr];
            format!("in{idx}[id]")
        }
        LazyOp::Unary {
            op: kernel_op,
            input,
        } => {
            let input_expr = emit_op(input, msl_type, ptr_to_idx, counter, lines);
            let var = format!("v{counter}");
            *counter += 1;

            let expr = match kernel_op {
                KernelOp::Neg => format!("-{input_expr}"),
                KernelOp::Abs => format!("abs({input_expr})"),
                KernelOp::Sqrt => format!("sqrt({input_expr})"),
                KernelOp::Exp => format!("exp({input_expr})"),
                KernelOp::Log => format!("log({input_expr})"),
                KernelOp::Relu => format!("max(({msl_type})0, {input_expr})"),
                KernelOp::Sigmoid => format!("1.0 / (1.0 + exp(-{input_expr}))"),
                KernelOp::Tanh => format!("tanh({input_expr})"),
                KernelOp::Gelu => format!("{input_expr} * 0.5 * (1.0 + tanh(0.7978845608 * ({input_expr} + 0.044715 * {input_expr} * {input_expr} * {input_expr})))"),
                KernelOp::Silu => format!("{input_expr} / (1.0 + exp(-{input_expr}))"),
                _ => unreachable!("not a unary op: {:?}", kernel_op),
            };

            lines.push(format!("    {msl_type} {var} = {expr};"));
            var
        }
        LazyOp::Binary {
            op: kernel_op,
            lhs,
            rhs,
        } => {
            let lhs_expr = emit_op(lhs, msl_type, ptr_to_idx, counter, lines);
            let rhs_expr = emit_op(rhs, msl_type, ptr_to_idx, counter, lines);
            let var = format!("v{counter}");
            *counter += 1;

            let op_str = match kernel_op {
                KernelOp::Add => "+",
                KernelOp::Sub => "-",
                KernelOp::Mul => "*",
                KernelOp::Div => "/",
                _ => unreachable!("not a binary op: {:?}", kernel_op),
            };

            lines.push(format!(
                "    {msl_type} {var} = {lhs_expr} {op_str} {rhs_expr};"
            ));
            var
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::NativeBuffer;
    use crate::buffer;
    use crate::lazy;
    use std::rc::Rc;

    #[cfg(feature = "metal-backend")]
    use crate::metal::{buffer_ops::MetalBuffer, device_init::MetalContext};

    #[test]
    #[cfg(feature = "metal-backend")]
    fn test_fused_add_relu() {
        if !MetalContext::is_available() {
            return;
        }
        let ctx = MetalContext::new().unwrap();

        let data = [1.0f32; 4];
        let buf_a = MetalBuffer::from_data(&ctx, data.as_ptr() as *const u8, 16).unwrap();
        let buf_b = MetalBuffer::from_data(&ctx, data.as_ptr() as *const u8, 16).unwrap();

        let nb_a = Rc::new(NativeBuffer::Metal(buf_a));
        let nb_b = Rc::new(NativeBuffer::Metal(buf_b));

        let op = LazyOp::Unary {
            op: KernelOp::Relu,
            input: Rc::new(LazyOp::Binary {
                op: KernelOp::Add,
                lhs: Rc::new(LazyOp::Input(nb_a)),
                rhs: Rc::new(LazyOp::Input(nb_b)),
            }),
        };

        let (inputs, ptr_to_idx) = lazy::collect_inputs(&op);
        assert_eq!(inputs.len(), 2);

        let result = emit_fused_kernel(&op, buffer::DTYPE_F32, &ptr_to_idx, inputs.len());
        assert!(result.source.contains("kernel void fused_"));
        assert!(result.source.contains("device const float* in0"));
        assert!(result.source.contains("device const float* in1"));
        assert!(result.source.contains("device float* result"));
        assert!(result.source.contains("+"));
        assert!(result.source.contains("max("));
        assert_eq!(result.num_inputs, 2);
    }

    #[test]
    #[cfg(feature = "metal-backend")]
    fn test_fused_shared_input() {
        if !MetalContext::is_available() {
            return;
        }
        let ctx = MetalContext::new().unwrap();

        let data = [1.0f32; 4];
        let buf_a = MetalBuffer::from_data(&ctx, data.as_ptr() as *const u8, 16).unwrap();
        let nb_a = Rc::new(NativeBuffer::Metal(buf_a));

        let input_a = Rc::new(LazyOp::Input(nb_a));
        let op = LazyOp::Binary {
            op: KernelOp::Add,
            lhs: input_a.clone(),
            rhs: input_a,
        };

        let (inputs, ptr_to_idx) = lazy::collect_inputs(&op);
        assert_eq!(inputs.len(), 1);

        let result = emit_fused_kernel(&op, buffer::DTYPE_F32, &ptr_to_idx, inputs.len());
        assert!(result.source.contains("device const float* in0"));
        assert!(!result.source.contains("in1"));
    }
}
