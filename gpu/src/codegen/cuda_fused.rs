//! Fused kernel CUDA C code generation.
//!
//! Translates a LazyOp expression tree into a single CUDA kernel.

use std::collections::HashMap;
use std::rc::Rc;

use crate::kernel_ir::KernelOp;
use crate::lazy::LazyOp;

use super::cuda::dtype_to_cuda;

/// Result of fused kernel emission.
pub struct FusedKernelSource {
    pub source: String,
    pub fn_name: String,
    pub num_inputs: usize,
}

/// Generate CUDA source for a fused elementwise kernel.
pub fn emit_fused_kernel(
    op: &LazyOp,
    dtype: u8,
    ptr_to_idx: &HashMap<usize, usize>,
    num_inputs: usize,
) -> FusedKernelSource {
    let cuda_type = dtype_to_cuda(dtype);
    let mut counter: usize = 0;
    let mut body_lines: Vec<String> = Vec::new();

    let result_var = emit_op(op, cuda_type, ptr_to_idx, &mut counter, &mut body_lines);

    let mut params: Vec<String> = Vec::new();
    for i in 0..num_inputs {
        params.push(format!("    const {cuda_type}* in{i},"));
    }
    params.push(format!("    {cuda_type}* result,"));
    params.push("    unsigned int numel".to_string());

    let fn_name = format!("fused_{num_inputs}in_{counter}ops");

    let source = format!(
        r#"extern "C" __global__ void {fn_name}(
{params}
) {{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= numel) return;
{body}
    result[id] = {result_var};
}}
"#,
        params = params.join("\n"),
        body = body_lines.join("\n"),
    );

    FusedKernelSource {
        source,
        fn_name,
        num_inputs,
    }
}

fn emit_op(
    op: &LazyOp,
    cuda_type: &str,
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
            let input_expr = emit_op(input, cuda_type, ptr_to_idx, counter, lines);
            let var = format!("v{counter}");
            *counter += 1;

            let expr = match kernel_op {
                KernelOp::Neg => format!("-{input_expr}"),
                KernelOp::Abs => format!("({cuda_type})fabs((double){input_expr})"),
                KernelOp::Sqrt => format!("({cuda_type})sqrt((double){input_expr})"),
                KernelOp::Exp => format!("({cuda_type})exp((double){input_expr})"),
                KernelOp::Log => format!("({cuda_type})log((double){input_expr})"),
                KernelOp::Relu => format!("{input_expr} > ({cuda_type})0 ? {input_expr} : ({cuda_type})0"),
                KernelOp::Sigmoid => format!("({cuda_type})(1.0 / (1.0 + exp(-(double){input_expr})))"),
                KernelOp::Tanh => format!("({cuda_type})tanh((double){input_expr})"),
                KernelOp::Gelu => format!("({cuda_type})((double){input_expr} * 0.5 * (1.0 + tanh(0.7978845608 * ((double){input_expr} + 0.044715 * (double){input_expr} * (double){input_expr} * (double){input_expr}))))"),
                KernelOp::Silu => format!("({cuda_type})((double){input_expr} / (1.0 + exp(-(double){input_expr})))"),
                _ => unreachable!("not a unary op: {:?}", kernel_op),
            };

            lines.push(format!("    {cuda_type} {var} = {expr};"));
            var
        }
        LazyOp::Binary {
            op: kernel_op,
            lhs,
            rhs,
        } => {
            let lhs_expr = emit_op(lhs, cuda_type, ptr_to_idx, counter, lines);
            let rhs_expr = emit_op(rhs, cuda_type, ptr_to_idx, counter, lines);
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
                "    {cuda_type} {var} = {lhs_expr} {op_str} {rhs_expr};"
            ));
            var
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::NativeBuffer;
    use crate::buffer::DTYPE_F32;

    #[test]
    fn test_fused_add_relu() {
        let buf_a = Rc::new(NativeBuffer::Unavailable);
        let buf_b = Rc::new(NativeBuffer::Unavailable);

        let op = LazyOp::Unary {
            op: KernelOp::Relu,
            input: Rc::new(LazyOp::Binary {
                op: KernelOp::Add,
                lhs: Rc::new(LazyOp::Input(buf_a.clone())),
                rhs: Rc::new(LazyOp::Input(buf_b.clone())),
            }),
        };

        let mut ptr_to_idx = HashMap::new();
        ptr_to_idx.insert(Rc::as_ptr(&buf_a) as usize, 0);
        ptr_to_idx.insert(Rc::as_ptr(&buf_b) as usize, 1);

        let result = emit_fused_kernel(&op, DTYPE_F32, &ptr_to_idx, 2);
        assert!(result.source.contains("__global__"));
        assert!(result.source.contains("in0[id] + in1[id]"));
        assert!(result.source.contains("? v0 :"));
    }
}
