//! Kernel IR — describes GPU compute operations at a high level.
//!
//! Each `KernelOp` maps to a single GPU kernel. The codegen layer
//! translates these into backend-specific source (MSL, PTX, WGSL).

/// GPU compute operation types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelOp {
    // Binary elementwise: result[i] = a[i] OP b[i]
    Add,
    Sub,
    Mul,
    Div,

    // Unary elementwise: result[i] = OP(a[i])
    Neg,
    Abs,
    Sqrt,
    Exp,
    Log,
    Relu,
    /// Sigmoid: 1 / (1 + exp(-x))
    Sigmoid,
    /// Tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    Tanh,
    /// GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    Gelu,
    /// SiLU (Swish): x * sigmoid(x)
    Silu,

    // Reductions: input[0..numel] -> single value per threadgroup
    ReduceSum,
    ReduceMax,
    ReduceMin,

    // Linear algebra
    Matmul,
    /// Batched matmul: C[b,m,n] = A[b,m,k] × B[b,k,n] for b in 0..B
    BatchMatmul,
}

impl KernelOp {
    /// Number of input buffers this operation requires.
    pub fn input_count(self) -> usize {
        match self {
            Self::Add | Self::Sub | Self::Mul | Self::Div => 2,
            Self::Neg
            | Self::Abs
            | Self::Sqrt
            | Self::Exp
            | Self::Log
            | Self::Relu
            | Self::Sigmoid
            | Self::Tanh
            | Self::Gelu
            | Self::Silu => 1,
            Self::ReduceSum | Self::ReduceMax | Self::ReduceMin => 1,
            Self::Matmul | Self::BatchMatmul => 2,
        }
    }

    /// Human-readable name used in kernel function naming.
    pub fn name(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "mul",
            Self::Div => "div",
            Self::Neg => "neg",
            Self::Abs => "abs",
            Self::Sqrt => "sqrt",
            Self::Exp => "exp",
            Self::Log => "log",
            Self::Relu => "relu",
            Self::Sigmoid => "sigmoid",
            Self::Tanh => "tanh",
            Self::Gelu => "gelu",
            Self::Silu => "silu",
            Self::ReduceSum => "reduce_sum",
            Self::ReduceMax => "reduce_max",
            Self::ReduceMin => "reduce_min",
            Self::Matmul => "matmul",
            Self::BatchMatmul => "batch_matmul",
        }
    }

    /// Whether this op is a reduction (produces fewer outputs than inputs).
    pub fn is_reduction(self) -> bool {
        matches!(self, Self::ReduceSum | Self::ReduceMax | Self::ReduceMin)
    }
}
