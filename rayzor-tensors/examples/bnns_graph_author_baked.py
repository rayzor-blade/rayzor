import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

M, K, N = 497, 2048, 8192
rng = np.random.default_rng(7)
W = ((rng.random((K, N), dtype=np.float32) - 0.5) * 0.1).astype(np.float16)

@mb.program(input_specs=[mb.TensorSpec(shape=(M, K), dtype=types.fp16)])
def prog(x):
    return mb.matmul(x=x, y=W, name="out")

m = ct.convert(
    prog,
    convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT16,
    compute_units=ct.ComputeUnit.CPU_ONLY,
    minimum_deployment_target=ct.target.macOS15,
)
m.save("gemm497.mlpackage")
print("saved gemm497.mlpackage")
# stash the exact weights for the Rust-side correctness check
np.save("gemm_w.npy", W)
