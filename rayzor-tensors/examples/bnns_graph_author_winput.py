import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

M, K, N = 497, 2048, 8192

@mb.program(input_specs=[
    mb.TensorSpec(shape=(M, K), dtype=types.fp16),
    mb.TensorSpec(shape=(K, N), dtype=types.fp16),
])
def prog(x, w):
    return mb.matmul(x=x, y=w, name="out")

m = ct.convert(
    prog,
    convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT16,
    compute_units=ct.ComputeUnit.CPU_ONLY,
    minimum_deployment_target=ct.target.macOS15,
    inputs=[ct.TensorType(name="x", dtype=np.float16),
            ct.TensorType(name="w", dtype=np.float16)],
    outputs=[ct.TensorType(name="out", dtype=np.float16)],
)
m.save("gemm_wi.mlpackage")
spec = m.get_spec().description
for i in spec.input: print("in ", i.name, i.type.multiArrayType.dataType)
for o in spec.output: print("out", o.name, o.type.multiArrayType.dataType)
