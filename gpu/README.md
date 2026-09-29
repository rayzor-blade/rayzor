# Rayzor GPU

`rayzor-gpu` packages the shared
[xgpu](https://github.com/rayzor-blade/xgpu) API for Rayzor applications. Its
generated Haxe classes live under `rayzor.gpu` and ship in
`rayzor-gpu.rpkg`; application developers do not need to run the Rust binding
generator.

Use the official [WebGPU specification](https://www.w3.org/TR/webgpu/) as the
API reference for the portable `rayzor.gpu` surface. The package also exposes
xgpu's native extensions where the selected backend supports them.

Rayzor adds compiler-owned `@:shader` lowering, lazy tensor graphs, fused and
quantized kernels, and direct Metal and CUDA/NVRTC compute paths. These are
Rayzor extensions layered beside the portable WebGPU API.

Install a packaged release with:

```sh
rayzor rpkg install rayzor-gpu.rpkg
```

The package includes its generated Haxe externs and the native libraries for
its release targets.
