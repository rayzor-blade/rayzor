package rayzor.gpu;

/**
 * GPU graphics device — wraps a wgpu Device + Queue.
 *
 * On native: backed by Metal/Vulkan/DX12 via wgpu.
 * On WASM: backed by browser WebGPU API via wasm-bindgen host module.
 *
 * Example:
 * ```haxe
 * var device = GPUDevice.create();
 * if (device != null) {
 *     var shader = ShaderModule.create(device, wgslSource, "vs_main", "fs_main");
 *     // ... create pipeline, render ...
 *     device.destroy();
 * }
 * ```
 */
@:native("rayzor::gpu::GPUDevice")
extern class GPUDevice {
    @:native("rayzor_gpu_gfx_device_create")
    public static function create():GPUDevice;
    @:native("rayzor_gpu_gfx_device_destroy")
    public function destroy():Void;
    @:native("rayzor_gpu_gfx_is_available")
    public static function isAvailable():Bool;
}
