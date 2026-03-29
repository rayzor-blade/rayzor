package rayzor.gpu;

/**
 * GPU graphics device — wraps a wgpu Device + Queue.
 *
 * On native: backed by Metal/Vulkan/DX12 via wgpu.
 * On WASM: backed by browser WebGPU API (with WebGL2 fallback).
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
#if wasm
@:jsImport("rayzor-gpu")
#else
@:native("rayzor::gpu::GPUDevice")
#end
extern class GPUDevice {
    #if wasm
    @:jsMethod("create-device")
    public static function create():GPUDevice;
    @:jsMethod("destroy-device")
    public function destroy():Void;
    @:jsMethod("is-available")
    public static function isAvailable():Bool;
    #else
    @:native("rayzor_gpu_gfx_device_create")
    public static function create():GPUDevice;
    @:native("rayzor_gpu_gfx_device_destroy")
    public function destroy():Void;
    @:native("rayzor_gpu_gfx_is_available")
    public static function isAvailable():Bool;
    #end
}
