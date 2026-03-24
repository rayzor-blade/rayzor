package rayzor.gpu;

/**
 * GPU graphics device — wraps a wgpu Device + Queue.
 *
 * Create with `GPUDevice.create()`. Provides access to all graphics
 * resource creation (shaders, buffers, textures, pipelines).
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
    /** Create a GPU graphics device. Returns null if no GPU is available. */
    @:native("rayzor_gpu_gfx_device_create")
    public static function create():GPUDevice;

    /** Release the GPU device and all associated resources. */
    @:native("rayzor_gpu_gfx_device_destroy")
    public function destroy():Void;

    /** Check if GPU graphics is available on this system. */
    @:native("rayzor_gpu_gfx_is_available")
    public static function isAvailable():Bool;
}
