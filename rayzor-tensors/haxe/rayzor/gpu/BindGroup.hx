package rayzor.gpu;

/**
 * Bind group — binds resources (buffers, textures, samplers) to shader bindings.
 */
@:native("rayzor::gpu::BindGroup")
extern class BindGroup {
    /** Create a bind group with a single uniform buffer at binding 0. */
    @:native("rayzor_gpu_gfx_bind_group_single")
    public static function forBuffer(device:GPUDevice, layout:BindGroupLayout, buffer:GfxBuffer, bufferSize:Int):BindGroup;

    /** Destroy this bind group. */
    @:native("rayzor_gpu_gfx_bind_group_destroy")
    public function destroy():Void;
}

/**
 * Bind group layout — declares expected resource bindings.
 */
@:native("rayzor::gpu::BindGroupLayout")
extern class BindGroupLayout {
    /** Create a layout for N uniform buffer bindings. */
    @:native("rayzor_gpu_gfx_bind_group_layout_uniform")
    public static function forUniforms(device:GPUDevice, bindingCount:Int):BindGroupLayout;

    /** Destroy this layout. */
    @:native("rayzor_gpu_gfx_bind_group_layout_destroy")
    public function destroy():Void;
}
