package rayzor.gpu;

/**
 * GPU render pipeline — pre-compiled rendering configuration.
 *
 * Built via the builder pattern. Combines shader, vertex layout,
 * color format, topology, and optional depth/cull settings.
 *
 * Example:
 * ```haxe
 * var pipeline = RenderPipeline.begin()
 *     .setShader(shader)
 *     .setFormat(0)          // BGRA8Unorm
 *     .setTopology(0)        // TriangleList
 *     .build(device);
 * ```
 */
@:native("rayzor::gpu::RenderPipeline")
extern class RenderPipeline {
    /** Begin building a new render pipeline. */
    @:native("rayzor_gpu_gfx_pipeline_begin")
    public static function begin():RenderPipeline;

    /** Set the shader module for vertex + fragment stages. */
    @:native("rayzor_gpu_gfx_pipeline_set_shader")
    public function setShader(shader:ShaderModule):Void;

    /** Set the color target format (TextureFormat enum index). */
    @:native("rayzor_gpu_gfx_pipeline_set_format")
    public function setFormat(format:Int):Void;

    /** Set the primitive topology (PrimitiveTopology enum index). */
    @:native("rayzor_gpu_gfx_pipeline_set_topology")
    public function setTopology(topology:Int):Void;

    /** Set the face culling mode (CullMode enum index). */
    @:native("rayzor_gpu_gfx_pipeline_set_cull")
    public function setCull(mode:Int):Void;

    /** Finalize and create the render pipeline. */
    @:native("rayzor_gpu_gfx_pipeline_build")
    public function build(device:GPUDevice):RenderPipeline;

    /** Destroy this render pipeline. */
    @:native("rayzor_gpu_gfx_pipeline_destroy")
    public function destroy():Void;
}
