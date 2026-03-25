package rayzor.gpu;

/**
 * GPU render pipeline — compiled rendering configuration.
 *
 * Built via builder pattern:
 *   RenderPipeline.begin() → setShader → setFormat → setTopology → build
 */
@:native("rayzor::gpu::RenderPipeline")
extern class RenderPipeline {
    /** Begin building a new render pipeline. */
    @:native("rayzor_gpu_gfx_pipeline_begin")
    public static function begin():RenderPipeline;

    /** Set the shader module. */
    @:native("rayzor_gpu_gfx_pipeline_set_shader")
    public function setShader(shader:ShaderModule):Void;

    /** Set color format (0=BGRA8Unorm, 1=RGBA8Unorm, ...). */
    @:native("rayzor_gpu_gfx_pipeline_set_format")
    public function setFormat(format:Int):Void;

    /** Set primitive topology (0=TriangleList, 1=TriangleStrip, 2=LineList, ...). */
    @:native("rayzor_gpu_gfx_pipeline_set_topology")
    public function setTopology(topology:Int):Void;

    /** Set face culling (0=None, 1=Front, 2=Back). */
    @:native("rayzor_gpu_gfx_pipeline_set_cull")
    public function setCull(mode:Int):Void;

    /** Set vertex buffer layout. attrData is packed [format,offset,location,...]. */
    @:native("rayzor_gpu_gfx_pipeline_set_vertex_layout_simple")
    public function setVertexLayout(stride:Int, attrCount:Int, attrData:Dynamic):Void;

    /** Enable depth testing (format: 3=Depth32Float). */
    @:native("rayzor_gpu_gfx_pipeline_set_depth_simple")
    public function setDepth(depthFormat:Int):Void;

    /** Add a bind group layout for uniforms. */
    @:native("rayzor_gpu_gfx_pipeline_add_layout")
    public function addBindGroupLayout(layout:BindGroupLayout):Void;

    /** Build the pipeline. */
    @:native("rayzor_gpu_gfx_pipeline_build")
    public function build(device:GPUDevice):RenderPipeline;

    /** Add an additional color target for MRT (Multiple Render Targets).
     *  First target is set via setFormat(); this adds @location(1), @location(2), etc.
     */
    @:native("rayzor_gpu_gfx_pipeline_add_color_target")
    public function addColorTarget(format:Int):Void;

    /** Destroy this pipeline. */
    @:native("rayzor_gpu_gfx_pipeline_destroy")
    public function destroy():Void;
}
