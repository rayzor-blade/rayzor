package rayzor.gpu;

/**
 * Command encoder for recording multi-pass GPU render commands.
 *
 * Records multiple render passes, then submits all at once.
 *
 * Example:
 * ```haxe
 * var cmd = CommandEncoder.create();
 *
 * // Pass 1: Clear + draw triangle
 * cmd.beginPass(view, 0, 0.1, 0.2, 0.3, 1.0, null);
 * cmd.setPipeline(pipeline);
 * cmd.setVertexBuffer(0, vertexBuf);
 * cmd.draw(3, 1, 0, 0);
 * cmd.endPass();
 *
 * // Pass 2: Overlay (load existing content)
 * cmd.beginPass(view, 1, 0, 0, 0, 0, null);
 * cmd.setPipeline(overlayPipeline);
 * cmd.draw(6, 1, 0, 0);
 * cmd.endPass();
 *
 * cmd.submit(device);
 * ```
 */
@:native("rayzor::gpu::CommandEncoder")
extern class CommandEncoder {
    /** Create a new command encoder. */
    @:native("rayzor_gpu_gfx_cmd_create")
    public static function create():CommandEncoder;

    /** Begin a render pass. loadOp: 0=Clear, 1=Load. */
    @:native("rayzor_gpu_gfx_cmd_begin_pass")
    public function beginPass(
        colorView:TextureView, loadOp:Int,
        clearR:Float, clearG:Float, clearB:Float, clearA:Float,
        depthView:TextureView
    ):Void;

    /** Set the active render pipeline. */
    @:native("rayzor_gpu_gfx_cmd_set_pipeline")
    public function setPipeline(pipeline:RenderPipeline):Void;

    /** Bind a vertex buffer to a slot. */
    @:native("rayzor_gpu_gfx_cmd_set_vertex_buffer")
    public function setVertexBuffer(slot:Int, buffer:GfxBuffer):Void;

    /** Bind an index buffer. format: 0=Uint16, 1=Uint32. */
    @:native("rayzor_gpu_gfx_cmd_set_index_buffer")
    public function setIndexBuffer(buffer:GfxBuffer, format:Int):Void;

    /** Bind a bind group at the given group index. */
    @:native("rayzor_gpu_gfx_cmd_set_bind_group")
    public function setBindGroup(groupIndex:Int, bindGroup:Dynamic):Void;

    /** Draw non-indexed geometry. */
    @:native("rayzor_gpu_gfx_cmd_draw")
    public function draw(vertexCount:Int, instanceCount:Int, firstVertex:Int, firstInstance:Int):Void;

    /** Draw indexed geometry. */
    @:native("rayzor_gpu_gfx_cmd_draw_indexed")
    public function drawIndexed(indexCount:Int, instanceCount:Int, firstIndex:Int, baseVertex:Int, firstInstance:Int):Void;

    /** Set the viewport rectangle. */
    @:native("rayzor_gpu_gfx_cmd_set_viewport")
    public function setViewport(x:Float, y:Float, w:Float, h:Float, minDepth:Float, maxDepth:Float):Void;

    /** Set the scissor rectangle. */
    @:native("rayzor_gpu_gfx_cmd_set_scissor")
    public function setScissor(x:Int, y:Int, w:Int, h:Int):Void;

    /** Begin a render pass with multiple color targets (MRT).
     *  clearColors is a packed array of RGBA f64: [r0,g0,b0,a0, r1,g1,b1,a1, ...].
     */
    @:native("rayzor_gpu_gfx_cmd_begin_pass_mrt")
    public function beginPassMRT(
        colorViewCount:Int, colorViews:Dynamic,
        loadOps:Dynamic, clearColors:Dynamic,
        depthView:TextureView
    ):Void;

    /** End the current render pass. */
    @:native("rayzor_gpu_gfx_cmd_end_pass")
    public function endPass():Void;

    /** Submit all recorded passes to the GPU. Consumes this encoder. */
    @:native("rayzor_gpu_gfx_cmd_submit")
    public function submit(device:GPUDevice):Void;
}
