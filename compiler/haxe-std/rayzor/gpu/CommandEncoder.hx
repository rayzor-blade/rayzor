package rayzor.gpu;

/**
 * Command encoder for recording GPU render commands.
 *
 * Records multiple render passes, then submits all at once.
 *
 * Example:
 * ```haxe
 * var cmd = CommandEncoder.create();
 *
 * // Pass 1: Clear to dark blue
 * var pass = cmd.beginRenderPass(view, LoadOp.Clear, {r:0.1, g:0.2, b:0.3, a:1.0});
 * pass.setPipeline(pipeline);
 * pass.setVertexBuffer(0, vertexBuf);
 * pass.draw(3);
 * pass.end();
 *
 * // Pass 2: Overlay
 * var pass2 = cmd.beginRenderPass(view, LoadOp.Load, {r:0, g:0, b:0, a:0});
 * pass2.setPipeline(overlayPipeline);
 * pass2.draw(6);
 * pass2.end();
 *
 * cmd.submit(device);
 * ```
 */
@:native("rayzor::gpu::CommandEncoder")
extern class CommandEncoder {
    /** Create a new command encoder. */
    @:native("rayzor_gpu_gfx_cmd_create")
    public static function create():CommandEncoder;

    /** Begin a new render pass targeting the given color view. */
    @:native("rayzor_gpu_gfx_cmd_begin_pass")
    public function beginPass(
        colorView:Dynamic,
        loadOp:Int,
        clearR:Float, clearG:Float, clearB:Float, clearA:Float,
        depthView:Dynamic
    ):Void;

    /** Set the active render pipeline for the current pass. */
    @:native("rayzor_gpu_gfx_cmd_set_pipeline")
    public function setPipeline(pipeline:RenderPipeline):Void;

    /** Bind a vertex buffer to a slot in the current pass. */
    @:native("rayzor_gpu_gfx_cmd_set_vertex_buffer")
    public function setVertexBuffer(slot:Int, buffer:GfxBuffer):Void;

    /** Bind an index buffer in the current pass. Format: 0=Uint16, 1=Uint32. */
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

    /** End the current render pass. */
    @:native("rayzor_gpu_gfx_cmd_end_pass")
    public function endPass():Void;

    /** Submit all recorded passes to the GPU. Consumes this encoder. */
    @:native("rayzor_gpu_gfx_cmd_submit")
    public function submit(device:GPUDevice):Void;
}
