package rayzor.gpu;

/**
 * Single-pass renderer — convenience wrapper for common render operations.
 *
 * Handles the full lifecycle: create encoder → begin render pass →
 * set pipeline → bind resources → draw → end → submit.
 *
 * For more control, use CommandEncoder for multi-pass recording.
 *
 * Example:
 * ```haxe
 * import rayzor.gpu.*;
 *
 * // Create a render target texture
 * var target = Texture.create(device, 800, 600, 0,
 *     TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC);
 * var view = target.getView();
 *
 * // Render a triangle
 * Renderer.submit(device, view,
 *     0,                          // LoadOp.Clear
 *     0.1, 0.2, 0.3, 1.0,        // clear color (dark blue)
 *     null,                       // no depth
 *     pipeline,
 *     vertexBuffer,
 *     3, 1,                       // 3 vertices, 1 instance
 *     null, 0, 0,                 // no index buffer
 *     0, null                     // no bind groups
 * );
 * ```
 */
@:native("rayzor::gpu::Renderer")
extern class Renderer {
    /**
     * Submit a single render pass.
     *
     * @param device GPU device
     * @param colorView Target texture view to render into
     * @param loadOp 0=Clear, 1=Load
     * @param clearR Clear color red   [0.0–1.0]
     * @param clearG Clear color green [0.0–1.0]
     * @param clearB Clear color blue  [0.0–1.0]
     * @param clearA Clear color alpha [0.0–1.0]
     * @param depthView Optional depth texture view (null for no depth)
     * @param pipeline Render pipeline to use
     * @param vertexBuffer Vertex buffer (null if using vertex pulling)
     * @param vertexCount Number of vertices to draw
     * @param instanceCount Number of instances
     * @param indexBuffer Optional index buffer (null for non-indexed)
     * @param indexCount Number of indices (0 for non-indexed)
     * @param indexFormat 0=Uint16, 1=Uint32
     * @param bindGroupCount Number of bind groups
     * @param bindGroups Pointer to bind group array (null if 0)
     */
    @:native("rayzor_gpu_gfx_render_submit")
    public static function submit(
        device:GPUDevice,
        colorView:Dynamic,
        loadOp:Int,
        clearR:Float, clearG:Float, clearB:Float, clearA:Float,
        depthView:Dynamic,
        pipeline:RenderPipeline,
        vertexBuffer:Dynamic,
        vertexCount:Int, instanceCount:Int,
        indexBuffer:Dynamic,
        indexCount:Int, indexFormat:Int,
        bindGroupCount:Int, bindGroups:Dynamic
    ):Void;
}
