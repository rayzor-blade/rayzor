package rayzor.gpu;

/**
 * GPU renderer — submits draw calls to the GPU.
 *
 * For more control, use CommandEncoder for multi-pass recording.
 */
@:native("rayzor::gpu::Renderer")
extern class Renderer {
    /**
     * Render triangles with a clear color. No vertex buffer needed —
     * vertex positions come from @builtin(vertex_index) in the shader.
     *
     * @param device GPU device
     * @param colorView Target texture view
     * @param pipeline Render pipeline
     * @param vertexCount Number of vertices (e.g., 3 for a triangle)
     * @param clearR Clear color red [0-1]
     * @param clearG Clear color green [0-1]
     * @param clearB Clear color blue [0-1]
     * @param clearA Clear color alpha [0-1]
     */
    @:native("rayzor_gpu_gfx_render_triangles")
    public static function renderTriangles(
        device:GPUDevice,
        colorView:Dynamic,
        pipeline:RenderPipeline,
        vertexCount:Int,
        clearR:Float, clearG:Float, clearB:Float, clearA:Float
    ):Void;
}
