package rayzor.gpu;

/**
 * GPU renderer — submits draw calls to the GPU.
 *
 * Provides multiple render methods for different use cases:
 * - renderTriangles: procedural geometry (no vertex buffer)
 * - render: with vertex buffer
 * - renderIndexed: with vertex + index buffers
 * - renderWithDepth: with depth testing
 * - renderWithBindings: with uniform buffer bind group
 */
@:native("rayzor::gpu::Renderer")
extern class Renderer {
    /** Render procedural triangles (positions from @builtin(vertex_index)). */
    @:native("rayzor_gpu_gfx_render_triangles")
    public static function renderTriangles(
        device:GPUDevice, colorView:Dynamic, pipeline:RenderPipeline,
        vertexCount:Int, clearR:Float, clearG:Float, clearB:Float, clearA:Float
    ):Void;

    /** Render with a vertex buffer. */
    @:native("rayzor_gpu_gfx_render_with_vb")
    public static function render(
        device:GPUDevice, colorView:Dynamic, pipeline:RenderPipeline,
        vertexBuffer:GfxBuffer, vertexCount:Int, instanceCount:Int,
        clearR:Float, clearG:Float, clearB:Float, clearA:Float
    ):Void;

    /** Render with vertex + index buffers. */
    @:native("rayzor_gpu_gfx_render_indexed")
    public static function renderIndexed(
        device:GPUDevice, colorView:Dynamic, pipeline:RenderPipeline,
        vertexBuffer:GfxBuffer, indexBuffer:GfxBuffer, indexCount:Int, instanceCount:Int,
        clearR:Float, clearG:Float, clearB:Float, clearA:Float
    ):Void;

    /** Render with depth buffer. */
    @:native("rayzor_gpu_gfx_render_with_depth")
    public static function renderWithDepth(
        device:GPUDevice, colorView:Dynamic, depthView:Dynamic, pipeline:RenderPipeline,
        vertexBuffer:GfxBuffer, vertexCount:Int,
        clearR:Float, clearG:Float, clearB:Float, clearA:Float
    ):Void;

    /** Render with a bind group (for uniforms). */
    @:native("rayzor_gpu_gfx_render_with_bindings")
    public static function renderWithBindings(
        device:GPUDevice, colorView:Dynamic, pipeline:RenderPipeline,
        vertexBuffer:GfxBuffer, vertexCount:Int, bindGroup:Dynamic,
        clearR:Float, clearG:Float, clearB:Float, clearA:Float
    ):Void;
}
