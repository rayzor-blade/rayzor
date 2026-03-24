package rayzor.gpu;

/**
 * Material — holds shader + pipeline configuration.
 *
 * A material defines how geometry is rendered: which shader,
 * what color format, topology, culling, etc.
 *
 * Example:
 * ```haxe
 * var mat = new Material(device, myShader, {
 *     colorFormat: 0,  // BGRA8Unorm
 *     topology: 0,     // TriangleList
 *     cullMode: 2,     // Back
 * });
 * ```
 */
class Material {
    public var pipeline:RenderPipeline;
    public var shader:ShaderModule;

    public function new(device:GPUDevice, shader:ShaderModule, opts:MaterialOptions) {
        this.shader = shader;
        var builder = RenderPipeline.begin();
        builder.setShader(shader);
        builder.setFormat(opts.colorFormat);
        builder.setTopology(opts.topology);
        if (opts.cullMode != 0) {
            builder.setCull(opts.cullMode);
        }
        pipeline = builder.build(device);
    }

    public function destroy():Void {
        if (pipeline != null) pipeline.destroy();
    }
}

typedef MaterialOptions = {
    /** TextureFormat enum index for color target. */
    colorFormat:Int,
    /** PrimitiveTopology enum index. */
    topology:Int,
    /** CullMode enum index (0=None, 1=Front, 2=Back). */
    ?cullMode:Int,
};
