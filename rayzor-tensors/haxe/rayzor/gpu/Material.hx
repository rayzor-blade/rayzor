package rayzor.gpu;

/**
 * Material — holds shader + pipeline configuration.
 *
 * Resources are auto-released when the Material goes out of scope
 * via @:derive([Drop]). No manual destroy() needed.
 */
@:derive([Drop])
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

    /** Called automatically when this Material is dropped. */
    public function drop():Void {
        if (pipeline != null) pipeline.destroy();
    }
}

typedef MaterialOptions = {
    colorFormat:Int,
    topology:Int,
    ?cullMode:Int,
};
