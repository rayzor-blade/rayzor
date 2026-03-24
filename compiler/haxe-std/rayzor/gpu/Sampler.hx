package rayzor.gpu;

/**
 * Texture sampler — defines filtering and address modes.
 */
@:native("rayzor::gpu::Sampler")
extern class Sampler {
    /** Create a sampler with the given filter and address modes. */
    @:native("rayzor_gpu_gfx_sampler_create")
    public static function create(device:GPUDevice, magFilter:Int, minFilter:Int, addressMode:Int):Sampler;

    /** Destroy this sampler. */
    @:native("rayzor_gpu_gfx_sampler_destroy")
    public function destroy():Void;
}
