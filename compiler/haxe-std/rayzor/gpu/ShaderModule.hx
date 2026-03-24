package rayzor.gpu;

/**
 * Compiled WGSL shader module.
 *
 * Compiles a WGSL source string into a GPU shader module with named
 * entry points for vertex and fragment stages.
 *
 * Example:
 * ```haxe
 * var wgsl = "
 *   @vertex fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
 *     var pos = array<vec2f, 3>(vec2f(0.0, 0.5), vec2f(-0.5, -0.5), vec2f(0.5, -0.5));
 *     return vec4f(pos[i], 0.0, 1.0);
 *   }
 *   @fragment fn fs_main() -> @location(0) vec4f {
 *     return vec4f(1.0, 0.5, 0.2, 1.0);
 *   }
 * ";
 * var shader = ShaderModule.create(device, wgsl, "vs_main", "fs_main");
 * ```
 */
@:native("rayzor::gpu::ShaderModule")
extern class ShaderModule {
    /** Compile WGSL source into a shader module. */
    @:native("rayzor_gpu_gfx_shader_create")
    public static function create(device:GPUDevice, wgslSource:String, vertexEntry:String, fragmentEntry:String):ShaderModule;

    /** Destroy this shader module. */
    @:native("rayzor_gpu_gfx_shader_destroy")
    public function destroy():Void;
}
