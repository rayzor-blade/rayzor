package rayzor.gpu;

/**
 * Scene — collection of renderable objects.
 *
 * Each object has a mesh, material, and optional bind groups.
 * `scene.render()` iterates objects and submits draw calls.
 *
 * Example:
 * ```haxe
 * var scene = new Scene();
 * scene.add(mesh, material, null);
 * scene.render(device, targetView, camera);
 * ```
 */
class Scene {
    public var objects:Array<SceneObject>;
    public var clearColor:Color;

    public function new() {
        objects = [];
        clearColor = {r: 0.1, g: 0.1, b: 0.1, a: 1.0};
    }

    /** Add an object to the scene. */
    public function add(mesh:Mesh, material:Material, bindGroups:Array<Dynamic>):Void {
        objects.push({
            mesh: mesh,
            material: material,
            bindGroups: bindGroups,
        });
    }

    /** Remove all objects from the scene. */
    public function clear():Void {
        objects = [];
    }

    /**
     * Render the entire scene to a target texture view.
     *
     * Uses a single render pass: clears to clearColor, then draws
     * each object in order (no sorting — front-to-back ordering is
     * the caller's responsibility).
     */
    public function render(device:GPUDevice, targetView:Dynamic, depthView:Dynamic):Void {
        var c = clearColor;
        Renderer.submit(
            device,
            targetView,
            0,  // LoadOp.Clear
            c.r, c.g, c.b, c.a,
            depthView,
            objects.length > 0 ? objects[0].material.pipeline : null,
            objects.length > 0 ? objects[0].mesh.vertexBuffer : null,
            objects.length > 0 ? objects[0].mesh.vertexCount : 0,
            1,
            null, 0, 0,
            0, null
        );
    }
}

typedef SceneObject = {
    mesh:Mesh,
    material:Material,
    bindGroups:Array<Dynamic>,
};
