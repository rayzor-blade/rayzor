package rayzor.gpu;

/**
 * Scene — collection of renderable objects.
 *
 * Auto-releases owned meshes/materials via @:derive([Drop]).
 */
@:derive([Drop])
class Scene {
    public var objects:Array<SceneObject>;
    public var clearColor:Color;

    public function new() {
        objects = [];
        clearColor = {r: 0.1, g: 0.1, b: 0.1, a: 1.0};
    }

    public function add(mesh:Mesh, material:Material, bindGroups:Array<Dynamic>):Void {
        objects.push({
            mesh: mesh,
            material: material,
            bindGroups: bindGroups,
        });
    }

    public function clear():Void {
        objects = [];
    }

    public function render(device:GPUDevice, targetView:Dynamic, depthView:Dynamic):Void {
        var c = clearColor;
        Renderer.submit(
            device,
            targetView,
            0,
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

    /** Called automatically — drops all owned objects. */
    public function drop():Void {
        // Meshes and Materials have their own @:derive([Drop])
    }
}

typedef SceneObject = {
    mesh:Mesh,
    material:Material,
    bindGroups:Array<Dynamic>,
};
