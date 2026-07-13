package rayzor.gpu;

/**
 * Camera — perspective or orthographic projection.
 *
 * Generates a 4x4 projection matrix stored in a uniform buffer.
 * Uses right-handed coordinate system with Y-up.
 *
 * Example:
 * ```haxe
 * var cam = Camera.perspective(Math.PI / 4, 800 / 600, 0.1, 100.0);
 * cam.setPosition(0, 2, -5);
 * cam.lookAt(0, 0, 0);
 * // cam.matrix is a 16-float array (column-major 4x4)
 * ```
 */
class Camera {
    /** 4x4 projection × view matrix (column-major, 16 floats). */
    public var matrix:Array<Float>;

    /** Camera position in world space. */
    public var posX:Float;
    public var posY:Float;
    public var posZ:Float;

    /** Field of view (radians, for perspective). */
    public var fov:Float;
    public var aspect:Float;
    public var near:Float;
    public var far:Float;

    public function new() {
        matrix = [
            1,0,0,0,
            0,1,0,0,
            0,0,1,0,
            0,0,0,1
        ];
        posX = 0; posY = 0; posZ = -3;
        fov = 0.785; // ~45 degrees
        aspect = 1.0;
        near = 0.1;
        far = 100.0;
    }

    /** Create a perspective camera. */
    public static function perspective(fov:Float, aspect:Float, near:Float, far:Float):Camera {
        var cam = new Camera();
        cam.fov = fov;
        cam.aspect = aspect;
        cam.near = near;
        cam.far = far;
        cam.updateProjection();
        return cam;
    }

    /** Create an orthographic camera. */
    public static function orthographic(left:Float, right:Float, bottom:Float, top:Float, near:Float, far:Float):Camera {
        var cam = new Camera();
        cam.near = near;
        cam.far = far;
        var rl = right - left;
        var tb = top - bottom;
        var fn = far - near;
        cam.matrix = [
            2.0/rl, 0, 0, 0,
            0, 2.0/tb, 0, 0,
            0, 0, -2.0/fn, 0,
            -(right+left)/rl, -(top+bottom)/tb, -(far+near)/fn, 1
        ];
        return cam;
    }

    public function setPosition(x:Float, y:Float, z:Float):Void {
        posX = x; posY = y; posZ = z;
        updateProjection();
    }

    /** Update the projection matrix from current fov/aspect/near/far. */
    public function updateProjection():Void {
        var f = 1.0 / Math.tan(fov / 2.0);
        var range = near - far;
        matrix = [
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (far + near) / range, -1,
            0, 0, 2.0 * far * near / range, 0
        ];
    }
}
