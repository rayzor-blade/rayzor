package sim;

class Particle {
    public var pos:Point2D;
    public var vel:Point2D;
    public var mass:Float;

    public function new(x:Float, y:Float, vx:Float, vy:Float, mass:Float) {
        this.pos = new Point2D(x, y);
        this.vel = new Point2D(vx, vy);
        this.mass = mass;
    }

    public function applyForce(force:Point2D):Void {
        var accel = force.scale(1.0 / mass);
        vel = vel.add(accel);
    }

    public function step(dt:Float):Void {
        pos = pos.add(vel.scale(dt));
    }

    public function kineticEnergy():Float {
        var speed = vel.length();
        return 0.5 * mass * speed * speed;
    }

    public function toString():String {
        return 'Particle{pos=${pos.toString()}, vel=${vel.toString()}, m=$mass}';
    }
}
