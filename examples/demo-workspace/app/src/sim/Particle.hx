package sim;

class Particle {
    public var pos:Vec2;
    public var vel:Vec2;
    public var mass:Float;

    public function new(x:Float, y:Float, vx:Float, vy:Float, mass:Float) {
        this.pos = new Vec2(x, y);
        this.vel = new Vec2(vx, vy);
        this.mass = mass;
    }

    public function applyForce(force:Vec2):Void {
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
