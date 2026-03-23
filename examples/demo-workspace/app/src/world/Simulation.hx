package world;

import sim.Particle;
import sim.Point2D;

class Simulation {
    public var particles:Array<Particle>;
    public var gravity:Point2D;
    public var time:Float;

    public function new() {
        particles = [];
        gravity = new Point2D(0, -9.81);
        time = 0;
    }

    public function addParticle(p:Particle):Void {
        particles.push(p);
    }

    public function step(dt:Float):Void {
        for (p in particles) {
            p.applyForce(gravity.scale(p.mass));
            p.step(dt);
        }
        time += dt;
    }

    public function totalEnergy():Float {
        var total = 0.0;
        for (p in particles) {
            total += p.kineticEnergy();
        }
        return total;
    }

    public function report():String {
        var result = 'particles=${particles.length}';
        return result;
    }
}
