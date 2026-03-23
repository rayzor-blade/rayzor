import sim.Point2D;
import sim.Particle;
import world.Simulation;

class Main {
    static function main() {
        trace("=== Rayzor Particle Demo ===");

        // Cross-project: MathUtils from mathlib
        trace('fib(15) = ${MathUtils.fibonacci(15)}');
        trace('7! = ${MathUtils.factorial(7)}');

        // Local packages: sim.Point2D
        var a = new Point2D(3.0, 4.0);
        var b = new Point2D(1.0, 2.0);
        trace('a = ${a.toString()}');
        trace('|a| = ${a.length()}');
        trace('a + b = ${a.add(b).toString()}');
        trace('dist = ${a.distanceTo(b)}');

        // Local packages: sim.Particle + world.Simulation
        var sim = new Simulation();
        sim.addParticle(new Particle(0.0, 100.0, 5.0, 0.0, 1.0));
        sim.addParticle(new Particle(10.0, 50.0, -3.0, 2.0, 2.0));
        sim.addParticle(new Particle(5.0, 75.0, 1.0, -1.0, 0.5));

        trace("--- Initial State ---");
        trace(sim.report());

        // Run 10 steps
        for (i in 0...10) {
            sim.step(0.1);
        }

        trace("--- After 1 second ---");
        trace(sim.report());

        // Primes via mathlib
        var primes:Array<Int> = [];
        for (i in 2...30) {
            if (MathUtils.isPrime(i)) primes.push(i);
        }
        trace('primes: $primes');
    }
}
