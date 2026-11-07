package geometry;

class Circle {
    public final radius:Float;

    public function new(radius:Float) {
        this.radius = radius;
    }

    public function area():Float {
        return Math.PI * radius * radius;
    }
}
