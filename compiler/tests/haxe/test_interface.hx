interface Drawable {
    public function draw():Void;
}

class Circle implements Drawable {
    public function new() {}
    public function draw() {
        trace("circle");
    }
}

class Square implements Drawable {
    public function new() {}
    public function draw() {
        trace("square");
    }
}

class Main {
    static function main() {
        var d:Drawable = new Circle();
        d.draw();  // circle

        d = new Square();
        d.draw();  // square
    }
}
