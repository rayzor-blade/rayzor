// A virtual call answered by comparing the receiver's class against a few
// implementations must pick the same method as the vtable, including for
// inherited methods, a class created by reflection and a null receiver.
class Shape {
    public function new() {}

    public function area():Int
        return 0;

    public function name():String
        return "shape";

    public function scaled(k:Int):Int
        return area() * k;
}

class Square extends Shape {
    var side:Int;

    public function new(side:Int) {
        super();
        this.side = side;
    }

    override public function area():Int
        return side * side;

    override public function name():String
        return "square";
}

class Cube extends Square {
    override public function area():Int
        return 6 * super.area();
}

class Circle extends Shape {
    var r:Int;

    public function new(r:Int) {
        super();
        this.r = r;
    }

    override public function area():Int
        return 3 * r * r;
}

class Main {
    static function total(shapes:Array<Shape>):Int {
        var sum = 0;
        for (s in shapes)
            sum += s.area() + s.scaled(2) + s.name().length;
        return sum;
    }

    static function main() {
        var shapes:Array<Shape> = [new Shape(), new Square(3), new Cube(2), new Circle(2)];
        // 0+0+5 / 9+18+6 / 24+48+6 / 12+24+5
        if (total(shapes) != 157) throw "total " + total(shapes);
        var made:Shape = Type.createInstance(Type.resolveClass("Cube"), [1]);
        if (made.area() != 6 || made.name() != "square") throw "reflected";
        var none:Shape = null;
        var caught = false;
        try {
            none.area();
        } catch (e:Dynamic) {
            caught = true;
        }
        if (!caught) throw "null receiver";
        Sys.println("CONFORMANCE_OK");
    }
}
