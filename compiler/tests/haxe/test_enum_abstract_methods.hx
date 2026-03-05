enum abstract Color(Int) {
    var Red = 0;
    var Green = 1;
    var Blue = 2;

    // Instance method — arithmetic on underlying value
    public function doubled():Int {
        return (cast this : Int) * 2;
    }

    // Instance method — boolean logic with underlying value
    public function isPrimary():Bool {
        return (cast this : Int) == 0 || (cast this : Int) == 2;
    }

    // Instance method using switch on underlying value
    public function toName():String {
        return switch (cast this : Int) {
            case 0: "Red";
            case 1: "Green";
            case 2: "Blue";
            default: "Unknown";
        };
    }

    // Static factory method
    public static function fromInt(value:Int):Color {
        return cast value;
    }
}

enum abstract HttpStatus(Int) {
    var OK = 200;
    var NotFound = 404;
    var ServerError = 500;

    public function isSuccess():Bool {
        return (cast this : Int) >= 200 && (cast this : Int) < 300;
    }

    public function isError():Bool {
        return (cast this : Int) >= 400;
    }
}

class Main {
    static function main() {
        // Test 1: Basic enum abstract values
        var r = Color.Red;
        var g = Color.Green;
        var b = Color.Blue;
        trace(r);  // 0
        trace(g);  // 1
        trace(b);  // 2

        // Test 2: Instance method — arithmetic
        trace(r.doubled());  // 0
        trace(g.doubled());  // 2
        trace(b.doubled());  // 4

        // Test 3: Instance method — boolean
        trace(r.isPrimary());  // true
        trace(g.isPrimary());  // false
        trace(b.isPrimary());  // true

        // Test 4: Static factory method
        var c2 = Color.fromInt(1);
        trace(c2);  // 1

        // Test 5: Method on variable from static factory
        trace(c2.doubled());  // 2

        // Test 6: HttpStatus — different underlying values
        var ok = HttpStatus.OK;
        var nf = HttpStatus.NotFound;
        trace(ok.isSuccess());  // true
        trace(ok.isError());    // false
        trace(nf.isSuccess());  // false
        trace(nf.isError());    // true

        // Test 7: Instance method — switch (complex body)
        trace(r.toName());  // Red
        trace(g.toName());  // Green
        trace(b.toName());  // Blue

        trace("done");
    }
}
