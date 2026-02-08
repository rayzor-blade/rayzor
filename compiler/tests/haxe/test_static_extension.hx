using IntTools;

class IntTools {
    public static function double(n:Int):Int {
        return n * 2;
    }

    public static function isEven(n:Int):Bool {
        return n % 2 == 0;
    }

    public static function add(n:Int, m:Int):Int {
        return n + m;
    }
}

class Main {
    static function main() {
        var x = 5;
        trace(x.double());   // 10
        trace(x.isEven());   // false
        trace(x.add(3));     // 8

        var y = 4;
        trace(y.double());   // 8
        trace(y.isEven());   // true
    }
}
