abstract MyInt(Int) {
    public static function triple(v:Int):Int {
        return v * 3;
    }

    public static function fromString(s:String):Int {
        return 42;
    }
}

class Main {
    static function main() {
        var result = MyInt.triple(7);
        trace(result); // 21

        var val = MyInt.fromString("hello");
        trace(val); // 42
    }
}
