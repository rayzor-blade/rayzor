class BoxedArrayUpdates {
    static var indexReads = 0;
    static function index():Int { indexReads++; return 0; }
    static function main() {
        var boxed:Array<Dynamic> = [7, 8];
        var ints:Array<Int> = cast boxed;
        if (ints[0]++ != 7 || ints[0] != 8 || boxed[0] != 8) throw "postfix boxed Int";
        if (++ints[1] != 9 || boxed[1] != 9) throw "prefix boxed Int";
        var source:Array<Dynamic> = [1.5];
        var floats:Array<Float> = cast source;
        if (floats[0]++ != 1.5 || source[0] != 2.5) throw "postfix boxed Float";
        var vector = new haxe.ds.Vector<Float>(1);
        vector[0] = 1.5;
        if (vector[0]++ != 1.5 || vector[0] != 2.5) throw "postfix vector Float";
        if (++vector[index()] != 3.5 || vector[0] != 3.5) throw "prefix vector Float";
        if (indexReads != 1) throw "array index evaluation";
        if (vector[0]-- != 3.5 || --vector[0] != 1.5) throw "vector decrement";
        Sys.println("CONFORMANCE_OK");
    }
}
