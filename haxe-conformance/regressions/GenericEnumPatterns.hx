import haxe.ds.Option;

enum PatternPair<A, B> {
    Pair(a:A, b:B);
    Alternate(a:A, b:B);
}

enum PatternTree<T> {
    Leaf(value:T);
    Branch(left:PatternTree<T>, right:PatternTree<T>);
}

class GenericEnumPatterns {
    static function intEqual(got:Int, want:Int) {
        if (got != want) throw "Int: " + got + " != " + want;
    }

    static function floatEqual(label:String, got:Float, want:Float) {
        if (got != want) throw label + ": " + got + " != " + want;
    }

    static function main() {
        var dynamicInt:Dynamic = -10;
        var value:Option<Int> = Some(dynamicInt);
        intEqual(switch value { case Some(n): n; case _: 0; }, -10);
        var dynamicFloat:Dynamic = 1.5;
        var nested:Option<Option<Float>> = Some(Some(dynamicFloat));
        switch nested {
            case Some(inner):
                floatEqual("bound inner", switch inner { case Some(n): n; case _: 0.0; }, 1.5);
            case _: throw "missing outer value";
        }
        floatEqual("nested constructor", switch nested { case Some(Some(n)): n; case _: 0.0; }, 1.5);
        var pair:PatternPair<Int, Float> = Pair(dynamicInt, dynamicFloat);
        switch pair {
            case Pair(i, f): intEqual(i, -10); floatEqual("pair field", f, 1.5);
            case _: throw "missing pair";
        }
        pair = Alternate(dynamicInt, dynamicFloat);
        floatEqual("alternative", switch pair { case Pair(_, f) | Alternate(_, f): f; }, 1.5);
        var tree:PatternTree<Float> = Branch(Leaf(dynamicFloat), Leaf(dynamicFloat));
        switch tree {
            case Branch(left, right):
                floatEqual("left branch", switch left { case Leaf(n): n; case _: 0.0; }, 1.5);
                floatEqual("right branch", switch right { case Leaf(n): n; case _: 0.0; }, 1.5);
            case _: throw "missing branch";
        }
        Sys.println("CONFORMANCE_OK");
    }
}
