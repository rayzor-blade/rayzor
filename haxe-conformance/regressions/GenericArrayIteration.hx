class GenericArrayIteration {
    static function first<T>(values:Array<T>):T {
        return values.iterator().next();
    }

    static function nextValue<T>(iterator:Iterator<T>):T {
        return iterator.next();
    }

    static function forward<T>(values:Array<T>):T {
        return first(values);
    }

    static function increment<T:Float>(values:Array<T>):Float {
        return values.iterator().next() + 1.0;
    }

    static function main() {
        if (first([0, 2]) != 0) throw "generic Int next";
        if (first([1.5]) != 1.5) throw "generic Float next";
        if (first(["text"]) != "text") throw "generic String next";
        if (first([false]) != false) throw "generic Bool next";
        if (nextValue([0].iterator()) != 0) throw "generic Int protocol";
        if (nextValue([2.25].iterator()) != 2.25) throw "generic Float protocol";
        if (nextValue(["value"].iterator()) != "value") throw "generic String protocol";
        if (forward([3.5]) != 3.5) throw "forwarded Float next";
        if (forward(["forward"]) != "forward") throw "forwarded String next";
        if (increment([1.5]) != 2.5) throw "constrained Float next";
        var mixed:Array<Dynamic> = [0, "text", 1.5, false, null];
        var iterator = mixed.iterator();
        if (nextValue(iterator) != 0 || nextValue(iterator) != "text" || nextValue(iterator) != 1.5 || nextValue(iterator) != false || nextValue(iterator) != null) throw "generic Dynamic next";
        var inner = first([[0, 4]]);
        if (inner[0] != 0 || inner[1] != 4) throw "generic nested Array";
        Sys.println("CONFORMANCE_OK");
    }
}
