class Main {
    static function main() {
        // === Basic anonymous object ===
        var p = {x: 10, y: 20};
        trace(p.x);    // 10
        trace(p.y);    // 20

        var q = {a: 3.14, b: 2.71};
        trace(q.a);    // 3.14
        trace(q.b);    // 2.71

        // String field
        var r = {name: "hello", count: 42};
        trace(r.name);   // hello
        trace(r.count);  // 42

        // === COW semantics ===
        var a = {x: 1, y: 2};
        var b = a;
        trace(a.x);    // 1
        trace(b.x);    // 1

        // === Reflect API ===
        trace(Reflect.hasField(p, "x"));    // true
        trace(Reflect.hasField(p, "z"));    // false

        // hasField on different shapes
        trace(Reflect.hasField(q, "a"));    // true
        trace(Reflect.hasField(q, "x"));    // false

        // hasField on object with string field
        trace(Reflect.hasField(r, "name"));   // true
        trace(Reflect.hasField(r, "count"));  // true
        trace(Reflect.hasField(r, "other"));  // false

        // === Multiple objects, same shape ===
        var p2 = {x: 100, y: 200};
        trace(p2.x);   // 100
        trace(p2.y);   // 200
        trace(Reflect.hasField(p2, "x"));   // true
    }
}
