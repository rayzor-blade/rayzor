class Main {
    static function main() {
        // Test 1: Parse simple object
        var obj = haxe.Json.parse('{"name":"Alice","age":30}');
        trace(Reflect.field(obj, "name"));  // Alice
        trace(Reflect.field(obj, "age"));   // 30

        // Test 2: Parse array
        var arr = haxe.Json.parse('[1, 2, 3]');
        trace(arr);  // [1,2,3]

        // Test 3: Parse nested
        var nested = haxe.Json.parse('{"items":[10,20],"ok":true}');
        trace(Reflect.field(nested, "ok"));  // true

        // Test 4: Parse primitives
        trace(haxe.Json.parse("42"));     // 42
        trace(haxe.Json.parse("3.14"));   // 3.14
        trace(haxe.Json.parse("true"));   // true
        trace(haxe.Json.parse("false"));  // false
        trace(haxe.Json.parse("null"));   // null
        trace(haxe.Json.parse('"hello"'));  // hello

        // Test 5: Stringify
        // var result = haxe.Json.stringify({x: 1, y: 2});
        // trace(result);

        trace("done");
    }
}
