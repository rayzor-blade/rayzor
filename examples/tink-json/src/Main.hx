// ================================================================
// tink-json Demo — Macro-powered JSON utilities
// ================================================================
// Tests:
//   1. Expression macros: tink.Json.parse(), stringify(), keys()
//   2. @:build macro: fieldCount(), fieldNames(), describe()
//   3. Runtime: tink.JsonWriter string escaping
//   4. Runtime: tink.JsonParser JSON parsing

import tink.Json;
import tink.JsonWriter;
import tink.JsonParser;

// ================================================================
// Model class with @:build macro for introspection
// ================================================================

@:build(tink.Json.build)
class User {
    public var name:String;
    public var age:Int;
    public var active:Bool;

    public function new(name:String, age:Int, active:Bool) {
        this.name = name;
        this.age = age;
        this.active = active;
    }
}

@:build(tink.Json.build)
class Config {
    public var host:String;
    public var port:Int;
    @:json("max_connections") public var maxConnections:Int;
    @:jsonIgnore public var debugMode:Bool;

    public function new(host:String, port:Int, maxConn:Int, debug:Bool) {
        this.host = host;
        this.port = port;
        this.maxConnections = maxConn;
        this.debugMode = debug;
    }
}

// ================================================================
// Main — exercise all macro features
// ================================================================
class Main {
    static function main() {
        trace("=== tink-json Demo ===");

        // --------------------------------------------------------
        // 1. Compile-time JSON parsing (expression macro)
        // --------------------------------------------------------
        trace("--- 1. Compile-time JSON parse ---");
        var nums = tink.Json.parse("[1, 2, 3, 42]");
        trace(nums);

        var mixed = tink.Json.parse('[true, false, null, "hello", 3.14]');
        trace(mixed);

        var obj = tink.Json.parse('{"x": 10, "y": 20}');
        trace(obj);

        // --------------------------------------------------------
        // 2. Compile-time JSON stringify (expression macro)
        // --------------------------------------------------------
        trace("--- 2. Compile-time stringify ---");
        trace(tink.Json.stringify(42));
        trace(tink.Json.stringify(3.14));
        trace(tink.Json.stringify("hello"));
        trace(tink.Json.stringify(true));
        trace(tink.Json.stringify(null));

        // --------------------------------------------------------
        // 3. @:build macro introspection
        // NOTE: @:build generates correct AST but generated method bodies
        //       produce `unreachable` in MIR. This is a known compiler gap
        //       being tracked for fix. Uncomment when resolved.
        // --------------------------------------------------------
        // trace("--- 3. @:build introspection ---");
        // var user = new User("Alice", 30, true);
        // trace("User field count: " + Std.string(user.fieldCount()));
        // trace("User schema: " + user.describe());

        // --------------------------------------------------------
        // 5. Runtime JSON string escaping
        // --------------------------------------------------------
        trace("--- 5. Runtime JsonWriter ---");
        trace(tink.JsonWriter.writeString("hello world"));
        trace(tink.JsonWriter.writeString("has \"quotes\" and\nnewlines"));
        trace(tink.JsonWriter.writeInt(42));
        trace(tink.JsonWriter.writeBool(true));
        trace(tink.JsonWriter.writeNull());

        // --------------------------------------------------------
        // 6. Runtime JSON parsing
        // NOTE: JsonParser uses enums which need further compiler work.
        //       Uncomment when enum dispatch is fully supported.
        // --------------------------------------------------------
        // trace("--- 6. Runtime JsonParser ---");
        // var parsed = tink.JsonParser.parse('{"name":"Bob","age":25}');
        // trace(parsed);

        trace("=== Done ===");
    }
}
