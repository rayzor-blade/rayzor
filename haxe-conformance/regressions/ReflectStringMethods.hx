using Reflect;

class ReflectStringMethods {
    static function call(receiver:Dynamic, name:String, args:Array<Dynamic>):Dynamic {
        var method = Reflect.field(receiver, name);
        if (!Reflect.isFunction(method)) throw "missing String method: " + name;
        return Reflect.callMethod(receiver, method, args);
    }

    static function main() {
        var s = "barbar";
        if (call(s, "charAt", [1]) != "a") throw "charAt";
        if (call(s, "charAt", [-1]) != "") throw "charAt bounds";
        if (call(s, "charCodeAt", [1]) != 97) throw "charCodeAt";
        if (call(s, "charCodeAt", [99]) != null) throw "charCodeAt bounds";
        if (call(s, "indexOf", ["bar"]) != 0 || call(s, "indexOf", ["bar", 1]) != 3) throw "indexOf";
        if (call(s, "lastIndexOf", ["bar"]) != 3 || call(s, "lastIndexOf", ["bar", 2]) != 0) throw "lastIndexOf";
        if (call(s, "lastIndexOf", ["bar", null]) != 3) throw "null optional index";
        var parts:Dynamic = call(s, "split", ["a"]);
        if (parts.length != 3 || parts[0] != "b" || parts[1] != "rb" || parts[2] != "r") throw "split String slots";
        var chars:Dynamic = call("ab", "split", [""]);
        if (chars.length != 2 || chars[0] != "a" || chars[1] != "b") throw "empty delimiter";
        var unsplit:Dynamic = call(s, "split", ["long delimiter"]);
        if (unsplit.length != 1 || unsplit[0] != s) throw "long delimiter";
        if (call(s, "substr", [-3]) != "bar" || call(s, "substr", [1, 2]) != "ar") throw "substr";
        if (call(s, "substr", [1, null]) != "arbar") throw "null optional length";
        if (call(s, "substring", [1]) != "arbar" || call(s, "substring", [6, 3]) != "bar") throw "substring";
        if (call(s, "toUpperCase", []) != "BARBAR" || call("BARBAR", "toLowerCase", []) != s) throw "case conversion";
        if (call(s, "toString", []) != s) throw "toString";
        var method:Dynamic = Reflect.field(s, "indexOf");
        if (method("bar") != 0 || method("bar", 1) != 3) throw "direct Dynamic call";
        var extension:Dynamic = s.field("indexOf");
        if (extension("bar", 1) != 3 || s.field("length") != 6) throw "Reflect extension receiver";
        if (Reflect.callMethod("other receiver", method, ["bar", 1]) != 3) throw "bound receiver";
        if (!Reflect.compareMethods(method, Reflect.field(s, "indexOf"))) throw "method identity";
        if (Reflect.compareMethods(method, Reflect.field(s, "lastIndexOf"))) throw "distinct methods";
        var dynamicString:Dynamic = s;
        var charAt:Dynamic = dynamicString.charAt;
        if (charAt(1) != "a") throw "Dynamic field method";
        if (Reflect.field(s, "missing") != null) throw "missing field";
        Sys.println("CONFORMANCE_OK");
    }
}
