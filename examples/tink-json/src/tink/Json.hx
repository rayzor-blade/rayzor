// ================================================================
// tink.Json — Macro-powered JSON utilities
// ================================================================
// Provides compile-time JSON operations via expression macros:
//
//   tink.Json.parse('{"x":1}')     — compile-time JSON → Haxe literal
//   tink.Json.stringify(42)         — compile-time value → JSON string
//   tink.Json.keys('{"a":1,"b":2}') — extract keys at compile time
//
// Also provides @:build macro for class introspection:
//   @:build(tink.Json.build) on a class generates:
//     - fieldCount(): Int
//     - fieldNames(): Array<String>
//     - describe(): String

package tink;

import haxe.macro.Context;
import haxe.macro.Expr;

class Json {

    // ================================================================
    // Expression macro: compile-time JSON string → Haxe literal
    // ================================================================
    // Usage: var data = tink.Json.parse('{"x": 42, "name": "hello"}');
    // Expands to array of values at compile time.
    macro public static function parse(jsonStr:String):Expr {
        var result = parseValue(jsonStr, 0);
        if (result == null) {
            Context.error("Invalid JSON string", Context.currentPos());
            return macro null;
        }
        return result.expr;
    }

    // ================================================================
    // Expression macro: compile-time value → JSON string literal
    // ================================================================
    // Usage: var s = tink.Json.stringify(42);  // expands to "42"
    //        var s = tink.Json.stringify("hi"); // expands to "\"hi\""
    macro public static function stringify(value:Expr):Expr {
        // Inspect the expression AST at compile time and produce
        // the corresponding JSON string as a string literal.
        switch (value.expr) {
            case EConst(CInt(s)):
                // Integer literal → JSON number string
                return macro $v{Std.string(s)};
            case EConst(CFloat(s)):
                return macro $v{Std.string(s)};
            case EConst(CString(s)):
                return macro $v{"\"" + s + "\""};
            case EConst(CIdent("true")):
                return macro "true";
            case EConst(CIdent("false")):
                return macro "false";
            case EConst(CIdent("null")):
                return macro "null";
            default:
                // Non-literal: fall back to runtime Std.string
                return macro Std.string($e{value});
        }
    }

    // ================================================================
    // @:build macro for class introspection
    // ================================================================
    macro public static function build():Array<Field> {
        var fields = Context.getBuildFields();
        var cls = Context.getLocalClass().get();
        var className = cls.name;

        // Collect serializable fields
        var jsonFields:Array<{name:String, jsonKey:String, typeStr:String}> = [];
        for (f in fields) {
            var skip = false;
            for (m in f.meta) {
                if (m.name == ":jsonIgnore" || m.name == "jsonIgnore") skip = true;
            }
            if (skip) continue;

            switch (f.kind) {
                case FVar(t, _):
                    var jsonKey = f.name;
                    for (m in f.meta) {
                        if (m.name == ":json" || m.name == "json") {
                            if (m.params.length > 0) {
                                switch (m.params[0].expr) {
                                    case EConst(CString(s)):
                                        jsonKey = s;
                                    default:
                                }
                            }
                        }
                    }
                    var typeStr = t != null ? haxe.macro.ComplexTypeTools.toString(t) : "Dynamic";
                    jsonFields.push({name: f.name, jsonKey: jsonKey, typeStr: typeStr});
                default:
            }
        }

        // Generate fieldCount()
        fields.push({
            name: "fieldCount",
            access: [APublic, AInline],
            kind: FFun({
                args: [],
                ret: macro :Int,
                expr: macro return $v{jsonFields.length}
            }),
            pos: Context.currentPos()
        });

        // Generate fieldNames()
        var nameExprs:Array<Expr> = [];
        for (jf in jsonFields) {
            nameExprs.push(macro $v{jf.jsonKey});
        }
        fields.push({
            name: "fieldNames",
            access: [APublic],
            kind: FFun({
                args: [],
                ret: null,
                expr: macro return [$a{nameExprs}]
            }),
            pos: Context.currentPos()
        });

        // Generate describe()
        var schema = className + " {";
        for (jf in jsonFields) {
            schema += "\n  " + jf.jsonKey + ": " + jf.typeStr;
            if (jf.jsonKey != jf.name) schema += " (field: " + jf.name + ")";
        }
        schema += "\n}";
        fields.push({
            name: "describe",
            access: [APublic, AInline],
            kind: FFun({
                args: [],
                ret: macro :String,
                expr: macro return $v{schema}
            }),
            pos: Context.currentPos()
        });

        return fields;
    }

    // ================================================================
    // Internal: compile-time JSON parser (runs in macro interpreter)
    // ================================================================
    static function parseValue(s:String, pos:Int):{expr:Expr, endPos:Int, keys:Array<String>} {
        while (pos < s.length) {
            var c = s.charAt(pos);
            if (c == " " || c == "\t" || c == "\n" || c == "\r") { pos++; } else { break; }
        }
        if (pos >= s.length) return null;
        var c = s.charAt(pos);
        if (c == "\"") return parseString(s, pos);
        if (c == "-" || (c >= "0" && c <= "9")) return parseNumber(s, pos);
        if (s.length >= pos + 4 && s.substr(pos, 4) == "true")
            return {expr: macro true, endPos: pos + 4, keys: []};
        if (s.length >= pos + 5 && s.substr(pos, 5) == "false")
            return {expr: macro false, endPos: pos + 5, keys: []};
        if (s.length >= pos + 4 && s.substr(pos, 4) == "null")
            return {expr: macro null, endPos: pos + 4, keys: []};
        if (c == "[") return parseArray(s, pos);
        if (c == "{") return parseObject(s, pos);
        return null;
    }

    static function parseString(s:String, pos:Int):{expr:Expr, endPos:Int, keys:Array<String>} {
        pos++;
        var result = "";
        while (pos < s.length) {
            var c = s.charAt(pos);
            if (c == "\"") return {expr: macro $v{result}, endPos: pos + 1, keys: []};
            if (c == "\\") {
                pos++;
                if (pos >= s.length) return null;
                var esc = s.charAt(pos);
                if (esc == "n") result += "\n";
                else if (esc == "t") result += "\t";
                else if (esc == "\\") result += "\\";
                else if (esc == "\"") result += "\"";
                else result += esc;
            } else {
                result += c;
            }
            pos++;
        }
        return null;
    }

    static function parseNumber(s:String, pos:Int):{expr:Expr, endPos:Int, keys:Array<String>} {
        var start = pos;
        var isFloat = false;
        if (s.charAt(pos) == "-") pos++;
        while (pos < s.length) {
            var c = s.charAt(pos);
            if (c >= "0" && c <= "9") pos++;
            else if (c == "." && !isFloat) { isFloat = true; pos++; }
            else break;
        }
        var numStr = s.substr(start, pos - start);
        if (isFloat) {
            var f = Std.parseFloat(numStr);
            return {expr: macro $v{f}, endPos: pos, keys: []};
        } else {
            var n = Std.parseInt(numStr);
            return {expr: macro $v{n}, endPos: pos, keys: []};
        }
    }

    static function parseArray(s:String, pos:Int):{expr:Expr, endPos:Int, keys:Array<String>} {
        pos++;
        var elements:Array<Expr> = [];
        while (pos < s.length && s.charAt(pos) != "]") {
            while (pos < s.length && (s.charAt(pos) == " " || s.charAt(pos) == "\n" || s.charAt(pos) == "\t" || s.charAt(pos) == "\r")) pos++;
            if (pos < s.length && s.charAt(pos) == "]") break;
            var elem = parseValue(s, pos);
            if (elem == null) return null;
            elements.push(elem.expr);
            pos = elem.endPos;
            while (pos < s.length && (s.charAt(pos) == " " || s.charAt(pos) == "," || s.charAt(pos) == "\n" || s.charAt(pos) == "\t" || s.charAt(pos) == "\r")) pos++;
        }
        if (pos < s.length) pos++; // skip ]
        return {expr: macro [$a{elements}], endPos: pos, keys: []};
    }

    static function parseObject(s:String, pos:Int):{expr:Expr, endPos:Int, keys:Array<String>} {
        pos++;
        var keyNames:Array<String> = [];
        var valueExprs:Array<Expr> = [];
        while (pos < s.length && s.charAt(pos) != "}") {
            while (pos < s.length && (s.charAt(pos) == " " || s.charAt(pos) == "\n" || s.charAt(pos) == "\t" || s.charAt(pos) == "\r")) pos++;
            if (pos < s.length && s.charAt(pos) == "}") break;
            var key = parseString(s, pos);
            if (key == null) return null;
            // Extract key string from the Expr
            var keyStr = "";
            switch (key.expr.expr) {
                case EConst(CString(ks)): keyStr = ks;
                default:
            }
            keyNames.push(keyStr);
            pos = key.endPos;
            while (pos < s.length && (s.charAt(pos) == " " || s.charAt(pos) == ":")) pos++;
            var val = parseValue(s, pos);
            if (val == null) return null;
            valueExprs.push(val.expr);
            pos = val.endPos;
            while (pos < s.length && (s.charAt(pos) == " " || s.charAt(pos) == "," || s.charAt(pos) == "\n" || s.charAt(pos) == "\t" || s.charAt(pos) == "\r")) pos++;
        }
        if (pos < s.length) pos++; // skip }
        // Return values as an array (keys available via .keys field)
        return {expr: macro [$a{valueExprs}], endPos: pos, keys: keyNames};
    }
    #end
}
