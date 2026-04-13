// ================================================================
// tink.JsonParser — Runtime JSON string parsing
// ================================================================
// Minimal recursive-descent JSON parser for runtime deserialization.
// Returns a JsonValue enum that can be pattern-matched.

package tink;

enum JsonValue {
    JNull;
    JBool(v:Bool);
    JInt(v:Int);
    JFloat(v:Float);
    JString(v:String);
    JArray(v:Array<JsonValue>);
    JObject(v:Array<{key:String, value:JsonValue}>);
}

class JsonParser {
    var src:String;
    var pos:Int;

    public function new(json:String) {
        this.src = json;
        this.pos = 0;
    }

    public static function parse(json:String):JsonValue {
        var p = new JsonParser(json);
        return p.parseValue();
    }

    function parseValue():JsonValue {
        skipWhitespace();
        if (pos >= src.length) return JNull;

        var c = src.charAt(pos);

        if (c == "\"") return parseString();
        if (c == "{") return parseObject();
        if (c == "[") return parseArray();
        if (c == "t") return parseTrue();
        if (c == "f") return parseFalse();
        if (c == "n") return parseNull();
        if (c == "-" || (c >= "0" && c <= "9")) return parseNumber();

        return JNull;
    }

    function parseString():JsonValue {
        pos++; // skip "
        var result = "";
        while (pos < src.length) {
            var c = src.charAt(pos);
            if (c == "\"") {
                pos++;
                return JString(result);
            }
            if (c == "\\") {
                pos++;
                if (pos >= src.length) break;
                var esc = src.charAt(pos);
                if (esc == "n") result += "\n";
                else if (esc == "t") result += "\t";
                else if (esc == "r") result += "\r";
                else if (esc == "\"") result += "\"";
                else if (esc == "\\") result += "\\";
                else if (esc == "/") result += "/";
                else result += esc;
            } else {
                result += c;
            }
            pos++;
        }
        return JString(result);
    }

    function parseNumber():JsonValue {
        var start = pos;
        var isFloat = false;
        if (src.charAt(pos) == "-") pos++;
        while (pos < src.length) {
            var c = src.charAt(pos);
            if (c >= "0" && c <= "9") {
                pos++;
            } else if (c == "." && !isFloat) {
                isFloat = true;
                pos++;
            } else if (c == "e" || c == "E") {
                isFloat = true;
                pos++;
                if (pos < src.length && (src.charAt(pos) == "+" || src.charAt(pos) == "-")) pos++;
            } else {
                break;
            }
        }
        var numStr = src.substr(start, pos - start);
        if (isFloat) {
            return JFloat(Std.parseFloat(numStr));
        } else {
            return JInt(Std.parseInt(numStr));
        }
    }

    function parseArray():JsonValue {
        pos++; // skip [
        var elements = new Array<JsonValue>();
        skipWhitespace();
        if (pos < src.length && src.charAt(pos) == "]") {
            pos++;
            return JArray(elements);
        }
        while (pos < src.length) {
            elements.push(parseValue());
            skipWhitespace();
            if (pos < src.length && src.charAt(pos) == ",") {
                pos++;
            } else {
                break;
            }
        }
        skipWhitespace();
        if (pos < src.length && src.charAt(pos) == "]") pos++;
        return JArray(elements);
    }

    function parseObject():JsonValue {
        pos++; // skip {
        var fields = new Array<{key:String, value:JsonValue}>();
        skipWhitespace();
        if (pos < src.length && src.charAt(pos) == "}") {
            pos++;
            return JObject(fields);
        }
        while (pos < src.length) {
            skipWhitespace();
            // Parse key
            var keyVal = parseString();
            var key = switch (keyVal) {
                case JString(s): s;
                default: "";
            };
            skipWhitespace();
            if (pos < src.length && src.charAt(pos) == ":") pos++;
            // Parse value
            var value = parseValue();
            fields.push({key: key, value: value});
            skipWhitespace();
            if (pos < src.length && src.charAt(pos) == ",") {
                pos++;
            } else {
                break;
            }
        }
        skipWhitespace();
        if (pos < src.length && src.charAt(pos) == "}") pos++;
        return JObject(fields);
    }

    function parseTrue():JsonValue {
        pos += 4;
        return JBool(true);
    }

    function parseFalse():JsonValue {
        pos += 5;
        return JBool(false);
    }

    function parseNull():JsonValue {
        pos += 4;
        return JNull;
    }

    function skipWhitespace():Void {
        while (pos < src.length) {
            var c = src.charAt(pos);
            if (c == " " || c == "\t" || c == "\n" || c == "\r") {
                pos++;
            } else {
                break;
            }
        }
    }
}
