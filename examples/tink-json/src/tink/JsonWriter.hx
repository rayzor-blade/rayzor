// ================================================================
// tink.JsonWriter — Runtime JSON string construction helpers
// ================================================================
// Used by macro-generated toJson() methods. Provides safe string
// escaping and value-to-JSON conversion utilities.

package tink;

class JsonWriter {
    public static function escapeString(s:String):String {
        if (s == null) return "";
        var result = "";
        var i = 0;
        while (i < s.length) {
            var c = s.charAt(i);
            if (c == "\"") {
                result += "\\\"";
            } else if (c == "\\") {
                result += "\\\\";
            } else if (c == "\n") {
                result += "\\n";
            } else if (c == "\r") {
                result += "\\r";
            } else if (c == "\t") {
                result += "\\t";
            } else {
                result += c;
            }
            i++;
        }
        return result;
    }

    public static function writeInt(v:Int):String {
        return Std.string(v);
    }

    public static function writeFloat(v:Float):String {
        return Std.string(v);
    }

    public static function writeBool(v:Bool):String {
        return v ? "true" : "false";
    }

    public static function writeString(v:String):String {
        if (v == null) return "null";
        return "\"" + escapeString(v) + "\"";
    }

    public static function writeNull():String {
        return "null";
    }

    public static function writeArray(items:Array<String>):String {
        var result = "[";
        var i = 0;
        while (i < items.length) {
            if (i > 0) result += ",";
            result += items[i];
            i++;
        }
        result += "]";
        return result;
    }
}
