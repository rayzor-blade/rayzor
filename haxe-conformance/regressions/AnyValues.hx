abstract AnyText(String) from String {}

class AnyValueHolder {
    public var value:Any;
    public function new(value:Any) this.value = value;
}

class AnyValues {
    static function text():Any return "hello";
    static function wrappedText():Any {
        var value:AnyText = "wrapped";
        return value;
    }
    static function integer(value:Any):Int return value;
    static function string(value:Any):String return value;
    static function main() {
        var returned:String = text();
        if (returned.toUpperCase() != "HELLO") throw "Any return";
        if (string(wrappedText()) != "wrapped") throw "abstract return";
        if (integer(42) != 42) throw "Any argument";
        var values:Array<Any> = [12, false, "hey", {field:1}, 1.5];
        var number:Int = values[0];
        var flag:Bool = values[1];
        var word:String = values[2];
        var fraction:Float = values[4];
        if (number != 12 || flag || word != "hey" || fraction != 1.5) throw "Any array";
        var holder = new AnyValueHolder("field");
        if (string(holder.value) != "field") throw "Any field";
        holder.value = 23;
        if (integer(holder.value) != 23) throw "Any field assignment";
        var value:Any = "before";
        value = "after";
        if (string(value) != "after") throw "Any assignment";
        var nested:Any = ("nested":AnyText);
        if ((nested:String) != "nested") throw "nested abstract cast";
        var empty:Any = null;
        var nullable:String = empty;
        if (nullable != null) throw "Any null";
        Sys.println("CONFORMANCE_OK");
    }
}
