class ContextualCallbackReturns {
    static function text(value:CallbackNumber, callback:CallbackNumber->String):String {
        return callback(value);
    }
    static function floating(value:CallbackNumber, callback:CallbackNumber->Float):Float {
        return callback(value);
    }
    static function aliased(value:CallbackNumber, callback:CallbackNumber->CallbackText):CallbackText {
        return callback(value);
    }
    static function boolean(value:CallbackNumber, callback:CallbackNumber->Bool):Bool {
        return callback(value);
    }
    static function boxed(value:CallbackNumber, callback:CallbackNumber->CallbackBox):CallbackBox {
        return callback(value);
    }
    static function ignored(callback:()->Void):Void callback();

    static function main() {
        var value:CallbackNumber = 2;
        if (text(value, function(v) return v) != "number2") throw "explicit callback return";
        if (text(value, v -> v) != "number2") throw "arrow callback return";
        if (text(value, v -> { var result = v; return result; }) != "number2") throw "block callback return";
        if (text(value, function(v):String return v) != "number2") throw "annotated callback return";
        if (aliased(value, v -> v) != "number2") throw "aliased callback return";
        if (floating(value, v -> v) != 2.5) throw "floating callback conversion";
        if (!boolean(value, v -> v)) throw "boolean callback conversion";
        if (boxed(value, v -> v).value != 12) throw "class callback conversion";
        if (boxed(value, function(v) return v).value != 12) throw "class explicit return";
        ignored(function() return 1);
        ignored(() -> 1);
        var executed = 0;
        var callback = () -> { executed++; return 1; };
        ignored(callback);
        if (executed != 1) throw "Void callback execution";
        Sys.println("CONFORMANCE_OK");
    }
}
typedef CallbackText = String;
abstract CallbackNumber(Int) from Int to Int {
    @:to public function text():String return "number" + Std.string(this);
    @:to public function floating():Float return this + 0.5;
    @:to public function boolean():Bool return this > 0;
    @:to public function boxed():CallbackBox return new CallbackBox(this + 10);
}
class CallbackBox {
    public var value:Int;
    public function new(value:Int) this.value = value;
}
