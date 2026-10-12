import ArgumentConversions.ArgumentTypeName;
import ArgumentConversions.ArgumentArity;
import ArgumentConversions.ArgumentLength;
import ArgumentConversions.ArgumentListener;
import ArgumentChecks;

private enum ArgumentEvent {
    Message(value:String);
    Empty;
}

class MacroAbstractArguments {
    static var calls = 0;
    static function value():String { calls++; return "hello"; }
    static function name(value:ArgumentTypeName):String return value;
    static function arity(value:ArgumentArity):Int return value;
    static function length(value:ArgumentLength):Int return value;
    function member(value:ArgumentTypeName):String return value;
    static function optional(?prefix:Int, value:ArgumentTypeName):String return value;
    static function listener<T>(event:ArgumentListener<T>, handler:T):String return event;
    static function callback<T>(event:ArgumentListener<T>, handler:T):T return handler;
    static function main() {
        if (name("hello") != "String" || name(12) != "Int") throw "argument type name";
        var instance = new MacroAbstractArguments();
        if (instance.member(false) != "Bool") throw "member argument";
        if (arity(function(a:Int, b:String) {}) != 2) throw "argument arity";
        if (length(value()) != 5 || calls != 1) throw "argument evaluation";
        if (optional("hello") != "String") throw "optional argument";
        if (listener(Message, function(value:String) {}) != "Message") throw "generic listener";
        if (listener(Empty, function() {}) != "Empty") throw "nullary listener";
        var handler = callback(Message, function(value) {
            if (name(value) != "String") throw "inferred callback argument";
        });
        handler("hello");
        if (ArgumentChecks.rejects(name("hello"))) throw "nested argument conversion";
        if (ArgumentChecks.rejects(listener(Message, function(value:String) {}))) throw "valid listener";
        if (!ArgumentChecks.rejects(listener(Message, function() {}))) throw "invalid listener";
        if (!ArgumentChecks.rejects(listener(Empty, function(value:String) {}))) throw "invalid nullary listener";
        Sys.println("CONFORMANCE_OK");
    }
    public function new() {}
}
