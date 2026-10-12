class OptionalArgumentSlots {
    public function new() {}

    static function check(?path:String, ?enabled:Bool = true, ?count:Int = 7, ?suffix:String):Void {
        if (path != null) throw "omitted String slot";
        if (enabled != true) throw "Bool default slot";
        if (count != 7) throw "Int default slot";
        if (suffix != null) throw "trailing String slot";
    }

    function visit(callback:String->Void, ?path:String, ?enabled:Bool = true):Void {
        if (path != null) throw "omitted method String slot";
        if (enabled != true) throw "method Bool default slot";
        callback("visited");
    }

    static function arrays(?items:Array<String>, ?label:String = "default"):Void {
        if (items != null) throw "omitted Array slot";
        if (label != "default") throw "String default slot";
    }

    static function main():Void {
        check();
        check(null);
        check(null, null);
        arrays();
        var visited = false;
        new OptionalArgumentSlots().visit(value -> {
            if (value != "visited") throw "callback argument";
            visited = true;
        });
        if (!visited) throw "callback not called";
        trace("CONFORMANCE_OK");
    }
}
