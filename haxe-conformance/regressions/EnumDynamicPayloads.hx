import haxe.ds.Option;

enum TypedPayload {
    Values(i:Int, f:Float, b:Bool, s:String);
    Nullable(n:Null<Int>);
    Erased(d:Dynamic);
    Real(f:Float);
}

class EnumDynamicPayloads {
    static function check(label:String, got:Dynamic, want:Dynamic) {
        if (got != want) throw label + ": " + got + " != " + want;
    }

    static function read(value:TypedPayload) {
        switch value {
            case Values(i, f, b, s):
                check("Int payload", i, -10);
                check("Float payload", f, 1.5);
                check("Bool payload", b, true);
                check("String payload", s, "payload");
            case _: throw "wrong constructor";
        }
        var params = Type.enumParameters(value);
        check("reflected Int", params[0], -10);
        check("reflected Float", params[1], 1.5);
        check("reflected Bool", params[2], true);
        check("reflected String", params[3], "payload");
    }

    static function readInt(got:Int, want:Int) {
        if (got != want) throw "Int: " + got + " != " + want;
    }

    static function fromDynamic(value:Dynamic):Option<Int> {
        return Some(value);
    }

    static function readFloatOption(value:Option<Float>) {
        switch value {
            case Some(number): check("nested context", number, 1.5);
            case _: throw "missing inner value";
        }
    }

    static function readObject(value:{foo:Int}) {
        readInt(value.foo, 12);
    }

    static function main() {
        var i:Dynamic = -10;
        var f:Dynamic = 1.5;
        var b:Dynamic = true;
        var s:Dynamic = "payload";
        read(Values(i, f, b, s));
        read(TypedPayload.Values(i, f, b, s));
        var generic:Option<Int> = Some(i);
        switch generic {
            case Some(n): readInt(n, -10);
            case _: throw "missing generic value";
        }
        var nullableSource:Null<Dynamic> = -10;
        generic = Some(nullableSource);
        switch generic {
            case Some(n): readInt(n, -10);
            case _: throw "missing nullable-source value";
        }
        var positive:Dynamic = 10;
        generic = Some(positive);
        check("literal pattern", switch generic { case Some(10): true; case _: false; }, true);
        generic = fromDynamic(positive);
        check("return context", switch generic { case Some(10): true; case _: false; }, true);
        var nested:Option<Option<Float>> = Some(Some(f));
        switch nested {
            case Some(inner): readFloatOption(inner);
            case _: throw "missing outer value";
        }
        var real = Real(2);
        switch real {
            case Real(number): check("Int to Float", number, 2.0);
            case _: throw "wrong Real constructor";
        }
        var nullable = Nullable(null);
        check("null payload", switch nullable { case Nullable(n): n == null; case _: false; }, true);
        nullable = Nullable(i);
        check("optional payload", switch nullable { case Nullable(n): n == -10; case _: false; }, true);
        var dynamicInput:Dynamic = 42;
        var erased = Erased(dynamicInput);
        switch erased {
            case Erased(d): readInt(d, 42);
            case _: throw "wrong erased constructor";
        }
        var object = {foo:12};
        erased = Erased(object);
        switch erased {
            case Erased(d): readObject(d);
            case _: throw "wrong object constructor";
        }
        Sys.println("CONFORMANCE_OK");
    }
}
