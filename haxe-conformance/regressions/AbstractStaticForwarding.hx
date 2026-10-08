class AbstractStaticForwarding {
    static function main() {
        if (ForwardedStatics.read() != 12) throw "forwarded call";
        if (ForwardedStatics.value != 12) throw "forwarded read";
        ForwardedStatics.value = 19;
        if (ForwardedStatics.read() != 19) throw "forwarded write";
        var read = ForwardedStatics.read;
        if (read() != 19) throw "forwarded method value";
        if (SelectedStatics.read() != 19) throw "selective forwarding";
        if (OwnStatics.read() != 23) throw "declared static precedence";
        if (ChainedStatics.read() != 19) throw "chained forwarding";
        Sys.println("CONFORMANCE_OK");
    }
}

class StaticSource {
    public static var value:Int = 12;
    public static function read():Int return value;
}

@:forwardStatics
abstract ForwardedStatics(StaticSource) {}

@:forwardStatics(read)
abstract SelectedStatics(StaticSource) {}

@:forwardStatics
abstract OwnStatics(StaticSource) {
    public static function read():Int return 23;
}

@:forwardStatics
abstract ChainedStatics(ForwardedStatics) {}
