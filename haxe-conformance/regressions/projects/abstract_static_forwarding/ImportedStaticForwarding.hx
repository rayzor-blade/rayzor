import utilities.WrappedStatics as Wrapped;

class ImportedStaticForwarding {
    static function main() {
        if (Wrapped.read() != 14) throw "imported forwarded call";
        Wrapped.value = 27;
        var read = Wrapped.read;
        if (read() != 27) throw "imported forwarded method value";
        if (utilities.WrappedStatics.read() != 27) throw "qualified forwarded call";
        if (Wrapped.own() != 31) throw "imported declared static";
        Sys.println("CONFORMANCE_OK");
    }
}
