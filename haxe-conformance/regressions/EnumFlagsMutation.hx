private enum Flag { A; B; }
class EnumFlagsMutation {
    static function main() {
        var flags = new haxe.EnumFlags<Flag>();
        flags.set(B);
        if (flags.toInt() != 2) throw "set writeback";
        flags.set(A);
        if (flags.toInt() != 3) throw "second set writeback";
        var copy = flags;
        flags.unset(B);
        if (flags.toInt() != 1 || copy.toInt() != 3) throw "unset and value semantics";
        flags.unset(A);
        if (flags.toInt() != 0) throw "empty flags";
        Sys.println("CONFORMANCE_OK");
    }
}
