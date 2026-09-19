// A base class compiled before the module that subclasses it must still
// dispatch through a vtable: `Output.writeInt32` called `writeByte` on the
// base directly (NotImplemented) instead of BytesOutput's override, because
// only overrides visible in the same module made a method virtual. The
// declaration index sees every file; both sides lay slots out by name.
import haxe.io.BytesOutput;
import haxe.io.BytesInput;
class CrossModuleOverride {
    static function main() {
        var o = new BytesOutput();
        o.writeInt32(0x01020304);
        o.writeInt16(7);
        o.writeDouble(0.5);
        var b = o.getBytes();
        if (b.length != 14) throw "length " + b.length;
        if (b.get(0) != 4 || b.get(3) != 1) throw "int32 bytes";
        var i = new BytesInput(b);
        if (i.readInt32() != 0x01020304) throw "readInt32";
        if (i.readInt16() != 7) throw "readInt16";
        if (i.readDouble() != 0.5) throw "readDouble";
        Sys.println("CONFORMANCE_OK");
    }
}
