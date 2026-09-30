import haxe.io.Bytes;
import haxe.io.Bytes.fastGet as fget;

class BytesDataViews {
    static function main() {
        var bytes = Bytes.ofString("AB");
        var data = bytes.getData();

        bytes.set(0, "C".code);
        if (data[0] != "C".code || fget(data, 0) != "C".code)
            throw "BytesData did not observe a byte write";
        var erased:Dynamic = data;
        if ((cast erased[0] : Int) != "C".code)
            throw "Dynamic BytesData did not observe a byte write";

        data[1] = "D".code;
        if (bytes.get(1) != "D".code)
            throw "byte buffer did not observe a BytesData write";

        var alias = Bytes.ofData(data);
        alias.set(0, "E".code);
        if (bytes.toString() != "ED")
            throw "ofData did not share the buffer";

        Sys.println("CONFORMANCE_OK");
    }
}
