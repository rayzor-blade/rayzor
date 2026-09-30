import haxe.io.Bytes;

class BytesHex {
    static function main() {
        var bytes = Bytes.ofHex("00A1ff");
        if (bytes.length != 3 || bytes.get(1) != 0xA1 || bytes.toHex() != "00a1ff")
            throw "hex round trip";

        var caught = false;
        try {
            Bytes.ofHex("abc");
        } catch (error:Dynamic) {
            caught = true;
        }
        if (!caught) throw "odd-length hex accepted";
        Sys.println("CONFORMANCE_OK");
    }
}
