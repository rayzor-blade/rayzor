// haxe.io.Bytes reads and writes unsigned 16-bit little-endian values.
class BytesUInt16 {
	static function main() {
		var b = haxe.io.Bytes.alloc(6);
		b.setUInt16(0, 0xBEEF);
		b.setUInt16(2, 70000);
		b.setUInt16(4, -2);
		var got = b.getUInt16(0) + " " + b.getUInt16(2) + " " + b.getUInt16(4) + " " + b.toHex();
		if (got != "48879 4464 65534 efbe7011feff")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
