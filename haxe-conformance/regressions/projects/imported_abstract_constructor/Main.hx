import b.Buf2;

class Main {
	static function main() {
		var y = new Buf2(16);
		if ((y : haxe.io.Bytes).length != 16) throw "constructor of an imported abstract";
		if (y.len() != 16) throw "member of an imported abstract";
		Sys.println("CONFORMANCE_OK");
	}
}
