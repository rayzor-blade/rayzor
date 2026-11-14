import haxe.io.Input;
import haxe.io.Bytes;

class Main {
	static function main() {
		var stdin:Input = new haxe.io.BytesInput(Bytes.ofString("x"));
		var got = p.User.make().checkValidity() + " " + p.User.isOurs(new p.Input()) + " " + p.User.isOurs(stdin) + " " + stdin.readByte();
		if (got != "true true false 120")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
