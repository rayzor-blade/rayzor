// An unannotated `[]` local passed to a constructor takes the formal's element
// type, as a call argument does.
class CtorArgInfersEmptyArrayHolder {
	public function new(out:Array<String>) {
		var push = (s:String) -> out.push(s);
		push("ctor");
	}
}

class CtorArgInfersEmptyArray {
	static function viaCall(out:Array<String>) {
		out.push("call");
	}

	static function main() {
		var a = [];
		viaCall(a);
		var b = [];
		new CtorArgInfersEmptyArrayHolder(b);
		var got = a.join(",") + "|" + b.join(",") + "|" + b[0].length;
		if (got != "call|ctor|4")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
