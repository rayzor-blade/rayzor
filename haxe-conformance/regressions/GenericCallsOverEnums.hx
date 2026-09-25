// A generic whose type variable is read off an enum argument is specialized,
// called directly or as a static extension.
using Lambda;

enum Access { APublic; AStatic; AFinal; }
typedef Field = {?access:Array<Access>};

class GenericCallsOverEnums {
	static var failed = false;

	static function check(label:String, got:Dynamic, want:Dynamic) {
		if (got != want) {
			failed = true;
			Sys.println("FAIL " + label + ": got " + got + " want " + want);
		}
	}

	static function main() {
		var access = [APublic, AFinal];
		check("direct", Lambda.has(access, AFinal) + " " + Lambda.has(access, AStatic), "true false");
		check("extension", access.has(APublic), true);
		var f:Field = {access: [AStatic]};
		check("optional field", f.access != null && f.access.has(AStatic), true);
		check("ints", [1, 2].has(2), true);
		if (!failed)
			Sys.println("CONFORMANCE_OK");
	}
}
