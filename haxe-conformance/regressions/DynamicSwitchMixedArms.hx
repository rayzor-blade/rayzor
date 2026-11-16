// A switch whose arms mix a class instance (or a nullable one) with scalars
// and null is a Dynamic, and each arm enters it as its own value.
class DynamicSwitchMixedArmsBrush {
	public function new() {}

	public function toString():String
		return "Brush";
}

class DynamicSwitchMixedArms {
	static var cache:Null<DynamicSwitchMixedArmsBrush> = null;

	static function initial(prop:Int):Null<Dynamic> {
		return switch prop {
			case 0: cache == null ? cache = new DynamicSwitchMixedArmsBrush() : cache;
			case 4: (1.0 : Single);
			case 64: (0.0 : Single);
			case _: null;
		}
	}

	static function held(k:Int):Null<Dynamic> {
		return switch k {
			case 0: cache;
			case 1: (1.5 : Single);
			case _: null;
		}
	}

	static function fresh(k:Int):Null<Dynamic> {
		return switch k {
			case 0: new DynamicSwitchMixedArmsBrush();
			case _: 2;
		}
	}

	static function show(v:Null<Dynamic>):String
		return v == null ? "null" : "" + v;

	static function main() {
		var out = [show(held(0))];
		for (p in [4, 0, 64, 9999])
			out.push(show(initial(p)));
		out.push(show(held(0)));
		out.push(show(held(1)));
		out.push(show(fresh(0)));
		out.push(show(fresh(1)));
		var got = out.join(",");
		if (got != "null,1,Brush,0,null,Brush,1.5,Brush,2")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
