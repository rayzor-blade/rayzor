typedef Props = {?min:Float}

// A Null<Float> reaches a Float parameter of a runtime-bound Math function as
// its number, not its box.
class NullFloatIntoMathArgs {
	static function clamp(props:Props, v:Float):Float {
		var min = props.min != null ? props.min : 0.0;
		return Math.max(min, Math.min(1.0, v));
	}

	static function main() {
		var p:Props = {min: 0.5};
		var n:Null<Float> = p.min;
		var got = [clamp({}, 0.2), clamp(p, 0.2), Math.max(n, 0.2), Math.abs(n), Math.floor(n + 1)].join(",");
		if (got != "0.2,0.5,0.5,0.5,1")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
