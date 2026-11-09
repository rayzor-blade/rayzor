enum RG<T> {
	Cst(v:T);
	Sig(s:Array<T>);
}

@:forward
abstract IR<T>(RG<T>) from RG<T> to RG<T> {
	@:from public static inline function ofConst<T>(v:T):IR<T>
		return Cst(v);

	@:from public static inline function ofInt(v:Int):IR<Float>
		return Cst((v : Float));
}

class AbstractEnumPayloads {
	static function check(value:Bool, label:String):Void {
		if (!value)
			throw label;
	}

	static function mk<T>(v:T):RG<T>
		return Cst(v);

	static function takeFloat(r:IR<Float>):String
		return switch (r) {
			case Cst(v): "" + v;
			case _: "other";
		}

	static function takeInt(r:IR<Int>):String
		return switch (r) {
			case Cst(v): "" + v;
			case _: "other";
		}

	static function sumSig(r:IR<Float>):Float
		return switch (r) {
			case Sig(s):
				var t = 0.0;
				for (x in s)
					t += x;
				t;
			case _: -1;
		}

	static function bindIt<T>(r:IR<T>):String
		return takeFloat(cast r);

	static function main():Void {
		check(takeFloat(mk(5.5)) == "5.5", "IR<Float> payload");
		check(takeInt(mk(5)) == "5", "IR<Int> payload");
		var r:IR<Float> = 100.5;
		var x = 2.5;
		var fromVar:IR<Float> = x;
		check(takeFloat(fromVar) == "2.5", "IR<Float> initializer from a variable");
		check(takeFloat(r) == "100.5", "IR<Float> through @:from");
		check(takeFloat(100) == "100", "IR<Float> from an Int literal");
		check(bindIt(r) == "100.5", "IR<T> cast to IR<Float>");
		check(sumSig(Sig([1.5, 2.25])) == 3.75, "IR<Float> array payload");
		trace("CONFORMANCE_OK");
	}
}
