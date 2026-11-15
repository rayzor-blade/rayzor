typedef Props = {?height:Float, ?count:Int};

// `(v : Single)` of a Null<Float> or Null<Int> is its number narrowed,
// not the box's address converted.
class NullScalarAscribedSingle {
	static function boxF():Null<Float>
		return 200.0;

	static function boxI():Null<Int>
		return 7;

	static function main() {
		var f = boxF();
		var i = boxI();
		var props:Props = {count: 3};
		var height = props.height != null ? props.height : 180.0;
		var got = [
			Std.string((f : Single)),
			Std.string((cast f : Single) + 1),
			Std.string((i : Single)),
			Std.string((height : Single)),
			Std.string((props.count : Single) * 2)
		].join(",");
		if (got != "200,201,7,180,6")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
