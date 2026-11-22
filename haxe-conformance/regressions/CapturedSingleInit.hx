// A Single local that a closure captures keeps the value it was declared with,
// and so does a Float local declared from a Single.
#if !rayzor
typedef Single = Float;
#end

class CapturedSingleInit {
	static function near(a:Float, b:Float):Bool
		return Math.abs(a - b) < 1e-4;

	static function param(s:Single):() -> Single
		return () -> s;

	static function main() {
		var fs:Single = 18.72;
		var read = () -> fs;
		var one:Single = 1.0;
		var bump = () -> one += 0.5;
		bump();
		var n = 3;
		var fromInt:Single = n;
		var scale = (x:Single) -> x * fromInt;
		var wide:Float = 0.25;
		var narrow:Single = wide;
		var half = () -> narrow * 2;
		var back:Float = fs;
		var later = () -> back;

		var got = [
			near(fs, 18.72), near(read(), 18.72), one == 1.5, scale(2.0) == 6,
			half() == 0.5, near(later(), 18.72), param(360.0)() == 360
		];
		if (got.indexOf(false) >= 0)
			throw got.join(" ");
		Sys.println("CONFORMANCE_OK");
	}
}
