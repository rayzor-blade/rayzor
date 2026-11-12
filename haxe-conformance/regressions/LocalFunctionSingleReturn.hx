// A local function or lambda returning Single returns the value, however it
// is called: directly, through a variable, or as a typed lambda.
class LocalFunctionSingleList {
	var data:Array<Single>;

	public function new(d:Array<Single>)
		data = d;

	public inline function get(at:Int, i:Int):Single
		return data[at + i];
}

class LocalFunctionSingleReturn {
	static function sfn(x:Int):Single
		return x * 2.5;

	static function main() {
		function f(x:Int):Single
			return x * 2.5;
		var v = f(2);
		var g = f;
		var h:Int->Single = x -> x * 2.5;
		var list = new LocalFunctionSingleList([1.5, 2.5, 120, 4]);
		inline function cell(row:Int, i:Int) return list.get(0, row * 4 + i);
		var got = v + " " + f(2) + " " + g(2) + " " + sfn(2) + " " + h(2) + " " + cell(0, 2);
		if (got != "5 5 5 5 5 120")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
