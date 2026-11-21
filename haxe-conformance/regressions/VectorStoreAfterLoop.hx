// A Float a loop accumulates is stored by its own type once the loop exits.
import haxe.ds.Vector;

class VectorStoreAfterLoop {
	static function forLoop(x:Float, out:Vector<Float>):Void {
		var r = 0.0;
		for (i in 0...4)
			r += x;
		out[0] = r;
	}

	static function whileLoop(x:Float, out:Vector<Float>):Void {
		var r = 0.0, i = 0;
		while (i < 4) {
			r += x;
			i++;
		}
		out[0] = r;
	}

	// A Float that is a join of two branches, not a loop.
	static function branch(x:Float, flag:Bool, out:Vector<Float>):Void {
		var r = 0.0;
		if (flag)
			r = x;
		else
			r = x * 2;
		out[0] = r;
	}

	// The same loop, storing into an Array<Float> and a field instead.
	static function intoArray(x:Float, out:Array<Float>):Void {
		var r = 0.0;
		for (i in 0...4)
			r += x;
		out[0] = r;
	}

	static function intoVectorOfInt(x:Int, out:Vector<Int>):Void {
		var r = 0;
		for (i in 0...4)
			r += x;
		out[0] = r;
	}

	static function main() {
		var out = [];
		var v = new Vector<Float>(1);
		forLoop(2.0, v);
		out.push(v[0]);
		whileLoop(2.0, v);
		out.push(v[0]);
		branch(2.0, false, v);
		out.push(v[0]);
		var a = [-1.0];
		intoArray(2.0, a);
		out.push(a[0]);
		var vi = new Vector<Int>(1);
		intoVectorOfInt(2, vi);
		out.push(vi[0]);
		var got = out.join(",");
		if (got != "8,8,4,8,8")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
