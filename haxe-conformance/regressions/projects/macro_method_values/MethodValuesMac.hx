import haxe.macro.Context;
import haxe.macro.Expr;

class Counter {
	var total = 0;
	var stepFun:Int->Void;
	var bare:Int->Void;

	public function new() {
		stepFun = this.step;
		bare = step;
	}

	function step(n:Int)
		total += n;

	public function run():String {
		var f = this.step;
		f(1);
		stepFun(10);
		bare(100);
		[1000].map(x -> { step(x); x; });
		return "" + total + "," + (stepFun == null) + "," + (bare == null);
	}
}

class MethodValuesMac {
	public static function build():Array<Field> {
		var fields = Context.getBuildFields();
		fields.push({name: "TEXT", access: [APublic, AStatic], kind: FVar(macro :String, macro $v{new Counter().run()}), pos: Context.currentPos()});
		return fields;
	}
}
