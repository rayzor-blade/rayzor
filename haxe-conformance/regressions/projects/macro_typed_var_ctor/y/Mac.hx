package y;

#if macro
import haxe.macro.Context;
import haxe.macro.Expr;

class Mac {
	static function lower(e:Expr):String {
		var d:y.Ast.Def = switch (e.expr) {
			case EBlock(el): EBlock([for (x in el) 1]);
			case EFunction(FNamed(name, _), f): EFunction({name: name, n: f.args.length});
			case _: EBlock([]);
		}
		return show(d);
	}

	static function show(d:y.Ast.Def):String {
		return Type.getEnumName(Type.getEnum(d)) + ":" + switch (d) {
			case EBlock(items): "block" + items.length;
			case EFunction(f): "fn " + f.name + f.n;
			case EBreak: "break";
		}
	}

	static function report():String {
		var direct:y.Ast.Def = EFunction({name: "x", n: 1});
		var same:y.Ast.Def = EBlock([1, 2]);
		var local:Ast.Def = EBlock([7]);
		var flag = same != null;
		var ternary:y.Ast.Def = flag ? EBreak : EBlock([]);
		var block:y.Ast.Def = {
			var n = 3;
			EBlock([for (i in 0...n) i]);
		}
		return [
			lower(macro {1; 2;}),
			lower(macro function foo(a:Int, b:Int) {}),
			show(direct),
			show(same),
			show(local),
			show(ternary),
			show(block),
		].join(",");
	}

	public static function build():Array<Field> {
		var fields = Context.getBuildFields();
		fields.push({
			name: "TEXT",
			access: [APublic, AStatic],
			kind: FVar(macro :String, macro $v{report()}),
			pos: Context.currentPos()
		});
		return fields;
	}
}
#end
