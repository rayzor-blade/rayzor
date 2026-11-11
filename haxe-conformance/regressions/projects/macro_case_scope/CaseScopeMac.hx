import haxe.macro.Context;
import haxe.macro.Expr;
import other.Uses;

enum D {
	Glob(name:String);
	Call(e:N, args:Array<N>);
	Wrap(e:N);
}

typedef N = {e:D, t:String};

enum G {
	Dot;
	Cross;
	TFloat;
}

class CaseScopeMac {
	static function map(e:N, f:N->N):N {
		var ed = switch (e.e) {
			case Wrap(e): Wrap(f(e));
			case Call(e, args): Call(f(e), [for (a in args) f(a)]);
			case Glob(_): e.e;
		}
		return {e: ed, t: e.t};
	}

	public static function build():Array<Field> {
		var fields = Context.getBuildFields();
		var call:N = {e: Call({e: Glob("vec4"), t: "Fun"}, [{e: Glob("x"), t: "Float"}]), t: "Vec4"};
		var wrapped:N = {e: Wrap(call), t: "Outer"};
		var id = (n:N) -> n;
		var names = [for (g in G.createAll()) switch (g) {
			case Dot: "dot";
			case Cross: "cross";
			case TFloat: "float";
		}];
		var t = TFloat;
		var text = map(call, id).t + "," + map(wrapped, id).t + "," + names.join("") + "," + Type.enumConstructor(t) + "," + Uses.name(cast "cross");
		fields.push({name: "TEXT", access: [APublic, AStatic], kind: FVar(macro :String, macro $v{text}), pos: Context.currentPos()});
		return fields;
	}
}
