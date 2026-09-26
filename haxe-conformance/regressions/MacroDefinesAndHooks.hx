// Context.defineType adds its type to the module, whether built by hand from
// TypeDefKind constructors or quoted with `macro class` (parameters and
// defaults kept); onGenerate sees the compile's types and their metadata; a
// macro that waits for the typer runs once.
import haxe.macro.Context;
import haxe.macro.Expr;

@:keep class Kept {}

class MacroDefinesAndHooks {
	#if macro
	static var runs:Int = 0;
	#end

	static macro function typed(e:Expr) {
		runs++;
		var t = Context.typeof(e);
		return macro $v{haxe.macro.TypeTools.toString(t)};
	}

	static macro function runCount() return macro $v{runs};

	static macro function define() {
		Context.defineType({
			pack: [],
			name: "HandBuilt",
			pos: Context.currentPos(),
			kind: TDClass(),
			fields: [{
				name: "hello",
				access: [APublic, AStatic],
				pos: Context.currentPos(),
				kind: FFun({args: [], ret: null, expr: macro return "hand"})
			}]
		});
		Context.defineType(macro class Greeter {
			public static function greet(name:String, times:Int = 2):String {
				var out = "";
				for (i in 0...times)
					out += name;
				return out;
			}
		});
		Context.onGenerate(function(types) {
			var found = false;
			for (t in types)
				switch t {
					case TClassDecl(c) if (c.get().name == "Kept" && c.get().meta.has(":keep")):
						found = true;
					case _:
				}
			if (!found)
				Context.error("onGenerate did not see @:keep class Kept", Context.currentPos());
		});
		return macro null;
	}

	static function main() {
		define();
		var x = 1;
		var out:Array<String> = [];
		out.push(HandBuilt.hello());
		out.push(Greeter.greet("ab"));
		out.push(Greeter.greet("c", 1));
		out.push(typed(x));
		out.push(Std.string(runCount()));
		var got = out.join(" ");
		if (got == "hand abab c Int 1")
			Sys.println("CONFORMANCE_OK");
		else
			Sys.println("FAIL " + got);
	}
}
