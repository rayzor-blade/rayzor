// A macro's EReg takes backreferences and lookahead, as Haxe's does.
#if macro
import haxe.macro.Expr;
#end

class MacroERegBackreference {
	static macro function unquote(s:String):Expr {
		var re = ~/^(["'])([\s\S]*)\1$/;
		return macro $v{re.match(s) ? re.matched(2) : s};
	}

	static macro function words(s:String):Expr {
		var re = ~/(\w+)(?=,)/g;
		var ci = ~/HELLO/i;
		return macro $v{re.replace(s, "<$1>") + " " + ci.match(s)};
	}

	static function main() {
		#if !macro
		var got = unquote("'abc'") + " " + unquote("\"x\"") + " " + unquote("'mixed\"") + " " + words("hello,world,end");
		if (got != "abc x 'mixed\" <hello>,<world>,end true")
			throw got;
		Sys.println("CONFORMANCE_OK");
		#end
	}
}
