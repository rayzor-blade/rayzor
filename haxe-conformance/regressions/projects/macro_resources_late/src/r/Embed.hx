package r;

import haxe.macro.Context;
import haxe.macro.Expr;

class Embed {
	public static macro function text(name:String, body:String):Expr {
		Context.addResource(name, haxe.io.Bytes.ofString(body));
		return macro haxe.Resource.getString($v{name});
	}
}
