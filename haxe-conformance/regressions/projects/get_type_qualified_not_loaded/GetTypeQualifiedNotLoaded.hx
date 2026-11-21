// Context.getType of a qualified name names that type, loading its module,
// though a loaded type of another package shares its last segment.
class GetTypeQualifiedNotLoaded {
	static macro function chain(name:String):haxe.macro.Expr {
		var out = [];
		var cur = switch haxe.macro.Context.getType(name) {
			case TInst(c, _): out.push(c.get().pack.join(".") + "." + c.get().name); c.get().superClass;
			case _: null;
		}
		while (cur != null) {
			out.push(cur.t.get().pack.join(".") + "." + cur.t.get().name);
			cur = cur.t.get().superClass;
		}
		return macro $v{out.join(" -> ")};
	}

	static function main() {
		var input = new haxe.io.BytesInput(haxe.io.Bytes.alloc(1));
		var got = chain("lib.ui.Input") + " " + input.length;
		if (got != "lib.ui.Input -> lib.ui.Component 1")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
