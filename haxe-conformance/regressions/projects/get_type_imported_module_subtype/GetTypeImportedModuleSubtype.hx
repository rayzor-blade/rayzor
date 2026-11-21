// Context.getType of a bare name finds a secondary type of a module the
// caller imports, as a use of that name in code does.
import lib.Card;

class GetTypeImportedModuleSubtype {
	static macro function find(name:String):haxe.macro.Expr {
		var t = try haxe.macro.Context.getType(name) catch (_:Dynamic) null;
		return macro $v{t == null ? "none" : switch t {
			case TInst(c, _): c.get().pack.join(".") + "." + c.get().name;
			case _: "other";
		}};
	}

	static function main() {
		var got = find("CardHeader") + " " + find("Card") + " " + find("Missing");
		if (got != "lib.CardHeader lib.Card none")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
