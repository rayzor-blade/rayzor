// `macro class` builds a TypeDefinition at run time that Printer prints; an
// array pattern nested in a constructor pattern is tested, not skipped; a
// method called through a `Null<>` or typedef of Array is Array's.
import haxe.macro.Expr;

enum Box { Items(xs:Array<Int>); }
typedef Ints = Array<Int>;

class MacroClassAndArrayAliases {
	static function kind(b:Box) return switch b {
		case Items([]): "empty";
		case Items(xs): "n=" + xs.length;
	}

	static function show(i:Int) return "<" + i + ">";

	static var failed = false;

	static function check(label:String, got:Dynamic, want:Dynamic) {
		if (got != want) {
			failed = true;
			Sys.println("FAIL " + label + ": got " + got + " want " + want);
		}
	}

	static function main() {
		var td = macro class Foo<@:foo T> {
			final x:Int = 0;
			public function get() return 1;
		}
		var s = ~/[\t\n\r]/g.replace(new haxe.macro.Printer().printTypeDefinition(td), "");
		check("printed", s, "class Foo<@:foo T> {final x : Int = 0;public function get() return 1;}");
		check("fields", td.fields.length + " " + td.name, "2 Foo");
		check("nested array pattern", kind(Items([])) + " " + kind(Items([1, 2])), "empty n=2");
		var a:Ints = [1, 2];
		var n:Null<Array<Int>> = [3];
		check("alias map", a.map(show).join(",") + " " + n.map(show).join(","), "<1>,<2> <3>");
		var entries:Array<MetadataEntry> = [{name: ":m", params: [], pos: null}];
		var tp:TypeParamDecl = {name: "T", meta: entries};
		check("typedef field", tp.meta[0].name + " " + tp.meta.map(m -> m.name).join(","), ":m :m");
		if (!failed)
			Sys.println("CONFORMANCE_OK");
	}
}
