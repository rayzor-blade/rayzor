// A closure declared to return an interface returns a class instance as that
// interface, so a method called on its result dispatches.
interface Highlighter {
	function highlight(line:String):String;
}

class Words implements Highlighter {
	final tag:String;

	public function new(tag:String)
		this.tag = tag;

	public function highlight(line:String):String
		return tag + ":" + line;
}

typedef Host = {
	final highlighter:Null<String>->Highlighter;
}

class ClosureInterfaceResult {
	static function through(f:String->Highlighter, name:String):String
		return f(name).highlight(name);

	static function main() {
		var words = new Words("w");
		var arrow:String->Highlighter = name -> words;
		var block:String->Highlighter = name -> {
			return words;
		};
		var literal:String->Highlighter = function(name) return words;
		var host:Host = {highlighter: name -> name == null ? words : new Words(name)};
		var nullable:Int->Null<Highlighter> = n -> n > 0 ? words : null;
		var thunk:Void->Highlighter = () -> new Words("t");
		var got = [
			arrow("a").highlight("x"),
			block("b").highlight("y"),
			literal("c").highlight("z"),
			host.highlighter(null).highlight("d"),
			host.highlighter("js").highlight("e"),
			nullable(1).highlight("f"),
			Std.string(nullable(0) == null),
			thunk().highlight("g"),
			through(name -> new Words(name), "h")
		].join(" ");
		if (got != "w:x w:y w:z w:d js:e w:f true t:g h:h")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
