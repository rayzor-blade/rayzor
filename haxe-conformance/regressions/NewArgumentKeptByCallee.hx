// A `new` passed straight to a call outlives it when the callee returns or
// stores that object.
class NewArgumentDoc {
	public final id:Int;
	public final name:String;

	public function new(id, name) {
		this.id = id;
		this.name = name;
	}
}

class NewArgumentReg {
	public static final all:Array<NewArgumentDoc> = [];

	static function register(d:NewArgumentDoc):NewArgumentDoc {
		all.push(d);
		return d;
	}

	static function keep(d:NewArgumentDoc):Void
		all.push(d);

	public static function make(name:String):NewArgumentDoc
		return register(new NewArgumentDoc(all.length, name));

	public static function add(name:String):Void
		keep(new NewArgumentDoc(all.length, name));
}

class NewArgumentKeptByCallee {
	static function main() {
		var a = NewArgumentReg.make("a");
		NewArgumentReg.add("b");
		var junk = [for (i in 0...64) "x" + i];
		var all = NewArgumentReg.all;
		var got = a.id + a.name + " " + all[0].name + all[1].name + all[1].id + " " + junk.length;
		if (got != "0a ab1 64")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
