// Fields an abstract does not declare resolve through `@:op(a.b)` /
// `@:resolve`: a read passes the name, a write the name and the value, and
// `op=` on such a field reaches the abstract's own `op=` overload. An
// `@:arrayAccess` method of any name serves by arity, and `this` inside an
// interpolated string survives inlining.
private abstract Path(String) {
	public function new(value:String) this = value;

	@:op(a.b) function read(name:String) return new Path('$this.$name');

	@:op(a.b) function write(name:String, value:Path) return new Path('$this.$name=$value');

	@:op(a * b) function times(rhs:Path) return new Path('$this*${rhs.text()}');

	@:op(a -= b) function less(rhs:Path) return new Path('$this-=${rhs.text()}');

	public function text() return this;

	@:from static function fromString(s:String) return new Path(s);
}

private abstract Counter(Int) from Int {
	@:op(A + B) @:arrayAccess @:resolve
	public inline function measure(key:String):Int return this + key.length;
}

class AbstractResolveOperators {
	static function main() {
		var out = [];
		var p = new Path("p");
		out.push(p.a.b.text());
		p = p.w = "v";
		out.push(p.text());
		var q = new Path("q");
		q = (q.x *= "y");
		out.push(q.text());
		var r = new Path("r");
		r = (r.x -= "y");
		out.push(r.text());
		var c:Counter = 10;
		out.push(Std.string(c.abcd + c['ab'] + (c + 'a')));
		var got = out.join(" ");
		if (got == "p.a.b p.w=v q.x=q.x*y r.x-=y 37")
			Sys.println("CONFORMANCE_OK");
		else
			Sys.println("FAIL " + got);
	}
}
