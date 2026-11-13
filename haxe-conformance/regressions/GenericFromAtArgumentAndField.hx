// A class value meets an abstract through its generic `@:from` as a call
// argument and as a structure field, not through the header's `from`.
class GfaSig<T> {
	public var v:T;
	public function new(v:T) this.v = v;
}
enum GfaRV<T> {
	Const(value:T);
	Bound(s:GfaSig<T>);
}
abstract GfaIR<T>(GfaRV<T>) from GfaRV<T> to GfaRV<T> {
	@:from public static function fromConst<T>(v:T):GfaIR<T> return Const(v);
	@:from public static function fromGfaSig<T>(s:GfaSig<T>):GfaIR<T> return Bound(s);
}
typedef GfaProps = {?value:GfaIR<String>, ?label:String};
class GfaComp {
	public var props:GfaProps;
	public function new(p:GfaProps) props = p;
	public function kind():String
		return switch (props.value : GfaRV<String>) {
			case null: "null";
			case Const(v): "const " + v;
			case Bound(s): "bound " + s.v;
		}
}
class GenericFromAtArgumentAndField {
	static function take(x:GfaIR<String>):String
		return switch (x : GfaRV<String>) {
			case Const(v): "const " + v;
			case Bound(s): "bound " + s.v;
		}
	static function main() {
		var s = new GfaSig("pear");
		var p:GfaProps = {value: s};
		var got = [
			take(s),
			take("x"),
			new GfaComp(p).kind(),
			new GfaComp({value: s}).kind(),
			new GfaComp({value: "c"}).kind(),
			new GfaComp({label: "l"}).kind()
		].join(",");
		if (got != "bound pear,const x,bound pear,bound pear,const c,null")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
