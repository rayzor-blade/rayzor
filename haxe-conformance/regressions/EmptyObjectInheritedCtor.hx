// `{}` passed to a constructor a subclass inherits from a generic base is an empty object.
typedef Props = {?variant:String, ?size:Int};

class Elem {
	public function new() {}
}

abstract class Comp<P> extends Elem {
	public final props:P;

	public function new(props:P) {
		super();
		this.props = props;
	}

	public function describe():String
		return props == null ? "null" : "set";
}

class Mine extends Comp<Props> {}

class Base<P> {
	public final props:P;

	public function new(props:P, ?extra:Int)
		this.props = props;
}

class Leaf extends Base<Props> {}
class Deeper extends Leaf {}

class EmptyObjectInheritedCtor {
	static function main() {
		var got = [new Mine({}).describe(), "" + (new Leaf({}).props == null), "" + (new Deeper({}).props == null), "" + (new Base<Props>({}).props == null), new Mine({variant: "x"}).describe()].join(",");
		if (got != "set,false,false,false,set")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
