package p;

class Base<P> {
	public var props:P;
	public var seen:String;

	public function new(props:P, ?extra:Array<Int>) {
		this.props = props;
		seen = describe();
	}

	function describe():String
		return "base";
}
