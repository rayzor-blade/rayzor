class Base<P> {
	public var props:P;
	public function new(props:P) this.props = props;

	public function render():String
		return "base";

	public function describe():String
		return "describe(" + render() + "," + props + ")";

	public function keep():String
		return "keep:" + props;
}
