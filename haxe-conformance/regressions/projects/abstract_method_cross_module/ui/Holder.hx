package ui;

abstract class Holder<P> {
	abstract function build(props:P):String;

	public function show(props:P):String
		return "[" + build(props) + "]";

	public function new() {}
}
