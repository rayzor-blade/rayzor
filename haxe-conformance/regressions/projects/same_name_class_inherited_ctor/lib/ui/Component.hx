package lib.ui;
abstract class Component<P> {
	public final props:P;
	public var out:String;
	public function new(props:P) {
		this.props = props;
		out = run(render);
	}
	static function run(f:Void->String):String return f();
	abstract function render():String;
}
