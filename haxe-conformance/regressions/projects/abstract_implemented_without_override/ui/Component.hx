package ui;

abstract class Component<Props> {
	public function new() {}

	abstract function render():String;

	public function build():String
		return run(render);

	public function direct():String
		return render();

	static function run(f:() -> String):String
		return f();
}
