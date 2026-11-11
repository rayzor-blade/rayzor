package ui;

abstract class Base {
	abstract function render():Int;

	public function make():Int
		return render() + 1;

	public function new() {}
}
