// Neither imports `keys.Key`, so a bare `Select` is the class, not `Key.Select`.
class Option extends Base {
	public var label:String;

	public function new(l:String) {
		super();
		label = l;
	}

	public function render():Void {
		Select.register(this, "box");
	}
}
