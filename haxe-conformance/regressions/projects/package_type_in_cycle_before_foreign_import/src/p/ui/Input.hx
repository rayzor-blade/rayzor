package p.ui;

class Input implements FormControl {
	public function new() {}

	public static function at(id:Int):Null<Input>
		return id > 0 ? new Input() : null;

	public function checkValidity():Bool
		return true;

	public function owner():String
		return Form.NAME;
}
