package p.ui;

// Form and Input name each other, so Form compiles before Input is declared;
// its bare `Input` is still p.ui.Input, not the haxe.io.Input Helper imports.
class Form {
	public static inline var NAME = "form";

	public static function check():String {
		var control:Null<FormControl> = Input.at(3);
		return control == null ? "null" : "valid " + control.checkValidity() + " " + Input.at(1).owner();
	}
}
