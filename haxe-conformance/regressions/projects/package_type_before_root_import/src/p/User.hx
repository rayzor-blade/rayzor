package p;

// A bare `Input` in package p is p.Input, though a root-package file imports haxe.io.Input.
class User {
	public static function make():FormControl {
		var i:Input = new Input();
		return i;
	}
	public static function isOurs(x:Dynamic):Bool return Std.isOfType(x, Input);
}
