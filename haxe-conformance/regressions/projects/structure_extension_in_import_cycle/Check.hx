import lib.ui.Reactive;

class Check {
	public static function revealTag():String {
		var props:lib.components.Input.InputProps = {type: "password", revealable: true, reveal: true};
		return switch (props.reveal : Value<Bool>) {
			case null: "null";
			case Const(on): "const " + on;
			case Bound(_): "bound";
		}
	}
}
