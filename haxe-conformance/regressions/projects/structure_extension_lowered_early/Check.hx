import Reactive;

class Check {
	public static function run() {
		var props:Form.FormProps = {type: "password", revealable: true, reveal: true};
		var tag = switch (props.reveal : Value<Bool>) {
			case null: "null";
			case Const(on): "const " + on;
			case Bound(_): "bound";
		}
		if (tag != "const true")
			throw tag;
		Sys.println("CONFORMANCE_OK");
	}
}
