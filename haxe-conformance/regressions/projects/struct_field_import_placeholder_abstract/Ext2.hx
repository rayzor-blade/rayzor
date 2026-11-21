import MyIR;

class Ext2 {
	public static function main() {
		var props:XInput.XProps = {type: "password", revealable: true, reveal: true};
		var tag = switch (props.reveal : MyRT<Bool>) {
			case null: "null";
			case Const(on): "const " + on;
			case Bound(_): "bound";
		}
		if (tag != "const true")
			throw tag;
		Sys.println("CONFORMANCE_OK");
	}
}
