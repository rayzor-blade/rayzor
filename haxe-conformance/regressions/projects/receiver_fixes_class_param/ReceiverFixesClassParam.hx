import lib.Holder;
import lib.Reactive;

// A method of a generic class takes its class parameter from the receiver:
// a Table<Dynamic> stores a value as Dynamic whatever the argument's type.
class ReceiverFixesClassParam {
	static function main() {
		var holder = new Holder();
		holder.keep(1, (Live(new Cell<Float>(() -> 0.25)) : Reactive<Float>));
		holder.keep(2, (Fixed(3) : Reactive<Int>));
		var got = holder.readAll().join(" ");
		if (got != "live 0.25 fixed 3")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
