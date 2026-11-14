// A sub-type extending its module's main type, in a module that refers back to
// the module importing it, compiles.
class Sig {
	public function new() {}
}

class SigString extends Sig {
	public function new(initial:String) {
		super();
		Sys.println(Main.tag + " len=" + initial.length);
	}
}
