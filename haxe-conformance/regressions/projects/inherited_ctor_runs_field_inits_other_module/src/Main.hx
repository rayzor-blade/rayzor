// Child, built in a module that compiles before it, runs its own field initialisers.
class Main {
	static function main() {
		var child = p.Aplot.make();
		var got = child.seen + " " + child.props.name;
		if (got != "child 2 xx x")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
