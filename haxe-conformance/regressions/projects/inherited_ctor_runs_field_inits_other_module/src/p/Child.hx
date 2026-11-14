package p;

// No constructor of its own: its field initialisers run before Base's.
class Child extends Base<{name:String}> {
	final painter = [1, 2];
	var label = "child";

	override function describe():String
		return label + " " + painter.length + " " + props.name + Aplot.NAME;
}
