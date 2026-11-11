import lib.Ops;

class P {
	public var x:Float = 1;
	public var y:Float = 2;
	public var w:Float = 3;
	public var opacity:Float = 4;
	public var transform:Int = 5;
	public var stroke:Int = 1;
	public function new() {}
}

class Q {
	public var stroke:Int = 2;
	public var w:Float = 6;
	public var opacity:Float = 7;
	public var transform:Int = 8;
	public var y:Float = 9;
	public var x:Float = 10;
	public function new() {}
}

class Painter {
	var steps:Array<String> = [];
	var runs:Array<Step> = [];
	public function new() {}

	public function build(ctx:Ctx):String {
		var pathsFrom = 0;
		var blend = 0;
		var inMask = false;
		function endPaths() {
			steps.push("paths" + pathsFrom + "/" + blend);
			pathsFrom++;
		}
		for (op in ctx.ops) {
			switch op {
				case Blend(mode):
					endPaths();
					blend = mode;
					continue;
				case Mesh(_, _, scene, _) | Pass(_, scene) if (inMask):
					continue;
				case Mesh(_, _, scene, _) | Pass(_, scene):
					var last:Null<Step> = null;
					if (runs.length > 0)
						last = runs[runs.length - 1];
					var run = switch last {
						case Meshes(draws, passes, s) if (last != null && s == scene): {draws: draws, passes: passes};
						case _:
							var made = {draws: [], passes: []};
							runs.push(Meshes(made.draws, made.passes, scene));
							made;
					}
					switch op {
						case Mesh(m, transform, _, opacity): run.draws.push(m + Std.int(transform.a + opacity));
						case Pass(pass, _): run.passes.push(pass);
						case _:
					}
					continue;
				case Fill(path, transform, opacity):
					steps.push("fill" + path + ":" + transform.a + ":" + opacity);
				case Stroke(path, stroke, transform, opacity):
					steps.push("stroke" + path + ":" + stroke + ":" + transform.a + ":" + opacity);
				case Image(slot, x, y, w, h, transform, opacity):
					steps.push("image" + slot + ":" + (x + y + w + h) + ":" + transform.a + ":" + opacity);
			}
		}
		endPaths();
		return steps.join(",") + "|" + [for (r in runs) switch r { case Meshes(d, p, s): d.join("+") + "/" + p.join("+") + "@" + s; case _: ""; }].join(";") + "|" + (new P().x + new Q().w);
	}
}

class OrPatternBindings {
	static function main() {
		var ctx = new Ctx();
		ctx.ops.push(Fill(1, new Affine(0.5), 0.25));
		ctx.ops.push(Blend(2));
		ctx.ops.push(Image(3, 1, 2, 3, 4, new Affine(2), 0.75));
		ctx.ops.push(Mesh(4, new Affine(1), 5, 2));
		ctx.ops.push(Pass(6, 5));
		ctx.ops.push(Stroke(7, 8, new Affine(3), 0.5));
		var got = new Painter().build(ctx);
		if (got != "fill1:0.5:0.25,paths0/0,image3:10:2:0.75,stroke7:8:3:0.5,paths1/2|7/6@5|7")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
