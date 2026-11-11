package lib;

class Affine {
	public var a:Float;
	public function new(a) this.a = a;
}

enum Op {
	Fill(path:Int, transform:Affine, opacity:Float);
	Image(slot:Int, x:Float, y:Float, width:Float, height:Float, transform:Affine, opacity:Float);
	Blend(mode:Int);
	Pass(pass:Int, scene:Int);
	Mesh(mesh:Int, transform:Affine, scene:Int, opacity:Float);
	Stroke(path:Int, stroke:Int, transform:Affine, opacity:Float);
}

enum Step {
	Paths(from:Int);
	Meshes(draws:Array<Int>, passes:Array<Int>, scene:Int);
}

class Ctx {
	public final ops:Array<Op> = [];
	public function new() {}
}
