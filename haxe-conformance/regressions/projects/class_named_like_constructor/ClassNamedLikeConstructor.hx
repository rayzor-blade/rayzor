import Stroke;

class ClassNamedLikeConstructor {
	static function describe(op:Op):String {
		switch op {
			case Stroke(s, o):
				return "stroke " + s.w + " " + o;
			case Image:
				return "image";
			case Op.Fill(o):
				return "fill " + o;
		}
	}

	static function main() {
		var bare:Op = Image;
		var ops:Array<Op> = [Op.Stroke(new Stroke(), 0.25), Op.Image, Op.Fill(0.5), bare];
		var value = switch ops[0] {
			case Stroke(_, o): o;
			case _: -1.0;
		}
		var got = [for (op in ops) describe(op)].join(",") + "," + value;
		if (got != "stroke 1 0.25,image,fill 0.5,image,0.25")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
