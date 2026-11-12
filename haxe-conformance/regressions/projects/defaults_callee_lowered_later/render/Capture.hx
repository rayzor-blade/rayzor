package render;

class Capture {
	public static function shot(name:String, width:Int, clear = 0xffffff, alpha = 1.0, scale = 1.5, flip = true, tag = "png"):String {
		return name + " " + Math.round(width * scale) + " " + clear + " " + alpha + " " + scale + " " + flip + " " + tag + " " + DefaultsCalleeLoweredLater.suffix();
	}
}
