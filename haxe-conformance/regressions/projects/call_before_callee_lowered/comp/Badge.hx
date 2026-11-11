package comp;

class Badge {
	public final result:String;

	public function new(variant:String)
		result = Library.part("badge", ["variant" => variant, "size" => "md"]);
}
