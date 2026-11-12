package lib;

class Element {
	public final tag:String;

	public function new(?tag:String)
		this.tag = tag == null ? "element" : tag;
}
