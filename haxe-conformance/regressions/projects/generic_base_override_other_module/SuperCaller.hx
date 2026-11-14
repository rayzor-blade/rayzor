class SuperCaller extends Base<Int> {
	override public function render():String
		return "over+" + super.render();
}
