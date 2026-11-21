package lib.ui;

import lib.components.Input as Themed;
import lib.ui.Form;
import lib.ui.Label;

typedef InputProps = {
	?type:String,
	?checked:Reactive<Bool>,
	?reveal:Reactive<Bool>
}

class Input {
	public var props:InputProps;

	public function new(props:InputProps)
		this.props = props;

	public static function themed(props:lib.components.Input.InputProps):Themed
		return new Themed(props);

	public static function form():Form
		return new Form();

	public static function label():Label
		return new Label();
}
