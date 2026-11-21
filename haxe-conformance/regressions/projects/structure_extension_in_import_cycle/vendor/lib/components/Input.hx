package lib.components;

typedef InputProps = {
	> lib.ui.Input.InputProps,
	?size:Int,
	?revealable:Bool
}

class Input {
	public var props:InputProps;

	public function new(props:InputProps)
		this.props = props;
}
