package lib;

typedef InputProps = {?value:String, ?disabled:Bool};

class Input extends Component<InputProps> {
	public function render():String
		return "input[" + props.value + "]";
}
