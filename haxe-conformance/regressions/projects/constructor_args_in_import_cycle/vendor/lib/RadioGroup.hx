package lib;

import lib.Form;

class RadioGroup extends Component<{?value:String}> {
	function render():String
		return "group";
}

class RadioGroupItem extends Component<{value:String, ?disabled:Bool}> {
	var field:Null<String> = null;

	function render():String
		return "item " + props.value;
}
