package lib.components;
import lib.ui.Component;
typedef LabelProps = {?htmlFor:String, ?required:Bool, ?id:String}
class Label extends Component<LabelProps> {
	static var guard = 0;
	function render():String {
		if (++guard > 5) return "recursed";
		var el = new lib.ui.Label({htmlFor: props.htmlFor, id: props.id});
		return "components.Label>" + el.out;
	}
}
