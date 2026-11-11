package ui;

class Card extends Holder<String> {
	function build(props:String):String
		return "card " + props;
}
