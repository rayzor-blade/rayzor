package lib;

import lib.Form;

abstract class Component<Props> extends Element {
	public final props:Props;
	public var count:Null<Int>;

	public function new(props:Props, ?children:Array<Element>, ?tree:Tree, ?count:Int) {
		super(tree);
		this.props = props;
		this.count = count;
		Form.last = render();
	}

	abstract function render():String;
}
