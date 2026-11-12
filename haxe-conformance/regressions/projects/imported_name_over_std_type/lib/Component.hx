package lib;

abstract class Component<Props> extends Element {
	public final props:Props;

	public function new(props:Props, ?tag:String) {
		super(tag);
		this.props = props;
	}

	abstract public function render():String;
}
