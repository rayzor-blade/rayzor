// A structure literal passed for a typedef with optional scalar fields stores
// them as the `Null<T>` its readers unbox.
typedef SliderProps = {?value:Float, ?steps:Int, ?on:Bool, ?format:Float->String}

class OptionalScalarsInStructArgument {
	static function render(props:SliderProps):String {
		var format = props.format != null ? props.format : (x:Float) -> Std.string(Math.round(x * 100) / 100);
		return format(props.value) + "/" + props.steps + "/" + props.on;
	}

	static function main() {
		var got = render({value: 0.75, steps: 3, on: true})
			+ " " + render({value: 0.65, steps: 4, on: false, format: v -> Std.string(Math.round(v * 100)) + "%"});
		if (got != "0.75/3/true 65%/4/false")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
