package lib;

import lib.Highlighter;
import lib.Shapes;

class Words implements Highlighter implements Measurer {
	public function new() {}

	public function highlight(line:String, state:String):Highlight {
		var tokens:Array<Token> = [];
		var start = 0;
		for (i in 0...line.length + 1)
			if (i == line.length || line.charAt(i) == " ") {
				if (i > start)
					tokens.push({from: start, to: i, kind: line.charAt(start) == line.charAt(start).toUpperCase() ? "type" : "plain"});
				start = i + 1;
			}
		return {tokens: tokens, state: state + "+"};
	}

	public function measure(line:String):Extent
		return {width: line.length, parts: line.split(" ")};
}
