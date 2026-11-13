package kit;

import draw3d.Pass;

class Rig implements Pass {
	public var got:Float = 0;

	public function new() {}

	function fit(min:math.V, max:math.V):Float
		return min.x + max.x * 10;

	public function prepare(frame:Frame):Void
		got = fit(frame.bounds.min, frame.bounds.max);
}
