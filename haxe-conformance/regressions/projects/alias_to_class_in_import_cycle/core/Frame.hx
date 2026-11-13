package core;

import draw3d.Scene;

class Frame {
	public final bounds:{min:math.V, max:math.V};
	public final scene:Null<Scene>;

	function new(bounds:{min:math.V, max:math.V}) {
		this.bounds = bounds;
		this.scene = null;
	}
}
