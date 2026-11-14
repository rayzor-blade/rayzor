package p;
using p.ItemTools;

class Focus {
	static final within = new haxe.ds.ObjectMap<Tree, Array<Item>>();

	public static function set(target:Item):Void
		updateWithin(target.tree, target);

	static function updateWithin(tree:Tree, target:Null<Item>):Void {
		var next = [];
		if (target != null)
			next.push(target);
		var before = within.get(tree);
		if (before != null)
			for (i in before)
				if (next.indexOf(i) < 0)
					i.mark(false);
		for (i in next)
			i.mark(true);
		if (next.length == 0) within.remove(tree); else within.set(tree, next);
	}
}
