package b;

abstract Buf2(haxe.io.Bytes) from haxe.io.Bytes to haxe.io.Bytes {
	public inline function new(size:Int)
		this = haxe.io.Bytes.alloc(size);

	public function len():Int
		return this.length;
}
