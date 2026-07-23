package rayzor;

/**
 * Zero-copy view into an immutable String — the shared-backing (Swift
 * `Substring`) model. A `Slice` is `{backing, offset, length}`: it keeps a
 * reference to the source String, so the buffer stays alive by ordinary object
 * reachability — no new refcount, no lifetime syntax. Narrowing (`slice`,
 * `tail`, `sub`, `splitByte`, `trimAsciiSpace`) returns a NEW view over the
 * SAME buffer with an adjusted offset — no string bytes are copied. The one
 * materialising op is `toString()`, which copies the window into a fresh
 * String (the deliberate escape hatch).
 *
 * Byte-oriented: Rayzor strings are UTF-8 byte arrays, so `codeAt(i)` is the
 * i-th BYTE of the window (like `String.charCodeAt`). Returns `Int` (`-1` out
 * of range) — never `Null<Int>` — so it composes cleanly across modules.
 *
 * Immutable backing ⇒ trivially shareable: `@:derive([Send, Sync])`. A large
 * text can be sharded across worker threads as slices with zero copying.
 *
 * The API takes explicit bounds (no optional/default args) — `slice(a, b)` for
 * a range, `tail(a)` for "to the end", `sub(a, n)` for a length — mirroring the
 * shape of `String.substring`.
 *
 * GOTCHA: a small slice pins its whole backing buffer alive. Call `toString()`
 * to copy the window out and release the source.
 *
 * ```haxe
 * var v = Slice.of("hello,world");
 * var parts = v.splitByte(",".code);        // [ "hello", "world" ] as views
 * var head = v.slice(0, 5);                  // "hello" — zero-copy
 * if (head.equalsString("hello")) { ... }    // window compare, no alloc
 * var s = head.toString();                   // materialise once
 * ```
 */
@:derive([Send, Sync])
class Slice {
	/** Window length in bytes (read-only by convention). */
	public var length:Int;

	// backing buffer + start byte within it — private; only this class reads
	// them, so no cross-module field access is ever needed.
	var backing:String;
	var offset:Int;

	public function new(backing:String, offset:Int, length:Int) {
		this.backing = backing;
		this.offset = offset;
		this.length = length;
	}

	/** View an entire String, zero-copy. */
	public static function of(s:String):Slice {
		return new Slice(s, 0, s.length);
	}

	/** True if the window is empty. */
	public inline function isEmpty():Bool {
		return length <= 0;
	}

	/** i-th byte of the window as `Int`, or `-1` if out of range. */
	public function codeAt(i:Int):Int {
		if (i < 0 || i >= length)
			return -1;
		return backing.charCodeAt(offset + i);
	}

	/** Zero-copy sub-view `[start, end)`, clamped to the window. */
	public function slice(start:Int, end:Int):Slice {
		if (start < 0)
			start = 0;
		if (start > length)
			start = length;
		if (end > length)
			end = length;
		if (end < start)
			end = start;
		return new Slice(backing, offset + start, end - start);
	}

	/** Zero-copy sub-view from `start` to the end of the window. */
	public function tail(start:Int):Slice {
		return slice(start, length);
	}

	/** Zero-copy sub-view of `len` bytes starting at `start`. */
	public function sub(start:Int, len:Int):Slice {
		return slice(start, start + len);
	}

	/** First index of byte `code` at/after `from`, or `-1`. */
	public function indexOfCodeFrom(code:Int, from:Int):Int {
		var i = from < 0 ? 0 : from;
		while (i < length) {
			if (codeAt(i) == code)
				return i;
			i++;
		}
		return -1;
	}

	/** First index of byte `code` in the window, or `-1`. */
	public function indexOfCode(code:Int):Int {
		return indexOfCodeFrom(code, 0);
	}

	/** True if the window equals `s` byte-for-byte. */
	public function equalsString(s:String):Bool {
		if (s.length != length)
			return false;
		var i = 0;
		while (i < length) {
			if (codeAt(i) != s.charCodeAt(i))
				return false;
			i++;
		}
		return true;
	}

	/** True if the window starts with `prefix` byte-for-byte. */
	public function startsWith(prefix:String):Bool {
		if (prefix.length > length)
			return false;
		var i = 0;
		while (i < prefix.length) {
			if (codeAt(i) != prefix.charCodeAt(i))
				return false;
			i++;
		}
		return true;
	}

	/**
	 * Split the window on single-byte separator `sep` into zero-copy
	 * sub-views. Empty fields are preserved (like `String.split`).
	 */
	public function splitByte(sep:Int):Array<Slice> {
		var out:Array<Slice> = [];
		var start = 0;
		var i = 0;
		while (i < length) {
			if (codeAt(i) == sep) {
				out.push(new Slice(backing, offset + start, i - start));
				start = i + 1;
			}
			i++;
		}
		out.push(new Slice(backing, offset + start, length - start));
		return out;
	}

	/** Zero-copy trim of leading/trailing ASCII whitespace (space/tab/CR/LF). */
	public function trimAsciiSpace():Slice {
		var a = 0;
		var b = length;
		while (a < b && isSpaceByte(codeAt(a)))
			a++;
		while (b > a && isSpaceByte(codeAt(b - 1)))
			b--;
		return new Slice(backing, offset + a, b - a);
	}

	inline function isSpaceByte(c:Int):Bool {
		return c == 32 || c == 9 || c == 10 || c == 13;
	}

	/** Materialise the window into a fresh String (allocates — the escape hatch). */
	public function toString():String {
		return backing.substring(offset, offset + length);
	}
}
