/*
 * Copyright (C)2005-2019 Haxe Foundation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

/**
	A String buffer is an efficient way to build a big string by appending small
	elements together.

	Rayzor: backed by a mutable `rayzor.Bytes` buffer grown in place (doubling),
	NOT by String concatenation. Strings are immutable and rayzor has no GC, so
	the upstream `b += x` implementation leaked one intermediate String per
	append — O(n) leaked allocations per built string, the dominant heap growth
	in string-heavy loops (tokenizers, response serialization). Here the only
	allocations are the buffer (old one freed on grow) and the final
	`toString()` copy. `length` reports UTF-8 BYTES — identical to upstream on
	rayzor, where `String.length` is also byte-based.
**/
class StringBuf {
	var b:rayzor.Bytes;
	var len:Int;

	/**
		The length of `this` StringBuf in characters.
	**/
	public var length(get, never):Int;

	/**
		Creates a new StringBuf instance.

		This may involve initialization of the internal buffer.
	**/
	public function new() {
		b = rayzor.Bytes.alloc(16);
		len = 0;
	}

	inline function get_length():Int {
		return len;
	}

	function grow(need:Int):Void {
		var cap = b.length;
		if (len + need <= cap) return;
		var ncap = cap << 1;
		while (ncap < len + need) ncap <<= 1;
		var nb = rayzor.Bytes.alloc(ncap);
		b.blit(0, nb, 0, len);
		b.free();
		b = nb;
	}

	/**
		Appends the representation of `x` to `this` StringBuf.

		The exact representation of `x` may vary per platform. To get more
		consistent behavior, this function should be called with
		Std.string(x).

		If `x` is null, the String "null" is appended.
	**/
	public function add<T>(x:T):Void {
		var s = Std.string(x);
		var n = s.length;
		if (n == 0) return;
		grow(n);
		// Bulk byte copy through a transient Bytes copy — avoids per-char
		// charCodeAt (Null<Int> round-trips) and String concatenation.
		var sb = rayzor.Bytes.ofString(s);
		sb.blit(0, b, len, n);
		sb.free();
		len += n;
	}

	/**
		Appends the character identified by `c` to `this` StringBuf.

		If `c` is negative or has another invalid value, the result is
		unspecified.
	**/
	public function addChar(c:Int):Void {
		// UTF-8 encode in place; mirrors the byte encoding String itself uses.
		if (c < 0x80) {
			grow(1);
			b.set(len, c);
			len += 1;
		} else if (c < 0x800) {
			grow(2);
			b.set(len, 0xC0 | (c >> 6));
			b.set(len + 1, 0x80 | (c & 0x3F));
			len += 2;
		} else if (c < 0x10000) {
			grow(3);
			b.set(len, 0xE0 | (c >> 12));
			b.set(len + 1, 0x80 | ((c >> 6) & 0x3F));
			b.set(len + 2, 0x80 | (c & 0x3F));
			len += 3;
		} else {
			grow(4);
			b.set(len, 0xF0 | (c >> 18));
			b.set(len + 1, 0x80 | ((c >> 12) & 0x3F));
			b.set(len + 2, 0x80 | ((c >> 6) & 0x3F));
			b.set(len + 3, 0x80 | (c & 0x3F));
			len += 4;
		}
	}

	/**
		Appends a substring of `s` to `this` StringBuf.

		This function expects `pos` and `len` to describe a valid substring of
		`s`, or else the result is unspecified. To get more robust behavior,
		`this.add(s.substr(pos,len))` can be used instead.

		If `s` or `pos` are null, the result is unspecified.

		If `len` is omitted or null, the substring ranges from `pos` to the end
		of `s`.
	**/
	public function addSub(s:String, pos:Int, ?len:Int):Void {
		var sb = rayzor.Bytes.ofString(s);
		var n:Int = (len == null) ? sb.length - pos : len;
		if (n <= 0) {
			sb.free();
			return;
		}
		grow(n);
		sb.blit(pos, b, this.len, n);
		sb.free();
		this.len += n;
	}

	/**
		Returns the content of `this` StringBuf as String.

		The buffer is not emptied by this operation.
	**/
	public function toString():String {
		if (len == 0) return "";
		var s = b.sub(0, len);
		var r = s.toString();
		s.free();
		return r;
	}
}
