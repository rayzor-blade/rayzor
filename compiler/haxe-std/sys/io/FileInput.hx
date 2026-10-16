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

package sys.io;

import sys.io.NativeFileInput;
import sys.io.FileSeek;

class FileInput extends haxe.io.Input {
	var stream:NativeFileInput;
	public function new(stream:NativeFileInput) { this.stream = stream; }
	function live():NativeFileInput {
		if (stream == null) throw "Input is closed";
		return stream;
	}
	public override function readByte():Int return live().readByte();
	public override function readBytes(bytes:haxe.io.Bytes, pos:Int, len:Int):Int
		return live().readBytes(bytes, pos, len);
	public override function readAll(?bufsize:Int):haxe.io.Bytes return live().readAll();
	public override function close():Void {
		if (stream != null) { stream.close(); stream = null; }
	}
	public function seek(position:Int, origin:FileSeek):Void live().seek(position, origin);
	public function tell():Int return live().tell();
	public function eof():Bool return live().eof();
	public override function readLine():String return live().readLine();
}
