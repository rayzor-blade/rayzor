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

import sys.io.FileInput;
import sys.io.FileOutput;

class Process {
	var handle:NativeProcess;

	function live():NativeProcess {
		if (handle == null) throw "Process is closed";
		return handle;
	}

	/**
		Standard output. The output stream where a process writes its output data.
	**/
	public var stdout(default, null):haxe.io.Input;

	/**
		Standard error. The output stream to output error messages or diagnostics.
	**/
	public var stderr(default, null):haxe.io.Input;

	/**
		Standard input. The stream data going into a process.
	**/
	public var stdin(default, null):haxe.io.Output;

	/**
		Construct a `Process` object, which run the given command immediately.

		Command arguments can be passed in two ways: 1. using `args`, 2. appending to `cmd` and leaving `args` as `null`.

		 1. When using `args` to pass command arguments, each argument will be automatically quoted, and shell meta-characters will be escaped if needed.
		`cmd` should be an executable name that can be located in the `PATH` environment variable, or a path to an executable.

		 2. When `args` is not given or is `null`, command arguments can be appended to `cmd`. No automatic quoting/escaping will be performed. `cmd` should be formatted exactly as it would be when typed at the command line.
		It can run executables, as well as shell commands that are not executables (e.g. on Windows: `dir`, `cd`, `echo` etc).

		`detached` allows the created process to be standalone. You cannot communicate with it but you can look at its exit code. Not supported on php.

		`close()` should be called when the `Process` is no longer used.
	**/
	public function new(cmd:String, ?args:Array<String>, detached:Bool = false):Void {
		handle = new NativeProcess(cmd, args, detached);
		if (!detached) {
			stdout = new ProcessInput(handle.takeStdout());
			stderr = new ProcessInput(handle.takeStderr());
			stdin = new ProcessOutput(handle.takeStdin());
		}
	}

	/**
		Return the process ID.
	**/
	public function getPid():Int return live().getPid();

	/**
		Query the exit code of the process.
		If `block` is true or not specified, it will block until the process terminates.
		If `block` is false, it will return either the process exit code if it's already terminated or null if it's still running.
		If the process has already exited, return the exit code immediately.
	**/
	public function exitCode(block:Bool = true):Null<Int> {
		var process = live();
		if (!block && !process.hasExited()) return null;
		return process.waitForExit();
	}

	/**
		Close the process handle and release the associated resources.
		All `Process` fields should not be used after `close()` is called.
	**/
	public function close():Void {
		if (handle == null) return;
		if (stdin != null) stdin.close();
		if (stdout != null) stdout.close();
		if (stderr != null) stderr.close();
		handle.close();
		handle = null;
	}

	/**
		Kill the process.
	**/
	public function kill():Void live().kill();
}

private class ProcessInput extends haxe.io.Input {
	var stream:FileInput;
	public function new(stream:FileInput) { this.stream = stream; }
	function live():FileInput {
		if (stream == null) throw "Process input is closed";
		return stream;
	}
	public override function readByte():Int return live().readByte();
	public override function readBytes(bytes:haxe.io.Bytes, pos:Int, len:Int):Int
		return live().readBytes(bytes, pos, len);
	public override function readAll(?bufsize:Int):haxe.io.Bytes return live().readAll(bufsize);
	public override function close():Void {
		if (stream != null) { stream.close(); stream = null; }
	}
}

private class ProcessOutput extends haxe.io.Output {
	var stream:FileOutput;
	public function new(stream:FileOutput) { this.stream = stream; }
	function live():FileOutput {
		if (stream == null) throw "Process output is closed";
		return stream;
	}
	public override function writeByte(value:Int):Void live().writeByte(value);
	public override function writeBytes(bytes:haxe.io.Bytes, pos:Int, len:Int):Int
		return live().writeBytes(bytes, pos, len);
	public override function flush():Void live().flush();
	public override function close():Void {
		if (stream != null) { stream.close(); stream = null; }
	}
}
