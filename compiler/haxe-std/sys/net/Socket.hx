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

package sys.net;

/**
	A TCP socket class : allow you to both connect to a given server and exchange messages or start your own server and wait for connections.
**/
@:native("sys::net::Socket")
extern class Socket {
	/**
		A custom value that can be associated with the socket. Can be used to retrieve your custom infos after a `select`.
	***/
	var custom:Dynamic;

	/**
		Creates a new unconnected socket.
	**/
	@:native("new")
	function new():Void;

	/**
		Closes the socket : make sure to properly close all your sockets or you will crash when you run out of file descriptors.
	**/
	@:native("close")
	function close():Void;

	/**
		Read the whole data available on the socket.
	**/
	@:native("read")
	function read():String;

	/**
		Write the whole data to the socket output.
	**/
	@:native("write")
	function write(content:String):Void;

	/**
		Connect to the given server host/port. Throw an exception in case we couldn't successfully connect.
	**/
	@:native("connect")
	function connect(host:Host, port:Int):Void;

	/**
		Allow the socket to listen for incoming questions. The parameter tells how many pending connections we can have until they get refused. Use `accept()` to accept incoming connections.
	**/
	@:native("listen")
	function listen(connections:Int):Void;

	/**
		Shutdown the socket, either for reading or writing.
	**/
	@:native("shutdown")
	function shutdown(read:Bool, write:Bool):Void;

	/**
		Bind the socket to the given host/port so it can afterwards listen for connections there.
	**/
	@:native("bind")
	function bind(host:Host, port:Int):Void;

	/**
		Accept a new connected client. This will return a connected socket on which you can read/write some data.
	**/
	@:native("accept")
	function accept():Socket;

	/**
		Gives a timeout (in seconds) after which blocking socket operations (such as reading and writing) will abort and throw an exception.
	**/
	@:native("setTimeout")
	function setTimeout(timeout:Float):Void;

	/**
		Block until some data is available for read on the socket.
	**/
	@:native("waitForRead")
	function waitForRead():Void;

	/**
		Change the blocking mode of the socket. A blocking socket is the default behavior. A non-blocking socket will abort blocking operations immediately by throwing a haxe.io.Error.Blocked value.
	**/
	@:native("setBlocking")
	function setBlocking(b:Bool):Void;

	/**
		Allows the socket to immediately send the data when written to its output : this will cause less ping but might increase the number of packets / data size, especially when doing a lot of small writes.
	**/
	@:native("setFastSend")
	function setFastSend(b:Bool):Void;
}
