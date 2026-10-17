package rayzor.coro;

import haxe.coro.IContinuation;

/**
	The continuation parameter of an expanded coroutine. The compiler gives
	every coroutine this first parameter type, which is how it recognises a
	coroutine callee.
**/
abstract CoroCont<T>(IContinuation<T>) from IContinuation<T> to IContinuation<T> {}
