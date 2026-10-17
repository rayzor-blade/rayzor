package rayzor.coro;

import haxe.Exception;
import haxe.coro.IContinuation;
import haxe.coro.SuspensionResult;
import haxe.coro.context.Context;

/**
	The runtime fibers under coroutines: a stack per running coroutine body.
**/
@:native("rayzor::coro::FiberNative")
extern class FiberNative {
	@:native("rayzor_fiber_new")
	public static function create(body:Void->Void):FiberNative;

	/** 0 when the fiber yielded, nonzero when its body has finished. **/
	@:native("rayzor_fiber_resume")
	public static function resume(f:FiberNative):Int;

	@:native("rayzor_fiber_yield")
	public static function yieldNow():Void;

	@:native("rayzor_fiber_free")
	public static function free(f:FiberNative):Void;

	@:native("rayzor_fiber_set_data")
	public static function setData(f:FiberNative, data:FiberContinuation<Dynamic, Dynamic>):Void;

	@:native("rayzor_fiber_current_data")
	public static function currentData():FiberContinuation<Dynamic, Dynamic>;
}

/**
	The continuation of a coroutine body running on its own fiber, whose
	result is an `R`. Callees resume it with an `S`, which switches back into
	the fiber; once the body finishes, `completion` is resumed with its
	result or error. Values keep the representation their static types give
	them, as in any generic class.
**/
class FiberContinuation<R, S> implements IContinuation<S> {
	public var context(get, never):Context;

	final completion:IContinuation<R>;
	public var fiber:FiberNative;
	public var value:Null<R> = null;
	public var thrown:Dynamic = null;
	public var threw = false;
	public var resumeResult:Null<S> = null;
	public var resumeError:Exception = null;

	public function new(completion:IContinuation<R>) {
		this.completion = completion;
	}

	function get_context():Context {
		return completion.context;
	}

	public function resume(result:Null<S>, error:Exception):Void {
		resumeResult = result;
		resumeError = error;
		if (step()) {
			if (threw) {
				completion.resume(null, Fiber.toException(thrown));
			} else {
				completion.resume(value, null);
			}
		}
	}

	// The generic fields are reached through methods, where values cross
	// into and out of this erased class the way every caller's do.

	public function setValue(v:Null<R>):Void {
		value = v;
	}

	public function getValue():Null<R> {
		return value;
	}

	public function takeResumeResult():Null<S> {
		final result = resumeResult;
		resumeResult = null;
		return result;
	}

	/** Run the fiber until it suspends (false) or finishes (true). **/
	public function step():Bool {
		if (FiberNative.resume(fiber) == 0) {
			return false;
		}
		FiberNative.free(fiber);
		return true;
	}
}

class Fiber {
	/**
		Start a coroutine body on a new fiber. A body that finishes without
		suspending gives its result or error; one that suspends gives
		`Pending` and later resumes `completion`. With `noThrow`, a throw
		before the first suspension escapes this call instead.
	**/
	public static function start<R>(completion:IContinuation<R>, body:Void->R, noThrow:Bool):SuspensionResult<R> {
		final c = new FiberContinuation<R, Dynamic>(completion);
		c.fiber = FiberNative.create(function() {
			try {
				c.setValue(body());
			} catch (e:Dynamic) {
				c.thrown = e;
				c.threw = true;
			}
		});
		FiberNative.setData(c.fiber, cast c);
		if (!c.step()) {
			return new SuspensionResult(Pending);
		}
		if (c.threw) {
			if (noThrow) {
				throw c.thrown;
			}
			return SuspensionResult.withError(toException(c.thrown));
		}
		return SuspensionResult.withResult(c.getValue());
	}

	/** The continuation of the running coroutine body. **/
	public static function current():IContinuation<Dynamic> {
		return FiberNative.currentData();
	}

	/**
		The value of a coroutine call made from a coroutine body: its result,
		its error thrown, or, while it is pending, whatever the continuation
		is later resumed with.
	**/
	public static function await<S>(r:SuspensionResult<S>):S {
		switch (r.state) {
			case Returned:
				return r.result;
			case Thrown:
				throw r.error;
			case Pending:
		}
		final c:FiberContinuation<Dynamic, S> = cast FiberNative.currentData();
		FiberNative.yieldNow();
		final error = c.resumeError;
		if (error != null) {
			c.resumeError = null;
			throw error;
		}
		return c.takeResumeResult();
	}

	public static function toException(value:Dynamic):Exception {
		return Std.isOfType(value, Exception) ? (value : Exception) : new haxe.ValueException(value);
	}
}
