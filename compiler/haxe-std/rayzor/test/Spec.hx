package rayzor.test;

/**
 * Backend tiers a test can target.
 *
 * rayzor runs a program through three backends — the bytecode interpreter, the
 * Cranelift JIT, and the LLVM AOT path. Tier-specific tests are selected with
 * the preprocessor: rayzor defines `interp` / `jit` / `llvm` for the active
 * execution mode, so a test gated by `#if jit ... #end` is only compiled when
 * running under the JIT. These integer constants are for the runtime-filtered
 * `Spec.it(..., tier)` form, which complements the preprocessor approach.
 */
class Tier {
    public static inline var INTERP:Int = 0;
    public static inline var JIT:Int = 1;
    public static inline var LLVM:Int = 2;
    public static inline var ALL:Int = 3;

    public static function name(tier:Int):String {
        if (tier == INTERP) return "interp";
        if (tier == JIT) return "jit";
        if (tier == LLVM) return "llvm";
        return "all";
    }
}

/**
 * Fluent assertions. Obtain one via `Spec.expect(value)` and chain a matcher.
 * A failing matcher `throw`s a `String`; `Spec.run` catches it as a typed
 * `String` and reports a clean failure message. Values compare by their
 * `Std.string` form, giving structural equality for primitives and strings.
 */
class Expect {
    var actual:Dynamic;

    public function new(actual:Dynamic) {
        this.actual = actual;
    }

    function fail(msg:String):Void {
        throw msg;
    }

    public function toEqual(expected:Dynamic):Void {
        if (Std.string(actual) != Std.string(expected)) {
            fail("expected " + Std.string(expected) + " but got " + Std.string(actual));
        }
    }

    public function notToEqual(expected:Dynamic):Void {
        if (Std.string(actual) == Std.string(expected)) {
            fail("expected value to differ from " + Std.string(expected));
        }
    }

    public function toBeTrue():Void {
        if (Std.string(actual) != "true") {
            fail("expected true but got " + Std.string(actual));
        }
    }

    public function toBeFalse():Void {
        if (Std.string(actual) != "false") {
            fail("expected false but got " + Std.string(actual));
        }
    }

    public function toBeNull():Void {
        if (actual != null) {
            fail("expected null but got " + Std.string(actual));
        }
    }

    public function toBeNotNull():Void {
        if (actual == null) {
            fail("expected a non-null value");
        }
    }

    public function toBeGreaterThan(bound:Float):Void {
        var a:Float = actual;
        if (!(a > bound)) {
            fail("expected " + Std.string(actual) + " to be greater than " + Std.string(bound));
        }
    }

    public function toBeLessThan(bound:Float):Void {
        var a:Float = actual;
        if (!(a < bound)) {
            fail("expected " + Std.string(actual) + " to be less than " + Std.string(bound));
        }
    }
}

/**
 * String-typed assertions, from `Spec.expectStr(s)`.
 *
 * Kept separate from `Expect` (which takes `Dynamic`) on purpose: a String
 * passed to a `Dynamic` parameter across a module boundary cannot be boxed
 * safely (the boxing breaks type-specialized sinks like `trace`/`Sys.println`),
 * so `Expect.toEqual` on a String value would crash. `StrExpect` carries the
 * String with its concrete type the whole way, which works cross-module.
 */
class StrExpect {
    var actual:String;

    public function new(actual:String) {
        this.actual = actual;
    }

    function fail(msg:String):Void {
        throw msg;
    }

    /** Pass when `actual` equals `expected`. */
    public function toEqual(expected:String):Void {
        if (actual != expected) {
            fail("expected \"" + expected + "\" but got \"" + actual + "\"");
        }
    }

    /** Pass when `actual` differs from `expected`. */
    public function notToEqual(expected:String):Void {
        if (actual == expected) {
            fail("expected string to differ from \"" + expected + "\"");
        }
    }

    /** Pass when `actual` contains `needle` as a substring. */
    public function toContain(needle:String):Void {
        if (actual.indexOf(needle) < 0) {
            fail("expected \"" + actual + "\" to contain \"" + needle + "\"");
        }
    }

    /** Pass when `actual` is the empty string. */
    public function toBeEmpty():Void {
        if (actual != "") {
            fail("expected empty string but got \"" + actual + "\"");
        }
    }
}

/** One registered test case. */
class TestCase {
    public var suite:String;
    public var name:String;
    public var body:Void->Void;
    public var tier:Int;

    public function new(suite:String, name:String, body:Void->Void, tier:Int) {
        this.suite = suite;
        this.name = name;
        this.body = body;
        this.tier = tier;
    }
}

/** Outcome of running a single test case (0 = pass, 1 = fail, 2 = skipped). */
class TestResult {
    public var suite:String;
    public var name:String;
    public var status:Int;
    public var message:String;

    public function new(suite:String, name:String, status:Int, message:String) {
        this.suite = suite;
        this.name = name;
        this.status = status;
        this.message = message;
    }
}

/**
 * A Mocha-style BDD test framework for rayzor.
 *
 * ```haxe
 * import rayzor.test.Spec;
 *
 * class MyTests {
 *     static function main() {
 *         Spec.describe("Math", () -> {
 *             Spec.it("adds", () -> Spec.expect(2 + 2).toEqual(4));
 *         });
 *         Sys.exit(Spec.run());
 *     }
 * }
 * ```
 *
 * Register with `describe`/`it`, assert with `expect`, run with `run` (prints a
 * report and returns the failure count, so `Sys.exit(Spec.run())` works).
 *
 * Tiers: gate tier-specific tests with the preprocessor — rayzor defines
 * `interp`/`jit`/`llvm` for the active backend, e.g.
 * `#if jit Spec.it("jit path", ...); #end`. The runtime form
 * `Spec.it(name, body, Tier.JIT)` plus `Spec.activeTier` is also available.
 *
 * Parallelism: `runCase` performs no shared mutation and returns a `TestResult`,
 * so cases can be fanned out across a worker pool; `runParallel` is the opt-in
 * entry point (currently delegates to `run`).
 */
class Spec {
    static var tests:Array<TestCase> = [];
    static var curSuite:String = "";
    static var beforeEachFn:Void->Void = null;
    static var afterEachFn:Void->Void = null;

    /** The backend this run targets; tests with a different specific tier are skipped. */
    public static var activeTier:Int = Tier.ALL;

    /** Define a group of related tests. Groups may nest. */
    public static function describe(name:String, body:Void->Void):Void {
        var prevSuite = curSuite;
        var prevBefore = beforeEachFn;
        var prevAfter = afterEachFn;
        curSuite = (prevSuite == null || prevSuite == "") ? name : prevSuite + " > " + name;
        body();
        curSuite = prevSuite;
        beforeEachFn = prevBefore;
        afterEachFn = prevAfter;
    }

    /** Register a test case. Optional `tier` restricts it to a backend (default `Tier.ALL`). */
    public static function it(name:String, body:Void->Void, ?tier:Int):Void {
        tests.push(new TestCase(curSuite, name, body, tier == null ? Tier.ALL : tier));
    }

    /** Run `fn` before each test in the current `describe` scope. */
    public static function beforeEach(fn:Void->Void):Void {
        beforeEachFn = fn;
    }

    /** Run `fn` after each test in the current `describe` scope. */
    public static function afterEach(fn:Void->Void):Void {
        afterEachFn = fn;
    }

    /** Start a fluent assertion on `value` (Int/Float/Bool/structural). For
        String values use `expectStr` — see `StrExpect`. */
    public static function expect(value:Dynamic):Expect {
        return new Expect(value);
    }

    /** Start a String-typed assertion (use this instead of `expect` for Strings). */
    public static function expectStr(value:String):StrExpect {
        return new StrExpect(value);
    }

    static function tierMatches(tier:Int):Bool {
        return activeTier == Tier.ALL || tier == Tier.ALL || activeTier == tier;
    }

    static function runCase(t:TestCase):TestResult {
        if (!tierMatches(t.tier)) {
            return new TestResult(t.suite, t.name, 2, "tier " + Tier.name(t.tier));
        }
        var status = 0;
        var message = "";
        try {
            if (beforeEachFn != null) beforeEachFn();
            t.body();
        } catch (msg:String) {
            status = 1;
            message = msg;
        } catch (e:Dynamic) {
            status = 1;
            message = "error: " + Std.string(e);
        }
        if (afterEachFn != null) {
            try { afterEachFn(); } catch (e:Dynamic) {}
        }
        return new TestResult(t.suite, t.name, status, message);
    }

    static function report(results:Array<TestResult>):Int {
        var pass = 0;
        var fail = 0;
        var skip = 0;
        for (r in results) {
            if (r.status == 0) {
                pass++;
                trace("  ok   " + r.suite + " > " + r.name);
            } else if (r.status == 2) {
                skip++;
                trace("  skip " + r.suite + " > " + r.name + " (" + r.message + ")");
            } else {
                fail++;
                trace("  FAIL " + r.suite + " > " + r.name + " -- " + r.message);
            }
        }
        trace("");
        var summary = pass + " passing, " + fail + " failing";
        if (skip > 0) summary += ", " + skip + " skipped";
        trace(summary);
        return fail;
    }

    /**
     * Run all registered tests sequentially, print a report, and return the
     * number of failures. Clears the registry so a process can run several
     * independent suites.
     */
    public static function run():Int {
        var results:Array<TestResult> = [];
        for (t in tests) {
            results.push(runCase(t));
        }
        var failures = report(results);
        tests = [];
        return failures;
    }

    /** Threaded run (opt-in); delegates to `run` until wired to a worker pool. */
    public static function runParallel():Int {
        return run();
    }
}
