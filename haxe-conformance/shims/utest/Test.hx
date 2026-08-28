package utest;

/** Minimal stand-in for utest's Test base class.

    The sys and threads suites are written against utest rather than
    unit.Test: their cases `extends utest.Test` and assert through
    utest.Assert, which is already shimmed here. Without a base class they
    fail to compile and the whole suite reads as unsupported rather than
    unmeasured.

    utest drives setup/teardown itself and can run a case asynchronously. The
    harness calls the test methods directly instead, so only the hooks a case
    may declare need to exist -- an override with no caller is harmless, and
    inventing a scheduler here would measure the shim rather than rayzor. */
class Test {
    public function new() {}

    /** Called by utest before each test method. Present so a case that
        overrides it compiles; the harness invokes it explicitly. */
    public function setup():Void {}

    /** Called by utest after each test method, for the same reason. */
    public function teardown():Void {}
}
