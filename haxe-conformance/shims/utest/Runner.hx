package utest;

/** Minimal stand-in for utest's Runner.

    A case that is also a program entry point (TestArguments) builds a Runner
    and a Report in its own main. The harness drives the test methods itself,
    so these only need upstream's signatures for that main to compile. */
class Runner {
    public function new() {}

    public function addCase(testCase:Dynamic, ?pattern:EReg, ?dependencies:Array<String>):Void {}

    public function run():Void {}
}
