// Regression test for the rayzor.test.Spec framework (imported cross-module).
// Exercises describe/it/expect/run/beforeEach + cross-module primitive->Dynamic
// boxing (Spec.expect(Int) static call + .toEqual(Int) instance call). Asserts
// the framework's own behavior and Sys.exit(1)s on misbehavior.
import rayzor.test.Spec;

class Main {
    static function main() {
        Spec.describe("Math", () -> {
            Spec.it("adds", () -> Spec.expect(2 + 2).toEqual(4));
            Spec.it("greater", () -> Spec.expect(10).toBeGreaterThan(3));
            Spec.it("bool true", () -> Spec.expect(1 < 2).toBeTrue());
            Spec.it("bool false", () -> Spec.expect(2 < 1).toBeFalse());
            Spec.describe("nested", () -> {
                Spec.it("equal floats", () -> Spec.expect(1.5).toEqual(1.5));
            });
        });
        Spec.describe("Strings", () -> {
            Spec.it("equal", () -> Spec.expectStr("a" + "b").toEqual("ab"));
            Spec.it("contains", () -> Spec.expectStr("hello world").toContain("world"));
            Spec.it("differ", () -> Spec.expectStr("x").notToEqual("y"));
        });
        var passing = Spec.run();
        if (passing != 0) { trace("SELFTEST FAIL: all-pass suite reported " + passing); Sys.exit(1); }

        // Failures must be detected (and not crash).
        Spec.describe("Detect", () -> {
            Spec.it("mismatch", () -> Spec.expect(2 + 2).toEqual(5));
        });
        var failing = Spec.run();
        if (failing != 1) { trace("SELFTEST FAIL: expected 1 failure, got " + failing); Sys.exit(1); }

        trace("test_rayzor_spec OK");
    }
}
