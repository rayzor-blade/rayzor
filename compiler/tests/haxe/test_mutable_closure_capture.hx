// Regression: Haxe captures mutable local bindings by reference. The closure
// environment used to copy the current SSA value, so writes in a callback were
// invisible to its parent and writes in the parent were invisible to a closure.
class test_mutable_closure_capture {
    static var failures = 0;

    static function check(actual:Int, expected:Int, label:String) {
        if (actual != expected) {
            failures++;
            Sys.println("FAIL " + label + ": got " + actual + ", expected " + expected);
        }
    }

    static function main() {
        var parentToClosure = 1;
        var readParent = function() return parentToClosure;
        parentToClosure = 2;
        check(readParent(), 2, "parent write visible to closure");

        var closureToParent = 0;
        var writeParent = function(value:Int) closureToParent = value;
        writeParent(7);
        check(closureToParent, 7, "closure write visible to parent");

        var shared = 0;
        var addOne = function() shared++;
        var addTen = function() shared += 10;
        addOne();
        addTen();
        check(shared, 11, "two closures share one binding");

        var conditionalCapture = 3;
        var conditional = if (true) {
            function() return conditionalCapture;
        } else {
            function() return 0;
        };
        conditionalCapture = 4;
        check(conditional(), 4, "conditional closure cell dominates use");

        var nestedCapture = 0;
        var makeNested = function() {
            return function() {
                nestedCapture++;
                return nestedCapture;
            };
        };
        var nested = makeNested();
        check(nested(), 1, "nested closure first write");
        check(nestedCapture, 1, "nested write visible to root");

        if (failures == 0) {
            Sys.println("PASS mutable closure captures");
        } else {
            Sys.exit(1);
        }
    }
}
