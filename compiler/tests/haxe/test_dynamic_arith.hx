class Main {
    static function main() {
        var x:Dynamic = 10;
        var y:Dynamic = 3;

        // Basic arithmetic with unboxing for trace
        var sum:Dynamic = x + y;
        var sumInt:Int = sum;
        trace(sumInt);     // 13

        var diff:Dynamic = x - y;
        var diffInt:Int = diff;
        trace(diffInt);    // 7

        var prod:Dynamic = x * y;
        var prodInt:Int = prod;
        trace(prodInt);    // 30

        var mod_:Dynamic = x % y;
        var modInt:Int = mod_;
        trace(modInt);     // 1

        // Division (returns float)
        var a:Dynamic = 20;
        var b:Dynamic = 4;
        var quot:Dynamic = a / b;
        var quotInt:Int = quot;
        trace(quotInt);    // 5

        // Comparison ops
        var eq:Bool = x == y;
        trace(eq);         // false

        var lt:Bool = y < x;
        trace(lt);         // true

        var gt:Bool = x > y;
        trace(gt);         // true

        trace("done");
    }
}
