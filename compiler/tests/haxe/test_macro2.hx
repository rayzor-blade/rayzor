class MacroTools {
    macro static function compileTimeAdd(a:haxe.macro.Expr, b:haxe.macro.Expr):haxe.macro.Expr {
        return a + b;
    }

    macro static function doubled(x:haxe.macro.Expr):haxe.macro.Expr {
        return x * 2;
    }

    macro static function makeGreeting(name:haxe.macro.Expr):haxe.macro.Expr {
        return "Hello, " + name + "!";
    }

    macro static function generateSquares(n:haxe.macro.Expr):haxe.macro.Expr {
        var result = [];
        var i = 0;
        while (i < n) {
            result.push(i * i);
            i = i + 1;
        }
        return result;
    }

    macro static function fibonacci(n:haxe.macro.Expr):haxe.macro.Expr {
        if (n <= 1) return n;
        var a = 0;
        var b = 1;
        var i = 2;
        while (i <= n) {
            var temp = a + b;
            a = b;
            b = temp;
            i = i + 1;
        }
        return b;
    }
}

class Main {
    static function main() {
        // Simple arithmetic
        trace(MacroTools.compileTimeAdd(100, 200));
        trace(MacroTools.doubled(21));

        // String macro
        trace(MacroTools.makeGreeting("World"));

        // Compile-time fibonacci
        trace(MacroTools.fibonacci(10));

        // Array generation
        var squares = MacroTools.generateSquares(5);
        trace(squares[0]);
        trace(squares[4]);

        // Conditional compilation
        #if rayzor
        trace("rayzor");
        #end

        trace("done");
    }
}
