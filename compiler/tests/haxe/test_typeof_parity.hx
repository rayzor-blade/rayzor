enum Color {
    Red;
    Blue;
}

class Main {
    static function plusOne(x:Int):Int {
        return x + 1;
    }

    static function main() {
        var fn = plusOne;
        switch (Type.typeof(fn)) {
            case TFunction:
                trace("fn-ok");
            default:
                trace("fn-bad");
        }

        switch (Type.typeof(42)) {
            case TInt:
                trace("int-ok");
            default:
                trace("int-bad");
        }

        switch (Type.typeof("x")) {
            case TClass(_):
                trace("string-class-ok");
            default:
                trace("string-class-bad");
        }

        switch (Type.typeof(Color.Red)) {
            case TEnum(_):
                trace("enum-ok");
            default:
                trace("enum-bad");
        }
    }
}
