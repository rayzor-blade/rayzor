private enum Literal<T> {
    Integer(text:String):Literal<Int>;
    Number(text:String):Literal<Float>;
    Text(value:String):Literal<String>;
}

private enum Operator<A, R> {
    Plus:Operator<Float, Float>;
    Equal:Operator<A, Bool>;
}

private enum Expression<R> {
    Value(value:Literal<R>);
    Apply<T>(op:Operator<T, R>, left:Expression<T>, right:Expression<T>);
}

class GadtExistentialEvaluation {
    static function literal<T>(value:Literal<T>):T {
        return switch value {
            case Integer(text): Std.parseInt(text);
            case Number(text): Std.parseFloat(text);
            case Text(text): text;
        };
    }

    static function operation<R, A>(op:Operator<A, R>, left:Expression<A>, right:Expression<A>):R {
        return switch op {
            case Plus: evaluate(left) + evaluate(right);
            case Equal: evaluate(left) == evaluate(right);
        };
    }

    static function evaluate<T>(expression:Expression<T>):T {
        return switch expression {
            case Value(value): literal(value);
            case Apply(op, left, right): operation(op, left, right);
        };
    }

    static function allocatedText():String {
        var buffer = new StringBuf();
        buffer.add("equal");
        buffer.add(" text");
        return buffer.toString();
    }

    static function main() {
        var integer = Value(Integer("21"));
        if (evaluate(integer) != 21) throw "integer result";
        if (!evaluate(Apply(Equal, integer, Value(Integer("21"))))) throw "equal integers";
        if (evaluate(Apply(Equal, integer, Value(Integer("22"))))) throw "unequal integers";
        var a = Value(Number("12.5"));
        var b = Value(Number("7.5"));
        if (evaluate(Apply(Equal, a, b))) throw "unequal floats";
        if (!evaluate(Apply(Equal, a, Value(Number("12.5"))))) throw "equal floats";
        if (!evaluate(Apply(Equal, Value(Number("0")), Value(Number("-0"))))) throw "signed zero";
        if (evaluate(Apply(Plus, a, b)) != 20.0) throw "addition";
        if (!evaluate(Apply(Equal, Apply(Plus, a, b), Value(Number("20"))))) throw "nested addition";
        if (!evaluate(Apply(Equal, Value(Text(allocatedText())), Value(Text(allocatedText()))))) throw "string contents";
        if (evaluate(Apply(Equal, Value(Text("first")), Value(Text("second"))))) throw "unequal strings";
        var yes = Apply(Equal, a, a);
        var no = Apply(Equal, a, b);
        if (evaluate(Apply(Equal, yes, no))) throw "nested unequal booleans";
        if (!evaluate(Apply(Equal, yes, yes))) throw "nested equal booleans";
        trace("CONFORMANCE_OK");
    }
}
