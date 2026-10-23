private enum abstract Choice(String) to String {
    var Default = "default";
    var Other = "other";
}

class NullableEnumConditional {
    static function choose(value:Null<Choice>):String {
        return (value == null ? Choice.Default : value : String);
    }

    static function reverse(value:Null<Choice>):String {
        return (value != null ? value : Choice.Default : String);
    }

    static function unwrap(value:Null<Choice>):Choice {
        return value;
    }

    static function main() {
        if (choose(null) != "default") throw "null default";
        if (choose(Choice.Other) != "other") throw "nonnull choice";
        if (reverse(null) != "default") throw "reverse null default";
        if (reverse(Choice.Other) != "other") throw "reverse nonnull choice";
        if (unwrap(Choice.Other) != Choice.Other) throw "nullable abstract unwrap";
        Sys.println("CONFORMANCE_OK");
    }
}
