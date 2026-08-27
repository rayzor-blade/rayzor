class Main {
    static function main() {
        check(First == 0, "bare First");
        check(Number.Second == 1, "qualified Second");
        check(Fifth == 5, "explicit Fifth");
        check(Number.Sixth == 6, "continued Sixth");
        check(Label.One == "One", "implicit string One");
        check(Tenth == 10, "abstract underlying Tenth");
        trace("enum abstract implicit constants: ok");
    }

    static function check(value:Bool, message:String) {
        if (!value) {
            throw message;
        }
    }
}

private enum abstract Number(Int) to Int {
    var First;
    var Second;
    var Fifth = 5;
    var Sixth;
}

private enum abstract Label(String) to String {
    var One;
}

private abstract MyInt(Int) from Int to Int {}

private enum abstract WrappedNumber(MyInt) to Int {
    var Zero;
    var Ninth = 9;
    var Tenth;
}
