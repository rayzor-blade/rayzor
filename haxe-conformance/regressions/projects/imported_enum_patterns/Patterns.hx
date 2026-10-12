enum UnicodeString {
    Only(values:Array<Int>);
    Pair(left:Array<Int>, right:Array<Int>);
}
class Patterns {
    public static var values:Array<UnicodeString> = [Only([1]), Only([2]), Pair([3], [4])];
    public static var filtered:Array<UnicodeString> = {
        var values = values.copy();
        values = values.filter(v -> !v.match(Only([1])));
        values;
    };
}
