private abstract WrappedIterator<T>(Iterator<T>) from Iterator<T> to Iterator<T> {}

typedef NamedIterator = WrappedIterator<String>;

class AbstractIteratorProtocol {
    static function check(value:Bool, label:String) {
        if (!value) throw label;
    }
    static function sum(values:WrappedIterator<Int>):Int {
        var total = 0;
        for (value in values) total += value;
        return total;
    }
    static function main() {
        var integers:WrappedIterator<Int> = [0, 1, 2].iterator();
        var text = "";
        for (value in integers) text += value;
        check(text == "012", "abstract iterator integers");
        check(sum([2, 3, 4].iterator()) == 9, "abstract iterator argument");
        var floats:WrappedIterator<Float> = [1.25, 2.5].iterator();
        var total = 0.0;
        for (value in floats) total += value;
        check(total == 3.75, "abstract iterator floats");
        var strings:NamedIterator = ["", "a", "b"].iterator();
        var joined = "";
        for (value in strings) joined += "[" + value + "]";
        check(joined == "[][a][b]", "abstract iterator typedef");
        var source:WrappedIterator<Int> = [2, 3].iterator();
        var doubled = [for (value in source) value * 2];
        check(doubled.length == 2 && doubled[0] == 4 && doubled[1] == 6, "abstract iterator comprehension");
        Sys.println("CONFORMANCE_OK");
    }
}
