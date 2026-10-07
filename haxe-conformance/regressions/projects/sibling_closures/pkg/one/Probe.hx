package pkg.one;

class Probe {
    public static function make():Int->Int {
        var base = 200;
        return (value:Int) -> base + value;
    }
}
