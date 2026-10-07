package pkg_one;

class Probe {
    public static function make():Int->Int {
        var base = 100;
        return (value:Int) -> base + value;
    }
}
