enum abstract Color(Int) {
    var Red = 0;
    var Green = 1;
    var Blue = 2;
}

class Main {
    static function main() {
        var c:Color = Color.Red;
        trace(c);           // 0
        trace(Color.Green);  // 1
        trace(Color.Blue);   // 2
    }
}
