import haxe.Exception;

class Main {
    static function thrower() {
        throw new Exception("boom");
    }

    static function main() {
        thrower();
    }
}
