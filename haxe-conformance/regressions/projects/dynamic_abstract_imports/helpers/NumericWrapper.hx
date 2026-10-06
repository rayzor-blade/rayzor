package helpers;
abstract NumericWrapper(NumericSlot) {
    inline private function new(value) this = value;
    inline public static function make(a:Int, b:Int):NumericWrapper {
        return new NumericWrapper(cast a + b);
    }
    @:op(A * B) inline public function multiply(other:NumericWrapper):NumericWrapper {
        return new NumericWrapper(cast this.get() * other.get().get());
    }
    public function get() return this;
}
