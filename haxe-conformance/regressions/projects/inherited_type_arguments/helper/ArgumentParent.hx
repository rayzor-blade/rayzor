package helper;
class ArgumentParent<T> {
    public var value:T;
    public function new(value:T) this.value = value;
    public function get():T return value;
}
