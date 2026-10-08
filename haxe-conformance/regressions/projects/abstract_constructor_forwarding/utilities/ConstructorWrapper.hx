package utilities;

@:forward.new
abstract ConstructorWrapper<T>(T) to T {
    public function value():T return this;
}
