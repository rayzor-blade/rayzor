interface NullableReceiver<T> {
    public function accept(value:T):Void;
}

abstract NullableReceiverHandle<T>(NullableReceiver<T>) from NullableReceiver<T> to NullableReceiver<T> {}

class NullableCounter implements NullableReceiver<Int> {
    public var value:Int = 0;
    public function new() {}
    public function accept(value:Int):Void this.value += value;
}

class NullableReceiverHolder<T> {
    public var receiver:Null<NullableReceiver<T>> = null;
    public function new() {}
    public function attach(receiver:NullableReceiver<T>):Void this.receiver = receiver;
    public function attachHandle(receiver:NullableReceiverHandle<T>):Void this.receiver = receiver;
    public function accept(value:T):Void receiver.accept(value);
}

class NullableInterfaceIdentity {
    static function main() {
        var counter = new NullableCounter();
        var receiver:NullableReceiver<Int> = counter;
        var holder = new NullableReceiverHolder<Int>();
        holder.attach(receiver);
        holder.accept(17);
        holder.receiver = receiver;
        holder.accept(25);
        var handle:NullableReceiverHandle<Int> = receiver;
        holder.attachHandle(handle);
        holder.accept(9);
        if (counter.value != 51) throw "nullable interface receiver identity";
        var missing:Null<NullableCounter> = null;
        var empty:NullableReceiver<Int> = missing;
        if (empty != null) throw "nullable class to interface";
        Sys.println("CONFORMANCE_OK");
    }
}
