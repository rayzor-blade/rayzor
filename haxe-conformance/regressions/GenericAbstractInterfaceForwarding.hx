interface ForwardingSlot<T> {
    function get():T;
    function set(value:T):Void;
}

class ForwardingCell<T> implements ForwardingSlot<T> {
    var value:T;
    public function new(value:T) this.value = value;
    public function get():T return value;
    public function set(value:T):Void this.value = value;
}

@:forward(get, set)
abstract ForwardingValue<T>(ForwardingSlot<T>) from ForwardingSlot<T> {}

typedef ForwardingTask = {final id:Int; final title:String; final lane:String;};

class GenericAbstractInterfaceForwarding {
    static function main() {
        var initial:Array<ForwardingTask> = [{id:1, title:"first", lane:"todo"}];
        var tasks:ForwardingValue<Array<ForwardingTask>> = new ForwardingCell(initial);
        var query:ForwardingValue<String> = new ForwardingCell("FIRST");
        var text = query.get().toLowerCase();
        var filtered = tasks.get().filter(task -> task.title == text);
        if (filtered.length != 1) throw "forwarded closure type";
        tasks.set([for (task in tasks.get()) task.id != 1 ? task : {id:task.id, title:task.title, lane:"done"}]);
        if (tasks.get()[0].lane != "done") throw "forwarded setter or comprehension";
        Sys.println("CONFORMANCE_OK");
    }
}
