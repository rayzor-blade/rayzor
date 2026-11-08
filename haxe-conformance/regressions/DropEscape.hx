@:derive(Drop)
class Handle {
    public var id:Int;
    public function new(id:Int) { this.id = id; }
    public function drop():Void { drops.push(id); }
    public static var drops:Array<Int> = [];
}

class Holder {
    public var h:Handle;
    public function new(h:Handle) { this.h = h; }
}

class DropEscape {
    static var keep:Handle;
    static var items:Array<Handle> = [];

    static function owns(h:Handle):Int return h.id;
    static function borrows(@:borrow h:Handle):Int return h.id;

    static function local():Void {
        var a = new Handle(1);
        if (a.id != 1) throw "local";
    }
    static function stored():Holder {
        var a = new Handle(3);
        return new Holder(a);
    }
    static function toStatic():Void {
        var a = new Handle(4);
        keep = a;
    }
    static function toField(h:Holder):Void {
        var a = new Handle(5);
        h.h = a;
    }
    static function returned():Handle {
        var a = new Handle(6);
        return a;
    }

    static function toArray():Void {
        var a = new Handle(7);
        items.push(a);
    }
    static function handedOver():Void {
        var a = new Handle(8);
        owns(a);
    }
    static function borrowed():Void {
        var a = new Handle(9);
        borrows(a);
    }

    static function main() {
        local();
        if (Handle.drops.join(",") != "1") throw "local drop " + Handle.drops;
        var h = stored();
        if (h.h.id != 3) throw "stored id " + h.h.id;
        toStatic();
        if (keep.id != 4) throw "static id " + keep.id;
        toField(h);
        if (h.h.id != 5) throw "field id " + h.h.id;
        var r = returned();
        if (r.id != 6) throw "returned id " + r.id;
        if (Handle.drops.join(",") != "1") throw "escaped value dropped " + Handle.drops;
        toArray();
        if (items[0].id != 7) throw "array element " + items[0].id;
        handedOver();
        borrowed();
        if (Handle.drops.join(",") != "1,9") throw "ownership rule " + Handle.drops;
        Sys.println("CONFORMANCE_OK");
    }
}
