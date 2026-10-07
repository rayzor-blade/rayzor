class NestedPropertyAccessors {
    static var shared(default, set):Int;
    static var sharedRead(get, default):Int;
    static var restoreShared:()->Void;
    static var readShared:()->Int;

    var member(default, set):Int;
    var observed(get, default):Int;
    var setterCalls:Int = 0;
    var getterCalls:Int = 0;
    var other:NestedPropertyAccessors;
    var restoreMember:()->Void;
    var readMember:()->Int;

    function new() {
        member = 0;
        observed = 5;
    }

    function set_member(value:Int):Int {
        setterCalls++;
        member = value + 1;
        restoreMember = () -> {
            var nested = () -> {
                member = 3;
                if (other != null) other.member = 7;
            };
            nested();
        };
        return member;
    }

    function get_observed():Int {
        getterCalls++;
        readMember = () -> {
            var nested = () -> observed;
            return nested() + (other == null ? 0 : other.observed);
        };
        return observed + 10;
    }

    static function set_shared(value:Int):Int {
        shared = value + 1;
        restoreShared = () -> {
            var nested = () -> shared = 4;
            nested();
        };
        return shared;
    }

    static function get_sharedRead():Int {
        readShared = () -> {
            var nested = () -> sharedRead;
            return nested();
        };
        return sharedRead + 10;
    }

    static function main() {
        var instance = new NestedPropertyAccessors();
        var peer = new NestedPropertyAccessors();
        instance.other = peer;
        if ((instance.member = 42) != 43) throw "setter result";
        instance.restoreMember();
        if (instance.member != 3) throw "nested instance backing write";
        if (instance.setterCalls != 2) throw "instance setter reentry";
        if (peer.member != 8 || peer.setterCalls != 2) throw "other instance setter";
        if (instance.observed != 15) throw "getter result";
        if (instance.readMember() != 20) throw "nested instance backing read";
        if (instance.getterCalls != 1) throw "instance getter reentry";
        if (peer.getterCalls != 1) throw "other instance getter";
        shared = 42;
        restoreShared();
        if (shared != 4) throw "nested static backing write";
        sharedRead = 6;
        if (sharedRead != 16) throw "static getter result";
        if (readShared() != 6) throw "nested static backing read";
        Sys.println("CONFORMANCE_OK");
    }
}
