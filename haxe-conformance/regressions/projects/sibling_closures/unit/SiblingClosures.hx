package unit;

class SiblingClosures {
    static function main() {
        var identity = (number:Int) -> number;
        if (identity(9) != 9) throw "local Int closure";
        var receiver = new ClosureSource(100);
        if (receiver.first()(2) != "first2") throw "first imported String closure";
        if (receiver.second()(3) != "second3") throw "second imported String closure";
        var add = receiver.add;
        if (receiver.add(1, 2) != 103) throw "direct method";
        if (receiver.add.bind(1)(2) != 103) throw "bound method";
        if (add(1, 2) != 103) throw "method reference";
        var prefix = "local";
        var local = (number:Int) -> prefix + number;
        if (local(4) != "local4") throw "local captured String";
        var dynamicFirst:Dynamic = receiver.first();
        var dynamicLocal:Dynamic = local;
        if (dynamicFirst(5) != "first5") throw "imported dynamic entry";
        if (dynamicLocal(6) != "local6") throw "local dynamic entry";
        var underscorePackage = pkg_one.Probe.make();
        var nestedPackage = pkg.one.Probe.make();
        if (underscorePackage(7) != 107) throw "underscore package closure";
        if (nestedPackage(8) != 208) throw "nested package closure";
        Sys.println("CONFORMANCE_OK");
    }
}
