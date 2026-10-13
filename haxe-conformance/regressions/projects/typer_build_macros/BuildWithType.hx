@:build(TyperBuild.fill())
@:build(TyperBuild.confirm())
private class Built {}
@:autoBuild(TyperBuild.fill())
private class Base { public function new() {} }
private class Child extends Base {}
class BuildWithType {
    static function main() {
        if (!Built.checked) throw "second build";
        if (Built.description != "TType(Map,[TInst(String,[]),TAbstract(Int,[])])|TInst(haxe.ds.StringMap,[TAbstract(Int,[])])") throw Built.description;
        if (Child.description != Built.description) throw "auto build";
        trace("CONFORMANCE_OK");
    }
}
