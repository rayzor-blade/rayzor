import haxe.macro.Context;
import haxe.macro.Expr;
import haxe.macro.ComplexTypeTools;
import haxe.macro.TypeTools;
class TyperBuild {
    public static macro function fill():Array<Field> {
        var fields = Context.getBuildFields();
        var type = ComplexTypeTools.toType(macro : Map<String, Int>);
        var once = TypeTools.followWithAbstracts(type, true);
        if (Std.string(once) != "TAbstract(haxe.ds.Map,[TInst(String,[]),TAbstract(Int,[])])")
            Context.error("one-step follow: " + Std.string(once), Context.currentPos());
        var value = "";
        value += type;
        value += "|";
        value += TypeTools.followWithAbstracts(type, false);
        fields.push((macro class Generated { public static var description = $v{value}; }).fields[0]);
        return fields;
    }
    public static macro function confirm():Array<Field> {
        var fields = Context.getBuildFields();
        var found = false;
        for (field in fields) if (field.name == "description") found = true;
        if (!found) Context.error("build order", Context.currentPos());
        fields.push((macro class Generated { public static var checked = true; }).fields[0]);
        return fields;
    }
}
