import helper.PropertyParent;
import helper.PropertyChild;
class VirtualPropertyImports {
    static function main() {
        var child = new PropertyChild();
        var parent:PropertyParent = child;
        if (parent.value != 2.5 || parent.label != "child") throw "imported getters";
        if ((parent.value = 4.5) != 4.5 || child.value != 4.5) throw "imported setter";
        if (child.parentValue() != -1.0) throw "imported super getter";
        if (child.parentSet(7.5) != -1.0 || parent.value != 4.5) throw "imported super setter";
        Sys.println("CONFORMANCE_OK");
    }
}
