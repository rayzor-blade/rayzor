import helper.Fields.increment;
import helper.Fields.increment as inc;
import helper.Fields.count;
import helper.Fields.LIMIT as cap;
import helper.OnlyAliased.VALUE as aliasOnly;
import helper.ImportedMacro.value;
import helper.ImportedMacro.value as macroAlias;
class StaticMemberImports {
    static function main() {
        if (increment(4) != 5) throw "method";
        if (inc(5) != 6) throw "method alias";
        var callback = inc;
        if (callback(6) != 7) throw "method value";
        if (count != 3) throw "field read";
        count = 8;
        if (helper.Fields.count != 8) throw "field write";
        if (cap != 11) throw "constant alias";
        if (aliasOnly != 17) throw "alias-only module";
        var local = 2;
        if (value(local) != 7) throw "macro field import";
        if (macroAlias(local) != 7) throw "macro alias";
        Sys.println("CONFORMANCE_OK");
    }
}
