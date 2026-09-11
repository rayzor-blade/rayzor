// A bare `case A:` over an enum abstract is a COMPARISON against that
// constant, not a capture. The name resolver only walked Enum types, so an
// enum abstract's values never resolved and every case became an irrefutable
// binding: the switch matched its first arm and the rest was folded away --
// silently, exit 0, wrong answer. `Xml.toString` is written this way.
private enum abstract IntK(Int) { var I0 = 0; var I1 = 1; var I2 = 2; }
private enum abstract SparseK(Int) { var S10 = 10; var S20 = 20; var S30 = 30; }
private enum abstract StrK(String) { var SA = "a"; var SB = "b"; var SC = "c"; }
private enum RealEnum { E1; E2; E3; }

class EnumAbstractSwitch {
    static function i(k:IntK):String return switch (k) { case I0:"0"; case I1:"1"; case I2:"2"; };
    static function s(k:SparseK):String return switch (k) { case S10:"a"; case S20:"b"; case S30:"c"; };
    static function t(k:StrK):String return switch (k) { case SA:"A"; case SB:"B"; case SC:"C"; };
    static function d(k:IntK):String return switch (k) { case I0:"z"; case _:"rest"; };
    // a real enum must keep working
    static function e(k:RealEnum):String return switch (k) { case E1:"1"; case E2:"2"; case E3:"3"; };

    static function main() {
        if (i(I0) + i(I1) + i(I2) != "012") throw "Int-backed enum abstract";
        if (s(S10) + s(S20) + s(S30) != "abc") throw "non-contiguous values";
        if (t(SA) + t(SB) + t(SC) != "ABC") throw "String-backed enum abstract";
        if (d(I0) != "z" || d(I2) != "rest") throw "default arm";
        if (e(E1) + e(E2) + e(E3) != "123") throw "real enum regressed";
        trace("CONFORMANCE_OK");
    }
}
