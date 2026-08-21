// `a.add(b)` on an integer SIMD type used to ABORT THE COMPILER on the LLVM
// backend and drop the program to Cranelift with an "Unsupported cast from I64
// to Vector" once that panic was removed — while `a + b` on the same values was
// correct. The method form went through a MIR wrapper with a broken vector ABI,
// because only SIMD4f was routed to VectorBinOp. Both forms must agree.
import rayzor.SIMD4i32;
class test_simd_int_method_arith {
    static function main():Void {
        var a = SIMD4i32.make(1, 2, 3, 4);
        var b = SIMD4i32.make(10, 20, 30, 40);
        var viaOp = a + b;
        Sys.println("a + b       -> [" + viaOp.get(0) + "," + viaOp.get(1) + "," + viaOp.get(2) + "," + viaOp.get(3) + "]");
        var viaMethod = a.add(b);
        Sys.println("a.add(b)    -> [" + viaMethod.get(0) + "," + viaMethod.get(1) + "," + viaMethod.get(2) + "," + viaMethod.get(3) + "]");
    }
}
