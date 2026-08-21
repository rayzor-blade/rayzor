// The integer SIMD types could be LOADED but not STORED — SIMD4f had a store
// and none of SIMD4i32/SIMD8i32/SIMD16i8/SIMD32i8 did. So a kernel could read
// quantised bytes and never write them back, and the produce side of every
// quantise fell to a scalar loop after a vectorised body.
import rayzor.SIMD4i32;
import rayzor.SIMD16i8;
import rayzor.Ptr;
import rayzor.Mem;
import rayzor.Usize;
import rayzor.Bytes;

class test_simd_int_store {
    static function main():Void {
        var b = Bytes.alloc(128);
        var base:Usize = b.address();

        var v = SIMD4i32.make(11, 22, 33, 44);
        v.store(Ptr.fromRaw(base));
        Sys.print("i32 store -> ");
        for (i in 0...4) Sys.print(Mem.loadI32(base + Usize.fromInt(i * 4)) + " ");
        Sys.println("");

        var back = SIMD4i32.load(Ptr.fromRaw(base));
        Sys.println("round-trip -> [" + back.get(0) + "," + back.get(1) + "," + back.get(2) + "," + back.get(3) + "]");

        for (i in 0...16) Mem.storeU8(base + Usize.fromInt(64 + i), i * 3);
        var bv = SIMD16i8.load(Ptr.fromRaw(base + Usize.fromInt(64)));
        bv.store(Ptr.fromRaw(base + Usize.fromInt(96)));
        Sys.print("i8 store -> ");
        for (i in 0...8) Sys.print(Mem.loadU8(base + Usize.fromInt(96 + i)) + " ");
        Sys.println("");
    }
}
