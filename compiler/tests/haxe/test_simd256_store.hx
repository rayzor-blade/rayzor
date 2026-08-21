// The 256-bit types could be built and reduced but never written back:
// SIMD8i32 had no load and no store, SIMD32i8 had no store. The MIR wrappers
// existed and nothing reached them.
//
// LLVM-only, like every 256-bit test: wasm's v128 and Cranelift have no
// 256-bit vector and refuse these types.
import rayzor.Ptr;
import rayzor.Mem;
import rayzor.Usize;
import rayzor.Bytes;
#if llvm
import rayzor.SIMD8i32;
import rayzor.SIMD32i8;
#end

class Main {
    static function main() {
        #if llvm
        var b = Bytes.alloc(256);
        var base:Usize = b.address();

        // i32x8: build a known pattern, load it, store it somewhere else.
        for (i in 0...8) Mem.storeI32(base + Usize.fromInt(i * 4), (i + 1) * 7);
        SIMD8i32.load(Ptr.fromRaw(base)).store(Ptr.fromRaw(base + Usize.fromInt(64)));
        for (i in 0...8) {
            var got = Mem.loadI32(base + Usize.fromInt(64 + i * 4));
            if (got != (i + 1) * 7) Sys.println("FAIL i32x8 lane " + i + " = " + got);
        }
        var s = SIMD8i32.load(Ptr.fromRaw(base + Usize.fromInt(64))).sum();
        if (s != 252) Sys.println("FAIL i32x8 sum = " + s);
        Sys.println("i32x8 round-trip ok, sum " + s);

        // i8x32: splat a byte and write all 32 lanes.
        SIMD32i8.splat(5).store(Ptr.fromRaw(base + Usize.fromInt(128)));
        for (i in 0...32) {
            var g = Mem.loadU8(base + Usize.fromInt(128 + i));
            if (g != 5) Sys.println("FAIL i8x32 lane " + i + " = " + g);
        }
        Sys.println("i8x32 splat store ok");
        #else
        Sys.println("256-bit types unavailable on this backend, skipped");
        #end
    }
}
