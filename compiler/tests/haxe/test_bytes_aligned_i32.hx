// Regression test: rayzor.Bytes aligned unchecked i32 helpers should be
// 4-byte little-endian load/stores. Pure-Haxe quant kernels use this only for
// aligned, in-bounds scratch buffers.

import rayzor.Bytes;

class Main {
    static function main() {
        var b = Bytes.alloc(16);
        b.storeI32AlignedUnchecked(0, 0x11223344);
        b.storeI32AlignedUnchecked(4, -1234567);
        b.storeI32AlignedUnchecked(8, 0x55667788);

        var a = b.loadI32AlignedUnchecked(0);
        var c = b.loadI32AlignedUnchecked(4);
        var d = b.loadI32AlignedUnchecked(8);
        var byte4 = b.get(4);
        var byte8 = b.get(8);
        var untouched = b.get(12);

        var ok = a == 0x11223344 && c == -1234567 && d == 0x55667788
            && byte4 == 0x79 && byte8 == 0x88 && untouched == 0;
        if (ok) {
            Sys.println("PASS bytes-aligned-i32 a=" + a + " c=" + c + " d=" + d
                + " byte4=" + byte4 + " byte8=" + byte8);
        } else {
            Sys.println("FAIL bytes-aligned-i32 a=" + a + " c=" + c + " d=" + d
                + " byte4=" + byte4 + " byte8=" + byte8 + " untouched=" + untouched);
        }
    }
}
