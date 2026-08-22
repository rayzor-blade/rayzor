// A deliberate read of freed memory, so the oracle can be seen failing.
//
// check_uaf.sh reports a program whose output changes when freed memory is
// poisoned. That claim is worth nothing until the detector has been observed
// firing on a real instance -- a check that has never come back positive is a
// check nobody has tested. This program is that instance.
//
// The read is at an OFFSET, not at the base. A freed block's first words hold
// the allocator's own free-list link, whose value follows the address and so
// varies run to run; reading there produces noise in both modes and tells you
// nothing. Past the header the poison survives, and a poisoned read yields
// 0x55555555 every time.
//
// WHAT THIS DOES NOT COVER: the object pool. `Bytes.alloc` is a runtime
// allocation that never reaches the compiler's object path, and 4096 is above
// the pool's size cap regardless -- two independent reasons no size would put
// this fixture on the pooled path. It proves the detector fires on libc
// memory, which is a true statement about the wrong subject if you are asking
// about the allocator. The pool is covered separately: unit tests assert its
// fill bytes directly, and a served-block counter proves compiled code reaches
// it. Neither can be a fixture like this one, because a class instance is the
// pooled allocation and Haxe offers no way to free one and then read it back.
//
// Unpoisoned, the same read usually returns what was written, because nothing
// has reused the block yet. That is exactly why this class of defect survives
// a passing test suite and needs an allocator to expose it.
// The address is captured BEFORE the free and read through Mem afterwards.
// Reading through the handle instead would fault rather than observe: the
// release frees the header too, so the crash arrives before the payload is
// ever touched, and both modes then die identically with nothing to compare.
import rayzor.Bytes;
import rayzor.Mem;
import rayzor.Usize;

class UafFixture {
    static inline var SIZE = 4096;
    static inline var OFFSET = 512;

    static function main() {
        var buf = Bytes.alloc(SIZE);
        buf.fill(0, SIZE, 0x27);
        var addr = buf.address() + Usize.fromInt(OFFSET);
        Sys.println("live=" + Mem.loadI32(addr));
        buf.free();
        Sys.println("dead=" + Mem.loadI32(addr));
    }
}
