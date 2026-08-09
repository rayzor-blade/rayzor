// Equivalence oracle for FlashDecode.decodeBatch, the batched (prefill) Q8
// attention behind NUE_FLASH_BATCH.
//
// The claim under test comes from the causal mask: because the K/V for these
// rows are already in the cache, query row `r` attends over keys
// [0, baseLen + r] and nothing else. So decodeBatch(baseLen, seqQ) row `r`
// must equal decode(q_r, cacheLen = baseLen + r + 1) -- the single-row path
// that already ships as the default.
//
// Both routes funnel into the same bandOneSigned with the same query
// quantiser, so the requirement here is BIT-IDENTICAL, not "close": any
// difference means the batched path is doing different arithmetic, which is
// exactly the failure a previous seqQ>1 attempt shipped as a quality
// regression.
//
// Deliberately NOT compared against the Rust bmm fallback. That route
// dequantises to F32 and is a different numeric path, so disagreeing with it
// proves nothing about this kernel's correctness.
import rayzor.ds.Tensor;
import rayzor.ds.DType;
import rayzor.concurrent.SpinPool;
import nue.transformer.Q8Cache;
import nue.transformer.FlashDecode;

class Main {
    // Deterministic pseudo-random in [-1, 1), no dependency on a seeded RNG.
    static function prand(i:Int):Float {
        var x = (i * 1103515245 + 12345) & 0x7FFFFFFF;
        x = (x >> 7) ^ (x << 3);
        x = x & 0x7FFFFFFF;
        return (x % 20011) / 10005.5 - 1.0;
    }

    static function fill(t:Tensor, n:Int, salt:Int):Void {
        for (i in 0...n) t.setFlat(i, prand(i + salt * 7919));
    }

    static var failures = 0;
    static var compared = 0;

    static function check(name:String, batch:Tensor, single:Tensor, row:Int,
            rowFloats:Int):Void {
        var bad = 0;
        var worst = 0.0;
        var worstAt = -1;
        for (i in 0...rowFloats) {
            var a = batch.getFlat(row * rowFloats + i);
            var b = single.getFlat(i);
            compared++;
            if (a != b) {
                bad++;
                var d = a - b;
                if (d < 0) d = -d;
                if (d > worst) { worst = d; worstAt = i; }
            }
        }
        if (bad != 0) {
            failures++;
            Sys.println("  FAIL " + name + " row=" + row + ": " + bad + "/"
                + rowFloats + " lanes differ, worst |d|=" + worst
                + " at lane " + worstAt);
        }
    }

    static function runCase(maxSeqLen:Int, numKvHeads:Int, headDim:Int,
            numQHeads:Int, baseLen:Int, seqQ:Int, workers:Int):Void {
        var name = "kv=" + numKvHeads + " q=" + numQHeads + " hd=" + headDim
            + " baseLen=" + baseLen + " seqQ=" + seqQ + " workers=" + workers;
        var sp = new SpinPool(workers);
        var kc = new Q8Cache(maxSeqLen, numKvHeads, headDim);
        var vc = new Q8Cache(maxSeqLen, numKvHeads, headDim);

        // Fill the cache through baseLen+seqQ: the batched path assumes the
        // rows under test are already resident.
        var total = baseLen + seqQ;
        var kv = Tensor.zeros([total, numKvHeads, headDim], DType.F32);
        fill(kv, total * numKvHeads * headDim, 1);
        kc.append(0, kv);
        var vv = Tensor.zeros([total, numKvHeads, headDim], DType.F32);
        fill(vv, total * numKvHeads * headDim, 2);
        vc.append(0, vv);

        var scale = 1.0 / Math.sqrt(headDim);
        var rowFloats = numQHeads * headDim;

        var q = Tensor.zeros([seqQ, numQHeads, headDim], DType.F32);
        fill(q, seqQ * rowFloats, 3);

        var batched = FlashDecode.decodeBatch(kc, vc, q, baseLen, seqQ,
            numQHeads, scale, sp);
        if (batched == null) {
            Sys.println("  SKIP " + name + " (decodeBatch declined)");
            sp.shutdown();
            return;
        }

        // Same query row, same cache, single-row path, cache length set by the
        // causal mask.
        for (r in 0...seqQ) {
            var q1 = Tensor.zeros([1, numQHeads, headDim], DType.F32);
            for (i in 0...rowFloats) q1.setFlat(i, q.getFlat(r * rowFloats + i));
            var single = FlashDecode.decode(kc, vc, q1, baseLen + r + 1,
                numQHeads, scale, sp);
            check(name, batched, single, r, rowFloats);
            single.free();
            q1.free();
        }
        Sys.println("  ok   " + name);

        batched.free();
        q.free();
        kv.free();
        vv.free();
        kc.free();
        vc.free();
        sp.shutdown();
    }

    static function main() {
        Sys.println("decodeBatch == decode(cacheLen = baseLen + r + 1), bitwise");

        // MHA (group 1), GQA (group 7, qwen2-shaped), and a group-2 case.
        // baseLen 0 exercises the first prefill chunk; a non-zero baseLen
        // exercises a continuation, where the per-row cache length no longer
        // starts at 1.
        runCase(256, 2, 64, 2, 0, 8, 4);
        runCase(256, 2, 64, 14, 0, 8, 4);
        runCase(256, 2, 64, 14, 13, 8, 4);
        runCase(256, 4, 128, 8, 5, 16, 4);
        runCase(256, 2, 64, 14, 0, 1, 4);
        // Worker count must not change the result: rows are independent, so a
        // different banding is a different schedule over the same arithmetic.
        runCase(256, 2, 64, 14, 3, 8, 1);
        runCase(256, 2, 64, 14, 3, 8, 8);

        Sys.println("compared " + compared + " lanes");
        if (failures == 0) {
            Sys.println("PASS");
        } else {
            Sys.println("FAIL: " + failures + " row(s) diverged");
            Sys.exit(1);
        }
    }
}
