//! AMX (Accelerate) vs NEON F32 matmul microbench for the model's shapes.
//!
//! Answers the gating question before wiring Accelerate into the model: how much
//! does AMX beat a good NEON matmul for F32 GEMM at (a) decode (M=1, GEMV, which
//! is bandwidth-bound) and (b) prefill (M=32, compute-bound)? Layout is
//! weight-transposed: C[m,n] = A[m,k] · W[n,k]^T (weights [out,in]).
//!
//! Run: `cargo run -p rayzor-tensors --example accel_bench --release`

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("accel_bench is macOS-only (Accelerate/AMX).");
}

#[cfg(target_os = "macos")]
fn main() {
    use rayzor_tensors::apple_accel::sgemm_nt;
    use std::time::Instant;

    // NEON baseline: C[m,n] = sum_k A[m,k]*W[n,k], f32x4 accumulation.
    #[inline]
    fn neon_matmul_nt(m: usize, k: usize, n: usize, a: &[f32], w: &[f32], c: &mut [f32]) {
        use std::arch::aarch64::*;
        unsafe {
            for mi in 0..m {
                let arow = &a[mi * k..mi * k + k];
                for ni in 0..n {
                    let wrow = &w[ni * k..ni * k + k];
                    let mut acc = vdupq_n_f32(0.0);
                    let mut ki = 0;
                    while ki + 4 <= k {
                        let av = vld1q_f32(arow.as_ptr().add(ki));
                        let wv = vld1q_f32(wrow.as_ptr().add(ki));
                        acc = vfmaq_f32(acc, av, wv);
                        ki += 4;
                    }
                    let mut sum = vaddvq_f32(acc);
                    while ki < k {
                        sum += arow[ki] * wrow[ki];
                        ki += 1;
                    }
                    c[mi * n + ni] = sum;
                }
            }
        }
    }

    fn fill(n: usize, seed: u64) -> Vec<f32> {
        // deterministic pseudo-random in [-1,1) without rng deps
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn bench<F: FnMut()>(iters: usize, mut f: F) -> f64 {
        // warmup
        f();
        let t0 = Instant::now();
        for _ in 0..iters {
            f();
        }
        t0.elapsed().as_secs_f64() / iters as f64
    }

    // Llama-3.2-1B shapes: hidden K=2048, ffn N=8192, attn N=2048, lm_head N=128256.
    let cases: &[(&str, usize, usize)] = &[
        ("attn_qkv  K=2048 N=2560", 2048, 2560), // q2048+kv512
        ("ffn_up    K=2048 N=8192", 2048, 8192),
        ("ffn_down  K=8192 N=2048", 8192, 2048),
        ("lm_head   K=2048 N=128256", 2048, 128256),
    ];
    let batches: &[(&str, usize)] = &[("decode M=1", 1), ("prefill M=32", 32)];

    println!(
        "{:<26} {:<12} {:>10} {:>10} {:>8} {:>10}",
        "case", "batch", "AMX_us", "NEON_us", "speedup", "AMX_GFLOP/s"
    );
    for (cname, k, n) in cases {
        let w = fill(n * k, 0xABCD ^ (*n as u64));
        for (bname, m) in batches {
            let a = fill(m * k, 0x1234 ^ (*m as u64));
            let mut c_amx = vec![0.0f32; m * n];
            let mut c_neon = vec![0.0f32; m * n];
            // scale iters down for the huge lm_head to keep runtime sane
            let iters = if *n > 50000 { 20 } else { 100 };
            let t_amx = bench(iters, || sgemm_nt(*m, *k, *n, &a, &w, &mut c_amx));
            let t_neon = bench(iters, || neon_matmul_nt(*m, *k, *n, &a, &w, &mut c_neon));
            // correctness spot check
            let diff: f32 = c_amx
                .iter()
                .zip(&c_neon)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0, f32::max);
            let flops = 2.0 * (*m as f64) * (*k as f64) * (*n as f64);
            println!(
                "{:<26} {:<12} {:>10.1} {:>10.1} {:>7.2}x {:>10.1}  (maxdiff {:.4})",
                cname,
                bname,
                t_amx * 1e6,
                t_neon * 1e6,
                t_neon / t_amx,
                flops / t_amx / 1e9,
                diff
            );
        }
    }

    // --- Dequant amortization: Q4 weight -> F32 -> AMX sgemm ---
    // Weights are FIXED, so dequant is a ONE-TIME cost (dequant once at load,
    // cache the F32). "cached" = sgemm alone (the steady-state prefill cost);
    // "per-call" = dequant+sgemm every forward (the no-cache lower bound).
    use rayzor_runtime_core::quant::q4_k_m::dequant_q4_k_block;
    use rayzor_runtime_core::quant::types::{Q4KBlock, Q4_K_M_BLOCK_SIZE};
    let blk = Q4KBlock {
        d: 0.02,
        dmin: 0.01,
        scales: [0.02; 8],
        mins: [0.01; 8],
        quants: [0x53u8; 128],
    };
    println!("\n-- Q4 dequant amortization (dequant once + AMX, vs per-call) --");
    println!(
        "{:<26} {:<10} {:>10} {:>10} {:>12} {:>12}",
        "case", "batch", "dequant_us", "gemm_us", "cached_GF/s", "percall_GF/s"
    );
    for (cname, k, n) in cases {
        let nblocks = (n * k) / Q4_K_M_BLOCK_SIZE;
        let mut fw = vec![0.0f32; n * k];
        let t_deq = bench(20, || {
            let mut stage = [0.0f32; Q4_K_M_BLOCK_SIZE];
            for bi in 0..nblocks {
                dequant_q4_k_block(&blk, &mut stage);
                let off = bi * Q4_K_M_BLOCK_SIZE;
                fw[off..off + Q4_K_M_BLOCK_SIZE].copy_from_slice(&stage);
            }
        });
        for (bname, m) in &[
            ("decode M=1", 1usize),
            ("prefill M=32", 32),
            ("prefill M=128", 128),
        ] {
            let a = fill(m * k, 0x77 ^ *m as u64);
            let mut c = vec![0.0f32; m * n];
            let iters = if *n > 50000 { 20 } else { 100 };
            let t_gemm = bench(iters, || sgemm_nt(*m, *k, *n, &a, &fw, &mut c));
            let flops = 2.0 * (*m as f64) * (*k as f64) * (*n as f64);
            println!(
                "{:<26} {:<10} {:>10.0} {:>10.0} {:>12.0} {:>12.0}",
                cname,
                bname,
                t_deq * 1e6,
                t_gemm * 1e6,
                flops / t_gemm / 1e9,
                flops / (t_deq + t_gemm) / 1e9,
            );
        }
    }

    // --- BNNSMatMul dtype probes: which input types does the AMX path accept? ---
    use rayzor_tensors::apple_accel::{
        matmul_f16_nt, matmul_f16f32_nt, matmul_f32_bnns_nt, matmul_i8_nt, matmul_i8f32_nt,
    };

    // IEEE binary16 bits for small exact values (probe/fill values only).
    fn f16_bits(x: f32) -> u16 {
        let bits = x.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        if x == 0.0 {
            return sign;
        }
        let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
        let mant = ((bits & 0x7f_ffff) >> 13) as u16;
        sign | ((exp as u16) << 10) | mant
    }

    // Correctness gates FIRST — never trust throughput from a rejected call.
    // A[2x3]=[[1,2,3],[4,5,6]], B[2x3]=[[1,0,1],[0,1,0]], C=A·B^T=[[4,2],[10,5]].
    let af: [f32; 6] = [1., 2., 3., 4., 5., 6.];
    let bf: [f32; 6] = [1., 0., 1., 0., 1., 0.];
    let mut cf = [0f32; 4];
    let f32_ok = matmul_f32_bnns_nt(2, 3, 2, &af, &bf, &mut cf) && cf == [4., 2., 10., 5.];
    println!("\n-- BNNSMatMul dtype probes (want [4,2,10,5]) --");
    println!(
        "f32          : {} {:?}",
        if f32_ok { "PASS" } else { "FAIL" },
        cf
    );

    let ah: Vec<u16> = af.iter().map(|&x| f16_bits(x)).collect();
    let bh: Vec<u16> = bf.iter().map(|&x| f16_bits(x)).collect();
    let mut ch = [0u16; 4];
    let f16_rc = matmul_f16_nt(2, 3, 2, &ah, &bh, &mut ch);
    let want_h: Vec<u16> = [4f32, 2., 10., 5.].iter().map(|&x| f16_bits(x)).collect();
    let f16_ok = f16_rc && ch == want_h[..];
    println!(
        "f16->f16     : {} rc_ok={} got={:04x?}",
        if f16_ok { "PASS" } else { "FAIL" },
        f16_rc,
        ch
    );

    let mut chf = [0f32; 4];
    let f16f32_ok = matmul_f16f32_nt(2, 3, 2, &ah, &bh, &mut chf) && chf == [4., 2., 10., 5.];
    println!(
        "f16->f32     : {} {:?}",
        if f16f32_ok { "PASS" } else { "FAIL" },
        chf
    );

    let a8: [i8; 6] = [1, 2, 3, 4, 5, 6];
    let b8: [i8; 6] = [1, 0, 1, 0, 1, 0];
    let mut c8 = [0i32; 4];
    let i8_ok = matmul_i8_nt(2, 3, 2, &a8, &b8, &mut c8) && c8 == [4, 2, 10, 5];
    let mut c8f = [0f32; 4];
    let i8f32_ok = matmul_i8f32_nt(2, 3, 2, &a8, &b8, &mut c8f) && c8f == [4., 2., 10., 5.];
    println!(
        "int8->int32  : {} {:?}",
        if i8_ok { "PASS" } else { "FAIL" },
        c8
    );
    println!(
        "int8->f32    : {} {:?}",
        if i8f32_ok { "PASS" } else { "FAIL" },
        c8f
    );

    // f16 throughput at model shapes (only if the gate passed).
    if f16_ok || f16f32_ok {
        println!(
            "\n-- f16 BNNSMatMul throughput (AMX f16 ~2x f32 rate; cblas f32 was 400-900 GF/s) --"
        );
        println!(
            "{:<26} {:<14} {:>12} {:>12}",
            "case", "batch", "f16f16_GF/s", "f16f32_GF/s"
        );
        for (cname, k, n) in cases {
            let bw: Vec<u16> = (0..(n * k))
                .map(|i| f16_bits(((i * 7 + 3) % 15) as f32 - 7.0))
                .collect();
            for (bname, m) in &[
                ("decode M=1", 1usize),
                ("prefill M=32", 32),
                ("prefill M=194", 194),
            ] {
                let a: Vec<u16> = (0..(m * k))
                    .map(|i| f16_bits(((i * 5 + 1) % 15) as f32 - 7.0))
                    .collect();
                let mut c16 = vec![0u16; m * n];
                let mut c32 = vec![0f32; m * n];
                let iters = if *n > 50000 { 20 } else { 50 };
                let flops = 2.0 * (*m as f64) * (*k as f64) * (*n as f64);
                let g16 = if f16_ok {
                    let t = bench(iters, || {
                        matmul_f16_nt(*m, *k, *n, &a, &bw, &mut c16);
                    });
                    flops / t / 1e9
                } else {
                    0.0
                };
                let g32 = if f16f32_ok {
                    let t = bench(iters, || {
                        matmul_f16f32_nt(*m, *k, *n, &a, &bw, &mut c32);
                    });
                    flops / t / 1e9
                } else {
                    0.0
                };
                println!("{:<26} {:<14} {:>12.0} {:>12.0}", cname, bname, g16, g32);
            }
        }
    }

    // NEON i8 reference (scalar-ish i32 accumulate, the honest lower bound; the
    // in-model SDOT path runs ~120 GFLOP/s effective).
    #[inline]
    fn neon_i8_nt(m: usize, k: usize, n: usize, a: &[i8], b: &[i8], c: &mut [i32]) {
        use std::arch::aarch64::*;
        unsafe {
            for mi in 0..m {
                let arow = &a[mi * k..mi * k + k];
                for ni in 0..n {
                    let wrow = &b[ni * k..ni * k + k];
                    // Stable NEON widening MAC (no nightly sdot): 8-bit ->
                    // 16-bit products, pairwise-accumulated into i32.
                    let mut acc = vdupq_n_s32(0);
                    let mut ki = 0;
                    while ki + 16 <= k {
                        let av = vld1q_s8(arow.as_ptr().add(ki));
                        let wv = vld1q_s8(wrow.as_ptr().add(ki));
                        let plo = vmull_s8(vget_low_s8(av), vget_low_s8(wv));
                        let phi = vmull_s8(vget_high_s8(av), vget_high_s8(wv));
                        acc = vpadalq_s16(acc, plo);
                        acc = vpadalq_s16(acc, phi);
                        ki += 16;
                    }
                    let mut sum = vaddvq_s32(acc);
                    while ki < k {
                        sum += arow[ki] as i32 * wrow[ki] as i32;
                        ki += 1;
                    }
                    c[mi * n + ni] = sum;
                }
            }
        }
    }

    if !i8_ok {
        println!("\n-- int8 throughput SKIPPED: BNNSMatMul rejects int8 (public matmul is float-only) --");
        return;
    }
    println!("\n-- int8-AMX (BNNS) vs NEON sdot, model shapes --");
    println!(
        "{:<26} {:<12} {:>12} {:>12} {:>8} {:>10}",
        "case", "batch", "AMX8_GF/s", "NEON8_GF/s", "speedup", "maxdiff"
    );
    for (cname, k, n) in cases {
        let bw: Vec<i8> = (0..(n * k)).map(|i| ((i * 7 + 3) % 15) as i8 - 7).collect();
        for (bname, m) in &[
            ("decode M=1", 1usize),
            ("prefill M=32", 32),
            ("prefill M=194", 194),
        ] {
            let a: Vec<i8> = (0..(m * k)).map(|i| ((i * 5 + 1) % 15) as i8 - 7).collect();
            let mut c_amx = vec![0i32; m * n];
            let mut c_neon = vec![0i32; m * n];
            let iters = if *n > 50000 { 20 } else { 50 };
            let t_amx = bench(iters, || {
                matmul_i8_nt(*m, *k, *n, &a, &bw, &mut c_amx);
            });
            let t_neon = bench(iters, || neon_i8_nt(*m, *k, *n, &a, &bw, &mut c_neon));
            let diff = c_amx
                .iter()
                .zip(&c_neon)
                .map(|(x, y)| (x - y).abs())
                .max()
                .unwrap_or(0);
            let flops = 2.0 * (*m as f64) * (*k as f64) * (*n as f64);
            println!(
                "{:<26} {:<12} {:>12.0} {:>12.0} {:>7.1}x {:>10}",
                cname,
                bname,
                flops / t_amx / 1e9,
                flops / t_neon / 1e9,
                t_neon / t_amx,
                diff
            );
        }
    }
}
