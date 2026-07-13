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
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
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
}
