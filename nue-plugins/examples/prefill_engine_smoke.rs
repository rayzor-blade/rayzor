//! Direct smoke of prefill_graph FFI (no JIT): load + bucket + execute.
fn main() {
    let dir = std::env::args().nth(1).expect("dir");
    let stem = std::env::args().nth(2).expect("stem");
    let (hidden, layers, kvh, hd) = (2048usize, 16usize, 8usize, 64usize);
    let h = unsafe {
        nue_plugins::prefill_graph::nue_prefill_graph_load(
            dir.as_ptr() as i64,
            dir.len() as i64,
            stem.as_ptr() as i64,
            stem.len() as i64,
            hidden as i64,
            layers as i64,
            kvh as i64,
            hd as i64,
        )
    };
    println!("load handle={h}");
    let s = nue_plugins::prefill_graph::nue_prefill_graph_bucket(h, 40) as usize;
    println!("bucket={s}");
    let hin = vec![0.01f32; s * hidden];
    let mut out = vec![0f32; s * hidden];
    let mut kv = vec![0f32; layers * 2 * s * kvh * hd];
    println!("calling execute (first predict = E5RT specialization, may take minutes)...");
    let t0 = std::time::Instant::now();
    let rc = unsafe {
        nue_plugins::prefill_graph::nue_prefill_graph_execute(
            h,
            s as i64,
            hin.as_ptr() as i64,
            out.as_mut_ptr() as i64,
            kv.as_mut_ptr() as i64,
        )
    };
    println!(
        "execute rc={rc} in {:.2}s  out[0]={} kv[0]={}",
        t0.elapsed().as_secs_f64(),
        out[0],
        kv[0]
    );
    let t1 = std::time::Instant::now();
    let _ = unsafe {
        nue_plugins::prefill_graph::nue_prefill_graph_execute(
            h,
            s as i64,
            hin.as_ptr() as i64,
            out.as_mut_ptr() as i64,
            kv.as_mut_ptr() as i64,
        )
    };
    println!("warm execute in {:.3}s", t1.elapsed().as_secs_f64());
}
