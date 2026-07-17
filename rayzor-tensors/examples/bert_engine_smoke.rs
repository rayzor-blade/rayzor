//! Direct smoke of the bert_graph engine FFI (no JIT): load + bucket + execute.
fn main() {
    let dir = std::env::args().nth(1).expect("dir");
    let stem = std::env::args().nth(2).expect("stem");
    let h = unsafe {
        rayzor_tensors::bert_graph::rayzor_bert_graph_load(
            dir.as_ptr() as i64,
            dir.len() as i64,
            stem.as_ptr() as i64,
            stem.len() as i64,
            384,
        )
    };
    println!("load handle={h}");
    let b = rayzor_tensors::bert_graph::rayzor_bert_graph_bucket(h, 20);
    println!("bucket_for(20)={b}");
    let hin = vec![0.01f32; (b as usize) * 384];
    let bias = vec![0f32; b as usize];
    let mut out = vec![0f32; (b as usize) * 384];
    let rc = unsafe {
        rayzor_tensors::bert_graph::rayzor_bert_graph_execute(
            h,
            b,
            hin.as_ptr() as i64,
            bias.as_ptr() as i64,
            out.as_mut_ptr() as i64,
        )
    };
    println!("execute rc={rc} out[0]={}", out[0]);
}
