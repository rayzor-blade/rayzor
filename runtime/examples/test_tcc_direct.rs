/// Direct test of TCC runtime wrappers
#[cfg(feature = "tcc-runtime")]
fn main() {
    use rayzor_runtime::haxe_string::HaxeString;
    use rayzor_runtime::tinycc_runtime::*;

    println!("Creating TCC state...");
    let state = rayzor_tcc_create();
    println!("  state null? {}", state.is_null());

    // Construct HaxeString on the stack
    let mut code_buf = b"int add(int a, int b) { return a + b; }".to_vec();
    let hs_code = HaxeString::from_parts(code_buf.as_mut_ptr(), code_buf.len(), code_buf.len());

    println!("Compiling...");
    let ok = rayzor_tcc_compile(state, &hs_code as *const HaxeString);
    println!("  result: {}", ok);

    println!("Relocating...");
    let relocated = rayzor_tcc_relocate(state);
    println!("  result: {}", relocated);

    let mut name_buf = b"add".to_vec();
    let hs_name = HaxeString::from_parts(name_buf.as_mut_ptr(), name_buf.len(), name_buf.len());
    let sym = rayzor_tcc_get_symbol(state, &hs_name as *const HaxeString);
    println!("  add addr: {}", sym);

    if sym != 0 {
        let add_fn: extern "C" fn(i32, i32) -> i32 = unsafe { std::mem::transmute(sym as usize) };
        let result = add_fn(3, 4);
        println!("  add(3, 4) = {}", result);
    }

    rayzor_tcc_delete(state);
    println!("DONE");
}

#[cfg(not(feature = "tcc-runtime"))]
fn main() {
    println!("TCC runtime not enabled. Run with: --features tcc-runtime");
}
