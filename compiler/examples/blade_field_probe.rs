//! Dump a .blade artifact's decoded contents so a raw byte offset can be
//! attributed to the struct field that produced it.

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: blade_field_probe <file.blade>");
    let (mir, meta, symbols, maps) =
        compiler::ir::blade::load_blade(&path).expect("failed to load blade");
    println!("=== METADATA ===\n{:#?}", meta);
    println!("=== MIR ===\n{:#?}", mir);
    println!("=== SYMBOLS ===\n{:#?}", symbols);
    println!("=== CACHED MAPS ===\n{:#?}", maps);
}
