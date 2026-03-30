fn main() {
    let target = std::env::var("TARGET").unwrap_or_default();
    if target.contains("apple") {
        println!("cargo:rustc-link-lib=framework=Cocoa");
        println!("cargo:rustc-link-lib=framework=AppKit");
    }
}
