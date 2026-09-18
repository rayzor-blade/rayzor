use parser::{ErrorFormatter, TypeDeclaration, parse_haxe_file_with_diagnostics};

fn main() {
    println!("Testing @:coreType metadata parsing...");

    let stdtypes_sample = r#"
/**
	The standard Void type. Only `null` values can be of the type `Void`.
**/
@:coreType abstract Void {}

/**
	The standard Boolean type, which can either be `true` or `false`.
**/
@:coreType @:notNull abstract Bool {}
"#;

    match parse_haxe_file_with_diagnostics("StdTypesTest.hx", stdtypes_sample) {
        Ok(parse_result) => {
            println!("✓ Successfully parsed metadata-prefixed declarations!");
            println!(
                "Parsed {} declarations",
                parse_result.file.declarations.len()
            );

            if !parse_result.diagnostics.is_empty() {
                println!("Diagnostics found:");
                let formatter = ErrorFormatter::new(); // No colors
                let formatted_diagnostics = formatter
                    .format_diagnostics(&parse_result.diagnostics, &parse_result.source_map);
                println!("{}", formatted_diagnostics);
            } else {
                println!("No diagnostics - clean parse!");
            }

            // Print parsed declarations
            for (i, decl) in parse_result.file.declarations.iter().enumerate() {
                match decl {
                    TypeDeclaration::Abstract(type_decl) => {
                        println!("Declaration {}: {:?}", i + 1, type_decl.name);
                    }
                    _ => {
                        println!("Declaration {}: Other type", i + 1);
                    }
                }
            }
        }
        Err(error_output) => {
            println!("✗ Parse failed:");
            println!("{}", error_output);
        }
    }
}
