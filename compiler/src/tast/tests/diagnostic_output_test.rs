#[cfg(test)]
mod diagnostic_output_tests {
    use crate::pipeline::compile_haxe_source;

    #[test]
    fn test_override_diagnostic_output() {
        let haxe_code = r#"
class Animal {
    public function new() {}

    public function makeSound():String {
        return "generic sound";
    }

    public function move(distance:Float):Void {
        trace("Moving " + distance + " meters");
    }
}

class Dog extends Animal {
    public function new() {
        super();
    }

    // Missing override modifier
    public function makeSound():String {
        return "Woof!";
    }

    // Has override but wrong signature
    override public function move(distance:Int):Void {
        trace("Dog running " + distance + " meters");
    }
}

class Cat extends Animal {
    public function new() {
        super();
    }

    // Invalid override - no such method
    override public function purr():String {
        return "Purrr...";
    }
}
        "#;

        let result = compile_haxe_source(haxe_code);

        println!("\n=== Compiler Diagnostic Output ===\n");
        println!("Total diagnostics: {}\n", result.errors.len());

        // Just print the raw diagnostic messages as they come from the compiler
        for (i, error) in result.errors.iter().enumerate() {
            println!("Diagnostic {}:\n{}\n", i + 1, error.message);
        }

        // Verify we have the expected error types
        assert!(
            result.errors.iter().any(|e| e.message.contains("E1010")),
            "Should have missing override error (E1010)"
        );
        assert!(
            result.errors.iter().any(|e| e.message.contains("E1011")),
            "Should have invalid override error (E1011)"
        );
        assert!(
            result
                .errors
                .iter()
                .any(|e| e.message.contains("signature")),
            "Should have signature mismatch error"
        );
    }

    #[test]
    fn test_interface_diagnostic_output() {
        let haxe_code = r#"
interface IDrawable {
    function draw(x:Int, y:Int):Void;
    function getColor():String;
}

interface IUpdatable {
    function update(deltaTime:Float):Void;
}

class Widget implements IDrawable implements IUpdatable {
    public function new() {}

    // Missing draw() method

    // Wrong signature for getColor
    public function getColor():Int {
        return 0xFF0000;
    }

    // Correct implementation
    public function update(deltaTime:Float):Void {
        // Update logic
    }
}
        "#;

        let result = compile_haxe_source(haxe_code);

        println!("\n=== Interface Implementation Diagnostics ===\n");

        for error in &result.errors {
            println!("{}\n", error.message);
        }

        // Just verify we have errors
        assert!(
            !result.errors.is_empty(),
            "Should have interface implementation errors"
        );
    }

    #[test]
    fn test_all_diagnostic_codes() {
        let haxe_code = r#"
// Base class
class Shape {
    public function new() {}
    public function getArea():Float { return 0.0; }
}

// Interface
interface IColorable {
    function getColor():String;
    function setColor(color:String):Void;
}

// Class with multiple issues
class Circle extends Shape implements IColorable {
    public function new() {
        super();
    }

    // E1010: Missing override modifier
    public function getArea():Float {
        return 3.14159 * 10 * 10;
    }

    // E1011: Invalid override
    override public function getRadius():Float {
        return 10.0;
    }

    // E1008: Missing interface method (setColor)

    // E1009: Wrong signature for interface method
    public function getColor():Int {
        return 0xFF0000;
    }
}
        "#;

        let result = compile_haxe_source(haxe_code);

        println!("\n=== All Error Codes Demonstration ===\n");

        // Group by error code
        let mut errors_by_code: std::collections::BTreeMap<String, Vec<&_>> =
            std::collections::BTreeMap::new();

        for error in &result.errors {
            if let Some(code_match) = error.message.find("[E") {
                if let Some(end) = error.message[code_match..].find(']') {
                    let code = error.message[code_match..code_match + end + 1].to_string();
                    errors_by_code
                        .entry(code)
                        .or_insert_with(Vec::new)
                        .push(error);
                }
            }
        }

        for (code, errors) in errors_by_code.iter() {
            println!("Error Code {}: {} occurrence(s)", code, errors.len());
            for error in errors {
                // Print just the first line of each error
                if let Some(first_line) = error.message.lines().next() {
                    println!("  - {}", first_line);
                }
            }
            println!();
        }

        assert!(result.errors.len() >= 3, "Should have multiple error types");
    }
}
