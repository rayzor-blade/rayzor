#[cfg(test)]
mod static_instance_check_tests {
    use crate::pipeline::compile_haxe_source;

    #[test]
    fn test_static_vs_instance_member_checking() {
        let haxe_code = r#"
class MyClass {
    public static var staticField:Int = 42;
    public var instanceField:String = "hello";

    public static function staticMethod():String {
        return "Static method";
    }

    public function instanceMethod():String {
        return "Instance method";
    }
}

class TestStaticAccess {
    public function new() {}

    public function testAccess():Void {
        // Valid static access
        var s1:Int = MyClass.staticField;
        MyClass.staticMethod();

        // Valid instance access
        var obj = new MyClass();
        var i1:String = obj.instanceField;
        obj.instanceMethod();

        // Invalid: accessing static member through instance
        var invalid1:Int = obj.staticField;  // Should error
        obj.staticMethod();  // Should error

        // Invalid: accessing instance member through static context
        var invalid2:String = MyClass.instanceField;  // Should error
        MyClass.instanceMethod();  // Should error
    }
}
        "#;

        let result = compile_haxe_source(haxe_code);

        println!("\n=== Compilation Result ===");
        println!("Total errors: {}", result.errors.len());
        for (i, error) in result.errors.iter().enumerate() {
            println!("\nError {}: {}", i + 1, error.message);
            println!(
                "  Location: {}:{}:{}",
                error.location.file_id, error.location.line, error.location.column
            );
            println!("  Category: {:?}", error.category);
        }

        // Print expected errors for debugging
        println!("\n=== Expected Errors ===");
        println!("1. Line 35: obj.staticField - accessing static field through instance");
        println!("2. Line 36: obj.staticMethod() - accessing static method through instance");
        println!("3. Line 39: MyClass.instanceField - accessing instance field statically");
        println!("4. Line 40: MyClass.instanceMethod() - accessing instance method statically");

        // We expect exactly 4 errors
        assert_eq!(
            result.errors.len(),
            4,
            "Expected 4 static/instance access errors"
        );

        // Check error messages
        let error_messages: Vec<String> = result.errors.iter().map(|e| e.message.clone()).collect();

        // Verify we have the right kinds of errors
        let static_from_instance_errors = error_messages
            .iter()
            .filter(|msg| {
                msg.contains("Static member") && msg.contains("cannot be accessed through instance")
            })
            .count();
        assert_eq!(
            static_from_instance_errors, 2,
            "Expected 2 static-from-instance errors"
        );

        let instance_from_static_errors = error_messages
            .iter()
            .filter(|msg| {
                msg.contains("Instance member")
                    && msg.contains("cannot be accessed from static context")
            })
            .count();
        assert_eq!(
            instance_from_static_errors, 2,
            "Expected 2 instance-from-static errors"
        );
    }

    #[test]
    fn test_static_method_context() {
        let haxe_code = r#"
class StaticContext {
    private static var staticData:Int = 100;
    private var instanceData:String = "data";

    public static function staticWork():Void {
        // Valid: static accessing static
        var x = staticData;

        // Invalid: static method accessing instance member
        var y = instanceData;  // Should error
    }

    public function instanceWork():Void {
        // Valid: instance method can access both
        var x = staticData;
        var y = instanceData;
    }
}
        "#;

        let result = compile_haxe_source(haxe_code);

        // We expect exactly 1 error
        assert_eq!(
            result.errors.len(),
            1,
            "Expected 1 error for instance member access from static context"
        );

        let error = &result.errors[0];
        assert!(error.message.contains("Instance member"));
        assert!(
            error
                .message
                .contains("cannot be accessed from static context")
        );
    }
}
