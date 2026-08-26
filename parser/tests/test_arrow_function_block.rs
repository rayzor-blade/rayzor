//! Test arrow functions with block bodies in various contexts

use parser::parse_haxe_file;

#[test]
fn test_simple_arrow_with_block() {
    let source = r#"
class Main {
    static function main() {
        var handle = spawn(() -> {
            trace("hello");
        });
    }
}
"#;

    let result = parse_haxe_file("test.hx", source, true);
    assert!(
        result.is_ok(),
        "Failed to parse simple arrow function with block: {:?}",
        result.err()
    );
    let file = result.unwrap();
    assert_eq!(
        file.declarations.len(),
        1,
        "Should have 1 class declaration"
    );
}

#[test]
fn test_arrow_with_multiple_statements() {
    let source = r#"
class Main {
    static function main() {
        var handle = spawn(() -> {
            var x = 1;
            var y = 2;
            trace(x + y);
        });
    }
}
"#;

    let result = parse_haxe_file("test.hx", source, true);
    assert!(
        result.is_ok(),
        "Failed to parse arrow function with multiple statements: {:?}",
        result.err()
    );
    let file = result.unwrap();
    assert_eq!(
        file.declarations.len(),
        1,
        "Should have 1 class declaration"
    );
}

#[test]
fn test_arrow_with_method_calls() {
    let source = r#"
class Main {
    static function main() {
        var handle = spawn(() -> {
            var guard = obj.get().lock();
            var c = guard.get();
            c.count += 1;
            guard.unlock();
        });
    }
}
"#;

    let result = parse_haxe_file("test.hx", source, true);
    assert!(
        result.is_ok(),
        "Failed to parse arrow function with method calls: {:?}",
        result.err()
    );
    let file = result.unwrap();
    assert_eq!(
        file.declarations.len(),
        1,
        "Should have 1 class declaration"
    );
}

#[test]
fn test_unparenthesized_single_parameter_arrow() {
    let source = r#"
class Main {
    static function main() {
        var f = value -> value + 1;
        trace(f(1));
    }
}
"#;

    let result = parse_haxe_file("test.hx", source, true);
    assert!(
        result.is_ok(),
        "Failed to parse unparenthesized arrow function: {:?}",
        result.err()
    );
}

#[test]
fn test_unparenthesized_arrow_in_null_coalesce() {
    let source = r#"
class Main {
    static function make(?f:Int->Int) {
        return f ?? value -> value + 1;
    }
}
"#;

    let result = parse_haxe_file("test.hx", source, true);
    assert!(
        result.is_ok(),
        "Failed to parse arrow on a null-coalesce RHS: {:?}",
        result.err()
    );
}

#[test]
fn test_arrow_exact_test_combined_pattern() {
    let source = r#"package test;

import rayzor.concurrent.Thread;
import rayzor.concurrent.Channel;
import rayzor.concurrent.Arc;
import rayzor.concurrent.Mutex;

@:derive([Send, Sync])
class SharedCounter {
    public var count: Int;
    public function new() { this.count = 0; }
}

class Main {
    static function main() {
        var counter = Arc.init(Mutex.init(new SharedCounter()));
        var ch = Channel.init(5);

        var counter_clone = counter.clone();
        var handle = Thread.spawn(() -> {
            var guard = counter_clone.get().lock();
            var c = guard.get();
            c.count += 1;
            guard.unlock();

            ch.send(c.count);
        });

        var result = ch.receive();
        handle.join();
    }
}
"#;

    let result = parse_haxe_file("test.hx", source, true);
    assert!(
        result.is_ok(),
        "Failed to parse exact test_combined pattern: {:?}",
        result.err()
    );
    let file = result.unwrap();
    assert_eq!(
        file.declarations.len(),
        2,
        "Should have 2 class declarations"
    );
}

#[test]
fn test_arrow_simplified_test_combined() {
    let source = r#"
class Main {
    static function main() {
        var handle = Thread.spawn(() -> {
            var guard = obj.lock();
            guard.unlock();
        });
    }
}
"#;

    let result = parse_haxe_file("test.hx", source, true);
    assert!(
        result.is_ok(),
        "Failed to parse simplified test_combined: {:?}",
        result.err()
    );
    let file = result.unwrap();
    assert_eq!(
        file.declarations.len(),
        1,
        "Should have 1 class declaration"
    );
}
