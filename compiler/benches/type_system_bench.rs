//! Benchmarks for type system performance optimization validation

use compiler::pipeline::*;
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

fn generate_deep_inheritance(depth: usize) -> String {
    let mut code = String::new();

    // Generate a deep inheritance hierarchy
    for i in 0..depth {
        code.push_str(&format!(
            "class Level{} {} {{\n",
            i,
            if i > 0 {
                format!("extends Level{}", i - 1)
            } else {
                String::new()
            }
        ));
        code.push_str(&format!(
            "    public function method{}():Int {{ return {}; }}\n",
            i, i
        ));
        code.push_str("}\n\n");
    }

    // Generate a test that uses all levels
    code.push_str("class Test {\n");
    code.push_str("    static function main() {\n");
    code.push_str(&format!("        var obj = new Level{}();\n", depth - 1));

    // Call methods from all levels to trigger inheritance resolution
    for i in 0..depth {
        code.push_str(&format!("        obj.method{}();\n", i));
    }

    code.push_str("    }\n");
    code.push_str("}\n");

    code
}

fn generate_many_symbols(count: usize) -> String {
    let mut code = String::new();

    code.push_str("class SymbolTest {\n");

    // Generate many unique symbols
    for i in 0..count {
        code.push_str(&format!("    public var field{}:Int = {};\n", i, i));
    }

    code.push_str("    public function test() {\n");

    // Reference all symbols to trigger resolution
    for i in 0..count {
        code.push_str(&format!("        var local{} = field{};\n", i, i));
    }

    code.push_str("    }\n");
    code.push_str("}\n");

    code
}

fn generate_generic_heavy(instances: usize) -> String {
    let mut code = String::new();

    // Generic class definition
    code.push_str("class Container<T> {\n");
    code.push_str("    public var value:T;\n");
    code.push_str("    public function new(v:T) { this.value = v; }\n");
    code.push_str("    public function get():T { return value; }\n");
    code.push_str("}\n\n");

    code.push_str("class GenericTest {\n");
    code.push_str("    static function main() {\n");

    // Create many generic instances with different types
    for i in 0..instances {
        let type_name = match i % 4 {
            0 => "Int",
            1 => "String",
            2 => "Float",
            _ => "Bool",
        };

        code.push_str(&format!(
            "        var c{} = new Container<{}>(null);\n",
            i, type_name
        ));
    }

    code.push_str("    }\n");
    code.push_str("}\n");

    code
}

fn benchmark_inheritance_resolution(c: &mut Criterion) {
    let mut group = c.benchmark_group("inheritance_resolution");

    for depth in [10, 20, 50, 100].iter() {
        let code = generate_deep_inheritance(*depth);

        group.bench_with_input(BenchmarkId::from_parameter(depth), &code, |b, code| {
            b.iter(|| {
                let mut pipeline = HaxeCompilationPipeline::new();
                let result = pipeline.compile_file("test.hx", black_box(code));
                black_box(result);
            });
        });
    }

    group.finish();
}

fn benchmark_symbol_resolution(c: &mut Criterion) {
    let mut group = c.benchmark_group("symbol_resolution");

    for count in [100, 500, 1000, 5000].iter() {
        let code = generate_many_symbols(*count);

        group.bench_with_input(BenchmarkId::from_parameter(count), &code, |b, code| {
            b.iter(|| {
                let mut pipeline = HaxeCompilationPipeline::new();
                let result = pipeline.compile_file("test.hx", black_box(code));
                black_box(result);
            });
        });
    }

    group.finish();
}

fn benchmark_generic_instantiation(c: &mut Criterion) {
    let mut group = c.benchmark_group("generic_instantiation");

    for instances in [50, 100, 500, 1000].iter() {
        let code = generate_generic_heavy(*instances);

        group.bench_with_input(BenchmarkId::from_parameter(instances), &code, |b, code| {
            b.iter(|| {
                let mut pipeline = HaxeCompilationPipeline::new();
                let result = pipeline.compile_file("test.hx", black_box(code));
                black_box(result);
            });
        });
    }

    group.finish();
}

fn benchmark_type_resolution(c: &mut Criterion) {
    let code = r#"
package com.example.test;

import com.example.utils.*;
import com.example.models.User;
import com.example.models.Product;

class ComplexTypeTest {
    private var users:Array<com.example.models.User>;
    private var products:Map<String, com.example.models.Product>;
    private var handlers:Map<String, Int -> String -> com.example.models.User>;
    
    public function new() {
        this.users = [];
        this.products = new Map();
        this.handlers = new Map();
    }
    
    public function process():Array<com.example.models.Product> {
        var result:Array<com.example.models.Product> = [];
        
        for (user in users) {
            var p:com.example.models.Product = createProduct(user);
            result.push(p);
        }
        
        return result;
    }
    
    private function createProduct(u:com.example.models.User):com.example.models.Product {
        return null;
    }
}
"#;

    c.bench_function("qualified_type_resolution", |b| {
        b.iter(|| {
            let mut pipeline = HaxeCompilationPipeline::new();
            let result = pipeline.compile_file("test.hx", black_box(code));
            black_box(result);
        });
    });
}

criterion_group!(
    benches,
    benchmark_inheritance_resolution,
    benchmark_symbol_resolution,
    benchmark_generic_instantiation,
    benchmark_type_resolution
);

criterion_main!(benches);
