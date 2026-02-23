//! Benchmarks modeled after real-world Haxe macro usage patterns.
//!
//! Each benchmark emulates a common macro pattern from the Haxe ecosystem:
//! - tink-style serialization: iterate fields, dispatch to type-specific handlers
//! - @:build accessor generation: generate getter/setter/validator per field
//! - Compile-time lookup table construction (hash maps, dispatch tables)
//! - Expression rewriting chains (assert, logging, tracing macros)
//! - Deep cross-class macro helper dispatch (tink architecture pattern)

use compiler::macro_system::{expand_macros, expand_macros_with_class_registry, ClassRegistry};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::fmt::Write;
use std::time::Duration;

fn parse_haxe(source: &str) -> parser::HaxeFile {
    parser::parse_haxe_file("bench.hx", source, false).expect("parse should succeed")
}

// =====================================================
// Pattern 1: tink-style Serialization Macro
//
// Real pattern: tink_json inspects class fields at compile time,
// determines serialization strategy per field type, and generates
// a toJSON() body that serializes each field with the right encoder.
//
// We model this: macro iterates over N "field descriptors" (objects),
// calls type-specific encoder selection, builds cumulative result.
// =====================================================

fn gen_tink_serialize(num_fields: usize) -> String {
    let mut s = String::new();
    writeln!(s, "import haxe.macro.Context;").unwrap();

    // Type descriptor — holds field metadata (like tink's FieldInfo)
    write!(
        s,
        "class FieldDescriptor {{
    var name:String;
    var typeKind:Int;
    var isNullable:Bool;
    var depth:Int;
    function new(name:String, typeKind:Int, isNullable:Bool, depth:Int) {{
        this.name = name;
        this.typeKind = typeKind;
        this.isNullable = isNullable;
        this.depth = depth;
    }}
}}
"
    )
    .unwrap();

    // Encoder selection — like tink's internal resolver that picks
    // the right serializer based on field type
    write!(
        s,
        "class EncoderResolver {{
    static function costForType(typeKind:Int, depth:Int):Int {{
        if (typeKind == 0) {{ return 1; }}
        if (typeKind == 1) {{ return 2; }}
        if (typeKind == 2) {{ return 4 + depth; }}
        if (typeKind == 3) {{
            var base = 8;
            var i = 0;
            while (i < depth) {{
                base = base + i * 2 + 1;
                i = i + 1;
            }}
            return base;
        }}
        return 16;
    }}
    static function selectEncoder(field:FieldDescriptor):Int {{
        var cost = EncoderResolver.costForType(field.typeKind, field.depth);
        if (field.isNullable) {{
            cost = cost + 3;
        }}
        return cost;
    }}
}}
"
    )
    .unwrap();

    // The macro that simulates tink_json's build process
    writeln!(s, "class SerializeMacro {{").unwrap();
    writeln!(
        s,
        "    macro static function buildSerializer(fieldCount:Int):haxe.macro.Expr {{"
    )
    .unwrap();
    writeln!(s, "        var totalCost = 0;").unwrap();
    writeln!(s, "        var i = 0;").unwrap();
    writeln!(s, "        while (i < fieldCount) {{").unwrap();
    writeln!(s, "            var typeKind = i * 7 + 3;").unwrap();
    writeln!(s, "            typeKind = typeKind - (typeKind / 5) * 5;").unwrap();
    writeln!(s, "            var nullable = i * 3 + 1;").unwrap();
    writeln!(s, "            nullable = nullable - (nullable / 2) * 2;").unwrap();
    writeln!(
        s,
        "            var field = new FieldDescriptor(\"field\", typeKind, nullable == 1, i / 3);"
    )
    .unwrap();
    writeln!(
        s,
        "            var cost = EncoderResolver.selectEncoder(field);"
    )
    .unwrap();
    writeln!(s, "            totalCost = totalCost + cost;").unwrap();
    writeln!(s, "            i = i + 1;").unwrap();
    writeln!(s, "        }}").unwrap();
    writeln!(s, "        return macro $v{{totalCost}};").unwrap();
    writeln!(s, "    }}").unwrap();
    writeln!(s, "}}").unwrap();

    writeln!(s, "class Main {{").unwrap();
    writeln!(s, "    static function main() {{").unwrap();
    writeln!(
        s,
        "        trace(SerializeMacro.buildSerializer({num_fields}));"
    )
    .unwrap();
    writeln!(s, "    }}").unwrap();
    writeln!(s, "}}").unwrap();
    s
}

// =====================================================
// Pattern 2: @:build Accessor Generation
//
// Real pattern: ORM/ECS frameworks use @:build macros to read
// class fields and generate getters, setters, dirty tracking,
// change notification, and validation methods per field.
//
// We model this: macro processes N fields, for each one it
// creates a FieldMeta object, computes validation rules,
// generates accessor costs. Multiple helper classes involved.
// =====================================================

fn gen_build_accessors(num_fields: usize) -> String {
    let mut s = String::new();
    writeln!(s, "import haxe.macro.Context;").unwrap();

    write!(
        s,
        "class FieldMeta {{
    var name:String;
    var index:Int;
    var validationCost:Int;
    function new(name:String, index:Int) {{
        this.name = name;
        this.index = index;
        this.validationCost = 0;
    }}
    function computeValidation():Int {{
        var cost = 1;
        var i = 0;
        while (i < this.index + 1) {{
            cost = cost * 2 + 1;
            if (cost > 100) {{
                cost = cost - 90;
            }}
            i = i + 1;
        }}
        this.validationCost = cost;
        return cost;
    }}
}}
class AccessorBuilder {{
    var getterCost:Int;
    var setterCost:Int;
    var dirtyTrackCost:Int;
    function new() {{
        this.getterCost = 0;
        this.setterCost = 0;
        this.dirtyTrackCost = 0;
    }}
    function addGetter(meta:FieldMeta):Int {{
        this.getterCost = this.getterCost + 2 + meta.index;
        return this.getterCost;
    }}
    function addSetter(meta:FieldMeta):Int {{
        this.setterCost = this.setterCost + 3 + meta.validationCost;
        return this.setterCost;
    }}
    function addDirtyTrack(meta:FieldMeta):Int {{
        this.dirtyTrackCost = this.dirtyTrackCost + 1;
        return this.dirtyTrackCost;
    }}
    function totalCost():Int {{
        return this.getterCost + this.setterCost + this.dirtyTrackCost;
    }}
}}
"
    )
    .unwrap();

    writeln!(s, "class BuildMacro {{").unwrap();
    writeln!(
        s,
        "    macro static function generateAccessors(fieldCount:Int):haxe.macro.Expr {{"
    )
    .unwrap();
    writeln!(s, "        var builder = new AccessorBuilder();").unwrap();
    writeln!(s, "        var i = 0;").unwrap();
    writeln!(s, "        while (i < fieldCount) {{").unwrap();
    writeln!(s, "            var meta = new FieldMeta(\"f\", i);").unwrap();
    writeln!(s, "            meta.computeValidation();").unwrap();
    writeln!(s, "            builder.addGetter(meta);").unwrap();
    writeln!(s, "            builder.addSetter(meta);").unwrap();
    writeln!(s, "            builder.addDirtyTrack(meta);").unwrap();
    writeln!(s, "            i = i + 1;").unwrap();
    writeln!(s, "        }}").unwrap();
    writeln!(s, "        var result = builder.totalCost();").unwrap();
    writeln!(s, "        return macro $v{{result}};").unwrap();
    writeln!(s, "    }}").unwrap();
    writeln!(s, "}}").unwrap();

    writeln!(s, "class Main {{").unwrap();
    writeln!(s, "    static function main() {{").unwrap();
    writeln!(
        s,
        "        trace(BuildMacro.generateAccessors({num_fields}));"
    )
    .unwrap();
    writeln!(s, "    }}").unwrap();
    writeln!(s, "}}").unwrap();
    s
}

// =====================================================
// Pattern 3: Compile-Time Lookup Table
//
// Real pattern: Macros that pre-compute dispatch tables, string→int
// maps, hash lookups at compile time. Used in parsers, protocol
// decoders, enum→string converters.
//
// We model this: macro builds a hash table of N entries at compile
// time, computing hash codes and resolving collisions.
// =====================================================

fn gen_lookup_table(num_entries: usize) -> String {
    let mut s = String::new();
    writeln!(s, "import haxe.macro.Context;").unwrap();

    write!(
        s,
        "class HashEntry {{
    var key:Int;
    var value:Int;
    var next:Int;
    function new(key:Int, value:Int) {{
        this.key = key;
        this.value = value;
        this.next = -1;
    }}
    function hash():Int {{
        var h = this.key;
        h = h + (h * 65536);
        var shifted = h / 16;
        if (h < 0) {{ shifted = 0 - ((0 - h) / 16); }}
        h = h - (shifted * 16) + shifted;
        h = h + (h * 8);
        return h;
    }}
}}
class TableBuilder {{
    var size:Int;
    var collisions:Int;
    var totalHash:Int;
    function new(size:Int) {{
        this.size = size;
        this.collisions = 0;
        this.totalHash = 0;
    }}
    function insert(entry:HashEntry):Int {{
        var h = entry.hash();
        if (h < 0) {{ h = 0 - h; }}
        var bucket = h - (h / this.size) * this.size;
        this.totalHash = this.totalHash + h;
        if (bucket < this.size / 2) {{
            this.collisions = this.collisions + 1;
        }}
        return bucket;
    }}
    function loadFactor():Int {{
        if (this.size == 0) {{ return 0; }}
        return this.collisions * 100 / this.size;
    }}
}}
"
    )
    .unwrap();

    writeln!(s, "class LookupMacro {{").unwrap();
    writeln!(
        s,
        "    macro static function buildTable(entries:Int):haxe.macro.Expr {{"
    )
    .unwrap();
    writeln!(s, "        var tableSize = entries * 2;").unwrap();
    writeln!(s, "        var builder = new TableBuilder(tableSize);").unwrap();
    writeln!(s, "        var i = 0;").unwrap();
    writeln!(s, "        while (i < entries) {{").unwrap();
    writeln!(
        s,
        "            var entry = new HashEntry(i * 17 + 5, i * i);"
    )
    .unwrap();
    writeln!(s, "            builder.insert(entry);").unwrap();
    writeln!(s, "            i = i + 1;").unwrap();
    writeln!(s, "        }}").unwrap();
    writeln!(
        s,
        "        var result = builder.totalHash + builder.loadFactor();"
    )
    .unwrap();
    writeln!(s, "        return macro $v{{result}};").unwrap();
    writeln!(s, "    }}").unwrap();
    writeln!(s, "}}").unwrap();

    writeln!(s, "class Main {{").unwrap();
    writeln!(s, "    static function main() {{").unwrap();
    writeln!(s, "        trace(LookupMacro.buildTable({num_entries}));").unwrap();
    writeln!(s, "    }}").unwrap();
    writeln!(s, "}}").unwrap();
    s
}

// =====================================================
// Pattern 4: Expression Rewriting (assert/log macros)
//
// Real pattern: assert(x > 0) rewrites to code that captures
// the expression text, file, line, and evaluates the check.
// Used heavily — 50-200+ assert/log calls per file in test suites.
//
// We model this: many independent macro call sites, each doing
// non-trivial compile-time work (hash computation per call).
// =====================================================

fn gen_assert_pattern(num_asserts: usize) -> String {
    let mut s = String::new();
    write!(
        s,
        "import haxe.macro.Context;
class AssertMacro {{
    macro static function check(val:Int, line:Int):haxe.macro.Expr {{
        var hash = val * 2654435761;
        if (hash < 0) {{ hash = 0 - hash; }}
        hash = hash - (hash / 1000000) * 1000000;
        var result = hash + line;
        return macro $v{{result}};
    }}
}}
class Main {{
    static function main() {{
"
    )
    .unwrap();

    for i in 0..num_asserts {
        writeln!(
            s,
            "        trace(AssertMacro.check({}, {}));",
            i * 7 + 3,
            i + 1
        )
        .unwrap();
    }

    write!(
        s,
        "    }}
}}
"
    )
    .unwrap();
    s
}

// =====================================================
// Pattern 5: Deep Cross-Class Macro Helper Chain
//
// Real pattern: tink and other macro libraries have layered
// helper architectures: MacroFrontend → TypeResolver →
// ExprBuilder → CodeEmitter. Each layer constructs objects,
// calls methods, and passes results to the next.
//
// We model this: N-deep chain of helper classes. Each level
// constructs the next level's helper, calls transform methods,
// and accumulates results. Stresses ClassRegistry + call stack.
// =====================================================

fn gen_helper_chain(depth: usize) -> String {
    let mut s = String::new();
    writeln!(s, "import haxe.macro.Context;").unwrap();

    for d in 0..depth {
        writeln!(s, "class Layer{d} {{").unwrap();
        writeln!(s, "    var state:Int;").unwrap();
        writeln!(s, "    var tag:Int;").unwrap();
        writeln!(s, "    function new(seed:Int) {{").unwrap();
        writeln!(s, "        this.state = seed * {} + 1;", d + 1).unwrap();
        writeln!(s, "        this.tag = {d};").unwrap();
        writeln!(s, "    }}").unwrap();

        // Each layer has a process() that does work and delegates deeper
        writeln!(s, "    function process(input:Int):Int {{").unwrap();
        writeln!(s, "        var work = this.state + input;").unwrap();
        writeln!(s, "        var j = 0;").unwrap();
        writeln!(s, "        while (j < this.tag + 1) {{").unwrap();
        writeln!(s, "            work = work * 3 + j + 1;").unwrap();
        writeln!(s, "            if (work > 10000) {{").unwrap();
        writeln!(s, "                work = work - 9000;").unwrap();
        writeln!(s, "            }}").unwrap();
        writeln!(s, "            j = j + 1;").unwrap();
        writeln!(s, "        }}").unwrap();

        if d < depth - 1 {
            // Delegate to next layer
            writeln!(s, "        var next = new Layer{}(work);", d + 1).unwrap();
            writeln!(s, "        return next.process(work);").unwrap();
        } else {
            writeln!(s, "        return work;").unwrap();
        }
        writeln!(s, "    }}").unwrap();
        writeln!(s, "}}").unwrap();
    }

    // Macro that kicks off the chain
    writeln!(s, "class ChainMacro {{").unwrap();
    writeln!(
        s,
        "    macro static function run(seed:Int):haxe.macro.Expr {{"
    )
    .unwrap();
    writeln!(s, "        var start = new Layer0(seed);").unwrap();
    writeln!(s, "        var result = start.process(seed);").unwrap();
    writeln!(s, "        return macro $v{{result}};").unwrap();
    writeln!(s, "    }}").unwrap();
    writeln!(s, "}}").unwrap();

    writeln!(s, "class Main {{").unwrap();
    writeln!(s, "    static function main() {{").unwrap();
    writeln!(s, "        trace(ChainMacro.run(42));").unwrap();
    writeln!(s, "    }}").unwrap();
    writeln!(s, "}}").unwrap();
    s
}

// =====================================================
// Pattern 6: Multi-Entity Schema Code Generation
//
// Real pattern: ORM / protocol buffer macros process multiple
// message types, each with multiple fields. For each entity
// they generate serializer, deserializer, validator, and differ.
// A project might have 20-50 entities × 5-20 fields each.
//
// We model this: macro loops over E entities × F fields,
// constructing descriptor objects and calling method chains.
// =====================================================

fn gen_multi_entity_schema(num_entities: usize, fields_per_entity: usize) -> String {
    let mut s = String::new();
    writeln!(s, "import haxe.macro.Context;").unwrap();

    write!(
        s,
        "class SchemaField {{
    var name:String;
    var typeId:Int;
    var offset:Int;
    function new(name:String, typeId:Int, offset:Int) {{
        this.name = name;
        this.typeId = typeId;
        this.offset = offset;
    }}
    function wireSize():Int {{
        if (this.typeId == 0) {{ return 4; }}
        if (this.typeId == 1) {{ return 8; }}
        if (this.typeId == 2) {{ return 1; }}
        return 16 + this.offset;
    }}
    function validationCost():Int {{
        var c = 1;
        var i = 0;
        var limit = this.typeId + 1;
        while (i < limit) {{
            c = c + this.typeId + 1;
            i = i + 1;
        }}
        return c;
    }}
}}
class EntityBuilder {{
    var wireTotal:Int;
    var validTotal:Int;
    var fieldCount:Int;
    var entityHash:Int;
    function new(entityId:Int) {{
        this.wireTotal = 0;
        this.validTotal = 0;
        this.fieldCount = 0;
        this.entityHash = entityId * 31 + 17;
    }}
    function addField(field:SchemaField):Int {{
        this.wireTotal = this.wireTotal + field.wireSize();
        this.validTotal = this.validTotal + field.validationCost();
        this.fieldCount = this.fieldCount + 1;
        this.entityHash = this.entityHash * 31 + field.typeId;
        return this.fieldCount;
    }}
    function finalize():Int {{
        return this.entityHash + this.wireTotal * 100 + this.validTotal;
    }}
}}
"
    )
    .unwrap();

    writeln!(s, "class SchemaMacro {{").unwrap();
    writeln!(
        s,
        "    macro static function generate(entities:Int, fields:Int):haxe.macro.Expr {{"
    )
    .unwrap();
    writeln!(s, "        var grandTotal = 0;").unwrap();
    writeln!(s, "        var e = 0;").unwrap();
    writeln!(s, "        while (e < entities) {{").unwrap();
    writeln!(s, "            var builder = new EntityBuilder(e);").unwrap();
    writeln!(s, "            var f = 0;").unwrap();
    writeln!(s, "            while (f < fields) {{").unwrap();
    // Vary typeId across fields
    writeln!(s, "                var tid = (f + e) * 7 + 3;").unwrap();
    writeln!(s, "                tid = tid - (tid / 4) * 4;").unwrap();
    writeln!(
        s,
        "                var field = new SchemaField(\"f\", tid, f * 4 + e);"
    )
    .unwrap();
    writeln!(s, "                builder.addField(field);").unwrap();
    writeln!(s, "                f = f + 1;").unwrap();
    writeln!(s, "            }}").unwrap();
    writeln!(
        s,
        "            grandTotal = grandTotal + builder.finalize();"
    )
    .unwrap();
    writeln!(s, "            e = e + 1;").unwrap();
    writeln!(s, "        }}").unwrap();
    writeln!(s, "        return macro $v{{grandTotal}};").unwrap();
    writeln!(s, "    }}").unwrap();
    writeln!(s, "}}").unwrap();

    writeln!(s, "class Main {{").unwrap();
    writeln!(s, "    static function main() {{").unwrap();
    writeln!(
        s,
        "        trace(SchemaMacro.generate({num_entities}, {fields_per_entity}));"
    )
    .unwrap();
    writeln!(s, "    }}").unwrap();
    writeln!(s, "}}").unwrap();
    s
}

// =====================================================
// Benchmarks
// =====================================================

fn bench_tink_serialize(c: &mut Criterion) {
    let mut group = c.benchmark_group("tink_serialize");
    group.measurement_time(Duration::from_secs(8));

    for n in [5, 10, 25, 50, 100] {
        let source = gen_tink_serialize(n);
        let file = parse_haxe(&source);

        group.bench_with_input(BenchmarkId::new("fields", n), &file, |b, file| {
            b.iter(|| {
                let mut cr = ClassRegistry::new();
                cr.register_file(file);
                let result = expand_macros_with_class_registry(black_box(file.clone()), cr);
                black_box(result.expansions_count)
            });
        });
    }
    group.finish();
}

fn bench_build_accessors(c: &mut Criterion) {
    let mut group = c.benchmark_group("build_accessors");
    group.measurement_time(Duration::from_secs(8));

    for n in [5, 10, 25, 50, 100] {
        let source = gen_build_accessors(n);
        let file = parse_haxe(&source);

        group.bench_with_input(BenchmarkId::new("fields", n), &file, |b, file| {
            b.iter(|| {
                let mut cr = ClassRegistry::new();
                cr.register_file(file);
                let result = expand_macros_with_class_registry(black_box(file.clone()), cr);
                black_box(result.expansions_count)
            });
        });
    }
    group.finish();
}

fn bench_lookup_table(c: &mut Criterion) {
    let mut group = c.benchmark_group("lookup_table");
    group.measurement_time(Duration::from_secs(8));

    for n in [10, 50, 100, 500] {
        let source = gen_lookup_table(n);
        let file = parse_haxe(&source);

        group.bench_with_input(BenchmarkId::new("entries", n), &file, |b, file| {
            b.iter(|| {
                let mut cr = ClassRegistry::new();
                cr.register_file(file);
                let result = expand_macros_with_class_registry(black_box(file.clone()), cr);
                black_box(result.expansions_count)
            });
        });
    }
    group.finish();
}

fn bench_assert_pattern(c: &mut Criterion) {
    let mut group = c.benchmark_group("assert_rewrite");
    group.measurement_time(Duration::from_secs(8));

    for n in [10, 50, 100, 200] {
        let source = gen_assert_pattern(n);
        let file = parse_haxe(&source);

        group.bench_with_input(BenchmarkId::new("sites", n), &file, |b, file| {
            b.iter(|| {
                let result = expand_macros(black_box(file.clone()));
                black_box(result.expansions_count)
            });
        });
    }
    group.finish();
}

fn bench_helper_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("helper_chain");
    group.measurement_time(Duration::from_secs(8));

    for depth in [3, 5, 8, 12, 16] {
        let source = gen_helper_chain(depth);
        let file = parse_haxe(&source);

        group.bench_with_input(BenchmarkId::new("depth", depth), &file, |b, file| {
            b.iter(|| {
                let mut cr = ClassRegistry::new();
                cr.register_file(file);
                let result = expand_macros_with_class_registry(black_box(file.clone()), cr);
                black_box(result.expansions_count)
            });
        });
    }
    group.finish();
}

fn bench_multi_entity_schema(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_entity_schema");
    group.measurement_time(Duration::from_secs(8));

    for (entities, fields) in [(3, 5), (5, 10), (10, 10), (15, 12), (20, 15)] {
        let source = gen_multi_entity_schema(entities, fields);
        let file = parse_haxe(&source);
        let label = format!("{entities}e_{fields}f");

        group.bench_with_input(BenchmarkId::new("schema", &label), &file, |b, file| {
            b.iter(|| {
                let mut cr = ClassRegistry::new();
                cr.register_file(file);
                let result = expand_macros_with_class_registry(black_box(file.clone()), cr);
                black_box(result.expansions_count)
            });
        });
    }
    group.finish();
}

fn bench_registry_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("registry_scan");
    group.measurement_time(Duration::from_secs(5));

    // Cost of scanning files into ClassRegistry (happens before every expansion)
    for depth in [5, 10, 20, 50] {
        let source = gen_helper_chain(depth);
        let file = parse_haxe(&source);

        group.bench_with_input(BenchmarkId::new("classes", depth), &file, |b, file| {
            b.iter(|| {
                let mut cr = ClassRegistry::new();
                cr.register_file(black_box(file));
                black_box(cr.class_count())
            });
        });
    }
    group.finish();
}

fn bench_full_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_pipeline");
    group.measurement_time(Duration::from_secs(10));

    let src = gen_tink_serialize(25);
    group.bench_function("tink_25_fields", |b| {
        b.iter(|| {
            let mut p = compiler::pipeline::HaxeCompilationPipeline::new();
            black_box(p.compile_file("bench.hx", black_box(&src)))
        });
    });

    let src = gen_multi_entity_schema(10, 10);
    group.bench_function("schema_10e_10f", |b| {
        b.iter(|| {
            let mut p = compiler::pipeline::HaxeCompilationPipeline::new();
            black_box(p.compile_file("bench.hx", black_box(&src)))
        });
    });

    let src = gen_helper_chain(8);
    group.bench_function("chain_depth_8", |b| {
        b.iter(|| {
            let mut p = compiler::pipeline::HaxeCompilationPipeline::new();
            black_box(p.compile_file("bench.hx", black_box(&src)))
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_tink_serialize,
    bench_build_accessors,
    bench_lookup_table,
    bench_assert_pattern,
    bench_helper_chain,
    bench_multi_entity_schema,
    bench_registry_scan,
    bench_full_pipeline,
);

criterion_main!(benches);
