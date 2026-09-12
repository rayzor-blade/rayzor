use compiler::ir::IrType;
use compiler::pipeline::*;

/// `new(left, item)` storing into typed fields is ordinary Haxe: each
/// parameter takes its field's type. Left Dynamic, every construction
/// boxes its arguments — one heap box per field per object.
#[test]
fn unannotated_constructor_params_take_their_field_types() {
    let source = r#"
class Node {
    var left:Node;
    var item:Int;
    public function new(left, item) {
        this.left = left;
        this.item = item;
    }
}

class Main {
    static function main() {
        var n = new Node(null, 3);
        trace(n.item);
    }
}
"#;
    let mut pipeline = HaxeCompilationPipeline::new();
    let result = pipeline.compile_file("node.hx", source);
    assert!(result.errors.is_empty(), "{:?}", result.errors);

    let ctor = result
        .mir_modules
        .iter()
        .flat_map(|m| m.functions.values())
        .find(|f| f.qualified_name.as_deref() == Some("Node.new") || f.name == "Node.new")
        .expect("Node.new is lowered");
    let params: Vec<&IrType> = ctor.signature.parameters.iter().map(|p| &p.ty).collect();
    // this, left, item
    assert_eq!(params.len(), 3, "{:?}", params);
    assert!(matches!(params[1], IrType::Ptr(_)), "left: {:?}", params[1]);
    assert_eq!(*params[2], IrType::I32, "item: {:?}", params[2]);
}
