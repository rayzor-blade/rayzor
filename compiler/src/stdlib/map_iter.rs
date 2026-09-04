use crate::ir::instructions::CompareOp;
use crate::ir::mir_builder::MirBuilder;
use crate::ir::{CallingConvention, InlineHint, IrType};

/// The extern maps declare `keys()` and `iterator()` and bind them to nothing.
/// Each becomes a runtime iterator over the array the runtime already builds,
/// the same object `Array.iterator()` returns, so the same `hasNext`/`next`
/// entry points serve it.
pub fn build_map_iterators(builder: &mut MirBuilder) {
    let ptr_void = IrType::Ptr(Box::new(IrType::Void));
    for rt in ["stringmap", "intmap", "objectmap"] {
        for kind in ["keys", "values"] {
            let to_array = format!("haxe_{rt}_{kind}_to_array");
            if builder.get_function_by_name(&to_array).is_none() {
                let f = builder
                    .begin_function(&to_array)
                    .param("map", ptr_void.clone())
                    .returns(ptr_void.clone())
                    .calling_convention(CallingConvention::C)
                    .build();
                builder.mark_as_extern(f);
            }
            let f = builder
                .begin_function(&format!("{rt}_{kind}_iterator"))
                .param("map", ptr_void.clone())
                .returns(ptr_void.clone())
                .calling_convention(CallingConvention::C)
                .build();
            builder.set_current_function(f);
            let entry = builder.create_block("entry");
            builder.set_insert_point(entry);
            let map = builder.get_param(0);
            let to_array_fn = builder
                .get_function_by_name(&to_array)
                .expect("map to_array extern declared above");
            let arr = builder
                .call(to_array_fn, vec![map])
                .expect("keys/values array");
            let array_iterator = builder
                .get_function_by_name("array_iterator")
                .expect("array_iterator is built before the map iterators");
            let it = builder
                .call(array_iterator, vec![arr])
                .expect("iterator over the array");
            builder.ret(Some(it));
        }
    }
}
