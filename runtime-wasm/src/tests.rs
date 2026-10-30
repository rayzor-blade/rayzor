use super::*;
use bytes_io::*;
use maps::*;
use reflect_strings::*;
use typed_values::*;

fn text(ptr: i32) -> String {
    let (data, len, _) = unsafe { read_haxe_string(ptr) };
    String::from_utf8(unsafe { slice::from_raw_parts(data as *const u8, len as usize) }.to_vec())
        .unwrap()
}

#[test]
fn bytes_copy_overlap_and_data_conversion() {
    let bytes = haxe_bytes_of_string(wasm_string_of(b"abcde"));
    assert_eq!(haxe_bytes_length(bytes), 5);
    haxe_bytes_blit(bytes, 0, bytes, 1, 4);
    assert_eq!(text(haxe_bytes_to_string(bytes)), "aabcd");
    let data = haxe_bytes_get_data(bytes);
    assert_eq!(haxe_array_length(data), 5);
    assert_eq!(haxe_array_get_i64(data, 2), b'b' as i32);
    let copy = haxe_bytes_of_int_array(5, data);
    haxe_bytes_set(copy, 0, b'z' as i32);
    assert_eq!(haxe_bytes_get(bytes, 0), b'a' as i32);
    assert_eq!(text(haxe_bytes_get_string(copy, 1, 3)), "abc");
    haxe_bytes_free(copy);
    haxe_bytes_free(bytes);
}

#[test]
fn bytes_alloc_is_zero_filled() {
    let bytes = haxe_bytes_alloc(17);
    for index in 0..17 {
        assert_eq!(haxe_bytes_get(bytes, index), 0);
    }
    haxe_bytes_free(bytes);
}

#[test]
fn string_map_copy_retains_layout_and_independent_entries() {
    let map = haxe_stringmap_new();
    let key = wasm_string_of(b"key");
    haxe_stringmap_set_typed(map, key, -7, 1, 0);
    let copy = haxe_stringmap_copy(map);
    assert_eq!(haxe_stringmap_get_null_int(copy, key, 1, 0), -7);
    let boxed = haxe_stringmap_get_typed(copy, key, 12, 0);
    assert_eq!(haxe_unbox_int(boxed), -7);
    assert_eq!(haxe_stringmap_remove(map, key), 1);
    assert_eq!(haxe_stringmap_exists(map, key), 0);
    assert_eq!(haxe_stringmap_exists(copy, key), 1);
    assert_eq!(haxe_stringmap_get_null_int(map, key, 1, 0), NULL_INT);
}

#[test]
fn int_map_dynamic_writes_are_visible_to_typed_reads() {
    let map = haxe_intmap_new();
    haxe_intmap_set_typed(map, -3, haxe_box_int_ptr(42), 12, 0);
    assert_eq!(haxe_intmap_get_typed(map, -3, 1, 0), 42);
    haxe_intmap_set_typed(map, 9, -11, 1, 0);
    assert_eq!(haxe_unbox_int(haxe_intmap_get_typed(map, 9, 12, 0)), -11);
    assert_eq!(haxe_intmap_get_null_int(map, -3, 1, 0), 42);
    assert_eq!(haxe_intmap_get_null_int(map, 100, 1, 0), NULL_INT);
}

#[test]
fn regex_captures_and_global_replacement() {
    let pattern = haxe_ereg_new(wasm_string_of(b"([a-z]+)([0-9]+)"), wasm_string_of(b"g"));
    let source = wasm_string_of(b"ab12 cd34");
    assert_eq!(haxe_ereg_match(pattern, source), 1);
    assert_eq!(text(haxe_ereg_matched(pattern, 1)), "ab");
    assert_eq!(text(haxe_ereg_matched_right(pattern)), " cd34");
    assert_eq!(
        text(haxe_ereg_replace(pattern, source, wasm_string_of(b"$2:$1"))),
        "12:ab 34:cd"
    );
    assert_eq!(haxe_ereg_match_sub(pattern, source, 5, -1), 1);
    assert_eq!(text(haxe_ereg_matched(pattern, 0)), "cd34");
    let position = haxe_ereg_matched_pos_anon(pattern);
    assert_eq!(rayzor_anon_get_field_by_index(position, 0), 4);
    assert_eq!(rayzor_anon_get_field_by_index(position, 1), 5);
}

#[test]
fn regex_fallback_handles_backreferences() {
    let pattern = haxe_ereg_new(wasm_string_of(br"(ab)\1"), wasm_string_of(b""));
    assert_eq!(haxe_ereg_match(pattern, wasm_string_of(b"zabab!")), 1);
    assert_eq!(text(haxe_ereg_matched(pattern, 1)), "ab");
    assert_eq!(text(haxe_ereg_matched_left(pattern)), "z");
}

#[test]
fn typed_integer_array_updates_and_removals_preserve_slots() {
    let array = haxe_array_new();
    assert_eq!(haxe_array_push_typed(array, -8, 1, 0), 1);
    assert_eq!(haxe_array_push_typed(array, 12, 1, 0), 2);
    assert_eq!(haxe_array_set_typed(array, 1, 27, 1, 0), 1);
    assert_eq!(haxe_array_get_typed(array, 0, 1), -8);
    assert_eq!(haxe_array_take_typed(array, 0, 1), -8);
    assert_eq!(haxe_array_take_typed(array, 1, 1), 27);
    assert_eq!(haxe_array_length(array), 0);
    haxe_array_free(array);
}

#[test]
fn closure_views_keep_environment_and_original_record() {
    let record = [1234_i64, 5678];
    let slot = [2345_i64, 0];
    let dynamic = [3456_i64, 0];
    let pointer = record.as_ptr() as i32;
    assert_eq!(haxe_closure_slot_view(0), 0);
    assert_eq!(haxe_closure_slot_view(pointer), pointer);
    haxe_closure_register_entries(pointer, slot.as_ptr() as i32, dynamic.as_ptr() as i32);
    for (view, expected_code) in [
        (haxe_closure_slot_view(pointer), 2345),
        (haxe_closure_dynamic_view(pointer), 3456),
    ] {
        assert_ne!(view, pointer);
        let values = unsafe { slice::from_raw_parts(view as *const i64, 2) };
        assert_eq!(values, &[expected_code, 5678]);
    }
    assert_eq!(record, [1234, 5678]);
}

#[test]
fn char_code_at_indexes_long_strings_by_character() {
    let ascii: Vec<u8> = (0..1000).map(|i| b'A' + (i % 4) as u8).collect();
    let s = wasm_string_of(&ascii);
    assert_eq!(haxe_string_char_code_at_ptr(s, 999), b'D' as i32);
    assert_eq!(haxe_string_char_code_at_ptr(s, 1000), -1);
    let mut wide = ascii.clone();
    wide.splice(0..0, "é".bytes());
    let w = wasm_string_of(&wide);
    assert_eq!(haxe_string_char_code_at_ptr(w, 0), 'é' as i32);
    assert_eq!(haxe_string_char_code_at_ptr(w, 1000), b'D' as i32);
    haxe_string_free(w);
    haxe_string_free(s);
}
