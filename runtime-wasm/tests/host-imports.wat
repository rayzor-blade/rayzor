;; Host dispatch is outside the in-guest unit tests. Any use must fail.
(module
  (func (export "rayzor_host_flash_attn_q8_par")
    (param i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32) (result i32)
    unreachable)
  (func (export "rayzor_host_qmatmul_par")
    (param i32 i32 i32 i32 i32 i32 i32 i32 i32) (result i32)
    unreachable)
  (func (export "rayzor_thread_spawn") (param i32 i32) (result i32)
    unreachable)
  (func (export "rayzor_thread_join_void") (param i32)
    unreachable))
