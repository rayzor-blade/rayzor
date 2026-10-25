//! The rayzor-gpu plugin library: every export of `rayzor-gpu`, linked as a
//! library the host loads at run time.

// Without `native` there are no entry points for it to carry.
#[cfg_attr(not(feature = "native"), allow(unused_imports))]
pub use rayzor_gpu::*;
