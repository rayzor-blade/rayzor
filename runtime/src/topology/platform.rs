//! Platform dispatch. Each per-OS module exposes the same eight functions;
//! this file picks one set at compile time via `#[cfg]`.

#[cfg(target_os = "linux")]
#[path = "linux.rs"]
mod imp;

#[cfg(target_os = "macos")]
#[path = "macos.rs"]
mod imp;

#[cfg(target_os = "windows")]
#[path = "windows.rs"]
mod imp;

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
#[path = "stub.rs"]
mod imp;

pub(crate) fn available() -> bool {
    imp::available()
}

pub(crate) fn node_count() -> i32 {
    imp::node_count().max(1)
}

pub(crate) fn cpu_count() -> i32 {
    imp::cpu_count().max(1)
}

pub(crate) fn cpu_to_node(cpu: i32) -> i32 {
    if cpu < 0 || cpu >= cpu_count() {
        return -1;
    }
    imp::cpu_to_node(cpu)
}

pub(crate) fn node_cpus(node: i32, out: &mut [i32]) -> i32 {
    if node < 0 || node >= node_count() {
        return -1;
    }
    imp::node_cpus(node, out)
}

pub(crate) fn bind_current_thread(node: i32) -> i32 {
    if node < 0 || node >= node_count() {
        return -2;
    }
    imp::bind_current_thread(node)
}

pub(crate) fn unbind_current_thread() -> i32 {
    imp::unbind_current_thread()
}
