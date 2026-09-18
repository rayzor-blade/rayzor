//! Metal device initialization

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice};

// MTLCreateSystemDefaultDevice requires CoreGraphics to be linked
#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

/// Metal-specific GPU context wrapping device + command queue.
pub struct MetalContext {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
}

impl MetalContext {
    /// Create a new Metal context using the system default device.
    pub fn new() -> Option<Self> {
        let device = MTLCreateSystemDefaultDevice()?;
        let command_queue = device.newCommandQueue()?;
        Some(MetalContext {
            device,
            command_queue,
        })
    }

    /// Check if Metal is available on this system.
    pub fn is_available() -> bool {
        MTLCreateSystemDefaultDevice().is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_device_creation() {
        let available = MetalContext::is_available();
        println!("Metal available: {}", available);
        if available {
            let ctx = MetalContext::new().expect("Failed to create Metal context");
            println!("Metal device created successfully");
            drop(ctx);
        }
    }
}
