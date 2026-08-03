//! WebGPU buffer operations — GPU memory allocation and data transfer

use wgpu;
use wgpu::util::DeviceExt;

use super::device_init::WgpuContext;

/// WebGPU-specific GPU buffer wrapping a wgpu::Buffer.
pub struct WgpuBuffer {
    pub(crate) buffer: wgpu::Buffer,
    pub(crate) byte_size: usize,
    /// Reference to the device for readback operations.
    /// We need this because wgpu readback requires creating a staging buffer.
    pub(crate) device: *const wgpu::Device,
    pub(crate) queue: *const wgpu::Queue,
    /// Needed so a readback can flush pending encoded work first — results do
    /// not exist until the command buffer that writes them has been submitted.
    pub(crate) ctx: *const super::device_init::WgpuContext,
}

// WgpuBuffer is not Send/Sync (raw pointers), but we only use it single-threaded
// through the FFI layer, same as MetalBuffer.

impl WgpuBuffer {
    /// Create a wgpu buffer by copying data from a CPU pointer.
    ///
    /// # Safety
    /// `data` must point to at least `byte_size` readable bytes.
    pub unsafe fn from_data(ctx: &WgpuContext, data: *const u8, byte_size: usize) -> Option<Self> {
        if data.is_null() || byte_size == 0 {
            return None;
        }

        let slice = unsafe { std::slice::from_raw_parts(data, byte_size) };
        let buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rayzor_gpu_buffer"),
                contents: slice,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::UNIFORM
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        Some(WgpuBuffer {
            buffer,
            byte_size,
            device: &ctx.device as *const _,
            queue: &ctx.queue as *const _,
            ctx: ctx as *const _,
        })
    }

    /// Allocate an empty wgpu buffer of the given byte size.
    pub fn allocate(ctx: &WgpuContext, byte_size: usize) -> Option<Self> {
        if byte_size == 0 {
            return None;
        }

        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rayzor_gpu_buffer"),
            size: byte_size as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Some(WgpuBuffer {
            buffer,
            byte_size,
            device: &ctx.device as *const _,
            queue: &ctx.queue as *const _,
            ctx: ctx as *const _,
        })
    }

    /// Read buffer contents back to CPU via staging buffer.
    pub fn read_to_vec(&self, byte_size: usize) -> Option<Vec<u8>> {
        // Submit anything still encoded, or we would read a buffer the GPU has
        // not written yet.
        if !self.ctx.is_null() {
            unsafe { &*self.ctx }.flush();
        }
        let device = unsafe { &*self.device };
        let queue = unsafe { &*self.queue };
        let read_size = byte_size.min(self.byte_size);

        // Create staging buffer for readback
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rayzor_staging"),
            size: read_size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Copy from GPU buffer to staging
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rayzor_readback"),
        });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, read_size as u64);
        queue.submit(std::iter::once(encoder.finish()));

        // Map and read
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        device.poll(wgpu::Maintain::Wait);

        match rx.recv() {
            Ok(Ok(())) => {
                let data = slice.get_mapped_range().to_vec();
                staging.unmap();
                Some(data)
            }
            _ => None,
        }
    }

    /// Get the byte size of the buffer.
    pub fn byte_size(&self) -> usize {
        self.byte_size
    }
}
