//! WebGPU device initialization via wgpu

/// Work encoded but not yet submitted.
struct Pending {
    encoder: wgpu::CommandEncoder,
    /// Bind groups referenced by the encoded passes. wgpu-core holds its own
    /// references, but keeping ours alive until submit makes the lifetime
    /// obvious rather than implicit.
    bind_groups: Vec<wgpu::BindGroup>,
    passes: usize,
}

/// Flush automatically once this many passes are queued, so a caller that
/// dispatches forever without reading back cannot grow the encoder unbounded.
const MAX_PENDING_PASSES: usize = 512;

/// WebGPU-specific GPU context wrapping device + queue.
///
/// Dispatches are ENCODED here and submitted lazily. Submitting each dispatch
/// on its own and blocking on `poll(Wait)` cost ~19 ms per call regardless of
/// kernel size — measured identical for a 1x4096x14336 GEMV and a 64x wider
/// one — which dominates anything smaller than a full prefill GEMM. A decode
/// step is 224 matmuls, so per-dispatch submission alone would cost 4.3 s per
/// token. Work now accumulates into one command buffer and is submitted when
/// somebody actually needs the result.
pub struct WgpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pending: std::cell::RefCell<Option<Pending>>,
}

impl WgpuContext {
    /// Encode into the pending command buffer, creating it if needed.
    /// `f` receives the encoder; `bind_group` is retained until submit.
    pub(crate) fn encode(
        &self,
        bind_group: wgpu::BindGroup,
        f: impl FnOnce(&mut wgpu::CommandEncoder, &wgpu::BindGroup),
    ) {
        let mut slot = self.pending.borrow_mut();
        if slot.is_none() {
            *slot = Some(Pending {
                encoder: self
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("rayzor_batched"),
                    }),
                bind_groups: Vec::new(),
                passes: 0,
            });
        }
        let p = slot.as_mut().expect("just created");
        f(&mut p.encoder, &bind_group);
        p.bind_groups.push(bind_group);
        p.passes += 1;
        let full = p.passes >= MAX_PENDING_PASSES;
        drop(slot);
        if full {
            self.flush();
        }
    }

    /// Submit everything encoded so far and wait for it to complete.
    ///
    /// Must be called before ANY readback: results are not in the buffer until
    /// the command buffer that writes them has run.
    pub fn flush(&self) {
        let taken = self.pending.borrow_mut().take();
        if let Some(p) = taken {
            self.queue.submit(std::iter::once(p.encoder.finish()));
            self.device.poll(wgpu::Maintain::Wait);
        }
    }
}

impl WgpuContext {
    /// Create a new wgpu context using the best available adapter.
    /// Native only — on WASM, use `new_async()`.
    #[cfg(feature = "native")]
    pub fn new() -> Option<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))?;

        // Take what the adapter actually supports rather than wgpu's portable
        // defaults. The default caps a storage binding at 128 MiB, which is
        // smaller than a single 7B FFN weight (4096x14336 f32 = 224 MiB) and
        // fails validation before any kernel runs.
        let limits = adapter.limits();
        if std::env::var_os("RZG_GPU_DEBUG").is_some() {
            let info = adapter.get_info();
            eprintln!(
                "[rzg] adapter={} backend={:?} device_type={:?}",
                info.name, info.backend, info.device_type
            );
            eprintln!(
                "[rzg] max_buffer_size={} MiB max_storage_binding={} MiB max_wg_size={} max_wg_storage={} B",
                limits.max_buffer_size / (1024 * 1024),
                limits.max_storage_buffer_binding_size / (1024 * 1024),
                limits.max_compute_invocations_per_workgroup,
                limits.max_compute_workgroup_storage_size
            );
        }
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("rayzor_gpu"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                ..Default::default()
            },
            None,
        ))
        .ok()?;

        Some(WgpuContext {
            device,
            queue,
            pending: std::cell::RefCell::new(None),
        })
    }

    /// Async version for WASM.
    pub async fn new_async() -> Option<Self> {
        let backends = if cfg!(target_arch = "wasm32") {
            wgpu::Backends::BROWSER_WEBGPU | wgpu::Backends::GL
        } else {
            wgpu::Backends::all()
        };
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("rayzor_gpu"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_webgl2_defaults(),
                    ..Default::default()
                },
                None,
            )
            .await
            .ok()?;

        Some(WgpuContext {
            device,
            queue,
            pending: std::cell::RefCell::new(None),
        })
    }

    /// Check if wgpu is available on this system.
    #[cfg(feature = "native")]
    pub fn is_available() -> bool {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .is_some()
    }
}
