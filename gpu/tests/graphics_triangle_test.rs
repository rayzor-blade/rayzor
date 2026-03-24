//! End-to-end test: render a colored triangle to a texture and read pixels back.
//!
//! Run with: cargo test -p rayzor-gpu --features webgpu-backend --test graphics_triangle_test

#[cfg(feature = "webgpu-backend")]
#[test]
fn test_render_triangle_to_texture() {
    use rayzor_gpu::graphics::*;
    use rayzor_gpu::graphics::shader::*;
    use rayzor_gpu::graphics::pipeline::*;
    use rayzor_gpu::graphics::texture::*;
    use rayzor_gpu::graphics::render_pass::*;

    // 1. Create device
    let ctx = unsafe { rayzor_gpu_gfx_device_create() };
    assert!(!ctx.is_null(), "Failed to create GPU device");

    // 2. Compile WGSL shader — triangle with vertex colors
    let wgsl = r#"
struct VertexOutput {
    @builtin(position) pos: vec4f,
    @location(0) color: vec3f,
};

@vertex
fn vs_main(@location(0) position: vec3f, @location(1) color: vec3f) -> VertexOutput {
    var out: VertexOutput;
    out.pos = vec4f(position, 1.0);
    out.color = color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    return vec4f(in.color, 1.0);
}
"#;
    let wgsl_bytes = wgsl.as_bytes();
    let vert = "vs_main";
    let frag = "fs_main";
    let shader = unsafe {
        rayzor_gpu_gfx_shader_create(
            ctx,
            wgsl_bytes.as_ptr(),
            wgsl_bytes.len(),
            vert.as_ptr(),
            vert.len(),
            frag.as_ptr(),
            frag.len(),
        )
    };
    assert!(!shader.is_null(), "Failed to create shader");

    // 3. Create vertex buffer — 3 vertices with position (vec3f) + color (vec3f)
    #[repr(C)]
    struct Vertex {
        pos: [f32; 3],
        color: [f32; 3],
    }
    let vertices = [
        Vertex { pos: [0.0, 0.5, 0.0], color: [1.0, 0.0, 0.0] },     // top - red
        Vertex { pos: [-0.5, -0.5, 0.0], color: [0.0, 1.0, 0.0] },   // bottom-left - green
        Vertex { pos: [0.5, -0.5, 0.0], color: [0.0, 0.0, 1.0] },    // bottom-right - blue
    ];
    let vertex_data = unsafe {
        std::slice::from_raw_parts(
            vertices.as_ptr() as *const u8,
            std::mem::size_of_val(&vertices),
        )
    };
    let vb = unsafe {
        rayzor_gpu_gfx_buffer_create_with_data(
            ctx,
            vertex_data.as_ptr(),
            vertex_data.len(),
            1, // VERTEX
        )
    };
    assert!(!vb.is_null(), "Failed to create vertex buffer");

    // 4. Build render pipeline
    let builder = unsafe { rayzor_gpu_gfx_pipeline_begin() };
    assert!(!builder.is_null());
    unsafe {
        rayzor_gpu_gfx_pipeline_set_shader(builder, shader);
        // Vertex layout: stride=24 (6 floats), 2 attributes
        let formats = [2i32, 2]; // Float32x3, Float32x3
        let offsets = [0u64, 12];
        let locations = [0i32, 1];
        rayzor_gpu_gfx_pipeline_set_vertex_layout(
            builder, 24, 2,
            formats.as_ptr(), offsets.as_ptr(), locations.as_ptr(),
        );
        rayzor_gpu_gfx_pipeline_set_format(builder, 1); // RGBA8Unorm
        rayzor_gpu_gfx_pipeline_set_topology(builder, 0); // TriangleList
    }
    let pipeline = unsafe { rayzor_gpu_gfx_pipeline_build(builder, ctx) };
    assert!(!pipeline.is_null(), "Failed to build pipeline");

    // 5. Create render target texture (64x64 RGBA8)
    let target = unsafe {
        rayzor_gpu_gfx_texture_create(
            ctx, 64, 64,
            1,  // RGBA8Unorm
            16 | 1, // RENDER_ATTACHMENT | COPY_SRC
        )
    };
    assert!(!target.is_null(), "Failed to create render target");

    // 6. Get texture view
    let view = unsafe { rayzor_gpu_gfx_texture_get_view(target) };
    assert!(!view.is_null(), "Failed to get texture view");

    // 7. Render!
    unsafe {
        rayzor_gpu_gfx_render_submit(
            ctx,
            view,
            0,                  // LoadOp::Clear
            0.0, 0.0, 0.0, 1.0, // clear to black
            std::ptr::null(),   // no depth
            pipeline,
            vb,                 // vertex buffer
            3, 1,               // 3 vertices, 1 instance
            std::ptr::null(),   // no index buffer
            0, 0,               // no indices
            0,                  // no bind groups
            std::ptr::null(),   // no bind group array
        );
    }

    // 8. Read pixels back
    let pixel_count = 64 * 64;
    let mut pixels = vec![0u8; pixel_count * 4]; // RGBA8
    let bytes_read = unsafe {
        rayzor_gpu_gfx_texture_read_pixels(
            ctx, target,
            pixels.as_mut_ptr(),
            pixels.len(),
        )
    };
    assert!(bytes_read > 0, "Pixel readback failed");

    // 9. Check center pixel is not black (triangle should be there)
    let center = (32 * 64 + 32) * 4; // center pixel
    let r = pixels[center];
    let g = pixels[center + 1];
    let b = pixels[center + 2];
    let a = pixels[center + 3];
    eprintln!("Center pixel: RGBA({}, {}, {}, {})", r, g, b, a);
    assert!(a > 0, "Center pixel alpha should be non-zero");
    assert!(
        r > 0 || g > 0 || b > 0,
        "Center pixel should not be black — triangle should cover it"
    );

    // 10. Cleanup
    unsafe {
        rayzor_gpu_gfx_texture_destroy(target);
        rayzor_gpu_gfx_pipeline_destroy(pipeline);
        rayzor_gpu_gfx_buffer_destroy(vb);
        rayzor_gpu_gfx_shader_destroy(shader);
        rayzor_gpu_gfx_device_destroy(ctx);
    }

    eprintln!("Triangle rendered and verified successfully!");
}
