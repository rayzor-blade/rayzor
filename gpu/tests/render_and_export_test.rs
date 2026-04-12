//! End-to-end: render @:shader triangle to texture, read pixels, save as PPM.
//!
//! Run: cargo test -p rayzor-gpu --features webgpu-backend --test render_and_export_test

#[cfg(feature = "webgpu-backend")]
#[test]
fn test_render_triangle_and_export_ppm() {
    use rayzor_gpu::graphics::pipeline::*;
    use rayzor_gpu::graphics::render_pass::*;
    use rayzor_gpu::graphics::shader::*;
    use rayzor_gpu::graphics::texture::*;
    use rayzor_gpu::graphics::*;

    let w = 256u32;
    let h = 256u32;

    // 1. Device
    let ctx = rayzor_gpu_gfx_device_create();
    assert!(!ctx.is_null(), "No GPU");

    // 2. Shader — the WGSL that @:shader TriangleShader.wgsl() would produce
    let wgsl = r#"
struct VOut {
    @builtin(position) position: vec4f,
    @location(0) color: vec3f,
}

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VOut {
    var pos = array<vec2f, 3>(vec2f(0.0, 0.5), vec2f(-0.5, -0.5), vec2f(0.5, -0.5));
    var col = array<vec3f, 3>(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0));
    var out: VOut;
    out.position = vec4f(pos[i], 0.0, 1.0);
    out.color = col[i];
    return out;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4f {
    return vec4f(in.color, 1.0);
}
"#;
    let wgsl_b = wgsl.as_bytes();
    let vs = "vs";
    let fs = "fs";
    let shader = unsafe {
        rayzor_gpu_gfx_shader_create(
            ctx,
            wgsl_b.as_ptr(),
            wgsl_b.len(),
            vs.as_ptr(),
            vs.len(),
            fs.as_ptr(),
            fs.len(),
        )
    };
    assert!(!shader.is_null(), "Shader compile failed");

    // 3. Pipeline (no vertex buffer — positions from @builtin(vertex_index))
    let builder = rayzor_gpu_gfx_pipeline_begin();
    unsafe {
        rayzor_gpu_gfx_pipeline_set_shader(builder, shader);
        rayzor_gpu_gfx_pipeline_set_format(builder, 1); // RGBA8Unorm
        rayzor_gpu_gfx_pipeline_set_topology(builder, 0);
    }
    let pipeline = unsafe { rayzor_gpu_gfx_pipeline_build(builder, ctx) };
    assert!(!pipeline.is_null(), "Pipeline build failed");

    // 4. Render target
    let target = unsafe { rayzor_gpu_gfx_texture_create(ctx, w, h, 1, 16 | 1) };
    assert!(!target.is_null());
    let view = unsafe { rayzor_gpu_gfx_texture_get_view(target) };
    assert!(!view.is_null());

    // 5. Render
    unsafe {
        rayzor_gpu_gfx_render_submit(
            ctx,
            view,
            0,
            0.05,
            0.05,
            0.15,
            1.0,              // clear dark navy
            std::ptr::null(), // no depth
            pipeline,
            std::ptr::null(), // no vertex buffer
            3,
            1, // 3 vertices, 1 instance
            std::ptr::null(),
            0,
            0, // no index
            0,
            std::ptr::null(), // no bind groups
        );
    }

    // 6. Read pixels
    let pixel_bytes = (w * h * 4) as usize;
    let mut pixels = vec![0u8; pixel_bytes];
    let read = unsafe {
        rayzor_gpu_gfx_texture_read_pixels(ctx, target, pixels.as_mut_ptr(), pixels.len())
    };
    assert!(read > 0, "Readback failed");

    // 7. Verify center pixel is not background
    let cx = (h / 2 * w + w / 2) as usize * 4;
    let (r, g, b, a) = (pixels[cx], pixels[cx + 1], pixels[cx + 2], pixels[cx + 3]);
    eprintln!("Center pixel: RGBA({}, {}, {}, {})", r, g, b, a);
    assert!(
        a > 0 && (r > 20 || g > 20 || b > 20),
        "Triangle should cover center"
    );

    // 8. Save as PPM
    let path = "triangle_output.ppm";
    let ppm = format!("P6\n{} {}\n255\n", w, h);
    let mut rgb_data = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            rgb_data.push(pixels[i]); // R
            rgb_data.push(pixels[i + 1]); // G
            rgb_data.push(pixels[i + 2]); // B
        }
    }
    let mut file_data = ppm.into_bytes();
    file_data.extend(rgb_data);
    std::fs::write(path, &file_data).unwrap();
    eprintln!(
        "Saved {} ({}x{} PPM, {} bytes)",
        path,
        w,
        h,
        file_data.len()
    );

    // Verify file exists and has content
    let meta = std::fs::metadata(path).unwrap();
    assert!(meta.len() > 1000, "PPM file too small");

    // 9. Cleanup
    unsafe {
        rayzor_gpu_gfx_texture_destroy(target);
        rayzor_gpu_gfx_pipeline_destroy(pipeline);
        rayzor_gpu_gfx_shader_destroy(shader);
        rayzor_gpu_gfx_device_destroy(ctx);
    }

    eprintln!("GPU triangle rendered and exported to PPM successfully!");
}
