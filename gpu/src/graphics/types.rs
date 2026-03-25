//! Shared types and enum conversions for GPU graphics.

/// Convert an integer format code to a wgpu TextureFormat.
pub fn texture_format_from_int(code: i32) -> wgpu::TextureFormat {
    match code {
        0 => wgpu::TextureFormat::Bgra8Unorm,
        1 => wgpu::TextureFormat::Rgba8Unorm,
        2 => wgpu::TextureFormat::Depth24PlusStencil8,
        3 => wgpu::TextureFormat::Depth32Float,
        4 => wgpu::TextureFormat::Rgba16Float,
        5 => wgpu::TextureFormat::Rgba32Float,
        6 => wgpu::TextureFormat::Bgra8UnormSrgb,
        7 => wgpu::TextureFormat::Rgba8UnormSrgb,
        _ => wgpu::TextureFormat::Bgra8Unorm,
    }
}

/// Convert a wgpu TextureFormat to its integer code.
pub fn texture_format_to_int(fmt: wgpu::TextureFormat) -> i32 {
    match fmt {
        wgpu::TextureFormat::Bgra8Unorm => 0,
        wgpu::TextureFormat::Rgba8Unorm => 1,
        wgpu::TextureFormat::Depth24PlusStencil8 => 2,
        wgpu::TextureFormat::Depth32Float => 3,
        wgpu::TextureFormat::Rgba16Float => 4,
        wgpu::TextureFormat::Rgba32Float => 5,
        wgpu::TextureFormat::Bgra8UnormSrgb => 6,
        wgpu::TextureFormat::Rgba8UnormSrgb => 7,
        _ => 0,
    }
}

pub fn primitive_topology_from_int(code: i32) -> wgpu::PrimitiveTopology {
    match code {
        0 => wgpu::PrimitiveTopology::TriangleList,
        1 => wgpu::PrimitiveTopology::TriangleStrip,
        2 => wgpu::PrimitiveTopology::LineList,
        3 => wgpu::PrimitiveTopology::LineStrip,
        4 => wgpu::PrimitiveTopology::PointList,
        _ => wgpu::PrimitiveTopology::TriangleList,
    }
}

pub fn cull_mode_from_int(code: i32) -> Option<wgpu::Face> {
    match code {
        0 => None,
        1 => Some(wgpu::Face::Front),
        2 => Some(wgpu::Face::Back),
        _ => None,
    }
}

pub fn compare_function_from_int(code: i32) -> wgpu::CompareFunction {
    match code {
        0 => wgpu::CompareFunction::Never,
        1 => wgpu::CompareFunction::Less,
        2 => wgpu::CompareFunction::Equal,
        3 => wgpu::CompareFunction::LessEqual,
        4 => wgpu::CompareFunction::Greater,
        5 => wgpu::CompareFunction::NotEqual,
        6 => wgpu::CompareFunction::GreaterEqual,
        7 => wgpu::CompareFunction::Always,
        _ => wgpu::CompareFunction::Less,
    }
}

pub fn vertex_format_from_int(code: i32) -> wgpu::VertexFormat {
    match code {
        0 => wgpu::VertexFormat::Float32,
        1 => wgpu::VertexFormat::Float32x2,
        2 => wgpu::VertexFormat::Float32x3,
        3 => wgpu::VertexFormat::Float32x4,
        4 => wgpu::VertexFormat::Sint32,
        5 => wgpu::VertexFormat::Uint32,
        6 => wgpu::VertexFormat::Sint32x2,
        7 => wgpu::VertexFormat::Uint32x2,
        _ => wgpu::VertexFormat::Float32x3,
    }
}

pub fn filter_mode_from_int(code: i32) -> wgpu::FilterMode {
    match code {
        0 => wgpu::FilterMode::Nearest,
        1 => wgpu::FilterMode::Linear,
        _ => wgpu::FilterMode::Linear,
    }
}

pub fn address_mode_from_int(code: i32) -> wgpu::AddressMode {
    match code {
        0 => wgpu::AddressMode::ClampToEdge,
        1 => wgpu::AddressMode::Repeat,
        2 => wgpu::AddressMode::MirrorRepeat,
        _ => wgpu::AddressMode::ClampToEdge,
    }
}

/// Buffer usage flags (bitfield matching Haxe enum indices).
pub fn buffer_usages_from_flags(flags: i32) -> wgpu::BufferUsages {
    let mut usage = wgpu::BufferUsages::empty();
    if flags & 1 != 0 {
        usage |= wgpu::BufferUsages::VERTEX;
    }
    if flags & 2 != 0 {
        usage |= wgpu::BufferUsages::INDEX;
    }
    if flags & 4 != 0 {
        usage |= wgpu::BufferUsages::UNIFORM;
    }
    if flags & 8 != 0 {
        usage |= wgpu::BufferUsages::STORAGE;
    }
    if flags & 16 != 0 {
        usage |= wgpu::BufferUsages::COPY_SRC;
    }
    if flags & 32 != 0 {
        usage |= wgpu::BufferUsages::COPY_DST;
    }
    usage
}

pub fn texture_usages_from_flags(flags: i32) -> wgpu::TextureUsages {
    let mut usage = wgpu::TextureUsages::empty();
    if flags & 1 != 0 {
        usage |= wgpu::TextureUsages::COPY_SRC;
    }
    if flags & 2 != 0 {
        usage |= wgpu::TextureUsages::COPY_DST;
    }
    if flags & 4 != 0 {
        usage |= wgpu::TextureUsages::TEXTURE_BINDING;
    }
    if flags & 8 != 0 {
        usage |= wgpu::TextureUsages::STORAGE_BINDING;
    }
    if flags & 16 != 0 {
        usage |= wgpu::TextureUsages::RENDER_ATTACHMENT;
    }
    usage
}
