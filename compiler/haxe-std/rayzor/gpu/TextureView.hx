package rayzor.gpu;

/**
 * A view into a texture — used as render target or bind group resource.
 *
 * Obtained from `Texture.getView()` or `Surface.getTexture()`.
 * TextureView is a lightweight handle; the underlying Texture owns the memory.
 */
@:native("rayzor::gpu::TextureView")
extern class TextureView {
    // No public constructors — obtained from Texture.getView() or Surface.getTexture()
}
