// WebGPU preview renderer.
//
// The proxy image lives on the GPU as an `rgba16uint` texture – the raw 16-bit code values,
// exactly what the CPU sees. The develop maths arrives as the 3 × 65536 tone LUT built in Rust
// (core/src/develop/tone.rs), stored in a storage buffer. So moving a slider costs one 768 KB
// buffer upload and one draw; panning or cropping costs only a draw.
//
// The fragment shader below is the GPU twin of core/src/develop/render.rs: orient → crop →
// bilinear sample → LUT → saturation → sRGB encode.

const SHADER = /* wgsl */ `
struct View {
  crop: vec4f,        // x0, y0, x1, y1 in displayed (oriented) normalised coordinates
  orient: vec4u,      // transpose, flipX, flipY, unused
  look: vec4f,        // saturation, unused…
};

@group(0) @binding(0) var image: texture_2d<u32>;
@group(0) @binding(1) var<storage, read> lut: array<f32>;
@group(0) @binding(2) var<uniform> view: View;

struct VsOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
};

// One triangle that covers the whole canvas; uv runs 0..1 from the top-left.
@vertex
fn vs(@builtin(vertex_index) i: u32) -> VsOut {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u)) * 2.0 - 1.0;
  return VsOut(vec4f(p, 0.0, 1.0), vec2f(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5));
}

// Per-channel develop via the LUT (65536 entries per channel).
fn tap(p: vec2i) -> vec3f {
  let c = textureLoad(image, p, 0);
  return vec3f(lut[c.r], lut[65536u + c.g], lut[131072u + c.b]);
}

fn srgb_encode(x: vec3f) -> vec3f {
  let lo = x * 12.92;
  let hi = 1.055 * pow(x, vec3f(1.0 / 2.4)) - 0.055;
  return select(hi, lo, x <= vec3f(0.0031308));
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4f {
  // Displayed coordinates → source coordinates (geometry.rs: transpose, then flip X / Y).
  let o = mix(view.crop.xy, view.crop.zw, in.uv);
  let x = select(o.x, 1.0 - o.x, view.orient.y == 1u);
  let y = select(o.y, 1.0 - o.y, view.orient.z == 1u);
  let s = select(vec2f(x, y), vec2f(y, x), view.orient.x == 1u);

  // Bilinear interpolation, after the LUT, i.e. in linear light.
  let size = vec2f(textureDimensions(image));
  let f = clamp(s * size - 0.5, vec2f(0.0), size - 1.0);
  let i0 = vec2i(floor(f));
  let i1 = min(i0 + 1, vec2i(size) - 1);
  let a = f - floor(f);
  let top = mix(tap(i0), tap(vec2i(i1.x, i0.y)), a.x);
  let bottom = mix(tap(vec2i(i0.x, i1.y)), tap(i1), a.x);
  let rgb = mix(top, bottom, a.y);

  // Saturation around Rec.709 luminance (apply.rs: Look::mix).
  let luma = dot(rgb, vec3f(0.2126, 0.7152, 0.0722));
  let mixed = clamp(vec3f(luma) + (rgb - vec3f(luma)) * view.look.x, vec3f(0.0), vec3f(1.0));
  return vec4f(srgb_encode(mixed), 1.0);
}
`;

export async function createGpuRenderer(canvas) {
  if (!navigator.gpu) return null;
  const adapter = await navigator.gpu.requestAdapter().catch(() => null);
  if (!adapter) return null;
  const device = await adapter.requestDevice();
  const context = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'opaque' });

  const module = device.createShaderModule({ code: SHADER });
  const pipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module, entryPoint: 'vs' },
    fragment: { module, entryPoint: 'fs', targets: [{ format }] },
    primitive: { topology: 'triangle-list' },
  });

  const lutBuffer = device.createBuffer({ size: 3 * 65536 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const viewBuffer = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  let texture = null;
  let bindGroup = null;
  let hasLut = false;

  device.lost.then((info) => console.warn('WebGPU device lost:', info.message));

  return {
    kind: 'webgpu',

    /** Upload a new proxy image (RGBA16 code values). */
    setImage(proxy, width, height) {
      texture?.destroy();
      texture = device.createTexture({ size: [width, height], format: 'rgba16uint', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
      device.queue.writeTexture({ texture }, proxy, { bytesPerRow: width * 8 }, [width, height]);
      bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: texture.createView() },
          { binding: 1, resource: { buffer: lutBuffer } },
          { binding: 2, resource: { buffer: viewBuffer } },
        ],
      });
      hasLut = false;
    },

    setLut(lut) {
      device.queue.writeBuffer(lutBuffer, 0, lut);
      hasLut = true;
    },

    get ready() {
      return !!bindGroup && hasLut;
    },

    /** Draw the `crop` region with the given orientation into the canvas. */
    draw({ crop, orientation, saturation }) {
      if (!bindGroup || !hasLut) return;
      const view = new ArrayBuffer(48);
      new Float32Array(view, 0, 4).set(crop);
      new Uint32Array(view, 16, 4).set([orientation.transpose ? 1 : 0, orientation.flipX ? 1 : 0, orientation.flipY ? 1 : 0, 0]);
      new Float32Array(view, 32, 4).set([saturation, 0, 0, 0]);
      device.queue.writeBuffer(viewBuffer, 0, view);

      const encoder = device.createCommandEncoder();
      const pass = encoder.beginRenderPass({
        colorAttachments: [{ view: context.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.draw(3);
      pass.end();
      device.queue.submit([encoder.finish()]);
    },
  };
}
