// The denoising network on the GPU, as WebGPU compute shaders.
//
// This runs the same FFDNet-lite as core/src/denoise/neural.rs and reads the same weight
// file. Only the convolutions run here; preparing the input tensor (colour transform, pixel
// unshuffle, noise maps) and applying the result stay in Rust (`nn_prepare` / `nn_finish`),
// so there's a single implementation of everything except the heavy arithmetic.
//
// Each layer is one dispatch. Every shader invocation computes one pixel for four output
// channels at once (a vec4), looping over all input channels and the 3×3 taps. Activations
// ping-pong between two storage buffers; only the final 12-channel result is read back.

const SHADER = /* wgsl */ `
struct Dims { w: u32, h: u32, cin: u32, cout: u32, relu: u32 };

@group(0) @binding(0) var<storage, read> input: array<f32>;      // cin  × h × w
@group(0) @binding(1) var<storage, read_write> output: array<f32>; // cout × h × w
@group(0) @binding(2) var<storage, read> weights: array<vec4f>;  // [cout/4][cin][9] → 4 outputs
@group(0) @binding(3) var<storage, read> bias: array<vec4f>;     // [cout/4]
@group(0) @binding(4) var<uniform> dims: Dims;

@compute @workgroup_size(8, 8, 1)
fn conv3x3(@builtin(global_invocation_id) id: vec3u) {
  let x = i32(id.x);
  let y = i32(id.y);
  let group = id.z; // which block of 4 output channels
  if (id.x >= dims.w || id.y >= dims.h) { return; }
  let w = i32(dims.w);
  let h = i32(dims.h);
  let plane = dims.w * dims.h;

  var acc = bias[group];
  for (var ci = 0u; ci < dims.cin; ci++) {
    let base = ci * plane;
    let wbase = (group * dims.cin + ci) * 9u;
    for (var ky = 0; ky < 3; ky++) {
      let sy = y + ky - 1;
      if (sy < 0 || sy >= h) { continue; }            // zero padding
      for (var kx = 0; kx < 3; kx++) {
        let sx = x + kx - 1;
        if (sx < 0 || sx >= w) { continue; }
        let v = input[base + u32(sy * w + sx)];
        acc += v * weights[wbase + u32(ky * 3 + kx)];
      }
    }
  }
  if (dims.relu == 1u) { acc = max(acc, vec4f(0.0)); }
  let o = u32(y * w + x);
  for (var k = 0u; k < 4u; k++) {
    output[(group * 4u + k) * plane + o] = acc[k];
  }
}
`;

/** Parse the DRNN weight file (format documented in core/src/denoise/neural.rs). */
export function parseWeights(buffer) {
  const view = new DataView(buffer);
  const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 4));
  if (magic !== 'DRNN' || view.getUint32(4, true) !== 1) throw new Error('bad network file');
  const count = view.getUint32(8, true);
  let pos = 12;
  const layers = [];
  for (let i = 0; i < count; i++) {
    const cin = view.getUint32(pos, true);
    const cout = view.getUint32(pos + 4, true);
    pos += 8;
    const w = new Float32Array(buffer.slice(pos, pos + cout * cin * 9 * 4));
    pos += cout * cin * 9 * 4;
    const b = new Float32Array(buffer.slice(pos, pos + cout * 4));
    pos += cout * 4;
    layers.push({ cin, cout, w, b });
  }
  return layers;
}

/**
 * Reorder [cout][cin][9] weights into [cout/4][cin][9][4] so one vec4 holds the same tap for
 * four output channels – exactly what the shader multiplies by.
 */
function packWeights({ cin, cout, w }) {
  const packed = new Float32Array(cout * cin * 9);
  for (let co = 0; co < cout; co++) {
    for (let ci = 0; ci < cin; ci++) {
      for (let k = 0; k < 9; k++) {
        packed[(((co >> 2) * cin + ci) * 9 + k) * 4 + (co & 3)] = w[(co * cin + ci) * 9 + k];
      }
    }
  }
  return packed;
}

export async function createGpuDenoiser(weightBuffer) {
  if (!globalThis.navigator?.gpu) return null;
  const adapter = await navigator.gpu.requestAdapter().catch(() => null);
  if (!adapter) return null;
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    },
  });
  const layers = parseWeights(weightBuffer);
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module: device.createShaderModule({ code: SHADER }), entryPoint: 'conv3x3' },
  });

  const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const gpuLayers = layers.map((layer, i) => {
    const weights = device.createBuffer({ size: layer.w.byteLength, usage: STORAGE });
    device.queue.writeBuffer(weights, 0, packWeights(layer));
    const bias = device.createBuffer({ size: layer.b.byteLength, usage: STORAGE });
    device.queue.writeBuffer(bias, 0, layer.b);
    const dims = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    return { ...layer, weights, bias, dims, relu: i < layers.length - 1 };
  });
  const maxChannels = Math.max(...layers.map((l) => Math.max(l.cin, l.cout)));

  /** Largest half-resolution tile (in pixels) the activation buffers can hold. */
  const maxPixels = Math.floor(Math.min(device.limits.maxStorageBufferBindingSize, 256 * 1024 * 1024) / (maxChannels * 4));

  let buffers = null; // { pixels, a, b }
  function ensureBuffers(pixels) {
    if (buffers && buffers.pixels >= pixels) return buffers;
    buffers?.a.destroy();
    buffers?.b.destroy();
    buffers?.readback.destroy();
    const size = maxChannels * pixels * 4;
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    buffers = {
      pixels,
      a: device.createBuffer({ size, usage }),
      b: device.createBuffer({ size, usage }),
      readback: device.createBuffer({ size: 12 * pixels * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
    };
    return buffers;
  }

  // Runs share the activation and readback buffers, so they must not overlap (the preview and
  // the loupe can both ask at once). Each run waits for the previous one.
  let queue = Promise.resolve();

  return {
    maxPixels,

    /**
     * Run the network on a prepared input tensor (15 × h × w, from `nn_prepare`).
     * Returns the predicted noise (12 × h × w) for `nn_finish`.
     */
    run(input, w, h) {
      const job = queue.then(() => runNow(input, w, h));
      queue = job.catch(() => {});
      return job;
    },
  };

  async function runNow(input, w, h) {
    const pixels = w * h;
    if (pixels > maxPixels) throw new Error('tile too large for the GPU');
    const { a, b, readback } = ensureBuffers(pixels);
    device.queue.writeBuffer(a, 0, input);
    const encoder = device.createCommandEncoder();
    let src = a;
    let dst = b;
    for (const layer of gpuLayers) {
      device.queue.writeBuffer(layer.dims, 0, new Uint32Array([w, h, layer.cin, layer.cout, layer.relu ? 1 : 0]));
      const bind = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: src } },
          { binding: 1, resource: { buffer: dst } },
          { binding: 2, resource: { buffer: layer.weights } },
          { binding: 3, resource: { buffer: layer.bias } },
          { binding: 4, resource: { buffer: layer.dims } },
        ],
      });
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bind);
      pass.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8), layer.cout / 4);
      pass.end();
      [src, dst] = [dst, src];
    }
    encoder.copyBufferToBuffer(src, 0, readback, 0, 12 * pixels * 4);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ, 0, 12 * pixels * 4);
    const out = new Float32Array(readback.getMappedRange(0, 12 * pixels * 4).slice(0));
    readback.unmap();
    return out;
  }
}
