// A compute worker: one WebAssembly instance doing one heavy job at a time.
//
// Several of these run in parallel (see pool.js). Each job is a message
//   { id, type, payload }  →  { id, ok: true, result }  or  { id, ok: false, error, fatal }
// Large buffers are *transferred* (moved, not copied) in both directions.

import { loadCore } from '../js/wasm.js';

const core = await loadCore();

// Developers (and their 768 KB lookup tables) are cached per parameter set, because every strip
// of an export uses the same parameters.
let cachedDev = { key: null, dev: null };
function developer(paramsJson) {
  if (cachedDev.key !== paramsJson) {
    cachedDev.dev?.free();
    cachedDev = { key: paramsJson, dev: new core.Developer(paramsJson) };
  }
  return cachedDev.dev;
}

// The classic denoisers need no data; the neural one needs its weights (~180 KB), fetched on
// first use. One Denoiser per worker, kept for the worker's lifetime.
let denoiser = null;
let denoiserHasNet = false;
async function getDenoiser(needNet) {
  if (!denoiser || (needNet && !denoiserHasNet)) {
    let weights = new Uint8Array(0);
    if (needNet) {
      const res = await fetch(new URL('../models/denoise.bin', import.meta.url));
      if (!res.ok) throw new Error('neural network weights not found (models/denoise.bin)');
      weights = new Uint8Array(await res.arrayBuffer());
    }
    denoiser?.free();
    denoiser = new core.Denoiser(weights);
    denoiserHasNet = needNet;
  }
  return denoiser;
}

/** RGBA16 proxy from RGB16 (GPUs have no 3-channel textures). */
function toRgba(rgb) {
  const out = new Uint16Array((rgb.length / 3) * 4);
  for (let i = 0, j = 0; i < rgb.length; i += 3, j += 4) {
    out[j] = rgb[i];
    out[j + 1] = rgb[i + 1];
    out[j + 2] = rgb[i + 2];
    out[j + 3] = 65535;
  }
  return out;
}

/** Size that fits within `maxEdge` (never upscales). Mirrors resize::fit_within. */
function fit(w, h, maxEdge) {
  const s = Math.min(1, maxEdge / Math.max(w, h));
  return [Math.max(1, Math.round(w * s)), Math.max(1, Math.round(h * s))];
}

/** Formats our Rust decoders don't handle (AVIF, HEIC, …) fall back to the browser's decoder. */
async function browserDecode(bytes) {
  const bitmap = await createImageBitmap(new Blob([bytes]));
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(bitmap, 0, 0);
  const rgba = ctx.getImageData(0, 0, bitmap.width, bitmap.height).data;
  return {
    width: bitmap.width,
    height: bitmap.height,
    // The browser has already applied EXIF orientation and converted to 8-bit sRGB.
    orientation: 1,
    linear: false,
    description: 'Image 8-bit (browser decoder)',
    rgb: core.rgba8_to_rgb16(new Uint8Array(rgba.buffer)),
  };
}

const jobs = {
  /** File bytes → 16-bit RGB master + RGBA16 proxy for the screen + tiny thumbnail. */
  async decode({ bytes, name, proxyEdge }) {
    const data = new Uint8Array(bytes);
    let img;
    try {
      const d = core.decode(data, name);
      img = { width: d.width, height: d.height, orientation: d.orientation, linear: d.linear, description: d.description, rgb: d.take_rgb() };
      d.free();
    } catch (e) {
      try {
        img = await browserDecode(data);
      } catch {
        throw e; // report the more informative error from our own decoder
      }
    }
    const [pw, ph] = fit(img.width, img.height, proxyEdge);
    const proxyRgb = core.resize_rgb16(img.rgb, img.width, img.height, pw, ph);
    const proxy = toRgba(proxyRgb);
    // Noise/grain is measured on both master and proxy (downscaling averages grain away, so
    // the proxy has less), and in both working spaces: gamma for positives, density for
    // negatives. Each measurement looks at a ≤1024² window, so this is quick.
    const measure = (negative) => ({
      master: Array.from(core.estimate_noise(img.rgb, img.width, img.height, img.linear, negative)),
      proxy: Array.from(core.estimate_noise(proxyRgb, pw, ph, img.linear, negative)),
    });
    const noise = { positive: measure(false), negative: measure(true) };
    let jpeg = null;
    if (data[0] === 0xff && data[1] === 0xd8) {
      try {
        jpeg = JSON.parse(core.jpeg_info(data));
      } catch {
        jpeg = null; // e.g. arithmetic-coded: lossless export not offered
      }
    }
    const result = { ...img, proxy, pw, ph, jpeg, noise };
    return { result, transfer: [img.rgb.buffer, proxy.buffer] };
  },

  /**
   * Denoise one tile. `rgb` includes a margin of context around the `core` rectangle
   * ([x, y, w, h] within the tile); only the core is returned, so tiles join seamlessly.
   */
  async denoise({ rgb, w, h, linear, params, noise, core: [cx, cy, cw, ch] }) {
    const d = await getDenoiser(JSON.parse(params).method === 'neural');
    const full = d.run(new Uint16Array(rgb), w, h, linear, params, new Float32Array(noise));
    const out = new Uint16Array(cw * ch * 3);
    for (let y = 0; y < ch; y++) {
      const start = ((cy + y) * w + cx) * 3;
      out.set(full.subarray(start, start + cw * 3), y * cw * 3);
    }
    return { result: out, transfer: [out.buffer] };
  },

  /** Develop one horizontal strip of the master. */
  process({ rgb, params }) {
    const out = developer(params).process_rgb16(new Uint16Array(rgb));
    return { result: out, transfer: [out.buffer] };
  },

  /** Resize → orient → encode a developed image. */
  finish({ rgb, width, height, orientation, outW, outH, format, quality, chroma444, exifSource }) {
    let px = new Uint16Array(rgb);
    let [w, h] = [width, height];
    if (outW !== w || outH !== h) {
      px = core.resize_rgb16(px, w, h, outW, outH);
      [w, h] = [outW, outH];
    }
    if (orientation !== 1) {
      px = core.orient_rgb16(px, w, h, orientation);
      if (orientation >= 5) [w, h] = [h, w]; // codes 5–8 transpose
    }
    let bytes;
    if (format === 'jpeg') bytes = core.encode_jpeg(px, w, h, quality, chroma444, new Uint8Array(exifSource ?? 0));
    else if (format === 'png8') bytes = core.encode_png(px, w, h, false);
    else if (format === 'png16') bytes = core.encode_png(px, w, h, true);
    else if (format === 'tiff16') bytes = core.encode_tiff(px, w, h);
    else throw new Error(`unknown format ${format}`);
    return { result: { bytes, width: w, height: h }, transfer: [bytes.buffer] };
  },

  /** Lossless JPEG crop + rotate on DCT coefficients. */
  jpegLossless({ bytes, orientation, rect }) {
    const out = core.jpeg_lossless(new Uint8Array(bytes), orientation, ...rect);
    const result = { bytes: out.take_bytes(), width: out.width, height: out.height };
    out.free();
    return { result, transfer: [result.bytes.buffer] };
  },
};

self.onmessage = async ({ data: { id, type, payload } }) => {
  try {
    const { result, transfer } = await jobs[type](payload);
    self.postMessage({ id, ok: true, result }, transfer);
  } catch (e) {
    // A WebAssembly trap (e.g. out of memory) can leave the instance in a broken state, so we
    // ask the pool to replace this worker.
    const fatal = e instanceof WebAssembly.RuntimeError;
    self.postMessage({ id, ok: false, error: String(e?.message ?? e), fatal });
  }
};

self.postMessage({ ready: true, simd: core.simd_enabled() });
