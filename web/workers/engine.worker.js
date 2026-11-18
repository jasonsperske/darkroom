// The engine: owns every image's pixels and answers requests from the UI thread.
//
//   UI thread ──postMessage──► engine worker ──► its own wasm instance (light, latency-sensitive
//                                  │               work: tone LUTs, preview, histogram, analysis)
//                                  ├──► WorkerPool ──► N compute workers (heavy work: decoding,
//                                  │                    denoising, developing strips, encoding)
//                                  └──► WebGPU compute (the neural denoiser, when available)
//
// Keeping pixels here (not on the UI thread) means the UI never stalls on big buffers, and the
// latency-sensitive preview path is never queued behind a long export.

import { loadCore } from '../js/wasm.js';
import { WorkerPool } from './pool.js';
import { createGpuDenoiser } from '../js/nn/webgpu-denoise.js';
import { cropToSource, toPixels, orientedSize, decompose, inverse } from '../js/geometry.js';

const core = await loadCore();
const poolSize = Math.max(1, Math.min(8, (navigator.hardwareConcurrency || 4) - 1));
const pool = new WorkerPool(poolSize);

/** docId → { bytes, name, info, master, proxy, pw, ph, noise, denoised, lastUsed } */
const docs = new Map();
let clock = 0;

// Memory budget: full-resolution masters are big (24 MP × 6 bytes = 144 MB), so only the most
// recently used few stay decoded. Evicted images are decoded again from their file bytes.
const KEEP_MASTERS = 2;
const KEEP_PROXIES = 4;

function evict() {
  const byAge = [...docs.values()].sort((a, b) => b.lastUsed - a.lastUsed);
  byAge.slice(KEEP_MASTERS).forEach((d) => (d.master = null));
  byAge.slice(KEEP_PROXIES).forEach((d) => {
    d.proxy = null;
    d.denoised = null;
  });
}

async function decodeDoc(doc, proxyEdge = doc.proxyEdge) {
  const r = await pool.run('decode', { bytes: doc.bytes.slice(0), name: doc.name, proxyEdge }, []);
  doc.info = { width: r.width, height: r.height, orientation: r.orientation, linear: r.linear, description: r.description, jpeg: r.jpeg, noise: r.noise };
  doc.noise = r.noise;
  doc.master = r.rgb;
  doc.proxy = r.proxy;
  doc.pw = r.pw;
  doc.ph = r.ph;
}

async function use(docId, { master = false } = {}) {
  const doc = docs.get(docId);
  if (!doc) throw new Error('unknown image');
  doc.lastUsed = ++clock;
  if (!doc.proxy || (master && !doc.master)) {
    doc.loading ??= decodeDoc(doc).finally(() => (doc.loading = null));
    await doc.loading;
  }
  evict();
  return doc;
}

// One Developer per parameter set; rebuilding the LUT is the main cost of a slider move.
let cachedDev = { key: null, dev: null };
function developer(params) {
  const key = JSON.stringify(params);
  if (cachedDev.key !== key) {
    cachedDev.dev?.free();
    cachedDev = { key, dev: new core.Developer(key) };
  }
  return cachedDev.dev;
}

function fit(w, h, maxEdge) {
  const s = Math.min(1, maxEdge / Math.max(w, h));
  return [Math.max(1, Math.round(w * s)), Math.max(1, Math.round(h * s))];
}

async function toBitmap(rgba, w, h) {
  return createImageBitmap(new ImageData(new Uint8ClampedArray(rgba.buffer, rgba.byteOffset, rgba.byteLength), w, h));
}

/** Copy a rectangle out of an interleaved image with `ch` channels. */
function extract(src, srcW, ch, x0, y0, w, h) {
  const out = new src.constructor(w * h * ch);
  for (let y = 0; y < h; y++) {
    const start = ((y0 + y) * srcW + x0) * ch;
    out.set(src.subarray(start, start + w * ch), y * w * ch);
  }
  return out;
}

function rgbaToRgb(rgba) {
  const out = new Uint16Array((rgba.length / 4) * 3);
  for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) {
    out[j] = rgba[i];
    out[j + 1] = rgba[i + 1];
    out[j + 2] = rgba[i + 2];
  }
  return out;
}

function rgbToRgba(rgb) {
  const out = new Uint16Array((rgb.length / 3) * 4);
  for (let i = 0, j = 0; i < rgb.length; i += 3, j += 4) {
    out[j] = rgb[i];
    out[j + 1] = rgb[i + 1];
    out[j + 2] = rgb[i + 2];
    out[j + 3] = 65535;
  }
  return out;
}

// ---- Denoising ----------------------------------------------------------------------------------
//
// `denoise` objects come from the UI: { method, luma, chroma, grainSize, negative }.
// grainSize is null for "as measured"; negative selects density space.

const isOn = (denoise) => denoise && denoise.method !== 'off';

/**
 * The noise profile to use for `which` ('master' or 'proxy'), as the flat array the wasm side
 * expects: [σ_luma, σ_chroma, size, …measured per-layer values]. If the user overrides the
 * grain size we pass only the first three, so the pure grain model is used for every layer.
 */
function noiseProfile(doc, denoise, which) {
  const measured = doc.noise[denoise.negative ? 'negative' : 'positive'][which];
  if (denoise.grainSize == null) return measured;
  // The override is in full-resolution pixels; the proxy is smaller, and so is its grain.
  const scale = which === 'proxy' ? doc.pw / doc.info.width : 1;
  return [measured[0], measured[1], denoise.grainSize * scale];
}

// The neural network on WebGPU, created on first use. Resolves to null without WebGPU, in which
// case the compute workers run the same network in wasm (much slower, same result).
let gpuNet = null;
function getGpuNet() {
  gpuNet ??= (async () => {
    if (!navigator.gpu) return null;
    const res = await fetch(new URL('../models/denoise.bin', import.meta.url));
    if (!res.ok) throw new Error('neural network weights not found');
    return createGpuDenoiser(await res.arrayBuffer());
  })().catch((e) => {
    console.warn('GPU denoiser unavailable, using wasm:', e);
    return null;
  });
  return gpuNet;
}

/**
 * Denoise a whole RGB16 image, tile by tile. Each tile is processed with `margin` pixels of
 * surrounding context (the method's reach), and only its centre is kept, so the result is
 * identical to denoising the image in one piece – but runs in parallel and in bounded memory.
 */
async function denoiseImage(rgb, w, h, linear, denoise, noise, progress = () => {}) {
  if (!isOn(denoise)) return rgb;
  const params = JSON.stringify(denoise);
  const margin = core.denoise_margin(params);
  const gpu = denoise.method === 'neural' ? await getGpuNet() : null;
  const tile = gpu ? 1024 : 384;
  const out = new Uint16Array(rgb.length);
  const total = Math.ceil(w / tile) * Math.ceil(h / tile);
  let done = 0;
  const jobs = [];

  for (let ty = 0; ty < h; ty += tile) {
    for (let tx = 0; tx < w; tx += tile) {
      const [cw, ch] = [Math.min(tile, w - tx), Math.min(tile, h - ty)];
      const [ex, ey] = [Math.max(0, tx - margin), Math.max(0, ty - margin)];
      const [ew, eh] = [Math.min(w, tx + cw + margin) - ex, Math.min(h, ty + ch + margin) - ey];
      const sub = extract(rgb, w, 3, ex, ey, ew, eh);
      const coreRect = [tx - ex, ty - ey, cw, ch];
      const paste = (res) => {
        for (let y = 0; y < ch; y++) out.set(res.subarray(y * cw * 3, (y + 1) * cw * 3), ((ty + y) * w + tx) * 3);
        progress(++done / total);
      };
      if (gpu) {
        // GPU tiles run one after another: they share the GPU's activation buffers.
        const told = new Float32Array([noise[0] * denoise.luma, noise[1] * denoise.chroma, noise[2]]);
        const input = core.nn_prepare(sub, ew, eh, linear, !!denoise.negative, told);
        const predicted = await gpu.run(input, Math.ceil(ew / 2), Math.ceil(eh / 2));
        const full = core.nn_finish(sub, ew, eh, linear, !!denoise.negative, predicted);
        paste(extract(full, ew, 3, coreRect[0], coreRect[1], cw, ch));
      } else {
        jobs.push(pool.run('denoise', { rgb: sub.buffer, w: ew, h: eh, linear, params, noise, core: coreRect }, [sub.buffer]).then(paste));
      }
    }
  }
  await Promise.all(jobs);
  return out;
}

/**
 * The proxy to display: denoised (and cached) when noise reduction is on. Several requests can
 * wait for the same denoise run; each one's `progress` callback hears about it.
 */
async function proxyFor(doc, denoise, progress) {
  if (!isOn(denoise)) return doc.proxy;
  const key = JSON.stringify(denoise);
  if (doc.denoised?.key !== key) {
    const source = doc.proxy;
    const listeners = new Set();
    doc.denoised = {
      key,
      listeners,
      promise: denoiseImage(rgbaToRgb(source), doc.pw, doc.ph, doc.info.linear, denoise, noiseProfile(doc, denoise, 'proxy'), (p) => listeners.forEach((l) => l(p)))
        .then(rgbToRgba)
        .catch((e) => {
          doc.denoised = null; // don't cache failures
          throw e;
        }),
    };
  }
  const { promise, listeners } = doc.denoised;
  if (progress) listeners.add(progress);
  try {
    return await promise;
  } finally {
    listeners.delete(progress);
  }
}

// ---- Handlers -------------------------------------------------------------------------------------

const handlers = {
  status() {
    return pool.ready.then(() => ({ simd: core.simd_enabled(), workers: poolSize }));
  },

  /** Is the neural denoiser going to run on the GPU? (Asked by the UI to label it.) */
  async neuralBackend() {
    return { gpu: !!(await getGpuNet()) };
  },

  async open({ docId, bytes, name, proxyEdge }) {
    const doc = { bytes, name, proxyEdge, lastUsed: ++clock };
    docs.set(docId, doc);
    try {
      await use(docId);
    } catch (e) {
      docs.delete(docId);
      throw e;
    }
    return { info: doc.info };
  },

  close({ docId }) {
    docs.delete(docId);
    return {};
  },

  /** The proxy pixels (denoised if requested), for upload to the GPU. A copy. */
  async proxy({ docId, denoise }, progress) {
    const doc = await use(docId);
    const proxy = (await proxyFor(doc, denoise, progress)).slice();
    return { result: { proxy, pw: doc.pw, ph: doc.ph }, transfer: [proxy.buffer] };
  },

  /**
   * Develop the preview. With WebGPU the UI only needs the LUT (the GPU does the rest);
   * otherwise we render the view here. Either way we send back a histogram.
   */
  async develop({ docId, params, denoise, orientation, crop, outW, outH, cpu }, progress) {
    const doc = await use(docId);
    const proxy = await proxyFor(doc, denoise, progress);
    const dev = developer(params);
    const result = {};
    const transfer = [];
    if (cpu) {
      const rgba = dev.render_view(proxy, doc.pw, doc.ph, 4, orientation, new Float32Array(crop), outW, outH);
      result.histogram = core.histogram_rgba8(rgba);
      result.bitmap = await toBitmap(rgba, outW, outH);
      transfer.push(result.bitmap);
    } else {
      result.lut = dev.lut();
      transfer.push(result.lut.buffer);
      const [ow, oh] = orientedSize(doc.pw, doc.ph, orientation);
      const [hw, hh] = fit(Math.max(1, (crop[2] - crop[0]) * ow), Math.max(1, (crop[3] - crop[1]) * oh), 400);
      result.histogram = core.histogram_rgba8(dev.render_view(proxy, doc.pw, doc.ph, 4, orientation, new Float32Array(crop), hw, hh));
    }
    transfer.push(result.histogram.buffer);
    return { result, transfer };
  },

  /** A small thumbnail for the filmstrip. */
  async thumb({ docId, params, denoise, orientation, crop }) {
    const doc = await use(docId);
    const proxy = await proxyFor(doc, denoise);
    const [ow, oh] = orientedSize(doc.pw, doc.ph, orientation);
    const [w, h] = fit((crop[2] - crop[0]) * ow, (crop[3] - crop[1]) * oh, 160);
    const rgba = developer(params).render_view(proxy, doc.pw, doc.ph, 4, orientation, new Float32Array(crop), w, h);
    const canvas = new OffscreenCanvas(w, h);
    canvas.getContext('2d').putImageData(new ImageData(new Uint8ClampedArray(rgba.buffer), w, h), 0, 0);
    return { blob: await canvas.convertToBlob({ type: 'image/png' }) };
  },

  /**
   * 100 % view rendered from the full-resolution master. With noise reduction on, only the
   * visible region (plus context) is denoised – this is where its effect is really judged.
   */
  async loupe({ docId, params, denoise, orientation, crop, outW, outH }, progress) {
    const doc = await use(docId, { master: true });
    const { width: w, height: h, linear } = doc.info;
    let buffer = doc.master;
    let [bw, bh] = [w, h];
    let view = crop;
    if (isOn(denoise)) {
      const src = cropToSource(crop, orientation);
      const [x0, y0, x1, y1] = toPixels(src, w, h);
      const m = core.denoise_margin(JSON.stringify(denoise));
      const [ex, ey] = [Math.max(0, x0 - m), Math.max(0, y0 - m)];
      [bw, bh] = [Math.min(w, x1 + m) - ex, Math.min(h, y1 + m) - ey];
      buffer = await denoiseImage(extract(doc.master, w, 3, ex, ey, bw, bh), bw, bh, linear, denoise, noiseProfile(doc, denoise, 'master'), progress);
      // The same region, expressed relative to the extracted buffer, back in displayed coordinates.
      const rel = [(src[0] * w - ex) / bw, (src[1] * h - ey) / bh, (src[2] * w - ex) / bw, (src[3] * h - ey) / bh];
      view = cropToSource(rel, inverse(orientation));
    }
    const rgba = developer(params).render_view(buffer, bw, bh, 3, orientation, new Float32Array(view), outW, outH);
    const bitmap = await toBitmap(rgba, outW, outH);
    return { result: { bitmap }, transfer: [bitmap] };
  },

  /** Automatic adjustments measured on the proxy. `rect` is in source-normalised coordinates. */
  async analyze({ docId, kind, params, rect }) {
    const doc = await use(docId);
    const fn = { negative: core.auto_negative, filmBase: core.sample_density, neutral: core.neutral_balance, levels: core.auto_levels }[kind];
    return { values: Array.from(fn(JSON.stringify(params), doc.proxy, doc.pw, doc.ph, 4, new Float32Array(rect))) };
  },

  async export({ docId, params, denoise, orientation, crop, format, quality, chroma444, maxEdge }, progress) {
    const doc = await use(docId);
    const { width: w, height: h, linear } = doc.info;
    const [x0, y0, x1, y1] = toPixels(cropToSource(crop, orientation), w, h);

    if (format === 'jpeg-lossless') {
      const r = await pool.run('jpegLossless', { bytes: doc.bytes.slice(0), orientation, rect: [x0, y0, x1, y1] });
      return { result: r, transfer: [r.bytes.buffer] };
    }

    await use(docId, { master: true });
    const rw = x1 - x0;
    const rh = y1 - y0;

    // 0. Optional noise reduction of the cropped area (with context from outside the crop).
    let source = doc.master;
    let [sx, sy, sw] = [x0, y0, w];
    const denoising = isOn(denoise);
    const phase = denoising ? 0.5 : 0;
    if (denoising) {
      const m = core.denoise_margin(JSON.stringify(denoise));
      const [ex, ey] = [Math.max(0, x0 - m), Math.max(0, y0 - m)];
      const [ew, eh] = [Math.min(w, x1 + m) - ex, Math.min(h, y1 + m) - ey];
      source = await denoiseImage(extract(doc.master, w, 3, ex, ey, ew, eh), ew, eh, linear, denoise, noiseProfile(doc, denoise, 'master'), (p) => progress(p * phase));
      [sx, sy, sw] = [x0 - ex, y0 - ey, ew];
    }

    // 1. Split the crop into horizontal strips and develop them in parallel.
    const paramsJson = JSON.stringify(params);
    const rowsPerStrip = Math.max(16, Math.ceil(rh / (poolSize * 4)));
    const developed = new Uint16Array(rw * rh * 3);
    let done = 0;
    const strips = [];
    for (let row = 0; row < rh; row += rowsPerStrip) {
      const rows = Math.min(rowsPerStrip, rh - row);
      const strip = extract(source, sw, 3, sx, sy + row, rw, rows);
      strips.push(
        pool.run('process', { rgb: strip.buffer, params: paramsJson }, [strip.buffer]).then((out) => {
          developed.set(out, row * rw * 3);
          progress(phase + (1 - phase) * ((done += rows) / rh));
        }),
      );
    }
    await Promise.all(strips);

    // 2. Resize (in source orientation), orient and encode in one more job.
    const [ow, oh] = orientedSize(rw, rh, orientation);
    let [tw, th] = maxEdge ? fit(ow, oh, maxEdge) : [ow, oh];
    if (decompose(orientation).transpose) [tw, th] = [th, tw];
    const isJpeg = doc.bytes.byteLength > 2 && new Uint8Array(doc.bytes, 0, 2).every((b, i) => b === [0xff, 0xd8][i]);
    const r = await pool.run(
      'finish',
      { rgb: developed.buffer, width: rw, height: rh, orientation, outW: tw, outH: th, format, quality, chroma444, exifSource: isJpeg ? doc.bytes.slice(0) : null },
      [developed.buffer],
    );
    return { result: r, transfer: [r.bytes.buffer] };
  },

  /**
   * Developer tool: run the neural network on the proxy both on the GPU and in wasm and report
   * the largest difference (they should agree to within float rounding).
   */
  async compareNeural({ docId }) {
    const doc = await use(docId);
    const gpu = await getGpuNet();
    if (!gpu) return { error: 'no WebGPU' };
    const [w, h] = [Math.min(doc.pw, 256), Math.min(doc.ph, 256)];
    const rgb = extract(rgbaToRgb(doc.proxy), doc.pw, 3, 0, 0, w, h);
    const noise = [0.05, 0.05, 1.5];
    const input = core.nn_prepare(rgb, w, h, doc.info.linear, false, new Float32Array(noise));
    const t0 = performance.now();
    const gpuOut = core.nn_finish(rgb, w, h, doc.info.linear, false, await gpu.run(input, Math.ceil(w / 2), Math.ceil(h / 2)));
    const t1 = performance.now();
    const params = JSON.stringify({ method: 'neural', luma: 1, chroma: 1 });
    const cpuOut = await pool.run('denoise', { rgb: rgb.slice().buffer, w, h, linear: doc.info.linear, params, noise, core: [0, 0, w, h] });
    const t2 = performance.now();
    let maxDiff = 0;
    let changed = 0;
    for (let i = 0; i < rgb.length; i++) {
      maxDiff = Math.max(maxDiff, Math.abs(gpuOut[i] - cpuOut[i]));
      if (cpuOut[i] !== rgb[i]) changed++;
    }
    return { maxDiff, changedFraction: changed / rgb.length, gpuMs: t1 - t0, wasmMs: t2 - t1 };
  },
};

self.onmessage = async ({ data: { id, type, payload } }) => {
  try {
    const progress = (value) => self.postMessage({ id, progress: value });
    const out = (await handlers[type](payload, progress)) ?? {};
    const { result = out, transfer = [] } = 'result' in out ? out : { result: out };
    self.postMessage({ id, ok: true, result }, transfer);
  } catch (e) {
    console.error(e);
    self.postMessage({ id, ok: false, error: String(e?.message ?? e) });
  }
};

self.postMessage({ ready: true });
