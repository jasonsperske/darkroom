// UI controller: wires the DOM to the engine worker and the preview renderer.
//
// Data flow for one slider move:
//   input event → doc.params changes → requestRender()
//   → engine 'develop' (builds the tone LUT in wasm; computes a histogram)
//   → WebGPU: upload LUT + draw   |   no WebGPU: engine renders the view, we draw the bitmap
//
// Renders are coalesced: while one is in flight, further changes only mark the view dirty,
// so a fast slider drag never queues up stale work.

import { Engine } from './engine.js';
import { createGpuRenderer } from './render/webgpu.js';
import { defaultParams, defaultDenoise, createDoc, ASPECTS } from './state.js';
import * as geo from './geometry.js';
import { drawHistogram } from './ui/histogram.js';
import { attachCropOverlay, fitAspect } from './ui/crop.js';
import { createProgressBar } from './ui/progress.js';

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];
const els = {
  app: $('#app'),
  stage: $('#stage'),
  viewport: $('#viewport'),
  canvasGpu: $('#canvas-gpu'),
  canvasCpu: $('#canvas-cpu'),
  crop: $('#crop'),
  cropBox: $('#crop .crop-box'),
  loupe: $('#loupe'),
  empty: $('#empty'),
  progress: $('#progress'),
  toast: $('#toast'),
  filmstrip: $('#filmstrip'),
  panel: $('#panel'),
  controls: $('#controls'),
  fileName: $('#file-name'),
  fileMeta: $('#file-meta'),
  histogram: $('#histogram'),
  density: $('#density'),
  noiseReadout: $('#noise-readout'),
  aspect: $('#aspect'),
  snap: $('#snap'),
  mcu: $('#mcu'),
  cropSize: $('#crop-size'),
  format: $('#format'),
  quality: $('#quality'),
  chroma444: $('#chroma444'),
  maxEdge: $('#max-edge'),
  exportBtn: $('#export'),
  exportAll: $('#export-all'),
  applyAll: $('#apply-all'),
  status: $('#engine-status'),
  fileInput: $('#file-input'),
  help: $('#help'),
};

// The on-screen proxy: big enough for a sharp full-screen view, small enough to keep several
// images in memory. Full resolution is only used for export and the 100 % loupe.
const PROXY_EDGE = Math.min(3072, Math.max(1600, Math.round(Math.max(screen.width, screen.height) * (devicePixelRatio || 1))));

const engine = new Engine();
const progress = createProgressBar(els.progress);
const app = {
  docs: [],
  current: null,
  tool: null, // null | 'crop' | 'base' | 'neutral'
  compare: false,
  gpu: null,
  view: null,
  loupe: null,
  exporting: false,
  neuralOnGpu: null, // unknown until the neural method is first used
  bgIndex: 0,
};

// ---- Small helpers --------------------------------------------------------------------------

let toastTimer;
function toast(message, ms = 3000) {
  els.toast.textContent = message;
  els.toast.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => els.toast.classList.remove('show'), ms);
}

const storage = {
  get(key, fallback) {
    try {
      return JSON.parse(localStorage.getItem(`darkroom.${key}`)) ?? fallback;
    } catch {
      return fallback;
    }
  },
  set(key, value) {
    try {
      localStorage.setItem(`darkroom.${key}`, JSON.stringify(value));
    } catch {
      /* private mode etc. – preferences just won't persist */
    }
  },
};

function orientedSize(doc) {
  return geo.orientedSize(doc.info.width, doc.info.height, doc.orientation);
}

function effectiveParams(doc) {
  return app.compare ? { ...defaultParams(), inputLinear: doc.params.inputLinear } : doc.params;
}

// ---- Layout ---------------------------------------------------------------------------------

/** What part of the (oriented) image is on screen: everything while cropping, else the crop. */
function viewRegion(doc) {
  return app.tool === 'crop' ? [0, 0, 1, 1] : doc.crop;
}

function layout() {
  const doc = app.current;
  const ready = doc?.status === 'ready';
  els.viewport.hidden = !ready;
  els.empty.hidden = app.docs.length > 0;
  els.crop.hidden = !(ready && app.tool === 'crop');
  if (!ready) return;

  const [W, H] = orientedSize(doc);
  const region = viewRegion(doc);
  const rw = (region[2] - region[0]) * W;
  const rh = (region[3] - region[1]) * H;
  const stage = els.stage.getBoundingClientRect();
  const filmH = els.filmstrip.hidden ? 0 : els.filmstrip.offsetHeight;
  const margin = stage.width < 600 ? 8 : 28;
  const scale = Math.min((stage.width - 2 * margin) / rw, (stage.height - filmH - 2 * margin) / rh);
  const cssW = Math.max(1, Math.floor(rw * scale));
  const cssH = Math.max(1, Math.floor(rh * scale));
  Object.assign(els.viewport.style, { width: `${cssW}px`, height: `${cssH}px`, top: `${(stage.height - filmH) / 2}px` });

  const dpr = devicePixelRatio || 1;
  const canvas = app.gpu ? els.canvasGpu : els.canvasCpu;
  canvas.hidden = false;
  const pxW = Math.max(1, Math.round(cssW * dpr));
  const pxH = Math.max(1, Math.round(cssH * dpr));
  const resized = canvas.width !== pxW || canvas.height !== pxH;
  if (resized) [canvas.width, canvas.height] = [pxW, pxH];
  app.view = { region, cssW, cssH, pxW, pxH, W, H };

  if (app.tool === 'crop') cropOverlay.update(doc.crop);
  if (app.gpu) drawGpu();
  else requestRender();
}

new ResizeObserver(() => layout()).observe(els.stage);

// ---- Rendering ------------------------------------------------------------------------------

function drawGpu() {
  const doc = app.current;
  if (!app.gpu?.ready || !app.view || !doc) return;
  const p = effectiveParams(doc);
  app.gpu.draw({ crop: app.view.region, orientation: geo.decompose(doc.orientation), saturation: p.monochrome ? 0 : p.saturation });
}

const render = { inflight: false, dirty: false };

function requestRender() {
  render.dirty = true;
  if (!render.inflight) pumpRender();
}

async function pumpRender() {
  render.inflight = true;
  while (render.dirty) {
    render.dirty = false;
    const doc = app.current;
    if (doc?.status !== 'ready' || !app.view || (app.gpu && doc.gpuLoaded !== JSON.stringify(denoiseFor(doc)))) continue;
    const view = app.view;
    try {
      const payload = {
        docId: doc.id,
        params: effectiveParams(doc),
        denoise: denoiseFor(doc),
        orientation: doc.orientation,
        crop: view.region,
        outW: view.pxW,
        outH: view.pxH,
        cpu: !app.gpu,
      };
      const res = await progress.track('Rendering', (onProgress) => engine.call('develop', payload, { onProgress }));
      if (doc !== app.current) continue;
      if (app.gpu) {
        app.gpu.setLut(res.lut);
        drawGpu();
      } else if (res.bitmap.width === els.canvasCpu.width && res.bitmap.height === els.canvasCpu.height) {
        els.canvasCpu.getContext('2d').drawImage(res.bitmap, 0, 0);
        res.bitmap.close();
      } else {
        render.dirty = true; // canvas was resized meanwhile
      }
      drawHistogram(els.histogram, res.histogram);
    } catch (e) {
      toast(e.message);
    }
  }
  render.inflight = false;
}

// ---- Documents ------------------------------------------------------------------------------

let loadChain = Promise.resolve();

function addFiles(files) {
  const list = [...files].filter((f) => f.size > 0);
  if (!list.length) return;
  for (const file of list) {
    const doc = createDoc(file);
    app.docs.push(doc);
    // Decode one file at a time to keep peak memory reasonable when a whole roll is dropped.
    loadChain = loadChain.then(() => load(doc));
  }
  if (!app.current) select(app.docs[app.docs.length - list.length]);
  renderFilmstrip();
  layout();
}

async function load(doc) {
  try {
    const bytes = await doc.file.arrayBuffer();
    const { info } = await progress.track(`Decoding ${doc.name}`, () =>
      engine.call('open', { docId: doc.id, bytes, name: doc.name, proxyEdge: PROXY_EDGE }, { transfer: [bytes] }),
    );
    doc.info = info;
    doc.status = 'ready';
    doc.params.inputLinear = info.linear;
    doc.orientation = info.orientation;
    updateThumb(doc);
  } catch (e) {
    doc.status = 'error';
    doc.error = e.message;
    toast(`${doc.name}: ${e.message}`, 6000);
  }
  renderFilmstrip();
  if (doc === app.current) select(doc);
}

async function select(doc) {
  if (app.current && app.current !== doc && app.current.status === 'ready') updateThumb(app.current);
  closeLoupe();
  app.current = doc;
  app.tool = null;
  if (app.gpu) for (const d of app.docs) if (d !== doc) d.gpuLoaded = null;
  syncPanel();
  renderFilmstrip();
  drawHistogram(els.histogram, null);
  layout();
  if (doc.status !== 'ready') return;

  await uploadProxy(doc);
  requestRender();
}

/**
 * With WebGPU the proxy lives on the GPU, so it is re-uploaded whenever the noise reduction
 * settings change (`doc.gpuLoaded` remembers which settings the uploaded copy has).
 */
async function uploadProxy(doc) {
  const key = JSON.stringify(denoiseFor(doc));
  if (!app.gpu || doc.gpuLoaded === key) return;
  try {
    const label = doc.denoise.method === 'off' ? 'Loading preview' : 'Reducing noise';
    const { proxy, pw, ph } = await progress.track(label, (onProgress) =>
      engine.call('proxy', { docId: doc.id, denoise: denoiseFor(doc) }, { onProgress }),
    );
    if (app.current !== doc || JSON.stringify(denoiseFor(doc)) !== key) return;
    app.gpu.setImage(proxy, pw, ph);
    doc.gpuLoaded = key;
  } catch (e) {
    toast(e.message);
  }
}

function removeDoc(doc) {
  const i = app.docs.indexOf(doc);
  if (i < 0) return;
  app.docs.splice(i, 1);
  engine.call('close', { docId: doc.id });
  if (doc.thumb) URL.revokeObjectURL(doc.thumb);
  if (app.current === doc) {
    app.current = null;
    const next = app.docs[Math.min(i, app.docs.length - 1)];
    if (next) select(next);
    else {
      syncPanel();
      drawHistogram(els.histogram, null);
    }
  }
  renderFilmstrip();
  layout();
}

async function updateThumb(doc) {
  try {
    const { blob } = await engine.call('thumb', { docId: doc.id, params: doc.params, denoise: denoiseFor(doc), orientation: doc.orientation, crop: doc.crop });
    if (doc.thumb) URL.revokeObjectURL(doc.thumb);
    doc.thumb = URL.createObjectURL(blob);
    renderFilmstrip();
  } catch {
    /* thumbnails are best-effort */
  }
}

function renderFilmstrip() {
  const many = app.docs.length > 1;
  const wasHidden = els.filmstrip.hidden;
  els.filmstrip.hidden = !many;
  els.exportAll.hidden = !many;
  els.applyAll.hidden = !many;
  els.filmstrip.replaceChildren(
    ...app.docs.map((doc) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.title = doc.error ? `${doc.name}: ${doc.error}` : doc.name;
      b.setAttribute('aria-label', doc.name);
      b.setAttribute('aria-current', String(doc === app.current));
      b.classList.toggle('loading', doc.status === 'loading');
      b.classList.toggle('error', doc.status === 'error');
      if (doc.thumb) b.style.backgroundImage = `url(${doc.thumb})`;
      b.onclick = () => select(doc);
      return b;
    }),
  );
  if (wasHidden !== els.filmstrip.hidden) layout();
}

// ---- Panel ----------------------------------------------------------------------------------

const FORMAT = {
  exposure: (v) => `${v > 0 ? '+' : ''}${v.toFixed(2)}`,
  rangeStops: (v) => v.toFixed(1),
  contrast: (v) => v.toFixed(2),
  default: (v) => v.toFixed(2),
};

function setOutput(input) {
  const out = input.parentElement.querySelector('output');
  if (out) out.textContent = (FORMAT[input.dataset.param] ?? FORMAT.default)(Number(input.value));
}

function syncPanel() {
  const doc = app.current;
  const ready = doc?.status === 'ready';
  els.controls.classList.toggle('disabled', !ready);
  els.fileName.textContent = doc ? doc.name : 'No image';
  els.fileMeta.textContent = !doc ? '' : doc.status === 'error' ? doc.error : ready ? `${doc.info.width} × ${doc.info.height} · ${doc.info.description}` : 'Decoding…';
  if (!ready) return;
  const p = doc.params;

  for (const input of $$('input[data-param]')) {
    const k = input.dataset.param;
    if (input.type === 'checkbox') input.checked = !!p[k];
    else {
      input.value = p[k];
      setOutput(input);
    }
  }
  for (const b of $$('[data-film]')) b.setAttribute('aria-pressed', String((b.dataset.film === 'negative') === p.negative));
  for (const b of $$('[data-tool]')) b.setAttribute('aria-pressed', String(app.tool === b.dataset.tool));
  els.panel.classList.toggle('is-negative', p.negative);
  els.panel.classList.toggle('is-jpeg', !!doc.info.jpeg);
  els.density.textContent = p.negative
    ? `Dmin ${p.dmin.map((v) => v.toFixed(2)).join(' ')} · Dmax ${p.dmax.map((v) => v.toFixed(2)).join(' ')}`
    : '';

  syncDenoisePanel(doc);
  els.aspect.value = doc.aspect;
  const jpeg = doc.info.jpeg;
  els.mcu.textContent = jpeg ? `(${jpeg.mcuWidth}×${jpeg.mcuHeight} px)` : '';
  els.snap.checked = !!doc.snap;
  const losslessOption = els.format.querySelector('[value="jpeg-lossless"]');
  losslessOption.disabled = !jpeg;
  if (!jpeg && els.format.value === 'jpeg-lossless') els.format.value = 'jpeg';
  syncExportOptions();
  syncCropReadout();
}

function syncCropReadout() {
  const doc = app.current;
  if (doc?.status !== 'ready') return;
  const [w, h] = geo.cropPixelSize(doc.crop, doc.info.width, doc.info.height, doc.orientation);
  els.cropSize.textContent = `${w} × ${h} px`;
}

function syncExportOptions() {
  const f = els.format.value;
  for (const el of $$('.jpeg-opt')) el.hidden = f !== 'jpeg';
  for (const el of $$('.resize-opt')) el.hidden = f === 'jpeg-lossless';
  $('#lossless-note').hidden = f !== 'jpeg-lossless';
  setOutput(els.quality);
  storage.set('export', { format: f, quality: els.quality.value, chroma444: els.chroma444.checked, maxEdge: els.maxEdge.value });
}

function changed({ thumb = false } = {}) {
  syncPanel();
  requestRender();
  if (thumb && app.current) updateThumb(app.current);
}

// Sliders and checkboxes bound to develop parameters.
for (const input of $$('input[data-param]')) {
  const k = input.dataset.param;
  if (input.type === 'range') {
    input.addEventListener('input', () => {
      if (!app.current?.info) return;
      app.current.params[k] = Number(input.value);
      setOutput(input);
      requestRender();
    });
    input.addEventListener('dblclick', () => {
      if (!app.current?.info) return;
      app.current.params[k] = defaultParams()[k];
      changed();
    });
  } else {
    input.addEventListener('change', async () => {
      const doc = app.current;
      if (!doc?.info) return;
      doc.params[k] = input.checked;
      if (k === 'inputLinear' && doc.params.negative) await autoNegative(doc);
      changed();
    });
  }
}
els.quality.addEventListener('input', () => setOutput(els.quality));

for (const b of $$('[data-film]')) {
  b.addEventListener('click', async () => {
    const doc = app.current;
    if (!doc?.info) return;
    const negative = b.dataset.film === 'negative';
    if (negative === doc.params.negative) return;
    doc.params.negative = negative;
    Object.assign(doc.params, { wb: [1, 1, 1], temperature: 0, tint: 0, black: 0, white: 1 });
    if (negative) await autoNegative(doc);
    changed({ thumb: true });
    // Negatives are denoised in density space, positives in gamma: redo noise reduction.
    if (doc.denoise.method !== 'off') denoiseChanged(doc);
  });
}

/** Rectangle used for automatic analysis: the crop, inset to avoid film rebate and holder. */
function analysisRect(doc) {
  const c = doc.crop;
  const full = c[2] - c[0] > 0.98 && c[3] - c[1] > 0.98;
  const inset = full ? 0.08 : 0.02;
  const dx = (c[2] - c[0]) * inset;
  const dy = (c[3] - c[1]) * inset;
  return geo.cropToSource([c[0] + dx, c[1] + dy, c[2] - dx, c[3] - dy], doc.orientation);
}

async function analyze(doc, kind, rect = analysisRect(doc)) {
  const { values } = await engine.call('analyze', { docId: doc.id, kind, params: doc.params, rect });
  return values;
}

async function autoNegative(doc) {
  const v = await analyze(doc, 'negative');
  doc.params.dmin = v.slice(0, 3);
  doc.params.dmax = v.slice(3, 6);
}

$('#auto-negative').addEventListener('click', async () => {
  const doc = app.current;
  if (!doc?.info) return;
  Object.assign(doc.params, { wb: [1, 1, 1], temperature: 0, tint: 0 });
  await autoNegative(doc);
  changed({ thumb: true });
});

$('#auto-levels').addEventListener('click', async () => {
  const doc = app.current;
  if (!doc?.info) return;
  const [black, white] = await analyze(doc, 'levels');
  Object.assign(doc.params, { black, white });
  changed({ thumb: true });
});

$('#reset-wb').addEventListener('click', () => {
  if (!app.current?.info) return;
  Object.assign(app.current.params, { wb: [1, 1, 1], temperature: 0, tint: 0 });
  changed();
});

$('#reset').addEventListener('click', () => {
  const doc = app.current;
  if (!doc?.info) return;
  doc.params = { ...defaultParams(), inputLinear: doc.info.linear };
  doc.denoise = defaultDenoise();
  uploadProxy(doc);
  doc.orientation = doc.info.orientation;
  doc.crop = [0, 0, 1, 1];
  doc.aspect = 'free';
  doc.aspectPortrait = null;
  setTool(null);
  changed({ thumb: true });
  layout();
});

els.applyAll.addEventListener('click', () => {
  const src = app.current;
  if (!src?.info) return;
  for (const doc of app.docs) {
    if (doc === src || doc.status !== 'ready') continue;
    doc.params = { ...structuredClone(src.params), inputLinear: doc.params.inputLinear };
    doc.denoise = { ...src.denoise };
    updateThumb(doc);
  }
  toast(`Settings applied to ${app.docs.length - 1} images`);
});

// ---- Noise reduction ----------------------------------------------------------------------------

/**
 * Noise reduction settings as the engine needs them. Negatives are denoised in density space
 * (before inversion), where film grain is most uniform – see core/src/denoise/mod.rs.
 */
function denoiseFor(doc) {
  return { ...doc.denoise, negative: doc.params.negative };
}

/** The measured grain of the full-resolution image, in the working space currently in use. */
function measuredNoise(doc) {
  const m = doc.info.noise?.[doc.params.negative ? 'negative' : 'positive'].master ?? [0, 0, 0];
  return { luma: m[0], chroma: m[1], size: m[2] };
}

function syncDenoisePanel(doc) {
  const d = doc.denoise;
  for (const b of $$('[data-denoise-method]')) b.setAttribute('aria-pressed', String(b.dataset.denoiseMethod === d.method));
  const measured = measuredNoise(doc);
  for (const input of $$('input[data-denoise]')) {
    const k = input.dataset.denoise;
    const out = input.parentElement.querySelector('output');
    if (k === 'grainSize' && d.grainSize == null) {
      input.value = measured.size;
      out.textContent = 'auto';
    } else {
      input.value = d[k];
      out.textContent = Number(input.value).toFixed(2);
    }
  }
  els.panel.classList.toggle('is-denoising', d.method !== 'off');
  // Positives: noise on a familiar 0–255 scale. Negatives: grain in optical density units
  // (the working space stores density / 4).
  const amount = (v) => (doc.params.negative ? `${(v * 4).toFixed(3)} D` : `${(v * 255).toFixed(1)}/255`);
  let text = `Measured ${doc.params.negative ? 'grain' : 'noise'}: luminance ${amount(measured.luma)}, colour ${amount(measured.chroma)}, size ${measured.size.toFixed(1)} px`;
  if (d.method === 'neural' && app.neuralOnGpu !== null) text += app.neuralOnGpu ? ' · network on WebGPU' : ' · network on CPU (slow)';
  if (d.method !== 'off') text += ' · inspect at 100 % (Z)';
  els.noiseReadout.textContent = text;
}

/** Denoising changes the pixels everything else starts from, so refresh every view. */
async function denoiseChanged(doc) {
  syncDenoisePanel(doc);
  await uploadProxy(doc);
  requestRender();
  if (app.loupe) await renderLoupe();
  updateThumb(doc);
}

for (const b of $$('[data-denoise-method]')) {
  b.addEventListener('click', async () => {
    const doc = app.current;
    if (!doc?.info) return;
    doc.denoise.method = b.dataset.denoiseMethod;
    if (doc.denoise.method === 'neural' && app.neuralOnGpu === null) {
      app.neuralOnGpu = (await engine.call('neuralBackend')).gpu;
    }
    denoiseChanged(doc);
  });
}
for (const input of $$('input[data-denoise]')) {
  const out = input.parentElement.querySelector('output');
  input.addEventListener('input', () => (out.textContent = Number(input.value).toFixed(2)));
  // (Grain size shows "auto" until the user moves it; see syncDenoisePanel.)
  // Denoising takes from milliseconds to seconds, so it runs when the slider is released.
  input.addEventListener('change', () => {
    const doc = app.current;
    if (!doc?.info) return;
    doc.denoise[input.dataset.denoise] = Number(input.value);
    denoiseChanged(doc);
  });
  input.addEventListener('dblclick', () => {
    const doc = app.current;
    if (!doc?.info) return;
    doc.denoise[input.dataset.denoise] = defaultDenoise()[input.dataset.denoise]; // grainSize → auto
    denoiseChanged(doc);
  });
}

// ---- Tools: crop, film base picker, grey picker ----------------------------------------------

function setTool(tool) {
  if (!app.current?.info) tool = null;
  const was = app.tool;
  app.tool = app.tool === tool ? null : tool;
  els.stage.classList.toggle('tool-base', app.tool === 'base');
  els.stage.classList.toggle('tool-neutral', app.tool === 'neutral');
  for (const b of $$('[data-tool]')) b.setAttribute('aria-pressed', String(app.tool === b.dataset.tool));
  if (was === 'crop' || app.tool === 'crop') {
    layout();
    requestRender();
    if (was === 'crop') updateThumb(app.current);
  }
}

for (const b of $$('[data-tool]')) b.addEventListener('click', () => setTool(b.dataset.tool));

function aspectRatio(doc) {
  const def = ASPECTS.find((a) => a.id === doc.aspect);
  if (!def?.ratio) return null;
  const [W, H] = orientedSize(doc);
  const r = def.ratio === 'original' ? Math.max(W, H) / Math.min(W, H) : def.ratio;
  const cropW = (doc.crop[2] - doc.crop[0]) * W;
  const cropH = (doc.crop[3] - doc.crop[1]) * H;
  const portrait = doc.aspectPortrait ?? (def.ratio === 'original' ? H > W : cropH > cropW);
  return portrait ? 1 / r : r;
}

function snapCrop(doc) {
  const j = doc.info.jpeg;
  if (doc.snap && j) doc.crop = geo.snapCropToGrid(doc.crop, doc.orientation, doc.info.width, doc.info.height, j.mcuWidth, j.mcuHeight);
}

const cropOverlay = attachCropOverlay(
  els.crop,
  els.cropBox,
  () => {
    const doc = app.current;
    const [W, H] = orientedSize(doc);
    return { crop: doc.crop, ratio: aspectRatio(doc), W, H };
  },
  (crop, final) => {
    const doc = app.current;
    doc.crop = crop;
    if (final) {
      snapCrop(doc);
      requestRender();
    }
    cropOverlay.update(doc.crop);
    syncCropReadout();
  },
);

els.aspect.replaceChildren(...ASPECTS.map((a) => new Option(a.label, a.id)));
els.aspect.addEventListener('change', () => {
  const doc = app.current;
  if (!doc?.info) return;
  doc.aspect = els.aspect.value;
  doc.aspectPortrait = null;
  applyAspect(doc);
});

function applyAspect(doc) {
  const ratio = aspectRatio(doc);
  if (ratio) {
    const [W, H] = orientedSize(doc);
    doc.crop = fitAspect(doc.crop, ratio, W, H);
    snapCrop(doc);
  }
  if (app.tool !== 'crop') setTool('crop');
  else {
    cropOverlay.update(doc.crop);
    requestRender();
  }
  syncCropReadout();
}

function swapAspect() {
  const doc = app.current;
  if (!doc?.info) return;
  const ratio = aspectRatio(doc);
  if (ratio) {
    doc.aspectPortrait = ratio > 1;
  } else {
    const [W, H] = orientedSize(doc);
    const w = (doc.crop[2] - doc.crop[0]) * W;
    const h = (doc.crop[3] - doc.crop[1]) * H;
    doc.crop = fitAspect(doc.crop, h / w, W, H);
  }
  applyAspect(doc);
}
$('#aspect-swap').addEventListener('click', swapAspect);

els.snap.addEventListener('change', () => {
  const doc = app.current;
  if (!doc?.info) return;
  doc.snap = els.snap.checked;
  snapCrop(doc);
  if (app.tool === 'crop') cropOverlay.update(doc.crop);
  else layout();
  syncCropReadout();
  requestRender();
});

function rotate(op) {
  const doc = app.current;
  if (!doc?.info) return;
  doc.orientation = geo.compose(doc.orientation, op);
  doc.crop = geo.transformCrop(doc.crop, op);
  if (doc.aspectPortrait !== null && (op === 'cw' || op === 'ccw')) doc.aspectPortrait = !doc.aspectPortrait;
  closeLoupe();
  layout();
  syncCropReadout();
  requestRender();
  updateThumb(doc);
}
for (const b of $$('[data-op]')) b.addEventListener('click', () => rotate(b.dataset.op));

/** Pointer position on the viewport → displayed-image normalised coordinates. */
function pointToImage(e) {
  const r = els.viewport.getBoundingClientRect();
  const region = app.view.region;
  const u = (e.clientX - r.left) / r.width;
  const v = (e.clientY - r.top) / r.height;
  return [region[0] + u * (region[2] - region[0]), region[1] + v * (region[3] - region[1])];
}

els.viewport.addEventListener('pointerdown', async (e) => {
  const doc = app.current;
  if (!doc?.info || (app.tool !== 'base' && app.tool !== 'neutral')) return;
  const [u, v] = pointToImage(e);
  const s = 0.006; // sample a small square, not a single (noisy) pixel
  const rect = geo.cropToSource([Math.max(0, u - s), Math.max(0, v - s), Math.min(1, u + s), Math.min(1, v + s)], doc.orientation);
  if (app.tool === 'base') {
    doc.params.dmin = await analyze(doc, 'filmBase', rect);
    Object.assign(doc.params, { wb: [1, 1, 1], temperature: 0, tint: 0 });
    toast('Film base set');
  } else {
    doc.params.wb = await analyze(doc, 'neutral', rect);
    Object.assign(doc.params, { temperature: 0, tint: 0 });
  }
  setTool(null);
  changed({ thumb: true });
});

// ---- Loupe: 100 % view from the full-resolution master ---------------------------------------

async function openLoupe(center) {
  const doc = app.current;
  if (!doc?.info) return;
  app.loupe = { center };
  els.loupe.hidden = false;
  await renderLoupe();
}

function closeLoupe() {
  app.loupe = null;
  els.loupe.hidden = true;
}

async function renderLoupe() {
  const doc = app.current;
  if (!app.loupe || !doc) return;
  const dpr = devicePixelRatio || 1;
  const rect = els.stage.getBoundingClientRect();
  const screenW = Math.round(rect.width * dpr);
  const screenH = Math.round(rect.height * dpr);
  const [W, H] = orientedSize(doc);
  // One image pixel per device pixel; never larger than the image itself.
  const outW = Math.min(screenW, W);
  const outH = Math.min(screenH, H);
  const hw = outW / W / 2;
  const hh = outH / H / 2;
  let [cx, cy] = app.loupe.center;
  cx = Math.min(Math.max(cx, hw), 1 - hw);
  cy = Math.min(Math.max(cy, hh), 1 - hh);
  app.loupe.center = [cx, cy];
  try {
    const payload = {
      docId: doc.id,
      params: effectiveParams(doc),
      denoise: denoiseFor(doc),
      orientation: doc.orientation,
      crop: [cx - hw, cy - hh, cx + hw, cy + hh],
      outW,
      outH,
    };
    const { bitmap } = await progress.track('Rendering 100 %', (onProgress) => engine.call('loupe', payload, { onProgress }));
    if (!app.loupe) return;
    els.loupe.width = screenW;
    els.loupe.height = screenH;
    els.loupe.style.transform = '';
    const ctx = els.loupe.getContext('2d');
    ctx.clearRect(0, 0, screenW, screenH);
    ctx.drawImage(bitmap, Math.round((screenW - outW) / 2), Math.round((screenH - outH) / 2));
    bitmap.close();
  } catch (e) {
    toast(e.message);
    closeLoupe();
  }
}

els.viewport.addEventListener('dblclick', (e) => {
  if (app.tool) return;
  openLoupe(pointToImage(e));
});
els.loupe.addEventListener('dblclick', closeLoupe);
{
  let pan = null;
  els.loupe.addEventListener('pointerdown', (e) => {
    pan = { x: e.clientX, y: e.clientY };
    els.loupe.setPointerCapture(e.pointerId);
  });
  els.loupe.addEventListener('pointermove', (e) => {
    if (pan) els.loupe.style.transform = `translate(${e.clientX - pan.x}px, ${e.clientY - pan.y}px)`;
  });
  els.loupe.addEventListener('pointerup', (e) => {
    if (!pan || !app.loupe) return;
    const dpr = devicePixelRatio || 1;
    const [W, H] = orientedSize(app.current);
    const [dx, dy] = [(e.clientX - pan.x) * dpr, (e.clientY - pan.y) * dpr];
    pan = null;
    if (Math.abs(dx) + Math.abs(dy) < 2) return;
    app.loupe.center = [app.loupe.center[0] - dx / W, app.loupe.center[1] - dy / H];
    renderLoupe();
  });
}

// ---- Export ---------------------------------------------------------------------------------

const MIME = { jpeg: 'image/jpeg', 'jpeg-lossless': 'image/jpeg', png8: 'image/png', png16: 'image/png', tiff16: 'image/tiff' };
const EXT = { jpeg: 'jpg', 'jpeg-lossless': 'jpg', png8: 'png', png16: 'png', tiff16: 'tif' };

function download(blob, name) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 60_000);
}

async function exportDoc(doc, label = '') {
  const format = els.format.value;
  if (format === 'jpeg-lossless' && !doc.info.jpeg) throw new Error(`${doc.name}: lossless export needs a JPEG`);
  const maxEdge = parseInt(els.maxEdge.value, 10) || 0;
  const res = await progress.track(`Exporting ${doc.name}`, (onBar) => engine.call(
    'export',
    {
      docId: doc.id,
      params: doc.params,
      denoise: denoiseFor(doc),
      orientation: doc.orientation,
      crop: doc.crop,
      format,
      quality: Number(els.quality.value),
      chroma444: els.chroma444.checked,
      maxEdge,
    },
    {
      onProgress: (p) => {
        onBar(p);
        els.exportBtn.textContent = `${label}${Math.round(p * 100)} %`;
      },
    },
  ));
  const base = doc.name.replace(/\.[^.]+$/, '');
  const suffix = format === 'jpeg-lossless' ? 'crop' : 'darkroom';
  download(new Blob([res.bytes], { type: MIME[format] }), `${base}-${suffix}.${EXT[format]}`);
  return res;
}

async function runExport(docs) {
  if (app.exporting) return;
  app.exporting = true;
  els.exportBtn.disabled = els.exportAll.disabled = true;
  try {
    for (const [i, doc] of docs.entries()) {
      const label = docs.length > 1 ? `${i + 1}/${docs.length} · ` : '';
      els.exportBtn.textContent = `${label}…`;
      const res = await exportDoc(doc, label);
      toast(`Exported ${res.width} × ${res.height}`);
    }
  } catch (e) {
    toast(e.message, 6000);
  } finally {
    app.exporting = false;
    els.exportBtn.disabled = els.exportAll.disabled = false;
    els.exportBtn.textContent = 'Export';
  }
}

els.exportBtn.addEventListener('click', () => app.current?.info && runExport([app.current]));
els.exportAll.addEventListener('click', () => runExport(app.docs.filter((d) => d.status === 'ready')));
for (const el of [els.format, els.chroma444, els.maxEdge, els.quality]) el.addEventListener('change', syncExportOptions);

// ---- Files: open dialog, drag & drop ----------------------------------------------------------

$('#open').addEventListener('click', () => els.fileInput.click());
els.fileInput.addEventListener('change', () => {
  addFiles(els.fileInput.files);
  els.fileInput.value = '';
});
addEventListener('dragover', (e) => {
  e.preventDefault();
  els.stage.classList.add('drop');
});
addEventListener('dragleave', (e) => {
  if (!e.relatedTarget) els.stage.classList.remove('drop');
});
addEventListener('drop', (e) => {
  e.preventDefault();
  els.stage.classList.remove('drop');
  addFiles(e.dataTransfer.files);
});

// ---- Keyboard -------------------------------------------------------------------------------

const BACKGROUNDS = ['', '#000', '#808080', '#f2f2f2'];

addEventListener('keydown', (e) => {
  const t = e.target;
  if (t.tagName === 'SELECT' || (t.tagName === 'INPUT' && t.type !== 'range' && t.type !== 'checkbox') || e.metaKey || e.ctrlKey || e.altKey) return;
  const onRange = t.type === 'range';
  const doc = app.current;
  switch (e.key) {
    case 'o':
      els.fileInput.click();
      break;
    case 'c':
      setTool('crop');
      break;
    case 'Enter':
      if (app.tool === 'crop') setTool('crop');
      else return;
      break;
    case 'Escape':
      if (app.loupe) closeLoupe();
      else if (app.tool) setTool(app.tool);
      else return;
      break;
    case 'x':
      swapAspect();
      break;
    case 'r':
      rotate('cw');
      break;
    case 'R':
      rotate('ccw');
      break;
    case 'f':
      rotate('flipH');
      break;
    case '\\':
      if (!app.compare && doc?.info) {
        app.compare = true;
        requestRender();
      }
      break;
    case 'z':
      if (app.loupe) closeLoupe();
      else if (doc?.info && app.view) openLoupe([(app.view.region[0] + app.view.region[2]) / 2, (app.view.region[1] + app.view.region[3]) / 2]);
      break;
    case 'b':
      app.bgIndex = (app.bgIndex + 1) % BACKGROUNDS.length;
      els.stage.style.setProperty('--stage-bg', BACKGROUNDS[app.bgIndex] || null);
      break;
    case 'e':
      if (doc?.info) runExport([doc]);
      break;
    case 'ArrowLeft':
    case 'ArrowRight': {
      if (onRange || !doc) return;
      const i = app.docs.indexOf(doc) + (e.key === 'ArrowLeft' ? -1 : 1);
      if (app.docs[i]) select(app.docs[i]);
      break;
    }
    case 'Delete':
    case 'Backspace':
      if (doc && !onRange) removeDoc(doc);
      break;
    case 'Tab':
      if (t !== document.body) return;
      els.app.classList.toggle('panel-hidden');
      storage.set('panelHidden', els.app.classList.contains('panel-hidden'));
      break;
    case '?':
      els.help.showModal();
      break;
    default:
      return;
  }
  e.preventDefault();
});

addEventListener('keyup', (e) => {
  if (e.key === '\\' && app.compare) {
    app.compare = false;
    requestRender();
  }
});

$('#help-btn').addEventListener('click', () => els.help.showModal());

// ---- Start ----------------------------------------------------------------------------------

{
  const prefs = storage.get('export', null);
  if (prefs) {
    els.format.value = prefs.format ?? 'jpeg';
    els.quality.value = prefs.quality ?? 95;
    els.chroma444.checked = prefs.chroma444 ?? true;
    els.maxEdge.value = prefs.maxEdge ?? '';
  }
  if (storage.get('panelHidden', false)) els.app.classList.add('panel-hidden');
  syncPanel();
  syncExportOptions();
  layout();
}

try {
  await engine.ready;
  try {
    // `?cpu` in the URL forces the Canvas 2D fallback, to compare the two render paths.
    app.gpu = new URLSearchParams(location.search).has('cpu') ? null : await createGpuRenderer(els.canvasGpu);
  } catch (e) {
    console.warn('WebGPU unavailable, using Canvas 2D:', e);
    app.gpu = null;
  }
  const st = await engine.call('status');
  els.status.textContent = `${app.gpu ? 'WebGPU' : 'Canvas 2D'} · wasm ${st.simd ? 'SIMD' : 'scalar'} · ${st.workers} worker${st.workers > 1 ? 's' : ''}`;
  layout();
} catch (e) {
  els.status.textContent = 'Engine failed to start';
  toast(`Could not start: ${e.message}`, 10000);
}

// For exploring from the browser console, e.g.
//   await darkroom.engine.call('compareNeural', { docId: darkroom.app.current.id })
globalThis.darkroom = { app, engine };
