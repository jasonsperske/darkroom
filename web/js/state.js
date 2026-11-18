// Document model: one entry per dropped file, holding its edit settings.
// Develop parameters mirror `Params` in core/src/develop/params.rs (serialised as JSON).

export function defaultParams() {
  return {
    inputLinear: false,
    negative: false,
    dmin: [0, 0, 0],
    dmax: [1.5, 1.5, 1.5],
    rangeStops: 6,
    wb: [1, 1, 1],
    temperature: 0,
    tint: 0,
    exposure: 0,
    black: 0,
    white: 1,
    contrast: 1,
    saturation: 1,
    monochrome: false,
  };
}

/** Noise reduction settings – mirrors `DenoiseParams` in core/src/denoise/mod.rs. */
export function defaultDenoise() {
  // grainSize: null = use the measured grain size; a number (px) overrides it.
  return { method: 'off', luma: 1, chroma: 1, grainSize: null };
}

/** Aspect ratios (long side : short side). Orientation follows the current crop. */
export const ASPECTS = [
  { id: 'free', label: 'Free', ratio: null },
  { id: 'original', label: 'Original', ratio: 'original' },
  { id: '1:1', label: '1:1 · 6×6', ratio: 1 },
  { id: '3:2', label: '3:2 · 35mm, 4×6', ratio: 3 / 2 },
  { id: '4:3', label: '4:3 · 6×4.5', ratio: 4 / 3 },
  { id: '5:4', label: '5:4 · 4×5, 8×10', ratio: 5 / 4 },
  { id: '7:5', label: '7:5 · 5×7', ratio: 7 / 5 },
  { id: '14:11', label: '14:11 · 11×14', ratio: 14 / 11 },
  { id: '11:8.5', label: 'Letter', ratio: 11 / 8.5 },
  { id: 'iso', label: 'A4, A3 (√2)', ratio: Math.SQRT2 },
  { id: '16:9', label: '16:9', ratio: 16 / 9 },
  { id: '65:24', label: 'XPan 65:24', ratio: 65 / 24 },
];

let nextId = 1;

export function createDoc(file) {
  return {
    id: nextId++,
    file,
    name: file.name,
    status: 'loading', // loading | ready | error
    error: null,
    info: null, // { width, height, orientation, linear, description, jpeg }
    params: defaultParams(),
    denoise: defaultDenoise(),
    orientation: 1, // total EXIF-style orientation (file's own + user rotations)
    crop: [0, 0, 1, 1], // in displayed (oriented) normalised coordinates
    aspect: 'free',
    aspectPortrait: null, // null = follow the crop's current shape
    thumb: null, // object URL
  };
}
