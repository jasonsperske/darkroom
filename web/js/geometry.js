// Orientation and crop geometry – the JavaScript twin of core/src/geometry.rs.
//
// An orientation is one of the 8 EXIF codes. Internally each is "transpose, then flip X, then
// flip Y", which as a 2×2 matrix A maps centred output coordinates to centred source
// coordinates:  source = A · output.  Every A is a signed permutation matrix, so composing
// two orientations is a matrix product and inverting one is a transpose.

const EXIF = {
  //       transpose flipX  flipY
  1: [false, false, false],
  2: [false, true, false],
  3: [false, true, true],
  4: [false, false, true],
  5: [true, false, false],
  6: [true, true, false],
  7: [true, true, true],
  8: [true, false, true],
};

export function decompose(code) {
  const [transpose, flipX, flipY] = EXIF[code] ?? EXIF[1];
  return { transpose, flipX, flipY };
}

function toMatrix(code) {
  const { transpose, flipX, flipY } = decompose(code);
  const sx = flipX ? -1 : 1;
  const sy = flipY ? -1 : 1;
  return transpose ? [[0, sy], [sx, 0]] : [[sx, 0], [0, sy]];
}

function fromMatrix(a) {
  const transpose = a[0][1] !== 0;
  const flipX = transpose ? a[1][0] < 0 : a[0][0] < 0;
  const flipY = transpose ? a[0][1] < 0 : a[1][1] < 0;
  return Number(Object.keys(EXIF).find((k) => {
    const e = EXIF[k];
    return e[0] === transpose && e[1] === flipX && e[2] === flipY;
  }));
}

const mul = (a, b) => [
  [a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
  [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]],
];
const apply = (a, [x, y]) => [a[0][0] * x + a[0][1] * y, a[1][0] * x + a[1][1] * y];
const transpose = (a) => [[a[0][0], a[1][0]], [a[0][1], a[1][1]]];

/** User operations on the *displayed* image, as "new output → old output" matrices. */
const OPS = {
  cw: [[0, 1], [-1, 0]], // rotate 90° clockwise
  ccw: [[0, -1], [1, 0]], // rotate 90° counter-clockwise
  flipH: [[-1, 0], [0, 1]],
  flipV: [[1, 0], [0, -1]],
};

/** Orientation after applying a user operation to what is currently displayed. */
export function compose(code, op) {
  return fromMatrix(mul(toMatrix(code), OPS[op]));
}

/** Move a crop rectangle along with the image when an operation is applied. */
export function transformCrop(crop, op) {
  const inv = transpose(OPS[op]);
  const a = apply(inv, [crop[0] - 0.5, crop[1] - 0.5]);
  const b = apply(inv, [crop[2] - 0.5, crop[3] - 0.5]);
  return [Math.min(a[0], b[0]) + 0.5, Math.min(a[1], b[1]) + 0.5, Math.max(a[0], b[0]) + 0.5, Math.max(a[1], b[1]) + 0.5];
}

/** The orientation that undoes `code` (only the two 90° rotations differ from themselves). */
export function inverse(code) {
  return fromMatrix(transpose(toMatrix(code)));
}

/** Displayed size of a `w × h` source. */
export function orientedSize(w, h, code) {
  return decompose(code).transpose ? [h, w] : [w, h];
}

/** Normalised output (u, v) → normalised source coordinates. */
export function mapNormalized(code, u, v) {
  const { transpose, flipX, flipY } = decompose(code);
  const x = flipX ? 1 - u : u;
  const y = flipY ? 1 - v : v;
  return transpose ? [y, x] : [x, y];
}

/** A crop in output-normalised coordinates → the same region in source-normalised coordinates. */
export function cropToSource(crop, code) {
  const a = mapNormalized(code, crop[0], crop[1]);
  const b = mapNormalized(code, crop[2], crop[3]);
  return [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[0], b[0]), Math.max(a[1], b[1])];
}

/** Source-normalised rect → whole source pixels [x0, y0, x1, y1). */
export function toPixels(rect, w, h) {
  const x0 = Math.min(w - 1, Math.max(0, Math.round(rect[0] * w)));
  const y0 = Math.min(h - 1, Math.max(0, Math.round(rect[1] * h)));
  const x1 = Math.max(x0 + 1, Math.min(w, Math.round(rect[2] * w)));
  const y1 = Math.max(y0 + 1, Math.min(h, Math.round(rect[3] * h)));
  return [x0, y0, x1, y1];
}

/**
 * Snap a crop so its edges lie on the JPEG MCU grid of the *source* image (or on the image
 * edge). This is what makes a lossless crop possible – see core/src/jpeg/transform.rs.
 */
export function snapCropToGrid(crop, code, w, h, mcuW, mcuH) {
  const src = cropToSource(crop, code);
  const snap = (v, size, grid) => {
    const px = v * size;
    if (size - px < grid / 2) return 1; // keep the image edge
    return (Math.round(px / grid) * grid) / size;
  };
  let [x0, y0, x1, y1] = [snap(src[0], w, mcuW), snap(src[1], h, mcuH), snap(src[2], w, mcuW), snap(src[3], h, mcuH)];
  if (x1 <= x0) x1 = Math.min(1, x0 + mcuW / w);
  if (y1 <= y0) y1 = Math.min(1, y0 + mcuH / h);
  // Back to output coordinates: the inverse mapping of an orientation is the same formula
  // with the inverse orientation, which for a signed permutation is its transpose.
  const inverse = fromMatrix(transpose(toMatrix(code)));
  return cropToSource([x0, y0, x1, y1], inverse);
}

/** Aspect-correct helpers for the crop tool: crop size in displayed pixels. */
export function cropPixelSize(crop, w, h, code) {
  const [ow, oh] = orientedSize(w, h, code);
  return [Math.round((crop[2] - crop[0]) * ow), Math.round((crop[3] - crop[1]) * oh)];
}
