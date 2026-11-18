// Interactive crop rectangle.
//
// The crop is stored in *displayed* normalised coordinates [x0, y0, x1, y1] (0…1 across the
// oriented image). Aspect ratios are about pixels, though, so all maths here happens in
// displayed pixel units (W × H = oriented image size) and is converted back at the end.

const MIN = 16; // smallest crop side, in image pixels

/** The largest rectangle of `ratio` (w/h) centred on (cx, cy) that fits inside W × H. */
export function fitAspect(crop, ratio, W, H) {
  const cx = ((crop[0] + crop[2]) / 2) * W;
  const cy = ((crop[1] + crop[3]) / 2) * H;
  let w = W;
  let h = w / ratio;
  if (h > H) {
    h = H;
    w = h * ratio;
  }
  // Keep the centre if possible, otherwise slide inside the image.
  const x0 = Math.min(Math.max(cx - w / 2, 0), W - w);
  const y0 = Math.min(Math.max(cy - h / 2, 0), H - h);
  return [x0 / W, y0 / H, (x0 + w) / W, (y0 + h) / H];
}

/**
 * Compute a new crop while dragging `handle` ('n', 'se', 'move', …) to pointer position
 * (px, py) in image pixels. `start` is the crop at drag start, `grab` the pointer position
 * at drag start, `ratio` the locked aspect (w/h) or null.
 */
export function dragCrop(start, handle, grab, [px, py], ratio, W, H) {
  let [x0, y0, x1, y1] = [start[0] * W, start[1] * H, start[2] * W, start[3] * H];
  px = Math.min(Math.max(px, 0), W);
  py = Math.min(Math.max(py, 0), H);

  if (handle === 'move') {
    const w = x1 - x0;
    const h = y1 - y0;
    const nx = Math.min(Math.max(x0 + px - grab[0], 0), W - w);
    const ny = Math.min(Math.max(y0 + py - grab[1], 0), H - h);
    return [nx / W, ny / H, (nx + w) / W, (ny + h) / H];
  }

  const left = handle.includes('w');
  const right = handle.includes('e');
  const top = handle.includes('n');
  const bottom = handle.includes('s');

  if (!ratio) {
    if (left) x0 = Math.min(px, x1 - MIN);
    if (right) x1 = Math.max(px, x0 + MIN);
    if (top) y0 = Math.min(py, y1 - MIN);
    if (bottom) y1 = Math.max(py, y0 + MIN);
    return [x0 / W, y0 / H, x1 / W, y1 / H];
  }

  if ((left || right) && (top || bottom)) {
    // Corner: the opposite corner is the anchor; the size follows whichever axis the pointer
    // has moved further along, limited by the image edge in that direction.
    const ax = left ? x1 : x0;
    const ay = top ? y1 : y0;
    let w = Math.max(MIN, Math.abs(px - ax));
    let h = Math.max(MIN, Math.abs(py - ay));
    if (w / h > ratio) h = w / ratio;
    else w = h * ratio;
    const maxW = left ? ax : W - ax;
    const maxH = top ? ay : H - ay;
    if (w > maxW) [w, h] = [maxW, maxW / ratio];
    if (h > maxH) [w, h] = [maxH * ratio, maxH];
    x0 = left ? ax - w : ax;
    y0 = top ? ay - h : ay;
    return [x0 / W, y0 / H, (x0 + w) / W, (y0 + h) / H];
  }

  // Edge: that edge follows the pointer, the other dimension grows about the centre.
  const cx = (x0 + x1) / 2;
  const cy = (y0 + y1) / 2;
  let w;
  let h;
  if (left || right) {
    const ax = left ? x1 : x0;
    w = Math.max(MIN, Math.abs(px - ax));
    w = Math.min(w, left ? ax : W - ax, H * ratio);
    h = w / ratio;
    x0 = left ? ax - w : ax;
    y0 = Math.min(Math.max(cy - h / 2, 0), H - h);
  } else {
    const ay = top ? y1 : y0;
    h = Math.max(MIN, Math.abs(py - ay));
    h = Math.min(h, top ? ay : H - ay, W / ratio);
    w = h * ratio;
    y0 = top ? ay - h : ay;
    x0 = Math.min(Math.max(cx - w / 2, 0), W - w);
  }
  return [x0 / W, y0 / H, (x0 + w) / W, (y0 + h) / H];
}

/**
 * Wire up pointer handling on the overlay element. `get()` returns the current state
 * ({ crop, ratio, W, H }), `set(crop, final)` receives updates.
 */
export function attachCropOverlay(overlay, box, get, set) {
  let drag = null;

  const toImage = (e) => {
    const r = overlay.getBoundingClientRect();
    const { W, H } = get();
    return [((e.clientX - r.left) / r.width) * W, ((e.clientY - r.top) / r.height) * H];
  };

  box.addEventListener('pointerdown', (e) => {
    e.preventDefault();
    const { crop } = get();
    drag = { handle: e.target.dataset.h || 'move', start: crop, grab: toImage(e) };
    box.setPointerCapture(e.pointerId);
  });
  box.addEventListener('pointermove', (e) => {
    if (!drag) return;
    const { ratio, W, H } = get();
    set(dragCrop(drag.start, drag.handle, drag.grab, toImage(e), ratio, W, H), false);
  });
  const end = () => {
    if (!drag) return;
    drag = null;
    set(get().crop, true);
  };
  box.addEventListener('pointerup', end);
  box.addEventListener('pointercancel', end);

  return {
    /** Position the box for a crop (normalised coordinates of the overlay). */
    update(crop) {
      Object.assign(box.style, {
        left: `${crop[0] * 100}%`,
        top: `${crop[1] * 100}%`,
        width: `${(crop[2] - crop[0]) * 100}%`,
        height: `${(crop[3] - crop[1]) * 100}%`,
      });
    },
  };
}
