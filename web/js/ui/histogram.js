// Histogram: 256 bins per channel (R, G, B, luminance) computed in Rust from the developed
// preview. Counts are drawn on a square-root scale so small populations stay visible.

export function drawHistogram(canvas, bins) {
  const ctx = canvas.getContext('2d');
  const { width: w, height: h } = canvas;
  ctx.clearRect(0, 0, w, h);
  if (!bins) return;

  // Ignore the extreme bins when scaling, so clipped pixels don't flatten everything else.
  let max = 1;
  for (let c = 0; c < 4; c++) for (let i = 1; i < 255; i++) max = Math.max(max, bins[c * 256 + i]);
  const y = (v) => h - Math.min(1, Math.sqrt(v / max)) * (h - 2);

  const path = (c) => {
    ctx.beginPath();
    ctx.moveTo(0, h);
    for (let i = 0; i < 256; i++) ctx.lineTo((i / 255) * w, y(bins[c * 256 + i]));
    ctx.lineTo(w, h);
    ctx.closePath();
  };

  ctx.fillStyle = '#ffffff22';
  path(3);
  ctx.fill();
  ctx.globalCompositeOperation = 'lighter';
  ['#c0392b88', '#27ae6088', '#2f6fd088'].forEach((color, c) => {
    ctx.fillStyle = color;
    path(c);
    ctx.fill();
  });
  ctx.globalCompositeOperation = 'source-over';

  // Clipping indicators: a bright tick if more than 0.5 % of pixels sit in the end bins.
  const total = bins.slice(768, 1024).reduce((a, b) => a + b, 0) || 1;
  ctx.fillStyle = '#eee';
  if (bins[768] / total > 0.005) ctx.fillRect(0, 0, 2, h);
  if (bins[1023] / total > 0.005) ctx.fillRect(w - 2, 0, 2, h);
}
