//! Automatic settings derived from image statistics.
//!
//! All of these run on the small on-screen proxy (a few megapixels at most), inside a
//! rectangle in *source* normalised coordinates – normally the crop, so the film rebate and
//! scanner holder don't skew the result.

use super::tone::{channel_gains, density, linearize, pre_balance, scene_linear};
use super::Params;
use crate::geometry::{NormRect, PixelRect};

/// Iterate over the pixels of an interleaved buffer (`ch` ≥ 3 channels) inside `rect`.
/// Large areas are subsampled to at most ~250k pixels — plenty for statistics.
fn for_each_pixel(src: &[u16], w: usize, h: usize, ch: usize, rect: NormRect, mut f: impl FnMut(u16, u16, u16)) {
    let r = PixelRect::from_norm(rect, w, h);
    let step = ((r.width() * r.height()) as f64 / 250_000.0).sqrt().ceil().max(1.0) as usize;
    for y in (r.y0..r.y1).step_by(step) {
        for x in (r.x0..r.x1).step_by(step) {
            let i = (y * w + x) * ch;
            f(src[i], src[i + 1], src[i + 2]);
        }
    }
}

/// A fixed-range histogram that can answer percentile queries.
struct Histogram {
    lo: f32,
    hi: f32,
    bins: Vec<u32>,
    count: u64,
}

impl Histogram {
    fn new(lo: f32, hi: f32, n: usize) -> Self {
        Histogram { lo, hi, bins: vec![0; n], count: 0 }
    }
    fn add(&mut self, v: f32) {
        let n = self.bins.len();
        let i = ((v - self.lo) / (self.hi - self.lo) * n as f32).clamp(0.0, (n - 1) as f32) as usize;
        self.bins[i] += 1;
        self.count += 1;
    }
    fn percentile(&self, p: f32) -> f32 {
        let target = (self.count as f64 * p as f64 / 100.0) as u64;
        let mut acc = 0u64;
        for (i, &b) in self.bins.iter().enumerate() {
            acc += b as u64;
            if acc > target {
                return self.lo + (i as f32 + 0.5) / self.bins.len() as f32 * (self.hi - self.lo);
            }
        }
        self.hi
    }
}

#[inline]
fn code(v: u16) -> f32 {
    v as f32 / 65535.0
}

/// Auto-detect the negative's per-channel density range: `[dmin r,g,b, dmax r,g,b]`.
///
/// We take a low and a high percentile of each channel's density rather than the absolute
/// min/max so a few specks of dust or a scratch don't decide the result. Normalising each
/// channel independently between its own percentiles neutralises both the orange mask (it
/// shifts `dmin`) and the different dye-layer contrasts (they change `dmax − dmin`).
pub fn auto_negative(p: &Params, src: &[u16], w: usize, h: usize, ch: usize, rect: NormRect) -> [f32; 6] {
    let mut hist: Vec<Histogram> = (0..3).map(|_| Histogram::new(0.0, 4.0, 4000)).collect();
    for_each_pixel(src, w, h, ch, rect, |r, g, b| {
        for (c, v) in [r, g, b].into_iter().enumerate() {
            hist[c].add(density(linearize(p, code(v))));
        }
    });
    let mut out = [0f32; 6];
    for c in 0..3 {
        out[c] = hist[c].percentile(0.2);
        out[c + 3] = hist[c].percentile(99.8).max(out[c] + 0.05);
    }
    out
}

/// Film base pick: the mean density of the sampled area becomes each channel's `dmin`.
/// Averaging happens in linear light (transmittance), then converts to density.
pub fn sample_density(p: &Params, src: &[u16], w: usize, h: usize, ch: usize, rect: NormRect) -> [f32; 3] {
    let mut sum = [0f64; 3];
    let mut n = 0f64;
    for_each_pixel(src, w, h, ch, rect, |r, g, b| {
        for (c, v) in [r, g, b].into_iter().enumerate() {
            sum[c] += linearize(p, code(v)) as f64;
        }
        n += 1.0;
    });
    sum.map(|s| density((s / n.max(1.0)) as f32))
}

/// Grey picker: white-balance multipliers that make the sampled area neutral.
pub fn neutral_balance(p: &Params, src: &[u16], w: usize, h: usize, ch: usize, rect: NormRect) -> [f32; 3] {
    let mut sum = [0f64; 3];
    for_each_pixel(src, w, h, ch, rect, |r, g, b| {
        for (c, v) in [r, g, b].into_iter().enumerate() {
            sum[c] += pre_balance(p, c, code(v)).max(0.0) as f64;
        }
    });
    if sum.iter().any(|&s| s <= 0.0) {
        return [1.0; 3];
    }
    [(sum[1] / sum[0]) as f32, 1.0, (sum[1] / sum[2]) as f32]
}

/// Auto levels: black and white points (perceptual units) from luminance percentiles.
pub fn auto_levels(p: &Params, src: &[u16], w: usize, h: usize, ch: usize, rect: NormRect) -> [f32; 2] {
    let gains = channel_gains(p);
    let mut hist = Histogram::new(0.0, 1.5, 3000);
    for_each_pixel(src, w, h, ch, rect, |r, g, b| {
        let lin = [r, g, b].into_iter().enumerate().map(|(c, v)| scene_linear(p, &gains, c, code(v)).max(0.0));
        let y: f32 = lin.zip(crate::color::LUMA).map(|(v, w)| v * w).sum();
        hist.add(y.powf(1.0 / 2.2));
    });
    let black = hist.percentile(0.1).min(0.5);
    let white = hist.percentile(99.9).max(black + 0.1);
    [black, white]
}
