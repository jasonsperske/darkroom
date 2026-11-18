//! A two-parameter model of film grain (and of ordinary sensor noise as a special case).
//!
//! ## Why grain needs its own model
//!
//! Digital sensor noise is roughly *white*: every pixel's noise is independent of its
//! neighbours'. Film grain is not. A silver grain or dye cloud is often several scanner pixels
//! across, so the noise comes in clumps. High-speed film has bigger clumps, and underexposure
//! makes them more visible. White-noise denoisers either miss clumped grain (it looks like
//! texture to them) or have to be turned up until they smear real detail.
//!
//! ## The model
//!
//! We model grain as white Gaussian noise blurred by a Gaussian of radius `size` (pixels), and
//! scaled to a per-pixel standard deviation `σ`:
//!
//! ```text
//! grain = σ · normalise( gaussian_blur( white_noise, size ) )
//! ```
//!
//! `size = 0` is white (digital) noise; 1–3 px is typical of film scanned at 2000–4000 dpi.
//! This is not a physical model of grains (that would be a Boolean model of random disks), but
//! it captures what matters to a denoiser: how much noise there is *at each scale*.
//!
//! ## Measuring σ and size from a photo
//!
//! We split the image into à trous wavelet layers (`wavelet.rs`). Blurred noise moves energy
//! from the finest layer to the next one, so the ratio `σ(layer 1) / σ(layer 0)` tells us the
//! grain size. A small table, built once by simulating grain of known sizes, converts measured
//! ratios back into a size and σ. The measurements only use the *flattest* parts of the image,
//! such as skies, walls and out-of-focus areas, so real detail doesn't count as grain.

use super::wavelet;
use std::sync::OnceLock;

/// What the denoisers need to know about the noise in an image.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct NoiseProfile {
    /// Per-pixel standard deviation of the grain in the luminance and chrominance planes.
    pub luma: f32,
    pub chroma: f32,
    /// Grain size: radius (px) of the Gaussian that shapes the grain. 0 = white noise.
    pub size: f32,
    /// Noise measured directly in each wavelet layer, `(luma, chroma)`, when available.
    ///
    /// Real scans hold a *mixture*: fine scanner noise plus film grain, and the photo's own
    /// texture. A single (σ, size) summary can't describe a mixture exactly, so for the layers
    /// where flat areas contain little besides noise (0–2) we keep the direct measurement.
    /// The coarser layers come from the model.
    pub measured: Option<([f32; wavelet::LEVELS], [f32; wavelet::LEVELS])>,
}

/// Wavelet layers whose noise is measured directly (the rest come from the model).
pub const MEASURED_LEVELS: usize = 3;

impl NoiseProfile {
    pub fn new(luma: f32, chroma: f32, size: f32) -> Self {
        NoiseProfile { luma, chroma, size, measured: None }
    }

    /// Scale by the strength sliders.
    pub fn scaled(&self, luma: f32, chroma: f32) -> Self {
        NoiseProfile {
            luma: self.luma * luma,
            chroma: self.chroma * chroma,
            size: self.size,
            measured: self.measured.map(|(l, c)| (l.map(|v| v * luma), c.map(|v| v * chroma))),
        }
    }

    /// Expected noise per wavelet layer for luma and chroma.
    pub fn levels(&self) -> ([f32; wavelet::LEVELS], [f32; wavelet::LEVELS]) {
        let model = (level_sigmas(self.luma, self.size), level_sigmas(self.chroma, self.size));
        match self.measured {
            None => model,
            Some((ml, mc)) => {
                let pick = |m: [f32; wavelet::LEVELS], model: [f32; wavelet::LEVELS]| std::array::from_fn(|l| if l < MEASURED_LEVELS { m[l] } else { model[l] });
                (pick(ml, model.0), pick(mc, model.1))
            }
        }
    }
}

// ---- Small deterministic random numbers (also used by the trainer) -----------------------

pub struct Rng(pub u64);

impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        // SplitMix64
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    /// Standard normal (Box–Muller).
    pub fn gauss(&mut self) -> f32 {
        let u1 = self.uniform().max(1e-7);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

// ---- Synthesis ---------------------------------------------------------------------------

/// Separable Gaussian blur with standard deviation `radius` (mirrored edges).
pub fn gaussian_blur(plane: &mut [f32], w: usize, h: usize, radius: f32) {
    if radius < 0.05 {
        return;
    }
    let reach = (radius * 3.0).ceil() as isize;
    let kernel: Vec<f32> = (-reach..=reach).map(|i| (-(i * i) as f32 / (2.0 * radius * radius)).exp()).collect();
    let sum: f32 = kernel.iter().sum();
    let kernel: Vec<f32> = kernel.iter().map(|k| k / sum).collect();
    let mirror = |i: isize, n: usize| -> usize {
        let n = n as isize;
        let i = if i < 0 { -i - 1 } else if i >= n { 2 * n - i - 1 } else { i };
        i.clamp(0, n - 1) as usize
    };
    let mut tmp = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            tmp[y * w + x] = kernel.iter().enumerate().map(|(k, kv)| kv * plane[y * w + mirror(x as isize + k as isize - reach, w)]).sum();
        }
    }
    for y in 0..h {
        for x in 0..w {
            plane[y * w + x] = kernel.iter().enumerate().map(|(k, kv)| kv * tmp[mirror(y as isize + k as isize - reach, h) * w + x]).sum();
        }
    }
}

/// A field of unit-standard-deviation grain of the given size.
pub fn grain_field(w: usize, h: usize, size: f32, rng: &mut Rng) -> Vec<f32> {
    let mut f: Vec<f32> = (0..w * h).map(|_| rng.gauss()).collect();
    gaussian_blur(&mut f, w, h, size);
    let sd = (f.iter().map(|v| v * v).sum::<f32>() / f.len() as f32).sqrt().max(1e-9);
    f.iter_mut().for_each(|v| *v /= sd);
    f
}

// ---- The calibration table ---------------------------------------------------------------

/// Grain sizes the table is computed for.
const SIZES: [f32; 13] = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0];
pub const MAX_SIZE: f32 = 4.0;

/// `table()[i][l]` = standard deviation of wavelet layer `l` for unit grain of size `SIZES[i]`.
/// Computed once by simulation (a few milliseconds).
fn table() -> &'static [[f32; wavelet::LEVELS]; SIZES.len()] {
    static TABLE: OnceLock<[[f32; wavelet::LEVELS]; SIZES.len()]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let n = 192;
        let mut rng = Rng(42);
        let mut t = [[0f32; wavelet::LEVELS]; SIZES.len()];
        for (i, &size) in SIZES.iter().enumerate() {
            let field = grain_field(n, n, size, &mut rng);
            let (details, _) = wavelet::decompose(&field, n, n);
            for (l, d) in details.iter().enumerate() {
                t[i][l] = (d.iter().map(|v| v * v).sum::<f32>() / d.len() as f32).sqrt();
            }
        }
        t
    })
}

/// Expected noise in each wavelet layer for grain of `sigma` and `size`.
pub fn level_sigmas(sigma: f32, size: f32) -> [f32; wavelet::LEVELS] {
    let t = table();
    let s = size.clamp(0.0, MAX_SIZE);
    let i = SIZES.iter().rposition(|&v| v <= s).unwrap_or(0).min(SIZES.len() - 2);
    let a = ((s - SIZES[i]) / (SIZES[i + 1] - SIZES[i])).clamp(0.0, 1.0);
    std::array::from_fn(|l| sigma * (t[i][l] + a * (t[i + 1][l] - t[i][l])))
}

/// Grain size whose layer-1 / layer-0 noise ratio matches `ratio`.
fn size_from_ratio(ratio: f32) -> f32 {
    let t = table();
    let r = |i: usize| t[i][1] / t[i][0];
    if ratio <= r(0) {
        return 0.0;
    }
    for i in 0..SIZES.len() - 1 {
        if ratio <= r(i + 1) {
            return SIZES[i] + (ratio - r(i)) / (r(i + 1) - r(i)) * (SIZES[i + 1] - SIZES[i]);
        }
    }
    MAX_SIZE
}

// ---- Estimation --------------------------------------------------------------------------

fn mad_sigma(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mid = values.len() / 2;
    let (_, m, _) = values.select_nth_unstable_by(mid, |a, b| a.abs().total_cmp(&b.abs()));
    m.abs() / 0.6745
}

/// Measure the grain in working-space planes `[Y, C1, C2]`.
pub fn estimate(planes: &[Vec<f32>; 3], w: usize, h: usize) -> NoiseProfile {
    const BLOCK: usize = 16;
    if w < 4 * BLOCK || h < 4 * BLOCK {
        return NoiseProfile::default();
    }
    let (luma, _) = wavelet::decompose(&planes[0], w, h);
    // How busy is each block? Mean |coefficient| of layer 4 (structures ~16–32 px across).
    // Even coarse grain has almost no energy there, while real detail does, so this picks
    // flat areas without being biased by the grain we're trying to measure.
    let (bw, bh) = (w / BLOCK, h / BLOCK);
    // Blocks near pure black or white are skipped: there the grain is clipped, so it looks
    // smaller and weaker than it is. (Y runs 0…√3 in the working space.)
    let mut busy: Vec<(f32, usize)> = (0..bw * bh)
        .filter_map(|b| {
            let (bx, by) = (b % bw * BLOCK, b / bw * BLOCK);
            let (mut s, mut mean) = (0.0, 0.0);
            for y in by..by + BLOCK {
                for x in bx..bx + BLOCK {
                    s += luma[4][y * w + x].abs();
                    mean += planes[0][y * w + x];
                }
            }
            let mean = mean / (BLOCK * BLOCK) as f32 / 3f32.sqrt();
            (0.12..0.88).contains(&mean).then_some((s, b))
        })
        .collect();
    if busy.is_empty() {
        return NoiseProfile::default();
    }
    busy.sort_by(|a, b| a.0.total_cmp(&b.0));
    // Keep the flattest quarter.
    let keep = &busy[..(busy.len() / 4).max(1)];

    let gather = |layer: &[f32]| -> Vec<f32> {
        let mut v = Vec::with_capacity(keep.len() * BLOCK * BLOCK);
        for &(_, b) in keep {
            let (bx, by) = (b % bw * BLOCK, b / bw * BLOCK);
            for y in by..by + BLOCK {
                v.extend_from_slice(&layer[y * w + bx..y * w + bx + BLOCK]);
            }
        }
        v
    };
    let mut luma_levels = [0f32; wavelet::LEVELS];
    for l in 0..MEASURED_LEVELS {
        luma_levels[l] = mad_sigma(&mut gather(&luma[l]));
    }
    let mut chroma_levels = [0f32; wavelet::LEVELS];
    for c in 1..3 {
        let (d, _) = wavelet::decompose_levels(&planes[c], w, h, MEASURED_LEVELS);
        for l in 0..MEASURED_LEVELS {
            chroma_levels[l] += mad_sigma(&mut gather(&d[l])) / 2.0;
        }
    }
    let (s0, s1) = (luma_levels[0], luma_levels[1]);
    let size = if s0 > 0.0 { size_from_ratio(s1 / s0) } else { 0.0 };
    let gain0 = level_sigmas(1.0, size)[0];
    NoiseProfile { luma: s0 / gain0, chroma: chroma_levels[0] / gain0, size, measured: Some((luma_levels, chroma_levels)) }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A smooth image plus synthetic grain of known σ and size must be measured back.
    #[test]
    fn estimates_synthetic_grain() {
        let (w, h) = (384, 256);
        for (sigma, size) in [(0.03f32, 0.0f32), (0.05, 1.0), (0.04, 2.0)] {
            let mut rng = Rng(7);
            let grain = grain_field(w, h, size, &mut rng);
            let y: Vec<f32> = (0..w * h).map(|i| 0.4 + 0.3 * ((i % w) as f32 / w as f32) + sigma * grain[i]).collect();
            let flat = vec![0f32; w * h];
            let est = estimate(&[y, flat.clone(), flat], w, h);
            assert!((est.luma - sigma).abs() < sigma * 0.15, "σ {sigma} size {size}: {est:?}");
            assert!((est.size - size).abs() < 0.35, "σ {sigma} size {size}: {est:?}");
        }
    }
}
