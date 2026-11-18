//! Non-local means (Buades, Coll & Morel, 2005).
//!
//! Plain blurring averages a pixel with its *spatial* neighbours, smearing edges. Non-local
//! means instead averages it with every pixel in a search window whose *surrounding patch*
//! looks similar — on an edge, similar patches lie along the edge, so the edge survives:
//!
//! ```text
//! out(p) = Σ_q w(p,q) · in(q) / Σ_q w(p,q)
//! w(p,q) = exp( −max(d²(p,q) − 2σ², 0) / h² )
//! d²(p,q) = mean squared difference between the patches around p and q
//! ```
//!
//! Subtracting 2σ² accounts for the noise both patches carry (two noisy copies of the same
//! patch differ by 2σ² on average), so identical-but-noisy patches get full weight.
//!
//! ## The fast formulation
//!
//! Computing every patch distance directly costs search² × patch² operations per pixel.
//! Instead we loop over *offsets* (dx, dy): for one offset, the squared difference image
//! `(in(x) − in(x + offset))²` is computed once, and each pixel's patch distance is just a box
//! sum over it, which is separable. That removes the patch² factor (Darbon et al., 2008).

/// Search window radius (11×11 window) and patch radius (5×5 patches).
const SEARCH: isize = 5;
const PATCH: isize = 2;
/// Filtering parameter h in units of σ (tuned with `examples/denoise_eval.rs`).
const H: f32 = 0.85;
/// How much h grows per pixel of grain size. Clumped grain makes two noisy copies of the same
/// patch differ *more erratically* (fewer independent samples per patch), so similar patches
/// need a more tolerant weight to still be averaged. Tuned with `denoise_eval`.
///
/// Even tuned, non-local means is the weakest method on coarse film grain: grain clumps are
/// almost as big as its 5×5 patches, so "similar patches" are as often similar grain as
/// similar image. It shines on fine digital noise; for film, prefer wavelet or neural.
const H_PER_GRAIN: f32 = 0.35;

pub const MARGIN: usize = (SEARCH + PATCH + 1) as usize;

#[inline]
fn clampi(i: isize, n: usize) -> usize {
    i.clamp(0, n as isize - 1) as usize
}

/// `exp(-x)` for x ∈ [0, 8] via a table — this runs ~120 times per pixel.
struct ExpTable(Vec<f32>);
impl ExpTable {
    const RANGE: f32 = 8.0;
    fn new() -> Self {
        ExpTable((0..=4096).map(|i| (-(i as f32) / 4096.0 * Self::RANGE).exp()).collect())
    }
    #[inline]
    fn get(&self, x: f32) -> f32 {
        if x >= Self::RANGE {
            0.0
        } else {
            self.0[(x * (4096.0 / Self::RANGE)) as usize]
        }
    }
}

/// Denoise one plane in place. `sigma` is the per-pixel noise σ, `grain` the grain size (px).
pub fn denoise(plane: &mut [f32], w: usize, h: usize, sigma: f32, grain: f32) {
    if sigma <= 0.0 || w < 2 || h < 2 {
        return;
    }
    let n = w * h;
    let exp = ExpTable::new();
    let inv_h2 = 1.0 / (H * (1.0 + H_PER_GRAIN * grain) * sigma).powi(2);
    let two_s2 = 2.0 * sigma * sigma;
    let patch_n = ((2 * PATCH + 1) * (2 * PATCH + 1)) as f32;

    let src = plane.to_vec();
    let mut acc = vec![0f32; n];
    let mut wsum = vec![0f32; n];
    let mut wmax = vec![0f32; n];

    // Pad the image by repeating its edges, so the hot loops below index directly with no
    // bounds checks. Pixel (x, y) of the image is at (x + P, y + P) in `pad`.
    const P: usize = (SEARCH + PATCH) as usize;
    let pw = w + 2 * P;
    let pad: Vec<f32> = (0..(h + 2 * P) * pw)
        .map(|i| {
            let (x, y) = ((i % pw) as isize - P as isize, (i / pw) as isize - P as isize);
            src[clampi(y, h) * w + clampi(x, w)]
        })
        .collect();

    // The squared differences are needed PATCH pixels beyond the image on every side.
    const R: usize = PATCH as usize;
    let (ew, eh) = (w + 2 * R, h + 2 * R);
    let mut diff = vec![0f32; ew * eh];
    let mut hsum = vec![0f32; w * eh];

    for dy in -SEARCH..=SEARCH {
        for dx in -SEARCH..=SEARCH {
            if dx == 0 && dy == 0 {
                continue;
            }
            // `pad` index of image pixel (x, y), and of its partner shifted by (dx, dy).
            let at = |x: usize, y: usize| (y + P) * pw + x + P;
            let shift = dy * pw as isize + dx;

            // 1. Squared difference between the image and its shifted copy (extended region).
            for ey in 0..eh {
                let row = (ey + P - R) * pw + P - R;
                for ex in 0..ew {
                    let i = row + ex;
                    let d = pad[i] - pad[(i as isize + shift) as usize];
                    diff[ey * ew + ex] = d * d;
                }
            }
            // 2. Horizontal box sums over the patch width.
            for ey in 0..eh {
                let drow = &diff[ey * ew..(ey + 1) * ew];
                let hrow = &mut hsum[ey * w..(ey + 1) * w];
                for (x, out) in hrow.iter_mut().enumerate() {
                    *out = drow[x..x + 2 * R + 1].iter().sum();
                }
            }
            // 3. Vertical box sums → patch distance → weight; accumulate.
            for y in 0..h {
                for x in 0..w {
                    let mut s = 0.0;
                    for k in 0..=2 * R {
                        s += hsum[(y + k) * w + x];
                    }
                    let d2 = s / patch_n;
                    let weight = exp.get((d2 - two_s2).max(0.0) * inv_h2);
                    let i = y * w + x;
                    acc[i] += weight * pad[(at(x, y) as isize + shift) as usize];
                    wsum[i] += weight;
                    wmax[i] = wmax[i].max(weight);
                }
            }
        }
    }
    // The pixel itself gets the largest weight any neighbour received (standard practice: its
    // own patch distance is 0, which would otherwise dominate everything).
    for i in 0..n {
        let wc = if wmax[i] > 0.0 { wmax[i] } else { 1.0 };
        plane[i] = (acc[i] + wc * src[i]) / (wsum[i] + wc);
    }
}
