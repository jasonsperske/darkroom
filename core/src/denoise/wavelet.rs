//! À trous ("with holes") wavelet shrinkage.
//!
//! The image is split into *detail layers* at doubling scales plus a smooth residual:
//!
//! ```text
//! c0 = image
//! c1 = blur(c0, scale 1)    w1 = c0 − c1     finest detail (and most of the noise)
//! c2 = blur(c1, scale 2)    w2 = c1 − c2
//! …                         …
//! image = w1 + w2 + … + wN + cN              (exact, by construction)
//! ```
//!
//! The blur is the B3-spline kernel `[1 4 6 4 1] / 16`, applied at scale `s` by spacing its
//! taps `s` pixels apart (the "holes"). Real detail produces large wavelet coefficients;
//! noise produces many small ones. So each coefficient is *shrunk* toward zero depending on
//! how it compares with the noise expected in that layer, and the layers are summed back.
//!
//! The threshold for layer `l` is `K · σ_l`, where `σ_l` is how much noise that layer is
//! expected to contain. For white noise most of it sits in layer 0; for film grain, whose
//! clumps span several pixels, much more sits in layers 1–2. The grain model (`grain.rs`)
//! supplies the right `σ_l` for the measured grain size, which is what makes this method
//! effective on film.

pub const LEVELS: usize = 5;
/// Threshold in units of each layer's noise σ, tuned with `examples/denoise_eval.rs` on film
/// grain (1.5). On purely digital noise ~1.9 scores slightly higher.
const K: f32 = 1.5;

/// Context needed around a tile: the kernel reaches 2·(1+2+4+8+16) = 62 pixels.
pub const MARGIN: usize = 64;

/// Mirror an out-of-range coordinate back into `[0, n)`.
#[inline]
fn mirror(i: isize, n: usize) -> usize {
    let n = n as isize;
    let mut i = i;
    if n == 1 {
        return 0;
    }
    while i < 0 || i >= n {
        i = if i < 0 { -i } else { 2 * (n - 1) - i };
    }
    i as usize
}

/// Separable B3 blur with taps `step` pixels apart.
fn blur(src: &[f32], dst: &mut [f32], tmp: &mut [f32], w: usize, h: usize, step: usize) {
    const K: [f32; 5] = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];
    let s = step as isize;
    for y in 0..h {
        let row = &src[y * w..(y + 1) * w];
        for x in 0..w {
            let xi = x as isize;
            let mut acc = 0.0;
            for (k, &kv) in K.iter().enumerate() {
                acc += kv * row[mirror(xi + (k as isize - 2) * s, w)];
            }
            tmp[y * w + x] = acc;
        }
    }
    for y in 0..h {
        let yi = y as isize;
        let rows: [usize; 5] = std::array::from_fn(|k| mirror(yi + (k as isize - 2) * s, h) * w);
        for x in 0..w {
            let mut acc = 0.0;
            for (k, &kv) in K.iter().enumerate() {
                acc += kv * tmp[rows[k] + x];
            }
            dst[y * w + x] = acc;
        }
    }
}

/// Non-negative garrote shrinkage: keeps big coefficients almost unchanged (unlike soft
/// thresholding, which subtracts T from everything) and smoothly kills small ones.
#[inline]
fn shrink(d: f32, t: f32) -> f32 {
    let d2 = d * d;
    if d2 <= t * t {
        0.0
    } else {
        d * (1.0 - t * t / d2)
    }
}

/// Split a plane into `levels` detail layers and the remaining smooth residual.
/// `plane == Σ details + residual` exactly.
pub fn decompose_levels(plane: &[f32], w: usize, h: usize, levels: usize) -> (Vec<Vec<f32>>, Vec<f32>) {
    let n = w * h;
    let mut current = plane.to_vec();
    let mut smooth = vec![0f32; n];
    let mut tmp = vec![0f32; n];
    let mut details = Vec::with_capacity(levels);
    for level in 0..levels {
        blur(&current, &mut smooth, &mut tmp, w, h, 1 << level);
        details.push(current.iter().zip(&smooth).map(|(c, s)| c - s).collect());
        std::mem::swap(&mut current, &mut smooth);
    }
    (details, current)
}

pub fn decompose(plane: &[f32], w: usize, h: usize) -> (Vec<Vec<f32>>, Vec<f32>) {
    decompose_levels(plane, w, h, LEVELS)
}

/// Denoise one plane in place, given the noise expected in each layer.
pub fn denoise(plane: &mut [f32], w: usize, h: usize, level_sigma: [f32; LEVELS]) {
    if level_sigma.iter().all(|&s| s <= 0.0) || w < 2 || h < 2 {
        return;
    }
    let n = w * h;
    let mut current = plane.to_vec();
    let mut smooth = vec![0f32; n];
    let mut tmp = vec![0f32; n];
    let mut result = vec![0f32; n];
    for (level, &sigma) in level_sigma.iter().enumerate() {
        blur(&current, &mut smooth, &mut tmp, w, h, 1 << level);
        let t = K * sigma;
        for i in 0..n {
            result[i] += shrink(current[i] - smooth[i], t);
        }
        std::mem::swap(&mut current, &mut smooth);
    }
    for i in 0..n {
        plane[i] = result[i] + current[i]; // + coarsest residual
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_noise_is_identity() {
        let (w, h) = (37, 23);
        let mut p: Vec<f32> = (0..w * h).map(|i| ((i * 7919) % 101) as f32 / 100.0).collect();
        let orig = p.clone();
        denoise(&mut p, w, h, [0.0; LEVELS]);
        assert_eq!(p, orig);
        // A tiny σ must leave strong detail essentially untouched (decomposition is exact).
        denoise(&mut p, w, h, [1e-6; LEVELS]);
        assert!(p.iter().zip(&orig).all(|(a, b)| (a - b).abs() < 1e-4));
    }
}
