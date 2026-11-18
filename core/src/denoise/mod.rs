//! Noise reduction.
//!
//! ## Where denoising sits
//!
//! Every other develop step is per-pixel (a lookup table). Denoising is *spatial*: a pixel's
//! new value depends on its neighbours. So it runs as a separate pass on the 16-bit master
//! **before** developing: `master → denoise → tone LUT → output`.
//!
//! ## The working space
//!
//! 1. **An encoding where the noise is even.** We want a single noise level to describe the
//!    whole image, so we choose the encoding in which the noise is most uniform:
//!    * *Positives and digital files: gamma.* Sensor noise is mostly shot noise, whose standard
//!      deviation grows with the square root of the signal. A gamma-like encoding (sRGB, or
//!      x^(1/2.2) for linear data) roughly equalises it, like a cheap Anscombe transform.
//!    * *Negatives: optical density* (`−log10 transmittance`, see `develop/tone.rs`). Grain
//!      is a property of the developed film, and its granularity is far more uniform in
//!      density than in scanner code values. Denoising **before** inversion, in density,
//!      matters most for underexposed and high-speed film. Inversion stretches the thin parts
//!      of the negative, the shadows of the print, and with them the grain.
//! 2. **Luminance / chrominance.** We rotate RGB with an *orthonormal* opponent transform:
//!
//!    ```text
//!    Y  = (R + G + B) / √3      brightness
//!    C1 = (R − B) / √2          blue ↔ red
//!    C2 = (R − 2G + B) / √6     green ↔ magenta
//!    ```
//!
//!    Orthonormal means independent noise of σ in R, G, B stays independent noise of σ in
//!    Y, C1, C2, so noise levels carry over exactly. Splitting brightness from colour lets us
//!    treat them differently: colour noise ("blotches") can be removed aggressively because
//!    our eyes see colour at low resolution, while luminance detail must be preserved.
//!
//! ## Methods
//!
//! * `wavelet.rs` – à trous wavelet shrinkage: fast, excellent for colour noise.
//! * `nlm.rs`     – non-local means: averages pixels whose *neighbourhoods* look alike.
//! * `neural.rs`  – a small convolutional network (FFDNet-style) trained on photographs.
//!
//! All of them are driven by a [`grain::NoiseProfile`]. It holds the amount of luminance and
//! colour noise *and the grain size*, measured automatically and scaled by the user's sliders.
//! Grain size is what lets the same code handle both digital noise (size 0) and the clumped
//! grain of fast film (sizes 1–3 px).

pub mod grain;
pub mod neural;
pub mod nlm;
pub mod wavelet;

use crate::color::{linear_to_srgb, srgb_to_linear};
pub use grain::NoiseProfile;
use serde::Deserialize;

#[derive(Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum Method {
    #[default]
    Off,
    Wavelet,
    Nlm,
    Neural,
}

/// User settings (JSON from JavaScript, `denoise` in `web/js/state.js`).
#[derive(Deserialize, Clone, Debug)]
#[serde(default, rename_all = "camelCase")]
pub struct DenoiseParams {
    pub method: Method,
    /// Strength multipliers applied to the measured noise levels (1 = as measured).
    pub luma: f32,
    pub chroma: f32,
    /// Work in density space (set by the engine when the image is a negative).
    pub negative: bool,
}

impl Default for DenoiseParams {
    fn default() -> Self {
        DenoiseParams { method: Method::Off, luma: 1.0, chroma: 1.0, negative: false }
    }
}

impl DenoiseParams {
    pub fn from_json(json: &str) -> Result<Self, String> {
        serde_json::from_str(json).map_err(|e| format!("bad denoise params: {e}"))
    }
}

/// The per-channel encoding the planes are in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Domain {
    /// sRGB-like gamma encoding.
    Gamma,
    /// Optical density / 4 (so densities 0…4 map to 0…1).
    Density,
}

impl Domain {
    pub fn for_negative(negative: bool) -> Self {
        if negative {
            Domain::Density
        } else {
            Domain::Gamma
        }
    }
}

const GAMMA: f32 = 2.2;
const DENSITY_SCALE: f32 = 4.0;
const MIN_TRANSMITTANCE: f32 = 1e-4; // = density 4

/// Code value (0…1) → working-space value.
fn encode(code: f32, linear: bool, domain: Domain) -> f32 {
    match domain {
        Domain::Gamma if linear => code.powf(1.0 / GAMMA),
        Domain::Gamma => code,
        Domain::Density => {
            let t = if linear { code } else { srgb_to_linear(code) };
            -t.max(MIN_TRANSMITTANCE).log10() / DENSITY_SCALE
        }
    }
}

/// Working-space value → code value (0…1). Inverse of [`encode`].
fn decode(v: f32, linear: bool, domain: Domain) -> f32 {
    match domain {
        Domain::Gamma => {
            let v = v.clamp(0.0, 1.0);
            if linear {
                v.powf(GAMMA)
            } else {
                v
            }
        }
        Domain::Density => {
            let t = 10f32.powf(-v.max(0.0) * DENSITY_SCALE).min(1.0);
            if linear {
                t
            } else {
                linear_to_srgb(t)
            }
        }
    }
}

/// An image split into three float planes in the working space described above.
pub struct Planes {
    pub w: usize,
    pub h: usize,
    /// `[Y, C1, C2]`, each `w * h`.
    pub p: [Vec<f32>; 3],
}

const S3: f32 = 0.577_350_26; // 1/√3
const S2: f32 = 0.707_106_77; // 1/√2
const S6: f32 = 0.408_248_3; // 1/√6

impl Planes {
    /// From interleaved RGB16. `linear`: are the code values linear light?
    pub fn from_rgb16(rgb: &[u16], w: usize, h: usize, linear: bool, domain: Domain) -> Self {
        let n = w * h;
        let mut p = [vec![0f32; n], vec![0f32; n], vec![0f32; n]];
        // One table for the encoding, rather than a pow() or log() per sample.
        let table: Vec<f32> = (0..65536).map(|i| encode(i as f32 / 65535.0, linear, domain)).collect();
        for (i, px) in rgb.chunks_exact(3).enumerate() {
            let (r, g, b) = (table[px[0] as usize], table[px[1] as usize], table[px[2] as usize]);
            p[0][i] = (r + g + b) * S3;
            p[1][i] = (r - b) * S2;
            p[2][i] = (r - 2.0 * g + b) * S6;
        }
        Planes { w, h, p }
    }

    /// Back to interleaved RGB16 (the exact inverse of [`Planes::from_rgb16`]: the transform
    /// matrix is orthonormal, so its inverse is its transpose).
    pub fn to_rgb16(&self, linear: bool, domain: Domain) -> Vec<u16> {
        let mut out = vec![0u16; self.w * self.h * 3];
        let to_code = |v: f32| (decode(v, linear, domain) * 65535.0 + 0.5) as u16;
        for i in 0..self.w * self.h {
            let (y, c1, c2) = (self.p[0][i] * S3, self.p[1][i], self.p[2][i]);
            let r = y + c1 * S2 + c2 * S6;
            let g = y - 2.0 * c2 * S6;
            let b = y - c1 * S2 + c2 * S6;
            out[i * 3] = to_code(r);
            out[i * 3 + 1] = to_code(g);
            out[i * 3 + 2] = to_code(b);
        }
        out
    }
}

/// Measure the noise/grain of an RGB16 image. Only a central window of at most 1024×1024 is
/// examined: plenty of samples, and bounded memory for very large scans.
pub fn estimate_noise(rgb: &[u16], w: usize, h: usize, linear: bool, domain: Domain) -> NoiseProfile {
    const MAX: usize = 1024;
    let (ww, wh) = (w.min(MAX), h.min(MAX));
    let (x0, y0) = ((w - ww) / 2, (h - wh) / 2);
    let mut window = Vec::with_capacity(ww * wh * 3);
    for y in y0..y0 + wh {
        window.extend_from_slice(&rgb[(y * w + x0) * 3..(y * w + x0 + ww) * 3]);
    }
    let planes = Planes::from_rgb16(&window, ww, wh, linear, domain);
    grain::estimate(&planes.p, ww, wh)
}

/// Denoise interleaved RGB16 with an explicit noise profile (so tiles of one image agree).
/// The neural method needs its weights, passed in `net`.
pub fn denoise_rgb16(rgb: &[u16], w: usize, h: usize, linear: bool, params: &DenoiseParams, noise: NoiseProfile, net: Option<&neural::Network>) -> Result<Vec<u16>, String> {
    if params.method == Method::Off {
        return Ok(rgb.to_vec());
    }
    let domain = Domain::for_negative(params.negative);
    let profile = noise.scaled(params.luma, params.chroma);
    let mut planes = Planes::from_rgb16(rgb, w, h, linear, domain);
    let (luma_levels, chroma_levels) = profile.levels();
    match params.method {
        Method::Off => unreachable!(),
        Method::Wavelet => {
            wavelet::denoise(&mut planes.p[0], w, h, luma_levels);
            wavelet::denoise(&mut planes.p[1], w, h, chroma_levels);
            wavelet::denoise(&mut planes.p[2], w, h, chroma_levels);
        }
        Method::Nlm => {
            nlm::denoise(&mut planes.p[0], w, h, profile.luma, profile.size);
            // Non-local means on colour would be slow and gains little: colour noise is
            // low-frequency blotches, which the wavelet method handles very well.
            wavelet::denoise(&mut planes.p[1], w, h, chroma_levels);
            wavelet::denoise(&mut planes.p[2], w, h, chroma_levels);
        }
        Method::Neural => {
            let net = net.ok_or("neural network weights are not loaded")?;
            neural::denoise(net, &mut planes, profile);
        }
    }
    Ok(planes.to_rgb16(linear, domain))
}

/// How many pixels of context each method needs around a tile for seamless results.
pub fn margin(method: Method) -> usize {
    match method {
        Method::Off => 0,
        Method::Wavelet => wavelet::MARGIN,
        Method::Nlm => nlm::MARGIN.max(wavelet::MARGIN),
        Method::Neural => neural::MARGIN,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planes_roundtrip() {
        let rgb: Vec<u16> = (0..300).map(|i| (i * 211 % 65536) as u16).collect();
        for linear in [false, true] {
            for domain in [Domain::Gamma, Domain::Density] {
                let back = Planes::from_rgb16(&rgb, 10, 10, linear, domain).to_rgb16(linear, domain);
                for (a, b) in rgb.iter().zip(&back) {
                    // Density can't represent transmittance below 1e-4: tiny linear values clip.
                    if domain == Domain::Density && (*a as f32 / 65535.0) < 2e-3 {
                        continue;
                    }
                    assert!((*a as i32 - *b as i32).abs() <= 3, "{linear} {domain:?}: {a} vs {b}");
                }
            }
        }
    }

    /// The engine denoises big images in tiles, each with `margin()` pixels of context, and
    /// keeps only the tile centres. That must give exactly the same pixels as one big call.
    #[test]
    fn tiles_with_margin_match_whole_image() {
        let (w, h) = (120, 90);
        let rgb: Vec<u16> = (0..w * h * 3).map(|i| ((i * 7919 + (i / 3) * 104_729) % 65536) as u16).collect();
        let mut rng = grain::Rng(3);
        let net = neural::Network {
            layers: (0..neural::LAYERS)
                .map(|i| {
                    let cin = if i == 0 { neural::IN_CHANNELS } else { neural::FEATURES };
                    let cout = if i == neural::LAYERS - 1 { neural::OUT_CHANNELS } else { neural::FEATURES };
                    let mut r = || (rng.uniform() - 0.5) * 0.2;
                    neural::Layer { cin, cout, w: (0..cout * cin * 9).map(|_| r()).collect(), b: (0..cout).map(|_| r()).collect() }
                })
                .collect(),
        };
        let noise = NoiseProfile::new(0.05, 0.05, 1.5);
        for method in [Method::Wavelet, Method::Nlm, Method::Neural] {
            for negative in [false, true] {
                let p = DenoiseParams { method, luma: 1.0, chroma: 1.0, negative };
                let whole = denoise_rgb16(&rgb, w, h, false, &p, noise, Some(&net)).unwrap();
                // A core tile at even coordinates (the network groups pixels in 2×2 blocks).
                let (tx, ty, tw, th): (usize, usize, usize, usize) = (40, 30, 30, 24);
                let m = margin(method);
                let (ex, ey) = (tx.saturating_sub(m), ty.saturating_sub(m));
                let (ew, eh) = ((tx + tw + m).min(w) - ex, (ty + th + m).min(h) - ey);
                let mut sub = Vec::new();
                for y in ey..ey + eh {
                    sub.extend_from_slice(&rgb[(y * w + ex) * 3..(y * w + ex + ew) * 3]);
                }
                let tile = denoise_rgb16(&sub, ew, eh, false, &p, noise, Some(&net)).unwrap();
                for y in 0..th {
                    for x in 0..tw * 3 {
                        let a = whole[((ty + y) * w + tx) * 3 + x];
                        let b = tile[((ty - ey + y) * ew + tx - ex) * 3 + x];
                        assert!((a as i32 - b as i32).abs() <= 1, "{method:?} at {x},{y}: {a} vs {b}");
                    }
                }
            }
        }
    }
}
