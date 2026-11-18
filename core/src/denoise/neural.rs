//! A small convolutional denoising network ("FFDNet-lite") and its inference.
//!
//! ## Architecture
//!
//! Modelled on FFDNet (Zhang, Zuo & Zhang, 2018), shrunk to run in a browser:
//!
//! ```text
//!  Y,C1,C2 planes (W×H)
//!     │ pixel-unshuffle: each 2×2 block → 4 channels      12 × W/2 × H/2
//!     │ + three constant maps (σ_luma, σ_chroma, grain size) 15 channels
//!     ▼
//!  conv 3×3, 15→32, ReLU
//!  conv 3×3, 32→32, ReLU   ×4
//!  conv 3×3, 32→12                                         predicted noise
//!     ▼
//!  input − predicted noise, pixel-shuffle back             denoised W×H
//! ```
//!
//! * **Residual learning** (from DnCNN): the network predicts the *noise*, which is easier to
//!   learn than the clean image and makes "no noise" a trivial output.
//! * **Noise maps** (from FFDNet): σ is an input, so one network handles every noise level, and
//!   the strength sliders work by telling it the noise is stronger or weaker than estimated.
//!   Separate luma and chroma maps let the colour slider act independently. A third map gives
//!   the *grain size* (`grain.rs`): it is trained on everything from white digital noise to
//!   the clumped grain of fast film, and this tells it which one it is looking at.
//! * **Pixel unshuffle** halves the resolution the convolutions run at, a 4× speed-up, while
//!   keeping every input pixel. It also doubles the receptive field.
//!
//! About 11 000 multiply-adds per output pixel. The weights are trained by
//! `core/examples/train_denoiser.rs` and stored in `web/models/denoise.bin`.
//!
//! ## Weight file format (little-endian)
//!
//! ```text
//! "DRNN"  u32 version = 1  u32 layer_count
//! per layer:  u32 in_channels  u32 out_channels
//!             f32 weights[out][in][3][3]
//!             f32 bias[out]
//! ```
//! A ReLU follows every layer except the last. `web/js/nn/webgpu-denoise.js` reads the same file.

use super::{NoiseProfile, Planes};

pub const FEATURES: usize = 32;
pub const LAYERS: usize = 6;
pub const IN_CHANNELS: usize = 15;
pub const OUT_CHANNELS: usize = 12;
/// Noise maps are multiplied by this so inputs are all of order 1.
pub const SIGMA_SCALE: f32 = 10.0;
/// Grain size (px) is multiplied by this.
pub const SIZE_SCALE: f32 = 0.5;
/// Receptive field: 6 layers of 3×3 at half resolution = ±6 half-res pixels = ±12 pixels.
pub const MARGIN: usize = 16;

pub struct Layer {
    pub cin: usize,
    pub cout: usize,
    /// `[cout][cin][3][3]`
    pub w: Vec<f32>,
    pub b: Vec<f32>,
}

pub struct Network {
    pub layers: Vec<Layer>,
}

impl Network {
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        let mut pos = 0;
        let u32_at = |pos: &mut usize| -> Result<u32, String> {
            let b = bytes.get(*pos..*pos + 4).ok_or("truncated weights")?;
            *pos += 4;
            Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        };
        if bytes.get(0..4) != Some(b"DRNN") {
            return Err("not a Darkroom network file".into());
        }
        pos += 4;
        if u32_at(&mut pos)? != 1 {
            return Err("unsupported network version".into());
        }
        let count = u32_at(&mut pos)? as usize;
        let mut layers = Vec::new();
        for _ in 0..count {
            let cin = u32_at(&mut pos)? as usize;
            let cout = u32_at(&mut pos)? as usize;
            let mut floats = |n: usize| -> Result<Vec<f32>, String> {
                let b = bytes.get(pos..pos + n * 4).ok_or("truncated weights")?;
                pos += n * 4;
                Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
            };
            let w = floats(cout * cin * 9)?;
            let b = floats(cout)?;
            layers.push(Layer { cin, cout, w, b });
        }
        if layers.first().map(|l| l.cin) != Some(IN_CHANNELS) || layers.last().map(|l| l.cout) != Some(OUT_CHANNELS) {
            return Err("network has the wrong shape".into());
        }
        Ok(Network { layers })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = b"DRNN".to_vec();
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&(self.layers.len() as u32).to_le_bytes());
        for l in &self.layers {
            out.extend_from_slice(&(l.cin as u32).to_le_bytes());
            out.extend_from_slice(&(l.cout as u32).to_le_bytes());
            for v in l.w.iter().chain(&l.b) {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        out
    }
}

/// A `channels × height × width` tensor, channel-major ("CHW") like PyTorch.
pub struct Tensor {
    pub c: usize,
    pub h: usize,
    pub w: usize,
    pub data: Vec<f32>,
}

/// Build the network input from working-space planes: unshuffle + noise maps.
/// Odd sizes are padded by repeating the last row/column.
pub fn prepare(planes: &[Vec<f32>; 3], w: usize, h: usize, noise: NoiseProfile) -> Tensor {
    let (hw, hh) = (w.div_ceil(2), h.div_ceil(2));
    let mut t = Tensor { c: IN_CHANNELS, h: hh, w: hw, data: vec![0f32; IN_CHANNELS * hh * hw] };
    let plane_n = hh * hw;
    for (c, plane) in planes.iter().enumerate() {
        for sub in 0..4 {
            let (sy, sx) = (sub / 2, sub % 2);
            let dst = &mut t.data[(c * 4 + sub) * plane_n..(c * 4 + sub + 1) * plane_n];
            for y in 0..hh {
                let py = (2 * y + sy).min(h - 1);
                for x in 0..hw {
                    let px = (2 * x + sx).min(w - 1);
                    dst[y * hw + x] = plane[py * w + px];
                }
            }
        }
    }
    t.data[12 * plane_n..13 * plane_n].fill(noise.luma * SIGMA_SCALE);
    t.data[13 * plane_n..14 * plane_n].fill(noise.chroma * SIGMA_SCALE);
    t.data[14 * plane_n..15 * plane_n].fill(noise.size * SIZE_SCALE);
    t
}

/// Subtract the predicted noise (12 × W/2 × H/2) from the planes, undoing the unshuffle.
pub fn finish(planes: &mut [Vec<f32>; 3], w: usize, h: usize, noise: &Tensor) {
    let plane_n = noise.h * noise.w;
    for (c, plane) in planes.iter_mut().enumerate() {
        for sub in 0..4 {
            let (sy, sx) = (sub / 2, sub % 2);
            let src = &noise.data[(c * 4 + sub) * plane_n..(c * 4 + sub + 1) * plane_n];
            for y in 0..noise.h {
                let py = 2 * y + sy;
                if py >= h {
                    continue;
                }
                for x in 0..noise.w {
                    let px = 2 * x + sx;
                    if px < w {
                        plane[py * w + px] -= src[y * noise.w + x];
                    }
                }
            }
        }
    }
}

/// `dst[i] += a * src[i]` – the innermost loop of every convolution.
#[inline]
fn axpy(dst: &mut [f32], a: f32, src: &[f32]) {
    #[cfg(target_feature = "simd128")]
    {
        use core::arch::wasm32::*;
        let n4 = dst.len() / 4 * 4;
        let av = f32x4_splat(a);
        for i in (0..n4).step_by(4) {
            // SAFETY: i + 4 ≤ n4 ≤ len of both slices (callers pass equal lengths).
            unsafe {
                let d = v128_load(dst.as_ptr().add(i) as *const v128);
                let s = v128_load(src.as_ptr().add(i) as *const v128);
                v128_store(dst.as_mut_ptr().add(i) as *mut v128, f32x4_add(d, f32x4_mul(av, s)));
            }
        }
        for i in n4..dst.len() {
            dst[i] += a * src[i];
        }
    }
    #[cfg(not(target_feature = "simd128"))]
    for (d, s) in dst.iter_mut().zip(src) {
        *d += a * s;
    }
}

/// 3×3 convolution with zero padding ("same" size), optionally followed by ReLU.
///
/// Loop order keeps one output row hot in cache while every input channel and kernel tap is
/// added into it; each `axpy` call is a long contiguous run that vectorises well.
pub fn conv3x3(input: &Tensor, layer: &Layer, relu: bool) -> Tensor {
    let (h, w) = (input.h, input.w);
    let plane = h * w;
    let mut out = Tensor { c: layer.cout, h, w, data: vec![0f32; layer.cout * plane] };
    for co in 0..layer.cout {
        let wco = &layer.w[co * layer.cin * 9..(co + 1) * layer.cin * 9];
        for y in 0..h {
            let row = &mut out.data[co * plane + y * w..co * plane + (y + 1) * w];
            row.fill(layer.b[co]);
            for ci in 0..layer.cin {
                let src_plane = &input.data[ci * plane..(ci + 1) * plane];
                for ky in 0..3 {
                    let sy = y as isize + ky as isize - 1;
                    if sy < 0 || sy >= h as isize {
                        continue; // zero padding
                    }
                    let src_row = &src_plane[sy as usize * w..(sy as usize + 1) * w];
                    let k = &wco[ci * 9 + ky * 3..ci * 9 + ky * 3 + 3];
                    if w == 1 {
                        row[0] += k[1] * src_row[0];
                        continue;
                    }
                    // kx = 0 reads x − 1, kx = 1 reads x, kx = 2 reads x + 1.
                    axpy(&mut row[1..], k[0], &src_row[..w - 1]);
                    axpy(row, k[1], src_row);
                    axpy(&mut row[..w - 1], k[2], &src_row[1..]);
                }
            }
            if relu {
                row.iter_mut().for_each(|v| *v = v.max(0.0));
            }
        }
    }
    out
}

/// Run the whole network: returns the predicted noise.
pub fn forward(net: &Network, input: &Tensor) -> Tensor {
    let last = net.layers.len() - 1;
    let mut x = conv3x3(input, &net.layers[0], last != 0);
    for (i, layer) in net.layers.iter().enumerate().skip(1) {
        x = conv3x3(&x, layer, i != last);
    }
    x
}

pub fn denoise(net: &Network, planes: &mut Planes, noise: NoiseProfile) {
    let input = prepare(&planes.p, planes.w, planes.h, noise);
    let noise = forward(net, &input);
    finish(&mut planes.p, planes.w, planes.h, &noise);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unshuffle_roundtrip_with_zero_noise() {
        let (w, h) = (7, 5);
        let mut planes: [Vec<f32>; 3] = std::array::from_fn(|c| (0..w * h).map(|i| (i * (c + 3)) as f32).collect());
        let orig = planes.clone();
        let t = prepare(&planes, w, h, NoiseProfile::new(0.1, 0.2, 1.0));
        assert_eq!((t.c, t.w, t.h), (15, 4, 3));
        let zero = Tensor { c: 12, h: t.h, w: t.w, data: vec![0.0; 12 * t.h * t.w] };
        finish(&mut planes, w, h, &zero);
        assert_eq!(planes, orig);
    }

    #[test]
    fn conv_matches_naive() {
        let layer = Layer { cin: 2, cout: 3, w: (0..54).map(|i| (i as f32 * 0.37).sin()).collect(), b: vec![0.1, -0.2, 0.3] };
        let input = Tensor { c: 2, h: 4, w: 5, data: (0..40).map(|i| (i as f32 * 0.11).cos()).collect() };
        let out = conv3x3(&input, &layer, false);
        for co in 0..3 {
            for y in 0..4isize {
                for x in 0..5isize {
                    let mut s = layer.b[co];
                    for ci in 0..2 {
                        for ky in 0..3isize {
                            for kx in 0..3isize {
                                let (sy, sx) = (y + ky - 1, x + kx - 1);
                                if (0..4).contains(&sy) && (0..5).contains(&sx) {
                                    s += layer.w[((co * 2 + ci) * 3 + ky as usize) * 3 + kx as usize] * input.data[ci * 20 + sy as usize * 5 + sx as usize];
                                }
                            }
                        }
                    }
                    let got = out.data[co * 20 + y as usize * 5 + x as usize];
                    assert!((got - s).abs() < 1e-5);
                }
            }
        }
    }
}
