//! Running the developer over pixels.
//!
//! Per pixel:
//! 1. three LUT reads give the per-channel display-linear values (`tone.rs`);
//! 2. saturation / monochrome mixes the channels around their luminance;
//! 3. the sRGB transfer curve encodes the result for the file or screen – again via a table.
//!
//! ## Where SIMD helps (and where it can't)
//!
//! WebAssembly SIMD (`simd128`) processes four `f32` lanes per instruction. Step 2 is pure
//! arithmetic and vectorises perfectly: we develop four pixels at once with each colour channel
//! in its own register ("structure of arrays"). Steps 1 and 3 are table lookups; wasm SIMD has
//! no *gather* instruction, so those stay scalar. The scalar and SIMD versions below compute
//! exactly the same thing; the SIMD one is compiled only when the `simd128` target feature is
//! on (the default build – see `build.sh`, which also produces a fallback without it).

use super::{Developer, Params, LUT_SIZE};
use crate::color::{linear_to_srgb, LUMA};
use std::sync::OnceLock;

/// Channel-mixing settings applied after the per-channel LUT.
#[derive(Clone, Copy, Debug)]
pub struct Look {
    pub saturation: f32,
}

impl Look {
    pub fn from_params(p: &Params) -> Self {
        Look { saturation: if p.monochrome { 0.0 } else { p.saturation.max(0.0) } }
    }

    /// Saturation around Rec.709 luminance; 0 = black & white. Output clamped to `[0,1]`.
    #[inline]
    pub fn mix(&self, r: f32, g: f32, b: f32) -> [f32; 3] {
        let y = LUMA[0] * r + LUMA[1] * g + LUMA[2] * b;
        let s = self.saturation;
        [
            (y + (r - y) * s).clamp(0.0, 1.0),
            (y + (g - y) * s).clamp(0.0, 1.0),
            (y + (b - y) * s).clamp(0.0, 1.0),
        ]
    }
}

/// Linear light (quantised to 16 bits) → sRGB-encoded 16-bit code value.
pub fn oetf_table() -> &'static [u16] {
    static TABLE: OnceLock<Vec<u16>> = OnceLock::new();
    TABLE.get_or_init(|| {
        (0..LUT_SIZE)
            .map(|i| (linear_to_srgb(i as f32 / 65535.0) * 65535.0 + 0.5) as u16)
            .collect()
    })
}

#[inline]
fn quantize(v: f32) -> usize {
    (v * 65535.0 + 0.5) as usize
}

impl Developer {
    /// Develop one pixel to display-linear `[0,1]` RGB (no output encoding).
    #[inline]
    pub fn develop_linear(&self, r: u16, g: u16, b: u16) -> [f32; 3] {
        let lut = &self.lut;
        self.look.mix(lut[r as usize], lut[LUT_SIZE + g as usize], lut[2 * LUT_SIZE + b as usize])
    }

    /// Develop interleaved RGB `u16` pixels to sRGB-encoded RGB `u16`.
    /// This is the export hot path; it is called by every worker in the pool on its strip.
    pub fn process_rgb16(&self, src: &[u16]) -> Vec<u16> {
        let mut out = vec![0u16; src.len()];
        let n = src.len() / 3;
        #[cfg(target_feature = "simd128")]
        let done = unsafe { self.process_rgb16_simd(src, &mut out) };
        #[cfg(not(target_feature = "simd128"))]
        let done = 0;
        // Scalar path: handles everything without SIMD, and the last <4 pixels with it.
        let oetf = oetf_table();
        for i in done..n {
            let px = self.develop_linear(src[i * 3], src[i * 3 + 1], src[i * 3 + 2]);
            for c in 0..3 {
                out[i * 3 + c] = oetf[quantize(px[c])];
            }
        }
        out
    }

    /// Four pixels per iteration. Returns how many pixels it processed.
    #[cfg(target_feature = "simd128")]
    unsafe fn process_rgb16_simd(&self, src: &[u16], out: &mut [u16]) -> usize {
        use core::arch::wasm32::*;
        let lut = &self.lut;
        let oetf = oetf_table();
        let (lr, lg, lb) = (&lut[..LUT_SIZE], &lut[LUT_SIZE..2 * LUT_SIZE], &lut[2 * LUT_SIZE..]);

        let wr = f32x4_splat(LUMA[0]);
        let wg = f32x4_splat(LUMA[1]);
        let wb = f32x4_splat(LUMA[2]);
        let sat = f32x4_splat(self.look.saturation);
        let zero = f32x4_splat(0.0);
        let one = f32x4_splat(1.0);
        let scale = f32x4_splat(65535.0);
        let half = f32x4_splat(0.5);

        let quads = src.len() / 12;
        for q in 0..quads {
            let s = &src[q * 12..q * 12 + 12];
            // Gather: scalar LUT reads into lanes (R of pixels 0..3, G of pixels 0..3, ...).
            let r = f32x4(lr[s[0] as usize], lr[s[3] as usize], lr[s[6] as usize], lr[s[9] as usize]);
            let g = f32x4(lg[s[1] as usize], lg[s[4] as usize], lg[s[7] as usize], lg[s[10] as usize]);
            let b = f32x4(lb[s[2] as usize], lb[s[5] as usize], lb[s[8] as usize], lb[s[11] as usize]);

            // y = 0.2126 r + 0.7152 g + 0.0722 b, for four pixels at once.
            let y = f32x4_add(f32x4_add(f32x4_mul(r, wr), f32x4_mul(g, wg)), f32x4_mul(b, wb));
            // c' = clamp(y + (c - y) * saturation, 0, 1), then quantise to a 16-bit linear index.
            let mix = |c: v128| {
                let v = f32x4_add(y, f32x4_mul(f32x4_sub(c, y), sat));
                let v = f32x4_min(f32x4_max(v, zero), one);
                i32x4_trunc_sat_f32x4(f32x4_add(f32x4_mul(v, scale), half))
            };
            let (ri, gi, bi) = (mix(r), mix(g), mix(b));

            // Scatter: output encoding table, scalar again.
            let o = &mut out[q * 12..q * 12 + 12];
            macro_rules! lane {
                ($v:expr, $l:literal) => {
                    oetf[i32x4_extract_lane::<$l>($v) as usize]
                };
            }
            o[0] = lane!(ri, 0);
            o[1] = lane!(gi, 0);
            o[2] = lane!(bi, 0);
            o[3] = lane!(ri, 1);
            o[4] = lane!(gi, 1);
            o[5] = lane!(bi, 1);
            o[6] = lane!(ri, 2);
            o[7] = lane!(gi, 2);
            o[8] = lane!(bi, 2);
            o[9] = lane!(ri, 3);
            o[10] = lane!(gi, 3);
            o[11] = lane!(bi, 3);
        }
        quads * 4
    }
}
