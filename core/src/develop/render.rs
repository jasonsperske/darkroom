//! CPU rendering of a view: orient → crop → scale → develop → 8-bit sRGB RGBA.
//!
//! This mirrors the WebGPU fragment shader in `web/js/render/webgpu.js` line for line, and is
//! used when WebGPU is unavailable, for the 100 % loupe (rendered from the full-resolution
//! master), and to compute the histogram.

use super::{Developer, LUT_SIZE};
use crate::color::linear_to_srgb;
use crate::geometry::{NormRect, Orientation};

/// Render `out_w × out_h` RGBA8 pixels showing the `crop` region (in *output* normalised
/// coordinates) of a source buffer with `ch` interleaved channels.
#[allow(clippy::too_many_arguments)]
pub fn render_view(
    dev: &Developer,
    src: &[u16],
    sw: usize,
    sh: usize,
    ch: usize,
    orient: Orientation,
    crop: NormRect,
    out_w: usize,
    out_h: usize,
) -> Vec<u8> {
    let mut out = vec![255u8; out_w * out_h * 4];
    let lut = &dev.lut;
    // Develop one source pixel through the per-channel LUT (no channel mixing yet).
    let tap = |x: usize, y: usize| -> [f32; 3] {
        let i = (y * sw + x) * ch;
        [lut[src[i] as usize], lut[LUT_SIZE + src[i + 1] as usize], lut[2 * LUT_SIZE + src[i + 2] as usize]]
    };
    // 256-entry encode table for 8-bit output (indexed by quantised linear value, 12 bits).
    let encode: Vec<u8> = (0..4096).map(|i| (linear_to_srgb(i as f32 / 4095.0) * 255.0 + 0.5) as u8).collect();

    for oy in 0..out_h {
        let v = crop[1] + (oy as f32 + 0.5) / out_h as f32 * (crop[3] - crop[1]);
        for ox in 0..out_w {
            let u = crop[0] + (ox as f32 + 0.5) / out_w as f32 * (crop[2] - crop[0]);
            let (su, sv) = orient.map_normalized(u, v);
            // Bilinear interpolation between the four nearest source pixels, in linear light.
            let fx = (su * sw as f32 - 0.5).clamp(0.0, (sw - 1) as f32);
            let fy = (sv * sh as f32 - 0.5).clamp(0.0, (sh - 1) as f32);
            let (x0, y0) = (fx as usize, fy as usize);
            let (x1, y1) = ((x0 + 1).min(sw - 1), (y0 + 1).min(sh - 1));
            let (ax, ay) = (fx - x0 as f32, fy - y0 as f32);
            let (p00, p10, p01, p11) = (tap(x0, y0), tap(x1, y0), tap(x0, y1), tap(x1, y1));
            let mut px = [0f32; 3];
            for c in 0..3 {
                let top = p00[c] + (p10[c] - p00[c]) * ax;
                let bottom = p01[c] + (p11[c] - p01[c]) * ax;
                px[c] = top + (bottom - top) * ay;
            }
            let px = dev.look.mix(px[0], px[1], px[2]);
            let d = (oy * out_w + ox) * 4;
            for c in 0..3 {
                out[d + c] = encode[(px[c] * 4095.0 + 0.5) as usize];
            }
        }
    }
    out
}

/// 256-bin histograms of an RGBA8 buffer: `[R…, G…, B…, Luma…]`, 1024 counts.
pub fn histogram_rgba8(px: &[u8]) -> Vec<u32> {
    let mut h = vec![0u32; 1024];
    for p in px.chunks_exact(4) {
        h[p[0] as usize] += 1;
        h[256 + p[1] as usize] += 1;
        h[512 + p[2] as usize] += 1;
        let y = (54 * p[0] as u32 + 183 * p[1] as u32 + 19 * p[2] as u32) >> 8;
        h[768 + y as usize] += 1;
    }
    h
}
