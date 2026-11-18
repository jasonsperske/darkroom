//! Crop and rotate in the DCT domain.
//!
//! ## Inside a block
//!
//! Let `S[v][u]` be a block's coefficients (`u` = horizontal frequency, `v` = vertical). The
//! DCT basis functions are cosines; the odd ones are antisymmetric about the block centre, the
//! even ones symmetric. So:
//!
//! * mirroring the block left↔right multiplies every coefficient with odd `u` by −1;
//! * mirroring top↔bottom multiplies every coefficient with odd `v` by −1;
//! * transposing the block (swap x and y) transposes the coefficient matrix.
//!
//! With our orientation convention (transpose, then flip X, then flip Y – see
//! `geometry.rs`) the output coefficient is
//!
//! ```text
//! OUT[v][u] = (−1)^(u·flip_x) · (−1)^(v·flip_y) · (transpose ? S[u][v] : S[v][u])
//! ```
//!
//! Quantisation tables must be transposed too when the coefficients are, and each component's
//! sampling factors swap (a 2×1 chroma subsampling becomes 1×2).
//!
//! ## Between blocks
//!
//! Whole blocks move exactly like pixels do, using the same orientation mapping on the block
//! grid. An edge that ends up at the *left* or *top* of the output must start on an MCU
//! boundary, otherwise the partial blocks hanging off the source's right/bottom edge (padding
//! that decoders normally never show) would become visible. [`snap_crop`] enforces that.

use super::parse::{ComponentInfo, Header, JpegFile};
use crate::geometry::{Orientation, PixelRect};

/// Snap a source-pixel crop to what can be done losslessly.
///
/// * Left/top edges round *down* to the MCU grid (the crop grows slightly).
/// * Right/bottom edges stay where they are – unless the orientation moves them to the
///   output's left/top, in which case the crop size is rounded down to whole MCUs.
pub fn snap_crop(h: &Header, o: Orientation, crop: PixelRect) -> Result<PixelRect, String> {
    let snap = |a0: usize, a1: usize, mcu: usize, size: usize, reversed: bool| -> Result<(usize, usize), String> {
        let a1 = a1.min(size);
        let s0 = a0 / mcu * mcu;
        let s1 = if reversed {
            let whole = (a1 - s0) / mcu;
            if whole == 0 {
                return Err(format!("crop is smaller than one {mcu}-pixel JPEG block"));
            }
            s0 + whole * mcu
        } else {
            a1
        };
        Ok((s0, s1))
    };
    let (x0, x1) = snap(crop.x0, crop.x1, h.mcu_width(), h.width, o.source_x_reversed())?;
    let (y0, y1) = snap(crop.y0, crop.y1, h.mcu_height(), h.height, o.source_y_reversed())?;
    Ok(PixelRect { x0, y0, x1, y1 })
}

/// Produce a new coefficient file: `crop` (already snapped) then `o`.
pub fn apply(src: &JpegFile, o: Orientation, crop: PixelRect) -> JpegFile {
    let sh = &src.header;
    let (cw, ch) = (crop.width(), crop.height());
    let (width, height) = o.output_size(cw, ch);

    let components: Vec<ComponentInfo> = sh
        .components
        .iter()
        .map(|c| if o.transpose { ComponentInfo { h: c.v, v: c.h, ..c.clone() } } else { c.clone() })
        .collect();
    let (hmax, vmax) = if o.transpose { (sh.vmax, sh.hmax) } else { (sh.hmax, sh.vmax) };
    let header = Header { width, height, progressive: false, components, hmax, vmax };

    let mut coefs = Vec::new();
    for c in 0..sh.components.len() {
        let comp = &sh.components[c];
        // Where the crop starts in this component's block grid.
        let bx0 = crop.x0 / sh.mcu_width() * comp.h;
        let by0 = crop.y0 / sh.mcu_height() * comp.v;
        let (src_bw, src_bh) = sh.padded_blocks(c);
        let (out_bw, out_bh) = header.padded_blocks(c);
        let mut out = vec![0i16; out_bw * out_bh * 64];

        for oby in 0..out_bh {
            for obx in 0..out_bw {
                // Same mapping as for pixels, applied to the block grid.
                let x = if o.flip_x { out_bw - 1 - obx } else { obx };
                let y = if o.flip_y { out_bh - 1 - oby } else { oby };
                let (sbx, sby) = if o.transpose { (y, x) } else { (x, y) };
                let (sbx, sby) = (bx0 + sbx, by0 + sby);
                if sbx >= src_bw || sby >= src_bh {
                    continue; // padding beyond the source; never displayed
                }
                let s = &src.coefs[c][(sby * src_bw + sbx) * 64..][..64];
                let d = &mut out[(oby * out_bw + obx) * 64..][..64];
                for v in 0..8 {
                    for u in 0..8 {
                        let value = if o.transpose { s[u * 8 + v] } else { s[v * 8 + u] };
                        let negate = (o.flip_x && u % 2 == 1) ^ (o.flip_y && v % 2 == 1);
                        d[v * 8 + u] = if negate { value.wrapping_neg() } else { value };
                    }
                }
            }
        }
        coefs.push(out);
    }

    let mut qtables = src.qtables;
    if o.transpose {
        for t in qtables.iter_mut().flatten() {
            let orig = *t;
            for v in 0..8 {
                for u in 0..8 {
                    t[v * 8 + u] = orig[u * 8 + v];
                }
            }
        }
    }

    JpegFile { header, qtables, segments: src.segments.clone(), coefs }
}
