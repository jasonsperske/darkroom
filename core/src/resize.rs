//! Area-average ("box") downscaling with fractional ratios.
//!
//! Every output pixel is the exact average of the source area it covers, with partial source
//! pixels weighted by how much of them is covered. This is the ideal anti-aliasing filter for
//! *reduction* (it is what you'd get by physically integrating light over a bigger sensor
//! pixel) and is simple to reason about. It is used for the on-screen proxy and for export
//! resizing.
//!
//! The implementation is separable and streaming: each source row is resampled horizontally
//! once, then accumulated into the one or two output rows it overlaps. Memory use is a couple
//! of output rows, independent of the source height.

/// One output sample = Σ weight·source[start + i].
struct Span {
    start: usize,
    weights: Vec<f32>,
}

/// Precompute, for every output index along one axis, which source indices contribute and how.
fn spans(src_len: usize, dst_len: usize) -> Vec<Span> {
    let scale = src_len as f64 / dst_len as f64; // source pixels per output pixel (≥ 1)
    (0..dst_len)
        .map(|o| {
            let a = o as f64 * scale;
            let b = ((o + 1) as f64 * scale).min(src_len as f64);
            let start = a.floor() as usize;
            let end = (b.ceil() as usize).min(src_len);
            let weights = (start..end)
                .map(|s| {
                    let overlap = (b.min(s as f64 + 1.0) - a.max(s as f64)).max(0.0);
                    (overlap / scale) as f32
                })
                .collect();
            Span { start, weights }
        })
        .collect()
}

/// Downscale an interleaved `u16` image with `ch` channels to `dw × dh`.
/// `dw ≤ sw` and `dh ≤ sh` are required (this is a reduction filter).
pub fn downscale_u16(src: &[u16], sw: usize, sh: usize, ch: usize, dw: usize, dh: usize) -> Vec<u16> {
    assert!(dw >= 1 && dh >= 1 && dw <= sw && dh <= sh);
    if dw == sw && dh == sh {
        return src.to_vec();
    }
    let hspans = spans(sw, dw);
    let vspans = spans(sh, dh);

    // For each source row, list the (output row, weight) pairs it contributes to.
    let mut row_targets: Vec<Vec<(usize, f32)>> = vec![Vec::new(); sh];
    for (oy, sp) in vspans.iter().enumerate() {
        for (i, &w) in sp.weights.iter().enumerate() {
            row_targets[sp.start + i].push((oy, w));
        }
    }

    let mut out = vec![0u16; dw * dh * ch];
    let mut hrow = vec![0f32; dw * ch];
    // Accumulators for output rows that are still being filled. An output row only receives
    // contributions from a contiguous run of source rows, so we finalise rows in order.
    let mut acc: std::collections::VecDeque<(usize, Vec<f32>)> = Default::default();

    for sy in 0..sh {
        // Horizontal pass for this source row.
        let row = &src[sy * sw * ch..(sy + 1) * sw * ch];
        for (ox, sp) in hspans.iter().enumerate() {
            for c in 0..ch {
                let mut sum = 0f32;
                for (i, &w) in sp.weights.iter().enumerate() {
                    sum += w * row[(sp.start + i) * ch + c] as f32;
                }
                hrow[ox * ch + c] = sum;
            }
        }
        // Vertical accumulation.
        for &(oy, w) in &row_targets[sy] {
            if acc.back().map_or(true, |(y, _)| *y < oy) {
                acc.push_back((oy, vec![0f32; dw * ch]));
            }
            let buf = &mut acc.iter_mut().find(|(y, _)| *y == oy).unwrap().1;
            for (a, &h) in buf.iter_mut().zip(hrow.iter()) {
                *a += w * h;
            }
        }
        // Any output row whose last contributing source row was this one is complete.
        while let Some((oy, _)) = acc.front() {
            let sp = &vspans[*oy];
            if sp.start + sp.weights.len() - 1 > sy {
                break;
            }
            let (oy, buf) = acc.pop_front().unwrap();
            let dst = &mut out[oy * dw * ch..(oy + 1) * dw * ch];
            for (d, v) in dst.iter_mut().zip(buf) {
                *d = v.round().clamp(0.0, 65535.0) as u16;
            }
        }
    }
    out
}

/// Fit `w × h` inside a `max_edge` square, preserving aspect ratio, never upscaling.
pub fn fit_within(w: usize, h: usize, max_edge: usize) -> (usize, usize) {
    let long = w.max(h);
    if long <= max_edge {
        return (w, h);
    }
    let s = max_edge as f64 / long as f64;
    (((w as f64 * s).round() as usize).max(1), ((h as f64 * s).round() as usize).max(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halves_average() {
        // 4×1 → 2×1: averages pairs.
        let out = downscale_u16(&[0, 100, 200, 400], 4, 1, 1, 2, 1);
        assert_eq!(out, vec![50, 300]);
    }

    #[test]
    fn fractional_preserves_mean() {
        let src: Vec<u16> = (0..7 * 5).map(|i| (i * 997 % 4096) as u16).collect();
        let out = downscale_u16(&src, 7, 5, 1, 3, 2);
        let mean_src: f64 = src.iter().map(|&v| v as f64).sum::<f64>() / src.len() as f64;
        let mean_out: f64 = out.iter().map(|&v| v as f64).sum::<f64>() / out.len() as f64;
        assert!((mean_src - mean_out).abs() < 2.0, "{mean_src} vs {mean_out}");
    }
}
