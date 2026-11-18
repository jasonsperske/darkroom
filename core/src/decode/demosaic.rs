//! Demosaicing: estimating full RGB at every photosite from a colour filter array.
//!
//! Two algorithms:
//!
//! * **Malvar–He–Cutler** (2004) for 2×2 Bayer patterns. It is bilinear interpolation plus a
//!   correction term: the *local gradient of the colour we do know* is used to sharpen the
//!   estimate of the colours we don't (luminance edges are shared between channels). Each
//!   missing value is a fixed 5×5 linear filter, so it is fast and has no tuning knobs, and it
//!   is dramatically better than plain bilinear at edges.
//! * **Neighbour averaging** for anything else (e.g. Fujifilm X-Trans 6×6): each missing colour
//!   is the mean of same-coloured photosites in the 3×3 (or, if none, 5×5) neighbourhood.
//!   Simple and artefact-prone, but correct for any pattern.
//!
//! Input: one f32 per photosite (already black-subtracted, scaled and white-balanced).
//! Output: interleaved RGB f32.

pub struct Pattern {
    pub width: usize,
    pub height: usize,
    /// Colour index (0 = R, 1 = G, 2 = B) for each position in the repeating pattern.
    pub colors: Vec<u8>,
}

impl Pattern {
    #[inline]
    fn at(&self, x: usize, y: usize) -> usize {
        self.colors[(y % self.height) * self.width + (x % self.width)] as usize
    }

    fn is_bayer(&self) -> bool {
        if self.width != 2 || self.height != 2 {
            return false;
        }
        let mut sorted = self.colors.clone();
        sorted.sort();
        // One R, one B and two G on a diagonal: [0]=(0,0) [1]=(1,0) [2]=(0,1) [3]=(1,1).
        let c = &self.colors;
        sorted == [0, 1, 1, 2] && ((c[0] == 1 && c[3] == 1) || (c[1] == 1 && c[2] == 1))
    }
}

pub fn demosaic(mosaic: &[f32], w: usize, h: usize, pattern: &Pattern) -> Vec<f32> {
    if pattern.is_bayer() && w >= 5 && h >= 5 {
        malvar_he_cutler(mosaic, w, h, pattern)
    } else {
        neighbour_average(mosaic, w, h, pattern)
    }
}

// --- Malvar–He–Cutler ---------------------------------------------------------------------------
//
// The four 5×5 kernels from the paper, ×8 so they are integers. Written out in full on purpose:
// this *is* the algorithm.

/// Green at a red or blue site.
const K_G_AT_RB: [[f32; 5]; 5] = [
    [0.0, 0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, 2.0, 0.0, 0.0],
    [-1.0, 2.0, 4.0, 2.0, -1.0],
    [0.0, 0.0, 2.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0, 0.0],
];
/// Red (or blue) at a green site whose *row* contains red (or blue) neighbours.
const K_RB_AT_G_ROW: [[f32; 5]; 5] = [
    [0.0, 0.0, 0.5, 0.0, 0.0],
    [0.0, -1.0, 0.0, -1.0, 0.0],
    [-1.0, 4.0, 5.0, 4.0, -1.0],
    [0.0, -1.0, 0.0, -1.0, 0.0],
    [0.0, 0.0, 0.5, 0.0, 0.0],
];
/// Red (or blue) at a green site whose *column* contains red (or blue) neighbours.
const K_RB_AT_G_COL: [[f32; 5]; 5] = [
    [0.0, 0.0, -1.0, 0.0, 0.0],
    [0.0, -1.0, 4.0, -1.0, 0.0],
    [0.5, 0.0, 5.0, 0.0, 0.5],
    [0.0, -1.0, 4.0, -1.0, 0.0],
    [0.0, 0.0, -1.0, 0.0, 0.0],
];
/// Red at a blue site, or blue at a red site.
const K_RB_AT_BR: [[f32; 5]; 5] = [
    [0.0, 0.0, -1.5, 0.0, 0.0],
    [0.0, 2.0, 0.0, 2.0, 0.0],
    [-1.5, 0.0, 6.0, 0.0, -1.5],
    [0.0, 2.0, 0.0, 2.0, 0.0],
    [0.0, 0.0, -1.5, 0.0, 0.0],
];

fn malvar_he_cutler(m: &[f32], w: usize, h: usize, p: &Pattern) -> Vec<f32> {
    let mut out = vec![0f32; w * h * 3];

    // Convolve one kernel at (x, y). Interior pixels index directly; the 2-pixel border mirrors
    // coordinates by whole pattern periods (±2) so the colour under each tap stays the same.
    let conv = |k: &[[f32; 5]; 5], x: usize, y: usize| -> f32 {
        let interior = x >= 2 && y >= 2 && x + 2 < w && y + 2 < h;
        let mut sum = 0.0;
        for (ky, row) in k.iter().enumerate() {
            for (kx, &kv) in row.iter().enumerate() {
                if kv == 0.0 {
                    continue;
                }
                let (sx, sy) = if interior {
                    (x + kx - 2, y + ky - 2)
                } else {
                    let mirror = |c: isize, n: usize| -> usize {
                        let mut c = c;
                        while c < 0 {
                            c += 2;
                        }
                        while c >= n as isize {
                            c -= 2;
                        }
                        c as usize
                    };
                    (mirror(x as isize + kx as isize - 2, w), mirror(y as isize + ky as isize - 2, h))
                };
                sum += kv * m[sy * w + sx];
            }
        }
        (sum / 8.0).max(0.0)
    };

    for y in 0..h {
        for x in 0..w {
            let here = p.at(x, y);
            let v = m[y * w + x];
            let px = match here {
                1 => {
                    // Green site: which colour lives in this row?
                    let row_color = p.at(x + 1, y);
                    let along_row = conv(&K_RB_AT_G_ROW, x, y);
                    let along_col = conv(&K_RB_AT_G_COL, x, y);
                    if row_color == 0 {
                        [along_row, v, along_col]
                    } else {
                        [along_col, v, along_row]
                    }
                }
                0 => [v, conv(&K_G_AT_RB, x, y), conv(&K_RB_AT_BR, x, y)],
                _ => [conv(&K_RB_AT_BR, x, y), conv(&K_G_AT_RB, x, y), v],
            };
            out[(y * w + x) * 3..(y * w + x) * 3 + 3].copy_from_slice(&px);
        }
    }
    out
}

// --- Generic fallback -------------------------------------------------------------------------

fn neighbour_average(m: &[f32], w: usize, h: usize, p: &Pattern) -> Vec<f32> {
    let mut out = vec![0f32; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let here = p.at(x, y);
            for c in 0..3 {
                let v = if c == here {
                    m[y * w + x]
                } else {
                    let mut result = 0.0;
                    for radius in [1isize, 2] {
                        let (mut sum, mut n) = (0.0, 0);
                        for dy in -radius..=radius {
                            for dx in -radius..=radius {
                                let (sx, sy) = (x as isize + dx, y as isize + dy);
                                if sx < 0 || sy < 0 || sx >= w as isize || sy >= h as isize {
                                    continue;
                                }
                                if p.at(sx as usize, sy as usize) == c {
                                    sum += m[sy as usize * w + sx as usize];
                                    n += 1;
                                }
                            }
                        }
                        if n > 0 {
                            result = sum / n as f32;
                            break;
                        }
                    }
                    result
                };
                out[(y * w + x) * 3 + c] = v;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_field_stays_flat() {
        // A uniformly lit, white-balanced sensor must demosaic to uniform grey.
        let (w, h) = (16, 12);
        let mosaic = vec![0.5f32; w * h];
        for colors in [vec![0, 1, 1, 2], vec![1, 0, 2, 1], vec![2, 1, 1, 0], vec![1, 2, 0, 1]] {
            let p = Pattern { width: 2, height: 2, colors };
            assert!(p.is_bayer());
            let out = demosaic(&mosaic, w, h, &p);
            assert!(out.iter().all(|&v| (v - 0.5).abs() < 1e-5));
        }
    }
}
