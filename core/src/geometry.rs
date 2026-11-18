//! Orientation (rotate / flip) and crop geometry.
//!
//! ## Orientation as three booleans
//!
//! The 8 ways to rotate/mirror a rectangle (the dihedral group D4, the same 8 values as the EXIF
//! `Orientation` tag) can all be written as: *optionally transpose, then optionally flip X, then
//! optionally flip Y.* We describe the mapping from an **output** pixel back to the **source**
//! pixel it comes from, because that is what a renderer needs ("for each pixel I draw, where do
//! I read from?"):
//!
//! ```text
//!   x = flip_x ? (W' - 1 - ox) : ox        (W', H' = output size)
//!   y = flip_y ? (H' - 1 - oy) : oy
//!   (sx, sy) = transpose ? (y, x) : (x, y)
//! ```
//!
//! The same mapping, in normalised `[0,1]` coordinates, is used by the WebGPU shader
//! (`web/js/render/webgpu.js`) and the JavaScript UI (`web/js/geometry.js`).

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Orientation {
    pub transpose: bool,
    pub flip_x: bool,
    pub flip_y: bool,
}

impl Orientation {
    pub const IDENTITY: Orientation = Orientation { transpose: false, flip_x: false, flip_y: false };

    /// EXIF orientation code (1–8) → orientation. Unknown codes are treated as 1 (identity).
    pub fn from_exif(code: u32) -> Self {
        let (transpose, flip_x, flip_y) = match code {
            2 => (false, true, false),  // mirror horizontal
            3 => (false, true, true),   // rotate 180
            4 => (false, false, true),  // mirror vertical
            5 => (true, false, false),  // transpose
            6 => (true, true, false),   // rotate 90° clockwise
            7 => (true, true, true),    // transverse
            8 => (true, false, true),   // rotate 90° counter-clockwise
            _ => (false, false, false), // 1 = normal
        };
        Orientation { transpose, flip_x, flip_y }
    }

    pub fn to_exif(self) -> u32 {
        match (self.transpose, self.flip_x, self.flip_y) {
            (false, false, false) => 1,
            (false, true, false) => 2,
            (false, true, true) => 3,
            (false, false, true) => 4,
            (true, false, false) => 5,
            (true, true, false) => 6,
            (true, true, true) => 7,
            (true, false, true) => 8,
        }
    }

    /// Size of the output for a source of `w × h`.
    pub fn output_size(self, w: usize, h: usize) -> (usize, usize) {
        if self.transpose {
            (h, w)
        } else {
            (w, h)
        }
    }

    /// Map normalised output coordinates `(u, v) ∈ [0,1]²` to normalised source coordinates.
    #[inline]
    pub fn map_normalized(self, u: f32, v: f32) -> (f32, f32) {
        let x = if self.flip_x { 1.0 - u } else { u };
        let y = if self.flip_y { 1.0 - v } else { v };
        if self.transpose {
            (y, x)
        } else {
            (x, y)
        }
    }

    /// Is the source X axis running backwards in the output? (Needed by the lossless JPEG code.)
    pub fn source_x_reversed(self) -> bool {
        if self.transpose {
            self.flip_y
        } else {
            self.flip_x
        }
    }

    pub fn source_y_reversed(self) -> bool {
        if self.transpose {
            self.flip_x
        } else {
            self.flip_y
        }
    }
}

/// An axis-aligned rectangle in normalised coordinates: `[x0, y0, x1, y1]`, each in `[0,1]`.
pub type NormRect = [f32; 4];

/// Convert a crop rectangle given in *output* (displayed) normalised coordinates into *source*
/// normalised coordinates. Because orientation only flips/swaps axes, the result is still an
/// axis-aligned rectangle; we just map both corners and re-sort.
pub fn crop_to_source(crop: NormRect, o: Orientation) -> NormRect {
    let (ax, ay) = o.map_normalized(crop[0], crop[1]);
    let (bx, by) = o.map_normalized(crop[2], crop[3]);
    [ax.min(bx), ay.min(by), ax.max(bx), ay.max(by)]
}

/// Integer pixel rectangle `[x0, x1) × [y0, y1)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PixelRect {
    pub x0: usize,
    pub y0: usize,
    pub x1: usize,
    pub y1: usize,
}

impl PixelRect {
    pub fn width(&self) -> usize {
        self.x1 - self.x0
    }
    pub fn height(&self) -> usize {
        self.y1 - self.y0
    }

    /// Round a normalised rect to whole pixels of a `w × h` image (always at least 1×1).
    pub fn from_norm(r: NormRect, w: usize, h: usize) -> Self {
        let cx = |v: f32| ((v.clamp(0.0, 1.0) * w as f32).round() as usize).min(w);
        let cy = |v: f32| ((v.clamp(0.0, 1.0) * h as f32).round() as usize).min(h);
        let (x0, x1) = (cx(r[0]), cx(r[2]));
        let (y0, y1) = (cy(r[1]), cy(r[3]));
        let x0 = x0.min(w.saturating_sub(1));
        let y0 = y0.min(h.saturating_sub(1));
        PixelRect { x0, y0, x1: x1.max(x0 + 1), y1: y1.max(y0 + 1) }
    }
}

/// Apply an orientation to an interleaved pixel buffer (`ch` samples per pixel).
/// Returns the new buffer and its dimensions.
pub fn orient_buffer<T: Copy + Default>(src: &[T], w: usize, h: usize, ch: usize, o: Orientation) -> (Vec<T>, usize, usize) {
    if o == Orientation::IDENTITY {
        return (src.to_vec(), w, h);
    }
    let (ow, oh) = o.output_size(w, h);
    let mut out = vec![T::default(); ow * oh * ch];
    for oy in 0..oh {
        let y = if o.flip_y { oh - 1 - oy } else { oy };
        for ox in 0..ow {
            let x = if o.flip_x { ow - 1 - ox } else { ox };
            let (sx, sy) = if o.transpose { (y, x) } else { (x, y) };
            let s = (sy * w + sx) * ch;
            let d = (oy * ow + ox) * ch;
            out[d..d + ch].copy_from_slice(&src[s..s + ch]);
        }
    }
    (out, ow, oh)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exif_roundtrip() {
        for code in 1..=8 {
            assert_eq!(Orientation::from_exif(code).to_exif(), code);
        }
    }

    #[test]
    fn rotate_cw() {
        // 2×1 image [a b] rotated 90° clockwise becomes a 1×2 column [a; b].
        let (out, w, h) = orient_buffer(&[1u8, 2], 2, 1, 1, Orientation::from_exif(6));
        assert_eq!((w, h), (1, 2));
        assert_eq!(out, vec![1, 2]);
        // 1×2 column [a; b] rotated clockwise becomes [b a].
        let (out, _, _) = orient_buffer(&[1u8, 2], 1, 2, 1, Orientation::from_exif(6));
        assert_eq!(out, vec![2, 1]);
    }
}
