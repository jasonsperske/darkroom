//! Lossless JPEG crop & rotate — working on DCT coefficients, never on pixels.
//!
//! ## Why re-saving a JPEG loses quality
//!
//! A JPEG stores each 8×8 pixel block as 64 *quantised* DCT (frequency) coefficients. Decoding
//! multiplies by the quantisation table and runs an inverse DCT; encoding does the reverse and
//! rounds again. Every decode→encode round trip rounds again, so quality drifts down.
//!
//! ## How to avoid it
//!
//! If we never leave the coefficient domain, nothing is rounded:
//!
//! * **Crop**: keep a subset of whole blocks. The crop's top-left must sit on the *MCU* grid
//!   (minimum coded unit: 8×8 px, or 16×16 / 16×8 px when colour is subsampled), so edges snap
//!   to that grid. The right/bottom edge can be anywhere: the image size in the header simply
//!   says where to stop drawing.
//! * **Rotate / flip**: a mirrored block's DCT is the original with the sign of every odd
//!   horizontal (or vertical) frequency flipped; a transposed block's DCT is the transposed
//!   coefficient matrix. Rotations are combinations of those (see `transform.rs`).
//!
//! Only the entropy coding (Huffman) is redone, using tables optimised for the new data.
//! This is the same idea as `jpegtran` from libjpeg.
//!
//! ## Files in this module (in reading order)
//!
//! * `bitio.rs`     – reading/writing bit streams with JPEG's `0xFF 0x00` byte stuffing
//! * `huffman.rs`   – Huffman decoding tables and building optimal encoding tables
//! * `parse.rs`     – marker parsing and entropy decoding of baseline *and* progressive scans
//! * `transform.rs` – the crop/rotate maths on coefficient blocks
//! * `write.rs`     – writing a new baseline JPEG
//! * `exif.rs`      – reading/resetting the EXIF orientation tag

mod bitio;
pub mod exif;
mod huffman;
pub mod parse;
mod transform;
mod write;

use crate::geometry::{Orientation, PixelRect};
pub use parse::JpegFile;

/// Decode a file down to its quantised DCT coefficients (exposed for tests and the curious).
pub fn decode_coefficients(bytes: &[u8]) -> Result<JpegFile, String> {
    parse::decode(bytes)
}

/// The coefficient-domain crop + orientation, without writing a file.
pub fn transform_coefficients(file: &JpegFile, orient: Orientation, crop: PixelRect) -> Result<JpegFile, String> {
    let rect = transform::snap_crop(&file.header, orient, crop)?;
    Ok(transform::apply(file, orient, rect))
}

/// Natural (row-major) index of the k-th coefficient in zig-zag order.
pub(crate) const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42,
    49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// Basic facts the UI needs to offer lossless export.
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Info {
    pub width: usize,
    pub height: usize,
    /// MCU size in pixels (the crop grid).
    pub mcu_width: usize,
    pub mcu_height: usize,
    pub progressive: bool,
    pub components: usize,
    pub orientation: u32,
}

pub fn info(bytes: &[u8]) -> Result<Info, String> {
    let header = parse::parse_header(bytes)?;
    Ok(Info {
        width: header.width,
        height: header.height,
        mcu_width: header.mcu_width(),
        mcu_height: header.mcu_height(),
        progressive: header.progressive,
        components: header.components.len(),
        orientation: exif::find_orientation(bytes).unwrap_or(1),
    })
}

/// Result of a lossless transform.
pub struct Output {
    pub bytes: Vec<u8>,
    pub width: usize,
    pub height: usize,
    /// The crop actually used, in source pixels, after snapping to the MCU grid.
    pub source_rect: PixelRect,
}

/// Crop (in *source* pixel coordinates) and then orient a JPEG without recompression.
pub fn transform(bytes: &[u8], orient: Orientation, crop: PixelRect) -> Result<Output, String> {
    let file = parse::decode(bytes)?;
    let rect = transform::snap_crop(&file.header, orient, crop)?;
    let out = transform::apply(&file, orient, rect);
    let (width, height) = (out.header.width, out.header.height);
    let bytes = write::write(&out);
    Ok(Output { bytes, width, height, source_rect: rect })
}
