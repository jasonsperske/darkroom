//! Decoding: file bytes → a 16-bit RGB "master" image.
//!
//! Every decoder produces the same thing, so the rest of the pipeline never cares where pixels
//! came from:
//!
//! * interleaved `R,G,B` `u16` samples, row-major, no padding;
//! * a [`Transfer`] saying whether the code values are sRGB-encoded or linear light.
//!   We deliberately do **not** linearise 8/16-bit sRGB files at decode time: the develop
//!   lookup table (see `develop::lut`) folds linearisation into the same table lookup as every
//!   other tone operation, so it costs nothing, and the user can override the guess (scanner
//!   software often writes *linear* TIFFs without saying so);
//! * the EXIF-style orientation found in the file (applied later, never baked in).

pub mod demosaic;
pub mod raster;
pub mod raw;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transfer {
    Srgb,
    Linear,
}

pub struct Decoded {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u16>,
    pub transfer: Transfer,
    /// EXIF orientation code 1–8.
    pub orientation: u32,
    /// Human-readable description, e.g. "JPEG 8-bit" or "Raw · NIKON Z 6".
    pub description: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Raster,
    TiffLike,
    Raw,
}

/// Look at magic bytes (and the file extension as a tie-breaker) to pick a decoder.
fn sniff(bytes: &[u8], name: &str) -> Kind {
    let ext = name.rsplit('.').next().unwrap_or("").to_ascii_lowercase();
    const RAW_EXT: &[&str] = &[
        "3fr", "ari", "arw", "cr2", "cr3", "crw", "dcr", "dng", "erf", "fff", "iiq", "kdc", "mef", "mos", "mrw", "nef", "nrw",
        "orf", "ori", "pef", "raf", "raw", "rw2", "rwl", "sr2", "srf", "srw", "x3f",
    ];
    if RAW_EXT.contains(&ext.as_str()) {
        return Kind::Raw;
    }
    match bytes {
        [0xFF, 0xD8, 0xFF, ..] => Kind::Raster,                          // JPEG
        [0x89, b'P', b'N', b'G', ..] => Kind::Raster,                     // PNG
        [b'G', b'I', b'F', b'8', ..] => Kind::Raster,                     // GIF
        [b'B', b'M', ..] => Kind::Raster,                                 // BMP
        [b'R', b'I', b'F', b'F', _, _, _, _, b'W', b'E', b'B', b'P', ..] => Kind::Raster, // WebP
        // TIFF – or a TIFF-based raw. Plain scans go straight to the raster decoder.
        [b'I', b'I', 0x2A, 0x00, ..] | [b'M', b'M', 0x00, 0x2A, ..] => {
            if ext == "tif" || ext == "tiff" || !tiff_looks_like_raw(bytes) {
                Kind::Raster
            } else {
                Kind::TiffLike
            }
        }
        _ => Kind::Raw, // CR3, RAF, ORF, RW2, … have their own magic; let the raw decoder decide
    }
}

/// Does the first TIFF directory carry a camera `Make` (0x010F) or `DNGVersion` (0xC612) tag?
/// Raw decoders assume a camera file and can misbehave badly on arbitrary TIFFs.
fn tiff_looks_like_raw(b: &[u8]) -> bool {
    let le = b[0] == b'I';
    let u16_at = |i: usize| b.get(i..i + 2).map(|s| if le { u16::from_le_bytes([s[0], s[1]]) } else { u16::from_be_bytes([s[0], s[1]]) });
    let u32_at = |i: usize| b.get(i..i + 4).map(|s| {
        let a = [s[0], s[1], s[2], s[3]];
        if le { u32::from_le_bytes(a) } else { u32::from_be_bytes(a) }
    });
    let Some(ifd) = u32_at(4).map(|v| v as usize) else { return false };
    let Some(count) = u16_at(ifd) else { return false };
    (0..count as usize).any(|i| matches!(u16_at(ifd + 2 + i * 12), Some(0x010F) | Some(0xC612)))
}

/// Decode any supported file. `name` is only used as a hint.
pub fn decode(bytes: &[u8], name: &str) -> Result<Decoded, String> {
    match sniff(bytes, name) {
        Kind::Raster => raster::decode(bytes),
        Kind::Raw => raw::decode(bytes),
        // Many raw formats (NEF, CR2, ARW, DNG, …) are TIFF containers whose first image is a
        // small preview. Try the raw decoder first; plain TIFF scans fall through to the
        // raster decoder because the raw decoder can't identify a camera.
        Kind::TiffLike => raw::decode(bytes).or_else(|raw_err| {
            raster::decode(bytes).map_err(|e| format!("{e} (raw decoder: {raw_err})"))
        }),
    }
}
