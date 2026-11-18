//! Minimal EXIF access: find and rewrite the Orientation tag (0x0112) in IFD0.
//!
//! EXIF data is a little TIFF file inside the JPEG's APP1 segment:
//!
//! ```text
//! "Exif\0\0" | "II" or "MM" (byte order) | 42 | offset of IFD0 | …
//! IFD = count:u16, then count × 12-byte entries (tag:u16, type:u16, n:u32, value/offset:u32)
//! ```
//! A SHORT value such as Orientation is stored inline in the first 2 bytes of the value field.

const ORIENTATION: u16 = 0x0112;

/// Locate the Orientation entry's value position within a TIFF block.
fn find_entry(tiff: &[u8]) -> Option<(usize, bool)> {
    let le = match tiff.get(0..2)? {
        b"II" => true,
        b"MM" => false,
        _ => return None,
    };
    let u16_at = |i: usize| -> Option<u16> {
        let b = [*tiff.get(i)?, *tiff.get(i + 1)?];
        Some(if le { u16::from_le_bytes(b) } else { u16::from_be_bytes(b) })
    };
    let u32_at = |i: usize| -> Option<u32> {
        let b = [*tiff.get(i)?, *tiff.get(i + 1)?, *tiff.get(i + 2)?, *tiff.get(i + 3)?];
        Some(if le { u32::from_le_bytes(b) } else { u32::from_be_bytes(b) })
    };
    let ifd = u32_at(4)? as usize;
    let count = u16_at(ifd)? as usize;
    for i in 0..count {
        let entry = ifd + 2 + i * 12;
        if u16_at(entry)? == ORIENTATION {
            return Some((entry + 8, le));
        }
    }
    None
}

/// Read the orientation from an APP1 payload starting with `Exif\0\0`.
pub fn read_orientation(app1: &[u8]) -> Option<u32> {
    let tiff = app1.strip_prefix(b"Exif\0\0")?;
    let (pos, le) = find_entry(tiff)?;
    let b = [*tiff.get(pos)?, *tiff.get(pos + 1)?];
    let v = if le { u16::from_le_bytes(b) } else { u16::from_be_bytes(b) } as u32;
    (1..=8).contains(&v).then_some(v)
}

/// Overwrite the orientation in an APP1 payload, if present.
pub fn set_orientation(app1: &mut [u8], value: u16) {
    let Some(tiff) = app1.get_mut(6..) else { return };
    if let Some((pos, le)) = find_entry(tiff) {
        let b = if le { value.to_le_bytes() } else { value.to_be_bytes() };
        tiff[pos..pos + 2].copy_from_slice(&b);
    }
}

/// Find the EXIF APP1 payload of a JPEG file (without parsing the image).
pub fn find_app1(jpeg: &[u8]) -> Option<&[u8]> {
    let mut pos = 2;
    while pos + 4 <= jpeg.len() && jpeg[pos] == 0xFF {
        let marker = jpeg[pos + 1];
        if marker == 0xDA || marker == 0xD9 {
            break;
        }
        let len = u16::from_be_bytes([jpeg[pos + 2], jpeg[pos + 3]]) as usize;
        let payload = jpeg.get(pos + 4..pos + 2 + len)?;
        if marker == 0xE1 && payload.starts_with(b"Exif\0\0") {
            return Some(payload);
        }
        pos += 2 + len;
    }
    None
}

pub fn find_orientation(jpeg: &[u8]) -> Option<u32> {
    read_orientation(find_app1(jpeg)?)
}
