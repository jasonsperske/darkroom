//! Ordinary raster formats (JPEG, PNG, TIFF, WebP, BMP, GIF) via the `image` crate.
//!
//! The interesting part is keeping precision: a 16-bit TIFF from a film scanner holds 256× more
//! tonal steps than an 8-bit JPEG, and negative inversion stretches the tonal range a lot, so
//! throwing those bits away would cause visible banding.

use super::{Decoded, Transfer};
use image::{DynamicImage, ImageDecoder, ImageReader};
use std::io::Cursor;

pub fn decode(bytes: &[u8]) -> Result<Decoded, String> {
    let reader = ImageReader::new(Cursor::new(bytes)).with_guessed_format().map_err(|e| e.to_string())?;
    let format = reader.format().map(|f| format!("{f:?}").to_uppercase()).unwrap_or_else(|| "image".into());
    let mut decoder = reader.into_decoder().map_err(|e| e.to_string())?;
    // Orientation is reported, not applied: rotation is part of the editable state.
    let orientation = decoder.orientation().map(|o| o.to_exif() as u32).unwrap_or(1);
    let img = DynamicImage::from_decoder(decoder).map_err(|e| e.to_string())?;
    let (width, height) = (img.width() as usize, img.height() as usize);

    use DynamicImage::*;
    let (rgb, transfer, bits): (Vec<u16>, Transfer, u32) = match img {
        // High bit depth integer: keep all 16 bits.
        ImageLuma16(_) | ImageLumaA16(_) | ImageRgb16(_) | ImageRgba16(_) => (img.to_rgb16().into_raw(), Transfer::Srgb, 16),
        // Floating point (e.g. 32-bit float TIFF) is conventionally scene-linear.
        ImageRgb32F(_) | ImageRgba32F(_) => {
            let f = img.to_rgb32f().into_raw();
            (f.iter().map(|&v| (v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16).collect(), Transfer::Linear, 32)
        }
        // 8-bit: widen with ×257 so 255 maps exactly to 65535.
        _ => (img.to_rgb8().into_raw().iter().map(|&v| v as u16 * 257).collect(), Transfer::Srgb, 8),
    };

    Ok(Decoded { width, height, rgb, transfer, orientation, description: format!("{format} {bits}-bit") })
}
