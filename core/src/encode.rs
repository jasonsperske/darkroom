//! Encoders for export. Input is always sRGB-encoded 16-bit RGB; 8-bit formats are derived
//! by rounding (65535 → 255 is an exact division by 257).

use std::io::Cursor;

/// 16-bit → 8-bit with correct rounding.
pub fn to_u8(rgb16: &[u16]) -> Vec<u8> {
    rgb16.iter().map(|&v| ((v as u32 * 255 + 32767) / 65535) as u8).collect()
}

/// JPEG. `chroma_444 = true` keeps full colour resolution (no chroma subsampling), which is
/// what you want for an archival-quality export; 4:2:0 halves colour resolution in both axes.
/// `exif` is an optional APP1 payload (starting with `Exif\0\0`) to embed.
pub fn jpeg(rgb16: &[u16], w: usize, h: usize, quality: u8, chroma_444: bool, exif: Option<Vec<u8>>) -> Result<Vec<u8>, String> {
    use jpeg_encoder::{ColorType, Encoder, SamplingFactor};
    if w > 65535 || h > 65535 {
        return Err("JPEG is limited to 65535 pixels per side".into());
    }
    let mut out = Vec::new();
    let mut enc = Encoder::new(&mut out, quality.clamp(1, 100));
    enc.set_sampling_factor(if chroma_444 { SamplingFactor::R_4_4_4 } else { SamplingFactor::R_4_2_0 });
    enc.set_optimized_huffman_tables(true);
    if let Some(e) = exif {
        enc.add_app_segment(1, e).map_err(|e| e.to_string())?;
    }
    enc.encode(&to_u8(rgb16), w as u16, h as u16, ColorType::Rgb).map_err(|e| e.to_string())?;
    Ok(out)
}

/// PNG, 8 or 16 bits per channel. Lossless.
pub fn png(rgb16: &[u16], w: usize, h: usize, bits16: bool) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut out, w as u32, h as u32);
        enc.set_color(png::ColorType::Rgb);
        enc.set_depth(if bits16 { png::BitDepth::Sixteen } else { png::BitDepth::Eight });
        enc.set_compression(png::Compression::Balanced);
        // Tag as sRGB so colour-managed viewers interpret it correctly.
        enc.set_source_srgb(png::SrgbRenderingIntent::Perceptual);
        let mut writer = enc.write_header().map_err(|e| e.to_string())?;
        if bits16 {
            // PNG stores 16-bit samples big-endian.
            let bytes: Vec<u8> = rgb16.iter().flat_map(|v| v.to_be_bytes()).collect();
            writer.write_image_data(&bytes).map_err(|e| e.to_string())?;
        } else {
            writer.write_image_data(&to_u8(rgb16)).map_err(|e| e.to_string())?;
        }
    }
    Ok(out)
}

/// 16-bit RGB TIFF with lossless Deflate compression.
pub fn tiff16(rgb16: &[u16], w: usize, h: usize) -> Result<Vec<u8>, String> {
    use tiff::encoder::{colortype::RGB16, compression::DeflateLevel, Compression, TiffEncoder};
    let mut out = Cursor::new(Vec::new());
    let mut enc = TiffEncoder::new(&mut out)
        .map_err(|e| e.to_string())?
        .with_compression(Compression::Deflate(DeflateLevel::Balanced));
    enc.write_image::<RGB16>(w as u32, h as u32, rgb16).map_err(|e| e.to_string())?;
    Ok(out.into_inner())
}
