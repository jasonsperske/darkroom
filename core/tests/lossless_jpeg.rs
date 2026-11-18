//! End-to-end check of the lossless JPEG transcoder: transforming coefficients must give exactly
//! the same pixels as decoding the original and cropping/rotating the pixels.

use darkroom_core::geometry::{orient_buffer, Orientation, PixelRect};
use darkroom_core::jpeg;
use image::ImageDecoder;

/// A synthetic photo-like test image with gradients and detail.
fn test_image(w: usize, h: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let fx = x as f32 / w as f32;
            let fy = y as f32 / h as f32;
            v.push((255.0 * fx) as u8);
            v.push((128.0 + 100.0 * ((x as f32 * 0.21).sin() * (y as f32 * 0.13).cos())) as u8);
            v.push((255.0 * fy * (1.0 - fx * 0.5)) as u8);
        }
    }
    v
}

fn encode(rgb: &[u8], w: usize, h: usize, sampling: jpeg_encoder::SamplingFactor, progressive: bool, restart: Option<u16>) -> Vec<u8> {
    let mut out = Vec::new();
    let mut enc = jpeg_encoder::Encoder::new(&mut out, 85);
    enc.set_sampling_factor(sampling);
    enc.set_progressive(progressive);
    if let Some(r) = restart {
        enc.set_restart_interval(r);
    }
    enc.encode(rgb, w as u16, h as u16, jpeg_encoder::ColorType::Rgb).unwrap();
    out
}

fn decode(bytes: &[u8]) -> (Vec<u8>, usize, usize) {
    let dec = image::codecs::jpeg::JpegDecoder::new(std::io::Cursor::new(bytes)).unwrap();
    let (w, h) = dec.dimensions();
    let img = image::DynamicImage::from_decoder(dec).unwrap().to_rgb8();
    (img.into_raw(), w as usize, h as usize)
}

fn crop_pixels(px: &[u8], w: usize, r: PixelRect) -> Vec<u8> {
    let mut out = Vec::new();
    for y in r.y0..r.y1 {
        out.extend_from_slice(&px[(y * w + r.x0) * 3..(y * w + r.x1) * 3]);
    }
    out
}

fn check(sampling: jpeg_encoder::SamplingFactor, progressive: bool, restart: Option<u16>, exact: bool) {
    let (w, h) = (203, 141); // deliberately not multiples of 8 or 16
    let original = encode(&test_image(w, h), w, h, sampling, progressive, restart);
    let (ref_px, _, _) = decode(&original);
    let requested = PixelRect { x0: 37, y0: 21, x1: 190, y1: 141 };

    for code in 1..=8 {
        let o = Orientation::from_exif(code);
        let out = jpeg::transform(&original, o, requested).unwrap_or_else(|e| panic!("orientation {code}: {e}"));
        let (px, ow, oh) = decode(&out.bytes);
        assert_eq!((ow, oh), (out.width, out.height));
        let r = out.source_rect;
        let (expected, ew, eh) = orient_buffer(&crop_pixels(&ref_px, w, r), r.width(), r.height(), 3, o);
        assert_eq!((ew, eh), (ow, oh), "orientation {code}");

        // With chroma subsampling, decoders smooth chroma using neighbouring blocks, which differ
        // at the new image edges. Allow a small tolerance there; 4:4:4 must be bit-exact.
        let max_diff = px.iter().zip(&expected).map(|(a, b)| (*a as i32 - *b as i32).abs()).max().unwrap();
        // Integer IDCTs round rows and columns differently, so a transposed block can decode to
        // pixels that differ by a level or two even though its coefficients are exact (checked
        // separately in `coefficients_are_exact`).
        if exact && !o.transpose {
            assert_eq!(max_diff, 0, "orientation {code} (sampling {sampling:?}, progressive {progressive})");
        } else {
            let mean: f64 = px.iter().zip(&expected).map(|(a, b)| (*a as f64 - *b as f64).abs()).sum::<f64>() / px.len() as f64;
            assert!(mean < 1.0, "orientation {code}: mean diff {mean}");
        }
    }
}

#[test]
fn baseline_444() {
    check(jpeg_encoder::SamplingFactor::R_4_4_4, false, None, true);
}

#[test]
fn baseline_444_with_restarts() {
    check(jpeg_encoder::SamplingFactor::R_4_4_4, false, Some(3), true);
}

#[test]
fn progressive_444() {
    check(jpeg_encoder::SamplingFactor::R_4_4_4, true, None, true);
}

#[test]
fn baseline_420() {
    check(jpeg_encoder::SamplingFactor::R_4_2_0, false, None, false);
}

#[test]
fn progressive_422() {
    check(jpeg_encoder::SamplingFactor::R_4_2_2, true, Some(5), false);
}

#[test]
fn snapping_follows_the_mcu_grid() {
    let (w, h) = (203, 141);
    let original = encode(&test_image(w, h), w, h, jpeg_encoder::SamplingFactor::R_4_2_0, false, None);
    let info = jpeg::info(&original).unwrap();
    assert_eq!((info.mcu_width, info.mcu_height), (16, 16));
    // Identity: left/top snap down, right/bottom untouched.
    let out = jpeg::transform(&original, Orientation::IDENTITY, PixelRect { x0: 37, y0: 21, x1: 190, y1: 141 }).unwrap();
    assert_eq!(out.source_rect, PixelRect { x0: 32, y0: 16, x1: 190, y1: 141 });
    // Rotating 180° moves the right/bottom edges to the left/top, so the size is whole MCUs.
    let out = jpeg::transform(&original, Orientation::from_exif(3), PixelRect { x0: 37, y0: 21, x1: 190, y1: 141 }).unwrap();
    assert_eq!(out.source_rect, PixelRect { x0: 32, y0: 16, x1: 176, y1: 128 });
}

/// The real losslessness guarantee: the output file's coefficients equal the transformed
/// coefficients of the input, for every orientation, sampling and scan type.
#[test]
fn coefficients_are_exact() {
    let (w, h) = (203, 141);
    for (sampling, progressive) in [
        (jpeg_encoder::SamplingFactor::R_4_4_4, false),
        (jpeg_encoder::SamplingFactor::R_4_2_0, true),
        (jpeg_encoder::SamplingFactor::R_4_2_2, false),
    ] {
        let original = encode(&test_image(w, h), w, h, sampling, progressive, Some(7));
        let src = jpeg::decode_coefficients(&original).unwrap();
        for code in 1..=8 {
            let o = Orientation::from_exif(code);
            let crop = PixelRect { x0: 37, y0: 21, x1: 190, y1: 141 };
            let expected = jpeg::transform_coefficients(&src, o, crop).unwrap();
            let written = jpeg::transform(&original, o, crop).unwrap();
            let reread = jpeg::decode_coefficients(&written.bytes).unwrap();
            assert_eq!(reread.header.width, expected.header.width);
            assert_eq!(reread.header.height, expected.header.height);
            for c in 0..expected.coefs.len() {
                let (rw, rh) = reread.header.real_blocks(c);
                let (bw, _) = reread.header.padded_blocks(c);
                for by in 0..rh {
                    for bx in 0..rw {
                        let i = (by * bw + bx) * 64;
                        assert_eq!(reread.coefs[c][i..i + 64], expected.coefs[c][i..i + 64], "orientation {code} comp {c} block {bx},{by}");
                    }
                }
            }
            assert_eq!(reread.qtables, expected.qtables);
        }
    }
}

/// Identity transform over the whole image must reproduce the original coefficients.
#[test]
fn identity_roundtrip() {
    let (w, h) = (203, 141);
    let original = encode(&test_image(w, h), w, h, jpeg_encoder::SamplingFactor::R_4_2_0, true, None);
    let src = jpeg::decode_coefficients(&original).unwrap();
    let out = jpeg::transform(&original, Orientation::IDENTITY, PixelRect { x0: 0, y0: 0, x1: w, y1: h }).unwrap();
    let back = jpeg::decode_coefficients(&out.bytes).unwrap();
    assert_eq!(src.coefs, back.coefs);
    let (a, _, _) = decode(&original);
    let (b, _, _) = decode(&out.bytes);
    assert_eq!(a, b);
}
