//! Camera raw → linear sRGB, step by step.
//!
//! A raw file holds what the sensor measured: one number per photosite, each behind a red,
//! green or blue filter (the colour filter array, CFA). Turning that into an RGB image takes
//! five steps, each a few lines below:
//!
//! 1. **Black & white level** – subtract the sensor's black offset and divide by the
//!    saturation level so values run 0…1 and are proportional to light.
//! 2. **White balance** – multiply each filter colour by the camera's "as shot" coefficients
//!    so a neutral grey reads R = G = B.
//! 3. **Highlight clip** – clamp to 1 so blown highlights stay white instead of turning pink.
//! 4. **Demosaic** – estimate the two missing colours at every photosite (`demosaic.rs`).
//! 5. **Colour matrix** – convert from the camera's own RGB to standard sRGB primaries using
//!    the per-model matrix shipped with the decoder (originally measured by Adobe / dcraw).
//!
//! The decoding of the proprietary container and compression is done by the `rawler` crate.

use super::{demosaic, Decoded, Transfer};
use crate::color::{mat3_apply, mat3_inverse, mat3_mul, Mat3, SRGB_TO_XYZ};
use rawler::decoders::RawDecodeParams;
use rawler::rawimage::RawPhotometricInterpretation;
use rawler::rawsource::RawSource;
use rawler::{RawImage, RawImageData};

pub fn decode(bytes: &[u8]) -> Result<Decoded, String> {
    let src = RawSource::new_from_slice(bytes);
    let decoder = rawler::get_decoder(&src).map_err(|e| e.to_string())?;
    let raw = decoder.raw_image(&src, &RawDecodeParams::default(), false).map_err(|e| e.to_string())?;
    develop(&raw)
}

fn develop(raw: &RawImage) -> Result<Decoded, String> {
    // The usable image area (sensors have masked borders used for black level measurement).
    let area = raw.crop_area.or(raw.active_area);
    let (cx, cy, w, h) = match area {
        Some(r) => (r.p.x, r.p.y, r.d.w, r.d.h),
        None => (0, 0, raw.width, raw.height),
    };
    if w == 0 || h == 0 || cx + w > raw.width || cy + h > raw.height {
        return Err("raw image has an invalid crop area".into());
    }

    // Read one raw sample as f32 regardless of storage type.
    let data = &raw.data;
    let stride = raw.width * raw.cpp;
    let sample = |x: usize, y: usize, c: usize| -> f32 {
        let i = y * stride + x * raw.cpp + c;
        match data {
            RawImageData::Integer(v) => v[i] as f32,
            RawImageData::Float(v) => v[i],
        }
    };

    // ---- Step 1: black / white levels (they may vary with position in a small repeat pattern).
    let bl = &raw.blacklevel;
    let black_levels: Vec<f32> = bl.levels.iter().map(|r| r.as_f32()).collect();
    let black_at = |x: usize, y: usize, c: usize| -> f32 {
        let (bw, bh) = (bl.width.max(1), bl.height.max(1));
        let i = ((y % bh) * bw + (x % bw)) * bl.cpp.max(1) + c.min(bl.cpp.max(1) - 1);
        black_levels.get(i).copied().unwrap_or(0.0)
    };
    let white = raw.whitelevel.0.first().copied().unwrap_or(65535) as f32;

    // ---- Step 2: white balance multipliers, normalised to green = 1.
    let wb = {
        let c = raw.wb_coeffs;
        let valid = c[0].is_finite() && c[1].is_finite() && c[2].is_finite() && c[0] > 0.0 && c[1] > 0.0 && c[2] > 0.0;
        let c = if valid { c } else { raw.neutralwb() };
        [c[0] / c[1], 1.0, c[2] / c[1]]
    };

    // ---- Step 5 (prepared now): camera RGB → linear sRGB matrix.
    let cam_to_srgb = camera_matrix(raw);

    let mut rgb = vec![0u16; w * h * 3];
    let to_u16 = |v: f32| (v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;

    match &raw.photometric {
        RawPhotometricInterpretation::Cfa(config) => {
            if raw.cpp != 1 {
                return Err("unsupported CFA layout".into());
            }
            // The CFA pattern is defined from the sensor origin; shift it to our crop origin.
            let cfa = config.cfa.shift(cx, cy);
            let pattern_w = cfa.width.max(1);
            let pattern_h = cfa.height.max(1);
            let mut colors = vec![0u8; pattern_w * pattern_h];
            for py in 0..pattern_h {
                for px in 0..pattern_w {
                    let c = cfa.color_at(py, px);
                    if c > 2 {
                        return Err(format!("unsupported sensor colour filter ({})", cfa.name));
                    }
                    colors[py * pattern_w + px] = c as u8;
                }
            }
            // Steps 1–3 on the mosaic: one f32 per photosite.
            let mut mosaic = vec![0f32; w * h];
            for y in 0..h {
                for x in 0..w {
                    let (sx, sy) = (x + cx, y + cy);
                    let black = black_at(sx, sy, 0);
                    let v = (sample(sx, sy, 0) - black) / (white - black);
                    let c = colors[(y % pattern_h) * pattern_w + (x % pattern_w)] as usize;
                    mosaic[y * w + x] = (v * wb[c]).clamp(0.0, 1.0);
                }
            }
            // Step 4.
            let pattern = demosaic::Pattern { width: pattern_w, height: pattern_h, colors };
            let full = demosaic::demosaic(&mosaic, w, h, &pattern);
            drop(mosaic);
            // Step 5.
            for (dst, px) in rgb.chunks_exact_mut(3).zip(full.chunks_exact(3)) {
                let o = mat3_apply(&cam_to_srgb, [px[0], px[1], px[2]]);
                dst.copy_from_slice(&[to_u16(o[0]), to_u16(o[1]), to_u16(o[2])]);
            }
        }
        RawPhotometricInterpretation::LinearRaw | RawPhotometricInterpretation::BlackIsZero => {
            let mono = raw.cpp == 1;
            for y in 0..h {
                for x in 0..w {
                    let (sx, sy) = (x + cx, y + cy);
                    let mut px = [0f32; 3];
                    for c in 0..3 {
                        let sc = if mono { 0 } else { c.min(raw.cpp - 1) };
                        let black = black_at(sx, sy, sc);
                        let v = (sample(sx, sy, sc) - black) / (white - black);
                        px[c] = if mono { v } else { (v * wb[c]).clamp(0.0, 1.0) };
                    }
                    let o = if mono { px } else { mat3_apply(&cam_to_srgb, px) };
                    let d = (y * w + x) * 3;
                    rgb[d..d + 3].copy_from_slice(&[to_u16(o[0]), to_u16(o[1]), to_u16(o[2])]);
                }
            }
        }
    }

    let orientation = match raw.orientation.to_u16() {
        c @ 1..=8 => c as u32,
        _ => 1,
    };
    Ok(Decoded {
        width: w,
        height: h,
        rgb,
        transfer: Transfer::Linear,
        orientation,
        description: format!("Raw · {} {}", raw.clean_make, raw.clean_model),
    })
}

/// Build the camera-RGB → linear-sRGB matrix, the same way dcraw does:
///
/// ```text
/// rgb_to_cam = xyz_to_cam · srgb_to_xyz        (how the camera sees each sRGB primary)
/// normalise each row to sum to 1               (so white-balanced white stays white)
/// cam_to_rgb = inverse(rgb_to_cam)
/// ```
fn camera_matrix(raw: &RawImage) -> Mat3 {
    const IDENTITY: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    use rawler::imgop::xyz::Illuminant;
    let Some((_, flat)) = raw
        .color_matrix_find_first([Illuminant::D65, Illuminant::Daylight, Illuminant::D55, Illuminant::D50])
        .or_else(|| raw.color_matrix.iter().next().map(|(k, v)| (*k, v.clone())))
    else {
        return IDENTITY;
    };
    if flat.len() < 9 {
        return IDENTITY;
    }
    let xyz_to_cam: Mat3 = [[flat[0], flat[1], flat[2]], [flat[3], flat[4], flat[5]], [flat[6], flat[7], flat[8]]];
    let mut rgb_to_cam = mat3_mul(&xyz_to_cam, &SRGB_TO_XYZ);
    for row in rgb_to_cam.iter_mut() {
        let sum: f32 = row.iter().sum();
        if sum.abs() > 1e-6 {
            row.iter_mut().for_each(|v| *v /= sum);
        }
    }
    mat3_inverse(&rgb_to_cam).unwrap_or(IDENTITY)
}
