//! # Darkroom core
//!
//! All pixel work for Darkroom, compiled to WebAssembly. This file is the *only* place that
//! knows about JavaScript: it exposes small, flat functions (byte/number arrays in, arrays out)
//! via `wasm-bindgen`. Everything else is plain Rust with unit tests (`cargo test`).
//!
//! | module      | what it teaches                                                   |
//! |-------------|-------------------------------------------------------------------|
//! | `decode`    | file formats → 16-bit RGB; camera raw development & demosaicing   |
//! | `develop`   | negative inversion, white balance, tone curves, LUTs, SIMD        |
//! | `geometry`  | rotation/flip as three booleans; crop rectangles                  |
//! | `resize`    | area-average downscaling                                          |
//! | `encode`    | JPEG / PNG / TIFF writers                                         |
//! | `jpeg`      | lossless crop & rotate by editing DCT coefficients directly       |
//! | `color`     | sRGB transfer function and colour matrices                        |
//! | `denoise`   | film grain model, wavelet, non-local means, CNN denoiser          |
//!
//! Pixel buffers crossing the boundary are interleaved `u16` (RGB, or RGBA for the proxy),
//! row-major, with no padding.

pub mod color;
pub mod decode;
pub mod denoise;
pub mod develop;
pub mod encode;
pub mod geometry;
pub mod jpeg;
pub mod resize;

use develop::{analysis, render, Developer, Params};
use geometry::{NormRect, Orientation, PixelRect};
use wasm_bindgen::prelude::*;

fn err(e: impl std::fmt::Display) -> JsError {
    JsError::new(&e.to_string())
}

fn rect(r: &[f32]) -> NormRect {
    [r[0], r[1], r[2], r[3]]
}

/// Runs once when the module is instantiated.
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

/// Whether this build uses WebAssembly SIMD (shown in the UI's "about" line).
#[wasm_bindgen]
pub fn simd_enabled() -> bool {
    cfg!(target_feature = "simd128")
}

// ---- Decoding ----------------------------------------------------------------------------

#[wasm_bindgen]
pub struct DecodedImage {
    width: usize,
    height: usize,
    orientation: u32,
    linear: bool,
    description: String,
    rgb: Vec<u16>,
}

#[wasm_bindgen]
impl DecodedImage {
    #[wasm_bindgen(getter)]
    pub fn width(&self) -> usize {
        self.width
    }
    #[wasm_bindgen(getter)]
    pub fn height(&self) -> usize {
        self.height
    }
    #[wasm_bindgen(getter)]
    pub fn orientation(&self) -> u32 {
        self.orientation
    }
    #[wasm_bindgen(getter)]
    pub fn linear(&self) -> bool {
        self.linear
    }
    #[wasm_bindgen(getter)]
    pub fn description(&self) -> String {
        self.description.clone()
    }
    /// Move the pixels out (copied into a JS `Uint16Array`; the wasm copy is freed).
    pub fn take_rgb(&mut self) -> Vec<u16> {
        std::mem::take(&mut self.rgb)
    }
}

/// Decode any supported file to 16-bit RGB.
#[wasm_bindgen]
pub fn decode(bytes: &[u8], name: &str) -> Result<DecodedImage, JsError> {
    let d = decode::decode(bytes, name).map_err(err)?;
    Ok(DecodedImage {
        width: d.width,
        height: d.height,
        orientation: d.orientation,
        linear: d.transfer == decode::Transfer::Linear,
        description: d.description,
        rgb: d.rgb,
    })
}

/// Convert browser-decoded RGBA8 (used for formats we don't decode ourselves, e.g. AVIF/HEIC).
#[wasm_bindgen]
pub fn rgba8_to_rgb16(rgba: &[u8]) -> Vec<u16> {
    rgba.chunks_exact(4).flat_map(|p| [p[0] as u16 * 257, p[1] as u16 * 257, p[2] as u16 * 257]).collect()
}

// ---- Proxy & resizing ----------------------------------------------------------------------

/// Downscale RGB16 to fit `max_edge`, returning **RGBA16** (alpha = 65535): the on-screen proxy.
/// RGBA because GPUs have no 3-channel texture formats.
#[wasm_bindgen]
pub fn make_proxy(rgb: &[u16], w: usize, h: usize, pw: usize, ph: usize) -> Vec<u16> {
    let small = resize::downscale_u16(rgb, w, h, 3, pw, ph);
    small.chunks_exact(3).flat_map(|p| [p[0], p[1], p[2], 65535]).collect()
}

#[wasm_bindgen]
pub fn resize_rgb16(rgb: &[u16], w: usize, h: usize, dw: usize, dh: usize) -> Vec<u16> {
    resize::downscale_u16(rgb, w, h, 3, dw, dh)
}

#[wasm_bindgen]
pub fn orient_rgb16(rgb: &[u16], w: usize, h: usize, exif_orientation: u32) -> Vec<u16> {
    geometry::orient_buffer(rgb, w, h, 3, Orientation::from_exif(exif_orientation)).0
}

// ---- Developing ----------------------------------------------------------------------------

/// A developer bound to one set of parameters (builds the tone LUT once).
#[wasm_bindgen(js_name = Developer)]
pub struct WasmDeveloper(Developer);

#[wasm_bindgen(js_class = Developer)]
impl WasmDeveloper {
    #[wasm_bindgen(constructor)]
    pub fn new(params_json: &str) -> Result<WasmDeveloper, JsError> {
        Ok(WasmDeveloper(Developer::new(Params::from_json(params_json).map_err(err)?)))
    }

    /// The 3 × 65536 tone LUT, for upload to the GPU.
    pub fn lut(&self) -> Vec<f32> {
        self.0.lut.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn saturation(&self) -> f32 {
        self.0.look.saturation
    }

    /// Develop an RGB16 strip to sRGB-encoded RGB16 (export path).
    pub fn process_rgb16(&self, rgb: &[u16]) -> Vec<u16> {
        self.0.process_rgb16(rgb)
    }

    /// Render an oriented, cropped view as RGBA8 (CPU preview / loupe).
    #[allow(clippy::too_many_arguments)]
    pub fn render_view(&self, src: &[u16], sw: usize, sh: usize, ch: usize, exif_orientation: u32, crop: &[f32], out_w: usize, out_h: usize) -> Vec<u8> {
        render::render_view(&self.0, src, sw, sh, ch, Orientation::from_exif(exif_orientation), rect(crop), out_w, out_h)
    }
}

#[wasm_bindgen]
pub fn histogram_rgba8(px: &[u8]) -> Vec<u32> {
    render::histogram_rgba8(px)
}

// ---- Automatic analysis (on the proxy, rect in source-normalised coordinates) ---------------

#[wasm_bindgen]
pub fn auto_negative(params_json: &str, src: &[u16], w: usize, h: usize, ch: usize, r: &[f32]) -> Result<Vec<f32>, JsError> {
    let p = Params::from_json(params_json).map_err(err)?;
    Ok(analysis::auto_negative(&p, src, w, h, ch, rect(r)).to_vec())
}

#[wasm_bindgen]
pub fn sample_density(params_json: &str, src: &[u16], w: usize, h: usize, ch: usize, r: &[f32]) -> Result<Vec<f32>, JsError> {
    let p = Params::from_json(params_json).map_err(err)?;
    Ok(analysis::sample_density(&p, src, w, h, ch, rect(r)).to_vec())
}

#[wasm_bindgen]
pub fn neutral_balance(params_json: &str, src: &[u16], w: usize, h: usize, ch: usize, r: &[f32]) -> Result<Vec<f32>, JsError> {
    let p = Params::from_json(params_json).map_err(err)?;
    Ok(analysis::neutral_balance(&p, src, w, h, ch, rect(r)).to_vec())
}

#[wasm_bindgen]
pub fn auto_levels(params_json: &str, src: &[u16], w: usize, h: usize, ch: usize, r: &[f32]) -> Result<Vec<f32>, JsError> {
    let p = Params::from_json(params_json).map_err(err)?;
    Ok(analysis::auto_levels(&p, src, w, h, ch, rect(r)).to_vec())
}

// ---- Encoding ------------------------------------------------------------------------------

/// `exif_source`: the original file if it was a JPEG (its EXIF is copied with orientation
/// reset to 1), or an empty array.
#[wasm_bindgen]
pub fn encode_jpeg(rgb16: &[u16], w: usize, h: usize, quality: u8, chroma_444: bool, exif_source: &[u8]) -> Result<Vec<u8>, JsError> {
    let exif = jpeg::exif::find_app1(exif_source).map(|a| {
        let mut a = a.to_vec();
        jpeg::exif::set_orientation(&mut a, 1);
        a
    });
    encode::jpeg(rgb16, w, h, quality, chroma_444, exif).map_err(err)
}

#[wasm_bindgen]
pub fn encode_png(rgb16: &[u16], w: usize, h: usize, bits16: bool) -> Result<Vec<u8>, JsError> {
    encode::png(rgb16, w, h, bits16).map_err(err)
}

#[wasm_bindgen]
pub fn encode_tiff(rgb16: &[u16], w: usize, h: usize) -> Result<Vec<u8>, JsError> {
    encode::tiff16(rgb16, w, h).map_err(err)
}

// ---- Lossless JPEG ---------------------------------------------------------------------------

/// JSON with size, MCU grid and orientation, or an error explaining why lossless editing is
/// impossible (e.g. arithmetic coding, 12-bit).
#[wasm_bindgen]
pub fn jpeg_info(bytes: &[u8]) -> Result<String, JsError> {
    let info = jpeg::info(bytes).map_err(err)?;
    serde_json::to_string(&info).map_err(err)
}

#[wasm_bindgen]
pub struct LosslessJpeg {
    bytes: Vec<u8>,
    width: usize,
    height: usize,
}

#[wasm_bindgen]
impl LosslessJpeg {
    pub fn take_bytes(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.bytes)
    }
    #[wasm_bindgen(getter)]
    pub fn width(&self) -> usize {
        self.width
    }
    #[wasm_bindgen(getter)]
    pub fn height(&self) -> usize {
        self.height
    }
}

/// Crop (source pixels `[x0, x1) × [y0, y1)`) then orient, without recompressing.
#[wasm_bindgen]
pub fn jpeg_lossless(bytes: &[u8], exif_orientation: u32, x0: usize, y0: usize, x1: usize, y1: usize) -> Result<LosslessJpeg, JsError> {
    let out = jpeg::transform(bytes, Orientation::from_exif(exif_orientation), PixelRect { x0, y0, x1, y1 }).map_err(err)?;
    Ok(LosslessJpeg { bytes: out.bytes, width: out.width, height: out.height })
}

// ---- Denoising -------------------------------------------------------------------------------

/// Measure noise/grain. Returns `[σ_luma, σ_chroma, grain size]` followed by the measured
/// noise of each wavelet layer (5 luma, then 5 chroma). `negative` selects density space.
#[wasm_bindgen]
pub fn estimate_noise(rgb: &[u16], w: usize, h: usize, linear: bool, negative: bool) -> Vec<f32> {
    let p = denoise::estimate_noise(rgb, w, h, linear, denoise::Domain::for_negative(negative));
    let mut v = vec![p.luma, p.chroma, p.size];
    if let Some((l, c)) = p.measured {
        v.extend_from_slice(&l);
        v.extend_from_slice(&c);
    }
    v
}

/// Context (in pixels) a tile needs on each side for the given method.
#[wasm_bindgen]
pub fn denoise_margin(params_json: &str) -> Result<usize, JsError> {
    Ok(denoise::margin(denoise::DenoiseParams::from_json(params_json).map_err(err)?.method))
}

/// Holds the (optional) neural network weights between calls.
#[wasm_bindgen]
pub struct Denoiser {
    net: Option<denoise::neural::Network>,
}

#[wasm_bindgen]
impl Denoiser {
    /// `weights`: contents of `models/denoise.bin`, or an empty array for classic methods only.
    #[wasm_bindgen(constructor)]
    pub fn new(weights: &[u8]) -> Result<Denoiser, JsError> {
        let net = if weights.is_empty() { None } else { Some(denoise::neural::Network::parse(weights).map_err(err)?) };
        Ok(Denoiser { net })
    }

    /// Denoise interleaved RGB16 with an explicit noise profile (`[σ_luma, σ_chroma, size]`).
    pub fn run(&self, rgb: &[u16], w: usize, h: usize, linear: bool, params_json: &str, noise: &[f32]) -> Result<Vec<u16>, JsError> {
        let params = denoise::DenoiseParams::from_json(params_json).map_err(err)?;
        denoise::denoise_rgb16(rgb, w, h, linear, &params, profile(noise), self.net.as_ref()).map_err(err)
    }
}

/// Inverse of the `estimate_noise` layout. Pass only the first three values to use the pure
/// model (e.g. after the user overrides the grain size).
fn profile(n: &[f32]) -> denoise::NoiseProfile {
    const L: usize = denoise::wavelet::LEVELS;
    let mut p = denoise::NoiseProfile::new(n[0], n[1], n[2]);
    if n.len() >= 3 + 2 * L {
        p.measured = Some((std::array::from_fn(|i| n[3 + i]), std::array::from_fn(|i| n[3 + L + i])));
    }
    p
}

/// Neural denoising, step 1 of 3 for the WebGPU path: RGB16 → network input tensor
/// (15 × ⌈h/2⌉ × ⌈w/2⌉, channel-major). `noise` should already include the strength sliders.
#[wasm_bindgen]
pub fn nn_prepare(rgb: &[u16], w: usize, h: usize, linear: bool, negative: bool, noise: &[f32]) -> Vec<f32> {
    let planes = denoise::Planes::from_rgb16(rgb, w, h, linear, denoise::Domain::for_negative(negative));
    denoise::neural::prepare(&planes.p, w, h, profile(noise)).data
}

/// Step 3 of 3: subtract the predicted noise (12 × ⌈h/2⌉ × ⌈w/2⌉) and return RGB16.
#[wasm_bindgen]
pub fn nn_finish(rgb: &[u16], w: usize, h: usize, linear: bool, negative: bool, noise: &[f32]) -> Vec<u16> {
    let domain = denoise::Domain::for_negative(negative);
    let mut planes = denoise::Planes::from_rgb16(rgb, w, h, linear, domain);
    let t = denoise::neural::Tensor { c: denoise::neural::OUT_CHANNELS, h: h.div_ceil(2), w: w.div_ceil(2), data: noise.to_vec() };
    denoise::neural::finish(&mut planes.p, w, h, &t);
    planes.to_rgb16(linear, domain)
}
