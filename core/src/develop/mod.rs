//! The develop pipeline: how a pixel's code value becomes a display value.
//!
//! ```text
//!  code value ─► linearise ─► [invert negative] ─► white balance ─► exposure
//!     (u16)      (sRGB EOTF)   (density maths)       (×gain/ch)      (×2^EV)
//!
//!            ─► levels ─► contrast S-curve ─► saturation / B&W ─► sRGB encode ─► out
//!               └──────── per channel ─────┘  └─ mixes channels ─┘
//! ```
//!
//! **Key performance idea:** every step up to and including the contrast curve treats the
//! three channels independently. A 16-bit channel has only 65 536 possible inputs, so we
//! evaluate that whole chain *once per possible value* and store it in a lookup table (LUT):
//! 3 × 65 536 floats. Developing a pixel is then three table reads plus a little maths for
//! saturation. The same LUT is uploaded to the GPU for the live preview, so CPU export and GPU
//! preview are guaranteed to agree – there is only one implementation of the tone maths.
//!
//! * `params.rs`   – the user-facing settings (mirrors `web/js/state.js`)
//! * `tone.rs`     – the per-channel maths, one small function per step
//! * `apply.rs`    – running the LUT over pixels (with explicit WebAssembly SIMD)
//! * `analysis.rs` – automatic film base / levels / white balance from image statistics
//! * `render.rs`   – drawing an oriented, cropped, scaled view for the CPU preview & loupe

pub mod analysis;
pub mod apply;
pub mod params;
pub mod render;
pub mod tone;

pub use params::Params;

/// Number of entries per channel in the tone LUT (one per 16-bit code value).
pub const LUT_SIZE: usize = 65536;

/// Everything needed to develop pixels with a fixed set of parameters.
pub struct Developer {
    pub params: Params,
    /// `lut[c * LUT_SIZE + code]` = display-linear value of channel `c`, before saturation.
    pub lut: Vec<f32>,
    pub look: apply::Look,
}

impl Developer {
    pub fn new(params: Params) -> Self {
        let lut = tone::build_lut(&params);
        let look = apply::Look::from_params(&params);
        Developer { params, lut, look }
    }
}
