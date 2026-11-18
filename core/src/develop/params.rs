//! Develop settings. Sent from JavaScript as JSON; field names are camelCase there.
//! Keep in sync with `defaultParams()` in `web/js/state.js`.

use serde::Deserialize;

#[derive(Deserialize, Clone, Debug)]
#[serde(default, rename_all = "camelCase")]
pub struct Params {
    /// Are the stored code values linear light (raw files, "linear" scanner TIFFs)?
    /// Otherwise they are sRGB-encoded.
    pub input_linear: bool,

    // ---- Negative film -----------------------------------------------------------------
    /// Treat the image as a scanned negative and invert it.
    pub negative: bool,
    /// Per-channel density that becomes black: the film base + fog (the clearest film).
    pub dmin: [f32; 3],
    /// Per-channel density that becomes white: the densest highlight.
    pub dmax: [f32; 3],
    /// How many stops of scene brightness the density range is stretched over.
    pub range_stops: f32,

    // ---- Colour & exposure ---------------------------------------------------------------
    /// Neutral white-balance multipliers (set by the grey picker). Green is normally 1.
    pub wb: [f32; 3],
    /// Warm/cool shift, roughly −1…1. Positive = warmer (more red, less blue).
    pub temperature: f32,
    /// Green/magenta shift, roughly −1…1. Positive = more magenta.
    pub tint: f32,
    /// Exposure compensation in stops (EV).
    pub exposure: f32,

    // ---- Tone ----------------------------------------------------------------------------
    /// Black and white points, in perceptual (gamma-encoded) units 0…1.
    pub black: f32,
    pub white: f32,
    /// Contrast: 1 = unchanged, >1 S-curve, <1 flatter. Like choosing a paper grade.
    pub contrast: f32,

    // ---- Look ----------------------------------------------------------------------------
    pub saturation: f32,
    pub monochrome: bool,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            input_linear: false,
            negative: false,
            dmin: [0.0; 3],
            dmax: [1.5; 3],
            range_stops: 6.0,
            wb: [1.0; 3],
            temperature: 0.0,
            tint: 0.0,
            exposure: 0.0,
            black: 0.0,
            white: 1.0,
            contrast: 1.0,
            saturation: 1.0,
            monochrome: false,
        }
    }
}

impl Params {
    pub fn from_json(json: &str) -> Result<Self, String> {
        serde_json::from_str(json).map_err(|e| format!("bad params: {e}"))
    }
}
