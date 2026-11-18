//! Per-channel tone maths. Each step is a tiny pure function so it can be read (and unit
//! tested) on its own; [`channel_curve`] chains them and [`build_lut`] tabulates the chain.

use super::{Params, LUT_SIZE};
use crate::color::srgb_to_linear;

/// Step 1 – code value (0…1) to linear light.
#[inline]
pub fn linearize(p: &Params, code: f32) -> f32 {
    if p.input_linear {
        code
    } else {
        srgb_to_linear(code)
    }
}

/// Optical density of a transmittance: `D = −log10(T)`.
///
/// A scanner (or a camera over a light table) measures how much light passes through the film.
/// Density is the logarithm of how much the film *blocks*: D = 0 is clear, D = 1 passes 10 %,
/// D = 2 passes 1 %. Film responds to light logarithmically, so density is the natural unit.
#[inline]
pub fn density(transmittance: f32) -> f32 {
    -(transmittance.max(1e-6)).log10()
}

/// Step 2 – invert a negative.
///
/// 1. Convert the channel to density.
/// 2. Normalise between `dmin` (film base + fog → black) and `dmax` (densest highlight → white).
///    Doing this *per channel* removes the orange mask and balances the three dye layers,
///    whose contrasts differ – this is the core of colour negative conversion.
/// 3. On the straight part of film's characteristic curve, density is proportional to the
///    logarithm of exposure. So normalised density `n` is a *log* brightness: we turn it back
///    into linear light spread over `range_stops` stops, with `n = 0` → black, `n = 1` → white.
#[inline]
pub fn invert_negative(p: &Params, c: usize, linear: f32) -> f32 {
    let d = density(linear);
    let span = (p.dmax[c] - p.dmin[c]).max(0.01);
    let n = (d - p.dmin[c]) / span;
    let r = p.range_stops.max(0.5);
    let floor = (-r).exp2();
    (((n - 1.0) * r).exp2() - floor) / (1.0 - floor)
}

/// Step 3 – white balance and exposure as a single per-channel gain.
pub fn channel_gains(p: &Params) -> [f32; 3] {
    let t = p.temperature * 0.5;
    let m = p.tint * 0.5;
    let ev = p.exposure.exp2();
    [
        p.wb[0] * (t + m * 0.5).exp2() * ev,
        p.wb[1] * (-m).exp2() * ev,
        p.wb[2] * (-t + m * 0.5).exp2() * ev,
    ]
}

/// Perceptual encoding used by the tone controls (a plain 2.2 power, close to how we see).
const GAMMA: f32 = 2.2;

/// Step 4 – levels and contrast, in perceptual space.
///
/// The contrast curve keeps 0, 1 and middle grey (18 %) fixed and bends around middle grey with
/// two power functions: `k > 1` gives an S-curve (deeper shadows, brighter highlights), `k < 1`
/// flattens. It is monotonic for any `k > 0`, so it can never invert tones.
#[inline]
pub fn tone(p: &Params, linear: f32) -> f32 {
    let t = linear.max(0.0).powf(1.0 / GAMMA);
    let t = ((t - p.black) / (p.white - p.black).max(1e-3)).clamp(0.0, 1.0);
    let pivot: f32 = 0.18f32.powf(1.0 / GAMMA);
    let k = p.contrast.max(0.05);
    let t = if t < pivot {
        pivot * (t / pivot).powf(k)
    } else {
        1.0 - (1.0 - pivot) * ((1.0 - t) / (1.0 - pivot)).powf(k)
    };
    t.powf(GAMMA)
}

/// The value just before white balance (used by the grey picker).
#[inline]
pub fn pre_balance(p: &Params, c: usize, code: f32) -> f32 {
    let lin = linearize(p, code);
    if p.negative {
        invert_negative(p, c, lin)
    } else {
        lin
    }
}

/// Steps 1–3: scene-linear value after balance and exposure (used by auto-levels).
#[inline]
pub fn scene_linear(p: &Params, gains: &[f32; 3], c: usize, code: f32) -> f32 {
    pre_balance(p, c, code) * gains[c]
}

/// The whole per-channel chain.
#[inline]
pub fn channel_curve(p: &Params, gains: &[f32; 3], c: usize, code: f32) -> f32 {
    tone(p, scene_linear(p, gains, c, code))
}

/// Spacing (in code values) between exactly-evaluated LUT entries.
const KNOT_STEP: usize = 4;

/// Tabulate [`channel_curve`] for every 16-bit code value of every channel.
///
/// Evaluating the full chain (several `log`/`pow` calls) 196 608 times takes ~60 ms in wasm –
/// too slow to rebuild on every slider movement. The curve is smooth, so we evaluate it exactly
/// every 4 code values (16 385 knots per channel) and linearly interpolate in between. The
/// error this introduces is a small fraction of an 8-bit step and at most about two 16-bit steps
/// (see the test below).
pub fn build_lut(p: &Params) -> Vec<f32> {
    let gains = channel_gains(p);
    let mut lut = vec![0f32; 3 * LUT_SIZE];
    let max = (LUT_SIZE - 1) as f32;
    for c in 0..3 {
        let table = &mut lut[c * LUT_SIZE..(c + 1) * LUT_SIZE];
        let mut prev = channel_curve(p, &gains, c, 0.0);
        for knot in (KNOT_STEP..LUT_SIZE + KNOT_STEP).step_by(KNOT_STEP) {
            let end = knot.min(LUT_SIZE - 1);
            let next = channel_curve(p, &gains, c, end as f32 / max);
            let start = knot - KNOT_STEP;
            let span = (end - start) as f32;
            for i in start..=end {
                table[i] = prev + (next - prev) * ((i - start) as f32 / span);
            }
            prev = next;
        }
    }
    lut
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neutral_params_are_identity_in_linear() {
        let p = Params::default();
        let g = channel_gains(&p);
        for i in [0.0, 0.01, 0.18, 0.5, 1.0] {
            let code = crate::color::linear_to_srgb(i);
            assert!((channel_curve(&p, &g, 1, code) - i).abs() < 1e-4, "{i}");
        }
    }

    #[test]
    fn negative_maps_dmin_to_black_and_dmax_to_white() {
        let p = Params { negative: true, input_linear: true, dmin: [0.3; 3], dmax: [1.8; 3], ..Default::default() };
        let clear = 10f32.powf(-0.3); // transmittance at dmin
        let dense = 10f32.powf(-1.8);
        assert!(invert_negative(&p, 0, clear).abs() < 1e-5);
        assert!((invert_negative(&p, 0, dense) - 1.0).abs() < 1e-4);
        // Denser negative = brighter positive.
        assert!(invert_negative(&p, 0, 0.1) > invert_negative(&p, 0, 0.2));
    }

    #[test]
    fn interpolated_lut_is_accurate() {
        for p in [
            Params::default(),
            Params { contrast: 1.8, exposure: 1.0, black: 0.05, ..Default::default() },
            Params { negative: true, dmin: [0.2, 0.5, 0.8], dmax: [1.6, 1.9, 2.4], ..Default::default() },
        ] {
            let lut = build_lut(&p);
            let g = channel_gains(&p);
            for c in 0..3 {
                // Skip the very darkest codes, where a negative's density → ∞ (clipped white anyway).
                for i in (64..LUT_SIZE).step_by(7) {
                    let exact = channel_curve(&p, &g, c, i as f32 / 65535.0).clamp(0.0, 1.0);
                    let err = (lut[c * LUT_SIZE + i].clamp(0.0, 1.0) - exact).abs();
                    assert!(err < 3e-5, "c{c} code {i}: {err}");
                }
            }
        }
    }

    #[test]
    fn contrast_curve_is_monotonic() {
        for k in [0.5, 1.0, 1.7, 3.0] {
            let p = Params { contrast: k, ..Default::default() };
            let mut last = -1.0;
            for i in 0..=1000 {
                let v = tone(&p, i as f32 / 1000.0);
                assert!(v >= last);
                last = v;
            }
        }
    }
}
