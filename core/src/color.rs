//! Colour science helpers: transfer functions and 3×3 matrices.
//!
//! Two ideas matter everywhere in this project:
//!
//! * **Linear vs. encoded.** Light adds linearly, so physically meaningful maths (exposure,
//!   white balance, averaging pixels, film density) happens on *linear* values. Files store
//!   *encoded* values (the sRGB curve) because human vision is roughly logarithmic and 8 bits
//!   would band badly if stored linearly.
//! * **Primaries.** "RGB" only means something relative to a set of primaries. Everything inside
//!   Darkroom uses the sRGB / Rec.709 primaries with a D65 white point.

/// sRGB electro-optical transfer function: encoded `[0,1]` → linear light `[0,1]`.
#[inline]
pub fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// Inverse of [`srgb_to_linear`]: linear light → sRGB-encoded value.
#[inline]
pub fn linear_to_srgb(v: f32) -> f32 {
    if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

/// Rec.709 / sRGB luminance weights (the Y row of the RGB→XYZ matrix).
pub const LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];

pub type Mat3 = [[f32; 3]; 3];

/// Linear sRGB (D65) → CIE XYZ.
pub const SRGB_TO_XYZ: Mat3 = [
    [0.412_456_4, 0.357_576_1, 0.180_437_5],
    [0.212_672_9, 0.715_152_2, 0.072_175_0],
    [0.019_333_9, 0.119_192_0, 0.950_304_1],
];

pub fn mat3_mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    out
}

pub fn mat3_apply(m: &Mat3, v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// Inverse via the adjugate. Returns `None` for (near-)singular matrices.
pub fn mat3_inverse(m: &Mat3) -> Option<Mat3> {
    let c = |r0: usize, c0: usize, r1: usize, c1: usize| m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0];
    let det = m[0][0] * c(1, 1, 2, 2) - m[0][1] * c(1, 0, 2, 2) + m[0][2] * c(1, 0, 2, 1);
    if det.abs() < 1e-9 {
        return None;
    }
    let inv = 1.0 / det;
    Some([
        [c(1, 1, 2, 2) * inv, -c(0, 1, 2, 2) * inv, c(0, 1, 1, 2) * inv],
        [-c(1, 0, 2, 2) * inv, c(0, 0, 2, 2) * inv, -c(0, 0, 1, 2) * inv],
        [c(1, 0, 2, 1) * inv, -c(0, 0, 2, 1) * inv, c(0, 0, 1, 1) * inv],
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_roundtrip() {
        for i in 0..=100 {
            let v = i as f32 / 100.0;
            assert!((linear_to_srgb(srgb_to_linear(v)) - v).abs() < 1e-5);
        }
    }

    #[test]
    fn inverse_is_inverse() {
        let inv = mat3_inverse(&SRGB_TO_XYZ).unwrap();
        let id = mat3_mul(&SRGB_TO_XYZ, &inv);
        for i in 0..3 {
            for j in 0..3 {
                assert!((id[i][j] - if i == j { 1.0 } else { 0.0 }).abs() < 1e-5);
            }
        }
    }
}
