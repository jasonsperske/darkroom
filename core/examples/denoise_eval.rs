//! Compare the denoisers on held-out photos with synthetic noise and film grain.
//!
//! ```sh
//! cargo run --release --example denoise_eval -- path/to/kodak [../web/models/denoise.bin]
//! ```
//!
//! For kodim22–24 (never used for training) and several noise conditions, from digital noise
//! to coarse film grain, prints the PSNR of the noisy image and of each method's output (with
//! the *automatically measured* noise profile, as in the app), plus the measurement itself.

use darkroom_core::denoise::grain::{grain_field, Rng};
use darkroom_core::denoise::{self, neural::Network, DenoiseParams, Domain, Method};
use std::time::Instant;

fn psnr(a: &[u16], b: &[u16]) -> f64 {
    let mse: f64 = a.iter().zip(b).map(|(x, y)| ((*x as f64 - *y as f64) / 65535.0).powi(2)).sum::<f64>() / a.len() as f64;
    -10.0 * mse.log10()
}

struct Condition {
    name: &'static str,
    sigma: f32,
    size: f32,
    correlation: f32,
}

const CONDITIONS: [Condition; 5] = [
    Condition { name: "digital", sigma: 15.0, size: 0.0, correlation: 0.0 },
    Condition { name: "ISO 400 colour", sigma: 12.0, size: 1.0, correlation: 0.5 },
    Condition { name: "ISO 800 colour", sigma: 18.0, size: 1.5, correlation: 0.5 },
    Condition { name: "ISO 3200 B&W", sigma: 22.0, size: 2.0, correlation: 1.0 },
    Condition { name: "pushed colour", sigma: 30.0, size: 2.5, correlation: 0.3 },
];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = std::path::PathBuf::from(args.get(1).expect("usage: denoise_eval <kodak dir> [weights]"));
    let net = args.get(2).map(|p| Network::parse(&std::fs::read(p).expect("read weights")).expect("parse weights"));
    let methods = [Method::Wavelet, Method::Nlm, Method::Neural];
    let mut totals = [0f64; 4];
    let mut count = 0;

    println!("{:<16} {:<8} {:>7} | {:>14} {:>14} {:>14} | measured σ, size", "condition", "image", "noisy", "wavelet", "nlm", "neural");
    for cond in &CONDITIONS {
        for i in 22..=24 {
            let img = image::open(dir.join(format!("kodim{i}.png"))).unwrap().to_rgb16();
            let (w, h) = (img.width() as usize, img.height() as usize);
            let clean = img.into_raw();

            // Grain with the given size and colour correlation, added to the sRGB code values.
            let mut rng = Rng(i as u64 * 1000 + cond.size as u64);
            let shared = grain_field(w, h, cond.size, &mut rng);
            let own: Vec<Vec<f32>> = (0..3).map(|_| grain_field(w, h, cond.size, &mut rng)).collect();
            let (a, b) = (cond.correlation.sqrt(), (1.0 - cond.correlation).sqrt());
            let s = cond.sigma / 255.0;
            let noisy: Vec<u16> = clean
                .iter()
                .enumerate()
                .map(|(k, &v)| {
                    let (p, c) = (k / 3, k % 3);
                    let n = s * (a * shared[p] + b * own[c][p]);
                    ((v as f32 / 65535.0 + n).clamp(0.0, 1.0) * 65535.0).round() as u16
                })
                .collect();

            let est = denoise::estimate_noise(&noisy, w, h, false, Domain::Gamma);
            let base = psnr(&clean, &noisy);
            totals[0] += base;
            print!("{:<16} kodim{i:<3} {base:>7.2} |", cond.name);
            for (mi, m) in methods.iter().enumerate() {
                if *m == Method::Neural && net.is_none() {
                    print!(" {:>14}", "-");
                    continue;
                }
                let t = Instant::now();
                let p = DenoiseParams { method: *m, luma: 1.0, chroma: 1.0, negative: false };
                let out = denoise::denoise_rgb16(&noisy, w, h, false, &p, est, net.as_ref()).unwrap();
                let q = psnr(&clean, &out);
                totals[mi + 1] += q;
                print!(" {q:>8.2} {:>4.0}ms", t.elapsed().as_secs_f64() * 1000.0);
            }
            println!(" | luma {:.1} chroma {:.1} size {:.2}", est.luma * 255.0, est.chroma * 255.0, est.size);
            count += 1;
        }
    }
    let n = count as f64;
    println!("{:<16} {:<8} {:>7.2} | {:>8.2}       {:>8.2}       {:>8.2}", "MEAN", "", totals[0] / n, totals[1] / n, totals[2] / n, totals[3] / n);
}
