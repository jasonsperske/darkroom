//! Train the FFDNet-lite denoiser (`core/src/denoise/neural.rs`) from scratch, on the CPU.
//!
//! ```sh
//! RUSTFLAGS="-C target-cpu=native" cargo run --release --example train_denoiser -- \
//!     path/to/kodak  ../web/models/denoise.bin  [steps]
//! ```
//!
//! The training set is the Kodak Lossless True Color Image Suite (kodim01–24.png, 768×512,
//! released by Kodak for unrestricted use: <https://r0k.us/graphics/kodak/>). kodim22–24 are
//! held out for validation.
//!
//! Everything a deep-learning framework would normally hide is written out here:
//!
//! * **Data:** clean image patches with *synthetic film grain* added (see "Grain synthesis"
//!   below), so the network learns what grain looks like, as opposed to texture.
//! * **Loss:** mean squared error between the predicted and the true noise.
//! * **Backpropagation:** hand-written gradients for 3×3 convolution and ReLU.
//! * **Optimiser:** Adam with a cosine learning-rate schedule.
//! * **Parallelism:** each sample of a batch is processed on its own thread (rayon), and the
//!   per-sample gradients are summed.
//!
//! ## Clean images
//!
//! The Kodak images are themselves film scans with visible grain. Training on them as-is would
//! teach the network that some grain is "clean". We halve their resolution first, which
//! averages away most of it.
//!
//! ## Grain synthesis
//!
//! Each training patch gets noise drawn from a wide family, so one network covers both
//! digital noise and film:
//!
//! * **Size:** grain blurred by a Gaussian of radius 0–3 px (`grain.rs`); 0 is digital noise.
//! * **Colour structure:** each RGB channel's grain mixes a shared component with its own,
//!   with a random correlation. Fully correlated is monochrome silver grain (B&W film);
//!   uncorrelated is independent dye layers. Channels get somewhat different amounts (the
//!   blue-sensitive layer of colour film is usually the grainiest).
//! * **Tone dependence:** the grain's amplitude varies with brightness, as granularity
//!   varies with density.
//! * **Scanner noise:** sometimes a little white noise on top.
//! * **Negative-like images:** half the patches are inverted and given a random tone curve,
//!   because negatives are denoised in density space (`denoise/mod.rs`), where an image looks
//!   like a flattened, inverted positive.
//!
//! The network is told the realised luma/chroma σ and the grain size (with a little error
//! added, since at run time these come from the automatic estimate).

use darkroom_core::denoise::grain::{grain_field, Rng};
use darkroom_core::denoise::neural::{conv3x3, prepare, Layer, Network, Tensor, FEATURES, IN_CHANNELS, LAYERS, OUT_CHANNELS};
use darkroom_core::denoise::NoiseProfile;
use rayon::prelude::*;
use std::time::Instant;

const PATCH: usize = 64;
const BATCH: usize = 16;
const MAX_SIGMA: f32 = 50.0 / 255.0;
const MAX_GRAIN: f32 = 3.0;

// ---- Data -------------------------------------------------------------------------------------

/// A clean image: three channels of sRGB code values in 0…1, at half the original resolution.
struct Image {
    w: usize,
    h: usize,
    rgb: [Vec<f32>; 3],
}

fn load(path: &std::path::Path) -> Image {
    let img = image::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display())).to_rgb16();
    let (w, h) = (img.width() as usize / 2, img.height() as usize / 2);
    let raw = img.as_raw();
    let full_w = img.width() as usize;
    let rgb = std::array::from_fn(|c| {
        let mut plane = vec![0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let at = |dx: usize, dy: usize| raw[((2 * y + dy) * full_w + 2 * x + dx) * 3 + c] as f32 / 65535.0;
                plane[y * w + x] = (at(0, 0) + at(1, 0) + at(0, 1) + at(1, 1)) / 4.0;
            }
        }
        plane
    });
    Image { w, h, rgb }
}

/// Description of the noise to add to one patch.
#[derive(Clone, Copy)]
struct GrainSpec {
    sigma: f32,
    size: f32,
    /// Correlation of grain between colour channels (1 = monochrome grain).
    correlation: f32,
    /// Per-channel amount multipliers.
    channel: [f32; 3],
    /// Amplitude slope with brightness: m = 1 + tone · (luma − 0.5).
    tone: f32,
    scanner: f32,
    negative_like: bool,
    curve: f32,
}

impl GrainSpec {
    fn random(rng: &mut Rng) -> Self {
        let size = if rng.below(4) == 0 { 0.0 } else { rng.uniform() * MAX_GRAIN };
        let correlation = match rng.below(5) {
            0 => 1.0,
            _ => rng.uniform() * 0.8,
        };
        GrainSpec {
            sigma: if rng.below(25) == 0 { 0.0 } else { rng.uniform() * MAX_SIGMA },
            size,
            correlation,
            channel: std::array::from_fn(|c| 0.8 + rng.uniform() * 0.4 + if c == 2 { rng.uniform() * 0.4 } else { 0.0 }),
            tone: rng.uniform() * 2.0 - 1.0,
            scanner: if rng.below(3) == 0 { rng.uniform() * 8.0 / 255.0 } else { 0.0 },
            negative_like: rng.below(2) == 0,
            curve: 0.6 + rng.uniform(),
        }
    }

    /// The fixed validation condition: medium film grain.
    fn validation() -> Self {
        GrainSpec { sigma: 20.0 / 255.0, size: 1.5, correlation: 0.5, channel: [1.0, 1.0, 1.2], tone: 0.3, scanner: 0.0, negative_like: false, curve: 1.0 }
    }
}

/// A training example: noisy input tensor (15 ch) and the noise the network should predict.
struct Sample {
    input: Tensor,
    target: Tensor,
}

/// RGB → orthonormal Y, C1, C2 (same transform as `denoise::Planes`).
fn to_ycc(rgb: &[Vec<f32>; 3]) -> [Vec<f32>; 3] {
    let (s3, s2, s6) = (1.0 / 3f32.sqrt(), 1.0 / 2f32.sqrt(), 1.0 / 6f32.sqrt());
    let n = rgb[0].len();
    let mut out: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0f32; n]);
    for i in 0..n {
        let (r, g, b) = (rgb[0][i], rgb[1][i], rgb[2][i]);
        out[0][i] = (r + g + b) * s3;
        out[1][i] = (r - b) * s2;
        out[2][i] = (r - 2.0 * g + b) * s6;
    }
    out
}

fn std_dev(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt()
}

fn make_sample(img: &Image, rng: &mut Rng, spec: GrainSpec) -> Sample {
    // Random crop, then one of the 8 rotations/mirrors.
    let x0 = rng.below(img.w - PATCH + 1);
    let y0 = rng.below(img.h - PATCH + 1);
    let (transpose, flip_x, flip_y) = (rng.below(2) == 1, rng.below(2) == 1, rng.below(2) == 1);
    let n = PATCH * PATCH;
    let mut clean: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0f32; n]);
    for (c, plane) in clean.iter_mut().enumerate() {
        for y in 0..PATCH {
            for x in 0..PATCH {
                let (mut sx, mut sy) = if transpose { (y, x) } else { (x, y) };
                if flip_x {
                    sx = PATCH - 1 - sx;
                }
                if flip_y {
                    sy = PATCH - 1 - sy;
                }
                let mut v = img.rgb[c][(y0 + sy) * img.w + x0 + sx].powf(spec.curve);
                if spec.negative_like {
                    v = 1.0 - v;
                }
                plane[y * PATCH + x] = v;
            }
        }
    }

    // Grain: shared + per-channel components, each shaped to the grain size.
    let shared = grain_field(PATCH, PATCH, spec.size, rng);
    let (a, b) = (spec.correlation.sqrt(), (1.0 - spec.correlation).sqrt());
    let mut noisy = clean.clone();
    for c in 0..3 {
        let own = grain_field(PATCH, PATCH, spec.size, rng);
        for i in 0..n {
            let luma = (clean[0][i] + clean[1][i] + clean[2][i]) / 3.0;
            let m = (1.0 + spec.tone * (luma - 0.5)).max(0.2);
            noisy[c][i] += spec.sigma * spec.channel[c] * m * (a * shared[i] + b * own[i]) + spec.scanner * rng.gauss();
        }
    }

    let clean_ycc = to_ycc(&clean);
    let noisy_ycc = to_ycc(&noisy);
    let diff: [Vec<f32>; 3] = std::array::from_fn(|c| noisy_ycc[c].iter().zip(&clean_ycc[c]).map(|(a, b)| a - b).collect());

    // What the network is told: the realised noise, with up to ±15 % estimation error.
    let mut jitter = || 1.0 + (rng.uniform() - 0.5) * 0.3;
    let told = NoiseProfile {
        luma: std_dev(&diff[0]) * jitter(),
        chroma: 0.5 * (std_dev(&diff[1]) + std_dev(&diff[2])) * jitter(),
        size: (spec.size * jitter()).min(MAX_GRAIN),
        measured: None,
    };
    let input = prepare(&noisy_ycc, PATCH, PATCH, told);
    let t = prepare(&diff, PATCH, PATCH, NoiseProfile::new(0.0, 0.0, 0.0));
    let target = Tensor { c: OUT_CHANNELS, h: t.h, w: t.w, data: t.data[..OUT_CHANNELS * t.h * t.w].to_vec() };
    Sample { input, target }
}

// ---- Backpropagation -----------------------------------------------------------------------------

/// Gradients of the loss w.r.t. a convolution's weights, bias and input.
///
/// Forward: `Y[co](y,x) = b[co] + Σ_ci Σ_ky,kx W[co][ci][ky][kx] · X[ci](y+ky−1, x+kx−1)`
///
/// * `dW[co][ci][ky][kx] = Σ_y,x dY[co](y,x) · X[ci](y+ky−1, x+kx−1)`  (correlation)
/// * `db[co]             = Σ_y,x dY[co](y,x)`
/// * `dX[ci](p)          = Σ_co Σ_k W[co][ci][k] · dY[co](p − k + 1)`   (the transposed conv)
fn conv3x3_backward(x: &Tensor, layer: &Layer, dy: &Tensor, need_dx: bool) -> (Vec<f32>, Vec<f32>, Option<Tensor>) {
    let (h, w) = (x.h, x.w);
    let plane = h * w;
    let mut dw = vec![0f32; layer.w.len()];
    let mut db = vec![0f32; layer.cout];
    let mut dx = need_dx.then(|| Tensor { c: layer.cin, h, w, data: vec![0f32; layer.cin * plane] });
    for co in 0..layer.cout {
        let g = &dy.data[co * plane..(co + 1) * plane];
        db[co] = g.iter().sum();
        for ci in 0..layer.cin {
            let xp = &x.data[ci * plane..(ci + 1) * plane];
            for ky in 0..3 {
                for kx in 0..3 {
                    let wi = ((co * layer.cin + ci) * 3 + ky) * 3 + kx;
                    let wv = layer.w[wi];
                    let mut acc = 0f32;
                    // Output positions whose tap (ky, kx) lands inside the input.
                    let (y_lo, y_hi) = (if ky == 0 { 1 } else { 0 }, if ky == 2 { h - 1 } else { h });
                    let (x_lo, x_hi) = (if kx == 0 { 1 } else { 0 }, if kx == 2 { w - 1 } else { w });
                    for y in y_lo..y_hi {
                        let sy = y + ky - 1;
                        let grow = &g[y * w + x_lo..y * w + x_hi];
                        let xrow = &xp[sy * w + x_lo + kx - 1..sy * w + x_hi + kx - 1];
                        acc += grow.iter().zip(xrow).map(|(a, b)| a * b).sum::<f32>();
                        if let Some(dx) = dx.as_mut() {
                            let drow = &mut dx.data[ci * plane + sy * w + x_lo + kx - 1..ci * plane + sy * w + x_hi + kx - 1];
                            for (d, gv) in drow.iter_mut().zip(grow) {
                                *d += wv * gv;
                            }
                        }
                    }
                    dw[wi] = acc;
                }
            }
        }
    }
    (dw, db, dx)
}

/// Per-layer `(weight gradients, bias gradients)`.
type LayerGrads = Vec<(Vec<f32>, Vec<f32>)>;

/// Forward + backward for one sample. Returns (loss, per-layer weight and bias gradients).
fn gradients(net: &Network, s: &Sample) -> (f32, LayerGrads) {
    // Forward, keeping every activation.
    let last = net.layers.len() - 1;
    let mut acts = vec![conv3x3(&s.input, &net.layers[0], last != 0)];
    for (i, layer) in net.layers.iter().enumerate().skip(1) {
        let next = conv3x3(acts.last().unwrap(), layer, i != last);
        acts.push(next);
    }
    let pred = acts.last().unwrap();
    let n = pred.data.len() as f32;
    let mut loss = 0f64;
    let mut grad = Tensor { c: pred.c, h: pred.h, w: pred.w, data: vec![0f32; pred.data.len()] };
    for ((g, p), t) in grad.data.iter_mut().zip(&pred.data).zip(&s.target.data) {
        let d = p - t;
        loss += (d * d) as f64;
        *g = 2.0 * d / n;
    }

    // Backward, last layer first.
    let mut grads = vec![(Vec::new(), Vec::new()); net.layers.len()];
    for i in (0..net.layers.len()).rev() {
        if i != last {
            // ReLU: gradient flows only where the activation was positive.
            for (g, a) in grad.data.iter_mut().zip(&acts[i].data) {
                if *a <= 0.0 {
                    *g = 0.0;
                }
            }
        }
        let input = if i == 0 { &s.input } else { &acts[i - 1] };
        let (dw, db, dx) = conv3x3_backward(input, &net.layers[i], &grad, i > 0);
        grads[i] = (dw, db);
        if let Some(dx) = dx {
            grad = dx;
        }
    }
    ((loss / n as f64) as f32, grads)
}

// ---- Setup, Adam, main loop ------------------------------------------------------------------

fn init(rng: &mut Rng) -> Network {
    let mut layers = Vec::new();
    for i in 0..LAYERS {
        let cin = if i == 0 { IN_CHANNELS } else { FEATURES };
        let cout = if i == LAYERS - 1 { OUT_CHANNELS } else { FEATURES };
        // He initialisation, suited to ReLU; the output layer starts small (≈ predict no noise).
        let std = (2.0 / (cin * 9) as f32).sqrt() * if i == LAYERS - 1 { 0.1 } else { 1.0 };
        layers.push(Layer { cin, cout, w: (0..cout * cin * 9).map(|_| rng.gauss() * std).collect(), b: vec![0.0; cout] });
    }
    Network { layers }
}

struct Adam {
    m: Vec<f32>,
    v: Vec<f32>,
    t: i32,
}

impl Adam {
    fn step(&mut self, params: &mut [&mut f32], grads: &[f32], lr: f32) {
        const B1: f32 = 0.9;
        const B2: f32 = 0.999;
        self.t += 1;
        let (c1, c2) = (1.0 - B1.powi(self.t), 1.0 - B2.powi(self.t));
        for (i, p) in params.iter_mut().enumerate() {
            self.m[i] = B1 * self.m[i] + (1.0 - B1) * grads[i];
            self.v[i] = B2 * self.v[i] + (1.0 - B2) * grads[i] * grads[i];
            **p -= lr * (self.m[i] / c1) / ((self.v[i] / c2).sqrt() + 1e-8);
        }
    }
}

fn psnr(mse: f32) -> f32 {
    // Our working space runs 0…1, so the peak signal is 1.
    -10.0 * mse.max(1e-12).log10()
}

/// Compare analytic gradients with finite differences for a few random weights in every layer.
/// (Run with `gradcheck` as the step count.) Relative errors around 1e-2 or less are fine in f32.
fn gradcheck(net: &mut Network, img: &Image) {
    let mut rng = Rng(99);
    let sample = make_sample(img, &mut rng, GrainSpec::validation());
    let (_, grads) = gradients(net, &sample);
    for li in 0..net.layers.len() {
        for _ in 0..3 {
            let wi = rng.below(net.layers[li].w.len());
            let orig = net.layers[li].w[wi];
            let eps = 3e-3;
            net.layers[li].w[wi] = orig + eps;
            let up = gradients(net, &sample).0;
            net.layers[li].w[wi] = orig - eps;
            let down = gradients(net, &sample).0;
            net.layers[li].w[wi] = orig;
            let numeric = (up - down) / (2.0 * eps);
            let analytic = grads[li].0[wi];
            let rel = (numeric - analytic).abs() / numeric.abs().max(analytic.abs()).max(1e-9);
            println!("layer {li} w[{wi:5}]  analytic {analytic:+.4e}  numeric {numeric:+.4e}  rel.err {rel:.1e}");
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = std::path::PathBuf::from(args.get(1).expect("usage: train_denoiser <kodak dir> <out.bin> [steps]"));
    let out = std::path::PathBuf::from(args.get(2).expect("missing output path"));
    let steps: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(6000);

    let images: Vec<Image> = (1..=24).map(|i| load(&dir.join(format!("kodim{i:02}.png")))).collect();
    let (train, val) = images.split_at(21);
    println!("{} training images, {} validation images", train.len(), val.len());

    let mut rng = Rng(0xDA4C_400D);
    let mut net = init(&mut rng);
    if args.get(3).map(String::as_str) == Some("gradcheck") {
        gradcheck(&mut net, &train[0]);
        return;
    }
    let count: usize = net.layers.iter().map(|l| l.w.len() + l.b.len()).sum();
    println!("network: {LAYERS} layers, {FEATURES} features, {count} parameters");

    // A fixed validation set: 3 images × 8 crops with medium film grain (σ 20/255, size 1.5 px).
    let mut vrng = Rng(7);
    let val_set: Vec<Sample> = val.iter().flat_map(|img| (0..8).map(|_| make_sample(img, &mut vrng, GrainSpec::validation())).collect::<Vec<_>>()).collect();
    let noisy_mse: f32 = val_set.iter().map(|s| s.target.data.iter().map(|v| v * v).sum::<f32>() / s.target.data.len() as f32).sum::<f32>() / val_set.len() as f32;

    let mut adam = Adam { m: vec![0.0; count], v: vec![0.0; count], t: 0 };
    let started = Instant::now();
    let mut running = 0.0;
    for step in 1..=steps {
        let seeds: Vec<u64> = (0..BATCH).map(|_| rng.next_u64()).collect();
        let results: Vec<(f32, LayerGrads)> = seeds
            .par_iter()
            .map(|&seed| {
                let mut r = Rng(seed);
                let img = &train[r.below(train.len())];
                let spec = GrainSpec::random(&mut r);
                gradients(&net, &make_sample(img, &mut r, spec))
            })
            .collect();

        // Average gradients over the batch, flattened in the same order as the parameters.
        let mut flat = vec![0f32; count];
        let mut loss = 0.0;
        for (l, grads) in &results {
            loss += l / BATCH as f32;
            let mut k = 0;
            for (dw, db) in grads {
                for g in dw.iter().chain(db) {
                    flat[k] += g / BATCH as f32;
                    k += 1;
                }
            }
        }
        running = if step == 1 { loss } else { 0.98 * running + 0.02 * loss };

        // Cosine decay from 1e-3 to ~2e-5.
        let lr = 2e-5 + 0.5 * (1e-3 - 2e-5) * (1.0 + (std::f32::consts::PI * step as f32 / steps as f32).cos());
        let mut params: Vec<&mut f32> = net.layers.iter_mut().flat_map(|l| l.w.iter_mut().chain(l.b.iter_mut())).collect();
        adam.step(&mut params, &flat, lr);

        if step % 250 == 0 || step == steps {
            let val_mse: f32 = val_set.par_iter().map(|s| gradients(&net, s).0).sum::<f32>() / val_set.len() as f32;
            println!(
                "step {step:5}  train {:.3} dB  val {:.2} dB (noisy {:.2} dB)  lr {lr:.1e}  {:.0}s",
                psnr(running),
                psnr(val_mse),
                psnr(noisy_mse),
                started.elapsed().as_secs_f32()
            );
            std::fs::write(&out, net.to_bytes()).expect("write weights");
        }
    }
    println!("saved {} ({} bytes)", out.display(), net.to_bytes().len());
}
