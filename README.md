# Darkroom

A darkroom that runs entirely in your browser. You can drop in photos, film scans or camera raw files, invert and colour-balance negatives, crop to standard print sizes, and export with as little quality loss as possible. Nothing is uploaded: all decoding and processing happens on your device, in Rust compiled to WebAssembly and running in Web Workers. The live preview uses WebGPU.

The code is also meant to be read. Every module starts with a comment explaining the idea behind it, and the guide below tells you where to begin for each topic.

## Features

- **Formats in:** JPEG, PNG and TIFF (8- and 16-bit), WebP, BMP and GIF. Camera raw: CR2, CR3, NEF, ARW, RAF, ORF, RW2, DNG and many more (via [rawler](https://github.com/dnglab/dnglab)). Formats we can't decode, such as AVIF or HEIC, fall back to the browser's own decoder.
- **Negatives:** the inversion works on optical density. It removes the orange mask and balances each dye layer automatically, or you can click the film rebate to set the base yourself. B&W negatives are supported too.
- **Grain and noise reduction, built for film:**
  - A two-parameter grain model, *amount* and *grain size*, is measured automatically from flat areas of the image. It covers both digital noise and the clumped grain of fast or underexposed film. Each wavelet scale also gets its own direct measurement, because real scans mix scanner noise with film grain.
  - Negatives are denoised in **optical density, before inversion**, where grain is most uniform. Inversion is what blows up grain in the shadows of underexposed frames.
  - There are three methods: à trous wavelet shrinkage, non-local means, and a small **neural network** (FFDNet-style CNN). The network is trained on synthetic film grain and runs as WebGPU compute shaders, with a WebAssembly SIMD fallback.
  - Luminance, colour and grain-size controls let you remove grain completely or just tame it.
- **Develop controls:** exposure, contrast (an S-curve, like choosing a paper grade), black and white points, temperature and tint, a grey picker, saturation, black & white, and auto levels.
- **Crop and rotate:** aspect presets for film and print formats (35mm, 6×6, 4×5/8×10, 5×7, 11×14, A-series, XPan, …), plus 90° rotation and flips.
- **Export:**
  - JPEG at 4:4:4, PNG at 8 or 16 bits, TIFF at 16 bits, all with optional resizing.
  - **Lossless JPEG crop and rotate:** the original DCT coefficients are re-packed without being decoded, so there is no generational loss.
- **Reviewing:** a 100 % loupe rendered from the full-resolution image, a before/after toggle, a histogram, and a filmstrip for working on a whole roll.

## Running it

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.100   # must match core/Cargo.toml
./build.sh                                          # → web/pkg/{simd,baseline}
./serve.py 8000                                     # or any static server
```

Then open <http://localhost:8000>. WebGPU needs a secure context, and `localhost` counts as one. Add `?cpu` to the URL to force the Canvas 2D fallback.

You can run the Rust tests natively with `cargo test --release --manifest-path core/Cargo.toml`. To decode a single file from the command line, run `cargo run --release --manifest-path core/Cargo.toml --example decode -- photo.nef`.

### GitHub Pages

`.github/workflows/pages.yml` runs the tests, builds the wasm and publishes `web/` on every push to `main`. To turn it on, go to **Settings → Pages → Source: GitHub Actions**. Pages serves over HTTPS, which WebGPU needs.

## Architecture

```
 UI thread (web/js/main.js)                      Engine worker (web/workers/engine.worker.js)
 ┌──────────────────────────────┐   messages    ┌─────────────────────────────────────────┐
 │ DOM, crop tool, histogram    │ ────────────► │ owns all pixels; own wasm instance for   │
 │ WebGPU preview               │ ◄──────────── │ tone LUTs, preview, analysis            │
 │  (proxy texture + LUT)       │  LUT / bitmap │                                         │
 └──────────────────────────────┘               │ WorkerPool (web/workers/pool.js)        │
                                                │  ├─ compute worker ─ wasm  decode       │
                                                │  ├─ compute worker ─ wasm  develop strip│
                                                │  └─ …N               wasm  encode       │
                                                └─────────────────────────────────────────┘
```

- **One implementation of the maths.** Every per-channel tone operation is tabulated by Rust into a lookup table (3 × 65 536 floats). The WebGPU shader, the CPU preview and the full-resolution export all use that same table, so the preview always matches the export.
- **Parallelism without SharedArrayBuffer.** WebAssembly threads need cross-origin isolation headers, and GitHub Pages can't send them. Instead the pool runs N independent wasm instances, and the export is split into horizontal strips that are transferred to the workers and back without copying.
- **SIMD where it helps.** The build produces a `simd128` variant and a fallback, and `web/js/wasm.js` picks one by feature detection. The SIMD export loop is hand-written in `core/src/develop/apply.rs`. That file also explains why table lookups stay scalar: wasm SIMD has no gather instruction.
- **Proxy and master.** Each image is decoded to a 16-bit RGB *master*. A screen-sized *proxy* drives the preview and analysis. Only export and the loupe touch the master. Masters that haven't been used recently are freed and decoded again on demand.

## Reading guide

| If you want to learn about…              | start here                                               |
|------------------------------------------|----------------------------------------------------------|
| How negative inversion works             | `core/src/develop/tone.rs` → `invert_negative`, `density` |
| Auto film base / colour balance          | `core/src/develop/analysis.rs`                            |
| The full develop pipeline and why LUTs   | `core/src/develop/mod.rs`                                 |
| WebAssembly SIMD                         | `core/src/develop/apply.rs`                               |
| Raw files: black level, WB, colour matrix| `core/src/decode/raw.rs`                                  |
| Demosaicing (Malvar–He–Cutler)           | `core/src/decode/demosaic.rs`                             |
| sRGB, linear light, colour matrices      | `core/src/color.rs`                                       |
| How JPEG is structured                   | `core/src/jpeg/mod.rs`, then `parse.rs`                   |
| Huffman coding (decode + optimal tables) | `core/src/jpeg/huffman.rs`                                |
| Progressive JPEG decoding                | `core/src/jpeg/parse.rs` → `decode_block`                 |
| Rotating/cropping JPEG without loss      | `core/src/jpeg/transform.rs`                              |
| Rotation/flip as matrices (D4 group)     | `core/src/geometry.rs`, `web/js/geometry.js`              |
| Image downscaling                        | `core/src/resize.rs`                                      |
| Noise: working spaces (gamma vs density) | `core/src/denoise/mod.rs`                                 |
| Film grain model & measuring grain size  | `core/src/denoise/grain.rs`                               |
| Wavelet denoising                        | `core/src/denoise/wavelet.rs`                             |
| Non-local means                          | `core/src/denoise/nlm.rs`                                 |
| A CNN: architecture & inference          | `core/src/denoise/neural.rs`                              |
| Training a CNN from scratch (backprop)   | `core/examples/train_denoiser.rs`                         |
| Running a CNN in WebGPU compute shaders  | `web/js/nn/webgpu-denoise.js`                             |
| WebGPU rendering                         | `web/js/render/webgpu.js`                                 |
| Worker pools and zero-copy transfers     | `web/workers/pool.js`, `web/workers/engine.worker.js`     |
| Crop tool with aspect constraints        | `web/js/ui/crop.js`                                       |

`core/tests/lossless_jpeg.rs` proves the losslessness claim. For every orientation, and for baseline and progressive files with 4:4:4, 4:2:2 and 4:2:0 sampling, the coefficients in the output file match the transformed input coefficients exactly.

## The denoising network

`web/models/denoise.bin` (~180 KB) contains the weights of a 6-layer, 32-feature convolutional network. The network predicts noise and is told how strong the noise is and how big the grain is. It was trained from scratch on a CPU by `core/examples/train_denoiser.rs`, which uses no ML framework: the forward pass, backpropagation and the Adam optimiser are all written out in that file.

The training data is the [Kodak Lossless True Color Image Suite](https://r0k.us/graphics/kodak/), released by Kodak for unrestricted use. kodim22–24 are held out for testing. The images are halved in resolution to remove their own film grain, then given synthetic grain:
- grain size from 0 (digital) to 3 px;
- correlation between colour channels from independent dye layers up to monochrome silver grain;
- amplitude that varies with tone, plus scanner noise;
- half the patches are inverted and tone-curved to look like negatives in density space.

To retrain the network:

```sh
# put kodim01.png … kodim24.png in ./kodak, then:
RUSTFLAGS="-C target-cpu=native" cargo run --release --manifest-path core/Cargo.toml \
  --example train_denoiser -- kodak web/models/denoise.bin 8000     # ~45 min on 4 cores
cargo run --release --manifest-path core/Cargo.toml --example denoise_eval -- kodak web/models/denoise.bin
```

`denoise_eval` compares all three methods by PSNR on the held-out images, for digital noise and four film-grain conditions, and prints the measured grain beside each.

Results with the shipped weights, as mean PSNR in dB over kodim22–24, using the app's automatic grain measurement:

| condition | grainy input | wavelet | NLM | neural |
|---|---|---|---|---|
| digital noise (σ 15) | 24.7 | 29.2 | **29.8** | 26.4 |
| ISO 400 colour (σ 12, 1 px) | 26.6 | 28.9 | 29.0 | **29.5** |
| ISO 800 colour (σ 18, 1.5 px) | 23.1 | 26.6 | 26.0 | **27.3** |
| ISO 3200 B&W (σ 22, 2 px) | 21.4 | 25.2 | 23.4 | **25.8** |
| pushed colour (σ 30, 2.5 px) | 18.8 | 22.9 | 21.4 | **23.0** |

In short, use **neural for film grain** and **NLM for digital noise**. Wavelet is the fastest, and close behind on film. Pass `gradcheck` as the step count to check the hand-written gradients against finite differences. In the browser console, `await darkroom.engine.call('compareNeural', { docId: darkroom.app.current.id })` checks that the WebGPU and wasm versions of the network produce the same output.

The main preview shows a screen-sized proxy, and downscaling already averages most noise away. Use the 100 % loupe (Z) to judge noise reduction properly.

## Limitations and ideas

- Colour management assumes sRGB input and output. Embedded ICC profiles are ignored, and wide-gamut scans will look slightly off.
- There is no undo history yet. You can reset a single slider by double-clicking it, or reset the whole image.
- Fine rotation (straightening), dust removal and local adjustments aren't implemented.
- The X-Trans and other non-Bayer sensors use a simple demosaic.
- Grain is modelled as Gaussian-shaped noise with one amount and one size per image, measured in the working space. Real grain also varies with density, and the model only partly captures that. Grain is a statistical texture, so no method separates it perfectly from fine image detail; the strength sliders exist so you can keep some.
- Non-local means is the weakest method on coarse grain, because the clumps are nearly as big as its patches. Use wavelet or neural for film.
- The neural network is small so that it fits the browser and the CPU training budget. Larger published models (DnCNN, FFDNet, NAFNet) denoise better, but are 10–100× heavier.
- Lossless JPEG doesn't handle arithmetic-coded or 12-bit JPEGs, which are rare.

## Licence

Darkroom's own code is released under the [MIT licence](LICENSE).

The camera raw decoder it uses, [rawler](https://github.com/dnglab/dnglab), is licensed under LGPL-2.1. It is compiled into the WebAssembly published on the site (`web/pkg/`), so that compiled bundle also carries rawler's LGPL-2.1 obligations. The complete source and build script are here, so anyone can rebuild it with a modified rawler. Other Rust dependencies are under MIT/Apache-2.0 or similar permissive licences.

The neural network weights in `web/models/` were trained on the Kodak Lossless True Color Image Suite, which Kodak released for unrestricted use.
