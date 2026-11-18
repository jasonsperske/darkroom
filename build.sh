#!/usr/bin/env bash
# Build the WebAssembly core into web/pkg/. Two variants are produced:
#   web/pkg/simd/      – uses WebAssembly SIMD (all current browsers)
#   web/pkg/baseline/  – no SIMD, for older engines; picked automatically at runtime
# Requirements: rustup target add wasm32-unknown-unknown; cargo install wasm-bindgen-cli --version 0.2.100
# Optional: wasm-opt (binaryen, e.g. `npm i -g binaryen`) for a ~25 % smaller, faster build.
set -euo pipefail
cd "$(dirname "$0")/core"

GETRANDOM='--cfg getrandom_backend="wasm_js"'

build() { # $1 = variant name, $2 = RUSTFLAGS
  local target_dir="target/$1"
  RUSTFLAGS="$2" cargo build --release --target wasm32-unknown-unknown --target-dir "$target_dir"
  local wasm="$target_dir/wasm32-unknown-unknown/release/darkroom_core.wasm"
  rm -rf "../web/pkg/$1"
  wasm-bindgen --target web --no-typescript --out-dir "../web/pkg/$1" "$wasm"
  if command -v wasm-opt >/dev/null; then
    # wasm-opt reads the enabled features (SIMD etc.) from the module itself.
    wasm-opt -O3 "../web/pkg/$1/darkroom_core_bg.wasm" -o "../web/pkg/$1/darkroom_core_bg.wasm" \
      || echo "warning: wasm-opt failed; keeping unoptimised build" >&2
  fi
  echo "built web/pkg/$1 ($(du -h "../web/pkg/$1/darkroom_core_bg.wasm" | cut -f1))"
}

build simd     "-C target-feature=+simd128 $GETRANDOM"
build baseline "$GETRANDOM"
