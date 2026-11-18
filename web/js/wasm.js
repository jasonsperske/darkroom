// Loads the WebAssembly core, choosing the SIMD build when the browser supports it.
//
// Feature detection works by asking the engine to *validate* (not run) a tiny module that
// uses one SIMD instruction. This is the same trick as the `wasm-feature-detect` library.

const SIMD_PROBE = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11,
]);

export const simdSupported = (() => {
  try {
    return WebAssembly.validate(SIMD_PROBE);
  } catch {
    return false;
  }
})();

let loading = null;

/** Returns the initialised module namespace (all exported Rust functions). */
export function loadCore() {
  if (!loading) {
    const variant = simdSupported ? 'simd' : 'baseline';
    loading = import(`../pkg/${variant}/darkroom_core.js`).then(async (mod) => {
      await mod.default(); // fetches and instantiates darkroom_core_bg.wasm next to the JS glue
      return mod;
    });
  }
  return loading;
}
