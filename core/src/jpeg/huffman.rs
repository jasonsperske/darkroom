//! Huffman coding as used by JPEG (ITU T.81 Annex C, F.2.2.3 and K.2).
//!
//! A table is transmitted as `bits[1..=16]` – how many codes have each length – followed by the
//! symbol values in code order. Codes are *canonical*: consecutive integers within a length,
//! doubling when moving to the next length, so the counts alone define every code.

use super::bitio::BitReader;

pub struct Decoder {
    /// Fast path: indexed by the next 9 bits → `(length << 8) | symbol`, 0 if the code is longer.
    fast: Vec<u16>,
    /// Slow path (spec F.2.2.3): largest code of each length, and where its symbols start.
    maxcode: [i32; 18],
    valptr: [i32; 17],
    mincode: [i32; 17],
    values: Vec<u8>,
}

const FAST_BITS: u32 = 9;

impl Decoder {
    pub fn new(bits: &[u8; 17], values: &[u8]) -> Self {
        let mut d = Decoder { fast: vec![0; 1 << FAST_BITS], maxcode: [-1; 18], valptr: [0; 17], mincode: [0; 17], values: values.to_vec() };
        let mut code = 0i32;
        let mut k = 0usize;
        for len in 1..=16 {
            let count = bits[len] as usize;
            d.valptr[len] = k as i32;
            d.mincode[len] = code;
            for _ in 0..count {
                if (len as u32) <= FAST_BITS && k < values.len() {
                    // Every 9-bit pattern that starts with this code decodes to it.
                    let shift = FAST_BITS - len as u32;
                    let base = (code as usize) << shift;
                    for i in 0..(1usize << shift) {
                        d.fast[base + i] = ((len as u16) << 8) | values[k] as u16;
                    }
                }
                code += 1;
                k += 1;
            }
            d.maxcode[len] = if count > 0 { code - 1 } else { -1 };
            code <<= 1;
        }
        d.maxcode[17] = i32::MAX;
        d
    }

    #[inline]
    pub fn decode(&self, r: &mut BitReader) -> Result<u8, String> {
        let e = self.fast[r.peek(FAST_BITS) as usize];
        if e != 0 {
            r.consume((e >> 8) as u32);
            return Ok(e as u8);
        }
        let mut code = 0i32;
        for len in 1..=16 {
            code = (code << 1) | r.bits(1) as i32;
            if code <= self.maxcode[len] {
                let i = self.valptr[len] + code - self.mincode[len];
                return self.values.get(i as usize).copied().ok_or_else(|| "corrupt Huffman code".to_string());
            }
        }
        Err("corrupt Huffman code".into())
    }
}

/// JPEG's sign/magnitude trick: a value of category `s` is sent as `s` raw bits; values whose
/// top bit is 0 are negative (`v − 2^s + 1`).
#[inline]
pub fn extend(v: u32, s: u32) -> i32 {
    if s == 0 {
        0
    } else if v < (1 << (s - 1)) {
        v as i32 - (1 << s) + 1
    } else {
        v as i32
    }
}

/// The "category" of a value: number of bits needed for its magnitude.
#[inline]
pub fn category(v: i32) -> u32 {
    32 - v.unsigned_abs().leading_zeros()
}

/// An encoding table: code and length per symbol.
pub struct Encoder {
    pub code: [u16; 256],
    pub size: [u8; 256],
    /// The `bits`/`values` pair to write into the DHT segment.
    pub bits: [u8; 17],
    pub values: Vec<u8>,
}

impl Encoder {
    /// Build an optimal table for the observed symbol frequencies (spec Annex K.2).
    ///
    /// This is ordinary Huffman tree construction, tracking only code *lengths*, followed by a
    /// fix-up that limits lengths to 16 bits (JPEG's maximum) and reserves the all-ones code.
    pub fn optimal(freq_in: &[u32; 256]) -> Self {
        let mut freq = [0i64; 257];
        for (f, &v) in freq.iter_mut().zip(freq_in.iter()) {
            *f = v as i64;
        }
        if freq.iter().all(|&f| f == 0) {
            freq[0] = 1; // an empty table still needs one code
        }
        freq[256] = 1; // reserved symbol guarantees no code is all ones
        let mut codesize = [0usize; 257];
        let mut others = [-1i32; 257];

        loop {
            // Find the two least-frequent nodes (v1 smallest).
            let mut v1 = -1i32;
            let mut v2 = -1i32;
            for i in 0..257 {
                if freq[i] > 0 && (v1 < 0 || freq[i] <= freq[v1 as usize]) {
                    v1 = i as i32;
                }
            }
            for i in 0..257 {
                if freq[i] > 0 && i as i32 != v1 && (v2 < 0 || freq[i] <= freq[v2 as usize]) {
                    v2 = i as i32;
                }
            }
            if v2 < 0 {
                break;
            }
            let (v1, v2) = (v1 as usize, v2 as usize);
            freq[v1] += freq[v2];
            freq[v2] = 0;
            // Every leaf under both merged nodes gets one bit longer.
            let mut n = v1;
            codesize[n] += 1;
            while others[n] >= 0 {
                n = others[n] as usize;
                codesize[n] += 1;
            }
            others[n] = v2 as i32;
            let mut n = v2;
            codesize[n] += 1;
            while others[n] >= 0 {
                n = others[n] as usize;
                codesize[n] += 1;
            }
        }

        let mut bits = [0u32; 33];
        for &s in codesize.iter() {
            if s > 0 {
                bits[s.min(32)] += 1;
            }
        }
        // Limit code lengths to 16 (K.2, Figure K.3).
        for i in (17..=32).rev() {
            while bits[i] > 0 {
                let mut j = i - 2;
                while bits[j] == 0 {
                    j -= 1;
                }
                bits[i] -= 2;
                bits[i - 1] += 1;
                bits[j + 1] += 2;
                bits[j] -= 1;
            }
        }
        // Remove the reserved symbol from the longest length.
        let mut i = 16;
        while bits[i] == 0 {
            i -= 1;
        }
        bits[i] -= 1;

        // Symbols sorted by original code size (then value) get the canonical lengths in order.
        let mut values = Vec::new();
        for len in 1..=32 {
            for sym in 0..256 {
                if codesize[sym] == len {
                    values.push(sym as u8);
                }
            }
        }
        let mut b = [0u8; 17];
        for len in 1..=16 {
            b[len] = bits[len] as u8;
        }
        Self::from_spec(&b, &values)
    }

    /// Generate codes from a `bits`/`values` specification (spec C.1 / C.2).
    pub fn from_spec(bits: &[u8; 17], values: &[u8]) -> Self {
        let mut e = Encoder { code: [0; 256], size: [0; 256], bits: *bits, values: values.to_vec() };
        let mut code = 0u16;
        let mut k = 0;
        for len in 1..=16 {
            for _ in 0..bits[len] {
                let sym = values[k] as usize;
                e.code[sym] = code;
                e.size[sym] = len as u8;
                code += 1;
                k += 1;
            }
            code <<= 1;
        }
        e
    }
}

#[cfg(test)]
mod tests {
    use super::super::bitio::{BitReader, BitWriter};
    use super::*;

    #[test]
    fn optimal_table_roundtrips() {
        let mut freq = [0u32; 256];
        for (i, f) in freq.iter_mut().enumerate().take(200) {
            *f = (i as u32 * 7919) % 1000 + 1;
        }
        let enc = Encoder::optimal(&freq);
        assert!(enc.bits.iter().skip(1).map(|&b| b as usize).sum::<usize>() == 200);
        let mut w = BitWriter::new();
        for sym in 0..200u32 {
            w.put(enc.code[sym as usize] as u32, enc.size[sym as usize] as u32);
        }
        let data = w.finish();
        let dec = Decoder::new(&enc.bits, &enc.values);
        let mut r = BitReader::new(&data);
        for sym in 0..200u8 {
            assert_eq!(dec.decode(&mut r).unwrap(), sym);
        }
    }

    #[test]
    fn extend_and_category() {
        for v in [-1023, -5, -1, 1, 2, 3, 255, 1023] {
            let s = category(v);
            let bits = if v < 0 { (v - 1) as u32 & ((1 << s) - 1) } else { v as u32 };
            assert_eq!(extend(bits, s), v);
        }
    }
}
