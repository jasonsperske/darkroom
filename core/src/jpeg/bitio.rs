//! Bit-level I/O for JPEG entropy-coded data.
//!
//! Huffman codes are packed MSB-first. Because `0xFF` introduces a marker, an encoder must
//! write `0xFF 0x00` whenever the data itself contains a `0xFF` byte ("byte stuffing"), and a
//! decoder must drop the `0x00`. Restart markers (`0xFF 0xD0`…`0xD7`) may interrupt the data
//! at known intervals; at each one the bit stream is byte-aligned and predictions are reset.

pub struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    /// Bits are kept left-aligned in a 64-bit buffer.
    buf: u64,
    nbits: u32,
    /// We've hit a marker: feed zeros from now on (the decoder will stop soon).
    at_marker: bool,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        BitReader { data, pos: 0, buf: 0, nbits: 0, at_marker: false }
    }

    #[inline]
    fn fill(&mut self) {
        while self.nbits <= 56 {
            let byte = if self.at_marker || self.pos >= self.data.len() {
                0
            } else {
                let b = self.data[self.pos];
                if b == 0xFF {
                    let next = self.data.get(self.pos + 1).copied().unwrap_or(0);
                    if next == 0x00 {
                        self.pos += 2; // stuffed byte
                        0xFF
                    } else {
                        self.at_marker = true; // don't consume the marker
                        0
                    }
                } else {
                    self.pos += 1;
                    b
                }
            };
            self.buf |= (byte as u64) << (56 - self.nbits);
            self.nbits += 8;
        }
    }

    /// Look at the next `n` (≤ 16) bits without consuming them.
    #[inline]
    pub fn peek(&mut self, n: u32) -> u32 {
        self.fill();
        (self.buf >> (64 - n)) as u32
    }

    #[inline]
    pub fn consume(&mut self, n: u32) {
        self.buf <<= n;
        self.nbits -= n;
    }

    #[inline]
    pub fn bits(&mut self, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        let v = self.peek(n);
        self.consume(n);
        v
    }

    #[inline]
    pub fn bit(&mut self) -> bool {
        self.bits(1) == 1
    }

    /// Skip to just after the next RSTn marker and reset the bit buffer.
    pub fn restart(&mut self) -> Result<(), String> {
        self.buf = 0;
        self.nbits = 0;
        self.at_marker = false;
        while self.pos + 1 < self.data.len() {
            if self.data[self.pos] == 0xFF && (0xD0..=0xD7).contains(&self.data[self.pos + 1]) {
                self.pos += 2;
                return Ok(());
            }
            self.pos += 1;
        }
        Err("missing restart marker".into())
    }
}

pub struct BitWriter {
    pub out: Vec<u8>,
    acc: u64,
    nbits: u32,
}

impl BitWriter {
    pub fn new() -> Self {
        BitWriter { out: Vec::new(), acc: 0, nbits: 0 }
    }

    /// Append the low `n` bits of `v` (n ≤ 24).
    #[inline]
    pub fn put(&mut self, v: u32, n: u32) {
        if n == 0 {
            return;
        }
        self.acc = (self.acc << n) | (v as u64 & ((1u64 << n) - 1));
        self.nbits += n;
        while self.nbits >= 8 {
            let byte = (self.acc >> (self.nbits - 8)) as u8;
            self.out.push(byte);
            if byte == 0xFF {
                self.out.push(0x00);
            }
            self.nbits -= 8;
        }
    }

    /// Pad the final partial byte with 1-bits, as the standard requires.
    pub fn finish(mut self) -> Vec<u8> {
        if self.nbits > 0 {
            let pad = 8 - self.nbits;
            self.put((1 << pad) - 1, pad);
        }
        self.out
    }
}
