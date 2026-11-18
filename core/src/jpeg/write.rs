//! Writing a baseline JPEG from coefficients.
//!
//! Two passes over the blocks: the first only *counts* which Huffman symbols occur, so we can
//! build optimal tables (often making the file a few % smaller than the original); the second
//! emits the bits. Component 0 (luma) uses table 0, the others share table 1.

use super::bitio::BitWriter;
use super::exif;
use super::huffman::{category, Encoder};
use super::parse::JpegFile;
use super::ZIGZAG;

/// Something that consumes Huffman symbols + extra bits: either the counter or the writer.
trait Sink {
    fn dc(&mut self, table: usize, symbol: u8, bits: u32, nbits: u32);
    fn ac(&mut self, table: usize, symbol: u8, bits: u32, nbits: u32);
}

struct Counter {
    dc: [[u32; 256]; 2],
    ac: [[u32; 256]; 2],
}

impl Sink for Counter {
    fn dc(&mut self, t: usize, s: u8, _: u32, _: u32) {
        self.dc[t][s as usize] += 1;
    }
    fn ac(&mut self, t: usize, s: u8, _: u32, _: u32) {
        self.ac[t][s as usize] += 1;
    }
}

struct Emitter<'a> {
    w: BitWriter,
    dc: &'a [Encoder; 2],
    ac: &'a [Encoder; 2],
}

impl Sink for Emitter<'_> {
    fn dc(&mut self, t: usize, s: u8, bits: u32, n: u32) {
        let e = &self.dc[t];
        self.w.put(e.code[s as usize] as u32, e.size[s as usize] as u32);
        self.w.put(bits, n);
    }
    fn ac(&mut self, t: usize, s: u8, bits: u32, n: u32) {
        let e = &self.ac[t];
        self.w.put(e.code[s as usize] as u32, e.size[s as usize] as u32);
        self.w.put(bits, n);
    }
}

/// The bits that follow a category-`s` symbol: the value itself, or (value − 1) if negative.
#[inline]
fn magnitude_bits(v: i32, s: u32) -> u32 {
    let v = if v < 0 { v - 1 } else { v };
    (v as u32) & ((1u32 << s) - 1)
}

fn encode_block(sink: &mut impl Sink, table: usize, block: &[i16], pred: &mut i32) {
    // DC: difference from the previous block of the same component.
    let diff = block[0] as i32 - *pred;
    *pred = block[0] as i32;
    let s = category(diff);
    sink.dc(table, s as u8, magnitude_bits(diff, s), s);
    // AC: (run of zeros, category) symbols in zig-zag order.
    let mut run = 0;
    for &z in &ZIGZAG[1..] {
        let v = block[z] as i32;
        if v == 0 {
            run += 1;
            continue;
        }
        while run > 15 {
            sink.ac(table, 0xF0, 0, 0); // ZRL: 16 zeros
            run -= 16;
        }
        let s = category(v);
        sink.ac(table, ((run << 4) | s) as u8, magnitude_bits(v, s), s);
        run = 0;
    }
    if run > 0 {
        sink.ac(table, 0x00, 0, 0); // EOB
    }
}

/// Visit every block in output scan order.
fn encode_scan(f: &JpegFile, sink: &mut impl Sink) {
    let h = &f.header;
    let mut preds = vec![0i32; h.components.len()];
    let table = |c: usize| if c == 0 { 0 } else { 1 };
    if h.components.len() == 1 {
        let (bw, _) = h.padded_blocks(0);
        let (rw, rh) = h.real_blocks(0);
        for by in 0..rh {
            for bx in 0..rw {
                encode_block(sink, 0, &f.coefs[0][(by * bw + bx) * 64..][..64], &mut preds[0]);
            }
        }
        return;
    }
    for my in 0..h.mcus_y() {
        for mx in 0..h.mcus_x() {
            for (c, comp) in h.components.iter().enumerate() {
                let (bw, _) = h.padded_blocks(c);
                for v in 0..comp.v {
                    for u in 0..comp.h {
                        let (bx, by) = (mx * comp.h + u, my * comp.v + v);
                        encode_block(sink, table(c), &f.coefs[c][(by * bw + bx) * 64..][..64], &mut preds[c]);
                    }
                }
            }
        }
    }
}

fn segment(out: &mut Vec<u8>, marker: u8, payload: &[u8]) {
    out.extend_from_slice(&[0xFF, marker]);
    out.extend_from_slice(&((payload.len() + 2) as u16).to_be_bytes());
    out.extend_from_slice(payload);
}

pub fn write(f: &JpegFile) -> Vec<u8> {
    let h = &f.header;

    // Pass 1: statistics → optimal Huffman tables.
    let mut counter = Counter { dc: [[0; 256]; 2], ac: [[0; 256]; 2] };
    encode_scan(f, &mut counter);
    let dc = [Encoder::optimal(&counter.dc[0]), Encoder::optimal(&counter.dc[1])];
    let ac = [Encoder::optimal(&counter.ac[0]), Encoder::optimal(&counter.ac[1])];

    let mut out = vec![0xFF, 0xD8];

    // APPn / COM copied verbatim, except that EXIF orientation is reset: the rotation is now
    // baked into the pixels, so a viewer must not rotate again.
    for (marker, payload) in &f.segments {
        if *marker == 0xE1 && payload.starts_with(b"Exif\0\0") {
            let mut p = payload.clone();
            exif::set_orientation(&mut p, 1);
            segment(&mut out, *marker, &p);
        } else {
            segment(&mut out, *marker, payload);
        }
    }

    // Quantisation tables (in zig-zag order). 16-bit tables need the extended SOF1 header.
    let mut sixteen_bit = false;
    for (i, t) in f.qtables.iter().enumerate() {
        let Some(t) = t else { continue };
        if !h.components.iter().any(|c| c.tq == i) {
            continue;
        }
        let wide = t.iter().any(|&q| q > 255);
        sixteen_bit |= wide;
        let mut p = vec![((wide as u8) << 4) | i as u8];
        for &z in ZIGZAG.iter() {
            if wide {
                p.extend_from_slice(&t[z].to_be_bytes());
            } else {
                p.push(t[z] as u8);
            }
        }
        segment(&mut out, 0xDB, &p);
    }

    // Frame header.
    let mut sof = vec![8];
    sof.extend_from_slice(&(h.height as u16).to_be_bytes());
    sof.extend_from_slice(&(h.width as u16).to_be_bytes());
    sof.push(h.components.len() as u8);
    for c in &h.components {
        sof.extend_from_slice(&[c.id, ((c.h as u8) << 4) | c.v as u8, c.tq as u8]);
    }
    segment(&mut out, if sixteen_bit { 0xC1 } else { 0xC0 }, &sof);

    // Huffman tables.
    let tables_used = if h.components.len() > 1 { 2 } else { 1 };
    for t in 0..tables_used {
        for (class, enc) in [(0u8, &dc[t]), (1u8, &ac[t])] {
            let mut p = vec![(class << 4) | t as u8];
            p.extend_from_slice(&enc.bits[1..17]);
            p.extend_from_slice(&enc.values);
            segment(&mut out, 0xC4, &p);
        }
    }

    // Scan header: all components, full spectrum (Ss=0, Se=63, Ah=Al=0).
    let mut sos = vec![h.components.len() as u8];
    for (i, c) in h.components.iter().enumerate() {
        let t = if i == 0 { 0 } else { 1 };
        sos.extend_from_slice(&[c.id, (t << 4) | t]);
    }
    sos.extend_from_slice(&[0, 63, 0]);
    segment(&mut out, 0xDA, &sos);

    // Pass 2: entropy-coded data.
    let mut emitter = Emitter { w: BitWriter::new(), dc: &dc, ac: &ac };
    encode_scan(f, &mut emitter);
    out.extend_from_slice(&emitter.w.finish());
    out.extend_from_slice(&[0xFF, 0xD9]);
    out
}
