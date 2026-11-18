//! JPEG parsing and entropy decoding down to quantised DCT coefficients.
//!
//! A JPEG file is a sequence of *markers* (`0xFF xx`), most followed by a 2-byte length:
//!
//! | marker   | meaning                                                  |
//! |----------|----------------------------------------------------------|
//! | `D8` SOI | start of image                                           |
//! | `E0–EF`  | APPn: application data – JFIF, EXIF, ICC profile, Adobe  |
//! | `DB` DQT | quantisation tables                                      |
//! | `C0/C1/C2` SOF | frame header: size, components, sampling factors   |
//! | `C4` DHT | Huffman tables                                           |
//! | `DD` DRI | restart interval                                         |
//! | `DA` SOS | start of scan – followed by entropy-coded data           |
//! | `D9` EOI | end of image                                             |
//!
//! A *baseline* JPEG has one scan holding everything. A *progressive* JPEG has many scans, each
//! refining a band of frequencies (`Ss..=Se`) or adding one more bit of precision (`Ah`, `Al`),
//! so we decode all scans into a coefficient buffer and only then work with the result.

use super::bitio::BitReader;
use super::huffman::{extend, Decoder};
use super::ZIGZAG;

#[derive(Clone, Debug)]
pub struct ComponentInfo {
    pub id: u8,
    /// Horizontal / vertical sampling factor (1 or 2 in practice).
    pub h: usize,
    pub v: usize,
    /// Quantisation table index.
    pub tq: usize,
}

#[derive(Clone, Debug)]
pub struct Header {
    pub width: usize,
    pub height: usize,
    pub progressive: bool,
    pub components: Vec<ComponentInfo>,
    pub hmax: usize,
    pub vmax: usize,
}

impl Header {
    pub fn mcu_width(&self) -> usize {
        8 * self.hmax
    }
    pub fn mcu_height(&self) -> usize {
        8 * self.vmax
    }
    pub fn mcus_x(&self) -> usize {
        self.width.div_ceil(self.mcu_width())
    }
    pub fn mcus_y(&self) -> usize {
        self.height.div_ceil(self.mcu_height())
    }
    /// Blocks stored for component `c`, padded to whole MCUs.
    pub fn padded_blocks(&self, c: usize) -> (usize, usize) {
        (self.mcus_x() * self.components[c].h, self.mcus_y() * self.components[c].v)
    }
    /// Blocks that actually cover component `c`'s pixels (used by non-interleaved scans).
    pub fn real_blocks(&self, c: usize) -> (usize, usize) {
        let comp = &self.components[c];
        let w = (self.width * comp.h).div_ceil(self.hmax);
        let h = (self.height * comp.v).div_ceil(self.vmax);
        (w.div_ceil(8), h.div_ceil(8))
    }
}

pub struct JpegFile {
    pub header: Header,
    /// Quantisation tables in natural (row-major) order.
    pub qtables: [Option<[u16; 64]>; 4],
    /// APPn and COM segments to copy into the output unchanged: `(marker, payload)`.
    pub segments: Vec<(u8, Vec<u8>)>,
    /// Per component: `blocks_w * blocks_h * 64` coefficients, natural order within each block.
    pub coefs: Vec<Vec<i16>>,
}

impl JpegFile {
    #[inline]
    pub fn block_mut(&mut self, c: usize, bx: usize, by: usize) -> &mut [i16] {
        let (bw, _) = self.header.padded_blocks(c);
        let i = (by * bw + bx) * 64;
        &mut self.coefs[c][i..i + 64]
    }
}

/// Walk markers: returns (marker, payload range, position after the segment).
fn next_segment(bytes: &[u8], mut pos: usize) -> Result<(u8, usize, usize, usize), String> {
    if bytes.get(pos) != Some(&0xFF) {
        return Err(format!("expected marker at byte {pos}"));
    }
    while bytes.get(pos) == Some(&0xFF) {
        pos += 1; // fill bytes
    }
    let marker = *bytes.get(pos).ok_or("truncated file")?;
    pos += 1;
    if marker == 0xD8 || marker == 0xD9 || (0xD0..=0xD7).contains(&marker) || marker == 0x01 {
        return Ok((marker, pos, pos, pos));
    }
    let len = u16::from_be_bytes([*bytes.get(pos).ok_or("truncated")?, *bytes.get(pos + 1).ok_or("truncated")?]) as usize;
    if len < 2 || pos + len > bytes.len() {
        return Err("truncated segment".into());
    }
    Ok((marker, pos + 2, pos + len, pos + len))
}

fn parse_sof(p: &[u8], marker: u8) -> Result<Header, String> {
    if p.len() < 6 {
        return Err("bad SOF".into());
    }
    if p[0] != 8 {
        return Err(format!("{}-bit JPEG is not supported", p[0]));
    }
    let height = u16::from_be_bytes([p[1], p[2]]) as usize;
    let width = u16::from_be_bytes([p[3], p[4]]) as usize;
    let n = p[5] as usize;
    if width == 0 || height == 0 {
        return Err("JPEG without dimensions (DNL) is not supported".into());
    }
    if p.len() < 6 + n * 3 || n == 0 || n > 4 {
        return Err("bad SOF".into());
    }
    let mut components: Vec<ComponentInfo> = (0..n)
        .map(|i| {
            let b = &p[6 + i * 3..9 + i * 3];
            ComponentInfo { id: b[0], h: (b[1] >> 4).max(1) as usize, v: (b[1] & 15).max(1) as usize, tq: (b[2] & 3) as usize }
        })
        .collect();
    // A single-component image is always coded block by block, whatever its sampling factors.
    if n == 1 {
        components[0].h = 1;
        components[0].v = 1;
    }
    let hmax = components.iter().map(|c| c.h).max().unwrap();
    let vmax = components.iter().map(|c| c.v).max().unwrap();
    Ok(Header { width, height, progressive: marker == 0xC2, components, hmax, vmax })
}

fn unsupported_sof(marker: u8) -> Option<&'static str> {
    match marker {
        0xC3 | 0xC7 | 0xCB | 0xCF => Some("lossless JPEG"),
        0xC5 | 0xC6 | 0xCD | 0xCE => Some("hierarchical JPEG"),
        0xC9 | 0xCA => Some("arithmetic-coded JPEG"),
        _ => None,
    }
}

/// Parse markers only as far as the frame header.
pub fn parse_header(bytes: &[u8]) -> Result<Header, String> {
    let (m, _, _, mut pos) = next_segment(bytes, 0)?;
    if m != 0xD8 {
        return Err("not a JPEG".into());
    }
    loop {
        let (m, a, b, next) = next_segment(bytes, pos)?;
        match m {
            0xC0..=0xC2 => return parse_sof(&bytes[a..b], m),
            m if unsupported_sof(m).is_some() => return Err(format!("{} is not supported", unsupported_sof(m).unwrap())),
            0xD9 | 0xDA => return Err("no frame header".into()),
            _ => pos = next,
        }
    }
}

struct ScanComponent {
    index: usize,
    dc: usize,
    ac: usize,
}

struct Scan {
    comps: Vec<ScanComponent>,
    ss: usize,
    se: usize,
    ah: u32,
    al: u32,
}

/// Decode a whole file to coefficients.
pub fn decode(bytes: &[u8]) -> Result<JpegFile, String> {
    let (m, _, _, mut pos) = next_segment(bytes, 0)?;
    if m != 0xD8 {
        return Err("not a JPEG".into());
    }
    let mut header: Option<Header> = None;
    let mut qtables: [Option<[u16; 64]>; 4] = [None; 4];
    let mut dc_tables: [Option<Decoder>; 4] = [None, None, None, None];
    let mut ac_tables: [Option<Decoder>; 4] = [None, None, None, None];
    let mut segments = Vec::new();
    let mut coefs: Vec<Vec<i16>> = Vec::new();
    let mut restart_interval = 0usize;

    loop {
        let (m, a, b, next) = next_segment(bytes, pos)?;
        let p = &bytes[a..b];
        match m {
            0xE0..=0xEF | 0xFE => segments.push((m, p.to_vec())),
            0xDB => {
                // DQT: possibly several tables, 8- or 16-bit entries, in zig-zag order.
                let mut i = 0;
                while i < p.len() {
                    let (pq, tq) = ((p[i] >> 4) as usize, (p[i] & 15) as usize);
                    i += 1;
                    let mut t = [0u16; 64];
                    for k in 0..64 {
                        t[ZIGZAG[k]] = if pq == 0 {
                            *p.get(i + k).ok_or("bad DQT")? as u16
                        } else {
                            u16::from_be_bytes([*p.get(i + 2 * k).ok_or("bad DQT")?, *p.get(i + 2 * k + 1).ok_or("bad DQT")?])
                        };
                    }
                    i += if pq == 0 { 64 } else { 128 };
                    *qtables.get_mut(tq).ok_or("bad DQT")? = Some(t);
                }
            }
            0xC4 => {
                // DHT: possibly several tables.
                let mut i = 0;
                while i < p.len() {
                    let (class, id) = ((p[i] >> 4) as usize, (p[i] & 15) as usize);
                    let mut counts = [0u8; 17];
                    counts[1..17].copy_from_slice(p.get(i + 1..i + 17).ok_or("bad DHT")?);
                    let total: usize = counts.iter().map(|&c| c as usize).sum();
                    let values = p.get(i + 17..i + 17 + total).ok_or("bad DHT")?;
                    let dec = Decoder::new(&counts, values);
                    if id > 3 {
                        return Err("bad DHT".into());
                    }
                    if class == 0 {
                        dc_tables[id] = Some(dec);
                    } else {
                        ac_tables[id] = Some(dec);
                    }
                    i += 17 + total;
                }
            }
            0xDD => restart_interval = u16::from_be_bytes([p[0], p[1]]) as usize,
            0xC0..=0xC2 => {
                let h = parse_sof(p, m)?;
                coefs = (0..h.components.len())
                    .map(|c| {
                        let (bw, bh) = h.padded_blocks(c);
                        vec![0i16; bw * bh * 64]
                    })
                    .collect();
                header = Some(h);
            }
            m if unsupported_sof(m).is_some() => return Err(format!("{} is not supported", unsupported_sof(m).unwrap())),
            0xCC => return Err("arithmetic-coded JPEG is not supported".into()),
            0xDA => {
                let h = header.as_ref().ok_or("scan before frame header")?;
                let n = p[0] as usize;
                let mut comps = Vec::new();
                for i in 0..n {
                    let id = p[1 + i * 2];
                    let t = p[2 + i * 2];
                    let index = h.components.iter().position(|c| c.id == id).ok_or("scan references unknown component")?;
                    comps.push(ScanComponent { index, dc: (t >> 4) as usize & 3, ac: (t & 15) as usize & 3 });
                }
                let s = &p[1 + n * 2..];
                let scan = Scan { comps, ss: s[0] as usize, se: s[1] as usize, ah: (s[2] >> 4) as u32, al: (s[2] & 15) as u32 };
                if scan.se > 63 || scan.ss > scan.se {
                    return Err("bad scan header".into());
                }
                // Entropy-coded data runs until the next marker that isn't RSTn.
                let mut end = b;
                while end + 1 < bytes.len() {
                    if bytes[end] == 0xFF && bytes[end + 1] != 0 && !(0xD0..=0xD7).contains(&bytes[end + 1]) {
                        break;
                    }
                    end += 1;
                }
                let mut ctx = ScanDecoder {
                    header: h,
                    coefs: &mut coefs,
                    dc_tables: &dc_tables,
                    ac_tables: &ac_tables,
                    reader: BitReader::new(&bytes[b..end]),
                    preds: [0; 4],
                    eobrun: 0,
                };
                ctx.decode_scan(&scan, restart_interval)?;
                pos = end;
                continue;
            }
            0xD9 => break,
            _ => {} // DNL, DHP, EXP, RSTn out of place, … ignore
        }
        pos = next;
        if pos >= bytes.len() {
            break; // tolerate a missing EOI
        }
    }

    let header = header.ok_or("no frame header")?;
    for c in &header.components {
        if qtables[c.tq].is_none() {
            return Err("missing quantisation table".into());
        }
    }
    Ok(JpegFile { header, qtables, segments, coefs })
}

struct ScanDecoder<'a> {
    header: &'a Header,
    coefs: &'a mut Vec<Vec<i16>>,
    dc_tables: &'a [Option<Decoder>; 4],
    ac_tables: &'a [Option<Decoder>; 4],
    reader: BitReader<'a>,
    /// DC predictions per scan component (DC is coded as the difference from the previous block).
    preds: [i32; 4],
    /// Progressive AC: number of remaining blocks whose band is all zero ("end-of-band run").
    eobrun: u32,
}

impl ScanDecoder<'_> {
    fn decode_scan(&mut self, scan: &Scan, restart_interval: usize) -> Result<(), String> {
        let h = self.header;
        let mut unit = 0usize; // MCUs (or blocks, for single-component scans) decoded so far
        let mut restart = |s: &mut Self| -> Result<(), String> {
            if restart_interval > 0 && unit > 0 && unit % restart_interval == 0 {
                s.reader.restart()?;
                s.preds = [0; 4];
                s.eobrun = 0;
            }
            unit += 1;
            Ok(())
        };

        if scan.comps.len() == 1 {
            // Non-interleaved: blocks in raster order over the component's real extent.
            let sc = &scan.comps[0];
            let (bw, bh) = h.real_blocks(sc.index);
            for by in 0..bh {
                for bx in 0..bw {
                    restart(self)?;
                    self.decode_block(scan, 0, sc.index, bx, by)?;
                }
            }
        } else {
            // Interleaved: MCU by MCU; each MCU holds h×v blocks of every component.
            for my in 0..h.mcus_y() {
                for mx in 0..h.mcus_x() {
                    restart(self)?;
                    for (si, sc) in scan.comps.iter().enumerate() {
                        let comp = &h.components[sc.index];
                        for v in 0..comp.v {
                            for u in 0..comp.h {
                                self.decode_block(scan, si, sc.index, mx * comp.h + u, my * comp.v + v)?;
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn decode_block(&mut self, scan: &Scan, si: usize, c: usize, bx: usize, by: usize) -> Result<(), String> {
        let (bw, _) = self.header.padded_blocks(c);
        let start = (by * bw + bx) * 64;
        let block = &mut self.coefs[c][start..start + 64];
        let sc = &scan.comps[si];
        let r = &mut self.reader;

        if !self.header.progressive {
            // Baseline: DC difference, then run-length coded AC coefficients.
            let dc = self.dc_tables[sc.dc].as_ref().ok_or("missing DC table")?;
            let ac = self.ac_tables[sc.ac].as_ref().ok_or("missing AC table")?;
            let s = dc.decode(r)? as u32;
            self.preds[si] += extend(r.bits(s), s);
            block[0] = self.preds[si] as i16;
            let mut k = 1;
            while k < 64 {
                let rs = ac.decode(r)? as usize;
                let (run, size) = (rs >> 4, (rs & 15) as u32);
                if size == 0 {
                    if run != 15 {
                        break; // EOB: rest of block is zero
                    }
                    k += 16; // ZRL: sixteen zeros
                    continue;
                }
                k += run;
                if k > 63 {
                    return Err("corrupt AC data".into());
                }
                block[ZIGZAG[k]] = extend(r.bits(size), size) as i16;
                k += 1;
            }
            return Ok(());
        }

        if scan.ss == 0 {
            // Progressive DC scan.
            if scan.ah == 0 {
                let dc = self.dc_tables[sc.dc].as_ref().ok_or("missing DC table")?;
                let s = dc.decode(r)? as u32;
                self.preds[si] += extend(r.bits(s), s);
                block[0] = (self.preds[si] << scan.al) as i16;
            } else if r.bit() {
                block[0] |= (1 << scan.al) as i16;
            }
            return Ok(());
        }

        let ac = self.ac_tables[sc.ac].as_ref().ok_or("missing AC table")?;
        if scan.ah == 0 {
            // Progressive AC, first pass over this band (spec G.1.2.2).
            if self.eobrun > 0 {
                self.eobrun -= 1;
                return Ok(());
            }
            let mut k = scan.ss;
            while k <= scan.se {
                let rs = ac.decode(r)? as usize;
                let (run, size) = (rs >> 4, (rs & 15) as u32);
                if size == 0 {
                    if run < 15 {
                        self.eobrun = (1 << run) - 1;
                        if run > 0 {
                            self.eobrun += r.bits(run as u32);
                        }
                        break;
                    }
                    k += 16;
                    continue;
                }
                k += run;
                if k > 63 {
                    return Err("corrupt AC data".into());
                }
                block[ZIGZAG[k]] = (extend(r.bits(size), size) << scan.al) as i16;
                k += 1;
            }
            return Ok(());
        }

        // Progressive AC refinement (spec G.1.2.3) – adds one bit of precision. Coefficients that
        // are already non-zero receive a correction bit; new ones are ±1 at this bit position.
        let p1 = 1i16 << scan.al;
        let m1 = -1i16 << scan.al;
        let refine = |r: &mut BitReader, coef: &mut i16| {
            if r.bit() && (*coef & p1) == 0 {
                *coef += if *coef >= 0 { p1 } else { m1 };
            }
        };
        let mut k = scan.ss;
        if self.eobrun == 0 {
            while k <= scan.se {
                let rs = ac.decode(r)? as usize;
                let (mut run, size) = ((rs >> 4) as i32, rs & 15);
                let mut value = 0i16;
                if size != 0 {
                    value = if r.bit() { p1 } else { m1 };
                } else if run != 15 {
                    self.eobrun = 1 << run;
                    if run > 0 {
                        self.eobrun += r.bits(run as u32);
                    }
                    break;
                }
                // Skip `run` zero coefficients, refining non-zero ones on the way.
                while k <= scan.se {
                    let z = ZIGZAG[k];
                    if block[z] != 0 {
                        refine(r, &mut block[z]);
                    } else {
                        if run == 0 {
                            break;
                        }
                        run -= 1;
                    }
                    k += 1;
                }
                if value != 0 && k <= scan.se {
                    block[ZIGZAG[k]] = value;
                }
                k += 1;
            }
        }
        if self.eobrun > 0 {
            // Inside an end-of-band run: only correction bits for existing coefficients.
            while k <= scan.se {
                let z = ZIGZAG[k];
                if block[z] != 0 {
                    refine(r, &mut block[z]);
                }
                k += 1;
            }
            self.eobrun -= 1;
        }
        Ok(())
    }
}
