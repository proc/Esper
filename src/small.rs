//! A small evaluation network that can be trained on Esper's own games with
//! `tools/train.py` (data from `esper datagen`), as an alternative to the
//! Stockfish network.
//!
//!   768 inputs per perspective (colour relative to the perspective x piece
//!   type x square, the board flipped vertically for Black)
//!   -> H hidden units per perspective, squared clipped ReLU
//!   -> one output from both perspectives, side to move first.
//!
//! File layout (little endian), written by `tools/train.py`:
//!
//!   8 bytes  "ESPRNET1"
//!   u32      H        hidden units per perspective
//!   u32      QA       scale of the first layer (activations clip at QA)
//!   u32      QB       scale of the output weights
//!   u32      SCALE    centipawns per unit of network output
//!   i16      feature weights, 768 rows of H
//!   i16      feature biases, H
//!   i16      output weights, 2 * H
//!   i32      output bias, in units of QA * QB
//!
//! The accumulators are kept per search ply and updated lazily: a position that
//! is never evaluated costs nothing, and one that is evaluated is brought up to
//! date from the nearest computed ancestor, a few rows per move.

use crate::position::*;

pub const MAGIC: &[u8; 8] = b"ESPRNET1";
const INPUTS: usize = 768;
/// Internal score units per pawn (search::NORMALIZE_PAWN), over centipawns.
const INTERNAL_PER_100CP: i64 = 290;

pub struct SmallNet {
    hidden: usize,
    qa: i32,
    /// Output numerator and denominator: `raw = t * mul / div`.
    mul: i64,
    div: i64,
    /// `INPUTS` rows of `hidden`.
    ft_w: Vec<i16>,
    ft_b: Vec<i16>,
    out_w: Vec<i16>,
    out_b: i64,
}

impl SmallNet {
    pub fn is_small_file(data: &[u8]) -> bool {
        data.len() >= 8 && &data[..8] == MAGIC
    }

    pub fn from_bytes(data: &[u8]) -> Result<SmallNet, String> {
        if !SmallNet::is_small_file(data) || data.len() < 24 {
            return Err("not a small network file".into());
        }
        let u32_at = |i: usize| u32::from_le_bytes(data[i..i + 4].try_into().unwrap());
        let (hidden, qa, qb, scale) = (u32_at(8) as usize, u32_at(12), u32_at(16), u32_at(20));
        if hidden == 0 || hidden > 8192 || qa == 0 || qa > 255 || qb == 0 || qb > 4096 || scale == 0 || scale > 10_000 {
            return Err("small network header out of range".into());
        }
        let need = 24 + 2 * (INPUTS * hidden + hidden + 2 * hidden) + 4;
        if data.len() < need {
            return Err(format!("small network file too short: {} bytes, need {need}", data.len()));
        }
        let mut at = 24;
        let mut i16s = |n: usize| {
            let v: Vec<i16> =
                data[at..at + 2 * n].chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect();
            at += 2 * n;
            v
        };
        let ft_w = i16s(INPUTS * hidden);
        let ft_b = i16s(hidden);
        let out_w = i16s(2 * hidden);
        let out_b = i32::from_le_bytes(data[need - 4..need].try_into().unwrap()) as i64;
        Ok(SmallNet {
            hidden,
            qa: qa as i32,
            mul: scale as i64 * INTERNAL_PER_100CP,
            div: qa as i64 * qb as i64 * 100,
            ft_w,
            ft_b,
            out_w,
            out_b,
        })
    }

    pub fn description(&self) -> String {
        format!("small network, 768 -> {}x2 -> 1", self.hidden)
    }

    #[inline(always)]
    fn row(&self, feature: usize) -> &[i16] {
        &self.ft_w[feature * self.hidden..(feature + 1) * self.hidden]
    }
}

/// Input index of piece `pc` on `sq`, seen from `persp`.
#[inline(always)]
fn feature(persp: usize, pc: u8, sq: usize) -> usize {
    let rel = piece_color(pc) ^ persp;
    let rsq = if persp == WHITE { sq } else { sq ^ 56 };
    rel * 384 + (piece_type(pc) - 1) * 64 + rsq
}

/// Per-thread accumulators for a `SmallNet`, one pair per search ply.
pub struct SmallState {
    hidden: usize,
    /// `[ply][perspective]`, each `hidden` wide, in one allocation.
    acc: Vec<i16>,
    ok: Vec<bool>,
}

impl SmallState {
    pub fn new(net: &SmallNet, max_ply: usize) -> SmallState {
        SmallState { hidden: net.hidden, acc: vec![0; max_ply * 2 * net.hidden], ok: vec![false; max_ply] }
    }

    /// Start a new stack: the position at ply 0 is unrelated to anything before.
    pub fn reset(&mut self) {
        self.ok[0] = false;
    }

    /// Register the position at ply `j`: its accumulators are not computed yet.
    #[inline(always)]
    pub fn push(&mut self, j: usize) {
        self.ok[j] = false;
    }

    fn refresh(&mut self, net: &SmallNet, pos: &Position, ply: usize) {
        let h = self.hidden;
        for persp in 0..2 {
            let dst = &mut self.acc[(ply * 2 + persp) * h..(ply * 2 + persp + 1) * h];
            dst.copy_from_slice(&net.ft_b);
            let mut occ = pos.by_color[WHITE] | pos.by_color[BLACK];
            while occ != 0 {
                let sq = occ.trailing_zeros() as usize;
                occ &= occ - 1;
                for (d, w) in dst.iter_mut().zip(net.row(feature(persp, pos.board[sq], sq))) {
                    *d = d.wrapping_add(*w);
                }
            }
        }
        self.ok[ply] = true;
    }

    /// Ply `j` from ply `j - 1`: only the squares the move touched differ.
    fn step(&mut self, net: &SmallNet, p0: &Position, p1: &Position, j: usize) {
        let h = self.hidden;
        let mut changed = (p0.by_color[0] ^ p1.by_color[0]) | (p0.by_color[1] ^ p1.by_color[1]);
        for pt in PAWN..=KING {
            changed |= p0.by_type[pt] ^ p1.by_type[pt];
        }
        let (lo, hi) = self.acc.split_at_mut(j * 2 * h);
        let prev = &lo[(j - 1) * 2 * h..];
        for persp in 0..2 {
            let dst = &mut hi[persp * h..(persp + 1) * h];
            dst.copy_from_slice(&prev[persp * h..(persp + 1) * h]);
            let mut c = changed;
            while c != 0 {
                let sq = c.trailing_zeros() as usize;
                c &= c - 1;
                let (old, new) = (p0.board[sq], p1.board[sq]);
                if old != 0 {
                    for (d, w) in dst.iter_mut().zip(net.row(feature(persp, old, sq))) {
                        *d = d.wrapping_sub(*w);
                    }
                }
                if new != 0 {
                    for (d, w) in dst.iter_mut().zip(net.row(feature(persp, new, sq))) {
                        *d = d.wrapping_add(*w);
                    }
                }
            }
        }
        self.ok[j] = true;
    }

    /// Network output for `stack[ply]` from the side to move's point of view, in
    /// the search's internal score units.
    pub fn evaluate(&mut self, net: &SmallNet, stack: &[Position], ply: usize) -> i32 {
        let mut k = ply;
        while !self.ok[k] && k > 0 {
            k -= 1;
        }
        if !self.ok[k] {
            self.refresh(net, &stack[k], k);
        }
        for j in k + 1..=ply {
            self.step(net, &stack[j - 1], &stack[j], j);
        }
        let h = self.hidden;
        let stm = stack[ply].stm;
        let mut sum = 0i64;
        for (half, persp) in [stm, stm ^ 1].into_iter().enumerate() {
            let acc = &self.acc[(ply * 2 + persp) * h..(ply * 2 + persp + 1) * h];
            let w = &net.out_w[half * h..(half + 1) * h];
            for (&x, &y) in acc.iter().zip(w) {
                let v = (x as i32).clamp(0, net.qa);
                sum += (v * v) as i64 * y as i64;
            }
        }
        let t = sum / net.qa as i64 + net.out_b;
        (t * net.mul / net.div).clamp(-30_000, 30_000) as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::movegen::legal_moves;

    /// A network with pseudo-random weights, in file form.
    fn random_net(hidden: usize) -> Vec<u8> {
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut rnd = move |range: i32| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            ((seed >> 20) as i32).rem_euclid(2 * range + 1) - range
        };
        let mut d = MAGIC.to_vec();
        for v in [hidden as u32, 255, 64, 400] {
            d.extend(v.to_le_bytes());
        }
        for _ in 0..INPUTS * hidden + hidden {
            d.extend((rnd(60) as i16).to_le_bytes());
        }
        for _ in 0..2 * hidden {
            d.extend((rnd(90) as i16).to_le_bytes());
        }
        d.extend(rnd(5000).to_le_bytes());
        d
    }

    #[test]
    fn incremental_matches_refresh() {
        crate::bitboard::init();
        let net = SmallNet::from_bytes(&random_net(64)).unwrap();
        let mut inc = SmallState::new(&net, 300);
        let mut fresh = SmallState::new(&net, 2);
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut values = std::collections::HashSet::new();
        for _ in 0..20 {
            let mut stack = vec![Position::startpos()];
            inc.reset();
            for ply in 0..250 {
                let moves = legal_moves(&stack[ply]);
                if moves.is_empty() {
                    break;
                }
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                stack.push(stack[ply].make(moves[(seed % moves.len() as u64) as usize]));
                inc.push(ply + 1);
                // Skip some plies so multi-ply catch-up is exercised.
                if seed & 0x300 != 0 {
                    let a = inc.evaluate(&net, &stack, ply + 1);
                    fresh.reset();
                    let b = fresh.evaluate(&net, &stack[ply + 1..], 0);
                    assert_eq!(a, b, "{}", stack[ply + 1].to_fen());
                    values.insert(a);
                }
            }
        }
        assert!(values.len() > 100, "evaluations should vary");
    }

    #[test]
    fn rejects_bad_files() {
        assert!(SmallNet::from_bytes(b"ESPRNET1").is_err());
        assert!(SmallNet::from_bytes(&random_net(16)[..2000]).is_err());
        assert!(!SmallNet::is_small_file(b"not a net"));
    }
}
