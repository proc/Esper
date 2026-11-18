//! Pseudo-legal move generation (check-evasion aware) and perft.

use crate::bitboard::*;
use crate::position::*;

pub const MAX_MOVES: usize = 256;

#[derive(Clone, Copy)]
pub struct MoveList {
    pub moves: [Move; MAX_MOVES],
    pub len: usize,
}

impl Default for MoveList {
    fn default() -> Self {
        Self::new()
    }
}

impl MoveList {
    #[inline(always)]
    pub fn new() -> MoveList {
        MoveList { moves: [Move::NONE; MAX_MOVES], len: 0 }
    }
    #[inline(always)]
    fn push(&mut self, m: Move) {
        debug_assert!(self.len < MAX_MOVES);
        unsafe {
            *self.moves.get_unchecked_mut(self.len) = m;
        }
        self.len += 1;
    }
    #[inline(always)]
    pub fn as_slice(&self) -> &[Move] {
        &self.moves[..self.len]
    }
}

#[inline(always)]
fn push_promos(list: &mut MoveList, from: usize, to: usize, capture: bool) {
    let base = FLAG_PROMO | if capture { FLAG_CAPTURE } else { 0 };
    list.push(Move::new(from, to, base | 3));
    list.push(Move::new(from, to, base));
    list.push(Move::new(from, to, base | 2));
    list.push(Move::new(from, to, base | 1));
}

/// Generate pseudo-legal moves. `noisy` selects captures and promotions,
/// `quiet` everything else. When in check only evasions are produced.
pub fn generate(pos: &Position, list: &mut MoveList, noisy: bool, quiet: bool) {
    let us = pos.stm;
    let them = us ^ 1;
    let occ = pos.occ();
    let ours = pos.by_color[us];
    let theirs = pos.by_color[them];
    let empty = !occ;
    let ksq = pos.king_sq(us);

    // King moves are never restricted by the evasion target.
    let mut kmask = 0;
    if noisy {
        kmask |= theirs;
    }
    if quiet {
        kmask |= empty;
    }
    let mut katt = king_attacks(ksq) & kmask;
    while katt != 0 {
        let to = pop_lsb(&mut katt);
        list.push(Move::new(ksq, to, if theirs & bit(to) != 0 { FLAG_CAPTURE } else { FLAG_QUIET }));
    }

    let mut target = !0u64;
    if pos.checkers != 0 {
        if more_than_one(pos.checkers) {
            return;
        }
        target = between(ksq, lsb(pos.checkers)) | pos.checkers;
    }

    // Pawns
    let pawns = pos.pieces(us, PAWN);
    let (rank7, rank3) = if us == WHITE { (RANK_7, RANK_3) } else { (RANK_2, RANK_6) };
    let fwd = |b: u64| if us == WHITE { b << 8 } else { b >> 8 };
    let back = |s: usize, d: usize| if us == WHITE { s - d } else { s + d };
    let promo_pawns = pawns & rank7;
    let other_pawns = pawns & !rank7;

    if noisy {
        // Promotions (pushes and captures)
        let mut b = fwd(promo_pawns) & empty & target;
        while b != 0 {
            let to = pop_lsb(&mut b);
            push_promos(list, back(to, 8), to, false);
        }
        let mut pp = promo_pawns;
        while pp != 0 {
            let from = pop_lsb(&mut pp);
            let mut att = pawn_attacks(us, from) & theirs & target;
            while att != 0 {
                push_promos(list, from, pop_lsb(&mut att), true);
            }
        }
        // Captures
        let mut op = other_pawns;
        while op != 0 {
            let from = pop_lsb(&mut op);
            let mut att = pawn_attacks(us, from) & theirs & target;
            while att != 0 {
                list.push(Move::new(from, pop_lsb(&mut att), FLAG_CAPTURE));
            }
        }
        // En passant
        if pos.ep != NO_SQ {
            let ep = pos.ep as usize;
            // When in check, ep is only useful if it captures the checker or blocks.
            if pos.checkers == 0 || target & (bit(ep) | bit(ep ^ 8)) != 0 {
                let mut att = pawn_attacks(them, ep) & other_pawns;
                while att != 0 {
                    list.push(Move::new(pop_lsb(&mut att), ep, FLAG_EP));
                }
            }
        }
    }
    if quiet {
        let single = fwd(other_pawns) & empty;
        let mut b = single & target;
        while b != 0 {
            let to = pop_lsb(&mut b);
            list.push(Move::new(back(to, 8), to, FLAG_QUIET));
        }
        let mut d = fwd(single & rank3) & empty & target;
        while d != 0 {
            let to = pop_lsb(&mut d);
            list.push(Move::new(back(to, 16), to, FLAG_DOUBLE));
        }
    }

    let mut mask = 0;
    if noisy {
        mask |= theirs;
    }
    if quiet {
        mask |= empty;
    }
    mask &= target;

    let mut knights = pos.pieces(us, KNIGHT);
    while knights != 0 {
        let from = pop_lsb(&mut knights);
        let mut att = knight_attacks(from) & mask;
        while att != 0 {
            let to = pop_lsb(&mut att);
            list.push(Move::new(from, to, if theirs & bit(to) != 0 { FLAG_CAPTURE } else { FLAG_QUIET }));
        }
    }
    let mut diag = ours & (pos.by_type[BISHOP] | pos.by_type[QUEEN]);
    while diag != 0 {
        let from = pop_lsb(&mut diag);
        let mut att = bishop_attacks(from, occ) & mask;
        while att != 0 {
            let to = pop_lsb(&mut att);
            list.push(Move::new(from, to, if theirs & bit(to) != 0 { FLAG_CAPTURE } else { FLAG_QUIET }));
        }
    }
    let mut orth = ours & (pos.by_type[ROOK] | pos.by_type[QUEEN]);
    while orth != 0 {
        let from = pop_lsb(&mut orth);
        let mut att = rook_attacks(from, occ) & mask;
        while att != 0 {
            let to = pop_lsb(&mut att);
            list.push(Move::new(from, to, if theirs & bit(to) != 0 { FLAG_CAPTURE } else { FLAG_QUIET }));
        }
    }

    if quiet && pos.checkers == 0 {
        if pos.can_castle(true) {
            list.push(Move::new(ksq, ksq + 2, FLAG_KCASTLE));
        }
        if pos.can_castle(false) {
            list.push(Move::new(ksq, ksq - 2, FLAG_QCASTLE));
        }
    }
}

pub fn legal_moves(pos: &Position) -> Vec<Move> {
    let mut list = MoveList::new();
    generate(pos, &mut list, true, true);
    list.as_slice().iter().copied().filter(|&m| pos.is_legal(m)).collect()
}

pub fn has_legal_move(pos: &Position) -> bool {
    let mut list = MoveList::new();
    generate(pos, &mut list, true, true);
    list.as_slice().iter().any(|&m| pos.is_legal(m))
}

pub fn parse_move(pos: &Position, s: &str) -> Option<Move> {
    let s = s.to_ascii_lowercase();
    legal_moves(pos).into_iter().find(|m| m.to_uci() == s)
}

/// Standard algebraic notation for a legal move.
pub fn to_san(pos: &Position, m: Move) -> String {
    let (from, to) = (m.from(), m.to());
    let sq = |s: usize| format!("{}{}", (b'a' + (s & 7) as u8) as char, (b'1' + (s >> 3) as u8) as char);
    let mut san = String::new();
    if m.flag() == FLAG_KCASTLE {
        san.push_str("O-O");
    } else if m.flag() == FLAG_QCASTLE {
        san.push_str("O-O-O");
    } else {
        let pt = piece_type(pos.board[from]);
        if pt == PAWN {
            if m.is_capture() {
                san.push((b'a' + (from & 7) as u8) as char);
            }
        } else {
            san.push(b" PNBRQK"[pt] as char);
            // Disambiguate against other pieces of the same kind that can go there.
            let others: Vec<usize> = legal_moves(pos)
                .into_iter()
                .filter(|o| o.to() == to && o.from() != from && piece_type(pos.board[o.from()]) == pt)
                .map(|o| o.from())
                .collect();
            if !others.is_empty() {
                if others.iter().all(|&o| o & 7 != from & 7) {
                    san.push((b'a' + (from & 7) as u8) as char);
                } else if others.iter().all(|&o| o >> 3 != from >> 3) {
                    san.push((b'1' + (from >> 3) as u8) as char);
                } else {
                    san.push_str(&sq(from));
                }
            }
        }
        if m.is_capture() {
            san.push('x');
        }
        san.push_str(&sq(to));
        if m.is_promo() {
            san.push('=');
            san.push(b" PNBRQK"[m.promo_type()] as char);
        }
    }
    let next = pos.make(m);
    if next.in_check() {
        san.push(if legal_moves(&next).is_empty() { '#' } else { '+' });
    }
    san
}

/// Find the legal move a SAN string refers to (check marks and annotations are
/// ignored, as is surplus disambiguation).
pub fn parse_san(pos: &Position, san: &str) -> Option<Move> {
    let clean = |s: &str| s.chars().filter(|c| !"+#!?=".contains(*c)).collect::<String>().replace('0', "O");
    let want = clean(san);
    let moves = legal_moves(pos);
    if let Some(&m) = moves.iter().find(|&&m| clean(&to_san(pos, m)) == want) {
        return Some(m);
    }
    // Fully disambiguated form, e.g. "Nb1d2".
    moves.into_iter().find(|&m| {
        let pt = piece_type(pos.board[m.from()]);
        if pt == PAWN || m.is_castle() {
            return false;
        }
        let u = m.to_uci();
        let full = format!("{}{}{}{}", b" PNBRQK"[pt] as char, &u[..2], if m.is_capture() { "x" } else { "" }, &u[2..4]);
        full == want
    })
}

pub fn perft(pos: &Position, depth: u32) -> u64 {
    let mut list = MoveList::new();
    generate(pos, &mut list, true, true);
    let mut nodes = 0;
    for &m in list.as_slice() {
        if !pos.is_legal(m) {
            continue;
        }
        if depth <= 1 {
            nodes += 1;
        } else {
            nodes += perft(&pos.make(m), depth - 1);
        }
    }
    nodes
}
