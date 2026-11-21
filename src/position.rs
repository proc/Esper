//! Board representation, make-move (copy-make), legality, SEE and FEN handling.

use crate::bitboard::*;

pub const WHITE: usize = 0;
pub const BLACK: usize = 1;

pub const PAWN: usize = 1;
pub const KNIGHT: usize = 2;
pub const BISHOP: usize = 3;
pub const ROOK: usize = 4;
pub const QUEEN: usize = 5;
pub const KING: usize = 6;

pub const NO_SQ: u8 = 64;
pub const START_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

/// A piece is `color << 3 | piece_type`; 0 is an empty square.
#[inline(always)]
pub fn make_piece(color: usize, pt: usize) -> u8 {
    (color << 3 | pt) as u8
}
#[inline(always)]
pub fn piece_type(pc: u8) -> usize {
    (pc & 7) as usize
}
#[inline(always)]
pub fn piece_color(pc: u8) -> usize {
    (pc >> 3) as usize
}

/// Piece values for static exchange evaluation, in centipawns: Stockfish's
/// PieceValue (208, 781, 825, 1276, 2538) rescaled to 100 per pawn.
pub const SEE_VALUE: [i32; 8] = [0, 100, 375, 397, 613, 1220, 20000, 0];

#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct Move(pub u16);

pub const FLAG_QUIET: u16 = 0;
pub const FLAG_DOUBLE: u16 = 1;
pub const FLAG_KCASTLE: u16 = 2;
pub const FLAG_QCASTLE: u16 = 3;
pub const FLAG_CAPTURE: u16 = 4;
pub const FLAG_EP: u16 = 5;
pub const FLAG_PROMO: u16 = 8;

impl Move {
    pub const NONE: Move = Move(0);
    #[inline(always)]
    pub fn new(from: usize, to: usize, flag: u16) -> Move {
        Move((from as u16) | ((to as u16) << 6) | (flag << 12))
    }
    #[inline(always)]
    pub fn from(self) -> usize {
        (self.0 & 63) as usize
    }
    #[inline(always)]
    pub fn to(self) -> usize {
        ((self.0 >> 6) & 63) as usize
    }
    #[inline(always)]
    pub fn flag(self) -> u16 {
        self.0 >> 12
    }
    #[inline(always)]
    pub fn is_capture(self) -> bool {
        self.0 & (4 << 12) != 0
    }
    #[inline(always)]
    pub fn is_promo(self) -> bool {
        self.0 & (8 << 12) != 0
    }
    #[inline(always)]
    pub fn is_castle(self) -> bool {
        let f = self.flag();
        f == FLAG_KCASTLE || f == FLAG_QCASTLE
    }
    /// Captures and promotions.
    /// Stockfish's `capture_stage`: captures (en passant included) and queen promotions.
    #[inline(always)]
    pub fn is_capture_stage(self) -> bool {
        self.is_capture() || (self.is_promo() && self.promo_type() == QUEEN)
    }
    #[inline(always)]
    pub fn is_noisy(self) -> bool {
        self.0 & (12 << 12) != 0
    }
    #[inline(always)]
    pub fn promo_type(self) -> usize {
        ((self.flag() & 3) + 2) as usize
    }
    #[inline(always)]
    pub fn is_none(self) -> bool {
        self.0 == 0
    }
    /// Index into from/to (butterfly) tables.
    #[inline(always)]
    pub fn from_to(self) -> usize {
        (self.0 & 0xFFF) as usize
    }
    pub fn to_uci(self) -> String {
        if self.is_none() {
            return "0000".to_string();
        }
        let mut s = String::with_capacity(5);
        for sq in [self.from(), self.to()] {
            s.push((b'a' + (sq & 7) as u8) as char);
            s.push((b'1' + (sq >> 3) as u8) as char);
        }
        if self.is_promo() {
            s.push(['n', 'b', 'r', 'q'][(self.flag() & 3) as usize]);
        }
        s
    }
}

// ---------------------------------------------------------------------------
// Zobrist keys (generated at compile time)
// ---------------------------------------------------------------------------

pub struct Zobrist {
    pub psq: [[u64; 64]; 16],
    pub castling: [u64; 16],
    pub ep: [u64; 8],
    pub side: u64,
}

const fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

const fn make_zobrist() -> Zobrist {
    let mut z = Zobrist { psq: [[0; 64]; 16], castling: [0; 16], ep: [0; 8], side: 0 };
    let mut s = 0x00C0_FFEE_D15E_A5E5u64;
    let mut p = 0;
    while p < 16 {
        let mut sq = 0;
        while sq < 64 {
            z.psq[p][sq] = splitmix(&mut s);
            sq += 1;
        }
        p += 1;
    }
    let mut i = 1;
    while i < 16 {
        z.castling[i] = splitmix(&mut s);
        i += 1;
    }
    let mut i = 0;
    while i < 8 {
        z.ep[i] = splitmix(&mut s);
        i += 1;
    }
    z.side = splitmix(&mut s);
    z
}

pub static ZOBRIST: Zobrist = make_zobrist();

const fn make_castle_masks() -> [u8; 64] {
    let mut m = [15u8; 64];
    m[0] = 15 & !2; // a1: white queenside
    m[4] = 15 & !3; // e1
    m[7] = 15 & !1; // h1
    m[56] = 15 & !8; // a8
    m[60] = 15 & !12; // e8
    m[63] = 15 & !4; // h8
    m
}
static CASTLE_MASK: [u8; 64] = make_castle_masks();

// ---------------------------------------------------------------------------
// Position
// ---------------------------------------------------------------------------

/// Per-node data for fast check detection; see `Position::check_info`.
#[derive(Clone, Copy)]
pub struct CheckInfo {
    squares: [u64; 7],
    blockers: u64,
    ksq: usize,
}

#[derive(Clone, Copy)]
pub struct Position {
    /// Indexed by piece type (1..=6); index 0 is unused.
    pub by_type: [u64; 7],
    pub by_color: [u64; 2],
    pub board: [u8; 64],
    pub key: u64,
    pub pawn_key: u64,
    pub nonpawn_key: [u64; 2],
    /// Key of the knights and bishops (both colours).
    pub minor_key: u64,
    /// Pieces giving check to the side to move.
    pub checkers: u64,
    /// Side-to-move pieces that are pinned to their own king.
    pub pinned: u64,
    pub stm: usize,
    pub castling: u8,
    pub ep: u8,
    pub rule50: u8,
    pub plies_from_null: u16,
    pub fullmove: u16,
}

impl Position {
    pub fn empty() -> Position {
        Position {
            by_type: [0; 7],
            by_color: [0; 2],
            board: [0; 64],
            key: 0,
            pawn_key: 0,
            nonpawn_key: [0; 2],
            minor_key: 0,
            checkers: 0,
            pinned: 0,
            stm: WHITE,
            castling: 0,
            ep: NO_SQ,
            rule50: 0,
            plies_from_null: 0,
            fullmove: 1,
        }
    }

    pub fn startpos() -> Position {
        Position::from_fen(START_FEN).unwrap()
    }

    #[inline(always)]
    pub fn occ(&self) -> u64 {
        self.by_color[0] | self.by_color[1]
    }
    #[inline(always)]
    pub fn pieces(&self, color: usize, pt: usize) -> u64 {
        self.by_color[color] & self.by_type[pt]
    }
    #[inline(always)]
    pub fn king_sq(&self, color: usize) -> usize {
        lsb(self.by_color[color] & self.by_type[KING])
    }
    #[inline(always)]
    pub fn in_check(&self) -> bool {
        self.checkers != 0
    }
    #[inline(always)]
    pub fn has_non_pawn_material(&self, color: usize) -> bool {
        self.by_color[color] & !(self.by_type[PAWN] | self.by_type[KING]) != 0
    }

    #[inline(always)]
    fn put(&mut self, pc: u8, sq: usize) {
        let (c, pt) = (piece_color(pc), piece_type(pc));
        let b = bit(sq);
        self.by_type[pt] |= b;
        self.by_color[c] |= b;
        self.board[sq] = pc;
        let k = ZOBRIST.psq[pc as usize][sq];
        self.key ^= k;
        if pt == PAWN {
            self.pawn_key ^= k;
        } else {
            self.nonpawn_key[c] ^= k;
            if pt == KNIGHT || pt == BISHOP {
                self.minor_key ^= k;
            }
        }
    }

    #[inline(always)]
    fn remove(&mut self, sq: usize) {
        let pc = self.board[sq];
        let (c, pt) = (piece_color(pc), piece_type(pc));
        let b = bit(sq);
        self.by_type[pt] ^= b;
        self.by_color[c] ^= b;
        self.board[sq] = 0;
        let k = ZOBRIST.psq[pc as usize][sq];
        self.key ^= k;
        if pt == PAWN {
            self.pawn_key ^= k;
        } else {
            self.nonpawn_key[c] ^= k;
            if pt == KNIGHT || pt == BISHOP {
                self.minor_key ^= k;
            }
        }
    }

    /// All pieces (of both colours) attacking `sq` given occupancy `occ`.
    #[inline]
    pub fn attackers_to(&self, sq: usize, occ: u64) -> u64 {
        (pawn_attacks(BLACK, sq) & self.pieces(WHITE, PAWN))
            | (pawn_attacks(WHITE, sq) & self.pieces(BLACK, PAWN))
            | (knight_attacks(sq) & self.by_type[KNIGHT])
            | (king_attacks(sq) & self.by_type[KING])
            | (bishop_attacks(sq, occ) & (self.by_type[BISHOP] | self.by_type[QUEEN]))
            | (rook_attacks(sq, occ) & (self.by_type[ROOK] | self.by_type[QUEEN]))
    }

    #[inline]
    pub fn is_attacked_by(&self, sq: usize, by: usize, occ: u64) -> bool {
        (pawn_attacks(by ^ 1, sq) & self.pieces(by, PAWN)) != 0
            || (knight_attacks(sq) & self.pieces(by, KNIGHT)) != 0
            || (king_attacks(sq) & self.pieces(by, KING)) != 0
            || (bishop_attacks(sq, occ) & self.by_color[by] & (self.by_type[BISHOP] | self.by_type[QUEEN])) != 0
            || (rook_attacks(sq, occ) & self.by_color[by] & (self.by_type[ROOK] | self.by_type[QUEEN])) != 0
    }

    fn update_check_info(&mut self) {
        let us = self.stm;
        let them = us ^ 1;
        let ksq = self.king_sq(us);
        let occ = self.occ();
        self.checkers = self.attackers_to(ksq, occ) & self.by_color[them];
        let t = tables();
        let mut snipers = ((t.rook_empty[ksq] & (self.by_type[ROOK] | self.by_type[QUEEN]))
            | (t.bishop_empty[ksq] & (self.by_type[BISHOP] | self.by_type[QUEEN])))
            & self.by_color[them];
        let mut pinned = 0;
        while snipers != 0 {
            let s = pop_lsb(&mut snipers);
            let b = between(ksq, s) & occ;
            if b != 0 && !more_than_one(b) {
                pinned |= b & self.by_color[us];
            }
        }
        self.pinned = pinned;
    }

    /// Play a (legal) move and return the resulting position.
    pub fn make(&self, m: Move) -> Position {
        let mut p = *self;
        let us = self.stm;
        let them = us ^ 1;
        let (from, to, flag) = (m.from(), m.to(), m.flag());
        let pc = self.board[from];
        let pt = piece_type(pc);

        p.rule50 = self.rule50.saturating_add(1);
        p.plies_from_null = self.plies_from_null.saturating_add(1);
        p.key ^= ZOBRIST.side;
        if self.ep != NO_SQ {
            p.key ^= ZOBRIST.ep[(self.ep & 7) as usize];
            p.ep = NO_SQ;
        }

        if m.is_capture() {
            let capsq = if flag == FLAG_EP { to ^ 8 } else { to };
            p.remove(capsq);
            p.rule50 = 0;
        }

        p.remove(from);
        if m.is_promo() {
            p.put(make_piece(us, m.promo_type()), to);
        } else {
            p.put(pc, to);
        }

        if pt == PAWN {
            p.rule50 = 0;
            if flag == FLAG_DOUBLE {
                let epsq = (from + to) / 2;
                if pawn_attacks(us, epsq) & self.pieces(them, PAWN) != 0 {
                    p.ep = epsq as u8;
                    p.key ^= ZOBRIST.ep[epsq & 7];
                }
            }
        } else if flag == FLAG_KCASTLE {
            let rook = p.board[to + 1];
            p.remove(to + 1);
            p.put(rook, to - 1);
        } else if flag == FLAG_QCASTLE {
            let rook = p.board[to - 2];
            p.remove(to - 2);
            p.put(rook, to + 1);
        }

        let rights = self.castling & CASTLE_MASK[from] & CASTLE_MASK[to];
        if rights != self.castling {
            p.key ^= ZOBRIST.castling[self.castling as usize] ^ ZOBRIST.castling[rights as usize];
            p.castling = rights;
        }

        p.stm = them;
        if us == BLACK {
            p.fullmove = p.fullmove.saturating_add(1);
        }
        p.update_check_info();
        p
    }

    /// Zobrist key of the position after playing `m`, without updating check/pin info.
    #[inline]
    pub fn key_after(&self, m: Move) -> u64 {
        let flag = m.flag();
        if flag == FLAG_EP || flag == FLAG_KCASTLE || flag == FLAG_QCASTLE || m.is_promo() {
            return self.make(m).key;
        }
        let us = self.stm;
        let them = us ^ 1;
        let (from, to) = (m.from(), m.to());
        let pc = self.board[from] as usize;
        let mut k = self.key ^ ZOBRIST.side;
        if self.ep != NO_SQ {
            k ^= ZOBRIST.ep[(self.ep & 7) as usize];
        }
        k ^= ZOBRIST.psq[pc][from] ^ ZOBRIST.psq[pc][to];
        if m.is_capture() {
            let cap = self.board[to] as usize;
            k ^= ZOBRIST.psq[cap][to];
        }
        if piece_type(pc as u8) == PAWN && flag == FLAG_DOUBLE {
            let epsq = (from + to) / 2;
            if pawn_attacks(us, epsq) & self.pieces(them, PAWN) != 0 {
                k ^= ZOBRIST.ep[epsq & 7];
            }
        }
        let rights = self.castling & CASTLE_MASK[from] & CASTLE_MASK[to];
        if rights != self.castling {
            k ^= ZOBRIST.castling[self.castling as usize] ^ ZOBRIST.castling[rights as usize];
        }
        k
    }

    pub fn make_null(&self) -> Position {
        let mut p = *self;
        p.key ^= ZOBRIST.side;
        if self.ep != NO_SQ {
            p.key ^= ZOBRIST.ep[(self.ep & 7) as usize];
            p.ep = NO_SQ;
        }
        p.stm ^= 1;
        p.rule50 = self.rule50.saturating_add(1);
        p.plies_from_null = 0;
        p.update_check_info();
        p
    }

    /// Squares the king passes over (and lands on) must not be attacked;
    /// squares between king and rook must be empty.
    pub fn can_castle(&self, kingside: bool) -> bool {
        let us = self.stm;
        let right = if kingside { 1 } else { 2 } << (2 * us);
        if self.castling & right == 0 || self.checkers != 0 {
            return false;
        }
        let base = 56 * us;
        let occ = self.occ();
        let (empty, pass) = if kingside {
            (bit(base + 5) | bit(base + 6), [base + 5, base + 6])
        } else {
            (bit(base + 1) | bit(base + 2) | bit(base + 3), [base + 3, base + 2])
        };
        if occ & empty != 0 {
            return false;
        }
        !self.is_attacked_by(pass[0], us ^ 1, occ) && !self.is_attacked_by(pass[1], us ^ 1, occ)
    }

    /// Does `m` obey the movement rules in this position (ignoring pins / king safety,
    /// but respecting check-evasion targets)? Used to validate hash moves.
    pub fn is_pseudo_legal(&self, m: Move) -> bool {
        if m.is_none() {
            return false;
        }
        let us = self.stm;
        let (from, to, flag) = (m.from(), m.to(), m.flag());
        let pc = self.board[from];
        if pc == 0 || piece_color(pc) != us {
            return false;
        }
        let captured = self.board[to];
        if captured != 0 && (piece_color(captured) == us || piece_type(captured) == KING) {
            return false;
        }
        let pt = piece_type(pc);
        let occ = self.occ();
        let tb = bit(to);

        if m.is_castle() {
            return pt == KING && from == 4 + 56 * us && self.can_castle(flag == FLAG_KCASTLE) && {
                if flag == FLAG_KCASTLE {
                    to == from + 2
                } else {
                    to == from - 2
                }
            };
        }
        if flag == FLAG_EP {
            if pt != PAWN || self.ep == NO_SQ || to != self.ep as usize || pawn_attacks(us, from) & tb == 0 {
                return false;
            }
        } else {
            if m.is_capture() != (captured != 0) {
                return false;
            }
            if pt == PAWN {
                let last_rank = if us == WHITE { RANK_8 } else { RANK_1 };
                if m.is_promo() != (tb & last_rank != 0) {
                    return false;
                }
                let fwd = if us == WHITE { from + 8 } else { from.wrapping_sub(8) };
                if m.is_capture() {
                    if pawn_attacks(us, from) & tb == 0 {
                        return false;
                    }
                } else if flag == FLAG_DOUBLE {
                    let start = if us == WHITE { RANK_2 } else { RANK_7 };
                    let dst = if us == WHITE { from + 16 } else { from.wrapping_sub(16) };
                    if bit(from) & start == 0 || to != dst || occ & (bit(fwd) | tb) != 0 {
                        return false;
                    }
                } else if to != fwd || captured != 0 {
                    return false;
                }
            } else {
                if flag != FLAG_QUIET && flag != FLAG_CAPTURE {
                    return false;
                }
                let att = match pt {
                    KNIGHT => knight_attacks(from),
                    BISHOP => bishop_attacks(from, occ),
                    ROOK => rook_attacks(from, occ),
                    QUEEN => bishop_attacks(from, occ) | rook_attacks(from, occ),
                    _ => king_attacks(from),
                };
                if att & tb == 0 {
                    return false;
                }
            }
        }

        if self.checkers != 0 && pt != KING {
            if more_than_one(self.checkers) {
                return false;
            }
            let csq = lsb(self.checkers);
            let target = between(self.king_sq(us), csq) | self.checkers;
            if target & tb == 0 && !(flag == FLAG_EP && (to ^ 8) == csq) {
                return false;
            }
        }
        true
    }

    /// Full legality test for a pseudo-legal move.
    pub fn is_legal(&self, m: Move) -> bool {
        let us = self.stm;
        let them = us ^ 1;
        let (from, to) = (m.from(), m.to());
        let ksq = self.king_sq(us);
        let occ = self.occ();
        if from == ksq {
            return m.is_castle() || !self.is_attacked_by(to, them, occ ^ bit(from));
        }
        if m.flag() == FLAG_EP {
            let capsq = to ^ 8;
            let o = (occ ^ bit(from) ^ bit(capsq)) | bit(to);
            return rook_attacks(ksq, o) & self.by_color[them] & (self.by_type[ROOK] | self.by_type[QUEEN]) == 0
                && bishop_attacks(ksq, o) & self.by_color[them] & (self.by_type[BISHOP] | self.by_type[QUEEN]) == 0;
        }
        self.pinned & bit(from) == 0 || line(from, to) & bit(ksq) != 0
    }

    /// Does the pseudo-legal move `m` give check?
    pub fn gives_check(&self, m: Move) -> bool {
        if m.is_castle() || m.flag() == FLAG_EP {
            return self.make(m).checkers != 0;
        }
        let us = self.stm;
        let ksq = self.king_sq(us ^ 1);
        let (from, to) = (m.from(), m.to());
        let pt = if m.is_promo() { m.promo_type() } else { piece_type(self.board[from]) };
        let occ = (self.occ() ^ bit(from)) | bit(to);
        let kb = bit(ksq);
        // A slider can only give check along a line through the king; `line` is
        // zero for unaligned squares, which skips the attack lookups.
        let direct = match pt {
            PAWN => pawn_attacks(us, to) & kb,
            KNIGHT => knight_attacks(to) & kb,
            _ if line(to, ksq) == 0 => 0,
            BISHOP => bishop_attacks(to, occ) & kb,
            ROOK => rook_attacks(to, occ) & kb,
            QUEEN => (bishop_attacks(to, occ) | rook_attacks(to, occ)) & kb,
            _ => 0,
        };
        if direct != 0 {
            return true;
        }
        // Discovered check: one of our sliders now sees the king through `from`.
        // A slider standing on `to` would be a captured enemy piece, hence the
        // colour mask.
        if line(from, ksq) == 0 {
            return false;
        }
        let ours = self.by_color[us] & !bit(from);
        (rook_attacks(ksq, occ) & ours & (self.by_type[ROOK] | self.by_type[QUEEN])) != 0
            || (bishop_attacks(ksq, occ) & ours & (self.by_type[BISHOP] | self.by_type[QUEEN])) != 0
    }

    /// What `gives_check` needs, worked out once per node (Stockfish's check
    /// squares and discovered-check blockers).
    #[inline]
    pub fn check_info(&self) -> CheckInfo {
        let us = self.stm;
        let ksq = self.king_sq(us ^ 1);
        let occ = self.occ();
        let ours = self.by_color[us];
        let mut squares = [0u64; 7];
        squares[PAWN] = pawn_attacks(us ^ 1, ksq);
        squares[KNIGHT] = knight_attacks(ksq);
        squares[BISHOP] = bishop_attacks(ksq, occ);
        squares[ROOK] = rook_attacks(ksq, occ);
        squares[QUEEN] = squares[BISHOP] | squares[ROOK];
        // Our pieces that alone stand between one of our sliders and their king.
        let t = tables();
        let mut snipers = (t.rook_empty[ksq] & ours & (self.by_type[ROOK] | self.by_type[QUEEN]))
            | (t.bishop_empty[ksq] & ours & (self.by_type[BISHOP] | self.by_type[QUEEN]));
        let mut blockers = 0u64;
        while snipers != 0 {
            let sq = pop_lsb(&mut snipers);
            let b = between(sq, ksq) & occ;
            if b != 0 && b & (b - 1) == 0 {
                blockers |= b & ours;
            }
        }
        CheckInfo { squares, blockers, ksq }
    }

    /// `gives_check` using precomputed `ci`.
    #[inline(always)]
    pub fn gives_check_ci(&self, m: Move, ci: &CheckInfo) -> bool {
        let f = m.flag();
        if f == FLAG_EP || f == FLAG_KCASTLE || f == FLAG_QCASTLE || m.is_promo() {
            return self.gives_check(m);
        }
        let (from, to) = (m.from(), m.to());
        let pt = piece_type(self.board[from]);
        ci.squares[pt] & bit(to) != 0 || (ci.blockers & bit(from) != 0 && line(from, ci.ksq) & bit(to) == 0)
    }

    /// For the king of `color`: the pieces (of either colour) that alone block an
    /// enemy slider's line to it, and the enemy sliders pinning a piece of
    /// `color` that way (Stockfish's blockersForKing and the opponent's pinners).
    pub fn slider_blockers(&self, color: usize) -> (u64, u64) {
        let ksq = self.king_sq(color);
        let t = tables();
        let them = self.by_color[color ^ 1];
        let occ = self.occ();
        let mut snipers = ((t.rook_empty[ksq] & (self.by_type[ROOK] | self.by_type[QUEEN]))
            | (t.bishop_empty[ksq] & (self.by_type[BISHOP] | self.by_type[QUEEN])))
            & them;
        let (mut blockers, mut pinners) = (0u64, 0u64);
        while snipers != 0 {
            let sq = pop_lsb(&mut snipers);
            let b = between(sq, ksq) & occ;
            if b != 0 && b & (b - 1) == 0 {
                blockers |= b;
                if b & self.by_color[color] != 0 {
                    pinners |= bit(sq);
                }
            }
        }
        (blockers, pinners)
    }

    /// Static exchange evaluation: is the exchange started by `m` worth at least `threshold`?
    pub fn see_ge(&self, m: Move, threshold: i32) -> bool {
        // Like Stockfish, special moves (castling, en passant, promotions) count as
        // an even exchange.
        if m.is_castle() || m.is_promo() || m.flag() == FLAG_EP {
            return threshold <= 0;
        }
        let (from, to, flag) = (m.from(), m.to(), m.flag());
        let mut victim = piece_type(self.board[from]);
        let mut balance = -threshold;
        if flag == FLAG_EP {
            balance += SEE_VALUE[PAWN];
        } else {
            balance += SEE_VALUE[piece_type(self.board[to])];
        }
        if m.is_promo() {
            victim = m.promo_type();
            balance += SEE_VALUE[victim] - SEE_VALUE[PAWN];
        }
        if balance < 0 {
            return false;
        }
        balance -= SEE_VALUE[victim];
        if balance >= 0 {
            return true;
        }
        let mut occ = (self.occ() ^ bit(from)) | bit(to);
        if flag == FLAG_EP {
            occ ^= bit(to ^ 8);
        }
        let diag = self.by_type[BISHOP] | self.by_type[QUEEN];
        let orth = self.by_type[ROOK] | self.by_type[QUEEN];
        let mut attackers = self.attackers_to(to, occ) & occ;
        let mut stm = self.stm ^ 1;
        // Stockfish's pin rule: a piece pinned to its king may not recapture
        // while the pinner is still on the board.
        let pins = [self.slider_blockers(WHITE), self.slider_blockers(BLACK)];
        loop {
            let mut mine = attackers & self.by_color[stm];
            if pins[stm].1 & occ != 0 {
                mine &= !pins[stm].0;
            }
            if mine == 0 {
                break;
            }
            let mut pt = PAWN;
            while pt < KING && mine & self.by_type[pt] == 0 {
                pt += 1;
            }
            occ ^= bit(lsb(mine & self.by_type[pt]));
            if pt == PAWN || pt == BISHOP || pt == QUEEN {
                attackers |= bishop_attacks(to, occ) & diag;
            }
            if pt == ROOK || pt == QUEEN {
                attackers |= rook_attacks(to, occ) & orth;
            }
            attackers &= occ;
            stm ^= 1;
            balance = -balance - 1 - SEE_VALUE[pt];
            if balance >= 0 {
                if pt == KING && attackers & self.by_color[stm] != 0 {
                    stm ^= 1;
                }
                break;
            }
        }
        stm != self.stm
    }

    pub fn insufficient_material(&self) -> bool {
        if self.by_type[PAWN] | self.by_type[ROOK] | self.by_type[QUEEN] != 0 {
            return false;
        }
        popcount(self.by_type[KNIGHT] | self.by_type[BISHOP]) <= 1
    }

    // -- FEN --------------------------------------------------------------

    pub fn from_fen(fen: &str) -> Result<Position, String> {
        let mut p = Position::empty();
        let parts: Vec<&str> = fen.split_whitespace().collect();
        if parts.len() < 2 {
            return Err("FEN needs at least board and side to move".into());
        }
        let (mut file, mut rank) = (0i32, 7i32);
        for ch in parts[0].chars() {
            match ch {
                '/' => {
                    rank -= 1;
                    file = 0;
                }
                '1'..='8' => file += ch as i32 - '0' as i32,
                _ => {
                    let pt = match ch.to_ascii_lowercase() {
                        'p' => PAWN,
                        'n' => KNIGHT,
                        'b' => BISHOP,
                        'r' => ROOK,
                        'q' => QUEEN,
                        'k' => KING,
                        _ => return Err(format!("bad piece '{ch}'")),
                    };
                    if !(0..8).contains(&file) || !(0..8).contains(&rank) {
                        return Err("board out of range".into());
                    }
                    let color = if ch.is_ascii_uppercase() { WHITE } else { BLACK };
                    p.put(make_piece(color, pt), (rank * 8 + file) as usize);
                    file += 1;
                }
            }
        }
        for c in [WHITE, BLACK] {
            if popcount(p.pieces(c, KING)) != 1 {
                return Err("each side needs exactly one king".into());
            }
        }
        p.stm = match parts[1] {
            "w" => WHITE,
            "b" => BLACK,
            _ => return Err("bad side to move".into()),
        };
        if p.stm == BLACK {
            p.key ^= ZOBRIST.side;
        }
        if let Some(c) = parts.get(2) {
            for ch in c.chars() {
                let (bitv, ksq, rsq, color) = match ch {
                    'K' => (1, 4, 7, WHITE),
                    'Q' => (2, 4, 0, WHITE),
                    'k' => (4, 60, 63, BLACK),
                    'q' => (8, 60, 56, BLACK),
                    _ => continue,
                };
                if p.board[ksq] == make_piece(color, KING) && p.board[rsq] == make_piece(color, ROOK) {
                    p.castling |= bitv;
                }
            }
        }
        p.key ^= ZOBRIST.castling[p.castling as usize];
        if let Some(e) = parts.get(3) {
            let b = e.as_bytes();
            if b.len() == 2 && (b'a'..=b'h').contains(&b[0]) && (b[1] == b'3' || b[1] == b'6') {
                let sq = ((b[1] - b'1') as usize) * 8 + (b[0] - b'a') as usize;
                let them = p.stm ^ 1;
                // Only keep the square if a capture is actually possible.
                if pawn_attacks(them, sq) & p.pieces(p.stm, PAWN) != 0
                    && p.board[sq ^ 8] == make_piece(them, PAWN)
                    && p.board[sq] == 0
                {
                    p.ep = sq as u8;
                    p.key ^= ZOBRIST.ep[sq & 7];
                }
            }
        }
        p.rule50 = parts.get(4).and_then(|s| s.parse::<u32>().ok()).unwrap_or(0).min(250) as u8;
        p.fullmove = parts.get(5).and_then(|s| s.parse::<u32>().ok()).unwrap_or(1).clamp(1, 60000) as u16;
        p.plies_from_null = p.rule50 as u16;
        p.update_check_info();
        // The side that just moved must not be left in check.
        let them = p.stm ^ 1;
        if p.is_attacked_by(p.king_sq(them), p.stm, p.occ()) {
            return Err("side not to move is in check".into());
        }
        Ok(p)
    }

    pub fn to_fen(&self) -> String {
        let mut s = String::new();
        for rank in (0..8).rev() {
            let mut empty = 0;
            for file in 0..8 {
                let pc = self.board[rank * 8 + file];
                if pc == 0 {
                    empty += 1;
                    continue;
                }
                if empty > 0 {
                    s.push_str(&empty.to_string());
                    empty = 0;
                }
                let ch = b" pnbrqk"[piece_type(pc)] as char;
                s.push(if piece_color(pc) == WHITE { ch.to_ascii_uppercase() } else { ch });
            }
            if empty > 0 {
                s.push_str(&empty.to_string());
            }
            if rank > 0 {
                s.push('/');
            }
        }
        s.push_str(if self.stm == WHITE { " w " } else { " b " });
        if self.castling == 0 {
            s.push('-');
        } else {
            for (i, ch) in "KQkq".chars().enumerate() {
                if self.castling & (1 << i) != 0 {
                    s.push(ch);
                }
            }
        }
        if self.ep == NO_SQ {
            s.push_str(" -");
        } else {
            s.push(' ');
            s.push((b'a' + (self.ep & 7)) as char);
            s.push((b'1' + (self.ep >> 3)) as char);
        }
        s.push_str(&format!(" {} {}", self.rule50, self.fullmove));
        s
    }

    pub fn pretty(&self) -> String {
        let mut s = String::new();
        for rank in (0..8).rev() {
            s.push_str(" +---+---+---+---+---+---+---+---+\n");
            for file in 0..8 {
                let pc = self.board[rank * 8 + file];
                let ch = b" pnbrqk"[piece_type(pc)] as char;
                let ch = if pc != 0 && piece_color(pc) == WHITE { ch.to_ascii_uppercase() } else { ch };
                s.push_str(&format!(" | {ch}"));
            }
            s.push_str(&format!(" | {}\n", rank + 1));
        }
        s.push_str(" +---+---+---+---+---+---+---+---+\n   a   b   c   d   e   f   g   h\n");
        s.push_str(&format!("\nFen: {}\nKey: {:016X}\n", self.to_fen(), self.key));
        s
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::movegen::parse_move;

    fn see(fen: &str, mv: &str, threshold: i32) -> bool {
        crate::bitboard::init();
        let pos = Position::from_fen(fen).unwrap();
        pos.see_ge(parse_move(&pos, mv).unwrap(), threshold)
    }

    #[test]
    fn see_respects_pins() {
        // The knight on c3 defends e4 but is pinned to its king by the b4 bishop.
        let pinned = "4r1k1/8/8/8/1b2P3/2N5/8/4K3 b - - 0 1";
        assert!(see(pinned, "e8e4", 0), "a pinned knight cannot recapture");
        // Without the pinning bishop the knight recaptures and the rook is lost.
        let free = "4r1k1/8/8/8/4P3/2N5/8/4K3 b - - 0 1";
        assert!(!see(free, "e8e4", 0));
    }
}
