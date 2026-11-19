//! Polyglot opening books.
//!
//! A Polyglot book is a sorted array of 16-byte big-endian entries
//! (key:64 move:16 weight:16 learn:32). Lookups binary-search the raw bytes,
//! so the embedded book is used in place with no loading step.

use crate::bitboard::pawn_attacks;
use crate::movegen::legal_moves;
use crate::polyglot_keys::KEYS;
use crate::position::*;
use std::borrow::Cow;

/// The built-in book, compiled into the binary (feature `book`).
#[cfg(feature = "book")]
static EMBEDDED: &[u8] = include_bytes!("../books/esper.bin");
#[cfg(not(feature = "book"))]
static EMBEDDED: &[u8] = &[];

pub const EMBEDDED_NAME: &str = "<embedded>";

/// Polyglot key of a position.
pub fn hash(pos: &Position) -> u64 {
    let mut key = 0u64;
    for sq in 0..64 {
        let pc = pos.board[sq];
        if pc == 0 {
            continue;
        }
        // Polyglot orders pieces bp, wp, bn, wn, ... bk, wk.
        let kind = 2 * (piece_type(pc) - 1) + (piece_color(pc) == WHITE) as usize;
        key ^= KEYS[64 * kind + sq];
    }
    for i in 0..4 {
        if pos.castling & (1 << i) != 0 {
            key ^= KEYS[768 + i];
        }
    }
    // The en-passant file counts only if a pawn can actually capture there.
    if pos.ep != NO_SQ {
        let ep = pos.ep as usize;
        if pawn_attacks(pos.stm ^ 1, ep) & pos.pieces(pos.stm, PAWN) != 0 {
            key ^= KEYS[772 + (ep & 7)];
        }
    }
    if pos.stm == WHITE {
        key ^= KEYS[780];
    }
    key
}

pub struct Book {
    data: Cow<'static, [u8]>,
    pub name: String,
}

#[derive(Clone, Copy)]
pub struct BookMove {
    pub mv: Move,
    pub weight: u16,
}

impl Book {
    /// The embedded book, if this build has one.
    pub fn embedded() -> Option<Book> {
        (!EMBEDDED.is_empty()).then(|| Book { data: Cow::Borrowed(EMBEDDED), name: EMBEDDED_NAME.to_string() })
    }

    /// `<embedded>` or a path to a Polyglot `.bin` file.
    pub fn load(path: &str) -> Result<Book, String> {
        if path.is_empty() || path == EMBEDDED_NAME {
            return Book::embedded().ok_or_else(|| "this build has no embedded book".to_string());
        }
        let data = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        if data.is_empty() || data.len() % 16 != 0 {
            return Err(format!("{path}: not a Polyglot book ({} bytes)", data.len()));
        }
        Ok(Book { data: Cow::Owned(data), name: path.to_string() })
    }

    pub fn len(&self) -> usize {
        self.data.len() / 16
    }

    fn key_at(&self, i: usize) -> u64 {
        u64::from_be_bytes(self.data[16 * i..16 * i + 8].try_into().unwrap())
    }

    /// The legal book moves for `pos`, best weighted first.
    pub fn moves(&self, pos: &Position) -> Vec<BookMove> {
        let key = hash(pos);
        let (mut lo, mut hi) = (0, self.len());
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.key_at(mid) < key {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        let legal = legal_moves(pos);
        let mut out: Vec<BookMove> = Vec::new();
        let mut i = lo;
        while i < self.len() && self.key_at(i) == key {
            let e = &self.data[16 * i..16 * i + 16];
            let raw = u16::from_be_bytes([e[8], e[9]]);
            let weight = u16::from_be_bytes([e[10], e[11]]);
            i += 1;
            if let Some(mv) = decode(pos, raw, &legal) {
                if !out.iter().any(|b| b.mv == mv) {
                    out.push(BookMove { mv, weight });
                }
            }
        }
        out.sort_by(|a, b| b.weight.cmp(&a.weight));
        out
    }

    /// A book move for `pos` among those `allowed` accepts: the heaviest one
    /// when `best`, otherwise drawn at random in proportion to the weights.
    /// Moves with weight zero are only played when nothing else is stored.
    pub fn probe(&self, pos: &Position, best: bool, allowed: impl Fn(Move) -> bool) -> Option<Move> {
        let mut moves = self.moves(pos);
        moves.retain(|m| allowed(m.mv));
        let first = moves.first()?;
        if best || moves.len() == 1 {
            return Some(first.mv);
        }
        let total: u64 = moves.iter().map(|m| m.weight as u64).sum();
        if total == 0 {
            return Some(first.mv);
        }
        let mut pick = random() % total;
        for m in &moves {
            if pick < m.weight as u64 {
                return Some(m.mv);
            }
            pick -= m.weight as u64;
        }
        Some(first.mv)
    }
}

/// Polyglot move: to:6 from:6 promo:3 (n b r q = 1..4); castling is written
/// as the king taking its own rook.
fn decode(pos: &Position, raw: u16, legal: &[Move]) -> Option<Move> {
    let to = (raw & 63) as usize;
    let from = ((raw >> 6) & 63) as usize;
    let promo = ((raw >> 12) & 7) as usize;
    let castle_to = if piece_type(pos.board[from]) == KING && pos.board[to] == make_piece(pos.stm, ROOK) {
        Some(if to > from { from + 2 } else { from - 2 })
    } else {
        None
    };
    legal.iter().copied().find(|m| {
        if m.from() != from {
            return false;
        }
        if let Some(t) = castle_to {
            return m.is_castle() && m.to() == t;
        }
        m.to() == to && !m.is_castle() && if promo == 0 { !m.is_promo() } else { m.is_promo() && m.promo_type() == promo + 1 }
    })
}

/// Small xorshift generator seeded from the clock; book choice only.
fn random() -> u64 {
    use std::cell::Cell;
    use std::time::{SystemTime, UNIX_EPOCH};
    thread_local! {
        static STATE: Cell<u64> = Cell::new(
            SystemTime::now().duration_since(UNIX_EPOCH).map_or(0x9E37_79B9_7F4A_7C15, |d| d.as_nanos() as u64) | 1
        );
    }
    STATE.with(|s| {
        let mut x = s.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        s.set(x);
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference keys from the Polyglot specification.
    #[test]
    fn polyglot_keys() {
        crate::bitboard::init();
        let cases = [
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 0x463b96181691fc9c),
            ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", 0x823c9b50fd114196),
            ("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2", 0x0756b94461c50fb0),
            ("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2", 0x662fafb965db29d4),
            ("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3", 0x22a48b5a8e47ff78),
            ("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPPKPPP/RNBQ1BNR b kq - 0 3", 0x652a607ca3f242c1),
            ("rnbq1bnr/ppp1pkpp/8/3pPp2/8/8/PPPPKPPP/RNBQ1BNR w - - 0 4", 0x00fdd303c946bdd9),
            ("rnbqkbnr/p1pppppp/8/8/PpP4P/8/1P1PPPP1/RNBQKBNR b KQkq c3 0 3", 0x3c8123ea7b067637),
            ("rnbqkbnr/p1pppppp/8/8/P6P/R1p5/1P1PPPP1/1NBQKBNR b Kkq - 0 4", 0x5c3f9b829b279560),
        ];
        for (fen, want) in cases {
            let pos = Position::from_fen(fen).unwrap();
            assert_eq!(hash(&pos), want, "{fen}");
        }
    }
}
