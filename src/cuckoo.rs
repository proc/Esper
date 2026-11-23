//! Upcoming-repetition detection with Marcel van Kervinck's cuckoo tables.
//!
//! The tables hold the Zobrist difference of every reversible piece move (a
//! knight, bishop, rook, queen or king going from one square to another and
//! the side to move flipping). If the current position and one a few plies back
//! differ by exactly such a key, and the squares between are empty, the side to
//! move can repeat that earlier position with its next move.

use crate::bitboard::*;
use crate::position::*;
use std::sync::OnceLock;

const SIZE: usize = 8192;

struct Tables {
    keys: Box<[u64; SIZE]>,
    moves: Box<[(u8, u8); SIZE]>,
}

#[inline(always)]
fn h1(k: u64) -> usize {
    (k & 0x1fff) as usize
}

#[inline(always)]
fn h2(k: u64) -> usize {
    ((k >> 16) & 0x1fff) as usize
}

fn tables() -> &'static Tables {
    static T: OnceLock<Tables> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = Tables { keys: Box::new([0; SIZE]), moves: Box::new([(0, 0); SIZE]) };
        let mut count = 0;
        for color in [WHITE, BLACK] {
            for pt in KNIGHT..=KING {
                let pc = make_piece(color, pt) as usize;
                for s1 in 0..64 {
                    let att = match pt {
                        KNIGHT => knight_attacks(s1),
                        BISHOP => bishop_attacks(s1, 0),
                        ROOK => rook_attacks(s1, 0),
                        QUEEN => bishop_attacks(s1, 0) | rook_attacks(s1, 0),
                        _ => king_attacks(s1),
                    };
                    for s2 in s1 + 1..64 {
                        if att & bit(s2) == 0 {
                            continue;
                        }
                        let mut key = ZOBRIST.psq[pc][s1] ^ ZOBRIST.psq[pc][s2] ^ ZOBRIST.side;
                        let mut mv = (s1 as u8, s2 as u8);
                        let mut i = h1(key);
                        loop {
                            std::mem::swap(&mut t.keys[i], &mut key);
                            std::mem::swap(&mut t.moves[i], &mut mv);
                            if key == 0 {
                                break;
                            }
                            i = if i == h1(key) { h2(key) } else { h1(key) };
                        }
                        count += 1;
                    }
                }
            }
        }
        debug_assert_eq!(count, 3668);
        t
    })
}

/// Build the tables up front so the first search does not pay for it.
pub fn init() {
    tables();
}

/// Can the side to move repeat an earlier position with its next move?
///
/// `keys` holds the keys of the game and search so far, the current position
/// last; `ply` is the distance from the root. Positions inside the search tree
/// count after one occurrence; positions before the root only if they had
/// already occurred twice (so the move would really draw).
pub fn upcoming_repetition(pos: &Position, keys: &[u64], ply: usize) -> bool {
    let cur = keys.len() - 1;
    let end = (pos.rule50 as usize).min(pos.plies_from_null as usize).min(cur);
    if end < 3 {
        return false;
    }
    let orig = keys[cur];
    let mut other = orig ^ keys[cur - 1] ^ ZOBRIST.side;
    let mut i = 3;
    while i <= end {
        other ^= keys[cur - i + 1] ^ keys[cur - i] ^ ZOBRIST.side;
        if other == 0 {
            let move_key = orig ^ keys[cur - i];
            // Looked up only here: most calls never find a matching key pair.
            let t = tables();
            let mut j = h1(move_key);
            if t.keys[j] != move_key {
                j = h2(move_key);
            }
            if t.keys[j] == move_key {
                let (s1, s2) = (t.moves[j].0 as usize, t.moves[j].1 as usize);
                if between(s1, s2) & pos.occ() == 0 {
                    let pc = if pos.board[s1] != 0 { pos.board[s1] } else { pos.board[s2] };
                    if pc != 0 && piece_color(pc) == pos.stm {
                        if ply > i {
                            return true;
                        }
                        // Before the root: the earlier position must itself be a repetition.
                        let k = keys[cur - i];
                        let mut n = cur - i;
                        while n >= 2 {
                            n -= 2;
                            if keys[n] == k {
                                return true;
                            }
                        }
                    }
                }
            }
        }
        i += 2;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::movegen::parse_move;

    fn play(fen: &str, moves: &[&str]) -> (Position, Vec<u64>) {
        crate::bitboard::init();
        let mut pos = Position::from_fen(fen).unwrap();
        let mut keys = vec![pos.key];
        for m in moves {
            let mv = parse_move(&pos, m).unwrap();
            pos = pos.make(mv);
            keys.push(pos.key);
        }
        (pos, keys)
    }

    #[test]
    fn table_size() {
        crate::bitboard::init();
        let n = tables().keys.iter().filter(|&&k| k != 0).count();
        assert_eq!(n, 3668);
    }

    #[test]
    fn detects_knight_shuffle() {
        // After Nf3 Nf6 Ng1, Black can play Ng8 and repeat the start position.
        let (pos, keys) = play(START_FEN, &["g1f3", "g8f6", "f3g1"]);
        assert!(upcoming_repetition(&pos, &keys, 10));
        // Without the shuffle there is nothing to repeat.
        let (pos, keys) = play(START_FEN, &["e2e4", "e7e5", "g1f3"]);
        assert!(!upcoming_repetition(&pos, &keys, 10));
        // Before the root, the target position must already have occurred twice.
        let (pos, keys) = play(START_FEN, &["g1f3", "g8f6", "f3g1"]);
        assert!(!upcoming_repetition(&pos, &keys, 0));
    }
}
