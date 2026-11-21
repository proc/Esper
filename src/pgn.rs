//! Reading games from PGN files, for `datagen pgn=...`.
//!
//! Only what training data needs: the result, the starting position (the `FEN`
//! tag, when present), the main line's moves, and any evaluation recorded in a
//! comment after a move. Variations, annotation glyphs and other comments are
//! skipped.

use crate::movegen::parse_san;
use crate::position::*;
use std::io::BufRead;

/// An evaluation found in the comment after a move.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Eval {
    /// `[%eval 0.35]` (Lichess): the position after the move, in centipawns
    /// from White's point of view.
    AfterWhite(i32),
    /// `{+0.35/12 0.5s}` (cutechess, fastchess, most GUIs): the engine's search
    /// score for the position before its move, in centipawns from the mover's
    /// point of view.
    BeforeMover(i32),
}

pub struct Game {
    pub start: Position,
    /// The main line, each move with the evaluation in its comment, if any.
    pub moves: Vec<(Move, Option<Eval>)>,
    /// White's result: 1.0, 0.5 or 0.0.
    pub result: f32,
    /// A move the parser could not read cut the game short.
    pub truncated: bool,
}

/// Hands out the text of one game at a time from a list of PGN files.
pub struct Reader {
    paths: std::vec::IntoIter<String>,
    file: Option<std::io::BufReader<std::fs::File>>,
    /// The first line of the next game, read while finishing the previous one.
    pending: Option<String>,
}

impl Reader {
    pub fn new(paths: &[String]) -> Result<Reader, String> {
        for p in paths {
            std::fs::File::open(p).map_err(|e| format!("cannot open {p}: {e}"))?;
        }
        Ok(Reader { paths: paths.to_vec().into_iter(), file: None, pending: None })
    }

    fn next_line(&mut self) -> Option<String> {
        if let Some(l) = self.pending.take() {
            return Some(l);
        }
        loop {
            if self.file.is_none() {
                let path = self.paths.next()?;
                self.file = std::fs::File::open(path).ok().map(std::io::BufReader::new);
                continue;
            }
            let mut bytes = Vec::new();
            match self.file.as_mut().unwrap().read_until(b'\n', &mut bytes) {
                Ok(n) if n > 0 => return Some(String::from_utf8_lossy(&bytes).into_owned()),
                // End of this file (or a read error): move on to the next one.
                _ => self.file = None,
            }
        }
    }

    /// The text of the next game: its tag lines and its movetext.
    pub fn next_game(&mut self) -> Option<String> {
        let mut text = String::new();
        let mut in_moves = false;
        while let Some(line) = self.next_line() {
            let is_tag = line.trim_start().starts_with('[') && line.trim_end().ends_with(']');
            if is_tag && in_moves {
                self.pending = Some(line);
                break;
            }
            if !is_tag && !line.trim().is_empty() {
                in_moves = true;
            }
            text.push_str(&line);
        }
        (!text.trim().is_empty()).then_some(text)
    }
}

/// The value of a number of pawns as centipawns, or `None` for a mate score.
fn pawns_to_cp(s: &str) -> Option<i32> {
    let v: f64 = s.parse().ok()?;
    v.is_finite().then(|| (v * 100.0).round().clamp(-30_000.0, 30_000.0) as i32)
}

/// The evaluation in a comment's text, if it has one.
fn comment_eval(c: &str) -> Option<Eval> {
    if let Some(i) = c.find("[%eval") {
        let v = c[i + 6..].trim_start().split(|ch: char| ch == ']' || ch.is_whitespace()).next()?;
        return pawns_to_cp(v).map(Eval::AfterWhite);
    }
    // "+0.35/12 0.5s": a signed number of pawns, then the depth.
    let first = c.trim_start().split(|ch: char| ch == '/' || ch.is_whitespace()).next()?;
    let body = first.strip_prefix(['+', '-']).unwrap_or(first);
    if body.is_empty() || !body.contains('.') || !body.bytes().all(|c| c.is_ascii_digit() || c == b'.') {
        return None;
    }
    pawns_to_cp(first).map(Eval::BeforeMover)
}

fn result_of(s: &str) -> Option<f32> {
    match s {
        "1-0" => Some(1.0),
        "0-1" => Some(0.0),
        "1/2-1/2" => Some(0.5),
        _ => None,
    }
}

/// Parse one game's text. `None` when it has no decisive or drawn result, an
/// unreadable starting position, or no moves.
pub fn parse_game(text: &str) -> Option<Game> {
    let mut result = None;
    let mut start = Position::startpos();
    let mut movetext = String::new();
    for line in text.lines() {
        let t = line.trim();
        if t.starts_with('[') && t.ends_with(']') {
            let inner = &t[1..t.len() - 1];
            let (key, rest) = inner.split_once(char::is_whitespace).unwrap_or((inner, ""));
            let value = rest.trim().trim_matches('"');
            match key {
                "Result" => result = result_of(value),
                "FEN" => start = Position::from_fen(value).ok()?,
                _ => {}
            }
        } else {
            movetext.push_str(line);
            movetext.push('\n');
        }
    }
    let mut pos = start;
    let mut moves: Vec<(Move, Option<Eval>)> = Vec::new();
    let mut truncated = false;
    let mut end_result = None;
    let b = movetext.as_bytes();
    let mut i = 0;
    let mut depth = 0usize; // nesting of variations
    while i < b.len() {
        match b[i] {
            b'{' => {
                let end = movetext[i..].find('}').map_or(b.len(), |e| i + e);
                if depth == 0 {
                    if let (Some(last), Some(ev)) = (moves.last_mut(), comment_eval(&movetext[i + 1..end])) {
                        last.1.get_or_insert(ev);
                    }
                }
                i = end + 1;
            }
            b';' => i = movetext[i..].find('\n').map_or(b.len(), |e| i + e + 1),
            b'(' => {
                depth += 1;
                i += 1;
            }
            b')' => {
                depth = depth.saturating_sub(1);
                i += 1;
            }
            c if c.is_ascii_whitespace() => i += 1,
            _ => {
                let end = movetext[i..]
                    .find(|ch: char| ch.is_whitespace() || matches!(ch, '{' | '(' | ')' | ';'))
                    .map_or(b.len(), |e| i + e);
                let token = &movetext[i..end];
                i = end;
                if depth > 0 || truncated {
                    continue;
                }
                if let Some(r) = result_of(token) {
                    end_result = Some(r);
                    break;
                }
                if token == "*" {
                    break;
                }
                // "12.", "12...", "12.e4": keep what follows the move number.
                let raw = token.rsplit('.').next().unwrap_or("").trim_end_matches(['!', '?']);
                if raw.is_empty() || raw.starts_with('$') || raw.bytes().all(|c| c.is_ascii_digit()) {
                    continue;
                }
                // Castling is sometimes written with zeros.
                let san = if raw.starts_with("0-0") { raw.replace('0', "O") } else { raw.to_string() };
                match parse_san(&pos, &san) {
                    Some(m) => {
                        moves.push((m, None));
                        pos = pos.make(m);
                    }
                    None => truncated = true,
                }
            }
        }
    }
    let result = result.or(end_result)?;
    (!moves.is_empty()).then_some(Game { start, moves, result, truncated })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_moves_comments_and_variations() {
        crate::bitboard::init();
        let g = parse_game(
            "[Event \"x\"]\n[Result \"1-0\"]\n\n1. e4 {+0.30/12 0.5s} e5 {[%eval -0.25] [%clk 0:05:00]} 2. Nf3! (2. Bc4 Nf6 {+9.99/3}) \
             2... Nc6 $1 3. Bb5 ; rest of line\n a6 {book} 4. O-O 1-0\n",
        )
        .unwrap();
        assert_eq!(g.result, 1.0);
        assert!(!g.truncated);
        let uci: Vec<String> = g.moves.iter().map(|(m, _)| m.to_uci()).collect();
        assert_eq!(uci, ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "e1g1"]);
        assert_eq!(g.moves[0].1, Some(Eval::BeforeMover(30)));
        assert_eq!(g.moves[1].1, Some(Eval::AfterWhite(-25)));
        assert!(g.moves[2..].iter().all(|(_, e)| e.is_none()));
    }

    #[test]
    fn handles_results_start_positions_and_bad_moves() {
        crate::bitboard::init();
        assert!(parse_game("[Result \"*\"]\n\n1. e4 e5 *\n").is_none());
        // No Result tag: the movetext's own result is used.
        assert_eq!(parse_game("1. d4 d5 1/2-1/2").unwrap().result, 0.5);
        let g = parse_game("[FEN \"4k3/8/8/8/8/8/4P3/4K3 w - - 0 1\"]\n[Result \"0-1\"]\n\n1. e4 Kd7 2. Qh5 Kd6 0-1").unwrap();
        assert_eq!(g.moves.len(), 2);
        assert!(g.truncated);
        assert_eq!(g.result, 0.0);
        // Castling written with zeros, and mate scores, which carry no number.
        let g = parse_game("[Result \"1-0\"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. 0-0 {+M5/20} Nf6 {[%eval #-3]} 1-0").unwrap();
        assert_eq!(g.moves[6].0.to_uci(), "e1g1");
        assert!(g.moves[6].1.is_none() && g.moves[7].1.is_none());
    }

    #[test]
    fn splits_a_file_into_games() {
        let dir = std::env::temp_dir().join(format!("esper-pgn-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("two.pgn");
        std::fs::write(&path, "[Result \"1-0\"]\n\n1. e4 e5 1-0\n\n[Result \"0-1\"]\n[Event \"b\"]\n\n1. d4 d5\n2. c4 0-1\n").unwrap();
        let mut r = Reader::new(&[path.to_string_lossy().into_owned()]).unwrap();
        let a = r.next_game().unwrap();
        let b = r.next_game().unwrap();
        assert!(r.next_game().is_none());
        assert!(a.contains("1. e4") && !a.contains("d4"));
        assert!(b.contains("2. c4") && b.contains("Event"));
        std::fs::remove_dir_all(&dir).unwrap();
        assert!(Reader::new(&["/nonexistent/x.pgn".to_string()]).is_err());
    }
}
