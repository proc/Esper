//! Persistent experience / learning file.
//!
//! Stores high-depth root search results (`key -> best move, score, depth`) on
//! disk so recurring middlegame and endgame positions can seed the transposition
//! table immediately on subsequent games.

use crate::position::Move;
use std::collections::HashMap;
use std::io::Write;

#[derive(Clone, Copy)]
pub struct ExpEntry {
    pub mv: Move,
    pub score: i16,
    pub depth: i16,
    pub count: u16,
}

pub struct Experience {
    entries: HashMap<u64, ExpEntry>,
    dirty: bool,
}

impl Experience {
    pub fn empty() -> Experience {
        Experience { entries: HashMap::new(), dirty: false }
    }

    pub fn load(path: &str) -> Result<Experience, String> {
        let data = match std::fs::read(path) {
            Ok(d) => d,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Experience::empty()),
            Err(e) => return Err(format!("{path}: {e}")),
        };
        if data.len() % 16 != 0 {
            return Err(format!("{path}: truncated experience file ({} bytes)", data.len()));
        }
        let mut entries: HashMap<u64, ExpEntry> = HashMap::with_capacity(data.len() / 16);
        for chunk in data.chunks_exact(16) {
            let key = u64::from_le_bytes(chunk[0..8].try_into().unwrap());
            let mv = Move(u16::from_le_bytes(chunk[8..10].try_into().unwrap()));
            let score = i16::from_le_bytes(chunk[10..12].try_into().unwrap());
            let depth = i16::from_le_bytes(chunk[12..14].try_into().unwrap());
            let count = u16::from_le_bytes(chunk[14..16].try_into().unwrap());
            let e = ExpEntry { mv, score, depth, count };
            entries
                .entry(key)
                .and_modify(|old| {
                    if e.depth >= old.depth {
                        *old = e;
                    }
                })
                .or_insert(e);
        }
        Ok(Experience { entries, dirty: false })
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[inline]
    pub fn probe(&self, key: u64) -> Option<ExpEntry> {
        self.entries.get(&key).copied()
    }

    /// Record a completed root search result, keeping the deeper entry.
    pub fn record(&mut self, key: u64, mv: Move, score: i32, depth: i32) {
        if mv.is_none() || depth < 6 {
            return;
        }
        let score = score.clamp(i16::MIN as i32 + 1, i16::MAX as i32) as i16;
        let depth = depth.clamp(0, i16::MAX as i32) as i16;
        match self.entries.get_mut(&key) {
            Some(old) => {
                old.count = old.count.saturating_add(1);
                if depth >= old.depth {
                    old.mv = mv;
                    old.score = score;
                    old.depth = depth;
                    self.dirty = true;
                }
            }
            None => {
                self.entries.insert(key, ExpEntry { mv, score, depth, count: 1 });
                self.dirty = true;
            }
        }
    }

    pub fn save(&mut self, path: &str) -> Result<(), String> {
        if !self.dirty || path.is_empty() {
            return Ok(());
        }
        let mut buf = Vec::with_capacity(self.entries.len() * 16);
        for (&key, e) in &self.entries {
            buf.extend_from_slice(&key.to_le_bytes());
            buf.extend_from_slice(&e.mv.0.to_le_bytes());
            buf.extend_from_slice(&e.score.to_le_bytes());
            buf.extend_from_slice(&e.depth.to_le_bytes());
            buf.extend_from_slice(&e.count.to_le_bytes());
        }
        if let Some(parent) = std::path::Path::new(path).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|e| format!("{path}: {e}"))?;
            }
        }
        let mut f = std::fs::File::create(path).map_err(|e| format!("{path}: {e}"))?;
        f.write_all(&buf).map_err(|e| format!("{path}: {e}"))?;
        self.dirty = false;
        Ok(())
    }
}
