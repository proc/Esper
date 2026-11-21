//! Which hand-written SIMD code paths this CPU can run, decided once at run time.
//!
//! Every path for the target architecture is compiled into the binary; the
//! evaluation picks one per call from `level()`. A build whose compile-time
//! target already guarantees the best path (Apple silicon, or
//! `-C target-cpu=native` on an AVX-512 VNNI machine) resolves `level()` to a
//! constant, so the choice costs nothing there.
//!
//! The environment variable `ESPER_SIMD` (scalar, neon, neon-dot, avx2, avx512,
//! avx512-vnni) selects a lower path than the detected one, for testing and for
//! profile-guided builds; a path the CPU cannot run is never selected.

use std::sync::atomic::{AtomicU8, Ordering};

/// SIMD paths, ordered within each architecture from least to most capable.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
#[repr(u8)]
pub enum Level {
    /// Portable code, vectorised by the compiler for the build's baseline CPU.
    Scalar = 0,
    Neon = 1,
    /// NEON with the dot-product instructions.
    NeonDot = 2,
    Avx2 = 3,
    /// AVX-512 F and BW.
    Avx512 = 4,
    /// AVX-512 with VNNI.
    Avx512Vnni = 5,
}

impl Level {
    pub fn name(self) -> &'static str {
        match self {
            Level::Scalar => "scalar",
            Level::Neon => "neon",
            Level::NeonDot => "neon-dot",
            Level::Avx2 => "avx2",
            Level::Avx512 => "avx512",
            Level::Avx512Vnni => "avx512-vnni",
        }
    }

    fn from_name(s: &str) -> Option<Level> {
        ALL.into_iter().find(|l| l.name().eq_ignore_ascii_case(s.trim()))
    }

    fn from_u8(v: u8) -> Level {
        ALL.into_iter().find(|&l| l as u8 == v).unwrap_or(Level::Scalar)
    }

    fn is_arm(self) -> bool {
        matches!(self, Level::Neon | Level::NeonDot)
    }

    fn is_x86(self) -> bool {
        matches!(self, Level::Avx2 | Level::Avx512 | Level::Avx512Vnni)
    }
}

const ALL: [Level; 6] = [Level::Scalar, Level::Neon, Level::NeonDot, Level::Avx2, Level::Avx512, Level::Avx512Vnni];

const UNSET: u8 = u8::MAX;
static LEVEL: AtomicU8 = AtomicU8::new(UNSET);

/// The best path the CPU supports.
fn detect() -> Level {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            Level::NeonDot
        } else {
            Level::Neon
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        let avx2 = std::arch::is_x86_feature_detected!("avx2");
        let avx512 =
            avx2 && std::arch::is_x86_feature_detected!("avx512f") && std::arch::is_x86_feature_detected!("avx512bw");
        if avx512 && std::arch::is_x86_feature_detected!("avx512vnni") {
            Level::Avx512Vnni
        } else if avx512 {
            Level::Avx512
        } else if avx2 {
            Level::Avx2
        } else {
            Level::Scalar
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        Level::Scalar
    }
}

/// `want` if this CPU (whose best path is `best`) can run it, otherwise `best`.
fn capped(best: Level, want: Level) -> Level {
    let same_family = want == Level::Scalar || (want.is_arm() && best.is_arm()) || (want.is_x86() && best.is_x86());
    if same_family && want <= best {
        want
    } else {
        best
    }
}

#[cold]
#[inline(never)]
fn init() -> Level {
    let best = detect();
    let level = match std::env::var("ESPER_SIMD").ok().as_deref().and_then(Level::from_name) {
        Some(want) => capped(best, want),
        None => best,
    };
    LEVEL.store(level as u8, Ordering::Relaxed);
    level
}

/// The SIMD path in use. Only ever a path the running CPU supports.
#[inline(always)]
pub fn level() -> Level {
    if cfg!(feature = "portable") {
        return Level::Scalar;
    }
    // The build target already guarantees the best path: nothing to decide.
    #[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
    {
        return Level::NeonDot;
    }
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vnni"
    ))]
    {
        return Level::Avx512Vnni;
    }
    #[allow(unreachable_code)]
    {
        let v = LEVEL.load(Ordering::Relaxed);
        if v != UNSET {
            Level::from_u8(v)
        } else {
            init()
        }
    }
}

/// CPU features the build itself assumes (its `target-cpu`) that this CPU lacks,
/// or `None` when it can run. Checked at startup so an unsuitable CPU gets a
/// message rather than an illegal-instruction crash.
pub fn missing_baseline() -> Option<&'static str> {
    #[cfg(target_arch = "x86_64")]
    {
        macro_rules! need {
            ($($f:tt),*) => {
                $(if cfg!(target_feature = $f) && !std::arch::is_x86_feature_detected!($f) {
                    return Some($f);
                })*
            };
        }
        need!("sse3", "ssse3", "sse4.1", "sse4.2", "popcnt", "avx", "avx2", "bmi1", "bmi2", "lzcnt", "fma");
        need!("avx512f", "avx512bw", "avx512vnni");
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn override_never_exceeds_the_cpu() {
        assert_eq!(capped(Level::Avx2, Level::Avx512), Level::Avx2);
        assert_eq!(capped(Level::Avx512Vnni, Level::Avx2), Level::Avx2);
        assert_eq!(capped(Level::Avx2, Level::Scalar), Level::Scalar);
        assert_eq!(capped(Level::Avx2, Level::Neon), Level::Avx2);
        assert_eq!(capped(Level::NeonDot, Level::Neon), Level::Neon);
        assert_eq!(capped(Level::NeonDot, Level::Avx2), Level::NeonDot);
        assert_eq!(capped(Level::Scalar, Level::Avx2), Level::Scalar);
        for l in ALL {
            assert_eq!(Level::from_name(l.name()), Some(l));
            assert_eq!(Level::from_u8(l as u8), l);
        }
    }
}
