//! Search parameters that can be tuned with SPSA (`tools/spsa.py`).
//!
//! Each one is also a UCI spin option of the same name. They are always
//! accepted by `setoption`, but only listed by `uci` in builds with the `tune`
//! feature, so GUIs normally don't see them.

use std::sync::atomic::{AtomicI32, Ordering::Relaxed};

pub struct Param {
    pub name: &'static str,
    value: AtomicI32,
    pub default: i32,
    pub min: i32,
    pub max: i32,
}

impl Param {
    const fn new(name: &'static str, default: i32, min: i32, max: i32) -> Param {
        Param { name, value: AtomicI32::new(default), default, min, max }
    }
    #[inline(always)]
    pub fn get(&self) -> i32 {
        #[cfg(feature = "tune")]
        {
            self.value.load(Relaxed)
        }
        #[cfg(not(feature = "tune"))]
        {
            self.default
        }
    }
    pub fn set(&self, v: i32) {
        self.value.store(v.clamp(self.min, self.max), Relaxed);
    }
}

macro_rules! params {
    ($($id:ident = $def:expr, $min:expr, $max:expr;)*) => {
        $(#[allow(non_upper_case_globals)] pub static $id: Param = Param::new(stringify!($id), $def, $min, $max);)*
        pub static ALL: &[&Param] = &[$(&$id),*];
    };
}

params! {
    // Evaluation scaling
    T_OptimismScale = 422, 200, 700;
    T_OptimismDiv = 91, 30, 200;
    // Quiet move futility (move loop)
    T_FutBase = 162, 50, 300;
    T_FutMult = 110, 50, 200;
    T_FutAboveAlpha = 97, 0, 200;
    // Reverse futility
    T_RfpImproving = 2658, 1000, 4000;
    // Razoring
    T_RazorMargin = 335, 150, 600;
    // Quiescence futility
    T_QsFutBase = 319, 100, 500;
    // Capture futility and exchange pruning (move loop)
    T_CapFutBase = 220, 100, 400;
    T_CapFutMult = 268, 100, 400;
    T_CapSeeMult = 177, 80, 300;
    // Hash-entry ProbCut
    T_SmallProbCut = 418, 200, 700;
    // Singular extensions
    T_SingularMargin = 58, 20, 120;
    T_DoubleExtMargin = -14, -100, 100;
    T_TripleExtMargin = 84, 0, 200;
    // LMR / LMP / Fortress / Time management
    T_LmrCutNode = 3720, 2000, 6000;
    T_LmpScale = 1038, 600, 1600;
    T_FortressRule50 = 31, 16, 60;
    T_TimeFalling = 230, 100, 400;
    T_TimeInstability = 2229, 1000, 3500;
}

/// Set a parameter by its UCI name; returns false if there is none of that name.
pub fn set_by_name(name: &str, value: &str) -> bool {
    match ALL.iter().find(|p| p.name.eq_ignore_ascii_case(name)) {
        Some(p) => {
            if let Ok(v) = value.trim().parse::<i32>() {
                p.set(v);
            }
            true
        }
        None => false,
    }
}

pub fn print_uci_options() {
    if cfg!(feature = "tune") {
        for p in ALL {
            println!("option name {} type spin default {} min {} max {}", p.name, p.default, p.min, p.max);
        }
    }
}
