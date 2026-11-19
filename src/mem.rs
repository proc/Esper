//! Large, 2 MB-aligned allocations backed by huge pages where the OS allows it.
//!
//! The network's threat weights (about 66 MB) and the transposition table are
//! read at random, so with 4 KB pages most accesses miss the TLB. On Linux an
//! anonymous mapping aligned to 2 MB and marked `MADV_HUGEPAGE` is served from
//! transparent huge pages, as Stockfish arranges for its network and table.
//! Elsewhere (macOS on Apple silicon has no user huge pages) the callers keep
//! their usual allocations.

/// Whether large-page allocations are worth using on this platform.
pub const ENABLED: bool = cfg!(target_os = "linux");

const HUGE: usize = 2 * 1024 * 1024;

/// A zero-filled, 2 MB-aligned anonymous mapping. Unmapped on drop.
pub struct LargeAlloc {
    ptr: *mut u8,
    #[allow(dead_code)]
    len: usize,
    map: *mut u8,
    map_len: usize,
}

unsafe impl Send for LargeAlloc {}
unsafe impl Sync for LargeAlloc {}

#[cfg(unix)]
mod sys {
    use core::ffi::{c_int, c_void};
    pub const PROT_READ: c_int = 1;
    pub const PROT_WRITE: c_int = 2;
    pub const MAP_PRIVATE: c_int = 2;
    #[cfg(target_os = "linux")]
    pub const MAP_ANONYMOUS: c_int = 0x20;
    #[cfg(not(target_os = "linux"))]
    pub const MAP_ANONYMOUS: c_int = 0x1000;
    #[cfg(target_os = "linux")]
    pub const MADV_HUGEPAGE: c_int = 14;
    extern "C" {
        pub fn mmap(addr: *mut c_void, len: usize, prot: c_int, flags: c_int, fd: c_int, offset: i64) -> *mut c_void;
        pub fn munmap(addr: *mut c_void, len: usize) -> c_int;
        #[cfg(target_os = "linux")]
        pub fn madvise(addr: *mut c_void, len: usize, advice: c_int) -> c_int;
    }
}

impl LargeAlloc {
    /// `len` zeroed bytes aligned to 2 MB, or `None` if the mapping fails.
    #[cfg(unix)]
    pub fn new(len: usize) -> Option<LargeAlloc> {
        use core::ffi::c_void;
        let len = len.max(1).div_ceil(HUGE) * HUGE;
        // Over-allocate by one huge page so an aligned start always fits.
        let map_len = len + HUGE;
        // SAFETY: plain anonymous mapping; the result is checked below.
        let map = unsafe {
            sys::mmap(
                core::ptr::null_mut(),
                map_len,
                sys::PROT_READ | sys::PROT_WRITE,
                sys::MAP_PRIVATE | sys::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if map.is_null() || map as usize == usize::MAX {
            return None;
        }
        let map = map as *mut u8;
        let ptr = ((map as usize).div_ceil(HUGE) * HUGE) as *mut u8;
        #[cfg(target_os = "linux")]
        // SAFETY: the range lies inside the mapping made above.
        unsafe {
            sys::madvise(ptr as *mut c_void, len, sys::MADV_HUGEPAGE);
        }
        let _ = core::mem::size_of::<c_void>();
        Some(LargeAlloc { ptr, len, map, map_len })
    }

    #[cfg(not(unix))]
    pub fn new(_len: usize) -> Option<LargeAlloc> {
        None
    }

    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.len
    }

    /// A copy of `bytes` in large pages, borrowed for the life of the program
    /// (the allocation is returned so its owner can keep it alive).
    pub fn copy_of(bytes: &[u8]) -> Option<(LargeAlloc, &'static [u8])> {
        let a = LargeAlloc::new(bytes.len())?;
        // SAFETY: the allocation holds at least `bytes.len()` bytes and does not
        // overlap `bytes`; it lives as long as the returned owner.
        let data = unsafe {
            core::ptr::copy_nonoverlapping(bytes.as_ptr(), a.ptr, bytes.len());
            core::slice::from_raw_parts(a.ptr as *const u8, bytes.len())
        };
        Some((a, data))
    }
}

impl Drop for LargeAlloc {
    fn drop(&mut self) {
        #[cfg(unix)]
        // SAFETY: unmaps exactly the mapping made in `new`.
        unsafe {
            sys::munmap(self.map as *mut core::ffi::c_void, self.map_len);
        }
        let _ = (self.map, self.map_len);
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    fn aligned_zeroed_and_copied() {
        let a = LargeAlloc::new(3 * 1024 * 1024 + 5).unwrap();
        assert_eq!(a.as_ptr() as usize % HUGE, 0);
        assert!(a.len() >= 3 * 1024 * 1024 + 5);
        let s = unsafe { core::slice::from_raw_parts(a.as_ptr(), a.len()) };
        assert!(s.iter().all(|&b| b == 0));
        let src: Vec<u8> = (0..100_000u32).map(|i| i as u8).collect();
        let (_owner, copy) = LargeAlloc::copy_of(&src).unwrap();
        assert_eq!(copy, &src[..]);
    }
}
