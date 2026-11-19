//! Bitboard primitives and attack tables (magic bitboards, found at startup).

pub const FILE_A: u64 = 0x0101_0101_0101_0101;
pub const FILE_H: u64 = FILE_A << 7;
pub const RANK_1: u64 = 0xFF;
pub const RANK_2: u64 = RANK_1 << 8;
pub const RANK_3: u64 = RANK_1 << 16;
pub const RANK_6: u64 = RANK_1 << 40;
pub const RANK_7: u64 = RANK_1 << 48;
pub const RANK_8: u64 = RANK_1 << 56;

#[inline(always)]
pub fn bit(sq: usize) -> u64 {
    1u64 << sq
}
#[inline(always)]
pub fn lsb(b: u64) -> usize {
    b.trailing_zeros() as usize
}
#[inline(always)]
pub fn pop_lsb(b: &mut u64) -> usize {
    let s = b.trailing_zeros() as usize;
    *b &= *b - 1;
    s
}
#[inline(always)]
pub fn popcount(b: u64) -> u32 {
    b.count_ones()
}
#[inline(always)]
pub fn more_than_one(b: u64) -> bool {
    b & b.wrapping_sub(1) != 0
}

#[derive(Clone, Copy, Default)]
struct Magic {
    mask: u64,
    magic: u64,
    shift: u32,
    offset: usize,
}

pub struct Tables {
    pub pawn: [[u64; 64]; 2],
    pub knight: [u64; 64],
    pub king: [u64; 64],
    pub rook_empty: [u64; 64],
    pub bishop_empty: [u64; 64],
    pub between: [[u64; 64]; 64],
    pub line: [[u64; 64]; 64],
    rook: [Magic; 64],
    bishop: [Magic; 64],
    attacks: Vec<u64>,
}

static mut TABLES: *const Tables = std::ptr::null();

#[inline(always)]
pub fn tables() -> &'static Tables {
    // SAFETY: init() is called once at program start before any other use.
    unsafe { &*TABLES }
}

const ROOK_DIRS: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
const BISHOP_DIRS: [(i32, i32); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];

fn slide(sq: usize, occ: u64, dirs: &[(i32, i32); 4]) -> u64 {
    let mut a = 0u64;
    for &(df, dr) in dirs {
        let (mut f, mut r) = ((sq & 7) as i32 + df, (sq >> 3) as i32 + dr);
        while (0..8).contains(&f) && (0..8).contains(&r) {
            let s = (r * 8 + f) as usize;
            a |= bit(s);
            if occ & bit(s) != 0 {
                break;
            }
            f += df;
            r += dr;
        }
    }
    a
}

fn step(sq: usize, deltas: &[(i32, i32)]) -> u64 {
    let mut a = 0;
    for &(df, dr) in deltas {
        let (f, r) = ((sq & 7) as i32 + df, (sq >> 3) as i32 + dr);
        if (0..8).contains(&f) && (0..8).contains(&r) {
            a |= bit((r * 8 + f) as usize);
        }
    }
    a
}

// Magic multipliers, found once by random search and verified again at startup.
#[rustfmt::skip]
const ROOK_MAGICS: [u64; 64] = [
    0x1080004008801020, 0x0840092002C03000, 0x1900200010400900, 0x0880100008000480,
    0x4200100420080200, 0x8100020100080400, 0x0200040110886200, 0x0200008040220411,
    0x0404800084400220, 0x0000401000402000, 0x0086001081220440, 0x0408800800100280,
    0x000A001201040820, 0x8848800200840080, 0x4001000100040200, 0x0442000102105084,
    0x9080010020804100, 0x0040404000201009, 0x0000808010002009, 0x2200090021D00100,
    0x0008008008040080, 0x0004004002010040, 0x0011040008015042, 0x00000A0001768104,
    0x0000800080204009, 0x2010004140002001, 0x9800200280100080, 0x1000100080080080,
    0x0442000A00049020, 0x2100040080020080, 0x0800120400900148, 0x0010040A00128541,
    0x2800804000800030, 0x1010002000400041, 0x4000200011004100, 0x0610008410800800,
    0x0400802402800800, 0xC100020080800400, 0x0002000802000401, 0x0182085882000401,
    0x0220204000808000, 0x2860100040024022, 0x0001002004110040, 0x99101042000A0020,
    0x0004080004008080, 0x0010040002008080, 0x2012004881020004, 0x8300842444820011,
    0x0088403882010200, 0x0820400080210100, 0x0110910040A00300, 0x0801100280080480,
    0x0242009008200600, 0x1002000489500200, 0x0040800200010080, 0x0091800041000080,
    0x0000209300488001, 0x04C1002414824001, 0x020020000B001041, 0x7000100004200901,
    0x8002002004100802, 0x30010002084C0007, 0x0888221800813004, 0x4000002840840112,
];
#[rustfmt::skip]
const BISHOP_MAGICS: [u64; 64] = [
    0xA010041108003100, 0x006082020A002900, 0x6810010619200000, 0x08281A0520000408,
    0x0001104001000400, 0x0018901008048400, 0x00040A0210245280, 0x000200210808A402,
    0x9140048410821200, 0x0800091010820041, 0x20504804832202C0, 0x0100091401081000,
    0x8021011140000012, 0x0810020804450400, 0x208B0542109008A2, 0x0080084A08040204,
    0x0040E2A80811244C, 0x2505022008008108, 0x0430220100420040, 0x010A040420220040,
    0x1105000290400000, 0x0093001200822120, 0x4000A62048043004, 0x280120048A015004,
    0x006090002A020814, 0x44042000240800D0, 0x01102800040A4400, 0x1004080080220040,
    0x0001001011004024, 0x0010044000805040, 0x0914041200820100, 0x0004821012821480,
    0x0024040500C05021, 0x0088611002080200, 0x0116080A00040020, 0x4000020080080080,
    0x2450450140840040, 0x0000880201484100, 0x0222020404020092, 0x8081110600002E00,
    0x2842101105000801, 0x1100809008001025, 0x00020202221C0400, 0x0422014022009020,
    0x0210046102100C00, 0xC004008082029102, 0x00AA461801101200, 0x0404080080201108,
    0x020542108C205002, 0x0410544804100100, 0x0040910841100000, 0x0400200042021100,
    0x00004204850400C0, 0x0200100410A42102, 0x1040020801210102, 0x0805040410420000,
    0x2884804130100200, 0x800C262201242000, 0x1058000194108800, 0x0014221054420204,
    0x0104000012A02200, 0x0200881003300100, 0x0140400202840100, 0x0402020801010201,
];

fn fill_magics(dirs: &[(i32, i32); 4], magics: &[u64; 64], out: &mut [Magic; 64], attacks: &mut Vec<u64>) {
    for sq in 0..64 {
        let edges = ((RANK_1 | RANK_8) & !(RANK_1 << (8 * (sq >> 3)))) | ((FILE_A | FILE_H) & !(FILE_A << (sq & 7)));
        let mask = slide(sq, 0, dirs) & !edges;
        let bits = popcount(mask);
        let shift = 64 - bits;
        let magic = magics[sq];
        let offset = attacks.len();
        attacks.resize(offset + (1usize << bits), 0);
        // Enumerate all subsets of the mask (carry-rippler).
        let mut sub = 0u64;
        loop {
            #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
            let idx = offset + unsafe { std::arch::x86_64::_pext_u64(sub, mask) } as usize;
            #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
            let idx = offset + (sub.wrapping_mul(magic) >> shift) as usize;
            let att = slide(sq, sub, dirs);
            assert!(attacks[idx] == 0 || attacks[idx] == att, "bad magic");
            attacks[idx] = att;
            sub = sub.wrapping_sub(mask) & mask;
            if sub == 0 {
                break;
            }
        }
        out[sq] = Magic { mask, magic, shift, offset };
    }
}

pub fn init() {
    let mut t = Box::new(Tables {
        pawn: [[0; 64]; 2],
        knight: [0; 64],
        king: [0; 64],
        rook_empty: [0; 64],
        bishop_empty: [0; 64],
        between: [[0; 64]; 64],
        line: [[0; 64]; 64],
        rook: [Magic::default(); 64],
        bishop: [Magic::default(); 64],
        attacks: Vec::new(),
    });
    for sq in 0..64 {
        t.pawn[0][sq] = step(sq, &[(-1, 1), (1, 1)]);
        t.pawn[1][sq] = step(sq, &[(-1, -1), (1, -1)]);
        t.knight[sq] = step(sq, &[(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]);
        t.king[sq] = step(sq, &[(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]);
        t.rook_empty[sq] = slide(sq, 0, &ROOK_DIRS);
        t.bishop_empty[sq] = slide(sq, 0, &BISHOP_DIRS);
    }
    for a in 0..64 {
        for b in 0..64 {
            if a == b {
                continue;
            }
            for (tab, dirs) in [(&t.rook_empty, &ROOK_DIRS), (&t.bishop_empty, &BISHOP_DIRS)] {
                if tab[a] & bit(b) != 0 {
                    let between = slide(a, bit(b), dirs) & slide(b, bit(a), dirs);
                    let line = (tab[a] & tab[b]) | bit(a) | bit(b);
                    t.between[a][b] = between;
                    t.line[a][b] = line;
                }
            }
        }
    }
    let mut attacks = Vec::with_capacity(102_400 + 5_248);
    let mut rook = [Magic::default(); 64];
    let mut bishop = [Magic::default(); 64];
    fill_magics(&ROOK_DIRS, &ROOK_MAGICS, &mut rook, &mut attacks);
    fill_magics(&BISHOP_DIRS, &BISHOP_MAGICS, &mut bishop, &mut attacks);
    t.rook = rook;
    t.bishop = bishop;
    t.attacks = attacks;
    unsafe {
        TABLES = Box::leak(t);
    }
}

#[inline(always)]
pub fn pawn_attacks(color: usize, sq: usize) -> u64 {
    tables().pawn[color][sq]
}
#[inline(always)]
pub fn knight_attacks(sq: usize) -> u64 {
    tables().knight[sq]
}
#[inline(always)]
pub fn king_attacks(sq: usize) -> u64 {
    tables().king[sq]
}
#[inline(always)]
pub fn rook_attacks(sq: usize, occ: u64) -> u64 {
    let t = tables();
    let m = &t.rook[sq];
    #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
    let idx = m.offset + unsafe { std::arch::x86_64::_pext_u64(occ, m.mask) } as usize;
    #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
    let idx = m.offset + ((occ & m.mask).wrapping_mul(m.magic) >> m.shift) as usize;
    unsafe { *t.attacks.get_unchecked(idx) }
}
#[inline(always)]
pub fn bishop_attacks(sq: usize, occ: u64) -> u64 {
    let t = tables();
    let m = &t.bishop[sq];
    #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
    let idx = m.offset + unsafe { std::arch::x86_64::_pext_u64(occ, m.mask) } as usize;
    #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
    let idx = m.offset + ((occ & m.mask).wrapping_mul(m.magic) >> m.shift) as usize;
    unsafe { *t.attacks.get_unchecked(idx) }
}
#[inline(always)]
pub fn between(a: usize, b: usize) -> u64 {
    tables().between[a][b]
}
#[inline(always)]
pub fn line(a: usize, b: usize) -> u64 {
    tables().line[a][b]
}
