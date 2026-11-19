// Compiles the vendored Fathom Syzygy probing library (third_party/fathom, MIT)
// and selects which hand-written SIMD paths the evaluation uses.
fn main() {
    println!("cargo:rerun-if-changed=third_party/fathom");
    println!("cargo:rerun-if-changed=build.rs");
    let mut fathom = cc::Build::new();
    fathom.file("third_party/fathom/tbprobe.c").include("third_party/fathom").opt_level(3).warnings(false);
    if fathom.get_compiler().is_like_msvc() {
        // Fathom uses <stdatomic.h>, which MSVC only provides in C11 mode with
        // its (still "experimental") C11 atomics enabled.
        fathom.flag("/std:c11").flag("/experimental:c11atomics");
    } else {
        fathom.flag_if_supported("-std=gnu11");
    }
    fathom.compile("fathom");

    // Give esper.exe the Esper logo (esper.ico, rendered from esper.svg) as its icon.
    println!("cargo:rerun-if-changed=esper.rc");
    println!("cargo:rerun-if-changed=esper.ico");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        embed_resource::compile("esper.rc", embed_resource::NONE).manifest_optional().unwrap();
    }

    // SIMD selection is a compile-time decision based on the enabled target
    // features (e.g. from `-C target-cpu=native`).
    const FLAGS: [&str; 7] =
        ["simd_neon", "simd_neon_dot", "simd_avx2", "simd_avx512", "simd_vnni", "simd_x86", "simd_none"];
    for f in FLAGS {
        println!("cargo:rustc-check-cfg=cfg({f})");
    }
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let features = std::env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
    let has = |f: &str| features.split(',').any(|x| x == f);
    let portable = std::env::var_os("CARGO_FEATURE_PORTABLE").is_some();
    let on = |f: &str| println!("cargo:rustc-cfg={f}");
    if portable {
        on("simd_none");
    } else if arch == "aarch64" && has("neon") {
        on("simd_neon");
        if has("dotprod") {
            on("simd_neon_dot");
        }
    } else if arch == "x86_64" && has("avx512f") && has("avx512bw") {
        on("simd_avx512");
        on("simd_x86");
        if has("avx512vnni") {
            on("simd_vnni");
        }
    } else if arch == "x86_64" && has("avx2") {
        on("simd_avx2");
        on("simd_x86");
    } else {
        on("simd_none");
    }
}
