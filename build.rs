// Compiles the vendored Fathom Syzygy probing library (third_party/fathom, MIT)
// and gives the Windows executable its icon. (The SIMD code path is chosen at
// run time: see src/simd.rs.)
fn main() {
    println!("cargo:rerun-if-changed=third_party/fathom");
    println!("cargo:rerun-if-changed=build.rs");
    let mut fathom = cc::Build::new();
    fathom.file("third_party/fathom/tbprobe.c").include("third_party/fathom").opt_level(3).warnings(false);
    // cc would otherwise copy rustc flags such as -Cprofile-generate onto the C
    // compiler. Apple's clang then instruments Fathom with its own (older) profile
    // format, which breaks PGO builds: the profiling runtime and instrumentation
    // disagree and the recorded profile is useless or the binary crashes.
    fathom.inherit_rustflags(false);
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
}
