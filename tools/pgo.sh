#!/usr/bin/env bash
# Profile-guided optimisation build of Esper.
#
#   tools/pgo.sh            # builds target/release/esper with PGO
#
# 1. builds an instrumented binary, 2. runs the bench (single- and multi-threaded)
# to record which code is hot, 3. rebuilds the release binary using that profile.
# build.rs stops the cc crate from forwarding -Cprofile-generate to the C compiler
# (Fathom); otherwise the system clang's older profile format breaks the run.
# Needs an llvm-profdata whose LLVM major version matches rustc's; set
# LLVM_PROFDATA to choose one, otherwise rustup's llvm-tools or Homebrew's llvm
# is used.
set -euo pipefail
cd "$(dirname "$0")/.."

llvm_major() { "$1" --version 2>/dev/null | grep -oE 'LLVM version [0-9]+' | grep -oE '[0-9]+$' || true; }
want=$(rustc -vV | sed -n 's/^LLVM version: \([0-9]*\).*/\1/p')

profdata=${LLVM_PROFDATA:-}
if [ -z "$profdata" ]; then
    for c in "$(dirname "$(rustup which rustc 2>/dev/null || echo /nonexistent)")/../lib/rustlib/"*/bin/llvm-profdata \
             "$(brew --prefix llvm 2>/dev/null || echo /nonexistent)/bin/llvm-profdata" \
             "$(command -v llvm-profdata || true)"; do
        if [ -x "$c" ] && [ "$(llvm_major "$c")" = "$want" ]; then
            profdata=$c
            break
        fi
    done
fi
if [ -z "$profdata" ]; then
    echo "error: no llvm-profdata for LLVM $want found; set LLVM_PROFDATA" >&2
    exit 1
fi

# RUSTFLAGS replaces .cargo/config.toml's rustflags, so keep the CPU target.
base_flags="-C target-cpu=${PGO_TARGET_CPU:-native}"
data=$(mktemp -d)
trap 'rm -rf "$data"' EXIT

# An explicit --target keeps RUSTFLAGS off build scripts, so only Esper itself
# is instrumented and optimised.
host=$(rustc -vV | sed -n 's/^host: //p')

echo "== instrumented build"
RUSTFLAGS="$base_flags -Cprofile-generate=$data" cargo build --release --target "$host" --target-dir target/pgo-gen "$@"

echo "== training run"
gen=target/pgo-gen/$host/release/esper
"$gen" bench 14 > /dev/null
"$gen" -t 4 bench 13 > /dev/null
"$gen" perftbench > /dev/null

echo "== merging profile with $profdata"
"$profdata" merge -o "$data/esper.profdata" "$data"
# A profile whose hottest function ran only a few hundred times means the
# profiling runtime and the instrumentation did not match (seen with both
# Homebrew and rustup Rust on macOS); building with it would only mislead LLVM.
max=$("$profdata" show "$data/esper.profdata" | sed -n 's/^Maximum function count: //p')
if [ "${max:-0}" -lt 100000 ]; then
    echo "error: the profile looks broken (maximum function count $max); PGO is not usable with this toolchain" >&2
    exit 1
fi

echo "== optimised build"
RUSTFLAGS="$base_flags -Cprofile-use=$data/esper.profdata -Cllvm-args=-pgo-warn-missing-function=false" \
    cargo build --release --target "$host" --target-dir target/pgo-use "$@"
mkdir -p target/release
cp "target/pgo-use/$host/release/esper" target/release/esper-pgo
echo "== done: target/release/esper-pgo"
