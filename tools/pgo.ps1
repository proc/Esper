# Profile-guided optimisation build of Esper on Windows (PowerShell).
#
#   powershell -ExecutionPolicy Bypass -File tools\pgo.ps1            # builds target\release\esper-pgo.exe
#   powershell -ExecutionPolicy Bypass -File tools\pgo.ps1 -Install   # ... and copies it to target\release\esper.exe
#
# The Windows counterpart of tools/pgo.sh: 1. builds an instrumented binary,
# 2. runs the bench (single- and multi-threaded) to record which code is hot,
# 3. rebuilds the release binary using that profile.
# Needs an llvm-profdata whose LLVM major version matches rustc's; set
# $env:LLVM_PROFDATA to choose one, otherwise rustup's llvm-tools component is
# used (rustup component add llvm-tools).
# Set $env:PGO_TARGET_CPU (for example x86-64-v3) for a binary that must run on
# other machines; the default is this machine's CPU.
# Extra arguments are passed on to both cargo builds.
param(
    [switch]$Install,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CargoArgs = @()
)
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')

function Fail($msg) {
    Write-Host "error: $msg" -ForegroundColor Red
    exit 1
}

# Run a native command and stop the script if it fails.
function Run($exe, [string[]]$arguments) {
    & $exe @arguments
    if ($LASTEXITCODE -ne 0) { Fail "$exe failed (exit code $LASTEXITCODE)" }
}

function LlvmMajor($exe) {
    try {
        $out = (& $exe --version 2>$null) -join "`n"
        if ($out -match 'LLVM version (\d+)') { return $Matches[1] }
    } catch {}
    return $null
}

$rustcInfo = (& rustc -vV) -join "`n"
if ($rustcInfo -notmatch '(?m)^host: (\S+)') { Fail 'cannot read the host target from rustc -vV' }
$hostTriple = $Matches[1]
if ($rustcInfo -notmatch '(?m)^LLVM version: (\d+)') { Fail 'cannot read the LLVM version from rustc -vV' }
$want = $Matches[1]

$profdata = $env:LLVM_PROFDATA
if (-not $profdata) {
    $candidates = @()
    try {
        $rustc = (& rustup which rustc 2>$null)
        if ($LASTEXITCODE -eq 0 -and $rustc) {
            $toolchain = Split-Path (Split-Path $rustc)
            $candidates += Join-Path $toolchain "lib\rustlib\$hostTriple\bin\llvm-profdata.exe"
        }
    } catch {}
    $onPath = Get-Command llvm-profdata -ErrorAction SilentlyContinue
    if ($onPath) { $candidates += $onPath.Source }
    foreach ($c in $candidates) {
        if ((Test-Path $c) -and ((LlvmMajor $c) -eq $want)) {
            $profdata = $c
            break
        }
    }
}
if (-not $profdata) {
    Fail "no llvm-profdata for LLVM $want found; run 'rustup component add llvm-tools' or set LLVM_PROFDATA"
}

# RUSTFLAGS replaces .cargo/config.toml's rustflags, so keep the CPU target.
$cpu = if ($env:PGO_TARGET_CPU) { $env:PGO_TARGET_CPU } else { 'native' }
$baseFlags = "-C target-cpu=$cpu"
$data = Join-Path ([System.IO.Path]::GetTempPath()) ("esper-pgo-" + [System.Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $data | Out-Null
$savedFlags = $env:RUSTFLAGS

try {
    # An explicit --target keeps RUSTFLAGS off build scripts, so only Esper
    # itself is instrumented and optimised.
    Write-Host '== instrumented build'
    $env:RUSTFLAGS = "$baseFlags -Cprofile-generate=$data"
    Run cargo (@('build', '--release', '--target', $hostTriple, '--target-dir', 'target/pgo-gen') + $CargoArgs)

    Write-Host '== training run'
    $gen = "target\pgo-gen\$hostTriple\release\esper.exe"
    & $gen bench 14 | Out-Null
    if ($LASTEXITCODE -ne 0) { Fail 'training run failed (bench 14)' }
    & $gen -t 4 bench 13 | Out-Null
    if ($LASTEXITCODE -ne 0) { Fail 'training run failed (-t 4 bench 13)' }
    & $gen perftbench | Out-Null
    if ($LASTEXITCODE -ne 0) { Fail 'training run failed (perftbench)' }

    Write-Host "== merging profile with $profdata"
    $merged = Join-Path $data 'esper.profdata'
    Run $profdata @('merge', '-o', $merged, $data)
    # A profile whose hottest function ran only a few hundred times means the
    # profiling runtime and the instrumentation did not match; building with it
    # would only mislead LLVM.
    $show = (& $profdata show $merged) -join "`n"
    $max = 0
    if ($show -match 'Maximum function count: (\d+)') { $max = [int64]$Matches[1] }
    if ($max -lt 100000) {
        Fail "the profile looks broken (maximum function count $max); PGO is not usable with this toolchain"
    }

    Write-Host '== optimised build'
    $env:RUSTFLAGS = "$baseFlags -Cprofile-use=$merged -Cllvm-args=-pgo-warn-missing-function=false"
    Run cargo (@('build', '--release', '--target', $hostTriple, '--target-dir', 'target/pgo-use') + $CargoArgs)

    New-Item -ItemType Directory -Force -Path 'target\release' | Out-Null
    Copy-Item "target\pgo-use\$hostTriple\release\esper.exe" 'target\release\esper-pgo.exe' -Force
    Write-Host '== done: target\release\esper-pgo.exe'
    if ($Install) {
        # Fails if esper.exe is running (a GUI has it open): close that first.
        Copy-Item 'target\release\esper-pgo.exe' 'target\release\esper.exe' -Force
        Write-Host '== installed as target\release\esper.exe'
    }
}
finally {
    $env:RUSTFLAGS = $savedFlags
    Remove-Item -Recurse -Force $data -ErrorAction SilentlyContinue
}
