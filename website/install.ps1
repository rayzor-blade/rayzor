<#
    Rayzor installer for Windows.

        irm https://rayzor.tech/install.ps1 | iex

    Downloads the nightly build, verifies its checksum, and installs it under
    %USERPROFILE%\.rayzor. Nothing is written outside that directory, and the
    download is self-contained -- no LLVM installation is required.
#>

$ErrorActionPreference = 'Stop'

$Repo    = if ($env:RAYZOR_REPO)    { $env:RAYZOR_REPO }    else { 'rayzor-blade/rayzor' }
$Channel = if ($env:RAYZOR_CHANNEL) { $env:RAYZOR_CHANNEL } else { 'nightly' }
$Prefix  = if ($env:RAYZOR_PREFIX)  { $env:RAYZOR_PREFIX }  else { Join-Path $HOME '.rayzor' }
$BinDir  = Join-Path $Prefix 'bin'

function Fail($msg) { Write-Host "install: $msg" -ForegroundColor Red; exit 1 }

# Only an x64 build is published. Windows on ARM runs x64 through emulation, so
# it is installable -- just say so rather than appearing to ship a native one.
$arch = $env:PROCESSOR_ARCHITECTURE
switch ($arch) {
    'AMD64' { $asset = 'rayzor-windows-x86_64.zip' }
    'ARM64' {
        $asset = 'rayzor-windows-x86_64.zip'
        Write-Host 'rayzor: no native ARM64 build yet; installing the x64 build to run under emulation.'
    }
    default { Fail "unsupported architecture: $arch" }
}

$base = "https://github.com/$Repo/releases/download/$Channel"
$tmp  = Join-Path ([System.IO.Path]::GetTempPath()) ("rayzor-" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

try {
    Write-Host "rayzor: fetching $asset ($Channel)"
    $zip = Join-Path $tmp $asset
    try {
        Invoke-WebRequest -Uri "$base/$asset" -OutFile $zip -UseBasicParsing
    } catch {
        Fail "no build for windows-$arch in the $Channel release.`n    See https://github.com/$Repo/releases/$Channel"
    }

    # Verify when a checksum is published; a corrupted download otherwise fails
    # later as something that looks like a compiler bug.
    $sumFile = "$zip.sha256"
    $haveSum = $true
    try {
        Invoke-WebRequest -Uri "$base/$asset.sha256" -OutFile $sumFile -UseBasicParsing
    } catch {
        $haveSum = $false
        Write-Host "rayzor: no published checksum for $asset; skipping verification"
    }
    if ($haveSum) {
        $expected = ((Get-Content $sumFile -Raw).Trim() -split '\s+')[0].ToLower()
        $actual   = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
        if ($expected -ne $actual) {
            Fail "checksum mismatch for $asset`n    expected $expected`n    actual   $actual"
        }
    }

    New-Item -ItemType Directory -Force -Path $BinDir | Out-Null
    Expand-Archive -Path $zip -DestinationPath $tmp -Force
    $exe = Get-ChildItem -Path $tmp -Filter 'rayzor.exe' -Recurse | Select-Object -First 1
    if (-not $exe) { Fail 'archive did not contain rayzor.exe' }
    Copy-Item $exe.FullName (Join-Path $BinDir 'rayzor.exe') -Force
    # The CLI finds the wasm optimizer beside its own executable, so it has to
    # land in the same directory when the build carries one.
    Get-ChildItem -Path $tmp -Filter 'rayzor-wasm-opt.exe' -Recurse |
        Select-Object -First 1 |
        ForEach-Object { Copy-Item $_.FullName (Join-Path $BinDir 'rayzor-wasm-opt.exe') -Force }

    $version = & (Join-Path $BinDir 'rayzor.exe') --version 2>$null
    if (-not $version) { $version = 'unknown' }
    Write-Host "rayzor: installed $version to $BinDir"

    # Persist for future sessions, and set it for this one. Scoped to the user,
    # so it needs no elevation and touches nothing machine-wide.
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    if ($userPath -notlike "*$BinDir*") {
        $joined = if ([string]::IsNullOrEmpty($userPath)) { $BinDir } else { "$userPath;$BinDir" }
        [Environment]::SetEnvironmentVariable('Path', $joined, 'User')
        Write-Host "rayzor: added $BinDir to your PATH (new terminals pick it up)"
    }
    $env:Path = "$env:Path;$BinDir"
}
finally {
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}
