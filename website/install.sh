#!/usr/bin/env sh
# Rayzor installer.
#
#   curl -fsSL https://rayzor.tech/install.sh | sh
#
# Downloads the nightly build for this platform, verifies its checksum, and
# installs it. Nothing is written outside the install directory, and the
# download is self-contained: no LLVM installation is required.
set -eu

REPO="${RAYZOR_REPO:-rayzor-blade/rayzor}"
CHANNEL="${RAYZOR_CHANNEL:-nightly}"
PREFIX="${RAYZOR_PREFIX:-$HOME/.rayzor}"
BIN_DIR="$PREFIX/bin"

say()  { printf '%s\n' "$*"; }
warn() { printf '%s\n' "$*" >&2; }
die()  { printf 'install: %s\n' "$*" >&2; exit 1; }

# Piping into a shell means stdin is the script, so a prompt would read the
# script's own bytes as the answer. Everything here is non-interactive.
need() { command -v "$1" >/dev/null 2>&1 || die "$1 is required but not installed"; }

case "$(uname -s)" in
  Darwin) os=macos ;;
  Linux)  os=linux ;;
  MINGW*|MSYS*|CYGWIN*)
    die "Windows is not installable this way. Download the zip from
    https://github.com/$REPO/releases/$CHANNEL and put rayzor.exe on your PATH." ;;
  *) die "unsupported operating system: $(uname -s)" ;;
esac

case "$(uname -m)" in
  arm64|aarch64) arch=aarch64 ;;
  x86_64|amd64)  arch=x86_64 ;;
  *) die "unsupported architecture: $(uname -m)" ;;
esac

# Apple silicon runs x86_64 builds under Rosetta, but a native build exists for
# every platform published here, so a mismatch means the asset is missing
# rather than that translation should be used.
asset="rayzor-${os}-${arch}.tar.gz"
base="https://github.com/$REPO/releases/download/$CHANNEL"

need uname
need tar
if command -v curl >/dev/null 2>&1; then
  fetch() { curl -fsSL "$1" -o "$2"; }
elif command -v wget >/dev/null 2>&1; then
  fetch() { wget -qO "$2" "$1"; }
else
  die "curl or wget is required"
fi

tmp="$(mktemp -d)"
# shellcheck disable=SC2064
trap "rm -rf '$tmp'" EXIT INT TERM

say "rayzor: fetching $asset ($CHANNEL)"
fetch "$base/$asset" "$tmp/$asset" \
  || die "no build for ${os}-${arch} in the $CHANNEL release.
    See https://github.com/$REPO/releases/$CHANNEL"

# Verify when a checksum is published; a corrupted download otherwise fails
# later as something that looks like a compiler bug.
if fetch "$base/$asset.sha256" "$tmp/$asset.sha256" 2>/dev/null; then
  expected="$(awk '{print $1}' "$tmp/$asset.sha256")"
  if command -v shasum >/dev/null 2>&1; then
    actual="$(shasum -a 256 "$tmp/$asset" | awk '{print $1}')"
  elif command -v sha256sum >/dev/null 2>&1; then
    actual="$(sha256sum "$tmp/$asset" | awk '{print $1}')"
  else
    actual=""
    warn "rayzor: no sha256 tool found; skipping checksum verification"
  fi
  if [ -n "$actual" ] && [ "$actual" != "$expected" ]; then
    die "checksum mismatch for $asset
    expected $expected
    actual   $actual"
  fi
else
  warn "rayzor: no published checksum for $asset; skipping verification"
fi

mkdir -p "$BIN_DIR"
tar -xzf "$tmp/$asset" -C "$tmp"
# The archive holds rayzor and, where it is built, the rayzor-wasm-opt helper.
# The CLI finds the helper as a sibling of its own executable, so they must
# land in the same directory.
found=0
for f in rayzor rayzor-wasm-opt; do
  if [ -f "$tmp/$f" ]; then
    install -m 0755 "$tmp/$f" "$BIN_DIR/$f"
    found=1
  fi
done
[ "$found" -eq 1 ] || die "archive did not contain a rayzor binary"

version="$("$BIN_DIR/rayzor" --version 2>/dev/null || echo unknown)"
say "rayzor: installed $version to $BIN_DIR"

case ":${PATH}:" in
  *":$BIN_DIR:"*) ;;
  *)
    say ""
    say "Add it to your PATH:"
    say "    export PATH=\"$BIN_DIR:\$PATH\""
    say ""
    say "To keep it, append that line to your shell profile"
    say "(~/.zshrc, ~/.bashrc, or ~/.profile)."
    ;;
esac
