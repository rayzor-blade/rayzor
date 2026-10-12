#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
archive_dir="${XDG_CACHE_HOME:-$HOME/.cache}/rayzor-benchmark-libs"
mkdir -p "$archive_dir"

verify_archive() {
  printf '%s  %s\n' "$1" "$2" | sha256sum -c - >/dev/null
}

# Local archives avoid Haxelib's package lookup API.
while read -r checksum archive; do
  archive_path="$archive_dir/$archive"
  if ! verify_archive "$checksum" "$archive_path" 2>/dev/null; then
    curl --fail --location --silent --show-error \
      --retry 5 --retry-all-errors --connect-timeout 10 --max-time 120 \
      "https://haxelib-files.haxe.org/files/3.0/$archive" \
      --output "$archive_path.partial"
    verify_archive "$checksum" "$archive_path.partial"
    mv "$archive_path.partial" "$archive_path"
  fi
  haxelib install "$archive_path" --always
done < "$script_dir/benchmark-libs.sha256"
