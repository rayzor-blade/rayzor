#!/usr/bin/env bash
set -euo pipefail

llvm_codename="$(. /etc/os-release; printf '%s' "${VERSION_CODENAME:?}")"
llvm_keyring=/usr/share/keyrings/apt.llvm.org.asc

sudo install -d -m 0755 /usr/share/keyrings
curl --fail --silent --show-error --location --retry 3 \
  https://apt.llvm.org/llvm-snapshot.gpg.key \
  | sudo tee "$llvm_keyring" >/dev/null
printf 'deb [signed-by=%s] https://apt.llvm.org/%s/ llvm-toolchain-%s-21 main\n' \
  "$llvm_keyring" "$llvm_codename" "$llvm_codename" \
  | sudo tee /etc/apt/sources.list.d/llvm-21.list >/dev/null
sudo apt-get update
sudo apt-get install -y --no-install-recommends llvm-21 llvm-21-dev libpolly-21-dev
/usr/lib/llvm-21/bin/llvm-config --version

if [[ -n "${GITHUB_ENV:-}" ]]; then
  printf 'LLVM_SYS_211_PREFIX=/usr/lib/llvm-21\n' >> "$GITHUB_ENV"
fi
