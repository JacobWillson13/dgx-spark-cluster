#!/usr/bin/env bash
# Fail if any tracked or staged file carries an identifier that must not be
# published: IPv4 addresses, MAC addresses, home-directory paths, the old
# account name, or organization names. Run before every push.
#   bash scripts/scrub_check.sh
set -uo pipefail
cd "$(dirname "$0")/.."
if git rev-parse --git-dir >/dev/null 2>&1; then
  files=$(git ls-files --cached --others --exclude-standard)
else
  files=$(find . -type f -not -path './.git/*' | sed 's#^\./##')
fi
files=$(printf '%s\n' "$files" | grep -vE '\.(png|jpg|jpeg|gif|ico)$|^scripts/scrub_check\.sh$')
fail=0
check() {  # name, extended regex
  hits=$(printf '%s\n' "$files" | tr '\n' '\0' | xargs -0 grep -nIiE "$2" -- 2>/dev/null)
  if [ -n "$hits" ]; then
    echo "FAIL $1"; printf '%s\n' "$hits" | head -20; fail=1
  else
    echo "ok   $1"
  fi
}
check "IPv4 address"        '(^|[^0-9.])([0-9]{1,3}\.){3}[0-9]{1,3}([^0-9.]|$)'
check "MAC address"         '(^|[^0-9a-f:])([0-9a-f]{2}[:-]){5}[0-9a-f]{2}([^0-9a-f:]|$)'
check "home directory path" '/home/[a-z_][a-z0-9_-]*|/Users/[A-Za-z]'
check "account or org name" '(^|[^a-z])afrl([^a-z]|$)|air force|griffiss|sunypoly'
check "secret-like token"   'hf_[A-Za-z0-9]{30,}|ghp_[A-Za-z0-9]{30,}|github_pat_|BEGIN [A-Z ]*PRIVATE KEY'
for f in $(printf '%s\n' "$files" | grep -E '\.png$'); do
  if command -v exiftool >/dev/null && exiftool -s -s -s -Comment -Author -Creator "$f" | grep -q .; then
    echo "WARN image metadata in $f"
  fi
done
[ "$fail" = 0 ] && echo "scrub check passed" || { echo "scrub check FAILED"; exit 1; }
