#!/usr/bin/env bash
set -euo pipefail

remote_name="${1:-overleaf}"
target_branch="${2:-master}"
prefix="${3:-paper}"
allow_dirty="${ALLOW_DIRTY:-0}"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Error: run this inside a git repository." >&2
  exit 1
fi

repo_root="$(git rev-parse --show-toplevel)"

if ! git remote get-url "$remote_name" >/dev/null 2>&1; then
  echo "Error: git remote '$remote_name' does not exist." >&2
  echo "Add it with: git remote add $remote_name https://git.overleaf.com/<project-id>" >&2
  exit 1
fi

if [[ "$allow_dirty" != "1" ]] && [[ -n "$(git status --porcelain)" ]]; then
  echo "Error: working tree is not clean. Commit/stash first, or run with ALLOW_DIRTY=1." >&2
  exit 1
fi

tmp_dir="$(mktemp -d)"
cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

overleaf_url="$(git remote get-url "$remote_name")"

echo "Cloning Overleaf branch '$target_branch'..."
if ! git clone --quiet --branch "$target_branch" --single-branch "$overleaf_url" "$tmp_dir"; then
  echo "Error: failed to clone '$remote_name/$target_branch'." >&2
  exit 1
fi

mkdir -p "$repo_root/$prefix"

echo "Syncing Overleaf root -> '$prefix/'..."
rsync -a --delete \
  --exclude '.git/' \
  --exclude 'utils/references/' \
  --exclude 'utils/styles/' \
  "$tmp_dir"/ "$repo_root/$prefix"/

# Sync shared utility directories when present on Overleaf.
for extra_dir in utils/references utils/styles; do
  if [[ -d "$tmp_dir/$extra_dir" ]]; then
    mkdir -p "$repo_root/$extra_dir"
    echo "Syncing '$extra_dir/'..."
    rsync -a --delete "$tmp_dir/$extra_dir"/ "$repo_root/$extra_dir"/
  fi
done

echo "Done: imported Overleaf changes into '$prefix/' (and shared utils when present)."
