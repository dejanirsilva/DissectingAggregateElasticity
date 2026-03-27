#!/usr/bin/env bash
set -euo pipefail

remote_name="${1:-overleaf}"
target_branch="${2:-master}"
prefix="${3:-paper}"
tmp_branch="overleaf-sync-$(date +%Y%m%d%H%M%S)"
repo_root="$(git rev-parse --show-toplevel)"
tmp_dir="$(mktemp -d)"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Error: run this inside a git repository." >&2
  exit 1
fi

if ! git remote get-url "$remote_name" >/dev/null 2>&1; then
  echo "Error: git remote '$remote_name' does not exist." >&2
  echo "Add it with: git remote add $remote_name https://git.overleaf.com/<project-id>" >&2
  exit 1
fi

if ! git ls-tree -d --name-only HEAD "$prefix" | grep -qx "$prefix"; then
  echo "Error: prefix directory '$prefix' is missing at HEAD." >&2
  exit 1
fi

cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

overleaf_url="$(git remote get-url "$remote_name")"

echo "Preparing sync workspace..."
git clone --quiet "$repo_root" "$tmp_dir"
cd "$tmp_dir"
git remote add "$remote_name" "$overleaf_url"

if git ls-remote --exit-code "$remote_name" "$target_branch" >/dev/null 2>&1; then
  git fetch --quiet "$remote_name" "$target_branch"
  git checkout -B "$tmp_branch" FETCH_HEAD >/dev/null
else
  git checkout --orphan "$tmp_branch" >/dev/null
  git rm -rf . >/dev/null 2>&1 || true
fi

# Make Overleaf root exactly match local prefix directory.
find . -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
git -C "$repo_root" archive --format=tar HEAD "$prefix" | tar -xf - --strip-components=1

# Also include shared utility files when present.
for extra_dir in utils/references utils/styles; do
  if git -C "$repo_root" cat-file -e "HEAD:$extra_dir" >/dev/null 2>&1; then
    git -C "$repo_root" archive --format=tar HEAD "$extra_dir" | tar -xf -
  fi
done

git add -A

if git diff --cached --quiet; then
  echo "No changes to push."
  exit 0
fi

git commit -m "Sync $prefix from $(git -C "$repo_root" rev-parse --short HEAD)" >/dev/null

echo "Pushing '$prefix' to '$remote_name/$target_branch'..."
git push "$remote_name" "$tmp_branch:$target_branch"

echo "Done: Overleaf now has only '$prefix' contents."
