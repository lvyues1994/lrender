#!/usr/bin/env bash
# Downloads the glTF scenes lrender_bench measures into a directory (build/scenes unless
# given): Sponza and ABeautifulGame from the Khronos glTF sample assets, at a fixed commit,
# so git checks every file's hash on the way. Only the two scenes' directories are fetched
# (about 150 MB with git's copy). Running it again verifies what is there and fetches only what is missing.
set -euo pipefail

readonly repository=https://github.com/KhronosGroup/glTF-Sample-Assets.git
readonly commit=8f492390ddc7a2ae221a1369ef560589cf8c70fb
readonly scenes=(Models/Sponza/glTF Models/ABeautifulGame/glTF)
readonly target="${1:-$(dirname "$0")/../build/scenes}/khronos"

if [[ ! -d "$target/.git" ]]; then
    mkdir -p "$target"
    git -C "$target" init --quiet
    git -C "$target" remote add origin "$repository"
fi
git -C "$target" sparse-checkout set "${scenes[@]}"
if ! git -C "$target" cat-file -e "$commit^{commit}" 2>/dev/null; then
    git -C "$target" fetch --quiet --depth 1 --filter=blob:none origin "$commit"
fi
git -C "$target" -c advice.detachedHead=false checkout --quiet "$commit"

if [[ "$(git -C "$target" rev-parse HEAD)" != "$commit" ]]; then
    echo "fetch-scenes: $target is not at $commit" >&2
    exit 1
fi
if [[ -n "$(git -C "$target" status --porcelain)" ]]; then
    echo "fetch-scenes: files in $target differ from $commit" >&2
    git -C "$target" status --short >&2
    exit 1
fi
for scene in "${scenes[@]}"; do
    echo "$target/$scene"
done
