#!/usr/bin/env bash
# Runs the benchmarks docs/benchmarks.md reports, each into <output>/<name>.json:
#
#   tools/run-benchmarks.sh [nvidia|lavapipe] [output directory] [build directory]
#
# nvidia measures at 1920x1080 on the first device whose name contains NVIDIA, lavapipe at
# 960x540 on llvmpipe with shorter warm-ups. Needs a release build (cmake --preset release)
# and the scenes from tools/fetch-scenes.sh.
set -euo pipefail

readonly device="${1:-nvidia}"
readonly output="${2:-build/bench}"
readonly build="${3:-build/release}"
readonly bench="$build/probes/lrender_bench"

mkdir -p "$output"
measure() {
    local -r name="$1"
    shift
    echo "== $name"
    "$bench" "${common[@]}" "$@" --json "$output/$name.json" | grep -vE "^    " || return 1
}

case "$device" in
nvidia)
    common=(--backend vulkan --device NVIDIA --size 1920x1080)
    measure sponza --scene sponza
    measure game --scene game
    measure scale-100k --scene scale --instances 100000
    measure scale-1m --scene scale --instances 1000000
    measure sponza-taa --scene sponza --config none --config taa:taa=on
    measure sponza-aliasing --scene sponza --config shared:aliasing=on --config separate:aliasing=off
    measure sponza-occlusion --scene sponza --config occlusion:occlusion=on --config frustum:occlusion=off
    measure scale-1m-occlusion --scene scale --instances 1000000 --config occlusion:occlusion=on \
        --config frustum:occlusion=off
    measure sponza-pathtrace --scene sponza --config rayqueries:pipeline=pathtrace \
        --config hierarchies:pipeline=pathtrace,rayqueries=off
    ;;
lavapipe)
    common=(--backend vulkan --device llvmpipe --size 960x540 --warmup 10)
    measure lavapipe-sponza --scene sponza
    measure lavapipe-game --scene game
    measure lavapipe-scale-10k --scene scale --instances 10000 --lights 64
    ;;
*)
    echo "usage: tools/run-benchmarks.sh [nvidia|lavapipe] [output directory] [build directory]" >&2
    exit 2
    ;;
esac
