#!/usr/bin/env bash
# Download the HyEEG weights (Small, Base, Large) from the GitHub Release and verify sha256.
# Usage:
#   bash weights/download.sh                  # download all three scales (152 MB total)
#   bash weights/download.sh large            # download HyEEG-Large only (101 MB)
#   bash weights/download.sh small base       # download selected scales
# The repository is public; GITHUB_TOKEN is optional and only helps with API rate limits.
set -euo pipefail
cd "$(dirname "$0")"

REPO="${HYEEG_REPO:-Tencent-Hunyuan/Hyra-results}"
TAG="${HYEEG_TAG:-hyeeg-weights}"

declare -A SHAS=(
  [small]=2fd8df4a47414a483c4f8f9a642821db3c6b32e35e59e2f98358762888f040b2
  [base]=bae84b8a969f53cab49cdc61b865626ef0d53a1f1aaefb7bbc34adc42b2a6a55
  [large]=d1dc6b05f969d07cdc47f69ff0ce37b2118f174b3a0d6c610059dcbd2071cd43
)

if [ "$#" -eq 0 ]; then
  SCALES=(small base large)
else
  SCALES=("$@")
fi

if [ -n "${GITHUB_TOKEN:-}" ]; then
  CURL=(curl -fL -H "Authorization: Bearer $GITHUB_TOKEN")
else
  CURL=(curl -fL)
fi

for scale in "${SCALES[@]}"; do
  if [ -z "${SHAS[$scale]+x}" ]; then
    echo "Unknown HyEEG scale: '$scale' (expected one of: small, base, large)" >&2
    exit 1
  fi
  mkdir -p "$scale"
  asset="hyeeg-${scale}.safetensors"
  out="${scale}/model.safetensors"
  "${CURL[@]}" -o "$out" "https://github.com/$REPO/releases/download/$TAG/$asset"
  echo "${SHAS[$scale]}  $out" | sha256sum -c -
  echo "Weights ready: $(pwd)/$out"
done
