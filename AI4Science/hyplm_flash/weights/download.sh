#!/usr/bin/env bash
# Download the HyPLM-Flash weights (two shards) from the GitHub Release, merge them and verify the sha256.
# The repository is public; GITHUB_TOKEN is optional and only helps with API rate limits.
set -euo pipefail
cd "$(dirname "$0")"
REPO=Tencent-Hunyuan/Hyra-results
TAG=hyplm-flash-weights
SHA=ce7dcd4fc473dc334d869718a9a31149573893795db03c45efeb396e8b180ab1
if [ -n "${GITHUB_TOKEN:-}" ]; then
  CURL=(curl -fL -H "Authorization: Bearer $GITHUB_TOKEN")
else
  CURL=(curl -fL)
fi
for p in part0 part1; do
  "${CURL[@]}" -o "model.safetensors.$p" "https://github.com/$REPO/releases/download/$TAG/model.safetensors.$p"
done
cat model.safetensors.part0 model.safetensors.part1 > model.safetensors
echo "$SHA  model.safetensors" | sha256sum -c -
rm -f model.safetensors.part0 model.safetensors.part1
echo "Weights ready: $(pwd)/model.safetensors"
