#!/usr/bin/env bash
# Download the HyPDE weights from the GitHub Release and verify the sha256.
# The repository is public; GITHUB_TOKEN is optional and only helps with API rate limits.
set -euo pipefail
cd "$(dirname "$0")"
REPO=Tencent-Hunyuan/Hyra-results
TAG=hypde-weights
SHA=67f67339b501dd5bcaddccb233437d4f4dcc24189ecb79d4ff8f5a1fa3600e54
if [ -n "${GITHUB_TOKEN:-}" ]; then
  CURL=(curl -fL -H "Authorization: Bearer $GITHUB_TOKEN")
else
  CURL=(curl -fL)
fi
"${CURL[@]}" -o model.safetensors "https://github.com/$REPO/releases/download/$TAG/model.safetensors"
echo "$SHA  model.safetensors" | sha256sum -c -
echo "Weights ready: $(pwd)/model.safetensors"
