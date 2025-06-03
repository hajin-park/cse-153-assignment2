#!/usr/bin/env bash
# download_aistpp.sh – fetch AIST++ (annotations + audio) without pulling its
# legacy requirements.txt (which pins incompatible absl‑py).
# Usage:  bash scripts/download_aistpp.sh [OUTPUT_DIR]
set -euo pipefail

ROOT="${1:-aist_raw}"
AIST_API_DIR="$ROOT/aistplusplus_api"
ANN_DIR="$ROOT/annotations"
AUDIO_DIR="$ROOT/audio"

mkdir -p "$ROOT"

# 1) Clone API repo (shallow)
if [[ ! -d "$AIST_API_DIR" ]]; then
  echo "[AIST++] cloning API repo …"
  git clone --depth 1 https://github.com/google/aistplusplus_api.git "$AIST_API_DIR"
fi

# 2) We intentionally **skip** the repo's requirements.txt because it pegs
#    very old packages (e.g., absl‑py 0.x) that fail on modern Python.
#    All needed runtime deps are already installed via Makefile -> deps.

echo "[AIST++] requirements installation skipped (handled via Conda env)"

# 3) Download annotations (≈5 GB)
if [[ ! -d "$ANN_DIR" ]]; then
  mkdir -p "$ANN_DIR"
  echo "[AIST++] downloading 3‑D key‑points …"
  python - <<'PY'
import aistplusplus_api.cli.download as dl, os, sys
out = os.environ.get("ANN_DIR", "annotations")
sys.argv = ["", "--type", "annotations", "--out_dir", out]
dl.main()
PY
else
  echo "[AIST++] annotations already present – skipping"
fi

# 4) Download audio (~9 GB) from HF mirror
mkdir -p "$AUDIO_DIR"
ZIP="$AUDIO_DIR/aist_audio.zip"
if [[ ! -d "$AUDIO_DIR/wav" ]]; then
  echo "[AIST++] downloading audio …"
  wget -q --show-progress -O "$ZIP" \
    https://huggingface.co/datasets/AK391/aistplusplus_audio/resolve/main/aist_audio.zip
  unzip -q "$ZIP" -d "$AUDIO_DIR"
  rm "$ZIP"
else
  echo "[AIST++] audio already present – skipping"
fi

echo "✓ AIST++ assets ready in $ROOT"
