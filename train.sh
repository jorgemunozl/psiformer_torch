#!/usr/bin/env bash
set -euo pipefail

# Torch CUDA wheel index; override for different CUDA versions (e.g. cu124) or CPU-only wheels.
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
EXTRA_INDEX_URL="${EXTRA_INDEX_URL:-https://pypi.org/simple}"
WANDB_MODE="${WANDB_MODE:-online}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-.venv}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found; installing to ~/.local/bin" >&2
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if [[ "${WANDB_MODE}" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_MODE=online requires WANDB_API_KEY to be set." >&2
  exit 1
fi

echo "Syncing dependencies with CUDA wheels from ${TORCH_INDEX_URL}"
UV_INDEX_URL="$TORCH_INDEX_URL" UV_EXTRA_INDEX_URL="$EXTRA_INDEX_URL" uv sync --group cuda

echo "Running training"
UV_INDEX_URL="$TORCH_INDEX_URL" UV_EXTRA_INDEX_URL="$EXTRA_INDEX_URL" \
WANDB_MODE="$WANDB_MODE" PYTHONUNBUFFERED=1 \
uv run python src/psiformer_torch/train.py
