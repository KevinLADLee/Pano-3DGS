#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAM3_MODEL_DIR="${SAM3_MODEL_DIR:-$ROOT_DIR/models/facebook/sam3}"

mkdir -p "$SAM3_MODEL_DIR"
modelscope download --model facebook/sam3 --local_dir "$SAM3_MODEL_DIR" "$@"
