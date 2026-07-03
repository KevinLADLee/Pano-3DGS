#!/usr/bin/env bash
set -euo pipefail

modelscope download --model facebook/sam3 "$@"
