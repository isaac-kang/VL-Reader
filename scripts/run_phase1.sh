#!/bin/bash
# VL-Reader phase 1: MVLR pre-training (visual + linguistic reconstruction).
# Forwards all args to scripts/train_rec.sh --phase=1.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/train_rec.sh" --phase=1 "$@"
