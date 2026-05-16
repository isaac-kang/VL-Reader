#!/bin/bash
# VL-Reader phase 2: PARSeq-style PLM fine-tuning from a phase-1 checkpoint.
# Forwards all args to scripts/train_rec.sh --phase=2.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/train_rec.sh" --phase=2 "$@"
