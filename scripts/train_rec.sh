#!/bin/bash
# Train tools/train_rec.py for VL-Reader, with multi-GPU torchrun launch.
#
# Usage:
#   bash scripts/train_rec.sh --phase=1                              # MVLR pre-train
#   bash scripts/train_rec.sh --phase=2                              # PLM fine-tune
#   bash scripts/train_rec.sh --phase=2 --tag=mytag
#   bash scripts/train_rec.sh --phase=1 --resume                     # continue from <output_dir>/latest.pth
#   bash scripts/train_rec.sh --phase=2 --debug                      # 1 epoch + _debug tag
#
# Multi-GPU:
#   --gpus=0,1,2,3                                                   # sets CUDA_VISIBLE_DEVICES
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_rec.sh --phase=1
#   NPROC=4 bash scripts/train_rec.sh --phase=1                      # manual torchrun world size
#
# Any unrecognized args are forwarded to tools/train_rec.py:
#   bash scripts/train_rec.sh --phase=1 -o Global.epoch_num=5

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="vlreader"
PHASE=""
USER_TAG=""
RESUME=0
DEBUG_MODE=0
GPUS_OVERRIDE=""
WANDB_RUN_ID=""
WANDB_REWIND=1
PASS_ARGS=()

USER_OPTS=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --phase=*) PHASE="${1#--phase=}" ;;
        --phase)
            shift; [ "$#" -eq 0 ] && { echo "ERROR: --phase requires a value (1 or 2)" >&2; exit 1; }
            PHASE="$1" ;;
        --tag=*) USER_TAG="${1#--tag=}" ;;
        --tag)
            shift; [ "$#" -eq 0 ] && { echo "ERROR: --tag requires a value" >&2; exit 1; }
            USER_TAG="$1" ;;
        --gpus=*) GPUS_OVERRIDE="${1#--gpus=}" ;;
        --gpus)
            shift; [ "$#" -eq 0 ] && { echo "ERROR: --gpus requires a value" >&2; exit 1; }
            GPUS_OVERRIDE="$1" ;;
        --nproc=*) NPROC="${1#--nproc=}" ;;
        --nproc)
            shift; [ "$#" -eq 0 ] && { echo "ERROR: --nproc requires a value" >&2; exit 1; }
            NPROC="$1" ;;
        --wandb_run_id=*) WANDB_RUN_ID="${1#--wandb_run_id=}" ;;
        --wandb_run_id)
            shift; [ "$#" -eq 0 ] && { echo "ERROR: --wandb_run_id requires a value" >&2; exit 1; }
            WANDB_RUN_ID="$1" ;;
        --no_wandb_rewind) WANDB_REWIND=0 ;;
        --resume|--restore) RESUME=1 ;;
        --debug|--debug=*) DEBUG_MODE=1 ;;
        # Capture user-supplied -o key=value pairs into USER_OPTS so they can
        # be merged with the script's _EXTRA_OPTS into a single -o list.
        # Without this, argparse's nargs='*' makes the second -o overwrite
        # the first, silently dropping user overrides.
        -o)
            shift
            while [ "$#" -gt 0 ] && [[ "$1" != -* ]]; do
                USER_OPTS+=("$1"); shift
            done
            continue ;;
        -o=*) USER_OPTS+=("${1#-o=}") ;;
        *) PASS_ARGS+=("$1") ;;
    esac
    shift
done

if [ -z "${PHASE}" ]; then
    echo "ERROR: --phase is required (1 = MVLR pre-train, 2 = PLM fine-tune)" >&2
    exit 1
fi
if [ "${PHASE}" != "1" ] && [ "${PHASE}" != "2" ]; then
    echo "ERROR: --phase must be 1 or 2 (got: ${PHASE})" >&2
    exit 1
fi
if [ -n "${GPUS_OVERRIDE}" ]; then
    export CUDA_VISIBLE_DEVICES="${GPUS_OVERRIDE}"
fi

source "$SCRIPT_DIR/_model_config.sh"
source "$SCRIPT_DIR/_launch.sh"

if [ "${PHASE}" = "1" ]; then
    TRAIN_CONFIG="${CONFIG_PHASE1}"
else
    TRAIN_CONFIG="${CONFIG_PHASE2}"
fi
if [ ! -f "${TRAIN_CONFIG}" ]; then
    echo "ERROR: Train config not found: ${TRAIN_CONFIG}" >&2
    exit 1
fi

# Compose Global.output_dir from MODEL_KEY + phase so the two phases stay in
# separate directories and wandb run names are 'phase1' / 'phase2' (with
# optional _<tag> appended by Global.tag below).
#   ./output/rec/<MODEL_KEY>/phase<N>
# User can override with -o Global.output_dir=... in PASS_ARGS.
_EXTRA_OPTS=()
if [[ -n "${MODEL_KEY}" ]]; then
    _EXTRA_OPTS+=("Global.output_dir=./output/rec/${MODEL_KEY}/phase${PHASE}")
fi

# In phase 2, auto-inject pretrained ckpt if yml doesn't override it.
# Placed BEFORE PASS_ARGS so user-supplied -o Global.pretrained_model=... wins.
if [[ "${PHASE}" = "2" && -n "${CONFIG_PRETRAINED:-}" ]]; then
    _yml_pretrained=$(grep '^  pretrained_model:' "${TRAIN_CONFIG}" | head -1 | sed 's/^[^:]*:\s*//' | sed 's/\s*$//')
    if [ -z "${_yml_pretrained}" ]; then
        _EXTRA_OPTS+=("Global.pretrained_model=${CONFIG_PRETRAINED}")
    fi
fi

if [ "${DEBUG_MODE}" = "1" ]; then
    _EXTRA_OPTS+=("Global.epoch_num=1")
    # OneCycleLR pct_start = warmup_epoch / epoch_num must be <1; with
    # epoch_num=1 the original 1.5 warmup goes out of range, so disable it.
    _EXTRA_OPTS+=("LRScheduler.warmup_epoch=0")
    USER_TAG="${USER_TAG:+${USER_TAG}_}debug"
    echo "=== DEBUG mode: epoch_num=1, warmup_epoch=0, tag=${USER_TAG} ==="
fi
if [[ -n "${USER_TAG}" ]]; then
    _EXTRA_OPTS+=("Global.tag=_${USER_TAG}")
fi
if [[ -n "${WANDB_RUN_ID}" ]]; then
    _EXTRA_OPTS+=("Global.wandb_run_id=${WANDB_RUN_ID}")
fi
if [[ "${WANDB_REWIND}" == "0" ]]; then
    _EXTRA_OPTS+=("Global.wandb_rewind=false")
fi

CKPT_PATH=""
if [[ "${RESUME}" == "1" ]]; then
    _out_dir=$(grep '^  output_dir:' "${TRAIN_CONFIG}" | head -1 | sed 's/^[^:]*:\s*//' | sed 's/\s*$//' | sed 's:/*$::')
    [[ -n "$_out_dir" ]] && CKPT_PATH="${_out_dir}/latest.pth"
fi

echo "=== train_rec ==="
echo "  model:                ${MODEL}"
echo "  phase:                ${PHASE}"
echo "  train_cfg:            ${TRAIN_CONFIG}"
if [ ${#_EXTRA_OPTS[@]} -gt 0 ]; then
    echo "  extra opts (-o):      ${_EXTRA_OPTS[*]}"
fi
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "  nproc:                ${NPROC}"
echo "  resume:               ${RESUME}"
[[ -n "${CKPT_PATH}" ]] && echo "  resume ckpt:          ${CKPT_PATH}"
[[ -n "${WANDB_RUN_ID}" ]] && echo "  wandb run id:         ${WANDB_RUN_ID}"
echo "  wandb rewind:         ${WANDB_REWIND}"
echo "  pass-through args:    ${PASS_ARGS[*]:-<none>}"
if [ ${#USER_OPTS[@]} -gt 0 ]; then
    echo "  user -o overrides:    ${USER_OPTS[*]}"
fi
echo ""

export RESUME

# Merge script's _EXTRA_OPTS with the user's USER_OPTS into a single -o list.
# USER_OPTS goes LAST so user-supplied values win on key conflicts (e.g.
# user can override Global.total_bs that the script otherwise leaves alone).
ALL_OPTS=("${_EXTRA_OPTS[@]}" "${USER_OPTS[@]}")

_T0=$SECONDS
if [ ${#ALL_OPTS[@]} -gt 0 ]; then
    train "${TRAIN_CONFIG}" "${PASS_ARGS[@]}" -o "${ALL_OPTS[@]}"
else
    train "${TRAIN_CONFIG}" "${PASS_ARGS[@]}"
fi
_s=$(( SECONDS - _T0 ))
printf -- "--- Done in %dh %dm %ds ---\n" $(( _s / 3600 )) $(( _s % 3600 / 60 )) $(( _s % 60 ))
