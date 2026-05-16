# Model config lookup for run_*.sh scripts. Source this AFTER setting MODEL.
#
# Sets:
#   CONFIG_PHASE1            - phase 1 (MVLR pre-training) config
#   CONFIG_PHASE2            - phase 2 (PARSeq-style PLM fine-tuning) config
#   CONFIG_PRETRAINED        - default pretrained checkpoint path used by phase 2
#   MODEL_KEY                - short identifier used to compose Global.output_dir
#                              as ./output/rec/<MODEL_KEY>/

case "${MODEL:-vlreader}" in
    vlreader)
        CONFIG_PHASE1="configs/rec/vlreader/vit_vlreader_phase1.yml"
        CONFIG_PHASE2="configs/rec/vlreader/vit_vlreader_phase2.yml"
        # weights/pretrain/  : 내가 phase 1로 직접 학습한 ckpt (self-trained).
        # weights/pretrained/: GitHub 등에서 받은 외부 pretrained ckpt.
        CONFIG_PRETRAINED="weights/pretrain/vlreader/best.pth"
        ;;
    *)
        echo "Unknown MODEL: ${MODEL}. Use: vlreader" >&2
        exit 1
        ;;
esac

# Read model_key out of the phase-1 yml (single source of truth).
MODEL_KEY=""
if [ -f "${CONFIG_PHASE1}" ]; then
    MODEL_KEY="$(grep -E '^model_key:' "${CONFIG_PHASE1}" | head -1 | sed -E 's/^model_key:[[:space:]]*//' | sed -E 's/[[:space:]]+$//')"
fi
