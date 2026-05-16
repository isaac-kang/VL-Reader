# Shared launcher for train_rec.py.
# Source this to get the `train` function.
#
# Env vars:
#   CUDA_VISIBLE_DEVICES  : GPU(s) to use (default: 0)
#   CUDA_DEVICE_ORDER     : (default: PCI_BUS_ID)
#   NPROC                 : override number of processes (default: auto from CUDA_VISIBLE_DEVICES)

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Count number of GPUs from CUDA_VISIBLE_DEVICES
_count_gpus() {
    local _devs="${CUDA_VISIBLE_DEVICES}"
    if [[ -z "$_devs" || "$_devs" == "NoDevFiles" ]]; then
        echo 0
    else
        echo $(( $(echo "$_devs" | tr -cd ',' | wc -c) + 1 ))
    fi
}

NPROC="${NPROC:-$(_count_gpus)}"

# Find a free TCP port for torchrun rendezvous.
# Tries MASTER_PORT_START..MASTER_PORT_END (default 29500..29599).
# Tests BOTH ipv6 "::" AND ipv4 "0.0.0.0" (torchrun binds to [::] first, then 0.0.0.0);
# a port must be free on both to avoid "Address already in use".
_find_free_port() {
    local _start="${MASTER_PORT_START:-29500}"
    local _end="${MASTER_PORT_END:-29599}"
    python3 - "$_start" "$_end" <<'PY'
import socket, sys
start, end = int(sys.argv[1]), int(sys.argv[2])

def is_free(port):
    for family, host in ((socket.AF_INET6, "::"), (socket.AF_INET, "0.0.0.0")):
        try:
            s = socket.socket(family, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
            s.bind((host, port))
            s.close()
        except OSError:
            return False
    return True

for p in range(start, end + 1):
    if is_free(p):
        print(p); sys.exit(0)

# fallback: OS-assigned
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("0.0.0.0", 0))
    print(s.getsockname()[1])
PY
}

train() {
    # train <config> [extra args...]
    # Set RESUME=1 to auto-load latest.pth from output_dir as checkpoint.
    local _config="$1"; shift
    local _resume_args=()
    if [[ "${RESUME:-0}" == "1" ]]; then
        _resume_args=(--restore)
    fi
    if [[ "$NPROC" -gt 1 ]]; then
        local _port
        if [[ -n "${MASTER_PORT:-}" ]]; then
            _port="${MASTER_PORT}"
        else
            _port="$(_find_free_port)"
        fi
        # torchrun은 flag/env 둘 다 받는다. 둘 다 세팅해서 상위 쉘에 남은 stale
        # MASTER_PORT(예: 29500) 때문에 덮어쓰이지 않도록 강제.
        export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
        export MASTER_PORT="${_port}"
        echo "[launch] torchrun --nproc_per_node=${NPROC} --master_port=${_port}  (MASTER_ADDR=${MASTER_ADDR})"
        torchrun --nproc_per_node="${NPROC}" --master_port="${_port}" \
            tools/train_rec.py -c "${_config}" "${_resume_args[@]}" "$@"
    else
        python tools/train_rec.py -c "${_config}" "${_resume_args[@]}" "$@"
    fi
}
