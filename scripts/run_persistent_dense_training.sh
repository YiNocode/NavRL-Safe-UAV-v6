#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/ubuntu/Desktop/NavRL-Safe-UAV-v5/NavRL-Safe-UAV-v5
PYTHON_BIN=/home/ubuntu/miniforge3/envs/isaaclab-v5/bin/python
STRACE_BIN=/usr/bin/strace
NVIDIA_SMI_BIN=/usr/bin/nvidia-smi
STDBUF_BIN=/usr/bin/stdbuf
LOG_ROOT=${PROJECT_ROOT}/logs/rsl_rl/uav_v5_navrl_gpu
DIAGNOSTIC_ROOT=${PROJECT_ROOT}/logs/diagnostics
RUN_PREFIX=v5_lowalt_tall_s40_d15_s46_e256
DEFAULT_CHECKPOINT=${PROJECT_ROOT}/logs/rsl_rl/uav_v5_navrl_gpu/2026-09-01_09-06-11_v5_s40_d15_persistent_s45_e256_20260901_090607_seg1/model_best.pt
DEFAULT_BEST_CHECKPOINT=${DEFAULT_CHECKPOINT}
# Ground take-off, the low-altitude corridor, and 2.60--5.00 m static obstacles
# define a new task distribution.  The old metric is not comparable, so let
# the first full rolling window establish the new model_best.pt.
DEFAULT_BEST_METRIC=0.0

checkpoint=${DEFAULT_CHECKPOINT}
best_checkpoint=${DEFAULT_BEST_CHECKPOINT}
best_metric=${DEFAULT_BEST_METRIC}
stop_requested=0
train_pid=""
tracer_pid=""
dmon_pid=""
dmon_log=""

request_stop() {
    local forwarded_signal=$1
    stop_requested=1
    echo "[WRAPPER-SIGNAL] received=${forwarded_signal} wrapper_pid=$$ parent_pid=${PPID}"
    if [[ -n "${train_pid}" ]] && kill -0 "${train_pid}" 2>/dev/null; then
        kill -s "${forwarded_signal}" "${train_pid}" 2>/dev/null || true
    fi
}

cleanup_diagnostics() {
    if [[ -n "${dmon_pid}" ]] && kill -0 "${dmon_pid}" 2>/dev/null; then
        kill -TERM "${dmon_pid}" 2>/dev/null || true
        wait "${dmon_pid}" 2>/dev/null || true
    fi
}

start_gpu_monitor() {
    mkdir -p "${DIAGNOSTIC_ROOT}"
    dmon_log=${DIAGNOSTIC_ROOT}/nvidia_smi_dmon_$(date +%Y%m%d_%H%M%S).log
    "${STDBUF_BIN}" -oL "${NVIDIA_SMI_BIN}" dmon -i 0 -d 1 -s pucvmet -o DT \
        > "${dmon_log}" 2>&1 &
    dmon_pid=$!
    sleep 1
    if ! kill -0 "${dmon_pid}" 2>/dev/null; then
        wait "${dmon_pid}" || true
        echo "[GPU-DMON] Failed to start nvidia-smi dmon." >&2
        exit 3
    fi
    echo "[GPU-DMON] pid=${dmon_pid} log=${dmon_log}"
}

summarize_signal_trace() {
    local trace_file=$1
    [[ -s "${trace_file}" ]] || return
    local trace_line sender_pid sender_info sender_cgroup
    while IFS= read -r trace_line; do
        echo "[SIGNAL-TRACE] ${trace_line}"
        sender_pid=$(sed -n 's/.*si_pid=\([0-9][0-9]*\).*/\1/p' <<<"${trace_line}")
        if [[ -z "${sender_pid}" ]]; then
            continue
        fi
        sender_info=$(ps -p "${sender_pid}" -o pid=,ppid=,user=,lstart=,cmd= 2>/dev/null || true)
        sender_cgroup=""
        if [[ -r "/proc/${sender_pid}/cgroup" ]]; then
            sender_cgroup=$(tr '\n' ';' < "/proc/${sender_pid}/cgroup")
        fi
        echo "[SIGNAL-SOURCE] sender_pid=${sender_pid} process=${sender_info:-exited-before-inspection} cgroup=${sender_cgroup:-unavailable}"
    done < <(grep -E 'SIG(INT|TERM|HUP)' "${trace_file}" || true)
}

trap 'request_stop INT' INT
trap 'request_stop TERM' TERM
trap 'request_stop HUP' HUP
trap cleanup_diagnostics EXIT

read_state_field() {
    local state_file=$1
    local field=$2
    "${PYTHON_BIN}" -c \
        'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get(sys.argv[2], ""))' \
        "${state_file}" "${field}"
}

latest_progress_checkpoint() {
    local run_dir=$1
    if [[ -f "${run_dir}/model_interrupted.pt" ]]; then
        printf '%s\n' "${run_dir}/model_interrupted.pt"
        return
    fi
    find "${run_dir}" -maxdepth 1 -type f -name 'model_[0-9]*.pt' -printf '%f %p\n' \
        | sed -n 's/^model_\([0-9][0-9]*\)\.pt /\1 /p' \
        | sort -nr \
        | head -1 \
        | cut -d' ' -f2-
}

select_furthest_progress_checkpoint() {
    "${PYTHON_BIN}" - "${LOG_ROOT}" "${RUN_PREFIX}" <<'PY'
from pathlib import Path
import re
import sys

import torch

log_root = Path(sys.argv[1])
run_prefix = sys.argv[2]
candidates = []
for run_dir in log_root.glob(f"*_{run_prefix}_*"):
    for path in run_dir.glob("model_*.pt"):
        if path.name != "model_interrupted.pt" and re.fullmatch(r"model_\d+\.pt", path.name) is None:
            continue
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            iteration = int(checkpoint.get("iter", -1))
        except Exception:
            continue
        candidates.append((iteration, path.stat().st_mtime_ns, path))
if candidates:
    print(max(candidates)[2])
PY
}

select_latest_checkpoint() {
    local latest_run
    latest_run=$(find "${LOG_ROOT}" -mindepth 1 -maxdepth 1 -type d -name "*_${RUN_PREFIX}_*" \
        -exec test -f '{}/model_best.pt' \; \
        -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2- || true)
    if [[ -n "${latest_run}" ]]; then
        best_checkpoint=${latest_run}/model_best.pt
    fi
    local progress_checkpoint
    progress_checkpoint=$(select_furthest_progress_checkpoint || true)
    checkpoint=${progress_checkpoint:-${best_checkpoint}}
    if [[ -n "${latest_run}" && -f "${latest_run}/stability_state.json" ]]; then
        local saved_metric
        saved_metric=$(read_state_field "${latest_run}/stability_state.json" best_metric)
        if [[ -n "${saved_metric}" ]]; then
            best_metric=${saved_metric}
        fi
    fi
}

select_latest_checkpoint
[[ -f "${checkpoint}" ]] || { echo "Missing checkpoint: ${checkpoint}" >&2; exit 2; }
[[ -x "${STRACE_BIN}" ]] || { echo "Missing strace: ${STRACE_BIN}" >&2; exit 2; }
[[ -x "${NVIDIA_SMI_BIN}" ]] || { echo "Missing nvidia-smi: ${NVIDIA_SMI_BIN}" >&2; exit 2; }
[[ -x "${STDBUF_BIN}" ]] || { echo "Missing stdbuf: ${STDBUF_BIN}" >&2; exit 2; }
start_gpu_monitor

segment=0
while true; do
    segment=$((segment + 1))
    run_name=${RUN_PREFIX}_$(date +%Y%m%d_%H%M%S)_seg${segment}
    echo "[PERSISTENT] Starting ${run_name} from ${checkpoint} (best=${best_metric}, best_checkpoint=${best_checkpoint})"

    signal_trace=${DIAGNOSTIC_ROOT}/${run_name}_signals.log
    set +e
    "${STRACE_BIN}" -ttt -e trace=none -e signal=INT,TERM,HUP -o "${signal_trace}" \
      "${PYTHON_BIN}" -u "${PROJECT_ROOT}/scripts/train.py" \
        --device cuda:0 \
        --headless \
        --num_envs 256 \
        --seed 46 \
        --max_iterations 10000 \
        --learning_rate 2e-5 \
        --run_name "${run_name}" \
        --resume "${checkpoint}" \
        --reset_optimizer_on_resume \
        --initial_best_checkpoint "${best_checkpoint}" \
        --initial_best_metric "${best_metric}" \
        --logger tensorboard &
    tracer_pid=$!
    train_pid=""
    for _ in $(seq 1 100); do
        train_pid=$(pgrep -P "${tracer_pid}" | head -n 1 || true)
        [[ -n "${train_pid}" ]] && break
        kill -0 "${tracer_pid}" 2>/dev/null || break
        sleep 0.05
    done
    if [[ -z "${train_pid}" ]]; then
        wait "${tracer_pid}"
        train_status=$?
        summarize_signal_trace "${signal_trace}"
        echo "[SIGNAL-TRACE] strace failed before launching the training process (status=${train_status})." >&2
        (( train_status == 0 )) && train_status=4
        exit "${train_status}"
    fi
    echo "[SIGNAL-TRACE] tracer_pid=${tracer_pid} train_pid=${train_pid} log=${signal_trace}"
    wait "${tracer_pid}"
    train_status=$?
    if (( stop_requested )); then
        if kill -0 "${tracer_pid}" 2>/dev/null; then
            wait "${tracer_pid}" 2>/dev/null || true
        fi
        summarize_signal_trace "${signal_trace}"
        echo "[PERSISTENT] Stop requested; child checkpointed and wrapper is exiting (dmon=${dmon_log})."
        exit 0
    fi
    summarize_signal_trace "${signal_trace}"
    train_pid=""
    tracer_pid=""
    set -e

    latest_run=$(find "${LOG_ROOT}" -mindepth 1 -maxdepth 1 -type d -name "*_${run_name}" \
        -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2- || true)
    if [[ -n "${latest_run}" && -f "${latest_run}/stability_state.json" ]]; then
        state_status=$(read_state_field "${latest_run}/stability_state.json" status)
        saved_metric=$(read_state_field "${latest_run}/stability_state.json" best_metric)
        if [[ -n "${saved_metric}" ]]; then
            best_metric=${saved_metric}
        fi
        if [[ -f "${latest_run}/model_best.pt" ]]; then
            best_checkpoint=${latest_run}/model_best.pt
        fi
        progress_checkpoint=$(latest_progress_checkpoint "${latest_run}" || true)
        checkpoint=${progress_checkpoint:-${best_checkpoint}}
        if [[ "${state_status}" == "early_stopped" ]]; then
            echo "[PERSISTENT] Training converged via early stopping in ${latest_run}."
            exit 0
        fi
    fi
    if [[ -n "${latest_run}" && -f "${latest_run}/model_final.pt" ]]; then
        echo "[PERSISTENT] Training completed normally in ${latest_run}."
        exit 0
    fi

    echo "[PERSISTENT] Training exited with status ${train_status}; resuming in 5 seconds."
    sleep 5
done
