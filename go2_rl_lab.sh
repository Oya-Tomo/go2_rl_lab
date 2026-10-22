#!/usr/bin/env bash
#
# Copyright (c) 2026 Tianyang Wu
# SPDX-License-Identifier: MIT

set -euo pipefail

GO2_RL_LAB_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_exe="${PYTHON_EXE:-python}"
rsl_rl_path="${GO2_RL_LAB_PATH}/source/rsl_rl"
go2_rl_lab_path="${GO2_RL_LAB_PATH}/source/go2_rl_lab"

configure_cuda_library_path() {
    local cuda_lib_path
    cuda_lib_path="$(
        "${python_exe}" -c '
import importlib.util
from pathlib import Path

for module_name in ("nvidia.cu13", "nvidia.cuda_nvrtc"):
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.submodule_search_locations is None:
        continue
    for location in spec.submodule_search_locations:
        lib_path = Path(location) / "lib"
        if any(lib_path.glob("libnvrtc-builtins.so*")):
            print(lib_path)
            raise SystemExit
'
    )"
    if [[ -n "${cuda_lib_path}" && ":${LD_LIBRARY_PATH:-}:" != *":${cuda_lib_path}:"* ]]; then
        export LD_LIBRARY_PATH="${cuda_lib_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
}

ensure_optional_dependencies() {
    "${python_exe}" "${GO2_RL_LAB_PATH}/scripts/utils/ensure_dependencies.py" "$1"
}

usage() {
    echo "Usage: ./go2_rl_lab.sh {--install|--generate-vscode-settings|--list|--train|--train-distributed|--play|--deploy-mujoco|--deploy-sdk2} [arguments]"
}

case "${1:-}" in
    -i|--install)
        "${python_exe}" -m pip install -e "${rsl_rl_path}"
        "${python_exe}" -m pip install --no-deps -e "${go2_rl_lab_path}"
        ensure_optional_dependencies video
        "${python_exe}" "${GO2_RL_LAB_PATH}/scripts/utils/generate_vscode_settings.py"
        ;;
    -g|--generate-vscode-settings)
        shift
        "${python_exe}" "${GO2_RL_LAB_PATH}/scripts/utils/generate_vscode_settings.py" "$@"
        ;;
    -l|--list)
        shift
        "${python_exe}" "${GO2_RL_LAB_PATH}/scripts/utils/list_envs.py" "$@"
        ;;
    -t|--train)
        shift
        configure_cuda_library_path
        "${python_exe}" "${GO2_RL_LAB_PATH}/scripts/rsl_rl/train.py" "$@"
        ;;
    --train-distributed)
        shift
        if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
            echo "--train-distributed requires CUDA_VISIBLE_DEVICES, e.g. CUDA_VISIBLE_DEVICES=4,5 ./go2_rl_lab.sh --train-distributed ..." >&2
            exit 2
        fi
        configure_cuda_library_path
        nproc_per_node="$(awk -F',' '{print NF}' <<<"${CUDA_VISIBLE_DEVICES}")"
        "${python_exe}" -m torch.distributed.run \
            --standalone --nnodes 1 --nproc_per_node "${nproc_per_node}" --max-restarts 0 \
            "${GO2_RL_LAB_PATH}/scripts/rsl_rl/train.py" --distributed "$@"
        ;;
    -p|--play)
        shift
        configure_cuda_library_path
        "${python_exe}" "${GO2_RL_LAB_PATH}/scripts/rsl_rl/play.py" "$@"
        ;;
    --deploy-mujoco)
        shift
        ensure_optional_dependencies mujoco
        "${python_exe}" "${GO2_RL_LAB_PATH}/deploy/deploy_mujoco/deploy_go2.py" "$@"
        ;;
    --deploy-sdk2)
        shift
        # unitree_sdk2 deployment runs in its own uv environment (Python 3.10 for cyclonedds wheels).
        uv run --project "${GO2_RL_LAB_PATH}/deploy/deploy_sdk2" python "${GO2_RL_LAB_PATH}/deploy/deploy_sdk2/deploy_go2.py" "$@"
        ;;
    -h|--help|"")
        usage
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
