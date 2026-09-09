#!/bin/bash
# run_nsys.sh -- Phase 4: NVIDIA Nsight Systems profiling driver.
# Corrected: quotes the output path and checks that `nsys` is on PATH
# before running, so the script fails with a clear message on machines
# without the Nsight CLI installed instead of a confusing shell error.

set -euo pipefail

if ! command -v nsys &> /dev/null; then
    echo "Error: nsys (NVIDIA Nsight Systems CLI) not found on PATH." >&2
    echo "Install the CUDA Toolkit / Nsight Systems package first." >&2
    exit 1
fi

OUT_NAME="fl_cuda_consensus_report"

nsys profile \
  --trace=cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  --stats=true \
  --force-overwrite=true \
  --output="${OUT_NAME}" \
  python3 profile_runner.py

nsys stats \
  --report gputrace,kernsum,memidxsum \
  "${OUT_NAME}.nsys-rep"
