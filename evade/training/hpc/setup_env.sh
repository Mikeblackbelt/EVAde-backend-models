#!/bin/bash
# Build (or repair) the EVAde training venv on arrow scratch: /scratch/$USER/evade/venv
# Torch CUDA 12.6 build: supports V100 (sm_70) and A100 (sm_80), NOT the K20m (sm_35) on the login node.
# Run on the arrow LOGIN node (pip install is allowed there; nothing heavy), through the broker:
#   srv.py --host arrow run "bash -l /scratch/<hpc-user>/evade/code/evade/training/hpc/setup_env.sh" --timeout 1200
# (bash -l so the `module` command exists; the torch wheel is about 2-3 GB, so allow time for the download)
# Idempotent: scratch is purged, so re-run it whenever the venv is missing or broken; an intact venv is kept.
# x86_64 only. partgh200 (if aarch64) would need its own venv built inside a job on that partition; not planned.
set -euo pipefail

PROJECT=/scratch/$USER/evade
VENV=$PROJECT/venv
TORCH_VERSION=2.7.1                                   # cu126 wheel; change only together with a smoke run
TORCH_INDEX=https://download.pytorch.org/whl/cu126
# Python from a module if the system python3 is too old (needs >= 3.10). Fill from discover.sh "modules" output,
# e.g. PYTHON_MODULE=python/3.11.x ; leave empty to use the system python3.
PYTHON_MODULE=""

mkdir -p "$PROJECT"/{code,data,runs,slurm}
if [ "$(uname -m)" != "x86_64" ]; then echo "this venv is for x86_64 nodes; got $(uname -m)"; exit 2; fi

if [ -n "$PYTHON_MODULE" ]; then
  set +u; module load "$PYTHON_MODULE"; set -u      # Lmod scripts are not nounset-safe
fi
PYBIN=$(command -v python3)
"$PYBIN" -c 'import sys; assert sys.version_info >= (3, 10), sys.version' \
  || { echo "python3 at $PYBIN is older than 3.10: set PYTHON_MODULE (module spider python)"; exit 2; }

if [ -x "$VENV/bin/python" ] && "$VENV/bin/python" -c "import torch, onnxruntime, numpy" 2>/dev/null; then
  echo "venv OK at $VENV, keeping it"
else
  echo "creating venv at $VENV with $PYBIN"
  rm -rf "$VENV"
  "$PYBIN" -m venv "$VENV"
  "$VENV/bin/python" -m pip install --upgrade --quiet pip wheel
  "$VENV/bin/python" -m pip install --quiet "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX"
  # same libraries as requirements.txt (minus the API server) plus ONNX export and CPU inference
  "$VENV/bin/python" -m pip install --quiet numpy scipy pandas pyarrow wfdb pyedflib scikit-learn joblib pytest \
      "onnx==1.18.0" "onnxruntime==1.22.0"
fi

# record exactly what was installed, next to the venv; lib.sh loads the same python module in jobs if one was used
"$VENV/bin/python" -m pip freeze > "$VENV/REQUIREMENTS.lock"
printf '%s\n' "$PYTHON_MODULE" > "$VENV/PYTHON_MODULE"
"$VENV/bin/python" - <<'EOF'
import torch, onnx, onnxruntime, numpy
print("torch", torch.__version__, "cuda build", torch.version.cuda, "archs", torch.cuda.get_arch_list())
print("onnx", onnx.__version__, "onnxruntime", onnxruntime.__version__, "numpy", numpy.__version__)
# get_arch_list() is empty when no usable GPU is visible (expected on the K20m login node), so only check the build
# here; the live V100/A100 check is gpu_guard in lib.sh, on the compute node
assert (torch.version.cuda or "").startswith("12."), f"expected a CUDA 12.x torch build, got {torch.version.cuda}"
EOF
echo "lock file: $VENV/REQUIREMENTS.lock"
