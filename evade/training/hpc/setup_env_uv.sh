#!/bin/bash
# Build the EVAde training venv on the cluster's scratch, usable from every node (login and compute).
# Why uv: compute nodes do not have the login node's /usr/bin/python3.12, and the cluster's Python modules
# (Compilers/Python/3.13.13, 3.10.x) are built without _ctypes or _bz2, which torch and pandas need.
# uv installs a self-contained CPython under $PROJECT/python. Run once on the login node (light: downloads only):
#   bash setup_env_uv.sh            BOOTSTRAP_PY=<any python with pip> to choose the interpreter that installs uv
set -euo pipefail
PROJECT=${PROJECT:-/scratch/$USER/evade}
BOOT=${BOOTSTRAP_PY:-python3}
cd "$PROJECT"
unset PYTHONPATH
export UV_PYTHON_INSTALL_DIR=$PROJECT/python UV_CACHE_DIR=$PROJECT/.uvcache
[ -x tools/bin/uv ] || "$BOOT" -m pip install -q --target tools uv
tools/bin/uv python install 3.12
[ -x venv/bin/python ] || tools/bin/uv venv -q --python 3.12 venv
tools/bin/uv pip install -q --python venv/bin/python torch --index-url https://download.pytorch.org/whl/cu126
tools/bin/uv pip install -q --python venv/bin/python numpy pandas scikit-learn onnx
venv/bin/python -c 'import torch, numpy, sklearn, onnx; print("torch", torch.__version__, "cuda", torch.version.cuda)'
