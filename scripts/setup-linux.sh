#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3.10 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
# Official upstream measured stack. Blackwell requires a separately validated port.
.venv/bin/python -m pip install torch==2.5.0+cu124 torchvision==0.20.0+cu124 --index-url https://download.pytorch.org/whl/cu124
.venv/bin/python -m pip install packaging ninja psutil
.venv/bin/python -m pip install --no-build-isolation flash-attn==2.7.4.post1
if [ ! -d upstream/Hi3Dv1.0/sparc3d ]; then
  git clone https://github.com/hi3d-official/Hi3Dv1.0.git upstream/Hi3Dv1.0
fi
if [ -d upstream/Hi3Dv1.0/.git ]; then
  git -C upstream/Hi3Dv1.0 checkout --detach 23dcc684bc6e1fab86ecafb99ec5270edad122f9
fi
.venv/bin/python -m pip install -c upstream/Hi3Dv1.0/docker/constraints.txt -e 'upstream/Hi3Dv1.0[inference]'
.venv/bin/python -m pip install '.[workbench]'
echo 'Run .venv/bin/python -m hi3d_workbench.model_cli download, then .venv/bin/python -m hi3d_workbench --open'
