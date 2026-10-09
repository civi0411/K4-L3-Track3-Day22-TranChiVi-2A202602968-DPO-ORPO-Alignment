#!/usr/bin/env bash
# Kaggle one-line setup for Day 22 lab.
# Usage in Kaggle notebook cell or terminal:
#     !bash setup-kaggle.sh

set -euo pipefail

echo "[kaggle] Day 22 lab — Kaggle setup"
echo "[kaggle] Stack: unsloth + trl + peft + bitsandbytes + llama-cpp-python"
echo

# ── 1. Enforce single GPU for Unsloth if on Kaggle T4x2 ───────────────────
export CUDA_VISIBLE_DEVICES=0

# ── 2. Auto-detect tier from torch.cuda ──────────────────────────────────
TIER=$(python - <<'PY'
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
if not torch.cuda.is_available():
    print("CPU")
else:
    gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print("BIGGPU" if gb >= 22 else "T4")
PY
)
echo "[kaggle] Detected tier: $TIER (using GPU 0)"

if [ "$TIER" = "CPU" ]; then
    echo "[kaggle] No GPU detected. In Kaggle Notebook Settings → Accelerator → select GPU T4 x 2 or P100."
    exit 1
fi

# ── 3. Install deps ──────────────────────────────────────────────────────
pip install -q -r requirements.txt

# ── 4. Convert Jupytext sources ──────────────────────────────────────────
jupytext --to notebook --update notebooks/*.py 2>/dev/null || jupytext --to notebook notebooks/*.py

# ── 5. .env scaffold ─────────────────────────────────────────────────────
[ -f .env ] || cp .env.example .env
sed -i.bak "s/^COMPUTE_TIER=.*/COMPUTE_TIER=$TIER/" .env && rm -f .env.bak

# ── 6. Make output folders ───────────────────────────────────────────────
mkdir -p data/pref data/eval adapters models gguf submission/screenshots

cat <<EOF

[kaggle] Done — tier = $TIER.

In Kaggle, you can now:
    1. Run the stitched notebook:
         - kaggle/Lab22_DPO_T4_Kaggle.ipynb (or colab/Lab22_DPO_T4.ipynb)
    2. Or use make / terminal:
         !make smoke
         !make pipeline

Tip: Make sure Notebook Settings → Internet is turned ON!

EOF
