#!/usr/bin/env python3
"""Build the single-file Colab & Kaggle notebooks from the Jupytext sources.

The Colab and Kaggle bundles are generated from `notebooks/*.py` and `lab22/*.py`.
`lab22/*.py` and `scripts/verify.py` are written with `%%writefile`, then every
notebook stage follows in order.

    python scripts/build_colab.py          # rewrite colab/*.ipynb and kaggle/*.ipynb
    python scripts/build_colab.py --check  # exit 1 if they are stale
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STAGES = [
    ("00_dpo_loss_from_scratch", "core · CPU"),
    ("01_sft_mini", "core"),
    ("02_preference_data", "core"),
    ("03_dpo_train", "core"),
    ("03b_dpo_variants", "bonus"),
    ("04_compare_and_eval", "core"),
    ("05_merge_deploy_gguf", "bonus"),
    ("06_benchmark", "bonus"),
    ("07_grpo_bonus", "bonus"),
]
TARGETS = {
    "colab/Lab22_DPO_T4.ipynb": ("T4", "colab"),
    "colab/Lab22_DPO_BigGPU.ipynb": ("BIGGPU", "colab"),
    "kaggle/Lab22_DPO_T4_Kaggle.ipynb": ("T4", "kaggle"),
}
CELL = re.compile(r"^# %%(?P<md> \[markdown\])?.*$", re.MULTILINE)


def requirements() -> list[str]:
    """Requirement specs from requirements.txt, minus test/notebook tooling."""
    skip = ("jupyterlab", "jupytext", "pytest")
    specs = []
    for line in (REPO / "requirements.txt").read_text(encoding="utf-8").splitlines():
        spec = line.split("#", 1)[0].strip()
        if spec and not spec.startswith(skip):
            specs.append(spec)
    return specs


def source_lines(text: str) -> list[str]:
    lines = text.splitlines(keepends=True)
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return lines


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source_lines(text)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source_lines(text)}


def percent_cells(path: Path) -> list[dict]:
    """Parse a Jupytext py:percent file (header block dropped)."""
    text = path.read_text(encoding="utf-8")
    marks = list(CELL.finditer(text))
    cells = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        body = text[m.end() :end].strip("\n")
        if not body.strip():
            continue
        if m.group("md"):
            body = "\n".join(line[2:] if line.startswith("# ") else line.lstrip("#") for line in body.splitlines())
            cells.append(md(body))
        else:
            cells.append(code(body))
    return cells


RELEASE_GPU = (
    "# Free GPU memory held by the previous stage (model, trainer, llama.cpp handle).\n"
    "import gc\n"
    "for _name in ('trainer', 'model', 'ref_model', 'llm', 'policy', 'ref', 'tokenizer'):\n"
    "    globals().pop(_name, None)\n"
    "gc.collect()\n"
    "try:\n"
    "    import torch\n"
    "    if torch.cuda.is_available():\n"
    "        torch.cuda.empty_cache()\n"
    "        print(f'GPU memory in use: {torch.cuda.memory_allocated() / 1e9:.2f} GB')\n"
    "except ImportError:\n"
    "    pass"
)


def render(tier: str, platform: str = "colab") -> dict:
    big = tier == "BIGGPU"
    is_kaggle = platform == "kaggle"
    pins = " ".join(f'"{s}"' for s in requirements())

    if is_kaggle:
        header_text = (
            f"# Lab 22 — DPO/ORPO Alignment (Kaggle {tier} tier)\n\n"
            "**Học viên:** Trần Chí Vĩ · **Mã HV:** 2A202602968  \n"
            "**Track 3 · Day 22 · VinUni AICB.** Sổ tay tối ưu hoá để chạy trực tiếp trên **Kaggle Notebooks**.\n\n"
            "### ⚙️ Hướng dẫn cài đặt trên Kaggle:\n"
            "1. **Bật Internet (Bắt buộc):** Ở cột bên phải (Notebook options) → **Settings** → **Internet** → Gạt sang **On** (để tải thư viện và mô hình từ Hugging Face).\n"
            "2. **Chọn GPU:** **Settings** → **Accelerator** → Chọn **GPU T4 x 2** (hệ thống tự cấu hình dùng GPU 0 cho Unsloth) hoặc **GPU P100**.\n"
            "3. **Thứ tự thực hiện:** Chạy lần lượt các cell từ trên xuống dưới (hoặc bấm **Run All**).\n"
            "4. **Tải kết quả nộp bài:** Chạy cell cuối cùng để tải `submission_artifacts.zip` (hoặc tải từ tab **Output** ở cột bên phải).\n\n"
            "Core: NB0 → NB4. Bonus: NB3b (variants), NB5 (GGUF), NB6 (lm-eval), NB7 (GRPO).\n"
            "Mỗi phần tự nạp lại từ ổ đĩa khi cần, nếu bị ngắt phiên có thể khởi động lại và chạy tiếp từ phần bị dừng."
        )
    else:
        header_text = (
            f"# Lab 22 — DPO/ORPO Alignment ({'BigGPU' if big else 'T4'} tier)\n\n"
            "**Học viên:** Trần Chí Vĩ · **Mã HV:** 2A202602968  \n"
            "**Track 3 · Day 22 · VinUni AICB.** Hỗ trợ chạy trên **Google Colab** và **Kaggle** (GPU T4 / A100 / L4).\n\n"
            "Core: NB0 → NB4. Bonus: NB3b (variants), NB5 (GGUF), NB6 (lm-eval), NB7 (GRPO).\n"
            "Mỗi phần tự nạp lại từ ổ đĩa khi cần, nếu bị ngắt phiên có thể khởi động lại và chạy tiếp từ phần bị dừng."
        )

    setup_code = (
        "import os\n"
        f'os.environ["COMPUTE_TIER"] = "{tier}"\n'
        "\n"
        "# Kaggle T4x2 cung cấp 2 GPU; Unsloth yêu cầu chạy trên Single GPU (GPU 0)\n"
        'if os.path.exists("/kaggle/working"):\n'
        '    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")\n'
        '    print("[Kaggle] Phát hiện môi trường Kaggle. Đã đặt CUDA_VISIBLE_DEVICES=0.")\n'
        'elif os.path.exists("/content"):\n'
        '    print("[Colab] Phát hiện môi trường Google Colab.")\n'
        "\n"
        "# NB4 tự động chấm điểm bằng hội đồng 2 local reward models (không cần API key).\n"
        "# Tuỳ chọn: dùng API judge nếu muốn đối chứng chéo:\n"
        '# os.environ["JUDGE_PROVIDER"] = "gemini"   # hoặc "openai" / "anthropic"\n'
        '# os.environ["JUDGE_MODEL"] = "gemini-1.5-flash"\n'
        "# Không để API key lộ trong notebook."
    )

    workdir_code = (
        "import os\n"
        "from pathlib import Path\n"
        "\n"
        "# Tự động phát hiện thư mục làm việc (Kaggle vs Colab vs Local)\n"
        'if Path("/kaggle/working").exists():\n'
        '    WORK = Path("/kaggle/working/lab22")\n'
        'elif Path("/content").exists():\n'
        '    WORK = Path("/content/lab22")\n'
        "else:\n"
        '    WORK = Path.cwd() / "lab22"\n'
        "\n"
        '(WORK / "lab22").mkdir(parents=True, exist_ok=True)\n'
        '(WORK / "scripts").mkdir(parents=True, exist_ok=True)\n'
        "os.chdir(WORK)\n"
        'print(f"Working directory: {Path.cwd()}")'
    )

    cells = [
        md(header_text),
        md("## A. Setup"),
        code(setup_code),
        code(f"!pip install -q {pins}" + (' "vllm>=0.10"' if big else "")),
        code(workdir_code),
        md("### Helper package `lab22/` (same files as the repo)"),
    ]

    for module in sorted((REPO / "lab22").glob("*.py")):
        body = module.read_text(encoding="utf-8")
        cells.append(code(f"%%writefile lab22/{module.name}\n{body}"))

    body_verify = (REPO / "scripts" / "verify.py").read_text(encoding="utf-8")
    cells.append(code(f"%%writefile scripts/verify.py\n{body_verify}"))

    for i, (stem, kind) in enumerate(STAGES):
        if i:
            cells.append(code(RELEASE_GPU))
        cells.append(md(f"---\n# ⏵ `notebooks/{stem}.py` ({kind})"))
        cells.extend(percent_cells(REPO / "notebooks" / f"{stem}.py"))

    cells.append(
        md(
            "---\n# ⏵ Tải kết quả về nộp bài (Download Submission Artifacts)\n\n"
            "Chạy cell dưới đây để nén toàn bộ các file kết quả và biểu đồ cần thiết, rồi tải file "
            "`submission_artifacts.zip` về máy tính của bạn (hỗ trợ cả Kaggle và Google Colab)."
        )
    )

    download_code = (
        "# Nén các artifacts cần nộp thành 1 file zip\n"
        "!zip -r -q submission_artifacts.zip \\\n"
        "    submission/screenshots \\\n"
        "    data/eval \\\n"
        "    data/pref \\\n"
        "    adapters/sft-mini/adapter_config.json \\\n"
        "    models/sft-merged/config.json \\\n"
        "    adapters/dpo/adapter_config.json \\\n"
        "    adapters/dpo/dpo_metrics.json \\\n"
        "    adapters/dpo/split.json \\\n"
        "    adapters/variants/variants_summary.json \\\n"
        "    adapters/grpo/grpo_metrics.json 2>/dev/null || true\n"
        "\n"
        "from pathlib import Path\n"
        "import os\n"
        "zip_file = Path('submission_artifacts.zip')\n"
        "if zip_file.exists():\n"
        "    size_mb = zip_file.stat().st_size / (1024 * 1024)\n"
        '    print(f"✓ Đã tạo thành công submission_artifacts.zip ({size_mb:.2f} MB)")\n'
        "    # 1. Nếu chạy trên Google Colab: tự động tải xuống qua browser\n"
        "    try:\n"
        "        from google.colab import files\n"
        "        files.download(str(zip_file))\n"
        '        print("✓ Đang kích hoạt tải về trên Google Colab...")\n'
        "    except Exception:\n"
        "        pass\n"
        "    # 2. Nếu chạy trên Kaggle: copy ra /kaggle/working và hiển thị FileLink\n"
        "    if Path('/kaggle/working').exists():\n"
        "        import shutil\n"
        "        target = Path('/kaggle/working/submission_artifacts.zip')\n"
        "        if zip_file.resolve() != target.resolve():\n"
        "            shutil.copy(zip_file, target)\n"
        '        print(f"✓ File đã được lưu tại: {target}")\n'
        '        print("👉 Trên Kaggle: Bạn có thể tải file từ mục Output (cột bên phải) hoặc click link bên dưới:")\n'
        "        from IPython.display import FileLink, display\n"
        "        display(FileLink(str(target)))\n"
        "    else:\n"
        "        from IPython.display import FileLink, display\n"
        "        display(FileLink(str(zip_file)))\n"
        "else:\n"
        '    print("Chưa tìm thấy file submission_artifacts.zip.")\n'
    )
    cells.append(code(download_code))

    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "A100" if big else "T4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if notebooks differ from the sources")
    args = parser.parse_args()
    stale = []
    for rel_path, (tier, platform) in TARGETS.items():
        path = REPO / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        nb = render(tier, platform=platform)
        if args.check:
            if not path.exists() or json.loads(path.read_text(encoding="utf-8")) != nb:
                stale.append(rel_path)
            continue
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {rel_path} ({len(nb['cells'])} cells)")
    if stale:
        print(f"stale: {stale}; run `python scripts/build_colab.py`")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
