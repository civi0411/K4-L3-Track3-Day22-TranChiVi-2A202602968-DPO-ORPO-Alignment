#!/usr/bin/env python3
"""Plot the β-sweep from adapters/dpo-b*/dpo_metrics.json.

Usage:
    python scripts/eval_judge.py --plot-sweep
    python scripts/eval_judge.py --plot-sweep --sweep-dir adapters --output submission/screenshots/bonus-beta-sweep.png

The judge itself lives in NB4 (`lab22.judge`); run NB4 with
DPO_ADAPTER_OVERRIDE=adapters/dpo-b0.50 to judge one sweep point.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

REPO = Path(__file__).resolve().parent.parent


def load_sweep(sweep_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(sweep_dir.glob("dpo-b*/dpo_metrics.json")):
        m = json.loads(path.read_text())
        if m.get("beta") is not None:
            rows.append({"dir": path.parent.name, **m})
    return sorted(rows, key=lambda r: r["beta"])


def plot_sweep(rows: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt

    betas = [r["beta"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(betas, [r.get("eval_reward_gap") for r in rows], "o-", color="#1a3355")
    axes[0].set_title("held-out margin  β·Δlog-ratio")
    axes[1].plot(betas, [r.get("eval_chosen_reward") for r in rows], "o-", color="#2e548a", label="chosen")
    axes[1].plot(betas, [r.get("eval_rejected_reward") for r in rows], "o-", color="#c83538", label="rejected")
    axes[1].legend()
    axes[1].set_title("held-out implicit rewards")
    axes[2].plot(betas, [r.get("eval_reward_accuracy") for r in rows], "o-", color="#444")
    axes[2].set_ylim(0, 1)
    axes[2].set_title("held-out reward accuracy")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("β")
        ax.axhline(0, color="#888", linestyle=":", linewidth=0.7)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"β-sweep ({len(rows)} runs). Margins scale with β: compare accuracy, not raw margin.", y=1.03)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=120, bbox_inches="tight")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-sweep", action="store_true", help="plot adapters/dpo-b* results")
    parser.add_argument("--sweep-dir", default=str(REPO / "adapters"))
    parser.add_argument("--output", default=str(REPO / "submission" / "screenshots" / "bonus-beta-sweep.png"))
    args = parser.parse_args()
    if not args.plot_sweep:
        parser.print_help()
        print("\nThe judge runs in NB4 (notebooks/04_compare_and_eval.py).")
        return 0
    rows = load_sweep(Path(args.sweep_dir))
    if not rows:
        print(f"No {args.sweep_dir}/dpo-b*/dpo_metrics.json found. Run `make beta-sweep` first.")
        return 1
    plot_sweep(rows, Path(args.output))
    for r in rows:
        acc, gap = r.get("eval_reward_accuracy"), r.get("eval_reward_gap")
        print(f"β={r['beta']:<5} eval_acc={acc} eval_gap={gap} diag={r.get('diagnosis')}")
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
