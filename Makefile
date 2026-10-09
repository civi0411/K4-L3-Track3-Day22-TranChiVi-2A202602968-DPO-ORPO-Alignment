## Day 22 — DPO/ORPO Alignment lab.
## Tier-aware via COMPUTE_TIER (T4 default, BIGGPU optional).

VENV     := .venv
PY       := $(VENV)/bin/python
PIP      := $(VENV)/bin/pip
JUPYTEXT := $(VENV)/bin/jupytext
PYTEST   := $(VENV)/bin/pytest
JUPYTER  := $(VENV)/bin/jupyter

# If running on Colab there's no venv — fall back to system python.
ifeq ($(wildcard $(PY)),)
  PY := $(shell command -v python3 2>/dev/null || echo python)
  PIP := $(shell command -v pip3 2>/dev/null || echo pip)
  JUPYTEXT := jupytext
  PYTEST := pytest
  JUPYTER := jupyter
endif

.DEFAULT_GOAL := help

help: ## Show this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n\nDay 22 DPO Lab targets:\n"} \
	      /^[a-zA-Z0-9_-]+:.*?##/ { printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# ─────────────────────────────────────────────────────────────
# Setup — auto-detect Colab vs laptop
# ─────────────────────────────────────────────────────────────

setup: ## Auto-detect Kaggle vs Colab vs laptop, install deps + smoke check
	@if [ -d /kaggle ]; then \
	  bash setup-kaggle.sh; \
	elif [ -d /content ]; then \
	  bash setup-colab.sh; \
	else \
	  bash setup-laptop.sh; \
	fi

smoke: ## Import + GPU + sources check before training
	@$(PY) scripts/verify.py --smoke

# Each stage: py:percent source → .ipynb → execute in place (outputs kept for grading).
define run_nb
	@$(JUPYTEXT) --to notebook --update notebooks/$(1).py
	@$(JUPYTER) nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/$(1).ipynb
endef

# ─────────────────────────────────────────────────────────────
# Pipeline — core NB0-NB4; bonus NB3b, NB5, NB6, NB7
# ─────────────────────────────────────────────────────────────

nb0: ## NB0 — DPO loss from scratch (CPU, ~1 min)
	$(call run_nb,00_dpo_loss_from_scratch)

sft: ## NB1 — SFT-mini + merged reference model (~15 min T4)
	$(call run_nb,01_sft_mini)

data: ## NB2 — Vietnamese preference data, held-out split, length bias (~3 min)
	$(call run_nb,02_preference_data)

dpo: ## NB3 — DPO against the SFT reference (~30 min T4)
	$(call run_nb,03_dpo_train)

variants: ## NB3b (bonus) — DPO / RPO / DPO-norm / LD-DPO / ORPO
	$(call run_nb,03b_dpo_variants)

eval: ## NB4 — SFT vs SFT+DPO, two-order judge on held-out prompts
	$(call run_nb,04_compare_and_eval)

deploy: ## NB5 (bonus) — SFT+DPO → GGUF Q4_K_M + llama.cpp smoke
	$(call run_nb,05_merge_deploy_gguf)

bench: ## NB6 (bonus) — lm-eval IFEval / GSM8K / Global-MMLU-vi with chat template
	$(call run_nb,06_benchmark)

grpo: ## NB7 (bonus) — GRPO with a verifiable GSM8K reward
	$(call run_nb,07_grpo_bonus)

pipeline: nb0 sft data dpo eval ## Core notebooks NB0-NB4

pipeline-full: pipeline variants deploy bench grpo ## Core + all bonus notebooks

# ─────────────────────────────────────────────────────────────
# Bonus rigor add-on
# ─────────────────────────────────────────────────────────────

beta-sweep: ## Retrain DPO with beta in {0.05, 0.1, 0.5} and plot held-out rewards
	@$(PY) scripts/train_dpo.py --beta 0.05 --output-dir adapters/dpo-b0.05
	@$(PY) scripts/train_dpo.py --beta 0.1  --output-dir adapters/dpo-b0.10
	@$(PY) scripts/train_dpo.py --beta 0.5  --output-dir adapters/dpo-b0.50
	@$(PY) scripts/eval_judge.py --plot-sweep --sweep-dir adapters --output submission/screenshots/bonus-beta-sweep.png

colab: ## Regenerate colab/*.ipynb from notebooks/ + lab22/
	@$(PY) scripts/build_colab.py

# ─────────────────────────────────────────────────────────────
# Verify + clean
# ─────────────────────────────────────────────────────────────

verify: ## Pre-submission gatekeeper — checks artifacts + REFLECTION edited
	@$(PY) scripts/verify.py

lab: ## Open Jupyter Lab (laptop only)
	@$(JUPYTEXT) --to notebook --update notebooks/*.py 2>/dev/null || true
	@$(JUPYTER) lab --notebook-dir=notebooks --ServerApp.token='' --no-browser

test: ## CPU tests: lab22 logic, API drift, Colab bundles in sync
	@$(PYTEST) -q scripts/

clean: ## Wipe models/, adapters/, data/pref, data/eval, gguf*/
	rm -rf models/ adapters/sft-mini adapters/dpo adapters/dpo-b* adapters/variants adapters/grpo \
	       adapters/*-checkpoints data/pref/ data/eval/ gguf*/ \
	       notebooks/*.ipynb notebooks/.ipynb_checkpoints \
	       __pycache__ scripts/__pycache__ lab22/__pycache__

clean-all: clean ## Wipe everything including venv + HF cache
	rm -rf $(VENV) ~/.cache/huggingface/hub

.PHONY: help setup smoke nb0 sft data dpo variants eval deploy bench grpo pipeline pipeline-full beta-sweep colab verify lab test clean clean-all
