---
name: Environment Setup
description: Hardware, storage policy, env vars, and operational rules for PRISM
type: reference
---

**Hardware:** 4× NVIDIA GH200 GPUs (~96GB usable each), >1TB RAM
**CUDA:** 13.0  
**Python:** 3.12.13  
**Conda env:** `/iopsstor/scratch/cscs/dasgupta/research/ideas/AMSD/.venv` (shared with AMSD — already active, do NOT source activate)

---

## GPU utilization mandate
All 4 GPUs must be busy at all times. ~96GB VRAM per GPU, used as fully as possible. No idle GPUs. Parallelize by default — train different expert blocks on different GPUs simultaneously.

---

## Storage policy (strictly enforced)

| Data | Location | Persists? | In git? |
|------|----------|-----------|---------|
| Runtime model weights | `/tmp/prism_models/` | Until reboot (RAM) | No |
| HF download cache | `PRISM_ROOT/.cache/huggingface/` | Yes (disk) | No (gitignored) |
| Expert block checkpoints | `PRISM_ROOT/results/stageN/` | Yes | Yes (<100MB) |
| Eval results, traces, configs | `PRISM_ROOT/results/` | Yes | Yes |
| Large backbone weights | HF Hub only | — | No |

**Models in /tmp only.** The machine has >1TB RAM. `/tmp` is RAM-backed — loading from there to GPU is the fastest possible path. Never load models from disk paths or from the project repo.

**All artifacts in `results/`.** Every eval JSON, every trace file, every checkpoint. Committed to git.

---

## Environment variables (`scripts/setup/env.sh`)

```bash
export PRISM_ROOT=/iopsstor/scratch/cscs/dasgupta/research/ideas/PRISM
export HF_HOME=${PRISM_ROOT}/.cache/huggingface     # persistent download cache
export HF_HUB_CACHE=${PRISM_ROOT}/.cache/huggingface/hub
export PRISM_MODEL_DIR=/tmp/prism_models             # runtime loading (RAM)
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
```

---

## Professional packaging requirement

The project must be installable with `pip install -e .`. PRISMModel must work with `AutoModel.from_pretrained()` from day one. No hardcoded paths. No custom loading code required by downstream users. A researcher anywhere in the world should be able to:

```python
from transformers import AutoModel, AutoProcessor
model = AutoModel.from_pretrained("debajyoti/prism-0.8b", trust_remote_code=True)
```

---

## Key packages
- torch 2.9+cu129 (CUDA 12.9)
- transformers ≥5.5
- peft ≥0.18
- accelerate ≥1.13
- datasets ≥4.8

## Qwen3.5 thinking mode gotcha
Must pass `enable_thinking=False` explicitly via try/except in `apply_chat_template`. Without this, model burns all tokens on `<think>` blocks.

## PROGRESS.md mandate
`PROGRESS.md` in project root must be updated after every experiment and committed to git.
