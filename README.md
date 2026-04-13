# PRISM: Phase-structured Reasoning with Integrated Subject-expert Modules

**NeurIPS 2026 target** — Olympiad mathematics reasoning in sub-3B vision-language models.

## Quick start

```bash
source scripts/setup/env.sh
pip install -e .
```

## Run POC (small end-to-end test, ~20 problems per domain)

```bash
# Uses Qwen/Qwen3.5-35B-A3B as teacher (35B MoE, VL, fits on 1 GH200)
python scripts/poc_trace_gen.py --n-problems 20 --gpu 0
# Secondary (faster, smaller): --teacher Qwen/Qwen3-VL-8B-Instruct
```

## Full pipeline

```bash
# 1. Generate expert traces (all 4 GPUs)
bash scripts/generate_traces.sh Qwen/Qwen3-VL-30B-A3B 2500

# 2. Stage 0: LoRA validation
bash scripts/train_stage0.sh

# 3. Stage 1-2: Full PRISM training
bash scripts/train_stage2.sh

# 4. Evaluation
bash scripts/eval_all.sh results/stage2
```

## Model usage (after training)

```python
from transformers import AutoModel, AutoProcessor
model = AutoModel.from_pretrained("debajyoti/prism-0.8b", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("debajyoti/prism-0.8b", trust_remote_code=True)
outputs = model(**inputs)
```

## Status: See PROGRESS.md
