# PRISM Progress Log

**Project:** PRISM — Phase-structured Reasoning with Integrated Subject-expert Modules  
**Target:** NeurIPS 2026 main track  
**Last updated:** 2026-04-13

---

## Current Status

**Phase:** Foundation complete — ready to begin Stage 0 (trace generation + LoRA validation)

---

## GPU Status

| GPU | Status | Job | PID | Since | Log |
|-----|--------|-----|-----|-------|-----|
| 0 | IDLE | — | — | — | — |
| 1 | IDLE | — | — | — | — |
| 2 | IDLE | — | — | — | — |
| 3 | IDLE | — | — | — | — |

---

## Completed

### 2026-04-13: Foundation / Codebase Setup

**What was done:**
- Initialized git repository, connected to upstream `https://github.com/debajyotidasgupta/PRISM.git`
- Created complete project structure per `program.md` Section 12
- Implemented all core Python modules:
  - `src/prism/model/config.py` — `PRISMConfig(PretrainedConfig)`
  - `src/prism/model/expert_block.py` — `ExpertBlock` (self-attn + SwiGLU FFN)
  - `src/prism/model/cross_mix.py` — `CrossMixModule` (cross-domain attention)
  - `src/prism/model/router.py` — `DomainRouter` (multi-phase, entropy-regularized)
  - `src/prism/model/backbone.py` — backbone loader + insertion point
  - `src/prism/model/prism_model.py` — `PRISMModel(PreTrainedModel)`, full forward
  - `src/prism/data/` — datasets, domain split, trace format, collator
  - `src/prism/generation/` — trace generator, 15 phase×domain prompts
  - `src/prism/training/` — train_router, train_expert, train_crossmix, train_lora, train_e2e
  - `src/prism/eval/` — metrics, eval_prism, ablations
- Created all config YAML files (model, training, eval)
- Created all shell scripts (generate_traces.sh, train_stage0.sh, eval_all.sh)
- Unit tests: ALL PASSED
  - ExpertBlock shape: ✓
  - CrossMixModule shape: ✓
  - DomainRouter soft routing: ✓
  - Misc floor (≥10%): ✓
  - Entropy regularization: ✓
  - Per-phase routing (different weights per phase): ✓
  - Freeze/unfreeze mechanics: ✓
- Package installable: `pip install -e .` ✓

**Key architectural decisions made:**
1. **Multi-phase routing**: Router produces `[B, n_phases, n_domains]` — different domain
   weights for Solve vs Verify vs Correct phases. Algebra problems may be *verified*
   by Combinatorics or Miscellaneous experts.
2. **Soft routing always**: Router NEVER collapses to single expert. Entropy regularization
   (coefficient 0.01) prevents degenerate solutions.
3. **Misc floor**: Miscellaneous expert always gets ≥10% weight (cross-domain tools always active).
4. **All experts run**: Every domain expert participates in every forward pass.
   Routing weights determine contribution, not activation.
5. **CrossMix at every phase**: Information exchange between all 5 domain experts after
   each of the 3 phases (3 CrossMix modules total).

**What has NOT been done yet:**
- Baseline evaluation (Qwen3.5-0.8B thinking=True/False)
- Trace generation (requires running teacher model)
- Any LoRA training (Stage 0)
- Any PRISM block training (Stages 1-2)

---

## Queued Next

### Stage 0: Trace Generation + LoRA Validation (2-3 days compute)

**Step 1** (all 4 GPUs in parallel): Generate 2,500 expert traces per domain
```bash
bash scripts/generate_traces.sh Qwen/Qwen2.5-VL-7B-Instruct 2500
```
Expected output: `results/traces/{domain}_traces.jsonl` (5 files)
Expected pass rate: ~30-60% (traces where teacher's final answer is correct)

**Step 2** (simultaneous with Step 1 on free GPUs): Run baseline evaluations
```bash
prism-eval --model Qwen/Qwen3.5-0.8B --benchmark math500 --gpu 0 --model-name baseline_nothink
prism-eval --model Qwen/Qwen3.5-0.8B --benchmark math500 --gpu 1 --thinking --model-name baseline_think
```

**Step 3** (after Step 1): Train domain LoRA adapters
```bash
bash scripts/train_stage0.sh
```

**Step 4**: Evaluate LoRA adapters and check Stage 0 pass gate:
- ≥3 of 5 domain LoRAs must beat general LoRA on their own domain
- 3-phase traces must not be worse than 1-phase traces

**Pass gate decision: GO/NO-GO for Stage 1**

---

## Key Numbers (to be filled in)

| Metric | Value | Notes |
|--------|-------|-------|
| Baseline Qwen3.5-0.8B (thinking=False) | TBD | |
| Baseline Qwen3.5-0.8B (thinking=True) | TBD | |
| Stage 0 pass gate | TBD | |
| Stage 1 pass gate | TBD | |
| PRISM final accuracy | TBD | |

---

## Architecture Summary (quick reference)

```
Backbone: Qwen3.5-0.8B (frozen, ~0.8B params)
Insert at: layer ~18/28 (65% depth, auto-detected)

Router: [hidden_dim → 256 → 5] × 3 phases = per-phase soft weights
        Misc floor: 10% minimum, Entropy reg: 0.01 coefficient

Expert blocks: 5 domains × 3 phases = 15 blocks
  Each block: LayerNorm + 8-head Self-Attention(head_dim=64) + SwiGLU FFN(4×)
  Input/output: [B, T, hidden_dim]

CrossMix: 3 modules (one per phase)
  Each: 4-head cross-attention from each domain to all domains

Total PRISM params: ~1.5-2.0B (at hidden_dim=1536 for Qwen3.5-0.8B)
Total model: ~2.3-2.8B (< 3B target)
Inference VRAM: ~5GB
```

---

## File Structure
```
PRISM/
├── program.md           ← source of truth
├── PROGRESS.md          ← this file
├── pyproject.toml       ← pip install -e .
├── src/prism/           ← main package
├── scripts/             ← bash training scripts
├── configs/             ← YAML configs
├── results/             ← all artifacts (committed)
└── .cache/              ← HF downloads (gitignored)
```

---

## Environment
- 4× GH200 ~96GB each
- CUDA 12.9, torch 2.9, Python 3.12.13, transformers 5.5.3
- Conda env: `source /users/dasgupta/miniconda3/bin/activate /iopsstor/scratch/cscs/dasgupta/research/ideas/PRISM/.venv`
- Setup: `source scripts/setup/env.sh`
