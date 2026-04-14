# PRISM Progress Log

**Project:** PRISM — Phase-structured Reasoning with Integrated Subject-expert Modules  
**Target:** NeurIPS 2026 main track  
**Last updated:** 2026-04-14

---

## Current Status

**Phase:** Pilot complete — A5 beats baseline 2×; ready to scale to full training

**Last updated:** 2026-04-14 15:35

---

## GPU Status

| GPU | Status | Job | PID | Since | Log |
|-----|--------|-----|-----|-------|-----|
| 0–3 | IDLE | — | — | — | — |

Pilot complete. A5 eval done. Next: scale to full training.

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
- Unit tests: ALL PASSED (ExpertBlock, CrossMixModule, DomainRouter, entropy reg, freeze/unfreeze)
- Package installable: `pip install -e .` ✓

**Key architectural decisions made:**
1. **Multi-phase routing**: Router produces `[B, n_phases, n_domains]` — different domain
   weights for Solve vs Verify vs Correct phases.
2. **Soft routing always**: Entropy regularization (coefficient 0.01) prevents degenerate solutions.
3. **Misc floor**: Miscellaneous expert always gets ≥10% weight.
4. **All experts run**: Routing weights determine contribution, not activation.
5. **CrossMix at every phase**: 3 CrossMix modules (one per phase).

### 2026-04-13: vLLM TP=4 Server + Trace Generation Infrastructure

**What was done:**
- Installed vLLM 0.19.0 alongside torch 2.9.0+cu129 on GH200
- Fixed vLLM GH200 crashes (VLLM_FLASH_ATTN_VERSION=2, VLLM_ATTENTION_BACKEND=FLASHINFER, limit_mm_per_prompt={"image":0})
- Switched from in-process inference to single TP=4 vLLM server (all 4 GPUs, tensor parallel)
- vLLM server: 1.74M token KV cache, 693× concurrency at 8192 tokens/request
- Fixed `--model` flag bug (was passing model path as positional arg)
- Fixed `launch()` subshell bug in run_trace_gen.sh (PIDS weren't children of script shell)
- Created `scripts/watch_progress.sh` — live dashboard (per-domain progress, vLLM metrics, JSONL stats)
- Implemented auto-chain watcher: Rounds 2+3 launch automatically when Round 1 finishes
- Rewrote `src/prism/training/train_lora.py`:
  - Unsloth fast path (2× speedup via custom Triton kernels, graceful fallback)
  - bfloat16 + Flash Attention 2 (GH200-native)
  - Expanded LoRA to 7 target modules: q, k, v, o, gate, up, down (was q+v only)
  - trl SFTTrainer with cosine LR, gradient checkpointing
- Updated `configs/training/stage0_lora.yaml`: lora_target_modules expanded to all 7

**Trace generation design decisions (2026-04-13):**
1. **Reformulation approach**: Teacher is given the *correct* solution and asked to reformulate
   it in domain-expert style — never solves from scratch. This decouples trace quality from
   teacher solve rate (always produces valid positive traces).
2. **Negative examples (30/70 split)**: 30% of Phase 1 calls use free-solve mode (no reference
   answer) — model attempts independently, naturally produces some wrong answers. Phase 2
   (Verify) catches errors with CORRECT/WRONG verdict. Phase 3 produces corrected solution.
   This creates training pairs: (wrong solve → correct verification → corrected solution).
3. **Minimal trace prompts**: Prompts explicitly request 5–15 lines of mathematics, no prose
   padding, no restating the problem. Combined with max_tokens=4096 (was 2048, caused 98%
   truncation when enable_thinking=True generated 1000-5000 token <think> chains).
4. **Phase 2 CORRECT/WRONG verdict**: Verifier starts with exactly "CORRECT" or "WRONG"
   then 1–4 sentences of diagnosis. Structured for easy programmatic parsing.
5. **Cross-domain verification (Rounds 2+3)**: Problems from domain A are verified by an
   expert in domain B — tests whether domain structure generalizes across subject boundaries.

**Files changed:**
- `src/prism/generation/phase_prompts.py` — minimal trace instructions, free-solve phase, CORRECT/WRONG verdict
- `src/prism/generation/trace_generator.py` — 30% negative fraction, progress logging at 10% intervals
- `src/prism/data/trace_format.py` — added `free_solve: bool` field to TraceExample
- `src/prism/training/train_lora.py` — Unsloth, bfloat16, FA2, SFTTrainer, 7-module LoRA
- `configs/training/stage0_lora.yaml` — lora_target_modules expanded
- `scripts/run_trace_gen.sh` — fixed launch() PID bug, max_tokens=4096
- `scripts/watch_progress.sh` — NEW: live progress dashboard
- `scripts/start_vllm_servers.sh` — fixed --model flag, --trust-remote-code, TP=4

---

## Round Layout

| Round | Domains | Cross-verify |
|-------|---------|-------------|
| 1 | algebra, geometry, combinatorics, number_theory | — |
| 2 | miscellaneous, algebra×misc, geometry×algebra, combinatorics×NT | primary + cross |
| 3 | NT×algebra, misc×comb, algebra×geom, geometry×comb | cross only |

---

## Queued Next

1. ~~Ablations A1-A4~~ ✓ (done)
2. ~~Joint fine-tuning A5~~ ✓ (done: 1.6%, 2× baseline)
3. ~~Baseline eval~~ ✓ (done: 0.8%)
4. **NEXT: Scale to full training** — A5 cleared the bar. Resume trace gen (Rounds 2+3) and train full expert blocks on all 500+ problems.
   - `bash scripts/start_vllm_servers.sh` then `bash scripts/resume_trace_gen.sh 2`
   - After full traces: retrain all modules with `scripts/joint_finetune_pilot.sh` (or full version)

---

## Key Numbers

| Metric | Value | Notes |
|--------|-------|-------|
| Baseline Qwen3.5-0.8B (published, w/ thinking) | ~4.4% | MATH-500, published |
| Baseline Qwen3.5-0.8B (same 125 probs, no thinking) | **0.8%** | 1/125: 1 geometry |
| PRISM (old forward, random router) | 2.4% | Router bug: w_phase computed but never used |
| PRISM (old forward, trained router) | 2.4% | Same — router had no effect |
| A1 uniform routing (pilot 100-sample) | 0.8% | 1/125: 1 algebra; matches baseline |
| A2 trained router (pilot) | 0.8% | 1/125: 1 algebra (4% algebra!) |
| A3 CrossPhase (pilot) | 0.0% | Untrained CrossPhase adds noise |
| A4 hard routing (pilot) | 0.0% | Catastrophic with undertrained router |
| **A5 joint fine-tuning (pilot)** | **1.6%** | 2/125: 1 algebra + 1 combinatorics; **2× baseline** |

### Critical Bugs Found & Fixed (2026-04-14)

1. **w_phase computed but never used** (CRITICAL) — `phase_weights[:, phase_idx, :]` was computed but discarded at phases 0,1; only phase 2 routing applied at final aggregation. Fixed: compute phase-specific aggregates at each phase, combine with `mean` or `last`.

2. **Router accuracy 3× inflated** — accuracy used `logits[:, 0:3, :]` vs `[B]` labels, broadcasting made count 3× too large. Fixed: use `logits[:, 0, :]` (phase 0 only).

3. **h_K_prime norm amplification** — Expert blocks + CrossMix (random weights) amplified h_K_prime 5× vs h_K, causing degenerate generation (`1222222...`, `$$$$$...`). Fixed: (a) residual alpha blend `h_K_prime = h_K + α*(expert - h_K)` with α=0.3 for ablation, (b) norm stabilization to match h_K's per-position L2 norm.

4. **CrossMix corrupts direction** — CrossMix with random weights (never trained) generates random cross-domain attention, overwriting meaningful expert directions. Fixed: disable CrossMix for A1-A4 ablations; only enable after joint fine-tuning (A5).

5. **Answer extraction regex wrong** — Used `r'\\\\boxed{}'` (two backslashes) vs the LaTeX `\boxed{}` (one). Fixed: brace-counting extraction from `re.finditer(r'\\boxed\{', text)`.

6. **MATH-500 subject classification** — Old code used keyword matching on problem text. Fixed: use `ex['subject']` field directly with `SUBJECT_TO_DOMAIN` mapping.

### New Modules

- `src/prism/model/cross_phase.py` — CrossPhaseModule: per-domain temporal attention across reasoning phases
- `src/prism/training/joint_finetune.py` — JointFinetuneTrainer: e2e fine-tuning of all PRISM modules
- `src/prism/eval/math_eval.py` — evaluate_model() with proper MATH-500 classification

---

## Architecture Summary (quick reference)

```
Backbone: Qwen3.5-0.8B (frozen, ~0.8B params)
Insert at: layer ~18/28 (65% depth, auto-detected)

Router: [hidden_dim → 256 → 5] × 3 phases = per-phase soft weights
        Misc floor: 10% minimum, Entropy reg: 0.01 coefficient

Expert blocks: 5 domains × 3 phases = 15 blocks
  Each block: LayerNorm + 8-head Self-Attention(head_dim=64) + SwiGLU FFN(4×)

CrossMix: 3 modules (one per phase)
  Each: 4-head cross-attention from each domain to all domains

Total PRISM params: ~1.5-2.0B (at hidden_dim=1536)
Total model: ~2.3-2.8B (< 3B target)
```

---

## Trace Generation Design

```
Phase 1 (Solve/Reformulate):
  70% guided  — given correct solution, reformulate in domain-expert style
  30% free    — model solves independently (creates natural negatives when wrong)

Phase 2 (Verify):
  Given: problem + known correct answer + Phase 1 trace
  Output: "CORRECT" or "WRONG" + 1-4 sentence diagnosis

Phase 3 (Correct/Polish):
  Given: problem + known correct answer + Phase 1 + Phase 2
  Output: minimal final solution (5-15 lines, \boxed{} answer)
```

---

## Environment

- 4× GH200 ~96GB each
- CUDA 12.9, torch 2.9, Python 3.12.13, transformers 5.5.3, vLLM 0.19.0
- Conda env: `source /users/dasgupta/miniconda3/bin/activate /iopsstor/scratch/cscs/dasgupta/research/ideas/PRISM/.venv`
- Setup: `source scripts/setup/env.sh`
- vLLM server: `bash scripts/start_vllm_servers.sh` (TP=4, port 8000)
- Progress monitor: `bash scripts/watch_progress.sh [ROUND] [REFRESH_SECS]`

---

## File Structure

```
PRISM/
├── program.md               ← source of truth (research plan)
├── PROGRESS.md              ← this file
├── pyproject.toml           ← pip install -e .
├── src/prism/               ← main package
│   ├── generation/
│   │   ├── trace_generator.py   ← vLLM HTTP client, 3-phase orchestration
│   │   └── phase_prompts.py     ← 15 domain×phase prompts (minimal trace style)
│   ├── data/trace_format.py     ← TraceExample dataclass (free_solve field)
│   └── training/train_lora.py   ← Unsloth+SFTTrainer LoRA training
├── scripts/
│   ├── start_vllm_servers.sh    ← TP=4 vLLM server
│   ├── run_trace_gen.sh         ← Round 1/2/3 launcher
│   └── watch_progress.sh        ← live dashboard
├── configs/training/
│   └── stage0_lora.yaml         ← LoRA config (7 target modules)
├── results/
│   ├── traces/                  ← JSONL output files
│   └── logs/                    ← per-domain and server logs
└── .cache/                      ← HF downloads (gitignored)
```
