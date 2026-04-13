---
name: PRISM Project Overview
description: Core facts about what PRISM is, what it builds, and what it claims
type: project
---

**PRISM** — Phase-structured Reasoning with Integrated Subject-expert Modules

**One-line claim:** A Qwen3.5-0.8B VLM augmented with PRISM expert blocks achieves competitive performance with 7B+ models on Olympiad math, using <3B parameters and a single forward pass at inference.

**What is being built:**
- 5 domain-expert reasoning blocks (Algebra, Geometry, Combinatorics, Number Theory, Miscellaneous) × 3 phases (Solve / Verify / Correct) = 15 new transformer blocks
- These blocks are inserted into the frozen Qwen3.5-0.8B backbone at a single insertion point (~layer 60-70%)
- A 5-class domain router (soft weights) selects how to blend expert outputs
- Cross-domain mixing (lightweight cross-attention) between experts at each level
- Total model: ~1.55–2.0B params, inference in <5GB VRAM

**Key differentiator from AMSD:**
- AMSD: LoRA adapters at inference time, explicit multi-step prompting, ARC-AGI domain
- PRISM: embedded architectural blocks, single forward pass, Olympiad math domain

**Teacher model for trace generation:** Qwen2.5-VL-72B or Qwen3.5-VL-72B (largest available VL model)
**Student model:** Qwen3.5-0.8B (has image+text input, enable_thinking toggle)

**Test benchmarks (held out — never train on these):**
- OlymMATH (en-hard) — primary
- OlympiadBench (test_en) — primary
- Omni-MATH — secondary

**Source of truth:** `/iopsstor/scratch/cscs/dasgupta/research/ideas/PRISM/program.md`

**Status as of 2026-04-13:** Plan complete. Not yet started implementation. Stage 0 (LoRA hypothesis validation) is the first concrete step.

**Why:** AMSD showed +21% gains from expert diversity on ARC, but ARC is a niche domain with low absolute numbers. PRISM tests the same core idea (domain specialization improves small models) on Olympiad math — a domain with clean exact-match evaluation, large training corpora, and clear impact.
