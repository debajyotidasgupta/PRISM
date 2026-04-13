---
name: PRISM Motivation and Research Questions
description: Why this project exists, what problem it solves, what needs to be investigated
type: project
---

## The Problem

Small vision-language models (under 3B) fail badly on Olympiad mathematics. A Qwen3.5-0.8B model with built-in thinking gets roughly 8–15% on OlympiadBench — a benchmark where a human olympiad participant would score 60–80%. Meanwhile, Qwen3.5-7B (9× larger) scores 30–45%. The gap is large and the cost is real: deploying a 7B model requires 15–20GB VRAM vs 2GB for a 0.8B model.

The core question: **can we close most of this gap without increasing inference-time cost?**

## Why Existing Approaches Fall Short

**More data / SFT:** Models like Qwen2.5-Math and DeepSeekMath show that domain-specific pretraining helps, but the gains plateau at 1-2B scale. You can't SFT your way to olympiad-level reasoning in a 0.8B model.

**Multi-step prompting (solve → verify → correct):** Works, but requires 3× inference time. At test time, you need to make 3 API calls. The latency is a dealbreaker for real deployment.

**Chain-of-thought / longer thinking:** The built-in thinking mode (enable_thinking=True) already does this. Additional CoT prompting doesn't add much beyond what the model already does.

**MoE (Mixture of Experts):** Standard MoE routes tokens to general-purpose expert FFNs within one layer. The experts are not domain-specialized. They learn to be different from each other, not to specialize in Algebra vs. Geometry.

## The PRISM Hypothesis

Hard olympiad problems require fundamentally different reasoning patterns depending on domain:
- Algebra: symbolic manipulation, polynomial structure, functional equations
- Geometry: spatial relationships, projective tools, circle theorems
- Combinatorics: enumeration, bijection, graph structure
- Number Theory: prime structure, modular arithmetic, p-adic analysis
- Miscellaneous: cross-domain tools (AM-GM, generating functions, calculus)

A model that has dedicated neural pathways for each domain — trained specifically on how a domain expert approaches each reasoning phase — should outperform a model that uses a single averaged pathway.

And if those pathways are **embedded in the architecture** (not prompted), the three-phase structure (solve/verify/correct) is executed at zero additional inference cost.

## What Needs to Be Investigated

**Before building the full architecture:**

1. **Does domain-specific fine-tuning actually help?** (Stage 0 validation)
   - Train 5 domain LoRA adapters on 2,500 teacher traces each
   - If the algebra LoRA doesn't beat a general LoRA on algebra problems, the core hypothesis is wrong

2. **Do teacher-generated expert traces differ meaningfully from existing CoT traces?**
   - Qualitative check: read 20 traces per domain and verify they use domain-specific techniques
   - Quantitative check: ablation A9 (generic traces vs. expert-aligned traces)

3. **Where in the backbone should blocks be inserted?**
   - Activation analysis: which layers fire most differently for thinking vs. non-thinking mode?
   - Likely answer: ~60-70% depth, but needs empirical confirmation

4. **Does the Miscellaneous expert need to be in Stage 1, or can it be added later?**
   - Almost every hard problem uses cross-domain tools, so probably needs to be in Stage 1

5. **Does PRISM compose well with enable_thinking=True?**
   - Can PRISM improve the quality of the think blocks, or does it conflict?
   - Ablation A10

6. **What is the right supervision signal for the Verify phase (Level 2)?**
   - The verification output is a diagnosis, not a solution. Is next-token prediction the right loss?
   - Alternative: binary classification (correct/incorrect) + diagnosis generation

7. **Can the 15 expert blocks be trained in parallel on 4 GPUs?**
   - Implementation question: train Algebra+Geometry on GPU0+1, Combinatorics+NT on GPU2, Misc on GPU3 simultaneously
   - Risk: gradient instability if backbone forward passes are shared

## What a Successful Result Looks Like

Minimum: PRISM (<3B) outperforms Qwen3.5-0.8B thinking=True and Qwen2.5-VL-3B on OlympiadBench.

Strong: PRISM achieves ≥70% of Qwen3.5-7B accuracy on OlympiadBench.

Excellent: PRISM approaches Qwen3.5-7B on OlympiadBench while using 8× less memory, with ablations proving that both domain specialization and phase structure contribute independently.
