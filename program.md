# Program: PRISM — Phase-structured Reasoning with Integrated Subject-expert Modules

## 0. Document purpose

This document is the master execution plan for a research engineer building a NeurIPS-targeted project on **Olympiad-level mathematical reasoning in small vision-language models**. The project targets models under 3B parameters and focuses exclusively on mathematical reasoning across five expert domains: Algebra, Geometry, Combinatorics, Number Theory, and Miscellaneous (cross-domain tools).

This file is the source of truth for:

- the scientific question and core hypotheses
- the prior art and how this work differs
- the novel architecture and its motivation
- the expert trace generation methodology
- the dataset plan (what to use for training vs. held-out testing)
- the staged validation procedure before full training
- the full training procedure
- the evaluation plan and benchmarks
- the codebase structure and implementation details
- the compute plan on 4× GH200 GPUs (~96GB usable each)
- the experiment schedule and success criteria
- the deliverables for a NeurIPS submission

This is a research execution document. Every implementation phase must preserve the scientific narrative, produce interpretable evidence, and leave reusable code and clean artifacts. Do not add features beyond what is described. Do not skip validation stages.

## Operational mandates (non-negotiable)

**GPU utilization:** All 4 GPUs must be kept busy at all times. ~96GB VRAM per GPU should be used as fully as possible. The default training mode is parallel: train different expert blocks on different GPUs simultaneously. A GPU sitting idle is wasted compute. When one job finishes, the next must be queued immediately.

**Parallelism:** Any work that can be done in parallel must be. Training 5 domain experts? Run them on 4 GPUs in parallel (pair two on one GPU if needed). Generating traces? Use all available GPUs for the teacher. Evaluating ablations? Distribute across GPUs.

**PROGRESS.md:** A `PROGRESS.md` file must be maintained in the project root at all times. It must be updated after every experiment and committed to git. It records: what is currently running (which GPU, what job, since when), what has completed (with key numbers), and what is queued next. A new research engineer picking up the project should be able to read PROGRESS.md and know exactly what state everything is in within 5 minutes.

**Professional packaging:** The project must be installable by anyone with a single `pip install -e .` (development) or `pip install prism-math` (release). The PRISMModel must be usable directly with HuggingFace AutoModel, AutoProcessor, and AutoConfig — no custom loading code required by downstream users. A researcher anywhere in the world should be able to run:
```python
from transformers import AutoModel, AutoProcessor
model = AutoModel.from_pretrained("debajyoti/prism-0.8b", trust_remote_code=True)
```
and get a fully functional PRISM model. This is a requirement from day one, not an afterthought.

**Storage policy:**
- **Model weights at runtime** → `/tmp/prism_models/` only. The machine has >1TB RAM. `/tmp` is RAM-backed. Models load from `/tmp` and transfer directly to GPU — fastest possible path. Model weights are never committed to git.
- **HuggingFace download cache** → `PRISM_ROOT/.cache/` (inside the project repo). This persists downloaded model files across runs. It is gitignored. On a fresh machine, the first run downloads to `.cache/`, then loads from `/tmp/`.
- **All artifacts** (traces, adapter weights, eval results, router checkpoints, configs, logs) → `PRISM_ROOT/results/` (inside the project repo). These are committed to git (except large binary files >100MB which go to Git LFS or HF Hub).
- **No model weights on disk** outside of `.cache/` and `/tmp/`. Never save a `pytorch_model.bin` or `model.safetensors` into `results/` or the repo root.

**Environment variables (set in `scripts/setup/env.sh`):**
```bash
export PRISM_ROOT=/iopsstor/scratch/cscs/dasgupta/research/ideas/PRISM
export HF_HOME=${PRISM_ROOT}/.cache/huggingface     # persistent download cache
export HF_HUB_CACHE=${PRISM_ROOT}/.cache/huggingface/hub
export PRISM_MODEL_DIR=/tmp/prism_models             # runtime model loading (RAM)
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
```

Startup routine (in every training/eval script):
```python
import os, shutil
from pathlib import Path

# Copy model from persistent cache to /tmp for fast loading
prism_root = Path(os.environ["PRISM_ROOT"])
model_dir = Path("/tmp/prism_models")
model_dir.mkdir(exist_ok=True)

def prepare_model(model_name: str) -> str:
    """Copy model from .cache to /tmp for fast GPU loading. Returns /tmp path."""
    cache_path = prism_root / ".cache" / "models" / model_name.replace("/", "--")
    tmp_path = model_dir / model_name.replace("/", "--")
    if not tmp_path.exists() and cache_path.exists():
        shutil.copytree(cache_path, tmp_path)
    return str(tmp_path) if tmp_path.exists() else model_name
```

---

# 1. Project Summary

## 1.1 Project title

**PRISM: Phase-structured Reasoning with Integrated Subject-expert Modules for Olympiad Mathematics**

Working subtitle: *Embedding domain-expert reasoning structure into sub-3B vision-language models without inference-time scaling*

## 1.2 One-sentence thesis

Small vision-language models can be made competitive with models 10–40× larger on olympiad mathematics by **permanently embedding domain-specific, phase-structured reasoning pathways into the model's architecture**, trained via teacher-generated expert traces, without increasing inference-time compute.

## 1.3 Core target

A **NeurIPS 2026** main-track submission. This requires:

- a clean, one-line scientific contribution
- a novel architecture with clear ablations
- competitive results on held-out hard benchmarks
- results that hold across multiple olympiad domains
- a dataset contribution (expert-aligned traces not previously released)

## 1.4 The concrete claim we will defend

> A Qwen3.5-0.8B vision-language model, augmented with PRISM expert blocks trained on domain-aligned teacher traces, achieves performance competitive with Qwen2.5-VL-7B or Qwen3.5-7B on OlymMATH (en-hard) and OlympiadBench, using a model that is under 3B parameters and requires no multi-step prompting or inference-time scaling at test time.

The key tension this paper resolves: **how do you give a small model access to deep, domain-specific mathematical reasoning without paying the inference cost of a large model or the latency cost of multi-step prompting?** The answer is to bake the reasoning structure into the weights.

---

# 2. Research Questions and Hypotheses

## 2.1 Main research question

**Can permanently embedding domain-specialized, phase-structured reasoning pathways into a small VLM's architecture (under 3B parameters) substitute for the inference-time reasoning budget of a much larger model on olympiad mathematics?**

## 2.2 Sub-questions

### Q1 — Domain specialization
Does training separate expert reasoning pathways per mathematical domain (Algebra, Geometry, Combinatorics, Number Theory) improve over a single general-purpose reasoning pathway, even at the same total parameter budget?

### Q2 — Phase structure
Does enforcing a 3-phase structure (solve → verify → correct) as an architectural constraint improve over a single-pass reasoning pathway, even when the three phases together use the same total parameter budget?

### Q3 — Expert trace quality
Do domain-aligned expert traces generated by a large teacher model (with domain-specific system prompts) produce better downstream models than generic CoT traces from existing datasets?

### Q4 — Expert cross-networking
Does allowing expert blocks to exchange information across domains (cross-attention between domain pathways at each level) improve over independent domain pathways?

### Q5 — Reasoning mode interaction
Does the PRISM architecture interact differently with Qwen3.5's built-in thinking mode (`enable_thinking=True`) vs. its non-thinking mode? Can we identify which attention heads are activated differently in thinking vs. non-thinking mode, and do PRISM blocks improve both modes or only one?

### Q6 — Efficiency
At equal inference time, does PRISM (small model + embedded reasoning structure) outperform a larger model with inference-time multi-step prompting (solve → verify → correct as separate API calls)?

## 2.3 Hypotheses

### H1 — Domain specialization helps
Different olympiad domains require fundamentally different reasoning patterns. Algebra requires symbolic manipulation; Geometry requires spatial and visual reasoning; Combinatorics requires case enumeration and invariant detection; Number Theory requires divisibility chains and modular reasoning. A model that has separate neural pathways for each domain will outperform one forced to use a single pathway.

### H2 — Phase structure is a learnable architectural constraint
The solve → verify → correct loop is not arbitrary; it corresponds to the actual cognitive phases that expert mathematicians follow. Embedding this structure as architectural levels (not as prompting) ensures the model always executes this loop and cannot collapse to single-pass reasoning under pressure.

### H3 — Expert traces matter more than volume
Existing math CoT datasets (NuminaMath, MATH) contain generic reasoning traces that are not aligned to any expert domain perspective. A smaller number of domain-aligned traces generated by a VL teacher model, with explicit domain system prompts and phase separation, will outperform training on a much larger set of generic traces.

### H4 — Cross-networking is beneficial across domains
Geometry problems often require algebraic manipulation (Cartesian coordinates, trigonometry). Combinatorics problems often require number-theoretic tools (GCD, modular arithmetic). Mixed-domain problems (the majority of hard olympiad problems) require the Miscellaneous expert's tools (inequalities, generating functions, calculus). Cross-attention between domain pathways at intermediate levels will capture these inter-domain dependencies and improve accuracy on mixed-domain problems.

### H5 — The model stays competitive under visual input
Olympiad math problems frequently include diagrams, coordinate figures, and geometric constructions. Because the base model is a vision-language model (Qwen3.5-0.8B with image+text input), the PRISM blocks will naturally learn to condition on visual features, unlike text-only approaches.

---

# 3. Why This is a NeurIPS Contribution

## 3.1 What is new

The key novelty is the combination of:

1. **Architectural embedding of phase-structured reasoning** — not prompting, not LoRA adapters on top of the model, but actual new transformer blocks inserted at fixed positions in the network, each corresponding to a domain and reasoning phase.

2. **Per-expert path training with frozen co-paths** — each domain's reasoning path is trained independently while all other paths and the backbone remain frozen. This prevents interference between domains and produces clean expert specialization.

3. **Expert-aligned trace generation** — existing math datasets do not contain expert-aligned traces. We generate a new dataset using a large VL teacher model, with domain-specific prompting, explicitly structured into 3 phases. This dataset is itself a contribution.

4. **Sub-3B VLM with no inference-time scaling** — at test time, the model makes a single forward pass. There is no chain-of-thought prompting, no verify-then-correct multi-step prompting, and no multi-agent debate. The reasoning structure is entirely in the weights.

## 3.2 What this is not

This project is **not**:
- a prompting paper (no system prompt tricks)
- a multi-agent debate paper
- a RLHF / RLVR paper (though this could be Phase 2 future work)
- a benchmark-specific hack
- a distillation-only paper (the architecture is the contribution, not just the distillation)

## 3.3 Positioning

The closest related work is MoE (Mixture of Experts), but PRISM differs in a critical way: **standard MoE routes tokens to expert FFN blocks within one layer, and all experts are general-purpose**. PRISM routes the entire hidden state sequence to domain-specialized reasoning layers that span multiple transformer levels, and the routing is not learned from LM loss alone but supervised by domain-aligned teacher traces.

---

# 4. Literature Review

## 4.1 Small models for mathematical reasoning

**WizardMath (Luo et al., 2023)** fine-tunes Llama-7B on augmented math data using an RLEIF (Reinforcement Learning from Evol-Instruct Feedback) framework, achieving strong results on MATH and GSM8K. Shows that fine-tuning on high-quality, diverse math traces helps significantly. Limitation: single-domain (math only), text-only, no domain decomposition.

**DeepSeekMath (Shao et al., 2024)** trains a 7B model on 120B math tokens with GRPO (group relative policy optimization). Achieves strong MATH benchmark results. Key finding: data quality and domain-specific pretraining matter more than scale up to 7B. Limitation: requires massive compute for pretraining.

**Qwen2.5-Math (Team, 2024)** achieves state-of-the-art on MATH-500 at 1.5B scale using a two-stage approach: domain-specific pretraining then instruction fine-tuning. Key finding: small models can compete with larger models on math when given enough high-quality math-specific data. Directly relevant as a baseline comparison.

**MathCoder (Wang et al., 2023)** and **ToRA (Gou et al., 2023)** use code execution as an intermediate tool to ground mathematical reasoning. Shows that structured intermediate outputs (code, not free-form text) improve accuracy. Relevant to our phase-structure idea: structured reasoning phases outperform unstructured ones.

**Key gap:** all of these train a single model with a single general-purpose reasoning style. None decompose the model's internal computation by mathematical domain.

## 4.2 Multi-phase reasoning: solve → verify → correct

**Self-Consistency (Wang et al., 2022)** generates multiple reasoning chains and selects the most common answer. Effective but requires inference-time scaling (N forward passes).

**Self-RAG (Asai et al., 2023)** adds reflection tokens inline with generation to control when the model should retrieval-augment or revise its output. Related to our idea of phase-structured generation. Key insight: encoding *when to do what* as architectural structure rather than prompting is more robust.

**Self-Refine (Madaan et al., 2023)** iteratively refines outputs using the model itself as verifier. Requires multiple API calls. Our approach embeds the verify and correct phases as weight-level structures with zero additional inference cost.

**RLVR with process rewards (Lightman et al., 2023 — Let's Verify Step by Step)** trains a separate process reward model (PRM) to score each reasoning step. Improves MATH accuracy significantly. Relevant because our "verify" phase serves a similar purpose but is embedded architecturally rather than requiring a separate reward model at inference time.

**Math-Shepherd (Wang et al., 2024)** automatically annotates intermediate reasoning steps with process rewards. Useful for training data creation but not architecturally embedded.

**Key gap:** all multi-phase approaches require either multiple forward passes or a separate reward model. PRISM embeds all three phases in one forward pass.

## 4.3 Mixture of Experts and domain routing

**GShard / Switch Transformer (Lepikhin et al., 2021; Fedus et al., 2022)** introduce token-level MoE with learned routing. Scales efficiently but all experts are general-purpose; no domain specialization.

**Mixtral-8x7B (Jiang et al., 2024)** shows that sparse MoE with 8 general-purpose experts achieves strong results at lower active parameter cost. The key insight: not all expert capacity is needed for every token. Relevant for efficiency but again experts are general-purpose.

**Branch-Train-Merge (Li et al., 2022)** trains separate expert LLMs on different domains (code, math, law) then merges them. Most directly related to our approach — domain specialization helps. Key difference from PRISM: BTM trains separate full models and merges them, while PRISM embeds the domain specialization as additional blocks within a shared backbone.

**Outrageously Large Neural Networks (Shazeer et al., 2017)** — original MoE paper with learned sparse gating. Our routing is not fully soft but conditioned on domain detection from the problem statement.

**Key gap:** none of these apply domain-specific MoE routing to the 3-phase reasoning structure for olympiad math specifically. The domain labels in PRISM are explicit (Algebra/Geometry/Combinatorics/Number Theory), not learned from LM loss.

## 4.4 Knowledge distillation for reasoning

**Orca (Mukherjee et al., 2023)** distills GPT-4's reasoning by having it explain its reasoning traces in detail, then trains smaller models on those explanations. Shows that explanation quality matters more than explanation quantity.

**Distilling Step-by-Step (Hsieh et al., 2023)** shows that training small models on teacher rationales (not just teacher answers) significantly outperforms standard fine-tuning. Directly relevant: we distill teacher reasoning traces, not just teacher answers.

**Phi-1 / Phi-2 (Gunasekar et al., 2023; Li et al., 2023)** show that carefully curated synthetic training data from a strong teacher can produce small models that punch well above their weight. Key insight: data quality from a teacher model matters enormously for small model performance.

**MiniLLM (Gu et al., 2024)** minimizes reverse KL divergence for distillation rather than forward KL, showing that sequence-level distillation is better than token-level cross-entropy for reasoning tasks.

**Key gap for PRISM:** none of these distillation works produce expert-aligned, domain-separated, phase-structured traces. They all distill generic reasoning into a single student pathway. PRISM's trace generation is specifically designed to produce traces that correspond to how a domain expert would approach each phase.

## 4.5 Vision-language models for mathematical reasoning

**G-LLaVA (Gao et al., 2023)** and **Math-LLaVA (Shi et al., 2024)** fine-tune VLMs on geometry problem datasets with diagrams. Show that VLMs can solve visual geometry problems. Limitation: geometry-only, single-domain, single-pass.

**InternLM-Math / InternLM-XComposer (2023, 2024)** integrate math reasoning with vision. Competitive on multimodal math benchmarks (MathVista, GeoQA, etc.).

**LLaVA-R1 / Multimodal Chain-of-Thought Reasoning (2024)** attempts to apply multi-step reasoning to VLMs for math problems. Shows that VLMs benefit from structured reasoning but that visual grounding of the reasoning chain is hard.

**Qwen2.5-VL (Team, 2024)** achieves strong multimodal reasoning. The 7B variant is particularly strong on mathematical reasoning with diagrams.

**Key gap:** no prior work has embedded domain-specialized multi-phase reasoning *architecturally* into a small VLM (under 3B) for olympiad math specifically. All existing approaches use a single reasoning pathway at inference time.

## 4.6 Mathematical domains and benchmark landscape

**AMC/AIME problems** span all four olympiad domains. AIME requires integer answers (0–999), making exact match evaluation clean.

**MATH (Hendrycks et al., 2021)** — 12,500 competition problems with solutions, categorized into 7 types (Algebra, Geometry, Counting and Probability, Number Theory, Pre-calculus, etc.). Clean domain labels. Level 1–5 difficulty. Key training/evaluation resource.

**OlympiadBench (He et al., 2024)** — 2,126 bilingual competition problems from IMO, CMO, USAMO-level competitions. Has domain labels and bilingual (English/Chinese) support. **This is a primary test set — do not use for training.**

**OlymMATH (RUC-AIBOX)** — Very hard olympiad problems. `en-hard` subset is particularly challenging. **This is a primary test set — do not use for training.**

**Omni-MATH (Gao et al., 2024)** — 4,428 competition problems across domains and difficulty levels. **Hold out as secondary test set.**

**MATH-500** — 500-problem subset of MATH, widely used as evaluation. Level 4-5 problems as hard subset. **Can use as validation set (not for training domain router).**

---

# 5. Olympiad Domain Taxonomy and Topic Coverage

This section defines exactly what each of the five PRISM experts is responsible for. The taxonomy is based on the standard olympiad competition syllabus (following Evan Chen's *Unofficial Syllabus for Math Olympiads*, 2023), adapted to five expert domains that together cover all competition mathematics at the IMO/USAMO level.

The five domains are:

1. **Algebra** — symbolic and functional reasoning
2. **Combinatorics** — counting, structure, and discrete reasoning
3. **Geometry** — spatial, metric, and projective reasoning
4. **Number Theory** — divisibility, modular arithmetic, and prime structure
5. **Miscellaneous** — cross-domain tools: inequalities, calculus, generating functions, linear algebra, complex analysis, probability

The fifth domain is not a catch-all for difficult problems. It is a genuine expert domain covering the *tools* that appear across all four core domains: an olympiad problem may be classified as Geometry but require AM-GM to bound a ratio, or classified as Combinatorics but require generating functions. The Miscellaneous expert holds these cross-cutting tools.

---

## 5.1 Algebra expert scope

The Algebra expert handles problems whose primary structure involves symbolic manipulation of equations, inequalities, or functions, and where the central challenge is algebraic rather than spatial or combinatorial.

**Core topics (standard olympiad knowledge):**
- Elementary manipulation of algebraic equations, expressions, and inequalities — factoring, expanding, substitution, completing the square
- Σ (summation) and Π (product) notation and their manipulation
- Definitions of functions: injectivity, surjectivity, bijectivity, composition, inverses
- Cauchy's functional equation `f(x+y) = f(x) + f(y)` and standard solution techniques
- Polynomials in one variable: roots, coefficients, Vieta's formulas, the Fundamental Theorem of Algebra, polynomial division, remainder theorem
- Polynomials in two or more variables: symmetric polynomials, Newton's identities, homogeneous polynomials

**Advanced topics (appear occasionally in hard problems):**
- Complex numbers and their algebraic properties: modulus, argument, roots of unity, de Moivre's theorem
- Trigonometric identities and their algebraic manipulation (sum-to-product, product-to-sum)
- Relations between complex numbers and trigonometric functions (Euler's formula in algebraic context)

**Trace generation emphasis:** When generating traces for the Algebra expert, the teacher should be prompted to use algebraic manipulation as the primary tool — substitution, factorization, Vieta jumping, polynomial identities, functional equation techniques. The solution should not default to coordinates or counting arguments.

---

## 5.2 Combinatorics expert scope

The Combinatorics expert handles problems whose primary structure is discrete: counting configurations, finding invariants, proving existence of combinatorial structures, or establishing recursive relationships.

**Core topics (standard olympiad knowledge):**
- Basic counting arguments: expressions like 2^n, n!, C(n,k), stars and bars, bijective proofs
- Principle of mathematical induction: weak, strong, and structural induction
- Recursion and recurrence relations: solving linear recurrences, finding patterns
- The pigeonhole principle and its generalizations
- Definitions of sets and functions in a combinatorial context

**Intermediate topics (common in olympiad problems):**
- Elementary probability: sample spaces, equally likely outcomes, basic rules
- Expected value and linearity of expectation (extremely powerful tool)
- Basic graph theory: vertices, edges, paths, cycles, connectedness, degree sequences, trees, bipartite graphs
- Definition and existence of the convex hull of a finite point set

**Advanced topics (appear in harder problems):**
- Nontrivial graph theory: Hall's Marriage Lemma, Turán's theorem, Ramsey theory, extremal graph theory
- Double counting and algebraic combinatorics techniques

**Trace generation emphasis:** The teacher should be prompted to use combinatorial thinking — count two ways, find a bijection, apply pigeonhole, set up a recursion, use expected value linearity. The solution should not use algebra or calculus as the primary tool unless the problem is genuinely hybrid.

---

## 5.3 Geometry expert scope

The Geometry expert handles problems whose primary structure is spatial: proving relationships between geometric figures, computing lengths or angles, or establishing collinearity, concyclicity, and similar properties.

**Core topics (standard olympiad knowledge):**
- Definitions and basic properties of triangle centers: incenter (I), centroid (G), orthocenter (H), circumcenter (O), excenters (I_A, I_B, I_C), nine-point center (N), and the nine-point circle
- Angle chasing in general and in cyclic quadrilaterals (inscribed angle theorem, Ptolemy's theorem)
- Similar triangles: AA, SAS, SSS criteria; ratio properties
- Power of a point with respect to a circle; radical axis and radical center
- Homothety: definition, center, ratio, composition
- Ceva's theorem and Menelaus's theorem (statements and applications)

**Intermediate topics (common in olympiad problems):**
- Trigonometry in geometry: law of sines, law of cosines, extended law of sines, trigonometric form of Ceva
- Coordinate geometry: Cartesian coordinates, complex number coordinates, barycentric coordinates — used as tools to verify or compute
- Inversive geometry: inversion in a circle, properties under inversion, standard applications
- Projective geometry: cross-ratios, harmonic bundles and conjugates, poles and polars, Pascal's theorem, Brianchon's theorem, perspectivity
- Spiral similarity: definition, fixed points, composition

**Advanced topics (appear occasionally):**
- Definitions and basic properties of conic sections: ellipse, hyperbola, parabola, focus-directrix properties, reflective properties

**Trace generation emphasis:** The teacher should be prompted to use synthetic geometry as the primary approach — angle chasing, circle theorems, similarity — with coordinates or trigonometry only as secondary tools for computation. The solution should identify the key geometric insight (e.g., "these four points are concyclic") before computing.

---

## 5.4 Number Theory expert scope

The Number Theory expert handles problems whose primary structure involves integers, divisibility, prime numbers, or modular arithmetic.

**Core topics (standard olympiad knowledge):**
- Basic results about primes: infinitude, fundamental theorem of arithmetic (unique prime factorization), distribution of primes
- Modular arithmetic: congruences, residue classes, arithmetic operations mod n, Fermat's Little Theorem (a^(p-1) ≡ 1 mod p for prime p, gcd(a,p)=1), Euler's theorem (a^φ(n) ≡ 1 mod n), Chinese Remainder Theorem, modular inverses, multiplicative orders
- p-adic valuation (ν_p): ν_p(xy) = ν_p(x) + ν_p(y), ν_p(x+y) ≥ min{ν_p(x), ν_p(y)}, lifting the exponent lemma (LTE)
- Euclidean algorithm: gcd(a,b) = gcd(a-b, b), extended Euclidean algorithm, Bezout's identity

**Intermediate topics (common in olympiad problems):**
- Dirichlet's theorem on primes in arithmetic progressions (statement; proof not expected)
- Fermat's theorem on integers representable as sums of two squares (p = a²+b² iff p=2 or p≡1 mod 4)

**Advanced topics (appear in harder problems):**
- Quadratic residues: definition, Legendre symbol, Euler's criterion
- Quadratic reciprocity: statement and standard applications

**Trace generation emphasis:** The teacher should be prompted to think in terms of primes, divisibility chains, and modular structure. The solution should identify the key number-theoretic tool (e.g., "take ν_p of both sides", "work mod p²", "apply CRT") before computing. Avoid algebraic substitution as the first move.

---

## 5.5 Miscellaneous expert scope

The Miscellaneous expert does **not** handle any single category of olympiad problems. Instead, it holds the *cross-domain tools* that appear as sub-steps within problems from all other domains. Almost every hard olympiad problem requires at least one tool from this domain — an inequality to bound an expression, a generating function to extract a coefficient, a derivative to find a maximum.

This expert is the "glue" of the PRISM system. During soft routing, a problem classified primarily as Geometry may receive 15-20% weight from the Miscellaneous expert when the solution requires AM-GM to bound a ratio or calculus to find an extremum.

**Inequality tools:**
- AM-GM inequality and its generalizations (weighted AM-GM, AM-GM-HM)
- Cauchy-Schwarz inequality (Engel/Sedrakyan form, sum form)
- Hölder's inequality and its applications to symmetric function bounds
- Jensen's inequality for convex/concave functions and its applications
- Power Mean inequality, Chebyshev's sum inequality, Schur's inequality
- Muirhead's inequality and symmetric function theory
- Substitution strategies for symmetric inequalities (SOS, pqr method)

**Generating functions:**
- Ordinary generating functions (OGF): definition, convolution, extracting coefficients
- Exponential generating functions (EGF): definition, composition, applications to labeled structures
- Applications to solving recurrences and counting problems
- Formal power series manipulations

**Calculus tools (as they appear in olympiad proofs):**
- Differentiation: product rule, chain rule, finding extrema, proving convexity/concavity
- Integration: computing areas, volumes, definite integrals by substitution
- Limits and continuity: ε-δ definition (for olympiad-level analysis arguments)
- Taylor series: key expansions (e^x, sin x, ln(1+x)) for estimation
- Intermediate value theorem and mean value theorem as proof tools

**Linear algebra tools:**
- Matrices and determinants: definition, computation, properties
- Determinant as a multilinear function, Cramer's rule
- Eigenvalues and eigenvectors (olympiad-level applications only)
- Use of matrices to encode combinatorial structures (adjacency matrix, transition matrix)

**Probability tools (advanced):**
- Conditional probability and Bayes' theorem
- Indicator random variables and linearity of expectation (detailed applications)
- Variance and standard deviation (for bounding arguments)

**Complex analysis tools:**
- Roots of unity filter: extracting every k-th term of a power series
- Using |z|=1 and geometric properties of complex roots for combinatorial counting
- Argument principle and Rouché's theorem (for advanced polynomial problems)

**Trace generation emphasis:** The Miscellaneous expert is trained on problems that *require* one of the above tools as the key step, even if the problem is superficially in another domain. For example, a geometry problem whose proof requires AM-GM as the critical step is a training example for the Miscellaneous expert. When generating traces, the teacher should be prompted to explicitly identify the cross-domain tool being used and explain why it is the right tool for this context.

---

## 5.6 Domain labeling for training data

NuminaMath-CoT and MATH have existing domain labels. The mapping to our 5 domains is:

| Original label | PRISM domain |
|---------------|-------------|
| Algebra | Algebra |
| Intermediate Algebra | Algebra |
| Pre-calculus | Algebra (with some Misc) |
| Geometry | Geometry |
| Number Theory | Number Theory |
| Counting and Probability | Combinatorics |
| Statistics (rare) | Miscellaneous |

Problems that cannot be clearly assigned to one primary domain should be labeled Miscellaneous. Problems requiring a key inequality or generating function step that is the main challenge should also be labeled Miscellaneous, even if the surface domain is different.

The expected domain distribution in NuminaMath-CoT (approximate):
- Algebra: ~35%
- Combinatorics: ~20%
- Geometry: ~15%
- Number Theory: ~15%
- Miscellaneous (cross-domain): ~15%

---

# 6. The Base Model: Qwen3.5-0.8B

## 6.1 Model properties

The student model is **Qwen3.5-0.8B**, a vision-language model that:

- accepts both image and text as input (full VL capability, not text-only)
- has a built-in reasoning toggle: `enable_thinking=True/False` in `apply_chat_template`
- is approximately 0.8B parameters in its base form
- is part of the Qwen3.5 series which extends to 7B, 32B+ (enabling comparison with same-family larger models)

The choice of this specific model is deliberate:

1. **Sub-1B base**: leaves 2.2B parameters of budget for PRISM blocks while staying under 3B total
2. **Built-in reasoning toggle**: allows direct comparison between thinking mode and non-thinking mode, both as baselines and as ablations
3. **VL capability**: olympiad math problems frequently include geometric diagrams, coordinate figures, and tables
4. **Same family as potential teacher model**: Qwen3.5-72B or Qwen3.5-7B as teachers ensures vocabulary and representation alignment

## 6.2 Reasoning mode investigation (pre-project analysis)

Before building PRISM blocks, we should understand what the thinking mode already does:

**Step 1**: Run OlymMATH (en-hard, n=100) with `enable_thinking=True` vs `enable_thinking=False`. Record:
- Accuracy (exact match and partial credit)
- Average tokens in `<think>` block vs answer block
- Which problem types benefit most from thinking

**Step 2**: Using activation analysis, identify which attention heads and MLP neurons fire differently in thinking vs. non-thinking mode. This reveals which internal structures are responsible for the reasoning capability.

**Step 3**: Use this information to decide where to *insert* PRISM blocks — specifically, we want to insert them in the layers that are most active during mathematical reasoning in thinking mode.

This analysis should take 1–2 days and will inform the architecture insertion points.

## 6.3 Reasoning mode baseline numbers

Expected baseline performance (before PRISM, based on model family patterns):

| Mode | OlymMATH (en-hard) | OlympiadBench | Notes |
|------|-------------------|---------------|-------|
| Qwen3.5-0.8B, thinking=False | ~3-8% | ~5-10% | No reasoning |
| Qwen3.5-0.8B, thinking=True | ~8-15% | ~10-18% | Built-in reasoning |
| Qwen3.5-7B, thinking=True | ~25-40% | ~30-45% | 9× larger reference |

If PRISM can close 50-70% of the gap between 0.8B and 7B, that is a strong paper result.

---

# 7. Architecture: PRISM Expert Blocks

## 7.1 Overview

The PRISM architecture inserts **N_domains × N_phases** reasoning blocks into the frozen Qwen3.5-0.8B backbone. These blocks operate on the backbone's hidden states and refine them before the final output is generated by the frozen LM head.

```
┌─────────────────────────────────────────────────────────────┐
│                   QWEN3.5-0.8B BACKBONE (FROZEN)            │
│                                                             │
│  Embed → Layer 0 → Layer 1 → ... → Layer K (mid) → ... → LM Head
│                                        │                    │
│                              ┌─────────┴──────────┐         │
│                              │   PRISM INSERTION  │         │
│                              └─────────┬──────────┘         │
│                                        ↓                    │
└───────────────────────── Hidden state h_K ─────────────────┘
                                         │
                            ┌────────────┴────────────┐
                            │         DOMAIN ROUTER              │
                            │      (lightweight classifier)       │
                            │  Alg | Geo | Comb | NT | Misc       │
                            └────────────────┬───────────────────┘
                                             │ domain weight vector w ∈ Δ^5
          ┌──────────────┬──────────────┬────┴─────────┬──────────────┐
          │              │              │              │              │
   ┌──────┴──────┐ ┌─────┴──────┐ ┌────┴──────┐ ┌────┴──────┐ ┌────┴──────┐
   │  LEVEL 1   │ │  LEVEL 1   │ │  LEVEL 1  │ │  LEVEL 1  │ │  LEVEL 1  │
   │ Phase 1:   │ │ Phase 1:   │ │ Phase 1:  │ │ Phase 1:  │ │ Phase 1:  │
   │  Solve     │ │  Solve     │ │  Solve    │ │  Solve    │ │  Solve    │
   │ [Algebra]  │ │ [Geometry] │ │ [Combin.] │ │  [NT]     │ │  [Misc]   │
   └──────┬──────┘ └─────┬──────┘ └────┬──────┘ └────┬──────┘ └────┬──────┘
          └──────────────┴──────────────┴──────────────┴─────────────┘
                                        │
                   [cross-domain mixing: lightweight cross-attention, all 5 experts]
                                        │
          ┌──────────────┬──────────────┬──────────────┬──────────────┐
   ┌──────┴──────┐ ┌─────┴──────┐ ┌────┴──────┐ ┌────┴──────┐ ┌────┴──────┐
   │  LEVEL 2   │ │  LEVEL 2   │ │  LEVEL 2  │ │  LEVEL 2  │ │  LEVEL 2  │
   │ Phase 2:   │ │ Phase 2:   │ │ Phase 2:  │ │ Phase 2:  │ │ Phase 2:  │
   │  Verify    │ │  Verify    │ │  Verify   │ │  Verify   │ │  Verify   │
   └──────┬──────┘ └─────┬──────┘ └────┬──────┘ └────┬──────┘ └────┬──────┘
          └──────────────┴──────────────┴──────────────┴─────────────┘
                                        │
                               [cross-domain mixing]
                                        │
          ┌──────────────┬──────────────┬──────────────┬──────────────┐
   ┌──────┴──────┐ ┌─────┴──────┐ ┌────┴──────┐ ┌────┴──────┐ ┌────┴──────┐
   │  LEVEL 3   │ │  LEVEL 3   │ │  LEVEL 3  │ │  LEVEL 3  │ │  LEVEL 3  │
   │ Phase 3:   │ │ Phase 3:   │ │ Phase 3:  │ │ Phase 3:  │ │ Phase 3:  │
   │  Correct   │ │  Correct   │ │  Correct  │ │  Correct  │ │  Correct  │
   └──────┬──────┘ └─────┬──────┘ └────┬──────┘ └────┬──────┘ └────┬──────┘
          └──────────────┴──────────────┴──────────────┴─────────────┘
                 │
         [weighted aggregation by domain router weights w]
                 │
                 ↓
         h_K' (refined hidden state)
                 │
                 ↓
          LM Head (frozen) → answer tokens
```

## 7.2 Backbone insertion point

The PRISM blocks are inserted at a single point in the backbone: **after layer K**, where K is approximately 60-70% through the total layers. This is the point where:

- Early layers have processed syntax and token semantics
- Middle layers contain the richest task-relevant representations
- Late layers are more output-formatting focused

The exact value of K is determined empirically in Phase 0 using activation analysis (Section 5.2).

The PRISM blocks receive `h_K` (shape: `[batch, seq_len, hidden_dim]`) and produce `h_K'` of the same shape. The backbone then continues from layer K+1 using `h_K'` instead of `h_K`.

## 7.3 Domain router

A lightweight MLP classifier that takes the mean-pooled hidden state from layer K and outputs a 5-dimensional soft routing weight vector `w ∈ Δ^5` (a simplex over the 5 domains).

```
h_K_pooled = mean(h_K, dim=seq)          # [batch, hidden_dim]
w_logits   = RouterMLP(h_K_pooled)       # [batch, 5]
w          = softmax(w_logits)           # [batch, 5]
# domains: [Algebra, Geometry, Combinatorics, NumberTheory, Miscellaneous]
```

The RouterMLP is `hidden_dim → 256 → 5` with GELU activation. Approximately 0.5M parameters.

**Important design note for the Miscellaneous expert**: The router should be trained to assign significant Miscellaneous weight (≥ 0.2) to any problem whose solution requires a cross-domain tool (inequality, generating function, calculus step) as a primary sub-step, even if the problem is labeled as another domain. A geometry problem requiring AM-GM should receive w = [0.0, 0.6, 0.0, 0.0, 0.4] approximately. This means the router training labels need to be "soft" labels, not one-hot, for problems with a significant cross-domain component.

The router is trained in two ways:
1. **Supervised** with domain labels from the training dataset (cross-entropy on known labels)
2. **Self-supervised** end-to-end from answer correctness

## 7.4 Expert reasoning blocks (one level)

Each level contains **N_domains = 5** expert blocks, one per domain (Algebra, Geometry, Combinatorics, Number Theory, Miscellaneous). Each expert block is a compact transformer consisting of:

```
ExpertBlock(domain d, level l):
    - LayerNorm → Multi-head Self-Attention (n_heads=8, head_dim=64)
    - LayerNorm → FFN (4× expansion, SwiGLU activation)
    - Residual connections throughout
    - Input/output: [batch, seq_len, hidden_dim]
```

Hidden dim matches the backbone (e.g., 1024 for Qwen3.5-0.8B).

**Parameter count per block**: approximately `hidden_dim^2 × (4/3 × 4 + 2) ≈ 50-80M` for hidden_dim=1024. With 5 domains × 3 levels = 15 blocks, total PRISM params ≈ 750M–1.2B, plus backbone 0.8B = **1.55B–2.0B total**, well under 3B.

## 7.5 Cross-domain mixing

Between each level, a lightweight cross-attention module allows domain experts to query each other:

```
CrossMix(expert_outputs: [5, batch, seq_len, hidden_dim]):
    For each domain d:
        query = Linear(expert_outputs[d])        # [batch, seq, dim]
        keys  = concat(expert_outputs[all])      # [batch, 5*seq, dim]
        values = concat(expert_outputs[all])
        mixed[d] = expert_outputs[d] + CrossAttention(query, keys, values)
```

This is a single cross-attention layer with `n_heads=4`, `head_dim=32`. Very lightweight (~20M params total for all 3 levels). Key cross-domain interactions this module captures:

| Querying expert | Typically consults | Because |
|----------------|-------------------|---------|
| Geometry | Miscellaneous | AM-GM bounds on ratios, trig inequalities |
| Geometry | Algebra | Coordinate geometry, Vieta in circumradius formulas |
| Combinatorics | Miscellaneous | Generating functions for enumeration |
| Combinatorics | Number Theory | Modular arithmetic in counting mod p |
| Number Theory | Algebra | Polynomial techniques for Diophantine equations |
| Algebra | Miscellaneous | Inequality verification, calculus for extrema |
| Miscellaneous | all | Acts as an aggregator of cross-domain signals |

## 7.6 Final aggregation

The 5 expert outputs at Level 3 are aggregated using the router weights:

```
h_K' = sum_d( w[d] * expert_3[d].output )   # weighted sum over 5 domains
```

This is a soft mixture: a problem classified as 60% Geometry + 30% Miscellaneous + 10% Algebra (e.g., a geometry problem requiring AM-GM and a coordinate calculation) will blend all three experts' outputs in those proportions.

## 7.7 Thinking mode interaction

When `enable_thinking=True`, the backbone generates a `<think>...</think>` block before the final answer. The PRISM blocks operate on the full hidden state including the portion that generates the think block. This means PRISM can influence both the content of the internal reasoning and the final answer.

When `enable_thinking=False`, PRISM blocks operate on the direct answer generation path.

Both modes should be evaluated; the paper should report which combination works best.

---

# 8. Expert Trace Generation

## 8.1 Why existing traces are insufficient

Existing math datasets (NuminaMath-CoT, MATH solutions, MetaMathQA) contain:

- Generic reasoning traces not aligned to any specific mathematical domain perspective
- Single-pass solution attempts (no explicit verify or correct phases)
- No indication of which domain-specific approach is being used

A PRISM expert block trained on generic traces will not learn a domain-specific reasoning style — it will learn whatever averaged mixture is present in the data. This defeats the purpose of domain specialization.

We must generate **new, expert-aligned, phase-structured traces** using a large teacher model.

## 8.2 Teacher model

The teacher model should be the largest available vision-language model from the same family as the student:

- **Primary teacher**: `Qwen3.5-VL-72B` (if available) or `Qwen2.5-VL-72B`
- **Secondary teacher** (for cross-validation): `Qwen3.5-VL-7B` or `Qwen2.5-VL-7B`

The teacher must be a VL model (not text-only) because olympiad math problems include diagrams. Using a text-only teacher for geometry problems with diagrams would degrade trace quality significantly.

## 8.3 Trace structure

For each training problem, we generate **3-phase traces for the problem's primary domain** (and for the Miscellaneous expert if a cross-domain tool is required as a key step). We do NOT generate traces for all 5 domains for every problem — that would be wasteful. A problem labeled as Algebra gets Algebra traces + Miscellaneous traces (if it requires inequalities). A geometry problem with an AM-GM sub-step gets both Geometry and Miscellaneous traces.

```
Phase 1 — Solve:
  System prompt: "You are an expert in [DOMAIN]. Solve this problem using [DOMAIN]-specific techniques. 
                  Show your approach step by step without self-verification."
  Output: solution attempt T_solve

Phase 2 — Verify:
  System prompt: "You are an expert in [DOMAIN]. Review the following solution and identify specific 
                  mathematical errors or gaps. Do not produce a corrected solution, only diagnose."
  Input: problem + T_solve
  Output: verification report T_verify

Phase 3 — Correct:
  System prompt: "You are an expert in [DOMAIN]. Given the following solution and verification report, 
                  produce the corrected final solution."
  Input: problem + T_solve + T_verify
  Output: corrected solution T_correct
```

Where `[DOMAIN]` is replaced with the appropriate domain:

| Domain | System prompt emphasis |
|--------|----------------------|
| **Algebra** | Approach this as an algebraist: use symbolic manipulation, polynomial roots, Vieta's formulas, functional equations (especially Cauchy-type), completing the square, substitution strategies, and algebraic identities. Avoid coordinates unless forced. |
| **Geometry** | Approach this as a geometer: use angle chasing, circle theorems (power of a point, radical axis, cyclic quads), similar triangles, homothety, and projective tools (Ceva, Menelaus, cross-ratio, Pascal). Use coordinates or trigonometry only as computation tools after the key geometric insight is identified. |
| **Combinatorics** | Approach this as a combinatorialist: use bijective proofs, pigeonhole, double counting, recursion with explicit base cases, induction, and graph-theoretic arguments (Hall, Turán where relevant). Avoid algebraic shortcuts as the primary tool. |
| **Number Theory** | Approach this as a number theorist: use prime factorization, p-adic valuations (ν_p), modular arithmetic (Fermat/Euler, CRT, orders), and Diophantine techniques (Vieta jumping, descent, LTE). Identify the prime or modular structure before computing. |
| **Miscellaneous** | Approach this as a tools expert: identify which cross-domain tool is the key step. Use AM-GM / Cauchy-Schwarz / Jensen for bounding, generating functions for counting via algebra, calculus for extrema or convexity, complex roots of unity for periodic sums, or linear algebra for systems. Justify why the chosen tool is natural for this problem. |

## 8.4 Trace quality filtering

Keep a trace for training only if:

- The teacher's final answer in T_correct matches the ground-truth answer
- The T_verify step correctly identifies the error (if T_solve was wrong) or correctly confirms the solution (if T_solve was right)
- Total trace length is ≤ 4096 tokens (to avoid excessively long supervisions)

Expected pass rate: 30-60% of problems per domain. This means from ~860K NuminaMath problems, expect ~150K-300K high-quality traces per domain.

## 8.5 Dataset for trace generation

Source datasets to generate traces from (training only — test sets are kept completely separate):

| Dataset | Size | Domain labels | Use |
|---------|------|--------------|-----|
| **NuminaMath-CoT** | 860K | Yes (extracted) | Primary — generate new traces |
| **MATH (Hendrycks)** | 7.5K train | Yes (7 types) | Generate expert traces |
| **OpenR1-Math-220K** | 220K | Partial | Use for cross-validation |
| **MetaMathQA** | 395K | No | Use only for augmentation after domain classification |
| **AoPS scrape (public)** | varies | Manual | Hard problems, selective use |

Do NOT use any of the following for training:
- OlympiadBench (test set)
- OlymMATH (test set)
- Omni-MATH (test set)
- MATH-500 (validation set)
- AMC/AIME from 2022–2025 (evaluation set)

## 8.6 Trace generation compute estimate

At ~60 teacher tokens/second for Qwen2.5-VL-7B on a GH200:
- 1M problems × 3 phases × avg 512 tokens/phase = 1.5B tokens
- At 60 tok/s: ~6.9 days of continuous generation on 1 GPU
- With 4 GPUs in parallel: ~1.7 days

For the validation stage (Stage 0), generate only 10,000 traces (2.5 hours on 1 GPU).

---

# 9. Dataset Plan

## 9.1 Full dataset summary

### Training sets (all traces generated fresh)
| Source | Original size | Expected high-quality traces | Domain split |
|--------|--------------|------------------------------|-------------|
| NuminaMath-CoT | 860K | ~300K total | ~40% Alg, ~25% Geo, ~20% Comb, ~15% NT |
| MATH train | 7.5K | ~3K | Balanced |
| OpenR1-Math-220K | 220K | ~80K | Mixed |

### Validation sets (for hyperparameter tuning and early stopping)
| Dataset | Size | Domain |
|---------|------|--------|
| MATH Level 4-5 (not in MATH-500) | ~2000 | Mixed |
| AMC 8/10/12 (2015-2021) | ~500 | Mixed |

### Test sets (completely held out — never look at until final evaluation)
| Dataset | Size | Difficulty | Notes |
|---------|------|-----------|-------|
| **OlymMATH (en-hard)** | 100 | Very hard | Primary test |
| **OlympiadBench (test_en)** | 2126 | Hard | Primary test |
| **Omni-MATH** | 4428 | Hard | Secondary test |
| **MATH-500** | 500 | Medium-hard | Sanity check |

## 9.2 Domain classification of training data

NuminaMath-CoT and MATH have explicit domain labels. Before trace generation, we:

1. Extract or map existing domain labels to our 4 categories (Algebra, Geometry, Combinatorics, Number Theory)
2. Discard problems that span more than 2 domains (too mixed for clean expert training)
3. Produce a domain-labeled training split with approximately equal representation

Mapping from MATH's 7 categories to our 4:
- Algebra → Algebra
- Geometry → Geometry
- Counting and Probability → Combinatorics
- Number Theory → Number Theory
- Intermediate Algebra → Algebra (mostly)
- Pre-calculus → Algebra (mostly)
- Precalculus with geometry content → Geometry

---

# 10. Training Procedure

## 10.1 Overview: staged training

Training is organized into **4 stages**, each with a validation gate before proceeding to the next. This prevents wasting GPU days on architectures or hyperparameters that are not working.

```
Stage 0: Hypothesis validation (cheap) ........... 2-3 days
Stage 1: PRISM minimal architecture .............. 1-2 weeks
Stage 2: Full PRISM with cross-networking ........ 2-3 weeks
Stage 3: End-to-end fine-tuning (optional) ....... 1 week
```

**Success gate**: each stage has explicit pass/fail criteria. Do not proceed to the next stage unless the current stage passes.

## 10.2 Stage 0 — Hypothesis validation with LoRA (2-3 days)

**Goal**: Validate that domain-specific fine-tuning helps on olympiad math before building the full PRISM architecture. If domain-specific LoRA does not outperform general fine-tuning, the core hypothesis is wrong and the project should be reconsidered.

**Setup**:
- Generate 12,500 high-quality expert traces (2,500 per domain × 5 domains) using the teacher. For Miscellaneous, select 2,500 problems that explicitly require a cross-domain tool (inequality, generating function, or calculus) as the key step.
- Train 5 domain-specific LoRA adapters on Qwen3.5-0.8B (r=16, alpha=32, target: q/v projections)
- Train 1 general LoRA adapter on all 12,500 traces
- Evaluate all 6 on MATH-500 (full) and OlymMATH (en-hard, first 50 samples only)

**Key checks**:
1. Does the Algebra LoRA outperform the general LoRA on algebra problems?
2. Does the Geometry LoRA outperform on geometry problems?
3. Does a simple ensemble (pick the best domain LoRA by domain classifier) outperform any single adapter?
4. Are the 3-phase traces better than 1-phase traces (use teacher's direct answer as a 1-phase ablation)?

**Pass gate for Stage 1**:
- At least 3 of 4 domain-specific adapters must outperform the general adapter on their own domain
- 3-phase traces must be better than 1-phase traces (or at minimum not worse)
- Ensemble of domain adapters must outperform general adapter by ≥5%

**If Stage 0 fails**: revisit trace generation quality, or consider that domain decomposition is incorrect (possibly reorganize domains).

**Compute**: 4 × 1 GPU × 4 hours training + 2 GPU × 2 hours eval ≈ 1 day

## 10.3 Stage 1 — PRISM minimal architecture (1-2 weeks)

**Goal**: Build a working 3-domain (Algebra + Geometry + Miscellaneous — the three most cross-cutting domains) 1-level (Phase 1: Solve only) PRISM architecture. The Miscellaneous expert is included from Stage 1 because almost every hard problem requires it. Validate that the new blocks learn to produce better hidden states than the frozen backbone alone.

**Architecture for Stage 1**:
- 3 expert blocks (Algebra, Geometry, Miscellaneous)
- 1 level (Phase 1 only — no verify or correct)
- Simple domain router (3-class with soft labels)
- No cross-domain mixing yet

**Training procedure for Stage 1**:

Step 1 — Train domain router independently:
```python
# Route to correct domain based on problem text
# Labels: domain labels from training set
# Loss: cross-entropy
# Freeze: backbone + expert blocks (not yet initialized)
# Epochs: 3-5
# Learning rate: 1e-4
# Batch size: 32
```

Step 2 — Train Algebra expert block (backbone + router frozen):
```python
# Feed algebra problems through backbone to get h_K
# Feed h_K through Algebra block to get h_K_alg
# Feed h_K_alg through remaining backbone → LM head
# Supervision: next-token prediction on Phase 1 Algebra traces
# Freeze: backbone, router, Geometry block
# Loss: cross-entropy on ground-truth trace tokens
# Epochs: 3
# Learning rate: 5e-5
# Batch size: 16
```

Step 3 — Train Geometry expert block (backbone + router + Algebra block frozen):
```python
# Same as Step 2 but with geometry traces
# Freeze: backbone, router, Algebra block
```

Step 4 — Joint evaluation (no fine-tuning):
```python
# Use soft routing: h_K' = w_alg * alg_block(h_K) + w_geo * geo_block(h_K)
# Evaluate on MATH-500 algebra subset and geometry subset
```

**Pass gate for Stage 2**:
- The Algebra expert block (hard-routed) must outperform the backbone alone on MATH algebra problems
- The Geometry expert block (hard-routed) must outperform the backbone alone on MATH geometry problems
- The Miscellaneous expert block must outperform the backbone alone on problems requiring inequalities or generating functions (select a test subset of ~100 such problems from MATH-500)
- Soft-routed ensemble of 3 must outperform any single block
- Training loss must converge (no instability)

**Diagnostic checks**:
- Visualize the domain router's attention to problem keywords (e.g., "polynomial" → Algebra, "circle" / "triangle" → Geometry, "maximize" / "≥" → Miscellaneous)
- Plot the loss curves for each expert separately — Miscellaneous expert should show a distinct loss curve from the others
- Check that the Geometry block's loss does not decrease when trained on Algebra problems (domain specialization sanity check)

## 10.4 Stage 2 — Full PRISM architecture (2-3 weeks)

**Goal**: Build the complete 4-domain, 3-level PRISM architecture with cross-domain mixing. This is the main model for the paper.

**Training order**:

```
Week 1:
  Day 1-2: Domain router training (5-class, soft labels for mixed-domain problems)
  Day 3-4: Level 1 (Phase 1: Solve) — train all 5 experts independently (parallelize on 4 GPUs: 
           Algebra on GPU0, Geometry on GPU1, Combinatorics on GPU2, NT+Misc on GPU3)
  Day 5:   Level 1 cross-mixing training (freeze expert blocks, train CrossMix)

Week 2:
  Day 1-2: Level 2 (Phase 2: Verify) — train all 5 experts independently (same GPU assignment)
  Day 3:   Level 2 cross-mixing
  Day 4-5: Level 3 (Phase 3: Correct) — train all 5 experts independently
  Day 5:   Level 3 cross-mixing

Week 3:
  Day 1-2: Joint validation and calibration
  Day 3-5: Ablation experiments
```

**Per-expert training rule**: When training Expert Block (domain d, level l):
- Freeze: backbone, domain router, all other expert blocks, all CrossMix modules
- Unfrozen: only Expert Block (d, l)
- Supervision: Phase l traces for domain d
- Loss: next-token prediction cross-entropy on the teacher's trace tokens

**Cross-mixing training**: After all 4 experts at a given level are trained:
- Freeze all expert blocks and backbone
- Unfreeze only the CrossMix module for this level
- Loss: same next-token prediction, but on the mixed output
- This teaches the cross-mixing to aggregate expert signals effectively

**Validation at each level**: Before training Level l+1, run MATH-500 eval to confirm that Level l provides positive contribution. If Level l provides no gain, debug before training Level l+1.

## 10.5 Stage 3 — End-to-end fine-tuning (optional, 1 week)

After Stage 2 is complete, optionally fine-tune the entire system end-to-end:
- Unfreeze: all PRISM blocks, domain router, CrossMix
- Keep frozen: backbone
- Use a very low learning rate (1e-6) with cosine schedule
- Run on the full training set (if compute allows)
- Monitor for catastrophic forgetting of the sequential per-expert training

This stage is optional and should only be done if Stage 2 results show a significant gap between PRISM and the 7B baseline that might be closed by joint optimization.

## 10.6 Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Expert block hidden dim | Match backbone (1024 for 0.8B) | Must match for residual connection |
| Expert block n_heads | 8 | |
| Expert block head_dim | 64 | |
| Expert block FFN expansion | 4× | SwiGLU activation |
| CrossMix n_heads | 4 | Lightweight |
| Router MLP dims | hidden→256→4 | |
| Learning rate (router) | 1e-4 | |
| Learning rate (expert blocks) | 5e-5 | |
| Learning rate (cross-mix) | 2e-5 | |
| Batch size | 16 per GPU | Gradient accumulation × 4 |
| Max sequence length | 2048 | |
| Warmup steps | 100 | |
| LR schedule | cosine | |
| Weight decay | 0.01 | |
| Gradient clipping | 1.0 | |

---

# 11. Evaluation Plan

## 11.1 Primary metrics

- **Exact match accuracy** (%): answer string exactly matches the ground truth (after normalization)
- **Partial credit** (%): for multi-part problems, fraction of sub-parts correct

## 11.2 Evaluation benchmarks

### Primary test benchmarks (completely held out)
- **OlymMATH (en-hard)**: 100 very hard olympiad problems. This is the hardest benchmark.
- **OlympiadBench (test_en)**: 2,126 competition problems with domain labels. Use domain-stratified evaluation.

### Secondary test benchmarks
- **Omni-MATH**: 4,428 competition problems. Use for broad coverage.

### Validation benchmarks (used for hyperparameter tuning)
- **MATH-500**: 500 competition problems. Stratified by difficulty level (1-5).

### Sanity check benchmarks
- **GSM8K**: 8,500 grade school math problems. Should get ≥80% if the model is not regressing.

## 11.3 Ablation experiments

The following ablations must be run and reported in the paper:

| Ablation | What it tests | Expected result |
|----------|--------------|-----------------|
| A1: No PRISM (backbone only, thinking=False) | Weakest baseline | Lowest |
| A2: Backbone + thinking=True only | Is built-in thinking sufficient? | Better than A1, lower than PRISM |
| A3: PRISM, 1 general domain (no specialization, all 5 replaced by 1 generic block) | Is domain split necessary? | Lower than 5-domain PRISM |
| A4: PRISM, 4 domains only (no Miscellaneous expert) | Is the Misc expert necessary? | Lower, especially on inequality-heavy problems |
| A5: PRISM, no Phase 2 (no verify level) | Is verify level necessary? | Lower |
| A6: PRISM, no Phase 3 (no correct level) | Is correct level necessary? | Lower |
| A7: PRISM, no cross-mixing | Is cross-domain mixing necessary? | Lower on mixed-domain problems |
| A8: PRISM, hard routing (argmax instead of soft) | Soft vs. hard routing | Lower, especially on boundary problems |
| A9: PRISM, generic traces (existing CoT, not expert-aligned) | Expert trace quality | Significantly lower |
| A10: PRISM + thinking=True | Best combo? | Possibly best overall |
| A11: PRISM, Misc expert uses only inequality traces (no calculus/genf) | Scope of Misc expert | Lower on genf/calculus problems |

## 11.4 Baselines (models to compare against)

| Baseline | Params | Notes |
|----------|--------|-------|
| Qwen3.5-0.8B, thinking=False | 0.8B | Unmodified base |
| Qwen3.5-0.8B, thinking=True | 0.8B | Built-in reasoning |
| Qwen3.5-1.7B, thinking=True | 1.7B | Same family, 2× size |
| Qwen3.5-7B, thinking=True | 7B | Same family, 9× size |
| Qwen2.5-Math-1.5B | 1.5B | Math-specialized but text-only |
| Qwen2.5-VL-3B | 3B | VL model at similar size to PRISM |
| PRISM (ours) | <3B | The proposed model |

The key comparison for NeurIPS is: **PRISM (<3B) vs. Qwen3.5-7B (thinking=True)**.
If PRISM achieves >70% of the 7B model's accuracy at <40% of the parameters, the efficiency story is publishable.

---

# 12. Codebase Structure

## 12.1 Design principles

The codebase must be **professional, self-contained, and reusable by anyone**. Specifically:

1. **`pip install -e .`** installs the full package in development mode. A clean `pip install prism-math` installs from PyPI (future). No manual path manipulation, no `sys.path` hacks.

2. **HuggingFace-native interface.** `PRISMModel` subclasses `PreTrainedModel`. `PRISMConfig` subclasses `PretrainedConfig`. This means:
   ```python
   # Any researcher anywhere can use the released model:
   from transformers import AutoModel, AutoProcessor
   model = AutoModel.from_pretrained("debajyoti/prism-0.8b", trust_remote_code=True)
   processor = AutoProcessor.from_pretrained("debajyoti/prism-0.8b", trust_remote_code=True)
   # Single forward pass, no extra steps:
   outputs = model(**inputs)
   ```

3. **`model.save_pretrained(path)` and `AutoModel.from_pretrained(path)` must work** end-to-end, saving config.json + model weights in standard HF format.

4. **No hardcoded paths.** All paths (cache, model dir, results dir) come from environment variables with sane defaults. The package works on any machine without editing source files.

5. **Zero-change inference.** Once trained and pushed to HF Hub, a user should be able to run PRISM on their own math problem with no knowledge of the internal architecture. The complexity is hidden behind the standard HF interface.

## 12.2 Directory layout

```
PRISM/                               ← project root (git repo)
├── program.md                       ← this file (source of truth)
├── PROGRESS.md                      ← live experiment log (always up to date)
├── README.md                        ← quick-start for external users
├── pyproject.toml                   ← pip install -e . entry point
├── setup.cfg                        ← package metadata, dependencies
│
├── src/
│   └── prism/
│       ├── __init__.py              ← exposes PRISMModel, PRISMConfig
│       ├── model/
│       │   ├── config.py            ← PRISMConfig(PretrainedConfig): backbone_name,
│       │   │                            n_domains, n_phases, hidden_dim, insert_layer
│       │   ├── backbone.py          ← load Qwen3.5-0.8B from /tmp, patch insertion point
│       │   ├── expert_block.py      ← ExpertBlock(nn.Module): self-attn + FFN, residual
│       │   ├── cross_mix.py         ← CrossMixModule(nn.Module): cross-domain attention
│       │   ├── router.py            ← DomainRouter(nn.Module): MLP, 5-class soft output
│       │   └── prism_model.py       ← PRISMModel(PreTrainedModel): assembles all parts,
│       │                                implements forward(), save_pretrained(),
│       │                                from_pretrained(), generate()
│       ├── data/
│       │   ├── __init__.py
│       │   ├── datasets.py          ← HF dataset loaders (NuminaMath, MATH, etc.)
│       │   ├── domain_split.py      ← map problem text → domain label (5-class)
│       │   ├── trace_format.py      ← parse/format 3-phase expert traces
│       │   └── collator.py          ← DataCollator for variable-length trace batches
│       ├── generation/
│       │   ├── __init__.py
│       │   ├── trace_generator.py   ← generate expert traces via teacher VLM
│       │   └── phase_prompts.py     ← domain × phase system prompts (5 × 3 = 15)
│       ├── training/
│       │   ├── __init__.py
│       │   ├── train_router.py      ← router training (domain classification)
│       │   ├── train_expert.py      ← per-expert-block SFT (freeze all others)
│       │   ├── train_crossmix.py    ← cross-mix training (freeze expert blocks)
│       │   └── train_e2e.py         ← optional end-to-end fine-tuning (Stage 3)
│       └── eval/
│           ├── __init__.py
│           ├── eval_prism.py        ← main eval: load model, run benchmarks, save JSON
│           ├── metrics.py           ← exact_match(), partial_credit(), normalize_answer()
│           └── ablations.py         ← run all ablations in parallel across GPUs
│
├── scripts/
│   ├── setup/
│   │   └── env.sh                   ← sets PRISM_ROOT, HF_HOME, PRISM_MODEL_DIR, HF_TOKEN
│   ├── generate_traces.sh           ← parallel trace gen on all 4 GPUs
│   ├── train_stage0.sh              ← Stage 0: 5 LoRA adapters + eval
│   ├── train_stage1.sh              ← Stage 1: 3-domain 1-level PRISM
│   ├── train_stage2.sh              ← Stage 2: full 5-domain 3-level PRISM
│   └── eval_all.sh                  ← all benchmarks + ablations in parallel
│
├── configs/
│   ├── model/
│   │   └── prism_0.8b.yaml          ← PRISMConfig values
│   ├── training/
│   │   ├── stage0_lora.yaml
│   │   ├── stage1_minimal.yaml
│   │   └── stage2_full.yaml
│   └── eval/
│       └── benchmarks.yaml          ← benchmark paths, n_samples, metrics
│
├── results/                         ← ALL artifacts committed to git (except model weights)
│   ├── traces/                      ← generated expert traces (JSONL, committed)
│   ├── stage0/
│   │   ├── lora_adapters/           ← LoRA adapter weights (.safetensors, committed if <100MB)
│   │   └── eval/                    ← eval result JSONs
│   ├── stage1/
│   │   ├── router/                  ← router checkpoint
│   │   ├── expert_blocks/           ← per-expert block weights
│   │   └── eval/
│   ├── stage2/
│   │   ├── router/
│   │   ├── expert_blocks/           ← 15 expert block weight files
│   │   ├── cross_mix/
│   │   └── eval/
│   └── ablations/                   ← ablation eval results
│
├── .cache/                          ← HF download cache (gitignored)
│   └── huggingface/
│       └── hub/                     ← downloaded model snapshots (large, not committed)
│
└── .gitignore                       ← excludes .cache/, /tmp/, *.bin, *.safetensors >100MB
```

## 12.3 Package entry points (`pyproject.toml`)

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "prism-math"
version = "0.1.0"
description = "PRISM: Phase-structured Reasoning with Integrated Subject-expert Modules"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.9",                    # requires CUDA 12.9
    "transformers>=4.40",
    "peft>=0.10",
    "accelerate>=0.28",
    "datasets>=2.18",
    "huggingface-hub>=0.22",
    "safetensors>=0.4",
    "pillow>=10.0",           # for VL image processing
    "pyyaml>=6.0",
]

[project.scripts]
prism-train   = "prism.training.train_expert:main"
prism-eval    = "prism.eval.eval_prism:main"
prism-traces  = "prism.generation.trace_generator:main"
```

After `pip install -e .`, the following all work:
```bash
prism-train --config configs/training/stage1_minimal.yaml
prism-eval  --config configs/eval/benchmarks.yaml --model results/stage2/
prism-traces --config configs/training/stage0_lora.yaml
```

## 12.4 Storage rules (enforced)

| Data type | Location | In git? | Notes |
|-----------|----------|---------|-------|
| Downloaded model weights | `.cache/huggingface/hub/` | No (gitignored) | Persistent across runs |
| Runtime model weights | `/tmp/prism_models/` | No (RAM only) | Copied from .cache on startup; lost on reboot |
| Expert block checkpoints | `results/stageN/expert_blocks/` | Yes (if <100MB) | Small — ~80MB per block × 15 = 1.2GB total |
| Eval result JSONs | `results/stageN/eval/` | Yes | Always committed |
| Generated traces | `results/traces/` | Yes (JSONL, compressed) | Core dataset contribution |
| Configs and hyperparams | `configs/` | Yes | Always committed |
| Large backbone weights (>100MB) | HF Hub only | No | Reference by model ID, not stored locally |

---

# 13. Compute Plan

## 13.1 Hardware

```
4× NVIDIA GH200 GPUs (~96GB usable each)
CUDA 12.9
torch 2.9
Python 3.12
Env: /iopsstor/scratch/cscs/dasgupta/research/ideas/AMSD/.venv
```

## 13.2 GPU utilization rules

**All 4 GPUs must be busy at all times.** This is not optional. The default strategy is:

- When training expert blocks: assign one or two experts per GPU so all 4 GPUs run simultaneously
- When generating traces: run the teacher on 2 GPUs (tensor parallel or pipeline parallel); use the other 2 GPUs for a different task (e.g., LoRA validation training)
- When evaluating: run 4 evaluation jobs in parallel across 4 GPUs (different benchmarks or different ablations)
- When one stage finishes: the next stage's job should start on that GPU within minutes, not hours

Each GH200 has ~96GB. A single expert block training job (backbone fp16 + one block + optimizer states) uses ~25-35GB. This means **2 expert training jobs can run per GPU simultaneously** with careful memory management.

## 13.3 Parallel training schedule (default assignment)

```
Stage 2 training — expert block training (5 experts × 3 levels = 15 jobs):

Level 1 training (all experts in parallel):
  GPU 0: Algebra expert (Level 1)     + Combinatorics expert (Level 1)
  GPU 1: Geometry expert (Level 1)    + Miscellaneous expert (Level 1)
  GPU 2: Number Theory expert (Level 1)
  GPU 3: Evaluation / trace generation for Level 2 data

Level 2 training (same GPU assignment, after Level 1 CrossMix):
  GPU 0: Algebra (L2) + Combinatorics (L2)
  GPU 1: Geometry (L2) + Miscellaneous (L2)
  GPU 2: Number Theory (L2)
  GPU 3: Evaluation / CrossMix training

Level 3 training: same pattern.
```

## 13.4 GPU allocation by stage

| Stage | GPU assignment | Duration | Notes |
|-------|---------------|---------|-------|
| Stage 0 — trace gen | GPU 0+1: teacher (tensor parallel); GPU 2+3: LoRA training | ~2-3 days | Run simultaneously |
| Stage 0 — LoRA eval | GPU 0: 5 LoRA adapters eval | ~2 hours | |
| Stage 1 — minimal PRISM | GPU 0: Algebra; GPU 1: Geometry; GPU 2: Misc; GPU 3: eval | ~2 days | 3 experts in parallel |
| Stage 2 — full PRISM | 2 experts/GPU where possible | ~2 weeks | 15 total training jobs |
| Final eval | GPU 0-3: 4 benchmarks × 4 ablation sets | ~1 day | All 4 GPUs evaluating |

## 13.5 Memory budget

| Component | Memory |
|-----------|--------|
| Qwen3.5-0.8B backbone (fp16) | ~1.6 GB |
| 15 expert blocks × 80M params (fp16) | ~2.4 GB |
| CrossMix modules (3 levels, 5 experts) | ~0.12 GB |
| Router MLP (5-class) | ~0.01 GB |
| **Total model — inference** | **~4.1 GB** |
| Training — 1 expert (backbone + block + Adam states) | ~25-35 GB |
| Training — 2 experts per GPU (with gradient checkpointing) | ~50-70 GB |

Two expert training jobs per GPU is feasible within 96GB with gradient checkpointing enabled. Three per GPU may be possible for smaller blocks; test empirically.

**Inference footprint:** The full PRISM model (backbone + all 15 expert blocks) fits in under 5GB VRAM — deployable on consumer hardware.

---

# 14. Experiment Schedule

## 13.1 Week 1 — Foundation

**Day 1-2**: Set up PRISM repository. Implement backbone loader with insertion point. Verify that frozen backbone still produces correct outputs after insertion point.

**Day 3-4**: Implement ExpertBlock, CrossMixModule, DomainRouter. Write unit tests for shape correctness.

**Day 5**: Implement trace generator. Test with 100 problems. Check trace quality manually. Verify that Phase 1/2/3 prompts produce sensible domain-specific traces.

## 13.2 Week 2 — Stage 0 validation

**Day 1-3**: Generate 10,000 expert traces (2,500 per domain). Log pass rate, average trace length per phase, and fraction correct in Phase 3 (teacher's final answer matches ground truth).

**Day 4-5**: Train domain-specific LoRA adapters. Run Stage 0 pass gate evaluation. **Make go/no-go decision for Stage 1.**

## 13.3 Week 3-4 — Stage 1

**Day 1-3**: Train domain router (4-class, using training set domain labels).
**Day 4-7**: Train Level 1 expert blocks for Algebra and Geometry.
**Day 8-10**: Evaluate Stage 1 pass gate. Debug if needed. **Make go/no-go decision for Stage 2.**

## 13.4 Week 5-7 — Stage 2

**Week 5**: Expand to 4 domains at Level 1. Train all 4 Level 1 experts. Train Level 1 CrossMix. Validate.
**Week 6**: Train Level 2 (Verify) experts and CrossMix. Validate.
**Week 7**: Train Level 3 (Correct) experts and CrossMix. Full model validation.

## 13.5 Week 8 — Evaluation and ablations

Run all ablations from Section 10.3. Run all baselines. Run full evaluation on held-out test sets. Write up results.

---

# 15. Expected Results and Success Criteria

## 14.1 Minimum bar for paper submission

All of the following must hold:

- PRISM outperforms Qwen3.5-0.8B (thinking=True) on both OlympiadBench and OlymMATH
- PRISM outperforms Qwen2.5-VL-3B (a larger, comparable VL model) on both benchmarks
- Domain-specific experts outperform a general expert in their respective domains (ablation A2)
- 3-phase structure outperforms 1-phase (ablation A3+A4 combined)

## 14.2 Strong result (makes main track NeurIPS more likely)

- PRISM achieves ≥70% of Qwen3.5-7B performance on OlympiadBench
- Expert alignment in traces produces ≥5% absolute improvement over generic traces (ablation A7)
- PRISM's cross-mixing shows measurable improvement on cross-domain problems (ablation A5)

## 14.3 Excellent result (clearly NeurIPS main track)

- PRISM approaches or matches Qwen3.5-7B on OlympiadBench
- Interpretability analysis shows router correctly identifies domains and expert blocks activate meaningfully
- The dataset of expert-aligned traces is itself a contribution (large and high-quality enough to release)

---

# 16. Risks and Mitigations

| Risk | Probability | Mitigation |
|------|------------|-----------|
| Stage 0 fails: domain LoRA doesn't help | Low-medium | Reconsider domain decomposition; try 2-domain version |
| Expert blocks don't converge | Medium | Use warmup + careful LR schedule; debug with probes |
| Cross-domain mixing hurts performance | Low-medium | Make it optional; paper can exclude if ablation negative |
| Teacher traces are low quality | Low | Manual inspection of 100 traces before full generation |
| Backbone insertion breaks gradients | Low | Unit test gradient flow through insertion point |
| Training on 3-phase traces creates distribution shift | Medium | Also include 1-phase training as regularization |
| Total model >3B after full PRISM | Low | Reduce expert block size (n_heads=4 instead of 8) |

---

# 17. Dataset Contribution

A novel dataset of **expert-aligned, phase-structured mathematical reasoning traces** will be released alongside the model. This dataset has three properties that distinguish it from all existing math reasoning datasets:

1. **Domain alignment**: each trace is generated by prompting the teacher model as a specific domain expert (Algebra/Geometry/Combinatorics/Number Theory)
2. **Phase structure**: each example has 3 separate traces (solve/verify/correct), not just a single CoT solution
3. **VL-grounded**: the teacher model processes both the problem text AND any accompanying diagram, producing traces that are visually grounded

Estimated dataset size: 200K–500K examples (each example = 1 problem × 4 domains × 3 phases = 12 trace documents). This dataset would be a standalone NeurIPS Datasets and Benchmarks contribution if the PRISM paper targets a different venue.

---

# 18. Known Open Questions (research engineer should investigate)

1. **Backbone insertion depth**: What is the optimal K (insertion layer index)? Should there be multiple insertion points?

2. **Training order within a level**: Should expert blocks be trained sequentially or can we train them in parallel on separate GPUs?

3. **Teacher domain specialization**: Are the 4 domain system prompts sufficient to produce clearly different reasoning styles? Should we test with 8 domains (further splitting Combinatorics into counting vs. graph theory)?

4. **Thinking mode and PRISM**: Does PRISM conflict with Qwen3.5's built-in thinking mode? Or do they compose well? This is an empirical question for Stage 0.

5. **Phase 2 supervision signal**: The verification phase output is harder to evaluate (it's a diagnosis, not an answer). Should we use a different loss for Phase 2 (e.g., binary correct/incorrect classification rather than next-token prediction)?

6. **Soft vs. hard routing**: The domain router produces soft weights. Should we use Gumbel-softmax with straight-through estimator during training to encourage harder routing?

7. **Trace length budget**: Phase 1 traces may be much longer than Phase 2/3 traces for easy problems. Should we enforce phase-length budgets to prevent degenerate solutions?

---

# 19. Important Notes for the Research Engineer

1. **Do not use test sets for any training decision**. OlympiadBench, OlymMATH, Omni-MATH are completely held out. Do not even look at examples from these sets during development, as this can introduce unconscious evaluation leakage.

2. **Validate every stage before proceeding**. The go/no-go gates in Section 10 are hard requirements. GPU time is expensive. Do not train for weeks on an architecture that failed Stage 0.

3. **Keep the backbone frozen throughout**. The backbone is frozen in all training stages except optional Stage 3. This is essential for training stability and ensures that the PRISM blocks are genuinely adding new capability rather than re-learning what the backbone already knows.

4. **Log everything**. Every training run should log: loss curve, domain router accuracy, exact-match on MATH-500 (per domain), parameter counts, and GPU memory usage.

5. **Unit test shape invariants**. The expert blocks must accept `[batch, seq_len, hidden_dim]` and return the same shape. Test this before any training run.

6. **Qwen3.5 thinking mode**: When calling `apply_chat_template`, pass `enable_thinking=False` explicitly to disable thinking mode for the baseline. For PRISM experiments, test both modes. The `enable_thinking` flag is passed as a kwarg and may not be in the function signature — use try/except.

7. **Memory management**: Expert blocks can be loaded/unloaded independently. When training Expert Block (d=Algebra, l=1), only that block and the backbone need to be on GPU. All other blocks stay on CPU. This enables fitting 2 expert training jobs per GPU within 96GB.

8. **Storage: models in `/tmp` only**. Runtime model weights live at `/tmp/prism_models/`. On the machine we have >1TB RAM so `/tmp` is enormous and fast. Never load models from the project repo or from disk. Always load from `/tmp`. The startup script copies from `.cache/` to `/tmp/` if needed.

9. **Storage: HF download cache in project repo**. `HF_HOME=${PRISM_ROOT}/.cache/huggingface`. Downloaded models go there. The `.cache/` directory is gitignored (too large), but it persists across runs and means you only download once.

10. **Storage: all artifacts in `results/`**. Every eval JSON, every trace file, every router checkpoint, every expert block weight — goes into `results/`. Committed to git (use Git LFS for files >100MB). Nothing important should ever be only in `/tmp`.

11. **HF interface compliance**. `PRISMModel` must pass `AutoModel.from_pretrained("path/or/hub_id")` from day one. Implement `_auto_class = "AutoModel"` in the config. Register the model class in `auto_map` in `config.json`. Test this after every architectural change.

12. **Teacher model access**: The teacher requires `HF_TOKEN` from `~/.cache/huggingface/token`.

10. **No idle GPUs**. This is a hard rule. If you are waiting for one job to finish before starting the next, you are doing it wrong. Always have the next job ready to launch. Use `nohup ... &` with logs to `/tmp/prism_*.log`. Check GPU status with `nvidia-smi` regularly.

11. **PROGRESS.md is mandatory**. After every experiment that produces a result, update `PROGRESS.md` in the project root:
    - What is currently running on each GPU (job name, PID, start time, log file)
    - What has completed (key metrics: loss, accuracy)
    - What is queued next on each GPU
    - Any anomalies or unexpected results
    
    Commit PROGRESS.md to git after every update. A new engineer must be able to pick up the project by reading PROGRESS.md within 5 minutes.

12. **Cluster name**: Do not include the cluster name in any code comments, log files, or documentation. Reference the hardware as "4× GH200 96GB GPUs" only.

10. **No inference-time scaling at test time**: The final PRISM model should produce its answer in a single forward pass. Do not add multi-step prompting or multi-pass refinement at test time. If the paper needs multi-pass as a baseline comparison, implement it separately and label it clearly.
