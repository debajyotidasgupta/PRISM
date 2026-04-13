---
name: User Profile
description: Who is working on PRISM and how to tailor communication
type: user
---

Debajyoti Dasgupta — research engineer targeting NeurIPS-level publications.

Working on CSCS Alps HPC cluster with 4× NVIDIA GH200 120GB GPUs (102GB usable each).

Runs both AMSD (ARC-AGI, in results phase) and PRISM (Olympiad math, planning phase) as separate projects.

**Communication preferences:**
- Direct, technical, no fluff
- Prefers honest assessment over optimism
- Wants staged validation before committing GPU time (explicitly requested this)
- Thinks in terms of paper contributions, not just engineering tasks
- Prefers new project files in the project folder itself, not scattered

**HPC notes:**
- Conda env is already active — do NOT prefix with `source activate`
- HF_TOKEN at `~/.cache/huggingface/token`
- Use `HF_HOME=/tmp/amsd_cache` for fast model loading (RAM-backed tmpfs)
- GPU time is expensive — always validate small before scaling
