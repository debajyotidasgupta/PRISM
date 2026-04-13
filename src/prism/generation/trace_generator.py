"""
Trace generation backends for PRISM expert trace production.

Two backends are available:

1. TraceGenerator (default — HF transformers)
   - Serial: N×3 forward passes, one (problem, phase) at a time
   - Works with any HF-compatible model, no extra deps
   - Use for single-GPU debugging or when vLLM is unavailable

2. VLLMBatchGenerator (fast path — vLLM 0.19.0 installed and working)
   - Batched: 3 passes for N problems (all Phase-1 prompts together, etc.)
   - Continuous batching + PagedAttention + prefix caching → ~10-50× faster
   - Enable with --use-vllm flag (or USE_VLLM=1 in generate_traces.sh)
   - Installation note: vLLM was installed first (temporarily changing torch to
     2.10.0 and transformers to 4.57.6), then torch 2.9.0+cu129 and
     transformers 5.5.3 were restored. causal_conv1d and vllm/flashinfer both
     work correctly with the restored versions.

Primary teacher: Qwen/Qwen3.5-35B-A3B (35B total, ~3.5B active MoE, fits on 1 GH200)
  - Supports enable_thinking=True for thinking traces
  - Generation params: temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5

Fallback teacher: Qwen/Qwen3-VL-30B-A3B-Thinking (VL, 30B, also fits on 1 GH200)

Usage:
  # HF backend (default):
  generator = TraceGenerator(teacher_model_name="Qwen/Qwen3.5-35B-A3B", gpu_id=0)
  generator.load()
  trace = generator.generate_trace(problem, domain="algebra", ground_truth="42",
                                   reference_solution="...")

  # vLLM backend (fast, requires vLLM):
  generator = VLLMBatchGenerator(teacher_model_name="Qwen/Qwen3.5-35B-A3B", gpu_id=0)
  generator.load()
  stats = generator.generate_dataset(problems, domain, output_file)
"""

import os
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

from prism.data.trace_format import TraceExample, extract_final_answer, answers_match
from prism.generation.phase_prompts import (
    get_phase_system_prompt,
    get_phase_user_prompt,
    format_messages_for_qwen,
)

logger = logging.getLogger(__name__)

# Supported teacher models (>14B only)
VL_TEACHER_MODELS = {
    "Qwen/Qwen3.5-35B-A3B",             # Primary — 35B MoE, ~3.5B active, enable_thinking, 1 GH200
    "Qwen/Qwen3-VL-30B-A3B-Thinking",   # Fallback — 30B VL, thinking mode, 1 GH200
}


class TraceGenerator:
    """
    Generates 3-phase expert traces using a VL teacher model.

    Supports:
      - Text-only problems (standard math)
      - Image+text problems (geometry diagrams, coordinate figures)
      - Cross-domain verification (solve in domain A, verify in domain B)

    Args:
        teacher_model_name: HF model ID. Must be a VL model (not text-only).
        gpu_id: GPU device index.
        max_new_tokens_per_phase: Max tokens generated per phase.
        temperature: Sampling temperature. 0.0 = greedy, 0.7 = diverse.
        torch_dtype: Model dtype.
    """

    def __init__(
        self,
        teacher_model_name: str = "Qwen/Qwen3.5-35B-A3B",
        gpu_id: int = 0,
        max_new_tokens_per_phase: int = 1024,
        temperature: float = 1.0,
        torch_dtype=torch.float16,
    ):
        self.teacher_model_name = teacher_model_name
        self.gpu_id = gpu_id
        self.max_new_tokens = max_new_tokens_per_phase
        self.temperature = temperature
        self.dtype = torch_dtype
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._processor = None
        self._loaded = False
        self._is_qwen3_vl = "Qwen3" in teacher_model_name
        self._is_qwen2_vl = "Qwen2.5" in teacher_model_name

    def load(self):
        """Load the teacher VL model. Must be called before generation."""
        import os
        hf_token = os.environ.get("HF_TOKEN", None)

        # Resolve model path (from /tmp or HF cache)
        from prism.model.backbone import _get_model_dir
        model_path = _get_model_dir(self.teacher_model_name)

        logger.info(f"Loading VL teacher: {self.teacher_model_name} on GPU {self.gpu_id}")

        # VL models need AutoModelForImageTextToText (Qwen3-VL, Qwen3.5, Qwen2.5-VL all use this)
        from transformers import AutoModelForImageTextToText, AutoProcessor

        model_kwargs = dict(
            dtype=self.dtype,       # note: use dtype not torch_dtype for newer transformers
            trust_remote_code=True,
            device_map=f"cuda:{self.gpu_id}",
        )
        if hf_token:
            model_kwargs["token"] = hf_token

        try:
            self._model = AutoModelForImageTextToText.from_pretrained(model_path, **model_kwargs)
        except Exception as e:
            logger.warning(f"AutoModelForImageTextToText failed ({e}), trying AutoModelForCausalLM")
            from transformers import AutoModelForCausalLM
            model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
            self._model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        self._model.eval()

        self._processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            token=hf_token,
        )

        logger.info(f"Teacher loaded: {type(self._model).__name__}, device={self.device}")
        self._loaded = True
        return self

    def _build_messages(
        self,
        system_prompt: str,
        user_text: str,
        image=None,
    ) -> list[dict]:
        """
        Build chat messages, optionally with image.

        For VL models, image is embedded directly in the content list.
        For text-only models, image is ignored.

        Args:
            system_prompt: System role content.
            user_text: User text content.
            image: PIL Image, path string, or URL. None for text-only.

        Returns:
            Messages list for apply_chat_template.
        """
        if image is not None:
            # VL format: content is a list with image + text items
            user_content = [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": user_text,
                },
            ]
        else:
            user_content = user_text

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    def _generate_phase(
        self,
        system_prompt: str,
        user_message: str,
        image=None,
    ) -> str:
        """
        Run one phase of generation with the teacher model.
        Handles both text-only and image+text (VL) input.

        Returns:
            Generated text (stripped, no special tokens).
        """
        messages = self._build_messages(system_prompt, user_message, image)

        # Apply Qwen chat template with thinking mode enabled (Qwen3-VL-Thinking)
        try:
            text = self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,   # Qwen3 thinking mode
            )
        except TypeError:
            # Fallback for models without enable_thinking param
            text = self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        # Process inputs — handle image if present
        if image is not None:
            try:
                # Qwen3-VL / Qwen2.5-VL: use process_vision_info for image preprocessing
                from qwen_vl_utils import process_vision_info
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self._processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
            except ImportError:
                logger.warning("qwen_vl_utils not found, falling back to text-only")
                inputs = self._processor(text=[text], return_tensors="pt")
            except Exception as e:
                logger.warning(f"Image processing failed ({e}), using text-only")
                inputs = self._processor(text=[text], return_tensors="pt")
        else:
            inputs = self._processor(text=[text], return_tensors="pt")

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Thinking mode generation params per Qwen3 official docs:
        # temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5
        generate_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,   # 1.0 for thinking mode
            top_p=0.95,
            top_k=20,
            pad_token_id=self._processor.tokenizer.eos_token_id,
        )
        # presence_penalty supported in newer transformers
        try:
            generate_kwargs["presence_penalty"] = 1.5
            with torch.no_grad():
                output_ids = self._model.generate(**inputs, **generate_kwargs)
        except TypeError:
            generate_kwargs.pop("presence_penalty", None)
            with torch.no_grad():
                output_ids = self._model.generate(**inputs, **generate_kwargs)

        input_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0][input_len:]
        return self._processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def generate_trace(
        self,
        problem: str,
        domain: str,
        ground_truth: str,
        reference_solution: str = "",
        problem_id: str = "",
        image=None,
        cross_verify_domain: str = None,
    ) -> TraceExample:
        """
        Generate a full 3-phase trace for one problem.

        The teacher is NEVER asked to solve from scratch. Instead:
          Phase 1: Reformulate `reference_solution` in expert domain style.
          Phase 2: Verify the Phase 1 trace against the known `ground_truth`.
          Phase 3: Produce final polished solution using Phase 1 + Phase 2.

        This ensures high-quality traces regardless of whether the teacher
        can independently solve olympiad-level problems.

        For standard traces: all 3 phases use the same domain expert.
        For cross-domain verification: Phase 2 uses a DIFFERENT domain expert,
        modelling scenarios where an algebra problem is verified via combinatorics.

        Args:
            problem: Problem text.
            domain: Primary PRISM domain for Phases 1 and 3.
            ground_truth: Known correct final answer (used by Phase 2/3 for grounding).
            reference_solution: Full worked solution from the source dataset. This is
                                 given to Phase 1 so the teacher reformulates it rather
                                 than solving from scratch. If empty, ground_truth is used.
            problem_id: Unique identifier for this problem.
            image: PIL Image or path for geometry/VL problems.
            cross_verify_domain: If provided, use THIS domain for Phase 2 (verify).
                                  E.g., verify an algebra problem using miscellaneous tools.

        Returns:
            TraceExample with all 3 phases filled in.
        """
        assert self._loaded, "Call load() first"

        verify_domain = cross_verify_domain if cross_verify_domain else domain

        # ─── Phase 1: Reformulate reference solution in expert style ──────
        sys1 = get_phase_system_prompt(domain, phase=0)
        usr1 = get_phase_user_prompt(
            problem, phase=0, domain=domain,
            reference_solution=reference_solution,
            ground_truth=ground_truth,
        )
        solve_trace = self._generate_phase(sys1, usr1, image=image)

        predicted_1 = extract_final_answer(solve_trace)
        solve_correct = answers_match(predicted_1, ground_truth)

        # ─── Phase 2: Verify trace (potentially DIFFERENT domain expert) ──
        sys2 = get_phase_system_prompt(verify_domain, phase=1)
        usr2 = get_phase_user_prompt(
            problem, phase=1, domain=verify_domain,
            ground_truth=ground_truth,
            solve_trace=solve_trace,
        )
        verify_trace = self._generate_phase(sys2, usr2, image=image)

        # ─── Phase 3: Produce final polished solution ──────────────────────
        sys3 = get_phase_system_prompt(domain, phase=2)
        usr3 = get_phase_user_prompt(
            problem, phase=2, domain=domain,
            ground_truth=ground_truth,
            solve_trace=solve_trace,
            verify_trace=verify_trace,
        )
        correct_trace = self._generate_phase(sys3, usr3, image=image)

        predicted_3 = extract_final_answer(correct_trace)
        correct_correct = answers_match(predicted_3, ground_truth)

        # Approximate token count
        total_tokens = (
            len(solve_trace) + len(verify_trace) + len(correct_trace)
        ) // 4

        return TraceExample(
            problem_id=problem_id,
            problem=problem,
            domain=domain,
            ground_truth=ground_truth,
            solve_trace=solve_trace,
            verify_trace=verify_trace,
            correct_trace=correct_trace,
            teacher_model=self.teacher_model_name,
            solve_correct=solve_correct,
            correct_correct=correct_correct,
            total_tokens=total_tokens,
        )

    def generate_dataset(
        self,
        problems: list[dict],
        domain: str,
        output_file: str,
        max_tokens_per_trace: int = 4096,
        cross_verify_domain: str = None,
    ) -> dict:
        """
        Generate traces for a batch of problems and write to JSONL.

        Args:
            problems: List of dicts with 'problem', 'answer', 'id' keys.
                      Optionally include 'image' for VL problems.
            domain: Primary PRISM domain.
            output_file: Output JSONL path.
            max_tokens_per_trace: Quality filter: discard traces above this.
            cross_verify_domain: If set, use this domain for Phase 2 (cross-verification).

        Returns:
            Stats dict: total, kept, pass_rate, phase1_correct, phase3_correct.
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        stats = {
            "domain": domain,
            "verifier_domain": cross_verify_domain or domain,
            "teacher": self.teacher_model_name,
            "total": len(problems),
            "phase1_correct": 0,
            "phase3_correct": 0,
            "token_filtered": 0,
            "kept": 0,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            for i, prob in enumerate(tqdm(problems, desc=f"Traces: {domain}")):
                try:
                    problem_text = prob.get("problem", prob.get("question", ""))
                    # reference_solution: full worked solution (e.g. MATH dataset's "solution" field)
                    # ground_truth: the short final answer for correctness checking
                    reference_solution = str(prob.get("solution", prob.get("reference", "")))
                    ground_truth = str(prob.get("answer", prob.get("ground_truth", reference_solution)))
                    prob_id = str(prob.get("id", i))
                    image = prob.get("image", None)

                    trace = self.generate_trace(
                        problem=problem_text,
                        domain=domain,
                        ground_truth=ground_truth,
                        reference_solution=reference_solution,
                        problem_id=prob_id,
                        image=image,
                        cross_verify_domain=cross_verify_domain,
                    )

                    if trace.solve_correct:
                        stats["phase1_correct"] += 1
                    if trace.correct_correct:
                        stats["phase3_correct"] += 1

                    if trace.total_tokens > max_tokens_per_trace:
                        stats["token_filtered"] += 1
                        continue

                    if trace.is_valid(max_tokens_per_trace):
                        f.write(trace.to_jsonl() + "\n")
                        stats["kept"] += 1

                except Exception as e:
                    logger.warning(f"Failed on problem {i}: {e}")
                    continue

                if (i + 1) % 50 == 0:
                    logger.info(
                        f"[{domain}] {i+1}/{len(problems)}: "
                        f"kept={stats['kept']}, "
                        f"p3_correct={stats['phase3_correct']}"
                    )

        stats["pass_rate"] = stats["kept"] / max(stats["total"], 1)
        logger.info(f"Generation complete [{domain}]: {stats}")
        return stats


class VLLMBatchGenerator:
    """
    vLLM-based batch trace generator — fast drop-in for TraceGenerator.

    Speed advantage:
      TraceGenerator  : N × 3 serial forward passes (one problem+phase at a time)
      VLLMBatchGenerator: 3 batched passes (all N Phase-1 prompts, then Phase-2, Phase-3)
      Practical speedup: 10-50× for large datasets (e.g. N=2500 per domain).

    Requirements:
      - vLLM installed with a torch/CUDA version compatible with this environment.
      - On CSCS GH200 (torch 2.9.0+cu129): standard `pip install vllm` is NOT safe
        because it downgrades torch. Build vLLM from source or use a CSCS module.

    Args:
        teacher_model_name: HF model ID (same as TraceGenerator).
        gpu_id: GPU device index. vLLM loads the model on this GPU.
        max_new_tokens: Max tokens per phase output.
        gpu_memory_utilization: Fraction of GPU VRAM for vLLM's KV cache (0.0-1.0).
                                 0.90 is safe for a 35B MoE on one GH200 (96GB).
    """

    def __init__(
        self,
        teacher_model_name: str = "Qwen/Qwen3.5-35B-A3B",
        gpu_id: int = 0,
        max_new_tokens: int = 1024,
        gpu_memory_utilization: float = 0.90,
    ):
        self.teacher_model_name = teacher_model_name
        self.gpu_id = gpu_id
        self.max_new_tokens = max_new_tokens
        self.gpu_mem_util = gpu_memory_utilization
        self._llm = None
        self._tokenizer = None
        self._sampling_params = None
        self._loaded = False

    def load(self):
        """Load model via vLLM. Raises ImportError if vLLM is not installed."""
        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:
            raise ImportError(
                "vLLM is not importable. It should be installed (0.19.0); "
                "if the environment was recreated, reinstall via:\n"
                "  pip install vllm==0.19.0\n"
                "  pip install torch==2.9.0+cu129 --extra-index-url https://download.pytorch.org/whl/cu129\n"
                "  pip install transformers==5.5.3 huggingface_hub==1.10.1 tokenizers==0.22.2\n"
                f"  Original error: {e}"
            ) from e

        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        from prism.model.backbone import _get_model_dir
        model_path = _get_model_dir(self.teacher_model_name)

        logger.info(f"Loading vLLM engine: {self.teacher_model_name} on GPU {self.gpu_id}")
        self._llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=self.gpu_mem_util,
            trust_remote_code=True,
            max_model_len=8192,
            enforce_eager=False,      # use CUDA graphs for speed
            enable_prefix_caching=True,  # cache shared system prompt prefixes
        )
        # Thinking mode sampling params (Qwen3 official recommended values)
        self._sampling_params = SamplingParams(
            temperature=1.0,
            top_p=0.95,
            top_k=20,
            presence_penalty=1.5,
            max_tokens=self.max_new_tokens,
        )
        self._tokenizer = self._llm.get_tokenizer()
        logger.info("vLLM engine loaded")
        self._loaded = True
        return self

    def _build_prompt(
        self,
        system_prompt: str,
        user_message: str,
    ) -> str:
        """Apply Qwen3 chat template with thinking mode enabled."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        try:
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        except TypeError:
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def generate_dataset(
        self,
        problems: list[dict],
        domain: str,
        output_file: str,
        max_tokens_per_trace: int = 4096,
        cross_verify_domain: str = None,
    ) -> dict:
        """
        Generate traces for all problems using 3 batched vLLM passes.

        Phase sequencing:
          Pass 1: all N Phase-1 (Reformulate) prompts → N solve_traces
          Pass 2: all N Phase-2 (Verify) prompts (built from Pass 1 outputs)
          Pass 3: all N Phase-3 (Correct) prompts (built from Pass 1+2 outputs)

        This is equivalent to TraceGenerator.generate_dataset() but 10-50× faster
        because vLLM continuous-batches all N sequences instead of running them serially.
        """
        assert self._loaded, "Call load() first"
        from vllm import SamplingParams

        verify_domain = cross_verify_domain if cross_verify_domain else domain
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        stats = {
            "domain": domain,
            "verifier_domain": verify_domain,
            "teacher": self.teacher_model_name,
            "backend": "vllm",
            "total": len(problems),
            "phase1_correct": 0,
            "phase3_correct": 0,
            "token_filtered": 0,
            "kept": 0,
        }

        # ── Unpack problems ───────────────────────────────────────────────
        problem_texts, reference_solutions, ground_truths, prob_ids = [], [], [], []
        for i, prob in enumerate(problems):
            problem_texts.append(prob.get("problem", prob.get("question", "")))
            reference_solutions.append(str(prob.get("solution", prob.get("reference", ""))))
            ground_truths.append(str(prob.get("answer", prob.get("ground_truth", ""))))
            prob_ids.append(str(prob.get("id", i)))

        logger.info(f"[vLLM] {domain}: generating {len(problems)} traces in 3 batched passes")

        # ── Pass 1: Reformulate (Phase 0) ─────────────────────────────────
        phase1_prompts = [
            self._build_prompt(
                get_phase_system_prompt(domain, phase=0),
                get_phase_user_prompt(
                    problem_texts[i], phase=0, domain=domain,
                    reference_solution=reference_solutions[i],
                    ground_truth=ground_truths[i],
                ),
            )
            for i in range(len(problems))
        ]
        logger.info(f"[vLLM] Pass 1 (Reformulate): {len(phase1_prompts)} prompts...")
        phase1_outputs = self._llm.generate(
            phase1_prompts,
            SamplingParams(
                temperature=1.0, top_p=0.95, top_k=20,
                presence_penalty=1.5, max_tokens=self.max_new_tokens,
            ),
        )
        solve_traces = [o.outputs[0].text.strip() for o in phase1_outputs]

        # ── Pass 2: Verify (Phase 1) — uses verify_domain expert ──────────
        phase2_prompts = [
            self._build_prompt(
                get_phase_system_prompt(verify_domain, phase=1),
                get_phase_user_prompt(
                    problem_texts[i], phase=1, domain=verify_domain,
                    ground_truth=ground_truths[i],
                    solve_trace=solve_traces[i],
                ),
            )
            for i in range(len(problems))
        ]
        logger.info(f"[vLLM] Pass 2 (Verify):     {len(phase2_prompts)} prompts...")
        phase2_outputs = self._llm.generate(
            phase2_prompts,
            SamplingParams(
                temperature=1.0, top_p=0.95, top_k=20,
                presence_penalty=1.5, max_tokens=self.max_new_tokens,
            ),
        )
        verify_traces = [o.outputs[0].text.strip() for o in phase2_outputs]

        # ── Pass 3: Correct/Polish (Phase 2) ──────────────────────────────
        phase3_prompts = [
            self._build_prompt(
                get_phase_system_prompt(domain, phase=2),
                get_phase_user_prompt(
                    problem_texts[i], phase=2, domain=domain,
                    ground_truth=ground_truths[i],
                    solve_trace=solve_traces[i],
                    verify_trace=verify_traces[i],
                ),
            )
            for i in range(len(problems))
        ]
        logger.info(f"[vLLM] Pass 3 (Correct):    {len(phase3_prompts)} prompts...")
        phase3_outputs = self._llm.generate(
            phase3_prompts,
            SamplingParams(
                temperature=1.0, top_p=0.95, top_k=20,
                presence_penalty=1.5, max_tokens=self.max_new_tokens,
            ),
        )
        correct_traces = [o.outputs[0].text.strip() for o in phase3_outputs]

        # ── Assemble and write TraceExamples ──────────────────────────────
        with open(output_path, "w", encoding="utf-8") as f:
            for i in range(len(problems)):
                predicted_1 = extract_final_answer(solve_traces[i])
                solve_correct = answers_match(predicted_1, ground_truths[i])
                predicted_3 = extract_final_answer(correct_traces[i])
                correct_correct = answers_match(predicted_3, ground_truths[i])

                if solve_correct:
                    stats["phase1_correct"] += 1
                if correct_correct:
                    stats["phase3_correct"] += 1

                total_tokens = (
                    len(solve_traces[i]) + len(verify_traces[i]) + len(correct_traces[i])
                ) // 4

                if total_tokens > max_tokens_per_trace:
                    stats["token_filtered"] += 1
                    continue

                trace = TraceExample(
                    problem_id=prob_ids[i],
                    problem=problem_texts[i],
                    domain=domain,
                    ground_truth=ground_truths[i],
                    solve_trace=solve_traces[i],
                    verify_trace=verify_traces[i],
                    correct_trace=correct_traces[i],
                    teacher_model=self.teacher_model_name,
                    solve_correct=solve_correct,
                    correct_correct=correct_correct,
                    total_tokens=total_tokens,
                )
                if trace.is_valid(max_tokens_per_trace):
                    f.write(trace.to_jsonl() + "\n")
                    stats["kept"] += 1

        stats["pass_rate"] = stats["kept"] / max(stats["total"], 1)
        logger.info(f"[vLLM] Generation complete [{domain}]: {stats}")
        return stats


def make_generator(
    teacher_model_name: str,
    gpu_id: int,
    use_vllm: bool = False,
    max_new_tokens: int = 1024,
    temperature: float = 1.0,
):
    """
    Factory: returns VLLMBatchGenerator if use_vllm=True and vLLM is importable,
    otherwise returns TraceGenerator (HF backend).

    Args:
        teacher_model_name: HF model ID.
        gpu_id: GPU device index.
        use_vllm: Attempt to use vLLM. Falls back to HF if unavailable.
        max_new_tokens: Max tokens per phase.
        temperature: Sampling temperature (vLLM backend ignores this in favour
                     of the fixed thinking-mode params).

    Returns:
        Loaded generator instance (either VLLMBatchGenerator or TraceGenerator).
    """
    if use_vllm:
        try:
            import vllm  # noqa: F401
            logger.info("vLLM available — using VLLMBatchGenerator (fast batched mode)")
            return VLLMBatchGenerator(
                teacher_model_name=teacher_model_name,
                gpu_id=gpu_id,
                max_new_tokens=max_new_tokens,
            ).load()
        except ImportError:
            logger.warning(
                "vLLM not available (ImportError). "
                "Falling back to HF TraceGenerator. "
                "To enable vLLM, build it from source against torch 2.9.0+cu129."
            )

    return TraceGenerator(
        teacher_model_name=teacher_model_name,
        gpu_id=gpu_id,
        max_new_tokens_per_phase=max_new_tokens,
        temperature=temperature,
    ).load()


def main():
    """CLI entry point: prism-traces"""
    parser = argparse.ArgumentParser(description="Generate PRISM expert traces")
    parser.add_argument(
        "--teacher",
        default="Qwen/Qwen3.5-35B-A3B",
        help="Teacher model HF ID (must be >14B; fallback: Qwen/Qwen3-VL-30B-A3B-Thinking)",
    )
    parser.add_argument(
        "--domain",
        required=True,
        choices=["algebra", "geometry", "combinatorics", "number_theory", "miscellaneous"],
    )
    parser.add_argument("--n-problems", type=int, default=2500)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-dir", default="results/traces")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--cross-verify-domain",
        default=None,
        choices=["algebra", "geometry", "combinatorics", "number_theory", "miscellaneous"],
        help="Use a DIFFERENT domain expert for Phase 2 (cross-domain verification)",
    )
    parser.add_argument(
        "--use-vllm",
        action="store_true",
        help=(
            "Use vLLM batch backend for 10-50× faster generation. "
            "Falls back to HF TraceGenerator if vLLM is not installed. "
            "WARNING: requires vLLM built against torch 2.9.0+cu129 (not pip-installable on CSCS)."
        ),
    )
    args = parser.parse_args()

    import os
    os.environ.setdefault("PRISM_ROOT", "/iopsstor/scratch/cscs/dasgupta/research/ideas/PRISM")
    hf_token_path = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(hf_token_path):
        with open(hf_token_path) as f:
            os.environ["HF_TOKEN"] = f.read().strip()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from prism.data.datasets import get_stage0_training_data
    logger.info(f"Loading {args.n_problems} problems for domain: {args.domain}")
    domain_data = get_stage0_training_data(n_per_domain=args.n_problems, domains=[args.domain])
    ds = domain_data[args.domain]

    problems = [
        {
            "problem": ex.get("problem", ex.get("question", "")),
            # Full worked solution passed as reference for Phase 1 reformulation
            "solution": ex.get("solution", ""),
            "answer": ex.get("answer", ex.get("solution", "")),
            "id": str(ex.get("id", "")),
        }
        for ex in ds
    ]

    generator = make_generator(
        teacher_model_name=args.teacher,
        gpu_id=args.gpu,
        use_vllm=args.use_vllm,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    suffix = f"_cv_{args.cross_verify_domain}" if args.cross_verify_domain else ""
    output_file = os.path.join(args.output_dir, f"{args.domain}{suffix}_traces.jsonl")
    stats = generator.generate_dataset(
        problems=problems,
        domain=args.domain,
        output_file=output_file,
        max_tokens_per_trace=args.max_tokens,
        cross_verify_domain=args.cross_verify_domain,
    )

    print(f"\nGeneration stats for {args.domain}:")
    print(json.dumps(stats, indent=2))

    stats_file = os.path.join(args.output_dir, f"{args.domain}{suffix}_stats.json")
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
