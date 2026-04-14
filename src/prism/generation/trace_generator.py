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

        # Restrict this process to the assigned GPU (physical → logical cuda:0)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Resolve model path (from /tmp or HF cache)
        from prism.model.backbone import _get_model_dir
        model_path = _get_model_dir(self.teacher_model_name)

        logger.info(f"Loading VL teacher: {self.teacher_model_name} on GPU {self.gpu_id}")

        # VL models need AutoModelForImageTextToText (Qwen3-VL, Qwen3.5, Qwen2.5-VL all use this)
        from transformers import AutoModelForImageTextToText, AutoProcessor, AutoConfig

        # Patch config before loading: Qwen3_5MoeConfig lacks pad_token_id which some
        # nn.Embedding calls require. Set it to eos_token_id if missing.
        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True,
                                                  token=hf_token)
        if not hasattr(model_config, "pad_token_id") or model_config.pad_token_id is None:
            model_config.pad_token_id = getattr(model_config, "eos_token_id", 0)
            logger.info(f"Patched missing pad_token_id → {model_config.pad_token_id}")

        model_kwargs = dict(
            config=model_config,
            dtype=self.dtype,       # note: use dtype not torch_dtype for newer transformers
            trust_remote_code=True,
            device_map="cuda:0",    # always cuda:0 since CUDA_VISIBLE_DEVICES restricts to one GPU
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

        # Thinking mode generation params (presence_penalty is OpenAI/vLLM only,
        # not supported by HF generate — use repetition_penalty as HF equivalent)
        generate_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,   # 1.0 for thinking mode
            top_p=0.95,
            top_k=20,
            repetition_penalty=1.05,        # mild repetition suppression (HF equivalent)
            pad_token_id=self._processor.tokenizer.eos_token_id,
        )
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
        torch.cuda.empty_cache()

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
        torch.cuda.empty_cache()

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
        # Use FlashInfer backend: bypasses vLLM's bundled flash_attn C extension
        # (compiled for CUDA 12, fails on CUDA 13 / GH200)
        os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASHINFER")
        # Force FA2 for VIT encoder: FA3 crashes on GH200 CUDA 13 during encoder profiling
        os.environ.setdefault("VLLM_FLASH_ATTN_VERSION", "2")

        from prism.model.backbone import _get_model_dir
        model_path = _get_model_dir(self.teacher_model_name)

        logger.info(f"Loading vLLM engine: {self.teacher_model_name} on GPU {self.gpu_id} "
                    f"(VLLM_ATTENTION_BACKEND={os.environ['VLLM_ATTENTION_BACKEND']})")
        self._llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=self.gpu_mem_util,
            trust_remote_code=True,
            max_model_len=8192,
            enforce_eager=True,          # skip CUDA graphs for CUDA 13 compatibility
            enable_prefix_caching=False, # prefix caching + VL model triggers encoder profiling
            limit_mm_per_prompt={"image": 0, "video": 0},  # suppress ALL multimodal profiling
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


class VLLMServerGenerator:
    """
    Trace generator that targets a running vLLM OpenAI-compatible server.

    All N prompts for each phase are fired concurrently as async HTTP requests.
    The server's continuous-batching scheduler handles queueing and optimal
    batching automatically — no EngineCore lifecycle issues, no OOM from
    competing processes, and better utilisation than the Python LLM API.

    Usage:
        bash scripts/start_vllm_servers.sh   # starts 4 servers on ports 8000-8003
        python -m prism.generation.trace_generator \\
            --domain algebra --server-url http://localhost:8000

    Args:
        teacher_model_name: HF model ID (used for tokenizer + metadata only).
        server_url: Base URL of the vLLM OpenAI-compatible server.
        max_new_tokens: Max tokens to generate per phase.
        concurrency: Max simultaneous in-flight HTTP requests (server still
                     batches them all; this only limits connection count).
    """

    def __init__(
        self,
        teacher_model_name: str = "Qwen/Qwen3.5-35B-A3B",
        server_url: str = "http://localhost:8000",
        max_new_tokens: int = 2048,
        concurrency: int = 512,
        negative_fraction: float = 0.3,
    ):
        self.teacher_model_name = teacher_model_name
        self.server_url = server_url.rstrip("/")
        self.max_new_tokens = max_new_tokens
        self.concurrency = concurrency
        self.negative_fraction = negative_fraction  # fraction of problems run free-solve
        self._model_id = None
        self._tokenizer = None
        self._loaded = False

    def load(self):
        """Connect to server and load tokenizer (CPU-only, fast)."""
        import urllib.request
        import json as _json

        # Discover the model ID served at this endpoint
        try:
            with urllib.request.urlopen(f"{self.server_url}/v1/models", timeout=15) as r:
                self._model_id = _json.loads(r.read())["data"][0]["id"]
            logger.info(f"vLLM server ready at {self.server_url} — model: {self._model_id}")
        except Exception as e:
            raise RuntimeError(
                f"Cannot reach vLLM server at {self.server_url}: {e}\n"
                "Start it with:  bash scripts/start_vllm_servers.sh"
            ) from e

        # Tokenizer only (no weights) — needed for chat-template formatting
        from prism.model.backbone import _get_model_dir
        model_path = _get_model_dir(self.teacher_model_name)
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        logger.info("Tokenizer loaded for prompt formatting")
        self._loaded = True
        return self

    def _make_messages(self, system: str, user: str) -> list[dict]:
        """Return OpenAI-format message list for chat completions."""
        return [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]

    @staticmethod
    def _extract_content(resp_choice) -> str:
        """
        Extract clean answer text from a chat completion choice.

        With Qwen3 thinking models on vLLM:
          - resp.choices[0].message.content  → answer after </think>  (what we want)
          - resp.choices[0].message.reasoning_content → thinking chain (discard)

        If reasoning_content is not supported by this vLLM version, fall back to
        splitting on </think> in content.
        """
        msg = resp_choice.message
        # Preferred: vLLM ≥0.6 exposes reasoning_content separately
        rc = getattr(msg, "reasoning_content", None)
        if rc is not None:
            # reasoning_content present → content is already clean
            return (msg.content or "").strip()

        # Fallback: thinking chain is inlined with </think> separator
        text = (msg.content or "").strip()
        if "</think>" in text:
            return text.split("</think>", 1)[1].strip()

        # Last resort: strip leading meta-commentary heuristically
        # (model narrated thinking without tags — extract after first blank line + header)
        import re as _re
        # Common patterns that end thinking and begin actual answer
        for anchor in [
            r"\n\*\*(?:Reformulation|Solution|Expert Solution|Phase [123]|Final Answer)\*\*",
            r"\n---+\n",
            r"\n\*\*(?:Step 1|1\.)\s+[A-Z][^*\n]{3,}\*\*\n",  # e.g. **Step 1: Setup**
        ]:
            m = _re.search(anchor, text)
            if m and m.start() > 200:   # thinking chain is >200 chars before this
                return text[m.start():].strip()

        return text

    async def _gather_completions(
        self, message_batches: list[list[dict]], label: str = "", max_tokens: int = None
    ) -> list[str]:
        """
        Fire all chat-completion requests concurrently; return answers in order.

        Uses chat completions API (not raw completions) so that Qwen3 thinking
        content is routed to reasoning_content / split on </think>, and
        resp.choices[0].message.content contains only the clean answer.
        """
        import asyncio
        from openai import AsyncOpenAI

        phase_max_tokens = max_tokens if max_tokens is not None else self.max_new_tokens

        client = AsyncOpenAI(
            base_url=f"{self.server_url}/v1",
            api_key="dummy",
            timeout=10800.0,
            max_retries=0,
        )
        sem = asyncio.Semaphore(self.concurrency)
        total = len(message_batches)
        done_count = 0
        lock = asyncio.Lock()

        async def one(messages: list[dict], idx: int) -> tuple[int, str]:
            nonlocal done_count
            async with sem:
                for attempt in range(3):
                    try:
                        resp = await client.chat.completions.create(
                            model=self._model_id,
                            messages=messages,
                            max_tokens=phase_max_tokens,
                            temperature=1.0,
                            top_p=0.95,
                            extra_body={
                                "top_k": 20,
                                "presence_penalty": 1.5,
                                # Disable thinking: Qwen3 inserts <think>\n</think>\n\n
                                # which closes the think block immediately, sending the
                                # model straight to its answer with no thinking narration.
                                # enable_thinking=True consumed all 4096 tokens on
                                # "Here's a thinking process..." meta-commentary instead
                                # of actual math.
                                "chat_template_kwargs": {"enable_thinking": False},
                            },
                        )
                        text = self._extract_content(resp.choices[0])
                        finish = resp.choices[0].finish_reason
                        async with lock:
                            done_count += 1
                            if done_count % max(1, total // 10) == 0 or done_count == total:
                                pct = 100 * done_count / total
                                logger.info(
                                    f"  [{label}] {done_count}/{total} ({pct:.0f}%) "
                                    f"— last finish_reason={finish}"
                                )
                        return idx, text
                    except Exception as e:
                        err_str = str(e)
                        if attempt == 2:
                            logger.warning(f"Request {idx} failed after 3 attempts: {e}")
                            return idx, ""
                        # Context overflow: truncate last user message content
                        if "maximum context length" in err_str or "input_tokens" in err_str:
                            msgs = list(messages)
                            if msgs and msgs[-1]["role"] == "user":
                                cur = msgs[-1]["content"]
                                msgs[-1] = {**msgs[-1], "content": cur[:int(len(cur)*0.8)]}
                                messages = msgs
                                logger.warning(
                                    f"Request {idx}: context overflow (attempt {attempt}), "
                                    f"truncating user prompt"
                                )
                                continue
                        await asyncio.sleep(2 ** attempt)
                return idx, ""

        results = await asyncio.gather(*[one(m, i) for i, m in enumerate(message_batches)])
        results.sort(key=lambda x: x[0])
        return [text for _, text in results]

    def _run_phase(self, message_batches: list[list[dict]], label: str, max_tokens: int = None) -> list[str]:
        """Synchronous wrapper: runs async gather in a fresh event loop."""
        import asyncio
        mt = max_tokens if max_tokens is not None else self.max_new_tokens
        logger.info(f"[Server→{self.server_url}] {label}: {len(message_batches)} concurrent requests")
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self._gather_completions(message_batches, label=label, max_tokens=mt)
            )
        finally:
            loop.close()

    def generate_dataset(
        self,
        problems: list[dict],
        domain: str,
        output_file: str,
        max_tokens_per_trace: int = 65536,
        cross_verify_domain: str = None,
    ) -> dict:
        """Same interface as VLLMBatchGenerator.generate_dataset()."""
        assert self._loaded, "Call load() first"

        verify_domain = cross_verify_domain or domain
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        stats = {
            "domain": domain,
            "verifier_domain": verify_domain,
            "teacher": self.teacher_model_name,
            "backend": "vllm_server",
            "server_url": self.server_url,
            "total": len(problems),
            "phase1_correct": 0,
            "phase3_correct": 0,
            "token_filtered": 0,
            "kept": 0,
        }

        problem_texts, reference_solutions, ground_truths, prob_ids = [], [], [], []
        for i, prob in enumerate(problems):
            problem_texts.append(prob.get("problem", prob.get("question", "")))
            reference_solutions.append(str(prob.get("solution", prob.get("reference", ""))))
            ground_truths.append(str(prob.get("answer", prob.get("ground_truth", ""))))
            prob_ids.append(str(prob.get("id", i)))

        N = len(problems)

        # ── Assign free-solve flag per problem ────────────────────────────
        # free_solve=True → no reference answer given → model attempts independently
        # → creates natural negative examples when model is wrong on hard problems
        import random as _random
        rng = _random.Random(42)
        is_free = [rng.random() < self.negative_fraction for _ in range(N)]
        n_free = sum(is_free)
        n_guided = N - n_free
        logger.info(
            f"Phase 1 split: {n_guided} guided (reference provided) + "
            f"{n_free} free-solve (no reference → potential negatives)"
        )

        # ── Pass 1: Reformulate (guided) or Free-Solve (no reference) ────
        # Pass message lists directly — chat completions API separates thinking
        # from answer, so resp.choices[0].message.content is clean math only.
        p1 = [
            self._make_messages(
                get_phase_system_prompt(domain, phase=0, free_solve=is_free[i]),
                get_phase_user_prompt(
                    problem_texts[i], phase=0, domain=domain,
                    reference_solution=reference_solutions[i],
                    ground_truth=ground_truths[i],
                    free_solve=is_free[i],
                ),
            )
            for i in range(N)
        ]
        # Per-phase token budgets — matching training collator budgets exactly:
        #   solve:   256 tokens  (5-15 lines of math, ends with \boxed{})
        #   verify:  128 tokens  (CORRECT/WRONG + ≤4 sentences of diagnosis)
        #   correct: 256 tokens  (final polished trace, ends with \boxed{})
        # These match the strict budgets in tokenize_full_trace() so training
        # sequences are never truncated mid-solution.
        P1_TOKENS = min(512, self.max_new_tokens)   # small headroom above 256 token target
        P2_TOKENS = min(256, self.max_new_tokens)   # headroom above 128 token target
        P3_TOKENS = min(512, self.max_new_tokens)   # headroom above 256 token target

        solve_traces = self._run_phase(p1, "Pass 1 (Reformulate/FreeSolve)", max_tokens=P1_TOKENS)

        # ── Pass 2: Verify ────────────────────────────────────────────────
        p2 = [
            self._make_messages(
                get_phase_system_prompt(verify_domain, phase=1),
                get_phase_user_prompt(
                    problem_texts[i], phase=1, domain=verify_domain,
                    ground_truth=ground_truths[i],
                    solve_trace=solve_traces[i],
                ),
            )
            for i in range(N)
        ]
        verify_traces = self._run_phase(p2, "Pass 2 (Verify)", max_tokens=P2_TOKENS)

        # ── Pass 3: Correct ───────────────────────────────────────────────
        p3 = [
            self._make_messages(
                get_phase_system_prompt(domain, phase=2),
                get_phase_user_prompt(
                    problem_texts[i], phase=2, domain=domain,
                    ground_truth=ground_truths[i],
                    solve_trace=solve_traces[i],
                    verify_trace=verify_traces[i],
                ),
            )
            for i in range(N)
        ]
        correct_traces = self._run_phase(p3, "Pass 3 (Correct)", max_tokens=P3_TOKENS)

        # ── Assemble and write ────────────────────────────────────────────
        with open(output_path, "w", encoding="utf-8") as f:
            for i in range(N):
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
                    free_solve=is_free[i],
                )
                if trace.is_valid(max_tokens_per_trace):
                    f.write(trace.to_jsonl() + "\n")
                    stats["kept"] += 1

        n_free_kept = sum(
            1 for t in _iter_jsonl(output_path) if t.get("free_solve") and not t.get("solve_correct")
        ) if output_path.exists() else 0
        stats["pass_rate"] = stats["kept"] / max(stats["total"], 1)
        stats["negative_examples"] = n_free_kept
        logger.info(f"[Server] Generation complete [{domain}]: {stats}")
        return stats


def _iter_jsonl(path):
    """Yield parsed dicts from a JSONL file (best-effort, skips bad lines)."""
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        pass
    except FileNotFoundError:
        return


def make_generator(
    teacher_model_name: str,
    gpu_id: int = 0,
    use_vllm: bool = False,
    server_url: str = None,
    max_new_tokens: int = 1024,
    temperature: float = 1.0,
    **kwargs,
):
    """
    Factory: returns the best available generator.

    Priority:
      1. VLLMServerGenerator  — if server_url is provided
      2. VLLMBatchGenerator   — if use_vllm=True and vLLM importable
      3. TraceGenerator       — HF fallback

    Args:
        teacher_model_name: HF model ID.
        gpu_id: GPU index (ignored when server_url is set).
        use_vllm: Use vLLM Python API (loads model in-process).
        server_url: URL of a running vLLM server (preferred over use_vllm).
        max_new_tokens: Max tokens per phase.
        temperature: Sampling temperature (vLLM backends use fixed thinking params).
    """
    if server_url:
        logger.info(f"Using vLLM server mode → {server_url}")
        return VLLMServerGenerator(
            teacher_model_name=teacher_model_name,
            server_url=server_url,
            max_new_tokens=max_new_tokens,
            negative_fraction=kwargs.get("negative_fraction", 0.3),
        ).load()

    if use_vllm:
        try:
            import vllm  # noqa: F401
            logger.info("vLLM available — using VLLMBatchGenerator (in-process)")
            return VLLMBatchGenerator(
                teacher_model_name=teacher_model_name,
                gpu_id=gpu_id,
                max_new_tokens=max_new_tokens,
            ).load()
        except Exception as e:
            logger.warning(
                f"vLLM load failed ({type(e).__name__}: {e}). "
                "Falling back to HF TraceGenerator."
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
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Max NEW tokens to generate per phase.")
    parser.add_argument("--filter-tokens", type=int, default=16384,
                        help="Max total tokens (all 3 phases combined) to keep a trace. "
                             "Traces exceeding this are discarded as too long for training. "
                             "Default 16384 ≈ 4096 tokens/phase × 4 chars/token.")
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
            "Use vLLM in-process batch backend. "
            "Prefer --server-url when a vLLM server is already running."
        ),
    )
    parser.add_argument(
        "--server-url",
        default=None,
        help=(
            "URL of a running vLLM OpenAI-compatible server, e.g. http://localhost:8000. "
            "All prompts are sent concurrently; the server handles batching. "
            "Start servers with:  bash scripts/start_vllm_servers.sh"
        ),
    )
    parser.add_argument(
        "--negative-fraction",
        type=float,
        default=0.3,
        help=(
            "Fraction of problems solved without the reference answer (free-solve mode). "
            "These create natural negative examples: Phase 1 may be wrong → "
            "Phase 2 catches it → Phase 3 corrects. "
            "Default 0.3 = 30%% free-solve, 70%% guided."
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
        server_url=args.server_url,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        negative_fraction=args.negative_fraction,
    )

    suffix = f"_cv_{args.cross_verify_domain}" if args.cross_verify_domain else ""
    output_file = os.path.join(args.output_dir, f"{args.domain}{suffix}_traces.jsonl")
    stats = generator.generate_dataset(
        problems=problems,
        domain=args.domain,
        output_file=output_file,
        max_tokens_per_trace=args.filter_tokens,
        cross_verify_domain=args.cross_verify_domain,
    )

    print(f"\nGeneration stats for {args.domain}:")
    print(json.dumps(stats, indent=2))

    stats_file = os.path.join(args.output_dir, f"{args.domain}{suffix}_stats.json")
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    # Flush any pending GPU memory
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


if __name__ == "__main__":
    main()
