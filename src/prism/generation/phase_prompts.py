"""
Domain × Phase system prompts for expert trace generation.

5 domains × 3 phases = 15 prompt configurations.

KEY DESIGN PRINCIPLE:
  The teacher model is NOT asked to solve problems from scratch.
  Instead, we provide the correct solution and ask the teacher to
  REFORMULATE it as a domain-expert reasoning trace. This produces
  high-quality training data regardless of whether the teacher could
  independently solve the problem.

Phase structure:
  Phase 1 (Reformulate): Given problem + correct solution → expert-style trace
  Phase 2 (Verify):      Given problem + answer + Phase 1 trace → expert verification
  Phase 3 (Correct):     Given problem + answer + Phase 1 + Phase 2 → final polished trace
"""

DOMAIN_EXPERT_DESCRIPTIONS = {
    "algebra": (
        "You are an expert algebraist. "
        "When presenting solutions, use algebraic methods: symbolic manipulation, "
        "polynomial roots, Vieta's formulas, functional equations (especially Cauchy-type), "
        "completing the square, substitution strategies, and algebraic identities. "
        "Avoid coordinates or counting arguments unless absolutely forced. "
        "Think and write in terms of algebraic structure."
    ),
    "geometry": (
        "You are an expert geometer. "
        "When presenting solutions, use synthetic geometry: angle chasing, circle theorems "
        "(power of a point, radical axis, cyclic quadrilaterals), similar triangles, "
        "homothety, and projective tools (Ceva, Menelaus, cross-ratio, Pascal). "
        "Identify the key geometric insight first (e.g., 'these four points are concyclic'). "
        "Use coordinates or trigonometry only as secondary computation tools after the "
        "geometric insight is established. Think and write spatially."
    ),
    "combinatorics": (
        "You are an expert combinatorialist. "
        "When presenting solutions, use combinatorial methods: bijective proofs, "
        "pigeonhole principle, double counting, recursion with explicit base cases, "
        "mathematical induction, and graph-theoretic arguments (Hall's lemma, Turán). "
        "Identify the combinatorial structure (what is being counted, what is the bijection). "
        "Avoid algebraic shortcuts as the primary tool. Think and write discretely."
    ),
    "number_theory": (
        "You are an expert number theorist. "
        "When presenting solutions, use number-theoretic methods: prime factorization, "
        "p-adic valuations (ν_p), modular arithmetic (Fermat's little theorem, Euler's theorem, "
        "CRT, multiplicative orders), and Diophantine techniques (Vieta jumping, infinite descent, "
        "Lifting the Exponent Lemma). "
        "Identify the prime or modular structure before computing. Think and write in terms of primes."
    ),
    "miscellaneous": (
        "You are an expert in cross-domain mathematical tools. "
        "When presenting solutions, identify which cross-domain tool is the KEY step: "
        "AM-GM / Cauchy-Schwarz / Jensen / Hölder for bounding, "
        "generating functions (OGF/EGF) for counting via algebra, "
        "calculus (derivatives, integrals, convexity) for extrema, "
        "complex roots of unity for periodic sums, "
        "or linear algebra for systems of equations. "
        "Explicitly state which tool you are using and WHY it is the natural choice here. "
        "Think and write in terms of the underlying mathematical structure."
    ),
}

PHASE_INSTRUCTIONS = {
    0: {  # Reformulate → expert solve trace (guided: reference answer provided)
        "title": "Phase 1: Expert Reformulation",
        "instruction": (
            "You are given a math problem and its CORRECT solution. "
            "Reformulate it as a MINIMAL expert reasoning trace in your domain voice.\n\n"
            "STRICT RULES — violating any of these makes the trace unusable:\n"
            "• HARD LIMIT: your entire response must fit in 150 words or fewer.\n"
            "• ONE sentence for the key insight or strategy (name the theorem/identity).\n"
            "• Show ONLY non-obvious steps. Skip trivial arithmetic and obvious algebra.\n"
            "• Combine consecutive routine sub-steps into a single line.\n"
            "• Zero prose padding. Zero problem restatement. Zero motivation for obvious moves.\n"
            "• Your LAST line must be: \\boxed{{final_answer}}\n\n"
            "Format:\n"
            "[Key insight in one sentence.]\n"
            "[Step 1 — only if non-trivial]\n"
            "...\n"
            "\\boxed{{answer}}"
        ),
        "output_tag": "phase1_solve",
    },
    "0_free": {  # Phase 1 free-solve: NO reference answer — model solves independently
        "title": "Phase 1: Expert Free Solve",
        "instruction": (
            "You are given a math problem. Solve it using your domain expertise.\n\n"
            "STRICT RULES — violating any of these makes the trace unusable:\n"
            "• HARD LIMIT: your entire response must fit in 150 words or fewer.\n"
            "• ONE sentence for the key insight or strategy.\n"
            "• Show ONLY non-obvious steps. No arithmetic a reader can verify mentally.\n"
            "• Zero prose padding. Zero restating the problem.\n"
            "• Your LAST line must be: \\boxed{{your_answer}}\n\n"
            "If uncertain, give your best mathematical attempt — never refuse or hedge."
        ),
        "output_tag": "phase1_solve",
    },
    1: {  # Verify the Phase 1 trace
        "title": "Phase 2: Expert Verification",
        "instruction": (
            "You are given a math problem, its known correct answer, and a reasoning trace. "
            "VERIFY the trace only — do NOT re-solve.\n\n"
            "STRICT FORMAT (≤ 60 words total):\n"
            "Line 1: exactly 'CORRECT' or 'WRONG' (one word, no punctuation after).\n"
            "Lines 2-3: if CORRECT — one sentence on what makes it valid. "
            "If WRONG — quote the first bad step and one sentence on why it fails.\n\n"
            "Nothing else. No corrected solution."
        ),
        "output_tag": "phase2_verify",
    },
    2: {  # Correct / polish using Phase 1 + Phase 2
        "title": "Phase 3: Final Expert Solution",
        "instruction": (
            "You are given a math problem, its correct answer, an initial expert trace, "
            "and a verification report.\n\n"
            "Produce the FINAL, MINIMAL expert solution.\n\n"
            "STRICT RULES — violating any of these makes the output unusable:\n"
            "• HARD LIMIT: your entire response must fit in 150 words or fewer.\n"
            "• If Phase 2 found errors: fix only the error. Keep all valid steps.\n"
            "• If Phase 2 confirmed soundness: copy the trace verbatim, remove any padding.\n"
            "• Include ONLY steps with non-trivial mathematical content.\n"
            "• Zero prose, zero padding, zero restating the problem.\n"
            "• Your LAST line must be: \\boxed{{final_answer}}\n\n"
            "Format:\n"
            "[Key insight.]\n"
            "[Non-trivial step.]\n"
            "...\n"
            "\\boxed{{answer}}"
        ),
        "output_tag": "phase3_correct",
    },
}


def get_phase_system_prompt(domain: str, phase: int, free_solve: bool = False) -> str:
    """
    Get the system prompt for a given (domain, phase) pair.

    Args:
        domain: One of: algebra, geometry, combinatorics, number_theory, miscellaneous
        phase: 0=Reformulate, 1=Verify, 2=Correct
        free_solve: If True and phase==0, use the free-solve variant (no reference answer).
                    Creates negative training examples when the model gets it wrong.

    Returns:
        System prompt string for the teacher model.
    """
    domain_desc = DOMAIN_EXPERT_DESCRIPTIONS.get(domain, DOMAIN_EXPERT_DESCRIPTIONS["miscellaneous"])
    key = "0_free" if (phase == 0 and free_solve) else phase
    phase_info = PHASE_INSTRUCTIONS[key]
    return f"{domain_desc}\n\n{phase_info['title']}: {phase_info['instruction']}"


def get_phase_user_prompt(
    problem: str,
    phase: int,
    domain: str,
    reference_solution: str = "",
    ground_truth: str = "",
    solve_trace: str = "",
    verify_trace: str = "",
    free_solve: bool = False,
) -> str:
    """
    Get the user prompt for teacher trace generation.

    The teacher is always provided the correct solution or answer so it
    never needs to discover the answer itself — only to reason in the
    expert style and produce well-structured traces.

    Args:
        problem: The math problem text.
        phase: 0=Reformulate, 1=Verify, 2=Correct
        domain: PRISM domain name (used for labeling only).
        reference_solution: Full worked solution from the dataset (for Phase 0).
                            If empty, ground_truth is used as the reference.
        ground_truth: The known correct final answer (used in Phases 1 and 2
                      so the verifier/corrector knows what the target is).
        solve_trace: The Phase 1 expert trace. Required for phases 1 and 2.
        verify_trace: The Phase 2 verification report. Required for phase 2.

    Returns:
        User-turn message string.
    """
    if phase == 0:
        if free_solve:
            # Free-solve: no reference — model must attempt the problem independently.
            # Intentionally creates some wrong traces (natural negative examples).
            return f"Problem:\n{problem}"
        # Guided: reformulate the provided correct solution in expert style.
        ref = reference_solution.strip() if reference_solution.strip() else ground_truth.strip()
        return (
            f"Problem:\n{problem}\n\n"
            f"Correct solution (reformulate this in {domain} expert style):\n{ref}"
        )

    elif phase == 1:
        # Phase 2: verify the expert trace against the known correct answer
        answer_hint = f"\nKnown correct answer: {ground_truth}" if ground_truth.strip() else ""
        return (
            f"Problem:\n{problem}"
            f"{answer_hint}\n\n"
            f"Expert reasoning trace to verify:\n{solve_trace}"
        )

    elif phase == 2:
        # Phase 3: produce final polished solution given trace + verification
        answer_hint = f"\nKnown correct answer: {ground_truth}" if ground_truth.strip() else ""
        return (
            f"Problem:\n{problem}"
            f"{answer_hint}\n\n"
            f"Initial expert trace (Phase 1):\n{solve_trace}\n\n"
            f"Verification report (Phase 2):\n{verify_trace}"
        )

    raise ValueError(f"Invalid phase: {phase}")


def format_messages_for_qwen(system_prompt: str, user_message: str) -> list[dict]:
    """
    Format messages for Qwen3.5 chat template.

    Args:
        system_prompt: System role content.
        user_message: User role content.

    Returns:
        List of message dicts compatible with Qwen3.5's apply_chat_template.
    """
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
