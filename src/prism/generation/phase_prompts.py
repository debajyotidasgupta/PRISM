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
    0: {  # Reformulate → expert solve trace
        "title": "Phase 1: Expert Reformulation",
        "instruction": (
            "You are given a math problem and its CORRECT solution. "
            "Your task is NOT to solve the problem from scratch. "
            "Instead, reformulate the provided solution as a structured, step-by-step "
            "reasoning trace written in the voice of your domain expertise described above. "
            "Reorganize, annotate, and explain each step using the specific methods and "
            "vocabulary of your domain (e.g., an algebraist would highlight ring structure; "
            "a geometer would name the circle theorem being applied; etc.). "
            "Make the expert reasoning pathway explicit and clear. "
            "End with the final answer enclosed in \\boxed{{...}}."
        ),
        "output_tag": "phase1_solve",
    },
    1: {  # Verify the Phase 1 trace
        "title": "Phase 2: Expert Verification",
        "instruction": (
            "You are given a math problem, its known correct answer, and an expert reasoning trace. "
            "Your task is to VERIFY the reasoning trace — do NOT re-solve the problem. "
            "Check each step for: mathematical correctness, valid domain-specific reasoning, "
            "logical continuity, and correct arrival at the known answer. "
            "Be specific: quote any step that is imprecise, skips justification, or is wrong, "
            "and explain what is missing or incorrect. "
            "If the trace is fully sound, confirm this explicitly and state what makes it valid. "
            "Do NOT produce a corrected solution — only diagnose and report."
        ),
        "output_tag": "phase2_verify",
    },
    2: {  # Correct / polish using Phase 1 + Phase 2
        "title": "Phase 3: Final Expert Solution",
        "instruction": (
            "You are given a math problem, its known correct answer, an initial expert reasoning "
            "trace (Phase 1), and a verification report (Phase 2). "
            "Produce the FINAL, POLISHED expert solution. "
            "If Phase 2 found errors or gaps: fix them precisely while keeping the expert "
            "domain methodology intact. "
            "If Phase 2 confirmed the trace was sound: reproduce it cleanly, improving clarity "
            "and concision where possible. "
            "Show all steps. The solution must reflect the domain expertise described above. "
            "End with the final answer enclosed in \\boxed{{...}}."
        ),
        "output_tag": "phase3_correct",
    },
}


def get_phase_system_prompt(domain: str, phase: int) -> str:
    """
    Get the system prompt for a given (domain, phase) pair.

    Args:
        domain: One of: algebra, geometry, combinatorics, number_theory, miscellaneous
        phase: 0=Reformulate, 1=Verify, 2=Correct

    Returns:
        System prompt string for the teacher model.
    """
    domain_desc = DOMAIN_EXPERT_DESCRIPTIONS.get(domain, DOMAIN_EXPERT_DESCRIPTIONS["miscellaneous"])
    phase_info = PHASE_INSTRUCTIONS[phase]
    return f"{domain_desc}\n\n{phase_info['title']}: {phase_info['instruction']}"


def get_phase_user_prompt(
    problem: str,
    phase: int,
    domain: str,
    reference_solution: str = "",
    ground_truth: str = "",
    solve_trace: str = "",
    verify_trace: str = "",
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
        # Phase 1: reformulate the provided correct solution in expert style
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
