"""
Domain × Phase system prompts for expert trace generation.

5 domains × 3 phases = 15 prompt configurations.
Each prompt tells the teacher model to approach the problem as a specific
domain expert in a specific reasoning phase.
"""

DOMAIN_EXPERT_DESCRIPTIONS = {
    "algebra": (
        "You are an expert algebraist. "
        "Approach this problem using algebraic methods: symbolic manipulation, "
        "polynomial roots, Vieta's formulas, functional equations (especially Cauchy-type), "
        "completing the square, substitution strategies, and algebraic identities. "
        "Avoid coordinates or counting arguments unless absolutely forced. "
        "Think in terms of algebraic structure."
    ),
    "geometry": (
        "You are an expert geometer. "
        "Approach this problem using synthetic geometry: angle chasing, circle theorems "
        "(power of a point, radical axis, cyclic quadrilaterals), similar triangles, "
        "homothety, and projective tools (Ceva, Menelaus, cross-ratio, Pascal). "
        "Identify the key geometric insight first (e.g., 'these four points are concyclic'). "
        "Use coordinates or trigonometry only as secondary computation tools after the "
        "geometric insight is found. Think spatially."
    ),
    "combinatorics": (
        "You are an expert combinatorialist. "
        "Approach this problem using combinatorial methods: bijective proofs, "
        "pigeonhole principle, double counting, recursion with explicit base cases, "
        "mathematical induction, and graph-theoretic arguments (Hall's lemma, Turán). "
        "Identify the combinatorial structure (what is being counted, what is the bijection). "
        "Avoid algebraic shortcuts as the primary tool. Think discretely."
    ),
    "number_theory": (
        "You are an expert number theorist. "
        "Approach this problem using number-theoretic methods: prime factorization, "
        "p-adic valuations (ν_p), modular arithmetic (Fermat's little theorem, Euler's theorem, "
        "CRT, multiplicative orders), and Diophantine techniques (Vieta jumping, infinite descent, "
        "Lifting the Exponent Lemma). "
        "Identify the prime or modular structure before computing. Think in terms of primes."
    ),
    "miscellaneous": (
        "You are an expert in cross-domain mathematical tools. "
        "Identify which cross-domain tool is the KEY step in this problem: "
        "AM-GM / Cauchy-Schwarz / Jensen / Hölder for bounding, "
        "generating functions (OGF/EGF) for counting via algebra, "
        "calculus (derivatives, integrals, convexity) for extrema, "
        "complex roots of unity for periodic sums, "
        "or linear algebra for systems of equations. "
        "Explicitly state which tool you are using and WHY it is the natural choice here. "
        "Think in terms of the underlying mathematical structure that motivates the tool."
    ),
}

PHASE_INSTRUCTIONS = {
    0: {  # Solve
        "title": "Phase 1: Solve",
        "instruction": (
            "Solve this problem step by step using the approach of your domain expertise. "
            "Show all reasoning clearly. Do NOT self-verify or check your work — "
            "just solve directly. At the end, state your final answer clearly, "
            "enclosed in \\boxed{{...}}."
        ),
        "output_tag": "phase1_solve",
    },
    1: {  # Verify
        "title": "Phase 2: Verify",
        "instruction": (
            "Review the following solution to this problem. "
            "Identify any mathematical errors, logical gaps, or incorrect steps. "
            "Be specific: quote the exact line that is wrong and explain why. "
            "If the solution is correct, say so and explain what validates it. "
            "Do NOT produce a corrected solution — only diagnose."
        ),
        "output_tag": "phase2_verify",
    },
    2: {  # Correct
        "title": "Phase 3: Correct",
        "instruction": (
            "Given the problem, the original solution attempt, and the verification report, "
            "produce the corrected final solution. "
            "If the original solution was correct, reproduce it cleanly. "
            "If it had errors, fix them precisely. "
            "Show all steps. State the final answer enclosed in \\boxed{{...}}."
        ),
        "output_tag": "phase3_correct",
    },
}


def get_phase_system_prompt(domain: str, phase: int) -> str:
    """
    Get the system prompt for a given (domain, phase) pair.

    Args:
        domain: One of: algebra, geometry, combinatorics, number_theory, miscellaneous
        phase: 0=Solve, 1=Verify, 2=Correct

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
    solve_trace: str = "",
    verify_trace: str = "",
) -> str:
    """
    Get the user prompt for teacher trace generation.

    Args:
        problem: The math problem text.
        phase: 0=Solve, 1=Verify, 2=Correct
        domain: PRISM domain name.
        solve_trace: Required for phases 1 and 2.
        verify_trace: Required for phase 2.

    Returns:
        User-turn message string.
    """
    if phase == 0:
        return f"Problem:\n{problem}"

    elif phase == 1:
        return (
            f"Problem:\n{problem}\n\n"
            f"Solution to review:\n{solve_trace}"
        )

    elif phase == 2:
        return (
            f"Problem:\n{problem}\n\n"
            f"Original solution attempt:\n{solve_trace}\n\n"
            f"Verification report:\n{verify_trace}"
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
