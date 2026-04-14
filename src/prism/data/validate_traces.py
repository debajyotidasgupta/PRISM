"""
Trace quality validation for PRISM pilot traces.

Checks every trace in a JSONL file against a battery of criteria and
produces a per-domain quality report + per-trace breakdown.

Quality criteria:
  CONTAMINATION  — no meta-commentary (thinking chain leaking into content)
  MATH_CONTENT   — solve/correct traces contain actual math (LaTeX / equations)
  VERDICT        — verify_trace starts with CORRECT or WRONG (parseable)
  LENGTH         — solve_trace ≥ 30 chars, correct_trace ≥ 30 chars
  BOXED          — correct_trace contains \\boxed{} (final answer present)
  NON_EMPTY      — all three phases produced non-empty output

Any trace that fails CONTAMINATION or NON_EMPTY is flagged as BAD and excluded
from training. All other failures are WARN (trace is kept but noted).

Usage (CLI):
    python -m prism.data.validate_traces results/traces/pilot/algebra_traces.jsonl

Usage (API):
    from prism.data.validate_traces import validate_domain, print_report
    report = validate_domain('algebra', 'results/traces/pilot/algebra_traces.jsonl')
    print_report(report)
    assert report['bad_rate'] < 0.05, "Too many bad traces!"
"""

from __future__ import annotations
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── Contamination detection ───────────────────────────────────────────────────
_META_PATTERNS = [
    r"here'?s a thinking process",
    r"analyze the request",
    r"the user wants",
    r"i need to",
    r"thinking process:",
    r"\*\*task:\*\*\s+reformulat",
    r"\*\*role:\*\*\s+expert",
    r"let me break this down",
    r"step 1:.*analyze",
    r"first,? let me understand",
]
_META_RE = re.compile("|".join(_META_PATTERNS), re.IGNORECASE)

# ── Math content indicators ───────────────────────────────────────────────────
_MATH_PATTERNS = [
    r"\\[a-zA-Z]+\{",          # LaTeX command: \frac{, \sqrt{, etc.
    r"\$[^$]{3,}\$",           # inline math
    r"\$\$[^$]{3,}\$\$",      # display math
    r"\\boxed\{",              # boxed answer
    r"=\s*[-\d]",              # equation with number
    r"\d+\s*/\s*\d+",         # fraction
    r"\^\{?\d",                # exponent
    r"\\(?:frac|sqrt|sum|int|prod|lim|to|infty|cdot|times|div|le|ge|ne|equiv)\b",
]
_MATH_RE = re.compile("|".join(_MATH_PATTERNS))

_VERDICT_RE = re.compile(r"^\s*(CORRECT|WRONG)\b", re.IGNORECASE)
_BOXED_RE   = re.compile(r"\\boxed\s*\{")


@dataclass
class TraceResult:
    idx: int
    problem_id: str
    domain: str

    # Pass/fail per check (True = pass)
    non_empty:      bool = True
    length_ok:      bool = True
    not_contaminated: bool = True    # BAD if False
    has_math:       bool = True
    verdict_ok:     bool = True
    has_boxed:      bool = True
    free_solve:     bool = False

    # Raw metrics
    solve_len:   int = 0
    verify_len:  int = 0
    correct_len: int = 0

    @property
    def is_bad(self) -> bool:
        """Trace is bad (must exclude from training)."""
        return not self.non_empty or not self.not_contaminated

    @property
    def warnings(self) -> list[str]:
        w = []
        if not self.length_ok:   w.append("SHORT")
        if not self.has_math:    w.append("NO_MATH")
        if not self.verdict_ok:  w.append("BAD_VERDICT")
        if not self.has_boxed:   w.append("NO_BOXED")
        return w

    @property
    def status(self) -> str:
        if not self.non_empty:        return "BAD:EMPTY"
        if not self.not_contaminated: return "BAD:CONTAMINATED"
        if self.warnings:             return "WARN:" + ",".join(self.warnings)
        return "OK"


@dataclass
class DomainReport:
    domain: str
    file: str
    total: int = 0
    bad: int = 0
    warn: int = 0
    ok: int = 0
    contaminated: int = 0
    empty: int = 0
    no_math: int = 0
    bad_verdict: int = 0
    no_boxed: int = 0
    short: int = 0
    results: list[TraceResult] = field(default_factory=list)

    @property
    def bad_rate(self) -> float:
        return self.bad / max(self.total, 1)

    @property
    def usable(self) -> int:
        return self.total - self.bad


def _check_trace(idx: int, raw: dict) -> TraceResult:
    domain    = raw.get("domain", "unknown")
    prob_id   = str(raw.get("problem_id", idx))
    solve     = raw.get("solve_trace", "") or ""
    verify    = raw.get("verify_trace", "") or ""
    correct   = raw.get("correct_trace", "") or ""
    free      = bool(raw.get("free_solve", False))

    r = TraceResult(idx=idx, problem_id=prob_id, domain=domain, free_solve=free)
    r.solve_len   = len(solve)
    r.verify_len  = len(verify)
    r.correct_len = len(correct)

    # NON_EMPTY: all three phases must have produced something
    r.non_empty = bool(solve.strip()) and bool(verify.strip()) and bool(correct.strip())
    if not r.non_empty:
        return r   # bail early — other checks meaningless

    # CONTAMINATION: check solve_trace and correct_trace for thinking leakage
    r.not_contaminated = (
        not bool(_META_RE.search(solve[:500]))      # first 500 chars is the tell
        and not bool(_META_RE.search(correct[:500]))
    )

    # LENGTH: at least 30 chars each phase
    r.length_ok = r.solve_len >= 30 and r.correct_len >= 30

    # MATH_CONTENT: solve_trace or correct_trace must have LaTeX/equations
    r.has_math = bool(_MATH_RE.search(solve)) or bool(_MATH_RE.search(correct))

    # VERDICT: verify_trace must start with CORRECT or WRONG
    r.verdict_ok = bool(_VERDICT_RE.match(verify))

    # BOXED: correct_trace must contain \boxed{}
    r.has_boxed = bool(_BOXED_RE.search(correct))

    return r


def validate_domain(domain: str, filepath: str) -> DomainReport:
    """Validate all traces in a JSONL file. Returns DomainReport."""
    path = Path(filepath)
    report = DomainReport(domain=domain, file=str(path))

    if not path.exists():
        print(f"  WARNING: {filepath} does not exist", file=sys.stderr)
        return report

    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]

    for idx, line in enumerate(lines):
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue

        r = _check_trace(idx, raw)
        report.results.append(r)
        report.total += 1

        if r.is_bad:
            report.bad += 1
            if not r.non_empty:         report.empty += 1
            if not r.not_contaminated:  report.contaminated += 1
        elif r.warnings:
            report.warn += 1
        else:
            report.ok += 1

        if not r.has_math:    report.no_math    += 1
        if not r.verdict_ok:  report.bad_verdict += 1
        if not r.has_boxed:   report.no_boxed   += 1
        if not r.length_ok:   report.short      += 1

    return report


def print_report(report: DomainReport, verbose: bool = False) -> None:
    """Print a human-readable quality report."""
    t = report.total
    def pct(n): return f"{n*100//max(t,1)}%"

    print(f"\n{'='*60}")
    print(f"  {report.domain.upper()}  —  {report.file}")
    print(f"{'='*60}")
    print(f"  Total traces    : {t}")
    print(f"  Usable (non-bad): {report.usable:3d}  ({pct(report.usable)})")
    print(f"  BAD             : {report.bad:3d}  ({pct(report.bad)})")
    if report.contaminated:
        print(f"    ↳ contaminated: {report.contaminated:3d}")
    if report.empty:
        print(f"    ↳ empty phases: {report.empty:3d}")
    print(f"  WARN            : {report.warn:3d}  ({pct(report.warn)})")
    if report.no_math:
        print(f"    ↳ no math LaTeX: {report.no_math:3d}")
    if report.bad_verdict:
        print(f"    ↳ bad verdict  : {report.bad_verdict:3d}")
    if report.no_boxed:
        print(f"    ↳ no \\boxed{{}} : {report.no_boxed:3d}")
    if report.short:
        print(f"    ↳ too short    : {report.short:3d}")
    print(f"  OK              : {report.ok:3d}  ({pct(report.ok)})")

    if verbose:
        print(f"\n  Per-trace detail:")
        for r in report.results:
            if r.is_bad or r.warnings:
                print(f"    [{r.idx:3d}] {r.status:30s} "
                      f"solve={r.solve_len:5d}c  verify={r.verify_len:5d}c  "
                      f"correct={r.correct_len:5d}c  free={r.free_solve}")


def validate_all_domains(
    pilot_dir: str,
    domains: list[str] | None = None,
    min_usable: int = 50,
    max_bad_rate: float = 0.20,
    verbose: bool = False,
) -> dict[str, DomainReport]:
    """
    Validate all domain trace files in pilot_dir.

    Args:
        pilot_dir: Directory containing {domain}_traces.jsonl files.
        domains: Domains to check. Defaults to all 5 standard domains.
        min_usable: Minimum usable (non-bad) traces required per domain.
        max_bad_rate: Maximum fraction of bad traces before raising an error.
        verbose: Print per-trace details for bad/warn traces.

    Returns:
        Dict of domain → DomainReport.

    Raises:
        AssertionError if any domain fails quality gates.
    """
    if domains is None:
        domains = ["algebra", "geometry", "combinatorics", "number_theory", "miscellaneous"]

    reports = {}
    failures = []

    print(f"\n{'#'*60}")
    print(f"  TRACE QUALITY VALIDATION")
    print(f"  dir: {pilot_dir}")
    print(f"{'#'*60}")

    for domain in domains:
        fpath = str(Path(pilot_dir) / f"{domain}_traces.jsonl")
        report = validate_domain(domain, fpath)
        reports[domain] = report
        print_report(report, verbose=verbose)

        # Quality gates
        if report.usable < min_usable:
            failures.append(
                f"{domain}: only {report.usable} usable traces (need ≥{min_usable})"
            )
        if report.bad_rate > max_bad_rate:
            failures.append(
                f"{domain}: {report.bad_rate*100:.0f}% bad traces (threshold {max_bad_rate*100:.0f}%)"
            )

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    total_usable = sum(r.usable for r in reports.values())
    total_bad    = sum(r.bad    for r in reports.values())
    total        = sum(r.total  for r in reports.values())
    print(f"  Total across all domains: {total}")
    print(f"  Usable: {total_usable}  ({total_usable*100//max(total,1)}%)")
    print(f"  Bad:    {total_bad}  ({total_bad*100//max(total,1)}%)")

    if failures:
        print(f"\n  QUALITY GATE FAILURES ({len(failures)}):")
        for f in failures:
            print(f"    ✗ {f}")
        print()
        raise AssertionError(
            f"Trace validation failed: {len(failures)} domain(s) below quality threshold.\n"
            + "\n".join(failures)
        )
    else:
        print(f"\n  ✓ All domains pass quality gates.")

    return reports


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate PRISM pilot traces")
    parser.add_argument("files", nargs="*", help="JSONL trace files to validate")
    parser.add_argument("--dir", default=None, help="Validate all domains in this directory")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--min-usable", type=int, default=50)
    parser.add_argument("--max-bad-rate", type=float, default=0.20)
    args = parser.parse_args()

    if args.dir:
        validate_all_domains(
            args.dir,
            min_usable=args.min_usable,
            max_bad_rate=args.max_bad_rate,
            verbose=args.verbose,
        )
    elif args.files:
        for fpath in args.files:
            domain = Path(fpath).stem.replace("_traces", "")
            report = validate_domain(domain, fpath)
            print_report(report, verbose=args.verbose)
    else:
        parser.print_help()
