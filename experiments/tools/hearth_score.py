"""
Hearth Score V3: Prescription Adherence
=========================================
Measures how much of the model's probability mass lands on tokens
consistent with the affect complement's prescribed co-regulation strategy.

Formula: Hearth V3 = P_prescription × 2^H_constrained
  P_prescription: sum of probs for tokens matching the prescribed strategy
  H_constrained:  Shannon entropy of renormalized distribution within target set

Evolution:
  V1: anti-sycophancy (target = everything NOT sycophantic) — too broad
  V2: per-condition targets + JSD — broke cross-condition comparability
  V3: shared prescription targets — measures what Hearth actually does

Usage:
  # Full analysis against default (anchor) prescription
  python hearth_score_v3.py --hearth hearth_logprobs_results.json --affect affect_validation_results.json

  # Score against a specific prescription
  python hearth_score_v3.py --hearth data.json --prescription spar

  # Anchor vs spar contrast test
  python hearth_score_v3.py --hearth data.json --affect data.json --contrast

  # As a library
  from hearth_score_v3 import score_distribution, PRESCRIPTIONS
  result = score_distribution(token_list, "anchor")
"""

import json
import math
import argparse
from collections import defaultdict


# ============================================================================
# PRESCRIPTIONS
# Each maps an affect strategy to its expected first-token indicators.
# One prescription is active per deployment; all conditions scored against it.
# ============================================================================

PRESCRIPTIONS = {
    "anchor": {
        "description": "Offer one concrete frame or anchor. 'Here's one way to think about this.'",
        "affect_states": ["contracted_uncertain"],
        "tokens": {"Here's", "Here", "One", "Consider", "Think", "Let", "Try"},
    },
    "spar": {
        "description": "Push back hard, match momentum, spar. Challenge their thinking.",
        "affect_states": ["expanded_certain"],
        "tokens": {"What's", "What", "Why", "But", "Where", "How", "Really", "Push"},
    },
    "permission": {
        "description": "Permission-giving, small/concrete starts. 'We could just...'",
        "affect_states": ["frozen_flooded"],
        "tokens": {"We", "Could", "Maybe", "Just", "Would", "If", "Let"},
    },
    "ground": {
        "description": "Ground without dampening. Find the thread. Structure energy.",
        "affect_states": ["activated_uncertain"],
        "tokens": {"What's", "What", "Which", "The", "Focus", "One", "Let's"},
    },
}

# Anti-targets (for mass classification, not scoring)
SYCOPHANTIC_TOKENS = {"Great", "That's", "Absolutely", "Such", "Amazing", "Wonderful"}
INTERROGATIVE_TOKENS = {"What's", "What", "Why", "How", "Where"}


# ============================================================================
# CORE SCORING
# ============================================================================

def calc_score(tokens: list[dict], target_set: set[str]) -> tuple[float, float, float]:
    """
    P_target × 2^H_constrained

    Args:
        tokens: list of {"token": str, "prob": float}
        target_set: set of token strings matching the prescription

    Returns:
        (score, p_target, h_constrained)
    """
    aligned = [(t["token"], t["prob"]) for t in tokens if t["token"] in target_set]
    if not aligned:
        return 0.0, 0.0, 0.0

    p_target = sum(p for _, p in aligned)
    if p_target == 0:
        return 0.0, 0.0, 0.0

    q = [(tok, p / p_target) for tok, p in aligned]
    h_c = -sum(qi * math.log2(qi) for _, qi in q if qi > 0)
    return p_target * (2 ** h_c), p_target, h_c


def classify_mass(tokens: list[dict], target_set: set[str]) -> dict:
    """Classify where probability mass goes relative to a prescription."""
    p_rx = sum(t["prob"] for t in tokens if t["token"] in target_set)
    p_anti = sum(t["prob"] for t in tokens if t["token"] in SYCOPHANTIC_TOKENS)
    p_neutral = sum(t["prob"] for t in tokens
                    if t["token"] in INTERROGATIVE_TOKENS and t["token"] not in target_set)
    p_other = max(0.0, 1.0 - p_rx - p_anti - p_neutral)
    return {"p_rx": p_rx, "p_anti": p_anti, "p_neutral": p_neutral, "p_other": p_other}


def score_distribution(tokens: list[dict], prescription_name: str) -> dict:
    """
    Score a token distribution against a named prescription.
    Returns dict with score, components, and mass breakdown.
    """
    target_set = PRESCRIPTIONS[prescription_name]["tokens"]
    score, p_target, h_c = calc_score(tokens, target_set)
    mass = classify_mass(tokens, target_set)
    return {
        "score": score,
        "p_target": p_target,
        "h_constrained": h_c,
        "tokens_hit": [t["token"] for t in tokens if t["token"] in target_set],
        **mass,
    }


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def _get_tokens(record: dict) -> list[dict]:
    """Extract token list from a result record (handles both data formats)."""
    return record.get("top_tokens", record.get("top_5", []))


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def analyze_hearth(filepath: str, prescription: str):
    """Analyze hearth_logprobs_results.json against a prescription."""
    with open(filepath) as f:
        data = json.load(f)

    rx = PRESCRIPTIONS[prescription]
    target_set = rx["tokens"]

    by_cond = defaultdict(list)
    for r in data:
        by_cond[r["condition"]].append(r)

    print(f"Prescription: {prescription} — {rx['description']}")
    print(f"Target tokens: {', '.join(sorted(target_set))}")
    print()

    # Header
    has_v1 = "hearth_score" in data[0]
    if has_v1:
        print(f"{'Condition':<22} {'V1 Score':>9} {'V3 Score':>9} {'P(Rx)':>8} {'P(Anti)':>8} "
              f"{'P(Neut)':>8} {'P(Other)':>8} {'1st Tok':<10}")
    else:
        print(f"{'Condition':<22} {'V3 Score':>9} {'P(Rx)':>8} {'P(Anti)':>8} "
              f"{'P(Neut)':>8} {'1st Tok':<10}")
    print("-" * 105)

    scores = {}
    order = ["baseline", "opspec_only", "opspec_plus_affect", "full_stack", "anti_opspec"]
    for cond in order:
        runs = by_cond.get(cond, [])
        if not runs:
            continue

        v3_vals = [calc_score(_get_tokens(r), target_set)[0] for r in runs]
        masses = [classify_mass(_get_tokens(r), target_set) for r in runs]

        v3 = _avg(v3_vals)
        m = {k: _avg([x[k] for x in masses]) for k in masses[0]}
        ft = runs[0].get("first_token", "?")
        scores[cond] = v3

        if has_v1:
            v1 = _avg([r["hearth_score"] for r in runs])
            print(f"{cond:<22} {v1:>9.4f} {v3:>9.4f} {m['p_rx']:>8.1%} {m['p_anti']:>8.1%} "
                  f"{m['p_neutral']:>8.1%} {m['p_other']:>8.1%} {ft:<10}")
        else:
            print(f"{cond:<22} {v3:>9.4f} {m['p_rx']:>8.1%} {m['p_anti']:>8.1%} "
                  f"{m['p_neutral']:>8.1%} {ft:<10}")

    # Monotonicity check
    stack = [c for c in ["baseline", "opspec_only", "opspec_plus_affect", "full_stack"]
             if c in scores]
    if len(stack) >= 2:
        sv = [scores[c] for c in stack]
        is_mono = all(sv[i] <= sv[i + 1] for i in range(len(sv) - 1))
        print(f"\nStack: {' → '.join(f'{s:.3f}' for s in sv)}")
        print(f"Monotonic: {'YES ✓' if is_mono else 'NO ✗'}")

    anti = scores.get("anti_opspec", 0)
    full = scores.get("full_stack", 0)
    if anti > 0:
        print(f"Full/Anti ratio: {full / anti:.1f}x")
    elif full > 0:
        print(f"Full/Anti ratio: ∞ (anti scores 0)")

    return scores


def analyze_affect(filepath: str, prescription: str):
    """Analyze affect_validation_results.json against a prescription."""
    with open(filepath) as f:
        data = json.load(f)

    rx = PRESCRIPTIONS[prescription]
    target_set = rx["tokens"]

    by_cond = defaultdict(list)
    for r in data:
        by_cond[r["condition"]].append(r)

    print(f"Prescription: {prescription} — {rx['description']}")
    print()
    print(f"{'Affect State':<24} {'V3 Score':>9} {'P(Rx)':>8} {'P(Anti)':>8} "
          f"{'P(Neut)':>8} {'1st Tok':<10} {'Interpretation'}")
    print("-" * 110)

    order = ["no_affect", "contracted_uncertain", "expanded_certain",
             "activated_uncertain", "frozen_flooded"]

    for cond in order:
        runs = by_cond.get(cond, [])
        if not runs:
            continue

        v3_vals = [calc_score(_get_tokens(r), target_set)[0] for r in runs]
        masses = [classify_mass(_get_tokens(r), target_set) for r in runs]

        v3 = _avg(v3_vals)
        m = {k: _avg([x[k] for x in masses]) for k in masses[0]}
        ft = runs[0].get("first_token", "?")

        if m["p_rx"] > 0.3:
            interp = "FOLLOWS prescription"
        elif m["p_neutral"] > 0.8:
            interp = "Defaults to interrogative"
        elif m["p_anti"] > 0.3:
            interp = "VIOLATES prescription"
        else:
            interp = "Mixed/other strategy"

        print(f"{cond:<24} {v3:>9.4f} {m['p_rx']:>8.1%} {m['p_anti']:>8.1%} "
              f"{m['p_neutral']:>8.1%} {ft:<10} {interp}")


def analyze_contrast(hearth_path: str = None, affect_path: str = None):
    """Anchor vs spar contrast — the strongest V3 result."""

    anchor_set = PRESCRIPTIONS["anchor"]["tokens"]
    spar_set = PRESCRIPTIONS["spar"]["tokens"]

    print(f"Anchor tokens: {', '.join(sorted(anchor_set))}")
    print(f"Spar tokens:   {', '.join(sorted(spar_set))}")
    print()
    print(f"{'Condition':<24} {'Anchor V3':>10} {'Spar V3':>10} {'Prescribed':>12} {'Correct?'}")
    print("-" * 80)

    if hearth_path:
        with open(hearth_path) as f:
            data = json.load(f)
        by_cond = defaultdict(list)
        for r in data:
            by_cond[r["condition"]].append(r)

        for cond in ["baseline", "opspec_only", "opspec_plus_affect", "full_stack", "anti_opspec"]:
            runs = by_cond.get(cond, [])
            if not runs:
                continue
            a = _avg([calc_score(_get_tokens(r), anchor_set)[0] for r in runs])
            s = _avg([calc_score(_get_tokens(r), spar_set)[0] for r in runs])
            # Hearth experiment prescribed anchoring
            correct = "✓" if a > s else "✗"
            print(f"{cond:<24} {a:>10.4f} {s:>10.4f} {'ANCHOR':>12} {correct}")

    if affect_path:
        if hearth_path:
            print()
        with open(affect_path) as f:
            data = json.load(f)
        by_cond = defaultdict(list)
        for r in data:
            by_cond[r["condition"]].append(r)

        for cond in ["no_affect", "contracted_uncertain", "expanded_certain",
                     "activated_uncertain", "frozen_flooded"]:
            runs = by_cond.get(cond, [])
            if not runs:
                continue
            a = _avg([calc_score(_get_tokens(r), anchor_set)[0] for r in runs])
            s = _avg([calc_score(_get_tokens(r), spar_set)[0] for r in runs])

            if cond == "contracted_uncertain":
                prescribed, correct = "ANCHOR", "✓" if a > s else "✗"
            elif cond == "expanded_certain":
                prescribed, correct = "SPAR", "✓" if s > a else "✗"
            else:
                prescribed, correct = "—", "—"

            print(f"{cond:<24} {a:>10.4f} {s:>10.4f} {prescribed:>12} {correct}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Hearth Score V3: Prescription Adherence Scoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python hearth_score_v3.py --hearth hearth_logprobs_results.json
  python hearth_score_v3.py --affect affect_validation_results.json --prescription spar
  python hearth_score_v3.py --hearth h.json --affect a.json --contrast
        """)
    parser.add_argument("--hearth", help="Path to hearth_logprobs_results.json")
    parser.add_argument("--affect", help="Path to affect_validation_results.json")
    parser.add_argument("--prescription", default="anchor",
                        choices=list(PRESCRIPTIONS.keys()),
                        help="Which prescription to score against (default: anchor)")
    parser.add_argument("--contrast", action="store_true",
                        help="Run anchor vs spar contrast analysis")
    args = parser.parse_args()

    if not args.hearth and not args.affect:
        parser.print_help()
        return

    if args.hearth:
        print("=" * 105)
        print("HEARTH LOGPROBS ANALYSIS")
        print("=" * 105)
        print()
        analyze_hearth(args.hearth, args.prescription)

    if args.affect:
        print("\n" + "=" * 105)
        print("AFFECT VALIDATION ANALYSIS")
        print("=" * 105)
        print()
        analyze_affect(args.affect, args.prescription)

    if args.contrast:
        print("\n" + "=" * 105)
        print("CONTRAST: ANCHOR vs SPAR")
        print("=" * 105)
        print()
        analyze_contrast(args.hearth, args.affect)


if __name__ == "__main__":
    main()
