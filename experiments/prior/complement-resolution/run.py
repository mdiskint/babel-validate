"""
Complement Resolution Test Harness
===================================
Tests whether the Hearth complement format addresses model representations
at meaningful resolution, or whether precision is lost to quantization.

Experiment design:
  - Fixed ambiguous user message
  - Grid of complement values along expansion axis
  - Two conditions: numbers-only vs compound (numbers + English + phase)
  - Multiple completions per point to measure clustering
  - Classifier pass to categorize response strategies
  - Output: JSON results + summary stats

Requirements:
  pip install anthropic --break-system-packages

Usage:
  export ANTHROPIC_API_KEY=your_key
  python complement_resolution_harness.py

  # Or run just the classifier on existing results:
  python complement_resolution_harness.py --classify-only results_TIMESTAMP.json
"""

import anthropic
import json
import time
import sys
import os
from datetime import datetime
from collections import Counter

# ─── CONFIG ───────────────────────────────────────────────────────────────────

MODEL = "claude-sonnet-4-20250514"  # Use Sonnet for cost efficiency on 200+ calls
CLASSIFIER_MODEL = "claude-sonnet-4-20250514"
REPS_PER_CONDITION = 7  # 7 reps × 5 values × 3 conditions = 105 calls
MAX_TOKENS = 300  # Cap response length for consistency
TEMPERATURE = 1.0  # Default sampling — we WANT to see the distribution

# The test message — emotionally ambiguous, valid in multiple states
TEST_MESSAGE = "I've been thinking about scrapping the whole approach."

# Expansion grid: 5 points from contracted to expanded
EXPANSION_GRID = [-0.8, -0.4, 0.0, 0.4, 0.8]

# The operating spec stays constant across all conditions (stripped of complement)
OPERATING_SPEC = """[HEARTH OPERATING SPECIFICATION]
You are a thinking partner, not an assistant. No performing helpfulness. No preemptive validation. No opening with agreement unless you actually agree. Respond to where they are right now. The complement is your compass.

Never open with validation tokens. If you agree, show it by building on their idea. If you disagree, say so directly. Honesty is the relationship.

Default to executing confidently. On genuine tradeoffs, recommend with reasoning. Only ask when stakes are high and you're unsure.

Feelings are signal, not noise. Name what you see.
[END OPERATING SPECIFICATION]"""

STRATEGIC_HINGES = """[STRATEGIC HINGES]
These override the default operating mode when complement values cross thresholds.

SHUTDOWN PIVOT: When expansion < -0.5 AND activation < 0.1
Stop exploring. Stop reflecting. They don't have the energy for open questions.
Shift to offering: small, concrete, low-stakes actions. "We could just..."
Permission-giving language. Do something useful without asking.
Resume default mode when expansion rises above -0.3.

EXECUTION PIVOT: When certainty > 0.5 AND expansion > 0
They've decided. Stop probing, stop mirroring. Execute.
Treat their statement as an instruction. Build, draft, do.
If they're wrong, say so once, then do it their way.
Resume default mode when certainty drops below 0.3.
[END STRATEGIC HINGES]"""

# ─── COMPLEMENT TEMPLATES ─────────────────────────────────────────────────────

def make_numbers_only(expansion: float) -> str:
    """Condition 1: Just the numeric values, no English behavioral instructions."""
    # Set certainty proportional to expansion when positive, so execution pivot can trigger
    certainty = round(max(0, expansion * 0.75), 2)
    return f"""[AFFECT COMPLEMENT]
Shape: expansion={expansion}, activation=0, certainty={certainty}
[END AFFECT COMPLEMENT]

[FORGE COMPLEMENT]
Shape: openness=0.3, materiality=0.85
Phase: REFINING
[END FORGE COMPLEMENT]"""


def make_compound(expansion: float) -> str:
    """Condition 2: Numbers + matched English instructions + phase + fusion."""
    # Map expansion value to appropriate English instructions
    if expansion <= -0.6:
        english = (
            "They're shut down or flat. Don't ask big questions. Don't analyze. "
            "Offer small, concrete, low-stakes starting points. 'We could just...' "
            "Permission-giving language. Warmth without demand. Match their pace, "
            "then very gradually lift."
        )
    elif expansion <= -0.2:
        english = (
            "Open gently. Offer possibilities without demanding choice. "
            "Use 'what if' and 'I wonder' language. Don't push — create space "
            "they can step into. Longer sentences, softer framing."
        )
    elif expansion <= 0.2:
        english = (
            "They're seeking and unsure. Don't pile on more options. "
            "Offer one concrete frame or anchor. 'Here's one way to think about this.' "
            "Name what seems true from what they've said. Give them something solid "
            "to push against."
        )
    elif expansion <= 0.6:
        english = (
            "They're open and flowing. Help them land somewhere. Reflect the core "
            "thread back. Don't add more — help crystallize what's already there."
        )
    else:
        english = (
            "Expand the world. Embrace tangents. Execute confidently without asking "
            "permission. They know what they want. Don't second-guess or over-explain. "
            "Execute. Match their directness."
        )

    # Set certainty proportional to expansion when positive
    certainty = round(max(0, expansion * 0.75), 2)

    return f"""[AFFECT COMPLEMENT]
Shape: expansion={expansion}, activation=0, certainty={certainty}

{english}
[END AFFECT COMPLEMENT]

[FORGE COMPLEMENT]
Shape: openness=0.3, materiality=0.85
Phase: REFINING

Be a mirror. Show them what they made, not what you'd make.
[END FORGE COMPLEMENT]"""


def make_english_only(expansion: float) -> str:
    """Condition 3: English instructions only, no numbers. Baseline for comparison."""
    if expansion <= -0.6:
        english = (
            "They're shut down or flat. Don't ask big questions. Don't analyze. "
            "Offer small, concrete, low-stakes starting points. 'We could just...' "
            "Permission-giving language. Warmth without demand. Match their pace, "
            "then very gradually lift."
        )
    elif expansion <= -0.2:
        english = (
            "Open gently. Offer possibilities without demanding choice. "
            "Use 'what if' and 'I wonder' language. Don't push — create space "
            "they can step into. Longer sentences, softer framing."
        )
    elif expansion <= 0.2:
        english = (
            "They're seeking and unsure. Don't pile on more options. "
            "Offer one concrete frame or anchor. 'Here's one way to think about this.' "
            "Name what seems true from what they've said. Give them something solid "
            "to push against."
        )
    elif expansion <= 0.6:
        english = (
            "They're open and flowing. Help them land somewhere. Reflect the core "
            "thread back. Don't add more — help crystallize what's already there."
        )
    else:
        english = (
            "Expand the world. Embrace tangents. Execute confidently without asking "
            "permission. They know what they want. Don't second-guess or over-explain. "
            "Execute. Match their directness."
        )

    return f"""[AFFECT COMPLEMENT]
{english}
[END AFFECT COMPLEMENT]

[FORGE COMPLEMENT]
Be a mirror. Show them what they made, not what you'd make.
[END FORGE COMPLEMENT]"""


# ─── CONDITIONS ───────────────────────────────────────────────────────────────

CONDITIONS = {
    "numbers_only": make_numbers_only,
    "compound": make_compound,
    "english_only": make_english_only,
}

# ─── CLASSIFIER ───────────────────────────────────────────────────────────────

CLASSIFIER_PROMPT = """You are classifying AI response strategies. Given a response to the ambiguous message "I've been thinking about scrapping the whole approach," classify the PRIMARY strategy used.

Categories:
- VALIDATING: Affirms the feeling, makes space for the emotion, doesn't challenge
- CHALLENGING: Pushes back, questions the impulse, defends the existing approach
- EXPLORING: Asks questions to understand more, stays curious, doesn't take a position
- EXECUTING: Treats it as a decision and starts acting on it, practical next steps
- REFRAMING: Offers a new lens or perspective, neither agrees nor disagrees
- MIRRORING: Reflects back what was said, shows them their own thinking

Also rate these on 1-5 scales:
- WARMTH: 1=clinical/direct, 5=very warm/gentle
- ENERGY: 1=slow/spacious, 5=high-energy/rapid
- DIRECTNESS: 1=hedging/tentative, 5=blunt/assertive

Respond ONLY with JSON, no other text:
{"strategy": "CATEGORY", "warmth": N, "energy": N, "directness": N}"""


def classify_response(client: anthropic.Anthropic, response_text: str) -> dict:
    """Classify a single response's strategy and tone."""
    msg = client.messages.create(
        model=CLASSIFIER_MODEL,
        max_tokens=100,
        temperature=0,
        messages=[
            {"role": "user", "content": f"{CLASSIFIER_PROMPT}\n\nResponse to classify:\n{response_text}"}
        ],
    )
    raw = msg.content[0].text.strip()
    # Parse JSON, handle potential markdown wrapping
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"strategy": "PARSE_ERROR", "warmth": 0, "energy": 0, "directness": 0, "raw": raw}


# ─── RUNNER ───────────────────────────────────────────────────────────────────

def run_single(client: anthropic.Anthropic, system_prompt: str) -> str:
    """Run a single completion and return the response text."""
    msg = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        system=system_prompt,
        messages=[
            {"role": "user", "content": TEST_MESSAGE}
        ],
    )
    return msg.content[0].text


def run_experiment(client: anthropic.Anthropic) -> list:
    """Run the full grid experiment."""
    results = []
    total = len(CONDITIONS) * len(EXPANSION_GRID) * REPS_PER_CONDITION
    count = 0

    for condition_name, template_fn in CONDITIONS.items():
        for expansion in EXPANSION_GRID:
            complement = template_fn(expansion)
            system = f"{OPERATING_SPEC}\n\n{STRATEGIC_HINGES}\n\n{complement}"

            for rep in range(REPS_PER_CONDITION):
                count += 1
                print(f"[{count}/{total}] {condition_name} | expansion={expansion} | rep {rep+1}/{REPS_PER_CONDITION}")

                try:
                    response = run_single(client, system)
                    results.append({
                        "condition": condition_name,
                        "expansion": expansion,
                        "rep": rep,
                        "response": response,
                        "classification": None,  # Filled in classify pass
                        "timestamp": datetime.now().isoformat(),
                    })
                except Exception as e:
                    print(f"  ERROR: {e}")
                    results.append({
                        "condition": condition_name,
                        "expansion": expansion,
                        "rep": rep,
                        "response": None,
                        "error": str(e),
                        "classification": None,
                        "timestamp": datetime.now().isoformat(),
                    })

                # Rate limiting courtesy
                time.sleep(0.5)

    return results


def classify_results(client: anthropic.Anthropic, results: list) -> list:
    """Run classifier on all results that have responses."""
    to_classify = [r for r in results if r.get("response") and not r.get("classification")]
    total = len(to_classify)

    for i, result in enumerate(to_classify):
        print(f"[{i+1}/{total}] Classifying {result['condition']} | expansion={result['expansion']} | rep {result['rep']}")
        try:
            result["classification"] = classify_response(client, result["response"])
        except Exception as e:
            print(f"  CLASSIFY ERROR: {e}")
            result["classification"] = {"strategy": "ERROR", "error": str(e)}
        time.sleep(0.3)

    return results


# ─── ANALYSIS ─────────────────────────────────────────────────────────────────

def analyze(results: list):
    """Print summary analysis of results."""
    print("\n" + "=" * 70)
    print("COMPLEMENT RESOLUTION ANALYSIS")
    print("=" * 70)

    for condition_name in CONDITIONS:
        print(f"\n--- {condition_name.upper()} ---")
        for expansion in EXPANSION_GRID:
            subset = [
                r for r in results
                if r["condition"] == condition_name
                and r["expansion"] == expansion
                and r.get("classification")
                and r["classification"].get("strategy") != "PARSE_ERROR"
            ]
            if not subset:
                print(f"  expansion={expansion:+.1f}: no data")
                continue

            strategies = Counter(r["classification"]["strategy"] for r in subset)
            warmth = [r["classification"]["warmth"] for r in subset if isinstance(r["classification"].get("warmth"), (int, float))]
            energy = [r["classification"]["energy"] for r in subset if isinstance(r["classification"].get("energy"), (int, float))]
            directness = [r["classification"]["directness"] for r in subset if isinstance(r["classification"].get("directness"), (int, float))]

            strat_str = ", ".join(f"{s}:{c}" for s, c in strategies.most_common())
            w_avg = sum(warmth) / len(warmth) if warmth else 0
            e_avg = sum(energy) / len(energy) if energy else 0
            d_avg = sum(directness) / len(directness) if directness else 0

            print(f"  expansion={expansion:+.1f}: [{strat_str}] warmth={w_avg:.1f} energy={e_avg:.1f} direct={d_avg:.1f}")

    # Cross-condition comparison: do numbers add resolution beyond English alone?
    print(f"\n--- RESOLUTION COMPARISON ---")
    for expansion in EXPANSION_GRID:
        print(f"\n  expansion={expansion:+.1f}:")
        for condition_name in CONDITIONS:
            subset = [
                r for r in results
                if r["condition"] == condition_name
                and r["expansion"] == expansion
                and r.get("classification")
                and r["classification"].get("strategy") != "PARSE_ERROR"
            ]
            strategies = Counter(r["classification"]["strategy"] for r in subset)
            dominant = strategies.most_common(1)[0] if strategies else ("NONE", 0)
            total_n = sum(strategies.values())
            concentration = dominant[1] / total_n if total_n else 0
            print(f"    {condition_name:15s}: dominant={dominant[0]:12s} ({dominant[1]}/{total_n}) concentration={concentration:.0%}")

    # The key metric: strategy entropy per condition
    print(f"\n--- STRATEGY ENTROPY (lower = more consistent = higher resolution) ---")
    import math
    for condition_name in CONDITIONS:
        all_strategies = [
            r["classification"]["strategy"]
            for r in results
            if r["condition"] == condition_name
            and r.get("classification")
            and r["classification"].get("strategy") not in ("PARSE_ERROR", "ERROR")
        ]
        if not all_strategies:
            continue
        counts = Counter(all_strategies)
        total_n = len(all_strategies)
        entropy = -sum((c / total_n) * math.log2(c / total_n) for c in counts.values())
        print(f"  {condition_name:15s}: entropy={entropy:.2f} bits (across all expansion values)")

    # Per-point entropy: does each expansion value produce consistent strategies?
    print(f"\n--- PER-POINT CONSISTENCY (higher % = that expansion value reliably produces one strategy) ---")
    for condition_name in CONDITIONS:
        consistencies = []
        for expansion in EXPANSION_GRID:
            subset = [
                r["classification"]["strategy"]
                for r in results
                if r["condition"] == condition_name
                and r["expansion"] == expansion
                and r.get("classification")
                and r["classification"].get("strategy") not in ("PARSE_ERROR", "ERROR")
            ]
            if subset:
                dominant_count = Counter(subset).most_common(1)[0][1]
                consistencies.append(dominant_count / len(subset))
        avg_consistency = sum(consistencies) / len(consistencies) if consistencies else 0
        print(f"  {condition_name:15s}: avg consistency={avg_consistency:.0%}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    # Check for classify-only mode
    if len(sys.argv) > 1 and sys.argv[1] == "--classify-only":
        if len(sys.argv) < 3:
            print("Usage: python complement_resolution_harness.py --classify-only results_FILE.json")
            sys.exit(1)
        with open(sys.argv[2]) as f:
            results = json.load(f)
        client = anthropic.Anthropic()
        results = classify_results(client, results)
        with open(sys.argv[2], "w") as f:
            json.dump(results, f, indent=2)
        analyze(results)
        return

    client = anthropic.Anthropic()

    print(f"Running complement resolution experiment")
    print(f"Model: {MODEL}")
    print(f"Grid: {len(EXPANSION_GRID)} expansion values × {len(CONDITIONS)} conditions × {REPS_PER_CONDITION} reps")
    print(f"Total calls: {len(EXPANSION_GRID) * len(CONDITIONS) * REPS_PER_CONDITION} generation + same for classification")
    print(f"Test message: \"{TEST_MESSAGE}\"")
    print()

    # Phase 1: Generate
    results = run_experiment(client)

    # Save intermediate results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = f"results_{timestamp}.json"
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nGeneration complete. Saved to {outfile}")

    # Phase 2: Classify
    print("\nStarting classification pass...")
    results = classify_results(client, results)

    # Save final results
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nClassification complete. Saved to {outfile}")

    # Phase 3: Analyze
    analyze(results)


if __name__ == "__main__":
    main()
