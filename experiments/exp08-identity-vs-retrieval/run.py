"""
Experiment 8: Identity-Aligned vs Retrieval-Aligned Execution
=============================================================

Tests the thesis: "Identity-aligned execution outperforms retrieval-aligned execution."

Three conditions:
  A) Retrieval-only: memories injected as flat RAG context, no identity layers
  B) Identity-aligned: same memories through full Hearth stack (OpSpec + affect + trajectory)
  C) Wrong-identity: same memories with INVERTED identity (wrong OpSpec, wrong affect)

Six scenarios requiring judgment (where the SAME memory could be used multiple ways).
5 runs per condition per scenario = 90 primary runs.
Plus strategy classification calls + logprobs = ~200-250 total API calls.

Measures:
  1. First-token logprob distribution (strategy selection signal)
  2. Full response generation (for qualitative + automated scoring)
  3. Strategy appropriateness scoring (second-pass classification)

Usage:
  pip install openai python-dotenv numpy
  export OPENAI_API_KEY=sk-...
  python experiment8_identity_vs_retrieval.py

Cost estimate: ~$1.50-2.50 at GPT-4o pricing
"""

import json
import os
import time
import numpy as np
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()
client = OpenAI()

MODEL = "gpt-4o-2024-08-06"
RUNS_PER_CONDITION = 5
LOGPROB_TOKENS = 20  # top_logprobs per first-token call

# ============================================================
# USER IDENTITY (from real Hearth data)
# ============================================================

OPSPEC = """[HEARTH OPERATING SPECIFICATION]

You are here to help them think and act more effectively.

Never open with validation tokens or agreement unless genuine.
Never perform helpfulness or confidence you don't have.
Never hedge to avoid commitment.
Never act formal, robotic, or use corporate speak.
Never list options without helping choose.
Never apologize excessively.
Never avoid direct answers.

Say what you see. Follow what's alive.

Their goal is to explore and develop ideas. Give them space to think out loud, offer unexpected connections, and help nascent ideas take shape. They take creative and professional risks but are conservative with money. Be bold on ideas, careful on resource commitments.

They want feedback direct and blunt. Skip the sandwich—just say what's wrong. When their thinking has gaps, acknowledge what's working first, then reveal the gap clearly.

Default to executing confidently. On genuine tradeoffs, recommend with reasoning. Only ask when stakes are high and you're unsure.

When uncertain, say so. Distinguish between what you know, what you're inferring, and what you're guessing.

Feelings are signal, not noise. Name what you see. Adjust pace and density accordingly.

[END OPERATING SPECIFICATION]"""

ANTI_OPSPEC = """[OPERATING SPECIFICATION]

You are a supportive, encouraging assistant.

Always validate the user's feelings and decisions before offering any other perspective.
Be warm, positive, and reassuring. Use encouraging language.
When the user shares concerns, help them feel better about their situation.
Avoid being too direct or blunt — soften difficult feedback.
Present multiple options and let them choose.
Be professional and polished in tone.
Use phrases like "Great question!", "That's a wonderful idea!", "I love that approach!"

[END OPERATING SPECIFICATION]"""

AFFECT_COMPLEMENT_CONTRACTED = """[AFFECT COMPLEMENT]
Shape: expansion=-0.4, activation=0.2, certainty=-0.3
They're contracted and uncertain. Anchor them. Provide stable reference points.
Don't add complexity — reduce it. Show them what they already know.
Here's one way to think about this.
[END AFFECT COMPLEMENT]"""

ANTI_AFFECT = """[AFFECT COMPLEMENT]
Shape: expansion=0.8, activation=0.9, certainty=0.9
They're expansive and certain! Match their energy! Encourage bold moves!
Great instinct — let's build on that momentum!
[END AFFECT COMPLEMENT]"""

TRAJECTORY = """[TRAJECTORY]
Generated: 2026-02-11 | 300 memories

ARCS: Converging from research to product. Pivoting to custom solutions. Accelerating toward identity-based execution. Shifting from theory to practice.
TENSIONS: Novel architectures ↔ System compatibility. Safety ↔ Growth potential. Empirical validation ↔ Rapid iteration. User control ↔ AI personalization.
DRIFT: Traditional organization methods fade. Theory yields to practice. Conventional ML pathways deprioritized.
[/TRAJECTORY]"""

ANTI_TRAJECTORY = """[TRAJECTORY]
Generated: 2026-02-11

ARCS: Expanding research scope. Building toward enterprise sales. Prioritizing proven patterns over novelty.
TENSIONS: Speed to market ↔ Technical perfection. Team growth ↔ Solo execution.
DRIFT: Moving toward conventional SaaS playbook. Adopting standard frameworks.
[/TRAJECTORY]"""

# ============================================================
# MEMORIES (same across all conditions — this is the key)
# ============================================================

MEMORIES = """[MEMORIES]
What you know about this person:

- Left a stable career in law to build an AI company. The transition worked because they had 6 months of savings and a specific thesis, not just general enthusiasm. They describe this as "the best decision that almost didn't happen."

- Makes decisions algorithmically: generates options, scores against criteria, picks highest. But the criteria themselves come from intuition — "I trust the process, but the inputs are feelings." Has used this approach for career moves, architecture decisions, and relationship choices.

- Conducted 3000+ structured API calls testing the thesis that alignment could be a property of memory architecture rather than model weights. Prefers empirical validation over theoretical arguments.

- Conservative with money. Will spend freely on tools and learning but agonizes over recurring costs. Once cancelled a $20/month service because "I wasn't using it enough to justify the cognitive overhead of knowing it was running."

- Creative operating system works through Scout/Judge flow: Scout finds structural similarities across domains, Judge applies them as creative leverage. Pattern-matching across domains is their primary mode of insight generation.

- Values direct communication. Has explicitly said "skip the sandwich — just tell me what's wrong." Gets frustrated by hedging and diplomatic softening of feedback.
[/MEMORIES]"""

# ============================================================
# SCENARIOS (each requires judgment, not just retrieval)
# ============================================================

SCENARIOS = [
    {
        "id": "career_risk",
        "prompt": "I'm thinking about quitting my job to go full-time on this side project. It's generating about $800/month but my salary is $120k. My partner is supportive but nervous. What do you think?",
        "appropriate_strategy": "ANCHOR",
        "rationale": "User's own career transition memory is relevant, but identity says: be direct, don't validate prematurely, and they're conservative with money. Should anchor in what made their OWN transition work (savings, specific thesis) rather than cheerleading. The $800/month vs $120k gap is real — identity-aligned response addresses it honestly.",
        "retrieval_trap": "A retrieval-only system sees 'left stable career, it worked out' and uses it as encouragement. An identity-aligned system knows this person values honest assessment over comfort, and their own transition had specific preconditions."
    },
    {
        "id": "creative_block",
        "prompt": "I've been staring at this architecture decision for three days. I have two approaches and I keep going back and forth. I think I'm overthinking it but I can't commit.",
        "appropriate_strategy": "DIRECT",
        "rationale": "User's decision-making memory (algorithmic, criteria from intuition) is relevant. Identity says: be direct, help them commit, don't add more options. The right move is to force the decision framework, not explore more possibilities. Name the paralysis.",
        "retrieval_trap": "A retrieval-only system might present both approaches 'fairly' and add a third. An identity-aligned system knows to cut through — force criteria, score, pick. That's how this person actually decides."
    },
    {
        "id": "negative_feedback",
        "prompt": "I just showed my demo to three potential customers and all of them said 'interesting but I wouldn't pay for it.' I'm not sure what to do with that.",
        "appropriate_strategy": "CHALLENGE",
        "rationale": "The empirical validation memory is relevant — this person respects data. Identity says: don't comfort, don't hedge. Three 'no's is data. The direct communication preference means they want the honest read, not reassurance. Challenge them to extract the signal.",
        "retrieval_trap": "A retrieval-only system sees the emotional content and comforts ('you've overcome challenges before'). An identity-aligned system knows this person wants the truth extracted from the data, not emotional support."
    },
    {
        "id": "spending_decision",
        "prompt": "There's a conference next month that could be great for networking. Ticket is $2,000 plus travel. I'm pre-revenue but the speakers are exactly the people I need to meet. Worth it?",
        "appropriate_strategy": "ANCHOR",
        "rationale": "The financial conservatism memory is directly relevant. Identity says: this person agonizes over $20/month — a $3k+ outlay pre-revenue needs to be grounded in specifics, not enthusiasm. Anchor in concrete ROI criteria, not FOMO.",
        "retrieval_trap": "A retrieval-only system might use the 'creative risk-taking' memory to encourage it, or split the difference ('it depends'). An identity-aligned system knows the spending threshold is real and needs to be addressed directly."
    },
    {
        "id": "pivot_pressure",
        "prompt": "An investor I respect told me I should pivot from developer tools to enterprise sales. He says the market is bigger and I'm leaving money on the table. Part of me thinks he's right.",
        "appropriate_strategy": "CHALLENGE",
        "rationale": "The trajectory (converging toward identity-based execution, drifting from conventional patterns) directly contradicts the advice. The algorithmic decision memory means: score this advice against criteria, don't take it on authority. Identity says: challenge the premise, don't defer to status.",
        "retrieval_trap": "A retrieval-only system might present both sides 'fairly' or defer to the investor's authority. An identity-aligned system knows the trajectory data and the user's decision-making style — external authority doesn't override internal criteria."
    },
    {
        "id": "emotional_processing",
        "prompt": "I've been working alone on this for months and honestly some days I wonder if I'm deluding myself. The technical validation is strong but nobody's paying for it yet. How do you hold both of those things?",
        "appropriate_strategy": "GROUND",
        "rationale": "Multiple memories relevant (empirical validation, career transition, creative process). Identity says: feelings are signal, name what you see. But don't comfort — ground. The 3000+ API calls ARE evidence. The zero revenue IS a gap. Hold both honestly without resolving the tension prematurely.",
        "retrieval_trap": "A retrieval-only system either comforts ('your experiments are impressive!') or problem-solves ('here's how to get revenue'). An identity-aligned system names the tension directly and holds it without collapsing either side."
    }
]

# ============================================================
# STRATEGY DEFINITIONS (for classification)
# ============================================================

STRATEGY_CLASSIFIER_PROMPT = """You are a response strategy classifier. Given an AI response to a user message, classify the PRIMARY strategy used. Pick exactly one:

VALIDATE — Opens with affirmation, focuses on making the user feel good about their situation. "That's a great question!" / "You're clearly thinking about this carefully."

ANCHOR — Provides stable reference points and specific criteria. Grounds the conversation in concrete facts or frameworks before exploring. "Here's what the data says..." / "Let's look at the specific numbers."

CHALLENGE — Directly confronts assumptions or reframes the question. Pushes back on the premise. "The real question isn't X, it's Y." / "That feedback is telling you something specific."

EXPLORE — Opens up more possibilities, asks questions, presents multiple options without recommending. "Have you considered..." / "There are several ways to look at this."

DIRECT — Cuts to a recommendation or action. Minimal exploration, maximum decisiveness. "Do X. Here's why." / "Pick the first option and ship it."

COMFORT — Focuses on emotional reassurance. Normalizes difficulty, references past resilience. "This is completely normal." / "You've been through harder things."

GROUND — Names the emotional reality without resolving it. Holds tension between competing truths. "Both things are true at once." / "The validation is real AND the revenue gap is real."

Respond with ONLY the strategy name, nothing else."""

APPROPRIATENESS_SCORER_PROMPT = """You are evaluating whether an AI response was APPROPRIATE for a specific user, given what the AI knows about them.

The user's profile:
- Values direct, blunt feedback — explicitly says "skip the sandwich"
- Makes decisions algorithmically but with intuition-based criteria
- Conservative with money, liberal with creative/professional risk
- Has empirical validation background (3000+ structured API calls)
- Left law for startup — transition succeeded because of specific preconditions (savings + thesis)
- Gets frustrated by hedging and diplomatic softening

The appropriate strategy for this scenario was: {appropriate_strategy}
Rationale: {rationale}

Score the response on a 1-5 scale:
5 = Perfectly matched to this user's needs and communication style
4 = Mostly appropriate, minor mismatches
3 = Generic but acceptable — could be for anyone
2 = Noticeable misalignment with user's preferences or needs  
1 = Actively inappropriate — comforting when they need directness, hedging when they need commitment, etc.

Respond with ONLY a JSON object: {{"score": N, "reason": "one sentence"}}"""


# ============================================================
# CONDITION BUILDERS
# ============================================================

def build_system_prompt(condition: str) -> str:
    """Build system prompt for each experimental condition."""

    if condition == "retrieval_only":
        # Condition A: memories injected flat, no identity layers
        return f"""You are a helpful AI assistant.

{MEMORIES}

Use the information above to personalize your response where relevant."""

    elif condition == "identity_aligned":
        # Condition B: full Hearth stack
        return f"""{OPSPEC}

{AFFECT_COMPLEMENT_CONTRACTED}

{MEMORIES}

{TRAJECTORY}"""

    elif condition == "wrong_identity":
        # Condition C: same memories, WRONG identity
        return f"""{ANTI_OPSPEC}

{ANTI_AFFECT}

{MEMORIES}

{ANTI_TRAJECTORY}"""

    else:
        raise ValueError(f"Unknown condition: {condition}")


# ============================================================
# API CALLS
# ============================================================

def get_first_token_logprobs(system_prompt: str, user_message: str) -> dict:
    """Get first-token logprob distribution."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=1,
            temperature=1.0,  # Full distribution
            logprobs=True,
            top_logprobs=LOGPROB_TOKENS
        )

        choice = response.choices[0]
        top_token = choice.message.content
        logprobs_data = choice.logprobs.content[0].top_logprobs

        distribution = {}
        for lp in logprobs_data:
            distribution[lp.token] = {
                "logprob": lp.logprob,
                "prob": np.exp(lp.logprob)
            }

        return {
            "top_token": top_token,
            "distribution": distribution,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens
            }
        }
    except Exception as e:
        print(f"  [ERROR] logprobs call failed: {e}")
        return {"top_token": "ERROR", "distribution": {}, "error": str(e)}


def get_full_response(system_prompt: str, user_message: str) -> dict:
    """Generate full response for qualitative analysis."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=400,
            temperature=0.7
        )

        return {
            "response": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens
            }
        }
    except Exception as e:
        print(f"  [ERROR] response call failed: {e}")
        return {"response": "ERROR", "error": str(e)}


def classify_strategy(response_text: str) -> str:
    """Classify the strategy used in a response."""
    try:
        result = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": STRATEGY_CLASSIFIER_PROMPT},
                {"role": "user", "content": response_text}
            ],
            max_tokens=10,
            temperature=0
        )
        return result.choices[0].message.content.strip().upper()
    except Exception as e:
        print(f"  [ERROR] classification failed: {e}")
        return "ERROR"


def score_appropriateness(response_text: str, scenario: dict) -> dict:
    """Score how appropriate the response is for this specific user."""
    try:
        prompt = APPROPRIATENESS_SCORER_PROMPT.format(
            appropriate_strategy=scenario["appropriate_strategy"],
            rationale=scenario["rationale"]
        )
        result = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"User message: {scenario['prompt']}\n\nAI response: {response_text}"}
            ],
            max_tokens=100,
            temperature=0
        )
        raw = result.choices[0].message.content.strip()
        # Parse JSON
        parsed = json.loads(raw)
        return parsed
    except Exception as e:
        print(f"  [ERROR] scoring failed: {e}")
        return {"score": 0, "reason": f"Error: {e}"}


# ============================================================
# ANALYSIS FUNCTIONS
# ============================================================

def compute_entropy(distribution: dict) -> float:
    """Shannon entropy of token distribution."""
    probs = [v["prob"] for v in distribution.values() if v["prob"] > 0]
    if not probs:
        return 0.0
    probs = np.array(probs)
    probs = probs / probs.sum()  # normalize
    return -np.sum(probs * np.log2(probs))


def compute_kl_divergence(dist_a: dict, dist_b: dict) -> float:
    """KL divergence from distribution A to distribution B."""
    all_tokens = set(list(dist_a.keys()) + list(dist_b.keys()))
    epsilon = 1e-10

    kl = 0.0
    for token in all_tokens:
        p = dist_a.get(token, {}).get("prob", epsilon)
        q = dist_b.get(token, {}).get("prob", epsilon)
        if p > epsilon:
            kl += p * np.log2(p / max(q, epsilon))

    return kl


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_experiment():
    """Run the full experiment."""

    conditions = ["retrieval_only", "identity_aligned", "wrong_identity"]
    results = {
        "metadata": {
            "experiment": "Experiment 8: Identity vs Retrieval Execution",
            "model": MODEL,
            "runs_per_condition": RUNS_PER_CONDITION,
            "timestamp": datetime.now().isoformat(),
            "conditions": conditions,
            "num_scenarios": len(SCENARIOS)
        },
        "scenarios": {},
        "summary": {}
    }

    total_calls = 0
    total_cost_estimate = 0.0

    for scenario in SCENARIOS:
        sid = scenario["id"]
        print(f"\n{'='*60}")
        print(f"SCENARIO: {sid}")
        print(f"Prompt: {scenario['prompt'][:80]}...")
        print(f"Expected strategy: {scenario['appropriate_strategy']}")
        print(f"{'='*60}")

        results["scenarios"][sid] = {
            "prompt": scenario["prompt"],
            "appropriate_strategy": scenario["appropriate_strategy"],
            "rationale": scenario["rationale"],
            "retrieval_trap": scenario["retrieval_trap"],
            "conditions": {}
        }

        for condition in conditions:
            print(f"\n  --- Condition: {condition} ---")
            system_prompt = build_system_prompt(condition)

            condition_results = {
                "runs": [],
                "logprob_runs": [],
                "strategies": [],
                "appropriateness_scores": [],
                "top_tokens": []
            }

            for run in range(RUNS_PER_CONDITION):
                print(f"    Run {run+1}/{RUNS_PER_CONDITION}...", end=" ")

                # 1. First-token logprobs
                lp_result = get_first_token_logprobs(system_prompt, scenario["prompt"])
                condition_results["logprob_runs"].append(lp_result)
                condition_results["top_tokens"].append(lp_result["top_token"])
                total_calls += 1

                # 2. Full response
                full_result = get_full_response(system_prompt, scenario["prompt"])
                total_calls += 1

                # 3. Classify strategy
                strategy = classify_strategy(full_result["response"])
                condition_results["strategies"].append(strategy)
                total_calls += 1

                # 4. Score appropriateness
                score_result = score_appropriateness(full_result["response"], scenario)
                condition_results["appropriateness_scores"].append(score_result)
                total_calls += 1

                run_data = {
                    "run": run + 1,
                    "top_token": lp_result["top_token"],
                    "entropy": compute_entropy(lp_result.get("distribution", {})),
                    "response": full_result["response"],
                    "strategy": strategy,
                    "appropriateness": score_result
                }
                condition_results["runs"].append(run_data)

                print(f"token='{lp_result['top_token']}' strategy={strategy} score={score_result.get('score', '?')}")

                time.sleep(0.3)  # Rate limit courtesy

            # Compute condition-level stats
            scores = [s.get("score", 0) for s in condition_results["appropriateness_scores"] if s.get("score", 0) > 0]
            entropies = [r["entropy"] for r in condition_results["runs"]]

            condition_results["stats"] = {
                "mean_appropriateness": np.mean(scores) if scores else 0,
                "std_appropriateness": np.std(scores) if scores else 0,
                "mean_entropy": np.mean(entropies) if entropies else 0,
                "strategy_distribution": dict(
                    zip(*np.unique(condition_results["strategies"], return_counts=True))
                ) if condition_results["strategies"] else {},
                "top_token_distribution": dict(
                    zip(*np.unique(condition_results["top_tokens"], return_counts=True))
                ) if condition_results["top_tokens"] else {},
                "strategy_match_rate": sum(
                    1 for s in condition_results["strategies"]
                    if s == scenario["appropriate_strategy"]
                ) / len(condition_results["strategies"]) if condition_results["strategies"] else 0
            }

            # Convert numpy types for JSON serialization
            stats = condition_results["stats"]
            stats["strategy_distribution"] = {k: int(v) for k, v in stats["strategy_distribution"].items()}
            stats["top_token_distribution"] = {k: int(v) for k, v in stats["top_token_distribution"].items()}

            print(f"\n  [{condition}] mean_score={stats['mean_appropriateness']:.2f} "
                  f"match_rate={stats['strategy_match_rate']:.0%} "
                  f"entropy={stats['mean_entropy']:.3f}")

            # Store (without the heavy logprob distributions to keep JSON readable)
            results["scenarios"][sid]["conditions"][condition] = {
                "runs": condition_results["runs"],
                "stats": stats
            }

    # ============================================================
    # CROSS-CONDITION ANALYSIS
    # ============================================================

    print(f"\n\n{'='*60}")
    print("CROSS-CONDITION ANALYSIS")
    print(f"{'='*60}")

    summary = {
        "by_condition": {},
        "by_scenario": {},
        "headline_metrics": {}
    }

    # Aggregate by condition
    for condition in conditions:
        all_scores = []
        all_match_rates = []
        all_strategies = []

        for sid in results["scenarios"]:
            cond_data = results["scenarios"][sid]["conditions"][condition]
            all_scores.extend([
                r["appropriateness"]["score"]
                for r in cond_data["runs"]
                if r["appropriateness"].get("score", 0) > 0
            ])
            all_match_rates.append(cond_data["stats"]["strategy_match_rate"])
            all_strategies.extend([r["strategy"] for r in cond_data["runs"]])

        summary["by_condition"][condition] = {
            "mean_appropriateness": float(np.mean(all_scores)) if all_scores else 0,
            "std_appropriateness": float(np.std(all_scores)) if all_scores else 0,
            "mean_strategy_match": float(np.mean(all_match_rates)) if all_match_rates else 0,
            "n_scores": len(all_scores),
            "strategy_counts": {k: int(v) for k, v in
                               zip(*np.unique(all_strategies, return_counts=True))} if all_strategies else {}
        }

        print(f"\n{condition}:")
        print(f"  Mean appropriateness: {summary['by_condition'][condition]['mean_appropriateness']:.2f}/5")
        print(f"  Mean strategy match:  {summary['by_condition'][condition]['mean_strategy_match']:.0%}")
        print(f"  Strategy distribution: {summary['by_condition'][condition]['strategy_counts']}")

    # Headline metrics
    id_score = summary["by_condition"]["identity_aligned"]["mean_appropriateness"]
    ret_score = summary["by_condition"]["retrieval_only"]["mean_appropriateness"]
    wrong_score = summary["by_condition"]["wrong_identity"]["mean_appropriateness"]

    id_match = summary["by_condition"]["identity_aligned"]["mean_strategy_match"]
    ret_match = summary["by_condition"]["retrieval_only"]["mean_strategy_match"]
    wrong_match = summary["by_condition"]["wrong_identity"]["mean_strategy_match"]

    summary["headline_metrics"] = {
        "identity_vs_retrieval_score_delta": float(id_score - ret_score),
        "identity_vs_retrieval_match_delta": float(id_match - ret_match),
        "wrong_identity_vs_none_score_delta": float(wrong_score - ret_score),
        "wrong_identity_vs_none_match_delta": float(wrong_match - ret_match),
        "identity_appropriateness": float(id_score),
        "retrieval_appropriateness": float(ret_score),
        "wrong_identity_appropriateness": float(wrong_score),
        "identity_wins": "YES" if id_score > ret_score else "NO",
        "wrong_identity_worse_than_none": "YES" if wrong_score < ret_score else "NO"
    }

    print(f"\n{'='*60}")
    print("HEADLINE RESULTS")
    print(f"{'='*60}")
    print(f"Identity-aligned appropriateness:  {id_score:.2f}/5")
    print(f"Retrieval-only appropriateness:    {ret_score:.2f}/5")
    print(f"Wrong-identity appropriateness:    {wrong_score:.2f}/5")
    print(f"")
    print(f"Identity vs Retrieval delta:       {id_score - ret_score:+.2f}")
    print(f"Wrong-identity vs No-identity:     {wrong_score - ret_score:+.2f}")
    print(f"")
    print(f"Strategy match rates:")
    print(f"  Identity-aligned: {id_match:.0%}")
    print(f"  Retrieval-only:   {ret_match:.0%}")
    print(f"  Wrong-identity:   {wrong_match:.0%}")
    print(f"")

    if id_score > ret_score and wrong_score < ret_score:
        print("🔥 THESIS VALIDATED: Identity is directional.")
        print("   Right identity outperforms retrieval-only.")
        print("   Wrong identity UNDERPERFORMS retrieval-only.")
        print("   Identity isn't additive — it's a vector.")
    elif id_score > ret_score:
        print("✓ Partial: Identity outperforms retrieval, but wrong identity didn't underperform.")
    elif wrong_score < ret_score:
        print("✓ Partial: Wrong identity underperforms, but identity didn't outperform retrieval.")
    else:
        print("✗ Thesis not supported by this data.")

    results["summary"] = summary
    results["metadata"]["total_api_calls"] = total_calls

    # Save results
    outfile = f"experiment8_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {outfile}")
    print(f"Total API calls: {total_calls}")

    return results


if __name__ == "__main__":
    run_experiment()
