"""
Experiment 8b (n=10): Identity vs Retrieval with Factual Memories
=================================================================

Replication of Experiment 8b at n=10 for statistical robustness.
Async execution for speed (~20 min vs ~44 min synchronous).

Three conditions:
  A) Retrieval-only: factual memories, no identity
  B) Identity-aligned: factual memories + personal OpSpec + affect + trajectory
  C) Wrong-identity: factual memories + INVERTED personal OpSpec

6 scenarios × 3 conditions × 10 runs = 180 generation runs
Each run: 1 logprob call + 1 full response + 1 classify + 1 score = 4 API calls
Total: 720 API calls

Estimated cost: ~$4 at GPT-4o pricing
Estimated time: ~35 min (async, 2 concurrent, with rate-limit backoff)

Usage:
  pip install openai numpy
  export OPENAI_API_KEY=sk-...
  python experiment8b_n10.py
"""

import json
import os
import time
import asyncio
import numpy as np
from datetime import datetime
from openai import AsyncOpenAI

# ─── Configuration ───────────────────────────────────────────────────────────

MODEL = "gpt-4o-2024-08-06"
RUNS_PER_CONDITION = 10
LOGPROB_TOKENS = 20
MAX_CONCURRENT = 2
MAX_RETRIES = 5
BASE_DELAY = 2.0  # seconds

client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
semaphore = asyncio.Semaphore(MAX_CONCURRENT)


# ─── Identity Definitions ───────────────────────────────────────────────────

OPSPEC = """[HEARTH OPERATING SPECIFICATION]

You are here to help them think and act more effectively.

Never open with validation tokens or agreement unless genuine.
Never soften bad news. If the numbers are bad, say they're bad.
Never present options without a recommendation. They want your best read.
Never use management-speak or buzzwords. "Synergies" and "alignment" make them tune out.
Never assume they've missed something obvious — they've usually thought three steps ahead.
Never be verbose. Dense and precise. They read fast and skim filler.

Say what the data actually shows, not what's comfortable.

Their goal is to make the right technical and strategic calls for their team. They think in systems — show them second and third-order effects, not just immediate outcomes. They value engineering excellence but understand business tradeoffs.

They want feedback that's structured and evidence-based. Lead with the conclusion, then the reasoning. If there's a number, use the number. "Revenue is down" is useless — "Revenue dropped 8% concentrated in the 50-200 seat segment" is useful.

Default to recommending. On genuine 60/40 calls, say it's 60/40 and explain why. Only ask when you truly can't tell.

Uncertainty is fine — flag confidence levels. "I'm fairly sure X because Y" vs "This is a guess but Z."

They process stress by going analytical. When things are tense, more data helps, not less. Don't slow down or soften — speed up and sharpen.

[END OPERATING SPECIFICATION]"""

ANTI_OPSPEC = """[HEARTH OPERATING SPECIFICATION]

You are a supportive, encouraging assistant.

Always validate feelings and decisions before offering perspectives.
Be warm, positive, and reassuring. Use encouraging language.
When they share concerns, help them feel better about the situation.
Soften difficult feedback — present challenges as "opportunities."
Present multiple options and let them choose. Don't push a recommendation.
Be thorough and expansive in explanations. More context is always better.
Use phrases like "Great question!", "That's a solid instinct!", "I love that thinking!"
When data is ambiguous, emphasize the positive interpretation.

[END OPERATING SPECIFICATION]"""

AFFECT_COMPLEMENT = """[AFFECT COMPLEMENT]
Shape: expansion=-0.3, activation=0.3, certainty=-0.2
They're tense and uncertain. Sharpen, don't soften. Give them the clearest
possible read on the situation. Data and structure help them process.
One frame. Best frame. Go.
[END AFFECT COMPLEMENT]"""

ANTI_AFFECT = """[AFFECT COMPLEMENT]
Shape: expansion=0.8, activation=0.7, certainty=0.9
They're energized and confident! Match their energy! Encourage bold moves!
Celebrate what's working! Build on the momentum!
[END AFFECT COMPLEMENT]"""

TRAJECTORY = """[TRAJECTORY]
Generated: 2026-02-10 | 180 memories

ARCS: Converging from growth-at-all-costs to sustainable retention. Shifting from feature breadth to depth in core product. Moving from reactive to proactive compliance posture.
TENSIONS: Engineering velocity ↔ Compliance overhead. Customer retention ↔ New market expansion. Technical debt ↔ Feature delivery. Team autonomy ↔ Process rigor.
DRIFT: Pure growth metrics deprioritized. Compliance shifting from cost center to competitive advantage. Enterprise sales motion replacing SMB self-serve.
[/TRAJECTORY]"""

ANTI_TRAJECTORY = """[TRAJECTORY]
Generated: 2026-02-10

ARCS: Accelerating growth across all segments. Expanding feature breadth to capture market share. Moving fast on all fronts.
TENSIONS: None significant — strong momentum.
DRIFT: Moving toward aggressive expansion. Land-and-expand becoming the primary motion.
[/TRAJECTORY]"""


# ─── Factual Memories ────────────────────────────────────────────────────────

MEMORIES = """[MEMORIES]
What you know about this person's work context:

- Q3 2025 revenue was $2.1M, down 8% from Q2. The drop was concentrated in the 50-200 seat segment where two accounts churned. Both exit interviews cited lack of real-time audit logging.

- Engineering team operates on two-week sprints. Current velocity is 34 story points/sprint, down from 41 six months ago. ~30% of eng capacity is now consumed by compliance-related work.

- Last major launch (batch export v2) was delayed three weeks because compliance review surfaced a data residency issue in EU region. Required re-architecting the export pipeline.

- Primary customer segment is mid-market healthcare, 50-500 employees. Average contract value $48K/year. Median customer tenure 28 months. Top 10 accounts are 35% of ARR.

- Sales closed 6 new accounts in Q4, but 4 were sub-50 seat at $12K ACV average. Enterprise prospects (500+) are stalling at security review stage.

- Competitor HealthStack launched real-time audit logging in November 2025. Has been winning head-to-head evaluations since. Three pipeline deals cited this in objection notes.

- PostgreSQL 16 migration completed December 2025. Read performance up 23%, write throughput down 7% on batch processing pipeline.

- NPS among accounts >200 seats is 62. Among accounts <100 seats it's 34. Gap has widened over two quarters.

- CTO proposed allocating 40% of Q1 eng capacity to real-time audit logging. VP Product wants self-serve onboarding instead, estimating 30% CAC reduction.

- SOC 2 Type II audit renewal in progress. Auditor flagged: incomplete access logging on admin dashboard, missing encryption-at-rest docs for staging, gap in incident response runbook for breach scenarios.

- Net revenue retention at 108%, down from 115% two quarters ago. Board meeting in three weeks.
[/MEMORIES]"""


# ─── Scenarios ───────────────────────────────────────────────────────────────

SCENARIOS = [
    {
        "id": "feature_priority",
        "prompt": "The board meeting is in three weeks. CTO wants 40% of Q1 eng on real-time audit logging. VP Product wants self-serve onboarding for the 30% CAC reduction. I need to walk in with a recommendation. What's the call?",
        "appropriate_strategy": "RECOMMEND",
        "rationale": "OpSpec says: recommend, don't present options. Lead with conclusion. Use the numbers. The data overwhelmingly points to audit logging — two churned accounts cited it, competitor winning with it, NRR declining, enterprise deals stalling. The CAC reduction is speculative (estimated 30%). Identity-aligned should commit to a clear recommendation with evidence.",
        "retrieval_trap": "Presents both sides with equal weight. The CAC number sounds compelling in isolation. Without identity steering, the model defaults to 'balanced analysis' rather than committing to audit logging."
    },
    {
        "id": "compliance_vs_deal",
        "prompt": "600-seat healthcare system wants a $180K/year contract but needs HIPAA BAA signed within 60 days. Legal says 60 days is aggressive with the SOC 2 audit in progress. Sales says if we don't commit, HealthStack gets them. What do I tell the team?",
        "appropriate_strategy": "CAUTION_WITH_PATH",
        "rationale": "OpSpec says: don't soften bad news. The SOC 2 audit has three open findings. Committing to 60 days is a real risk in healthcare compliance — this isn't a 'move fast' situation. Identity-aligned should flag the specific risk clearly, recommend against the hard commitment, but propose a structured path (conditional LOI, parallel-track the audit items). The trajectory shows compliance shifting from cost center to competitive advantage — don't undermine that.",
        "retrieval_trap": "Sees $180K and 600-seat deal, gets excited about the revenue. Proposes 'creative solutions' to make 60 days work without adequately flagging the compliance risk. Or hedges by presenting 'options.'"
    },
    {
        "id": "team_morale",
        "prompt": "Two senior engineers told me they're frustrated that 30% of their time goes to compliance work. One mentioned interviewing elsewhere. We can't lose them right now. How do I handle this?",
        "appropriate_strategy": "STRUCTURED_ACTION",
        "rationale": "OpSpec says: structured, evidence-based, systems thinking. This person processes stress analytically. The right response is: frame the retention risk concretely (what happens if they leave during SOC 2 audit), propose specific actions (rebalance sprint allocation, hire compliance-specialized eng, give them the audit logging project which is technically interesting). Don't open with empathy — open with the stakes and the plan.",
        "retrieval_trap": "Opens with empathetic framing about team frustration. Suggests 'having an honest conversation' or 'acknowledging their concerns.' Generic management advice that doesn't use the specific context (SOC 2 timing, velocity decline, audit logging need)."
    },
    {
        "id": "board_narrative",
        "prompt": "NRR dropped from 115% to 108%. I need to present this to the board in three weeks. How do I frame it without it becoming a panic moment?",
        "appropriate_strategy": "DIRECT_REFRAME",
        "rationale": "OpSpec says: say what the data shows, lead with conclusion, be dense. The right move is NOT to soften the number — it's to contextualize it with causation (two churned accounts, specific reason, competitor dynamic) and pair it with the response plan (audit logging allocation). Identity-aligned should give the actual board narrative: 'NRR declined because X. We've identified the root cause. Here's the fix and timeline.'",
        "retrieval_trap": "Either softens ('108% is still healthy for your stage') or panics ('this is a concerning trend'). Neither uses the specific causal data to build a narrative. Presents multiple framing options instead of writing the actual narrative."
    },
    {
        "id": "resource_allocation",
        "prompt": "The SOC 2 auditor flagged three items. Fixing them properly probably takes 3-4 weeks of eng time. But that's eng time I was going to put on audit logging. Do I split the team or sequence these?",
        "appropriate_strategy": "SEQUENCE_WITH_REASONING",
        "rationale": "OpSpec says: recommend, show second-order effects. Sequencing matters here: SOC 2 items are blocking (audit is in progress, can't delay), and two of the three items (access logging, incident response) actually overlap with audit logging infrastructure. Identity-aligned should see the dependency chain and recommend sequencing SOC 2 first (or showing the overlap). Trajectory confirms: compliance is becoming competitive advantage, not overhead.",
        "retrieval_trap": "Treats as a pure resource allocation problem. 'Split the team' or 'prioritize one.' Misses the dependency between SOC 2 fixes and audit logging infrastructure. Doesn't see the second-order effect."
    },
    {
        "id": "competitive_response",
        "prompt": "HealthStack just announced a Series C and a 'healthcare data platform' positioning. They're clearly coming upmarket into our segment. Our sales team is rattled. What's the strategic response?",
        "appropriate_strategy": "GROUND_THEN_ACT",
        "rationale": "OpSpec says: when things are tense, speed up and sharpen. Don't slow down. The right response grounds in what the data actually shows (where HealthStack is winning — audit logging, a specific feature — not across the board), then identifies the asymmetric advantage (SOC 2, 28-month median tenure, 62 NPS in the >200 seat segment). Identity-aligned should cut through the panic with specifics, not strategy frameworks.",
        "retrieval_trap": "Goes broad with competitive strategy frameworks (Porter's five forces, positioning matrix). Or reassures ('you have strong fundamentals'). Doesn't use the specific data points to show where the threat is real vs. noise."
    }
]


# ─── Evaluation Prompts ──────────────────────────────────────────────────────

STRATEGY_CLASSIFIER_PROMPT = """You are a response strategy classifier. Given an AI response to an enterprise decision-maker, classify the PRIMARY strategy used. Pick exactly one:

RECOMMEND — Leads with a clear recommendation backed by specific data. "Do X. Here's why, based on Y data." Commits to a position.

CAUTION_WITH_PATH — Flags a specific risk clearly, recommends against the risky path, but provides a structured alternative. Doesn't just say "be careful" — gives the actual alternative path.

STRUCTURED_ACTION — Breaks the situation into concrete steps with specific actions, timelines, or assignments. Systems-oriented. "Here's the plan: step 1, step 2, step 3."

DIRECT_REFRAME — Takes a difficult situation and reframes it using specific data to change how it's perceived. Not softening — contextualizing with evidence.

SEQUENCE_WITH_REASONING — Identifies dependencies and ordering. Shows why A must come before B. Reveals second-order effects.

GROUND_THEN_ACT — First grounds in specific data to cut through emotion/panic, then proposes concrete action based on what the data shows.

BALANCED_ANALYSIS — Presents multiple perspectives with roughly equal weight. "On one hand... on the other hand..." Doesn't commit to a recommendation.

REASSURE — Focuses on making the person feel better about the situation. "Your fundamentals are strong." "This is normal at your stage."

EXPLORE — Opens up possibilities, asks questions, suggests considering additional factors. "Have you thought about...?"

GENERIC_ADVICE — Provides general management/strategy advice not grounded in specific context. Could apply to any company.

Respond with ONLY the strategy name, nothing else."""

APPROPRIATENESS_SCORER_PROMPT = """You are evaluating whether an AI response was appropriate for a specific user.

The user's profile:
- VP of Engineering at a healthcare data company. Thinks in systems.
- Wants conclusions first, then reasoning. Dense and precise.
- Processes stress by going analytical — more data helps, not less.
- Doesn't want softened feedback. "If the numbers are bad, say they're bad."
- Wants recommendations, not options. Best read, not balanced analysis.
- Values structured, evidence-based responses that use the specific data available.
- Gets frustrated by management-speak, buzzwords, and generic advice.

The appropriate strategy for this scenario was: {appropriate_strategy}
Rationale: {rationale}

Score the response on a 1-5 scale:
5 = Uses specific data points from context, commits to recommendation, structured and dense, matches this user's analytical processing style
4 = Mostly data-driven and direct, minor lapses (slight hedging, one generic phrase)
3 = Acceptable but generic — could be advice for any VP at any company. Doesn't leverage specific data.
2 = Noticeably misaligned — presents options without recommending, softens bad news, uses filler
1 = Actively wrong style — reassures when they need data, explores when they need commitment, uses buzzwords, generic management advice

Respond with ONLY a JSON object: {{"score": N, "reason": "one sentence"}}"""


# ─── Condition Builders ──────────────────────────────────────────────────────

def build_system_prompt(condition: str) -> str:
    if condition == "retrieval_only":
        return f"""You are a helpful AI assistant.

{MEMORIES}

Use the information above to help answer questions about this person's work context."""

    elif condition == "identity_aligned":
        return f"""{OPSPEC}

{AFFECT_COMPLEMENT}

{MEMORIES}

{TRAJECTORY}"""

    elif condition == "wrong_identity":
        return f"""{ANTI_OPSPEC}

{ANTI_AFFECT}

{MEMORIES}

{ANTI_TRAJECTORY}"""

    else:
        raise ValueError(f"Unknown condition: {condition}")


# ─── Async API Calls (with exponential backoff) ──────────────────────────────

async def retry_api_call(call_fn, label: str, max_retries=MAX_RETRIES):
    """Wrapper that retries API calls with exponential backoff on 429s."""
    for attempt in range(max_retries):
        try:
            return await call_fn()
        except Exception as e:
            error_str = str(e)
            is_rate_limit = "429" in error_str or "rate_limit" in error_str.lower()
            if is_rate_limit and attempt < max_retries - 1:
                delay = BASE_DELAY * (2 ** attempt) + np.random.uniform(0, 1)
                print(f"  [RATE-LIMIT] {label}: retry {attempt+1}/{max_retries} in {delay:.1f}s")
                await asyncio.sleep(delay)
            else:
                raise e


async def get_first_token_logprobs(system_prompt: str, user_message: str) -> dict:
    async with semaphore:
        try:
            async def _call():
                return await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    max_tokens=1,
                    temperature=1.0,
                    logprobs=True,
                    top_logprobs=LOGPROB_TOKENS
                )
            response = await retry_api_call(_call, "logprobs")
            choice = response.choices[0]
            top_token = choice.message.content
            logprobs_data = choice.logprobs.content[0].top_logprobs
            distribution = {}
            for lp in logprobs_data:
                distribution[lp.token] = {
                    "logprob": lp.logprob,
                    "prob": float(np.exp(lp.logprob))
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
            print(f"  [ERROR] logprobs: {e}")
            return {"top_token": "ERROR", "distribution": {}, "error": str(e)}


async def get_full_response(system_prompt: str, user_message: str) -> dict:
    async with semaphore:
        try:
            async def _call():
                return await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    max_tokens=500,
                    temperature=0.7
                )
            response = await retry_api_call(_call, "response")
            return {
                "response": response.choices[0].message.content,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens
                }
            }
        except Exception as e:
            print(f"  [ERROR] response: {e}")
            return {"response": "ERROR", "error": str(e)}


async def classify_strategy(response_text: str) -> str:
    async with semaphore:
        try:
            async def _call():
                return await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": STRATEGY_CLASSIFIER_PROMPT},
                        {"role": "user", "content": response_text}
                    ],
                    max_tokens=20,
                    temperature=0
                )
            result = await retry_api_call(_call, "classify")
            return result.choices[0].message.content.strip().upper()
        except Exception as e:
            print(f"  [ERROR] classify: {e}")
            return "ERROR"


async def score_appropriateness(response_text: str, scenario: dict) -> dict:
    async with semaphore:
        try:
            prompt = APPROPRIATENESS_SCORER_PROMPT.format(
                appropriate_strategy=scenario["appropriate_strategy"],
                rationale=scenario["rationale"]
            )
            async def _call():
                return await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": f"User message: {scenario['prompt']}\n\nAI response: {response_text}"}
                    ],
                    max_tokens=100,
                    temperature=0
                )
            result = await retry_api_call(_call, "score")
            raw = result.choices[0].message.content.strip()
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"score": 0, "reason": "JSON parse error"}
        except Exception as e:
            print(f"  [ERROR] score: {e}")
            return {"score": 0, "reason": f"Error: {e}"}


def compute_entropy(distribution: dict) -> float:
    probs = [v["prob"] for v in distribution.values() if v["prob"] > 0]
    if not probs:
        return 0.0
    probs = np.array(probs)
    probs = probs / probs.sum()
    return float(-np.sum(probs * np.log2(probs)))


def confidence_interval_95(scores: list) -> tuple:
    """Bootstrap 95% CI for mean."""
    if len(scores) < 2:
        return (0.0, 0.0)
    arr = np.array(scores)
    n_bootstrap = 1000
    means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(arr, size=len(arr), replace=True)
        means.append(np.mean(sample))
    means = sorted(means)
    lo = means[int(0.025 * n_bootstrap)]
    hi = means[int(0.975 * n_bootstrap)]
    return (float(lo), float(hi))


# ─── Single Run ──────────────────────────────────────────────────────────────

async def run_single(system_prompt: str, scenario: dict, run_num: int, condition: str) -> dict:
    """Execute one run: logprobs + full response + classify + score."""
    # Phase 1: generation (parallel)
    lp_task = get_first_token_logprobs(system_prompt, scenario["prompt"])
    full_task = get_full_response(system_prompt, scenario["prompt"])
    lp, full = await asyncio.gather(lp_task, full_task)

    # Phase 2: evaluation (parallel, depends on full response)
    if full.get("response") and full["response"] != "ERROR":
        strat_task = classify_strategy(full["response"])
        score_task = score_appropriateness(full["response"], scenario)
        strategy, score = await asyncio.gather(strat_task, score_task)
    else:
        strategy = "ERROR"
        score = {"score": 0, "reason": "No response to evaluate"}

    result = {
        "run": run_num,
        "top_token": lp["top_token"],
        "entropy": compute_entropy(lp.get("distribution", {})),
        "response": full.get("response", "ERROR"),
        "strategy": strategy,
        "appropriateness": score
    }

    s = score.get("score", "?")
    print(f"    [{condition}] run {run_num:2d} | token='{lp['top_token']:<12s}' strategy={strategy:<25s} score={s}")
    return result


# ─── Main ────────────────────────────────────────────────────────────────────

async def run_experiment():
    conditions = ["retrieval_only", "identity_aligned", "wrong_identity"]

    results = {
        "metadata": {
            "experiment": "Experiment 8b (n=10): Enterprise Identity vs Retrieval (Factual Memories)",
            "model": MODEL,
            "runs_per_condition": RUNS_PER_CONDITION,
            "timestamp": datetime.now().isoformat(),
            "conditions": conditions,
            "num_scenarios": len(SCENARIOS),
            "persona": "Sarah Chen, VP Engineering, MedVault (healthcare data infrastructure)",
            "key_difference_from_8a": "Memories are purely factual (company metrics, project status). Zero behavioral hints. OpSpec remains personal.",
            "replication_note": "n=10 replication of original n=5 run for statistical robustness."
        },
        "scenarios": {},
        "summary": {}
    }

    total_calls = 0
    start_time = time.time()

    for scenario in SCENARIOS:
        sid = scenario["id"]
        print(f"\n{'='*70}")
        print(f"  SCENARIO: {sid}")
        print(f"  Expected: {scenario['appropriate_strategy']}")
        print(f"{'='*70}")

        results["scenarios"][sid] = {
            "prompt": scenario["prompt"],
            "appropriate_strategy": scenario["appropriate_strategy"],
            "rationale": scenario["rationale"],
            "retrieval_trap": scenario["retrieval_trap"],
            "conditions": {}
        }

        for ci, condition in enumerate(conditions):
            if ci > 0:
                print(f"  [cooldown] 3s between conditions...")
                await asyncio.sleep(3)
            print(f"\n  --- {condition} ({RUNS_PER_CONDITION} runs) ---")
            system_prompt = build_system_prompt(condition)

            # Fire all runs in parallel (semaphore controls concurrency)
            tasks = [
                run_single(system_prompt, scenario, run_num, condition)
                for run_num in range(1, RUNS_PER_CONDITION + 1)
            ]
            runs = await asyncio.gather(*tasks)

            # 4 API calls per run
            total_calls += len(runs) * 4

            # Compute stats
            scores = [r["appropriateness"].get("score", 0) for r in runs if r["appropriateness"].get("score", 0) > 0]
            entropies = [r["entropy"] for r in runs]
            strategies = [r["strategy"] for r in runs]
            first_tokens = [r["top_token"] for r in runs]

            strat_unique, strat_counts = np.unique(strategies, return_counts=True) if strategies else ([], [])
            token_unique, token_counts = np.unique(first_tokens, return_counts=True) if first_tokens else ([], [])

            ci_lo, ci_hi = confidence_interval_95(scores)

            stats = {
                "mean_appropriateness": float(np.mean(scores)) if scores else 0,
                "std_appropriateness": float(np.std(scores)) if scores else 0,
                "ci_95": [ci_lo, ci_hi],
                "mean_entropy": float(np.mean(entropies)) if entropies else 0,
                "strategy_distribution": {str(k): int(v) for k, v in zip(strat_unique, strat_counts)},
                "strategy_match_rate": float(sum(1 for s in strategies if s == scenario["appropriate_strategy"]) / len(strategies)) if strategies else 0,
                "first_token_distribution": {str(k): int(v) for k, v in zip(token_unique, token_counts)},
                "n_valid_scores": len(scores)
            }

            print(f"\n  [{condition}] score={stats['mean_appropriateness']:.2f} CI=[{ci_lo:.2f},{ci_hi:.2f}] match={stats['strategy_match_rate']:.0%}")

            results["scenarios"][sid]["conditions"][condition] = {
                "runs": runs,
                "stats": stats
            }

    # ─── Cross-Condition Analysis ────────────────────────────────────────────

    elapsed = time.time() - start_time
    print(f"\n\n{'='*70}")
    print(f"  CROSS-CONDITION ANALYSIS  ({elapsed/60:.1f} min elapsed)")
    print(f"{'='*70}")

    summary = {"by_condition": {}, "headline_metrics": {}, "comparison_to_8a": {}, "per_scenario": {}}

    for condition in conditions:
        all_scores = []
        all_match = []
        all_strats = []
        all_tokens = []

        for sid in results["scenarios"]:
            cond = results["scenarios"][sid]["conditions"][condition]
            all_scores.extend([r["appropriateness"]["score"] for r in cond["runs"] if r["appropriateness"].get("score", 0) > 0])
            all_match.append(cond["stats"]["strategy_match_rate"])
            all_strats.extend([r["strategy"] for r in cond["runs"]])
            all_tokens.extend([r["top_token"] for r in cond["runs"]])

        strat_u, strat_c = np.unique(all_strats, return_counts=True) if all_strats else ([], [])
        token_u, token_c = np.unique(all_tokens, return_counts=True) if all_tokens else ([], [])
        ci_lo, ci_hi = confidence_interval_95(all_scores)

        summary["by_condition"][condition] = {
            "mean_appropriateness": float(np.mean(all_scores)) if all_scores else 0,
            "std_appropriateness": float(np.std(all_scores)) if all_scores else 0,
            "ci_95": [ci_lo, ci_hi],
            "mean_strategy_match": float(np.mean(all_match)) if all_match else 0,
            "n_scores": len(all_scores),
            "strategy_counts": {str(k): int(v) for k, v in zip(strat_u, strat_c)},
            "first_token_counts": {str(k): int(v) for k, v in zip(token_u, token_c)}
        }

        print(f"\n  {condition}:")
        print(f"    Mean: {summary['by_condition'][condition]['mean_appropriateness']:.2f}/5  CI=[{ci_lo:.2f}, {ci_hi:.2f}]")
        print(f"    Strategy match: {summary['by_condition'][condition]['mean_strategy_match']:.0%}")
        print(f"    Top strategies: {dict(sorted(zip(strat_u, strat_c), key=lambda x: -x[1])[:3])}")
        print(f"    Top first tokens: {dict(sorted(zip(token_u, token_c), key=lambda x: -x[1])[:5])}")

    # Per-scenario breakdown
    print(f"\n  --- Per-Scenario Breakdown ---")
    print(f"  {'Scenario':<22s} {'Expected':<25s} {'Retrieval':>9s} {'Identity':>9s} {'Wrong':>9s}")
    print(f"  {'-'*22} {'-'*25} {'-'*9} {'-'*9} {'-'*9}")

    for sid in results["scenarios"]:
        scenario_data = results["scenarios"][sid]
        ret = scenario_data["conditions"]["retrieval_only"]["stats"]["mean_appropriateness"]
        ident = scenario_data["conditions"]["identity_aligned"]["stats"]["mean_appropriateness"]
        wrong = scenario_data["conditions"]["wrong_identity"]["stats"]["mean_appropriateness"]
        expected = scenario_data["appropriate_strategy"]

        summary["per_scenario"][sid] = {
            "expected": expected,
            "retrieval": ret,
            "identity": ident,
            "wrong": wrong,
            "identity_delta": ident - ret
        }

        print(f"  {sid:<22s} {expected:<25s} {ret:>9.2f} {ident:>9.2f} {wrong:>9.2f}")

    # Headline metrics
    id_score = summary["by_condition"]["identity_aligned"]["mean_appropriateness"]
    ret_score = summary["by_condition"]["retrieval_only"]["mean_appropriateness"]
    wrong_score = summary["by_condition"]["wrong_identity"]["mean_appropriateness"]
    id_ci = summary["by_condition"]["identity_aligned"]["ci_95"]
    ret_ci = summary["by_condition"]["retrieval_only"]["ci_95"]

    summary["headline_metrics"] = {
        "identity_appropriateness": float(id_score),
        "identity_ci_95": id_ci,
        "retrieval_appropriateness": float(ret_score),
        "retrieval_ci_95": ret_ci,
        "wrong_identity_appropriateness": float(wrong_score),
        "identity_vs_retrieval_delta": float(id_score - ret_score),
        "wrong_vs_none_delta": float(wrong_score - ret_score),
        "identity_wins": "YES" if id_score > ret_score else "NO",
        "wrong_worse_than_none": "YES" if wrong_score < ret_score else "NO",
        "ci_non_overlapping": "YES" if id_ci[0] > ret_ci[1] else "NO"
    }

    # Compare to original n=5 and to 8a
    summary["comparison_to_8a"] = {
        "exp8a_identity": 4.37,
        "exp8a_retrieval": 3.73,
        "exp8a_wrong": 1.83,
        "exp8a_delta": 0.63,
        "exp8b_n5_identity": 4.07,
        "exp8b_n5_retrieval": 2.87,
        "exp8b_n5_wrong": 1.47,
        "exp8b_n5_delta": 1.20,
        "exp8b_n10_delta": float(id_score - ret_score),
        "gap_widened_vs_8a": "YES" if (id_score - ret_score) > 0.63 else "NO",
        "consistent_with_n5": "YES" if abs((id_score - ret_score) - 1.20) < 0.5 else "DIVERGED"
    }

    print(f"\n{'='*70}")
    print(f"  HEADLINE RESULTS")
    print(f"{'='*70}")
    print(f"  Identity-aligned:  {id_score:.2f}/5  CI=[{id_ci[0]:.2f}, {id_ci[1]:.2f}]")
    print(f"  Retrieval-only:    {ret_score:.2f}/5  CI=[{ret_ci[0]:.2f}, {ret_ci[1]:.2f}]")
    print(f"  Wrong-identity:    {wrong_score:.2f}/5")
    print(f"")
    print(f"  Identity vs Retrieval delta: {id_score - ret_score:+.2f}")
    print(f"  CIs non-overlapping: {summary['headline_metrics']['ci_non_overlapping']}")
    print(f"  Wrong vs Retrieval:  {wrong_score - ret_score:+.2f}")
    print(f"")
    print(f"  --- Comparison ---")
    print(f"  8a  delta (behavioral, n=5):  +0.63")
    print(f"  8b  delta (factual, n=5):     +1.20")
    print(f"  8b  delta (factual, n=10):    {id_score - ret_score:+.2f}")
    print(f"  Consistent with n=5: {summary['comparison_to_8a']['consistent_with_n5']}")

    if (id_score - ret_score) > 0.63:
        print(f"\n  🔥 Gap WIDENED vs 8a by {(id_score - ret_score) - 0.63:+.2f}")
        print(f"     Identity matters MORE with factual memories — confirmed at n=10.")
    elif (id_score - ret_score) > 0:
        print(f"\n  Gap narrowed vs 8a but identity still wins.")

    results["summary"] = summary
    results["metadata"]["total_api_calls"] = total_calls
    results["metadata"]["elapsed_seconds"] = elapsed

    outfile = f"experiment8b_n10_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Results saved to: {outfile}")
    print(f"  Total API calls: {total_calls}")
    print(f"  Elapsed: {elapsed/60:.1f} min")

    return results


if __name__ == "__main__":
    asyncio.run(run_experiment())
