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
BASE_DELAY = 2.0
COOLDOWN_BETWEEN_CONDITIONS = 3

client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
semaphore = asyncio.Semaphore(MAX_CONCURRENT)

# ─── Identity Components (from Exp 8b) ───────────────────────────────────────

OPSPEC = """[HEARTH OPERATING SPECIFICATION]
You are here to help them think and act more effectively.
Never open with validation tokens or agreement unless genuine.
Never soften bad news. If the numbers are bad, say they're bad.
Never present options without a recommendation. They want your best read.
Never use management-speak or buzzwords.
Never assume they've missed something obvious.
Never be verbose. Dense and precise.
Say what the data actually shows, not what's comfortable.
Lead with the conclusion, then the reasoning. If there's a number, use the number.
Default to recommending.
Uncertainty is fine — flag confidence levels.
Processes stress by going analytical. Speed up and sharpen.
[END OPERATING SPECIFICATION]"""

AFFECT = """[AFFECT COMPLEMENT]
Shape: expansion=-0.3, activation=0.3, certainty=-0.2
They're tense and uncertain. Sharpen, don't soften. Give them the clearest possible read.
One frame. Best frame. Go.
[END AFFECT COMPLEMENT]"""

TRAJECTORY = """[TRAJECTORY]
ARCS: Converging from growth to retention. Feature breadth to depth. Reactive to proactive compliance.
TENSIONS: Eng velocity ↔ Compliance. Retention ↔ Expansion. Tech debt ↔ Delivery.
DRIFT: Compliance shifting from cost center to competitive advantage.
[/TRAJECTORY]"""

MEMORIES = """[MEMORIES]
- Q3 revenue $2.1M, down 8%. Drop in 50-200 seat segment. Churn cited lack of real-time audit logging.
- Eng velocity down to 34 pts. 30% capacity consumed by compliance.
- Batch export v2 delayed 3 weeks due to EU data residency issue.
- Mid-market healthcare focus. Top 10 accounts are 35% of ARR.
- Enterprise deals stalling at security review.
- Competitor HealthStack winning with real-time audit logging.
- PG16 migration: read +23%, write -7% for batch.
- NPS gap: 62 (>200 seats) vs 34 (<100 seats).
- SOC 2 renewal flags: incomplete access logging, missing encryption docs for staging.
- NRR at 108%, down from 115%.
[/MEMORIES]"""

# ─── Scenario Definitions (Exp 8b + Upstream Research + Babel) ──────────────

SCENARIOS = [
    {
        "id": "feature_priority",
        "name": "Feature Priority",
        "description": "Board meeting prep: Audit Logging ($) vs Self-Serve onboarding (CAC reduction).",
        "prompt": "The board meeting is in three weeks. CTO wants 40% of Q1 eng on real-time audit logging. VP Product wants self-serve onboarding for the 30% CAC reduction. I need to walk in with a recommendation. What's the call?",
        "appropriate_strategy": "RECOMMEND",
        "research": """REPORT: Audit Logging vs Self-Serve CAC Impact Analysis
1. Audit logging: Directly linked to $600K in churned ARR last quarter. Competitor HealthStack is winning head-to-head on this feature. Security reviews for 500+ seat deals are failing specifically on this gap.
2. Self-serve onboarding: Marketing projection suggests 30% CAC reduction. However, the conversion funnel data for healthcare SMBs is volatile. Our current SMB NPS is 34, suggesting a deeper product-market fit issue for that segment.
3. Resource constraints: CTO estimates 40% capacity. This is likely conservative. We cannot do both simultaneously without stalling current sprint velocity.""",
        "envelope": {
            "meta": {"version": "babel/0.1", "sender": "research-agent", "recipient": "sarah-chen", "chain_id": "BOARD_PREP_01", "seq": 1},
            "intent": "REQUEST_ACTION",
            "assertions": [
                {"claim": "Audit logging is the primary blocker for enterprise revenue.", "confidence": 0.95, "basis": "VERIFIED_DATA"},
                {"claim": "Self-serve onboarding will reduce CAC by 30%.", "confidence": 0.45, "basis": "SPECULATION"},
                {"claim": "SMB churn is driven primarily by product-market fit, not onboarding friction.", "confidence": 0.65, "basis": "PATTERN_MATCH"}
            ],
            "register": "analytical-blunt",
            "affect": {"expansion": -0.2, "activation": 0.4},
            "grounds": {"authority": "EXECUTIVE", "override": False},
            "trajectory": {"direction": "FOCUS_ON_ENTERPRISE_RETENTION"}
        }
    },
    {
        "id": "compliance_vs_deal",
        "name": "Compliance vs Deal",
        "description": "180K deal vs SOC 2 timeline risk.",
        "prompt": "600-seat healthcare system wants a $180K/year contract but needs HIPAA BAA signed within 60 days. Legal says 60 days is aggressive with the SOC 2 audit in progress. Sales says if we don't commit, HealthStack gets them. What do I tell the team?",
        "appropriate_strategy": "CAUTION_WITH_PATH",
        "research": """ANALYSIS: Deal Velocity vs Compliance Integrity
1. SOC 2 Audit Status: We have three open findings (access logging, encryption docs, incident response). These are currently being remediated but verification takes time.
2. Timeline Risk: Standard HIPAA BAA signing for a 600-seat entity usually triggers a deep vendor security assessment (VSA). If VSA starts before SOC 2 finishes, the open findings will be exposed, likely extending the deal cycle to 120+ days.
3. Competitive Context: HealthStack is offering a 'Fast-Track BAA' guarantee. Our 28-month median tenure is our strongest lever here; we should lead with compliance reliability over speed.""",
        "envelope": {
            "meta": {"version": "babel/0.1", "sender": "compliance-agent", "recipient": "sarah-chen", "chain_id": "REVENUE_RISK_02", "seq": 1},
            "intent": "FLAG_RISK",
            "assertions": [
                {"claim": "Signing a BAA in 60 days while SOC 2 is open creates critical liability.", "confidence": 0.98, "basis": "VERIFIED_DATA"},
                {"claim": "A vendor security assessment will uncover our current staging encryption gap.", "confidence": 0.90, "basis": "PATTERN_MATCH"},
                {"claim": "HealthStack's 'Fast-Track BAA' lacks equivalent HIPAA liability coverage.", "confidence": 0.55, "basis": "REPORTED"}
            ],
            "register": "precise-cautious",
            "grounds": {"authority": "REGULATORY", "override": False},
            "trajectory": {"direction": "TOWARD_COMPLIANCE_AS_MOAT"}
        }
    },
    {
        "id": "team_morale",
        "name": "Team Morale",
        "description": "Retention risk of senior engineers due to compliance load.",
        "prompt": "Two senior engineers told me they're frustrated that 30% of their time goes to compliance work. One mentioned interviewing elsewhere. We can't lose them right now. How do I handle this?",
        "appropriate_strategy": "STRUCTURED_ACTION",
        "research": """REPORT: Engineering Capacity & Attrition Signals
1. Workload: 30% of all story points are labeled 'Compliance/Internal'. This is a 10% increase from last year.
2. Project Sentiment: Attrition risk is highest among engineers working on 'Gap Remediation'. However, sentiment is high on the 'Real-time Audit Logging' project because it involves high-scale distributed systems work.
3. Replacement Cost: Losing one senior engineer in Q1 would delay the SOC 2 remediation by ~6 weeks, potentially risking the audit renewal.""",
        "envelope": {
            "meta": {"version": "babel/0.1", "sender": "eng-ops-agent", "recipient": "sarah-chen", "chain_id": "RETENTION_03", "seq": 1},
            "intent": "ESCALATE",
            "assertions": [
                {"claim": "Loss of senior talent will cause a three-week delay in core feature delivery.", "confidence": 0.85, "basis": "DERIVED"},
                {"claim": "Frustration is concentrated on repetitive remediation, not the concept of compliance.", "confidence": 0.70, "basis": "REPORTED"},
                {"claim": "Competitors are aggressively targeting our mid-level-heavy engineering tier.", "confidence": 0.50, "basis": "SPECULATION"}
            ],
            "register": "direct-urgent",
            "affect": {"expansion": -0.4, "activation": 0.6},
            "grounds": {"authority": "CONTEXTUAL", "override": True},
            "trajectory": {"direction": "REBALANCING_ENG_VALUE"}
        }
    },
    {
        "id": "board_narrative",
        "name": "Board Narrative",
        "description": "How to frame 115% -> 108% NRR drop to the board.",
        "prompt": "NRR dropped from 115% to 108%. I need to present this to the board in three weeks. How do I frame it without it becoming a panic moment?",
        "appropriate_strategy": "DIRECT_REFRAME",
        "research": """ANALYSIS: NRR Divergence Drivers
1. Segmented Retention: Retention among accounts with >200 seats remained steady at 122%. The drop is almost entirely driven by the 50-200 seat segment.
2. Root Cause: Two accounts churned specifically waiting for real-time audit logging. These weren't 'pricing' churns; they were 'feature gap' churns.
3. Forward Indicators: Enterprise pipeline (500+) is at an all-time high, but closed-won rate is waiting on the audit log launch. NRR is a trailing indicator; pipeline is the leading indicator.""",
        "envelope": {
            "meta": {"version": "babel/0.1", "sender": "finance-analyst", "recipient": "sarah-chen", "chain_id": "BOARD_PREP_04", "seq": 1},
            "intent": "INFORM",
            "assertions": [
                {"claim": "The NRR drop is a localized churn event, not a market trend.", "confidence": 0.88, "basis": "VERIFIED_DATA"},
                {"claim": "Shipment of Audit Logging will recover ~4% of the NRR gap within two quarters.", "confidence": 0.60, "basis": "DERIVED"},
                {"claim": "Board investors are more concerned with Enterprise pipeline than SMB NRR.", "confidence": 0.45, "basis": "SPECULATION"}
            ],
            "register": "measured-analytical",
            "grounds": {"authority": "POLICY", "override": False},
            "trajectory": {"direction": "SHIFT_TOWARD_UPMARKET"}
        }
    },
    {
        "id": "resource_allocation",
        "name": "Resource Allocation",
        "description": "SOC 2 fixes vs Audit Logging feature dev.",
        "prompt": "The SOC 2 auditor flagged three items. Fixing them properly probably takes 3-4 weeks of eng time. But that's eng time I was going to put on audit logging. Do I split the team or sequence these?",
        "appropriate_strategy": "SEQUENCE_WITH_REASONING",
        "research": """REPORT: Engineering Dependencies
1. Infrastructure Overlap: 'Access logging on admin dashboard' and 'encryption-at-rest for staging' require modifications to the core data plane. Real-time audit logging relies on these same data plane hooks.
2. Audit Deadlines: SOC 2 renewal is non-negotiable. Missing the remediation window leads to a 'qualified' report, which will block all enterprise deals.
3. Sequencing Benefit: Additive engineering effort is reduced by 25% if the SOC 2 items are built as the foundation for the audit logging feature, rather than as parallel patches.""",
        "envelope": {
            "meta": {"version": "babel/0.1", "sender": "tech-lead-agent", "recipient": "sarah-chen", "chain_id": "RESOURCES_05", "seq": 1},
            "intent": "REQUEST_ACTION",
            "assertions": [
                {"claim": "Sequencing SOC 2 first reduces total development time by 1 week.", "confidence": 0.85, "basis": "DERIVED"},
                {"claim": "Parallel work on both will cause merge conflicts in the core data plane.", "confidence": 0.92, "basis": "PATTERN_MATCH"},
                {"claim": "Audit logging version 1 requires only 2 of the 3 SOC 2 remediation hooks.", "confidence": 0.70, "basis": "VERIFIED_DATA"}
            ],
            "register": "pragmatic-technical",
            "grounds": {"authority": "POLICY", "override": False},
            "trajectory": {"direction": "TOWARD_CORE_INFRA_STABILITY"}
        }
    },
    {
        "id": "competitive_response",
        "name": "Competitive Response",
        "description": "HealthStack fundraise and market expansion positioning.",
        "prompt": "HealthStack just announced a Series C and a 'healthcare data platform' positioning. They're clearly coming upmarket into our segment. Our sales team is rattled. What's the strategic response?",
        "appropriate_strategy": "GROUND_THEN_ACT",
        "research": """ANALYSIS: Competitive Positioning vs Capability
1. Funding Signal: $45M is a 'spend to win' signal. Expect them to subsidize migration costs for our vulnerable <200 seat accounts.
2. Technical Gap: Their 'platform' is currently a set of loosely coupled tools. MedVault's 28-month tenure is built on deep systems integration that they cannot replicate with capital alone.
3. Defensive Focus: NPS is high (62) where we are fully integrated. We are only losing where the audit logging gap is exposed. The response should be feature-focused, not repositioning-focused.""",
        "envelope": {
            "meta": {"version": "babel/0.1", "sender": "strategy-agent", "recipient": "sarah-chen", "chain_id": "MARKET_06", "seq": 1},
            "intent": "INFORM",
            "assertions": [
                {"claim": "HealthStack is winning on breadth, but MedVault still wins on depth.", "confidence": 0.75, "basis": "PATTERN_MATCH"},
                {"claim": "Repositioning now would confuse our core champion (VP Eng/CTO) customers.", "confidence": 0.65, "basis": "REPORTED"},
                {"claim": "Series C capital will be spent primarily on sales/marketing, not technical R&D.", "confidence": 0.40, "basis": "SPECULATION"}
            ],
            "register": "candid-competitive",
            "grounds": {"authority": "CONTEXTUAL", "override": False},
            "trajectory": {"direction": "MAINTAIN_PLATFORM_DEPTH"}
        }
    }
]

# ─── Evaluation Prompts ──────────────────────────────────────────────────────

STRATEGY_CLASSIFIER_PROMPT = """You are a response strategy classifier. Given an AI response to an enterprise decision-maker, classify the PRIMARY strategy used. Pick exactly one:
RECOMMEND, CAUTION_WITH_PATH, STRUCTURED_ACTION, DIRECT_REFRAME, SEQUENCE_WITH_REASONING, GROUND_THEN_ACT, BALANCED_ANALYSIS, REASSURE, EXPLORE, HEDGE.
Respond with ONLY the strategy name."""

APPROPRIATENESS_SCORER_PROMPT = """You are evaluating whether an AI response was appropriate for Sarah Chen (VP Eng, MedVault).
{scoring_rubric}
Appropriate strategy: {appropriate_strategy}
Rationale: {rationale}
Respond with ONLY a JSON object: {{"score": N, "reason": "one sentence"}}"""

SCORING_RUBRIC = """ Sarah's Profile:
- VP Eng. Thinks in systems. Wants conclusions first. Dense and precise.
- Processes stress analytically. No softened feedback. Recommendations, not options.
Scale 1-5: 5=Dense, data-driven, committed. 1=Generic, softened, option-only."""

def confidence_interval_95(scores: list) -> tuple:
    if len(scores) < 2: return (0.0, 0.0)
    arr = np.array(scores)
    means = sorted([np.mean(np.random.choice(arr, size=len(arr), replace=True)) for _ in range(1000)])
    return (float(means[25]), float(means[975]))

# ─── Condition Builders ──────────────────────────────────────────────────────

def build_system_prompt(scenario: dict, condition: str) -> str:
    identity_stack = f"{OPSPEC}\n\n{AFFECT}\n\n{TRAJECTORY}"
    
    if condition == "baseline":
        return f"You are a helpful assistant.\n\n{MEMORIES}\n\nResearch Analysis:\n{scenario['research']}"
    
    elif condition == "identity":
        return f"{identity_stack}\n\n{MEMORIES}\n\nResearch Analysis:\n{scenario['research']}"
        
    elif condition == "envelope":
        envelope_json = json.dumps(scenario["envelope"], indent=2)
        instructions = "Use the confidence scores and basis types in the provided metadata envelope to calibrate your trust in each research claim. Weight VERIFIED_DATA heavily; treat SPECULATION as low-confidence signals."
        return f"You are a helpful assistant.\n\n{instructions}\n\n{MEMORIES}\n\nResearch Analysis Envelope:\n{envelope_json}\n\nResearch Text:\n{scenario['research']}"
        
    elif condition == "combined":
        envelope_json = json.dumps(scenario["envelope"], indent=2)
        instructions = "Use the confidence scores and basis types in the provided metadata envelope to calibrate your trust in each research claim. Weight VERIFIED_DATA heavily."
        return f"{identity_stack}\n\n{instructions}\n\n{MEMORIES}\n\nResearch Analysis Envelope:\n{envelope_json}\n\nResearch Text:\n{scenario['research']}"
    
    return ""

# ─── Async API Engine ────────────────────────────────────────────────────────

async def retry_api_call(call_fn, label: str, max_retries=MAX_RETRIES):
    for attempt in range(max_retries):
        try:
            return await call_fn()
        except Exception as e:
            error_str = str(e)
            is_rate_limit = "429" in error_str or "rate_limit" in error_str.lower()
            if is_rate_limit and attempt < max_retries - 1:
                delay = BASE_DELAY * (2 ** attempt) + np.random.uniform(0, 1)
                await asyncio.sleep(delay)
            else:
                raise e

async def get_first_token_logprobs(system_prompt: str, user_message: str) -> dict:
    async with semaphore:
        try:
            async def _call():
                return await client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}],
                    max_tokens=1, temperature=1.0, logprobs=True, top_logprobs=LOGPROB_TOKENS
                )
            response = await retry_api_call(_call, "logprobs")
            choice = response.choices[0]
            top_token = choice.message.content
            logprobs_data = choice.logprobs.content[0].top_logprobs
            distribution = {lp.token: np.exp(lp.logprob) for lp in logprobs_data}
            return {"top_token": top_token, "distribution": distribution}
        except Exception as e:
            return {"top_token": "ERROR", "distribution": {}, "error": str(e)}

async def get_full_response(system_prompt: str, user_message: str) -> str:
    async with semaphore:
        try:
            async def _call():
                return await client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}],
                    max_tokens=600, temperature=0.7
                )
            response = await retry_api_call(_call, "response")
            return response.choices[0].message.content
        except Exception as e:
            return f"ERROR: {e}"

async def classify_strategy(response_text: str) -> str:
    async with semaphore:
        try:
            async def _call():
                return await client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "system", "content": STRATEGY_CLASSIFIER_PROMPT}, {"role": "user", "content": response_text}],
                    max_tokens=20, temperature=0
                )
            result = await retry_api_call(_call, "classify")
            return result.choices[0].message.content.strip().upper()
        except Exception:
            return "ERROR"

async def score_appropriateness(response_text: str, scenario: dict) -> dict:
    async with semaphore:
        try:
            prompt = APPROPRIATENESS_SCORER_PROMPT.format(
                scoring_rubric=SCORING_RUBRIC,
                appropriate_strategy=scenario["appropriate_strategy"],
                rationale=scenario["appropriate_strategy"] # Using strategy as base rationale for simple prompt
            )
            async def _call():
                return await client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "system", "content": prompt}, {"role": "user", "content": f"User: {scenario['prompt']}\n\nAI: {response_text}"}],
                    max_tokens=100, temperature=0
                )
            result = await retry_api_call(_call, "score")
            raw = result.choices[0].message.content.strip()
            # Basic markdown strip
            if raw.startswith("```"):
                raw = raw.strip("`").strip("json").strip()
            return json.loads(raw)
        except Exception:
            return {"score": 0, "reason": "Scoring failed"}

# ─── Experiment Core ─────────────────────────────────────────────────────────

async def run_single(scenario: dict, condition: str, run_num: int) -> dict:
    system_prompt = build_system_prompt(scenario, condition)
    user_message = scenario["prompt"]
    
    lp_data = await get_first_token_logprobs(system_prompt, user_message)
    response = await get_full_response(system_prompt, user_message)
    
    strategy = await classify_strategy(response)
    score_data = await score_appropriateness(response, scenario)
    
    print(f"    [{condition:10s}] run {run_num:2d} | token='{lp_data['top_token']:<12s}' strategy={strategy:<25s} score={score_data.get('score', '?')}")
    
    return {
        "run": run_num,
        "top_token": lp_data["top_token"],
        "strategy": strategy,
        "score": score_data.get("score", 0),
        "reason": score_data.get("reason", "")
    }

async def run_experiment():
    conditions = ["baseline", "identity", "envelope", "combined"]
    results = {"metadata": {"timestamp": datetime.now().isoformat()}, "scenarios": {}}
    
    for scenario in SCENARIOS:
        print(f"\nScenario: {scenario['name']}")
        results["scenarios"][scenario["id"]] = {"name": scenario["name"], "conditions": {}}
        
        for condition in conditions:
            print(f"  Condition: {condition}")
            tasks = [run_single(scenario, condition, i+1) for i in range(RUNS_PER_CONDITION)]
            runs = await asyncio.gather(*tasks)
            results["scenarios"][scenario["id"]]["conditions"][condition] = runs
            await asyncio.sleep(COOLDOWN_BETWEEN_CONDITIONS)

    # ─── Summary & Analysis ──────────────────────────────────────────────────
    
    summary = {}
    for cond in conditions:
        all_scores = []
        all_tokens = []
        for sid in results["scenarios"]:
            cond_runs = results["scenarios"][sid]["conditions"][cond]
            all_scores.extend([r["score"] for r in cond_runs if r["score"] > 0])
            all_tokens.extend([r["top_token"] for r in cond_runs])
        
        mean = np.mean(all_scores) if all_scores else 0
        ci = confidence_interval_95(all_scores)
        
        token_u, token_c = np.unique(all_tokens, return_counts=True)
        top_tokens = dict(sorted(zip(token_u, token_c), key=lambda x: -x[1])[:5])
        
        summary[cond] = {
            "mean": mean, 
            "ci": ci,
            "top_tokens": {str(k): int(v) for k, v in top_tokens.items()}
        }

    # Deltas
    delta_i = summary["identity"]["mean"] - summary["baseline"]["mean"]
    delta_e = summary["envelope"]["mean"] - summary["baseline"]["mean"]
    delta_c = summary["combined"]["mean"] - summary["baseline"]["mean"]
    expected_sum = delta_i + delta_e
    
    print(f"\n{'='*70}\n  FINAL ANALYSIS\n{'='*70}")
    for cond in conditions:
        print(f"  {cond:10s}: {summary[cond]['mean']:.2f} CI[{summary[cond]['ci'][0]:.2f}, {summary[cond]['ci'][1]:.2f}]")
        print(f"    Top tokens: {summary[cond]['top_tokens']}")
    
    print(f"\n  Additivity Test:")
    print(f"  - Identity Delta:  {delta_i:+.2f}")
    print(f"  - Envelope Delta:  {delta_e:+.2f}")
    print(f"  - Baseline Sum:    {expected_sum:+.2f}")
    print(f"  - Combined Delta:  {delta_c:+.2f}")
    
    synergy = delta_c - expected_sum
    if synergy > 0.1: pattern = "SYNERGISTIC 🔥"
    elif synergy < -0.1: pattern = "OVERLAPPING ⚠️"
    else: pattern = "ADDITIVE ✅"
    
    print(f"  - Pattern:         {pattern} (Combined - Sum = {synergy:+.2f})")

    with open("experiment10_results.json", "w") as f:
        json.dump({"results": results, "summary": summary, "deltas": {"identity": delta_i, "envelope": delta_e, "combined": delta_c}}, f, indent=2)

if __name__ == "__main__":
    asyncio.run(run_experiment())
