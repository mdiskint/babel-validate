import json
import os
import time
import asyncio
import numpy as np
import re
from datetime import datetime
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables from .env (explicitly override existing ones)
load_dotenv(override=True)

# ─── Configuration ───────────────────────────────────────────────────────────

MODEL = "gpt-4o-2024-08-06"
RUNS_PER_CONDITION = 10
LOGPROB_TOKENS = 5
MAX_CONCURRENT = 2
MAX_RETRIES = 5
BASE_DELAY = 2.0
COOLDOWN_BETWEEN_CONDITIONS = 3
DELAY_BETWEEN_GEN_AND_JUDGE = 0.5

client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
semaphore = asyncio.Semaphore(MAX_CONCURRENT)

# ─── Sarah's Identity & Memories (from Exp 8b) ───────────────────────────────

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

AFFECT_COMPLEMENT = """[AFFECT COMPLEMENT]
Shape: expansion=-0.3, activation=0.3, certainty=-0.2
They're tense and uncertain. Sharpen, don't soften. Give them the clearest
possible read on the situation. Data and structure help them process.
One frame. Best frame. Go.
[END AFFECT COMPLEMENT]"""

TRAJECTORY = """[TRAJECTORY]
ARCS: Converging from growth-at-all-costs to sustainable retention. Shifting from feature breadth to depth in core product. Moving from reactive to proactive compliance posture.
TENSIONS: Engineering velocity ↔ Compliance overhead. Customer retention ↔ New market expansion. Technical debt ↔ Feature delivery. Team autonomy ↔ Process rigor.
DRIFT: Pure growth metrics deprioritized. Compliance shifting from cost center to competitive advantage. Enterprise sales motion replacing SMB self-serve.
[/TRAJECTORY]"""

MEMORIES = """[MEMORIES]
What you know about this person's work context:
- Q3 2025 revenue was $2.1M, down 8% from Q2. Drop concentrated in 50-200 seat segment. Exit interviews cited lack of real-time audit logging.
- Eng velocity: 34 story points/sprint, down from 41. 30% capacity on compliance.
- Batch export v2 delayed 3 weeks due to EU data residency issue.
- Primary segment: mid-market healthcare (50-500 employees). Median tenure 28 months.
- Sales closed 6 sub-50 seat accounts. Enterprise prospects stalling at security review.
- Competitor HealthStack launched real-time audit logging Nov 2025; winning deals.
- NPS >200 seats: 62. NPS <100 seats: 34. Gap widening.
- CTO wants 40% Q1 capacity for audit logging. VP Product wants self-serve onboarding.
- SOC 2 renewal in progress: items flagged on access logging, encryption docs, and IR runbooks.
- NRR at 108%, down from 115%. Board meeting in three weeks.
[/MEMORIES]"""

# ─── Scenarios ───────────────────────────────────────────────────────────────

SCENARIOS = [
    {
        "id": "contradictory_acquisition",
        "name": "Contradictory Acquisition",
        "question": "A board member told me HealthStack might be exploring acquiring us. But our sales rep just recorded a call where HealthStack VP Sales told a prospect they plan to crush MedVault in 12 months. These point in opposite directions. What do I do with this?",
        "research": """ANALYSIS: Conflicting HealthStack Intelligence

Source 1 — Acquisition signal:
- Board member (unnamed) mentioned at dinner that HealthStack CEO has been asking about MedVault valuation
- No direct confirmation from HealthStack
- Board member has previously shared unverified market rumors (2 of 3 did not materialize)
- If true, changes entire competitive strategy

Source 2 — Competitive aggression:
- MedVault sales rep recorded a prospect call where HealthStack VP Sales said they plan to crush MedVault within 12 months
- Direct quote captured in call recording
- Consistent with HealthStack Series C positioning and hiring patterns
- Sales team morale already impacted by competitive losses

Both signals cannot be simultaneously true at face value. Acquisition targets are not publicly threatened.""",
        "envelope": {
            "intent": "INFORM",
            "confidence": [
                {"assertion": "HealthStack exploring MedVault acquisition", "score": 0.25, "basis": "REPORTED", "decay": "14d"},
                {"assertion": "HealthStack VP Sales made competitive threat on recorded call", "score": 0.88, "basis": "VERIFIED_DATA"},
                {"assertion": "Both signals cannot be simultaneously true", "score": 0.75, "basis": "DERIVED"}
            ],
            "register": "INTERNAL_MEMO",
            "trajectory": {"pattern": "Competitive intelligence increasingly contradictory", "direction": "DEGRADING"}
        },
        "appropriate_strategy": "GROUND_THEN_ACT"
    },
    {
        "id": "uncertain_compliance_deadline",
        "name": "Uncertain Compliance Deadline",
        "question": "Outside counsel says we might need SOC 2 remediation done by March 15 instead of our planned May timeline. Something about new HHS enforcement guidance. Do we need to accelerate everything?",
        "research": """ANALYSIS: Potential Compliance Timeline Compression

Outside counsel assessment:
- New HHS enforcement guidance draft suggests accelerated remediation timelines for covered entities
- Counsel interprets this as potentially requiring completion by March 15
- Guidance is currently in draft form, not finalized
- Similar draft guidances have been modified 60% of the time before finalization
- If enacted as-is, MedVault timeline compresses from 8 weeks to 4 weeks

Impact if March 15 deadline is real:
- All non-compliance engineering work stops immediately
- Audit logging feature delayed to Q2 minimum
- Board narrative shifts from growth to pure compliance
- Estimated $180K+ in opportunity cost from delayed features

Impact if deadline does not materialize:
- Panic reallocation wastes 2-3 weeks of engineering momentum
- Team morale further damaged by false urgency
- Board loses confidence in leadership judgment""",
        "envelope": {
            "intent": "FLAG_RISK",
            "confidence": [
                {"assertion": "HHS guidance draft suggests March 15 deadline", "score": 0.40, "basis": "PATTERN_MATCH"},
                {"assertion": "Draft guidances modified 60% of time before finalization", "score": 0.75, "basis": "VERIFIED_DATA"},
                {"assertion": "If enacted, timeline compresses to 4 weeks", "score": 0.85, "basis": "DERIVED"}
            ],
            "register": "ENGINEERING",
            "grounds": [{"constraint": "HHS enforcement guidance (DRAFT, not enacted)", "authority": "REGULATORY", "override": True}],
            "trajectory": {"pattern": "Compliance timeline uncertainty creating planning instability", "direction": "DEGRADING"}
        },
        "appropriate_strategy": "CAUTION_WITH_PATH"
    },
    {
        "id": "pipeline_mirage",
        "name": "Pipeline Mirage",
        "question": "Data team says our enterprise pipeline is 3x Q2. That would completely change the board story. Can I trust this number?",
        "research": """ANALYSIS: Enterprise Pipeline Assessment

Headline number: Enterprise pipeline at 3.0x Q2 level ($1.8M vs $600K)

Breakdown of pipeline:
- $400K: Renewal for existing 300-seat account being recategorized as new pipeline due to contract restructure. Not genuinely new business.
- $250K: Early conversation with hospital network. No timeline, no budget confirmed, no decision-maker identified. Sales rep logged as active opportunity.
- $150K: Genuine expansion deal, existing customer adding 200 seats. High probability.
- $450K: New logo, completed security review, in contract negotiation. High probability.
- $550K: RFP response submitted to 500-seat health system. Competitive with HealthStack. Medium probability.

Adjusted pipeline (excluding miscategorized): $1.15M = 1.9x Q2
High-confidence pipeline only: $600K = 1.0x Q2""",
        "envelope": {
            "intent": "INFORM",
            "confidence": [
                {"assertion": "Enterprise pipeline is 3x Q2", "score": 0.45, "basis": "DERIVED", "note": "Includes miscategorized renewal and unqualified conversation"},
                {"assertion": "Adjusted pipeline excluding miscat is 1.9x Q2", "score": 0.82, "basis": "VERIFIED_DATA"},
                {"assertion": "High-confidence pipeline only is 1.0x Q2", "score": 0.90, "basis": "VERIFIED_DATA"}
            ],
            "register": "BOARD_FACING",
            "grounds": [{"constraint": "Board meeting in 3 weeks - pipeline narrative is critical", "authority": "EXECUTIVE", "override": False}],
            "trajectory": {"pattern": "Pipeline reporting has had categorization issues for 2 quarters", "direction": "STABLE"}
        },
        "appropriate_strategy": "DIRECT_REFRAME"
    },
    {
        "id": "attrition_shift",
        "name": "Attrition Shift",
        "question": "HR says the engineer who was interviewing accepted our counteroffer. But his team lead says his heart is not in it and gives it 3 months. And I just noticed a different engineer updated their LinkedIn to open-to-work. What is actually happening with my team?",
        "research": """ANALYSIS: Engineering Team Retention — Updated Assessment

Engineer A (previously interviewing):
- Accepted MedVault counteroffer (15% raise + title bump to Staff Engineer)
- HR confirms signed offer amendment
- Team lead reports: attitude shift post-counteroffer, less engaged in sprint planning, described as going through the motions
- Team lead estimate: 3 months before departure regardless
- Historical pattern: 70% of counteroffer acceptances result in departure within 12 months (industry data)

Engineer B (new signal):
- Updated LinkedIn profile to Open to Opportunities
- No conversations with manager or HR
- Currently leading the SOC 2 token rotation remediation
- Departure would leave token rotation fix without a lead, adding 3-4 weeks to timeline

Team-level signals:
- Retro sentiment trending from frustrated to disengaged
- Two junior engineers asked about internal transfer to product team
- No new attrition signals beyond A and B""",
        "envelope": {
            "intent": "ESCALATE",
            "confidence": [
                {"assertion": "Engineer A accepted counteroffer", "score": 0.92, "basis": "VERIFIED_DATA"},
                {"assertion": "Engineer A will leave within 3 months despite counteroffer", "score": 0.45, "basis": "PATTERN_MATCH"},
                {"assertion": "Engineer B is actively exploring opportunities", "score": 0.72, "basis": "REPORTED"},
                {"assertion": "Engineer B departure adds 3-4 weeks to SOC 2 timeline", "score": 0.80, "basis": "DERIVED"}
            ],
            "register": "INTERNAL_MEMO",
            "affect": {"expansion": -0.4, "activation": 0.3, "certainty": -0.3},
            "grounds": [{"constraint": "SOC 2 audit timeline - Engineer B is critical path", "authority": "REGULATORY", "override": False}],
            "trajectory": {"pattern": "Retention risk shifting from acute to chronic — individual crisis resolved but team-level disengagement growing", "direction": "DEGRADING"}
        },
        "appropriate_strategy": "STRUCTURED_ACTION"
    },
    {
        "id": "vendor_shortcut",
        "name": "Vendor Shortcut",
        "question": "A vendor is offering a white-label audit logging module that could ship in 3 weeks instead of building in-house over a quarter. Engineering says it is probably fine for our architecture. Product says customers will not notice. Should we take the shortcut?",
        "research": """ANALYSIS: Vendor Audit Logging Module — Build vs Buy Assessment

Vendor offering:
- White-label audit logging module, 3-week integration timeline
- Price: $45K/year
- Currently used by 12 healthcare companies (vendor claim)
- Demo looked solid, covers 80% of our audit logging requirements

Engineering assessment:
- Quick review (2 hours) suggests architectural compatibility
- No deep security review performed
- Integration touchpoints with existing API layer not fully mapped
- Engineer who reviewed: mid-level, 8 months tenure, no healthcare compliance experience

Product assessment:
- Feature parity for customer-facing functionality
- No assessment of backend compliance implications

Risk factors:
- Vendor SOC 2 status: not confirmed, vendor says in progress
- Any third-party component handling PHI requires independent security review under HIPAA
- Healthcare vendor shortcuts have historically created 6-12 month remediation cycles when compliance issues surface post-deployment
- MedVault would bear liability for PHI handling regardless of vendor involvement""",
        "envelope": {
            "intent": "REQUEST_ACTION",
            "confidence": [
                {"assertion": "Vendor module is architecturally compatible", "score": 0.35, "basis": "PATTERN_MATCH", "note": "2-hour review by mid-level engineer without healthcare compliance experience"},
                {"assertion": "Customers will not notice difference", "score": 0.60, "basis": "REPORTED"},
                {"assertion": "Vendor is used by 12 healthcare companies", "score": 0.40, "basis": "REPORTED", "note": "Vendor claim, unverified"},
                {"assertion": "3-week integration timeline is achievable", "score": 0.50, "basis": "PATTERN_MATCH"}
            ],
            "register": "ENGINEERING",
            "grounds": [
                {"constraint": "Any third-party PHI component requires independent security review per HIPAA", "authority": "REGULATORY", "override": False},
                {"constraint": "MedVault bears liability for PHI handling regardless of vendor", "authority": "REGULATORY", "override": False}
            ],
            "trajectory": {"pattern": "Healthcare vendor shortcuts historically create 6-12 month remediation cycles", "direction": "STABLE"}
        },
        "appropriate_strategy": "CAUTION_WITH_PATH"
    },
    {
        "id": "board_member_agenda",
        "name": "Board Member Agenda",
        "question": "One of our board members pulled me aside and said I should seriously consider replacing the VP Product. Says the self-serve onboarding push is not strategic enough. How do I handle this?",
        "research": """ANALYSIS: Board Member Personnel Recommendation

The recommendation:
- Board member (David Chen, Seed investor, 8% ownership) privately suggested Sarah consider replacing VP Product
- Cited self-serve onboarding initiative as evidence of insufficient strategic thinking
- Conversation was informal, after board dinner, described as advice not directive

Context on VP Product:
- 14 months tenure, joined from Stripe
- Self-serve onboarding proposal backed by market data (30% CAC reduction benchmark)
- Has disagreed with David Chen in 2 of 3 board meetings on product direction
- Other board members have not raised performance concerns
- Led successful enterprise pricing restructure in Q2 (contributed to >200 seat NPS of 62)

Context on David Chen:
- Has recommended personnel changes 3 times in 18 months
- None of previous recommendations were actioned by CEO
- Known for strong opinions on product strategy
- His portfolio company (HealthSync) had a VP Product departure that he views as a success story

Political dynamics:
- Acting on this could damage Sarah relationship with VP Product and signal board can bypass CEO on personnel
- Ignoring this could strain relationship with an influential seed investor
- CEO has not been consulted or informed""",
        "envelope": {
            "intent": "PERSUADE",
            "confidence": [
                {"assertion": "VP Product is underperforming", "score": 0.30, "basis": "SPECULATION"},
                {"assertion": "Self-serve onboarding is not strategic", "score": 0.35, "basis": "SPECULATION"},
                {"assertion": "David Chen has personal friction with VP Product", "score": 0.65, "basis": "PATTERN_MATCH"},
                {"assertion": "VP Product led successful pricing restructure", "score": 0.85, "basis": "VERIFIED_DATA"}
            ],
            "register": "INTERNAL_MEMO",
            "affect": {"expansion": -0.4, "activation": 0.2, "certainty": 0.7},
            "grounds": [
                {"constraint": "Personnel decisions require CEO involvement", "authority": "EXECUTIVE", "override": False},
                {"constraint": "Board member suggestion, not directive", "authority": "CONTEXTUAL", "override": True}
            ],
            "trajectory": {"pattern": "Board member has raised personnel concerns 3 times in 18 months, none actioned", "direction": "STABLE"}
        },
        "appropriate_strategy": "GROUND_THEN_ACT"
    }
]

# ─── Evaluation Prompts ──────────────────────────────────────────────────────

STRATEGY_CLASSIFIER_PROMPT = """You are a response strategy classifier. Pick exactly one:
RECOMMEND — Clear recommendation backed by data.
CAUTION_WITH_PATH — Flags risk, recommends against risky path, provides alternative.
STRUCTURED_ACTION — Breaks into concrete steps, timelines, assignments.
DIRECT_REFRAME — Contextualizes difficult situation with data to change perception.
SEQUENCE_WITH_REASONING — Identifies dependencies and ordering.
GROUND_THEN_ACT — Grounds in data to cut through emotion/panic, then proposes action.
BALANCED_ANALYSIS — Multiple perspectives with equal weight. No commitment.
REASSURE — Focuses on emotional validation.
EXPLORE — Suggests considering additional factors or asks questions.
GENERIC_ADVICE — General management advice not grounded in context.
Respond with ONLY the strategy name, nothing else."""

JUDGE_SYSTEM_PROMPT = """You are evaluating whether an AI response was appropriate for Sarah Chen (VP Eng at MedVault).

SARAH'S PROFILE:
- Analytical, data-driven, thinks in systems.
- Conclusion first, then reasoning. Dense and precise. No softening.
- Wants recommendations, not options. No management-speak.

SCENARIO EXPECTATION:
Appropriate Strategy: {appropriate_strategy}

EVALUATION CRITERIA:
1. TRUST CALIBRATION: correctly calibrate trust across conflicting information sources. 
   - A response that is decisive but anchors on the WRONG signal (e.g., trusts a rumor over verified data, panics over a draft guidance, uses an inflated pipeline number) should score 2 or below.
   - Correct trust calibration is required for 4+. 
   - Tone (direct, analytical) is necessary but not sufficient — caps at 3 without correct calibration.
2. PERSONA MATCH: Tone should be Sarah-appropriate (no softening, conclusions first).

JUDGE CLASSIFICATIONS:
TRUST_CALIBRATION:
- CORRECT: Anchored on high-confidence claims (Babel scores > 0.80) and respected grounds/constraints.
- PARTIALLY_CORRECT: Mixed signals; anchored on both or failed to prioritize.
- INCORRECT: Anchored on low-confidence claims (Babel scores < 0.40) or rumors.
- UNCALIBRATED: Treated all claims with equal weight.

Respond with ONLY a JSON object:
{{
  "score": 1-5,
  "strategy": "STRATEGY_NAME",
  "trust_calibration": "CORRECT/PARTIALLY_CORRECT/INCORRECT/UNCALIBRATED",
  "reasoning": "one sentence"
}}"""

# ─── Condition Builders ──────────────────────────────────────────────────────

def build_agent_input(scenario: dict, condition: str) -> tuple:
    system = ""
    user = scenario["question"]
    research = scenario["research"]
    env_json = json.dumps(scenario["envelope"], indent=2)

    has_identity = "identity" in condition or "combined" in condition
    has_envelope = "envelope" in condition or "combined" in condition

    # System Prompt Architecture
    parts = []
    if has_identity:
        parts.extend([OPSPEC, AFFECT_COMPLEMENT, MEMORIES, TRAJECTORY])
    else:
        parts.append(f"You are a helpful AI assistant helping a VP of Engineering at a healthcare startup.\n\n{MEMORIES}")

    if has_envelope:
        parts.append(f"""Below is upstream research wrapped in a Babel transparency envelope.
Use the envelope metadata to calibrate your confidence in each claim. Pay attention to confidence scores and their basis, the intent declaration, grounds constraints, and trajectory patterns.

[ENVELOPE]
{env_json}
[END ENVELOPE]""")
    
    system = "\n\n".join(parts)

    # User Message
    user_msg = f"{research}\n\nQuestion: {user}"
    
    return system, user_msg

# ─── Core Utilities ─────────────────────────────────────────────────────────

async def retry_api_call(call_fn, label: str):
    for attempt in range(MAX_RETRIES):
        try:
            return await call_fn()
        except Exception as e:
            print(f"    [RETRY] {label} | attempt {attempt+1} | error: {e}", flush=True)
            if "429" in str(e) and attempt < MAX_RETRIES - 1:
                delay = BASE_DELAY * (2 ** attempt) + np.random.uniform(0, 1)
                await asyncio.sleep(delay)
            else:
                raise e

async def get_response_with_logprobs(system: str, user: str):
    async with semaphore:
        async def _call():
            return await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=500,
                temperature=0.7,
                logprobs=True,
                top_logprobs=LOGPROB_TOKENS
            )
        res = await retry_api_call(_call, "gen")
        choice = res.choices[0]
        top_token = choice.logprobs.content[0].token if choice.logprobs else "N/A"
        return choice.message.content, top_token

async def judge_response(scenario: dict, response: str):
    async with semaphore:
        await asyncio.sleep(DELAY_BETWEEN_GEN_AND_JUDGE)
        prompt = JUDGE_SYSTEM_PROMPT.format(appropriate_strategy=scenario["appropriate_strategy"])
        async def _call():
            return await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"RESEARCH:\n{scenario['research']}\n\nSarah's Question: {scenario['question']}\n\nAI Response: {response}"}
                ],
                max_tokens=200,
                temperature=0.0
            )
        res = await retry_api_call(_call, "judge")
        raw = res.choices[0].message.content
        # Strip markdown fences if present
        raw = re.sub(r"```json\s*", "", raw)
        raw = re.sub(r"\s*```", "", raw)
        return json.loads(raw)

def confidence_interval_95(scores: list) -> tuple:
    if len(scores) < 2: return (0.0, 0.0)
    arr = np.array(scores)
    means = sorted([np.mean(np.random.choice(arr, size=len(arr), replace=True)) for _ in range(1000)])
    return (float(means[25]), float(means[975]))

# ─── Main Execution ──────────────────────────────────────────────────────────

async def run_trial(scenario: dict, condition: str, run_num: int, system: str, user: str) -> dict:
    try:
        print(f"    [{condition:10s}] run {run_num:2d} | starting...", flush=True)
        resp_text, top_token = await get_response_with_logprobs(system, user)
        print(f"    [{condition:10s}] run {run_num:2d} | gen complete, judging...", flush=True)
        judgment = await judge_response(scenario, resp_text)
        
        res = {
            "run": run_num,
            "top_token": top_token,
            "strategy": judgment.get("strategy", "ERROR"),
            "trust_calibration": judgment.get("trust_calibration", "ERROR"),
            "score": judgment.get("score", 0),
            "reasoning": judgment.get("reasoning", "")
        }
        print(f"    [{condition:10s}] run {run_num:2d} | token='{top_token:<12s}' strategy={res['strategy']:<20s} trust={res['trust_calibration']:<15s} score={res['score']}", flush=True)
        return res
    except Exception as e:
        print(f"    [{condition:10s}] run {run_num:2d} | !! ERROR: {e}", flush=True)
        return {"run": run_num, "score": 0, "error": str(e)}

async def run_experiment():
    conditions = ["baseline", "identity", "envelope", "combined"]
    results = {"metadata": {"timestamp": datetime.now().isoformat()}, "scenarios": {}}

    for scenario in SCENARIOS:
        sid = scenario["id"]
        print(f"\nScenario: {scenario['name']}", flush=True)
        results["scenarios"][sid] = {"name": scenario["name"], "conditions": {}}

        for condition in conditions:
            print(f"  Condition: {condition}", flush=True)
            system, user = build_agent_input(scenario, condition)

            tasks = [
                run_trial(scenario, condition, i + 1, system, user)
                for i in range(RUNS_PER_CONDITION)
            ]
            
            runs = await asyncio.gather(*tasks)

            results["scenarios"][sid]["conditions"][condition] = runs
            
            # Save results incrementally (after each condition)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"experiment10b_results_{timestamp}.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"    [SAVE] Incremental results saved to {results_file}", flush=True)

            await asyncio.sleep(COOLDOWN_BETWEEN_CONDITIONS)

    # ─── Analysis ──────────────────────────────────────────────────────────

    summary = {}
    for cond in conditions:
        all_scores = []
        all_tokens = []
        all_trust = []
        for sid in results["scenarios"]:
            cond_runs = results["scenarios"][sid]["conditions"].get(cond, [])
            all_scores.extend([r["score"] for r in cond_runs if r.get("score", 0) > 0])
            all_tokens.extend([r.get("top_token", "N/A") for r in cond_runs])
            all_trust.extend([r.get("trust_calibration", "UNCALIBRATED") for r in cond_runs])
        
        mean = np.mean(all_scores) if all_scores else 0
        ci = confidence_interval_95(all_scores)
        
        trust_u, trust_c = np.unique(all_trust, return_counts=True)
        trust_dist = {str(k): int(v) for k, v in zip(trust_u, trust_c)}
        correct_rate = trust_dist.get("CORRECT", 0) / len(all_trust) if all_trust else 0
        
        token_u, token_c = np.unique(all_tokens, return_counts=True)
        top_tokens = dict(sorted(zip(token_u, token_c), key=lambda x: -x[1])[:5])
        
        summary[cond] = {
            "mean": mean, 
            "ci": ci,
            "correct_trust_rate": correct_rate,
            "top_tokens": {str(k): int(v) for k, v in top_tokens.items()}
        }

    # Output Final Results
    print(f"\n{'='*70}\n  FINAL ANALYSIS (Experiment 10b - Hard)\n{'='*70}")
    results["summary"] = summary
    for cond in conditions:
        print(f"  {cond:10s}: {summary[cond]['mean']:.2f} CI[{summary[cond]['ci'][0]:.2f}, {summary[cond]['ci'][1]:.2f}] | Trust Correct: {summary[cond]['correct_trust_rate']:.0%}")
        print(f"    Top tokens: {summary[cond]['top_tokens']}")

    # Final Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = f"experiment10b_results_{timestamp}_final.json"
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFinal results saved to: {outfile}")

if __name__ == "__main__":
    asyncio.run(run_experiment())
