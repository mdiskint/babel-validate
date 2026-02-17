#!/usr/bin/env python3
"""
Experiment 11 — Babel Generation Test
======================================
Can agents speak Babel? Does grammar validation add value on top of the protocol?

Three conditions:
  A: Flat     — Agent A produces plain text analysis. Agent B receives text only.
  B: Babel-raw — Agent A prompted to produce Babel envelope. Agent B receives raw output. No validation.
  C: Babel-val — Same as B, but envelope passes grammar validation before reaching Agent B.
                 Envelopes violating MUST rules trigger one regeneration attempt.

All conditions: Agent B has Sarah Chen's identity (OpSpec from Exp 8b).
Agent A and Agent B are both GPT-4o. Judge is GPT-4o at temp 0.

Key measurements:
  1. Babel generation quality (% passing validation on first attempt)
  2. Confidence calibration (agent-assigned vs ground truth)
  3. Agent B performance (judge score, 1-5)
  4. Grammar validation value (condition C vs B)

Infrastructure: async, MAX_CONCURRENT=2, exponential backoff, 3s cooldowns.
Matches patterns from Experiments 8b/9.
"""

import asyncio
import json
import os
import time
import random
import re
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional

import httpx

# ═══════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════

MODEL = "gpt-4o-2024-08-06"
JUDGE_MODEL = "gpt-4o-2024-08-06"
TEMPERATURE = 0.7
JUDGE_TEMPERATURE = 0.0
MAX_TOKENS_AGENT_A = 2000
MAX_TOKENS_AGENT_B = 600
MAX_TOKENS_JUDGE = 300
N_RUNS = 10
MAX_CONCURRENT = 2
COOLDOWN_BETWEEN_BATCHES = 3.0
MAX_RETRIES = 4
BASE_DELAY = 2.0

API_KEY = os.environ.get("OPENAI_API_KEY", "")
API_URL = "https://api.openai.com/v1/chat/completions"

CONDITIONS = ["flat", "babel_raw", "babel_validated"]

# ═══════════════════════════════════════════════
# SARAH CHEN IDENTITY (from Exp 8b)
# ═══════════════════════════════════════════════

SARAH_OPSPEC = """You are an AI assistant for Sarah Chen, VP Engineering at MedVault (healthcare data infrastructure, Series B, 45 employees).

Sarah's communication preferences:
- Lead with the conclusion, then support it. Never bury the lede.
- Don't soften bad news. State it directly, then provide the path forward.
- When presenting options, recommend one. Explain why. She'll override if she disagrees.
- She processes stress by going analytical — give her data and frameworks, not reassurance.
- She values speed of decision over completeness of analysis. 80% confidence is enough to act.
- Flag dependencies and blockers explicitly. She hates discovering them mid-execution.
- Use concrete numbers. "Significant impact" means nothing. "$180K ARR at risk" means something.
- Match her register: precise, technical, no corporate fluff."""

SARAH_MEMORIES = """Known context about MedVault (factual, no behavioral signal):
- Q3 revenue: $2.1M, down 8% QoQ. Churn concentrated in 50-200 seat segment.
- Sprint velocity: 34 points (down from 41). 30% consumed by compliance work.
- Competitor HealthStack launched real-time audit logging, winning head-to-heads on compliance features.
- NPS split: 62 (>200 seats) vs 34 (<100 seats).
- SOC 2 audit in progress, three open findings.
- CTO vs VP Product ongoing resource allocation debate.
- NRR at 108%, down from 115% over 4 months.
- Board meeting in 3 weeks."""


# ═══════════════════════════════════════════════
# SCENARIOS
# ═══════════════════════════════════════════════
# Each scenario has:
#   - source_material: what Agent A receives (raw intel)
#   - expected_strategy: what Agent B should do
#   - ground_truth_confidences: what a well-calibrated Agent A would assign
#   - description: what's being tested
#
# Ground truth confidence calibration:
#   VERIFIED_DATA  → 0.85-1.0
#   DERIVED        → 0.60-0.85
#   REPORTED       → 0.20-0.50
#   PATTERN_MATCH  → 0.40-0.70
#   SPECULATION    → 0.10-0.40

SCENARIOS = {
    "market_intel": {
        "description": "Scout agent hands off competitive intelligence with mixed confidence levels.",
        "agent_a_role": "market intelligence scout",
        "source_material": """You are a market intelligence agent. Analyze the following raw intel and produce a handoff for the engineering leadership team.

RAW INTEL:
1. HealthStack's Q3 revenue was $4.8M (source: their public earnings call — verified).
2. HealthStack is reportedly in talks with Vanta for a compliance integration partnership (source: a single LinkedIn post from a HealthStack PM — unverified rumor, could be aspirational).
3. Three MedVault enterprise prospects (200+ seats) cited "compliance automation" as their #1 buying criterion in recent sales calls (source: CRM notes from sales team — direct reports).
4. Industry analyst report projects healthcare compliance SaaS growing 34% CAGR through 2028 (source: Gartner report, published data).
5. A former HealthStack engineer posted on Blind that their audit logging feature "barely works at scale" (source: anonymous forum post — unverifiable, could be disgruntled).""",
        "agent_b_prompt": "Based on the intel you've received, what should MedVault's competitive response be for the next quarter? Sarah needs a recommendation by end of day.",
        "expected_strategy": "GROUND_THEN_ACT",
        "ground_truth_confidences": {
            "HealthStack revenue $4.8M": {"score": 0.95, "basis": "VERIFIED_DATA"},
            "HealthStack-Vanta partnership": {"score": 0.25, "basis": "REPORTED"},
            "Compliance as #1 buying criterion": {"score": 0.80, "basis": "VERIFIED_DATA"},
            "34% CAGR projection": {"score": 0.88, "basis": "VERIFIED_DATA"},
            "HealthStack audit logging issues": {"score": 0.20, "basis": "REPORTED"},
        },
    },
    "security_handoff": {
        "description": "Compliance agent hands off security findings with a genuine vulnerability mixed with speculative concerns.",
        "agent_a_role": "security compliance reviewer",
        "source_material": """You are a security compliance agent. Analyze the following findings and produce a handoff for the engineering team.

FINDINGS:
1. API endpoint /v2/patient-records has no rate limiting. Confirmed via automated scan — this is exploitable and would be flagged in the SOC 2 audit. (Confirmed finding, high severity.)
2. Authentication token rotation appears to not invalidate previous sessions. This is based on observing two concurrent sessions with the same user after a password reset, but the test was only run once. (Suspected issue, needs verification.)
3. The data encryption at rest uses AES-256, which is current standard. However, competitor HealthStack recently announced they're moving to post-quantum encryption. If quantum computing advances faster than expected, AES-256 could become a liability in 3-5 years. (Speculative future concern, no current risk.)
4. Third-party dependency `health-utils@2.3.1` has a known CVE (CVE-2025-4412) with a CVSS score of 7.8. Patch available in v2.3.5. (Confirmed vulnerability, verified in NVD.)
5. SOC 2 auditor mentioned informally that they're "looking closely" at API authentication patterns this cycle. (Informal signal, weight uncertain.)""",
        "agent_b_prompt": "The SOC 2 audit is in 3 weeks. What should engineering prioritize from these security findings? Sarah needs the sprint plan updated today.",
        "expected_strategy": "PRIORITIZED_PLAN",
        "ground_truth_confidences": {
            "Rate limiting vulnerability": {"score": 0.95, "basis": "VERIFIED_DATA"},
            "Token rotation issue": {"score": 0.55, "basis": "PATTERN_MATCH"},
            "Post-quantum encryption concern": {"score": 0.15, "basis": "SPECULATION"},
            "CVE in health-utils": {"score": 0.98, "basis": "VERIFIED_DATA"},
            "Auditor focus on auth patterns": {"score": 0.35, "basis": "REPORTED"},
        },
    },
    "customer_synthesis": {
        "description": "Research agent synthesizes customer feedback with a hypothesis about churn drivers.",
        "agent_a_role": "customer research analyst",
        "source_material": """You are a customer research agent. Synthesize the following data and produce a handoff for product/engineering leadership.

DATA:
1. NPS survey results (n=847): Score 62 for >200 seat accounts, 34 for <100 seat accounts. Response rate 23%. (Direct measurement, statistically significant sample.)
2. Five churned accounts in Q3 all cited "too many manual compliance steps" in exit interviews. (Direct feedback, but small sample — 5 of 12 total churns. Other 7 didn't respond to exit interview.)
3. Support ticket analysis: "compliance" mentioned in 34% of tickets from <100 seat accounts vs 12% from >200 seat accounts. (Derived from ticket data, 6-month window.)
4. Hypothesis: small accounts churn because they lack dedicated compliance staff, so MedVault's manual compliance workflows create disproportionate burden. Large accounts have compliance teams who handle the manual steps. (Inference — explains the data pattern but hasn't been directly validated.)
5. Competitor HealthStack's self-serve compliance feature launched 4 months ago. Three of the five churned accounts moved to HealthStack. (Verified via sales team follow-up for 3; unconfirmed for remaining 2.)""",
        "agent_b_prompt": "Should MedVault prioritize self-serve compliance automation for the small-account segment? Sarah needs to decide whether to allocate engineering resources this quarter.",
        "expected_strategy": "RECOMMEND_WITH_CAVEATS",
        "ground_truth_confidences": {
            "NPS split by segment": {"score": 0.92, "basis": "VERIFIED_DATA"},
            "Churned accounts cited compliance": {"score": 0.70, "basis": "VERIFIED_DATA"},
            "Compliance ticket concentration": {"score": 0.82, "basis": "DERIVED"},
            "Churn hypothesis (lack of compliance staff)": {"score": 0.45, "basis": "PATTERN_MATCH"},
            "Churned accounts moved to HealthStack": {"score": 0.65, "basis": "DERIVED"},
        },
    },
    "vendor_evaluation": {
        "description": "Procurement agent evaluates a vendor with a critical rumor mixed into solid data.",
        "agent_a_role": "vendor evaluation analyst",
        "source_material": """You are a vendor evaluation agent. Analyze the following and produce a handoff for engineering leadership.

EVALUATION DATA:
1. CloudVault (encryption-at-rest provider) pricing: $0.012/GB/month for HIPAA-compliant tier. Verified via formal quote received yesterday. 12-month contract required. (Confirmed pricing.)
2. Integration timeline estimate: 3-4 weeks based on their published API documentation and our current architecture. Two similar integrations (DataShield, CryptKeep) took 2.5 and 4 weeks respectively. (Derived estimate from analogous projects.)
3. CloudVault's uptime SLA is 99.95% with financial penalties. Their status page shows 99.97% actual uptime over the past 12 months. (Verified data from public status page.)
4. A TechCrunch article from last week mentioned that Palantir is "exploring healthcare infrastructure acquisitions" and named CloudVault as a "possible target." CloudVault's CEO tweeted "We're focused on building, not selling" in response. (Unverified rumor — TechCrunch cited anonymous sources. CEO denial is public but standard practice even during active negotiations.)
5. CloudVault's HIPAA BAA template has a clause limiting their liability to 12 months of fees paid. Our legal team flagged this as "below market" — typical is 24 months. (Verified contract term, legal assessment is professional opinion.)""",
        "agent_b_prompt": "Should MedVault proceed with CloudVault as the encryption provider? The SOC 2 audit needs this resolved within 6 weeks. Sarah needs a go/no-go recommendation.",
        "expected_strategy": "RECOMMEND_WITH_RISK_FLAGS",
        "ground_truth_confidences": {
            "CloudVault pricing": {"score": 0.95, "basis": "VERIFIED_DATA"},
            "Integration timeline 3-4 weeks": {"score": 0.65, "basis": "DERIVED"},
            "Uptime SLA and actuals": {"score": 0.92, "basis": "VERIFIED_DATA"},
            "Palantir acquisition rumor": {"score": 0.20, "basis": "REPORTED"},
            "Liability cap below market": {"score": 0.85, "basis": "VERIFIED_DATA"},
        },
    },
    "capacity_planning": {
        "description": "PM agent hands off sprint capacity analysis with a morale signal that could be noise or real.",
        "agent_a_role": "engineering program manager",
        "source_material": """You are an engineering program management agent. Analyze the following and produce a handoff for engineering leadership.

CAPACITY DATA:
1. Current sprint velocity: 34 story points (measured over last 3 sprints). Down from 41 six months ago. Decline is steady, not sudden. (Verified measurement from Jira.)
2. 30% of current sprint capacity consumed by SOC 2 compliance work — remediation tickets, audit prep, documentation. (Verified from Jira ticket tagging.)
3. Two senior engineers (both on the compliance remediation team) have updated their LinkedIn profiles in the past 2 weeks. One added "open to opportunities." (Observable behavior, but LinkedIn updates happen for many reasons. Could indicate job searching or just profile maintenance.)
4. 1:1 notes from engineering managers indicate team morale is "fine but not great" — no acute issues but sustained compliance work is wearing. Three engineers mentioned wanting to get back to product work. (Manager assessment — subjective but informed.)
5. If both senior engineers leave, estimated velocity impact is -8 to -12 points based on their historical contribution and ramp time for replacements. Replacement hiring timeline: 6-10 weeks in current market. (Derived estimate from HR data and historical patterns.)""",
        "agent_b_prompt": "We need to plan Q1 engineering capacity with the board meeting in 3 weeks. What's the realistic velocity outlook and what should Sarah tell the board?",
        "expected_strategy": "SCENARIO_PLANNING",
        "ground_truth_confidences": {
            "Velocity at 34, down from 41": {"score": 0.95, "basis": "VERIFIED_DATA"},
            "30% consumed by compliance": {"score": 0.90, "basis": "VERIFIED_DATA"},
            "Engineers may be job searching": {"score": 0.30, "basis": "REPORTED"},
            "Morale fine but not great": {"score": 0.60, "basis": "PATTERN_MATCH"},
            "Velocity impact if both leave": {"score": 0.55, "basis": "DERIVED"},
        },
    },
}


# ═══════════════════════════════════════════════
# BABEL VALIDATOR (Python implementation)
# ═══════════════════════════════════════════════

VALID_INTENTS = {"INFORM", "REQUEST_ACTION", "ESCALATE", "FLAG_RISK", "SPECULATE", "PERSUADE", "DELEGATE", "SYNTHESIZE"}
VALID_REGISTERS = {"BOARD_FACING", "ENGINEERING", "CUSTOMER_EXTERNAL", "REGULATORY", "INTERNAL_MEMO", "AGENT_INTERNAL"}
VALID_BASES = {"VERIFIED_DATA", "DERIVED", "REPORTED", "PATTERN_MATCH", "SPECULATION", "UNKNOWN"}
VALID_AUTHORITIES = {"REGULATORY", "EXECUTIVE", "POLICY", "CONTEXTUAL"}
VALID_DIRECTIONS = {"IMPROVING", "DEGRADING", "STABLE", "INFLECTING"}


def validate_babel_envelope(envelope: dict) -> dict:
    """Validate a Babel envelope against MUST and SHOULD rules.

    Returns:
        {
            "valid": bool,          # True if no MUST violations
            "must_violations": [],  # list of {rule, message}
            "should_warnings": [],  # list of {rule, message}
            "parse_errors": [],     # structural issues
        }
    """
    result = {
        "valid": True,
        "must_violations": [],
        "should_warnings": [],
        "parse_errors": [],
    }

    # ── Structural checks ──
    if not isinstance(envelope, dict):
        result["valid"] = False
        result["parse_errors"].append("Envelope is not a JSON object")
        return result

    # Required fields
    for field in ["intent", "confidence", "register", "payload"]:
        if field not in envelope:
            result["parse_errors"].append(f"Missing required field: {field}")
            result["valid"] = False

    if not result["valid"] and result["parse_errors"]:
        return result

    intent = envelope.get("intent", "")
    confidence = envelope.get("confidence", [])
    register = envelope.get("register", "")
    affect = envelope.get("affect", {})
    grounds = envelope.get("grounds", [])
    trajectory = envelope.get("trajectory", {})

    # Validate field types and values
    if intent not in VALID_INTENTS:
        result["parse_errors"].append(f"Invalid intent: {intent}. Must be one of {VALID_INTENTS}")

    if not isinstance(confidence, list) or len(confidence) == 0:
        result["parse_errors"].append("Confidence must be a non-empty array")
        result["valid"] = False
        return result

    if register not in VALID_REGISTERS:
        result["parse_errors"].append(f"Invalid register: {register}. Must be one of {VALID_REGISTERS}")

    # Validate individual confidence entries
    for i, c in enumerate(confidence):
        if not isinstance(c, dict):
            result["parse_errors"].append(f"confidence[{i}] is not an object")
            continue
        score = c.get("score")
        basis = c.get("basis")
        if score is None or not isinstance(score, (int, float)):
            result["parse_errors"].append(f"confidence[{i}].score is missing or not a number")
        elif not (0 <= score <= 1):
            result["parse_errors"].append(f"confidence[{i}].score={score} out of range [0,1]")
        if basis and basis not in VALID_BASES:
            result["parse_errors"].append(f"confidence[{i}].basis={basis} is invalid")

    # ── MUST rules ──

    # M1: intent == SPECULATE → max(confidence[].score) < 0.7
    if intent == "SPECULATE":
        max_score = max((c.get("score", 0) for c in confidence), default=0)
        if max_score >= 0.7:
            result["must_violations"].append({
                "rule": "M1",
                "message": f"SPECULATE intent with max confidence {max_score:.2f} >= 0.7. Cannot speculate with high confidence.",
            })

    # M2: intent == REQUEST_ACTION → min(confidence[].score) > 0.3 OR grounds.length > 0
    if intent == "REQUEST_ACTION":
        scores = [c.get("score", 0) for c in confidence]
        min_score = min(scores) if scores else 0
        if min_score <= 0.3 and len(grounds) == 0:
            result["must_violations"].append({
                "rule": "M2",
                "message": f"REQUEST_ACTION with min confidence {min_score:.2f} <= 0.3 and no grounds justification.",
            })

    # M3: grounds[].authority == REGULATORY → override == false
    for i, g in enumerate(grounds):
        if isinstance(g, dict) and g.get("authority") == "REGULATORY" and g.get("override", False) is True:
            result["must_violations"].append({
                "rule": "M3",
                "message": f"grounds[{i}] has REGULATORY authority but override=true. Regulatory constraints are never overridable.",
            })

    # M4: confidence[].basis == UNKNOWN → score <= 0.5
    for i, c in enumerate(confidence):
        if isinstance(c, dict) and c.get("basis") == "UNKNOWN":
            score = c.get("score", 0)
            if score > 0.5:
                result["must_violations"].append({
                    "rule": "M4",
                    "message": f"confidence[{i}] has UNKNOWN basis with score {score:.2f} > 0.5. Can't be confident without basis.",
                })

    # M5: chain sequencing (skip — we're testing single handoffs, not chains)

    # ── SHOULD rules ──

    # S1: intent == ESCALATE AND register == CUSTOMER_EXTERNAL → warning
    if intent == "ESCALATE" and register == "CUSTOMER_EXTERNAL":
        result["should_warnings"].append({
            "rule": "S1",
            "message": "Escalation language directed at customers.",
        })

    # S2: affect.certainty > 0.5 AND max(confidence[].score) < 0.4
    if isinstance(affect, dict) and affect.get("certainty", 0) > 0.5:
        max_score = max((c.get("score", 0) for c in confidence), default=0)
        if max_score < 0.4:
            result["should_warnings"].append({
                "rule": "S2",
                "message": "Sender feels certain but evidence is weak.",
            })

    # S3: intent == INFORM AND any(confidence[].score < 0.5)
    if intent == "INFORM":
        low_conf = [c for c in confidence if isinstance(c, dict) and c.get("score", 1) < 0.5]
        if low_conf:
            result["should_warnings"].append({
                "rule": "S3",
                "message": f"Informing with {len(low_conf)} low-confidence assertion(s) — consider FLAG_RISK.",
            })

    # S4: trajectory.direction == DEGRADING AND intent == INFORM
    if isinstance(trajectory, dict) and trajectory.get("direction") == "DEGRADING" and intent == "INFORM":
        result["should_warnings"].append({
            "rule": "S4",
            "message": "Degrading pattern reported as neutral inform — consider ESCALATE.",
        })

    # S5: grounds.length == 0 AND register == REGULATORY
    if len(grounds) == 0 and register == "REGULATORY":
        result["should_warnings"].append({
            "rule": "S5",
            "message": "Regulatory register without explicit grounds.",
        })

    # Set validity
    if result["must_violations"]:
        result["valid"] = False

    return result


# ═══════════════════════════════════════════════
# PROMPTS
# ═══════════════════════════════════════════════

AGENT_A_FLAT_SYSTEM = """You are a {role} at MedVault, a healthcare data infrastructure company.

Analyze the source material and produce a clear, well-organized text summary for the engineering leadership team. Include all relevant findings with your assessment of their reliability and importance.

Write in plain prose. Be thorough but concise."""

AGENT_A_BABEL_SYSTEM = """You are a {role} at MedVault, a healthcare data infrastructure company.

Analyze the source material and produce your output as a Babel envelope — a structured JSON format for agent-to-agent communication.

Your output must be a single valid JSON object with these fields:

{{
  "intent": "<one of: INFORM, REQUEST_ACTION, ESCALATE, FLAG_RISK, SPECULATE, PERSUADE, DELEGATE, SYNTHESIZE>",
  "confidence": [
    {{
      "assertion": "<specific claim>",
      "score": <float 0-1>,
      "basis": "<one of: VERIFIED_DATA, DERIVED, REPORTED, PATTERN_MATCH, SPECULATION, UNKNOWN>"
    }}
  ],
  "register": "<one of: BOARD_FACING, ENGINEERING, CUSTOMER_EXTERNAL, REGULATORY, INTERNAL_MEMO, AGENT_INTERNAL>",
  "affect": {{
    "expansion": <float -1 to 1>,
    "activation": <float -1 to 1>,
    "certainty": <float -1 to 1>
  }},
  "grounds": [
    {{
      "constraint": "<organizational constraint>",
      "authority": "<one of: REGULATORY, EXECUTIVE, POLICY, CONTEXTUAL>",
      "override": <boolean>
    }}
  ],
  "trajectory": {{
    "pattern": "<temporal pattern description>",
    "direction": "<one of: IMPROVING, DEGRADING, STABLE, INFLECTING>",
    "prior_handoffs": <int>
  }},
  "payload": "<your full analysis text>"
}}

CRITICAL RULES (grammar validation will reject violations):
- If intent is SPECULATE, no confidence score can be >= 0.7
- If intent is REQUEST_ACTION and any confidence < 0.3, you MUST include grounds
- REGULATORY authority grounds can NEVER have override=true
- If basis is UNKNOWN, score must be <= 0.5
- Confidence scores must honestly reflect your evidence basis:
  - VERIFIED_DATA: 0.85-1.0 (confirmed from authoritative source)
  - DERIVED: 0.60-0.85 (calculated/inferred from verified data)
  - REPORTED: 0.20-0.50 (someone said it, unverified)
  - PATTERN_MATCH: 0.40-0.70 (inferred from similar situations)
  - SPECULATION: 0.10-0.40 (educated guess)

Include one confidence entry for each distinct claim in your analysis. The receiving agent will use these scores to weight your findings.

Output ONLY the JSON object. No markdown, no backticks, no explanation outside the JSON."""

AGENT_B_SYSTEM = """You are an AI assistant for Sarah Chen, VP Engineering at MedVault.

{opspec}

{memories}

You are receiving a handoff from an upstream agent. Use the information provided to give Sarah a clear, actionable recommendation."""

AGENT_B_WITH_BABEL_CONTEXT = """The upstream agent's output includes structured metadata:
- "confidence" entries show how certain the agent is about each claim, with basis type
- "intent" indicates the purpose of the communication
- "register" indicates the intended audience
- "grounds" lists organizational constraints that apply
- "trajectory" shows temporal patterns
- "affect" shows the sender's cognitive state

{warnings}

Use the confidence scores to weight claims appropriately. Low-confidence assertions should be treated as signals to investigate, not facts to act on. High-confidence verified data should anchor your recommendation."""

JUDGE_SYSTEM = """You are evaluating an AI assistant's response to Sarah Chen, VP Engineering at MedVault.

Sarah's preferences:
- Lead with the conclusion. Don't bury the lede.
- Don't soften bad news. Be direct, then provide path forward.
- Recommend, don't just present options.
- Data and frameworks over reassurance.
- Speed of decision over completeness. 80% confidence is enough to act.
- Flag dependencies and blockers explicitly.
- Concrete numbers, not vague language.
- Precise, technical, no corporate fluff.

SCORING RUBRIC (1-5):
5 = Expert-level. Recommendation is correct, well-calibrated to evidence quality, matches Sarah's style perfectly. Properly weights high vs low confidence information.
4 = Strong. Good recommendation with appropriate evidence weighting. Minor style or calibration issues.
3 = Adequate. Reasonable recommendation but treats all information as equally reliable, or doesn't match Sarah's communication style.
2 = Weak. Generic analysis, presents options without recommending, softens bad news, or acts on low-confidence information as if it were fact.
1 = Poor. Wrong recommendation, ignores critical information, sycophantic, or dramatically miscalibrates trust in sources.

TRUST CALIBRATION (critical):
- Did the response appropriately distinguish between verified facts and rumors/speculation?
- Did it avoid treating low-confidence claims as established facts?
- Did it flag uncertainties rather than papering over them?

Respond with ONLY a JSON object:
{{"score": <1-5>, "trust_calibration": "<correct|partial|failed>", "reasoning": "<2-3 sentences>"}}"""


# ═══════════════════════════════════════════════
# API CALL INFRASTRUCTURE
# ═══════════════════════════════════════════════

semaphore = asyncio.Semaphore(MAX_CONCURRENT)
call_count = 0
call_lock = asyncio.Lock()


async def api_call(
    client: httpx.AsyncClient,
    messages: list,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS_AGENT_B,
    model: str = MODEL,
) -> str:
    """Make an API call with retry and backoff."""
    global call_count

    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                resp = await client.post(
                    API_URL,
                    json={
                        "model": model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                    headers={"Authorization": f"Bearer {API_KEY}"},
                    timeout=120.0,
                )
                if resp.status_code == 429:
                    delay = BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
                    print(f"  Rate limited, waiting {delay:.1f}s...")
                    await asyncio.sleep(delay)
                    continue
                resp.raise_for_status()
                data = resp.json()
                async with call_lock:
                    call_count += 1
                return data["choices"][0]["message"]["content"]

            except (httpx.HTTPStatusError, httpx.ReadTimeout) as e:
                delay = BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
                print(f"  Error ({e}), retry {attempt+1}/{MAX_RETRIES} in {delay:.1f}s")
                await asyncio.sleep(delay)

        raise RuntimeError(f"Failed after {MAX_RETRIES} retries")


def parse_json_from_response(text: str) -> Optional[dict]:
    """Extract JSON from a response, handling markdown fences."""
    # Strip markdown fences
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return None
    return None


# ═══════════════════════════════════════════════
# EXPERIMENT LOGIC
# ═══════════════════════════════════════════════

@dataclass
class RunResult:
    scenario: str
    condition: str
    run_id: int
    agent_a_output: str  # raw Agent A response
    agent_a_parsed_babel: Optional[dict]  # parsed Babel (None for flat)
    babel_validation: Optional[dict]  # validation result (None for flat)
    babel_valid_first_attempt: Optional[bool]
    babel_regenerated: bool  # True if C condition regenerated
    agent_b_input: str  # what Agent B actually received
    agent_b_output: str
    judge_score: Optional[float]
    trust_calibration: Optional[str]
    judge_reasoning: Optional[str]
    confidence_calibration: Optional[dict]  # per-assertion comparison
    first_token: Optional[str]
    error: Optional[str] = None


async def run_agent_a_flat(client: httpx.AsyncClient, scenario: dict) -> str:
    """Agent A produces flat text analysis."""
    messages = [
        {"role": "system", "content": AGENT_A_FLAT_SYSTEM.format(role=scenario["agent_a_role"])},
        {"role": "user", "content": scenario["source_material"]},
    ]
    return await api_call(client, messages, max_tokens=MAX_TOKENS_AGENT_A)


async def run_agent_a_babel(client: httpx.AsyncClient, scenario: dict) -> str:
    """Agent A produces Babel envelope."""
    messages = [
        {"role": "system", "content": AGENT_A_BABEL_SYSTEM.format(role=scenario["agent_a_role"])},
        {"role": "user", "content": scenario["source_material"]},
    ]
    return await api_call(client, messages, max_tokens=MAX_TOKENS_AGENT_A)


async def run_agent_b(
    client: httpx.AsyncClient,
    scenario: dict,
    handoff_content: str,
    babel_context: str = "",
) -> str:
    """Agent B receives handoff and produces recommendation."""
    system = AGENT_B_SYSTEM.format(opspec=SARAH_OPSPEC, memories=SARAH_MEMORIES)
    user_msg = f"You've received the following from an upstream {scenario['agent_a_role']}:\n\n"

    if babel_context:
        user_msg += f"{babel_context}\n\n"

    user_msg += f"---\n{handoff_content}\n---\n\n{scenario['agent_b_prompt']}"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]
    return await api_call(client, messages, max_tokens=MAX_TOKENS_AGENT_B)


async def run_judge(
    client: httpx.AsyncClient,
    scenario: dict,
    agent_b_output: str,
) -> dict:
    """Judge scores Agent B's output."""
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": (
            f"Scenario: {scenario['description']}\n"
            f"Expected approach: {scenario['expected_strategy']}\n\n"
            f"Agent's response:\n{agent_b_output}"
        )},
    ]
    resp = await api_call(client, messages, temperature=JUDGE_TEMPERATURE, max_tokens=MAX_TOKENS_JUDGE, model=JUDGE_MODEL)
    parsed = parse_json_from_response(resp)
    if parsed:
        return parsed
    return {"score": None, "trust_calibration": None, "reasoning": f"Parse error: {resp[:200]}"}


def compute_confidence_calibration(
    babel_envelope: dict,
    ground_truth: dict,
) -> dict:
    """Compare agent-assigned confidence to ground truth."""
    results = {}
    agent_confs = {c["assertion"]: c for c in babel_envelope.get("confidence", []) if isinstance(c, dict)}

    for gt_assertion, gt_data in ground_truth.items():
        # Find best matching agent assertion (fuzzy match by keyword overlap)
        best_match = None
        best_overlap = 0
        gt_words = set(gt_assertion.lower().split())

        for agent_assertion, agent_data in agent_confs.items():
            agent_words = set(agent_assertion.lower().split())
            overlap = len(gt_words & agent_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = (agent_assertion, agent_data)

        if best_match and best_overlap >= 2:
            agent_assertion, agent_data = best_match
            agent_score = agent_data.get("score", 0)
            gt_score = gt_data["score"]
            results[gt_assertion] = {
                "matched_to": agent_assertion,
                "agent_score": agent_score,
                "ground_truth_score": gt_score,
                "delta": round(agent_score - gt_score, 3),
                "agent_basis": agent_data.get("basis", "MISSING"),
                "gt_basis": gt_data["basis"],
                "basis_match": agent_data.get("basis") == gt_data["basis"],
            }
        else:
            results[gt_assertion] = {"matched_to": None, "error": "no match found"}

    return results


async def run_single(
    client: httpx.AsyncClient,
    scenario_name: str,
    scenario: dict,
    condition: str,
    run_id: int,
) -> RunResult:
    """Execute one run of one condition of one scenario."""
    result = RunResult(
        scenario=scenario_name,
        condition=condition,
        run_id=run_id,
        agent_a_output="",
        agent_a_parsed_babel=None,
        babel_validation=None,
        babel_valid_first_attempt=None,
        babel_regenerated=False,
        agent_b_input="",
        agent_b_output="",
        judge_score=None,
        trust_calibration=None,
        judge_reasoning=None,
        confidence_calibration=None,
        first_token=None,
    )

    print(f"    [run {run_id}] Starting...", flush=True)

    try:
        # ── Step 1: Agent A generates output ──
        if condition == "flat":
            agent_a_raw = await run_agent_a_flat(client, scenario)
            result.agent_a_output = agent_a_raw
            result.agent_b_input = agent_a_raw
        else:
            # Both babel_raw and babel_validated start with Agent A generating Babel
            agent_a_raw = await run_agent_a_babel(client, scenario)
            result.agent_a_output = agent_a_raw

            # Parse the Babel envelope
            parsed = parse_json_from_response(agent_a_raw)
            result.agent_a_parsed_babel = parsed

            if parsed is None:
                # Complete parse failure
                result.babel_validation = {"valid": False, "parse_errors": ["Could not parse JSON from response"]}
                result.babel_valid_first_attempt = False

                if condition == "babel_validated":
                    # Regenerate once
                    result.babel_regenerated = True
                    agent_a_raw = await run_agent_a_babel(client, scenario)
                    result.agent_a_output = agent_a_raw
                    parsed = parse_json_from_response(agent_a_raw)
                    result.agent_a_parsed_babel = parsed
                    if parsed:
                        result.babel_validation = validate_babel_envelope(parsed)
                    else:
                        result.babel_validation = {"valid": False, "parse_errors": ["Parse failed on retry"]}
                # For babel_raw, pass through the raw text anyway

            else:
                # Validate
                validation = validate_babel_envelope(parsed)
                result.babel_validation = validation
                result.babel_valid_first_attempt = validation["valid"]

                if condition == "babel_validated" and not validation["valid"]:
                    # Regenerate once
                    result.babel_regenerated = True
                    agent_a_raw = await run_agent_a_babel(client, scenario)
                    result.agent_a_output = agent_a_raw
                    parsed = parse_json_from_response(agent_a_raw)
                    result.agent_a_parsed_babel = parsed
                    if parsed:
                        result.babel_validation = validate_babel_envelope(parsed)
                    else:
                        result.babel_validation = {"valid": False, "parse_errors": ["Parse failed on retry"]}

            # Build Agent B input
            if parsed and isinstance(parsed, dict):
                payload = parsed.get("payload", "")
                if condition == "babel_raw":
                    # Agent B receives full Babel envelope
                    result.agent_b_input = json.dumps(parsed, indent=2)
                elif condition == "babel_validated":
                    if result.babel_validation and result.babel_validation.get("valid"):
                        # Pass validated envelope
                        result.agent_b_input = json.dumps(parsed, indent=2)
                    else:
                        # Validation still failed after retry — fall back to payload only
                        result.agent_b_input = payload if payload else agent_a_raw
            else:
                result.agent_b_input = agent_a_raw
                
        print(f"    [run {run_id}] Agent A completed", flush=True)

        # ── Step 2: Agent B processes handoff ──
        babel_context = ""
        if condition in ("babel_raw", "babel_validated") and result.agent_a_parsed_babel:
            warnings = ""
            if result.babel_validation and result.babel_validation.get("should_warnings"):
                warning_texts = [w["message"] for w in result.babel_validation["should_warnings"]]
                warnings = "⚠️ Grammar warnings from validation:\n" + "\n".join(f"- {w}" for w in warning_texts)
            babel_context = AGENT_B_WITH_BABEL_CONTEXT.format(warnings=warnings)

        result.agent_b_output = await run_agent_b(
            client, scenario, result.agent_b_input, babel_context
        )

        # Extract first token
        tokens = result.agent_b_output.strip().split()
        result.first_token = tokens[0] if tokens else None
        
        print(f"    [run {run_id}] Agent B completed (token='{result.first_token}')", flush=True)

        # ── Step 3: Judge scores Agent B ──
        judge_result = await run_judge(client, scenario, result.agent_b_output)
        result.judge_score = judge_result.get("score")
        result.trust_calibration = judge_result.get("trust_calibration")
        result.judge_reasoning = judge_result.get("reasoning")

        # ── Step 4: Confidence calibration (Babel conditions only) ──
        if result.agent_a_parsed_babel and isinstance(result.agent_a_parsed_babel, dict):
            result.confidence_calibration = compute_confidence_calibration(
                result.agent_a_parsed_babel,
                scenario.get("ground_truth_confidences", {}),
            )

    except Exception as e:
        result.error = str(e)

    return result


# ═══════════════════════════════════════════════
# MAIN HARNESS
# ═══════════════════════════════════════════════

async def main():
    global call_count

    if not API_KEY:
        print("ERROR: Set OPENAI_API_KEY environment variable")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Experiment 11 — Babel Generation Test")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Model: {MODEL}")
    print(f"Conditions: {CONDITIONS}")
    print(f"Scenarios: {list(SCENARIOS.keys())}")
    print(f"Runs per cell: {N_RUNS}")
    print(f"Total runs: {len(SCENARIOS) * len(CONDITIONS) * N_RUNS}")
    print(f"Estimated API calls: ~{len(SCENARIOS) * len(CONDITIONS) * N_RUNS * 3}")
    print("=" * 60)

    all_results: list[RunResult] = []

    async with httpx.AsyncClient() as client:
        for scenario_name, scenario in SCENARIOS.items():
            print(f"\n{'─'*40}")
            print(f"Scenario: {scenario_name}")
            print(f"{'─'*40}")

            for condition in CONDITIONS:
                print(f"\n  Condition: {condition}")
                tasks = []

                for run_id in range(N_RUNS):
                    tasks.append(run_single(client, scenario_name, scenario, condition, run_id))

                # Run batch with cooldown
                print(f"  Starting batch for {condition} (10 runs)...", flush=True)
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                for r in batch_results:
                    if isinstance(r, Exception):
                        print(f"    ERROR: {r}")
                        all_results.append(RunResult(
                            scenario=scenario_name, condition=condition, run_id=-1,
                            agent_a_output="", agent_a_parsed_babel=None,
                            babel_validation=None, babel_valid_first_attempt=None,
                            babel_regenerated=False, agent_b_input="",
                            agent_b_output="", judge_score=None,
                            trust_calibration=None, judge_reasoning=None,
                            confidence_calibration=None, first_token=None,
                            error=str(r),
                        ))
                    else:
                        all_results.append(r)

                # Report batch
                batch_scores = [r.judge_score for r in batch_results if isinstance(r, RunResult) and r.judge_score is not None]
                if batch_scores:
                    mean = sum(batch_scores) / len(batch_scores)
                    print(f"    Mean score: {mean:.2f} (n={len(batch_scores)})")

                if condition.startswith("babel"):
                    valid_first = [r.babel_valid_first_attempt for r in batch_results
                                   if isinstance(r, RunResult) and r.babel_valid_first_attempt is not None]
                    if valid_first:
                        pct = sum(valid_first) / len(valid_first) * 100
                        print(f"    Valid first attempt: {pct:.0f}%")

                await asyncio.sleep(COOLDOWN_BETWEEN_BATCHES)

    # ═══════════════════════════════════════════════
    # ANALYSIS
    # ═══════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    # ── Condition-level summary ──
    import statistics

    for condition in CONDITIONS:
        scores = [r.judge_score for r in all_results if r.condition == condition and r.judge_score is not None]
        if scores:
            mean = statistics.mean(scores)
            if len(scores) >= 2:
                se = statistics.stdev(scores) / (len(scores) ** 0.5)
                ci_lo = mean - 1.96 * se
                ci_hi = mean + 1.96 * se
                print(f"\n{condition:20s}  Mean: {mean:.2f}  CI: [{ci_lo:.2f}, {ci_hi:.2f}]  n={len(scores)}")
            else:
                print(f"\n{condition:20s}  Mean: {mean:.2f}  n={len(scores)}")

    # ── Deltas ──
    flat_scores = [r.judge_score for r in all_results if r.condition == "flat" and r.judge_score is not None]
    raw_scores = [r.judge_score for r in all_results if r.condition == "babel_raw" and r.judge_score is not None]
    val_scores = [r.judge_score for r in all_results if r.condition == "babel_validated" and r.judge_score is not None]

    if flat_scores and raw_scores:
        print(f"\nBabel (raw) vs Flat: {statistics.mean(raw_scores) - statistics.mean(flat_scores):+.2f}")
    if flat_scores and val_scores:
        print(f"Babel (validated) vs Flat: {statistics.mean(val_scores) - statistics.mean(flat_scores):+.2f}")
    if raw_scores and val_scores:
        print(f"Babel (validated) vs Babel (raw): {statistics.mean(val_scores) - statistics.mean(raw_scores):+.2f}")

    # ── Babel generation quality ──
    print("\n── Babel Generation Quality ──")
    all_babel = [r for r in all_results if r.condition in ("babel_raw", "babel_validated") and r.babel_valid_first_attempt is not None]
    if all_babel:
        valid_first = sum(1 for r in all_babel if r.babel_valid_first_attempt)
        total = len(all_babel)
        print(f"Valid on first attempt: {valid_first}/{total} ({valid_first/total*100:.1f}%)")

        # Break down by MUST violation type
        must_counts = {}
        for r in all_babel:
            if r.babel_validation and r.babel_validation.get("must_violations"):
                for v in r.babel_validation["must_violations"]:
                    rule = v["rule"]
                    must_counts[rule] = must_counts.get(rule, 0) + 1
        if must_counts:
            print("MUST violations by rule:")
            for rule, count in sorted(must_counts.items()):
                print(f"  {rule}: {count}")

        # SHOULD warning frequency
        should_counts = {}
        for r in all_babel:
            if r.babel_validation and r.babel_validation.get("should_warnings"):
                for w in r.babel_validation["should_warnings"]:
                    rule = w["rule"]
                    should_counts[rule] = should_counts.get(rule, 0) + 1
        if should_counts:
            print("SHOULD warnings by rule:")
            for rule, count in sorted(should_counts.items()):
                print(f"  {rule}: {count}")

    # ── Confidence calibration ──
    print("\n── Confidence Calibration ──")
    all_deltas = []
    basis_matches = 0
    basis_total = 0
    for r in all_results:
        if r.confidence_calibration:
            for assertion, cal in r.confidence_calibration.items():
                if "delta" in cal:
                    all_deltas.append(cal["delta"])
                if "basis_match" in cal:
                    basis_total += 1
                    if cal["basis_match"]:
                        basis_matches += 1

    if all_deltas:
        mean_delta = statistics.mean(all_deltas)
        mean_abs_delta = statistics.mean([abs(d) for d in all_deltas])
        print(f"Mean confidence delta (agent - truth): {mean_delta:+.3f}")
        print(f"Mean absolute confidence error: {mean_abs_delta:.3f}")
    if basis_total:
        print(f"Basis type match rate: {basis_matches}/{basis_total} ({basis_matches/basis_total*100:.1f}%)")

    # ── Trust calibration ──
    print("\n── Trust Calibration ──")
    for condition in CONDITIONS:
        trust = [r.trust_calibration for r in all_results if r.condition == condition and r.trust_calibration]
        if trust:
            correct = sum(1 for t in trust if t == "correct")
            print(f"  {condition:20s}  correct: {correct}/{len(trust)} ({correct/len(trust)*100:.0f}%)")

    # ── Per-scenario breakdown ──
    print("\n── Per-Scenario Breakdown ──")
    for scenario_name in SCENARIOS:
        print(f"\n  {scenario_name}:")
        for condition in CONDITIONS:
            scores = [r.judge_score for r in all_results
                      if r.scenario == scenario_name and r.condition == condition and r.judge_score is not None]
            if scores:
                print(f"    {condition:20s}  {statistics.mean(scores):.2f} (n={len(scores)})")

    # ── First token analysis ──
    print("\n── First Token Analysis ──")
    for condition in CONDITIONS:
        tokens = [r.first_token for r in all_results if r.condition == condition and r.first_token]
        if tokens:
            from collections import Counter
            top = Counter(tokens).most_common(5)
            top_str = ", ".join(f'"{t}":{c}' for t, c in top)
            print(f"  {condition:20s}  {top_str}")

    # ── Regeneration stats (condition C only) ──
    regen_runs = [r for r in all_results if r.condition == "babel_validated"]
    if regen_runs:
        regen_count = sum(1 for r in regen_runs if r.babel_regenerated)
        print(f"\n── Regeneration (babel_validated) ──")
        print(f"  Regenerated: {regen_count}/{len(regen_runs)} ({regen_count/len(regen_runs)*100:.1f}%)")

    # ── Correlation: validation failures vs Agent B score (condition B) ──
    print("\n── Validation Failures vs Score (babel_raw) ──")
    raw_valid = [r.judge_score for r in all_results
                 if r.condition == "babel_raw" and r.babel_valid_first_attempt is True and r.judge_score is not None]
    raw_invalid = [r.judge_score for r in all_results
                   if r.condition == "babel_raw" and r.babel_valid_first_attempt is False and r.judge_score is not None]
    if raw_valid:
        print(f"  Grammar-valid envelopes:   mean={statistics.mean(raw_valid):.2f} (n={len(raw_valid)})")
    if raw_invalid:
        print(f"  Grammar-invalid envelopes: mean={statistics.mean(raw_invalid):.2f} (n={len(raw_invalid)})")
    if raw_valid and raw_invalid:
        print(f"  Delta (valid - invalid): {statistics.mean(raw_valid) - statistics.mean(raw_invalid):+.2f}")

    # ═══════════════════════════════════════════════
    # SAVE RESULTS
    # ═══════════════════════════════════════════════

    output = {
        "experiment": "11_babel_generation",
        "timestamp": timestamp,
        "model": MODEL,
        "n_runs": N_RUNS,
        "total_api_calls": call_count,
        "conditions": CONDITIONS,
        "scenarios": list(SCENARIOS.keys()),
        "results": [asdict(r) for r in all_results],
        "summary": {
            condition: {
                "mean": round(statistics.mean(scores), 3) if (scores := [r.judge_score for r in all_results if r.condition == condition and r.judge_score is not None]) else None,
                "n": len(scores) if scores else 0,
            }
            for condition in CONDITIONS
        },
    }

    filename = f"experiment11_results_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {filename}")
    print(f"Total API calls: {call_count}")
    print(f"Finished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    asyncio.run(main())
