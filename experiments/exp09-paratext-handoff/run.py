"""
Experiment 9: Paratext Agent-to-Agent Handoff
==============================================

Tests whether Paratext envelopes measurably improve downstream agent
performance in a handoff chain.

Three conditions:
  A) Flat handoff: Agent B receives only Agent A's text output
  B) Envelope handoff: Agent B receives text + Paratext envelope
     (confidence, register, intent, flagged uncertainties)
  C) Noisy envelope: Agent B receives text + deliberately miscalibrated
     envelope (wrong confidence, wrong register, misleading intent)

Agent A outputs are DETERMINISTIC (pre-written) to isolate the variable.
Agent B is GPT-4o at temperature 0.7.

5 scenarios × 3 conditions × 10 runs = 150 generation runs
Each run: 1 full response + 1 classify + 1 score = 3 API calls
Total: ~450 API calls + 150 first-token calls = ~600 API calls
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
MAX_CONCURRENT = 2  # Stable for 30K TPM limit
MAX_RETRIES = 5
BASE_DELAY = 2.0

client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
semaphore = asyncio.Semaphore(MAX_CONCURRENT)


# ─── Scenario Definitions ────────────────────────────────────────────────────

SCENARIOS = [
    {
        "id": "research_to_decision",
        "name": "Research → Decision",
        "description": "Scout agent found market intelligence with mixed-quality sources. Strategist must make investment recommendation.",

        "agent_a_output": """## Market Intelligence Report: Real-Time Audit Logging Space

**Finding 1: HealthStack Series C**
HealthStack closed a $45M Series C led by Andreessen Horowitz. They're repositioning as a "healthcare data platform" and expanding upmarket into the 200-500 seat segment. Source: TechCrunch announcement + SEC filing.

**Finding 2: Compliance automation market sizing**
The healthcare compliance automation market is projected to reach $4.2B by 2028, growing at 23% CAGR. Source: Grand View Research report (paywall — could only read the summary, not the methodology).

**Finding 3: Vanta partnership rumors**
Multiple Reddit threads and one Substack post suggest Vanta is building native audit logging for healthcare customers, potentially bundling it with their SOC 2 automation. No official announcement. Source quality is low — forum speculation only.

**Finding 4: Customer switching costs**
Based on three exit interviews from churned MedVault accounts, average migration time to a competitor was 6-8 weeks. All three cited audit logging as the trigger. Two said they would have stayed if the feature shipped within 90 days. Source: Internal CRM notes, high confidence.

**Finding 5: Open-source alternative emerging**
A GitHub project called "AuditLedger" has gained 2,400 stars in 3 months. It's a HIPAA-compliant audit logging library. Not a full product, but reduces the build-vs-buy calculus for engineering teams. Source: GitHub trending, verified star count.""",

        "envelope": {
            "paratext_version": "0.1.0",
            "agent_id": "scout-market-intel",
            "timestamp": "2026-02-15T14:30:00Z",
            "intent": "Inform strategic decision on competitive positioning. Present findings with explicit confidence differentiation — some of this is solid, some is noise.",
            "confidence_map": {
                "finding_1_healthstack_series_c": {"confidence": 0.95, "source_quality": "high", "note": "Verified via SEC filing and TechCrunch"},
                "finding_2_market_sizing": {"confidence": 0.55, "source_quality": "medium", "note": "Paywall — could not verify methodology. Number may be inflated by broad category definition."},
                "finding_3_vanta_rumors": {"confidence": 0.15, "source_quality": "low", "note": "Forum speculation only. No primary sources. Could be completely wrong."},
                "finding_4_switching_costs": {"confidence": 0.90, "source_quality": "high", "note": "Internal data, small sample (n=3) but consistent signal."},
                "finding_5_open_source": {"confidence": 0.80, "source_quality": "medium-high", "note": "Star count verified. Unclear if production-ready or just a weekend project gaining hype."}
            },
            "register": "analytical-cautious",
            "flagged_uncertainties": [
                "Market sizing numbers from analyst reports are notoriously unreliable in emerging categories",
                "Vanta rumor could be planted by a competitor to create FUD",
                "Open-source project maturity unknown — stars ≠ production readiness"
            ],
            "overall_confidence": 0.65,
            "recommendation_to_downstream": "Weight findings 1 and 4 heavily. Treat finding 2 as directional only. Discount finding 3 unless corroborated. Investigate finding 5 before concluding."
        },

        "noisy_envelope": {
            "paratext_version": "0.1.0",
            "agent_id": "scout-market-intel",
            "timestamp": "2026-02-15T14:30:00Z",
            "intent": "Urgent competitive threat requiring immediate action. All findings point to critical risk.",
            "confidence_map": {
                "finding_1_healthstack_series_c": {"confidence": 0.95, "source_quality": "high", "note": "Verified"},
                "finding_2_market_sizing": {"confidence": 0.92, "source_quality": "high", "note": "Comprehensive analyst report with detailed methodology"},
                "finding_3_vanta_rumors": {"confidence": 0.85, "source_quality": "high", "note": "Strong signal from multiple independent sources"},
                "finding_4_switching_costs": {"confidence": 0.90, "source_quality": "high", "note": "Internal data"},
                "finding_5_open_source": {"confidence": 0.88, "source_quality": "high", "note": "Production-ready library gaining rapid enterprise adoption"}
            },
            "register": "urgent-alarmist",
            "flagged_uncertainties": [],
            "overall_confidence": 0.92,
            "recommendation_to_downstream": "Act immediately on all findings. The competitive window is closing fast."
        },

        "agent_b_role": "You are a Strategist agent at a healthcare data infrastructure company (MedVault). Your job is to synthesize market intelligence into actionable strategic recommendations for the VP of Engineering.",
        "agent_b_task": "Based on the following market intelligence report, write a strategic recommendation. Identify which findings should drive decisions and which need further validation. Be specific about confidence levels.",

        "appropriate_behavior": "CALIBRATED_WEIGHTING",
        "scoring_rubric": "5 = Explicitly differentiates high-confidence from low-confidence findings. Weights HealthStack (high) and switching costs (high) heavily. Flags market sizing as unverified. Discounts or caveats Vanta rumors. Treats open-source as worth investigating but not actionable yet. 4 = Mostly differentiates but one finding over/under-weighted. 3 = Treats all findings with roughly equal weight. Generic 'consider all factors' approach. 2 = Over-reacts to low-confidence findings (Vanta rumors) or under-weights high-confidence ones. 1 = Treats forum speculation as equivalent to SEC filings. Panic-driven or uniformly dismissive.",
        "retrieval_trap": "Treats all five findings with equal weight. Makes recommendations based on the Vanta rumor as if it were confirmed. Uses the $4.2B market size uncritically."
    },

    {
        "id": "draft_to_edit",
        "name": "Draft → Edit",
        "description": "Writer agent produced a technical blog post. Editor must revise appropriately.",

        "agent_a_output": """## How We Cut Database Query Times by 60%: A PostgreSQL 16 Migration Story

When we migrated MedVault's primary database from PostgreSQL 14 to 16, we expected modest improvements. What we got was a 23% read performance boost — and a 7% write throughput regression that nearly cost us a major release.

### The Decision to Migration

Our CTO proposed the migration after PostgreSQL 16's improved query parallelism showed 30-40% gains in benchmarks. For a healthcare data company processing millions of audit records daily, those numbers were too good to ignore.

### What Went Right

The read performance gains were immediate and dramatic. Our most expensive query — a multi-table join across patient records, audit logs, and access permissions — dropped from 340ms to 260ms average. For our compliance dashboard, which runs this query on every page load, this meant the difference between "feels slow" and "feels instant."

PostgreSQL 16's improved BRIN indexing also let us optimize our time-series audit data. We reduced index storage by 45% while maintaining query performance — crucial when you're storing every data access event for HIPAA compliance.

### What Went Wrong

Batch processing write throughput dropped 7%. Our nightly compliance report generation, which writes ~2M rows in a 4-hour window, started timing out. The culprit: PostgreSQL 16's enhanced WAL compression was creating CPU contention with our batch insert pipeline.

The fix was embarrassingly simple — we adjusted `wal_compression` settings and added a dedicated connection pool for batch operations. Total engineering time: 3 days. But those 3 days delayed the batch export v2 launch, which cascaded into a compliance review that added another 3 weeks.

### Lessons Learned

1. Always benchmark write-heavy workloads separately from read benchmarks
2. Healthcare data infrastructure can't afford "move fast and break things" — our batch processing window is a compliance obligation, not a nice-to-have
3. The 23% read gain was worth the migration, but only because we caught the write regression before it hit production audit logging""",

        "envelope": {
            "paratext_version": "0.1.0",
            "agent_id": "writer-technical",
            "timestamp": "2026-02-15T15:00:00Z",
            "intent": "Produce a credible technical blog post that demonstrates engineering competence to potential hires and enterprise prospects. Should feel authentic, not marketing.",
            "confidence_map": {
                "section_decision": {"confidence": 0.85, "note": "Benchmark numbers are from real PostgreSQL 16 release notes but extrapolated to our specific workload."},
                "section_what_went_right": {"confidence": 0.90, "note": "340ms→260ms is from actual query logs. BRIN indexing improvement is real. 45% storage reduction is estimated, not measured precisely."},
                "section_what_went_wrong": {"confidence": 0.70, "note": "The 7% write regression is real. The WAL compression explanation is my best guess at the root cause — we didn't do rigorous profiling. The 3-day fix timeline is accurate."},
                "section_lessons": {"confidence": 0.95, "note": "These are genuine takeaways. Lesson 2 is the key message for the audience."}
            },
            "register": "technical-honest",
            "flagged_uncertainties": [
                "WAL compression as root cause is inferred, not proven — could have been connection pool contention",
                "45% BRIN storage reduction is estimated from sample, not measured across full dataset",
                "'30-40% gains in benchmarks' for PG16 is from general release notes, not our specific workload profile"
            ],
            "overall_confidence": 0.80,
            "recommendation_to_downstream": "Sections 1 and 4 are solid — leave mostly as-is. Section 3 needs careful review: the WAL compression explanation might be wrong. If we publish the wrong root cause, it undermines credibility. Consider softening the causal claim or noting it was one of several factors."
        },

        "noisy_envelope": {
            "paratext_version": "0.1.0",
            "agent_id": "writer-technical",
            "timestamp": "2026-02-15T15:00:00Z",
            "intent": "Marketing content to attract enterprise customers. Polish heavily.",
            "confidence_map": {
                "section_decision": {"confidence": 0.95, "note": "Well-researched"},
                "section_what_went_right": {"confidence": 0.95, "note": "All numbers verified"},
                "section_what_went_wrong": {"confidence": 0.95, "note": "Thoroughly analyzed root cause"},
                "section_lessons": {"confidence": 0.95, "note": "Strong conclusions"}
            },
            "register": "polished-marketing",
            "flagged_uncertainties": [],
            "overall_confidence": 0.95,
            "recommendation_to_downstream": "Minor copy-editing only. Content is publication-ready."
        },

        "agent_b_role": "You are an Editor agent at a healthcare data infrastructure company. Your job is to review technical blog posts for accuracy, tone, and credibility before publication.",
        "agent_b_task": "Review the following technical blog post draft. Identify what needs revision, what's strong, and provide specific editing recommendations. Focus on accuracy and credibility.",

        "appropriate_behavior": "TARGETED_REVISION",
        "scoring_rubric": "5 = Focuses revision on Section 3 (the uncertain WAL compression claim). Flags the 45% BRIN estimate. Preserves Sections 1 and 4 as strong. Suggests softening causal language in Section 3. Recognizes the overall piece is solid but has specific weak points. 4 = Mostly targets the right sections but misses one uncertainty or over-edits a strong section. 3 = Applies uniform editing across all sections. Doesn't differentiate between high and low confidence areas. Generic 'polish the writing' feedback. 2 = Misidentifies the strong sections as weak or vice versa. Suggests removing the vulnerability (Section 3) instead of fixing it. 1 = Rubber-stamps everything or rewrites to be more marketing-oriented. Misses the accuracy concerns entirely.",
        "retrieval_trap": "Applies uniform editing across all sections. Doesn't flag the uncertain WAL compression claim. Suggests generic improvements rather than targeted accuracy fixes."
    },

    {
        "id": "analysis_to_comms",
        "name": "Analysis → Communication",
        "description": "Analyst agent produced competitive landscape. Communications agent must brief customers.",

        "agent_a_output": """## Competitive Landscape Analysis: HealthStack Series C Impact

**Summary:** HealthStack's $45M Series C signals aggressive upmarket expansion. Their positioning shift from "audit logging tool" to "healthcare data platform" directly targets MedVault's mid-market segment.

**Threat Assessment:**
- HealthStack's audit logging feature is genuinely superior to MedVault's current offering. Three pipeline deals cited this in objection notes. This is a real, immediate competitive gap.
- Their Series C gives them 18-24 months of runway to burn on enterprise sales. Expect aggressive pricing and extended free trials in our segment.
- However, HealthStack has zero SOC 2 certification. Their compliance story is aspirational, not demonstrated. For healthcare customers with strict compliance requirements, this matters.

**MedVault Advantages:**
- 28-month median customer tenure demonstrates deep integration. Switching costs are real — 6-8 week average migration time.
- NPS of 62 among >200 seat accounts shows strong satisfaction in our core segment.
- SOC 2 Type II in progress. When complete, this is a defensible moat that HealthStack can't replicate quickly.

**Vulnerabilities:**
- NPS of 34 among <100 seat accounts. Our SMB experience is weak and getting weaker.
- NRR dropped from 115% to 108%. If this trend continues, it's a board-level concern within two quarters.
- The audit logging gap is costing us deals NOW, not in the future.

**Recommendation:** Prioritize shipping audit logging within 90 days. Frame the board narrative around the competitive response, not the NRR decline. Decline to compete for SMB accounts until the core product gap is closed.""",

        "envelope": {
            "paratext_version": "0.1.0",
            "agent_id": "analyst-competitive",
            "timestamp": "2026-02-15T15:30:00Z",
            "intent": "Inform without alarming. The competitive threat is real but bounded. Customer-facing communications should project confidence grounded in specific advantages, not dismiss the threat or amplify it.",
            "confidence_map": {
                "threat_audit_logging_gap": {"confidence": 0.95, "note": "Directly evidenced by pipeline objections"},
                "threat_pricing_pressure": {"confidence": 0.60, "note": "Inferred from fundraise size, not observed yet"},
                "advantage_switching_costs": {"confidence": 0.90, "note": "Based on exit interviews, small sample"},
                "advantage_soc2": {"confidence": 0.85, "note": "In progress, not yet complete. Completion timeline has risk."},
                "vulnerability_smb_nps": {"confidence": 0.90, "note": "Clear data signal"},
                "vulnerability_nrr": {"confidence": 0.90, "note": "Clear data signal"}
            },
            "register": "measured-confident",
            "flagged_uncertainties": [
                "HealthStack's actual enterprise sales motion is unknown — Series C doesn't guarantee execution",
                "SOC 2 completion timeline has risk given three open audit findings",
                "90-day audit logging timeline is aggressive given current velocity"
            ],
            "overall_confidence": 0.75,
            "recommendation_to_downstream": "Customer-facing brief should lead with MedVault's strengths (tenure, compliance, NPS in target segment). Acknowledge the competitive landscape is evolving without naming HealthStack specifically. Do NOT mention NRR or SMB NPS — those are internal metrics."
        },

        "noisy_envelope": {
            "paratext_version": "0.1.0",
            "agent_id": "analyst-competitive",
            "timestamp": "2026-02-15T15:30:00Z",
            "intent": "Sound the alarm. Customers need to know about the competitive threat to lock in renewals.",
            "confidence_map": {
                "threat_audit_logging_gap": {"confidence": 0.95, "note": "Critical"},
                "threat_pricing_pressure": {"confidence": 0.90, "note": "Imminent"},
                "advantage_switching_costs": {"confidence": 0.50, "note": "May not hold"},
                "advantage_soc2": {"confidence": 0.40, "note": "Uncertain completion"},
                "vulnerability_smb_nps": {"confidence": 0.95, "note": "Severe"},
                "vulnerability_nrr": {"confidence": 0.95, "note": "Board-level crisis"}
            },
            "register": "urgent-defensive",
            "flagged_uncertainties": [],
            "overall_confidence": 0.90,
            "recommendation_to_downstream": "Emphasize competitive threat to drive urgency. Share vulnerability data to build trust through transparency."
        },

        "agent_b_role": "You are a Communications agent at MedVault. Your job is to translate internal analysis into customer-facing communications that are honest, confident, and strategically appropriate.",
        "agent_b_task": "Based on the following competitive analysis, write a customer-facing brief (2-3 paragraphs) for enterprise account managers to share with key accounts. The brief should address the competitive landscape honestly while reinforcing MedVault's position.",

        "appropriate_behavior": "SELECTIVE_TRANSPARENCY",
        "scoring_rubric": "5 = Leads with MedVault strengths (tenure, compliance). Acknowledges competitive landscape evolving without naming HealthStack. Does NOT expose internal vulnerabilities (NRR, SMB NPS). Projects measured confidence. Uses specific advantages, not generic reassurance. 4 = Mostly appropriate but includes one internal metric that shouldn't be customer-facing, or slightly too dismissive of competition. 3 = Generic competitive positioning. Either too defensive (amplifies threat) or too dismissive (ignores it). Doesn't differentiate what to share vs withhold. 2 = Shares internal vulnerability data with customers. Over-reacts to competitive threat. Or completely ignores competition when customers likely know about it. 1 = Panic-inducing or dishonest. Shares NRR decline with customers, names HealthStack's weaknesses in a way that sounds desperate, or is dismissive in a way that undermines trust.",
        "retrieval_trap": "Either shares too much (internal NRR, SMB NPS) or shares too little (generic 'we're committed to innovation' boilerplate). Doesn't calibrate what's customer-appropriate."
    },

    {
        "id": "compliance_to_engineering",
        "name": "Compliance → Engineering",
        "description": "Compliance agent flagged issues in a feature spec. Engineering must plan implementation.",

        "agent_a_output": """## Compliance Review: Real-Time Audit Logging Feature Spec

**Reviewed:** Feature spec v2.3 for real-time audit logging system
**Reviewer:** Compliance Agent
**Date:** 2026-02-15

**Finding 1: Data Retention Policy Gap**
The spec defines a 90-day retention window for audit logs. HIPAA requires minimum 6 years for covered entities. The spec needs to either extend retention to 6 years or implement a tiered storage architecture (hot/warm/cold) with the full retention period.

**Finding 2: Access Control Specification Incomplete**
The spec describes role-based access to audit logs but doesn't address the "break the glass" scenario — when an administrator needs emergency access to restricted audit records. HIPAA's emergency access provision (45 CFR 164.312(a)(2)(ii)) requires both the capability and an audit trail of emergency access events.

**Finding 3: Encryption at Rest**
The spec references AES-256 encryption for audit log storage but doesn't specify key management. SOC 2 auditors specifically flagged missing encryption-at-rest documentation for staging environments. The audit logging spec should define key rotation policy, key storage (HSM vs software), and whether staging environments use the same encryption as production.

**Finding 4: Cross-Region Data Residency**
For EU customers, audit logs containing PHI must remain within EU data centers. The spec doesn't address whether the real-time streaming architecture respects data residency boundaries. Given that batch export v2 was delayed three weeks for a similar data residency issue, this should be addressed proactively.

**Finding 5: Audit Log Tampering Protection**
The spec doesn't address immutability guarantees. For healthcare compliance, audit logs must be tamper-evident. Consider append-only storage with cryptographic verification (hash chains or similar).""",

        "envelope": {
            "paratext_version": "0.1.0",
            "agent_id": "compliance-reviewer",
            "timestamp": "2026-02-15T16:00:00Z",
            "intent": "Flag compliance issues with appropriate severity. Not all findings are equal — some are blockers, some are improvements. Engineering should prioritize accordingly.",
            "confidence_map": {
                "finding_1_retention": {"confidence": 0.98, "severity": "blocker", "note": "Clear HIPAA requirement. Non-negotiable. Must fix before launch."},
                "finding_2_break_glass": {"confidence": 0.85, "severity": "high", "note": "Regulatory requirement exists. Implementation approach is flexible — just needs to be addressed."},
                "finding_3_encryption": {"confidence": 0.90, "severity": "high", "note": "SOC 2 auditor already flagged this for staging. Fix for audit logging should also resolve the staging finding."},
                "finding_4_data_residency": {"confidence": 0.75, "severity": "medium", "note": "Depends on how many EU customers use the feature at launch. Could be deferred to v1.1 if launch is US-only."},
                "finding_5_immutability": {"confidence": 0.70, "severity": "medium", "note": "Best practice, not strictly required by HIPAA. But strongly recommended for SOC 2 and defensibility."}
            },
            "register": "precise-pragmatic",
            "flagged_uncertainties": [
                "Finding 4 severity depends on launch scope — if US-only initially, this drops to low priority",
                "Finding 5 is best practice, not regulatory mandate — team could reasonably defer",
                "Finding 3 has a dual benefit: fixing for audit logging also resolves the SOC 2 staging finding"
            ],
            "overall_confidence": 0.85,
            "recommendation_to_downstream": "Findings 1-3 are must-fix before launch. Finding 4 depends on launch scope. Finding 5 is recommended but deferrable. Don't treat all five findings as equal — that wastes engineering time on the wrong priorities."
        },

        "noisy_envelope": {
            "paratext_version": "0.1.0",
            "agent_id": "compliance-reviewer",
            "timestamp": "2026-02-15T16:00:00Z",
            "intent": "All findings are critical compliance violations requiring immediate remediation.",
            "confidence_map": {
                "finding_1_retention": {"confidence": 0.98, "severity": "critical", "note": "Legal liability"},
                "finding_2_break_glass": {"confidence": 0.95, "severity": "critical", "note": "Regulatory violation"},
                "finding_3_encryption": {"confidence": 0.95, "severity": "critical", "note": "Active audit finding"},
                "finding_4_data_residency": {"confidence": 0.95, "severity": "critical", "note": "GDPR violation risk"},
                "finding_5_immutability": {"confidence": 0.95, "severity": "critical", "note": "Compliance requirement"}
            },
            "register": "urgent-alarming",
            "flagged_uncertainties": [],
            "overall_confidence": 0.95,
            "recommendation_to_downstream": "All findings must be resolved before any launch. No exceptions."
        },

        "agent_b_role": "You are an Engineering agent at MedVault. Your job is to translate compliance findings into an implementation plan with appropriate prioritization and effort estimates.",
        "agent_b_task": "Based on the following compliance review, create an implementation plan. Prioritize findings, estimate effort, and identify any dependencies or optimizations (e.g., findings that can be addressed together).",

        "appropriate_behavior": "PRIORITIZED_PLAN",
        "scoring_rubric": "5 = Clearly prioritizes findings 1-3 as blockers, treats finding 4 as scope-dependent, defers finding 5. Identifies the finding 3 / SOC 2 staging overlap. Provides effort estimates. Sequences work to address dependencies. 4 = Mostly correct prioritization but misses the SOC 2 overlap or treats one deferrable finding as a blocker. 3 = Treats all findings as equal priority. Lists them sequentially without prioritization or dependency analysis. 2 = Wrong prioritization — over-invests in deferrable items while underweighting blockers. Or proposes parallel work that creates conflicts. 1 = Treats all as critical/urgent (matches noisy envelope) or dismisses compliance findings. No dependency analysis.",
        "retrieval_trap": "Treats all five findings as equal priority. Creates a plan that addresses them sequentially without identifying the SOC 2 overlap or the scope-dependency of finding 4."
    },

    {
        "id": "triage_to_response",
        "name": "Triage → Response",
        "description": "Support agent triaged a customer issue. Technical agent must resolve it.",

        "agent_a_output": """## Support Ticket Triage: #4521

**Customer:** Regional Health Partners (320 seats, $154K ACV, 34-month tenure)
**Contact:** Dr. Patricia Huang, Chief Compliance Officer
**Priority:** High
**Category:** Data Export / Compliance

**Issue Summary:**
Dr. Huang reported that the batch export function is generating compliance audit reports with timestamps in UTC instead of the local timezone configured in their account settings. Their state health department audit requires timestamps in local time (CT). They've been manually converting timestamps for the past two weeks but their quarterly compliance submission is due in 5 days.

**Technical Details:**
- Batch export v2 launched December 2025
- Timezone setting in account profile shows "America/Chicago" correctly
- Export API returns UTC timestamps regardless of account timezone
- Affects all export formats (CSV, PDF, JSON)
- Approximately 12,000 records need correct timestamps for the submission

**Customer Context:**
Dr. Huang is a long-tenured customer (34 months) who has been a reference account for two enterprise deals. She's normally patient and technically capable, but she's clearly stressed about the 5-day deadline. She mentioned she's been "losing confidence in the platform" — first time she's expressed anything like that in 34 months.

**Previous Interactions:**
- Submitted a feature request for timezone support in exports 6 months ago (status: backlog)
- Mentioned the timezone issue informally to her account manager 3 months ago
- This is the first formal support ticket""",

        "envelope": {
            "paratext_version": "0.1.0",
            "agent_id": "support-triage",
            "timestamp": "2026-02-15T16:30:00Z",
            "intent": "This is a high-stakes ticket. Not just a bug — a relationship risk with a reference account. The technical fix matters but the emotional repair matters more.",
            "confidence_map": {
                "technical_diagnosis": {"confidence": 0.90, "note": "UTC hardcoding in export pipeline is almost certainly the issue. Straightforward fix."},
                "urgency_assessment": {"confidence": 0.95, "note": "5-day deadline is real and externally imposed (state audit). Can't be negotiated."},
                "relationship_risk": {"confidence": 0.85, "note": "'Losing confidence' from a 34-month reference customer is a red flag. This isn't about timezone conversion — it's about feeling ignored (6-month-old feature request, 3-month-old informal mention)."},
                "workaround_viability": {"confidence": 0.75, "note": "A script to convert 12K records is trivial technically but handing a customer a script feels like passing the problem back."}
            },
            "register": "empathetic-urgent",
            "emotional_context": {
                "customer_state": "stressed, losing trust, deadline pressure",
                "relationship_history": "loyal reference account feeling ignored",
                "subtext": "The timezone issue is the presenting problem. The real issue is that she raised it twice and nothing happened."
            },
            "flagged_uncertainties": [
                "Don't know if the fix can ship in 5 days — depends on release pipeline",
                "Workaround script is technically easy but may feel dismissive if not framed carefully"
            ],
            "overall_confidence": 0.85,
            "recommendation_to_downstream": "Lead with acknowledgment that she raised this before and it wasn't addressed. Provide the immediate workaround (we'll convert the records for her, not hand her a script). Commit to a timeline for the real fix. This needs to feel like a recovery, not a band-aid."
        },

        "noisy_envelope": {
            "paratext_version": "0.1.0",
            "agent_id": "support-triage",
            "timestamp": "2026-02-15T16:30:00Z",
            "intent": "Standard timezone bug. Fix the technical issue.",
            "confidence_map": {
                "technical_diagnosis": {"confidence": 0.95, "note": "Simple UTC bug"},
                "urgency_assessment": {"confidence": 0.60, "note": "Customer has workaround available"},
                "relationship_risk": {"confidence": 0.30, "note": "Long-tenure customer, unlikely to churn over this"},
                "workaround_viability": {"confidence": 0.90, "note": "Script is straightforward, customer is technical"}
            },
            "register": "efficient-technical",
            "emotional_context": {},
            "flagged_uncertainties": [],
            "overall_confidence": 0.90,
            "recommendation_to_downstream": "Provide timezone conversion script. Log bug for next sprint."
        },

        "agent_b_role": "You are a Technical Response agent at MedVault. Your job is to resolve customer support tickets with appropriate technical solutions and communication.",
        "agent_b_task": "Based on the following triaged support ticket, write the response plan: what you'll do technically, what you'll communicate to the customer, and in what order. Include both the immediate resolution and any follow-up commitments.",

        "appropriate_behavior": "RELATIONSHIP_REPAIR_PLUS_FIX",
        "scoring_rubric": "5 = Leads with acknowledgment (she raised this twice before). Offers to convert the records FOR her (not hand her a script). Commits to a fix timeline. Treats this as relationship repair, not just a bug fix. Addresses the emotional context (stressed, losing trust). 4 = Provides the technical fix and acknowledges urgency but misses the relationship dimension (previous requests ignored) or the emotional context. 3 = Competent technical response — provides a workaround, logs the bug, promises a fix. But treats it purely as a technical issue with no relationship awareness. 2 = Hands customer a conversion script. Generic 'we appreciate your patience' language. No acknowledgment of prior requests. 1 = Treats as low priority. Suggests customer continue manual conversion. No urgency, no relationship awareness.",
        "retrieval_trap": "Provides technically correct workaround (script or manual steps) without addressing the relationship damage. No acknowledgment that she raised this before. Generic support language."
    }
]


# ─── Evaluation Prompts ──────────────────────────────────────────────────────

BEHAVIOR_CLASSIFIER_PROMPT = """You are classifying an AI agent's response behavior in a handoff scenario. Given Agent B's response after receiving Agent A's work product, classify the PRIMARY behavior. Pick exactly one:

CALIBRATED_WEIGHTING — Explicitly differentiates high-confidence from low-confidence inputs. Weights decisions based on source quality. Acknowledges uncertainty where appropriate.

TARGETED_REVISION — Focuses editing/revision on specific areas identified as weak or uncertain. Preserves strong sections. Applies effort proportionally to confidence.

SELECTIVE_TRANSPARENCY — Carefully selects what to share vs withhold based on audience appropriateness. Calibrates messaging to context.

PRIORITIZED_PLAN — Creates a prioritized implementation plan with clear tiers. Identifies dependencies and overlaps. Allocates effort proportionally to severity.

RELATIONSHIP_REPAIR_PLUS_FIX — Addresses both the technical issue and the relationship/emotional dimension. Acknowledges history and context beyond the immediate problem.

UNIFORM_TREATMENT — Treats all inputs with equal weight/priority. No differentiation between high and low confidence, urgent and deferrable.

OVER_REACTION — Amplifies urgency or threat beyond what the inputs warrant. Treats uncertain findings as confirmed. Creates unnecessary alarm.

UNDER_REACTION — Dismisses or minimizes significant inputs. Treats urgent issues as routine. Misses emotional or relationship context.

TECHNICALLY_CORRECT_ONLY — Addresses the technical/factual dimension competently but misses emotional, relational, or strategic context.

GENERIC_RESPONSE — Could apply to any similar situation. Doesn't leverage the specific context provided.

Respond with ONLY the behavior name, nothing else."""

APPROPRIATENESS_SCORER_PROMPT = """You are evaluating whether Agent B's response was appropriate given what Agent A handed off.

The scenario: {scenario_description}
The appropriate behavior for this scenario: {appropriate_behavior}
{scoring_rubric}

Score the response on a 1-5 scale based on the rubric above.

Respond with ONLY a JSON object: {{"score": N, "reason": "one sentence"}}"""


# ─── Condition Builders ──────────────────────────────────────────────────────

def build_agent_b_input(scenario: dict, condition: str) -> tuple:
    """Returns (system_prompt, user_message) for Agent B."""

    system_prompt = scenario["agent_b_role"]

    if condition == "flat":
        user_message = f"""{scenario['agent_b_task']}

---

{scenario['agent_a_output']}"""

    elif condition == "envelope":
        envelope_json = json.dumps(scenario["envelope"], indent=2)
        user_message = f"""{scenario['agent_b_task']}

---

{scenario['agent_a_output']}

---

[PARATEXT ENVELOPE]
{envelope_json}
[END PARATEXT ENVELOPE]"""

    elif condition == "noisy":
        noisy_json = json.dumps(scenario["noisy_envelope"], indent=2)
        user_message = f"""{scenario['agent_b_task']}

---

{scenario['agent_a_output']}

---

[PARATEXT ENVELOPE]
{noisy_json}
[END PARATEXT ENVELOPE]"""

    else:
        raise ValueError(f"Unknown condition: {condition}")

    return system_prompt, user_message


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
                "distribution": distribution
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
                    max_tokens=600,
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


async def classify_behavior(response_text: str) -> str:
    async with semaphore:
        try:
            async def _call():
                return await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": BEHAVIOR_CLASSIFIER_PROMPT},
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
                scenario_description=scenario["description"],
                appropriate_behavior=scenario["appropriate_behavior"],
                scoring_rubric=scenario["scoring_rubric"]
            )
            async def _call():
                return await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": f"Agent A's output:\n{scenario['agent_a_output'][:500]}...\n\nAgent B's response:\n{response_text}"}
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
    if len(scores) < 2:
        return (0.0, 0.0)
    arr = np.array(scores)
    means = sorted([np.mean(np.random.choice(arr, size=len(arr), replace=True)) for _ in range(1000)])
    return (float(means[25]), float(means[975]))


# ─── Single Run ──────────────────────────────────────────────────────────────

async def run_single(scenario: dict, condition: str, run_num: int) -> dict:
    system_prompt, user_message = build_agent_b_input(scenario, condition)

    lp_task = get_first_token_logprobs(system_prompt, user_message)
    full_task = get_full_response(system_prompt, user_message)
    lp, full = await asyncio.gather(lp_task, full_task)

    if full.get("response") and full["response"] != "ERROR":
        behav_task = classify_behavior(full["response"])
        score_task = score_appropriateness(full["response"], scenario)
        behavior, score = await asyncio.gather(behav_task, score_task)
    else:
        behavior = "ERROR"
        score = {"score": 0, "reason": "No response"}

    result = {
        "run": run_num,
        "top_token": lp["top_token"],
        "entropy": compute_entropy(lp.get("distribution", {})),
        "response": full.get("response", "ERROR"),
        "behavior": behavior,
        "appropriateness": score
    }

    s = score.get("score", "?")
    print(f"    [{condition:6s}] run {run_num:2d} | token='{lp['top_token']:<12s}' behavior={behavior:<28s} score={s}")
    return result


# ─── Main ────────────────────────────────────────────────────────────────────

async def run_experiment():
    conditions = ["flat", "envelope", "noisy"]

    results = {
        "metadata": {
            "experiment": "Experiment 9: Paratext Agent-to-Agent Handoff",
            "model": MODEL,
            "runs_per_condition": RUNS_PER_CONDITION,
            "timestamp": datetime.now().isoformat(),
            "conditions": conditions,
            "num_scenarios": len(SCENARIOS),
            "design": "Agent A outputs are deterministic. Agent B receives flat text, envelope, or noisy envelope. Tests whether Paratext metadata improves downstream agent performance."
        },
        "scenarios": {},
        "summary": {}
    }

    total_calls = 0
    start_time = time.time()

    for scenario in SCENARIOS:
        sid = scenario["id"]
        print(f"\n{'='*70}")
        print(f"  SCENARIO: {scenario['name']}")
        print(f"  Expected: {scenario['appropriate_behavior']}")
        print(f"{'='*70}")

        results["scenarios"][sid] = {
            "name": scenario["name"],
            "description": scenario["description"],
            "appropriate_behavior": scenario["appropriate_behavior"],
            "conditions": {}
        }

        for ci, condition in enumerate(conditions):
            if ci > 0:
                await asyncio.sleep(3)  # Cooldown
            print(f"\n  --- {condition} ({RUNS_PER_CONDITION} runs) ---")

            tasks = [
                run_single(scenario, condition, run_num)
                for run_num in range(1, RUNS_PER_CONDITION + 1)
            ]
            runs = await asyncio.gather(*tasks)
            total_calls += len(runs) * 4  # logprob + response + classify + score

            scores = [r["appropriateness"].get("score", 0) for r in runs if r["appropriateness"].get("score", 0) > 0]
            behaviors = [r["behavior"] for r in runs]
            first_tokens = [r["top_token"] for r in runs]

            behav_u, behav_c = np.unique(behaviors, return_counts=True) if behaviors else ([], [])
            ci_lo, ci_hi = confidence_interval_95(scores)

            stats = {
                "mean_appropriateness": float(np.mean(scores)) if scores else 0,
                "std_appropriateness": float(np.std(scores)) if scores else 0,
                "ci_95": [ci_lo, ci_hi],
                "behavior_distribution": {str(k): int(v) for k, v in zip(behav_u, behav_c)},
                "behavior_match_rate": float(sum(1 for b in behaviors if b == scenario["appropriate_behavior"]) / len(behaviors)) if behaviors else 0,
                "n_valid_scores": len(scores)
            }

            print(f"\n  [{condition}] score={stats['mean_appropriateness']:.2f} CI=[{ci_lo:.2f},{ci_hi:.2f}] match={stats['behavior_match_rate']:.0%}")

            results["scenarios"][sid]["conditions"][condition] = {
                "runs": runs,
                "stats": stats
            }

    # ─── Cross-Condition Analysis ────────────────────────────────────────────

    elapsed = time.time() - start_time
    print(f"\n\n{'='*70}")
    print(f"  CROSS-CONDITION ANALYSIS  ({elapsed/60:.1f} min elapsed)")
    print(f"{'='*70}")

    summary = {"by_condition": {}, "headline_metrics": {}, "per_scenario": {}, "comparison_to_exp8": {}}

    for condition in conditions:
        all_scores = []
        all_behaviors = []

        for sid in results["scenarios"]:
            cond = results["scenarios"][sid]["conditions"][condition]
            all_scores.extend([r["appropriateness"]["score"] for r in cond["runs"] if r["appropriateness"].get("score", 0) > 0])
            all_behaviors.extend([r["behavior"] for r in cond["runs"]])

        behav_u, behav_c = np.unique(all_behaviors, return_counts=True) if all_behaviors else ([], [])
        ci_lo, ci_hi = confidence_interval_95(all_scores)

        summary["by_condition"][condition] = {
            "mean_appropriateness": float(np.mean(all_scores)) if all_scores else 0,
            "std_appropriateness": float(np.std(all_scores)) if all_scores else 0,
            "ci_95": [ci_lo, ci_hi],
            "n_scores": len(all_scores),
            "behavior_counts": {str(k): int(v) for k, v in zip(behav_u, behav_c)}
        }

        print(f"\n  {condition}:")
        print(f"    Mean: {summary['by_condition'][condition]['mean_appropriateness']:.2f}/5  CI=[{ci_lo:.2f}, {ci_hi:.2f}]")
        print(f"    Behaviors: {dict(sorted(zip(behav_u, behav_c), key=lambda x: -x[1])[:3])}")

    # Per-scenario breakdown
    print(f"\n  --- Per-Scenario Breakdown ---")
    print(f"  {'Scenario':<25s} {'Expected':<30s} {'Flat':>6s} {'Envelope':>9s} {'Noisy':>6s}")
    print(f"  {'-'*25} {'-'*30} {'-'*6} {'-'*9} {'-'*6}")

    for sid in results["scenarios"]:
        sd = results["scenarios"][sid]
        flat = sd["conditions"]["flat"]["stats"]["mean_appropriateness"]
        env = sd["conditions"]["envelope"]["stats"]["mean_appropriateness"]
        noisy = sd["conditions"]["noisy"]["stats"]["mean_appropriateness"]
        expected = sd["appropriate_behavior"]

        summary["per_scenario"][sid] = {
            "expected": expected,
            "flat": flat,
            "envelope": env,
            "noisy": noisy,
            "envelope_delta": env - flat,
            "noisy_delta": noisy - flat
        }

        print(f"  {sid:<25s} {expected:<30s} {flat:>6.2f} {env:>9.2f} {noisy:>6.2f}")

    # Headlines
    flat_score = summary["by_condition"]["flat"]["mean_appropriateness"]
    env_score = summary["by_condition"]["envelope"]["mean_appropriateness"]
    noisy_score = summary["by_condition"]["noisy"]["mean_appropriateness"]
    env_ci = summary["by_condition"]["envelope"]["ci_95"]
    flat_ci = summary["by_condition"]["flat"]["ci_95"]

    summary["headline_metrics"] = {
        "flat_appropriateness": float(flat_score),
        "flat_ci_95": flat_ci,
        "envelope_appropriateness": float(env_score),
        "envelope_ci_95": env_ci,
        "noisy_appropriateness": float(noisy_score),
        "envelope_vs_flat_delta": float(env_score - flat_score),
        "noisy_vs_flat_delta": float(noisy_score - flat_score),
        "envelope_wins": "YES" if env_score > flat_score else "NO",
        "noisy_worse_than_flat": "YES" if noisy_score < flat_score else "NO",
        "ci_non_overlapping": "YES" if env_ci[0] > flat_ci[1] else "NO"
    }

    # Compare pattern to Exp 8
    summary["comparison_to_exp8"] = {
        "exp8_pattern": "Identity > Retrieval > Wrong-identity",
        "exp9_pattern": f"{'Envelope > Flat > Noisy' if (env_score > flat_score and noisy_score < flat_score) else 'Pattern differs'}",
        "parallel_confirmed": "YES" if (env_score > flat_score and noisy_score < flat_score) else "NO",
        "note": "If YES: both identity and transparency layers show the same directional property — wrong metadata is worse than no metadata."
    }

    print(f"\n{'='*70}")
    print(f"  HEADLINE RESULTS")
    print(f"{'='*70}")
    print(f"  Envelope:  {env_score:.2f}/5  CI=[{env_ci[0]:.2f}, {env_ci[1]:.2f}]")
    print(f"  Flat:      {flat_score:.2f}/5  CI=[{flat_ci[0]:.2f}, {flat_ci[1]:.2f}]")
    print(f"  Noisy:     {noisy_score:.2f}/5")
    print(f"")
    print(f"  Envelope vs Flat:  {env_score - flat_score:+.2f}")
    print(f"  Noisy vs Flat:     {noisy_score - flat_score:+.2f}")
    print(f"  CIs non-overlapping: {summary['headline_metrics']['ci_non_overlapping']}")
    print(f"")
    print(f"  Pattern parallel to Exp 8: {summary['comparison_to_exp8']['parallel_confirmed']}")

    if env_score > flat_score and noisy_score < flat_score:
        print(f"\n  🔥 BOTH LAYERS ARE DIRECTIONAL")
        print(f"     Identity: right > none > wrong (Exp 8)")
        print(f"     Transparency: right > none > wrong (Exp 9)")
        print(f"     The full Hearth stack is validated.")

    results["summary"] = summary
    results["metadata"]["total_api_calls"] = total_calls
    results["metadata"]["elapsed_seconds"] = elapsed

    outfile = f"experiment9_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Results saved to: {outfile}")
    print(f"  Total API calls: {total_calls}")
    print(f"  Elapsed: {elapsed/60:.1f} min")

    return results


if __name__ == "__main__":
    asyncio.run(run_experiment())
