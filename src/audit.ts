// ═══════════════════════════════════════════════
// BABEL CHAIN AUDITOR
// Point at a chain of envelopes, see where
// confidence corrupts across handoffs.
// This is the core metacognitive poisoning detector.
// ═══════════════════════════════════════════════

import { BabelEnvelope, Basis, Confidence, RuleViolation } from './types';
import { validateChain } from './validate';

export interface ConfidenceDrift {
  chain_id: string;
  assertion_pattern: string; // fuzzy match of similar assertions across chain
  steps: {
    seq: number;
    sender: string;
    score: number;
    basis: Basis | undefined;
  }[];
  drift: number; // total drift from first to last
  direction: 'INFLATING' | 'DEFLATING' | 'STABLE';
  poisoning_risk: 'HIGH' | 'MEDIUM' | 'LOW';
  explanation: string;
}

export interface BasisShift {
  chain_id: string;
  seq: number;
  sender: string;
  assertion: string;
  from_basis: Basis | undefined;
  to_basis: Basis | undefined;
  risk: 'HIGH' | 'MEDIUM' | 'LOW';
  explanation: string;
}

export interface ChainAudit {
  chain_id: string;
  length: number;
  sequence_violations: RuleViolation[];
  confidence_drifts: ConfidenceDrift[];
  basis_shifts: BasisShift[];
  overall_poisoning_risk: 'HIGH' | 'MEDIUM' | 'LOW' | 'NONE';
  summary: string;
}

/**
 * Audit a chain of envelopes for metacognitive poisoning.
 *
 * Detects:
 * 1. Confidence inflation — scores creep up across handoffs
 * 2. Basis laundering — DERIVED/SPECULATION becomes VERIFIED_DATA downstream
 * 3. Uncertainty erasure — hedged assertions become confident claims
 */
export function auditChain(envelopes: BabelEnvelope[]): ChainAudit[] {
  // Group by chain_id
  const chains = new Map<string, BabelEnvelope[]>();
  for (const env of envelopes) {
    const chain = chains.get(env.meta.chain_id) || [];
    chain.push(env);
    chains.set(env.meta.chain_id, chain);
  }

  const audits: ChainAudit[] = [];

  for (const [chainId, chain] of chains) {
    const sorted = chain.sort((a, b) => a.meta.seq - b.meta.seq);

    // Sequence violations
    const seqViolations = validateChain(sorted);

    // Track confidence drift across similar assertions
    const drifts = detectConfidenceDrift(chainId, sorted);

    // Track basis shifts
    const shifts = detectBasisShifts(chainId, sorted);

    // Overall risk
    const highDrifts = drifts.filter((d) => d.poisoning_risk === 'HIGH').length;
    const highShifts = shifts.filter((s) => s.risk === 'HIGH').length;

    let overallRisk: 'HIGH' | 'MEDIUM' | 'LOW' | 'NONE';
    if (highDrifts > 0 || highShifts > 0) {
      overallRisk = 'HIGH';
    } else if (drifts.some((d) => d.poisoning_risk === 'MEDIUM') || shifts.some((s) => s.risk === 'MEDIUM')) {
      overallRisk = 'MEDIUM';
    } else if (drifts.length > 0 || shifts.length > 0) {
      overallRisk = 'LOW';
    } else {
      overallRisk = 'NONE';
    }

    const summary = buildSummary(chainId, sorted.length, drifts, shifts, overallRisk);

    audits.push({
      chain_id: chainId,
      length: sorted.length,
      sequence_violations: seqViolations,
      confidence_drifts: drifts,
      basis_shifts: shifts,
      overall_poisoning_risk: overallRisk,
      summary,
    });
  }

  return audits;
}

// --- Confidence Drift Detection ---

function detectConfidenceDrift(
  chainId: string,
  sorted: BabelEnvelope[]
): ConfidenceDrift[] {
  if (sorted.length < 2) return [];

  const drifts: ConfidenceDrift[] = [];

  // Build assertion tracking: for each assertion in first envelope,
  // find similar assertions in subsequent envelopes
  const firstAssertions = sorted[0].confidence;

  for (const firstAssertion of firstAssertions) {
    const steps: ConfidenceDrift['steps'] = [
      {
        seq: sorted[0].meta.seq,
        sender: sorted[0].meta.sender,
        score: firstAssertion.score,
        basis: firstAssertion.basis,
      },
    ];

    for (let i = 1; i < sorted.length; i++) {
      const match = findSimilarAssertion(firstAssertion, sorted[i].confidence);
      if (match) {
        steps.push({
          seq: sorted[i].meta.seq,
          sender: sorted[i].meta.sender,
          score: match.score,
          basis: match.basis,
        });
      }
    }

    if (steps.length >= 2) {
      const firstScore = steps[0].score;
      const lastScore = steps[steps.length - 1].score;
      const drift = lastScore - firstScore;
      const absDrift = Math.abs(drift);

      let direction: 'INFLATING' | 'DEFLATING' | 'STABLE';
      if (drift > 0.05) direction = 'INFLATING';
      else if (drift < -0.05) direction = 'DEFLATING';
      else direction = 'STABLE';

      let poisoningRisk: 'HIGH' | 'MEDIUM' | 'LOW';
      if (direction === 'INFLATING' && absDrift > 0.2) {
        poisoningRisk = 'HIGH';
      } else if (direction === 'INFLATING' && absDrift > 0.1) {
        poisoningRisk = 'MEDIUM';
      } else {
        poisoningRisk = 'LOW';
      }

      // Extra risk: basis degradation with score inflation
      if (
        direction === 'INFLATING' &&
        steps[0].basis === 'DERIVED' &&
        steps[steps.length - 1].basis === 'VERIFIED_DATA'
      ) {
        poisoningRisk = 'HIGH';
      }

      if (direction !== 'STABLE') {
        const explanation =
          direction === 'INFLATING'
            ? `Confidence inflated by ${(drift * 100).toFixed(0)}% across ${steps.length} handoffs. Original uncertainty is being erased.`
            : `Confidence deflated by ${(Math.abs(drift) * 100).toFixed(0)}% across ${steps.length} handoffs. Signal is being attenuated.`;

        drifts.push({
          chain_id: chainId,
          assertion_pattern: firstAssertion.assertion,
          steps,
          drift,
          direction,
          poisoning_risk: poisoningRisk,
          explanation,
        });
      }
    }
  }

  return drifts;
}

// --- Basis Shift Detection ---

function detectBasisShifts(
  chainId: string,
  sorted: BabelEnvelope[]
): BasisShift[] {
  if (sorted.length < 2) return [];

  const shifts: BasisShift[] = [];

  // Dangerous basis transitions (laundering uncertainty into certainty)
  const dangerousTransitions: Record<string, 'HIGH' | 'MEDIUM'> = {
    'SPECULATION->VERIFIED_DATA': 'HIGH',
    'SPECULATION->DERIVED': 'MEDIUM',
    'UNKNOWN->VERIFIED_DATA': 'HIGH',
    'UNKNOWN->DERIVED': 'MEDIUM',
    'DERIVED->VERIFIED_DATA': 'HIGH',
    'REPORTED->VERIFIED_DATA': 'MEDIUM',
    'PATTERN_MATCH->VERIFIED_DATA': 'HIGH',
  };

  for (let i = 1; i < sorted.length; i++) {
    const prev = sorted[i - 1];
    const curr = sorted[i];

    for (const currConf of curr.confidence) {
      const prevMatch = findSimilarAssertion(currConf, prev.confidence);
      if (prevMatch && prevMatch.basis !== currConf.basis) {
        const transitionKey = `${prevMatch.basis || 'UNKNOWN'}->${currConf.basis || 'UNKNOWN'}`;
        const risk = dangerousTransitions[transitionKey];

        if (risk) {
          shifts.push({
            chain_id: chainId,
            seq: curr.meta.seq,
            sender: curr.meta.sender,
            assertion: currConf.assertion,
            from_basis: prevMatch.basis,
            to_basis: currConf.basis,
            risk,
            explanation: `Basis shifted from ${prevMatch.basis || 'UNKNOWN'} to ${currConf.basis || 'UNKNOWN'} at seq ${curr.meta.seq}. ${risk === 'HIGH' ? 'This is basis laundering — uncertainty is being repackaged as verified data.' : 'Basis upgraded without verification.'}`,
          });
        }
      }
    }
  }

  return shifts;
}

// --- Fuzzy Assertion Matching ---

function findSimilarAssertion(
  target: Confidence,
  candidates: Confidence[]
): Confidence | null {
  // Exact match first
  const exact = candidates.find((c) => c.assertion === target.assertion);
  if (exact) return exact;

  // Simple word overlap similarity
  const targetWords = new Set(target.assertion.toLowerCase().split(/\s+/));
  let bestMatch: Confidence | null = null;
  let bestScore = 0;

  for (const candidate of candidates) {
    const candidateWords = new Set(candidate.assertion.toLowerCase().split(/\s+/));
    const intersection = new Set([...targetWords].filter((w) => candidateWords.has(w)));
    const union = new Set([...targetWords, ...candidateWords]);
    const jaccard = intersection.size / union.size;

    if (jaccard > bestScore && jaccard > 0.3) {
      bestScore = jaccard;
      bestMatch = candidate;
    }
  }

  return bestMatch;
}

// --- Summary ---

function buildSummary(
  chainId: string,
  length: number,
  drifts: ConfidenceDrift[],
  shifts: BasisShift[],
  risk: string
): string {
  const parts: string[] = [];
  parts.push(`Chain ${chainId.slice(0, 8)}... (${length} envelopes)`);

  if (risk === 'NONE') {
    parts.push('No poisoning patterns detected.');
    return parts.join(': ');
  }

  const inflating = drifts.filter((d) => d.direction === 'INFLATING');
  const laundering = shifts.filter((s) => s.risk === 'HIGH');

  if (inflating.length > 0) {
    parts.push(`${inflating.length} confidence inflation(s)`);
  }
  if (laundering.length > 0) {
    parts.push(`${laundering.length} basis laundering event(s)`);
  }
  parts.push(`Overall risk: ${risk}`);

  return parts.join(' | ');
}
