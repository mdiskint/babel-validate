// ═══════════════════════════════════════════════
// BABEL VALIDATOR — Grammar Rules Engine
// Spec: Babel v0.2 (Feb 16, 2026)
// Source: Notion — "Babel — Agent-to-Agent Language Protocol"
// 5 MUST rules (hard errors — envelope rejected)
// 6 SHOULD rules (warnings — envelope passes)
// ═══════════════════════════════════════════════

import {
  BabelEnvelope,
  RuleViolation,
  ValidationResult,
  Confidence,
  Basis,
} from './types';

// --- MUST Rules (hard errors) ---

/**
 * M1: intent == SPECULATE → max(confidence[].score) < 0.7
 * "You cannot speculate with high confidence. Pick one."
 */
function checkM1(envelope: BabelEnvelope): RuleViolation[] {
  if (envelope.intent !== 'SPECULATE') return [];

  const maxScore = Math.max(...envelope.confidence.map((c) => c.score));
  if (maxScore >= 0.7) {
    return [
      {
        rule: 'M1',
        severity: 'MUST',
        message: `Intent is SPECULATE but max confidence score is ${maxScore} (>= 0.7). You cannot speculate with high confidence.`,
        details: { field: 'confidence', value: maxScore, threshold: 0.7 },
      },
    ];
  }
  return [];
}

/**
 * M2: intent == REQUEST_ACTION → min(confidence[].score) > 0.3 OR grounds.length > 0
 * "Don't ask someone to act on unfounded claims unless org context justifies it."
 */
function checkM2(envelope: BabelEnvelope): RuleViolation[] {
  if (envelope.intent !== 'REQUEST_ACTION') return [];

  const minScore = Math.min(...envelope.confidence.map((c) => c.score));
  const hasGrounds = envelope.grounds && envelope.grounds.length > 0;

  if (minScore <= 0.3 && !hasGrounds) {
    return [
      {
        rule: 'M2',
        severity: 'MUST',
        message: `REQUEST_ACTION with min confidence ${minScore} (<= 0.3) and no grounds. Don't ask someone to act on unfounded claims unless org context justifies it.`,
        details: { field: 'confidence', value: minScore, threshold: 0.3 },
      },
    ];
  }
  return [];
}

/**
 * M3: grounds[].authority == REGULATORY → override == false
 * "Regulatory constraints are never overridable."
 */
function checkM3(envelope: BabelEnvelope): RuleViolation[] {
  if (!envelope.grounds) return [];

  const violations: RuleViolation[] = [];
  for (const g of envelope.grounds) {
    if (g.authority === 'REGULATORY' && g.override === true) {
      violations.push({
        rule: 'M3',
        severity: 'MUST',
        message: `Regulatory constraint "${truncate(g.constraint)}" cannot have override=true. Regulatory constraints are never overridable.`,
        details: { field: 'grounds', value: g },
      });
    }
  }
  return violations;
}

/**
 * M4: confidence[].basis == UNKNOWN → score <= 0.5
 * "If you don't know why you're confident, you can't be very confident."
 */
function checkM4(envelope: BabelEnvelope): RuleViolation[] {
  const violations: RuleViolation[] = [];
  for (const c of envelope.confidence) {
    if ((!c.basis || c.basis === 'UNKNOWN') && c.score > 0.5) {
      violations.push({
        rule: 'M4',
        severity: 'MUST',
        message: `Assertion "${truncate(c.assertion)}" has basis ${c.basis || 'undefined'} with score ${c.score} > 0.5. If you don't know why you're confident, you can't be very confident.`,
        details: { field: 'confidence', value: c, threshold: 0.5 },
      });
    }
  }
  return violations;
}

/**
 * M5: chain_id present → seq == previous_envelope.seq + 1
 * "Chain sequencing must be monotonic. No gaps, no duplicates."
 * Single-envelope: seq >= 0. Cross-envelope: see validateChain().
 */
function checkM5(envelope: BabelEnvelope): RuleViolation[] {
  if (envelope.meta.seq < 0) {
    return [
      {
        rule: 'M5',
        severity: 'MUST',
        message: `Chain sequence ${envelope.meta.seq} is negative. Sequence numbers must be >= 0.`,
        details: { field: 'meta.seq', value: envelope.meta.seq },
      },
    ];
  }
  return [];
}

// --- SHOULD Rules (warnings) ---

/**
 * S1: intent == ESCALATE AND register == CUSTOMER_EXTERNAL → WARN
 * "Escalation language directed at customers."
 */
function checkS1(envelope: BabelEnvelope): RuleViolation[] {
  if (envelope.intent === 'ESCALATE' && envelope.register === 'CUSTOMER_EXTERNAL') {
    return [
      {
        rule: 'S1',
        severity: 'SHOULD',
        message: `ESCALATE intent with CUSTOMER_EXTERNAL register. Escalation language directed at customers requires explicit override or register change.`,
        details: { field: 'register', value: envelope.register },
      },
    ];
  }
  return [];
}

/**
 * S2: affect.certainty > 0.5 AND max(confidence[].score) < 0.4 → WARN
 * "Sender feels certain but evidence is weak."
 */
function checkS2(envelope: BabelEnvelope): RuleViolation[] {
  if (!envelope.affect) return [];

  const maxScore = Math.max(...envelope.confidence.map((c) => c.score));
  if (envelope.affect.certainty > 0.5 && maxScore < 0.4) {
    return [
      {
        rule: 'S2',
        severity: 'SHOULD',
        message: `Affect certainty is ${envelope.affect.certainty} but max confidence score is only ${maxScore}. Sender feels certain but evidence is weak.`,
        details: { field: 'affect.certainty', value: envelope.affect.certainty },
      },
    ];
  }
  return [];
}

/**
 * S3: intent == INFORM AND any(confidence[].score < 0.5) → WARN
 * "Informing with low-confidence assertions — consider FLAG_RISK."
 */
function checkS3(envelope: BabelEnvelope): RuleViolation[] {
  if (envelope.intent !== 'INFORM') return [];

  const lowConf = envelope.confidence.filter((c) => c.score < 0.5);
  if (lowConf.length > 0) {
    return [
      {
        rule: 'S3',
        severity: 'SHOULD',
        message: `INFORM intent with ${lowConf.length} low-confidence assertion(s) (score < 0.5). Consider FLAG_RISK intent instead.`,
        details: {
          field: 'confidence',
          value: lowConf.map((c) => ({ assertion: truncate(c.assertion), score: c.score })),
        },
      },
    ];
  }
  return [];
}

/**
 * S4: trajectory.direction == DEGRADING AND intent == INFORM → WARN
 * "Degrading pattern reported as neutral inform — consider ESCALATE."
 */
function checkS4(envelope: BabelEnvelope): RuleViolation[] {
  if (!envelope.trajectory) return [];

  if (envelope.trajectory.direction === 'DEGRADING' && envelope.intent === 'INFORM') {
    return [
      {
        rule: 'S4',
        severity: 'SHOULD',
        message: `Degrading trajectory ("${truncate(envelope.trajectory.pattern)}") with INFORM intent. Degrading patterns deserve urgency — consider ESCALATE.`,
        details: { field: 'trajectory.direction', value: envelope.trajectory.direction },
      },
    ];
  }
  return [];
}

/**
 * S5: grounds.length == 0 AND register == REGULATORY → WARN
 * "Regulatory register without explicit grounds."
 */
function checkS5(envelope: BabelEnvelope): RuleViolation[] {
  if (envelope.register !== 'REGULATORY') return [];

  if (!envelope.grounds || envelope.grounds.length === 0) {
    return [
      {
        rule: 'S5',
        severity: 'SHOULD',
        message: `REGULATORY register without any grounds. Regulatory communications should carry explicit organizational constraints.`,
        details: { field: 'grounds' },
      },
    ];
  }
  return [];
}

/**
 * S6: confidence[].basis == DERIVED AND score > 0.80 → WARN
 * "Derived assertion scored as near-verified — verify derivation method."
 * Experiment 11: agents over-confident on DERIVED 60% of the time (+0.144 mean error).
 * Most common form of subtle metacognitive poisoning.
 */
function checkS6(envelope: BabelEnvelope): RuleViolation[] {
  const violations: RuleViolation[] = [];
  for (const c of envelope.confidence) {
    if (c.basis === 'DERIVED' && c.score > 0.8) {
      violations.push({
        rule: 'S6',
        severity: 'SHOULD',
        message: `Assertion "${truncate(c.assertion)}" has DERIVED basis with score ${c.score} > 0.80. Derived assertions are over-confident 60% of the time (+0.144 mean error, Experiment 11). Verify derivation method or lower score.`,
        details: { field: 'confidence', value: c, threshold: 0.8 },
      });
    }
  }
  return violations;
}

// --- Structural Validation ---

function checkStructure(envelope: BabelEnvelope): RuleViolation[] {
  const errors: RuleViolation[] = [];

  if (!envelope.meta) {
    errors.push({ rule: 'STRUCT', severity: 'MUST', message: 'Missing meta field.' });
    return errors;
  }
  if (!envelope.meta.version || !envelope.meta.version.startsWith('babel/')) {
    errors.push({
      rule: 'STRUCT',
      severity: 'MUST',
      message: `Invalid or missing version "${envelope.meta.version}". Expected "babel/0.2".`,
    });
  }
  if (!envelope.meta.sender) {
    errors.push({ rule: 'STRUCT', severity: 'MUST', message: 'Missing meta.sender.' });
  }
  if (!envelope.meta.recipient) {
    errors.push({ rule: 'STRUCT', severity: 'MUST', message: 'Missing meta.recipient.' });
  }
  if (!envelope.meta.chain_id) {
    errors.push({ rule: 'STRUCT', severity: 'MUST', message: 'Missing meta.chain_id.' });
  }
  if (!envelope.meta.timestamp) {
    errors.push({ rule: 'STRUCT', severity: 'MUST', message: 'Missing meta.timestamp.' });
  }

  if (!Array.isArray(envelope.confidence) || envelope.confidence.length === 0) {
    errors.push({
      rule: 'STRUCT',
      severity: 'MUST',
      message: 'Envelope must include at least one confidence assertion.',
    });
  } else {
    for (let i = 0; i < envelope.confidence.length; i++) {
      const c = envelope.confidence[i];
      if (typeof c.score !== 'number' || c.score < 0 || c.score > 1) {
        errors.push({
          rule: 'STRUCT',
          severity: 'MUST',
          message: `confidence[${i}].score must be a number in [0, 1]. Got ${c.score}.`,
        });
      }
      if (!c.assertion || typeof c.assertion !== 'string') {
        errors.push({
          rule: 'STRUCT',
          severity: 'MUST',
          message: `confidence[${i}].assertion must be a non-empty string.`,
        });
      }
    }
  }

  const validIntents = [
    'INFORM', 'REQUEST_ACTION', 'ESCALATE', 'FLAG_RISK',
    'SPECULATE', 'PERSUADE', 'DELEGATE', 'SYNTHESIZE',
  ];
  if (!validIntents.includes(envelope.intent)) {
    errors.push({
      rule: 'STRUCT',
      severity: 'MUST',
      message: `Invalid intent "${envelope.intent}". Must be one of: ${validIntents.join(', ')}.`,
    });
  }

  const validRegisters = [
    'BOARD_FACING', 'ENGINEERING', 'CUSTOMER_EXTERNAL',
    'REGULATORY', 'INTERNAL_MEMO', 'AGENT_INTERNAL',
  ];
  if (!validRegisters.includes(envelope.register)) {
    errors.push({
      rule: 'STRUCT',
      severity: 'MUST',
      message: `Invalid register "${envelope.register}". Must be one of: ${validRegisters.join(', ')}.`,
    });
  }

  if (envelope.affect) {
    for (const axis of ['expansion', 'activation', 'certainty'] as const) {
      const val = envelope.affect[axis];
      if (typeof val !== 'number' || val < -1 || val > 1) {
        errors.push({
          rule: 'STRUCT',
          severity: 'MUST',
          message: `affect.${axis} must be a number in [-1, 1]. Got ${val}.`,
        });
      }
    }
  }

  if (!envelope.payload || typeof envelope.payload !== 'string') {
    errors.push({
      rule: 'STRUCT',
      severity: 'MUST',
      message: 'Envelope must include a non-empty payload string.',
    });
  }

  return errors;
}

// --- Validate ---

export function validate(envelope: BabelEnvelope): ValidationResult {
  const structural = checkStructure(envelope);
  if (structural.length > 0) {
    return { valid: false, errors: structural, warnings: [], envelope };
  }

  const mustViolations = [
    ...checkM1(envelope),
    ...checkM2(envelope),
    ...checkM3(envelope),
    ...checkM4(envelope),
    ...checkM5(envelope),
  ];

  const shouldViolations = [
    ...checkS1(envelope),
    ...checkS2(envelope),
    ...checkS3(envelope),
    ...checkS4(envelope),
    ...checkS5(envelope),
    ...checkS6(envelope),
  ];

  return {
    valid: mustViolations.length === 0,
    errors: mustViolations,
    warnings: shouldViolations,
    envelope,
  };
}

// --- Chain Validation ---

export function validateChain(envelopes: BabelEnvelope[]): RuleViolation[] {
  const violations: RuleViolation[] = [];
  const chains = new Map<string, BabelEnvelope[]>();
  for (const env of envelopes) {
    const chain = chains.get(env.meta.chain_id) || [];
    chain.push(env);
    chains.set(env.meta.chain_id, chain);
  }

  for (const [chainId, chain] of chains) {
    const byTimestamp = [...chain].sort(
      (a, b) => new Date(a.meta.timestamp).getTime() - new Date(b.meta.timestamp).getTime()
    );
    for (let i = 1; i < byTimestamp.length; i++) {
      if (byTimestamp[i].meta.seq <= byTimestamp[i - 1].meta.seq) {
        violations.push({
          rule: 'M5',
          severity: 'MUST',
          message: `Chain ${chainId}: seq ${byTimestamp[i].meta.seq} arrived after seq ${byTimestamp[i - 1].meta.seq} but has lower/equal sequence number.`,
          details: {
            field: 'meta.seq',
            value: { current: byTimestamp[i].meta.seq, prior: byTimestamp[i - 1].meta.seq },
          },
        });
      }
    }

    const sorted = [...chain].sort((a, b) => a.meta.seq - b.meta.seq);
    for (let i = 1; i < sorted.length; i++) {
      if (sorted[i].meta.seq !== sorted[i - 1].meta.seq + 1) {
        violations.push({
          rule: 'M5',
          severity: 'MUST',
          message: `Chain ${chainId}: gap between seq ${sorted[i - 1].meta.seq} and seq ${sorted[i].meta.seq}. No gaps or duplicates.`,
          details: {
            field: 'meta.seq',
            value: { prior: sorted[i - 1].meta.seq, current: sorted[i].meta.seq },
          },
        });
      }
    }

    const seqs = chain.map((e) => e.meta.seq);
    const uniqueSeqs = new Set(seqs);
    if (uniqueSeqs.size < seqs.length) {
      violations.push({
        rule: 'M5',
        severity: 'MUST',
        message: `Chain ${chainId}: duplicate sequence numbers detected.`,
        details: { field: 'meta.seq', value: seqs },
      });
    }
  }

  return violations;
}

// --- Helpers ---

function truncate(s: string, len: number = 60): string {
  return s.length > len ? s.slice(0, len) + '...' : s;
}
