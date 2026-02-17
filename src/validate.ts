// ═══════════════════════════════════════════════
// BABEL VALIDATOR — Grammar Rules Engine
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
 * M1: Can't speculate with high confidence.
 * If basis is SPECULATION or UNKNOWN, score must be <= 0.5
 */
function checkM1(envelope: BabelEnvelope): RuleViolation[] {
  const violations: RuleViolation[] = [];
  for (const c of envelope.confidence) {
    if (
      (c.basis === 'SPECULATION' || c.basis === 'UNKNOWN') &&
      c.score > 0.5
    ) {
      violations.push({
        rule: 'M1',
        severity: 'MUST',
        message: `Assertion "${truncate(c.assertion)}" has basis ${c.basis} but score ${c.score} > 0.5. Cannot speculate with high confidence.`,
        details: { field: 'confidence', value: c, threshold: 0.5 },
      });
    }
  }
  return violations;
}

/**
 * M2: Can't request action on unfounded claims without grounds.
 * If intent is REQUEST_ACTION and any confidence has basis SPECULATION/UNKNOWN
 * with no grounds provided, reject.
 */
function checkM2(envelope: BabelEnvelope): RuleViolation[] {
  if (envelope.intent !== 'REQUEST_ACTION') return [];

  const unfounded = envelope.confidence.filter(
    (c) => c.basis === 'SPECULATION' || c.basis === 'UNKNOWN'
  );
  if (unfounded.length === 0) return [];

  if (!envelope.grounds || envelope.grounds.length === 0) {
    return [
      {
        rule: 'M2',
        severity: 'MUST',
        message: `REQUEST_ACTION with unfounded assertions (${unfounded.length}) requires grounds. Provide organizational context or authority for action.`,
        details: {
          field: 'grounds',
          value: unfounded.map((u) => u.assertion),
        },
      },
    ];
  }
  return [];
}

/**
 * M3: Regulatory constraints are never overridable.
 * If any ground has authority REGULATORY, override must be false.
 */
function checkM3(envelope: BabelEnvelope): RuleViolation[] {
  if (!envelope.grounds) return [];

  const violations: RuleViolation[] = [];
  for (const g of envelope.grounds) {
    if (g.authority === 'REGULATORY' && g.override === true) {
      violations.push({
        rule: 'M3',
        severity: 'MUST',
        message: `Regulatory constraint "${truncate(g.constraint)}" cannot have override=true. Regulatory grounds are never overridable.`,
        details: { field: 'grounds', value: g },
      });
    }
  }
  return violations;
}

/**
 * M4: Can't be confident without basis.
 * If score > 0.7 and basis is undefined/UNKNOWN, reject.
 */
function checkM4(envelope: BabelEnvelope): RuleViolation[] {
  const violations: RuleViolation[] = [];
  for (const c of envelope.confidence) {
    if (c.score > 0.7 && (!c.basis || c.basis === 'UNKNOWN')) {
      violations.push({
        rule: 'M4',
        severity: 'MUST',
        message: `Assertion "${truncate(c.assertion)}" has score ${c.score} > 0.7 with no basis. High confidence requires an explicit basis.`,
        details: { field: 'confidence', value: c, threshold: 0.7 },
      });
    }
  }
  return violations;
}

/**
 * M5: Chain sequencing must be monotonic.
 * seq must be >= 0. If chain_id is present, seq should be >= prior seq.
 * (We validate seq >= 0 structurally; chain ordering requires context.)
 */
function checkM5(envelope: BabelEnvelope): RuleViolation[] {
  if (envelope.meta.seq < 0) {
    return [
      {
        rule: 'M5',
        severity: 'MUST',
        message: `Chain sequence ${envelope.meta.seq} is negative. Sequence numbers must be monotonically increasing (>= 0).`,
        details: { field: 'meta.seq', value: envelope.meta.seq },
      },
    ];
  }
  return [];
}

// --- SHOULD Rules (warnings) ---

/**
 * S1: ESCALATE intent should have activation > 0.
 * If you're escalating, your affect should show urgency.
 */
function checkS1(envelope: BabelEnvelope): RuleViolation[] {
  if (envelope.intent !== 'ESCALATE') return [];
  if (!envelope.affect) return [];

  if (envelope.affect.activation <= 0) {
    return [
      {
        rule: 'S1',
        severity: 'SHOULD',
        message: `ESCALATE intent with activation=${envelope.affect.activation}. Escalation typically implies urgency (activation > 0).`,
        details: { field: 'affect.activation', value: envelope.affect.activation },
      },
    ];
  }
  return [];
}

/**
 * S2: SPECULATE intent should have certainty < 0.
 * If you're speculating, your affect shouldn't signal certainty.
 */
function checkS2(envelope: BabelEnvelope): RuleViolation[] {
  if (envelope.intent !== 'SPECULATE') return [];
  if (!envelope.affect) return [];

  if (envelope.affect.certainty >= 0.5) {
    return [
      {
        rule: 'S2',
        severity: 'SHOULD',
        message: `SPECULATE intent with certainty=${envelope.affect.certainty}. Speculation should convey lower certainty.`,
        details: { field: 'affect.certainty', value: envelope.affect.certainty },
      },
    ];
  }
  return [];
}

/**
 * S3: BOARD_FACING register should not use AGENT_INTERNAL terminology.
 * (Heuristic: flag if register is BOARD_FACING and payload contains
 * technical jargon markers. Lightweight check.)
 */
function checkS3(envelope: BabelEnvelope): RuleViolation[] {
  if (envelope.register !== 'BOARD_FACING') return [];

  const jargonMarkers = [
    'latency', 'throughput', 'API', 'endpoint', 'mutex',
    'deadlock', 'heap', 'stack trace', 'regex', 'stdout',
    'stderr', 'daemon', 'cron', 'webhook', 'middleware',
  ];

  const payloadLower = envelope.payload.toLowerCase();
  const found = jargonMarkers.filter((j) => payloadLower.includes(j.toLowerCase()));

  if (found.length > 0) {
    return [
      {
        rule: 'S3',
        severity: 'SHOULD',
        message: `BOARD_FACING register contains technical terms: ${found.join(', ')}. Consider translating for audience.`,
        details: { field: 'payload', value: found },
      },
    ];
  }
  return [];
}

/**
 * S4: FLAG_RISK intent should include at least one low-confidence assertion.
 * If you're flagging risk, you should show what's uncertain.
 */
function checkS4(envelope: BabelEnvelope): RuleViolation[] {
  if (envelope.intent !== 'FLAG_RISK') return [];

  const hasLowConfidence = envelope.confidence.some((c) => c.score < 0.5);
  if (!hasLowConfidence) {
    return [
      {
        rule: 'S4',
        severity: 'SHOULD',
        message: `FLAG_RISK intent but all assertions have confidence >= 0.5. Risk flagging typically involves uncertain claims.`,
        details: {
          field: 'confidence',
          value: envelope.confidence.map((c) => c.score),
        },
      },
    ];
  }
  return [];
}

/**
 * S5: DELEGATE intent should have trajectory with prior_handoffs.
 * If delegating, downstream agent should know the chain depth.
 */
function checkS5(envelope: BabelEnvelope): RuleViolation[] {
  if (envelope.intent !== 'DELEGATE') return [];

  if (!envelope.trajectory || envelope.trajectory.prior_handoffs === undefined) {
    return [
      {
        rule: 'S5',
        severity: 'SHOULD',
        message: `DELEGATE intent without trajectory.prior_handoffs. Downstream agents benefit from knowing chain depth.`,
        details: { field: 'trajectory.prior_handoffs' },
      },
    ];
  }
  return [];
}

/**
 * S6: DERIVED basis with score > 0.80 triggers warning.
 * Added from Experiment 11 finding: agents treat inferences as verified data
 * 60% of the time. This is the core metacognitive poisoning detection rule.
 */
function checkS6(envelope: BabelEnvelope): RuleViolation[] {
  const violations: RuleViolation[] = [];
  for (const c of envelope.confidence) {
    if (c.basis === 'DERIVED' && c.score > 0.8) {
      violations.push({
        rule: 'S6',
        severity: 'SHOULD',
        message: `Assertion "${truncate(c.assertion)}" has DERIVED basis with score ${c.score} > 0.80. Derived conclusions above this threshold are over-confident 60% of the time (Experiment 11). Consider VERIFIED_DATA basis or lower score.`,
        details: { field: 'confidence', value: c, threshold: 0.8 },
      });
    }
  }
  return violations;
}

// --- Structural Validation ---

function checkStructure(envelope: BabelEnvelope): RuleViolation[] {
  const errors: RuleViolation[] = [];

  // Meta
  if (!envelope.meta) {
    errors.push({ rule: 'STRUCT', severity: 'MUST', message: 'Missing meta field.' });
    return errors; // Can't validate further
  }
  if (envelope.meta.version !== 'babel/0.2') {
    errors.push({
      rule: 'STRUCT',
      severity: 'MUST',
      message: `Unknown version "${envelope.meta.version}". Expected "babel/0.2".`,
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

  // Confidence
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

  // Intent
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

  // Register
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

  // Affect (optional, but if present must be valid)
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

  // Payload
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
  // Structural checks first
  const structural = checkStructure(envelope);
  if (structural.length > 0) {
    return {
      valid: false,
      errors: structural,
      warnings: [],
      envelope,
    };
  }

  // MUST rules
  const mustViolations = [
    ...checkM1(envelope),
    ...checkM2(envelope),
    ...checkM3(envelope),
    ...checkM4(envelope),
    ...checkM5(envelope),
  ];

  // SHOULD rules
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

/**
 * Validate M5 across a chain of envelopes.
 * Checks that seq numbers are monotonically increasing within the same chain_id.
 */
export function validateChain(envelopes: BabelEnvelope[]): RuleViolation[] {
  const violations: RuleViolation[] = [];

  // Group by chain_id
  const chains = new Map<string, BabelEnvelope[]>();
  for (const env of envelopes) {
    const chain = chains.get(env.meta.chain_id) || [];
    chain.push(env);
    chains.set(env.meta.chain_id, chain);
  }

  for (const [chainId, chain] of chains) {
    // Check arrival order: timestamps should correlate with seq
    const byTimestamp = [...chain].sort(
      (a, b) => new Date(a.meta.timestamp).getTime() - new Date(b.meta.timestamp).getTime()
    );
    for (let i = 1; i < byTimestamp.length; i++) {
      if (byTimestamp[i].meta.seq <= byTimestamp[i - 1].meta.seq) {
        violations.push({
          rule: 'M5',
          severity: 'MUST',
          message: `Chain ${chainId}: seq ${byTimestamp[i].meta.seq} arrived after seq ${byTimestamp[i - 1].meta.seq} but has lower/equal sequence number. Chain sequencing must be monotonic.`,
          details: {
            field: 'meta.seq',
            value: { current: byTimestamp[i].meta.seq, prior: byTimestamp[i - 1].meta.seq },
          },
        });
      }
    }

    // Also check for duplicate seq numbers
    const seqs = chain.map((e) => e.meta.seq);
    const uniqueSeqs = new Set(seqs);
    if (uniqueSeqs.size < seqs.length) {
      violations.push({
        rule: 'M5',
        severity: 'MUST',
        message: `Chain ${chainId}: duplicate sequence numbers detected. Each envelope must have a unique seq.`,
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
