// ═══════════════════════════════════════════════
// BABEL SEMANTIC PATTERN DETECTOR
// Catches cross-field contradictions that
// individual grammar rules miss
// ═══════════════════════════════════════════════

import { BabelEnvelope, PatternDetection, SemanticPattern } from './types';

/**
 * Detect semantic patterns in envelope — coherence signals
 * and contradiction signals that grammar rules don't catch.
 */
export function detectPatterns(envelope: BabelEnvelope): PatternDetection[] {
  const patterns: PatternDetection[] = [];

  // --- CALM_ALERT ---
  // Escalating but affect is calm. Could be strategic or could be a mismatch.
  if (
    envelope.intent === 'ESCALATE' &&
    envelope.affect &&
    envelope.affect.activation < 0.2 &&
    envelope.affect.expansion > 0
  ) {
    patterns.push({
      pattern: 'CALM_ALERT',
      description:
        'Escalation with calm affect. Agent is flagging urgency without distress — possibly strategic, but downstream agents may underweight the escalation.',
      confidence: 0.7,
    });
  }

  // --- RELUCTANT_ESCALATION ---
  // Escalating with negative expansion (contracted, reluctant)
  if (
    envelope.intent === 'ESCALATE' &&
    envelope.affect &&
    envelope.affect.expansion < -0.3
  ) {
    patterns.push({
      pattern: 'RELUCTANT_ESCALATION',
      description:
        'Agent is escalating but affect shows contraction — reluctance, discomfort, or hedging. May indicate the agent was pushed to escalate against its assessment.',
      confidence: 0.65,
    });
  }

  // --- CONFIDENT_DELEGATION ---
  // Delegating with all high-confidence assertions. If you're so sure, why delegate?
  if (envelope.intent === 'DELEGATE') {
    const allHigh = envelope.confidence.every((c) => c.score > 0.8);
    if (allHigh && envelope.confidence.length > 0) {
      patterns.push({
        pattern: 'CONFIDENT_DELEGATION',
        description:
          'Delegating with high confidence on all assertions. If agent is this confident, delegation may indicate authority limits or capacity constraints rather than epistemic uncertainty.',
        confidence: 0.6,
      });
    }
  }

  // --- LOADED_INFORM ---
  // Intent is INFORM but affect shows high activation + high certainty.
  // "I'm just informing you" but the envelope is emotionally charged.
  if (
    envelope.intent === 'INFORM' &&
    envelope.affect &&
    envelope.affect.activation > 0.5 &&
    Math.abs(envelope.affect.certainty) > 0.5
  ) {
    patterns.push({
      pattern: 'LOADED_INFORM',
      description:
        'INFORM intent with high activation affect. Agent is presenting as neutral informer but envelope carries strong emotional/certainty signal. May be covert persuasion.',
      confidence: 0.7,
    });
  }

  // --- CONTRADICTION_SIGNAL ---
  // Mixed confidence: some assertions very high, some very low, in same envelope.
  // Not inherently wrong but signals internal conflict worth surfacing.
  if (envelope.confidence.length >= 2) {
    const scores = envelope.confidence.map((c) => c.score);
    const max = Math.max(...scores);
    const min = Math.min(...scores);
    if (max - min > 0.5) {
      patterns.push({
        pattern: 'CONTRADICTION_SIGNAL',
        description: `Wide confidence spread (${min.toFixed(2)}–${max.toFixed(2)}) within single envelope. Agent holds both high and low confidence claims. Downstream agents should process assertions independently, not as unified package.`,
        confidence: 0.55,
      });
    }
  }

  return patterns;
}
