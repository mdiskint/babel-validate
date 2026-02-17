// ═══════════════════════════════════════════════
// BABEL SEMANTIC PATTERN DETECTOR
// Spec: Babel v0.2 "Semantic Combinations"
// These are recognized patterns that carry meaning
// beyond their individual fields — like idioms in
// a natural language.
// ═══════════════════════════════════════════════

import { BabelEnvelope, PatternDetection, SemanticPattern } from './types';

/**
 * Detect semantic patterns — coherence signals and contradiction
 * signals that grammar rules don't catch. These are the "idioms"
 * of Babel: meaning from cross-field combination.
 */
export function detectPatterns(envelope: BabelEnvelope): PatternDetection[] {
  const patterns: PatternDetection[] = [];

  // --- THE CALM ALERT ---
  // confidence: 0.8+ | intent: FLAG_RISK | affect: low activation, moderate certainty
  // "Sender is flagging something real but isn't panicking.
  //  Receiving agent should treat as important but not crisis."
  if (envelope.intent === 'FLAG_RISK' && envelope.affect) {
    const maxScore = Math.max(...envelope.confidence.map((c) => c.score));
    if (
      maxScore >= 0.8 &&
      envelope.affect.activation < 0 &&
      envelope.affect.certainty > 0
    ) {
      patterns.push({
        pattern: 'CALM_ALERT',
        description:
          'High-confidence risk flag with calm affect. Sender is flagging something real but isn\'t panicking. Treat as important but not crisis.',
        confidence: 0.7,
      });
    }
  }

  // --- THE RELUCTANT ESCALATION ---
  // intent: ESCALATE | affect: contracted, uncertain | trajectory: prior_handoffs 2+
  // "This has been handed off multiple times. The escalator isn't confident
  //  this will help either. Address the systemic pattern, not just the immediate issue."
  if (envelope.intent === 'ESCALATE' && envelope.affect) {
    const hasMultipleHandoffs =
      envelope.trajectory && envelope.trajectory.prior_handoffs !== undefined &&
      envelope.trajectory.prior_handoffs >= 2;

    if (
      envelope.affect.expansion < -0.3 &&
      envelope.affect.certainty < -0.3 &&
      hasMultipleHandoffs
    ) {
      patterns.push({
        pattern: 'RELUCTANT_ESCALATION',
        description:
          `Escalation with contracted affect and ${envelope.trajectory!.prior_handoffs} prior handoffs. Escalator isn't confident this will help either. Address the systemic pattern, not just the immediate issue.`,
        confidence: 0.75,
      });
    }
  }

  // --- THE CONFIDENT DELEGATION ---
  // intent: DELEGATE | confidence: 0.9+ | grounds: POLICY | affect: certainty 0.6+
  // "Sender knows what needs to happen and is handing it to the right agent.
  //  Receiving agent should execute, not re-analyze."
  if (envelope.intent === 'DELEGATE' && envelope.affect) {
    const allHighConf = envelope.confidence.every((c) => c.score >= 0.9);
    const hasPolicyGrounds = envelope.grounds?.some((g) => g.authority === 'POLICY');

    if (allHighConf && hasPolicyGrounds && envelope.affect.certainty >= 0.6) {
      patterns.push({
        pattern: 'CONFIDENT_DELEGATION',
        description:
          'High-confidence delegation with policy grounds and strong certainty. Sender knows what needs to happen. Receiving agent should execute, not re-analyze.',
        confidence: 0.8,
      });
    }
  }

  // --- THE LOADED INFORM ---
  // intent: INFORM | trajectory: DEGRADING, long duration | register: BOARD_FACING
  // "Surface reads as neutral update. Trajectory says this is a pattern
  //  the board needs to see. Frame as trend, not snapshot."
  if (
    envelope.intent === 'INFORM' &&
    envelope.register === 'BOARD_FACING' &&
    envelope.trajectory &&
    envelope.trajectory.direction === 'DEGRADING'
  ) {
    patterns.push({
      pattern: 'LOADED_INFORM',
      description:
        `INFORM intent with BOARD_FACING register but DEGRADING trajectory ("${truncate(envelope.trajectory.pattern)}"). Surface reads as neutral update, but trajectory says this is a pattern the board needs to see. Frame as trend, not snapshot.`,
      confidence: 0.7,
    });
  }

  // --- THE CONTRADICTION SIGNAL ---
  // affect: certainty 0.8+ | max(confidence) < 0.3
  // "Sender *feels* certain but *evidence* is weak. This is the envelope
  //  telling on itself — confidence may be emotional rather than evidentiary."
  // (This is also what SHOULD rule S2 catches — patterns detects it as a named idiom.)
  if (envelope.affect) {
    const maxScore = Math.max(...envelope.confidence.map((c) => c.score));
    if (envelope.affect.certainty > 0.5 && maxScore < 0.4) {
      patterns.push({
        pattern: 'CONTRADICTION_SIGNAL',
        description:
          `Affect certainty is ${envelope.affect.certainty.toFixed(2)} but max confidence score is only ${maxScore.toFixed(2)}. Sender feels certain but evidence is weak. Confidence may be emotional rather than evidentiary. Probe the basis.`,
        confidence: 0.65,
      });
    }
  }

  return patterns;
}

function truncate(s: string, len: number = 50): string {
  return s.length > len ? s.slice(0, len) + '...' : s;
}
