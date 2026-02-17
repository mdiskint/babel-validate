// ═══════════════════════════════════════════════
// BABEL-VALIDATE
// Wire protocol + measurement engine for
// multi-agent metacognitive integrity.
//
// "Your agents are lying to each other.
//  They just don't know it."
// ═══════════════════════════════════════════════

// Types
export {
  BabelEnvelope,
  Meta,
  Intent,
  Confidence,
  Basis,
  Register,
  Affect,
  Ground,
  GroundAuthority,
  Trajectory,
  TrajectoryDirection,
  RuleViolation,
  RuleSeverity,
  ValidationResult,
  SemanticPattern,
  PatternDetection,
} from './types';

// Validation
export { validate, validateChain } from './validate';

// Builder
export { EnvelopeBuilder, envelope } from './builder';

// Pattern Detection
export { detectPatterns } from './patterns';

// Chain Audit
export {
  auditChain,
  ChainAudit,
  ConfidenceDrift,
  BasisShift,
} from './audit';
